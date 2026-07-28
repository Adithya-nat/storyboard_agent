"""The motion node: submit/poll/download per scene, and every degrade path
(no client / unreachable / partial failure / budget timeout / idempotent resume).
Uses a fake engine client — no HTTP, no ffmpeg."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agents.motion import motion as motion_node
from app.artifacts.store import ArtifactStore
from app.core.config import Settings
from app.graph.state import Deps


class FakeMotion:
    def __init__(
        self,
        *,
        health_ok=True,
        tier="parallax",
        motion_kind=None,
        model=None,
        policy=None,
        fail_orders=(),
        never_done=False,
    ):
        self.health_ok = health_ok
        self.tier = tier
        self.motion_kind = motion_kind
        self.model = model
        self.policy = policy
        self.fail_orders = set(fail_orders)
        self.never_done = never_done
        self.submitted: list[dict] = []
        self._tasks: dict[str, dict] = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def health(self):
        if not self.health_ok:
            return None
        body = {"tiers": ["parallax", "kenburns"]}
        if self.policy:
            body["policy"] = self.policy
        return body

    async def submit(self, image, meta):
        self.submitted.append(meta)
        tid = f"t{meta['order']}"
        self._tasks[tid] = meta
        return tid

    async def get_task(self, tid):
        meta = self._tasks[tid]
        if self.never_done:
            return {"status": "rendering"}
        if meta["order"] in self.fail_orders:
            return {"status": "failed", "error": "boom"}
        return {
            "status": "done",
            "tier": self.tier,
            "motion_kind": self.motion_kind,
            "renderer": self.tier,
            "model": self.model,
            "source_seconds": 4.0,
            "pipeline_version": "2",
            "attempts": [{"renderer": self.tier, "outcome": "succeeded"}],
        }

    async def download(self, tid):
        return b"CLIP-" + str(self._tasks[tid]["order"]).encode()


def _deps(tmp_path, motion, **over):
    settings = Settings(
        motion_engine_enabled=True,
        motion_max_concurrency=over.get("concurrency", 4),
        motion_stage_budget_seconds=over.get("budget", 10.0),
        motion_poll_interval_seconds=0.01,
        motion_clip_seconds=8.0,
        use_mock_providers=True,
    )
    store = ArtifactStore(tmp_path, "job1")
    deps = Deps(
        settings=settings, store=store,
        llm=SimpleNamespace(), image=SimpleNamespace(), tts=None,
        reporter=None, motion=motion,
    )
    return deps, store


def _frames(store, n=2):
    frames = []
    for o in range(1, n + 1):
        store.save_bytes(f"frame_{o:02d}.png", f"PNG{o}".encode())
        frames.append({"order": o, "path": store.abspath(f"frame_{o:02d}.png"),
                       "caption": f"c{o}", "status": "ok"})
    return frames


async def test_motion_success_saves_clips(tmp_path):
    motion = FakeMotion(tier="parallax")
    deps, store = _deps(tmp_path, motion)
    frames = _frames(store, 2)
    out = await motion_node({"job_id": "job1", "frames": frames}, deps)

    clips = {c["order"]: c for c in out["motion_clips"]}
    assert set(clips) == {1, 2}
    assert all(c["status"] == "ok" and c["tier"] == "parallax" for c in clips.values())
    assert store.exists("motion_01.mp4") and store.exists("motion_02.mp4")
    # first scene is a hero (min order)
    hero = next(m for m in motion.submitted if m["order"] == 1)
    assert hero["hero_hint"] is True
    manifest = store.load_json(store.abspath("motion_manifest.json"))
    assert manifest["pipeline_version"] == "2"
    assert [scene["order"] for scene in manifest["scenes"]] == [1, 2]


@pytest.mark.parametrize("scene_count", [15, 30])
async def test_motion_manifest_scales_to_full_storyboards(tmp_path, scene_count):
    motion = FakeMotion(
        tier="veo",
        motion_kind="generative",
        model="veo-3.1-lite-generate-preview",
        policy="generative_all",
    )
    deps, store = _deps(
        tmp_path, motion, concurrency=30, budget=30.0
    )

    out = await motion_node(
        {"job_id": f"job-{scene_count}", "frames": _frames(store, scene_count)},
        deps,
    )
    manifest = store.load_json(store.abspath("motion_manifest.json"))

    assert len(out["motion_clips"]) == scene_count
    assert len(manifest["scenes"]) == scene_count
    assert all(scene["motion_kind"] == "generative" for scene in manifest["scenes"])


async def test_motion_no_client_is_noop(tmp_path):
    deps, store = _deps(tmp_path, None)
    frames = _frames(store, 2)
    out = await motion_node({"job_id": "job1", "frames": frames}, deps)
    assert out["motion_clips"] == []
    assert out["warnings"] and "not configured" in out["warnings"][0]


async def test_motion_engine_unreachable_degrades(tmp_path):
    motion = FakeMotion(health_ok=False)
    deps, store = _deps(tmp_path, motion)
    frames = _frames(store, 2)
    out = await motion_node({"job_id": "job1", "frames": frames}, deps)
    assert out["motion_clips"] == []
    assert any("unreachable" in w for w in out["warnings"])
    assert motion.submitted == []  # never even tried to submit


async def test_motion_partial_failure_keeps_others(tmp_path):
    motion = FakeMotion(fail_orders={2})
    deps, store = _deps(tmp_path, motion)
    frames = _frames(store, 2)
    out = await motion_node({"job_id": "job1", "frames": frames}, deps)
    orders = {c["order"] for c in out["motion_clips"]}
    assert orders == {1}  # scene 2 failed
    assert any("scene 2" in w for w in out["warnings"])


async def test_motion_budget_timeout_degrades(tmp_path):
    motion = FakeMotion(never_done=True)
    deps, store = _deps(tmp_path, motion, budget=0.2)
    frames = _frames(store, 2)
    out = await motion_node({"job_id": "job1", "frames": frames}, deps)
    assert out["motion_clips"] == []
    assert any("budget" in w for w in out["warnings"])


async def test_motion_idempotent_resume_skips_existing(tmp_path):
    motion = FakeMotion()
    deps, store = _deps(tmp_path, motion)
    frames = _frames(store, 2)
    store.save_bytes("motion_01.mp4", b"already-here")  # from a prior run
    out = await motion_node({"job_id": "job1", "frames": frames}, deps)

    clips = {c["order"]: c for c in out["motion_clips"]}
    assert clips[1]["tier"] == "cached"        # reused, not re-billed
    assert [m["order"] for m in motion.submitted] == [2]  # only scene 2 submitted


async def test_generative_policy_upgrades_unversioned_existing_clip(tmp_path):
    motion = FakeMotion(
        tier="veo",
        motion_kind="generative",
        model="veo-3.1-lite-generate-preview",
        policy="generative_all",
    )
    deps, store = _deps(tmp_path, motion)
    frames = _frames(store, 1)
    store.save_bytes("motion_01.mp4", b"legacy-fallback")

    out = await motion_node({"job_id": "job1", "frames": frames}, deps)

    assert [m["order"] for m in motion.submitted] == [1]
    assert out["motion_clips"][0]["motion_kind"] == "generative"
    assert out["motion_manifest_path"].endswith("motion_manifest.json")


async def test_generative_policy_marks_compositor_as_degraded(tmp_path):
    motion = FakeMotion(
        tier="compositor",
        motion_kind="composited",
        policy="generative_all",
    )
    deps, store = _deps(tmp_path, motion)
    out = await motion_node({"job_id": "job1", "frames": _frames(store, 1)}, deps)

    assert out["motion_clips"][0]["status"] == "degraded"
    assert any("non-generative fallback" in warning for warning in out["warnings"])
