"""Content-policy blocks: classification, the rewrite/redraw rescue ladder, and resume safety."""

from __future__ import annotations

import base64
import json
import os
import time
from types import SimpleNamespace

import pytest

from app.adapters import omni
from app.adapters.omni import ContentBlocked, OmniVideoAdapter, SceneSpec
from app.core.config import Settings

THIRD_PARTY = Exception("Error code: 400 - {'error': {'message': \"Input blocked: Google's guardrails related to "
                        "third-party content.\", 'code': 'content_blocked'}}")
PROHIBITED = Exception("Error code: 400 - {'error': {'message': 'Request blocked due to prohibited content "
                       "guidelines.', 'code': 'prohibited_content'}}")
JPEG = b"\xff\xd8\xff" + b"0" * 16
SPEC = SceneSpec(order=1, scene="a hero turns invisible", mode="dialogue", characters=("A",),
                 lines=(("A", "hello", ""),), voices={"A": "MALE"}, seconds=4.0)


class _Fake:
    def __init__(self, items):
        self.items, self.calls = list(items), []

    def create(self, **kw):
        self.calls.append(kw)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _video(i="int-1"):
    return SimpleNamespace(id=i, output_video=SimpleNamespace(uri=None, data=base64.b64encode(b"MP4").decode()))


def _adapter(tmp_path, items, **kw):
    s = Settings(gemini_api_key="k", use_mock_providers=False, omni_rpm_limit=100000, omni_speaker_check=False, **kw)
    a = OmniVideoAdapter(s, journal_dir=tmp_path / "ops")
    fake = _Fake(items)
    a._client = SimpleNamespace(interactions=fake, models=None, files=None)
    return a, fake


# --- adapter --------------------------------------------------------------- #
def test_block_codes():
    assert omni.block_code(THIRD_PARTY) == "content_blocked"
    assert omni.block_code(PROHIBITED) == "prohibited_content"
    assert omni.block_code(Exception("429 RESOURCE_EXHAUSTED: requests per day")) is None


def test_content_block_raises_without_retrying(tmp_path):
    a, fake = _adapter(tmp_path, [THIRD_PARTY])
    with pytest.raises(ContentBlocked) as e:
        a.generate(JPEG, SPEC, key="j-1")
    assert e.value.code == "content_blocked" and len(fake.calls) == 1
    assert not list((tmp_path / "ops").glob("*.json"))  # nothing billed, fence cleared


def test_stale_fence_is_released(tmp_path):
    a, fake = _adapter(tmp_path, [_video()])
    j = a._journal("j-2")
    j.write_text(json.dumps({"model": "gemini-omni-1.2-flash", "state": "submission_uncertain"}))
    old = time.time() - (a.settings.omni_timeout_seconds + 600)
    os.utime(j, (old, old))
    assert a.generate(JPEG, SPEC, key="j-2").data == b"MP4"


def test_fence_from_an_earlier_process_is_released(tmp_path):
    a, fake = _adapter(tmp_path, [_video()])
    a._journal("j-4").write_text(json.dumps({"model": "gemini-omni-1.2-flash", "state": "submission_uncertain",
                                              "boot": "123-earlier-process"}))
    assert a.generate(JPEG, SPEC, key="j-4").data == b"MP4"


def test_blocked_voice_rerender_keeps_first_clip(tmp_path, monkeypatch):
    a, fake = _adapter(tmp_path, [_video("a"), PROHIBITED])
    monkeypatch.setattr(a, "_wrong_voice", lambda *_: True)
    clip = a.generate(JPEG, SPEC, key="j-3")
    assert clip.data == b"MP4" and clip.renders == 1


# --- video stage ----------------------------------------------------------- #
async def _stage(artifact_root, job, behaviour, *, image=None):
    from app.adapters.llm import LLMAdapter
    from app.agents.designer import _mock_shotlist
    from app.agents.script_writer import _mock_script
    from app.artifacts.store import ArtifactStore
    from app.core.images import render_card
    from app.graph.state import Deps

    s = Settings(gemini_api_key="", use_mock_providers=True, ai_video_retry_delay_seconds=0)
    store = ArtifactStore(artifact_root, job)
    script = _mock_script("x", 3, "dialogue")
    calls: list[tuple[int, str, bytes]] = []

    class _V:
        def generate(self, img, spec, *, key):
            calls.append((spec.order, spec.scene, img))
            exc = behaviour(spec, img, sum(1 for o, *_ in calls if o == spec.order))
            if exc:
                raise exc
            return omni.OmniClip(data=b"MP4", model="gemini-omni-1.2-flash", seconds=5, cost_usd=0.5, renders=1)

    deps = Deps(settings=s, store=store, llm=LLMAdapter(s), image=image, video=_V())
    frames = [{"order": i, "path": store.save_bytes(f"frame_{i:02d}.png", render_card(str(i))), "caption": "",
               "status": "ok"} for i in (1, 2, 3)]
    state = {"job_id": job, "frames": frames, "script_path": store.save_json("script.json", script),
             "shotlist_path": store.save_json("shotlist.json", _mock_shotlist(script)), "language": "en"}
    from app.agents.video_gen import video_gen

    return video_gen, state, deps, calls


@pytest.fixture
async def client():
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _no_ffprobe(monkeypatch):
    monkeypatch.setattr("app.agents.video_gen.has_audio", lambda p: True)


def _third_party_until_rewritten(spec, img, n):
    return ContentBlocked(str(THIRD_PARTY), "content_blocked") if spec.order == 2 and "(rewritten)" not in spec.scene else None


async def test_third_party_block_is_rewritten_and_recorded(artifact_root):
    video_gen, state, deps, calls = await _stage(artifact_root, "rescue-1", _third_party_until_rewritten)
    out = await video_gen(state, deps)
    assert len(out["motion_clips"]) == 3
    assert [c[1].endswith("(rewritten)") for c in calls if c[0] == 2] == [False, True]  # no unchanged retry for IP
    rescue = json.loads(open(deps.store.abspath("rescue_02.json")).read())
    assert rescue["code"] == "content_blocked" and rescue["redrawn_frame"] is False
    script = json.loads(open(state["script_path"]).read())
    assert next(sc for sc in script["scenes"] if sc["order"] == 2)["description"].endswith("(rewritten)")
    clip2 = next(c for c in out["motion_clips"] if c["order"] == 2)
    assert clip2["motion_reason"] == "rewritten to meet content rules"


async def test_flaky_prohibited_block_gets_one_unchanged_retry(artifact_root):
    video_gen, state, deps, calls = await _stage(
        artifact_root, "rescue-2", lambda spec, img, n: ContentBlocked("x", "prohibited_content") if spec.order == 3 and n == 1 else None)
    out = await video_gen(state, deps)
    assert len(out["motion_clips"]) == 3 and not deps.store.exists("rescue_03.json")
    assert [c[1] for c in calls if c[0] == 3][0] == [c[1] for c in calls if c[0] == 3][1]  # same prompt twice


async def test_frame_is_redrawn_when_the_image_is_the_trigger(artifact_root, monkeypatch):
    new_frame = b"\x89PNG-new"
    monkeypatch.setattr("app.agents.video_gen.render_frame", lambda *a, **k: new_frame)

    def block_old_frame(spec, img, n):
        return ContentBlocked("x", "content_blocked") if spec.order == 1 and img != new_frame else None

    video_gen, state, deps, calls = await _stage(artifact_root, "rescue-3", block_old_frame, image=object())
    out = await video_gen(state, deps)
    assert len(out["motion_clips"]) == 3
    assert deps.store.exists("frame_01.blocked.png")
    assert open(deps.store.abspath("frame_01.png"), "rb").read() == new_frame
    assert json.loads(open(deps.store.abspath("rescue_01.json")).read())["redrawn_frame"] is True


async def test_scene_blocked_after_every_rewrite_stops_with_a_clear_reason(artifact_root):
    from app.agents.video_gen import AIVideoUnavailable

    video_gen, state, deps, calls = await _stage(
        artifact_root, "rescue-4", lambda spec, img, n: ContentBlocked("x", "content_blocked") if spec.order == 2 else None)
    with pytest.raises(AIVideoUnavailable, match="third-party content rule.*scene\\(s\\) 2.*original characters"):
        await video_gen(state, deps)
    assert len([c for c in calls if c[0] == 2]) == 3  # original + 2 rewrites (no image adapter -> no redraws)
    manifest = json.loads(open(deps.store.abspath("motion_manifest.json")).read())
    assert sorted(s["order"] for s in manifest["scenes"]) == [1, 3]  # finished clips kept


async def test_clip_finished_after_the_manifest_is_adopted_not_rebilled(artifact_root, monkeypatch):
    video_gen, state, deps, calls = await _stage(artifact_root, "rescue-5", lambda *a: None)
    deps.store.save_bytes("motion_02.mp4", b"MP4")  # written by a process that was then stopped
    monkeypatch.setattr("app.agents.video_gen._probe_seconds", lambda p: 7.0 if p.endswith("motion_02.mp4") else 0.0)
    out = await video_gen(state, deps)
    assert sorted(c[0] for c in calls) == [1, 3]
    assert next(c for c in out["motion_clips"] if c["order"] == 2)["motion_reason"] == "recovered after a restart"


# --- finish with stills (user's choice) ------------------------------------ #
async def test_allow_stills_finishes_with_existing_clips_only(artifact_root):
    video_gen, state, deps, calls = await _stage(artifact_root, "stills-1", lambda *a: None)
    deps.store.save_bytes("motion_01.mp4", b"MP4")
    deps.store.save_json("motion_manifest.json", {"scenes": [{
        "order": 1, "path": deps.store.abspath("motion_01.mp4"), "status": "ok", "tier": "omni",
        "motion_kind": "generative", "renderer": "omni", "model": "m", "source_seconds": 5.0,
        "pipeline_version": "omni-1", "attempts": [], "motion_reason": None, "has_audio": True}]})
    deps.allow_stills = True
    out = await video_gen(state, deps)
    assert calls == []  # no new AI video spend
    assert [c["order"] for c in out["motion_clips"]] == [1] and out["stills_used"] == [2, 3]
    assert any("your choice" in w for w in out["warnings"])


async def test_status_offers_finishing_with_stills_and_resume_passes_the_choice(client, artifact_root, monkeypatch):
    from app.artifacts.store import ArtifactStore
    from app.db import repository as repo
    from app.db.session import get_sessionmaker
    from app.schemas import JobStatus

    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
        await repo.set_status(s, job.id, JobStatus.FAILED,
                              error="AIVideoUnavailable: Google's third-party content rule blocked scene(s) 2")
    store = ArtifactStore(artifact_root, job.id)
    for i in (1, 2, 3):
        store.save_bytes(f"frame_{i:02d}.png", b"png")
    store.save_bytes("frame_02.blocked.png", b"png")  # not a scene of its own
    for i in (1, 3):
        store.save_bytes(f"motion_{i:02d}.mp4", b"MP4")
    store.save_json("motion_manifest.json", {"scenes": [{"order": 1}, {"order": 3}]})

    gap = (await client.get(f"/storyboard/{job.id}")).json()["video_gap"]
    assert gap == {"ready": 2, "total": 3, "missing": [2], "reason": "blocked"}

    seen = {}

    async def _capture(*a, **k):
        seen.update(k)

    monkeypatch.setattr("app.api.routes.run_job", _capture)
    r = await client.post(f"/storyboard/{job.id}/resume", json={"allow_stills": True})
    assert r.status_code == 202 and seen["allow_stills"] is True


def test_spend_limit_429_saturates_the_governor(tmp_path, monkeypatch):
    from app.observability import spend_rate

    g = spend_rate.SpendRateGovernor(9.0)
    monkeypatch.setattr("app.adapters.omni.get_governor", lambda limit: g)
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda *_a, **_k: None)
    a, fake = _adapter(tmp_path, [Exception("429 You exceeded your spend-based rate limit"), _video()])
    monkeypatch.setattr(g, "acquire", lambda est, **k: 0.0)
    assert a.generate(JPEG, SPEC, key="j-9").data == b"MP4"
    assert g.spent() >= 9.0 - 1e-6  # next calls wait for the window
