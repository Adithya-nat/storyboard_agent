"""Recut: the dependency closure, takes, the director's edits, and the API end to end (mock mode)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.adapters import omni
from app.agents.music import _fits
from app.agents.redo_director import RedoPlan, SceneEdit, ShotEdit, apply_plan, edited
from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.redo import takes as tk
from app.redo.targets import Item, closure, media_path
from app.schemas import DialogueLine, JobStatus, Script
from app.workers.runner import run_job

PIPE = ["brief", "script_writer", "dialogue_pass", "reflection", "designer", "image_gen", "narration", "motion",
        "captions", "music", "assembler", "film_review", "publish_kit"]


@pytest.fixture
def stub_video(monkeypatch):
    async def _fake_assemble(timeline, output_path):
        Path(output_path).write_bytes(b"FAKE-MP4")
        return output_path

    monkeypatch.setattr("app.agents.assembler.assemble_video", _fake_assemble)


@pytest.fixture
async def client():
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def launched(monkeypatch):
    """Capture the background run instead of starting it, so tests can await it."""
    calls: list[tuple] = []

    async def _capture(*a, **k):
        calls.append((a, k))

    monkeypatch.setattr("app.api.redo.run_job", _capture)
    return calls


async def _finished_film(prompt="a lighthouse keeper finds a message in a bottle") -> str:
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, prompt, 30)
    await run_job(job.id, prompt, 30)
    return job.id


async def _status(job_id: str) -> str:
    async with get_sessionmaker()() as s:
        return (await repo.get_job(s, job_id)).status


# --- closure ----------------------------------------------------------------- #
def _keys(inv):
    return {i.key for i in inv.items}


def test_a_clip_rebuilds_only_its_captions_and_the_edit():
    inv = closure([Item("clip", 4)], [1, 2, 3, 4, 5], PIPE)
    assert _keys(inv) == {"clip-4", "captions-4", "edit"}
    assert inv.stages == ["motion", "captions", "assembler", "film_review"]


def test_a_frame_also_rebuilds_its_clip_and_the_cover():
    inv = closure([Item("frame", 2)], [1, 2, 3], PIPE)
    assert _keys(inv) == {"frame-2", "clip-2", "captions-2", "edit", "release"}


def test_the_story_rebuilds_everything_downstream():
    inv = closure([Item("story")], [1, 2], PIPE)
    assert {"look", "frame-1", "clip-2", "scene-1", "score", "release", "edit"} <= _keys(inv)
    assert inv.stages[:3] == ["script_writer", "dialogue_pass", "reflection"]


def test_without_the_video_stage_clips_drop_out():
    inv = closure([Item("frame", 1)], [1, 2], [s for s in PIPE if s != "motion"])
    assert "clip-1" not in _keys(inv) and "motion" not in inv.stages


def test_a_release_rewrite_leaves_the_edit_alone():
    assert _keys(closure([Item("release")], [1], PIPE)) == {"release"}


def test_media_paths_never_leave_the_job_folder(tmp_path):
    (tmp_path / "frame_01.png").write_bytes(b"x")
    assert media_path(tmp_path, "frame_01.png") is not None
    assert media_path(tmp_path, "../frame_01.png") is None
    assert media_path(tmp_path, "script.json") is None


# --- takes ------------------------------------------------------------------- #
def test_archive_then_bring_back_round_trips_and_forgets_the_clip_entry(tmp_path):
    (tmp_path / "frame_02.png").write_bytes(b"OLD-FRAME")
    (tmp_path / "motion_02.mp4").write_bytes(b"OLD-CLIP")
    (tmp_path / "motion_manifest.json").write_text(json.dumps({"scenes": [{"order": 1}, {"order": 2}]}))
    made = tk.archive(tmp_path, [Item("frame", 2), Item("clip", 2), Item("clip", 3)], cut=2)
    assert {t["id"] for t in made} == {"c2-frame-2", "c2-clip-2"}
    assert not (tmp_path / "frame_02.png").exists()
    assert [c["order"] for c in json.loads((tmp_path / "motion_manifest.json").read_text())["scenes"]] == [1]

    (tmp_path / "frame_02.png").write_bytes(b"NEW-FRAME")
    group = tk.take_group(tmp_path, "c2-frame-2")
    assert {t["kind"] for t in group} == {"frame", "clip"}  # a frame comes back with its clip
    tk.bring_back(tmp_path, group)
    assert (tmp_path / "frame_02.png").read_bytes() == b"OLD-FRAME"
    assert (tmp_path / "takes" / "c2-frame-2" / "frame_02.png").exists()  # still restorable later


def test_cuts_are_numbered_and_snapshot_their_timing(tmp_path):
    (tmp_path / "script.json").write_text(json.dumps({"scenes": [{"order": 1, "seconds": 8}]}))
    assert tk.start_cut(tmp_path, picked=[Item("clip", 1)], items=[], note="n", summary="s", cost_now=5.0) == 2
    assert tk.start_cut(tmp_path, picked=[], items=[], note="", summary="", cost_now=6.5) == 3
    cuts = tk.history(tmp_path)
    assert [c["cut"] for c in cuts] == [1, 2, 3]
    assert cuts[0]["scene_seconds"] == {"1": 8.0} and cuts[1]["cost_end"] == 6.5


def test_a_score_still_fits_until_a_cut_point_moves():
    assert _fits(None, 30, [0, 8])
    assert _fits({"seconds": 30, "starts": [0, 8]}, 30.4, [0, 8.5])
    assert not _fits({"seconds": 30, "starts": [0, 8]}, 33, [0, 11])


# --- the director's edits -------------------------------------------------------- #
def _script() -> Script:
    from app.agents.script_writer import _mock_script

    return _mock_script("x", 3, "dialogue")


def test_edits_change_only_allowed_scenes_and_keep_known_speakers():
    script = _script()
    speaker = script.characters[0].name
    plan = RedoPlan(summary="s", scenes=[
        SceneEdit(order=2, description="a dance-off", image_prompt="two dancers mid-spin",
                  shots=[ShotEdit(index=0, action="they spin", lines=[DialogueLine(speaker=speaker, text="Your move.")])]),
        SceneEdit(order=3, description="should not land"),
        SceneEdit(order=1, shots=[ShotEdit(index=0, lines=[DialogueLine(speaker="STRANGER", text="hi")])]),
    ])
    before3 = script.scenes[2].description
    before1 = [l.text for l in script.scenes[0].shots[0].lines]
    out, _, changed = edited(script, None, plan, allowed={1, 2})
    assert changed == {2: True, 1: False}
    assert out.scenes[1].description == "a dance-off" and out.scenes[1].shots[0].action == "they spin"
    assert out.scenes[1].lines[0].text == "Your move."
    assert out.scenes[2].description == before3
    assert [l.text for l in out.scenes[0].shots[0].lines] == before1  # unknown speaker: lines kept


def test_a_picked_note_replaces_the_old_direction(tmp_path):
    from app.artifacts.store import ArtifactStore

    store = ArtifactStore(tmp_path, "j")
    store.save_json("script.json", _script())
    apply_plan(store.job_dir, RedoPlan(summary="s", score_note="lighter, playful strings"), [Item("score")])
    apply_plan(store.job_dir, RedoPlan(summary="s"), [Item("score")])
    assert json.loads((store.job_dir / "direction.json").read_text())["score"] == ""


# --- API end to end ------------------------------------------------------------ #
async def test_recut_one_frame_reuses_everything_else_and_keeps_the_old_take(client, stub_video, artifact_root, launched):
    job_id = await _finished_film()
    job_dir = artifact_root / job_id
    script_mtime = (job_dir / "script.json").stat().st_mtime_ns
    kept = (job_dir / "frame_01.png").stat().st_mtime_ns
    old = (job_dir / "frame_02.png").read_bytes()

    plan = (await client.post(f"/storyboard/{job_id}/recut/plan", json={"targets": [{"kind": "frame", "unit": 2}]})).json()
    assert plan["directed"] is False and plan["plan_id"] is None
    assert {(i["kind"], i["unit"]) for i in plan["items"]} >= {("frame", 2)}

    r = await client.post(f"/storyboard/{job_id}/recut", json={"targets": [{"kind": "frame", "unit": 2}]})
    assert r.status_code == 202 and r.json()["cut"] == 2
    assert await _status(job_id) == JobStatus.QUEUED.value
    (args, kwargs), = launched
    assert kwargs["fresh"] is True and args[3] is True  # resume, on a new thread
    await run_job(*args, **kwargs)

    assert await _status(job_id) == JobStatus.COMPLETED.value
    assert (job_dir / "script.json").stat().st_mtime_ns == script_mtime
    assert (job_dir / "frame_01.png").stat().st_mtime_ns == kept
    assert (job_dir / "frame_02.png").exists()
    assert (job_dir / "takes" / "c2-frame-2" / "frame_02.png").read_bytes() == old

    m = (await client.get(f"/storyboard/{job_id}/recut")).json()
    assert m["can_recut"] and [c["cut"] for c in m["cuts"]] == [1, 2] and m["cuts"][1]["current"]
    scene2 = next(s for s in m["scenes"] if s["order"] == 2)
    take = scene2["takes"][0]
    assert take["restorable"] and take["urls"]
    assert (await client.get(take["urls"][0].split("?")[0])).status_code == 200

    # Bring the old frame back: no new picture is drawn.
    r = await client.post(f"/storyboard/{job_id}/takes/{take['id']}/restore")
    assert r.status_code == 202 and r.json()["cut"] == 3
    (args, kwargs) = launched[-1]
    await run_job(*args, **kwargs)
    assert (job_dir / "frame_02.png").read_bytes() == old
    assert await _status(job_id) == JobStatus.COMPLETED.value


async def test_a_note_goes_through_the_director_and_lands_in_the_script(client, stub_video, artifact_root, launched):
    job_id = await _finished_film()
    job_dir = artifact_root / job_id
    (job_dir / "rescue_01.json").write_text("{}")
    plan = (await client.post(f"/storyboard/{job_id}/recut/plan",
                              json={"targets": [{"kind": "scene", "unit": 1}], "note": "make it raining"})).json()
    assert plan["directed"] and plan["plan_id"]
    diff = plan["scenes"][0]
    assert diff["order"] == 1 and "make it raining" in diff["description_after"]

    r = await client.post(f"/storyboard/{job_id}/recut", json={"plan_id": plan["plan_id"]})
    assert r.status_code == 202
    script = json.loads((job_dir / "script.json").read_text())
    assert "make it raining" in script["scenes"][0]["description"]
    assert not (job_dir / "rescue_01.json").exists()  # the new text supersedes the old content-policy rewrite
    cuts = tk.history(job_dir)
    assert cuts[-1]["note"] == "make it raining"


async def test_recut_needs_a_stopped_film_and_a_plan_for_scene_rewrites(client, stub_video, launched):
    job_id = await _finished_film()
    r = await client.post(f"/storyboard/{job_id}/recut", json={"targets": [{"kind": "scene", "unit": 1}]})
    assert r.status_code == 422
    r = await client.post(f"/storyboard/{job_id}/recut", json={"targets": [{"kind": "frame", "unit": 99}]})
    assert r.status_code == 422
    async with get_sessionmaker()() as s:
        await repo.set_status(s, job_id, JobStatus.RUNNING)
    r = await client.post(f"/storyboard/{job_id}/recut/plan", json={"targets": [{"kind": "frame", "unit": 1}]})
    assert r.status_code == 409 and "Pause" in r.json()["detail"]
    assert (await client.get(f"/storyboard/{job_id}/recut")).json()["can_recut"] is False
    assert not launched


async def test_review_frames_sets_a_checkpoint_before_filming(client, stub_video, artifact_root, launched, monkeypatch):
    monkeypatch.setattr("app.api.redo.pipeline_stage_names", lambda s: PIPE)
    job_id = await _finished_film()
    r = await client.post(f"/storyboard/{job_id}/recut",
                          json={"targets": [{"kind": "frame", "unit": 1}], "review_frames": True})
    assert r.status_code == 202
    async with get_sessionmaker()() as s:
        assert (await repo.get_job(s, job_id)).breakpoints == ["motion"]


# --- video stage: a blocked scene's history ---------------------------------- #
async def test_a_blocked_scene_resumes_from_its_last_rewrite(artifact_root, monkeypatch):
    from app.adapters.llm import LLMAdapter
    from app.agents.designer import _mock_shotlist
    from app.agents.video_gen import video_gen
    from app.artifacts.store import ArtifactStore
    from app.core.config import Settings
    from app.core.images import render_card
    from app.graph.state import Deps

    monkeypatch.setattr("app.agents.video_gen.has_audio", lambda p: True)
    s = Settings(gemini_api_key="", use_mock_providers=True, ai_video_retry_delay_seconds=0, omni_max_concurrency=1)
    store = ArtifactStore(artifact_root, "blocked-1")
    script = _script()
    seen: list[str] = []
    allow = {"on": False}

    class _V:
        def generate(self, img, spec, *, key):
            seen.append(spec.scene)
            if spec.order == 2 and not allow["on"]:
                raise omni.ContentBlocked("blocked", "prohibited_content")
            return omni.OmniClip(data=b"MP4", model="gemini-omni-1.2-flash", seconds=5, cost_usd=0.5, renders=1)

    deps = Deps(settings=s, store=store, llm=LLMAdapter(s), image=None, video=_V())
    frames = [{"order": i, "path": store.save_bytes(f"frame_{i:02d}.png", render_card(str(i))), "caption": "",
               "status": "ok"} for i in (1, 2, 3)]
    state = {"job_id": "blocked-1", "frames": frames, "script_path": store.save_json("script.json", script),
             "shotlist_path": store.save_json("shotlist.json", _mock_shotlist(script)), "language": "en"}
    with pytest.raises(Exception):
        await video_gen(state, deps)
    attempts = json.loads((store.job_dir / "rescue_02.attempts.json").read_text())
    assert attempts["code"] == "prohibited_content" and attempts["rewrites"]

    seen.clear()
    allow["on"] = True
    out = await video_gen(state, deps)
    assert seen == [attempts["rewrites"][-1]["scene"]]  # only scene 2, and from its last rewrite
    assert len(out["motion_clips"]) == 3
    assert store.exists("rescue_02.json") and not store.exists("rescue_02.attempts.json")
