"""Run controls: pause at a unit boundary, resume without re-billing, breakpoints,
withdrawing a pause mid-drain, and the checkpoint (per-unit state + cost to finish)."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import update

from app.adapters import omni
from app.core.config import Settings
from app.db import repository as repo
from app.db.models import Job
from app.db.session import get_sessionmaker
from app.schemas import JobStatus
from app.workers import control
from app.workers.control import JobPaused, RunControl


@pytest.fixture
async def client():
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _no_ffprobe(monkeypatch):
    monkeypatch.setattr("app.agents.video_gen.has_audio", lambda p: True)


class _Ctl(RunControl):
    """Control whose pause flag the fake provider flips (the fake runs in a worker thread)."""

    def __init__(self, job_id: str) -> None:
        super().__init__(job_id)
        self.flag = False

    async def requested(self) -> bool:
        return self.flag


async def _video_stage(artifact_root, job: str, scenes: int, pause_on: int | None):
    from app.adapters.llm import LLMAdapter
    from app.agents.designer import _mock_shotlist
    from app.agents.script_writer import _mock_script
    from app.artifacts.store import ArtifactStore
    from app.core.images import render_card
    from app.graph.state import Deps

    s = Settings(gemini_api_key="", use_mock_providers=True, ai_video_retry_delay_seconds=0, omni_max_concurrency=1)
    store = ArtifactStore(artifact_root, job)
    script = _mock_script("x", scenes, "dialogue")
    calls: list[int] = []
    ctl = _Ctl(job)

    class _V:
        def generate(self, img, spec, *, key):
            calls.append(spec.order)
            if spec.order == pause_on:
                ctl.flag = True  # the user hits Pause while this scene is rendering
            return omni.OmniClip(data=b"MP4", model="gemini-omni-1.2-flash", seconds=5, cost_usd=0.5, renders=1)

    deps = Deps(settings=s, store=store, llm=LLMAdapter(s), image=None, video=_V(), control=ctl)
    frames = [{"order": i, "path": store.save_bytes(f"frame_{i:02d}.png", render_card(str(i))), "caption": "",
               "status": "ok"} for i in range(1, scenes + 1)]
    state = {"job_id": job, "frames": frames, "script_path": store.save_json("script.json", script),
             "shotlist_path": store.save_json("shotlist.json", _mock_shotlist(script)), "language": "en"}
    from app.agents.video_gen import video_gen

    return video_gen, state, deps, calls, ctl


# --- unit boundary: the core cost guarantee --------------------------------- #
async def test_pause_mid_video_keeps_the_inflight_clip_and_resume_never_rebills(artifact_root):
    video_gen, state, deps, calls, ctl = await _video_stage(artifact_root, "pause-1", 6, pause_on=3)

    with pytest.raises(JobPaused) as paused:
        await video_gen(state, deps)
    assert paused.value.stage == "motion"
    assert "3 of 6" in paused.value.message and "3 left" in paused.value.message
    # Scene 3 was already submitted (billed either way), so it finished and was kept;
    # scenes 4-6 never started.
    assert calls == [1, 2, 3]
    assert all(deps.store.exists(f"motion_{i:02d}.mp4") for i in (1, 2, 3))
    assert not deps.store.exists("motion_04.mp4")

    ctl.flag = False  # Resume
    out = await video_gen(state, deps)
    assert calls == [1, 2, 3, 4, 5, 6]  # each scene billed exactly once across pause/resume
    assert [c["order"] for c in out["motion_clips"]] == [1, 2, 3, 4, 5, 6]


async def test_pause_before_a_scene_submits_resumes_at_that_scene(artifact_root):
    video_gen, state, deps, calls, ctl = await _video_stage(artifact_root, "pause-2", 6, pause_on=2)
    with pytest.raises(JobPaused):
        await video_gen(state, deps)
    assert calls == [1, 2]

    ctl.flag = False
    await video_gen(state, deps)
    assert calls == [1, 2, 3, 4, 5, 6]  # resumed at scene 3, nothing repeated


async def test_pause_during_frames_keeps_finished_frames_and_skips_the_neighbour_fill(artifact_root, monkeypatch):
    from app.adapters.llm import LLMAdapter
    from app.agents.designer import _mock_shotlist
    from app.agents.image_gen import image_gen
    from app.agents.script_writer import _mock_script
    from app.artifacts.store import ArtifactStore
    from app.graph.state import Deps

    s = Settings(gemini_api_key="", use_mock_providers=True, image_max_concurrency=1)
    store = ArtifactStore(artifact_root, "pause-3")
    script = _mock_script("x", 4, "dialogue")
    ctl = _Ctl("pause-3")
    rendered: list[int] = []

    class _Img:
        def generate_image(self, prompt, refs=None, seed=None):
            rendered.append(len(rendered) + 1)
            if len(rendered) == 2:
                ctl.flag = True
            return b"\x89PNG-frame"

    deps = Deps(settings=s, store=store, llm=LLMAdapter(s), image=_Img(), control=ctl)
    monkeypatch.setattr("app.agents.image_gen._identity_mismatch", lambda *a, **k: False)
    state = {"job_id": "pause-3", "script_path": store.save_json("script.json", script),
             "shotlist_path": store.save_json("shotlist.json", _mock_shotlist(script))}

    with pytest.raises(JobPaused) as paused:
        await image_gen(state, deps)
    assert paused.value.stage == "image_gen" and "2 of 4" in paused.value.message
    assert [store.exists(f"frame_{i:02d}.png") for i in (1, 2, 3, 4)] == [True, True, False, False]

    ctl.flag = False
    out = await image_gen(state, deps)
    assert len(rendered) == 4  # frames 1-2 reused, 3-4 rendered once
    assert all(f["status"] == "ok" for f in out["frames"])


# --- whole run: breakpoints + resume ---------------------------------------- #
@pytest.fixture
def stub_video(monkeypatch):
    from pathlib import Path

    async def _fake_assemble(timeline, output_path):
        Path(output_path).write_bytes(b"FAKE-MP4")
        return output_path

    monkeypatch.setattr("app.agents.assembler.assemble_video", _fake_assemble)


async def test_breakpoint_parks_the_run_before_the_stage_then_resume_finishes(stub_video, artifact_root):
    from app.workers.runner import run_job

    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "a lighthouse keeper's last night", 30)
        await repo.set_breakpoints(s, job.id, ["image_gen"])

    await run_job(job.id, job.prompt, 30)

    async with get_sessionmaker()() as s:
        parked = await repo.get_job(s, job.id)
        stages = {st.name: st for st in await repo.get_stages(s, job.id)}
    assert parked.status == JobStatus.PAUSED.value
    assert parked.pause_requested_at is None and not parked.breakpoints  # consumed
    assert stages["image_gen"].status == "paused"
    assert stages["designer"].status == "succeeded"
    job_dir = artifact_root / job.id
    assert (job_dir / "script.json").exists() and not list(job_dir.glob("frame_*.png"))
    script_mtime = (job_dir / "script.json").stat().st_mtime_ns

    await run_job(job.id, job.prompt, 30, resume=True)

    async with get_sessionmaker()() as s:
        done = await repo.get_job(s, job.id)
    assert done.status == JobStatus.COMPLETED.value
    assert (job_dir / "script.json").stat().st_mtime_ns == script_mtime  # nothing upstream redone


# --- API: pause / withdraw / breakpoints / checkpoint ------------------------ #
async def _job(status: JobStatus) -> str:
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
        await repo.set_status(s, job.id, status)
    return job.id


async def test_pause_endpoint_then_resume_while_draining_withdraws_the_pause(client):
    job_id = await _job(JobStatus.RUNNING)

    r = await client.post(f"/storyboard/{job_id}/pause")
    assert r.status_code == 202 and r.json()["status"] == "pausing"
    body = (await client.get(f"/storyboard/{job_id}")).json()
    assert body["status"] == "pausing" and body["pause_requested"] is True
    assert (await client.post(f"/storyboard/{job_id}/pause")).status_code == 202  # idempotent

    r = await client.post(f"/storyboard/{job_id}/resume")
    assert r.status_code == 202 and r.json()["status"] == "running"
    async with get_sessionmaker()() as s:
        assert not await repo.pause_requested(s, job_id)
        assert not await repo.park_job(s, job_id)  # the draining runner now carries on
    control.forget(job_id)


async def test_pause_needs_a_running_film(client):
    job_id = await _job(JobStatus.COMPLETED)
    assert (await client.post(f"/storyboard/{job_id}/pause")).status_code == 409


async def test_resume_from_paused_starts_a_runner(client, monkeypatch):
    job_id = await _job(JobStatus.PAUSED)
    seen = {}

    async def _capture(*a, **k):
        seen["args"] = a

    monkeypatch.setattr("app.api.routes.run_job", _capture)
    r = await client.post(f"/storyboard/{job_id}/resume")
    assert r.status_code == 202 and seen["args"][3] is True  # resume=True


async def test_breakpoints_validate_stage_names_and_skip_started_stages(client):
    job_id = await _job(JobStatus.RUNNING)
    r = await client.put(f"/storyboard/{job_id}/breakpoints", json={"stages": ["nope"]})
    assert r.status_code == 422
    async with get_sessionmaker()() as s:
        await repo.upsert_stage_running(s, job_id, "brief", 1)
    r = await client.put(f"/storyboard/{job_id}/breakpoints", json={"stages": ["brief", "music", "music"]})
    assert r.json()["breakpoints"] == ["music"]
    assert (await client.get(f"/storyboard/{job_id}")).json()["breakpoints"] == ["music"]


async def test_checkpoint_reports_units_resume_point_and_cost_to_finish(client, artifact_root, monkeypatch):
    from app.agents.designer import _mock_shotlist
    from app.graph import build

    monkeypatch.setattr("app.api.controls.pipeline_stage_names",
                        lambda s: build.pipeline_stage_names(s.model_copy(update={"video_provider": "omni"})))
    from app.agents.script_writer import _mock_script
    from app.artifacts.store import ArtifactStore

    job_id = await _job(JobStatus.PAUSED)
    store = ArtifactStore(artifact_root, job_id)
    script = _mock_script("x", 4, "dialogue")
    store.save_json("script.json", script)
    store.save_json("shotlist.json", _mock_shotlist(script))
    for i in (1, 2, 3, 4):
        store.save_bytes(f"frame_{i:02d}.png", b"png")
    for i in (1, 2):
        store.save_bytes(f"motion_{i:02d}.mp4", b"MP4")
    async with get_sessionmaker()() as s:
        for seq, name in enumerate(["brief", "script_writer", "dialogue_pass", "reflection", "designer",
                                    "image_gen", "narration"], start=1):
            await repo.upsert_stage_running(s, job_id, name, seq)
            await repo.flip_stage_if_running(s, job_id, name, "succeeded")
        await repo.pause_stage(s, job_id, "motion", 8, "Paused after 2 of 4 AI clips; 2 left")

    cp = (await client.get(f"/storyboard/{job_id}/checkpoint")).json()
    assert cp["paused_stage"] == "motion" and cp["resume_stage"] == "motion" and cp["resume_unit"] == 3
    motion = next(st for st in cp["stages"] if st["name"] == "motion")
    assert motion["units"]["done"] == [1, 2] and motion["units"]["pending"] == [3, 4]
    assert motion["remaining_usd"] == pytest.approx(motion["units"]["unit_cost_usd"] * 2, abs=1e-3)
    assert motion["remaining_usd"] > 0
    frames = next(st for st in cp["stages"] if st["name"] == "image_gen")
    assert frames["remaining_usd"] == 0 and frames["units"]["done"] == [1, 2, 3, 4]
    assert cp["remaining_usd"] >= motion["remaining_usd"]


# --- lifecycle: expiry + restarts ------------------------------------------- #
async def test_a_paused_film_outlives_the_normal_ttl():
    job_id = await _job(JobStatus.PAUSED)
    old = datetime.now(timezone.utc) - timedelta(days=2)
    async with get_sessionmaker()() as s:
        await s.execute(update(Job).where(Job.id == job_id).values(created_at=old))
        await s.commit()
        assert await repo.delete_expired_jobs(s, 86400, 7 * 86400) == []
        await s.execute(update(Job).where(Job.id == job_id).values(updated_at=old - timedelta(days=6)))
        await s.commit()
        assert await repo.delete_expired_jobs(s, 86400, 7 * 86400) == [job_id]


async def test_restart_while_pausing_parks_the_film_as_paused():
    job_id = await _job(JobStatus.RUNNING)
    async with get_sessionmaker()() as s:
        await repo.upsert_stage_running(s, job_id, "motion", 8)
        assert await repo.request_pause(s, job_id)
        await repo.reap_orphaned_jobs(s)
        job = await repo.get_job(s, job_id)
        stages = await repo.get_stages(s, job_id)
    assert job.status == JobStatus.PAUSED.value
    assert stages[0].status == "paused"


# --- Tamil: English renderings for the UI ----------------------------------- #
async def test_tamil_strings_are_glossed_once_and_cached(artifact_root):
    from app.adapters.llm import LLMAdapter
    from app.agents import gloss
    from app.artifacts.store import ArtifactStore
    from app.graph.state import Deps

    s = Settings(gemini_api_key="", use_mock_providers=True)
    calls: list[str] = []

    class _LLM(LLMAdapter):
        def complete_json(self, *, system, user, response_model, mock_factory, role="script"):
            calls.append(user)
            return response_model(items=[{"i": 0, "en": "The last bus"}])

    deps = Deps(settings=s, store=ArtifactStore(artifact_root, "gloss-1"), llm=_LLM(s), image=None)
    out = await gloss.ensure(deps, ["கடைசி பஸ்", "Already English", ""])
    assert out == {"கடைசி பஸ்": "The last bus"}
    await gloss.ensure(deps, ["கடைசி பஸ்"])
    assert len(calls) == 1  # cached: a resume or second read costs nothing


async def test_storyboard_content_carries_english_renderings(client, artifact_root):
    from app.artifacts.store import ArtifactStore

    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30, language="ta")
        store = ArtifactStore(artifact_root, job.id)
        path = store.save_json("script.json", {"title": "கடைசி பஸ்", "logline": "", "scenes": []})
        store.save_json("gloss.json", {"கடைசி பஸ்": "The last bus"})
        await repo.replace_artifacts(s, job.id, [{"kind": "script", "path": path}])
        await repo.set_status(s, job.id, JobStatus.COMPLETED)

    body = (await client.get(f"/storyboard/{job.id}/storyboard")).json()
    assert body["title"] == "கடைசி பஸ்" and body["title_en"] == "The last bus"


# --- Hosted mode: run controls are per-user ---------------------------------- #
async def test_controls_hide_another_users_film(client, monkeypatch):
    from types import SimpleNamespace

    from app.core import auth
    from app.core.config import get_settings
    from tests import test_hosted as h

    live = get_settings()
    for k, v in {"auth_mode": "jwt", "auth_jwks_url": "http://web.test/api/auth/jwks", "auth_issuer": h.ISS,
                 "allowed_users": ["ada@example.com", "bob@example.com"], "asset_signing_secret": "s" * 32}.items():
        monkeypatch.setattr(live, k, v)
    monkeypatch.setattr(auth, "_jwks_client", lambda url: SimpleNamespace(
        get_signing_key_from_jwt=lambda t: SimpleNamespace(key=h._PRIV.public_key())))

    async with get_sessionmaker()() as sess:
        job = await repo.create_job(sess, "p", 30, user_id="ada")
        await repo.set_status(sess, job.id, JobStatus.RUNNING)
    bob = h.bearer(sub="bob", email="bob@example.com")
    assert (await client.post(f"/storyboard/{job.id}/pause", headers=bob)).status_code == 404
    assert (await client.get(f"/storyboard/{job.id}/checkpoint", headers=bob)).status_code == 404
    r = await client.put(f"/storyboard/{job.id}/breakpoints", json={"stages": []}, headers=bob)
    assert r.status_code == 404
    ada = h.bearer(sub="ada", email="ada@example.com")
    assert (await client.post(f"/storyboard/{job.id}/pause", headers=ada)).status_code == 202
    control.forget(job.id)


def test_mock_omni_makes_a_local_clip_without_the_network(tmp_path):
    from app.adapters.omni import OmniVideoAdapter, SceneSpec
    from app.core.images import render_card

    s = Settings(gemini_api_key="", use_mock_providers=True)
    adapter = OmniVideoAdapter(s, journal_dir=tmp_path)
    adapter._client = None  # any client use would go through make_client(), which refuses in mock mode
    clip = adapter.generate(render_card("x"), SceneSpec(order=1, scene="x", seconds=3.0), key="k")
    assert clip.data[4:8] == b"ftyp" and clip.cost_usd == 0.0 and clip.model == "mock-omni"


# --- Review fixes (CodeRabbit on creator-agent#3) ---------------------------- #
from tests.test_hosted import bearer, captured, hosted  # noqa: E402,F401 - reused hosted fixtures


async def test_a_film_still_pausing_counts_toward_the_active_limit(hosted, client, captured):  # noqa: F811
    async with get_sessionmaker()() as s:
        film_a = await repo.create_job(s, "a", 30, user_id="u1")
        await repo.set_status(s, film_a.id, JobStatus.RUNNING)
        assert await repo.request_pause(s, film_a.id)  # A drains its paid clips: still a live runner
    r = await client.post("/storyboard", json={"prompt": "film b", "duration": 30}, headers=bearer())
    assert r.status_code == 429 and not captured
    control.forget(film_a.id)


async def test_checkpoint_history_only_averages_the_callers_own_runs(hosted, client):  # noqa: F811
    from app.db.models import RunLedger, SpendEvent

    async with get_sessionmaker()() as s:
        for jid, uid, cost in (("other-run", "someone-else", 9.0), ("own-run", "u1", 1.0)):
            s.add(RunLedger(job_id=jid, user_id=uid, status="completed",
                            started_at=datetime.now(timezone.utc)))
            s.add(SpendEvent(job_id=jid, user_id=uid, stage="music", category="music", model="m", cost_usd=cost))
        await s.commit()
        mine = await repo.create_job(s, "p", 30, user_id="u1")
        await repo.set_status(s, mine.id, JobStatus.PAUSED)
    cp = (await client.get(f"/storyboard/{mine.id}/checkpoint", headers=bearer())).json()
    music = next(st for st in cp["stages"] if st["name"] == "music")
    assert music["remaining_usd"] == pytest.approx(1.0)  # not 5.0, the cross-user average


async def test_a_pause_signalled_during_the_db_read_still_stops_the_next_unit(monkeypatch):
    from app.workers.control import signal_pause

    async def read_then_signal(session, job_id):
        signal_pause(job_id)  # the API process signals while our read is in flight
        return False

    monkeypatch.setattr("app.db.repository.pause_requested", read_then_signal)
    ctl = RunControl("race-1", get_sessionmaker(), poll_seconds=0)
    assert await ctl.requested() is True
    control.forget("race-1")


async def test_retry_wait_ends_early_on_a_pause_recorded_only_in_the_db():
    job_id = await _job(JobStatus.RUNNING)
    async with get_sessionmaker()() as s:
        assert await repo.request_pause(s, job_id)  # another process: DB flag, no local event
    control.clear_pause(job_id)
    started = time.monotonic()
    assert await RunControl(job_id, get_sessionmaker(), poll_seconds=0.05).wait(30) is True
    assert time.monotonic() - started < 2
    control.forget(job_id)


async def test_a_failed_narration_line_is_retried_after_a_pause(artifact_root):
    from app.adapters.errors import ProviderError
    from app.adapters.llm import LLMAdapter
    from app.adapters.tts import silent_wav
    from app.agents.narration import voice_lines
    from app.artifacts.store import ArtifactStore
    from app.graph.state import Deps

    s = Settings(gemini_api_key="", use_mock_providers=True)
    ctl = _Ctl("narr-1")
    calls: list[str] = []

    class _TTS:
        def synthesize(self, text):
            calls.append(text)
            if text == "one" and calls.count("one") == 1:
                raise ProviderError("tts down")  # the first try at line 1 fails
            ctl.flag = text == "two" and calls.count("two") == 1  # pause lands after line 2 starts
            return silent_wav(1.0)

    deps = Deps(settings=s, store=ArtifactStore(artifact_root, "narr-1"), llm=LLMAdapter(s), image=None,
                tts=_TTS(), control=ctl)
    with pytest.raises(JobPaused):
        await voice_lines([(1, "one"), (2, "two"), (3, "three")], deps)
    ctl.flag = False
    await voice_lines([(1, "one"), (2, "two"), (3, "three")], deps)
    assert calls == ["one", "two", "one", "three"]  # line 1 retried for real; line 2 not re-billed
    assert not deps.store.exists("narration_01.failed")


# --- Review fixes, round 2 (CodeRabbit re-review on creator-agent#3) --------- #
def test_hosted_mode_rejects_a_longer_paused_retention():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="PAUSED_TTL_SECONDS"):
        Settings(auth_mode="jwt", auth_jwks_url="http://web.test/jwks", paused_ttl_seconds=7 * 86400)
    assert Settings(auth_mode="none", paused_ttl_seconds=7 * 86400).paused_ttl_seconds  # self-host opt-in


async def test_the_sweeper_never_deletes_a_film_that_started_after_its_scan():
    from app.db.repository import _purge_expired

    job_id = await _job(JobStatus.FAILED)
    async with get_sessionmaker()() as s:
        candidates = [job_id]  # the scan saw it failed and expired...
        assert await repo.mark_queued(s, job_id)  # ...then a resume admitted it
        assert await repo.try_acquire_job(s, job_id)  # ...and its runner started
        assert await _purge_expired(s, candidates) == []  # re-checked under the lock: kept
        assert (await repo.get_job(s, job_id)).status == JobStatus.RUNNING.value
        await repo.set_status(s, job_id, JobStatus.FAILED)
        assert await _purge_expired(s, candidates) == [job_id]  # not running: the 24h rule applies


async def test_a_parking_runner_finishes_its_writes_before_the_job_is_resumable(stub_video, monkeypatch):
    from app.workers import runner

    order: list[str] = []
    real_park, real_ledger = repo.park_job, runner._finish_ledger

    async def park(session, job_id):
        order.append("park")
        return await real_park(session, job_id)

    async def ledger(*a, **k):
        order.append(f"ledger:{a[3]}")
        return await real_ledger(*a, **k)

    monkeypatch.setattr("app.db.repository.park_job", park)
    monkeypatch.setattr(runner, "_finish_ledger", ledger)
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "a quiet harbour at dawn", 30)
        await repo.set_breakpoints(s, job.id, ["image_gen"])
    await runner.run_job(job.id, job.prompt, 30)
    assert order == ["ledger:paused", "park"]  # nothing of the old run lands after parking


async def test_a_withdrawn_pause_does_not_re_pause_from_the_cached_flag():
    from app.workers.runner import _drive

    job_id = await _job(JobStatus.RUNNING)  # running, no pause flag: park_job refuses
    ctl = RunControl(job_id, get_sessionmaker(), poll_seconds=3600)
    ctl._db_flag, ctl._checked = True, time.monotonic()  # cached "paused" from before the withdraw
    seen: list[bool] = []

    class _Graph:
        calls = 0

        async def ainvoke(self, init, config=None):
            _Graph.calls += 1
            if _Graph.calls == 1:
                raise JobPaused("motion", "paused")
            seen.append(await ctl.requested())
            return {"ok": True}

    final, paused = await _drive(_Graph(), {"job_id": job_id}, {}, False, checkpointed=False, control=ctl)
    assert paused is None and final == {"ok": True}
    assert seen == [False]  # the cache was reset: the next check read the DB
    control.forget(job_id)


async def test_the_failure_marker_survives_a_failed_save_of_the_retried_line(artifact_root, monkeypatch):
    from app.adapters.errors import ProviderError
    from app.adapters.llm import LLMAdapter
    from app.adapters.tts import silent_wav
    from app.agents.narration import voice_lines
    from app.artifacts.store import ArtifactStore
    from app.graph.state import Deps

    s = Settings(gemini_api_key="", use_mock_providers=True)
    tries: list[int] = []

    class _TTS:
        def synthesize(self, text):
            tries.append(1)
            if len(tries) == 1:
                raise ProviderError("tts down")
            return silent_wav(2.0)

    store = ArtifactStore(artifact_root, "narr-2")
    deps = Deps(settings=s, store=store, llm=LLMAdapter(s), image=None, tts=_TTS())
    await voice_lines([(1, "one")], deps)  # fails: silence + marker
    assert store.exists("narration_01.failed")

    real_save = store.save_bytes

    def broken_save(name, data):
        raise OSError("disk full")

    monkeypatch.setattr(store, "save_bytes", broken_save)
    with pytest.raises(OSError):
        await voice_lines([(1, "one")], deps)  # the retry works, but saving it fails
    assert store.exists("narration_01.failed")  # so the silence is still marked for a retry
    monkeypatch.setattr(store, "save_bytes", real_save)
    await voice_lines([(1, "one")], deps)
    assert not store.exists("narration_01.failed") and len(tries) == 3
