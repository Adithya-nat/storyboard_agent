"""Resume + idempotency — the load-bearing correctness guarantees.

Covers the whole-job resume path (skip done work, regenerate only what's missing,
no divergent re-script, no duplicate artifact rows) and the /resume endpoint's
status-code contract.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.adapters.errors import PermanentProviderError
from app.adapters.gemini import GeminiImageAdapter
from app.adapters.llm import LLMAdapter
from app.agents.designer import designer
from app.agents.image_gen import image_gen
from app.agents.image_gen import image_gen as real_image_gen
from app.agents.script_writer import script_writer
from app.artifacts.store import ArtifactStore
from app.core.config import get_settings
from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.graph.build import build_graph
from app.graph.checkpoint import make_checkpointer
from app.graph.state import Deps
from app.main import app
from app.schemas import JobStatus
from app.workers.runner import run_job


@pytest.fixture
def stub_video(monkeypatch):
    async def _fake_assemble(timeline, output_path):
        Path(output_path).write_bytes(b"FAKE-MP4")
        return output_path

    monkeypatch.setattr("app.agents.assembler.assemble_video", _fake_assemble)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_resume_skips_done_work_and_regenerates_missing(stub_video, artifact_root):
    prompt = "a comet passes over a sleeping village"
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, prompt, 30)
    await run_job(job.id, prompt, 30)

    job_dir = artifact_root / job.id
    script_mtime = (job_dir / "script.json").stat().st_mtime_ns
    shotlist_mtime = (job_dir / "shotlist.json").stat().st_mtime_ns
    kept_frame = job_dir / "frame_01.png"
    kept_mtime = kept_frame.stat().st_mtime_ns

    # Simulate an interrupted job: drop the video + a few frames, mark interrupted.
    (job_dir / "storyboard.mp4").unlink()
    for n in (5, 6, 7):
        (job_dir / f"frame_{n:02d}.png").unlink()
    async with get_sessionmaker()() as s:
        await repo.set_status(s, job.id, JobStatus.INTERRUPTED)

    await run_job(job.id, prompt, 30, resume=True)

    async with get_sessionmaker()() as s:
        done = await repo.get_job(s, job.id)

    assert done.status == JobStatus.COMPLETED.value
    # Upstream artifacts NOT regenerated -> idempotency held, no divergent script.
    assert (job_dir / "script.json").stat().st_mtime_ns == script_mtime
    assert (job_dir / "shotlist.json").stat().st_mtime_ns == shotlist_mtime
    # A surviving frame was reused, not rewritten.
    assert kept_frame.stat().st_mtime_ns == kept_mtime
    # Deleted frames regenerated; full set present again; video rebuilt.
    assert len(list(job_dir.glob("frame_*.png"))) == 15
    assert (job_dir / "storyboard.mp4").exists()
    # Re-persist REPLACED rather than appended -> no duplicate artifact rows.
    counts = Counter(a.kind for a in done.artifacts)
    assert counts["frame"] == 15
    assert counts["script"] == 1
    assert counts["video"] == 1


async def test_resume_continues_from_crash_checkpoint(stub_video, artifact_root, monkeypatch):
    """The REAL crash-resume path: a node raises mid-graph so the checkpoint is
    left pending at image_gen, then resume continues via ainvoke(None) — the only
    path where the LangGraph checkpointer (not idempotency) is load-bearing."""
    prompt = "a storm at sea swallows a paper boat"
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, prompt, 30)

    calls = {"n": 0}

    async def _flaky(state, deps):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("crash mid-image_gen")
        return await real_image_gen(state, deps)

    # Patch the name build_graph binds (it does `from ...image_gen import image_gen`).
    monkeypatch.setattr("app.graph.build.image_gen", _flaky)

    # First run crashes inside image_gen -> FAILED, designer already checkpointed.
    await run_job(job.id, prompt, 30)
    async with get_sessionmaker()() as s:
        failed = await repo.get_job(s, job.id)
    assert failed.status == JobStatus.FAILED.value

    job_dir = artifact_root / job.id
    assert (job_dir / "shotlist.json").exists()  # designer finished before the crash
    shotlist_mtime = (job_dir / "shotlist.json").stat().st_mtime_ns

    # The crash must have left a checkpoint PENDING at image_gen (else the resume
    # would silently fall back to a fresh re-run and the checkpointer is dead code).
    settings = get_settings()
    deps = Deps(
        settings=settings,
        store=ArtifactStore(settings.artifact_root, job.id),
        llm=LLMAdapter(settings),
        image=GeminiImageAdapter(settings),
    )
    async with make_checkpointer(settings) as cp:
        snap = await build_graph(deps, checkpointer=cp).aget_state(
            {"configurable": {"thread_id": job.id}}
        )
    assert snap.next == ("image_gen",)

    # Resume -> hits the snapshot.next branch (ainvoke(None)) and finishes.
    await run_job(job.id, prompt, 30, resume=True)
    async with get_sessionmaker()() as s:
        done = await repo.get_job(s, job.id)

    assert done.status == JobStatus.COMPLETED.value
    assert calls["n"] == 2  # image_gen ran exactly once more (the continuation)
    assert (job_dir / "shotlist.json").stat().st_mtime_ns == shotlist_mtime  # not re-run
    assert len(list(job_dir.glob("frame_*.png"))) == 15
    counts = Counter(a.kind for a in done.artifacts)
    assert counts["frame"] == 15
    assert counts["video"] == 1


class _AlwaysFailsImage:
    def generate_image(self, *a, **k):
        raise PermanentProviderError("safety block")


async def test_resume_retries_degraded_frame(artifact_root):
    """A degraded frame keeps its `.degraded` marker, so resume RE-TRIES it rather
    than skipping it — the final status/warnings stay honest across a restart."""
    settings = get_settings()
    store = ArtifactStore(artifact_root, "ig-degraded-retry")
    ok_deps = Deps(
        settings=settings, store=store,
        llm=LLMAdapter(settings), image=GeminiImageAdapter(settings), max_frames=3,
    )
    sw = await script_writer({"prompt": "x", "duration": 30}, ok_deps)
    dz = await designer({"script_path": sw["script_path"], "duration": 30}, ok_deps)
    state = {
        "shotlist_path": dz["shotlist_path"],
        "style_ref_path": dz["style_ref_path"],
        "duration": 30,
    }

    # Run 1: failing adapter -> all frames degraded + markers written.
    fail_deps = Deps(
        settings=settings, store=store,
        llm=LLMAdapter(settings), image=_AlwaysFailsImage(), max_frames=3,
    )
    out1 = await image_gen(state, fail_deps)
    assert all(f["status"] == "degraded" for f in out1["frames"])
    assert store.exists("frame_01.degraded")

    # Run 2 (resume): working adapter -> degraded frames re-tried to ok, markers cleared.
    out2 = await image_gen(state, ok_deps)
    assert all(f["status"] == "ok" for f in out2["frames"])
    assert out2["warnings"] == []
    assert not store.exists("frame_01.degraded")


async def test_run_job_survives_unavailable_checkpointer(stub_video, monkeypatch):
    """If the checkpointer can't be acquired, the job must still COMPLETE (resume
    correctness rests on idempotency, not the checkpointer) — never hang or fail."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _broken(settings):
        raise RuntimeError("checkpointer down")
        yield  # pragma: no cover - unreachable, makes this a valid async CM

    monkeypatch.setattr("app.workers.runner.make_checkpointer", _broken)

    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "resilient prompt", 30)
    await run_job(job.id, "resilient prompt", 30)

    async with get_sessionmaker()() as s:
        done = await repo.get_job(s, job.id)
    assert done.status == JobStatus.COMPLETED.value
    assert sum(1 for a in done.artifacts if a.kind == "frame") == 15


async def test_resume_endpoint_status_codes(client, monkeypatch):
    # Stub the background runner: this test checks the endpoint contract only.
    async def _noop(*a, **k):
        return None

    monkeypatch.setattr("app.api.routes.run_job", _noop)

    # unknown -> 404
    assert (await client.post("/storyboard/does-not-exist/resume")).status_code == 404

    async def _seed(status):
        async with get_sessionmaker()() as s:
            job = await repo.create_job(s, "p", 30)
            await repo.set_status(s, job.id, status)
        return job.id

    interrupted = await _seed(JobStatus.INTERRUPTED)
    assert (await client.post(f"/storyboard/{interrupted}/resume")).status_code == 202

    failed = await _seed(JobStatus.FAILED)
    assert (await client.post(f"/storyboard/{failed}/resume")).status_code == 202

    running = await _seed(JobStatus.RUNNING)
    assert (await client.post(f"/storyboard/{running}/resume")).status_code == 409

    completed = await _seed(JobStatus.COMPLETED)
    assert (await client.post(f"/storyboard/{completed}/resume")).status_code == 409
