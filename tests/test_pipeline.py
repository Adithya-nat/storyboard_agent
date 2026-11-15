"""End-to-end runner test with mock providers and a stubbed video step.

Covers: script_writer -> image_gen -> assembler wiring + DB persistence + final
status. ffmpeg is stubbed so the test is hermetic (real ffmpeg is exercised via
the README verification flow, not unit tests).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.schemas import JobStatus


@pytest.fixture
def stub_video(monkeypatch):
    async def _fake_assemble(timeline, output_path):
        Path(output_path).write_bytes(b"FAKE-MP4")
        return output_path

    monkeypatch.setattr("app.agents.assembler.assemble_video", _fake_assemble)


async def test_full_pipeline_completes(stub_video):
    from app.workers.runner import run_job

    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "a lonely robot finds a flower", 30)

    await run_job(job.id, "a lonely robot finds a flower", 30)

    async with get_sessionmaker()() as s:
        done = await repo.get_job(s, job.id)

    assert done.status == JobStatus.COMPLETED.value
    # duration 30 @ 2s/frame -> 15 frames, plus one each of the designer + output
    # artifacts. Assert by kind/count, not a brittle exact-list.
    counts = Counter(a.kind for a in done.artifacts)
    assert counts["frame"] == 15
    assert counts["script"] == 1
    assert counts["shotlist"] == 1
    assert counts["style_bible"] == 1
    assert counts["style_ref"] == 1
    assert counts["video"] == 1
    video = next(a for a in done.artifacts if a.kind == "video")
    assert Path(video.path).exists()


async def test_pipeline_degraded_completes_with_warnings(stub_video, monkeypatch):
    """4A end-to-end: frames fail -> video still produced -> completed_with_warnings.

    This drives the *runner* through the degraded path (the _persist
    `degraded -> COMPLETED_WITH_WARNINGS` branch), which the per-agent and
    happy-path tests don't exercise together.
    """
    from app.adapters.errors import PermanentProviderError
    from app.adapters.gemini import GeminiImageAdapter
    from app.workers.runner import run_job

    def _always_fail(self, *a, **k):
        raise PermanentProviderError("safety block")

    monkeypatch.setattr(GeminiImageAdapter, "generate_image", _always_fail)

    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "forbidden prompt", 30)

    await run_job(job.id, "forbidden prompt", 30)

    async with get_sessionmaker()() as s:
        done = await repo.get_job(s, job.id)

    assert done.status == JobStatus.COMPLETED_WITH_WARNINGS.value
    assert done.warnings  # non-empty, surfaced to the client
    assert all(a.status == "degraded" for a in done.artifacts if a.kind == "frame")
    assert any(a.kind == "video" for a in done.artifacts)  # video still produced


async def test_pipeline_records_stages_and_logs(stub_video):
    """Observability end-to-end: a full mock run records all 4 stages (direct
    writes) succeeded + in order, and captures logs (best-effort, drained here)."""
    from app.observability import LogDrainer
    from app.observability.logs import reset_log_queue
    from app.workers.runner import run_job

    reset_log_queue()
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "a robot waters a plant", 30)

    await run_job(job.id, "a robot waters a plant", 30)

    # Stages are DIRECT synchronous writes -> assert ordering + all succeeded.
    async with get_sessionmaker()() as s:
        stages = await repo.get_stages(s, job.id)
    assert [st.name for st in stages] == [
        "script_writer", "reflection", "designer", "image_gen", "narration",
        "captions", "music", "assembler", "publish_kit",
    ]
    # Narrated + TTS/music pinned off in tests -> those stages report skipped.
    skipped = {"narration", "captions", "music"}
    assert all(st.status == ("skipped" if st.name in skipped else "succeeded") for st in stages)
    img = next(st for st in stages if st.name == "image_gen")
    assert img.progress_current == img.progress_total == 15  # 30s @ 2s/frame

    # Logs are BEST-EFFORT via the queue: lifespan (the drainer) doesn't run under
    # ASGITransport, so DRAIN explicitly before asserting — never race the drainer.
    drainer = LogDrainer(get_sessionmaker(), batch=1000)
    await drainer.drain_once()
    async with get_sessionmaker()() as s:
        logs = await repo.get_logs(s, job.id)
    assert logs, "expected captured log lines (image_gen progress)"
    assert any(lg.stage == "image_gen" for lg in logs)


async def test_pipeline_failure_sets_failed_status(monkeypatch):
    from app.workers.runner import run_job

    async def _boom(timeline, output_path):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr("app.agents.assembler.assemble_video", _boom)

    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)

    await run_job(job.id, "p", 30)

    async with get_sessionmaker()() as s:
        done = await repo.get_job(s, job.id)
    assert done.status == JobStatus.FAILED.value
    assert "ffmpeg exploded" in done.error
