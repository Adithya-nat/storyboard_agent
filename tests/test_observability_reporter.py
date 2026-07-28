"""ProgressReporter -> job_stages: the direct (reliable) write path.

Covers the stage state machine: running -> succeeded / failed / degraded, the
flip-iff-running guard (degraded is not overwritten by stage_finish), progress
updates, and the resume re-entry reset.
"""

from __future__ import annotations

from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.observability import bind_stage, reset_stage
from app.observability.reporter import ProgressReporter


async def _new_job() -> str:
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
    return job.id


async def _stage(job_id: str, name: str):
    async with get_sessionmaker()() as s:
        return {st.name: st for st in await repo.get_stages(s, job_id)}[name]


async def test_start_then_finish_succeeded():
    job_id = await _new_job()
    r = ProgressReporter(get_sessionmaker())
    await r.stage_start(job_id, "script_writer", 1)
    st = await _stage(job_id, "script_writer")
    assert st.status == "running" and st.started_at is not None and st.seq == 1

    await r.stage_finish(job_id, "script_writer")
    st = await _stage(job_id, "script_writer")
    assert st.status == "succeeded" and st.ended_at is not None


async def test_fail_records_exception_message():
    job_id = await _new_job()
    r = ProgressReporter(get_sessionmaker())
    await r.stage_start(job_id, "designer", 2)
    await r.stage_fail(job_id, "designer", RuntimeError("boom"))
    st = await _stage(job_id, "designer")
    assert st.status == "failed"
    assert "RuntimeError" in st.message and "boom" in st.message


async def test_finish_does_not_overwrite_degraded():
    """image_gen marks itself degraded; the wrapper's stage_finish must no-op."""
    job_id = await _new_job()
    r = ProgressReporter(get_sessionmaker())
    await r.stage_start(job_id, "image_gen", 3)
    token = bind_stage(job_id, "image_gen")
    try:
        await r.mark_degraded("3 frames filled")
    finally:
        reset_stage(token)
    await r.stage_finish(job_id, "image_gen")  # flip-iff-running -> no-op
    st = await _stage(job_id, "image_gen")
    assert st.status == "degraded" and st.message == "3 frames filled"


async def test_progress_updates_counts():
    job_id = await _new_job()
    r = ProgressReporter(get_sessionmaker())
    await r.stage_start(job_id, "image_gen", 3)
    token = bind_stage(job_id, "image_gen")
    try:
        await r.progress(2, 5, "halfway")
    finally:
        reset_stage(token)
    st = await _stage(job_id, "image_gen")
    assert st.progress_current == 2 and st.progress_total == 5 and st.message == "halfway"


async def test_resume_reentry_resets_to_running():
    job_id = await _new_job()
    r = ProgressReporter(get_sessionmaker())
    await r.stage_start(job_id, "script_writer", 1)
    await r.stage_finish(job_id, "script_writer")
    await r.stage_start(job_id, "script_writer", 1)  # resume re-run
    st = await _stage(job_id, "script_writer")
    assert st.status == "running" and st.ended_at is None


async def test_progress_and_log_noop_without_bound_stage():
    """Called outside a stage context (e.g. a pure unit test) -> no DB, no crash."""
    r = ProgressReporter(get_sessionmaker())
    await r.progress(1, 2)        # no contextvar -> returns immediately
    await r.mark_degraded("x")    # ditto
    r.log("orphan")               # ditto (sync)
