"""Job/artifact repository + the orphaned-job reaper (the late-caught critical gap)."""

from __future__ import annotations

from app.db import repository as repo
from app.schemas import JobStatus


async def test_create_and_status_transitions(session):
    job = await repo.create_job(session, "a prompt", 30)
    assert job.status == JobStatus.QUEUED.value

    await repo.set_status(session, job.id, JobStatus.RUNNING)
    fetched = await repo.get_job(session, job.id)
    assert fetched.status == JobStatus.RUNNING.value

    await repo.set_status(
        session, job.id, JobStatus.COMPLETED_WITH_WARNINGS, warnings=["frame 2: failed"]
    )
    fetched = await repo.get_job(session, job.id)
    assert fetched.status == JobStatus.COMPLETED_WITH_WARNINGS.value
    assert fetched.warnings == ["frame 2: failed"]


async def test_add_artifact(session):
    job = await repo.create_job(session, "p", 30)
    await repo.add_artifact(session, job.id, kind="frame", path="/x/1.png", order_index=1)
    fetched = await repo.get_job(session, job.id)
    assert len(fetched.artifacts) == 1
    assert fetched.artifacts[0].kind == "frame"


async def test_reaper_marks_running_jobs_interrupted(session):
    job = await repo.create_job(session, "p", 30)
    await repo.set_status(session, job.id, JobStatus.RUNNING)

    reaped = await repo.reap_orphaned_jobs(session)
    assert reaped == 1

    fetched = await repo.get_job(session, job.id)
    assert fetched.status == JobStatus.INTERRUPTED.value
    assert "restarted" in fetched.error


async def test_reaper_ignores_terminal_jobs(session):
    job = await repo.create_job(session, "p", 30)
    await repo.set_status(session, job.id, JobStatus.COMPLETED)
    reaped = await repo.reap_orphaned_jobs(session)
    assert reaped == 0


async def test_reaper_reconciles_dangling_stages(session):
    """A restart must not leave a stage stuck 'running' (phantom in the UI)."""
    job = await repo.create_job(session, "p", 30)
    await repo.set_status(session, job.id, JobStatus.RUNNING)
    await repo.upsert_stage_running(session, job.id, "script_writer", 1)
    await repo.flip_stage_if_running(session, job.id, "script_writer", "succeeded")
    await repo.upsert_stage_running(session, job.id, "image_gen", 3)  # left running

    await repo.reap_orphaned_jobs(session)

    stages = {st.name: st for st in await repo.get_stages(session, job.id)}
    assert stages["image_gen"].status == "interrupted"  # dangling -> interrupted
    assert stages["image_gen"].ended_at is not None
    assert stages["script_writer"].status == "succeeded"  # finished -> untouched


async def test_try_acquire_job_guards_concurrent_runs(session):
    job = await repo.create_job(session, "p", 30)
    # queued -> running: acquired.
    assert await repo.try_acquire_job(session, job.id) is True
    fetched = await repo.get_job(session, job.id)
    assert fetched.status == JobStatus.RUNNING.value
    # already running: a second runner cannot acquire it.
    assert await repo.try_acquire_job(session, job.id) is False


async def test_replace_artifacts_is_idempotent(session):
    job = await repo.create_job(session, "p", 30)
    await repo.replace_artifacts(
        session,
        job.id,
        [
            {"kind": "frame", "path": "/x/1.png", "status": "ok", "order_index": 1},
            {"kind": "script", "path": "/x/s.json"},
        ],
    )
    # Re-persist a superset -> rows are REPLACED, not appended (no duplicates).
    await repo.replace_artifacts(
        session,
        job.id,
        [
            {"kind": "frame", "path": "/x/1.png", "status": "ok", "order_index": 1},
            {"kind": "script", "path": "/x/s.json"},
            {"kind": "video", "path": "/x/v.mp4"},
        ],
    )
    fetched = await repo.get_job(session, job.id)
    assert sorted(a.kind for a in fetched.artifacts) == ["frame", "script", "video"]
