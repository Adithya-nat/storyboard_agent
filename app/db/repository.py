"""Data-access helpers for jobs + artifacts + stages + logs. Keeps SQLAlchemy out
of agents."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Artifact, Job, JobLog, JobStage
from app.schemas import TERMINAL_STATUSES, JobStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def create_job(session: AsyncSession, prompt: str, duration: int) -> Job:
    job = Job(prompt=prompt, duration=duration, status=JobStatus.QUEUED.value)
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def get_job(session: AsyncSession, job_id: str) -> Job | None:
    result = await session.execute(
        select(Job).options(selectinload(Job.artifacts)).where(Job.id == job_id)
    )
    return result.scalar_one_or_none()


async def set_status(
    session: AsyncSession,
    job_id: str,
    status: JobStatus,
    *,
    error: str | None = None,
    warnings: list[str] | None = None,
) -> None:
    values: dict = {"status": status.value}
    if error is not None:
        values["error"] = error
    if warnings is not None:
        values["warnings"] = warnings
    await session.execute(update(Job).where(Job.id == job_id).values(**values))
    await session.commit()


async def set_job_tokens(
    session: AsyncSession,
    job_id: str,
    *,
    total: int,
    prompt: int = 0,
    output: int = 0,
) -> None:
    """Persist whole-job token totals (Workstream 4)."""
    await session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(total_tokens=total, prompt_tokens=prompt, output_tokens=output)
    )
    await session.commit()


async def set_stage_tokens(
    session: AsyncSession, job_id: str, by_stage: dict[str, int]
) -> None:
    """Persist per-stage token totals — one UPDATE per named stage (≤5 rows/job)."""
    for name, tokens in by_stage.items():
        await session.execute(
            update(JobStage)
            .where(JobStage.job_id == job_id, JobStage.name == name)
            .values(tokens=tokens)
        )
    await session.commit()


async def add_artifact(
    session: AsyncSession,
    job_id: str,
    *,
    kind: str,
    path: str,
    status: str = "ok",
    order_index: int | None = None,
) -> Artifact:
    artifact = Artifact(
        job_id=job_id, kind=kind, path=path, status=status, order_index=order_index
    )
    session.add(artifact)
    await session.commit()
    await session.refresh(artifact)
    return artifact


async def replace_artifacts(
    session: AsyncSession, job_id: str, artifacts: list[dict]
) -> None:
    """Atomically replace a job's artifact rows (clear-then-insert).

    Makes (re)persistence idempotent: a *resumed* job re-persists the SAME
    artifacts, and replace-not-append stops them duplicating in the table.
    Each item: ``{"kind", "path", "status"?, "order_index"?}``.
    """
    await session.execute(delete(Artifact).where(Artifact.job_id == job_id))
    for a in artifacts:
        session.add(
            Artifact(
                job_id=job_id,
                kind=a["kind"],
                path=a["path"],
                status=a.get("status", "ok"),
                order_index=a.get("order_index"),
            )
        )
    await session.commit()


async def try_acquire_job(session: AsyncSession, job_id: str) -> bool:
    """Atomically flip a job to RUNNING iff it is not already running.

    The concurrency guard for resume: an UPDATE ... WHERE status != 'running'
    that affects 0 rows means another runner already owns the job, so the caller
    must abort. Prevents two concurrent POST /resume runners racing one job's
    artifacts. Clears any stale error from a prior interrupted/failed attempt.
    """
    result = await session.execute(
        update(Job)
        .where(Job.id == job_id, Job.status != JobStatus.RUNNING.value)
        .values(status=JobStatus.RUNNING.value, error=None)
    )
    await session.commit()
    return (result.rowcount or 0) > 0


async def reap_orphaned_jobs(session: AsyncSession) -> int:
    """Mark any job stuck in queued/running as ``interrupted``.

    Called at startup. The in-process runner means a process restart (very
    common under ``uvicorn --reload``) would otherwise leave a job 'running'
    forever and clients polling into the void — a silent failure. This converts
    that into an explicit, visible terminal state.
    """
    live = [JobStatus.QUEUED.value, JobStatus.RUNNING.value]
    result = await session.execute(
        update(Job)
        .where(Job.status.in_(live))
        .values(
            status=JobStatus.INTERRUPTED.value,
            error="Job interrupted: service restarted before completion.",
        )
    )
    # Reconcile dangling stage rows too, so a restart never leaves the front end
    # staring at a phantom "running" stage for an interrupted job.
    await session.execute(
        update(JobStage)
        .where(JobStage.status.in_(["pending", "running"]))
        .values(status="interrupted", ended_at=_utcnow())
    )
    await session.commit()
    return result.rowcount or 0


async def delete_expired_jobs(session: AsyncSession, ttl_seconds: int) -> list[str]:
    """Delete jobs older than ``ttl_seconds`` + all their child rows; return the
    deleted ids so the caller can remove their on-disk artifact directories.

    Children are deleted EXPLICITLY (not via ON DELETE CASCADE) so this behaves
    identically on SQLite (FKs off by default) and Postgres. The candidate set is
    tiny for a single-user service, so we filter in Python to dodge cross-DB tz
    comparison quirks on ``created_at``."""
    now = datetime.now(timezone.utc)
    rows = (await session.execute(select(Job.id, Job.created_at))).all()
    expired: list[str] = []
    for job_id, created in rows:
        if created.tzinfo is None:  # SQLite hands back naive datetimes
            created = created.replace(tzinfo=timezone.utc)
        if now - created >= timedelta(seconds=ttl_seconds):
            expired.append(job_id)
    if not expired:
        return []
    await session.execute(delete(JobLog).where(JobLog.job_id.in_(expired)))
    await session.execute(delete(JobStage).where(JobStage.job_id.in_(expired)))
    await session.execute(delete(Artifact).where(Artifact.job_id.in_(expired)))
    await session.execute(delete(Job).where(Job.id.in_(expired)))
    await session.commit()
    return expired


# --------------------------------------------------------------------------- #
# Stages (job_stages): the live "which stage is running" summary.
# Written by ProgressReporter with DIRECT awaited calls (reliable, low volume).
# --------------------------------------------------------------------------- #
async def upsert_stage_running(
    session: AsyncSession, job_id: str, name: str, seq: int
) -> None:
    """Mark a stage RUNNING — insert on first entry, reset on a resume re-run."""
    existing = await session.execute(
        select(JobStage).where(JobStage.job_id == job_id, JobStage.name == name)
    )
    stage = existing.scalar_one_or_none()
    if stage is None:
        session.add(
            JobStage(job_id=job_id, name=name, seq=seq, status="running", started_at=_utcnow())
        )
    else:  # resume: the idempotent node runs again -> show it re-processing
        stage.status = "running"
        stage.seq = seq
        stage.started_at = _utcnow()
        stage.ended_at = None
        stage.message = None
    await session.commit()


async def flip_stage_if_running(
    session: AsyncSession, job_id: str, name: str, new_status: str
) -> None:
    """Conditional flip running -> new_status. No-ops if the node already set a
    terminal status (degraded/failed) — same guard idiom as ``try_acquire_job``."""
    await session.execute(
        update(JobStage)
        .where(
            JobStage.job_id == job_id,
            JobStage.name == name,
            JobStage.status == "running",
        )
        .values(status=new_status, ended_at=_utcnow())
    )
    await session.commit()


async def end_stage(
    session: AsyncSession,
    job_id: str,
    name: str,
    status: str,
    *,
    message: str | None = None,
) -> None:
    """Unconditionally end a stage (used for failed + degraded)."""
    values: dict = {"status": status, "ended_at": _utcnow()}
    if message is not None:
        values["message"] = message
    await session.execute(
        update(JobStage)
        .where(JobStage.job_id == job_id, JobStage.name == name)
        .values(**values)
    )
    await session.commit()


async def update_stage_progress(
    session: AsyncSession,
    job_id: str,
    name: str,
    current: int,
    total: int,
    message: str | None = None,
) -> None:
    """Advance a stage's progress. MONOTONIC: image_gen reports progress from
    concurrent per-frame tasks, so writes can arrive out of order; the
    ``progress_current < current`` guard makes a stale (lower) write a no-op, so
    the final value lands on ``total`` instead of whichever update committed last."""
    values: dict = {"progress_current": current, "progress_total": total}
    if message is not None:
        values["message"] = message
    await session.execute(
        update(JobStage)
        .where(
            JobStage.job_id == job_id,
            JobStage.name == name,
            (JobStage.progress_current.is_(None)) | (JobStage.progress_current < current),
        )
        .values(**values)
    )
    await session.commit()


async def get_stages(session: AsyncSession, job_id: str) -> list[JobStage]:
    """All stages for a job, in pipeline order (shared by GET /{id} + SSE)."""
    result = await session.execute(
        select(JobStage).where(JobStage.job_id == job_id).order_by(JobStage.seq)
    )
    return list(result.scalars().all())


# --------------------------------------------------------------------------- #
# Logs (job_logs): append-only, integer-cursor stream.
# Written via the bounded queue + LogDrainer (best-effort, high volume).
# --------------------------------------------------------------------------- #
async def insert_logs(session: AsyncSession, items) -> None:
    """Bulk-insert log lines. ``items`` are LogItem-shaped (``job_id``, ``stage``,
    ``level``, ``message``, ``created_at``) — duck-typed so this layer stays free
    of any ``app.observability`` import (avoids a layering inversion)."""
    session.add_all(
        JobLog(
            job_id=i.job_id,
            stage=i.stage,
            level=i.level,
            message=i.message,
            created_at=i.created_at,
        )
        for i in items
    )
    await session.commit()


async def get_logs(
    session: AsyncSession,
    job_id: str,
    *,
    after_id: int = 0,
    stage: str | None = None,
    limit: int = 500,
) -> list[JobLog]:
    """Log lines for a job after a cursor (shared by GET /{id}/logs + SSE poll).

    ``after_id`` is the streaming cursor (0 = from the start); ``stage`` filters to
    one stage's expand-view; ``limit`` bounds a single page/poll."""
    q = select(JobLog).where(JobLog.job_id == job_id, JobLog.id > after_id)
    if stage is not None:
        q = q.where(JobLog.stage == stage)
    q = q.order_by(JobLog.id).limit(limit)
    result = await session.execute(q)
    return list(result.scalars().all())


def is_terminal(status: str) -> bool:
    return status in {s.value for s in TERMINAL_STATUSES}
