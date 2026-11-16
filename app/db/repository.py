"""Data-access helpers for jobs + artifacts + stages + logs. Keeps SQLAlchemy out
of agents."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Artifact, Job, JobLog, JobStage, RunLedger, SpendEvent
from app.schemas import TERMINAL_STATUSES, JobStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def create_job(
    session: AsyncSession, prompt: str, duration: int, style: str = "narrated", language: str = "en",
    user_id: str | None = None,
) -> Job:
    job = Job(prompt=prompt, duration=duration, style=style, language=language,
              status=JobStatus.QUEUED.value, user_id=user_id)
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


async def upsert_artifact(
    session: AsyncSession,
    job_id: str,
    *,
    kind: str,
    path: str,
    status: str = "ok",
    order_index: int | None = None,
) -> None:
    """Insert one artifact, replacing any existing row with the same
    (job, kind, order). Used to publish a frame the moment it renders, so a
    retried/resumed stage never shows the same frame twice mid-run."""
    await session.execute(
        delete(Artifact).where(
            Artifact.job_id == job_id,
            Artifact.kind == kind,
            Artifact.order_index == order_index,
        )
    )
    session.add(
        Artifact(job_id=job_id, kind=kind, path=path, status=status, order_index=order_index)
    )
    await session.commit()


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
        .where(Job.id == job_id, Job.status.not_in(_ACTIVE))
        .values(status=JobStatus.RUNNING.value, error=None, pause_requested_at=None)
    )
    await session.commit()
    return (result.rowcount or 0) > 0


# --------------------------------------------------------------------------- #
# Run controls (pause / breakpoints). Every transition is one conditional UPDATE,
# so a pause, a withdrawn pause and the runner parking the job can't interleave.
# --------------------------------------------------------------------------- #
_ACTIVE = (JobStatus.RUNNING.value, JobStatus.PAUSING.value)


async def request_pause(session: AsyncSession, job_id: str) -> bool:
    """running -> pausing. False if the job isn't running (nothing to pause)."""
    result = await session.execute(
        update(Job)
        .where(Job.id == job_id, Job.status == JobStatus.RUNNING.value)
        .values(status=JobStatus.PAUSING.value, pause_requested_at=_utcnow())
    )
    await session.commit()
    return (result.rowcount or 0) > 0


async def withdraw_pause(session: AsyncSession, job_id: str) -> bool:
    """pausing -> running (the user changed their mind while units were draining)."""
    result = await session.execute(
        update(Job)
        .where(Job.id == job_id, Job.status == JobStatus.PAUSING.value)
        .values(status=JobStatus.RUNNING.value, pause_requested_at=None)
    )
    await session.commit()
    return (result.rowcount or 0) > 0


async def pause_requested(session: AsyncSession, job_id: str) -> bool:
    flag = await session.scalar(select(Job.pause_requested_at).where(Job.id == job_id))
    return flag is not None


async def park_job(session: AsyncSession, job_id: str) -> bool:
    """Park a drained job: -> paused, iff the pause is still wanted. False means the pause
    was withdrawn while draining, so the runner carries on instead of parking."""
    result = await session.execute(
        update(Job)
        .where(Job.id == job_id, Job.status.in_(_ACTIVE), Job.pause_requested_at.is_not(None))
        .values(status=JobStatus.PAUSED.value, pause_requested_at=None)
    )
    await session.commit()
    return (result.rowcount or 0) > 0


async def set_breakpoints(session: AsyncSession, job_id: str, stages: list[str]) -> list[str]:
    clean = list(dict.fromkeys(s for s in stages if s))
    await session.execute(update(Job).where(Job.id == job_id).values(breakpoints=clean or None))
    await session.commit()
    return clean


async def consume_breakpoint(session: AsyncSession, job_id: str, stage: str) -> bool:
    """If ``stage`` is a breakpoint: remove it and raise the pause flag (one transaction).
    The row is locked while it's read and rewritten, so a concurrent ``set_breakpoints``
    waits and then lands on top instead of being overwritten by this stale list."""
    bps = await session.scalar(select(Job.breakpoints).where(Job.id == job_id).with_for_update()) or []
    if stage not in bps:
        return False
    await session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(
            breakpoints=[b for b in bps if b != stage] or None,
            pause_requested_at=_utcnow(),
            status=JobStatus.PAUSING.value,
        )
    )
    await session.commit()
    return True


async def reap_orphaned_jobs(session: AsyncSession) -> int:
    """Mark any job stuck in queued/running as ``interrupted``.

    Called at startup. The in-process runner means a process restart (very
    common under ``uvicorn --reload``) would otherwise leave a job 'running'
    forever and clients polling into the void — a silent failure. This converts
    that into an explicit, visible terminal state.
    """
    # A job that was draining toward a pause keeps that intent: it parks as paused
    # (its in-flight units are lost with the process, but nothing else is).
    pausing = select(Job.id).where(Job.status == JobStatus.PAUSING.value)
    await session.execute(
        update(JobStage)
        .where(JobStage.job_id.in_(pausing), JobStage.status.in_(["pending", "running"]))
        .values(status="paused", ended_at=_utcnow(), message="Paused (service restarted while pausing)")
    )
    await session.execute(
        update(Job)
        .where(Job.status == JobStatus.PAUSING.value)
        .values(status=JobStatus.PAUSED.value, pause_requested_at=None)
    )
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
    # ...and the usage ledger, or the dashboard shows the run as "running" forever.
    await session.execute(
        update(RunLedger)
        .where(RunLedger.status.in_(live))
        .values(status=JobStatus.INTERRUPTED.value, finished_at=_utcnow())
    )
    await session.commit()
    return result.rowcount or 0


async def delete_expired_jobs(
    session: AsyncSession, ttl_seconds: int, paused_ttl_seconds: int | None = None
) -> list[str]:
    """Delete jobs older than ``ttl_seconds`` + all their child rows; return the
    deleted ids so the caller can remove their on-disk artifact directories.

    Children are deleted EXPLICITLY (not via ON DELETE CASCADE) so this behaves
    identically on SQLite (FKs off by default) and Postgres. The candidate set is
    tiny for a single-user service, so we filter in Python to dodge cross-DB tz
    comparison quirks on ``created_at``."""
    now = datetime.now(timezone.utc)
    rows = (await session.execute(select(Job.id, Job.created_at, Job.updated_at, Job.status))).all()
    expired: list[str] = []
    for job_id, created, updated, status in rows:
        if status in _ACTIVE:  # never delete a film mid-run (or mid-pause)
            continue
        if status == JobStatus.PAUSED.value and paused_ttl_seconds is not None:
            # A paused film is kept from when it was paused, so a long pause is safe.
            created, ttl = (updated or created), paused_ttl_seconds
        else:
            ttl = ttl_seconds
        if created.tzinfo is None:  # SQLite hands back naive datetimes
            created = created.replace(tzinfo=timezone.utc)
        if now - created >= timedelta(seconds=ttl):
            expired.append(job_id)
    if not expired:
        return []
    return await _purge_expired(session, expired)


async def _purge_expired(session: AsyncSession, candidates: list[str]) -> list[str]:
    """Delete expired candidates, re-checked under a row lock.

    A resume may have started a candidate after the scan above read its status. Locking
    the rows and re-checking means either the runner acquired first (the row is now
    running, so it is skipped) or the sweeper did (``try_acquire_job`` waits on the lock,
    then finds no row and the runner aborts). A running film is never deleted."""
    expired = list((await session.execute(
        select(Job.id).where(Job.id.in_(candidates), Job.status.not_in(_ACTIVE)).with_for_update()
    )).scalars().all())
    if not expired:
        await session.commit()
        return []
    await session.execute(delete(JobLog).where(JobLog.job_id.in_(expired)))
    await session.execute(delete(JobStage).where(JobStage.job_id.in_(expired)))
    await session.execute(delete(Artifact).where(Artifact.job_id.in_(expired)))
    await session.execute(delete(Job).where(Job.id.in_(expired)))
    # 24h rule: the prompt is content too; only cost/count metrics outlive the job.
    await session.execute(update(RunLedger).where(RunLedger.job_id.in_(expired)).values(prompt=""))
    await session.commit()
    return expired


async def next_expiry(
    session: AsyncSession, ttl_seconds: int, paused_ttl_seconds: int | None = None
) -> datetime | None:
    """The next moment ``delete_expired_jobs`` could delete something (same rules); None = nothing to wait for."""
    deadlines = []
    for created, updated, status in (await session.execute(select(Job.created_at, Job.updated_at, Job.status))).all():
        if status in _ACTIVE:  # a running film is deleted on a later sweep, once it stops
            continue
        start, ttl = created, ttl_seconds
        if status == JobStatus.PAUSED.value and paused_ttl_seconds is not None:
            start, ttl = (updated or created), paused_ttl_seconds
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        deadlines.append(start + timedelta(seconds=ttl))
    return min(deadlines, default=None)


ACTIVE = (JobStatus.QUEUED.value, JobStatus.RUNNING.value, JobStatus.PAUSING.value)


async def job_ids(session: AsyncSession) -> set[str]:
    return set((await session.execute(select(Job.id))).scalars().all())


async def mark_queued(session: AsyncSession, job_id: str) -> bool:
    """Admit a stopped job for a (re)run; False if another request already admitted or started it."""
    result = await session.execute(
        update(Job).where(Job.id == job_id, Job.status.not_in(ACTIVE)).values(status=JobStatus.QUEUED.value)
    )
    await session.commit()
    return result.rowcount == 1


async def count_active_jobs(session: AsyncSession, user_id: str | None = None) -> int:
    q = select(func.count()).select_from(Job).where(Job.status.in_(ACTIVE))
    if user_id is not None:
        q = q.where(Job.user_id == user_id)
    return int(await session.scalar(q) or 0)


async def list_jobs(session: AsyncSession, user_id: str | None, since: datetime, limit: int = 50) -> list[Job]:
    q = select(Job).where(Job.created_at >= since).order_by(Job.created_at.desc()).limit(limit)
    if user_id is not None:
        q = q.where(Job.user_id == user_id)
    return list((await session.execute(q)).scalars().all())


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


async def pause_stage(session: AsyncSession, job_id: str, name: str, seq: int, message: str) -> None:
    """Mark a stage paused, creating its row when the pause landed before it started."""
    stage = (
        await session.execute(select(JobStage).where(JobStage.job_id == job_id, JobStage.name == name))
    ).scalar_one_or_none()
    if stage is None:
        session.add(JobStage(job_id=job_id, name=name, seq=seq, status="paused", message=message,
                             started_at=_utcnow(), ended_at=_utcnow()))
    else:
        stage.status, stage.message, stage.ended_at = "paused", message, _utcnow()
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


# --------------------------------------------------------------------------- #
# Spend + run ledger (persistent; never swept)
# --------------------------------------------------------------------------- #
async def add_spend_events(session: AsyncSession, job_id: str, events) -> None:
    user_id = await session.scalar(select(Job.user_id).where(Job.id == job_id))
    for e in events:
        session.add(SpendEvent(
            job_id=job_id, user_id=user_id, stage=e.stage, category=e.category, provider=e.provider,
            model=e.model, cost_usd=e.cost_usd, units=e.units, created_at=e.at,
        ))
    await session.commit()


async def set_job_cost(session: AsyncSession, job_id: str, total: float = 0.0, by_stage: dict[str, float] | None = None) -> None:
    """Job + stage cost = sum of persisted spend events, so resumed runs accumulate correctly."""
    del total
    job_total = await session.scalar(select(func.sum(SpendEvent.cost_usd)).where(SpendEvent.job_id == job_id))
    await session.execute(update(Job).where(Job.id == job_id).values(cost_usd=round(float(job_total or 0.0), 6)))
    for name in by_stage or {}:
        stage_total = await session.scalar(
            select(func.sum(SpendEvent.cost_usd)).where(SpendEvent.job_id == job_id, SpendEvent.stage == name)
        )
        await session.execute(
            update(JobStage).where(JobStage.job_id == job_id, JobStage.name == name).values(cost_usd=round(float(stage_total or 0.0), 6))
        )
    await session.commit()


async def job_spend_by_category(session: AsyncSession, job_id: str) -> dict[str, float]:
    rows = await session.execute(
        select(SpendEvent.category, func.sum(SpendEvent.cost_usd)).where(SpendEvent.job_id == job_id).group_by(SpendEvent.category)
    )
    return {c: round(v or 0.0, 6) for c, v in rows.all()}


async def upsert_run(session: AsyncSession, job_id: str, **values) -> None:
    row = await session.get(RunLedger, job_id)
    if row is None:
        session.add(RunLedger(job_id=job_id, **values))
    else:
        for k, v in values.items():
            setattr(row, k, v)
    await session.commit()


def month_start(now: datetime | None = None) -> datetime:
    now = now or _utcnow()
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


async def spend_since(session: AsyncSession, since: datetime, user_id: str | None = None) -> float:
    q = select(func.sum(SpendEvent.cost_usd)).where(SpendEvent.created_at >= since)
    if user_id is not None:
        q = q.where(SpendEvent.user_id == user_id)
    return float(await session.scalar(q) or 0.0)


async def runs_since(session: AsyncSession, since: datetime | None, user_id: str | None = None) -> list[RunLedger]:
    q = select(RunLedger).order_by(RunLedger.started_at.desc())
    if since is not None:
        q = q.where(RunLedger.started_at >= since)
    if user_id is not None:
        q = q.where(RunLedger.user_id == user_id)
    return list((await session.execute(q)).scalars().all())


async def spend_events_since(session: AsyncSession, since: datetime | None, user_id: str | None = None) -> list[SpendEvent]:
    q = select(SpendEvent)
    if since is not None:
        q = q.where(SpendEvent.created_at >= since)
    if user_id is not None:
        q = q.where(SpendEvent.user_id == user_id)
    return list((await session.execute(q)).scalars().all())


async def spend_events_for_job(session: AsyncSession, job_id: str) -> list[SpendEvent]:
    return list((await session.execute(select(SpendEvent).where(SpendEvent.job_id == job_id))).scalars().all())


async def list_runs(
    session: AsyncSession, limit: int = 50, before: datetime | None = None, user_id: str | None = None
) -> list[RunLedger]:
    q = select(RunLedger).order_by(RunLedger.started_at.desc()).limit(limit)
    if before is not None:
        q = q.where(RunLedger.started_at < before)
    if user_id is not None:
        q = q.where(RunLedger.user_id == user_id)
    return list((await session.execute(q)).scalars().all())
