"""SQLAlchemy models: jobs + artifacts.

Design note — artifacts by reference:
    The ``artifacts`` table stores METADATA and a filesystem ``path`` only.
    The actual image/video bytes live under ARTIFACT_ROOT, never in a DB row.
    This keeps the DB small and lets sub-agents/tools fetch only what they need.

    jobs (1) ──────< (N) artifacts
      id  PK                 id        PK
      status                 job_id    FK ─┘
      prompt                 kind  (script|frame|video|...)
      duration               path
      error                  status (ok|degraded)
      warnings (JSON)        order_index

    jobs (1) ──< (N) job_stages       jobs (1) ──< (N) job_logs
      id  PK       id      PK           id  PK      id   PK (INTEGER autoincrement = SSE cursor)
                   job_id  FK ─┘                    job_id FK ─┘
                   name (script_writer|…)           stage  (NULL = job-level)
                   seq, status, message             level, message, created_at
                   progress_current/total
                   started_at, ended_at
                   UNIQUE(job_id, name)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    duration: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    warnings: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    # Token accounting (Workstream 4): whole-job totals; the per-stage breakdown lives on
    # JobStage.tokens. Default 0 so pre-existing rows + mock runs (no usage) stay valid.
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    style: Mapped[str] = mapped_column(String(16), nullable=False, default="narrated")
    language: Mapped[str] = mapped_column(String(8), nullable=False, default="en")
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)  # None = local mode
    # Run controls: set while a pause is requested (draining) and cleared when the job
    # parks or the pause is withdrawn. ``breakpoints`` = stages to pause BEFORE.
    pause_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    breakpoints: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    artifacts: Mapped[list["Artifact"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="Artifact.order_index",
    )


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ok")
    order_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    job: Mapped[Job] = relationship(back_populates="artifacts")


class JobStage(Base):
    """One pipeline node's live status for a job — the "which stage is running"
    summary the front end shows collapsed.

    Mutable + small (≤4 rows/job): written by DIRECT awaited DB calls from the
    ProgressReporter, so the load-bearing "current stage" signal never depends on
    the best-effort (queued) log path. ``UNIQUE(job_id, name)`` makes the per-node
    row a stable upsert target.
    """

    __tablename__ = "job_stages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(32), nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)  # 1..N pipeline order
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress_current: Mapped[int | None] = mapped_column(Integer, nullable=True)
    progress_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)  # per-stage total
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (UniqueConstraint("job_id", "name", name="uq_job_stages_job_name"),)


class JobLog(Base):
    """One append-only log line for a job, optionally scoped to a stage — the
    detail shown when the front end expands a stage.

    ``id`` is an INTEGER autoincrement PK used as the streaming cursor (SSE
    ``Last-Event-ID`` + "give me logs after X"). INTEGER — not BigInteger — on
    PURPOSE: SQLite only auto-populates a PK as the rowid alias when the declared
    type is exactly ``INTEGER``, and the aiosqlite test path relies on that to
    fill the cursor. On Postgres this is an IDENTITY column. UUIDs were rejected
    here because they don't order, and ordering IS the cursor.
    """

    __tablename__ = "job_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    stage: Mapped[str | None] = mapped_column(String(32), nullable=True)  # None = job-level
    level: Mapped[str] = mapped_column(String(8), nullable=False, default="info")
    message: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


# Hot lookup paths.
Index("ix_artifacts_job_id", Artifact.job_id)          # all artifacts for a job, ordered
Index("ix_job_stages_job_id", JobStage.job_id)         # all stages for a job
Index("ix_job_logs_job_id_id", JobLog.job_id, JobLog.id)  # incremental cursor: job_id=? AND id>?


# --- Spend history: NOT linked to jobs, so the TTL sweeper never deletes it ---------- #
class SpendEvent(Base):
    __tablename__ = "spend_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    category: Mapped[str] = mapped_column(String(16), nullable=False)
    provider: Mapped[str] = mapped_column(String(16), nullable=False, default="google")
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    units: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class RunLedger(Base):
    """One row per run (upserted as the run progresses) for the usage dashboard."""

    __tablename__ = "run_ledger"

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")  # blanked after 24h
    style: Mapped[str] = mapped_column(String(16), nullable=False, default="narrated")
    language: Mapped[str] = mapped_column(String(8), nullable=False, default="en")
    duration: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cost_by_category: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    video_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    scenes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    omni_scenes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fallback_scenes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    judge_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    film_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    runtime_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    models: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# --- Film library: films a user chose to show on the homepage. NOT linked to jobs, so it
# outlives the 24-hour sweep; its media lives under the library folder, not the job's.
class LibraryFilm(Base):
    __tablename__ = "library_films"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # url slug
    job_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # rendering -> draft -> published; failed when media or the poster could not be made
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="rendering")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    native_title: Mapped[str | None] = mapped_column(String(160), nullable=True)
    tagline: Mapped[str | None] = mapped_column(String(160), nullable=True)
    logline: Mapped[str] = mapped_column(Text, nullable=False, default="")
    genre: Mapped[str] = mapped_column(String(48), nullable=False, default="")
    prompt: Mapped[str | None] = mapped_column(Text, nullable=True)  # None = creator kept it private
    language: Mapped[str] = mapped_column(String(8), nullable=False, default="en")
    style: Mapped[str] = mapped_column(String(16), nullable=False, default="dialogue")
    length_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    scenes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    made_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    poster_look: Mapped[str] = mapped_column(String(24), nullable=False, default="auto")
    poster_frame: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    poster_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    posters_made: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
