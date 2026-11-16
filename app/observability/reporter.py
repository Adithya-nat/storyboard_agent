"""ProgressReporter: the nodes' + graph-wrapper's handle to the observability tables.

    stage_start / stage_finish / stage_fail   -> job_stages  (DIRECT awaited; reliable)
    progress / mark_degraded                  -> job_stages  (DIRECT awaited)
    log(message, level)                       -> job_logs    (ENQUEUE; best-effort, shares
                                                              the exact path the stdlib bridge uses)

``progress`` / ``mark_degraded`` / ``log`` read the current (job, stage) from the
contextvar, so a node never hard-codes its own stage name. Stage writes are
DIRECT (the load-bearing "which stage" signal must not depend on the queue); log
writes are queued (high-volume, best-effort).

Owns its own sessionmaker (never a request session), mirroring how
``workers/runner.py`` keeps DB writes out of the pure agents.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.artifacts.rows import artifact_rows
from app.core.logging import get_logger
from app.db import repository as repo
from app.observability.context import current
from app.observability.logs import LogItem, enqueue

log = get_logger(__name__)


class ProgressReporter:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sm = sessionmaker

    # --- stage transitions (called by the graph wrapper) ------------------- #
    async def stage_start(self, job_id: str, name: str, seq: int) -> None:
        async with self._sm() as s:
            await repo.upsert_stage_running(s, job_id, name, seq)

    async def stage_finish(self, job_id: str, name: str) -> None:
        # Flip running -> succeeded ONLY if still running, so a node that already
        # set 'degraded' (image_gen) or 'failed' is not overwritten.
        async with self._sm() as s:
            await repo.flip_stage_if_running(s, job_id, name, "succeeded")

    async def stage_fail(self, job_id: str, name: str, exc: BaseException) -> None:
        async with self._sm() as s:
            await repo.end_stage(
                s, job_id, name, "failed", message=f"{type(exc).__name__}: {exc}"
            )

    async def stage_pause(self, job_id: str, name: str, seq: int, message: str) -> None:
        """The stage stopped for a pause (mid-stage, or before it started)."""
        async with self._sm() as s:
            await repo.pause_stage(s, job_id, name, seq, message)

    async def record_stage_tokens(self, job_id: str, name: str) -> None:
        """Persist this stage's tokens + spend live at stage end (best-effort)."""
        from app.observability.usage import get_collector

        try:
            collector = get_collector()
            if collector is None:
                return
            events = collector.drain_events()
            async with self._sm() as s:
                if collector.by_stage.get(name):
                    await repo.set_stage_tokens(s, job_id, {name: collector.by_stage[name]})
                if events:
                    await repo.add_spend_events(s, job_id, events)
                await repo.set_job_cost(s, job_id, collector.cost_usd, {name: collector.cost_by_stage.get(name, 0.0)})
        except Exception:  # noqa: BLE001 - accounting never breaks a stage
            pass

    async def publish_artifacts(self, job_id: str, state: dict) -> None:
        """Make everything the run has produced so far fetchable mid-run (the
        live generation UI shows real frames, cast and audio as they land).
        Same rows as the final persist, so it's a no-op for the end result.
        Best-effort: a failed publish never breaks the stage."""
        try:
            rows = artifact_rows(state)
            if not rows:
                return
            async with self._sm() as s:
                await repo.replace_artifacts(s, job_id, rows)
        except Exception:  # noqa: BLE001 - presentation only, never fatal
            log.warning("live artifact publish failed for job %s", job_id, exc_info=True)

    # --- intra-stage updates (called by nodes, via the contextvar) --------- #
    async def publish_frame(self, order: int, path: str) -> None:
        """Publish one finished frame immediately (best-effort)."""
        ctx = current()
        if ctx is None:
            return
        try:
            async with self._sm() as s:
                await repo.upsert_artifact(s, ctx.job_id, kind="frame", path=path, order_index=order)
        except Exception:  # noqa: BLE001 - presentation only, never fatal
            log.warning("live frame publish failed for job %s", ctx.job_id, exc_info=True)

    async def progress(self, current_n: int, total: int, message: str | None = None) -> None:
        ctx = current()
        if ctx is None:
            return
        async with self._sm() as s:
            await repo.update_stage_progress(s, ctx.job_id, ctx.stage, current_n, total, message)

    async def mark_degraded(self, message: str | None = None) -> None:
        ctx = current()
        if ctx is None:
            return
        async with self._sm() as s:
            await repo.end_stage(s, ctx.job_id, ctx.stage, "degraded", message=message)

    async def mark_skipped(self, message: str | None = None) -> None:
        """The stage doesn't apply to this job (e.g. narration in a pure-dialogue film)."""
        ctx = current()
        if ctx is None:
            return
        async with self._sm() as s:
            await repo.end_stage(s, ctx.job_id, ctx.stage, "skipped", message=message)

    def log(self, message: str, level: str = "info") -> None:
        """Best-effort, non-blocking. Sync on purpose (just an enqueue) so hot
        loops don't await a DB round-trip per line."""
        ctx = current()
        if ctx is None:
            return
        enqueue(LogItem(ctx.job_id, ctx.stage, level, message, datetime.now(timezone.utc)))
