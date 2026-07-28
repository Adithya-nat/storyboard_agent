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

from app.db import repository as repo
from app.observability.context import current
from app.observability.logs import LogItem, enqueue


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

    async def record_stage_tokens(self, job_id: str, name: str) -> None:
        """Persist this stage's running token total (live, at stage end). Reads the
        bound UsageCollector; no-op if none is bound. Exception-safe: token accounting
        must never mask a node's own error when called from the wrapper's finally."""
        from app.observability.usage import get_collector

        try:
            collector = get_collector()
            if collector is None:
                return
            tokens = collector.by_stage.get(name)
            if tokens:
                async with self._sm() as s:
                    await repo.set_stage_tokens(s, job_id, {name: tokens})
        except Exception:  # noqa: BLE001 - degrade, don't die
            pass

    # --- intra-stage updates (called by nodes, via the contextvar) --------- #
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

    def log(self, message: str, level: str = "info") -> None:
        """Best-effort, non-blocking. Sync on purpose (just an enqueue) so hot
        loops don't await a DB round-trip per line."""
        ctx = current()
        if ctx is None:
            return
        enqueue(LogItem(ctx.job_id, ctx.stage, level, message, datetime.now(timezone.utc)))
