"""Append-only log pipeline: capture -> bounded queue -> async drainer -> job_logs.

    log.* (inside a bound stage)        reporter.log(...)
            │                                  │
            ▼  DBLogHandler.emit()             ▼
            └─────────────► enqueue(LogItem) ◄─┘   (one shared, thread-safe path)
                                   │  queue.Queue (bounded, drop-OLDEST on full)
                                   ▼
                          LogDrainer (one async task, started in lifespan)
                                   │  batch INSERT
                                   ▼
                               job_logs

Best-effort by design: a log line may be delayed or (under extreme flood)
dropped — but a drop is surfaced as a synthetic per-job line (never silent), and
the drainer survives DB errors so it can't die and strand the queue. The
load-bearing "which stage is running" signal does NOT flow through here;
``ProgressReporter`` writes ``job_stages`` with direct awaited calls.

Thread-safety: ``DBLogHandler.emit`` may run on a worker thread (``to_thread``),
so the queue is a ``queue.Queue`` (thread-safe) and ``emit`` only ever enqueues —
it never touches the event loop or the DB.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

from app.observability.context import current


@dataclass
class LogItem:
    job_id: str
    stage: str | None
    level: str
    message: str
    created_at: datetime


# --- the single global queue + drop bookkeeping ----------------------------- #
# Recreated by configure_log_queue() at startup so its bound tracks settings.
_queue: "queue.Queue[LogItem]" = queue.Queue(maxsize=10_000)
# Dropped lines are tracked PER JOB so the synthetic "truncated" notice can be
# attributed to the exact job that lost lines (drop-oldest knows the victim).
_dropped_by_job: dict[str, int] = {}


def configure_log_queue(maxsize: int) -> None:
    global _queue, _dropped_by_job
    _queue = queue.Queue(maxsize=maxsize)
    _dropped_by_job = {}


def enqueue(item: LogItem) -> None:
    """Non-blocking, thread-safe. On overflow, evict the OLDEST and record the
    loss against its job so it surfaces later as a synthetic line."""
    try:
        _queue.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        evicted = _queue.get_nowait()
        _dropped_by_job[evicted.job_id] = _dropped_by_job.get(evicted.job_id, 0) + 1
    except queue.Empty:
        pass
    try:
        _queue.put_nowait(item)
    except queue.Full:  # still full after evict (racing producers) -> drop the new one
        _dropped_by_job[item.job_id] = _dropped_by_job.get(item.job_id, 0) + 1


def _drain_items(limit: int) -> list[LogItem]:
    out: list[LogItem] = []
    for _ in range(limit):
        try:
            out.append(_queue.get_nowait())
        except queue.Empty:
            break
    return out


def _take_dropped() -> dict[str, int]:
    global _dropped_by_job
    snapshot, _dropped_by_job = _dropped_by_job, {}
    return snapshot


def reset_log_queue() -> None:
    """Test helper: clear the global queue + drop counters between tests so a
    leftover item (referencing a dropped job_id) can't violate an FK later."""
    while True:
        try:
            _queue.get_nowait()
        except queue.Empty:
            break
    _take_dropped()


def queue_size() -> int:
    return _queue.qsize()


class DBLogHandler(logging.Handler):
    """Routes stdlib log records emitted INSIDE a bound stage into ``job_logs``.

    ``emit`` is synchronous and may run on a worker thread, so it only enqueues
    (thread-safe, non-blocking) — never the DB, never the loop. No bound stage =>
    no-op, so unit tests that call nodes directly capture nothing.
    """

    def emit(self, record: logging.LogRecord) -> None:
        ctx = current()
        if ctx is None:
            return
        try:
            enqueue(
                LogItem(
                    job_id=ctx.job_id,
                    stage=ctx.stage,
                    level=record.levelname.lower(),
                    message=record.getMessage(),
                    created_at=datetime.now(timezone.utc),
                )
            )
        except Exception:  # logging must never break the caller
            self.handleError(record)


class LogDrainer:
    """One async task that drains the queue into ``job_logs`` in batches.

    Survives per-batch DB errors (logs to stderr, continues) so it can never die
    and strand the queue. ``drain_once`` is public so tests + the lifespan
    shutdown can force a flush without racing the interval loop.
    """

    def __init__(self, sessionmaker, *, batch: int = 200, interval: float = 0.75) -> None:
        self._sm = sessionmaker
        self._batch = batch
        self._interval = interval
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._run())

    async def astop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None
        await self.drain_once()  # flush anything enqueued during shutdown

    async def _run(self) -> None:
        while not self._stop.is_set():
            await self.drain_once()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass

    async def drain_once(self) -> int:
        # Local import avoids an import cycle at module load (repository -> models;
        # observability is a higher layer that may be imported very early).
        from app.db import repository as repo

        items = _drain_items(self._batch)
        for job_id, n in _take_dropped().items():
            items.append(
                LogItem(
                    job_id=job_id,
                    stage=None,
                    level="warning",
                    message=f"{n} log line(s) dropped (queue overflow)",
                    created_at=datetime.now(timezone.utc),
                )
            )
        if not items:
            return 0
        try:
            async with self._sm() as session:
                await repo.insert_logs(session, items)
            return len(items)
        except Exception as exc:  # noqa: BLE001 - logs are best-effort; never die
            print(f"[log-drainer] batch insert failed ({len(items)} lines): {exc}", file=sys.stderr)
            return 0
