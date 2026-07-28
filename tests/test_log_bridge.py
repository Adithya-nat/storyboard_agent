"""The append-only log pipeline: handler -> bounded queue -> drainer -> job_logs.

Covers: the stdlib bridge only captures inside a bound stage, drop-oldest on
overflow, the drainer writes rows with a monotonic integer cursor, the drainer
survives a DB error (best-effort, never dies), and a drop surfaces as a synthetic
per-job line (never silent).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.observability import DBLogHandler, LogDrainer, bind_stage, reset_stage
from app.observability.logs import (
    LogItem,
    configure_log_queue,
    enqueue,
    queue_size,
    reset_log_queue,
)


def _item(job_id: str = "j", stage: str | None = "image_gen", msg: str = "hi") -> LogItem:
    return LogItem(job_id, stage, "info", msg, datetime.now(timezone.utc))


async def _new_job() -> str:
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
    return job.id


def test_handler_captures_only_inside_a_bound_stage():
    reset_log_queue()
    handler = DBLogHandler()
    logger = logging.getLogger("app.test_bridge")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        token = bind_stage("job-1", "designer")
        try:
            logger.info("inside the stage")  # captured
        finally:
            reset_stage(token)
        logger.info("outside the stage")  # NOT captured (no context)
    finally:
        logger.removeHandler(handler)
    assert queue_size() == 1


def test_drop_oldest_on_overflow():
    configure_log_queue(maxsize=2)
    enqueue(_item(msg="a"))
    enqueue(_item(msg="b"))
    enqueue(_item(msg="c"))  # full -> evicts oldest ("a")
    assert queue_size() == 2


async def test_drainer_writes_rows_with_monotonic_cursor():
    reset_log_queue()
    job_id = await _new_job()
    enqueue(LogItem(job_id, "image_gen", "info", "frame 1", datetime.now(timezone.utc)))
    enqueue(LogItem(job_id, "image_gen", "warning", "frame 2 failed", datetime.now(timezone.utc)))

    drainer = LogDrainer(get_sessionmaker(), batch=100)
    assert await drainer.drain_once() == 2

    async with get_sessionmaker()() as s:
        logs = await repo.get_logs(s, job_id)
    assert [lg.message for lg in logs] == ["frame 1", "frame 2 failed"]
    assert logs[0].id >= 1 and logs[1].id > logs[0].id  # INTEGER autoincrement cursor


async def test_drainer_survives_db_error(monkeypatch):
    reset_log_queue()
    enqueue(_item(job_id="whatever"))

    async def _boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(repo, "insert_logs", _boom)
    drainer = LogDrainer(get_sessionmaker(), batch=100)
    assert await drainer.drain_once() == 0  # swallowed, did not raise


async def test_dropped_lines_surface_as_synthetic_row():
    reset_log_queue()
    configure_log_queue(maxsize=1)
    job_id = await _new_job()
    enqueue(LogItem(job_id, "image_gen", "info", "keep", datetime.now(timezone.utc)))
    enqueue(LogItem(job_id, "image_gen", "info", "evict-keep", datetime.now(timezone.utc)))

    drainer = LogDrainer(get_sessionmaker(), batch=100)
    await drainer.drain_once()

    async with get_sessionmaker()() as s:
        logs = await repo.get_logs(s, job_id)
    messages = [lg.message for lg in logs]
    assert any("dropped" in m for m in messages)  # truncation is visible, not silent
