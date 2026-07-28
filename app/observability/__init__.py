"""Observability: per-stage status (job_stages) + streamable logs (job_logs).

Public surface:
    StageCtx, bind_stage, reset_stage, current   — the stage contextvar
    LogItem, DBLogHandler, LogDrainer, enqueue   — the log-capture pipeline
    configure_log_queue, reset_log_queue, queue_size
    ProgressReporter                             — nodes/wrapper write here
"""

from __future__ import annotations

from app.observability.context import StageCtx, bind_stage, current, reset_stage
from app.observability.logs import (
    DBLogHandler,
    LogDrainer,
    LogItem,
    configure_log_queue,
    enqueue,
    queue_size,
    reset_log_queue,
)
from app.observability.reporter import ProgressReporter

__all__ = [
    "StageCtx",
    "bind_stage",
    "reset_stage",
    "current",
    "LogItem",
    "DBLogHandler",
    "LogDrainer",
    "enqueue",
    "configure_log_queue",
    "reset_log_queue",
    "queue_size",
    "ProgressReporter",
]
