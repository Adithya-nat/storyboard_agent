"""Contextvar binding the "current job + stage" for log attribution.

Set once per node by the graph wrapper (``graph/build.py``); read by the
``ProgressReporter`` convenience methods (so a node never hard-codes its own
stage name) AND by the ``DBLogHandler`` (so every stdlib ``log.*`` emitted inside
a stage is auto-tagged to that stage).

Why a contextvar works for our execution shape: ``asyncio.gather`` and
``asyncio.to_thread`` BOTH copy the current context, so a stage bound at the node
level is visible inside the parallel per-frame attempts (gather) and inside the
image adapter's threaded ``generate_image`` call (to_thread). That is exactly how
``app.adapters.gemini``'s re-roll ``log.info`` lines get attributed to image_gen.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass


@dataclass(frozen=True)
class StageCtx:
    job_id: str
    stage: str


_current: contextvars.ContextVar[StageCtx | None] = contextvars.ContextVar(
    "storyboard_stage", default=None
)


def bind_stage(job_id: str, stage: str) -> contextvars.Token:
    """Bind (job_id, stage) for the current context. Returns a Token to reset."""
    return _current.set(StageCtx(job_id=job_id, stage=stage))


def reset_stage(token: contextvars.Token) -> None:
    """Restore the previous binding. MUST run in a ``finally`` so stage
    attribution can't bleed into the next node or the post-graph runner logs."""
    _current.reset(token)


def current() -> StageCtx | None:
    return _current.get()
