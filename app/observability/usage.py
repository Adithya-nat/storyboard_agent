"""Per-job token accounting, attributed per stage.

Mirrors ``observability/context.py``: a contextvar holds a mutable ``UsageCollector`` for
the current job; the text/image/TTS adapters record their ``usage_metadata`` into it from
inside their ``asyncio.to_thread`` call. ``asyncio.gather`` + ``asyncio.to_thread`` BOTH
copy the context (see context.py), so the SAME collector object is visible across the
parallel per-frame image calls — and a ``threading.Lock`` makes the concurrent ``+=`` from
those executor threads safe.

Stage attribution is free: ``record_usage`` reads the current (job, stage) from
``observability.context`` (which the graph wrapper already binds per node), so every call
lands under the stage that made it.

Decoupled by design: if no collector is bound (unit tests, the eval harness), recording is
a no-op, so adapters never depend on a job being in flight.
"""

from __future__ import annotations

import contextvars
import threading
from dataclasses import dataclass, field

from app.observability.context import current


@dataclass
class UsageCollector:
    """Running token totals for one job, plus a per-stage breakdown."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    by_stage: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, *, prompt: int = 0, output: int = 0, total: int | None = None, stage: str | None = None) -> None:
        amount = total if total is not None else (prompt + output)
        with self._lock:  # image_gen records from concurrent executor threads
            self.total_tokens += amount
            self.prompt_tokens += prompt
            self.output_tokens += output
            if stage:
                self.by_stage[stage] = self.by_stage.get(stage, 0) + amount


_current: contextvars.ContextVar[UsageCollector | None] = contextvars.ContextVar(
    "storyboard_usage", default=None
)


def bind_collector(collector: UsageCollector) -> contextvars.Token:
    """Bind a collector for the current context (call BEFORE graph.ainvoke so every
    copied context — gather/to_thread — shares it). Returns a Token to reset."""
    return _current.set(collector)


def reset_collector(token: contextvars.Token) -> None:
    _current.reset(token)


def get_collector() -> UsageCollector | None:
    return _current.get()


def _safe_int(v: object) -> int:
    try:
        return int(v) if v is not None else 0  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def record(*, prompt: int = 0, output: int = 0, total: int | None = None) -> None:
    """Add a usage record to the bound collector under the current stage. No-op when no
    collector is bound."""
    collector = _current.get()
    if collector is None:
        return
    ctx = current()
    collector.add(prompt=prompt, output=output, total=total, stage=ctx.stage if ctx else None)


def record_usage(usage_metadata: object) -> None:
    """Record a google-genai ``usage_metadata`` defensively (every field is Optional and
    can be absent across SDK/model variants — read with getattr like the rest of the
    adapter does). ``None`` (mock responses, edge cases) is a no-op."""
    if usage_metadata is None:
        return
    prompt = _safe_int(getattr(usage_metadata, "prompt_token_count", None))
    output = _safe_int(getattr(usage_metadata, "candidates_token_count", None))
    total_raw = getattr(usage_metadata, "total_token_count", None)
    record(prompt=prompt, output=output, total=_safe_int(total_raw) if total_raw is not None else None)
