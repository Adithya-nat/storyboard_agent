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
from app.core.config import get_settings
from app.observability.cost import SpendEvent, gemini_usage_cost
from app.observability.spend_rate import get_governor

# Categories whose cost is reserved up front with the spend-rate governor (not re-recorded).
RESERVED_CATEGORIES = {"video", "image"}


@dataclass
class UsageCollector:
    """Running token totals for one job, plus a per-stage breakdown."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    by_stage: dict[str, int] = field(default_factory=dict)
    # AI spend (2026-11): one SpendEvent per priced provider call, plus running totals.
    cost_usd: float = 0.0
    cost_by_stage: dict[str, float] = field(default_factory=dict)
    events: list[SpendEvent] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add_spend(self, event: SpendEvent) -> None:
        with self._lock:  # concurrent executor threads (image_gen, narration)
            self.events.append(event)
            self.cost_usd += event.cost_usd
            if event.stage:
                self.cost_by_stage[event.stage] = (
                    self.cost_by_stage.get(event.stage, 0.0) + event.cost_usd
                )

    def drain_events(self) -> list[SpendEvent]:
        """Pop every event recorded so far (the runner persists them incrementally)."""
        with self._lock:
            out, self.events = self.events, []
            return out

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


def record_spend(
    *,
    category: str,
    model: str,
    cost_usd: float,
    units: dict[str, float] | None = None,
    provider: str = "google",
) -> None:
    """Record one priced call under the current stage. No-op without a bound collector."""
    if category not in RESERVED_CATEGORIES:
        get_governor(get_settings().spend_rate_limit_usd_per_10min).record(float(cost_usd or 0.0))
    collector = _current.get()
    if collector is None:
        return
    ctx = current()
    collector.add_spend(
        SpendEvent(
            category=category,
            provider=provider,
            model=model,
            cost_usd=float(cost_usd or 0.0),
            units=dict(units or {}),
            stage=ctx.stage if ctx else None,
        )
    )


def record_gemini_call(category: str, model: str, usage_metadata: object) -> None:
    """Record tokens AND priced spend for one google-genai call (text, image, TTS,
    transcription). The single entry point adapters use after a real call."""
    record_usage(usage_metadata)
    if usage_metadata is None:
        return
    cost, units = gemini_usage_cost(model, usage_metadata)
    record_spend(category=category, model=model, cost_usd=cost, units=units)
