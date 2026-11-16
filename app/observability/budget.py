"""Monthly AI budget: month-to-date spend vs the cap, and per-run estimates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db import repository as repo
from app.observability.cost import estimate_job_cost

MIN_RUNS_FOR_HISTORY = 3


@dataclass
class BudgetStatus:
    month_spent_usd: float
    cap_usd: float
    remaining_usd: float
    warn: bool
    over: bool
    mode: str
    estimates: dict[str, dict[str, float]]  # style -> {"30": usd, "60": usd}
    videos_left: dict[str, int]             # "<style>_<duration>" -> count

    def dict(self) -> dict:
        return asdict(self)


async def history_cost_per_second(session: AsyncSession, style: str) -> float | None:
    runs = [
        r for r in await repo.runs_since(session, None)
        if r.style == style and r.status.startswith("completed") and r.duration and r.cost_usd > 0
    ][:20]
    if len(runs) < MIN_RUNS_FOR_HISTORY:
        return None
    return sum(r.cost_usd / r.duration for r in runs) / len(runs)


async def estimate(session: AsyncSession, *, style: str, duration: int, omni: bool = True) -> float:
    per_s = await history_cost_per_second(session, style)
    return estimate_job_cost(duration=duration, style=style, omni_enabled=omni, history_avg_per_second=per_s)


async def budget_status(session: AsyncSession, settings: Settings, user_id: str | None = None) -> BudgetStatus:
    spent = await repo.spend_since(session, repo.month_start(), user_id=user_id)
    cap = settings.budget_cap_usd
    remaining = max(0.0, cap - spent)
    estimates: dict[str, dict[str, float]] = {}
    left: dict[str, int] = {}
    for style in ("narrated", "dialogue", "mixed"):
        estimates[style] = {}
        for d in (30, 60):
            e = await estimate(session, style=style, duration=d)
            estimates[style][str(d)] = round(e, 2)
            left[f"{style}_{d}"] = int(remaining // e) if e > 0 else 0
    return BudgetStatus(
        month_spent_usd=round(spent, 4),
        cap_usd=cap,
        remaining_usd=round(remaining, 4),
        warn=spent >= cap * settings.budget_warn_fraction,
        over=spent >= cap,
        mode=settings.budget_mode,
        estimates=estimates,
        videos_left=left,
    )
