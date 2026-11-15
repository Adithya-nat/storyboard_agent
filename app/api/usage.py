"""Usage dashboard API: budget, aggregate stats, per-run history, estimates."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db import repository as repo
from app.db.session import get_session
from app.observability.budget import budget_status, estimate

router = APIRouter(tags=["usage"])

Range = Literal["7d", "30d", "month", "all"]


class RunRow(BaseModel):
    job_id: str
    prompt: str
    style: str
    language: str
    duration: int
    status: str
    cost_usd: float
    cost_by_category: dict[str, float] = Field(default_factory=dict)
    total_tokens: int
    video_seconds: float
    images: int
    scenes: int
    omni_scenes: int
    fallback_scenes: int
    judge_score: int | None
    runtime_seconds: float
    models: list[str]
    started_at: datetime | None
    finished_at: datetime | None


class UsageSummary(BaseModel):
    range: str
    runs: int
    completed_runs: int
    success_rate: float
    total_cost_usd: float
    avg_cost_per_run: float
    avg_cost_per_second: float
    avg_runtime_seconds: float
    avg_judge_score: float | None
    omni_share: float
    by_category: dict[str, float]
    by_model: dict[str, float]
    by_provider: dict[str, float]
    by_day: list[dict]
    avg_cost_by_style_duration: dict[str, float]
    by_language: dict[str, int]


def _since(r: Range) -> datetime | None:
    now = datetime.now(timezone.utc)
    return {"7d": now - timedelta(days=7), "30d": now - timedelta(days=30), "month": repo.month_start(now)}.get(r)


def _row(r) -> RunRow:
    return RunRow(**{k: getattr(r, k) for k in RunRow.model_fields})


@router.get("/budget")
async def get_budget(session: AsyncSession = Depends(get_session)) -> dict:
    return (await budget_status(session, get_settings())).dict()


@router.get("/estimate")
async def get_estimate(
    style: Literal["narrated", "dialogue", "mixed"] = "dialogue",
    duration: int = Query(30),
    session: AsyncSession = Depends(get_session),
) -> dict:
    if duration not in (30, 60):
        raise HTTPException(status_code=422, detail="duration must be 30 or 60")
    return {"style": style, "duration": duration, "estimated_usd": round(await estimate(session, style=style, duration=duration), 2)}


@router.get("/usage/runs", response_model=list[RunRow])
async def get_runs(
    limit: int = Query(50, ge=1, le=500),
    before: datetime | None = None,
    session: AsyncSession = Depends(get_session),
) -> list[RunRow]:
    return [_row(r) for r in await repo.list_runs(session, limit=limit, before=before)]


@router.get("/usage/summary", response_model=UsageSummary)
async def get_summary(range: Range = "30d", session: AsyncSession = Depends(get_session)) -> UsageSummary:
    since = _since(range)
    runs = await repo.runs_since(session, since)
    events = await repo.spend_events_since(session, since)
    finished = [r for r in runs if r.finished_at is not None]
    completed = [r for r in finished if r.status.startswith("completed")]
    total = sum(e.cost_usd for e in events)

    def group(key) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for e in events:
            out[key(e)] += e.cost_usd
        return {k: round(v, 4) for k, v in sorted(out.items(), key=lambda kv: -kv[1])}

    days: dict[str, dict] = {}
    for e in events:
        d = (e.created_at or datetime.now(timezone.utc)).date().isoformat()
        bucket = days.setdefault(d, {"date": d, "total": 0.0})
        bucket["total"] = round(bucket["total"] + e.cost_usd, 4)
        bucket[e.category] = round(bucket.get(e.category, 0.0) + e.cost_usd, 4)

    sd: dict[str, list[float]] = defaultdict(list)
    for r in completed:
        sd[f"{r.style}_{r.duration}"].append(r.cost_usd)
    lang: dict[str, int] = defaultdict(int)
    for r in runs:
        lang[r.language] += 1
    scored = [r.judge_score for r in completed if r.judge_score is not None]
    scenes = sum(r.scenes for r in completed)
    seconds = sum(r.duration for r in completed)
    return UsageSummary(
        range=range,
        runs=len(runs),
        completed_runs=len(completed),
        success_rate=round(len(completed) / len(finished), 4) if finished else 0.0,
        total_cost_usd=round(total, 4),
        avg_cost_per_run=round(sum(r.cost_usd for r in completed) / len(completed), 4) if completed else 0.0,
        avg_cost_per_second=round(sum(r.cost_usd for r in completed) / seconds, 4) if seconds else 0.0,
        avg_runtime_seconds=round(sum(r.runtime_seconds for r in completed) / len(completed), 1) if completed else 0.0,
        avg_judge_score=round(sum(scored) / len(scored), 1) if scored else None,
        omni_share=round(sum(r.omni_scenes for r in completed) / scenes, 4) if scenes else 0.0,
        by_category=group(lambda e: e.category),
        by_model=group(lambda e: e.model),
        by_provider=group(lambda e: e.provider),
        by_day=sorted(days.values(), key=lambda d: d["date"]),
        avg_cost_by_style_duration={k: round(sum(v) / len(v), 4) for k, v in sd.items()},
        by_language=dict(lang),
    )
