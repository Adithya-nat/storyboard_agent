"""Usage dashboard API: budget, aggregate stats, per-run history, estimates."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import User, allowed_user
from app.core.config import get_settings
from app.db import repository as repo
from app.db.session import get_session
from app.graph.build import pipeline_stage_names
from app.observability.budget import budget_status, estimate

router = APIRouter(tags=["usage"])

Range = Literal["7d", "30d", "month", "all"]
# "me" = the caller's own runs; "all" = every user of this deployment, as aggregates only
# (no prompts, no per-user rows; the runs list stays private to its owner).
Scope = Literal["me", "all"]


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
    film_score: int | None = None
    runtime_seconds: float
    models: list[str]
    started_at: datetime | None
    finished_at: datetime | None


class UsageSummary(BaseModel):
    range: str
    scope: str = "me"
    creators: int = 0  # distinct users with a run in range
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
async def get_budget(session: AsyncSession = Depends(get_session), user: User = Depends(allowed_user)) -> dict:
    return (await budget_status(session, get_settings(), user.scope)).dict()


@router.get("/estimate")
async def get_estimate(
    style: Literal["narrated", "dialogue", "mixed"] = "dialogue",
    duration: int = Query(30),
    session: AsyncSession = Depends(get_session),
    user: User = Depends(allowed_user),
) -> dict:
    if duration not in (30, 60):
        raise HTTPException(status_code=422, detail="duration must be 30 or 60")
    return {"style": style, "duration": duration, "estimated_usd": round(await estimate(session, style=style, duration=duration), 2)}


@router.get("/usage/runs", response_model=list[RunRow])
async def get_runs(
    limit: int = Query(50, ge=1, le=500),
    before: datetime | None = None,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(allowed_user),
) -> list[RunRow]:
    return [_row(r) for r in await repo.list_runs(session, limit=limit, before=before, user_id=user.scope)]


@router.get("/usage/summary", response_model=UsageSummary)
async def get_summary(
    range: Range = "30d",
    scope: Scope = "me",
    session: AsyncSession = Depends(get_session),
    user: User = Depends(allowed_user),
) -> UsageSummary:
    since = _since(range)
    user_id = None if scope == "all" else user.scope
    runs = await repo.runs_since(session, since, user_id)
    events = await repo.spend_events_since(session, since, user_id)
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
        scope=scope,
        creators=len({r.user_id or "local" for r in runs}),
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


# --- Insights: the studio's story so far (all time) ----------------------------------- #
# Everything here is lifetime, bucketed in the caller's own timezone. "me" carries your
# film list with job ids; "all" is the same shape with no ids and no prompts.
ACTIVITY_DAYS = 371  # 53 full weeks, the calendar grid
TIMELINE_FILMS = 60
MIN_CREATORS_FOR_RANK = 3  # below this a percentile would describe one specific person


class DayCount(BaseModel):
    date: str
    runs: int
    films: int
    cost_usd: float


class FilmPoint(BaseModel):
    n: int  # 1-based order among this view's finished films
    job_id: str | None = None  # only in your own view
    started_at: datetime | None
    film_score: int | None
    judge_score: int | None
    cost_usd: float
    duration: int
    style: str
    language: str
    ai_share: float | None  # share of scenes the AI video model filmed


class SweetSpot(BaseModel):
    key: str  # "<style>_<duration>"
    films: int
    avg_cost_usd: float
    avg_film_score: float | None
    avg_judge_score: float | None


class StageCost(BaseModel):
    stage: str
    cost_usd: float
    avg_seconds: float | None = None  # from runs still on disk (last 24h); None = no timing yet
    timed_runs: int = 0


class LibraryEntry(BaseModel):
    id: str
    title: str
    genre: str
    language: str
    length_seconds: int
    published_at: datetime | None
    poster: str  # public path on this API


class StudioStats(BaseModel):
    creators: int
    runs: int
    films: int
    screen_seconds: int
    scenes: int
    ai_scenes: int
    images: int
    video_seconds: float
    tokens: int
    spend_usd: float
    published: int
    library: list[LibraryEntry]  # your view only; empty in the project view
    genres: dict[str, int]  # published films by genre
    renders: int  # AI video renders (incl. retakes)
    rendered_seconds: float  # AI footage generated
    shooting_ratio: float | None  # rendered seconds : seconds of AI scenes kept
    songs: int
    languages: dict[str, int]
    styles: dict[str, int]
    outcomes: dict[str, int]
    first_at: datetime | None
    last_at: datetime | None
    active_days: int
    current_streak: int
    longest_streak: int
    activity: list[DayCount]
    punchcard: list[list[int]]  # [weekday 0=Mon][hour 0-23] runs started
    stages: list[StageCost]
    timeline: list[FilmPoint]
    sweet_spots: list[SweetSpot]


class Rank(BaseModel):
    """Where you sit among creators: the share of OTHER creators you're ahead of (0-1)."""

    creators: int
    films: float
    score: float | None  # by average film score
    value: float | None  # by cost per second of finished film (cheaper is ahead)


class UsageInsights(BaseModel):
    tz: str
    stage_order: list[str]
    me: StudioStats
    all: StudioStats
    rank: Rank | None


def _zone(tz: str | None) -> ZoneInfo:
    try:
        return ZoneInfo(tz or "UTC")
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def _local(dt: datetime | None, zone: ZoneInfo) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:  # SQLite hands back naive UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(zone)


def _done(r) -> bool:
    return r.status.startswith("completed")


def _avg(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 1) if xs else None


def streaks(days: set[date], today: date) -> tuple[int, int]:
    """(current, longest) runs of consecutive active days. A streak is still alive
    through today if you made something yesterday."""
    longest = run = 0
    prev: date | None = None
    for d in sorted(days):
        run = run + 1 if prev is not None and d - prev == timedelta(days=1) else 1
        longest = max(longest, run)
        prev = d
    current = 0
    d = today if today in days else today - timedelta(days=1)
    while d in days:
        current += 1
        d -= timedelta(days=1)
    return current, longest


def _stages(spend: list, timings: list, order: list[str]) -> list[StageCost]:
    cost: dict[str, float] = defaultdict(float)
    for _u, stage, _c, usd, _units, _j in spend:
        cost[stage or "other"] += float(usd or 0.0)
    secs: dict[str, list[float]] = defaultdict(list)
    for _u, stage, s in timings:
        secs[stage].append(s)
    names = set(cost) | set(secs)
    keys = [k for k in order if k in names] + sorted(k for k in names if k not in order)
    return [
        StageCost(stage=k, cost_usd=round(cost.get(k, 0.0), 4),
                  avg_seconds=round(sum(secs[k]) / len(secs[k]), 1) if secs.get(k) else None,
                  timed_runs=len(secs.get(k, [])))
        for k in keys
        if cost.get(k, 0.0) > 0 or secs.get(k)
    ]


def _units(spend: list, category: str, key: str, jobs: set[str] | None = None) -> float:
    total = 0.0
    for _u, _s, c, _usd, units, job in spend:
        if c == category and isinstance(units, dict) and (jobs is None or job in jobs):
            try:
                total += float(units.get(key) or 0)
            except (TypeError, ValueError):
                pass
    return total


def studio_stats(runs: list, zone: ZoneInfo, *, spend: list, timings: list, library: list,
                 order: list[str], with_ids: bool, now: datetime | None = None) -> StudioStats:
    now = _local(now or datetime.now(timezone.utc), zone)
    today = now.date()
    runs = sorted(runs, key=lambda r: _local(r.started_at, zone) or now)
    films = [r for r in runs if _done(r)]

    day_set: set[date] = set()
    days: dict[str, DayCount] = {}
    punch = [[0] * 24 for _ in range(7)]
    horizon = today - timedelta(days=ACTIVITY_DAYS - 1)
    for r in runs:
        at = _local(r.started_at, zone)
        if at is None:
            continue
        day_set.add(at.date())
        punch[at.weekday()][at.hour] += 1
        if at.date() >= horizon:
            k = at.date().isoformat()
            dc = days.setdefault(k, DayCount(date=k, runs=0, films=0, cost_usd=0.0))
            dc.runs += 1
            dc.films += int(_done(r))
            dc.cost_usd = round(dc.cost_usd + r.cost_usd, 4)
    current, longest = streaks(day_set, today)

    def count(key) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for r in films:
            out[key(r)] += 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    outcomes: dict[str, int] = defaultdict(int)
    for r in runs:
        outcomes[r.status] += 1

    timeline = [
        FilmPoint(
            n=i + 1, job_id=r.job_id if with_ids else None, started_at=r.started_at,
            film_score=r.film_score, judge_score=r.judge_score, cost_usd=round(r.cost_usd, 4),
            duration=r.duration, style=r.style, language=r.language,
            ai_share=round(r.omni_scenes / r.scenes, 4) if r.scenes else None,
        )
        for i, r in enumerate(films)
    ][-TIMELINE_FILMS:]

    groups: dict[str, list] = defaultdict(list)
    for r in films:
        groups[f"{r.style}_{r.duration}"].append(r)
    spots = [
        SweetSpot(
            key=k, films=len(v), avg_cost_usd=round(sum(r.cost_usd for r in v) / len(v), 4),
            avg_film_score=_avg([r.film_score for r in v if r.film_score is not None]),
            avg_judge_score=_avg([r.judge_score for r in v if r.judge_score is not None]),
        )
        for k, v in sorted(groups.items())
    ]
    starts = [r.started_at for r in runs if r.started_at is not None]
    genres: dict[str, int] = defaultdict(int)
    for f in library:
        genres[f.genre or "Other"] += 1
    rendered = round(_units(spend, "video", "seconds"), 1)
    # Shooting ratio, finished films only: AI footage rendered (retakes included) over the
    # seconds of finished film that came from AI scenes (runtime x AI share).
    shot = [r for r in films if r.scenes and r.omni_scenes]
    kept = sum(r.duration * (r.omni_scenes / r.scenes) for r in shot)
    shot_seconds = _units(spend, "video", "seconds", {r.job_id for r in shot})
    return StudioStats(
        creators=len({r.user_id or "local" for r in runs}),
        runs=len(runs),
        films=len(films),
        screen_seconds=sum(r.duration for r in films),
        scenes=sum(r.scenes for r in films),
        ai_scenes=sum(r.omni_scenes for r in films),
        images=sum(r.images for r in runs),
        video_seconds=round(sum(r.video_seconds for r in runs), 1),
        tokens=sum(r.total_tokens for r in runs),
        spend_usd=round(sum(r.cost_usd for r in runs), 4),
        published=len(library),
        library=[
            LibraryEntry(id=f.id, title=f.title, genre=f.genre, language=f.language,
                         length_seconds=f.length_seconds, published_at=f.published_at,
                         poster=f"/library/public/{f.id}/poster.jpg?v={f.poster_version}")
            for f in library
        ] if with_ids else [],
        genres=dict(sorted(genres.items(), key=lambda kv: -kv[1])),
        renders=int(_units(spend, "video", "renders")),
        rendered_seconds=rendered,
        shooting_ratio=round(shot_seconds / kept, 2) if kept and shot_seconds else None,
        songs=int(_units(spend, "music", "songs")),
        languages=count(lambda r: r.language),
        styles=count(lambda r: r.style),
        outcomes=dict(outcomes),
        first_at=starts[0] if starts else None,
        last_at=starts[-1] if starts else None,
        active_days=len(day_set),
        current_streak=current,
        longest_streak=longest,
        activity=sorted(days.values(), key=lambda d: d.date),
        punchcard=punch,
        stages=_stages(spend, timings, order),
        timeline=timeline,
        sweet_spots=spots,
    )


def ahead_of(mine: float | None, others: list[float], *, lower_is_better: bool = False) -> float | None:
    """Share of the other creators you're strictly ahead of."""
    if mine is None or not others:
        return None
    beat = sum(1 for o in others if (mine < o if lower_is_better else mine > o))
    return round(beat / len(others), 4)


def rank_of(me: str, runs: list) -> Rank | None:
    per: dict[str, list] = defaultdict(list)
    for r in runs:
        per[r.user_id or "local"].append(r)
    if me not in per or len(per) < MIN_CREATORS_FOR_RANK:
        return None

    def films(rs):
        return [r for r in rs if _done(r)]

    def score(rs):
        return _avg([r.film_score for r in films(rs) if r.film_score is not None])

    def value(rs):
        secs = sum(r.duration for r in films(rs))
        return sum(r.cost_usd for r in films(rs)) / secs if secs else None

    others = [rs for u, rs in per.items() if u != me]
    return Rank(
        creators=len(per),
        films=ahead_of(len(films(per[me])), [len(films(rs)) for rs in others]) or 0.0,
        score=ahead_of(score(per[me]), [s for rs in others if (s := score(rs)) is not None]),
        value=ahead_of(value(per[me]), [v for rs in others if (v := value(rs)) is not None], lower_is_better=True),
    )


@router.get("/usage/insights", response_model=UsageInsights)
async def get_insights(
    tz: str = Query("UTC", max_length=64),
    session: AsyncSession = Depends(get_session),
    user: User = Depends(allowed_user),
) -> UsageInsights:
    zone = _zone(tz)
    order = pipeline_stage_names(get_settings()) + ["other"]
    everyone = await repo.runs_since(session, None, None)
    spend = await repo.spend_rows(session)
    library = await repo.library_rows(session)
    timings = await repo.stage_timings(session)

    def own(rows: list, uid=lambda x: x[0]) -> list:
        return rows if user.scope is None else [x for x in rows if uid(x) == user.scope]

    me = studio_stats(own(everyone, lambda r: r.user_id), zone, spend=own(spend), timings=own(timings),
                      library=own(library, lambda f: f.user_id), order=order, with_ids=True)
    project = studio_stats(everyone, zone, spend=spend, timings=timings, library=library,
                           order=order, with_ids=False)
    return UsageInsights(
        tz=zone.key, stage_order=order, me=me, all=project,
        rank=rank_of(user.scope, everyone) if user.scope is not None else None,
    )
