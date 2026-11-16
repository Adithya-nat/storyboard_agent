"""HTTP surface — the async job API (curl/Postman friendly).

    POST /storyboard            -> 202 {job_id, status}        (enqueue, returns fast)
    GET  /storyboard/{id}       -> 200 status + STAGES + artifacts | 404
    GET  /storyboard/{id}/logs  -> 200 paginated log lines (?stage=&after=&limit=) | 404
    GET  /storyboard/{id}/events-> 200 SSE live stream (status|stage|log|done) | 404
    GET  /storyboard/{id}/video -> 200 mp4 | 409 not-ready | 404

Generation runs in the background (minutes). A client can either POLL (GET status
+ GET logs?after=cursor) or SUBSCRIBE (GET events, SSE) for live progress.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.adapters.turnstile import verify_turnstile
from app.core.auth import User, allowed_user, media_user
from app.core.keys import looks_like_key
from app.artifacts.rows import assets_version
from app.core.images import raqm_available
from app.observability.budget import budget_status, estimate
from app.core.config import get_settings
from app.graph.build import pipeline_stage_names
from app.db import repository as repo
from app.db.session import get_session, get_sessionmaker
from app.schemas import (
    ResumeRequest,
    VideoGap,
    ArtifactRef,
    JobCreatedResponse,
    JobListItem,
    JobStatus,
    JobStatusResponse,
    LogLine,
    LogsResponse,
    StageRef,
    StoryboardContent,
    StoryboardRequest,
    StoryboardScene,
    CastMember,
    PublishKit,
    SceneLine,
    SceneShotView,
    BriefView,
    FilmReviewView,
)
from app.workers.runner import run_job

router = APIRouter(prefix="/storyboard", tags=["storyboard"])

# Per-IP rate limiter for the public create endpoint (registered on the app in
# app/main.py). Disabled in tests via settings.rate_limit_enabled so the shared
# in-memory counter can't bleed across test cases.
_settings = get_settings()
def _rate_key(request: Request) -> str:
    # Signed-in users are limited per account; local mode per IP.
    return getattr(request.state, "user_id", None) or get_remote_address(request)


limiter = Limiter(
    key_func=_rate_key,
    enabled=_settings.rate_limit_enabled and bool(_settings.rate_limit_create.strip()),
)
_CREATE_RATE_LIMIT = _settings.rate_limit_create.strip() or "100/minute"


# --------------------------------------------------------------------------- #
# ORM -> schema mappers (shared by the pull endpoints + the SSE stream).
# --------------------------------------------------------------------------- #
def _stage_ref(st) -> StageRef:
    return StageRef(
        name=st.name,
        seq=st.seq,
        status=st.status,
        message=st.message,
        progress_current=st.progress_current,
        progress_total=st.progress_total,
        tokens=st.tokens,
        cost_usd=st.cost_usd,
        started_at=st.started_at,
        ended_at=st.ended_at,
    )


def _log_line(lg) -> LogLine:
    return LogLine(
        id=lg.id, stage=lg.stage, level=lg.level, message=lg.message, created_at=lg.created_at
    )


def _current_stage(stages) -> str | None:
    """The name of the stage currently 'running', if any (else None)."""
    return next((st.name for st in stages if st.status == "running"), None)


def _expires_at(job) -> datetime:
    """When a job (and its artifacts) expire = created_at + the TTL."""
    created = job.created_at
    if created.tzinfo is None:  # SQLite can hand back a naive datetime
        created = created.replace(tzinfo=timezone.utc)
    return created + timedelta(seconds=get_settings().job_ttl_seconds)


async def _load_live_job(session: AsyncSession, job_id: str):
    """Fetch a job; 404 if unknown, 410 if past its 1-hour TTL.

    The background sweeper deletes expired rows on its own cadence; this makes
    expiry immediate + explicit to clients even before the next sweep runs."""
    job = await repo.get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if datetime.now(timezone.utc) >= _expires_at(job):
        raise HTTPException(
            status_code=410, detail="job expired (everything is deleted 24 hours after creation)"
        )
    return job


async def _load_owned_job(session: AsyncSession, job_id: str, user: User):
    """A live job the caller owns; someone else's job is a plain 404."""
    job = await _load_live_job(session, job_id)
    if user.scope is not None and job.user_id != user.scope:
        raise HTTPException(status_code=404, detail="job not found")
    return job


def _request_key(request: Request) -> str | None:
    """The tester's Gemini key from X-Gemini-Key (BYOK). Held in memory for the run only."""
    settings = get_settings()
    key = (request.headers.get("x-gemini-key") or "").strip()
    if key and not looks_like_key(key):
        raise HTTPException(status_code=400, detail="That doesn't look like a Gemini API key.")
    if not key and settings.key_mode == "byok" and not settings.use_mock_providers:
        raise HTTPException(
            status_code=400,
            detail={"code": "key_required", "message": "Add your Gemini API key to make films."},
        )
    return key or None


def _artifact_url(job_id: str, artifact_id: str) -> str:
    """Relative URL the browser uses to stream an artifact's bytes. Relative so it
    works behind any proxy/host; the frontend prefixes its API base."""
    return f"/storyboard/{job_id}/artifacts/{artifact_id}"


def _file_response(path: str) -> FileResponse:
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="artifact file missing on disk")
    media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type)


@router.post("", status_code=202, response_model=JobCreatedResponse)
@limiter.limit(_CREATE_RATE_LIMIT)
async def create_storyboard(
    request: Request,
    body: StoryboardRequest,
    background: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(allowed_user),
) -> JobCreatedResponse:
    # Guard the public, un-authed endpoint against quota-draining abuse: a per-IP
    # rate limit (the decorator) + a Cloudflare Turnstile check (both no-ops when
    # disabled for local/mock dev).
    if not await verify_turnstile(
        get_settings(), body.turnstile_token, get_remote_address(request)
    ):
        raise HTTPException(status_code=403, detail="Turnstile verification failed")
    settings = get_settings()
    api_key = _request_key(request)
    if settings.hosted:
        if await repo.count_active_jobs(session, user.scope) >= settings.max_active_jobs_per_user:
            raise HTTPException(status_code=429, detail="You already have a film rendering. Wait for it to finish.")
        if await repo.count_active_jobs(session) >= settings.max_active_jobs:
            raise HTTPException(status_code=429, detail="The studio is busy right now. Try again in a few minutes.")
    style = body.style or settings.default_style
    language = body.language or settings.default_language
    if language == "ta" and settings.require_raqm_for_tamil and not raqm_available():
        raise HTTPException(
            status_code=422,
            detail="Tamil captions need libraqm for correct text shaping (brew install libraqm). See SETUP.md.",
        )
    omni = True
    status = await budget_status(session, settings, user.scope)
    cost = await estimate(session, style=style, duration=body.duration)
    cap = settings.budget_cap_usd
    if status.month_spent_usd + cost > cap:
        cheaper = await estimate(session, style=style, duration=body.duration, omni=False)
        if (
            settings.budget_mode == "degrade"
            and not settings.require_ai_video
            and status.month_spent_usd + cheaper <= cap
        ):
            omni = False
        else:
            raise HTTPException(
                status_code=402,
                detail=(
                    f"Monthly AI budget reached: ${status.month_spent_usd:.2f} of "
                    f"${cap:.0f} spent; this video needs ~${cost:.2f}. "
                    + ("Your monthly cap resets on the 1st." if settings.hosted
                       else "Raise MONTHLY_BUDGET_USD or wait for next month.")
                ),
            )
    job = await repo.create_job(session, body.prompt, body.duration, style, language, user_id=user.scope)
    background.add_task(run_job, job.id, body.prompt, body.duration, False, style, language, omni,
                        api_key=api_key, user_id=user.scope)
    return JobCreatedResponse(job_id=job.id, status=JobStatus(job.status))


@router.get("", response_model=list[JobListItem])
async def list_storyboards(
    session: AsyncSession = Depends(get_session), user: User = Depends(allowed_user)
) -> list[JobListItem]:
    """The caller's films from the last 24 hours (newest first)."""
    since = datetime.now(timezone.utc) - timedelta(seconds=get_settings().job_ttl_seconds)
    return [
        JobListItem(
            job_id=j.id, status=JobStatus(j.status), prompt=j.prompt, duration=j.duration,
            style=j.style, language=j.language, cost_usd=round(j.cost_usd or 0.0, 4),
            created_at=j.created_at, expires_at=_expires_at(j),
        )
        for j in await repo.list_jobs(session, user.scope, since)
    ]


@router.post("/{job_id}/resume", status_code=202, response_model=JobCreatedResponse)
async def resume_storyboard(
    job_id: str,
    request: Request,
    background: BackgroundTasks,
    body: ResumeRequest | None = None,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(allowed_user),
) -> JobCreatedResponse:
    """Resume an interrupted/failed job. Idempotent nodes skip work already done.

    409 if it's already running or already finished cleanly (nothing to resume).
    NOTE: this status check is UX only — the real concurrency guard is the atomic
    ``try_acquire_job`` inside ``run_job``, so two racing resumes can't both run.
    """
    job = await _load_owned_job(session, job_id, user)
    api_key = _request_key(request)

    status = JobStatus(job.status)
    if status == JobStatus.RUNNING:
        raise HTTPException(status_code=409, detail="job is already running")
    if status in (JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_WARNINGS):
        raise HTTPException(
            status_code=409,
            detail=f"job already finished (status={job.status}); nothing to resume",
        )

    allow_stills = bool(body and body.allow_stills)
    background.add_task(run_job, job.id, job.prompt, job.duration, True, job.style, job.language,
                        allow_stills=allow_stills, api_key=api_key, user_id=job.user_id)
    return JobCreatedResponse(job_id=job.id, status=JobStatus.RUNNING)


def _video_gap(job) -> VideoGap | None:
    """For a film stopped by missing AI clips: how much is filmed, so the user can finish with stills."""
    if job.status != JobStatus.FAILED.value or "AIVideoUnavailable" not in (job.error or ""):
        return None
    root = Path(get_settings().artifact_root) / job.id
    try:
        manifest = json.loads((root / "motion_manifest.json").read_text())
    except (OSError, ValueError):
        return None
    total = len([p for p in root.glob("frame_*.png") if p.stem.count(".") == 0])
    ready = {s["order"] for s in manifest.get("scenes", []) if (root / f"motion_{s['order']:02d}.mp4").exists()}
    missing = [o for o in range(1, total + 1) if o not in ready]
    if not ready or not missing:
        return None
    reason = "blocked" if "blocked" in job.error else "unavailable"
    return VideoGap(ready=len(ready), total=total, missing=missing, reason=reason)


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_storyboard(
    job_id: str, session: AsyncSession = Depends(get_session),
    user: User = Depends(allowed_user),
) -> JobStatusResponse:
    job = await _load_owned_job(session, job_id, user)
    stages = await repo.get_stages(session, job_id)
    return JobStatusResponse(
        job_id=job.id,
        status=JobStatus(job.status),
        prompt=job.prompt,
        duration=job.duration,
        created_at=job.created_at,
        expires_at=_expires_at(job),
        error=job.error,
        warnings=job.warnings or [],
        current_stage=_current_stage(stages),
        total_tokens=job.total_tokens,
        prompt_tokens=job.prompt_tokens,
        output_tokens=job.output_tokens,
        cost_usd=round(job.cost_usd or 0.0, 4),
        style=job.style,
        language=job.language,
        pipeline=pipeline_stage_names(get_settings()),
        stages=[_stage_ref(st) for st in stages],
        video_gap=_video_gap(job),
        artifacts=[
            ArtifactRef(
                id=a.id,
                kind=a.kind,
                url=_artifact_url(job.id, a.id),
                status=a.status,
                order=a.order_index,
            )
            for a in job.artifacts
        ],
    )


@router.get("/{job_id}/logs", response_model=LogsResponse)
async def list_logs(
    job_id: str,
    after: int = 0,
    stage: str | None = None,
    limit: int = 500,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(allowed_user),
) -> LogsResponse:
    """Paginated log pull — the expand-view + a curl-friendly replay of the stream.

    ``after`` is the cursor (the last id you've seen; 0 = from the start);
    ``stage`` filters to one stage; poll by passing back ``next_after``.
    """
    await _load_owned_job(session, job_id, user)
    limit = max(1, min(limit, 2000))
    logs = await repo.get_logs(session, job_id, after_id=after, stage=stage, limit=limit)
    next_after = logs[-1].id if logs else after
    return LogsResponse(
        job_id=job_id, logs=[_log_line(lg) for lg in logs], next_after=next_after
    )


@router.get("/{job_id}/events")
async def stream_events(job_id: str, request: Request, user: User = Depends(media_user)) -> EventSourceResponse:
    """Server-Sent Events: live status + stage + log stream for one job.

    On connect: emits a status+stage snapshot, then REPLAYS logs since
    ``Last-Event-ID`` (or from the start). Then polls the DB every
    ``log_poll_interval_seconds`` for new logs (id>cursor) + changed stages,
    emitting diffs. Closes with a ``done`` event once the job is terminal AND the
    log stream has drained. ``log`` events carry ``id:`` (the cursor); ``status``/
    ``stage`` events are idempotent snapshots, safe to re-send on reconnect.

    Uses its OWN short-lived session per poll (not a request-scoped one held open
    for minutes), so it sees the runner's committed writes and survives a future
    move to a separate worker process.
    """
    settings = get_settings()
    if not settings.enable_log_streaming:
        raise HTTPException(status_code=503, detail="log streaming is disabled")

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as s:  # existence + expiry check before streaming
        await _load_owned_job(s, job_id, user)

    poll = max(0.01, settings.log_poll_interval_seconds)
    batch = settings.log_drain_batch

    async def event_gen():
        # Resume support: a reconnecting EventSource sends the last id it saw.
        hdr = request.headers.get("last-event-id")
        cursor = int(hdr) if (hdr and hdr.lstrip("-").isdigit() and int(hdr) >= 0) else 0
        last_sig: tuple | None = None
        last_assets: str | None = None
        terminal_quiet = 0

        while True:
            if await request.is_disconnected():
                break

            async with sessionmaker() as s:
                job = await repo.get_job(s, job_id)
                stages = await repo.get_stages(s, job_id)
                logs = await repo.get_logs(s, job_id, after_id=cursor, limit=batch)

            if job is None:  # deleted mid-stream -> stop cleanly
                break

            # Status + stage snapshot, only when something changed (idempotent).
            sig = (
                job.status,
                tuple(
                    (st.name, st.status, st.progress_current, st.progress_total)
                    for st in stages
                ),
            )
            if sig != last_sig:
                yield {
                    "event": "status",
                    "data": json.dumps(
                        {"status": job.status, "current_stage": _current_stage(stages)}
                    ),
                }
                yield {
                    "event": "stage",
                    "data": json.dumps([_stage_ref(st).model_dump(mode="json") for st in stages]),
                }
                last_sig = sig

            # Live assets: a fingerprint of the artifact set; clients refetch
            # GET /{id}/live when it changes. Unknown events are ignored by
            # older clients, so this is purely additive.
            assets = assets_version(job.artifacts)
            if assets != last_assets:
                yield {"event": "assets", "data": json.dumps({"version": assets})}
                last_assets = assets

            for lg in logs:
                yield {"id": str(lg.id), "event": "log", "data": _log_line(lg).model_dump_json()}
                cursor = lg.id

            # Close only after the job is terminal AND two consecutive quiet polls
            # (gives the best-effort log drainer a cycle to flush any stragglers).
            if repo.is_terminal(job.status) and not logs:
                terminal_quiet += 1
                if terminal_quiet >= 2:
                    yield {"event": "done", "data": json.dumps({"status": job.status})}
                    break
            else:
                terminal_quiet = 0

            await asyncio.sleep(poll)

    return EventSourceResponse(event_gen())


@router.get("/{job_id}/video")
async def get_video(
    job_id: str, session: AsyncSession = Depends(get_session), user: User = Depends(media_user)
) -> FileResponse:
    job = await _load_owned_job(session, job_id, user)

    video = next((a for a in job.artifacts if a.kind == "video"), None)
    if video is None:
        if repo.is_terminal(job.status):
            raise HTTPException(status_code=404, detail="no video produced for this job")
        raise HTTPException(
            status_code=409, detail=f"video not ready (status={job.status})"
        )
    if not os.path.exists(video.path):
        raise HTTPException(status_code=404, detail="video file missing on disk")

    return FileResponse(video.path, media_type="video/mp4", filename=f"{job_id}.mp4")


@router.get("/{job_id}/artifacts/{artifact_id}")
async def get_artifact(
    job_id: str, artifact_id: str, session: AsyncSession = Depends(get_session),
    user: User = Depends(media_user),
) -> FileResponse:
    """Stream one artifact's bytes (frame / style_ref / script / ...).

    The browser-safe way to fetch images: the raw filesystem path is never exposed
    to clients, only this opaque id-addressed URL."""
    job = await _load_owned_job(session, job_id, user)
    art = next((a for a in job.artifacts if a.id == artifact_id), None)
    if art is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    return _file_response(art.path)


@router.get("/{job_id}/frames/{order}")
async def get_frame(
    job_id: str, order: int, session: AsyncSession = Depends(get_session),
    user: User = Depends(media_user),
) -> FileResponse:
    """Convenience: stream the frame at a given order (1..N)."""
    job = await _load_owned_job(session, job_id, user)
    art = next(
        (a for a in job.artifacts if a.kind == "frame" and a.order_index == order),
        None,
    )
    if art is None:
        raise HTTPException(status_code=404, detail="frame not found")
    return _file_response(art.path)


@router.get("/{job_id}/storyboard", response_model=StoryboardContent)
async def get_storyboard_content(
    job_id: str, session: AsyncSession = Depends(get_session), user: User = Depends(allowed_user)
) -> StoryboardContent:
    """The assembled storyboard for the result page: the video + per-scene frames
    with the EXACT captions the assembler burned into the mp4.

    Captions come from the shotlist's per-shot narration (what image_gen wrote into
    each frame's caption), falling back to the script's scene narration. The
    title / logline / style come from the script.
    """
    job = await _load_owned_job(session, job_id, user)
    by_kind = {a.kind: a for a in job.artifacts}

    def _read_json(art) -> dict:
        if art is None or not os.path.exists(art.path):
            return {}
        try:
            return json.loads(Path(art.path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    script = _read_json(by_kind.get("script"))
    job_dir = Path(by_kind["script"].path).parent if by_kind.get("script") else None

    def _job_json(name: str) -> dict:
        p = job_dir / name if job_dir else None
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p and p.exists() else {}
        except (OSError, json.JSONDecodeError):
            return {}

    brief_raw, review_raw = _job_json("brief.json"), _job_json("film_review.json")
    scene_audio = {int(a["order"]): a for a in (_read_json(by_kind.get("scene_audio")) or []) if "order" in a}
    script_scenes = {int(sc["order"]): sc for sc in script.get("scenes", []) if "order" in sc}
    shotlist = _read_json(by_kind.get("shotlist"))
    motion_manifest = _read_json(by_kind.get("motion_manifest"))
    motion_meta_by_order = {
        int(scene["order"]): scene
        for scene in motion_manifest.get("scenes", [])
        if "order" in scene
    }

    # Canonical caption source = the shotlist narration the assembler burned in;
    # fall back to script scenes if the shotlist is missing.
    narration_by_order: dict[int, str] = {
        s["order"]: s.get("narration", "")
        for s in shotlist.get("shots", [])
        if "order" in s
    }
    if not narration_by_order:
        narration_by_order = {
            s["order"]: s.get("narration", "")
            for s in script.get("scenes", [])
            if "order" in s
        }

    frames = sorted(
        (a for a in job.artifacts if a.kind == "frame"),
        key=lambda a: a.order_index if a.order_index is not None else 0,
    )
    # Per-scene motion clips (motion engine), keyed by the same order_index as
    # frames. Absent (or engine disabled) => video_url stays None and the UI shows
    # the still frame.
    motion_by_order = {
        a.order_index: a for a in job.artifacts if a.kind == "motion"
    }
    scenes = []
    by_renderer: dict[str, int] = {}
    by_model: dict[str, int] = {}
    kind_counts = {
        "generative": 0,
        "local_generative": 0,
        "composited": 0,
        "kenburns": 0,
    }
    for f in frames:
        order = f.order_index or 0
        motion_artifact = motion_by_order.get(f.order_index)
        meta = motion_meta_by_order.get(order, {})
        kind = meta.get("motion_kind")
        renderer = meta.get("renderer")
        model = meta.get("model")
        if kind in kind_counts:
            kind_counts[kind] += 1
        if renderer:
            by_renderer[renderer] = by_renderer.get(renderer, 0) + 1
        if model:
            by_model[model] = by_model.get(model, 0) + 1
        audio = scene_audio.get(order, {})
        src = script_scenes.get(order, {})
        scenes.append(StoryboardScene(
            order=order,
            narration=audio.get("caption") or narration_by_order.get(order, ""),
            mode=src.get("mode", "narration"),
            lines=[SceneLine(**ln) for ln in src.get("lines", [])],
            shots=[
                SceneShotView(size=sh.get("size", "medium"), action=sh.get("action", ""),
                              lines=[SceneLine(**ln) for ln in sh.get("lines", [])])
                for sh in src.get("shots", [])
            ],
            location=src.get("location", ""),
            transcript=audio.get("transcript", "") or "",
            voice=audio.get("voice") or ("narrator" if "narration" in by_kind else "silent"),
            frame_url=_artifact_url(job_id, f.id),
            video_url=(
                _artifact_url(job_id, motion_artifact.id)
                if motion_artifact is not None
                else None
            ),
            status=(
                "degraded"
                if motion_artifact is not None and motion_artifact.status == "degraded"
                else f.status
            ),
            motion_kind=kind,
            motion_renderer=renderer,
            motion_model=meta.get("model"),
            motion_reason=meta.get("motion_reason"),
        ))

    style_ref = by_kind.get("style_ref")
    video = by_kind.get("video")
    cover = by_kind.get("cover")
    portraits = {a.kind.split(":", 1)[1]: a for a in job.artifacts if a.kind.startswith("cast:")}
    cast = [
        CastMember(
            name=c.get("name", ""), look=c.get("look", ""), voice_description=c.get("voice_description", ""),
            portrait_url=_artifact_url(job_id, portraits[c.get("name", "").upper()[:26]].id)
            if c.get("name", "").upper()[:26] in portraits else None,
        )
        for c in script.get("characters", [])
    ]
    return StoryboardContent(
        job_id=job_id,
        status=JobStatus(job.status),
        title=script.get("title", ""),
        logline=script.get("logline", ""),
        style=script.get("style", ""),
        story_style=job.style,
        language=job.language,
        duration=job.duration,
        cover_url=_artifact_url(job_id, cover.id) if cover else None,
        cost_usd=round(job.cost_usd or 0.0, 4),
        cast=cast,
        video_url=f"/storyboard/{job_id}/video" if video is not None else None,
        style_ref_url=_artifact_url(job_id, style_ref.id) if style_ref else None,
        total_tokens=job.total_tokens,
        narrated="narration" in by_kind,
        brief=BriefView(
            format=brief_raw.get("format", ""), hook=brief_raw.get("hook", ""), logline=brief_raw.get("logline", ""),
            setting=brief_raw.get("setting", ""), era=(brief_raw.get("speech") or {}).get("era", "modern"),
            period=(brief_raw.get("speech") or {}).get("period", ""), dialect=(brief_raw.get("speech") or {}).get("dialect", ""),
        ) if brief_raw else None,
        film_review=FilmReviewView.model_validate(review_raw) if review_raw else None,
        motion_coverage={
            "total": len(frames),
            "animated": len(motion_by_order),
            **kind_counts,
            "still": max(0, len(frames) - len(motion_by_order)),
            "by_renderer": by_renderer,
            "by_model": by_model,
        },
        scenes=scenes,
    )


@router.get("/{job_id}/publish", response_model=PublishKit)
async def get_publish_kit(
    job_id: str, session: AsyncSession = Depends(get_session), user: User = Depends(allowed_user)
) -> PublishKit:
    """Post-ready copy (title, hook, per-platform captions + hashtags)."""
    job = await _load_owned_job(session, job_id, user)
    art = next((a for a in job.artifacts if a.kind == "publish"), None)
    if art is None or not os.path.exists(art.path):
        raise HTTPException(status_code=404 if repo.is_terminal(job.status) else 409, detail="publish kit not ready")
    return PublishKit.model_validate_json(Path(art.path).read_text(encoding="utf-8"))
