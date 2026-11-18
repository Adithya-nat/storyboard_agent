"""Run controls: pause a running film, set breakpoints, and see what resuming costs.

    POST /storyboard/{id}/pause        -> 202 {status: pausing}  (parks within moments)
    PUT  /storyboard/{id}/breakpoints  -> 200 {breakpoints}      (stages to pause BEFORE)
    GET  /storyboard/{id}/checkpoint   -> 200 Checkpoint         (per-unit state + cost to finish)

Resume stays on POST /storyboard/{id}/resume (routes.py). Called in the moment before the
job parks, it withdraws the pause and the same runner carries on.

The checkpoint is read from disk, the same source of truth the stages use to skip
finished work. So "done" here means "won't be paid for again", by construction.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes import _load_owned_job
from app.core.auth import User, allowed_user
from app.core.config import get_settings
from app.db import repository as repo
from app.db.models import RunLedger, SpendEvent
from app.db.session import get_session
from app.graph.build import pipeline_stage_names
from app.observability.cost import token_cost, video_cost
from app.schemas import (
    BreakpointsRequest,
    Checkpoint,
    CheckpointStage,
    JobStatus,
    UnitProgress,
)
from app.workers import control

router = APIRouter(prefix="/storyboard", tags=["run controls"])

# Stages that spend per unit, and where their finished units live on disk.
UNIT_STAGES = {"image_gen": "frames", "motion": "AI clips"}
_FINISHED = {"succeeded", "degraded", "skipped"}
# Typical image output size for a 9:16 frame when there's no spend history yet.
_FRAME_IMAGE_TOKENS = 1290


@router.post("/{job_id}/pause", status_code=202)
async def pause_storyboard(
    job_id: str, session: AsyncSession = Depends(get_session), user: User = Depends(allowed_user)
) -> dict:
    """Pause now: the runner stops the current step and parks the job as ``paused``. Paid
    calls already submitted finish in the background and are kept (checkpoint ``in_flight``).
    Idempotent while pausing or paused."""
    job = await _load_owned_job(session, job_id, user)
    if job.status in (JobStatus.PAUSING.value, JobStatus.PAUSED.value):
        return {"job_id": job_id, "status": job.status}
    if not await repo.request_pause(session, job_id):
        raise HTTPException(status_code=409, detail=f"only a running film can be paused (status={job.status})")
    control.signal_pause(job_id)
    return {"job_id": job_id, "status": JobStatus.PAUSING.value}


@router.put("/{job_id}/breakpoints")
async def set_breakpoints(
    job_id: str,
    body: BreakpointsRequest,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(allowed_user),
) -> dict:
    """Replace the job's breakpoints: the run pauses right BEFORE each listed stage."""
    job = await _load_owned_job(session, job_id, user)
    if job.status in (JobStatus.COMPLETED.value, JobStatus.COMPLETED_WITH_WARNINGS.value):
        raise HTTPException(status_code=409, detail=f"film already finished (status={job.status})")
    pipeline = pipeline_stage_names(get_settings())
    unknown = [s for s in body.stages if s not in pipeline]
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown stage(s): {', '.join(unknown)}")
    started = {st.name for st in await repo.get_stages(session, job_id) if st.status != "paused"}
    bps = await repo.set_breakpoints(session, job_id, [s for s in body.stages if s not in started])
    return {"job_id": job_id, "breakpoints": bps}


@router.get("/{job_id}/checkpoint", response_model=Checkpoint)
async def get_checkpoint(
    job_id: str, session: AsyncSession = Depends(get_session), user: User = Depends(allowed_user)
) -> Checkpoint:
    job = await _load_owned_job(session, job_id, user)
    settings = get_settings()
    root = Path(settings.artifact_root) / job_id
    pipeline = pipeline_stage_names(settings)
    stages = {st.name: st for st in await repo.get_stages(session, job_id)}
    history = await _stage_history(session, user.scope)
    bps = list(job.breakpoints or [])
    # Read in-flight units after the last await, right before the (synchronous) disk scan,
    # so a unit that finishes meanwhile isn't counted as both done and pending.
    flying = control.in_flight(job_id)

    rows: list[CheckpointStage] = []
    used_history = False
    for name in pipeline:
        st = stages.get(name)
        status = st.status if st is not None else "pending"
        units = _units(root, name, flying.get(name, []), settings) if name in UNIT_STAGES else None
        if status in _FINISHED:
            remaining = 0.0
        elif units is not None and units.total:
            remaining = units.unit_cost_usd * len(units.pending)
        else:
            remaining = history.get(name, 0.0)
            used_history = used_history or name in history
        rows.append(CheckpointStage(
            name=name, status=status, cost_usd=round((st.cost_usd if st else 0.0) or 0.0, 4),
            remaining_usd=round(remaining, 4), units=units, breakpoint=name in bps,
        ))

    stopped = next((r for r in rows if r.status == "paused"), None)
    if stopped is None and job.status == JobStatus.PAUSING.value:
        stopped = next((r for r in rows if r.status == "running"), None)
    nxt = next((r for r in rows if r.status not in _FINISHED), None)
    resume_unit = None
    if nxt is not None and nxt.units is not None:
        waiting = nxt.units.pending or nxt.units.in_flight
        resume_unit = min(waiting) if waiting else None

    return Checkpoint(
        job_id=job_id,
        status=JobStatus(job.status),
        pause_requested=job.pause_requested_at is not None or job.status == JobStatus.PAUSING.value,
        paused_stage=stopped.name if stopped else None,
        paused_message=stages[stopped.name].message if stopped and stopped.name in stages else None,
        resume_stage=nxt.name if nxt else None,
        resume_unit=resume_unit,
        breakpoints=bps,
        spent_usd=round(job.cost_usd or 0.0, 4),
        remaining_usd=round(sum(r.remaining_usd for r in rows), 4),
        estimate_basis="history" if used_history else "units",
        stages=rows,
    )


# --------------------------------------------------------------------------- #
def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _units(root: Path, stage: str, flying: list[int], settings) -> UnitProgress:
    shots = sorted(int(s["order"]) for s in _read(root / "shotlist.json").get("shots", []) if "order" in s)
    if stage == "image_gen":
        done = [o for o in shots if (root / f"frame_{o:02d}.png").exists()
                and not (root / f"frame_{o:02d}.degraded").exists()]
        unit = token_cost(settings.gemini_image_model, image_tokens=_FRAME_IMAGE_TOKENS)
        pending = [o for o in shots if o not in done and o not in flying]
        return UnitProgress(noun="frames", total=len(shots), done=done, in_flight=[o for o in flying if o not in done],
                            pending=pending, unit_cost_usd=round(unit, 4))
    # motion: one clip per scene; price each pending scene at its own planned length.
    done = [o for o in shots if (root / f"motion_{o:02d}.mp4").exists()]
    seconds = {int(sc["order"]): float(sc.get("seconds") or settings.seconds_per_frame)
               for sc in _read(root / "script.json").get("scenes", []) if "order" in sc}
    model = settings.omni_model_chain[0] if settings.omni_model_chain else ""
    pending = [o for o in shots if o not in done and o not in flying]

    def clip(o: int) -> float:
        return video_cost(model, min(15.0, max(3.0, seconds.get(o, settings.seconds_per_frame))), settings.omni_resolution)

    unit = sum(clip(o) for o in pending) / len(pending) if pending else (clip(shots[0]) if shots else 0.0)
    return UnitProgress(noun="AI clips", total=len(shots), done=done, in_flight=[o for o in flying if o not in done],
                        pending=pending, unit_cost_usd=round(unit, 4))


async def _stage_history(session: AsyncSession, scope: str | None, runs: int = 20) -> dict[str, float]:
    """Average spend per stage over the caller's last finished runs (spend events outlive
    jobs). Hosted mode only ever averages the caller's own runs, never other users'."""
    query = select(RunLedger.job_id).where(RunLedger.status.like("completed%"))
    if scope is not None:
        query = query.where(RunLedger.user_id == scope)
    ids = (await session.execute(query.order_by(RunLedger.started_at.desc()).limit(runs))).scalars().all()
    if not ids:
        return {}
    events = (await session.execute(
        select(SpendEvent.job_id, SpendEvent.stage, SpendEvent.cost_usd).where(SpendEvent.job_id.in_(ids))
    )).all()
    per_job: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for jid, stage, cost in events:
        if stage:
            per_job[stage][jid] += cost or 0.0
    return {stage: sum(v.values()) / len(ids) for stage, v in per_job.items()}
