"""Recut: redo picked parts of a paused, failed or finished film.

    GET  /storyboard/{id}/recut               -> RecutMap (scenes, film items, takes, cuts)
    GET  /storyboard/{id}/recut/media/{name}  -> a current or archived media file
    POST /storyboard/{id}/recut/plan          -> RecutPlanResponse (director edits + estimate; no media spend)
    POST /storyboard/{id}/recut               -> 202: archive, apply, re-run only what changed
    POST /storyboard/{id}/takes/{take}/restore -> 202: bring a take back, rebuild what depends on it
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.errors import ProviderError
from app.adapters.llm import LLMAdapter
from app.agents.designer import sheet_name
from app.agents.redo_director import RedoPlan, apply_plan, direct, edited, scene_history
from app.api.controls import _FRAME_IMAGE_TOKENS, _read, _stage_history
from app.api.routes import (
    _admission,
    _check_active_limits,
    _file_response,
    _load_owned_job,
    _request_key,
)
from app.core.auth import User, allowed_user, media_user
from app.core.config import get_settings
from app.core.keys import bind_key, redact, release_key
from app.db import repository as repo
from app.db.session import get_session
from app.graph.build import pipeline_stage_names
from app.observability.cost import token_cost, video_cost
from app.observability.logs import LogItem
from app.observability.usage import UsageCollector, bind_collector, reset_collector
from app.redo import takes as tk
from app.redo.targets import LABEL, PER_SCENE, Invalidation, Item, closure, media_path, parse
from app.schemas import (
    CastMember,
    CutView,
    JobStatus,
    RecutEstimate,
    RecutExtra,
    RecutFilmItem,
    RecutMap,
    RecutNote,
    RecutPlanRequest,
    RecutPlanResponse,
    RecutRef,
    RecutRequest,
    RecutScene,
    RecutShot,
    RecutStarted,
    SceneDiff,
    SceneLine,
    ShotDiff,
    TakeView,
)
from app.workers.runner import run_job

router = APIRouter(prefix="/storyboard", tags=["recut"])

_RECUTTABLE = {
    JobStatus.PAUSED.value, JobStatus.FAILED.value, JobStatus.INTERRUPTED.value,
    JobStatus.COMPLETED.value, JobStatus.COMPLETED_WITH_WARNINGS.value,
}
# Rough wall-clock per unit, for the "~3 min" hint.
_SECONDS = {"frame": 25, "clip": 90, "voice": 8, "story": 420, "look": 150, "score": 45, "release": 20, "edit": 75}
_DEFAULT_USD = {"story": 0.4, "look": 1.0, "score": 0.1, "release": 0.02, "edit": 0.05, "voice": 0.01}
_HISTORY_STAGES = {"story": ("script_writer", "dialogue_pass", "reflection"), "look": ("designer",),
                   "score": ("music",), "release": ("publish_kit",), "edit": ("film_review",)}


def _job_dir(job_id: str) -> Path:
    return Path(get_settings().artifact_root) / job_id


def _media(job_id: str, job_dir: Path, name: str) -> str | None:
    path = job_dir / name
    if not path.is_file():
        return None
    return f"/storyboard/{job_id}/recut/media/{name}?v={int(path.stat().st_mtime)}"


def _scene_orders(job_dir: Path) -> list[int]:
    return sorted(int(s["order"]) for s in _read(job_dir / "script.json").get("scenes", []) if "order" in s)


def _why_not(status: str) -> str | None:
    if status in _RECUTTABLE:
        return None
    if status in (JobStatus.RUNNING.value, JobStatus.QUEUED.value):
        return "Pause the film first, then pick what to redo."
    if status == JobStatus.PAUSING.value:
        return "The film is still pausing; recut opens once it stops."
    return f"This film can't be recut right now ({status})."


def _take_view(job_id: str, job_dir: Path, t: dict) -> TakeView:
    urls = [u for name in t.get("files", []) if (u := _media(job_id, job_dir, f"takes/{t['id']}/{name}"))]
    return TakeView(id=t["id"], cut=t["cut"], kind=t["kind"], unit=t.get("unit"), label=t["label"], at=t["at"],
                    urls=urls, restorable=t["kind"] in tk.RESTORABLE and bool(urls))


# --------------------------------------------------------------------------- #
@router.get("/{job_id}/recut", response_model=RecutMap)
async def get_recut_map(
    job_id: str, session: AsyncSession = Depends(get_session), user: User = Depends(allowed_user)
) -> RecutMap:
    job = await _load_owned_job(session, job_id, user)
    settings = get_settings()
    job_dir = _job_dir(job_id)
    pipeline = pipeline_stage_names(settings)
    script = _read(job_dir / "script.json")
    manifest = {int(c["order"]): c for c in _read(job_dir / "motion_manifest.json").get("scenes", []) if "order" in c}
    takes = tk.list_takes(job_dir)
    by_unit: dict[tuple[str, int | None], list[dict]] = {}
    for t in takes:
        by_unit.setdefault(("scene" if t.get("unit") is not None else t["kind"], t.get("unit")), []).append(t)

    scenes: list[RecutScene] = []
    for sc in script.get("scenes", []):
        n = int(sc.get("order", 0))
        nn = f"{n:02d}"
        clip = manifest.get(n)
        clip_url = _media(job_id, job_dir, f"motion_{nn}.mp4")
        hist = scene_history(job_dir, n)
        issues = list((clip or {}).get("qc") or []) if clip_url else []
        if "accepted_rewrite" in hist and clip_url:
            issues.append(f"rewritten to meet Google's rules: {hist['accepted_rewrite'].get('change') or ''}".strip())
        if clip_url:
            status = "qc" if (clip or {}).get("qc") else "ok"
        elif "blocked" in hist:
            status = "blocked"
            issues.insert(0, f"blocked by Google ({hist['blocked'].get('code')}) after "
                             f"{len(hist['blocked'].get('failed_rewrites') or [])} rewrite(s)")
        elif (job_dir / f"frame_{nn}.png").exists():
            status = "missing"
        else:
            status = "pending"
        scenes.append(RecutScene(
            order=n, mode=sc.get("mode", "narration"), beat_role=sc.get("beat_role", ""), location=sc.get("location", ""),
            planned_seconds=float(sc.get("seconds") or 0),
            seconds=round(float((clip or {}).get("source_seconds") or sc.get("seconds") or 0), 2) if clip_url
            else float(sc.get("seconds") or 0),
            description=sc.get("description", ""), narration=sc.get("narration", ""),
            shots=[RecutShot(size=s.get("size", "medium"), action=s.get("action", ""), camera=s.get("camera", ""),
                             seconds=float(s.get("seconds") or 0),
                             lines=[SceneLine(speaker=l.get("speaker", ""), text=l.get("text", ""),
                                              delivery=l.get("delivery", "")) for l in s.get("lines", [])])
                   for s in sc.get("shots", [])],
            frame_url=_media(job_id, job_dir, f"frame_{nn}.png"),
            clip_url=clip_url,
            voice_url=_media(job_id, job_dir, f"narration_{nn}.wav"),
            status=status, issues=issues,
            takes=[_take_view(job_id, job_dir, t) for t in by_unit.get(("scene", n), [])
                   if t["kind"] in tk.RESTORABLE],
        ))

    publish = _read(job_dir / "publish.json")
    music_url = _media(job_id, job_dir, "music.mp3") or _media(job_id, job_dir, "music.wav")
    film = [
        RecutFilmItem(kind="story", label="Story", ready=bool(script),
                      detail=" — ".join(x for x in (script.get("title", ""), script.get("logline", "")) if x)),
        RecutFilmItem(kind="look", label="Look", ready=(job_dir / "shotlist.json").exists(),
                      detail=(_read(job_dir / "shotlist.json").get("style_bible") or "")[:220],
                      url=_media(job_id, job_dir, "style_ref.png")),
        RecutFilmItem(kind="score", label="Score", ready=bool(music_url), url=music_url,
                      cues=list(_read(job_dir / "music.json").get("starts") or [])),
        RecutFilmItem(kind="release", label="Release", ready=bool(publish),
                      detail=publish.get("title", ""), url=_media(job_id, job_dir, "cover.png")),
    ]
    for item in film:
        item.takes = [_take_view(job_id, job_dir, t) for t in by_unit.get((item.kind, None), [])]

    kinds = []
    if script:
        kinds += ["story", "scene", "frame"]
        if "motion" in pipeline:
            kinds.append("clip")
        if settings.enable_tts and any(s.get("mode") == "narration" for s in script.get("scenes", [])):
            kinds.append("voice")
    if (job_dir / "shotlist.json").exists():
        kinds.append("look")
    if settings.music_enabled:
        kinds.append("score")
    kinds.append("release")

    cast = [CastMember(name=c.get("display_name") or c.get("name", ""), look=c.get("look", ""),
                       portrait_url=_media(job_id, job_dir, sheet_name(c.get("name", ""), 0)))
            for c in script.get("characters", [])]
    reason = _why_not(job.status) or (None if script else "Nothing to recut yet: the script isn't written.")
    return RecutMap(
        job_id=job_id, status=JobStatus(job.status), can_recut=reason is None, reason=reason,
        language=job.language, title=script.get("title", ""), pipeline=pipeline, kinds=kinds,
        scenes=scenes, film=film, cuts=_cuts(job_id, job_dir, job.cost_usd or 0.0), cast=cast,
    )


def _cuts(job_id: str, job_dir: Path, cost_now: float) -> list[CutView]:
    rows = tk.history(job_dir) or [{"cut": 1, "kind": "first", "summary": "First cut", "cost_start": 0.0}]
    out = []
    for i, c in enumerate(rows):
        current = i == len(rows) - 1
        if current:
            video = _media(job_id, job_dir, "storyboard_muxed.mp4") or _media(job_id, job_dir, "storyboard.mp4")
            seconds = tk.scene_seconds(job_dir)
            cost = cost_now - float(c.get("cost_start") or 0)
        else:
            nxt = f"takes/c{c['cut'] + 1}-edit"
            video = _media(job_id, job_dir, f"{nxt}/storyboard_muxed.mp4") or _media(job_id, job_dir, f"{nxt}/storyboard.mp4")
            seconds = c.get("scene_seconds") or {}
            cost = float(c.get("cost_end") or 0) - float(c.get("cost_start") or 0)
        out.append(CutView(cut=c["cut"], kind=c.get("kind", "recut"), at=c.get("at", ""), summary=c.get("summary", ""),
                           note=c.get("note", ""), picked=c.get("picked", []), items=c.get("items", []),
                           cost_usd=round(max(0.0, cost), 4), scene_seconds=seconds, video_url=video, current=current))
    return out


@router.get("/{job_id}/recut/media/{name:path}")
async def recut_media(
    job_id: str, name: str, session: AsyncSession = Depends(get_session), user: User = Depends(media_user)
) -> FileResponse:
    await _load_owned_job(session, job_id, user)
    path = media_path(_job_dir(job_id), name)
    if path is None:
        raise HTTPException(status_code=404, detail="file not found")
    return _file_response(str(path))


# --------------------------------------------------------------------------- #
def _picked(refs: list[RecutRef], scenes: list[int]) -> list[Item]:
    try:
        items = [parse(r.kind, r.unit, scenes) for r in refs]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return sorted(set(items))


def _expand(picked: list[Item], plan: RedoPlan | None, changed: dict[int, bool], job_dir: Path,
            pipeline: list[str]) -> list[Item]:
    """The director's scene edits become concrete media items: a new clip, a new frame only when
    the opening picture changed, a new voice for narrated scenes."""
    if plan is None:
        return picked
    modes = {int(s["order"]): s.get("mode") for s in _read(job_dir / "script.json").get("scenes", []) if "order" in s}
    allowed = {i.unit for i in picked if i.kind in PER_SCENE}
    out = {i for i in picked if i.kind != "scene"}
    edited_orders = {o for o in changed if o in allowed}
    for n in edited_orders:
        if "motion" in pipeline:
            out.add(Item("clip", n))
        if changed[n]:
            out.add(Item("frame", n))
        if modes.get(n) == "narration":
            out.add(Item("voice", n))
    out |= {i for i in picked if i.kind == "scene" and i.unit not in edited_orders}
    return sorted(out)


async def _unit_costs(session: AsyncSession, scope: str | None, job_dir: Path) -> tuple[dict, dict[int, float]]:
    settings = get_settings()
    history = await _stage_history(session, scope)
    costs = {k: sum(history.get(s, 0.0) for s in stages) or _DEFAULT_USD[k] for k, stages in _HISTORY_STAGES.items()}
    costs["frame"] = token_cost(settings.gemini_image_model, image_tokens=_FRAME_IMAGE_TOKENS)
    costs["voice"] = _DEFAULT_USD["voice"]
    model = settings.omni_model_chain[0] if settings.omni_model_chain else ""
    clips = {int(s["order"]): video_cost(model, min(15.0, max(3.0, float(s.get("seconds") or settings.seconds_per_frame))),
                                         settings.omni_resolution)
             for s in _read(job_dir / "script.json").get("scenes", []) if "order" in s}
    return costs, clips


def _item_usd(item: Item, costs: dict, clips: dict[int, float]) -> float:
    if item.kind == "clip":
        return clips.get(item.unit or 0, 0.0)
    return costs.get(item.kind, 0.0)


def _estimate(inv: Invalidation, costs: dict, clips: dict[int, float], *, directed: bool) -> RecutEstimate:
    settings = get_settings()
    usd = sum(_item_usd(i, costs, clips) for i in inv.items) + (0.02 if directed else 0.0)
    n = lambda k: len(inv.units(k))  # noqa: E731
    secs = (math.ceil(n("frame") / max(1, settings.image_max_concurrency)) * _SECONDS["frame"]
            + math.ceil(n("clip") / max(1, settings.omni_max_concurrency)) * _SECONDS["clip"]
            + n("voice") * _SECONDS["voice"]
            + sum(_SECONDS[k] for k in ("story", "look", "score", "release", "edit") if inv.has(k)))
    return RecutEstimate(usd=round(usd, 2), seconds=int(secs))


def _lines(lines) -> list[SceneLine]:
    return [SceneLine(speaker=l.speaker, text=l.text, delivery=l.delivery) for l in lines]


def _diffs(job_dir: Path, plan: RedoPlan, allowed: set[int]) -> tuple[list[SceneDiff], dict[int, bool]]:
    from app.schemas import Script, Shotlist

    raw = _read(job_dir / "script.json")
    if not raw:
        return [], {}
    before = Script.model_validate(raw)
    raw_sl = _read(job_dir / "shotlist.json")
    after, _, changed = edited(before.model_copy(deep=True), Shotlist.model_validate(raw_sl) if raw_sl else None,
                               plan, allowed)
    out = []
    for n in sorted(o for o in changed if o in allowed):
        b = next(s for s in before.scenes if s.order == n)
        a = next(s for s in after.scenes if s.order == n)
        shots = [ShotDiff(index=i, seconds=sb.seconds, size_before=sb.size, size_after=sa.size,
                          action_before=sb.action, action_after=sa.action,
                          lines_before=_lines(sb.lines), lines_after=_lines(sa.lines))
                 for i, (sb, sa) in enumerate(zip(b.shots, a.shots))]
        out.append(SceneDiff(order=n, description_before=b.description, description_after=a.description,
                             redraw_frame=changed[n], shots=shots))
    return out, {n: changed[n] for n in changed if n in allowed}


@router.post("/{job_id}/recut/plan", response_model=RecutPlanResponse)
async def plan_recut(
    job_id: str,
    body: RecutPlanRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(allowed_user),
) -> RecutPlanResponse:
    job = await _load_owned_job(session, job_id, user)
    if (why := _why_not(job.status)) is not None:
        raise HTTPException(status_code=409, detail=why)
    settings = get_settings()
    job_dir = _job_dir(job_id)
    scenes = _scene_orders(job_dir)
    if not scenes:
        raise HTTPException(status_code=409, detail="Nothing to recut yet: the script isn't written.")
    pipeline = pipeline_stage_names(settings)
    picked = _picked(body.targets, scenes)
    note = body.note.strip()
    directed = bool(note) or any(i.kind == "scene" for i in picked)

    plan: RedoPlan | None = None
    plan_id = None
    diffs: list[SceneDiff] = []
    changed: dict[int, bool] = {}
    if directed:
        api_key = _request_key(request)
        plan = await _run_director(session, job_id, job.language, job_dir, picked, note, api_key)
        diffs, changed = _diffs(job_dir, plan, {i.unit for i in picked if i.kind in PER_SCENE})
        plan_id = uuid4().hex[:12]
        tk.save_plan(job_dir, plan_id, {"picked": [{"kind": i.kind, "unit": i.unit} for i in picked], "note": note,
                                        "plan": plan.model_dump(), "at": datetime.now(timezone.utc).isoformat()})
    effective = _expand(picked, plan, changed, job_dir, pipeline)
    inv = closure(effective, scenes, pipeline)
    costs, clips = await _unit_costs(session, user.scope, job_dir)

    extra: list[RecutExtra] = []
    for e in (plan.extra if plan else []):
        try:
            it = parse(e.kind, e.unit, scenes)
        except ValueError:
            continue
        if it in inv.items or it.kind == "scene" or any(x.kind == it.kind and x.unit == it.unit for x in extra):
            continue
        extra_inv = closure(effective + [it], scenes, pipeline)
        delta = sum(_item_usd(i, costs, clips) for i in extra_inv.items if i not in inv.items)
        extra.append(RecutExtra(kind=it.kind, unit=it.unit, label=it.label, reason=e.reason, usd=round(delta, 2)))

    notes = [RecutNote(kind=k, text=getattr(plan, f"{k}_note")) for k in ("story", "look", "score", "release")
             if plan and getattr(plan, f"{k}_note").strip()]
    return RecutPlanResponse(
        plan_id=plan_id, summary=plan.summary if plan else _plain_summary(picked), directed=directed,
        scenes=diffs, notes=notes, extra=extra,
        items=[RecutRef(kind=i.kind, unit=i.unit) for i in inv.items if i.kind in LABEL and i.kind not in ("captions", "edit")],
        stages=inv.stages, estimate=_estimate(inv, costs, clips, directed=directed),
        review_frames_available=inv.has("frame") and inv.has("clip"),
    )


def _plain_summary(picked: list[Item]) -> str:
    return "A fresh take of " + ", ".join(i.label.lower() if i.unit is None else i.label for i in picked) + "."


async def _run_director(session, job_id: str, language: str, job_dir: Path, picked: list[Item], note: str,
                        api_key: str | None) -> RedoPlan:
    settings = get_settings()
    if api_key:
        settings = settings.model_copy(update={"gemini_api_key": api_key})
    key_token = bind_key(api_key)
    collector = UsageCollector()
    usage_token = bind_collector(collector)
    try:
        return await asyncio.to_thread(direct, LLMAdapter(settings), job_dir, picked, note, language)
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=f"The director couldn't plan this: {redact(str(exc))}") from None
    finally:
        reset_collector(usage_token)
        release_key(api_key, key_token)
        events = collector.drain_events()
        if events:
            await repo.add_spend_events(session, job_id, events)
            await repo.set_job_cost(session, job_id)


# --------------------------------------------------------------------------- #
async def _log(session: AsyncSession, job_id: str, stage: str | None, message: str) -> None:
    await repo.insert_logs(session, [LogItem(job_id, stage, "info", message, datetime.now(timezone.utc))])


async def _admit(session: AsyncSession, job, user: User) -> None:
    if (why := _why_not(job.status)) is not None:
        raise HTTPException(status_code=409, detail=why)
    async with _admission():
        await _check_active_limits(session, user)
        if not await repo.mark_queued(session, job.id):
            raise HTTPException(status_code=409, detail="The film is already running.")


async def _launch(background: BackgroundTasks, job, api_key: str | None) -> None:
    background.add_task(run_job, job.id, job.prompt, job.duration, True, job.style, job.language,
                        api_key=api_key, user_id=job.user_id, fresh=True)


@router.post("/{job_id}/recut", status_code=202, response_model=RecutStarted)
async def start_recut(
    job_id: str,
    body: RecutRequest,
    request: Request,
    background: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(allowed_user),
) -> RecutStarted:
    job = await _load_owned_job(session, job_id, user)
    settings = get_settings()
    job_dir = _job_dir(job_id)
    scenes = _scene_orders(job_dir)
    pipeline = pipeline_stage_names(settings)
    saved = tk.load_plan(job_dir, body.plan_id) if body.plan_id else None
    if body.plan_id and saved is None:
        raise HTTPException(status_code=404, detail="That plan has expired; plan the recut again.")
    plan = RedoPlan.model_validate(saved["plan"]) if saved else None
    refs = [RecutRef(**r) for r in saved["picked"]] if saved else body.targets
    if not refs:
        raise HTTPException(status_code=422, detail="Pick something to redo.")
    picked = _picked(refs, scenes)
    if plan is None and any(i.kind == "scene" for i in picked):
        raise HTTPException(status_code=422, detail="Rewriting a scene needs a plan from the director first.")
    accepted = _picked(body.accept_extra, scenes) if body.accept_extra else []
    api_key = _request_key(request)
    previous = job.status
    await _admit(session, job, user)
    try:
        changed = _diffs(job_dir, plan, {i.unit for i in picked if i.kind in PER_SCENE})[1] if plan else {}
        effective = sorted(set(_expand(picked, plan, changed, job_dir, pipeline)) | {i for i in accepted if i.kind != "scene"})
        inv = closure(effective, scenes, pipeline)
        note = saved.get("note", "") if saved else ""
        summary = plan.summary if plan else _plain_summary(picked)
        cut = tk.start_cut(job_dir, picked=effective, items=inv.items, note=note, summary=summary,
                           cost_now=job.cost_usd or 0.0, created_at=job.created_at.isoformat())
        tk.archive(job_dir, inv.items + [Item("scene", n) for n in changed], cut)
        if plan is not None:
            apply_plan(job_dir, plan, picked)
        await _prepare_run(session, job, inv, cut, summary, review=body.review_frames)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - give the film back in the state it was in
        await repo.set_status(session, job_id, JobStatus(previous), error=None)
        raise HTTPException(status_code=500, detail=f"Recut couldn't start: {redact(str(exc))}") from None
    await _launch(background, job, api_key)
    return RecutStarted(job_id=job_id, status=JobStatus.RUNNING, cut=cut)


async def _prepare_run(session: AsyncSession, job, inv: Invalidation, cut: int, summary: str, *, review: bool) -> None:
    await repo.reset_stages(session, job.id, inv.stages)
    bps = list(job.breakpoints or [])
    if review and inv.has("frame") and inv.has("clip") and "motion" not in bps:
        await repo.set_breakpoints(session, job.id, bps + ["motion"])
    await _log(session, job.id, inv.stages[0] if inv.stages else None, f"Cut {cut}: {summary}")


@router.post("/{job_id}/takes/{take_id}/restore", status_code=202, response_model=RecutStarted)
async def restore_take(
    job_id: str,
    take_id: str,
    request: Request,
    background: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(allowed_user),
) -> RecutStarted:
    job = await _load_owned_job(session, job_id, user)
    job_dir = _job_dir(job_id)
    group = tk.take_group(job_dir, take_id)
    if not group:
        raise HTTPException(status_code=404, detail="That take can't be brought back.")
    scenes = _scene_orders(job_dir)
    pipeline = pipeline_stage_names(get_settings())
    restored = sorted({Item(t["kind"], t.get("unit")) for t in group})
    api_key = _request_key(request)
    previous = job.status
    await _admit(session, job, user)
    try:
        inv = closure(restored, scenes, pipeline)
        rebuild = [i for i in inv.items if i not in restored]
        labels = ", ".join(i.label for i in restored)
        summary = f"Brought back {labels} from cut {group[0]['cut'] - 1}"
        cut = tk.start_cut(job_dir, picked=restored, items=inv.items, note="", summary=summary,
                           cost_now=job.cost_usd or 0.0, kind="restore", created_at=job.created_at.isoformat())
        tk.archive(job_dir, inv.items, cut)
        tk.bring_back(job_dir, group)
        inv = Invalidation(picked=restored, items=inv.items,
                           stages=[s for s in inv.stages if s not in ("image_gen", "motion") or any(
                               i.kind in ("frame", "clip") for i in rebuild)])
        await _prepare_run(session, job, inv, cut, summary, review=False)
    except Exception as exc:  # noqa: BLE001
        await repo.set_status(session, job_id, JobStatus(previous), error=None)
        raise HTTPException(status_code=500, detail=f"Couldn't bring that take back: {redact(str(exc))}") from None
    await _launch(background, job, api_key)
    return RecutStarted(job_id=job_id, status=JobStatus.RUNNING, cut=cut)
