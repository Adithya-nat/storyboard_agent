"""motion agent: ordered CLEAN frames -> per-scene motion clips (external engine).

Sits between image_gen and assembler ONLY when settings.motion_engine_enabled.
Each frame is sent to the storyboard motion engine (parallax / Veo / Ken Burns,
decided engine-side) and the returned clip is saved as ``motion_NN.mp4`` in the
job dir. The assembler later splices those clips into the film.

Never fails the job. Every failure path degrades to "no clip for this scene",
which the assembler + UI treat as the still image (today's exact behavior):
  - engine client not wired / unreachable  -> mark stage degraded, produce nothing
  - a single frame errors or times out      -> warning, that scene keeps its still
  - the whole stage exceeds its budget       -> whatever finished is kept, rest still
  - ANY unexpected error                     -> caught, empty result, job continues

Idempotent (resume): a ``motion_NN.mp4`` already on disk is reused, never re-billed.
"""

from __future__ import annotations

import asyncio

from app.core.logging import get_logger
from app.graph.state import Deps, MotionClipMeta, StoryboardState

log = get_logger(__name__)

_OK_TIERS = {"veo", "parallax"}
_GENERATIVE_KINDS = {"generative", "local_generative"}
_MANIFEST_VERSION = "2"


async def motion(state: StoryboardState, deps: Deps) -> StoryboardState:
    reporter = deps.reporter
    frames = sorted(state.get("frames", []) or [], key=lambda f: f["order"])
    if not frames:
        return {"motion_clips": [], "warnings": []}
    if deps.motion is None:
        return {"motion_clips": [], "warnings": ["motion: engine not configured; scenes keep stills"]}

    try:
        return await _run(state, deps, frames)
    except Exception as exc:  # noqa: BLE001 - motion NEVER fails the job
        log.warning("motion stage errored (%s); scenes keep their stills", exc, exc_info=True)
        if reporter is not None:
            await reporter.mark_degraded(f"motion stage error ({exc}); scenes kept stills")
        return {"motion_clips": [], "warnings": [f"motion: stage error ({exc}); scenes kept stills"]}


async def _run(state: StoryboardState, deps: Deps, frames: list) -> StoryboardState:
    store = deps.store
    reporter = deps.reporter
    total = len(frames)

    health = await deps.motion.health()
    if health is None:
        msg = "motion engine unreachable; scenes keep their still Ken Burns"
        log.warning(msg)
        if reporter is not None:
            reporter.log(msg)
            await reporter.mark_degraded("motion engine unreachable")
        return {"motion_clips": [], "warnings": [f"motion: {msg}"]}

    beat_by_order = _load_beats(store, state)
    prompt_by_order, narr_by_order = _load_shots(store, state)
    min_order = frames[0]["order"]
    policy = health.get("policy") or "legacy"
    prior_manifest = _load_motion_manifest(store)
    prior_scenes = {
        int(scene["order"]): scene
        for scene in prior_manifest.get("scenes", [])
        if "order" in scene
    }

    if reporter is not None:
        reporter.log(f"animating {total} scene(s) via motion engine (tiers={health.get('tiers')})")
        await reporter.progress(0, total)

    sem = asyncio.Semaphore(max(1, deps.settings.motion_max_concurrency))
    clips: dict[int, MotionClipMeta] = {}
    warnings: list[str] = []
    done = 0
    step_lock = asyncio.Lock()

    async def animate(frame) -> None:
        nonlocal done
        order = frame["order"]
        name = f"motion_{order:02d}.mp4"
        async with sem:
            try:
                prior = prior_scenes.get(order, {})
                reusable = store.exists(name) and (
                    policy != "generative_all"
                    or (
                        prior_manifest.get("pipeline_version") == _MANIFEST_VERSION
                        and prior.get("motion_kind") in _GENERATIVE_KINDS
                    )
                )
                if reusable:
                    clips[order] = MotionClipMeta(
                        order=order,
                        path=store.abspath(name),
                        status="ok",
                        tier=prior.get("tier") or "cached",
                        motion_kind=prior.get("motion_kind"),
                        renderer=prior.get("renderer") or "cached",
                        model=prior.get("model"),
                        source_seconds=prior.get("source_seconds"),
                        pipeline_version=prior.get("pipeline_version"),
                        attempts=prior.get("attempts") or [],
                        motion_reason=prior.get("motion_reason"),
                    )
                else:
                    meta = {
                        "job_id": state["job_id"],
                        "order": order,
                        "image_prompt": prompt_by_order.get(order, ""),
                        "narration": narr_by_order.get(order, frame.get("caption", "")),
                        "beat_role": beat_by_order.get(order, ""),
                        "hero_hint": order == min_order
                        or beat_by_order.get(order, "").strip().lower() == "climax",
                        "duration_hint": deps.settings.motion_clip_seconds,
                    }
                    image = store.load_bytes(frame["path"])
                    task, data = await _animate_one(deps, image, meta)
                    tier = task.get("tier") or "unknown"
                    motion_kind = task.get("motion_kind")
                    path = store.save_bytes(name, data)
                    status = (
                        "ok"
                        if motion_kind in _GENERATIVE_KINDS
                        or (motion_kind is None and tier in _OK_TIERS)
                        else "degraded"
                    )
                    attempts = task.get("attempts") or []
                    reason = next(
                        (
                            attempt.get("reason")
                            for attempt in attempts
                            if attempt.get("outcome") == "failed" and attempt.get("reason")
                        ),
                        None,
                    )
                    clips[order] = MotionClipMeta(
                        order=order,
                        path=path,
                        status=status,
                        tier=tier,
                        motion_kind=motion_kind,
                        renderer=task.get("renderer") or tier,
                        model=task.get("model"),
                        source_seconds=task.get("source_seconds"),
                        pipeline_version=task.get("pipeline_version"),
                        attempts=attempts,
                        motion_reason=reason,
                    )
                    if reporter is not None:
                        reporter.log(f"scene {order}/{total}: animated ({tier})")
            except Exception as exc:  # noqa: BLE001 - degrade this scene only
                log.warning("motion for scene %s failed (%s); keeping still", order, exc)
                warnings.append(f"motion: scene {order} not animated ({exc}); kept still")
            finally:
                async with step_lock:
                    done += 1
                    if reporter is not None:
                        await reporter.progress(done, total)

    async with deps.motion:
        tasks = [asyncio.create_task(animate(f)) for f in frames]
        try:
            async with asyncio.timeout(deps.settings.motion_stage_budget_seconds):
                await asyncio.gather(*tasks)
        except TimeoutError:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            missing = total - len(clips)
            warnings.append(
                f"motion: stage budget exceeded; {missing} scene(s) kept their stills"
            )
            log.warning("motion stage budget exceeded; %d scene(s) kept stills", missing)

    ordered = [clips[o] for o in sorted(clips)]
    fallback_count = sum(
        1 for clip in ordered if clip.get("status") == "degraded"
    )
    if policy == "generative_all" and fallback_count:
        warnings.append(
            f"motion: {fallback_count} scene(s) used non-generative fallback"
        )

    if warnings and reporter is not None:
        await reporter.mark_degraded(f"{len(warnings)} scene(s) not animated (kept stills)")

    manifest_path = store.save_json(
        "motion_manifest.json",
        {
            "pipeline_version": _MANIFEST_VERSION,
            "policy": policy,
            "scenes": [dict(clip) for clip in ordered],
        },
    )
    log.info("motion: %d/%d scene(s) animated", len(ordered), total)
    return {
        "motion_clips": ordered,
        "motion_manifest_path": manifest_path,
        "warnings": warnings,
    }


async def _animate_one(deps: Deps, image: bytes, meta: dict) -> tuple[dict, bytes]:
    """Submit one frame and poll to completion; return (task metadata, mp4 bytes).

    Bounded by the caller's overall asyncio.timeout budget — this loop has no
    private deadline, so it can't outlive the stage.
    """
    poll = max(0.1, deps.settings.motion_poll_interval_seconds)
    task_id = await deps.motion.submit(image, meta)
    while True:
        task = await deps.motion.get_task(task_id)
        status = task.get("status")
        if status == "done":
            return task, await deps.motion.download(task_id)
        if status == "failed":
            raise RuntimeError(f"engine task failed: {task.get('error')}")
        await asyncio.sleep(poll)


def _load_motion_manifest(store) -> dict:
    if not store.exists("motion_manifest.json"):
        return {}
    try:
        return store.load_json(store.abspath("motion_manifest.json"))
    except Exception:  # noqa: BLE001 - a stale/corrupt manifest means regenerate
        return {}


def _load_beats(store, state: StoryboardState) -> dict[int, str]:
    path = state.get("script_path")
    if not path:
        return {}
    try:
        data = store.load_json(path)
        return {
            s["order"]: s.get("beat_role", "")
            for s in data.get("scenes", [])
            if "order" in s
        }
    except Exception:  # noqa: BLE001 - metadata is best-effort context
        return {}


def _load_shots(store, state: StoryboardState) -> tuple[dict[int, str], dict[int, str]]:
    path = state.get("shotlist_path")
    if not path:
        return {}, {}
    try:
        data = store.load_json(path)
        shots = data.get("shots", [])
        prompts = {s["order"]: s.get("image_prompt", "") for s in shots if "order" in s}
        narr = {s["order"]: s.get("narration", "") for s in shots if "order" in s}
        return prompts, narr
    except Exception:  # noqa: BLE001
        return {}, {}
