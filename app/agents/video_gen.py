"""AI video stage (Gemini Omni): every frame becomes a generated clip; dialogue scenes
are spoken on camera. Runs under the ``motion`` stage name.

A scene blocked by a content policy is rewritten to comply (and, if needed, its start
frame redrawn) instead of failing. Billing and quota errors still stop the job for Resume.
Each finished clip is recorded in the manifest immediately, so an interruption never
re-bills it.
"""

from __future__ import annotations

import asyncio
import dataclasses

from pydantic import BaseModel

from app.adapters.errors import ProviderError
from app.adapters.omni import ContentBlocked, SceneSpec, ShotSpec, has_audio
from app.agents.image_gen import render_frame
from app.core.logging import get_logger
from app.graph.state import Deps, MotionClipMeta, StoryboardState
from app.prompts import render
from app.schemas import Script, Shotlist

log = get_logger(__name__)

MANIFEST_VERSION = "omni-2"  # omni-2: one multi-shot clip per scene


class AIVideoUnavailable(RuntimeError):
    """A scene couldn't get an AI clip and stills aren't allowed (require_ai_video)."""


_CAMERA = {
    "setup": "slow establishing push-in",
    "inciting": "gentle handheld drift toward the subject",
    "rising": "steady tracking move that follows the action",
    "climax": "dynamic, intimate camera move that heightens the tension",
    "resolution": "slow, calm pull-back",
}
_BLOCK_LABEL = {
    "content_blocked": "third-party content rule (existing characters, brands or real people)",
    "prohibited_content": "prohibited content rule (violence, weapons or harm)",
    "safety": "safety filter on the generated video (usually physical confrontation or someone shown in danger)",
}
_RULES = {"content_blocked": "omni/rescue_rules_third_party", "safety": "omni/rescue_rules_safety"}


class _Line(BaseModel):
    speaker: str
    text: str
    delivery: str = ""


class SceneRescue(BaseModel):
    scene: str
    image_prompt: str
    lines: list[_Line] = []
    change: str = ""


def _label(code: str) -> str:
    return _BLOCK_LABEL.get(code, f"content rule ({code})")


def apply_rescue(spec: SceneSpec, r: SceneRescue | dict) -> SceneSpec:
    """Take the rewritten scene; lines change only when the rewrite returns the same speakers in order."""
    r = SceneRescue.model_validate(r)
    lines, shots = spec.lines, spec.shots
    if spec.mode == "dialogue" and r.lines:
        new = tuple((l.speaker.strip().upper(), l.text, l.delivery) for l in r.lines)
        if [s for s, _, _ in new] == [s for s, _, _ in spec.lines]:
            lines, it = new, iter(new)
            shots = tuple(dataclasses.replace(sh, lines=tuple(next(it) for _ in sh.lines)) for sh in spec.shots)
    return dataclasses.replace(spec, scene=r.scene, lines=lines, shots=shots)


def _rewrite(deps: Deps, spec: SceneSpec, code: str, message: str, attempt: int) -> SceneRescue:
    rules = render(_RULES.get(code, "omni/rescue_rules_prohibited"))
    note = "" if attempt == 1 else (
        "A first rewrite was ALSO blocked. Be more conservative: remove the triggering element "
        "from the shot entirely and keep the action simple."
    )
    lines = "\n".join(f"{s}: {t} ({d})" for s, t, d in spec.lines) or "(none)"
    if spec.shots:
        shots = "\n".join(f"- {sh.size}: {sh.action}" for sh in spec.shots)
        scene_text = f"{spec.scene}\nShots:\n{shots}"
    else:
        scene_text = spec.scene

    def mock(_u: str) -> SceneRescue:
        return SceneRescue(scene=f"{spec.scene} (rewritten)", image_prompt=spec.scene,
                           lines=[_Line(speaker=s, text=t, delivery=d) for s, t, d in spec.lines], change="mock rewrite")

    return deps.llm.complete_json(
        system=render("omni/rescue_system", rules=rules),
        user=render("omni/rescue_user", code=code, message=message[:500], attempt_note=note, mode=spec.mode,
                    language=spec.language, scene=scene_text, lines=lines, characters=", ".join(spec.characters) or "none"),
        response_model=SceneRescue,
        role="designer",
        mock_factory=mock,
    )


def _shot_specs(sc) -> tuple[ShotSpec, ...]:
    if sc is None:
        return ()
    return tuple(
        ShotSpec(size=sh.size, action=sh.action, camera=sh.camera, seconds=sh.seconds,
                 lines=tuple((l.speaker, l.text, l.delivery) for l in sh.lines))
        for sh in sc.shots
    )


def _setting(sc, places: dict[str, str]) -> str:
    if sc is None:
        return ""
    look = places.get(sc.location.strip().upper(), "")
    parts = [p for p in (look or sc.location.replace("_", " ").lower(), sc.time_of_day) if p]
    return ", ".join(parts)


def _probe_seconds(path: str) -> float:
    import subprocess

    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True,
        ).stdout
        return float(out.strip() or 0)
    except Exception:  # noqa: BLE001
        return 0.0


async def video_gen(state: StoryboardState, deps: Deps) -> StoryboardState:
    store, reporter = deps.store, deps.reporter
    frames = sorted(state.get("frames") or [], key=lambda f: f["order"])
    if not frames or deps.video is None:
        log.info("AI video off; scenes use their stills")
        if reporter is not None:
            await reporter.mark_skipped("AI video off")
        return {"motion_clips": []}

    script = Script.model_validate(store.load_json(state["script_path"]))
    shotlist = Shotlist.model_validate(store.load_json(state["shotlist_path"]))
    shots = {s.order: s for s in shotlist.shots}
    scenes = {sc.order: sc for sc in script.scenes}
    voices = {c.name.upper(): c.voice_description for c in script.characters}
    places = {loc.tag.strip().upper(): loc.look for loc in shotlist.locations}
    holds = {a["order"]: a.get("hold") for a in state.get("scene_audio") or []}
    language = state.get("language") or script.language
    prior = {c["order"]: c for c in (store.load_json(store.abspath("motion_manifest.json")) or {}).get("scenes", [])} \
        if store.exists("motion_manifest.json") else {}

    total, done = len(frames), 0
    clips: dict[int, MotionClipMeta] = {}
    warnings: list[str] = []
    blocked: dict[int, ContentBlocked] = {}
    stills: list[int] = []
    strict = deps.settings.require_ai_video and not deps.allow_stills
    sem = asyncio.Semaphore(max(1, deps.settings.omni_max_concurrency))
    lock = asyncio.Lock()

    async def note(msg: str) -> None:
        log.info(msg)  # the log bridge copies it into the stage's log

    def save_manifest() -> str:
        ordered = [dict(clips[o]) for o in sorted(clips)]
        return store.save_json("motion_manifest.json",
                               {"pipeline_version": MANIFEST_VERSION, "policy": "omni", "scenes": ordered})

    async def record(order: int, meta: MotionClipMeta) -> None:
        async with lock:
            clips[order] = meta
            save_manifest()  # survives an interruption: Resume never re-bills this clip

    async def gen(order: int, image: bytes, spec: SceneSpec):
        async with sem:
            return await asyncio.to_thread(deps.video.generate, image, spec, key=f"{state['job_id']}-{order}")

    async def rescue(order: int, image: bytes, spec: SceneSpec, exc: ContentBlocked):
        await note(f"scene {order}: blocked by Google's {_label(exc.code)}; rewriting the scene to comply")
        if exc.code != "content_blocked":  # this classifier is inconsistent: one unchanged retry first
            try:
                return await gen(order, image, spec), None
            except ContentBlocked as e:
                exc = e
        for attempt in (1, 2):
            try:
                rewrite = await asyncio.to_thread(_rewrite, deps, spec, exc.code, str(exc), attempt)
            except ProviderError as e:
                log.warning("scene %s: rewrite failed (%s)", order, e)
                break
            new_spec = apply_rescue(spec, rewrite)
            log.info("scene %s: rewrite %d tried: %s", order, attempt, rewrite.change[:200])
            redraws = ((False, True) if attempt == 1 else (True,)) if deps.image is not None else (False,)
            for redraw in redraws:
                img = image
                if redraw:
                    try:
                        img = await asyncio.to_thread(render_frame, deps, state, rewrite.image_prompt, spec.characters,
                                                    shots[order].location if order in shots else "")
                    except ProviderError as e:
                        log.warning("scene %s: redrawing the frame failed (%s)", order, e)
                        continue
                try:
                    clip = await gen(order, img, new_spec)
                except ContentBlocked as e:
                    exc = e
                    continue
                await note(f"scene {order}: rewrite accepted{' with a redrawn frame' if redraw else ''}: {rewrite.change}")
                return clip, (rewrite, redraw, img, exc.code)
        raise exc

    async def save_rescue(order: int, info) -> None:
        rewrite, redraw, img, code = info
        if redraw:
            store.save_bytes(f"frame_{order:02d}.blocked.png", store.load_bytes(store.abspath(f"frame_{order:02d}.png")))
            store.save_bytes(f"frame_{order:02d}.png", img)
        store.save_json(f"rescue_{order:02d}.json", {"code": code, "redrawn_frame": redraw, **rewrite.model_dump()})
        async with lock:  # keep the script panel truthful about what's said on screen
            s = Script.model_validate(store.load_json(state["script_path"]))
            for sc in s.scenes:
                if sc.order == order:
                    sc.description = rewrite.scene
                    if sc.mode == "dialogue" and rewrite.lines and len(rewrite.lines) == len(sc.shot_lines() or sc.lines):
                        for line, new in zip(sc.shot_lines() or sc.lines, rewrite.lines):
                            line.text, line.delivery = new.text, new.delivery or line.delivery
                        sc.lines = sc.shot_lines() or sc.lines
            store.save_json("script.json", s)

    if reporter is not None:
        reporter.log(f"generating {total} AI video clip(s) with {deps.settings.omni_model_chain[0]}")
        await reporter.progress(0, total)

    async def one(frame) -> None:
        nonlocal done
        order = frame["order"]
        name = f"motion_{order:02d}.mp4"
        sc = scenes.get(order)
        try:
            if store.exists(name) and order in prior:  # resume: never re-bill a finished clip
                clips[order] = MotionClipMeta(**prior[order])
                return
            if store.exists(name) and (secs := _probe_seconds(store.abspath(name))) > 0:
                # Finished after the manifest was last written (e.g. the process was stopped): adopt it.
                await record(order, MotionClipMeta(
                    order=order, path=store.abspath(name), status="ok", tier="omni", motion_kind="generative",
                    renderer="omni", model=deps.settings.omni_model_chain[0], source_seconds=secs,
                    pipeline_version=MANIFEST_VERSION, attempts=[], motion_reason="recovered after a restart",
                    has_audio=has_audio(store.abspath(name)),
                ))
                await note(f"scene {order}/{total}: reused a clip finished before the restart")
                return
            if deps.allow_stills:  # the user chose to finish now with this scene's still
                stills.append(order)
                await note(f"scene {order}/{total}: using its still image (finishing without this clip)")
                return
            spec = SceneSpec(
                order=order,
                scene=(shots[order].image_prompt if order in shots else (sc.description if sc else "")),
                motion=_CAMERA.get((sc.beat_role if sc else "").lower(), ""),
                mode=sc.mode if sc else "narration",
                language=language,
                characters=tuple(sc.characters) if sc else (),
                lines=tuple((l.speaker, l.text, l.delivery) for l in sc.lines) if sc else (),
                voices={s: voices.get(s, "") for s in (sc.characters if sc else [])},
                seconds=float(holds.get(order) or (sc.seconds if sc else 0) or deps.settings.seconds_per_frame),
                shots=_shot_specs(sc),
                setting=_setting(sc, places),
            )
            if store.exists(f"rescue_{order:02d}.json"):  # resume: start from the accepted rewrite
                spec = apply_rescue(spec, store.load_json(store.abspath(f"rescue_{order:02d}.json")))
            image = store.load_bytes(frame["path"])
            try:
                clip, info = await gen(order, image, spec), None
            except ContentBlocked as exc:
                clip, info = await rescue(order, image, spec, exc)
            path = store.save_bytes(name, clip.data)
            if info is not None:
                await save_rescue(order, info)
            blocked.pop(order, None)
            await record(order, MotionClipMeta(
                order=order, path=path, status="ok", tier="omni", motion_kind="generative",
                renderer="omni", model=clip.model, source_seconds=float(clip.seconds),
                pipeline_version=MANIFEST_VERSION, attempts=[],
                motion_reason="rewritten to meet content rules" if info else None,
                has_audio=has_audio(path), qc=list(clip.qc), renders=clip.renders,
            ))
            if reporter is not None:
                shots_note = f", {len(spec.shots)} shots" if len(spec.shots) > 1 else ""
                qc_note = f"; QC: {'; '.join(clip.qc)}" if clip.qc else ""
                reporter.log(f"scene {order}/{total}: {clip.seconds}s clip{shots_note} ({clip.model}, {clip.renders} render(s)){qc_note}")
        except ContentBlocked as exc:
            log.warning("AI video for scene %s still blocked after rewrites (%s)", order, exc)
            blocked[order] = exc
            warnings.append(f"video: scene {order} blocked by Google's {_label(exc.code)}")
        except ProviderError as exc:
            log.warning("AI video for scene %s failed (%s); keeping the still", order, exc)
            warnings.append(f"video: scene {order} not generated ({exc}); kept still")
        finally:
            async with lock:
                done += 1
                if reporter is not None:
                    await reporter.progress(done, total)

    await asyncio.gather(*(one(f) for f in frames))
    # Quota/rate failures usually clear after a pause; content blocks were already rescued inline.
    retry = [f for f in frames if f["order"] not in clips and f["order"] not in blocked]
    if retry and strict:
        log.info("retrying %d scene(s) in %.0fs", len(retry), deps.settings.ai_video_retry_delay_seconds)
        await asyncio.sleep(deps.settings.ai_video_retry_delay_seconds)
        warnings[:] = [w for w in warnings if "blocked by Google" in w]
        done = total - len(retry)
        await asyncio.gather(*(one(f) for f in retry))
    missing = [f["order"] for f in frames if f["order"] not in clips and f["order"] not in stills]
    ordered = [clips[o] for o in sorted(clips)]
    manifest = save_manifest()
    if missing and strict:
        kept = f"The {len(ordered)} finished clip(s) are kept"
        if all(o in blocked for o in missing):
            codes = sorted({blocked[o].code for o in missing})
            advice = (
                "Change the idea to use original characters (no existing heroes, brands or real people) and start a new film."
                if codes == ["content_blocked"] else
                "Select Resume to try that scene again (this filter is inconsistent), or Finish with stills."
            )
            raise AIVideoUnavailable(
                f"Google's {', '.join(_label(c) for c in codes)} blocked scene(s) {', '.join(map(str, missing))} "
                f"even after rewriting them. {kept}. {advice}"
            )
        reason = next((w.split("(", 1)[-1].rstrip(")") for w in reversed(warnings) if "not generated" in w),
                      "provider unavailable")
        raise AIVideoUnavailable(
            f"AI video couldn't be generated for scene(s) {', '.join(map(str, missing))} ({reason}). "
            f"{kept}; select Resume to finish later (daily AI-video quota resets at midnight Pacific)."
        )
    if stills:
        warnings.append(f"video: scene(s) {', '.join(map(str, sorted(stills)))} use their still image (your choice)")
    if warnings and reporter is not None:
        await reporter.mark_degraded(f"{len(warnings)} scene(s) kept stills")
    log.info("AI video: %d/%d scene(s) generated", len(ordered), total)
    return {"motion_clips": ordered, "motion_manifest_path": manifest, "warnings": warnings,
            "stills_used": sorted(stills)}
