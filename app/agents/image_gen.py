"""image_gen agent: Shotlist shots -> one CLEAN frame image per shot.

Three phases so we NEVER ship a gap OR an ugly placeholder card to the user:

    Phase A  attempt (bounded pool)   Skip clean on-disk frames (resume/idempotency).
             Generate the rest conditioned on style_ref. Collect CLEAN bytes / failures.
    Phase B  fill gaps                Each failure -> reuse the nearest successful
             neighbor's CLEAN bytes (no gap, no jarring card). All failed, no
             neighbor -> a styled placeholder card (base case only).
    Phase C  save                     Write CLEAN frame_NN.png. Failed/filled keep a
             .degraded marker so resume re-attempts real generation, and report
             status=degraded (honest in GET).

Frames are saved CLEAN (no burned caption): the assembler burns/overlays the
caption later, so a borrowed neighbor never carries the wrong caption, and Wave 2
(caption as ffmpeg overlay) needs no frame changes.

Concurrency: an asyncio.Semaphore bounds how many image calls run at once; the
shared token bucket inside the adapter bounds the per-minute rate (so re-rolls +
concurrency provably stay under the model's RPM). Every frame is conditioned on the
designer's single style_ref.png — the load-bearing consistency lever.
"""

from __future__ import annotations

import asyncio
import os

from app.adapters.errors import ProviderError
from app.core.images import render_card
from app.core.logging import get_logger
from app.graph.state import Deps, FrameMeta, StoryboardState
from pydantic import BaseModel

from app.observability import usage
from app.prompts import render
from app.schemas import Shot, Shotlist
from app.workers.control import PauseRequested, gate, keep, paused

log = get_logger(__name__)


async def image_gen(state: StoryboardState, deps: Deps) -> StoryboardState:
    store = deps.store
    reporter = deps.reporter  # None in pure unit tests -> all calls below are skipped
    shotlist = Shotlist.model_validate(store.load_json(state["shotlist_path"]))
    shots = sorted(shotlist.shots, key=lambda s: s.order)
    existing_orders = {s.order for s in shots}
    total = len(shots)

    ref_image, sheets = load_refs(store, state)
    plates = load_plates(store, state)
    cast_looks = _cast_looks(store, state)

    def refs_for(shot) -> list:
        return frame_refs(ref_image, sheets, shot.characters, plates.get(shot.location.strip().upper()))

    def prompt_for(shot) -> str:
        return frame_prompt(shot.image_prompt, shot.characters, sheets, cast_looks, shot.location.strip().upper() in plates)

    sem = asyncio.Semaphore(max(1, deps.settings.image_max_concurrency))

    if reporter is not None:
        reporter.log(
            f"generating {total} frame(s) "
            f"(≤{deps.settings.image_max_concurrency} concurrent)"
        )
        await reporter.progress(0, total)

    # --- Phase A: attempt every missing/degraded shot (bounded pool) -------- #
    clean_bytes: dict[int, bytes] = {}   # generated THIS run (un-captioned)
    reused_clean: set[int] = set()       # already clean on disk -> skipped, kept
    failures: dict[int, str] = {}
    held: set[int] = set()  # frames a pause stopped before they were generated
    done = 0  # frames resolved (generated / reused / failed) -> drives progress

    async def attempt(shot: Shot) -> None:
        nonlocal done
        frame_name = f"frame_{shot.order:02d}.png"
        marker_name = f"frame_{shot.order:02d}.degraded"
        # Idempotency (resume): a clean frame already on disk is kept, not re-billed.
        if store.exists(frame_name) and not store.exists(marker_name):
            reused_clean.add(shot.order)
            if reporter is not None:
                reporter.log(f"frame {shot.order}/{total}: reused clean frame (resume)")
        else:
            async with sem:
                try:
                    await gate(deps)  # a pause stops frames that haven't started
                except PauseRequested:
                    held.add(shot.order)
                    return
                try:
                    raw = await keep(deps, lambda: _render_and_save(shot), "image_gen", shot.order)
                    clean_bytes[shot.order] = raw
                    if reporter is not None:
                        reporter.log(f"frame {shot.order}/{total}: generated")
                        await reporter.publish_frame(shot.order, store.abspath(frame_name))
                except ProviderError as exc:
                    # The bridge captures this log.warning into job_logs (image_gen
                    # stage) via the contextvar — no explicit reporter.log needed.
                    log.warning("frame %s generation failed: %s", shot.order, exc)
                    failures[shot.order] = str(exc)
        done += 1  # single-threaded event loop: safe (no await between read+write)
        if reporter is not None:
            await reporter.progress(done, total)

    async def _render_and_save(shot: Shot) -> bytes:
        # Saved straight away so a pause can't lose a paid frame (Phase C re-saves the same bytes).
        raw = await _render_shot(shot)
        store.save_bytes(f"frame_{shot.order:02d}.png", raw)
        store.remove(f"frame_{shot.order:02d}.degraded")
        return raw

    async def _render_shot(shot: Shot) -> bytes:
        raw = await asyncio.to_thread(
            deps.image.generate_image,
            prompt_for(shot),
            refs=refs_for(shot),
            seed=deps.settings.image_seed,
        )
        if await asyncio.to_thread(_identity_mismatch, deps, raw, shot.characters, sheets, cast_looks):
            log.info("frame %s: a character didn't match their reference; re-rendering once", shot.order)
            raw = await asyncio.to_thread(
                deps.image.generate_image, prompt_for(shot), refs=refs_for(shot), seed=deps.settings.image_seed
            )
        return raw

    await asyncio.gather(*(attempt(s) for s in shots))

    if held:
        # Keep every frame finished before the pause (clean, no degraded marker). Failed
        # frames are NOT neighbour-filled here: resume retries them for real.
        for order, raw in clean_bytes.items():
            store.save_bytes(f"frame_{order:02d}.png", raw)
            store.remove(f"frame_{order:02d}.degraded")
        kept = len(clean_bytes) + len(reused_clean)
        log.info("frames paused: %d/%d kept; resume starts at frame %d", kept, total, min(held))
        raise paused("image_gen", kept, total, "frames")

    # QA toggle: force one degraded frame so the completed_with_warnings UI is
    # testable (mock mode otherwise only ever yields clean frames). Treat the first
    # shot as a failure -> Phase B neighbour-fills it -> status=degraded.
    if deps.settings.mock_force_degraded_frame and shots:
        victim = shots[0].order
        clean_bytes.pop(victim, None)
        reused_clean.discard(victim)
        failures.setdefault(victim, "forced degraded frame (QA toggle)")

    # --- Phase B helpers: nearest CLEAN neighbor (memory or clean on disk) -- #
    def clean_source(order: int) -> bytes | None:
        if order in clean_bytes:
            return clean_bytes[order]
        if order in reused_clean:
            return store.load_bytes(store.abspath(f"frame_{order:02d}.png"))
        return None

    def nearest_clean(order: int) -> bytes | None:
        available = [o for o in existing_orders if clean_source(o) is not None]
        if not available:
            return None
        best = min(available, key=lambda o: (abs(o - order), o))
        return clean_source(best)

    # --- Phase C: save clean frames; fill gaps seamlessly ------------------- #
    frames: list[FrameMeta] = []
    warnings: list[str] = []
    for shot in shots:
        order = shot.order
        frame_name = f"frame_{order:02d}.png"
        marker_name = f"frame_{order:02d}.degraded"

        if order in reused_clean:
            frames.append(
                FrameMeta(order=order, path=store.abspath(frame_name),
                          caption=shot.narration, status="ok")
            )
            continue

        if order in clean_bytes:
            path = store.save_bytes(frame_name, clean_bytes[order])
            store.remove(marker_name)  # clear any stale marker from a prior degraded run
            frames.append(FrameMeta(order=order, path=path, caption=shot.narration, status="ok"))
            continue

        # Failed -> seamless neighbor-fill (no gap, no ugly card). Only the
        # all-frames-failed base case (no clean neighbor anywhere) shows a placeholder.
        fill = nearest_clean(order)
        if fill is not None:
            data, how = fill, "reused a neighbouring frame"
        else:
            data, how = render_card(shot.image_prompt, shot.narration, degraded=True), "used a placeholder"
        path = store.save_bytes(frame_name, data)
        store.write_marker(marker_name)  # survive restart -> re-try real gen on resume
        warnings.append(
            f"frame {order}: image generation failed "
            f"({failures.get(order, 'unknown')}); {how} to avoid a gap"
        )
        frames.append(FrameMeta(order=order, path=path, caption=shot.narration, status="degraded"))

    # Mark the STAGE degraded (not the job) so the front end shows it amber and the
    # wrapper's stage_finish no-ops. The runner still derives the job's
    # completed_with_warnings status from the degraded frames in _persist.
    if warnings and reporter is not None:
        await reporter.mark_degraded(
            f"{len(warnings)} of {total} frame(s) degraded (neighbour-filled / placeholder)"
        )

    return {"frames": frames, "warnings": warnings}


def load_plates(store, state) -> dict[str, bytes]:
    return {t.upper(): store.load_bytes(p) for t, p in (state.get("location_paths") or {}).items() if os.path.exists(p)}


def load_refs(store, state) -> tuple[bytes | None, dict[str, list[bytes]]]:
    """Style ref + each character's reference sheets (missing => text-only for that part)."""
    style_ref_path = state.get("style_ref_path")
    ref_image = store.load_bytes(style_ref_path) if style_ref_path and os.path.exists(style_ref_path) else None
    sheets = {
        name: [store.load_bytes(p) for p in paths if os.path.exists(p)]
        for name, paths in (state.get("cast_paths") or {}).items()
    }
    return ref_image, sheets


def frame_refs(ref_image: bytes | None, sheets: dict, characters, plate: bytes | None = None) -> list:
    out: list = [("Style reference:", ref_image)] if ref_image else []
    if plate:
        out.append(("Location reference (the exact set: layout, props, light):", plate))
    for name in characters:
        for k, img in enumerate(sheets.get(name.upper(), [])):
            out.append((f"{name.upper()} — character reference {k + 1}:", img))
    return out[:14]


def frame_prompt(image_prompt: str, characters, sheets: dict, looks: dict, plate: bool = False) -> str:
    names = [n.upper() for n in characters if sheets.get(n.upper())]
    if plate:
        image_prompt += "\n\nThe set must match the labelled location reference exactly (layout, props, light)."
    if not names:
        return image_prompt
    cast = "\n".join(f"- {n}: {looks.get(n, '')}" for n in names)
    return image_prompt + "\n\n" + render("image_gen/cast_rules", cast=cast)


def render_frame(deps: Deps, state, image_prompt: str, characters, location: str = "") -> bytes:
    """One cast-conditioned frame (with the identity check + one re-render). Blocking."""
    ref_image, sheets = load_refs(deps.store, state)
    plate = load_plates(deps.store, state).get(location.strip().upper())
    looks = _cast_looks(deps.store, state)
    prompt = frame_prompt(image_prompt, characters, sheets, looks, plate is not None)
    refs = frame_refs(ref_image, sheets, characters, plate)
    raw = deps.image.generate_image(prompt, refs=refs, seed=deps.settings.image_seed)
    if _identity_mismatch(deps, raw, characters, sheets, looks):
        raw = deps.image.generate_image(prompt, refs=refs, seed=deps.settings.image_seed)
    return raw


def _cast_looks(store, state) -> dict[str, str]:
    from app.schemas import Script

    if not state.get("script_path"):
        return {}
    script = Script.model_validate(store.load_json(state["script_path"]))
    return {c.name.upper(): c.look for c in script.characters}


class _CharMatch(BaseModel):
    name: str
    present: bool
    matches: bool


class _IdentityCheck(BaseModel):
    characters: list[_CharMatch]


_client = None


def _check_client(settings):
    """One long-lived client (an inline temporary gets garbage-collected and closed mid-call)."""
    global _client
    if _client is None:
        from app.adapters.genai_client import make_client

        _client = make_client(settings, timeout_seconds=120)
    return _client


def _identity_mismatch(deps: Deps, frame: bytes, characters, sheets: dict, looks: dict) -> bool:
    """True when a named character in the frame clearly doesn't match their reference sheet."""
    names = [n.upper() for n in characters if sheets.get(n.upper())]
    if not names or deps.settings.mock_mode or not deps.settings.image_identity_check:
        return False
    try:
        from google.genai import types

        from app.adapters.gemini import image_mime

        contents: list = [types.Part.from_bytes(data=frame, mime_type=image_mime(frame))]
        for n in names:
            contents += [f"{n} reference:", types.Part.from_bytes(data=sheets[n][0], mime_type=image_mime(sheets[n][0]))]
        contents.append(render("image_gen/identity_check", cast=", ".join(f"{n} ({looks.get(n, '')})" for n in names)))
        model = deps.settings.image_identity_check_model
        resp = _check_client(deps.settings).models.generate_content(
            model=model, contents=contents,
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=_IdentityCheck),
        )
        usage.record_gemini_call("text", model, getattr(resp, "usage_metadata", None))
        verdict = _IdentityCheck.model_validate_json(resp.text)
        return any(not (c.present and c.matches) for c in verdict.characters)  # every named character is expected
    except Exception as exc:  # noqa: BLE001 - the check never blocks a frame
        log.warning("identity check skipped (%s)", exc)
        return False
