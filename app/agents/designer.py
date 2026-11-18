"""designer agent: Script -> Shotlist + style bible + style ref + character reference sheets.

Every artifact is load-or-produce (resume-safe). Failures degrade with a warning:
shotlist -> derived from the script; style ref / a sheet -> text-only conditioning.
"""

from __future__ import annotations

import asyncio
import re

from app.adapters.errors import ProviderError
from app.workers.control import JobPaused, PauseRequested, gate, keep
from app.core.logging import get_logger
from app.graph.state import Deps, StoryboardState
from app.prompts import render
from app.agents.redo_director import direction
from app.schemas import Script, Shot, Shotlist

log = get_logger(__name__)

SHEET_VIEWS = (
    "front-facing portrait, head and shoulders",
    "three-quarter view, waist up",
    "full body, standing, three-quarter view",
)
MAX_CAST = 4  # Nano Banana 2.1 holds character consistency for up to 4 characters


def _shots_from_script(script: Script) -> list[Shot]:
    return [
        Shot(
            order=s.order,
            image_prompt=f"{script.style}. {s.description}".strip(". "),
            location=s.location,
            narration=s.narration,
            characters=list(s.characters),
        )
        for s in script.scenes
    ]


def _mock_shotlist(script: Script) -> Shotlist:
    bible = (
        f"STYLE BIBLE (mock): {script.style or 'clean cinematic'}. Consistent palette, "
        f"lighting, and character design across all {len(script.scenes)} shots. "
        f"Title: {script.title}."
    )
    return Shotlist(style_bible=bible, shots=_shots_from_script(script))


def _shotlist_from_script(script: Script) -> Shotlist:
    """Fallback when the designer LLM fails."""
    return Shotlist(
        style_bible=script.style or "Consistent cinematic style across all shots.",
        shots=_shots_from_script(script),
    )


def _ref_prompt(shotlist: Shotlist) -> str:
    establishing = shotlist.shots[0].image_prompt if shotlist.shots else ""
    return render("designer/style_ref", style_bible=shotlist.style_bible, establishing=establishing).strip()


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper() or "CHAR"


def sheet_name(character: str, view: int) -> str:
    return f"char_{_slug(character)}_{view + 1}.png"


async def _build_cast(
    script: Script, shotlist: Shotlist, style_ref: bytes | None, deps: Deps
) -> tuple[dict[str, list[str]], list[str]]:
    """Reference sheets per main character. View 1 anchors identity for the other views."""
    store, warnings = deps.store, []
    cast = [c for c in script.characters if c.look.strip()][:MAX_CAST]
    sem = asyncio.Semaphore(max(1, deps.settings.image_max_concurrency))
    held_sheets: list[str] = []  # sheets a pause stopped before they started

    async def one(char) -> tuple[str, list[str]]:
        paths: list[str] = []
        anchor: bytes | None = None
        for v, view in enumerate(SHEET_VIEWS):
            name = sheet_name(char.name, v)
            if store.exists(name):
                paths.append(store.abspath(name))
                anchor = anchor or store.load_bytes(store.abspath(name))
                continue
            refs = [b for b in (style_ref, anchor) if b]
            prompt = render(
                "designer/character_sheet",
                name=char.name, look=char.look, view=view, style_bible=shotlist.style_bible[:1500],
            )
            try:
                async with sem:
                    await gate(deps)
                    img, path = await keep(deps, lambda: _draw(name, prompt, refs))
            except PauseRequested:
                held_sheets.append(name)
                break
            except ProviderError as exc:
                warnings.append(f"designer: {char.name} reference sheet failed ({exc}); text-only for that view")
                continue
            paths.append(path)
            anchor = anchor or img
        return char.name.upper(), paths

    async def _draw(name: str, prompt: str, refs: list[bytes]) -> tuple[bytes, str]:
        img = await asyncio.to_thread(deps.image.generate_image, prompt, refs=refs)
        return img, store.save_bytes(name, img)  # saved at once: a pause can't lose it

    results = await asyncio.gather(*(one(c) for c in cast))
    if held_sheets:
        kept = sum(len(p) for _, p in results)
        raise JobPaused("designer", f"Paused while casting; {kept} reference sheet(s) kept")
    cast_paths = {name: paths for name, paths in results if paths}
    if cast:
        log.info("cast ready: %s", ", ".join(f"{n} ({len(p)} views)" for n, p in cast_paths.items()) or "none")
    return cast_paths, warnings


def plate_name(tag: str) -> str:
    return f"loc_{_slug(tag)}.png"


async def _build_locations(shotlist: Shotlist, style_ref: bytes | None, deps: Deps) -> tuple[dict[str, str], list[str]]:
    """One empty-set reference plate per location tag (load-or-produce)."""
    store, warnings = deps.store, []
    used = {sh.location.strip().upper() for sh in shotlist.shots if sh.location.strip()}
    specs = {loc.tag.strip().upper(): loc for loc in shotlist.locations if loc.tag.strip().upper() in used}
    sem = asyncio.Semaphore(max(1, deps.settings.image_max_concurrency))
    held_plates: list[str] = []

    async def one(tag: str, look: str) -> tuple[str, str | None]:
        name = plate_name(tag)
        if store.exists(name):
            return tag, store.abspath(name)
        prompt = render("designer/location_plate", tag=tag, look=look, style_bible=shotlist.style_bible[:1500])
        async def draw() -> str:
            img = await asyncio.to_thread(deps.image.generate_image, prompt, refs=[b for b in (style_ref,) if b])
            return store.save_bytes(name, img)  # saved at once: a pause can't lose it

        try:
            async with sem:
                await gate(deps)
                return tag, await keep(deps, draw)
        except PauseRequested:
            held_plates.append(tag)
            return tag, None
        except ProviderError as exc:
            warnings.append(f"designer: location plate {tag} failed ({exc}); text-only set for that place")
            return tag, None

    results = await asyncio.gather(*(one(t, specs[t].look if t in specs else t) for t in sorted(used)))
    if held_plates:
        kept = sum(1 for _, p in results if p)
        raise JobPaused("designer", f"Paused while building sets; {kept} location plate(s) kept")
    paths = {t: p for t, p in results if p}
    if paths:
        log.info("locations ready: %s", ", ".join(paths))
    return paths, warnings


async def designer(state: StoryboardState, deps: Deps) -> StoryboardState:
    store = deps.store
    warnings: list[str] = []
    script = Script.model_validate(store.load_json(state["script_path"]))

    if store.exists("shotlist.json"):
        log.info("reusing the existing shotlist + style bible (resume)")
        shotlist = Shotlist.model_validate(store.load_json(store.abspath("shotlist.json")))
        shotlist_path = store.abspath("shotlist.json")
        style_bible_path = (
            store.abspath("style_bible.md")
            if store.exists("style_bible.md")
            else store.save_text("style_bible.md", shotlist.style_bible)
        )
    else:
        try:
            shotlist = await asyncio.to_thread(
                deps.llm.complete_json,
                system=render("designer/system"),
                user=render(
                    "designer/user",
                    script_json=script.model_dump_json(indent=2),
                    n=len(script.scenes),
                    first_order=script.scenes[0].order if script.scenes else 1,
                    direction=direction(store, "look"),
                ),
                response_model=Shotlist,
                role="designer",
                mock_factory=lambda _u: _mock_shotlist(script),
            )
        except ProviderError as exc:
            log.warning("designer shotlist failed, deriving from script: %s", exc)
            warnings.append(f"designer: shotlist generation failed ({exc}); used script scenes directly")
            shotlist = _shotlist_from_script(script)
        _sync_shots(shotlist, script)
        shotlist_path = store.save_json("shotlist.json", shotlist)
        style_bible_path = store.save_text("style_bible.md", shotlist.style_bible)
        log.info("locked the look: %d shots + a style bible", len(shotlist.shots))

    if store.exists("style_ref.png"):
        log.info("reusing the existing style reference (resume)")
        style_ref_path: str | None = store.abspath("style_ref.png")
    else:
        try:
            ref = await asyncio.to_thread(
                deps.image.generate_image, _ref_prompt(shotlist), seed=deps.settings.image_seed
            )
            style_ref_path = store.save_bytes("style_ref.png", ref)
            log.info("rendered the style-reference image that anchors every frame")
        except ProviderError as exc:
            log.warning("designer style_ref failed, falling back to text-only: %s", exc)
            warnings.append(f"designer: style reference failed ({exc}); frames use text-only style")
            style_ref_path = None

    style_ref = store.load_bytes(style_ref_path) if style_ref_path else None
    # return_exceptions: on a pause, the other branch still drains its in-flight images
    # (they're billed either way) before the stage stops.
    built = await asyncio.gather(
        _build_cast(script, shotlist, style_ref, deps), _build_locations(shotlist, style_ref, deps),
        return_exceptions=True,
    )
    for result in built:
        if isinstance(result, BaseException) and not isinstance(result, JobPaused):
            raise result
    if pauses := [r for r in built if isinstance(r, JobPaused)]:
        raise JobPaused("designer", "; ".join(p.message for p in pauses))
    (cast_paths, cast_warnings), (location_paths, loc_warnings) = built

    return {
        "shotlist_path": shotlist_path,
        "style_bible_path": style_bible_path,
        "style_ref_path": style_ref_path,
        "cast_paths": cast_paths,
        "location_paths": location_paths,
        "warnings": warnings + cast_warnings + loc_warnings,
    }


def _sync_shots(shotlist: Shotlist, script: Script) -> None:
    """Shots carry the script's narration + visible characters (the model may drop them)."""
    by_order = {s.order: s for s in script.scenes}
    for shot in shotlist.shots:
        scene = by_order.get(shot.order)
        if scene is None:
            continue
        shot.narration = scene.narration
        shot.location = scene.location or shot.location
        shot.characters = [c.upper() for c in (shot.characters or scene.characters)]
