"""designer agent: Script -> Shotlist + style bible + ONE style reference image.

This is the #1 anti-"AI slop" lever. Independent per-frame generations drift in
character, palette, and composition. The designer locks the look ONCE:

    script.json
        │
        ▼  (LLM) art-direct: write a detailed style_bible + per-shot image_prompts
    shotlist.json  +  style_bible.md
        │
        ▼  (image) render ONE style_ref.png from the bible
    style_ref.png  ──▶ image_gen conditions EVERY frame on it (ref_image=)

Degrade, don't die (Phase 2 decision): the style ref enhances consistency but is
not worth killing a multi-minute job over. If the shotlist LLM fails we fall back
to the raw script scenes; if the style-ref image fails we drop to text-only style
conditioning. Both paths surface a warning — never silent.

Idempotent (resume): every artifact is load-or-produce, so re-entering this node
reuses the existing shotlist/bible/ref instead of regenerating a divergent one.
"""

from __future__ import annotations

import asyncio

from app.adapters.errors import ProviderError
from app.core.logging import get_logger
from app.graph.state import Deps, StoryboardState
from app.schemas import Script, Shot, Shotlist

log = get_logger(__name__)

_SYSTEM = (
    "<role>\n"
    "You are a meticulous art director and cinematographer for short-form film. You are "
    "handed a finished script — a STORY with an arc (a beginning, middle, and end). Your "
    "job is to lock ONE consistent visual world so the finished film reads as a single "
    "coherent story, never a set of unrelated images.\n"
    "</role>\n\n"
    "<task>\n"
    "Produce STRICT JSON matching the Shotlist schema:\n"
    "(1) 'style_bible' — a detailed, concrete set of visual rules EVERY frame must obey: "
    "medium & render style, colour palette, lighting, lens/composition language, and the "
    "EXACT recurring look of each named character and setting (so they stay identical "
    "shot to shot). Be specific and reusable — this is the anti-drift contract.\n"
    "(2) 'shots' — EXACTLY one shot per script scene, preserving each scene's 'order' and "
    "'narration'. Each 'image_prompt' is rendered ALONE by the image model (one prompt at "
    "a time, with no memory of the other shots), so it must be SELF-CONTAINED: restate "
    "that scene visually AND bake in the style-bible rules — including each subject's "
    "exact appearance — so the shot generates stand-alone yet still matches the others. "
    "Use each scene's 'beat_role' in the arc to choose framing and energy: establish wide "
    "on setup beats, tighten and raise tension through rising beats, give the climax its "
    "most dramatic composition, and let the resolution land calmly.\n"
    "</task>\n\n"
    "<rules>\n"
    "DO:\n"
    "  - Hold characters, wardrobe, palette, and setting identical across every shot.\n"
    "  - Let composition track the story's rising and falling tension.\n"
    "  - Be concrete and filmable; describe what is IN FRAME, not a mood word.\n"
    "DON'T:\n"
    "  - Don't restyle between shots or redesign a character mid-story.\n"
    "  - Don't drop the narration or change the shot order.\n"
    "  - Don't use generic AI cliches or vague 'cinematic, beautiful' filler.\n"
    "</rules>\n\n"
    "Be specific, not generic."
)


def _mock_shotlist(script: Script) -> Shotlist:
    """Deterministic offline shotlist so the pipeline runs without an API key."""
    n = len(script.scenes)
    bible = (
        f"STYLE BIBLE (mock): {script.style or 'clean cinematic'}. Consistent "
        f"palette, lighting, and character design held identical across all {n} "
        f"shots. Title: {script.title}."
    )
    return Shotlist(
        style_bible=bible,
        shots=[
            Shot(
                order=s.order,
                image_prompt=f"{script.style}. {s.description}".strip(". "),
                narration=s.narration,
            )
            for s in script.scenes
        ],
    )


def _shotlist_from_script(script: Script) -> Shotlist:
    """Fallback when the designer LLM fails: derive a usable shotlist from the
    script directly (less enrichment, but the job still runs)."""
    bible = script.style or "Consistent cinematic style across all shots."
    return Shotlist(
        style_bible=bible,
        shots=[
            Shot(
                order=s.order,
                image_prompt=f"{script.style}. {s.description}".strip(". "),
                narration=s.narration,
            )
            for s in script.scenes
        ],
    )


def _ref_prompt(shotlist: Shotlist) -> str:
    """Prompt for the single style-reference image that anchors every frame."""
    establishing = shotlist.shots[0].image_prompt if shotlist.shots else ""
    return (
        "A single STYLE & CHARACTER reference frame (a 'model sheet' still) that locks the "
        "look of an entire short film: the exact colour palette, lighting, render style, and "
        "character/setting design every later frame must match. Show the main character(s) "
        "clearly and in-style, and render the establishing shot in this look. Clean and "
        "cohesive, with no text, captions, or watermarks.\n"
        f"STYLE BIBLE: {shotlist.style_bible}\n"
        f"ESTABLISHING SHOT: {establishing}"
    ).strip()


async def designer(state: StoryboardState, deps: Deps) -> StoryboardState:
    store = deps.store
    warnings: list[str] = []

    # 1) Shotlist + style bible (load-or-produce) ------------------------- #
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
        script = Script.model_validate(store.load_json(state["script_path"]))
        try:
            shotlist = await asyncio.to_thread(
                deps.llm.complete_json,
                system=_SYSTEM,
                user=(
                    "<script_json>\n"
                    f"{script.model_dump_json(indent=2)}\n"
                    "</script_json>\n\n"
                    f"Write a Shotlist with exactly {len(script.scenes)} shots "
                    f"(order {script.scenes[0].order if script.scenes else 1}..), "
                    "one per scene, preserving each scene's narration."
                ),
                response_model=Shotlist,
                mock_factory=lambda _u: _mock_shotlist(script),
            )
        except ProviderError as exc:
            log.warning("designer shotlist failed, deriving from script: %s", exc)
            warnings.append(
                f"designer: shotlist generation failed ({exc}); used script scenes directly"
            )
            shotlist = _shotlist_from_script(script)
        shotlist_path = store.save_json("shotlist.json", shotlist)
        style_bible_path = store.save_text("style_bible.md", shotlist.style_bible)
        log.info("locked the look: %d shots + a style bible", len(shotlist.shots))

    # 2) ONE style-reference image (load-or-produce; degrade, don't die) -- #
    if store.exists("style_ref.png"):
        log.info("reusing the existing style reference (resume)")
        style_ref_path: str | None = store.abspath("style_ref.png")
    else:
        try:
            ref = await asyncio.to_thread(
                deps.image.generate_image,
                _ref_prompt(shotlist),
                seed=deps.settings.image_seed,
            )
            style_ref_path = store.save_bytes("style_ref.png", ref)
            log.info("rendered the style-reference image that anchors every frame")
        except ProviderError as exc:
            log.warning("designer style_ref failed, falling back to text-only: %s", exc)
            warnings.append(
                f"designer: style reference failed ({exc}); frames use text-only style"
            )
            style_ref_path = None

    return {
        "shotlist_path": shotlist_path,
        "style_bible_path": style_bible_path,
        "style_ref_path": style_ref_path,
        "warnings": warnings,
    }
