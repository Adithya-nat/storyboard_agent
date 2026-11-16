"""brief agent: raw idea -> CreativeBrief (format, hook, speech register, scene plan)."""

from __future__ import annotations

import asyncio

from app.adapters.errors import ProviderError
from app.core.logging import get_logger
from app.graph.state import SCENE_MAX_SECONDS, SCENE_MIN_SECONDS, Deps, StoryboardState, scene_range
from app.prompts import render
from app.schemas import LANGUAGE_NAMES, CreativeBrief, PlannedScene, SpeechRegister

log = get_logger(__name__)

_STYLE_RULE = {
    "dialogue": "Every scene is mode 'dialogue': characters speak on camera (silent shots inside a scene are fine; a monologue counts).",
    "narrated": "Every scene is mode 'narration': a narrator speaks over the pictures; characters are seen, not heard.",
    "mixed": "Use mode 'narration' to open and close the story and mode 'dialogue' for the key confrontations (about half the scenes).",
}
_PURPOSES = ("hook", "setup", "complication", "turn", "climax", "payoff")


def speech_block(brief: CreativeBrief | None, language: str) -> str:
    """The <speech_register> guide for this film's language and era."""
    sp = brief.speech if brief else SpeechRegister()
    lang = language if language in ("en", "ta") else "en"
    return render(
        f"register/{lang}_{sp.era if sp.era in ('modern', 'period') else 'modern'}",
        period=sp.period or "present day", dialect=sp.dialect or "everyday conversational",
        slang=sp.slang or "as fits each character", avoid=sp.avoid or "nothing specific",
    )


def default_modes(n: int, style: str) -> list[str]:
    if style == "narrated":
        return ["narration"] * n
    if style != "mixed" or n < 3:
        return ["dialogue"] * n
    return ["narration"] + ["dialogue"] * (n - 2) + ["narration"]


def normalize_brief(brief: CreativeBrief, *, duration: int, style: str, n_override: int | None = None) -> CreativeBrief:
    """Scene count within range, 5-15s each, seconds summing to the runtime, modes per style."""
    lo, hi = scene_range(duration)
    if n_override is not None:
        lo = hi = n_override
    scenes = sorted(brief.scenes, key=lambda s: s.order)[:hi] or [PlannedScene(order=1, purpose="setup", seconds=duration)]
    while len(scenes) < lo:  # split the longest scene until the count is legal
        i = max(range(len(scenes)), key=lambda k: scenes[k].seconds)
        half = scenes[i].model_copy(update={"seconds": scenes[i].seconds / 2})
        scenes[i:i + 1] = [half, half.model_copy(update={"purpose": f"{half.purpose} (continued)"})]
    total = sum(max(0.1, s.seconds) for s in scenes)
    lo_s, hi_s = (SCENE_MIN_SECONDS, SCENE_MAX_SECONDS) if n_override is None else (1.0, SCENE_MAX_SECONDS)
    modes = default_modes(len(scenes), style)
    for i, s in enumerate(scenes):
        s.order = i + 1
        s.seconds = round(min(hi_s, max(lo_s, max(0.1, s.seconds) * duration / total)), 1)
        if style != "mixed" or s.mode not in ("dialogue", "narration"):
            s.mode = modes[i]
    if style == "mixed" and len(scenes) >= 2 and len({s.mode for s in scenes}) == 1:
        for s, m in zip(scenes, modes):
            s.mode = m
    brief.scenes = scenes
    return brief


def mock_brief(prompt: str, n: int, duration: int, style: str) -> CreativeBrief:
    modes = default_modes(n, style)
    return CreativeBrief(
        format="short drama",
        purpose="(mock) a small, warm story with a clear payoff",
        audience="general",
        hook=f"(mock) a striking first image: {prompt[:60]}",
        logline=f"(mock) {prompt[:120]}",
        setting="(mock) present day",
        speech=SpeechRegister(era="modern", period="present day", dialect="everyday conversational", slang="light"),
        scenes=[
            PlannedScene(order=i + 1, purpose=_PURPOSES[min(i, len(_PURPOSES) - 1)], location="MAIN_LOCATION",
                         seconds=duration / n, mode=modes[i])
            for i in range(n)
        ],
    )


async def brief(state: StoryboardState, deps: Deps) -> StoryboardState:
    store = deps.store
    if store.exists("brief.json"):
        log.info("reusing the existing creative brief (resume)")
        return {"brief_path": store.abspath("brief.json")}
    style = state.get("style") or deps.settings.default_style
    language = state.get("language") or deps.settings.default_language
    duration = state["duration"]
    n = deps.frame_count(duration, style)
    lo, hi = scene_range(duration) if deps.max_frames is None else (n, n)
    lang_name = LANGUAGE_NAMES.get(language, language)
    warnings: list[str] = []
    try:
        result: CreativeBrief = await asyncio.to_thread(
            deps.llm.complete_json,
            system=render("brief/system", duration=duration, style_rule=_STYLE_RULE.get(style, _STYLE_RULE["dialogue"]),
                          lo=lo, hi=hi, n=n, language=lang_name),
            user=render("brief/user", idea=state["prompt"], duration=duration, style=style, language=lang_name),
            response_model=CreativeBrief,
            role="script",
            mock_factory=lambda _u: mock_brief(state["prompt"], n, duration, style),
        )
    except ProviderError as exc:
        log.warning("creative brief failed (%s); using a default plan", exc)
        warnings.append(f"brief: planning failed ({exc}); used a default scene plan")
        result = mock_brief(state["prompt"], n, duration, style)
    result = normalize_brief(result, duration=duration, style=style, n_override=deps.max_frames)
    path = store.save_json("brief.json", result)
    sp = result.speech
    log.info(
        "brief: %s — %d scenes (%s); %s speech, %s; hook: %s",
        result.format, len(result.scenes), ", ".join(f"{s.seconds:g}s" for s in result.scenes),
        sp.era, sp.dialect or sp.period, result.hook[:100],
    )
    return {"brief_path": path, "warnings": warnings}
