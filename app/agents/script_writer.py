"""script_writer agent: prompt -> structured Script (cast + scenes, per story style/language)."""

from __future__ import annotations

import asyncio

from app.core.logging import get_logger
from app.graph.state import Deps, StoryboardState
from app.prompts import render
from app.schemas import LANGUAGE_NAMES, Character, DialogueLine, Scene, Script

log = get_logger(__name__)

BEAT_ROLES = ("setup", "inciting", "rising", "climax", "resolution")
STYLES = ("narrated", "dialogue", "mixed")
# Speakable words per second; Tamil words are longer (agglutinative).
WORDS_PER_SECOND = {"en": 2.6, "ta": 1.8}


def _system_prompt(n: int, style: str = "narrated", language: str = "en", seconds_per_scene: float = 4.0) -> str:
    style = style if style in STYLES else "narrated"
    language = language if language in WORDS_PER_SECOND else "en"
    wps = WORDS_PER_SECOND[language]
    has_dialogue = style in ("dialogue", "mixed")
    return render(
        "script_writer/system",
        n=n,
        style_block=render(f"script_writer/style_{style}"),
        language_block=render(f"script_writer/language_{language}"),
        beat_roles=", ".join(BEAT_ROLES),
        narr_words=max(6, round(seconds_per_scene * wps * 0.85)),
        dialogue_rules=render(
            "script_writer/dialogue_rules",
            wps=wps,
            scene_seconds=round(seconds_per_scene),
            scene_words=max(5, round((seconds_per_scene - 1.0) * wps)),
        )
        + "\n"
        if has_dialogue
        else "",
    )


def _beat_roles(n: int) -> list[str]:
    """setup -> inciting -> rising... -> climax -> resolution, scaled to n beats."""
    if n <= 1:
        return ["setup"]
    roles: list[str] = []
    for i in range(n):
        frac = i / (n - 1)
        if i == 0:
            roles.append("setup")
        elif i == 1 and n >= 5:
            roles.append("inciting")
        elif frac >= 0.88:
            roles.append("resolution")
        elif frac >= 0.72:
            roles.append("climax")
        else:
            roles.append("rising")
    return roles


def scene_modes(n: int, style: str) -> list[str]:
    """Default per-scene modes. Mixed: dialogue on the inciting beat and the climax."""
    if style == "dialogue":
        return ["dialogue"] * n
    if style != "mixed" or n < 2:
        return ["narration"] * n
    modes = ["dialogue" if r in ("inciting", "climax") else "narration" for r in _beat_roles(n)]
    if "dialogue" not in modes:
        modes[max(0, n - 2)] = "dialogue"
    return modes


_MOCK_CAST = {
    "en": [
        Character(name="MAYA", look="woman in her sixties, silver braid, navy sweater (mock)",
                  voice_description="warm weathered alto, gentle Scottish accent (mock)"),
        Character(name="LEO", look="teenage boy, curly dark hair, red rain jacket (mock)",
                  voice_description="bright, slightly hoarse teenage voice (mock)"),
    ],
    "ta": [
        Character(name="KAMALA", display_name="கமலா", look="Tamil woman in her sixties, maroon saree (mock)",
                  voice_description="warm alto, speaks Tamil (mock)"),
        Character(name="ARUN", display_name="அருண்", look="Tamil teenage boy, blue rain jacket (mock)",
                  voice_description="bright teenage voice, speaks Tamil (mock)"),
    ],
}


def _mock_script(prompt: str, n: int, style: str = "narrated", language: str = "en") -> Script:
    """Deterministic offline script so the pipeline runs without an API key."""
    roles, modes = _beat_roles(n), scene_modes(n, style)
    cast = _MOCK_CAST.get(language, _MOCK_CAST["en"])
    a, b = cast[0].name, cast[1].name
    scenes = []
    for i in range(1, n + 1):
        dialogue = modes[i - 1] == "dialogue"
        scenes.append(
            Scene(
                order=i,
                beat_role=roles[i - 1],
                mode=modes[i - 1],
                description=f"Scene {i}: {prompt} — beat {i} of {n}.",
                characters=[a, b] if dialogue else [a],
                narration="" if dialogue else f"Beat {i}: {prompt[:60]}",
                lines=[
                    DialogueLine(speaker=a, text=f"Line {i}a about {prompt[:30]}.", delivery="curious"),
                    DialogueLine(speaker=b, text=f"Line {i}b in reply.", delivery="quiet"),
                ]
                if dialogue
                else [],
            )
        )
    return Script(
        title=f"Mock: {prompt[:40]}",
        logline=prompt,
        story_style=style,
        language=language,
        style="clean cinematic, consistent palette (mock)",
        arc=(
            f"Beginning: we meet {prompt[:48]} and what they want. Middle: it develops "
            "and complicates beat by beat. End: it resolves with a clear payoff."
        ),
        characters=list(cast),
        scenes=scenes,
    )


def normalize_script(script: Script, *, style: str, language: str, n: int) -> Script:
    """Enforce pipeline invariants on model output: requested style/language, modes that
    match the style, <=2 known speakers per dialogue scene, orders 1..n."""
    script.story_style, script.language = style, language
    known = {c.name.strip().upper() for c in script.characters}
    scenes = sorted(script.scenes, key=lambda x: x.order)[: n or None]
    for idx, sc in enumerate(scenes, start=1):
        sc.order = idx
        sc.characters = [c.strip().upper() for c in sc.characters if c.strip()]
        if style in ("narrated", "dialogue"):
            sc.mode = "narration" if style == "narrated" else "dialogue"
        for line in sc.lines:
            line.speaker = line.speaker.strip().upper()
        if sc.mode == "dialogue":
            speakers = list(dict.fromkeys(l.speaker for l in sc.lines))[:2]
            sc.lines = [l for l in sc.lines if l.speaker in speakers and l.text.strip()]
            for spk in speakers:
                if spk not in known:
                    script.characters.append(Character(name=spk, look=""))
                    known.add(spk)
                if spk not in sc.characters:
                    sc.characters.append(spk)
            if not sc.lines:  # nothing said -> narrate the beat instead
                sc.mode = "narration"
                sc.narration = sc.narration or sc.description
        else:
            if not sc.narration.strip() and sc.lines:
                sc.narration = " ".join(l.text for l in sc.lines)
            sc.lines = []
    script.scenes = scenes
    return script


async def script_writer(state: StoryboardState, deps: Deps) -> StoryboardState:
    # Resume: reuse the script; a re-run would diverge from frames already built.
    if deps.store.exists("script.json"):
        log.info("reusing the existing script (resume)")
        return {"script_path": deps.store.abspath("script.json")}

    style = state.get("style") or deps.settings.default_style
    language = state.get("language") or deps.settings.default_language
    n = deps.frame_count(state["duration"], style)
    seconds = state["duration"] / max(1, n)
    lang_name = LANGUAGE_NAMES.get(language, language)
    log.info("writing a %d-beat %s script in %s: %r", n, style, lang_name, state["prompt"][:80])
    script: Script = await asyncio.to_thread(
        deps.llm.complete_json,
        system=_system_prompt(n, style, language, seconds),
        user=render(
            "script_writer/user",
            idea=state["prompt"], duration=state["duration"], n=n,
            seconds=round(seconds), style=style, language=lang_name,
        ),
        response_model=Script,
        role="script",
        mock_factory=lambda _u: _mock_script(state["prompt"], n, style, language),
    )
    script = normalize_script(script, style=style, language=language, n=n)
    path = deps.store.save_json("script.json", script)
    log.info(
        "script ready: %r — %d scenes (%d dialogue), cast: %s",
        script.title, len(script.scenes), sum(sc.mode == "dialogue" for sc in script.scenes),
        ", ".join(c.name for c in script.characters) or "none",
    )
    return {"script_path": path}
