"""script_writer agent: prompt -> structured Script (cast + scenes, per story style/language)."""

from __future__ import annotations

import asyncio

from app.core.logging import get_logger
from app.agents.brief import mock_brief, normalize_brief, speech_block
from app.agents.redo_director import direction
from app.graph.state import SCENE_MAX_SECONDS, Deps, StoryboardState
from app.prompts import render
from app.schemas import LANGUAGE_NAMES, Character, CreativeBrief, DialogueLine, Scene, SceneShot, Script

log = get_logger(__name__)

BEAT_ROLES = ("setup", "inciting", "rising", "climax", "resolution")
STYLES = ("narrated", "dialogue", "mixed")
# Speakable words per second; Tamil words are longer (agglutinative).
WORDS_PER_SECOND = {"en": 2.6, "ta": 1.8}


def word_budget(seconds: float, language: str, mode: str) -> int:
    """Max spoken words for a scene: dialogue leaves ~1.5s of air, narration ~15%."""
    wps = WORDS_PER_SECOND.get(language, 2.6)
    if mode == "dialogue":
        return max(4, round((seconds - 1.5) * wps * 0.9))
    return max(6, round(seconds * wps * 0.85))


def _system_prompt(
    n: int, style: str = "narrated", language: str = "en", seconds_per_scene: float = 10.0,
    brief: CreativeBrief | None = None,
) -> str:
    style = style if style in STYLES else "narrated"
    language = language if language in WORDS_PER_SECOND else "en"
    plan = [(s.seconds, s.mode) for s in brief.scenes] if brief else [(seconds_per_scene, "dialogue" if style != "narrated" else "narration")] * n
    budgets = "\n".join(
        f"  - scene {i}: {sec:g}s, {mode} -> at most {word_budget(sec, language, mode)} words" for i, (sec, mode) in enumerate(plan, 1)
    )
    has_dialogue = style in ("dialogue", "mixed")
    return render(
        "script_writer/system",
        n=n,
        style_block=render(f"script_writer/style_{style}"),
        language_block=render(f"script_writer/language_{language}"),
        register_block=speech_block(brief, language),
        beat_roles=", ".join(BEAT_ROLES),
        budgets=budgets,
        dialogue_rules=render("script_writer/dialogue_rules", wps=WORDS_PER_SECOND[language]) + "\n" if has_dialogue else "",
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


def _mock_script(
    prompt: str, n: int, style: str = "narrated", language: str = "en", modes: list[str] | None = None,
    seconds: list[float] | None = None,
) -> Script:
    """Deterministic offline script so the pipeline runs without an API key."""
    roles, modes = _beat_roles(n), modes or scene_modes(n, style)
    cast = _MOCK_CAST.get(language, _MOCK_CAST["en"])
    a, b = cast[0].name, cast[1].name
    scenes = []
    for i in range(1, n + 1):
        dialogue = modes[i - 1] == "dialogue"
        secs = (seconds or [])[i - 1] if seconds and i <= len(seconds) else 8.0
        lines = [
            DialogueLine(speaker=a, text=f"Line {i}a about {prompt[:30]}.", delivery="curious"),
            DialogueLine(speaker=b, text=f"Line {i}b in reply.", delivery="quiet"),
        ] if dialogue else []
        shots = [
            SceneShot(size="wide", action=f"{a} and {b} in the main location (mock)", seconds=secs * 0.4, lines=lines[:1]),
            SceneShot(size="close", action=f"{b} reacts (mock)", seconds=secs * 0.35, lines=lines[1:]),
            SceneShot(size="insert", action="a key prop (mock)", seconds=secs * 0.25),
        ]
        scenes.append(
            Scene(
                order=i,
                beat_role=roles[i - 1],
                mode=modes[i - 1],
                location="MAIN_LOCATION",
                description=f"Scene {i}: {prompt} — beat {i} of {n}.",
                seconds=secs,
                shots=shots,
                characters=[a, b] if dialogue else [a],
                narration="" if dialogue else f"Beat {i}: {prompt[:60]}",
                lines=list(lines),
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


def _fit_shots(sc: Scene) -> None:
    """Every scene has >= 1 shot whose seconds add up to the scene's seconds."""
    if not sc.shots:
        sc.shots = [SceneShot(size="medium", action=sc.description, seconds=sc.seconds or 4.0, lines=list(sc.lines))]
    if not sc.seconds:
        sc.seconds = sum(max(0.5, sh.seconds) for sh in sc.shots)
    sc.seconds = round(min(SCENE_MAX_SECONDS, sc.seconds), 1)
    total = sum(max(0.5, sh.seconds) for sh in sc.shots)
    for sh in sc.shots:
        sh.seconds = round(max(0.5, sh.seconds) * sc.seconds / total, 2)


def normalize_script(
    script: Script, *, style: str, language: str, n: int, brief: CreativeBrief | None = None,
) -> Script:
    """Enforce pipeline invariants on model output: requested style/language, modes that
    match the style (or the brief's plan), <=2 known speakers per dialogue scene, shots that
    fill each scene, scene.lines = the shots' lines, orders 1..n."""
    script.story_style, script.language = style, language
    known = {c.name.strip().upper() for c in script.characters}
    plan = {p.order: p for p in brief.scenes} if brief else {}
    scenes = sorted(script.scenes, key=lambda x: x.order)[: n or None]
    for idx, sc in enumerate(scenes, start=1):
        sc.order = idx
        if idx in plan:
            sc.seconds = plan[idx].seconds or sc.seconds
            sc.location = sc.location or plan[idx].location
        sc.characters = [c.strip().upper() for c in sc.characters if c.strip()]
        if style in ("narrated", "dialogue"):
            sc.mode = "narration" if style == "narrated" else "dialogue"
        elif idx in plan:
            sc.mode = plan[idx].mode
        if not sc.shots and sc.lines:  # older/flat output: one shot carries the lines
            sc.shots = [SceneShot(size="medium", action=sc.description, seconds=sc.seconds or 4.0, lines=list(sc.lines))]
        lines = sc.shot_lines() if sc.shots else list(sc.lines)
        for line in lines:
            line.speaker = line.speaker.strip().upper()
        if sc.mode == "dialogue":
            speakers = list(dict.fromkeys(l.speaker for l in lines if l.text.strip()))[:2]
            for shot in sc.shots:
                shot.lines = [l for l in shot.lines if l.speaker in speakers and l.text.strip()]
            for spk in speakers:
                if spk not in known:
                    script.characters.append(Character(name=spk, look=""))
                    known.add(spk)
                if spk not in sc.characters:
                    sc.characters.append(spk)
            sc.lines = sc.shot_lines() if sc.shots else [l for l in lines if l.speaker in speakers and l.text.strip()]
            sc.narration = ""
            if not sc.lines:  # nothing said -> narrate the scene instead
                sc.mode = "narration"
                sc.narration = sc.description
        else:
            if not sc.narration.strip() and lines:
                sc.narration = " ".join(l.text for l in lines)
            sc.lines = []
            for shot in sc.shots:
                shot.lines = []
        _fit_shots(sc)
    script.scenes = scenes
    return script


def load_brief(state: StoryboardState, deps: Deps, style: str, n: int) -> CreativeBrief:
    """The run's creative brief; a default plan when called without the brief stage."""
    if deps.store.exists("brief.json"):
        return CreativeBrief.model_validate(deps.store.load_json(deps.store.abspath("brief.json")))
    duration = state.get("duration") or 30
    b = mock_brief(state.get("prompt", ""), n, duration, style)
    return normalize_brief(b, duration=duration, style=style, n_override=n)


async def script_writer(state: StoryboardState, deps: Deps) -> StoryboardState:
    # Resume: reuse the script; a re-run would diverge from frames already built.
    if deps.store.exists("script.json"):
        log.info("reusing the existing script (resume)")
        return {"script_path": deps.store.abspath("script.json")}

    style = state.get("style") or deps.settings.default_style
    language = state.get("language") or deps.settings.default_language
    brief = load_brief(state, deps, style, deps.frame_count(state["duration"], style))
    n = len(brief.scenes)
    lang_name = LANGUAGE_NAMES.get(language, language)
    log.info("writing a %d-scene %s screenplay in %s: %r", n, style, lang_name, state["prompt"][:80])
    script: Script = await asyncio.to_thread(
        deps.llm.complete_json,
        system=_system_prompt(n, style, language, brief=brief),
        user=render(
            "script_writer/user",
            idea=state["prompt"], brief_json=brief.model_dump_json(indent=2), duration=state["duration"],
            n=n, style=style, language=lang_name, direction=direction(deps.store, "story"),
        ),
        response_model=Script,
        role="script",
        mock_factory=lambda _u: _mock_script(
            state["prompt"], n, style, language, [p.mode for p in brief.scenes], [p.seconds for p in brief.scenes]
        ),
    )
    script = normalize_script(script, style=style, language=language, n=n, brief=brief)
    path = deps.store.save_json("script.json", script)
    log.info(
        "screenplay ready: %r — %d scenes, %d shots, %d lines; cast: %s",
        script.title, len(script.scenes), sum(len(sc.shots) for sc in script.scenes),
        sum(len(sc.lines) for sc in script.scenes), ", ".join(c.name for c in script.characters) or "none",
    )
    return {"script_path": path}
