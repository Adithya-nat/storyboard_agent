"""dialogue_pass agent: rewrite all spoken words of the film in one pass, in the brief's register.

One call sees the whole film, so lines answer each other and callbacks land. It never changes
speakers, structure or plot; a scene whose rewrite blows its word budget keeps its old lines.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from app.adapters.errors import ProviderError
from app.agents.brief import speech_block
from app.agents.script_writer import load_brief, word_budget
from app.core.logging import get_logger
from app.graph.state import Deps, StoryboardState
from app.prompts import render
from app.schemas import LANGUAGE_NAMES, CreativeBrief, Script

log = get_logger(__name__)

BUDGET_SLACK = 1.25  # a rewrite may run this much over the scene's word budget


class LineRewrite(BaseModel):
    id: str = Field(description="The id exactly as given, e.g. L2.1.1 or N3.")
    text: str = Field(description="The rewritten spoken words, in the film's language.")
    delivery: str = Field(default="", description="Short English acting note (lines only).")


class DialoguePassResult(BaseModel):
    notes: str = Field(default="", description="One sentence on what changed.")
    lines: list[LineRewrite] = Field(default_factory=list)
    narration: list[LineRewrite] = Field(default_factory=list)


def _words(text: str) -> int:
    return len(text.split())


def screenplay_listing(script: Script) -> str:
    out: list[str] = []
    for sc in script.scenes:
        budget = word_budget(sc.seconds or 8.0, script.language, sc.mode)
        out.append(f"SCENE {sc.order} ({sc.seconds:g}s, {sc.mode}, max {budget} words) at {sc.location or 'location'} — {sc.goal or sc.description}")
        if sc.mode == "narration":
            out.append(f"  [N{sc.order}] NARRATOR: \"{sc.narration}\"")
        for si, shot in enumerate(sc.shots, 1):
            out.append(f"  shot {si} [{shot.size}] {shot.action}")
            for li, line in enumerate(shot.lines, 1):
                out.append(f"    [L{sc.order}.{si}.{li}] {line.speaker} ({line.delivery}): \"{line.text}\"")
    return "\n".join(out)


def apply_pass(script: Script, result: DialoguePassResult) -> tuple[Script, list[int]]:
    """Apply rewrites by id; scenes that bust their budget keep their old words."""
    lines = {r.id.strip().upper(): r for r in result.lines}
    narr = {r.id.strip().upper(): r for r in result.narration}
    kept: list[int] = []
    for sc in script.scenes:
        budget = word_budget(sc.seconds or 8.0, script.language, sc.mode) * BUDGET_SLACK
        if sc.mode == "narration":
            new = narr.get(f"N{sc.order}")
            if new and new.text.strip():
                if _words(new.text) <= max(budget, _words(sc.narration)):
                    sc.narration = new.text.strip()
                else:
                    kept.append(sc.order)
            continue
        edits = {}
        for si, shot in enumerate(sc.shots, 1):
            for li, line in enumerate(shot.lines, 1):
                new = lines.get(f"L{sc.order}.{si}.{li}")
                if new and new.text.strip():
                    edits[(si, li)] = new
        old_words = sum(_words(l.text) for l in sc.shot_lines())
        new_words = sum(_words(edits[(si, li)].text) if (si, li) in edits else _words(l.text)
                        for si, shot in enumerate(sc.shots, 1) for li, l in enumerate(shot.lines, 1))
        if new_words > max(budget, old_words):
            kept.append(sc.order)
            continue
        for (si, li), new in edits.items():
            line = sc.shots[si - 1].lines[li - 1]
            line.text, line.delivery = new.text.strip(), new.delivery.strip() or line.delivery
        sc.lines = sc.shot_lines()
    return script, kept


def _cast(script: Script) -> str:
    return "\n".join(f"{c.name} ({c.display_name or c.name}): {c.voice_description} — {c.look}" for c in script.characters)


async def dialogue_pass(state: StoryboardState, deps: Deps) -> StoryboardState:
    store = deps.store
    if store.exists("dialogue_pass.json"):
        log.info("reusing the dialogue pass (resume)")
        return {}
    script = Script.model_validate(store.load_json(state["script_path"]))
    if not any(sc.narration.strip() or sc.lines for sc in script.scenes):
        if deps.reporter is not None:
            await deps.reporter.mark_skipped("nothing spoken")
        return {}
    style = state.get("style") or script.story_style
    brief: CreativeBrief = load_brief(state, deps, style, len(script.scenes))
    language = script.language
    try:
        result: DialoguePassResult = await asyncio.to_thread(
            deps.llm.complete_json,
            system=render("dialogue_pass/system", language=LANGUAGE_NAMES.get(language, language),
                          register_block=speech_block(brief, language)),
            user=render("dialogue_pass/user", brief=brief.model_dump_json(indent=2), cast=_cast(script),
                        screenplay=screenplay_listing(script)),
            response_model=DialoguePassResult,
            role="script",
            mock_factory=lambda _u: DialoguePassResult(notes="(mock) no changes"),
        )
    except ProviderError as exc:
        log.warning("dialogue pass failed (%s); keeping the writer's lines", exc)
        return {"warnings": [f"dialogue_pass: polish failed ({exc}); kept the original lines"]}
    script, kept = apply_pass(script, result)
    store.save_json("script.json", script)
    store.save_json("dialogue_pass.json", {**result.model_dump(), "kept_scenes": kept})
    log.info(
        "dialogue pass (%s %s): %d line(s), %d narration passage(s) rewritten%s — %s",
        brief.speech.era, brief.speech.dialect or language, len(result.lines), len(result.narration),
        f"; scene(s) {kept} kept (over budget)" if kept else "", result.notes[:160],
    )
    return {}
