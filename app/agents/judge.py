"""Narrative judge: scores a screenplay (vs its creative brief) before any media is made.

Shared by the eval harness and the reflection node. ``overall`` is a weighted mean computed
here, never read from the model, so a threshold means the same thing everywhere.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from app.adapters.llm import LLMAdapter
from app.core.config import Settings, get_settings
from app.prompts import render
from app.schemas import CreativeBrief, JudgeReport, Script, Shotlist

# Weights sum to 1.0. Structure + continuity + hook carry the "random clips" complaint;
# visual_concreteness matters least for story cohesion.
DIMENSION_WEIGHTS: dict[str, float] = {
    "structure": 0.20,
    "continuity": 0.20,
    "clarity": 0.15,
    "hook": 0.13,
    "brief_adherence": 0.12,
    "payoff": 0.12,
    "visual_concreteness": 0.08,
}
# Mixed in only when they apply; the base weights scale down by their total.
DIALOGUE_WEIGHTS: dict[str, float] = {"dialogue": 0.10, "dialogue_thread": 0.10}
REGISTER_WEIGHT = 0.08  # whenever anything is spoken
DIALOGUE_WEIGHT = sum(DIALOGUE_WEIGHTS.values())


def _system(has_dialogue: bool) -> str:
    base = render("judge/system")
    return base + "\n\n" + render("judge/dialogue_dimension") if has_dialogue else base


def compute_overall(report: JudgeReport, has_dialogue: bool = False, has_speech: bool | None = None) -> int:
    """Canonical weighted mean (0-100); never trust the model's arithmetic."""
    extra = dict(DIALOGUE_WEIGHTS) if has_dialogue else {}
    if has_speech if has_speech is not None else has_dialogue:
        extra["register_fit"] = REGISTER_WEIGHT
    base = sum(getattr(report, dim) * w for dim, w in DIMENSION_WEIGHTS.items())
    mixed = sum(getattr(report, dim) * w for dim, w in extra.items())
    return round(base * (1 - sum(extra.values())) + mixed)


def _format_target(script: Script, shotlist: Shotlist | None, brief: CreativeBrief | None = None) -> str:
    parts = []
    if brief is not None:
        parts += ["<creative_brief>", brief.model_dump_json(indent=2), "</creative_brief>"]
    data = script.model_dump()
    for sc in data["scenes"]:
        if any(sh["lines"] for sh in sc["shots"]):
            sc.pop("lines")  # the shots already carry them
    parts += ["<script_json>", json.dumps(data, indent=2, ensure_ascii=False), "</script_json>"]
    if shotlist is not None:
        parts += ["<shotlist_json>", shotlist.model_dump_json(indent=2), "</shotlist_json>"]
    return "\n".join(parts)


def _mock_factory(score: int) -> Callable[[str], JudgeReport]:
    """Deterministic offline judge: every dimension == ``score`` so mock runs are
    reproducible and the reflection loop's branches are testable end-to-end."""

    def factory(_user: str) -> JudgeReport:
        dims = {d: score for d in (*DIMENSION_WEIGHTS, *DIALOGUE_WEIGHTS, "register_fit")}
        return JudgeReport(
            **dims,
            retelling="(mock) the story as written.",
            weakest_dimension="structure",
            revision_guidance="(mock) tighten the causal links between beats.",
            rationale="(mock judge)",
        )

    return factory


class NarrativeJudge:
    """Scores a Script (optionally with its Shotlist) for story cohesion."""

    def __init__(self, llm: LLMAdapter | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or LLMAdapter(self.settings)

    def score(self, script: Script, shotlist: Shotlist | None = None, brief: CreativeBrief | None = None) -> JudgeReport:
        has_dialogue = any(sc.mode == "dialogue" and sc.lines for sc in script.scenes)
        has_speech = has_dialogue or any(sc.narration.strip() for sc in script.scenes)
        report = self.llm.complete_json(
            system=_system(has_dialogue),
            user=_format_target(script, shotlist, brief),
            response_model=JudgeReport,
            mock_factory=_mock_factory(self.settings.mock_quality_score),
            role="judge",
        )
        report.overall = compute_overall(report, has_dialogue, has_speech)
        return report
