"""Narrative judge: score a Script on story cohesion BEFORE any media is generated.

Reused by TWO callers:
  - the eval harness (``evals/run.py``) — MEASURE how good the prompts are.
  - the reflection node (``app/agents/reflection.py``) — the runtime guard that
    critiques->revises a weak script until it clears a threshold.

Design choices that keep the score meaningful:
  - Five 0-100 dimensions, weighted toward the user's actual complaint (structure +
    continuity). The canonical ``overall`` is computed HERE as a weighted mean, never
    read from the model — so a threshold means the same thing in eval and reflection,
    and the model can't fudge its own arithmetic.
  - Temperature 0 (deterministic-as-possible scoring) when the adapter forwards it.
  - A deterministic ``mock_factory`` so the whole thing runs offline with no API key.
"""

from __future__ import annotations

from collections.abc import Callable

from app.adapters.llm import LLMAdapter
from app.core.config import Settings, get_settings
from app.prompts import render
from app.schemas import JudgeReport, Script, Shotlist

# Weights sum to 1.0. Structure + continuity dominate: they ARE the "no start/mid/end"
# complaint. visual_concreteness matters least for STORY cohesion (the designer + image
# model carry the look).
DIMENSION_WEIGHTS: dict[str, float] = {
    "structure": 0.30,
    "continuity": 0.25,
    "clarity": 0.20,
    "payoff": 0.15,
    "visual_concreteness": 0.10,
}

DIALOGUE_WEIGHT = 0.12  # only when the script has dialogue; the other weights scale down


def _system(has_dialogue: bool) -> str:
    base = render("judge/system")
    return base + "\n\n" + render("judge/dialogue_dimension") if has_dialogue else base


def compute_overall(report: JudgeReport, has_dialogue: bool = False) -> int:
    """Canonical weighted mean (0-100); never trust the model's arithmetic."""
    base = sum(getattr(report, dim) * w for dim, w in DIMENSION_WEIGHTS.items())
    if not has_dialogue:
        return round(base)
    return round(base * (1 - DIALOGUE_WEIGHT) + report.dialogue * DIALOGUE_WEIGHT)


def _format_target(script: Script, shotlist: Shotlist | None) -> str:
    parts = ["<script_json>", script.model_dump_json(indent=2), "</script_json>"]
    if shotlist is not None:
        parts += ["<shotlist_json>", shotlist.model_dump_json(indent=2), "</shotlist_json>"]
    return "\n".join(parts)


def _mock_factory(score: int) -> Callable[[str], JudgeReport]:
    """Deterministic offline judge: every dimension == ``score`` so mock runs are
    reproducible and the reflection loop's branches are testable end-to-end."""

    def factory(_user: str) -> JudgeReport:
        return JudgeReport(
            structure=score,
            continuity=score,
            clarity=score,
            visual_concreteness=score,
            payoff=score,
            dialogue=score,
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

    def score(self, script: Script, shotlist: Shotlist | None = None) -> JudgeReport:
        has_dialogue = any(sc.mode == "dialogue" and sc.lines for sc in script.scenes)
        report = self.llm.complete_json(
            system=_system(has_dialogue),
            user=_format_target(script, shotlist),
            response_model=JudgeReport,
            mock_factory=_mock_factory(self.settings.mock_quality_score),
            role="judge",
        )
        report.overall = compute_overall(report, has_dialogue)
        return report
