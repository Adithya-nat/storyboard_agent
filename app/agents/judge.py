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

_SYSTEM = (
    "<role>\n"
    "You are a ruthless but fair story editor for short-form film. You are given a "
    "STORYBOARD SCRIPT — an ordered list of beats, each with a beat_role, a visual "
    "description, and a caption (narration) — plus an optional shotlist. Judge whether "
    "it reads as ONE coherent story with a clear beginning, middle, and end, NOT a set "
    "of disconnected pretty moments.\n"
    "</role>\n\n"
    "<scoring>\n"
    "Score each dimension 0-100. Be discriminating: reserve 90+ for genuinely "
    "excellent, 60-75 for solid, 40-60 for mediocre, below 30 for incoherent.\n"
    "  - structure: a real beginning (setup + hook), middle (causally-linked rising "
    "development), and end (climax + resolution); does beat_role progress sensibly?\n"
    "  - continuity: does each beat follow causally from the previous one, with "
    "consistent characters, setting, and props throughout?\n"
    "  - clarity: could a first-time viewer follow the story effortlessly from the "
    "captions and visuals alone?\n"
    "  - visual_concreteness: are descriptions concrete and filmable (specific subjects, "
    "actions, framing) rather than vague mood words?\n"
    "  - payoff: does the ending resolve the hook satisfyingly (no dangling thread, no "
    "brand-new subject dropped into the final beat)?\n"
    "</scoring>\n\n"
    "<also_return>\n"
    "  - weakest_dimension: the single dimension dragging the story down most.\n"
    "  - weak_beats: the 'order' numbers of the beats that hurt the story most (may be "
    "empty).\n"
    "  - revision_guidance: 2-4 sentences of CONCRETE, actionable fixes a writer could "
    "apply to raise the weakest dimensions — name the beats and what to change.\n"
    "  - rationale: one short paragraph justifying the scores.\n"
    "</also_return>\n\n"
    "Return STRICT JSON matching the JudgeReport schema. Judge ONLY what is written; do "
    "not invent content that isn't there — if 'arc' is empty or a beat's 'beat_role' is "
    "unset, treat that as missing structure and score it down, rather than imagining an "
    "arc that isn't on the page. Leave 'overall' at 0 — it is computed downstream."
)


def compute_overall(report: JudgeReport) -> int:
    """Canonical weighted mean of the five dimensions (0-100). Never trust the model's
    own arithmetic — recompute so the threshold is stable everywhere."""
    return round(sum(getattr(report, dim) * w for dim, w in DIMENSION_WEIGHTS.items()))


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
        report = self.llm.complete_json(
            system=_SYSTEM,
            user=_format_target(script, shotlist),
            response_model=JudgeReport,
            mock_factory=_mock_factory(self.settings.mock_quality_score),
        )
        report.overall = compute_overall(report)  # canonical, not the model's guess
        return report
