"""W2: narrative judge + eval-harness quality wiring."""

from __future__ import annotations

from pathlib import Path

from app.agents.judge import (
    DIMENSION_WEIGHTS,
    NarrativeJudge,
    compute_overall,
)
from app.artifacts.store import ArtifactStore
from app.core.config import Settings, get_settings
from app.graph.state import Deps
from app.adapters.gemini import GeminiImageAdapter
from app.adapters.llm import LLMAdapter
from app.schemas import JudgeReport, Scene, Script
from evals.rubric_fixtures import BAD_SCRIPT, GOOD_SCRIPT
from evals.run import _quality


def _script(score_hint: str = "x") -> Script:
    return Script(
        title="t",
        logline="l",
        arc="a",
        scenes=[Scene(order=i, beat_role="rising", description=f"d{i}", narration=f"n{i}") for i in (1, 2, 3)],
    )


def test_dimension_weights_sum_to_one():
    assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-9


def test_judge_report_schema_parses_and_defaults():
    r = JudgeReport.model_validate({"structure": 70})
    assert r.structure == 70
    assert r.overall == 0  # computed later, not required from the model
    assert r.weak_beats == []


def test_compute_overall_is_weighted_mean():
    r = JudgeReport(structure=100, continuity=0, clarity=0, visual_concreteness=0, payoff=0)
    # only the structure weight (0.30) contributes
    assert compute_overall(r) == round(100 * DIMENSION_WEIGHTS["structure"])
    flat = JudgeReport(structure=80, continuity=80, clarity=80, visual_concreteness=80, payoff=80)
    assert compute_overall(flat) == 80


def test_narrative_judge_mock_returns_report():
    settings = Settings(use_mock_providers=True, mock_quality_score=77)
    report = NarrativeJudge(LLMAdapter(settings), settings).score(_script())
    # all dims == mock score -> weighted mean == score
    assert report.overall == 77
    assert report.structure == 77


class _StubLLM:
    """Returns a preset JudgeReport, ignoring the mock_factory — used to prove the
    judge recomputes `overall` canonically instead of trusting the model."""

    def __init__(self, report: JudgeReport):
        self.report = report
        self.calls = 0

    def complete_json(self, **kwargs):
        self.calls += 1
        return self.report


def test_overall_is_canonical_not_model_supplied():
    lying = JudgeReport(
        structure=10, continuity=10, clarity=10, visual_concreteness=10, payoff=10,
        overall=999,  # the model lies; the judge must overwrite this
        weakest_dimension="continuity",
        revision_guidance="fix beat 2",
    )
    judge = NarrativeJudge(_StubLLM(lying), get_settings())
    out = judge.score(_script())
    assert out.overall == 10  # recomputed weighted mean, NOT 999
    assert out.weakest_dimension == "continuity"
    assert out.revision_guidance == "fix beat 2"


def test_quality_helper_scores_existing_script(artifact_root):
    settings = get_settings()
    store = ArtifactStore(artifact_root, "judge-q")
    store.save_json("script.json", _script())
    deps = Deps(settings=settings, store=store, llm=LLMAdapter(settings), image=GeminiImageAdapter(settings))
    report, err = _quality(store, deps)
    assert err is None
    assert report is not None and report.overall == settings.mock_quality_score


def test_quality_helper_no_script_is_a_finding(artifact_root):
    settings = get_settings()
    store = ArtifactStore(artifact_root, "judge-empty")
    deps = Deps(settings=settings, store=store, llm=LLMAdapter(settings), image=GeminiImageAdapter(settings))
    report, err = _quality(store, deps)
    assert report is None
    assert err and "no script.json" in err


def test_rubric_fixtures_are_well_formed():
    """The bad fixture is structurally weaker by construction (empty arc, no beat roles,
    no causal through-line) — the real-model comparison lives in evals/rubric_fixtures.py."""
    assert GOOD_SCRIPT.arc and all(s.beat_role for s in GOOD_SCRIPT.scenes)
    assert BAD_SCRIPT.arc == "" and all(s.beat_role == "" for s in BAD_SCRIPT.scenes)
    assert len(GOOD_SCRIPT.scenes) == len(BAD_SCRIPT.scenes)
