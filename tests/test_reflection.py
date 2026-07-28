"""W3: reflection node — critique/revise loop, accept-best, idempotency, degrade."""

from __future__ import annotations

import pytest

from app.adapters.errors import PermanentProviderError
from app.adapters.gemini import GeminiImageAdapter
from app.adapters.llm import LLMAdapter
from app.agents.reflection import reflection
from app.artifacts.store import ArtifactStore
from app.core.config import Settings
from app.graph.state import Deps
from app.schemas import JudgeReport, Scene, Script


def _script(title: str = "orig", n: int = 3) -> Script:
    return Script(
        title=title,
        logline="l",
        arc="a",
        scenes=[Scene(order=i, beat_role="rising", description=f"d{i}", narration=f"n{i}") for i in range(1, n + 1)],
    )


class _ScriptedLLM:
    """Drive the judge score sequence + reviser outputs deterministically.

    Judge calls (response_model=JudgeReport) pop the next score; reviser calls
    (response_model=Script) pop the next revised script (or echo via mock_factory).
    """

    def __init__(self, judge_scores, revised_scripts=None, raise_revise_on=None, raise_judge_on=None):
        self.judge_scores = list(judge_scores)
        self.revised_scripts = list(revised_scripts or [])
        self.raise_revise_on = raise_revise_on  # 1-based revise-call index to raise on
        self.raise_judge_on = raise_judge_on    # 1-based judge-call index to raise on
        self.judge_calls = 0
        self.revise_calls = 0

    def complete_json(self, *, system, user, response_model, mock_factory):
        if response_model is JudgeReport:
            self.judge_calls += 1
            if self.raise_judge_on == self.judge_calls:
                raise PermanentProviderError("judge down")
            score = self.judge_scores[min(self.judge_calls - 1, len(self.judge_scores) - 1)]
            return JudgeReport(
                structure=score, continuity=score, clarity=score,
                visual_concreteness=score, payoff=score, revision_guidance="g",
            )
        if response_model is Script:
            self.revise_calls += 1
            if self.raise_revise_on == self.revise_calls:
                raise PermanentProviderError("reviser down")
            if self.revised_scripts:
                return self.revised_scripts[min(self.revise_calls - 1, len(self.revised_scripts) - 1)]
            return mock_factory(user)  # echo current script
        raise AssertionError(response_model)


def _deps(artifact_root, job_id, *, llm, threshold=85, max_iters=2, enabled=True):
    settings = Settings(
        use_mock_providers=True,
        reflection_enabled=enabled,
        reflection_threshold=threshold,
        reflection_max_iters=max_iters,
    )
    store = ArtifactStore(artifact_root, job_id)
    return Deps(settings=settings, store=store, llm=llm, image=GeminiImageAdapter(settings)), store


def _seed(store, script: Script):
    store.save_json("script.json", script)
    return {"script_path": store.abspath("script.json")}


async def test_disabled_is_noop(artifact_root):
    deps, store = _deps(artifact_root, "ref-off", llm=_ScriptedLLM([10]), enabled=False)
    state = _seed(store, _script())
    out = await reflection(state, deps)
    assert out == {}
    assert not store.exists("reflection.json")


async def test_idempotent_skip_when_report_exists(artifact_root):
    llm = _ScriptedLLM([10])
    deps, store = _deps(artifact_root, "ref-idem", llm=llm)
    _seed(store, _script())
    store.save_json("reflection.json", {"iterations": 0})
    out = await reflection({"script_path": store.abspath("script.json")}, deps)
    assert out["reflection_path"].endswith("reflection.json")
    assert llm.judge_calls == 0  # never judged -> truly skipped


async def test_accept_immediately_when_above_threshold(artifact_root):
    llm = _ScriptedLLM([90])  # initial >= threshold 85
    deps, store = _deps(artifact_root, "ref-accept", llm=llm)
    _seed(store, _script("orig"))
    out = await reflection({"script_path": store.abspath("script.json")}, deps)
    assert llm.revise_calls == 0  # no revision needed
    report = store.load_json(out["reflection_path"])
    assert report["iterations"] == 0 and report["met_threshold"] is True
    assert out["warnings"] == []


async def test_rising_scores_converge(artifact_root):
    llm = _ScriptedLLM([70, 88])  # initial 70 -> revise -> 88 >= 85 stop
    deps, store = _deps(artifact_root, "ref-converge", llm=llm)
    _seed(store, _script())
    out = await reflection({"script_path": store.abspath("script.json")}, deps)
    assert llm.revise_calls == 1 and llm.judge_calls == 2
    report = store.load_json(out["reflection_path"])
    assert report["final_overall"] == 88 and report["met_threshold"] is True


async def test_flat_low_hits_cap_and_warns(artifact_root):
    llm = _ScriptedLLM([50, 50, 50])
    deps, store = _deps(artifact_root, "ref-cap", llm=llm, threshold=85, max_iters=2)
    _seed(store, _script())
    out = await reflection({"script_path": store.abspath("script.json")}, deps)
    report = store.load_json(out["reflection_path"])
    assert report["iterations"] == 2 and report["met_threshold"] is False
    assert any("below threshold" in w for w in out["warnings"])


async def test_accept_best_keeps_highest_not_last(artifact_root):
    # threshold high so both iters run: 50 -> 90 -> 60. Best is 90, not the last (60).
    better = _script("better")
    worse = _script("worse")
    llm = _ScriptedLLM([50, 90, 60], revised_scripts=[better, worse])
    deps, store = _deps(artifact_root, "ref-best", llm=llm, threshold=99, max_iters=2)
    _seed(store, _script("orig"))
    out = await reflection({"script_path": store.abspath("script.json")}, deps)
    report = store.load_json(out["reflection_path"])
    assert report["final_overall"] == 90
    # script.json overwritten with the BEST-scoring revision ("better"), not "worse".
    assert Script.model_validate(store.load_json(store.abspath("script.json"))).title == "better"


async def test_initial_judge_failure_skips(artifact_root):
    llm = _ScriptedLLM([50], raise_judge_on=1)
    deps, store = _deps(artifact_root, "ref-judgefail", llm=llm)
    _seed(store, _script("orig"))
    out = await reflection({"script_path": store.abspath("script.json")}, deps)
    assert "reflection_path" not in out  # nothing written -> resume can retry
    assert any("judge unavailable" in w for w in out["warnings"])
    assert Script.model_validate(store.load_json(store.abspath("script.json"))).title == "orig"


async def test_reviser_failure_keeps_best(artifact_root):
    llm = _ScriptedLLM([50], raise_revise_on=1)
    deps, store = _deps(artifact_root, "ref-revisefail", llm=llm)
    _seed(store, _script("orig"))
    out = await reflection({"script_path": store.abspath("script.json")}, deps)
    report = store.load_json(out["reflection_path"])
    assert report["final_overall"] == 50  # kept the initial best
    assert any("revision step failed" in w for w in out["warnings"])


async def test_revision_changing_scene_count_is_rejected(artifact_root):
    bad = _script("bad", n=5)  # different scene count than the 3-scene original
    llm = _ScriptedLLM([50, 95], revised_scripts=[bad])
    deps, store = _deps(artifact_root, "ref-count", llm=llm, threshold=85, max_iters=1)
    _seed(store, _script("orig", n=3))
    out = await reflection({"script_path": store.abspath("script.json")}, deps)
    assert any("scene count" in w for w in out["warnings"])
    # original kept; the higher score from the bad revision was never adopted.
    kept = Script.model_validate(store.load_json(store.abspath("script.json")))
    assert kept.title == "orig" and len(kept.scenes) == 3


async def test_mock_mode_below_threshold_loops_and_accepts_best(artifact_root):
    """End-to-end with the real (mock) LLM: fixed mock score < threshold -> cap+best."""
    settings = Settings(
        use_mock_providers=True, mock_quality_score=60,
        reflection_threshold=85, reflection_max_iters=2,
    )
    store = ArtifactStore(artifact_root, "ref-mock")
    deps = Deps(settings=settings, store=store, llm=LLMAdapter(settings),
                image=GeminiImageAdapter(settings))
    _seed(store, _script("orig"))
    out = await reflection({"script_path": store.abspath("script.json")}, deps)
    report = store.load_json(out["reflection_path"])
    assert report["final_overall"] == 60 and report["met_threshold"] is False
