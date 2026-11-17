"""A content-block rewrite must reach the per-shot actions a multi-shot clip is prompted from.

Regression: job d535ded9 scene 4 was blocked three times because the rewrite replaced only
the scene summary and dialogue, while the video prompt was still built from the original
shot actions ("execution strike", "crimson juice leaking from cracked armor").
"""

from __future__ import annotations

from types import SimpleNamespace

from app.adapters.omni import SceneSpec, ShotSpec, build_prompt
from app.agents.video_gen import SceneRescue, _rewrite, apply_rescue
from app.core.config import Settings
from app.prompts import render

SHOTS = (
    ShotSpec(size="medium", action="Titan raises his war club for an execution strike", seconds=3.0,
             lines=(("TITAN", "Harvest's over.", "boom"),)),
    ShotSpec(size="close", action="Melon kneels in a puddle of crimson juice leaking from cracked armor", seconds=4.0,
             lines=(("MELON", "Under pressure, it cuts stone.", "grit"),)),
)
SPEC = SceneSpec(order=4, scene="a warehouse duel", mode="dialogue", characters=("TITAN", "MELON"),
                 lines=(("TITAN", "Harvest's over.", "boom"), ("MELON", "Under pressure, it cuts stone.", "grit")),
                 seconds=12.0, shots=SHOTS, setting="neon fruit warehouse, midnight")


def _rescue(**kw) -> dict:
    base = {
        "scene": "a tense standoff in the warehouse",
        "image_prompt": "two armoured fruit heroes face off",
        "lines": [{"speaker": "TITAN", "text": "Harvest's over."}, {"speaker": "MELON", "text": "Under pressure."}],
        "change": "removed the strike and the juice",
    }
    return {**base, **kw}


def test_rewritten_shot_actions_replace_the_blocked_ones_in_the_video_prompt():
    out = apply_rescue(SPEC, _rescue(shots=["Titan plants his staff and glares", "Melon steadies himself on wet concrete"]))
    assert [sh.action for sh in out.shots] == ["Titan plants his staff and glares", "Melon steadies himself on wet concrete"]
    # lines and timing survive the swap
    assert [sh.lines for sh in out.shots] == [(("TITAN", "Harvest's over.", ""),), (("MELON", "Under pressure.", ""),)]
    assert [sh.seconds for sh in out.shots] == [3.0, 4.0]
    prompt = build_prompt(out, 12)
    assert "execution strike" not in prompt and "crimson juice" not in prompt
    assert "plants his staff" in prompt


def test_shot_actions_are_kept_when_the_rewrite_does_not_match_the_shot_count():
    for shots in ([], ["only one"], ["a", ""]):
        out = apply_rescue(SPEC, _rescue(shots=shots))
        assert [sh.action for sh in out.shots] == [sh.action for sh in SHOTS]


def test_an_older_saved_rescue_without_shots_still_loads():
    out = apply_rescue(SPEC, _rescue())  # rescue_NN.json written before this field existed
    assert out.scene == "a tense standoff in the warehouse"
    assert [sh.action for sh in out.shots] == [sh.action for sh in SHOTS]
    assert SceneRescue.model_validate(_rescue()).shots == []


def test_the_rewrite_prompt_asks_for_every_shot():
    seen = {}

    class _LLM:
        def complete_json(self, *, system, user, response_model, mock_factory, role):
            seen["user"] = user
            return mock_factory(user)

    deps = SimpleNamespace(llm=_LLM(), settings=Settings())
    r = _rewrite(deps, SPEC, "prohibited_content", "blocked", 1)
    assert "exactly 2 rewritten actions" in seen["user"]
    assert "1. medium: Titan raises his war club" in seen["user"]
    assert len(r.shots) == 2

    single = SceneSpec(order=1, scene="one take", mode="narration")
    _rewrite(deps, single, "prohibited_content", "blocked", 1)
    assert "one continuous shot" in seen["user"]


def test_rescue_prompts_render():
    assert "{{" not in render("omni/rescue_shots", count=3)
