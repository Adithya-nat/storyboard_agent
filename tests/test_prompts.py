"""Prompt files: every agent prompt lives in app/prompts and renders strictly."""

from __future__ import annotations

import pytest

from app.prompts import all_prompts, render, variables


def test_every_prompt_file_loads_and_strips_header():
    names = all_prompts()
    assert {"script_writer/system", "judge/system", "designer/system", "reflection/reviser_system",
            "publish_kit/system", "music/score", "captions/attribute"} <= set(names)
    for name in names:
        dummy = {v: "X" for v in variables(name)}
        text = render(name, **dummy)
        assert text and not text.lstrip().startswith("<!--"), name
        assert "{{" not in text, name


def test_missing_variable_raises():
    with pytest.raises(KeyError, match="missing variables"):
        render("script_writer/user", idea="x")


def test_judge_bands_are_verbatim():
    # Load-bearing for reflection_threshold: rewording silently retunes the loop.
    text = render("judge/system")
    assert "reserve 90+ for genuinely excellent, 60-75 for solid, 40-60 for mediocre, below 30 for incoherent" in " ".join(text.split())


def test_name_variable_does_not_clash_with_loader_argument():
    out = render("designer/character_sheet", name="MAYA", look="silver braid", view="front", style_bible="moody")
    assert "MAYA" in out and "silver braid" in out
