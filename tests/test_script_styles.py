"""Story style + language: prompts, scene counts, normalization invariants."""

from __future__ import annotations

from app.agents.script_writer import _mock_script, _system_prompt, normalize_script, scene_modes
from app.graph.state import scene_range, scenes_for_duration
from app.schemas import Character, DialogueLine, Scene, Script


def test_prompt_selects_style_and_language_blocks():
    p = _system_prompt(5, "dialogue", "ta")
    assert "STYLE = DIALOGUE" in p and "LANGUAGE = TAMIL" in p
    assert "voice_description" in p and "FEMALE, 60s:" in p  # voice-contrast rule (G1)
    n = _system_prompt(5, "narrated", "en")
    assert "STYLE = NARRATED" in n and "FEMALE, 60s:" not in n


def test_scene_counts_per_style():
    # A scene is one multi-shot take of 5-15s, so a film holds a few real scenes.
    assert scenes_for_duration(60, "narrated") == 8
    assert scenes_for_duration(60, "mixed") == 6
    assert scenes_for_duration(60, "dialogue") == 5
    assert scenes_for_duration(30, "dialogue") == 3
    assert scene_range(60) == (4, 12)
    assert scenes_for_duration(10, "dialogue") == 2  # a conversation needs >= 2 scenes


def test_mixed_puts_dialogue_on_inciting_and_climax():
    modes = scene_modes(10, "mixed")
    assert modes.count("dialogue") >= 2 and modes[0] == "narration" and modes[-1] == "narration"


def test_mock_scripts_follow_style():
    d = _mock_script("a storm", 4, "dialogue", "ta")
    assert all(sc.mode == "dialogue" and sc.lines for sc in d.scenes)
    assert d.language == "ta" and d.characters[0].name == "KAMALA"
    n = _mock_script("a storm", 4, "narrated")
    assert all(sc.mode == "narration" and sc.narration and not sc.lines for sc in n.scenes)


def test_normalize_enforces_invariants():
    raw = Script(
        title="t", logline="l",
        characters=[Character(name="maya", look="x")],
        scenes=[
            Scene(order=3, description="a", mode="narration", narration="n3"),
            Scene(order=1, description="b", mode="dialogue", lines=[
                DialogueLine(speaker="maya", text="hi"), DialogueLine(speaker="leo", text="yo"),
                DialogueLine(speaker="zed", text="third speaker dropped"),
            ]),
            Scene(order=2, description="c", mode="dialogue", lines=[]),
        ],
    )
    s = normalize_script(raw, style="mixed", language="en", n=3)
    assert [sc.order for sc in s.scenes] == [1, 2, 3]
    first = s.scenes[0]
    assert {l.speaker for l in first.lines} == {"MAYA", "LEO"}  # <=2 speakers, uppercased
    assert s.character("LEO") is not None  # unknown speaker added to the cast
    assert s.scenes[1].mode == "narration" and s.scenes[1].narration  # empty dialogue -> narration
    assert s.story_style == "mixed"


def test_normalize_forces_style_modes():
    s = normalize_script(_mock_script("x", 3, "mixed"), style="narrated", language="en", n=3)
    assert all(sc.mode == "narration" and not sc.lines and sc.narration for sc in s.scenes)


def test_dialogue_word_budget_tracks_scene_length():
    p = _system_prompt(5, "dialogue", "ta", 6.0)
    assert "scene 1: 6s, dialogue -> at most 7 words" in p  # (6s - 1.5s) x 1.8 wps x 0.9
    assert "at most 8 words" in _system_prompt(5, "dialogue", "en", 5.0)  # (5 - 1.5) x 2.6 x 0.9
