"""Coherence pipeline: creative brief, scenes->shots screenplay, dialogue pass, speech register."""

from __future__ import annotations

from app.adapters.gemini import GeminiImageAdapter
from app.adapters.llm import LLMAdapter
from app.agents.brief import brief, mock_brief, normalize_brief, speech_block
from app.agents.dialogue_pass import DialoguePassResult, LineRewrite, apply_pass, screenplay_listing
from app.agents.script_writer import _mock_script, normalize_script, script_writer
from app.artifacts.store import ArtifactStore
from app.core.config import Settings
from app.graph.state import Deps
from app.schemas import CreativeBrief, DialogueLine, PlannedScene, Scene, SceneShot, Script, SpeechRegister


def _brief(scenes) -> CreativeBrief:
    return CreativeBrief(format="drama", purpose="p", hook="h", logline="l", setting="s", scenes=scenes)


def test_normalize_brief_clamps_count_and_fills_runtime():
    many = _brief([PlannedScene(order=i, purpose="x", seconds=2, mode="dialogue") for i in range(1, 20)])
    b = normalize_brief(many, duration=60, style="dialogue")
    assert 4 <= len(b.scenes) <= 12
    assert all(5 <= s.seconds <= 15 for s in b.scenes)
    one = normalize_brief(_brief([PlannedScene(order=1, purpose="all", seconds=60)]), duration=60, style="dialogue")
    assert len(one.scenes) >= 4 and abs(sum(s.seconds for s in one.scenes) - 60) < 1


def test_normalize_brief_modes_follow_style():
    b = normalize_brief(mock_brief("x", 5, 60, "mixed"), duration=60, style="narrated")
    assert {s.mode for s in b.scenes} == {"narration"}
    m = normalize_brief(_brief([PlannedScene(order=i, purpose="x", seconds=12, mode="dialogue") for i in range(1, 6)]),
                        duration=60, style="mixed")
    assert {"narration", "dialogue"} <= {s.mode for s in m.scenes}  # mixed always mixes


def test_speech_block_picks_language_and_era():
    modern = speech_block(_brief([]).model_copy(update={"speech": SpeechRegister(era="modern", dialect="Chennai")}), "ta")
    assert "PRESENT DAY" in modern and "Chennai" in modern and "Tamil script" in modern.replace("TAMIL SCRIPT", "Tamil script")
    period = speech_block(_brief([]).model_copy(update={"speech": SpeechRegister(era="period", period="Madras, 1972")}), "ta")
    assert "1972" in period and "NEVER modern slang" in period
    assert "OF ITS TIME" in speech_block(_brief([]).model_copy(update={"speech": SpeechRegister(era="period")}), "en")


def test_normalize_script_shots_fill_scene_and_flatten_lines():
    raw = Script(title="t", logline="l", scenes=[Scene(
        order=1, description="kitchen", mode="dialogue", seconds=10, shots=[
            SceneShot(size="wide", action="a", seconds=1, lines=[DialogueLine(speaker="riya", text="You're early.")]),
            SceneShot(size="close", action="b", seconds=1),
            SceneShot(size="close", action="c", seconds=2, lines=[
                DialogueLine(speaker="dev", text="I quit."), DialogueLine(speaker="zed", text="dropped")]),
        ])])
    s = normalize_script(raw, style="dialogue", language="en", n=1)
    sc = s.scenes[0]
    assert abs(sum(sh.seconds for sh in sc.shots) - sc.seconds) < 0.05
    assert [l.speaker for l in sc.lines] == ["RIYA", "DEV"]  # <=2 speakers, silent shot kept
    assert sc.shots[1].lines == []


def test_normalize_script_flat_output_becomes_one_shot():
    raw = Script(title="t", logline="l", scenes=[Scene(order=1, description="d", mode="dialogue",
                                                       lines=[DialogueLine(speaker="A", text="hi")])])
    sc = normalize_script(raw, style="dialogue", language="en", n=1).scenes[0]
    assert len(sc.shots) == 1 and sc.lines[0].text == "hi" and sc.seconds > 0


def test_normalize_script_caps_scene_at_15s():
    raw = _mock_script("x", 1, "dialogue", seconds=[40])
    assert normalize_script(raw, style="dialogue", language="en", n=1).scenes[0].seconds == 15


def test_dialogue_pass_applies_by_id_and_rejects_over_budget():
    s = normalize_script(_mock_script("storm", 2, "dialogue"), style="dialogue", language="en", n=2)
    listing = screenplay_listing(s)
    assert "[L1.1.1]" in listing and "[L2.2.1]" in listing
    result = DialoguePassResult(lines=[
        LineRewrite(id="L1.1.1", text="Wait, you're home already?", delivery="teasing"),
        LineRewrite(id="L2.1.1", text=" ".join(["word"] * 80)),  # blows scene 2's budget
    ])
    s, kept = apply_pass(s, result)
    assert s.scenes[0].shots[0].lines[0].text == "Wait, you're home already?"
    assert s.scenes[0].lines[0].delivery == "teasing"  # scene.lines re-flattened
    assert kept == [2] and "word word" not in s.scenes[1].lines[0].text


async def test_brief_then_screenplay_in_mock_mode(artifact_root):
    settings = Settings(use_mock_providers=True)
    store = ArtifactStore(artifact_root, "brief-1")
    deps = Deps(settings=settings, store=store, llm=LLMAdapter(settings), image=GeminiImageAdapter(settings))
    state = {"prompt": "two friends argue at a station", "duration": 60, "style": "dialogue", "language": "ta"}
    out = await brief(state, deps)
    b = CreativeBrief.model_validate(store.load_json(out["brief_path"]))
    sw = await script_writer(state, deps)
    s = Script.model_validate(store.load_json(sw["script_path"]))
    assert len(s.scenes) == len(b.scenes) == 5
    assert all(sc.shots and sc.seconds == p.seconds for sc, p in zip(s.scenes, b.scenes))
    assert (await brief(state, deps))["brief_path"] == out["brief_path"]  # resume reuses it


async def test_designer_renders_location_plates_and_frames_use_them(artifact_root):
    from app.agents.designer import designer, plate_name
    from app.agents.image_gen import frame_refs, load_plates

    settings = Settings(use_mock_providers=True)
    store = ArtifactStore(artifact_root, "loc-1")
    deps = Deps(settings=settings, store=store, llm=LLMAdapter(settings), image=GeminiImageAdapter(settings))
    state = {"prompt": "a kite on a beach", "duration": 30, "style": "dialogue", "language": "en"}
    state.update(await brief(state, deps))
    state.update(await script_writer(state, deps))
    out = await designer(state, deps)
    assert out["location_paths"] == {"MAIN_LOCATION": store.abspath(plate_name("MAIN_LOCATION"))}
    plates = load_plates(store, out)
    labels = [label for label, _ in frame_refs(None, {}, [], plates["MAIN_LOCATION"])]
    assert labels and labels[0].startswith("Location reference")
    again = await designer(state, deps)  # resume: reuses the plate
    assert again["location_paths"] == out["location_paths"]
