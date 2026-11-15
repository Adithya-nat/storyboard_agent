"""Dialogue / Tamil stories end to end in mock mode, plus captions + motion meta units."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.adapters.gemini import GeminiImageAdapter
from app.adapters.llm import LLMAdapter
from app.adapters.tts import silent_wav
from app.agents.captions import captions, word_error_rate
from app.agents.motion import _dialogue_meta, clip_seconds
from app.artifacts.store import ArtifactStore
from app.core.config import Settings
from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.graph.state import Deps
from app.schemas import JobStatus, Script


@pytest.fixture
def stub_video(monkeypatch):
    async def _fake_assemble(timeline, output_path):
        Path(output_path).write_bytes(b"\x00")
        return output_path

    monkeypatch.setattr("app.agents.assembler.assemble_video", _fake_assemble)


class _TTS:
    def __init__(self):
        self.narr, self.dialogue = 0, 0

    def synthesize(self, text, **_k):
        self.narr += 1
        return silent_wav(1.0)

    def synthesize_dialogue(self, turns, voices):
        self.dialogue += 1
        assert set(t.speaker for t in turns) <= set(voices)
        return silent_wav(2.0)


async def _run(style: str, language: str = "en"):
    from app.workers.runner import run_job

    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "a keeper and a runaway in a storm", 30, style, language)
    await run_job(job.id, job.prompt, 30, False, style, language)
    async with get_sessionmaker()() as s:
        return await repo.get_job(s, job.id), await repo.get_stages(s, job.id)


async def test_dialogue_job_completes_with_publish_kit_and_ledger(stub_video):
    job, stages = await _run("dialogue")
    assert job.status == JobStatus.COMPLETED.value and job.style == "dialogue"
    kinds = {a.kind for a in job.artifacts}
    assert {"script", "publish", "cover", "scene_audio", "video"} <= kinds
    assert any(k.startswith("cast:") for k in kinds)  # character sheets persisted
    by = {st.name: st.status for st in stages}
    assert by["captions"] == "succeeded" and by["narration"] == "skipped"
    script = Script.model_validate_json(Path(next(a.path for a in job.artifacts if a.kind == "script")).read_text())
    assert all(sc.mode == "dialogue" for sc in script.scenes)
    async with get_sessionmaker()() as s:
        runs = await repo.list_runs(s)
    assert runs and runs[0].job_id == job.id and runs[0].status == "completed" and runs[0].style == "dialogue"


async def test_tamil_job_keeps_language_end_to_end(stub_video):
    job, _ = await _run("narrated", "ta")
    assert job.language == "ta"
    script = json.loads(Path(next(a.path for a in job.artifacts if a.kind == "script")).read_text())
    assert script["language"] == "ta" and script["characters"][0]["name"] == "KAMALA"


def _deps(root, job_id, **kw) -> Deps:
    s = Settings(gemini_api_key="", use_mock_providers=True, enable_tts=True, **kw)
    return Deps(settings=s, store=ArtifactStore(root, job_id), llm=LLMAdapter(s), image=GeminiImageAdapter(s), tts=_TTS())


async def test_captions_voices_dialogue_off_screen_without_clips(artifact_root):
    from app.agents.script_writer import _mock_script

    deps = _deps(artifact_root, "cap-off")
    script = _mock_script("x", 2, "dialogue")
    path = deps.store.save_json("script.json", script)
    pending = [{"order": sc.order, "voice": "pending", "hold": 3.0, "caption": sc.caption_text()} for sc in script.scenes]
    out = await captions({"script_path": path, "scene_audio": pending, "duration": 30}, deps)
    assert [a["voice"] for a in out["scene_audio"]] == ["off_screen", "off_screen"]
    assert all(a["path"] and a["hold"] >= 2.0 for a in out["scene_audio"])
    assert deps.tts.dialogue == 2
    assert any("off-screen" in w for w in out["warnings"])


async def test_captions_on_camera_uses_clip_length_and_transcript(artifact_root, monkeypatch):
    from app.agents.script_writer import _mock_script

    deps = _deps(artifact_root, "cap-on")

    class _Tx:
        def transcribe(self, wav, language):
            return "totally different words were said here"

        def attribute(self, wav, expected, language):
            return [("MAYA", "what was really said")]

    deps.transcriber = _Tx()
    script = _mock_script("x", 1, "dialogue")
    path = deps.store.save_json("script.json", script)
    clip = deps.store.save_bytes("motion_01.mp4", b"\x00")

    async def _dur(_p):
        return 6.5

    async def _extract(_v, wav):
        Path(wav).write_bytes(silent_wav(1.0))
        return wav

    monkeypatch.setattr("app.agents.captions.probe_duration", _dur)
    monkeypatch.setattr("app.agents.captions.extract_audio", _extract)
    state = {
        "script_path": path, "duration": 30,
        "scene_audio": [{"order": 1, "voice": "pending", "hold": 3.0, "caption": "c"}],
        "motion_clips": [{"order": 1, "path": clip, "has_audio": True}],
    }
    out = await captions(state, deps)
    meta = out["scene_audio"][0]
    assert meta["voice"] == "on_camera" and meta["hold"] == 6.5
    assert meta["caption"] == "Maya: what was really said"  # transcript diverged -> attributed lines
    assert deps.tts.dialogue == 0


def test_word_error_rate_handles_tamil_and_speaker_tags():
    assert word_error_rate("You shouldn't be out here.", "Speaker 1: You shouldn't be out here") == 0.0
    ref = "இந்த புயலில் நீ வெளியே இருக்கக்கூடாது"
    assert 0 < word_error_rate(ref, "இந்த புயல்ல நீ வெளியே இருக்கக்கூடாது") <= 0.25


def test_motion_meta_for_dialogue_and_clip_seconds():
    from app.agents.script_writer import _mock_script

    sc = _mock_script("x", 1, "dialogue").scenes[0]
    meta = _dialogue_meta(sc, {"MAYA": "FEMALE, 60s: warm", "LEO": "MALE, teen: bright"})
    assert meta["mode"] == "dialogue" and len(meta["lines"]) == 2
    assert meta["cast_voices"] == {"MAYA": "FEMALE, 60s: warm", "LEO": "MALE, teen: bright"}
    assert (clip_seconds(2.1, 4), clip_seconds(7.2, 4), clip_seconds(40, 4), clip_seconds(None, 4)) == (3, 8, 15, 4)


async def test_subtitles_off_skips_transcription_and_burns_nothing(artifact_root, monkeypatch):
    from app.agents.script_writer import _mock_script

    deps = _deps(artifact_root, "no-subs", burn_subtitles=False)

    class _Tx:
        def transcribe(self, *a):
            raise AssertionError("no transcription when subtitles are off")

    deps.transcriber = _Tx()
    script = _mock_script("x", 1, "dialogue")
    path = deps.store.save_json("script.json", script)
    clip = deps.store.save_bytes("motion_01.mp4", b"\x00")

    async def _dur(_p):
        return 6.0

    monkeypatch.setattr("app.agents.captions.probe_duration", _dur)
    out = await captions({"script_path": path, "duration": 30,
                          "scene_audio": [{"order": 1, "voice": "pending", "hold": 3.0, "caption": "c"}],
                          "motion_clips": [{"order": 1, "path": clip, "has_audio": True}]}, deps)
    assert out["scene_audio"][0]["voice"] == "on_camera" and out["scene_audio"][0]["hold"] == 6.0
