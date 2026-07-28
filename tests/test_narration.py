"""W5: synced narration — duration helpers, audio-driven holds, idempotency, degrade.

The timing math + degrade ladder are covered WITHOUT ffmpeg; the real mux is exercised
(ffmpeg-gated) by tests/test_tts.py.
"""

from __future__ import annotations

import pytest

from app.adapters.errors import PermanentProviderError
from app.adapters.gemini import GeminiImageAdapter
from app.adapters.llm import LLMAdapter
from app.adapters.tts import (
    GeminiTTSAdapter,
    _pcm_to_wav,
    concat_wavs,
    silent_wav,
    wav_duration_seconds,
)
from app.adapters.video import concat_offsets, ffmpeg_available, xfade_offsets
from app.agents import assembler as assembler_mod
from app.agents.assembler import _add_synced_voiceover, _synthesize_beats, assembler
from app.artifacts.store import ArtifactStore
from app.core.config import Settings
from app.core.images import render_card
from app.graph.state import Deps, FrameMeta

_needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")


def _settings(**kw):
    base = dict(gemini_api_key="", use_mock_providers=True, enable_motion=False)
    base.update(kw)
    return Settings(**base)


def _deps(artifact_root, job_id, settings, tts=None):
    return Deps(
        settings=settings,
        store=ArtifactStore(artifact_root, job_id),
        llm=LLMAdapter(settings),
        image=GeminiImageAdapter(settings),
        tts=tts if tts is not None else GeminiTTSAdapter(settings),
    )


def _frames(store, n=3):
    out = []
    for i in range(1, n + 1):
        p = store.save_bytes(f"frame_{i:02d}.png", render_card(f"scene {i}"))
        out.append(FrameMeta(order=i, path=p, caption=f"narration line {i}", status="ok"))
    return out


# --------------------------------------------------------------------------- #
# Duration helpers (pure)
# --------------------------------------------------------------------------- #
def test_wav_duration_seconds():
    one_sec = _pcm_to_wav(b"\x01\x02" * 24000)  # 24000 frames @ 24kHz = 1.0s
    assert abs(wav_duration_seconds(one_sec) - 1.0) < 1e-6
    assert abs(wav_duration_seconds(silent_wav(2.5)) - 2.5) < 1e-3


def test_concat_wavs_sums_durations():
    a, b = silent_wav(1.0), silent_wav(0.5)
    assert abs(wav_duration_seconds(concat_wavs([a, b])) - 1.5) < 1e-3
    assert wav_duration_seconds(concat_wavs([])) > 0  # empty -> short valid track


# --------------------------------------------------------------------------- #
# Offset math: audio placement == video timeline (single source of truth)
# --------------------------------------------------------------------------- #
def test_concat_offsets_are_cumulative():
    assert concat_offsets([1.5, 2.0, 1.0]) == pytest.approx([0.0, 1.5, 3.5])


def test_xfade_offsets_match_video_timeline():
    holds, d = [2.0, 2.0, 2.0, 2.0], 0.4
    assert xfade_offsets(holds, d) == pytest.approx([0.0, 1.6, 3.2, 4.8])
    assert xfade_offsets([], d) == []


# --------------------------------------------------------------------------- #
# Audio-driven holds + idempotency
# --------------------------------------------------------------------------- #
class _CountingTTS:
    def __init__(self, dur=1.0):
        self.calls = 0
        self.dur = dur

    def synthesize(self, text: str) -> bytes:
        self.calls += 1
        return silent_wav(self.dur)


async def test_synthesize_beats_drives_holds(artifact_root):
    settings = _settings(enable_tts=True, narration_floor_seconds=1.5, narration_pad_seconds=0.4)
    tts = _CountingTTS(dur=3.0)
    deps = _deps(artifact_root, "narr-holds", settings, tts=tts)
    frames = _frames(deps.store)

    beats, warns = await _synthesize_beats(frames, deps)
    assert warns == []
    assert tts.calls == 3
    # hold = max(floor 1.5, audio 3.0 + pad 0.4) = 3.4
    assert all(abs(hold - 3.4) < 1e-6 for _, hold in beats)


async def test_synthesize_beats_is_idempotent_on_resume(artifact_root):
    settings = _settings(enable_tts=True)
    tts = _CountingTTS()
    deps = _deps(artifact_root, "narr-idem", settings, tts=tts)
    frames = _frames(deps.store)

    await _synthesize_beats(frames, deps)
    assert tts.calls == 3
    # Second pass (resume): narration_NN.wav exists -> no new TTS calls.
    await _synthesize_beats(frames, deps)
    assert tts.calls == 3


class _FailTTS:
    def synthesize(self, text: str) -> bytes:
        raise PermanentProviderError("tts down")


async def test_synthesize_beats_degrades_per_beat(artifact_root):
    settings = _settings(enable_tts=True, narration_floor_seconds=1.5)
    deps = _deps(artifact_root, "narr-fail", settings, tts=_FailTTS())
    frames = _frames(deps.store)

    beats, warns = await _synthesize_beats(frames, deps)
    assert len(beats) == 3  # every image still gets a (silent) beat
    assert len(warns) == 3 and all("voiceover failed" in w for w in warns)


# --------------------------------------------------------------------------- #
# Degrade ladder for the synced track (no ffmpeg needed when both raise)
# --------------------------------------------------------------------------- #
async def test_voiceover_all_paths_fail_leaves_silent(artifact_root, monkeypatch):
    settings = _settings(enable_tts=True)
    deps = _deps(artifact_root, "narr-silent", settings, tts=_CountingTTS())
    frames = _frames(deps.store)
    beats, _ = await _synthesize_beats(frames, deps)
    offsets = concat_offsets([h for _, h in beats])

    async def boom(*a, **k):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(assembler_mod, "build_narration_track", boom)
    monkeypatch.setattr(assembler_mod, "mux_audio", boom)

    warnings: list[str] = []
    await _add_synced_voiceover(beats, offsets, deps, "/nonexistent/v.mp4", warnings)
    assert any("synced track failed" in w for w in warnings)
    assert any("video is silent" in w for w in warnings)


@_needs_ffmpeg
async def test_synced_voiceover_falls_back_to_single_track(artifact_root, monkeypatch):
    """Synced track raises -> the single continuous-track fallback still muxes audio."""
    settings = _settings(enable_tts=True)
    deps = _deps(artifact_root, "narr-fallback", settings, tts=_CountingTTS())
    frames = _frames(deps.store)
    out = await assembler({"frames": frames, "duration": 30}, deps)  # builds a real mp4
    beats, _ = await _synthesize_beats(frames, deps)
    offsets = concat_offsets([h for _, h in beats])

    async def boom(*a, **k):
        raise RuntimeError("synced builder down")

    monkeypatch.setattr(assembler_mod, "build_narration_track", boom)

    warnings: list[str] = []
    await _add_synced_voiceover(beats, offsets, deps, out["video_path"], warnings)
    assert any("synced track failed" in w for w in warnings)
    assert not any("silent" in w for w in warnings)  # fallback succeeded


# --------------------------------------------------------------------------- #
# Audio-driven holds change runtime (narrated != silent exact-duration)
# --------------------------------------------------------------------------- #
@_needs_ffmpeg
async def test_narrated_runtime_tracks_narration_not_duration(artifact_root):
    import json
    import subprocess

    def _dur(path: str) -> float:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", path,
        ])
        return float(json.loads(out)["format"]["duration"])

    deps = _deps(artifact_root, "rt-narr", _settings(enable_tts=True))
    narrated = await assembler({"frames": _frames(deps.store), "duration": 30}, deps)
    # 3 mock beats @ ~1.5s holds -> well under the 30s a silent exact-duration build makes.
    assert _dur(narrated["video_path"]) < 20
    assert narrated.get("narration_path")  # assembler reports the muxed track


async def test_assembler_rejects_empty_frames(artifact_root):
    deps = _deps(artifact_root, "empty", _settings(enable_tts=False))
    with pytest.raises(ValueError, match="no frames"):
        await assembler({"frames": [], "duration": 30}, deps)


@_needs_ffmpeg
async def test_assembler_narrated_motion_path(artifact_root):
    """Exercise the Ken Burns + per-beat-hold branch (enable_motion ON, narration ON)."""
    deps = _deps(artifact_root, "narr-motion", _settings(enable_tts=True, enable_motion=True))
    out = await assembler({"frames": _frames(deps.store), "duration": 30}, deps)
    assert out.get("narration_path")
    assert out["video_path"].endswith("storyboard.mp4")


@_needs_ffmpeg
async def test_assembler_silent_motion_path(artifact_root):
    """enable_motion ON + narration OFF -> uniform exact-duration holds (Ken Burns)."""
    deps = _deps(artifact_root, "silent-motion", _settings(enable_tts=False, enable_motion=True))
    out = await assembler({"frames": _frames(deps.store), "duration": 30}, deps)
    assert "narration_path" not in out  # silent
    assert out["video_path"].endswith("storyboard.mp4")


@_needs_ffmpeg
async def test_runner_persists_narration_end_to_end(monkeypatch):
    """Default prod config (enable_tts=True): the full chain runs end-to-end —
    assembler -> narration_path -> graph state -> _persist -> 'narration' artifact ->
    narrated=True. conftest pins ENABLE_TTS off, so this is the only test that composes it.
    """
    from httpx import ASGITransport, AsyncClient

    from app.db import repository as repo
    from app.db.session import get_sessionmaker
    from app.main import app
    from app.workers import runner as runner_mod

    # enable_tts on; artifact_root / enable_motion come from the test env (conftest).
    settings = Settings(use_mock_providers=True, enable_tts=True)
    monkeypatch.setattr(runner_mod, "get_settings", lambda: settings)

    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "a tiny narrated story", 30)
    await runner_mod.run_job(job.id, "a tiny narrated story", 30)

    async with get_sessionmaker()() as s:
        j = await repo.get_job(s, job.id)
    kinds = {a.kind for a in j.artifacts}
    assert "narration" in kinds and "video" in kinds

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        body = (await c.get(f"/storyboard/{job.id}/storyboard")).json()
    assert body["narrated"] is True


async def test_content_api_reports_narrated():
    from httpx import ASGITransport, AsyncClient

    from app.db import repository as repo
    from app.db.session import get_sessionmaker
    from app.main import app

    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
        await repo.add_artifact(s, job.id, kind="narration", path="/x/narration_track.wav")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        body = (await c.get(f"/storyboard/{job.id}/storyboard")).json()
    assert body["narrated"] is True
