"""Phase 3 TTS voiceover: mock synthesis yields a valid WAV, and the assembler
muxes a real audio stream into the video when enable_tts is on.
"""

from __future__ import annotations

import io
import json
import subprocess
import wave
from types import SimpleNamespace

import pytest

from app.adapters.errors import PermanentProviderError
from app.adapters.gemini import GeminiImageAdapter
from app.adapters.llm import LLMAdapter
from app.adapters.rate_limit import reset_tts_limiter
from app.adapters.tts import GeminiTTSAdapter, _pcm_to_wav
from app.adapters.video import ffmpeg_available
from app.agents.assembler import assembler
from app.artifacts.store import ArtifactStore
from app.core.config import Settings
from app.core.images import render_card
from app.graph.state import Deps, FrameMeta

_needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")


def _wav_info(data: bytes) -> tuple[int, int]:
    with wave.open(io.BytesIO(data)) as w:
        return w.getnframes(), w.getframerate()


def _mock_tts() -> GeminiTTSAdapter:
    return GeminiTTSAdapter(Settings(gemini_api_key="", use_mock_providers=True))


def test_mock_synthesize_returns_valid_wav():
    frames, rate = _wav_info(_mock_tts().synthesize("a lonely lighthouse keeper"))
    assert rate == 24000
    assert frames > 0


def test_empty_text_returns_silent_wav():
    frames, _ = _wav_info(_mock_tts().synthesize("   "))
    assert frames > 0  # silent, but a valid track


def test_pcm_to_wav_roundtrip():
    frames, rate = _wav_info(_pcm_to_wav(b"\x01\x02" * 1000))
    assert frames == 1000
    assert rate == 24000


# --------------------------------------------------------------------------- #
# Issue 1: real-codepath retry behaviour against a fake google-genai client.
# A 429 no longer instantly degrades a beat to silence — it retries (honoring
# the server's retryDelay) and recovers; a permanent error is not retried.
# --------------------------------------------------------------------------- #
def _audio_resp(data: bytes):
    """An Interactions TTS response: base64 audio on ``output_audio.data``."""
    import base64

    return SimpleNamespace(
        output_audio=SimpleNamespace(data=base64.b64encode(data).decode(), mime_type="audio/wav"),
        usage=None,
    )


class _FakeInteractions:
    """Replays a script of items: an Exception is raised, bytes -> audio resp. Records
    every request body so tests can assert the exact request shape."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0
        self.requests: list[dict] = []

    def create(self, **kw):
        self.requests.append(kw)
        item = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return _audio_resp(item)


def _real_tts(script, **overrides) -> GeminiTTSAdapter:
    reset_tts_limiter()
    settings = Settings(
        gemini_api_key="fake-key",  # disable mock_mode -> real retry codepath
        use_mock_providers=False,
        tts_rpm_limit=100_000,  # effectively unthrottled so tests stay fast
        tts_max_retries=5,
        tts_retry_cap_seconds=20.0,
        **overrides,
    )
    a = GeminiTTSAdapter(settings)
    fake = _FakeInteractions(script)
    a._client = SimpleNamespace(interactions=fake, models=fake)  # .models.calls kept for old asserts
    return a


_QUOTA_429 = Exception(
    "429 RESOURCE_EXHAUSTED. Quota exceeded for gemini-2.5-flash-tts. "
    "Please retry in 3.017691049s. 'retryDelay': '3s'"
)


def test_tts_retries_429_then_succeeds(monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda *_a, **_k: None)
    a = _real_tts([_QUOTA_429, b"\x01\x02" * 200])
    out = a.synthesize("a lonely lighthouse keeper")
    assert _wav_info(out)[0] > 0  # recovered to a real WAV, NOT silence
    assert a._client.models.calls == 2  # retried exactly once


def test_tts_permanent_error_not_retried(monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda *_a, **_k: None)
    a = _real_tts([Exception("400 INVALID_ARGUMENT: unsupported voice")])
    with pytest.raises(PermanentProviderError):
        a.synthesize("hello")
    assert a._client.models.calls == 1  # gave up immediately, no wasted retry


def test_tts_exhausts_retries_then_raises_transient(monkeypatch):
    from app.adapters.errors import TransientProviderError

    monkeypatch.setattr("tenacity.nap.time.sleep", lambda *_a, **_k: None)
    a = _real_tts([_QUOTA_429])  # 429 forever
    with pytest.raises(TransientProviderError):
        a.synthesize("never succeeds")
    assert a._client.models.calls == 5  # tts_max_retries attempts, then propagate


def test_tts_acquires_rate_limiter_each_call(monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda *_a, **_k: None)
    seen = {"n": 0}

    class _SpyBucket:
        def acquire(self, *_a, **_k):
            seen["n"] += 1
            return True

    monkeypatch.setattr("app.adapters.tts.get_tts_limiter", lambda *_a, **_k: _SpyBucket())
    a = _real_tts([_QUOTA_429, b"\x03\x04" * 100])
    a.synthesize("paced please")
    assert seen["n"] == 2  # one acquire before each attempt (the retry included)


def _has_audio_stream(path: str) -> bool:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_type", "-of", "json", path,
    ])
    return bool(json.loads(out).get("streams"))


@_needs_ffmpeg
async def test_assembler_muxes_voiceover(artifact_root):
    # Concat path (faster) + TTS on; mock TTS -> silent wav, real ffmpeg mux.
    settings = Settings(
        gemini_api_key="", use_mock_providers=True, enable_tts=True, enable_motion=False
    )
    store = ArtifactStore(artifact_root, "tts-job")
    deps = Deps(
        settings=settings,
        store=store,
        llm=LLMAdapter(settings),
        image=GeminiImageAdapter(settings),
        tts=GeminiTTSAdapter(settings),
    )
    frames = []
    for i in range(1, 4):
        p = store.save_bytes(f"frame_{i:02d}.png", render_card(f"scene {i}"))
        frames.append(FrameMeta(order=i, path=p, caption=f"narration line {i}", status="ok"))

    out = await assembler({"frames": frames, "duration": 30}, deps)
    assert _has_audio_stream(out["video_path"]), "no audio stream muxed into the video"


@_needs_ffmpeg
async def test_no_voiceover_when_disabled(artifact_root):
    settings = Settings(
        gemini_api_key="", use_mock_providers=True, enable_tts=False, enable_motion=False
    )
    store = ArtifactStore(artifact_root, "no-tts-job")
    deps = Deps(
        settings=settings,
        store=store,
        llm=LLMAdapter(settings),
        image=GeminiImageAdapter(settings),
        tts=GeminiTTSAdapter(settings),
    )
    frames = [
        FrameMeta(order=i, path=store.save_bytes(f"frame_{i:02d}.png", render_card(f"s{i}")),
                  caption=f"line {i}", status="ok")
        for i in range(1, 4)
    ]
    out = await assembler({"frames": frames, "duration": 30}, deps)
    assert not _has_audio_stream(out["video_path"])  # silent when TTS is off


# --------------------------------------------------------------------------- #
# 2026-11: 3.8 TTS models return a full WAV (RIFF), cascade on daily quota, designed
# voice ids, multi-speaker dialogue, and style annotations (never spoken text).
# --------------------------------------------------------------------------- #
def test_riff_wav_is_passed_through_not_double_wrapped(monkeypatch):
    from app.adapters.tts import ensure_wav

    real_wav = _pcm_to_wav(b"\x05\x06" * 4800)  # what gemini-3.8-flash-tts returns
    a = _real_tts([real_wav])
    out = a.synthesize("hello there")
    assert out == real_wav  # byte-identical: no second header
    assert ensure_wav(real_wav) == real_wav
    frames, rate = _wav_info(out)
    assert (frames, rate) == (4800, 24000)


def test_headerless_pcm_still_wrapped():
    from app.adapters.tts import ensure_wav

    frames, rate = _wav_info(ensure_wav(b"\x01\x02" * 2400))
    assert (frames, rate) == (2400, 24000)


def test_request_shape_uses_v1beta_interactions_and_prebuilt_voice():
    a = _real_tts([_pcm_to_wav(b"\x00\x00" * 10)])
    a.synthesize("The keeper climbs.")
    req = a._client.interactions.requests[0]
    assert req["api_version"] == "v1beta"  # Interactions only lives on v1beta (SDK 3.x defaults to v1)
    assert req["model"] == "gemini-3.8-flash-tts"
    assert req["response_format"] == {"type": "audio"}
    assert req["generation_config"]["speech_config"] == [{"voice": "Kore"}]
    item = req["input"][0]["content"][0]
    assert item == {"type": "text", "text": "The keeper climbs."}  # no style prepended


def test_designed_voice_id_and_style_annotation():
    a = _real_tts([_pcm_to_wav(b"\x00\x00" * 10)], narrator_voice_id="voices/abc123")
    a.synthesize("Night falls.", style="hushed, reverent")
    req = a._client.interactions.requests[0]
    assert req["generation_config"]["speech_config"] == [{"voice": "voices/abc123"}]
    item = req["input"][0]["content"][0]
    assert item["text"] == "Night falls."  # style is metadata, never read aloud
    assert item["annotations"] == [{"type": "speech_metadata", "style": "hushed, reverent"}]


def test_daily_quota_cascades_to_next_model(monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda *_a, **_k: None)
    daily = Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for metric generate_requests_per_model_per_day")
    a = _real_tts([daily, _pcm_to_wav(b"\x01\x01" * 50)])
    out = a.synthesize("cascade me")
    assert _wav_info(out)[0] == 50
    models = [r["model"] for r in a._client.interactions.requests]
    assert models == ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"]  # no same-model retry


def test_all_models_daily_exhausted_raises_transient(monkeypatch):
    from app.adapters.errors import TransientProviderError

    monkeypatch.setattr("tenacity.nap.time.sleep", lambda *_a, **_k: None)
    daily = Exception("429 quota exceeded: requests per day")
    a = _real_tts([daily])
    with pytest.raises(TransientProviderError):
        a.synthesize("nothing left")
    assert a._client.interactions.calls == 2  # one try per model, then give up


def test_multi_speaker_dialogue_request_shape():
    from app.adapters.tts import DialogueTurn

    a = _real_tts([_pcm_to_wav(b"\x00\x00" * 10)])
    a.synthesize_dialogue(
        [DialogueTurn("MAYA", "Did you hear that?", "whispered"), DialogueTurn("LEO", "Stay here.")],
        {"MAYA": "Kore", "LEO": "Puck"},
    )
    req = a._client.interactions.requests[0]
    assert req["generation_config"]["speech_config"] == {
        "speakers": [{"speaker": "MAYA", "voice": "Kore"}, {"speaker": "LEO", "voice": "Puck"}]
    }
    c0, c1 = req["input"][0]["content"]
    assert c0["annotations"] == [{"type": "speech_metadata", "speaker": "MAYA", "style": "whispered"}]
    assert c1["annotations"] == [{"type": "speech_metadata", "speaker": "LEO"}]


def test_multi_speaker_rejects_three_speakers():
    from app.adapters.tts import DialogueTurn

    a = _real_tts([_pcm_to_wav(b"\x00\x00" * 10)])
    with pytest.raises(ValueError):
        a.synthesize_dialogue(
            [DialogueTurn("A", "1"), DialogueTurn("B", "2"), DialogueTurn("C", "3")],
            {"A": "Kore", "B": "Puck", "C": "Charon"},
        )


def test_legacy_gemini_tts_model_env_is_tried_first():
    s = Settings(gemini_api_key="k", gemini_tts_model="gemini-2.5-flash-preview-tts")
    assert s.tts_models[0] == "gemini-2.5-flash-preview-tts"
    assert Settings(gemini_api_key="k").tts_models == ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"]
