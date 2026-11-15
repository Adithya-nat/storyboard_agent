"""Real API smoke tests (~$0.50). Run: pytest -m live. Skipped by default and in CI."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def settings():
    from dotenv import dotenv_values

    from app.core.config import Settings

    env = dotenv_values(".env")
    if not env.get("GEMINI_API_KEY"):
        pytest.skip("no GEMINI_API_KEY in .env")
    os.environ.pop("USE_MOCK_PROVIDERS", None)
    return Settings(_env_file=".env", use_mock_providers=False, gemini_api_key=env["GEMINI_API_KEY"])


def test_script_writer_model_returns_valid_script(settings):
    from app.adapters.llm import LLMAdapter
    from app.agents.script_writer import _system_prompt, normalize_script
    from app.schemas import Script

    script = LLMAdapter(settings).complete_json(
        system=_system_prompt(3, "dialogue", "en", 8.0),
        user="<idea>a lost umbrella finds its owner</idea> 3 scenes.",
        response_model=Script, mock_factory=lambda _u: None, role="script",
    )
    script = normalize_script(script, style="dialogue", language="en", n=3)
    assert len(script.scenes) == 3 and all(sc.lines for sc in script.scenes)


def test_image_model_returns_vertical_image(settings):
    import io

    from PIL import Image

    from app.adapters.gemini import GeminiImageAdapter

    data = GeminiImageAdapter(settings.model_copy(update={"image_size": "1K"})).generate_image("a paper boat on a pond")
    w, h = Image.open(io.BytesIO(data)).size
    assert h > w


def test_tts_returns_riff_wav(settings):
    from app.adapters.tts import GeminiTTSAdapter, wav_duration_seconds

    wav = GeminiTTSAdapter(settings).synthesize("A short test line.")
    assert wav[:4] == b"RIFF" and wav[44:48] != b"RIFF" and wav_duration_seconds(wav) > 0.5


def test_tamil_tts_and_transcription(settings):
    from app.adapters.transcribe import TranscribeAdapter
    from app.adapters.tts import GeminiTTSAdapter

    wav = GeminiTTSAdapter(settings).synthesize("இன்று மழை பெய்கிறது.")
    text = TranscribeAdapter(settings).transcribe(wav, "ta")
    assert "மழை" in text


def test_music_returns_audio(settings):
    from app.adapters.music import LyriaMusicAdapter

    data, ext, model = LyriaMusicAdapter(settings).compose("A calm five second instrumental sting. Instrumental only.")
    assert len(data) > 1000 and ext in ("mp3", "wav")


def test_omni_short_clip_has_audio(settings, tmp_path):
    from app.adapters.omni import OmniVideoAdapter, SceneSpec, has_audio

    s = settings.model_copy(update={"omni_resolution": "360p", "omni_speaker_check": False})
    from app.adapters.gemini import GeminiImageAdapter

    frame = GeminiImageAdapter(settings.model_copy(update={"image_size": "1K"})).generate_image("a candle on a table, night")
    clip = OmniVideoAdapter(s, journal_dir=tmp_path).generate(
        frame, SceneSpec(order=1, scene="a candle flickering on a table", seconds=3), key="live-smoke"
    )
    p = tmp_path / "c.mp4"
    p.write_bytes(clip.data)
    assert clip.seconds == 3 and has_audio(str(p))
