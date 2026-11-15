"""Transcribe / music / image refs / model validation / Tamil captions / audio mix."""

from __future__ import annotations

import io
import json
import subprocess
from types import SimpleNamespace

import pytest
from PIL import Image

from app.adapters.music import LyriaMusicAdapter
from app.adapters.transcribe import TranscribeAdapter
from app.adapters.tts import silent_wav
from app.adapters.video import build_mix, ffmpeg_available
from app.core.config import Settings
from app.core.images import needs_complex_shaping, raqm_available, render_caption_overlay
from app.core.model_check import validate_models

_needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")


class _Rec:
    def __init__(self, items):
        self.items, self.requests = list(items), []

    def create(self, **kw):
        self.requests.append(kw)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_transcribe_request_shape_and_fallback_language():
    s = Settings(gemini_api_key="k", use_mock_providers=False, transcribe_fallback_languages=["ta"])
    a = TranscribeAdapter(s)
    rec = _Rec([SimpleNamespace(output_text="hello there", usage=None)])
    a._client = SimpleNamespace(interactions=rec)
    assert a.transcribe(b"RIFF....WAVE", "en") == "hello there"
    req = rec.requests[0]
    assert req["api_version"] == "v1beta" and req["model"] == "gemini-3.5-transcribe"
    assert req["generation_config"]["transcription_config"]["mode"]["diarization_mode"] == "speaker"

    called = {}
    a._transcribe_llm = lambda wav, lang: called.setdefault("lang", lang) or "tamil text"
    a.transcribe(b"x", "ta")
    assert called["lang"] == "ta"  # routed to the fallback model


def test_music_cascades_on_daily_quota(monkeypatch):
    import base64

    s = Settings(gemini_api_key="k", use_mock_providers=False)
    a = LyriaMusicAdapter(s)
    daily = Exception("429 RESOURCE_EXHAUSTED quota exceeded: requests per day")
    ok = SimpleNamespace(output_audio=SimpleNamespace(data=base64.b64encode(b"ID3fake-mp3").decode()), usage=None)
    rec = _Rec([daily, ok])
    a._client = SimpleNamespace(interactions=rec)
    data, ext, model = a.compose("instrumental")
    assert (data, ext, model) == (b"ID3fake-mp3", "mp3", "lyria-3-pro-preview")
    assert [r["model"] for r in rec.requests] == ["lyria-3.5", "lyria-3-pro-preview"]


def test_image_adapter_sends_every_ref_with_real_mime(monkeypatch):
    from app.adapters.gemini import GeminiImageAdapter, image_mime

    jpeg = b"\xff\xd8\xff" + b"0" * 10
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 10
    assert (image_mime(jpeg), image_mime(png)) == ("image/jpeg", "image/png")
    s = Settings(gemini_api_key="k", use_mock_providers=False, image_rpm_limit=100000)
    a = GeminiImageAdapter(s)
    seen = {}

    def gen(**kw):
        seen.update(kw)
        part = SimpleNamespace(inline_data=SimpleNamespace(data=b"IMG"))
        return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))], usage_metadata=None)

    a._client = SimpleNamespace(models=SimpleNamespace(generate_content=gen))
    assert a.generate_image("p", ref_image=png, refs=[jpeg, jpeg]) == b"IMG"
    mimes = [c.inline_data.mime_type for c in seen["contents"][1:]]
    assert mimes == ["image/png", "image/jpeg", "image/jpeg"]
    assert seen["config"].image_config.image_size == "2K"


async def test_model_validation_prunes_missing_chain_entries(monkeypatch):
    s = Settings(gemini_api_key="k", use_mock_providers=False)
    visible = ["gemini-4-carbon", "gemini-4-argon", "gemini-nano-banana-2.1", "gemini-3.8-flash-tts",
               "gemini-3.5-transcribe", "gemini-3.8-flash", "lyria-3-clip-preview"]

    class _C:
        def __init__(self, **_k):
            self.models = SimpleNamespace(list=lambda: [SimpleNamespace(name=f"models/{m}") for m in visible])

    monkeypatch.setattr("google.genai.Client", _C)
    report = await validate_models(s)
    assert report["checked"] and report["roles"]["tts"]["missing"] == ["gemini-3.8-flash-lite-tts"]
    assert s.tts_model_chain == ["gemini-3.8-flash-tts"]
    assert s.music_model_chain == ["lyria-3-clip-preview"]


def test_tamil_caption_uses_shaped_font():
    text = "இந்த புயலில் நீ வெளியே இருக்கக்கூடாது"
    assert needs_complex_shaping(text) and not needs_complex_shaping("hello")
    if not raqm_available():
        pytest.skip("libraqm not installed (brew install libraqm)")
    img = Image.open(io.BytesIO(render_caption_overlay(text, size=(720, 1280))))
    assert img.getbbox() is not None  # something was drawn
    from PIL import ImageFont

    from app.core.config import get_settings

    path = get_settings().tamil_font_path
    raqm = ImageFont.truetype(path, 40, layout_engine=ImageFont.Layout.RAQM)
    basic = ImageFont.truetype(path, 40, layout_engine=ImageFont.Layout.BASIC)
    # Shaping reorders vowel signs/forms conjuncts, so the shaped run differs from the naive one.
    assert raqm.getmask(text).size != basic.getmask(text).size or raqm.getbbox(text) != basic.getbbox(text)


@_needs_ffmpeg
async def test_mix_ducks_music_under_speech(artifact_root):
    d = artifact_root / "mix"
    d.mkdir(parents=True, exist_ok=True)
    speech = d / "s.wav"
    speech.write_bytes(silent_wav(1.0))
    music = d / "m.wav"
    music.write_bytes(silent_wav(3.0))
    out = d / "out.wav"
    await build_mix(speech=[(str(speech), 0.5)], ambience=[(str(speech), 0.0)], music=str(music),
                    total_seconds=5.0, output_path=str(out))
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(out)]))
    assert abs(float(probe["format"]["duration"]) - 5.0) < 0.15  # music looped to the full runtime


def test_labelled_refs_are_interleaved_before_each_image():
    from app.adapters.gemini import GeminiImageAdapter

    s = Settings(gemini_api_key="k", use_mock_providers=False, image_rpm_limit=100000)
    a = GeminiImageAdapter(s)
    seen = {}

    def gen(**kw):
        seen.update(kw)
        part = SimpleNamespace(inline_data=SimpleNamespace(data=b"IMG"))
        return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))], usage_metadata=None)

    a._client = SimpleNamespace(models=SimpleNamespace(generate_content=gen))
    png = b"\x89PNG\r\n\x1a\n0000"
    a.generate_image("p", refs=[("Style reference:", png), ("LEO — character reference 1:", png)])
    c = seen["contents"]
    assert c[0] == "p" and c[1] == "Style reference:" and c[3] == "LEO — character reference 1:"
    assert c[2].inline_data.mime_type == "image/png"


async def test_identity_mismatch_triggers_one_rerender(artifact_root, monkeypatch):
    from app.agents import image_gen as ig
    from app.agents.designer import _mock_shotlist
    from app.agents.script_writer import _mock_script
    from app.artifacts.store import ArtifactStore
    from app.core.images import render_card
    from app.graph.state import Deps

    s = Settings(gemini_api_key="", use_mock_providers=True)
    store = ArtifactStore(artifact_root, "idcheck")
    script = _mock_script("x", 1, "dialogue")
    sheet = store.save_bytes("char_MAYA_1.png", render_card("MAYA"))
    calls = {"n": 0}

    class _Img:
        def generate_image(self, prompt, refs=None, seed=None):
            calls["n"] += 1
            calls["prompt"], calls["refs"] = prompt, refs
            return render_card("frame")

    verdicts = iter([True, False])
    monkeypatch.setattr(ig, "_identity_mismatch", lambda *a, **k: next(verdicts))
    deps = Deps(settings=s, store=store, llm=None, image=_Img())
    state = {"job_id": "idcheck", "shotlist_path": store.save_json("shotlist.json", _mock_shotlist(script)),
             "script_path": store.save_json("script.json", script), "cast_paths": {"MAYA": [sheet]}}
    await ig.image_gen(state, deps)
    assert calls["n"] == 2  # first render failed the identity check, re-rendered once
    assert "match each one to ITS OWN labelled reference" in calls["prompt"]
    assert any(isinstance(r, tuple) and r[0].startswith("MAYA") for r in calls["refs"])


def test_mixed_script_caption_uses_a_font_per_word(monkeypatch):
    if not raqm_available():
        pytest.skip("libraqm not installed")
    from app.core import images

    used = []
    real = images._load_font

    def spy(size, text=""):
        f = real(size, text)
        used.append("tamil" if text and needs_complex_shaping(text) else "latin")
        return f

    monkeypatch.setattr(images, "_load_font", spy)
    images.render_caption_overlay("KATHIR: வண்டி வரப்போகுது", size=(720, 1280))
    assert "tamil" in used and "latin" in used


def test_dialogue_captions_use_display_names():
    from app.agents.script_writer import _mock_script

    s = _mock_script("x", 1, "dialogue", "ta")
    cap = s.scenes[0].caption_text(s.display_names())
    assert cap.startswith("கமலா:") and "KAMALA" not in cap
