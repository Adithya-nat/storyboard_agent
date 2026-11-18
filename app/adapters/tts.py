"""Gemini TTS (Interactions API): narration lines and <=2-speaker dialogue.

Daily-quota 429s cascade down ``tts_models``; 3.8 models return RIFF WAV (passed through
untouched); style goes in a speech_metadata annotation, never in the spoken text."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.adapters.errors import PermanentProviderError, TransientProviderError
from app.adapters.genai_client import (
    b64bytes,
    interactions_create,
    is_daily_quota,
    make_client,
    record_interaction,
)
from app.adapters.llm import _is_transient, parse_retry_after, wait_retry_after
from app.adapters.rate_limit import get_tts_limiter
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.observability import usage

log = get_logger(__name__)

_TTS_RATE = 24000  # Gemini TTS PCM: 24 kHz, 16-bit, mono


class DailyQuotaExhausted(TransientProviderError):
    """This model's daily TTS budget is spent — try the next model in the chain."""


@dataclass(frozen=True)
class DialogueTurn:
    speaker: str
    text: str
    style: str = ""


class GeminiTTSAdapter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = make_client(self.settings, timeout_seconds=120)
        return self._client

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def narrator_voice(self) -> str:
        return self.settings.narrator_voice_id.strip() or self.settings.tts_voice

    def synthesize(self, text: str, *, style: str = "", voice: str | None = None) -> bytes:
        """Narration text -> WAV bytes. Silent WAV in mock mode / for empty text."""
        if self.settings.mock_mode or not text.strip():
            if self.settings.mock_mode and text.strip() and self.settings.mock_token_usage:
                usage.record(total=self.settings.mock_token_usage)
            return _silent_wav(1.0)
        content = [_text_item(text, style=style)]
        speech_config = [{"voice": voice or self.narrator_voice()}]
        return self._synthesize_chain(content, speech_config)

    def synthesize_dialogue(
        self, turns: list[DialogueTurn], voices: dict[str, str]
    ) -> bytes:
        """Up to TWO speakers in one call (the model's multi-speaker limit). Speakers
        beyond two must be split by the caller. Silent WAV in mock mode."""
        turns = [t for t in turns if t.text.strip()]
        if self.settings.mock_mode or not turns:
            return _silent_wav(max(1.0, 0.4 * len(turns)))
        speakers = list(dict.fromkeys(t.speaker for t in turns))
        if len(speakers) > 2:
            raise ValueError("synthesize_dialogue supports at most 2 speakers per call")
        if len(speakers) == 1:
            # Single speaker: plain voice config, styles kept per turn.
            content = [_text_item(t.text, style=t.style) for t in turns]
            return self._synthesize_chain(content, [{"voice": voices[speakers[0]]}])
        content = [_text_item(t.text, style=t.style, speaker=t.speaker) for t in turns]
        speech_config = {
            "speakers": [{"speaker": s, "voice": voices[s]} for s in speakers]
        }
        return self._synthesize_chain(content, speech_config)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _synthesize_chain(self, content: list[dict], speech_config) -> bytes:
        models = self.settings.tts_models or ["gemini-3.8-flash-tts"]
        last: Exception | None = None
        for model in models:
            try:
                return self._with_retries(model, content, speech_config)
            except DailyQuotaExhausted as exc:
                log.warning("TTS model %s daily quota spent; cascading to the next model", model)
                last = exc
                continue
        assert last is not None
        raise TransientProviderError(f"all TTS models exhausted for today: {last}")

    def _with_retries(self, model: str, content, speech_config) -> bytes:
        retrying = Retrying(
            # Per-minute 429s / 5xx retry the SAME model; a daily-quota 429 must not
            # (it propagates to _synthesize_chain, which cascades to the next model).
            retry=retry_if_exception(
                lambda e: isinstance(e, TransientProviderError)
                and not isinstance(e, DailyQuotaExhausted)
            ),
            stop=stop_after_attempt(max(1, self.settings.tts_max_retries)),
            wait=wait_retry_after(
                wait_exponential(multiplier=1, min=2, max=20),
                cap=self.settings.tts_retry_cap_seconds,
            ),
            reraise=True,
        )
        return retrying(self._synthesize_once, model, content, speech_config)

    def _synthesize_once(self, model: str, content, speech_config) -> bytes:
        # RPM guard: pace calls under this model's per-minute cap (capacity=1, no burst).
        get_tts_limiter(self.settings.tts_rpm_limit, model=model).acquire()
        try:
            interaction = interactions_create(
                self.client,
                model=model,
                input=[{"type": "user_input", "content": content}],
                response_format={"type": "audio"},
                generation_config={"speech_config": speech_config},
            )
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            if is_daily_quota(exc):
                raise DailyQuotaExhausted(str(exc)) from exc
            if _is_transient(exc):
                raise TransientProviderError(
                    str(exc), retry_after=parse_retry_after(exc)
                ) from exc
            raise PermanentProviderError(str(exc)) from exc

        record_interaction("tts", model, interaction)
        audio = getattr(interaction, "output_audio", None)
        data = b64bytes(getattr(audio, "data", None))
        if not data:
            raise PermanentProviderError("Gemini TTS returned no audio.")
        return ensure_wav(data)


def _text_item(text: str, *, style: str = "", speaker: str = "") -> dict:
    item: dict = {"type": "text", "text": text}
    meta: dict = {}
    if speaker:
        meta["speaker"] = speaker
    if style:
        meta["style"] = style
    if meta:
        item["annotations"] = [{"type": "speech_metadata", **meta}]
    return item


def ensure_wav(data: bytes, *, rate: int = _TTS_RATE) -> bytes:
    """Return a WAV: pass a RIFF/WAVE blob through untouched; wrap headerless PCM."""
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return data
    return _pcm_to_wav(data, rate=rate)


def _pcm_to_wav(pcm: bytes, *, rate: int = _TTS_RATE, channels: int = 1, width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _silent_wav(seconds: float, *, rate: int = _TTS_RATE) -> bytes:
    return _pcm_to_wav(b"\x00\x00" * int(rate * seconds), rate=rate)


def silent_wav(seconds: float) -> bytes:
    """Public silence helper (a failed beat keeps the image on screen, just unvoiced)."""
    return _silent_wav(max(0.0, seconds))


def wav_duration_seconds(data: bytes) -> float:
    """Length of a WAV blob in seconds (frames / framerate, read from its header).
    Drives per-beat holds so an image stays on screen exactly as long as its narration."""
    with wave.open(io.BytesIO(data), "rb") as w:
        frames, rate = w.getnframes(), w.getframerate()
    return frames / rate if rate else 0.0


def concat_wavs(parts: list[bytes]) -> bytes:
    """Concatenate WAV blobs sequentially into one WAV (same format assumed). The
    narration FALLBACK: when the precisely-aligned track can't be built we still get one
    continuous voiceover from the audio we already synthesized — no extra TTS calls."""
    if not parts:
        return _silent_wav(0.5)
    buf = io.BytesIO()
    out: wave.Wave_write | None = None
    try:
        for p in parts:
            with wave.open(io.BytesIO(p), "rb") as w:
                if out is None:
                    out = wave.open(buf, "wb")
                    out.setnchannels(w.getnchannels())
                    out.setsampwidth(w.getsampwidth())
                    out.setframerate(w.getframerate())
                out.writeframes(w.readframes(w.getnframes()))
    finally:
        if out is not None:
            out.close()
    return buf.getvalue()
