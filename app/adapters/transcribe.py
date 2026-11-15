"""Transcription for dialogue captions: what a clip actually says, attributed to the cast."""

from __future__ import annotations

from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.adapters.errors import PermanentProviderError, TransientProviderError
from app.adapters.genai_client import b64str, interactions_create, make_client, record_interaction
from app.adapters.llm import _is_transient, parse_retry_after
from app.adapters.rate_limit import get_text_limiter
from app.core.config import Settings, get_settings
from app.observability import usage
from app.prompts import render


class SpokenLine(BaseModel):
    speaker: str = Field(description="One of the given character names.")
    text: str = Field(description="Exactly what this speaker said, in the original language and script.")


class SpokenLines(BaseModel):
    lines: list[SpokenLine]


class TranscribeAdapter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = make_client(self.settings, timeout_seconds=180)
        return self._client

    def transcribe(self, wav: bytes, language: str = "en") -> str:
        """Verbatim transcript (speaker-diarized when the primary model is used)."""
        if self.settings.mock_mode:
            return ""
        if language in self.settings.transcribe_fallback_languages:
            return self._transcribe_llm(wav, language)
        try:
            return self._transcribe_primary(wav)
        except PermanentProviderError:
            return self._transcribe_llm(wav, language)

    def attribute(self, wav: bytes, expected: list[tuple[str, str]], language: str = "en") -> list[tuple[str, str]]:
        """Assign what was actually said to the cast (structured), guided by the script lines."""
        if self.settings.mock_mode:
            return list(expected)
        from google.genai import types

        model = self.settings.transcribe_fallback_model
        prompt = render(
            "captions/attribute",
            speakers=", ".join(dict.fromkeys(s for s, _ in expected)),
            script="\n".join(f"{s}: {t}" for s, t in expected),
        )
        get_text_limiter(self.settings.text_rpm_limit).acquire()
        resp = self.client.models.generate_content(
            model=model,
            contents=[types.Part.from_bytes(data=wav, mime_type="audio/wav"), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=SpokenLines),
        )
        usage.record_gemini_call("transcribe", model, getattr(resp, "usage_metadata", None))
        return [(l.speaker, l.text) for l in SpokenLines.model_validate_json(resp.text).lines]

    @retry(
        retry=retry_if_exception_type(TransientProviderError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=20),
        reraise=True,
    )
    def _transcribe_primary(self, wav: bytes) -> str:
        model = self.settings.transcribe_model
        try:
            it = interactions_create(
                self.client,
                model=model,
                input=[{"type": "audio", "data": b64str(wav), "mime_type": "audio/wav"}],
                generation_config={
                    "transcription_config": {"mode": {"type": "verbatim", "diarization_mode": "speaker"}}
                },
            )
        except Exception as exc:  # noqa: BLE001
            if _is_transient(exc):
                raise TransientProviderError(str(exc), retry_after=parse_retry_after(exc)) from exc
            raise PermanentProviderError(str(exc)) from exc
        record_interaction("transcribe", model, it)
        return it.output_text or ""

    def _transcribe_llm(self, wav: bytes, language: str) -> str:
        from google.genai import types

        model = self.settings.transcribe_fallback_model
        get_text_limiter(self.settings.text_rpm_limit).acquire()
        resp = self.client.models.generate_content(
            model=model,
            contents=[types.Part.from_bytes(data=wav, mime_type="audio/wav"), render("captions/transcribe")],
        )
        usage.record_gemini_call("transcribe", model, getattr(resp, "usage_metadata", None))
        return resp.text or ""
