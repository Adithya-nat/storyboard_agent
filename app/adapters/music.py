"""Lyria music bed: one instrumental score per film (Interactions API)."""

from __future__ import annotations

from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from app.adapters.errors import PermanentProviderError, TransientProviderError
from app.adapters.genai_client import b64bytes, interactions_create, is_daily_quota, make_client
from app.adapters.llm import _is_transient, parse_retry_after, wait_retry_after
from app.adapters.rate_limit import get_named_limiter
from app.adapters.tts import silent_wav
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.observability import usage
from app.observability.cost import unit_cost

log = get_logger(__name__)


class _DailyExhausted(TransientProviderError):
    pass


class LyriaMusicAdapter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = make_client(self.settings, timeout_seconds=300)
        return self._client

    def compose(self, prompt: str) -> tuple[bytes, str, str]:
        """Returns (audio bytes, file extension, model used). Mock: a short silent WAV."""
        if self.settings.mock_mode:
            return silent_wav(2.0), "wav", "mock"
        last: Exception | None = None
        for model in self.settings.music_model_chain:
            try:
                data = Retrying(
                    retry=retry_if_exception(
                        lambda e: isinstance(e, TransientProviderError) and not isinstance(e, _DailyExhausted)
                    ),
                    stop=stop_after_attempt(3),
                    wait=wait_retry_after(wait_exponential(min=2, max=20), cap=20.0),
                    reraise=True,
                )(self._compose_once, model, prompt)
                return data, _ext(data), model
            except (_DailyExhausted, PermanentProviderError) as exc:
                log.warning("music model %s unavailable (%s); trying the next", model, exc)
                last = exc
        raise TransientProviderError(f"no music model available: {last}")

    def _compose_once(self, model: str, prompt: str) -> bytes:
        get_named_limiter(f"music:{model}", 100).acquire()
        try:
            it = interactions_create(self.client, model=model, input=prompt)
        except Exception as exc:  # noqa: BLE001
            if is_daily_quota(exc):
                raise _DailyExhausted(str(exc)) from exc
            if _is_transient(exc):
                raise TransientProviderError(str(exc), retry_after=parse_retry_after(exc)) from exc
            raise PermanentProviderError(str(exc)) from exc
        data = b64bytes(getattr(getattr(it, "output_audio", None), "data", None))
        if not data:
            raise PermanentProviderError("Lyria returned no audio")
        usage.record_spend(category="music", model=model, cost_usd=unit_cost(model), units={"songs": 1})
        return data


def _ext(data: bytes) -> str:
    return "wav" if data[:4] == b"RIFF" else "mp3"
