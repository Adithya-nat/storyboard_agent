"""Text-LLM adapter (Gemini). Returns schema-validated pydantic objects.

All provider specifics + retry/backoff + JSON re-ask live here ONCE; agents call
``complete_json`` and get a typed object back. In mock mode (no API key) the
agent-supplied ``mock_factory`` produces a deterministic object so the pipeline
runs offline.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.adapters.errors import PermanentProviderError, TransientProviderError
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.observability import usage

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("429", "resource_exhausted", "rate", "503", "500", "unavailable"))


# Gemini 429s carry their own back-off hint in TWO spots of the error string:
# the human message ("Please retry in 3.017691049s") and the structured
# RetryInfo detail ("'retryDelay': '3s'"). Match either; .search returns the
# first (the precise message value), so we wait what the server actually asked.
_RETRY_AFTER_RE = re.compile(
    r"(?:retry in\s+|retryDelay['\"]?\s*[:=]\s*['\"]?)(\d+(?:\.\d+)?)\s*s",
    re.IGNORECASE,
)


def parse_retry_after(exc: Exception) -> float | None:
    """Pull the provider's suggested back-off (seconds) out of a 429, or None."""
    m = _RETRY_AFTER_RE.search(str(exc))
    if not m:
        return None
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return None


def wait_retry_after(fallback: Callable[[object], float], *, cap: float) -> Callable[[object], float]:
    """tenacity ``wait`` strategy: honor a ``TransientProviderError.retry_after``
    hint (CAPPED) when the provider gave one, else defer to ``fallback``.

    Capping is load-bearing, not cosmetic: a DAILY-quota 429 (100 RPD) can carry
    a huge ``retryDelay`` — we must NOT sleep that long inside a running job, so
    we cap the honored wait low enough that a daily exhaustion exhausts retries
    quickly and the caller degrades (silent beat) instead of hanging for minutes.
    """

    def _wait(retry_state) -> float:
        outcome = getattr(retry_state, "outcome", None)
        exc = outcome.exception() if outcome is not None else None
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is not None and retry_after >= 0:
            return min(float(retry_after), cap)
        return fallback(retry_state)

    return _wait


class LLMAdapter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None  # lazy

    @property
    def client(self):
        if self._client is None:
            from google import genai  # imported lazily so mock mode needs no key

            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        mock_factory: Callable[[str], T],
    ) -> T:
        if self.settings.mock_mode:
            log.info("LLMAdapter mock_mode -> %s", response_model.__name__)
            if self.settings.mock_token_usage:
                usage.record(total=self.settings.mock_token_usage)
            return mock_factory(user)
        return self._complete_json_real(system, user, response_model)

    @retry(
        retry=retry_if_exception_type(TransientProviderError),
        stop=stop_after_attempt(3),
        # Honor a 429's own retryDelay (capped) when present, else exponential.
        wait=wait_retry_after(wait_exponential(multiplier=1, min=2, max=20), cap=20.0),
        reraise=True,
    )
    def _complete_json_real(self, system: str, user: str, response_model: type[T]) -> T:
        try:
            resp = self.client.models.generate_content(
                model=self.settings.gemini_text_model,
                contents=f"{system}\n\n{user}",
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_model,
                },
            )
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            if _is_transient(exc):
                raise TransientProviderError(
                    str(exc), retry_after=parse_retry_after(exc)
                ) from exc
            raise PermanentProviderError(str(exc)) from exc

        usage.record_usage(getattr(resp, "usage_metadata", None))
        try:
            return response_model.model_validate_json(resp.text)
        except ValidationError as exc:
            # Bad JSON is permanent for this call; tenacity won't retry it.
            raise PermanentProviderError(
                f"LLM returned invalid {response_model.__name__}: {exc}"
            ) from exc
