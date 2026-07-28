"""Shared adapter exceptions, so agents can catch provider-agnostic errors."""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for any external-provider failure."""


class TransientProviderError(ProviderError):
    """Retryable: rate limit (429), 5xx, network blip.

    ``retry_after`` (seconds) carries the provider's OWN back-off hint when it
    gives one — Gemini's 429 ``RetryInfo.retryDelay`` ("Please retry in 3.017s").
    The retry policy waits EXACTLY that long instead of guessing, so we stop
    hammering a per-minute quota. ``None`` when the provider offered no hint.
    Keyword-only + defaulted so ``TransientProviderError(str(exc))`` and the
    ``EmptyImageError(...)`` subclass keep constructing with just a message.
    """

    def __init__(self, *args: object, retry_after: float | None = None) -> None:
        super().__init__(*args)
        self.retry_after = retry_after


class EmptyImageError(TransientProviderError):
    """The image model returned NO image part, but it was NOT a genuine block
    (e.g. finish_reason NO_IMAGE / STOP / IMAGE_OTHER). Image models stochastically
    answer with text-only on a perfectly safe prompt; a re-roll almost always
    yields a real image, so this is treated as retryable (subclasses Transient)."""


class PermanentProviderError(ProviderError):
    """Not retryable: safety block, invalid request, bad output."""
