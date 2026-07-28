"""Issue 1 (TTS rate-limit fix): the shared limiter registry + the retry-after
back-off helpers that keep per-beat narration under Gemini's hard 10 RPM cap.

    get_tts_limiter (capacity=1)  ──▶ paces calls UNDER the per-minute cap
    parse_retry_after / wait_retry_after  ──▶ on a 429, wait the server's own
                                              retryDelay (capped), not a guess
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from app.adapters.errors import TransientProviderError
from app.adapters.llm import parse_retry_after, wait_retry_after
from app.adapters.rate_limit import (
    get_image_limiter,
    get_tts_limiter,
    reset_image_limiter,
    reset_tts_limiter,
)


# --------------------------------------------------------------------------- #
# Named-limiter registry: TTS is a SEPARATE ceiling, capacity=1 (no burst),
# and the image limiter's public API + default capacity are unchanged.
# --------------------------------------------------------------------------- #
def test_tts_limiter_capacity_one_and_separate_from_image():
    reset_tts_limiter()
    reset_image_limiter()
    tts = get_tts_limiter(8)
    img = get_image_limiter(60)
    assert tts._capacity == 1.0  # explicit no-burst
    assert tts is not img  # different per-model quotas -> different buckets
    assert get_tts_limiter(8) is tts  # cached while rpm unchanged
    assert get_tts_limiter(9) is not tts  # rebuilt on rpm change


def test_image_limiter_default_capacity_preserved():
    reset_image_limiter()
    img = get_image_limiter(60)
    # Original single-bucket behaviour: capacity defaults to max(1, rpm/10).
    assert img._capacity == 6.0
    assert get_image_limiter(60) is img


def test_tts_limiter_paces_calls():
    reset_tts_limiter()
    tb = get_tts_limiter(600)  # 10 tokens/sec, capacity 1
    start = time.monotonic()
    for _ in range(5):
        assert tb.acquire(timeout=5)
    # First token is free (capacity 1); the next 4 wait ~0.1s each.
    assert time.monotonic() - start >= 0.3


# --------------------------------------------------------------------------- #
# parse_retry_after: pull the server's hint out of BOTH 429 string shapes.
# --------------------------------------------------------------------------- #
def test_parse_retry_after_message_form():
    exc = Exception("You exceeded your quota. Please retry in 3.017691049s.")
    assert parse_retry_after(exc) == pytest.approx(3.017691049)


def test_parse_retry_after_retrydelay_detail_form():
    exc = Exception("...RetryInfo', 'retryDelay': '7s'}]}}")
    assert parse_retry_after(exc) == 7.0


def test_parse_retry_after_prefers_precise_message_value():
    # Real 429s carry BOTH; .search finds the precise message value first.
    exc = Exception("Please retry in 3.017691049s. ... 'retryDelay': '3s'")
    assert parse_retry_after(exc) == pytest.approx(3.017691049)


def test_parse_retry_after_none_when_absent():
    assert parse_retry_after(Exception("500 internal, no hint")) is None


# --------------------------------------------------------------------------- #
# wait_retry_after: honor a capped hint, else fall back to the base strategy.
# --------------------------------------------------------------------------- #
def _retry_state(exc: Exception | None):
    outcome = SimpleNamespace(exception=lambda: exc)
    return SimpleNamespace(outcome=outcome)


def test_wait_uses_small_hint_verbatim():
    w = wait_retry_after(lambda _s: 99.0, cap=20.0)
    assert w(_retry_state(TransientProviderError("x", retry_after=3.0))) == 3.0


def test_wait_caps_a_huge_daily_quota_hint():
    # A 100 RPD exhaustion can return a giant delay; we must NOT sleep that long.
    w = wait_retry_after(lambda _s: 99.0, cap=20.0)
    assert w(_retry_state(TransientProviderError("x", retry_after=3600.0))) == 20.0


def test_wait_falls_back_without_a_hint():
    w = wait_retry_after(lambda _s: 42.0, cap=20.0)
    assert w(_retry_state(TransientProviderError("x"))) == 42.0  # retry_after None


def test_wait_falls_back_for_non_provider_exception():
    w = wait_retry_after(lambda _s: 42.0, cap=20.0)
    assert w(_retry_state(ValueError("unrelated"))) == 42.0  # no retry_after attr
