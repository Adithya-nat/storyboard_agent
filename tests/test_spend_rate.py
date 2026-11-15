"""Spend-rate governor: pace calls under the rolling 10-minute spend limit."""

from __future__ import annotations

import pytest

from app.adapters.llm import parse_retry_after
from app.observability.spend_rate import SpendRateGovernor, is_spend_rate_limited

SPEND_429 = Exception(
    "429 RESOURCE_EXHAUSTED. {'message': \"You exceeded your spend-based rate limit. Your spending rate "
    "has exceeded the allowed limit for your account's billing history and tier.\"}"
)


def test_admits_until_limit_then_waits_for_window(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr("app.observability.spend_rate.time.monotonic", lambda: clock["t"])
    g = SpendRateGovernor(10.0, window=600)
    assert g.acquire(4) == 0 and g.acquire(4) == 0 and g.spent() == 8
    slept = []

    def fake_sleep(s):
        slept.append(s)
        clock["t"] += s

    waited = g.acquire(4, sleep=fake_sleep)  # 12 > 10: must wait for the first $4 to age out
    assert waited >= 600 and slept
    assert g.spent() <= 10


def test_oversized_estimate_admitted_into_empty_window():
    g = SpendRateGovernor(5.0)
    assert g.acquire(7.0) < 0.01  # a single call bigger than the limit still runs alone


def test_settle_refunds_unbilled_reservation():
    g = SpendRateGovernor(10.0)
    g.acquire(6.0)
    g.settle(6.0, 0.0)
    assert g.spent() == pytest.approx(0.0)


def test_spend_rate_429_detected_and_backs_off_a_minute():
    assert is_spend_rate_limited(SPEND_429)
    assert parse_retry_after(SPEND_429) == 60.0


def test_saturate_blocks_until_the_window_rolls_over():
    from app.observability.spend_rate import SpendRateGovernor

    clock = {"t": 0.0}
    g = SpendRateGovernor(9.0, window=600)
    import app.observability.spend_rate as sr

    real = sr.time.monotonic
    sr.time.monotonic = lambda: clock["t"]
    try:
        g.record(1.0)
        g.saturate()
        assert g.spent() == 9.0
        slept = []

        def fake_sleep(s):
            slept.append(s)
            clock["t"] += s

        g.acquire(1.0, sleep=fake_sleep)
        assert clock["t"] >= 600  # waited for the window, not forever
    finally:
        sr.time.monotonic = real
