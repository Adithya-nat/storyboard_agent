"""Process-wide spend-rate governor: keeps rolling 10-minute spend under the provider's
spend-based rate limit (Gemini Tier 1 = $10 / 10 min, Tier 2+ = $200) by pacing expensive
calls instead of letting them fail with 429s."""

from __future__ import annotations

import threading
import time
from collections import deque

WINDOW_SECONDS = 600.0


class SpendRateGovernor:
    def __init__(self, limit_usd: float, window: float = WINDOW_SECONDS) -> None:
        self.limit = limit_usd
        self.window = window
        self._events: deque[tuple[float, float]] = deque()  # (monotonic time, usd)
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] > self.window:
            self._events.popleft()

    def spent(self) -> float:
        with self._lock:
            self._prune(time.monotonic())
            return sum(c for _, c in self._events)

    def acquire(self, estimate_usd: float, *, timeout: float = 1800.0, sleep=time.sleep) -> float:
        """Block until ``estimate_usd`` fits in the window, then reserve it. Returns seconds waited.
        An estimate larger than the whole limit is admitted once the window is empty."""
        if self.limit <= 0 or estimate_usd <= 0:
            return 0.0
        start = time.monotonic()
        while True:
            with self._lock:
                now = time.monotonic()
                self._prune(now)
                used = sum(c for _, c in self._events)
                if used + estimate_usd <= self.limit or not self._events:
                    self._events.append((now, estimate_usd))
                    return now - start
                wait = self.window - (now - self._events[0][0]) + 0.5
            if time.monotonic() - start + wait > timeout:
                raise TimeoutError("spend-rate window did not free up in time")
            sleep(min(wait, 30.0))

    def settle(self, estimate_usd: float, actual_usd: float) -> None:
        """Replace a reservation with the actual cost (no-op when equal)."""
        delta = actual_usd - estimate_usd
        if abs(delta) > 1e-9:
            with self._lock:
                self._events.append((time.monotonic(), delta))

    def saturate(self) -> None:
        """The provider says the window is full: block new spend until it rolls over."""
        with self._lock:
            used = sum(c for _, c in self._events)
            self._events.append((time.monotonic(), max(0.0, self.limit - used)))

    def record(self, usd: float) -> None:
        if usd > 0:
            with self._lock:
                self._events.append((time.monotonic(), usd))


_governor: SpendRateGovernor | None = None
_glock = threading.Lock()


def get_governor(limit_usd: float) -> SpendRateGovernor:
    global _governor
    with _glock:
        if _governor is None or _governor.limit != limit_usd:
            _governor = SpendRateGovernor(limit_usd)
        return _governor


def is_spend_rate_limited(exc: BaseException) -> bool:
    return "spend-based rate limit" in str(exc).lower() or "spend rate limit" in str(exc).lower()
