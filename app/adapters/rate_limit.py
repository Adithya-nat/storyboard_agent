"""Process-global, thread-safe token-bucket rate limiter for image API calls.

A single shared bucket bounds the TOTAL request rate across everything that hits
the image model — every frame, the designer's style reference, AND every
retry/re-roll — because each caller must ``acquire()`` a token before its HTTP
request. This is the one load-bearing RPM guard:

    asyncio.Semaphore (in image_gen)  ──▶  caps how many calls run AT ONCE
    TokenBucket (here, shared)        ──▶  caps how many calls run PER MINUTE

So even when re-rolls + concurrency stack up we provably stay under the model's
RPM (Nano Banana 2 = 100; we run a default ceiling of 60).

Thread-safe (``threading`` primitives, not asyncio) because the sync Gemini
adapter runs inside ``asyncio.to_thread`` worker threads — an asyncio limiter
would be acquired in the wrong context.

    tokens refill continuously at rate/60 per second, capped at `capacity`
    acquire(): refill → if enough, take & return → else sleep the shortfall, retry
"""

from __future__ import annotations

import threading
import time


class TokenBucket:
    """Classic token bucket. ``rate_per_minute`` tokens trickle in continuously;
    a short burst up to ``capacity`` is allowed, then callers are paced."""

    def __init__(self, rate_per_minute: float, *, capacity: float | None = None) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be > 0")
        self._rate = rate_per_minute / 60.0  # tokens per second
        # Default burst = ~6s of throughput (>=1), so we never serialize hard but
        # also never dump a big spike at the ceiling.
        self._capacity = capacity if capacity is not None else max(1.0, rate_per_minute / 10.0)
        self._tokens = self._capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._updated
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._updated = now

    def acquire(self, tokens: float = 1.0, *, timeout: float | None = None) -> bool:
        """Block until ``tokens`` are available (or ``timeout`` elapses).

        Returns True when tokens were granted, False on timeout. Sleeps in small
        slices so it stays responsive and never deadlocks.
        """
        if tokens > self._capacity:
            # Asking for more than the bucket can ever hold would block forever.
            tokens = self._capacity
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                wait = (tokens - self._tokens) / self._rate
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                wait = min(wait, remaining)
            time.sleep(min(max(wait, 0.0), 1.0))


# --- Process-global named registry (one ceiling per call-class) ------------- #
# Each model family (image, tts) hits a SEPARATE per-model quota, so each gets its
# own bucket. Keyed by name; (re)created when its rpm or capacity changes.
_lock = threading.Lock()
_buckets: dict[str, tuple[TokenBucket, float, float | None]] = {}


def _get_or_create(name: str, rate_per_minute: float, *, capacity: float | None = None) -> TokenBucket:
    with _lock:
        existing = _buckets.get(name)
        if existing is None or existing[1] != rate_per_minute or existing[2] != capacity:
            bucket = TokenBucket(rate_per_minute, capacity=capacity)
            _buckets[name] = (bucket, rate_per_minute, capacity)
            return bucket
        return existing[0]


def _reset(name: str) -> None:
    with _lock:
        _buckets.pop(name, None)


def get_image_limiter(rate_per_minute: float) -> TokenBucket:
    """Return the shared image-call limiter, (re)created if the RPM changed.

    capacity defaults (=max(1, rpm/10)) so a short burst is allowed, unchanged
    from the original single-bucket behaviour (Nano Banana 2 = 100 RPM)."""
    return _get_or_create("image", rate_per_minute)


def reset_image_limiter() -> None:
    """Drop the shared image limiter (tests: isolate rate state between cases)."""
    _reset("image")


def get_tts_limiter(rate_per_minute: float) -> TokenBucket:
    """Return the shared TTS-call limiter — a SEPARATE ceiling from image (TTS is
    its own 10 RPM per-model quota).

    ``capacity=1.0`` on PURPOSE (no burst): a burst can trip a fixed-window
    10/min quota even when the average is safe, and throughput is irrelevant for
    a background narration job — so we err all the way toward margin and pace
    strictly one call at a time."""
    return _get_or_create("tts", rate_per_minute, capacity=1.0)


def reset_tts_limiter() -> None:
    """Drop the shared TTS limiter (tests: isolate rate state between cases)."""
    _reset("tts")
