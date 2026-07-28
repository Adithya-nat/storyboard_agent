"""Phase 3 reliability: adapter re-roll vs block, and image_gen neighbor-fill /
idempotency / bounded concurrency. These are the fixes that eliminate the silent
'degraded frame' gaps without ever shipping an ugly placeholder to the user.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from app.adapters.errors import (
    EmptyImageError,
    PermanentProviderError,
)
from app.adapters.gemini import GeminiImageAdapter
from app.adapters.llm import LLMAdapter
from app.adapters.rate_limit import TokenBucket, reset_image_limiter
from app.agents.image_gen import image_gen
from app.artifacts.store import ArtifactStore
from app.core.config import Settings
from app.graph.state import Deps
from app.schemas import Shot, Shotlist


# --------------------------------------------------------------------------- #
# Fake google-genai response/client
# --------------------------------------------------------------------------- #
def _resp(*, image: bytes | None = None, finish=None, block=None):
    parts = [SimpleNamespace(inline_data=SimpleNamespace(data=image))] if image else []
    candidate = SimpleNamespace(content=SimpleNamespace(parts=parts), finish_reason=finish)
    return SimpleNamespace(candidates=[candidate], prompt_feedback=SimpleNamespace(block_reason=block))


class _FakeModels:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def generate_content(self, **_kw):
        r = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return r


def _adapter(responses, **overrides) -> GeminiImageAdapter:
    reset_image_limiter()
    settings = Settings(
        gemini_api_key="fake-key",          # disable mock_mode -> real codepath
        use_mock_providers=False,
        image_gen_max_retries=overrides.get("retries", 4),
        image_rpm_limit=100_000,            # effectively unthrottled in tests
        image_aspect_ratio="9:16",
    )
    a = GeminiImageAdapter(settings)
    a._client = SimpleNamespace(models=_FakeModels(responses))
    return a


# --------------------------------------------------------------------------- #
# Adapter: classification (fast, no retry sleep)
# --------------------------------------------------------------------------- #
def test_generate_once_returns_image_bytes():
    a = _adapter([_resp(image=b"PNG", finish="STOP")])
    assert a._generate_once("p", None, None) == b"PNG"


def test_empty_no_image_is_retryable():
    a = _adapter([_resp(finish="NO_IMAGE")])
    with pytest.raises(EmptyImageError):
        a._generate_once("p", None, None)


def test_safety_block_is_permanent():
    a = _adapter([_resp(finish="IMAGE_SAFETY")])
    with pytest.raises(PermanentProviderError):
        a._generate_once("p", None, None)


def test_prompt_block_reason_is_permanent():
    a = _adapter([_resp(finish="STOP", block="PROHIBITED_CONTENT")])
    with pytest.raises(PermanentProviderError):
        a._generate_once("p", None, None)


# --------------------------------------------------------------------------- #
# Adapter: retry integration (re-roll an empty, then succeed)
# --------------------------------------------------------------------------- #
def test_generate_image_rerolls_empty_then_succeeds(monkeypatch):
    # No real backoff sleeps in the test.
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda *_a, **_k: None)
    a = _adapter([_resp(finish="NO_IMAGE"), _resp(image=b"PNG2", finish="STOP")])
    out = a.generate_image("safe prompt")
    assert out == b"PNG2"
    assert a._client.models.calls == 2  # re-rolled exactly once


def test_generate_image_block_does_not_reroll(monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda *_a, **_k: None)
    a = _adapter([_resp(finish="IMAGE_SAFETY"), _resp(image=b"NEVER")])
    with pytest.raises(PermanentProviderError):
        a.generate_image("blocked prompt")
    assert a._client.models.calls == 1  # gave up immediately, no wasted re-roll


# --------------------------------------------------------------------------- #
# Token bucket: provably bounds rate
# --------------------------------------------------------------------------- #
def test_token_bucket_paces_calls():
    # 600 rpm = 10/sec, tiny capacity -> the 5th token must wait ~ (5-cap)/rate.
    tb = TokenBucket(600, capacity=1)
    start = time.monotonic()
    for _ in range(5):
        assert tb.acquire(timeout=5)
    elapsed = time.monotonic() - start
    assert elapsed >= 0.3  # 4 refills @ 0.1s each, minus the initial token


# --------------------------------------------------------------------------- #
# image_gen: neighbor-fill, idempotency, bounded concurrency
# --------------------------------------------------------------------------- #
def _deps(artifact_root, job_id, image, *, concurrency=4) -> Deps:
    settings = Settings(
        gemini_api_key="",
        use_mock_providers=True,
        image_max_concurrency=concurrency,
    )
    return Deps(
        settings=settings,
        store=ArtifactStore(artifact_root, job_id),
        llm=LLMAdapter(settings),
        image=image,
    )


def _save_shotlist(store, n) -> str:
    shotlist = Shotlist(
        style_bible="bible",
        shots=[Shot(order=i, image_prompt=f"p{i}", narration=f"caption {i}") for i in range(1, n + 1)],
    )
    return store.save_json("shotlist.json", shotlist)


class _FailOnPrompt:
    """Succeeds with unique per-prompt bytes, except prompts in `fail`."""

    def __init__(self, fail: set[str]):
        self.fail = fail
        self.called: list[str] = []

    def generate_image(self, prompt, **_k):
        self.called.append(prompt)
        if prompt in self.fail:
            raise PermanentProviderError("blocked")
        return f"IMG:{prompt}".encode()


async def test_neighbor_fill_reuses_clean_neighbor_no_placeholder(artifact_root):
    img = _FailOnPrompt(fail={"p3"})
    deps = _deps(artifact_root, "nf", img)
    path = _save_shotlist(deps.store, 5)
    out = await image_gen({"shotlist_path": path, "style_ref_path": None, "duration": 30}, deps)

    frames = {f["order"]: f for f in out["frames"]}
    assert len(frames) == 5
    # Frame 3 failed -> filled from nearest clean neighbour (order 2, tie-break lower).
    assert frames[3]["status"] == "degraded"
    assert deps.store.load_bytes(frames[3]["path"]) == b"IMG:p2"  # a real neighbour, NOT a card
    # Everyone else is a clean success.
    assert all(frames[o]["status"] == "ok" for o in (1, 2, 4, 5))
    assert any("frame 3" in w for w in out["warnings"])


async def test_idempotent_skip_keeps_clean_frame_and_does_not_recall(artifact_root):
    img = _FailOnPrompt(fail={"p2"})  # would fail p2 if called
    deps = _deps(artifact_root, "idem", img)
    # Pre-place a clean frame_02 with NO degraded marker -> must be skipped.
    deps.store.save_bytes("frame_02.png", b"PREEXISTING")
    path = _save_shotlist(deps.store, 3)
    out = await image_gen({"shotlist_path": path, "style_ref_path": None, "duration": 30}, deps)

    frames = {f["order"]: f for f in out["frames"]}
    assert frames[2]["status"] == "ok"            # reused, not degraded
    assert "p2" not in img.called                 # never re-billed
    assert deps.store.load_bytes(frames[2]["path"]) == b"PREEXISTING"


class _SlowCountingImage:
    """Tracks peak concurrent in-flight calls across worker threads."""

    def __init__(self):
        self.current = 0
        self.peak = 0
        self._lock = threading.Lock()

    def generate_image(self, prompt, **_k):
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)
        time.sleep(0.03)
        with self._lock:
            self.current -= 1
        return f"IMG:{prompt}".encode()


async def test_concurrency_never_exceeds_limit(artifact_root):
    img = _SlowCountingImage()
    deps = _deps(artifact_root, "conc", img, concurrency=3)
    path = _save_shotlist(deps.store, 12)
    out = await image_gen({"shotlist_path": path, "style_ref_path": None, "duration": 60}, deps)

    assert len(out["frames"]) == 12
    assert all(f["status"] == "ok" for f in out["frames"])
    assert img.peak <= 3  # semaphore held the line
    assert img.peak > 1   # but it WAS concurrent (not accidentally serialized)
