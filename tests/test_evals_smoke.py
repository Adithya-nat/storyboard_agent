"""Smoke test: the manual eval harness still runs end-to-end in mock mode.

Guards against bit-rot in evals/run.py (it's not in the normal CI path otherwise).
Uses a short duration so it's fast; asserts the schema/scaling heuristics pass.
"""

from __future__ import annotations

from app.core.config import Settings
from evals.run import evaluate


async def test_evals_harness_runs_in_mock_mode(tmp_path):
    settings = Settings(use_mock_providers=True)
    res = await evaluate("a tiny test idea about a curious fox", 6, tmp_path, settings)

    checks = {name: ok for name, ok, _ in res["checks"]}
    # 6s @ 2s/frame -> 3 frames; schema + scaling + caption checks must pass.
    # (video isn't asserted so the smoke test stays portable to ffmpeg-less envs.)
    assert checks["script"]
    assert checks["shotlist"]
    assert checks["frames"]
    assert checks["captions"]
