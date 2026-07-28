"""The node wrapper (graph/build.py::_instrument) transparency contract.

The wrapper must be invisible to the pipeline's behaviour: same return value, same
exception, and a literal pass-through when there's no reporter (so the existing 64
tests + agent unit tests are unaffected). It also binds the stage contextvar
during the node and resets it after (no attribution bleed).
"""

from __future__ import annotations

import pytest

from app.graph.build import _instrument
from app.observability.context import current


class _FakeReporter:
    def __init__(self) -> None:
        self.calls: list = []

    async def stage_start(self, job_id, name, seq):
        self.calls.append(("start", job_id, name, seq))

    async def stage_finish(self, job_id, name):
        self.calls.append(("finish", job_id, name))

    async def stage_fail(self, job_id, name, exc):
        self.calls.append(("fail", job_id, name, str(exc)))

    async def record_stage_tokens(self, job_id, name):
        self.calls.append(("tokens", job_id, name))


def test_passthrough_is_identity_when_no_reporter():
    async def node(state):
        return {"ok": True}

    assert _instrument("x", 1, node, reporter=None) is node  # zero overhead


async def test_returns_unchanged_and_emits_start_finish():
    rep = _FakeReporter()

    async def node(state):
        return {"result": 42}

    wrapped = _instrument("image_gen", 3, node, rep)
    out = await wrapped({"job_id": "j1"})
    assert out == {"result": 42}
    assert ("start", "j1", "image_gen", 3) in rep.calls
    assert ("finish", "j1", "image_gen") in rep.calls
    assert ("tokens", "j1", "image_gen") in rep.calls  # live per-stage token write


async def test_reraises_and_marks_failed():
    rep = _FakeReporter()

    async def node(state):
        raise ValueError("kaboom")

    wrapped = _instrument("designer", 2, node, rep)
    with pytest.raises(ValueError, match="kaboom"):
        await wrapped({"job_id": "j1"})
    assert ("fail", "j1", "designer", "kaboom") in rep.calls
    assert not any(c[0] == "finish" for c in rep.calls)  # never finished on failure
    assert ("tokens", "j1", "designer") in rep.calls  # finally still records spend


async def test_binds_contextvar_during_node_and_resets_after():
    rep = _FakeReporter()
    seen = {}

    async def node(state):
        ctx = current()
        seen["during"] = None if ctx is None else (ctx.job_id, ctx.stage)
        return {}

    wrapped = _instrument("image_gen", 3, node, rep)
    await wrapped({"job_id": "jZ"})
    assert seen["during"] == ("jZ", "image_gen")
    assert current() is None  # reset in finally, even on the success path


async def test_contextvar_reset_even_on_exception():
    rep = _FakeReporter()

    async def node(state):
        raise RuntimeError("x")

    wrapped = _instrument("assembler", 4, node, rep)
    with pytest.raises(RuntimeError):
        await wrapped({"job_id": "jE"})
    assert current() is None  # finally ran despite the raise
