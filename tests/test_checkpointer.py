"""Checkpointer factory: a checkpointed graph round-trips AND the state survives
a fresh connection (proves the FILE-backed sqlite saver, not an ephemeral
:memory: one that would lose the checkpoint POST /resume needs)."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.core.config import get_settings
from app.graph.checkpoint import make_checkpointer


class _S(TypedDict, total=False):
    n: int


async def _inc(s: _S) -> _S:
    return {"n": s.get("n", 0) + 1}


def _graph(checkpointer):
    g = StateGraph(_S)
    g.add_node("inc", _inc)
    g.add_edge(START, "inc")
    g.add_edge("inc", END)
    return g.compile(checkpointer=checkpointer)


async def test_checkpointer_persists_across_connections():
    settings = get_settings()
    cfg = {"configurable": {"thread_id": "cp-roundtrip"}}

    async with make_checkpointer(settings) as cp:
        out = await _graph(cp).ainvoke({"n": 0}, cfg)
        assert out["n"] == 1

    # Fresh connection (new make_checkpointer context) still sees the checkpoint.
    async with make_checkpointer(settings) as cp2:
        snapshot = await _graph(cp2).aget_state(cfg)
        assert snapshot.values.get("n") == 1
        assert snapshot.next == ()  # finished thread
