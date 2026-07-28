"""Build the storyboard StateGraph.

    START ─▶ script_writer ─▶ reflection ─▶ designer ─▶ image_gen ─▶ assembler ─▶ END
              (LLM)           (LLM judge/   (LLM +       (Gemini,      (ffmpeg)
                               revise loop)  1 ref img)   sequential)

The `reflection` node (Workstream 3) critiques + revises the script for story
coherence BEFORE the look is locked; the loop is internal (no graph cycle). The
`designer` node (Phase 2) then locks one visual style (style_bible + style_ref)
between scripting and image generation. The shape stays strictly linear.

Deps (adapters, store) are bound into each node via functools.partial so the
graph state remains small + serializable (checkpoint-ready). Pass a `checkpointer`
to persist node-boundary state for resume (thread_id == job_id).

Observability: when ``deps.reporter`` is set, each node is wrapped by
``_instrument`` — it binds the (job_id, stage) contextvar (so every log.* inside
the node is auto-attributed) and emits stage start/finish/fail to ``job_stages``.
The wrapper is TRANSPARENT (returns the node's dict unchanged, re-raises
unchanged) and a pure pass-through when ``reporter is None``, so existing graph
builds + agent unit tests are unaffected.
"""

from __future__ import annotations

import functools

from langgraph.graph import END, START, StateGraph

from app.agents.assembler import assembler
from app.agents.designer import designer
from app.agents.image_gen import image_gen
from app.agents.motion import motion
from app.agents.reflection import reflection
from app.agents.script_writer import script_writer
from app.core.config import Settings
from app.graph.state import Deps, StoryboardState
from app.observability.context import bind_stage, reset_stage


def pipeline_stage_names(settings: Settings) -> list[str]:
    """The ordered pipeline stage names — the SINGLE source of truth shared by the
    graph builder (below) and the API (so the front end's stage list can never
    disagree with what actually runs). The ``motion`` stage exists only when the
    external motion engine is enabled; otherwise the pipeline is exactly today's."""
    names = ["script_writer", "reflection", "designer", "image_gen"]
    if settings.motion_engine_enabled:
        names.append("motion")
    names.append("assembler")
    return names


# Name -> node function. Resolved as module globals at call time so tests that
# monkeypatch e.g. app.graph.build.image_gen (or .motion) win.
def _node_for(name: str):
    return {
        "script_writer": script_writer,
        "reflection": reflection,
        "designer": designer,
        "image_gen": image_gen,
        "motion": motion,
        "assembler": assembler,
    }[name]


def _instrument(name: str, seq: int, fn, reporter, deps=None):
    """Wrap a bound node with stage transitions + contextvar binding.

    No reporter => return ``fn`` unchanged (zero overhead, identical behaviour).
    ``deps`` is optional (isolated wrapper tests omit it); only the QA
    failure-injection toggle reads it.
    """
    if reporter is None:
        return fn

    async def wrapped(state):
        job_id = state["job_id"]
        token = bind_stage(job_id, name)  # log attribution for this stage
        try:
            await reporter.stage_start(job_id, name, seq)
            # QA toggle: raise inside the named stage so the failed + Resume UI is
            # testable (mock mode never fails on its own).
            if deps is not None and deps.settings.mock_force_stage_failure == name:
                raise RuntimeError(
                    f"forced failure at stage '{name}' (QA: MOCK_FORCE_STAGE_FAILURE)"
                )
            result = await fn(state)
            # Flip running -> succeeded ONLY if the node didn't already set a
            # terminal status (image_gen may mark itself 'degraded').
            await reporter.stage_finish(job_id, name)
            return result
        except Exception as exc:
            await reporter.stage_fail(job_id, name, exc)
            raise  # re-raise unchanged so the runner still sets the job FAILED
        finally:
            # Persist this stage's token total live (best-effort), then clear the stage
            # binding so attribution can't bleed into the next node.
            await reporter.record_stage_tokens(job_id, name)
            reset_stage(token)

    return wrapped


def build_graph(deps: Deps, checkpointer=None):
    g = StateGraph(StoryboardState)
    reporter = deps.reporter
    # Pipeline order -> the `seq` column (front end sorts stages by it). Derived
    # from pipeline_stage_names(settings) so the graph and the API agree, and the
    # `motion` stage is present iff the motion engine is enabled. Node names
    # resolve as module globals at call time (tests monkeypatch app.graph.build.*).
    names = pipeline_stage_names(deps.settings)
    for seq, name in enumerate(names, start=1):
        bound = functools.partial(_node_for(name), deps=deps)
        g.add_node(name, _instrument(name, seq, bound, reporter, deps))

    g.add_edge(START, names[0])
    for prev, nxt in zip(names, names[1:]):
        g.add_edge(prev, nxt)
    g.add_edge(names[-1], END)
    return g.compile(checkpointer=checkpointer)
