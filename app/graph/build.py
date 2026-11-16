"""Build the storyboard StateGraph (linear; stage order from pipeline_stage_names).

    brief -> script_writer -> dialogue_pass -> reflection -> designer -> image_gen -> narration -> [motion]
      -> captions -> music -> assembler -> publish_kit

Each node is wrapped by ``_instrument`` (stage status + log attribution) when a
reporter is set; deps are bound with functools.partial so state stays serializable.
"""

from __future__ import annotations

import functools

from langgraph.graph import END, START, StateGraph

from app.agents.assembler import assembler
from app.agents.brief import brief
from app.agents.dialogue_pass import dialogue_pass
from app.agents.film_review import film_review
from app.agents.gloss import ensure_for_script
from app.agents.captions import captions
from app.agents.designer import designer
from app.agents.image_gen import image_gen
from app.agents.motion import motion
from app.agents.music import music
from app.agents.narration import narration
from app.agents.publish_kit import publish_kit
from app.agents.reflection import reflection
from app.agents.script_writer import script_writer
from app.agents.video_gen import video_gen
from app.core.config import Settings
from app.graph.state import Deps, StoryboardState
from app.observability.context import bind_stage, reset_stage
from app.workers.control import JobPaused


def uses_video_stage(settings: Settings) -> bool:
    """AI video (Omni) is the default; the legacy engine only when enabled."""
    provider = settings.video_provider
    return provider == "omni" or (provider == "engine" and settings.motion_engine_enabled)


def pipeline_stage_names(settings: Settings) -> list[str]:
    """Ordered stage names — shared by the graph and the API. ``motion`` exists only
    when the motion engine is enabled."""
    names = ["brief", "script_writer", "dialogue_pass", "reflection", "designer", "image_gen", "narration"]
    if uses_video_stage(settings):
        names.append("motion")
    names += ["captions", "music", "assembler", "film_review", "publish_kit"]
    return names


# Resolved at call time so tests that monkeypatch app.graph.build.<node> win.
def _node_for(name: str):
    return {
        "brief": brief,
        "script_writer": script_writer,
        "dialogue_pass": dialogue_pass,
        "reflection": reflection,
        "designer": designer,
        "image_gen": image_gen,
        "narration": narration,
        "motion": motion,
        "captions": captions,
        "music": music,
        "assembler": assembler,
        "film_review": film_review,
        "publish_kit": publish_kit,
    }[name]


# Run after a stage succeeds, inside it (so their cost is attributed to it). English
# renderings of a Tamil film's text: once the script is final, and again after the
# video stage, whose content-policy rewrites can change a line.
_POST_HOOKS = {"reflection": ensure_for_script, "motion": ensure_for_script}


def _instrument(name: str, seq: int, fn, reporter, deps=None):
    """Wrap a bound node with stage transitions + contextvar binding.

    No reporter => return ``fn`` unchanged (zero overhead, identical behaviour).
    ``deps`` is optional (isolated wrapper tests omit it); only the QA
    failure-injection toggle reads it.
    """
    if reporter is None:
        return fn

    control = getattr(deps, "control", None) if deps is not None else None

    async def wrapped(state):
        job_id = state["job_id"]
        token = bind_stage(job_id, name)  # log attribution for this stage
        try:
            # Stage boundary: honour a breakpoint on this stage, or a pause that landed
            # after the previous stage's last unit, BEFORE anything here spends.
            if control is not None:
                if await control.take_breakpoint(name):
                    raise JobPaused(name, "Stopped at your checkpoint, before this stage", breakpoint=True)
                if await control.requested():
                    raise JobPaused(name, "Paused before this stage started")
            await reporter.stage_start(job_id, name, seq)
            # QA toggle: raise inside the named stage so the failed + Resume UI is
            # testable (mock mode never fails on its own).
            if deps is not None and deps.settings.mock_force_stage_failure == name:
                raise RuntimeError(
                    f"forced failure at stage '{name}' (QA: MOCK_FORCE_STAGE_FAILURE)"
                )
            result = await fn(state)
            if deps is not None and name in _POST_HOOKS:
                await _POST_HOOKS[name]({**state, **(result if isinstance(result, dict) else {})}, deps)
            # Flip running -> succeeded ONLY if the node didn't already set a
            # terminal status (image_gen may mark itself 'degraded').
            await reporter.stage_finish(job_id, name)
            # Live assets: expose what this stage produced right away (getattr:
            # minimal test reporters don't implement it).
            publish = getattr(reporter, "publish_artifacts", None)
            if publish is not None and isinstance(result, dict):
                await publish(job_id, {**state, **result})
            return result
        except JobPaused as p:
            await reporter.stage_pause(job_id, name, seq, p.message)
            raise  # the runner parks the job
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
        node = video_gen if name == "motion" and deps.settings.video_provider == "omni" else _node_for(name)
        bound = functools.partial(node, deps=deps)
        g.add_node(name, _instrument(name, seq, bound, reporter, deps))

    g.add_edge(START, names[0])
    for prev, nxt in zip(names, names[1:]):
        g.add_edge(prev, nxt)
    g.add_edge(names[-1], END)
    return g.compile(checkpointer=checkpointer)
