"""The 'motion' stage exists in the graph + pipeline names ONLY when enabled."""

from __future__ import annotations

from types import SimpleNamespace

from app.core.config import Settings
from app.graph.build import build_graph, pipeline_stage_names
from app.graph.state import Deps

_BASE = ["brief", "script_writer", "dialogue_pass", "reflection", "designer", "image_gen", "narration", "captions", "music", "assembler", "film_review", "publish_kit"]


def _deps(enabled: bool) -> Deps:
    return Deps(
        settings=Settings(motion_engine_enabled=enabled, use_mock_providers=True),
        store=SimpleNamespace(), llm=SimpleNamespace(), image=SimpleNamespace(),
        tts=None, reporter=None, motion=None,
    )


def test_pipeline_names_default_excludes_motion():
    assert pipeline_stage_names(Settings(motion_engine_enabled=False)) == _BASE


def test_pipeline_names_enabled_inserts_motion_after_narration():
    names = pipeline_stage_names(Settings(motion_engine_enabled=True))
    assert names == _BASE[:7] + ["motion"] + _BASE[7:]
    # Narration sets clip lengths before motion; captions needs the clips.
    assert names.index("narration") < names.index("motion") < names.index("captions")


def test_graph_has_motion_node_when_enabled():
    nodes = set(build_graph(_deps(True)).get_graph().nodes)
    assert "motion" in nodes


def test_graph_omits_motion_by_default():
    nodes = set(build_graph(_deps(False)).get_graph().nodes)
    assert "motion" not in nodes
