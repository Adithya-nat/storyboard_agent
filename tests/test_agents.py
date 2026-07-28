"""Agent-level unit tests (providers mocked / stubbed)."""

from __future__ import annotations

import pytest

from app.adapters.errors import PermanentProviderError
from app.adapters.gemini import GeminiImageAdapter
from app.adapters.llm import LLMAdapter
from app.agents.designer import designer
from app.agents.image_gen import image_gen
from app.agents.script_writer import (
    BEAT_ROLES,
    _beat_roles,
    _system_prompt,
    script_writer,
)
from app.artifacts.store import ArtifactStore
from app.core.config import get_settings
from app.graph.state import Deps
from app.schemas import Scene, Script


def _deps(artifact_root, job_id, image_adapter=None) -> Deps:
    settings = get_settings()
    return Deps(
        settings=settings,
        store=ArtifactStore(artifact_root, job_id),
        llm=LLMAdapter(settings),
        image=image_adapter or GeminiImageAdapter(settings),
        max_frames=3,
    )


async def test_script_writer_produces_valid_script(artifact_root):
    deps = _deps(artifact_root, "job-sw")
    out = await script_writer({"prompt": "a lonely robot", "duration": 30}, deps)

    script = Script.model_validate(deps.store.load_json(out["script_path"]))
    assert len(script.scenes) == 3
    assert all(s.narration for s in script.scenes)


async def test_script_writer_populates_arc_and_beat_roles(artifact_root):
    """W1: the script now plans an arc and tags each beat's structural role."""
    deps = _deps(artifact_root, "job-arc")
    out = await script_writer({"prompt": "a lonely robot", "duration": 30}, deps)
    script = Script.model_validate(deps.store.load_json(out["script_path"]))

    assert script.arc.strip(), "arc should be populated (plan-then-write)"
    roles = [s.beat_role for s in script.scenes]
    assert all(r in BEAT_ROLES for r in roles), roles
    assert roles[0] == "setup"
    assert roles[-1] == "resolution"


def test_system_prompt_builds_and_mentions_structure():
    """The prompt builder interpolates n without the str.format brace trap and
    carries the narrative scaffold."""
    prompt = _system_prompt(15)
    assert "15" in prompt
    for token in ("BEGINNING", "MIDDLE", "END", "arc", "beat_role", "DON'T"):
        assert token in prompt
    # No stray unescaped format fields left behind.
    assert "{" not in prompt and "}" not in prompt


def test_beat_roles_shape():
    assert _beat_roles(1) == ["setup"]
    for n in (2, 3, 5, 15, 30):
        roles = _beat_roles(n)
        assert len(roles) == n
        assert roles[0] == "setup"
        assert roles[-1] == "resolution"
        assert all(r in BEAT_ROLES for r in roles)
    # A long story reaches a climax before resolving.
    assert "climax" in _beat_roles(15)


def test_script_schema_backward_compatible():
    """Scripts written before arc/beat_role existed must still validate (defaults)."""
    legacy = {
        "title": "t",
        "logline": "l",
        "scenes": [{"order": 1, "description": "d", "narration": "n"}],
    }
    script = Script.model_validate(legacy)
    assert script.arc == ""
    assert script.scenes[0].beat_role == ""
    # And a fully-specified scene round-trips.
    s = Scene(order=2, beat_role="climax", description="d2", narration="n2")
    assert s.beat_role == "climax"


async def test_image_gen_mock_all_ok(artifact_root):
    deps = _deps(artifact_root, "job-ig")
    sw = await script_writer({"prompt": "sunrise over a city", "duration": 30}, deps)
    dz = await designer({"script_path": sw["script_path"], "duration": 30}, deps)
    out = await image_gen(
        {
            "shotlist_path": dz["shotlist_path"],
            "style_ref_path": dz["style_ref_path"],
            "duration": 30,
        },
        deps,
    )

    assert len(out["frames"]) == 3
    assert all(f["status"] == "ok" for f in out["frames"])
    assert out["warnings"] == []


class _AlwaysFailsImage:
    """Stub image adapter that always raises -> exercises the 4A degraded path."""

    def generate_image(self, *a, **k):
        raise PermanentProviderError("safety block")


async def test_image_gen_degraded_on_failure(artifact_root):
    deps = _deps(artifact_root, "job-degraded", image_adapter=_AlwaysFailsImage())
    sw = await script_writer({"prompt": "forbidden scene", "duration": 30}, deps)
    # designer's style_ref also fails on this adapter -> text-only fallback (None).
    dz = await designer({"script_path": sw["script_path"], "duration": 30}, deps)
    assert dz["style_ref_path"] is None
    out = await image_gen(
        {
            "shotlist_path": dz["shotlist_path"],
            "style_ref_path": dz["style_ref_path"],
            "duration": 30,
        },
        deps,
    )

    # Every frame still produced (placeholder), but flagged degraded + warned.
    assert len(out["frames"]) == 3
    assert all(f["status"] == "degraded" for f in out["frames"])
    assert len(out["warnings"]) == 3
    assert "image generation failed" in out["warnings"][0]


def test_llm_adapter_mock_returns_typed_model():
    """Structured-output contract: complete_json returns a validated pydantic model."""
    adapter = LLMAdapter(get_settings())
    result = adapter.complete_json(
        system="sys",
        user="make a script",
        response_model=Script,
        mock_factory=lambda _u: Script(title="t", logline="l", scenes=[]),
    )
    assert isinstance(result, Script)
