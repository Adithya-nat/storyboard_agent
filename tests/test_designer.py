"""designer agent: artifact production, degrade-don't-die fallbacks, idempotency."""

from __future__ import annotations

from pathlib import Path

from app.adapters.errors import PermanentProviderError
from app.adapters.gemini import GeminiImageAdapter
from app.adapters.llm import LLMAdapter
from app.agents.designer import designer
from app.agents.script_writer import script_writer
from app.artifacts.store import ArtifactStore
from app.core.config import get_settings
from app.graph.state import Deps
from app.schemas import Shotlist


def _deps(artifact_root, job_id, *, llm=None, image=None, max_frames=3) -> Deps:
    settings = get_settings()
    return Deps(
        settings=settings,
        store=ArtifactStore(artifact_root, job_id),
        llm=llm or LLMAdapter(settings),
        image=image or GeminiImageAdapter(settings),
        max_frames=max_frames,
    )


async def _script(deps, prompt="a quiet town wakes up"):
    return await script_writer({"prompt": prompt, "duration": 30}, deps)


async def test_designer_mock_produces_all_artifacts(artifact_root):
    deps = _deps(artifact_root, "dz-ok")
    sw = await _script(deps)
    out = await designer({"script_path": sw["script_path"], "duration": 30}, deps)

    assert Path(out["shotlist_path"]).exists()
    assert Path(out["style_bible_path"]).exists()
    assert out["style_ref_path"] and Path(out["style_ref_path"]).exists()
    assert out["warnings"] == []

    sl = Shotlist.model_validate(deps.store.load_json(out["shotlist_path"]))
    assert len(sl.shots) == 3  # 1:1 with script scenes (max_frames=3)
    assert sl.style_bible.strip()


class _FailImage:
    def generate_image(self, *a, **k):
        raise PermanentProviderError("image provider down")


async def test_designer_style_ref_failure_falls_back_to_text_only(artifact_root):
    deps = _deps(artifact_root, "dz-noref", image=_FailImage())
    sw = await _script(deps)
    out = await designer({"script_path": sw["script_path"], "duration": 30}, deps)

    # Shotlist + bible still produced (LLM mock ok); only the style ref failed.
    assert Path(out["shotlist_path"]).exists()
    assert out["style_ref_path"] is None  # text-only conditioning
    assert any("style reference failed" in w for w in out["warnings"])


class _FailLLM:
    def complete_json(self, **k):
        raise PermanentProviderError("llm provider down")


async def test_designer_shotlist_llm_failure_derives_from_script(artifact_root):
    # Produce the script with a working (mock) LLM first...
    deps_ok = _deps(artifact_root, "dz-fallback")
    sw = await _script(deps_ok)
    # ...then run the designer with a FAILING LLM against the same store.
    deps_fail = _deps(artifact_root, "dz-fallback", llm=_FailLLM())
    out = await designer({"script_path": sw["script_path"], "duration": 30}, deps_fail)

    sl = Shotlist.model_validate(deps_fail.store.load_json(out["shotlist_path"]))
    assert len(sl.shots) == 3  # derived directly from the script scenes
    assert any("shotlist generation failed" in w for w in out["warnings"])


class _CountingLLM:
    """Wrap the real (mock) LLM and count calls to prove idempotent skip."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def complete_json(self, **k):
        self.calls += 1
        return self.inner.complete_json(**k)


async def test_designer_idempotent_reuses_shotlist(artifact_root):
    settings = get_settings()
    counting = _CountingLLM(LLMAdapter(settings))
    deps = Deps(
        settings=settings,
        store=ArtifactStore(artifact_root, "dz-idem"),
        llm=counting,
        image=GeminiImageAdapter(settings),
        max_frames=3,
    )
    sw = await script_writer({"prompt": "x", "duration": 30}, deps)  # LLM call #1
    await designer({"script_path": sw["script_path"], "duration": 30}, deps)  # call #2
    after_first = counting.calls
    # Second designer run must NOT call the LLM again (shotlist.json exists).
    await designer({"script_path": sw["script_path"], "duration": 30}, deps)
    assert counting.calls == after_first
