"""Frame-count scaling (1 frame / 2s) + the assembler hold-time floor."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.adapters.gemini import GeminiImageAdapter
from app.adapters.llm import LLMAdapter
from app.agents.assembler import assembler
from app.agents.script_writer import script_writer
from app.artifacts.store import ArtifactStore
from app.core.config import get_settings
from app.graph.state import Deps, FrameMeta, frames_for_duration
from app.schemas import Script


def test_frames_for_duration():
    assert frames_for_duration(30, 2.0) == 15
    assert frames_for_duration(60, 2.0) == 30
    assert frames_for_duration(10, 2.0) == 5
    assert frames_for_duration(1, 2.0) == 1  # never zero


def _deps(artifact_root, job_id, max_frames=None) -> Deps:
    settings = get_settings()
    return Deps(
        settings=settings,
        store=ArtifactStore(artifact_root, job_id),
        llm=LLMAdapter(settings),
        image=GeminiImageAdapter(settings),
        max_frames=max_frames,
    )


@pytest.mark.parametrize("duration,expected", [(30, 15), (60, 30)])
async def test_script_writer_scales_with_duration(artifact_root, duration, expected):
    deps = _deps(artifact_root, f"fc-{duration}")  # max_frames=None -> derived
    out = await script_writer({"prompt": "a city at dawn", "duration": duration}, deps)
    script = Script.model_validate(deps.store.load_json(out["script_path"]))
    assert len(script.scenes) == expected


async def test_assembler_per_frame_floor(artifact_root, monkeypatch):
    captured: dict = {}

    async def _fake_assemble(timeline, output_path):
        captured["timeline"] = timeline
        Path(output_path).write_bytes(b"x")
        return output_path

    monkeypatch.setattr("app.agents.assembler.assemble_video", _fake_assemble)

    deps = _deps(artifact_root, "fc-floor")
    # 15 frames but only 10s of duration -> raw 0.67s/frame, floored to 2.0s.
    # Real clean frames on disk: the assembler now reads + captions each one.
    from app.core.images import render_card

    frames = []
    for i in range(1, 16):
        path = deps.store.save_bytes(f"frame_{i:02d}.png", render_card(f"f{i}"))
        frames.append(FrameMeta(order=i, path=path, caption="c", status="ok"))
    await assembler({"frames": frames, "duration": 10}, deps)

    durations = [d for _, d in captured["timeline"]]
    assert all(d >= deps.settings.seconds_per_frame for d in durations)
    assert durations[0] == 2.0
