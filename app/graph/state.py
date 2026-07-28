"""Shared state for the storyboard LangGraph.

Holds only JSON-serializable *references* (paths) + metadata — never image/video
bytes. This is the context-engineering contract: nodes pass artifacts by
reference, so the graph state (and any checkpoint) stays small.

Non-serializable dependencies (adapters, store) are bound into the nodes via
functools.partial in graph/build.py, NOT carried in state.

    StoryboardState (references only)
      job_id, prompt, duration
      script_path        ── script_writer
      shotlist_path      ┐
      style_bible_path   ├ designer
      style_ref_path     ┘  (None => style-ref gen failed => text-only conditioning)
      frames[]           ── image_gen
      video_path         ── assembler
      warnings[]         ── accumulated across nodes (reducer, see below)
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, TypedDict

from app.adapters.gemini import GeminiImageAdapter
from app.adapters.llm import LLMAdapter
from app.adapters.motion_engine import MotionEngineClient
from app.adapters.tts import GeminiTTSAdapter
from app.artifacts.store import ArtifactStore
from app.core.config import Settings
from app.observability.reporter import ProgressReporter


class FrameMeta(TypedDict):
    order: int
    path: str
    caption: str
    status: str  # ok | degraded


class MotionClipMeta(TypedDict, total=False):
    order: int
    path: str    # abspath of motion_NN.mp4 in the job dir
    status: str  # ok | degraded (degraded = engine fell back to its own Ken Burns)
    tier: str    # veo | parallax | kenburns | cached
    motion_kind: str
    renderer: str
    model: str | None
    source_seconds: float | None
    pipeline_version: str | None
    attempts: list[dict]
    motion_reason: str | None


class StoryboardState(TypedDict, total=False):
    job_id: str
    prompt: str
    duration: int
    script_path: str | None
    # Reflection output (Workstream 3): the critique/revise report. The chosen script is
    # written back over script.json in place, so script_path is unchanged.
    reflection_path: str | None
    # Designer outputs (Phase 2)
    shotlist_path: str | None
    style_bible_path: str | None
    style_ref_path: str | None  # None => style-ref generation failed => text-only style
    frames: list[FrameMeta]
    # Per-scene motion clips from the external engine (Workstream: motion engine).
    # Written only by the `motion` node when motion_engine_enabled; absent otherwise.
    motion_clips: list[MotionClipMeta]
    motion_manifest_path: str | None
    video_path: str | None
    narration_path: str | None  # narration track muxed under the video (None = silent)
    # Multiple nodes (designer + image_gen) can emit warnings; the reducer makes
    # them ACCUMULATE instead of the last writer overwriting earlier warnings.
    warnings: Annotated[list[str], operator.add]


def frames_for_duration(duration: int, seconds_per_frame: float) -> int:
    """Frame count from video length: 1 frame per ``seconds_per_frame`` seconds.

    At 2.0s/frame: 30s -> 15 frames, 60s -> 30 frames. Always at least 1.
    """
    if seconds_per_frame <= 0:
        raise ValueError("seconds_per_frame must be > 0")
    return max(1, round(duration / seconds_per_frame))


@dataclass
class Deps:
    """Injected, non-serializable collaborators for the graph nodes."""

    settings: Settings
    store: ArtifactStore
    llm: LLMAdapter
    image: GeminiImageAdapter
    # TTS is optional (Phase 3, off unless settings.enable_tts). Default None so
    # tests/agents that never voice-over don't have to construct one.
    tts: GeminiTTSAdapter | None = None
    # Optional override for the frame count. When None (production default) the
    # count is derived from duration via frames_for_duration(). Tests/evals can
    # pin a small number for speed.
    max_frames: int | None = None
    # Observability sink (per-stage status + logs). Optional + default None so
    # agent unit tests and graph builds without streaming are unaffected; the
    # node wrapper in graph/build.py becomes a pure pass-through when it's None.
    reporter: ProgressReporter | None = None
    # Motion engine client (per-scene animation). Set by the runner only when
    # settings.motion_engine_enabled; None => the motion node no-ops (degrade).
    motion: MotionEngineClient | None = None

    def frame_count(self, duration: int) -> int:
        if self.max_frames is not None:
            return self.max_frames
        return frames_for_duration(duration, self.settings.seconds_per_frame)
