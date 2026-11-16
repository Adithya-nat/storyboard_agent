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
import math
from dataclasses import dataclass
from typing import Annotated, TypedDict

from app.adapters.gemini import GeminiImageAdapter
from app.adapters.llm import LLMAdapter
from app.adapters.motion_engine import MotionEngineClient
from app.adapters.music import LyriaMusicAdapter
from app.adapters.omni import OmniVideoAdapter
from app.adapters.transcribe import TranscribeAdapter
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
    has_audio: bool  # the clip carries generated audio (Omni)
    qc: list[str]    # continuity issues left in the delivered clip (empty = passed)
    renders: int


class SceneAudioMeta(TypedDict, total=False):
    """How one scene is voiced + how long it holds (written by narration / captions)."""

    order: int
    voice: str          # narrator | on_camera | off_screen | silent
    path: str | None    # speech WAV for narrator / off_screen scenes (None for on_camera)
    hold: float         # seconds this scene stays on screen
    caption: str        # final on-screen caption (transcript-backed for dialogue)
    transcript: str


class StoryboardState(TypedDict, total=False):
    job_id: str
    prompt: str
    duration: int
    style: str     # narrated | dialogue | mixed
    language: str  # en | ta
    brief_path: str | None  # creative brief (format, hook, speech register, scene plan)
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
    stills_used: list[int]  # scenes the user chose to finish with their still image
    # Per-scene voicing + holds (narration stage writes narrator scenes; captions stage
    # finalizes dialogue scenes after motion). Keyed by order.
    scene_audio: list[SceneAudioMeta]
    cast_paths: dict[str, list[str]]  # character name -> reference sheet paths
    location_paths: dict[str, str]  # location tag -> set reference plate
    music_path: str | None
    scene_audio_path: str | None
    publish_path: str | None
    film_review_path: str | None  # final-film coherence review
    cover_path: str | None
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


# A scene is one continuous place/time filmed as a multi-shot take (one Omni clip, <=15s).
SCENE_MIN_SECONDS, SCENE_MAX_SECONDS = 5.0, 15.0
_STYLE_SCENE_SECONDS = {"narrated": 8.0, "mixed": 10.0, "dialogue": 11.0}


def scene_range(duration: int) -> tuple[int, int]:
    """Allowed scene counts for a runtime: every scene between 5 and 15 seconds."""
    lo = max(1, math.ceil(duration / SCENE_MAX_SECONDS))
    return lo, max(lo, math.floor(duration / SCENE_MIN_SECONDS))


def scenes_for_duration(duration: int, style: str = "narrated") -> int:
    """Default scene count (30s dialogue -> 3, 60s dialogue -> 5, 60s narrated -> 8)."""
    lo, hi = scene_range(duration)
    n = round(duration / _STYLE_SCENE_SECONDS.get(style, 8.0))
    n = min(hi, max(lo, n))
    return max(2, n) if style in ("dialogue", "mixed") and hi >= 2 else n


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
    # 2026-11: transcription (dialogue captions) + music bed. Optional (None => the
    # captions stage falls back to the script lines; the music stage skips).
    transcriber: TranscribeAdapter | None = None
    music: LyriaMusicAdapter | None = None
    video: OmniVideoAdapter | None = None  # AI video (None => scenes keep stills)
    allow_stills: bool = False  # user chose to finish with stills for scenes that have no clip

    def frame_count(self, duration: int, style: str = "narrated") -> int:
        if self.max_frames is not None:
            return self.max_frames
        return scenes_for_duration(duration, style)
