"""Pydantic schemas for jobs, API I/O, and the artifacts that flow between agents.

These are the contracts of the "context engineering" design: agents read/write
small structured artifacts (script, shotlist, frame metadata) and pass them by
*reference*. Binary media (images, video) never lives in these objects — only
paths/ids do.

Status state machine (job):

    queued ──▶ running ──▶ completed
                  │   └───▶ completed_with_warnings   (some frames degraded)
                  ├───────▶ failed                    (unrecoverable error)
                  └───────▶ interrupted               (process died mid-job; reaper)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


TERMINAL_STATUSES = {
    JobStatus.COMPLETED,
    JobStatus.COMPLETED_WITH_WARNINGS,
    JobStatus.FAILED,
    JobStatus.INTERRUPTED,
}


# --------------------------------------------------------------------------- #
# API request / response
# --------------------------------------------------------------------------- #
class StoryboardRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=2000)
    duration: Literal[30, 60] = 30
    # Cloudflare Turnstile token from the create form. Verified server-side only
    # when turnstile_enabled (off in local/mock dev), so it stays optional here.
    turnstile_token: str | None = None


class JobCreatedResponse(BaseModel):
    job_id: str
    status: JobStatus


class ArtifactRef(BaseModel):
    """A reference to a stored artifact. Never inlines the binary, and never leaks
    the server filesystem path — the browser fetches bytes via ``url``."""

    id: str
    kind: str  # script | shotlist | style_bible | style_ref | frame | video
    url: str  # GET path that streams the bytes: /storyboard/{job}/artifacts/{id}
    status: str = "ok"  # ok | degraded
    order: int | None = None  # frame ordering within the storyboard


class StageRef(BaseModel):
    """One pipeline stage's live status — the collapsed/summary view a front end
    shows while a job runs."""

    name: str  # script_writer | reflection | designer | image_gen | assembler
    seq: int
    status: str  # pending | running | succeeded | failed | degraded | interrupted
    message: str | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    tokens: int | None = None  # tokens this stage consumed (None if not yet recorded)
    started_at: datetime | None = None
    ended_at: datetime | None = None


class LogLine(BaseModel):
    """One log line — the detail shown when a stage is expanded / an SSE log event.
    ``id`` is the streaming cursor (pass as ``?after=`` or ``Last-Event-ID``)."""

    id: int
    stage: str | None = None
    level: str
    message: str
    created_at: datetime


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    prompt: str
    duration: int
    created_at: datetime
    expires_at: datetime  # created_at + TTL; drives the UI expiry countdown
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    current_stage: str | None = None  # name of the stage currently 'running', if any
    # Token accounting (Workstream 4): whole-job totals; per-stage is on each StageRef.
    total_tokens: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    # Ordered pipeline stage names for THIS job (includes "motion" only when the
    # motion engine is enabled). The front end derives its stage list/progress from
    # this so a static client list can't disagree with what actually runs.
    pipeline: list[str] = Field(default_factory=list)
    stages: list[StageRef] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)


class LogsResponse(BaseModel):
    """Paginated pull of a job's logs (the non-streaming expand-view + curl path)."""

    job_id: str
    logs: list[LogLine] = Field(default_factory=list)
    next_after: int  # cursor for the next pull (?after=); == the last id returned


class StoryboardScene(BaseModel):
    """One finished beat for the result page: a frame image + the caption the
    assembler burned into the video for that beat."""

    order: int
    narration: str
    frame_url: str | None = None  # None if the frame artifact is missing
    # Per-scene motion clip (motion engine). None => show the still frame_url.
    # The UI plays this and uses frame_url as the poster/fallback.
    video_url: str | None = None
    status: str = "ok"  # ok | degraded
    motion_kind: str | None = None
    motion_renderer: str | None = None
    motion_model: str | None = None
    motion_reason: str | None = None


class MotionCoverage(BaseModel):
    total: int = 0
    animated: int = 0
    generative: int = 0
    local_generative: int = 0
    composited: int = 0
    kenburns: int = 0
    still: int = 0
    by_renderer: dict[str, int] = Field(default_factory=dict)
    by_model: dict[str, int] = Field(default_factory=dict)


class StoryboardContent(BaseModel):
    """The assembled storyboard the result page renders: the video plus per-scene
    frames and their captions, sourced so on-screen text matches the mp4."""

    job_id: str
    status: JobStatus
    title: str = ""
    logline: str = ""
    style: str = ""
    duration: int
    video_url: str | None = None
    style_ref_url: str | None = None
    total_tokens: int = 0  # tokens consumed generating this storyboard
    narrated: bool = False  # a synced narration track was muxed under the video
    motion_coverage: MotionCoverage | None = None
    scenes: list[StoryboardScene] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Inter-agent artifacts (the "script" produced by the script_writer)
# --------------------------------------------------------------------------- #
class Scene(BaseModel):
    """One beat of the story = one storyboard frame."""

    order: int = Field(description="1-based position of this beat in story order.")
    # Declared BEFORE description so the model commits to WHY this beat exists before it
    # writes WHAT we see. Optional + default "" = backward-compatible with on-disk scripts
    # written before this field existed. (Field description is sent to the model via the
    # structured-output schema — keep it in sync with the script_writer/reviser prompts.)
    beat_role: str = Field(
        default="",
        description=(
            "This beat's structural role in the arc: one of setup, inciting, rising, "
            "climax, resolution."
        ),
    )
    description: str = Field(
        description=(
            "What the CAMERA SEES — concrete, filmable action and staging. Keep the same "
            "character(s) and setting identical across every scene."
        )
    )
    narration: str = Field(
        description=(
            "One short caption sentence (aim for under ~15 words) that moves the story "
            "forward; TTS-ready."
        )
    )


class Script(BaseModel):
    title: str = Field(description="Short title for the film.")
    logline: str = Field(description="One-sentence summary of the whole story.")
    style: str = Field(
        default="",
        description="Short visual-style note, e.g. 'moody 90s anime, teal palette'.",
    )
    # Declared BEFORE scenes so the model PLANS the arc, then writes beats that deliver it
    # (chain-of-thought inside the strict-JSON call). Optional + default "" keeps old
    # artifacts valid.
    arc: str = Field(
        default="",
        description=(
            "One paragraph stating the beginning -> middle -> end (who, what they want, "
            "what changes, how it resolves). Plan this FIRST, then write scenes that "
            "deliver it."
        ),
    )
    scenes: list[Scene] = Field(
        description="The ordered beats that deliver the arc, in story order."
    )


# --------------------------------------------------------------------------- #
# Designer artifacts (Phase 2 — defined now so the schema is stable)
# --------------------------------------------------------------------------- #
class Shot(BaseModel):
    order: int = Field(
        description="Matches the source scene's order (1-based, story order)."
    )
    image_prompt: str = Field(
        description=(
            "Self-contained prompt for the image model: restate the scene visually AND "
            "bake in the style-bible look (each subject's exact appearance), so the frame "
            "renders stand-alone yet matches every other shot."
        )
    )
    narration: str = Field(
        description="The source scene's caption, carried through unchanged."
    )


class Shotlist(BaseModel):
    style_bible: str = Field(
        description=(
            "Detailed, concrete visual rules EVERY frame must obey (render style, palette, "
            "lighting, lens/composition, and each character/setting's exact recurring "
            "look). The anti-drift contract."
        )
    )
    shots: list[Shot] = Field(
        description=(
            "Exactly one shot per script scene, preserving each scene's order and "
            "narration."
        )
    )


# --------------------------------------------------------------------------- #
# Narrative judge (Workstream 2/3) — scores story cohesion BEFORE media gen.
# Shared by the eval harness (measure) and the reflection node (runtime guard).
# --------------------------------------------------------------------------- #
class JudgeReport(BaseModel):
    """A story-cohesion score for a Script (0-100 per dimension).

    The model fills the five dimension scores + the qualitative fields; the canonical
    ``overall`` is (re)computed by the judge as a weighted mean (never trust model
    arithmetic), so a threshold means the same thing in eval and in reflection.
    """

    structure: int = Field(
        default=0,
        description="0-100: beginning/middle/end present; does beat_role progress sensibly?",
    )
    continuity: int = Field(
        default=0,
        description="0-100: each beat follows causally; characters/setting/props consistent.",
    )
    clarity: int = Field(
        default=0, description="0-100: a first-time viewer can follow it effortlessly."
    )
    visual_concreteness: int = Field(
        default=0, description="0-100: concrete and filmable, not vague mood words."
    )
    payoff: int = Field(
        default=0, description="0-100: the ending resolves the hook satisfyingly."
    )
    # Recomputed as a weighted mean by the judge — never trusted from the model.
    overall: int = Field(
        default=0,
        description="Leave 0 — computed downstream as a weighted mean; do not fill.",
    )
    weakest_dimension: str = Field(
        default="", description="The single dimension dragging the story down most."
    )
    weak_beats: list[int] = Field(
        default_factory=list,
        description="The 'order' numbers of the beats that hurt the story most (may be empty).",
    )
    revision_guidance: str = Field(
        default="",
        description=(
            "2-4 sentences of concrete, actionable fixes; name the beats and what to "
            "change. (Consumed by the reflection node.)"
        ),
    )
    rationale: str = Field(
        default="", description="One short paragraph justifying the scores."
    )
