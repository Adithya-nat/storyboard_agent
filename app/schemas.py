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

from pydantic import BaseModel, Field, field_validator


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
StoryStyle = Literal["narrated", "dialogue", "mixed"]
Language = Literal["en", "ta"]
SceneMode = Literal["narration", "dialogue"]
LANGUAGE_NAMES = {"en": "English", "ta": "Tamil"}


class StoryboardRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=2000)
    duration: Literal[30, 60] = 30
    # Story style: a narrator over cinematic scenes, characters speaking on camera, or a
    # mix (narrator frames the story, key beats cut to dialogue). None -> server default.
    style: StoryStyle | None = None
    # The whole video (narration, dialogue, captions, publish kit) is in ONE language.
    language: Language | None = None
    # Cloudflare Turnstile token from the create form. Verified server-side only
    # when turnstile_enabled (off in local/mock dev), so it stays optional here.
    turnstile_token: str | None = None


class VideoGap(BaseModel):
    """A film stopped because some scenes have no AI clip; it can finish with their stills."""

    ready: int
    total: int
    missing: list[int]
    reason: Literal["blocked", "unavailable"]


class ResumeRequest(BaseModel):
    allow_stills: bool = False  # finish now: scenes without a clip use their still image


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
    cost_usd: float | None = None  # estimated AI spend of this stage (None if not yet recorded)
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
    cost_usd: float = 0.0  # estimated AI spend so far (live while running)
    style: str = "narrated"
    language: str = "en"
    # Ordered pipeline stage names for THIS job (includes "motion" only when the
    # motion engine is enabled). The front end derives its stage list/progress from
    # this so a static client list can't disagree with what actually runs.
    pipeline: list[str] = Field(default_factory=list)
    stages: list[StageRef] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    video_gap: VideoGap | None = None  # set when the film can be finished with stills


class LogsResponse(BaseModel):
    """Paginated pull of a job's logs (the non-streaming expand-view + curl path)."""

    job_id: str
    logs: list[LogLine] = Field(default_factory=list)
    next_after: int  # cursor for the next pull (?after=); == the last id returned


class SceneLine(BaseModel):
    """One spoken dialogue line as shown on the result page (script vs what was said)."""

    speaker: str
    text: str
    delivery: str = ""


class SceneShotView(BaseModel):
    size: str = "medium"
    action: str = ""
    lines: list[SceneLine] = Field(default_factory=list)


class BriefView(BaseModel):
    """What the film set out to be (from the creative brief)."""

    format: str = ""
    hook: str = ""
    logline: str = ""
    setting: str = ""
    era: str = "modern"
    period: str = ""
    dialect: str = ""


class FilmReviewView(BaseModel):
    """A model's first-time-viewer review of the finished film."""

    score: int = 0
    coherence: int = 0
    continuity: int = 0
    dialogue: int = 0
    hook: int = 0
    human_made: int = 0
    retelling: str = ""
    confusing_moments: list[str] = Field(default_factory=list)


class StoryboardScene(BaseModel):
    """One finished beat for the result page: a frame image + the caption the
    assembler burned into the video for that beat."""

    order: int
    narration: str
    mode: str = "narration"  # narration | dialogue
    lines: list[SceneLine] = Field(default_factory=list)  # the screenplay lines (dialogue)
    shots: list[SceneShotView] = Field(default_factory=list)  # the scene's coverage, in order
    location: str = ""
    transcript: str = ""  # what the clip actually says (dialogue scenes, from transcription)
    # How this scene's voice was produced: narrator | on_camera | off_screen | silent
    voice: str = "narrator"
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
    story_style: str = "narrated"
    language: str = "en"
    duration: int
    video_url: str | None = None
    cover_url: str | None = None
    cost_usd: float = 0.0
    cast: list["CastMember"] = Field(default_factory=list)
    style_ref_url: str | None = None
    total_tokens: int = 0  # tokens consumed generating this storyboard
    narrated: bool = False  # a synced narration track was muxed under the video
    motion_coverage: MotionCoverage | None = None
    brief: BriefView | None = None
    film_review: FilmReviewView | None = None
    scenes: list[StoryboardScene] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Inter-agent artifacts (the "script" produced by the script_writer)
# --------------------------------------------------------------------------- #
class Character(BaseModel):
    """A recurring character. ``look`` and ``voice_description`` are FIXED strings reused
    verbatim in every image and video prompt — the anti-drift contract for both the
    character's appearance and their voice."""

    name: str = Field(
        description=(
            "Short UPPERCASE speaker tag used everywhere this character speaks, e.g. MAYA. "
            "For Tamil stories use a Tamil name written in Latin capitals (e.g. KAMALA)."
        )
    )
    look: str = Field(
        description=(
            "English visual description, fixed for the whole film: age (adults stated as 18+), build, "
            "face, hair, clothing, signature colours. Original fictional person — never a real or "
            "famous person."
        )
    )
    display_name: str = Field(
        default="",
        description=(
            "The character's name as viewers read it, written in the film's language and script "
            "(e.g. கதிர் in a Tamil film, Kathir in an English one). Used in on-screen captions."
        ),
    )
    voice_description: str = Field(
        default="",
        description=(
            "English description of how this character SOUNDS, fixed for the whole film. Start "
            "with a gender+age tag, e.g. 'FEMALE, 60s: warm weathered alto, slow and calm'. "
            "No regional accents."
        ),
    )


class DialogueLine(BaseModel):
    speaker: str = Field(description="The speaking character's name tag, exactly as in characters[].name.")
    text: str = Field(
        description=(
            "The exact words spoken, in the film's language. Short and speakable: one or two "
            "sentences."
        )
    )
    delivery: str = Field(
        default="",
        description="Short English acting note for how it is said, e.g. 'whispered, nervous'.",
    )


ShotSize = Literal["wide", "medium", "close", "extreme_close", "over_shoulder", "insert", "two_shot"]
Transition = Literal["cut", "match_cut", "dissolve"]


class SceneShot(BaseModel):
    """One camera set-up inside a scene. A scene's shots are filmed as ONE continuous take
    with cuts, so the room, light and voices stay the same across them."""

    size: ShotSize = Field(default="medium", description="Shot size / framing.")
    action: str = Field(
        description=(
            "English: who is in frame and what they physically do, concrete and filmable. A shot "
            "may be a silent reaction or an insert of a prop."
        )
    )
    camera: str = Field(
        default="static",
        description="English camera note, motivated by the action (often 'static'; or 'slow push-in', 'pan to door').",
    )
    seconds: float = Field(default=3.0, description="Screen time of this shot, 1.5-6 seconds.")
    lines: list[DialogueLine] = Field(
        default_factory=list,
        description=(
            "Lines spoken during this shot (may be empty for silent shots). The speaker may be "
            "off-screen while the camera holds on the listener."
        ),
    )

    @field_validator("size", mode="before")
    @classmethod
    def _known_size(cls, v):
        return v if v in ShotSize.__args__ else "medium"


class CastMember(BaseModel):
    """A character as shown on the status/result pages (cast card)."""

    name: str
    look: str = ""
    voice_description: str = ""
    portrait_url: str | None = None


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
    mode: SceneMode = Field(
        default="narration",
        description=(
            "'narration' = the narrator speaks this beat over the picture; 'dialogue' = the "
            "characters speak on camera."
        ),
    )
    location: str = Field(
        default="",
        description="Short English location tag reused verbatim by every scene set there, e.g. 'MEENA_FLAT_KITCHEN'.",
    )
    time_of_day: str = Field(default="", description="English, e.g. 'night', 'dawn'.")
    goal: str = Field(
        default="",
        description="English: what the scene's lead wants here, the obstacle, and how the scene turns.",
    )
    description: str = Field(
        description=(
            "What the CAMERA SEES in the scene's OPENING shot — concrete, filmable staging, written "
            "in English. Keep the same character(s) and setting identical across every scene."
        )
    )
    seconds: float = Field(default=0, description="Screen time of the whole scene (sum of its shots), at most 15.")
    transition_in: Transition = Field(
        default="cut",
        description="How we enter this scene: 'cut' (default), 'match_cut' (a shape or action rhymes across the cut), 'dissolve' (time jump only).",
    )
    shots: list[SceneShot] = Field(
        default_factory=list,
        description="2-5 shots that cover the scene, in order. Dialogue lives inside the shots.",
    )
    exit_hook: str = Field(
        default="",
        description="English: the question or action that pulls the viewer into the next scene.",
    )
    characters: list[str] = Field(
        default_factory=list,
        description="Names (characters[].name) visible in this scene; at most 2 for dialogue scenes.",
    )
    narration: str = Field(
        default="",
        description=(
            "Narration scenes: the narrator's words for this scene in the film's language, within the "
            "scene's word budget, moving the story forward. Dialogue scenes: leave empty."
        ),
    )
    lines: list[DialogueLine] = Field(
        default_factory=list,
        description="Leave empty — filled by the pipeline from the shots' lines.",
    )

    @field_validator("transition_in", mode="before")
    @classmethod
    def _known_transition(cls, v):
        return v if v in Transition.__args__ else "cut"  # older artifacts may carry retired values

    def shot_lines(self) -> list[DialogueLine]:
        return [line for shot in self.shots for line in shot.lines]

    def spoken_text(self) -> str:
        """Everything spoken in this beat (narration or the joined dialogue)."""
        if self.mode == "dialogue" and self.lines:
            return " ".join(line.text for line in self.lines)
        return self.narration

    def caption_text(self, names: dict[str, str] | None = None) -> str:
        """The default on-screen caption: narration, or dialogue labelled with display names."""
        if self.mode == "dialogue" and self.lines:
            names = names or {}
            return "\n".join(f"{names.get(line.speaker, line.speaker)}: {line.text}" for line in self.lines)
        return self.narration


class Script(BaseModel):
    title: str = Field(description="Short title for the film, in the film's language.")
    logline: str = Field(description="One-sentence summary of the whole story, in the film's language.")
    # Set by the pipeline from the request (not by the model's judgement).
    story_style: str = Field(default="narrated", description="narrated | dialogue | mixed (as requested).")
    language: str = Field(default="en", description="Film language code: en | ta (as requested).")
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
    characters: list[Character] = Field(
        default_factory=list,
        description=(
            "The 1-4 recurring characters, defined ONCE here and referenced by name in "
            "scenes. Every speaking character must be listed."
        ),
    )
    scenes: list[Scene] = Field(
        description="The ordered beats that deliver the arc, in story order."
    )

    def display_names(self) -> dict[str, str]:
        """Speaker tag -> the name shown on screen (falls back to a title-cased tag)."""
        return {c.name.upper(): (c.display_name.strip() or c.name.title()) for c in self.characters}

    def character(self, name: str) -> Character | None:
        key = name.strip().upper()
        return next((c for c in self.characters if c.name.strip().upper() == key), None)


# --------------------------------------------------------------------------- #
# Creative brief (what the creator is trying to make, before any writing)
# --------------------------------------------------------------------------- #
Era = Literal["modern", "period"]


class SpeechRegister(BaseModel):
    era: Era = Field(
        default="modern",
        description="'modern' = present-day setting; 'period' = any clearly past (or historical/mythic) setting.",
    )
    period: str = Field(default="present day", description="English, e.g. 'present day', 'Madras, 1972', 'Chola court, 11th century'.")
    dialect: str = Field(
        default="",
        description=(
            "English: how these people actually talk, e.g. 'colloquial Chennai Tamil, young friends', "
            "'Madurai Tamil, rural elders', 'casual American English, Gen Z'."
        ),
    )
    slang: str = Field(
        default="",
        description="English: slang and code-mixing level that fits the characters, era and setting (or 'none').",
    )
    avoid: str = Field(default="", description="English: words, slang or tech that would be anachronistic or off-register.")


class PlannedScene(BaseModel):
    order: int = Field(description="1-based scene order.")
    purpose: str = Field(description="English: what this scene does for the story (hook, setup, turn, payoff...).")
    location: str = Field(default="", description="English location tag, reused for every scene in the same place.")
    seconds: float = Field(description="Screen time for this scene, 5-15 seconds.")
    mode: SceneMode = Field(default="dialogue", description="'dialogue' (characters speak on camera) or 'narration' (narrator over picture).")


class CreativeBrief(BaseModel):
    """The creator's intent, decided once and obeyed by every later agent."""

    format: str = Field(description="English, e.g. 'short drama', 'monologue', 'comedy skit', 'heist thriller', 'emotional slice of life'.")
    purpose: str = Field(description="English: why someone would watch and share this; the feeling it leaves.")
    audience: str = Field(default="", description="English: who it's for.")
    hook: str = Field(description="English: exactly what the viewer sees and hears in the first 2 seconds that stops the scroll.")
    logline: str = Field(description="English: one sentence — who wants what, what stands in the way, how it turns out.")
    setting: str = Field(description="English: place and time, concrete.")
    speech: SpeechRegister = Field(default_factory=SpeechRegister)
    user_structure: str = Field(
        default="",
        description=(
            "If the idea already spells out beats, timecodes or a structure, restate it here faithfully "
            "(English). Otherwise empty."
        ),
    )
    originality_note: str = Field(
        default="",
        description=(
            "If the idea names an existing film, character, franchise or real person: the original names "
            "and look that replace them while keeping the premise. Otherwise empty."
        ),
    )
    scenes: list[PlannedScene] = Field(description="The scene plan, in order.")


# --------------------------------------------------------------------------- #
# Designer artifacts (Phase 2 — defined now so the schema is stable)
# --------------------------------------------------------------------------- #
class LocationSpec(BaseModel):
    tag: str = Field(description="The scene 'location' tag, verbatim.")
    look: str = Field(
        description=(
            "English, fixed for the whole film: layout, key furniture and props and where they sit, "
            "materials, colours, light sources. Every shot set here must match it."
        )
    )


class Shot(BaseModel):
    order: int = Field(
        description="Matches the source scene's order (1-based, story order)."
    )
    location: str = Field(default="", description="The scene's location tag, verbatim.")
    image_prompt: str = Field(
        description=(
            "Self-contained prompt for the image model: restate the scene visually AND "
            "bake in the style-bible look (each subject's exact appearance), so the frame "
            "renders stand-alone yet matches every other shot."
        )
    )
    narration: str = Field(
        default="",
        description="The source scene's narration, carried through unchanged (empty for dialogue scenes).",
    )
    characters: list[str] = Field(
        default_factory=list,
        description="Names of the characters visible in this shot (from the script), for reference sheets.",
    )


class Shotlist(BaseModel):
    style_bible: str = Field(
        description=(
            "Detailed, concrete visual rules EVERY frame must obey (render style, palette, "
            "lighting, lens/composition, and each character/setting's exact recurring "
            "look). The anti-drift contract."
        )
    )
    locations: list[LocationSpec] = Field(
        default_factory=list,
        description="One entry per distinct scene location tag: the fixed set every scene there reuses.",
    )
    shots: list[Shot] = Field(
        description=(
            "Exactly one shot per script scene (its OPENING frame), preserving each scene's order and "
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

    # Declared first: the judge retells the film before scoring it (a table read).
    retelling: str = Field(
        default="",
        description=(
            "Retell the story in 2-3 sentences using ONLY what a viewer would see and hear (actions and "
            "spoken words), as a first-time viewer would summarise it."
        ),
    )
    hook: int = Field(default=0, description="0-100: do the first 2 seconds stop the scroll and raise a question?")
    brief_adherence: int = Field(
        default=0, description="0-100: does it deliver the creative brief (format, purpose, structure, logline)?"
    )
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
    dialogue: int = Field(
        default=0,
        description="0-100: natural, in-character, speakable lines (only scored when the film has dialogue).",
    )
    dialogue_thread: int = Field(
        default=0,
        description="0-100: within each scene every line answers the one before; threads carry across scenes (dialogue only).",
    )
    register_fit: int = Field(
        default=0,
        description="0-100: spoken words fit the era, place, dialect and characters (today's slang vs period speech).",
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


# --------------------------------------------------------------------------- #
# Publish kit (social posting copy)
# --------------------------------------------------------------------------- #
class PlatformPost(BaseModel):
    caption: str = Field(description="Post caption for this platform, in the film's language.")
    hashtags: list[str] = Field(default_factory=list, description="3-8 hashtags, each starting with '#'.")


class PublishKit(BaseModel):
    title: str = Field(description="Punchy title, max 60 characters, in the film's language.")
    hook: str = Field(description="One scroll-stopping line, max 100 characters, in the film's language.")
    tiktok: PlatformPost
    reels: PlatformPost
    shorts: PlatformPost
    cover_order: int = Field(default=0, description="Leave 0; set by the pipeline.")


# Resolve forward references (StoryboardContent.cast -> CastMember, defined later).
StoryboardContent.model_rebuild()
