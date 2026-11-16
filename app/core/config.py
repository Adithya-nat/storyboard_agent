"""Settings from env / .env. An empty GEMINI_API_KEY switches to offline mock mode."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Gemini ---
    gemini_api_key: str = ""
    # Text + image models are config-driven so upgrading is a .env change, no code
    gemini_text_model: str = "gemini-3.8-flash"
    gemini_image_model: str = "gemini-3-pro-image"  # G0 bake-off winner (quality-first)
    use_mock_providers: bool = False

    # --- Per-role text models (2026-11 modernization) ---
    # Each LLM role routes to its own model so the creative writer can be the most
    script_model: str = "gemini-4-carbon"   # script_writer + reflection reviser
    judge_model: str = "gemini-4-argon"     # NarrativeJudge (+ G1 audio/video judging)
    designer_model: str = "gemini-4-argon"  # shotlist / style bible / cast
    publish_model: str = "gemini-4-argon"   # publish kit (title, captions, hashtags)
    # Thinking depth per role (Gemini 3+/4: "low" | "medium" | "high"; "" = model default).
    script_thinking: str = "high"
    judge_thinking: str = "medium"
    designer_thinking: str = "medium"
    publish_thinking: str = "low"
    # Sampling temperature per role. None = don't send (Gemini 3+ recommend the default 1.0;
    script_temperature: float | None = None
    judge_temperature: float | None = None
    designer_temperature: float | None = None
    publish_temperature: float | None = None
    # Shared pacing for ALL text-model calls (Gemini 4 is 30 RPM on this account; 25 = margin).
    text_rpm_limit: int = 25
    # Per-request HTTP timeout for text calls (Gemini 4 Carbon can think for minutes).
    llm_timeout_seconds: float = 600.0
    # Validate every configured model id against models.list() once at startup.
    validate_models_on_startup: bool = True

    # --- Story style + language (per-job, these are the defaults) ---
    default_style: str = "dialogue"  # narrated | dialogue | mixed
    default_language: str = "en"      # en | ta
    # Pillow needs libraqm to shape complex scripts (Tamil). When it's missing, Tamil
    require_raqm_for_tamil: bool = True
    tamil_font_path: str = "app/assets/fonts/NotoSansTamil-Bold.ttf"
    tamil_only_hashtags: bool = False  # Tamil films: add a few English discovery tags unless set

    # --- Gemini TTS narration (synced voiceover, ON by default) ---
    # Each narration line is voiced and its image is held for exactly the narration's
    enable_tts: bool = True
    tts_model_chain: Annotated[list[str], NoDecode] = [
        "gemini-3.8-flash-tts",
        "gemini-3.8-flash-lite-tts",
    ]
    # Legacy single-model setting; if set in the env it is tried FIRST.
    gemini_tts_model: str = ""
    tts_voice: str = "Kore"  # prebuilt voice used when no designed narrator voice is set
    # Persistent custom narrator voice (Voice Design; scripts/design_voice.py creates it).
    narrator_voice_id: str = ""
    # Prebuilt voices for off-screen dialogue (multi-speaker fallback), assigned per
    dialogue_fallback_voices: Annotated[list[str], NoDecode] = ["Puck", "Kore", "Charon", "Zephyr"]

    # --- Transcription (captions from what the clip actually says) ---
    transcribe_model: str = "gemini-3.5-transcribe"
    # Used when the primary model can't handle a language (decided by G1 for Tamil) or fails.
    transcribe_fallback_model: str = "gemini-3.8-flash"
    # Languages routed straight to the fallback model (CSV/JSON list; G1 decides "ta").
    transcribe_fallback_languages: Annotated[list[str], NoDecode] = ["ta"]

    # --- Music bed (Lyria) ---
    music_enabled: bool = True
    music_model_chain: Annotated[list[str], NoDecode] = [
        "lyria-3.5",
        "lyria-3-pro-preview",
        "lyria-3-clip-preview",
    ]
    music_gain_db: float = -14.0     # score level before ducking
    ambience_gain_db: float = -18.0  # Omni clip ambience under narration
    music_duck_ratio: float = 8.0    # sidechain compression ratio while speech plays

    # --- Monthly AI budget guard (app-side estimate; Google/Anthropic billing is authoritative) ---
    monthly_budget_usd: float = 300.0
    # Google's spend-based rate limit per rolling 10 min (Tier 1 = $10, Tier 2+ = $200).
    spend_rate_limit_usd_per_10min: float = 9.0
    budget_mode: str = "block"   # block -> 402 when over; degrade -> run without Omni
    budget_warn_fraction: float = 0.8
    # Per-beat hold = max(narration_floor_seconds, audio_length + narration_pad_seconds).
    narration_floor_seconds: float = 1.5
    narration_pad_seconds: float = 0.4
    # TTS is a HARD 10 RPM / 100 RPD per-model quota. A shared token bucket paces
    tts_rpm_limit: int = 8  # < 10, with margin; the TTS bucket uses capacity=1 (no burst)
    tts_max_retries: int = 5  # transient 429/5xx attempts before a beat degrades to silence
    # Cap the honored retryDelay: a DAILY-quota 429 can carry a huge delay, and we must
    tts_retry_cap_seconds: float = 20.0

    # --- Database ---
    database_url: str = (
        "postgresql+asyncpg://storyboard:storyboard@localhost:5432/storyboard"
    )

    # --- Storage ---
    artifact_root: Path = Path("./artifacts_store")

    # --- Image generation: reliability + format + throughput (Phase 3) ---
    # Re-rolls (empty/no-image responses) + transient retries before giving up to a
    image_gen_max_retries: int = 5
    # Vertical 9:16 short-form. Sent straight to the image model (ImageConfig), so
    image_aspect_ratio: str = "9:16"
    # Output resolution tier for the image model ("1K" | "2K" | "4K"). 2K 9:16 =
    image_size: str = "2K"
    # Re-render a frame once when a vision check finds a character not matching their sheet.
    image_identity_check: bool = True
    image_identity_check_model: str = "gemini-3.8-flash"
    # Image-model thinking level ("" = model default, which is "medium" on Nano Banana 2.1).
    image_thinking_level: str = ""
    # Final film size (every tier normalizes to this).
    video_width: int = 1080
    video_height: int = 1920
    # Bounded concurrency: at most this many image calls in flight (asyncio.Semaphore).
    image_max_concurrency: int = 4
    # Global RPM ceiling enforced by a shared token bucket across ALL image calls
    image_rpm_limit: int = 60

    # --- Frame scaling (Phase 2) ---
    # Frame count = round(duration / seconds_per_frame). At 2.0s: 30s -> 15 frames,
    seconds_per_frame: float = 4.0

    # --- Motion (Phase 3 — Ken Burns) ---
    # ON: per-frame slow zoom (ffmpeg zoompan) + crossfades + caption overlay, so
    enable_motion: bool = True
    crossfade_seconds: float = 0.4
    editor_hard_cuts: bool = True  # cut between scenes; dissolve only where the script marks a time jump

    # --- Captions ---
    # Optional path to a .ttf for nicer subtitles. When unset we use Pillow's
    caption_font_path: str | None = None

    # --- Observability: per-stage status + log streaming ---
    # Live status/log streaming for the front end. The SSE endpoint polls the DB
    enable_log_streaming: bool = True
    log_poll_interval_seconds: float = 0.75
    # Captured log lines flow through a bounded, thread-safe queue drained by one
    log_queue_max: int = 10_000
    log_drain_batch: int = 200
    # Minimum level the stdlib-logging bridge captures into job_logs.
    log_capture_level: str = "INFO"

    # Optional fixed seed forwarded to the image model for reproducibility.
    image_seed: int | None = None

    # --- Frontend integration (CORS) ---
    # Browser origins allowed to call this API (the Next.js frontend). Accepts a
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000"]

    # --- Job retention (TTL) ---
    # Product rule: everything a job makes is deleted 24h after creation.
    job_ttl_seconds: int = 86400
    # The sweeper sleeps until the next job expires, at most this long (lets Neon scale to zero).
    job_sweep_interval_seconds: float = 3600.0

    # --- Hosting: sign-in + bring-your-own-key ---
    auth_mode: str = "none"        # none = local single user | jwt = Better Auth tokens from the web app
    auth_jwks_url: str = ""        # e.g. https://app.example.com/api/auth/jwks
    auth_issuer: str = ""          # the web app's BETTER_AUTH_URL
    auth_audience: str = "creator-agent-api"
    # Who may make films when auth_mode=jwt: emails, "@domain.com", or "*". Empty = nobody.
    allowed_users: Annotated[list[str], NoDecode] = []
    key_mode: str = "server"       # server = GEMINI_API_KEY above | byok = each request sends X-Gemini-Key
    asset_signing_secret: str = "" # signs media/SSE links; random per process when empty
    media_token_ttl_seconds: int = 43200  # 12h; the web app refreshes it before expiry
    user_monthly_budget_usd: float = 50.0  # per-user cap when auth_mode=jwt
    max_active_jobs_per_user: int = 1      # jwt mode only
    max_active_jobs: int = 3               # jwt mode only (server CPU / ffmpeg)

    # --- Abuse / cost guard (the create endpoint is public + un-authed) ---
    # Per-IP rate limit on POST /storyboard (slowapi syntax; empty -> disabled).
    rate_limit_create: str = "5/minute;30/day"
    # Master switch for rate limiting (tests set this false so the shared in-memory
    rate_limit_enabled: bool = True
    # Cloudflare Turnstile: verify a token from the create form server-side. OFF
    turnstile_enabled: bool = False
    turnstile_secret_key: str = ""
    turnstile_verify_url: str = (
        "https://challenges.cloudflare.com/turnstile/v0/siteverify"
    )

    # --- Story quality: narrative judge + reflection (Workstream 2/3) ---
    # A reflection node critiques+revises the script BEFORE media gen. The judge scores
    reflection_enabled: bool = True
    reflection_threshold: int = 90  # G0: strong scripts score ~93 on the Argon judge
    reflection_max_iters: int = 2
    # Deterministic judge score in MOCK mode (no API key). Drives the eval harness and
    mock_quality_score: int = 82

    # --- QA: force failure states (mock mode only ever yields clean frames) ---
    # Inject one degraded frame so the completed_with_warnings UI is testable.
    mock_force_degraded_frame: bool = False
    # Synthetic per-call token count in MOCK mode (no real usage_metadata). >0 makes the
    mock_token_usage: int = 0
    # Raise inside the named stage (e.g. "image_gen") so the failed + Resume UI is
    mock_force_stage_failure: str = ""

    # --- AI video (Gemini Omni, called directly; the preferred path) ---
    # omni   -> each scene is generated by Omni (incl. on-camera dialogue)
    video_provider: str = "omni"
    # Burn subtitles into the film (and transcribe dialogue clips so they match the speech).
    burn_subtitles: bool = False
    # Every scene must be AI video: a scene that can't get a clip (after a paced retry) stops
    # the job for a later Resume (finished clips are kept) instead of shipping a still.
    require_ai_video: bool = True
    ai_video_retry_delay_seconds: float = 90.0
    omni_model_chain: Annotated[list[str], NoDecode] = ["gemini-omni-1.2-flash", "gemini-omni-1.1-flash"]
    omni_resolution: str = "1080p"       # 360p | 720p | 1080p | 4k
    omni_rpm_limit: int = 8
    omni_max_concurrency: int = 4
    omni_timeout_seconds: float = 600.0
    omni_speaker_check: bool = True      # re-render a dialogue clip once if a voice is wrong
    film_review_enabled: bool = True     # a model watches the finished film and scores coherence
    film_review_model: str = "gemini-4-argon"
    omni_scene_qc: bool = True           # watch each clip vs its shot plan; re-render once on a failure
    omni_speaker_check_model: str = "gemini-3.8-flash"

    # --- Motion engine (external service: per-scene "live photo" motion clips) ---
    # DISTINCT from `enable_motion` above: `enable_motion` = the local Ken Burns
    motion_engine_enabled: bool = False
    motion_engine_url: str = "http://localhost:8090"
    # Splice the engine's clips into the final storyboard.mp4 (in place of Ken Burns
    motion_in_film: bool = True
    motion_clip_seconds: float = 4.0            # duration_hint sent to the engine
    motion_stage_budget_seconds: float = 1800.0 # overall ceiling for the motion stage
    motion_poll_interval_seconds: float = 2.0   # how often to poll a task's status
    motion_http_timeout_seconds: float = 30.0
    motion_max_concurrency: int = 30            # submit/poll all scene tasks in parallel

    @field_validator(
        "cors_origins",
        "tts_model_chain",
        "dialogue_fallback_voices",
        "transcribe_fallback_languages",
        "music_model_chain",
        "omni_model_chain",
        "allowed_users",
        mode="before",
    )
    @classmethod
    def _split_csv(cls, v: object) -> object:
        """Allow list settings as CSV ('a,b') in addition to a JSON list."""
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("["):  # JSON -> let pydantic handle it
                return v
            return [o.strip() for o in s.split(",") if o.strip()]
        return v

    @field_validator("database_url", mode="after")
    @classmethod
    def _asyncpg_url(cls, v: str) -> str:
        """Accept a pasted Neon/Postgres URL: force asyncpg, map sslmode -> ssl."""
        return normalize_database_url(v)

    @model_validator(mode="after")
    def _hosted_rules(self) -> "Settings":
        if self.auth_mode not in ("none", "jwt"):
            raise ValueError("AUTH_MODE must be 'none' or 'jwt'")
        if self.key_mode not in ("server", "byok"):
            raise ValueError("KEY_MODE must be 'server' or 'byok'")
        if self.auth_mode == "jwt":
            if not self.auth_jwks_url:
                raise ValueError("AUTH_MODE=jwt needs AUTH_JWKS_URL")
            if self.job_ttl_seconds > 86400:
                raise ValueError("hosted mode keeps jobs for at most 24 hours (JOB_TTL_SECONDS <= 86400)")
        return self

    @property
    def hosted(self) -> bool:
        return self.auth_mode == "jwt"

    @property
    def budget_cap_usd(self) -> float:
        return self.user_monthly_budget_usd if self.hosted else self.monthly_budget_usd

    @property
    def mock_mode(self) -> bool:
        """Mock when explicitly requested OR when no API key is configured."""
        return self.use_mock_providers or not self.gemini_api_key.strip()

    # --- Role routing helpers ---------------------------------------------- #
    def role_model(self, role: str) -> str:
        """Model id for an LLM role (script | judge | designer | publish), falling back
        to the legacy ``gemini_text_model`` when the role's model is empty."""
        return (getattr(self, f"{role}_model", "") or "").strip() or self.gemini_text_model

    def role_thinking(self, role: str) -> str:
        return (getattr(self, f"{role}_thinking", "") or "").strip()

    def role_temperature(self, role: str) -> float | None:
        return getattr(self, f"{role}_temperature", None)

    @property
    def tts_models(self) -> list[str]:
        """TTS models to try in order (legacy GEMINI_TTS_MODEL first when set)."""
        chain = [m for m in self.tts_model_chain if m]
        if self.gemini_tts_model.strip() and self.gemini_tts_model.strip() not in chain:
            chain.insert(0, self.gemini_tts_model.strip())
        return chain


def normalize_database_url(url: str) -> str:
    if not url.startswith(("postgres://", "postgresql://", "postgresql+asyncpg://")):
        return url
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    parts = urlsplit(url)
    query = []
    for k, val in parse_qsl(parts.query):
        if k == "channel_binding":
            continue
        if k == "sslmode":
            k, val = "ssl", ("require" if val in ("require", "verify-ca", "verify-full", "prefer") else val)
        query.append((k, val))
    return urlunsplit(("postgresql+asyncpg", parts.netloc, parts.path, urlencode(query), parts.fragment))


@lru_cache
def get_settings() -> Settings:
    return Settings()
