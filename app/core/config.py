"""Application settings, loaded from environment / .env via pydantic-settings.

API keys are placeholders the user fills in later. When ``GEMINI_API_KEY`` is
empty we automatically flip into MOCK mode so the whole pipeline still runs
end-to-end locally (and in tests) without real API calls.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Gemini ---
    gemini_api_key: str = ""
    # Text + image models are config-driven so upgrading is a .env change, no code
    # edit. Resolve the exact IDs for your tier with client.models.list() (see
    # README). Defaults are the known-good 2.5 family; Phase 3 recommends upgrading
    # text -> best Gemini-3 model, image -> Nano Banana 2 (Gemini 3.1 Flash Image).
    gemini_text_model: str = "gemini-3-flash-preview"
    gemini_image_model: str = "gemini-3.1-flash-image"
    use_mock_providers: bool = False

    # --- Gemini TTS narration (Workstream 5: per-beat SYNCED voiceover, ON by default) ---
    # Each caption is voiced and its image is held for exactly the narration's length
    # (audio-driven holds), so audio and visuals are synced per beat. Flash-TTS is ~100
    # RPD and this makes one call PER beat (15-30/job), so a narrated user is limited to
    # ~3-6 jobs/day — the graceful fallback (silent beat -> single track -> silent video)
    # keeps a job from ever failing on quota.
    enable_tts: bool = True
    gemini_tts_model: str = "gemini-2.5-flash-preview-tts"
    tts_voice: str = "Kore"
    # Per-beat hold = max(narration_floor_seconds, audio_length + narration_pad_seconds).
    narration_floor_seconds: float = 1.5
    narration_pad_seconds: float = 0.4
    # TTS is a HARD 10 RPM / 100 RPD per-model quota. A shared token bucket paces
    # calls UNDER the per-minute cap (exactly like image_rpm_limit does for image);
    # on the rare 429 we honor the server's retryDelay (capped) instead of guessing.
    tts_rpm_limit: int = 8  # < 10, with margin; the TTS bucket uses capacity=1 (no burst)
    tts_max_retries: int = 5  # transient 429/5xx attempts before a beat degrades to silence
    # Cap the honored retryDelay: a DAILY-quota 429 can carry a huge delay, and we must
    # not sleep that long inside a job — cap it so daily exhaustion degrades fast.
    tts_retry_cap_seconds: float = 20.0

    # --- Database ---
    database_url: str = (
        "postgresql+asyncpg://storyboard:storyboard@localhost:5432/storyboard"
    )

    # --- Storage ---
    artifact_root: Path = Path("./artifacts_store")

    # --- Image generation: reliability + format + throughput (Phase 3) ---
    # Re-rolls (empty/no-image responses) + transient retries before giving up to a
    # neighbor-fill. The image model stochastically returns text-only on safe
    # prompts; re-rolling almost always recovers a real image.
    image_gen_max_retries: int = 5
    # Vertical 9:16 short-form. Sent straight to the image model (ImageConfig), so
    # frames come back the right shape instead of square-padded (no pillarbox bars).
    image_aspect_ratio: str = "9:16"
    # Bounded concurrency: at most this many image calls in flight (asyncio.Semaphore).
    image_max_concurrency: int = 4
    # Global RPM ceiling enforced by a shared token bucket across ALL image calls
    # (frames + style_ref + re-rolls). Kept comfortably under Nano Banana 2's 100 RPM
    # so a single user never spikes near the limit. (1K RPD / 30 frames ~= 33 jobs/day.)
    image_rpm_limit: int = 60

    # --- Frame scaling (Phase 2) ---
    # Frame count = round(duration / seconds_per_frame). At 2.0s: 30s -> 15 frames,
    # 60s -> 30 frames. Also the floor each frame is held in the assembled video.
    seconds_per_frame: float = 2.0

    # --- Motion (Phase 3 — Ken Burns) ---
    # ON: per-frame slow zoom (ffmpeg zoompan) + crossfades + caption overlay, so
    # held stills read as filmed motion. OFF: the simple concat path (burned-in
    # captions, no motion) — used by hermetic tests and as a no-ffmpeg-features
    # fallback. Per-clip hold is solved so total == target duration even with xfades.
    enable_motion: bool = True
    crossfade_seconds: float = 0.4

    # --- Captions ---
    # Optional path to a .ttf for nicer subtitles. When unset we use Pillow's
    # scalable default font (load_default(size=...)) — still centered + properly sized.
    caption_font_path: str | None = None

    # --- Observability: per-stage status + log streaming ---
    # Live status/log streaming for the front end. The SSE endpoint polls the DB
    # every log_poll_interval_seconds (invisible latency for a minutes-long job —
    # and it survives a future move to a separate worker process, unlike an
    # in-process bus). OFF -> the /events endpoint is disabled (pull still works).
    enable_log_streaming: bool = True
    log_poll_interval_seconds: float = 0.75
    # Captured log lines flow through a bounded, thread-safe queue drained by one
    # async task. Drop-OLDEST on overflow so a log flood never OOMs or blocks a
    # job; the drop is surfaced as a synthetic log line, never silent.
    log_queue_max: int = 10_000
    log_drain_batch: int = 200
    # Minimum level the stdlib-logging bridge captures into job_logs.
    log_capture_level: str = "INFO"

    # Optional fixed seed forwarded to the image model for reproducibility.
    # OFF by default: image-endpoint seed support is version-dependent, so
    # reference-image conditioning (style_ref.png) is the load-bearing consistency
    # lever; seed is best-effort if set.
    image_seed: int | None = None

    # --- Frontend integration (CORS) ---
    # Browser origins allowed to call this API (the Next.js frontend). Accepts a
    # JSON list or a comma-separated string in the env (see the validator below).
    # NoDecode: stop pydantic-settings from JSON-parsing this env var before the
    # _split_origins validator runs (CORS_ORIGINS=http://localhost:3000 in .env
    # would otherwise fail to parse under pydantic-settings >=2.14).
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000"]

    # --- Job retention (TTL) ---
    # A job + its artifacts are kept this long AFTER CREATION, then expired
    # (GET -> 410 Gone) and swept from the DB + disk. The frontend shows a live
    # countdown derived from created_at + this value.
    job_ttl_seconds: int = 3600
    # How often the background sweeper purges expired jobs (seconds).
    job_sweep_interval_seconds: float = 300.0

    # --- Abuse / cost guard (the create endpoint is public + un-authed) ---
    # Per-IP rate limit on POST /storyboard (slowapi syntax; empty -> disabled).
    rate_limit_create: str = "5/minute;30/day"
    # Master switch for rate limiting (tests set this false so the shared in-memory
    # limiter can't carry counts across tests and 429 a legitimate assertion).
    rate_limit_enabled: bool = True
    # Cloudflare Turnstile: verify a token from the create form server-side. OFF
    # by default so local/mock dev (and tests) never need a key.
    turnstile_enabled: bool = False
    turnstile_secret_key: str = ""
    turnstile_verify_url: str = (
        "https://challenges.cloudflare.com/turnstile/v0/siteverify"
    )

    # --- Story quality: narrative judge + reflection (Workstream 2/3) ---
    # A reflection node critiques+revises the script BEFORE media gen. The judge scores
    # narrative cohesion 0-100; the loop revises until `overall >= reflection_threshold`
    # or `reflection_max_iters` is hit, then keeps the best-scoring version (accept-best).
    # Threshold/iters are deliberately modest: LLM-judge scores are noisy and the text
    # model shares the same ~10 RPM quota, so the loop is `1 + max_iters*(revise+judge)`
    # calls/job (= up to 5 at max_iters=2). Raising the threshold mostly buys more calls.
    reflection_enabled: bool = True
    reflection_threshold: int = 95
    reflection_max_iters: int = 5
    # Deterministic judge score in MOCK mode (no API key). Drives the eval harness and
    # the reflection loop offline; set LOW (< threshold) to exercise the cap+accept-best
    # path in an end-to-end mock run, HIGH (>= threshold) to exercise early-accept.
    mock_quality_score: int = 82

    # --- QA: force failure states (mock mode only ever yields clean frames) ---
    # Inject one degraded frame so the completed_with_warnings UI is testable.
    mock_force_degraded_frame: bool = False
    # Synthetic per-call token count in MOCK mode (no real usage_metadata). >0 makes the
    # end-to-end token UI demoable/testable without an API key. 0 -> mock runs show 0.
    mock_token_usage: int = 0
    # Raise inside the named stage (e.g. "image_gen") so the failed + Resume UI is
    # testable. Empty -> off.
    mock_force_stage_failure: str = ""

    # --- Motion engine (external service: per-scene "live photo" motion clips) ---
    # DISTINCT from `enable_motion` above: `enable_motion` = the local Ken Burns
    # zoompan vs plain concat inside the assembler. `motion_engine_enabled` = call
    # the separate storyboard_engine service to animate each frame. OFF by default
    # so the graph, API, and every existing test are byte-identical to today.
    motion_engine_enabled: bool = False
    motion_engine_url: str = "http://localhost:8090"
    # Splice the engine's clips into the final storyboard.mp4 (in place of Ken Burns
    # still-zooms). Only effective when `enable_motion` is also true — the concat
    # path has no per-clip renderer to swap. Per-scene fallback to zoompan if a clip
    # is missing, so the film never regresses.
    motion_in_film: bool = True
    motion_clip_seconds: float = 4.0            # duration_hint sent to the engine
    motion_stage_budget_seconds: float = 1800.0 # overall ceiling for the motion stage
    motion_poll_interval_seconds: float = 2.0   # how often to poll a task's status
    motion_http_timeout_seconds: float = 30.0
    motion_max_concurrency: int = 30            # submit/poll all scene tasks in parallel

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        """Allow CORS_ORIGINS as CSV ('a,b') in addition to a JSON list."""
        if isinstance(v, str):
            s = v.strip()
            if not s or s.startswith("["):  # empty or JSON -> let pydantic handle it
                return v
            return [o.strip() for o in s.split(",") if o.strip()]
        return v

    @property
    def mock_mode(self) -> bool:
        """Mock when explicitly requested OR when no API key is configured."""
        return self.use_mock_providers or not self.gemini_api_key.strip()


@lru_cache
def get_settings() -> Settings:
    return Settings()
