"""AI spend: a dated price table (scheduled price changes apply automatically) and per-call
spend events. Estimates only; provider billing is authoritative. Unknown models price at $0."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

log = logging.getLogger("app.observability.cost")

# Spend categories shown on the dashboard.
CATEGORIES = ("text", "image", "video", "tts", "transcribe", "music")

_PER_M = 1_000_000.0


@dataclass(frozen=True)
class TokenPrice:
    """Per-1M-token prices. ``output_image``/``output_audio``/``output_video`` price the
    media tokens a multimodal model emits; text/thinking output uses ``output``."""

    input: float
    output: float = 0.0
    output_image: float = 0.0
    output_audio: float = 0.0
    output_video: float = 0.0


@dataclass(frozen=True)
class PriceRule:
    model: str                 # exact id, or a prefix ending in "*"
    effective_from: date
    tokens: TokenPrice | None = None
    per_second: dict[str, float] | None = None  # video: resolution -> $/s
    per_unit: float | None = None               # music: $ per song

    def matches(self, model: str) -> bool:
        if self.model.endswith("*"):
            return model.startswith(self.model[:-1])
        return model == self.model


_D0 = date(2026, 1, 1)

PRICE_RULES: tuple[PriceRule, ...] = (
    # --- Gemini text ---
    PriceRule("gemini-4-carbon", _D0, TokenPrice(2.00, 10.00)),
    PriceRule("gemini-4-carbon", date(2026, 12, 1), TokenPrice(4.00, 20.00)),
    PriceRule("gemini-4-argon", _D0, TokenPrice(2.00, 10.00)),
    PriceRule("gemini-4-argon", date(2026, 12, 1), TokenPrice(4.00, 20.00)),
    PriceRule("gemini-3.8-flash", _D0, TokenPrice(0.75, 3.75)),
    PriceRule("gemini-3.8-flash", date(2027, 1, 1), TokenPrice(1.50, 7.50)),
    PriceRule("gemini-3-flash-preview", _D0, TokenPrice(0.50, 3.00)),
    # --- Gemini image (token-priced; the image-token rate is what makes a frame cost) ---
    PriceRule("gemini-nano-banana-2.1", _D0, TokenPrice(1.50, 7.50, output_image=30.00)),
    PriceRule("gemini-3-pro-image", _D0, TokenPrice(2.00, 12.00, output_image=120.00)),
    PriceRule("gemini-3.1-flash-image", _D0, TokenPrice(0.50, 3.00, output_image=60.00)),
    # --- TTS / transcription ---
    PriceRule("gemini-3.8-flash-tts", _D0, TokenPrice(0.50, output_audio=9.00)),
    PriceRule("gemini-3.8-flash-tts", date(2027, 1, 1), TokenPrice(1.00, output_audio=18.00)),
    PriceRule("gemini-3.8-flash-lite-tts", _D0, TokenPrice(0.25, output_audio=4.50)),
    PriceRule("gemini-3.5-transcribe", _D0, TokenPrice(2.00, 12.00)),
    # --- Video (Omni): effective $/s by resolution (pricing-page footnote) ---
    PriceRule(
        "gemini-omni-*",
        _D0,
        per_second={"360p": 0.023, "720p": 0.07, "1080p": 0.104, "4k": 0.209},
    ),
    # --- Music (Lyria): per song ---
    PriceRule("lyria-3.5", _D0, per_unit=0.08),
    PriceRule("lyria-3-pro-preview", _D0, per_unit=0.08),
    PriceRule("lyria-3-clip-preview", _D0, per_unit=0.04),
    # --- Claude (only used if adopted by the G0 bake-off) ---
    PriceRule("claude-fable-5-2", _D0, TokenPrice(8.00, 40.00)),
    PriceRule("claude-opus-5-5", _D0, TokenPrice(4.00, 20.00)),
    PriceRule("claude-sonnet-5-5", _D0, TokenPrice(2.00, 10.00)),
    PriceRule("claude-haiku-5-5", _D0, TokenPrice(0.10, 0.50)),
)

_warned_unpriced: set[str] = set()


def _strip_provider(model: str) -> str:
    return model.split(":", 1)[1] if ":" in model else model


def price_rule(model: str, on: date | None = None) -> PriceRule | None:
    """The rule in effect for ``model`` on ``on`` (latest effective_from <= on)."""
    model = _strip_provider(model)
    on = on or datetime.now(timezone.utc).date()
    candidates = [r for r in PRICE_RULES if r.matches(model) and r.effective_from <= on]
    if not candidates:
        if model not in _warned_unpriced:
            _warned_unpriced.add(model)
            log.warning("no price for model %r; its spend is recorded as $0", model)
        return None
    return max(candidates, key=lambda r: r.effective_from)


def token_cost(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    image_tokens: int = 0,
    audio_tokens: int = 0,
    video_tokens: int = 0,
    on: date | None = None,
) -> float:
    """Cost of one token-priced call. ``output_tokens`` = text + thinking output; media
    output tokens are passed separately (they are priced differently)."""
    rule = price_rule(model, on)
    if rule is None or rule.tokens is None:
        return 0.0
    p = rule.tokens
    return (
        input_tokens * p.input
        + output_tokens * p.output
        + image_tokens * p.output_image
        + audio_tokens * p.output_audio
        + video_tokens * p.output_video
    ) / _PER_M


def video_cost(model: str, seconds: float, resolution: str = "1080p", on: date | None = None) -> float:
    rule = price_rule(model, on)
    if rule is None or not rule.per_second:
        return 0.0
    rate = rule.per_second.get(resolution.lower(), rule.per_second.get("720p", 0.0))
    return max(0.0, seconds) * rate


def unit_cost(model: str, units: float = 1.0, on: date | None = None) -> float:
    rule = price_rule(model, on)
    if rule is None or rule.per_unit is None:
        return 0.0
    return units * rule.per_unit


@dataclass
class SpendEvent:
    """One priced provider call. ``units`` is human-readable usage (tokens, images,
    seconds, songs) for the dashboard; ``cost_usd`` is the estimate."""

    category: str
    provider: str
    model: str
    cost_usd: float
    units: dict[str, float] = field(default_factory=dict)
    stage: str | None = None
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _modality_counts(details) -> dict[str, int]:
    """Sum token counts by modality from a google-genai ``*_tokens_details`` list."""
    out: dict[str, int] = {}
    for d in details or []:
        modality = getattr(getattr(d, "modality", None), "name", None) or str(getattr(d, "modality", ""))
        out[modality.upper()] = out.get(modality.upper(), 0) + int(getattr(d, "token_count", 0) or 0)
    return out


def gemini_usage_cost(model: str, usage_metadata, on: date | None = None) -> tuple[float, dict[str, float]]:
    """Price a google-genai ``usage_metadata`` (text, image or TTS call).

    Output media tokens (IMAGE / AUDIO / VIDEO) come from ``candidates_tokens_details``;
    the remaining candidate tokens plus ``thoughts_token_count`` are text/thinking output.
    Defensive: any missing field counts as 0.
    """
    if usage_metadata is None:
        return 0.0, {}
    prompt = int(getattr(usage_metadata, "prompt_token_count", 0) or 0)
    candidates = int(getattr(usage_metadata, "candidates_token_count", 0) or 0)
    thoughts = int(getattr(usage_metadata, "thoughts_token_count", 0) or 0)
    media = _modality_counts(getattr(usage_metadata, "candidates_tokens_details", None))
    image = media.get("IMAGE", 0)
    audio = media.get("AUDIO", 0)
    video = media.get("VIDEO", 0)
    text_out = max(0, candidates - image - audio - video) + thoughts
    cost = token_cost(
        model,
        input_tokens=prompt,
        output_tokens=text_out,
        image_tokens=image,
        audio_tokens=audio,
        video_tokens=video,
        on=on,
    )
    units = {"input_tokens": prompt, "output_tokens": text_out}
    if image:
        units["image_tokens"] = image
    if audio:
        units["audio_tokens"] = audio
    return cost, units


# --- Pre-flight estimates (budget guard + UI "estimated cost") -------------- #
# Calibrated from the plan's per-video math; refined by real run_ledger averages once
# enough runs exist (see estimate_job_cost(history=...)).
_BASE_ESTIMATE_PER_SECOND = {
    # style -> USD per finished second; measured on real 30s films (Nov 2026: $6.02-7.60).
    # Used until a style has MIN_RUNS_FOR_HISTORY completed runs of its own.
    "narrated": 0.225,
    "dialogue": 0.230,
    "mixed": 0.230,
}


def estimate_job_cost(
    *,
    duration: int,
    style: str = "narrated",
    omni_enabled: bool = True,
    history_avg_per_second: float | None = None,
) -> float:
    """Estimated USD for one run. Uses the real average from past runs when provided."""
    if history_avg_per_second is not None and history_avg_per_second > 0:
        per_s = history_avg_per_second
    else:
        per_s = _BASE_ESTIMATE_PER_SECOND.get(style, 0.155)
    if not omni_enabled:
        per_s = max(0.0, per_s - 0.15)  # Omni share of the measured per-second cost
    return round(duration * per_s, 4)
