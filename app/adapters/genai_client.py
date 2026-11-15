"""Shared Interactions-API helpers. google-genai 3.x defaults to API v1, but Interactions
is only served on v1beta, so every call goes through ``interactions_create``."""

from __future__ import annotations

import base64

from app.core.config import Settings
from app.observability import usage
from app.observability.cost import token_cost

INTERACTIONS_API_VERSION = "v1beta"


def make_client(settings: Settings, *, timeout_seconds: float | None = None, attempts: int | None = None):
    """``attempts=1`` turns off the SDK's own retries (it honors Retry-After uncapped and silently)."""
    from google import genai  # lazy: mock mode needs no SDK import
    from google.genai import types

    kwargs: dict = {"api_key": settings.gemini_api_key}
    opts: dict = {}
    if timeout_seconds:
        opts["timeout"] = int(timeout_seconds * 1000)
    if attempts is not None:
        opts["retry_options"] = types.HttpRetryOptions(attempts=attempts)
    if opts:
        kwargs["http_options"] = types.HttpOptions(**opts)
    return genai.Client(**kwargs)


def interactions_create(client, **body):
    """``client.interactions.create`` pinned to the API version Interactions lives on."""
    body.setdefault("api_version", INTERACTIONS_API_VERSION)
    return client.interactions.create(**body)


def b64bytes(data) -> bytes:
    if data is None:
        return b""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    return base64.b64decode(data)


def b64str(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _by_modality(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items or []:
        mod = str(getattr(it, "modality", "") or "").lower()
        out[mod] = out.get(mod, 0) + int(getattr(it, "tokens", 0) or 0)
    return out


def interactions_usage_cost(model: str, usage_obj) -> tuple[float, dict[str, float], int, int]:
    """Price an Interactions ``usage`` object. Returns (cost, units, prompt, output)."""
    if usage_obj is None:
        return 0.0, {}, 0, 0
    prompt = int(getattr(usage_obj, "total_input_tokens", 0) or 0)
    out_mod = _by_modality(getattr(usage_obj, "output_tokens_by_modality", None))
    thoughts = int(getattr(usage_obj, "total_thought_tokens", 0) or 0)
    audio = out_mod.get("audio", 0)
    image = out_mod.get("image", 0)
    video = out_mod.get("video", 0)
    text_out = out_mod.get("text", 0) + thoughts
    total_out = int(getattr(usage_obj, "total_output_tokens", 0) or 0)
    if not out_mod and total_out:
        text_out = total_out + thoughts
    cost = token_cost(
        model,
        input_tokens=prompt,
        output_tokens=text_out,
        image_tokens=image,
        audio_tokens=audio,
        video_tokens=video,
    )
    units: dict[str, float] = {"input_tokens": prompt, "output_tokens": text_out}
    if audio:
        units["audio_tokens"] = audio
    return cost, units, prompt, text_out + audio + image + video


def record_interaction(category: str, model: str, interaction) -> float:
    """Record tokens + spend for one Interactions call; returns the cost."""
    cost, units, prompt, output = interactions_usage_cost(model, getattr(interaction, "usage", None))
    usage.record(prompt=prompt, output=output)
    usage.record_spend(category=category, model=model, cost_usd=cost, units=units)
    return cost


def is_daily_quota(exc: BaseException) -> bool:
    """A 429 that means the model's DAILY budget is gone (cascade to the next model)
    rather than a per-minute blip (wait + retry)."""
    msg = str(exc).lower()
    return ("429" in msg or "resource_exhausted" in msg or "quota" in msg) and (
        "per day" in msg or "perday" in msg or "per_day" in msg or "daily" in msg
    )
