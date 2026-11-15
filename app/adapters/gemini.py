"""Gemini image-generation adapter.

    generate_image(prompt, ref_image=None, seed=None, caption="") -> PNG bytes

Consistency lever: pass the one-time ``ref_image`` (style reference) + the style
text in the prompt so every frame matches. Vertical short-form via
``image_config.aspect_ratio`` so frames come back the right shape (no pillarbox).

Reliability (Phase 3) — the fix for silent "degraded" frames:

    generate_content
        ├─ image part present ──────────────────────────▶ return bytes
        └─ NO image part ─▶ classify finish_reason / block_reason
                ├─ genuine block (SAFETY / PROHIBITED / RECITATION / BLOCKLIST / SPII)
                │       ──▶ PermanentProviderError  (don't waste re-rolls)
                └─ transient empty (NO_IMAGE / STOP / OTHER / unset)
                        ──▶ EmptyImageError  ──▶ tenacity RE-ROLLS (up to N)

Image models stochastically return text-only on a safe prompt; the old code read
that as a permanent safety block and replaced the frame with a placeholder. Now we
re-roll, which almost always recovers a real image.

Every real attempt (frames + style_ref + re-rolls) takes a token from the shared
``get_image_limiter`` bucket, so concurrency + retries provably stay under the
model's RPM. In mock mode a deterministic labelled card is returned.
"""

from __future__ import annotations

from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.adapters.errors import (
    EmptyImageError,
    PermanentProviderError,
    TransientProviderError,
)
from app.adapters.llm import _is_transient, parse_retry_after, wait_retry_after
from app.adapters.rate_limit import get_image_limiter
from app.core.config import Settings, get_settings
from app.core.images import render_card
from app.core.logging import get_logger
from app.observability import usage
from app.observability.spend_rate import get_governor

log = get_logger(__name__)

# FinishReason / block-reason names that mean a genuine, non-retryable block.
# Names are matched case-insensitively and cover the 2.x + 3.x image variants.
_BLOCK_REASONS = {
    "SAFETY",
    "IMAGE_SAFETY",
    "PROHIBITED_CONTENT",
    "IMAGE_PROHIBITED_CONTENT",
    "RECITATION",
    "IMAGE_RECITATION",
    "BLOCKLIST",
    "SPII",
}


class GeminiImageAdapter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    def generate_image(
        self,
        prompt: str,
        *,
        ref_image: bytes | None = None,
        refs: list[bytes | tuple[str, bytes]] | None = None,
        seed: int | None = None,
        caption: str = "",
    ) -> bytes:
        """Prompt (+ optional reference images) -> image bytes (JPEG or PNG; callers
        sniff the format with ``image_mime``). ``refs`` carries the style reference plus
        any character sheets for this shot (Nano Banana 2.1 accepts up to 14);
        ``ref_image`` is the legacy single-reference form and is prepended to ``refs``."""
        if self.settings.mock_mode:
            log.info("GeminiImageAdapter mock_mode -> card")
            if self.settings.mock_token_usage:
                usage.record(total=self.settings.mock_token_usage)
            return render_card(prompt, caption)

        # Settings-driven retry: re-roll transient empties (EmptyImageError is a
        # TransientProviderError) AND transient 429/5xx, up to image_gen_max_retries.
        # Genuine blocks raise PermanentProviderError (not in the retry set) and
        # propagate immediately so the caller can neighbor-fill.
        retrying = Retrying(
            retry=retry_if_exception_type(TransientProviderError),
            stop=stop_after_attempt(max(1, self.settings.image_gen_max_retries)),
            # Honor a 429's own retryDelay (capped) when present, else exponential.
            # Empty-image re-rolls carry no retry_after, so they keep backing off.
            wait=wait_retry_after(wait_exponential(multiplier=1, min=1, max=20), cap=20.0),
            reraise=True,
        )
        all_refs = ([ref_image] if ref_image is not None else []) + list(refs or [])
        return retrying(self._generate_once, prompt, all_refs[:14], seed)

    def _generate_once(self, prompt: str, refs: list | bytes | None, seed: int | None) -> bytes:
        from google.genai import types

        # RPM guard: every attempt (incl. re-rolls + the style_ref) takes a token.
        get_image_limiter(self.settings.image_rpm_limit).acquire()
        get_governor(self.settings.spend_rate_limit_usd_per_10min).acquire(_image_estimate(self.settings))

        if isinstance(refs, (bytes, bytearray)):
            refs = [bytes(refs)]  # legacy single-reference call shape
        contents: list = [prompt]
        for ref in refs or []:
            # A (label, bytes) ref is preceded by its label so the model knows whose image it is.
            label, data = ref if isinstance(ref, tuple) else (None, ref)
            if label:
                contents.append(label)
            contents.append(types.Part.from_bytes(data=data, mime_type=image_mime(data)))

        # response_modalities MUST include "IMAGE" or the model returns text only.
        # image_config.aspect_ratio asks for the right shape up front (vertical 9:16),
        # so frames don't come back square + padded. seed is forwarded only if set.
        image_config: dict = {"aspect_ratio": self.settings.image_aspect_ratio}
        if self.settings.image_size:
            image_config["image_size"] = self.settings.image_size
        config_kwargs: dict = {
            "response_modalities": ["IMAGE"],
            "image_config": types.ImageConfig(**image_config),
        }
        if self.settings.image_thinking_level:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=self.settings.image_thinking_level
            )
        if seed is not None:
            config_kwargs["seed"] = seed

        try:
            resp = self.client.models.generate_content(
                model=self.settings.gemini_image_model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            if _is_transient(exc):
                raise TransientProviderError(
                    str(exc), retry_after=parse_retry_after(exc)
                ) from exc
            raise PermanentProviderError(str(exc)) from exc

        usage.record_gemini_call(
            "image", self.settings.gemini_image_model, getattr(resp, "usage_metadata", None)
        )
        image_bytes = _extract_image_bytes(resp)
        if image_bytes is not None:
            return image_bytes

        # No image part — re-roll a transient empty, give up on a genuine block.
        is_block, reason = _classify_no_image(resp)
        if is_block:
            raise PermanentProviderError(f"Gemini blocked the image (reason={reason}).")
        log.info("image gen returned no image part (reason=%s); re-rolling", reason)
        raise EmptyImageError(f"Gemini returned no image part (reason={reason}).")


def _image_estimate(settings: Settings) -> float:
    """Rough per-image cost for spend-rate pacing (actual cost is recorded from usage)."""
    big = settings.image_size.upper() == "4K"
    if "pro" in settings.gemini_image_model:
        return 0.24 if big else 0.134
    return 0.113 if big else 0.05


def image_mime(data: bytes) -> str:
    """Sniff an image's real MIME type from its magic bytes (defaults to PNG)."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _extract_image_bytes(resp) -> bytes | None:
    """Pull the first inline image part out of a google-genai response."""
    candidates = getattr(resp, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return inline.data
    return None


def _enum_name(value) -> str:
    """Robust name for an enum/str finish_reason across SDK versions."""
    return getattr(value, "name", str(value)).upper()


def _classify_no_image(resp) -> tuple[bool, str]:
    """Decide whether a no-image response is a genuine block (don't retry) or a
    transient empty (re-roll). Reads prompt_feedback.block_reason + the candidate
    finish_reason defensively — these fields/enums shift across SDK versions.

    Returns (is_block, reason_name).
    """
    # Prompt-level rejection: the whole request was blocked -> always permanent.
    pf = getattr(resp, "prompt_feedback", None)
    block_reason = getattr(pf, "block_reason", None) if pf is not None else None
    if block_reason:
        return True, f"prompt_block:{_enum_name(block_reason)}"

    # Candidate-level finish reason: only the known block set is permanent.
    for candidate in getattr(resp, "candidates", None) or []:
        finish = getattr(candidate, "finish_reason", None)
        if finish:
            name = _enum_name(finish)
            return name in _BLOCK_REASONS, name
    return False, "NO_CANDIDATES"
