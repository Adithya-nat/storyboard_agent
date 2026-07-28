"""Pillow helpers: frame geometry, subtitle captions, and placeholder cards.

Real frames come from Gemini. These helpers (1) normalize any frame to the
vertical short-form canvas with a cover-crop so there are NO pillarbox bars, and
(2) render a proper centered, lower-third **subtitle** with a scrim + outline —
the fix for the old tiny, corner-pinned default-bitmap-font caption. The caption
drawing (`_draw_caption`) is written ONCE here and reused by the burned-in path
(Wave 1) and, later, the ffmpeg overlay strip (Wave 2).

Placeholder cards keep the full pipeline + final video working without an API key
(mock mode) and give a visible, labelled frame in the rare all-frames-failed case.
"""

from __future__ import annotations

import hashlib
import io

from PIL import Image, ImageDraw, ImageFont

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

FRAME_SIZE = (720, 1280)  # 9:16 vertical short-form (W, H)


# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #
def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """A scalable font at ``size`` px. Prefer a configured .ttf, else Pillow's
    bundled default (load_default(size=...) returns a real sized TrueType font —
    the bare bitmap default could NOT be scaled, which is why captions were tiny)."""
    path = get_settings().caption_font_path
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            log.warning("caption_font_path %r not loadable; using default font", path)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow without the size kwarg
        return ImageFont.load_default()


# Map common "smart" punctuation to ASCII so the default font never renders a
# tofu box (□) in a caption — LLM narration loves em-dashes and curly quotes.
_PUNCT = {
    "—": "-", "–": "-",      # em / en dash
    "‘": "'", "’": "'",      # curly single quotes / apostrophe
    "“": '"', "”": '"',      # curly double quotes
    "…": "...",                    # ellipsis
    " ": " ",                      # non-breaking space
}


def _sanitize(text: str) -> str:
    return "".join(_PUNCT.get(ch, ch) for ch in text)


def _color_from_text(text: str) -> tuple[int, int, int]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Keep it muted/dark so overlaid caption text stays readable.
    return (40 + h[0] % 120, 40 + h[1] % 120, 40 + h[2] % 120)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def _fit(img: "Image.Image", size: tuple[int, int]) -> "Image.Image":
    """Scale to COVER ``size`` (fill the frame) then center-crop to exactly ``size``.

    Cover-crop (not scale-to-fit + pad) is what removes the black pillarbox bars:
    a near-square Gemini image fills the vertical canvas and the overflow is
    trimmed equally from both sides instead of leaving bars.
    """
    target_w, target_h = size
    if img.width == 0 or img.height == 0:
        return Image.new("RGB", size, (0, 0, 0))
    scale = max(target_w / img.width, target_h / img.height)
    new_w = max(target_w, round(img.width * scale))
    new_h = max(target_h, round(img.height * scale))
    scaled = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return scaled.crop((left, top, left + target_w, top + target_h))


# --------------------------------------------------------------------------- #
# Captions (shared renderer)
# --------------------------------------------------------------------------- #
def _wrap_to_width(draw: "ImageDraw.ImageDraw", text: str, font, max_width: int) -> list[str]:
    """Greedy word-wrap by MEASURED pixel width (not a fixed char count)."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if not current or draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_caption(img: "Image.Image", caption: str, size: tuple[int, int]) -> None:
    """Draw a centered, lower-third subtitle (scrim band + outlined text) IN PLACE.

    Shared by burn_caption (Wave 1) and the overlay strip (Wave 2). No-op for an
    empty caption.

        ┌───────────────────────┐
        │                       │
        │        (image)        │
        │                       │
        │  ░░ centered subtitle ░░  │  <- scrim band, ~6% above bottom
        └───────────────────────┘
    """
    if not caption.strip():
        return
    width, height = size
    draw = ImageDraw.Draw(img, "RGBA")

    font_size = max(20, round(width * 0.060))  # ~43px at 720w
    font = _load_font(font_size)
    max_text_w = int(width * 0.88)
    lines = _wrap_to_width(draw, _sanitize(caption.strip()), font, max_text_w)[:3]

    ascent, descent = font.getmetrics()
    line_h = ascent + descent + round(font_size * 0.18)
    pad = round(font_size * 0.55)
    band_h = line_h * len(lines) + pad * 2
    bottom_margin = round(height * 0.06)
    band_top = height - band_h - bottom_margin

    # Semi-transparent scrim for legibility on any background.
    draw.rectangle([0, band_top, width, band_top + band_h], fill=(0, 0, 0, 140))

    # Centered, outlined lines.
    stroke = max(2, font_size // 14)
    y = band_top + pad
    for line in lines:
        text_w = draw.textlength(line, font=font)
        x = (width - text_w) / 2
        draw.text(
            (x, y),
            line,
            font=font,
            fill=(255, 255, 255),
            stroke_width=stroke,
            stroke_fill=(0, 0, 0),
        )
        y += line_h


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def burn_caption(
    image_bytes: bytes, caption: str, *, size: tuple[int, int] = FRAME_SIZE
) -> bytes:
    """Normalize a frame to ``size`` (cover-crop, no bars) and burn a centered
    subtitle along the lower third. (Wave 2 moves the caption to an ffmpeg overlay
    so the Ken Burns zoom doesn't drift it; the rendering function is shared.)"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = _fit(img, size)
    _draw_caption(img, caption, size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_caption_overlay(caption: str, *, size: tuple[int, int] = FRAME_SIZE) -> bytes:
    """A fully transparent RGBA PNG with ONLY the subtitle strip drawn, for the
    Ken Burns pipeline: ffmpeg overlays it on top of the zooming frame so the
    caption stays fixed (a burned-in caption would scale/drift with the zoom).
    Same renderer as burn_caption, so captions look identical either way."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    _draw_caption(img, caption, size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_card(
    title: str,
    subtitle: str = "",
    *,
    degraded: bool = False,
    size: tuple[int, int] = FRAME_SIZE,
) -> bytes:
    """Render a labelled card as PNG bytes (mock frame, or the rare all-failed
    placeholder). Centered so it reads as intentional, with the subtitle rendered
    by the same caption renderer as real frames."""
    width, height = size
    bg = (90, 30, 30) if degraded else _color_from_text(title)
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)

    banner = "FRAME UNAVAILABLE" if degraded else "MOCK FRAME"
    banner_font = _load_font(max(18, round(width * 0.045)))
    bw = draw.textlength(banner, font=banner_font)
    draw.text(((width - bw) / 2, round(height * 0.07)), banner, font=banner_font, fill=(230, 230, 230))

    title_font = _load_font(max(20, round(width * 0.058)))
    lines = _wrap_to_width(draw, _sanitize(title), title_font, int(width * 0.84))[:6]
    ascent, descent = title_font.getmetrics()
    line_h = ascent + descent + round(width * 0.058 * 0.2)
    y = height // 2 - (line_h * len(lines)) // 2
    for line in lines:
        tw = draw.textlength(line, font=title_font)
        draw.text(((width - tw) / 2, y), line, font=title_font, fill=(255, 255, 255))
        y += line_h

    _draw_caption(img, subtitle, size)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
