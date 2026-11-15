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
import re

from PIL import Image, ImageDraw, ImageFont

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

FRAME_SIZE = (720, 1280)  # legacy default (mock cards); the film uses frame_size()


def frame_size() -> tuple[int, int]:
    s = get_settings()
    return (s.video_width, s.video_height)


_TAMIL = re.compile(r"[\u0B80-\u0BFF]")


def needs_complex_shaping(text: str) -> bool:
    return bool(_TAMIL.search(text))


def raqm_available() -> bool:
    from PIL import features

    return bool(features.check("raqm"))


# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #
def _load_font(size: int, text: str = "") -> ImageFont.FreeTypeFont:
    """A scalable font at ``size`` px: the Tamil font (raqm-shaped) for Tamil text,
    else a configured .ttf, else Pillow's bundled scalable default."""
    if needs_complex_shaping(text):
        tamil = get_settings().tamil_font_path
        try:
            engine = ImageFont.Layout.RAQM if raqm_available() else ImageFont.Layout.BASIC
            return ImageFont.truetype(tamil, size, layout_engine=engine)
        except OSError:
            log.warning("Tamil font %r not loadable; Tamil captions may render incorrectly", tamil)
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
    for para in text.split("\n"):  # one block per speaker line
        lines += _wrap_para(draw, para, font, max_width)
    return lines


def _wrap_para(draw: "ImageDraw.ImageDraw", text: str, font, max_width: int) -> list[str]:
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
    """Draw a centered lower-third subtitle (scrim + outlined text) in place. Each word uses a
    font that has its glyphs (Tamil font for Tamil words, Latin font otherwise)."""
    if not caption.strip():
        return
    width, height = size
    draw = ImageDraw.Draw(img, "RGBA")
    font_size = max(20, round(width * 0.055))
    text = _sanitize(caption.strip())
    latin = _load_font(font_size)
    tamil = _load_font(font_size, "த") if needs_complex_shaping(text) else latin
    font_for = lambda w: tamil if needs_complex_shaping(w) else latin  # noqa: E731
    space = draw.textlength(" ", font=latin)
    max_text_w = int(width * 0.88)

    lines: list[list[str]] = []
    for para in text.split("\n"):
        cur: list[str] = []
        cur_w = 0.0
        for word in para.split():
            w = draw.textlength(word, font=font_for(word))
            if cur and cur_w + space + w > max_text_w:
                lines.append(cur)
                cur, cur_w = [word], w
            else:
                cur_w += (space if cur else 0) + w
                cur.append(word)
        if cur:
            lines.append(cur)
    lines = lines[:4]

    ascent = max(latin.getmetrics()[0], tamil.getmetrics()[0])
    descent = max(latin.getmetrics()[1], tamil.getmetrics()[1])
    line_h = ascent + descent + round(font_size * 0.18)
    pad = round(font_size * 0.55)
    band_h = line_h * len(lines) + pad * 2
    band_top = height - band_h - round(height * 0.06)
    draw.rectangle([0, band_top, width, band_top + band_h], fill=(0, 0, 0, 140))

    stroke = max(2, font_size // 14)
    y = band_top + pad
    for words in lines:
        widths = [draw.textlength(w, font=font_for(w)) for w in words]
        x = (width - (sum(widths) + space * (len(words) - 1))) / 2
        for w, ww in zip(words, widths):
            f = font_for(w)
            # Baseline-align mixed fonts on the shared ascent.
            draw.text((x, y + ascent - f.getmetrics()[0]), w, font=f, fill=(255, 255, 255),
                      stroke_width=stroke, stroke_fill=(0, 0, 0))
            x += ww + space
        y += line_h


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def burn_caption(image_bytes: bytes, caption: str, *, size: tuple[int, int] | None = None) -> bytes:
    """Normalize a frame to ``size`` (cover-crop, no bars) and burn a centered
    subtitle along the lower third. (Wave 2 moves the caption to an ffmpeg overlay
    so the Ken Burns zoom doesn't drift it; the rendering function is shared.)"""
    size = size or frame_size()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = _fit(img, size)
    _draw_caption(img, caption, size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_caption_overlay(caption: str, *, size: tuple[int, int] | None = None) -> bytes:
    """A fully transparent RGBA PNG with ONLY the subtitle strip drawn, for the
    Ken Burns pipeline: ffmpeg overlays it on top of the zooming frame so the
    caption stays fixed (a burned-in caption would scale/drift with the zoom).
    Same renderer as burn_caption, so captions look identical either way."""
    size = size or frame_size()
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
