"""Caption rendering + frame geometry (Phase 3): subtitles must be centered and
properly sized (not the old tiny corner text), and frames must cover-fill the
vertical canvas with no pillarbox bars.
"""

from __future__ import annotations

import io

from PIL import Image

from app.core.images import FRAME_SIZE, burn_caption


def _png(img: "Image.Image") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _white_columns(img: "Image.Image", y0: int, y1: int, step: int = 4) -> list[int]:
    """Columns (x) that contain near-white text pixels in the y-band [y0, y1)."""
    px = img.load()
    width, _ = img.size
    cols: list[int] = []
    for x in range(0, width, step):
        for y in range(y0, y1, step):
            r, g, b = px[x, y][:3]
            if r > 200 and g > 200 and b > 200:
                cols.append(x)
                break
    return cols


def test_frame_size_is_vertical_9_16():
    w, h = FRAME_SIZE
    assert (w, h) == (720, 1280)
    assert h > w  # portrait


def test_caption_centered_and_not_cornered():
    base = Image.new("RGB", FRAME_SIZE, (128, 128, 128))
    out = burn_caption(_png(base), "Two ghost runners trapped in an endless loop")
    img = Image.open(io.BytesIO(out)).convert("RGB")
    w, h = img.size

    cols = _white_columns(img, int(h * 0.70), h)
    assert cols, "no caption text was rendered in the lower third"
    lo, hi = min(cols), max(cols)
    center = (lo + hi) / 2
    # The old bug: tiny text pinned at x=40 (left corner). Now: centered subtitle.
    assert lo > w * 0.06, "caption still hugging the left edge"
    assert w * 0.35 < center < w * 0.65, "caption not horizontally centered"
    # And it spans a real width (proper font size, not ~11px bitmap).
    assert (hi - lo) > w * 0.30, "caption text suspiciously narrow (font too small?)"


def test_cover_crop_fills_vertical_no_bars():
    # A wide (landscape) source must cover-fill the portrait frame — no black bars.
    wide = Image.new("RGB", (1000, 400), (200, 20, 20))
    out = burn_caption(_png(wide), "")  # no caption -> pure geometry check
    img = Image.open(io.BytesIO(out)).convert("RGB")
    assert img.size == FRAME_SIZE

    px = img.load()
    w, h = img.size
    for x, y in [(2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3)]:
        r, g, b = px[x, y]
        assert r > 150 and g < 80 and b < 80, f"pillarbox bar at {(x, y)} = {(r, g, b)}"
