"""Phase 3 Ken Burns: caption overlay transparency, xfade offset math (no ffmpeg),
and the real zoompan+overlay+xfade pipeline (ffmpeg-gated) — dims, exact duration,
and that motion actually happens.
"""

from __future__ import annotations

import io
import json
import subprocess

import pytest
from PIL import Image, ImageDraw

from app.adapters.video import (
    _xfade_filter,
    assemble_kenburns,
    ffmpeg_available,
)
from app.core.images import FRAME_SIZE, render_caption_overlay

_needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")


# --------------------------------------------------------------------------- #
# Unit (no ffmpeg)
# --------------------------------------------------------------------------- #
def test_caption_overlay_is_transparent_except_band():
    out = render_caption_overlay("A clear subtitle line")
    img = Image.open(io.BytesIO(out))
    assert img.mode == "RGBA"
    assert img.size == FRAME_SIZE
    px = img.load()
    w, h = img.size
    # Top of the frame is fully transparent (image shows through under motion).
    assert px[w // 2, int(h * 0.2)][3] == 0
    # The lower-third band carries opaque scrim/text pixels.
    band_alpha = max(px[w // 2, y][3] for y in range(int(h * 0.80), h, 4))
    assert band_alpha > 0


def test_empty_caption_overlay_fully_transparent():
    img = Image.open(io.BytesIO(render_caption_overlay("")))
    assert img.getextrema()[3] == (0, 0)  # alpha channel all-zero


def test_xfade_offsets_make_total_equal_target():
    n, d = 4, 0.4
    holds = [2.0] * n
    fc, final = _xfade_filter(n, holds, d)
    assert final == "x3"
    # offset_j = j*(hold-d): 1.6, 3.2, 4.8
    for expected in ("offset=1.600", "offset=3.200", "offset=4.800"):
        assert expected in fc
    # total = n*hold - (n-1)*d = 8 - 1.2 = 6.8 ; last transition 4.8 + hold 2.0 = 6.8
    assert "[0:v][1:v]xfade" in fc and "[x1][2:v]xfade" in fc


# --------------------------------------------------------------------------- #
# Real ffmpeg (gated)
# --------------------------------------------------------------------------- #
def _probe(path: str) -> tuple[int, int, float]:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration",
        "-of", "json", path,
    ])
    data = json.loads(out)
    s = data["streams"][0]
    return int(s["width"]), int(s["height"]), float(data["format"]["duration"])


def _detail_frame(path) -> str:
    """A striped frame so a zoom produces a clearly measurable pixel change."""
    img = Image.new("RGB", FRAME_SIZE, (12, 12, 12))
    draw = ImageDraw.Draw(img)
    for y in range(0, FRAME_SIZE[1], 40):
        draw.rectangle([0, y, FRAME_SIZE[0], y + 20], fill=(240, 240, 240))
    img.save(path)
    return str(path)


def _clips(tmp_path, n, hold=2.0):
    clips = []
    for i in range(1, n + 1):
        frame = _detail_frame(tmp_path / f"frame_{i}.png")
        ov = tmp_path / f"ov_{i}.png"
        ov.write_bytes(render_caption_overlay(f"caption {i}"))
        clips.append((frame, str(ov), hold))
    return clips


def _signature(png_path: str) -> list[int]:
    im = Image.open(png_path).convert("L").resize((32, 32))
    return list(im.tobytes())


def _extract(video: str, t: float, out: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{t}", "-i", video, "-frames:v", "1", out],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


@_needs_ffmpeg
async def test_kenburns_dims_duration_and_motion(tmp_path):
    clips = _clips(tmp_path, 3, hold=2.0)
    out = tmp_path / "out.mp4"
    await assemble_kenburns(clips, str(out), crossfade=0.4)

    w, h, dur = _probe(str(out))
    assert (w, h) == (720, 1280)  # vertical, no bars
    # total = 3*2.0 - 2*0.4 = 5.2s
    assert abs(dur - 5.2) < 0.4

    # Motion: two instants within the first clip differ (the zoom is happening).
    a, b = str(tmp_path / "a.png"), str(tmp_path / "b.png")
    _extract(str(out), 0.2, a)
    _extract(str(out), 1.6, b)
    sa, sb = _signature(a), _signature(b)
    mad = sum(abs(x - y) for x, y in zip(sa, sb)) / len(sa)
    assert mad > 2.0, f"frames look static (mean abs diff {mad:.2f}) — zoom not applied"


@_needs_ffmpeg
async def test_kenburns_single_clip(tmp_path):
    clips = _clips(tmp_path, 1, hold=2.5)
    out = tmp_path / "single.mp4"
    await assemble_kenburns(clips, str(out), crossfade=0.4)
    w, h, dur = _probe(str(out))
    assert (w, h) == (720, 1280)
    assert abs(dur - 2.5) < 0.4
