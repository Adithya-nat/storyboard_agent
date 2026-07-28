"""Splicing engine motion clips into the assembled film (ffmpeg-gated).

Asserts a mixed timeline (some scenes have a motion clip, some fall back to the
Ken Burns still-zoom) still hits the exact solved duration + 720x1280 contract,
and that a motion clip SHORTER than its hold is freeze-padded (not truncated)."""

from __future__ import annotations

import json
import subprocess

import pytest
from PIL import Image, ImageStat

from app.adapters.video import (
    _motion_summary,
    assemble_kenburns,
    ffmpeg_available,
    xfade_offsets,
)
from app.core.images import render_caption_overlay

_needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")


def _still(path, color=(90, 90, 160)):
    Image.new("RGB", (360, 640), color).save(path)


def _overlay(path):
    path.write_bytes(render_caption_overlay("a caption"))


def _motion_clip(path, seconds, w=320, h=568):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"testsrc=size={w}x{h}:rate=30:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )


def _black_then_white(path, w=320, h=568):
    """8s clip: 5s black then 3s white. The MOTION (the black->white change) lives
    past the first 1.5s, so a trim-to-head splice sees only black (frozen) while a
    correct retime surfaces the later bright content."""
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r=30:d=5",
         "-f", "lavfi", "-i", f"color=c=white:s={w}x{h}:r=30:d=3",
         "-filter_complex", "[0][1]concat=n=2:v=1[v]",
         "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )


def _last_frame_mean_luma(clip_path, tmp_path):
    frame = tmp_path / "last.png"
    subprocess.run(
        ["ffmpeg", "-y", "-sseof", "-0.1", "-i", str(clip_path), "-frames:v", "1", str(frame)],
        check=True, capture_output=True,
    )
    return ImageStat.Stat(Image.open(frame).convert("L")).mean[0]


def _probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:format=duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    d = json.loads(out)
    st = d["streams"][0]
    return int(st["width"]), int(st["height"]), float(d["format"]["duration"])


@_needs_ffmpeg
async def test_mixed_motion_and_still_hits_exact_duration(tmp_path):
    f0, f1 = tmp_path / "f0.png", tmp_path / "f1.png"
    ov = tmp_path / "ov.png"
    m0 = tmp_path / "motion_00.mp4"
    _still(f0); _still(f1, (160, 90, 90)); _overlay(ov)
    _motion_clip(m0, 2.0)

    d, hold = 0.4, 2.0
    clips = [(str(f0), str(ov), hold), (str(f1), str(ov), hold)]
    out = tmp_path / "film.mp4"

    # scene 0 uses the motion clip; scene 1 falls back to the still zoom (None).
    await assemble_kenburns(clips, str(out), crossfade=d, motion_paths=[str(m0), None])

    total = sum([hold, hold]) - d  # xfade overlap
    w, h, dur = _probe(out)
    assert (w, h) == (720, 1280)
    assert abs(dur - total) < 0.2
    # offsets are the SAME source of truth the narration uses -> sync preserved
    assert xfade_offsets([hold, hold], d)[1] == pytest.approx(hold - d)


@_needs_ffmpeg
async def test_motion_clip_shorter_than_hold_is_freeze_padded(tmp_path):
    f0 = tmp_path / "f0.png"
    ov = tmp_path / "ov.png"
    m0 = tmp_path / "motion_00.mp4"
    _still(f0); _overlay(ov)
    _motion_clip(m0, 1.0)  # 1s clip, held for 2s -> last frame frozen for the gap

    hold = 2.0
    out = tmp_path / "film.mp4"
    await assemble_kenburns([(str(f0), str(ov), hold)], str(out), crossfade=0.4, motion_paths=[str(m0)])

    w, h, dur = _probe(out)
    assert (w, h) == (720, 1280)
    assert abs(dur - hold) < 0.2  # padded up to the full hold, not truncated to 1s


@_needs_ffmpeg
async def test_long_motion_clip_retimed_to_exact_hold(tmp_path):
    # An 8s engine clip spliced into a short 1.5s hold must come out EXACTLY 1.5s
    # (the xfade + narration-sync math depends on it), 720x1280.
    f0 = tmp_path / "f0.png"; ov = tmp_path / "ov.png"; m0 = tmp_path / "motion_00.mp4"
    _still(f0); _overlay(ov); _motion_clip(m0, 8.0)
    out = tmp_path / "film.mp4"; hold = 1.5
    await assemble_kenburns(
        [(str(f0), str(ov), hold)], str(out), crossfade=0.4, motion_paths=[str(m0)]
    )
    w, h, dur = _probe(out)
    assert (w, h) == (720, 1280)
    assert abs(dur - hold) < 0.2


@_needs_ffmpeg
async def test_retimed_clip_surfaces_arc_not_frozen_head(tmp_path):
    # REGRESSION: the old trim=duration=hold kept only the SLOW first ~1.5s of an 8s
    # move -> a static film scene. With a black(5s)->white(3s) source, trim-to-head
    # sees only black; a correct retime windows into the later content, so the last
    # frame is bright. (Non-cyclic source, so last-frame check is valid here; live
    # verification samples a mid window because orbit_sway is cyclic.)
    f0 = tmp_path / "f0.png"; ov = tmp_path / "ov.png"; m0 = tmp_path / "motion_00.mp4"
    _still(f0); _overlay(ov); _black_then_white(m0)
    out = tmp_path / "film.mp4"; hold = 1.5
    await assemble_kenburns(
        [(str(f0), str(ov), hold)], str(out), crossfade=0.4, motion_paths=[str(m0)]
    )
    mean = _last_frame_mean_luma(out, tmp_path)
    assert mean > 100, f"retimed clip shows only the black head (mean luma {mean:.0f})"


def test_motion_summary_formatting():
    # The honest label that replaced the old unconditional "kenburns, N clips".
    assert (
        _motion_summary(15, 15, 0)
        == "15 scenes: 15 motion, 0 motion->kenburns fallback, 0 still-zoom"
    )
    assert (
        _motion_summary(3, 1, 1)
        == "3 scenes: 1 motion, 1 motion->kenburns fallback, 1 still-zoom"
    )
    assert _motion_summary(1, 0, 0).startswith("1 scene:")  # singular


@_needs_ffmpeg
async def test_assembled_log_reports_real_motion_counts(tmp_path, caplog):
    # The assembly log must state how many scenes actually used a motion clip, so a
    # silent all-fallback is loud. (This is the signal T7 verification reads.)
    import logging

    f0, f1 = tmp_path / "f0.png", tmp_path / "f1.png"
    ov = tmp_path / "ov.png"; m0 = tmp_path / "motion_00.mp4"
    _still(f0); _still(f1, (160, 90, 90)); _overlay(ov); _motion_clip(m0, 2.0)
    clips = [(str(f0), str(ov), 2.0), (str(f1), str(ov), 2.0)]
    out = tmp_path / "film.mp4"
    with caplog.at_level(logging.INFO, logger="app.adapters.video"):
        # scene 0 uses the motion clip; scene 1 has none (still-zoom).
        await assemble_kenburns(clips, str(out), crossfade=0.4, motion_paths=[str(m0), None])
    msg = "\n".join(r.getMessage() for r in caplog.records)
    assert "2 scenes: 1 motion" in msg
    assert "0 motion->kenburns fallback" in msg
    assert "1 still-zoom" in msg
    assert "kenburns, 2 clips" not in msg  # the old misleading label is gone


@_needs_ffmpeg
async def test_unreadable_motion_clip_falls_back_to_kenburns(tmp_path):
    # A clip whose duration can't be probed must degrade to a Ken Burns zoom (which
    # still moves), never a frozen head -> the film still renders at the exact hold.
    f0 = tmp_path / "f0.png"; ov = tmp_path / "ov.png"; bad = tmp_path / "motion_00.mp4"
    _still(f0); _overlay(ov); bad.write_bytes(b"not a video")
    out = tmp_path / "film.mp4"; hold = 1.5
    await assemble_kenburns(
        [(str(f0), str(ov), hold)], str(out), crossfade=0.4, motion_paths=[str(bad)]
    )
    w, h, dur = _probe(out)
    assert (w, h) == (720, 1280)
    assert abs(dur - hold) < 0.2
