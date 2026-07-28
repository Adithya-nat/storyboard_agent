"""Video adapter: the error branches (always) + a real ffmpeg roundtrip (if present)."""

from __future__ import annotations

import pytest

from app.adapters.video import VideoAssemblyError, assemble_video, ffmpeg_available
from app.core.images import render_card


async def test_assemble_rejects_empty_frames(tmp_path):
    with pytest.raises(VideoAssemblyError, match="no frames"):
        await assemble_video([], str(tmp_path / "out.mp4"))


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
async def test_assemble_real_video(tmp_path):
    frames = []
    for i in range(1, 4):
        p = tmp_path / f"f{i}.png"
        p.write_bytes(render_card(f"frame {i}", "caption"))
        frames.append((str(p), 2.0))

    out = tmp_path / "out.mp4"
    await assemble_video(frames, str(out))
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
async def test_assemble_with_relative_paths(tmp_path, monkeypatch):
    """Regression: RELATIVE frame paths (default ARTIFACT_ROOT=./artifacts_store)
    must not be doubled by the concat demuxer. Mirrors the real layout where the
    concat file sits in the same dir as the frames."""
    monkeypatch.chdir(tmp_path)
    job_dir = tmp_path / "artifacts_store" / "job123"
    job_dir.mkdir(parents=True)

    frames = []
    for i in range(1, 4):
        rel = f"artifacts_store/job123/frame_{i:02d}.png"
        (tmp_path / rel).write_bytes(render_card(f"frame {i}", "cap"))
        frames.append((rel, 2.0))  # relative path, as the store produces

    out = "artifacts_store/job123/out.mp4"
    await assemble_video(frames, out)
    assert (tmp_path / out).exists() and (tmp_path / out).stat().st_size > 0
