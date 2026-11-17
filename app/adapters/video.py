"""Video assembly adapter.

Two paths:

  assemble_video    (fallback / hermetic tests) — concat demuxer, stills held N s,
                    captions already burned in. No motion.

  assemble_kenburns (Phase 3 default) — per-frame slow zoom + a fixed caption
                    overlay, joined with crossfades so held stills read as filmed:

    clean frame_NN  ─loop─▶ scale 2x cover ─▶ zoompan (slow zoom-in) ─▶ 720x1280
                                                      │
                       caption_NN.png (transparent) ─▶ overlay (fixed) ─▶ clip_NN.mp4
                                                      │
    clip_00 clip_01 ... clip_NN ─▶ xfade chain (crossfade d) ─▶ storyboard.mp4

Per-clip hold is solved UP FRONT so the crossfades don't shrink the runtime:
    hold = (target + (n-1)*d) / n   →   total == target exactly.

ffmpeg runs as ASYNC subprocesses so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from app.core.logging import get_logger

log = get_logger(__name__)

# Max playback speed when retiming an engine motion clip into a (short) scene hold.
# Above this, motion reads as fast-forward; below the cap we window the arc's
# dynamic middle instead of speeding up further. 2.5x still reads as real video.
_MOTION_MAX_SPEED = 2.5
_TRIM_TOLERANCE = 1.5  # a clip up to this much longer than its hold is trimmed, not retimed


def _out_size() -> tuple[int, int]:
    from app.core.config import get_settings

    s = get_settings()
    return s.video_width, s.video_height


class VideoAssemblyError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _motion_summary(n: int, motion: int, fallback: int) -> str:
    """Per-film render summary — the signal that says whether the motion engine's
    clips actually landed in the film vs. silently degraded to Ken Burns. Replaces
    the old unconditional 'kenburns, N clips' log that hid a total fallback."""
    still = n - motion - fallback
    return (
        f"{n} scene{'s' if n != 1 else ''}: {motion} motion, "
        f"{fallback} motion->kenburns fallback, {still} still-zoom"
    )


async def _run_ffmpeg(cmd: list[str]) -> None:
    """Run an ffmpeg command async; raise VideoAssemblyError with tail logs on failure."""
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise VideoAssemblyError(
            f"ffmpeg failed (exit {proc.returncode}): "
            f"{stderr.decode(errors='replace')[-1000:]}"
        )


async def _probe_duration(path: str) -> float | None:
    """Container duration in seconds via ffprobe, or None if it can't be read.

    Backend-local (the engine has its own ``ffprobe_duration``; this repo doesn't).
    Used to RETIME a motion clip into a scene's hold so the whole move is visible
    instead of only its slow ease-in head.
    """
    if shutil.which("ffprobe") is None:
        return None
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate()
    if proc.returncode != 0:
        return None
    try:
        d = float(out.decode().strip())
    except ValueError:
        return None
    return d if d > 0 else None


# --------------------------------------------------------------------------- #
# Fallback: plain concat of pre-captioned stills (no motion)
# --------------------------------------------------------------------------- #
async def assemble_video(frames: list[tuple[str, float]], output_path: str) -> str:
    """frames: list of (image_path, duration_seconds). Returns output_path."""
    W, H = _out_size()
    if not frames:
        raise VideoAssemblyError("Cannot assemble video: no frames provided.")
    if not ffmpeg_available():
        raise VideoAssemblyError("ffmpeg not found on PATH. Install ffmpeg.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    concat_file = out.parent / "concat.txt"

    # Every frame gets an explicit `duration` so the last frame is honored and
    # total runtime == sum(durations) exactly. ABSOLUTE paths: the concat demuxer
    # resolves a relative `file` entry relative to the concat file's OWN dir, which
    # would double a path like 'artifacts_store/<job>/frame.png'.
    lines: list[str] = []
    for path, duration in frames:
        abs_path = str(Path(path).resolve())
        safe = abs_path.replace("'", "'\\''")
        lines.append(f"file '{safe}'")
        lines.append(f"duration {duration:.3f}")
    concat_file.write_text("\n".join(lines), encoding="utf-8")

    await _run_ffmpeg([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
               f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        str(out),
    ])
    log.info("Assembled video (concat) -> %s", out)
    return str(out)


# --------------------------------------------------------------------------- #
# Ken Burns: per-frame zoom + caption overlay + crossfades
# --------------------------------------------------------------------------- #
async def assemble_kenburns(
    clips: list[tuple[str, str, float]],
    output_path: str,
    *,
    crossfade: float | list[float] = 0.4,
    fps: int = 30,
    motion_paths: list[str | None] | None = None,
) -> str:
    """clips: list of (clean_frame_path, caption_overlay_png, hold_seconds).

    Renders one clip per frame then crossfades them together. When
    ``motion_paths[i]`` is set, that scene's clip is built from the external
    engine's motion video (retimed to the scene's hold + caption overlaid) instead
    of a Ken Burns zoom on the still; a per-scene render failure falls back to the
    zoom, so the film never regresses.
    """
    if not clips:
        raise VideoAssemblyError("Cannot assemble video: no frames provided.")
    if not ffmpeg_available():
        raise VideoAssemblyError("ffmpeg not found on PATH. Install ffmpeg.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    workdir = out.parent

    clip_paths: list[Path] = []
    holds: list[float] = []
    motion_count = 0    # scenes rendered from an engine motion clip
    fallback_count = 0  # scenes that HAD a motion clip but it failed -> Ken Burns
    for i, (frame, overlay, hold) in enumerate(clips):
        clip_path = workdir / f"clip_{i:02d}.mp4"
        motion = motion_paths[i] if motion_paths and i < len(motion_paths) else None
        if motion:
            try:
                await _render_motion_clip(motion, overlay, hold, str(clip_path), fps)
                motion_count += 1
            except VideoAssemblyError as exc:
                log.warning(
                    "motion clip %d failed (%s); falling back to Ken Burns zoom", i, exc
                )
                await _render_kenburns_clip(frame, overlay, hold, str(clip_path), fps)
                fallback_count += 1
        else:
            await _render_kenburns_clip(frame, overlay, hold, str(clip_path), fps)
        clip_paths.append(clip_path)
        holds.append(hold)

    if len(clip_paths) == 1:
        # Nothing to crossfade — the single clip IS the video.
        shutil.copyfile(clip_paths[0], out)
        log.info(
            "Assembled video (%s) -> %s", _motion_summary(1, motion_count, fallback_count), out
        )
        return str(out)

    inputs: list[str] = []
    for p in clip_paths:
        inputs += ["-i", str(p)]
    filter_complex, final = _xfade_filter(len(clip_paths), holds, crossfade)
    await _run_ffmpeg([
        "ffmpeg", "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{final}]",
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        str(out),
    ])
    log.info(
        "Assembled video (%s) -> %s",
        _motion_summary(len(clip_paths), motion_count, fallback_count),
        out,
    )
    return str(out)


async def _render_kenburns_clip(
    frame: str, overlay: str, hold: float, out: str, fps: int
) -> None:
    """One frame -> a slow zoom-in clip with the (fixed) caption overlaid on top."""
    total_frames = max(1, round(hold * fps))
    # Reach ~1.12x zoom by the last frame; sub-pixel increments on a 2x-upscaled
    # source keep the zoom smooth (the classic zoompan-jitter fix).
    zinc = 0.12 / max(1, total_frames - 1)
    W, H = _out_size()
    filter_complex = (
        f"[0:v]scale={W*2}:{H*2}:force_original_aspect_ratio=increase,crop={W*2}:{H*2},"
        f"zoompan=z='min(zoom+{zinc:.6f},1.12)':d={total_frames}:"
        "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"s={W}x{H}:fps={fps},setsar=1[zo];"
        "[zo][1:v]overlay=0:0:format=auto,format=yuv420p[v]"
    )
    await _run_ffmpeg([
        "ffmpeg", "-y",
        "-loop", "1", "-t", f"{hold:.3f}", "-i", frame,
        "-loop", "1", "-t", f"{hold:.3f}", "-i", overlay,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-t", f"{hold:.3f}",
        "-r", str(fps),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out,
    ])


async def _render_motion_clip(
    motion: str, overlay: str, hold: float, out: str, fps: int
) -> None:
    """One engine motion clip -> RETIMED to exactly ``hold`` with the caption
    overlaid, matching the Ken Burns clip contract (720x1280 @ fps) so it slots
    into the same xfade chain.

    Why retime, not trim: engine clips are ~8s and the move eases in, but scene
    holds are narration-driven and short (~1.5s). The old ``trim=duration=hold``
    kept only the SLOW first ~1.5s of the arc, so the film looked static. Instead:

        clip src (~8s) ──▶ take a window of length min(src, hold*MAX_SPEED)
                           centered on the DYNAMIC MIDDLE (skips the ease-in head)
                        ──▶ setpts-retime that window to exactly ``hold`` seconds
                        ──▶ tpad/trim/fps pin the output to hold @ fps (exact)

    Speed is capped at MAX_SPEED so motion still reads as video; for short holds
    windowing the middle is the NORMAL path (an 8s arc can't fully play in 1.5s
    without looking fast-forwarded). A probe failure raises so the caller falls
    back to a Ken Burns zoom (which still moves) rather than a frozen head.
    """
    src = await _probe_duration(motion)
    if src is None:
        raise VideoAssemblyError(f"could not probe motion clip duration: {motion}")
    if hold <= src <= hold + _TRIM_TOLERANCE:
        # Generated at the scene's length (Omni): play at 1x so lip-sync holds; trim the tail.
        window, start, factor = hold, 0.0, 1.0
    else:
        # Show as much of the arc as the speed cap allows, centered on the dynamic middle.
        window = min(src, hold * _MOTION_MAX_SPEED)
        start = max(0.0, (src - window) / 2.0)
        factor = hold / window
    W, H = _out_size()
    filter_complex = (
        f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
        f"fps={fps},trim=start={start:.3f}:duration={window:.3f},"
        f"setpts={factor:.5f}*(PTS-STARTPTS),"
        f"tpad=stop_mode=clone:stop_duration={hold:.3f},trim=duration={hold:.3f},"
        f"fps={fps},setpts=PTS-STARTPTS[m];"
        "[m][1:v]overlay=0:0:format=auto,format=yuv420p[v]"
    )
    await _run_ffmpeg([
        "ffmpeg", "-y",
        "-i", motion,
        "-loop", "1", "-t", f"{hold:.3f}", "-i", overlay,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-t", f"{hold:.3f}",
        "-r", str(fps),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out,
    ])


async def mux_audio(video_path: str, audio_path: str, output_path: str) -> str:
    """Lay a narration track under the (silent) video. The VIDEO is the master
    length: shorter audio is padded with silence, longer audio is trimmed. Video
    is stream-copied (no re-encode); audio is encoded to AAC."""
    if not ffmpeg_available():
        raise VideoAssemblyError("ffmpeg not found on PATH. Install ffmpeg.")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    await _run_ffmpeg([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-filter_complex", "[1:a]apad[a]",  # pad audio with silence...
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac",
        "-shortest",                         # ...then trim to the video length
        output_path,
    ])
    log.info("Muxed narration audio -> %s", output_path)
    return output_path


async def build_narration_track(segments: list[tuple[str, float]], output_path: str) -> str:
    """Lay each beat's WAV at its start offset on ONE audio track (synced narration).

    ``segments`` = list of (wav_path, start_offset_seconds), where the offsets come from
    ``xfade_offsets`` / ``concat_offsets`` so each beat's voice starts exactly when its
    image appears. Implemented as a per-input ``adelay`` + a single ``amix``.
    """
    if not segments:
        raise VideoAssemblyError("no narration segments to build a track from")
    if not ffmpeg_available():
        raise VideoAssemblyError("ffmpeg not found on PATH. Install ffmpeg.")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    inputs: list[str] = []
    filters: list[str] = []
    labels: list[str] = []
    for i, (path, offset) in enumerate(segments):
        inputs += ["-i", str(path)]
        ms = max(0, int(round(offset * 1000)))
        filters.append(f"[{i}:a]adelay={ms}:all=1[a{i}]")
        labels.append(f"[a{i}]")
    # normalize=0 keeps each delayed voice at full volume (they don't overlap by design).
    filters.append(f"{''.join(labels)}amix=inputs={len(segments)}:normalize=0[out]")

    await _run_ffmpeg([
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(filters),
        "-map", "[out]",
        str(output_path),
    ])
    log.info("Built synced narration track (%d beats) -> %s", len(segments), output_path)
    return str(output_path)


def _overlaps(n: int, d: float | list[float]) -> list[float]:
    """Per-boundary overlap (index j = into clip j); 0 = hard cut."""
    return [0.0] + (list(d) if isinstance(d, list) else [d] * max(0, n - 1))


def xfade_offsets(holds: list[float], d: float | list[float]) -> list[float]:
    """Start offset of each clip on the timeline. ``d`` is one crossfade for every cut, or a
    per-cut list (0 = hard cut). The single source of truth for video AND audio placement."""
    if not holds:
        return []
    ov = _overlaps(len(holds), d)
    offsets = [0.0]
    running = holds[0]
    for j in range(1, len(holds)):
        offsets.append(max(0.0, running - ov[j]))
        running = running + holds[j] - ov[j]
    return offsets


def concat_offsets(holds: list[float]) -> list[float]:
    """Start offset of each still in the plain-concat (no-motion) timeline = the
    cumulative sum of the preceding holds."""
    offsets: list[float] = []
    acc = 0.0
    for h in holds:
        offsets.append(acc)
        acc += h
    return offsets


def _xfade_filter(n: int, holds: list[float], d: float | list[float]) -> tuple[str, str]:
    """Chained filter_complex: hard cuts concat, dissolves xfade, at ``xfade_offsets``.

    Every input is normalized to one timebase first: concat outputs AVTB (1/1000000)
    while the clips are 1/15360, and xfade refuses inputs whose timebases differ, so a
    dissolve after a hard cut would fail. settb only relabels timestamps (no re-encode).
    """
    offsets, ov = xfade_offsets(holds, d), _overlaps(n, d)
    parts = [f"[{j}:v]settb=AVTB[s{j}]" for j in range(n)]
    prev = "s0"
    for j in range(1, n):
        out_label = f"x{j}"
        if ov[j] > 0:
            parts.append(
                f"[{prev}][s{j}]xfade=transition=fade:duration={ov[j]:.3f}:offset={offsets[j]:.3f}[{out_label}]"
            )
        else:
            parts.append(f"[{prev}][s{j}]concat=n=2:v=1:a=0[{out_label}]")
        prev = out_label
    return ";".join(parts), prev


async def probe_duration(path: str) -> float | None:
    return await _probe_duration(path)


async def extract_audio(video_path: str, wav_path: str) -> str:
    """Mono 24 kHz WAV of a clip's audio track."""
    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
    await _run_ffmpeg(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "24000", wav_path])
    return wav_path


def _db(gain_db: float) -> str:
    return f"{10 ** (gain_db / 20):.4f}"


async def build_mix(
    *,
    speech: list[tuple],
    ambience: list[tuple],
    music: str | None,
    total_seconds: float,
    output_path: str,
    ambience_db: float = -18.0,
    music_db: float = -14.0,
    duck_ratio: float = 8.0,
) -> str:
    """One stereo-ready track: speech at full level, ambience under it, and a looped
    score sidechain-ducked whenever anyone speaks. Offsets come from xfade/concat_offsets."""
    if not ffmpeg_available():
        raise VideoAssemblyError("ffmpeg not found on PATH. Install ffmpeg.")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    total = max(0.5, total_seconds)
    inputs: list[str] = []
    filters: list[str] = []

    def add(path: str) -> int:
        inputs.extend(["-i", str(path)])
        return len(inputs) // 2 - 1

    def bus(items: list[tuple], label: str, gain: str = "1") -> str | None:
        if not items:
            return None
        parts = []
        for path, offset, *rest in items:
            i = add(path)
            ms = max(0, int(round(offset * 1000)))
            dur = rest[0] if rest and rest[0] else None
            # Clip audio is cut to its scene with 60ms fades, so cuts never click or bleed.
            edge = (f"atrim=0:{dur:.3f},afade=t=in:d=0.06,afade=t=out:st={max(0.0, dur - 0.06):.3f}:d=0.06,"
                    if dur else "")
            filters.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=mono,{edge}volume={gain},adelay={ms}:all=1[{label}{i}]")
            parts.append(f"[{label}{i}]")
        filters.append(f"{''.join(parts)}amix=inputs={len(parts)}:normalize=0[{label}]")
        return label

    sp = bus(speech, "sp")
    amb = bus(ambience, "am", _db(ambience_db))
    mus = None
    if music:
        i = add(music)
        filters.append(
            f"[{i}:a]aresample=48000,aformat=channel_layouts=mono,aloop=loop=-1:size=2e9,"
            f"atrim=0:{total:.3f},volume={_db(music_db)},"
            f"afade=t=in:d=1.0,afade=t=out:st={max(0.0, total - 2.0):.3f}:d=2.0[mraw]"
        )
        if sp:
            # sidechaincompress ends with its shortest input; pad speech to the full runtime.
            filters.append(f"[sp]apad=whole_dur={total:.3f},asplit=2[spm][spk]")
            sp = "spm"
            filters.append(f"[mraw][spk]sidechaincompress=threshold=0.02:ratio={duck_ratio}:attack=20:release=400[mus]")
        else:
            filters.append("[mraw]anull[mus]")
        mus = "mus"
    buses = [f"[{b}]" for b in (sp, amb, mus) if b]
    if not buses:
        raise VideoAssemblyError("nothing to mix")
    filters.append(f"{''.join(buses)}amix=inputs={len(buses)}:normalize=0,alimiter=limit=0.95,atrim=0:{total:.3f}[out]")
    await _run_ffmpeg(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[out]", str(output_path)])
    log.info("Mixed audio (%d speech, %d ambience, music=%s) -> %s", len(speech), len(ambience), bool(music), output_path)
    return str(output_path)
