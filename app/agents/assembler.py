"""assembler agent: frames/clips + per-scene audio -> final mp4.

Holds and captions come from scene_audio (narration + captions stages). Audio mix:
speech (narrator, off-screen, or on-camera clip audio) at full level, Omni ambience
under narrated scenes, and the score ducked under speech. Audio never fails a job:
mix -> narration-only track -> silent video.
"""

from __future__ import annotations

import os

from app.adapters.tts import concat_wavs
from app.adapters.video import (
    assemble_kenburns,
    assemble_video,
    build_mix,
    build_narration_track,
    concat_offsets,
    extract_audio,
    mux_audio,
    xfade_offsets,
)
from app.agents.narration import voice_lines
from app.core.images import burn_caption, render_caption_overlay
from app.core.logging import get_logger
from app.graph.state import Deps, StoryboardState
from app.schemas import Script

log = get_logger(__name__)


async def assembler(state: StoryboardState, deps: Deps) -> StoryboardState:
    frames = sorted(state["frames"], key=lambda f: f["order"])
    if not frames:
        raise ValueError("assembler: no frames to assemble")
    store, settings = deps.store, deps.settings
    output_path = str(store.job_dir / "storyboard.mp4")
    warnings: list[str] = []
    n = len(frames)
    log.info("assembling %d frame(s) into the film", n)

    audio = state.get("scene_audio")
    if audio is None and settings.enable_tts and deps.tts is not None:
        # Called without the narration stage (direct use): narrate the frame captions.
        audio, warns = await voice_lines([(f["order"], f["caption"]) for f in frames], deps)
        warnings += warns
    by_order = {a["order"]: a for a in audio or []}
    voiced = any(a.get("voice") in ("narrator", "off_screen", "on_camera") for a in by_order.values())
    captions = (
        [by_order.get(f["order"], {}).get("caption") or f["caption"] for f in frames]
        if settings.burn_subtitles
        else [""] * len(frames)
    )
    holds: list[float] | None = (
        [float(by_order.get(f["order"], {}).get("hold") or settings.seconds_per_frame) for f in frames]
        if voiced
        else None
    )

    motion_paths = _motion_paths(frames, state, deps)
    if settings.enable_motion:
        d = _transitions(frames, state, deps)
        floor = 2 * max(d) if d else 0.0
        if holds is None:  # silent path: total == target duration exactly
            holds = [max(floor, (state["duration"] + sum(d)) / n)] * n
        else:
            holds = [max(floor, h) for h in holds]
        clips = [
            (f["path"], store.save_bytes(f"capov_{f['order']:02d}.png", render_caption_overlay(cap)), hold)
            for f, cap, hold in zip(frames, captions, holds)
        ]
        await assemble_kenburns(clips, output_path, crossfade=d, motion_paths=motion_paths)
        offsets = xfade_offsets(holds, d)
    else:
        if holds is None:
            holds = [max(settings.seconds_per_frame, state["duration"] / n)] * n
        timeline = [
            (store.save_bytes(f"cap_{f['order']:02d}.png", burn_caption(store.load_bytes(f["path"]), cap)), hold)
            for f, cap, hold in zip(frames, captions, holds)
        ]
        await assemble_video(timeline, output_path)
        offsets = concat_offsets(holds)
        motion_paths = None  # the concat path has no clip audio either

    track = await _lay_audio(frames, holds, offsets, by_order, motion_paths, state, deps, output_path, warnings)
    scene_audio_path = store.save_json(
        "scene_audio.json",
        [{**by_order.get(f["order"], {"order": f["order"], "voice": "silent"}), "caption": cap, "hold": h}
         for f, cap, h in zip(frames, captions, holds)],
    )
    result: StoryboardState = {"video_path": output_path, "scene_audio_path": scene_audio_path, "warnings": warnings}
    if track:
        result["narration_path"] = track
    return result


def _transitions(frames, state: StoryboardState, deps: Deps) -> list[float]:
    """Per-cut overlap: hard cuts (0) like an editor, a dissolve only where the script marks a time jump."""
    d = deps.settings.crossfade_seconds
    if not deps.settings.editor_hard_cuts:
        return [d] * (len(frames) - 1)
    kinds: dict[int, str] = {}
    if state.get("script_path") and os.path.exists(state["script_path"]):
        script = Script.model_validate(deps.store.load_json(state["script_path"]))
        kinds = {sc.order: sc.transition_in for sc in script.scenes}
    return [d if kinds.get(f["order"]) == "dissolve" else 0.0 for f in frames[1:]]


def _motion_paths(frames, state: StoryboardState, deps: Deps) -> list[str | None] | None:
    """Per-scene engine clip paths aligned to frames (only files that exist)."""
    if not deps.settings.motion_in_film:
        return None
    by_order = {
        c["order"]: c["path"]
        for c in state.get("motion_clips", []) or []
        if c.get("path") and os.path.exists(c["path"])
    }
    if not by_order:
        return None
    return [by_order.get(f["order"]) for f in frames]


async def _lay_audio(frames, holds, offsets, by_order, motion_paths, state, deps: Deps, video_path, warnings) -> str | None:
    store, settings = deps.store, deps.settings
    clip_audio = {c["order"]: c for c in state.get("motion_clips") or [] if c.get("has_audio")}
    speech: list[tuple[str, float]] = []
    ambience: list[tuple[str, float]] = []
    for i, f in enumerate(frames):
        meta = by_order.get(f["order"], {})
        voice = meta.get("voice")
        if voice in ("narrator", "off_screen") and meta.get("path"):
            speech.append((meta["path"], offsets[i]))  # a voiceover may run across the cut
        clip = motion_paths[i] if motion_paths else None
        if clip and f["order"] in clip_audio:
            wav = store.abspath(f"clipaudio_{f['order']:02d}.wav")
            try:
                await extract_audio(clip, wav)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"audio: scene {f['order']} clip audio unreadable ({exc})")
                continue
            (speech if voice == "on_camera" else ambience).append((wav, offsets[i], holds[i]))

    music = state.get("music_path") if settings.music_enabled else None
    if not (speech or ambience or music):
        return None
    total = (offsets[-1] + holds[-1]) if offsets else float(state["duration"])
    track = str(store.job_dir / "narration_track.wav")
    muxed = str(store.job_dir / "storyboard_muxed.mp4")
    try:
        await build_mix(
            speech=speech, ambience=ambience, music=music, total_seconds=total, output_path=track,
            ambience_db=settings.ambience_gain_db, music_db=settings.music_gain_db,
            duck_ratio=settings.music_duck_ratio,
        )
        await mux_audio(video_path, track, muxed)
        os.replace(muxed, video_path)
        return track
    except Exception as exc:  # noqa: BLE001
        log.warning("audio mix failed (%s); falling back to speech only", exc)
        warnings.append(f"audio: mix failed ({exc}); used speech only")
    if not speech:
        warnings.append("audio: the video is silent")
        return None
    try:
        await build_narration_track(speech, track)
    except Exception:  # noqa: BLE001
        try:
            store.save_bytes("narration_track.wav", concat_wavs([store.load_bytes(p) for p, _ in speech]))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"audio: voiceover unavailable ({exc}); the video is silent")
            return None
    try:
        await mux_audio(video_path, track, muxed)
        os.replace(muxed, video_path)
        return track
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"audio: voiceover unavailable ({exc}); the video is silent")
        return None
