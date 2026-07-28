"""assembler agent: ordered CLEAN frames -> final mp4 with SYNCED narration.

Two visual paths (unchanged):
  enable_motion ON  -> per-frame Ken Burns zoom + caption overlay + crossfades.
  enable_motion OFF -> burn captions + concat stills (hermetic tests, no motion).

Narration (Workstream 5, ON by default): a TTS voice reads EACH caption and that image is
held on screen for exactly the narration's length (audio-driven holds), so audio and
visuals are synced per beat. Each beat's start offset comes from the SAME function the
video uses (xfade_offsets / concat_offsets), so the voice can't drift from the picture.

    caption_NN ─▶ TTS ─▶ narration_NN.wav ─▶ hold_NN = max(floor, len+pad)
                                              │
        holds ─▶ kenburns/concat (video) ─────┼─▶ offsets ─▶ adelay+amix (one track) ─▶ mux

Idempotent: the per-beat WAVs are load-or-produce (narration_NN.wav), so a resume never
re-synthesizes against the ~100/day TTS cap.

Degrade, don't die: a beat whose TTS fails gets silence (image still shows); if the synced
track can't be built we fall back to one continuous track from the audio we already have;
if even that fails the video is left silent. Narration NEVER fails a job.

Silent path (enable_tts off / no tts): the original EXACT-duration uniform holds are used,
so existing duration behaviour + tests are unchanged.
"""

from __future__ import annotations

import asyncio
import os

from app.adapters.errors import ProviderError
from app.adapters.tts import concat_wavs, silent_wav, wav_duration_seconds
from app.adapters.video import (
    assemble_kenburns,
    assemble_video,
    build_narration_track,
    concat_offsets,
    mux_audio,
    xfade_offsets,
)
from app.core.images import burn_caption, render_caption_overlay
from app.core.logging import get_logger
from app.graph.state import Deps, StoryboardState

log = get_logger(__name__)


async def assembler(state: StoryboardState, deps: Deps) -> StoryboardState:
    frames = sorted(state["frames"], key=lambda f: f["order"])
    if not frames:
        raise ValueError("assembler: no frames to assemble")

    store = deps.store
    output_path = str(store.job_dir / "storyboard.mp4")
    n = len(frames)
    narrating = deps.settings.enable_tts and deps.tts is not None
    warnings: list[str] = []
    log.info("assembling %d frame(s) into the film", n)

    # 1) Narration first (when on): per-beat audio drives each image's on-screen hold.
    beat_audio: list[tuple[str, float]] = []  # (wav_path, hold) per frame, in order
    if narrating:
        log.info("voicing %d narration beat(s) under the montage", n)
        beat_audio, warns = await _synthesize_beats(frames, deps)
        warnings += warns
    holds: list[float] | None = [hold for _, hold in beat_audio] if narrating else None

    # 2) Build the (silent) video with those holds; capture each clip's start offset.
    if deps.settings.enable_motion:
        d = deps.settings.crossfade_seconds
        if holds is None:
            # Silent path: solve a uniform hold so total == target duration exactly.
            uniform = max(2 * d, (state["duration"] + (n - 1) * d) / n)
            holds = [uniform] * n
        else:
            holds = [max(2 * d, h) for h in holds]  # xfade math needs hold >= 2*d
        clips: list[tuple[str, str, float]] = []
        for f, hold in zip(frames, holds):
            overlay = store.save_bytes(
                f"capov_{f['order']:02d}.png", render_caption_overlay(f["caption"])
            )
            clips.append((f["path"], overlay, hold))
        # Splice per-scene motion clips (motion engine) in place of the still zoom,
        # aligned to frame order. Missing/absent clips stay None -> that scene keeps
        # its Ken Burns zoom (assemble_kenburns also falls back per-scene on error).
        motion_paths = _motion_paths(frames, state, deps)
        await assemble_kenburns(clips, output_path, crossfade=d, motion_paths=motion_paths)
        offsets = xfade_offsets(holds, d)
    else:
        if holds is None:
            per_frame = max(deps.settings.seconds_per_frame, state["duration"] / n)
            holds = [per_frame] * n
        timeline: list[tuple[str, float]] = []
        for f, hold in zip(frames, holds):
            captioned = burn_caption(store.load_bytes(f["path"]), f["caption"])
            cap_path = store.save_bytes(f"cap_{f['order']:02d}.png", captioned)
            timeline.append((cap_path, hold))
        await assemble_video(timeline, output_path)
        offsets = concat_offsets(holds)

    # 3) Lay the synced narration under the finished video (degrade, don't die).
    narration_path: str | None = None
    if narrating and beat_audio:
        narration_path = await _add_synced_voiceover(
            beat_audio, offsets, deps, output_path, warnings
        )

    result: StoryboardState = {"video_path": output_path, "warnings": warnings}
    if narration_path:
        result["narration_path"] = narration_path
    return result


def _motion_paths(frames, state: StoryboardState, deps: Deps) -> list[str | None] | None:
    """Per-scene engine motion-clip paths aligned to ``frames`` order, or None when
    there are none / the feature is off. Only existing files are used, so a swept
    or half-written clip silently degrades that scene to its still zoom."""
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


async def _synthesize_beats(
    frames, deps: Deps
) -> tuple[list[tuple[str, float]], list[str]]:
    """Synthesize (idempotently) one WAV per caption; return [(wav_path, hold)] + warnings.

    ``hold = max(floor, audio_len + pad)`` so each image stays up for its narration.
    SEQUENTIAL synthesis respects the ~100/day TTS cap; a per-beat failure becomes silence
    (the image still shows) instead of killing the job.
    """
    store = deps.store
    floor = deps.settings.narration_floor_seconds
    pad = deps.settings.narration_pad_seconds
    out: list[tuple[str, float]] = []
    warnings: list[str] = []
    silent = 0

    for f in frames:
        name = f"narration_{f['order']:02d}.wav"
        if store.exists(name):  # idempotent: resume reuses synthesized audio
            wav = store.load_bytes(store.abspath(name))
        else:
            try:
                wav = await asyncio.to_thread(deps.tts.synthesize, f["caption"])
            except ProviderError as exc:
                log.warning("narration for beat %s failed (%s); using silence", f["order"], exc)
                warnings.append(
                    f"narration: beat {f['order']} voiceover failed ({exc}); silent"
                )
                wav = silent_wav(floor)
                silent += 1
            store.save_bytes(name, wav)
        hold = max(floor, wav_duration_seconds(wav) + pad)
        out.append((store.abspath(name), hold))
    log.info("narration ready: %d voiced, %d silent", len(out) - silent, silent)
    return out, warnings


async def _add_synced_voiceover(
    beat_audio: list[tuple[str, float]],
    offsets: list[float],
    deps: Deps,
    video_path: str,
    warnings: list[str],
) -> str | None:
    """Build ONE aligned track (beat k at offset k) + mux it under the video. Degrade
    ladder: synced track -> single concatenated track -> silent video. Returns the
    narration track path when audio was laid down, or None when the video is left silent.
    """
    store = deps.store
    segments = [(wav_path, off) for (wav_path, _), off in zip(beat_audio, offsets)]
    track = str(store.job_dir / "narration_track.wav")
    muxed = str(store.job_dir / "storyboard_muxed.mp4")

    try:
        await build_narration_track(segments, track)
        await mux_audio(video_path, track, muxed)
        os.replace(muxed, video_path)
        return track
    except Exception as exc:  # noqa: BLE001 - degrade, don't die
        log.warning("synced narration failed (%s); falling back to a single track", exc)
        warnings.append(f"narration: synced track failed ({exc}); used one continuous track")

    # Fallback: one continuous track from the audio we already synthesized (no extra TTS).
    try:
        wavs = [store.load_bytes(p) for p, _ in beat_audio]
        store.save_bytes("narration_track.wav", concat_wavs(wavs))
        await mux_audio(video_path, track, muxed)
        os.replace(muxed, video_path)
        return track
    except Exception as exc:  # noqa: BLE001 - degrade, don't die
        log.warning("fallback narration also failed (%s); leaving video silent", exc)
        warnings.append("narration: voiceover unavailable; the video is silent")
        return None
