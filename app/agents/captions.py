"""captions stage: finalize dialogue scenes after motion (voicing + holds; subtitles optional).

On-camera (Omni clip with audio): transcribe; captions = script lines when they match,
else the lines actually spoken. Hold = clip length.
Off-screen (no generated clip): voice the lines with multi-speaker TTS, max 2 speakers.
Never fails the job.
"""

from __future__ import annotations

import asyncio
import os
import re
import unicodedata

from app.adapters.errors import ProviderError
from app.adapters.tts import DialogueTurn, silent_wav, wav_duration_seconds
from app.adapters.video import extract_audio, probe_duration
from app.core.logging import get_logger
from app.graph.state import Deps, SceneAudioMeta, StoryboardState
from app.schemas import Scene, Script

log = get_logger(__name__)

CAPTION_WER_TOLERANCE = 0.25  # above this, captions use what was actually said


def word_error_rate(ref: str, hyp: str) -> float:
    def words(s: str) -> list[str]:
        s = unicodedata.normalize("NFC", s.lower())
        s = re.sub(r"speaker\s*\d+\s*:", " ", s)
        # Drop punctuation/symbols only; keep combining marks (Tamil vowel signs).
        s = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in s)
        return s.split()

    r, h = words(ref), words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / len(r)


def fallback_voices(script: Script, voices: list[str]) -> dict[str, str]:
    """Stable prebuilt voice per character (cast order)."""
    return {c.name.upper(): voices[i % len(voices)] for i, c in enumerate(script.characters)} if voices else {}


async def captions(state: StoryboardState, deps: Deps) -> StoryboardState:
    store, settings = deps.store, deps.settings
    script = Script.model_validate(store.load_json(state["script_path"]))
    language = state.get("language") or script.language
    scenes = {sc.order: sc for sc in script.scenes}
    clips = {c["order"]: c for c in state.get("motion_clips") or [] if c.get("path")}
    audio = [dict(a) for a in state.get("scene_audio") or []]
    pending = [a for a in audio if a.get("voice") == "pending"]
    if not pending:
        log.info("captions: no dialogue scenes; nothing to do")
        if deps.reporter is not None:
            await deps.reporter.mark_skipped("no dialogue scenes")
        return {}

    voices = fallback_voices(script, settings.dialogue_fallback_voices)
    for order in scenes:
        deps_names[order] = script.display_names()
    warnings: list[str] = []

    async def finalize(meta: dict) -> None:
        sc: Scene = scenes[meta["order"]]
        clip = clips.get(sc.order)
        if clip and clip.get("has_audio") and os.path.exists(clip["path"]):
            await _on_camera(meta, sc, clip["path"], language, deps)
        else:
            warn = await _off_screen(meta, sc, voices, deps)
            if warn:
                warnings.append(warn)

    await asyncio.gather(*(finalize(m) for m in pending))
    on_cam = sum(a["voice"] == "on_camera" for a in audio)
    off = sum(a["voice"] == "off_screen" for a in audio)
    if off:
        warnings.append(f"dialogue: {off} scene(s) voiced off-screen (no on-camera clip)")
    log.info("captions: %d on-camera, %d off-screen dialogue scene(s)", on_cam, off)
    return {"scene_audio": audio, "warnings": warnings}


deps_names: dict[int, dict[str, str]] = {}


async def _on_camera(meta: dict, sc: Scene, clip_path: str, language: str, deps: Deps) -> None:
    meta["voice"] = "on_camera"
    meta["path"] = None
    meta["hold"] = await probe_duration(clip_path) or meta["hold"]
    if deps.transcriber is None or not deps.settings.burn_subtitles:
        return  # no subtitles -> no transcription calls
    wav_path = deps.store.abspath(f"dialogue_{sc.order:02d}_clip.wav")
    try:
        await extract_audio(clip_path, wav_path)
        wav = deps.store.load_bytes(wav_path)
        transcript = await asyncio.to_thread(deps.transcriber.transcribe, wav, language)
        meta["transcript"] = transcript
        if transcript and word_error_rate(sc.spoken_text(), transcript) > CAPTION_WER_TOLERANCE:
            expected = [(l.speaker, l.text) for l in sc.lines]
            said = await asyncio.to_thread(deps.transcriber.attribute, wav, expected, language)
            if said:
                names = deps_names.get(sc.order, {})
                meta["caption"] = "\n".join(f"{names.get(s.upper(), s)}: {t}" for s, t in said)
    except (ProviderError, Exception) as exc:  # noqa: BLE001 - captions fall back to the script
        log.warning("transcription for scene %s failed (%s); captions use the script", sc.order, exc)


async def _off_screen(meta: dict, sc: Scene, voices: dict[str, str], deps: Deps) -> str | None:
    meta["voice"] = "off_screen"
    name = f"dialogue_{sc.order:02d}.wav"
    store = deps.store
    if store.exists(name):
        wav = store.load_bytes(store.abspath(name))
    elif deps.tts is None or not deps.settings.enable_tts:
        meta["voice"] = "silent"
        return None
    else:
        turns = [DialogueTurn(l.speaker, l.text, l.delivery) for l in sc.lines]
        try:
            wav = await asyncio.to_thread(deps.tts.synthesize_dialogue, turns, voices)
        except (ProviderError, ValueError) as exc:
            wav = silent_wav(deps.settings.narration_floor_seconds)
            store.save_bytes(name, wav)
            meta["path"] = store.abspath(name)
            return f"dialogue: scene {sc.order} voices failed ({exc}); silent"
        store.save_bytes(name, wav)
    meta["path"] = store.abspath(name)
    meta["hold"] = max(deps.settings.narration_floor_seconds, wav_duration_seconds(wav) + deps.settings.narration_pad_seconds)
    return None
