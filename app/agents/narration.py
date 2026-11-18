"""narration stage: voice narrator scenes and set every scene's hold (before motion).

Narrator scenes: TTS with the brand voice -> hold = max(floor, audio + pad).
Dialogue scenes: hold = estimated speech time (finalized by captions after motion).
Idempotent via narration_NN.wav. A failed line becomes silence; never fails the job.
"""

from __future__ import annotations

import asyncio

from app.adapters.errors import ProviderError
from app.workers.control import PauseRequested, gate, keep, paused
from app.adapters.tts import silent_wav, wav_duration_seconds
from app.agents.script_writer import WORDS_PER_SECOND
from app.core.logging import get_logger
from app.graph.state import SCENE_MAX_SECONDS, Deps, SceneAudioMeta, StoryboardState
from app.schemas import Scene, Script

log = get_logger(__name__)

DIALOGUE_PAD_SECONDS = 1.0


def estimate_speech_seconds(text: str, language: str = "en") -> float:
    words = len(text.split())
    return words / WORDS_PER_SECOND.get(language, 2.6)


def dialogue_hold(scene: Scene, language: str = "en") -> float:
    """The scene's planned length, stretched if its lines need longer (one clip caps at 15s)."""
    speech = estimate_speech_seconds(scene.spoken_text(), language) + DIALOGUE_PAD_SECONDS
    return min(SCENE_MAX_SECONDS, max(3.0, scene.seconds, speech))


async def voice_lines(
    items: list[tuple[int, str]], deps: Deps, min_holds: dict[int, float] | None = None,
) -> tuple[list[SceneAudioMeta], list[str]]:
    """Voice (order, text) narration lines with the brand voice (idempotent per order)."""
    store, settings = deps.store, deps.settings
    floor, pad = settings.narration_floor_seconds, settings.narration_pad_seconds
    out: list[SceneAudioMeta] = []
    warnings: list[str] = []
    for i, (order, text) in enumerate(items):
        name = f"narration_{order:02d}.wav"
        failed = f"narration_{order:02d}.failed"  # silence stood in; resume retries it for real
        if store.exists(name) and not store.exists(failed):
            wav = store.load_bytes(store.abspath(name))
        else:
            try:
                await gate(deps)  # each line is its own paid call
            except PauseRequested:
                raise paused("narration", i, len(items), "narration lines") from None
            async def speak(name: str = name, failed: str = failed, text: str = text) -> bytes:
                wav = await asyncio.to_thread(deps.tts.synthesize, text)
                store.save_bytes(name, wav)
                store.remove(failed)  # only once the real voiceover is safely on disk
                return wav

            try:
                wav = await keep(deps, speak)
            except ProviderError as exc:
                log.warning("narration for scene %s failed (%s); using silence", order, exc)
                warnings.append(f"narration: scene {order} voiceover failed ({exc}); silent")
                wav = silent_wav(floor)
                store.write_marker(failed)
                store.save_bytes(name, wav)
        out.append(SceneAudioMeta(
            order=order, voice="narrator", path=store.abspath(name),
            hold=max(floor, wav_duration_seconds(wav) + pad, (min_holds or {}).get(order, 0.0)), caption=text,
        ))
    return out, warnings


async def narration(state: StoryboardState, deps: Deps) -> StoryboardState:
    store, settings = deps.store, deps.settings
    script = Script.model_validate(store.load_json(state["script_path"]))
    language = state.get("language") or script.language
    voicing = settings.enable_tts and deps.tts is not None
    uniform = max(settings.narration_floor_seconds, state["duration"] / max(1, len(script.scenes)))

    out: dict[int, SceneAudioMeta] = {}
    to_voice: list[tuple[int, str]] = []
    for sc in sorted(script.scenes, key=lambda s: s.order):
        if sc.mode == "dialogue" and sc.lines:
            out[sc.order] = SceneAudioMeta(order=sc.order, voice="pending", path=None,
                                           hold=dialogue_hold(sc, language), caption=sc.caption_text(script.display_names()))
        elif voicing and sc.narration.strip():
            to_voice.append((sc.order, sc.narration))
        else:
            out[sc.order] = SceneAudioMeta(order=sc.order, voice="silent", path=None,
                                           hold=sc.seconds or uniform, caption=sc.narration)
    planned = {sc.order: min(SCENE_MAX_SECONDS, sc.seconds) for sc in script.scenes if sc.seconds}
    voiced, warnings = await voice_lines(to_voice, deps, planned) if to_voice else ([], [])
    for meta in voiced:
        out[meta["order"]] = meta

    if deps.reporter is not None and not voiced:
        await deps.reporter.mark_skipped("no narrator scenes" if voicing else "narration off")
    log.info("narration: %d scene(s) voiced, %d dialogue scene(s) pending",
             len(voiced), sum(a["voice"] == "pending" for a in out.values()))
    return {"scene_audio": [out[k] for k in sorted(out)], "warnings": warnings}
