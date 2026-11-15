"""Startup check: every configured model id must be visible to the API key."""

from __future__ import annotations

import asyncio

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)


def configured_models(s: Settings) -> dict[str, list[str]]:
    return {
        "script": [s.role_model("script")],
        "judge": [s.role_model("judge")],
        "designer": [s.role_model("designer")],
        "publish": [s.role_model("publish")],
        "image": [s.gemini_image_model],
        "tts": list(s.tts_models),
        "transcribe": [s.transcribe_model, s.transcribe_fallback_model],
        "music": list(s.music_model_chain),
    }


async def validate_models(s: Settings) -> dict:
    """Returns {role: {"ok": [...], "missing": [...]}}; prunes missing ids from chains.
    A listing error keeps everything as configured (a network blip must not disable features)."""
    roles = configured_models(s)
    if s.mock_mode:
        return {"checked": False, "reason": "mock mode", "roles": roles}
    try:
        from google import genai

        from google.genai import types

        # v1beta lists every model; SDK 3.x's default (v1) omits Interactions-only ones (TTS, Lyria...).
        client = genai.Client(api_key=s.gemini_api_key, http_options=types.HttpOptions(api_version="v1beta"))
        visible = {
            (getattr(m, "name", "") or "").removeprefix("models/")
            for m in await asyncio.to_thread(lambda: list(client.models.list()))
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("could not list models (%s); using configuration as-is", exc)
        return {"checked": False, "reason": str(exc)[:200], "roles": roles}

    report: dict = {"checked": True, "roles": {}}
    for role, ids in roles.items():
        ok = [m for m in ids if m.split(":", 1)[-1] in visible or m.startswith("anthropic:")]
        missing = [m for m in ids if m not in ok]
        report["roles"][role] = {"ok": ok, "missing": missing}
        if missing:
            log.error("model(s) for %s not visible to this API key: %s", role, missing)
    pruned_tts = [m for m in s.tts_model_chain if m not in report["roles"]["tts"]["missing"]]
    if pruned_tts:
        s.tts_model_chain = pruned_tts
    pruned_music = [m for m in s.music_model_chain if m not in report["roles"]["music"]["missing"]]
    if pruned_music:
        s.music_model_chain = pruned_music
    return report
