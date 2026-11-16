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
        **({"video": list(s.omni_model_chain)} if s.video_provider == "omni" else {}),
    }


def list_visible_models(api_key: str) -> set[str]:
    """Every model id this key can see (free call; raises on a bad key)."""
    from google import genai
    from google.genai import types

    # v1beta lists every model; SDK 3.x's default (v1) omits Interactions-only ones (TTS, Lyria...).
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))
    return {(getattr(m, "name", "") or "").removeprefix("models/") for m in client.models.list()}


def _roles_report(roles: dict[str, list[str]], visible: set[str]) -> dict:
    out = {}
    for role, ids in roles.items():
        ok = [m for m in ids if m.split(":", 1)[-1] in visible or m.startswith("anthropic:")]
        out[role] = {"ok": ok, "missing": [m for m in ids if m not in ok]}
    return out


async def check_key(s: Settings, api_key: str) -> dict:
    """What a tester's key can do. Never raises; never echoes the key."""
    from app.core.keys import redact

    try:
        visible = await asyncio.to_thread(list_visible_models, api_key)
    except Exception as exc:  # noqa: BLE001
        msg = redact(str(exc))
        bad = any(t in msg.lower() for t in ("api key not valid", "api_key_invalid", "permission", "401", "403"))
        return {"ok": False, "roles": {}, "message": "Google rejected this key." if bad else f"Couldn't check the key: {msg[:160]}"}
    roles = _roles_report(configured_models(s), visible)
    core = ("script", "image", "designer")
    ok = all(roles[r]["ok"] for r in core if r in roles)
    video = roles.get("video")
    message = "Ready to make films."
    if not ok:
        message = "This key can't see the writing or image models it needs."
    elif video is not None and not video["ok"]:
        message = "Writing and images work, but AI video isn't available. Enable billing on this key's Google project."
    return {"ok": ok, "video_ok": bool(video is None or video["ok"]), "roles": roles, "message": message}


async def validate_models(s: Settings) -> dict:
    """Returns {role: {"ok": [...], "missing": [...]}}; prunes missing ids from chains.
    A listing error keeps everything as configured (a network blip must not disable features)."""
    roles = configured_models(s)
    if s.mock_mode:
        return {"checked": False, "reason": "mock mode", "roles": roles}
    try:
        visible = await asyncio.to_thread(list_visible_models, s.gemini_api_key)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not list models (%s); using configuration as-is", exc)
        return {"checked": False, "reason": str(exc)[:200], "roles": roles}

    report: dict = {"checked": True, "roles": _roles_report(roles, visible)}
    for role, r in report["roles"].items():
        if r["missing"]:
            log.error("model(s) for %s not visible to this API key: %s", role, r["missing"])
    pruned_tts = [m for m in s.tts_model_chain if m not in report["roles"]["tts"]["missing"]]
    if pruned_tts:
        s.tts_model_chain = pruned_tts
    pruned_music = [m for m in s.music_model_chain if m not in report["roles"]["music"]["missing"]]
    if pruned_music:
        s.music_model_chain = pruned_music
    return report
