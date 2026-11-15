"""Setup checker: is everything in place to generate videos?

    python -m scripts.doctor          # free checks
    python -m scripts.doctor --live   # + one tiny real call per model (~$0.20)

Exit code 0 = ready, 1 = something needs fixing (each failure prints how to fix it).
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys

from app.core.config import get_settings

OK, WARN, FAIL = "✓", "!", "✗"
results: list[tuple[str, str, str]] = []


def check(name: str, status: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))


def ffmpeg_checks() -> None:
    if not shutil.which("ffmpeg"):
        check("ffmpeg", FAIL, "install it: brew install ffmpeg")
        return
    filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    missing = [f for f in ("zoompan", "xfade", "sidechaincompress", "amix", "adelay") if f" {f} " not in filters]
    check("ffmpeg filters", FAIL if missing else OK, f"missing: {missing}" if missing else "zoompan, xfade, sidechaincompress")


def tamil_checks(s) -> None:
    from pathlib import Path

    from app.core.images import raqm_available

    check("Tamil text shaping (libraqm)", OK if raqm_available() else WARN,
          "" if raqm_available() else "Tamil videos are disabled until: brew install libraqm")
    check("Tamil font", OK if Path(s.tamil_font_path).exists() else FAIL, s.tamil_font_path)


async def db_checks(s) -> None:
    from sqlalchemy import text

    from app.db.session import get_engine

    try:
        async with get_engine().connect() as conn:
            head = (await conn.execute(text("select version_num from alembic_version"))).scalar()
        check("database", OK if head == "0004_cost_budget" else FAIL,
              f"migration {head}" + ("" if head == "0004_cost_budget" else " — run: alembic upgrade head"))
    except Exception as exc:  # noqa: BLE001
        check("database", FAIL, f"{str(exc)[:120]} — start it: docker compose up -d")


async def model_checks(s) -> None:
    from app.core.model_check import validate_models

    if s.mock_mode:
        check("Gemini API key", FAIL, "set GEMINI_API_KEY in storyboard_agent/.env (see SETUP.md)")
        return
    check("Gemini API key", OK)
    report = await validate_models(s.model_copy())
    if not report.get("checked"):
        check("model access", WARN, f"couldn't list models: {report.get('reason')}")
        return
    for role, r in report["roles"].items():
        check(f"model: {role}", FAIL if r["missing"] and not r["ok"] else WARN if r["missing"] else OK,
              ", ".join(r["ok"]) + (f" (not visible: {', '.join(r['missing'])})" if r["missing"] else ""))
    omni = [m for m in s.omni_model_chain]
    check("AI video provider", OK if s.video_provider == "omni" else WARN, f"{s.video_provider} ({', '.join(omni)})")


async def budget_checks(s) -> None:
    try:
        from app.db.session import get_sessionmaker
        from app.observability.budget import budget_status

        async with get_sessionmaker()() as session:
            b = await budget_status(session, s)
        check("monthly budget", FAIL if b.over else WARN if b.warn else OK,
              f"${b.month_spent_usd:.2f} of ${b.cap_usd:.0f} used this month")
    except Exception as exc:  # noqa: BLE001
        check("monthly budget", WARN, str(exc)[:100])


def live_checks(s) -> None:
    from google.genai import types

    from app.adapters.genai_client import interactions_create, make_client
    from app.adapters.tts import GeminiTTSAdapter

    c = make_client(s, timeout_seconds=300)
    for role in ("script", "judge"):
        m = s.role_model(role)
        try:
            c.models.generate_content(model=m, contents="Reply: ok")
            check(f"live: {m}", OK)
        except Exception as exc:  # noqa: BLE001
            check(f"live: {m}", FAIL, str(exc)[:120])
    try:
        r = c.models.generate_content(
            model=s.gemini_image_model, contents="a small red paper boat on a lake",
            config=types.GenerateContentConfig(response_modalities=["IMAGE"], image_config=types.ImageConfig(aspect_ratio="9:16", image_size="1K")),
        )
        check(f"live: {s.gemini_image_model}", OK if any(p.inline_data for p in r.candidates[0].content.parts) else FAIL)
    except Exception as exc:  # noqa: BLE001
        check(f"live: {s.gemini_image_model}", FAIL, str(exc)[:120])
    try:
        GeminiTTSAdapter(s).synthesize("Setup check.")
        check(f"live: {s.tts_models[0]}", OK)
    except Exception as exc:  # noqa: BLE001
        check(f"live: {s.tts_models[0]}", FAIL, str(exc)[:120])
    try:
        it = interactions_create(c, model=s.music_model_chain[0], input="A five second calm instrumental sting. Instrumental only.")
        check(f"live: {s.music_model_chain[0]}", OK if it.output_audio else FAIL)
    except Exception as exc:  # noqa: BLE001
        check(f"live: {s.music_model_chain[0]}", FAIL, str(exc)[:120])
    if s.video_provider == "omni":
        try:
            it = interactions_create(c, model=s.omni_model_chain[0], input="A candle flame flickering, single continuous shot.",
                                     response_format={"type": "video", "aspect_ratio": "9:16", "resolution": "360p", "duration": "3s"})
            check(f"live: {s.omni_model_chain[0]} (3s @360p)", OK if it.output_video else FAIL)
        except Exception as exc:  # noqa: BLE001
            check(f"live: {s.omni_model_chain[0]}", FAIL, str(exc)[:120])


async def main_async(live: bool) -> int:
    s = get_settings()
    print("Storyboard setup check\n")
    ffmpeg_checks()
    tamil_checks(s)
    await db_checks(s)
    await model_checks(s)
    await budget_checks(s)
    if live and not s.mock_mode:
        print("\nLive calls (paid, ~$0.20):")
        live_checks(s)
    fails = [r for r in results if r[0] == FAIL]
    print(f"\n{'Ready to generate.' if not fails else f'{len(fails)} problem(s) to fix — see SETUP.md.'}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="make one tiny real call per model (~$0.20)")
    return asyncio.run(main_async(ap.parse_args().live))


if __name__ == "__main__":
    sys.exit(main())
