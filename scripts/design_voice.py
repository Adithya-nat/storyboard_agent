"""Create your channel's persistent narrator voice (Voice Design) and save its id to .env.

    python -m scripts.design_voice --name "Channel narrator" \\
        --describe "A warm, cinematic documentary narrator in his forties, calm and intimate" \\
        --gender male --language en-US

Writes NARRATOR_VOICE_ID=... to .env and saves a preview to narrator_preview.wav.
Run once per voice (Tamil channel: --language ta-IN and describe a Tamil narrator).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from app.adapters.genai_client import b64bytes, make_client
from app.core.config import get_settings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Storyboard narrator")
    ap.add_argument("--describe", required=True, help="natural-language voice description")
    ap.add_argument("--gender", choices=["male", "female"], default="female")
    ap.add_argument("--language", default="en-US", help="BCP-47 code, e.g. en-US or ta-IN")
    ap.add_argument("--model", default="gemini-3.8-flash-tts")
    ap.add_argument("--env", default=".env")
    a = ap.parse_args()

    client = make_client(get_settings(), timeout_seconds=120)
    voice = client.voices.create(
        api_version="v1beta",  # Voice Design is v1beta-only; the SDK default (v1) 404s
        store=True,
        voice={
            "model": a.model,
            "type": "prompted",
            "display_name": a.name,
            "gender": a.gender,
            "language_code": a.language,
            "prompted": {"input": a.describe},
        },
    )
    if voice.sample_audio and voice.sample_audio.data:
        Path("narrator_preview.wav").write_bytes(b64bytes(voice.sample_audio.data))
        print("preview -> narrator_preview.wav")
    env = Path(a.env)
    text = env.read_text(encoding="utf-8") if env.exists() else ""
    line = f"NARRATOR_VOICE_ID={voice.id}"
    text = re.sub(r"^NARRATOR_VOICE_ID=.*$", line, text, flags=re.M) if "NARRATOR_VOICE_ID=" in text else text.rstrip() + f"\n{line}\n"
    env.write_text(text, encoding="utf-8")
    print(f"saved {line} to {env}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
