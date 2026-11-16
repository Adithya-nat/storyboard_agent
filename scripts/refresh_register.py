"""Propose fresh everyday slang for the modern speech registers (LIVE, ~$0.05). Run monthly:

    python -m scripts.refresh_register --language ta [--out suggestions.md]

Uses Gemini with Google Search grounding to find what people actually say now, and prints a
suggested replacement for the slang line of app/prompts/register/<lang>_modern.md, with sources.
It never edits the prompt files: review the suggestion and paste what you agree with.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.adapters.genai_client import make_client
from app.core.config import get_settings
from app.prompts import PROMPTS_DIR

_ASK = {
    "ta": (
        "Search for the colloquial Tamil slang and expressions young people in Tamil Nadu (Chennai, Madurai, "
        "Coimbatore) use in everyday speech and on Instagram/YouTube right now. Separate what is current from "
        "what now sounds dated. Write every word in Tamil script, and English loanwords as Tamil speakers say "
        "them, in Tamil script."
    ),
    "en": (
        "Search for the casual English slang and expressions Gen Z and millennials actually use in everyday "
        "speech and on social video right now, and which once-popular terms now sound dated or cringe."
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", choices=["ta", "en"], default="ta")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from google.genai import types

    current = (PROMPTS_DIR / "register" / f"{a.language}_modern.md").read_text(encoding="utf-8")
    settings = get_settings()
    resp = make_client(settings, timeout_seconds=300).models.generate_content(
        model=settings.role_model("script"),
        contents=(
            f"{_ASK[a.language]}\n\nHere is our current speech-register guide:\n<guide>\n{current}\n</guide>\n\n"
            "Return: (1) terms in the guide that now sound dated, (2) 10-15 current terms with a one-line meaning "
            "and who says them (age, region), (3) a suggested replacement for the guide's slang bullet, "
            "keeping its rule 'one or two touches per line at most'."
        ),
        config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]),
    )
    sources = []
    for cand in resp.candidates or []:
        meta = getattr(cand, "grounding_metadata", None)
        for chunk in getattr(meta, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if web and web.uri:
                sources.append(f"- {web.title or ''} {web.uri}")
    text = resp.text + ("\n\nSources:\n" + "\n".join(dict.fromkeys(sources)) if sources else "")
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
