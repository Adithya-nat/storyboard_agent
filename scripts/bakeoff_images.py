"""G0 image bake-off (LIVE, ~$0.60): Nano Banana 2.1 vs Nano Banana Pro on identical shots.

    python -m scripts.bakeoff_images --out DIR

Each model renders a character sheet, then 3 storyboard shots conditioned on it. A Gemini 4
Argon vision judge scores character consistency, prompt adherence, cinematic quality and
artifacts (1-10). Writes DIR/<model>_*.jpg, a side-by-side sheet and results.json.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field

from app.adapters.gemini import image_mime
from app.adapters.genai_client import make_client
from app.core.config import get_settings

MODELS = ["gemini-nano-banana-2.1", "gemini-3-pro-image"]
STYLE = "Cinematic live-action film still, photoreal, 35mm, warm dusk light with teal shadows, vertical 9:16."
SHEET = "Character reference: ANNA, a woman in her thirties, short auburn bob, green raincoat, small scar on her left eyebrow. Front-facing portrait, plain background."
SHOTS = [
    "ANNA runs across a rain-soaked night market, neon reflections in puddles, she looks back over her shoulder.",
    "ANNA kneels beside a stray cat under a striped awning, offering it half a bun, close-up on her face and hands.",
    "ANNA stands on a rooftop at dawn holding the cat, the city behind her, wind in her hair, wide shot.",
]


class Score(BaseModel):
    consistency: int = Field(description="1-10: ANNA looks like the same person as in the reference sheet")
    adherence: int = Field(description="1-10: the image shows exactly what the shot describes")
    quality: int = Field(description="1-10: cinematic, photoreal, well composed")
    artifacts: int = Field(description="1-10: 10 = no anatomy/text/AI artifacts")
    notes: str


def gen(client, model: str, prompt: str, refs: list[bytes]) -> tuple[bytes, float]:
    contents: list = [prompt] + [types.Part.from_bytes(data=r, mime_type=image_mime(r)) for r in refs]
    t = time.time()
    resp = client.models.generate_content(
        model=model, contents=contents,
        config=types.GenerateContentConfig(response_modalities=["IMAGE"],
                                           image_config=types.ImageConfig(aspect_ratio="9:16", image_size="2K")),
    )
    for p in resp.candidates[0].content.parts:
        if p.inline_data and p.inline_data.data:
            return p.inline_data.data, time.time() - t
    raise RuntimeError(f"{model}: no image")


def judge(client, sheet: bytes, shot: bytes, desc: str) -> Score:
    resp = client.models.generate_content(
        model="gemini-4-argon",
        contents=["Reference sheet:", types.Part.from_bytes(data=sheet, mime_type=image_mime(sheet)),
                  "Generated shot:", types.Part.from_bytes(data=shot, mime_type=image_mime(shot)),
                  f"The shot was meant to show: {desc}\nScore it strictly."],
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=Score),
    )
    return Score.model_validate_json(resp.text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    client = make_client(get_settings(), timeout_seconds=300)
    results: dict = {}
    rows = []
    for model in MODELS:
        sheet, _ = gen(client, model, f"{STYLE} {SHEET}", [])
        (out / f"{model}_sheet.jpg").write_bytes(sheet)
        shots, scores, times = [], [], []
        for i, desc in enumerate(SHOTS, 1):
            img, dt = gen(client, model, f"{STYLE} {desc} Keep ANNA identical to the reference sheet.", [sheet])
            (out / f"{model}_shot{i}.jpg").write_bytes(img)
            s = judge(client, sheet, img, desc)
            shots.append(img)
            scores.append(s.model_dump())
            times.append(round(dt, 1))
            print(model, i, s.model_dump(exclude={"notes"}), f"{dt:.1f}s", flush=True)
        avg = {k: round(sum(s[k] for s in scores) / len(scores), 2) for k in ("consistency", "adherence", "quality", "artifacts")}
        results[model] = {"avg": avg, "overall": round(sum(avg.values()) / 4, 2), "seconds": times, "scores": scores}
        rows.append([sheet] + shots)
    thumbs = [[Image.open(io.BytesIO(b)).convert("RGB").resize((270, 480)) for b in row] for row in rows]
    sheet_img = Image.new("RGB", (270 * 4, 480 * len(thumbs)), "white")
    for r, row in enumerate(thumbs):
        for c, im in enumerate(row):
            sheet_img.paste(im, (c * 270, r * 480))
    sheet_img.save(out / "comparison.jpg", quality=88)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({m: (v["avg"], v["overall"]) for m, v in results.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
