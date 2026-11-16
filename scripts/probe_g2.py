"""Gate G2 — scene-level multi-shot Omni probe (LIVE, ~$4-5 at 720p). Run manually:

    python -m scripts.probe_g2 --out /path/to/dir [--only en_a,en_b,en_c,ta_a,ta_b]

Variants per language: (a) one 12s timecoded multi-shot scene from a start frame; (b) the same
with character sheets as subject references; (c, EN) an 8s extension of (a) via
previous_interaction_id. Measures shot-plan adherence, line WER, in-clip voice consistency,
lip-sync and identity vs sheets. Decision rule lives in the plan; this only measures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from google.genai import types
from pydantic import BaseModel, Field

from app.adapters.genai_client import b64str, make_client
from app.core.config import get_settings
from scripts.probe_g1 import (
    JUDGE, contact_sheet, download_video, extract_audio, ffprobe, gen_image, log, omni, sniff_mime,
    transcribe, wer,
)

SECONDS = 12
STYLE = (
    "Cinematic live-action film still, photoreal, 35mm, warm practical kitchen light at night, "
    "shallow depth of field, vertical 9:16. Original fictional characters (not real people)."
)

SPECS = {
    "en": {
        "chars": {
            "RIYA": "a woman in her mid-twenties, shoulder-length wavy black hair, oversized mustard hoodie",
            "DEV": "a man in his late twenties, short cropped hair, light stubble, creased white office shirt",
        },
        "voices": {
            "RIYA": "FEMALE, 20s: bright, quick, teasing mid-range voice",
            "DEV": "MALE, late 20s: low, tired, hesitant baritone",
        },
        "setting": "a cramped city apartment kitchen at night, a pot of instant noodles on the stove",
        "shots": [
            ("0-3", "MEDIUM-WIDE: RIYA stirs noodles at the stove; DEV walks in and drops his laptop bag",
             [("RIYA", "You're home early. Like, suspiciously early.")]),
            ("3-7", "CLOSE-UP on DEV, avoiding her eyes, rubbing the back of his neck",
             [("DEV", "Yeah, so... I kind of quit today.")]),
            ("7-9", "CLOSE-UP on RIYA: she freezes, spoon in mid-air, silent", []),
            ("9-12", "TWO-SHOT across the kitchen counter",
             [("RIYA", "You quit? Without a backup plan?")]),
        ],
        "extension": [
            ("0-4", "CLOSE-UP on DEV, a guilty half-smile", [("DEV", "I have a plan. It's just not a good one.")]),
            ("4-8", "TWO-SHOT: RIYA laughs despite herself and shakes her head, silent", []),
        ],
        "language_note": "",
    },
    "ta": {
        "chars": {
            "MEENA": "a Tamil woman in her mid-twenties, long black hair in a loose braid, mustard cotton kurti",
            "KATHIR": "a Tamil man in his late twenties, short hair, light stubble, creased white office shirt",
        },
        "voices": {
            "MEENA": "FEMALE, 20s: bright, quick, teasing voice, speaks colloquial Chennai Tamil",
            "KATHIR": "MALE, late 20s: low, tired, hesitant voice, speaks colloquial Chennai Tamil",
        },
        "setting": "a cramped Chennai flat kitchen at night, a pot of noodles on a gas stove",
        "shots": [
            ("0-3", "MEDIUM-WIDE: MEENA stirs noodles at the stove; KATHIR walks in and drops his laptop bag",
             [("MEENA", "என்ன டா, இவ்ளோ சீக்கிரம் வந்துட்ட?")]),
            ("3-7", "CLOSE-UP on KATHIR, avoiding her eyes, rubbing the back of his neck",
             [("KATHIR", "அது... இன்னைக்கு வேலைய ரிசைன் பண்ணிட்டேன்.")]),
            ("7-9", "CLOSE-UP on MEENA: she freezes, spoon in mid-air, silent", []),
            ("9-12", "TWO-SHOT across the kitchen counter",
             [("MEENA", "என்னது? பிளான் பி கூட இல்லாமலா?")]),
        ],
        "extension": [],
        "language_note": "All dialogue is spoken in colloquial Chennai Tamil (தமிழ்), pronounced exactly as written.",
    },
}


def scene_prompt(spec, shots, *, with_refs: bool, extension: bool = False) -> str:
    voices = spec["voices"]
    head = (
        "The scene continues in the same place with the same two people and voices."
        if extension else
        ("The FIRST image is the opening frame of this video. The other images are character references: "
         + ", ".join(f"image {i + 2} = {n}" for i, n in enumerate(spec["chars"])) + ". Keep each character identical to their reference. "
         if with_refs else "Start from this exact frame. ")
        + f"Cinematic live-action scene in {spec['setting']}, cut like a film scene with these shots:"
    )
    rows = []
    for t, shot, lines in shots:
        said = " ".join(f'{s} ({voices[s]}) says: "{x}"' for s, x in lines) or "No dialogue."
        rows.append(f"[{t}s] {shot}. {said}")
    cast = "Cast: " + "; ".join(f"{n} — {d}; voice {voices[n]}" for n, d in spec["chars"].items()) + "."
    return "\n".join([
        head, *rows, cast,
        "Lip-sync accurately and speak each quoted line EXACTLY as written, word for word, in that "
        "character's voice. Each character keeps the same voice in every shot. " + spec["language_note"],
        "Sound: quiet kitchen room tone continuous across cuts. No music. No narrator. No on-screen text or subtitles.",
    ])


class ShotCheck(BaseModel):
    planned: int = Field(description="Planned shot number (1-based)")
    present: bool = Field(description="A shot matching this plan appears, in order")
    start_s: float = Field(default=0, description="Approx start time of that shot in the clip")
    note: str = ""


class SceneJudge(BaseModel):
    cuts_detected: int = Field(description="Number of camera cuts you see")
    shots: list[ShotCheck]
    lines_exact: int = Field(description="1-5: lines spoken exactly as written")
    right_speakers: int = Field(description="1-5: each line spoken by the right character")
    lip_sync: int = Field(description="1-5")
    voice_consistency: int = Field(description="1-5: each character's voice stays the same throughout")
    set_continuity: int = Field(description="1-5: room, light and props stay consistent across cuts")
    identity_match: int = Field(description="1-5: characters match the reference images (if given); else the opening frame")
    feels_like_a_scene: int = Field(description="1-5: reads as one continuous, human-edited film scene")
    notes: str


def judge_scene(client, mp4: Path, spec, shots, refs: list[bytes]) -> SceneJudge:
    plan = "\n".join(f"{i}. [{t}s] {s} — lines: {', '.join(f'{a}: {b}' for a, b in ls) or 'none'}"
                     for i, (t, s, ls) in enumerate(shots, 1))
    parts: list = [types.Part.from_bytes(data=mp4.read_bytes(), mime_type="video/mp4")]
    for r in refs:
        parts.append(types.Part.from_bytes(data=r, mime_type=sniff_mime(r)))
    parts.append(
        "You are a strict film editor reviewing an AI-generated scene. The first item is the clip; any "
        f"images after it are character references ({', '.join(spec['chars'])}, in that order).\n"
        f"Planned shots:\n{plan}\nVoices: {json.dumps(spec['voices'], ensure_ascii=False)}\nScore strictly; 5 = flawless."
    )
    resp = client.models.generate_content(
        model=JUDGE, contents=parts,
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=SceneJudge,
                                           thinking_config=types.ThinkingConfig(thinking_level="medium")),
    )
    return SceneJudge.model_validate_json(resp.text)


def make_assets(client, spec, out: Path, tag: str) -> tuple[list[bytes], bytes]:
    sheets = []
    for name, look in spec["chars"].items():
        p = out / f"{tag}_sheet_{name}.img"
        if not p.exists():
            data, _ = gen_image(client, f"{STYLE} Character reference portrait, head and shoulders, plain background: {name}, {look}.", [])
            p.write_bytes(data)
            log(f"{tag} sheet {name} ok")
        sheets.append(p.read_bytes())
    p = out / f"{tag}_frame.img"
    if not p.exists():
        who = "; ".join(f"{n} (reference image {i + 1}): {d}" for i, (n, d) in enumerate(spec["chars"].items()))
        data, _ = gen_image(client, f"{STYLE} {spec['setting']}. {spec['shots'][0][1]}. Characters — {who}. Match each to their reference exactly.", sheets)
        p.write_bytes(data)
        log(f"{tag} start frame ok")
    return sheets, p.read_bytes()


def measure(client, mp4: Path, spec, shots, refs, lang: str, rec: dict) -> None:
    rec["probe"] = ffprobe(mp4)
    contact_sheet(mp4)
    ref = " ".join(x for _, _, ls in shots for _, x in ls)
    if rec["probe"]["has_audio"]:
        wav = extract_audio(mp4)
        models = ["gemini-3.5-transcribe"] + (["gemini-3.8-flash"] if lang == "ta" else [])
        rec["transcripts"], rec["wer"] = {}, {}
        for m in models:
            try:
                tx = transcribe(client, wav, m)
                rec["transcripts"][m], rec["wer"][m] = tx, round(wer(ref, tx), 3)
            except Exception as e:  # noqa: BLE001
                rec["transcripts"][m] = f"ERR {e}"
    try:
        rec["judge"] = judge_scene(client, mp4, spec, shots, refs).model_dump()
    except Exception as e:  # noqa: BLE001
        rec["judge"] = {"error": str(e)[:300]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="en_a,en_b,en_c,ta_a,ta_b")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(","))
    client = make_client(get_settings(), timeout_seconds=900)
    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for lang, spec in SPECS.items():
        if not any(v.startswith(lang) for v in only):
            continue
        sheets, frame = make_assets(client, spec, out, lang)
        img = lambda b: {"type": "image", "data": b64str(b), "mime_type": sniff_mime(b)}  # noqa: E731
        first_id = None
        for variant, refs in (("a", []), ("b", sheets)):
            tag = f"{lang}_{variant}"
            if tag not in only:
                continue
            rec: dict = {}
            try:
                inputs = [img(frame), *[img(s) for s in refs],
                          {"type": "text", "text": scene_prompt(spec, spec["shots"], with_refs=bool(refs))}]
                it, dt = omni(client, inputs=inputs, seconds=SECONDS, store=True)
                rec["latency_s"], rec["interaction_id"] = round(dt, 1), it.id
                first_id = it.id if variant == "a" else first_id
                mp4 = out / f"{tag}.mp4"
                download_video(client, it, mp4)
                measure(client, mp4, spec, spec["shots"], sheets, lang, rec)
            except Exception as e:  # noqa: BLE001
                rec["error"] = str(e)[:500]
            results[tag] = rec
            results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            log(tag, json.dumps({k: rec.get(k) for k in ("wer", "error")}, ensure_ascii=False),
                {k: v for k, v in (rec.get("judge") or {}).items() if k not in ("notes", "shots")})
        if f"{lang}_c" in only and spec["extension"] and first_id:
            rec = {}
            try:
                it, dt = omni(client, inputs=scene_prompt(spec, spec["extension"], with_refs=False, extension=True),
                              seconds=8, store=True, previous=first_id)
                rec["latency_s"] = round(dt, 1)
                mp4 = out / f"{lang}_c.mp4"
                download_video(client, it, mp4)
                measure(client, mp4, spec, spec["shots"] + spec["extension"], sheets, lang, rec)
            except Exception as e:  # noqa: BLE001
                rec["error"] = str(e)[:500]
            results[f"{lang}_c"] = rec
            results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            log(f"{lang}_c", {k: v for k, v in (rec.get("judge") or {}).items() if k not in ("notes", "shots")}, rec.get("error"))
    log("done ->", results_path)


if __name__ == "__main__":
    main()
