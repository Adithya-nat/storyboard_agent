"""Gate G1 — on-camera dialogue probe (LIVE, costs ~$3-5). Run manually:

    python -m scripts.probe_g1 --out /path/to/dir [--skip-images] [--only en_a,en_b,ta]

Measures whether Gemini Omni can carry storyboard DIALOGUE scenes:
  * en_a  English, 3 independent image-to-video clips with fixed voice descriptions
  * en_b  English, a multi-turn chain (clip 1 + 2 extensions via previous_interaction_id)
  * ta    Tamil, 2 independent clips

For every clip it records: audio stream present (ffprobe), transcript (gemini-3.5-transcribe
with speaker diarization; Tamil also via gemini-3.8-flash), per-clip WER vs the script lines,
and an Argon video judge (lip-sync / acting / exact-lines 1-5). Per variant, an Argon audio
judge scores each character's voice consistency across clips (1-5). Results -> results.json
plus contact sheets (3 frames per clip) for human inspection.

Decision rule (plan G1): PASS = WER <= 15% AND lip-sync >= 4 AND voice consistency >= 4 for
the better of en_a / en_b. This script only MEASURES; the decision is made with the user.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

from google.genai import types
from pydantic import BaseModel, Field

from app.adapters.genai_client import b64bytes, b64str, interactions_create, make_client
from app.core.config import get_settings

OMNI = "gemini-omni-1.2-flash"
IMG = "gemini-nano-banana-2.1"
JUDGE = "gemini-4-argon"
RES = "720p"

STYLE = (
    "Cinematic live-action film still, photoreal, 35mm, moody teal-and-amber storm light, "
    "shallow depth of field, vertical 9:16 composition. Original fictional characters (not "
    "real people)."
)

EN = {
    "chars": {
        "MAYA": "a weathered lighthouse keeper in her sixties, silver braid, navy wool sweater",
        "LEO": "a soaked fifteen-year-old runaway boy, dark curly hair, red rain jacket",
    },
    "voices": {
        "MAYA": "warm, weathered alto voice of a woman in her sixties, gentle Scottish accent",
        "LEO": "bright, slightly hoarse voice of a fifteen-year-old boy, American accent",
    },
    "scenes": [
        ("Inside the lighthouse doorway during a storm, MAYA holds a lantern, LEO stands dripping.",
         [("MAYA", "You shouldn't be out here in this storm."), ("LEO", "I had nowhere else to go.")]),
        ("At the top of the lighthouse beside the great lamp, both look out at the dark sea.",
         [("LEO", "Why keep the light burning? No ships come anymore."),
          ("MAYA", "Because someone might still be lost.")]),
        ("MAYA hands LEO a long match next to the lamp; warm light on both faces.",
         [("MAYA", "Here. Your turn to light it."), ("LEO", "Then nobody gets lost tonight.")]),
    ],
    "language_note": "",
}

TA = {
    "chars": {
        "KAMALA": "a weathered Tamil lighthouse keeper in her sixties, silver hair bun, maroon cotton saree and grey shawl",
        "ARUN": "a soaked fifteen-year-old Tamil boy, short dark hair, blue rain jacket",
    },
    "voices": {
        "KAMALA": "warm, weathered alto voice of a Tamil woman in her sixties, natural Chennai Tamil",
        "ARUN": "bright, slightly hoarse voice of a fifteen-year-old Tamil boy, natural Chennai Tamil",
    },
    "scenes": [
        ("Inside the lighthouse doorway during a storm, KAMALA holds a lantern, ARUN stands dripping.",
         [("KAMALA", "இந்த புயலில் நீ வெளியே இருக்கக்கூடாது."), ("ARUN", "எனக்கு போக வேறு இடமில்லை.")]),
        ("At the top of the lighthouse beside the great lamp, both look out at the dark sea.",
         [("ARUN", "கப்பல்கள் வருவதே இல்லை. ஏன் இன்னும் விளக்கை எரிய விடுகிறீர்கள்?"),
          ("KAMALA", "யாராவது இன்னும் வழி தவறியிருக்கலாம்.")]),
    ],
    "language_note": "All dialogue is spoken in Tamil (தமிழ்) with natural, fluent Tamil pronunciation. ",
}


# --------------------------------------------------------------------------- #
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def frame_prompt(spec, scene_desc: str) -> str:
    who = "; ".join(f"{n}: {d}" for n, d in spec["chars"].items())
    return f"{STYLE} Characters — {who}. Scene: {scene_desc} Both characters clearly visible, faces toward camera."


def gen_image(client, prompt: str, refs: list[bytes]) -> tuple[bytes, str]:
    contents: list = [prompt]
    for r in refs:
        contents.append(types.Part.from_bytes(data=r, mime_type=sniff_mime(r)))
    resp = client.models.generate_content(
        model=IMG,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="9:16", image_size="2K"),
        ),
    )
    for p in resp.candidates[0].content.parts:
        if p.inline_data and p.inline_data.data:
            return p.inline_data.data, p.inline_data.mime_type or sniff_mime(p.inline_data.data)
    raise RuntimeError("no image")


def sniff_mime(b: bytes) -> str:
    if b[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def dialogue_prompt(spec, scene_desc: str, lines, seconds: int) -> str:
    step = seconds / max(1, len(lines))
    timed = " ".join(
        f'[{round(i*step)}-{round((i+1)*step)}s] {spk} ({spec["voices"][spk]}) says: "{txt}"'
        for i, (spk, txt) in enumerate(lines)
    )
    return (
        "Single continuous shot, no scene cuts. Animate this exact frame as cinematic live-action "
        f"footage. {scene_desc} The characters speak their lines on camera with accurate lip-sync, "
        "natural acting and expressions. Speak the quoted lines EXACTLY as written, word for word. "
        f"{spec['language_note']}Dialogue: {timed} "
        "Sound design: howling wind and rain against glass, distant waves. "
        "No music. No narrator. No on-screen text or subtitles."
    )


def omni(client, *, inputs, seconds: int, store: bool, previous: str | None = None):
    body = dict(
        model=OMNI,
        input=inputs,
        response_format={
            "type": "video",
            "aspect_ratio": "9:16",
            "resolution": RES,
            "duration": f"{seconds}s",
            "delivery": "uri",
        },
        # URI delivery REQUIRES store=true (API: "store=true is required when response
        # format has video delivery set to URI").
        store=True,
    )
    if previous:
        body["previous_interaction_id"] = previous
    else:
        body["generation_config"] = {"video_config": {"task": "image_to_video"}}
    t = time.time()
    it = interactions_create(client, **body)
    return it, time.time() - t


def download_video(client, interaction, dest: Path) -> None:
    vid = interaction.output_video
    uri = getattr(vid, "uri", None)
    if uri:
        name = uri.split("/")[-1].split("?")[0].split(":")[0]
        for _ in range(120):
            info = client.files.get(name=f"files/{name}")
            state = getattr(getattr(info, "state", None), "name", str(getattr(info, "state", "")))
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise RuntimeError("file processing failed")
            time.sleep(3)
        client.files.download(file=uri, destination=str(dest))
    else:
        dest.write_bytes(b64bytes(vid.data))


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,duration",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    streams = json.loads(out).get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    return {
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "width": v.get("width"), "height": v.get("height"),
        "duration": float(v.get("duration") or 0),
    }


def extract_audio(path: Path) -> bytes:
    wav = path.with_suffix(".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True)
    return wav.read_bytes()


def contact_sheet(path: Path) -> Path:
    sheet = path.with_name(path.stem + "_sheet.jpg")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vf",
                    "fps=3/{d},scale=360:-1,tile=3x1".format(d=max(1, int(ffprobe(path)["duration"]) or 1)),
                    "-frames:v", "1", str(sheet)], check=True)
    return sheet


def transcribe(client, wav: bytes, model: str) -> str:
    inp = [{"type": "audio", "data": b64str(wav), "mime_type": "audio/wav"}]
    kw = {}
    if "transcribe" in model:
        kw["generation_config"] = {"transcription_config": {"mode": {"type": "verbatim", "diarization_mode": "speaker"}}}
    else:
        inp.append({"type": "text", "text": "Transcribe all speech verbatim in its original language and script, one line per speaker turn, prefixed 'Speaker N:'. Output only the transcript."})
    it = interactions_create(client, model=model, input=inp, **kw)
    return it.output_text or ""


def _norm_words(s: str) -> list[str]:
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"speaker\s*\d+\s*:", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return [w for w in s.split() if w]


def wer(ref: str, hyp: str) -> float:
    r, h = _norm_words(ref), _norm_words(hyp)
    if not r:
        return 0.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / len(r)


class ClipJudge(BaseModel):
    lip_sync: int = Field(description="1-5: mouths move in sync with the spoken words")
    acting: int = Field(description="1-5: natural expressions/gestures fitting the lines")
    lines_exact: int = Field(description="1-5: the quoted lines are spoken exactly (5 = word for word)")
    right_speakers: int = Field(description="1-5: each line is spoken by the correct character")
    frame_fidelity: int = Field(description="1-5: the video keeps the look of the starting frame/characters")
    audio_quality: int = Field(description="1-5: clean, intelligible speech; fitting ambience; no music")
    notes: str


class CharVoice(BaseModel):
    name: str
    consistency: int = Field(description="1-5 voice consistency across ALL clips")
    matches_description: int = Field(description="1-5 how well the voice matches its description")


class VoiceJudge(BaseModel):
    # NB: no dict fields — Gemini Developer API structured output rejects additionalProperties.
    per_character: list[CharVoice]
    notes: str


def judge_clip(client, mp4: Path, lines, voices) -> ClipJudge:
    script = "\n".join(f'{s} ({voices[s]}): "{t}"' for s, t in lines)
    resp = client.models.generate_content(
        model=JUDGE,
        contents=[
            types.Part.from_bytes(data=mp4.read_bytes(), mime_type="video/mp4"),
            "You are a strict film QA reviewer. Watch AND listen to this clip. The intended dialogue was:\n"
            f"{script}\nScore it. Be critical; 5 means flawless.",
        ],
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=ClipJudge,
                                           thinking_config=types.ThinkingConfig(thinking_level="medium")),
    )
    return ClipJudge.model_validate_json(resp.text)


def judge_voices(client, wavs: list[bytes], spec) -> VoiceJudge:
    parts: list = []
    for i, w in enumerate(wavs, 1):
        parts += [f"Clip {i}:", types.Part.from_bytes(data=w, mime_type="audio/wav")]
    desc = "\n".join(f"{k}: {v}" for k, v in spec["voices"].items())
    parts.append(
        "These clips are consecutive scenes of one short film with the same characters:\n"
        f"{desc}\nFor EACH character, does their voice sound like the SAME person across all clips? "
        "Score 1-5 (5 = clearly the same voice every time). Be strict."
    )
    resp = client.models.generate_content(
        model=JUDGE, contents=parts,
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=VoiceJudge,
                                           thinking_config=types.ThinkingConfig(thinking_level="medium")),
    )
    return VoiceJudge.model_validate_json(resp.text)


# --------------------------------------------------------------------------- #
def make_frames(client, spec, out: Path, tag: str, skip: bool) -> list[bytes]:
    frames: list[bytes] = []
    for i, (desc, _lines) in enumerate(spec["scenes"], 1):
        p = out / f"{tag}_frame_{i}.img"
        if skip and p.exists():
            frames.append(p.read_bytes())
            continue
        refs = frames[:1]  # scene 1 is the character reference for the rest
        prompt = frame_prompt(spec, desc) + (" Keep both characters IDENTICAL to the reference image." if refs else "")
        data, _ = gen_image(client, prompt, refs)
        p.write_bytes(data)
        frames.append(data)
        log(f"{tag} frame {i} ok ({len(data)//1024} KB)")
    return frames


def analyse(client, mp4: Path, lines, spec, lang: str, rec: dict) -> bytes:
    rec["probe"] = ffprobe(mp4)
    contact_sheet(mp4)
    wav = extract_audio(mp4) if rec["probe"]["has_audio"] else b""
    ref = " ".join(t for _, t in lines)
    if wav:
        models = ["gemini-3.5-transcribe"] + (["gemini-3.8-flash"] if lang == "ta" else [])
        rec["transcripts"] = {}
        rec["wer"] = {}
        for m in models:
            try:
                tx = transcribe(client, wav, m)
                rec["transcripts"][m] = tx
                rec["wer"][m] = round(wer(ref, tx), 3)
            except Exception as e:  # noqa: BLE001
                rec["transcripts"][m] = f"ERR {e}"
    try:
        rec["judge"] = judge_clip(client, mp4, lines, spec["voices"]).model_dump()
    except Exception as e:  # noqa: BLE001
        rec["judge"] = {"error": str(e)[:300]}
    return wav


def run_independent(client, spec, frames, out: Path, tag: str, lang: str, secs: int) -> dict:
    res = {"clips": []}
    wavs = []
    for i, ((desc, lines), frame) in enumerate(zip(spec["scenes"], frames), 1):
        rec: dict = {"scene": i, "lines": lines}
        prompt = dialogue_prompt(spec, desc, lines, secs)
        try:
            it, dt = omni(client, inputs=[{"type": "image", "data": b64str(frame), "mime_type": sniff_mime(frame)},
                                         {"type": "text", "text": prompt}], seconds=secs, store=False)
            rec["latency_s"] = round(dt, 1)
            mp4 = out / f"{tag}_clip_{i}.mp4"
            download_video(client, it, mp4)
            wavs.append(analyse(client, mp4, lines, spec, lang, rec))
            log(f"{tag} clip {i}: {rec.get('probe')} wer={rec.get('wer')} judge={ {k: v for k, v in rec.get('judge', {}).items() if k != 'notes'} }")
        except Exception as e:  # noqa: BLE001
            rec["error"] = str(e)[:500]
            log(f"{tag} clip {i} ERROR {rec['error']}")
        res["clips"].append(rec)
    wavs = [w for w in wavs if w]
    if len(wavs) >= 2:
        try:
            res["voice_consistency"] = judge_voices(client, wavs, spec).model_dump()
        except Exception as e:  # noqa: BLE001
            res["voice_consistency"] = {"error": str(e)[:300]}
    return res


def run_chain(client, spec, frames, out: Path, tag: str, secs: int) -> dict:
    res = {"clips": [], "extension_accepts_image": None}
    prev = None
    wavs = []
    for i, ((desc, lines), frame) in enumerate(zip(spec["scenes"], frames), 1):
        rec: dict = {"scene": i, "lines": lines}
        prompt = dialogue_prompt(spec, desc, lines, secs)
        try:
            if prev is None:
                inputs = [{"type": "image", "data": b64str(frame), "mime_type": sniff_mime(frame)}, {"type": "text", "text": prompt}]
                it, dt = omni(client, inputs=inputs, seconds=secs, store=True)
            else:
                cont = "Continue the film with a cut to the next scene. " + prompt.replace("Single continuous shot, no scene cuts. Animate this exact frame as cinematic live-action footage. ", "")
                try:
                    inputs = [{"type": "image", "data": b64str(frame), "mime_type": sniff_mime(frame)}, {"type": "text", "text": cont}]
                    it, dt = omni(client, inputs=inputs, seconds=secs, store=True, previous=prev)
                    if res["extension_accepts_image"] is None:
                        res["extension_accepts_image"] = True
                except Exception as e:  # noqa: BLE001
                    log(f"{tag} extension with image rejected ({str(e)[:160]}); retrying text-only")
                    res["extension_accepts_image"] = False
                    it, dt = omni(client, inputs=cont, seconds=secs, store=True, previous=prev)
            prev = it.id
            rec["latency_s"] = round(dt, 1)
            mp4 = out / f"{tag}_clip_{i}.mp4"
            download_video(client, it, mp4)
            cum_lines = [ln for _d, ls in spec["scenes"][:i] for ln in ls]  # extensions return the CUMULATIVE video
            wavs.append(analyse(client, mp4, cum_lines, spec, "en", rec))
            log(f"{tag} clip {i}: {rec.get('probe')} wer={rec.get('wer')} judge={ {k: v for k, v in rec.get('judge', {}).items() if k != 'notes'} }")
        except Exception as e:  # noqa: BLE001
            rec["error"] = str(e)[:500]
            log(f"{tag} clip {i} ERROR {rec['error']}")
            res["clips"].append(rec)
            break
        res["clips"].append(rec)
    wavs = [w for w in wavs if w]
    if len(wavs) >= 2:
        try:
            res["voice_consistency"] = judge_voices(client, wavs, spec).model_dump()
        except Exception as e:  # noqa: BLE001
            res["voice_consistency"] = {"error": str(e)[:300]}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-images", action="store_true", help="reuse frames already in --out")
    ap.add_argument("--only", default="en_a,en_b,ta")
    ap.add_argument("--seconds", type=int, default=7)
    ap.add_argument("--rejudge-voices", default="", help="variant tag whose saved clip wavs to re-judge (e.g. en_b)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = get_settings()
    if s.mock_mode:
        print("GEMINI_API_KEY not set — G1 is a live probe.", file=sys.stderr)
        return 2
    client = make_client(s, timeout_seconds=900)
    only = set(x for x in args.only.split(",") if x)
    rp = out / "results.json"
    results: dict = json.loads(rp.read_text()) if rp.exists() else {}
    results.update({"omni_model": OMNI, "resolution": RES, "seconds": args.seconds})
    if args.rejudge_voices:
        tag = args.rejudge_voices
        spec = TA if tag.startswith("ta") else EN
        wavs = [p.read_bytes() for p in sorted(out.glob(f"{tag}_clip_*.wav"))]
        if tag == "en_b" and wavs:
            wavs = wavs[-1:]  # cumulative chain: the final clip contains every scene
        vj = judge_voices(client, wavs, spec).model_dump()
        results.setdefault(tag, {})["voice_consistency"] = vj
        log("voices", tag, vj)
    if only & {"en_a", "en_b"}:
        en_frames = make_frames(client, EN, out, "en", args.skip_images)
        if "en_a" in only:
            results["en_a"] = run_independent(client, EN, en_frames, out, "en_a", "en", args.seconds)
            (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
        if "en_b" in only:
            results["en_b"] = run_chain(client, EN, en_frames, out, "en_b", args.seconds)
            (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    if "ta" in only:
        ta_frames = make_frames(client, TA, out, "ta", args.skip_images)
        results["ta"] = run_independent(client, TA, ta_frames, out, "ta", "ta", args.seconds)
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    log("done ->", out / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
