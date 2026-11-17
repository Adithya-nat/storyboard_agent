"""Make a finished film ready for the homepage library.

A job's files are deleted 24 hours after it starts, so publishing copies what the homepage
needs into the library folder (outside the sweep): a web encode of the film, a 12-second
preview, a still, the film's frames (so the creator can re-pick the poster frame later)
and the poster itself.

The poster is the film's own frame with movie-title lettering set on top by the image
model, the same way a designer would turn a still into a one-sheet.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.adapters.gemini import GeminiImageAdapter
from app.adapters.video import probe_duration
from app.core.config import Settings
from app.core.logging import get_logger
from app.prompts import render

log = get_logger(__name__)

POSTER_SIZE = (720, 1290)  # what the homepage cards are laid out for
FRAME_SIZE = (540, 960)
PREVIEW_SECONDS = 12
MEDIA_FILES = {"poster.jpg", "frame.jpg", "preview.mp4", "film.mp4"}
_FRAME_FILE = re.compile(r"^frames/(\d{1,3})\.jpg$")


def film_dir(settings: Settings, film_id: str) -> Path:
    return settings.library_dir / film_id


def media_path(settings: Settings, film_id: str, name: str) -> Path | None:
    """The file for a public media name (or ``frames/N.jpg``), or None for anything else."""
    if name in MEDIA_FILES or _FRAME_FILE.match(name):
        return film_dir(settings, film_id) / name
    return None


def frame_orders(settings: Settings, film_id: str) -> list[int]:
    d = film_dir(settings, film_id) / "frames"
    if not d.exists():
        return []
    return sorted(int(p.stem) for p in d.glob("*.jpg") if p.stem.isdigit())


# --------------------------------------------------------------------------- #
# Looks
# --------------------------------------------------------------------------- #
def looks() -> dict[str, str]:
    """Poster looks from prompts/library/looks.md: {id: instruction}."""
    out: dict[str, str] = {}
    for block in render("library/looks").split("## ")[1:]:
        key, _, body = block.partition("\n")
        out[key.strip()] = " ".join(body.split())
    return out


LOOK_IDS = tuple(looks())


# --------------------------------------------------------------------------- #
# Facts about the film, read from the job's artifacts
# --------------------------------------------------------------------------- #
@dataclass
class FilmFacts:
    title: str
    native_title: str | None
    logline: str
    genre: str
    tone: str
    language: str
    style: str
    scenes: int
    cover_order: int
    frames: dict[int, str]  # order -> path


def _read_json(path: str | None) -> dict:
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _genre_from(fmt: str) -> str:
    g = " ".join((fmt or "").split())
    for word in ("trailer", "short film", "short"):
        g = re.sub(rf"\b{word}\b", "", g, flags=re.I).strip(" ,-")
    g = g[:1].upper() + g[1:] if g else ""
    return g[:48]


def film_facts(job) -> FilmFacts:
    by_kind: dict[str, str] = {}
    frames: dict[int, str] = {}
    for a in job.artifacts:
        if a.kind == "frame" and a.order_index is not None and a.status == "ok":
            frames[int(a.order_index)] = a.path
        else:
            by_kind.setdefault(a.kind, a.path)
    script = _read_json(by_kind.get("script"))
    publish = _read_json(by_kind.get("publish"))
    root = Path(by_kind["video"]).parent if by_kind.get("video") else None
    brief = _read_json(str(root / "brief.json")) if root else {}
    gloss = _read_json(str(root / "gloss.json")) if root else {}

    language = script.get("language") or job.language or "en"
    raw_title = script.get("title") or publish.get("title") or "Untitled"
    if language == "en":
        title, native = raw_title, None
    else:
        title, native = gloss.get(raw_title) or raw_title, raw_title
    logline = brief.get("logline") or gloss.get(script.get("logline", "")) or script.get("logline", "")
    scenes = script.get("scenes") or []
    cover = int(publish.get("cover_order") or 0) or (min(frames) if frames else 1)
    return FilmFacts(
        title=title[:120],
        native_title=native[:160] if native else None,
        logline=logline,
        genre=_genre_from(brief.get("format", "")),
        tone=brief.get("purpose", "") or script.get("style", ""),
        language=language,
        style=script.get("story_style") or job.style,
        scenes=len(scenes) or len(frames),
        cover_order=cover if cover in frames or not frames else min(frames),
        frames=frames,
    )


def made_minutes(stages) -> int:
    """Wall-clock render time: the sum of stage run times (pauses don't count)."""
    secs = 0.0
    for st in stages:
        if st.started_at and st.ended_at:
            secs += max(0.0, (st.ended_at - st.started_at).total_seconds())
    return max(1, round(secs / 60)) if secs else 0


def slugify(title: str) -> str:
    s = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s[:48].strip("-") or "film"


# --------------------------------------------------------------------------- #
# Media
# --------------------------------------------------------------------------- #
def _fit(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Cover-crop to exactly ``size`` (centre)."""
    w, h = size
    scale = max(w / img.width, h / img.height)
    img = img.convert("RGB").resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    left, top = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((left, top, left + w, top + h))


def save_jpeg(data_or_path: bytes | str | Path, dest: Path, size: tuple[int, int], quality: int = 86) -> None:
    src = Image.open(io.BytesIO(data_or_path)) if isinstance(data_or_path, bytes) else Image.open(data_or_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.jpg")
    _fit(src, size).save(tmp, "JPEG", quality=quality, optimize=True, progressive=True)
    tmp.replace(dest)


async def _ffmpeg(*args: str) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-v", "error", "-y", *args,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        log.warning("library encode failed: %s", err.decode(errors="replace")[-400:])
    return proc.returncode == 0


_H264 = ("-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-preset", "medium", "-movflags", "+faststart")


async def prepare_media(facts: FilmFacts, video_path: str, dest: Path) -> int:
    """Web encodes + still + frames into ``dest``. Returns the film length in seconds."""
    dest.mkdir(parents=True, exist_ok=True)
    length = await probe_duration(video_path) or 0.0

    film = dest / "film.mp4"
    if not await _ffmpeg("-i", video_path, "-vf", "scale=720:1280:flags=lanczos", *_H264,
                         "-crf", "26", "-maxrate", "1400k", "-bufsize", "2800k",
                         "-c:a", "aac", "-b:a", "128k", str(film)):
        shutil.copyfile(video_path, film)

    # The preview starts on the poster scene, like a trailer cut to the money shot.
    start = 0.0
    if length > PREVIEW_SECONDS and facts.scenes:
        start = (facts.cover_order - 1) / facts.scenes * length
        start = max(0.0, min(start, length - PREVIEW_SECONDS))
    if not await _ffmpeg("-ss", f"{start:.2f}", "-i", video_path, "-t", str(PREVIEW_SECONDS),
                         "-vf", "scale=540:960:flags=lanczos", *_H264, "-crf", "26",
                         "-maxrate", "900k", "-bufsize", "1800k", "-c:a", "aac", "-b:a", "96k",
                         str(dest / "preview.mp4")):
        shutil.copyfile(film, dest / "preview.mp4")

    for order, path in facts.frames.items():
        await asyncio.to_thread(save_jpeg, path, dest / "frames" / f"{order}.jpg", (720, 1280), 88)
    cover = dest / "frames" / f"{facts.cover_order}.jpg"
    if cover.exists():
        await asyncio.to_thread(save_jpeg, cover, dest / "frame.jpg", FRAME_SIZE, 84)
    return round(length)


# --------------------------------------------------------------------------- #
# Poster
# --------------------------------------------------------------------------- #
def poster_prompt(*, title: str, native_title: str | None, tagline: str | None, genre: str,
                  tone: str, logline: str, look: str) -> str:
    native_line = (
        f'- Under or beside it, the Tamil title "{native_title}" as a secondary line in a matching Tamil typeface.'
        if native_title else ""
    )
    tagline_rule = (
        f'Tagline: "{tagline}", small, where a tagline sits on a real poster.'
        if tagline else "One short tagline you write for this film (under eight words, English), small, where a tagline sits on a real poster."
    )
    return render(
        "library/poster",
        title=title, native_line=native_line, tagline_rule=tagline_rule,
        genre=genre or "drama", tone=tone or "cinematic", logline=logline or title,
        look=looks().get(look) or looks()["auto"],
    )


async def make_poster(settings: Settings, frame_jpg: Path, prompt: str, dest: Path) -> None:
    """One image call: the frame in, the poster out (saved as ``dest``)."""
    adapter = GeminiImageAdapter(settings)
    data = frame_jpg.read_bytes()
    out = await asyncio.to_thread(adapter.generate_image, prompt, refs=[("Frame from the film", data)])
    await asyncio.to_thread(save_jpeg, out, dest, POSTER_SIZE, 88)
