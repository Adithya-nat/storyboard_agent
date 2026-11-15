"""GET /storyboard/{id}/live: the assets a run has produced SO FAR.

Feeds the live generation UI (the "production line"): the real title and beats
once the script exists, the judge's scores, the look reference and cast, each
frame as it renders, audio tracks, then the cover. Everything is optional and
read defensively, so it works at any point mid-run, on any schema revision of
the JSON files, and on finished jobs too. Read-only; never blocks the pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes import _artifact_url, _load_live_job
from app.artifacts.rows import assets_version
from app.core.config import get_settings
from app.db.session import get_session

router = APIRouter(prefix="/storyboard", tags=["storyboard"])


class LiveBeat(BaseModel):
    order: int
    text: str
    speaker: str | None = None


class LiveCast(BaseModel):
    name: str
    portrait_url: str | None = None


class LiveMedia(BaseModel):
    order: int
    url: str
    status: str = "ok"


class LiveBrief(BaseModel):
    format: str = ""
    hook: str = ""
    logline: str = ""


class LiveReview(BaseModel):
    score: int | None = None
    retelling: str = ""


class LiveAssets(BaseModel):
    job_id: str
    # Changes whenever the artifact set changes; the SSE ``assets`` event carries
    # the same value so clients refetch only when something new landed.
    version: str
    title: str | None = None
    logline: str | None = None
    beats: list[LiveBeat] = Field(default_factory=list)
    scores: dict[str, int] = Field(default_factory=dict)
    score_overall: int | None = None
    style_ref_url: str | None = None
    cast: list[LiveCast] = Field(default_factory=list)
    frames: list[LiveMedia] = Field(default_factory=list)
    clips: list[LiveMedia] = Field(default_factory=list)
    narration_url: str | None = None
    music_url: str | None = None
    cover_url: str | None = None
    caption: str | None = None
    # Creative brief + test-screening review, when this pipeline has those
    # stages (brief.json / film_review.json in the job folder); else null.
    brief: LiveBrief | None = None
    review: LiveReview | None = None


def _read_json(art) -> dict:
    if art is None or not os.path.exists(art.path):
        return {}
    try:
        data = json.loads(Path(art.path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _beats(script: dict) -> list[LiveBeat]:
    out: list[LiveBeat] = []
    for sc in script.get("scenes") or []:
        if not isinstance(sc, dict) or not isinstance(sc.get("order"), int):
            continue
        text, speaker = (sc.get("narration") or "").strip(), None
        if not text:
            for ln in sc.get("lines") or []:
                if isinstance(ln, dict) and (ln.get("text") or "").strip():
                    text, speaker = ln["text"].strip(), ln.get("speaker") or None
                    break
        if not text:
            text = (sc.get("description") or "").strip()
        if text:
            out.append(LiveBeat(order=sc["order"], text=text[:240], speaker=speaker))
    return sorted(out, key=lambda b: b.order)


def _scores(reflection: dict) -> tuple[dict[str, int], int | None]:
    report = reflection.get("report") if isinstance(reflection.get("report"), dict) else {}
    # Any 0..100 integer dimension, whatever this revision of the judge calls
    # them; zero means "not scored for this film" (e.g. dialogue on narration).
    scores = {
        k: v
        for k, v in report.items()
        if k != "overall" and isinstance(v, int) and not isinstance(v, bool) and 0 < v <= 100
    }
    overall = reflection.get("final_overall", report.get("overall"))
    return scores, overall if isinstance(overall, int) and not isinstance(overall, bool) else None


def _brief(raw: dict) -> LiveBrief | None:
    fields = {k: raw[k].strip() for k in ("format", "hook", "logline") if isinstance(raw.get(k), str)}
    return LiveBrief(**fields) if any(fields.values()) else None


def _review(raw: dict) -> LiveReview | None:
    score = raw.get("score")
    score = score if isinstance(score, int) and not isinstance(score, bool) else None
    retelling = raw.get("retelling") if isinstance(raw.get("retelling"), str) else ""
    return LiveReview(score=score, retelling=retelling.strip()) if score is not None or retelling else None


def _caption(kit: dict) -> str | None:
    for key in ("hook", "title"):
        if isinstance(kit.get(key), str) and kit[key].strip():
            return kit[key].strip()
    return None


@router.get("/{job_id}/live", response_model=LiveAssets)
async def get_live_assets(job_id: str, session: AsyncSession = Depends(get_session)) -> LiveAssets:
    job = await _load_live_job(session, job_id)
    arts = list(job.artifacts)
    by_kind = {a.kind: a for a in arts}

    def url(kind: str) -> str | None:
        a = by_kind.get(kind)
        return _artifact_url(job_id, a.id) if a else None

    def media(kind: str) -> list[LiveMedia]:
        rows = {
            a.order_index: a for a in arts if a.kind == kind and a.order_index is not None
        }
        return [
            LiveMedia(order=o, url=_artifact_url(job_id, a.id), status=a.status)
            for o, a in sorted(rows.items())
        ]

    script = _read_json(by_kind.get("script"))
    job_dir = Path(get_settings().artifact_root) / job_id

    def job_json(name: str) -> dict:
        # Stage outputs that aren't artifact rows (brief, film review) live in
        # the job folder; read them the same defensive way.
        return _read_json(SimpleNamespace(path=str(job_dir / name)))

    scores, overall = _scores(_read_json(by_kind.get("reflection")))
    portraits = {a.kind.split(":", 1)[1]: a for a in arts if a.kind.startswith("cast:")}
    cast = []
    for c in script.get("characters") or []:
        name = (c.get("name") or "") if isinstance(c, dict) else ""
        if not name:
            continue
        p = portraits.get(name.upper()[:26])
        cast.append(LiveCast(name=name, portrait_url=_artifact_url(job_id, p.id) if p else None))

    return LiveAssets(
        job_id=job_id,
        version=assets_version(arts),
        title=(script.get("title") or None),
        logline=(script.get("logline") or None),
        beats=_beats(script),
        scores=scores,
        score_overall=overall,
        style_ref_url=url("style_ref"),
        cast=cast,
        frames=media("frame"),
        clips=media("motion"),
        narration_url=url("narration"),
        music_url=url("music"),
        cover_url=url("cover"),
        caption=_caption(_read_json(by_kind.get("publish"))),
        brief=_brief(job_json("brief.json")),
        review=_review(job_json("film_review.json")),
    )
