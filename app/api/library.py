"""Film library: creators publish finished films to the homepage.

    GET    /library/public                -> published films (no sign-in; the homepage reads it)
    GET    /library/public/{id}/{file}    -> poster.jpg | frame.jpg | preview.mp4 | film.mp4
    GET    /library/draft/{job_id}        -> form defaults + this job's library entry, if any
    POST   /library                       -> 202 copy the film into the library + make a poster
    GET    /library/{id}                  -> the creator's entry (status, poster version, frames)
    PATCH  /library/{id}                  -> edit title / logline / genre / tagline / prompt shown
    POST   /library/{id}/poster           -> 202 a new poster (another look or frame)
    POST   /library/{id}/publish          -> show it on the homepage
    POST   /library/{id}/unpublish        -> take it off the homepage (keeps the draft)
    DELETE /library/{id}                  -> remove it and its files
    GET    /library/{id}/media/{file}     -> the creator's own files, drafts included (?t= works)
"""

from __future__ import annotations

import asyncio
import secrets
import shutil
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes import _load_owned_job, _request_key
from app.core.auth import User, allowed_user, media_user
from app.core.config import get_settings
from app.core.keys import bind_key, release_key
from app.core.logging import get_logger
from app.db import repository as repo
from app.db.models import LibraryFilm
from app.db.session import get_session, get_sessionmaker
from app.library import service
from app.observability.context import bind_stage, reset_stage
from app.observability.usage import UsageCollector, bind_collector, reset_collector

log = get_logger(__name__)
router = APIRouter(prefix="/library", tags=["library"])

Look = Literal[service.LOOK_IDS]  # type: ignore[valid-type]
Status = Literal["rendering", "draft", "published", "failed"]


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class LibraryMedia(BaseModel):
    poster: str
    frame: str
    preview: str
    film: str


class PublicFilm(BaseModel):
    id: str
    title: str
    native_title: str | None = None
    tagline: str | None = None
    logline: str
    genre: str
    prompt: str | None = None
    language: str
    style: str
    length_seconds: int
    scenes: int
    made_minutes: int
    cost_usd: float
    published_at: datetime | None = None
    media: LibraryMedia


class LibraryEntry(PublicFilm):
    job_id: str
    status: Status
    error: str | None = None
    poster_look: str
    poster_frame: int
    poster_version: int
    posters_left: int
    frames: list[int] = []


class LibraryDetails(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    native_title: str | None = Field(default=None, max_length=160)
    tagline: str | None = Field(default=None, max_length=160)
    logline: str = Field(default="", max_length=600)
    genre: str = Field(default="", max_length=48)
    show_prompt: bool = True


class LibraryCreate(LibraryDetails):
    job_id: str
    look: Look = "auto"  # type: ignore[valid-type]
    frame: int | None = None  # scene order; None = the film's cover frame


class LibraryEdit(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    native_title: str | None = Field(default=None, max_length=160)
    tagline: str | None = Field(default=None, max_length=160)
    logline: str | None = Field(default=None, max_length=600)
    genre: str | None = Field(default=None, max_length=48)
    show_prompt: bool | None = None


class PosterRequest(BaseModel):
    look: Look = "auto"  # type: ignore[valid-type]
    frame: int | None = None
    tagline: str | None = Field(default=None, max_length=160)


class LibraryDraft(BaseModel):
    """Form defaults read from the film, plus the existing entry if it was published before."""

    title: str
    native_title: str | None = None
    logline: str
    genre: str
    prompt: str
    language: str
    cover_frame: int
    frames: list[int]
    looks: list[str]
    entry: LibraryEntry | None = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _media(base: str, version: int) -> LibraryMedia:
    return LibraryMedia(
        poster=f"{base}/poster.jpg?v={version}",
        frame=f"{base}/frame.jpg",
        preview=f"{base}/preview.mp4",
        film=f"{base}/film.mp4",
    )


def _public(f: LibraryFilm) -> PublicFilm:
    return PublicFilm(
        id=f.id, title=f.title, native_title=f.native_title, tagline=f.tagline, logline=f.logline,
        genre=f.genre, prompt=f.prompt, language=f.language, style=f.style,
        length_seconds=f.length_seconds, scenes=f.scenes, made_minutes=f.made_minutes,
        cost_usd=round(f.cost_usd, 2), published_at=f.published_at,
        media=_media(f"/library/public/{f.id}", f.poster_version),
    )


def _entry(f: LibraryFilm) -> LibraryEntry:
    settings = get_settings()
    pub = _public(f).model_dump()
    pub["media"] = _media(f"/library/{f.id}/media", f.poster_version)
    return LibraryEntry(
        **pub, job_id=f.job_id, status=f.status, error=f.error, poster_look=f.poster_look,
        poster_frame=f.poster_frame, poster_version=f.poster_version,
        posters_left=max(0, settings.library_poster_limit - f.posters_made),
        frames=service.frame_orders(settings, f.id),
    )


def _owns(user: User, f: LibraryFilm) -> bool:
    return user.scope is None or f.user_id == user.scope


async def _owned(session: AsyncSession, film_id: str, user: User) -> LibraryFilm:
    f = await session.get(LibraryFilm, film_id)
    if f is None or not _owns(user, f):
        raise HTTPException(status_code=404, detail="film not found")
    return f


async def _for_job(session: AsyncSession, job_id: str, user: User) -> LibraryFilm | None:
    q = select(LibraryFilm).where(LibraryFilm.job_id == job_id)
    if user.scope is not None:
        q = q.where(LibraryFilm.user_id == user.scope)
    return (await session.execute(q.limit(1))).scalar_one_or_none()


async def _unique_id(session: AsyncSession, title: str) -> str:
    base = service.slugify(title)
    if await session.get(LibraryFilm, base) is None:
        return base
    while True:
        cand = f"{base}-{secrets.token_hex(2)}"
        if await session.get(LibraryFilm, cand) is None:
            return cand


def _check_poster_budget(f: LibraryFilm) -> None:
    if f.posters_made >= get_settings().library_poster_limit:
        raise HTTPException(status_code=429, detail="This film has used all its poster tries.")


def _cors(response: Response) -> None:
    """The homepage lives on another origin (GitHub Pages) and only reads published films."""
    response.headers["Access-Control-Allow-Origin"] = "*"


# Background work holds one lock per film so two poster requests can't race.
_locks: dict[str, asyncio.Lock] = {}


async def _run_poster_job(film_id: str, api_key: str | None, *, with_media: bool, video_path: str | None,
                          facts: service.FilmFacts | None) -> None:
    settings = get_settings()
    if api_key:
        settings = settings.model_copy(update={"gemini_api_key": api_key})
    key_token = bind_key(api_key)
    lock = _locks.setdefault(film_id, asyncio.Lock())
    collector = UsageCollector()
    c_token = bind_collector(collector)
    s_token = None
    step = "media" if with_media else "poster"
    sm = get_sessionmaker()
    try:
        async with lock:
            async with sm() as session:
                f = await session.get(LibraryFilm, film_id)
                if f is None:
                    return
                s_token = bind_stage(f.job_id, "library_poster")
                dest = service.film_dir(settings, film_id)
                if with_media and facts is not None and video_path:
                    f.length_seconds = await service.prepare_media(facts, video_path, dest)
                    await session.commit()
                step = "poster"
                frame = dest / "frames" / f"{f.poster_frame}.jpg"
                if not frame.exists():
                    raise RuntimeError("the poster frame is missing")
                prompt = service.poster_prompt(
                    title=f.title, native_title=f.native_title, tagline=f.tagline, genre=f.genre,
                    tone=facts.tone if facts else f.genre, logline=f.logline, look=f.poster_look,
                )
                await service.make_poster(settings, frame, prompt, dest / "poster.jpg")
                f.poster_version += 1
                f.posters_made += 1
                f.status = "published" if f.published_at else "draft"
                f.error = None
                await session.commit()
    except Exception as exc:  # noqa: BLE001 - surfaced to the creator on the entry
        log.warning("library poster failed for %s: %s", film_id, exc)
        async with sm() as session:
            f = await session.get(LibraryFilm, film_id)
            if f is not None:
                has_poster = (service.film_dir(settings, film_id) / "poster.jpg").exists()
                f.status = ("published" if f.published_at else "draft") if has_poster else "failed"
                f.error = ("The poster could not be made. Try another look or frame." if step == "poster"
                           else "The film could not be copied to the library. Remove it and try again.")
                await session.commit()
    finally:
        events = collector.drain_events()
        if events:
            try:
                async with sm() as session:
                    f = await session.get(LibraryFilm, film_id)
                    if f is not None:
                        await repo.add_spend_events(session, f.job_id, events)
            except Exception:  # noqa: BLE001
                log.warning("could not record poster spend", exc_info=True)
        if s_token is not None:
            reset_stage(s_token)
        reset_collector(c_token)
        release_key(api_key, key_token)


# --------------------------------------------------------------------------- #
# Public (the homepage)
# --------------------------------------------------------------------------- #
@router.get("/public", response_model=list[PublicFilm])
async def public_films(response: Response, session: AsyncSession = Depends(get_session)) -> list[PublicFilm]:
    _cors(response)
    response.headers["Cache-Control"] = "public, max-age=60"
    rows = await session.execute(
        select(LibraryFilm).where(LibraryFilm.status == "published").order_by(LibraryFilm.published_at.desc()).limit(60)
    )
    return [_public(f) for f in rows.scalars()]


@router.get("/public/{film_id}/{name:path}")
async def public_media(film_id: str, name: str, session: AsyncSession = Depends(get_session)) -> FileResponse:
    f = await session.get(LibraryFilm, film_id)
    path = service.media_path(get_settings(), film_id, name)
    if f is None or f.status != "published" or path is None or name.startswith("frames/") or not path.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "public, max-age=300"})


# --------------------------------------------------------------------------- #
# The creator's side
# --------------------------------------------------------------------------- #
@router.get("/draft/{job_id}", response_model=LibraryDraft)
async def library_draft(job_id: str, session: AsyncSession = Depends(get_session),
                        user: User = Depends(allowed_user)) -> LibraryDraft:
    existing = await _for_job(session, job_id, user)
    job = await _load_owned_job(session, job_id, user)
    facts = service.film_facts(job)
    return LibraryDraft(
        title=facts.title, native_title=facts.native_title, logline=facts.logline, genre=facts.genre,
        prompt=job.prompt, language=facts.language, cover_frame=facts.cover_order,
        frames=sorted(facts.frames), looks=list(service.LOOK_IDS),
        entry=_entry(existing) if existing else None,
    )


@router.post("", status_code=202, response_model=LibraryEntry)
async def create_entry(request: Request, body: LibraryCreate, background: BackgroundTasks,
                       session: AsyncSession = Depends(get_session),
                       user: User = Depends(allowed_user)) -> LibraryEntry:
    job = await _load_owned_job(session, body.job_id, user)
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="Only finished films can go in the library.")
    video = next((a for a in job.artifacts if a.kind == "video"), None)
    if video is None:
        raise HTTPException(status_code=409, detail="This film has no video yet.")
    if await _for_job(session, body.job_id, user) is not None:
        raise HTTPException(status_code=409, detail="This film is already in the library.")
    api_key = _request_key(request)
    facts = service.film_facts(job)
    frame = body.frame if body.frame in facts.frames else facts.cover_order
    stages = await repo.get_stages(session, job.id)
    f = LibraryFilm(
        id=await _unique_id(session, body.title), job_id=job.id, user_id=user.scope, status="rendering",
        title=body.title.strip(), native_title=(body.native_title or "").strip() or None,
        tagline=(body.tagline or "").strip() or None, logline=body.logline.strip(),
        genre=body.genre.strip(), prompt=job.prompt if body.show_prompt else None,
        language=facts.language, style=facts.style, scenes=facts.scenes,
        made_minutes=service.made_minutes(stages), cost_usd=float(job.cost_usd or 0.0),
        poster_look=body.look, poster_frame=frame, created_at=datetime.now(timezone.utc),
    )
    session.add(f)
    await session.commit()
    background.add_task(_run_poster_job, f.id, api_key, with_media=True, video_path=video.path, facts=facts)
    return _entry(f)


@router.get("/{film_id}", response_model=LibraryEntry)
async def get_entry(film_id: str, session: AsyncSession = Depends(get_session),
                    user: User = Depends(allowed_user)) -> LibraryEntry:
    return _entry(await _owned(session, film_id, user))


@router.patch("/{film_id}", response_model=LibraryEntry)
async def edit_entry(film_id: str, body: LibraryEdit, session: AsyncSession = Depends(get_session),
                     user: User = Depends(allowed_user)) -> LibraryEntry:
    f = await _owned(session, film_id, user)
    for field in ("title", "logline", "genre"):
        v = getattr(body, field)
        if v is not None:
            setattr(f, field, v.strip())
    for field in ("native_title", "tagline"):
        if field in body.model_fields_set:
            setattr(f, field, (getattr(body, field) or "").strip() or None)
    if body.show_prompt is not None:
        if body.show_prompt and f.prompt is None:
            job = await repo.get_job(session, f.job_id)
            if job is None:
                raise HTTPException(status_code=410, detail="The prompt was deleted with the film's files after 24 hours.")
            f.prompt = job.prompt
        elif not body.show_prompt:
            f.prompt = None
    await session.commit()
    return _entry(f)


@router.post("/{film_id}/poster", status_code=202, response_model=LibraryEntry)
async def new_poster(film_id: str, request: Request, body: PosterRequest, background: BackgroundTasks,
                     session: AsyncSession = Depends(get_session),
                     user: User = Depends(allowed_user)) -> LibraryEntry:
    f = await _owned(session, film_id, user)
    if f.status == "rendering":
        raise HTTPException(status_code=409, detail="A poster is already being made.")
    if f.status == "failed" and not service.frame_orders(get_settings(), f.id):
        raise HTTPException(status_code=409, detail="This film's files didn't copy. Remove it and publish again.")
    _check_poster_budget(f)
    api_key = _request_key(request)
    frames = service.frame_orders(get_settings(), f.id)
    if body.frame is not None and body.frame in frames:
        f.poster_frame = body.frame
    f.poster_look = body.look
    if "tagline" in body.model_fields_set:
        f.tagline = (body.tagline or "").strip() or None
    f.status = "rendering"
    f.error = None
    await session.commit()
    background.add_task(_run_poster_job, f.id, api_key, with_media=False, video_path=None, facts=None)
    return _entry(f)


@router.post("/{film_id}/publish", response_model=LibraryEntry)
async def publish(film_id: str, session: AsyncSession = Depends(get_session),
                  user: User = Depends(allowed_user)) -> LibraryEntry:
    f = await _owned(session, film_id, user)
    if f.status not in ("draft", "published"):
        raise HTTPException(status_code=409, detail="Wait for the poster before publishing.")
    f.status = "published"
    f.published_at = f.published_at or datetime.now(timezone.utc)
    await session.commit()
    return _entry(f)


@router.post("/{film_id}/unpublish", response_model=LibraryEntry)
async def unpublish(film_id: str, session: AsyncSession = Depends(get_session),
                    user: User = Depends(allowed_user)) -> LibraryEntry:
    f = await _owned(session, film_id, user)
    if f.status == "published":
        f.status = "draft"
    f.published_at = None
    await session.commit()
    return _entry(f)


@router.delete("/{film_id}", status_code=204)
async def delete_entry(film_id: str, session: AsyncSession = Depends(get_session),
                       user: User = Depends(allowed_user)) -> Response:
    f = await _owned(session, film_id, user)
    if f.status == "rendering":
        raise HTTPException(status_code=409, detail="Wait for the poster to finish first.")
    await session.delete(f)
    await session.commit()
    shutil.rmtree(service.film_dir(get_settings(), film_id), ignore_errors=True)
    _locks.pop(film_id, None)
    return Response(status_code=204)


@router.get("/{film_id}/media/{name:path}")
async def owner_media(film_id: str, name: str, session: AsyncSession = Depends(get_session),
                      user: User = Depends(media_user)) -> FileResponse:
    f = await _owned(session, film_id, user)
    path = service.media_path(get_settings(), f.id, name)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, headers={"Cache-Control": "private, max-age=60"})
