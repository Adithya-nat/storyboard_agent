"""film_review agent: a model watches the FINISHED film as a first-time viewer and scores coherence.

This is the measure of "feels like a human-made film" (the script judge only reads the page).
It informs the ledger and evals; it never fails or re-renders the film.
"""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.graph.state import Deps, StoryboardState
from app.observability import usage
from app.prompts import render
from app.schemas import LANGUAGE_NAMES

log = get_logger(__name__)


class FilmReview(BaseModel):
    retelling: str = ""
    who_wants_what: str = ""
    what_changed: str = ""
    coherence: int = Field(default=0, description="0-100")
    continuity: int = Field(default=0, description="0-100")
    dialogue: int = Field(default=0, description="0-100")
    hook: int = Field(default=0, description="0-100")
    human_made: int = Field(default=0, description="0-100")
    confusing_moments: list[str] = Field(default_factory=list)

    @property
    def score(self) -> int:
        """One number for the ledger: coherence and continuity carry the 'random clips' complaint."""
        return round(0.35 * self.coherence + 0.25 * self.continuity + 0.15 * self.dialogue
                     + 0.10 * self.hook + 0.15 * self.human_made)


def _small(video: str) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "review.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video, "-vf", "scale=-2:480", "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "32", "-c:a", "aac", "-b:a", "64k", str(out)], check=True)
        return out.read_bytes()


def _review(deps: Deps, video: str, language: str) -> FilmReview:
    from google.genai import types

    from app.adapters.genai_client import make_client

    model = deps.settings.film_review_model
    client = make_client(deps.settings, timeout_seconds=300)
    resp = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=_small(video), mime_type="video/mp4"),
                  render("film_review/system", language=LANGUAGE_NAMES.get(language, language))],
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=FilmReview),
    )
    usage.record_gemini_call("text", model, getattr(resp, "usage_metadata", None))
    return FilmReview.model_validate_json(resp.text)


async def film_review(state: StoryboardState, deps: Deps) -> StoryboardState:
    store = deps.store
    if store.exists("film_review.json"):
        return {"film_review_path": store.abspath("film_review.json")}
    video = state.get("video_path")
    if not deps.settings.film_review_enabled or deps.settings.mock_mode or not video or not Path(video).exists():
        if deps.reporter is not None:
            await deps.reporter.mark_skipped("film review off")
        return {}
    language = state.get("language") or "en"
    try:
        review = await asyncio.to_thread(_review, deps, video, language)
    except Exception as exc:  # noqa: BLE001 - a review never fails the film
        log.warning("film review skipped (%s)", exc)
        return {"warnings": [f"film_review: unavailable ({exc})"]}
    path = store.save_json("film_review.json", {**review.model_dump(), "score": review.score})
    log.info("film review %d/100 — coherence %d, continuity %d, dialogue %d, hook %d, human-made %d",
             review.score, review.coherence, review.continuity, review.dialogue, review.hook, review.human_made)
    log.info("viewer retelling: %s", review.retelling[:300])
    for moment in review.confusing_moments[:5]:
        log.info("confusing: %s", moment[:200])
    return {"film_review_path": path}
