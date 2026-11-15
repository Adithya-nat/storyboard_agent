"""music stage: one instrumental Lyria score per film (idempotent; never fails the job)."""

from __future__ import annotations

import asyncio

from app.adapters.errors import ProviderError
from app.core.logging import get_logger
from app.graph.state import Deps, StoryboardState
from app.prompts import render
from app.schemas import Script

log = get_logger(__name__)


def total_runtime(state: StoryboardState) -> float:
    holds = [a.get("hold") or 0 for a in state.get("scene_audio") or []]
    return sum(holds) or float(state["duration"])


async def music(state: StoryboardState, deps: Deps) -> StoryboardState:
    store = deps.store
    for ext in ("mp3", "wav"):
        if store.exists(f"music.{ext}"):
            return {"music_path": store.abspath(f"music.{ext}")}
    if not deps.settings.music_enabled or deps.music is None:
        log.info("music: disabled; the film ships without a score")
        if deps.reporter is not None:
            await deps.reporter.mark_skipped("music off")
        return {"music_path": None}

    script = Script.model_validate(store.load_json(state["script_path"]))
    seconds = round(total_runtime(state))
    climax = next((i for i, sc in enumerate(script.scenes) if sc.beat_role == "climax"), len(script.scenes) - 2)
    prompt = render(
        "music/score",
        seconds=seconds,
        logline=script.logline,
        arc=script.arc,
        look=script.style,
        climax_at=round(seconds * (climax + 0.5) / max(1, len(script.scenes))),
    )
    try:
        data, ext, model = await asyncio.to_thread(deps.music.compose, prompt)
    except ProviderError as exc:
        log.warning("music failed (%s); the film ships without a score", exc)
        return {"music_path": None, "warnings": [f"music: score unavailable ({exc}); no music"]}
    path = store.save_bytes(f"music.{ext}", data)
    log.info("score ready (%s, %s)", model, ext)
    return {"music_path": path}
