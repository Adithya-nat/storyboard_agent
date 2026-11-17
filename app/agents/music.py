"""music stage: one instrumental Lyria score per film (idempotent; never fails the job)."""

from __future__ import annotations

import asyncio

from app.adapters.errors import ProviderError
from app.agents.redo_director import direction
from app.core.logging import get_logger
from app.graph.state import Deps, StoryboardState
from app.prompts import render
from app.schemas import Script

log = get_logger(__name__)


def total_runtime(state: StoryboardState) -> float:
    holds = [a.get("hold") or 0 for a in state.get("scene_audio") or []]
    return sum(holds) or float(state["duration"])


def _fits(timing: dict | None, seconds: float, starts: list[float]) -> bool:
    """A score still fits when the runtime and every cut point moved by under a second."""
    if not timing:
        return True  # older films: no record, keep the score
    old = timing.get("starts") or []
    return (abs(float(timing.get("seconds", seconds)) - seconds) < 1.0 and len(old) == len(starts)
            and all(abs(a - b) < 1.0 for a, b in zip(old, starts)))


async def music(state: StoryboardState, deps: Deps) -> StoryboardState:
    store = deps.store
    holds = [a.get("hold") or 0 for a in sorted(state.get("scene_audio") or [], key=lambda a: a["order"])]
    runtime = total_runtime(state)
    cut_points = [round(sum(holds[:i]), 2) for i in range(len(holds))] if all(holds) else []
    for ext in ("mp3", "wav"):
        if store.exists(f"music.{ext}"):
            timing = store.load_json(store.abspath("music.json")) if store.exists("music.json") else None
            if _fits(timing, runtime, cut_points):
                return {"music_path": store.abspath(f"music.{ext}")}
            log.info("music: the cut's timing changed (now %.0fs); rescoring to the new cuts", runtime)
            break
    if not deps.settings.music_enabled or deps.music is None:
        log.info("music: disabled; the film ships without a score")
        if deps.reporter is not None:
            await deps.reporter.mark_skipped("music off")
        return {"music_path": None}

    script = Script.model_validate(store.load_json(state["script_path"]))
    seconds = round(runtime)
    starts = [sum(holds[:i]) for i in range(len(holds))] if all(holds) and len(holds) == len(script.scenes) else []
    climax = next((i for i, sc in enumerate(script.scenes) if sc.beat_role == "climax"), len(script.scenes) - 2)
    climax_at = (starts[climax] + holds[climax] / 2) if starts else seconds * (climax + 0.5) / max(1, len(script.scenes))
    prompt = render(
        "music/score",
        seconds=seconds,
        logline=script.logline,
        arc=script.arc,
        look=script.style,
        climax_at=round(climax_at),
        cues=", ".join(f"{t:.0f}s" for t in starts[1:]) or "evenly through the film",
        direction=direction(store, "score"),
    )
    try:
        data, ext, model = await asyncio.to_thread(deps.music.compose, prompt)
    except ProviderError as exc:
        kept = next((store.abspath(f"music.{e}") for e in ("mp3", "wav") if store.exists(f"music.{e}")), None)
        if kept:
            log.warning("rescoring failed (%s); keeping the earlier score", exc)
            return {"music_path": kept, "warnings": [f"music: rescoring failed ({exc}); kept the earlier score"]}
        log.warning("music failed (%s); the film ships without a score", exc)
        return {"music_path": None, "warnings": [f"music: score unavailable ({exc}); no music"]}
    from app.redo.takes import archive, current_cut
    from app.redo.targets import Item

    archive(store.job_dir, [Item("score")], current_cut(store.job_dir))  # the old score stays restorable
    path = store.save_bytes(f"music.{ext}", data)
    store.save_json("music.json", {"seconds": round(runtime, 2), "starts": cut_points})
    log.info("score ready (%s, %s)", model, ext)
    return {"music_path": path}
