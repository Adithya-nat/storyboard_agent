"""publish_kit stage: title, hook and per-platform captions/hashtags + a cover frame."""

from __future__ import annotations

import asyncio
import shutil

from app.adapters.errors import ProviderError
from app.core.logging import get_logger
from app.graph.state import Deps, StoryboardState
from app.prompts import render
from app.schemas import LANGUAGE_NAMES, PlatformPost, PublishKit, Script

log = get_logger(__name__)


def _hashtag_rule(language: str, tamil_only: bool) -> str:
    if language == "ta":
        return render("publish_kit/hashtags_ta_only" if tamil_only else "publish_kit/hashtags_ta")
    return render("publish_kit/hashtags_en")


def _mock_kit(script: Script) -> PublishKit:
    post = PlatformPost(caption=script.logline[:140], hashtags=["#shorts", "#story"])
    return PublishKit(title=script.title[:60], hook=script.logline[:100], tiktok=post, reels=post, shorts=post)


def cover_order(script: Script) -> int:
    climax = [sc.order for sc in script.scenes if sc.beat_role == "climax"]
    return climax[0] if climax else (script.scenes[len(script.scenes) // 2].order if script.scenes else 1)


async def publish_kit(state: StoryboardState, deps: Deps) -> StoryboardState:
    store = deps.store
    if store.exists("publish.json"):
        return {"publish_path": store.abspath("publish.json"),
                "cover_path": store.abspath("cover.png") if store.exists("cover.png") else None}

    script = Script.model_validate(store.load_json(state["script_path"]))
    language = state.get("language") or script.language
    warnings: list[str] = []
    try:
        kit = await asyncio.to_thread(
            deps.llm.complete_json,
            system=render(
                "publish_kit/system",
                language_name=LANGUAGE_NAMES.get(language, language),
                hashtag_rule=_hashtag_rule(language, deps.settings.tamil_only_hashtags),
            ),
            user=render(
                "publish_kit/user", title=script.title, logline=script.logline,
                arc=script.arc, style=script.style, duration=state["duration"],
            ),
            response_model=PublishKit,
            role="publish",
            mock_factory=lambda _u: _mock_kit(script),
        )
    except ProviderError as exc:
        log.warning("publish kit failed (%s); using the script title", exc)
        warnings.append(f"publish: copy generation failed ({exc}); used the script title")
        kit = _mock_kit(script)

    kit.cover_order = cover_order(script)
    frame = next((f for f in state.get("frames") or [] if f["order"] == kit.cover_order), None)
    cover_path = None
    if frame:
        cover_path = store.abspath("cover.png")
        shutil.copyfile(frame["path"], cover_path)
    path = store.save_json("publish.json", kit)
    log.info("publish kit ready: %r", kit.title)
    return {"publish_path": path, "cover_path": cover_path, "warnings": warnings}
