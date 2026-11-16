"""English renderings of a Tamil film's text, for the app UI only.

The film itself stays Tamil (speech, captions, post copy). The UI stays English, so
it shows each Tamil title, logline and line with an English rendering beside it.

Cost: one small call to the cheapest text model, run after the script is final. It is
cached per string in ``gloss.json``, so a resume, a later rewrite or a second read only
translates strings that are new. It's best-effort: a failure leaves the Tamil text
standing alone and never affects the film.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable

from pydantic import BaseModel

from app.adapters.errors import ProviderError
from app.core.logging import get_logger
from app.prompts import render

log = get_logger(__name__)

GLOSS_FILE = "gloss.json"
TAMIL = re.compile(r"[஀-௿]")


class _Item(BaseModel):
    i: int
    en: str


class _Gloss(BaseModel):
    items: list[_Item] = []


def script_texts(script: dict) -> list[str]:
    """Every viewer-facing string of a script (title, logline, narration, spoken lines)."""
    out = [script.get("title") or "", script.get("logline") or ""]
    for sc in script.get("scenes") or []:
        out.append(sc.get("narration") or "")
        for line in sc.get("lines") or []:
            out.append(line.get("text") or "")
        for sh in sc.get("shots") or []:
            for line in sh.get("lines") or []:
                out.append(line.get("text") or "")
    return out


def load(store) -> dict[str, str]:
    return (store.load_json(store.abspath(GLOSS_FILE)) or {}) if store.exists(GLOSS_FILE) else {}


async def ensure(deps, texts: Iterable[str]) -> dict[str, str]:
    """Translate any Tamil strings not glossed yet; return the whole map."""
    store = deps.store
    have = load(store)
    todo = [t for t in dict.fromkeys(texts) if t and TAMIL.search(t) and t not in have]
    if not todo:
        return have

    def mock(_u: str) -> _Gloss:
        return _Gloss(items=[_Item(i=i, en=t) for i, t in enumerate(todo)])

    try:
        out = await asyncio.to_thread(
            deps.llm.complete_json,
            system=render("gloss/system"),
            user=render("gloss/user", items="\n".join(f"{i}. {t}" for i, t in enumerate(todo))),
            response_model=_Gloss,
            role="gloss",
            mock_factory=mock,
        )
    except ProviderError as exc:
        log.warning("English renderings skipped (%s); the UI shows the Tamil text alone", exc)
        return have
    for item in out.items:
        if 0 <= item.i < len(todo) and item.en.strip():
            have[todo[item.i]] = item.en.strip()
    store.save_json(GLOSS_FILE, have)
    log.info("English renderings: %d new Tamil string(s)", len(todo))
    return have


async def ensure_for_script(state: dict, deps) -> None:
    """Graph post-hook: gloss the final script of a Tamil film (no-op otherwise)."""
    if (state.get("language") or "en") != "ta" or not state.get("script_path"):
        return
    try:
        await ensure(deps, script_texts(deps.store.load_json(state["script_path"]) or {}))
    except Exception:  # noqa: BLE001 - presentation only, never fails a stage
        log.warning("English renderings failed", exc_info=True)
