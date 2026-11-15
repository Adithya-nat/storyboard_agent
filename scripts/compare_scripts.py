"""Old-vs-new screenplay comparison (LIVE, text only, ~$0.5 per prompt per side).

    python -m scripts.compare_scripts write --out DIR --tag new      # in this repo
    python -m scripts.compare_scripts write --out DIR --tag old      # same file, run in a checkout of main
    python -m scripts.compare_scripts judge --out DIR                # judge both with judge v2 + blind A/B page

``write`` runs whichever text stages this checkout has (brief, script_writer, dialogue_pass,
reflection) for every CREATOR_PROMPT, in parallel.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import random
from pathlib import Path

from app.adapters.gemini import GeminiImageAdapter
from app.adapters.llm import LLMAdapter
from app.artifacts.store import ArtifactStore
from app.core.config import get_settings
from app.graph.state import Deps
from evals.golden_prompts import CREATOR_PROMPTS


async def _write_one(i: int, prompt: str, duration: int, style: str, language: str, root: Path) -> None:
    settings = get_settings()
    store = ArtifactStore(root, f"p{i}")
    deps = Deps(settings=settings, store=store, llm=LLMAdapter(settings), image=GeminiImageAdapter(settings))
    state = {"job_id": f"p{i}", "prompt": prompt, "duration": duration, "style": style, "language": language}
    stages = []
    try:
        from app.agents.brief import brief
        stages.append(brief)
    except ImportError:
        pass
    from app.agents.script_writer import script_writer
    stages.append(script_writer)
    try:
        from app.agents.dialogue_pass import dialogue_pass
        stages.append(dialogue_pass)
    except ImportError:
        pass
    from app.agents.reflection import reflection
    stages.append(reflection)
    for stage in stages:
        out = await stage(state, deps)
        state.update({k: v for k, v in (out or {}).items() if k != "warnings"})
        print(f"p{i} {stage.__name__} done", flush=True)


async def write(out: Path, tag: str) -> None:
    root = out / tag
    await asyncio.gather(*(_write_one(i, *p, root) for i, p in enumerate(CREATOR_PROMPTS, 1)))


def _screenplay_html(script: dict) -> str:
    names = {c["name"].upper(): (c.get("display_name") or c["name"]) for c in script.get("characters", [])}
    rows = [f"<h4>{html.escape(script.get('title', ''))}</h4>"]
    for sc in script["scenes"]:
        head = f"Scene {sc['order']}"
        if sc.get("seconds"):
            head += f" · {sc['seconds']:g}s"
        if sc.get("location"):
            head += f" · {html.escape(sc['location'])}"
        rows.append(f"<div class='scene'><div class='sh'>{head}</div>")
        shots = sc.get("shots") or [{"action": sc["description"], "lines": sc.get("lines", [])}]
        if sc.get("mode") == "narration" and sc.get("narration"):
            rows.append(f"<p class='nar'>NARRATOR: {html.escape(sc['narration'])}</p>")
        for sh in shots:
            size = (sh.get("size") or "").upper().replace("_", " ")
            rows.append(f"<p class='act'>{html.escape(size + ': ' if size else '')}{html.escape(sh.get('action', ''))}</p>")
            for ln in sh.get("lines", []):
                who = names.get(ln["speaker"].upper(), ln["speaker"])
                rows.append(f"<p class='ln'><b>{html.escape(who)}</b>: {html.escape(ln['text'])}</p>")
        rows.append("</div>")
    return "\n".join(rows)


def judge(out: Path) -> None:
    from app.agents.judge import NarrativeJudge
    from app.schemas import CreativeBrief, Script

    settings = get_settings()
    j = NarrativeJudge(LLMAdapter(settings), settings)
    rng = random.Random(1114)
    results, cards, key = [], [], {}
    for i, (prompt, duration, style, language) in enumerate(CREATOR_PROMPTS, 1):
        brief = CreativeBrief.model_validate_json((out / "new" / f"p{i}" / "brief.json").read_text())
        row = {"prompt": prompt[:80], "duration": duration, "style": style, "language": language}
        scripts = {}
        for tag in ("old", "new"):
            raw = json.loads((out / tag / f"p{i}" / "script.json").read_text())
            scripts[tag] = raw
            rep = j.score(Script.model_validate(raw), None, brief)
            row[tag] = {k: getattr(rep, k) for k in ("overall", "hook", "brief_adherence", "structure", "continuity",
                                                     "clarity", "payoff", "dialogue", "dialogue_thread", "register_fit")}
            row[tag]["retelling"] = rep.retelling
            print(f"p{i} {tag}: {row[tag]['overall']}", flush=True)
        results.append(row)
        order = ["old", "new"]
        rng.shuffle(order)
        key[f"p{i}"] = {"A": order[0], "B": order[1]}
        cards.append(
            f"<section><h3>{i}. {html.escape(style)} · {html.escape(language)} · {duration}s</h3>"
            f"<details><summary>Idea</summary><pre>{html.escape(prompt)}</pre></details>"
            f"<div class='pair'><div><h4>A</h4>{_screenplay_html(scripts[order[0]])}</div>"
            f"<div><h4>B</h4>{_screenplay_html(scripts[order[1]])}</div></div>"
            f"<p class='pick'>Which plays more like a real creator's film? "
            f"<label><input type=radio name=p{i} value=A> A</label> <label><input type=radio name=p{i} value=B> B</label>"
            f" <input class=note data-p=p{i} placeholder='why (optional)'></p></section>"
        )
    (out / "judge_v2.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (out / "ab_key.json").write_text(json.dumps(key, indent=2))
    page = """<!doctype html><meta charset=utf-8><title>Script blind A/B</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>body{font:15px/1.5 system-ui,'Noto Sans Tamil',sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;background:#141a2e;color:#efe8da}
section{border-top:1px solid #333a55;padding:16px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:800px){.pair{grid-template-columns:1fr}}.pair>div{background:#1c2340;padding:12px;border-radius:8px}
.sh{color:#f2b544;font-weight:600;margin-top:10px}.act{color:#a9b0c8;margin:4px 0;font-size:13px}.ln{margin:2px 0 2px 12px}
.nar{font-style:italic;color:#d6cfbf}pre{white-space:pre-wrap;font-size:12px;color:#a9b0c8}.pick{font-weight:600}
.note{width:50%;background:#1c2340;color:#efe8da;border:1px solid #333a55;padding:4px}button{background:#f2b544;border:0;padding:8px 14px;font-weight:700}</style>
<h1>Which screenplay is better?</h1><p>Each pair is the same idea written by two versions of the pipeline (randomly labelled). Pick per idea, then copy your picks back to Claude.</p>
""" + "\n".join(cards) + """
<p><button onclick="out.value=[...document.querySelectorAll('input[type=radio]:checked')].map(r=>r.name+'='+r.value+' '+(document.querySelector('.note[data-p='+r.name+']').value||'')).join('\\n')">Show my picks</button></p>
<textarea id=out rows=6 style="width:100%;background:#1c2340;color:#efe8da"></textarea>"""
    (out / "blind_ab.html").write_text(page)
    print("wrote", out / "blind_ab.html")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write", "judge"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="new")
    a = ap.parse_args()
    out = Path(a.out)
    if a.cmd == "write":
        asyncio.run(write(out, a.tag))
    else:
        judge(out)


if __name__ == "__main__":
    main()
