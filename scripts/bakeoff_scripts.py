"""G0 script bake-off (LIVE, ~$8-10): Gemini 4 Carbon/Argon vs Claude Opus 5.5/Fable 5.2.

    python -m scripts.bakeoff_scripts --out DIR

Every model writes the same 12 scripts (2 prompts x 3 styles x 2 languages) with the
production script_writer prompt. A cross-provider panel (Gemini 4 Argon + Claude Opus 5.5)
scores each with the NarrativeJudge rubric; the canonical overall is computed in code.
Writes DIR/results.json and DIR/blind_ab.html (best Gemini vs best Claude, sides shuffled)
for the user's blind picks. Claude refusal fallbacks are deliberately OFF here: a fallback
would swap models and contaminate the comparison.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from dotenv import dotenv_values

from app.adapters.llm import LLMAdapter
from app.agents.judge import _format_target, _system, compute_overall
from app.agents.script_writer import _system_prompt, normalize_script
from app.core.config import get_settings
from app.graph.state import scenes_for_duration
from app.prompts import render
from app.schemas import LANGUAGE_NAMES, JudgeReport, Script

GEMINI = ["gemini-4-carbon", "gemini-4-argon"]
CLAUDE = ["claude-opus-5-5", "claude-fable-5-2"]
PROMPTS = [
    ("A lonely lighthouse keeper befriends a storm petrel over one winter", 30),
    ("Two rival tea masters settle their feud in a single ceremony", 30),
]
STYLES = ["dialogue", "mixed", "narrated"]
LANGS = ["en", "ta"]


def _claude() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=dotenv_values(".env").get("ANTHROPIC_API_KEY"), max_retries=4)


def write_script(model: str, idea: str, duration: int, style: str, lang: str) -> tuple[Script | None, float, str]:
    s = get_settings()
    n = scenes_for_duration(duration, s.seconds_per_frame, style)
    system = _system_prompt(n, style, lang, duration / n)
    user = render("script_writer/user", direction="", idea=idea, duration=duration, n=n, seconds=round(duration / n),
                  style=style, language=LANGUAGE_NAMES[lang])
    t = time.time()
    try:
        if model.startswith("claude"):
            resp = _claude().messages.parse(
                model=model, max_tokens=16000, system=system,
                messages=[{"role": "user", "content": user}],
                output_format=Script, output_config={"effort": "high"},
            )
            if resp.stop_reason == "refusal" or resp.parsed_output is None:
                return None, time.time() - t, f"stop_reason={resp.stop_reason}"
            script = resp.parsed_output
        else:
            s2 = s.model_copy(update={"script_model": model, "script_thinking": "high"})
            script = LLMAdapter(s2).complete_json(system=system, user=user, response_model=Script,
                                                  mock_factory=lambda _u: None, role="script")
        return normalize_script(script, style=style, language=lang, n=n), time.time() - t, ""
    except Exception as exc:  # noqa: BLE001
        return None, time.time() - t, f"{type(exc).__name__}: {str(exc)[:200]}"


def judge(script: Script) -> dict:
    has_dialogue = any(sc.mode == "dialogue" and sc.lines for sc in script.scenes)
    system, user = _system(has_dialogue), _format_target(script, None)
    scores = {}
    try:
        s = get_settings().model_copy(update={"judge_model": "gemini-4-argon"})
        g = LLMAdapter(s).complete_json(system=system, user=user, response_model=JudgeReport,
                                        mock_factory=lambda _u: None, role="judge")
        scores["gemini-4-argon"] = compute_overall(g, has_dialogue)
    except Exception as exc:  # noqa: BLE001
        scores["gemini-4-argon_error"] = str(exc)[:200]
    try:
        r = _claude().messages.parse(model="claude-opus-5-5", max_tokens=8000, system=system,
                                     messages=[{"role": "user", "content": user}],
                                     output_format=JudgeReport, output_config={"effort": "medium"})
        if r.parsed_output is not None:
            scores["claude-opus-5-5"] = compute_overall(r.parsed_output, has_dialogue)
    except Exception as exc:  # noqa: BLE001
        scores["claude-opus-5-5_error"] = str(exc)[:200]
    vals = [v for k, v in scores.items() if not k.endswith("_error")]
    scores["panel"] = round(sum(vals) / len(vals), 1) if vals else None
    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    cases = [(i, idea, d, st, lg) for i, (idea, d) in enumerate(PROMPTS) for st in STYLES for lg in LANGS]
    jobs = [(m, c) for m in GEMINI + CLAUDE for c in cases]
    results: dict = {m: {} for m in GEMINI + CLAUDE}

    def run(job):
        model, (i, idea, d, st, lg) = job
        key = f"p{i}_{st}_{lg}"
        script, secs, err = write_script(model, idea, d, st, lg)
        rec = {"seconds": round(secs, 1), "error": err}
        if script is not None:
            (out / f"{model}_{key}.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")
            rec["scores"] = judge(script)
        print(model, key, rec.get("scores", {}).get("panel"), err, f"{secs:.0f}s", flush=True)
        return model, key, rec

    with ThreadPoolExecutor(max_workers=6) as pool:
        for model, key, rec in pool.map(run, jobs):
            results[model][key] = rec

    summary = {}
    for m, recs in results.items():
        panel = [r["scores"]["panel"] for r in recs.values() if r.get("scores", {}).get("panel") is not None]
        by_judge = {}
        for j in ("gemini-4-argon", "claude-opus-5-5"):
            v = [r["scores"][j] for r in recs.values() if j in r.get("scores", {})]
            by_judge[j] = round(sum(v) / len(v), 1) if v else None
        summary[m] = {"panel_avg": round(sum(panel) / len(panel), 1) if panel else None,
                      "by_judge": by_judge, "ok": len(panel), "failed": len(recs) - len(panel),
                      "avg_seconds": round(sum(r["seconds"] for r in recs.values()) / max(1, len(recs)), 1)}
    best_g = max(GEMINI, key=lambda m: summary[m]["panel_avg"] or 0)
    best_c = max(CLAUDE, key=lambda m: summary[m]["panel_avg"] or 0)
    pairs, rng = [], random.Random(7)
    for (i, idea, d, st, lg) in cases:
        key = f"p{i}_{st}_{lg}"
        a, b = out / f"{best_g}_{key}.json", out / f"{best_c}_{key}.json"
        if a.exists() and b.exists():
            flip = rng.random() < 0.5
            pairs.append({"key": key, "idea": idea, "A": b if flip else a, "B": a if flip else b,
                          "A_model": best_c if flip else best_g, "B_model": best_g if flip else best_c})
    _write_blind_page(out / "blind_ab.html", pairs)
    (out / "results.json").write_text(json.dumps({
        "summary": summary, "best_gemini": best_g, "best_claude": best_c, "per_case": results,
        "blind_key": [{"key": p["key"], "A": p["A_model"], "B": p["B_model"]} for p in pairs],
    }, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=1))
    return 0


def _render_script(path: Path) -> str:
    s = Script.model_validate_json(path.read_text(encoding="utf-8"))
    rows = [f"<h3>{html.escape(s.title)}</h3><p><i>{html.escape(s.logline)}</i></p>"]
    for sc in s.scenes:
        if sc.mode == "dialogue":
            body = "<br>".join(f"<b>{html.escape(l.speaker)}</b>: {html.escape(l.text)}" for l in sc.lines)
        else:
            body = f"<span class=n>NARRATOR:</span> {html.escape(sc.narration)}"
        rows.append(f"<div class=sc><small>{sc.order} · {html.escape(sc.beat_role)}</small> "
                    f"<div class=d>{html.escape(sc.description)}</div>{body}</div>")
    return "".join(rows)


def _write_blind_page(path: Path, pairs: list[dict]) -> None:
    items = []
    for n, p in enumerate(pairs):
        items.append(f"""<section><h2>{n + 1}. {html.escape(p['idea'])} <small>({p['key'].split('_', 1)[1]})</small></h2>
<div class=row><div class=col><h4>A</h4>{_render_script(p['A'])}</div><div class=col><h4>B</h4>{_render_script(p['B'])}</div></div>
<p>Better story: <label><input type=radio name=q{n} value=A> A</label> <label><input type=radio name=q{n} value=B> B</label></p></section>""")
    path.write_text(f"""<!doctype html><meta charset=utf-8><title>Script blind test</title>
<style>body{{font:15px/1.5 system-ui,'Noto Sans Tamil';max-width:1200px;margin:24px auto;padding:0 16px}}
.row{{display:flex;gap:16px}}.col{{flex:1;border:1px solid #ccc;border-radius:8px;padding:12px}}
.sc{{margin:8px 0}}.d{{color:#666;font-size:13px}}.n{{color:#888}}section{{margin-bottom:40px}}
#code{{position:sticky;bottom:0;background:#ffd;padding:12px;font:16px monospace}}</style>
<h1>Which script is better? (models hidden)</h1>{''.join(items)}
<div id=code>Your picks code: <b id=c>—</b> (paste this back to Claude)</div>
<script>document.addEventListener('change',()=>{{let s='';for(let i=0;i<{len(pairs)};i++){{const v=document.querySelector('input[name=q'+i+']:checked');s+=v?v.value:'-';}}document.getElementById('c').textContent=s;}});</script>""",
                    encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
