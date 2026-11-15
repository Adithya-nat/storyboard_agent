"""Golden-prompt eval runner (manual).

    python -m evals.run            # real models (needs GEMINI_API_KEY)
    python -m evals.run --mock     # offline smoke run (placeholder frames)
    python -m evals.run --limit 2  # first N prompts only

For each prompt it drives the full graph (script -> designer -> image_gen ->
assembler), dumps every artifact under ``evals/output/<timestamp>/<slug>/`` for
human review, and runs cheap heuristics (schema parses, frame count matches the
duration, captions are a readable length, one style reference exists). Exits
non-zero if any heuristic fails — handy for ad-hoc gating, but this is NOT wired
into CI (mocked unit tests are).
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from datetime import datetime
from pathlib import Path

from app.adapters.gemini import GeminiImageAdapter
from app.adapters.llm import LLMAdapter
from app.agents.judge import NarrativeJudge
from app.artifacts.store import ArtifactStore
from app.core.config import Settings, get_settings
from app.graph.build import build_graph
from app.graph.state import Deps, scenes_for_duration
from app.schemas import CreativeBrief, JudgeReport, Script, Shotlist
from evals.golden_prompts import GOLDEN_PROMPTS

_MAX_CAPTION_CHARS = 120  # a 2s on-screen caption should be short enough to read


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "prompt"


Check = tuple[str, bool, str]  # (name, ok, detail)


def _checks(store: ArtifactStore, duration: int, settings: Settings, video_path: str | None) -> list[Check]:
    expected = (len(CreativeBrief.model_validate(store.load_json(store.abspath("brief.json"))).scenes)
                if store.exists("brief.json") else scenes_for_duration(duration, settings.default_style))
    out: list[Check] = []

    def add(name: str, fn) -> None:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001 - a failed check is a finding, not a crash
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        out.append((name, ok, detail))

    def script_check():
        s = Script.model_validate(store.load_json(store.abspath("script.json")))
        return len(s.scenes) == expected, f"{len(s.scenes)} scenes (want {expected})"

    def shotlist_check():
        sl = Shotlist.model_validate(store.load_json(store.abspath("shotlist.json")))
        ok = len(sl.shots) == expected and bool(sl.style_bible.strip())
        return ok, f"{len(sl.shots)} shots (want {expected}), bible={len(sl.style_bible)} chars"

    def bible_check():
        text = store.load_text(store.abspath("style_bible.md"))
        return bool(text.strip()), f"{len(text)} chars"

    def style_ref_check():
        p = Path(store.abspath("style_ref.png"))
        size = p.stat().st_size if p.exists() else 0
        return size > 0, f"{size} bytes" + ("" if size else " (text-only fallback)")

    def frames_check():
        n = len(list(store.job_dir.glob("frame_*.png")))
        return n == expected, f"{n} frames (want {expected})"

    def caption_check():
        sl = Shotlist.model_validate(store.load_json(store.abspath("shotlist.json")))
        longest = max((len(s.narration) for s in sl.shots), default=0)
        return longest <= _MAX_CAPTION_CHARS, f"longest caption {longest} chars (max {_MAX_CAPTION_CHARS})"

    def video_check():
        ok = bool(video_path) and Path(video_path).exists() and Path(video_path).stat().st_size > 0
        return ok, (video_path or "no video (ffmpeg missing?)")

    add("script", script_check)
    add("shotlist", shotlist_check)
    add("style_bible", bible_check)
    add("style_ref", style_ref_check)
    add("frames", frames_check)
    add("captions", caption_check)
    add("video", video_check)
    return out


def _quality(store: ArtifactStore, deps: Deps) -> tuple[JudgeReport | None, str | None]:
    """Score the PRE-MEDIA artifacts (script + shotlist) for story cohesion — the W2
    measurement. A judge failure is a finding, not a crash."""
    if not store.exists("script.json"):
        return None, "no script.json to judge"
    try:
        script = Script.model_validate(store.load_json(store.abspath("script.json")))
        shotlist = (
            Shotlist.model_validate(store.load_json(store.abspath("shotlist.json")))
            if store.exists("shotlist.json")
            else None
        )
        brief = (CreativeBrief.model_validate(store.load_json(store.abspath("brief.json")))
                 if store.exists("brief.json") else None)
        return NarrativeJudge(deps.llm, deps.settings).score(script, shotlist, brief), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


async def evaluate(prompt: str, duration: int, out_root: Path, settings: Settings) -> dict:
    slug = _slug(prompt)
    store = ArtifactStore(out_root, slug)
    deps = Deps(
        settings=settings,
        store=store,
        llm=LLMAdapter(settings),
        image=GeminiImageAdapter(settings),
    )
    graph = build_graph(deps)  # one-shot eval: no checkpointer needed
    error: str | None = None
    video_path: str | None = None
    try:
        final = await graph.ainvoke(
            {"job_id": slug, "prompt": prompt, "duration": duration, "warnings": []}
        )
        video_path = final.get("video_path")
    except Exception as exc:  # noqa: BLE001 - record + still inspect on-disk artifacts
        error = f"{type(exc).__name__}: {exc}"

    judge, judge_error = _quality(store, deps)
    return {
        "prompt": prompt,
        "duration": duration,
        "slug": slug,
        "dir": str(store.job_dir),
        "error": error,
        "checks": _checks(store, duration, settings, video_path),
        "judge": judge,
        "judge_error": judge_error,
    }


def _print_summary(results: list[dict], out_root: Path) -> None:
    print("\n" + "=" * 72)
    print(f"EVAL SUMMARY  ({len(results)} prompts)  ->  {out_root}")
    print("=" * 72)
    for r in results:
        passed = sum(1 for _, ok, _ in r["checks"] if ok)
        total = len(r["checks"])
        flag = "OK " if passed == total and not r["error"] else "!! "
        print(f"\n{flag}[{passed}/{total}] {r['prompt'][:60]}  ({r['duration']}s)")
        print(f"    dir: {r['dir']}")
        if r["error"]:
            print(f"    ERROR during run: {r['error']}")
        for name, ok, detail in r["checks"]:
            print(f"    [{'x' if ok else ' '}] {name:12} {detail}")
        jr = r.get("judge")
        if jr is not None:
            print(
                f"    QUALITY overall={jr.overall:3d}  struct={jr.structure} "
                f"cont={jr.continuity} clar={jr.clarity} vis={jr.visual_concreteness} "
                f"pay={jr.payoff}"
            )
            if jr.weakest_dimension:
                print(f"      weakest: {jr.weakest_dimension}")
            if jr.revision_guidance:
                print(f"      fix: {jr.revision_guidance[:140]}")
        elif r.get("judge_error"):
            print(f"    QUALITY (skipped): {r['judge_error']}")
    scored = [r["judge"].overall for r in results if r.get("judge") is not None]
    if scored:
        mean = sum(scored) / len(scored)
        print(f"\nMEAN STORY-QUALITY (overall): {mean:.1f} / 100  (n={len(scored)})")
    print("=" * 72)


async def _run_all(goldens: list[tuple[str, int]], out_root: Path, settings: Settings) -> list[dict]:
    results = []
    for prompt, duration in goldens:
        print(f"running: {prompt[:60]} ({duration}s) ...")
        results.append(await evaluate(prompt, duration, out_root, settings))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Manual golden-prompt eval harness.")
    ap.add_argument("--mock", action="store_true", help="force mock providers (smoke run)")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N prompts")
    ap.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="fail (exit 1) if the MEAN story-quality (overall) is below this (0-100)",
    )
    args = ap.parse_args()

    settings = Settings(use_mock_providers=True) if args.mock else get_settings()
    if settings.mock_mode and not args.mock:
        print(
            "WARNING: no GEMINI_API_KEY -> MOCK mode. Frames are placeholders; this is "
            "a smoke run, not a real quality check.\n"
        )

    out_root = Path("evals/output") / datetime.now().strftime("%Y%m%d-%H%M%S")
    goldens = GOLDEN_PROMPTS[: args.limit] if args.limit else GOLDEN_PROMPTS
    results = asyncio.run(_run_all(goldens, out_root, settings))
    _print_summary(results, out_root)

    failed = any(not ok for r in results for _, ok, _ in r["checks"]) or any(r["error"] for r in results)
    if args.min_score is not None:
        scored = [r["judge"].overall for r in results if r.get("judge") is not None]
        mean = (sum(scored) / len(scored)) if scored else 0.0
        if mean < args.min_score:
            print(f"\nQUALITY GATE FAILED: mean {mean:.1f} < --min-score {args.min_score}")
            failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
