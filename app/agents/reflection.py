"""reflection node: critique -> revise the script until it tells a coherent story.

Runs AFTER script_writer, BEFORE designer — fix the STORY before the look is locked.

    script.json ──▶ judge ──▶ overall >= threshold? ──yes──▶ keep ──▶ script.json
                      ▲                 │ no, iters left
                      │                 ▼
                   re-judge ◀── revise (uses judge.revision_guidance)   [accept-best]

The loop lives ENTIRELY inside this node (no LangGraph cycle), so the checkpoint/resume
contract the project built is untouched. We keep the BEST-scoring version seen (accept-
best), not necessarily the last — a noisy judge can score a revision lower.

Idempotent (resume): a ``reflection.json`` sentinel means "already reflected" -> skip. The
chosen script OVERWRITES ``script.json`` so the designer + everything downstream use the
improved version. ``script_writer`` is also idempotent, so on resume it reuses the
(already-improved) script and reflection's sentinel skips a second pass.

Degrade, don't die: a judge/reviser ProviderError logs a warning, keeps the best script so
far, and proceeds. Reflection must NEVER fail a job. Mirrors the designer's fallbacks.
"""

from __future__ import annotations

import asyncio

from app.adapters.errors import ProviderError
from app.agents.judge import NarrativeJudge
from app.core.logging import get_logger
from app.graph.state import Deps, StoryboardState
from app.schemas import JudgeReport, Script

log = get_logger(__name__)

_REVISER_SYSTEM = (
    "<role>\n"
    "You are a master script doctor. You are given a STORYBOARD SCRIPT that an editor "
    "scored as not-yet-coherent, plus the editor's notes. Rewrite the script so it tells "
    "ONE clear story with a real beginning, middle, and end, fixing the SPECIFIC "
    "weaknesses called out — concentrate your effort on the named weak beats and the "
    "weakest dimension.\n"
    "</role>\n\n"
    "<hard_constraints>\n"
    "  - Keep EXACTLY the same number of scenes and the same 'order' values.\n"
    "  - Keep the same overall premise and subject — improve it, don't replace it.\n"
    "  - Strengthen causality: each beat must follow from the one before.\n"
    "  - Write a coherent 'arc' and set each scene's 'beat_role' "
    "(setup/inciting/rising/climax/resolution).\n"
    "  - 'description' stays concrete and filmable; 'narration' is one short caption.\n"
    "</hard_constraints>\n\n"
    "Return STRICT JSON matching the Script schema."
)


def _revise(deps: Deps, script: Script, report: JudgeReport) -> Script:
    """One revision pass: rewrite the script to address the judge's guidance.

    Mock mode returns the script unchanged (deterministic offline loop)."""
    return deps.llm.complete_json(
        system=_REVISER_SYSTEM,
        user=(
            "<editor_notes>\n"
            f"Apply these fixes: {report.revision_guidance or report.rationale}\n"
            f"Weakest dimension: {report.weakest_dimension}; weak beats: {report.weak_beats}\n"
            "</editor_notes>\n\n"
            "<current_script_json>\n"
            f"{script.model_dump_json(indent=2)}\n"
            "</current_script_json>"
        ),
        response_model=Script,
        mock_factory=lambda _u: script,
    )


async def reflection(state: StoryboardState, deps: Deps) -> StoryboardState:
    store = deps.store
    settings = deps.settings

    if not settings.reflection_enabled:
        log.info("reflection disabled; using the script as written")
        return {}  # no-op: designer reads script.json exactly as script_writer left it

    # Idempotency (resume): already reflected -> reuse, don't re-judge/re-bill.
    if store.exists("reflection.json"):
        log.info("reusing the earlier reflection result (resume)")
        return {"reflection_path": store.abspath("reflection.json")}

    script = Script.model_validate(store.load_json(state["script_path"]))
    judge = NarrativeJudge(deps.llm, settings)
    warnings: list[str] = []

    # Initial score. If we can't even judge, proceed with the script untouched.
    try:
        best_report = await asyncio.to_thread(judge.score, script)
    except ProviderError as exc:
        log.warning("reflection: initial judge failed (%s); skipping reflection", exc)
        return {"warnings": [f"reflection: judge unavailable ({exc}); script left as-is"]}

    log.info(
        "judge scored the story %d/100 — structure %d, continuity %d, clarity %d, "
        "payoff %d, visual %d (threshold %d)",
        best_report.overall, best_report.structure, best_report.continuity,
        best_report.clarity, best_report.payoff, best_report.visual_concreteness,
        settings.reflection_threshold,
    )

    best_script = script
    iters = 0
    while (
        best_report.overall < settings.reflection_threshold
        and iters < settings.reflection_max_iters
    ):
        iters += 1
        log.info(
            "story %d/100 below threshold %d — revising (iter %d/%d), weakest: %s",
            best_report.overall, settings.reflection_threshold, iters,
            settings.reflection_max_iters, best_report.weakest_dimension,
        )
        try:
            revised = await asyncio.to_thread(_revise, deps, best_script, best_report)
            # Guard the downstream invariant: frame/scene count must not drift.
            if len(revised.scenes) != len(best_script.scenes):
                log.warning(
                    "reflection: revision changed scene count (%d -> %d); rejecting it",
                    len(best_script.scenes), len(revised.scenes),
                )
                warnings.append("reflection: a revision changed the scene count; kept prior version")
                continue
            new_report = await asyncio.to_thread(judge.score, revised)
        except ProviderError as exc:
            log.warning("reflection: revise/judge failed on iter %d (%s); keeping best", iters, exc)
            warnings.append(f"reflection: revision step failed ({exc}); kept best-so-far")
            break
        log.info(
            "reflection iter %d: %d -> %d (threshold %d)",
            iters, best_report.overall, new_report.overall, settings.reflection_threshold,
        )
        if new_report.overall > best_report.overall:  # accept-best
            best_script, best_report = revised, new_report

    if best_report.overall >= settings.reflection_threshold:
        log.info("story accepted at %d/100 after %d revision(s)", best_report.overall, iters)
    else:
        log.info(
            "kept the best story at %d/100 (below threshold %d) after %d revision(s)",
            best_report.overall, settings.reflection_threshold, iters,
        )

    # Persist the chosen script back in place (overwrite) + a report artifact.
    store.save_json("script.json", best_script)
    reflection_path = store.save_json(
        "reflection.json",
        {
            "iterations": iters,
            "final_overall": best_report.overall,
            "threshold": settings.reflection_threshold,
            "met_threshold": best_report.overall >= settings.reflection_threshold,
            "report": best_report.model_dump(),
        },
    )
    if best_report.overall < settings.reflection_threshold:
        warnings.append(
            f"reflection: best story score {best_report.overall} stayed below threshold "
            f"{settings.reflection_threshold} after {iters} revision(s); used best version"
        )

    return {"reflection_path": reflection_path, "warnings": warnings}
