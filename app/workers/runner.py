"""Background job runner: drives the graph for one job and persists results.

Atomic acquire -> run graph (checkpointer when available) -> persist artifacts, status,
tokens, spend and the run-ledger row. Resume re-runs idempotent nodes cheaply.

Pause: the runner cancels the graph the moment a pause is requested and parks the job
as ``paused`` (see app/workers/control.py). Paid calls already submitted finish in the
background; the runner books their spend once they land. A breakpoint parks the job the
same way, before its stage starts.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, suppress
from datetime import datetime, timezone
from uuid import uuid4

from app.adapters.gemini import GeminiImageAdapter
from app.adapters.llm import LLMAdapter
from app.adapters.motion_engine import MotionEngineClient
from app.adapters.music import LyriaMusicAdapter
from app.adapters.omni import OmniVideoAdapter
from app.adapters.transcribe import TranscribeAdapter
from app.adapters.tts import GeminiTTSAdapter
from app.artifacts.rows import artifact_rows
from app.artifacts.store import ArtifactStore
from app.core.config import get_settings
from app.core.keys import bind_key, redact, release_key
from app.core.model_check import validate_models
from app.core.logging import get_logger
from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.graph.build import build_graph
from app.graph.checkpoint import make_checkpointer
from app.graph.state import Deps
from app.observability.reporter import ProgressReporter
from app.observability.usage import UsageCollector, bind_collector, reset_collector
from app.schemas import JobStatus
from app.workers.control import JobPaused, RunControl, clear_pause, forget, settle

log = get_logger(__name__)


async def run_job(
    job_id: str,
    prompt: str,
    duration: int,
    resume: bool = False,
    style: str | None = None,
    language: str | None = None,
    allow_motion: bool = True,
    allow_stills: bool = False,
    api_key: str | None = None,
    user_id: str | None = None,
    fresh: bool = False,
) -> None:
    key_token = bind_key(api_key)
    try:
        await _run_job(job_id, prompt, duration, resume, style, language, allow_motion, allow_stills,
                       api_key, user_id, fresh)
    finally:
        release_key(api_key, key_token)


async def _run_job(
    job_id: str,
    prompt: str,
    duration: int,
    resume: bool,
    style: str | None,
    language: str | None,
    allow_motion: bool,
    allow_stills: bool,
    api_key: str | None,
    user_id: str | None,
    fresh: bool = False,
) -> None:
    settings = get_settings()
    if api_key:  # BYOK: this job's own settings copy; the key is never written anywhere
        settings = settings.model_copy(update={"gemini_api_key": api_key})
        if settings.validate_models_on_startup and not settings.mock_mode:
            await validate_models(settings)
    sessionmaker = get_sessionmaker()

    # Atomic acquire — the real concurrency guard. If another runner already owns
    # the job (status already 'running'), this affects 0 rows and we abort, so two
    # concurrent POST /resume calls can't race one job's artifacts.
    async with sessionmaker() as session:
        acquired = await repo.try_acquire_job(session, job_id)
    if not acquired:
        log.warning("job %s already running; runner aborting (concurrent resume?)", job_id)
        return
    clear_pause(job_id)  # a fresh/resumed run starts unpaused (the DB flag was cleared on acquire)

    # Token accounting: bind a collector for THIS job BEFORE the graph runs, so every
    # context copied by asyncio.gather/to_thread (the parallel image calls) shares it.
    style = style or settings.default_style
    language = language or settings.default_language
    started = datetime.now(timezone.utc)
    async with sessionmaker() as session:
        await repo.upsert_run(
            session, job_id, prompt=prompt, style=style, language=language, user_id=user_id,
            duration=duration, status="running", started_at=started,
        )

    collector = UsageCollector()
    usage_token = bind_collector(collector)
    paused: JobPaused | None = None
    try:
        try:
            store = ArtifactStore(settings.artifact_root, job_id)
            deps = Deps(
                settings=settings,
                store=store,
                llm=LLMAdapter(settings),
                image=GeminiImageAdapter(settings),
                tts=GeminiTTSAdapter(settings),
                transcriber=TranscribeAdapter(settings),
                music=LyriaMusicAdapter(settings) if settings.music_enabled else None,
                # Per-stage status + log sink. Owns its own sessionmaker (never the
                # request session), like the rest of the runner's DB writes.
                reporter=ProgressReporter(sessionmaker),
                # External motion engine — wired only when enabled (else the motion
                # node isn't in the graph and this is unused).
                # Budget degrade mode runs without the (paid) motion engine.
                motion=MotionEngineClient(settings)
                if settings.video_provider == "engine" and settings.motion_engine_enabled and allow_motion
                else None,
                video=OmniVideoAdapter(settings) if settings.video_provider == "omni" and allow_motion else None,
                allow_stills=allow_stills,
                control=RunControl(job_id, sessionmaker),
            )
            config = {"configurable": {"thread_id": job_id}}
            init = {
                "job_id": job_id,
                "prompt": prompt,
                "duration": duration,
                "style": style,
                "language": language,
                "warnings": [],
            }
            # Best-effort checkpointer: if it can't be acquired, run WITHOUT one. The
            # job must never hang or fail because of the checkpointer — idempotent
            # nodes still make resume correct (via a fresh re-run).
            async with AsyncExitStack() as stack:
                try:
                    checkpointer = await stack.enter_async_context(make_checkpointer(settings))
                except Exception as exc:  # noqa: BLE001 - degrade, don't die
                    log.warning(
                        "checkpointer unavailable (%s); running without it (resume via idempotency)",
                        exc,
                    )
                    checkpointer = None
                graph = build_graph(deps, checkpointer=checkpointer)

                async def settle_pause() -> None:
                    # Every write this runner owes happens BEFORE the job becomes
                    # resumable, so a fast resume's new run can't be overwritten.
                    if deps.control.held is not None:  # written here: the graph can't be cancelled mid-write
                        await deps.reporter.stage_pause(job_id, *deps.control.held)
                    async with sessionmaker() as session:
                        await _persist_tokens(session, job_id, collector)
                    await _finish_ledger(job_id, {}, collector, JobStatus.PAUSED.value, started)

                final, paused = await _drive(
                    graph, init, config, resume, checkpointed=checkpointer is not None,
                    control=deps.control, settle_pause=settle_pause, fresh=fresh,
                )
        except Exception as exc:  # noqa: BLE001 - any failure -> visible FAILED status
            log.exception("job %s failed", job_id)
            async with sessionmaker() as session:
                # Best-effort: a failed job still shows the tokens/spend it used.
                await _persist_tokens(session, job_id, collector)
                await repo.set_status(
                    session, job_id, JobStatus.FAILED, error=redact(f"{type(exc).__name__}: {exc}")
                )
            await _finish_ledger(job_id, {}, collector, "failed", started)
            return
        if paused is not None:
            # Parked: a resume may already own this job, so only additive writes from here.
            log.info("job %s paused at %s: %s", job_id, paused.stage, paused.message)
            await _book_late_spend(job_id, collector)
            return
    finally:
        reset_collector(usage_token)
        if paused is None:  # a parked run already released its in-memory state before parking
            forget(job_id)

    status = await _persist(job_id, final, collector)
    await _finish_ledger(job_id, final, collector, status, started)


async def _drive(
    graph, init: dict, config: dict, resume: bool, *, checkpointed: bool,
    control: RunControl | None = None, settle_pause=None, fresh: bool = False,
) -> tuple[dict, JobPaused | None]:
    """Run the graph; on a pause, park the job. Returns (final_state, pause_or_None).

    Before parking, the runner finishes every write it owes (``settle_pause``) and drops
    its in-memory state: once ``park_job`` commits, a resume can start a new runner at
    any moment, and only additive writes of this one may land after that.

    If the pause was withdrawn before the job parked, ``park_job`` refuses (the flag is
    gone) and we simply continue: as a resume, so finished nodes and units are skipped
    and nothing is re-billed.
    """
    job_id = init["job_id"]
    for _ in range(20):  # bounded: a withdraw/re-pause tug-of-war can't spin forever
        # Paid calls an earlier pause left running land on disk first, so this run
        # sees them as done instead of paying for them again.
        await settle(job_id)
        try:
            run = _invoke(graph, init, config, resume, checkpointed=checkpointed, fresh=fresh)
            return await _until_paused(run, control), None
        except JobPaused as paused:
            if settle_pause is not None:
                await settle_pause()
            forget(job_id)  # signals are re-created on demand; units still landing stay listed
            async with get_sessionmaker()() as session:
                if await repo.park_job(session, job_id):
                    return {}, paused
                await repo.upsert_run(session, job_id, status="running")  # back from "paused"
            clear_pause(job_id)
            if control is not None:
                control.reset()  # else its cached "paused" answer re-pauses every retry
            log.info("job %s: pause withdrawn while draining; continuing from %s", job_id, paused.stage)
            resume, fresh = True, False
    raise RuntimeError("pause/resume toggled too many times in one run; resume it again")


async def _until_paused(run, control: RunControl | None) -> dict:
    """Await the graph, cancelling it the moment a pause is requested (JobPaused)."""
    if control is None:
        return await run
    task = asyncio.ensure_future(run)
    watch = asyncio.ensure_future(control.until_requested())
    try:
        done, _ = await asyncio.wait({task, watch}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        watch.cancel()
        if not task.done():
            task.cancel()  # a pause, or this runner itself was cancelled
            with suppress(asyncio.CancelledError, JobPaused):
                await task
    if task in done or not task.cancelled():
        return task.result()  # finished (or failed, or hit a breakpoint) first
    raise JobPaused(control.stage, "Paused partway through; finished work is kept")


async def _book_late_spend(job_id: str, collector: UsageCollector) -> None:
    """After parking: wait for paid calls still in flight, then record what they cost.
    Spend rows are additive and job cost is their sum, so this is safe beside a resume."""
    await settle(job_id)
    events = collector.drain_events()
    if not events:
        return
    try:
        async with get_sessionmaker()() as session:
            await repo.add_spend_events(session, job_id, events)
            await repo.set_job_cost(session, job_id, by_stage={e.stage: 0.0 for e in events if e.stage})
    except Exception:  # noqa: BLE001 - degrade, don't die
        log.warning("late spend for paused job %s not recorded", job_id, exc_info=True)


async def _invoke(graph, init: dict, config: dict, resume: bool, *, checkpointed: bool, fresh: bool = False) -> dict:
    """Run (or resume) the graph.

    Without a checkpointer: a plain idempotent invoke covers both fresh runs and
    resumes (re-running reuses every existing artifact, regenerates only the gaps).

    With a checkpointer:
    - Fresh job: invoke from the start on thread_id=job_id.
    - Resume with a PENDING checkpoint (process died mid-graph): continue in place
      with ``None`` — the checkpointer skips finished nodes; idempotency handles
      the half-done image_gen loop.
    - Resume with a finished/absent checkpoint: re-run from the start on a NEW
      thread so LangGraph actually re-executes; idempotent nodes reuse every
      existing artifact, so only the deleted/missing pieces are regenerated.
    - ``fresh`` (a recut): always that new-thread re-run, since the saved checkpoint may sit
      past a stage whose files the recut just replaced.
    """
    if not checkpointed:
        return await graph.ainvoke(init)

    if resume:
        snapshot = await graph.aget_state(config)
        if snapshot.next and not fresh:  # a node is pending -> continue from the checkpoint
            return await graph.ainvoke(None, config)
        fresh = {"configurable": {"thread_id": f"{init['job_id']}:r:{uuid4().hex[:8]}"}}
        return await graph.ainvoke(init, fresh)
    return await graph.ainvoke(init, config)


async def _persist_tokens(session, job_id: str, collector: UsageCollector) -> None:
    """Persist tokens + any undrained spend. Best-effort: never breaks status handling."""
    try:
        await repo.set_job_tokens(
            session, job_id,
            total=collector.total_tokens, prompt=collector.prompt_tokens, output=collector.output_tokens,
        )
        if collector.by_stage:
            await repo.set_stage_tokens(session, job_id, collector.by_stage)
        events = collector.drain_events()
        if events:
            await repo.add_spend_events(session, job_id, events)
        await repo.set_job_cost(session, job_id, collector.cost_usd, collector.cost_by_stage)
    except Exception:  # noqa: BLE001 - degrade, don't die
        log.warning("usage persistence failed for job %s", job_id, exc_info=True)


async def _finish_ledger(job_id: str, final: dict, collector: UsageCollector, status: str, started) -> None:
    """Write the run's final dashboard row (best-effort)."""
    try:
        clips = final.get("motion_clips") or []
        omni = sum(1 for c in clips if "omni" in str(c.get("renderer") or c.get("model") or "").lower())
        scenes = len(final.get("frames") or [])
        judge = None
        if final.get("reflection_path"):
            import json

            with open(final["reflection_path"], encoding="utf-8") as fh:
                judge = json.load(fh).get("final_overall")
        film = None
        if final.get("film_review_path"):
            import json

            with open(final["film_review_path"], encoding="utf-8") as fh:
                film = json.load(fh).get("score")
        async with get_sessionmaker()() as session:
            by_cat = await repo.job_spend_by_category(session, job_id)
            events = await repo.spend_events_for_job(session, job_id)
            await repo.upsert_run(
                session, job_id,
                status=status,
                cost_usd=round(sum(by_cat.values()), 6),
                cost_by_category=by_cat,
                total_tokens=collector.total_tokens,
                video_seconds=sum(float(e.units.get("seconds", 0)) for e in events if e.category == "video"),
                images=sum(1 for e in events if e.category == "image"),
                scenes=scenes,
                omni_scenes=omni,
                fallback_scenes=max(0, len(clips) - omni) if clips else 0,
                judge_score=judge,
                film_score=film,
                runtime_seconds=(datetime.now(timezone.utc) - started).total_seconds(),
                models=sorted({e.model for e in events}),
                finished_at=datetime.now(timezone.utc),
            )
    except Exception:  # noqa: BLE001
        log.warning("run ledger update failed for job %s", job_id, exc_info=True)


async def _persist(job_id: str, final: dict, collector: UsageCollector | None = None) -> str:
    sessionmaker = get_sessionmaker()
    frames = final.get("frames", []) or []
    warnings = final.get("warnings", []) or []
    degraded = any(f["status"] == "degraded" for f in frames)

    # Build the full artifact set, then REPLACE (clear-then-insert) so a resumed
    # job re-persisting the same artifacts doesn't duplicate rows.
    artifacts = artifact_rows(final)

    async with sessionmaker() as session:
        await repo.replace_artifacts(session, job_id, artifacts)
        if collector is not None:
            await _persist_tokens(session, job_id, collector)
        motion_degraded = any(
            c.get("status") == "degraded"
            for c in final.get("motion_clips", []) or []
        )
        status = (
            JobStatus.COMPLETED_WITH_WARNINGS
            if degraded or motion_degraded or final.get("stills_used")
            else JobStatus.COMPLETED
        )
        await repo.set_status(session, job_id, status, warnings=warnings)
    log.info(
        "job %s -> %s (%d frames, %d warnings)",
        job_id,
        status.value,
        len(frames),
        len(warnings),
    )
    return status.value
