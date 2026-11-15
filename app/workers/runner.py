"""Background job runner: drives the graph for one job and persists results.

Atomic acquire -> run graph (checkpointer when available) -> persist artifacts, status,
tokens, spend and the run-ledger row. Resume re-runs idempotent nodes cheaply.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from datetime import datetime, timezone
from uuid import uuid4

from app.adapters.gemini import GeminiImageAdapter
from app.adapters.llm import LLMAdapter
from app.adapters.motion_engine import MotionEngineClient
from app.adapters.music import LyriaMusicAdapter
from app.adapters.omni import OmniVideoAdapter
from app.adapters.transcribe import TranscribeAdapter
from app.adapters.tts import GeminiTTSAdapter
from app.artifacts.store import ArtifactStore
from app.core.config import get_settings
from app.core.logging import get_logger
from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.graph.build import build_graph
from app.graph.checkpoint import make_checkpointer
from app.graph.state import Deps
from app.observability.reporter import ProgressReporter
from app.observability.usage import UsageCollector, bind_collector, reset_collector
from app.schemas import JobStatus

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
) -> None:
    settings = get_settings()
    sessionmaker = get_sessionmaker()

    # Atomic acquire — the real concurrency guard. If another runner already owns
    # the job (status already 'running'), this affects 0 rows and we abort, so two
    # concurrent POST /resume calls can't race one job's artifacts.
    async with sessionmaker() as session:
        acquired = await repo.try_acquire_job(session, job_id)
    if not acquired:
        log.warning("job %s already running; runner aborting (concurrent resume?)", job_id)
        return

    # Token accounting: bind a collector for THIS job BEFORE the graph runs, so every
    # context copied by asyncio.gather/to_thread (the parallel image calls) shares it.
    style = style or settings.default_style
    language = language or settings.default_language
    started = datetime.now(timezone.utc)
    async with sessionmaker() as session:
        await repo.upsert_run(
            session, job_id, prompt=prompt, style=style, language=language,
            duration=duration, status="running", started_at=started,
        )

    collector = UsageCollector()
    usage_token = bind_collector(collector)
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
                final = await _invoke(
                    graph, init, config, resume, checkpointed=checkpointer is not None
                )
        except Exception as exc:  # noqa: BLE001 - any failure -> visible FAILED status
            log.exception("job %s failed", job_id)
            async with sessionmaker() as session:
                # Best-effort: a failed job still shows the tokens/spend it used.
                await _persist_tokens(session, job_id, collector)
                await repo.set_status(
                    session, job_id, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}"
                )
            await _finish_ledger(job_id, {}, collector, "failed", started)
            return
    finally:
        reset_collector(usage_token)

    status = await _persist(job_id, final, collector)
    await _finish_ledger(job_id, final, collector, status, started)


async def _invoke(graph, init: dict, config: dict, resume: bool, *, checkpointed: bool) -> dict:
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
    """
    if not checkpointed:
        return await graph.ainvoke(init)

    if resume:
        snapshot = await graph.aget_state(config)
        if snapshot.next:  # a node is pending -> continue from the checkpoint
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
    artifacts: list[dict] = []
    for kind in (
        "script_path",
        "reflection_path",
        "shotlist_path",
        "style_bible_path",
        "style_ref_path",
        "motion_manifest_path",
    ):
        if final.get(kind):
            artifacts.append({"kind": kind.removesuffix("_path"), "path": final[kind]})
    for f in frames:
        artifacts.append(
            {"kind": "frame", "path": f["path"], "status": f["status"], "order_index": f["order"]}
        )
    # Per-scene motion clips (motion engine). Served for free by the existing
    # /artifacts/{id} route; mapped to scenes[].video_url in get_storyboard_content.
    for c in final.get("motion_clips", []) or []:
        artifacts.append(
            {
                "kind": "motion",
                "path": c["path"],
                "status": c.get("status", "ok"),
                "order_index": c["order"],
            }
        )
    if final.get("video_path"):
        artifacts.append({"kind": "video", "path": final["video_path"]})
    if final.get("narration_path"):
        artifacts.append({"kind": "narration", "path": final["narration_path"]})
    for kind in ("publish_path", "cover_path", "music_path", "scene_audio_path"):
        if final.get(kind):
            artifacts.append({"kind": kind.removesuffix("_path"), "path": final[kind]})
    for name, paths in (final.get("cast_paths") or {}).items():
        if paths:
            artifacts.append({"kind": f"cast:{name[:26]}", "path": paths[0]})

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
