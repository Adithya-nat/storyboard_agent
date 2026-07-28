"""Background job runner: drives the graph for one job and persists results.

    try_acquire (atomic) ─▶ run graph (checkpointer, thread_id=job_id) ─▶ persist
            │ already running                     │ on exception
            └─▶ abort (concurrent resume)         └─▶ set FAILED (+ error)

Resume: POST /resume calls run_job(resume=True). The graph continues from the
last checkpoint if one is pending; otherwise it re-invokes from the start, which
is correct AND cheap because every node is idempotent (load-or-produce) — a
resumed image_gen skips frames already on disk instead of re-billing them.

Runs as a fire-and-forget task (FastAPI BackgroundTasks). It owns its OWN DB
session — never the request session. Keeps DB writes here, out of the agents, so
agents stay pure + unit-testable.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from uuid import uuid4

from app.adapters.gemini import GeminiImageAdapter
from app.adapters.llm import LLMAdapter
from app.adapters.motion_engine import MotionEngineClient
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


async def run_job(job_id: str, prompt: str, duration: int, resume: bool = False) -> None:
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
                # Per-stage status + log sink. Owns its own sessionmaker (never the
                # request session), like the rest of the runner's DB writes.
                reporter=ProgressReporter(sessionmaker),
                # External motion engine — wired only when enabled (else the motion
                # node isn't in the graph and this is unused).
                motion=MotionEngineClient(settings) if settings.motion_engine_enabled else None,
            )
            config = {"configurable": {"thread_id": job_id}}
            init = {
                "job_id": job_id,
                "prompt": prompt,
                "duration": duration,
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
                # Best-effort: a failed job still shows the tokens it spent.
                await _persist_tokens(session, job_id, collector)
                await repo.set_status(
                    session, job_id, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}"
                )
            return
    finally:
        reset_collector(usage_token)

    await _persist(job_id, final, collector)


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
    """Persist the job total + per-stage token breakdown. Best-effort: token accounting
    must NEVER break a job's status handling."""
    try:
        await repo.set_job_tokens(
            session,
            job_id,
            total=collector.total_tokens,
            prompt=collector.prompt_tokens,
            output=collector.output_tokens,
        )
        if collector.by_stage:
            await repo.set_stage_tokens(session, job_id, collector.by_stage)
    except Exception:  # noqa: BLE001 - degrade, don't die
        log.warning("token persistence failed for job %s", job_id, exc_info=True)


async def _persist(job_id: str, final: dict, collector: UsageCollector | None = None) -> None:
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
            if degraded or motion_degraded
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
