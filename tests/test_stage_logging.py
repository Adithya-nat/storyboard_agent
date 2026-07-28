"""Issue 2: EVERY pipeline stage emits at least one captured log line.

script_writer / reflection / designer previously logged nothing on their happy
paths, so the front end showed "No logs yet for this stage." for them (image_gen
+ assembler logged, the other three were silent). This is the regression guard:
a full mock run — with the same DBLogHandler main.py's lifespan attaches in prod
(runner.py does NOT attach it) — must capture >=1 job_logs row for EVERY stage,
and the reflection stage must surface the judge SCORE the user specifically asked
to see.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.observability import DBLogHandler, LogDrainer
from app.observability.logs import reset_log_queue

_STAGES = ("script_writer", "reflection", "designer", "image_gen", "assembler")


async def test_every_stage_emits_at_least_one_log(monkeypatch):
    # Hermetic: stub ffmpeg exactly like test_pipeline (no real video encode).
    async def _fake_assemble(timeline, output_path):
        Path(output_path).write_bytes(b"FAKE-MP4")
        return output_path

    monkeypatch.setattr("app.agents.assembler.assemble_video", _fake_assemble)
    from app.workers.runner import run_job

    reset_log_queue()
    # Attach the stdlib->job_logs bridge exactly as main.py's lifespan does; the
    # runner alone does not. main.py also runs configure_logging() (root=INFO);
    # under run_job in tests it isn't called, so pin the "app" logger to INFO or
    # the INFO records are filtered before reaching the handler (as in prod).
    handler = DBLogHandler()
    handler.setLevel(logging.INFO)
    app_logger = logging.getLogger("app")
    prev_level = app_logger.level
    app_logger.addHandler(handler)
    app_logger.setLevel(logging.INFO)
    try:
        async with get_sessionmaker()() as s:
            job = await repo.create_job(s, "a robot waters a plant", 30)
        await run_job(job.id, "a robot waters a plant", 30)
    finally:
        app_logger.removeHandler(handler)
        app_logger.setLevel(prev_level)

    # Logs are best-effort via the queue; drain explicitly (no lifespan drainer here).
    await LogDrainer(get_sessionmaker(), batch=1000).drain_once()
    async with get_sessionmaker()() as s:
        logs = await repo.get_logs(s, job.id)

    stages_with_logs = {lg.stage for lg in logs}
    for stage in _STAGES:
        assert stage in stages_with_logs, f"stage {stage!r} emitted NO captured logs"

    # The user's specific complaint: the judge/reflection score must be visible.
    reflection_msgs = [lg.message for lg in logs if lg.stage == "reflection"]
    assert any("judge scored" in m for m in reflection_msgs), (
        "reflection stage did not surface the judge score"
    )
