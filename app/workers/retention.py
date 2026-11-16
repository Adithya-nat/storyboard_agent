"""24-hour retention: everything a job makes is deleted 24h after creation."""

from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings
from app.core.logging import get_logger
from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.graph.checkpoint import checkpoint_job_ids, purge_checkpoints

log = get_logger(__name__)

MIN_SLEEP = 60.0
_JOB_DIR = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


async def _sweep_leftovers(settings: Settings) -> int:
    """Retry cleanup a previous sweep couldn't finish: files and checkpoints whose job row is gone."""
    root = Path(settings.artifact_root)
    # List first, then read live ids: a job row always exists before its files and checkpoints.
    dirs = [d for d in root.iterdir() if d.is_dir() and _JOB_DIR.match(d.name)] if root.exists() else []
    try:
        threads = await checkpoint_job_ids(settings)
    except Exception:  # noqa: BLE001
        log.warning("could not list checkpoints for cleanup", exc_info=True)
        threads = set()
    async with get_sessionmaker()() as session:
        live = await repo.job_ids(session)
    stale_dirs = [d for d in dirs if d.name not in live]
    for d in stale_dirs:
        shutil.rmtree(d, ignore_errors=True)
    stale_threads = sorted(threads - live)
    if stale_threads:
        await purge_checkpoints(settings, stale_threads)
    return len(stale_dirs) + len(stale_threads)


async def sweep_once(settings: Settings) -> float:
    """Purge expired jobs (rows, files, graph checkpoints); return seconds until the next expiry."""
    async with get_sessionmaker()() as session:
        expired = await repo.delete_expired_jobs(session, settings.job_ttl_seconds, settings.paused_ttl_seconds)
    for job_id in expired:
        shutil.rmtree(Path(settings.artifact_root) / job_id, ignore_errors=True)
    if expired:
        try:
            await purge_checkpoints(settings, expired)
        except Exception:  # noqa: BLE001 - best effort; rows and files are already gone
            log.warning("checkpoint purge failed for %d job(s)", len(expired), exc_info=True)
        log.info("retention: deleted %d expired job(s)", len(expired))
    try:
        if n := await _sweep_leftovers(settings):
            log.info("retention: removed %d leftover file folder(s)/checkpoint(s)", n)
    except Exception:  # noqa: BLE001 - retried on the next sweep
        log.warning("leftover cleanup failed; will retry", exc_info=True)
    async with get_sessionmaker()() as session:
        nxt = await repo.next_expiry(session, settings.job_ttl_seconds)
    cap = settings.job_sweep_interval_seconds
    if nxt is None:
        return cap
    wait = (nxt - datetime.now(timezone.utc)).total_seconds()
    return min(cap, max(MIN_SLEEP, wait + 1))
