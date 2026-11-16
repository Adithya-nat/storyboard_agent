"""24-hour retention: everything a job makes is deleted 24h after creation."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings
from app.core.logging import get_logger
from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.graph.checkpoint import purge_checkpoints

log = get_logger(__name__)

MIN_SLEEP = 60.0


async def sweep_once(settings: Settings) -> float:
    """Purge expired jobs (rows, files, graph checkpoints); return seconds until the next expiry."""
    async with get_sessionmaker()() as session:
        expired = await repo.delete_expired_jobs(session, settings.job_ttl_seconds)
    for job_id in expired:
        shutil.rmtree(Path(settings.artifact_root) / job_id, ignore_errors=True)
    if expired:
        try:
            await purge_checkpoints(settings, expired)
        except Exception:  # noqa: BLE001 - best effort; rows and files are already gone
            log.warning("checkpoint purge failed for %d job(s)", len(expired), exc_info=True)
        log.info("retention: deleted %d expired job(s)", len(expired))
    async with get_sessionmaker()() as session:
        nxt = await repo.next_expiry(session, settings.job_ttl_seconds)
    cap = settings.job_sweep_interval_seconds
    if nxt is None:
        return cap
    wait = (nxt - datetime.now(timezone.utc)).total_seconds()
    return min(cap, max(MIN_SLEEP, wait + 1))
