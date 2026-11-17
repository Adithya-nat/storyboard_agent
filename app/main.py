"""FastAPI entrypoint.

Startup runs the orphaned-job reaper: because jobs run in-process, a restart
(e.g. uvicorn --reload) would otherwise strand 'running' jobs forever. The reaper
flips them to 'interrupted' so clients get an explicit terminal state.

Startup also wires the observability log bridge: a DBLogHandler attached to the
``app`` logger (so app.* logs — including the gemini adapter's re-roll lines —
land in job_logs, but sqlalchemy/httpx/uvicorn noise does not) + the async
LogDrainer that batch-writes the captured queue into the DB.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.routes import limiter
from app.api.routes import router as storyboard_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.api.usage import router as usage_router
from app.api.live import router as live_router
from app.api.account import router as account_router
from app.api.controls import router as controls_router
from app.api.library import router as library_router
from app.core.images import raqm_available
from app.core.keys import RedactKeysFilter
from app.core.model_check import validate_models
from app.graph.checkpoint import setup_checkpointer
from app.workers.retention import sweep_once
from app.observability import DBLogHandler, LogDrainer, configure_log_queue

log = get_logger(__name__)


async def _ttl_sweeper(settings) -> None:
    """Deletes expired jobs on time, sleeping until the next expiry (so Neon can scale to zero)."""
    while True:
        try:
            delay = await sweep_once(settings)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - never let the sweeper loop die silently
            log.exception("retention sweep failed; will retry")
            delay = settings.job_sweep_interval_seconds
        await asyncio.sleep(delay)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()

    async with get_sessionmaker()() as session:
        reaped = await repo.reap_orphaned_jobs(session)
    if reaped:
        log.warning("Reaped %d orphaned job(s) -> interrupted on startup", reaped)

    # Observability bridge: capture app.* logs into job_logs via a bounded queue +
    # one async drainer. Attached to the `app` logger (NOT root) so it grabs the
    # adapter/agent logs but not framework noise. OFF -> pull/SSE still serve rows,
    # they just won't auto-capture incidental logs.
    drainer: LogDrainer | None = None
    handler: DBLogHandler | None = None
    if settings.enable_log_streaming:
        configure_log_queue(settings.log_queue_max)
        handler = DBLogHandler()
        handler.addFilter(RedactKeysFilter())
        handler.setLevel(getattr(logging, settings.log_capture_level.upper(), logging.INFO))
        logging.getLogger("app").addHandler(handler)
        drainer = LogDrainer(
            get_sessionmaker(),
            batch=settings.log_drain_batch,
            interval=settings.log_poll_interval_seconds,
        )
        drainer.start()

    # Create the checkpointer tables ONCE, now, while startup is quiet — the
    # Postgres saver's CREATE INDEX CONCURRENTLY can't run alongside live request
    # traffic. Best-effort + timeout: if it fails, jobs still run and resume via
    # idempotency (the load-bearing layer), just without graph-level checkpoints.
    try:
        await asyncio.wait_for(setup_checkpointer(settings), timeout=30)
    except Exception as exc:  # noqa: BLE001 - never block startup on the checkpointer
        log.warning(
            "checkpointer setup skipped (%s); resume falls back to idempotent re-run", exc
        )

    app.state.model_check = {"checked": False, "reason": "disabled"}
    if settings.validate_models_on_startup:
        app.state.model_check = await validate_models(settings)

    # Background TTL sweeper: reclaim expired jobs' DB rows + artifact files.
    sweeper_task = asyncio.create_task(_ttl_sweeper(settings))

    try:
        yield
    finally:
        sweeper_task.cancel()
        try:
            await sweeper_task
        except asyncio.CancelledError:
            pass
        # Flush + stop the drainer so a clean shutdown doesn't lose queued logs.
        if drainer is not None:
            await drainer.astop()
        if handler is not None:
            logging.getLogger("app").removeHandler(handler)


app = FastAPI(title="Creator Agent", version="0.2.0", lifespan=lifespan)

# CORS: the Next.js frontend runs on a different origin, so the browser requires
# these headers to call the API at all (status, SSE, logs, and file downloads are
# all plain GETs; POST create carries a JSON body + Turnstile token).
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting: register the limiter + its 429 handler (the limiter itself lives
# in app.api.routes, next to the decorated create endpoint).
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(storyboard_router)
app.include_router(usage_router)
app.include_router(live_router)
app.include_router(account_router)
app.include_router(controls_router)
app.include_router(library_router)


@app.get("/health")
async def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "mock_mode": s.mock_mode,
        "auth_mode": s.auth_mode,
        "key_mode": s.key_mode,
        "models": getattr(app.state, "model_check", {"checked": False}),
        "tamil_shaping": raqm_available(),
    }
