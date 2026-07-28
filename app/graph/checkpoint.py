"""LangGraph checkpointer factory.

Picks the saver that matches the configured database so resume persistence works
identically in dev/tests and prod:

    postgresql*  -> AsyncPostgresSaver   (prod; uses psycopg)
    sqlite*      -> AsyncSqliteSaver      (dev/tests; FILE-backed, not :memory:)
    other        -> InMemorySaver         (last-resort; no cross-process resume)

⚠ Postgres setup() runs `CREATE INDEX CONCURRENTLY`, which blocks until EVERY
in-flight transaction finishes. In a live app (GET polling holds open read
transactions) that wait never ends — running it per-job deadlocks the whole
pipeline. So setup() runs exactly ONCE at startup (`setup_checkpointer`, called
from the app lifespan, before any request traffic); `make_checkpointer` only
*connects* per job (fast) and writes to tables that already exist.

The checkpointer is layered ON TOP of per-node artifact idempotency — idempotency
is what guarantees correctness; the checkpointer just skips already-finished nodes
at the graph level. So callers treat it as best-effort (run without it on failure).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)


def _psycopg_url(database_url: str) -> str:
    """The app talks to Postgres via SQLAlchemy+asyncpg; the LangGraph postgres
    saver talks via psycopg. Strip the driver tag so psycopg accepts the DSN."""
    return database_url.replace("+asyncpg", "").replace("+psycopg", "")


@asynccontextmanager
async def make_checkpointer(settings: Settings) -> AsyncIterator[object]:
    url = settings.database_url
    if url.startswith("postgresql"):
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        # No setup() here — see module docstring. Connect only; the tables were
        # created once at startup by setup_checkpointer().
        async with AsyncPostgresSaver.from_conn_string(_psycopg_url(url)) as saver:
            yield saver
    elif url.startswith("sqlite"):
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        # sqlite setup() is plain CREATE TABLE IF NOT EXISTS (no CONCURRENTLY),
        # so it's instant and safe to run per-use.
        path = settings.artifact_root / "checkpoints.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
            await saver.setup()
            yield saver
    else:
        from langgraph.checkpoint.memory import InMemorySaver

        log.warning("No DB-backed checkpointer for %s; using in-memory (no resume).", url)
        yield InMemorySaver()


async def setup_checkpointer(settings: Settings) -> None:
    """Create the Postgres checkpoint tables/indexes ONCE, at app startup.

    Must run when no other transactions are in flight: the saver's migrations use
    `CREATE INDEX CONCURRENTLY`, which blocks until every concurrent transaction
    finishes. Startup (before the app serves requests) is the only safe window —
    running it per-job deadlocks against the very GET-polling it serves. No-op for
    sqlite/memory (those set themselves up in make_checkpointer).
    """
    if not settings.database_url.startswith("postgresql"):
        return
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(_psycopg_url(settings.database_url)) as saver:
        await saver.setup()
