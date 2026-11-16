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
    return database_url.replace("+asyncpg", "").replace("+psycopg", "").replace("ssl=", "sslmode=")


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


_PG_TABLES = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")
_SQLITE_TABLES = ("writes", "checkpoints")


async def purge_checkpoints(settings: Settings, job_ids: list[str]) -> int:
    """Delete saved graph state (scripts, dialogue) for expired jobs, incl. resume threads."""
    if not job_ids:
        return 0
    url = settings.database_url
    if url.startswith("postgresql"):
        import psycopg

        n = 0
        async with await psycopg.AsyncConnection.connect(_psycopg_url(url), autocommit=True) as conn:
            for table in _PG_TABLES:
                exists = await (await conn.execute("SELECT to_regclass(%s)", (table,))).fetchone()
                if not exists or exists[0] is None:
                    continue
                for jid in job_ids:
                    cur = await conn.execute(
                        f"DELETE FROM {table} WHERE thread_id = %s OR thread_id LIKE %s", (jid, f"{jid}:r:%")
                    )
                    n += cur.rowcount or 0
        return n
    if url.startswith("sqlite"):
        import asyncio
        import sqlite3

        path = settings.artifact_root / "checkpoints.sqlite"
        if not path.exists():
            return 0

        def _purge() -> int:
            n = 0
            with sqlite3.connect(path) as conn:
                tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                for table in _SQLITE_TABLES:
                    if table not in tables:
                        continue
                    for jid in job_ids:
                        n += conn.execute(
                            f"DELETE FROM {table} WHERE thread_id = ? OR thread_id LIKE ?", (jid, f"{jid}:r:%")
                        ).rowcount
            return n

        return await asyncio.to_thread(_purge)
    return 0


async def checkpoint_job_ids(settings: Settings) -> set[str]:
    """Job ids that still have saved graph state (resume threads collapse to their job)."""
    url = settings.database_url
    threads: list[str] = []
    if url.startswith("postgresql"):
        import psycopg

        async with await psycopg.AsyncConnection.connect(_psycopg_url(url), autocommit=True) as conn:
            for table in _PG_TABLES:
                exists = await (await conn.execute("SELECT to_regclass(%s)", (table,))).fetchone()
                if exists and exists[0] is not None:
                    rows = await (await conn.execute(f"SELECT DISTINCT thread_id FROM {table}")).fetchall()
                    threads += [r[0] for r in rows]
    elif url.startswith("sqlite"):
        import asyncio
        import sqlite3

        path = settings.artifact_root / "checkpoints.sqlite"
        if path.exists():
            def _list() -> list[str]:
                with sqlite3.connect(path) as conn:
                    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    return [r[0] for t in _SQLITE_TABLES if t in tables
                            for r in conn.execute(f"SELECT DISTINCT thread_id FROM {t}")]
            threads = await asyncio.to_thread(_list)
    return {t.split(":", 1)[0] for t in threads if t}
