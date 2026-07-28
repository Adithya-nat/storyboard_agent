"""Async SQLAlchemy engine + session factory.

The engine is created lazily from settings so tests can override DATABASE_URL
(e.g. aiosqlite) before the first connection.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings


@lru_cache
def get_engine() -> AsyncEngine:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, future=True)
    if settings.database_url.startswith("sqlite"):
        # SQLite (local dev + tests) serializes writers; without these, a poll
        # read overlapping a write raises "database is locked". WAL lets a reader
        # and a writer coexist; busy_timeout makes a writer wait, not fail. Harmless
        # for prod, which is Postgres. (No-op for Postgres — gated on the URL.)
        @event.listens_for(engine.sync_engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA busy_timeout=5000")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

    return engine


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one session per request."""
    async with get_sessionmaker()() as session:
        yield session
