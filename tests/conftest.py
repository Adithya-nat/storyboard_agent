"""Test fixtures.

Env is configured BEFORE any app import so the lru_cached settings/engine pick up
the sqlite test DB + mock providers. Each test gets a fresh schema.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# --- configure env before importing app code -------------------------------- #
_TMP = Path(tempfile.mkdtemp(prefix="storyboard_test_"))
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TMP/'test.db'}"
os.environ["USE_MOCK_PROVIDERS"] = "true"
os.environ["GEMINI_API_KEY"] = ""
os.environ["ARTIFACT_ROOT"] = str(_TMP / "artifacts")
# Rate limiting off in tests: the shared in-memory limiter would otherwise carry
# counts across tests and could 429 a legitimate create assertion.
os.environ["RATE_LIMIT_ENABLED"] = "false"
# Hermetic by default: integration tests stub the concat assembler. Ken Burns
# (real ffmpeg) is exercised by dedicated, ffmpeg-gated tests instead.
os.environ["ENABLE_MOTION"] = "false"
# Narration is ON by default in prod (Workstream 5), but the existing pipeline tests
# assert the silent exact-duration path; pin it OFF so they keep that path. Dedicated
# narration tests opt back in by constructing Settings(enable_tts=True) explicitly.
os.environ["ENABLE_TTS"] = "false"
# Fast SSE polling so streaming tests finish in tens of ms, not seconds.
os.environ["LOG_POLL_INTERVAL_SECONDS"] = "0.02"
# Motion engine OFF in tests: the graph must not grow a 'motion' node or reach out
# to localhost:8090, even if a developer's real .env enables it (os.environ wins
# over .env in pydantic-settings). Belt-and-braces with the code default (False).
os.environ["MOTION_ENGINE_ENABLED"] = "false"

from app.db.models import Base  # noqa: E402
from app.db.session import get_engine, get_sessionmaker  # noqa: E402
from app.observability.logs import configure_log_queue, reset_log_queue  # noqa: E402


@pytest.fixture(autouse=True)
async def _schema():
    """Fresh tables for every test."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture(autouse=True)
def _fresh_log_queue():
    """Reset the module-global log queue between tests so a leftover item (which
    references a now-dropped job_id) can't leak into another test."""
    configure_log_queue(10_000)
    yield
    reset_log_queue()


@pytest.fixture
async def session():
    async with get_sessionmaker()() as s:
        yield s


@pytest.fixture
def artifact_root() -> Path:
    return Path(os.environ["ARTIFACT_ROOT"])
