"""W4: token capture (collector + adapters), persistence, API exposure, migration."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.adapters.llm import LLMAdapter
from app.core.config import Settings
from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.main import app
from app.observability import usage
from app.observability.context import bind_stage, reset_stage
from app.observability.usage import UsageCollector, bind_collector, get_collector, reset_collector
from app.schemas import Script


class _FakeUsageMeta:
    def __init__(self, prompt=None, candidates=None, total=None):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.total_token_count = total


# --------------------------------------------------------------------------- #
# Collector + record_usage (pure, no DB)
# --------------------------------------------------------------------------- #
def test_record_usage_parses_fields():
    collector = UsageCollector()
    token = bind_collector(collector)
    try:
        usage.record_usage(_FakeUsageMeta(prompt=10, candidates=20, total=30))
        assert collector.total_tokens == 30
        assert collector.prompt_tokens == 10
        assert collector.output_tokens == 20
    finally:
        reset_collector(token)


def test_record_usage_defensive_none_and_missing():
    collector = UsageCollector()
    token = bind_collector(collector)
    try:
        usage.record_usage(None)  # no metadata -> no-op
        usage.record_usage(_FakeUsageMeta())  # all None -> 0
        # total absent -> derive from prompt+candidates
        usage.record_usage(_FakeUsageMeta(prompt=5, candidates=7, total=None))
        assert collector.total_tokens == 12
    finally:
        reset_collector(token)


def test_record_usage_handles_non_numeric_fields():
    collector = UsageCollector()
    token = bind_collector(collector)
    try:
        usage.record_usage(_FakeUsageMeta(prompt="abc", candidates=None, total="xyz"))
        assert collector.total_tokens == 0  # _safe_int swallows garbage -> 0
    finally:
        reset_collector(token)


def test_record_is_noop_without_collector():
    # No collector bound (unit tests / eval harness) -> silent no-op, never raises.
    assert get_collector() is None
    usage.record(total=999)  # must not blow up


def test_per_stage_attribution_via_contextvar():
    collector = UsageCollector()
    ctoken = bind_collector(collector)
    stoken = bind_stage("job-x", "image_gen")
    try:
        usage.record(prompt=1, output=2, total=3)
        usage.record(total=7)
    finally:
        reset_stage(stoken)
        reset_collector(ctoken)
    assert collector.by_stage == {"image_gen": 10}
    assert collector.total_tokens == 10


def test_collector_is_threadsafe_under_concurrency():
    """image_gen records from concurrent executor threads -> the += must not race."""
    collector = UsageCollector()
    n_threads, per = 16, 500

    def worker():
        for _ in range(per):
            collector.add(total=1, stage="image_gen")

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert collector.total_tokens == n_threads * per
    assert collector.by_stage["image_gen"] == n_threads * per


# --------------------------------------------------------------------------- #
# Adapter integration (mock token injection)
# --------------------------------------------------------------------------- #
def test_llm_adapter_records_mock_tokens_under_stage():
    settings = Settings(use_mock_providers=True, mock_token_usage=42)
    collector = UsageCollector()
    ctoken = bind_collector(collector)
    stoken = bind_stage("job-y", "script_writer")
    try:
        LLMAdapter(settings).complete_json(
            system="s", user="u", response_model=Script,
            mock_factory=lambda _u: Script(title="t", logline="l", scenes=[]),
        )
    finally:
        reset_stage(stoken)
        reset_collector(ctoken)
    assert collector.by_stage == {"script_writer": 42}


def test_llm_adapter_zero_mock_tokens_records_nothing():
    settings = Settings(use_mock_providers=True, mock_token_usage=0)
    collector = UsageCollector()
    ctoken = bind_collector(collector)
    try:
        LLMAdapter(settings).complete_json(
            system="s", user="u", response_model=Script,
            mock_factory=lambda _u: Script(title="t", logline="l", scenes=[]),
        )
    finally:
        reset_collector(ctoken)
    assert collector.total_tokens == 0


# --------------------------------------------------------------------------- #
# Repository setters
# --------------------------------------------------------------------------- #
async def test_repo_set_job_and_stage_tokens():
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
        await repo.upsert_stage_running(s, job.id, "image_gen", 4)
        await repo.set_job_tokens(s, job.id, total=1234, prompt=900, output=334)
        await repo.set_stage_tokens(s, job.id, {"image_gen": 750})

    async with get_sessionmaker()() as s:
        j = await repo.get_job(s, job.id)
        stages = await repo.get_stages(s, job.id)
    assert (j.total_tokens, j.prompt_tokens, j.output_tokens) == (1234, 900, 334)
    assert stages[0].tokens == 750


# --------------------------------------------------------------------------- #
# API exposure
# --------------------------------------------------------------------------- #
@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_status_api_exposes_tokens(client):
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
        await repo.upsert_stage_running(s, job.id, "image_gen", 4)
        await repo.set_job_tokens(s, job.id, total=555, prompt=400, output=155)
        await repo.set_stage_tokens(s, job.id, {"image_gen": 300})

    body = (await client.get(f"/storyboard/{job.id}")).json()
    assert body["total_tokens"] == 555
    assert body["prompt_tokens"] == 400
    assert body["output_tokens"] == 155
    assert body["stages"][0]["tokens"] == 300


# --------------------------------------------------------------------------- #
# Runner end-to-end: tokens persisted for a real (mock) job
# --------------------------------------------------------------------------- #
async def test_runner_persists_tokens_end_to_end(monkeypatch):
    from app.workers import runner as runner_mod

    # mock_token_usage>0 -> every mock LLM/image call records synthetic tokens.
    # artifact_root / enable_motion / enable_tts come from the test env (conftest).
    settings = Settings(use_mock_providers=True, mock_token_usage=50)
    monkeypatch.setattr(runner_mod, "get_settings", lambda: settings)

    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "a tiny story", 30)
    await runner_mod.run_job(job.id, "a tiny story", 30)

    async with get_sessionmaker()() as s:
        j = await repo.get_job(s, job.id)
        stages = await repo.get_stages(s, job.id)
    assert j.total_tokens > 0  # tokens accumulated across stages
    assert any(st.tokens and st.tokens > 0 for st in stages)  # per-stage breakdown written


# --------------------------------------------------------------------------- #
# Migration applies cleanly on SQLite
# --------------------------------------------------------------------------- #
def test_migration_0003_applies_on_sqlite(tmp_path):
    import sqlite3

    repo_root = Path(__file__).resolve().parents[1]
    db = tmp_path / "m.db"
    env = {**os.environ, "DATABASE_URL": f"sqlite+aiosqlite:///{db}"}
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        capture_output=True, text=True, env=env, cwd=str(repo_root),
    )
    assert result.returncode == 0, result.stderr

    con = sqlite3.connect(db)
    try:
        jobs_cols = {r[1] for r in con.execute("PRAGMA table_info(jobs)")}
        stage_cols = {r[1] for r in con.execute("PRAGMA table_info(job_stages)")}
    finally:
        con.close()
    assert {"total_tokens", "prompt_tokens", "output_tokens"} <= jobs_cols
    assert "tokens" in stage_cols
