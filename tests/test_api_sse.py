"""GET /storyboard/{id}/events — the SSE live stream.

Discipline (advisor): SEED job_stages/job_logs directly and assert the endpoint
replays/streams/closes. Do NOT race a live background runner — that gives flaky
timing AND SQLite writer/reader contention. Live-tail is exercised with ONE
controlled insert at a known time. Every read loop is bounded so a logic bug
fails fast instead of hanging.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.main import app
from app.observability.logs import LogItem
from app.schemas import JobStatus

_MAX_LINES = 800  # hard cap so a broken stream fails the test instead of hanging


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _read_until_done(client, url: str, headers: dict | None = None) -> list[str]:
    lines: list[str] = []
    async with client.stream("GET", url, headers=headers or {}) as resp:
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            lines.append(line)
            if "done" in line or len(lines) > _MAX_LINES:
                break
    return lines


async def test_terminal_job_snapshots_replays_then_closes(client):
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
        await repo.upsert_stage_running(s, job.id, "script_writer", 1)
        await repo.flip_stage_if_running(s, job.id, "script_writer", "succeeded")
        await repo.insert_logs(
            s,
            [LogItem(job.id, "script_writer", "info", "hello-stream", datetime.now(timezone.utc))],
        )
        await repo.set_status(s, job.id, JobStatus.COMPLETED)

    text = "\n".join(await _read_until_done(client, f"/storyboard/{job.id}/events"))
    assert "event: status" in text
    assert "event: stage" in text
    assert "event: log" in text and "hello-stream" in text
    assert "event: done" in text


async def test_last_event_id_resumes_after_cursor(client):
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
        await repo.insert_logs(
            s,
            [
                LogItem(job.id, "script_writer", "info", "line-1", datetime.now(timezone.utc)),
                LogItem(job.id, "script_writer", "info", "line-2", datetime.now(timezone.utc)),
            ],
        )
        await repo.set_status(s, job.id, JobStatus.COMPLETED)
        logs = await repo.get_logs(s, job.id)
    first_id = logs[0].id

    text = "\n".join(
        await _read_until_done(
            client, f"/storyboard/{job.id}/events", headers={"Last-Event-ID": str(first_id)}
        )
    )
    assert "line-2" in text       # strictly after the cursor -> replayed
    assert "line-1" not in text   # at/before the cursor -> skipped


async def test_live_tail_picks_up_a_later_insert(client):
    """ONE controlled insert while streaming a RUNNING job -> appears, then done."""
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
        await repo.set_status(s, job.id, JobStatus.RUNNING)

    async def insert_later():
        await asyncio.sleep(0.05)  # after the stream is established + first poll
        async with get_sessionmaker()() as s:
            await repo.insert_logs(
                s, [LogItem(job.id, "image_gen", "info", "late-line", datetime.now(timezone.utc))]
            )
            await repo.set_status(s, job.id, JobStatus.COMPLETED)

    task = asyncio.create_task(insert_later())
    try:
        text = "\n".join(await _read_until_done(client, f"/storyboard/{job.id}/events"))
    finally:
        await task
    assert "late-line" in text
    assert "event: done" in text


async def test_unknown_job_404(client):
    r = await client.get("/storyboard/does-not-exist/events")
    assert r.status_code == 404
