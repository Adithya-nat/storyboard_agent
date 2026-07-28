"""GET /storyboard/{id}/logs — the paginated, stage-filtered expand-view pull."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.main import app
from app.observability.logs import LogItem


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _seed_logs(job_id: str):
    now = datetime.now(timezone.utc)
    async with get_sessionmaker()() as s:
        await repo.insert_logs(
            s,
            [
                LogItem(job_id, "script_writer", "info", "wrote script", now),
                LogItem(job_id, "image_gen", "info", "frame 1", now),
                LogItem(job_id, "image_gen", "warning", "frame 2 failed", now),
            ],
        )


async def test_logs_all_with_cursor(client):
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
    await _seed_logs(job.id)

    body = (await client.get(f"/storyboard/{job.id}/logs")).json()
    assert len(body["logs"]) == 3
    assert body["next_after"] == body["logs"][-1]["id"]


async def test_logs_stage_filter_and_after(client):
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
    await _seed_logs(job.id)

    body = (await client.get(f"/storyboard/{job.id}/logs?stage=image_gen")).json()
    assert [lg["message"] for lg in body["logs"]] == ["frame 1", "frame 2 failed"]

    first_id = body["logs"][0]["id"]
    body2 = (
        await client.get(f"/storyboard/{job.id}/logs?stage=image_gen&after={first_id}")
    ).json()
    assert [lg["message"] for lg in body2["logs"]] == ["frame 2 failed"]


async def test_logs_unknown_job_404(client):
    r = await client.get("/storyboard/does-not-exist/logs")
    assert r.status_code == 404
