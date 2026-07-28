"""GET /storyboard/{id} now carries the per-stage summary + current_stage."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_status_includes_ordered_stages_and_current(client):
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
        await repo.upsert_stage_running(s, job.id, "script_writer", 1)
        await repo.flip_stage_if_running(s, job.id, "script_writer", "succeeded")
        await repo.upsert_stage_running(s, job.id, "designer", 2)  # currently running

    resp = await client.get(f"/storyboard/{job.id}")
    assert resp.status_code == 200
    body = resp.json()

    assert body["current_stage"] == "designer"
    assert [st["name"] for st in body["stages"]] == ["script_writer", "designer"]  # by seq
    assert body["stages"][0]["status"] == "succeeded"
    assert body["stages"][1]["status"] == "running"


async def test_status_no_stages_yet_is_empty(client):
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)

    body = (await client.get(f"/storyboard/{job.id}")).json()
    assert body["stages"] == []
    assert body["current_stage"] is None
