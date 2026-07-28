"""API surface tests: status codes + the 404/409 branches."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.main import app
from app.schemas import JobStatus


@pytest.fixture
def stub_video(monkeypatch):
    async def _fake_assemble(timeline, output_path):
        Path(output_path).write_bytes(b"FAKE-MP4")
        return output_path

    monkeypatch.setattr("app.agents.assembler.assemble_video", _fake_assemble)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200


async def test_create_returns_202_with_job_id(client, stub_video):
    r = await client.post("/storyboard", json={"prompt": "a robot", "duration": 30})
    assert r.status_code == 202
    body = r.json()
    assert body["job_id"]
    assert body["status"] == JobStatus.QUEUED.value


async def test_create_rejects_short_prompt(client):
    r = await client.post("/storyboard", json={"prompt": "x", "duration": 30})
    assert r.status_code == 422


async def test_get_unknown_job_404(client):
    r = await client.get("/storyboard/does-not-exist")
    assert r.status_code == 404


async def test_get_video_before_ready_409(client):
    # Seed a running job (no video artifact yet).
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
        await repo.set_status(s, job.id, JobStatus.RUNNING)

    r = await client.get(f"/storyboard/{job.id}/video")
    assert r.status_code == 409


async def test_get_status_exposes_degraded_artifacts(client):
    # Seed a completed_with_warnings job with one degraded frame.
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
        await repo.add_artifact(s, job.id, kind="frame", path="/x/1.png", status="degraded", order_index=1)
        await repo.set_status(s, job.id, JobStatus.COMPLETED_WITH_WARNINGS, warnings=["frame 1: failed"])

    r = await client.get(f"/storyboard/{job.id}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == JobStatus.COMPLETED_WITH_WARNINGS.value
    assert body["warnings"] == ["frame 1: failed"]
    assert any(a["status"] == "degraded" for a in body["artifacts"])
