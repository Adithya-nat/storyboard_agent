"""Frontend-facing endpoints: artifact streaming, structured storyboard content
(captions sourced from the shotlist so they match the mp4), and 1-hour TTL (410)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import update

from app.artifacts.store import ArtifactStore
from app.db import repository as repo
from app.db.models import Job
from app.db.session import get_sessionmaker
from app.main import app
from app.schemas import JobStatus


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _seed_completed_job(artifact_root) -> str:
    """A completed job with script.json + shotlist.json + two frames on disk."""
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "a lonely robot", 30)
        store = ArtifactStore(artifact_root, job.id)
        script_path = store.save_json(
            "script.json",
            {
                "title": "Robot",
                "logline": "A lonely robot.",
                "style": "moody",
                "scenes": [
                    {"order": 1, "description": "d1", "narration": "Beat one."},
                    {"order": 2, "description": "d2", "narration": "Beat two."},
                ],
            },
        )
        shotlist_path = store.save_json(
            "shotlist.json",
            {
                "style_bible": "bible",
                "shots": [
                    {"order": 1, "image_prompt": "p1", "narration": "Beat one."},
                    {"order": 2, "image_prompt": "p2", "narration": "Beat two."},
                ],
            },
        )
        f1 = store.save_bytes("frame_01.png", b"PNG1")
        f2 = store.save_bytes("frame_02.png", b"PNG2")
        await repo.add_artifact(s, job.id, kind="script", path=script_path)
        await repo.add_artifact(s, job.id, kind="shotlist", path=shotlist_path)
        await repo.add_artifact(s, job.id, kind="frame", path=f1, order_index=1)
        await repo.add_artifact(s, job.id, kind="frame", path=f2, order_index=2)
        await repo.set_status(s, job.id, JobStatus.COMPLETED)
    return job.id


async def test_storyboard_content_joins_frames_and_captions(client, artifact_root):
    job_id = await _seed_completed_job(artifact_root)
    body = (await client.get(f"/storyboard/{job_id}/storyboard")).json()

    assert body["title"] == "Robot"
    assert body["logline"] == "A lonely robot."
    assert body["video_url"] is None  # no video artifact seeded
    assert [s["order"] for s in body["scenes"]] == [1, 2]
    # Caption comes from the shotlist narration (what the assembler burns in).
    assert body["scenes"][0]["narration"] == "Beat one."

    # The frame_url is fetchable and returns the bytes (never the raw FS path).
    frame_url = body["scenes"][0]["frame_url"]
    assert "/artifacts/" in frame_url
    fr = await client.get(frame_url)
    assert fr.status_code == 200
    assert fr.content == b"PNG1"


async def test_frame_by_order(client, artifact_root):
    job_id = await _seed_completed_job(artifact_root)
    r = await client.get(f"/storyboard/{job_id}/frames/2")
    assert r.status_code == 200
    assert r.content == b"PNG2"


async def test_frame_unknown_order_404(client, artifact_root):
    job_id = await _seed_completed_job(artifact_root)
    r = await client.get(f"/storyboard/{job_id}/frames/99")
    assert r.status_code == 404


async def test_status_includes_created_and_expires(client):
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
    body = (await client.get(f"/storyboard/{job.id}")).json()
    assert "created_at" in body and "expires_at" in body
    assert body["expires_at"] > body["created_at"]  # same ISO format -> lexical ok


async def test_expired_job_returns_410(client):
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
        old = datetime.now(timezone.utc) - timedelta(hours=2)
        await s.execute(update(Job).where(Job.id == job.id).values(created_at=old))
        await s.commit()
    r = await client.get(f"/storyboard/{job.id}")
    assert r.status_code == 410


async def test_delete_expired_jobs_sweeps_db(artifact_root):
    async with get_sessionmaker()() as s:
        fresh = await repo.create_job(s, "fresh", 30)
        stale = await repo.create_job(s, "stale", 30)
        old = datetime.now(timezone.utc) - timedelta(hours=2)
        await s.execute(update(Job).where(Job.id == stale.id).values(created_at=old))
        await s.commit()

    async with get_sessionmaker()() as s:
        deleted = await repo.delete_expired_jobs(s, ttl_seconds=3600)
    assert stale.id in deleted and fresh.id not in deleted

    async with get_sessionmaker()() as s:
        assert await repo.get_job(s, stale.id) is None
        assert await repo.get_job(s, fresh.id) is not None
