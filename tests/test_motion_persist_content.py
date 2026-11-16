"""Motion clips -> artifact rows -> scenes[].video_url, and the pipeline field
(on/off). Flag-off leaves video_url None + no 'motion' stage (regression contract)."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.main import app
from app.workers.runner import _persist

_BASE_PIPELINE = ["brief", "script_writer", "dialogue_pass", "reflection", "designer", "image_gen", "narration", "captions", "music", "assembler", "film_review", "publish_kit"]


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _make_job() -> str:
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
        return job.id


async def test_persist_maps_motion_clips_to_video_url(client, tmp_path):
    job_id = await _make_job()
    manifest = tmp_path / "motion_manifest.json"
    manifest.write_text(
        """{"pipeline_version":"2","scenes":[{"order":1,"motion_kind":"generative",
        "renderer":"veo","model":"veo-3.1-lite-generate-preview",
        "source_seconds":4.0,"attempts":[]}]}"""
    )
    final = {
        "frames": [
            {"order": 1, "path": "/x/frame_01.png", "status": "ok"},
            {"order": 2, "path": "/x/frame_02.png", "status": "ok"},
        ],
        "motion_clips": [
            {
                "order": 1,
                "path": "/x/motion_01.mp4",
                "status": "ok",
                "tier": "veo",
                "motion_kind": "generative",
                "renderer": "veo",
                "model": "veo-3.1-lite-generate-preview",
            },
        ],
        "motion_manifest_path": str(manifest),
        "warnings": [],
    }
    await _persist(job_id, final)

    body = (await client.get(f"/storyboard/{job_id}/storyboard")).json()
    scenes = {sc["order"]: sc for sc in body["scenes"]}
    assert scenes[1]["video_url"] is not None                 # has a motion clip
    assert scenes[1]["video_url"].endswith("/artifacts/" + _artifact_id(scenes[1]["video_url"]))
    assert scenes[2]["video_url"] is None                     # no clip -> still image
    assert scenes[1]["motion_kind"] == "generative"
    assert scenes[1]["motion_renderer"] == "veo"
    assert scenes[1]["motion_model"] == "veo-3.1-lite-generate-preview"
    assert body["motion_coverage"]["generative"] == 1
    assert body["motion_coverage"]["still"] == 1
    assert body["motion_coverage"]["by_model"] == {
        "veo-3.1-lite-generate-preview": 1
    }

    # The motion artifact is a first-class row served by the generic artifacts route.
    status = (await client.get(f"/storyboard/{job_id}")).json()
    kinds = [a["kind"] for a in status["artifacts"]]
    assert kinds.count("motion") == 1
    assert kinds.count("motion_manifest") == 1


async def test_no_motion_clips_leaves_video_url_none(client):
    job_id = await _make_job()
    final = {"frames": [{"order": 1, "path": "/x/frame_01.png", "status": "ok"}], "warnings": []}
    await _persist(job_id, final)
    body = (await client.get(f"/storyboard/{job_id}/storyboard")).json()
    assert body["scenes"][0]["video_url"] is None


async def test_pipeline_field_excludes_motion_by_default(client):
    job_id = await _make_job()
    body = (await client.get(f"/storyboard/{job_id}")).json()
    assert body["pipeline"] == _BASE_PIPELINE


async def test_pipeline_field_includes_motion_when_enabled(client, monkeypatch):
    # NOTE: the route reads pipeline_stage_names(get_settings()); get_settings is
    # lru_cached with MOTION_ENGINE_ENABLED=false pinned in conftest, so flip the
    # env AND clear the cache (and clear again after) or the route won't see it.
    from app.core.config import get_settings

    monkeypatch.setenv("MOTION_ENGINE_ENABLED", "true")
    get_settings.cache_clear()
    try:
        job_id = await _make_job()
        body = (await client.get(f"/storyboard/{job_id}")).json()
        assert body["pipeline"] == _BASE_PIPELINE[:7] + ["motion"] + _BASE_PIPELINE[7:]
    finally:
        get_settings.cache_clear()


def _artifact_id(url: str) -> str:
    return url.rsplit("/", 1)[-1]
