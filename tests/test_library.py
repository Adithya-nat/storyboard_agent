"""Film library: publish a finished film to the homepage with a generated poster."""

from __future__ import annotations

import io

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.api import library as library_api
from app.artifacts.store import ArtifactStore
from app.core.config import get_settings
from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.library import service
from app.main import app
from app.prompts import render
from app.schemas import JobStatus


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _png(color=(200, 120, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (90, 160), color).save(buf, "PNG")
    return buf.getvalue()


async def _seed(artifact_root, *, language="ta", status=JobStatus.COMPLETED) -> str:
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "Grand Theft Auto Chennai Official Trailer", 60, "dialogue", language)
        store = ArtifactStore(artifact_root, job.id)
        script = store.save_json("script.json", {
            "title": "மீட்டர் போடு ராஜா", "logline": "tamil logline", "language": language,
            "story_style": "dialogue", "style": "neon",
            "scenes": [{"order": i} for i in (1, 2, 3)],
        })
        store.save_json("brief.json", {"format": "stylized crime-comedy trailer",
                                       "purpose": "a swaggering chase", "logline": "Two hustlers steal gold."})
        store.save_json("gloss.json", {"மீட்டர் போடு ராஜா": "Drop the Meter, Raja"})
        publish = store.save_json("publish.json", {"title": "x", "hook": "y", "cover_order": 2,
                                                   "tiktok": {"caption": "", "hashtags": []},
                                                   "reels": {"caption": "", "hashtags": []},
                                                   "shorts": {"caption": "", "hashtags": []}})
        video = store.save_bytes("storyboard.mp4", b"not really a video")
        await repo.add_artifact(s, job.id, kind="script", path=script)
        await repo.add_artifact(s, job.id, kind="publish", path=publish)
        await repo.add_artifact(s, job.id, kind="video", path=video)
        for i in (1, 2, 3):
            p = store.save_bytes(f"frame_0{i}.png", _png((40 * i, 80, 120)))
            await repo.add_artifact(s, job.id, kind="frame", path=p, order_index=i)
        await repo.set_status(s, job.id, status)
    return job.id


def test_every_look_renders_into_the_poster_prompt():
    assert set(service.LOOK_IDS) == {"auto", "blockbuster", "thriller", "retro", "comedy", "arthouse"}
    text = service.poster_prompt(title="Drop the Meter, Raja", native_title="மீட்டர் போடு ராஜா", tagline=None,
                                 genre="Crime comedy", tone="swagger", logline="Gold heist.", look="retro")
    assert '"Drop the Meter, Raja"' in text and "மீட்டர் போடு ராஜா" in text
    assert "hand-lettered" in text and "{{" not in text
    assert "No real studio" in text
    # an English film has no secondary-title line
    en = render("library/poster", title="T", native_line="", tagline_rule="Tagline: x", genre="g",
                tone="t", logline="l", look="k")
    assert "Tamil title" not in en


def test_slugify_keeps_urls_ascii():
    assert service.slugify("Grand Theft Auto: Chennai!") == "grand-theft-auto-chennai"
    assert service.slugify("மீட்டர்") == "film"


async def test_draft_defaults_come_from_the_film(client, artifact_root):
    job_id = await _seed(artifact_root)
    body = (await client.get(f"/library/draft/{job_id}")).json()
    assert body["title"] == "Drop the Meter, Raja"
    assert body["native_title"] == "மீட்டர் போடு ராஜா"
    assert body["logline"] == "Two hustlers steal gold."
    assert body["genre"] == "Stylized crime-comedy"
    assert body["cover_frame"] == 2 and body["frames"] == [1, 2, 3]
    assert body["entry"] is None


async def test_publish_flow_end_to_end(client, artifact_root):
    job_id = await _seed(artifact_root)
    r = await client.post("/library", json={
        "job_id": job_id, "title": "Grand Theft Auto Chennai", "native_title": "மீட்டர் போடு ராஜா",
        "logline": "Two hustlers steal gold.", "genre": "Crime comedy", "look": "blockbuster",
        "show_prompt": False,
    })
    assert r.status_code == 202, r.text
    film_id = r.json()["id"]
    assert film_id == "grand-theft-auto-chennai"

    # The background task ran inside the request: media copied, poster made (mock image).
    entry = (await client.get(f"/library/{film_id}")).json()
    assert entry["status"] == "draft", entry
    assert entry["poster_version"] == 1 and entry["poster_frame"] == 2
    assert entry["frames"] == [1, 2, 3]
    assert entry["prompt"] is None
    d = service.film_dir(get_settings(), film_id)
    for name in ("poster.jpg", "frame.jpg", "film.mp4", "preview.mp4", "frames/1.jpg"):
        assert (d / name).exists(), name
    assert Image.open(d / "poster.jpg").size == service.POSTER_SIZE

    # Not on the homepage until the creator publishes.
    assert (await client.get("/library/public")).json() == []
    assert (await client.get(f"/library/public/{film_id}/poster.jpg")).status_code == 404
    # ...but the creator can see their draft poster.
    assert (await client.get(f"/library/{film_id}/media/poster.jpg")).status_code == 200

    pub = (await client.post(f"/library/{film_id}/publish")).json()
    assert pub["status"] == "published" and pub["published_at"]
    feed = await client.get("/library/public")
    assert feed.headers["access-control-allow-origin"] == "*"
    [card] = feed.json()
    assert card["title"] == "Grand Theft Auto Chennai"
    assert card["media"]["poster"] == f"/library/public/{film_id}/poster.jpg?v=1"
    assert "job_id" not in card and "status" not in card
    assert (await client.get(f"/library/public/{film_id}/film.mp4")).status_code == 200
    # frames stay private
    assert (await client.get(f"/library/public/{film_id}/frames/1.jpg")).status_code == 404
    assert (await client.get(f"/library/public/{film_id}/../../x")).status_code == 404

    # Another poster, from another frame: stays published, version bumps.
    again = await client.post(f"/library/{film_id}/poster", json={"look": "retro", "frame": 3})
    assert again.status_code == 202
    entry = (await client.get(f"/library/{film_id}")).json()
    assert entry["status"] == "published" and entry["poster_version"] == 2
    assert entry["poster_look"] == "retro" and entry["poster_frame"] == 3

    # Show the prompt after all.
    edited = (await client.patch(f"/library/{film_id}", json={"show_prompt": True, "genre": "Heist"})).json()
    assert edited["prompt"] == "Grand Theft Auto Chennai Official Trailer" and edited["genre"] == "Heist"

    # Unpublish hides it; delete removes the files.
    await client.post(f"/library/{film_id}/unpublish")
    assert (await client.get("/library/public")).json() == []
    assert (await client.delete(f"/library/{film_id}")).status_code == 204
    assert not d.exists()


async def test_one_entry_per_film_and_only_finished_films(client, artifact_root):
    running = await _seed(artifact_root, status=JobStatus.RUNNING)
    r = await client.post("/library", json={"job_id": running, "title": "T"})
    assert r.status_code == 409

    job_id = await _seed(artifact_root)
    assert (await client.post("/library", json={"job_id": job_id, "title": "Same"})).status_code == 202
    assert (await client.post("/library", json={"job_id": job_id, "title": "Same"})).status_code == 409
    # a second film with the same title gets its own slug
    other = await _seed(artifact_root)
    r2 = await client.post("/library", json={"job_id": other, "title": "Same"})
    assert r2.json()["id"].startswith("same-") and r2.json()["id"] != "same"


async def test_poster_tries_are_capped(client, artifact_root, monkeypatch):
    monkeypatch.setattr(get_settings(), "library_poster_limit", 2)
    job_id = await _seed(artifact_root)
    film_id = (await client.post("/library", json={"job_id": job_id, "title": "Capped"})).json()["id"]
    assert (await client.post(f"/library/{film_id}/poster", json={})).status_code == 202
    r = await client.post(f"/library/{film_id}/poster", json={})
    assert r.status_code == 429
    assert (await client.get(f"/library/{film_id}")).json()["posters_left"] == 0


async def test_a_failed_poster_keeps_the_film_and_says_why(client, artifact_root, monkeypatch):
    job_id = await _seed(artifact_root)

    async def boom(*_a, **_k):
        raise RuntimeError("blocked")

    monkeypatch.setattr(library_api.service, "make_poster", boom)
    film_id = (await client.post("/library", json={"job_id": job_id, "title": "Boom"})).json()["id"]
    entry = (await client.get(f"/library/{film_id}")).json()
    assert entry["status"] == "failed"
    assert "poster" in entry["error"]
    assert entry["frames"] == [1, 2, 3]  # media copied, so another try can work
    monkeypatch.undo()
    assert (await client.post(f"/library/{film_id}/poster", json={"look": "auto"})).status_code == 202
    assert (await client.get(f"/library/{film_id}")).json()["status"] == "draft"
