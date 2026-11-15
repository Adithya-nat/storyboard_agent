"""Budget guard (402 / degrade), usage summary + runs, publish endpoint, Tamil gate."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.main import app
from app.observability.cost import SpendEvent


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def _spend(job_id: str, usd: float, category: str = "video", model: str = "gemini-omni-1.2-flash"):
    async with get_sessionmaker()() as s:
        await repo.add_spend_events(s, job_id, [SpendEvent(category=category, provider="google", model=model, cost_usd=usd)])
        await repo.upsert_run(s, job_id, prompt="p", style="dialogue", duration=30, status="completed",
                              cost_usd=usd, cost_by_category={category: usd}, scenes=5, omni_scenes=5,
                              runtime_seconds=120, finished_at=datetime.now(timezone.utc), models=[model])


@pytest.fixture
def small_budget(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "monthly_budget_usd", 10.0)
    monkeypatch.setattr(s, "budget_mode", "block")
    return s


async def test_budget_blocks_with_402_when_over(client, small_budget, monkeypatch):
    monkeypatch.setattr("app.api.routes.run_job", lambda *a, **k: None)
    await _spend("old-1", 9.5)
    r = await client.post("/storyboard", json={"prompt": "a test story", "duration": 30})
    assert r.status_code == 402 and "budget" in r.json()["detail"].lower()
    b = (await client.get("/budget")).json()
    assert b["month_spent_usd"] == 9.5 and b["warn"] and b["cap_usd"] == 10.0


async def test_budget_degrade_mode_runs_without_motion(client, small_budget, monkeypatch):
    seen = {}

    async def fake_run(*args, **_k):
        seen["args"] = args

    monkeypatch.setattr("app.api.routes.run_job", fake_run)
    monkeypatch.setattr(small_budget, "budget_mode", "degrade")
    monkeypatch.setattr(small_budget, "require_ai_video", False)
    await _spend("old-2", 7.0)  # 30s dialogue ~$4.5 with Omni, ~$1 without
    r = await client.post("/storyboard", json={"prompt": "a test story", "duration": 30, "style": "dialogue"})
    assert r.status_code == 202
    assert seen["args"][-1] is False  # allow_motion=False


async def test_create_records_style_and_language(client, monkeypatch):
    monkeypatch.setattr("app.api.routes.run_job", lambda *a, **k: None)
    r = await client.post("/storyboard", json={"prompt": "a test story", "duration": 60, "style": "mixed", "language": "ta"})
    assert r.status_code == 202
    body = (await client.get(f"/storyboard/{r.json()['job_id']}")).json()
    assert (body["style"], body["language"], body["cost_usd"]) == ("mixed", "ta", 0.0)


async def test_tamil_rejected_without_text_shaping(client, monkeypatch):
    monkeypatch.setattr("app.api.routes.raqm_available", lambda: False)
    r = await client.post("/storyboard", json={"prompt": "a test story", "language": "ta"})
    assert r.status_code == 422 and "libraqm" in r.json()["detail"]


async def test_usage_summary_and_runs(client):
    await _spend("run-a", 4.0)
    await _spend("run-b", 6.0, category="image", model="gemini-nano-banana-2.1")
    s = (await client.get("/usage/summary?range=month")).json()
    assert s["runs"] == 2 and s["total_cost_usd"] == 10.0 and s["avg_cost_per_run"] == 5.0
    assert s["by_category"] == {"image": 6.0, "video": 4.0}
    assert s["avg_cost_by_style_duration"] == {"dialogue_30": 5.0}
    assert s["by_day"] and s["by_day"][0]["total"] == 10.0
    runs = (await client.get("/usage/runs?limit=1")).json()
    assert len(runs) == 1 and runs[0]["job_id"] in {"run-a", "run-b"}
    est = (await client.get("/estimate?style=dialogue&duration=60")).json()
    assert est["estimated_usd"] > 0


async def test_spend_survives_ttl_sweep(client):
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
    await _spend(job.id, 3.0)
    async with get_sessionmaker()() as s:
        await repo.delete_expired_jobs(s, ttl_seconds=-1)
        assert await repo.get_job(s, job.id) is None
        assert await repo.spend_since(s, repo.month_start()) == 3.0


async def test_job_cost_accumulates_across_resumed_runs():
    from app.observability.cost import SpendEvent as Ev

    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
        await repo.add_spend_events(s, job.id, [Ev(category="text", provider="google", model="m", cost_usd=0.10, stage="script_writer")])
        await repo.set_job_cost(s, job.id, 0.10, {"script_writer": 0.10})
        # a resumed run's fresh collector only knows its own spend
        await repo.add_spend_events(s, job.id, [Ev(category="image", provider="google", model="m", cost_usd=0.02, stage="designer")])
        await repo.set_job_cost(s, job.id, 0.02, {"designer": 0.02})
        job = await repo.get_job(s, job.id)
        assert round(job.cost_usd, 4) == 0.12


async def test_degrade_mode_refuses_when_ai_video_is_required(client, small_budget, monkeypatch):
    monkeypatch.setattr("app.api.routes.run_job", lambda *a, **k: None)
    monkeypatch.setattr(small_budget, "budget_mode", "degrade")
    monkeypatch.setattr(small_budget, "require_ai_video", True)
    await _spend("old-3", 7.0)
    r = await client.post("/storyboard", json={"prompt": "a test story", "duration": 30, "style": "dialogue"})
    assert r.status_code == 402  # never a stills-only film
