"""Hosted mode: sign-in tokens, per-user jobs, media links, BYOK keys, 24h retention."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from httpx import ASGITransport, AsyncClient
from sqlalchemy import update

from app.adapters import rate_limit
from app.core import auth
from app.core.config import get_settings
from app.core.keys import REDACTED, RedactKeysFilter, bind_key, current_scope, release_key
from app.db import repository as repo
from app.db.models import Job, RunLedger
from app.db.session import get_sessionmaker
from app.graph.checkpoint import purge_checkpoints
from app.main import app
from app.schemas import JobStatus
from app.workers import retention

KEY = "AIza" + "B" * 35
ISS = "http://web.test"
AUD = "creator-agent-api"
_PRIV = Ed25519PrivateKey.generate()


def token(sub="u1", email="ada@example.com", exp_in=600, aud=AUD, iss=ISS) -> str:
    now = int(time.time())
    return jwt.encode(
        {"sub": sub, "email": email, "name": "Ada", "aud": aud, "iss": iss, "iat": now, "exp": now + exp_in},
        _PRIV, algorithm="EdDSA",
    )


def bearer(**kw) -> dict:
    return {"authorization": f"Bearer {token(**kw)}"}


@pytest.fixture
def hosted(monkeypatch):
    s = get_settings()
    for k, v in {
        "auth_mode": "jwt", "auth_jwks_url": "http://web.test/api/auth/jwks", "auth_issuer": ISS,
        "allowed_users": ["ada@example.com", "@team.dev"], "asset_signing_secret": "s" * 32,
    }.items():
        monkeypatch.setattr(s, k, v)
    monkeypatch.setattr(auth, "_jwks_client", lambda url: SimpleNamespace(
        get_signing_key_from_jwt=lambda t: SimpleNamespace(key=_PRIV.public_key())))
    return s


@pytest.fixture
def captured(monkeypatch):
    calls = []

    async def fake_run_job(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr("app.api.routes.run_job", fake_run_job)
    return calls


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def _job(user_id=None, prompt="p") -> Job:
    async with get_sessionmaker()() as s:
        return await repo.create_job(s, prompt, 30, user_id=user_id)


# --- sign-in -------------------------------------------------------------------------- #
async def test_local_mode_needs_no_sign_in(client):
    r = await client.get("/me")
    assert r.status_code == 200 and r.json()["allowed"] and r.json()["auth_mode"] == "none"


@pytest.mark.parametrize(
    "headers, code",
    [({}, 401), ({"authorization": "Bearer junk"}, 401)],
)
async def test_hosted_rejects_missing_or_bad_token(hosted, client, headers, code):
    assert (await client.get("/storyboard", headers=headers)).status_code == code


async def test_hosted_rejects_expired_or_wrong_audience(hosted, client):
    assert (await client.get("/storyboard", headers=bearer(exp_in=-10))).status_code == 401
    assert (await client.get("/storyboard", headers=bearer(aud="other"))).status_code == 401
    assert (await client.get("/storyboard", headers=bearer(iss="http://evil"))).status_code == 401


async def test_allowlist(hosted, client):
    r = await client.get("/storyboard", headers=bearer(email="eve@else.com"))
    assert r.status_code == 403 and r.json()["detail"]["code"] == "not_allowlisted"
    assert (await client.get("/storyboard", headers=bearer(email="bob@team.dev"))).status_code == 200
    me = (await client.get("/me", headers=bearer(email="eve@else.com"))).json()
    assert me["allowed"] is False and me["media_token"] is None


# --- per-user jobs ---------------------------------------------------------------------- #
async def test_other_users_job_is_404_everywhere(hosted, client):
    job = await _job(user_id="someone-else")
    h = bearer()
    for path in ("", "/logs", "/storyboard", "/live", "/publish", "/video", "/frames/1"):
        assert (await client.get(f"/storyboard/{job.id}{path}", headers=h)).status_code == 404, path
    assert (await client.post(f"/storyboard/{job.id}/resume", headers={**h, "x-gemini-key": KEY})).status_code == 404


async def test_list_shows_only_my_recent_jobs(hosted, client):
    mine, theirs = await _job(user_id="u1", prompt="mine"), await _job(user_id="u2")
    rows = (await client.get("/storyboard", headers=bearer())).json()
    assert [r["job_id"] for r in rows] == [mine.id] and theirs.id not in json.dumps(rows)


async def test_usage_is_scoped_to_the_user(hosted, client):
    async with get_sessionmaker()() as s:
        await repo.upsert_run(s, "j1", user_id="u1", prompt="a")
        await repo.upsert_run(s, "j2", user_id="u2", prompt="b")
    rows = (await client.get("/usage/runs", headers=bearer())).json()
    assert [r["job_id"] for r in rows] == ["j1"]


# --- media links ------------------------------------------------------------------------ #
async def test_media_token_round_trip(hosted):
    t, _ = auth.make_media_token(auth.User(id="u1"), hosted)
    assert auth.read_media_token(t, hosted).id == "u1"
    assert auth.read_media_token(t, hosted, now=time.time() + hosted.media_token_ttl_seconds + 5) is None
    uid, exp, sig = t.rsplit(".", 2)
    assert auth.read_media_token(f"u2.{exp}.{sig}", hosted) is None


async def test_media_route_accepts_token_for_own_job_only(hosted, client):
    mine, theirs = await _job(user_id="u1"), await _job(user_id="u2")
    t = (await client.get("/me", headers=bearer())).json()["media_token"]
    assert (await client.get(f"/storyboard/{mine.id}/video?t={t}")).status_code == 409  # not ready, but authorized
    assert (await client.get(f"/storyboard/{theirs.id}/video?t={t}")).status_code == 404
    assert (await client.get(f"/storyboard/{mine.id}/video?t=bad.1.x")).status_code == 401


# --- BYOK --------------------------------------------------------------------------------- #
async def test_byok_requires_a_key(hosted, client, captured, monkeypatch):
    monkeypatch.setattr(hosted, "key_mode", "byok")
    monkeypatch.setattr(hosted, "use_mock_providers", False)
    body = {"prompt": "a robot", "duration": 30}
    r = await client.post("/storyboard", json=body, headers=bearer())
    assert r.status_code == 400 and r.json()["detail"]["code"] == "key_required"
    r = await client.post("/storyboard", json=body, headers={**bearer(), "x-gemini-key": "short"})
    assert r.status_code == 400
    r = await client.post("/storyboard", json=body, headers={**bearer(), "x-gemini-key": KEY})
    assert r.status_code == 202
    (_, kwargs), = captured
    assert kwargs["api_key"] == KEY and kwargs["user_id"] == "u1"


async def test_byok_resume_needs_the_key_again(hosted, client, captured, monkeypatch):
    monkeypatch.setattr(hosted, "key_mode", "byok")
    monkeypatch.setattr(hosted, "use_mock_providers", False)
    job = await _job(user_id="u1")
    async with get_sessionmaker()() as s:
        await repo.set_status(s, job.id, JobStatus.FAILED, error="AIVideoUnavailable: quota")
    r = await client.post(f"/storyboard/{job.id}/resume", headers=bearer())
    assert r.status_code == 400 and r.json()["detail"]["code"] == "key_required"
    r = await client.post(f"/storyboard/{job.id}/resume", headers={**bearer(), "x-gemini-key": KEY})
    assert r.status_code == 202
    (_, kwargs), = captured
    assert kwargs["api_key"] == KEY and kwargs["user_id"] == "u1"


async def test_one_active_film_per_user(hosted, client, captured):
    body = {"prompt": "a robot", "duration": 30}
    assert (await client.post("/storyboard", json=body, headers=bearer())).status_code == 202
    assert (await client.post("/storyboard", json=body, headers=bearer())).status_code == 429


async def test_key_reaches_the_job_and_never_lands_in_the_db(monkeypatch):
    from app.workers import runner

    seen = {}

    def fake_build_graph(deps, checkpointer=None):
        seen["key"] = deps.settings.gemini_api_key
        seen["scope"] = current_scope()
        raise RuntimeError(f"provider said: bad request for key={deps.settings.gemini_api_key}")

    monkeypatch.setattr(runner, "build_graph", fake_build_graph)
    job = await _job(user_id="u1")
    await runner.run_job(job.id, "p", 30, api_key=KEY, user_id="u1")
    assert seen["key"] == KEY and seen["scope"]
    assert current_scope() == ""  # released after the job
    async with get_sessionmaker()() as s:
        stored = await repo.get_job(s, job.id)
    assert stored.status == JobStatus.FAILED.value
    assert KEY not in stored.error and REDACTED in stored.error


def test_limiters_are_per_key():
    rate_limit.reset_all_limiters()
    shared = rate_limit.get_text_limiter(25)
    tok = bind_key(KEY)
    try:
        mine = rate_limit.get_text_limiter(25)
    finally:
        release_key(KEY, tok)
    assert mine is not shared and rate_limit.get_text_limiter(25) is shared


def test_mock_mode_never_calls_google(monkeypatch, tmp_path):
    from app.adapters import omni
    from app.adapters.genai_client import make_client
    from app.core.images import render_card

    s = get_settings().model_copy(update={"gemini_api_key": KEY, "use_mock_providers": True})
    with pytest.raises(RuntimeError):
        make_client(s)
    # Mock AI video is a local still-clip: free, and no client is ever built.
    monkeypatch.setattr(omni, "make_client", lambda *a, **k: pytest.fail("mock mode built a Gemini client"))
    clip = omni.OmniVideoAdapter(s, journal_dir=tmp_path).generate(
        render_card("x"), omni.SceneSpec(order=1, scene="x", seconds=3.0), key="k")
    assert clip.cost_usd == 0.0 and clip.model == "mock-omni"


def test_log_filter_redacts_keys():
    rec = logging.LogRecord("app", logging.INFO, __file__, 1, "calling with %s", (KEY,), None)
    RedactKeysFilter().filter(rec)
    assert KEY not in rec.getMessage() and REDACTED in rec.getMessage()


async def test_key_check_needs_a_key(client):
    assert (await client.post("/keys/check")).status_code == 400


# --- 24h retention ------------------------------------------------------------------------ #
async def _age(job_id: str, hours: float) -> None:
    async with get_sessionmaker()() as s:
        old = datetime.now(timezone.utc) - timedelta(hours=hours)
        await s.execute(update(Job).where(Job.id == job_id).values(created_at=old))
        await s.commit()


async def test_sweep_deletes_files_blanks_prompt_and_schedules_next(tmp_path, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "artifact_root", tmp_path)
    stale, fresh = await _job(prompt="secret idea"), await _job(prompt="new")
    async with get_sessionmaker()() as session:
        await repo.upsert_run(session, stale.id, prompt="secret idea", cost_usd=1.5)
    await _age(stale.id, 25)
    await _age(fresh.id, 23)
    (tmp_path / stale.id).mkdir()
    (tmp_path / stale.id / "clip.mp4").write_bytes(b"x")

    delay = await retention.sweep_once(s)

    assert not (tmp_path / stale.id).exists()
    async with get_sessionmaker()() as session:
        assert await repo.get_job(session, stale.id) is None
        row = await session.get(RunLedger, stale.id)
        assert row.prompt == "" and row.cost_usd == 1.5
    assert 60 <= delay <= 3601  # wakes when the fresh job hits 24h (~1h away)


async def test_purge_checkpoints_sqlite(tmp_path, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "artifact_root", tmp_path)
    with sqlite3.connect(tmp_path / "checkpoints.sqlite") as c:
        c.execute("CREATE TABLE checkpoints (thread_id TEXT)")
        c.execute("CREATE TABLE writes (thread_id TEXT)")
        c.executemany("INSERT INTO checkpoints VALUES (?)", [("j1",), ("j1:r:abcd",), ("j2",)])
        c.execute("INSERT INTO writes VALUES ('j1')")
    assert await purge_checkpoints(s, ["j1"]) == 3
    with sqlite3.connect(tmp_path / "checkpoints.sqlite") as c:
        assert [r[0] for r in c.execute("SELECT thread_id FROM checkpoints")] == ["j2"]


def test_hosted_settings_cap_retention():
    from app.core.config import Settings

    with pytest.raises(ValueError):
        Settings(auth_mode="jwt", auth_jwks_url="http://x/jwks", job_ttl_seconds=90000)
    with pytest.raises(ValueError):
        Settings(auth_mode="jwt")


async def test_resume_respects_the_active_film_limit(hosted, client, captured):
    running = await _job(user_id="u1")
    stopped = await _job(user_id="u1")
    async with get_sessionmaker()() as s:
        await repo.set_status(s, running.id, JobStatus.RUNNING)
        await repo.set_status(s, stopped.id, JobStatus.FAILED, error="x")
    r = await client.post(f"/storyboard/{stopped.id}/resume", headers=bearer())
    assert r.status_code == 429 and not captured


async def test_concurrent_creates_admit_only_one(hosted, client, captured):
    import asyncio

    body = {"prompt": "a robot", "duration": 30}
    codes = sorted(r.status_code for r in await asyncio.gather(
        *(client.post("/storyboard", json=body, headers=bearer()) for _ in range(4))))
    assert codes == [202, 429, 429, 429]


async def test_sweep_retries_leftover_files_and_checkpoints(tmp_path, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "artifact_root", tmp_path)
    live = await _job()
    gone = "0b0b0b0b-0000-4000-8000-000000000000"
    for jid in (live.id, gone):
        (tmp_path / jid).mkdir()
    (tmp_path / "_omni_ops").mkdir()
    with sqlite3.connect(tmp_path / "checkpoints.sqlite") as c:
        c.execute("CREATE TABLE checkpoints (thread_id TEXT)")
        c.executemany("INSERT INTO checkpoints VALUES (?)", [(live.id,), (gone,), (f"{gone}:r:1",)])
    await retention.sweep_once(s)
    assert (tmp_path / live.id).exists() and not (tmp_path / gone).exists() and (tmp_path / "_omni_ops").exists()
    with sqlite3.connect(tmp_path / "checkpoints.sqlite") as c:
        assert [r[0] for r in c.execute("SELECT thread_id FROM checkpoints")] == [live.id]


async def test_two_resumes_of_one_film_start_one_runner(client, captured):
    import asyncio

    job = await _job()
    async with get_sessionmaker()() as s:
        await repo.set_status(s, job.id, JobStatus.FAILED, error="x")
    codes = sorted(r.status_code for r in await asyncio.gather(
        *(client.post(f"/storyboard/{job.id}/resume") for _ in range(3))))
    assert codes[0] == 202 and all(c == 409 for c in codes[1:]) and len(captured) == 1
    async with get_sessionmaker()() as s:  # a started runner is never knocked back to queued
        await repo.set_status(s, job.id, JobStatus.RUNNING)
        assert await repo.mark_queued(s, job.id) is False


async def test_next_sweep_ignores_running_films_and_honours_paused_ttl():
    from datetime import datetime, timedelta, timezone

    running, paused = await _job(), await _job()
    async with get_sessionmaker()() as s:
        await repo.set_status(s, running.id, JobStatus.RUNNING)
        await repo.set_status(s, paused.id, JobStatus.PAUSED)
    await _age(running.id, 30)
    await _age(paused.id, 30)
    async with get_sessionmaker()() as s:
        assert await repo.next_expiry(s, 86400) < datetime.now(timezone.utc)  # paused counts at 24h
        nxt = await repo.next_expiry(s, 86400, paused_ttl_seconds=7 * 86400)
    assert nxt is not None and nxt > datetime.now(timezone.utc) + timedelta(days=5)


def test_allowlist_accepts_json_or_csv(monkeypatch):
    from app.core.config import Settings

    monkeypatch.setenv("ALLOWED_USERS", '["a@x.com", "@team.dev"]')
    assert Settings().allowed_users == ["a@x.com", "@team.dev"]
    monkeypatch.setenv("ALLOWED_USERS", "a@x.com, @team.dev")
    assert Settings().allowed_users == ["a@x.com", "@team.dev"]


def test_tls_verification_mode_is_kept():
    from app.core.config import normalize_database_url
    from app.graph.checkpoint import _psycopg_url

    url = normalize_database_url("postgresql://u:p@h/db?sslmode=verify-full")
    assert url.endswith("?ssl=verify-full") and _psycopg_url(url).endswith("?sslmode=verify-full")


def test_neon_url_is_normalized():
    from app.core.config import normalize_database_url

    url = normalize_database_url("postgresql://u:p@ep-1.neon.tech/db?sslmode=require&channel_binding=require")
    assert url == "postgresql+asyncpg://u:p@ep-1.neon.tech/db?ssl=require"
