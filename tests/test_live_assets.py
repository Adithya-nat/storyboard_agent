"""Live assets: artifacts published as stages finish (and per frame), served by
GET /storyboard/{id}/live, signalled over SSE with an ``assets`` event.

Contract: purely additive. The final persist is unchanged (same rows from the
same pure mapping), publishing is best-effort, and a reporter that doesn't
implement it is still accepted by the stage wrapper.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.adapters.llm import LLMAdapter
from app.agents.image_gen import image_gen
from app.artifacts.rows import artifact_rows, assets_version
from app.artifacts.store import ArtifactStore
from app.core.config import Settings
from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.graph.build import _instrument
from app.graph.state import Deps
from app.main import app
from app.observability.reporter import ProgressReporter
from app.schemas import JobStatus, Shot, Shotlist


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# --------------------------------------------------------------------------- #
# artifact_rows: one mapping for the final persist AND the live publish
# --------------------------------------------------------------------------- #
def test_artifact_rows_partial_state_only_lists_what_exists():
    assert artifact_rows({"job_id": "j"}) == []
    rows = artifact_rows(
        {
            "script_path": "/a/script.json",
            "style_ref_path": "/a/ref.png",
            "cast_paths": {"MAYA": ["/a/maya.png"], "EMPTY": []},
            "frames": [{"path": "/a/f1.png", "status": "ok", "order": 1}],
        }
    )
    kinds = [(r["kind"], r.get("order_index")) for r in rows]
    assert kinds == [("script", None), ("style_ref", None), ("frame", 1), ("cast:MAYA", None)]


def test_assets_version_is_order_independent_and_changes_on_new_rows():
    class A:
        def __init__(self, i, status="ok"):
            self.id, self.status = i, status

    v1 = assets_version([A("x"), A("y")])
    assert v1 == assets_version([A("y"), A("x")])
    assert v1 != assets_version([A("x"), A("y"), A("z")])
    assert v1 != assets_version([A("x"), A("y", "degraded")])


# --------------------------------------------------------------------------- #
# repository.upsert_artifact: a re-rendered frame replaces, never duplicates
# --------------------------------------------------------------------------- #
async def test_upsert_artifact_replaces_same_kind_and_order(session):
    job_id = (await repo.create_job(session, "p", 30)).id
    await repo.upsert_artifact(session, job_id, kind="frame", path="/a", order_index=1)
    await repo.upsert_artifact(session, job_id, kind="frame", path="/b", order_index=1)
    await repo.upsert_artifact(session, job_id, kind="frame", path="/c", order_index=2)
    async with get_sessionmaker()() as fresh:
        got = await repo.get_job(fresh, job_id)
    frames = sorted((a.order_index, a.path) for a in got.artifacts if a.kind == "frame")
    assert frames == [(1, "/b"), (2, "/c")]


# --------------------------------------------------------------------------- #
# Stage wrapper: publishes the merged state after a successful stage
# --------------------------------------------------------------------------- #
class _PublishingReporter:
    def __init__(self) -> None:
        self.published: list = []

    async def stage_start(self, *a):
        pass

    async def stage_finish(self, *a):
        pass

    async def stage_fail(self, *a):
        pass

    async def record_stage_tokens(self, *a):
        pass

    async def publish_artifacts(self, job_id, state):
        self.published.append((job_id, state))


async def test_wrapper_publishes_merged_state_after_success():
    rep = _PublishingReporter()

    async def node(state):
        return {"style_ref_path": "/r.png"}

    out = await _instrument("designer", 5, node, rep)({"job_id": "j1", "script_path": "/s.json"})
    assert out == {"style_ref_path": "/r.png"}  # return value untouched
    assert rep.published == [("j1", {"job_id": "j1", "script_path": "/s.json", "style_ref_path": "/r.png"})]


async def test_wrapper_does_not_publish_when_stage_fails():
    rep = _PublishingReporter()

    async def node(state):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await _instrument("designer", 5, node, rep)({"job_id": "j1"})
    assert rep.published == []


async def test_reporter_publish_artifacts_replaces_rows_and_never_raises(session):
    job_id = (await repo.create_job(session, "p", 30)).id
    rep = ProgressReporter(get_sessionmaker())
    await rep.publish_artifacts(job_id, {"script_path": "/s.json"})
    await rep.publish_artifacts(job_id, {"script_path": "/s.json", "style_ref_path": "/r.png"})
    async with get_sessionmaker()() as fresh:
        got = await repo.get_job(fresh, job_id)
        assert sorted(a.kind for a in got.artifacts) == ["script", "style_ref"]
    # A malformed state is swallowed (presentation only, never fatal).
    await rep.publish_artifacts(job_id, {"frames": [{"oops": 1}]})


# --------------------------------------------------------------------------- #
# image_gen: each clean frame is saved + published the moment it renders
# --------------------------------------------------------------------------- #
class _RecordingReporter:
    def __init__(self) -> None:
        self.frames: list[tuple[int, str]] = []

    def log(self, *a, **k):
        pass

    async def progress(self, *a, **k):
        pass

    async def mark_degraded(self, *a, **k):
        pass

    async def publish_frame(self, order, path):
        self.frames.append((order, path))


class _Image:
    def generate_image(self, prompt, **_k):
        return f"IMG-{prompt}".encode()


async def test_image_gen_publishes_each_frame_and_result_is_unchanged(artifact_root):
    settings = Settings(gemini_api_key="", use_mock_providers=True, image_max_concurrency=2)
    store = ArtifactStore(artifact_root, "live-frames")
    rep = _RecordingReporter()
    deps = Deps(settings=settings, store=store, llm=LLMAdapter(settings), image=_Image(), reporter=rep)
    shotlist = Shotlist(
        style_bible="bible",
        shots=[Shot(order=i, image_prompt=f"p{i}", narration=f"c{i}") for i in (1, 2, 3)],
    )
    out = await image_gen({"shotlist_path": store.save_json("shotlist.json", shotlist)}, deps)

    assert sorted(o for o, _ in rep.frames) == [1, 2, 3]
    # The published file is the final frame (same path, same bytes).
    final = {f["order"]: f["path"] for f in out["frames"]}
    for order, path in rep.frames:
        assert path == final[order]
        assert store.load_bytes(path).startswith(b"IMG-")


# --------------------------------------------------------------------------- #
# GET /storyboard/{id}/live
# --------------------------------------------------------------------------- #
async def test_live_empty_running_job(client):
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
        await repo.set_status(s, job.id, JobStatus.RUNNING)
    body = (await client.get(f"/storyboard/{job.id}/live")).json()
    assert body["job_id"] == job.id
    assert body["title"] is None and body["beats"] == [] and body["frames"] == []
    assert body["version"]


async def test_live_mid_run_assets_and_version_bump(client, artifact_root):
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
        await repo.set_status(s, job.id, JobStatus.RUNNING)
        store = ArtifactStore(artifact_root, job.id)
        script = store.save_json(
            "script.json",
            {
                "title": "The Keeper",
                "logline": "A keeper finds a bottle.",
                "characters": [{"name": "Maya", "look": "x"}, {"name": "Ravi", "look": "y"}],
                "scenes": [
                    {"order": 2, "narration": "", "lines": [{"speaker": "MAYA", "text": "Who sent this?"}]},
                    {"order": 1, "narration": "The lamp turns.", "lines": []},
                    {"order": 3, "description": "Waves at dawn."},
                ],
            },
        )
        # A newer judge revision with extra dimensions: read generically.
        reflection = store.save_json(
            "reflection.json",
            {"final_overall": 84, "report": {"hook": 80, "structure": 90, "dialogue": 0,
                                             "overall": 84, "weak_beats": [2], "retelling": "…"}},
        )
        maya = store.save_bytes("cast_maya.png", b"M")
        await repo.add_artifact(s, job.id, kind="script", path=script)
        await repo.add_artifact(s, job.id, kind="reflection", path=reflection)
        await repo.add_artifact(s, job.id, kind="cast:MAYA", path=maya)
        await repo.upsert_artifact(s, job.id, kind="frame", path=store.save_bytes("frame_02.png", b"F2"), order_index=2)

    body = (await client.get(f"/storyboard/{job.id}/live")).json()
    assert body["title"] == "The Keeper"
    assert [(b["order"], b["text"], b["speaker"]) for b in body["beats"]] == [
        (1, "The lamp turns.", None),
        (2, "Who sent this?", "MAYA"),
        (3, "Waves at dawn.", None),
    ]
    assert body["scores"] == {"hook": 80, "structure": 90}
    assert body["score_overall"] == 84
    assert [c["name"] for c in body["cast"]] == ["Maya", "Ravi"]
    assert body["cast"][0]["portrait_url"] and body["cast"][1]["portrait_url"] is None
    assert [f["order"] for f in body["frames"]] == [2]
    assert (await client.get(body["frames"][0]["url"])).content == b"F2"

    async with get_sessionmaker()() as s:
        await repo.upsert_artifact(s, job.id, kind="frame", path=store.save_bytes("frame_01.png", b"F1"), order_index=1)
    body2 = (await client.get(f"/storyboard/{job.id}/live")).json()
    assert body2["version"] != body["version"]
    assert [f["order"] for f in body2["frames"]] == [1, 2]


async def test_live_tolerates_corrupt_json(client, artifact_root):
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
        store = ArtifactStore(artifact_root, job.id)
        path = store.save_bytes("script.json", b"{not json")
        await repo.add_artifact(s, job.id, kind="script", path=path)
    r = await client.get(f"/storyboard/{job.id}/live")
    assert r.status_code == 200 and r.json()["beats"] == []


async def test_live_unknown_job_404(client):
    assert (await client.get("/storyboard/nope/live")).status_code == 404


# --------------------------------------------------------------------------- #
# SSE: an ``assets`` event carries the same version as /live
# --------------------------------------------------------------------------- #
async def test_sse_emits_assets_version(client, artifact_root):
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
        store = ArtifactStore(artifact_root, job.id)
        await repo.add_artifact(s, job.id, kind="script", path=store.save_json("script.json", {"title": "T"}))
        await repo.set_status(s, job.id, JobStatus.COMPLETED)

    version = (await client.get(f"/storyboard/{job.id}/live")).json()["version"]
    lines: list[str] = []
    async with client.stream("GET", f"/storyboard/{job.id}/events") as resp:
        async for line in resp.aiter_lines():
            lines.append(line)
            if "done" in line or len(lines) > 400:
                break
    text = "\n".join(lines)
    assert "event: assets" in text
    assert version in text
