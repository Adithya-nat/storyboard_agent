"""ArtifactStore save/load roundtrip — guards against path drift / orphan blobs."""

from __future__ import annotations

from app.artifacts.store import ArtifactStore
from app.schemas import Scene, Script


def test_bytes_roundtrip(artifact_root):
    store = ArtifactStore(artifact_root, "job-bytes")
    path = store.save_bytes("frame.png", b"\x89PNG-data")
    assert store.load_bytes(path) == b"\x89PNG-data"


def test_json_roundtrip_with_pydantic(artifact_root):
    store = ArtifactStore(artifact_root, "job-json")
    script = Script(
        title="t", logline="l", style="s",
        scenes=[Scene(order=1, description="d", narration="n")],
    )
    path = store.save_json("script.json", script)
    loaded = Script.model_validate(store.load_json(path))
    assert loaded == script


def test_atomic_write_leaves_no_temp_files(artifact_root):
    store = ArtifactStore(artifact_root, "job-atomic")
    store.save_bytes("frame.png", b"data")
    # temp-file pattern from the atomic temp+rename must be cleaned up.
    leftovers = list(store.job_dir.glob(".tmp-*"))
    assert leftovers == []


def test_exists_and_degraded_marker(artifact_root):
    store = ArtifactStore(artifact_root, "job-marker")
    assert not store.exists("frame_01.png")
    store.save_bytes("frame_01.png", b"x")
    assert store.exists("frame_01.png")

    # marker lifecycle (drives image_gen resume: re-try degraded, skip clean).
    assert not store.exists("frame_01.degraded")
    store.write_marker("frame_01.degraded")
    assert store.exists("frame_01.degraded")
    store.remove("frame_01.degraded")
    assert not store.exists("frame_01.degraded")
    store.remove("frame_01.degraded")  # idempotent: no error when already gone
