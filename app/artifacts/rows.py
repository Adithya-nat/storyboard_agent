"""Run state -> artifact rows (shared by the final persist and the live publish)."""

from __future__ import annotations

import hashlib


def artifact_rows(state: dict) -> list[dict]:
    """Every artifact a (possibly partial) run state has produced, as
    ``replace_artifacts`` rows. Pure, so the final persist and the live per-stage
    publish (``ProgressReporter.publish_artifacts``) always agree on the set."""
    artifacts: list[dict] = []
    for kind in (
        "script_path",
        "reflection_path",
        "shotlist_path",
        "style_bible_path",
        "style_ref_path",
        "motion_manifest_path",
    ):
        if state.get(kind):
            artifacts.append({"kind": kind.removesuffix("_path"), "path": state[kind]})
    for f in state.get("frames", []) or []:
        artifacts.append(
            {"kind": "frame", "path": f["path"], "status": f["status"], "order_index": f["order"]}
        )
    # Per-scene motion clips (motion engine). Served for free by the existing
    # /artifacts/{id} route; mapped to scenes[].video_url in get_storyboard_content.
    for c in state.get("motion_clips", []) or []:
        artifacts.append(
            {
                "kind": "motion",
                "path": c["path"],
                "status": c.get("status", "ok"),
                "order_index": c["order"],
            }
        )
    if state.get("video_path"):
        artifacts.append({"kind": "video", "path": state["video_path"]})
    if state.get("narration_path"):
        artifacts.append({"kind": "narration", "path": state["narration_path"]})
    for kind in ("publish_path", "cover_path", "music_path", "scene_audio_path"):
        if state.get(kind):
            artifacts.append({"kind": kind.removesuffix("_path"), "path": state[kind]})
    for name, paths in (state.get("cast_paths") or {}).items():
        if paths:
            artifacts.append({"kind": f"cast:{name[:26]}", "path": paths[0]})
    return artifacts


def assets_version(artifacts) -> str:
    """Short, order-independent fingerprint of a job's artifact rows. Changes
    whenever an artifact lands (or is replaced), so live clients refetch
    ``GET /storyboard/{id}/live`` only when something new exists."""
    ids = sorted(f"{a.id}:{a.status}" for a in artifacts)
    return hashlib.sha1("|".join(ids).encode()).hexdigest()[:12]
