"""Takes and cuts: every file a recut replaces is moved (never deleted) into
``takes/<take id>/``, and every recut is recorded as a numbered cut.

    takes/index.json      {"takes": [{id, cut, kind, unit, label, files, at}]}
    recut/history.json    {"cuts":  [{cut, at, picked, items, note, summary, cost_start, scene_seconds}]}
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from app.redo.targets import Item, files_for

RESTORABLE = frozenset({"frame", "clip", "voice", "score"})


def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_takes(job_dir: Path) -> list[dict]:
    return _read(job_dir / "takes" / "index.json", {"takes": []}).get("takes", [])


def history(job_dir: Path) -> list[dict]:
    return _read(job_dir / "recut" / "history.json", {"cuts": []}).get("cuts", [])


def current_cut(job_dir: Path) -> int:
    cuts = history(job_dir)
    return cuts[-1]["cut"] if cuts else 1


def archive(job_dir: Path, items: list[Item], cut: int) -> list[dict]:
    """Move each item's current files into a take of ``cut``; returns the new take records."""
    takes = list_takes(job_dir)
    made: list[dict] = []
    for item in items:
        names = files_for(job_dir, item)
        if not names:
            continue
        take_id = f"c{cut}-{item.key}"
        dest = job_dir / "takes" / take_id
        dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            os.replace(job_dir / name, dest / name)
        rec = {"id": take_id, "cut": cut, "kind": item.kind, "unit": item.unit, "label": item.label,
               "files": names, "at": _now()}
        takes = [t for t in takes if t["id"] != take_id] + [rec]
        made.append(rec)
    _write(job_dir / "takes" / "index.json", {"takes": takes})
    gone = {i.unit for i in items if i.kind == "clip"}
    manifest = _read(job_dir / "motion_manifest.json", None)
    if gone and manifest:  # a clip that comes back later is re-measured, not trusted from the old entry
        manifest["scenes"] = [c for c in manifest.get("scenes", []) if c.get("order") not in gone]
        _write(job_dir / "motion_manifest.json", manifest)
    return made


def take_group(job_dir: Path, take_id: str) -> list[dict]:
    """The take plus its siblings from the same cut and scene (a frame comes back with its clip)."""
    takes = list_takes(job_dir)
    take = next((t for t in takes if t["id"] == take_id), None)
    if take is None or take["kind"] not in RESTORABLE:
        return []
    return [t for t in takes if t["cut"] == take["cut"] and t["unit"] == take["unit"] and t["kind"] in RESTORABLE]


def bring_back(job_dir: Path, group: list[dict]) -> None:
    """Copy a take group's files back into place (the take itself stays, so it can be restored again)."""
    import shutil

    for t in group:
        for name in t["files"]:
            src = job_dir / "takes" / t["id"] / name
            if src.exists():
                shutil.copy2(src, job_dir / name)


def scene_seconds(job_dir: Path) -> dict[str, float]:
    """Seconds per scene in the current cut: the real clip length, else the planned scene length."""
    out: dict[str, float] = {}
    script = _read(job_dir / "script.json", {})
    for sc in script.get("scenes", []):
        if "order" in sc:
            out[str(sc["order"])] = float(sc.get("seconds") or 0)
    for c in _read(job_dir / "motion_manifest.json", {}).get("scenes", []):
        if c.get("source_seconds") and (job_dir / f"motion_{int(c['order']):02d}.mp4").exists():
            out[str(c["order"])] = round(float(c["source_seconds"]), 2)
    return out


def start_cut(job_dir: Path, *, picked: list[Item], items: list[Item], note: str, summary: str,
              cost_now: float, kind: str = "recut", created_at: str | None = None) -> int:
    """Close the current cut (snapshot its timing) and open the next one; returns its number."""
    cuts = history(job_dir)
    if not cuts:
        cuts = [{"cut": 1, "kind": "first", "at": created_at or _now(), "picked": [], "items": [], "note": "",
                 "summary": "First cut", "cost_start": 0.0}]
    cuts[-1]["scene_seconds"] = scene_seconds(job_dir)
    cuts[-1]["cost_end"] = round(cost_now, 4)
    n = cuts[-1]["cut"] + 1
    cuts.append({"cut": n, "kind": kind, "at": _now(), "picked": [i.key for i in picked],
                 "items": [i.key for i in items], "note": note, "summary": summary, "cost_start": round(cost_now, 4)})
    _write(job_dir / "recut" / "history.json", {"cuts": cuts})
    return n


def save_plan(job_dir: Path, plan_id: str, data: dict) -> None:
    _write(job_dir / "recut" / f"plan_{plan_id}.json", data)


def load_plan(job_dir: Path, plan_id: str) -> dict | None:
    if not plan_id.isalnum():
        return None
    return _read(job_dir / "recut" / f"plan_{plan_id}.json", None)
