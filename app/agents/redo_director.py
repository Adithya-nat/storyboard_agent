"""Redo Director: turns a creator's note on picked parts of a film into concrete edits.

One structured text call. Its scene edits land in script.json / shotlist.json (the files the
stages already read); whole-film notes land in direction.json, which the story, look, score
and release prompts read. Nothing here spends on media.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from app.agents.brief import speech_block
from app.prompts import render
from app.redo.targets import Item
from app.schemas import CreativeBrief, DialogueLine, Script, ShotSize, Shotlist

DIRECTION_FILE = "direction.json"
NOTE_KINDS = ("story", "look", "score", "release")


class ShotEdit(BaseModel):
    index: int = Field(description="0-based position of the shot inside its scene.")
    action: str = Field(default="", description="English: what happens in this shot. Empty = unchanged.")
    size: ShotSize | None = Field(default=None, description="New shot size, or null to keep it.")
    camera: str = Field(default="", description="English camera move. Empty = unchanged.")
    lines: list[DialogueLine] = Field(
        default_factory=list,
        description="The shot's full new dialogue, in the film's language. Empty = keep the shot's lines.",
    )


class SceneEdit(BaseModel):
    order: int = Field(description="The scene number being changed.")
    description: str = Field(default="", description="English: the new scene description. Empty = unchanged.")
    image_prompt: str = Field(
        default="", description="English: full prompt for the scene's new opening frame. Empty = keep the frame."
    )
    narration: str = Field(
        default="", description="Narrated scenes only: the new voice-over line, in the film's language. Empty = unchanged."
    )
    shots: list[ShotEdit] = Field(default_factory=list, description="Only the shots that change.")


class ExtraItem(BaseModel):
    kind: str = Field(description="One of: frame, clip, voice, score, look, release.")
    unit: int | None = Field(default=None, description="Scene number for frame/clip/voice; null otherwise.")
    reason: str = Field(default="", description="One short line for the creator.")


class RedoPlan(BaseModel):
    summary: str = Field(description="One plain sentence for the creator: what will change.")
    scenes: list[SceneEdit] = Field(default_factory=list)
    story_note: str = ""
    look_note: str = ""
    score_note: str = ""
    release_note: str = ""
    extra: list[ExtraItem] = Field(default_factory=list)


def _read(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _scene_block(script: Script, shotlist: Shotlist | None, order: int, history: dict) -> str:
    sc = next((s for s in script.scenes if s.order == order), None)
    if sc is None:
        return ""
    shot = next((s for s in (shotlist.shots if shotlist else []) if s.order == order), None)
    rows = [f'<scene order="{order}" mode="{sc.mode}" seconds="{sc.seconds:g}" location="{sc.location}">',
            f"Description: {sc.description}"]
    if shot is not None:
        rows.append(f"Opening frame prompt: {shot.image_prompt}")
    for i, sh in enumerate(sc.shots):
        lines = "; ".join(f"{l.speaker}: {l.text}" for l in sh.lines) or "(no lines)"
        rows.append(f"  shot {i} [{sh.size}, {sh.seconds:g}s, {sh.camera}] {sh.action} | {lines}")
    if not sc.shots and sc.narration:
        rows.append(f"Narration: {sc.narration}")
    if history:
        rows.append(f"History: {json.dumps(history, ensure_ascii=False)[:1500]}")
    rows.append("</scene>")
    return "\n".join(rows)


def scene_history(job_dir: Path, order: int) -> dict:
    """What already happened to this scene: content blocks, accepted/failed rewrites, QC notes."""
    nn = f"{order:02d}"
    out: dict = {}
    if (r := _read(job_dir / f"rescue_{nn}.json")) is not None:
        out["accepted_rewrite"] = {"block": r.get("code"), "change": r.get("change")}
    if (a := _read(job_dir / f"rescue_{nn}.attempts.json")) is not None:
        out["blocked"] = {"code": a.get("code"), "failed_rewrites": [x.get("change") for x in a.get("rewrites", [])]}
    clip = next((c for c in (_read(job_dir / "motion_manifest.json", {}) or {}).get("scenes", [])
                 if c.get("order") == order), None)
    if clip and clip.get("qc"):
        out["qc_issues"] = clip["qc"]
    return out


def _mock_plan(script: Script, picked: list[Item], note: str) -> RedoPlan:
    """Mock mode: a plan that touches a shot, a line and (for frames) the opening picture."""
    edits, extra = [], []
    want = note or "a fresh take"
    for it in picked:
        if it.kind not in ("scene", "clip", "frame") or it.unit is None:
            continue
        sc = next((s for s in script.scenes if s.order == it.unit), None)
        if sc is None:
            continue
        shots = []
        if sc.shots:
            last = len(sc.shots) - 1
            lines = [DialogueLine(speaker=l.speaker, text=f"{l.text.rstrip('.')} ({want}).", delivery=l.delivery)
                     for l in sc.shots[last].lines]
            shots.append(ShotEdit(index=last, action=f"{sc.shots[last].action}, now {want}", lines=lines))
        edits.append(SceneEdit(order=it.unit, description=f"{sc.description} ({want})", shots=shots,
                               image_prompt=f"Opening frame: {want}" if it.kind == "frame" else ""))
        if it.kind == "clip":
            extra.append(ExtraItem(kind="frame", unit=it.unit, reason="the opening picture still shows the old action"))
    notes = {f"{k}_note": want for k in NOTE_KINDS if any(i.kind == k for i in picked)}
    return RedoPlan(summary=f"Mock plan: {want}", scenes=edits, extra=extra, **notes)


def direct(llm, job_dir: Path, picked: list[Item], note: str, language: str) -> RedoPlan:
    """Ask the director for a plan (sync; run it in a thread)."""
    script = Script.model_validate(_read(job_dir / "script.json"))
    raw_shotlist = _read(job_dir / "shotlist.json")
    shotlist = Shotlist.model_validate(raw_shotlist) if raw_shotlist else None
    raw_brief = _read(job_dir / "brief.json")
    brief = CreativeBrief.model_validate(raw_brief) if raw_brief else None
    orders = sorted({i.unit for i in picked if i.unit is not None})
    histories = {o: scene_history(job_dir, o) for o in orders}
    codes = {h["blocked"]["code"] for h in histories.values() if "blocked" in h} | {
        h["accepted_rewrite"]["block"] for h in histories.values() if "accepted_rewrite" in h}
    rules = "\n".join(render({"content_blocked": "omni/rescue_rules_third_party", "safety": "omni/rescue_rules_safety"}
                             .get(c, "omni/rescue_rules_prohibited")) for c in sorted(c for c in codes if c))
    neighbours = sorted({n for o in orders for n in (o - 1, o, o + 1)})
    blocks = "\n".join(_scene_block(script, shotlist, n, histories.get(n, {})) for n in neighbours)
    if not orders:  # whole-film items: give the director the outline
        blocks = "\n".join(f"{s.order}. [{s.beat_role}] {s.description}" for s in script.scenes)
    return llm.complete_json(
        system=render("redo/system", speech=speech_block(brief, language),
                      safety_rules=rules or "No scene in this request has been blocked before."),
        user=render(
            "redo/user", note=note.strip() or "(no note: give the picked items a fresh take)",
            picked=", ".join(i.label for i in picked), title=script.title, logline=script.logline,
            language=language, characters=", ".join(f"{c.name} ({c.look[:80]})" for c in script.characters) or "none",
            scenes=blocks, history="Neighbouring scenes are context only; change only picked scenes.",
        ),
        response_model=RedoPlan,
        role="designer",
        mock_factory=lambda _u: _mock_plan(script, picked, note),
    )


def edited(script: Script, shotlist: Shotlist | None, plan: RedoPlan,
           allowed: set[int] | None = None) -> tuple[Script, Shotlist | None, dict[int, bool]]:
    """Apply the plan's scene edits in memory. Returns (script, shotlist, {order: frame changed}).

    Only ``allowed`` scenes change (all when None). Lines naming an unknown speaker are dropped
    for that shot, so its old lines stay."""
    cast = {c.name.strip().upper() for c in script.characters}
    changed: dict[int, bool] = {}
    for edit in plan.scenes:
        sc = next((s for s in script.scenes if s.order == edit.order), None)
        if sc is None or (allowed is not None and edit.order not in allowed):
            continue
        if edit.description.strip():
            sc.description = edit.description.strip()
        if edit.narration.strip() and sc.mode == "narration":
            sc.narration = edit.narration.strip()
        for se in edit.shots:
            if not 0 <= se.index < len(sc.shots):
                continue
            shot = sc.shots[se.index]
            shot.action = se.action.strip() or shot.action
            shot.camera = se.camera.strip() or shot.camera
            shot.size = se.size or shot.size
            if se.lines and all(l.speaker.strip().upper() in cast for l in se.lines):
                shot.lines = [DialogueLine(speaker=l.speaker.strip().upper(), text=l.text, delivery=l.delivery)
                              for l in se.lines]
        if sc.shots:
            sc.lines = sc.shot_lines()
        redraw = bool(edit.image_prompt.strip())
        for s in (shotlist.shots if shotlist is not None else []):
            if s.order == edit.order:
                if redraw:
                    s.image_prompt = edit.image_prompt.strip()
                if sc.mode == "narration":
                    s.narration = sc.narration
        changed[edit.order] = redraw
    return script, shotlist, changed


def apply_plan(job_dir: Path, plan: RedoPlan, picked: list[Item] = ()) -> dict[int, bool]:
    """Write the plan's edits into script.json / shotlist.json / direction.json; returns
    {scene order: opening frame changed}. A picked whole-film item's note replaces the old one."""
    from app.artifacts.store import ArtifactStore

    store = ArtifactStore(job_dir.parent, job_dir.name)
    changed: dict[int, bool] = {}
    raw = _read(job_dir / "script.json")
    if raw:  # a story recut archived the script: there is nothing to edit in place
        raw_shotlist = _read(job_dir / "shotlist.json")
        allowed = {i.unit for i in picked if i.unit is not None} if picked else None
        script, shotlist, changed = edited(
            Script.model_validate(raw), Shotlist.model_validate(raw_shotlist) if raw_shotlist else None, plan, allowed)
        store.save_json("script.json", script)
        if shotlist is not None:
            store.save_json("shotlist.json", shotlist)
    kinds = {i.kind for i in picked}
    notes = {k: getattr(plan, f"{k}_note").strip() for k in NOTE_KINDS
             if k in kinds or getattr(plan, f"{k}_note").strip()}
    if notes:
        prior = _read(job_dir / DIRECTION_FILE, {}) or {}
        store.save_json(DIRECTION_FILE, {**prior, **notes})
    return changed


def direction(store, kind: str) -> str:
    """The creator's standing note for one crew member, ready to drop into its prompt ('' if none)."""
    if not store.exists(DIRECTION_FILE):
        return ""
    note = (store.load_json(store.abspath(DIRECTION_FILE)) or {}).get(kind, "").strip()
    return f"\nCreator's direction for this version (follow it): {note}\n" if note else ""
