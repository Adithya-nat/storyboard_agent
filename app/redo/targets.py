"""What a recut rebuilds: each pickable item, the files it owns, and what depends on it.

Every stage is load-or-produce, so redoing an item means moving its files out of the way
(into a take) and resuming: the stage regenerates exactly what is missing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

KINDS = ("story", "look", "scene", "frame", "clip", "voice", "score", "release")
PER_SCENE = frozenset({"scene", "frame", "clip", "voice"})
# Derived items: rebuilt by the closure, never picked directly.
DERIVED = ("captions", "edit")

LABEL = {
    "story": "Story", "look": "Look", "scene": "Scene", "frame": "Frame", "clip": "Clip",
    "voice": "Voice", "score": "Score", "release": "Release", "captions": "Captions", "edit": "Edit",
}

# The stages that do an item's work (reset to pending before the recut runs).
STAGES = {
    "story": ("script_writer", "dialogue_pass", "reflection"),
    "look": ("designer",),
    "frame": ("image_gen",),
    "clip": ("motion",),
    "voice": ("narration",),
    "captions": ("captions",),
    "score": ("music",),
    "edit": ("assembler", "film_review"),
    "release": ("publish_kit",),
}


def _scene_files(kind: str, n: int) -> list[str]:
    nn = f"{n:02d}"
    return {
        "frame": [f"frame_{nn}.png", f"frame_{nn}.degraded", f"frame_{nn}.blocked.png"],
        "clip": [f"motion_{nn}.mp4"],
        "voice": [f"narration_{nn}.wav", f"narration_{nn}.failed"],
        "captions": [f"dialogue_{nn}.wav", f"dialogue_{nn}_clip.wav", f"cap_{nn}.png", f"capov_{nn}.png"],
        # A rewritten scene drops the earlier content-policy rewrite: the new text replaces it.
        "scene": [f"rescue_{nn}.json", f"rescue_{nn}.attempts.json"],
    }.get(kind, [])


_FILM_FILES = {
    "story": ["script.json", "dialogue_pass.json", "reflection.json"],
    "look": ["shotlist.json", "style_bible.md", "style_ref.png"],
    "score": ["music.mp3", "music.wav", "music.json"],
    "edit": ["storyboard.mp4", "storyboard_muxed.mp4", "narration_track.wav", "scene_audio.json", "film_review.json"],
    "release": ["publish.json", "cover.png"],
}
_LOOK_GLOBS = ("char_*.png", "loc_*.png")


@dataclass(frozen=True, order=True)
class Item:
    kind: str
    unit: int | None = None

    @property
    def key(self) -> str:
        return f"{self.kind}-{self.unit}" if self.unit is not None else self.kind

    @property
    def label(self) -> str:
        return f"{LABEL[self.kind]} {self.unit}" if self.unit is not None else LABEL[self.kind]


@dataclass
class Invalidation:
    picked: list[Item]
    items: list[Item] = field(default_factory=list)  # picked + everything rebuilt because of them
    stages: list[str] = field(default_factory=list)  # pipeline order

    def units(self, kind: str) -> list[int]:
        return sorted(i.unit for i in self.items if i.kind == kind and i.unit is not None)

    def has(self, kind: str) -> bool:
        return any(i.kind == kind for i in self.items)


def parse(kind: str, unit: int | None, scenes: list[int]) -> Item:
    if kind not in KINDS:
        raise ValueError(f"unknown item {kind!r}")
    if kind in PER_SCENE:
        if unit not in scenes:
            raise ValueError(f"{LABEL[kind]} needs a scene number from {scenes}")
        return Item(kind, unit)
    return Item(kind)


def closure(picked: list[Item], scenes: list[int], pipeline: list[str]) -> Invalidation:
    """Picked items plus everything that must be rebuilt because of them."""
    out: set[Item] = set(picked)
    every = lambda kind: {Item(kind, n) for n in scenes}  # noqa: E731
    for it in picked:
        if it.kind == "story":
            out |= {Item("look")} | every("scene") | every("frame") | every("clip") | every("voice") | every("captions")
            out |= {Item("score"), Item("release")}
        elif it.kind == "look":
            out |= every("frame") | every("clip") | every("captions") | {Item("release")}
        elif it.kind == "scene":
            n = it.unit
            out |= {Item("frame", n), Item("clip", n), Item("voice", n), Item("captions", n)}
        elif it.kind == "frame":
            out |= {Item("clip", it.unit), Item("captions", it.unit), Item("release")}
        elif it.kind in ("clip", "voice"):
            out.add(Item("captions", it.unit))
    if any(i.kind != "release" for i in out):
        out.add(Item("edit"))
    if "motion" not in pipeline:
        out = {i for i in out if i.kind != "clip"}
    names = {s for i in out for s in STAGES.get(i.kind, ())}
    if any(i.kind == "scene" for i in out):
        names.add("narration")
    return Invalidation(
        picked=sorted(picked), items=sorted(out), stages=[s for s in pipeline if s in names],
    )


def files_for(job_dir: Path, item: Item) -> list[str]:
    """The item's files that exist right now (names relative to the job folder)."""
    if item.unit is not None:
        names = _scene_files(item.kind, item.unit)
    else:
        names = list(_FILM_FILES.get(item.kind, []))
        if item.kind == "look":
            names += sorted(p.name for g in _LOOK_GLOBS for p in job_dir.glob(g))
    return [n for n in names if (job_dir / n).exists()]


_MEDIA = re.compile(
    r"^(takes/[a-z0-9-]+/)?("
    r"frame_\d{2}(\.blocked)?\.png|motion_\d{2}\.mp4|narration_\d{2}\.wav|style_ref\.png|"
    r"char_[A-Z0-9_]+_\d\.png|loc_[A-Za-z0-9_]+\.png|music\.(mp3|wav)|storyboard(_muxed)?\.mp4|cover\.png"
    r")$"
)


def media_path(job_dir: Path, name: str) -> Path | None:
    """A servable media file inside the job folder (current or archived take), else None."""
    if not _MEDIA.match(name):
        return None
    path = (job_dir / name).resolve()
    if job_dir.resolve() not in path.parents or not path.is_file():
        return None
    return path
