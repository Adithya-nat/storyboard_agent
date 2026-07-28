"""ArtifactStore — filesystem persistence for artifacts, addressed by reference.

    save_*  ->  writes bytes/text/json under  <root>/<job_id>/<name>
            ->  returns the absolute path (the "reference")
    load_*  ->  reads back by path

Writes are ATOMIC (temp file + os.replace): a crash mid-write can never leave a
half-written file that a *resumed* job would later load as if it were complete.
This is the load-bearing guarantee behind per-node idempotency (Phase 2 resume).

The store deliberately knows nothing about the database. The DB only ever holds
the returned path + metadata (see app/db/models.py), keeping LLM context and DB
rows free of binary payloads. Swapping to S3/GCS later = reimplement this class.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel


class ArtifactStore:
    def __init__(self, root: Path | str, job_id: str) -> None:
        self.job_dir = Path(root) / job_id
        self.job_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.job_dir / name

    def _atomic_write(self, name: str, data: bytes) -> str:
        """Write ``data`` to <job_dir>/<name> atomically; return the path string."""
        path = self._path(name)
        # Temp file in the SAME directory so os.replace() is an atomic rename
        # (rename is only atomic within one filesystem).
        fd, tmp = tempfile.mkstemp(dir=self.job_dir, prefix=".tmp-", suffix=f"-{name}")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise
        return str(path)

    # --- writes ---------------------------------------------------------- #
    def save_bytes(self, name: str, data: bytes) -> str:
        return self._atomic_write(name, data)

    def save_text(self, name: str, text: str) -> str:
        return self._atomic_write(name, text.encode("utf-8"))

    def save_json(self, name: str, model: BaseModel | dict) -> str:
        payload = model.model_dump() if isinstance(model, BaseModel) else model
        return self.save_text(name, json.dumps(payload, indent=2, ensure_ascii=False))

    # --- reads ----------------------------------------------------------- #
    def load_bytes(self, path: str) -> bytes:
        return Path(path).read_bytes()

    def load_text(self, path: str) -> str:
        return Path(path).read_text(encoding="utf-8")

    def load_json(self, path: str) -> dict:
        return json.loads(self.load_text(path))

    # --- idempotency helpers (Phase 2 resume) ---------------------------- #
    def exists(self, name: str) -> bool:
        """True if an artifact with this name is already on disk."""
        return self._path(name).exists()

    def abspath(self, name: str) -> str:
        """The reference path for ``name`` (whether or not it exists yet)."""
        return str(self._path(name))

    def write_marker(self, name: str) -> None:
        """Drop a zero-byte sidecar marker (e.g. ``frame_03.degraded``).

        Used so a degraded frame's status survives a restart: on resume we can
        tell a real frame from a placeholder and re-try it, keeping the final
        job status/warnings honest instead of silently reporting COMPLETED.
        """
        self._path(name).write_text("", encoding="utf-8")

    def remove(self, name: str) -> None:
        """Delete an artifact/marker if present (no error if missing)."""
        self._path(name).unlink(missing_ok=True)
