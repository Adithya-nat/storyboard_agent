"""Client for the external storyboard motion engine (per-scene "live photo" clips).

Mirrors the httpx style of ``turnstile.py`` and the ``adapters/errors.py`` taxonomy
so the motion node reasons about failures like every other adapter. Used as an
async context manager for the duration of one motion-stage run so a single
connection pool serves all the submit/poll/download calls:

    async with deps.motion as mc:
        task_id = await mc.submit(image_bytes, meta)
        ...

``health`` is deliberately usable WITHOUT entering the context (its own 3s client)
so the node can probe the engine and degrade fast before opening the pool.
"""

from __future__ import annotations

import json

import httpx

from app.adapters.errors import ProviderError, TransientProviderError
from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)


class MotionEngineClient:
    def __init__(self, settings: Settings) -> None:
        self._base = settings.motion_engine_url.rstrip("/")
        self._timeout = settings.motion_http_timeout_seconds
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "MotionEngineClient":
        self._client = httpx.AsyncClient(base_url=self._base, timeout=self._timeout)
        return self

    async def __aexit__(self, *exc) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def health(self) -> dict | None:
        """Capability probe. Returns the /health JSON, or None on ANY error (so the
        node degrades in ~3s instead of hanging when the engine is down)."""
        try:
            async with httpx.AsyncClient(base_url=self._base, timeout=3.0) as c:
                r = await c.get("/health")
                r.raise_for_status()
                return r.json()
        except Exception:  # noqa: BLE001 - unreachable/malformed => degrade
            return None

    async def submit(self, image: bytes, meta: dict) -> str:
        """Enqueue one frame for animation; return the engine task id."""
        client = self._require_client()
        try:
            r = await client.post(
                "/animate",
                files={"image": ("frame.png", image, "image/png")},
                data={"meta": json.dumps(meta)},
            )
            r.raise_for_status()
            return r.json()["task_id"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise TransientProviderError(f"motion submit failed: {exc}") from exc

    async def get_task(self, task_id: str) -> dict:
        client = self._require_client()
        try:
            r = await client.get(f"/tasks/{task_id}")
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise TransientProviderError(f"motion get_task failed: {exc}") from exc

    async def download(self, task_id: str) -> bytes:
        client = self._require_client()
        try:
            r = await client.get(f"/tasks/{task_id}/video")
            r.raise_for_status()
            return r.content
        except httpx.HTTPError as exc:
            raise TransientProviderError(f"motion download failed: {exc}") from exc

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise ProviderError("MotionEngineClient used outside its async context")
        return self._client
