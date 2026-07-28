"""Cloudflare Turnstile verification — guards the public, un-authed create endpoint.

The frontend renders a Turnstile widget and sends its token with POST /storyboard.
We verify that token server-side against Cloudflare before spending Gemini quota.

OFF by default (``settings.turnstile_enabled``) so local/mock dev and tests never
need a key: when disabled, verification is a no-op that always passes.
"""

from __future__ import annotations

import httpx

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)


async def verify_turnstile(
    settings: Settings, token: str | None, remote_ip: str | None = None
) -> bool:
    """Return True if the request may proceed.

    - Turnstile disabled -> always True (dev/test).
    - Enabled but no token -> False.
    - Enabled with a token -> True iff Cloudflare's siteverify says ``success``.
      A network/parse error fails CLOSED (False) so a Turnstile outage can't be
      used to bypass the guard.
    """
    if not settings.turnstile_enabled:
        return True
    if not token:
        return False

    data = {"secret": settings.turnstile_secret_key, "response": token}
    if remote_ip:
        data["remoteip"] = remote_ip
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(settings.turnstile_verify_url, data=data)
        return bool(resp.json().get("success"))
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("Turnstile verification failed (treating as invalid): %s", exc)
        return False
