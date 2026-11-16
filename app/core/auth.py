"""Who is calling. AUTH_MODE=none -> one local user; jwt -> Better Auth tokens from the web app.

JSON routes take ``Authorization: Bearer <jwt>``. Media and SSE routes also accept a
short-lived signed ``?t=`` token, because <video> and EventSource can't send headers.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass
from functools import lru_cache

from fastapi import Depends, HTTPException, Request

from app.core.config import Settings, get_settings

LOCAL_USER_ID = "local"


@dataclass(frozen=True)
class User:
    id: str
    email: str = ""
    name: str = ""

    @property
    def local(self) -> bool:
        return self.id == LOCAL_USER_ID

    @property
    def scope(self) -> str | None:
        """user_id to filter rows by; None = single-user mode sees everything."""
        return None if self.local else self.id


LOCAL_USER = User(id=LOCAL_USER_ID, name="You")


def is_allowed(user: User, settings: Settings) -> bool:
    if user.local:
        return True
    email = user.email.lower()
    for entry in (e.strip().lower() for e in settings.allowed_users):
        if entry == "*" or (email and (entry == email or (entry.startswith("@") and email.endswith(entry)))):
            return True
    return False


@lru_cache
def _jwks_client(url: str):
    import jwt

    return jwt.PyJWKClient(url, cache_keys=True, lifespan=3600)


def verify_jwt(token: str, settings: Settings) -> User:
    import jwt

    try:
        key = _jwks_client(settings.auth_jwks_url).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            key.key,
            algorithms=["EdDSA", "ES256", "ES512", "RS256", "PS256"],
            audience=settings.auth_audience,
            issuer=settings.auth_issuer or None,
            options={"require": ["exp", "sub"], "verify_iss": bool(settings.auth_issuer)},
        )
    except jwt.PyJWKClientConnectionError as exc:
        raise HTTPException(status_code=503, detail="sign-in service unreachable") from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="sign in again (invalid or expired session)") from exc
    return User(id=str(claims["sub"]), email=str(claims.get("email") or ""), name=str(claims.get("name") or ""))


async def current_user(request: Request) -> User:
    """Signed-in user (not yet checked against the allowlist)."""
    settings = get_settings()
    if not settings.hosted:
        return LOCAL_USER
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="sign in to continue")
    user = await asyncio.to_thread(verify_jwt, header[7:].strip(), settings)
    request.state.user_id = user.id
    return user


async def allowed_user(user: User = Depends(current_user)) -> User:
    if not is_allowed(user, get_settings()):
        raise HTTPException(
            status_code=403,
            detail={"code": "not_allowlisted", "message": "Your account is waiting for beta access."},
        )
    return user


# --- media tokens -------------------------------------------------------------------- #
@lru_cache
def _fallback_secret() -> bytes:
    return secrets.token_bytes(32)


def _secret(settings: Settings) -> bytes:
    return settings.asset_signing_secret.encode() if settings.asset_signing_secret else _fallback_secret()


def _sign(payload: str, settings: Settings) -> str:
    mac = hmac.new(_secret(settings), payload.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode().rstrip("=")


def make_media_token(user: User, settings: Settings, now: float | None = None) -> tuple[str, int]:
    exp = int((now or time.time()) + settings.media_token_ttl_seconds)
    payload = f"{user.id}.{exp}"
    return f"{payload}.{_sign(payload, settings)}", exp


def read_media_token(token: str, settings: Settings, now: float | None = None) -> User | None:
    try:
        uid, exp, sig = token.rsplit(".", 2)
        if not hmac.compare_digest(sig, _sign(f"{uid}.{exp}", settings)):
            return None
        if int(exp) < (now or time.time()):
            return None
    except ValueError:
        return None
    return User(id=uid)


async def media_user(request: Request) -> User:
    """For media + SSE: the ?t= token, else the normal bearer check."""
    settings = get_settings()
    if not settings.hosted:
        return LOCAL_USER
    token = request.query_params.get("t")
    if token:
        user = read_media_token(token, settings)
        if user is None:
            raise HTTPException(status_code=401, detail="media link expired; reload the page")
        return user
    return await allowed_user(await current_user(request))
