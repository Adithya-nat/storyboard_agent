"""Who am I + check a Gemini key, for the web app's sign-in and key dialogs."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.core.auth import User, allowed_user, current_user, is_allowed, make_media_token
from app.core.config import get_settings
from app.core.keys import looks_like_key
from app.core.model_check import check_key

router = APIRouter(tags=["account"])


class Me(BaseModel):
    id: str
    email: str
    name: str
    allowed: bool
    auth_mode: str
    key_mode: str
    retention_hours: int
    media_token: str | None = None
    media_token_expires_at: datetime | None = None


@router.get("/me", response_model=Me)
async def me(user: User = Depends(current_user)) -> Me:
    s = get_settings()
    allowed = is_allowed(user, s)
    token, exp = make_media_token(user, s) if allowed and s.hosted else (None, None)
    return Me(
        id=user.id, email=user.email, name=user.name, allowed=allowed,
        auth_mode=s.auth_mode, key_mode=s.key_mode, retention_hours=s.job_ttl_seconds // 3600,
        media_token=token,
        media_token_expires_at=datetime.fromtimestamp(exp, timezone.utc) if exp else None,
    )


@router.post("/keys/check")
async def keys_check(request: Request, user: User = Depends(allowed_user)) -> dict:
    """Lists the models the key can see (free). The key is not stored or logged."""
    del user
    key = (request.headers.get("x-gemini-key") or "").strip()
    if not key or not looks_like_key(key):
        raise HTTPException(status_code=400, detail="Paste a Gemini API key first.")
    return await check_key(get_settings(), key)
