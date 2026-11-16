"""Bring-your-own-key plumbing: per-key rate-limit scope + secret redaction.

A tester's key lives only in memory for the job that uses it. The scope (a short
hash of the key) is bound in a contextvar so every limiter/governor in that job
uses the tester's own buckets, matching Google's per-key quotas.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from contextvars import ContextVar, Token

_scope: ContextVar[str] = ContextVar("key_scope", default="")
_secrets: dict[str, int] = {}
_lock = threading.Lock()

_KEY_SHAPE = re.compile(r"^[A-Za-z0-9_\-]{20,200}$")
_GOOGLE_KEY = re.compile(r"AIza[0-9A-Za-z_\-]{35}")
REDACTED = "[redacted-key]"


def looks_like_key(key: str) -> bool:
    return bool(_KEY_SHAPE.match(key))


def fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def current_scope() -> str:
    return _scope.get()


def bind_key(key: str | None) -> Token | None:
    """Scope limiters to this key and mark it for redaction. Pair with ``release_key``."""
    if not key:
        return None
    with _lock:
        _secrets[key] = _secrets.get(key, 0) + 1
    return _scope.set(fingerprint(key))


def release_key(key: str | None, token: Token | None) -> None:
    if token is not None:
        _scope.reset(token)
    if key:
        with _lock:
            n = _secrets.get(key, 0) - 1
            if n > 0:
                _secrets[key] = n
            else:
                _secrets.pop(key, None)


def redact(text: str) -> str:
    if not text:
        return text
    with _lock:
        live = list(_secrets)
    for key in live:
        text = text.replace(key, REDACTED)
    return _GOOGLE_KEY.sub(REDACTED, text)


class RedactKeysFilter(logging.Filter):
    """Handler filter: strips API keys from the message and any traceback."""

    _fmt = logging.Formatter()

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact(record.getMessage())
            record.args = None
            if record.exc_info and not record.exc_text:
                record.exc_text = self._fmt.formatException(record.exc_info)
            if record.exc_text:
                record.exc_text = redact(record.exc_text)
        except Exception:  # noqa: BLE001 - never drop a log line over redaction
            pass
        return True
