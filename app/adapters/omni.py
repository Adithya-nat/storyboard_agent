"""Gemini Omni image-to-video (Interactions API): one clip per scene, incl. on-camera dialogue.

Daily-quota 429s cascade down ``omni_model_chain``; per-minute 429s/5xx retry. A journal
fences each paid submission so a crash/timeout can't bill it twice. Dialogue clips get a
speaker check and are re-rendered once if a line is voiced by the wrong gender. Content-policy
blocks raise ``ContentBlocked`` so the video stage can rewrite the scene (see video_gen).
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from app.adapters.errors import PermanentProviderError, ProviderError, TransientProviderError
from app.adapters.genai_client import b64bytes, b64str, interactions_create, is_daily_quota, make_client
from app.adapters.llm import _is_transient, parse_retry_after, wait_retry_after
from app.adapters.rate_limit import get_named_limiter
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.observability import usage
from app.observability.cost import video_cost
from app.observability.spend_rate import get_governor, is_spend_rate_limited
from app.prompts import render

log = get_logger(__name__)

WORDS_PER_SECOND = {"en": 2.6, "ta": 1.8}
_BOOT = f"{os.getpid()}-{time.time_ns()}"  # tags fences with the process that wrote them


class SubmissionPending(ProviderError):
    """A paid submission's outcome is unknown; resubmitting could double-bill."""


class _DailyExhausted(TransientProviderError):
    pass


class ContentBlocked(PermanentProviderError):
    """Omni refused the input under a content policy. Blocked requests aren't billed."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code  # content_blocked (third-party/IP) | prohibited_content | safety


_BLOCK_CODE = re.compile(r"'code':\s*'([a-z_]+)'")
_BLOCK_HINTS = ("content_blocked", "prohibited_content", "guardrails", "blocked due to", "safety", "policy")


def block_code(exc: BaseException) -> str | None:
    """The content-policy code of a 400 rejection, or None when it isn't a content block."""
    msg = str(exc)
    low = msg.lower()
    if "400" not in msg or not any(h in low for h in _BLOCK_HINTS):
        return None
    m = _BLOCK_CODE.search(msg)
    if m and m.group(1) not in ("invalid_argument", "bad_request"):
        return m.group(1)
    return "content_blocked" if "third-party" in low else "prohibited_content"


@dataclass
class SceneSpec:
    order: int
    scene: str                      # what the camera sees (English)
    motion: str = ""                # camera/subject motion hint
    mode: str = "narration"         # narration | dialogue
    language: str = "en"
    characters: tuple[str, ...] = ()
    lines: tuple[tuple[str, str, str], ...] = ()  # (speaker, text, delivery)
    voices: dict | None = None      # speaker -> voice description
    seconds: float = 4.0


@dataclass
class OmniClip:
    data: bytes
    model: str
    seconds: int
    cost_usd: float
    renders: int


def _gender(voice: str) -> str | None:
    head = voice.strip().upper()
    return "female" if head.startswith("FEMALE") else "male" if head.startswith("MALE") else None


def clip_seconds(spec: SceneSpec) -> int:
    want = spec.seconds
    if spec.mode == "dialogue" and spec.lines:
        words = sum(len(t.split()) for _, t, _ in spec.lines)
        want = max(want, words / WORDS_PER_SECOND.get(spec.language, 2.6) + 1.0)
    return int(min(15, max(3, math.ceil(want))))


def timed_lines(spec: SceneSpec, seconds: int) -> str:
    voices = spec.voices or {}
    weights = [max(1, len(t.split())) for _, t, _ in spec.lines]
    usable, t, out = max(1.0, seconds - 0.5), 0.0, []
    for (speaker, text, delivery), w in zip(spec.lines, weights):
        end = min(seconds, t + usable * w / sum(weights))
        note = f", {delivery}" if delivery else ""
        out.append(f'[{math.floor(t)}-{math.ceil(end)}s] {speaker} ({voices.get(speaker, "")}{note}) says: "{text}"')
        t = end
    return "\n".join(out)


def build_prompt(spec: SceneSpec, seconds: int) -> str:
    voices = spec.voices or {}
    cast = ("In frame: " + "; ".join(f"{c} ({voices[c]})" if voices.get(c) else c for c in spec.characters) + ".") if spec.characters else ""
    motion = spec.motion or "subtle, natural cinematic motion; the camera moves gently"
    if spec.mode == "dialogue" and spec.lines:
        note = "All dialogue is spoken in Tamil (தமிழ்) with natural, fluent Tamil pronunciation." if spec.language == "ta" else ""
        return render("omni/dialogue", scene=spec.scene, motion=motion, cast=cast,
                      timed_lines=timed_lines(spec, seconds), language_note=note)
    return render("omni/narration", scene=spec.scene, motion=motion, cast=cast)


class OmniVideoAdapter:
    def __init__(self, settings: Settings | None = None, journal_dir: str | Path | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None
        self.journal_dir = Path(journal_dir or (Path(self.settings.artifact_root) / "_omni_ops"))

    @property
    def client(self):
        if self._client is None:
            # Our own retries + spend governor handle 429s visibly; the SDK's would sleep silently.
            self._client = make_client(self.settings, timeout_seconds=self.settings.omni_timeout_seconds, attempts=1)
        return self._client

    def generate(self, image: bytes, spec: SceneSpec, *, key: str) -> OmniClip:
        """Image + scene spec -> mp4 bytes (with audio). Raises ProviderError when no model can."""
        seconds = clip_seconds(spec)
        prompt = build_prompt(spec, seconds)
        cost, renders, last, first = 0.0, 0, None, None
        for model in self.settings.omni_model_chain:
            try:
                while True:
                    try:
                        data = self._with_retries(model, image, prompt, seconds, key)
                    except ProviderError:
                        if first is None:
                            raise
                        log.warning("scene %s: voice re-render failed; keeping the first clip", spec.order)
                        return OmniClip(data=first, model=model, seconds=seconds, cost_usd=round(cost, 4), renders=renders)
                    renders += 1
                    c = video_cost(model, seconds, self.settings.omni_resolution)
                    cost += c
                    usage.record_spend(category="video", model=model, cost_usd=c,
                                       units={"seconds": seconds, "renders": 1})
                    if renders == 1 and self._wrong_voice(data, spec):
                        log.warning("scene %s: wrong voice on a line; re-rendering once", spec.order)
                        first = data
                        continue
                    return OmniClip(data=data, model=model, seconds=seconds, cost_usd=round(cost, 4), renders=renders)
            except _DailyExhausted as exc:
                log.warning("omni model %s daily quota spent; trying the next", model)
                last = exc
        raise ProviderError(f"no Omni model available today: {last}")

    # ------------------------------------------------------------------ #
    def _with_retries(self, model: str, image: bytes, prompt: str, seconds: int, key: str) -> bytes:
        return Retrying(
            retry=retry_if_exception(
                lambda e: isinstance(e, TransientProviderError) and not isinstance(e, _DailyExhausted)
            ),
            stop=stop_after_attempt(3),
            wait=wait_retry_after(wait_exponential(min=5, max=60), cap=60.0),
            reraise=True,
        )(self._submit, model, image, prompt, seconds, key)

    def _journal(self, key: str) -> Path:
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        return self.journal_dir / f"{re.sub(r'[^A-Za-z0-9_-]', '', key)[:120]}.json"

    def _submit(self, model: str, image: bytes, prompt: str, seconds: int, key: str) -> bytes:
        jpath = self._journal(key)
        pending = json.loads(jpath.read_text()) if jpath.exists() else None
        if pending and pending.get("interaction_id"):
            it = self.client.interactions.get(pending["interaction_id"], api_version="v1beta")
            data = self._download(it)
            jpath.unlink(missing_ok=True)
            return data
        if pending and pending.get("model") == model:
            age = time.time() - jpath.stat().st_mtime
            alive = pending.get("boot") == _BOOT and age < self.settings.omni_timeout_seconds + 300
            if alive:
                raise SubmissionPending("previous Omni submission outcome unknown; not resubmitting (avoids double billing)")
            # Written by a process that has since stopped (or older than any live request): release it.
            log.warning("releasing a stale Omni fence for %s left by an earlier run", key)

        get_named_limiter(f"omni:{model}", self.settings.omni_rpm_limit, capacity=1.0).acquire()
        estimate = video_cost(model, seconds, self.settings.omni_resolution)
        governor = get_governor(self.settings.spend_rate_limit_usd_per_10min)
        waited = governor.acquire(estimate)
        if waited > 1:
            log.info("paced %.0fs to stay under the 10-minute spend-rate limit", waited)
        jpath.write_text(json.dumps({"model": model, "state": "submission_uncertain", "boot": _BOOT}))
        try:
            it = interactions_create(
                self.client,
                model=model,
                input=[
                    {"type": "image", "data": b64str(image), "mime_type": _mime(image)},
                    {"type": "text", "text": prompt},
                ],
                generation_config={"video_config": {"task": "image_to_video"}},
                response_format={
                    "type": "video",
                    "aspect_ratio": "9:16",
                    "resolution": self.settings.omni_resolution,
                    "duration": f"{seconds}s",
                    "delivery": "uri",  # >4MB clips need URI delivery, which requires store=true
                },
                store=True,
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if isinstance(exc, TimeoutError) or any(m in msg for m in ("timed out", "timeout", "deadline exceeded")):
                raise SubmissionPending("Omni submission timed out; fenced against double billing") from exc
            jpath.unlink(missing_ok=True)
            governor.settle(estimate, 0.0)  # nothing was billed
            code = block_code(exc)
            if code:
                raise ContentBlocked(str(exc), code) from exc
            if is_spend_rate_limited(exc):
                log.warning("Google's spend-rate limit reached; pausing AI video until the 10-minute window clears")
                governor.saturate()
            if is_daily_quota(exc):
                raise _DailyExhausted(str(exc)) from exc
            if _is_transient(exc):
                raise TransientProviderError(str(exc), retry_after=parse_retry_after(exc)) from exc
            raise PermanentProviderError(str(exc)) from exc
        jpath.write_text(json.dumps({"model": model, "interaction_id": it.id}))
        data = self._download(it)
        jpath.unlink(missing_ok=True)
        return data

    def _download(self, interaction) -> bytes:
        vid = getattr(interaction, "output_video", None)
        if vid is None:
            raise PermanentProviderError("Omni returned no video")
        uri = getattr(vid, "uri", None)
        if not uri:
            data = b64bytes(getattr(vid, "data", None))
            if not data:
                raise PermanentProviderError("Omni returned an empty video")
            return data
        name = uri.split("/")[-1].split("?")[0].split(":")[0]
        for _ in range(200):
            state = getattr(getattr(self.client.files.get(name=f"files/{name}"), "state", None), "name", "")
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise PermanentProviderError("Omni video processing failed")
            time.sleep(3)
        fd, path = tempfile.mkstemp(prefix="omni-", suffix=".mp4")
        os.close(fd)
        try:
            self.client.files.download(file=uri, destination=path)
            return Path(path).read_bytes()
        finally:
            Path(path).unlink(missing_ok=True)

    def _wrong_voice(self, data: bytes, spec: SceneSpec) -> bool:
        """True when a dialogue line is clearly voiced by the wrong gender (best-effort)."""
        if not (self.settings.omni_speaker_check and spec.mode == "dialogue" and spec.lines):
            return False
        expected = [_gender((spec.voices or {}).get(s, "")) for s, _, _ in spec.lines]
        if not any(expected):
            return False
        try:
            from google.genai import types

            wav = _audio_wav(data)
            lines = "\n".join(f"{i}. {s}: {t}" for i, (s, t, _) in enumerate(spec.lines))
            resp = self.client.models.generate_content(
                model=self.settings.omni_speaker_check_model,
                contents=[types.Part.from_bytes(data=wav, mime_type="audio/wav"), render("omni/speaker_check", lines=lines)],
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=_SpeakerCheck),
            )
            usage.record_gemini_call("transcribe", self.settings.omni_speaker_check_model, getattr(resp, "usage_metadata", None))
            heard = {v.index: v.gender.lower() for v in _SpeakerCheck.model_validate_json(resp.text).lines}
        except Exception as exc:  # noqa: BLE001 - never blocks a clip
            log.warning("speaker check skipped (%s)", exc)
            return False
        return any(e and heard.get(i) in ("male", "female") and heard[i] != e for i, e in enumerate(expected))


class _LineVoice(BaseModel):
    index: int
    gender: str


class _SpeakerCheck(BaseModel):
    lines: list[_LineVoice]


def _mime(data: bytes) -> str:
    return "image/jpeg" if data[:3] == b"\xff\xd8\xff" else "image/png"


def _audio_wav(mp4: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(mp4)
        f.flush()
        return subprocess.run(
            ["ffmpeg", "-v", "error", "-i", f.name, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", "-"],
            capture_output=True, check=True,
        ).stdout


def has_audio(path: str) -> bool:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True,
        ).stdout
        return bool(out.strip())
    except Exception:  # noqa: BLE001
        return False
