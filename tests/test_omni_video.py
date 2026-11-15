"""AI video (Omni) in the backend: prompts, request shape, billing fence, cascade, stage."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.adapters import omni
from app.adapters.errors import ProviderError
from app.adapters.omni import OmniVideoAdapter, SceneSpec, SubmissionPending, build_prompt, clip_seconds
from app.core.config import Settings
from app.graph.build import pipeline_stage_names

DIALOGUE = SceneSpec(
    order=1, scene="a lighthouse doorway in a storm", motion="slow push-in", mode="dialogue",
    characters=("MAYA", "LEO"),
    lines=(("MAYA", "You shouldn't be out here in this storm.", "stern"), ("LEO", "I had nowhere else to go.", "")),
    voices={"MAYA": "FEMALE, 60s: warm alto", "LEO": "MALE, teen: bright"}, seconds=4.0,
)


def test_prompts_and_durations():
    p = build_prompt(DIALOGUE, 6)
    assert 'MAYA (FEMALE, 60s: warm alto, stern) says: "You shouldn\'t be out here in this storm."' in p
    assert "No music" in p and "Single continuous shot" in p and "Tamil" not in p
    ta = SceneSpec(**{**DIALOGUE.__dict__, "language": "ta"})
    assert "Tamil (தமிழ்)" in build_prompt(ta, 6)
    narr = build_prompt(SceneSpec(order=2, scene="the sea", characters=("MAYA",)), 4)
    assert "No dialogue, no speech" in narr and "In frame: MAYA" in narr
    assert clip_seconds(SceneSpec(order=1, scene="x", seconds=1.2)) == 3
    assert clip_seconds(SceneSpec(order=1, scene="x", seconds=30)) == 15
    assert clip_seconds(DIALOGUE) == 7  # 14 words / 2.6 wps + 1s, rounded up


class _Fake:
    def __init__(self, items):
        self.items, self.calls = list(items), []

    def create(self, **kw):
        self.calls.append(kw)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _video(i="int-1"):
    return SimpleNamespace(id=i, output_video=SimpleNamespace(uri=None, data=base64.b64encode(b"MP4").decode()))


def _adapter(tmp_path, items, **kw):
    s = Settings(gemini_api_key="k", use_mock_providers=False, omni_rpm_limit=100000, omni_speaker_check=False, **kw)
    a = OmniVideoAdapter(s, journal_dir=tmp_path / "ops")
    fake = _Fake(items)
    a._client = SimpleNamespace(interactions=fake, models=None, files=None)
    return a, fake


JPEG = b"\xff\xd8\xff" + b"0" * 16


def test_request_shape_cost_and_single_image(tmp_path):
    a, fake = _adapter(tmp_path, [_video()])
    clip = a.generate(JPEG, DIALOGUE, key="job-1")
    req = fake.calls[0]
    assert req["api_version"] == "v1beta" and req["store"] is True
    assert req["response_format"] == {"type": "video", "aspect_ratio": "9:16", "resolution": "1080p",
                                      "duration": "7s", "delivery": "uri"}
    assert req["generation_config"] == {"video_config": {"task": "image_to_video"}}
    assert [i["type"] for i in req["input"]] == ["image", "text"] and req["input"][0]["mime_type"] == "image/jpeg"
    assert clip.data == b"MP4" and clip.cost_usd == pytest.approx(7 * 0.104)
    assert not list((tmp_path / "ops").glob("*.json"))  # journal cleared on success


def test_timeout_fenced_no_double_billing(tmp_path, monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda *_a, **_k: None)
    a, fake = _adapter(tmp_path, [TimeoutError("read timed out")])
    with pytest.raises(SubmissionPending):
        a.generate(JPEG, DIALOGUE, key="job-2")
    b, fake2 = _adapter(tmp_path, [_video()])
    with pytest.raises(SubmissionPending):  # same scene key: refuses to resubmit
        b.generate(JPEG, DIALOGUE, key="job-2")
    assert fake2.calls == []


def test_daily_quota_cascades(tmp_path):
    a, fake = _adapter(tmp_path, [Exception("429 RESOURCE_EXHAUSTED: requests per day"), _video()])
    clip = a.generate(JPEG, DIALOGUE, key="job-3")
    assert [c["model"] for c in fake.calls] == ["gemini-omni-1.2-flash", "gemini-omni-1.1-flash"]
    assert clip.model == "gemini-omni-1.1-flash"


def test_all_models_exhausted_raises(tmp_path):
    daily = Exception("429 quota: requests per day")
    a, _ = _adapter(tmp_path, [daily, daily])
    with pytest.raises(ProviderError):
        a.generate(JPEG, DIALOGUE, key="job-4")


def test_wrong_voice_rerenders_once(tmp_path, monkeypatch):
    a, fake = _adapter(tmp_path, [_video("a"), _video("b")])
    verdicts = iter([True, False])
    monkeypatch.setattr(a, "_wrong_voice", lambda *_: next(verdicts))
    clip = a.generate(JPEG, DIALOGUE, key="job-5")
    assert len(fake.calls) == 2 and clip.renders == 2 and clip.cost_usd == pytest.approx(2 * 7 * 0.104)


def test_omni_is_the_default_video_stage():
    names = pipeline_stage_names(Settings(video_provider="omni"))
    assert names.index("narration") < names.index("motion") < names.index("captions")
    assert "motion" not in pipeline_stage_names(Settings(video_provider="none"))


async def test_dialogue_run_with_ai_video_is_on_camera(monkeypatch):
    """Full mock run, Omni stubbed: dialogue scenes become on-camera clips with audio."""
    from app.db import repository as repo
    from app.db.session import get_sessionmaker
    from app.workers.runner import run_job

    class _FakeOmni:
        def __init__(self, *_a, **_k):
            self.calls = 0

        def generate(self, image, spec, *, key):
            self.calls += 1
            return omni.OmniClip(data=b"MP4", model="gemini-omni-1.2-flash", seconds=5, cost_usd=0.52, renders=1)

    async def _fake_assemble(timeline, output_path):
        Path(output_path).write_bytes(b"\x00")
        return output_path

    async def _dur(_p):
        return 5.0

    monkeypatch.setenv("VIDEO_PROVIDER", "omni")
    from app.core.config import get_settings

    get_settings.cache_clear()
    try:
        monkeypatch.setattr("app.workers.runner.OmniVideoAdapter", _FakeOmni)
        monkeypatch.setattr("app.agents.video_gen.has_audio", lambda p: True)
        monkeypatch.setattr("app.agents.captions.probe_duration", _dur)
        monkeypatch.setattr("app.agents.assembler.assemble_video", _fake_assemble)

        async def _no_audio(*a, **k):
            return None

        monkeypatch.setattr("app.agents.assembler._lay_audio", _no_audio)
        async with get_sessionmaker()() as s:
            job = await repo.create_job(s, "a keeper and a runaway", 30, "dialogue", "en")
        await run_job(job.id, job.prompt, 30, False, "dialogue", "en")
        async with get_sessionmaker()() as s:
            job = await repo.get_job(s, job.id)
            stages = {st.name: st.status for st in await repo.get_stages(s, job.id)}
        assert job.status == "completed" and stages["motion"] == "succeeded"
        audio = json.loads(Path(next(a.path for a in job.artifacts if a.kind == "scene_audio")).read_text())
        assert audio and all(a["voice"] == "on_camera" for a in audio)
        assert sum(1 for a in job.artifacts if a.kind == "motion") == len(audio)
    finally:
        get_settings.cache_clear()


async def _video_stage(artifact_root, job, fail_orders, attempts_to_fail=1, **settings_kw):
    from app.adapters.errors import ProviderError as PE
    from app.agents.script_writer import _mock_script
    from app.agents.designer import _mock_shotlist
    from app.agents.video_gen import video_gen
    from app.artifacts.store import ArtifactStore
    from app.core.images import render_card
    from app.graph.state import Deps

    s = Settings(gemini_api_key="", use_mock_providers=True, ai_video_retry_delay_seconds=0, **settings_kw)
    store = ArtifactStore(artifact_root, job)
    script = _mock_script("x", 3, "dialogue")
    calls: dict[int, int] = {}

    class _V:
        def generate(self, image, spec, *, key):
            calls[spec.order] = calls.get(spec.order, 0) + 1
            if spec.order in fail_orders and calls[spec.order] <= attempts_to_fail:
                raise PE("omni daily quota spent")
            return omni.OmniClip(data=b"MP4", model="gemini-omni-1.2-flash", seconds=5, cost_usd=0.5, renders=1)

    deps = Deps(settings=s, store=store, llm=None, image=None, video=_V())
    frames = [{"order": i, "path": store.save_bytes(f"frame_{i:02d}.png", render_card(str(i))), "caption": "", "status": "ok"} for i in (1, 2, 3)]
    state = {"job_id": job, "frames": frames, "script_path": store.save_json("script.json", script),
             "shotlist_path": store.save_json("shotlist.json", _mock_shotlist(script)), "language": "en"}
    return video_gen, state, deps, calls


async def test_failed_scene_is_retried_once_then_succeeds(artifact_root, monkeypatch):
    monkeypatch.setattr("app.agents.video_gen.has_audio", lambda p: True)
    video_gen, state, deps, calls = await _video_stage(artifact_root, "vg-retry", fail_orders={2})
    out = await video_gen(state, deps)
    assert len(out["motion_clips"]) == 3 and calls[2] == 2


async def test_scene_without_ai_video_stops_the_job_not_stills(artifact_root, monkeypatch):
    from app.agents.video_gen import AIVideoUnavailable

    monkeypatch.setattr("app.agents.video_gen.has_audio", lambda p: True)
    video_gen, state, deps, calls = await _video_stage(artifact_root, "vg-stop", fail_orders={3}, attempts_to_fail=99)
    with pytest.raises(AIVideoUnavailable, match="scene\\(s\\) 3"):
        await video_gen(state, deps)
    # finished clips are kept for Resume (no re-billing)
    assert deps.store.exists("motion_01.mp4") and deps.store.exists("motion_manifest.json")
    out = None
    calls.clear()
    video_gen2, state2, deps2, calls2 = await _video_stage(artifact_root, "vg-stop", fail_orders=set())
    out = await video_gen2(state2, deps2)
    assert set(calls2) == {3}  # resume only regenerates the missing scene
    assert len(out["motion_clips"]) == 3
