"""Price table + spend events + estimates."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from app.observability.cost import (
    estimate_job_cost,
    gemini_usage_cost,
    price_rule,
    token_cost,
    unit_cost,
    video_cost,
)
from app.observability.usage import UsageCollector, bind_collector, record_gemini_call, reset_collector


def test_gemini4_price_doubles_on_dec_1():
    nov = token_cost("gemini-4-carbon", input_tokens=1_000_000, output_tokens=1_000_000, on=date(2026, 11, 30))
    dec = token_cost("gemini-4-carbon", input_tokens=1_000_000, output_tokens=1_000_000, on=date(2026, 12, 1))
    assert (nov, dec) == (12.0, 24.0)


def test_flash_and_tts_double_on_jan_1():
    assert price_rule("gemini-3.8-flash", date(2026, 12, 31)).tokens.input == 0.75
    assert price_rule("gemini-3.8-flash", date(2027, 1, 1)).tokens.input == 1.50
    assert price_rule("gemini-3.8-flash-tts", date(2027, 1, 1)).tokens.output_audio == 18.00


def test_image_tokens_priced_at_image_rate():
    usage = SimpleNamespace(
        prompt_token_count=14, candidates_token_count=1987, thoughts_token_count=1039,
        candidates_tokens_details=[SimpleNamespace(modality=SimpleNamespace(name="IMAGE"), token_count=1680)],
    )
    cost, units = gemini_usage_cost("gemini-nano-banana-2.1", usage, on=date(2026, 11, 12))
    # 1680 image tokens @ $30/M + (307 text + 1039 thinking) @ $7.50/M + 14 in @ $1.50/M
    assert abs(cost - (1680 * 30 + 1346 * 7.5 + 14 * 1.5) / 1e6) < 1e-9
    assert units["image_tokens"] == 1680


def test_video_music_and_unknown_models():
    assert abs(video_cost("gemini-omni-1.2-flash", 10, "1080p") - 1.04) < 1e-9
    assert abs(video_cost("gemini-omni-1.1-flash", 10, "720p") - 0.70) < 1e-9
    assert unit_cost("lyria-3.5") == 0.08
    assert token_cost("totally-new-model", input_tokens=10**6) == 0.0
    assert price_rule("anthropic:claude-opus-5-5").tokens.input == 4.00


def test_spend_recorded_under_current_stage():
    from app.observability.context import bind_stage, reset_stage

    col = UsageCollector()
    tok = bind_collector(col)
    st = bind_stage("job1", "image_gen")
    try:
        usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=0, total_token_count=10)
        record_gemini_call("text", "gemini-4-argon", usage)
    finally:
        reset_stage(st)
        reset_collector(tok)
    assert len(col.events) == 1 and col.events[0].stage == "image_gen"
    assert col.cost_by_stage["image_gen"] == col.cost_usd > 0
    assert col.drain_events() and col.events == []


def test_estimate_uses_history_when_available():
    assert estimate_job_cost(duration=60, style="dialogue") > estimate_job_cost(duration=30, style="dialogue")
    assert estimate_job_cost(duration=60, style="dialogue", history_avg_per_second=0.2) == 12.0
    assert estimate_job_cost(duration=60, style="dialogue", omni_enabled=False) < estimate_job_cost(duration=60, style="dialogue") / 2
