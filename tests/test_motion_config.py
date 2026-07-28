from app.core.config import Settings


def test_motion_v2_defaults_are_bounded_and_feature_remains_off():
    settings = Settings(_env_file=None)
    assert settings.motion_engine_enabled is False
    assert settings.motion_clip_seconds == 4.0
    assert settings.motion_max_concurrency == 30
    assert settings.motion_stage_budget_seconds == 1800.0
