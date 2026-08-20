"""Settings parsing and validation."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.core.config import Settings, get_settings


def test_defaults_are_sensible():
    settings = Settings(_env_file=None)

    assert settings.app_name == "meeting-intelligence-api"
    assert settings.environment == "local"
    assert settings.log_level == "INFO"
    assert settings.log_json is True
    assert settings.api_prefix == "/api"
    assert settings.cors_origins == ["http://localhost:3000"]


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    monkeypatch.setenv("LOG_JSON", "false")
    monkeypatch.setenv("CORS_ORIGINS", '["https://example.com"]')

    settings = Settings(_env_file=None)

    assert settings.environment == "production"
    assert settings.log_level == "ERROR"
    assert settings.log_json is False
    assert settings.cors_origins == ["https://example.com"]


def test_invalid_environment_is_rejected(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "staging")

    with pytest.raises(PydanticValidationError):
        Settings(_env_file=None)


def test_invalid_log_level_is_rejected(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "VERBOSE")

    with pytest.raises(PydanticValidationError):
        Settings(_env_file=None)


def test_unknown_environment_variables_are_ignored(monkeypatch):
    monkeypatch.setenv("SOME_UNRELATED_VARIABLE", "value")

    assert Settings(_env_file=None).app_name == "meeting-intelligence-api"


def test_get_settings_is_cached():
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()


# --- blank provider keys ---------------------------------------------------


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_a_blank_anthropic_key_is_treated_as_unset(monkeypatch, blank):
    """`.env.example` ships `ANTHROPIC_API_KEY=`; that must not look configured."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", blank)

    assert Settings(_env_file=None).anthropic_api_key is None


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_openai_key_is_treated_as_unset(monkeypatch, blank):
    monkeypatch.setenv("OPENAI_API_KEY", blank)

    assert Settings(_env_file=None).openai_api_key is None


def test_a_real_key_is_still_read(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-example")

    settings = Settings(_env_file=None)

    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-ant-example"


def test_keys_default_to_none_when_absent(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = Settings(_env_file=None)

    assert settings.anthropic_api_key is None
    assert settings.openai_api_key is None


def test_the_shipped_env_example_yields_no_configured_keys(tmp_path):
    """Copying .env.example verbatim must leave both providers unconfigured."""
    from pathlib import Path

    example = Path(__file__).resolve().parents[2] / ".env.example"
    env_file = tmp_path / ".env"
    env_file.write_text(example.read_text())

    settings = Settings(_env_file=str(env_file))

    assert settings.anthropic_api_key is None
    assert settings.openai_api_key is None
