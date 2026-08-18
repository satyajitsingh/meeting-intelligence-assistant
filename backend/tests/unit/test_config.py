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
