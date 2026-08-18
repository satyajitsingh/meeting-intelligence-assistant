"""Application configuration.

All settings are read from environment variables (or a local ``.env`` file) so
that no credential or environment-specific value is ever committed to source.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "test", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Typed, validated application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "meeting-intelligence-api"
    environment: Environment = "local"

    log_level: LogLevel = "INFO"
    log_json: bool = True

    api_prefix: str = "/api"
    cors_origins: list[str] = ["http://localhost:3000"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that configuration is parsed and validated exactly once. Tests
    that need to change the environment must call ``get_settings.cache_clear()``.
    """
    return Settings()
