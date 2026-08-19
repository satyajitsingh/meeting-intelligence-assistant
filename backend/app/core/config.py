"""Application configuration.

All settings are read from environment variables (or a local ``.env`` file) so
that no credential or environment-specific value is ever committed to source.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
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

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_target_chars: int = Field(default=700, gt=0)

    # Answer generation. The key is never logged or returned in an error.
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-5"
    llm_timeout_seconds: float = Field(default=30.0, gt=0)

    # Speech to text. Only whisper-1 reports language and duration; the
    # gpt-4o-transcribe family returns text alone.
    openai_api_key: SecretStr | None = None
    openai_transcription_model: str = "whisper-1"
    max_audio_upload_mb: int = Field(default=25, gt=0)

    @property
    def max_audio_upload_bytes(self) -> int:
        return self.max_audio_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that configuration is parsed and validated exactly once. Tests
    that need to change the environment must call ``get_settings.cache_clear()``.
    """
    return Settings()
