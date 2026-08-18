"""Shared pytest fixtures."""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import create_app


@pytest.fixture
def settings() -> Settings:
    """Deterministic settings for tests: console logs, no .env file influence."""
    return Settings(environment="test", log_json=False, log_level="WARNING")


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    application = create_app(settings)
    # The health route resolves settings through the cached provider, so point
    # that dependency at the same test settings instance.
    application.dependency_overrides[get_settings] = lambda: settings
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
