"""Error translation: application errors become a uniform wire format."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.errors import AppError, NotFoundError, ValidationError


def test_app_error_carries_code_and_details():
    error = NotFoundError("Meeting not found.", details={"meeting_id": "m1"})

    body = error.to_response()

    assert error.status_code == 404
    assert body.error == "not_found"
    assert body.message == "Meeting not found."
    assert body.details == {"meeting_id": "m1"}


def test_validation_error_maps_to_422():
    assert ValidationError("bad input").status_code == 422


@pytest.fixture
def failing_client(app: FastAPI) -> TestClient:
    """An app with routes that raise each error class we translate."""

    @app.get("/boom/not-found")
    async def _not_found() -> None:
        raise NotFoundError("Meeting not found.", details={"meeting_id": "m1"})

    @app.get("/boom/app-error")
    async def _app_error() -> None:
        raise AppError("Something broke.")

    @app.get("/boom/unhandled")
    async def _unhandled() -> None:
        raise RuntimeError("unexpected failure")

    return TestClient(app, raise_server_exceptions=False)


def test_not_found_error_is_rendered_with_its_status_code(failing_client):
    response = failing_client.get("/boom/not-found")

    assert response.status_code == 404
    assert response.json() == {
        "error": "not_found",
        "message": "Meeting not found.",
        "details": {"meeting_id": "m1"},
    }


def test_base_app_error_defaults_to_500(failing_client):
    response = failing_client.get("/boom/app-error")

    assert response.status_code == 500
    assert response.json()["error"] == "internal_error"


def test_unhandled_exception_does_not_leak_internals(failing_client):
    response = failing_client.get("/boom/unhandled")

    assert response.status_code == 500
    assert response.json() == {
        "error": "internal_error",
        "message": "An unexpected error occurred.",
        "details": None,
    }
    assert "unexpected failure" not in response.text
