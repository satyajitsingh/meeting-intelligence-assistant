"""Health endpoint and request-context behaviour."""

from app import __version__
from app.api.middleware import REQUEST_ID_HEADER


def test_health_returns_service_metadata(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "meeting-intelligence-api",
        "version": __version__,
        "environment": "test",
    }


def test_health_response_carries_a_request_id(client):
    response = client.get("/health")

    assert response.headers[REQUEST_ID_HEADER]


def test_supplied_request_id_is_echoed_back(client):
    response = client.get("/health", headers={REQUEST_ID_HEADER: "abc-123"})

    assert response.headers[REQUEST_ID_HEADER] == "abc-123"


def test_generated_request_ids_are_unique_per_request(client):
    first = client.get("/health").headers[REQUEST_ID_HEADER]
    second = client.get("/health").headers[REQUEST_ID_HEADER]

    assert first != second


def test_unknown_route_uses_the_uniform_error_shape(client):
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {
        "error": "not_found",
        "message": "Not Found",
        "details": None,
    }


def test_wrong_method_uses_the_uniform_error_shape(client):
    response = client.post("/health")

    assert response.status_code == 405
    assert response.json()["error"] == "method_not_allowed"
