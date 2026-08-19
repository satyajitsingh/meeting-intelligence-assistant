"""POST /api/retrieval.

Runs against fake adapters via dependency overrides -- no embedding model is
constructed and no network call is made.
"""

import pytest

from app.api.middleware import REQUEST_ID_HEADER

TRANSCRIPT = (
    "[00:00:12] Sarah: We need to delay the release because the migration is unfinished.\n"
    "[00:00:31] John: Agreed, the migration script still fails on legacy accounts.\n"
    "[00:00:52] Amir: What happens to the marketing budget we already approved?\n"
    "[00:01:14] Sarah: The budget is unchanged, only the launch date moves.\n"
    "[00:01:38] John: I will update the launch plan by Friday and notify support.\n"
)


@pytest.fixture
def ingested(client):
    response = client.post(
        "/api/transcripts",
        json={"meeting_id": "m1", "title": "Release planning", "transcript": TRANSCRIPT},
    )
    assert response.status_code == 201
    return response.json()


def retrieve(client, **overrides):
    payload = {"meeting_id": "m1", "query": "marketing budget", "k": 5}
    payload.update(overrides)
    return client.post("/api/retrieval", json=payload)


# --- happy path ------------------------------------------------------------


def test_retrieval_returns_200(client, ingested):
    assert retrieve(client).status_code == 200


def test_response_echoes_the_request(client, ingested):
    body = retrieve(client).json()

    assert body["meeting_id"] == "m1"
    assert body["query"] == "marketing budget"


def test_response_has_the_documented_shape(client, ingested):
    body = retrieve(client).json()

    assert set(body) == {"meeting_id", "query", "results"}
    assert set(body["results"][0]) == {
        "chunk_id",
        "score",
        "text",
        "speakers",
        "start_seconds",
        "end_seconds",
        "utterance_ids",
    }


def test_results_include_chunk_ids(client, ingested):
    results = retrieve(client).json()["results"]

    assert results
    assert all(r["chunk_id"].startswith("m1:c") for r in results)


def test_results_include_utterance_ids(client, ingested):
    results = retrieve(client).json()["results"]

    for result in results:
        assert result["utterance_ids"]
        assert all(uid.startswith("m1:u") for uid in result["utterance_ids"])


def test_results_include_speakers(client, ingested):
    results = retrieve(client).json()["results"]

    for result in results:
        assert result["speakers"]
        assert set(result["speakers"]) <= {"Sarah", "John", "Amir"}


def test_results_include_timestamps(client, ingested):
    results = retrieve(client).json()["results"]

    for result in results:
        assert isinstance(result["start_seconds"], int)
        assert isinstance(result["end_seconds"], int)
        assert result["end_seconds"] >= result["start_seconds"]


def test_results_include_chunk_text_with_speaker_labels(client, ingested):
    top = retrieve(client).json()["results"][0]

    assert top["text"]
    assert all(":" in line for line in top["text"].splitlines())


def test_scores_are_preserved_and_ordered(client, ingested):
    results = retrieve(client, k=10).json()["results"]

    scores = [r["score"] for r in results]
    assert all(isinstance(s, float) for s in scores)
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)


def test_k_limits_the_number_of_results(client, ingested):
    assert len(retrieve(client, k=1).json()["results"]) == 1


def test_k_defaults_when_omitted(client, ingested):
    response = client.post("/api/retrieval", json={"meeting_id": "m1", "query": "marketing budget"})

    assert response.status_code == 200
    assert len(response.json()["results"]) <= 5


def test_response_never_exposes_embeddings(client, ingested):
    body = retrieve(client, k=10).json()

    serialised = str(body)
    assert "vector" not in serialised
    assert "embedding" not in serialised


def test_query_is_trimmed_in_the_response(client, ingested):
    body = retrieve(client, query="   marketing budget   ").json()

    assert body["query"] == "marketing budget"


# --- empty results ---------------------------------------------------------


def test_known_meeting_with_no_chunks_returns_200_and_empty_results(client, ingested, vector_store):
    import anyio

    anyio.run(lambda: vector_store.delete_meeting("m1"))

    response = retrieve(client)

    assert response.status_code == 200
    assert response.json()["results"] == []


# --- errors ----------------------------------------------------------------


def test_unknown_meeting_returns_the_uniform_404(client):
    response = retrieve(client, meeting_id="never-ingested")

    assert response.status_code == 404
    assert response.json() == {
        "error": "not_found",
        "message": "Transcript not found.",
        "details": {"meeting_id": "never-ingested"},
    }


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_returns_the_uniform_422(client, ingested, query):
    response = retrieve(client, query=query)

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_returns_the_uniform_422(client, ingested, k):
    response = retrieve(client, k=k)

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_blank_meeting_id_returns_the_uniform_422(client, ingested):
    response = retrieve(client, meeting_id="   ")

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_missing_fields_return_the_uniform_422(client):
    response = client.post("/api/retrieval", json={"meeting_id": "m1"})

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_non_integer_k_returns_the_uniform_422(client, ingested):
    response = retrieve(client, k="five")

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_error_bodies_carry_the_uniform_keys(client):
    body = retrieve(client, meeting_id="missing").json()

    assert set(body) == {"error", "message", "details"}


# --- cross-cutting ---------------------------------------------------------


def test_request_id_header_is_present(client, ingested):
    assert retrieve(client).headers[REQUEST_ID_HEADER]


def test_supplied_request_id_is_echoed(client, ingested):
    response = client.post(
        "/api/retrieval",
        json={"meeting_id": "m1", "query": "budget", "k": 5},
        headers={REQUEST_ID_HEADER: "trace-retrieval"},
    )

    assert response.headers[REQUEST_ID_HEADER] == "trace-retrieval"


def test_api_tests_never_construct_the_real_embedding_provider(app):
    from app.api.deps import get_retrieval_service

    assert get_retrieval_service in app.dependency_overrides


def test_endpoint_is_registered_under_the_api_prefix(client):
    assert "/api/retrieval" in client.get("/openapi.json").json()["paths"]


def test_retrieval_endpoint_calls_no_language_model(client, ingested):
    """Phase 6 is retrieval-only: the response carries no generated text."""
    body = retrieve(client).json()

    assert set(body) == {"meeting_id", "query", "results"}
    assert "answer" not in body
    assert "confidence" not in body


# --- container wiring ------------------------------------------------------


def test_container_shares_one_store_between_ingestion_and_retrieval(settings):
    """A second store here would silently retrieve nothing after ingestion.

    Constructing the container instantiates FastEmbedProvider, which resolves
    its dimension from static metadata and does not download the model.
    """
    from app.core.container import build_container

    container = build_container(settings)

    assert container.retrieval_service._vector_store is container.vector_store
    assert container.retrieval_service._vector_store is container.ingestion_service._vector_store
    assert container.retrieval_service._repository is container.repository
    assert container.retrieval_service._repository is container.ingestion_service._repository
    assert container.retrieval_service._embeddings is container.embeddings
    assert container.retrieval_service._embeddings is container.ingestion_service._embeddings


def test_container_does_not_load_the_embedding_model(settings):
    from app.core.container import build_container

    container = build_container(settings)

    assert container.embeddings._model is None
    assert container.vector_store.dimension == container.embeddings.dimension
