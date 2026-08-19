"""Transcript endpoints.

Runs entirely against fake adapters via dependency overrides, so no embedding
model is constructed and no network call is made.
"""

from app.api.middleware import REQUEST_ID_HEADER

MALFORMED = "[00:00:12] Sarah: Hello.\n[00:00:25] John Missing the colon.\n"


def post_transcript(client, sample_transcript, meeting_id="m1", title="Release planning"):
    return client.post(
        "/api/transcripts",
        json={"meeting_id": meeting_id, "title": title, "transcript": sample_transcript},
    )


# --- POST ------------------------------------------------------------------


def test_ingest_returns_201_with_a_summary(client, sample_transcript):
    response = post_transcript(client, sample_transcript)

    assert response.status_code == 201
    body = response.json()
    assert body["meeting_id"] == "m1"
    assert body["title"] == "Release planning"
    assert body["speakers"] == ["Sarah", "John", "Amir"]
    assert body["utterance_count"] == 4
    assert body["duration_seconds"] == 74
    assert body["chunk_count"] >= 1


def test_ingest_response_never_exposes_embeddings(client, sample_transcript):
    body = post_transcript(client, sample_transcript).json()

    assert set(body) == {
        "meeting_id",
        "title",
        "speakers",
        "utterance_count",
        "chunk_count",
        "duration_seconds",
    }


def test_reingesting_replaces_the_previous_transcript(client, sample_transcript):
    post_transcript(client, sample_transcript)

    response = client.post(
        "/api/transcripts",
        json={"meeting_id": "m1", "title": "Revised", "transcript": "[00:00:05] Amir: New.\n"},
    )

    assert response.status_code == 201
    detail = client.get("/api/transcripts/m1").json()
    assert detail["title"] == "Revised"
    assert len(detail["utterances"]) == 1


def test_malformed_transcript_returns_the_uniform_422(client):
    response = client.post(
        "/api/transcripts",
        json={"meeting_id": "m1", "title": "T", "transcript": MALFORMED},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "transcript_parse_error"
    assert "Line 2" in body["message"]
    assert body["details"]["line_number"] == 2


def test_empty_transcript_body_returns_the_uniform_422(client):
    response = client.post(
        "/api/transcripts", json={"meeting_id": "m1", "title": "T", "transcript": "   \n"}
    )

    assert response.status_code == 422
    assert response.json()["error"] == "empty_transcript"


def test_blank_meeting_id_is_rejected(client, sample_transcript):
    response = post_transcript(client, sample_transcript, meeting_id="   ")

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_missing_fields_are_rejected(client):
    response = client.post("/api/transcripts", json={"meeting_id": "m1"})

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_meeting_id_is_trimmed(client, sample_transcript):
    post_transcript(client, sample_transcript, meeting_id="  spaced  ")

    assert client.get("/api/transcripts/spaced").status_code == 200


def test_meeting_id_is_never_generated(client, sample_transcript):
    body = post_transcript(client, sample_transcript, meeting_id="chosen-by-caller").json()

    assert body["meeting_id"] == "chosen-by-caller"


# --- GET one ---------------------------------------------------------------


def test_get_returns_the_stored_utterances(client, sample_transcript):
    post_transcript(client, sample_transcript)

    body = client.get("/api/transcripts/m1").json()

    assert body["meeting_id"] == "m1"
    assert body["speakers"] == ["Sarah", "John", "Amir"]
    assert body["duration_seconds"] == 74
    assert len(body["utterances"]) == 4

    first = body["utterances"][0]
    assert first["id"] == "m1:u0"
    assert first["index"] == 0
    assert first["speaker"] == "Sarah"
    assert first["start_seconds"] == 12
    assert first["raw_timestamp"] == "00:00:12"
    assert first["display_timestamp"] == "00:00:12"
    assert first["text"].startswith("We need to delay")


def test_get_preserves_utterance_order(client, sample_transcript):
    post_transcript(client, sample_transcript)

    utterances = client.get("/api/transcripts/m1").json()["utterances"]

    assert [u["index"] for u in utterances] == [0, 1, 2, 3]


def test_get_does_not_expose_chunks_or_vectors(client, sample_transcript):
    post_transcript(client, sample_transcript)

    body = client.get("/api/transcripts/m1").json()

    assert set(body) == {
        "meeting_id",
        "title",
        "speakers",
        "duration_seconds",
        "utterances",
    }


def test_get_unknown_transcript_returns_the_uniform_404(client):
    response = client.get("/api/transcripts/missing")

    assert response.status_code == 404
    assert response.json() == {
        "error": "not_found",
        "message": "Transcript not found.",
        "details": {"meeting_id": "missing"},
    }


# --- GET list --------------------------------------------------------------


def test_list_is_empty_before_any_upload(client):
    response = client.get("/api/transcripts")

    assert response.status_code == 200
    assert response.json() == []


def test_list_returns_uploaded_meetings(client, sample_transcript):
    post_transcript(client, sample_transcript, meeting_id="m1", title="One")
    post_transcript(client, sample_transcript, meeting_id="m2", title="Two")

    body = client.get("/api/transcripts").json()

    assert [row["meeting_id"] for row in body] == ["m1", "m2"]
    assert body[0]["title"] == "One"
    assert body[0]["utterance_count"] == 4
    assert body[0]["duration_seconds"] == 74


def test_list_rows_omit_utterances(client, sample_transcript):
    post_transcript(client, sample_transcript)

    row = client.get("/api/transcripts").json()[0]

    assert set(row) == {
        "meeting_id",
        "title",
        "speakers",
        "utterance_count",
        "duration_seconds",
    }


def test_list_order_is_insertion_order(client, sample_transcript):
    for meeting_id in ["charlie", "alpha", "bravo"]:
        post_transcript(client, sample_transcript, meeting_id=meeting_id)

    body = client.get("/api/transcripts").json()

    assert [row["meeting_id"] for row in body] == ["charlie", "alpha", "bravo"]


# --- DELETE ----------------------------------------------------------------


def test_delete_returns_204(client, sample_transcript):
    post_transcript(client, sample_transcript)

    assert client.delete("/api/transcripts/m1").status_code == 204


def test_deleted_transcript_no_longer_gets(client, sample_transcript):
    post_transcript(client, sample_transcript)
    client.delete("/api/transcripts/m1")

    assert client.get("/api/transcripts/m1").status_code == 404


def test_deleted_transcript_leaves_the_listing(client, sample_transcript):
    post_transcript(client, sample_transcript, meeting_id="m1")
    post_transcript(client, sample_transcript, meeting_id="m2")

    client.delete("/api/transcripts/m1")

    assert [row["meeting_id"] for row in client.get("/api/transcripts").json()] == ["m2"]


def test_delete_clears_vector_search_results(client, sample_transcript, vector_store):
    post_transcript(client, sample_transcript)
    query = [0.0] * vector_store.dimension
    query[0] = 1.0

    import anyio

    before = len(anyio.run(lambda: vector_store.search(query, meeting_id="m1", k=100)))
    client.delete("/api/transcripts/m1")
    after = len(anyio.run(lambda: vector_store.search(query, meeting_id="m1", k=100)))

    assert before > 0
    assert after == 0


def test_delete_unknown_meeting_returns_the_uniform_404(client):
    response = client.delete("/api/transcripts/missing")

    assert response.status_code == 404
    assert response.json()["error"] == "not_found"
    assert response.json()["details"] == {"meeting_id": "missing"}


def test_deleting_twice_returns_404_the_second_time(client, sample_transcript):
    post_transcript(client, sample_transcript)

    assert client.delete("/api/transcripts/m1").status_code == 204
    assert client.delete("/api/transcripts/m1").status_code == 404


# --- cross-cutting ---------------------------------------------------------


def test_request_id_header_is_present_on_transcript_routes(client, sample_transcript):
    response = post_transcript(client, sample_transcript)

    assert response.headers[REQUEST_ID_HEADER]


def test_supplied_request_id_is_echoed(client, sample_transcript):
    response = client.post(
        "/api/transcripts",
        json={"meeting_id": "m1", "title": "T", "transcript": sample_transcript},
        headers={REQUEST_ID_HEADER: "trace-123"},
    )

    assert response.headers[REQUEST_ID_HEADER] == "trace-123"


def test_api_tests_never_construct_the_real_embedding_provider(app):
    """The overrides must fully replace the container-backed dependencies."""
    from app.api.deps import get_ingestion_service, get_transcript_repository, get_vector_store

    assert get_ingestion_service in app.dependency_overrides
    assert get_transcript_repository in app.dependency_overrides
    assert get_vector_store in app.dependency_overrides


def test_endpoints_are_registered_under_the_api_prefix(client):
    paths = client.get("/openapi.json").json()["paths"]

    assert "/api/transcripts" in paths
    assert "/api/transcripts/{meeting_id}" in paths
