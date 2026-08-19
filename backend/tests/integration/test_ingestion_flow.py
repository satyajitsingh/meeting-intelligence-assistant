"""End-to-end transcript lifecycle through the real FastAPI app.

Offline by design -- fake embeddings, in-memory store and repository -- so this
carries no ``integration`` marker and runs in the default suite. That marker is
reserved for tests that need a model download or network access.
"""

from app.api.middleware import REQUEST_ID_HEADER

TRANSCRIPT = (
    "[00:00:12] Sarah: We need to delay the release by two weeks.\n"
    "[00:00:31] John: Agreed. The migration script still fails on legacy accounts.\n"
    "[00:00:52] Amir: What does the delay mean for the marketing budget?\n"
    "[00:01:14] Sarah: The budget is unchanged, only the launch date moves.\n"
    "[00:01:38] John: I can have the migration fixed by Wednesday.\n"
    "[00:02:03] Amir: Then let's announce the new date publicly.\n"
    "[00:02:29] Sarah: Agreed. John, please update the launch plan by Friday.\n"
)


def test_full_transcript_lifecycle(client, vector_store):
    # 1. POST -- ingest
    created = client.post(
        "/api/transcripts",
        json={
            "meeting_id": "release-planning",
            "title": "Release planning",
            "transcript": TRANSCRIPT,
        },
    )
    assert created.status_code == 201
    summary = created.json()
    assert summary["meeting_id"] == "release-planning"
    assert summary["speakers"] == ["Sarah", "John", "Amir"]
    assert summary["utterance_count"] == 7
    assert summary["duration_seconds"] == 149
    assert summary["chunk_count"] >= 1
    assert created.headers[REQUEST_ID_HEADER]

    # 2. GET one -- utterances are available for citation resolution
    detail = client.get("/api/transcripts/release-planning")
    assert detail.status_code == 200
    utterances = detail.json()["utterances"]
    assert len(utterances) == 7
    assert [u["id"] for u in utterances] == [f"release-planning:u{i}" for i in range(7)]
    assert utterances[0]["display_timestamp"] == "00:00:12"
    assert utterances[-1]["speaker"] == "Sarah"

    # 3. GET list
    listing = client.get("/api/transcripts")
    assert listing.status_code == 200
    assert [row["meeting_id"] for row in listing.json()] == ["release-planning"]

    # chunks really are searchable at this point
    import anyio

    query = [0.0] * vector_store.dimension
    query[0] = 1.0
    stored = anyio.run(lambda: vector_store.search(query, meeting_id="release-planning", k=100))
    assert len(stored) == summary["chunk_count"]
    assert all(scored.chunk.meeting_id == "release-planning" for scored in stored)
    # every chunk still maps back to its source utterances
    assert all(scored.chunk.utterance_ids for scored in stored)

    # 4. DELETE
    assert client.delete("/api/transcripts/release-planning").status_code == 204

    # 5. GET -- now absent, with the uniform 404 body
    missing = client.get("/api/transcripts/release-planning")
    assert missing.status_code == 404
    assert missing.json() == {
        "error": "not_found",
        "message": "Transcript not found.",
        "details": {"meeting_id": "release-planning"},
    }

    assert client.get("/api/transcripts").json() == []
    emptied = anyio.run(lambda: vector_store.search(query, meeting_id="release-planning", k=100))
    assert emptied == []


def test_two_meetings_stay_isolated_through_the_api(client):
    for meeting_id, title in [("alpha", "Alpha sync"), ("bravo", "Bravo sync")]:
        response = client.post(
            "/api/transcripts",
            json={"meeting_id": meeting_id, "title": title, "transcript": TRANSCRIPT},
        )
        assert response.status_code == 201

    assert [row["meeting_id"] for row in client.get("/api/transcripts").json()] == [
        "alpha",
        "bravo",
    ]

    client.delete("/api/transcripts/alpha")

    assert client.get("/api/transcripts/alpha").status_code == 404
    assert client.get("/api/transcripts/bravo").status_code == 200
