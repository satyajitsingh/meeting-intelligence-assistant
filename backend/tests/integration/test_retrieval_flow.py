"""Ingestion through retrieval, across two meetings, via the real FastAPI app.

Offline by design -- fake embeddings, in-memory store and repository -- so this
carries no ``integration`` marker and runs in the default suite.
"""

RELEASE_PLANNING = (
    "[00:00:12] Sarah: We need to delay the release because the migration is unfinished.\n"
    "[00:00:31] John: Agreed. The migration script still fails on legacy accounts.\n"
    "[00:00:52] Amir: What happens to the marketing budget we already approved?\n"
    "[00:01:14] Sarah: The budget is unchanged, only the launch date moves.\n"
    "[00:01:38] John: I will update the launch plan by Friday and notify support.\n"
)

HIRING_REVIEW = (
    "[00:00:05] Priya: The hiring plan needs three more backend engineers this quarter.\n"
    "[00:00:24] Tom: Recruiting says the candidate pipeline is thin for senior roles.\n"
    "[00:00:47] Priya: Then we should raise the referral bonus before the next cycle.\n"
)


def ingest(client, meeting_id: str, title: str, transcript: str) -> None:
    response = client.post(
        "/api/transcripts",
        json={"meeting_id": meeting_id, "title": title, "transcript": transcript},
    )
    assert response.status_code == 201


def test_retrieval_is_scoped_to_the_requested_meeting(client):
    # 1. ingest the first meeting
    ingest(client, "release-planning", "Release planning", RELEASE_PLANNING)

    # 2. query it
    response = client.post(
        "/api/retrieval",
        json={
            "meeting_id": "release-planning",
            "query": "What was decided about the marketing budget?",
            "k": 5,
        },
    )
    assert response.status_code == 200
    body = response.json()

    # 3. at least one result
    results = body["results"]
    assert len(results) >= 1
    assert body["meeting_id"] == "release-planning"

    # 4. every chunk belongs to that meeting, and maps back to its utterances
    assert all(r["chunk_id"].startswith("release-planning:c") for r in results)
    assert all(uid.startswith("release-planning:u") for r in results for uid in r["utterance_ids"])
    assert all(r["utterance_ids"] for r in results)

    # 5. ingest a second, unrelated meeting
    ingest(client, "hiring-review", "Hiring review", HIRING_REVIEW)

    # 6. query the first meeting again
    again = client.post(
        "/api/retrieval",
        json={
            "meeting_id": "release-planning",
            "query": "hiring pipeline for senior engineers",
            "k": 10,
        },
    )
    assert again.status_code == 200
    second_results = again.json()["results"]

    # 7. nothing from meeting two leaks in, even for a question about it
    assert second_results
    assert all(r["chunk_id"].startswith("release-planning:") for r in second_results)
    assert not any("hiring-review" in r["chunk_id"] for r in second_results)
    assert not any("hiring-review" in uid for r in second_results for uid in r["utterance_ids"])


def test_each_meeting_retrieves_only_its_own_chunks(client):
    ingest(client, "release-planning", "Release planning", RELEASE_PLANNING)
    ingest(client, "hiring-review", "Hiring review", HIRING_REVIEW)

    def chunk_ids(meeting_id: str) -> set[str]:
        response = client.post(
            "/api/retrieval",
            json={"meeting_id": meeting_id, "query": "what was decided", "k": 20},
        )
        assert response.status_code == 200
        return {r["chunk_id"] for r in response.json()["results"]}

    first = chunk_ids("release-planning")
    second = chunk_ids("hiring-review")

    assert first and second
    assert first.isdisjoint(second)


def test_deleting_a_meeting_removes_it_from_retrieval(client):
    ingest(client, "release-planning", "Release planning", RELEASE_PLANNING)
    ingest(client, "hiring-review", "Hiring review", HIRING_REVIEW)

    assert client.delete("/api/transcripts/release-planning").status_code == 204

    gone = client.post(
        "/api/retrieval", json={"meeting_id": "release-planning", "query": "budget", "k": 5}
    )
    assert gone.status_code == 404

    survivor = client.post(
        "/api/retrieval", json={"meeting_id": "hiring-review", "query": "hiring", "k": 5}
    )
    assert survivor.status_code == 200
    assert survivor.json()["results"]


def test_reingesting_replaces_what_retrieval_returns(client):
    ingest(client, "m1", "Original", RELEASE_PLANNING)
    ingest(client, "m1", "Replacement", "[00:00:05] Priya: Completely different content.\n")

    body = client.post(
        "/api/retrieval", json={"meeting_id": "m1", "query": "marketing budget", "k": 20}
    ).json()

    assert len(body["results"]) == 1
    assert body["results"][0]["speakers"] == ["Priya"]
    assert body["results"][0]["utterance_ids"] == ["m1:u0"]
