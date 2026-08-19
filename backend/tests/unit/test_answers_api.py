"""POST /api/answers.

Runs against a FakeLLMProvider through dependency overrides: no Anthropic
client is constructed and no request leaves the process.
"""

import pytest

from app.adapters.llm.base import LLMProviderError
from app.adapters.llm.fake import FakeLLMProvider
from app.api.middleware import REQUEST_ID_HEADER
from app.domain.models import GeneratedAnswer, GeneratedCitation
from app.services.generation import INSUFFICIENT_EVIDENCE_ANSWER

TRANSCRIPT = (
    "[00:00:12] Sarah: We need to delay the release because migration is unfinished.\n"
    "[00:00:31] John: Agreed. The migration script still fails.\n"
    "[00:00:52] Amir: What happens to the marketing budget?\n"
    "[00:01:14] Sarah: The budget is unchanged.\n"
    "[00:01:38] John: I will update the launch plan by Friday.\n"
)

GROUNDED = GeneratedAnswer(
    answer="The budget is unchanged; only the launch date moves.",
    citations=[
        GeneratedCitation(utterance_id="m1:u2"),
        GeneratedCitation(utterance_id="m1:u3"),
    ],
    insufficient_evidence=False,
)


@pytest.fixture
def ingested(client):
    response = client.post(
        "/api/transcripts",
        json={"meeting_id": "m1", "title": "Release planning", "transcript": TRANSCRIPT},
    )
    assert response.status_code == 201
    return response.json()


def ask(client, **overrides):
    payload = {
        "meeting_id": "m1",
        "question": "What was decided about the marketing budget?",
        "k": 5,
    }
    payload.update(overrides)
    return client.post("/api/answers", json=payload)


# --- happy path ------------------------------------------------------------


def test_returns_200(client, ingested):
    assert ask(client).status_code == 200


def test_response_has_the_documented_shape(client, ingested):
    body = ask(client).json()

    assert set(body) == {
        "meeting_id",
        "question",
        "answer",
        "citations",
        "insufficient_evidence",
    }


def test_response_echoes_the_request(client, ingested):
    body = ask(client).json()

    assert body["meeting_id"] == "m1"
    assert body["question"] == "What was decided about the marketing budget?"


def test_answer_text_comes_from_the_provider(client, ingested, llm_provider):
    llm_provider.response = GROUNDED

    assert ask(client).json()["answer"] == GROUNDED.answer


def test_citations_carry_utterance_ids(client, ingested, llm_provider):
    llm_provider.response = GROUNDED

    citations = ask(client).json()["citations"]

    assert [c["utterance_id"] for c in citations] == ["m1:u2", "m1:u3"]


def test_citations_carry_resolved_evidence_fields(client, ingested, llm_provider):
    llm_provider.response = GROUNDED

    citation = ask(client).json()["citations"][0]

    assert set(citation) == {
        "utterance_id",
        "speaker",
        "timestamp",
        "start_seconds",
        "quote",
    }


def test_citation_speaker_comes_from_the_transcript(client, ingested, llm_provider):
    llm_provider.response = GROUNDED

    citations = ask(client, k=10).json()["citations"]

    assert [c["speaker"] for c in citations] == ["Amir", "Sarah"]


def test_citation_timestamp_comes_from_the_transcript(client, ingested, llm_provider):
    llm_provider.response = GROUNDED

    citations = ask(client, k=10).json()["citations"]

    assert [c["timestamp"] for c in citations] == ["00:00:52", "00:01:14"]


def test_citation_start_seconds_comes_from_the_transcript(client, ingested, llm_provider):
    llm_provider.response = GROUNDED

    citations = ask(client, k=10).json()["citations"]

    assert [c["start_seconds"] for c in citations] == [52, 74]


def test_citation_quote_is_the_exact_source_text(client, ingested, llm_provider):
    llm_provider.response = GROUNDED

    citations = ask(client, k=10).json()["citations"]
    stored = {u["id"]: u["text"] for u in client.get("/api/transcripts/m1").json()["utterances"]}

    for citation in citations:
        assert citation["quote"] == stored[citation["utterance_id"]]


def test_invented_citation_does_not_appear_in_the_response(client, ingested, llm_provider):
    """Regression for Phase 7, which returned m1:u999 verbatim."""
    llm_provider.response = GeneratedAnswer(
        answer="The budget is unchanged.",
        citations=[
            GeneratedCitation(utterance_id="m1:u3"),
            GeneratedCitation(utterance_id="m1:u999"),
        ],
        insufficient_evidence=False,
    )

    citations = ask(client, k=10).json()["citations"]

    assert citations == [
        {
            "utterance_id": "m1:u3",
            "speaker": "Sarah",
            "timestamp": "00:01:14",
            "start_seconds": 74,
            "quote": "The budget is unchanged.",
        }
    ]


def test_a_wholly_invented_citation_list_yields_no_citations(client, ingested, llm_provider):
    llm_provider.response = GeneratedAnswer(
        answer="Confidently wrong.",
        citations=[GeneratedCitation(utterance_id="m1:u999")],
        insufficient_evidence=False,
    )

    body = ask(client).json()

    assert body["citations"] == []
    assert body["answer"] == "Confidently wrong."


def test_a_citation_from_another_meeting_is_discarded(client, ingested, llm_provider):
    llm_provider.response = GeneratedAnswer(
        answer="Cross-meeting leak attempt.",
        citations=[GeneratedCitation(utterance_id="other-meeting:u0")],
        insufficient_evidence=False,
    )

    assert ask(client).json()["citations"] == []


def test_duplicate_citations_appear_once(client, ingested, llm_provider):
    llm_provider.response = GeneratedAnswer(
        answer="The budget is unchanged.",
        citations=[
            GeneratedCitation(utterance_id="m1:u3"),
            GeneratedCitation(utterance_id="m1:u3"),
            GeneratedCitation(utterance_id="m1:u3"),
        ],
        insufficient_evidence=False,
    )

    citations = ask(client, k=10).json()["citations"]

    assert [c["utterance_id"] for c in citations] == ["m1:u3"]


def test_no_generated_quote_text_reaches_the_response(client, ingested, llm_provider):
    """Every quote must match a stored utterance exactly."""
    llm_provider.response = GROUNDED

    citations = ask(client, k=10).json()["citations"]
    stored = {u["text"] for u in client.get("/api/transcripts/m1").json()["utterances"]}

    assert citations
    assert all(c["quote"] in stored for c in citations)


def test_insufficient_evidence_is_exposed(client, ingested, llm_provider):
    llm_provider.response = GeneratedAnswer(
        answer="The meeting does not cover that.", citations=[], insufficient_evidence=True
    )

    body = ask(client).json()

    assert body["insufficient_evidence"] is True
    assert body["citations"] == []


def test_no_evidence_returns_the_stable_insufficient_answer(client, ingested, vector_store):
    import anyio

    anyio.run(lambda: vector_store.delete_meeting("m1"))

    body = ask(client).json()

    assert body["answer"] == INSUFFICIENT_EVIDENCE_ANSWER
    assert body["insufficient_evidence"] is True
    assert body["citations"] == []


def test_no_evidence_does_not_call_the_provider(client, ingested, vector_store, llm_provider):
    import anyio

    anyio.run(lambda: vector_store.delete_meeting("m1"))

    ask(client)

    assert llm_provider.call_count == 0


def test_question_is_trimmed_in_the_response(client, ingested):
    body = ask(client, question="   What about the budget?   ").json()

    assert body["question"] == "What about the budget?"


def test_response_exposes_no_chunks_or_vectors(client, ingested):
    body = ask(client).json()

    serialised = str(body)
    assert "chunk" not in serialised
    assert "score" not in serialised
    assert "m1:c" not in serialised


def test_the_provider_receives_evidence_context(client, ingested, llm_provider):
    ask(client)

    call = llm_provider.last_call
    assert call is not None
    assert "[m1:u" in call.context
    assert call.allowed_utterance_ids


# --- errors ----------------------------------------------------------------


def test_unknown_meeting_returns_the_uniform_404(client):
    response = ask(client, meeting_id="never-ingested")

    assert response.status_code == 404
    assert response.json() == {
        "error": "not_found",
        "message": "Transcript not found.",
        "details": {"meeting_id": "never-ingested"},
    }


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_blank_question_returns_the_uniform_422(client, ingested, question):
    response = ask(client, question=question)

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_returns_the_uniform_422(client, ingested, k):
    response = ask(client, k=k)

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_missing_fields_return_the_uniform_422(client):
    response = client.post("/api/answers", json={"meeting_id": "m1"})

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_provider_failure_returns_the_uniform_502(client, ingested, app, answer_service):
    from app.api.deps import get_answer_generation_service
    from app.services.generation import AnswerGenerationService

    failing = AnswerGenerationService(
        retrieval=answer_service._retrieval,
        repository=answer_service._repository,
        llm=FakeLLMProvider(error=LLMProviderError("provider unavailable")),
    )
    app.dependency_overrides[get_answer_generation_service] = lambda: failing

    response = ask(client)

    assert response.status_code == 502
    assert response.json()["error"] == "llm_provider_error"


def test_error_bodies_carry_the_uniform_keys(client):
    body = ask(client, meeting_id="missing").json()

    assert set(body) == {"error", "message", "details"}


# --- cross-cutting ---------------------------------------------------------


def test_request_id_header_is_present(client, ingested):
    assert ask(client).headers[REQUEST_ID_HEADER]


def test_supplied_request_id_is_echoed(client, ingested):
    response = client.post(
        "/api/answers",
        json={"meeting_id": "m1", "question": "budget", "k": 5},
        headers={REQUEST_ID_HEADER: "trace-answers"},
    )

    assert response.headers[REQUEST_ID_HEADER] == "trace-answers"


def test_api_tests_never_construct_the_real_provider(app):
    from app.api.deps import get_answer_generation_service

    assert get_answer_generation_service in app.dependency_overrides


def test_no_anthropic_request_is_made_during_api_tests(client, ingested, llm_provider):
    ask(client)

    assert isinstance(llm_provider, FakeLLMProvider)
    assert llm_provider.call_count == 1


def test_endpoint_is_registered_under_the_api_prefix(client):
    assert "/api/answers" in client.get("/openapi.json").json()["paths"]


def test_retrieval_endpoint_remains_available(client, ingested):
    """The debugging/evaluation endpoint is unaffected by answer generation."""
    response = client.post("/api/retrieval", json={"meeting_id": "m1", "query": "budget", "k": 3})

    assert response.status_code == 200
    assert response.json()["results"]
