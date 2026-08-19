"""AnswerGenerationService orchestration."""

import pytest

from app.adapters.embeddings.fake import FakeEmbeddingProvider
from app.adapters.llm.base import LLMProviderError
from app.adapters.llm.fake import FakeLLMProvider
from app.adapters.repository.memory import InMemoryTranscriptRepository
from app.adapters.vectorstore.memory import InMemoryVectorStore
from app.core.errors import NotFoundError
from app.domain.models import GeneratedAnswer, GeneratedCitation
from app.services.generation import INSUFFICIENT_EVIDENCE_ANSWER, AnswerGenerationService
from app.services.ingestion import IngestionService
from app.services.retrieval import DEFAULT_K, InvalidRetrievalRequestError, RetrievalService

pytestmark = pytest.mark.anyio

DIMENSION = 256

SAMPLE = (
    "[00:00:12] Sarah: We need to delay the release because migration is unfinished.\n"
    "[00:00:31] John: Agreed. The migration script still fails.\n"
    "[00:00:52] Amir: What happens to the marketing budget?\n"
    "[00:01:14] Sarah: The budget is unchanged.\n"
    "[00:01:38] John: I will update the launch plan by Friday.\n"
)


class RecordingRetrievalService(RetrievalService):
    """Records retrieve() arguments and how often it ran."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calls: list[tuple[str, str, int]] = []

    async def retrieve(self, *, meeting_id: str, query: str, k: int = DEFAULT_K):
        self.calls.append((meeting_id, query, k))
        return await super().retrieve(meeting_id=meeting_id, query=query, k=k)


async def build(
    llm: FakeLLMProvider | None = None,
    transcripts: dict[str, str] | None = None,
    target_chars: int = 150,
):
    embeddings = FakeEmbeddingProvider(dimension=DIMENSION)
    vector_store = InMemoryVectorStore(dimension=DIMENSION)
    repository = InMemoryTranscriptRepository()

    ingestion = IngestionService(
        embeddings=embeddings,
        vector_store=vector_store,
        repository=repository,
        target_chars=target_chars,
    )
    for meeting_id, text in (transcripts or {"m1": SAMPLE}).items():
        await ingestion.ingest(meeting_id=meeting_id, title=meeting_id, transcript_text=text)

    retrieval = RecordingRetrievalService(
        embeddings=embeddings, vector_store=vector_store, repository=repository
    )
    llm = llm if llm is not None else FakeLLMProvider()
    service = AnswerGenerationService(retrieval=retrieval, repository=repository, llm=llm)
    return service, retrieval, llm, vector_store, repository


# --- retrieval delegation --------------------------------------------------


async def test_retrieval_is_called_exactly_once():
    service, retrieval, _, _, _ = await build()

    await service.answer(meeting_id="m1", question="What about the budget?")

    assert len(retrieval.calls) == 1


async def test_retrieval_receives_meeting_question_and_k():
    service, retrieval, _, _, _ = await build()

    await service.answer(meeting_id="m1", question="What about the budget?", k=3)

    assert retrieval.calls == [("m1", "What about the budget?", 3)]


async def test_k_defaults_to_the_retrieval_default():
    service, retrieval, _, _, _ = await build()

    await service.answer(meeting_id="m1", question="budget")

    assert retrieval.calls[0][2] == DEFAULT_K


async def test_there_is_no_second_retrieval_attempt():
    """No agent loop, no retry, no widening on a weak result."""
    service, retrieval, _, _, _ = await build()

    await service.answer(meeting_id="m1", question="something entirely unrelated")

    assert len(retrieval.calls) == 1


# --- LLM delegation --------------------------------------------------------


async def test_the_llm_is_called_once_when_evidence_exists():
    service, _, llm, _, _ = await build()

    await service.answer(meeting_id="m1", question="What about the budget?")

    assert llm.call_count == 1


async def test_the_llm_receives_the_trimmed_question():
    service, _, llm, _, _ = await build()

    await service.answer(meeting_id="m1", question="   What about the budget?   ")

    assert llm.last_call is not None
    assert llm.last_call.question == "What about the budget?"


async def test_allowed_ids_exactly_match_the_context_utterances():
    service, _, llm, _, _ = await build()

    await service.answer(meeting_id="m1", question="budget", k=10)

    call = llm.last_call
    assert call is not None
    for utterance_id in call.allowed_utterance_ids:
        assert f"[{utterance_id} |" in call.context
    assert call.context.count("[m1:u") == len(call.allowed_utterance_ids)


async def test_allowed_ids_are_unique_and_in_transcript_order():
    service, _, llm, _, _ = await build()

    await service.answer(meeting_id="m1", question="budget migration launch", k=10)

    ids = llm.last_call.allowed_utterance_ids
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids, key=lambda i: int(i.rsplit("u", 1)[1]))


async def test_context_contains_only_retrieved_evidence():
    service, _, llm, _, _ = await build()

    await service.answer(meeting_id="m1", question="budget", k=1)

    call = llm.last_call
    assert call is not None
    assert call.allowed_utterance_ids
    assert len(call.allowed_utterance_ids) < 5


async def test_context_is_utterance_level_not_chunk_level():
    service, _, llm, _, _ = await build()

    await service.answer(meeting_id="m1", question="budget", k=10)

    assert "m1:c" not in llm.last_call.context


async def test_full_transcript_is_not_sent_to_the_llm():
    """The model sees retrieved evidence, never the whole meeting."""
    service, _, llm, _, _ = await build(target_chars=120)

    await service.answer(meeting_id="m1", question="marketing budget", k=1)

    assert "I will update the launch plan by Friday." not in llm.last_call.context


# --- pass-through ----------------------------------------------------------


async def test_the_answer_text_is_returned_unchanged():
    """Validation touches citations, never the model's prose."""
    configured = GeneratedAnswer(
        answer="The budget is unchanged.",
        citations=[GeneratedCitation(utterance_id="m1:u3")],
        insufficient_evidence=False,
    )
    service, _, _, _, _ = await build(llm=FakeLLMProvider(configured))

    result = await service.answer(meeting_id="m1", question="budget")

    assert result.answer == "The budget is unchanged."
    assert result.insufficient_evidence is False


async def test_a_valid_citation_becomes_a_resolved_citation():
    configured = GeneratedAnswer(
        answer="The budget is unchanged.",
        citations=[GeneratedCitation(utterance_id="m1:u3")],
        insufficient_evidence=False,
    )
    service, _, _, _, _ = await build(llm=FakeLLMProvider(configured))

    citation = (await service.answer(meeting_id="m1", question="budget", k=10)).citations[0]

    assert citation.utterance_id == "m1:u3"
    assert citation.speaker == "Sarah"
    assert citation.timestamp == "00:01:14"
    assert citation.start_seconds == 74
    assert citation.quote == "The budget is unchanged."


async def test_hallucinated_citations_are_discarded():
    """The Phase 7 pass-through is now a rejection: m1:u999 must not survive."""
    configured = GeneratedAnswer(
        answer="Invented.",
        citations=[GeneratedCitation(utterance_id="m1:u999")],
        insufficient_evidence=False,
    )
    service, _, _, _, _ = await build(llm=FakeLLMProvider(configured))

    result = await service.answer(meeting_id="m1", question="budget")

    assert result.citations == []
    assert result.answer == "Invented."


async def test_mixed_valid_and_invented_citations_keep_only_the_valid_one():
    configured = GeneratedAnswer(
        answer="The budget is unchanged.",
        citations=[
            GeneratedCitation(utterance_id="m1:u3"),
            GeneratedCitation(utterance_id="m1:u999"),
        ],
        insufficient_evidence=False,
    )
    service, _, _, _, _ = await build(llm=FakeLLMProvider(configured))

    result = await service.answer(meeting_id="m1", question="budget", k=10)

    assert [c.utterance_id for c in result.citations] == ["m1:u3"]


async def test_citations_carry_evidence_resolved_from_the_transcript():
    service, _, _, _, repository = await build(
        llm=FakeLLMProvider(
            GeneratedAnswer(
                answer="Yes.",
                citations=[GeneratedCitation(utterance_id="m1:u3")],
                insufficient_evidence=False,
            )
        )
    )

    citation = (await service.answer(meeting_id="m1", question="budget", k=10)).citations[0]

    assert set(citation.model_dump()) == {
        "utterance_id",
        "speaker",
        "timestamp",
        "start_seconds",
        "quote",
    }

    transcript = await repository.get("m1")
    source = next(u for u in transcript.utterances if u.id == "m1:u3")
    assert citation.speaker == source.speaker
    assert citation.quote == source.text
    assert citation.timestamp == source.display_timestamp
    assert citation.start_seconds == source.start_seconds


async def test_validation_uses_exactly_the_ids_sent_to_the_llm():
    """A real utterance that was never shown to the model is still rejected."""
    service, _, llm, _, _ = await build(target_chars=120)

    # First call with k=1 to learn which evidence the model is shown.
    await service.answer(meeting_id="m1", question="marketing budget", k=1)
    shown = set(llm.last_call.allowed_utterance_ids)
    withheld = next(f"m1:u{i}" for i in range(5) if f"m1:u{i}" not in shown)

    llm.response = GeneratedAnswer(
        answer="Citing evidence I was never given.",
        citations=[GeneratedCitation(utterance_id=withheld)],
        insufficient_evidence=False,
    )
    result = await service.answer(meeting_id="m1", question="marketing budget", k=1)

    assert result.citations == []


async def test_duplicate_citations_appear_once():
    configured = GeneratedAnswer(
        answer="The budget is unchanged.",
        citations=[
            GeneratedCitation(utterance_id="m1:u3"),
            GeneratedCitation(utterance_id="m1:u3"),
        ],
        insufficient_evidence=False,
    )
    service, _, _, _, _ = await build(llm=FakeLLMProvider(configured))

    result = await service.answer(meeting_id="m1", question="budget", k=10)

    assert [c.utterance_id for c in result.citations] == ["m1:u3"]


# --- no evidence -----------------------------------------------------------


async def test_no_evidence_returns_the_insufficient_answer():
    service, _, _, vector_store, _ = await build()
    await vector_store.delete_meeting("m1")

    result = await service.answer(meeting_id="m1", question="budget")

    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.citations == []
    assert result.insufficient_evidence is True


async def test_no_evidence_does_not_call_the_llm():
    service, _, llm, vector_store, _ = await build()
    await vector_store.delete_meeting("m1")

    await service.answer(meeting_id="m1", question="budget")

    assert llm.call_count == 0


async def test_the_insufficient_wording_is_stable():
    assert INSUFFICIENT_EVIDENCE_ANSWER == (
        "I don't have enough evidence in this meeting to answer that."
    )


# --- error propagation -----------------------------------------------------


async def test_unknown_meeting_propagates_not_found():
    service, _, _, _, _ = await build()

    with pytest.raises(NotFoundError) as exc_info:
        await service.answer(meeting_id="never-ingested", question="budget")

    assert exc_info.value.details == {"meeting_id": "never-ingested"}


async def test_unknown_meeting_does_not_call_the_llm():
    service, _, llm, _, _ = await build()

    with pytest.raises(NotFoundError):
        await service.answer(meeting_id="missing", question="budget")

    assert llm.call_count == 0


@pytest.mark.parametrize("question", ["", "   "])
async def test_blank_question_is_rejected(question):
    service, _, _, _, _ = await build()

    with pytest.raises(InvalidRetrievalRequestError):
        await service.answer(meeting_id="m1", question=question)


@pytest.mark.parametrize("k", [0, -1])
async def test_invalid_k_is_rejected(k):
    service, _, _, _, _ = await build()

    with pytest.raises(InvalidRetrievalRequestError):
        await service.answer(meeting_id="m1", question="budget", k=k)


async def test_provider_failure_propagates():
    service, _, _, _, _ = await build(
        llm=FakeLLMProvider(error=LLMProviderError("provider unavailable"))
    )

    with pytest.raises(LLMProviderError) as exc_info:
        await service.answer(meeting_id="m1", question="budget")

    assert exc_info.value.status_code == 502


async def test_provider_failure_is_not_swallowed_into_an_insufficient_answer():
    service, _, _, _, _ = await build(llm=FakeLLMProvider(error=LLMProviderError("boom")))

    with pytest.raises(LLMProviderError):
        await service.answer(meeting_id="m1", question="budget")


# --- isolation and determinism ---------------------------------------------


async def test_evidence_never_crosses_meetings():
    other = "[00:00:05] Priya: We need three more backend engineers.\n"
    service, _, llm, _, _ = await build(transcripts={"m1": SAMPLE, "m2": other})

    await service.answer(meeting_id="m1", question="backend engineers", k=10)

    assert "m2:u" not in llm.last_call.context
    assert all(i.startswith("m1:") for i in llm.last_call.allowed_utterance_ids)


async def test_repeated_calls_build_identical_context():
    service, _, llm, _, _ = await build()

    await service.answer(meeting_id="m1", question="budget", k=10)
    await service.answer(meeting_id="m1", question="budget", k=10)

    assert llm.calls[0].context == llm.calls[1].context
    assert llm.calls[0].allowed_utterance_ids == llm.calls[1].allowed_utterance_ids


async def test_generation_requires_no_network():
    service, _, llm, _, _ = await build()

    assert isinstance(llm, FakeLLMProvider)
    assert await service.answer(meeting_id="m1", question="budget")
