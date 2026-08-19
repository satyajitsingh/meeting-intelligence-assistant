"""RetrievalService: validation, scoping and pass-through ranking."""

import pytest

from app.adapters.embeddings.fake import FakeEmbeddingProvider
from app.adapters.repository.memory import InMemoryTranscriptRepository
from app.adapters.vectorstore.memory import InMemoryVectorStore
from app.core.errors import NotFoundError
from app.domain.models import ScoredChunk
from app.services.ingestion import IngestionService
from app.services.retrieval import (
    DEFAULT_K,
    InvalidRetrievalRequestError,
    RetrievalConfigurationError,
    RetrievalService,
)

pytestmark = pytest.mark.anyio

DIMENSION = 256

SAMPLE = (
    "[00:00:12] Sarah: We need to delay the release because the migration is unfinished.\n"
    "[00:00:31] John: Agreed, the migration script still fails on legacy accounts.\n"
    "[00:00:52] Amir: What happens to the marketing budget we already approved?\n"
    "[00:01:14] Sarah: The budget is unchanged, only the launch date moves.\n"
    "[00:01:38] John: I will update the launch plan by Friday and notify support.\n"
)

OTHER = (
    "[00:00:05] Priya: The hiring plan needs three more backend engineers.\n"
    "[00:00:20] Tom: Recruiting says the pipeline is thin for senior roles.\n"
)


class RecordingEmbeddingProvider(FakeEmbeddingProvider):
    """Records every query it is asked to embed."""

    def __init__(self, dimension: int = DIMENSION) -> None:
        super().__init__(dimension=dimension)
        self.query_calls: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return await super().embed_query(text)


class RecordingVectorStore(InMemoryVectorStore):
    """Records the arguments every search was called with."""

    def __init__(self, dimension: int = DIMENSION) -> None:
        super().__init__(dimension=dimension)
        self.search_calls: list[tuple[list[float], str, int]] = []

    async def search(self, vector: list[float], *, meeting_id: str, k: int):
        self.search_calls.append((list(vector), meeting_id, k))
        return await super().search(vector, meeting_id=meeting_id, k=k)


class StubVectorStore(InMemoryVectorStore):
    """Returns a fixed result list, so pass-through can be asserted exactly."""

    def __init__(self, results: list[ScoredChunk], dimension: int = DIMENSION) -> None:
        super().__init__(dimension=dimension)
        self._results = results

    async def search(self, vector: list[float], *, meeting_id: str, k: int):
        return list(self._results)


async def build(
    embeddings: FakeEmbeddingProvider | None = None,
    vector_store: InMemoryVectorStore | None = None,
    transcripts: dict[str, str] | None = None,
    target_chars: int = 150,
):
    """Return a service wired to freshly ingested transcripts."""
    embeddings = embeddings or FakeEmbeddingProvider(dimension=DIMENSION)
    vector_store = vector_store or InMemoryVectorStore(dimension=DIMENSION)
    repository = InMemoryTranscriptRepository()

    ingestion = IngestionService(
        embeddings=embeddings,
        vector_store=vector_store,
        repository=repository,
        target_chars=target_chars,
    )
    for meeting_id, text in (transcripts or {"m1": SAMPLE}).items():
        await ingestion.ingest(meeting_id=meeting_id, title=meeting_id, transcript_text=text)

    service = RetrievalService(
        embeddings=embeddings, vector_store=vector_store, repository=repository
    )
    return service, embeddings, vector_store, repository


# --- construction ----------------------------------------------------------


def test_rejects_mismatched_dimensions_at_construction():
    with pytest.raises(RetrievalConfigurationError) as exc_info:
        RetrievalService(
            embeddings=FakeEmbeddingProvider(dimension=64),
            vector_store=InMemoryVectorStore(dimension=128),
            repository=InMemoryTranscriptRepository(),
        )

    assert exc_info.value.details == {
        "embedding_dimension": 64,
        "vector_store_dimension": 128,
    }


# --- happy path ------------------------------------------------------------


async def test_retrieves_chunks_for_a_known_meeting():
    service, _, _, _ = await build()

    results = await service.retrieve(meeting_id="m1", query="marketing budget")

    assert results
    assert all(isinstance(r, ScoredChunk) for r in results)
    assert all(r.chunk.meeting_id == "m1" for r in results)


async def test_results_carry_full_chunk_metadata():
    service, _, _, _ = await build()

    top = (await service.retrieve(meeting_id="m1", query="marketing budget"))[0]

    assert top.chunk.id.startswith("m1:c")
    assert top.chunk.utterance_ids
    assert all(uid.startswith("m1:u") for uid in top.chunk.utterance_ids)
    assert top.chunk.speakers
    assert top.chunk.text


async def test_default_k_is_applied():
    store = RecordingVectorStore()
    service, _, _, _ = await build(vector_store=store)
    store.search_calls.clear()

    await service.retrieve(meeting_id="m1", query="budget")

    assert store.search_calls[0][2] == DEFAULT_K


# --- delegation ------------------------------------------------------------


async def test_query_is_embedded_exactly_once():
    embeddings = RecordingEmbeddingProvider()
    service, _, _, _ = await build(embeddings=embeddings)
    embeddings.query_calls.clear()

    await service.retrieve(meeting_id="m1", query="marketing budget")

    assert embeddings.query_calls == ["marketing budget"]


async def test_query_is_embedded_with_embed_query_not_embed_documents():
    embeddings = RecordingEmbeddingProvider()
    service, _, _, _ = await build(embeddings=embeddings)
    embeddings.query_calls.clear()

    await service.retrieve(meeting_id="m1", query="budget")

    assert len(embeddings.query_calls) == 1


async def test_search_receives_the_requested_meeting_id():
    store = RecordingVectorStore()
    service, _, _, _ = await build(vector_store=store, transcripts={"m1": SAMPLE, "m2": OTHER})
    store.search_calls.clear()

    await service.retrieve(meeting_id="m2", query="hiring")

    assert store.search_calls[0][1] == "m2"


async def test_search_receives_the_requested_k():
    store = RecordingVectorStore()
    service, _, _, _ = await build(vector_store=store)
    store.search_calls.clear()

    await service.retrieve(meeting_id="m1", query="budget", k=3)

    assert store.search_calls[0][2] == 3


async def test_search_receives_exactly_the_embedded_query_vector():
    embeddings = RecordingEmbeddingProvider()
    store = RecordingVectorStore()
    service, _, _, _ = await build(embeddings=embeddings, vector_store=store)
    store.search_calls.clear()

    await service.retrieve(meeting_id="m1", query="marketing budget")

    expected = await FakeEmbeddingProvider(dimension=DIMENSION).embed_query("marketing budget")
    assert store.search_calls[0][0] == expected


async def test_query_is_trimmed_before_embedding():
    embeddings = RecordingEmbeddingProvider()
    service, _, _, _ = await build(embeddings=embeddings)
    embeddings.query_calls.clear()

    await service.retrieve(meeting_id="m1", query="   budget   ")

    assert embeddings.query_calls == ["budget"]


# --- pass-through ----------------------------------------------------------


async def test_ranking_is_returned_unchanged():
    service, _, _, _ = await build()

    results = await service.retrieve(meeting_id="m1", query="marketing budget", k=10)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


async def test_results_are_neither_reranked_nor_filtered():
    """Whatever the store returns is what the caller gets, in that order."""
    stub_service, _, _, _ = await build()
    stored = await stub_service.retrieve(meeting_id="m1", query="budget", k=10)

    reversed_results = list(reversed(stored))
    service = RetrievalService(
        embeddings=FakeEmbeddingProvider(dimension=DIMENSION),
        vector_store=StubVectorStore(reversed_results),
        repository=(await build())[3],
    )

    assert await service.retrieve(meeting_id="m1", query="budget", k=10) == reversed_results


async def test_scores_are_returned_unchanged():
    service, embeddings, store, _ = await build()

    results = await service.retrieve(meeting_id="m1", query="marketing budget", k=10)

    vector = await embeddings.embed_query("marketing budget")
    expected = await store.search(vector, meeting_id="m1", k=10)
    assert [r.score for r in results] == [r.score for r in expected]


async def test_chunk_text_is_not_modified():
    service, _, _, repository = await build()

    results = await service.retrieve(meeting_id="m1", query="budget", k=10)

    transcript = await repository.get("m1")
    assert transcript is not None
    for result in results:
        for line in result.chunk.text.splitlines():
            speaker, _, text = line.partition(": ")
            assert any(u.speaker == speaker and u.text == text for u in transcript.utterances)


async def test_overlapping_chunks_are_not_deduplicated():
    """Adjacent chunks share an utterance; both may legitimately be returned.

    At this target the chunker yields [u0,u1], [u1,u2,u3], [u3,u4], so u1 and
    u3 each appear twice. Retrieval must not collapse them.
    """
    service, _, _, _ = await build(target_chars=200)

    results = await service.retrieve(meeting_id="m1", query="migration budget", k=10)

    assert len(results) == 3
    seen = [uid for r in results for uid in r.chunk.utterance_ids]
    assert len(seen) > len(set(seen))


# --- validation ------------------------------------------------------------


async def test_unknown_meeting_raises_not_found():
    service, _, _, _ = await build()

    with pytest.raises(NotFoundError) as exc_info:
        await service.retrieve(meeting_id="never-ingested", query="budget")

    assert exc_info.value.status_code == 404
    assert exc_info.value.details == {"meeting_id": "never-ingested"}


async def test_unknown_meeting_is_rejected_before_embedding():
    embeddings = RecordingEmbeddingProvider()
    service, _, _, _ = await build(embeddings=embeddings)
    embeddings.query_calls.clear()

    with pytest.raises(NotFoundError):
        await service.retrieve(meeting_id="missing", query="budget")

    assert embeddings.query_calls == []


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
async def test_blank_query_is_rejected(query):
    service, _, _, _ = await build()

    with pytest.raises(InvalidRetrievalRequestError) as exc_info:
        await service.retrieve(meeting_id="m1", query=query)

    assert exc_info.value.status_code == 422


@pytest.mark.parametrize("meeting_id", ["", "   "])
async def test_blank_meeting_id_is_rejected(meeting_id):
    service, _, _, _ = await build()

    with pytest.raises(InvalidRetrievalRequestError):
        await service.retrieve(meeting_id=meeting_id, query="budget")


@pytest.mark.parametrize("k", [0, -1, -10])
async def test_non_positive_k_is_rejected(k):
    service, _, _, _ = await build()

    with pytest.raises(InvalidRetrievalRequestError) as exc_info:
        await service.retrieve(meeting_id="m1", query="budget", k=k)

    assert exc_info.value.details == {"k": k}


async def test_invalid_request_is_rejected_before_embedding():
    embeddings = RecordingEmbeddingProvider()
    service, _, _, _ = await build(embeddings=embeddings)
    embeddings.query_calls.clear()

    with pytest.raises(InvalidRetrievalRequestError):
        await service.retrieve(meeting_id="m1", query="budget", k=0)

    assert embeddings.query_calls == []


async def test_meeting_id_is_trimmed():
    service, _, _, _ = await build()

    results = await service.retrieve(meeting_id="  m1  ", query="budget")

    assert all(r.chunk.meeting_id == "m1" for r in results)


# --- empty and isolation ---------------------------------------------------


async def test_known_meeting_with_no_vectors_returns_an_empty_list():
    service, _, vector_store, _ = await build()
    await vector_store.delete_meeting("m1")

    assert await service.retrieve(meeting_id="m1", query="budget") == []


async def test_empty_result_is_not_an_error():
    service, _, vector_store, _ = await build()
    await vector_store.delete_meeting("m1")

    results = await service.retrieve(meeting_id="m1", query="anything at all")

    assert results == []


async def test_retrieval_never_crosses_meetings():
    service, _, _, _ = await build(transcripts={"m1": SAMPLE, "m2": OTHER})

    results = await service.retrieve(meeting_id="m1", query="hiring pipeline engineers", k=10)

    assert results
    assert all(r.chunk.meeting_id == "m1" for r in results)
    assert all(uid.startswith("m1:") for r in results for uid in r.chunk.utterance_ids)


async def test_each_meeting_returns_its_own_chunks():
    service, _, _, _ = await build(transcripts={"m1": SAMPLE, "m2": OTHER})

    first = await service.retrieve(meeting_id="m1", query="budget", k=10)
    second = await service.retrieve(meeting_id="m2", query="budget", k=10)

    assert {r.chunk.id for r in first}.isdisjoint({r.chunk.id for r in second})


async def test_deleting_a_meeting_empties_its_retrieval():
    service, _, vector_store, _ = await build(transcripts={"m1": SAMPLE, "m2": OTHER})
    await vector_store.delete_meeting("m1")

    assert await service.retrieve(meeting_id="m1", query="budget") == []
    assert await service.retrieve(meeting_id="m2", query="hiring")


# --- determinism -----------------------------------------------------------


async def test_repeated_retrieval_is_identical():
    service, _, _, _ = await build()

    first = await service.retrieve(meeting_id="m1", query="marketing budget", k=10)
    second = await service.retrieve(meeting_id="m1", query="marketing budget", k=10)

    assert [(r.chunk.id, r.score) for r in first] == [(r.chunk.id, r.score) for r in second]


async def test_retrieval_requires_no_network():
    """The whole path runs on the fake provider and in-memory stores."""
    service, embeddings, _, _ = await build()

    assert isinstance(embeddings, FakeEmbeddingProvider)
    assert await service.retrieve(meeting_id="m1", query="budget")
