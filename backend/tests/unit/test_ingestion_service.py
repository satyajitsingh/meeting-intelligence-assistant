"""IngestionService orchestration, isolation and failure consistency."""

import pytest

from app.adapters.embeddings.base import EmbeddingError
from app.adapters.embeddings.fake import FakeEmbeddingProvider
from app.adapters.repository.memory import InMemoryTranscriptRepository
from app.adapters.vectorstore.base import VectorStoreError
from app.adapters.vectorstore.memory import InMemoryVectorStore
from app.core.errors import NotFoundError
from app.domain.chunker import chunk_transcript
from app.domain.errors import EmptyTranscriptError, TranscriptParseError
from app.domain.models import Chunk, Transcript
from app.domain.parser import parse_transcript
from app.services.ingestion import IngestionConfigurationError, IngestionService

pytestmark = pytest.mark.anyio

DIMENSION = 128

SAMPLE = (
    "[00:00:12] Sarah: We need to delay the release because the migration is unfinished.\n"
    "[00:00:31] John: Agreed, the migration script still fails on legacy accounts.\n"
    "[00:00:52] Amir: What does the delay mean for the marketing budget we approved?\n"
    "[00:01:14] Sarah: The budget is unchanged, only the launch date moves.\n"
)

SHORTER = "[00:00:05] Amir: One line only.\n"


class RecordingEmbeddingProvider(FakeEmbeddingProvider):
    """Fake provider that remembers exactly what it was asked to embed."""

    def __init__(self, dimension: int = DIMENSION) -> None:
        super().__init__(dimension=dimension)
        self.document_calls: list[list[str]] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return await super().embed_documents(texts)


class FailingEmbeddingProvider(FakeEmbeddingProvider):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("provider unavailable")


class FailingVectorStore(InMemoryVectorStore):
    """Deletes normally, then fails on upsert.

    That combination is the interesting one: it reproduces a failure *after*
    the old vectors have already been cleared.
    """

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        raise VectorStoreError("store unavailable")


class OrderedEmbeddingProvider(FakeEmbeddingProvider):
    """Fake provider that appends to a shared call log."""

    def __init__(self, calls: list[str], dimension: int = DIMENSION) -> None:
        super().__init__(dimension=dimension)
        self.calls = calls

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append("embed")
        return await super().embed_documents(texts)


class OrderedVectorStore(InMemoryVectorStore):
    """In-memory store that appends to a shared call log."""

    def __init__(self, calls: list[str], dimension: int = DIMENSION) -> None:
        super().__init__(dimension=dimension)
        self.calls = calls

    async def delete_meeting(self, meeting_id: str) -> None:
        self.calls.append("delete_vectors")
        await super().delete_meeting(meeting_id)

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.calls.append("upsert_vectors")
        await super().upsert(chunks, vectors)


class OrderedRepository(InMemoryTranscriptRepository):
    """In-memory repository that appends to a shared call log."""

    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self.calls = calls

    async def save(self, transcript: Transcript) -> None:
        self.calls.append("save_transcript")
        await super().save(transcript)


def build_service(
    embeddings: FakeEmbeddingProvider | None = None,
    vector_store: InMemoryVectorStore | None = None,
    repository: InMemoryTranscriptRepository | None = None,
    target_chars: int = 700,
) -> tuple[
    IngestionService,
    FakeEmbeddingProvider,
    InMemoryVectorStore,
    InMemoryTranscriptRepository,
]:
    embeddings = embeddings or FakeEmbeddingProvider(dimension=DIMENSION)
    vector_store = vector_store or InMemoryVectorStore(dimension=DIMENSION)
    repository = repository or InMemoryTranscriptRepository()
    service = IngestionService(
        embeddings=embeddings,
        vector_store=vector_store,
        repository=repository,
        target_chars=target_chars,
    )
    return service, embeddings, vector_store, repository


async def stored_chunk_count(store: InMemoryVectorStore, meeting_id: str) -> int:
    query = [0.0] * DIMENSION
    query[0] = 1.0
    return len(await store.search(query, meeting_id=meeting_id, k=1000))


# --- construction ----------------------------------------------------------


def test_rejects_mismatched_dimensions_at_construction():
    with pytest.raises(IngestionConfigurationError) as exc_info:
        IngestionService(
            embeddings=FakeEmbeddingProvider(dimension=64),
            vector_store=InMemoryVectorStore(dimension=128),
            repository=InMemoryTranscriptRepository(),
        )

    assert exc_info.value.details == {
        "embedding_dimension": 64,
        "vector_store_dimension": 128,
    }


def test_accepts_matching_dimensions():
    service, *_ = build_service()

    assert service is not None


# --- happy path ------------------------------------------------------------


async def test_ingest_returns_a_complete_summary():
    service, _, _, _ = build_service()

    result = await service.ingest(meeting_id="m1", title="Release planning", transcript_text=SAMPLE)

    assert result.summary.meeting_id == "m1"
    assert result.summary.title == "Release planning"
    assert result.summary.speakers == ["Sarah", "John", "Amir"]
    assert result.summary.utterance_count == 4
    assert result.summary.duration_seconds == 74
    assert result.chunk_count >= 1


async def test_ingest_saves_the_parsed_transcript():
    service, _, _, repository = build_service()

    await service.ingest(meeting_id="m1", title="Release planning", transcript_text=SAMPLE)

    stored = await repository.get("m1")
    assert stored is not None
    assert stored.utterances[0].text.startswith("We need to delay")


async def test_ingest_stores_every_chunk_in_the_vector_store():
    service, _, vector_store, _ = build_service(target_chars=150)

    result = await service.ingest(meeting_id="m1", title="T", transcript_text=SAMPLE)

    assert await stored_chunk_count(vector_store, "m1") == result.chunk_count


async def test_chunk_count_matches_the_chunker():
    service, _, _, _ = build_service(target_chars=150)

    result = await service.ingest(meeting_id="m1", title="T", transcript_text=SAMPLE)

    expected = chunk_transcript(
        parse_transcript(SAMPLE, meeting_id="m1", title="T"), target_chars=150
    )
    assert result.chunk_count == len(expected)


async def test_chunk_count_is_deterministic():
    first, *_ = build_service(target_chars=150)
    second, *_ = build_service(target_chars=150)

    a = await first.ingest(meeting_id="m1", title="T", transcript_text=SAMPLE)
    b = await second.ingest(meeting_id="m1", title="T", transcript_text=SAMPLE)

    assert a.chunk_count == b.chunk_count


async def test_embedding_receives_chunk_text_with_speaker_labels():
    recorder = RecordingEmbeddingProvider()
    service, _, _, _ = build_service(embeddings=recorder, target_chars=150)

    await service.ingest(meeting_id="m1", title="T", transcript_text=SAMPLE)

    assert len(recorder.document_calls) == 1
    embedded = recorder.document_calls[0]
    expected = chunk_transcript(
        parse_transcript(SAMPLE, meeting_id="m1", title="T"), target_chars=150
    )
    assert embedded == [chunk.text for chunk in expected]
    assert all(
        line.split(":")[0] in {"Sarah", "John", "Amir"}
        for text in embedded
        for line in text.splitlines()
    )


async def test_target_chars_is_honoured():
    coarse, _, _, _ = build_service(target_chars=1000)
    fine, _, _, _ = build_service(target_chars=100)

    a = await coarse.ingest(meeting_id="m1", title="T", transcript_text=SAMPLE)
    b = await fine.ingest(meeting_id="m1", title="T", transcript_text=SAMPLE)

    assert b.chunk_count > a.chunk_count


# --- multiple meetings -----------------------------------------------------


async def test_multiple_meetings_are_stored_independently():
    service, _, vector_store, repository = build_service()

    await service.ingest(meeting_id="m1", title="One", transcript_text=SAMPLE)
    await service.ingest(meeting_id="m2", title="Two", transcript_text=SHORTER)

    assert len(await repository.list()) == 2
    assert await stored_chunk_count(vector_store, "m1") >= 1
    assert await stored_chunk_count(vector_store, "m2") == 1


async def test_chunks_are_scoped_to_their_meeting():
    service, _, vector_store, _ = build_service(target_chars=150)

    await service.ingest(meeting_id="m1", title="One", transcript_text=SAMPLE)
    await service.ingest(meeting_id="m2", title="Two", transcript_text=SHORTER)

    query = [0.0] * DIMENSION
    query[0] = 1.0
    results = await vector_store.search(query, meeting_id="m2", k=100)

    assert all(scored.chunk.meeting_id == "m2" for scored in results)


# --- re-ingestion ----------------------------------------------------------


async def test_reingesting_replaces_the_stored_transcript():
    service, _, _, repository = build_service()
    await service.ingest(meeting_id="m1", title="Original", transcript_text=SAMPLE)

    await service.ingest(meeting_id="m1", title="Revised", transcript_text=SHORTER)

    stored = await repository.get("m1")
    assert stored is not None
    assert stored.title == "Revised"
    assert len(stored.utterances) == 1


async def test_reingesting_a_shorter_transcript_leaves_no_stale_chunks():
    """Chunk IDs are positional, so upsert alone would strand old chunks."""
    service, _, vector_store, _ = build_service(target_chars=100)
    await service.ingest(meeting_id="m1", title="Original", transcript_text=SAMPLE)
    assert await stored_chunk_count(vector_store, "m1") > 1

    result = await service.ingest(meeting_id="m1", title="Revised", transcript_text=SHORTER)

    assert result.chunk_count == 1
    assert await stored_chunk_count(vector_store, "m1") == 1


async def test_reingesting_does_not_disturb_other_meetings():
    service, _, vector_store, repository = build_service(target_chars=150)
    await service.ingest(meeting_id="m1", title="One", transcript_text=SAMPLE)
    await service.ingest(meeting_id="m2", title="Two", transcript_text=SAMPLE)
    before = await stored_chunk_count(vector_store, "m2")

    await service.ingest(meeting_id="m1", title="One again", transcript_text=SHORTER)

    assert await stored_chunk_count(vector_store, "m2") == before
    stored = await repository.get("m2")
    assert stored is not None and stored.title == "Two"


async def test_reingesting_keeps_a_single_listing_entry():
    service, _, _, repository = build_service()

    await service.ingest(meeting_id="m1", title="One", transcript_text=SAMPLE)
    await service.ingest(meeting_id="m1", title="One again", transcript_text=SAMPLE)

    assert [s.meeting_id for s in await repository.list()] == ["m1"]


# --- error propagation -----------------------------------------------------


async def test_parser_errors_propagate():
    service, _, _, _ = build_service()

    with pytest.raises(TranscriptParseError) as exc_info:
        await service.ingest(meeting_id="m1", title="T", transcript_text="[00:01] Sarah Hello.\n")

    assert exc_info.value.line_number == 1


async def test_empty_transcript_errors_propagate():
    service, _, _, _ = build_service()

    with pytest.raises(EmptyTranscriptError):
        await service.ingest(meeting_id="m1", title="T", transcript_text="   \n  ")


async def test_a_parse_failure_stores_nothing():
    service, _, vector_store, repository = build_service()

    with pytest.raises(TranscriptParseError):
        await service.ingest(meeting_id="m1", title="T", transcript_text="bad input\n")

    assert await repository.get("m1") is None
    assert await stored_chunk_count(vector_store, "m1") == 0


async def test_a_parse_failure_leaves_a_previous_ingest_intact():
    service, _, vector_store, repository = build_service()
    await service.ingest(meeting_id="m1", title="Good", transcript_text=SAMPLE)
    before = await stored_chunk_count(vector_store, "m1")

    with pytest.raises(TranscriptParseError):
        await service.ingest(meeting_id="m1", title="Bad", transcript_text="nonsense\n")

    stored = await repository.get("m1")
    assert stored is not None and stored.title == "Good"
    assert await stored_chunk_count(vector_store, "m1") == before


# --- failure consistency ---------------------------------------------------


async def test_an_embedding_failure_does_not_save_the_transcript():
    service, _, vector_store, repository = build_service(
        embeddings=FailingEmbeddingProvider(dimension=DIMENSION)
    )

    with pytest.raises(EmbeddingError):
        await service.ingest(meeting_id="m1", title="T", transcript_text=SAMPLE)

    assert await repository.get("m1") is None
    assert await stored_chunk_count(vector_store, "m1") == 0


async def test_an_embedding_failure_leaves_a_previous_ingest_intact():
    """Embedding runs before any store is touched, so prior state survives."""
    repository = InMemoryTranscriptRepository()
    vector_store = InMemoryVectorStore(dimension=DIMENSION)
    good, _, _, _ = build_service(vector_store=vector_store, repository=repository)
    await good.ingest(meeting_id="m1", title="Good", transcript_text=SAMPLE)

    failing, _, _, _ = build_service(
        embeddings=FailingEmbeddingProvider(dimension=DIMENSION),
        vector_store=vector_store,
        repository=repository,
    )
    with pytest.raises(EmbeddingError):
        await failing.ingest(meeting_id="m1", title="Bad", transcript_text=SHORTER)

    stored = await repository.get("m1")
    assert stored is not None and stored.title == "Good"
    assert await stored_chunk_count(vector_store, "m1") > 0


async def test_a_vector_store_failure_does_not_save_the_transcript():
    service, _, _, repository = build_service(vector_store=FailingVectorStore(dimension=DIMENSION))

    with pytest.raises(VectorStoreError):
        await service.ingest(meeting_id="m1", title="T", transcript_text=SAMPLE)

    assert await repository.get("m1") is None


async def test_a_first_ingest_that_fails_at_upsert_stores_nothing():
    store = FailingVectorStore(dimension=DIMENSION)
    repository = InMemoryTranscriptRepository()
    service, _, _, _ = build_service(vector_store=store, repository=repository)

    with pytest.raises(VectorStoreError):
        await service.ingest(meeting_id="m1", title="T", transcript_text=SAMPLE)

    assert await repository.get("m1") is None
    assert await stored_chunk_count(store, "m1") == 0
    assert [s.meeting_id for s in await repository.list()] == []


# --- re-ingestion failure ordering -----------------------------------------


async def test_ingest_runs_its_steps_in_the_declared_order():
    """embed -> delete old vectors -> upsert new vectors -> save transcript."""
    calls: list[str] = []
    service = IngestionService(
        embeddings=OrderedEmbeddingProvider(calls),
        vector_store=OrderedVectorStore(calls),
        repository=OrderedRepository(calls),
    )

    await service.ingest(meeting_id="m1", title="T", transcript_text=SAMPLE)

    assert calls == ["embed", "delete_vectors", "upsert_vectors", "save_transcript"]


async def test_reingest_embeds_before_deleting_the_old_vectors():
    """Embedding must not be able to fail after old state was destroyed."""
    calls: list[str] = []
    service = IngestionService(
        embeddings=OrderedEmbeddingProvider(calls),
        vector_store=OrderedVectorStore(calls),
        repository=OrderedRepository(calls),
    )
    await service.ingest(meeting_id="m1", title="One", transcript_text=SAMPLE)
    calls.clear()

    await service.ingest(meeting_id="m1", title="Two", transcript_text=SHORTER)

    assert calls.index("embed") < calls.index("delete_vectors")


async def test_a_failed_reingest_does_not_save_the_new_transcript():
    """The repository must never point at vectors that were never stored."""
    store = FailingVectorStore(dimension=DIMENSION)
    repository = InMemoryTranscriptRepository()

    # Seed a good ingest through a working store sharing the same repository.
    good, _, _, _ = build_service(
        vector_store=InMemoryVectorStore(dimension=DIMENSION), repository=repository
    )
    await good.ingest(meeting_id="m1", title="Good", transcript_text=SAMPLE)

    failing, _, _, _ = build_service(vector_store=store, repository=repository)
    with pytest.raises(VectorStoreError):
        await failing.ingest(meeting_id="m1", title="Replacement", transcript_text=SHORTER)

    stored = await repository.get("m1")
    assert stored is not None
    assert stored.title == "Good"
    assert len(stored.utterances) == 4


async def test_a_failed_reingest_leaves_the_old_vectors_cleared():
    """Documented v1 limitation: no rollback, so re-posting is the recovery."""
    store = FailingVectorStore(dimension=DIMENSION)
    repository = InMemoryTranscriptRepository()
    service, _, _, _ = build_service(vector_store=store, repository=repository)

    # Seed vectors directly through the same store, bypassing the failing upsert.
    transcript = parse_transcript(SAMPLE, meeting_id="m1", title="Good")
    chunks = chunk_transcript(transcript, target_chars=700)
    embeddings = FakeEmbeddingProvider(dimension=DIMENSION)
    await InMemoryVectorStore.upsert(
        store, chunks, await embeddings.embed_documents([c.text for c in chunks])
    )
    await repository.save(transcript)
    assert await stored_chunk_count(store, "m1") > 0

    with pytest.raises(VectorStoreError):
        await service.ingest(meeting_id="m1", title="Replacement", transcript_text=SHORTER)

    assert await stored_chunk_count(store, "m1") == 0
    still_there = await repository.get("m1")
    assert still_there is not None and still_there.title == "Good"


async def test_a_failed_reingest_is_recoverable_by_reposting():
    """The accepted inconsistency is repaired by a successful re-ingest."""
    repository = InMemoryTranscriptRepository()
    store = InMemoryVectorStore(dimension=DIMENSION)
    service, _, _, _ = build_service(vector_store=store, repository=repository)
    await service.ingest(meeting_id="m1", title="Good", transcript_text=SAMPLE)

    failing, _, _, _ = build_service(
        vector_store=FailingVectorStore(dimension=DIMENSION), repository=repository
    )
    with pytest.raises(VectorStoreError):
        await failing.ingest(meeting_id="m1", title="Broken", transcript_text=SHORTER)

    result = await service.ingest(meeting_id="m1", title="Repaired", transcript_text=SHORTER)

    stored = await repository.get("m1")
    assert stored is not None and stored.title == "Repaired"
    assert await stored_chunk_count(store, "m1") == result.chunk_count


async def test_old_vectors_survive_a_parse_failure_during_reingest():
    service, _, vector_store, repository = build_service()
    await service.ingest(meeting_id="m1", title="Good", transcript_text=SAMPLE)
    before = await stored_chunk_count(vector_store, "m1")

    with pytest.raises(TranscriptParseError):
        await service.ingest(meeting_id="m1", title="Bad", transcript_text="nonsense\n")

    assert await stored_chunk_count(vector_store, "m1") == before
    stored = await repository.get("m1")
    assert stored is not None and stored.title == "Good"


async def test_old_vectors_survive_an_embedding_failure_during_reingest():
    repository = InMemoryTranscriptRepository()
    vector_store = InMemoryVectorStore(dimension=DIMENSION)
    good, _, _, _ = build_service(vector_store=vector_store, repository=repository)
    await good.ingest(meeting_id="m1", title="Good", transcript_text=SAMPLE)
    before = await stored_chunk_count(vector_store, "m1")

    failing, _, _, _ = build_service(
        embeddings=FailingEmbeddingProvider(dimension=DIMENSION),
        vector_store=vector_store,
        repository=repository,
    )
    with pytest.raises(EmbeddingError):
        await failing.ingest(meeting_id="m1", title="Bad", transcript_text=SHORTER)

    assert await stored_chunk_count(vector_store, "m1") == before
    stored = await repository.get("m1")
    assert stored is not None and stored.title == "Good"


# --- delete ----------------------------------------------------------------


async def test_delete_removes_transcript_and_chunks():
    service, _, vector_store, repository = build_service()
    await service.ingest(meeting_id="m1", title="T", transcript_text=SAMPLE)

    await service.delete("m1")

    assert await repository.get("m1") is None
    assert await stored_chunk_count(vector_store, "m1") == 0


async def test_delete_of_an_unknown_meeting_raises_not_found():
    service, _, _, _ = build_service()

    with pytest.raises(NotFoundError) as exc_info:
        await service.delete("never-ingested")

    assert exc_info.value.status_code == 404
    assert exc_info.value.details == {"meeting_id": "never-ingested"}


async def test_delete_leaves_other_meetings_intact():
    service, _, vector_store, repository = build_service()
    await service.ingest(meeting_id="m1", title="One", transcript_text=SAMPLE)
    await service.ingest(meeting_id="m2", title="Two", transcript_text=SAMPLE)

    await service.delete("m1")

    assert await repository.get("m2") is not None
    assert await stored_chunk_count(vector_store, "m2") > 0
