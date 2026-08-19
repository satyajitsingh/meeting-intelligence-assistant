"""InMemoryVectorStore: exact cosine search, scoped to one meeting."""

import pytest

from app.adapters.embeddings.fake import FakeEmbeddingProvider
from app.adapters.vectorstore.base import VectorStore, VectorStoreError
from app.adapters.vectorstore.memory import InMemoryVectorStore
from app.domain.chunker import chunk_transcript
from app.domain.models import Chunk, ScoredChunk, Utterance, make_utterance_id
from app.domain.parser import parse_transcript

pytestmark = pytest.mark.anyio

DIMENSION = 4

SAMPLE = """\
[00:00:12] Sarah: We need to delay the release because the migration is unfinished.
[00:00:25] John: Agreed, the migration script still fails on legacy accounts.
[00:00:40] Amir: What does that mean for the marketing budget we approved?
[00:01:02] Sarah: The budget is unchanged, only the launch date moves.
"""


def make_chunk(chunk_id: str, meeting_id: str = "m1", index: int = 0) -> Chunk:
    utterance = Utterance(
        id=make_utterance_id(meeting_id, index),
        meeting_id=meeting_id,
        index=index,
        speaker="Sarah",
        start_seconds=index * 10,
        raw_timestamp="00:00:00",
        text=f"Dialogue for {chunk_id}.",
    )
    return Chunk(
        id=chunk_id,
        meeting_id=meeting_id,
        index=index,
        text=f"Sarah: Dialogue for {chunk_id}.",
        utterance_ids=[utterance.id],
        speakers=["Sarah"],
        start_seconds=utterance.start_seconds,
        end_seconds=utterance.start_seconds,
    )


def store() -> InMemoryVectorStore:
    return InMemoryVectorStore(dimension=DIMENSION)


# Axis-aligned vectors make expected cosine scores obvious.
X = [1.0, 0.0, 0.0, 0.0]
Y = [0.0, 1.0, 0.0, 0.0]
Z = [0.0, 0.0, 1.0, 0.0]
XY = [1.0, 1.0, 0.0, 0.0]
ZERO = [0.0, 0.0, 0.0, 0.0]


# --- protocol and construction --------------------------------------------


def test_satisfies_the_vector_store_protocol():
    assert isinstance(store(), VectorStore)


def test_exposes_its_dimension():
    assert store().dimension == DIMENSION


@pytest.mark.parametrize("dimension", [0, -1])
def test_rejects_a_non_positive_dimension(dimension):
    with pytest.raises(VectorStoreError):
        InMemoryVectorStore(dimension=dimension)


# --- insert and search -----------------------------------------------------


async def test_insert_then_search_returns_the_chunk():
    vector_store = store()
    await vector_store.upsert([make_chunk("m1:c0")], [X])

    results = await vector_store.search(X, meeting_id="m1", k=5)

    assert len(results) == 1
    assert isinstance(results[0], ScoredChunk)
    assert results[0].chunk.id == "m1:c0"
    assert results[0].score == pytest.approx(1.0)


async def test_nearest_result_ranks_first():
    vector_store = store()
    await vector_store.upsert(
        [make_chunk("m1:c0"), make_chunk("m1:c1", index=1), make_chunk("m1:c2", index=2)],
        [Z, X, Y],
    )

    results = await vector_store.search(X, meeting_id="m1", k=3)

    assert results[0].chunk.id == "m1:c1"
    assert results[0].score == pytest.approx(1.0)


async def test_results_are_ordered_by_descending_score():
    vector_store = store()
    await vector_store.upsert(
        [make_chunk("m1:c0"), make_chunk("m1:c1", index=1), make_chunk("m1:c2", index=2)],
        [Y, XY, X],
    )

    results = await vector_store.search(X, meeting_id="m1", k=3)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].chunk.id == "m1:c2"
    assert results[1].chunk.id == "m1:c1"


async def test_scores_are_true_cosine_similarity():
    vector_store = store()
    await vector_store.upsert([make_chunk("m1:c0")], [XY])

    results = await vector_store.search(X, meeting_id="m1", k=1)

    assert results[0].score == pytest.approx(0.7071, abs=1e-4)


async def test_unnormalised_vectors_are_normalised_on_insert():
    vector_store = store()
    await vector_store.upsert([make_chunk("m1:c0")], [[5.0, 0.0, 0.0, 0.0]])

    results = await vector_store.search([100.0, 0.0, 0.0, 0.0], meeting_id="m1", k=1)

    assert results[0].score == pytest.approx(1.0)


async def test_scores_stay_within_the_cosine_range():
    vector_store = store()
    await vector_store.upsert([make_chunk("m1:c0")], [X])

    results = await vector_store.search([-1.0, 0.0, 0.0, 0.0], meeting_id="m1", k=1)

    assert -1.0 <= results[0].score <= 1.0
    assert results[0].score == pytest.approx(-1.0)


# --- top-k -----------------------------------------------------------------


async def test_search_returns_at_most_k_results():
    vector_store = store()
    chunks = [make_chunk(f"m1:c{i}", index=i) for i in range(5)]
    await vector_store.upsert(chunks, [X, Y, Z, XY, X])

    assert len(await vector_store.search(X, meeting_id="m1", k=2)) == 2


async def test_k_larger_than_the_store_returns_everything():
    vector_store = store()
    await vector_store.upsert([make_chunk("m1:c0"), make_chunk("m1:c1", index=1)], [X, Y])

    assert len(await vector_store.search(X, meeting_id="m1", k=100)) == 2


@pytest.mark.parametrize("k", [0, -1, -10])
async def test_rejects_a_non_positive_k(k):
    vector_store = store()
    await vector_store.upsert([make_chunk("m1:c0")], [X])

    with pytest.raises(VectorStoreError) as exc_info:
        await vector_store.search(X, meeting_id="m1", k=k)

    assert exc_info.value.details == {"k": k}


async def test_invalid_k_is_rejected_even_on_an_empty_store():
    with pytest.raises(VectorStoreError):
        await store().search(X, meeting_id="unknown", k=0)


# --- meeting isolation -----------------------------------------------------


async def test_search_is_scoped_to_one_meeting():
    vector_store = store()
    await vector_store.upsert(
        [make_chunk("m1:c0", meeting_id="m1"), make_chunk("m2:c0", meeting_id="m2")],
        [X, X],
    )

    results = await vector_store.search(X, meeting_id="m1", k=10)

    assert [r.chunk.id for r in results] == ["m1:c0"]


async def test_unknown_meeting_returns_no_results():
    vector_store = store()
    await vector_store.upsert([make_chunk("m1:c0")], [X])

    assert await vector_store.search(X, meeting_id="does-not-exist", k=5) == []


async def test_empty_store_returns_no_results():
    assert await store().search(X, meeting_id="m1", k=5) == []


async def test_upsert_files_each_chunk_under_its_own_meeting():
    vector_store = store()
    await vector_store.upsert(
        [make_chunk("m1:c0", meeting_id="m1"), make_chunk("m2:c0", meeting_id="m2")],
        [X, Y],
    )

    assert len(await vector_store.search(Y, meeting_id="m2", k=10)) == 1


# --- delete ----------------------------------------------------------------


async def test_delete_meeting_removes_its_chunks():
    vector_store = store()
    await vector_store.upsert([make_chunk("m1:c0")], [X])

    await vector_store.delete_meeting("m1")

    assert await vector_store.search(X, meeting_id="m1", k=5) == []


async def test_delete_meeting_leaves_other_meetings_intact():
    vector_store = store()
    await vector_store.upsert(
        [make_chunk("m1:c0", meeting_id="m1"), make_chunk("m2:c0", meeting_id="m2")],
        [X, X],
    )

    await vector_store.delete_meeting("m1")

    assert await vector_store.search(X, meeting_id="m1", k=5) == []
    assert len(await vector_store.search(X, meeting_id="m2", k=5)) == 1


async def test_deleting_an_unknown_meeting_is_a_no_op():
    vector_store = store()
    await vector_store.upsert([make_chunk("m1:c0")], [X])

    await vector_store.delete_meeting("never-stored")

    assert len(await vector_store.search(X, meeting_id="m1", k=5)) == 1


async def test_a_meeting_can_be_repopulated_after_deletion():
    vector_store = store()
    await vector_store.upsert([make_chunk("m1:c0")], [X])
    await vector_store.delete_meeting("m1")

    await vector_store.upsert([make_chunk("m1:c0")], [Y])

    results = await vector_store.search(Y, meeting_id="m1", k=5)
    assert len(results) == 1
    assert results[0].score == pytest.approx(1.0)


# --- upsert semantics ------------------------------------------------------


async def test_upsert_replaces_a_chunk_with_the_same_id():
    vector_store = store()
    await vector_store.upsert([make_chunk("m1:c0")], [X])

    await vector_store.upsert([make_chunk("m1:c0")], [Y])

    results = await vector_store.search(Y, meeting_id="m1", k=10)
    assert len(results) == 1
    assert results[0].score == pytest.approx(1.0)


async def test_repeated_upsert_does_not_duplicate_chunks():
    vector_store = store()
    chunks = [make_chunk("m1:c0"), make_chunk("m1:c1", index=1)]

    for _ in range(3):
        await vector_store.upsert(chunks, [X, Y])

    assert len(await vector_store.search(X, meeting_id="m1", k=100)) == 2


async def test_replacement_updates_the_stored_chunk_metadata():
    vector_store = store()
    original = make_chunk("m1:c0")
    await vector_store.upsert([original], [X])

    revised = original.model_copy(update={"text": "Sarah: Revised dialogue."})
    await vector_store.upsert([revised], [X])

    results = await vector_store.search(X, meeting_id="m1", k=1)
    assert results[0].chunk.text == "Sarah: Revised dialogue."


async def test_empty_upsert_is_a_no_op():
    vector_store = store()

    await vector_store.upsert([], [])

    assert await vector_store.search(X, meeting_id="m1", k=5) == []


# --- validation ------------------------------------------------------------


async def test_rejects_mismatched_chunk_and_vector_counts():
    vector_store = store()

    with pytest.raises(VectorStoreError) as exc_info:
        await vector_store.upsert([make_chunk("m1:c0"), make_chunk("m1:c1", index=1)], [X])

    assert exc_info.value.details == {"chunks": 2, "vectors": 1}


async def test_rejects_a_stored_vector_of_the_wrong_dimension():
    vector_store = store()

    with pytest.raises(VectorStoreError) as exc_info:
        await vector_store.upsert([make_chunk("m1:c0")], [[1.0, 0.0]])

    assert exc_info.value.details is not None
    assert exc_info.value.details["expected_dimension"] == DIMENSION
    assert exc_info.value.details["actual_dimension"] == 2


async def test_rejects_a_query_vector_of_the_wrong_dimension():
    vector_store = store()
    await vector_store.upsert([make_chunk("m1:c0")], [X])

    with pytest.raises(VectorStoreError):
        await vector_store.search([1.0, 0.0], meeting_id="m1", k=5)


async def test_validation_errors_are_reported_as_422():
    with pytest.raises(VectorStoreError) as exc_info:
        await store().search(X, meeting_id="m1", k=0)

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "vector_store_error"


# --- zero vectors ----------------------------------------------------------


async def test_zero_query_vector_scores_zero_rather_than_nan():
    vector_store = store()
    await vector_store.upsert([make_chunk("m1:c0")], [X])

    results = await vector_store.search(ZERO, meeting_id="m1", k=5)

    assert results[0].score == 0.0


async def test_zero_stored_vector_scores_zero_rather_than_nan():
    vector_store = store()
    await vector_store.upsert([make_chunk("m1:c0")], [ZERO])

    results = await vector_store.search(X, meeting_id="m1", k=5)

    assert results[0].score == 0.0


async def test_a_zero_vector_never_outranks_a_real_match():
    vector_store = store()
    await vector_store.upsert([make_chunk("m1:c0"), make_chunk("m1:c1", index=1)], [ZERO, X])

    results = await vector_store.search(X, meeting_id="m1", k=2)

    assert results[0].chunk.id == "m1:c1"


# --- deterministic ordering ------------------------------------------------


async def test_equal_scores_break_ties_by_insertion_order():
    vector_store = store()
    chunks = [make_chunk(f"m1:c{i}", index=i) for i in range(5)]
    await vector_store.upsert(chunks, [X] * 5)

    results = await vector_store.search(X, meeting_id="m1", k=5)

    assert [r.chunk.id for r in results] == [f"m1:c{i}" for i in range(5)]


async def test_tie_break_order_is_stable_across_repeated_searches():
    vector_store = store()
    await vector_store.upsert([make_chunk(f"m1:c{i}", index=i) for i in range(5)], [X] * 5)

    first = await vector_store.search(X, meeting_id="m1", k=5)
    second = await vector_store.search(X, meeting_id="m1", k=5)

    assert [r.chunk.id for r in first] == [r.chunk.id for r in second]


async def test_replacement_preserves_tie_break_position():
    vector_store = store()
    chunks = [make_chunk(f"m1:c{i}", index=i) for i in range(3)]
    await vector_store.upsert(chunks, [X] * 3)

    await vector_store.upsert([chunks[0]], [X])

    results = await vector_store.search(X, meeting_id="m1", k=3)
    assert [r.chunk.id for r in results] == ["m1:c0", "m1:c1", "m1:c2"]


# --- full chunk metadata ---------------------------------------------------


async def test_scored_chunk_retains_the_complete_chunk():
    vector_store = store()
    transcript = parse_transcript(SAMPLE, meeting_id="m1", title="Release planning")
    chunk = chunk_transcript(transcript, target_chars=200)[0]

    await vector_store.upsert([chunk], [X])
    result = (await vector_store.search(X, meeting_id="m1", k=1))[0]

    assert result.chunk == chunk
    assert result.chunk.utterance_ids == chunk.utterance_ids
    assert result.chunk.speakers == chunk.speakers
    assert result.chunk.start_seconds == chunk.start_seconds
    assert result.chunk.end_seconds == chunk.end_seconds
    assert result.chunk.text == chunk.text


async def test_stores_every_chunk_of_a_transcript_and_retrieves_them():
    transcript = parse_transcript(SAMPLE, meeting_id="m1", title="Release planning")
    chunks = chunk_transcript(transcript, target_chars=150)
    provider = FakeEmbeddingProvider(dimension=128)
    vector_store = InMemoryVectorStore(dimension=provider.dimension)

    vectors = await provider.embed_documents([c.text for c in chunks])
    await vector_store.upsert(chunks, vectors)

    query = await provider.embed_query("anything")
    results = await vector_store.search(query, meeting_id="m1", k=100)

    assert len(results) == len(chunks)
    assert {r.chunk.id for r in results} == {c.id for c in chunks}


async def test_search_finds_the_chunk_matching_the_question():
    """End-to-end ranking with the fake provider, no model download."""
    transcript = parse_transcript(SAMPLE, meeting_id="m1", title="Release planning")
    chunks = chunk_transcript(transcript, target_chars=120)
    provider = FakeEmbeddingProvider(dimension=512)
    vector_store = InMemoryVectorStore(dimension=provider.dimension)

    await vector_store.upsert(chunks, await provider.embed_documents([c.text for c in chunks]))

    query = await provider.embed_query("what happens to the marketing budget")
    top = (await vector_store.search(query, meeting_id="m1", k=1))[0]

    assert "budget" in top.chunk.text.lower()
