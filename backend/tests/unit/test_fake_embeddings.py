"""FakeEmbeddingProvider: deterministic, offline, and useful for ranking tests."""

import math
import subprocess
import sys

import pytest

from app.adapters.embeddings.base import EmbeddingError, EmbeddingProvider, ensure_dimension
from app.adapters.embeddings.fake import DEFAULT_DIMENSION, FakeEmbeddingProvider

pytestmark = pytest.mark.anyio


def cosine(a: list[float], b: list[float]) -> float:
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (norm_a * norm_b)


# --- protocol conformance --------------------------------------------------


def test_satisfies_the_embedding_provider_protocol():
    assert isinstance(FakeEmbeddingProvider(), EmbeddingProvider)


def test_default_dimension():
    assert FakeEmbeddingProvider().dimension == DEFAULT_DIMENSION


@pytest.mark.parametrize("dimension", [1, 8, 64, 384])
def test_dimension_is_configurable(dimension):
    assert FakeEmbeddingProvider(dimension=dimension).dimension == dimension


@pytest.mark.parametrize("dimension", [0, -1])
def test_rejects_a_non_positive_dimension(dimension):
    with pytest.raises(EmbeddingError):
        FakeEmbeddingProvider(dimension=dimension)


# --- dimensions ------------------------------------------------------------


@pytest.mark.parametrize("dimension", [1, 8, 64, 384])
async def test_documents_have_the_declared_dimension(dimension):
    provider = FakeEmbeddingProvider(dimension=dimension)

    vectors = await provider.embed_documents(["one", "two", "three"])

    assert all(len(v) == dimension for v in vectors)


@pytest.mark.parametrize("dimension", [1, 8, 64, 384])
async def test_queries_have_the_declared_dimension(dimension):
    provider = FakeEmbeddingProvider(dimension=dimension)

    assert len(await provider.embed_query("a question")) == dimension


async def test_vectors_are_plain_floats():
    vector = await FakeEmbeddingProvider().embed_query("budget")

    assert all(isinstance(value, float) for value in vector)


# --- determinism -----------------------------------------------------------


async def test_same_text_gives_the_same_vector():
    provider = FakeEmbeddingProvider()

    assert await provider.embed_query("delay the release") == await provider.embed_query(
        "delay the release"
    )


async def test_two_provider_instances_agree():
    assert await FakeEmbeddingProvider().embed_query(
        "budget"
    ) == await FakeEmbeddingProvider().embed_query("budget")


async def test_document_and_query_embedding_agree_for_identical_text():
    provider = FakeEmbeddingProvider()

    documents = await provider.embed_documents(["shared text"])

    assert documents[0] == await provider.embed_query("shared text")


def test_vectors_are_stable_across_processes():
    """Guards against Python's process-randomised hash() creeping back in."""
    probe = (
        "import asyncio;"
        "from app.adapters.embeddings.fake import FakeEmbeddingProvider;"
        "print(asyncio.run(FakeEmbeddingProvider().embed_query('budget review')))"
    )
    first = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    second = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert first.stdout == second.stdout
    assert first.stdout.strip() != ""


# --- discrimination --------------------------------------------------------


async def test_different_text_gives_a_different_vector():
    provider = FakeEmbeddingProvider()

    assert await provider.embed_query("budget forecast") != await provider.embed_query(
        "launch timeline"
    )


async def test_unrelated_texts_score_lower_than_overlapping_texts():
    """Feature hashing means shared vocabulary yields genuine similarity."""
    provider = FakeEmbeddingProvider(dimension=256)

    anchor = await provider.embed_query("we need to delay the release")
    overlapping = await provider.embed_query("the release will delay")
    unrelated = await provider.embed_query("hiring budget headcount approval")

    assert cosine(anchor, overlapping) > cosine(anchor, unrelated)


async def test_identical_text_scores_one():
    provider = FakeEmbeddingProvider(dimension=256)

    vector = await provider.embed_query("migration script fails")

    assert cosine(vector, vector) == pytest.approx(1.0)


async def test_embedding_is_case_insensitive():
    provider = FakeEmbeddingProvider()

    assert await provider.embed_query("Budget Review") == await provider.embed_query(
        "budget review"
    )


# --- normalisation ---------------------------------------------------------


async def test_vectors_are_unit_length():
    provider = FakeEmbeddingProvider(dimension=128)

    vector = await provider.embed_query("some meeting dialogue about scope")

    assert math.sqrt(sum(v * v for v in vector)) == pytest.approx(1.0)


# --- empty and degenerate input -------------------------------------------


async def test_empty_document_list_returns_empty_list():
    assert await FakeEmbeddingProvider().embed_documents([]) == []


@pytest.mark.parametrize("text", ["", "   ", "!!! ...", "\n\t"])
async def test_text_without_tokens_yields_the_zero_vector(text):
    provider = FakeEmbeddingProvider(dimension=16)

    assert await provider.embed_query(text) == [0.0] * 16


async def test_empty_string_among_documents_is_embedded_not_skipped():
    provider = FakeEmbeddingProvider(dimension=16)

    vectors = await provider.embed_documents(["real text", "", "more text"])

    assert len(vectors) == 3
    assert vectors[1] == [0.0] * 16


# --- shared dimension guard ------------------------------------------------


def test_ensure_dimension_accepts_matching_vectors():
    ensure_dimension([[0.0, 1.0], [1.0, 0.0]], 2, provider="test")


def test_ensure_dimension_reports_the_offending_position():
    with pytest.raises(EmbeddingError) as exc_info:
        ensure_dimension([[0.0, 1.0], [1.0]], 2, provider="test")

    assert exc_info.value.details == {
        "expected_dimension": 2,
        "actual_dimension": 1,
        "position": 1,
    }
