"""FastEmbedProvider against the real model.

Excluded from the default run because the first execution downloads model
weights. Run explicitly with::

    pytest -m integration
"""

import math

import pytest

from app.adapters.embeddings.base import EmbeddingProvider
from app.adapters.embeddings.local import DEFAULT_DIMENSION, DEFAULT_MODEL, FastEmbedProvider

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )


async def test_embeds_documents_with_the_declared_dimension():
    provider = FastEmbedProvider()

    vectors = await provider.embed_documents(
        ["Sarah: We need to delay the release.", "John: Agreed."]
    )

    assert len(vectors) == 2
    assert all(len(v) == provider.dimension for v in vectors)


async def test_embeds_a_query():
    provider = FastEmbedProvider()

    vector = await provider.embed_query("What did Sarah say about the release?")

    assert len(vector) == provider.dimension


async def test_semantically_related_text_scores_higher():
    provider = FastEmbedProvider()

    documents = await provider.embed_documents(
        [
            "Sarah: We need to delay the release by two weeks.",
            "Amir: The office coffee machine is broken again.",
        ]
    )
    query = await provider.embed_query("Why is the launch being pushed back?")

    assert cosine(query, documents[0]) > cosine(query, documents[1])


async def test_embedding_is_deterministic():
    provider = FastEmbedProvider()

    first = await provider.embed_query("budget review")
    second = await provider.embed_query("budget review")

    assert first == pytest.approx(second)


async def test_empty_document_list_short_circuits_without_loading_the_model():
    provider = FastEmbedProvider()

    assert await provider.embed_documents([]) == []
    assert provider._model is None


def test_reports_the_expected_model_and_dimension():
    provider = FastEmbedProvider()

    assert provider.model_name == DEFAULT_MODEL
    assert provider.dimension == DEFAULT_DIMENSION


def test_satisfies_the_embedding_provider_protocol():
    assert isinstance(FastEmbedProvider(), EmbeddingProvider)
