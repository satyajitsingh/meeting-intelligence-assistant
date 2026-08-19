"""Embedding provider port.

Embeddings are always generated from ``Chunk.text`` -- the speaker-labelled
rendering produced by the chunker -- so that speaker names participate in
similarity rather than living only in metadata.
"""

from typing import Protocol, runtime_checkable

from app.core.errors import AppError


class EmbeddingError(AppError):
    """An embedding provider failed to produce vectors."""

    code = "embedding_error"


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into dense vectors.

    Implementations must be consistent: every vector returned by either method
    has exactly ``dimension`` components.
    """

    @property
    def dimension(self) -> int:
        """Number of components in every vector this provider returns."""
        ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunk texts for storage.

        An empty ``texts`` list returns an empty list without calling out to
        any model.
        """
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single user question for search."""
        ...


def ensure_dimension(vectors: list[list[float]], dimension: int, *, provider: str) -> None:
    """Validate that every vector has the provider's declared dimension."""
    for position, vector in enumerate(vectors):
        if len(vector) != dimension:
            raise EmbeddingError(
                f"{provider} returned a vector of the wrong size.",
                details={
                    "expected_dimension": dimension,
                    "actual_dimension": len(vector),
                    "position": position,
                },
            )
