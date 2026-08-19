"""Exact in-memory vector store.

A meeting transcript produces on the order of a hundred chunks, so an exact
brute-force cosine search is both faster than building an ANN index and exactly
reproducible -- which is what makes retrieval tests able to assert rankings
rather than approximate overlap. Persistence and an approximate index are
documented as production enhancements rather than built here.
"""

from dataclasses import dataclass

import numpy as np

from app.adapters.vectorstore.base import VectorStoreError
from app.domain.models import Chunk, ScoredChunk


@dataclass(frozen=True)
class _Entry:
    """One stored chunk and its unit-length vector."""

    chunk: Chunk
    vector: np.ndarray


class InMemoryVectorStore:
    """Exact cosine similarity over chunks held in process memory.

    Storage is ``meeting_id -> chunk_id -> entry``. The nesting gives meeting
    scoping and per-chunk replacement directly, and because a dict preserves
    insertion order -- including when an existing key is reassigned -- the
    iteration order is stable, which is what breaks ties deterministically.
    """

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise VectorStoreError(
                "dimension must be greater than zero.", details={"dimension": dimension}
            )
        self._dimension = dimension
        self._meetings: dict[str, dict[str, _Entry]] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                "chunks and vectors must be the same length.",
                details={"chunks": len(chunks), "vectors": len(vectors)},
            )

        for chunk, raw_vector in zip(chunks, vectors, strict=True):
            vector = self._as_unit_vector(raw_vector, context=chunk.id)
            meeting = self._meetings.setdefault(chunk.meeting_id, {})
            # Assigning an existing key replaces the entry and keeps its
            # original position, so re-ingesting a transcript neither
            # duplicates chunks nor reshuffles tie-break order.
            meeting[chunk.id] = _Entry(chunk=chunk, vector=vector)

    async def search(self, vector: list[float], *, meeting_id: str, k: int) -> list[ScoredChunk]:
        if k <= 0:
            raise VectorStoreError("k must be greater than zero.", details={"k": k})

        query = self._as_unit_vector(vector, context="query")

        entries = list(self._meetings.get(meeting_id, {}).values())
        if not entries:
            return []

        matrix = np.vstack([entry.vector for entry in entries])
        # Both sides are unit length (or zero), so the dot product is the
        # cosine similarity and a zero vector scores 0.0 instead of NaN.
        scores = np.clip(matrix @ query, -1.0, 1.0)

        # A stable sort over insertion order makes equal scores deterministic.
        order = np.argsort(-scores, kind="stable")[:k]

        return [
            ScoredChunk(chunk=entries[position].chunk, score=float(scores[position]))
            for position in order
        ]

    async def delete_meeting(self, meeting_id: str) -> None:
        self._meetings.pop(meeting_id, None)

    def _as_unit_vector(self, values: list[float], *, context: str) -> np.ndarray:
        """Validate the dimension and normalise, leaving zero vectors alone."""
        if len(values) != self._dimension:
            raise VectorStoreError(
                "Vector has the wrong number of dimensions.",
                details={
                    "expected_dimension": self._dimension,
                    "actual_dimension": len(values),
                    "context": context,
                },
            )

        vector = np.asarray(values, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm
