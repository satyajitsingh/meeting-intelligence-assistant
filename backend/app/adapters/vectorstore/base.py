"""Vector store port.

Chunks are the retrieval unit, so the store holds whole :class:`Chunk` objects
rather than bare text: a search result carries its source utterance IDs with it,
which is what lets an answer cite a speaker turn instead of a passage.
"""

from typing import Protocol, runtime_checkable

from app.core.errors import ValidationError
from app.domain.models import Chunk, ScoredChunk


class VectorStoreError(ValidationError):
    """A vector store rejected an operation."""

    code = "vector_store_error"


@runtime_checkable
class VectorStore(Protocol):
    """Stores chunk vectors and answers nearest-neighbour queries.

    Every operation is scoped to a single meeting. Cross-meeting search is out
    of scope for v1 by design.
    """

    @property
    def dimension(self) -> int:
        """Vector size this store accepts."""
        ...

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Store chunks with their vectors, replacing any with the same chunk ID.

        ``chunks`` and ``vectors`` are positional pairs and must be the same
        length. Each chunk is filed under its own ``meeting_id``.
        """
        ...

    async def search(self, vector: list[float], *, meeting_id: str, k: int) -> list[ScoredChunk]:
        """Return the ``k`` closest chunks within one meeting, best first.

        Returns an empty list when the meeting holds no chunks.
        """
        ...

    async def delete_meeting(self, meeting_id: str) -> None:
        """Remove every chunk belonging to one meeting. A no-op if unknown."""
        ...
