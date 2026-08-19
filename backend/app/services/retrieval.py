"""Meeting-scoped dense retrieval.

Deliberately raw: the service embeds the question, searches one meeting, and
returns what the vector store ranked, unchanged. No reranking, deduplication,
neighbour expansion or query rewriting -- so evaluation can measure dense
retrieval on its own before anything is layered on top of it.
"""

from __future__ import annotations

from app.adapters.embeddings.base import EmbeddingProvider
from app.adapters.repository.base import TranscriptRepository
from app.adapters.vectorstore.base import VectorStore
from app.core.errors import AppError, NotFoundError, ValidationError
from app.core.logging import get_logger
from app.domain.models import ScoredChunk

logger = get_logger(__name__)

DEFAULT_K = 5


class RetrievalConfigurationError(AppError):
    """The service was wired with incompatible components.

    A wiring fault rather than bad user input, so it keeps the default 500.
    """

    code = "retrieval_configuration_error"


class InvalidRetrievalRequestError(ValidationError):
    """The caller supplied an unusable meeting, query or result count."""

    code = "invalid_retrieval_request"


class RetrievalService:
    """Finds the chunks of one meeting most similar to a question."""

    def __init__(
        self,
        *,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        repository: TranscriptRepository,
    ) -> None:
        if embeddings.dimension != vector_store.dimension:
            raise RetrievalConfigurationError(
                "Embedding provider and vector store disagree on dimension.",
                details={
                    "embedding_dimension": embeddings.dimension,
                    "vector_store_dimension": vector_store.dimension,
                },
            )

        self._embeddings = embeddings
        self._vector_store = vector_store
        self._repository = repository

    async def retrieve(
        self, *, meeting_id: str, query: str, k: int = DEFAULT_K
    ) -> list[ScoredChunk]:
        """Return the ``k`` chunks of ``meeting_id`` closest to ``query``.

        Search is scoped to exactly one meeting; there is no cross-meeting
        retrieval. A known meeting with no matching chunks returns an empty
        list rather than an error -- absence of evidence is a valid result.

        Raises:
            InvalidRetrievalRequestError: blank meeting, blank query, or k <= 0.
            NotFoundError: the meeting was never ingested.
        """
        meeting_id = meeting_id.strip()
        query = query.strip()

        # Validate before doing any work: an unusable request should never
        # cost an embedding call.
        if not meeting_id:
            raise InvalidRetrievalRequestError("meeting_id must not be empty.")

        if not query:
            raise InvalidRetrievalRequestError("query must not be empty.")

        if k <= 0:
            raise InvalidRetrievalRequestError("k must be greater than zero.", details={"k": k})

        # Checked before embedding, so an unknown meeting costs nothing.
        if await self._repository.get(meeting_id) is None:
            raise NotFoundError("Transcript not found.", details={"meeting_id": meeting_id})

        vector = await self._embeddings.embed_query(query)
        results = await self._vector_store.search(vector, meeting_id=meeting_id, k=k)

        logger.info(
            "retrieval.completed",
            meeting_id=meeting_id,
            k=k,
            result_count=len(results),
            top_score=results[0].score if results else None,
        )

        return results
