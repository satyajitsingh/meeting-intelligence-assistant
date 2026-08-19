"""Transcript ingestion.

Orchestration only: parse, chunk, embed, store. Every decision along the way
belongs to the component being called -- this service owns the *ordering*, and
the consistency guarantees that ordering buys.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.adapters.embeddings.base import EmbeddingProvider
from app.adapters.repository.base import TranscriptRepository
from app.adapters.vectorstore.base import VectorStore
from app.core.errors import AppError, NotFoundError
from app.core.logging import get_logger
from app.domain.chunker import DEFAULT_TARGET_CHARS, chunk_transcript
from app.domain.models import TranscriptSummary
from app.domain.parser import parse_transcript

logger = get_logger(__name__)


class IngestionConfigurationError(AppError):
    """The service was wired with incompatible components.

    A wiring fault rather than bad user input, so it keeps the default 500.
    """

    code = "ingestion_configuration_error"


class IngestionResult(BaseModel):
    """What ingestion produced, for the API layer to render."""

    model_config = ConfigDict(frozen=True)

    summary: TranscriptSummary
    chunk_count: int


class IngestionService:
    """Turns raw transcript text into stored, searchable chunks."""

    def __init__(
        self,
        *,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        repository: TranscriptRepository,
        target_chars: int = DEFAULT_TARGET_CHARS,
    ) -> None:
        if embeddings.dimension != vector_store.dimension:
            raise IngestionConfigurationError(
                "Embedding provider and vector store disagree on dimension.",
                details={
                    "embedding_dimension": embeddings.dimension,
                    "vector_store_dimension": vector_store.dimension,
                },
            )

        self._embeddings = embeddings
        self._vector_store = vector_store
        self._repository = repository
        self._target_chars = target_chars

    async def ingest(self, *, meeting_id: str, title: str, transcript_text: str) -> IngestionResult:
        """Parse, chunk, embed and store a transcript.

        Re-ingesting an existing ``meeting_id`` fully replaces the previous
        transcript and its chunks.

        Ordering *is* the consistency mechanism -- there is no rollback. The
        three steps that can fail without touching stored state run first, so
        existing data survives a malformed transcript or an embedding outage
        untouched. Stored state is only replaced once new vectors are in hand.

        Failure semantics, in the order the steps run:

        =========================  ===================  ====================
        Step that fails            Old vectors          Old transcript
        =========================  ===================  ====================
        parse / chunk              intact               intact
        embed                      intact               intact
        delete old vectors         intact               intact
        upsert new vectors         **cleared**          intact (not replaced)
        save transcript            replaced             replaced
        =========================  ===================  ====================

        A failed upsert therefore leaves the meeting with its previous
        transcript but no vectors. That is a known v1 limitation, accepted in
        preference to building rollback machinery: re-posting the transcript
        restores a consistent state.

        Raises:
            TranscriptParseError: the transcript is malformed.
            EmbeddingError: the provider failed to embed the chunks.
            VectorStoreError: the chunks could not be stored.
        """
        # 1-3: no side effects, so any failure here leaves stored state alone.
        transcript = parse_transcript(transcript_text, meeting_id=meeting_id, title=title)
        chunks = chunk_transcript(transcript, target_chars=self._target_chars)
        vectors = await self._embeddings.embed_documents([chunk.text for chunk in chunks])

        # 4: clear before upserting. Chunk IDs are positional, so upsert alone
        # would strand old chunks when the new transcript is shorter.
        await self._vector_store.delete_meeting(meeting_id)

        # 5: if this raises, step 6 never runs and the new transcript is not
        # saved -- the repository is never left pointing at absent vectors.
        await self._vector_store.upsert(chunks, vectors)

        # 6: last, so the repository entry only ever reflects stored vectors.
        await self._repository.save(transcript)

        logger.info(
            "transcript.ingested",
            meeting_id=meeting_id,
            utterance_count=len(transcript.utterances),
            chunk_count=len(chunks),
            speaker_count=len(transcript.speakers),
        )

        return IngestionResult(summary=transcript.summary(), chunk_count=len(chunks))

    async def delete(self, meeting_id: str) -> None:
        """Remove a transcript and every chunk derived from it.

        Raises:
            NotFoundError: the meeting was never ingested.
        """
        if await self._repository.get(meeting_id) is None:
            raise NotFoundError("Transcript not found.", details={"meeting_id": meeting_id})

        await self._purge(meeting_id)
        logger.info("transcript.deleted", meeting_id=meeting_id)

    async def _purge(self, meeting_id: str) -> None:
        await self._vector_store.delete_meeting(meeting_id)
        await self._repository.delete(meeting_id)
