"""Composition root.

The single place where concrete adapters are chosen and wired. Everything else
depends on Protocols, so swapping the embedding provider or vector store is a
change here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.adapters.embeddings.base import EmbeddingProvider
from app.adapters.embeddings.local import FastEmbedProvider
from app.adapters.repository.base import TranscriptRepository
from app.adapters.repository.memory import InMemoryTranscriptRepository
from app.adapters.vectorstore.base import VectorStore
from app.adapters.vectorstore.memory import InMemoryVectorStore
from app.core.config import Settings, get_settings
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService


@dataclass(frozen=True)
class Container:
    """The application's long-lived collaborators."""

    embeddings: EmbeddingProvider
    vector_store: VectorStore
    repository: TranscriptRepository
    ingestion_service: IngestionService
    retrieval_service: RetrievalService


def build_container(settings: Settings) -> Container:
    """Construct the object graph for one process."""
    embeddings = FastEmbedProvider(model_name=settings.embedding_model)

    # Derived, never configured separately, so the two cannot drift apart.
    vector_store = InMemoryVectorStore(dimension=embeddings.dimension)
    repository = InMemoryTranscriptRepository()

    ingestion_service = IngestionService(
        embeddings=embeddings,
        vector_store=vector_store,
        repository=repository,
        target_chars=settings.chunk_target_chars,
    )

    # Shares the very same store ingestion writes to -- a second instance
    # here would search an empty index and silently return nothing.
    retrieval_service = RetrievalService(
        embeddings=embeddings,
        vector_store=vector_store,
        repository=repository,
    )

    return Container(
        embeddings=embeddings,
        vector_store=vector_store,
        repository=repository,
        ingestion_service=ingestion_service,
        retrieval_service=retrieval_service,
    )


@lru_cache(maxsize=1)
def get_container() -> Container:
    """Return the process-wide container, building it on first use.

    Built on first request rather than at import time, which keeps the
    embedding model's import and download off the startup path.
    """
    return build_container(get_settings())
