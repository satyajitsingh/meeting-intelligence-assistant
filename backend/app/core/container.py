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
from app.adapters.llm.anthropic import AnthropicLLMProvider
from app.adapters.llm.base import LLMProvider
from app.adapters.repository.base import TranscriptRepository
from app.adapters.repository.memory import InMemoryTranscriptRepository
from app.adapters.stt.base import SpeechToTextProvider
from app.adapters.stt.openai import OpenAISpeechToTextProvider
from app.adapters.vectorstore.base import VectorStore
from app.adapters.vectorstore.memory import InMemoryVectorStore
from app.core.config import Settings, get_settings
from app.services.generation import AnswerGenerationService
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.services.transcription import TranscriptionService


@dataclass(frozen=True)
class Container:
    """The application's long-lived collaborators."""

    embeddings: EmbeddingProvider
    vector_store: VectorStore
    repository: TranscriptRepository
    llm: LLMProvider
    speech_to_text: SpeechToTextProvider
    ingestion_service: IngestionService
    retrieval_service: RetrievalService
    answer_service: AnswerGenerationService
    transcription_service: TranscriptionService


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

    # Constructed here but not connected: the SDK client is created on the
    # first generate_answer call, so startup makes no request and needs no key.
    llm = AnthropicLLMProvider(
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )

    answer_service = AnswerGenerationService(
        retrieval=retrieval_service,
        repository=repository,
        llm=llm,
    )

    # Also lazy: the OpenAI client is created on the first transcription, so
    # startup needs no key and makes no request.
    speech_to_text = OpenAISpeechToTextProvider(
        api_key=settings.openai_api_key,
        model=settings.openai_transcription_model,
    )

    transcription_service = TranscriptionService(
        provider=speech_to_text,
        max_upload_bytes=settings.max_audio_upload_bytes,
    )

    return Container(
        embeddings=embeddings,
        vector_store=vector_store,
        repository=repository,
        llm=llm,
        speech_to_text=speech_to_text,
        ingestion_service=ingestion_service,
        retrieval_service=retrieval_service,
        answer_service=answer_service,
        transcription_service=transcription_service,
    )


@lru_cache(maxsize=1)
def get_container() -> Container:
    """Return the process-wide container, building it on first use.

    Built on first request rather than at import time, which keeps the
    embedding model's import and download off the startup path.
    """
    return build_container(get_settings())
