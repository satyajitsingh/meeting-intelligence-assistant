"""FastAPI dependency providers.

Each provider reads from the process-wide container. Tests replace them through
``app.dependency_overrides``, which is how the API suite runs against fake
adapters without ever constructing the real embedding model.
"""

from typing import Annotated

from fastapi import Depends

from app.adapters.repository.base import TranscriptRepository
from app.adapters.vectorstore.base import VectorStore
from app.core.config import Settings, get_settings
from app.core.container import get_container
from app.services.generation import AnswerGenerationService
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.services.transcription import TranscriptionService


def get_ingestion_service() -> IngestionService:
    return get_container().ingestion_service


def get_answer_generation_service() -> AnswerGenerationService:
    return get_container().answer_service


def get_retrieval_service() -> RetrievalService:
    return get_container().retrieval_service


def get_transcription_service() -> TranscriptionService:
    return get_container().transcription_service


def get_transcript_repository() -> TranscriptRepository:
    return get_container().repository


def get_vector_store() -> VectorStore:
    return get_container().vector_store


SettingsDep = Annotated[Settings, Depends(get_settings)]
IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
RetrievalServiceDep = Annotated[RetrievalService, Depends(get_retrieval_service)]
AnswerGenerationServiceDep = Annotated[
    AnswerGenerationService, Depends(get_answer_generation_service)
]
TranscriptionServiceDep = Annotated[TranscriptionService, Depends(get_transcription_service)]
TranscriptRepositoryDep = Annotated[TranscriptRepository, Depends(get_transcript_repository)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
