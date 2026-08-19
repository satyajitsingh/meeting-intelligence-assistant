"""Shared pytest fixtures.

Every fixture here is offline. The API fixtures override the real dependency
providers with fake-backed collaborators, so no test constructs the embedding
model or reaches the network.
"""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.adapters.embeddings.fake import FakeEmbeddingProvider
from app.adapters.llm.fake import FakeLLMProvider
from app.adapters.repository.memory import InMemoryTranscriptRepository
from app.adapters.vectorstore.memory import InMemoryVectorStore
from app.api.deps import (
    get_answer_generation_service,
    get_ingestion_service,
    get_retrieval_service,
    get_transcript_repository,
    get_vector_store,
)
from app.core.config import Settings, get_settings
from app.main import create_app
from app.services.generation import AnswerGenerationService
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService

EMBEDDING_DIMENSION = 128


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on asyncio only; no trio in this project."""
    return "asyncio"


@pytest.fixture
def settings() -> Settings:
    """Deterministic settings for tests: console logs, no .env file influence."""
    return Settings(environment="test", log_json=False, log_level="WARNING")


@pytest.fixture
def embedding_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(dimension=EMBEDDING_DIMENSION)


@pytest.fixture
def vector_store(embedding_provider: FakeEmbeddingProvider) -> InMemoryVectorStore:
    return InMemoryVectorStore(dimension=embedding_provider.dimension)


@pytest.fixture
def transcript_repository() -> InMemoryTranscriptRepository:
    return InMemoryTranscriptRepository()


@pytest.fixture
def ingestion_service(
    embedding_provider: FakeEmbeddingProvider,
    vector_store: InMemoryVectorStore,
    transcript_repository: InMemoryTranscriptRepository,
) -> IngestionService:
    return IngestionService(
        embeddings=embedding_provider,
        vector_store=vector_store,
        repository=transcript_repository,
        target_chars=700,
    )


@pytest.fixture
def retrieval_service(
    embedding_provider: FakeEmbeddingProvider,
    vector_store: InMemoryVectorStore,
    transcript_repository: InMemoryTranscriptRepository,
) -> RetrievalService:
    """Shares the same store and repository the ingestion fixture writes to."""
    return RetrievalService(
        embeddings=embedding_provider,
        vector_store=vector_store,
        repository=transcript_repository,
    )


@pytest.fixture
def llm_provider() -> FakeLLMProvider:
    return FakeLLMProvider()


@pytest.fixture
def answer_service(
    retrieval_service: RetrievalService,
    transcript_repository: InMemoryTranscriptRepository,
    llm_provider: FakeLLMProvider,
) -> AnswerGenerationService:
    return AnswerGenerationService(
        retrieval=retrieval_service,
        repository=transcript_repository,
        llm=llm_provider,
    )


@pytest.fixture
def app(
    settings: Settings,
    ingestion_service: IngestionService,
    retrieval_service: RetrievalService,
    answer_service: AnswerGenerationService,
    transcript_repository: InMemoryTranscriptRepository,
    vector_store: InMemoryVectorStore,
) -> FastAPI:
    application = create_app(settings)
    application.dependency_overrides[get_settings] = lambda: settings
    application.dependency_overrides[get_ingestion_service] = lambda: ingestion_service
    application.dependency_overrides[get_retrieval_service] = lambda: retrieval_service
    application.dependency_overrides[get_answer_generation_service] = lambda: answer_service
    application.dependency_overrides[get_transcript_repository] = lambda: transcript_repository
    application.dependency_overrides[get_vector_store] = lambda: vector_store
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_transcript() -> str:
    return (
        "[00:00:12] Sarah: We need to delay the release because the migration is unfinished.\n"
        "[00:00:31] John: Agreed, the migration script still fails on legacy accounts.\n"
        "[00:00:52] Amir: What does the delay mean for the marketing budget we approved?\n"
        "[00:01:14] Sarah: The budget is unchanged, only the launch date moves.\n"
    )
