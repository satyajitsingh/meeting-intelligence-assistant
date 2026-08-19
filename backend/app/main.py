"""FastAPI application factory.

``create_app`` is the composition point for the HTTP layer: it configures
logging, installs middleware and exception handlers, and mounts routers. Later
phases add the ingestion and query routers here; nothing else about this file
needs to change.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.errors import register_exception_handlers
from app.api.middleware import RequestContextMiddleware
from app.api.routes import answers, health, retrieval, transcripts
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured FastAPI application.

    Accepting ``settings`` lets tests construct an app with an explicit
    configuration instead of mutating the environment.
    """
    settings = settings or get_settings()
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "app.startup",
            service=settings.app_name,
            version=__version__,
            environment=settings.environment,
        )
        yield
        logger.info("app.shutdown", service=settings.app_name)

    app = FastAPI(
        title="Meeting Intelligence API",
        description="Grounded question answering over meeting transcripts.",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(transcripts.router, prefix=settings.api_prefix)
    app.include_router(retrieval.router, prefix=settings.api_prefix)
    app.include_router(answers.router, prefix=settings.api_prefix)

    return app


app = create_app()
