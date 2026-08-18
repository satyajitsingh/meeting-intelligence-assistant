"""Health and readiness endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app import __version__
from app.core.config import Settings, get_settings

router = APIRouter(tags=["health"])

# Moves to app/api/deps.py once more than one route needs it.
SettingsDep = Annotated[Settings, Depends(get_settings)]


class HealthResponse(BaseModel):
    """Liveness payload used by local development and container health checks."""

    status: str
    service: str
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
async def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=__version__,
        environment=settings.environment,
    )
