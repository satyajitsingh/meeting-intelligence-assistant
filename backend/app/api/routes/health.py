"""Health and readiness endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__
from app.api.deps import SettingsDep

router = APIRouter(tags=["health"])


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
