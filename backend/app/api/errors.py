"""Application error hierarchy and FastAPI exception handlers.

Domain and service layers raise :class:`AppError` subclasses; they never import
FastAPI or construct HTTP responses. This module is the single place where an
application error is translated into a wire format, so every failure the API
returns has the same shape.
"""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette import status
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger

logger = get_logger(__name__)

_HTTP_ERROR_CODES: dict[int, str] = {
    status.HTTP_404_NOT_FOUND: "not_found",
    status.HTTP_405_METHOD_NOT_ALLOWED: "method_not_allowed",
}


class ErrorResponse(BaseModel):
    """Uniform error body returned by every failing endpoint."""

    error: str = Field(description="Stable, machine-readable error code.")
    message: str = Field(description="Human-readable explanation.")
    details: dict[str, Any] | None = Field(
        default=None, description="Optional structured context about the failure."
    )


class AppError(Exception):
    """Base class for all expected application errors."""

    code: str = "internal_error"
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_response(self) -> ErrorResponse:
        return ErrorResponse(error=self.code, message=self.message, details=self.details)


class NotFoundError(AppError):
    """A requested resource does not exist."""

    code = "not_found"
    status_code = status.HTTP_404_NOT_FOUND


class ValidationError(AppError):
    """Caller-supplied input was rejected by the application."""

    code = "validation_error"
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Render an :class:`AppError` using its own status code."""
    logger.warning(
        "request.app_error",
        error_code=exc.code,
        message=exc.message,
        details=exc.details,
        path=request.url.path,
    )
    return JSONResponse(status_code=exc.status_code, content=exc.to_response().model_dump())


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Render FastAPI's request-validation failures in the same shape."""
    logger.warning("request.invalid", path=request.url.path, errors=exc.errors())
    body = ErrorResponse(
        error="validation_error",
        message="Request validation failed.",
        details={"errors": exc.errors()},
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=body.model_dump(mode="json"),
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Render framework-raised HTTP errors (404, 405, ...) in the same shape."""
    body = ErrorResponse(
        error=_HTTP_ERROR_CODES.get(exc.status_code, "http_error"),
        message=str(exc.detail),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=body.model_dump(),
        headers=getattr(exc, "headers", None),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log unexpected failures with a stack trace and return a generic body."""
    logger.exception("request.unhandled_error", path=request.url.path)
    body = ErrorResponse(
        error="internal_error",
        message="An unexpected error occurred.",
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=body.model_dump(),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every handler in this module to ``app``."""
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)
