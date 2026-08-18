"""Translation of application errors into HTTP responses.

The error classes themselves live in :mod:`app.core.errors` so that the domain
and service layers can raise them without importing FastAPI. This module is the
single place where an error becomes a wire format, so every failure the API
returns has the same shape.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import AppError, ErrorResponse, NotFoundError, ValidationError
from app.core.logging import get_logger

# Re-exported so callers can import the error contract from one place.
__all__ = [
    "AppError",
    "ErrorResponse",
    "NotFoundError",
    "ValidationError",
    "register_exception_handlers",
]

logger = get_logger(__name__)

_HTTP_ERROR_CODES: dict[int, str] = {
    status.HTTP_404_NOT_FOUND: "not_found",
    status.HTTP_405_METHOD_NOT_ALLOWED: "method_not_allowed",
}


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
