"""Application error hierarchy.

These classes are deliberately free of any web-framework import so that the
domain and service layers can raise them. ``app/api/errors.py`` owns the
translation from these errors into HTTP responses.
"""

from http import HTTPStatus
from typing import Any

from pydantic import BaseModel, Field


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
    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_response(self) -> ErrorResponse:
        return ErrorResponse(error=self.code, message=self.message, details=self.details)


class NotFoundError(AppError):
    """A requested resource does not exist."""

    code = "not_found"
    status_code = HTTPStatus.NOT_FOUND


class ValidationError(AppError):
    """Caller-supplied input was rejected by the application."""

    code = "validation_error"
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY
