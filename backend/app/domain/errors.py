"""Domain-level errors raised while interpreting a transcript.

These subclass :class:`app.core.errors.ValidationError`, so the API layer maps
them to a 422 with the standard error body without any bespoke translation.
"""

from typing import Any

from app.core.errors import ValidationError


class TranscriptParseError(ValidationError):
    """A transcript could not be parsed.

    ``line_number`` is 1-based and refers to the offending line of the original
    text, so the caller can point a user straight at the problem.
    """

    code = "transcript_parse_error"

    def __init__(
        self,
        message: str,
        *,
        line_number: int | None = None,
        line: str | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if line_number is not None:
            details["line_number"] = line_number
        if line is not None:
            details["line"] = line

        if line_number is not None:
            message = f"Line {line_number}: {message}"

        super().__init__(message, details=details or None)
        self.line_number = line_number
        self.line = line


class InvalidTimestampError(TranscriptParseError):
    """A timestamp did not match the supported ``HH:MM:SS`` / ``MM:SS`` forms."""

    code = "invalid_timestamp"


class EmptyTranscriptError(TranscriptParseError):
    """The transcript contained no utterances."""

    code = "empty_transcript"
