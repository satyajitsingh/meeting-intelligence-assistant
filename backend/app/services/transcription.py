"""Audio transcription.

Validates an upload and delegates to a speech-to-text provider. Deliberately
does *not* ingest the result: transcription is lossy in exactly the places
citations depend on -- names, numbers, decisions -- so the text is returned for
a human to review, and ``POST /api/transcripts`` remains the single path into
the vector store.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from app.adapters.stt.base import SpeechToTextProvider
from app.core.errors import ValidationError
from app.core.logging import get_logger
from app.domain.models import TranscriptionResult

logger = get_logger(__name__)

DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024

SUPPORTED_EXTENSIONS = frozenset(
    {".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".oga", ".ogg", ".wav", ".webm"}
)

SUPPORTED_CONTENT_TYPES = frozenset(
    {
        "audio/flac",
        "audio/m4a",
        "audio/mp3",
        "audio/mp4",
        "audio/mpeg",
        "audio/ogg",
        "audio/wav",
        "audio/wave",
        "audio/webm",
        "audio/x-m4a",
        "audio/x-wav",
        "video/mp4",
        "video/webm",
    }
)

# Browsers frequently send one of these for recorded or dragged-in audio. They
# carry no information, so the filename extension decides instead.
UNINFORMATIVE_CONTENT_TYPES = frozenset({"", "application/octet-stream", "binary/octet-stream"})


class AudioValidationError(ValidationError):
    """An uploaded file was rejected before transcription was attempted."""

    code = "invalid_audio_upload"


class TranscriptionService:
    """Turns an uploaded audio file into reviewable transcript text."""

    def __init__(
        self,
        *,
        provider: SpeechToTextProvider,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    ) -> None:
        self._provider = provider
        self._max_upload_bytes = max_upload_bytes

    @property
    def max_upload_bytes(self) -> int:
        return self._max_upload_bytes

    async def transcribe(
        self,
        *,
        audio: bytes,
        filename: str | None,
        content_type: str | None,
    ) -> TranscriptionResult:
        """Validate an upload and transcribe it.

        Nothing is stored or indexed. The caller reviews the text and submits
        it through the transcript ingestion endpoint when satisfied.

        Raises:
            AudioValidationError: missing name, empty, oversized or wrong type.
            SpeechToTextProviderError: the provider failed.
        """
        name = self.validate(filename=filename, content_type=content_type, size_bytes=len(audio))

        result = await self._provider.transcribe(
            audio=audio, filename=name, content_type=content_type
        )

        logger.info(
            "audio.transcribed",
            filename=name,
            size_bytes=len(audio),
            language=result.language,
            duration_seconds=result.duration_seconds,
            text_length=len(result.text),
        )

        return result

    def validate(self, *, filename: str | None, content_type: str | None, size_bytes: int) -> str:
        """Check an upload is usable and return its cleaned filename."""
        name = (filename or "").strip()
        if not name:
            raise AudioValidationError(
                "A filename is required.", details={"reason": "missing_filename"}
            )

        if size_bytes == 0:
            raise AudioValidationError(
                "The uploaded file is empty.",
                details={"reason": "empty_file", "filename": name},
            )

        if size_bytes > self._max_upload_bytes:
            raise AudioValidationError(
                "The uploaded file is too large.",
                details={
                    "reason": "file_too_large",
                    "filename": name,
                    "size_bytes": size_bytes,
                    "max_bytes": self._max_upload_bytes,
                },
            )

        extension = PurePosixPath(name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise AudioValidationError(
                "Unsupported audio file type.",
                details={
                    "reason": "unsupported_extension",
                    "filename": name,
                    "extension": extension,
                    "supported": sorted(SUPPORTED_EXTENSIONS),
                },
            )

        # Only trust a content type that actually says something: browsers send
        # octet-stream for recorded audio often enough that rejecting on it
        # would break the common case.
        declared = (content_type or "").strip().lower()
        if declared not in UNINFORMATIVE_CONTENT_TYPES and declared not in (
            SUPPORTED_CONTENT_TYPES
        ):
            raise AudioValidationError(
                "Unsupported audio content type.",
                details={
                    "reason": "unsupported_content_type",
                    "filename": name,
                    "content_type": declared,
                },
            )

        return name
