"""Speech-to-text provider port.

A provider turns audio bytes into text. It performs no validation of its own --
size and format are checked before it is called -- and no vendor type crosses
this boundary.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Protocol, runtime_checkable

from app.core.errors import AppError
from app.domain.models import TranscriptionResult


class SpeechToTextProviderError(AppError):
    """The transcription provider failed or returned something unusable.

    Modelled as a bad gateway: the upload was fine, an upstream dependency was
    not.
    """

    code = "speech_to_text_provider_error"
    status_code = HTTPStatus.BAD_GATEWAY


@runtime_checkable
class SpeechToTextProvider(Protocol):
    """Transcribes recorded audio."""

    async def transcribe(
        self,
        *,
        audio: bytes,
        filename: str,
        content_type: str | None,
    ) -> TranscriptionResult:
        """Transcribe ``audio``.

        ``filename`` is passed through to the provider because most infer the
        container format from the extension.

        Raises:
            SpeechToTextProviderError: the provider failed or timed out.
        """
        ...
