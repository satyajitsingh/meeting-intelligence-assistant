"""OpenAI-backed speech to text.

The only real transcription provider in v1. A hosted API is used rather than a
local Whisper checkpoint so the project stays installable without a multi-
gigabyte model download.

Model note: only ``whisper-1`` supports ``verbose_json``, which is how the API
reports language and duration. The gpt-4o-transcribe family returns text alone
and rejects that format, so the request format is chosen from the configured
model rather than hard-coded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import SecretStr

from app.adapters.stt.base import SpeechToTextProviderError
from app.core.logging import get_logger
from app.domain.models import TranscriptionResult

if TYPE_CHECKING:  # pragma: no cover - avoids importing the SDK at module load
    from openai import AsyncOpenAI

logger = get_logger(__name__)

DEFAULT_MODEL = "whisper-1"
DEFAULT_TIMEOUT_SECONDS = 120.0

# Models that report language and duration. Everything else gets plain json.
VERBOSE_JSON_MODELS = frozenset({"whisper-1"})

FALLBACK_CONTENT_TYPE = "application/octet-stream"


class OpenAISpeechToTextProvider:
    """Transcribes audio with OpenAI's hosted speech-to-text models.

    The SDK client is created on first use, so importing the application,
    building the container and starting the server all make no network call and
    work with no API key present.
    """

    def __init__(
        self,
        *,
        api_key: SecretStr | None = None,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    async def transcribe(
        self,
        *,
        audio: bytes,
        filename: str,
        content_type: str | None,
    ) -> TranscriptionResult:
        client = self._ensure_client()

        # The SDK takes (name, bytes, content_type); the name matters because
        # the provider infers the container format from its extension.
        upload = (filename, audio, content_type or FALLBACK_CONTENT_TYPE)

        try:
            # Branched rather than parameterised: the two formats return
            # different SDK types, and only the verbose one carries language
            # and duration.
            if self._model in VERBOSE_JSON_MODELS:
                response: Any = await client.audio.transcriptions.create(
                    model=self._model,
                    file=upload,
                    response_format="verbose_json",
                    timeout=self._timeout_seconds,
                )
            else:
                response = await client.audio.transcriptions.create(
                    model=self._model,
                    file=upload,
                    response_format="json",
                    timeout=self._timeout_seconds,
                )
        except Exception as exc:
            # No exception detail in the message: SDK errors can echo request
            # headers, and the API key must never reach a log or error body.
            raise SpeechToTextProviderError(
                "The transcription provider is unavailable.",
                details={"model": self._model, "reason": type(exc).__name__},
            ) from exc

        return self._to_result(response)

    def _to_result(self, response: Any) -> TranscriptionResult:
        """Map the SDK response onto the domain model.

        Language and duration are read defensively: they are present only on
        verbose responses, and absent entirely on the gpt-4o-transcribe family.
        """
        text = getattr(response, "text", None)
        if not isinstance(text, str):
            raise SpeechToTextProviderError(
                "The transcription provider returned no text.",
                details={"model": self._model},
            )

        duration = getattr(response, "duration", None)

        return TranscriptionResult(
            text=text,
            language=getattr(response, "language", None),
            duration_seconds=float(duration) if duration is not None else None,
        )

    def _ensure_client(self) -> AsyncOpenAI:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - dependency declared
                raise SpeechToTextProviderError("The openai package is not installed.") from exc

            if self._api_key is None or not self._api_key.get_secret_value().strip():
                raise SpeechToTextProviderError(
                    "No OpenAI API key is configured. Set OPENAI_API_KEY to enable "
                    "audio transcription.",
                    details={"model": self._model},
                )

            self._client = AsyncOpenAI(api_key=self._api_key.get_secret_value())

        return self._client
