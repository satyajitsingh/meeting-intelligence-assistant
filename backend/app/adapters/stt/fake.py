"""Deterministic offline speech-to-text provider for tests."""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.models import TranscriptionResult


@dataclass(frozen=True)
class TranscriptionCall:
    """One recorded invocation, for assertions."""

    audio: bytes
    filename: str
    content_type: str | None

    @property
    def audio_length(self) -> int:
        return len(self.audio)


DEFAULT_RESULT = TranscriptionResult(
    text="[00:00:00] Speaker: A fake transcription.",
    language="en",
    duration_seconds=12.5,
)


class FakeSpeechToTextProvider:
    """Returns a configured result and records what it was asked to transcribe."""

    def __init__(
        self,
        result: TranscriptionResult | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.result = result if result is not None else DEFAULT_RESULT
        self.error = error
        self.calls: list[TranscriptionCall] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_call(self) -> TranscriptionCall | None:
        return self.calls[-1] if self.calls else None

    async def transcribe(
        self,
        *,
        audio: bytes,
        filename: str,
        content_type: str | None,
    ) -> TranscriptionResult:
        self.calls.append(
            TranscriptionCall(audio=audio, filename=filename, content_type=content_type)
        )

        if self.error is not None:
            raise self.error

        return self.result
