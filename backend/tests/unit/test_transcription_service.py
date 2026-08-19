"""TranscriptionService: upload validation and provider delegation."""

import pytest

from app.adapters.stt.base import SpeechToTextProviderError
from app.adapters.stt.fake import FakeSpeechToTextProvider
from app.domain.models import TranscriptionResult
from app.services.transcription import AudioValidationError, TranscriptionService

pytestmark = pytest.mark.anyio

AUDIO = b"fake-audio-bytes"
MAX_BYTES = 1024


def build(provider: FakeSpeechToTextProvider | None = None, max_upload_bytes: int = MAX_BYTES):
    provider = provider or FakeSpeechToTextProvider()
    service = TranscriptionService(provider=provider, max_upload_bytes=max_upload_bytes)
    return service, provider


async def transcribe(service, **overrides):
    args = {"audio": AUDIO, "filename": "meeting.m4a", "content_type": "audio/x-m4a"}
    args.update(overrides)
    return await service.transcribe(**args)


# --- delegation ------------------------------------------------------------


async def test_returns_the_provider_result():
    configured = TranscriptionResult(text="Transcribed.", language="en", duration_seconds=9.0)
    service, _ = build(FakeSpeechToTextProvider(configured))

    assert await transcribe(service) == configured


async def test_the_provider_receives_the_exact_bytes():
    service, provider = build()

    await transcribe(service, audio=b"\x00\x01\x02exact")

    assert provider.last_call is not None
    assert provider.last_call.audio == b"\x00\x01\x02exact"


async def test_the_provider_receives_the_filename():
    service, provider = build()

    await transcribe(service, filename="standup.wav", content_type="audio/wav")

    assert provider.last_call is not None
    assert provider.last_call.filename == "standup.wav"


async def test_the_provider_receives_the_content_type():
    service, provider = build()

    await transcribe(service, filename="clip.webm", content_type="audio/webm")

    assert provider.last_call is not None
    assert provider.last_call.content_type == "audio/webm"


async def test_the_filename_is_trimmed_before_use():
    service, provider = build()

    await transcribe(service, filename="  meeting.m4a  ")

    assert provider.last_call is not None
    assert provider.last_call.filename == "meeting.m4a"


async def test_the_provider_is_called_once():
    service, provider = build()

    await transcribe(service)

    assert provider.call_count == 1


# --- accepted formats ------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("meeting.mp3", "audio/mpeg"),
        ("meeting.m4a", "audio/x-m4a"),
        ("meeting.m4a", "audio/mp4"),
        ("meeting.wav", "audio/wav"),
        ("meeting.webm", "audio/webm"),
        ("meeting.ogg", "audio/ogg"),
        ("meeting.mp4", "video/mp4"),
        ("meeting.flac", "audio/flac"),
    ],
)
async def test_accepts_supported_audio(filename, content_type):
    service, _ = build()

    assert await transcribe(service, filename=filename, content_type=content_type)


@pytest.mark.parametrize("content_type", [None, "", "application/octet-stream"])
async def test_accepts_a_weak_content_type_when_the_extension_is_audio(content_type):
    """Browsers routinely send octet-stream for recorded or dragged-in audio."""
    service, _ = build()

    assert await transcribe(service, filename="meeting.m4a", content_type=content_type)


async def test_extension_matching_is_case_insensitive():
    service, _ = build()

    assert await transcribe(service, filename="MEETING.M4A", content_type=None)


# --- rejected uploads ------------------------------------------------------


@pytest.mark.parametrize("filename", [None, "", "   "])
async def test_rejects_a_missing_filename(filename):
    service, _ = build()

    with pytest.raises(AudioValidationError) as exc_info:
        await transcribe(service, filename=filename)

    assert exc_info.value.details["reason"] == "missing_filename"


async def test_rejects_an_empty_file():
    service, _ = build()

    with pytest.raises(AudioValidationError) as exc_info:
        await transcribe(service, audio=b"")

    assert exc_info.value.details["reason"] == "empty_file"


async def test_rejects_an_oversized_file():
    service, _ = build(max_upload_bytes=10)

    with pytest.raises(AudioValidationError) as exc_info:
        await transcribe(service, audio=b"x" * 11)

    details = exc_info.value.details
    assert details["reason"] == "file_too_large"
    assert details["size_bytes"] == 11
    assert details["max_bytes"] == 10


async def test_accepts_a_file_exactly_at_the_limit():
    service, _ = build(max_upload_bytes=10)

    assert await transcribe(service, audio=b"x" * 10)


@pytest.mark.parametrize("filename", ["notes.txt", "deck.pdf", "archive.zip", "noextension"])
async def test_rejects_an_unsupported_extension(filename):
    service, _ = build()

    with pytest.raises(AudioValidationError) as exc_info:
        await transcribe(service, filename=filename, content_type="audio/mpeg")

    assert exc_info.value.details["reason"] == "unsupported_extension"


async def test_rejects_an_unsupported_content_type_even_with_an_audio_extension():
    """An explicit non-audio type is a real signal and is not waived."""
    service, _ = build()

    with pytest.raises(AudioValidationError) as exc_info:
        await transcribe(service, filename="meeting.m4a", content_type="text/plain")

    assert exc_info.value.details["reason"] == "unsupported_content_type"


async def test_rejected_uploads_never_reach_the_provider():
    service, provider = build()

    with pytest.raises(AudioValidationError):
        await transcribe(service, filename="notes.txt")

    assert provider.call_count == 0


async def test_validation_errors_are_422():
    service, _ = build()

    with pytest.raises(AudioValidationError) as exc_info:
        await transcribe(service, audio=b"")

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "invalid_audio_upload"


# --- provider failure ------------------------------------------------------


async def test_provider_failure_propagates():
    service, _ = build(FakeSpeechToTextProvider(error=SpeechToTextProviderError("down")))

    with pytest.raises(SpeechToTextProviderError) as exc_info:
        await transcribe(service)

    assert exc_info.value.status_code == 502


# --- no side effects -------------------------------------------------------


async def test_transcribing_stores_nothing():
    """The service holds no repository or vector store to write to."""
    service, _ = build()

    await transcribe(service)

    assert not hasattr(service, "_repository")
    assert not hasattr(service, "_vector_store")


def test_the_upload_limit_is_exposed_for_the_route():
    service, _ = build(max_upload_bytes=4321)

    assert service.max_upload_bytes == 4321
