"""FakeSpeechToTextProvider: deterministic, recording, offline."""

import pytest

from app.adapters.stt.base import SpeechToTextProvider, SpeechToTextProviderError
from app.adapters.stt.fake import DEFAULT_RESULT, FakeSpeechToTextProvider
from app.domain.models import TranscriptionResult

pytestmark = pytest.mark.anyio

AUDIO = b"fake-m4a-bytes"


def call_args(**overrides):
    args = {"audio": AUDIO, "filename": "meeting.m4a", "content_type": "audio/x-m4a"}
    args.update(overrides)
    return args


def test_satisfies_the_provider_protocol():
    assert isinstance(FakeSpeechToTextProvider(), SpeechToTextProvider)


async def test_returns_the_default_result_when_unconfigured():
    assert await FakeSpeechToTextProvider().transcribe(**call_args()) == DEFAULT_RESULT


async def test_returns_the_configured_result():
    configured = TranscriptionResult(
        text="[00:00:01] Sarah: Hello.", language="en", duration_seconds=3.2
    )

    assert await FakeSpeechToTextProvider(configured).transcribe(**call_args()) is configured


async def test_is_deterministic_across_calls():
    provider = FakeSpeechToTextProvider()

    assert await provider.transcribe(**call_args()) == await provider.transcribe(**call_args())


async def test_records_the_filename():
    provider = FakeSpeechToTextProvider()

    await provider.transcribe(**call_args(filename="standup.wav"))

    assert provider.last_call is not None
    assert provider.last_call.filename == "standup.wav"


async def test_records_the_content_type():
    provider = FakeSpeechToTextProvider()

    await provider.transcribe(**call_args(content_type="audio/wav"))

    assert provider.last_call is not None
    assert provider.last_call.content_type == "audio/wav"


async def test_records_a_missing_content_type():
    provider = FakeSpeechToTextProvider()

    await provider.transcribe(**call_args(content_type=None))

    assert provider.last_call is not None
    assert provider.last_call.content_type is None


async def test_records_the_audio_bytes_and_length():
    provider = FakeSpeechToTextProvider()

    await provider.transcribe(**call_args(audio=b"0123456789"))

    assert provider.last_call is not None
    assert provider.last_call.audio == b"0123456789"
    assert provider.last_call.audio_length == 10


async def test_counts_calls():
    provider = FakeSpeechToTextProvider()

    assert provider.call_count == 0
    await provider.transcribe(**call_args())
    await provider.transcribe(**call_args())

    assert provider.call_count == 2


def test_last_call_is_none_before_any_call():
    assert FakeSpeechToTextProvider().last_call is None


async def test_can_simulate_provider_failure():
    provider = FakeSpeechToTextProvider(error=SpeechToTextProviderError("unavailable"))

    with pytest.raises(SpeechToTextProviderError):
        await provider.transcribe(**call_args())


async def test_a_failed_call_is_still_recorded():
    provider = FakeSpeechToTextProvider(error=SpeechToTextProviderError("boom"))

    with pytest.raises(SpeechToTextProviderError):
        await provider.transcribe(**call_args())

    assert provider.call_count == 1


async def test_can_return_a_result_without_language_or_duration():
    """Mirrors the gpt-4o-transcribe family, which reports neither."""
    provider = FakeSpeechToTextProvider(TranscriptionResult(text="Only text."))

    result = await provider.transcribe(**call_args())

    assert result.language is None
    assert result.duration_seconds is None


def test_the_fake_never_imports_a_client_library():
    import subprocess
    import sys

    probe = (
        "import sys, app.adapters.stt.fake;print('openai' in sys.modules or 'httpx' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"
