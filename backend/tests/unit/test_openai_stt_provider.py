"""OpenAISpeechToTextProvider, with the SDK boundary stubbed.

No network: a stub client is injected through the constructor seam, so these
tests assert what the provider sends and how it maps what comes back.
"""

from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app.adapters.stt.base import SpeechToTextProvider, SpeechToTextProviderError
from app.adapters.stt.openai import DEFAULT_MODEL, OpenAISpeechToTextProvider

pytestmark = pytest.mark.anyio

AUDIO = b"fake-m4a-bytes"


class StubTranscriptions:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class StubClient:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.audio = SimpleNamespace(
            transcriptions=StubTranscriptions(response=response, error=error)
        )


def verbose_response(text="Transcribed speech.", language="en", duration=42.5):
    return SimpleNamespace(text=text, language=language, duration=duration)


def build(response=None, error: Exception | None = None, **kwargs):
    client = StubClient(response=response, error=error)
    provider = OpenAISpeechToTextProvider(client=client, **kwargs)
    return provider, client


async def transcribe(provider, **overrides):
    args = {"audio": AUDIO, "filename": "meeting.m4a", "content_type": "audio/x-m4a"}
    args.update(overrides)
    return await provider.transcribe(**args)


def calls(client):
    return client.audio.transcriptions.calls


# --- protocol --------------------------------------------------------------


def test_satisfies_the_provider_protocol():
    provider, _ = build()

    assert isinstance(provider, SpeechToTextProvider)


# --- request shape ---------------------------------------------------------


def test_the_default_model_reports_language_and_duration():
    assert DEFAULT_MODEL == "whisper-1"


async def test_uses_the_default_model_when_unconfigured():
    provider, client = build(response=verbose_response())

    await transcribe(provider)

    assert calls(client)[0]["model"] == DEFAULT_MODEL
    assert provider.model == DEFAULT_MODEL


async def test_the_model_is_configurable():
    provider, client = build(response=verbose_response(), model="gpt-4o-transcribe")

    await transcribe(provider)

    assert calls(client)[0]["model"] == "gpt-4o-transcribe"


async def test_preserves_the_uploaded_filename():
    provider, client = build(response=verbose_response())

    await transcribe(provider, filename="standup.wav", content_type="audio/wav")

    name, _, _ = calls(client)[0]["file"]
    assert name == "standup.wav"


async def test_passes_the_exact_byte_content():
    provider, client = build(response=verbose_response())

    await transcribe(provider, audio=b"\x00\x01\x02exact")

    _, content, _ = calls(client)[0]["file"]
    assert content == b"\x00\x01\x02exact"


async def test_passes_the_content_type():
    provider, client = build(response=verbose_response())

    await transcribe(provider, content_type="audio/webm")

    _, _, content_type = calls(client)[0]["file"]
    assert content_type == "audio/webm"


async def test_substitutes_a_content_type_when_the_browser_sent_none():
    provider, client = build(response=verbose_response())

    await transcribe(provider, content_type=None)

    _, _, content_type = calls(client)[0]["file"]
    assert content_type == "application/octet-stream"


async def test_requests_verbose_json_for_whisper():
    """Only whisper-1 supports it, and it is how language and duration arrive."""
    provider, client = build(response=verbose_response(), model="whisper-1")

    await transcribe(provider)

    assert calls(client)[0]["response_format"] == "verbose_json"


async def test_requests_plain_json_for_models_that_reject_verbose():
    provider, client = build(response=SimpleNamespace(text="Text."), model="gpt-4o-transcribe")

    await transcribe(provider)

    assert calls(client)[0]["response_format"] == "json"


async def test_applies_the_configured_timeout():
    provider, client = build(response=verbose_response(), timeout_seconds=45.0)

    await transcribe(provider)

    assert calls(client)[0]["timeout"] == 45.0


# --- response mapping ------------------------------------------------------


async def test_parses_the_transcribed_text():
    provider, _ = build(response=verbose_response(text="The budget is unchanged."))

    assert (await transcribe(provider)).text == "The budget is unchanged."


async def test_parses_the_language_when_available():
    provider, _ = build(response=verbose_response(language="es"))

    assert (await transcribe(provider)).language == "es"


async def test_parses_the_duration_when_available():
    provider, _ = build(response=verbose_response(duration=123.4))

    assert (await transcribe(provider)).duration_seconds == pytest.approx(123.4)


async def test_duration_is_coerced_to_float():
    provider, _ = build(response=verbose_response(duration=90))

    result = await transcribe(provider)

    assert isinstance(result.duration_seconds, float)
    assert result.duration_seconds == 90.0


async def test_language_and_duration_are_none_when_the_model_omits_them():
    provider, _ = build(response=SimpleNamespace(text="Only text."))

    result = await transcribe(provider)

    assert result.text == "Only text."
    assert result.language is None
    assert result.duration_seconds is None


async def test_a_response_without_text_raises_provider_error():
    provider, _ = build(response=SimpleNamespace(language="en"))

    with pytest.raises(SpeechToTextProviderError) as exc_info:
        await transcribe(provider)

    assert "no text" in exc_info.value.message


# --- failure handling ------------------------------------------------------


async def test_sdk_exception_becomes_a_provider_error():
    provider, _ = build(error=RuntimeError("connection reset"))

    with pytest.raises(SpeechToTextProviderError) as exc_info:
        await transcribe(provider)

    assert exc_info.value.code == "speech_to_text_provider_error"
    assert exc_info.value.status_code == 502


async def test_timeout_becomes_a_provider_error():
    provider, _ = build(error=TimeoutError("timed out"))

    with pytest.raises(SpeechToTextProviderError) as exc_info:
        await transcribe(provider)

    assert exc_info.value.details["reason"] == "TimeoutError"


async def test_the_original_exception_is_chained():
    original = RuntimeError("connection reset")
    provider, _ = build(error=original)

    with pytest.raises(SpeechToTextProviderError) as exc_info:
        await transcribe(provider)

    assert exc_info.value.__cause__ is original


# --- secret handling -------------------------------------------------------


SECRET = "sk-openai-super-secret-value"


async def test_api_key_never_appears_in_a_provider_error():
    provider, _ = build(
        error=RuntimeError(f"401 unauthorized for key {SECRET}"),
        api_key=SecretStr(SECRET),
    )

    with pytest.raises(SpeechToTextProviderError) as exc_info:
        await transcribe(provider)

    error = exc_info.value
    assert SECRET not in error.message
    assert SECRET not in str(error.details)
    assert SECRET not in str(error.to_response().model_dump())


def test_the_api_key_is_held_as_a_secret():
    provider = OpenAISpeechToTextProvider(api_key=SecretStr(SECRET))

    assert SECRET not in repr(provider._api_key)
    assert str(provider._api_key) == "**********"


# --- lazy construction -----------------------------------------------------


def test_no_client_is_created_at_construction():
    provider = OpenAISpeechToTextProvider(api_key=SecretStr(SECRET))

    assert provider._client is None


async def test_a_missing_api_key_fails_only_when_transcribing():
    provider = OpenAISpeechToTextProvider(api_key=None)

    with pytest.raises(SpeechToTextProviderError) as exc_info:
        await transcribe(provider)

    assert "OPENAI_API_KEY" in exc_info.value.message


def test_constructing_without_a_key_does_not_raise():
    """Every other endpoint must work on a deployment with no OpenAI key."""
    assert OpenAISpeechToTextProvider(api_key=None) is not None
