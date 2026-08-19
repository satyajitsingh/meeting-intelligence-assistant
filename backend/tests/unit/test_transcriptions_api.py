"""POST /api/transcriptions.

Runs against a FakeSpeechToTextProvider through dependency overrides: no OpenAI
client is constructed and no request leaves the process.
"""

import pytest

from app.adapters.stt.base import SpeechToTextProviderError
from app.adapters.stt.fake import FakeSpeechToTextProvider
from app.api.middleware import REQUEST_ID_HEADER
from app.domain.models import TranscriptionResult
from app.services.transcription import TranscriptionService

AUDIO = b"fake-audio-payload"


def upload(client, filename="meeting.m4a", content_type="audio/x-m4a", content=AUDIO):
    return client.post("/api/transcriptions", files={"file": (filename, content, content_type)})


# --- happy path ------------------------------------------------------------


def test_returns_200_for_a_valid_upload(client):
    assert upload(client).status_code == 200


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("meeting.mp3", "audio/mpeg"),
        ("meeting.m4a", "audio/x-m4a"),
        ("meeting.wav", "audio/wav"),
        ("meeting.webm", "audio/webm"),
        ("meeting.ogg", "audio/ogg"),
        ("meeting.mp4", "video/mp4"),
    ],
)
def test_accepts_common_meeting_audio_formats(client, filename, content_type):
    assert upload(client, filename=filename, content_type=content_type).status_code == 200


def test_response_has_the_documented_shape(client):
    body = upload(client).json()

    assert set(body) == {"text", "language", "duration_seconds", "filename"}


def test_response_contains_the_transcribed_text(client, speech_to_text_provider):
    speech_to_text_provider.result = TranscriptionResult(
        text="[00:00:12] Sarah: We need to delay the release.",
        language="en",
        duration_seconds=74.0,
    )

    assert upload(client).json()["text"] == "[00:00:12] Sarah: We need to delay the release."


def test_response_contains_the_language(client, speech_to_text_provider):
    speech_to_text_provider.result = TranscriptionResult(text="Hola.", language="es")

    assert upload(client).json()["language"] == "es"


def test_response_contains_the_duration(client, speech_to_text_provider):
    speech_to_text_provider.result = TranscriptionResult(text="Text.", duration_seconds=123.4)

    assert upload(client).json()["duration_seconds"] == pytest.approx(123.4)


def test_response_contains_the_filename(client):
    assert (
        upload(client, filename="standup.wav", content_type="audio/wav").json()["filename"]
        == "standup.wav"
    )


def test_language_and_duration_may_be_null(client, speech_to_text_provider):
    speech_to_text_provider.result = TranscriptionResult(text="Only text.")

    body = upload(client).json()

    assert body["language"] is None
    assert body["duration_seconds"] is None


def test_the_provider_receives_the_uploaded_bytes(client, speech_to_text_provider):
    upload(client, content=b"\x00\x01\x02distinctive")

    call = speech_to_text_provider.last_call
    assert call is not None
    assert call.audio == b"\x00\x01\x02distinctive"
    assert call.filename == "meeting.m4a"


def test_a_weak_browser_content_type_is_accepted(client):
    response = upload(client, filename="recording.webm", content_type="application/octet-stream")

    assert response.status_code == 200


# --- no ingestion ----------------------------------------------------------


def test_transcribing_does_not_ingest_the_transcript(client, transcript_repository):
    import anyio

    upload(client)

    assert anyio.run(transcript_repository.list) == []
    assert client.get("/api/transcripts").json() == []


def test_transcribing_does_not_populate_the_vector_store(client, vector_store):
    import anyio

    upload(client)

    query = [0.0] * vector_store.dimension
    query[0] = 1.0
    assert anyio.run(lambda: vector_store.search(query, meeting_id="meeting.m4a", k=10)) == []


def test_the_response_carries_no_meeting_id(client):
    """Nothing was stored, so there is nothing to identify."""
    assert "meeting_id" not in upload(client).json()


# --- rejected uploads ------------------------------------------------------


@pytest.mark.parametrize("filename", ["notes.txt", "deck.pdf", "archive.zip"])
def test_unsupported_extension_returns_422(client, filename):
    response = upload(client, filename=filename, content_type="audio/mpeg")

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "invalid_audio_upload"
    assert body["details"]["reason"] == "unsupported_extension"


def test_unsupported_mime_type_returns_422(client):
    response = upload(client, filename="meeting.m4a", content_type="text/plain")

    assert response.status_code == 422
    assert response.json()["details"]["reason"] == "unsupported_content_type"


def test_empty_file_returns_422(client):
    response = upload(client, content=b"")

    assert response.status_code == 422
    assert response.json()["details"]["reason"] == "empty_file"


def test_too_large_file_returns_422(client, speech_to_text_provider, app, settings):
    from app.api.deps import get_transcription_service

    tiny = TranscriptionService(provider=speech_to_text_provider, max_upload_bytes=16)
    app.dependency_overrides[get_transcription_service] = lambda: tiny

    response = upload(client, content=b"x" * 64)

    assert response.status_code == 422
    body = response.json()
    assert body["details"]["reason"] == "file_too_large"
    assert body["details"]["max_bytes"] == 16


def test_an_oversized_upload_never_reaches_the_provider(client, speech_to_text_provider, app):
    from app.api.deps import get_transcription_service

    tiny = TranscriptionService(provider=speech_to_text_provider, max_upload_bytes=16)
    app.dependency_overrides[get_transcription_service] = lambda: tiny

    upload(client, content=b"x" * 64)

    assert speech_to_text_provider.call_count == 0


def test_a_missing_file_field_returns_422(client):
    response = client.post("/api/transcriptions")

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_rejected_uploads_never_reach_the_provider(client, speech_to_text_provider):
    upload(client, filename="notes.txt")

    assert speech_to_text_provider.call_count == 0


def test_error_bodies_carry_the_uniform_keys(client):
    body = upload(client, content=b"").json()

    assert set(body) == {"error", "message", "details"}


# --- provider failure ------------------------------------------------------


def test_provider_failure_returns_502(client, app, settings):
    from app.api.deps import get_transcription_service

    failing = TranscriptionService(
        provider=FakeSpeechToTextProvider(error=SpeechToTextProviderError("provider unavailable")),
        max_upload_bytes=settings.max_audio_upload_bytes,
    )
    app.dependency_overrides[get_transcription_service] = lambda: failing

    response = upload(client)

    assert response.status_code == 502
    assert response.json()["error"] == "speech_to_text_provider_error"


# --- cross-cutting ---------------------------------------------------------


def test_request_id_header_is_present(client):
    assert upload(client).headers[REQUEST_ID_HEADER]


def test_supplied_request_id_is_echoed(client):
    response = client.post(
        "/api/transcriptions",
        files={"file": ("meeting.m4a", AUDIO, "audio/x-m4a")},
        headers={REQUEST_ID_HEADER: "trace-transcribe"},
    )

    assert response.headers[REQUEST_ID_HEADER] == "trace-transcribe"


def test_api_tests_never_construct_the_real_provider(app):
    from app.api.deps import get_transcription_service

    assert get_transcription_service in app.dependency_overrides


def test_no_openai_request_is_made(client, speech_to_text_provider):
    upload(client)

    assert isinstance(speech_to_text_provider, FakeSpeechToTextProvider)
    assert speech_to_text_provider.call_count == 1


def test_endpoint_is_registered_under_the_api_prefix(client):
    assert "/api/transcriptions" in client.get("/openapi.json").json()["paths"]


def test_existing_endpoints_are_unaffected(client):
    paths = client.get("/openapi.json").json()["paths"]

    for path in ["/api/transcripts", "/api/retrieval", "/api/answers", "/health"]:
        assert path in paths
