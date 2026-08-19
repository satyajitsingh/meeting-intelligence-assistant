"""Audio transcription endpoint.

Transcribes an uploaded recording and returns the text. It does not ingest:
the caller reviews and corrects the transcript, then submits it through
``POST /api/transcripts``, which remains the single path into the vector store.
"""

from typing import Annotated, Any

from fastapi import APIRouter, File, UploadFile, status

from app.api.deps import TranscriptionServiceDep
from app.api.schemas import TranscriptionResponse
from app.core.errors import ErrorResponse
from app.services.transcription import AudioValidationError

router = APIRouter(prefix="/transcriptions", tags=["transcriptions"])

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse},
    status.HTTP_502_BAD_GATEWAY: {"model": ErrorResponse},
}

READ_CHUNK_BYTES = 64 * 1024

AudioUpload = Annotated[UploadFile, File(description="Audio recording of a meeting.")]


async def read_upload(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload, refusing to buffer more than ``max_bytes``.

    Read in chunks rather than with a single ``read()`` so an oversized body is
    rejected as soon as it crosses the limit instead of after it has all been
    held in memory.
    """
    chunks: list[bytes] = []
    total = 0

    while chunk := await file.read(READ_CHUNK_BYTES):
        total += len(chunk)
        if total > max_bytes:
            raise AudioValidationError(
                "The uploaded file is too large.",
                details={
                    "reason": "file_too_large",
                    "filename": file.filename,
                    "max_bytes": max_bytes,
                },
            )
        chunks.append(chunk)

    return b"".join(chunks)


@router.post(
    "",
    response_model=TranscriptionResponse,
    status_code=status.HTTP_200_OK,
    summary="Transcribe an audio recording",
    responses=ERROR_RESPONSES,
)
async def transcribe_audio(
    service: TranscriptionServiceDep, file: AudioUpload
) -> TranscriptionResponse:
    """Transcribe an uploaded recording into editable transcript text.

    The result is **not** stored or indexed. Review it, correct it, then submit
    it to ``POST /api/transcripts`` to make it searchable.
    """
    audio = await read_upload(file, service.max_upload_bytes)

    result = await service.transcribe(
        audio=audio, filename=file.filename, content_type=file.content_type
    )

    return TranscriptionResponse.from_result(result=result, filename=(file.filename or "").strip())
