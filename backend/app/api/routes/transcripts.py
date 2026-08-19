"""Transcript ingestion and lookup endpoints.

Routes translate between HTTP and the service layer and do nothing else: no
parsing, no chunking, no store access beyond a direct read for the two lookup
endpoints, which involve no orchestration.
"""

from typing import Any

from fastapi import APIRouter, Response, status

from app.api.deps import IngestionServiceDep, TranscriptRepositoryDep
from app.api.schemas import (
    IngestTranscriptRequest,
    IngestTranscriptResponse,
    TranscriptDetailResponse,
    TranscriptSummaryResponse,
)
from app.core.errors import ErrorResponse, NotFoundError

router = APIRouter(prefix="/transcripts", tags=["transcripts"])

NOT_FOUND_RESPONSE: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}
}


@router.post(
    "",
    response_model=IngestTranscriptResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a transcript",
    responses={status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse}},
)
async def ingest_transcript(
    payload: IngestTranscriptRequest, service: IngestionServiceDep
) -> IngestTranscriptResponse:
    """Parse, chunk, embed and store a transcript.

    Re-posting an existing ``meeting_id`` fully replaces the previous
    transcript and its chunks.
    """
    result = await service.ingest(
        meeting_id=payload.meeting_id,
        title=payload.title,
        transcript_text=payload.transcript,
    )
    return IngestTranscriptResponse.from_result(result.summary, result.chunk_count)


@router.get(
    "",
    response_model=list[TranscriptSummaryResponse],
    summary="List ingested transcripts",
)
async def list_transcripts(
    repository: TranscriptRepositoryDep,
) -> list[TranscriptSummaryResponse]:
    """Return a summary of every ingested meeting, in upload order."""
    summaries = await repository.list()
    return [TranscriptSummaryResponse.from_summary(summary) for summary in summaries]


@router.get(
    "/{meeting_id}",
    response_model=TranscriptDetailResponse,
    summary="Fetch one transcript",
    responses=NOT_FOUND_RESPONSE,
)
async def get_transcript(
    meeting_id: str, repository: TranscriptRepositoryDep
) -> TranscriptDetailResponse:
    """Return a transcript's metadata and its utterances."""
    transcript = await repository.get(meeting_id)
    if transcript is None:
        raise NotFoundError("Transcript not found.", details={"meeting_id": meeting_id})

    return TranscriptDetailResponse.from_transcript(transcript)


@router.delete(
    "/{meeting_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    # A 204 has no body, so suppress the default JSON content-type header.
    response_class=Response,
    summary="Delete a transcript",
    responses=NOT_FOUND_RESPONSE,
)
async def delete_transcript(meeting_id: str, service: IngestionServiceDep) -> None:
    """Remove a transcript and every chunk derived from it.

    Deleting an unknown meeting returns 404 rather than an idempotent 204, so a
    mistyped ``meeting_id`` surfaces instead of looking like a success.
    """
    await service.delete(meeting_id)
