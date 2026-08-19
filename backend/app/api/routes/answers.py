"""Grounded answer endpoint.

Returns an answer with the utterance IDs the model cited. Chunks and vectors
stay out of this response -- ``POST /api/retrieval`` remains available for
inspecting retrieval directly.
"""

from typing import Any

from fastapi import APIRouter, status

from app.api.deps import AnswerGenerationServiceDep
from app.api.schemas import AnswerRequest, AnswerResponse
from app.core.errors import ErrorResponse

router = APIRouter(prefix="/answers", tags=["answers"])

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
    status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse},
    status.HTTP_502_BAD_GATEWAY: {"model": ErrorResponse},
}


@router.post(
    "",
    response_model=AnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Answer a question about one meeting",
    responses=ERROR_RESPONSES,
)
async def answer_question(
    payload: AnswerRequest, service: AnswerGenerationServiceDep
) -> AnswerResponse:
    """Answer a question using only evidence retrieved from that meeting.

    Citations are validated against the transcript before they are returned, so
    speaker, timestamp and quote are always source data rather than generated
    text. When the meeting holds no relevant evidence, the response says so and
    sets ``insufficient_evidence`` rather than guessing.
    """
    validated = await service.answer(
        meeting_id=payload.meeting_id, question=payload.question, k=payload.k
    )

    return AnswerResponse.from_validated(
        meeting_id=payload.meeting_id, question=payload.question, validated=validated
    )
