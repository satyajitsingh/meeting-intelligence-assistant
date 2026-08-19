"""Retrieval endpoint.

Retrieval-only by design: it returns ranked chunks and never calls a language
model. Its purpose is to make dense-retrieval behaviour directly observable so
it can be evaluated before answer generation is layered on.
"""

from typing import Any

from fastapi import APIRouter, status

from app.api.deps import RetrievalServiceDep
from app.api.schemas import RetrievalResponse, RetrievalResultResponse, RetrieveRequest
from app.core.errors import ErrorResponse

router = APIRouter(prefix="/retrieval", tags=["retrieval"])

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
    status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse},
}


@router.post(
    "",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve relevant chunks from one meeting",
    responses=ERROR_RESPONSES,
)
async def retrieve(payload: RetrieveRequest, service: RetrievalServiceDep) -> RetrievalResponse:
    """Return the chunks of one meeting most similar to a question.

    A known meeting with no matching chunks returns ``200`` with an empty
    ``results`` list -- only an unknown meeting is a 404.
    """
    results = await service.retrieve(
        meeting_id=payload.meeting_id, query=payload.query, k=payload.k
    )

    return RetrievalResponse(
        meeting_id=payload.meeting_id,
        query=payload.query,
        results=[RetrievalResultResponse.from_scored_chunk(scored) for scored in results],
    )
