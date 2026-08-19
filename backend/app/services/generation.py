"""Grounded answer generation.

Orchestration only: retrieve, prepare evidence, ask the model, then check what
came back. The model contributes the answer prose and a list of utterance IDs;
every citation is validated and resolved against the stored transcript before
it reaches a caller.
"""

from __future__ import annotations

from app.adapters.llm.base import LLMProvider
from app.adapters.repository.base import TranscriptRepository
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.domain.models import ValidatedAnswer
from app.services.citations import CitationResolver
from app.services.context import build_evidence_context, select_evidence_utterances
from app.services.retrieval import DEFAULT_K, RetrievalService

logger = get_logger(__name__)

INSUFFICIENT_EVIDENCE_ANSWER = "I don't have enough evidence in this meeting to answer that."


class AnswerGenerationService:
    """Answers a question about one meeting, grounded in retrieved evidence."""

    def __init__(
        self,
        *,
        retrieval: RetrievalService,
        repository: TranscriptRepository,
        llm: LLMProvider,
        resolver: CitationResolver | None = None,
    ) -> None:
        self._retrieval = retrieval
        self._repository = repository
        self._llm = llm
        # Stateless and dependency-free, so it is built here rather than wired
        # through the composition root. Injectable for tests.
        self._resolver = resolver or CitationResolver()

    async def answer(
        self, *, meeting_id: str, question: str, k: int = DEFAULT_K
    ) -> ValidatedAnswer:
        """Retrieve evidence for ``question`` and generate a grounded answer.

        Retrieval runs exactly once; there is no reranking, no second attempt
        and no agent loop. If retrieval finds nothing, the model is not called
        at all -- there is no evidence to ground an answer in, and asking
        anyway would invite invention.

        Raises:
            InvalidRetrievalRequestError: blank meeting, blank question, k <= 0.
            NotFoundError: the meeting was never ingested.
            LLMProviderError: the provider failed or returned bad output.
        """
        scored_chunks = await self._retrieval.retrieve(meeting_id=meeting_id, query=question, k=k)

        if not scored_chunks:
            logger.info("answer.no_evidence", meeting_id=meeting_id.strip(), k=k)
            return ValidatedAnswer(
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                citations=[],
                insufficient_evidence=True,
            )

        meeting_id = meeting_id.strip()
        transcript = await self._repository.get(meeting_id)
        if transcript is None:  # pragma: no cover - retrieval already checked
            raise NotFoundError("Transcript not found.", details={"meeting_id": meeting_id})

        utterances = select_evidence_utterances(transcript, scored_chunks)
        context = build_evidence_context(utterances)
        allowed_utterance_ids = [utterance.id for utterance in utterances]

        generated = await self._llm.generate_answer(
            question=question.strip(),
            context=context,
            allowed_utterance_ids=allowed_utterance_ids,
        )

        # The model is not trusted to supply evidence: its IDs are checked
        # against the evidence it was actually shown, and speaker, timestamp
        # and quote are read back from the transcript.
        validated = self._resolver.resolve(
            generated=generated,
            transcript=transcript,
            allowed_utterance_ids=allowed_utterance_ids,
        )

        logger.info(
            "answer.generated",
            meeting_id=meeting_id,
            k=k,
            chunk_count=len(scored_chunks),
            evidence_count=len(utterances),
            cited_count=len(generated.citations),
            valid_citation_count=len(validated.citations),
            insufficient_evidence=validated.insufficient_evidence,
        )

        return validated
