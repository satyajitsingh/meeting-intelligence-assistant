"""Deterministic citation validation and evidence resolution.

The model is not trusted to supply evidence. It returns utterance identifiers;
everything a reader sees -- speaker, timestamp, verbatim quote -- is read back
from the stored transcript here. A citation survives only if it exists, belongs
to the requested meeting, and was among the evidence actually shown to the
model.

Nothing in this module calls a language model, and nothing scores or reranks.
"""

from __future__ import annotations

from enum import StrEnum

from app.core.logging import get_logger
from app.domain.models import (
    GeneratedAnswer,
    ResolvedCitation,
    Transcript,
    Utterance,
    ValidatedAnswer,
)

logger = get_logger(__name__)

# Utterance IDs are '<meeting_id>:u<index>' (see make_utterance_id).
_ID_SEPARATOR = ":u"


class InvalidCitationReason(StrEnum):
    """Why a model-supplied citation was discarded."""

    WRONG_MEETING = "wrong_meeting"
    NOT_FOUND = "not_found"
    NOT_ALLOWED = "not_allowed"


class CitationResolver:
    """Turns model-supplied IDs into evidence read from the transcript.

    Stateless and dependency-free, so it is constructed where it is used rather
    than wired through the composition root.
    """

    def resolve(
        self,
        *,
        generated: GeneratedAnswer,
        transcript: Transcript,
        allowed_utterance_ids: list[str],
    ) -> ValidatedAnswer:
        """Validate every citation and resolve it against ``transcript``.

        Invalid citations are discarded and logged rather than raised: one bad
        identifier should not destroy an otherwise usable answer. Duplicates
        collapse to their first occurrence.

        The answer text is never altered.
        """
        if generated.insufficient_evidence:
            # Nothing to resolve: an answer that reports insufficient evidence
            # must not carry citations, whatever the model returned.
            return ValidatedAnswer(
                answer=generated.answer, citations=[], insufficient_evidence=True
            )

        allowed = set(allowed_utterance_ids)
        by_id = {utterance.id: utterance for utterance in transcript.utterances}

        resolved: list[ResolvedCitation] = []
        processed: set[str] = set()

        for citation in generated.citations:
            utterance_id = citation.utterance_id

            # Deduplicate before validating, so a repeated bad ID is judged --
            # and logged -- exactly once.
            if utterance_id in processed:
                continue
            processed.add(utterance_id)

            reason = self._rejection_reason(
                utterance_id=utterance_id,
                transcript=transcript,
                by_id=by_id,
                allowed=allowed,
            )
            if reason is not None:
                # Identifiers and reasons only: these logs must never carry
                # transcript text.
                logger.warning(
                    "citation.invalid",
                    meeting_id=transcript.meeting_id,
                    utterance_id=utterance_id,
                    reason=str(reason),
                )
                continue

            resolved.append(ResolvedCitation.from_utterance(by_id[utterance_id]))

        if generated.citations and not resolved:
            # The answer claims to be supported but nothing survived checking.
            # Surfaced for observability; insufficient_evidence is deliberately
            # left as the model set it (see module docs in the phase notes).
            logger.warning(
                "answer.no_valid_citations",
                meeting_id=transcript.meeting_id,
                cited_count=len(generated.citations),
            )

        return ValidatedAnswer(
            answer=generated.answer,
            citations=resolved,
            insufficient_evidence=False,
        )

    @staticmethod
    def _rejection_reason(
        *,
        utterance_id: str,
        transcript: Transcript,
        by_id: dict[str, Utterance],
        allowed: set[str],
    ) -> InvalidCitationReason | None:
        """Return why a citation is invalid, or ``None`` when it is usable.

        Ordered most-specific first: an identifier that names a different
        meeting is a more informative failure than a generic miss.
        """
        meeting_id, separator, _ = utterance_id.rpartition(_ID_SEPARATOR)
        if separator and meeting_id != transcript.meeting_id:
            return InvalidCitationReason.WRONG_MEETING

        utterance = by_id.get(utterance_id)
        if utterance is None:
            return InvalidCitationReason.NOT_FOUND

        # Defensive: a transcript should never hold another meeting's
        # utterances, but the evidence contract is worth enforcing directly.
        if utterance.meeting_id != transcript.meeting_id:
            return InvalidCitationReason.WRONG_MEETING

        if utterance_id not in allowed:
            return InvalidCitationReason.NOT_ALLOWED

        return None
