"""Deterministic evidence preparation for answer generation.

Retrieval returns chunks; the model is shown utterances. This module performs
that conversion, and nothing else -- it does not validate citations or resolve
quotes for display, which is the next phase's job.
"""

from __future__ import annotations

from app.domain.models import ScoredChunk, Transcript, Utterance


def select_evidence_utterances(
    transcript: Transcript, scored_chunks: list[ScoredChunk]
) -> list[Utterance]:
    """Resolve retrieved chunks to their source utterances.

    Adjacent chunks overlap by one utterance by design, so the same utterance
    can arrive from several chunks. Each appears exactly once here, ordered by
    position in the transcript rather than by retrieval rank: meeting dialogue
    only reads correctly in chronological order, where a decision follows the
    proposal it answers.
    """
    by_id = {utterance.id: utterance for utterance in transcript.utterances}

    selected: dict[str, Utterance] = {}
    for scored in scored_chunks:
        for utterance_id in scored.chunk.utterance_ids:
            utterance = by_id.get(utterance_id)
            if utterance is not None:
                selected[utterance_id] = utterance

    return sorted(selected.values(), key=lambda utterance: utterance.index)


def render_evidence_block(utterance: Utterance) -> str:
    """Render one evidence unit as an ID-tagged, attributed quote."""
    header = f"[{utterance.id} | {utterance.display_timestamp} | {utterance.speaker}]"
    return f"{header}\n{utterance.text}"


def build_evidence_context(utterances: list[Utterance]) -> str:
    """Render evidence for the prompt.

    Retrieval scores are deliberately absent: they are ranking metadata, not
    meeting evidence, and showing them invites the model to reason about a
    confidence it has no basis to interpret.
    """
    return "\n\n".join(render_evidence_block(utterance) for utterance in utterances)
