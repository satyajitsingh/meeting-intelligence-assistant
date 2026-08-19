"""Deterministic, speaker-turn-aware chunking.

Chunks are the retrieval unit; utterances remain the citation unit. A chunk
therefore never splits an utterance, and always records the exact utterance IDs
it was built from so that an answer can cite a single speaker turn rather than
a whole passage.

The strategy is intentionally plain: accumulate consecutive utterances until
the rendered text would exceed a character target, then start the next chunk one
utterance earlier so adjacent chunks overlap by a single turn. That overlap
matters for meetings specifically -- a proposal and the "agreed" that answers it
must not land in different chunks.

No semantic segmentation, no tokenizer and no LLM: the transcript is already
segmented by speaker turn, which is structure available for free.
"""

from app.domain.errors import ChunkingError, EmptyTranscriptError
from app.domain.models import (
    UTTERANCE_SEPARATOR,
    Chunk,
    Transcript,
    Utterance,
    render_utterance,
)

DEFAULT_TARGET_CHARS = 700


def chunk_transcript(
    transcript: Transcript, *, target_chars: int = DEFAULT_TARGET_CHARS
) -> list[Chunk]:
    """Split a transcript into overlapping, speaker-aligned chunks.

    Args:
        transcript: The parsed transcript to split.
        target_chars: Soft budget for a chunk's rendered text. A single
            utterance longer than this becomes its own chunk and is allowed to
            exceed it, because splitting an utterance would destroy the
            citation unit.

    Returns:
        Chunks in transcript order. Consecutive chunks overlap by exactly one
        utterance wherever that overlap would carry new content.

    Raises:
        ChunkingError: ``target_chars`` is not positive.
        EmptyTranscriptError: the transcript has no utterances.
    """
    if target_chars <= 0:
        raise ChunkingError(
            "target_chars must be greater than zero.",
            details={"target_chars": target_chars},
        )

    utterances = transcript.utterances
    if not utterances:
        raise EmptyTranscriptError("Cannot chunk a transcript with no utterances.")

    chunks: list[Chunk] = []
    cursor = 0  # Index of the first utterance not yet covered by any chunk.

    while cursor < len(utterances):
        # Every chunk after the first reaches one utterance back for overlap.
        start = cursor - 1 if chunks and cursor > 0 else cursor
        end = _fill(utterances, start, target_chars)

        # The overlap utterance filled the chunk on its own, so this chunk would
        # repeat the previous one without adding anything. Drop the overlap.
        if end <= cursor:
            start = cursor
            end = _fill(utterances, start, target_chars)

        chunks.append(Chunk.from_utterances(len(chunks), utterances[start:end]))
        cursor = end

    return chunks


def _fill(utterances: list[Utterance], start: int, target_chars: int) -> int:
    """Return the exclusive end index of the chunk beginning at ``start``.

    Always consumes at least one utterance, which is what guarantees progress
    when a single utterance is longer than the target.
    """
    length = len(render_utterance(utterances[start]))
    end = start + 1

    while end < len(utterances):
        addition = len(UTTERANCE_SEPARATOR) + len(render_utterance(utterances[end]))
        if length + addition > target_chars:
            break
        length += addition
        end += 1

    return end
