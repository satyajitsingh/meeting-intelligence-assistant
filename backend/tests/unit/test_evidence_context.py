"""Deterministic evidence preparation from retrieved chunks."""

import pytest

from app.domain.chunker import chunk_transcript
from app.domain.models import ScoredChunk
from app.domain.parser import parse_transcript
from app.services.context import (
    build_evidence_context,
    render_evidence_block,
    select_evidence_utterances,
)

SAMPLE = (
    "[00:00:12] Sarah: We need to delay the release because migration is unfinished.\n"
    "[00:00:31] John: Agreed. The migration script still fails.\n"
    "[00:00:52] Amir: What happens to the marketing budget?\n"
    "[01:01:14] Sarah: The budget is unchanged.\n"
    "[01:01:38] John: I will update the launch plan by Friday.\n"
)


@pytest.fixture
def transcript():
    return parse_transcript(SAMPLE, meeting_id="m1", title="Release planning")


@pytest.fixture
def chunks(transcript):
    # Yields [u0,u1] [u1,u2,u3] [u3,u4] -- adjacent chunks share an utterance.
    return chunk_transcript(transcript, target_chars=150)


def scored(chunks, *indexes, score: float = 0.5):
    return [ScoredChunk(chunk=chunks[i], score=score) for i in indexes]


# --- selection -------------------------------------------------------------


def test_the_fixture_produces_overlapping_chunks(chunks):
    """Pin the layout the overlap tests below depend on."""
    assert [c.utterance_ids for c in chunks] == [
        ["m1:u0", "m1:u1"],
        ["m1:u1", "m1:u2", "m1:u3"],
        ["m1:u3", "m1:u4"],
    ]


def test_resolves_chunk_utterance_ids_to_transcript_utterances(transcript, chunks):
    selected = select_evidence_utterances(transcript, scored(chunks, 0))

    assert [u.id for u in selected] == chunks[0].utterance_ids
    assert all(u.text for u in selected)


def test_overlapping_utterances_appear_only_once(transcript, chunks):
    """Chunks 0 and 1 both contain u1; chunks 1 and 2 both contain u3."""
    assert "m1:u1" in chunks[0].utterance_ids
    assert "m1:u1" in chunks[1].utterance_ids

    selected = select_evidence_utterances(transcript, scored(chunks, 0, 1, 2))

    ids = [u.id for u in selected]
    assert len(ids) == len(set(ids))
    assert ids == ["m1:u0", "m1:u1", "m1:u2", "m1:u3", "m1:u4"]


def test_transcript_order_is_preserved_regardless_of_retrieval_order(transcript, chunks):
    """Retrieval rank must not reorder dialogue."""
    reversed_order = select_evidence_utterances(transcript, scored(chunks, 2, 1, 0))
    natural_order = select_evidence_utterances(transcript, scored(chunks, 0, 1, 2))

    assert [u.id for u in reversed_order] == [u.id for u in natural_order]
    assert [u.index for u in reversed_order] == sorted(u.index for u in reversed_order)


def test_only_retrieved_utterances_are_included(transcript, chunks):
    selected = select_evidence_utterances(transcript, scored(chunks, 2))

    assert [u.id for u in selected] == ["m1:u3", "m1:u4"]
    assert "m1:u0" not in {u.id for u in selected}


def test_no_chunks_yields_no_evidence(transcript):
    assert select_evidence_utterances(transcript, []) == []


def test_unknown_utterance_ids_are_skipped(transcript, chunks):
    """Defensive: a stale chunk must not crash context construction."""
    stale = chunks[0].model_copy(update={"utterance_ids": ["m1:u0", "m1:u99"]})

    selected = select_evidence_utterances(transcript, [ScoredChunk(chunk=stale, score=0.9)])

    assert [u.id for u in selected] == ["m1:u0"]


def test_selection_is_deterministic(transcript, chunks):
    first = select_evidence_utterances(transcript, scored(chunks, 0, 1, 2))
    second = select_evidence_utterances(transcript, scored(chunks, 0, 1, 2))

    assert [u.id for u in first] == [u.id for u in second]


# --- rendering -------------------------------------------------------------


def test_evidence_block_has_the_documented_shape(transcript):
    block = render_evidence_block(transcript.utterances[2])

    assert block == "[m1:u2 | 00:00:52 | Amir]\nWhat happens to the marketing budget?"


def test_context_includes_the_utterance_id(transcript, chunks):
    context = build_evidence_context(select_evidence_utterances(transcript, scored(chunks, 1)))

    assert "m1:u1" in context
    assert "m1:u2" in context
    assert "m1:u3" in context


def test_context_includes_the_normalised_timestamp(transcript, chunks):
    """Raw '01:01:14' and normalised form agree; the label is always HH:MM:SS."""
    context = build_evidence_context(select_evidence_utterances(transcript, scored(chunks, 2)))

    assert "01:01:14" in context
    assert "01:01:38" in context


def test_context_includes_the_speaker(transcript, chunks):
    context = build_evidence_context(select_evidence_utterances(transcript, scored(chunks, 1)))

    assert "Sarah" in context
    assert "Amir" in context


def test_context_includes_the_exact_utterance_text(transcript, chunks):
    selected = select_evidence_utterances(transcript, scored(chunks, 0, 1, 2))

    context = build_evidence_context(selected)

    for utterance in selected:
        assert utterance.text in context


def test_context_does_not_include_retrieval_scores(transcript, chunks):
    """Scores are ranking metadata, not meeting evidence."""
    context = build_evidence_context(
        select_evidence_utterances(transcript, scored(chunks, 0, 1, 2, score=0.87654))
    )

    assert "0.87" not in context
    assert "score" not in context.lower()


def test_context_does_not_include_chunk_ids(transcript, chunks):
    context = build_evidence_context(
        select_evidence_utterances(transcript, scored(chunks, 0, 1, 2))
    )

    assert "m1:c" not in context


def test_context_blocks_are_blank_line_separated(transcript, chunks):
    selected = select_evidence_utterances(transcript, scored(chunks, 0))

    context = build_evidence_context(selected)

    assert context.count("\n\n") == len(selected) - 1


def test_empty_evidence_renders_as_an_empty_string():
    assert build_evidence_context([]) == ""


def test_context_lists_evidence_in_chronological_order(transcript, chunks):
    context = build_evidence_context(
        select_evidence_utterances(transcript, scored(chunks, 2, 0, 1))
    )

    positions = [context.index(f"[m1:u{i} |") for i in range(5)]
    assert positions == sorted(positions)
