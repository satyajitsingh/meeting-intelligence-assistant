"""Speaker-turn-aware chunking behaviour."""

from itertools import pairwise

import pytest

from app.domain.chunker import DEFAULT_TARGET_CHARS, chunk_transcript
from app.domain.errors import ChunkingError, EmptyTranscriptError
from app.domain.models import Transcript, Utterance, make_utterance_id
from app.domain.parser import parse_transcript

SAMPLE = """\
[00:00:12] Sarah: We need to delay the release.
[00:00:25] John: Agreed.
[00:00:40] Sarah: John, please update the launch plan by Friday.
"""


def build_transcript(text: str, meeting_id: str = "m1") -> Transcript:
    return parse_transcript(text, meeting_id=meeting_id, title="Release planning")


def build_utterance(index: int, text: str, speaker: str = "Sarah", meeting_id: str = "m1"):
    return Utterance(
        id=make_utterance_id(meeting_id, index),
        meeting_id=meeting_id,
        index=index,
        speaker=speaker,
        start_seconds=index * 10,
        raw_timestamp="00:00:00",
        text=text,
    )


def transcript_of(utterances: list[Utterance], meeting_id: str = "m1") -> Transcript:
    return Transcript.from_utterances(meeting_id, "Synthetic", utterances)


def synthetic(count: int, chars: int = 100, meeting_id: str = "m1") -> Transcript:
    """A transcript of `count` utterances, each roughly `chars` long."""
    speakers = ["Sarah", "John", "Amir"]
    return transcript_of(
        [
            build_utterance(
                i, "x" * chars, speaker=speakers[i % len(speakers)], meeting_id=meeting_id
            )
            for i in range(count)
        ],
        meeting_id=meeting_id,
    )


# --- basic shape -----------------------------------------------------------


def test_short_transcript_fits_into_one_chunk():
    chunks = chunk_transcript(build_transcript(SAMPLE))

    assert len(chunks) == 1
    assert chunks[0].utterance_ids == ["m1:u0", "m1:u1", "m1:u2"]


def test_long_transcript_produces_multiple_chunks():
    chunks = chunk_transcript(synthetic(20, chars=100))

    assert len(chunks) > 1


def test_chunk_text_contains_speaker_labels():
    chunks = chunk_transcript(build_transcript(SAMPLE))

    assert chunks[0].text == (
        "Sarah: We need to delay the release.\n"
        "John: Agreed.\n"
        "Sarah: John, please update the launch plan by Friday."
    )


def test_chunk_text_has_one_turn_per_line():
    chunk = chunk_transcript(build_transcript(SAMPLE))[0]

    assert len(chunk.text.splitlines()) == 3


def test_chunk_text_contains_no_generated_summary():
    """Chunk text is verbatim dialogue only."""
    transcript = build_transcript(SAMPLE)
    chunk = chunk_transcript(transcript)[0]

    for utterance in transcript.utterances:
        assert utterance.text in chunk.text


# --- metadata --------------------------------------------------------------


def test_chunk_ids_are_stable_and_sequential():
    chunks = chunk_transcript(synthetic(20, chars=100))

    assert [c.id for c in chunks] == [f"m1:c{i}" for i in range(len(chunks))]
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_scoped_to_the_meeting():
    chunks = chunk_transcript(build_transcript(SAMPLE, meeting_id="budget-review"))

    assert chunks[0].id == "budget-review:c0"
    assert all(c.meeting_id == "budget-review" for c in chunks)


def test_chunk_meeting_id_matches_the_transcript():
    transcript = build_transcript(SAMPLE, meeting_id="standup")

    assert all(c.meeting_id == transcript.meeting_id for c in chunk_transcript(transcript))


def test_utterance_ids_are_preserved_in_order():
    transcript = synthetic(20, chars=100)
    chunks = chunk_transcript(transcript)

    valid_ids = [u.id for u in transcript.utterances]
    for chunk in chunks:
        positions = [valid_ids.index(uid) for uid in chunk.utterance_ids]
        assert positions == sorted(positions)
        assert positions == list(range(positions[0], positions[0] + len(positions)))


def test_speakers_are_unique_and_in_first_appearance_order():
    utterances = [
        build_utterance(0, "One.", speaker="Sarah"),
        build_utterance(1, "Two.", speaker="John"),
        build_utterance(2, "Three.", speaker="Sarah"),
        build_utterance(3, "Four.", speaker="Amir"),
    ]

    chunk = chunk_transcript(transcript_of(utterances))[0]

    assert chunk.speakers == ["Sarah", "John", "Amir"]


def test_start_and_end_seconds_come_from_the_boundary_utterances():
    chunks = chunk_transcript(build_transcript(SAMPLE))

    assert chunks[0].start_seconds == 12
    assert chunks[0].end_seconds == 40


def test_start_and_end_seconds_track_each_chunk():
    transcript = synthetic(20, chars=100)
    by_id = {u.id: u for u in transcript.utterances}

    for chunk in chunk_transcript(transcript):
        assert chunk.start_seconds == by_id[chunk.utterance_ids[0]].start_seconds
        assert chunk.end_seconds == by_id[chunk.utterance_ids[-1]].start_seconds


def test_chunks_are_immutable():
    from pydantic import ValidationError as PydanticValidationError

    chunk = chunk_transcript(build_transcript(SAMPLE))[0]

    with pytest.raises(PydanticValidationError):
        chunk.text = "changed"


# --- sizing ----------------------------------------------------------------


def test_chunks_respect_the_target_size():
    chunks = chunk_transcript(synthetic(30, chars=100), target_chars=400)

    for chunk in chunks:
        # A chunk may exceed the target only when it holds a single utterance.
        assert len(chunk.text) <= 400 or len(chunk.utterance_ids) == 1


def test_smaller_target_produces_more_chunks():
    transcript = synthetic(30, chars=100)

    coarse = chunk_transcript(transcript, target_chars=700)
    fine = chunk_transcript(transcript, target_chars=200)

    assert len(fine) > len(coarse)


def test_default_target_is_used_when_unspecified():
    transcript = synthetic(30, chars=100)

    assert chunk_transcript(transcript) == chunk_transcript(
        transcript, target_chars=DEFAULT_TARGET_CHARS
    )


def test_never_splits_an_individual_utterance():
    transcript = synthetic(20, chars=100)
    chunks = chunk_transcript(transcript, target_chars=250)

    rendered = {f"{u.speaker}: {u.text}" for u in transcript.utterances}
    for chunk in chunks:
        for line in chunk.text.splitlines():
            assert line in rendered


# --- overlap ---------------------------------------------------------------


def test_adjacent_chunks_overlap_by_one_utterance():
    chunks = chunk_transcript(synthetic(20, chars=100), target_chars=400)

    assert len(chunks) > 1
    for previous, current in pairwise(chunks):
        assert current.utterance_ids[0] == previous.utterance_ids[-1]


def test_first_chunk_has_no_overlap():
    transcript = synthetic(20, chars=100)
    chunks = chunk_transcript(transcript, target_chars=400)

    assert chunks[0].utterance_ids[0] == transcript.utterances[0].id


def test_overlap_is_exactly_one_utterance():
    chunks = chunk_transcript(synthetic(20, chars=100), target_chars=400)

    for previous, current in pairwise(chunks):
        shared = set(previous.utterance_ids) & set(current.utterance_ids)
        assert len(shared) == 1


# --- oversized utterances --------------------------------------------------


def test_utterance_longer_than_target_becomes_its_own_chunk():
    utterances = [
        build_utterance(0, "short one."),
        build_utterance(1, "y" * 2000, speaker="John"),
        build_utterance(2, "short three.", speaker="Amir"),
    ]

    chunks = chunk_transcript(transcript_of(utterances), target_chars=200)

    oversized = [c for c in chunks if "m1:u1" in c.utterance_ids]
    assert any(c.utterance_ids == ["m1:u1"] for c in oversized)


def test_consecutive_oversized_utterances_terminate():
    utterances = [build_utterance(i, "z" * 1000) for i in range(5)]

    chunks = chunk_transcript(transcript_of(utterances), target_chars=100)

    assert len(chunks) == 5
    assert [c.utterance_ids for c in chunks] == [[f"m1:u{i}"] for i in range(5)]


def test_single_utterance_transcript_yields_one_chunk():
    chunks = chunk_transcript(transcript_of([build_utterance(0, "Only one.")]))

    assert len(chunks) == 1
    assert chunks[0].utterance_ids == ["m1:u0"]


def test_tiny_target_still_terminates_and_covers_everything():
    transcript = synthetic(10, chars=50)

    chunks = chunk_transcript(transcript, target_chars=1)

    assert len(chunks) == 10
    assert {uid for c in chunks for uid in c.utterance_ids} == {u.id for u in transcript.utterances}


# --- coverage and duplication ----------------------------------------------


@pytest.mark.parametrize("target", [1, 50, 120, 200, 400, 700, 5000])
@pytest.mark.parametrize("count", [1, 2, 3, 7, 20])
def test_every_utterance_appears_in_at_least_one_chunk(count, target):
    transcript = synthetic(count, chars=100)

    chunks = chunk_transcript(transcript, target_chars=target)

    covered = {uid for chunk in chunks for uid in chunk.utterance_ids}
    assert covered == {u.id for u in transcript.utterances}


@pytest.mark.parametrize("target", [1, 50, 120, 200, 400, 700, 5000])
@pytest.mark.parametrize("count", [1, 2, 3, 7, 20])
def test_no_chunk_is_a_duplicate_of_another(count, target):
    chunks = chunk_transcript(synthetic(count, chars=100), target_chars=target)

    seen = [c.utterance_ids for c in chunks]
    assert len(seen) == len({tuple(ids) for ids in seen})


@pytest.mark.parametrize("target", [1, 50, 120, 200, 400, 700])
@pytest.mark.parametrize("count", [1, 2, 3, 7, 20])
def test_no_chunk_is_contained_within_its_predecessor(count, target):
    """The duplicate-only chunk that a naive overlap rule would emit."""
    chunks = chunk_transcript(synthetic(count, chars=100), target_chars=target)

    for previous, current in pairwise(chunks):
        assert not set(current.utterance_ids).issubset(set(previous.utterance_ids))


def test_overlap_is_dropped_when_it_would_add_nothing():
    """Two utterances that cannot share a chunk must not produce a repeat chunk."""
    utterances = [build_utterance(i, "w" * 400, speaker=f"S{i}") for i in range(4)]

    chunks = chunk_transcript(transcript_of(utterances), target_chars=700)

    assert [c.utterance_ids for c in chunks] == [[f"m1:u{i}"] for i in range(4)]


def test_final_chunk_is_emitted_once():
    transcript = synthetic(20, chars=100)
    chunks = chunk_transcript(transcript, target_chars=400)

    last_id = transcript.utterances[-1].id
    containing_last = [c for c in chunks if last_id in c.utterance_ids]
    assert len(containing_last) == 1
    assert containing_last[0] is chunks[-1]


def test_chunks_preserve_transcript_order():
    transcript = synthetic(20, chars=100)
    chunks = chunk_transcript(transcript, target_chars=400)

    starts = [c.start_seconds for c in chunks]
    assert starts == sorted(starts)


# --- determinism -----------------------------------------------------------


def test_output_is_deterministic_across_calls():
    transcript = synthetic(20, chars=100)

    assert chunk_transcript(transcript, target_chars=400) == chunk_transcript(
        transcript, target_chars=400
    )


def test_output_is_deterministic_across_reparsing():
    first = chunk_transcript(build_transcript(SAMPLE))
    second = chunk_transcript(build_transcript(SAMPLE))

    assert [c.model_dump() for c in first] == [c.model_dump() for c in second]


# --- validation ------------------------------------------------------------


@pytest.mark.parametrize("target", [0, -1, -700])
def test_rejects_non_positive_target_size(target):
    with pytest.raises(ChunkingError) as exc_info:
        chunk_transcript(build_transcript(SAMPLE), target_chars=target)

    assert exc_info.value.code == "chunking_error"
    assert exc_info.value.details == {"target_chars": target}


def test_rejects_a_transcript_with_no_utterances():
    empty = Transcript(
        meeting_id="m1", title="Empty", speakers=[], utterances=[], duration_seconds=0
    )

    with pytest.raises(EmptyTranscriptError):
        chunk_transcript(empty)


def test_chunking_errors_are_validation_failures():
    with pytest.raises(ChunkingError) as exc_info:
        chunk_transcript(build_transcript(SAMPLE), target_chars=0)

    assert exc_info.value.status_code == 422
