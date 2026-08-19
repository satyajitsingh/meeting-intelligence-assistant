"""Transcript domain models."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.domain.errors import ChunkingError
from app.domain.models import (
    Chunk,
    Transcript,
    Utterance,
    make_chunk_id,
    make_utterance_id,
    render_utterance,
    render_utterances,
)


def build_utterance(index: int, speaker: str = "Sarah", start_seconds: int = 0) -> Utterance:
    return Utterance(
        id=make_utterance_id("m1", index),
        meeting_id="m1",
        index=index,
        speaker=speaker,
        start_seconds=start_seconds,
        raw_timestamp="00:00:00",
        text="Some dialogue.",
    )


def test_utterance_id_format_is_stable():
    assert make_utterance_id("m1", 0) == "m1:u0"
    assert make_utterance_id("meeting-abc", 42) == "meeting-abc:u42"


def test_utterance_id_depends_only_on_meeting_and_index():
    assert make_utterance_id("m1", 3) == make_utterance_id("m1", 3)
    assert make_utterance_id("m1", 3) != make_utterance_id("m2", 3)
    assert make_utterance_id("m1", 3) != make_utterance_id("m1", 4)


def test_display_timestamp_normalises_to_hh_mm_ss():
    utterance = build_utterance(0, start_seconds=3930)

    assert utterance.display_timestamp == "01:05:30"


def test_utterance_is_immutable():
    utterance = build_utterance(0)

    with pytest.raises(PydanticValidationError):
        utterance.text = "changed"


@pytest.mark.parametrize("field", ["speaker", "text", "raw_timestamp"])
def test_utterance_rejects_empty_strings(field):
    values = {
        "id": "m1:u0",
        "meeting_id": "m1",
        "index": 0,
        "speaker": "Sarah",
        "start_seconds": 0,
        "raw_timestamp": "00:00:00",
        "text": "Some dialogue.",
        field: "",
    }

    with pytest.raises(PydanticValidationError):
        Utterance(**values)


@pytest.mark.parametrize(("field", "value"), [("index", -1), ("start_seconds", -1)])
def test_utterance_rejects_negative_numbers(field, value):
    values = {
        "id": "m1:u0",
        "meeting_id": "m1",
        "index": 0,
        "speaker": "Sarah",
        "start_seconds": 0,
        "raw_timestamp": "00:00:00",
        "text": "Some dialogue.",
        field: value,
    }

    with pytest.raises(PydanticValidationError):
        Utterance(**values)


def test_from_utterances_orders_speakers_by_first_appearance():
    utterances = [
        build_utterance(0, speaker="Sarah"),
        build_utterance(1, speaker="John"),
        build_utterance(2, speaker="Sarah"),
        build_utterance(3, speaker="Amir"),
    ]

    transcript = Transcript.from_utterances("m1", "Standup", utterances)

    assert transcript.speakers == ["Sarah", "John", "Amir"]


def test_from_utterances_derives_duration_from_the_latest_timestamp():
    utterances = [
        build_utterance(0, start_seconds=0),
        build_utterance(1, start_seconds=125),
        build_utterance(2, start_seconds=90),
    ]

    transcript = Transcript.from_utterances("m1", "Standup", utterances)

    assert transcript.duration_seconds == 125


def test_from_utterances_preserves_order():
    utterances = [build_utterance(i) for i in range(3)]

    transcript = Transcript.from_utterances("m1", "Standup", utterances)

    assert [u.index for u in transcript.utterances] == [0, 1, 2]


def test_summary_reports_metadata_without_dialogue():
    utterances = [
        build_utterance(0, speaker="Sarah", start_seconds=0),
        build_utterance(1, speaker="John", start_seconds=60),
    ]

    summary = Transcript.from_utterances("m1", "Release planning", utterances).summary()

    assert summary.meeting_id == "m1"
    assert summary.title == "Release planning"
    assert summary.speakers == ["Sarah", "John"]
    assert summary.utterance_count == 2
    assert summary.duration_seconds == 60
    assert "utterances" not in summary.model_dump()


# --- chunk model -----------------------------------------------------------


def test_chunk_id_format_is_stable():
    assert make_chunk_id("m1", 0) == "m1:c0"
    assert make_chunk_id("budget-review", 12) == "budget-review:c12"


def test_chunk_id_depends_only_on_meeting_and_index():
    assert make_chunk_id("m1", 3) == make_chunk_id("m1", 3)
    assert make_chunk_id("m1", 3) != make_chunk_id("m2", 3)
    assert make_chunk_id("m1", 3) != make_chunk_id("m1", 4)


def test_chunk_and_utterance_ids_do_not_collide():
    assert make_chunk_id("m1", 0) != make_utterance_id("m1", 0)


def test_render_utterance_includes_the_speaker_label():
    utterance = build_utterance(0, speaker="Sarah")

    assert render_utterance(utterance) == "Sarah: Some dialogue."


def test_render_utterances_joins_turns_with_newlines():
    utterances = [build_utterance(0, speaker="Sarah"), build_utterance(1, speaker="John")]

    assert render_utterances(utterances) == "Sarah: Some dialogue.\nJohn: Some dialogue."


def test_chunk_from_utterances_derives_its_metadata():
    utterances = [
        build_utterance(0, speaker="Sarah", start_seconds=12),
        build_utterance(1, speaker="John", start_seconds=25),
        build_utterance(2, speaker="Sarah", start_seconds=40),
    ]

    chunk = Chunk.from_utterances(0, utterances)

    assert chunk.id == "m1:c0"
    assert chunk.meeting_id == "m1"
    assert chunk.index == 0
    assert chunk.utterance_ids == ["m1:u0", "m1:u1", "m1:u2"]
    assert chunk.speakers == ["Sarah", "John"]
    assert chunk.start_seconds == 12
    assert chunk.end_seconds == 40
    assert chunk.text.splitlines() == [
        "Sarah: Some dialogue.",
        "John: Some dialogue.",
        "Sarah: Some dialogue.",
    ]


def test_chunk_is_immutable():
    chunk = Chunk.from_utterances(0, [build_utterance(0)])

    with pytest.raises(PydanticValidationError):
        chunk.text = "changed"


def test_chunk_rejects_an_empty_utterance_list():
    with pytest.raises(ChunkingError):
        Chunk.from_utterances(0, [])


def test_chunk_rejects_utterances_from_different_meetings():
    other = Utterance(
        id=make_utterance_id("m2", 0),
        meeting_id="m2",
        index=0,
        speaker="Amir",
        start_seconds=0,
        raw_timestamp="00:00:00",
        text="Different meeting.",
    )

    with pytest.raises(ChunkingError) as exc_info:
        Chunk.from_utterances(0, [build_utterance(0), other])

    assert exc_info.value.details == {"meeting_ids": ["m1", "m2"]}


def test_chunk_rejects_an_empty_utterance_id_list_directly():
    with pytest.raises(PydanticValidationError):
        Chunk(
            id="m1:c0",
            meeting_id="m1",
            index=0,
            text="Sarah: Hello.",
            utterance_ids=[],
            speakers=["Sarah"],
            start_seconds=0,
            end_seconds=0,
        )
