"""Transcript domain models."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.domain.models import Transcript, Utterance, make_utterance_id


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
