"""Transcript parsing behaviour."""

import pytest

from app.domain.errors import EmptyTranscriptError, InvalidTimestampError, TranscriptParseError
from app.domain.parser import parse_transcript

SAMPLE = """\
[00:00:12] Sarah: We need to delay the release.
[00:00:25] John: Agreed.
[00:00:40] Sarah: John, please update the launch plan by Friday.
"""


def parse(text: str, meeting_id: str = "m1", title: str = "Release planning"):
    return parse_transcript(text, meeting_id=meeting_id, title=title)


def test_parses_the_documented_format():
    transcript = parse(SAMPLE)

    assert transcript.meeting_id == "m1"
    assert transcript.title == "Release planning"
    assert len(transcript.utterances) == 3

    first = transcript.utterances[0]
    assert first.speaker == "Sarah"
    assert first.text == "We need to delay the release."
    assert first.start_seconds == 12
    assert first.raw_timestamp == "00:00:12"
    assert first.index == 0


def test_preserves_transcript_order():
    transcript = parse(SAMPLE)

    assert [u.index for u in transcript.utterances] == [0, 1, 2]
    assert [u.start_seconds for u in transcript.utterances] == [12, 25, 40]
    assert [u.speaker for u in transcript.utterances] == ["Sarah", "John", "Sarah"]


def test_parses_mm_ss_timestamps():
    transcript = parse("[00:12] Sarah: Hello.\n[05:30] John: Hi.\n")

    assert [u.start_seconds for u in transcript.utterances] == [12, 330]
    assert [u.raw_timestamp for u in transcript.utterances] == ["00:12", "05:30"]


def test_parses_hh_mm_ss_timestamps():
    transcript = parse("[01:05:30] Sarah: Still going.\n")

    assert transcript.utterances[0].start_seconds == 3930
    assert transcript.utterances[0].raw_timestamp == "01:05:30"


def test_preserves_the_original_timestamp_string():
    transcript = parse("[5:30] Sarah: Hello.\n")

    assert transcript.utterances[0].raw_timestamp == "5:30"
    assert transcript.utterances[0].start_seconds == 330
    assert transcript.utterances[0].display_timestamp == "00:05:30"


def test_extracts_speakers_in_order_of_first_appearance():
    text = "[00:01] Sarah: One.\n[00:02] John: Two.\n[00:03] Sarah: Three.\n[00:04] Amir: Four.\n"

    assert parse(text).speakers == ["Sarah", "John", "Amir"]


def test_calculates_duration_from_the_final_timestamp():
    assert parse(SAMPLE).duration_seconds == 40


def test_generates_stable_utterance_ids():
    transcript = parse(SAMPLE)

    assert [u.id for u in transcript.utterances] == ["m1:u0", "m1:u1", "m1:u2"]


def test_utterance_ids_are_reproducible_across_parses():
    first = parse(SAMPLE)
    second = parse(SAMPLE)

    assert [u.id for u in first.utterances] == [u.id for u in second.utterances]


def test_utterance_ids_are_scoped_to_the_meeting():
    transcript = parse(SAMPLE, meeting_id="budget-review")

    assert transcript.utterances[0].id == "budget-review:u0"
    assert all(u.meeting_id == "budget-review" for u in transcript.utterances)


def test_joins_multiline_dialogue_into_one_utterance():
    text = (
        "[00:00:12] Sarah: We need to delay the release\n"
        "because the migration is not finished.\n"
        "[00:00:25] John: Agreed.\n"
    )

    transcript = parse(text)

    assert len(transcript.utterances) == 2
    assert transcript.utterances[0].text == (
        "We need to delay the release because the migration is not finished."
    )


def test_multiline_dialogue_supports_several_continuation_lines():
    text = "[00:01] Sarah: One\ntwo\nthree\n"

    assert parse(text).utterances[0].text == "One two three"


def test_blank_lines_are_ignored():
    text = "\n[00:01] Sarah: Hello.\n\n\n[00:02] John: Hi.\n\n"

    transcript = parse(text)

    assert len(transcript.utterances) == 2
    assert transcript.utterances[1].text == "Hi."


def test_dialogue_may_contain_colons():
    transcript = parse("[00:01] Sarah: Note: we ship on Friday.\n")

    assert transcript.utterances[0].speaker == "Sarah"
    assert transcript.utterances[0].text == "Note: we ship on Friday."


def test_handles_missing_trailing_newline():
    assert len(parse("[00:01] Sarah: Hello.").utterances) == 1


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "\t\n  \n"])
def test_rejects_empty_transcripts(text):
    with pytest.raises(EmptyTranscriptError):
        parse(text)


def test_rejects_a_transcript_of_only_blank_and_unparsed_whitespace():
    with pytest.raises(EmptyTranscriptError) as exc_info:
        parse("\n \n")

    assert exc_info.value.code == "empty_transcript"


def test_rejects_a_malformed_line_with_its_line_number():
    text = "[00:01] Sarah: Hello.\n[00:02] John Hi there.\n[00:03] Amir: Bye.\n"

    with pytest.raises(TranscriptParseError) as exc_info:
        parse(text)

    error = exc_info.value
    assert error.line_number == 2
    assert "Line 2" in error.message
    assert error.details is not None
    assert error.details["line_number"] == 2


def test_rejects_an_unclosed_timestamp_bracket():
    with pytest.raises(TranscriptParseError) as exc_info:
        parse("[00:01 Sarah: Hello.\n")

    assert exc_info.value.line_number == 1


def test_rejects_an_invalid_timestamp_with_its_line_number():
    text = "[00:01] Sarah: Hello.\n[99:99] John: Hi.\n"

    with pytest.raises(InvalidTimestampError) as exc_info:
        parse(text)

    assert exc_info.value.line_number == 2


def test_rejects_a_missing_speaker():
    with pytest.raises(TranscriptParseError) as exc_info:
        parse("[00:01] : Hello there.\n")

    assert "Missing speaker" in exc_info.value.message
    assert exc_info.value.line_number == 1


def test_rejects_a_missing_dialogue_body():
    with pytest.raises(TranscriptParseError) as exc_info:
        parse("[00:01] Sarah:\n")

    assert "Missing dialogue" in exc_info.value.message
    assert exc_info.value.line_number == 1


def test_rejects_dialogue_before_the_first_timestamp():
    text = "Some preamble text.\n[00:01] Sarah: Hello.\n"

    with pytest.raises(TranscriptParseError) as exc_info:
        parse(text)

    assert exc_info.value.line_number == 1


def test_rejects_an_empty_meeting_id():
    with pytest.raises(TranscriptParseError):
        parse(SAMPLE, meeting_id="  ")


def test_parse_errors_are_reported_as_validation_failures():
    """Domain errors carry the HTTP contract without the domain importing FastAPI."""
    with pytest.raises(TranscriptParseError) as exc_info:
        parse("[00:01] Sarah:\n")

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "transcript_parse_error"


def test_summary_reflects_the_parsed_transcript():
    summary = parse(SAMPLE).summary()

    assert summary.speakers == ["Sarah", "John"]
    assert summary.utterance_count == 3
    assert summary.duration_seconds == 40
