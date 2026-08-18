"""Deterministic parser for the v1 transcript format.

The single supported format is one utterance per line::

    [00:00:12] Sarah: We need to delay the release.
    [00:00:25] John: Agreed.
    [00:00:40] Sarah: John, please update the launch plan by Friday.

Rules:

* A line beginning with ``[`` is treated as a new utterance header and must
  match ``[<timestamp>] <speaker>: <text>``. A line that opens with ``[`` but
  does not match is an error rather than being silently absorbed as dialogue.
* Any other non-blank line is continuation text belonging to the previous
  utterance, joined with a single space.
* Blank lines are ignored and do not terminate an utterance.
* Timestamps are ``HH:MM:SS`` or ``MM:SS``.

Supporting further formats (WebVTT, Zoom, Otter, speaker-only exports) is noted
in the README as a production enhancement.

This module performs no I/O: it takes text and returns a
:class:`~app.domain.models.Transcript`.
"""

import re
from dataclasses import dataclass, field

from app.domain.errors import EmptyTranscriptError, TranscriptParseError
from app.domain.models import Transcript, Utterance, make_utterance_id
from app.domain.timestamps import parse_timestamp

UTTERANCE_LINE_PATTERN = re.compile(
    r"^\[(?P<timestamp>[^\]]*)\]\s*(?P<speaker>[^:]*):(?P<text>.*)$"
)

HEADER_PREFIX = "["


@dataclass
class _PendingUtterance:
    """Mutable accumulator for an utterance and its continuation lines."""

    speaker: str
    start_seconds: int
    raw_timestamp: str
    line_number: int
    parts: list[str] = field(default_factory=list)

    def text(self) -> str:
        return " ".join(self.parts)


def parse_transcript(text: str, *, meeting_id: str, title: str) -> Transcript:
    """Parse raw transcript text into a :class:`Transcript`.

    Raises:
        EmptyTranscriptError: the input has no utterances.
        TranscriptParseError: a line is malformed; the error carries its
            1-based line number.
    """
    if not meeting_id.strip():
        raise TranscriptParseError("meeting_id must not be empty.")

    if not text.strip():
        raise EmptyTranscriptError("Transcript is empty.")

    pending: list[_PendingUtterance] = []
    current: _PendingUtterance | None = None

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith(HEADER_PREFIX):
            current = _parse_header(line, line_number)
            pending.append(current)
            continue

        if current is None:
            raise TranscriptParseError(
                "Expected an utterance in the form '[HH:MM:SS] Speaker: text' "
                "before any dialogue text.",
                line_number=line_number,
                line=raw_line,
            )

        current.parts.append(line)

    if not pending:
        raise EmptyTranscriptError("Transcript contains no utterances.")

    utterances = [
        Utterance(
            id=make_utterance_id(meeting_id, index),
            meeting_id=meeting_id,
            index=index,
            speaker=item.speaker,
            start_seconds=item.start_seconds,
            raw_timestamp=item.raw_timestamp,
            text=item.text(),
        )
        for index, item in enumerate(pending)
    ]

    return Transcript.from_utterances(meeting_id, title, utterances)


def _parse_header(line: str, line_number: int) -> _PendingUtterance:
    """Parse one ``[timestamp] speaker: text`` header line."""
    match = UTTERANCE_LINE_PATTERN.match(line)
    if match is None:
        raise TranscriptParseError(
            "Malformed utterance. Expected '[HH:MM:SS] Speaker: text'.",
            line_number=line_number,
            line=line,
        )

    raw_timestamp = match.group("timestamp").strip()
    speaker = match.group("speaker").strip()
    body = match.group("text").strip()

    if not speaker:
        raise TranscriptParseError("Missing speaker name.", line_number=line_number, line=line)

    if not body:
        raise TranscriptParseError("Missing dialogue text.", line_number=line_number, line=line)

    start_seconds = parse_timestamp(raw_timestamp, line_number=line_number)

    return _PendingUtterance(
        speaker=speaker,
        start_seconds=start_seconds,
        raw_timestamp=raw_timestamp,
        line_number=line_number,
        parts=[body],
    )
