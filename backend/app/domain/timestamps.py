"""Transcript timestamp parsing.

Two forms are supported in v1, matching the documented transcript format:

* ``HH:MM:SS`` -- e.g. ``01:05:30``
* ``MM:SS``    -- e.g. ``05:30``

Minutes and seconds must be below 60; the pattern enforces that directly, so a
value such as ``00:75:00`` is rejected rather than silently normalised.
"""

import re

from app.domain.errors import InvalidTimestampError

TIMESTAMP_PATTERN = re.compile(
    r"^(?:(?P<hours>\d{1,2}):)?(?P<minutes>[0-5]?\d):(?P<seconds>[0-5]\d)$"
)

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600


def parse_timestamp(value: str, *, line_number: int | None = None) -> int:
    """Convert a ``HH:MM:SS`` or ``MM:SS`` timestamp to whole seconds.

    ``line_number`` is threaded through purely so that a failure can report
    where in the source transcript it occurred.
    """
    match = TIMESTAMP_PATTERN.match(value.strip())
    if match is None:
        raise InvalidTimestampError(
            f"Invalid timestamp {value!r}. Expected HH:MM:SS or MM:SS.",
            line_number=line_number,
            line=value,
        )

    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    return hours * SECONDS_PER_HOUR + minutes * SECONDS_PER_MINUTE + seconds


def format_timestamp(total_seconds: int) -> str:
    """Render whole seconds as ``HH:MM:SS`` for display."""
    if total_seconds < 0:
        raise ValueError("total_seconds must not be negative.")

    hours, remainder = divmod(total_seconds, SECONDS_PER_HOUR)
    minutes, seconds = divmod(remainder, SECONDS_PER_MINUTE)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
