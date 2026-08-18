"""Timestamp parsing and formatting."""

import pytest

from app.domain.errors import InvalidTimestampError
from app.domain.timestamps import format_timestamp, parse_timestamp


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("00:00:00", 0),
        ("00:00:12", 12),
        ("00:01:00", 60),
        ("01:00:00", 3600),
        ("01:05:30", 3930),
        ("12:34:56", 45296),
    ],
)
def test_parses_hh_mm_ss(value, expected):
    assert parse_timestamp(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("00:00", 0),
        ("00:12", 12),
        ("01:00", 60),
        ("05:30", 330),
        ("59:59", 3599),
        ("5:30", 330),
    ],
)
def test_parses_mm_ss(value, expected):
    assert parse_timestamp(value) == expected


def test_surrounding_whitespace_is_tolerated():
    assert parse_timestamp("  00:01:30  ") == 90


@pytest.mark.parametrize(
    "value",
    [
        "",
        "12",
        "abc",
        "00:00:00:00",
        "00:60:00",
        "00:00:60",
        "1:2:3",
        "00-01-30",
        "1h30m",
        "-00:01",
    ],
)
def test_rejects_unsupported_values(value):
    with pytest.raises(InvalidTimestampError):
        parse_timestamp(value)


def test_error_reports_the_line_number_when_supplied():
    with pytest.raises(InvalidTimestampError) as exc_info:
        parse_timestamp("99:99", line_number=7)

    error = exc_info.value
    assert error.line_number == 7
    assert "Line 7" in error.message
    assert error.details == {"line_number": 7, "line": "99:99"}


def test_error_without_a_line_number_has_no_prefix():
    with pytest.raises(InvalidTimestampError) as exc_info:
        parse_timestamp("nope")

    assert exc_info.value.line_number is None
    assert not exc_info.value.message.startswith("Line")


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "00:00:00"), (12, "00:00:12"), (90, "00:01:30"), (3930, "01:05:30")],
)
def test_formats_as_hh_mm_ss(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_format_rejects_negative_values():
    with pytest.raises(ValueError):
        format_timestamp(-1)


@pytest.mark.parametrize("value", ["00:00:00", "00:01:30", "01:05:30"])
def test_parse_and_format_round_trip(value):
    assert format_timestamp(parse_timestamp(value)) == value
