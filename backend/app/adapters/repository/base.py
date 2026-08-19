"""Transcript repository port.

The repository holds parsed transcripts, which are the source of truth for
citations: retrieval returns chunks, and resolving a chunk back to a speaker,
timestamp and verbatim quote means reading its utterances from here.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain.models import Transcript, TranscriptSummary


@runtime_checkable
class TranscriptRepository(Protocol):
    """Stores parsed transcripts, keyed by meeting."""

    async def save(self, transcript: Transcript) -> None:
        """Store a transcript, replacing any existing one with the same ID."""
        ...

    async def get(self, meeting_id: str) -> Transcript | None:
        """Return a transcript, or ``None`` when the meeting is unknown."""
        ...

    async def delete(self, meeting_id: str) -> None:
        """Remove a transcript. A no-op when the meeting is unknown."""
        ...

    async def list(self) -> list[TranscriptSummary]:
        """Return a summary of every stored transcript, in insertion order."""
        ...
