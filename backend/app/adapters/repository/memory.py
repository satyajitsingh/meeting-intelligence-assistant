"""In-process transcript repository.

Transcripts live for the lifetime of the process. Persistence is documented as
a production enhancement rather than built here: for a single-process demo it
would add a storage engine, a migration story and a serialisation format
without changing any interface.
"""

from __future__ import annotations

from app.domain.models import Transcript, TranscriptSummary


class InMemoryTranscriptRepository:
    """Dict-backed :class:`TranscriptRepository`.

    Keyed by ``meeting_id``. Because a dict preserves insertion order -- and
    keeps an existing key's position when it is reassigned -- listing is stable
    and re-saving a meeting does not reorder the listing.
    """

    def __init__(self) -> None:
        self._transcripts: dict[str, Transcript] = {}

    async def save(self, transcript: Transcript) -> None:
        self._transcripts[transcript.meeting_id] = transcript

    async def get(self, meeting_id: str) -> Transcript | None:
        return self._transcripts.get(meeting_id)

    async def delete(self, meeting_id: str) -> None:
        self._transcripts.pop(meeting_id, None)

    async def list(self) -> list[TranscriptSummary]:
        return [transcript.summary() for transcript in self._transcripts.values()]
