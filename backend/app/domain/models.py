"""Core transcript domain models.

An :class:`Utterance` is the citation unit: every answer the system produces
points at utterance IDs, so the identifiers must be stable and reproducible for
a given transcript. Chunks (added in the next phase) are the retrieval unit and
reference utterances by ID.
"""

from pydantic import BaseModel, ConfigDict, Field

from app.domain.timestamps import format_timestamp


def make_utterance_id(meeting_id: str, index: int) -> str:
    """Build the stable identifier for an utterance.

    Derived purely from the meeting and the utterance's position, so re-parsing
    the same transcript always yields the same IDs.
    """
    return f"{meeting_id}:u{index}"


class Utterance(BaseModel):
    """A single speaker turn within a meeting."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, description="Stable identifier, '<meeting_id>:u<index>'.")
    meeting_id: str = Field(min_length=1)
    index: int = Field(ge=0, description="Zero-based position in the transcript.")
    speaker: str = Field(min_length=1)
    start_seconds: int = Field(ge=0, description="Offset from the start of the meeting.")
    raw_timestamp: str = Field(min_length=1, description="Timestamp exactly as written.")
    text: str = Field(min_length=1)

    @property
    def display_timestamp(self) -> str:
        """Normalised ``HH:MM:SS`` label for citation display."""
        return format_timestamp(self.start_seconds)


class TranscriptSummary(BaseModel):
    """Lightweight description of a transcript, without the dialogue."""

    meeting_id: str
    title: str
    speakers: list[str]
    utterance_count: int
    duration_seconds: int


class Transcript(BaseModel):
    """A parsed meeting transcript in original speaking order."""

    model_config = ConfigDict(frozen=True)

    meeting_id: str = Field(min_length=1)
    title: str
    speakers: list[str] = Field(description="Distinct speakers, in order of first appearance.")
    utterances: list[Utterance]
    duration_seconds: int = Field(ge=0)

    @classmethod
    def from_utterances(
        cls, meeting_id: str, title: str, utterances: list[Utterance]
    ) -> "Transcript":
        """Assemble a transcript, deriving its speaker list and duration.

        Speakers are ordered by first appearance rather than alphabetically, so
        the ordering reflects the meeting and stays deterministic.
        """
        speakers: list[str] = []
        for utterance in utterances:
            if utterance.speaker not in speakers:
                speakers.append(utterance.speaker)

        # No end times exist in the source format, so duration is the offset of
        # the last thing said. `max` rather than the final element keeps this
        # sane if a transcript's timestamps are not monotonic.
        duration_seconds = max((u.start_seconds for u in utterances), default=0)

        return cls(
            meeting_id=meeting_id,
            title=title,
            speakers=speakers,
            utterances=utterances,
            duration_seconds=duration_seconds,
        )

    def summary(self) -> TranscriptSummary:
        """Return the metadata-only view of this transcript."""
        return TranscriptSummary(
            meeting_id=self.meeting_id,
            title=self.title,
            speakers=self.speakers,
            utterance_count=len(self.utterances),
            duration_seconds=self.duration_seconds,
        )
