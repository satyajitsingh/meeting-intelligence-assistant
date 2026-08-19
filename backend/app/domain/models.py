"""Core transcript domain models.

An :class:`Utterance` is the citation unit: every answer the system produces
points at utterance IDs, so the identifiers must be stable and reproducible for
a given transcript. Chunks (added in the next phase) are the retrieval unit and
reference utterances by ID.
"""

from pydantic import BaseModel, ConfigDict, Field

from app.domain.errors import ChunkingError
from app.domain.timestamps import format_timestamp


def make_utterance_id(meeting_id: str, index: int) -> str:
    """Build the stable identifier for an utterance.

    Derived purely from the meeting and the utterance's position, so re-parsing
    the same transcript always yields the same IDs.
    """
    return f"{meeting_id}:u{index}"


def make_chunk_id(meeting_id: str, index: int) -> str:
    """Build the stable identifier for a retrieval chunk.

    Like utterance IDs, derived purely from the meeting and position so that
    re-chunking the same transcript always yields the same identifiers.
    """
    return f"{meeting_id}:c{index}"


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


UTTERANCE_SEPARATOR = "\n"


def render_utterance(utterance: Utterance) -> str:
    """Render one turn as ``<speaker>: <text>``.

    Speaker names are part of the rendered text, not just metadata, so that a
    question naming a participant can match on the name once this text is
    embedded.
    """
    return f"{utterance.speaker}: {utterance.text}"


def render_utterances(utterances: list[Utterance]) -> str:
    """Render consecutive turns as one newline-separated block."""
    return UTTERANCE_SEPARATOR.join(render_utterance(u) for u in utterances)


class Chunk(BaseModel):
    """A contiguous run of utterances, used as the retrieval unit.

    A chunk is never a citation: ``utterance_ids`` maps it back to the exact
    speaker turns it was built from, and answers cite those instead. Chunks may
    overlap by one utterance, so an utterance ID can appear in two chunks.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, description="Stable identifier, '<meeting_id>:c<index>'.")
    meeting_id: str = Field(min_length=1)
    index: int = Field(ge=0, description="Zero-based position in the chunk sequence.")
    text: str = Field(min_length=1, description="Speaker-labelled dialogue, one turn per line.")
    utterance_ids: list[str] = Field(
        min_length=1, description="Source utterance IDs, in transcript order."
    )
    speakers: list[str] = Field(description="Distinct speakers, in order of first appearance.")
    start_seconds: int = Field(ge=0, description="Start of the first utterance.")
    end_seconds: int = Field(ge=0, description="Start of the final utterance.")

    @classmethod
    def from_utterances(cls, index: int, utterances: list[Utterance]) -> "Chunk":
        """Build a chunk from consecutive utterances, deriving its metadata."""
        if not utterances:
            raise ChunkingError("A chunk must contain at least one utterance.")

        meeting_ids = {u.meeting_id for u in utterances}
        if len(meeting_ids) > 1:
            raise ChunkingError(
                "A chunk must not span multiple meetings.",
                details={"meeting_ids": sorted(meeting_ids)},
            )

        meeting_id = utterances[0].meeting_id

        speakers: list[str] = []
        for utterance in utterances:
            if utterance.speaker not in speakers:
                speakers.append(utterance.speaker)

        return cls(
            id=make_chunk_id(meeting_id, index),
            meeting_id=meeting_id,
            index=index,
            text=render_utterances(utterances),
            utterance_ids=[u.id for u in utterances],
            speakers=speakers,
            start_seconds=utterances[0].start_seconds,
            end_seconds=utterances[-1].start_seconds,
        )
