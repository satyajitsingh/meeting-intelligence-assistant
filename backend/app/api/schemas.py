"""API request and response models.

Kept separate from the domain models so the wire contract can evolve without
dragging the domain with it, and so responses expose only what a client needs --
notably never embeddings or raw vectors.
"""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from app.domain.models import ScoredChunk, Transcript, TranscriptSummary, Utterance
from app.services.retrieval import DEFAULT_K

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class IngestTranscriptRequest(BaseModel):
    """Body of ``POST /api/transcripts``."""

    meeting_id: NonEmptyString = Field(
        description="Caller-supplied identifier. Never generated for you.",
        examples=["release-planning-2026-08-19"],
    )
    title: NonEmptyString = Field(
        description="Human-readable meeting name.", examples=["Release planning"]
    )
    # Not stripped: leading and trailing whitespace is part of the line
    # structure the parser reads. Whitespace-only input is rejected by the
    # parser, which reports it as an empty transcript.
    transcript: str = Field(
        min_length=1,
        description="Raw transcript text in the documented '[HH:MM:SS] Speaker: text' format.",
        examples=["[00:00:12] Sarah: We need to delay the release."],
    )


class IngestTranscriptResponse(BaseModel):
    """Result of a successful ingestion."""

    meeting_id: str
    title: str
    speakers: list[str]
    utterance_count: int
    chunk_count: int
    duration_seconds: int

    @classmethod
    def from_result(
        cls, summary: TranscriptSummary, chunk_count: int
    ) -> "IngestTranscriptResponse":
        return cls(**summary.model_dump(), chunk_count=chunk_count)


class TranscriptSummaryResponse(BaseModel):
    """One row of ``GET /api/transcripts``."""

    meeting_id: str
    title: str
    speakers: list[str]
    utterance_count: int
    duration_seconds: int

    @classmethod
    def from_summary(cls, summary: TranscriptSummary) -> "TranscriptSummaryResponse":
        return cls(**summary.model_dump())


class UtteranceResponse(BaseModel):
    """A single speaker turn, as the transcript viewer renders it."""

    id: str
    index: int
    speaker: str
    start_seconds: int
    raw_timestamp: str = Field(description="Timestamp exactly as written in the upload.")
    display_timestamp: str = Field(description="Normalised HH:MM:SS label.")
    text: str

    @classmethod
    def from_utterance(cls, utterance: Utterance) -> "UtteranceResponse":
        return cls(
            id=utterance.id,
            index=utterance.index,
            speaker=utterance.speaker,
            start_seconds=utterance.start_seconds,
            raw_timestamp=utterance.raw_timestamp,
            display_timestamp=utterance.display_timestamp,
            text=utterance.text,
        )


class TranscriptDetailResponse(BaseModel):
    """Body of ``GET /api/transcripts/{meeting_id}``.

    Carries utterances -- the citation unit -- and deliberately not chunks or
    vectors, which are retrieval internals with no meaning to a client.
    """

    meeting_id: str
    title: str
    speakers: list[str]
    duration_seconds: int
    utterances: list[UtteranceResponse]

    @classmethod
    def from_transcript(cls, transcript: Transcript) -> "TranscriptDetailResponse":
        return cls(
            meeting_id=transcript.meeting_id,
            title=transcript.title,
            speakers=transcript.speakers,
            duration_seconds=transcript.duration_seconds,
            utterances=[UtteranceResponse.from_utterance(u) for u in transcript.utterances],
        )


class RetrieveRequest(BaseModel):
    """Body of ``POST /api/retrieval``."""

    meeting_id: NonEmptyString = Field(
        description="Meeting to search. Retrieval never spans meetings.",
        examples=["release-planning"],
    )
    query: NonEmptyString = Field(
        description="Natural-language question.",
        examples=["What was decided about the marketing budget?"],
    )
    k: int = Field(default=DEFAULT_K, gt=0, description="Maximum number of chunks to return.")


class RetrievalResultResponse(BaseModel):
    """One ranked chunk.

    ``utterance_ids`` is the bridge back to the citation units: retrieval
    returns chunks, but evidence is always resolved to individual utterances.
    """

    chunk_id: str
    score: float = Field(description="Cosine similarity; higher is closer.")
    text: str = Field(description="Speaker-labelled dialogue, exactly as stored.")
    speakers: list[str]
    start_seconds: int
    end_seconds: int
    utterance_ids: list[str]

    @classmethod
    def from_scored_chunk(cls, scored: ScoredChunk) -> "RetrievalResultResponse":
        return cls(
            chunk_id=scored.chunk.id,
            score=scored.score,
            text=scored.chunk.text,
            speakers=scored.chunk.speakers,
            start_seconds=scored.chunk.start_seconds,
            end_seconds=scored.chunk.end_seconds,
            utterance_ids=scored.chunk.utterance_ids,
        )


class RetrievalResponse(BaseModel):
    """Ranked retrieval results for one question against one meeting."""

    meeting_id: str
    query: str = Field(description="The question as searched, after trimming.")
    results: list[RetrievalResultResponse] = Field(
        description="Ranked best-first. Empty when the meeting has no matching chunks."
    )
