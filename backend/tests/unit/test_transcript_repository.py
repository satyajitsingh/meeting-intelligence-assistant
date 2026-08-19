"""InMemoryTranscriptRepository behaviour."""

import pytest

from app.adapters.repository.base import TranscriptRepository
from app.adapters.repository.memory import InMemoryTranscriptRepository
from app.domain.parser import parse_transcript

pytestmark = pytest.mark.anyio

SAMPLE = "[00:00:12] Sarah: Hello.\n[00:00:25] John: Hi there.\n"


def transcript(meeting_id: str = "m1", title: str = "Standup", text: str = SAMPLE):
    return parse_transcript(text, meeting_id=meeting_id, title=title)


def test_satisfies_the_repository_protocol():
    assert isinstance(InMemoryTranscriptRepository(), TranscriptRepository)


async def test_save_then_get_returns_the_transcript():
    repository = InMemoryTranscriptRepository()
    stored = transcript()

    await repository.save(stored)

    assert await repository.get("m1") == stored


async def test_get_returns_none_for_an_unknown_meeting():
    assert await InMemoryTranscriptRepository().get("nope") is None


async def test_get_returns_none_on_an_empty_repository():
    assert await InMemoryTranscriptRepository().get("m1") is None


async def test_stores_multiple_meetings_independently():
    repository = InMemoryTranscriptRepository()
    await repository.save(transcript(meeting_id="m1", title="One"))
    await repository.save(transcript(meeting_id="m2", title="Two"))

    first = await repository.get("m1")
    second = await repository.get("m2")

    assert first is not None and first.title == "One"
    assert second is not None and second.title == "Two"


async def test_save_replaces_a_transcript_with_the_same_meeting_id():
    repository = InMemoryTranscriptRepository()
    await repository.save(transcript(title="Original"))

    await repository.save(transcript(title="Revised", text="[00:00:05] Amir: Replaced.\n"))

    stored = await repository.get("m1")
    assert stored is not None
    assert stored.title == "Revised"
    assert len(stored.utterances) == 1
    assert stored.speakers == ["Amir"]


async def test_save_does_not_duplicate_a_meeting_in_the_listing():
    repository = InMemoryTranscriptRepository()

    for _ in range(3):
        await repository.save(transcript())

    assert len(await repository.list()) == 1


async def test_delete_removes_the_transcript():
    repository = InMemoryTranscriptRepository()
    await repository.save(transcript())

    await repository.delete("m1")

    assert await repository.get("m1") is None


async def test_delete_leaves_other_meetings_intact():
    repository = InMemoryTranscriptRepository()
    await repository.save(transcript(meeting_id="m1"))
    await repository.save(transcript(meeting_id="m2"))

    await repository.delete("m1")

    assert await repository.get("m1") is None
    assert await repository.get("m2") is not None


async def test_delete_of_an_unknown_meeting_is_a_no_op():
    repository = InMemoryTranscriptRepository()
    await repository.save(transcript())

    await repository.delete("never-saved")

    assert await repository.get("m1") is not None


async def test_a_meeting_can_be_saved_again_after_deletion():
    repository = InMemoryTranscriptRepository()
    await repository.save(transcript())
    await repository.delete("m1")

    await repository.save(transcript(title="Second time"))

    stored = await repository.get("m1")
    assert stored is not None and stored.title == "Second time"


async def test_list_is_empty_for_a_new_repository():
    assert await InMemoryTranscriptRepository().list() == []


async def test_list_returns_summaries_not_full_transcripts():
    repository = InMemoryTranscriptRepository()
    await repository.save(transcript())

    summaries = await repository.list()

    assert len(summaries) == 1
    assert summaries[0].meeting_id == "m1"
    assert summaries[0].title == "Standup"
    assert summaries[0].speakers == ["Sarah", "John"]
    assert summaries[0].utterance_count == 2
    assert summaries[0].duration_seconds == 25
    assert "utterances" not in summaries[0].model_dump()


async def test_list_preserves_insertion_order():
    repository = InMemoryTranscriptRepository()
    for meeting_id in ["charlie", "alpha", "bravo"]:
        await repository.save(transcript(meeting_id=meeting_id))

    assert [s.meeting_id for s in await repository.list()] == ["charlie", "alpha", "bravo"]


async def test_resaving_keeps_a_meetings_position_in_the_listing():
    repository = InMemoryTranscriptRepository()
    for meeting_id in ["one", "two", "three"]:
        await repository.save(transcript(meeting_id=meeting_id))

    await repository.save(transcript(meeting_id="one", title="Updated"))

    assert [s.meeting_id for s in await repository.list()] == ["one", "two", "three"]


async def test_list_order_is_stable_across_calls():
    repository = InMemoryTranscriptRepository()
    for meeting_id in ["a", "b", "c"]:
        await repository.save(transcript(meeting_id=meeting_id))

    assert [s.meeting_id for s in await repository.list()] == [
        s.meeting_id for s in await repository.list()
    ]


async def test_deleting_removes_the_meeting_from_the_listing():
    repository = InMemoryTranscriptRepository()
    await repository.save(transcript(meeting_id="m1"))
    await repository.save(transcript(meeting_id="m2"))

    await repository.delete("m1")

    assert [s.meeting_id for s in await repository.list()] == ["m2"]


async def test_repositories_do_not_share_state():
    first = InMemoryTranscriptRepository()
    second = InMemoryTranscriptRepository()

    await first.save(transcript())

    assert await second.get("m1") is None
