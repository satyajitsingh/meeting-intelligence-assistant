"""CitationResolver: deterministic validation and evidence resolution."""

import logging

import pytest

from app.domain.models import (
    GeneratedAnswer,
    GeneratedCitation,
    ResolvedCitation,
    ValidatedAnswer,
)
from app.domain.parser import parse_transcript
from app.services.citations import CitationResolver, InvalidCitationReason

SAMPLE = (
    "[00:00:12] Sarah: We need to delay the release because migration is unfinished.\n"
    "[00:00:31] John: Agreed. The migration script still fails.\n"
    "[00:00:52] Amir: What happens to the marketing budget?\n"
    "[00:01:14] Sarah: The budget is unchanged.\n"
    "[00:01:38] John: I will update the launch plan by Friday.\n"
)

ALL_IDS = [f"m1:u{i}" for i in range(5)]


@pytest.fixture
def transcript():
    return parse_transcript(SAMPLE, meeting_id="m1", title="Release planning")


@pytest.fixture
def resolver():
    return CitationResolver()


def generated(*utterance_ids, answer="An answer.", insufficient=False) -> GeneratedAnswer:
    return GeneratedAnswer(
        answer=answer,
        citations=[GeneratedCitation(utterance_id=i) for i in utterance_ids],
        insufficient_evidence=insufficient,
    )


def resolve(resolver, transcript, gen, allowed=None) -> ValidatedAnswer:
    return resolver.resolve(
        generated=gen,
        transcript=transcript,
        allowed_utterance_ids=list(ALL_IDS if allowed is None else allowed),
    )


# --- valid citations -------------------------------------------------------


def test_a_valid_citation_resolves(resolver, transcript):
    result = resolve(resolver, transcript, generated("m1:u3"))

    assert len(result.citations) == 1
    assert isinstance(result.citations[0], ResolvedCitation)
    assert result.citations[0].utterance_id == "m1:u3"


def test_quote_is_the_exact_transcript_text(resolver, transcript):
    result = resolve(resolver, transcript, generated("m1:u3"))

    assert result.citations[0].quote == "The budget is unchanged."


def test_speaker_comes_from_the_transcript(resolver, transcript):
    result = resolve(resolver, transcript, generated("m1:u2"))

    assert result.citations[0].speaker == "Amir"


def test_timestamp_comes_from_the_transcript(resolver, transcript):
    result = resolve(resolver, transcript, generated("m1:u3"))

    assert result.citations[0].timestamp == "00:01:14"


def test_start_seconds_comes_from_the_transcript(resolver, transcript):
    result = resolve(resolver, transcript, generated("m1:u3"))

    assert result.citations[0].start_seconds == 74


def test_every_resolved_field_matches_the_source_utterance(resolver, transcript):
    result = resolve(resolver, transcript, generated(*ALL_IDS))

    by_id = {u.id: u for u in transcript.utterances}
    for citation in result.citations:
        source = by_id[citation.utterance_id]
        assert citation.speaker == source.speaker
        assert citation.quote == source.text
        assert citation.timestamp == source.display_timestamp
        assert citation.start_seconds == source.start_seconds


def test_answer_text_is_unchanged(resolver, transcript):
    result = resolve(resolver, transcript, generated("m1:u3", answer="Verbatim prose."))

    assert result.answer == "Verbatim prose."


def test_several_valid_citations_all_resolve(resolver, transcript):
    result = resolve(resolver, transcript, generated("m1:u2", "m1:u3"))

    assert [c.utterance_id for c in result.citations] == ["m1:u2", "m1:u3"]


# --- invalid citations -----------------------------------------------------


def test_an_invented_id_is_discarded(resolver, transcript):
    result = resolve(resolver, transcript, generated("m1:u999"))

    assert result.citations == []


def test_an_id_from_another_meeting_is_discarded(resolver, transcript):
    result = resolve(resolver, transcript, generated("m2:u0"), allowed=[*ALL_IDS, "m2:u0"])

    assert result.citations == []


def test_an_existing_id_that_was_not_supplied_as_evidence_is_discarded(resolver, transcript):
    """The utterance is real, but the model was never shown it."""
    result = resolve(resolver, transcript, generated("m1:u4"), allowed=["m1:u0", "m1:u1"])

    assert result.citations == []


def test_a_malformed_id_is_discarded(resolver, transcript):
    result = resolve(resolver, transcript, generated("not-an-id"), allowed=["not-an-id"])

    assert result.citations == []


def test_a_reformatted_id_is_discarded(resolver, transcript):
    """IDs are matched exactly; near-misses are not repaired."""
    result = resolve(
        resolver, transcript, generated("m1:U3", "m1-u3", " m1:u3"), allowed=[*ALL_IDS]
    )

    assert result.citations == []


def test_mixed_valid_and_invalid_returns_only_the_valid(resolver, transcript):
    result = resolve(resolver, transcript, generated("m1:u999", "m1:u3", "m2:u0", "m1:u2"))

    assert [c.utterance_id for c in result.citations] == ["m1:u3", "m1:u2"]


def test_one_invalid_citation_does_not_fail_the_answer(resolver, transcript):
    result = resolve(resolver, transcript, generated("m1:u3", "m1:u999"))

    assert result.answer == "An answer."
    assert len(result.citations) == 1


def test_all_invalid_yields_no_citations_without_fabricating(resolver, transcript):
    result = resolve(resolver, transcript, generated("m1:u999", "m1:u998"))

    assert result.citations == []
    assert result.answer == "An answer."


def test_insufficient_evidence_is_not_flipped_when_citations_are_invalid(resolver, transcript):
    """Documented decision: the flag reports the model's judgement, not ours."""
    result = resolve(resolver, transcript, generated("m1:u999"))

    assert result.insufficient_evidence is False


# --- duplicates and ordering -----------------------------------------------


def test_duplicate_citations_are_collapsed(resolver, transcript):
    result = resolve(resolver, transcript, generated("m1:u3", "m1:u3", "m1:u3"))

    assert [c.utterance_id for c in result.citations] == ["m1:u3"]


def test_first_occurrence_order_is_preserved(resolver, transcript):
    result = resolve(resolver, transcript, generated("m1:u4", "m1:u1", "m1:u4", "m1:u0"))

    assert [c.utterance_id for c in result.citations] == ["m1:u4", "m1:u1", "m1:u0"]


def test_retrieval_order_is_not_resorted_into_transcript_order(resolver, transcript):
    """Citation order follows the model, not the transcript."""
    result = resolve(resolver, transcript, generated("m1:u3", "m1:u0"))

    assert [c.utterance_id for c in result.citations] == ["m1:u3", "m1:u0"]


def test_duplicate_invalid_ids_are_collapsed_too(resolver, transcript, caplog):
    with caplog.at_level(logging.WARNING):
        resolve(resolver, transcript, generated("m1:u999", "m1:u999"))

    invalid = [r for r in caplog.records if "citation.invalid" in r.getMessage()]
    assert len(invalid) == 1


# --- insufficient evidence -------------------------------------------------


def test_insufficient_evidence_forces_empty_citations(resolver, transcript):
    result = resolve(resolver, transcript, generated("m1:u3", insufficient=True))

    assert result.citations == []
    assert result.insufficient_evidence is True


def test_insufficient_evidence_keeps_the_answer_text(resolver, transcript):
    result = resolve(
        resolver,
        transcript,
        generated("m1:u3", answer="The meeting does not cover that.", insufficient=True),
    )

    assert result.answer == "The meeting does not cover that."


def test_no_citations_at_all_is_not_an_error(resolver, transcript):
    result = resolve(resolver, transcript, generated())

    assert result.citations == []
    assert result.insufficient_evidence is False


# --- purity ----------------------------------------------------------------


def test_the_generated_answer_is_not_mutated(resolver, transcript):
    gen = generated("m1:u3", "m1:u999")
    before = gen.model_dump()

    resolve(resolver, transcript, gen)

    assert gen.model_dump() == before
    assert [c.utterance_id for c in gen.citations] == ["m1:u3", "m1:u999"]


def test_the_allowed_id_list_is_not_mutated(resolver, transcript):
    allowed = ["m1:u3"]

    resolver.resolve(
        generated=generated("m1:u3", "m1:u999"),
        transcript=transcript,
        allowed_utterance_ids=allowed,
    )

    assert allowed == ["m1:u3"]


def test_resolution_is_deterministic(resolver, transcript):
    gen = generated("m1:u3", "m1:u999", "m1:u2")

    first = resolve(resolver, transcript, gen)
    second = resolve(resolver, transcript, gen)

    assert first == second


# --- logging ---------------------------------------------------------------


def test_invalid_citations_are_logged_with_the_reason(resolver, transcript, caplog):
    with caplog.at_level(logging.WARNING):
        resolve(resolver, transcript, generated("m1:u999"))

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "citation.invalid" in messages
    assert "m1:u999" in messages
    assert InvalidCitationReason.NOT_FOUND in messages


def test_wrong_meeting_is_logged_with_its_own_reason(resolver, transcript, caplog):
    with caplog.at_level(logging.WARNING):
        resolve(resolver, transcript, generated("m2:u0"), allowed=[*ALL_IDS, "m2:u0"])

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert InvalidCitationReason.WRONG_MEETING in messages


def test_not_allowed_is_logged_with_its_own_reason(resolver, transcript, caplog):
    with caplog.at_level(logging.WARNING):
        resolve(resolver, transcript, generated("m1:u4"), allowed=["m1:u0"])

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert InvalidCitationReason.NOT_ALLOWED in messages


def test_invalid_citation_logs_include_the_meeting_id(resolver, transcript, caplog):
    with caplog.at_level(logging.WARNING):
        resolve(resolver, transcript, generated("m1:u999"))

    assert "m1" in " ".join(r.getMessage() for r in caplog.records)


def test_invalid_citation_logs_never_contain_transcript_text(resolver, transcript, caplog):
    with caplog.at_level(logging.WARNING):
        resolve(resolver, transcript, generated("m1:u999", "m1:u4"), allowed=["m1:u0"])

    messages = " ".join(r.getMessage() for r in caplog.records)
    for utterance in transcript.utterances:
        assert utterance.text not in messages
        assert utterance.speaker not in messages


def test_an_answer_with_no_valid_citations_is_logged(resolver, transcript, caplog):
    with caplog.at_level(logging.WARNING):
        resolve(resolver, transcript, generated("m1:u999"))

    assert "answer.no_valid_citations" in " ".join(r.getMessage() for r in caplog.records)


def test_a_partially_valid_answer_is_not_flagged_as_uncited(resolver, transcript, caplog):
    with caplog.at_level(logging.WARNING):
        resolve(resolver, transcript, generated("m1:u3", "m1:u999"))

    assert "answer.no_valid_citations" not in " ".join(r.getMessage() for r in caplog.records)


def test_valid_citations_produce_no_warnings(resolver, transcript, caplog):
    with caplog.at_level(logging.WARNING):
        resolve(resolver, transcript, generated("m1:u2", "m1:u3"))

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
