"""FakeLLMProvider: deterministic, recording, offline."""

import pytest

from app.adapters.llm.base import LLMProvider, LLMProviderError
from app.adapters.llm.fake import DEFAULT_ANSWER, FakeLLMProvider
from app.domain.models import GeneratedAnswer, GeneratedCitation

pytestmark = pytest.mark.anyio


def call_args(**overrides):
    args = {
        "question": "What was decided about the budget?",
        "context": "[m1:u0 | 00:00:12 | Sarah]\nThe budget is unchanged.",
        "allowed_utterance_ids": ["m1:u0"],
    }
    args.update(overrides)
    return args


def test_satisfies_the_llm_provider_protocol():
    assert isinstance(FakeLLMProvider(), LLMProvider)


async def test_returns_the_default_answer_when_unconfigured():
    assert await FakeLLMProvider().generate_answer(**call_args()) == DEFAULT_ANSWER


async def test_returns_the_configured_response():
    configured = GeneratedAnswer(
        answer="The budget is unchanged.",
        citations=[GeneratedCitation(utterance_id="m1:u3")],
        insufficient_evidence=False,
    )

    assert await FakeLLMProvider(configured).generate_answer(**call_args()) is configured


async def test_is_deterministic_across_calls():
    provider = FakeLLMProvider()

    first = await provider.generate_answer(**call_args())
    second = await provider.generate_answer(**call_args())

    assert first == second


async def test_records_the_question():
    provider = FakeLLMProvider()

    await provider.generate_answer(**call_args(question="Who owns the launch plan?"))

    assert provider.last_call is not None
    assert provider.last_call.question == "Who owns the launch plan?"


async def test_records_the_context():
    provider = FakeLLMProvider()
    context = "[m1:u1 | 00:00:31 | John]\nAgreed."

    await provider.generate_answer(**call_args(context=context))

    assert provider.last_call is not None
    assert provider.last_call.context == context


async def test_records_the_allowed_utterance_ids():
    provider = FakeLLMProvider()

    await provider.generate_answer(**call_args(allowed_utterance_ids=["m1:u0", "m1:u2"]))

    assert provider.last_call is not None
    assert provider.last_call.allowed_utterance_ids == ["m1:u0", "m1:u2"]


async def test_recorded_allowed_ids_are_a_copy():
    """Mutating the caller's list must not rewrite the recorded call."""
    provider = FakeLLMProvider()
    ids = ["m1:u0"]

    await provider.generate_answer(**call_args(allowed_utterance_ids=ids))
    ids.append("m1:u9")

    assert provider.last_call is not None
    assert provider.last_call.allowed_utterance_ids == ["m1:u0"]


async def test_counts_calls():
    provider = FakeLLMProvider()

    assert provider.call_count == 0
    await provider.generate_answer(**call_args())
    await provider.generate_answer(**call_args())

    assert provider.call_count == 2
    assert len(provider.calls) == 2


def test_last_call_is_none_before_any_call():
    assert FakeLLMProvider().last_call is None


async def test_can_simulate_provider_failure():
    provider = FakeLLMProvider(error=LLMProviderError("provider unavailable"))

    with pytest.raises(LLMProviderError):
        await provider.generate_answer(**call_args())


async def test_failure_is_still_recorded():
    """A failed call is observable, so tests can assert it was attempted."""
    provider = FakeLLMProvider(error=LLMProviderError("boom"))

    with pytest.raises(LLMProviderError):
        await provider.generate_answer(**call_args())

    assert provider.call_count == 1


async def test_can_simulate_an_arbitrary_exception():
    provider = FakeLLMProvider(error=RuntimeError("unexpected"))

    with pytest.raises(RuntimeError):
        await provider.generate_answer(**call_args())


async def test_can_return_an_insufficient_evidence_answer():
    configured = GeneratedAnswer(
        answer="The meeting does not cover that.", citations=[], insufficient_evidence=True
    )

    result = await FakeLLMProvider(configured).generate_answer(**call_args())

    assert result.insufficient_evidence is True
    assert result.citations == []


def test_the_fake_never_imports_a_client_library():
    """A subprocess probe: importing the fake must not pull in the SDK."""
    import subprocess
    import sys

    probe = (
        "import sys, app.adapters.llm.fake;"
        "print('anthropic' in sys.modules or 'httpx' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"
