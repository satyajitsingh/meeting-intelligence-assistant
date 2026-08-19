"""AnthropicLLMProvider, with the SDK boundary stubbed.

No network: a stub client is injected through the constructor seam, so these
tests assert what the provider *sends* and how it handles what comes back.
"""

from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app.adapters.llm.anthropic import (
    DEFAULT_MODEL,
    EFFORT,
    MAX_TOKENS,
    AnthropicLLMProvider,
)
from app.adapters.llm.base import LLMProvider, LLMProviderError
from app.adapters.llm.prompts import SYSTEM_PROMPT
from app.domain.models import GeneratedAnswer, GeneratedCitation

pytestmark = pytest.mark.anyio

CONTEXT = (
    "[m1:u2 | 00:00:52 | Amir]\nWhat happens to the marketing budget?\n\n"
    "[m1:u3 | 00:01:14 | Sarah]\nThe budget is unchanged."
)
ALLOWED = ["m1:u2", "m1:u3"]

PARSED = GeneratedAnswer(
    answer="The budget is unchanged.",
    citations=[GeneratedCitation(utterance_id="m1:u3")],
    insufficient_evidence=False,
)


class StubMessages:
    """Captures parse() kwargs and returns a canned response."""

    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class StubClient:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.messages = StubMessages(response=response, error=error)


def ok_response(parsed=PARSED, stop_reason: str = "end_turn"):
    return SimpleNamespace(parsed_output=parsed, stop_reason=stop_reason)


def build(response=None, error: Exception | None = None, **kwargs):
    client = StubClient(response=response, error=error)
    provider = AnthropicLLMProvider(client=client, **kwargs)
    return provider, client


async def generate(provider):
    return await provider.generate_answer(
        question="What was decided about the marketing budget?",
        context=CONTEXT,
        allowed_utterance_ids=ALLOWED,
    )


# --- protocol --------------------------------------------------------------


def test_satisfies_the_llm_provider_protocol():
    provider, _ = build()

    assert isinstance(provider, LLMProvider)


# --- request shape ---------------------------------------------------------


def test_the_default_model_is_sonnet_class():
    assert DEFAULT_MODEL == "claude-sonnet-5"


async def test_uses_the_default_model_when_unconfigured():
    provider, client = build(response=ok_response())

    await generate(provider)

    assert client.messages.calls[0]["model"] == DEFAULT_MODEL
    assert provider.model == DEFAULT_MODEL


async def test_model_is_configurable():
    provider, client = build(response=ok_response(), model="claude-opus-5")

    await generate(provider)

    assert client.messages.calls[0]["model"] == "claude-opus-5"
    assert provider.model == "claude-opus-5"


async def test_sends_the_grounding_system_prompt():
    provider, client = build(response=ok_response())

    await generate(provider)

    assert client.messages.calls[0]["system"] == SYSTEM_PROMPT


async def test_includes_the_question_in_the_user_message():
    provider, client = build(response=ok_response())

    await generate(provider)

    content = client.messages.calls[0]["messages"][0]["content"]
    assert "What was decided about the marketing budget?" in content


async def test_includes_the_evidence_context_in_the_user_message():
    provider, client = build(response=ok_response())

    await generate(provider)

    content = client.messages.calls[0]["messages"][0]["content"]
    assert CONTEXT in content


async def test_includes_the_allowed_citation_ids_in_the_user_message():
    provider, client = build(response=ok_response())

    await generate(provider)

    content = client.messages.calls[0]["messages"][0]["content"]
    assert "ALLOWED CITATION IDS" in content
    for utterance_id in ALLOWED:
        assert utterance_id in content


async def test_requests_structured_output_bound_to_the_answer_schema():
    provider, client = build(response=ok_response())

    await generate(provider)

    assert client.messages.calls[0]["output_format"] is GeneratedAnswer


async def test_sends_effort_and_max_tokens():
    provider, client = build(response=ok_response())

    await generate(provider)

    call = client.messages.calls[0]
    assert call["output_config"] == {"effort": EFFORT}
    assert call["max_tokens"] == MAX_TOKENS


async def test_does_not_send_sampling_parameters():
    """Claude Sonnet 5 removed temperature/top_p/top_k; sending them is a 400."""
    provider, client = build(response=ok_response())

    await generate(provider)

    call = client.messages.calls[0]
    assert "temperature" not in call
    assert "top_p" not in call
    assert "top_k" not in call


async def test_applies_the_configured_timeout():
    provider, client = build(response=ok_response(), timeout_seconds=12.5)

    await generate(provider)

    assert client.messages.calls[0]["timeout"] == 12.5


async def test_sends_exactly_one_user_message():
    provider, client = build(response=ok_response())

    await generate(provider)

    messages = client.messages.calls[0]["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


# --- response handling -----------------------------------------------------


async def test_returns_the_parsed_structured_response():
    provider, _ = build(response=ok_response())

    assert await generate(provider) == PARSED


async def test_accepts_a_mapping_the_sdk_has_not_yet_validated():
    payload = {
        "answer": "The budget is unchanged.",
        "citations": [{"utterance_id": "m1:u3"}],
        "insufficient_evidence": False,
    }
    provider, _ = build(response=ok_response(parsed=payload))

    result = await generate(provider)

    assert result.answer == "The budget is unchanged."
    assert [c.utterance_id for c in result.citations] == ["m1:u3"]


async def test_missing_structured_output_raises_provider_error():
    provider, _ = build(response=ok_response(parsed=None))

    with pytest.raises(LLMProviderError) as exc_info:
        await generate(provider)

    assert "structured output" in exc_info.value.message


async def test_malformed_structured_output_raises_provider_error():
    provider, _ = build(response=ok_response(parsed={"unexpected": "shape"}))

    with pytest.raises(LLMProviderError) as exc_info:
        await generate(provider)

    assert exc_info.value.status_code == 502


async def test_a_refusal_raises_provider_error():
    provider, _ = build(response=ok_response(stop_reason="refusal"))

    with pytest.raises(LLMProviderError) as exc_info:
        await generate(provider)

    assert exc_info.value.details is not None
    assert exc_info.value.details["stop_reason"] == "refusal"


# --- failure handling ------------------------------------------------------


async def test_sdk_exception_becomes_a_provider_error():
    provider, _ = build(error=RuntimeError("connection reset"))

    with pytest.raises(LLMProviderError) as exc_info:
        await generate(provider)

    assert exc_info.value.code == "llm_provider_error"
    assert exc_info.value.status_code == 502


async def test_timeout_becomes_a_provider_error():
    provider, _ = build(error=TimeoutError("timed out"))

    with pytest.raises(LLMProviderError) as exc_info:
        await generate(provider)

    assert exc_info.value.details is not None
    assert exc_info.value.details["reason"] == "TimeoutError"


async def test_the_original_exception_is_chained():
    original = RuntimeError("connection reset")
    provider, _ = build(error=original)

    with pytest.raises(LLMProviderError) as exc_info:
        await generate(provider)

    assert exc_info.value.__cause__ is original


# --- secret handling -------------------------------------------------------


SECRET = "sk-ant-super-secret-value"


async def test_api_key_never_appears_in_a_provider_error():
    provider, _ = build(
        error=RuntimeError(f"401 unauthorized for key {SECRET}"),
        api_key=SecretStr(SECRET),
    )

    with pytest.raises(LLMProviderError) as exc_info:
        await generate(provider)

    error = exc_info.value
    assert SECRET not in error.message
    assert SECRET not in str(error.details)
    assert SECRET not in str(error.to_response().model_dump())


async def test_underlying_exception_text_is_not_copied_into_the_error():
    """SDK errors can echo request headers, so their text is never reused."""
    provider, _ = build(error=RuntimeError(f"x-api-key: {SECRET}"))

    with pytest.raises(LLMProviderError) as exc_info:
        await generate(provider)

    assert SECRET not in exc_info.value.message


def test_the_api_key_is_held_as_a_secret():
    provider = AnthropicLLMProvider(api_key=SecretStr(SECRET))

    assert SECRET not in repr(provider._api_key)
    assert str(provider._api_key) == "**********"


# --- lazy construction -----------------------------------------------------


def test_no_client_is_created_at_construction():
    provider = AnthropicLLMProvider(api_key=SecretStr(SECRET))

    assert provider._client is None


async def test_a_missing_api_key_fails_only_when_generating():
    provider = AnthropicLLMProvider(api_key=None)

    with pytest.raises(LLMProviderError) as exc_info:
        await generate(provider)

    assert "ANTHROPIC_API_KEY" in exc_info.value.message


def test_constructing_without_a_key_does_not_raise():
    """Ingestion and retrieval must work on a deployment with no LLM key."""
    assert AnthropicLLMProvider(api_key=None) is not None
