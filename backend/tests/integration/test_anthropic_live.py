"""Live Anthropic provider check.

Excluded from the default run: it needs ANTHROPIC_API_KEY and makes a real,
billable request. Run explicitly with::

    ANTHROPIC_API_KEY=sk-ant-... pytest -m integration
"""

import os

import pytest
from pydantic import SecretStr

from app.adapters.llm.anthropic import AnthropicLLMProvider

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

CONTEXT = (
    "[m1:u2 | 00:00:52 | Amir]\nWhat happens to the marketing budget?\n\n"
    "[m1:u3 | 00:01:14 | Sarah]\nThe budget is unchanged, only the launch date moves."
)
ALLOWED = ["m1:u2", "m1:u3"]


@pytest.fixture
def provider():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY is not set")
    return AnthropicLLMProvider(api_key=SecretStr(api_key))


async def test_answers_from_the_supplied_evidence(provider):
    result = await provider.generate_answer(
        question="What was decided about the marketing budget?",
        context=CONTEXT,
        allowed_utterance_ids=ALLOWED,
    )

    assert result.answer
    assert result.insufficient_evidence is False
    assert result.citations
    # The core grounding contract: no invented identifiers.
    assert all(c.utterance_id in ALLOWED for c in result.citations)


async def test_reports_insufficient_evidence_rather_than_guessing(provider):
    result = await provider.generate_answer(
        question="How many engineers were hired last quarter?",
        context=CONTEXT,
        allowed_utterance_ids=ALLOWED,
    )

    assert result.insufficient_evidence is True
    assert result.citations == []
