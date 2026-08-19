"""Deterministic offline language-model provider for tests."""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.models import GeneratedAnswer


@dataclass(frozen=True)
class LLMCall:
    """One recorded invocation, for assertions."""

    question: str
    context: str
    allowed_utterance_ids: list[str]


DEFAULT_ANSWER = GeneratedAnswer(
    answer="A fake grounded answer.",
    citations=[],
    insufficient_evidence=False,
)


class FakeLLMProvider:
    """Returns a configured answer and records what it was asked.

    Set ``error`` to make every call raise, which is how provider-failure
    handling is exercised without touching the network.
    """

    def __init__(
        self,
        response: GeneratedAnswer | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.response = response if response is not None else DEFAULT_ANSWER
        self.error = error
        self.calls: list[LLMCall] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_call(self) -> LLMCall | None:
        return self.calls[-1] if self.calls else None

    async def generate_answer(
        self,
        *,
        question: str,
        context: str,
        allowed_utterance_ids: list[str],
    ) -> GeneratedAnswer:
        self.calls.append(
            LLMCall(
                question=question,
                context=context,
                allowed_utterance_ids=list(allowed_utterance_ids),
            )
        )

        if self.error is not None:
            raise self.error

        return self.response
