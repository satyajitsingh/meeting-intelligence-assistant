"""Anthropic-backed answer generation.

The only real language-model provider in v1. Structured output is enforced by
the SDK's ``messages.parse`` helper, which constrains generation to the
:class:`~app.domain.models.GeneratedAnswer` schema and returns a validated
instance -- so there is no JSON parsing or repair code here.

Note on determinism: the current Claude models removed the sampling
parameters, and passing ``temperature`` returns a 400. Determinism comes from
constraining the output schema and keeping reasoning effort moderate, not from
sampling controls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import SecretStr, ValidationError

from app.adapters.llm.base import LLMProviderError
from app.adapters.llm.prompts import SYSTEM_PROMPT, build_user_message
from app.core.logging import get_logger
from app.domain.models import GeneratedAnswer

if TYPE_CHECKING:  # pragma: no cover - avoids importing the SDK at module load
    from anthropic import AsyncAnthropic
    from anthropic.types import OutputConfigParam

logger = get_logger(__name__)

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_TIMEOUT_SECONDS = 30.0

# Grounded extraction over a handful of short evidence blocks is not a
# reasoning-heavy task; medium effort keeps latency and cost sane while leaving
# adaptive thinking on. Sonnet 5 supports the full low..max effort range, so
# this needs no adjustment for the default model.
EFFORT: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
MAX_TOKENS = 8192


class AnthropicLLMProvider:
    """Generates grounded answers with Claude.

    The SDK client is created on first use, so importing the application,
    building the container and starting the server all make no network call and
    work with no API key present.
    """

    def __init__(
        self,
        *,
        api_key: SecretStr | None = None,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    async def generate_answer(
        self,
        *,
        question: str,
        context: str,
        allowed_utterance_ids: list[str],
    ) -> GeneratedAnswer:
        client = self._ensure_client()
        user_message = build_user_message(
            question=question,
            context=context,
            allowed_utterance_ids=allowed_utterance_ids,
        )

        try:
            response = await client.messages.parse(
                model=self._model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                output_format=GeneratedAnswer,
                output_config=self._output_config(),
                timeout=self._timeout_seconds,
            )
        except Exception as exc:
            # Deliberately no exception detail in the message: SDK errors can
            # echo request headers, and the API key must never reach a log line
            # or an error body.
            raise LLMProviderError(
                "The answer provider is unavailable.",
                details={"model": self._model, "reason": type(exc).__name__},
            ) from exc

        return self._extract_answer(response)

    @staticmethod
    def _output_config() -> OutputConfigParam:
        return {"effort": EFFORT}

    def _extract_answer(self, response: Any) -> GeneratedAnswer:
        """Pull the validated answer out of the SDK response."""
        if getattr(response, "stop_reason", None) == "refusal":
            raise LLMProviderError(
                "The answer provider declined to respond.",
                details={"model": self._model, "stop_reason": "refusal"},
            )

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise LLMProviderError(
                "The answer provider returned no structured output.",
                details={"model": self._model},
            )

        if isinstance(parsed, GeneratedAnswer):
            return parsed

        # Defensive: a future SDK version handing back a mapping rather than a
        # model instance should still fail loudly rather than leak an odd type.
        try:
            return GeneratedAnswer.model_validate(parsed)
        except ValidationError as exc:
            raise LLMProviderError(
                "The answer provider returned a malformed response.",
                details={"model": self._model},
            ) from exc

    def _ensure_client(self) -> AsyncAnthropic:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:  # pragma: no cover - dependency declared
                raise LLMProviderError("The anthropic package is not installed.") from exc

            if self._api_key is None:
                raise LLMProviderError(
                    "No Anthropic API key is configured. Set ANTHROPIC_API_KEY to "
                    "enable answer generation.",
                    details={"model": self._model},
                )

            self._client = AsyncAnthropic(api_key=self._api_key.get_secret_value())

        return self._client
