"""Language-model provider port.

A provider receives a question, a prepared evidence context and the exact set
of utterance IDs it is permitted to cite, and returns a structured answer. No
vendor type crosses this boundary: the service never sees an SDK message,
stop reason or token count.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Protocol, runtime_checkable

from app.core.errors import AppError
from app.domain.models import GeneratedAnswer


class LLMProviderError(AppError):
    """The language-model provider failed or returned something unusable.

    Modelled as a bad gateway: the request was fine, an upstream dependency
    was not.
    """

    code = "llm_provider_error"
    status_code = HTTPStatus.BAD_GATEWAY


@runtime_checkable
class LLMProvider(Protocol):
    """Generates a grounded, structured answer from supplied evidence."""

    async def generate_answer(
        self,
        *,
        question: str,
        context: str,
        allowed_utterance_ids: list[str],
    ) -> GeneratedAnswer:
        """Answer ``question`` using only ``context``.

        Implementations must constrain output to the
        :class:`~app.domain.models.GeneratedAnswer` schema and must tell the
        model that ``allowed_utterance_ids`` are the only citable identifiers.

        Raises:
            LLMProviderError: the provider failed, timed out, or returned a
                response that did not satisfy the schema.
        """
        ...
