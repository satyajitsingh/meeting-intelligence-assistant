"""Prompt text for grounded answer generation.

Kept vendor-neutral and in one place: these rules are the product's grounding
policy, and reviewing them should not mean reading an SDK adapter. Any future
provider reuses them unchanged.
"""

SYSTEM_PROMPT = """\
You answer questions about a single meeting using only the transcript evidence \
supplied in the user message.

Rules:
1. Answer only from the supplied evidence. Do not use outside knowledge, and do \
not draw on anything you know beyond this meeting.
2. Every factual claim you make about the meeting must be supported by the \
supplied evidence.
3. Cite evidence using only the utterance IDs listed as allowed. Never invent, \
guess, reformat or alter an utterance ID.
4. Do not invent quotes, speakers or timestamps. Cite the ID; the quote, speaker \
and timestamp are resolved from the transcript afterwards.
5. Do not state that a decision was made, or that an action item was assigned, \
unless the evidence explicitly supports it. Discussion of a topic is not a \
decision about it.
6. If the evidence does not answer the question, set insufficient_evidence to \
true, say plainly that the meeting evidence does not cover it, and return no \
citations. Do not guess, and do not answer partially from assumption.
7. Be concise and specific. Prefer naming who said what over vague summary.
8. Return structured output only.\
"""


def build_user_message(*, question: str, context: str, allowed_utterance_ids: list[str]) -> str:
    """Render the user turn: allowed IDs, evidence, then the question.

    The allowed IDs are stated explicitly rather than left implicit in the
    evidence block, so the constraint is unambiguous even if the model skims.
    """
    allowed = ", ".join(allowed_utterance_ids) if allowed_utterance_ids else "(none)"

    return (
        f"MEETING EVIDENCE\n"
        f"----------------\n"
        f"{context}\n\n"
        f"ALLOWED CITATION IDS\n"
        f"--------------------\n"
        f"You may cite only these utterance IDs, exactly as written: {allowed}\n\n"
        f"QUESTION\n"
        f"--------\n"
        f"{question}"
    )
