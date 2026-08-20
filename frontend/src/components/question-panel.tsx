"use client";

import { useState } from "react";
import type { AnswerResult } from "@/lib/api";
import { AnswerCard } from "@/components/answer-card";
import { Button, EmptyHint, SectionHeading, Spinner } from "@/components/ui";

const SUGGESTIONS = [
  "What decisions were made?",
  "What action items were assigned?",
  "What remains unresolved?",
];

export function QuestionPanel({
  answer,
  pending,
  onAsk,
  onViewEvidence,
}: {
  answer: AnswerResult | null;
  pending: boolean;
  onAsk: (question: string) => void;
  onViewEvidence: (utteranceId: string) => void;
}) {
  const [question, setQuestion] = useState("");
  const trimmed = question.trim();

  function submit() {
    if (!trimmed || pending) return;
    onAsk(trimmed);
  }

  return (
    <div className="flex flex-col gap-5">
      <div>
        <label htmlFor="question" className="block text-sm font-medium text-ink">
          Ask this meeting
        </label>

        <div className="mt-2 flex gap-2">
          <input
            id="question"
            type="text"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                submit();
              }
            }}
            placeholder="What was decided about the marketing budget?"
            disabled={pending}
            className="flex-1 rounded-lg border border-line bg-surface px-3.5 py-2.5 text-sm placeholder:text-muted/70 disabled:opacity-60"
          />
          <Button onClick={submit} pending={pending} disabled={!trimmed}>
            {pending ? "Asking…" : "Ask"}
          </Button>
        </div>

        {!answer && !pending ? (
          <div className="mt-3 flex flex-wrap gap-2">
            {SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                onClick={() => setQuestion(suggestion)}
                className="rounded-full border border-line px-3 py-1 text-xs text-muted transition-colors hover:border-accent/40 hover:text-accent"
              >
                {suggestion}
              </button>
            ))}
          </div>
        ) : null}
      </div>

      {pending ? (
        <p className="flex items-center gap-2 text-sm text-muted">
          <Spinner label="Finding evidence" />
          Searching the transcript for evidence…
        </p>
      ) : answer ? (
        <AnswerCard answer={answer} onViewEvidence={onViewEvidence} />
      ) : (
        <div>
          <SectionHeading>Answer</SectionHeading>
          <div className="mt-1.5">
            <EmptyHint>
              Ask a question and the answer will appear here, with the exact transcript lines
              that support it.
            </EmptyHint>
          </div>
        </div>
      )}
    </div>
  );
}
