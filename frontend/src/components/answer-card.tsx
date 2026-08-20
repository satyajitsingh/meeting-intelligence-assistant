"use client";

import type { AnswerResult } from "@/lib/api";
import { EvidenceCard } from "@/components/evidence-card";
import { SectionHeading } from "@/components/ui";

type Trust = "grounded" | "insufficient" | "unverified";

/**
 * Three distinct states, deliberately separated.
 *
 * "Insufficient" means the model reported the meeting does not cover the
 * question. "Unverified" means it answered but nothing survived citation
 * validation — a different and more concerning thing, so it must not be
 * presented as a trusted answer.
 */
export function trustLevel(answer: AnswerResult): Trust {
  if (answer.insufficient_evidence) return "insufficient";
  return answer.citations.length > 0 ? "grounded" : "unverified";
}

const BADGES: Record<Trust, { label: string; className: string }> = {
  grounded: {
    label: "Grounded in meeting evidence",
    className: "border-emerald-200 bg-emerald-50 text-emerald-800",
  },
  insufficient: {
    label: "Not enough evidence in this meeting",
    className: "border-line bg-canvas text-muted",
  },
  unverified: {
    label: "Answer could not be verified against meeting evidence",
    className: "border-amber-200 bg-amber-50 text-amber-900",
  },
};

export function AnswerCard({
  answer,
  onViewEvidence,
}: {
  answer: AnswerResult;
  onViewEvidence: (utteranceId: string) => void;
}) {
  const trust = trustLevel(answer);
  const badge = BADGES[trust];

  return (
    <div className="flex flex-col gap-5">
      <div>
        <SectionHeading>Question</SectionHeading>
        <p className="mt-1.5 text-sm text-muted">{answer.question}</p>
      </div>

      <div>
        <SectionHeading>Answer</SectionHeading>
        <p className="mt-1.5 whitespace-pre-wrap text-[15px] leading-relaxed text-ink">
          {answer.answer}
        </p>

        <p
          data-trust={trust}
          className={`mt-3 inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium ${badge.className}`}
        >
          {badge.label}
        </p>
      </div>

      {answer.citations.length > 0 ? (
        <div>
          <SectionHeading hint={`${answer.citations.length} cited`}>Evidence</SectionHeading>
          <ul className="mt-2 flex flex-col gap-3">
            {answer.citations.map((citation, index) => (
              <EvidenceCard
                key={`${citation.utterance_id}-${index}`}
                citation={citation}
                index={index}
                onView={onViewEvidence}
              />
            ))}
          </ul>
        </div>
      ) : (
        <p className="text-sm leading-relaxed text-muted">
          {trust === "insufficient"
            ? "Nothing in this transcript answers that question. Try rephrasing, or ask about something the meeting covered."
            : "No transcript evidence supported this answer, so treat it with caution."}
        </p>
      )}
    </div>
  );
}
