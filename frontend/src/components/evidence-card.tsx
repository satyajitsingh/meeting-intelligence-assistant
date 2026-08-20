"use client";

import type { ResolvedCitation } from "@/lib/api";

export function EvidenceCard({
  citation,
  index,
  onView,
}: {
  citation: ResolvedCitation;
  index: number;
  onView: (utteranceId: string) => void;
}) {
  return (
    <li className="rounded-lg border border-line bg-canvas p-4">
      <div className="flex items-baseline justify-between gap-3">
        <p className="text-xs font-semibold uppercase tracking-wider text-muted">
          Evidence {index + 1}
        </p>
        <p className="text-xs text-muted">
          <span className="font-medium text-ink">{citation.speaker}</span>
          <span aria-hidden="true"> · </span>
          <time className="font-mono">{citation.timestamp}</time>
        </p>
      </div>

      {/* Verbatim transcript text, resolved by the backend — never generated. */}
      <blockquote className="mt-2 border-l-2 border-line pl-3 text-sm leading-relaxed text-ink">
        “{citation.quote}”
      </blockquote>

      <button
        type="button"
        onClick={() => onView(citation.utterance_id)}
        className="mt-3 text-xs font-medium text-accent hover:underline"
      >
        View in transcript
        <span className="sr-only"> — {citation.speaker} at {citation.timestamp}</span>
      </button>
    </li>
  );
}
