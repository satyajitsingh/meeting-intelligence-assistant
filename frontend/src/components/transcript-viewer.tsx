"use client";

import type { TranscriptDetail } from "@/lib/api";
import { utteranceDomId } from "@/lib/format";

export function TranscriptViewer({ transcript }: { transcript: TranscriptDetail }) {
  return (
    <ol className="flex flex-col divide-y divide-line">
      {transcript.utterances.map((utterance) => (
        <li
          key={utterance.id}
          // Stable, sanitised target for the evidence "view in transcript" jump.
          id={utteranceDomId(utterance.id)}
          data-utterance-id={utterance.id}
          className="scroll-mt-24 rounded-md px-4 py-3"
        >
          <p className="flex items-baseline gap-2 text-xs">
            <time className="font-mono text-muted">{utterance.display_timestamp}</time>
            <span className="font-semibold text-ink">{utterance.speaker}</span>
          </p>
          <p className="mt-1 text-sm leading-relaxed text-ink">{utterance.text}</p>
        </li>
      ))}
    </ol>
  );
}
