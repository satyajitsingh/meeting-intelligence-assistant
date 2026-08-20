"use client";

import type { TranscriptionResult } from "@/lib/api";
import { Button } from "@/components/ui";
import { TRANSCRIPT_FORMAT_EXAMPLE, formatDuration } from "@/lib/format";

export function TranscriptEditor({
  value,
  onChange,
  onIngest,
  pending,
  source,
}: {
  value: string;
  onChange: (value: string) => void;
  onIngest: () => void;
  pending: boolean;
  source: TranscriptionResult | null;
}) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-baseline justify-between gap-4">
        <label htmlFor="transcript" className="text-sm font-medium text-ink">
          Transcript
        </label>
        <p className="text-xs text-muted">
          Review speaker names and timestamps before ingesting.
        </p>
      </div>

      {source ? (
        <p className="rounded-lg border border-line bg-canvas px-3 py-2 text-xs text-muted">
          Transcribed <span className="font-medium text-ink">{source.filename}</span>
          {source.language ? ` · ${source.language.toUpperCase()}` : ""}
          {source.duration_seconds ? ` · ${formatDuration(source.duration_seconds)}` : ""}
        </p>
      ) : null}

      <textarea
        id="transcript"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        disabled={pending}
        rows={16}
        spellCheck={false}
        placeholder={TRANSCRIPT_FORMAT_EXAMPLE}
        aria-describedby="transcript-format"
        className="w-full rounded-lg border border-line bg-surface p-3.5 font-mono text-[13px] leading-relaxed placeholder:text-muted/60 disabled:opacity-60"
      />

      <div
        id="transcript-format"
        className="rounded-lg border border-line bg-canvas px-3.5 py-3 text-xs text-muted"
      >
        <p className="font-medium text-ink">Required format</p>
        <pre className="mt-1.5 whitespace-pre-wrap font-mono text-[12px] leading-relaxed">
          {TRANSCRIPT_FORMAT_EXAMPLE}
        </pre>
        <p className="mt-2">
          One line per turn: a timestamp in brackets, the speaker, then a colon. Lines without a
          timestamp continue the previous turn.
        </p>
      </div>

      <div className="flex justify-end">
        <Button onClick={onIngest} pending={pending} disabled={!value.trim()}>
          {pending ? "Ingesting…" : "Ingest meeting"}
        </Button>
      </div>
    </div>
  );
}
