"use client";

import type { TranscriptionResult } from "@/lib/api";
import { AudioInput } from "@/components/audio-input";
import { TranscriptEditor } from "@/components/transcript-editor";

export type InputMethod = "paste" | "upload";

const METHODS: { id: InputMethod; label: string }[] = [
  { id: "paste", label: "Paste transcript" },
  { id: "upload", label: "Upload or record audio" },
];

export function NewMeeting({
  title,
  method,
  transcript,
  transcription,
  transcribing,
  ingesting,
  onTitleChange,
  onMethodChange,
  onTranscriptChange,
  onFile,
  onInvalidFile,
  onIngest,
  onCancel,
}: {
  title: string;
  method: InputMethod;
  transcript: string;
  transcription: TranscriptionResult | null;
  transcribing: boolean;
  ingesting: boolean;
  onTitleChange: (value: string) => void;
  onMethodChange: (method: InputMethod) => void;
  onTranscriptChange: (value: string) => void;
  onFile: (file: File | Blob, filename: string) => void;
  onInvalidFile: (message: string) => void;
  onIngest: () => void;
  onCancel: () => void;
}) {
  // Once audio has produced text, the review step is what matters, so the
  // editor replaces the picker rather than sitting below it.
  const showEditor = method === "paste" || transcript.length > 0;

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-8 py-8">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">New meeting</h2>
          <p className="mt-1 text-sm text-muted">
            Add a transcript directly, or transcribe a recording and review it first.
          </p>
        </div>
        <button
          type="button"
          onClick={onCancel}
          className="text-sm text-muted transition-colors hover:text-ink"
        >
          Cancel
        </button>
      </header>

      <div>
        <label htmlFor="meeting-title" className="block text-sm font-medium text-ink">
          Meeting title
        </label>
        <input
          id="meeting-title"
          type="text"
          value={title}
          onChange={(event) => onTitleChange(event.target.value)}
          placeholder="Release planning"
          className="mt-2 w-full rounded-lg border border-line bg-surface px-3.5 py-2.5 text-sm placeholder:text-muted/70"
        />
      </div>

      <div role="group" aria-label="Transcript source" className="flex gap-2">
        {METHODS.map((item) => (
          <button
            key={item.id}
            type="button"
            onClick={() => onMethodChange(item.id)}
            aria-pressed={method === item.id}
            className={`rounded-lg border px-3.5 py-2 text-sm font-medium transition-colors ${
              method === item.id
                ? "border-accent bg-accent-soft text-accent"
                : "border-line bg-surface text-muted hover:text-ink"
            }`}
          >
            {item.label}
          </button>
        ))}
      </div>

      {method === "upload" && !showEditor ? (
        <AudioInput pending={transcribing} onFile={onFile} onInvalidFile={onInvalidFile} />
      ) : null}

      {method === "upload" && transcribing ? (
        <AudioInput pending onFile={onFile} onInvalidFile={onInvalidFile} />
      ) : null}

      {showEditor && !transcribing ? (
        <TranscriptEditor
          value={transcript}
          onChange={onTranscriptChange}
          onIngest={onIngest}
          pending={ingesting}
          source={transcription}
        />
      ) : null}
    </div>
  );
}
