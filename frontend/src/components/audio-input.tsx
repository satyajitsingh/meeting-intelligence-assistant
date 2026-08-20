"use client";

import { useRef, useState } from "react";
import { Recorder } from "@/components/recorder";
import { Button, Spinner } from "@/components/ui";
import { AUDIO_ACCEPT, AUDIO_EXTENSIONS, formatBytes, hasAudioExtension } from "@/lib/format";

export function AudioInput({
  pending,
  onFile,
  onInvalidFile,
}: {
  pending: boolean;
  onFile: (file: File | Blob, filename: string) => void;
  onInvalidFile: (message: string) => void;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);

  function accept(file: File) {
    if (!hasAudioExtension(file.name)) {
      // Fail fast on the obvious case; the backend remains the real authority.
      onInvalidFile(
        `“${file.name}” is not a supported audio file. Use ${AUDIO_EXTENSIONS.join(", ")}.`,
      );
      return;
    }
    setSelected(`${file.name} · ${formatBytes(file.size)}`);
    onFile(file, file.name);
  }

  if (pending) {
    return (
      <div className="rounded-lg border border-line bg-canvas px-4 py-8 text-center">
        <p className="flex items-center justify-center gap-2 text-sm font-medium text-ink">
          <Spinner label="Transcribing" />
          Transcribing meeting…
        </p>
        <p className="mt-1 text-xs text-muted">This can take a moment for longer recordings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          const file = event.dataTransfer.files?.[0];
          if (file) accept(file);
        }}
        className={`rounded-lg border border-dashed px-4 py-8 text-center transition-colors ${
          dragging ? "border-accent bg-accent-soft" : "border-line bg-canvas"
        }`}
      >
        <p className="text-sm text-ink">Drop an audio file here</p>
        <p className="mt-1 text-xs text-muted">{AUDIO_EXTENSIONS.join(" · ")}</p>

        <div className="mt-4">
          <Button variant="secondary" onClick={() => inputRef.current?.click()}>
            Choose file
          </Button>
        </div>

        <input
          ref={inputRef}
          id="audio-file"
          type="file"
          accept={AUDIO_ACCEPT}
          className="sr-only"
          aria-label="Audio file"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) accept(file);
            // Reset so choosing the same file twice still fires a change event.
            event.target.value = "";
          }}
        />

        {selected ? <p className="mt-3 text-xs text-muted">Selected: {selected}</p> : null}
      </div>

      <div className="border-t border-line pt-4">
        <p className="mb-2 text-sm font-medium text-ink">Or record now</p>
        <Recorder disabled={pending} onRecorded={onFile} />
      </div>
    </div>
  );
}
