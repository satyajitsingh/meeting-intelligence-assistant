"use client";

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { Button } from "@/components/ui";
import { formatDuration } from "@/lib/format";

type RecorderState = "idle" | "recording" | "denied";

/** True when this browser can record audio at all. */
export function detectRecordingSupport(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.MediaRecorder !== "undefined" &&
    Boolean(navigator.mediaDevices?.getUserMedia)
  );
}

// Support never changes during a session, so the subscribe function is a no-op.
const noopSubscribe = () => () => {};

export function Recorder({
  disabled,
  onRecorded,
}: {
  disabled: boolean;
  onRecorded: (blob: Blob, filename: string) => void;
}) {
  // Read through useSyncExternalStore rather than an effect: it keeps the
  // browser-only check out of render while still giving the server a stable
  // snapshot, so there is no hydration mismatch and no setState-in-effect.
  const supported = useSyncExternalStore(noopSubscribe, detectRecordingSupport, () => true);

  const [state, setState] = useState<RecorderState>("idle");
  const [seconds, setSeconds] = useState(0);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    if (state !== "recording") return;
    const timer = setInterval(() => setSeconds((value) => value + 1), 1000);
    return () => clearInterval(timer);
  }, [state]);

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }, []);

  // Release the microphone if the component unmounts mid-recording.
  useEffect(() => stopStream, [stopStream]);

  async function start() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];

      const recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        stopStream();
        setState("idle");
        setSeconds(0);
        // A zero-length blob means the recording was cancelled before any audio
        // arrived; sending it would just earn a 422.
        if (blob.size > 0) {
          onRecorded(blob, `meeting-recording-${Date.now()}.webm`);
        }
      };

      recorderRef.current = recorder;
      recorder.start();
      setSeconds(0);
      setState("recording");
    } catch {
      stopStream();
      setState("denied");
    }
  }

  function stop() {
    recorderRef.current?.stop();
    recorderRef.current = null;
  }

  if (!supported) {
    return (
      <p className="text-sm text-muted">
        This browser cannot record audio. Upload an audio file instead.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-3">
        {state === "recording" ? (
          <Button variant="secondary" onClick={stop}>
            Stop recording
          </Button>
        ) : (
          <Button variant="secondary" onClick={start} disabled={disabled}>
            Start recording
          </Button>
        )}

        {/* State is conveyed by text, not by the dot alone. */}
        {state === "recording" ? (
          <p className="flex items-center gap-2 text-sm font-medium text-red-600">
            <span aria-hidden="true" className="size-2 animate-pulse rounded-full bg-red-600" />
            Recording · {formatDuration(seconds)}
          </p>
        ) : null}
      </div>

      {state === "denied" ? (
        <p className="text-sm text-red-700">
          Microphone access was blocked. Allow it in your browser settings, or upload a file
          instead.
        </p>
      ) : (
        <p className="text-xs text-muted">
          Records in your browser and sends the audio for transcription when you stop.
        </p>
      )}
    </div>
  );
}
