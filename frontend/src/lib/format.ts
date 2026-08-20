/** Small pure helpers shared across components. */

/** Accepted audio extensions, mirroring the backend's allow-list. */
export const AUDIO_EXTENSIONS = [
  ".mp3",
  ".m4a",
  ".wav",
  ".webm",
  ".ogg",
  ".mp4",
  ".flac",
] as const;

export const AUDIO_ACCEPT = AUDIO_EXTENSIONS.join(",");

/** The transcript format the backend parser accepts. */
export const TRANSCRIPT_FORMAT_EXAMPLE = `[00:00:12] Sarah: We need to delay the release.
[00:00:31] John: Agreed.`;

/**
 * Build a URL- and DOM-safe meeting id from a human title.
 *
 * A short random suffix keeps ids unique without asking the user to invent
 * one, and without a collision round-trip to the backend.
 */
export function meetingIdFromTitle(title: string, suffix?: string): string {
  const slug = title
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);

  const tail = suffix ?? Math.random().toString(36).slice(2, 8);
  return slug ? `${slug}-${tail}` : `meeting-${tail}`;
}

/**
 * DOM id for an utterance element.
 *
 * Utterance ids contain a colon (`m1:u3`), which is legal in an `id` attribute
 * but awkward in CSS selectors, so it is replaced.
 */
export function utteranceDomId(utteranceId: string): string {
  return `utterance-${utteranceId.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
}

/** Render seconds as `MM:SS`, or `H:MM:SS` past an hour. */
export function formatDuration(totalSeconds: number | null | undefined): string {
  if (totalSeconds === null || totalSeconds === undefined || Number.isNaN(totalSeconds)) {
    return "—";
  }

  const seconds = Math.max(0, Math.floor(totalSeconds));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;

  const pad = (value: number) => value.toString().padStart(2, "0");
  return hours > 0 ? `${hours}:${pad(minutes)}:${pad(remainder)}` : `${minutes}:${pad(remainder)}`;
}

/** Human-readable file size for upload feedback. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Cheap client-side check so obviously wrong files fail before upload. */
export function hasAudioExtension(filename: string): boolean {
  const lower = filename.toLowerCase();
  return AUDIO_EXTENSIONS.some((extension) => lower.endsWith(extension));
}
