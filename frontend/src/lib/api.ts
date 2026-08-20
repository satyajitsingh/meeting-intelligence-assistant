/**
 * Typed client for the Meeting Intelligence backend.
 *
 * Every network call lives here so components never touch `fetch`, and the
 * base URL is resolved in exactly one place.
 */

const DEFAULT_BASE_URL = "http://localhost:8000";

export function apiBaseUrl(): string {
  return (process.env.NEXT_PUBLIC_API_BASE_URL ?? DEFAULT_BASE_URL).replace(/\/$/, "");
}

/* -------------------------------------------------------------------------- */
/* Backend types                                                              */
/* -------------------------------------------------------------------------- */

export interface MeetingSummary {
  meeting_id: string;
  title: string;
  speakers: string[];
  utterance_count: number;
  duration_seconds: number;
}

export interface IngestedMeeting extends MeetingSummary {
  chunk_count: number;
}

export interface Utterance {
  id: string;
  index: number;
  speaker: string;
  start_seconds: number;
  raw_timestamp: string;
  display_timestamp: string;
  text: string;
}

export interface TranscriptDetail {
  meeting_id: string;
  title: string;
  speakers: string[];
  duration_seconds: number;
  utterances: Utterance[];
}

export interface TranscriptionResult {
  text: string;
  language: string | null;
  duration_seconds: number | null;
  filename: string;
}

/** A citation resolved from the transcript — never model-generated text. */
export interface ResolvedCitation {
  utterance_id: string;
  speaker: string;
  timestamp: string;
  start_seconds: number;
  quote: string;
}

export interface AnswerResult {
  meeting_id: string;
  question: string;
  answer: string;
  citations: ResolvedCitation[];
  insufficient_evidence: boolean;
}

/** The backend's uniform error body. */
interface ErrorBody {
  error: string;
  message: string;
  details: Record<string, unknown> | null;
}

/* -------------------------------------------------------------------------- */
/* Errors                                                                     */
/* -------------------------------------------------------------------------- */

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly details: Record<string, unknown> | null;

  constructor(
    message: string,
    options: { code?: string; status?: number; details?: Record<string, unknown> | null } = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.code = options.code ?? "unknown_error";
    this.status = options.status ?? 0;
    this.details = options.details ?? null;
  }
}

/**
 * Turn a failure into something worth showing a person.
 *
 * The backend's `message` is already human-readable, so it is preferred; the
 * fallbacks exist for transport failures, where there is no body at all.
 */
export function friendlyMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 0) {
      return "Cannot reach the backend. Check that the API is running.";
    }
    return error.message;
  }
  if (error instanceof Error) return error.message;
  return "Something went wrong.";
}

/* -------------------------------------------------------------------------- */
/* Transport                                                                  */
/* -------------------------------------------------------------------------- */

async function toApiError(response: Response): Promise<ApiError> {
  let body: Partial<ErrorBody> | null = null;
  try {
    body = (await response.json()) as Partial<ErrorBody>;
  } catch {
    body = null;
  }

  return new ApiError(body?.message ?? `Request failed (${response.status}).`, {
    code: body?.error,
    status: response.status,
    details: body?.details ?? null,
  });
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl()}${path}`, init);
  } catch {
    // Network-level failure: no status, no body.
    throw new ApiError("Cannot reach the backend. Check that the API is running.");
  }

  if (!response.ok) throw await toApiError(response);
  if (response.status === 204) return undefined as T;

  return (await response.json()) as T;
}

function jsonInit(body: unknown): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

/* -------------------------------------------------------------------------- */
/* Endpoints                                                                  */
/* -------------------------------------------------------------------------- */

export const api = {
  listMeetings(): Promise<MeetingSummary[]> {
    return request<MeetingSummary[]>("/api/transcripts");
  },

  getTranscript(meetingId: string): Promise<TranscriptDetail> {
    return request<TranscriptDetail>(`/api/transcripts/${encodeURIComponent(meetingId)}`);
  },

  ingestTranscript(input: {
    meetingId: string;
    title: string;
    transcript: string;
  }): Promise<IngestedMeeting> {
    return request<IngestedMeeting>(
      "/api/transcripts",
      jsonInit({
        meeting_id: input.meetingId,
        title: input.title,
        transcript: input.transcript,
      }),
    );
  },

  deleteMeeting(meetingId: string): Promise<void> {
    return request<void>(`/api/transcripts/${encodeURIComponent(meetingId)}`, {
      method: "DELETE",
    });
  },

  transcribeAudio(file: File | Blob, filename: string): Promise<TranscriptionResult> {
    const form = new FormData();
    form.append("file", file, filename);
    // No Content-Type header: the browser must set the multipart boundary.
    return request<TranscriptionResult>("/api/transcriptions", {
      method: "POST",
      body: form,
    });
  },

  askQuestion(input: {
    meetingId: string;
    question: string;
    k?: number;
  }): Promise<AnswerResult> {
    return request<AnswerResult>(
      "/api/answers",
      jsonInit({
        meeting_id: input.meetingId,
        question: input.question,
        k: input.k ?? 5,
      }),
    );
  },
};

export type Api = typeof api;
