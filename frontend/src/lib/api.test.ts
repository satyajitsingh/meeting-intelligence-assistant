import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, apiBaseUrl, friendlyMessage } from "./api";

const BASE = apiBaseUrl();

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("apiBaseUrl", () => {
  it("has a localhost default and no trailing slash", () => {
    expect(BASE).toBe("http://localhost:8000");
  });
});

describe("listMeetings", () => {
  it("requests the transcripts collection", async () => {
    fetchMock.mockResolvedValue(jsonResponse([]));

    await api.listMeetings();

    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/api/transcripts`, undefined);
  });

  it("returns the parsed list", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse([
        {
          meeting_id: "m1",
          title: "Release planning",
          speakers: ["Sarah"],
          utterance_count: 5,
          duration_seconds: 98,
        },
      ]),
    );

    const meetings = await api.listMeetings();

    expect(meetings[0].meeting_id).toBe("m1");
  });
});

describe("ingestTranscript", () => {
  it("posts snake_case fields the backend expects", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ meeting_id: "m1" }));

    await api.ingestTranscript({ meetingId: "m1", title: "T", transcript: "[00:01] A: hi" });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/api/transcripts`);
    expect(JSON.parse(init.body)).toEqual({
      meeting_id: "m1",
      title: "T",
      transcript: "[00:01] A: hi",
    });
  });
});

describe("askQuestion", () => {
  it("posts the meeting, question and k", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ citations: [] }));

    await api.askQuestion({ meetingId: "m1", question: "What was decided?" });

    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({
      meeting_id: "m1",
      question: "What was decided?",
      k: 5,
    });
  });
});

describe("transcribeAudio", () => {
  it("sends multipart form data without a content-type header", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ text: "hi" }));

    await api.transcribeAudio(new Blob(["audio"]), "meeting.webm");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/api/transcriptions`);
    expect(init.body).toBeInstanceOf(FormData);
    // The browser must set the multipart boundary itself.
    expect(init.headers).toBeUndefined();
  });
});

describe("deleteMeeting", () => {
  it("tolerates an empty 204 response", async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }));

    await expect(api.deleteMeeting("m1")).resolves.toBeUndefined();
  });

  it("encodes the meeting id", async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }));

    await api.deleteMeeting("a b/c");

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/api/transcripts/a%20b%2Fc`);
  });
});

describe("error handling", () => {
  it("turns the uniform error body into an ApiError", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(
        {
          error: "transcript_parse_error",
          message: "Line 2: Malformed utterance.",
          details: { line_number: 2 },
        },
        422,
      ),
    );

    await expect(api.listMeetings()).rejects.toMatchObject({
      code: "transcript_parse_error",
      status: 422,
      message: "Line 2: Malformed utterance.",
      details: { line_number: 2 },
    });
  });

  it("reports an unreachable backend rather than a fetch stack trace", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));

    const error = await api.listMeetings().catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(ApiError);
    expect(friendlyMessage(error)).toContain("Cannot reach the backend");
  });

  it("survives an error response with no json body", async () => {
    fetchMock.mockResolvedValue(new Response("nope", { status: 500 }));

    await expect(api.listMeetings()).rejects.toMatchObject({ status: 500 });
  });
});

describe("friendlyMessage", () => {
  it("prefers the backend message", () => {
    expect(friendlyMessage(new ApiError("Transcript not found.", { status: 404 }))).toBe(
      "Transcript not found.",
    );
  });

  it("falls back for unknown values", () => {
    expect(friendlyMessage("boom")).toBe("Something went wrong.");
  });
});
