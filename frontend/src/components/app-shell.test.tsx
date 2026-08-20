import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AnswerResult, TranscriptDetail } from "@/lib/api";

const listMeetings = vi.fn();
const getTranscript = vi.fn();
const ingestTranscript = vi.fn();
const deleteMeeting = vi.fn();
const askQuestion = vi.fn();
const transcribeAudio = vi.fn();

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      listMeetings: (...args: unknown[]) => listMeetings(...args),
      getTranscript: (...args: unknown[]) => getTranscript(...args),
      ingestTranscript: (...args: unknown[]) => ingestTranscript(...args),
      deleteMeeting: (...args: unknown[]) => deleteMeeting(...args),
      askQuestion: (...args: unknown[]) => askQuestion(...args),
      transcribeAudio: (...args: unknown[]) => transcribeAudio(...args),
    },
  };
});

const { AppShell } = await import("./app-shell");

const TRANSCRIPT_TEXT = `[00:00:12] Sarah: We need to delay the release because migration is unfinished.
[00:01:14] Sarah: The budget is unchanged.`;

const SUMMARY = {
  meeting_id: "release-planning-abc123",
  title: "Release planning",
  speakers: ["Sarah"],
  utterance_count: 2,
  duration_seconds: 74,
};

const DETAIL: TranscriptDetail = {
  meeting_id: SUMMARY.meeting_id,
  title: SUMMARY.title,
  speakers: ["Sarah"],
  duration_seconds: 74,
  utterances: [
    {
      id: "release-planning-abc123:u0",
      index: 0,
      speaker: "Sarah",
      start_seconds: 12,
      raw_timestamp: "00:00:12",
      display_timestamp: "00:00:12",
      text: "We need to delay the release because migration is unfinished.",
    },
    {
      id: "release-planning-abc123:u1",
      index: 1,
      speaker: "Sarah",
      start_seconds: 74,
      raw_timestamp: "00:01:14",
      display_timestamp: "00:01:14",
      text: "The budget is unchanged.",
    },
  ],
};

const GROUNDED: AnswerResult = {
  meeting_id: SUMMARY.meeting_id,
  question: "What was decided about the marketing budget?",
  answer: "The budget remains unchanged.",
  citations: [
    {
      utterance_id: "release-planning-abc123:u1",
      speaker: "Sarah",
      timestamp: "00:01:14",
      start_seconds: 74,
      quote: "The budget is unchanged.",
    },
  ],
  insufficient_evidence: false,
};

beforeEach(() => {
  vi.clearAllMocks();
  listMeetings.mockResolvedValue([]);
  getTranscript.mockResolvedValue(DETAIL);
  ingestTranscript.mockResolvedValue({ ...SUMMARY, chunk_count: 1 });
  askQuestion.mockResolvedValue(GROUNDED);
  deleteMeeting.mockResolvedValue(undefined);
});

describe("initial render", () => {
  it("shows the welcome state when there are no meetings", async () => {
    render(<AppShell />);

    expect(
      await screen.findByText("Ask questions about your meetings"),
    ).toBeInTheDocument();
  });

  it("lists meetings returned by the backend", async () => {
    listMeetings.mockResolvedValue([SUMMARY]);
    render(<AppShell />);

    expect(await screen.findByText("Release planning")).toBeInTheDocument();
    expect(screen.getByText(/2 utterances/)).toBeInTheDocument();
  });

  it("surfaces a readable error when the backend is unreachable", async () => {
    const { ApiError } = await import("@/lib/api");
    listMeetings.mockRejectedValue(new ApiError("Cannot reach the backend."));

    render(<AppShell />);

    expect(await screen.findByText(/cannot reach the backend/i)).toBeInTheDocument();
  });
});

describe("paste and ingest flow", () => {
  async function ingest(user: ReturnType<typeof userEvent.setup>) {
    await user.click(await screen.findByRole("button", { name: "New meeting" }));
    await user.type(screen.getByLabelText("Meeting title"), "Release planning");
    await user.click(screen.getByLabelText("Transcript"));
    await user.paste(TRANSCRIPT_TEXT);
    await user.click(screen.getByRole("button", { name: /ingest meeting/i }));
  }

  it("posts the transcript with a generated meeting id", async () => {
    const user = userEvent.setup();
    render(<AppShell />);

    await ingest(user);

    await waitFor(() => expect(ingestTranscript).toHaveBeenCalledOnce());
    const payload = ingestTranscript.mock.calls[0][0];
    expect(payload.title).toBe("Release planning");
    expect(payload.transcript).toContain("The budget is unchanged.");
    // Generated for the user, never typed by them.
    expect(payload.meetingId).toMatch(/^release-planning-[a-z0-9]+$/);
  });

  it("shows the required transcript format before ingesting", async () => {
    const user = userEvent.setup();
    render(<AppShell />);

    await user.click(await screen.findByRole("button", { name: "New meeting" }));

    expect(screen.getByText("Required format")).toBeInTheDocument();
    expect(
      screen.getByText(/Review speaker names and timestamps before ingesting/),
    ).toBeInTheDocument();
  });

  it("opens the ingested meeting and renders its utterances", async () => {
    const user = userEvent.setup();
    render(<AppShell />);

    await ingest(user);

    expect(await screen.findByText("The budget is unchanged.")).toBeInTheDocument();
    expect(getTranscript).toHaveBeenCalledWith(SUMMARY.meeting_id);
  });

  it("confirms success with the indexed counts", async () => {
    const user = userEvent.setup();
    render(<AppShell />);

    await ingest(user);

    expect(await screen.findByText(/2 utterances, 1 chunks/)).toBeInTheDocument();
  });

  it("surfaces a parser error without dumping raw json", async () => {
    const { ApiError } = await import("@/lib/api");
    ingestTranscript.mockRejectedValue(
      new ApiError("Line 2: Malformed utterance. Expected '[HH:MM:SS] Speaker: text'.", {
        code: "transcript_parse_error",
        status: 422,
      }),
    );
    const user = userEvent.setup();
    render(<AppShell />);

    await ingest(user);

    const message = await screen.findByText(/Line 2: Malformed utterance/);
    expect(message).toBeInTheDocument();
    expect(message.textContent).not.toContain("{");
  });

  it("disables the ingest button while the request is in flight", async () => {
    let release: (value: unknown) => void = () => {};
    ingestTranscript.mockReturnValue(new Promise((resolve) => (release = resolve)));
    const user = userEvent.setup();
    render(<AppShell />);

    await ingest(user);

    const button = screen.getByRole("button", { name: /ingesting/i });
    expect(button).toBeDisabled();
    release({ ...SUMMARY, chunk_count: 1 });
  });
});

describe("asking a question", () => {
  async function openMeeting(user: ReturnType<typeof userEvent.setup>) {
    listMeetings.mockResolvedValue([SUMMARY]);
    render(<AppShell />);
    await user.click(await screen.findByRole("button", { name: /^Release planning/ }));
    await screen.findByText("The budget is unchanged.");
  }

  it("sends the question for the selected meeting", async () => {
    const user = userEvent.setup();
    await openMeeting(user);

    await user.type(
      screen.getByLabelText("Ask this meeting"),
      "What was decided about the marketing budget?",
    );
    await user.click(screen.getByRole("button", { name: "Ask" }));

    await waitFor(() =>
      expect(askQuestion).toHaveBeenCalledWith({
        meetingId: SUMMARY.meeting_id,
        question: "What was decided about the marketing budget?",
      }),
    );
  });

  it("submits on Enter", async () => {
    const user = userEvent.setup();
    await openMeeting(user);

    await user.type(screen.getByLabelText("Ask this meeting"), "What was decided?{Enter}");

    await waitFor(() => expect(askQuestion).toHaveBeenCalledOnce());
  });

  it("renders the answer and its evidence card", async () => {
    const user = userEvent.setup();
    await openMeeting(user);

    await user.type(screen.getByLabelText("Ask this meeting"), "budget?{Enter}");

    expect(await screen.findByText("The budget remains unchanged.")).toBeInTheDocument();
    expect(screen.getByText("Grounded in meeting evidence")).toBeInTheDocument();
    const evidence = screen.getByText("Evidence 1").closest("li");
    expect(within(evidence as HTMLElement).getByText("00:01:14")).toBeInTheDocument();
  });

  it("scrolls to and highlights the cited utterance, matched by id", async () => {
    const scrollIntoView = vi.fn();
    Element.prototype.scrollIntoView = scrollIntoView;

    const user = userEvent.setup();
    await openMeeting(user);
    await user.type(screen.getByLabelText("Ask this meeting"), "budget?{Enter}");
    await screen.findByText("Evidence 1");

    await user.click(screen.getByRole("button", { name: /view in transcript/i }));

    const target = document.getElementById("utterance-release-planning-abc123-u1");
    expect(target).not.toBeNull();
    expect(scrollIntoView).toHaveBeenCalled();
    expect(target).toHaveClass("evidence-highlight");
  });

  it("clears a previous answer when another meeting is opened", async () => {
    const user = userEvent.setup();
    await openMeeting(user);
    await user.type(screen.getByLabelText("Ask this meeting"), "budget?{Enter}");
    await screen.findByText("The budget remains unchanged.");

    await user.click(screen.getByRole("button", { name: /^Release planning/ }));

    await waitFor(() =>
      expect(screen.queryByText("The budget remains unchanged.")).not.toBeInTheDocument(),
    );
  });

  it("surfaces an answer-provider failure", async () => {
    const { ApiError } = await import("@/lib/api");
    askQuestion.mockRejectedValue(
      new ApiError("The answer provider is unavailable.", { status: 502 }),
    );
    const user = userEvent.setup();
    await openMeeting(user);

    await user.type(screen.getByLabelText("Ask this meeting"), "budget?{Enter}");

    expect(await screen.findByText(/answer provider is unavailable/i)).toBeInTheDocument();
  });
});

describe("deleting a meeting", () => {
  it("asks for confirmation and calls the backend", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    listMeetings.mockResolvedValue([SUMMARY]);
    const user = userEvent.setup();
    render(<AppShell />);

    await user.click(await screen.findByRole("button", { name: /Delete Release planning/ }));

    await waitFor(() => expect(deleteMeeting).toHaveBeenCalledWith(SUMMARY.meeting_id));
  });

  it("does nothing when the confirmation is declined", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(false);
    listMeetings.mockResolvedValue([SUMMARY]);
    const user = userEvent.setup();
    render(<AppShell />);

    await user.click(await screen.findByRole("button", { name: /Delete Release planning/ }));

    expect(deleteMeeting).not.toHaveBeenCalled();
  });
});
