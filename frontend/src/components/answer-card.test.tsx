import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { AnswerResult } from "@/lib/api";
import { AnswerCard, trustLevel } from "./answer-card";

function answerWith(overrides: Partial<AnswerResult> = {}): AnswerResult {
  return {
    meeting_id: "m1",
    question: "What was decided about the marketing budget?",
    answer: "The budget remains unchanged.",
    citations: [
      {
        utterance_id: "m1:u3",
        speaker: "Sarah",
        timestamp: "00:01:14",
        start_seconds: 74,
        quote: "The budget is unchanged.",
      },
    ],
    insufficient_evidence: false,
    ...overrides,
  };
}

describe("trustLevel", () => {
  it("is grounded when citations survived validation", () => {
    expect(trustLevel(answerWith())).toBe("grounded");
  });

  it("is insufficient when the model reported no evidence", () => {
    expect(trustLevel(answerWith({ insufficient_evidence: true, citations: [] }))).toBe(
      "insufficient",
    );
  });

  it("is unverified when an answer claims evidence but cites nothing", () => {
    expect(trustLevel(answerWith({ citations: [] }))).toBe("unverified");
  });

  it("treats insufficient as insufficient even if citations leak through", () => {
    expect(trustLevel(answerWith({ insufficient_evidence: true }))).toBe("insufficient");
  });
});

describe("grounded answer", () => {
  it("renders the answer text", () => {
    render(<AnswerCard answer={answerWith()} onViewEvidence={vi.fn()} />);

    expect(screen.getByText("The budget remains unchanged.")).toBeInTheDocument();
  });

  it("shows the grounded trust badge", () => {
    render(<AnswerCard answer={answerWith()} onViewEvidence={vi.fn()} />);

    expect(screen.getByText("Grounded in meeting evidence")).toBeInTheDocument();
  });

  it("renders speaker, timestamp and the exact quote", () => {
    render(<AnswerCard answer={answerWith()} onViewEvidence={vi.fn()} />);

    expect(screen.getByText("Sarah")).toBeInTheDocument();
    expect(screen.getByText("00:01:14")).toBeInTheDocument();
    expect(screen.getByText(/The budget is unchanged\./)).toBeInTheDocument();
  });

  it("renders one card per citation", () => {
    const answer = answerWith({
      citations: [
        {
          utterance_id: "m1:u2",
          speaker: "Amir",
          timestamp: "00:00:52",
          start_seconds: 52,
          quote: "What happens to the marketing budget?",
        },
        answerWith().citations[0],
      ],
    });

    render(<AnswerCard answer={answer} onViewEvidence={vi.fn()} />);

    expect(screen.getByText("Evidence 1")).toBeInTheDocument();
    expect(screen.getByText("Evidence 2")).toBeInTheDocument();
  });

  it("never renders retrieval internals", () => {
    const { container } = render(<AnswerCard answer={answerWith()} onViewEvidence={vi.fn()} />);

    expect(container.textContent).not.toMatch(/chunk|score|vector/i);
  });
});

describe("evidence interaction", () => {
  it("maps a click to the utterance id, not the quote text", async () => {
    const onViewEvidence = vi.fn();
    render(<AnswerCard answer={answerWith()} onViewEvidence={onViewEvidence} />);

    await userEvent.click(screen.getByRole("button", { name: /view in transcript/i }));

    expect(onViewEvidence).toHaveBeenCalledExactlyOnceWith("m1:u3");
  });

  it("is reachable and activatable by keyboard", async () => {
    const onViewEvidence = vi.fn();
    render(<AnswerCard answer={answerWith()} onViewEvidence={onViewEvidence} />);

    await userEvent.tab();
    expect(screen.getByRole("button", { name: /view in transcript/i })).toHaveFocus();

    await userEvent.keyboard("{Enter}");
    expect(onViewEvidence).toHaveBeenCalledWith("m1:u3");
  });
});

describe("insufficient evidence", () => {
  const answer = answerWith({
    answer: "I don't have enough evidence in this meeting to answer that.",
    citations: [],
    insufficient_evidence: true,
  });

  it("shows the neutral insufficient badge", () => {
    render(<AnswerCard answer={answer} onViewEvidence={vi.fn()} />);

    expect(screen.getByText("Not enough evidence in this meeting")).toBeInTheDocument();
  });

  it("renders no evidence cards", () => {
    render(<AnswerCard answer={answer} onViewEvidence={vi.fn()} />);

    expect(screen.queryByText("Evidence 1")).not.toBeInTheDocument();
  });
});

describe("unverified answer", () => {
  const answer = answerWith({ citations: [] });

  it("warns that the answer could not be verified", () => {
    render(<AnswerCard answer={answer} onViewEvidence={vi.fn()} />);

    expect(
      screen.getByText("Answer could not be verified against meeting evidence"),
    ).toBeInTheDocument();
  });

  it("does not present it as grounded", () => {
    render(<AnswerCard answer={answer} onViewEvidence={vi.fn()} />);

    expect(screen.queryByText("Grounded in meeting evidence")).not.toBeInTheDocument();
  });

  it("still shows the answer text, with a caution", () => {
    render(<AnswerCard answer={answer} onViewEvidence={vi.fn()} />);

    expect(screen.getByText("The budget remains unchanged.")).toBeInTheDocument();
    expect(screen.getByText(/treat it with caution/i)).toBeInTheDocument();
  });
});
