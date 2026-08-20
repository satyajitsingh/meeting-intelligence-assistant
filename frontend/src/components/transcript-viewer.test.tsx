import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { TranscriptDetail } from "@/lib/api";
import { utteranceDomId } from "@/lib/format";
import { TranscriptViewer } from "./transcript-viewer";

const TRANSCRIPT: TranscriptDetail = {
  meeting_id: "m1",
  title: "Release planning",
  speakers: ["Sarah", "John"],
  duration_seconds: 98,
  utterances: [
    {
      id: "m1:u0",
      index: 0,
      speaker: "Sarah",
      start_seconds: 12,
      raw_timestamp: "00:00:12",
      display_timestamp: "00:00:12",
      text: "We need to delay the release.",
    },
    {
      id: "m1:u3",
      index: 3,
      speaker: "Sarah",
      start_seconds: 74,
      raw_timestamp: "00:01:14",
      display_timestamp: "00:01:14",
      text: "The budget is unchanged.",
    },
  ],
};

describe("TranscriptViewer", () => {
  it("renders each utterance as a readable block", () => {
    render(<TranscriptViewer transcript={TRANSCRIPT} />);

    expect(screen.getByText("We need to delay the release.")).toBeInTheDocument();
    expect(screen.getByText("The budget is unchanged.")).toBeInTheDocument();
  });

  it("shows the timestamp and speaker for each utterance", () => {
    render(<TranscriptViewer transcript={TRANSCRIPT} />);

    expect(screen.getByText("00:01:14")).toBeInTheDocument();
    expect(screen.getAllByText("Sarah")).toHaveLength(2);
  });

  it("gives each utterance a sanitised, stable dom id", () => {
    const { container } = render(<TranscriptViewer transcript={TRANSCRIPT} />);

    expect(container.querySelector(`#${utteranceDomId("m1:u3")}`)).not.toBeNull();
    expect(container.querySelector("#utterance-m1-u3")).not.toBeNull();
  });

  it("keeps the raw utterance id available for mapping", () => {
    const { container } = render(<TranscriptViewer transcript={TRANSCRIPT} />);

    expect(container.querySelector('[data-utterance-id="m1:u3"]')).not.toBeNull();
  });

  it("preserves transcript order", () => {
    const { container } = render(<TranscriptViewer transcript={TRANSCRIPT} />);

    const ids = Array.from(container.querySelectorAll("[data-utterance-id]")).map((node) =>
      node.getAttribute("data-utterance-id"),
    );
    expect(ids).toEqual(["m1:u0", "m1:u3"]);
  });
});
