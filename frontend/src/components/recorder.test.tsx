import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Recorder } from "./recorder";

const originalMediaRecorder = window.MediaRecorder;

afterEach(() => {
  if (originalMediaRecorder) {
    window.MediaRecorder = originalMediaRecorder;
  } else {
    Reflect.deleteProperty(window, "MediaRecorder");
  }
});

function stubSupport(supported: boolean) {
  if (supported) {
    // jsdom implements neither API, so both halves of the check are stubbed.
    window.MediaRecorder = function MediaRecorderStub() {} as unknown as typeof MediaRecorder;
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn() },
    });
  } else {
    Reflect.deleteProperty(window, "MediaRecorder");
  }
}

describe("Recorder", () => {
  it("explains itself when the browser cannot record", () => {
    stubSupport(false);

    render(<Recorder disabled={false} onRecorded={vi.fn()} />);

    expect(screen.getByText(/cannot record audio/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /start recording/i })).not.toBeInTheDocument();
  });

  it("offers recording when the browser supports it", () => {
    stubSupport(true);

    render(<Recorder disabled={false} onRecorded={vi.fn()} />);

    expect(screen.getByRole("button", { name: /start recording/i })).toBeInTheDocument();
  });

  it("disables recording while another request is in flight", () => {
    stubSupport(true);

    render(<Recorder disabled onRecorded={vi.fn()} />);

    expect(screen.getByRole("button", { name: /start recording/i })).toBeDisabled();
  });

  it("reports a blocked microphone as text, not colour alone", async () => {
    stubSupport(true);
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn().mockRejectedValue(new Error("denied")) },
    });

    const { default: userEvent } = await import("@testing-library/user-event");
    render(<Recorder disabled={false} onRecorded={vi.fn()} />);

    await userEvent.click(screen.getByRole("button", { name: /start recording/i }));

    expect(await screen.findByText(/microphone access was blocked/i)).toBeInTheDocument();
  });
});
