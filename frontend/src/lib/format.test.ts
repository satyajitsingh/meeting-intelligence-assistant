import { describe, expect, it } from "vitest";
import {
  formatBytes,
  formatDuration,
  hasAudioExtension,
  meetingIdFromTitle,
  utteranceDomId,
} from "./format";

describe("meetingIdFromTitle", () => {
  it("slugifies a title and appends a suffix", () => {
    expect(meetingIdFromTitle("Release Planning", "abc123")).toBe("release-planning-abc123");
  });

  it("strips punctuation and collapses separators", () => {
    expect(meetingIdFromTitle("Q3 Budget / Review!!", "x1")).toBe("q3-budget-review-x1");
  });

  it("falls back when the title has no usable characters", () => {
    expect(meetingIdFromTitle("!!!", "x1")).toBe("meeting-x1");
  });

  it("produces a distinct id for repeated titles", () => {
    expect(meetingIdFromTitle("Standup")).not.toBe(meetingIdFromTitle("Standup"));
  });

  it("only produces url-safe characters", () => {
    expect(meetingIdFromTitle("Ünïcode Tïtle")).toMatch(/^[a-z0-9-]+$/);
  });
});

describe("utteranceDomId", () => {
  it("sanitises the colon in an utterance id", () => {
    expect(utteranceDomId("m1:u3")).toBe("utterance-m1-u3");
  });

  it("is stable for the same input", () => {
    expect(utteranceDomId("release-planning:u12")).toBe(utteranceDomId("release-planning:u12"));
  });

  it("produces a valid css selector target", () => {
    expect(utteranceDomId("m1:u3")).toMatch(/^[a-zA-Z][\w-]*$/);
  });
});

describe("formatDuration", () => {
  it.each([
    [0, "0:00"],
    [74, "1:14"],
    [98, "1:38"],
    [3661, "1:01:01"],
  ])("formats %i seconds as %s", (seconds, expected) => {
    expect(formatDuration(seconds)).toBe(expected);
  });

  it("handles missing values", () => {
    expect(formatDuration(null)).toBe("—");
    expect(formatDuration(undefined)).toBe("—");
  });
});

describe("formatBytes", () => {
  it.each([
    [512, "512 B"],
    [2048, "2 KB"],
    [5 * 1024 * 1024, "5.0 MB"],
  ])("formats %i bytes as %s", (bytes, expected) => {
    expect(formatBytes(bytes)).toBe(expected);
  });
});

describe("hasAudioExtension", () => {
  it.each(["a.mp3", "a.m4a", "a.WAV", "recording.webm", "a.flac"])("accepts %s", (name) => {
    expect(hasAudioExtension(name)).toBe(true);
  });

  it.each(["notes.txt", "deck.pdf", "noextension"])("rejects %s", (name) => {
    expect(hasAudioExtension(name)).toBe(false);
  });
});
