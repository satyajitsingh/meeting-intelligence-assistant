"use client";

import type { MeetingSummary } from "@/lib/api";
import { Button, Spinner } from "@/components/ui";

export function MeetingSidebar({
  meetings,
  selectedId,
  loading,
  deletingId,
  onSelect,
  onNewMeeting,
  onDelete,
}: {
  meetings: MeetingSummary[];
  selectedId: string | null;
  loading: boolean;
  deletingId: string | null;
  onSelect: (meetingId: string) => void;
  onNewMeeting: () => void;
  onDelete: (meeting: MeetingSummary) => void;
}) {
  return (
    <aside className="flex w-72 shrink-0 flex-col border-r border-line bg-surface">
      <div className="border-b border-line px-5 py-5">
        <h1 className="text-sm font-semibold tracking-tight">Meeting Intelligence</h1>
        <p className="mt-0.5 text-xs text-muted">Grounded answers from your transcripts</p>
      </div>

      <div className="px-4 py-4">
        <Button onClick={onNewMeeting} className="w-full">
          New meeting
        </Button>
      </div>

      <nav aria-label="Meetings" className="flex-1 overflow-y-auto px-3 pb-4">
        <h2 className="px-2 pb-2 text-xs font-semibold uppercase tracking-wider text-muted">
          Meetings
        </h2>

        {loading ? (
          <p className="flex items-center gap-2 px-2 py-3 text-sm text-muted">
            <Spinner label="Loading meetings" />
            Loading meetings…
          </p>
        ) : meetings.length === 0 ? (
          <p className="px-2 py-3 text-sm leading-relaxed text-muted">
            No meetings yet. Create one to paste a transcript or upload audio.
          </p>
        ) : (
          <ul className="flex flex-col gap-1">
            {meetings.map((meeting) => {
              const active = meeting.meeting_id === selectedId;
              return (
                <li key={meeting.meeting_id} className="group relative">
                  <button
                    type="button"
                    onClick={() => onSelect(meeting.meeting_id)}
                    aria-current={active ? "true" : undefined}
                    className={`w-full rounded-lg px-3 py-2.5 pr-9 text-left transition-colors ${
                      active ? "bg-accent-soft text-accent" : "text-ink hover:bg-canvas"
                    }`}
                  >
                    <span className="block truncate text-sm font-medium">{meeting.title}</span>
                    <span className="mt-0.5 block text-xs text-muted">
                      {meeting.utterance_count} utterances · {meeting.speakers.length} speakers
                    </span>
                  </button>

                  <button
                    type="button"
                    onClick={() => onDelete(meeting)}
                    disabled={deletingId === meeting.meeting_id}
                    aria-label={`Delete ${meeting.title}`}
                    // Subtle: revealed on hover, but always reachable by keyboard.
                    className="absolute right-2 top-2.5 rounded p-1 text-muted opacity-0 transition hover:bg-red-50 hover:text-red-600 focus-visible:opacity-100 group-hover:opacity-100 disabled:opacity-40"
                  >
                    {deletingId === meeting.meeting_id ? <Spinner label="Deleting" /> : "🗑"}
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </nav>

      <p className="border-t border-line px-5 py-3 text-xs text-muted">
        Answers cite the transcript directly.
      </p>
    </aside>
  );
}
