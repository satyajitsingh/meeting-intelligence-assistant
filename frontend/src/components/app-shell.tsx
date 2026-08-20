"use client";

import { useCallback, useEffect, useState } from "react";
import {
  api,
  friendlyMessage,
  type AnswerResult,
  type MeetingSummary,
  type TranscriptDetail,
  type TranscriptionResult,
} from "@/lib/api";
import { MeetingSidebar } from "@/components/meeting-sidebar";
import { NewMeeting, type InputMethod } from "@/components/new-meeting";
import { QuestionPanel } from "@/components/question-panel";
import { Toaster, type ToastMessage, type ToastTone } from "@/components/toast";
import { TranscriptViewer } from "@/components/transcript-viewer";
import { EmptyHint, Panel, SectionHeading, Spinner } from "@/components/ui";
import { formatDuration, meetingIdFromTitle, utteranceDomId } from "@/lib/format";

type View = "welcome" | "new" | "meeting";

const HIGHLIGHT_MS = 1800;

/**
 * The single stateful component.
 *
 * All server state is fetched here and passed down, which keeps every other
 * component presentational and testable without a state library.
 */
export function AppShell() {
  const [view, setView] = useState<View>("welcome");
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  const [meetings, setMeetings] = useState<MeetingSummary[]>([]);
  const [loadingMeetings, setLoadingMeetings] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<TranscriptDetail | null>(null);
  const [loadingTranscript, setLoadingTranscript] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const [draftTitle, setDraftTitle] = useState("");
  const [draftText, setDraftText] = useState("");
  const [method, setMethod] = useState<InputMethod>("paste");
  const [transcription, setTranscription] = useState<TranscriptionResult | null>(null);
  const [transcribing, setTranscribing] = useState(false);
  const [ingesting, setIngesting] = useState(false);

  const [answer, setAnswer] = useState<AnswerResult | null>(null);
  const [asking, setAsking] = useState(false);

  const notify = useCallback((tone: ToastTone, text: string) => {
    setToasts((current) => [...current, { id: Date.now() + Math.random(), tone, text }]);
  }, []);

  const dismissToast = useCallback((id: number) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const refreshMeetings = useCallback(async (): Promise<MeetingSummary[]> => {
    try {
      const list = await api.listMeetings();
      setMeetings(list);
      return list;
    } catch (error) {
      notify("error", friendlyMessage(error));
      return [];
    } finally {
      setLoadingMeetings(false);
    }
  }, [notify]);

  // Inlined rather than calling refreshMeetings, so every state update sits
  // clearly behind an await and a late response cannot update an unmounted
  // component.
  useEffect(() => {
    let cancelled = false;

    async function loadInitialMeetings() {
      try {
        const list = await api.listMeetings();
        if (!cancelled) setMeetings(list);
      } catch (error) {
        if (!cancelled) notify("error", friendlyMessage(error));
      } finally {
        if (!cancelled) setLoadingMeetings(false);
      }
    }

    void loadInitialMeetings();
    return () => {
      cancelled = true;
    };
  }, [notify]);

  const openMeeting = useCallback(
    async (meetingId: string) => {
      setSelectedId(meetingId);
      setView("meeting");
      // Answers belong to the meeting that produced them.
      setAnswer(null);
      setLoadingTranscript(true);
      try {
        setTranscript(await api.getTranscript(meetingId));
      } catch (error) {
        setTranscript(null);
        notify("error", friendlyMessage(error));
      } finally {
        setLoadingTranscript(false);
      }
    },
    [notify],
  );

  function startNewMeeting() {
    setView("new");
    setSelectedId(null);
    setTranscript(null);
    setAnswer(null);
    setDraftTitle("");
    setDraftText("");
    setTranscription(null);
    setMethod("paste");
  }

  async function handleFile(file: File | Blob, filename: string) {
    setTranscribing(true);
    try {
      const result = await api.transcribeAudio(file, filename);
      setTranscription(result);
      setDraftText(result.text);
      if (!draftTitle.trim()) {
        setDraftTitle(filename.replace(/\.[^.]+$/, ""));
      }
      notify("success", "Audio transcribed. Review it before ingesting.");
    } catch (error) {
      notify("error", friendlyMessage(error));
    } finally {
      setTranscribing(false);
    }
  }

  async function handleIngest() {
    const title = draftTitle.trim() || "Untitled meeting";
    const text = draftText.trim();
    if (!text || ingesting) return;

    setIngesting(true);
    try {
      const created = await api.ingestTranscript({
        meetingId: meetingIdFromTitle(title),
        title,
        transcript: draftText,
      });
      await refreshMeetings();
      await openMeeting(created.meeting_id);
      notify(
        "success",
        `“${created.title}” indexed — ${created.utterance_count} utterances, ${created.chunk_count} chunks.`,
      );
    } catch (error) {
      notify("error", friendlyMessage(error));
    } finally {
      setIngesting(false);
    }
  }

  async function handleDelete(meeting: MeetingSummary) {
    const confirmed = window.confirm(
      `Delete “${meeting.title}”? Its transcript and index will be removed.`,
    );
    if (!confirmed) return;

    setDeletingId(meeting.meeting_id);
    try {
      await api.deleteMeeting(meeting.meeting_id);
      const remaining = await refreshMeetings();
      if (selectedId === meeting.meeting_id) {
        setSelectedId(null);
        setTranscript(null);
        setAnswer(null);
        setView(remaining.length > 0 ? "welcome" : "welcome");
      }
      notify("success", `“${meeting.title}” deleted.`);
    } catch (error) {
      notify("error", friendlyMessage(error));
    } finally {
      setDeletingId(null);
    }
  }

  async function handleAsk(question: string) {
    if (!selectedId || asking) return;

    setAsking(true);
    setAnswer(null);
    try {
      setAnswer(await api.askQuestion({ meetingId: selectedId, question }));
    } catch (error) {
      notify("error", friendlyMessage(error));
    } finally {
      setAsking(false);
    }
  }

  /**
   * Follow a citation into the transcript.
   *
   * Mapping is by utterance id only — never by matching quote text, which
   * would be guesswork.
   */
  const viewEvidence = useCallback((utteranceId: string) => {
    const element = document.getElementById(utteranceDomId(utteranceId));
    if (!element) return;

    element.scrollIntoView({ behavior: "smooth", block: "center" });
    element.classList.remove("evidence-highlight");
    // Force a reflow so re-adding the class restarts the animation.
    void element.offsetWidth;
    element.classList.add("evidence-highlight");
    setTimeout(() => element.classList.remove("evidence-highlight"), HIGHLIGHT_MS);
  }, []);

  return (
    <div className="flex h-screen overflow-hidden">
      <MeetingSidebar
        meetings={meetings}
        selectedId={selectedId}
        loading={loadingMeetings}
        deletingId={deletingId}
        onSelect={openMeeting}
        onNewMeeting={startNewMeeting}
        onDelete={handleDelete}
      />

      <main className="flex flex-1 flex-col overflow-hidden">
        <header className="flex items-center justify-between border-b border-line bg-surface px-8 py-4">
          <div>
            <h2 className="text-sm font-semibold tracking-tight">
              {transcript ? transcript.title : view === "new" ? "New meeting" : "Welcome"}
            </h2>
            {transcript ? (
              <p className="mt-0.5 text-xs text-muted">
                {transcript.speakers.join(", ")} · {transcript.utterances.length} utterances ·{" "}
                {formatDuration(transcript.duration_seconds)}
              </p>
            ) : null}
          </div>
          <span className="rounded-full border border-line px-3 py-1 text-xs text-muted">
            Evidence-backed answers
          </span>
        </header>

        <div className="flex-1 overflow-y-auto">
          {view === "new" ? (
            <NewMeeting
              title={draftTitle}
              method={method}
              transcript={draftText}
              transcription={transcription}
              transcribing={transcribing}
              ingesting={ingesting}
              onTitleChange={setDraftTitle}
              onMethodChange={setMethod}
              onTranscriptChange={setDraftText}
              onFile={handleFile}
              onInvalidFile={(message) => notify("error", message)}
              onIngest={handleIngest}
              onCancel={() => setView(selectedId ? "meeting" : "welcome")}
            />
          ) : view === "meeting" ? (
            <div className="mx-auto grid w-full max-w-6xl gap-6 px-8 py-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
              <Panel className="overflow-hidden">
                <div className="border-b border-line px-4 py-3">
                  <SectionHeading>Transcript</SectionHeading>
                </div>
                {loadingTranscript ? (
                  <p className="flex items-center gap-2 px-4 py-6 text-sm text-muted">
                    <Spinner label="Loading transcript" />
                    Loading transcript…
                  </p>
                ) : transcript ? (
                  <TranscriptViewer transcript={transcript} />
                ) : (
                  <p className="px-4 py-6 text-sm text-muted">No transcript loaded.</p>
                )}
              </Panel>

              <Panel className="h-fit p-5">
                <QuestionPanel
                  answer={answer}
                  pending={asking}
                  onAsk={handleAsk}
                  onViewEvidence={viewEvidence}
                />
              </Panel>
            </div>
          ) : (
            <div className="mx-auto flex w-full max-w-2xl flex-col gap-4 px-8 py-20 text-center">
              <h2 className="text-xl font-semibold tracking-tight">
                Ask questions about your meetings
              </h2>
              <EmptyHint>
                Add a meeting transcript, or upload a recording to transcribe. Every answer comes
                back with the exact lines that support it — speaker, timestamp and quote.
              </EmptyHint>
              <div className="mt-2 flex justify-center">
                <button
                  type="button"
                  onClick={startNewMeeting}
                  className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent/90"
                >
                  Create your first meeting
                </button>
              </div>
            </div>
          )}
        </div>
      </main>

      <Toaster toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}
