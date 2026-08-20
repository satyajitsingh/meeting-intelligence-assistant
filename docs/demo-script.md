# Demo script (90 seconds)

## Before recording

```bash
cp backend/.env.example backend/.env    # add ANTHROPIC_API_KEY
make dev-backend                        # terminal 1 — http://localhost:8000
make dev-frontend                       # terminal 2 — http://localhost:3000
```

- `ANTHROPIC_API_KEY` is **required** — the answer step returns `502` without it.
- Ingest once beforehand so the embedding model (~130 MB) is already cached, then delete the
  meeting. The first ingestion of a fresh install is slow and makes for a poor recording.
- Have `sample_data/release_planning.txt` open, ready to paste.

**Transcript:** [`sample_data/release_planning.txt`](../sample_data/release_planning.txt)

```
[00:00:12] Sarah: We need to delay the release because migration is unfinished.
[00:00:31] John: Agreed. The migration script still fails.
[00:00:52] Amir: What happens to the marketing budget?
[00:01:14] Sarah: The budget is unchanged.
[00:01:38] John: I will update the launch plan by Friday.
```

**Question:** `What was decided about the marketing budget?`

**Second question (optional):** `What salary increase did the team approve?`

---

## Beats

### 0:00–0:10 — What it is

> "Meeting Intelligence. You ask a question about a meeting, and every answer comes back with the
> transcript evidence behind it — speaker, timestamp, exact quote."

Show the empty state. Keep this short; the product should be visible, not described.

### 0:10–0:25 — Ingest

Click **New meeting** → title `Release Planning` → paste the transcript → **Ingest meeting**.

> "I'll paste a short release planning meeting. It's parsed into utterances, chunked on speaker
> turns, embedded, and indexed."

The toast reports the counts. The transcript renders as readable turns, not a wall of text.

### 0:25–0:45 — Ask

Type: `What was decided about the marketing budget?` → **Ask**.

> "The question is embedded, matched against this meeting only, and the retrieved lines are the
> only thing the model is allowed to use."

### 0:45–1:05 — The answer and its evidence

> "The answer, and underneath it the evidence: Sarah, at 1:14, saying 'The budget is unchanged.'
> That quote isn't generated — the model only returns an utterance ID. The speaker, timestamp and
> text are read back from the stored transcript."

Point at the green **Grounded in meeting evidence** badge.

### 1:05–1:15 — Follow the evidence

Click **View in transcript**.

> "And I can follow any citation straight to its source line."

The transcript scrolls and the line highlights briefly.

### 1:15–1:30 — Trust and audio

> "If a model cites something that isn't in the meeting, or a line it was never shown, that
> citation is rejected before it reaches the screen. An answer with no surviving evidence is
> labelled unverified rather than presented as fact."

If time allows, ask the salary question to show the *"Not enough evidence in this meeting"* state
— it demonstrates the system declining to answer, which is more convincing than another success.

Close on the audio input:

> "The same flow starts from an uploaded or recorded meeting: it's transcribed, you review and
> correct it, and only then is it indexed."

---

## What to capture as screenshots

Save to `docs/screenshots/` and reference from the README:

1. `01-empty-state.png` — the welcome screen
2. `02-transcript-review.png` — pasted transcript with the required-format hint
3. `03-grounded-answer.png` — answer plus evidence cards and the grounded badge
4. `04-evidence-highlight.png` — the transcript line highlighted after clicking through
5. `05-insufficient-evidence.png` — the declining-to-answer state

## Notes

- The insufficient-evidence beat is worth more than a second successful answer: it shows the
  system refusing to invent, which is the harder behaviour to get right.
- Don't demo audio live unless `OPENAI_API_KEY` is set and you have rehearsed it — raw
  speech-to-text has no diarisation, so the text usually needs editing into
  `[HH:MM:SS] Speaker: text` before it will ingest. Mentioning that honestly is better than
  fumbling it on camera.
