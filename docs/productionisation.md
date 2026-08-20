# Productionisation

How this v1 becomes a system a business could depend on. The target below is AWS, chosen because
the design already assumes hosted model providers, stateless request handling and adapter
boundaries that map cleanly onto managed services. The equivalent Azure or GCP mapping is
mechanical; the trade-offs are what matter.

Nothing here is implemented. This is the plan, not a description of the repository.

---

## Target architecture on AWS

### Frontend

The UI is a static Next.js build with no server-side data fetching, so **CloudFront in front of
S3** is sufficient and cheapest. If server components or middleware are introduced later, it moves
to **ECS/Fargate behind the same distribution** rather than being re-platformed.

The one real constraint: `NEXT_PUBLIC_API_BASE_URL` is baked in at build time, so each environment
needs its own build artefact. That is a deliberate trade — inlining removes a runtime config fetch
on every page load. The alternative, a `/config` endpoint read at boot, costs a round trip to
avoid a rebuild; not worth it for two environments, worth reconsidering at ten.

### API

**ECS/Fargate behind an ALB.** The service is stateless once the vector store moves out of
process, so it scales horizontally on CPU and request count.

Fargate over Lambda, for two specific reasons rather than preference:

- The embedding model is a ~130 MB ONNX artefact loaded into memory. On Lambda that is cold-start
  cost on every scale-out; on a warm container it is paid once.
- Answer generation is a synchronous call to an external model that can take seconds. Paying
  Lambda's per-millisecond rate to wait on someone else's inference is poor economics.

EKS only if the wider platform already runs Kubernetes. For a service this size it is overhead.

### Audio and transcription

Today audio is uploaded through the API and held in memory. That does not survive real files.

- **S3 presigned PUT** — the browser uploads directly, so audio never traverses the API. This
  removes the request-size ceiling and the memory pressure at once.
- **S3 event → SQS → worker** (Fargate task or Lambda) performs transcription.
- The client polls or receives a WebSocket/SSE notification when text is ready.

This changes the UX contract: transcription becomes a **job**, not a request. The current
synchronous endpoint is honest for a 25 MB cap and dishonest for a two-hour recording.

Ingestion (parse → chunk → embed) moves to the same queue for the same reason: embedding a long
transcript should not hold an HTTP connection open.

### Vector storage

The decision that most affects everything else. Two credible options:

| Option | Fits when | Cost |
|---|---|---|
| **Aurora PostgreSQL + pgvector** | Metadata and vectors want to be transactionally consistent; meetings are naturally scoped by tenant | One engine to operate; exact or IVFFlat/HNSW search; joins against meeting metadata come free |
| **OpenSearch Serverless vector engine** | Corpus grows past what a single Postgres instance serves comfortably; hybrid lexical+vector search is wanted | A second datastore to operate and keep consistent |

I would start with **Aurora + pgvector**. Meetings partition naturally by tenant and meeting ID, so
searches are always filtered to a small subset — exactly the case where a general-purpose vector
index is unnecessary. It also keeps transcript rows and vectors in one transaction, which removes
a whole class of "indexed but not stored" inconsistency that the current in-memory implementation
handles only by careful write ordering.

Move to OpenSearch when hybrid retrieval is justified by evaluation data, not before.

The `VectorStore` Protocol is the seam. No service code changes.

### Metadata

**Aurora PostgreSQL** — meetings, utterances, tenants, users, audit records. Utterances belong in
a table rather than being reconstructed from source text, because they are the citation unit and
must be immutable and addressable for as long as any answer references them.

### Secrets

**AWS Secrets Manager**, injected as environment variables by the task definition, with rotation.
The application already reads keys as `SecretStr` from settings and never logs or returns them, so
this is a deployment change, not a code change.

### LLM access

Two options, and they are not equivalent:

- **Amazon Bedrock** — keeps traffic inside the AWS network boundary, uses IAM rather than a
  long-lived key, and consolidates billing. Best when data residency or network isolation is a
  hard requirement.
- **Anthropic API directly** — access to the newest models first, and to features that reach the
  first-party API before partner platforms.

The `LLMProvider` Protocol makes this a one-adapter change. For a healthcare or life-sciences
buyer I would default to **Bedrock**, and treat the model-availability lag as the cost of a
defensible network story.

### Observability

- **CloudWatch Logs** — the application already emits single-line JSON with request IDs, so it is
  queryable via Logs Insights without changes.
- **OpenTelemetry → X-Ray or a vendor** — the spans worth having are retrieval, generation and
  citation validation. Answer latency is dominated by model time; without spans, every latency
  investigation guesses.
- **Metrics that reflect product health, not just uptime**: citation rejection rate,
  insufficient-evidence rate, and answers with zero valid citations. A rise in any of those means
  quality has regressed even while every request returns `200`.

### Security

- **Cognito or an enterprise OIDC provider** — meeting content is sensitive by default.
- **Tenant isolation enforced at the data layer**, not in application code. Every query carries a
  tenant predicate; row-level security in Postgres makes a missing filter fail closed rather than
  leak. The current per-meeting scoping is the right shape but the wrong enforcement point.
- **KMS** for encryption at rest, customer-managed keys where a buyer requires it.
- **Private subnets**, egress via NAT to provider endpoints only.
- **WAF** on the ALB.
- **Audit trail** — who asked what, of which meeting, and what evidence was returned. For
  regulated buyers this is a requirement, and it is also the dataset that makes evaluation
  possible later.

---

## Scaling

**Stateless API.** Once the vector store and repository move out of process, instances are
interchangeable and scale on CPU and queue depth.

**Async everything slow.** Transcription and ingestion become queued jobs with visibility
timeouts, retries with backoff, and a dead-letter queue. A poisoned audio file should surface for
inspection, not retry forever.

**Cache embeddings.** Chunk text is deterministic, so its embedding is too. Keying a cache on a
content hash makes re-ingesting an edited transcript cost only the changed chunks — the common
case, since the review step exists precisely so people edit transcripts.

**Batch embedding calls.** The provider already accepts a list; a long meeting should embed in
batches sized to the model, not one call per chunk.

**Bound concurrency to the providers.** Model APIs rate-limit. A semaphore plus a token-bucket per
provider, with `429` surfaced as backpressure rather than a failed user request.

**Read-heavy asymmetry.** Ingestion is rare and expensive; questions are frequent and cheap.
Scale the two independently — separate the ingestion workers from the API service.

**Cost control.** Cache identical question/meeting pairs briefly; use a smaller model for
retrieval-only paths; keep the effort setting low for what is fundamentally extraction.

---

## Data protection and privacy

Meeting transcripts are among the most sensitive unstructured data a company holds — personnel
discussions, commercial terms, and in a life-sciences context potentially patient-adjacent
information.

**Encryption.** TLS in transit; KMS at rest for S3, Aurora and backups. Customer-managed keys
where the buyer requires cryptographic separation.

**Retention and deletion.** Configurable per tenant, with deletion that actually deletes:
transcript rows, vectors, cached embeddings, uploaded audio, and derived artefacts. The current
`DELETE` already removes both transcript and vectors together, which is the right shape.

**PII.** Speaker names are PII by definition here. Options, in increasing strength: redact at
display for unauthorised roles; pseudonymise speakers at ingestion and map back at render time; or
detect and mask entity classes before any text leaves the boundary. The last conflicts with
verbatim citation, which is a genuine trade-off to make explicitly with the buyer rather than
silently.

**Tenant boundaries.** Enforced in the datastore, tested adversarially. The existing
cross-meeting citation rejection is the same instinct applied one level down.

**Operational logs must stay clean.** The citation-validation warnings deliberately carry
identifiers and reasons but never transcript text, because those lines are the most likely to be
shipped to a third-party aggregator. That discipline needs to extend to every log line added
later — it is easy to lose by accident.

**Provider data handling.** Both Anthropic and OpenAI offer zero-retention and no-training
arrangements on business terms; for a regulated buyer these should be contractual, and Bedrock
avoids the question for inference entirely.

**No compliance claims.** This project is not HIPAA or GDPR compliant and nothing here should be
read as certification. The controls above are the technical groundwork such a programme requires,
not a substitute for it.

---

## Model evaluation

**None of this exists in the repository yet, and that is the largest gap between this project and
something trustworthy.** Retrieval choices — dense-only, `k=5`, 700-character chunks, one-utterance
overlap — are reasoned but unmeasured.

What a real harness needs:

**A golden dataset.** 15–30 questions across several meetings, each labelled with the utterance
IDs that genuinely answer it, plus deliberate negatives — questions the transcript does not
answer. Building this by hand is unglamorous and is the single highest-value next task, because
every metric below depends on it.

**Retrieval quality**, measured before generation is involved at all:
- *recall@k* — did the retrieved chunks contain the labelled utterances? A generation failure
  caused by a retrieval miss is a different bug from a prompting failure, and conflating them
  wastes days.
- *MRR* — how far down the ranking the first correct chunk appeared.

**Citation quality**, which is where this design should differentiate:
- *citation precision* — of the citations returned, how many actually support the claim? Requires
  human or LLM-judge labelling.
- *citation validity rate* — how often the model proposes an ID that validation rejects. This one
  is free: the `citation.invalid` warnings already carry it. A rise means the prompt or the
  evidence format has regressed.

**Answer quality:**
- *groundedness* — is every claim supported by a cited utterance? An LLM judge with the transcript
  is a reasonable first approximation, calibrated against human labels on a sample.
- *insufficient-evidence accuracy* — the negatives matter as much as the positives. A system that
  confidently answers unanswerable questions is worse than one that retrieves poorly, because the
  failure is invisible.

**Operational metrics:** p50/p95 latency split by retrieval and generation, cost per answer, and
token usage per question.

**Regression gate.** The suite runs in CI against recorded provider responses so it is
deterministic and free, plus a small live run before release. Prompt and model changes are
evaluated, not eyeballed — the failure mode of prompt tuning is improving the example you are
looking at while quietly degrading everything else.

**Human review** on a sampled basis, feeding back into the golden set. Automated groundedness
scoring drifts; periodic human labelling is what keeps it honest.
