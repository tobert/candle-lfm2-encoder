# Brief for kaibo leader — building kaibo tools against lfm2d

**From:** lfm2d leader (session of 2026-08-11)
**Status:** design brief, not a spec. Amy explicitly wants kaibo leader to
**work the design with her directly** — she said she'll stop by. Treat the
open questions at the bottom as hers to answer, not as things to guess.

Amy's framing when she asked for this: *"clear notes for kaibo leader to
start some subagents working on kaibo tools for lfm2d (and other tokenizer /
classifier services, which should help drive our design)."* Note the
parenthetical — **lfm2d is the first instance of a category, not the
category.** If the kaibo-side tool ends up hard-wired to lfm2d's exact JSON,
we have built the wrong thing.

## Why this is a network service and not a crate

This is the load-bearing constraint, and it is measured, not assumed:

**kaibo cannot link candle at all today.** `candle-core` 0.11 hard-wires
`tokenizers` + `onig` (C, via `cc`) on every non-wasm target, which breaks
kaibo's "musl links with nothing but a Rust toolchain" invariant — the same
invariant that got `blake3` rejected for less. One upstream file
(`quantized/tokenizer.rs`) is the only consumer; feature-gating it is a small
PR someone could land, but until then, in-process is off the table.

So kaibo talks to lfm2d over HTTP. The client should be whatever kaibo's
existing HTTP stack already is — **no new C dependencies, no new TLS
backends.** If the natural client pulls in something that compiles C, stop
and raise it rather than working around it.

Two more measured facts that shape the design:

- **Model loads copy.** `from_mmaped_safetensors` memcpys every tensor into
  private anonymous RSS. N processes = N × 1.4 GiB; there is no shared-page
  trick and no warm-cache one-shot CLI. One resident daemon is the only
  shape that makes sense.
- **One forward saturates ~13.6 of 16 cores.** Concurrency does not raise
  throughput (15 embeds/s solo *or* parallel). lfm2d therefore serializes
  requests through a single worker on purpose. **Do not design kaibo-side
  fan-out expecting parallel speedup** — batching within one request is the
  future lever, not concurrent connections. Assume a queue.

## What is deployed right now

- Live in k3s, namespace `lfm2d`, on node `zorak`, tailnet-exposed via the
  Tailscale operator: **`http://lfm2d-1.taila4abc.ts.net:8088`**
- Loaded heads as of 2026-08-11: `kube_ordinal_v8` (sequence classifier,
  labels `informative` / `situation-normal` / `data-critical`) and
  `LFM2.5-Encoder-350M-Prompt-Router`.
- **Unauthenticated.** The boundary is the tailnet, which Amy has ruled
  trusted (her partner's box is the other account on it). Do not add
  credentials on the kaibo side without asking — and do not assume the
  absence of auth is an oversight to be fixed unilaterally.

## The API as it actually exists

```
GET  /healthz            liveness
GET  /readyz             readiness (flips to 503 before shutdown drains)
GET  /v1/models          [{id, kind, weight_hash, labels?, hidden_size}]
POST /embed              TEI-compat  {"inputs": str|[str], "kind"?: document|query}
POST /predict            TEI-compat  {"inputs": str|[str]}
POST /v1/classify        {"inputs": str|[str]} -> per-input {scores{label:p}, top, model_id, weight_hash}
POST /v1/route           {"input": str, "routes": [str]} -> RAW cosines, no softmax
POST /v1/cascade         {"clauses": [str]}   -> per-clause severity + winner + lane
```

Note `inputs` (TEI's spelling), not `texts` — I got this wrong on my first
probe today, so it is worth saying out loud.

### Invariants that must survive contact with kaibo

These are hard-won and each one has a memory or a ruling behind it:

1. **No thresholds, ever, in the library or the daemon.** Measured: benign
   clauses top out at 0.3415 while data-critical ones start at 0.3440 —
   the ranges *overlap*, so no global cutoff exists. `/v1/cascade`
   deliberately has **no `flagged` boolean**. Gates rank clauses *within* a
   statement. If a kaibo guard needs a yes/no, that policy lives in kaibo
   and should be visible as kaibo's choice, not smuggled into the service.
2. **`/v1/route` returns raw cosines with no softmax.** A softmax over
   caller-supplied routes is route-count arithmetic, not confidence.
3. **Every response carries `model_id` + `weight_hash`** (sha256 of the
   checkpoint). This is an audit trail, not decoration — today's deploy was
   verified by matching the served hash against the export byte-for-byte.
   **Kaibo should log the hash with any guard decision it records**, so a
   verdict can be tied to the exact weights that produced it.
4. **Loud failures.** Unknown severe labels are refused by name; a dead
   worker exits the process rather than serving a healthy-looking 200.
   Kaibo should not paper over a 5xx with a permissive default — a guard
   that fails open silently is worse than one that fails.

## The gap that blocks kaibo's actual use case

Kaibo's headline reason to want this — Amy's own framing — is **screening
foreign content crossing the membrane**: injection and PII/secrets in
untrusted repo reads and outbound batch payloads. That is a **token-level**
job: you need the spans, not a sentence-level score.

**lfm2d does not expose token classification today.** The library has it
(`src/token_classification.rs`: `Lfm2TokenClassifier::predict` → `Vec<Span>`,
plus a `credentials()` convenience), and the PII detector checkpoint is a
real secrets detector — its 161 BIOES labels include
`credential.api_key`, `credential.jwt`, `credential.private_key`,
`credential.connection_string`. But the daemon's `ModelKind` is only
`Embedder | Classifier | Router`, and there is no span endpoint.

**So the first concrete work item is daemon-side, not kaibo-side:** add a
token-classification head kind and a span endpoint. That is lfm2d's job — I
own it. Kaibo leader should design the *consumer* contract with Amy in
parallel so the endpoint gets shaped by a real caller instead of by my
guess. What I need from that conversation: what a span response has to
contain for kaibo to act on it (offsets in bytes or chars? entity type
taxonomy passed through raw, or collapsed to a kaibo-side category? does
kaibo need the whole span list, or only a
"were there any credential-class spans" answer?).

**Memory cost, so nobody is surprised:** the PII detector is a LiquidAI
shipped specialist with a *genuinely different trunk* from both the router
and our finetune — I verified this today, 0 of 148 trunk tensors match.
Adding it makes three resident trunks: ~4.2 GiB, up from the measured 2.78
GiB steady RSS. There is no trunk-sharing trick that avoids this for
heads that were not trained on a common frozen trunk. Budget accordingly.

## Where kaibo would actually call it

From the earlier recon of kaibo's own structure — **re-verify before
building, this was surveyed 2026-08-09 and kaibo has moved since**:

- **Foreign repo reads** all funnel through one sandbox job loop
  (`src/sandbox.rs`, ~458-494). One chokepoint, clean.
- **Interactive outbound** all passes `Watched::completion`
  (`src/completion_watch.rs`, ~286). One chokepoint, clean.
- **Batch / media outbound is five scattered `reqwest` call sites with no
  common gate.** This is the structural gap. A guard here needs a
  chokepoint to exist first — that refactor is probably prerequisite work,
  and it is kaibo's call whether it happens before or alongside.

## Designing for "and other tokenizer / classifier services"

Amy's parenthetical is the interesting part of this assignment. Suggested
shape, to be argued with her rather than accepted:

- The kaibo-side abstraction is **"a classifier service"**: something that
  takes text and returns labeled scores and/or spans, with model identity
  attached. lfm2d is one implementation; a TEI server, a local llama.cpp
  with a classifier head, or a future rewrite are others.
- lfm2d's `/embed` and `/predict` are **already TEI-shaped on purpose**, so
  a TEI-conformant client gets those two endpoints for free against other
  servers. The `/v1/*` endpoints are ours and have no standard. KServe
  V2/OIP is the only real standard in this space and it is tensor-clunky;
  we deliberately did not conform. Do not invent a fourth convention
  without a reason.
- **Discovery via `GET /v1/models`.** Labels are not hard-coded anywhere:
  the daemon reads them from the checkpoint. A kaibo tool that hard-codes
  `destructive` or `data-critical` will break the next time we ship a
  checkpoint — which just happened today, v6 → v8, and the label vocabulary
  changed wholesale. **Read the labels at runtime.**

## Open questions — Amy's, not ours

1. Span response contract (see above) — the one that unblocks real work.
2. Does kaibo call lfm2d over the tailnet, or does lfm2d get a local
   sidecar/UDS deployment next to kaibo? The daemon already serves both
   Unix socket and TCP from one binary, so this is a deployment choice, not
   a code change. A local registry is coming, which makes the sidecar
   cheaper than it was.
3. Fail-open vs fail-closed when lfm2d is unreachable, per chokepoint. My
   read is that these differ — a foreign-read screen and an outbound-payload
   screen do not deserve the same default — but that is a policy call.
4. Is the batch-outbound chokepoint refactor in scope, or does kaibo tooling
   start with the two clean chokepoints only?

## Coordination

- lfm2d leader (this session) owns the daemon, its endpoints, and the
  cluster deployment. Ask me for endpoint changes rather than working
  around their absence.
- **zorak is the machine agent** for host-level and cluster work, and will
  work with brak when it needs to. Route infrastructure asks that way.
- Repo rename to `lfm2d` is coming, and a local container registry with it.
  **Do not pin to the current GitHub repo name or to `ghcr.io` paths yet.**
