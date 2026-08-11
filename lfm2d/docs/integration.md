# Integrating with lfm2d

How to call `lfm2d` from another program, and the constraints its design
assumes. Written for the first consumers (Rust guard tooling), but nothing
here is client-specific.

## Why it's a network service

`candle-core` hard-wires `tokenizers` + `onig` (C, via `cc`) on every
non-wasm target. A consumer that requires a pure-Rust dependency graph — one
that links with nothing but a Rust toolchain, e.g. for static musl builds —
cannot link candle at all today. One upstream file
(`quantized/tokenizer.rs`) is the only consumer of that dependency, so
feature-gating it upstream would fix this; until then, in-process embedding
is unavailable to those callers and HTTP is the integration path.

Two measured facts shape the rest of the design:

- **Model loads copy.** `from_mmaped_safetensors` memcpys every tensor into
  private anonymous RSS. N processes = N × ~1.4 GiB per head; there is no
  shared-page trick and no warm-cache one-shot CLI. One resident daemon is
  the only sensible shape.
- **One forward saturates ~13.6 of 16 cores.** Concurrency does not raise
  throughput (measured: ~15 embeds/s solo *or* parallel). lfm2d therefore
  serializes requests through a single worker deliberately. **Do not design
  client-side fan-out expecting parallel speedup** — assume a queue.
  Batching within one request is the future lever, not concurrent
  connections. `lfm2d_worker_queue_depth` is exported so overload is
  visible before it is painful.

## Endpoints

```
GET  /healthz                 liveness
GET  /readyz                  readiness (503 before shutdown drains)
GET  /v1/models               [{id, kind, weight_hash, labels?, hidden_size}]
POST /embed                   TEI-compat  {"inputs": str|[str], "kind"?: document|query}
POST /predict                 TEI-compat  {"inputs": str|[str]}
POST /v1/classify             {"inputs": str|[str]}
POST /v1/route                {"input": str, "routes": [str]}  -> RAW cosines
POST /v1/cascade              {"clauses": [str]}
POST /v1/spans                {"inputs": str|[str], "model"?: str}
POST /v1/spans/credentials    same shape; credential.* entities only
```

The input field is `inputs` (TEI's spelling), not `texts`. `/v1/route` takes
singular `input` because it scores one text against many routes.

## Invariants

Each of these is load-bearing. Breaking them client-side reintroduces a
problem this service was built to avoid.

**1. No thresholds. Ranges overlap, measured.** Benign clauses top out at
0.3415 while data-critical ones start at 0.3440 — the ranges *overlap*, so
no global cutoff exists. `/v1/cascade` deliberately has **no `flagged`
boolean**. Rank items *within* a request; do not compare a score to a
constant. If your policy needs a yes/no, that decision belongs in your code
where it is visible as your choice.

**2. Read labels at runtime.** Label vocabularies change between
checkpoints, and have: `informative/mutating/destructive` became
`informative/situation-normal/data-critical` wholesale. Get them from
`GET /v1/models`. A hard-coded label string is a live breakage, not a
theoretical one.

**3. `severity_score` is an expected ordinal rank, range `0.0..=n`** for n
severe labels (so `0.0..=2.0` for a three-rung vocabulary) — *not* a
probability, and not bounded by 1.0. It was previously a sum over the severe
labels, which was not monotone in severity: mass moving up the ordinal scale
could lower the score. If you recorded values from an older build, they are
on a different scale.

**4. `/v1/route` returns raw cosines with no softmax.** A softmax over
caller-supplied routes is route-count arithmetic, not confidence.

**5. Every response carries `model_id` + `weight_hash`** (sha256 of the
checkpoint). This is an audit trail. Log the hash alongside any decision you
record, so a verdict can be tied to the exact weights that produced it.

**6. Failures are loud, and should stay loud.** An unknown severe label is
refused by name; a dead worker exits the process rather than serving a
healthy-looking 200. Do not paper over a 5xx with a permissive default — a
guard that fails open silently is worse than one that fails.

## Spans: the secrets-detection path

`/v1/spans` runs a token-classification head. The PII checkpoint is a
secrets detector as well as a PII detector — its 161 BIOES labels over 40
entity types include `credential.api_key`, `credential.jwt`,
`credential.private_key`, and `credential.connection_string`.

```json
POST /v1/spans   {"inputs": "export AWS_KEY=AKIA..."}
-> [[{"start": 15, "end": 35, "entity": "credential.api_key", "score": 0.94}]]
```

**The matched text is never returned.** Not by default, and not behind a
flag. Presidio omits it; GCP DLP makes it opt-in and says why (quotes are
sensitive); this API is stricter than both, because the caller already has
the text it just sent — echoing a credential back adds no information and
creates a second copy in every log downstream. Slice it yourself if you need
it.

**Offsets are UTF-8 byte offsets.** Not codepoints, not UTF-16 code units.
A Rust caller slices `&text[start..end]` directly and correctly. Python and
JavaScript callers must **not** index their own strings with these numbers —
`str` is codepoint-indexed and JS strings are UTF-16 — they must re-encode
to UTF-8 bytes (or operate on the bytes they already sent) before slicing.
Getting this wrong is silent on ASCII and wrong on the first non-ASCII
input, which for a secrets detector means redacting the wrong bytes.

**`score` is the minimum softmax confidence across the span's tokens**, not
the mean. A span is a conjunction of per-token decisions — it is wrong if
any one token is wrong — so its trustworthiness is that of its weakest
token, and averaging would hide precisely the coin-flip token indicating a
misplaced boundary. **These numbers read systematically lower than tools
that average** (Hugging Face's grouped-entity pipeline) or that report a
recognizer's own confidence (Presidio). Do not compare them across tools.
Per invariant 1, do not threshold them either.

`/v1/spans/credentials` returns only the `credential.*` family — a separate
endpoint rather than a flag, following Amazon Comprehend's split between
`DetectPiiEntities` and `ContainsPiiEntities`.

**Telemetry never records the payload.** No input text, no span offsets
(offsets reconstruct *where* a secret is), no matched substrings in traces,
logs, or metrics — only counts, durations, entity types, and model identity.
`--log-input-hash` is off by default, and when on it attaches to trace spans
only, never as a metric label.

## Configuration and deployment

`--token-classifier-dir` is repeatable; each head registers under its own
model id. With two or more loaded, name one in the request's `"model"`
field — omitting it is a 400 that lists the available ids rather than a
silent pick. No checkpoint is special-cased anywhere, so any
token-classification finetune is a config line.

**A checkpoint and its severe set travel together.** `--cascade-severe-label`
is repeatable and its **order is semantic: ascending severity, least severe
first.** The Nth severe label weighs N in the ranking. Reversing them
inverts the ranking silently, since both names are valid — verify against
`/v1/cascade` output, where a clearly destructive clause should score near
the top of the range.

The daemon serves JSON over a Unix socket and TCP from one binary, so
co-located callers can skip the TCP stack entirely.

**Memory:** roughly 1.4 GiB per resident head, and heads do not share
trunks unless they were trained on a common frozen trunk. Checked directly:
the base encoder and the shipped Prompt-Router have **0 of 148 trunk tensors
in common**, and a full finetune diverges from both. Budget per head.

## Standards

`/embed` and `/predict` are TEI-shaped so TEI-conformant clients work
against them unchanged. The `/v1/*` endpoints have no standard: TEI has no
token-classification endpoint at all, and KServe V2/OIP — the only real
standard here — is tensor-clunky and was deliberately not adopted. The span
shape follows the PII-service convention (Presidio / AWS / GCP) instead,
which is the closest thing to prior art for this job.
