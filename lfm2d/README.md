# lfm2d

HTTP sidecar/daemon serving `candle-lfm2-encoder` heads over a Unix domain
socket and/or TCP, from ONE process. Built because:

- candle's `from_mmaped_safetensors` copies every tensor into private
  anonymous RSS on load — N processes loading the same 350M checkpoint cost
  N × ~1.4 GiB, not one shared mapping.
- A single forward pass already saturates ~13.6 of this box's cores
  (measured) — per-request concurrency has no headroom to spend, so this
  daemon serves every request through ONE serial inference worker thread by
  design, not as an unoptimized bottleneck.
- `kaibo` cannot link `candle-core` at all (it hard-wires
  `tokenizers`+`onig`, breaking kaibo's musl static-build invariant) — an
  HTTP boundary is the only way kaibo can reach these models.

See `src/lib.rs`'s module docs for the full architecture writeup (worker
thread / channel design, the `InferenceEngine` trait that decouples router
tests from candle, and the two response conventions this API uses).

## Building

```sh
cargo build --release -p lfm2d
```

Container image (build context must be the repo ROOT, since `lfm2d`
depends on the parent crate via `path = ".."`):

```sh
podman build -t lfm2d:latest -f lfm2d/Containerfile .
```

See `deploy/lfm2d.container` for an example podman quadlet unit (models
volume-mounted, `CPUWeight=`/`Nice=` set, both socket and TCP examples,
OTLP env wiring) and `deploy/k8s.yaml` for a complete Deployment + Service
(probes, a measured memory recipe, CPU requests without limits, OTLP env
wiring — see "Deploying on k8s" below for the reasoning behind each).

## Configuration

CLI flags, each with an env-var fallback (`clap`'s `env` feature):

| Flag | Env var | Meaning |
| --- | --- | --- |
| `--embedder-dir` | `LFM2D_EMBEDDER_DIR` | `Lfm2Embedding`-shaped checkpoint dir; backs `/embed` |
| `--classifier-dir` | `LFM2D_CLASSIFIER_DIR` | `Lfm2SequenceClassifier`-shaped checkpoint dir; backs `/predict`, `/v1/classify`, `/v1/cascade` |
| `--router-dir` | `LFM2D_ROUTER_DIR` | Prompt-Router checkpoint dir; backs `/v1/route`, `/v1/cascade` |
| `--cascade-route` (repeatable) | `LFM2D_CASCADE_ROUTES` (comma-separated) | Candidate routes for `/v1/cascade` — server-side config, not a request field |
| `--cascade-severe-label` (repeatable) | `LFM2D_CASCADE_SEVERE_LABELS` (comma-separated) | Which classifier labels count toward the severity ranking sum; default `mutating,destructive` |
| `--socket-path` | `LFM2D_SOCKET_PATH` | Unix domain socket to serve on |
| `--bind-addr` | `LFM2D_BIND_ADDR` | TCP address to serve on, e.g. `127.0.0.1:8088` |
| `--threads` | `LFM2D_THREADS` | Size of rayon's global thread pool (candle's matmul runs on it transitively), set BEFORE any model load. Defaults to `std::thread::available_parallelism()` |

Standard OTEL env vars also apply (`OTEL_EXPORTER_OTLP_ENDPOINT`,
`OTEL_SERVICE_NAME` default `lfm2d`, ...) — see "Observability" below; none
of these are `clap` flags, they're read directly by the OTLP exporters and
`main.rs`.

At least one of `--socket-path`/`--bind-addr` and at least one of
`--embedder-dir`/`--classifier-dir`/`--router-dir` are required — startup
fails loudly (`exit 2`, config problem named) otherwise. Model loading
itself is synchronous and happens before the socket is ever bound: a bad
checkpoint path or an incompatible config also fails loudly at startup
(`exit 1`), never lazily on the first request.

Only ONE model per head kind is supported — `/predict`, `/v1/classify`,
`/v1/route` don't take a model-selection parameter, so a deployment serving
multiple classifiers, say, needs multiple `lfm2d` processes.

## API (v1)

- `GET /healthz` → `200 "ok"` — process alive, never touches the worker.
- `GET /readyz` → `200` once every configured model has loaded, `503`
  before.
- `GET /v1/models` → `[{id, kind, weight_hash, labels?, hidden_size}]` —
  every loaded model. `kind` is `embedder`/`classifier`/`router`. `labels`
  is present only for a classifier (an embedder/router has no fixed label
  set). `weight_hash` is sha256 over the checkpoint's `model.safetensors`,
  64 lowercase hex chars — the audit trail this whole daemon exists partly
  to satisfy (kaish approval-chain rulings).
- `POST /embed` — TEI-compatible-ish. `{"inputs": "text"}` or
  `{"inputs": ["a", "b"]}`, optional `"kind": "query"|"document"` (default
  `document`; this is the E5-style asymmetric-embedding side selector —
  see `candle_lfm2_encoder::TextKind`'s docs on why picking wrong quietly
  degrades retrieval). Response: `[[f32, ...]]`, one pooled vector per
  input, in input order. `model_id`/`weight_hash` travel as
  `X-Model-Id`/`X-Model-Weight-Hash` response headers, not in the JSON body
  — keeping the body wire-compatible with plain TEI clients that don't
  know these headers exist.
- `POST /predict` — TEI-ish sequence classification. `{"inputs": ...}` →
  per input, `[{"label", "score"}, ...]` covering ALL labels (full
  softmax, never just top-1), sorted by score descending. Audit headers
  same as `/embed`.
- `POST /v1/classify` — our full contract, not TEI-compat: `{"inputs":
  [...]}` → per input `{"scores": {label: prob}, "top": label, "model_id",
  "weight_hash"}`.
- `POST /v1/route` — `{"input": str, "routes": [str, ...]}` → `{"model_id",
  "weight_hash", "routes": [{"route", "cosine"}, ...]}`. **RAW COSINE
  ONLY** — this router's softmax is route-count arithmetic and carries no
  confidence information (see `candle_lfm2_encoder::routing`'s module
  docs), so it is never exposed on this endpoint.
- `POST /v1/cascade` — `{"clauses": [str, ...]}` (pre-decomposed; clause
  decomposition is the caller's job, e.g. via a kaish `Plan`) →
  moderations-FLAVORED but with **no `flagged` boolean and no threshold
  anywhere**:
  ```json
  {
    "winner": {"index": 1, "clause": "rm -rf .", "severity_scores": {"...": 0.9}},
    "lane": {"route": "shell", "cosine": 0.91},
    "clauses": [
      {"index": 0, "clause": "...", "severity_scores": {"...": 0.9}, "top_severity": "informative"},
      {"index": 1, "clause": "rm -rf .", "severity_scores": {"...": 0.9}, "top_severity": "destructive"}
    ],
    "models": [{"model_id": "...", "weight_hash": "..."}, {"model_id": "...", "weight_hash": "..."}]
  }
  ```
  This handler does NOT reimplement `candle_lfm2_encoder::cascade`'s
  rank-by-severity-route-the-winner aggregation — it calls straight through
  to the library's `Cascade::run`. **No global severity cutoff exists by
  measurement** (benign max 0.3415 vs data-critical min 0.3440 — see
  `src/cascade.rs`'s module docs in the parent crate): consumers must rank
  clauses WITHIN a statement, never threshold any number here against a
  fixed cutoff. `routes` and `severe_labels` are `--cascade-route`/
  `--cascade-severe-label` server config, not request fields — see
  `src/lib.rs`'s "cascade configuration is server-side" section for why
  this was a judgment call, not something the spec pinned down explicitly.

Errors: `{"error": {"message", "type"}}`. `type` is `"bad_request"` (400 —
malformed/empty input, or a call against a head this instance never
loaded) or `"internal"` (500 — a loaded model's forward pass failed, or the
worker thread itself died). Never a silently-wrong `200`.

## Being a good k8s/k3s container citizen

**Graceful shutdown.** On SIGTERM or SIGINT: `/readyz` flips to 503
immediately (so a Service's endpoint controller stops routing new traffic
here), every listener (Unix socket and/or TCP) stops accepting new
connections via axum's `with_graceful_shutdown`, in-flight requests are
allowed to finish (a forward pass here is sub-second), then the process
exits 0. The drain is bounded — `shutdown::DEFAULT_DRAIN_TIMEOUT` (10s) —
so a wedged request can't hang shutdown forever; `serve()` returns `Ok(())`
either way, exit code 0 regardless. Rust runs as PID 1 in the container, so
this handler is installed explicitly (`tokio::signal`) — there's no init
process to translate the signal for us. See `src/shutdown.rs` and
`src/main.rs`'s `install_signal_handlers`; `tests/graceful_shutdown.rs`
drives the sequence programmatically (via `server::begin_shutdown`) and
asserts all three parts: `/readyz` flips, the in-flight request completes
200, `serve()` resolves.

**Crash-only on worker death.** A worker-thread panic today would leave the
HTTP server answering 500s forever while `/healthz` still said 200 —
alive-but-useless, and kubelet never restarts a pod whose process hasn't
exited. `main.rs`'s production path spawns the worker via
`WorkerHandle::spawn_crash_on_panic` instead of the plain `spawn` the test
suite uses: a monitor thread calls `std::process::exit(1)` if (and only if)
the worker thread terminates via panic — NOT on the clean channel-closure
exit a graceful shutdown produces (see `worker_thread_outcome_is_a_crash`
in `src/worker.rs`, unit-tested directly). `tests/worker_crash_monitor.rs`
proves this through the REAL compiled binary: it spawns `lfm2d` itself
with `LFM2D_TEST_CRASH_ON_WORKER_PANIC=1` (a test-only env-gated branch in
`main.rs` that swaps in a panic-on-call stub engine instead of loading real
checkpoints — real models aren't needed to prove the process-exit
mechanism), fires one request over a real HTTP connection, and asserts the
CHILD PROCESS itself exits nonzero.

**Startup observability + `--threads`.** Startup logs (via `tracing`, not
`eprintln!`) the full effective config, each loaded model's `{id, kind,
weight_hash}`, `std::thread::available_parallelism()`, and a hand-rolled
container-runtime probe (`/.dockerenv` → docker, `/run/.containerenv` →
podman, `KUBERNETES_SERVICE_HOST` → kubernetes, `container` env var,
`/sys/fs/cgroup/cpu.max` when readable) — see `src/probe.rs`, unit-tested
with injectable root dir + env snapshot, no new dependency. `--threads N`
(env `LFM2D_THREADS`) sizes rayon's GLOBAL thread pool — which candle's
matmul runs on transitively — via
`rayon::ThreadPoolBuilder::num_threads(n).build_global()`, called BEFORE
any model load (rayon's global pool can only be built once, so this has to
win the race against candle's own lazy default); a build failure exits
loudly rather than silently falling back to the default pool size.

**OpenTelemetry: metrics + traces + logs over ONE pipeline** (`tracing` +
OTLP, not bare Prometheus). Human-readable stderr logging is always on
(`tracing-subscriber`'s `fmt` layer); when `OTEL_EXPORTER_OTLP_ENDPOINT` is
set, OTLP export layers/providers are added on top and ONE loud line is
logged either way ("OTLP export disabled (... unset)" or "OTLP export
enabled"). A missing/unreachable collector never crashes the daemon or
slows inference — every exporter is a BATCH exporter (bounded queue, drops
under backpressure) over gRPC/tonic with rustls (never openssl). One span
per HTTP request (`method`, `route`, `status`); within it, a `worker_call`
span carrying `queue_wait_ms` and `inference_ms` — the queue-wait vs.
compute split is the whole point of a serial worker (see `src/worker.rs`'s
module docs on how a `tracing::Span` is created request-side and recorded
onto from the worker OS thread). Resource attributes: `service.name`
(`OTEL_SERVICE_NAME`, default `lfm2d`), `service.version` (crate version),
and `lfm2d.model.<kind>_hash` per loaded model — set once, from
`main.rs`, AFTER models finish loading (weight hashes aren't known any
earlier; see `src/telemetry.rs`'s module docs). Metrics: `lfm2d.worker.queue_depth`
(observable gauge over an `AtomicUsize`, incremented on send, decremented
when the worker picks a command up), `lfm2d.request.duration_ms` (histogram,
by route+status), `lfm2d.inference.duration_ms` (histogram, by operation
kind), `lfm2d.requests` (counter). Verified against a real OTLP/gRPC
capture server with real checkpoints loaded (`kube_ordinal_v6` +
`LFM2.5-Encoder-350M-Prompt-Router`): spans arrived correctly nested
(`http_request` parenting `worker_call`) carrying `queue_wait_ms`/
`inference_ms`, all four metrics arrived, logs arrived with the effective
config and per-model weight hashes, and SIGTERM still drained and exited 0
cleanly despite transient export failures observed against one candidate
endpoint (see "Problems noted, not fixed" below — worth knowing, not a
code defect).

**Deploying on k8s** — `deploy/k8s.yaml` is a complete Deployment +
Service: `readinessProbe`/`livenessProbe` against `/readyz`/`/healthz`, a
`startupProbe` budgeted for a COLD PVC (local-disk load measured at
~1.4-3.2s for two heads here; a cold network-backed PVC can take far
longer, which is what the startup probe's generous budget is for — the
readiness/liveness probes only start once it succeeds),
`terminationGracePeriodSeconds` comfortably above the 10s drain cap,
`resources.requests` with a MEASURED memory number (not an estimate — see
the manifest's comment for the exact `/proc/<pid>/status` reading), CPU
`requests` WITHOUT `limits` (rationale in-manifest: rayon's thread pool
auto-sizes to the visible core count at startup and doesn't re-check a
limit applied later — a CPU limit just throttles the same thread count
against fewer cores, no benefit; this workload scales by replica count,
not by starving one replica), and OTLP env wiring. lfm2d is STATELESS — N
replicas behind the Service is the horizontal scaling path; a previous
version of this doc's k8s example pinned `replicas: 1` in a way that read
as a requirement, which was wrong and has been fixed (see `deploy/lfm2d.container`'s
trailing comment and `deploy/k8s.yaml` itself).

## Judgment calls worth knowing about

The task spec left a few things implicit; here's what was decided and why
(also in the relevant doc comments):

1. **Audit headers vs. audit body fields.** The spec requires "every
   inference response carries `{model_id, weight_hash}`" AND specifies
   `/embed`/`/predict` as bare TEI-compatible arrays with no room for those
   fields. Resolved by putting them in `X-Model-Id`/`X-Model-Weight-Hash`
   response headers for those two endpoints, and directly in the JSON body
   for `/v1/classify`/`/v1/route`/`/v1/cascade` (which are "our full
   contract," not TEI-compat).
2. **`/v1/cascade`'s `routes`/`severe_labels`.** The spec's request shape
   is `{"clauses": [...]}` only, but the library's `Cascade::run` also
   needs a route set and a severe-label set. Made these server-side startup
   config (`--cascade-route`, `--cascade-severe-label`) rather than
   request fields — a cascade specialist's lane set and severity
   definition are properties of the deployment, not something each caller
   should be re-specifying per call.
3. **`top_severity`'s meaning.** Read as "the classifier's own argmax
   label for this clause" (mirroring `/v1/classify`'s `top`), NOT the
   severe-label-set ranking sum (`ClauseVerdict::severity_score` in the
   library) — the latter isn't exposed as a separate number in v1, since a
   caller can already derive it from `severity_scores` plus their own
   knowledge of which labels they consider severe.
4. **One model per head kind.** No request carries a model-selection
   parameter, so this daemon serves at most one embedder, one classifier,
   one router at a time.

## Problems noted, not fixed (scope-limited by the task)

- No dtype/device flag — always F32 on CPU (the library's verified path).
  A `--dtype` flag would be a small addition if f16 throughput becomes
  worth the ~1.6× latency tradeoff the parent crate's `f16-halves-memory`
  finding documents.
- No trunk sharing across heads (`Lfm2Trunk::load_shared` + `from_trunk`)
  — each head loads its own trunk via `from_dir`. Correct for this
  service's general case (independently-trained checkpoints don't share a
  trunk), but a same-trunk deployment would pay for N trunks it doesn't
  need to.
- The quadlet's `HealthCmd` only proves the binary still execs, not that
  `/healthz` answers — the runtime image ships no `curl`/`wget` on purpose
  (minimal image; see the Containerfile). A real HTTP healthcheck should
  run from OUTSIDE the container against `/healthz`/`/readyz`.
- **Observed, not a code defect, but worth recording:** during live OTLP
  verification, the endpoint an `otlp-mcp` tool call returned
  (`127.0.0.1:36185`) accepted the TCP connection instantly but every
  actual gRPC export (logs, then traces) timed out against it — even at a
  20s `OTEL_EXPORTER_OTLP_TIMEOUT`, well past the 10s default — while the
  standard OTLP/gRPC port (`4317`, also listening on this box) accepted
  the exact same exporter code's traffic immediately and captured
  everything (spans correctly parented, all four metrics, logs with the
  effective config and weight hashes). Whatever the ephemeral port is
  multiplexing, it wasn't reliably forwarding gRPC under this session's
  conditions; `4317` was. Confirms the required behavior either way
  (export failures against the flaky endpoint did NOT crash the daemon or
  block SIGTERM's clean exit 0), but an operator pointing
  `OTEL_EXPORTER_OTLP_ENDPOINT` at a genuinely reachable collector — which
  `4317` was standing in for here — is a precondition this daemon can't
  itself fix.
- `WorkerHandle::spawn_crash_on_panic`'s monitor mechanism is proven
  end-to-end via `tests/worker_crash_monitor.rs`, but that test needs a
  `LFM2D_TEST_CRASH_ON_WORKER_PANIC`-gated stub-engine branch in `main.rs`
  to avoid depending on real checkpoints being present in CI. The pure
  decision logic (`worker_thread_outcome_is_a_crash`) is ALSO unit-tested
  directly, so the crash-vs-clean-exit distinction has a fast, model-free
  test in addition to the slower real-binary one.
