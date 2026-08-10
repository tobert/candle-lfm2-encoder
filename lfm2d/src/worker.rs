//! The single inference worker: one OS thread owns every loaded model.
//! Handlers never touch a model directly — they send a [`WorkerCommand`]
//! down an unbounded [`std::sync::mpsc`] channel and `.await` a
//! [`tokio::sync::oneshot`] reply. This is deliberate serialization, not an
//! accidental bottleneck: a single forward pass already saturates ~13.6 of
//! this box's cores (measured), so request-level concurrency has no
//! headroom to spend.
//!
//! [`InferenceEngine`] is the seam that lets router/handler tests run
//! without candle: [`crate::engine_real::RealEngine`] backs production,
//! [`crate::engine_stub::StubEngine`] backs the HTTP-contract test suite.
//! Both are driven through the exact same channel/thread plumbing here, so
//! a router test genuinely exercises the same code path production traffic
//! does, up to the engine implementation itself.
//!
//! # Observability
//!
//! Every command carries a `tracing::Span` (created by the async method
//! that enqueues it, so it nests under whatever HTTP-request span is
//! current) and an enqueue timestamp. The worker thread — NOT the async
//! task awaiting the reply — records `queue_wait_ms` (time spent sitting in
//! the channel) and `inference_ms` (time spent inside [`InferenceEngine`])
//! onto that span, and reports `inference_ms` to the
//! `lfm2d.inference.duration_ms` histogram: `tracing::Span` is `Send`/
//! `Clone` and not thread-affine, so recording from the worker OS thread is
//! the intended use, not a hack. A separate `AtomicUsize` (`queue_depth`)
//! backs the `lfm2d.worker.queue_depth` observable gauge — incremented in
//! [`WorkerHandle::send`], decremented the moment a command is dequeued.
//!
//! # Crash-only on worker death
//!
//! [`WorkerHandle::spawn`] never exits the process on its own — every
//! router/HTTP test in this crate relies on being able to make the worker
//! thread panic (simulating "a loaded model's forward pass corrupted
//! something") and observe a clean 500 afterward, in the SAME test
//! process. [`WorkerHandle::spawn_crash_on_panic`] is the production-only
//! variant `main.rs` calls instead: identical worker thread, PLUS a monitor
//! thread that calls `std::process::exit(1)` if (and only if) the worker
//! thread terminates via panic — never on the clean channel-closure exit a
//! graceful shutdown produces. See [`worker_thread_outcome_is_a_crash`] for
//! the (separately unit-tested) decision logic, and
//! `tests/worker_crash_monitor.rs` for the integration test that exercises
//! this through the real compiled binary.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::oneshot;
use tracing::Span;

use crate::types::{CascadeResponse, ClassifyResult, EmbedKind, LabelScore, ModelInfo, RouteResponse};

/// A worker-side failure, already classified into the two HTTP status
/// families the API contract promises: a caller-supplied problem (400) or
/// an internal/model failure (500). Mapped to [`crate::types::ApiError`] at
/// the handler boundary — see `server.rs`.
#[derive(Debug, Clone)]
pub enum WorkerError {
    /// The request itself, or the server's current configuration relative
    /// to it (e.g. `/predict` called with no classifier loaded), makes this
    /// call impossible to satisfy. Never the model's fault.
    BadRequest(String),
    /// A loaded model failed during inference, or the worker thread itself
    /// is gone. Loud, not swallowed: this crate's house rule is that a
    /// worker failure is a 500, never a silently-wrong 200.
    Internal(String),
}

impl WorkerError {
    pub fn message(&self) -> &str {
        match self {
            WorkerError::BadRequest(m) | WorkerError::Internal(m) => m,
        }
    }
}

/// Map the library's own error type onto the two-bucket [`WorkerError`]
/// taxonomy. [`candle_lfm2_encoder::Error::InvalidInput`] is, by the
/// library's own doc comment, "a caller-supplied argument that is
/// structurally invalid" — exactly a 400. Everything else (I/O, config
/// parse, tokenizer, candle numerics) is this SERVICE's problem to explain,
/// not the caller's mistake to fix by rephrasing the request — a 500.
impl From<candle_lfm2_encoder::Error> for WorkerError {
    fn from(err: candle_lfm2_encoder::Error) -> Self {
        match err {
            candle_lfm2_encoder::Error::InvalidInput(msg) => WorkerError::BadRequest(msg),
            other => WorkerError::Internal(other.to_string()),
        }
    }
}

/// Everything [`InferenceEngine::embed`] hands back: the pooled vectors
/// plus the audit pair that travels out-of-band as response headers (see
/// the crate root docs — `/embed`'s JSON body stays a bare TEI-compatible
/// array, so `model_id`/`weight_hash` cannot live in it).
#[derive(Debug, Clone)]
pub struct EmbedOutcome {
    pub vectors: Vec<Vec<f32>>,
    pub model_id: String,
    pub weight_hash: String,
}

/// As [`EmbedOutcome`], for `/predict`.
#[derive(Debug, Clone)]
pub struct PredictOutcome {
    pub per_input: Vec<Vec<LabelScore>>,
    pub model_id: String,
    pub weight_hash: String,
}

/// The engine seam. Every method is synchronous (candle inference is
/// CPU-bound, blocking work — there is nothing to `.await` inside it) and
/// runs entirely on the worker thread; `&self` because a loaded model is
/// never mutated after construction.
pub trait InferenceEngine: Send + 'static {
    fn list_models(&self) -> Vec<ModelInfo>;
    fn embed(&self, inputs: &[String], kind: EmbedKind) -> Result<EmbedOutcome, WorkerError>;
    fn predict(&self, inputs: &[String]) -> Result<PredictOutcome, WorkerError>;
    fn classify(&self, inputs: &[String]) -> Result<Vec<ClassifyResult>, WorkerError>;
    fn route(&self, input: &str, routes: &[String]) -> Result<RouteResponse, WorkerError>;
    fn cascade(&self, clauses: &[String]) -> Result<CascadeResponse, WorkerError>;
}

/// Metadata every [`WorkerCommand`] variant carries alongside its
/// request-specific fields — enqueue time (for `queue_wait_ms`) and the
/// span it should record onto (for both `queue_wait_ms` and
/// `inference_ms`; see the module docs' "Observability" section).
pub(crate) struct Enqueued {
    queued_at: Instant,
    span: Span,
}

/// One request to the worker thread, paired with the `oneshot` it must
/// reply on. `pub(crate)` — handlers go through [`WorkerHandle`]'s typed
/// async methods, never build this directly.
pub(crate) enum WorkerCommand {
    ListModels {
        reply: oneshot::Sender<Vec<ModelInfo>>,
        meta: Enqueued,
    },
    Embed {
        inputs: Vec<String>,
        kind: EmbedKind,
        reply: oneshot::Sender<Result<EmbedOutcome, WorkerError>>,
        meta: Enqueued,
    },
    Predict {
        inputs: Vec<String>,
        reply: oneshot::Sender<Result<PredictOutcome, WorkerError>>,
        meta: Enqueued,
    },
    Classify {
        inputs: Vec<String>,
        reply: oneshot::Sender<Result<Vec<ClassifyResult>, WorkerError>>,
        meta: Enqueued,
    },
    Route {
        input: String,
        routes: Vec<String>,
        reply: oneshot::Sender<Result<RouteResponse, WorkerError>>,
        meta: Enqueued,
    },
    Cascade {
        clauses: Vec<String>,
        reply: oneshot::Sender<Result<CascadeResponse, WorkerError>>,
        meta: Enqueued,
    },
}

/// A cheap-to-clone handle to the worker thread. Every axum handler holds
/// one (via `Arc` in [`crate::server::AppState`]); sending a command never
/// blocks the async runtime — the underlying channel is unbounded, and
/// awaiting the `oneshot` reply yields to the executor like any other
/// future.
#[derive(Clone)]
pub struct WorkerHandle {
    tx: std::sync::mpsc::Sender<WorkerCommand>,
    /// Commands sent but not yet dequeued by the worker thread — the
    /// `lfm2d.worker.queue_depth` observable gauge reads this (see
    /// `telemetry::register_queue_depth_gauge`). Incremented in
    /// [`Self::send`], decremented at the top of the worker loop's match.
    queue_depth: Arc<AtomicUsize>,
}

/// A single dispatch point every `WorkerCommand::Foo { .. }` variant's
/// worker-thread handling goes through: enter the span, record
/// `queue_wait_ms`, decrement the queue-depth gauge, time the actual
/// engine call, record `inference_ms` + the histogram, reply. Written once
/// as a closure-taking helper so the six near-identical match arms below
/// don't each re-implement this sequence (and can't drift out of sync).
fn run_timed<T>(
    meta: Enqueued,
    queue_depth: &AtomicUsize,
    operation: &'static str,
    f: impl FnOnce() -> T,
) -> T {
    let Enqueued { queued_at, span } = meta;
    let _guard = span.enter();
    let queue_wait = queued_at.elapsed();
    span.record("queue_wait_ms", queue_wait.as_secs_f64() * 1000.0);
    queue_depth.fetch_sub(1, Ordering::SeqCst);

    let t0 = Instant::now();
    let result = f();
    let inference = t0.elapsed();
    span.record("inference_ms", inference.as_secs_f64() * 1000.0);
    crate::telemetry::record_inference_duration(operation, inference);
    result
}

impl WorkerHandle {
    /// Spawn the worker thread over `engine` and return a handle to it. The
    /// thread runs until every [`WorkerHandle`] clone (and thus every
    /// sender) is dropped, at which point `rx` iteration ends and the
    /// thread exits cleanly. Production wants crash-only semantics on a
    /// worker panic — see [`Self::spawn_crash_on_panic`] — but THIS
    /// function deliberately does not install that behavior: several tests
    /// in `tests/router_stub.rs` intentionally panic the worker thread
    /// in-process to assert the 500 mapping, and must not take the whole
    /// test binary down with it.
    pub fn spawn<E: InferenceEngine>(engine: E) -> Self {
        Self::spawn_inner(engine, false)
    }

    /// As [`Self::spawn`], but additionally installs a monitor thread that
    /// calls `std::process::exit(1)` if the worker thread terminates via
    /// panic. Production-only (`main.rs`'s job, not this module's default):
    /// a worker-thread panic must not leave the HTTP server up answering
    /// 500s forever while `/healthz` still says 200 — kubelet only restarts
    /// a pod whose process actually exits. A clean exit (every
    /// [`WorkerHandle`] clone dropped, e.g. at the end of a graceful
    /// shutdown) is NOT treated as a crash — see
    /// [`worker_thread_outcome_is_a_crash`].
    pub fn spawn_crash_on_panic<E: InferenceEngine>(engine: E) -> Self {
        Self::spawn_inner(engine, true)
    }

    fn spawn_inner<E: InferenceEngine>(engine: E, exit_on_panic: bool) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<WorkerCommand>();
        let queue_depth = Arc::new(AtomicUsize::new(0));
        let worker_queue_depth = queue_depth.clone();
        let join = std::thread::Builder::new()
            .name("lfm2d-worker".to_string())
            .spawn(move || {
                for cmd in rx {
                    // Every branch replies exactly once. A dropped receiver
                    // (the HTTP request that asked was itself dropped,
                    // e.g. client disconnect) makes `.send()` return `Err`,
                    // which we discard — there is nothing to escalate to,
                    // the asker is gone.
                    match cmd {
                        WorkerCommand::ListModels { reply, meta } => {
                            let _ = reply.send(run_timed(meta, &worker_queue_depth, "list_models", || {
                                engine.list_models()
                            }));
                        }
                        WorkerCommand::Embed { inputs, kind, reply, meta } => {
                            let _ = reply.send(run_timed(meta, &worker_queue_depth, "embed", || {
                                engine.embed(&inputs, kind)
                            }));
                        }
                        WorkerCommand::Predict { inputs, reply, meta } => {
                            let _ = reply.send(run_timed(meta, &worker_queue_depth, "predict", || {
                                engine.predict(&inputs)
                            }));
                        }
                        WorkerCommand::Classify { inputs, reply, meta } => {
                            let _ = reply.send(run_timed(meta, &worker_queue_depth, "classify", || {
                                engine.classify(&inputs)
                            }));
                        }
                        WorkerCommand::Route { input, routes, reply, meta } => {
                            let _ = reply.send(run_timed(meta, &worker_queue_depth, "route", || {
                                engine.route(&input, &routes)
                            }));
                        }
                        WorkerCommand::Cascade { clauses, reply, meta } => {
                            let _ = reply.send(run_timed(meta, &worker_queue_depth, "cascade", || {
                                engine.cascade(&clauses)
                            }));
                        }
                    }
                }
            })
            .expect("failed to spawn the lfm2d inference worker thread");

        if exit_on_panic {
            std::thread::Builder::new()
                .name("lfm2d-worker-monitor".to_string())
                .spawn(move || {
                    let outcome = join.join();
                    if worker_thread_outcome_is_a_crash(&outcome) {
                        eprintln!(
                            "lfm2d: the inference worker thread panicked — exiting so the \
                             supervisor restarts this pod. An alive-but-broken worker must \
                             never keep answering 500s while /healthz still says 200."
                        );
                        std::process::exit(1);
                    }
                    // Ok(()) means the worker's `for cmd in rx` loop ended
                    // because every WorkerHandle (and thus every sender)
                    // was dropped — the expected end of a graceful
                    // shutdown, already underway on `main`'s task. Nothing
                    // to do here.
                })
                .expect("failed to spawn the lfm2d worker crash monitor thread");
        }

        Self { tx, queue_depth }
    }

    /// Clone of the queue-depth counter — `main.rs` passes this to
    /// [`crate::telemetry::register_queue_depth_gauge`].
    pub fn queue_depth_handle(&self) -> Arc<AtomicUsize> {
        self.queue_depth.clone()
    }

    /// `Err` only if the worker thread is gone (panicked or otherwise
    /// exited) — this is always a 500, never a caller mistake, so callers
    /// convert directly to [`WorkerError::Internal`].
    fn send(&self, cmd: WorkerCommand) -> Result<(), WorkerError> {
        self.queue_depth.fetch_add(1, Ordering::SeqCst);
        self.tx.send(cmd).map_err(|_| {
            // The send failed, so the worker will never dequeue this one —
            // undo the optimistic increment above rather than leaking a
            // phantom entry into the gauge.
            self.queue_depth.fetch_sub(1, Ordering::SeqCst);
            WorkerError::Internal("the inference worker thread is gone".to_string())
        })
    }

    async fn recv<T>(rx: oneshot::Receiver<T>) -> Result<T, WorkerError> {
        rx.await
            .map_err(|_| WorkerError::Internal("the inference worker dropped the reply channel without answering".to_string()))
    }

    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, WorkerError> {
        let span = tracing::info_span!("worker_call", operation = "list_models", queue_wait_ms = tracing::field::Empty, inference_ms = tracing::field::Empty);
        let (tx, rx) = oneshot::channel();
        self.send(WorkerCommand::ListModels { reply: tx, meta: Enqueued { queued_at: Instant::now(), span } })?;
        Self::recv(rx).await
    }

    pub async fn embed(&self, inputs: Vec<String>, kind: EmbedKind) -> Result<EmbedOutcome, WorkerError> {
        let span = tracing::info_span!("worker_call", operation = "embed", queue_wait_ms = tracing::field::Empty, inference_ms = tracing::field::Empty);
        let (tx, rx) = oneshot::channel();
        self.send(WorkerCommand::Embed { inputs, kind, reply: tx, meta: Enqueued { queued_at: Instant::now(), span } })?;
        Self::recv(rx).await?
    }

    pub async fn predict(&self, inputs: Vec<String>) -> Result<PredictOutcome, WorkerError> {
        let span = tracing::info_span!("worker_call", operation = "predict", queue_wait_ms = tracing::field::Empty, inference_ms = tracing::field::Empty);
        let (tx, rx) = oneshot::channel();
        self.send(WorkerCommand::Predict { inputs, reply: tx, meta: Enqueued { queued_at: Instant::now(), span } })?;
        Self::recv(rx).await?
    }

    pub async fn classify(&self, inputs: Vec<String>) -> Result<Vec<ClassifyResult>, WorkerError> {
        let span = tracing::info_span!("worker_call", operation = "classify", queue_wait_ms = tracing::field::Empty, inference_ms = tracing::field::Empty);
        let (tx, rx) = oneshot::channel();
        self.send(WorkerCommand::Classify { inputs, reply: tx, meta: Enqueued { queued_at: Instant::now(), span } })?;
        Self::recv(rx).await?
    }

    pub async fn route(&self, input: String, routes: Vec<String>) -> Result<RouteResponse, WorkerError> {
        let span = tracing::info_span!("worker_call", operation = "route", queue_wait_ms = tracing::field::Empty, inference_ms = tracing::field::Empty);
        let (tx, rx) = oneshot::channel();
        self.send(WorkerCommand::Route { input, routes, reply: tx, meta: Enqueued { queued_at: Instant::now(), span } })?;
        Self::recv(rx).await?
    }

    pub async fn cascade(&self, clauses: Vec<String>) -> Result<CascadeResponse, WorkerError> {
        let span = tracing::info_span!("worker_call", operation = "cascade", queue_wait_ms = tracing::field::Empty, inference_ms = tracing::field::Empty);
        let (tx, rx) = oneshot::channel();
        self.send(WorkerCommand::Cascade { clauses, reply: tx, meta: Enqueued { queued_at: Instant::now(), span } })?;
        Self::recv(rx).await?
    }
}

/// Pure decision logic for the crash monitor spawned by
/// [`WorkerHandle::spawn_crash_on_panic`], pulled out specifically so it's
/// unit-testable without ever calling `std::process::exit` from inside
/// `cargo test` (which would kill the test runner, not just fail one
/// test). `true` (a crash) iff the worker thread's closure unwound via
/// panic rather than returning normally.
fn worker_thread_outcome_is_a_crash(outcome: &std::thread::Result<()>) -> bool {
    outcome.is_err()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panicked_worker_thread_is_a_crash() {
        let outcome: std::thread::Result<()> = Err(Box::new("simulated panic payload"));
        assert!(worker_thread_outcome_is_a_crash(&outcome));
    }

    #[test]
    fn a_cleanly_returned_worker_thread_is_not_a_crash() {
        let outcome: std::thread::Result<()> = Ok(());
        assert!(!worker_thread_outcome_is_a_crash(&outcome));
    }
}
