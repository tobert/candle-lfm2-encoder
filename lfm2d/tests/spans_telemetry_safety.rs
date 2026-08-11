//! **The most important test in the `/v1/spans` change.** `POST /v1/spans`
//! and `POST /v1/spans/credentials` exist to find live credentials in
//! request text — which means request text contains live credentials by
//! definition. This file proves a known secret string, embedded in a
//! request, NEVER shows up anywhere in this crate's tracing output: not on
//! a span field, not on a log event, not stringified into a Debug dump.
//!
//! No OTLP collector is involved (the automated suite must not depend on
//! one) — a custom [`tracing_subscriber::Layer`] captures every field
//! value emitted on every span/event created while the request is served,
//! via [`tracing::subscriber::set_default`] (thread-local, so this composes
//! safely with `cargo test`'s parallel test threads and never touches the
//! process-wide default `main.rs`/other tests might install). Everything
//! this crate exports over OTLP is built from exactly these same
//! `tracing::Span`/`tracing::event!` calls (see `telemetry.rs`'s module
//! docs — the OTLP layers are additional subscriber layers over the same
//! spans/events, not a separate code path with its own data), so "never
//! appears in what this Layer captures" is equivalent to "never appears in
//! what an OTLP collector would receive."

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;
use tracing::field::{Field, Visit};
use tracing::span;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::Registry;

use lfm2d::engine_stub::{StubEngine, StubTokenClassifier};
use lfm2d::server::{build_router, AppState};
use lfm2d::worker::WorkerHandle;

/// Stringifies every field value it's handed (`{name}={value}`) into a
/// shared buffer — deliberately crude (Debug-formats everything, no
/// filtering) so this test is checking "does the substring appear ANYWHERE
/// tracing touched," not trusting a narrower, more easily-gamed check.
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<String>>>);

struct StringVisitor<'a>(&'a mut Vec<String>);

impl Visit for StringVisitor<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0.push(format!("{}={value:?}", field.name()));
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.push(format!("{}={value}", field.name()));
    }
}

impl<S: tracing::Subscriber> Layer<S> for Capture {
    fn on_new_span(&self, attrs: &span::Attributes<'_>, _id: &span::Id, _ctx: Context<'_, S>) {
        let mut buf = Vec::new();
        attrs.record(&mut StringVisitor(&mut buf));
        self.0.lock().unwrap().extend(buf);
    }

    fn on_record(&self, _id: &span::Id, values: &span::Record<'_>, _ctx: Context<'_, S>) {
        let mut buf = Vec::new();
        values.record(&mut StringVisitor(&mut buf));
        self.0.lock().unwrap().extend(buf);
    }

    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut buf = Vec::new();
        event.record(&mut StringVisitor(&mut buf));
        self.0.lock().unwrap().extend(buf);
    }
}

fn router_with(engine: StubEngine, log_input_hash: bool) -> axum::Router {
    let worker = WorkerHandle::spawn(engine).with_log_input_hash(log_input_hash);
    build_router(AppState { worker, ready: Arc::new(AtomicBool::new(true)) })
}

async fn post_json(router: axum::Router, path: &str, body: Value) -> (StatusCode, Value) {
    let resp = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

/// A secret distinctive enough that finding its literal bytes anywhere in
/// captured telemetry is unambiguous proof of a leak, not a coincidence.
const SECRET: &str = "sk-KNOWN_TEST_SECRET_do_not_leak_9f8e7d6c5b4a";

fn secret_bearing_text() -> String {
    format!("here is an api key: {SECRET} — please rotate it")
}

async fn run_request(path: &str, log_input_hash: bool) -> (StatusCode, Value, Vec<String>) {
    let capture = Capture::default();
    let subscriber = Registry::default().with(capture.clone());
    let _guard = tracing::subscriber::set_default(subscriber);

    let engine = StubEngine {
        token_classifiers: vec![StubTokenClassifier::new("pii-detector")],
        ..StubEngine::default()
    };
    let router = router_with(engine, log_input_hash);
    let (status, body) = post_json(router, path, json!({"inputs": [secret_bearing_text()]})).await;

    drop(_guard);
    let captured = capture.0.lock().unwrap().clone();
    (status, body, captured)
}

#[tokio::test]
async fn v1_spans_never_leaks_the_secret_string_into_telemetry() {
    let (status, _, captured) = run_request("/v1/spans", false).await;
    assert_eq!(status, StatusCode::OK);

    let joined = captured.join("\n");
    assert!(!joined.contains(SECRET), "the secret leaked into telemetry:\n{joined}");
    assert!(!joined.contains(&secret_bearing_text()), "the full input text leaked into telemetry:\n{joined}");
}

#[tokio::test]
async fn v1_spans_credentials_never_leaks_the_secret_string_into_telemetry() {
    let (status, _, captured) = run_request("/v1/spans/credentials", false).await;
    assert_eq!(status, StatusCode::OK);

    let joined = captured.join("\n");
    assert!(!joined.contains(SECRET), "the secret leaked into telemetry:\n{joined}");
    assert!(!joined.contains(&secret_bearing_text()), "the full input text leaked into telemetry:\n{joined}");
}

/// With `--log-input-hash` OFF (the default), no `input_hash` field is
/// recorded at all — not an empty one, none. Distinguishes "the hash is
/// empty" from "the feature is off," which matters because an empty-string
/// hash field would itself be a shape a careless refactor could fill in by
/// accident.
#[tokio::test]
async fn log_input_hash_off_by_default_records_no_input_hash_field() {
    let (status, _, captured) = run_request("/v1/spans", false).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        captured.iter().all(|line| !line.starts_with("input_hash=")),
        "input_hash must not be recorded when --log-input-hash is off: {captured:?}"
    );
}

/// With `--log-input-hash` ON, a hash IS recorded (proving the opt-in path
/// actually does something) — but it is a hash, not the secret: the
/// captured value must never equal or contain the secret bytes, and must
/// look like the 64-hex-char sha256 this crate's hasher produces (see
/// `hash::sha256_hex_bytes`), not a copy of the input.
#[tokio::test]
async fn log_input_hash_on_records_a_hash_never_the_secret_itself() {
    let (status, _, captured) = run_request("/v1/spans", true).await;
    assert_eq!(status, StatusCode::OK);

    let hash_lines: Vec<&String> = captured.iter().filter(|line| line.starts_with("input_hash=")).collect();
    assert!(!hash_lines.is_empty(), "--log-input-hash=true must record an input_hash field: {captured:?}");
    for line in &hash_lines {
        assert!(!line.contains(SECRET), "input_hash carried the secret verbatim: {line}");
        let hex = line.trim_start_matches("input_hash=");
        assert_eq!(hex.len(), 64, "expected a sha256 hex digest, got: {hex}");
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()), "not hex: {hex}");
    }

    // And the blanket substring check still holds even with hashing on —
    // the opt-in is for a HASH, never a path back to raw text.
    let joined = captured.join("\n");
    assert!(!joined.contains(SECRET), "the secret leaked into telemetry even with --log-input-hash on:\n{joined}");
}

/// Span offsets are exactly the coordinates that turn "a credential is
/// somewhere in this request" into "here are its exact bytes" — this
/// crate's telemetry must not record them either, only the response body
/// does. Cross-checked against the response body's own offsets (proving
/// this isn't vacuous — the stub really did produce a nonzero span for
/// this input) rather than just asserting "no numbers appear," which would
/// be too broad a claim (queue_wait_ms/inference_ms are legitimately
/// numeric).
#[tokio::test]
async fn span_offset_fields_never_appear_on_any_captured_span() {
    let (status, body, captured) = run_request("/v1/spans", false).await;
    assert_eq!(status, StatusCode::OK);
    let spans = body[0].as_array().expect("response body");
    assert!(!spans.is_empty(), "test setup must produce at least one span");

    for line in &captured {
        assert!(
            !line.starts_with("start=") && !line.starts_with("end=") && !line.starts_with("offsets="),
            "a span-offset-shaped field leaked into telemetry: {line}"
        );
    }
}
