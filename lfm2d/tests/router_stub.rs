//! HTTP-level router tests, driven over a real `axum::Router` + real
//! [`WorkerHandle`] + real worker thread, but backed by
//! [`StubEngine`] instead of any candle model — no weights on disk are
//! required for this file to run. Exercises: `/healthz`/`/readyz`
//! semantics, each endpoint's happy path and 400s, the error JSON shape,
//! and audit-header/audit-body passthrough.
//!
//! Requests are driven with `tower::ServiceExt::oneshot` per the task's own
//! suggestion — no real socket, no real port, just the `Service` trait
//! `axum::Router` implements. The same `Router` this crate builds is what
//! `serve()` binds a Unix socket and a TCP listener to in production (see
//! `src/server.rs`), so these tests exercise the actual routing/handler
//! code, not a parallel test-only copy of it.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

use lfm2d::engine_stub::StubEngine;
use lfm2d::server::{build_router, AppState};
use lfm2d::worker::WorkerHandle;

fn router_over(engine: StubEngine) -> axum::Router {
    router_with_ready(engine, true)
}

fn router_with_ready(engine: StubEngine, ready: bool) -> axum::Router {
    let worker = WorkerHandle::spawn(engine);
    build_router(AppState { worker, ready: Arc::new(AtomicBool::new(ready)) })
}

async fn get(router: axum::Router, path: &str) -> (StatusCode, String, axum::http::HeaderMap) {
    let resp = router
        .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(body.to_vec()).unwrap(), headers)
}

async fn post_json(
    router: axum::Router,
    path: &str,
    body: Value,
) -> (StatusCode, Value, axum::http::HeaderMap) {
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
    let headers = resp.headers().clone();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json, headers)
}

// ------------------------------------------------------------- healthz

#[tokio::test]
async fn healthz_is_200_ok_regardless_of_readiness() {
    let (status, body, _) = get(router_with_ready(StubEngine::fully_configured(), false), "/healthz").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, "ok");
}

// -------------------------------------------------------------- readyz

#[tokio::test]
async fn readyz_is_503_before_ready_and_200_after() {
    let (status, _, _) = get(router_with_ready(StubEngine::fully_configured(), false), "/readyz").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);

    let (status, _, _) = get(router_with_ready(StubEngine::fully_configured(), true), "/readyz").await;
    assert_eq!(status, StatusCode::OK);
}

// ----------------------------------------------------------- /v1/models

#[tokio::test]
async fn v1_models_lists_every_configured_head_with_hash_and_kind() {
    let (status, body, _) = get(router_over(StubEngine::fully_configured()), "/v1/models").await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_str(&body).unwrap();
    let models = v.as_array().unwrap();
    assert_eq!(models.len(), 3, "embedder + classifier + router");

    let kinds: Vec<&str> = models.iter().map(|m| m["kind"].as_str().unwrap()).collect();
    assert!(kinds.contains(&"embedder"));
    assert!(kinds.contains(&"classifier"));
    assert!(kinds.contains(&"router"));

    let classifier = models.iter().find(|m| m["kind"] == "classifier").unwrap();
    assert_eq!(
        classifier["labels"],
        json!(["destructive", "informative", "mutating"])
    );
    // weight_hash format: 64 hex chars, even for the fake stub hash.
    let hash = classifier["weight_hash"].as_str().unwrap();
    assert_eq!(hash.len(), 64);
    assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));

    let embedder = models.iter().find(|m| m["kind"] == "embedder").unwrap();
    assert!(embedder.get("labels").is_none(), "an embedder has no fixed label set");
}

#[tokio::test]
async fn v1_models_reflects_a_partially_configured_server() {
    let engine = StubEngine {
        router: Some(lfm2d::engine_stub::StubModel::new("only-router")),
        ..StubEngine::default()
    };
    let (status, body, _) = get(router_over(engine), "/v1/models").await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v.as_array().unwrap().len(), 1);
    assert_eq!(v[0]["kind"], "router");
}

// ---------------------------------------------------------------- /embed

#[tokio::test]
async fn embed_happy_path_returns_a_bare_array_with_audit_headers() {
    let (status, body, headers) =
        post_json(router_over(StubEngine::fully_configured()), "/embed", json!({"inputs": "hello"})).await;
    assert_eq!(status, StatusCode::OK);
    // Bare TEI-shaped array: no wrapper object.
    let vectors = body.as_array().expect("response must be a bare JSON array");
    assert_eq!(vectors.len(), 1, "one vector for one input");
    assert_eq!(headers.get("x-model-id").unwrap(), "stub-embedder");
    assert_eq!(headers.get("x-model-weight-hash").unwrap().len(), 64);
}

#[tokio::test]
async fn embed_accepts_a_batch_of_inputs() {
    let (status, body, _) =
        post_json(router_over(StubEngine::fully_configured()), "/embed", json!({"inputs": ["a", "b", "c"]})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body.as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn embed_kind_selects_query_vs_document_pooling() {
    let engine = StubEngine::fully_configured();
    let (_, doc_body, _) =
        post_json(router_over(engine.clone()), "/embed", json!({"inputs": "same text"})).await;
    let (_, query_body, _) = post_json(
        router_over(engine),
        "/embed",
        json!({"inputs": "same text", "kind": "query"}),
    )
    .await;
    // The stub encodes `kind` into vector element [1] — document (default)
    // and query must differ, proving the field is actually threaded
    // through to the engine and not silently ignored.
    assert_ne!(doc_body[0][1], query_body[0][1]);
}

#[tokio::test]
async fn embed_rejects_empty_inputs_with_400_and_error_shape() {
    let (status, body, _) =
        post_json(router_over(StubEngine::fully_configured()), "/embed", json!({"inputs": []})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]["message"].as_str().unwrap().contains("empty"));
    assert_eq!(body["error"]["type"], "bad_request");
}

#[tokio::test]
async fn embed_with_no_embedder_configured_is_400_not_500() {
    let mut engine = StubEngine::fully_configured();
    engine.embedder = None;
    let (status, body, _) = post_json(router_over(engine), "/embed", json!({"inputs": "hi"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]["message"].as_str().unwrap().contains("embedder"));
}

#[tokio::test]
async fn embed_malformed_json_body_is_400_with_the_error_shape() {
    let router = router_over(StubEngine::fully_configured());
    let resp = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/embed")
                .header("content-type", "application/json")
                .body(Body::from("{ this is not json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&bytes).expect("even a malformed-input error is our JSON shape");
    assert_eq!(v["error"]["type"], "bad_request");
}

// -------------------------------------------------------------- /predict

#[tokio::test]
async fn predict_happy_path_covers_every_label_sorted_descending() {
    let (status, body, headers) = post_json(
        router_over(StubEngine::fully_configured()),
        "/predict",
        json!({"inputs": ["kubectl delete ns prod"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let per_input = body.as_array().unwrap();
    assert_eq!(per_input.len(), 1);
    let labels = per_input[0].as_array().unwrap();
    assert_eq!(labels.len(), 3, "full softmax over ALL labels, never just top-1");
    let scores: Vec<f64> = labels.iter().map(|l| l["score"].as_f64().unwrap()).collect();
    assert!(scores.windows(2).all(|w| w[0] >= w[1]), "must be sorted descending: {scores:?}");
    assert_eq!(headers.get("x-model-id").unwrap(), "stub-classifier");
}

#[tokio::test]
async fn predict_rejects_empty_inputs() {
    let (status, _, _) =
        post_json(router_over(StubEngine::fully_configured()), "/predict", json!({"inputs": []})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

// ----------------------------------------------------------- /v1/classify

#[tokio::test]
async fn v1_classify_carries_scores_top_model_id_and_weight_hash_in_body() {
    let (status, body, _) = post_json(
        router_over(StubEngine::fully_configured()),
        "/v1/classify",
        json!({"inputs": ["rm -rf /"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let result = &body[0];
    assert!(result["scores"].as_object().unwrap().len() == 3);
    assert!(result["top"].is_string());
    assert_eq!(result["model_id"], "stub-classifier");
    assert_eq!(result["weight_hash"].as_str().unwrap().len(), 64);

    // Full softmax: scores must sum to ~1.0.
    let sum: f64 = result["scores"].as_object().unwrap().values().map(|v| v.as_f64().unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-3, "sum={sum}");
}

#[tokio::test]
async fn v1_classify_with_no_classifier_configured_is_400() {
    let mut engine = StubEngine::fully_configured();
    engine.classifier = None;
    let (status, body, _) =
        post_json(router_over(engine), "/v1/classify", json!({"inputs": ["hi"]})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]["message"].as_str().unwrap().contains("classifier"));
}

// --------------------------------------------------------------- /v1/route

#[tokio::test]
async fn v1_route_returns_raw_cosines_only_no_softmax_field() {
    let (status, body, _) = post_json(
        router_over(StubEngine::fully_configured()),
        "/v1/route",
        json!({"input": "kubectl delete ns prod", "routes": ["k8s", "shell"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["model_id"], "stub-router");
    assert_eq!(body["weight_hash"].as_str().unwrap().len(), 64);
    let routes = body["routes"].as_array().unwrap();
    assert_eq!(routes.len(), 2);
    for r in routes {
        assert!(r.get("score").is_none(), "RAW COSINE ONLY — no softmax score field");
        assert!(r.get("prob").is_none());
        assert!(r["cosine"].is_number());
    }
}

#[tokio::test]
async fn v1_route_rejects_empty_input_and_empty_routes() {
    let engine = StubEngine::fully_configured();
    let (status, _, _) =
        post_json(router_over(engine.clone()), "/v1/route", json!({"input": "", "routes": ["k8s"]})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, _, _) =
        post_json(router_over(engine), "/v1/route", json!({"input": "hi", "routes": []})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn v1_route_with_no_router_configured_is_400() {
    let mut engine = StubEngine::fully_configured();
    engine.router = None;
    let (status, _, _) = post_json(
        router_over(engine),
        "/v1/route",
        json!({"input": "hi", "routes": ["a"]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

// ------------------------------------------------------------- /v1/cascade

#[tokio::test]
async fn v1_cascade_matches_the_moderations_flavored_contract() {
    let (status, body, _) = post_json(
        router_over(StubEngine::fully_configured()),
        "/v1/cascade",
        json!({"clauses": ["cd vendor/legacy-plugin", "rm -rf ."]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    assert!(body.get("flagged").is_none(), "no flagged boolean anywhere");
    assert!(body.get("threshold").is_none(), "no threshold anywhere");

    assert!(body["winner"]["index"].is_number());
    assert!(body["winner"]["clause"].is_string());
    assert_eq!(body["winner"]["severity_scores"].as_object().unwrap().len(), 3);

    assert!(body["lane"]["route"].is_string());
    assert!(body["lane"]["cosine"].is_number());

    let clauses = body["clauses"].as_array().unwrap();
    assert_eq!(clauses.len(), 2);
    for c in clauses {
        assert!(c["index"].is_number());
        assert!(c["clause"].is_string());
        assert_eq!(c["severity_scores"].as_object().unwrap().len(), 3);
        assert!(c["top_severity"].is_string());
        assert!(c.get("flagged").is_none());
    }

    let models = body["models"].as_array().unwrap();
    assert_eq!(models.len(), 2, "classifier + router");
    for m in models {
        assert!(m["model_id"].is_string());
        assert_eq!(m["weight_hash"].as_str().unwrap().len(), 64);
    }
}

#[tokio::test]
async fn v1_cascade_rejects_empty_clauses() {
    let (status, body, _) =
        post_json(router_over(StubEngine::fully_configured()), "/v1/cascade", json!({"clauses": []})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["type"], "bad_request");
}

#[tokio::test]
async fn v1_cascade_without_a_router_is_400_not_500() {
    let mut engine = StubEngine::fully_configured();
    engine.router = None;
    let (status, body, _) =
        post_json(router_over(engine), "/v1/cascade", json!({"clauses": ["rm -rf ."]})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]["message"].as_str().unwrap().contains("router"));
}

#[tokio::test]
async fn v1_cascade_with_no_cascade_routes_configured_is_400() {
    let mut engine = StubEngine::fully_configured();
    engine.cascade_routes.clear();
    let (status, body, _) =
        post_json(router_over(engine), "/v1/cascade", json!({"clauses": ["rm -rf ."]})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]["message"].as_str().unwrap().contains("route"));
}

// ------------------------------------------------------------ worker fail

#[tokio::test]
async fn a_worker_failure_is_500_with_the_error_shape_never_a_silent_200() {
    let mut engine = StubEngine::fully_configured();
    engine.fail_with = Some("simulated forward-pass failure".to_string());
    let (status, body, _) = post_json(
        router_over(engine),
        "/v1/classify",
        json!({"inputs": ["hi"]}),
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(body["error"]["type"], "internal");
    assert!(body["error"]["message"].as_str().unwrap().contains("simulated"));
}

#[tokio::test]
async fn a_worker_thread_panic_is_500_for_the_inflight_request_and_500_after() {
    let engine = StubEngine {
        panic_with: Some("simulated worker thread death".to_string()),
        ..StubEngine::fully_configured()
    };
    let router = router_over(engine);

    // The request that triggers the panic: the worker unwinds mid-request,
    // dropping the oneshot sender without replying.
    let (status, body, _) =
        post_json(router.clone(), "/v1/classify", json!({"inputs": ["hi"]})).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(body["error"]["type"], "internal");
    assert!(
        body["error"]["message"].as_str().unwrap().contains("without answering"),
        "in-flight request must see the dropped-reply error, got: {body}"
    );

    // The worker thread is dead now; the command channel's receiver is gone.
    // Every later request must still be a loud 500, never a hang or a 200.
    let (status, body, _) = post_json(router, "/v1/classify", json!({"inputs": ["hi"]})).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(body["error"]["type"], "internal");
    assert!(
        body["error"]["message"].as_str().unwrap().contains("worker thread is gone"),
        "post-panic request must see the dead-worker error, got: {body}"
    );
}
