//! Wire-contract tests: every request/response type round-trips through
//! JSON, and the exact field names/shapes the API doc promises are pinned
//! with literal-string assertions — not just "it deserializes something".
//! A field rename or an accidental extra wrapper here is a silent contract
//! break for kaish/kaijutsu callers; these tests make it a compile-time-red
//! `cargo test`, not a discovery made in production.
//!
//! Maps that get string-compared (`scores`, `severity_scores`) are typed
//! `BTreeMap`, not `HashMap` — `HashMap`'s iteration order is randomized
//! per-process, which would make an exact-string assertion flaky by
//! construction.

use std::collections::BTreeMap;

use lfm2d::types::{
    ApiError, CascadeClause, CascadeLane, CascadeModelRef, CascadeRequest, CascadeResponse,
    CascadeWinner, ClassifyRequest, ClassifyResult, EmbedRequest, Inputs, LabelScore, ModelInfo,
    ModelKind, PredictRequest, RouteRequest, RouteResponse, RouteScore,
};

fn assert_json_eq<T: serde::Serialize>(value: &T, expected: &str) {
    let got: serde_json::Value = serde_json::to_value(value).expect("serialize");
    let want: serde_json::Value = serde_json::from_str(expected).expect("expected is valid JSON");
    assert!(
        json_approx_eq(&got, &want),
        "\n  got: {got}\n want: {want}"
    );
}

/// Structural equality, except `Number`s compare within a small epsilon —
/// every score in this API is an `f32`, and `serde_json::Value` always
/// stores numbers as `f64`, so `0.71f32` and the literal `0.71` parsed as
/// `f64` are two DIFFERENT f64 bit patterns even though no test here cares
/// about the difference. An exact `Value::eq` would make every float-typed
/// contract test flaky-by-construction on the least significant bits, not
/// on anything this test suite is actually pinning (field names/shape).
/// Pull an `f32` back out of a `serde_json::Value::Number` for a tolerant
/// comparison against a literal — see [`json_approx_eq`] for why exact
/// comparison isn't the right tool for an f32-through-f64 round trip.
fn approx(v: &serde_json::Value) -> f64 {
    v.as_f64().expect("expected a JSON number")
}

fn json_approx_eq(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    use serde_json::Value;
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => (x.as_f64().unwrap() - y.as_f64().unwrap()).abs() < 1e-5,
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| json_approx_eq(a, b))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| json_approx_eq(v, w)))
        }
        _ => a == b,
    }
}

// ---------------------------------------------------------------- /embed

#[test]
fn embed_request_accepts_a_single_string() {
    let req: EmbedRequest = serde_json::from_str(r#"{"inputs": "hello"}"#).expect("parse");
    assert_eq!(req.inputs.into_vec(), vec!["hello".to_string()]);
}

#[test]
fn embed_request_accepts_an_array() {
    let req: EmbedRequest = serde_json::from_str(r#"{"inputs": ["a", "b"]}"#).expect("parse");
    assert_eq!(req.inputs.into_vec(), vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn inputs_single_and_many_both_normalize_through_into_vec() {
    let one: Inputs = serde_json::from_str(r#""solo""#).expect("parse");
    assert_eq!(one.into_vec(), vec!["solo".to_string()]);
    let many: Inputs = serde_json::from_str(r#"["x", "y", "z"]"#).expect("parse");
    assert_eq!(many.into_vec(), vec!["x".to_string(), "y".to_string(), "z".to_string()]);
}

// --------------------------------------------------------------- /predict

#[test]
fn predict_request_round_trips() {
    let req: PredictRequest = serde_json::from_str(r#"{"inputs": ["kubectl delete ns prod"]}"#)
        .expect("parse");
    assert_eq!(req.inputs.into_vec(), vec!["kubectl delete ns prod".to_string()]);
}

#[test]
fn label_score_serializes_with_exact_field_names() {
    let ls = LabelScore { label: "destructive".to_string(), score: 0.71 };
    assert_json_eq(&ls, r#"{"label": "destructive", "score": 0.71}"#);
}

#[test]
fn predict_response_shape_is_a_list_of_label_lists() {
    // The wire response has NO wrapper object — TEI-ish: a bare
    // Vec<Vec<LabelScore>>, one inner list per input, covering ALL labels.
    let resp: Vec<Vec<LabelScore>> = vec![vec![
        LabelScore { label: "mutating".to_string(), score: 0.6 },
        LabelScore { label: "informative".to_string(), score: 0.4 },
    ]];
    assert_json_eq(
        &resp,
        r#"[[{"label": "mutating", "score": 0.6}, {"label": "informative", "score": 0.4}]]"#,
    );
}

// ------------------------------------------------------------- /v1/models

#[test]
fn model_kind_serializes_as_lowercase_snake_case() {
    assert_json_eq(&ModelKind::Embedder, r#""embedder""#);
    assert_json_eq(&ModelKind::Classifier, r#""classifier""#);
    assert_json_eq(&ModelKind::Router, r#""router""#);
}

#[test]
fn model_info_for_an_embedder_omits_labels() {
    let info = ModelInfo {
        id: "LFM2.5-Embedding-350M".to_string(),
        kind: ModelKind::Embedder,
        weight_hash: "a".repeat(64),
        labels: None,
        hidden_size: 1024,
    };
    assert_json_eq(
        &info,
        &format!(
            r#"{{"id": "LFM2.5-Embedding-350M", "kind": "embedder", "weight_hash": "{}", "hidden_size": 1024}}"#,
            "a".repeat(64)
        ),
    );
}

#[test]
fn model_info_for_a_classifier_carries_labels() {
    let info = ModelInfo {
        id: "kube_ordinal_v6".to_string(),
        kind: ModelKind::Classifier,
        weight_hash: "b".repeat(64),
        labels: Some(vec!["destructive".into(), "informative".into(), "mutating".into()]),
        hidden_size: 1024,
    };
    let v = serde_json::to_value(&info).unwrap();
    assert_eq!(v["labels"], serde_json::json!(["destructive", "informative", "mutating"]));
}

// ------------------------------------------------------------ /v1/classify

#[test]
fn classify_request_takes_an_inputs_array() {
    let req: ClassifyRequest = serde_json::from_str(r#"{"inputs": ["rm -rf /"]}"#).expect("parse");
    assert_eq!(req.inputs, vec!["rm -rf /".to_string()]);
}

#[test]
fn classify_result_pins_scores_top_model_id_and_weight_hash() {
    let mut scores = BTreeMap::new();
    scores.insert("destructive".to_string(), 0.71f32);
    scores.insert("informative".to_string(), 0.05f32);
    scores.insert("mutating".to_string(), 0.24f32);
    let result = ClassifyResult {
        scores,
        top: "destructive".to_string(),
        model_id: "kube_ordinal_v6".to_string(),
        weight_hash: "c".repeat(64),
    };
    assert_json_eq(
        &result,
        &format!(
            r#"{{
                "scores": {{"destructive": 0.71, "informative": 0.05, "mutating": 0.24}},
                "top": "destructive",
                "model_id": "kube_ordinal_v6",
                "weight_hash": "{}"
            }}"#,
            "c".repeat(64)
        ),
    );
}

// --------------------------------------------------------------- /v1/route

#[test]
fn route_request_uses_singular_input_not_inputs() {
    let req: RouteRequest =
        serde_json::from_str(r#"{"input": "kubectl delete ns prod", "routes": ["k8s", "shell"]}"#)
            .expect("parse");
    assert_eq!(req.input, "kubectl delete ns prod");
    assert_eq!(req.routes, vec!["k8s".to_string(), "shell".to_string()]);
}

#[test]
fn route_response_carries_model_id_weight_hash_and_raw_cosines_only() {
    let resp = RouteResponse {
        model_id: "LFM2.5-Encoder-350M-Prompt-Router".to_string(),
        weight_hash: "d".repeat(64),
        routes: vec![
            RouteScore { route: "k8s".to_string(), cosine: 0.978 },
            RouteScore { route: "shell".to_string(), cosine: -0.2 },
        ],
    };
    let v = serde_json::to_value(&resp).unwrap();
    // RAW COSINE ONLY — no "score"/"prob"/softmax field anywhere.
    assert!(v["routes"][0].get("score").is_none());
    assert!(v["routes"][0].get("prob").is_none());
    assert_eq!(v["routes"][0]["route"], "k8s");
    assert!((approx(&v["routes"][0]["cosine"]) - 0.978).abs() < 1e-5);
    assert_eq!(v["model_id"], "LFM2.5-Encoder-350M-Prompt-Router");
    assert_eq!(v["weight_hash"], "d".repeat(64));
}

// ------------------------------------------------------------- /v1/cascade

#[test]
fn cascade_request_takes_only_clauses() {
    let req: CascadeRequest =
        serde_json::from_str(r#"{"clauses": ["cd vendor/legacy-plugin", "rm -rf ."]}"#)
            .expect("parse");
    assert_eq!(req.clauses, vec!["cd vendor/legacy-plugin".to_string(), "rm -rf .".to_string()]);
}

#[test]
fn cascade_response_matches_the_moderations_flavored_contract_exactly() {
    let mut severity = BTreeMap::new();
    severity.insert("destructive".to_string(), 0.9f32);
    severity.insert("informative".to_string(), 0.02f32);
    severity.insert("mutating".to_string(), 0.08f32);

    let resp = CascadeResponse {
        winner: CascadeWinner {
            index: 1,
            clause: "rm -rf .".to_string(),
            severity_scores: severity.clone(),
        },
        lane: CascadeLane { route: "shell".to_string(), cosine: 0.91 },
        clauses: vec![
            CascadeClause {
                index: 0,
                clause: "cd vendor/legacy-plugin".to_string(),
                severity_scores: {
                    let mut m = BTreeMap::new();
                    m.insert("destructive".to_string(), 0.01f32);
                    m.insert("informative".to_string(), 0.95f32);
                    m.insert("mutating".to_string(), 0.04f32);
                    m
                },
                top_severity: "informative".to_string(),
            },
            CascadeClause {
                index: 1,
                clause: "rm -rf .".to_string(),
                severity_scores: severity,
                top_severity: "destructive".to_string(),
            },
        ],
        models: vec![
            CascadeModelRef {
                model_id: "kube_ordinal_v6".to_string(),
                weight_hash: "e".repeat(64),
            },
            CascadeModelRef {
                model_id: "LFM2.5-Encoder-350M-Prompt-Router".to_string(),
                weight_hash: "f".repeat(64),
            },
        ],
    };

    let v = serde_json::to_value(&resp).unwrap();
    // No "flagged" boolean, no threshold-shaped field, anywhere.
    assert!(v.get("flagged").is_none());
    assert!(v.get("threshold").is_none());
    for clause in v["clauses"].as_array().unwrap() {
        assert!(clause.get("flagged").is_none());
    }

    assert_eq!(v["winner"]["index"], 1);
    assert_eq!(v["winner"]["clause"], "rm -rf .");
    assert!((approx(&v["winner"]["severity_scores"]["destructive"]) - 0.9).abs() < 1e-5);
    // winner has no "top_severity" field — only per-clause entries do.
    assert!(v["winner"].get("top_severity").is_none());

    assert_eq!(v["lane"]["route"], "shell");
    assert!((approx(&v["lane"]["cosine"]) - 0.91).abs() < 1e-5);

    assert_eq!(v["clauses"][0]["top_severity"], "informative");
    assert_eq!(v["clauses"][1]["top_severity"], "destructive");

    assert_eq!(v["models"][0]["model_id"], "kube_ordinal_v6");
    assert_eq!(v["models"][1]["model_id"], "LFM2.5-Encoder-350M-Prompt-Router");
}

// ----------------------------------------------------------------- errors

#[test]
fn api_error_shape_pins_message_and_type_field_names() {
    let err = ApiError::bad_request("inputs must not be empty");
    assert_json_eq(
        &err,
        r#"{"error": {"message": "inputs must not be empty", "type": "bad_request"}}"#,
    );
}

#[test]
fn api_error_worker_failure_uses_the_internal_type() {
    let err = ApiError::internal("classifier forward pass failed: shape mismatch");
    let v = serde_json::to_value(&err).unwrap();
    assert_eq!(v["error"]["type"], "internal");
}
