//! PII token-classification parity against the checkpoint's own decode.
//!
//! Reference: `tests/reference/dump_pii_reference.py`, which calls
//! `pii_hybrid_decode.model_spans` — the PURE MODEL decode. The shipped
//! hybrid wraps this head in a regex tier that owns every credential type
//! outright, so comparing against hybrid output would be testing regexes
//! rather than the model.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{Device, Tensor};
use candle_lfm2_encoder::Lfm2TokenClassifier;
use serde::Deserialize;

const MODEL: &str = "LFM2.5-Encoder-350M-PII-Detector";

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    n: usize,
    id: String,
    category: String,
    text: String,
    model_spans: Vec<RefSpan>,
}

#[derive(Deserialize, Debug, PartialEq)]
struct RefSpan {
    start: usize,
    end: usize,
    #[serde(rename = "type")]
    label: String,
}

fn checkpoint() -> PathBuf {
    let base = match std::env::var("LFM2_MODELS_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".models"),
    };
    let dir = base.join(MODEL);
    assert!(
        dir.join("model.safetensors").is_file(),
        "missing weights at {}\n\n  hf download LiquidAI/{MODEL} --local-dir {}\n",
        dir.display(),
        dir.display(),
    );
    dir
}

fn reference() -> Reference {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pii_reference_spans.json");
    serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
        .expect("parse span reference")
}

fn tensors() -> HashMap<String, Tensor> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pii_reference.safetensors");
    candle_core::safetensors::load(&path, &Device::Cpu)
        .unwrap_or_else(|e| panic!("load {}: {e}", path.display()))
}

fn model() -> &'static Lfm2TokenClassifier {
    static M: std::sync::OnceLock<Lfm2TokenClassifier> = std::sync::OnceLock::new();
    M.get_or_init(|| Lfm2TokenClassifier::from_dir(checkpoint()).expect("load PII detector"))
}

/// The label set must come from config.json (161 labels / 40 types), NOT
/// from the checkpoint's stale `label_schema.json` (109 / 27). What that
/// file omits is exactly what a guard tool reads this model for.
#[test]
fn label_set_is_the_full_161_not_the_stale_schema_file() {
    let m = model();
    assert_eq!(m.num_labels(), 161, "classifier width must match config.json");

    let types = m.entity_types();
    assert_eq!(types.len(), 40, "BIOES over 40 types: 4*40 + 1 = 161");
    for required in [
        "credential.api_key",
        "credential.connection_string",
        "credential.jwt",
        "credential.password",
        "credential.private_key",
    ] {
        assert!(
            types.contains(&required),
            "{required} missing — label_schema.json (109 labels) drops every one of \
             these, and decoding through it would mislabel silently"
        );
    }
}

/// Per-token argmax, before any span logic. A divergence here is the model;
/// a divergence only in spans is the decode.
#[test]
fn per_token_predictions_match_the_reference() {
    let refs = reference();
    let t = tensors();
    let m = model();

    for case in &refs.cases {
        let want: Vec<u32> = t[&format!("case.{}.argmax", case.n)]
            .to_dtype(candle_core::DType::U32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let got = m.token_label_ids(&case.text).expect("token labels");
        assert_eq!(
            got.len(),
            want.len(),
            "case {} ({}): token count differs",
            case.n,
            case.id
        );
        let diffs = got
            .iter()
            .zip(&want)
            .filter(|(a, b)| **a as u32 != **b)
            .count();
        assert_eq!(diffs, 0, "case {} ({}): {diffs} token labels differ", case.n, case.id);
    }
}

#[test]
fn decoded_spans_match_the_reference_exactly() {
    let refs = reference();
    let m = model();

    for case in &refs.cases {
        let got = m.predict(&case.text).expect("predict");
        let got_simple: Vec<RefSpan> = got
            .iter()
            .map(|s| RefSpan {
                start: s.start,
                end: s.end,
                label: s.label.clone(),
            })
            .collect();
        assert_eq!(
            got_simple, case.model_spans,
            "case {} ({}, {}) spans differ.\n  text: {:?}\n  ours: {:?}",
            case.n,
            case.id,
            case.category,
            case.text,
            got.iter().map(|s| (s.text(&case.text), &s.label)).collect::<Vec<_>>()
        );
    }
}

/// Offsets must index the ORIGINAL string correctly, which is where a
/// byte-vs-char mismatch would surface — the reference includes non-ASCII
/// cases for exactly this reason.
#[test]
fn span_offsets_are_valid_utf8_boundaries_into_the_source() {
    let refs = reference();
    let m = model();

    for case in &refs.cases {
        for span in m.predict(&case.text).expect("predict") {
            assert!(
                case.text.is_char_boundary(span.start) && case.text.is_char_boundary(span.end),
                "case {}: span {}..{} is not on char boundaries of {:?}",
                case.n,
                span.start,
                span.end,
                case.text
            );
            // Must not panic, and must be non-empty.
            assert!(!span.text(&case.text).is_empty());
        }
    }
}

/// The guard question kaibo actually asks. Clean text must stay clean —
/// a boundary guard that cries wolf gets switched off.
#[test]
fn clean_text_produces_no_spans() {
    let refs = reference();
    let m = model();

    for case in refs.cases.iter().filter(|c| c.category == "clean") {
        let spans = m.predict(&case.text).expect("predict");
        assert!(
            spans.is_empty(),
            "clean case {} fired: {:?} on {:?}",
            case.id,
            spans.iter().map(|s| (s.text(&case.text), &s.label)).collect::<Vec<_>>(),
            case.text
        );
    }
}
