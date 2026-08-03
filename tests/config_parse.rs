//! Config parsing pinned against REAL checkpoint configs (fetched from the
//! Hub 2026-08-03, see fixtures/) — the family's checkpoints disagree with
//! each other in ways transcription from docs would have missed.

use candle_lfm2_encoder::{EncoderArch, LayerType, Lfm2EncoderConfig};

fn fixture(name: &str) -> Lfm2EncoderConfig {
    let path = format!("{}/tests/fixtures/{name}.config.json", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let cfg = Lfm2EncoderConfig::from_json(&bytes).unwrap_or_else(|e| panic!("parse {name}: {e}"));
    cfg.validate().unwrap_or_else(|e| panic!("validate {name}: {e}"));
    cfg
}

#[test]
fn embedding_checkpoint_is_a_bare_trunk() {
    let cfg = fixture("LFM2.5-Embedding-350M");
    assert_eq!(cfg.arch(), EncoderArch::BidirectionalModel);
    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.num_hidden_layers, 16);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert_eq!(cfg.vocab_size, 65536);
    assert_eq!(cfg.layer_types.len(), 16);
    assert!(cfg.num_labels().is_none(), "embedding trunk has no label head");
}

#[test]
fn encoder_base_is_masked_lm_with_14_layers() {
    // The 230M base is SHALLOWER (14 layers), not just narrower — a depth
    // assumption keyed to the 350M would corrupt a 230M load.
    let cfg = fixture("LFM2.5-Encoder-230M");
    assert_eq!(cfg.arch(), EncoderArch::MaskedLm);
    assert_eq!(cfg.num_hidden_layers, 14);
    assert_eq!(cfg.layer_types.len(), 14);
}

#[test]
fn pii_detector_has_bioes_labels_including_secrets() {
    let cfg = fixture("LFM2.5-Encoder-350M-PII-Detector");
    assert_eq!(cfg.arch(), EncoderArch::TokenClassification);
    let n = cfg.num_labels().expect("token-classification head has labels");
    assert_eq!(n, 161, "O + BIOES over the 40-type taxonomy");
    let labels = cfg.id2label.as_ref().unwrap();
    assert!(
        labels.values().any(|l| l == "B-credential.api_key"),
        "the PII taxonomy includes credentials — it doubles as a secrets detector"
    );
    // The shipped config carries a literal `"full_attn_idxs": null` — the
    // key exists, the value doesn't. Pin that it parses as None (and so
    // skips validate()'s cross-check) rather than assuming presence means
    // a value; layer_types remains the single source of layer kinds.
    assert!(cfg.full_attn_idxs.is_none());
}

#[test]
fn router_is_rule_projection_not_a_label_head() {
    let cfg = fixture("LFM2.5-Encoder-350M-Prompt-Router");
    assert_eq!(cfg.arch(), EncoderArch::SequenceRouting);
    assert!(cfg.rule_proj_dim.is_some(), "routing scores against rule projections");
    assert!(cfg.num_labels().is_none(), "no fixed label set — routing is zero-shot-shaped");
}

#[test]
fn hybrid_stack_is_mostly_conv_with_interleaved_attention() {
    let cfg = fixture("LFM2.5-Embedding-350M");
    let attn = cfg
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::FullAttention)
        .count();
    let conv = cfg.layer_types.len() - attn;
    assert!(conv > attn, "LFM2's stack is conv-heavy: {conv} conv vs {attn} attention");
}

#[test]
fn rope_theta_resolves_across_both_config_forms() {
    // Every fixture must yield a positive theta whether it came from the
    // flat key, the structured rope_parameters, or the default.
    for name in [
        "LFM2.5-Embedding-350M",
        "LFM2.5-Encoder-230M",
        "LFM2.5-Encoder-350M-PII-Detector",
        "LFM2.5-Encoder-350M-Prompt-Router",
    ] {
        let cfg = fixture(name);
        assert!(cfg.rope_theta() > 0.0, "{name}: rope_theta must resolve");
    }
}

#[test]
fn validate_catches_a_truncated_layer_list() {
    let mut cfg = fixture("LFM2.5-Embedding-350M");
    cfg.layer_types.pop();
    let err = cfg.validate().unwrap_err();
    assert!(err.contains("layer_types"), "{err}");
}
