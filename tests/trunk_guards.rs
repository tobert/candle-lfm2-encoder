//! Loud-failure guards, on a synthetic tiny checkpoint.
//!
//! These need no weights, so they run anywhere and cover paths the parity
//! tests structurally cannot: the parity suite loads the Embedding
//! checkpoint, which lives at the top level, so the `lfm2.` prefix that
//! EVERY head-carrying checkpoint uses was otherwise unexercised.
//!
//! The house rule these encode: a wrong load or a wrong call must crash,
//! not return plausible numbers. Each test names the corruption it prevents.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use lfm2_encoder::{Lfm2EncoderConfig, Lfm2Trunk};

const HIDDEN: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = HIDDEN / HEADS;
const FFN: usize = 128;
const VOCAB: usize = 100;
const K: usize = 3;

/// Two layers, one of each kind — enough to exercise both operators.
fn tiny_config() -> Lfm2EncoderConfig {
    let json = format!(
        r#"{{
            "architectures": ["Lfm2BidirectionalModel"],
            "vocab_size": {VOCAB},
            "hidden_size": {HIDDEN},
            "num_hidden_layers": 2,
            "num_attention_heads": {HEADS},
            "num_key_value_heads": {KV_HEADS},
            "layer_types": ["conv", "full_attention"],
            "max_position_embeddings": 512,
            "intermediate_size": {FFN},
            "block_multiple_of": 1,
            "conv_L_cache": {K},
            "conv_bias": false,
            "use_pos_enc": true,
            "rope_theta": 1000000.0
        }}"#
    );
    let cfg = Lfm2EncoderConfig::from_json(json.as_bytes()).expect("parse tiny config");
    assert_eq!(cfg.ffn_dim(), FFN);
    cfg
}

fn zeros(dims: &[usize], map: &mut HashMap<String, Tensor>, name: String) {
    map.insert(
        name,
        Tensor::zeros(dims, DType::F32, &Device::Cpu).unwrap(),
    );
}

/// Every tensor a trunk needs, under `prefix` (e.g. `""` or `"lfm2."`).
fn tiny_weights(prefix: &str) -> HashMap<String, Tensor> {
    let mut m = HashMap::new();
    let p = prefix;
    zeros(&[VOCAB, HIDDEN], &mut m, format!("{p}embed_tokens.weight"));
    zeros(&[HIDDEN], &mut m, format!("{p}embedding_norm.weight"));
    for i in 0..2 {
        let l = format!("{p}layers.{i}");
        zeros(&[HIDDEN], &mut m, format!("{l}.operator_norm.weight"));
        zeros(&[HIDDEN], &mut m, format!("{l}.ffn_norm.weight"));
        zeros(&[FFN, HIDDEN], &mut m, format!("{l}.feed_forward.w1.weight"));
        zeros(&[HIDDEN, FFN], &mut m, format!("{l}.feed_forward.w2.weight"));
        zeros(&[FFN, HIDDEN], &mut m, format!("{l}.feed_forward.w3.weight"));
        if i == 0 {
            zeros(&[HIDDEN, 1, K], &mut m, format!("{l}.conv.conv.weight"));
            zeros(&[3 * HIDDEN, HIDDEN], &mut m, format!("{l}.conv.in_proj.weight"));
            zeros(&[HIDDEN, HIDDEN], &mut m, format!("{l}.conv.out_proj.weight"));
        } else {
            let a = format!("{l}.self_attn");
            zeros(&[HEADS * HEAD_DIM, HIDDEN], &mut m, format!("{a}.q_proj.weight"));
            zeros(&[KV_HEADS * HEAD_DIM, HIDDEN], &mut m, format!("{a}.k_proj.weight"));
            zeros(&[KV_HEADS * HEAD_DIM, HIDDEN], &mut m, format!("{a}.v_proj.weight"));
            zeros(&[HIDDEN, HEADS * HEAD_DIM], &mut m, format!("{a}.out_proj.weight"));
            zeros(&[HEAD_DIM], &mut m, format!("{a}.q_layernorm.weight"));
            zeros(&[HEAD_DIM], &mut m, format!("{a}.k_layernorm.weight"));
        }
    }
    m
}

fn vb(map: HashMap<String, Tensor>) -> VarBuilder<'static> {
    VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
}

/// The prefix every head-carrying checkpoint (PII, Router, Policy-Linter)
/// stores its trunk under. The parity tests can't reach this path.
#[test]
fn autodetects_the_lfm2_prefix_used_by_head_carrying_checkpoints() {
    let cfg = tiny_config();
    let trunk = Lfm2Trunk::load(&cfg, vb(tiny_weights("lfm2."))).expect("load under lfm2.");

    let ids = Tensor::new(&[[1u32, 2, 3, 4]], &Device::Cpu).unwrap();
    let out = trunk.forward(&ids, None).expect("forward");
    assert_eq!(out.dims3().unwrap(), (1, 4, HIDDEN));
}

#[test]
fn autodetects_the_bare_prefix_used_by_the_embedding_checkpoint() {
    let cfg = tiny_config();
    let trunk = Lfm2Trunk::load(&cfg, vb(tiny_weights(""))).expect("load at top level");
    let ids = Tensor::new(&[[1u32, 2, 3]], &Device::Cpu).unwrap();
    assert_eq!(trunk.forward(&ids, None).unwrap().dims3().unwrap().1, 3);
}

/// A merged/multi-task checkpoint carrying the trunk in two places must
/// not be resolved by probe order — that would load a real trunk from the
/// wrong place and produce entirely plausible garbage.
#[test]
fn ambiguous_trunk_prefix_is_refused_not_guessed() {
    let cfg = tiny_config();
    let mut map = tiny_weights("");
    map.extend(tiny_weights("lfm2."));

    let err = Lfm2Trunk::load(&cfg, vb(map)).expect_err("two trunks must not silently resolve");
    let msg = err.to_string();
    assert!(msg.contains("ambiguous"), "{msg}");
    assert!(msg.contains("load_with_prefix"), "error should say how to disambiguate: {msg}");
}

#[test]
fn a_checkpoint_with_no_trunk_names_what_it_looked_for() {
    let cfg = tiny_config();
    let err = Lfm2Trunk::load(&cfg, vb(HashMap::new())).expect_err("empty checkpoint must fail");
    assert!(err.to_string().contains("embed_tokens.weight"), "{err}");
}

/// The corruption this prevents: a padded batch with no mask attends to
/// pad tokens as content AND runs them through the unmasked conv, so every
/// row comes back subtly wrong with no error anywhere.
#[test]
fn batched_forward_without_a_mask_is_refused() {
    let cfg = tiny_config();
    let trunk = Lfm2Trunk::load(&cfg, vb(tiny_weights(""))).unwrap();

    let batch = Tensor::new(&[[1u32, 2, 3, 4], [5, 6, 0, 0]], &Device::Cpu).unwrap();
    let err = trunk
        .forward(&batch, None)
        .expect_err("batch>1 without a mask must fail");
    assert!(err.to_string().contains("attention_mask"), "{err}");

    // Same batch WITH a mask is fine, and a single row without one is the
    // documented reproducible path.
    let mask = Tensor::new(&[[1u32, 1, 1, 1], [1, 1, 0, 0]], &Device::Cpu).unwrap();
    trunk.forward(&batch, Some(&mask)).expect("masked batch is fine");
    let single = Tensor::new(&[[1u32, 2, 3, 4]], &Device::Cpu).unwrap();
    trunk.forward(&single, None).expect("single row needs no mask");
}

/// A soft/fractional mask is not something these checkpoints were trained
/// with; accepting one would silently attenuate keys instead of masking
/// them. The contract is binary, and enforced.
#[test]
fn a_non_binary_attention_mask_is_refused() {
    let cfg = tiny_config();
    let trunk = Lfm2Trunk::load(&cfg, vb(tiny_weights(""))).unwrap();
    let ids = Tensor::new(&[[1u32, 2, 3, 4]], &Device::Cpu).unwrap();

    let soft = Tensor::new(&[[1.0f32, 0.5, 1.0, 0.0]], &Device::Cpu).unwrap();
    let err = trunk
        .forward(&ids, Some(&soft))
        .expect_err("a fractional mask must be refused");
    assert!(err.to_string().contains("1 (real) or 0 (pad)"), "{err}");

    // The same mask with binary values is accepted.
    let hard = Tensor::new(&[[1.0f32, 1.0, 1.0, 0.0]], &Device::Cpu).unwrap();
    trunk.forward(&ids, Some(&hard)).expect("binary f32 mask is fine");
}

/// `conv_bias` defaults to false when the key is absent, so a checkpoint
/// that ships a bias would have it silently dropped — a trained parameter
/// vanishing with no error.
#[test]
fn a_conv_bias_the_config_denies_is_refused() {
    let cfg = tiny_config();
    assert!(!cfg.conv_bias, "fixture premise: config says no conv bias");

    let mut map = tiny_weights("");
    zeros(&[HIDDEN], &mut map, "layers.0.conv.conv.bias".to_string());

    let err = Lfm2Trunk::load(&cfg, vb(map)).expect_err("undeclared conv bias must fail");
    let msg = err.to_string();
    assert!(msg.contains("conv_bias=false"), "{msg}");
}

/// A config JSON with the shape-defining fields as parameters — used by
/// [`check_compatible`]'s guards below to build a second config that
/// disagrees with [`tiny_config`] on exactly one field at a time.
/// `check_compatible` never calls `validate()` itself, so this deliberately
/// skips it too (a mismatched `layer_types` length is exactly one of the
/// things being tested).
fn other_config(hidden: usize, vocab: usize, layers: usize, heads: usize, kv_heads: usize, layer_types: &str) -> Lfm2EncoderConfig {
    let json = format!(
        r#"{{
            "architectures": ["Lfm2BidirectionalModel"],
            "vocab_size": {vocab},
            "hidden_size": {hidden},
            "num_hidden_layers": {layers},
            "num_attention_heads": {heads},
            "num_key_value_heads": {kv_heads},
            "layer_types": {layer_types},
            "max_position_embeddings": 512,
            "intermediate_size": {FFN},
            "block_multiple_of": 1,
            "conv_L_cache": {K},
            "conv_bias": false,
            "use_pos_enc": true,
            "rope_theta": 1000000.0
        }}"#
    );
    Lfm2EncoderConfig::from_json(json.as_bytes()).expect("parse other config")
}

/// A checkpoint config identical in shape to `tiny_config()` must be
/// declared compatible — this is the happy path `from_trunk` takes on every
/// well-formed head checkpoint, and the guards below are only meaningful in
/// contrast to it.
#[test]
fn check_compatible_accepts_a_config_with_the_same_shape() {
    let cfg = tiny_config();
    let trunk = Lfm2Trunk::load(&cfg, vb(tiny_weights("lfm2."))).expect("load trunk");

    let same = other_config(HIDDEN, VOCAB, 2, HEADS, KV_HEADS, r#"["conv", "full_attention"]"#);
    trunk.check_compatible(&same).expect("identical shape must be declared compatible");
}

/// The corruption this prevents: pairing a shared trunk with a head
/// checkpoint trained for a DIFFERENT model shape. Without this check, the
/// mismatch either panics three matmuls deep with a bare shape error, or —
/// if dimensions happen to coincide in a way that still multiplies — runs
/// to completion and hands back a plausible-looking, silently wrong number.
/// Every disagreeing field must be named, not just the first found.
#[test]
fn check_compatible_names_every_mismatched_field() {
    let cfg = tiny_config();
    let trunk = Lfm2Trunk::load(&cfg, vb(tiny_weights("lfm2."))).expect("load trunk");

    // Disagrees on hidden_size AND vocab_size, but keeps everything else
    // matching tiny_config().
    let mismatched = other_config(HIDDEN * 2, VOCAB + 1, 2, HEADS, KV_HEADS, r#"["conv", "full_attention"]"#);

    let err = trunk
        .check_compatible(&mismatched)
        .expect_err("a differently-shaped head checkpoint must be refused");
    assert!(err.contains("hidden_size"), "{err}");
    assert!(err.contains(&HIDDEN.to_string()), "{err}");
    assert!(err.contains(&(HIDDEN * 2).to_string()), "{err}");
    assert!(err.contains("vocab_size"), "{err}");
    // A field that DOES agree (num_hidden_layers) must not be named.
    assert!(!err.contains("num_hidden_layers"), "{err}");
}

/// Layer count/kind disagreement is its own, separately-testable case: two
/// configs can agree on every scalar field and still describe an
/// incompatible model if the hybrid conv/attention layout differs.
#[test]
fn check_compatible_catches_a_layer_type_layout_disagreement() {
    let cfg = tiny_config();
    let trunk = Lfm2Trunk::load(&cfg, vb(tiny_weights("lfm2."))).expect("load trunk");

    // Same layer COUNT, but attention/conv swapped.
    let swapped = other_config(HIDDEN, VOCAB, 2, HEADS, KV_HEADS, r#"["full_attention", "conv"]"#);
    let err = trunk
        .check_compatible(&swapped)
        .expect_err("a swapped conv/attention layout must be refused");
    assert!(err.contains("layer_types"), "{err}");

    // Different layer COUNT.
    let deeper = other_config(HIDDEN, VOCAB, 3, HEADS, KV_HEADS, r#"["conv", "conv", "full_attention"]"#);
    let err = trunk
        .check_compatible(&deeper)
        .expect_err("a different layer count must be refused");
    assert!(err.contains("num_hidden_layers"), "{err}");
}

/// [`Lfm2Trunk::load_shared`] is the directory-based entry point
/// `from_trunk` callers use; it must round-trip a real checkpoint directory
/// (config.json + model.safetensors) the same way every head's `from_dir`
/// does, and the trunk it hands back must be an `Arc` sized for sharing.
#[test]
fn load_shared_round_trips_a_directory_checkpoint() {
    let _ = tiny_config(); // sanity: this is the shape config.json below encodes
    let dir = tempfile::tempdir().expect("tempdir");
    // `Lfm2EncoderConfig` has no `Serialize` derive, so write the same JSON
    // literal `tiny_config()` itself parses, by hand.
    std::fs::write(
        dir.path().join("config.json"),
        r#"{
            "architectures": ["Lfm2BidirectionalModel"],
            "vocab_size": 100,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "layer_types": ["conv", "full_attention"],
            "max_position_embeddings": 512,
            "intermediate_size": 128,
            "block_multiple_of": 1,
            "conv_L_cache": 3,
            "conv_bias": false,
            "use_pos_enc": true,
            "rope_theta": 1000000.0
        }"#,
    )
    .expect("write config.json");
    candle_core::safetensors::save(&tiny_weights("lfm2."), dir.path().join("model.safetensors"))
        .expect("write model.safetensors");

    let trunk = Lfm2Trunk::load_shared(dir.path()).expect("load_shared");
    assert_eq!(std::sync::Arc::strong_count(&trunk), 1);

    let ids = Tensor::new(&[[1u32, 2, 3, 4]], &Device::Cpu).unwrap();
    let out = trunk.forward(&ids, None).expect("forward");
    assert_eq!(out.dims3().unwrap(), (1, 4, HIDDEN));
}
