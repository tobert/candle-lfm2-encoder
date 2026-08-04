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
use candle_lfm2_encoder::{Lfm2EncoderConfig, Lfm2Trunk};

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
