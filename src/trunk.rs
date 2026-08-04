//! The bidirectional hybrid trunk: embeddings → 14/16 interleaved
//! short-conv and full-attention blocks → final RMS norm.
//!
//! Adapted from candle-transformers' causal `models/lfm2.rs` (Copyright
//! The candle Authors, dual MIT/Apache-2.0 — the same license this crate
//! carries), with the encoder-branch differences taken from the
//! checkpoints' own `modeling_lfm2_bidirectional.py` rather than inferred:
//!
//! * **No causal mask.** Attention is full bidirectional; the only mask is
//!   an additive pad mask (`-1e9` on padded *keys*).
//! * **No causal short-conv.** The conv is centered: `padding = k / 2`
//!   symmetric, where the causal model left-pads `k - 1`. Same weights,
//!   different receptive field — this is the single change most likely to
//!   produce plausible-but-wrong embeddings if missed.
//! * **No KV cache and no conv state cache.** An encoder sees the whole
//!   sequence at once; there is no incremental decode to cache for.
//!
//! Everything else — SwiGLU FFN with `w1`/`w2`/`w3`, per-head RMS norm on
//! Q and K before RoPE, `operator_norm`/`ffn_norm` pre-norm residuals — is
//! shared with the causal branch and adapted as-is.

use candle_core::{DType, IndexOp, Module, Result, Tensor};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder};

use crate::config::{LayerType, Lfm2EncoderConfig};

/// Weight-name prefixes the family ships the trunk under, in probe order.
///
/// The bare Embedding checkpoint stores the trunk at the top level
/// (`embed_tokens.weight`); every head-carrying checkpoint nests it under
/// `lfm2.`. `model.` is what upstream causal LFM2 uses — accepted so a
/// causal checkpoint loads rather than mystifying the caller.
const TRUNK_PREFIXES: &[&str] = &["", "lfm2.", "model."];

/// A pad key's additive mask value. Matches the checkpoints' own
/// `_bidirectional_mask`, which uses -1e9 rather than -inf (an -inf row
/// would NaN out a fully-padded sequence in softmax; -1e9 degrades to a
/// uniform row instead).
const MASK_NEG: f64 = -1e9;

/// GQA head expansion: repeat each KV head `n_rep` times to line up with
/// the query heads.
///
/// The cat-on-dim-2-then-reshape is deliberate and load-bearing: catting
/// along the sequence axis lays each KV head's `n_rep` copies down
/// contiguously, so the reshape reinterprets them as `[kv0, kv0, kv1,
/// kv1, …]` — the repeat_interleave order the query heads expect. Catting
/// along the head axis instead would give `[kv0, kv1, kv0, kv1, …]`, which
/// is a plausible-looking pairing of the wrong heads.
fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs);
    }
    // Guard rather than assume: callers currently hand us contiguous
    // tensors, but this is cheap (candle short-circuits when already
    // contiguous) and the failure mode if that ever stops being true is
    // silently scrambled heads, not an error.
    let xs = xs.contiguous()?;
    let (b, n_kv, seq, dim) = xs.dims4()?;
    Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b, n_kv * n_rep, seq, dim))
}

/// SwiGLU feed-forward. LFM2 names the projections `w1` (gate), `w3` (up),
/// `w2` (down).
#[derive(Debug)]
struct FeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl FeedForward {
    fn load(cfg: &Lfm2EncoderConfig, vb: VarBuilder) -> Result<Self> {
        // ffn_dim(), NOT cfg.intermediate_size — see its docs; the config
        // key overstates the width on three of four checkpoints.
        let ffn = cfg.ffn_dim();
        let h = cfg.hidden_size;
        Ok(Self {
            w1: candle_nn::linear_no_bias(h, ffn, vb.pp("w1"))?,
            w2: candle_nn::linear_no_bias(ffn, h, vb.pp("w2"))?,
            w3: candle_nn::linear_no_bias(h, ffn, vb.pp("w3"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.w1.forward(xs)?)?;
        let up = self.w3.forward(xs)?;
        self.w2.forward(&(gate * up)?)
    }
}

/// Bidirectional attention with per-head QK norm and RoPE.
#[derive(Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    q_layernorm: RmsNorm,
    k_layernorm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(cfg: &Lfm2EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let (nh, nkv) = (cfg.num_attention_heads, cfg.num_key_value_heads);
        Ok(Self {
            q_proj: candle_nn::linear_no_bias(h, nh * head_dim, vb.pp("q_proj"))?,
            k_proj: candle_nn::linear_no_bias(h, nkv * head_dim, vb.pp("k_proj"))?,
            v_proj: candle_nn::linear_no_bias(h, nkv * head_dim, vb.pp("v_proj"))?,
            // Note the name: `out_proj`, not the `o_proj` most Llama-shaped
            // models use.
            out_proj: candle_nn::linear_no_bias(nh * head_dim, h, vb.pp("out_proj"))?,
            // Per-HEAD norm — these are [head_dim] vectors, not [hidden].
            q_layernorm: candle_nn::rms_norm(head_dim, cfg.norm_eps, vb.pp("q_layernorm"))?,
            k_layernorm: candle_nn::rms_norm(head_dim, cfg.norm_eps, vb.pp("k_layernorm"))?,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim,
        })
    }

    /// `rope` (not `rope_i`) — the rotate-half/NeoX layout HF's
    /// `apply_rotary_pos_emb` uses. The interleaved variant would also run
    /// and silently produce wrong numbers.
    fn apply_rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        candle_nn::rotary_emb::rope(&xs.contiguous()?, cos, sin)
    }

    fn forward(&self, xs: &Tensor, rope: Option<&(Tensor, Tensor)>, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq, _) = xs.dims3()?;

        let q = self
            .q_proj
            .forward(xs)?
            .reshape((b, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // QK norm first, then RoPE — the order HF's Lfm2Attention uses.
        let q = self.q_layernorm.forward(&q.contiguous()?)?;
        let k = self.k_layernorm.forward(&k.contiguous()?)?;
        let (q, k) = match rope {
            Some((cos, sin)) => (Self::apply_rope(&q, cos, sin)?, Self::apply_rope(&k, cos, sin)?),
            None => (q, k),
        };

        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(k, n_rep)?;
        let v = repeat_kv(v, n_rep)?;

        // Scores in f32 regardless of weight dtype: a bf16 softmax over a
        // 512-token row loses more than the matmul saves.
        let in_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        // The ONLY mask. No causal term: every token sees every token.
        let att = match mask {
            Some(m) => att.broadcast_add(m)?,
            None => att,
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        // `v` is already contiguous here: to_dtype materializes a fresh
        // buffer, as does repeat_kv's cat before it.
        let out = att.matmul(&v)?.to_dtype(in_dtype)?;

        let out = out
            .transpose(1, 2)?
            .reshape((b, seq, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{Conv1d, Conv1dConfig};

    /// The shifted multiply-add form of the short conv must equal candle's
    /// grouped `Conv1d` exactly — that equivalence is the whole license for
    /// replacing it (candle's grouped path costs a flat ~173 ms/call; see
    /// `examples/bench_ops.rs`). Covers even `k` too, which no shipped
    /// checkpoint exercises: every one is k=3.
    #[test]
    fn depthwise_conv_equals_candle_grouped_conv1d() {
        let dev = Device::Cpu;
        let hidden = 24usize;

        for k in [1usize, 2, 3, 4, 5] {
            for seq in [1usize, 3, 7, 16] {
                let weight = Tensor::randn(0f32, 1., (hidden, 1, k), &dev).unwrap();
                let bx = Tensor::randn(0f32, 1., (1, hidden, seq), &dev).unwrap();

                let ours = ShortConv {
                    // Projections are irrelevant here; depthwise_conv only
                    // touches the kernel fields.
                    in_proj: candle_nn::Linear::new(
                        Tensor::zeros((3 * hidden, hidden), DType::F32, &dev).unwrap(),
                        None,
                    ),
                    out_proj: candle_nn::Linear::new(
                        Tensor::zeros((hidden, hidden), DType::F32, &dev).unwrap(),
                        None,
                    ),
                    conv_weight: weight.clone(),
                    conv_bias: None,
                    kernel: k,
                    hidden_size: hidden,
                }
                .depthwise_conv(&bx, seq)
                .unwrap();

                let reference = Conv1d::new(
                    weight,
                    None,
                    Conv1dConfig {
                        padding: k / 2,
                        groups: hidden,
                        ..Default::default()
                    },
                )
                .forward(&bx)
                .unwrap();
                // Match the upstream trim for even k, which leaves an extra step.
                let reference = if reference.dim(2).unwrap() > seq {
                    reference.narrow(2, 0, seq).unwrap()
                } else {
                    reference
                };

                assert_eq!(ours.dims3().unwrap(), (1, hidden, seq), "k={k} seq={seq}");
                let max_d = (ours - reference)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .max(0)
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                assert!(max_d < 1e-5, "k={k} seq={seq}: max|Δ| = {max_d:e}");
            }
        }
    }

    /// A bias, when the config declares one, must land the same way.
    #[test]
    fn depthwise_conv_applies_bias_like_conv1d() {
        let dev = Device::Cpu;
        let (hidden, k, seq) = (16usize, 3usize, 9usize);
        let weight = Tensor::randn(0f32, 1., (hidden, 1, k), &dev).unwrap();
        let bias = Tensor::randn(0f32, 1., hidden, &dev).unwrap();
        let bx = Tensor::randn(0f32, 1., (1, hidden, seq), &dev).unwrap();

        let ours = ShortConv {
            in_proj: candle_nn::Linear::new(
                Tensor::zeros((3 * hidden, hidden), DType::F32, &dev).unwrap(),
                None,
            ),
            out_proj: candle_nn::Linear::new(
                Tensor::zeros((hidden, hidden), DType::F32, &dev).unwrap(),
                None,
            ),
            conv_weight: weight.clone(),
            conv_bias: Some(bias.clone()),
            kernel: k,
            hidden_size: hidden,
        }
        .depthwise_conv(&bx, seq)
        .unwrap();

        let reference = Conv1d::new(
            weight,
            Some(bias),
            Conv1dConfig {
                padding: k / 2,
                groups: hidden,
                ..Default::default()
            },
        )
        .forward(&bx)
        .unwrap();

        let max_d = (ours - reference)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(max_d < 1e-5, "max|Δ| = {max_d:e}");
    }
}

/// Non-causal short convolution — the gated `B * x` conv LFM2 uses in place
/// of attention in most layers.
///
/// The convolution is evaluated as `k` shifted, per-channel multiply-adds
/// rather than via [`candle_nn::Conv1d`]. That is not premature cleverness:
/// a depthwise conv here means `groups = hidden_size = 1024`, and candle's
/// grouped path costs a flat **~173 ms per call regardless of sequence
/// length** — it is dominated by per-group overhead, not arithmetic. Across
/// the 10 conv layers of a 350M checkpoint that was the entire ~1.9 s of a
/// single short embedding. The shifted form is the same arithmetic (a
/// depthwise conv IS a per-channel weighted sum of `k` shifts) and runs in
/// tens of microseconds. Parity with the Python reference is unchanged;
/// `tests/trunk_parity.rs` is what proves that.
#[derive(Debug)]
struct ShortConv {
    in_proj: Linear,
    out_proj: Linear,
    /// Per-channel taps, `(hidden, 1, k)` as stored in the checkpoint.
    conv_weight: Tensor,
    conv_bias: Option<Tensor>,
    kernel: usize,
    hidden_size: usize,
}

impl ShortConv {
    fn load(cfg: &Lfm2EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let k = cfg.conv_l_cache;
        let vb_c = vb.pp("conv");
        let weight = vb_c.get((h, 1, k), "weight")?;
        let bias = if cfg.conv_bias {
            Some(vb_c.get(h, "bias")?)
        } else {
            // `conv_bias` defaults to false when the key is absent. If the
            // checkpoint nonetheless ships a bias, the config is lying and
            // we would silently drop a trained parameter — refuse instead.
            if vb_c.contains_tensor("bias") {
                return Err(candle_core::Error::Msg(format!(
                    "checkpoint ships `{}conv.bias` but config says conv_bias=false; \
                     loading would silently drop a trained parameter",
                    vb_c.prefix()
                )));
            }
            None
        };
        Ok(Self {
            in_proj: candle_nn::linear_no_bias(h, 3 * h, vb.pp("in_proj"))?,
            out_proj: candle_nn::linear_no_bias(h, h, vb.pp("out_proj"))?,
            conv_weight: weight,
            conv_bias: bias,
            kernel: k,
            hidden_size: h,
        })
    }

    /// Centered depthwise convolution over `(batch, hidden, seq)`.
    ///
    /// Zero-pads `k/2` on each side, then accumulates `k` shifted windows
    /// scaled by their per-channel tap — identical arithmetic to a grouped
    /// `Conv1d`, minus the per-group dispatch that dominates it. Centered
    /// padding is the encoder branch's defining difference from the causal
    /// model, which pads `k - 1` on the left only.
    fn depthwise_conv(&self, bx: &Tensor, seq: usize) -> Result<Tensor> {
        let (b, h, _) = bx.dims3()?;
        let pad = self.kernel / 2;
        let padded = if pad == 0 {
            bx.clone()
        } else {
            let zeros = Tensor::zeros((b, h, pad), bx.dtype(), bx.device())?;
            Tensor::cat(&[&zeros, bx, &zeros], 2)?
        };

        let mut acc: Option<Tensor> = None;
        for j in 0..self.kernel {
            // Tap j for every channel: (hidden, 1, k) -> (1, hidden, 1).
            let tap = self.conv_weight.narrow(2, j, 1)?.reshape((1, h, 1))?;
            // Window j: for odd k this is x shifted by j - pad. Taking
            // exactly `seq` outputs also handles even k the way the
            // upstream trim does — by dropping the trailing step.
            let window = padded.narrow(2, j, seq)?;
            let term = window.broadcast_mul(&tap)?;
            acc = Some(match acc {
                None => term,
                Some(a) => (a + term)?,
            });
        }
        let out = acc.expect("kernel size is at least 1");
        match &self.conv_bias {
            Some(bias) => out.broadcast_add(&bias.reshape((1, h, 1))?),
            None => Ok(out),
        }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, seq, _) = xs.dims3()?;
        let h = self.hidden_size;

        // (b, seq, 3h) -> (b, 3h, seq), then split into B, C, x.
        let bcx = self.in_proj.forward(xs)?.transpose(1, 2)?;
        let b_gate = bcx.narrow(1, 0, h)?;
        let c_gate = bcx.narrow(1, h, h)?;
        let x_proj = bcx.narrow(1, 2 * h, h)?;

        let bx = (b_gate * &x_proj)?.contiguous()?;
        let conv_out = self.depthwise_conv(&bx, seq)?;

        let out = (c_gate * conv_out)?.transpose(1, 2)?.contiguous()?;
        self.out_proj.forward(&out)
    }
}

#[derive(Debug)]
enum Operator {
    Attention(Box<Attention>),
    ShortConv(Box<ShortConv>),
}

/// One block: pre-normed operator (attention or conv) + pre-normed FFN,
/// both residual.
#[derive(Debug)]
struct Block {
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
    operator: Operator,
    feed_forward: FeedForward,
}

impl Block {
    fn load(cfg: &Lfm2EncoderConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let operator = match cfg.layer_types[layer_idx] {
            LayerType::FullAttention => {
                Operator::Attention(Box::new(Attention::load(cfg, vb.pp("self_attn"))?))
            }
            LayerType::Conv => Operator::ShortConv(Box::new(ShortConv::load(cfg, vb.pp("conv"))?)),
        };
        Ok(Self {
            operator_norm: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.norm_eps,
                vb.pp("operator_norm"),
            )?,
            ffn_norm: candle_nn::rms_norm(cfg.hidden_size, cfg.norm_eps, vb.pp("ffn_norm"))?,
            operator,
            feed_forward: FeedForward::load(cfg, vb.pp("feed_forward"))?,
        })
    }

    fn forward(&self, xs: &Tensor, rope: Option<&(Tensor, Tensor)>, mask: Option<&Tensor>) -> Result<Tensor> {
        let normed = self.operator_norm.forward(xs)?;
        let delta = match &self.operator {
            Operator::Attention(attn) => attn.forward(&normed, rope, mask)?,
            // The conv takes no mask: in the eager/sdpa path the
            // checkpoints' `apply_mask_to_padding_states` is a no-op
            // (it bails on a 4D mask), so pad states DO flow through the
            // conv. See `Lfm2Trunk::forward` for what that means for
            // batching.
            Operator::ShortConv(conv) => conv.forward(&normed)?,
        };
        let xs = (xs + delta)?;
        let delta = self.feed_forward.forward(&self.ffn_norm.forward(&xs)?)?;
        xs + delta
    }
}

/// The bidirectional LFM2.5 encoder trunk. Produces per-token hidden
/// states; heads live on top of this.
#[derive(Debug)]
pub struct Lfm2Trunk {
    embed_tokens: Embedding,
    blocks: Vec<Block>,
    embedding_norm: RmsNorm,
    /// RoPE inverse frequencies, `(1, head_dim/2)`. Depends only on
    /// `head_dim` and `rope_theta`, so it is built once at load rather than
    /// per forward — these crates get embedded in daemons serving many
    /// short sequences, where per-call allocator churn is the cost that
    /// actually shows up.
    inv_freq: Tensor,
    use_pos_enc: bool,
    dtype: DType,
}

impl Lfm2Trunk {
    /// Load the trunk, probing [`TRUNK_PREFIXES`] for wherever this
    /// checkpoint hid it. Fails loudly and names what it looked for rather
    /// than falling back to an untied random init.
    pub fn load(cfg: &Lfm2EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let found: Vec<&&str> = TRUNK_PREFIXES
            .iter()
            .filter(|p| vb.contains_tensor(&format!("{p}embed_tokens.weight")))
            .collect();
        match found.as_slice() {
            [prefix] => Self::load_with_prefix(cfg, vb, prefix),
            [] => Err(candle_core::Error::Msg(format!(
                "no LFM2 trunk found: none of {TRUNK_PREFIXES:?} carry `embed_tokens.weight`"
            ))),
            // A merged or multi-task checkpoint could carry two. Picking
            // the first would load a real trunk from the wrong place and
            // never say so; make the caller name the one they meant via
            // `load_with_prefix`.
            many => Err(candle_core::Error::Msg(format!(
                "ambiguous LFM2 trunk: {many:?} all carry `embed_tokens.weight`; \
                 use load_with_prefix() to say which one you mean"
            ))),
        }
    }

    /// Load from an explicit prefix (`""` for the bare Embedding
    /// checkpoint, `"lfm2."` for the head-carrying ones). Head loaders use
    /// this once they've decided where the trunk lives.
    pub fn load_with_prefix(
        cfg: &Lfm2EncoderConfig,
        vb: VarBuilder,
        prefix: &str,
    ) -> Result<Self> {
        cfg.validate().map_err(candle_core::Error::Msg)?;
        let vb = if prefix.is_empty() {
            vb
        } else {
            vb.pp(prefix.trim_end_matches('.'))
        };

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let vb_l = vb.pp("layers");
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(cfg, i, vb_l.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let embedding_norm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.norm_eps, vb.pp("embedding_norm"))?;

        let head_dim = cfg.head_dim();
        let rope_theta = cfg.rope_theta();
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| (1f64 / rope_theta.powf(i as f64 / head_dim as f64)) as f32)
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), vb.device())?.reshape((1, ()))?;

        Ok(Self {
            embed_tokens,
            blocks,
            embedding_norm,
            inv_freq,
            use_pos_enc: cfg.use_pos_enc,
            dtype: vb.dtype(),
        })
    }

    /// RoPE tables for `seq` positions, built per call.
    ///
    /// `max_position_embeddings` is 128000 on every checkpoint while
    /// `sentence_bert_config.json` caps sequences at 512 — precomputing the
    /// full table would burn 16 MiB to serve 512 rows. Building `seq × 32`
    /// values per forward is noise next to a 350M-parameter pass.
    fn rope_tables(&self, seq: usize, device: &candle_core::Device) -> Result<(Tensor, Tensor)> {
        let positions = Tensor::arange(0u32, seq as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((seq, 1))?;
        let freqs = positions.matmul(&self.inv_freq.to_device(device)?)?;
        Ok((
            freqs.cos()?.to_dtype(self.dtype)?,
            freqs.sin()?.to_dtype(self.dtype)?,
        ))
    }

    /// Build the additive attention mask from a `(batch, seq)` padding
    /// mask of 1 = real token, 0 = pad. Shaped `(batch, 1, 1, seq)` so it
    /// broadcasts over heads and query positions — pad *keys* are
    /// suppressed for every query.
    fn additive_mask(pad_mask: &Tensor) -> Result<Tensor> {
        let (b, seq) = pad_mask.dims2()?;
        let keep = pad_mask.to_dtype(DType::F32)?;

        // The contract is 1 = real, 0 = pad. A fractional mask would sail
        // through as a soft mask the model was never trained for, so check
        // it: keep*(keep-1) is exactly zero iff every value is 0 or 1. Two
        // tiny ops on (batch, seq) — nothing next to the forward pass.
        let ones = Tensor::ones((b, seq), DType::F32, keep.device())?;
        let off_binary = (&keep * (&keep - &ones)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        if off_binary != 0.0 {
            return Err(candle_core::Error::Msg(format!(
                "attention_mask must be 1 (real) or 0 (pad); found a value that is \
                 neither (off-binary residual {off_binary:e}). A soft mask is not \
                 something these checkpoints were trained with."
            )));
        }

        ((ones - keep)? * MASK_NEG)?.reshape((b, 1, 1, seq))
    }

    /// Run the trunk. `input_ids` is `(batch, seq)`; `attention_mask` is an
    /// optional `(batch, seq)` padding mask (1 = real, 0 = pad). Returns
    /// per-token hidden states `(batch, seq, hidden)`.
    ///
    /// **Padding is not inert.** The conv layers are deliberately *not*
    /// masked — the checkpoints train with pad states flowing through the
    /// short conv, so we reproduce that. With `k = 3` centered, a pad
    /// bleeds one position into its real neighbour per conv layer, which
    /// means a sequence's embedding depends slightly on what it was
    /// batched with. Padding-free single-sequence calls are the
    /// reproducible path; batch when throughput matters more than
    /// bit-identical results.
    pub fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch, seq) = input_ids.dims2()?;
        // A multi-row batch with no mask is almost certainly a caller who
        // padded and forgot: pad tokens would be attended to as content AND
        // flow through the convs, quietly corrupting every row in the
        // batch. There is no output that would look wrong. Refuse.
        if attention_mask.is_none() && batch > 1 {
            return Err(candle_core::Error::Msg(format!(
                "forward() called with batch {batch} and no attention_mask: if these rows \
                 are padded the pad tokens are treated as content and silently corrupt \
                 every row. Pass a (batch, seq) mask, or call one row at a time."
            )));
        }
        let mask = attention_mask.map(Self::additive_mask).transpose()?;
        let rope = if self.use_pos_enc {
            Some(self.rope_tables(seq, input_ids.device())?)
        } else {
            // No checkpoint ships `use_pos_enc: false`; honoured rather
            // than assumed, and untested for that reason.
            None
        };

        let mut xs = self.embed_tokens.forward(input_ids)?;
        for block in &self.blocks {
            xs = block.forward(&xs, rope.as_ref(), mask.as_ref())?;
        }
        self.embedding_norm.forward(&xs)
    }

    /// CLS pooling: hidden state of position 0.
    ///
    /// The Embedding checkpoint's `1_Pooling/config.json` sets
    /// `pooling_mode_cls_token: true` — NOT mean pooling, which is the
    /// usual assumption for a bidirectional encoder and would produce
    /// quietly worse embeddings. Its `modules.json` also has no
    /// Normalize module, so this returns the raw pooled vector; L2-normalize
    /// at the call site if you want cosine similarity.
    pub fn pool_cls(hidden_states: &Tensor) -> Result<Tensor> {
        hidden_states.i((.., 0, ..))?.contiguous()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The device the weights live on — callers building input tensors need
    /// to match it.
    pub fn device(&self) -> &candle_core::Device {
        self.inv_freq.device()
    }
}
