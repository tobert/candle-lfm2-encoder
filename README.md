# candle-lfm2-encoder

LiquidAI's **LFM2.5 bidirectional encoders** on [candle] — pure Rust,
CPU-first, no Python, no C++.

Upstream candle-transformers implements the *causal* LFM2
(`models/lfm2.rs`). Nobody implements the encoder branch —
`Lfm2BidirectionalModel` and its task heads. This crate is that gap:

| checkpoint | head | why we want it |
|---|---|---|
| `LFM2.5-Embedding-350M` | pooled embedding (1024-dim) | text embeddings in-process |
| `LFM2.5-Encoder-350M-PII-Detector` | token classification (BIOES, 161 labels **including credentials/secrets**) | boundary screening of foreign prose & outbound payloads |
| `LFM2.5-Encoder-350M-Policy-Linter` | token classification | same lane, policy flavor |
| `LFM2.5-Encoder-350M-Prompt-Router` | sequence **routing** (`rule_proj_dim` — scores prompts against rule projections, zero-shot-shaped) | dynamic model-lane routing |
| `LFM2.5-Encoder-230M/350M` | masked-LM bases | fine-tune substrate |

## Status

**Milestone 2 done: the bidirectional trunk runs and matches the
reference.** `Lfm2Trunk` reproduces LiquidAI's own
`modeling_lfm2_bidirectional.py` to max|Δ| ≈ 4.6e-5 on f32 CPU, verified
against activations dumped from the real Embedding-350M weights
(`tests/reference/dump_trunk_reference.py`, transformers 4.56.2).

Milestone 1 (config) covers every family checkpoint. The fixtures keep
earning their keep — the 230M base is *shallower* (14 layers) not just
narrower; the PII config ships a literal `"full_attn_idxs": null`; the
Router is not a softmax classifier; `intermediate_size` **disagrees with
the shipped weights** on three of four checkpoints (says 6656, ships
4608 — see `Lfm2EncoderConfig::ffn_dim`); and the Policy-Linter turned
out to be a fifth architecture name, `Lfm2BidirForRuleMatching`.

Next: heads, in order — pooled embedding → token classification →
routing/rule-matching.

## Batching changes your embeddings (read this)

The short conv is deliberately **not** masked, because that is how these
checkpoints were trained: in the eager/sdpa path the reference's
`apply_mask_to_padding_states` is a no-op, so pad states flow through the
conv. With a centered `k = 3` kernel, each conv layer bleeds a pad one
position into its real neighbour.

**Consequence:** a sequence's embedding depends slightly on what it was
batched with. Unlike BERT, padding is not inert here. If you need
reproducible vectors — cache keys, stored embeddings, anything compared
across processes — embed one sequence at a time with no padding. Batch
when throughput matters more than bit-identical results.

`forward()` refuses a multi-row batch with no `attention_mask` rather
than silently treating pad tokens as content.

## Design intents

- **CPU-first.** Consumers embed this in long-lived server processes
  where a 350M encoder pass is tens of milliseconds. GPU (CUDA/Metal via
  candle features) is a bonus, never a requirement.
- **Pure Rust.** No C++ toolchain in the dependency tree.
- **Local weights by default.** Point at a directory holding
  `model.safetensors` + `tokenizer.json` + `config.json`; the optional
  `hub` feature adds Hub download.

## License & attribution

MIT OR Apache-2.0, matching candle. Trunk block implementations are
adapted from [candle-transformers]' `lfm2.rs` (© the candle authors, MIT
OR Apache-2.0); attribution retained in source where adapted.

[candle]: https://github.com/huggingface/candle
[candle-transformers]: https://github.com/huggingface/candle/tree/main/candle-transformers
