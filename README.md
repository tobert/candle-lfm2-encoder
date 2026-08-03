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

**Day 0.** Milestone 1 (config) done: every family checkpoint's
`config.json` parses and validates against real fixtures
(`tests/fixtures/`, fetched 2026-08-03), with head detection. The
fixtures already earned their keep — the 230M base is *shallower* (14
layers) not just narrower, the PII config ships a literal
`"full_attn_idxs": null`, and the Router is not a softmax classifier.

Next: the bidirectional trunk (adapting the hybrid conv+attention blocks
from candle-transformers' `lfm2.rs`, minus causal mask and KV cache),
then heads in order: embedding → token classification → routing.

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
