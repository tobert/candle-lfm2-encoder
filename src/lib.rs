//! LFM2.5 bidirectional encoders on [candle] — pure Rust, CPU-first.
//!
//! LiquidAI's LFM2.5 encoder branch is a family of small (230M/350M)
//! hybrid conv+attention models with task heads: text embedding, masked-LM
//! fine-tune bases, token classification (PII/secrets detection, policy
//! linting), and prompt routing. Upstream candle-transformers implements
//! the *causal* LFM2 (`lfm2.rs`); nothing implements the bidirectional
//! encoder branch. This crate is that gap.
//!
//! ## Milestones
//!
//! 1. **Config** (done): parse every family checkpoint's `config.json`,
//!    verified against real fixtures; head detection via [`config::EncoderArch`].
//! 2. **Trunk** (done): the bidirectional hybrid stack ([`Lfm2Trunk`]) —
//!    blocks adapted from candle-transformers' `lfm2.rs` (MIT/Apache-2.0,
//!    attribution retained), minus the causal mask and KV cache, plus the
//!    encoder branch's *centered* short conv. Verified to max|Δ| ≈ 5e-5
//!    against the checkpoint's own Python modeling code.
//! 3. **Heads**, in order of use: pooled embedding (`Embedding-350M`),
//!    token classification (`PII-Detector`, `Policy-Linter`), sequence
//!    routing (`Prompt-Router` — rule-projection scoring, see its
//!    `rule_proj_dim`; not a fixed-label softmax).
//! 4. **Quantized loads** (GGUF) if CPU f16/f32 latency disappoints.
//!
//! Design intents: CPU-first (consumers embed this in long-lived server
//! processes — kaijutsu's kernel, kaibo — where a 350M encoder pass is
//! tens of milliseconds); no C++ in the dependency tree; weights loaded
//! from a local dir by default (`hub` feature adds Hub download).
//!
//! [candle]: https://github.com/huggingface/candle

pub mod config;
pub mod trunk;

pub use config::{EncoderArch, LayerType, Lfm2EncoderConfig};
pub use trunk::Lfm2Trunk;
