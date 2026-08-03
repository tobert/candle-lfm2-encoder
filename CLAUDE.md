# candle-lfm2-encoder

LFM2.5 bidirectional encoders on candle. Read README.md for the what;
this file is the working context.

## Where things are

- **Adaptation source**: `~/src/research/candle/candle-transformers/src/models/lfm2.rs`
  (659 lines, the CAUSAL branch — config/blocks/attention are the parts we
  adapt; drop causal mask + KV cache). Dual MIT/Apache-2.0; keep
  attribution in adapted files. candle has no CONTRIBUTING.md and no AI
  policy as of 2026-08-03; upstreaming is a later decision Amy makes after
  reading whatever policy exists then.
- **Fixtures**: `tests/fixtures/*.config.json` — REAL Hub configs
  (2026-08-03). Refresh with curl from
  `https://huggingface.co/LiquidAI/<model>/raw/main/config.json`.
- **HF auth**: `hf` CLI is logged in as AmyTobey on this machine; weights
  download with `hf download LiquidAI/<model> --local-dir <dir>`.
- **Python reference**: HF transformers `models/lfm2/modeling_lfm2.py`
  (causal) — the bidirectional variants live in the checkpoints'
  `auto_map` custom code on the Hub; READ the checkpoint's own modeling
  file for head shapes before implementing a head, don't guess.

## Checkpoint facts (fixture-verified — trust these over docs)

- Hybrid stack via `layer_types` (conv-heavy, interleaved full_attention);
  the 230M base has **14** layers, 350M family has 16. hidden 1024,
  16 heads / 8 kv-heads, vocab 65536.
- PII detector: BIOES over 40 types, **161 labels incl.
  credential.api_key/jwt/private_key/connection_string** — it's a secrets
  detector too. Ships `"full_attn_idxs": null` (key present, value null).
- Prompt-Router: `Lfm2BidirForSequenceRouting`, `rule_proj_dim` — scores
  prompts against rule projections (zero-shot-shaped), NOT a fixed label
  softmax.
- rope theta appears BOTH flat (`rope_theta`) and structured
  (`rope_parameters.rope_theta`) across checkpoints; `Lfm2EncoderConfig::rope_theta()`
  resolves precedence, default 1e6.

## Consumers (the reason this exists)

kaijutsu (router eval → cast-lane routing; embedding swap candidate vs
bge-small) and kaibo (boundary guards on foreign-repo reads + outbound
batch payloads; candle-only there — no C++ allowed in kaibo's build).
Cross-project map lives in kaijutsu's `docs/issues.md` ("LFM2.5 encoder
family").

## Conventions

Amy's global CLAUDE.md applies (TDD, 改善, loud failures). Verify against
fixtures, not documentation — the fixtures have already contradicted
plausible assumptions three times on day 0.
