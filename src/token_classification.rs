//! Token classification — `LFM2.5-Encoder-350M-PII-Detector` and friends:
//! a per-token BIOES head over the checkpoint's label set.
//!
//! # The label set comes from `config.json`, never `label_schema.json`
//!
//! The PII checkpoint ships BOTH, and they disagree: `label_schema.json`
//! declares **109 labels over 27 types**, while `config.json` and the
//! actual `classifier.weight` are **161 labels over 40 types**. The schema
//! file is stale, and what it omits is not incidental — it is missing
//! `credential.connection_string`, `credential.jwt`, `credential.password`
//! and `credential.private_key`, i.e. every credential type a guard tool
//! would be reading this model for. Decoding 161-wide logits through that
//! file's `id2label` would mislabel silently.
//!
//! # This head is not the whole product
//!
//! The checkpoint also ships `pii_hybrid_decode.py`, which wraps this head
//! in a regex + validator tier and is what LiquidAI calls the intended
//! decode. That tier is deliberately NOT implemented here: it is product
//! policy rather than the model, and measurement does not flatter it —
//! on our eval set it fires on 30% of clean text against this head's 4%,
//! and its ordered regexes can mislabel a connection string as an email
//! (the email pattern claims `user:password@host` first, suppressing the
//! `credential.connection_string` this head gets right). Consumers who
//! want it should port it deliberately, with that trade in view.

use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use tokenizers::Tokenizer;

use crate::config::Lfm2EncoderConfig;
use crate::error::{Error, Result};
use crate::labels::order_labels;
use crate::trunk::Lfm2Trunk;

/// A detected entity, with byte offsets into the input string.
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    /// Byte offset of the first byte, into the text passed to `predict`.
    pub start: usize,
    /// Byte offset one past the last byte.
    pub end: usize,
    /// Entity type with the BIOES prefix stripped, e.g.
    /// `credential.api_key`.
    pub label: String,
}

impl Span {
    /// The matched text. Panics if the offsets are not char boundaries,
    /// which would mean the tokenizer and the input disagree.
    pub fn text<'a>(&self, source: &'a str) -> &'a str {
        &source[self.start..self.end]
    }
}

/// A loaded token-classification checkpoint.
#[derive(Debug)]
pub struct Lfm2TokenClassifier {
    trunk: Arc<Lfm2Trunk>,
    classifier: Linear,
    tokenizer: Tokenizer,
    /// Label string per class id, ordered by id.
    id2label: Vec<String>,
}

/// Parse+validate `config.json`, pull out `id2label`, and load the
/// tokenizer — the prefix shared by [`Lfm2TokenClassifier::from_dir_with`]
/// (which goes on to load its own trunk) and
/// [`Lfm2TokenClassifier::from_trunk`] (which reuses one instead).
fn load_head_metadata(dir: &Path) -> Result<(Lfm2EncoderConfig, Vec<String>, Tokenizer)> {
    let cfg_path = dir.join("config.json");
    let bytes = std::fs::read(&cfg_path).map_err(|e| Error::io(&cfg_path, e))?;
    let cfg = Lfm2EncoderConfig::from_json(&bytes).map_err(|source| Error::ConfigParse {
        path: cfg_path.clone(),
        source,
    })?;
    cfg.validate().map_err(|message| Error::ConfigInvalid {
        path: cfg_path.clone(),
        message,
    })?;

    let labels = cfg.id2label.as_ref().ok_or_else(|| Error::ConfigInvalid {
        path: cfg_path.clone(),
        message: "no id2label: this checkpoint carries no token-classification labels"
            .to_string(),
    })?;
    let id2label = order_labels(labels, &cfg_path)?;

    let tok_path = dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| Error::Tokenizer {
        path: tok_path.clone(),
        message: e.to_string(),
    })?;

    Ok((cfg, id2label, tokenizer))
}

impl Lfm2TokenClassifier {
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        Self::from_dir_with(dir, DType::F32, &Device::Cpu)
    }

    pub fn from_dir_with(dir: impl AsRef<Path>, dtype: DType, device: &Device) -> Result<Self> {
        let dir = dir.as_ref();
        let (cfg, id2label, tokenizer) = load_head_metadata(dir)?;

        let weights = dir.join("model.safetensors");
        if !weights.is_file() {
            return Err(Error::io(
                &weights,
                std::io::Error::new(std::io::ErrorKind::NotFound, "no model.safetensors"),
            ));
        }
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device)? };
        let trunk = Arc::new(Lfm2Trunk::load(&cfg, vb.clone())?);

        // A plain Linear over the trunk's per-token hidden states, WITH a
        // bias (unlike every projection inside the trunk). The checkpoint
        // also ships `class_weights` — a training-time class-balancing
        // artifact, not an inference parameter; loading it into anything
        // would be wrong.
        let classifier = candle_nn::linear(cfg.hidden_size, id2label.len(), vb.pp("classifier"))?;

        Ok(Self {
            trunk,
            classifier,
            tokenizer,
            id2label,
        })
    }

    /// Build a head over an already-loaded, possibly-shared trunk — loads
    /// ONLY the tokenizer and the classifier weights from `dir`, at the
    /// trunk's own dtype/device. `dir`'s own trunk weights (under `lfm2.`
    /// in the same `model.safetensors`) are never read; see
    /// [`crate::sequence_classification::Lfm2SequenceClassifier::from_trunk`]
    /// for the fuller rationale — this head mirrors it exactly.
    ///
    /// Errors loudly, naming every mismatched field, if `dir`'s config
    /// disagrees with the trunk's shape — see [`Lfm2Trunk::check_compatible`].
    pub fn from_trunk(trunk: Arc<Lfm2Trunk>, dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let (cfg, id2label, tokenizer) = load_head_metadata(dir)?;

        let cfg_path = dir.join("config.json");
        trunk
            .check_compatible(&cfg)
            .map_err(|message| Error::ConfigInvalid { path: cfg_path, message })?;

        let weights = dir.join("model.safetensors");
        if !weights.is_file() {
            return Err(Error::io(
                &weights,
                std::io::Error::new(std::io::ErrorKind::NotFound, "no model.safetensors"),
            ));
        }
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights], trunk.dtype(), trunk.device())?
        };
        let classifier = candle_nn::linear(cfg.hidden_size, id2label.len(), vb.pp("classifier"))?;

        Ok(Self {
            trunk,
            classifier,
            tokenizer,
            id2label,
        })
    }

    /// The underlying trunk, for callers that want raw hidden states or
    /// want to confirm two heads share the same loaded trunk.
    pub fn trunk(&self) -> &Lfm2Trunk {
        &self.trunk
    }

    /// Number of classes (161 for the PII detector).
    pub fn num_labels(&self) -> usize {
        self.id2label.len()
    }

    /// The distinct entity types, BIOES prefixes stripped.
    pub fn entity_types(&self) -> Vec<&str> {
        let mut types: Vec<&str> = self
            .id2label
            .iter()
            .filter_map(|l| l.split_once('-').map(|(_, t)| t))
            .collect();
        types.sort_unstable();
        types.dedup();
        types
    }

    /// Per-token predicted label ids, alongside each token's byte span.
    fn token_labels(&self, text: &str) -> Result<(Vec<usize>, Vec<(usize, usize)>)> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::Encode(e.to_string()))?;
        let ids = encoding.get_ids();
        let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();

        let device = self.trunk.device();
        let input = Tensor::new(ids, device)?.unsqueeze(0)?;
        let hidden = self.trunk.forward(&input, None)?;
        let logits = self.classifier.forward(&hidden)?.to_dtype(DType::F32)?;

        let best = logits.i(0)?.argmax(1)?.to_vec1::<u32>()?;
        Ok((best.into_iter().map(|i| i as usize).collect(), offsets))
    }

    /// Per-token predicted class ids, one per token including specials.
    ///
    /// Exposed so a divergence in the MODEL can be told apart from a
    /// divergence in the span DECODE — they have completely different
    /// causes and the distinction is worth being able to make directly.
    pub fn token_label_ids(&self, text: &str) -> Result<Vec<usize>> {
        Ok(self.token_labels(text)?.0)
    }

    /// Detect entities in `text`.
    ///
    /// Decoding follows the checkpoint's own `model_spans`, which is
    /// deliberately forgiving about malformed BIOES: a span starts on `B-`
    /// or `S-` or a change of type, and otherwise extends. Tokens with an
    /// empty offset span (the specials) close the current span, and
    /// leading/trailing whitespace is trimmed off the result.
    pub fn predict(&self, text: &str) -> Result<Vec<Span>> {
        let (label_ids, offsets) = self.token_labels(text)?;

        let mut spans: Vec<Span> = Vec::new();
        let mut current: Option<Span> = None;

        for (&label_id, &(start, end)) in label_ids.iter().zip(&offsets) {
            let label = self
                .id2label
                .get(label_id)
                .map(String::as_str)
                .unwrap_or("O");

            if end <= start || label == "O" {
                spans.extend(current.take());
                continue;
            }
            let (prefix, entity) = match label.split_once('-') {
                Some((p, t)) => (p, t),
                None => ("S", label),
            };

            let starts_new = matches!(prefix, "B" | "S")
                || current.as_ref().is_none_or(|c| c.label != entity);
            if starts_new {
                spans.extend(current.take());
                current = Some(Span {
                    start,
                    end,
                    label: entity.to_string(),
                });
            } else if let Some(c) = current.as_mut() {
                c.end = end;
            }
        }
        spans.extend(current);

        // Trim whitespace, then drop anything that trimmed to nothing.
        let bytes = text.as_bytes();
        for span in &mut spans {
            while span.start < span.end && bytes[span.start].is_ascii_whitespace() {
                span.start += 1;
            }
            while span.end > span.start && bytes[span.end - 1].is_ascii_whitespace() {
                span.end -= 1;
            }
        }
        spans.retain(|s| s.end > s.start);
        Ok(spans)
    }

    /// Spans whose type is a `credential.*` — the boundary-guard question.
    pub fn credentials(&self, text: &str) -> Result<Vec<Span>> {
        Ok(self
            .predict(text)?
            .into_iter()
            .filter(|s| s.label.starts_with("credential."))
            .collect())
    }
}
