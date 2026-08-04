//! `LFM2.5-ColBERT-350M`: late-interaction multi-vector retrieval.
//!
//! Where [`crate::embedding`] pools a passage down to one 1024-dim vector,
//! ColBERT keeps **one 128-dim vector per token** and scores with MaxSim:
//! every query token takes its best match in the document, and those
//! maxima are summed. That buys noticeably better retrieval at the cost of
//! storing ~hundreds of vectors per document instead of one.
//!
//! The conventions below were read out of PyLate's actual behaviour rather
//! than inferred from the config, because several of them are not what a
//! careful reading would predict:
//!
//! * **Query expansion pads with the EOS token, not a mask token.** This
//!   checkpoint's tokenizer has NO mask token (`mask_token_id` is null);
//!   the model card tells you to set `pad_token = eos_token`, making the
//!   expansion filler `<|im_end|>` (id 7). Padding with 0 — the config's
//!   nominal `pad_token_id` — would silently embed the wrong tokens.
//! * **`[Q]` and `[D]` are real vocabulary ids** (64400 / 64401), which is
//!   why this checkpoint's vocab is 64402 while the rest of the family is
//!   65536. They land at position 1, right after BOS.
//! * **Expansion tokens are masked as keys but still emitted as vectors**
//!   (`attend_to_expansion_tokens: false`). They cannot be attended TO, yet
//!   their own outputs are projected, normalized and scored — that is the
//!   whole point of query expansion, and it is exactly the case the
//!   checkpoint's modeling code warns breaks FlashAttention.
//! * **Documents drop punctuation vectors** via a skiplist of 32 token ids;
//!   queries do not.
//! * **Every vector is L2-normalized**, so MaxSim is a sum of cosines.

use std::collections::HashSet;
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::config::Lfm2EncoderConfig;
use crate::error::{Error, Result};
use crate::trunk::Lfm2Trunk;

/// `config_sentence_transformers.json`, the part that defines the pipeline.
#[derive(Debug, Deserialize)]
struct ColbertSettings {
    #[serde(default = "default_query_prefix")]
    query_prefix: String,
    #[serde(default = "default_document_prefix")]
    document_prefix: String,
    #[serde(default = "default_query_length")]
    query_length: usize,
    #[serde(default = "default_document_length")]
    document_length: usize,
    #[serde(default)]
    skiplist_words: Vec<String>,
    #[serde(default = "default_true")]
    do_query_expansion: bool,
    #[serde(default)]
    attend_to_expansion_tokens: bool,
}

fn default_query_prefix() -> String {
    "[Q] ".to_string()
}
fn default_document_prefix() -> String {
    "[D] ".to_string()
}
fn default_query_length() -> usize {
    32
}
fn default_document_length() -> usize {
    512
}
fn default_true() -> bool {
    true
}

/// One document or query as a bag of per-token vectors.
#[derive(Debug, Clone)]
pub struct MultiVector {
    /// Row-major `[n_tokens][dim]`, each row L2-normalized.
    pub vectors: Vec<Vec<f32>>,
}

impl MultiVector {
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// MaxSim: for each of `self`'s token vectors take its best match in
    /// `doc`, and sum. Asymmetric on purpose — call it as
    /// `query.max_sim(&document)`.
    ///
    /// Because every vector is unit length, each term is a cosine, so the
    /// score scales with query length (a 32-token query maxes out near
    /// 32.0). Scores are comparable ACROSS documents for one query, not
    /// across queries.
    pub fn max_sim(&self, doc: &MultiVector) -> f32 {
        if doc.is_empty() {
            return 0.0;
        }
        self.vectors
            .iter()
            .map(|q| {
                doc.vectors
                    .iter()
                    .map(|d| q.iter().zip(d).map(|(a, b)| a * b).sum::<f32>())
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .sum()
    }
}

/// A loaded ColBERT model: trunk + the separate `1_Dense` projection.
#[derive(Debug)]
pub struct ColbertModel {
    trunk: Lfm2Trunk,
    dense: Linear,
    tokenizer: Tokenizer,
    settings: ColbertSettings,
    /// Token ids whose vectors are dropped from DOCUMENTS.
    skiplist: HashSet<u32>,
    /// Filler for query expansion — the EOS id, per the model card.
    expansion_id: u32,
    dim: usize,
}

impl ColbertModel {
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        Self::from_dir_with(dir, DType::F32, &Device::Cpu)
    }

    pub fn from_dir_with(dir: impl AsRef<Path>, dtype: DType, device: &Device) -> Result<Self> {
        let dir = dir.as_ref();

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

        let st_path = dir.join("config_sentence_transformers.json");
        let st_bytes = std::fs::read(&st_path).map_err(|e| Error::io(&st_path, e))?;
        let settings: ColbertSettings =
            serde_json::from_slice(&st_bytes).map_err(|source| Error::ConfigParse {
                path: st_path.clone(),
                source,
            })?;

        let tok_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| Error::Tokenizer {
            path: tok_path.clone(),
            message: e.to_string(),
        })?;

        // Punctuation ids to drop from documents. A word the tokenizer
        // doesn't know as a single token simply has no id to skip.
        let skiplist: HashSet<u32> = settings
            .skiplist_words
            .iter()
            .filter_map(|w| tokenizer.token_to_id(w))
            .collect();

        // Query expansion pads with EOS, NOT with pad_token_id. The model
        // card instructs `tokenizer.pad_token = tokenizer.eos_token`, and
        // this checkpoint has no mask token at all.
        let expansion_id = cfg.eos_token_id.ok_or_else(|| Error::ConfigInvalid {
            path: cfg_path.clone(),
            message: "ColBERT query expansion pads with the EOS token, but this config \
                      carries no eos_token_id"
                .to_string(),
        })?;

        let weights = dir.join("model.safetensors");
        if !weights.is_file() {
            return Err(Error::io(
                &weights,
                std::io::Error::new(std::io::ErrorKind::NotFound, "no model.safetensors"),
            ));
        }
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device)? };
        let trunk = Lfm2Trunk::load(&cfg, vb)?;

        // The projection ships in its OWN safetensors file, which is why a
        // reader of model.safetensors alone concludes this is a bare trunk.
        let dense_path = dir.join("1_Dense").join("model.safetensors");
        if !dense_path.is_file() {
            return Err(Error::io(
                &dense_path,
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "ColBERT needs its 1_Dense projection; without it the trunk would \
                     silently emit unprojected 1024-dim vectors",
                ),
            ));
        }
        let dense_cfg_path = dir.join("1_Dense").join("config.json");
        let dense_cfg: DenseConfig = serde_json::from_slice(
            &std::fs::read(&dense_cfg_path).map_err(|e| Error::io(&dense_cfg_path, e))?,
        )
        .map_err(|source| Error::ConfigParse {
            path: dense_cfg_path.clone(),
            source,
        })?;
        if dense_cfg.bias {
            return Err(Error::ConfigInvalid {
                path: dense_cfg_path,
                message: "1_Dense declares a bias; only the bias-free projection is \
                          implemented and verified"
                    .to_string(),
            });
        }
        let dense_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[dense_path], dtype, device)? };
        let dense = candle_nn::linear_no_bias(
            dense_cfg.in_features,
            dense_cfg.out_features,
            dense_vb.pp("linear"),
        )?;

        Ok(Self {
            trunk,
            dense,
            tokenizer,
            settings,
            skiplist,
            expansion_id,
            dim: dense_cfg.out_features,
        })
    }

    /// Per-token vector width (128).
    pub fn dim(&self) -> usize {
        self.dim
    }

    fn encode_ids(&self, text: &str, prefix: &str, limit: usize) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(format!("{prefix}{text}"), true)
            .map_err(|e| Error::Encode(e.to_string()))?;
        let mut ids = encoding.get_ids().to_vec();
        ids.truncate(limit);
        Ok(ids)
    }

    /// Run the trunk + projection and L2-normalize each token vector.
    fn project(&self, ids: &[u32], mask: Option<&[u32]>) -> Result<Vec<Vec<f32>>> {
        let device = self.trunk.device();
        let input = Tensor::new(ids, device)?.unsqueeze(0)?;
        let mask_t = match mask {
            Some(m) => Some(Tensor::new(m, device)?.unsqueeze(0)?),
            None => None,
        };
        let hidden = self.trunk.forward(&input, mask_t.as_ref())?;
        let projected = self.dense.forward(&hidden)?.to_dtype(DType::F32)?;

        // L2-normalize per token: MaxSim is then a sum of cosines.
        let norms = projected.sqr()?.sum_keepdim(2)?.sqrt()?;
        let normalized = projected.broadcast_div(&norms)?;
        Ok(normalized.i(0)?.to_vec2::<f32>()?)
    }

    /// Encode a query: `[Q]` marker, expanded to exactly `query_length`
    /// tokens with EOS filler. Every position is returned, including the
    /// expansion tokens — they are masked as attention KEYS but their own
    /// outputs are what query expansion is for.
    pub fn encode_query(&self, text: &str) -> Result<MultiVector> {
        let mut ids = self.encode_ids(text, &self.settings.query_prefix, self.settings.query_length)?;
        let real = ids.len();

        if self.settings.do_query_expansion {
            ids.resize(self.settings.query_length, self.expansion_id);
        }
        let mask: Vec<u32> = (0..ids.len())
            .map(|i| u32::from(i < real || self.settings.attend_to_expansion_tokens))
            .collect();

        Ok(MultiVector {
            vectors: self.project(&ids, Some(&mask))?,
        })
    }

    /// Encode a document: `[D]` marker, no expansion, punctuation vectors
    /// dropped per the skiplist.
    pub fn encode_document(&self, text: &str) -> Result<MultiVector> {
        let ids = self.encode_ids(
            text,
            &self.settings.document_prefix,
            self.settings.document_length,
        )?;
        // Unpadded single sequence, so no mask is needed — and that is the
        // reproducible path (see the crate docs on padding).
        let all = self.project(&ids, None)?;

        let vectors = all
            .into_iter()
            .zip(&ids)
            .filter(|(_, id)| !self.skiplist.contains(id))
            .map(|(v, _)| v)
            .collect();
        Ok(MultiVector { vectors })
    }

    /// Token ids for a query, exposed so a tokenizer mismatch can be
    /// diagnosed as itself.
    pub fn query_ids(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = self.encode_ids(text, &self.settings.query_prefix, self.settings.query_length)?;
        if self.settings.do_query_expansion {
            ids.resize(self.settings.query_length, self.expansion_id);
        }
        Ok(ids)
    }

    /// Token ids for a document, before skiplist filtering.
    pub fn document_ids(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_ids(
            text,
            &self.settings.document_prefix,
            self.settings.document_length,
        )
    }
}

#[derive(Debug, Deserialize)]
struct DenseConfig {
    in_features: usize,
    out_features: usize,
    #[serde(default)]
    bias: bool,
}
