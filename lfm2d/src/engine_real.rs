//! [`crate::worker::InferenceEngine`] backed by real candle models —
//! what `main.rs` spawns in production. Loads each configured checkpoint
//! exactly once (via the library's own `from_dir` constructors — this
//! crate never touches a `VarBuilder` or a tensor directly), computes each
//! checkpoint's weight hash, and answers every [`crate::worker::WorkerCommand`]
//! by calling straight through to `candle-lfm2-encoder`'s public API.
//!
//! No trunk sharing: each head loads its own trunk via `from_dir`, even
//! though the library supports `Lfm2Trunk::load_shared` + `from_trunk` for
//! heads that share one checkpoint family. The three heads this daemon
//! serves (a generic embedder, an arbitrary sequence classifier, the
//! Prompt-Router) are independently-trained checkpoints in the general
//! case — `kube_ordinal_v6` and the Prompt-Router do NOT share a trunk (see
//! `src/cascade.rs`'s module docs) — so there is no trunk-sharing
//! opportunity to take here without assuming a same-trunk deployment this
//! service's config doesn't promise. Noted as a judgment call, not an
//! oversight: see `lfm2d/README.md`.

use std::path::Path;

use candle_lfm2_encoder::{
    resolve_severe_labels, Cascade, Lfm2Embedding, Lfm2EncoderConfig, Lfm2SequenceClassifier,
    Lfm2SequenceRouter, TextKind,
};

use crate::config::Cli;
use crate::hash::sha256_hex_file;
use crate::types::{
    CascadeClause, CascadeLane, CascadeModelRef, CascadeResponse, CascadeWinner, ClassifyResult,
    EmbedKind, LabelScore, ModelInfo, ModelKind, RouteResponse, RouteScore,
};
use crate::worker::{EmbedOutcome, InferenceEngine, PredictOutcome, WorkerError};

/// A loaded checkpoint's audit identity, computed once at load time.
#[derive(Debug, Clone)]
struct ModelMeta {
    id: String,
    weight_hash: String,
    hidden_size: usize,
}

/// Directory basename as the model id (`LFM2.5-Embedding-350M`,
/// `kube_ordinal_v6`, ...) — falls back to the full path if the directory
/// has no basename component (e.g. `/`), which should never happen for a
/// real checkpoint dir but must not panic if it somehow does.
fn model_id_from_dir(dir: &Path) -> String {
    dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| dir.display().to_string())
}

/// `hidden_size` is read directly from `config.json` rather than from any
/// loaded head object: none of [`Lfm2Embedding`]/[`Lfm2SequenceClassifier`]/
/// [`Lfm2SequenceRouter`]/[`candle_lfm2_encoder::Lfm2Trunk`] exposes it
/// uniformly (`Lfm2Embedding::dim` exists; the others don't), and every
/// head's checkpoint carries this same `config.json` regardless of which
/// head it is, so reading it directly is both simpler and uniform across
/// all three head kinds.
fn read_hidden_size(dir: &Path) -> Result<usize, String> {
    let cfg_path = dir.join("config.json");
    let bytes = std::fs::read(&cfg_path).map_err(|e| format!("reading {}: {e}", cfg_path.display()))?;
    let cfg = Lfm2EncoderConfig::from_json(&bytes).map_err(|e| format!("parsing {}: {e}", cfg_path.display()))?;
    Ok(cfg.hidden_size)
}

fn load_meta(dir: &Path) -> Result<ModelMeta, String> {
    let weights = dir.join("model.safetensors");
    let weight_hash = sha256_hex_file(&weights)
        .map_err(|e| format!("hashing weights at {}: {e}", weights.display()))?;
    Ok(ModelMeta { id: model_id_from_dir(dir), weight_hash, hidden_size: read_hidden_size(dir)? })
}

pub struct RealEngine {
    embedder: Option<(Lfm2Embedding, ModelMeta)>,
    classifier: Option<(Lfm2SequenceClassifier, ModelMeta)>,
    router: Option<(Lfm2SequenceRouter, ModelMeta)>,
    cascade_routes: Vec<String>,
    cascade_severe_labels: Vec<String>,
}

impl RealEngine {
    /// Load every head named in `cli`. Fails loudly and stops at the FIRST
    /// problem — a bad config, a missing file, an incompatible checkpoint —
    /// rather than starting a server that can only serve some of what it
    /// was told to. No lazy loading: every head is fully loaded (weights
    /// mmapped and read once for hashing) before this returns.
    ///
    /// If `--cascade-route` is configured, this ALSO validates now (not on
    /// first request) that both a classifier and a router are loaded and
    /// that every `--cascade-severe-label` names a real label on the loaded
    /// classifier — a typo'd severe-label name would otherwise silently
    /// zero out every clause's severity score on the first real request
    /// instead of failing at the moment the operator could still fix it.
    pub fn load(cli: &Cli) -> Result<Self, String> {
        let embedder = match &cli.embedder_dir {
            Some(dir) => {
                let model = Lfm2Embedding::from_dir(dir)
                    .map_err(|e| format!("loading embedder at {}: {e}", dir.display()))?;
                let meta = load_meta(dir)?;
                Some((model, meta))
            }
            None => None,
        };
        let classifier = match &cli.classifier_dir {
            Some(dir) => {
                let model = Lfm2SequenceClassifier::from_dir(dir)
                    .map_err(|e| format!("loading classifier at {}: {e}", dir.display()))?;
                let meta = load_meta(dir)?;
                Some((model, meta))
            }
            None => None,
        };
        let router = match &cli.router_dir {
            Some(dir) => {
                let model = Lfm2SequenceRouter::from_dir(dir)
                    .map_err(|e| format!("loading router at {}: {e}", dir.display()))?;
                let meta = load_meta(dir)?;
                Some((model, meta))
            }
            None => None,
        };

        if !cli.cascade_routes.is_empty() {
            let (classifier_model, _) = classifier.as_ref().ok_or_else(|| {
                "--cascade-route was given but no --classifier-dir was configured: \
                 /v1/cascade needs both a classifier and a router"
                    .to_string()
            })?;
            if router.is_none() {
                return Err(
                    "--cascade-route was given but no --router-dir was configured: \
                     /v1/cascade needs both a classifier and a router"
                        .to_string(),
                );
            }
            resolve_severe_labels(classifier_model.labels(), &cli.cascade_severe_labels)
                .map_err(|e| format!("--cascade-severe-label: {e}"))?;
        }

        Ok(Self {
            embedder,
            classifier,
            router,
            cascade_routes: cli.cascade_routes.clone(),
            cascade_severe_labels: cli.cascade_severe_labels.clone(),
        })
    }
}

/// Build the full per-label score map plus the argmax `(label, score)` in
/// one pass — shared by `predict`/`classify`/`cascade`'s per-clause
/// breakdown, all of which need exactly this from a `probs` vector plus its
/// label names.
fn scores_and_top(labels: &[String], probs: &[f32]) -> (std::collections::BTreeMap<String, f32>, String) {
    let mut map = std::collections::BTreeMap::new();
    let mut top = (String::new(), f32::NEG_INFINITY);
    for (label, &score) in labels.iter().zip(probs) {
        if score > top.1 {
            top = (label.clone(), score);
        }
        map.insert(label.clone(), score);
    }
    (map, top.0)
}

impl InferenceEngine for RealEngine {
    fn list_models(&self) -> Vec<ModelInfo> {
        let mut out = Vec::new();
        if let Some((_, meta)) = &self.embedder {
            out.push(ModelInfo {
                id: meta.id.clone(),
                kind: ModelKind::Embedder,
                weight_hash: meta.weight_hash.clone(),
                labels: None,
                hidden_size: meta.hidden_size,
            });
        }
        if let Some((model, meta)) = &self.classifier {
            out.push(ModelInfo {
                id: meta.id.clone(),
                kind: ModelKind::Classifier,
                weight_hash: meta.weight_hash.clone(),
                labels: Some(model.labels().to_vec()),
                hidden_size: meta.hidden_size,
            });
        }
        if let Some((_, meta)) = &self.router {
            out.push(ModelInfo {
                id: meta.id.clone(),
                kind: ModelKind::Router,
                weight_hash: meta.weight_hash.clone(),
                labels: None,
                hidden_size: meta.hidden_size,
            });
        }
        out
    }

    fn embed(&self, inputs: &[String], kind: EmbedKind) -> Result<EmbedOutcome, WorkerError> {
        let (model, meta) = self
            .embedder
            .as_ref()
            .ok_or_else(|| WorkerError::BadRequest("no embedder configured on this server".to_string()))?;
        let text_kind = match kind {
            EmbedKind::Document => TextKind::Document,
            EmbedKind::Query => TextKind::Query,
        };
        let vectors = inputs
            .iter()
            .map(|text| model.embed(text, text_kind).map_err(WorkerError::from))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(EmbedOutcome { vectors, model_id: meta.id.clone(), weight_hash: meta.weight_hash.clone() })
    }

    fn predict(&self, inputs: &[String]) -> Result<PredictOutcome, WorkerError> {
        let (model, meta) = self.classifier.as_ref().ok_or_else(|| {
            WorkerError::BadRequest("no classifier configured on this server".to_string())
        })?;
        let per_input = inputs
            .iter()
            .map(|text| {
                let probs = model.predict(text).map_err(WorkerError::from)?;
                let mut ranked: Vec<LabelScore> = model
                    .labels()
                    .iter()
                    .zip(probs)
                    .map(|(label, score)| LabelScore { label: label.clone(), score })
                    .collect();
                ranked.sort_by(|a, b| b.score.total_cmp(&a.score));
                Ok(ranked)
            })
            .collect::<Result<Vec<_>, WorkerError>>()?;
        Ok(PredictOutcome { per_input, model_id: meta.id.clone(), weight_hash: meta.weight_hash.clone() })
    }

    fn classify(&self, inputs: &[String]) -> Result<Vec<ClassifyResult>, WorkerError> {
        let (model, meta) = self.classifier.as_ref().ok_or_else(|| {
            WorkerError::BadRequest("no classifier configured on this server".to_string())
        })?;
        inputs
            .iter()
            .map(|text| {
                let probs = model.predict(text).map_err(WorkerError::from)?;
                let (scores, top) = scores_and_top(model.labels(), &probs);
                Ok(ClassifyResult { scores, top, model_id: meta.id.clone(), weight_hash: meta.weight_hash.clone() })
            })
            .collect()
    }

    fn route(&self, input: &str, routes: &[String]) -> Result<RouteResponse, WorkerError> {
        let (model, meta) = self
            .router
            .as_ref()
            .ok_or_else(|| WorkerError::BadRequest("no router configured on this server".to_string()))?;
        let cosines = model.route_cosines(input, routes).map_err(WorkerError::from)?;
        let routes = routes
            .iter()
            .zip(cosines)
            .map(|(route, cosine)| RouteScore { route: route.clone(), cosine })
            .collect();
        Ok(RouteResponse { model_id: meta.id.clone(), weight_hash: meta.weight_hash.clone(), routes })
    }

    fn cascade(&self, clauses: &[String]) -> Result<CascadeResponse, WorkerError> {
        let (classifier, classifier_meta) = self.classifier.as_ref().ok_or_else(|| {
            WorkerError::BadRequest("cascade requires a classifier configured on this server".to_string())
        })?;
        let (router, router_meta) = self.router.as_ref().ok_or_else(|| {
            WorkerError::BadRequest("cascade requires a router configured on this server".to_string())
        })?;
        if self.cascade_routes.is_empty() {
            return Err(WorkerError::BadRequest(
                "cascade has no routes configured on this server (--cascade-route)".to_string(),
            ));
        }

        // Rank-then-route: entirely the library's own aggregation, called
        // through unmodified — this engine only reshapes the result into
        // the wire contract below.
        let verdict = Cascade::new(classifier, router)
            .run(clauses, &self.cascade_routes, &self.cascade_severe_labels)
            .map_err(WorkerError::from)?;

        let labels = classifier.labels();
        let clause_rows: Vec<CascadeClause> = verdict
            .clauses
            .iter()
            .enumerate()
            .map(|(index, c)| {
                let (severity_scores, top_severity) = scores_and_top(labels, &c.severity_probs);
                CascadeClause { index, clause: c.clause.clone(), severity_scores, top_severity }
            })
            .collect();

        let winner_row = &verdict.clauses[verdict.winner];
        let (winner_scores, _) = scores_and_top(labels, &winner_row.severity_probs);

        Ok(CascadeResponse {
            winner: CascadeWinner { index: verdict.winner, clause: winner_row.clause.clone(), severity_scores: winner_scores },
            lane: CascadeLane {
                route: self.cascade_routes[verdict.winner_lane].clone(),
                cosine: winner_row.lane_cosines[verdict.winner_lane],
            },
            clauses: clause_rows,
            models: vec![
                CascadeModelRef { model_id: classifier_meta.id.clone(), weight_hash: classifier_meta.weight_hash.clone() },
                CascadeModelRef { model_id: router_meta.id.clone(), weight_hash: router_meta.weight_hash.clone() },
            ],
        })
    }
}
