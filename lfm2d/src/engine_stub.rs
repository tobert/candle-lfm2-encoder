//! [`crate::worker::InferenceEngine`] test double — no candle, no weights,
//! deterministic canned output. Exists so `tests/router_stub.rs` can drive
//! the real `axum` router (real [`crate::worker::WorkerHandle`], real
//! worker thread, real channel plumbing) and assert HTTP-level behavior
//! (status codes, JSON shape, header passthrough, error mapping) in
//! milliseconds, independent of whether any checkpoint is present on disk.
//!
//! Configuration mirrors what `--embedder-dir`/`--classifier-dir`/
//! `--router-dir` would produce in production: each head is independently
//! present-or-absent, so tests can exercise "endpoint called with that head
//! unconfigured" (→ 400) the same way a misconfigured real deployment
//! would.

use crate::types::{
    CascadeClause, CascadeLane, CascadeModelRef, CascadeResponse, CascadeWinner, ClassifyResult,
    EmbedKind, LabelScore, ModelInfo, ModelKind, RouteResponse, RouteScore,
};
use crate::worker::{EmbedOutcome, InferenceEngine, PredictOutcome, WorkerError};

/// A fake loaded model's identity — same shape as what a real load would
/// record, without ever touching a file.
#[derive(Debug, Clone)]
pub struct StubModel {
    pub id: String,
    pub weight_hash: String,
    pub hidden_size: usize,
}

impl StubModel {
    pub fn new(id: impl Into<String>) -> Self {
        // A recognizable, obviously-fake but correctly-shaped (64 hex
        // chars) weight hash, distinct per model id so tests can tell
        // models apart by hash alone.
        let id = id.into();
        let weight_hash = format!("{:0<64}", format!("stub-{id}-hash"))
            .chars()
            .map(|c| if c.is_ascii_hexdigit() { c.to_ascii_lowercase() } else { '0' })
            .collect();
        Self { id, weight_hash, hidden_size: 8 }
    }
}

#[derive(Debug, Clone, Default)]
pub struct StubEngine {
    pub embedder: Option<StubModel>,
    pub classifier: Option<StubModel>,
    /// Order is the label id order — `predict`/`classify` output follows
    /// it, same contract as the real
    /// [`candle_lfm2_encoder::Lfm2SequenceClassifier::labels`].
    pub classifier_labels: Vec<String>,
    pub router: Option<StubModel>,
    /// Server-side cascade config — see the crate root docs on why this is
    /// not per-request.
    pub cascade_routes: Vec<String>,
    pub cascade_severe_labels: Vec<String>,
    /// When set, every inference call fails with this message as a
    /// [`WorkerError::Internal`] — the test double's way of simulating "a
    /// loaded model's forward pass blew up," for asserting 500 mapping and
    /// the error JSON shape without needing a real model that can be made
    /// to fail on cue.
    pub fail_with: Option<String>,
    /// When set, every inference call panics with this message — simulating
    /// the worker thread itself dying (not a clean `Err`), so tests can
    /// assert both halves of the channel failure contract: the in-flight
    /// request's dropped reply channel and later requests' dead sender.
    pub panic_with: Option<String>,
    /// When set, every inference call blocks (`std::thread::sleep`, since
    /// this runs on the worker OS thread) for this long before doing
    /// anything else — including before `fail_with`/`panic_with` take
    /// effect. Exists so a test can hold a request "in flight" on purpose,
    /// e.g. `tests/graceful_shutdown.rs` proving an in-flight request still
    /// completes with 200 after shutdown has been requested.
    pub delay: Option<std::time::Duration>,
}

impl StubEngine {
    /// The everything-loaded configuration most router tests want: one of
    /// each head, three classifier labels (`destructive`/`informative`/
    /// `mutating` — `kube_ordinal_v6`'s real label set), and cascade wired
    /// up over two routes.
    pub fn fully_configured() -> Self {
        Self {
            embedder: Some(StubModel::new("stub-embedder")),
            classifier: Some(StubModel::new("stub-classifier")),
            classifier_labels: vec!["destructive".into(), "informative".into(), "mutating".into()],
            router: Some(StubModel::new("stub-router")),
            cascade_routes: vec!["shell".into(), "k8s".into()],
            cascade_severe_labels: vec!["mutating".into(), "destructive".into()],
            fail_with: None,
            panic_with: None,
            delay: None,
        }
    }

    fn check_fail(&self) -> Result<(), WorkerError> {
        if let Some(d) = self.delay {
            std::thread::sleep(d);
        }
        if let Some(msg) = &self.panic_with {
            panic!("{msg}");
        }
        if let Some(msg) = &self.fail_with {
            return Err(WorkerError::Internal(msg.clone()));
        }
        Ok(())
    }

    /// Deterministic, obviously-fake "scores": longer text skews toward the
    /// last label, shorter toward the first — enough spread that tests can
    /// assert a specific label wins without needing real inference.
    fn fake_scores(&self, text: &str) -> Vec<f32> {
        let n = self.classifier_labels.len().max(1);
        let bucket = text.len() % n;
        let mut raw = vec![0.1f32; n];
        raw[bucket] = 1.0;
        let sum: f32 = raw.iter().sum();
        raw.iter().map(|v| v / sum).collect()
    }
}

impl InferenceEngine for StubEngine {
    fn list_models(&self) -> Vec<ModelInfo> {
        let mut out = Vec::new();
        if let Some(m) = &self.embedder {
            out.push(ModelInfo {
                id: m.id.clone(),
                kind: ModelKind::Embedder,
                weight_hash: m.weight_hash.clone(),
                labels: None,
                hidden_size: m.hidden_size,
            });
        }
        if let Some(m) = &self.classifier {
            out.push(ModelInfo {
                id: m.id.clone(),
                kind: ModelKind::Classifier,
                weight_hash: m.weight_hash.clone(),
                labels: Some(self.classifier_labels.clone()),
                hidden_size: m.hidden_size,
            });
        }
        if let Some(m) = &self.router {
            out.push(ModelInfo {
                id: m.id.clone(),
                kind: ModelKind::Router,
                weight_hash: m.weight_hash.clone(),
                labels: None,
                hidden_size: m.hidden_size,
            });
        }
        out
    }

    fn embed(&self, inputs: &[String], kind: EmbedKind) -> Result<EmbedOutcome, WorkerError> {
        self.check_fail()?;
        let model = self
            .embedder
            .as_ref()
            .ok_or_else(|| WorkerError::BadRequest("no embedder configured on this server".to_string()))?;
        let kind_component = match kind {
            EmbedKind::Document => 0.0,
            EmbedKind::Query => 1.0,
        };
        let vectors = inputs
            .iter()
            .map(|s| {
                let mut v = vec![0f32; model.hidden_size];
                v[0] = s.len() as f32;
                v[1] = kind_component;
                v
            })
            .collect();
        Ok(EmbedOutcome { vectors, model_id: model.id.clone(), weight_hash: model.weight_hash.clone() })
    }

    fn predict(&self, inputs: &[String]) -> Result<PredictOutcome, WorkerError> {
        self.check_fail()?;
        let model = self.classifier.as_ref().ok_or_else(|| {
            WorkerError::BadRequest("no classifier configured on this server".to_string())
        })?;
        let per_input = inputs
            .iter()
            .map(|text| {
                let scores = self.fake_scores(text);
                let mut ranked: Vec<LabelScore> = self
                    .classifier_labels
                    .iter()
                    .zip(scores)
                    .map(|(label, score)| LabelScore { label: label.clone(), score })
                    .collect();
                ranked.sort_by(|a, b| b.score.total_cmp(&a.score));
                ranked
            })
            .collect();
        Ok(PredictOutcome { per_input, model_id: model.id.clone(), weight_hash: model.weight_hash.clone() })
    }

    fn classify(&self, inputs: &[String]) -> Result<Vec<ClassifyResult>, WorkerError> {
        self.check_fail()?;
        let model = self.classifier.as_ref().ok_or_else(|| {
            WorkerError::BadRequest("no classifier configured on this server".to_string())
        })?;
        inputs
            .iter()
            .map(|text| {
                let scores = self.fake_scores(text);
                let mut map = std::collections::BTreeMap::new();
                let mut top = (String::new(), f32::NEG_INFINITY);
                for (label, score) in self.classifier_labels.iter().zip(scores) {
                    if score > top.1 {
                        top = (label.clone(), score);
                    }
                    map.insert(label.clone(), score);
                }
                Ok(ClassifyResult {
                    scores: map,
                    top: top.0,
                    model_id: model.id.clone(),
                    weight_hash: model.weight_hash.clone(),
                })
            })
            .collect()
    }

    fn route(&self, input: &str, routes: &[String]) -> Result<RouteResponse, WorkerError> {
        self.check_fail()?;
        let model = self
            .router
            .as_ref()
            .ok_or_else(|| WorkerError::BadRequest("no router configured on this server".to_string()))?;
        let base = (input.len() % 7) as f32 / 7.0;
        let routes = routes
            .iter()
            .enumerate()
            .map(|(i, r)| RouteScore { route: r.clone(), cosine: (base - i as f32 * 0.2).clamp(-1.0, 1.0) })
            .collect();
        Ok(RouteResponse { model_id: model.id.clone(), weight_hash: model.weight_hash.clone(), routes })
    }

    fn cascade(&self, clauses: &[String]) -> Result<CascadeResponse, WorkerError> {
        self.check_fail()?;
        let classifier = self.classifier.as_ref().ok_or_else(|| {
            WorkerError::BadRequest("cascade requires a classifier configured on this server".to_string())
        })?;
        let router = self.router.as_ref().ok_or_else(|| {
            WorkerError::BadRequest("cascade requires a router configured on this server".to_string())
        })?;
        if self.cascade_routes.is_empty() {
            return Err(WorkerError::BadRequest(
                "cascade has no routes configured on this server (--cascade-route)".to_string(),
            ));
        }

        // Rank by the same fake_scores' severe-label sum the real engine
        // would compute over caller-configured severe labels.
        let severe_idx: Vec<usize> = self
            .cascade_severe_labels
            .iter()
            .filter_map(|l| self.classifier_labels.iter().position(|c| c == l))
            .collect();

        let mut clause_rows = Vec::with_capacity(clauses.len());
        let mut best: Option<(usize, f32)> = None;
        for (i, clause) in clauses.iter().enumerate() {
            let scores = self.fake_scores(clause);
            let severity_score: f32 = severe_idx.iter().map(|&j| scores[j]).sum();
            if best.is_none_or(|(_, s)| severity_score > s) {
                best = Some((i, severity_score));
            }
            let mut map = std::collections::BTreeMap::new();
            let mut top = (String::new(), f32::NEG_INFINITY);
            for (label, score) in self.classifier_labels.iter().zip(scores) {
                if score > top.1 {
                    top = (label.clone(), score);
                }
                map.insert(label.clone(), score);
            }
            clause_rows.push((i, clause.clone(), map, top.0));
        }

        let winner_idx = best.map(|(i, _)| i).unwrap_or(0);
        let winner_row = &clause_rows[winner_idx];
        let lane_cosines: Vec<f32> = self
            .cascade_routes
            .iter()
            .enumerate()
            .map(|(i, _)| ((winner_row.1.len() % 7) as f32 / 7.0 - i as f32 * 0.2).clamp(-1.0, 1.0))
            .collect();
        let winner_lane = (0..lane_cosines.len())
            .max_by(|&a, &b| lane_cosines[a].total_cmp(&lane_cosines[b]))
            .expect("cascade_routes checked non-empty above");

        Ok(CascadeResponse {
            winner: CascadeWinner {
                index: winner_idx,
                clause: winner_row.1.clone(),
                severity_scores: winner_row.2.clone(),
            },
            lane: CascadeLane {
                route: self.cascade_routes[winner_lane].clone(),
                cosine: lane_cosines[winner_lane],
            },
            clauses: clause_rows
                .into_iter()
                .map(|(index, clause, severity_scores, top_severity)| CascadeClause {
                    index,
                    clause,
                    severity_scores,
                    top_severity,
                })
                .collect(),
            models: vec![
                CascadeModelRef { model_id: classifier.id.clone(), weight_hash: classifier.weight_hash.clone() },
                CascadeModelRef { model_id: router.id.clone(), weight_hash: router.weight_hash.clone() },
            ],
        })
    }
}
