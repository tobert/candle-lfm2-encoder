//! Wire types for the `lfm2d` HTTP API — v1 (see the crate root docs for
//! the full endpoint list and the TEI-compat/audit-trail tension each type
//! resolves).
//!
//! Every map that gets compared as a JSON string in tests
//! ([`ClassifyResult::scores`], [`CascadeWinner::severity_scores`],
//! [`CascadeClause::severity_scores`]) is a [`BTreeMap`], not a
//! `std::collections::HashMap` — `HashMap`'s iteration order is randomized
//! per process (a DoS-hardening default), which would make an
//! exact-JSON-string assertion flaky by construction. A `BTreeMap`
//! serializes its keys in a fixed (alphabetical) order, so the wire output
//! is deterministic run to run.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

// --------------------------------------------------------------- shared

/// `{"inputs": "one string"}` or `{"inputs": ["many", "strings"]}` — the
/// TEI convention both `/embed` and `/predict` accept. Always normalizes to
/// a `Vec<String>` via [`Self::into_vec`]; the response is always an array
/// (batch of 1 for a single-string request), matching the endpoint docs'
/// `→ [[f32,...]]` / `→ [[{label,score},...]]` shape regardless of which
/// input form was used.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Inputs {
    One(String),
    Many(Vec<String>),
}

impl Inputs {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Inputs::One(s) => vec![s],
            Inputs::Many(v) => v,
        }
    }
}

/// Which kind of head a loaded model is — `GET /v1/models`'s `kind` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Embedder,
    Classifier,
    Router,
}

/// One entry in `GET /v1/models`'s response array.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub kind: ModelKind,
    pub weight_hash: String,
    /// Only present for a classifier — an embedder/router has no fixed
    /// label set (the router's "labels" are caller-supplied routes at call
    /// time, not a trained output width; see `src/routing.rs`'s module
    /// docs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<Vec<String>>,
    pub hidden_size: usize,
}

// --------------------------------------------------------------- /embed

/// Which side of `LFM2.5-Embedding-350M`'s asymmetric (E5-style) pair to
/// embed as — see [`candle_lfm2_encoder::TextKind`]'s docs: the same text
/// under `"query: "` vs `"document: "` embeds at cosine ≈ 0.70, so this is
/// not cosmetic. TEI's own `/embed` has no such parameter (single-purpose
/// embedders don't need one), so this is an ADDITIVE optional field a
/// strict TEI client simply never sets — defaulting to `Document`, the more
/// common "index this text" case, keeps wire compatibility for clients that
/// don't know it exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EmbedKind {
    #[default]
    Document,
    Query,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbedRequest {
    pub inputs: Inputs,
    #[serde(default)]
    pub kind: EmbedKind,
}

// -------------------------------------------------------------- /predict

#[derive(Debug, Clone, Deserialize)]
pub struct PredictRequest {
    pub inputs: Inputs,
}

/// One label's probability — `/predict`'s per-input list covers ALL
/// labels (full softmax), sorted by `score` descending.
#[derive(Debug, Clone, Serialize)]
pub struct LabelScore {
    pub label: String,
    pub score: f32,
}

// ----------------------------------------------------------- /v1/classify

/// Unlike `/embed`/`/predict`, `/v1/classify` takes an array only — no
/// bare-string shorthand. This is "our full contract," not a TEI-compat
/// shim, so there is no legacy client to stay bug-compatible with.
#[derive(Debug, Clone, Deserialize)]
pub struct ClassifyRequest {
    pub inputs: Vec<String>,
}

/// One input's full result: the full softmax (`scores`, every label), the
/// argmax (`top`), and the audit pair (`model_id`, `weight_hash`) — see the
/// crate root docs on why `/v1/classify` carries these in-body while
/// `/embed`/`/predict` carry them as response headers instead.
#[derive(Debug, Clone, Serialize)]
pub struct ClassifyResult {
    pub scores: BTreeMap<String, f32>,
    pub top: String,
    pub model_id: String,
    pub weight_hash: String,
}

// -------------------------------------------------------------- /v1/route

/// Singular `input`, not `inputs` — one prompt scored against many routes
/// in one forward pass (matching [`candle_lfm2_encoder::Lfm2SequenceRouter::route_cosines`]).
#[derive(Debug, Clone, Deserialize)]
pub struct RouteRequest {
    pub input: String,
    pub routes: Vec<String>,
}

/// One route's RAW cosine — no softmax probability anywhere on this
/// endpoint. The router's softmax is route-count arithmetic and carries no
/// confidence information (see `src/routing.rs`'s module docs, "the
/// softmax is saturated"); exposing it here would just be a footgun with a
/// plausible-looking field name.
#[derive(Debug, Clone, Serialize)]
pub struct RouteScore {
    pub route: String,
    pub cosine: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct RouteResponse {
    pub model_id: String,
    pub weight_hash: String,
    pub routes: Vec<RouteScore>,
}

// ------------------------------------------------------------ /v1/cascade

/// `{"clauses": [...]}` only — no `routes`, no `severe_labels` in the
/// request body. Both are fixed server-side configuration (`--cascade-route`
/// / `--cascade-severe-label`), not per-request input; see the crate root
/// docs' "cascade configuration is server-side" section for why.
#[derive(Debug, Clone, Deserialize)]
pub struct CascadeRequest {
    pub clauses: Vec<String>,
}

/// The winning (highest-severity) clause. NO `top_severity` field here —
/// unlike [`CascadeClause`] — because "winner" already names which clause
/// won; adding a second "top" concept at this level would just be noise.
#[derive(Debug, Clone, Serialize)]
pub struct CascadeWinner {
    pub index: usize,
    pub clause: String,
    pub severity_scores: BTreeMap<String, f32>,
}

/// The winning clause's argmax route — the cascade's single "route this
/// statement to X" answer.
#[derive(Debug, Clone, Serialize)]
pub struct CascadeLane {
    pub route: String,
    pub cosine: f32,
}

/// One clause's full per-label breakdown, plus which single label scored
/// highest (`top_severity`) — the ranking SUM used to pick the winner
/// (`severity_score` in [`candle_lfm2_encoder::ClauseVerdict`], the caller's
/// configured severe-label set summed) is deliberately not repeated here;
/// see the crate root docs.
#[derive(Debug, Clone, Serialize)]
pub struct CascadeClause {
    pub index: usize,
    pub clause: String,
    pub severity_scores: BTreeMap<String, f32>,
    pub top_severity: String,
}

/// One model's audit identity — `/v1/cascade`'s `models` array carries one
/// entry per head involved (classifier, router).
#[derive(Debug, Clone, Serialize)]
pub struct CascadeModelRef {
    pub model_id: String,
    pub weight_hash: String,
}

/// Moderations-FLAVORED but with NO `flagged` boolean and NO threshold
/// anywhere — see the crate root docs and `src/cascade.rs`'s module docs on
/// why: no global severity cutoff exists by measurement (benign max 0.3415
/// vs data-critical min 0.3440), so a caller ranks clauses WITHIN a
/// statement rather than gating on an absolute number this API would
/// otherwise seem to bless.
#[derive(Debug, Clone, Serialize)]
pub struct CascadeResponse {
    pub winner: CascadeWinner,
    pub lane: CascadeLane,
    pub clauses: Vec<CascadeClause>,
    pub models: Vec<CascadeModelRef>,
}

// ----------------------------------------------------------------- errors

/// `{"error": {"message", "type"}}` — every non-2xx response body.
#[derive(Debug, Clone, Serialize)]
pub struct ApiError {
    pub error: ApiErrorBody,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
}

impl ApiError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self { error: ApiErrorBody { message: message.into(), kind: "bad_request".to_string() } }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self { error: ApiErrorBody { message: message.into(), kind: "internal".to_string() } }
    }
}
