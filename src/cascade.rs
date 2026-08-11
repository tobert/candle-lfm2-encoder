//! The rank-then-route composite: v6 ordinal classifier ranks, router
//! routes — the cascade `training/router/findings_clause_decomposition.md`
//! measured 5/5 against 3/5 whole-statement top-1 and 0/5 max-cosine union.
//!
//! # Why two heads, and why neither alone
//!
//! [`crate::routing::Lfm2SequenceRouter::route_clauses`] answers a
//! **domain** question ("which specialist should look at this clause?") and
//! answers it well per clause. It cannot answer a **severity** question: a
//! clause carrying no intent at all still has to score against every lane,
//! and the findings doc measured it winning the union outright over a
//! genuinely dangerous co-clause (`cd vendor/legacy-plugin` at k8s +0.978
//! beating `rm -rf .` at shell +0.702 in the same statement — the
//! `a/b`-slug navigation trap). Confidence on the router's head is not
//! severity; it is route-count arithmetic (see [`crate::routing`]'s module
//! docs, finding 2), and per clause that is the DOMINANT failure mode.
//!
//! The v6 ordinal classifier (`kube_ordinal_v6`, an
//! `Lfm2BidirForSequenceClassification` fine-tune — see
//! [`crate::sequence_classification`]) answers severity and has no domain
//! opinion at all; it cannot tell a `git clone` from a `kubectl delete` from
//! an `rm -rf`, it can only rank how much each clause LOOKS like it destroys
//! or mutates state.
//!
//! Neither model can do the other's job. This module composes them: rank
//! every clause by severity, then route only the winner. On the findings
//! doc's five decomposable probes this resolved all three guard_evasion
//! "laundering" cases (a delete buried behind a benign opener) correctly,
//! where the router alone got 3/5 and the router's own max-cosine union got
//! 0/5.
//!
//! # Ranking and gating are different thresholds
//!
//! [`ClauseVerdict::severity_score`] is a RANKING signal — "which clause,
//! among these, matters most" — not a gating one. Two of the findings doc's
//! five correct selections are, on their own, unalarming numbers: `sudo
//! bash` at severity 0.207 and `npm install` at 0.132. A cascade that gated
//! on absolute severity would let both through; one that ranks clauses
//! within a statement gets the right answer. This module exposes ranked
//! severities and raw router cosines and deliberately bakes in NO
//! accept/reject decision anywhere — that threshold, and whether ranking and
//! gating even share one (the findings doc says they should not), is a
//! caller/gate decision, not this crate's.
//!
//! # No splitter, same refusal as [`crate::routing`]
//!
//! `clauses` must come from a real parser — kaish `Plan.commands[]` is what
//! this was built and measured against. This module ships nothing that
//! turns text into clauses, for the same reason `route_clauses` doesn't: a
//! regex on `&&`/`|` would disagree with the parser that actually decides
//! what runs, and disagreeing about clause boundaries is exactly the
//! silent-wrong-answer shape this crate refuses everywhere else. Feed it
//! real clauses, or feed it the whole statement as one clause.
//!
//! # Provenance
//!
//! Every number in [`CascadeVerdict`] traces back to the clause that
//! produced it: [`ClauseVerdict::clause`] on each per-clause row,
//! [`CascadeVerdict::winner`] naming the ranked clause, and
//! [`CascadeVerdict::firing_clause`] naming which clause supplied each
//! union-lane score — the same provenance contract
//! [`crate::routing::ClauseRouting::firing_clause`] gives the router alone.
//! A gate that fires must be able to quote which clause, and which number,
//! made it fire.

use crate::error::{Error, Result};
use crate::routing::Lfm2SequenceRouter;
use crate::sequence_classification::Lfm2SequenceClassifier;

/// One clause's full result from both heads — the atomic unit [`aggregate`]
/// operates over. Provenance-complete: everything a gate needs to quote why
/// it fired about THIS clause lives here, keyed by the clause's own text.
#[derive(Debug, Clone, PartialEq)]
pub struct ClauseVerdict {
    /// The clause's own text, kept so a caller/gate can quote it directly
    /// without re-indexing into the original clause list.
    pub clause: String,
    /// Full softmax over the classifier's labels, `[num_labels]`, in
    /// [`Lfm2SequenceClassifier::labels`] order.
    pub severity_probs: Vec<f32>,
    /// `sum(severity_probs[i] for i in the caller's severe label set)` — the
    /// RANKING signal (see the module docs' "Ranking and gating" section).
    /// NOT bounded away from small values for a correct selection: `sudo
    /// bash` at 0.207 and `npm install` at 0.132 are both correct winners in
    /// the findings doc's table.
    pub severity_score: f32,
    /// Every lane's raw cosine against the router, `[n_routes]`, in the
    /// caller's `routes` order — see
    /// [`crate::routing::Lfm2SequenceRouter::route_cosines`] for why raw
    /// cosine and not the (route-count-saturated) softmax.
    pub lane_cosines: Vec<f32>,
}

/// One statement's cascade result: every clause's own numbers
/// ([`Self::clauses`]), plus the aggregation [`aggregate`] derives from
/// them.
#[derive(Debug, Clone, PartialEq)]
pub struct CascadeVerdict {
    /// Per-clause results, input order preserved — every other field below
    /// is an index into this.
    pub clauses: Vec<ClauseVerdict>,
    /// Clause indices ordered by [`ClauseVerdict::severity_score`]
    /// descending. Ties (equal `severity_score`) keep the EARLIER clause
    /// first — see [`aggregate`]'s docs for why and how that's guaranteed.
    pub ranked: Vec<usize>,
    /// The max-severity clause's index — always `ranked[0]`.
    pub winner: usize,
    /// The winning clause's argmax lane, an index into the caller's
    /// `routes`. This is the cascade's answer to "which specialist should
    /// see this statement": the DOMAIN of the clause severity picked, not
    /// the domain of whichever clause happened to score highest on any one
    /// lane.
    pub winner_lane: usize,
    /// Per-lane max cosine across ALL clauses (not only the winner) — the
    /// recall channel for fan-out to always-on specialists. Same shape and
    /// rationale as [`crate::routing::ClauseRouting::union_cosines`]: a lane
    /// fires if ANY clause fires it, independent of which clause won the
    /// severity ranking. The findings doc measured this at 5/5 expected-lane
    /// recall even where the union's ARGMAX was wrong every time — recall
    /// and precision are different questions, and this field answers only
    /// the first.
    pub union_cosines: Vec<f32>,
}

impl CascadeVerdict {
    /// Which clause supplied lane `lane`'s union score — the provenance a
    /// gate quotes when it explains itself. Mirrors
    /// [`crate::routing::ClauseRouting::firing_clause`] exactly; kept
    /// separate (not shared code) because the two types have unrelated
    /// underlying storage (`clauses: Vec<ClauseVerdict>` here vs
    /// `clause_cosines: Vec<Vec<f32>>` there).
    ///
    /// Returns `None` only if `self.clauses` is empty (which [`aggregate`]
    /// never produces) — an out-of-range `lane` panics, same as indexing
    /// `union_cosines[lane]` directly would.
    pub fn firing_clause(&self, lane: usize) -> Option<usize> {
        (0..self.clauses.len()).max_by(|&a, &b| {
            self.clauses[a].lane_cosines[lane].total_cmp(&self.clauses[b].lane_cosines[lane])
        })
    }
}

/// Resolve caller-supplied severe-label names to indices into `id2label`
/// (the same order [`Lfm2SequenceClassifier::labels`] and
/// [`Lfm2SequenceClassifier::predict`] use). Loud error naming the bad
/// label and the checkpoint's actual labels if any entry doesn't match — a
/// typo here would otherwise silently zero out every clause's severity
/// score and rank nothing, which is exactly the silent-wrong-answer shape
/// this crate refuses.
///
/// Pure and testable without a loaded model — see `tests/cascade.rs`'s
/// contract tests. The set is intentionally never hard-coded to any one
/// checkpoint's labels: `{"mutating", "destructive"}` is `kube_ordinal_v6`'s
/// severe set per the findings doc, not a fact about this crate.
pub fn resolve_severe_labels<L: AsRef<str>>(
    id2label: &[String],
    severe_labels: &[L],
) -> Result<Vec<usize>> {
    if severe_labels.is_empty() {
        return Err(Error::InvalidInput(
            "cascade needs at least one severe label (e.g. {\"mutating\", \"destructive\"} for \
             kube_ordinal_v6) — an empty set would make every clause's severity score exactly \
             zero and rank nothing"
                .to_string(),
        ));
    }
    severe_labels
        .iter()
        .map(|l| {
            let name = l.as_ref();
            id2label.iter().position(|c| c == name).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "severe label {name:?} is not one of this classifier's labels {id2label:?}"
                ))
            })
        })
        .collect()
}

/// Per-label weights for the expected-ordinal-rank severity score: a
/// non-severe label weighs `0.0`, and the `i`th severe label (in the
/// CALLER's order) weighs `i + 1`.
///
/// # Why the caller's order, and why that matters
///
/// The rank of a label cannot be read off the checkpoint. `kube_ordinal_v6`
/// stores its labels as `["destructive", "informative", "mutating"]` — array
/// position is an artifact of training, not a severity ordering. So the
/// ordering has to come from the caller, and `severe_labels` is therefore
/// **ascending severity, least severe first**: `["situation-normal",
/// "data-critical"]`, not the reverse.
///
/// Passing them in the wrong order silently inverts the ranking, which is
/// precisely the kind of quiet wrong answer this crate refuses elsewhere —
/// and it cannot be detected from the label names alone. Two things
/// mitigate it: duplicates are rejected here (a repeated label has no
/// unambiguous rank), and callers SHOULD echo the resolved ranking at
/// startup so a mistake is visible immediately rather than at the first
/// surprising verdict. Long-lived callers that load a checkpoint once —
/// `lfm2d` in particular — are the ones this matters for; a one-shot CLI
/// run shows its own arguments.
///
/// # Why not just sum the severe probabilities
///
/// That was the original aggregation and it is **not monotone in severity**.
/// The severe set spans two rungs of an ordinal scale, so probability mass
/// moving from the middle rung to the top rung — a clause getting *more*
/// severe — leaves the sum unchanged at best, and lowers it whenever the
/// bottom rung picks up any mass. Observed live on the deployed v8 daemon
/// 2026-08-11: `psql -f drop_all.sql` (data-critical 0.600) scored 0.8299
/// and ranked BELOW `psql -f 001_init.sql` (data-critical 0.330) at 0.9298.
/// The model was right; the aggregation discarded the distinction.
///
/// Measured against the pinned probe set before switching (see
/// `training/cascade/rescore_aggregation.py`, which recomputes from the
/// recorded eval artifacts): winner selection went 33→35/36 on v6,
/// unchanged 33/36 on v7, and 34→35/36 on v8. Never worse on any checkpoint.
///
/// # Consequence for callers
///
/// [`ClauseVerdict::severity_score`] is no longer bounded by `1.0` — its
/// range is `0.0..=n` for `n` severe labels (so `0.0..=2.0` for the usual
/// three-rung vocabulary). It remains a RANKING signal with no meaningful
/// absolute cutoff, exactly as before; nothing in this crate thresholds it.
pub fn severity_rank_weights<L: AsRef<str>>(
    id2label: &[String],
    severe_labels: &[L],
) -> Result<Vec<f32>> {
    let severe_ids = resolve_severe_labels(id2label, severe_labels)?;

    let mut weights = vec![0.0f32; id2label.len()];
    for (rank, &id) in severe_ids.iter().enumerate() {
        if weights[id] != 0.0 {
            return Err(Error::InvalidInput(format!(
                "severe label {:?} appears more than once — its ordinal rank would be \
                 ambiguous; list each label once, in ascending severity order",
                id2label[id]
            )));
        }
        weights[id] = (rank + 1) as f32;
    }
    Ok(weights)
}

/// The expected ordinal rank: `sum(weight[i] * probs[i])`.
///
/// Kept separate from [`Cascade::run`] so the aggregation is testable
/// without a model, and so the monotonicity property has something to
/// assert against directly.
pub fn severity_score_from_probs(probs: &[f32], rank_weights: &[f32]) -> f32 {
    probs.iter().zip(rank_weights).map(|(p, w)| p * w).sum()
}

/// Pure aggregation: per-clause results → one statement verdict. No model
/// calls happen here — this is what makes the composite rule unit-testable
/// without weights (see `tests/cascade.rs`'s contract tests, and
/// [`Cascade::run`], the only caller that feeds it real model output).
///
/// # Ranking and its tie rule
///
/// Clauses are ordered by [`ClauseVerdict::severity_score`] descending.
/// **On an exact tie, the earlier clause (lower input index) wins** — a
/// deliberate, documented choice, not an accident of a particular sort
/// implementation: it is produced by a STABLE sort over a descending
/// comparator, which by definition leaves elements that compare equal in
/// their original relative order. See
/// `tests/cascade.rs::ties_break_toward_the_earlier_clause` for the
/// contract test pinning this.
///
/// # Errors
/// - `clauses` is empty — an empty clause list has no winner to report, and
///   reporting one would say "no risk" about a statement nobody scored.
/// - any clause's `lane_cosines` is empty, or clauses disagree on
///   `lane_cosines` length — both mean the clauses weren't scored against
///   one consistent route set, which this function cannot safely rank
///   lanes over.
pub fn aggregate(clauses: Vec<ClauseVerdict>) -> Result<CascadeVerdict> {
    if clauses.is_empty() {
        return Err(Error::InvalidInput(
            "cascade aggregation needs at least one clause: an empty clause list has no \
             winner to report, and reporting one would say 'no risk' about a statement that \
             was never scored"
                .to_string(),
        ));
    }

    let n_routes = clauses[0].lane_cosines.len();
    if n_routes == 0 {
        return Err(Error::InvalidInput(
            "cascade aggregation needs at least one route: a clause with zero lane cosines \
             has no domain to report"
                .to_string(),
        ));
    }
    for (i, c) in clauses.iter().enumerate() {
        if c.lane_cosines.len() != n_routes {
            return Err(Error::InvalidInput(format!(
                "clause {i} ({:?}) has {} lane cosines, but clause 0 has {n_routes} — every \
                 clause must be scored against the same route set for a union to mean anything",
                c.clause,
                c.lane_cosines.len()
            )));
        }
    }

    // Stable sort, descending: equal severity_score entries keep their
    // original relative order, which is exactly the documented tie rule.
    let mut ranked: Vec<usize> = (0..clauses.len()).collect();
    ranked.sort_by(|&a, &b| clauses[b].severity_score.total_cmp(&clauses[a].severity_score));
    let winner = ranked[0];
    let winner_lane = argmax(&clauses[winner].lane_cosines);

    let mut union_cosines = vec![f32::NEG_INFINITY; n_routes];
    for c in &clauses {
        for (lane, &cos) in c.lane_cosines.iter().enumerate() {
            union_cosines[lane] = union_cosines[lane].max(cos);
        }
    }

    Ok(CascadeVerdict {
        clauses,
        ranked,
        winner,
        winner_lane,
        union_cosines,
    })
}

/// Index of the maximum value, by [`f32::total_cmp`] (NaN-safe, matching
/// every other argmax in this crate — see `routing.rs`'s
/// `firing_clause`/`lanes_above`). Callers guarantee `v` is non-empty;
/// [`aggregate`] checks that before this is ever called.
fn argmax(v: &[f32]) -> usize {
    (0..v.len())
        .max_by(|&a, &b| v[a].total_cmp(&v[b]))
        .expect("caller guarantees a non-empty slice")
}

/// The rank-then-route composite over two already-loaded heads.
///
/// Borrows both — never owns or (re)loads either. The two checkpoints have
/// genuinely different trunks: `kube_ordinal_v6` is its own fine-tune, not
/// merely a different head over a shared base, so there is no trunk to
/// share between it and the Prompt-Router, and no reason for this type to
/// touch loading at all. Callers load both once, however they see fit
/// (`from_dir`, or `Lfm2Trunk::load_shared` + `from_trunk` if a checkpoint
/// happens to share a trunk with something else in the caller's process),
/// and hand this type references. This type has no opinion on checkpoint
/// paths, filesystem layout, or which checkpoint a caller is running — see
/// the module docs' "no splitter" section for the same reasoning applied to
/// clause boundaries.
#[derive(Debug)]
pub struct Cascade<'a> {
    classifier: &'a Lfm2SequenceClassifier,
    router: &'a Lfm2SequenceRouter,
}

impl<'a> Cascade<'a> {
    /// Borrow a loaded classifier and router. Both must outlive `self`.
    pub fn new(classifier: &'a Lfm2SequenceClassifier, router: &'a Lfm2SequenceRouter) -> Self {
        Self { classifier, router }
    }

    /// Run both heads over every clause — one classifier forward pass and
    /// one router forward pass PER CLAUSE, `clauses.len() * 2` passes total
    /// — then rank-then-route via [`aggregate`].
    ///
    /// `severe_labels` names which of the classifier's OWN labels count as
    /// "severe" for ranking (`{"mutating", "destructive"}` for
    /// `kube_ordinal_v6`, per the findings doc) — see
    /// [`resolve_severe_labels`]. This type hard-codes no checkpoint's
    /// label set; the caller names it every call.
    ///
    /// # Errors
    /// - empty `clauses` — see [`aggregate`]; checked FIRST, before either
    ///   head runs, so a caller's empty-input bug costs zero forward passes.
    /// - empty `severe_labels`, or an entry not in the classifier's labels —
    ///   see [`resolve_severe_labels`]; also checked before any clause is
    ///   scored.
    /// - empty `routes` — checked before any forward pass (the router would
    ///   also refuse, but only after a classifier pass had already run).
    /// - a non-finite severity score — a clause whose severity comes back
    ///   NaN/inf would otherwise WIN the ranking (`total_cmp` sorts NaN above
    ///   everything), turning a numerics bug into a confident verdict. The
    ///   router refuses non-finite intermediates (`require_finite` in
    ///   `routing.rs`); this is the same house rule on the severity side.
    /// - any per-clause model call failing — propagated from
    ///   [`Lfm2SequenceClassifier::predict`] /
    ///   [`Lfm2SequenceRouter::route_cosines`].
    pub fn run<C, S, L>(
        &self,
        clauses: &[C],
        routes: &[S],
        severe_labels: &[L],
    ) -> Result<CascadeVerdict>
    where
        C: AsRef<str>,
        S: AsRef<str>,
        L: AsRef<str>,
    {
        if clauses.is_empty() {
            return Err(Error::InvalidInput(
                "cascade needs at least one clause: an empty clause list has no winner to \
                 report, and reporting one would say 'no risk' about a statement that was \
                 never scored"
                    .to_string(),
            ));
        }
        if routes.is_empty() {
            return Err(Error::InvalidInput(
                "cascade needs at least one route: with no lanes there is nothing to \
                 route the winning clause to"
                    .to_string(),
            ));
        }
        // Expected ordinal rank, NOT a sum over the severe labels — the sum
        // is not monotone in severity. See `severity_rank_weights`.
        let rank_weights = severity_rank_weights(self.classifier.labels(), severe_labels)?;

        let mut verdicts = Vec::with_capacity(clauses.len());
        for clause in clauses {
            let text = clause.as_ref();
            let severity_probs = self.classifier.predict(text)?;
            let severity_score: f32 = severity_score_from_probs(&severity_probs, &rank_weights);
            if !severity_score.is_finite() {
                return Err(Error::InvalidInput(format!(
                    "clause {text:?} produced a non-finite severity score \
                     ({severity_score}) from probabilities {severity_probs:?}; refusing \
                     to rank it — NaN sorts above every finite severity and would win"
                )));
            }
            let lane_cosines = self.router.route_cosines(text, routes)?;
            verdicts.push(ClauseVerdict {
                clause: text.to_string(),
                severity_probs,
                severity_score,
                lane_cosines,
            });
        }
        aggregate(verdicts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verdict(clause: &str, severity_score: f32, lane_cosines: Vec<f32>) -> ClauseVerdict {
        ClauseVerdict {
            clause: clause.to_string(),
            severity_probs: Vec::new(), // unused by aggregate/argmax
            severity_score,
            lane_cosines,
        }
    }

    #[test]
    fn empty_clause_list_is_refused_not_treated_as_no_intent() {
        let err = aggregate(Vec::new()).expect_err("empty clause list must be refused");
        let msg = err.to_string();
        assert!(msg.contains("clause"), "{msg}");
    }

    #[test]
    fn max_severity_selects_the_winner_regardless_of_input_position() {
        let clauses = vec![
            verdict("ls", 0.01, vec![0.9, -0.5]),
            verdict("rm -rf /", 0.98, vec![-0.2, 0.9]),
            verdict("echo hi", 0.02, vec![0.1, -0.1]),
        ];
        let v = aggregate(clauses).expect("non-empty");
        assert_eq!(v.winner, 1, "the highest-severity clause must win");
        assert_eq!(v.ranked[0], 1);
        assert_eq!(v.clauses[v.winner].clause, "rm -rf /");
    }

    /// Documented tie rule: on an EXACT tie, the earlier (lower-index)
    /// clause wins, because `aggregate` sorts with a stable sort and a
    /// stable sort preserves the input order of elements that compare
    /// equal. This is a chosen behavior, not an accident — see the module
    /// docs on `aggregate`.
    #[test]
    fn ties_break_toward_the_earlier_clause() {
        let clauses = vec![
            verdict("DROP TABLE orders", 0.9, vec![0.5]),
            verdict("DROP TABLE customers", 0.9, vec![0.5]),
        ];
        let v = aggregate(clauses).expect("non-empty");
        assert_eq!(
            v.winner, 0,
            "an exact severity tie must resolve to the earlier clause"
        );
        assert_eq!(v.ranked, vec![0, 1]);
    }

    #[test]
    fn union_is_the_per_lane_max_over_clauses() {
        let clauses = vec![
            verdict("a", 0.1, vec![0.9, -0.2, 0.0]),
            verdict("b", 0.9, vec![-0.5, 0.8, -0.9]),
            verdict("c", 0.3, vec![0.1, 0.1, 0.95]),
        ];
        let v = aggregate(clauses).expect("non-empty");
        assert_eq!(v.union_cosines, vec![0.9, 0.8, 0.95]);
    }

    /// `firing_clause` must name whichever clause actually supplied a given
    /// lane's union score — the provenance a gate quotes when it explains
    /// itself. A wrong answer here would attribute a decision to an
    /// innocent clause.
    #[test]
    fn firing_clause_names_the_clause_that_supplied_the_union_score() {
        let clauses = vec![
            verdict("cd some/dir", 0.02, vec![0.97, -0.3]),
            verdict("rm -rf .", 0.95, vec![-0.1, 0.7]),
        ];
        let v = aggregate(clauses).expect("non-empty");

        // Lane 0 (k8s-shaped): the navigation clause (index 0) supplied it.
        assert_eq!(v.firing_clause(0), Some(0));
        // Lane 1 (shell-shaped): the delete (index 1) supplied it, even
        // though it lost the severity ranking's argmax on lane 0.
        assert_eq!(v.firing_clause(1), Some(1));
        // And the winner is still the delete, by severity — domain
        // (firing_clause/union) and ranking (winner) are independent
        // questions, which is the whole point of keeping both exposed.
        assert_eq!(v.winner, 1);
    }

    #[test]
    fn mismatched_lane_cosine_lengths_are_refused() {
        let clauses = vec![verdict("a", 0.1, vec![0.1, 0.2]), verdict("b", 0.2, vec![0.1])];
        let err = aggregate(clauses).expect_err("mismatched route-set lengths must be refused");
        let msg = err.to_string();
        assert!(msg.contains("route set") || msg.contains("lane cosines"), "{msg}");
    }

    #[test]
    fn zero_length_lane_cosines_are_refused() {
        let clauses = vec![verdict("a", 0.1, Vec::new())];
        let err = aggregate(clauses).expect_err("zero routes must be refused");
        assert!(err.to_string().contains("route"));
    }

    #[test]
    fn resolve_severe_labels_rejects_empty_set() {
        let labels = vec!["informative".to_string(), "mutating".to_string(), "destructive".to_string()];
        let empty: [&str; 0] = [];
        let err = resolve_severe_labels(&labels, &empty).expect_err("empty severe set must be refused");
        assert!(err.to_string().contains("severe label"));
    }

    #[test]
    fn resolve_severe_labels_rejects_an_unknown_label_and_names_it() {
        let labels = vec!["informative".to_string(), "mutating".to_string(), "destructive".to_string()];
        let err = resolve_severe_labels(&labels, &["mutating", "catastrophic"])
            .expect_err("an unrecognized label must be refused");
        let msg = err.to_string();
        assert!(msg.contains("catastrophic"), "{msg}");
    }

    #[test]
    fn resolve_severe_labels_returns_indices_in_the_ids_own_label_order() {
        let labels = vec!["destructive".to_string(), "informative".to_string(), "mutating".to_string()];
        let ids = resolve_severe_labels(&labels, &["mutating", "destructive"]).expect("both present");
        assert_eq!(ids, vec![2, 0]);
    }

    // ---------------------------------------------------------------
    // Severity ranking (expected ordinal rank). See the module docs for
    // why summing the severe rungs was replaced.

    #[test]
    fn severity_ranks_weight_by_position_in_the_caller_s_order() {
        // v8's vocabulary, stored in ordinal order.
        let labels = vec![
            "informative".to_string(),
            "situation-normal".to_string(),
            "data-critical".to_string(),
        ];
        let w = severity_rank_weights(&labels, &["situation-normal", "data-critical"])
            .expect("both present");
        // Non-severe labels weigh 0; severe labels weigh 1..=n in the
        // caller's ascending order.
        assert_eq!(w, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn severity_ranks_follow_the_callers_order_not_the_checkpoints() {
        // v6 stores its labels in a NON-ordinal order — this is the whole
        // reason rank cannot be derived from the checkpoint.
        let labels = vec![
            "destructive".to_string(),
            "informative".to_string(),
            "mutating".to_string(),
        ];
        let w = severity_rank_weights(&labels, &["mutating", "destructive"]).expect("both present");
        // destructive is stored FIRST but ranks HIGHEST (2.0); mutating is
        // stored last but is the middle rung (1.0).
        assert_eq!(w, vec![2.0, 0.0, 1.0]);
    }

    #[test]
    fn severity_ranks_reject_a_duplicate_label() {
        let labels = vec![
            "informative".to_string(),
            "situation-normal".to_string(),
            "data-critical".to_string(),
        ];
        let err = severity_rank_weights(&labels, &["data-critical", "data-critical"])
            .expect_err("a duplicated label has no unambiguous rank");
        let msg = err.to_string();
        assert!(msg.contains("data-critical"), "{msg}");
    }

    #[test]
    fn severity_ranks_reject_empty_and_unknown_like_the_set_does() {
        let labels = vec!["informative".to_string(), "mutating".to_string()];
        let empty: [&str; 0] = [];
        assert!(severity_rank_weights(&labels, &empty).is_err());
        let err = severity_rank_weights(&labels, &["mutating", "catastrophic"])
            .expect_err("unknown label must be refused");
        assert!(err.to_string().contains("catastrophic"));
    }

    /// THE REGRESSION TEST. Observed live on the deployed v8 daemon
    /// 2026-08-11, with the shipped sum-of-severe-rungs aggregation:
    ///
    ///     psql -f 001_init.sql   inf 0.0702  sn 0.6002  dc 0.3296
    ///     psql -f drop_all.sql   inf 0.1701  sn 0.2297  dc 0.6002
    ///
    /// The second clause is nearly twice as data-critical, and the sum
    /// ranked it BELOW the first (0.9298 vs 0.8299) because probability
    /// mass moving from the middle rung to the top rung leaves the sum
    /// unchanged at best, and lowers it whenever the bottom rung gains any.
    /// Expected ordinal rank is monotone by construction and cannot do this.
    #[test]
    fn a_more_data_critical_clause_outranks_a_less_data_critical_one() {
        let labels = vec![
            "informative".to_string(),
            "situation-normal".to_string(),
            "data-critical".to_string(),
        ];
        let w = severity_rank_weights(&labels, &["situation-normal", "data-critical"]).unwrap();

        let init = severity_score_from_probs(&[0.0702, 0.6002, 0.3296], &w);
        let drop_all = severity_score_from_probs(&[0.1701, 0.2297, 0.6002], &w);

        // The old sum ranked these the wrong way round; assert the sum
        // really would have, so this test documents the defect it guards.
        let old_sum_init = 0.6002 + 0.3296;
        let old_sum_drop = 0.2297 + 0.6002;
        assert!(
            old_sum_init > old_sum_drop,
            "the historical defect should reproduce: {old_sum_init} vs {old_sum_drop}"
        );

        assert!(
            drop_all > init,
            "dropping all tables must outrank a schema init: {drop_all} vs {init}"
        );
    }

    /// Monotonicity as a property, not just on the one observed pair:
    /// moving probability mass UP the ordinal scale must never lower the
    /// score. This is the invariant the sum violated.
    #[test]
    fn moving_mass_up_the_scale_never_lowers_the_score() {
        let labels = vec![
            "informative".to_string(),
            "situation-normal".to_string(),
            "data-critical".to_string(),
        ];
        let w = severity_rank_weights(&labels, &["situation-normal", "data-critical"]).unwrap();

        // Walk mass from `informative` all the way to `data-critical` in
        // steps, through the middle rung, and require a non-decreasing score.
        let mut previous = f32::NEG_INFINITY;
        for step in 0..=20 {
            let t = step as f32 / 20.0;
            // t=0: all informative. t=0.5: all situation-normal. t=1: all data-critical.
            let probs = if t <= 0.5 {
                let u = t * 2.0;
                [1.0 - u, u, 0.0]
            } else {
                let u = (t - 0.5) * 2.0;
                [0.0, 1.0 - u, u]
            };
            let score = severity_score_from_probs(&probs, &w);
            assert!(
                score >= previous - 1e-6,
                "score fell from {previous} to {score} while mass moved UP the scale (t={t})"
            );
            previous = score;
        }
    }
}
