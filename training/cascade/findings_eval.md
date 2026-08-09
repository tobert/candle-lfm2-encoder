# Cascade eval on adjudicated gold: the composite holds, the router is the weak link

Measured 2026-08-09 against the 53 adjudicated rows of `probes.json`
(marker COMPLETE, rulings R1–R8), checkpoints `kube_ordinal_v6` +
`LFM2.5-Encoder-350M-Prompt-Router` (`capability_noname` lanes,
severe = {mutating, destructive}), no-relabel mapping informative→
informative, mutating→situation-normal, destructive→data-critical.
Raw per-clause dump with checkpoint sha256 provenance:
`eval_v6_router_raw.json`. Run by a Claude Sonnet subagent over
`Cascade::run`; numbers gate-checked against the raw dump.

## Headline table

| metric | result |
|---|---|
| winner selection (36 multi-clause rows, bands a/b/c/e) | **33/36 = 91.7%** |
| per-clause severity vs gold (n=109) | 78.0% |
| … gold informative | 98.0% |
| … gold data-critical | 76.3% |
| … gold **situation-normal** | **36.4%** |
| winner's lane vs authored expected_lane | 24/36 = 66.7% |
| union recall (expected lane ≥ 0.5 anywhere) | 31/36 = 86.1% |
| benign-band max severity vs data-critical-winner min | 0.3415 vs 0.3440 |
| calibration separation (escalate_measured vs unanimous) | +0.026…+0.039 |

## The five readings

1. **The composite generalizes, imperfectly: 91.7% winner selection**
   against the pinned 5/5 it was built from. The three misses are each
   instructive, not random: `a6` loses the R8-ruled `bash` token by
   0.005 of severity (token scoring means a pipe-sink carries almost no
   signal — the named pipe-blindness limitation, now with a number);
   `c5` has a `cd` clause outscoring `psql -f` on SEVERITY (the nav
   trap exists in v6's severity space too, not just the router's lane
   space); `e4` flips two clauses that are both ≥0.99 severe (harmless
   for gating, wrong for ranking).

2. **The no-relabel hypothesis fails exactly in the middle class.**
   situation-normal accuracy is 36.4%, and 14/22 errors are
   situation-normal→informative: v6's `mutating` never fires for the
   R6 class of "durable artifact created" (`git clone`, `npm install`,
   `curl -o`, `aws s3 cp`, `git commit`). v6 reads the verb; R6 defined
   the middle class by effect. **This is the concrete, measured case for
   the corpus relabel** — the boundary R6 set for 2,093 rows is one the
   current model cannot express.

3. **The router, not ranking, is the lane bottleneck.** 10 of 12 lane
   misses happen on rows whose winner was selected CORRECTLY. Lane
   wording is the classifier (see `lane-wording-is-the-classifier`);
   the wording pass (board item: git lane misses clone AND rebase, k8s
   is a magnet) is now the highest-leverage router work, ahead of any
   architecture change.

4. **No global severity threshold exists.** The most severe benign-chain
   statement (0.3415, `git branch -a` chain) and the least severe
   data-critical winner (0.3440, `git rebase -i` per R7) are separated
   by 0.0025. Ranking within a statement works; thresholding across
   statements cannot. Any future gate must rank, then decide with more
   context than one number.

5. **Calibration is worse here than on recoverability.** Disputed
   clauses are less confident in the right direction, but the
   separation (+0.026 clause-level, +0.039 row-level) is a quarter of
   recoverability's already-unusable +0.100. Confidence-carried
   escalation stays blocked; flat-target training on measured-escalate
   rows remains the lever, and calibration must be a first-class eval
   metric for any v7.

Design property confirmed: all four R4-gap exfil rows (f2/f3/f4/f7)
stay non-severe exactly as designed — the severity axis has no opinion
on disclosure, which is why the PII lane must be always-on, never
routed to.

## Caveats

Single probe set, 53 rows, authored expected_lane not adjudicated
(b8/e8 lane calls flagged as judgment calls). The winner-selection
number excludes benign_chain (null winners by convention) and the two
single-clause null-winner rows (f3/f7 — convention pending a ruling).
f3's severity miss is a 5.5pp coin flip, worth ignoring until the
relabel lands.
