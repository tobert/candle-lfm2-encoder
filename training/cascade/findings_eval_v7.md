# Cascade regression with v7: same headline, redistributed — and one new trap

Measured 2026-08-09 evening against the same 53 adjudicated rows and
IDENTICAL router configuration as the v6 eval (lane strings and router
sha256 diffed byte-for-byte; only the severity checkpoint changed:
`kube_ordinal_v7`, which speaks gold's vocabulary natively — no class
mapping, unlike v6). Raw dump: `eval_v7_router_raw.json`. Run by a
Claude Sonnet subagent; v6 baseline REPRODUCED under the same scorer
before any v7 number was trusted (winner 33/36 with the same three
misses, per-clause 78.0%, per-class 98.0/36.4/76.3 — all exact).

## Headlines vs v6

| metric | v6 | v7 |
|---|---|---|
| winner selection (36 rows) | 33/36 | **33/36 — same rate, different misses** |
| per-clause severity overall | 78.0% | **78.0% (identical, redistributed)** |
| … gold informative | 98.0% | 87.8% (−10.2pp) |
| … gold situation-normal | 36.4% | **59.1% (+22.7pp)** |
| … gold data-critical | 76.3% | 76.3% |
| calibration separation, clause (one consistent formula, see caveat) | +0.039 | **+0.107 (~3×)** |
| calibration separation, row/winner | +0.102 | +0.076 |

- **a6 FIXED** — the pipe-sink `bash` clause now wins the ranking with a
  real severity signal (0.377 vs v6's noise-level 0.05). Its per-clause
  class is still wrong (informative, not situation-normal per R8).
- **e4 FIXED cleanly** — bare pod delete (data-critical 0.9998) now
  separates from deployment scale (situation-normal); v6 had them tied
  at two 0.99s in `mutating`.
- **NEW misses a5, c2** — `git pull` outranks a config-dir delete;
  `cd vendor/legacy-plugin` outranks `npm install`.
- **c5 got WORSE, and it is the flag of this eval**: `cd db/migrations`
  is now **data-critical 0.898** (v6: informative 0.274). A bare `cd`
  destroys nothing — R5 says navigation is informative — and the score
  pattern looks like a keyword trigger on the path (`db`, `migrations`).
  The nav trap the router findings named now exists, confidently, in
  v7's severity space. **v8 coverage must include bare-`cd` /
  navigation forms (R5) alongside the R6/R7/R9 forms.**

## Lane note (router unchanged)

Winner's-lane went 24/36 → 21/36 with the router byte-identical (shared
clauses' cosines match across dumps). Entirely a cascade coupling:
severity picks the winner, the winner determines which clause's lane is
scored — a5/b7/c2 flipped winners, so different lanes got checked.
Lane wording remains the router bottleneck (board item unchanged).

## Methodology caveat — the committed v6 calibration numbers were wrong-ish

`findings_eval.md`'s +0.026/+0.039 could NOT be reproduced from the raw
v6 dump; the original scorer was an ephemeral subagent script and its
confidence formula is unrecoverable (several plausible variants tried,
none match). Every OTHER v6 number reproduced exactly. The table above
uses one stated formula (top-1 softmax confidence, escalate_measured vs
unanimous) computed identically for both checkpoints, so the delta is
apples-to-apples even though the baseline line moved. Consequence
adopted: **eval scorers get committed with their findings from now on**
— this file's scorer lives at `score_cascade.py` in this directory.

## Reading

The corpus relabel bought exactly what the measured case predicted
(situation-normal +22.7pp, a6/e4 fixed, clause-level calibration ~3×)
and paid for it with informative accuracy (−10.2pp) and a new confident
nav-trap false positive. Net per-clause accuracy is a wash at 78.0%;
the composite's value moved from "can't express the middle class" to
"expresses it, with coverage gaps where the corpus lacks the forms"
(chains-with-backout, name-claims, bare navigation). All three gaps are
generation targets, not architecture changes.
