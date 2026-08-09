# v7 relabel — recoverability axis over the full kube pool

Decisions (Amy, 2026-08-09): three cheap blind families (deepseek,
gemini-flash, or-glm via kaibo), vote-proportional targets, NL-whole
rows pulled OUT of v7 train but still labeled — they become the training
set for a future NL head on the shared trunk. Labels move to the v7 axis
`informative < situation-normal < data-critical` under rulings R1–R8
(`rubric.md` is the verbatim labeler prompt).

The eval sets (kube_test, gen2 holdouts, subtle evalonly,
eval_realistic) are relabeled in the same blind run — v7 cannot be
scored against `mutating`. Chunks are shuffled so labelers see no group
signal. Note: adding eval_realistic to the reserved set means the pool
may shed a few more collision rows than v6's 2,093 — the build step
prints the count.

## Flow

1. `python3 relabel.py pilot 58` — blind chunks for the recoverability
   gold set. Run all three families over them with the rubric prompt.
2. `python3 relabel.py score-pilot ds=...jsonl gm=...jsonl glm=...jsonl`
   — per-family agreement vs adjudicated gold + ordinal-median ensemble.
   **Gate: do not spend on the full run until families look sane here**
   (blind three-family review of the cascade set ran 89/109 unanimous;
   a family far below that on gold means rubric wording work, not
   spending).
3. `python3 relabel.py build 100` — blind chunks for pool + eval sets;
   manifest with group/old-label kept OUT of the chunks.
4. Three families label every chunk (kaibo batch). Verdicts are strict
   JSONL {"id","label"}; the driver hard-fails on gaps, bad labels, or
   conflicting duplicate ids.
5. `python3 relabel.py apply ds=... gm=... glm=...` — folds votes:
   label = ordinal median, target = vote fractions, contested flag,
   flip-rate vs the no-relabel mapping per group. Writes
   `~/.local/share/lfm2-training-data/relabel_v7/pool_v7.jsonl` +
   `eval_v7/<set>.jsonl`.
6. `python3 ../form_tag.py tag pool_v7.jsonl --write` then
   `python3 relabel.py splits` — kube_train_v7 / kube_val_v7 (shellish
   only, stratified by source×label, seed 42) + `nl_v7.jsonl` set-aside.

v6's `kube_prep_splits.py` stays frozen; pool assembly for v7 lives in
`../kube_pool.py` (same gates, stable content-addressed ids).
Everything here prints aggregates only — never a raw row.
