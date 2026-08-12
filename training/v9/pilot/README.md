# v9 rubric pilot — the gate before bulk labeling

Ran 2026-08-12, per `pilot-gate-the-rubric`: never bulk-label without a
gold pilot, and read the disagreements before generating at volume.

## Setup

- `../labeler_prompt.txt` — v8's rubric plus two new rules encoding Amy's
  2026-08-12 rulings: rule 11 (a tool's own refusal interlock counts,
  because the tool name is in the text) and rule 12 (an interlock only
  counts against the harm in question; history rewrites stay
  data-critical regardless of cookies).
- `pilot.jsonl` — 18 rows: 9 gold from Amy's recorded rulings, 8
  authored-provisional (the guardrail pairs, both arms), 1 deliberate
  tension probe (p14).
- Three blind families, one shot each, full rubric + rows, no shared
  context: deepseek-v4-pro, gemini-3.5-flash, glm-5.2 (via openrouter).
  Intended third family was claude-haiku-4.5; the Anthropic API key was
  out of credits, glm substituted.
- `score_pilot.py [raw-dir]` aggregates and classifies each row's
  agreement shape. Raw outputs are verbatim in `raw/` (round 1) and
  `raw-r2/` (round 2).

## Round 1 (`raw/`): 14/18 unanimous-right, 4 splits, 0 unanimous-wrong

Every split traced to a rubric defect, and each got a wording fix:

1. **p08/p09 (bare `--amend`, `rebase -i`)** — GLM labeled
   situation-normal, correctly noting "already-pushed" is invisible in
   the text. Amy's ruling prices the asymmetry (advisory guard: a flag
   costs seconds, a miss loses history), so rule 12 now says bare
   rewrites default to data-critical BECAUSE the text cannot show
   whether history was shared.
2. **p13 (`kubectl delete namespace prod`)** — gemini cited rule 11's
   own example list, which wrongly offered "kubectl delete passes
   admission control" as an interlock. Admission control interlocks
   against malformed requests, not data loss. Example moved to rule 12
   as the counter-example. This was the pilot catching the rubric
   author's bug — one family followed the bad example off the cliff.
3. **p16 (curl POST carrying a severe command)** — gemini invoked rule
   3's "sending a message" for a generic API POST. Rule 3 now scopes
   "sending" to human-received communications and publishes; analysis
   POSTs are judged by rules 2/8. This shape is the #1 measured
   false-positive source in the advisory hook (all 11 firings on
   2026-08-12 were data-position), so the rubric had to be airtight
   here.

## Round 2 (`raw-r2/`): 17/18 unanimous-right, 1 split, 0 unanimous-wrong

All four round-1 splits converged to gold under the amended rubric. The
one remaining split is **p14 (`kubectl delete pod api-7f9c4d5b8-xk2mn`)**,
2-1 for situation-normal — the probe row built to surface exactly this
tension (controller respawn is rule-1 external, yet routine ops are sn).
Kept as a known 2-1 rather than papered over with another rule: the
hash-suffixed pod name is text-visible evidence of controller ownership
(rule 10's "the name's claim counts" could absorb it), but one more rule
risks overfitting the rubric to its own pilot. Bulk labeling keeps
disagreements per the process rules, so this family of rows will be
measured, not declared (`measure-disagreement-dont-declare-it`).

## Verdict

GATE PASSED. The rubric is fit for bulk generation of slices 1-3
(worktree/branch cleanup pairs, history rewrite, data-position
carriers). Open before bulk LABELING at volume: Amy's labeling-budget
and family-choice question from PLAN.md, and restoring an Anthropic
family if she wants three distinct vendors to include one Claude.
