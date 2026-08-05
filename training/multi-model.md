# Multi-model generation survey — 2026-08-05

Fixes the one-family blind spot: all gen2 rows and both holdouts were
generated AND labeled by claude-haiku-4-5, so holdout numbers measured
same-family generalization. This survey probed five outside families as
generators and cross-validated labels three ways.

## Transport (proven this session)

kaibo consult with `save_artifacts` → `kaibo://cas/<digest>` → copy to
`~/.local/share/lfm2-training-data/incoming/` → **sha256(file) ==
digest** → `validate_incoming.py` (schema gate) → `report_dataset_stats.py`
(dup/length aggregates). Rows never transit an exec-capable agent
context; subagents and read-only kaibo consults do the row work
(aggregates-only reporting).

## Roster and probe design

Standard probes: 30 command-shaped rows, self-chosen persona, house
rubric from `personas.md`. Subtle probes (kimi, glm): two-pass sessions —
a design turn proposing a 12-category boundary taxonomy, then a
generation turn producing 20 training rows + 10 eval-only rows. Locals
were down; anthropic casts skipped (same family as incumbent data).

## Scorecard (labels post-adjudication)

| family | rows | gates | near-dup | self-report | haiku agree | v3 agree | notes |
|---|---|---|---|---|---|---|---|
| deepseek-v4-pro | 30 | 0 err | 0% | exact | 93% | 93% | best chain variety (12 binaries); quoted fragments in answer |
| gpt-5.6-luna | 30 | 0 err | 0% | exact | 97% | 87% | novel persona; cleanest compliance; cheapest probe |
| gemini-3.5-flash | 30 | 0 err | 3.3% | exact | 100% | 93% | costliest probe (96k in + 22k reasoning) |
| qwen3-coder-next | 32 | 0 err | 0% | **wrong** | 88% | 87.5% | ignored "exactly 30"; 3 labels overturned; short rows |
| kimi-k2.7 subtle | 20+10 | 0 err | 0% | exact | 60–80% | 85%/70% | ALL 7 disputed labels upheld — haiku was wrong, not kimi |
| glm-5.2 subtle | 20+10 | 0 err | 0% | exact | 90–100% | 80%/**30%** | eval-only slice crushes v3, labels 10/10 validated |

Total: 182 rows, ~350k in / 85k out tokens across providers (< $1).

## Label adjudication

Blind haiku relabel of all 182 (shuffled, no labels/authors) agreed with
generators on 165 (91%). The 17 disagreements went to a blind
gpt-5.6-luna third vote; majority ruled: **13 generator labels upheld, 4
overturned** (all destructive→mutating: qwen 3, deepseek 1 — the etcd
snapshot-save direction and serviceaccount-adjacent deletes). Overturned
rows patched in `incoming/` with `contested: true` and an adjudication
note; the CAS artifacts remain the pristine originals.

Pattern worth watching: haiku sat alone on 13 of 17 disputes, always on
the lenient side of subtle destructive rows. The gen2 corpus is
haiku-labeled; boundary-row severity may be systematically soft there.

## Findings

1. **v3 generalizes across families on ordinary command-shaped rows**
   (87–97% agreement) — the family blind spot is small for the covered
   surface form, consistent with the form-coverage law.
2. **Adversarial read-shaped chains are an uncovered form.** GLM's
   eval-only probes (buried-destructive-in-long-chain, etcd-level
   finalizer bypass, read-chain-ending-in-write) score **3/10 with five
   severe destructive→informative misses** — the dangerous direction
   (under-flagging). Labels independently validated 10/10 by haiku.
   This slice is v4's first target.
3. **Audit-log deletion is EVAL-ONLY** (Amy's ruling, this session):
   kimi argued training on it teaches a brittle "audit = destructive"
   keyword override; probe first, train only if v4 still fails.
4. **Never trust generator self-reports** — qwen claimed 30 rows @
   12/13/5, delivered 32 @ 12/9/11. Gates on artifacts, always.
5. Kimi/GLM taxonomies converged independently (finalizer surgery, audit
   pruning, destructive-as-data, cascade choices, controller-vs-bare,
   helm edges, exec trailing risk) — evidence the boundary categories
   are real, not one model's idiosyncrasy.

## Disposition

- `incoming/kube_gen_{deepseek,gemini,gpt_luna,qwen}_sample.jsonl` and
  `incoming/kube_subtle_{kimi,glm}_train.jsonl` (142 rows): candidate
  training material for v4, pending the usual `kube_prep_splits.py`
  contradiction/leakage gates against the full corpus.
- `incoming/kube_subtle_{kimi,glm}_evalonly.jsonl` (20 rows): join the
  holdout suite; never train.
- Cross-check machinery: `crosscheck_prep.py` (prep/score),
  `validate_incoming.py` (schema gate). Verdicts and blind files under
  `~/.local/share/lfm2-training-data/crosscheck/`.

Models engaged: deepseek-v4-pro, gpt-5.6-luna, gemini-3.5-flash,
qwen3-coder-next, kimi-k2.7-code, glm-5.2 (generation/design);
claude-haiku-4-5 (blind relabel); gpt-5.6-luna (blind arbitration);
all via kaibo except the haiku subagent.
