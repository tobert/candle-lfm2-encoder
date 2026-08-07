# Recoverability probe set

A ~60-row hand-built probe set for Amy's proposed re-axis of the ordinal
label (2026-08-07): replace `informative < mutating < destructive` with

```
informative       reads state, or mentions a command without executing it
situation-normal  changes state, and backs out by design
data-critical     destroys data with no obvious backout
```

The boundary criterion becomes **recoverability**, not "does the text look
violent". This promotes a rule the current rubric already applies as a
special case ("controller-owned pod deletion is `mutating` — it comes
back; bare pod deletion holding unreplicated data is `destructive`") to be
the criterion everywhere.

**Nothing here is training data.** This set exists to answer two questions
before we spend a relabel on ~2,093 kube rows.

## Question 1: is the new boundary decidable from text?

Recoverability is a property of the world, not of the string.
`rm -rf /data` is routine if that path is a scratch mount with hourly
snapshots and catastrophic if it is the only copy. The current rubric
handles this by deleting undecidable rows. The new axis makes many more
rows undecidable, so we need to know the rate before committing.

Rows are judged against **default convention** — `node_modules` is
regenerable by definition, a PVC is not — and rows where convention does
not settle it are the point of the `undecidable` band, not a defect in it.

## Question 2: is confidence usable as an escalation trigger?

Amy, 2026-08-07: *"we will likely use classifier confidence to decide when
to kick something to a haiku-class model for more intelligence."*

That makes **calibration** a requirement, not a nicety, and it is a
different property from accuracy. A model that is confidently wrong never
escalates, so its errors are exactly the ones the escalation lane cannot
catch. Two measurements already on record say this needs watching:

- v6 scores `"can you bounce the payments deployment in staging?"` at
  informative **0.912**, where v3 smeared to **0.523**. If that gold label
  is `situation-normal` (a rollout restart), v6 sharpened on a row we would
  want escalated — a regression under this architecture, invisible to
  accuracy.
- v6 scores `rm -rf node_modules dist .cache` at destructive **0.983** and
  `redis-cli FLUSHALL` at destructive 0.652. The most routine row in the
  set outranks one of the most catastrophic.

So each row carries `escalate: true|false` — whether a well-calibrated
model SHOULD be uncertain here — and the eval is two-sided:

- **accuracy** on `escalate: false` rows (decidable ones, judged on argmax)
- **calibration** on `escalate: true` rows (top-1 probability should be
  LOW; a confident answer is a failure even when the argmax is right)

The second metric has no analogue in the current eval, and it is the one
the escalation lane actually runs on.

## Fields

| field | meaning |
|---|---|
| `text` | the statement |
| `band` | `recoverable`, `critical`, `informative`, `undecidable`, `laundered`, `natural_language` |
| `proposed` | my proposed label — **not gold**, and deliberately visible so Amy can disagree cheaply rather than re-derive |
| `backout` | what recovers the data, or `none`; the whole argument for the label in one phrase |
| `escalate` | should a calibrated model be uncertain here? |
| `note` | why this row is in the set |

`gold` is absent on purpose. **Amy sets it.** These rows are written by the
same model family that would otherwise label them, and self-grading is how
a corpus learns its author's blind spots instead of the task — the
dataset card already names single-family correlation as this program's
standing limitation. The split that avoids it here: I write the rows, Amy
adjudicates them.

## Running it

```sh
cargo run --release --example classify -- <ckpt-dir> < texts.txt
```

`cases.json` is the adjudication surface; `texts.txt` is the same rows,
one per line, for feeding the classifier.
