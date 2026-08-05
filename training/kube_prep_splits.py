#!/usr/bin/env python3
"""Gate-check the kube pool and build kube_train_v5/kube_val_v5.

Inputs:
  prepared/kube_slice_*.jsonl          (original 4 slices)
  gen2/kube_gen2_<persona>.jsonl       (9 training personas)
  gen2/kube_gen2_holdout_*.jsonl       (2 holdout personas — NEVER in train)
  incoming/kube_gen_*_sample.jsonl     (4 multi-model standard probes)
  incoming/kube_subtle_*_train.jsonl   (2 multi-model subtle batches)
  incoming/kube_chains_*_train.jsonl   (4 read-shaped-chain form, v5)
  incoming/kube_nl_*_train.jsonl       (2 multi-family NL top-up, v5)
  incoming/kube_subtle_*_evalonly.jsonl (eval-only probes — NEVER in train)
  prepared/kube_test.jsonl             (v1 test — stays untouched)

Gates (loud failure, no silent drops except exact dups):
  1. Every line valid JSON, valid label, non-empty text.
  2. Hard contradiction check across old+new train pool (same normalized
     text, different label) -> abort if any.
  3. Leakage: any train-pool row whose normalized text appears in
     kube_test.jsonl or the holdout files is DROPPED from the pool
     (counted, reported).
  4. Same-label exact dups deduped (keep first).

Output: prepared/kube_train_v5.jsonl (90%), prepared/kube_val_v5.jsonl
(10%), stratified by (source, label). Aggregates printed; never a row.
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
DATA = Path.home() / ".local/share/lfm2-training-data"
VALID = {"informative", "mutating", "destructive"}

def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())

def load(path: Path, source: str) -> list[dict]:
    out = []
    for lineno, line in enumerate(open(path), 1):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            sys.exit(f"GATE FAIL: {path.name}:{lineno} invalid JSON: {e}")
        if r.get("label") not in VALID:
            sys.exit(f"GATE FAIL: {path.name}:{lineno} bad label {r.get('label')!r}")
        if not str(r.get("text", "")).strip():
            sys.exit(f"GATE FAIL: {path.name}:{lineno} empty text")
        out.append({
            "text": r["text"], "label": r["label"], "source": source,
            "verb": r.get("verb"), "resource": r.get("resource"),
            "contested": bool(r.get("contested")),
        })
    if not out:
        sys.exit(f"GATE FAIL: {path.name} is empty")
    return out

pool = []
for p in sorted((DATA / "prepared").glob("kube_slice_*.jsonl")):
    pool.extend(load(p, "slice-" + p.stem.replace("kube_slice_", "")))
gen2_files = sorted((DATA / "gen2").glob("kube_gen2_*.jsonl"))
holdout_files = [p for p in gen2_files if "holdout" in p.name]
train_files = [p for p in gen2_files if "holdout" not in p.name]
EXPECTED_TRAIN_FILES = 9  # 6 command personas + 3 NL personas (v3)
if len(train_files) != EXPECTED_TRAIN_FILES or len(holdout_files) != 2:
    sys.exit(f"GATE FAIL: expected {EXPECTED_TRAIN_FILES} train + 2 holdout gen2 files, "
             f"got {len(train_files)} + {len(holdout_files)}")
for p in train_files:
    pool.extend(load(p, p.stem.replace("kube_gen2_", "gen2-")))

incoming_train = sorted((DATA / "incoming").glob("kube_gen_*_sample.jsonl")) + \
    sorted((DATA / "incoming").glob("kube_subtle_*_train.jsonl")) + \
    sorted((DATA / "incoming").glob("kube_chains_*_train.jsonl")) + \
    sorted((DATA / "incoming").glob("kube_nl_*_train.jsonl"))
incoming_evalonly = sorted((DATA / "incoming").glob("kube_subtle_*_evalonly.jsonl"))
# 4 standard probes + 2 subtle batches + 3 chains (gemini 503'd out) +
# 2 NL top-up, 2 eval-only
EXPECTED_INCOMING = (11, 2)
if (len(incoming_train), len(incoming_evalonly)) != EXPECTED_INCOMING:
    sys.exit(f"GATE FAIL: expected {EXPECTED_INCOMING} incoming train/evalonly files, "
             f"got ({len(incoming_train)}, {len(incoming_evalonly)})")
for p in incoming_train:
    pool.extend(load(p, "mm-" + p.stem.replace("kube_", "").replace("_sample", "")))

reserved = set()
for p in [DATA / "prepared/kube_test.jsonl", *holdout_files, *incoming_evalonly]:
    for line in open(p):
        reserved.add(norm(json.loads(line)["text"]))

# gate 2: contradictions
by_text = defaultdict(list)
for r in pool:
    by_text[norm(r["text"])].append(r)
contra = {t: rs for t, rs in by_text.items() if len({r["label"] for r in rs}) > 1}
if contra:
    print(f"GATE FAIL: {len(contra)} hard contradictions (same text, different label):")
    for t, rs in list(contra.items())[:20]:
        print("  " + "; ".join(f"{r['source']}:{r['label']}" for r in rs)
              + f"  verb={rs[0].get('verb')} resource={rs[0].get('resource')}")
    sys.exit(1)

# gates 3+4: leakage + dedup
seen, kept = set(), []
n_leak = n_dup = 0
for r in pool:
    key = norm(r["text"])
    if key in reserved:
        n_leak += 1
        continue
    if key in seen:
        n_dup += 1
        continue
    seen.add(key)
    kept.append(r)

print(f"pool: {len(pool)} rows -> kept {len(kept)} "
      f"(dropped {n_leak} test/holdout collisions, {n_dup} exact dups)")
print("by source:", dict(sorted(Counter(r["source"] for r in kept).items())))
print("by label:", dict(sorted(Counter(r["label"] for r in kept).items())))
print("distinct verbs:", len({str(r['verb']).lower() for r in kept}),
      " distinct resources:", len({str(r['resource']).lower() for r in kept}))

rng = random.Random(SEED)
strata = defaultdict(list)
for r in kept:
    strata[(r["source"], r["label"])].append(r)
train, val = [], []
for key in sorted(strata):
    g = strata[key]
    rng.shuffle(g)
    n_val = max(1, round(len(g) * 0.10))
    val.extend(g[:n_val])
    train.extend(g[n_val:])
rng.shuffle(train)
rng.shuffle(val)
for name, items in [("kube_train_v5", train), ("kube_val_v5", val)]:
    with open(DATA / "prepared" / f"{name}.jsonl", "w") as f:
        for r in items:
            f.write(json.dumps(r) + "\n")
    print(f"{name}: {len(items)} rows  "
          f"labels={dict(sorted(Counter(r['label'] for r in items).items()))}")
print("OK: gates passed, v5 splits written")
