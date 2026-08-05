#!/usr/bin/env python3
"""Cross-slice label contradiction check for kube_slice_*.jsonl.

Policy: aggregates and slice:line references only. Never prints the `text`
or `note` field of any row. Metadata fields (verb, resource, label,
contested) are short categorical values and may be printed.

Checks, in order of severity:
  1. HARD: identical normalized text, different label (across or within slices).
  2. SOFT: same (verb, resource) pair with a mixed label distribution --
     expected for context-dependent data, reported for eyeballing only.
Exit code 1 if any hard contradiction is found among non-contested rows,
2 if only contested-involved hard contradictions, 0 if clean.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DATA = Path.home() / ".local/share/lfm2-training-data/prepared"
SLICES = sorted(DATA.glob("kube_slice_*.jsonl"))

def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

rows = []  # (slice_name, lineno, norm_text, label, verb, resource, contested)
for path in SLICES:
    sname = path.stem.replace("kube_slice_", "")
    for i, line in enumerate(open(path), 1):
        r = json.loads(line)
        rows.append((sname, i, norm(r["text"]), r["label"], r.get("verb"),
                     r.get("resource"), bool(r.get("contested"))))

print(f"total rows: {len(rows)} across {len(SLICES)} slices")

# --- 1. hard contradictions: same normalized text, >1 distinct label
by_text = defaultdict(list)
for row in rows:
    by_text[row[2]].append(row)

exact_dups = {t: rs for t, rs in by_text.items() if len(rs) > 1}
print(f"\nnormalized-text duplicate groups (any label): {len(exact_dups)}")

hard = {t: rs for t, rs in exact_dups.items()
        if len({r[3] for r in rs}) > 1}
hard_noncontested = {t: rs for t, rs in hard.items()
                     if any(not r[6] for r in rs)}

print(f"HARD contradictions (same text, different label): {len(hard)}")
print(f"  ...involving at least one NON-contested row: {len(hard_noncontested)}")

for t, rs in sorted(hard.items(), key=lambda kv: -len(kv[1])):
    refs = ", ".join(f"{s}:{ln}={lab}{'(c)' if c else ''}"
                     for s, ln, _, lab, _, _, c in rs)
    verbs = sorted({str(r[4]) for r in rs})
    resources = sorted({str(r[5]) for r in rs})
    print(f"  verb={verbs} resource={resources} -> {refs}")

# duplicate groups with SAME label (pure dups, noise not contradiction)
same_label_dups = {t: rs for t, rs in exact_dups.items()
                   if len({r[3] for r in rs}) == 1}
cross_slice_same = sum(1 for rs in same_label_dups.values()
                       if len({r[0] for r in rs}) > 1)
print(f"\nsame-label duplicate groups: {len(same_label_dups)} "
      f"({cross_slice_same} span slices) -- dedup before training")

# --- 2. soft signal: (verb, resource) label mix
by_vr = defaultdict(lambda: defaultdict(int))
for s, ln, t, lab, v, res, c in rows:
    by_vr[(str(v), str(res))][lab] += 1

mixed = {vr: d for vr, d in by_vr.items() if len(d) > 1}
print(f"\n(verb, resource) pairs with mixed labels: {len(mixed)} of {len(by_vr)}")
print("(expected for context-dependent data; top 15 by row count)")
for (v, res), d in sorted(mixed.items(), key=lambda kv: -sum(kv[1].values()))[:15]:
    total = sum(d.values())
    dist = " ".join(f"{k}={n}" for k, n in sorted(d.items()))
    print(f"  {v} {res} (n={total}): {dist}")

# specific check from the signoff: bare `delete pod`
dp = [r for r in rows if r[4] == "delete" and str(r[5]).startswith("pod")]
dp_labels = defaultdict(int)
for r in dp:
    dp_labels[r[3]] += 1
print(f"\n'delete pod*' rows: {len(dp)}, labels: {dict(sorted(dp_labels.items()))}")

if hard_noncontested:
    sys.exit(1)
elif hard:
    sys.exit(2)
sys.exit(0)
