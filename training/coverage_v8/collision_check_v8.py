#!/usr/bin/env python3
"""Collision + contradiction check: v8 incoming rows vs the v7 pool + eval sets.

Adapted from ../kube_pool.py's reserved-set logic and
../kube_collision_check.py's hard-contradiction check, retargeted at the
v7 axis and pool_v7.jsonl/eval_v7/*.jsonl instead of the v6 kube_slice
files. Policy: aggregates and file:line references only, never row text
(CLAUDE.md data policy).

Checks:
  1. Exact-normalized-text collision against pool_v7 + every eval_v7 set
     (dropped -- these rows already exist or would leak an eval set).
  2. Exact-normalized-text collision WITHIN the incoming batch itself
     (dedup before folding).
  3. HARD contradiction: an incoming row's normalized text matches an
     existing pool_v7 row with a DIFFERENT label (would corrupt the pool
     if ever merged -- reported loudly even though this run never merges).

Usage:
    python3 collision_check_v8.py <incoming1.jsonl> [<incoming2.jsonl> ...]

Exit 0 clean, 1 if any hard contradiction found (non-fatal to this run --
gates the eventual merge, which is explicitly out of scope here).
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DATA = Path.home() / ".local/share/lfm2-training-data"
POOL_V7 = DATA / "relabel_v7/pool_v7.jsonl"
EVAL_V7_DIR = DATA / "relabel_v7/eval_v7"


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def load_reserved() -> tuple[dict[str, str], set[str]]:
    """norm(text) -> label for pool_v7; set of norm(text) across all eval_v7 sets."""
    pool_by_text = {}
    for line in open(POOL_V7):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        pool_by_text[norm(r["text"])] = r["label"]
    eval_texts = set()
    for p in sorted(EVAL_V7_DIR.glob("*.jsonl")):
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            eval_texts.add(norm(json.loads(line)["text"]))
    return pool_by_text, eval_texts


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    pool_by_text, eval_texts = load_reserved()
    print(f"reserved: {len(pool_by_text)} pool_v7 rows, {len(eval_texts)} distinct eval_v7 texts")

    seen_in_batch: dict[str, list[str]] = defaultdict(list)  # norm -> [file:line, ...]
    kept = 0
    n_pool_collision = 0
    n_eval_collision = 0
    n_intra_dup = 0
    n_hard_contradiction = 0

    for arg in sys.argv[1:]:
        path = Path(arg)
        for i, line in enumerate(open(path), 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = norm(r["text"])
            ref = f"{path.name}:{i}"

            if key in eval_texts:
                n_eval_collision += 1
                print(f"  EVAL COLLISION: {ref} matches an eval_v7 row -- drop")
                continue
            if key in pool_by_text:
                n_pool_collision += 1
                if pool_by_text[key] != r["label"]:
                    n_hard_contradiction += 1
                    print(f"  HARD CONTRADICTION: {ref} label={r['label']!r} "
                          f"vs pool_v7 label={pool_by_text[key]!r}")
                else:
                    print(f"  pool collision (same label, harmless dup): {ref}")
                continue
            if key in seen_in_batch:
                n_intra_dup += 1
                print(f"  INTRA-BATCH DUP: {ref} matches {seen_in_batch[key][0]}")
                seen_in_batch[key].append(ref)
                continue
            seen_in_batch[key].append(ref)
            kept += 1

    print(f"\nsummary: kept {kept}  pool_collisions {n_pool_collision} "
          f"(hard_contradictions {n_hard_contradiction})  eval_collisions {n_eval_collision} "
          f"intra_batch_dups {n_intra_dup}")
    sys.exit(1 if n_hard_contradiction else 0)


if __name__ == "__main__":
    main()
