#!/usr/bin/env python3
"""Shared kube-pool assembly for v7+.

kube_prep_splits.py (v6) stays FROZEN as the historical artifact; this
module is the same assembly + gates, importable, with stable
content-addressed row ids. Loud failure everywhere; aggregates only.

Groups:
  train pool  — the ~2,093 gated rows (post contradiction/leak/dup gates)
  eval sets   — kube_test, gen2 holdouts, subtle evalonly, eval_realistic
                (NEVER in train; relabeled to the v7 axis alongside the
                pool so v7 can be scored on the axis it is trained on)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DATA = Path.home() / ".local/share/lfm2-training-data"
VALID_V6 = {"informative", "mutating", "destructive"}

# v6 label -> v7 axis under the no-relabel mapping (for flip accounting only)
V6_TO_V7 = {
    "informative": "informative",
    "mutating": "situation-normal",
    "destructive": "data-critical",
}


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def row_id(text: str) -> str:
    return hashlib.sha256(norm(text).encode()).hexdigest()[:12]


def _load(path: Path, source: str, valid: set[str] | None = VALID_V6) -> list[dict]:
    """valid=None skips the label-vocabulary gate — eval sets carry their
    own historical vocabularies (e.g. eval_realistic's 'safe') and their
    old labels are used only for flip accounting."""
    out = []
    for lineno, line in enumerate(open(path), 1):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            sys.exit(f"GATE FAIL: {path.name}:{lineno} invalid JSON: {e}")
        if valid is not None and r.get("label") not in valid:
            sys.exit(f"GATE FAIL: {path.name}:{lineno} bad label {r.get('label')!r}")
        if not str(r.get("text", "")).strip():
            sys.exit(f"GATE FAIL: {path.name}:{lineno} empty text")
        out.append({
            "id": row_id(r["text"]),
            "text": r["text"], "label": r.get("label"), "source": source,
            "verb": r.get("verb"), "resource": r.get("resource"),
            "contested": bool(r.get("contested")),
        })
    if not out:
        sys.exit(f"GATE FAIL: {path.name} is empty")
    return out


def _train_files() -> list[tuple[Path, str]]:
    files = []
    for p in sorted((DATA / "prepared").glob("kube_slice_*.jsonl")):
        files.append((p, "slice-" + p.stem.replace("kube_slice_", "")))
    gen2 = sorted((DATA / "gen2").glob("kube_gen2_*.jsonl"))
    holdout = [p for p in gen2 if "holdout" in p.name]
    train = [p for p in gen2 if "holdout" not in p.name]
    if len(train) != 9 or len(holdout) != 2:
        sys.exit(f"GATE FAIL: expected 9 train + 2 holdout gen2 files, "
                 f"got {len(train)} + {len(holdout)}")
    for p in train:
        files.append((p, p.stem.replace("kube_gen2_", "gen2-")))
    incoming = sorted((DATA / "incoming").glob("kube_gen_*_sample.jsonl")) + \
        sorted((DATA / "incoming").glob("kube_subtle_*_train.jsonl")) + \
        sorted((DATA / "incoming").glob("kube_chains_*_train.jsonl")) + \
        sorted((DATA / "incoming").glob("kube_nl_*_train.jsonl")) + \
        sorted((DATA / "incoming").glob("kube_v6_*_train.jsonl"))
    evalonly = sorted((DATA / "incoming").glob("kube_subtle_*_evalonly.jsonl"))
    if (len(incoming), len(evalonly)) != (14, 2):
        sys.exit(f"GATE FAIL: expected (14, 2) incoming train/evalonly files, "
                 f"got ({len(incoming)}, {len(evalonly)})")
    for p in incoming:
        files.append((p, "mm-" + p.stem.replace("kube_", "").replace("_sample", "")))
    return files


def eval_set_files() -> list[tuple[Path, str]]:
    gen2 = sorted((DATA / "gen2").glob("kube_gen2_holdout_*.jsonl"))
    evalonly = sorted((DATA / "incoming").glob("kube_subtle_*_evalonly.jsonl"))
    files = [(DATA / "prepared/kube_test.jsonl", "kube_test"),
             (DATA / "prepared/eval_realistic.jsonl", "eval_realistic")]
    files += [(p, p.stem.replace("kube_gen2_", "")) for p in gen2]
    files += [(p, p.stem) for p in evalonly]
    return files


def load_train_pool() -> list[dict]:
    """Assemble + gate: contradictions abort; test/holdout leaks and exact
    dups dropped (counted). Returns kept rows with stable ids."""
    pool = []
    for p, source in _train_files():
        pool.extend(_load(p, source))

    reserved = set()
    for p, _ in eval_set_files():
        for line in open(p):
            line = line.strip()
            if line:
                reserved.add(norm(json.loads(line)["text"]))

    by_text = defaultdict(list)
    for r in pool:
        by_text[norm(r["text"])].append(r)
    contra = {t: rs for t, rs in by_text.items()
              if len({r["label"] for r in rs}) > 1}
    if contra:
        sys.exit(f"GATE FAIL: {len(contra)} hard contradictions in pool")

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
    print(f"pool: {len(pool)} -> kept {len(kept)} "
          f"(dropped {n_leak} eval-set collisions, {n_dup} exact dups)",
          file=sys.stderr)
    return kept


def load_eval_sets() -> dict[str, list[dict]]:
    """Eval sets keyed by set name, rows with stable ids. Deduped within
    a set only; sets may intentionally overlap each other."""
    out = {}
    for p, name in eval_set_files():
        rows, seen = [], set()
        for r in _load(p, name, valid=None):
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            rows.append(r)
        out[name] = rows
    return out

