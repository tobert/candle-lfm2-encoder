#!/usr/bin/env python3
"""Blind-relabel prep and scoring for cross-family label validation.

prep:  strip labels/authors/notes from incoming generated files into one
       shuffled blind file (id + text only) a relabeling subagent reads;
       the id->source key stays beside it on disk.
score: join a predictions file ({"id", "label"} per line) against the
       key and print AGGREGATE agreement per source file plus a pooled
       confusion matrix. Disagreement ids (never text) go to stdout; a
       texts-included disagreement file is written on disk for the
       second-opinion pass.

Prints aggregates only — never a raw row (CLAUDE.md data policy).

Usage:
    python crosscheck_prep.py prep <out_dir> <file.jsonl> [...]
    python crosscheck_prep.py score <out_dir> <predictions.jsonl>
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

VALID = {"informative", "mutating", "destructive"}


def prep(out_dir: Path, files: list[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({"id": f"{f.stem}:{i}", "text": r["text"], "label": r["label"]})
    random.Random(20260805).shuffle(rows)
    with (out_dir / "blind_rows.jsonl").open("w") as blind, (
        out_dir / "blind_key.jsonl"
    ).open("w") as key:
        for r in rows:
            blind.write(json.dumps({"id": r["id"], "text": r["text"]}) + "\n")
            key.write(json.dumps({"id": r["id"], "label": r["label"]}) + "\n")
    print(f"prep: {len(rows)} rows from {len(files)} files -> {out_dir}/blind_rows.jsonl")


def score(out_dir: Path, pred_path: Path) -> None:
    key = {}
    text_by_id = {}
    for line in (out_dir / "blind_key.jsonl").read_text().splitlines():
        r = json.loads(line)
        key[r["id"]] = r["label"]
    for line in (out_dir / "blind_rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        text_by_id[r["id"]] = r["text"]

    preds = {}
    for line in pred_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("label") in VALID:
            preds[r["id"]] = r["label"]

    missing = sorted(set(key) - set(preds))
    extra = sorted(set(preds) - set(key))
    per_file: dict[str, Counter] = defaultdict(Counter)
    confusion: Counter = Counter()
    disagreements = []
    for rid, gold in key.items():
        p = preds.get(rid)
        if p is None:
            continue
        src = rid.rsplit(":", 1)[0]
        per_file[src]["n"] += 1
        confusion[(gold, p)] += 1
        if p == gold:
            per_file[src]["agree"] += 1
        else:
            severe = {gold, p} == {"informative", "destructive"}
            per_file[src]["severe" if severe else "adjacent"] += 1
            disagreements.append({"id": rid, "text": text_by_id[rid],
                                  "generator_label": gold, "relabel": p})

    print(f"scored {len(preds)} predictions; missing={len(missing)} extra={len(extra)}")
    for src in sorted(per_file):
        c = per_file[src]
        print(f"  {src}: {c['agree']}/{c['n']} agree "
              f"({c['agree'] / c['n']:.0%}), adjacent={c['adjacent']}, severe={c['severe']}")
    print("pooled confusion (generator -> relabel):")
    for (g, p), n in sorted(confusion.items()):
        marker = " <-- " if g != p else ""
        print(f"  {g:12s} -> {p:12s} {n}{marker}")
    dis_path = out_dir / "disagreements.jsonl"
    with dis_path.open("w") as f:
        for d in disagreements:
            f.write(json.dumps(d) + "\n")
    print(f"{len(disagreements)} disagreements written to {dis_path} (ids: "
          + ", ".join(d["id"] for d in disagreements[:40]) + ")")


def main() -> None:
    if len(sys.argv) < 4 or sys.argv[1] not in {"prep", "score"}:
        raise SystemExit(__doc__)
    out_dir = Path(sys.argv[2])
    if sys.argv[1] == "prep":
        prep(out_dir, [Path(a) for a in sys.argv[3:]])
    else:
        score(out_dir, Path(sys.argv[3]))


if __name__ == "__main__":
    main()
