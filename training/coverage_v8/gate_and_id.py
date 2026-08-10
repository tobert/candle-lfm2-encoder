#!/usr/bin/env python3
"""Final gate pass: dedup incoming/*.jsonl against pool_v7 + eval_v7 + itself,
assign stable content-addressed ids (kube_pool.row_id convention), tag each
row with its target family, write gated_v8.jsonl. Aggregates only printed.

Family is inferred from the filename prefix (nav/durable/noundo/trustname/backout).
Drop order matches collision_check_v8.py: eval collision > pool collision >
intra-batch dup > keep. First occurrence wins in glob (alphabetical) order.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from kube_pool import DATA, row_id  # noqa: E402

POOL_V7 = DATA / "relabel_v7/pool_v7.jsonl"
EVAL_V7_DIR = DATA / "relabel_v7/eval_v7"


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def family_of(path: Path) -> str:
    name = path.stem
    for fam in ("nav", "durable", "noundo", "trustname", "backout"):
        if name.startswith(fam + "_"):
            return fam
    return "unknown"


def main() -> None:
    pool_by_text = {}
    for line in open(POOL_V7):
        line = line.strip()
        if line:
            r = json.loads(line)
            pool_by_text[norm(r["text"])] = r["label"]
    eval_texts = set()
    for p in sorted(EVAL_V7_DIR.glob("*.jsonl")):
        for line in open(p):
            line = line.strip()
            if line:
                eval_texts.add(norm(json.loads(line)["text"]))

    files = sorted((HERE / "incoming").glob("*.jsonl"))
    seen: dict[str, str] = {}  # norm -> ref, for intra-batch dedup
    kept: list[dict] = []
    dropped = Counter()
    fam_counts = defaultdict(Counter)

    for path in files:
        fam = family_of(path)
        for i, line in enumerate(open(path), 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = norm(r["text"])
            if key in eval_texts:
                dropped["eval_collision"] += 1
                continue
            if key in pool_by_text:
                dropped["pool_collision"] += 1
                continue
            if key in seen:
                dropped["intra_batch_dup"] += 1
                continue
            seen[key] = f"{path.name}:{i}"
            rid = row_id(r["text"])
            row = {
                "id": rid, "text": r["text"], "gen_label": r["label"],
                "verb": r.get("verb"), "resource": r.get("resource"),
                "gen_contested": bool(r.get("contested")),
                "gen_note": r.get("note"), "author": r.get("author"),
                "family": fam, "source_file": path.name,
            }
            kept.append(row)
            fam_counts[fam][r["label"]] += 1

    ids = [r["id"] for r in kept]
    if len(set(ids)) != len(ids):
        sys.exit(f"FAIL: {len(ids) - len(set(ids))} id collisions among kept rows "
                  f"(sha256[:12] birthday collision or genuine dup that slipped "
                  f"normalization) -- do not proceed until resolved")

    out = HERE / "gated_v8.jsonl"
    with out.open("w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")

    print(f"kept {len(kept)} rows -> {out}")
    print(f"dropped: {dict(dropped)}")
    print("per-family generator-label counts (pre-blind-relabel, for reference only):")
    for fam in sorted(fam_counts):
        print(f"  {fam}: n={sum(fam_counts[fam].values())} {dict(sorted(fam_counts[fam].items()))}")


if __name__ == "__main__":
    main()
