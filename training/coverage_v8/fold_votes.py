#!/usr/bin/env python3
"""Fold the three v8 blind verdict files into vote-proportional targets,
mirroring relabel_v7/relabel.py's fold_votes/vote_stats exactly (same
ordinal median, same target-fraction convention, same full-precision
target dict to avoid the 1e-6 sum-gate trip noted in v7's history).

Writes coverage_v8/coverage_v8.jsonl -- a NEW batch artifact, never merged
into pool_v7 or any split (out of scope for this run). Prints aggregates
only per family and per annotator; never a raw row.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
V7_LABELS = ["informative", "situation-normal", "data-critical"]
V7_INDEX = {l: i for i, l in enumerate(V7_LABELS)}


def read_verdicts(path: Path) -> dict[str, str]:
    out = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[r["id"]] = r["label"]
    return out


def main() -> None:
    reviewers = {
        "ds": read_verdicts(HERE / "verdicts_ds.jsonl"),
        "gm": read_verdicts(HERE / "verdicts_gm.jsonl"),
        "glm": read_verdicts(HERE / "verdicts_glm.jsonl"),
    }
    for name, v in reviewers.items():
        print(f"{name}: {len(v)} verdicts, marginal {dict(sorted(Counter(v.values()).items()))}")

    gated = {r["id"]: r for r in
             [json.loads(l) for l in open(HERE / "gated_v8.jsonl") if l.strip()]}
    ids = list(gated)

    known = set(ids)
    for name, v in reviewers.items():
        unknown = set(v) - known
        if unknown:
            sys.exit(f"FAIL: {name} has {len(unknown)} verdict ids not in gated_v8 "
                      f"(e.g. {sorted(unknown)[:3]})")
    missing = {name: [i for i in ids if i not in v] for name, v in reviewers.items()}
    missing = {n: m for n, m in missing.items() if m}
    if missing:
        detail = ", ".join(f"{n}: {len(m)}" for n, m in missing.items())
        sys.exit(f"FAIL: missing verdicts ({detail}) -- every reviewer must cover every id")

    folded = {}
    for i in ids:
        votes = {name: reviewers[name][i] for name in sorted(reviewers)}
        idxs = sorted(V7_INDEX[l] for l in votes.values())
        label = V7_LABELS[idxs[len(idxs) // 2]]
        target = {l: 0.0 for l in V7_LABELS}
        for l in votes.values():
            target[l] += 1.0 / len(votes)
        folded[i] = {
            "label": label,
            "target": dict(target),
            "votes": votes,
            "contested": len(set(votes.values())) > 1,
        }

    n = len(folded)
    unanimous = sum(1 for f in folded.values() if not f["contested"])
    three_way = sum(1 for f in folded.values() if len(set(f["votes"].values())) == 3)
    two_one = n - unanimous - three_way
    print(f"\noverall: n={n}  labels={dict(sorted(Counter(f['label'] for f in folded.values()).items()))}")
    print(f"  unanimous={unanimous} ({unanimous/n:.1%})  2-1={two_one} ({two_one/n:.1%})  "
          f"3-way={three_way} ({three_way/n:.1%})")

    # per-family breakdown
    by_family = defaultdict(list)
    for i in ids:
        by_family[gated[i]["family"]].append(i)
    print("\nper-family folded-label marginals:")
    for fam in sorted(by_family):
        fam_ids = by_family[fam]
        n_f = len(fam_ids)
        labels_f = Counter(folded[i]["label"] for i in fam_ids)
        unan_f = sum(1 for i in fam_ids if not folded[i]["contested"])
        print(f"  {fam}: n={n_f}  labels={dict(sorted(labels_f.items()))}  "
              f"unanimous={unan_f} ({unan_f/n_f:.1%})")

    # generator-label vs folded-label flip rate per family
    print("\ngenerator-label vs folded-label agreement (informal QA on the generators):")
    for fam in sorted(by_family):
        fam_ids = by_family[fam]
        agree = sum(1 for i in fam_ids if gated[i]["gen_label"] == folded[i]["label"])
        print(f"  {fam}: generator label matched folded label {agree}/{len(fam_ids)} "
              f"({agree/len(fam_ids):.1%})")

    # per-annotator marginal already printed above; per-annotator agreement with folded label
    print("\nper-annotator agreement with the folded (ordinal-median) label:")
    for name in sorted(reviewers):
        agree = sum(1 for i in ids if reviewers[name][i] == folded[i]["label"])
        print(f"  {name}: {agree}/{n} ({agree/n:.1%})")

    out = HERE / "coverage_v8.jsonl"
    with out.open("w") as fh:
        for i in ids:
            row = {
                "id": i, "text": gated[i]["text"], "label": folded[i]["label"],
                "target": folded[i]["target"], "votes": folded[i]["votes"],
                "contested": folded[i]["contested"], "family": gated[i]["family"],
                "gen_label": gated[i]["gen_label"], "author": gated[i]["author"],
                "source_file": gated[i]["source_file"],
            }
            fh.write(json.dumps(row) + "\n")
    print(f"\nwrote {n} rows -> {out} (NEW artifact -- not merged into pool_v7 or any split)")


if __name__ == "__main__":
    main()
