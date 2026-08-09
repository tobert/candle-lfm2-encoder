#!/usr/bin/env python3
"""v7 relabel driver: 3-family blind fresh labels -> vote-proportional targets.

Flow (see README.md):
  pilot           emit blind chunks for the 58-row recoverability gold set
  score-pilot     reviewer verdicts vs adjudicated gold (the rubric gate)
  build           emit blind chunks for the train pool + eval sets
  apply           3 reviewer verdict files -> pool_v7.jsonl + eval_v7/<set>.jsonl
  splits          form-tagged pool_v7 -> kube_train_v7 / kube_val_v7 / nl_v7

Blind means blind: chunks carry {"id","text"} ONLY — no current label, no
group membership. Verdicts are {"id","label"} on the v7 axis.
Targets are vote-proportional (R3: disagreement is measured, not declared);
the reported label is the ordinal median of the three votes.
Prints aggregates only — never a raw row (CLAUDE.md data policy).

Usage:
  python3 relabel.py pilot <chunk_rows>
  python3 relabel.py score-pilot <name>=<verdicts.jsonl> [...]
  python3 relabel.py build <chunk_rows>
  python3 relabel.py apply <name>=<verdicts.jsonl> x3
  python3 relabel.py splits
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kube_pool import DATA, V6_TO_V7, load_eval_sets, load_train_pool, row_id  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = DATA / "relabel_v7"
V7_LABELS = ["informative", "situation-normal", "data-critical"]  # ordinal, fixed
V7_INDEX = {l: i for i, l in enumerate(V7_LABELS)}
RECOV_CASES = HERE.parent / "recoverability" / "cases.json"
SEED = 42


def write_chunks(rows: list[dict], chunk_rows: int, prefix: str) -> None:
    chunks = [rows[i:i + chunk_rows] for i in range(0, len(rows), chunk_rows)]
    for n, chunk in enumerate(chunks, 1):
        with (HERE / f"tmp_{prefix}_{n}.jsonl").open("w") as fh:
            for r in chunk:
                fh.write(json.dumps({"id": r["id"], "text": r["text"]}) + "\n")
    print(f"{prefix}: {len(rows)} rows -> {len(chunks)} chunks of <= {chunk_rows} "
          f"in {HERE}/tmp_{prefix}_N.jsonl")


def read_verdicts(args: list[str]) -> dict[str, dict[str, str]]:
    """`name=path` args -> {reviewer: {id: label}}; loud on bad labels/dups."""
    out = {}
    for a in args:
        if "=" not in a:
            sys.exit(f"expected name=verdicts.jsonl, got {a!r}")
        name, path = a.split("=", 1)
        verdicts = {}
        for lineno, line in enumerate(open(path), 1):
            line = line.strip()
            if not line:
                continue
            v = json.loads(line)
            if v.get("label") not in V7_INDEX:
                sys.exit(f"{path}:{lineno}: bad label {v.get('label')!r}")
            if v["id"] in verdicts and verdicts[v["id"]] != v["label"]:
                sys.exit(f"{path}:{lineno}: id {v['id']} has conflicting verdicts")
            verdicts[v["id"]] = v["label"]
        if name in out:
            sys.exit(f"duplicate reviewer name {name!r}")
        out[name] = verdicts
        print(f"{name}: {len(verdicts)} verdicts, "
              f"marginal {dict(sorted(Counter(verdicts.values()).items()))}")
    return out


def fold_votes(ids: list[str], reviewers: dict[str, dict[str, str]]) -> dict[str, dict]:
    """Per id: votes, ordinal-median label, vote-proportional target."""
    known = set(ids)
    for name, verdicts in reviewers.items():
        unknown = set(verdicts) - known
        if unknown:
            # a model mutating an id (observed: one-hex-digit typo) must
            # surface as BOTH an unknown id and a gap, never silently
            sys.exit(f"FAIL: {name} has {len(unknown)} verdict ids not in the "
                     f"manifest (e.g. {sorted(unknown)[:3]}) — likely id "
                     f"transcription errors; fix or re-run those chunks")
    missing = defaultdict(list)
    for name, verdicts in reviewers.items():
        for i in ids:
            if i not in verdicts:
                missing[name].append(i)
    if missing:
        detail = ", ".join(f"{n}: {len(ids)}" for n, ids in missing.items())
        sys.exit(f"FAIL: missing verdicts ({detail}) — every reviewer must "
                 f"cover every id; re-run the gaps before folding")
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
            # full precision — rounding to 6 places made a 3-way split's
            # thirds sum to 0.999999, tripping the trainer's (correct)
            # 1e-6 target-sum gate
            "target": dict(target),
            "votes": votes,
            "contested": len(set(votes.values())) > 1,
        }
    return folded


def vote_stats(folded: dict[str, dict], group: str) -> None:
    n = len(folded)
    unanimous = sum(1 for f in folded.values() if not f["contested"])
    three_way = sum(1 for f in folded.values()
                    if len(set(f["votes"].values())) == 3)
    print(f"{group}: n={n}  labels={dict(sorted(Counter(f['label'] for f in folded.values()).items()))}  "
          f"unanimous={unanimous} ({unanimous/n:.1%})  2-1={n-unanimous-three_way}  "
          f"3-way={three_way}")


def cmd_pilot(chunk_rows: int) -> None:
    cases = json.loads(RECOV_CASES.read_text())
    rows = [{"id": row_id(c["text"]), "text": c["text"]} for c in cases]
    if len({r["id"] for r in rows}) != len(rows):
        sys.exit("FAIL: id collision in pilot set")
    write_chunks(rows, chunk_rows, "pilot")


def cmd_score_pilot(args: list[str]) -> None:
    cases = json.loads(RECOV_CASES.read_text())
    gold = {row_id(c["text"]): c for c in cases if c.get("gold")}
    print(f"gold rows: {len(gold)} of {len(cases)}")
    reviewers = read_verdicts(args)
    for name, verdicts in reviewers.items():
        hit = sum(1 for i, c in gold.items() if verdicts.get(i) == c["gold"])
        cov = sum(1 for i in gold if i in verdicts)
        conf = Counter((gold[i]["gold"], verdicts[i]) for i in gold if i in verdicts
                       and verdicts[i] != gold[i]["gold"])
        print(f"{name}: {hit}/{cov} = {hit/cov:.1%} vs gold" if cov else f"{name}: 0 coverage")
        for (g, v), c in conf.most_common(6):
            print(f"    gold {g} -> voted {v}: {c}")
    if len(reviewers) >= 2:
        ids = [i for i in gold if all(i in v for v in reviewers.values())]
        folded = fold_votes(ids, reviewers)
        med_hit = sum(1 for i in ids if folded[i]["label"] == gold[i]["gold"])
        print(f"ordinal-median of all reviewers: {med_hit}/{len(ids)} = "
              f"{med_hit/len(ids):.1%}")
        vote_stats(folded, "pilot")


def cmd_build(chunk_rows: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = load_train_pool()
    evals = load_eval_sets()
    manifest, rows, seen = {}, [], set()
    for r in pool:
        manifest[r["id"]] = {"group": "pool", "source": r["source"],
                             "old_label": r["label"]}
        rows.append(r)
        seen.add(r["id"])
    for name, ers in evals.items():
        for r in ers:
            if r["id"] not in manifest:
                manifest[r["id"]] = {"group": name, "source": name,
                                     "old_label": r["label"]}
            else:
                manifest[r["id"]].setdefault("also", []).append(name)
            if r["id"] not in seen:
                seen.add(r["id"])
                rows.append(r)
    (OUT / "manifest.json").write_text(json.dumps(manifest))
    rng = random.Random(SEED)
    rng.shuffle(rows)  # blind: no group ordering signal in the chunks
    write_chunks(rows, chunk_rows, "relabel")
    groups = Counter(m["group"] for m in manifest.values())
    print(f"manifest: {len(manifest)} unique ids  groups={dict(sorted(groups.items()))}")


def cmd_apply(args: list[str]) -> None:
    manifest = json.loads((OUT / "manifest.json").read_text())
    reviewers = read_verdicts(args)
    if len(reviewers) != 3:
        sys.exit(f"FAIL: expected exactly 3 reviewers, got {len(reviewers)}")
    folded = fold_votes(list(manifest), reviewers)

    # rebuild text by id from the sources (chunks are ephemeral; sources are truth)
    texts = {}
    for r in load_train_pool():
        texts[r["id"]] = r
    for name, ers in load_eval_sets().items():
        for r in ers:
            texts.setdefault(r["id"], r)

    by_group = defaultdict(list)
    for i, meta in manifest.items():
        if i not in texts:
            sys.exit(f"FAIL: manifest id {i} no longer resolves to a source row")
        f = folded[i]
        row = {"id": i, "text": texts[i]["text"], "label": f["label"],
               "target": f["target"], "votes": f["votes"],
               "contested": f["contested"], "source": meta["source"]}
        by_group[meta["group"]].append((meta, row))

    (OUT / "eval_v7").mkdir(exist_ok=True)
    for group, items in sorted(by_group.items()):
        dest = OUT / ("pool_v7.jsonl" if group == "pool"
                      else f"eval_v7/{group}.jsonl")
        with dest.open("w") as fh:
            for _, row in items:
                fh.write(json.dumps(row) + "\n")
        folded_g = {row["id"]: row for _, row in items}
        vote_stats({i: f for i, f in folded.items() if i in folded_g}, group)
        mappable = [(meta, row) for meta, row in items
                    if meta["old_label"] in V6_TO_V7]
        flips = sum(1 for meta, row in mappable
                    if V6_TO_V7[meta["old_label"]] != row["label"])
        flip_note = (f"flips vs no-relabel mapping: {flips}/{len(mappable)} "
                     f"= {flips/len(mappable):.1%}" if mappable
                     else "flips: n/a (old labels not on the v6 vocabulary)")
        print(f"    {flip_note}  -> {dest}")


def cmd_splits() -> None:
    pool_path = OUT / "pool_v7.jsonl"
    rows = [json.loads(l) for l in open(pool_path) if l.strip()]
    if any("form" not in r for r in rows):
        n = sum(1 for r in rows if "form" not in r)
        sys.exit(f"FAIL: {n}/{len(rows)} pool rows missing 'form' — run "
                 f"form_tag.py tag --write on {pool_path} first")
    nl = [r for r in rows if r["form"] == "nl_whole"]
    shellish = [r for r in rows if r["form"] != "nl_whole"]
    with (OUT / "nl_v7.jsonl").open("w") as fh:
        for r in nl:
            fh.write(json.dumps(r) + "\n")
    print(f"NL set-aside (future NL head): {len(nl)} rows -> {OUT/'nl_v7.jsonl'}")

    rng = random.Random(SEED)
    strata = defaultdict(list)
    for r in shellish:
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
    for name, items in [("kube_train_v7", train), ("kube_val_v7", val)]:
        with (DATA / "prepared" / f"{name}.jsonl").open("w") as f:
            for r in items:
                f.write(json.dumps(r) + "\n")
        contested = sum(1 for r in items if r["contested"])
        print(f"{name}: {len(items)} rows  "
              f"labels={dict(sorted(Counter(r['label'] for r in items).items()))}  "
              f"contested={contested} ({contested/len(items):.1%})")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "pilot":
        cmd_pilot(int(args[0]))
    elif cmd == "score-pilot":
        cmd_score_pilot(args)
    elif cmd == "build":
        cmd_build(int(args[0]))
    elif cmd == "apply":
        cmd_apply(args)
    elif cmd == "splits":
        cmd_splits()
    else:
        sys.exit(f"unknown command {cmd!r}\n{__doc__}")


if __name__ == "__main__":
    main()
