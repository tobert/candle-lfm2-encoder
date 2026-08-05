#!/usr/bin/env python3
"""Corpus label recheck: luna reviews every haiku label, GLM rules on unsure.

build: emit {"id","text","label"} chunks (repo tmp files, gitignored) for
       the reviewing consults, plus a manifest in the recheck dir.
apply: join one or more verdict files ({"id","verdict"}; verdict is
       "agree", a corrected label, or "unsure") against the manifest;
       print aggregate verdict counts per source file; write the
       unsure set as a new chunk for the second-opinion pass; patch
       corrected labels into the SOURCE files (contested=true + note).

Prints aggregates only — never a raw row (CLAUDE.md data policy).

Usage:
    python recheck_labels.py build <chunk_rows> <src.jsonl> [...]
    python recheck_labels.py apply <reviewer> <verdicts.jsonl> [...]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

VALID = {"informative", "mutating", "destructive"}
DATA = Path.home() / ".local/share/lfm2-training-data"
RECHECK = DATA / "recheck"
REPO_TMP = Path(__file__).resolve().parent


def build(chunk_rows: int, files: list[Path]) -> None:
    RECHECK.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({"id": f"{f.stem}:{i}", "text": r["text"], "label": r["label"]})
    manifest = {}
    for f in files:
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if line.strip():
                manifest[f"{f.stem}:{i}"] = str(f)
    (RECHECK / "manifest.json").write_text(json.dumps(manifest))
    chunks = [rows[i:i + chunk_rows] for i in range(0, len(rows), chunk_rows)]
    for n, chunk in enumerate(chunks, 1):
        out = REPO_TMP / f"tmp_recheck_{n}.jsonl"
        with out.open("w") as fh:
            for r in chunk:
                fh.write(json.dumps(r) + "\n")
    print(f"build: {len(rows)} rows from {len(files)} files -> "
          f"{len(chunks)} chunks of <= {chunk_rows} in {REPO_TMP}/tmp_recheck_N.jsonl")


def apply(reviewer: str, verdict_files: list[Path]) -> None:
    manifest = json.loads((RECHECK / "manifest.json").read_text())
    verdicts = {}
    bad = 0
    for vf in verdict_files:
        for line in vf.read_text().splitlines():
            if not line.strip():
                continue
            v = json.loads(line)
            if v.get("verdict") in VALID | {"agree", "unsure"} and v.get("id") in manifest:
                verdicts[v["id"]] = v["verdict"]
            else:
                bad += 1

    per_file: dict[str, Counter] = defaultdict(Counter)
    corrections: dict[str, list[tuple[int, str]]] = defaultdict(list)
    unsure_ids = []
    for rid, verdict in verdicts.items():
        src = manifest[rid]
        stem = Path(src).stem
        if verdict == "agree":
            per_file[stem]["agree"] += 1
        elif verdict == "unsure":
            per_file[stem]["unsure"] += 1
            unsure_ids.append(rid)
        else:
            per_file[stem]["corrected"] += 1
            corrections[src].append((int(rid.rsplit(":", 1)[1]), verdict))

    print(f"apply({reviewer}): {len(verdicts)} verdicts, {bad} malformed/unknown, "
          f"{len(manifest) - len(verdicts)} rows missing a verdict")
    for stem in sorted(per_file):
        c = per_file[stem]
        print(f"  {stem}: agree={c['agree']} corrected={c['corrected']} unsure={c['unsure']}")

    label_moves: Counter = Counter()
    for src, fixes in corrections.items():
        path = Path(src)
        lines = path.read_text().splitlines()
        for lineno, new_label in fixes:
            r = json.loads(lines[lineno - 1])
            old = r["label"]
            if old == new_label:
                continue
            r["label"] = new_label
            r["contested"] = True
            note = r.get("note", "")
            r["note"] = f"{note} [recheck 2026-08-05 {reviewer}: {old}->{new_label}]".strip()
            lines[lineno - 1] = json.dumps(r)
            label_moves[f"{old}->{new_label}"] += 1
        path.write_text("\n".join(lines) + "\n")
    print("label moves:", dict(label_moves) or "none")

    if unsure_ids:
        texts = {}
        labels = {}
        for src in {manifest[rid] for rid in unsure_ids}:
            for i, line in enumerate(Path(src).read_text().splitlines(), 1):
                rid = f"{Path(src).stem}:{i}"
                if rid in set(unsure_ids) and line.strip():
                    r = json.loads(line)
                    texts[rid], labels[rid] = r["text"], r["label"]
        out = REPO_TMP / "tmp_recheck_unsure.jsonl"
        with out.open("w") as fh:
            for rid in unsure_ids:
                fh.write(json.dumps({"id": rid, "text": texts[rid],
                                     "label": labels[rid]}) + "\n")
        print(f"{len(unsure_ids)} unsure rows -> {out}")


def main() -> None:
    if len(sys.argv) < 4 or sys.argv[1] not in {"build", "apply"}:
        raise SystemExit(__doc__)
    if sys.argv[1] == "build":
        build(int(sys.argv[2]), [Path(a) for a in sys.argv[3:]])
    else:
        apply(sys.argv[2], [Path(a) for a in sys.argv[3:]])


if __name__ == "__main__":
    main()
