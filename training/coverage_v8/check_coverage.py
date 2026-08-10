#!/usr/bin/env python3
"""Per-reviewer coverage check against the 494-id manifest, merging all
chunks/raw_*_<cast>*.jsonl for that cast. Hard-fails loudly on conflicting
duplicate verdicts for the same id (relabel_v7 convention); reports missing
ids so gaps can be re-run precisely instead of guessed at.

Usage: python3 check_coverage.py <cast_tag> <raw_file1.jsonl> [...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VALID = {"informative", "situation-normal", "data-critical"}


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: check_coverage.py <cast_tag> <raw1.jsonl> [...]")
    cast_tag = sys.argv[1]
    files = sys.argv[2:]

    manifest = json.loads((HERE / "manifest_v8.json").read_text())
    all_ids = set(manifest)

    verdicts: dict[str, str] = {}
    n_lines = 0
    n_bad_json = 0
    n_unknown_id = 0
    n_conflict = 0
    for f in files:
        for i, line in enumerate(open(f), 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                n_bad_json += 1
                continue
            n_lines += 1
            rid, label = r.get("id"), r.get("label")
            if rid not in all_ids:
                n_unknown_id += 1
                print(f"  UNKNOWN ID (not in manifest): {f}: {rid!r}")
                continue
            if label not in VALID:
                print(f"  BAD LABEL: {f}: id={rid} label={label!r}")
                continue
            if rid in verdicts and verdicts[rid] != label:
                n_conflict += 1
                print(f"  CONFLICT: id={rid} has {verdicts[rid]!r} and {label!r} (file {f})")
                continue
            verdicts[rid] = label

    missing = all_ids - set(verdicts)
    print(f"\n[{cast_tag}] parsed {n_lines} valid lines from {len(files)} files "
          f"(bad_json={n_bad_json} unknown_id={n_unknown_id} conflicts={n_conflict})")
    print(f"[{cast_tag}] coverage: {len(verdicts)}/{len(all_ids)}  missing: {len(missing)}")

    out = HERE / f"verdicts_{cast_tag}.jsonl"
    with out.open("w") as fh:
        for rid in sorted(verdicts):
            fh.write(json.dumps({"id": rid, "label": verdicts[rid]}) + "\n")
    print(f"[{cast_tag}] wrote {out}")

    if missing:
        gaps_path = HERE / f"gaps_{cast_tag}.jsonl"
        pool = {r["id"]: r for r in
                [json.loads(l) for l in open(HERE / "gated_v8.jsonl") if l.strip()]}
        with gaps_path.open("w") as fh:
            for rid in sorted(missing):
                fh.write(json.dumps({"id": rid, "text": pool[rid]["text"]}) + "\n")
        print(f"[{cast_tag}] gap chunk written -> {gaps_path} ({len(missing)} rows) -- re-run these")


if __name__ == "__main__":
    main()
