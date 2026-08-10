#!/usr/bin/env python3
"""Blind chunk builder for v8 coverage-batch labeling, mirroring
relabel_v7/relabel.py's cmd_build: shuffle so no family/label signal rides
along, write {"id","text"} only. Manifest (id -> family/gen_label) kept
locally, never sent to a labeler.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 42
CHUNK_ROWS = 100


def main() -> None:
    rows = [json.loads(l) for l in open(HERE / "gated_v8.jsonl") if l.strip()]
    manifest = {r["id"]: {"family": r["family"], "gen_label": r["gen_label"],
                           "author": r["author"]} for r in rows}
    (HERE / "manifest_v8.json").write_text(json.dumps(manifest))

    rng = random.Random(SEED)
    rng.shuffle(rows)
    chunks = [rows[i:i + CHUNK_ROWS] for i in range(0, len(rows), CHUNK_ROWS)]
    (HERE / "chunks").mkdir(exist_ok=True)
    for n, chunk in enumerate(chunks, 1):
        with (HERE / f"chunks/tmp_v8_{n}.jsonl").open("w") as fh:
            for r in chunk:
                fh.write(json.dumps({"id": r["id"], "text": r["text"]}) + "\n")
    print(f"{len(rows)} rows -> {len(chunks)} blind chunks of <= {CHUNK_ROWS} "
          f"in {HERE}/chunks/tmp_v8_N.jsonl; manifest -> {HERE}/manifest_v8.json")


if __name__ == "__main__":
    main()
