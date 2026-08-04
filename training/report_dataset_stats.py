#!/usr/bin/env python3
"""Print AGGREGATE stats for prepared command-safety JSONL files. Never
prints a raw row — only counts, ratios, and aggregates — so this is safe
to run and read even though the dataset itself never leaves
~/.local/share/lfm2-training-data (see CLAUDE.md: ".jsonl handled only by
scripts, and never read into an agent's context").

Computes, per file:
  - example count, label balance, tier distribution
  - distinct command "binaries" used (first token of each `&&`/`;`/`|`
    segment, after stripping env-var prefixes / sudo / a leading `cd ...
    &&`, with a path prefix reduced to its basename)
  - mean/median whitespace-ish token length of `text`
  - exact-duplicate rate (identical `text` strings)
  - near-duplicate rate: `text` normalized by replacing any token that
    matches a known name-pool value (from generate_command_dataset.py's
    pools) with a `<POOL>` placeholder, plus digit-runs with `<NUM>`, then
    counting how many rows collapse onto the same normalized signature.
    This is the honest measure of "did a bigger N buy new templates, or
    just more slot-fills of the same ones".

Usage:
    python report_dataset_stats.py --dir ~/.local/share/lfm2-training-data/prepared
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_command_dataset import _ALL_POOLS  # noqa: E402

_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S+\s+")
_DIGIT_RUN_RE = re.compile(r"\d+")


def build_pool_value_to_name() -> dict[str, str]:
    rev: dict[str, str] = {}
    for pool_name, values in _ALL_POOLS.items():
        for v in values:
            rev.setdefault(v, pool_name)
    return rev


POOL_LOOKUP = build_pool_value_to_name()


def strip_prefix_layers(segment: str) -> str:
    s = segment.strip()
    while True:
        m = _ENV_ASSIGN_RE.match(s)
        if m:
            s = s[m.end():].strip()
            continue
        if s.startswith("sudo "):
            s = s[len("sudo "):].strip()
            continue
        break
    return s


def extract_binaries(text: str) -> set[str]:
    """Split on shell-ish separators and pull each segment's leading
    command token (after prefix-stripping), reduced to a basename."""
    if text.lstrip().startswith("#"):
        # entirely a comment line -> no binary actually invoked
        return set()
    segments = re.split(r"&&|;|\|", text)
    out = set()
    for seg in segments:
        seg = strip_prefix_layers(seg)
        if not seg:
            continue
        if seg.startswith("cd "):
            # "cd DIR && real_cmd" already split by &&; a bare lone "cd x"
            # segment's binary is legitimately "cd"
            pass
        tok = seg.split()[0] if seg.split() else ""
        tok = tok.strip("'\"")
        if not tok:
            continue
        tok = tok.rsplit("/", 1)[-1]
        out.add(tok)
    return out


def normalize_for_near_dup(text: str) -> str:
    # drop a trailing "  # comment"
    text = re.sub(r"\s+#.*$", "", text)
    tokens = text.split()
    norm_tokens = []
    for tok in tokens:
        stripped = tok.strip("'\",")
        if stripped in POOL_LOOKUP:
            norm_tokens.append(f"<{POOL_LOOKUP[stripped]}>")
        elif _DIGIT_RUN_RE.search(stripped):
            norm_tokens.append(_DIGIT_RUN_RE.sub("<NUM>", stripped))
        else:
            norm_tokens.append(stripped.lower())
    return " ".join(norm_tokens)


def load(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def report_file(path: Path) -> None:
    rows = load(path)
    n = len(rows)
    if n == 0:
        print(f"{path.name}: EMPTY")
        return

    labels = Counter(r["label"] for r in rows)
    tiers = Counter(r.get("tier", "?") for r in rows)

    lengths = [len(r["text"].split()) for r in rows]
    mean_len = statistics.mean(lengths)
    median_len = statistics.median(lengths)

    all_binaries: set[str] = set()
    for r in rows:
        all_binaries |= extract_binaries(r["text"])

    texts = [r["text"] for r in rows]
    exact_unique = len(set(texts))
    exact_dup_rate = 1 - exact_unique / n

    near_sigs = [normalize_for_near_dup(t) for t in texts]
    near_unique = len(set(near_sigs))
    near_dup_rate = 1 - near_unique / n

    print(f"=== {path.name} ===")
    print(f"  n = {n}")
    print(f"  labels: {dict(labels)}  (safe={labels.get('safe',0)/n:.1%}, dangerous={labels.get('dangerous',0)/n:.1%})")
    print(f"  tiers:  {dict(tiers)}")
    print(f"  distinct command binaries: {len(all_binaries)}")
    print(f"  token-ish length: mean={mean_len:.1f} median={median_len:.1f}")
    print(f"  exact-duplicate rate:      {exact_dup_rate:.1%}  ({n - exact_unique} of {n} rows share text with another row)")
    print(f"  near-duplicate rate:       {near_dup_rate:.1%}  ({n - near_unique} of {n} rows share a normalized template)")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--glob", default="*.jsonl")
    args = p.parse_args()

    paths = sorted(args.dir.glob(args.glob))
    if not paths:
        raise SystemExit(f"no files matching {args.glob} in {args.dir}")
    for path in paths:
        report_file(path)


if __name__ == "__main__":
    main()
