#!/usr/bin/env python3
"""Evaluate an exported kube ordinal classifier on kube_test.jsonl.

Reports aggregates only: overall/per-label/per-slice/contested accuracy,
full confusion matrix, and the ordinal severity of errors. Never prints
row text. Misclassified rows are referenced by test-file line number.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))

from finetune_sequence_classifier import (  # noqa: E402
    Lfm2ForSequenceClassification,
    dot_free_checkpoint_dir,
    load_backbone,
)
from transformers import AutoTokenizer  # noqa: E402

SEVERITY = {"informative": 0, "mutating": 1, "destructive": 2}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path, help="exported classifier dir")
    p.add_argument("--base", required=True, type=Path, help="base checkpoint dir")
    p.add_argument("--test", required=True, type=Path, help="test JSONL")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=256)
    args = p.parse_args()

    config = json.loads((args.ckpt / "config.json").read_text())
    id2label = {int(k): v for k, v in config["id2label"].items()}
    label2id = {v: k for k, v in id2label.items()}

    base_dir = dot_free_checkpoint_dir(args.base)
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir))
    backbone, base_config = load_backbone(base_dir)
    model = Lfm2ForSequenceClassification(backbone, base_config.hidden_size, len(id2label))
    state = load_file(str(args.ckpt / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=True), None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    rows = [json.loads(l) for l in open(args.test)]
    print(f"test rows: {len(rows)}  labels: {sorted(label2id)}")

    preds = []
    with torch.no_grad():
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            enc = tokenizer(
                [r["text"] for r in batch], padding=True, truncation=True,
                max_length=args.max_len, return_tensors="pt",
            ).to(device)
            logits = model(enc["input_ids"], enc["attention_mask"])
            preds.extend(logits.float().argmax(-1).tolist())

    total = correct = 0
    by_slice = defaultdict(lambda: [0, 0])
    by_contested = defaultdict(lambda: [0, 0])
    confusion = Counter()
    sev_dist = Counter()
    wrong_refs = []
    for lineno, (r, p_id) in enumerate(zip(rows, preds), 1):
        gold, pred = r["label"], id2label[p_id]
        ok = gold == pred
        total += 1
        correct += ok
        by_slice[r.get("slice", "?")][ok] += 1
        by_contested[r.get("contested", False)][ok] += 1
        confusion[(gold, pred)] += 1
        dist = abs(SEVERITY[gold] - SEVERITY[pred])
        sev_dist[dist] += 1
        if not ok:
            wrong_refs.append(
                f"line {lineno}: {r.get('slice')}/{r.get('verb')} {r.get('resource')} "
                f"gold={gold} pred={pred}" + (" (contested)" if r.get("contested") else "")
            )

    print(f"\noverall accuracy: {correct}/{total} = {correct/total:.1%}")
    print("\nper-slice:")
    for s in sorted(by_slice):
        w, c = by_slice[s]
        print(f"  {s:>10s}: {c}/{c+w} = {c/(c+w):.1%}")
    print("contested split:")
    for k in sorted(by_contested):
        w, c = by_contested[k]
        name = "contested" if k else "clear"
        print(f"  {name:>10s}: {c}/{c+w} = {c/(c+w):.1%}")

    labels = sorted(label2id)
    print("\nconfusion (gold -> pred):")
    header = " ".join(f"{l[:6]:>7s}" for l in labels)
    print(f"  {'':>13s}{header}")
    for g in labels:
        cells = " ".join(f"{confusion[(g, p)]:>7d}" for p in labels)
        print(f"  {g:>13s}{cells}")

    print("\nerror severity distance (0=correct, 2=informative<->destructive):")
    for d in sorted(sev_dist):
        print(f"  distance {d}: {sev_dist[d]}")

    print(f"\nmisclassified rows ({len(wrong_refs)}) by metadata ref (no text):")
    for ref in wrong_refs:
        print(f"  {ref}")


if __name__ == "__main__":
    main()
