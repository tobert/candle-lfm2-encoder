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

import calib_metrics  # noqa: E402
from finetune_sequence_classifier import (  # noqa: E402
    Lfm2ForSequenceClassification,
    dot_free_checkpoint_dir,
    load_backbone,
)
from transformers import AutoTokenizer  # noqa: E402

# one ordinal map spanning both label vocabularies (v6 mutating/destructive,
# v7 situation-normal/data-critical) — same ranks, so mixed gold/pred pairs
# still measure distance; unknown labels fail loudly at the lookup
SEVERITY = {"informative": 0, "mutating": 1, "situation-normal": 1,
            "destructive": 2, "data-critical": 2}


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
    probs = []  # per-row softmax distribution, aligned with `preds`
    with torch.no_grad():
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            enc = tokenizer(
                [r["text"] for r in batch], padding=True, truncation=True,
                max_length=args.max_len, return_tensors="pt",
            ).to(device)
            logits = model(enc["input_ids"], enc["attention_mask"]).float()
            preds.extend(logits.argmax(-1).tolist())
            probs.extend(torch.softmax(logits, dim=-1).tolist())

    total = correct = 0
    by_slice = defaultdict(lambda: [0, 0])
    by_contested = defaultdict(lambda: [0, 0])
    confusion = Counter()
    sev_dist = Counter()
    wrong_refs = []

    # v7 calibration accumulators — only populated for rows carrying an
    # optional "target" distribution; rows without it never touch these,
    # so eval on legacy test files is unaffected.
    calib_total = 0
    unanimous_conf = []
    split_conf = []
    ece_confidences = []
    ece_correct = []  # top-1 label vs. argmax(target) ("majority label")
    ce_pairs = []  # (pred_dist, target_dist) for mean cross-entropy

    for lineno, (r, p_id, row_probs) in enumerate(zip(rows, preds, probs), 1):
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

        target = r.get("target")
        if target:
            calib_total += 1
            pred_dist = {id2label[j]: row_probs[j] for j in range(len(id2label))}
            top1_prob = pred_dist[pred]
            majority = calib_metrics.argmax_label(target)

            if calib_metrics.is_unanimous(target):
                unanimous_conf.append(top1_prob)
            else:
                split_conf.append(top1_prob)

            ece_confidences.append(top1_prob)
            ece_correct.append(pred == majority)
            ce_pairs.append((pred_dist, target))

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

    if calib_total:
        n_unanimous, n_split = len(unanimous_conf), len(split_conf)
        print(f"\ncalibration (target-bearing rows: {calib_total}/{total}):")
        print(f"  unanimous: {n_unanimous}  split: {n_split}")

        if n_unanimous and n_split:
            sep = calib_metrics.separation(unanimous_conf, split_conf)
            print(f"  separation (unanimous top-1 mean - split top-1 mean): {sep:+.3f}")
        else:
            print("  separation: n/a (need >=1 unanimous and >=1 split row)")

        if n_split:
            oc = calib_metrics.overconfident_split_count(split_conf)
            print(
                f"  overconfident splits (top-1 >= {calib_metrics.OVERCONFIDENT_THRESHOLD:.2f}): "
                f"{oc}/{n_split} = {oc/n_split:.1%}"
            )
        else:
            print("  overconfident splits: n/a (no split rows)")

        ece = calib_metrics.expected_calibration_error(ece_confidences, ece_correct)
        print(f"  ECE ({calib_metrics.ECE_BINS} bins, top-1 vs majority label): {ece:.4f}")

        mce = calib_metrics.mean_cross_entropy(ce_pairs)
        print(f"  mean cross-entropy (pred dist vs target dist): {mce:.4f}")


if __name__ == "__main__":
    main()
