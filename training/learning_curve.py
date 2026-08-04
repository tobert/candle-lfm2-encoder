#!/usr/bin/env python3
"""Learning-curve experiment: how much synthetic data does it take to get a
useful advisory commandline-safety classifier?

For each training-set size in --sizes, trains --seeds independent runs (same
hyperparameters, different data shuffle/init) using the same checkpoint-
loading and model-construction path as `finetune_sequence_classifier.py`
(imported from it directly — see that file and training/README.md for the
two loading gotchas this depends on getting right). Unlike that script, this
one does NOT export a checkpoint to disk: we only need accuracy numbers, and
the "best" checkpoint per run is a training the transient CPU state_dict
(cheap, no 1.4 GiB safetensors write). Nothing at any point is exported to
--out, only aggregate metrics are written as JSON.

Two evaluation sets are scored for every run, and BOTH are always reported
together, never separately:
  1. the held-out val split from the SAME generator as --train (optimistic
     — measures fitting the generator's own distribution).
  2. a hand-authored `eval_realistic.jsonl` in a deliberately different
     style (honest — measures generalisation). Records in this file carry a
     `tier` field (or one-hot `blatant`/`moderate`/`subtle`/`contested`
     boolean fields — both encodings are accepted). The GAP between (1) and
     (2), and the subtle-tier accuracy within (2), are the actual findings.

DATA HANDLING (strict, per the user's instructions):
  - This script never prints a "text" field from any dataset, in any log
    line, error message, or exception. Only aggregate counts/metrics leave
    stdout.
  - This script never executes any command string as a shell command,
    including the ones baked into the keyword-matcher baseline below (those
    are Python regex literals compared against DATA, never passed to a
    shell).
  - Checkpoints are never written to the repo; --runs-dir defaults outside
    the repo. Only a metrics JSON is written.

Baselines reported alongside the learning curve:
  - majority-class: always predict the most common training label.
  - keyword/regex matcher: ~dozen obvious dangerous-command patterns. If the
    fine-tune can't beat this on the subtle tier, that's the headline
    finding and this script says so.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_sequence_classifier import (  # noqa: E402
    Lfm2ForSequenceClassification,
    dot_free_checkpoint_dir,
    load_backbone,
    make_collate,
    set_seed,
)
from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

TIER_FIELDS = ("blatant", "moderate", "subtle", "contested")

# --------------------------------------------------------------------------
# Keyword/regex baseline — DATA ONLY, never executed as a shell command.
# A dozen obvious patterns a security-conscious grep-writer would reach for
# first. This is deliberately dumb: it's the bar the fine-tune must clear.
# --------------------------------------------------------------------------
DANGEROUS_PATTERNS = [
    re.compile(r"rm\s+-rf\s+/(\s|$)", re.IGNORECASE),
    re.compile(r"rm\s+-rf\s+--no-preserve-root", re.IGNORECASE),
    re.compile(r"dd\s+if=\S+\s+of=/dev/(sd|nvme|hd|xvd)", re.IGNORECASE),
    re.compile(r"mkfs\.\w+\s+/dev/", re.IGNORECASE),
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.IGNORECASE),
    re.compile(r"chmod\s+-R\s+777\s+/", re.IGNORECASE),
    re.compile(r"(curl|wget)[^\n]{0,120}\|\s*(sudo\s+)?(ba)?sh\b", re.IGNORECASE),
    re.compile(r">\s*/dev/(sd|nvme|hd|xvd)\w*", re.IGNORECASE),
    re.compile(r"iptables\s+-F\b", re.IGNORECASE),
    re.compile(r"\bdrop\s+table\b", re.IGNORECASE),
    re.compile(r"\bdelete\s+from\s+\w+\s*;?\s*$", re.IGNORECASE),
    re.compile(r"shutdown\s+-h\s+now\b", re.IGNORECASE),
    re.compile(r"\bnc\s+-e\b", re.IGNORECASE),
    re.compile(r"base64\s+-d\s*\|\s*(ba)?sh\b", re.IGNORECASE),
]


def keyword_matches(text: str) -> bool:
    return any(p.search(text) is not None for p in DANGEROUS_PATTERNS)


# --------------------------------------------------------------------------
# Data loading — realistic eval needs the tier field preserved, which the
# shared finetune_sequence_classifier.load_jsonl() deliberately drops (it
# only keeps text/label). Separate loader here.
# --------------------------------------------------------------------------


def load_split_jsonl(path: Path) -> list[dict]:
    """Like finetune_sequence_classifier.load_jsonl but keeps this module
    self-contained for the (text,label)-only splits (train/val)."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" not in obj or "label" not in obj:
                raise ValueError(
                    f"{path}:{lineno}: expected keys 'text' and 'label', got {sorted(obj)}"
                )
            if not isinstance(obj["label"], str):
                raise ValueError(f"{path}:{lineno}: label must be a string")
            items.append({"text": str(obj["text"]), "label": obj["label"]})
    if not items:
        raise ValueError(f"{path}: no examples found")
    return items


def resolve_tier(obj: dict, path: Path, lineno: int) -> str:
    if "tier" in obj:
        return str(obj["tier"])
    present = [k for k in TIER_FIELDS if k in obj]
    truthy = [k for k in present if bool(obj[k])]
    if len(truthy) == 1:
        return truthy[0]
    raise ValueError(
        f"{path}:{lineno}: could not resolve a tier — no 'tier' field and "
        f"one-hot fields {TIER_FIELDS} gave {len(truthy)} truthy values "
        "(expected exactly 1). Refusing to guess a tier; fix the data or "
        "this loader."
    )


def load_realistic_eval(path: Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" not in obj or "label" not in obj:
                raise ValueError(
                    f"{path}:{lineno}: expected keys 'text' and 'label', got {sorted(obj)}"
                )
            if not isinstance(obj["label"], str):
                raise ValueError(f"{path}:{lineno}: label must be a string")
            tier = resolve_tier(obj, path, lineno)
            items.append({"text": str(obj["text"]), "label": obj["label"], "tier": tier})
    if not items:
        raise ValueError(f"{path}: no examples found")
    return items


class TieredDataset(Dataset):
    def __init__(self, items: list[dict], label2id: dict[str, int]):
        self.items = items
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        return it["text"], self.label2id[it["label"]], it["tier"]


def make_tiered_collate(tokenizer, max_len: int):
    def collate(batch):
        texts, labels, tiers = zip(*batch)
        enc = tokenizer(
            list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        )
        return enc, torch.tensor(labels, dtype=torch.long), list(tiers)

    return collate


# --------------------------------------------------------------------------
# Baselines (no model — pure functions of the data)
# --------------------------------------------------------------------------


def majority_label(train_items: list[dict]) -> str:
    counts = Counter(it["label"] for it in train_items)
    return counts.most_common(1)[0][0]


def infer_dangerous_label(train_items: list[dict], labels_sorted: list[str]) -> str | None:
    """Data-driven pick of which label the keyword matcher's positive signal
    should map to: whichever label has the highest keyword-match rate among
    its own training examples. Returns None if the match rate is ~0 for all
    labels (keyword baseline degenerates to majority-only)."""
    match_rate = {}
    for label in labels_sorted:
        subset = [it for it in train_items if it["label"] == label]
        if not subset:
            continue
        match_rate[label] = sum(keyword_matches(it["text"]) for it in subset) / len(subset)
    if not match_rate:
        return None
    best_label, best_rate = max(match_rate.items(), key=lambda kv: kv[1])
    return best_label if best_rate > 0.0 else None


def evaluate_predictions(
    items: list[dict], preds: list[str], labels_sorted: list[str]
) -> dict:
    total = len(items)
    correct = sum(1 for it, p in zip(items, preds) if it["label"] == p)
    tier_correct: Counter = Counter()
    tier_total: Counter = Counter()
    confusion = {t: {p: 0 for p in labels_sorted} for t in labels_sorted}
    for it, p in zip(items, preds):
        gold = it["label"]
        confusion[gold][p] = confusion[gold].get(p, 0) + 1
        tier = it.get("tier")
        if tier is not None:
            tier_total[tier] += 1
            if gold == p:
                tier_correct[tier] += 1
    per_tier_acc = {
        t: (tier_correct[t] / tier_total[t] if tier_total[t] else None) for t in tier_total
    }
    return {
        "accuracy": correct / total if total else 0.0,
        "n": total,
        "per_tier_accuracy": per_tier_acc,
        "per_tier_support": dict(tier_total),
        "confusion": confusion,
    }


def run_baselines(
    train_pool: list[dict], val_items: list[dict], realistic_items: list[dict], labels_sorted: list[str]
) -> dict:
    maj = majority_label(train_pool)
    maj_val_preds = [maj] * len(val_items)
    maj_real_preds = [maj] * len(realistic_items)

    dangerous_label = infer_dangerous_label(train_pool, labels_sorted)
    other_labels = [l for l in labels_sorted if l != dangerous_label]
    default_label = majority_label([it for it in train_pool if it["label"] != dangerous_label]) \
        if dangerous_label and other_labels else maj

    def keyword_predict(text: str) -> str:
        if dangerous_label is not None and keyword_matches(text):
            return dangerous_label
        return default_label

    kw_val_preds = [keyword_predict(it["text"]) for it in val_items]
    kw_real_preds = [keyword_predict(it["text"]) for it in realistic_items]

    return {
        "majority_label": maj,
        "keyword_positive_label": dangerous_label,
        "keyword_default_label": default_label,
        "majority": {
            "val": evaluate_predictions(val_items, maj_val_preds, labels_sorted),
            "realistic": evaluate_predictions(realistic_items, maj_real_preds, labels_sorted),
        },
        "keyword": {
            "val": evaluate_predictions(val_items, kw_val_preds, labels_sorted),
            "realistic": evaluate_predictions(realistic_items, kw_real_preds, labels_sorted),
        },
    }


# --------------------------------------------------------------------------
# Training / evaluation for one (size, seed) run
# --------------------------------------------------------------------------


def epochs_for_size(n: int, pivot: int, min_epochs: int, max_epochs: int) -> int:
    return max(min_epochs, min(max_epochs, round(pivot / n)))


@torch.no_grad()
def eval_simple(model, loader, device, autocast_dtype) -> float:
    model.eval()
    correct = 0
    total = 0
    for enc, labels in loader:
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels = labels.to(device)
        with torch.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"
        ):
            logits = model(input_ids, attention_mask)
        preds = logits.float().argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / max(1, total)


@torch.no_grad()
def eval_tiered(model, loader, device, autocast_dtype, labels_sorted: list[str]) -> dict:
    model.eval()
    correct = 0
    total = 0
    tier_correct: Counter = Counter()
    tier_total: Counter = Counter()
    confusion = {t: {p: 0 for p in labels_sorted} for t in labels_sorted}
    for enc, labels, tiers in loader:
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels_dev = labels.to(device)
        with torch.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"
        ):
            logits = model(input_ids, attention_mask)
        preds = logits.float().argmax(dim=-1).cpu()
        correct += (preds == labels).sum().item()
        total += labels.numel()
        for pred_idx, gold_idx, tier in zip(preds.tolist(), labels.tolist(), tiers):
            gold = labels_sorted[gold_idx]
            pred = labels_sorted[pred_idx]
            confusion[gold][pred] = confusion[gold].get(pred, 0) + 1
            tier_total[tier] += 1
            if pred_idx == gold_idx:
                tier_correct[tier] += 1
    per_tier_acc = {t: (tier_correct[t] / tier_total[t] if tier_total[t] else None) for t in tier_total}
    return {
        "accuracy": correct / max(1, total),
        "n": total,
        "per_tier_accuracy": per_tier_acc,
        "per_tier_support": dict(tier_total),
        "confusion": confusion,
    }


def train_one_run(
    *,
    train_items: list[dict],
    val_items: list[dict],
    realistic_items: list[dict],
    base_dir: Path,
    tokenizer,
    device: torch.device,
    autocast_dtype,
    seed: int,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    lr: float,
    max_len: int,
    warmup_ratio: float,
) -> dict:
    set_seed(seed)

    labels_sorted = sorted({it["label"] for it in train_items})
    unseen_val = {it["label"] for it in val_items} - set(labels_sorted)
    if unseen_val:
        raise ValueError(f"val split has labels never seen in train: {sorted(unseen_val)}")
    unseen_real = {it["label"] for it in realistic_items} - set(labels_sorted)
    if unseen_real:
        raise ValueError(
            f"eval_realistic.jsonl has labels never seen in train: {sorted(unseen_real)}"
        )
    label2id = {label: i for i, label in enumerate(labels_sorted)}

    lfm2_backbone, base_config = load_backbone(base_dir)
    model = Lfm2ForSequenceClassification(
        lfm2_backbone, base_config.hidden_size, len(labels_sorted)
    ).to(device)

    collate = make_collate(tokenizer, max_len)
    tiered_collate = make_tiered_collate(tokenizer, max_len)

    from finetune_sequence_classifier import ClassificationDataset  # local import, no side effects

    train_ds = ClassificationDataset(train_items, label2id)
    val_ds = ClassificationDataset(val_items, label2id)
    real_ds = TieredDataset(realistic_items, label2id)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=generator
    )
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=collate)
    real_loader = DataLoader(
        real_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=tiered_collate
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    use_scaler = device.type == "cuda" and autocast_dtype == torch.float16
    scaler = torch.amp.GradScaler(device="cuda" if device.type == "cuda" else "cpu", enabled=use_scaler)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_val_acc = -1.0
    best_state = None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for enc, labels in train_loader:
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"
            ):
                logits = model(input_ids, attention_mask)
                loss = nn.functional.cross_entropy(logits, labels)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        val_acc = eval_simple(model, val_loader, device, autocast_dtype)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone().to("cpu") for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training produced no validation result")
    model.load_state_dict(best_state)
    model.to(device)

    real_metrics = eval_tiered(model, real_loader, device, autocast_dtype, labels_sorted)
    wall_s = time.time() - t0
    peak_mem_gib = (
        torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else None
    )

    del model, best_state, lfm2_backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "labels_sorted": labels_sorted,
        "epochs": epochs,
        "n_train": len(train_items),
        "n_val": len(val_items),
        "val_accuracy": best_val_acc,
        "realistic": real_metrics,
        "wall_s": wall_s,
        "peak_gpu_mem_gib": peak_mem_gib,
    }


# --------------------------------------------------------------------------
# Data discovery / waiting
# --------------------------------------------------------------------------


def expected_files(data_dir: Path, sizes: list[int]) -> dict[int, tuple[Path, Path]]:
    out = {}
    for n in sizes:
        out[n] = (data_dir / f"train_{n}.jsonl", data_dir / f"val_{n}.jsonl")
    return out


def wait_for_data(
    data_dir: Path, sizes: list[int], realistic_path: Path, interval_s: float, timeout_s: float
) -> None:
    files = expected_files(data_dir, sizes)
    t0 = time.time()
    attempt = 0
    while True:
        missing = []
        for n, (tr, va) in files.items():
            if not tr.is_file():
                missing.append(tr.name)
            if not va.is_file():
                missing.append(va.name)
        if not realistic_path.is_file():
            missing.append(realistic_path.name)
        if not missing:
            print(f"all data files present after {time.time() - t0:.0f}s", flush=True)
            return
        elapsed = time.time() - t0
        if elapsed > timeout_s:
            raise TimeoutError(
                f"timed out after {elapsed:.0f}s waiting for data files; still missing: {missing}"
            )
        attempt += 1
        print(
            f"[wait] attempt {attempt}, elapsed {elapsed:.0f}s — still missing {len(missing)} "
            f"file(s): {missing}",
            flush=True,
        )
        time.sleep(interval_s)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def print_curve_table(per_size_summary: list[dict]) -> None:
    header = f"{'n_train':>8} {'epochs':>7} {'val_acc (mean±sd)':>20} {'realistic_acc (mean±sd)':>24} {'gap':>8}"
    print(header)
    print("-" * len(header))
    for row in per_size_summary:
        val_s = f"{row['val_mean']:.3f}±{row['val_std']:.3f}"
        real_s = f"{row['real_mean']:.3f}±{row['real_std']:.3f}"
        gap = row["val_mean"] - row["real_mean"]
        print(
            f"{row['n_train']:>8} {row['epochs']:>7} {val_s:>20} {real_s:>24} {gap:>8.3f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("/home/atobey/.local/share/lfm2-training-data/prepared"))
    p.add_argument("--realistic-file", type=Path, default=None, help="default: <data-dir>/eval_realistic.jsonl")
    p.add_argument("--base", type=Path, default=Path("/home/atobey/src/candle-lfm2-encoder/.models/LFM2.5-Encoder-350M"))
    p.add_argument("--runs-dir", type=Path, default=Path("/home/atobey/.local/share/lfm2-training-data/runs"))
    p.add_argument("--out-name", type=str, default="learning_curve.json")
    p.add_argument("--sizes", type=int, nargs="+", default=[100, 250, 500, 1000, 2000, 4000])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--epoch-pivot", type=int, default=1000, help="epochs ~= pivot / n_train, clipped to [min,max]")
    p.add_argument("--epoch-min", type=int, default=3)
    p.add_argument("--epoch-max", type=int, default=10)
    p.add_argument("--wait", action="store_true", help="poll --data-dir until all expected files exist")
    p.add_argument("--wait-interval", type=float, default=30.0)
    p.add_argument("--wait-timeout", type=float, default=6 * 3600.0)
    p.add_argument("--force", action="store_true", help="ignore any existing results JSON and retrain everything")
    args = p.parse_args()

    realistic_path = args.realistic_file or (args.data_dir / "eval_realistic.jsonl")

    if args.wait:
        wait_for_data(args.data_dir, args.sizes, realistic_path, args.wait_interval, args.wait_timeout)
    else:
        files = expected_files(args.data_dir, args.sizes)
        missing = [
            str(pth)
            for n, (tr, va) in files.items()
            for pth in (tr, va)
            if not pth.is_file()
        ] + ([str(realistic_path)] if not realistic_path.is_file() else [])
        if missing:
            raise FileNotFoundError(
                f"missing expected data file(s) (pass --wait to poll instead): {missing}"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
    elif device.type == "cuda":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32
    print(f"device={device}  autocast_dtype={autocast_dtype}", flush=True)

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.runs_dir / args.out_name

    results: dict = {"runs": {}}
    if out_path.is_file() and not args.force:
        try:
            results = json.loads(out_path.read_text())
            print(f"resuming from existing {out_path} ({len(results.get('runs', {}))} runs already recorded)")
        except json.JSONDecodeError:
            print(f"existing {out_path} is not valid JSON, starting fresh")
            results = {"runs": {}}

    base_dir = dot_free_checkpoint_dir(args.base)
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir))

    realistic_items = load_realistic_eval(realistic_path)
    tier_counts = Counter(it["tier"] for it in realistic_items)
    label_counts_real = Counter(it["label"] for it in realistic_items)
    print(f"eval_realistic: n={len(realistic_items)}  tiers={dict(tier_counts)}  labels={dict(label_counts_real)}")

    sweep_t0 = time.time()
    files = expected_files(args.data_dir, args.sizes)

    largest_n = max(args.sizes)
    train_pool = load_split_jsonl(files[largest_n][0])
    val_pool_largest = load_split_jsonl(files[largest_n][1])
    labels_sorted_pool = sorted({it["label"] for it in train_pool})

    for n in args.sizes:
        train_path, val_path = files[n]
        train_items = load_split_jsonl(train_path)
        val_items = load_split_jsonl(val_path)
        n_epochs = epochs_for_size(n, args.epoch_pivot, args.epoch_min, args.epoch_max)
        label_counts = Counter(it["label"] for it in train_items)
        print(
            f"\n=== n_train={n} (file has {len(train_items)}), epochs={n_epochs}, "
            f"val n={len(val_items)}, train label counts={dict(label_counts)} ==="
        )

        for seed in args.seeds:
            key = f"n{n}_seed{seed}"
            if key in results["runs"] and not args.force:
                print(f"[skip] {key} already recorded")
                continue
            run_t0 = time.time()
            metrics = train_one_run(
                train_items=train_items,
                val_items=val_items,
                realistic_items=realistic_items,
                base_dir=base_dir,
                tokenizer=tokenizer,
                device=device,
                autocast_dtype=autocast_dtype,
                seed=seed,
                epochs=n_epochs,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                lr=args.lr,
                max_len=args.max_len,
                warmup_ratio=args.warmup_ratio,
            )
            print(
                f"  [{key}] val_acc={metrics['val_accuracy']:.4f}  "
                f"realistic_acc={metrics['realistic']['accuracy']:.4f}  "
                f"wall_s={metrics['wall_s']:.1f}  peak_gpu_gib={metrics['peak_gpu_mem_gib']}"
            )
            results["runs"][key] = {"n_train": n, "seed": seed, **metrics}
            out_path.write_text(json.dumps(results, indent=2))

    # ---------------- aggregation ----------------
    per_size_summary = []
    for n in args.sizes:
        val_accs = [r["val_accuracy"] for r in results["runs"].values() if r["n_train"] == n]
        real_accs = [r["realistic"]["accuracy"] for r in results["runs"].values() if r["n_train"] == n]
        epochs_used = next(r["epochs"] for r in results["runs"].values() if r["n_train"] == n)
        val_mean, val_std = mean_std(val_accs)
        real_mean, real_std = mean_std(real_accs)
        per_size_summary.append(
            {
                "n_train": n,
                "epochs": epochs_used,
                "val_mean": val_mean,
                "val_std": val_std,
                "real_mean": real_mean,
                "real_std": real_std,
                "n_seeds": len(val_accs),
            }
        )

    # per-tier breakdown at the largest size, averaged across seeds
    largest_runs = [r for r in results["runs"].values() if r["n_train"] == largest_n]
    tier_keys = sorted(tier_counts.keys())
    tier_summary = {}
    for t in tier_keys:
        vals = [
            r["realistic"]["per_tier_accuracy"].get(t)
            for r in largest_runs
            if r["realistic"]["per_tier_accuracy"].get(t) is not None
        ]
        if vals:
            m, s = mean_std(vals)
            tier_summary[t] = {"mean": m, "std": s, "n_seeds": len(vals), "support": tier_counts[t]}

    baselines = run_baselines(train_pool, val_pool_largest, realistic_items, labels_sorted_pool)

    total_wall_s = sum(r["wall_s"] for r in results["runs"].values())
    peak_mem_values = [r["peak_gpu_mem_gib"] for r in results["runs"].values() if r["peak_gpu_mem_gib"] is not None]
    peak_mem_overall = max(peak_mem_values) if peak_mem_values else None

    results["summary"] = {
        "per_size": per_size_summary,
        "tier_at_largest_n": {"n_train": largest_n, "tiers": tier_summary},
        "baselines": baselines,
        "total_wall_s_this_invocation": time.time() - sweep_t0,
        "total_wall_s_all_runs": total_wall_s,
        "peak_gpu_mem_gib": peak_mem_overall,
        "hyperparameters": {
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "lr": args.lr,
            "max_len": args.max_len,
            "warmup_ratio": args.warmup_ratio,
            "epoch_pivot": args.epoch_pivot,
            "epoch_min": args.epoch_min,
            "epoch_max": args.epoch_max,
            "seeds": args.seeds,
            "sizes": args.sizes,
        },
    }
    out_path.write_text(json.dumps(results, indent=2))

    print("\n================ LEARNING CURVE ================")
    print_curve_table(per_size_summary)

    print(f"\n================ TIER BREAKDOWN at n_train={largest_n} (mean over {len(largest_runs)} seeds) ================")
    for t, s in tier_summary.items():
        print(f"  {t:>10}  acc={s['mean']:.3f}±{s['std']:.3f}  support={s['support']}")

    print("\n================ BASELINES ================")
    maj = baselines["majority"]
    kw = baselines["keyword"]
    print(f"majority-class label = {baselines['majority_label']!r}")
    print(f"  val:       acc={maj['val']['accuracy']:.3f}")
    print(f"  realistic: acc={maj['realistic']['accuracy']:.3f}  per_tier={maj['realistic']['per_tier_accuracy']}")
    print(
        f"keyword matcher: positive_label={baselines['keyword_positive_label']!r} "
        f"default_label={baselines['keyword_default_label']!r}"
    )
    print(f"  val:       acc={kw['val']['accuracy']:.3f}")
    print(f"  realistic: acc={kw['realistic']['accuracy']:.3f}  per_tier={kw['realistic']['per_tier_accuracy']}")

    print(f"\ntotal wall time (this invocation): {results['summary']['total_wall_s_this_invocation']:.1f}s")
    print(f"total wall time (sum of all recorded runs): {total_wall_s:.1f}s")
    print(f"peak GPU memory across all runs: {peak_mem_overall} GiB")
    print(f"\nfull metrics JSON written to {out_path}")


if __name__ == "__main__":
    main()
