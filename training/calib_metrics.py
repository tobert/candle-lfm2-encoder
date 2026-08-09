#!/usr/bin/env python3
"""Pure calibration metrics for the v7 ordinal safety classifier eval.

No torch/model/dataset dependency by design: every function here takes
already-computed probability dicts, confidence floats, or boolean
correctness flags, so the whole module is unit-testable with synthetic
data (see tests/test_calib_metrics.py) and stays importable in contexts
that never load a model.

Row shape these functions assume (produced by the caller, not this
module):
  - `target`: dict[label -> probability], vote-proportional, sums to 1.
  - `pred`: dict[label -> probability], the model's softmax output.
  - a row is "unanimous" when its target is (near) one-hot, else "split".

Ties in argmax_label are broken alphabetically by label name — a
deliberate, deterministic choice, not a claim that any label is "more
correct" than another on a tie.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence, Tuple

UNANIMOUS_THRESHOLD = 0.999
OVERCONFIDENT_THRESHOLD = 0.88
ECE_BINS = 10

# Numerical clamp only (float32 softmax can underflow to an exact 0.0 for
# a confidently-wrong prediction). This is not a silent correctness
# fallback: a target-class probability that's genuinely ~0 SHOULD produce
# a very large cross-entropy; the clamp just keeps that finite instead of
# turning one bad row into an inf that swallows the whole mean silently.
_CE_EPSILON = 1e-12


def is_unanimous(target: Mapping[str, float], threshold: float = UNANIMOUS_THRESHOLD) -> bool:
    """True if one class carries essentially all of the vote mass in `target`."""
    if not target:
        raise ValueError("target distribution is empty")
    return max(target.values()) >= threshold


def argmax_label(dist: Mapping[str, float]) -> str:
    """Label with the highest probability in `dist`; ties broken alphabetically."""
    if not dist:
        raise ValueError("distribution is empty")
    top = max(dist.values())
    winners = sorted(k for k, v in dist.items() if v == top)
    return winners[0]


def separation(unanimous_confidences: Sequence[float], split_confidences: Sequence[float]) -> float:
    """Mean top-1 confidence on unanimous rows minus mean top-1 confidence on split rows.

    Both sequences must be non-empty — this metric is undefined without at
    least one row of each kind, and returning a placeholder value would
    silently misreport a missing comparison as a real one. Callers should
    check counts before calling and report "n/a" themselves when either
    side is empty.
    """
    if not unanimous_confidences:
        raise ValueError("separation requires at least one unanimous-row confidence")
    if not split_confidences:
        raise ValueError("separation requires at least one split-row confidence")
    mean_unanimous = sum(unanimous_confidences) / len(unanimous_confidences)
    mean_split = sum(split_confidences) / len(split_confidences)
    return mean_unanimous - mean_split


def overconfident_split_count(
    split_confidences: Sequence[float], threshold: float = OVERCONFIDENT_THRESHOLD
) -> int:
    """Count of split (contested) rows the model answered with top-1 prob >= threshold."""
    return sum(1 for c in split_confidences if c >= threshold)


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = ECE_BINS
) -> float:
    """Equal-width-bin ECE over [0, 1]: sum_b (|bin_b| / n) * |acc(bin_b) - mean_conf(bin_b)|.

    `correct` should already encode whatever correctness definition the
    caller wants (e.g. top-1 prediction vs. majority label) — this
    function only bins and aggregates.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    if not confidences:
        raise ValueError("expected_calibration_error requires at least one row")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    n = len(confidences)
    bins: list[list[Tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for c, ok in zip(confidences, correct):
        if not (0.0 <= c <= 1.0):
            raise ValueError(f"confidence {c!r} is out of [0, 1]")
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append((c, ok))

    ece = 0.0
    for b in bins:
        if not b:
            continue
        bin_conf = sum(c for c, _ in b) / len(b)
        bin_acc = sum(1 for _, ok in b if ok) / len(b)
        ece += (len(b) / n) * abs(bin_acc - bin_conf)
    return ece


def cross_entropy(pred: Mapping[str, float], target: Mapping[str, float]) -> float:
    """-sum_k target[k] * log(pred[k]), over classes with nonzero target mass.

    For a one-hot target this reduces exactly to -log(pred[gold_class]).
    """
    if not target:
        raise ValueError("target distribution is empty")
    ce = 0.0
    for k, t in target.items():
        if t <= 0.0:
            continue
        p = pred.get(k, 0.0)
        ce -= t * math.log(max(p, _CE_EPSILON))
    return ce


def mean_cross_entropy(pairs: Sequence[Tuple[Mapping[str, float], Mapping[str, float]]]) -> float:
    """Mean cross_entropy(pred, target) over a sequence of (pred, target) dict pairs."""
    if not pairs:
        raise ValueError("mean_cross_entropy requires at least one (pred, target) pair")
    return sum(cross_entropy(pred, target) for pred, target in pairs) / len(pairs)
