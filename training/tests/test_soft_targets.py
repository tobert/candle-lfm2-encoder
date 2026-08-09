"""Tests for the v7 soft-target (calibration-aware) training path added to
`finetune_sequence_classifier.py`:

- optional per-row `"target"` distribution on top of the existing `"label"`
- validation: target keys subset of the label set, values sum to 1 ± 1e-6
- soft cross-entropy loss, numerically equal to hard CE when every target
  is one-hot
- explicit (non-derived) label ordering via `resolve_label_order`

Run with: `python3 -m unittest discover training/tests` from the repo's
`training/` directory (or point discovery at this dir explicitly).

Stdlib `unittest` only. Uses `torch` directly with tiny synthetic tensors
— no model download, no GPU, no real corpus rows (synthetic strings only).
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _stub_missing_heavy_deps() -> None:
    """`finetune_sequence_classifier.py` imports `transformers` and
    `safetensors` at module level, even though the functions this test
    file exercises (JSONL/target validation, label ordering, the soft-CE
    loss, dataset/collate) are pure torch + stdlib and never touch either
    package. On a machine with only system `torch` (no project venv
    active), importing the module for these tests would otherwise fail
    for reasons unrelated to what's under test.

    Install a minimal stub only when the real package genuinely can't be
    imported, so this is a no-op (and the real symbols get exercised) on
    any machine where the actual training venv is active.
    """
    try:
        import safetensors.torch  # noqa: F401
    except ImportError:
        st = types.ModuleType("safetensors")
        st_torch = types.ModuleType("safetensors.torch")
        st_torch.save_file = lambda *a, **k: None
        st.torch = st_torch
        sys.modules["safetensors"] = st
        sys.modules["safetensors.torch"] = st_torch

    try:
        import transformers  # noqa: F401
    except ImportError:
        tr = types.ModuleType("transformers")
        tr.AutoModelForMaskedLM = object
        tr.AutoTokenizer = object
        tr.get_linear_schedule_with_warmup = lambda *a, **k: None
        sys.modules["transformers"] = tr


_stub_missing_heavy_deps()

import finetune_sequence_classifier as fsc  # noqa: E402


LABELS = ["informative", "situation-normal", "data-critical"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def write_jsonl(rows: list[dict]) -> Path:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for row in rows:
        f.write(json.dumps(row) + "\n")
    f.close()
    return Path(f.name)


# --------------------------------------------------------------------------
# load_jsonl: target parsing / validation
# --------------------------------------------------------------------------


class TestLoadJsonlTargets(unittest.TestCase):
    def test_label_only_row_gets_none_target(self):
        path = write_jsonl([{"text": "restart the pod", "label": "situation-normal"}])
        items = fsc.load_jsonl(path)
        self.assertEqual(len(items), 1)
        self.assertIsNone(items[0]["target"])
        self.assertEqual(items[0]["label"], "situation-normal")

    def test_valid_target_row_parses_and_normalizes_to_float(self):
        path = write_jsonl(
            [
                {
                    "text": "disk usage nominal",
                    "label": "informative",
                    "target": {"informative": 0.667, "situation-normal": 0.333, "data-critical": 0.0},
                }
            ]
        )
        items = fsc.load_jsonl(path)
        target = items[0]["target"]
        self.assertIsNotNone(target)
        self.assertEqual(set(target), {"informative", "situation-normal", "data-critical"})
        for v in target.values():
            self.assertIsInstance(v, float)
        self.assertAlmostEqual(sum(target.values()), 1.0, places=6)

    def test_target_sum_off_by_more_than_tolerance_raises(self):
        path = write_jsonl(
            [{"text": "x", "label": "informative", "target": {"informative": 0.5, "data-critical": 0.4}}]
        )
        with self.assertRaises(ValueError):
            fsc.load_jsonl(path)

    def test_target_sum_within_tolerance_is_accepted(self):
        # 1e-7 off, inside the 1e-6 tolerance band.
        path = write_jsonl(
            [
                {
                    "text": "x",
                    "label": "informative",
                    "target": {"informative": 0.5000001, "data-critical": 0.4999999},
                }
            ]
        )
        items = fsc.load_jsonl(path)  # must not raise
        self.assertAlmostEqual(sum(items[0]["target"].values()), 1.0, places=6)

    def test_target_not_a_dict_raises(self):
        path = write_jsonl([{"text": "x", "label": "informative", "target": [0.5, 0.5]}])
        with self.assertRaises(ValueError):
            fsc.load_jsonl(path)

    def test_target_empty_dict_raises(self):
        path = write_jsonl([{"text": "x", "label": "informative", "target": {}}])
        with self.assertRaises(ValueError):
            fsc.load_jsonl(path)

    def test_target_non_numeric_value_raises(self):
        path = write_jsonl(
            [{"text": "x", "label": "informative", "target": {"informative": "high", "data-critical": 0.0}}]
        )
        with self.assertRaises(ValueError):
            fsc.load_jsonl(path)

    def test_target_bool_value_raises(self):
        # bool is a numeric subtype in Python; must be explicitly rejected,
        # not silently accepted as 0/1.
        path = write_jsonl(
            [{"text": "x", "label": "informative", "target": {"informative": True, "data-critical": False}}]
        )
        with self.assertRaises(ValueError):
            fsc.load_jsonl(path)

    def test_row_without_label_still_rejected(self):
        path = write_jsonl([{"text": "x", "target": {"informative": 1.0}}])
        with self.assertRaises(ValueError):
            fsc.load_jsonl(path)


# --------------------------------------------------------------------------
# resolve_label_order
# --------------------------------------------------------------------------


class TestResolveLabelOrder(unittest.TestCase):
    def test_default_is_sorted_observed_labels_unchanged_behavior(self):
        items = [{"label": "zebra"}, {"label": "apple"}, {"label": "mango"}]
        self.assertEqual(fsc.resolve_label_order(items), ["apple", "mango", "zebra"])

    def test_explicit_order_is_honored_verbatim(self):
        items = [{"label": "data-critical"}, {"label": "informative"}]
        order = fsc.resolve_label_order(items, LABELS)
        self.assertEqual(order, LABELS)  # NOT sorted alphabetically

    def test_explicit_order_may_include_unobserved_label(self):
        # zero training rows for situation-normal is allowed: fixed-width
        # head survives an uneven split.
        items = [{"label": "informative"}, {"label": "data-critical"}]
        order = fsc.resolve_label_order(items, LABELS)
        self.assertEqual(order, LABELS)

    def test_explicit_order_missing_an_observed_label_raises(self):
        items = [{"label": "informative"}, {"label": "surprise-label"}]
        with self.assertRaises(ValueError):
            fsc.resolve_label_order(items, LABELS)

    def test_explicit_order_with_duplicates_raises(self):
        items = [{"label": "informative"}]
        with self.assertRaises(ValueError):
            fsc.resolve_label_order(items, ["informative", "informative", "data-critical"])


# --------------------------------------------------------------------------
# validate_target_keys (second-stage: needs the resolved label set)
# --------------------------------------------------------------------------


class TestValidateTargetKeys(unittest.TestCase):
    def test_keys_subset_of_label_set_passes(self):
        items = [
            {"label": "informative", "target": {"informative": 0.7, "situation-normal": 0.3}},
            {"label": "data-critical", "target": None},
        ]
        fsc.validate_target_keys(items, set(LABELS), "test-source")  # must not raise

    def test_unknown_key_raises_loudly(self):
        items = [{"label": "informative", "target": {"informative": 0.5, "not-a-real-class": 0.5}}]
        with self.assertRaises(ValueError):
            fsc.validate_target_keys(items, set(LABELS), "test-source")

    def test_rows_without_target_are_skipped(self):
        items = [{"label": "informative", "target": None}]
        fsc.validate_target_keys(items, {"informative"}, "test-source")  # must not raise


# --------------------------------------------------------------------------
# target_to_vector
# --------------------------------------------------------------------------


class TestTargetToVector(unittest.TestCase):
    def test_none_target_is_one_hot_at_label_id(self):
        vec = fsc.target_to_vector(2, None, LABEL2ID)
        self.assertTrue(torch.equal(vec, torch.tensor([0.0, 0.0, 1.0])))

    def test_explicit_target_maps_into_label2id_slot_order(self):
        target = {"data-critical": 0.1, "informative": 0.6, "situation-normal": 0.3}
        vec = fsc.target_to_vector(0, target, LABEL2ID)
        # LABEL2ID order is informative=0, situation-normal=1, data-critical=2
        self.assertTrue(torch.allclose(vec, torch.tensor([0.6, 0.3, 0.1])))

    def test_vector_length_matches_label2id_even_with_partial_target(self):
        # a target need not name every class explicitly if it legitimately
        # assigns zero mass elsewhere — but every named key must resolve.
        vec = fsc.target_to_vector(0, {"informative": 1.0}, LABEL2ID)
        self.assertEqual(vec.shape, (3,))
        self.assertTrue(torch.allclose(vec, torch.tensor([1.0, 0.0, 0.0])))


# --------------------------------------------------------------------------
# soft_cross_entropy: the core numeric-equivalence proof
# --------------------------------------------------------------------------


class TestSoftCrossEntropy(unittest.TestCase):
    def test_one_hot_targets_equal_hard_cross_entropy(self):
        torch.manual_seed(0)
        logits = torch.randn(8, 3)
        hard_labels = torch.randint(0, 3, (8,))
        one_hot = F.one_hot(hard_labels, num_classes=3).float()

        soft_loss = fsc.soft_cross_entropy(logits, one_hot)
        hard_loss = F.cross_entropy(logits, hard_labels)

        self.assertTrue(
            torch.allclose(soft_loss, hard_loss, atol=1e-6),
            f"soft={soft_loss.item()} hard={hard_loss.item()}",
        )

    def test_one_hot_equivalence_holds_across_many_random_batches(self):
        torch.manual_seed(1)
        for _ in range(20):
            batch = torch.randint(1, 16, (1,)).item()
            num_classes = torch.randint(2, 6, (1,)).item()
            logits = torch.randn(batch, num_classes) * 5.0  # exercise a wider logit range
            hard_labels = torch.randint(0, num_classes, (batch,))
            one_hot = F.one_hot(hard_labels, num_classes=num_classes).float()

            soft_loss = fsc.soft_cross_entropy(logits, one_hot)
            hard_loss = F.cross_entropy(logits, hard_labels)
            self.assertTrue(torch.allclose(soft_loss, hard_loss, atol=1e-5))

    def test_genuine_soft_target_matches_hand_computation(self):
        # 2-class, single row: logits [0, 0] -> softmax [0.5, 0.5].
        # target [0.667, 0.333] -> loss = -(0.667*ln(0.5) + 0.333*ln(0.5))
        #                                = -ln(0.5) = ln(2)
        logits = torch.tensor([[0.0, 0.0]])
        targets = torch.tensor([[0.667, 0.333]])
        loss = fsc.soft_cross_entropy(logits, targets)
        self.assertAlmostEqual(loss.item(), torch.log(torch.tensor(2.0)).item(), places=5)

    def test_soft_target_loss_between_the_two_hard_losses_it_blends(self):
        # A target that's a mixture of two one-hots should produce a loss
        # that's the same convex combination of the two hard CE losses
        # (cross-entropy is linear in the target distribution).
        torch.manual_seed(2)
        logits = torch.randn(1, 4)
        mix = torch.tensor([0.667, 0.333, 0.0, 0.0])
        blended = fsc.soft_cross_entropy(logits, mix.unsqueeze(0))

        loss_a = F.cross_entropy(logits, torch.tensor([0]))
        loss_b = F.cross_entropy(logits, torch.tensor([1]))
        expected = 0.667 * loss_a + 0.333 * loss_b

        self.assertTrue(torch.allclose(blended, expected, atol=1e-5))


# --------------------------------------------------------------------------
# ClassificationDataset + collate: end-to-end with a fake tokenizer
# --------------------------------------------------------------------------


class FakeTokenizer:
    """Minimal stand-in for a HF tokenizer's __call__: fixed-width dummy
    ids/mask so collate can be exercised with zero network/model access.
    """

    def __call__(self, texts, padding=True, truncation=True, max_length=8, return_tensors="pt"):
        n = len(texts)
        return {
            "input_ids": torch.ones(n, 4, dtype=torch.long),
            "attention_mask": torch.ones(n, 4, dtype=torch.long),
        }


class TestDatasetAndCollate(unittest.TestCase):
    def setUp(self):
        self.items = [
            {"text": "a", "label": "informative", "target": None},
            {
                "text": "b",
                "label": "situation-normal",
                "target": {"informative": 0.2, "situation-normal": 0.7, "data-critical": 0.1},
            },
            {"text": "c", "label": "data-critical", "target": None},
        ]
        self.ds = fsc.ClassificationDataset(self.items, LABEL2ID)
        self.collate = fsc.make_collate(FakeTokenizer(), max_len=8)

    def test_dataset_getitem_shapes(self):
        text, label_id, target_vec = self.ds[0]
        self.assertEqual(text, "a")
        self.assertEqual(label_id, 0)
        self.assertEqual(target_vec.shape, (3,))
        self.assertTrue(torch.equal(target_vec, torch.tensor([1.0, 0.0, 0.0])))

    def test_collate_stacks_labels_and_targets(self):
        batch = [self.ds[i] for i in range(len(self.ds))]
        enc, labels, targets = self.collate(batch)
        self.assertEqual(labels.tolist(), [0, 1, 2])
        self.assertEqual(targets.shape, (3, 3))
        self.assertTrue(torch.equal(targets[0], torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(targets[1], torch.tensor([0.2, 0.7, 0.1])))
        self.assertTrue(torch.equal(targets[2], torch.tensor([0.0, 0.0, 1.0])))
        # every row's target distribution still sums to 1 after collate
        self.assertTrue(torch.allclose(targets.sum(dim=-1), torch.ones(3)))

    def test_collate_batch_of_all_one_hot_rows_matches_hard_labels_via_argmax(self):
        # sanity: for label-only rows, argmax(target) == label id, which is
        # what today's eval (argmax vs majority label) already assumes.
        batch = [self.ds[0], self.ds[2]]
        _enc, labels, targets = self.collate(batch)
        self.assertTrue(torch.equal(targets.argmax(dim=-1), labels))


if __name__ == "__main__":
    unittest.main()
