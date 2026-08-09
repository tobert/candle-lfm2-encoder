#!/usr/bin/env python3
"""Unit tests for training/calib_metrics.py.

Synthetic distributions only — no model, no torch, no dataset files.
Run with: python3 -m unittest discover training/tests
(or: python3 -m unittest training.tests.test_calib_metrics)
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import calib_metrics as cm  # noqa: E402


class TestIsUnanimous(unittest.TestCase):
    def test_one_hot_is_unanimous(self):
        self.assertTrue(cm.is_unanimous({"informative": 1.0, "situation-normal": 0.0}))

    def test_near_one_hot_above_threshold(self):
        self.assertTrue(cm.is_unanimous({"informative": 0.9995, "situation-normal": 0.0005}))

    def test_split_is_not_unanimous(self):
        self.assertFalse(cm.is_unanimous({"informative": 0.6, "situation-normal": 0.4}))

    def test_exactly_at_default_threshold(self):
        self.assertTrue(cm.is_unanimous({"a": 0.999, "b": 0.001}))

    def test_just_under_threshold_is_split(self):
        self.assertFalse(cm.is_unanimous({"a": 0.998, "b": 0.002}))

    def test_custom_threshold(self):
        self.assertTrue(cm.is_unanimous({"a": 0.8, "b": 0.2}, threshold=0.8))
        self.assertFalse(cm.is_unanimous({"a": 0.79, "b": 0.21}, threshold=0.8))

    def test_empty_target_raises(self):
        with self.assertRaises(ValueError):
            cm.is_unanimous({})


class TestArgmaxLabel(unittest.TestCase):
    def test_simple_argmax(self):
        self.assertEqual(
            cm.argmax_label({"informative": 0.1, "situation-normal": 0.7, "data-critical": 0.2}),
            "situation-normal",
        )

    def test_tie_breaks_alphabetically(self):
        self.assertEqual(cm.argmax_label({"zeta": 0.5, "alpha": 0.5}), "alpha")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            cm.argmax_label({})


class TestSeparation(unittest.TestCase):
    def test_hand_computed_separation(self):
        # unanimous mean = (0.9 + 0.8) / 2 = 0.85
        # split mean = (0.6 + 0.5 + 0.4) / 3 = 0.5
        # separation = 0.85 - 0.5 = 0.35
        result = cm.separation([0.9, 0.8], [0.6, 0.5, 0.4])
        self.assertAlmostEqual(result, 0.35, places=9)

    def test_negative_separation_when_model_more_confident_on_splits(self):
        result = cm.separation([0.5], [0.9])
        self.assertAlmostEqual(result, -0.4, places=9)

    def test_zero_separation(self):
        result = cm.separation([0.7, 0.7], [0.7])
        self.assertAlmostEqual(result, 0.0, places=9)

    def test_empty_unanimous_raises(self):
        with self.assertRaises(ValueError):
            cm.separation([], [0.5])

    def test_empty_split_raises(self):
        with self.assertRaises(ValueError):
            cm.separation([0.5], [])


class TestOverconfidentSplitCount(unittest.TestCase):
    def test_counts_above_default_threshold(self):
        self.assertEqual(cm.overconfident_split_count([0.9, 0.5, 0.88, 0.879999]), 2)

    def test_none_overconfident(self):
        self.assertEqual(cm.overconfident_split_count([0.1, 0.2, 0.3]), 0)

    def test_empty_list_is_zero(self):
        self.assertEqual(cm.overconfident_split_count([]), 0)

    def test_custom_threshold(self):
        self.assertEqual(cm.overconfident_split_count([0.6, 0.7], threshold=0.65), 1)


class TestExpectedCalibrationError(unittest.TestCase):
    def test_perfect_calibration_is_zero(self):
        # Perfectly-calibrated construction: bin mean confidence == bin accuracy exactly.
        confidences = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        correct = [True] * 9 + [False]  # bin acc = 0.9, mean conf = 0.9 -> |diff| = 0
        ece = cm.expected_calibration_error(confidences, correct)
        self.assertAlmostEqual(ece, 0.0, places=9)

    def test_perfect_calibration_multi_bin(self):
        # Bin A: conf 0.2 x5, 1/5 correct (acc 0.2) -> matches
        # Bin B: conf 0.8 x5, 4/5 correct (acc 0.8) -> matches
        confidences = [0.2] * 5 + [0.8] * 5
        correct = [True, False, False, False, False] + [True, True, True, True, False]
        ece = cm.expected_calibration_error(confidences, correct)
        self.assertAlmostEqual(ece, 0.0, places=9)

    def test_fully_miscalibrated(self):
        # All predictions at confidence 1.0 but all wrong -> ECE == 1.0
        confidences = [1.0, 1.0, 1.0]
        correct = [False, False, False]
        ece = cm.expected_calibration_error(confidences, correct)
        self.assertAlmostEqual(ece, 1.0, places=9)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            cm.expected_calibration_error([0.5, 0.6], [True])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            cm.expected_calibration_error([], [])

    def test_out_of_range_confidence_raises(self):
        with self.assertRaises(ValueError):
            cm.expected_calibration_error([1.5], [True])
        with self.assertRaises(ValueError):
            cm.expected_calibration_error([-0.1], [True])

    def test_confidence_of_exactly_one_lands_in_last_bin(self):
        # Regression guard: idx = int(1.0 * 10) = 10 must clamp to bin 9, not raise/IndexError.
        ece = cm.expected_calibration_error([1.0], [True])
        self.assertAlmostEqual(ece, 0.0, places=9)


class TestCrossEntropy(unittest.TestCase):
    def test_one_hot_target_equals_neg_log_p(self):
        pred = {"informative": 0.7, "situation-normal": 0.2, "data-critical": 0.1}
        target = {"informative": 1.0, "situation-normal": 0.0, "data-critical": 0.0}
        result = cm.cross_entropy(pred, target)
        self.assertAlmostEqual(result, -math.log(0.7), places=9)

    def test_split_target_weighted_sum(self):
        pred = {"a": 0.5, "b": 0.5}
        target = {"a": 0.6, "b": 0.4}
        expected = -(0.6 * math.log(0.5) + 0.4 * math.log(0.5))
        result = cm.cross_entropy(pred, target)
        self.assertAlmostEqual(result, expected, places=9)

    def test_target_class_missing_from_pred_uses_epsilon(self):
        pred = {"a": 1.0}  # no "b" key at all -> treated as prob 0, clamped
        target = {"a": 0.0, "b": 1.0}
        result = cm.cross_entropy(pred, target)
        self.assertTrue(math.isfinite(result))
        self.assertGreater(result, 20.0)  # -log(1e-12) ~= 27.6

    def test_empty_target_raises(self):
        with self.assertRaises(ValueError):
            cm.cross_entropy({"a": 1.0}, {})


class TestMeanCrossEntropy(unittest.TestCase):
    def test_mean_over_rows(self):
        pairs = [
            ({"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 0.0}),  # CE = -log(1.0) = 0
            ({"a": 0.5, "b": 0.5}, {"a": 1.0, "b": 0.0}),  # CE = -log(0.5)
        ]
        expected = (0.0 + (-math.log(0.5))) / 2
        result = cm.mean_cross_entropy(pairs)
        self.assertAlmostEqual(result, expected, places=9)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            cm.mean_cross_entropy([])


if __name__ == "__main__":
    unittest.main()
