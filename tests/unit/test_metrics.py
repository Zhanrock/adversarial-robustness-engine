"""tests/unit/test_metrics.py — Evaluation metric function tests."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from adversarial_robustness.evaluation.metrics import (  # noqa: E402
    attack_success_rate,
    clean_accuracy,
    compute_all_metrics,
    mean_perturbation_norm,
    robustness_gap,
)
from adversarial_robustness.models.dummy_model import DummyClassifier  # noqa: E402


class TestCleanAccuracy(unittest.TestCase):
    def test_all_correct(self):
        self.assertEqual(clean_accuracy(np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3])), 1.0)

    def test_none_correct(self):
        self.assertEqual(clean_accuracy(np.array([1, 2, 3, 0]), np.array([0, 1, 2, 3])), 0.0)

    def test_half_correct(self):
        self.assertEqual(clean_accuracy(np.array([0, 9, 2, 9]), np.array([0, 1, 2, 3])), 0.5)

    def test_range(self):
        rng = np.random.default_rng(0)
        a = clean_accuracy(rng.integers(0, 5, 100), rng.integers(0, 5, 100))
        self.assertGreaterEqual(a, 0.0)
        self.assertLessEqual(a, 1.0)


class TestASR(unittest.TestCase):
    def test_perfect_attack(self):
        yc = np.array([0, 1, 2, 3])
        ya = np.array([1, 2, 3, 0])
        yt = np.array([0, 1, 2, 3])
        self.assertEqual(attack_success_rate(yc, ya, yt), 1.0)

    def test_no_effect(self):
        y = np.array([0, 1, 2, 3])
        self.assertEqual(attack_success_rate(y, y, y), 0.0)

    def test_already_wrong_not_counted(self):
        yc = np.array([0, 9, 2, 9])
        ya = np.array([1, 9, 3, 9])
        yt = np.array([0, 1, 2, 3])
        self.assertEqual(attack_success_rate(yc, ya, yt), 1.0)


class TestRobustnessGap(unittest.TestCase):
    def test_positive(self):
        self.assertAlmostEqual(robustness_gap(0.9, 0.6), 0.3, places=6)

    def test_zero(self):
        self.assertAlmostEqual(robustness_gap(0.8, 0.8), 0.0, places=6)


class TestPerturbationNorm(unittest.TestCase):
    def test_zero_pert(self):
        x = np.random.rand(4, 3, 8, 8).astype(np.float32)
        self.assertAlmostEqual(mean_perturbation_norm(x, x, p=2), 0.0, places=5)

    def test_positive(self):
        x = np.zeros((4, 3, 8, 8), dtype=np.float32)
        self.assertGreater(mean_perturbation_norm(x, x + 0.1, p=2), 0)

    def test_linf(self):
        x = np.zeros((2, 1, 4, 4), dtype=np.float32)
        xadv = x.copy()
        xadv[:, 0, 0, 0] = 0.5
        self.assertAlmostEqual(mean_perturbation_norm(x, xadv, p=np.inf), 0.5, places=5)


class TestComputeAllMetrics(unittest.TestCase):
    def _data(self):
        rng = np.random.default_rng(7)
        x = rng.random((20, 3, 8, 8)).astype(np.float32)
        xadv = np.clip(x + rng.uniform(-0.05, 0.05, x.shape), 0, 1).astype(np.float32)
        y = rng.integers(0, 5, 20).astype(np.int64)
        m = DummyClassifier(5, (3, 8, 8))
        return x, xadv, y, m.predict(x), m.predict(xadv)

    def test_all_keys_present(self):
        x, xadv, y, yc, ya = self._data()
        keys = compute_all_metrics(x, xadv, y, yc, ya, 0.05).keys()
        for k in [
            "clean_accuracy",
            "adversarial_accuracy",
            "attack_success_rate",
            "robustness_gap",
            "mean_l2_perturbation",
            "certified_robustness",
        ]:
            self.assertIn(k, keys)

    def test_all_finite(self):
        x, xadv, y, yc, ya = self._data()
        for k, v in compute_all_metrics(x, xadv, y, yc, ya, 0.05).items():
            self.assertTrue(np.isfinite(v), f"{k} not finite")

    def test_gap_consistent(self):
        x, xadv, y, yc, ya = self._data()
        m = compute_all_metrics(x, xadv, y, yc, ya, 0.05)
        self.assertAlmostEqual(
            m["robustness_gap"], m["clean_accuracy"] - m["adversarial_accuracy"], places=5
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
