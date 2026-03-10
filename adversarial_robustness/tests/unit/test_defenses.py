"""tests/unit/test_defenses.py — Defense module tests."""
import sys, os, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from adversarial_robustness.defenses.denoiser import GaussianDenoiser, MedianDenoiser, FeatureSqueezing
from adversarial_robustness.defenses.base_defense import DefensePipeline
from adversarial_robustness.defenses.adversarial_training import AdversarialTrainer
from adversarial_robustness.models.dummy_model import DummyClassifier

RNG   = np.random.default_rng(42)
MODEL = DummyClassifier(num_classes=5, input_shape=(3, 8, 8), seed=42)
X     = RNG.random((8, 3, 8, 8)).astype(np.float32)
Y     = RNG.integers(0, 5, 8).astype(np.int64)

def _contract(tc, defense, x):
    out = defense.preprocess(x)
    tc.assertEqual(out.shape, x.shape)
    tc.assertEqual(out.dtype, np.float32)
    tc.assertGreaterEqual(float(out.min()), -1e-6)
    tc.assertLessEqual(float(out.max()), 1.0+1e-6)
    # No in-place modification
    x_copy = x.copy()
    defense.preprocess(x)
    np.testing.assert_array_equal(x, x_copy)

class TestGaussianDenoiser(unittest.TestCase):
    def test_contract(self): _contract(self, GaussianDenoiser(0.05), X)
    def test_zero_sigma_identity(self):
        out = GaussianDenoiser(0.0).preprocess(X)
        np.testing.assert_allclose(out, X, atol=1e-6)
    def test_high_sigma_reduces_variance(self):
        noisy = np.clip(X + RNG.standard_normal(X.shape).astype(np.float32)*0.3, 0, 1)
        smoothed = GaussianDenoiser(2.0).preprocess(noisy)
        self.assertLess(smoothed.std(), noisy.std())
    def test_negative_sigma_raises(self):
        with self.assertRaises(ValueError): GaussianDenoiser(-0.1)
    def test_callable(self):
        out = GaussianDenoiser(0.05)(X)
        self.assertEqual(out.shape, X.shape)

class TestMedianDenoiser(unittest.TestCase):
    def test_contract(self): _contract(self, MedianDenoiser(3), X)
    def test_even_kernel_raises(self):
        with self.assertRaises(ValueError): MedianDenoiser(2)
    def test_kernel1_near_identity(self):
        out = MedianDenoiser(1).preprocess(X)
        np.testing.assert_allclose(out, X, atol=1e-5)

class TestFeatureSqueezing(unittest.TestCase):
    def test_contract(self): _contract(self, FeatureSqueezing(4), X)
    def test_2bit_few_unique(self):
        out = FeatureSqueezing(2).preprocess(X)
        self.assertLessEqual(len(np.unique(out.ravel())), 4)
    def test_8bit_near_identity(self):
        out = FeatureSqueezing(8).preprocess(X)
        np.testing.assert_allclose(out, X, atol=1/255+1e-6)
    def test_invalid_bit_depth_raises(self):
        with self.assertRaises(ValueError): FeatureSqueezing(0)
        with self.assertRaises(ValueError): FeatureSqueezing(9)

class TestDefensePipeline(unittest.TestCase):
    def test_applies_in_order(self):
        d1 = GaussianDenoiser(0.05); d2 = FeatureSqueezing(6)
        expected = d2.preprocess(d1.preprocess(X))
        actual   = DefensePipeline([d1, d2]).preprocess(X)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
    def test_empty_raises(self):
        with self.assertRaises(ValueError): DefensePipeline([])
    def test_repr(self):
        r = repr(DefensePipeline([GaussianDenoiser(), FeatureSqueezing()]))
        self.assertIn("DefensePipeline", r)

class TestAdversarialTrainer(unittest.TestCase):
    def _data(self, n=32):
        x = RNG.random((n, 3, 8, 8)).astype(np.float32)
        y = RNG.integers(0, 5, n).astype(np.int64)
        return x, y
    def test_training_runs(self):
        x, y = self._data()
        h = AdversarialTrainer(MODEL, attack="fgsm", epsilon=0.05,
                               epochs=2, batch_size=16).fit(x, y)
        self.assertEqual(len(h.epochs), 2)
    def test_metrics_in_range(self):
        x, y = self._data()
        h = AdversarialTrainer(MODEL, attack="pgd", epsilon=0.03,
                               epochs=2, batch_size=16, pgd_steps=3).fit(x, y)
        for e in h.epochs:
            self.assertGreaterEqual(e.train_clean_acc, 0.0)
            self.assertLessEqual(e.train_clean_acc, 1.0)
    def test_invalid_ratio_raises(self):
        with self.assertRaises(ValueError): AdversarialTrainer(MODEL, ratio=1.5)
    def test_invalid_attack_raises(self):
        with self.assertRaises(ValueError): AdversarialTrainer(MODEL, attack="cw")
    def test_summary_keys(self):
        x, y = self._data()
        h = AdversarialTrainer(MODEL, epochs=1, batch_size=16).fit(x, y)
        s = h.summary()
        self.assertIn("total_epochs", s)
        self.assertIn("best_epoch", s)

if __name__ == "__main__":
    unittest.main(verbosity=2)
