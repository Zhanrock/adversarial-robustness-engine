"""tests/unit/test_models.py — DummyClassifier & BaseModel tests."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from adversarial_robustness.models.base_model import BaseModel
from adversarial_robustness.models.dummy_model import DummyClassifier

RNG = np.random.default_rng(42)
MODEL = DummyClassifier(num_classes=5, input_shape=(3, 8, 8), seed=42)
X = RNG.random((8, 3, 8, 8)).astype(np.float32)
Y = RNG.integers(0, 5, 8).astype(np.int64)


class TestDummyClassifier(unittest.TestCase):
    def test_is_base_model(self):
        self.assertIsInstance(MODEL, BaseModel)

    def test_forward_shape(self):
        logits = MODEL.forward(X)
        self.assertEqual(logits.shape, (8, 5))

    def test_forward_dtype(self):
        self.assertEqual(MODEL.forward(X).dtype, np.float32)

    def test_predict_shape(self):
        self.assertEqual(MODEL.predict(X).shape, (8,))

    def test_predict_in_range(self):
        preds = MODEL.predict(X)
        self.assertGreaterEqual(int(preds.min()), 0)
        self.assertLess(int(preds.max()), 5)

    def test_gradients_shape(self):
        grad, _ = MODEL.get_gradients(X, Y)
        self.assertEqual(grad.shape, X.shape)

    def test_gradients_finite(self):
        grad, _ = MODEL.get_gradients(X, Y)
        self.assertTrue(np.all(np.isfinite(grad)))

    def test_loss_positive(self):
        _, loss = MODEL.get_gradients(X, Y)
        self.assertGreater(loss, 0)

    def test_accuracy_range(self):
        acc = MODEL.accuracy(X, Y)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_single_sample(self):
        x1 = X[:1]
        y1 = Y[:1]
        self.assertEqual(MODEL.forward(x1).shape, (1, 5))
        self.assertEqual(MODEL.predict(x1).shape, (1,))

    def test_train_eval(self):
        MODEL.train()
        self.assertTrue(MODEL._is_training)
        MODEL.eval()
        self.assertFalse(MODEL._is_training)

    def test_callable(self):
        self.assertEqual(MODEL(X).shape[1], 5)

    def test_repr(self):
        r = repr(MODEL)
        self.assertIn("DummyClassifier", r)

    def test_different_seeds_differ(self):
        m1 = DummyClassifier(5, (3, 8, 8), seed=1)
        m2 = DummyClassifier(5, (3, 8, 8), seed=2)
        self.assertFalse(np.allclose(m1._W, m2._W))

    def test_same_seed_equal(self):
        m1 = DummyClassifier(5, (3, 8, 8), seed=99)
        m2 = DummyClassifier(5, (3, 8, 8), seed=99)
        self.assertTrue(np.allclose(m1._W, m2._W))


if __name__ == "__main__":
    unittest.main(verbosity=2)
