"""tests/unit/test_attacks.py — FGSM, PGD, PatchAttack tests."""
import sys, os, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from adversarial_robustness.models.dummy_model import DummyClassifier
from adversarial_robustness.attacks.fgsm import FGSM
from adversarial_robustness.attacks.pgd import PGD
from adversarial_robustness.attacks.patch_attack import PatchAttack
from adversarial_robustness.attacks.base_attack import AttackResult

RNG   = np.random.default_rng(42)
MODEL = DummyClassifier(num_classes=5, input_shape=(3, 8, 8), seed=42)
X     = RNG.random((8, 3, 8, 8)).astype(np.float32)
Y     = RNG.integers(0, 5, 8).astype(np.int64)
EPS   = 0.05

def _contract(tc, attack, x, y):
    result = attack.generate(x, y)
    tc.assertIsInstance(result, AttackResult)
    tc.assertEqual(result.adversarial_examples.shape, x.shape)
    tc.assertEqual(result.adversarial_examples.dtype, np.float32)
    tc.assertGreaterEqual(float(result.adversarial_examples.min()), -1e-6)
    tc.assertLessEqual(float(result.adversarial_examples.max()), 1.0+1e-6)
    tc.assertGreaterEqual(result.attack_success_rate, 0.0)
    tc.assertLessEqual(result.attack_success_rate, 1.0)
    tc.assertGreaterEqual(result.mean_l2_norm, 0.0)
    tc.assertGreaterEqual(result.mean_linf_norm, 0.0)
    np.testing.assert_allclose(result.perturbations, result.adversarial_examples - x, atol=1e-5)

class TestFGSMContract(unittest.TestCase):
    def test_contract(self): _contract(self, FGSM(MODEL, EPS), X, Y)
    def test_linf_bound(self):
        r = FGSM(MODEL, EPS).generate(X, Y)
        self.assertLessEqual(float(np.abs(r.perturbations).max()), EPS+1e-5)
    def test_zero_epsilon_no_change(self):
        r = FGSM(MODEL, 0.0).generate(X, Y)
        np.testing.assert_allclose(r.adversarial_examples, X, atol=1e-6)
    def test_larger_eps_larger_pert(self):
        r_small = FGSM(MODEL, 0.01).generate(X, Y)
        r_large = FGSM(MODEL, 0.10).generate(X, Y)
        self.assertGreaterEqual(r_large.mean_linf_norm, r_small.mean_linf_norm)
    def test_negative_eps_raises(self):
        with self.assertRaises(ValueError): FGSM(MODEL, -0.1)
    def test_wrong_dims_raises(self):
        attack = FGSM(MODEL, EPS)
        with self.assertRaises(ValueError):
            attack.generate(np.zeros((8,8,8)), Y)
    def test_metadata_loss(self):
        r = FGSM(MODEL, EPS).generate(X, Y)
        self.assertIn("loss", r.metadata)
        self.assertGreater(r.metadata["loss"], 0)

class TestPGDContract(unittest.TestCase):
    def test_contract(self): _contract(self, PGD(MODEL, EPS, num_steps=5), X, Y)
    def test_linf_bound(self):
        r = PGD(MODEL, EPS, num_steps=10).generate(X, Y)
        per_sample = np.abs(r.perturbations).reshape(len(X),-1).max(axis=1)
        self.assertTrue(np.all(per_sample <= EPS+1e-5))
    def test_random_start_differs(self):
        r1 = PGD(MODEL, EPS, num_steps=5, random_start=False).generate(X, Y)
        r2 = PGD(MODEL, EPS, num_steps=5, random_start=True).generate(X, Y)
        self.assertFalse(np.allclose(r1.adversarial_examples, r2.adversarial_examples))
    def test_loss_history_length(self):
        r = PGD(MODEL, EPS, num_steps=7).generate(X, Y)
        self.assertEqual(len(r.metadata["loss_history"]), 7)
    def test_invalid_alpha_raises(self):
        with self.assertRaises(ValueError): PGD(MODEL, EPS, alpha=0.0)
    def test_invalid_steps_raises(self):
        with self.assertRaises(ValueError): PGD(MODEL, EPS, num_steps=0)

class TestPatchAttackContract(unittest.TestCase):
    def test_contract(self): _contract(self, PatchAttack(MODEL, patch_size=4, num_steps=10), X, Y)
    def test_patch_changes_pixels(self):
        r = PatchAttack(MODEL, patch_size=4, num_steps=20).generate(X, Y)
        self.assertGreater(float(np.abs(r.adversarial_examples - X).max()), 1e-4)
    def test_patch_accessible(self):
        attack = PatchAttack(MODEL, patch_size=4, num_steps=10)
        attack.generate(X, Y)
        self.assertIsNotNone(attack.optimised_patch)
        self.assertEqual(attack.optimised_patch.shape, (3, 4, 4))
    def test_patch_too_large_raises(self):
        attack = PatchAttack(MODEL, patch_size=100, num_steps=5)
        with self.assertRaises(ValueError): attack.generate(X, Y)
    def test_single_sample(self):
        r = PatchAttack(MODEL, patch_size=4, num_steps=5).generate(X[:1], Y[:1])
        self.assertEqual(r.adversarial_examples.shape, (1,3,8,8))

if __name__ == "__main__":
    unittest.main(verbosity=2)
