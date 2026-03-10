"""tests/integration/test_full_pipeline.py — Full pipeline integration tests."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from adversarial_robustness.attacks.fgsm import FGSM
from adversarial_robustness.attacks.patch_attack import PatchAttack
from adversarial_robustness.attacks.pgd import PGD
from adversarial_robustness.defenses.adversarial_training import AdversarialTrainer
from adversarial_robustness.defenses.base_defense import DefensePipeline
from adversarial_robustness.defenses.denoiser import FeatureSqueezing, GaussianDenoiser
from adversarial_robustness.evaluation.benchmarker import RobustnessBenchmarker
from adversarial_robustness.models.dummy_model import DummyClassifier

RNG = np.random.default_rng(0)
X = RNG.random((64, 3, 16, 16)).astype(np.float32)
Y = RNG.integers(0, 10, 64).astype(np.int64)
MODEL = DummyClassifier(num_classes=10, input_shape=(3, 16, 16), seed=0)


class TestFGSMWithDefense(unittest.TestCase):
    def test_defense_does_not_hurt(self):
        r = FGSM(MODEL, 0.05).generate(X, Y)
        adv_acc = MODEL.accuracy(r.adversarial_examples, Y)
        def_acc = MODEL.accuracy(GaussianDenoiser(0.07).preprocess(r.adversarial_examples), Y)
        self.assertGreaterEqual(def_acc, adv_acc - 0.05)


class TestBenchmarkSerialization(unittest.TestCase):
    def test_json_written(self):
        with tempfile.TemporaryDirectory() as d:
            b = RobustnessBenchmarker(
                MODEL, defenses=[GaussianDenoiser(0.05)], output_dir=d, model_name="test"
            )
            b.run(X, Y, [PGD(MODEL, 0.05, num_steps=5)], save_report=True)
            files = os.listdir(d)
            self.assertTrue(any(f.endswith(".json") for f in files))

    def test_json_structure(self):
        with tempfile.TemporaryDirectory() as d:
            b = RobustnessBenchmarker(MODEL, output_dir=d)
            b.run(X, Y, [FGSM(MODEL, 0.03)], save_report=True)
            jfile = next(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))
            data = json.load(open(jfile))
            self.assertIn("clean_accuracy", data)
            self.assertIn("results", data)


class TestAdversarialTrainingCycle(unittest.TestCase):
    def test_cycle(self):
        xt, yt = X[:48], Y[:48]
        xv, yv = X[48:], Y[48:]
        h = AdversarialTrainer(
            MODEL, attack="fgsm", epsilon=0.03, ratio=0.5, epochs=2, batch_size=16
        ).fit(xt, yt, xv, yv)
        self.assertEqual(len(h.epochs), 2)
        s = h.summary()
        self.assertEqual(s["total_epochs"], 2)
        self.assertGreaterEqual(s["final_val_adv_acc"], 0.0)


class TestMultiAttackBenchmark(unittest.TestCase):
    def test_all_attacks_present(self):
        attacks = [
            FGSM(MODEL, 0.05),
            PGD(MODEL, 0.05, num_steps=5),
            PatchAttack(MODEL, patch_size=6, num_steps=20),
        ]
        pipeline = DefensePipeline([GaussianDenoiser(0.05), FeatureSqueezing(6)])
        with tempfile.TemporaryDirectory() as d:
            r = RobustnessBenchmarker(MODEL, [pipeline], d).run(X, Y, attacks, save_report=False)
        self.assertEqual(len(r.results), 3)
        names = [x.attack_name for x in r.results]
        self.assertIn("FGSM", names)
        self.assertIn("PGD", names)
        self.assertIn("PatchAttack", names)

    def test_worst_case_le_clean(self):
        with tempfile.TemporaryDirectory() as d:
            r = RobustnessBenchmarker(MODEL, output_dir=d).run(
                X, Y, [FGSM(MODEL, 0.01), FGSM(MODEL, 0.1)], save_report=False
            )
        self.assertLessEqual(r.worst_case_robustness(), r.clean_accuracy)

    def test_report_gap_consistent(self):
        with tempfile.TemporaryDirectory() as d:
            r = RobustnessBenchmarker(MODEL, output_dir=d).run(
                X, Y, [FGSM(MODEL, 0.03)], save_report=False
            )
        res = r.results[0]
        self.assertAlmostEqual(
            res.robustness_gap, res.clean_accuracy - res.adversarial_accuracy, places=5
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
