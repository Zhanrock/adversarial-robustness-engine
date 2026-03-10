#!/usr/bin/env python3
"""
scripts/run_benchmark.py
-------------------------
Standalone script that runs a full adversarial robustness benchmark
against a synthetic dataset.  Replace ``generate_synthetic_data()``
with your real data loader for production use.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --epsilon 0.05 --samples 200
    python scripts/run_benchmark.py --config configs/experiment_fgsm.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from adversarial_robustness.attacks.fgsm import FGSM
from adversarial_robustness.attacks.pgd import PGD
from adversarial_robustness.attacks.patch_attack import PatchAttack
from adversarial_robustness.defenses.denoiser import GaussianDenoiser, FeatureSqueezing
from adversarial_robustness.defenses.base_defense import DefensePipeline
from adversarial_robustness.evaluation.benchmarker import RobustnessBenchmarker
from adversarial_robustness.models.dummy_model import DummyClassifier
from adversarial_robustness.utils.logger import configure_logging, get_logger


def generate_synthetic_data(n: int, input_shape: tuple, num_classes: int, seed: int = 42):
    """
    Generate synthetic test data for demonstration.

    In a real deployment, replace this with your actual dataset loader
    (e.g., torchvision.datasets.CIFAR10, custom VideoDataset, etc.)
    """
    rng = np.random.default_rng(seed)
    x = rng.random((n, *input_shape)).astype(np.float32)
    y = rng.integers(0, num_classes, n).astype(np.int64)
    return x, y


def main():
    parser = argparse.ArgumentParser(description="Adversarial Robustness Benchmark")
    parser.add_argument("--epsilon",  type=float, default=0.03)
    parser.add_argument("--samples",  type=int,   default=100)
    parser.add_argument("--classes",  type=int,   default=10)
    parser.add_argument("--pgd-steps",type=int,   default=20)
    parser.add_argument("--output",   default="reports/")
    parser.add_argument("--no-defense", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    configure_logging("DEBUG" if args.verbose else "INFO")
    logger = get_logger(__name__)

    input_shape = (3, 32, 32)

    logger.info("Generating %d synthetic test samples...", args.samples)
    x_test, y_test = generate_synthetic_data(
        args.samples, input_shape, args.classes
    )

    model = DummyClassifier(
        num_classes=args.classes, input_shape=input_shape, seed=42
    )

    # Attacks
    attacks = [
        FGSM(model, epsilon=args.epsilon),
        PGD(model, epsilon=args.epsilon, alpha=args.epsilon / 5,
            num_steps=args.pgd_steps),
        PatchAttack(model, patch_size=8, num_steps=100),
    ]

    # Defense pipeline (disable with --no-defense)
    defenses = (
        []
        if args.no_defense
        else [
            DefensePipeline([
                GaussianDenoiser(sigma=0.05),
                FeatureSqueezing(bit_depth=6),
            ])
        ]
    )

    # Run benchmark
    benchmarker = RobustnessBenchmarker(
        model,
        defenses=defenses,
        output_dir=args.output,
        model_name="DummyClassifier-CIFAR10",
    )
    report = benchmarker.run(x_test, y_test, attacks=attacks)

    logger.info(
        "Done. Worst-case robustness: %.2f%% | Best defended: %.2f%%",
        report.worst_case_robustness() * 100,
        report.best_defended_accuracy() * 100,
    )


if __name__ == "__main__":
    main()
