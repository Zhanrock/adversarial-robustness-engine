"""
adversarial_robustness/cli.py
------------------------------
CLI entry points registered in pyproject.toml.

Commands
--------
are-evaluate   Run a robustness evaluation benchmark
are-benchmark  Full benchmark suite with all attacks
"""

from __future__ import annotations

import argparse
import os


def evaluate() -> None:
    """Entry point: ``are-evaluate``"""
    parser = argparse.ArgumentParser(
        prog="are-evaluate",
        description="Evaluate adversarial robustness of a model",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--attack",
        choices=["fgsm", "pgd", "patch", "all"],
        default="all",
        help="Attack to evaluate against (default: all)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Override epsilon from config",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for evaluation reports",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100)",
    )
    parser.add_argument(
        "--no-defense",
        action="store_true",
        help="Disable preprocessor defenses",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Configure logging
    from adversarial_robustness.utils.logger import configure_logging

    configure_logging(level="DEBUG" if args.verbose else "INFO")

    from adversarial_robustness.utils.config import load_config
    from adversarial_robustness.utils.logger import get_logger

    logger = get_logger(__name__)

    # Load config
    if not os.path.exists(args.config):
        logger.warning("Config not found at %s, using defaults", args.config)
        cfg_epsilon = 0.03
    else:
        cfg = load_config(args.config)
        cfg_epsilon = cfg.attacks.fgsm.epsilon

    epsilon = args.epsilon if args.epsilon is not None else cfg_epsilon

    # Generate synthetic test data (replace with real data loader in production)
    import numpy as np

    from adversarial_robustness.attacks.fgsm import FGSM
    from adversarial_robustness.attacks.patch_attack import PatchAttack
    from adversarial_robustness.attacks.pgd import PGD
    from adversarial_robustness.defenses.denoiser import GaussianDenoiser
    from adversarial_robustness.evaluation.benchmarker import RobustnessBenchmarker
    from adversarial_robustness.models.dummy_model import DummyClassifier

    logger.info("Generating %d synthetic test samples...", args.num_samples)
    rng = np.random.default_rng(42)
    input_shape = (3, 32, 32)
    x_test = rng.random((args.num_samples, *input_shape)).astype(np.float32)
    y_test = rng.integers(0, 10, size=args.num_samples).astype(np.int64)

    model = DummyClassifier(num_classes=10, input_shape=input_shape)
    defenses = [] if args.no_defense else [GaussianDenoiser(sigma=0.05)]

    # Build attacks
    all_attacks = {
        "fgsm": FGSM(model, epsilon=epsilon),
        "pgd": PGD(model, epsilon=epsilon, num_steps=20),
        "patch": PatchAttack(model, patch_size=8, num_steps=50),
    }
    attacks = list(all_attacks.values()) if args.attack == "all" else [all_attacks[args.attack]]

    benchmarker = RobustnessBenchmarker(
        model,
        defenses=defenses,
        output_dir=args.output_dir,
        model_name="DummyClassifier",
    )
    report = benchmarker.run(x_test, y_test, attacks=attacks)
    logger.info("Evaluation complete. Report saved to: %s", args.output_dir)


def train() -> None:
    """Entry point: ``are-train``"""
    from adversarial_robustness.utils.logger import configure_logging

    configure_logging()
    from adversarial_robustness.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Adversarial training CLI — see README for usage details.")


def benchmark() -> None:
    """Entry point: ``are-benchmark``"""
    evaluate()


if __name__ == "__main__":
    evaluate()
