"""Evaluation utilities: benchmarking and metrics."""

from adversarial_robustness.evaluation.benchmarker import (
    AttackEvalResult,
    BenchmarkReport,
    RobustnessBenchmarker,
)
from adversarial_robustness.evaluation.metrics import (
    adversarial_accuracy,
    attack_success_rate,
    clean_accuracy,
    compute_all_metrics,
    mean_perturbation_norm,
    robustness_gap,
)

__all__ = [
    "RobustnessBenchmarker",
    "BenchmarkReport",
    "AttackEvalResult",
    "clean_accuracy",
    "adversarial_accuracy",
    "attack_success_rate",
    "robustness_gap",
    "mean_perturbation_norm",
    "compute_all_metrics",
]
