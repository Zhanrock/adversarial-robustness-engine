"""Evaluation utilities: benchmarking and metrics."""

from adversarial_robustness.evaluation.benchmarker import (
    RobustnessBenchmarker,
    BenchmarkReport,
    AttackEvalResult,
)
from adversarial_robustness.evaluation.metrics import (
    clean_accuracy,
    adversarial_accuracy,
    attack_success_rate,
    robustness_gap,
    mean_perturbation_norm,
    compute_all_metrics,
)

__all__ = [
    "RobustnessBenchmarker", "BenchmarkReport", "AttackEvalResult",
    "clean_accuracy", "adversarial_accuracy", "attack_success_rate",
    "robustness_gap", "mean_perturbation_norm", "compute_all_metrics",
]
