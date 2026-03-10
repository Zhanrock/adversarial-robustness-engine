"""
evaluation/metrics.py
----------------------
Standalone metric functions for adversarial robustness evaluation.
All functions operate on NumPy arrays and have no side effects,
making them easy to unit test and use in notebooks.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def clean_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of correctly classified clean examples."""
    return float(np.mean(y_pred == y_true))


def adversarial_accuracy(y_pred_adv: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of correctly classified adversarial examples."""
    return float(np.mean(y_pred_adv == y_true))


def attack_success_rate(
    y_pred_clean: np.ndarray,
    y_pred_adv: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """
    Fraction of originally-correct samples that the attack fools.

    ASR = |{i : y_clean[i]==y[i]  AND  y_adv[i]!=y[i]}|
          ─────────────────────────────────────────────
                  |{i : y_clean[i]==y[i]}|
    """
    was_correct   = y_pred_clean == y_true
    now_incorrect = y_pred_adv   != y_true
    denom = was_correct.sum()
    if denom == 0:
        return 0.0
    return float((was_correct & now_incorrect).sum() / denom)


def robustness_gap(clean_acc: float, adv_acc: float) -> float:
    """Difference between clean and adversarial accuracy."""
    return clean_acc - adv_acc


def mean_perturbation_norm(
    x_orig: np.ndarray,
    x_adv: np.ndarray,
    p: float = 2,
) -> float:
    """
    Mean L-p norm of perturbations (x_adv - x_orig) across the batch.

    Parameters
    ----------
    x_orig, x_adv: Arrays of shape (N, C, H, W).
    p:             Norm order (2 for L2, np.inf for L-inf).
    """
    diff = (x_adv - x_orig).reshape(len(x_orig), -1)
    norms = np.linalg.norm(diff, ord=p, axis=1)
    return float(norms.mean())


def certified_robustness_ratio(
    y_pred_adv: np.ndarray,
    y_true: np.ndarray,
    perturbation_norms: np.ndarray,
    epsilon: float,
) -> float:
    """
    Fraction of adversarial examples that both:
      (a) are within the ε-ball, AND
      (b) remain correctly classified.

    This is a proxy for certifiable robustness.
    """
    within_budget  = perturbation_norms <= epsilon
    still_correct  = y_pred_adv == y_true
    return float((within_budget & still_correct).sum() / max(len(y_true), 1))


def compute_all_metrics(
    x_orig: np.ndarray,
    x_adv: np.ndarray,
    y_true: np.ndarray,
    y_pred_clean: np.ndarray,
    y_pred_adv: np.ndarray,
    epsilon: float,
) -> Dict[str, float]:
    """
    Compute the full suite of robustness metrics in one call.

    Returns a flat dict suitable for logging and JSON serialisation.
    """
    perturbation_norms_l2   = (x_adv - x_orig).reshape(len(x_orig), -1)
    perturbation_norms_linf = np.abs(perturbation_norms_l2).max(axis=1)
    l2_norms = np.linalg.norm(perturbation_norms_l2, axis=1)

    return {
        "clean_accuracy":          clean_accuracy(y_pred_clean, y_true),
        "adversarial_accuracy":    adversarial_accuracy(y_pred_adv, y_true),
        "attack_success_rate":     attack_success_rate(y_pred_clean, y_pred_adv, y_true),
        "robustness_gap":          robustness_gap(
                                       clean_accuracy(y_pred_clean, y_true),
                                       adversarial_accuracy(y_pred_adv, y_true),
                                   ),
        "mean_l2_perturbation":    float(l2_norms.mean()),
        "mean_linf_perturbation":  float(perturbation_norms_linf.mean()),
        "max_linf_perturbation":   float(perturbation_norms_linf.max()),
        "certified_robustness":    certified_robustness_ratio(
                                       y_pred_adv, y_true, l2_norms, epsilon
                                   ),
    }
