"""
attacks/fgsm.py
---------------
Fast Gradient Sign Method (FGSM)
---------------------------------
Reference: Goodfellow et al. (2015) — "Explaining and Harnessing Adversarial Examples"
           https://arxiv.org/abs/1412.6572

FGSM generates adversarial examples in a single gradient step:

    x_adv = x + ε · sign(∇_x L(θ, x, y))

where:
  - x         is the clean input
  - ε         is the perturbation budget (L-inf)
  - ∇_x L     is the gradient of the loss w.r.t. the input
  - θ         are the model parameters

This is the fastest attack and serves as a baseline.  It is also the
foundation of adversarial training (Madry et al., 2017).
"""

from __future__ import annotations

import numpy as np

from adversarial_robustness.attacks.base_attack import AttackResult, BaseAttack
from adversarial_robustness.models.base_model import BaseModel
from adversarial_robustness.utils.logger import get_logger

logger = get_logger(__name__)


class FGSM(BaseAttack):
    """
    Fast Gradient Sign Method (FGSM) — single-step L-inf attack.

    Parameters
    ----------
    model:    Target model.
    epsilon:  L-inf perturbation budget.  Typical values: 0.01 – 0.1.
    clip_min: Minimum pixel value after perturbation (default: 0.0).
    clip_max: Maximum pixel value after perturbation (default: 1.0).

    Example
    -------
    >>> attack = FGSM(model, epsilon=0.03)
    >>> result = attack.generate(x_batch, labels)
    >>> print(result.attack_success_rate)
    """

    def generate(
        self,
        x: np.ndarray,
        labels: np.ndarray,
    ) -> AttackResult:
        """
        Generate FGSM adversarial examples.

        Parameters
        ----------
        x:      Clean input batch (N, C, H, W), values in [clip_min, clip_max].
        labels: True class indices (N,).

        Returns
        -------
        AttackResult with perturbed examples and attack statistics.
        """
        x = np.asarray(x, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        if x.ndim != 4:
            raise ValueError(f"Expected 4-D input (N, C, H, W), got shape {x.shape}")

        logger.debug("FGSM: batch_size=%d, epsilon=%.4f", x.shape[0], self.epsilon)

        # ── 1. Compute gradient of loss w.r.t. input ─────────────────────
        grad, loss = self.model.get_gradients(x, labels)

        # ── 2. Apply signed gradient perturbation ────────────────────────
        perturbation = self.epsilon * np.sign(grad)
        x_adv = self._clip(x + perturbation)

        logger.debug("FGSM: loss=%.4f", loss)

        result = self._compute_result(
            x,
            x_adv,
            labels,
            metadata={"loss": loss, "epsilon": self.epsilon},
        )

        logger.info(
            "FGSM complete — ASR: %.2f%% | mean L2: %.4f | mean Linf: %.4f",
            result.attack_success_rate * 100,
            result.mean_l2_norm,
            result.mean_linf_norm,
        )
        return result
