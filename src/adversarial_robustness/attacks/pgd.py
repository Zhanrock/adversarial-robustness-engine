"""
attacks/pgd.py
--------------
Projected Gradient Descent (PGD) Attack
-----------------------------------------
Reference: Madry et al. (2018) — "Towards Deep Learning Models Resistant to Adversarial Attacks"
           https://arxiv.org/abs/1706.06083

PGD is the iterative extension of FGSM.  It is considered the gold-standard
white-box attack and is widely used for adversarial training.

Algorithm:
    x_0 = x + Uniform(-ε, ε)          (random start, optional)
    for t in range(num_steps):
        x_{t+1} = Π_{x+S} [ x_t + α · sign(∇_x L(θ, x_t, y)) ]

where:
  - Π_{x+S}  projects back into the ε-ball around the original x (clipping)
  - α         is the step size per iteration
  - S         is the set {δ : ||δ||_inf ≤ ε}
"""

from __future__ import annotations

from typing import List

import numpy as np

from adversarial_robustness.attacks.base_attack import AttackResult, BaseAttack
from adversarial_robustness.models.base_model import BaseModel
from adversarial_robustness.utils.logger import get_logger

logger = get_logger(__name__)


class PGD(BaseAttack):
    """
    Projected Gradient Descent (PGD) — multi-step L-inf attack.

    Parameters
    ----------
    model:        Target model.
    epsilon:      L-inf perturbation budget.
    alpha:        Step size per iteration.  Rule of thumb: epsilon / num_steps * 2.5.
    num_steps:    Number of PGD iterations (default: 40).
    random_start: If True, initialise with random perturbation inside ε-ball.
    clip_min:     Minimum pixel value (default: 0.0).
    clip_max:     Maximum pixel value (default: 1.0).

    Example
    -------
    >>> attack = PGD(model, epsilon=0.03, alpha=0.007, num_steps=40)
    >>> result = attack.generate(x_batch, labels)
    """

    def __init__(
        self,
        model: BaseModel,
        epsilon: float,
        alpha: float = 0.007,
        num_steps: int = 40,
        random_start: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__(model, epsilon, clip_min, clip_max)

        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if num_steps < 1:
            raise ValueError(f"num_steps must be ≥ 1, got {num_steps}")

        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start

    def generate(
        self,
        x: np.ndarray,
        labels: np.ndarray,
    ) -> AttackResult:
        """
        Generate PGD adversarial examples.

        Parameters
        ----------
        x:      Clean input batch (N, C, H, W).
        labels: True class indices (N,).

        Returns
        -------
        AttackResult with the strongest found adversarial examples.
        """
        x = np.asarray(x, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        if x.ndim != 4:
            raise ValueError(f"Expected 4-D input (N, C, H, W), got shape {x.shape}")

        logger.debug(
            "PGD: batch_size=%d, epsilon=%.4f, alpha=%.4f, steps=%d, random_start=%s",
            x.shape[0],
            self.epsilon,
            self.alpha,
            self.num_steps,
            self.random_start,
        )

        # ── 1. Initialise starting point ─────────────────────────────────
        if self.random_start:
            # Random uniform initialisation within ε-ball
            noise = np.random.uniform(-self.epsilon, self.epsilon, size=x.shape).astype(np.float32)
            x_adv = self._clip(x + noise)
        else:
            x_adv = x.copy()

        # ── 2. PGD iteration loop ─────────────────────────────────────────
        loss_history: List[float] = []

        for step in range(self.num_steps):
            # Gradient of cross-entropy loss w.r.t. current adversarial example
            grad, loss = self.model.get_gradients(x_adv, labels)
            loss_history.append(loss)

            # Signed gradient step
            x_adv = x_adv + self.alpha * np.sign(grad)

            # Project back into ε-ball centred on original x
            delta = np.clip(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = self._clip(x + delta)

            if (step + 1) % 10 == 0:
                logger.debug("PGD step %d/%d | loss=%.4f", step + 1, self.num_steps, loss)

        result = self._compute_result(
            x,
            x_adv,
            labels,
            metadata={
                "alpha": self.alpha,
                "num_steps": self.num_steps,
                "random_start": self.random_start,
                "epsilon": self.epsilon,
                "final_loss": loss_history[-1] if loss_history else 0.0,
                "loss_history": loss_history,
            },
        )

        logger.info(
            "PGD complete — ASR: %.2f%% | mean L2: %.4f | mean Linf: %.4f | " "final_loss: %.4f",
            result.attack_success_rate * 100,
            result.mean_l2_norm,
            result.mean_linf_norm,
            loss_history[-1] if loss_history else 0.0,
        )
        return result

    def __repr__(self) -> str:
        return (
            f"PGD(epsilon={self.epsilon}, alpha={self.alpha}, "
            f"num_steps={self.num_steps}, random_start={self.random_start})"
        )
