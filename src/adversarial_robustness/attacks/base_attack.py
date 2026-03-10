"""
attacks/base_attack.py
-----------------------
Abstract base class for all adversarial attack implementations.
Enforces a consistent API so that attacks are interchangeable in
evaluation loops, adversarial training, and benchmarks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from adversarial_robustness.models.base_model import BaseModel
from adversarial_robustness.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AttackResult:
    """
    Container for the output of a single attack run.

    Attributes
    ----------
    adversarial_examples:  Perturbed inputs, same shape as originals.
    perturbations:         Difference x_adv - x_orig.
    success_flags:         Boolean array — True where attack fooled the model.
    attack_success_rate:   Fraction of inputs successfully attacked.
    mean_l2_norm:          Mean L2 norm of perturbations across the batch.
    mean_linf_norm:        Mean L-inf norm of perturbations across the batch.
    metadata:              Extra attack-specific metrics.
    """

    adversarial_examples: np.ndarray
    perturbations: np.ndarray
    success_flags: np.ndarray
    attack_success_rate: float
    mean_l2_norm: float
    mean_linf_norm: float
    metadata: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"AttackResult("
            f"ASR={self.attack_success_rate:.3f}, "
            f"mean_L2={self.mean_l2_norm:.4f}, "
            f"mean_Linf={self.mean_linf_norm:.4f})"
        )


class BaseAttack(ABC):
    """
    Abstract base class for adversarial attacks.

    All attacks accept a model and configuration at construction time,
    then expose a single ``generate()`` method that returns an
    ``AttackResult``.
    """

    def __init__(
        self,
        model: BaseModel,
        epsilon: float,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        model:    Target model to attack.
        epsilon:  Maximum L-inf perturbation budget.
        clip_min: Lower bound for pixel values after perturbation.
        clip_max: Upper bound for pixel values after perturbation.
        """
        if not 0.0 <= epsilon:
            raise ValueError(f"epsilon must be ≥ 0, got {epsilon}")
        if clip_min >= clip_max:
            raise ValueError("clip_min must be < clip_max")

        self.model = model
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max

    @abstractmethod
    def generate(
        self,
        x: np.ndarray,
        labels: np.ndarray,
    ) -> AttackResult:
        """
        Generate adversarial examples.

        Parameters
        ----------
        x:      Clean inputs (N, C, H, W), values in [clip_min, clip_max].
        labels: True class labels (N,).

        Returns
        -------
        AttackResult containing perturbed inputs and statistics.
        """

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _compute_result(
        self,
        x_orig: np.ndarray,
        x_adv: np.ndarray,
        labels: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> AttackResult:
        """Compute standard metrics and package into an AttackResult."""
        perturbations = x_adv - x_orig

        # Attack success: model prediction changes from correct label
        orig_preds = self.model.predict(x_orig)
        adv_preds = self.model.predict(x_adv)

        was_correct = orig_preds == labels
        now_incorrect = adv_preds != labels
        success_flags = was_correct & now_incorrect

        asr = float(success_flags.sum() / max(was_correct.sum(), 1))

        # Norm statistics (per-sample, then averaged)
        flat = perturbations.reshape(len(perturbations), -1)
        l2 = float(np.linalg.norm(flat, axis=1).mean())
        linf = float(np.abs(flat).max(axis=1).mean())

        return AttackResult(
            adversarial_examples=x_adv,
            perturbations=perturbations,
            success_flags=success_flags,
            attack_success_rate=asr,
            mean_l2_norm=l2,
            mean_linf_norm=linf,
            metadata=metadata or {},
        )

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.clip_min, self.clip_max)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"epsilon={self.epsilon}, "
            f"clip=[{self.clip_min}, {self.clip_max}])"
        )
