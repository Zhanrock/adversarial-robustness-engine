"""
attacks/patch_attack.py
------------------------
Adversarial Patch Attack
-------------------------
Reference: Brown et al. (2018) — "Adversarial Patch"
           https://arxiv.org/abs/1712.09665

Unlike FGSM/PGD (which apply imperceptible perturbations across the whole
image), a patch attack concentrates a visible perturbation in a small region.
This models real-world "physical" attacks such as:
  - Printed stickers placed on road signs / objects
  - "Cloaking patches" worn to fool person detectors
  - Logo-shaped patches that cause misclassification

Algorithm:
    Initialise patch p (random or solid colour)
    for t in range(num_steps):
        Place p at a random location in x
        Compute ∇_p L(model(x_with_patch), y_target)
        p ← clip(p + α · sign(∇_p))   # gradient ascent on adversarial loss
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from adversarial_robustness.attacks.base_attack import BaseAttack, AttackResult
from adversarial_robustness.models.base_model import BaseModel
from adversarial_robustness.utils.logger import get_logger

logger = get_logger(__name__)


class PatchAttack(BaseAttack):
    """
    Adversarial patch attack — visible, localised L-inf attack.

    Optimises a small patch that, when placed on any location in an image,
    causes misclassification.

    Parameters
    ----------
    model:          Target model.
    epsilon:        Maximum pixel perturbation for the patch (default 1.0
                    allows any pixel value within [clip_min, clip_max]).
    patch_size:     Side length of the square patch in pixels.
    num_steps:      Optimisation steps.
    step_size:      Gradient ascent step size.
    target_class:   If specified, attack tries to fool the model into
                    predicting this class (targeted attack).  None = untargeted.
    clip_min:       Minimum pixel value.
    clip_max:       Maximum pixel value.

    Example
    -------
    >>> attack = PatchAttack(model, patch_size=32, num_steps=200)
    >>> result = attack.generate(x_batch, labels)
    """

    def __init__(
        self,
        model: BaseModel,
        epsilon: float = 1.0,
        patch_size: int = 32,
        num_steps: int = 200,
        step_size: float = 0.01,
        target_class: Optional[int] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__(model, epsilon, clip_min, clip_max)

        if patch_size < 1:
            raise ValueError(f"patch_size must be ≥ 1, got {patch_size}")
        if num_steps < 1:
            raise ValueError(f"num_steps must be ≥ 1, got {num_steps}")
        if step_size <= 0:
            raise ValueError(f"step_size must be > 0, got {step_size}")

        self.patch_size    = patch_size
        self.num_steps     = num_steps
        self.step_size     = step_size
        self.target_class  = target_class
        self._rng          = np.random.default_rng(seed)

        # Patch initialised as random noise in [clip_min, clip_max]
        c = model.input_shape[0] if len(model.input_shape) >= 1 else 3
        self._patch: Optional[np.ndarray] = None
        self._channels = c

    def _init_patch(self) -> np.ndarray:
        """Initialise patch as random uniform noise."""
        return self._rng.uniform(
            self.clip_min, self.clip_max,
            size=(self._channels, self.patch_size, self.patch_size),
        ).astype(np.float32)

    def _apply_patch(
        self,
        x: np.ndarray,
        patch: np.ndarray,
        row: int,
        col: int,
    ) -> np.ndarray:
        """
        Place *patch* on each image in batch *x* at position (row, col).

        Parameters
        ----------
        x:     Batch (N, C, H, W).
        patch: Patch array (C, patch_size, patch_size).
        row:   Top-left row of patch placement.
        col:   Top-left column of patch placement.

        Returns
        -------
        Patched batch with same shape as x.
        """
        x_patched = x.copy()
        h, w = x.shape[2], x.shape[3]
        r_end = min(row + self.patch_size, h)
        c_end = min(col + self.patch_size, w)
        ph    = r_end - row
        pw    = c_end - col
        x_patched[:, :, row:r_end, col:c_end] = patch[:, :ph, :pw]
        return x_patched

    def _random_location(self, h: int, w: int) -> Tuple[int, int]:
        """Sample a random top-left corner for patch placement."""
        max_row = max(0, h - self.patch_size)
        max_col = max(0, w - self.patch_size)
        row = int(self._rng.integers(0, max_row + 1))
        col = int(self._rng.integers(0, max_col + 1))
        return row, col

    def generate(
        self,
        x: np.ndarray,
        labels: np.ndarray,
    ) -> AttackResult:
        """
        Optimise and apply an adversarial patch.

        Parameters
        ----------
        x:      Clean input batch (N, C, H, W).
        labels: True class indices (N,).

        Returns
        -------
        AttackResult with patched examples and statistics.
        """
        x      = np.asarray(x, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        n, c, h, w = x.shape

        if self.patch_size > min(h, w):
            raise ValueError(
                f"patch_size ({self.patch_size}) exceeds image dimensions "
                f"({h}x{w})"
            )

        logger.debug(
            "PatchAttack: batch=%d, patch=%dpx, steps=%d, targeted=%s",
            n, self.patch_size, self.num_steps,
            self.target_class is not None,
        )

        # ── 1. Initialise patch ───────────────────────────────────────────
        patch = self._init_patch()

        # Determine attack labels (targeted vs untargeted)
        if self.target_class is not None:
            attack_labels = np.full(n, self.target_class, dtype=np.int64)
        else:
            attack_labels = labels

        # ── 2. Patch optimisation loop ────────────────────────────────────
        for step in range(self.num_steps):
            row, col   = self._random_location(h, w)
            x_patched  = self._apply_patch(x, patch, row, col)

            # Gradient of loss w.r.t. whole image
            grad, loss = self.model.get_gradients(x_patched, attack_labels)

            # Extract gradient for the patch region only and average over batch
            grad_patch = grad[:, :, row:row + self.patch_size,
                                    col:col + self.patch_size]
            avg_grad   = grad_patch.mean(axis=0)        # (C, ph, pw)

            ph = min(self.patch_size, h - row)
            pw = min(self.patch_size, w - col)

            if self.target_class is not None:
                # Targeted: minimise loss → gradient descent
                patch[:, :ph, :pw] -= self.step_size * np.sign(avg_grad[:, :ph, :pw])
            else:
                # Untargeted: maximise loss → gradient ascent
                patch[:, :ph, :pw] += self.step_size * np.sign(avg_grad[:, :ph, :pw])

            # Clip patch values
            patch = np.clip(patch, self.clip_min, self.clip_max)

            if (step + 1) % 50 == 0:
                logger.debug("PatchAttack step %d/%d | loss=%.4f",
                             step + 1, self.num_steps, loss)

        self._patch = patch.copy()

        # ── 3. Apply final patch to all images (random locations) ─────────
        x_adv = np.zeros_like(x)
        for i in range(n):
            row, col       = self._random_location(h, w)
            x_adv[i:i+1]  = self._apply_patch(x[i:i+1], patch, row, col)

        result = self._compute_result(
            x, x_adv, labels,
            metadata={
                "patch_size":   self.patch_size,
                "num_steps":    self.num_steps,
                "target_class": self.target_class,
            },
        )

        logger.info(
            "PatchAttack complete — ASR: %.2f%% | mean L2: %.4f",
            result.attack_success_rate * 100,
            result.mean_l2_norm,
        )
        return result

    @property
    def optimised_patch(self) -> Optional[np.ndarray]:
        """Return the optimised patch after calling generate()."""
        return self._patch

    def __repr__(self) -> str:
        return (
            f"PatchAttack(patch_size={self.patch_size}, "
            f"num_steps={self.num_steps}, "
            f"target_class={self.target_class})"
        )
