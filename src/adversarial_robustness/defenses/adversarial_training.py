"""
defenses/adversarial_training.py
----------------------------------
Adversarial Training (Madry et al., 2018)
------------------------------------------
The gold-standard empirical defense.  During training, a fraction of each
mini-batch is replaced with adversarially perturbed examples generated
on-the-fly using PGD (or FGSM for speed).  This forces the model to learn
features that are robust to worst-case perturbations.

Key insight: a model trained on adversarial examples develops internal
representations that are less sensitive to adversarial perturbations,
because it has "seen" such perturbations during every weight update.

Reference: Madry et al. (2018) — "Towards Deep Learning Models Resistant
           to Adversarial Attacks"  https://arxiv.org/abs/1706.06083
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np

from adversarial_robustness.attacks.fgsm import FGSM
from adversarial_robustness.attacks.pgd import PGD
from adversarial_robustness.models.base_model import BaseModel
from adversarial_robustness.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics recorded at each epoch of adversarial training."""

    epoch: int
    train_clean_acc: float
    train_adv_acc: float
    val_clean_acc: float
    val_adv_acc: float
    epoch_time_sec: float
    loss: float = 0.0


@dataclass
class TrainingHistory:
    """Full training history, aggregated across epochs."""

    epochs: List[TrainingMetrics] = field(default_factory=list)

    def best_epoch(self) -> Optional[TrainingMetrics]:
        if not self.epochs:
            return None
        return max(self.epochs, key=lambda e: e.val_adv_acc)

    def summary(self) -> dict:
        if not self.epochs:
            return {}
        best = self.best_epoch()
        return {
            "total_epochs": len(self.epochs),
            "best_epoch": best.epoch if best else None,
            "best_val_adv_acc": best.val_adv_acc if best else None,
            "best_val_clean_acc": best.val_clean_acc if best else None,
            "final_val_adv_acc": self.epochs[-1].val_adv_acc,
            "final_val_clean_acc": self.epochs[-1].val_clean_acc,
        }


class AdversarialTrainer:
    """
    Adversarial training loop.

    Trains a model using a mixture of clean and adversarially perturbed
    examples.  Uses the DummyClassifier (numpy) for CPU-based training in
    environments without PyTorch; in production, wraps a PyTorch model via
    ``PytorchModelWrapper``.

    Parameters
    ----------
    model:        Model to train (must implement BaseModel.get_gradients).
    attack:       Attack used to generate adversarial examples ("fgsm" | "pgd").
    epsilon:      Perturbation budget for training attack.
    ratio:        Fraction of each batch that is adversarial (0.0–1.0).
    epochs:       Number of training epochs.
    batch_size:   Mini-batch size.
    learning_rate: SGD learning rate (used for DummyClassifier simulation).

    Example
    -------
    >>> trainer = AdversarialTrainer(model, attack="pgd", epsilon=0.03)
    >>> history = trainer.fit(x_train, y_train, x_val, y_val)
    >>> print(history.summary())
    """

    def __init__(
        self,
        model: BaseModel,
        attack: str = "pgd",
        epsilon: float = 0.03,
        ratio: float = 0.5,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        pgd_steps: int = 7,
        pgd_alpha: float = 0.01,
    ) -> None:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"ratio must be in [0.0, 1.0], got {ratio}")

        self.model = model
        self.ratio = ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Build attack
        if attack == "fgsm":
            self._attack: Union[FGSM, PGD] = FGSM(model, epsilon=epsilon)
        elif attack == "pgd":
            self._attack = PGD(model, epsilon=epsilon, alpha=pgd_alpha, num_steps=pgd_steps)
        else:
            raise ValueError(f"Unknown attack: '{attack}'. Use 'fgsm' or 'pgd'.")

        logger.info(
            "AdversarialTrainer initialised: attack=%s, epsilon=%.3f, ratio=%.2f, epochs=%d",
            attack,
            epsilon,
            ratio,
            epochs,
        )

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> TrainingHistory:
        """
        Run the adversarial training loop.

        Parameters
        ----------
        x_train, y_train: Training data and labels.
        x_val, y_val:     Optional validation data for epoch metrics.

        Returns
        -------
        TrainingHistory with per-epoch metrics.
        """
        history = TrainingHistory()
        n = len(x_train)

        logger.info("Starting adversarial training: %d samples, %d epochs", n, self.epochs)

        for epoch in range(1, self.epochs + 1):
            t0 = time.monotonic()

            # Shuffle
            idx = np.random.permutation(n)
            x_shuffled = x_train[idx]
            y_shuffled = y_train[idx]

            epoch_losses: List[float] = []
            adv_correct = clean_correct = total = 0

            # Mini-batch loop
            for start in range(0, n, self.batch_size):
                x_b = x_shuffled[start : start + self.batch_size]
                y_b = y_shuffled[start : start + self.batch_size]
                bs = len(x_b)
                total += bs

                # Generate adversarial portion
                n_adv = max(1, int(bs * self.ratio))
                x_adv_b = self._attack.generate(x_b[:n_adv], y_b[:n_adv]).adversarial_examples

                # Mixed batch
                x_mixed = np.concatenate([x_adv_b, x_b[n_adv:]], axis=0)
                y_mixed = y_b  # labels unchanged

                # Simulated gradient-descent weight update on DummyClassifier
                grad_x, loss = self.model.get_gradients(x_mixed, y_mixed)
                epoch_losses.append(loss)

                # Simulate a weight update: nudge W toward reducing loss
                # (This is a proxy for real backprop — see PytorchModelWrapper
                #  for actual torch optimizer integration.)
                flat_grad = grad_x.reshape(bs, -1)
                self.model._W -= (  # type: ignore[attr-defined]
                    self.learning_rate
                    * (flat_grad.T @ np.eye(bs, self.model.num_classes)[y_mixed])
                    / bs
                    if hasattr(self.model, "_W")
                    else None
                )

                # Track accuracy
                clean_correct += int((self.model.predict(x_b) == y_b).sum())
                adv_correct += int((self.model.predict(x_adv_b) == y_b[:n_adv]).sum())

            # Epoch metrics
            train_clean_acc = clean_correct / total
            train_adv_acc = adv_correct / max(1, int(total * self.ratio))

            val_clean_acc = val_adv_acc = 0.0
            if x_val is not None and y_val is not None:
                val_clean_acc = self.model.accuracy(x_val, y_val)
                val_adv = self._attack.generate(x_val, y_val)
                val_adv_acc = self.model.accuracy(val_adv.adversarial_examples, y_val)

            elapsed = time.monotonic() - t0
            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

            metrics = TrainingMetrics(
                epoch=epoch,
                train_clean_acc=train_clean_acc,
                train_adv_acc=train_adv_acc,
                val_clean_acc=val_clean_acc,
                val_adv_acc=val_adv_acc,
                epoch_time_sec=elapsed,
                loss=avg_loss,
            )
            history.epochs.append(metrics)

            logger.info(
                "Epoch %3d/%d | loss=%.4f | "
                "train_clean=%.3f | train_adv=%.3f | "
                "val_clean=%.3f | val_adv=%.3f | "
                "time=%.1fs",
                epoch,
                self.epochs,
                avg_loss,
                train_clean_acc,
                train_adv_acc,
                val_clean_acc,
                val_adv_acc,
                elapsed,
            )

        logger.info("Training complete. %s", history.summary())
        return history
