"""
models/base_model.py
---------------------
Abstract base class that all model wrappers in this framework must implement.
Defines the contract expected by the attack and defense modules so that
any model backend (PyTorch, ONNX, TFLite) can be plugged in without changing
attack/defense logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class BaseModel(ABC):
    """
    Abstract interface for classification models used in the robustness engine.

    All concrete models must implement ``forward()``, ``predict()``, and
    ``get_gradients()``.  The latter is used by gradient-based attacks
    (FGSM, PGD) to compute adversarial perturbations.
    """

    def __init__(self, num_classes: int, input_shape: Tuple[int, ...]) -> None:
        """
        Parameters
        ----------
        num_classes:  Number of output classes.
        input_shape:  Expected input shape as (C, H, W).
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self._is_training = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Run a forward pass.

        Parameters
        ----------
        x: Input array of shape (N, C, H, W), values in [0, 1].

        Returns
        -------
        Logits array of shape (N, num_classes).
        """

    @abstractmethod
    def get_gradients(self, x: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute loss gradients w.r.t. the input *x*.

        Parameters
        ----------
        x:      Input array (N, C, H, W).
        labels: True class indices (N,).

        Returns
        -------
        (gradients, loss_value) — gradients have the same shape as *x*.
        """

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Return predicted class indices for a batch.

        Parameters
        ----------
        x: Input array (N, C, H, W).

        Returns
        -------
        Integer array of shape (N,).
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def accuracy(self, x: np.ndarray, labels: np.ndarray) -> float:
        """Compute classification accuracy over a batch."""
        preds = self.predict(x)
        return float(np.mean(preds == labels))

    def train(self) -> "BaseModel":
        """Set model to training mode."""
        self._is_training = True
        return self

    def eval(self) -> "BaseModel":
        """Set model to evaluation mode."""
        self._is_training = False
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}, "
            f"input_shape={self.input_shape})"
        )
