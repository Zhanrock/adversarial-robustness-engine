"""
defenses/base_defense.py
-------------------------
Abstract base class for all defense (preprocessor) implementations.
Defenses are designed to be composable: they can be chained via
``DefensePipeline`` so that multiple preprocessing steps are applied
sequentially before images reach the model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from adversarial_robustness.utils.logger import get_logger

logger = get_logger(__name__)


class BaseDefense(ABC):
    """
    Abstract preprocessor defense.

    A defense transforms an input batch *x* → *x_clean* before the batch
    is fed to the model.  This design decouples defense logic from model
    logic, enabling modular composition and unit testing.
    """

    @abstractmethod
    def preprocess(self, x: np.ndarray) -> np.ndarray:
        """
        Apply defense preprocessing to a batch of inputs.

        Parameters
        ----------
        x: Input batch (N, C, H, W), values typically in [0, 1].

        Returns
        -------
        Preprocessed batch with the same shape as *x*.
        """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.preprocess(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class DefensePipeline(BaseDefense):
    """
    Chains multiple defenses sequentially.

    Example
    -------
    >>> pipeline = DefensePipeline([GaussianDenoiser(sigma=0.05),
    ...                             JPEGCompression(quality=75)])
    >>> x_clean = pipeline(x_adversarial)
    """

    def __init__(self, defenses: list[BaseDefense]) -> None:
        if not defenses:
            raise ValueError("DefensePipeline requires at least one defense.")
        self._defenses = list(defenses)

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        result = x
        for defense in self._defenses:
            result = defense.preprocess(result)
        return result

    def __repr__(self) -> str:
        names = " → ".join(d.__class__.__name__ for d in self._defenses)
        return f"DefensePipeline([{names}])"
