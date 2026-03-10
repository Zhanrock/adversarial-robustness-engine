"""
models/dummy_model.py
---------------------
A NumPy-backed dummy classifier used for:
  - Unit testing attacks and defenses without PyTorch
  - CI/CD pipelines without GPU dependencies
  - Rapid prototyping and benchmarking

The dummy model applies a fixed linear projection (random weights seeded
for reproducibility) followed by a softmax.  Gradients are computed
analytically via the cross-entropy loss gradient formula.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from adversarial_robustness.models.base_model import BaseModel
from adversarial_robustness.utils.logger import get_logger

logger = get_logger(__name__)


class DummyClassifier(BaseModel):
    """
    Lightweight linear classifier backed entirely by NumPy.

    Architecture:  Flatten → Linear(input_dim, num_classes) → Softmax
    Gradient:      d(cross_entropy) / d(x)  via chain rule

    Parameters
    ----------
    num_classes:  Number of output classes.
    input_shape:  Input shape as (C, H, W).
    seed:         Random seed for weight initialisation (default: 42).
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_shape: Tuple[int, ...] = (3, 32, 32),
        seed: int = 42,
    ) -> None:
        super().__init__(num_classes, input_shape)
        rng = np.random.default_rng(seed)

        self._input_dim = int(np.prod(input_shape))
        # Xavier initialisation
        scale = np.sqrt(2.0 / (self._input_dim + num_classes))
        self._W = rng.standard_normal((self._input_dim, num_classes)).astype(np.float32) * scale
        self._b = np.zeros(num_classes, dtype=np.float32)

        logger.debug(
            "DummyClassifier initialised: input_dim=%d, num_classes=%d",
            self._input_dim,
            num_classes,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flatten(self, x: np.ndarray) -> np.ndarray:
        """Flatten spatial dims: (N, C, H, W) → (N, C*H*W)."""
        return x.reshape(x.shape[0], -1).astype(np.float32)

    def _logits(self, x: np.ndarray) -> np.ndarray:
        """Compute raw logits: (N, num_classes)."""
        return (self._flatten(x) @ self._W + self._b).astype(np.float32)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return logits (N, num_classes)."""
        return self._logits(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predicted class indices (N,)."""
        return self._logits(x).argmax(axis=1).astype(np.int64)

    def get_gradients(self, x: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute cross-entropy loss gradient w.r.t. input *x*.

        The gradient is:
            dL/dx = (softmax(Wx+b) - one_hot(y)) @ W^T  reshaped to x.shape

        Returns
        -------
        (grad_x, loss_scalar)
        """
        n = x.shape[0]
        logits = self._logits(x)  # (N, C)
        probs = self._softmax(logits)  # (N, C)

        # Cross-entropy loss
        log_probs = np.log(probs[np.arange(n), labels] + 1e-12)
        loss = -log_probs.mean()

        # Gradient of cross-entropy w.r.t. logits
        dlogits = probs.copy()  # (N, C)
        dlogits[np.arange(n), labels] -= 1.0
        dlogits /= n

        # Gradient w.r.t. flattened input
        d_flat = dlogits @ self._W.T  # (N, input_dim)

        # Reshape back to input shape
        grad_x = d_flat.reshape(x.shape).astype(np.float32)

        return grad_x, float(loss)

    def __repr__(self) -> str:
        return (
            f"DummyClassifier(num_classes={self.num_classes}, "
            f"input_shape={self.input_shape}, "
            f"input_dim={self._input_dim})"
        )


class PytorchModelWrapper(BaseModel):
    """
    Wraps a real ``torch.nn.Module`` to satisfy the ``BaseModel`` interface.

    Usage
    -----
    >>> import torchvision.models as tv
    >>> torch_model = tv.resnet18(pretrained=False)
    >>> model = PytorchModelWrapper(torch_model, num_classes=10)
    >>> logits = model.forward(batch_numpy)
    """

    def __init__(
        self,
        module,  # torch.nn.Module
        num_classes: int,
        input_shape: Tuple[int, ...] = (3, 224, 224),
        device: str = "cpu",
    ) -> None:
        super().__init__(num_classes, input_shape)
        try:
            import torch

            self._torch = torch
        except ImportError as exc:
            raise ImportError(
                "PytorchModelWrapper requires PyTorch. "
                "Install with: pip install torch torchvision"
            ) from exc

        self._device = torch.device(device)
        self._module = module.to(self._device)
        self._loss_fn = torch.nn.CrossEntropyLoss()

    def _to_tensor(self, x: np.ndarray):
        return self._torch.tensor(x, dtype=self._torch.float32, device=self._device)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._module.eval()
        with self._torch.no_grad():
            logits = self._module(self._to_tensor(x))
        return logits.cpu().numpy()

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x)
        return logits.argmax(axis=1).astype(np.int64)

    def get_gradients(self, x: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, float]:
        self._module.eval()
        t_x = self._to_tensor(x).requires_grad_(True)
        t_y = self._torch.tensor(labels, dtype=self._torch.long, device=self._device)
        logits = self._module(t_x)
        loss = self._loss_fn(logits, t_y)
        loss.backward()
        grad = t_x.grad.cpu().numpy()
        return grad, float(loss.item())
