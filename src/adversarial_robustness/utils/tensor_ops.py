"""
utils/tensor_ops.py
--------------------
Thin abstraction layer over NumPy that mirrors the PyTorch tensor API used
throughout the codebase.  When PyTorch is available, real ``torch.Tensor``
objects are used; otherwise, NumPy arrays are wrapped in a lightweight
shim so that all attack / defense code runs identically in both environments.

This design means every attack and defense can be unit-tested without a GPU
or a full PyTorch installation, while still being production-ready when
torch is present.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

_TORCH_AVAILABLE = False
try:
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    pass


def is_torch_available() -> bool:
    return _TORCH_AVAILABLE


def get_device(preference: str = "auto") -> str:
    """Return the best available compute device string."""
    if not _TORCH_AVAILABLE:
        return "numpy"
    if preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return preference


# ---------------------------------------------------------------------------
# NumPy shim tensor class (used when torch is absent)
# ---------------------------------------------------------------------------


class NumpyTensor:
    """
    Minimal NumPy-backed tensor that mirrors the PyTorch Tensor API used
    in attack/defense implementations.
    """

    def __init__(self, data: np.ndarray, requires_grad: bool = False) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self.shape: Tuple[int, ...] = self.data.shape

    # ------------------------------------------------------------------
    # Arithmetic  (enables: tensor + tensor, tensor * scalar, etc.)
    # ------------------------------------------------------------------

    def __add__(self, other):
        o = other.data if isinstance(other, NumpyTensor) else other
        return NumpyTensor(self.data + o)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        o = other.data if isinstance(other, NumpyTensor) else other
        return NumpyTensor(self.data - o)

    def __rsub__(self, other):
        o = other.data if isinstance(other, NumpyTensor) else other
        return NumpyTensor(o - self.data)

    def __mul__(self, other):
        o = other.data if isinstance(other, NumpyTensor) else other
        return NumpyTensor(self.data * o)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        o = other.data if isinstance(other, NumpyTensor) else other
        return NumpyTensor(self.data / o)

    def __neg__(self):
        return NumpyTensor(-self.data)

    def __repr__(self) -> str:
        return f"NumpyTensor(shape={self.shape}, dtype={self.data.dtype})"

    # ------------------------------------------------------------------
    # PyTorch-compatible methods
    # ------------------------------------------------------------------

    def clone(self) -> "NumpyTensor":
        return NumpyTensor(self.data.copy(), requires_grad=self.requires_grad)

    def detach(self) -> "NumpyTensor":
        return NumpyTensor(self.data.copy(), requires_grad=False)

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def clamp(self, min_val: float, max_val: float) -> "NumpyTensor":
        return NumpyTensor(np.clip(self.data, min_val, max_val))

    def sign(self) -> "NumpyTensor":
        return NumpyTensor(np.sign(self.data))

    def abs(self) -> "NumpyTensor":
        return NumpyTensor(np.abs(self.data))

    def mean(self) -> "NumpyTensor":
        return NumpyTensor(np.array(self.data.mean()))

    def norm(self, p: Union[int, float] = 2) -> "NumpyTensor":
        return NumpyTensor(np.array(np.linalg.norm(self.data.ravel(), ord=p)))

    def max(self) -> "NumpyTensor":
        return NumpyTensor(np.array(self.data.max()))

    def min(self) -> "NumpyTensor":
        return NumpyTensor(np.array(self.data.min()))

    def argmax(self, dim: int = -1) -> "NumpyTensor":
        return NumpyTensor(self.data.argmax(axis=dim).astype(np.int64))

    def item(self) -> float:
        return float(self.data.ravel()[0])

    def unsqueeze(self, dim: int) -> "NumpyTensor":
        return NumpyTensor(np.expand_dims(self.data, axis=dim))

    def squeeze(self, dim: Optional[int] = None) -> "NumpyTensor":
        return NumpyTensor(
            np.squeeze(self.data, axis=dim) if dim is not None else np.squeeze(self.data)
        )

    def reshape(self, *shape) -> "NumpyTensor":
        return NumpyTensor(self.data.reshape(*shape))

    def permute(self, *dims) -> "NumpyTensor":
        return NumpyTensor(np.transpose(self.data, dims))

    def float(self) -> "NumpyTensor":
        return NumpyTensor(self.data.astype(np.float32))

    def long(self) -> "NumpyTensor":
        return NumpyTensor(self.data.astype(np.int64))

    def cpu(self) -> "NumpyTensor":
        return self.clone()

    def to(self, *args, **kwargs) -> "NumpyTensor":
        return self.clone()

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        return NumpyTensor(self.data[idx])


# ---------------------------------------------------------------------------
# Factory functions mirroring torch.* API
# ---------------------------------------------------------------------------


def zeros(*shape, **kwargs) -> NumpyTensor:
    return NumpyTensor(np.zeros(shape, dtype=np.float32))


def ones(*shape, **kwargs) -> NumpyTensor:
    return NumpyTensor(np.ones(shape, dtype=np.float32))


def rand(*shape, **kwargs) -> NumpyTensor:
    return NumpyTensor(np.random.rand(*shape).astype(np.float32))


def randn(*shape, **kwargs) -> NumpyTensor:
    return NumpyTensor(np.random.randn(*shape).astype(np.float32))


def from_numpy(arr: np.ndarray) -> NumpyTensor:
    return NumpyTensor(arr.astype(np.float32))


def stack(tensors, dim: int = 0) -> NumpyTensor:
    arrays = [t.data if isinstance(t, NumpyTensor) else t for t in tensors]
    return NumpyTensor(np.stack(arrays, axis=dim))


def cat(tensors, dim: int = 0) -> NumpyTensor:
    arrays = [t.data if isinstance(t, NumpyTensor) else t for t in tensors]
    return NumpyTensor(np.concatenate(arrays, axis=dim))
