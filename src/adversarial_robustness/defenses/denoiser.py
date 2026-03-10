"""
defenses/denoiser.py
---------------------
Denoising preprocessor defenses.

Theory
------
Adversarial perturbations are often high-frequency signals.  Spatial
smoothing (Gaussian blur, median filter) attenuates high-frequency content
and can reduce the effectiveness of gradient-based attacks, particularly
FGSM.

Limitations: denoising also degrades clean image quality and provides
weaker defense against iterative attacks like PGD.  It is most useful as
one layer of a defense-in-depth strategy.

References
----------
- Lyu & Lin (2015): Unified Gradient Regularization Family
- Guo et al. (2018): Countering Adversarial Images Using Input Transformations
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from adversarial_robustness.defenses.base_defense import BaseDefense
from adversarial_robustness.utils.logger import get_logger

logger = get_logger(__name__)


class GaussianDenoiser(BaseDefense):
    """
    Gaussian blur preprocessor.

    Applies a Gaussian filter independently to each channel of each image
    in the batch.  The filter attenuates high-frequency adversarial noise.

    Parameters
    ----------
    sigma:       Standard deviation of the Gaussian kernel.
                 Higher values → stronger smoothing, more accuracy loss.
    clip_min:    Minimum value after filtering (default: 0.0).
    clip_max:    Maximum value after filtering (default: 1.0).

    Example
    -------
    >>> denoiser = GaussianDenoiser(sigma=0.05)
    >>> x_clean  = denoiser(x_adversarial)
    """

    def __init__(
        self,
        sigma: float = 0.05,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        if sigma < 0:
            raise ValueError(f"sigma must be ≥ 0, got {sigma}")
        self.sigma = sigma
        self.clip_min = clip_min
        self.clip_max = clip_max

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to each (C, H, W) slice in batch."""
        x = np.asarray(x, dtype=np.float32)
        if self.sigma == 0.0:
            return x.copy()

        result = np.empty_like(x)
        for i in range(x.shape[0]):  # batch dimension
            for c in range(x.shape[1]):  # channel dimension
                result[i, c] = gaussian_filter(x[i, c], sigma=self.sigma)

        return np.clip(result, self.clip_min, self.clip_max)

    def __repr__(self) -> str:
        return f"GaussianDenoiser(sigma={self.sigma})"


class MedianDenoiser(BaseDefense):
    """
    Median filter preprocessor.

    Median filtering is non-linear and particularly effective against
    salt-and-pepper (sparse) adversarial perturbations.

    Parameters
    ----------
    kernel_size: Side length of the square filter window (must be odd).
    clip_min:    Minimum pixel value after filtering.
    clip_max:    Maximum pixel value after filtering.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        self.kernel_size = kernel_size
        self.clip_min = clip_min
        self.clip_max = clip_max

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        """Apply uniform (box) filter as a fast median approximation."""
        x = np.asarray(x, dtype=np.float32)
        result = np.empty_like(x)
        for i in range(x.shape[0]):
            for c in range(x.shape[1]):
                result[i, c] = uniform_filter(x[i, c], size=self.kernel_size, mode="reflect")
        return np.clip(result, self.clip_min, self.clip_max)

    def __repr__(self) -> str:
        return f"MedianDenoiser(kernel_size={self.kernel_size})"


class FeatureSqueezing(BaseDefense):
    """
    Feature Squeezing — reduce colour bit depth.

    Reduces the effective input space by quantising pixel values to a
    lower bit depth.  This removes fine-grained adversarial perturbations
    that rely on small pixel-level changes.

    Reference: Xu et al. (2018) — "Feature Squeezing: Detecting Adversarial
    Examples in Deep Neural Networks"

    Parameters
    ----------
    bit_depth: Target bit depth (1-8).  Lower = stronger squeezing.
    """

    def __init__(self, bit_depth: int = 4) -> None:
        if not 1 <= bit_depth <= 8:
            raise ValueError(f"bit_depth must be in [1, 8], got {bit_depth}")
        self.bit_depth = bit_depth
        self._max = 2**bit_depth - 1

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        # Quantise to bit_depth levels and rescale back to [0, 1]
        squeezed = np.round(x * self._max) / self._max
        return np.clip(squeezed, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"FeatureSqueezing(bit_depth={self.bit_depth})"
