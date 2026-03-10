"""
Defense modules.

Available defenses
------------------
GaussianDenoiser     — Gaussian blur preprocessing
MedianDenoiser       — Median filter preprocessing
FeatureSqueezing     — Colour bit-depth reduction
AdversarialTrainer   — PGD/FGSM adversarial training loop
DefensePipeline      — Chain multiple defenses
"""

from adversarial_robustness.defenses.base_defense import BaseDefense, DefensePipeline
from adversarial_robustness.defenses.denoiser import (
    GaussianDenoiser,
    MedianDenoiser,
    FeatureSqueezing,
)
from adversarial_robustness.defenses.adversarial_training import (
    AdversarialTrainer,
    TrainingHistory,
    TrainingMetrics,
)

__all__ = [
    "BaseDefense", "DefensePipeline",
    "GaussianDenoiser", "MedianDenoiser", "FeatureSqueezing",
    "AdversarialTrainer", "TrainingHistory", "TrainingMetrics",
]
