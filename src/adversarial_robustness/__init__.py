"""
Adversarial Robustness Engine
==============================
A defense framework for protecting safety-critical computer vision models
against adversarial attacks in real-time video analytics pipelines.

Modules
-------
attacks     — FGSM, PGD, and patch attack implementations
defenses    — Adversarial training, denoising preprocessors, feature squeezing
evaluation  — Robustness benchmarking and metric reporting
models      — Model wrappers and robust model architectures
utils       — Configuration, logging, tensor utilities
"""

__version__ = "1.0.0"
__author__ = "Adversarial Robustness Team"

from adversarial_robustness.utils.config import load_config
from adversarial_robustness.utils.logger import get_logger

__all__ = ["load_config", "get_logger", "__version__"]
