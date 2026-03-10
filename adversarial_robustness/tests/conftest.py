"""
tests/conftest.py
------------------
Shared pytest fixtures used across all test modules.
Fixtures are structured to mirror real-world use:
  - Fixed random seeds for reproducibility
  - Multiple batch sizes (edge cases + typical)
  - Both tiny (fast) and full-size inputs
"""

import sys
import os
import numpy as np
import pytest

# Ensure src/ is importable without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adversarial_robustness.models.dummy_model import DummyClassifier


# ---------------------------------------------------------------------------
# Random number generator (seeded for reproducibility)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def small_model() -> DummyClassifier:
    """Tiny 3×8×8 model — fast for unit tests."""
    return DummyClassifier(num_classes=5, input_shape=(3, 8, 8), seed=42)


@pytest.fixture(scope="session")
def standard_model() -> DummyClassifier:
    """Standard 3×32×32 model (CIFAR-like)."""
    return DummyClassifier(num_classes=10, input_shape=(3, 32, 32), seed=42)


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def small_batch(rng, small_model) -> tuple:
    """(x, y) batch for small_model — shape (8, 3, 8, 8)."""
    x = rng.random((8, *small_model.input_shape)).astype(np.float32)
    y = rng.integers(0, small_model.num_classes, size=8).astype(np.int64)
    return x, y


@pytest.fixture(scope="session")
def standard_batch(rng, standard_model) -> tuple:
    """(x, y) batch for standard_model — shape (16, 3, 32, 32)."""
    x = rng.random((16, *standard_model.input_shape)).astype(np.float32)
    y = rng.integers(0, standard_model.num_classes, size=16).astype(np.int64)
    return x, y


@pytest.fixture(scope="session")
def single_sample(rng, small_model) -> tuple:
    """(x, y) single sample — batch size 1."""
    x = rng.random((1, *small_model.input_shape)).astype(np.float32)
    y = rng.integers(0, small_model.num_classes, size=1).astype(np.int64)
    return x, y
