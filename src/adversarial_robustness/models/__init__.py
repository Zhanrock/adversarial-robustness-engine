"""Model wrappers and abstract base class."""

from adversarial_robustness.models.base_model import BaseModel
from adversarial_robustness.models.dummy_model import DummyClassifier, PytorchModelWrapper

__all__ = ["BaseModel", "DummyClassifier", "PytorchModelWrapper"]
