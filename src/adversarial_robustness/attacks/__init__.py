"""
Adversarial attack implementations.

Available attacks
-----------------
FGSM        — Fast Gradient Sign Method (single-step)
PGD         — Projected Gradient Descent (multi-step, strongest baseline)
PatchAttack — Adversarial patch (visible, physical-world attack)
"""

from adversarial_robustness.attacks.base_attack import AttackResult, BaseAttack
from adversarial_robustness.attacks.fgsm import FGSM
from adversarial_robustness.attacks.patch_attack import PatchAttack
from adversarial_robustness.attacks.pgd import PGD

__all__ = ["BaseAttack", "AttackResult", "FGSM", "PGD", "PatchAttack"]
