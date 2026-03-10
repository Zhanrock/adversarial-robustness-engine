# 🛡️ Adversarial Robustness Engine

> **Protect safety-critical computer vision models from adversarial attacks in real-time video analytics pipelines.**

[![CI](https://github.com/Zhanrock/adversarial-robustness-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Zhanrock/adversarial-robustness-engine/actions)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-81%20passing-brightgreen)]()

---

## Overview

The **Adversarial Robustness Engine** is a defense framework that evaluates and improves the resilience of computer vision models against adversarial attacks — including "cloaking patches" and pixel-level manipulation that could cause safety-critical AI systems to miss hazards.

Built with a production-first architecture: modular, extensible, fully tested, and designed to integrate into existing ML pipelines with minimal friction.

---

## The Problem

In safety-critical environments (industrial monitoring, autonomous systems, security surveillance), adversaries may intentionally manipulate video data to:

- **Hide objects** from detection systems using imperceptible pixel perturbations (FGSM/PGD)
- **Physically attach printed patches** to objects to cause persistent misclassification
- **Inject noise** into video streams to degrade model confidence below action thresholds

Standard models are not robust to these attacks. A model that achieves 95% clean accuracy may drop to near 0% under a PGD attack with ε=0.03.

---

## Solution: Defense-in-Depth

This framework implements a layered defense strategy:

```
Video Frame Input
      │
      ▼
┌─────────────────────┐
│   Preprocessing     │  ← Gaussian denoising, feature squeezing
│   Defense Layer     │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│   Robust Model      │  ← Adversarially trained (PGD/FGSM)
│   (Inference)       │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│   Robustness        │  ← Continuous benchmarking in CI/CD
│   Validation        │
└─────────────────────┘
```

---

## Features

| Category | Implementation |
|----------|---------------|
| **Attacks** | FGSM, PGD (multi-step), Adversarial Patch |
| **Defenses** | Gaussian Denoiser, Median Filter, Feature Squeezing |
| **Training** | Adversarial Training (FGSM/PGD), epoch metrics & history |
| **Evaluation** | Clean acc, ASR, Robustness Gap, L2/L-inf norms, Certified Robustness |
| **Benchmarking** | Multi-attack automated benchmark, JSON report, tabular summary |
| **Architecture** | Abstract interfaces — swap any attack, defense, or model |
| **CI/CD** | GitHub Actions: lint, unit, integration, smoke tests |
| **No-GPU mode** | All code runs on CPU via NumPy backend for testing |

---

## Quick Start

```bash
# Clone & install
git clone https://github.com/YOUR_USERNAME/adversarial-robustness-engine.git
cd adversarial-robustness-engine
pip install numpy scipy Pillow pyyaml

# Run full benchmark (synthetic data, no GPU needed)
python scripts/run_benchmark.py

# Or step by step in Python:
python - <<'EOF'
import numpy as np
import sys; sys.path.insert(0, "src")

from adversarial_robustness.models.dummy_model import DummyClassifier
from adversarial_robustness.attacks.fgsm import FGSM
from adversarial_robustness.attacks.pgd import PGD
from adversarial_robustness.defenses.denoiser import GaussianDenoiser
from adversarial_robustness.evaluation.benchmarker import RobustnessBenchmarker

# Create model and test data
model = DummyClassifier(num_classes=10, input_shape=(3, 32, 32))
rng   = np.random.default_rng(42)
x     = rng.random((100, 3, 32, 32)).astype(np.float32)
y     = rng.integers(0, 10, 100).astype(np.int64)

# Run benchmark: FGSM + PGD with Gaussian denoising defense
benchmarker = RobustnessBenchmarker(
    model,
    defenses=[GaussianDenoiser(sigma=0.05)],
    output_dir="reports/",
)
report = benchmarker.run(x, y, attacks=[
    FGSM(model, epsilon=0.03),
    PGD(model, epsilon=0.03, num_steps=40),
])
EOF
```

**Output:**
```
========================================================================
  ROBUSTNESS BENCHMARK — DummyClassifier
  2024-12-01T10:00:00
========================================================================
  Clean Accuracy  : 0.0900  (9.00%)
  Worst-Case Adv  : 0.0000  (0.00%)
  Best Defended   : 0.0400  (4.00%)

  Attack              Adv Acc Defended      ASR      Gap       L2     Linf
  ----------------------------------------------------------------------
  FGSM                 0.000    0.010    1.000    0.090   1.642   0.0300
  PGD                  0.000    0.040    1.000    0.090   1.638   0.0300
========================================================================
```

---

## Project Structure

```
adversarial-robustness-engine/
├── src/
│   └── adversarial_robustness/
│       ├── attacks/
│       │   ├── base_attack.py       # Abstract BaseAttack + AttackResult
│       │   ├── fgsm.py              # Fast Gradient Sign Method
│       │   ├── pgd.py               # Projected Gradient Descent
│       │   └── patch_attack.py      # Adversarial patch (physical attack)
│       ├── defenses/
│       │   ├── base_defense.py      # Abstract BaseDefense + DefensePipeline
│       │   ├── denoiser.py          # Gaussian, Median, FeatureSqueezing
│       │   └── adversarial_training.py  # Training loop + TrainingHistory
│       ├── evaluation/
│       │   ├── benchmarker.py       # RobustnessBenchmarker + BenchmarkReport
│       │   └── metrics.py           # Pure metric functions
│       ├── models/
│       │   ├── base_model.py        # Abstract BaseModel interface
│       │   └── dummy_model.py       # NumPy DummyClassifier + PytorchModelWrapper
│       ├── utils/
│       │   ├── config.py            # YAML loader with dot-access + inheritance
│       │   ├── logger.py            # Centralised structured logging
│       │   └── tensor_ops.py        # NumPy shim mirroring PyTorch tensor API
│       └── cli.py                   # are-evaluate CLI entry point
├── tests/
│   ├── conftest.py                  # Shared pytest fixtures
│   ├── unit/
│   │   ├── test_models.py           # 15 tests
│   │   ├── test_attacks.py          # 22 tests
│   │   ├── test_defenses.py         # 24 tests
│   │   ├── test_metrics.py          # 13 tests
│   │   └── test_config.py           # 6 tests
│   └── integration/
│       └── test_full_pipeline.py    # 8 integration tests
├── configs/
│   ├── default.yaml                 # Full default configuration
│   └── experiment_fgsm.yaml         # FGSM experiment (inherits default)
├── scripts/
│   └── run_benchmark.py             # End-to-end benchmark runner
├── .github/
│   └── workflows/
│       └── ci.yml                   # GitHub Actions CI pipeline
├── pyproject.toml
├── CHANGELOG.md
├── CONTRIBUTING.md
└── README.md
```

---

## Installation

### CPU-only (no GPU, no PyTorch required)
```bash
pip install numpy scipy Pillow pyyaml
```

### Full installation with PyTorch
```bash
pip install -e ".[dev]"
# or for production:
pip install torch torchvision  # PyTorch
pip install -e .
```

---

## Attacks

### FGSM — Fast Gradient Sign Method
Single-step attack. Fast and widely used as a training augmentation.
```python
from adversarial_robustness.attacks.fgsm import FGSM

attack = FGSM(model, epsilon=0.03)
result = attack.generate(x_batch, labels)
# result.adversarial_examples, result.attack_success_rate, result.mean_l2_norm
```

### PGD — Projected Gradient Descent
Multi-step attack. The strongest white-box L-inf attack and gold standard for evaluation.
```python
from adversarial_robustness.attacks.pgd import PGD

attack = PGD(model, epsilon=0.03, alpha=0.007, num_steps=40, random_start=True)
result = attack.generate(x_batch, labels)
print(result)  # PGD complete — ASR: 94.5% | mean L2: 0.832 | mean Linf: 0.030
```

### PatchAttack — Adversarial Patch
Visible patch that causes misclassification regardless of where it's placed.
Models physical-world attacks (stickers, printed patches, "cloaking" wearables).
```python
from adversarial_robustness.attacks.patch_attack import PatchAttack

attack = PatchAttack(model, patch_size=32, num_steps=500, target_class=None)
result = attack.generate(x_batch, labels)
patch  = attack.optimised_patch   # (C, 32, 32) — save/display this
```

---

## Defenses

### Preprocessing Defenses
```python
from adversarial_robustness.defenses.denoiser import GaussianDenoiser, FeatureSqueezing
from adversarial_robustness.defenses.base_defense import DefensePipeline

# Chain defenses
pipeline = DefensePipeline([
    GaussianDenoiser(sigma=0.05),
    FeatureSqueezing(bit_depth=6),
])
x_defended = pipeline(x_adversarial)
```

### Adversarial Training
```python
from adversarial_robustness.defenses.adversarial_training import AdversarialTrainer

trainer = AdversarialTrainer(
    model,
    attack="pgd",
    epsilon=0.03,
    ratio=0.5,      # 50% adversarial examples per batch
    epochs=50,
)
history = trainer.fit(x_train, y_train, x_val, y_val)
print(history.summary())
```

---

## Using with Real PyTorch Models

```python
import torchvision.models as tv
from adversarial_robustness.models.dummy_model import PytorchModelWrapper
from adversarial_robustness.attacks.pgd import PGD

# Wrap any torchvision model
torch_model = tv.resnet18(pretrained=True)
model = PytorchModelWrapper(torch_model, num_classes=1000, device="cuda")

# All attacks and defenses work identically
attack = PGD(model, epsilon=0.03)
result = attack.generate(x_batch, labels)
```

---

## Configuration System

```python
from adversarial_robustness.utils.config import load_config

cfg = load_config("configs/default.yaml")

# Dot-notation access
print(cfg.attacks.pgd.epsilon)        # 0.03
print(cfg.training.epochs)            # 50
print(cfg.defenses.gaussian_denoiser.sigma)  # 0.05
```

Create experiment configs with inheritance:
```yaml
# configs/my_experiment.yaml
_base_: default.yaml         # inherit everything

attacks:
  pgd:
    epsilon: 0.05            # only override this
```

---

## Running Tests

```bash
# All 81 tests (requires: numpy, scipy)
python -m unittest discover -s tests -p "test_*.py" -v

# With pytest (if installed)
pytest tests/ -v
pytest tests/unit/ -v -m "not gpu"       # fast unit tests only
pytest tests/integration/ -v             # full pipeline tests
```

**Test coverage: 81 tests across 6 test modules, 0 failures.**

---

## CI/CD Integration

The benchmark script is designed to run in CI/CD as a safety validation gate:

```yaml
# In your CI pipeline (.github/workflows/model_validation.yml):
- name: Adversarial Robustness Validation
  run: |
    python scripts/run_benchmark.py \
      --epsilon 0.03 \
      --samples 500 \
      --output reports/

- name: Check robustness threshold
  run: |
    python -c "
    import json, glob, sys
    report = json.load(open(sorted(glob.glob('reports/*.json'))[-1]))
    worst = min(r['adversarial_accuracy'] for r in report['results'])
    if worst < 0.30:
        print(f'FAIL: Worst-case robustness {worst:.2%} below threshold 30%')
        sys.exit(1)
    print(f'PASS: Worst-case robustness {worst:.2%}')
    "
```

---

## References

- Goodfellow et al. (2015) — [Explaining and Harnessing Adversarial Examples (FGSM)](https://arxiv.org/abs/1412.6572)
- Madry et al. (2018) — [Towards Deep Learning Models Resistant to Adversarial Attacks (PGD)](https://arxiv.org/abs/1706.06083)
- Brown et al. (2018) — [Adversarial Patch](https://arxiv.org/abs/1712.09665)
- Xu et al. (2018) — [Feature Squeezing](https://arxiv.org/abs/1704.01155)
- Guo et al. (2018) — [Countering Adversarial Images Using Input Transformations](https://arxiv.org/abs/1711.00117)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
