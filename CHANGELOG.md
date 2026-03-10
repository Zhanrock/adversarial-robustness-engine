# Changelog

All notable changes to this project are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2024-12-01

### Added
- **FGSM attack** — single-step L-inf adversarial perturbation
- **PGD attack** — multi-step projected gradient descent with optional random start
- **PatchAttack** — visible adversarial patch (targeted and untargeted modes)
- **GaussianDenoiser** — Gaussian blur preprocessing defense
- **MedianDenoiser** — median filter preprocessing defense
- **FeatureSqueezing** — colour bit-depth reduction defense
- **DefensePipeline** — composable defense chain
- **AdversarialTrainer** — PGD/FGSM adversarial training loop with epoch metrics
- **RobustnessBenchmarker** — automated multi-attack evaluation with JSON reports
- **Metric suite** — clean accuracy, ASR, robustness gap, L2/L-inf norms, certified robustness
- **DummyClassifier** — NumPy-backed model for CPU testing without PyTorch
- **PytorchModelWrapper** — wraps real `torch.nn.Module` for production use
- **YAML config system** — dot-notation access with inheritance support
- **Structured logging** — per-module logging with optional file output
- **CLI entry points** — `are-evaluate`, `are-benchmark`
- **GitHub Actions CI** — lint, unit tests (Python 3.9/3.10/3.11), integration tests, smoke test
- **81 unit + integration tests** — all passing on CPU without GPU

### Architecture
- Abstract `BaseModel`, `BaseAttack`, `BaseDefense` interfaces enable plug-in extensibility
- PyTorch-compatible API through `NumpyTensor` shim for zero-dependency testing
- YAML config inheritance allows experiment variants without full config duplication
