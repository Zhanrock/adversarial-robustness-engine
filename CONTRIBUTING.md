# Contributing

Thank you for contributing to the Adversarial Robustness Engine.

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/adversarial-robustness-engine.git
cd adversarial-robustness-engine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# Install in editable mode with all dev dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Unit tests only (fast, no GPU)
python -m pytest tests/unit/ -v -m "not gpu"

# Integration tests
python -m pytest tests/integration/ -v

# With coverage
python -m pytest tests/ --cov=adversarial_robustness --cov-report=html
```

## Code Style

This project uses Black, isort, and Flake8:

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/adversarial_robustness
```

## Adding a New Attack

1. Create `src/adversarial_robustness/attacks/my_attack.py`
2. Subclass `BaseAttack` and implement `generate()`
3. Export from `attacks/__init__.py`
4. Add unit tests in `tests/unit/test_attacks.py`
5. Add a benchmark integration test

## Adding a New Defense

1. Create `src/adversarial_robustness/defenses/my_defense.py`
2. Subclass `BaseDefense` and implement `preprocess()`
3. Export from `defenses/__init__.py`
4. Add unit tests
5. Verify `DefensePipeline` compatibility

## Pull Request Process

1. Branch from `develop`: `git checkout -b feat/my-feature`
2. Write tests first (TDD encouraged)
3. Ensure all CI checks pass
4. Submit PR with clear description of what changed and why
