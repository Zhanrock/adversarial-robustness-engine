"""
evaluation/benchmarker.py
--------------------------
Robustness Benchmarking Suite
------------------------------
Orchestrates a full evaluation run:
  1. Measures clean accuracy baseline
  2. Runs every configured attack
  3. Measures accuracy under each attack, with and without defenses
  4. Computes the "robustness gap" (clean_acc - adversarial_acc)
  5. Persists results as a structured JSON report

This is the primary entry point for automated model validation in CI/CD
pipelines and regression testing against "Safety Gold Standards".
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from adversarial_robustness.attacks.base_attack import AttackResult, BaseAttack
from adversarial_robustness.defenses.base_defense import BaseDefense
from adversarial_robustness.models.base_model import BaseModel
from adversarial_robustness.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Report data classes
# ---------------------------------------------------------------------------


@dataclass
class AttackEvalResult:
    attack_name: str
    attack_config: Dict
    clean_accuracy: float
    adversarial_accuracy: float
    defended_accuracy: float  # accuracy after defense applied
    attack_success_rate: float
    robustness_gap: float  # clean_acc - adversarial_acc
    defense_recovery: float  # defended_acc - adversarial_acc
    mean_l2_perturbation: float
    mean_linf_perturbation: float
    eval_time_sec: float
    num_samples: int


@dataclass
class BenchmarkReport:
    model_name: str
    timestamp: str
    dataset_info: Dict
    clean_accuracy: float
    results: List[AttackEvalResult] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def worst_case_robustness(self) -> float:
        if not self.results:
            return self.clean_accuracy
        return min(r.adversarial_accuracy for r in self.results)

    def best_defended_accuracy(self) -> float:
        if not self.results:
            return self.clean_accuracy
        return max(r.defended_accuracy for r in self.results)

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, output_dir: str, filename: Optional[str] = None) -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{self.model_name}_{ts}.json"
        path = Path(output_dir) / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Benchmark report saved: %s", path)
        return str(path)

    def print_summary(self) -> None:
        sep = "=" * 72
        print(f"\n{sep}")
        print(f"  ROBUSTNESS BENCHMARK — {self.model_name}")
        print(f"  {self.timestamp}")
        print(sep)
        print(f"  Clean Accuracy  : {self.clean_accuracy:.4f}  ({self.clean_accuracy*100:.2f}%)")
        print(
            f"  Worst-Case Adv  : {self.worst_case_robustness():.4f}  "
            f"({self.worst_case_robustness()*100:.2f}%)"
        )
        print(
            f"  Best Defended   : {self.best_defended_accuracy():.4f}  "
            f"({self.best_defended_accuracy()*100:.2f}%)"
        )
        print(
            f"\n  {'Attack':<18} {'Adv Acc':>8} {'Defended':>8} "
            f"{'ASR':>8} {'Gap':>8} {'L2':>8} {'Linf':>8}"
        )
        print(f"  {'-'*70}")
        for r in self.results:
            print(
                f"  {r.attack_name:<18} "
                f"{r.adversarial_accuracy:>7.3f}  "
                f"{r.defended_accuracy:>7.3f}  "
                f"{r.attack_success_rate:>7.3f}  "
                f"{r.robustness_gap:>7.3f}  "
                f"{r.mean_l2_perturbation:>7.4f}  "
                f"{r.mean_linf_perturbation:>7.4f}"
            )
        print(sep)


# ---------------------------------------------------------------------------
# Benchmarker
# ---------------------------------------------------------------------------


class RobustnessBenchmarker:
    """
    Orchestrates a full adversarial robustness evaluation.

    Parameters
    ----------
    model:      Model under evaluation.
    defenses:   Optional list of preprocessor defenses.
    output_dir: Directory for JSON report persistence.
    model_name: Human-readable model identifier.

    Example
    -------
    >>> benchmarker = RobustnessBenchmarker(model, defenses=[GaussianDenoiser()])
    >>> report = benchmarker.run(x_test, y_test, attacks=[fgsm, pgd])
    >>> report.print_summary()
    """

    def __init__(
        self,
        model: BaseModel,
        defenses: Optional[List[BaseDefense]] = None,
        output_dir: str = "reports",
        model_name: str = "model",
    ) -> None:
        self.model = model
        self.defenses = defenses or []
        self.output_dir = output_dir
        self.model_name = model_name

    def run(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
        attacks: List[BaseAttack],
        save_report: bool = True,
    ) -> BenchmarkReport:
        """
        Run the full benchmark suite.

        Parameters
        ----------
        x_test:  Test inputs (N, C, H, W).
        y_test:  True labels (N,).
        attacks: List of instantiated attack objects to evaluate against.
        save_report: Whether to persist the report to disk.

        Returns
        -------
        BenchmarkReport with all metrics.
        """
        logger.info(
            "Starting benchmark: model=%s, n_samples=%d, n_attacks=%d",
            self.model_name,
            len(x_test),
            len(attacks),
        )

        # ── 1. Clean accuracy ────────────────────────────────────────────
        clean_acc = self.model.accuracy(x_test, y_test)
        logger.info("Clean accuracy: %.4f", clean_acc)

        report = BenchmarkReport(
            model_name=self.model_name,
            timestamp=datetime.now().isoformat(),
            dataset_info={
                "num_samples": len(x_test),
                "input_shape": list(x_test.shape[1:]),
                "num_classes": int(y_test.max() + 1),
            },
            clean_accuracy=clean_acc,
            metadata={
                "defenses": [repr(d) for d in self.defenses],
            },
        )

        # ── 2. Per-attack evaluation ─────────────────────────────────────
        for attack in attacks:
            attack_name = attack.__class__.__name__
            logger.info("Evaluating attack: %s", attack_name)
            t0 = time.monotonic()

            # Generate adversarial examples
            attack_result: AttackResult = attack.generate(x_test, y_test)
            x_adv = attack_result.adversarial_examples

            # Adversarial accuracy (no defense)
            adv_acc = self.model.accuracy(x_adv, y_test)

            # Defended accuracy
            if self.defenses:
                x_defended = x_adv
                for defense in self.defenses:
                    x_defended = defense.preprocess(x_defended)
                defended_acc = self.model.accuracy(x_defended, y_test)
            else:
                defended_acc = adv_acc

            elapsed = time.monotonic() - t0

            result = AttackEvalResult(
                attack_name=attack_name,
                attack_config={"epsilon": attack.epsilon},
                clean_accuracy=clean_acc,
                adversarial_accuracy=adv_acc,
                defended_accuracy=defended_acc,
                attack_success_rate=attack_result.attack_success_rate,
                robustness_gap=clean_acc - adv_acc,
                defense_recovery=defended_acc - adv_acc,
                mean_l2_perturbation=attack_result.mean_l2_norm,
                mean_linf_perturbation=attack_result.mean_linf_norm,
                eval_time_sec=round(elapsed, 2),
                num_samples=len(x_test),
            )
            report.results.append(result)

            logger.info(
                "  %s → adv_acc=%.4f, defended=%.4f, ASR=%.4f, gap=%.4f",
                attack_name,
                adv_acc,
                defended_acc,
                attack_result.attack_success_rate,
                clean_acc - adv_acc,
            )

        if save_report:
            report.save(self.output_dir)

        report.print_summary()
        return report
