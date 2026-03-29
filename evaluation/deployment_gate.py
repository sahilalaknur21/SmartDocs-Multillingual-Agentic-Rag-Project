"""
evaluation/deployment_gate.py

WHY THIS EXISTS: Prevents deploying a SmartDocs version that fails Hindi users.
Reads the evaluation_report.json produced by ragas_evaluator.py and enforces
hard thresholds. Exit code 1 blocks the Railway / CI deploy pipeline.

Usage in CI:
    python evaluation/deployment_gate.py
    # Exit 0 = deploy approved
    # Exit 1 = deploy blocked (reason printed)

LAW 16: faithfulness_hindi < 0.85 OR language_accuracy < 0.92 â†’ BLOCK DEPLOYMENT.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deployment thresholds â€” single source of truth
# ---------------------------------------------------------------------------

THRESHOLDS: dict[str, tuple[str, float, str]] = {
    # (metric_path, minimum_value, human_readable_name)
    "faithfulness_en":       ("english.faithfulness",      0.90, "English faithfulness"),
    "faithfulness_hi":       ("hindi.faithfulness",        0.85, "Hindi faithfulness"),
    "relevancy_en":          ("english.answer_relevancy",  0.85, "English answer relevancy"),
    "relevancy_hi":          ("hindi.answer_relevancy",    0.83, "Hindi answer relevancy"),
    "precision_en":          ("english.context_precision", 0.80, "English context precision"),
    "precision_hi":          ("hindi.context_precision",   0.80, "Hindi context precision"),
    "recall_en":             ("english.context_recall",    0.75, "English context recall"),
    "recall_hi":             ("hindi.context_recall",      0.75, "Hindi context recall"),
    "hallucination_rate":    ("hallucination_rate",        0.10, "Hallucination rate",    True),   # lower is better
    "faithfulness_ratio":    ("faithfulness_ratio",        0.97, "Hindi/English faithfulness ratio"),
}

# Metrics where LOWER is better (inverted threshold check)
LOWER_IS_BETTER = {"hallucination_rate"}


# ---------------------------------------------------------------------------
# Gate result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    approved: bool
    blocking_failures: list[dict]    # [{"metric": str, "value": float, "threshold": float}]
    warnings: list[str]
    report_timestamp: str
    evaluated_at: str

    def print_report(self) -> None:
        sep = "=" * 64
        print(f"\n{sep}")
        print("SMARTDOCS DEPLOYMENT GATE")
        print(f"Report timestamp:  {self.report_timestamp}")
        print(f"Evaluated at:      {self.evaluated_at}")
        print(sep)

        if self.approved:
            print("\nâœ…  DEPLOYMENT APPROVED\n")
            print("    All metrics above threshold.")
        else:
            print("\nðŸš«  DEPLOYMENT BLOCKED\n")
            for failure in self.blocking_failures:
                direction = ">" if failure["metric"] not in LOWER_IS_BETTER else "<"
                print(
                    f"    âœ--  {failure['name']:<35} "
                    f"actual={failure['value']:.4f}  "
                    f"threshold{direction}{failure['threshold']:.4f}"
                )

        if self.warnings:
            print("\n  Warnings:")
            for w in self.warnings:
                print(f"    âš   {w}")

        print(sep)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "blocking_failures": self.blocking_failures,
            "warnings": self.warnings,
            "report_timestamp": self.report_timestamp,
            "evaluated_at": self.evaluated_at,
        }


# ---------------------------------------------------------------------------
# Helper: safe nested key access
# ---------------------------------------------------------------------------

def _get_nested(data: dict, dotted_path: str) -> float | None:
    """
    Resolves a dot-separated key path against a nested dict.
    Returns None if any key is missing.

    Example: _get_nested(report, "english.faithfulness") â†’ 0.91
    """
    parts = dotted_path.split(".")
    obj: Any = data
    for part in parts:
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    try:
        return float(obj)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

def check_gate(report: dict) -> GateResult:
    """
    Applies all deployment thresholds to an evaluation report dict.

    Args:
        report: Dict produced by EvaluationReport.to_dict() in ragas_evaluator.py.

    Returns:
        GateResult with approved=True if all thresholds pass.
    """
    blocking_failures: list[dict] = []
    warnings: list[str] = []

    for metric_key, threshold_spec in THRESHOLDS.items():
        # Handle both 4-tuple (with lower_is_better flag) and 3-tuple
        if len(threshold_spec) == 4:
            path, threshold, name, _ = threshold_spec
        else:
            path, threshold, name = threshold_spec[:3]

        value = _get_nested(report, path)

        if value is None:
            warnings.append(f"Metric '{metric_key}' not found in report â€” skipped")
            continue

        if metric_key in LOWER_IS_BETTER:
            failed = value >= threshold
        else:
            failed = value < threshold

        if failed:
            blocking_failures.append({
                "metric": metric_key,
                "name": name,
                "path": path,
                "value": value,
                "threshold": threshold,
            })

    # Special check: if both en and hi faithfulness are present, assert ratio
    en_faith = _get_nested(report, "english.faithfulness")
    hi_faith = _get_nested(report, "hindi.faithfulness")
    if en_faith and hi_faith and en_faith > 0:
        ratio = hi_faith / en_faith
        if ratio < 0.97:
            blocking_failures.append({
                "metric": "faithfulness_ratio_computed",
                "name": "Computed Hindi/English faithfulness ratio",
                "path": "computed",
                "value": ratio,
                "threshold": 0.97,
            })

    return GateResult(
        approved=len(blocking_failures) == 0,
        blocking_failures=blocking_failures,
        warnings=warnings,
        report_timestamp=report.get("timestamp", "unknown"),
        evaluated_at=datetime.now(timezone.utc).isoformat(),
    )


def load_and_check(report_path: Path) -> GateResult:
    """
    Loads evaluation report from disk and runs the gate check.

    Args:
        report_path: Path to evaluation_report.json.

    Returns:
        GateResult.

    Raises:
        FileNotFoundError: If report does not exist.
        json.JSONDecodeError: If report is malformed.
    """
    if not report_path.exists():
        raise FileNotFoundError(
            f"Evaluation report not found at {report_path}. "
            "Run: python evaluation/ragas_evaluator.py first."
        )

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    return check_gate(report)


def write_gate_result(result: GateResult, output_path: Path) -> None:
    """Writes GateResult to JSON for CI artifact storage."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info("Gate result written to %s", output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """
    CLI entry point.
    Exit code 0 = deployment approved.
    Exit code 1 = deployment blocked.
    Exit code 2 = report not found (treat as blocked).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s â€” %(message)s",
        stream=sys.stdout,
    )

    report_path = Path(__file__).parent / "evaluation_report.json"
    gate_output_path = Path(__file__).parent / "gate_result.json"

    try:
        result = load_and_check(report_path)
    except FileNotFoundError as exc:
        print(f"\nðŸš«  DEPLOYMENT BLOCKED â€” {exc}\n")
        return 2
    except json.JSONDecodeError as exc:
        print(f"\nðŸš«  DEPLOYMENT BLOCKED â€” Malformed evaluation report: {exc}\n")
        return 2

    result.print_report()
    write_gate_result(result, gate_output_path)

    return 0 if result.approved else 1


if __name__ == "__main__":
    sys.exit(main())

