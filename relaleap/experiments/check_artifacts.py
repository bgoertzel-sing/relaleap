"""Inspect existing RelaLeap comparison artifacts without rerunning experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACTS = ("summary.json", "metrics.csv", "notes.md")


def check_comparison_artifacts(
    comparison_dir: Path,
    *,
    require_baseline_comparison: bool = False,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Check a comparison artifact tree and return a compact pass/fail report."""

    checks: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        checks.append(_artifact_check(comparison_dir / name, f"comparison.{name}"))

    summary = _read_json(comparison_dir / "summary.json", checks, "comparison.summary")
    runs = summary.get("runs") if isinstance(summary, dict) else []
    run_reports = []
    if isinstance(runs, list):
        for entry in runs:
            if not isinstance(entry, dict):
                continue
            run_reports.append(_run_artifact_report(comparison_dir, entry))
    checks.extend(
        check
        for report in run_reports
        for check in report["artifacts"]
    )

    baseline_path = comparison_dir / "baseline_comparison.json"
    baseline_check = _artifact_check(
        baseline_path,
        "comparison.baseline_comparison.json",
    )
    baseline = None
    if baseline_check["exists"]:
        checks.append(baseline_check)
        baseline = _read_json(
            baseline_path,
            checks,
            "comparison.baseline_comparison",
        )
    elif require_baseline_comparison:
        checks.append(baseline_check)

    verdict = summary.get("verdict") if isinstance(summary, dict) else {}
    phase0_passed = (
        verdict.get("invariants_passed") if isinstance(verdict, dict) else None
    )
    phase0_failed = (
        verdict.get("failed_invariants", []) if isinstance(verdict, dict) else []
    )
    acceptance = (
        verdict.get("hep_alpha_acceptance", {}) if isinstance(verdict, dict) else {}
    )
    accepted_alpha = (
        acceptance.get("accepted_alpha") if isinstance(acceptance, dict) else None
    )

    failures = _artifact_failures(checks)
    if isinstance(summary, dict):
        if summary.get("status") != "ok":
            failures.append(
                {
                    "field": "comparison.status",
                    "expected": "ok",
                    "actual": summary.get("status"),
                }
            )
        if verdict.get("status") != "pass":
            failures.append(
                {
                    "field": "comparison.verdict.status",
                    "expected": "pass",
                    "actual": verdict.get("status"),
                }
            )
        if phase0_passed is not True:
            failures.append(
                {
                    "field": "comparison.verdict.invariants_passed",
                    "expected": True,
                    "actual": phase0_passed,
                }
            )
    if isinstance(baseline, dict) and baseline.get("status") != "pass":
        failures.append(
            {
                "field": "baseline_comparison.status",
                "expected": "pass",
                "actual": baseline.get("status"),
            }
        )

    report = {
        "status": "pass" if not failures else "fail",
        "comparison_dir": str(comparison_dir),
        "artifacts": checks,
        "summary_status": summary.get("status") if isinstance(summary, dict) else None,
        "verdict_status": verdict.get("status") if isinstance(verdict, dict) else None,
        "phase0_invariants": {
            "passed": phase0_passed,
            "count": verdict.get("invariant_count") if isinstance(verdict, dict) else None,
            "failed_count": len(phase0_failed) if isinstance(phase0_failed, list) else None,
        },
        "hep_alpha_acceptance": {
            "status": acceptance.get("status") if isinstance(acceptance, dict) else None,
            "accepted_alpha": accepted_alpha,
        },
        "baseline_comparison": _baseline_summary(baseline),
        "runs": run_reports,
        "failures": failures,
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def _run_artifact_report(
    comparison_dir: Path,
    entry: dict[str, Any],
) -> dict[str, Any]:
    config_path = entry.get("config_path", "")
    run_dir = comparison_dir / "runs" / Path(str(config_path)).stem
    return {
        "experiment_id": entry.get("experiment_id"),
        "run_dir": str(run_dir),
        "artifacts": [
            _artifact_check(run_dir / name, f"run.{entry.get('experiment_id')}.{name}")
            for name in REQUIRED_ARTIFACTS
        ],
    }


def _artifact_check(path: Path, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path),
        "exists": path.is_file(),
    }


def _artifact_failures(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for check in checks:
        if not check["exists"]:
            failures.append(
                {
                    "field": check["label"],
                    "expected": "file exists",
                    "actual": "missing",
                    "path": check["path"],
                }
            )
        if check.get("valid_json") is False:
            failures.append(
                {
                    "field": check["label"],
                    "expected": "valid JSON object",
                    "actual": check.get("error", "invalid JSON"),
                    "path": check["path"],
                }
            )
    return failures


def _read_json(
    path: Path,
    checks: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        checks.append(
            {
                "label": label,
                "path": str(path),
                "exists": True,
                "valid_json": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return {}
    if not isinstance(loaded, dict):
        checks.append(
            {
                "label": label,
                "path": str(path),
                "exists": True,
                "valid_json": False,
                "error": "expected JSON object",
            }
        )
        return {}
    return loaded


def _baseline_summary(baseline: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(baseline, dict):
        return {
            "present": False,
            "status": None,
            "mismatch_count": None,
        }
    mismatches = baseline.get("mismatches", [])
    return {
        "present": True,
        "status": baseline.get("status"),
        "mismatch_count": len(mismatches) if isinstance(mismatches, list) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect an existing RelaLeap comparison artifact directory."
    )
    parser.add_argument(
        "--comparison-dir",
        default=Path("results/comparisons/colab_phase0"),
        type=Path,
        help="Comparison output directory to inspect without rerunning experiments.",
    )
    parser.add_argument(
        "--require-baseline-comparison",
        action="store_true",
        help="Fail if baseline_comparison.json is missing.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional path to write the artifact check JSON report.",
    )
    args = parser.parse_args()
    report = check_comparison_artifacts(
        args.comparison_dir,
        require_baseline_comparison=args.require_baseline_comparison,
        out_path=args.out,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
