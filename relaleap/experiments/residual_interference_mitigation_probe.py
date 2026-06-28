"""Focused mitigation report for mechanism-factorized residual interference."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_SUMMARY = Path(
    "results/reports/mechanism_factorized_continual_learning_probe/summary.json"
)
DEFAULT_OUT_DIR = Path("results/reports/residual_interference_mitigation_probe")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "mitigation_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)

REQUIRED_SOURCE_ARMS = (
    "dense_active_rank",
    "contextual_topk1",
    "contextual_topk2",
    "random_frequency_matched_topk2",
)


def run_residual_interference_mitigation_probe(
    *,
    source_summary_path: Path = DEFAULT_SOURCE_SUMMARY,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Analyze support-width widening as a sparse interference mitigation."""

    start = time.time()
    source_summary = _read_json(source_summary_path)
    source_arm_rows = list(source_summary.get("arm_metrics", []))
    arm_rows = [
        row for row in source_arm_rows if row.get("arm") in set(REQUIRED_SOURCE_ARMS)
    ]
    gate_rows = _preflight_gates(source_summary_path, source_summary, arm_rows)
    mitigation_rows: list[dict[str, Any]] = []
    if all(row["passed"] for row in gate_rows):
        mitigation_rows = _mitigation_rows(arm_rows)
        gate_rows.extend(_claim_gates(mitigation_rows))

    status = "pass" if all(row["passed"] for row in gate_rows if row["severity"] == "hard") else "fail"
    claim_status = _claim_status(gate_rows)
    summary = {
        "status": status,
        "decision": (
            "residual_interference_mitigation_probe_recorded"
            if status == "pass"
            else "residual_interference_mitigation_probe_failed_closed"
        ),
        "claim_status": claim_status,
        "selected_next_step": _selected_next_step(claim_status),
        "requires_gpu_now": False,
        "backend_policy": "local CPU post-hoc mitigation decision; no RunPod/Colab spend",
        "source_summary_path": str(source_summary_path),
        "source_decision": source_summary.get("decision"),
        "source_claim_status": source_summary.get("claim_status"),
        "mitigation_under_test": "contextual_topk2_support_width_expansion",
        "required_source_arms": list(REQUIRED_SOURCE_ARMS),
        "mitigation_metrics": mitigation_rows,
        "gate_criteria": gate_rows,
        "primary_result": _primary_result(mitigation_rows),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _preflight_gates(
    source_summary_path: Path,
    source_summary: dict[str, Any],
    arm_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_arm = {str(row.get("arm")): row for row in arm_rows}
    missing = sorted(set(REQUIRED_SOURCE_ARMS) - set(by_arm))
    return [
        _criterion(
            "source_summary_exists",
            source_summary_path.is_file(),
            "hard",
            "mechanism-factorized CL summary exists",
            str(source_summary_path),
            "missing source summary",
        ),
        _criterion(
            "source_probe_passed",
            source_summary.get("status") == "pass",
            "hard",
            "source mechanism-factorized probe passed hard artifact gates",
            source_summary.get("status"),
            "source probe did not pass",
        ),
        _criterion(
            "required_dense_sparse_null_arms_present",
            not missing,
            "hard",
            "dense, top-k1, top-k2, and random-support controls present",
            "present" if not missing else missing,
            "missing required source arms",
        ),
    ]


def _mitigation_rows(arm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_arm = {str(row["arm"]): row for row in arm_rows}
    dense = by_arm["dense_active_rank"]
    topk1 = by_arm["contextual_topk1"]
    topk2 = by_arm["contextual_topk2"]
    random_null = by_arm["random_frequency_matched_topk2"]
    comparisons = [
        ("topk2_minus_topk1", topk2, topk1),
        ("topk2_minus_dense", topk2, dense),
        ("topk2_minus_random_support_null", topk2, random_null),
    ]
    rows: list[dict[str, Any]] = []
    for comparison, left, right in comparisons:
        rows.append(
            {
                "comparison": comparison,
                "left_arm": left["arm"],
                "right_arm": right["arm"],
                "target_ce_delta_delta": _delta(left, right, "mean_target_ce_delta"),
                "off_target_ce_drift_delta": _delta(left, right, "mean_off_target_ce_drift"),
                "off_target_kl_delta": _delta(left, right, "mean_off_target_kl"),
                "final_forgetting_delta": _delta(left, right, "mean_final_forgetting"),
                "left_target_ce_delta": left.get("mean_target_ce_delta"),
                "right_target_ce_delta": right.get("mean_target_ce_delta"),
                "left_off_target_ce_drift": left.get("mean_off_target_ce_drift"),
                "right_off_target_ce_drift": right.get("mean_off_target_ce_drift"),
                "left_off_target_kl": left.get("mean_off_target_kl"),
                "right_off_target_kl": right.get("mean_off_target_kl"),
                "left_final_forgetting": left.get("mean_final_forgetting"),
                "right_final_forgetting": right.get("mean_final_forgetting"),
            }
        )
    return rows


def _claim_gates(mitigation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_comparison = {str(row["comparison"]): row for row in mitigation_rows}
    topk1 = by_comparison.get("topk2_minus_topk1", {})
    dense = by_comparison.get("topk2_minus_dense", {})
    random_null = by_comparison.get("topk2_minus_random_support_null", {})
    return [
        _criterion(
            "topk2_improves_sparse_target_adaptation_vs_topk1",
            _lt(topk1.get("target_ce_delta_delta"), 0.0),
            "claim",
            "support-width mitigation improves sparse target CE adaptation versus top-k1",
            topk1.get("target_ce_delta_delta"),
            "top-k2 did not improve target adaptation versus top-k1",
        ),
        _criterion(
            "topk2_preserves_dense_off_target_ce_advantage",
            _leq(dense.get("off_target_ce_drift_delta"), 0.0),
            "claim",
            "top-k2 off-target CE drift remains no worse than dense",
            dense.get("off_target_ce_drift_delta"),
            "top-k2 loses the dense off-target CE advantage",
        ),
        _criterion(
            "topk2_preserves_dense_off_target_kl_advantage",
            _leq(dense.get("off_target_kl_delta"), 0.0),
            "claim",
            "top-k2 off-target KL remains no worse than dense",
            dense.get("off_target_kl_delta"),
            "top-k2 loses the dense off-target KL advantage",
        ),
        _criterion(
            "topk2_preserves_forgetting_advantage_vs_topk1",
            _leq(topk1.get("final_forgetting_delta"), 0.0),
            "claim",
            "top-k2 final forgetting remains no worse than top-k1",
            topk1.get("final_forgetting_delta"),
            "top-k2 forgetting exceeds top-k1",
        ),
        _criterion(
            "topk2_beats_random_support_null_on_target_and_forgetting",
            _lt(random_null.get("target_ce_delta_delta"), 0.0)
            and _leq(random_null.get("final_forgetting_delta"), 0.0),
            "claim",
            "top-k2 improves target adaptation and forgetting versus random support null",
            {
                "target_ce_delta_delta": random_null.get("target_ce_delta_delta"),
                "final_forgetting_delta": random_null.get("final_forgetting_delta"),
            },
            "top-k2 did not beat random support null on both target adaptation and forgetting",
        ),
        _criterion(
            "topk2_target_adaptation_still_not_dense_matched",
            not _leq(dense.get("target_ce_delta_delta"), 0.02),
            "interpretation",
            "record whether support-width mitigation still trails dense target adaptation",
            dense.get("target_ce_delta_delta"),
            "top-k2 is target-adaptation matched to dense; repeat before any escalation",
        ),
    ]


def _criterion(
    criterion: str,
    passed: bool,
    severity: str,
    requirement: str,
    observed: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "severity": severity,
        "requirement": requirement,
        "observed": observed,
        "failure_reason": "" if passed else failure_reason,
    }


def _claim_status(gate_rows: list[dict[str, Any]]) -> str:
    if any(not row["passed"] and row["severity"] == "hard" for row in gate_rows):
        return "residual_interference_mitigation_failed_closed"
    claim_fail = any(not row["passed"] and row["severity"] == "claim" for row in gate_rows)
    if claim_fail:
        return "support_width_mitigation_not_sufficient"
    dense_gap_recorded = any(
        row["criterion"] == "topk2_target_adaptation_still_not_dense_matched"
        and row["passed"]
        for row in gate_rows
    )
    if dense_gap_recorded:
        return "support_width_mitigation_partial_candidate_not_promoted"
    return "support_width_mitigation_candidate_requires_repeat"


def _selected_next_step(claim_status: str) -> str:
    if claim_status == "residual_interference_mitigation_failed_closed":
        return "repair source mechanism CL artifact before mitigation interpretation"
    if claim_status == "support_width_mitigation_not_sufficient":
        return "test a stricter sparse target-adaptation mitigation with the same dense/null controls"
    if claim_status == "support_width_mitigation_partial_candidate_not_promoted":
        return "design one sparse target-adaptation rescue that closes the remaining dense gap without losing top-k2 off-target advantages"
    return "repeat support-width mitigation on a second local seed before GPU validation"


def _primary_result(mitigation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_comparison = {str(row["comparison"]): row for row in mitigation_rows}
    topk1 = by_comparison.get("topk2_minus_topk1", {})
    dense = by_comparison.get("topk2_minus_dense", {})
    random_null = by_comparison.get("topk2_minus_random_support_null", {})
    return {
        "topk2_minus_topk1_target_ce_delta": topk1.get("target_ce_delta_delta"),
        "topk2_minus_topk1_off_target_ce_drift": topk1.get("off_target_ce_drift_delta"),
        "topk2_minus_dense_off_target_ce_drift": dense.get("off_target_ce_drift_delta"),
        "topk2_minus_dense_off_target_kl": dense.get("off_target_kl_delta"),
        "topk2_minus_random_support_null_target_ce_delta": random_null.get("target_ce_delta_delta"),
        "interpretation": (
            "Negative deltas favor contextual top-k2. This is a local support-width "
            "mitigation screen, not promotion evidence without repeats."
        ),
    }


def _delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    left_value = left.get(key)
    right_value = right.get(key)
    if left_value is None or right_value is None:
        return None
    return float(left_value) - float(right_value)


def _lt(value: Any, threshold: float) -> bool:
    if value is None:
        return False
    return float(value) < threshold


def _leq(value: Any, threshold: float) -> bool:
    if value is None:
        return False
    return float(value) <= threshold


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "mitigation_metrics.csv", summary["mitigation_metrics"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    result = summary["primary_result"]
    lines = [
        "# Residual-Interference Mitigation Probe",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Mitigation under test: `{summary['mitigation_under_test']}`",
        f"- Source summary: `{summary['source_summary_path']}`",
        "- Top-k2 minus top-k1 target CE delta: "
        f"`{result.get('topk2_minus_topk1_target_ce_delta')}`",
        "- Top-k2 minus dense off-target CE drift: "
        f"`{result.get('topk2_minus_dense_off_target_ce_drift')}`",
        "- Top-k2 minus dense off-target KL: "
        f"`{result.get('topk2_minus_dense_off_target_kl')}`",
        "- Top-k2 minus random-support-null target CE delta: "
        f"`{result.get('topk2_minus_random_support_null_target_ce_delta')}`",
        "",
        "This report treats support-width expansion as a residual-interference mitigation screen. It consumes the command-generated known-rule mechanism CL artifact and keeps dense and random-support controls in the gate.",
        "",
        "## Next Step",
        "",
        str(summary["selected_next_step"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", type=Path, default=DEFAULT_SOURCE_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_residual_interference_mitigation_probe(
        source_summary_path=args.source_summary,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
