"""Local failure localization for dense-teacher residual columnability.

This command consumes completed dense-teacher residual columnability assay
artifacts and separates upstream teacher adequacy, sparse value scale,
oracle-support representability, null sensitivity, and learned-router regret.
It is post-hoc local analysis only; it never promotes a GPU validation claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_DIRS = (
    Path("results/reports/dense_teacher_residual_columnability_assay"),
    Path("results/reports/dense_teacher_residual_columnability_assay_seed2"),
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_residual_columnability_failure_localization")

DECISION = "dense_teacher_residual_columnability_failure_localization_recorded"
FAIL_DECISION = "dense_teacher_residual_columnability_failure_localization_failed_closed"
REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_seeds.csv",
    "failure_axes.csv",
    "arm_comparison.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_dense_teacher_residual_columnability_failure_localization(
    *,
    source_dirs: tuple[Path, ...] = DEFAULT_SOURCE_DIRS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed seed-comparison localization report."""

    start = time.time()
    review = _strategy_review(strategy_review_path)
    seed_rows: list[dict[str, Any]] = []
    arm_rows: list[dict[str, Any]] = []
    for index, source_dir in enumerate(source_dirs, start=1):
        packet = _read_json(source_dir / "summary.json")
        seed_label = source_dir.name
        seed_rows.append(_seed_row(seed_label, source_dir, packet))
        arm_rows.extend(_arm_rows(seed_label, source_dir, packet))

    axes = _failure_axes(seed_rows, arm_rows)
    gates = _gate_rows(seed_rows, axes, review)
    hard_failures = [row for row in gates if row["required"] and not row["passed"]]
    scientific_failures = [row for row in gates if row["gate_type"] == "scientific" and not row["passed"]]
    status = "fail" if hard_failures else "pass"
    claim_status = (
        "dense_teacher_residual_columnability_failure_localized_gpu_blocked"
        if not hard_failures
        else "dense_teacher_residual_columnability_failure_localization_incomplete"
    )
    selected_next_step = (
        "add a bounded local sparse value-capacity and norm-control redesign pregate for the dense-teacher residual dictionary"
        if not hard_failures
        else "repair missing dense-teacher residual columnability assay artifacts before interpretation"
    )
    summary = {
        "status": status,
        "decision": FAIL_DECISION if hard_failures else DECISION,
        "claim_status": claim_status,
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "source_seed_count": len(seed_rows),
        "complete_source_seed_count": sum(1 for row in seed_rows if row["present"]),
        "source_seeds": seed_rows,
        "failure_axes": axes,
        "arm_comparison": arm_rows,
        "gate_criteria": gates,
        "failures": hard_failures + scientific_failures,
        "localization": _localization_summary(axes),
        "selected_next_step": selected_next_step,
        "strategy_review": review,
        "strategy_review_handling": _strategy_review_handling(review),
        "backend_policy": "local CPU artifact analysis only; RunPod and Colab remain blocked by failed/mixed local gates",
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _seed_row(seed_label: str, source_dir: Path, packet: dict[str, Any]) -> dict[str, Any]:
    arms = _arms(packet)
    oracle = arms.get("oracle_support_sparse_dictionary", {})
    learned = arms.get("learned_causal_router_sparse_dictionary", {})
    token = arms.get("token_position_router_null", {})
    shuffled = arms.get("shuffled_teacher_residual_null", {})
    base_ce = _float(packet.get("base_holdout_ce"))
    teacher_ce = _float(packet.get("dense_teacher_holdout_ce"))
    teacher_l2 = _float(oracle.get("teacher_residual_l2_mean"))
    oracle_l2 = _float(oracle.get("residual_l2_mean"))
    oracle_mse = _float(oracle.get("teacher_residual_reconstruction_mse"))
    shuffled_mse = _float(shuffled.get("teacher_residual_reconstruction_mse"))
    learned_ce = _float(learned.get("ce"))
    token_ce = _float(token.get("ce"))
    return {
        "seed_label": seed_label,
        "source_dir": str(source_dir),
        "present": bool(packet),
        "source_status": packet.get("status", ""),
        "source_decision": packet.get("decision", ""),
        "source_claim_status": packet.get("claim_status", ""),
        "base_ce": base_ce,
        "dense_teacher_ce": teacher_ce,
        "teacher_ce_improvement": _delta(base_ce, teacher_ce),
        "teacher_improves_base": _delta(base_ce, teacher_ce) is not None and _delta(base_ce, teacher_ce) > 0.0,
        "oracle_sparse_ce": _float(oracle.get("ce")),
        "learned_sparse_ce": learned_ce,
        "token_position_null_ce": token_ce,
        "learned_sparse_ce_advantage_vs_token_null": _delta(token_ce, learned_ce),
        "oracle_sparse_mse": oracle_mse,
        "shuffled_null_mse": shuffled_mse,
        "oracle_sparse_mse_advantage_vs_shuffled_null": _delta(shuffled_mse, oracle_mse),
        "oracle_residual_l2_mean": oracle_l2,
        "teacher_residual_l2_mean": teacher_l2,
        "oracle_to_teacher_l2_ratio": _ratio(oracle_l2, teacher_l2),
        "failed_source_scientific_gates": ",".join(
            row.get("criterion", "") for row in _list(packet.get("failures")) if row.get("gate_type") == "scientific"
        ),
    }


def _arm_rows(seed_label: str, source_dir: Path, packet: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for arm in _list(packet.get("arm_metrics")):
        rows.append(
            {
                "seed_label": seed_label,
                "source_dir": str(source_dir),
                "arm": arm.get("arm", ""),
                "ce": _float(arm.get("ce")),
                "ce_gap_vs_dense_teacher": _float(arm.get("ce_gap_vs_dense_teacher")),
                "teacher_residual_reconstruction_mse": _float(arm.get("teacher_residual_reconstruction_mse")),
                "oracle_support_regret": _float(arm.get("oracle_support_regret")),
                "residual_l2_mean": _float(arm.get("residual_l2_mean")),
                "teacher_residual_l2_mean": _float(arm.get("teacher_residual_l2_mean")),
                "residual_l2_ratio": _ratio(_float(arm.get("residual_l2_mean")), _float(arm.get("teacher_residual_l2_mean"))),
                "functional_churn": _float(arm.get("functional_churn")),
                "finite_update_commutator_proxy": _float(arm.get("finite_update_commutator_proxy")),
                "retention_proxy": _float(arm.get("retention_proxy")),
                "intervention_selectivity_proxy": _float(arm.get("intervention_selectivity_proxy")),
                "oracle_support_non_deployable": bool(arm.get("oracle_support_non_deployable")),
                "uses_oracle_support_at_eval": bool(arm.get("uses_oracle_support_at_eval")),
            }
        )
    return rows


def _failure_axes(seed_rows: list[dict[str, Any]], arm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    present = [row for row in seed_rows if row["present"]]
    teacher_fail_count = sum(1 for row in present if not row["teacher_improves_base"])
    oracle_edges = [_float(row["oracle_sparse_mse_advantage_vs_shuffled_null"]) for row in present]
    l2_ratios = [_float(row["oracle_to_teacher_l2_ratio"]) for row in present]
    learned_edges = [_float(row["learned_sparse_ce_advantage_vs_token_null"]) for row in present]
    return [
        _axis(
            "teacher_training_adequacy",
            teacher_fail_count == 0,
            "blocked_mixed" if teacher_fail_count else "supported",
            f"{teacher_fail_count}/{len(present)} source seeds fail dense_teacher_ce < base_ce",
            "teacher failure is upstream of sparse routing when present",
        ),
        _axis(
            "sparse_residual_scale",
            _all_at_least(l2_ratios, 0.5),
            "blocked",
            f"oracle/teacher residual L2 ratios={_format_values(l2_ratios)}; threshold >=0.5 on every seed",
            "oracle dictionary is under-scale relative to teacher corrections",
        ),
        _axis(
            "oracle_support_representability",
            _all_positive(oracle_edges, min_edge=0.02),
            "blocked_mixed",
            f"oracle MSE advantage over shuffled null={_format_values(oracle_edges)}; threshold >0.02 on every seed",
            "oracle-support sparse values do not robustly beat shuffled teacher-residual nulls",
        ),
        _axis(
            "learned_router_regret",
            _all_positive(learned_edges, min_edge=0.0),
            "secondary_mixed",
            f"learned sparse CE advantage over token-position null={_format_values(learned_edges)}",
            "learned routing is not the primary blocker unless oracle support is first representable",
        ),
        _axis(
            "null_sensitivity",
            not _any_small_or_negative(oracle_edges, threshold=0.02),
            "blocked",
            f"oracle-vs-shuffled edge small or negative on {_count_small_or_negative(oracle_edges, 0.02)}/{len(oracle_edges)} seeds",
            "teacher-residual nulls are too competitive for a columnability claim",
        ),
        _axis(
            "gpu_readiness",
            False,
            "blocked",
            "mixed local gates and sparse value-scale/representability failures keep requires_gpu_now=false",
            "GPU validation would amplify an unresolved local failure mode",
        ),
    ]


def _gate_rows(
    seed_rows: list[dict[str, Any]],
    axes: list[dict[str, Any]],
    review: dict[str, Any],
) -> list[dict[str, Any]]:
    present_count = sum(1 for row in seed_rows if row["present"])
    axes_by_name = {row["axis"]: row for row in axes}
    return [
        _gate("source_artifacts_present", present_count >= 2, True, "runtime", f"present_source_seeds={present_count}"),
        _gate("local_analysis_only", True, True, "runtime", "requires_gpu_now=false; RunPod/Colab unused"),
        _gate(
            "strategy_review_major_notify_recorded",
            bool(review.get("ben_notification_required")),
            False,
            "strategy",
            f"strategic_change_level={review.get('strategic_change_level')}; notify_ben={review.get('notify_ben')}",
        ),
        _gate(
            "teacher_training_replicates",
            bool(axes_by_name["teacher_training_adequacy"]["passed"]),
            False,
            "scientific",
            axes_by_name["teacher_training_adequacy"]["evidence"],
        ),
        _gate(
            "oracle_sparse_scale_adequate",
            bool(axes_by_name["sparse_residual_scale"]["passed"]),
            False,
            "scientific",
            axes_by_name["sparse_residual_scale"]["evidence"],
        ),
        _gate(
            "oracle_support_beats_shuffled_null_across_seeds",
            bool(axes_by_name["oracle_support_representability"]["passed"]),
            False,
            "scientific",
            axes_by_name["oracle_support_representability"]["evidence"],
        ),
    ]


def _axis(axis: str, passed: bool, localization: str, evidence: str, interpretation: str) -> dict[str, Any]:
    return {
        "axis": axis,
        "passed": bool(passed),
        "localization": localization,
        "evidence": evidence,
        "interpretation": interpretation,
    }


def _gate(criterion: str, passed: bool, required: bool, gate_type: str, evidence: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "required": required,
        "gate_type": gate_type,
        "evidence": evidence,
    }


def _localization_summary(axes: list[dict[str, Any]]) -> dict[str, Any]:
    blocked = [row["axis"] for row in axes if not row["passed"]]
    return {
        "primary_blockers": blocked,
        "interpretation": (
            "The two-seed evidence localizes the failure upstream of GPU validation: teacher adequacy is not stable, "
            "oracle sparse residuals are materially under-scale, and oracle support does not robustly beat shuffled "
            "teacher-residual nulls. Learned-router tuning should wait until value representability is repaired."
        ),
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {
        "present": path.is_file(),
        "path": str(path),
        "strategic_change_level": "",
        "notify_ben": False,
        "ben_notification_required": False,
        "recommended_next_action": "",
        "verdict": "",
    }
    if not path.is_file():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "notify_ben":
            data[key] = value.lower() == "true"
        elif key in {"strategic_change_level", "recommended_next_action", "verdict"}:
            data[key] = value
    data["ben_notification_required"] = (
        data.get("notify_ben") is True or data.get("strategic_change_level") == "major"
    )
    return data


def _strategy_review_handling(review: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommendation_disposition": "accepted",
        "ben_should_be_notified": bool(review.get("ben_notification_required")),
        "direction_shift": (
            "PC/core-periphery and teacher-support Transformer-ACSR stay closed locally; dense-teacher residual "
            "columnability now requires failure localization and value-capacity/norm-control repair before GPU."
        ),
        "deferred_or_rejected_recommendations": "",
    }


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_seeds.csv", summary["source_seeds"])
    _write_csv(out_dir / "failure_axes.csv", summary["failure_axes"])
    _write_csv(out_dir / "arm_comparison.csv", summary["arm_comparison"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _notes(summary: dict[str, Any]) -> str:
    lines = [
        "# Dense-Teacher Residual Columnability Failure Localization",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Ben should be notified: `{summary['strategy_review_handling']['ben_should_be_notified']}`",
        "",
        "## Interpretation",
        "",
        summary["localization"]["interpretation"],
        "",
        "## Next Step",
        "",
        str(summary["selected_next_step"]),
        "",
    ]
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _arms(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row.get("arm", ""): row for row in _list(packet.get("arm_metrics"))}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _ratio(left: float | None, right: float | None) -> float | None:
    if left is None or right in (None, 0.0):
        return None
    return left / right


def _all_positive(values: list[float | None], *, min_edge: float) -> bool:
    return bool(values) and all(value is not None and value > min_edge for value in values)


def _all_at_least(values: list[float | None], threshold: float) -> bool:
    return bool(values) and all(value is not None and value >= threshold for value in values)


def _any_small_or_negative(values: list[float | None], *, threshold: float) -> bool:
    return any(value is None or value <= threshold for value in values)


def _count_small_or_negative(values: list[float | None], threshold: float) -> int:
    return sum(1 for value in values if value is None or value <= threshold)


def _format_values(values: list[float | None]) -> str:
    return ",".join("none" if value is None else f"{value:.6f}" for value in values)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, action="append", dest="source_dirs")
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    source_dirs = tuple(args.source_dirs) if args.source_dirs else DEFAULT_SOURCE_DIRS
    summary = run_dense_teacher_residual_columnability_failure_localization(
        source_dirs=source_dirs,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({key: summary[key] for key in ("status", "decision", "claim_status", "selected_next_step")}, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
