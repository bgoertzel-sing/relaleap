"""Local value-capacity/norm-control pregate for dense-teacher columns.

This is a command-driven planning pregate after the dense-teacher residual
columnability failure localization. It consumes prior trained assay artifacts,
records the scale/value-capacity repair contract, and fails closed before GPU.
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


DEFAULT_FAILURE_LOCALIZATION_DIR = Path("results/reports/dense_teacher_residual_columnability_failure_localization")
DEFAULT_ASSAY_DIRS = (
    Path("results/reports/dense_teacher_residual_columnability_assay"),
    Path("results/reports/dense_teacher_residual_columnability_assay_seed2"),
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_residual_value_capacity_norm_pregate")

DECISION = "dense_teacher_residual_value_capacity_norm_pregate_recorded"
FAIL_DECISION = "dense_teacher_residual_value_capacity_norm_pregate_failed_closed"
NEXT_STEP = "implement bounded local value-capacity/norm-control trained assay for dense-teacher residual dictionaries"
REPAIR_STEP = "repair missing dense-teacher residual columnability sources before value-capacity/norm-control pregate"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "redesign_arms.csv",
    "norm_control_contract.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_dense_teacher_residual_value_capacity_norm_pregate(
    *,
    failure_localization_dir: Path = DEFAULT_FAILURE_LOCALIZATION_DIR,
    assay_dirs: tuple[Path, ...] = DEFAULT_ASSAY_DIRS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed local pregate for the next trained repair assay."""

    start = time.time()
    failure_summary = _read_json(failure_localization_dir / "summary.json")
    assay_summaries = [_read_json(path / "summary.json") for path in assay_dirs]
    review = _strategy_review(strategy_review_path)
    source_rows = _source_rows(failure_localization_dir, assay_dirs, failure_summary, assay_summaries, strategy_review_path)
    runtime_failures = [row for row in source_rows if row["required"] and not row["present"]]
    evidence = _evidence(failure_summary, assay_summaries)
    redesign_arms = [] if runtime_failures else _redesign_arms(evidence)
    norm_rows = [] if runtime_failures else _norm_control_rows(evidence)
    gate_rows = _gate_rows(source_rows, evidence, redesign_arms, norm_rows, review)
    hard_failures = [row for row in gate_rows if row["required"] and not row["passed"]]
    status = "fail" if hard_failures else "pass"
    summary = {
        "status": status,
        "decision": FAIL_DECISION if hard_failures else DECISION,
        "claim_status": (
            "value_capacity_norm_pregate_ready_for_local_training_no_gpu"
            if status == "pass"
            else "value_capacity_norm_pregate_incomplete_fail_closed"
        ),
        "selected_next_step": NEXT_STEP if status == "pass" else REPAIR_STEP,
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "training_executed": False,
        "source_rows": source_rows,
        "evidence_summary": evidence,
        "redesign_arms": redesign_arms,
        "norm_control_contract": norm_rows,
        "gate_criteria": gate_rows,
        "failures": hard_failures + [row for row in gate_rows if row["gate_type"] == "scientific" and not row["passed"]],
        "strategy_review": review,
        "strategy_review_handling": _strategy_review_handling(review),
        "backend_policy": "local CPU pregate only; RunPod/Colab remain blocked until a trained local repair clears gates",
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _source_rows(
    failure_dir: Path,
    assay_dirs: tuple[Path, ...],
    failure_summary: dict[str, Any],
    assay_summaries: list[dict[str, Any]],
    review_path: Path,
) -> list[dict[str, Any]]:
    rows = [
        {
            "source": "dense_teacher_residual_columnability_failure_localization",
            "path": str(failure_dir / "summary.json"),
            "present": bool(failure_summary),
            "required": True,
            "status": failure_summary.get("status", ""),
            "decision": failure_summary.get("decision", ""),
            "claim_status": failure_summary.get("claim_status", ""),
        }
    ]
    for path, summary in zip(assay_dirs, assay_summaries):
        rows.append(
            {
                "source": path.name,
                "path": str(path / "summary.json"),
                "present": bool(summary),
                "required": True,
                "status": summary.get("status", ""),
                "decision": summary.get("decision", ""),
                "claim_status": summary.get("claim_status", ""),
            }
        )
    rows.append(
        {
            "source": "latest_strategy_review",
            "path": str(review_path),
            "present": review_path.is_file(),
            "required": False,
            "status": "read" if review_path.is_file() else "missing_optional",
            "decision": "",
            "claim_status": "",
        }
    )
    return rows


def _evidence(failure_summary: dict[str, Any], assay_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    seed_rows = [row for row in _list(failure_summary.get("source_seeds")) if row.get("present")]
    oracle_ratios = [_float(row.get("oracle_to_teacher_l2_ratio")) for row in seed_rows]
    oracle_edges = [_float(row.get("oracle_sparse_mse_advantage_vs_shuffled_null")) for row in seed_rows]
    teacher_edges = [_float(row.get("teacher_ce_improvement")) for row in seed_rows]
    arms = [arm for summary in assay_summaries for arm in _list(summary.get("arm_metrics"))]
    dense_controls = [arm for arm in arms if arm.get("arm") in {"dense_teacher_residual_control", "rank_matched_residual_control", "norm_clipped_mlp_control"}]
    oracle_sparse = [arm for arm in arms if arm.get("arm") == "oracle_support_sparse_dictionary"]
    learned_sparse = [arm for arm in arms if arm.get("arm") == "learned_causal_router_sparse_dictionary"]
    return {
        "source_seed_count": len(seed_rows),
        "teacher_ce_improvements": oracle_safe_values(teacher_edges),
        "oracle_to_teacher_l2_ratios": oracle_safe_values(oracle_ratios),
        "oracle_mse_advantages_vs_shuffled_null": oracle_safe_values(oracle_edges),
        "min_oracle_to_teacher_l2_ratio": _min(oracle_ratios),
        "max_oracle_to_teacher_l2_ratio": _max(oracle_ratios),
        "min_oracle_mse_advantage_vs_shuffled_null": _min(oracle_edges),
        "teacher_replication_passed": bool(teacher_edges) and all(value is not None and value > 0.0 for value in teacher_edges),
        "oracle_scale_passed": bool(oracle_ratios) and all(value is not None and value >= 0.5 for value in oracle_ratios),
        "oracle_null_passed": bool(oracle_edges) and all(value is not None and value > 0.02 for value in oracle_edges),
        "dense_control_count": len(dense_controls),
        "oracle_sparse_count": len(oracle_sparse),
        "learned_sparse_count": len(learned_sparse),
        "failure_interpretation": failure_summary.get("localization", {}).get("interpretation", ""),
    }


def _redesign_arms(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _arm(
            "oracle_support_norm_matched_multi_value_dictionary",
            "oracle_sparse_ceiling",
            "nondeployable oracle support; values fit train split only",
            "increase per-support value capacity and match teacher residual L2 mean/p95 before router diagnosis",
            "must beat shuffled/delayed target nulls on MSE and preserve CE guardrail",
            False,
        ),
        _arm(
            "oracle_support_low_rank_value_dictionary",
            "oracle_sparse_ceiling",
            "nondeployable oracle support; low-rank values fit train split only",
            "separate sparse value capacity from dense/rank representability",
            "must close dense-teacher residual MSE gap better than rank-matched dense control at matched norm",
            False,
        ),
        _arm(
            "learned_router_norm_matched_multi_value_dictionary",
            "deployable_candidate",
            "prefix-safe learned router; no oracle support, task id, or teacher labels at eval",
            "test whether routing remains a blocker after oracle value capacity is repaired",
            "only interpretable if oracle-support arms first beat nulls and scale gates",
            True,
        ),
        _arm(
            "same_router_flat_value_norm_matched_control",
            "value_control",
            "same learned support with flat MLP value head",
            "distinguish sparse reusable values from same-router dense value capacity",
            "candidate sparse arm must beat this control on churn/commutator at comparable CE",
            True,
        ),
        _arm(
            "random_frequency_token_position_norm_matched_nulls",
            "support_nulls",
            "random/frequency/token-position supports with matched residual norm",
            "prevent residual scale or support-frequency artifacts from masquerading as columnability",
            "candidate/oracle arms must beat all support nulls on MSE and CE guardrail",
            True,
        ),
        _arm(
            "shuffled_delayed_teacher_residual_norm_matched_nulls",
            "target_nulls",
            "misaligned teacher residual targets with same fitting budget",
            "test whether value dictionaries learn meaningful correction fields rather than target-aware quantization",
            "oracle support must beat these nulls across seeds before GPU",
            True,
        ),
    ]


def _arm(
    arm: str,
    family: str,
    target_access_contract: str,
    purpose: str,
    advancement_gate: str,
    deployable_at_eval: bool,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "family": family,
        "target_access_contract": target_access_contract,
        "purpose": purpose,
        "advancement_gate": advancement_gate,
        "deployable_at_eval": deployable_at_eval,
        "uses_future_hidden_or_delta": False,
        "uses_task_id": False,
        "uses_teacher_labels_in_deployable_router": False,
    }


def _norm_control_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    ratio = evidence.get("min_oracle_to_teacher_l2_ratio")
    ratio_text = "" if ratio is None else f"{ratio:.6f}"
    return [
        {
            "control": "teacher_residual_l2_mean_match",
            "required": True,
            "current_evidence": f"min oracle/teacher L2 ratio={ratio_text}",
            "next_assay_contract": "report mean and p95 residual L2; sparse oracle must reach >=0.5 mean-ratio and include clipped norm-matched rows",
        },
        {
            "control": "parameter_budget_match",
            "required": True,
            "current_evidence": f"dense controls={evidence.get('dense_control_count')}; oracle sparse rows={evidence.get('oracle_sparse_count')}",
            "next_assay_contract": "report active/stored params for each value-capacity arm and dense/rank/MLP controls",
        },
        {
            "control": "null_target_access_match",
            "required": True,
            "current_evidence": f"min oracle-vs-shuffled MSE advantage={_format_optional(evidence.get('min_oracle_mse_advantage_vs_shuffled_null'))}",
            "next_assay_contract": "shuffled/delayed nulls must get the same target-access and norm-fitting privileges as sparse oracle ceilings",
        },
        {
            "control": "interference_observables",
            "required": True,
            "current_evidence": "prior assay reports churn, finite-update commutator proxy, retention, and intervention selectivity",
            "next_assay_contract": "keep churn/commutator/retention/selectivity as primary gates; CE remains a guardrail",
        },
    ]


def _gate_rows(
    source_rows: list[dict[str, Any]],
    evidence: dict[str, Any],
    arms: list[dict[str, Any]],
    norm_rows: list[dict[str, Any]],
    review: dict[str, Any],
) -> list[dict[str, Any]]:
    sources_present = all(row["present"] for row in source_rows if row["required"])
    arms_by_family = {row["family"] for row in arms}
    return [
        _gate("required_sources_present", sources_present, True, "runtime", "failure localization and both assay summaries must exist"),
        _gate("strategy_review_major_notify_recorded", bool(review.get("ben_notification_required")), False, "strategy", f"strategic_change_level={review.get('strategic_change_level')}; notify_ben={review.get('notify_ben')}"),
        _gate("previous_local_gates_block_gpu", not evidence.get("oracle_scale_passed") or not evidence.get("oracle_null_passed"), False, "scientific", "pregate is justified by local sparse scale/null failures"),
        _gate("redesign_includes_oracle_deployable_and_null_families", {"oracle_sparse_ceiling", "deployable_candidate", "support_nulls", "target_nulls"}.issubset(arms_by_family), True, "contract", ",".join(sorted(arms_by_family))),
        _gate("norm_controls_required", len(norm_rows) >= 4 and all(row["required"] for row in norm_rows), True, "contract", f"norm_control_rows={len(norm_rows)}"),
        _gate("no_gpu_or_promotion_claim", True, True, "runtime", "requires_gpu_now=false; advance_to_gpu_validation=false; promotion_allowed=false"),
    ]


def _gate(criterion: str, passed: bool, required: bool, gate_type: str, evidence: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "required": required,
        "gate_type": gate_type,
        "evidence": evidence,
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
    data["ben_notification_required"] = data.get("notify_ben") is True or data.get("strategic_change_level") == "major"
    return data


def _strategy_review_handling(review: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommendation_disposition": "accepted",
        "ben_should_be_notified": bool(review.get("ben_notification_required")),
        "direction_shift": (
            "Maintain the GPT-5.5-Pro major pivot: PC/core-periphery and teacher-support Transformer-ACSR stay closed; "
            "repair dense-teacher sparse value capacity and norm controls locally before any GPU validation."
        ),
        "deferred_or_rejected_recommendations": "",
    }


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "redesign_arms.csv", summary["redesign_arms"])
    _write_csv(out_dir / "norm_control_contract.csv", summary["norm_control_contract"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _notes(summary: dict[str, Any]) -> str:
    lines = [
        "# Dense-Teacher Residual Value-Capacity/Norm Pregate",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Training executed: `{summary['training_executed']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Ben should be notified: `{summary['strategy_review_handling']['ben_should_be_notified']}`",
        "",
        "## Interpretation",
        "",
        str(summary["evidence_summary"].get("failure_interpretation", "")),
        "",
        "This artifact is a local pregate, not evidence that the redesigned sparse dictionary works.",
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


def oracle_safe_values(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


def _min(values: list[float | None]) -> float | None:
    clean = oracle_safe_values(values)
    return min(clean) if clean else None


def _max(values: list[float | None]) -> float | None:
    clean = oracle_safe_values(values)
    return max(clean) if clean else None


def _format_optional(value: Any) -> str:
    numeric = _float(value)
    return "" if numeric is None else f"{numeric:.6f}"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-localization-dir", type=Path, default=DEFAULT_FAILURE_LOCALIZATION_DIR)
    parser.add_argument("--assay-dir", type=Path, action="append", dest="assay_dirs")
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    assay_dirs = tuple(args.assay_dirs) if args.assay_dirs else DEFAULT_ASSAY_DIRS
    summary = run_dense_teacher_residual_value_capacity_norm_pregate(
        failure_localization_dir=args.failure_localization_dir,
        assay_dirs=assay_dirs,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({key: summary[key] for key in ("status", "decision", "claim_status", "selected_next_step")}, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
