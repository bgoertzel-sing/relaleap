"""Gate the dense-teacher columnability scaffold before GPU validation."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_DENSE_PRIMARY_DIR = Path("results/reports/dense_primary_mechanism_assay")
DEFAULT_DISTILLATION_DIR = Path(
    "results/audits/token_larger_dense_teacher_residual_distillation_comparison"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_columnability_gate")

TEACHER_ARM = "parameter_matched_causal_mlp_control"
REQUIRED_SPARSE_STUDENT_ARMS = (
    "promoted_contextual_topk2_ce_mse_distill",
    "promoted_contextual_topk2_mse_only_distill",
)
REQUIRED_CONTROL_ARMS = (
    "dense_teacher_parameter_matched_mlp",
    "dense_rank_norm_control",
    "rank_matched_contextual_topk1",
    "random_support_topk2",
    "fixed_support_topk2",
    "token_position_only_router_topk2",
    "shuffled_feature_router_topk2",
    "shuffled_teacher_target_topk2",
)
REQUIRED_TEACHER_FIELDS = (
    "teacher_hidden_residual_export",
    "teacher_logit_residual_export",
    "teacher_residual_mse",
    "teacher_residual_r2",
    "teacher_residual_cosine",
)
REQUIRED_ACCOUNTING_FIELDS = (
    "stored_params",
    "active_params",
    "active_rank_or_topk",
    "residual_l2",
    "residual_norm_ratio",
    "flops_estimate",
)
REQUIRED_MECHANISM_FIELDS = (
    "ce_loss",
    "anchor_kl_or_logit_mse",
    "functional_churn",
    "intervention_fingerprint_purity",
    "support_regret",
    "commutator_norm",
)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "arm_contract.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_dense_teacher_columnability_gate(
    *,
    dense_primary_dir: Path = DEFAULT_DENSE_PRIMARY_DIR,
    distillation_dir: Path = DEFAULT_DISTILLATION_DIR,
    strategy_review: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Check whether existing artifacts can support dense-teacher columnability."""

    start = time.time()
    dense_primary = _read_json(dense_primary_dir / "summary.json")
    distillation = _read_json(distillation_dir / "summary.json")
    review = _read_strategy_review(strategy_review)
    source_rows = _source_rows(dense_primary_dir, dense_primary, distillation_dir, distillation, strategy_review, review)
    arm_rows = _arm_contract_rows(dense_primary, distillation)
    criteria = _gate_criteria(dense_primary, distillation, review, arm_rows)
    failures = [row for row in criteria if not row["passed"]]
    scientific_gate = "blocked" if failures else "ready_for_local_validation"
    decision = (
        "dense_teacher_columnability_scaffold_blocked_missing_contract"
        if failures
        else "dense_teacher_columnability_scaffold_ready_for_local_validation"
    )
    summary = {
        "status": "pass",
        "scientific_gate": scientific_gate,
        "decision": decision,
        "claim_status": (
            "dense_teacher_columnability_not_interpretable_until_contract_is_complete"
            if failures
            else "dense_teacher_columnability_contract_complete_no_gpu_claim"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "teacher_arm": TEACHER_ARM,
        "selected_next_step": _selected_next_step(failures),
        "source_rows": source_rows,
        "arm_contract": arm_rows,
        "criteria": criteria,
        "failures": failures,
        "strategy_review": review,
        "backend_policy": "RunPod remains blocked until this local columnability contract and validation gate pass.",
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, source_rows, arm_rows, criteria)
    return summary


def _source_rows(
    dense_primary_dir: Path,
    dense_primary: dict[str, Any],
    distillation_dir: Path,
    distillation: dict[str, Any],
    strategy_review: Path,
    review: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "source": "dense_primary_mechanism_assay",
            "path": str(dense_primary_dir / "summary.json"),
            "present": (dense_primary_dir / "summary.json").is_file(),
            "status": dense_primary.get("status", ""),
            "decision": dense_primary.get("decision", ""),
            "primary_arm": dense_primary.get("primary_arm", ""),
        },
        {
            "source": "dense_teacher_residual_distillation_comparison",
            "path": str(distillation_dir / "summary.json"),
            "present": (distillation_dir / "summary.json").is_file(),
            "status": distillation.get("status", ""),
            "decision": distillation.get("decision", ""),
            "primary_arm": _primary_variant(distillation),
        },
        {
            "source": "strategy_review",
            "path": str(strategy_review),
            "present": strategy_review.is_file(),
            "status": review.get("status", ""),
            "decision": review.get("recommended_next_action", ""),
            "primary_arm": "",
        },
    ]


def _arm_contract_rows(dense_primary: dict[str, Any], distillation: dict[str, Any]) -> list[dict[str, Any]]:
    observed = _observed_arm_packets(dense_primary, distillation)
    required_arms = (TEACHER_ARM,) + REQUIRED_SPARSE_STUDENT_ARMS + REQUIRED_CONTROL_ARMS
    rows: list[dict[str, Any]] = []
    for arm in required_arms:
        packet = observed.get(arm, {})
        observed_fields = set(packet)
        required_fields = set(REQUIRED_MECHANISM_FIELDS)
        if arm == TEACHER_ARM:
            required_fields |= set(REQUIRED_TEACHER_FIELDS)
        required_fields |= set(REQUIRED_ACCOUNTING_FIELDS)
        missing = sorted(field for field in required_fields if field not in observed_fields or packet.get(field) in ("", None))
        rows.append(
            {
                "arm": arm,
                "role": _arm_role(arm),
                "present": bool(packet),
                "required_field_count": len(required_fields),
                "missing_field_count": len(missing),
                "missing_fields": ",".join(missing),
                "has_teacher_residual_target": all(
                    field in observed_fields and packet.get(field) not in ("", None)
                    for field in REQUIRED_TEACHER_FIELDS
                )
                if arm == TEACHER_ARM
                else "",
                "has_param_rank_norm_accounting": all(
                    field in observed_fields and packet.get(field) not in ("", None)
                    for field in REQUIRED_ACCOUNTING_FIELDS
                ),
            }
        )
    return rows


def _observed_arm_packets(dense_primary: dict[str, Any], distillation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    packets: dict[str, dict[str, Any]] = {}
    for row in _list(dense_primary.get("candidate_scorecard")):
        arm = str(row.get("arm", ""))
        if arm:
            packets[arm] = dict(row)
    for row in _list(distillation.get("variant_rows")):
        arm = str(row.get("arm", ""))
        variant = str(row.get("variant", ""))
        mapped = {
            "promoted_contextual_router_support": "promoted_contextual_topk2_ce_mse_distill",
            "token_position_only_predicted_support": "token_position_only_router_topk2",
            "shuffled_predicted_support": "shuffled_feature_router_topk2",
        }.get(variant, arm or variant)
        if mapped:
            packets.setdefault(mapped, {}).update(row)
    return packets


def _gate_criteria(
    dense_primary: dict[str, Any],
    distillation: dict[str, Any],
    review: dict[str, Any],
    arm_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_arm = {row["arm"]: row for row in arm_rows}
    required_arm_set = {TEACHER_ARM, *REQUIRED_SPARSE_STUDENT_ARMS, *REQUIRED_CONTROL_ARMS}
    observed_complete = {arm for arm, row in by_arm.items() if row["present"] and row["missing_field_count"] == 0}
    teacher_row = by_arm.get(TEACHER_ARM, {})
    return [
        _criterion(
            "strategy_review_consumed",
            review.get("status") == "read",
            "latest strategy review is read and folded into this local gate",
            review.get("recommended_next_action", ""),
            "strategy review missing or unread",
        ),
        _criterion(
            "dense_primary_selected_parameter_matched_mlp_teacher",
            dense_primary.get("status") == "pass" and dense_primary.get("primary_arm") == TEACHER_ARM,
            "dense primary assay selected the parameter-matched causal MLP control as teacher/control",
            {"status": dense_primary.get("status"), "primary_arm": dense_primary.get("primary_arm")},
            "dense primary assay did not select the expected MLP teacher arm",
        ),
        _criterion(
            "teacher_residual_exports_available",
            teacher_row.get("has_teacher_residual_target") is True,
            "teacher hidden/logit residual exports plus MSE/R2/cosine fields are available",
            teacher_row,
            "teacher residual export fields are missing; existing artifacts expose aggregate/logit metrics only",
        ),
        _criterion(
            "param_rank_norm_flop_accounting_available",
            all(row["has_param_rank_norm_accounting"] is True for row in arm_rows),
            "every required arm declares stored/active params, active rank/top-k, residual norm, norm ratio, and FLOPs",
            {"missing_by_arm": {row["arm"]: row["missing_fields"] for row in arm_rows if row["missing_field_count"]}},
            "one or more required arms lacks parameter/rank/norm/FLOP accounting",
        ),
        _criterion(
            "required_sparse_student_and_null_arms_declared",
            required_arm_set.issubset(observed_complete),
            "promoted top-k2 MSE-only and CE+MSE students plus dense/rank/top-k1/random/fixed/token-position/shuffle nulls are complete",
            {"missing_or_incomplete": sorted(required_arm_set - observed_complete)},
            "required student/control/null arms are absent or incomplete",
        ),
        _criterion(
            "prior_distillation_not_promoted_as_evidence",
            distillation.get("status") in {"pass", "fail", ""}
            and distillation.get("decision", "") != "dense_teacher_residual_distillation_acsr_pilot_supported_not_promoted",
            "prior dense-teacher comparison is treated as context, not a completed columnability pass",
            {"status": distillation.get("status"), "decision": distillation.get("decision")},
            "prior dense-teacher artifact would be misread as a completed columnability gate",
        ),
    ]


def _criterion(
    criterion: str,
    passed: bool,
    threshold: str,
    actual: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": actual,
        "failure_reason": "" if passed else failure_reason,
    }


def _selected_next_step(failures: list[dict[str, Any]]) -> str:
    if failures:
        return (
            "add teacher residual export plus dense_teacher_columnability arm metrics for promoted top-k2 "
            "MSE-only/CE+MSE students and mandatory dense/rank/top-k1/random/fixed/token-position/shuffle nulls"
        )
    return "run the local dense_teacher_columnability validation command; keep RunPod blocked until it passes"


def _arm_role(arm: str) -> str:
    if arm == TEACHER_ARM:
        return "dense_teacher"
    if arm in REQUIRED_SPARSE_STUDENT_ARMS:
        return "sparse_student"
    return "control_or_null"


def _primary_variant(summary: dict[str, Any]) -> str:
    rows = _list(summary.get("variant_rows"))
    if not rows:
        return ""
    return str(rows[0].get("variant", ""))


def _read_strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    data: dict[str, Any] = {"status": "read", "path": str(path)}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"strategic_change_level", "notify_ben", "recommended_next_action", "verdict"}:
            data[key] = value.strip()
    return data


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    criteria: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_rows.csv", source_rows)
    _write_csv(out_dir / "arm_contract.csv", arm_rows)
    _write_csv(out_dir / "gate_criteria.csv", criteria)
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Dense Teacher Columnability Gate",
        "",
        f"Status: `{summary['status']}`",
        f"Scientific gate: `{summary['scientific_gate']}`",
        f"Decision: `{summary['decision']}`",
        f"Teacher arm: `{summary['teacher_arm']}`",
        "",
        "## Interpretation",
        "",
        "This is a local artifact-contract gate for dense-teacher columnability, not GPU evidence.",
        "The GPT-5.5-Pro recommendation to treat the parameter-matched causal MLP as a teacher/control is accepted.",
        "RunPod remains blocked until the local teacher residual export and null-arm contract is complete.",
        "",
        "## Next Step",
        "",
        summary["selected_next_step"],
        "",
        "## Failures",
        "",
    ]
    failures = summary.get("failures", [])
    if failures:
        for failure in failures:
            lines.append(f"- `{failure['criterion']}`: {failure['failure_reason']}")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-primary-dir", type=Path, default=DEFAULT_DENSE_PRIMARY_DIR)
    parser.add_argument("--distillation-dir", type=Path, default=DEFAULT_DISTILLATION_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_dense_teacher_columnability_gate(
        dense_primary_dir=args.dense_primary_dir,
        distillation_dir=args.distillation_dir,
        strategy_review=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
