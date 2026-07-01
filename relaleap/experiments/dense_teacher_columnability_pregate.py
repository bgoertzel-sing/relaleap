"""Pregate dense-teacher residual targets before sparse-column scaling."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_BRANCH_INVENTORY = Path("results/reports/mechanism_branch_inventory/summary.json")
DEFAULT_DISTILLATION_DIR = Path(
    "results/audits/token_larger_dense_teacher_residual_distillation_comparison"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_columnability_pregate")

REQUIRED_TENSORS = (
    "base_hidden.pt",
    "base_logits.pt",
    "teacher_hidden_residual.pt",
    "teacher_logit_residual.pt",
)
REQUIRED_NULL_ARMS = (
    "token_position_only_router_topk2",
    "random_support_topk2",
    "fixed_support_topk2",
    "shuffled_feature_router_topk2",
    "shuffled_teacher_target_topk2",
)
SPARSE_STUDENT_ARMS = (
    "promoted_contextual_topk2_ce_mse_distill",
    "promoted_contextual_topk2_mse_only_distill",
    "norm_budgeted_promoted_contextual_topk2_ce_mse_distill",
    "rank_matched_contextual_topk1",
)
NORM_BUDGET_ARM = "norm_budgeted_promoted_contextual_topk2_ce_mse_distill"
PRIMARY_STUDENT_ARM = "promoted_contextual_topk2_ce_mse_distill"
REQUIRED_ARTIFACTS = (
    "summary.json",
    "target_tensors.csv",
    "arm_rows.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_dense_teacher_columnability_pregate(
    *,
    branch_inventory_path: Path = DEFAULT_BRANCH_INVENTORY,
    distillation_dir: Path = DEFAULT_DISTILLATION_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
    min_teacher_ce_improvement: float = 0.25,
    min_sparse_teacher_residual_r2: float = 0.20,
    min_norm_utilization: float = 0.20,
    max_functional_churn: float = 0.50,
) -> dict[str, Any]:
    """Write a local fail-closed pregate for dense-teacher columnability."""

    start = time.time()
    branch_inventory = _read_json(branch_inventory_path)
    distillation_summary = _read_json(distillation_dir / "summary.json")
    strategy_review = _strategy_review(strategy_review_path)
    variant_rows = _list(distillation_summary.get("variant_rows"))
    arm_rows = _selected_arm_rows(variant_rows)
    target_rows = _target_tensor_rows(distillation_dir)
    criteria = _criteria(
        branch_inventory=branch_inventory,
        distillation_summary=distillation_summary,
        strategy_review=strategy_review,
        target_rows=target_rows,
        arm_rows=arm_rows,
        min_teacher_ce_improvement=min_teacher_ce_improvement,
        min_sparse_teacher_residual_r2=min_sparse_teacher_residual_r2,
        min_norm_utilization=min_norm_utilization,
        max_functional_churn=max_functional_churn,
    )
    failures = [row for row in criteria if not row["passed"]]
    scientific_gate = "ready_for_local_columnability_validation" if not failures else "blocked"
    decision = (
        "dense_teacher_columnability_pregate_ready_for_local_validation"
        if not failures
        else "dense_teacher_columnability_pregate_blocked"
    )
    summary = {
        "status": "pass",
        "scientific_gate": scientific_gate,
        "decision": decision,
        "claim_status": (
            "dense_teacher_sparse_columnability_prerequisites_pass_no_gpu_claim"
            if not failures
            else "dense_teacher_sparse_columnability_prerequisites_not_met"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "selected_next_step": (
            "run the local dense-teacher columnability validation gate with matched nulls"
            if not failures
            else "do not run GPU; local evidence shows dense-teacher sparse-columnability is blocked or incomplete"
        ),
        "source_rows": _source_rows(
            branch_inventory_path,
            branch_inventory,
            distillation_dir,
            distillation_summary,
            strategy_review_path,
            strategy_review,
        ),
        "target_tensors": target_rows,
        "arm_rows": arm_rows,
        "gate_criteria": criteria,
        "failures": failures,
        "thresholds": {
            "min_teacher_ce_improvement": min_teacher_ce_improvement,
            "min_sparse_teacher_residual_r2": min_sparse_teacher_residual_r2,
            "min_norm_utilization": min_norm_utilization,
            "max_functional_churn": max_functional_churn,
        },
        "strategy_review": strategy_review,
        "direction_shift": _direction_shift(strategy_review),
        "backend_policy": "local pregate only; RunPod and Colab remain blocked until a local columnability gate passes",
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _source_rows(
    branch_inventory_path: Path,
    branch_inventory: dict[str, Any],
    distillation_dir: Path,
    distillation_summary: dict[str, Any],
    strategy_review_path: Path,
    strategy_review: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "source": "mechanism_branch_inventory",
            "path": str(branch_inventory_path),
            "present": branch_inventory_path.is_file(),
            "status": branch_inventory.get("status", ""),
            "decision": branch_inventory.get("decision", ""),
            "selected_next_action": branch_inventory.get("selected_next_action", ""),
        },
        {
            "source": "dense_teacher_residual_distillation_comparison",
            "path": str(distillation_dir / "summary.json"),
            "present": (distillation_dir / "summary.json").is_file(),
            "status": distillation_summary.get("status", ""),
            "decision": distillation_summary.get("decision", ""),
            "selected_next_action": distillation_summary.get("selected_next_step", ""),
        },
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy_review_path.is_file(),
            "status": strategy_review.get("status", ""),
            "decision": strategy_review.get("recommended_next_action", ""),
            "selected_next_action": "",
        },
    ]


def _target_tensor_rows(distillation_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        return [
            {
                "tensor": name.removesuffix(".pt"),
                "path": str(distillation_dir / name),
                "present": (distillation_dir / name).is_file(),
                "loadable": False,
                "shape": "",
                "finite": False,
                "mean_l2": "",
                "status": f"torch_unavailable:{exc}",
            }
            for name in REQUIRED_TENSORS
        ]

    for name in REQUIRED_TENSORS:
        path = distillation_dir / name
        row: dict[str, Any] = {
            "tensor": name.removesuffix(".pt"),
            "path": str(path),
            "present": path.is_file(),
            "loadable": False,
            "shape": "",
            "finite": False,
            "mean_l2": "",
            "status": "missing",
        }
        if path.is_file():
            try:
                value = torch.load(path, map_location="cpu")
                row["loadable"] = True
                row["shape"] = "x".join(str(dim) for dim in tuple(value.shape))
                finite = bool(torch.isfinite(value).all().item())
                row["finite"] = finite
                if value.ndim >= 1:
                    flat = value.reshape(-1, value.shape[-1])
                    row["mean_l2"] = float(flat.norm(dim=-1).mean().item())
                else:
                    row["mean_l2"] = float(value.abs().item())
                row["status"] = "ok" if finite else "nonfinite"
            except Exception as exc:  # pragma: no cover - corrupt tensor path
                row["status"] = f"load_failed:{exc}"
        rows.append(row)
    return rows


def _selected_arm_rows(variant_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = set(SPARSE_STUDENT_ARMS) | set(REQUIRED_NULL_ARMS) | {"parameter_matched_causal_mlp_control"}
    rows: list[dict[str, Any]] = []
    for arm in sorted(wanted):
        row = _find_arm(variant_rows, arm)
        rows.append(
            {
                "arm": arm,
                "present": bool(row),
                "role": _arm_role(arm),
                "teacher_scale": _float_or_none(row.get("teacher_scale")),
                "ce_loss": _float_or_none(row.get("ce_loss")),
                "teacher_residual_r2": _float_or_none(row.get("teacher_residual_r2")),
                "teacher_residual_mse": _float_or_none(row.get("teacher_residual_mse")),
                "teacher_logit_mse": _float_or_none(row.get("teacher_logit_mse")),
                "residual_norm_ratio": _float_or_none(row.get("residual_norm_ratio")),
                "residual_norm_budget": _float_or_none(row.get("residual_norm_budget")),
                "residual_norm_budget_error": _float_or_none(row.get("residual_norm_budget_error")),
                "residual_norm_budget_overuse": _float_or_none(row.get("residual_norm_budget_overuse")),
                "functional_churn": _float_or_none(row.get("functional_churn")),
                "commutator_norm": _float_or_none(row.get("commutator_norm")),
                "support_regret": _float_or_none(row.get("support_regret")),
                "intervention_fingerprint_purity": _float_or_none(row.get("intervention_fingerprint_purity")),
            }
        )
    return rows


def _criteria(
    *,
    branch_inventory: dict[str, Any],
    distillation_summary: dict[str, Any],
    strategy_review: dict[str, Any],
    target_rows: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    min_teacher_ce_improvement: float,
    min_sparse_teacher_residual_r2: float,
    min_norm_utilization: float,
    max_functional_churn: float,
) -> list[dict[str, Any]]:
    arms = {row["arm"]: row for row in arm_rows}
    primary = arms.get(PRIMARY_STUDENT_ARM, {})
    norm_budget = arms.get(NORM_BUDGET_ARM, {})
    nulls = [arms.get(arm, {}) for arm in REQUIRED_NULL_ARMS]
    sparse_rows = [arms.get(arm, {}) for arm in SPARSE_STUDENT_ARMS]
    target_shapes = {row["tensor"]: row["shape"] for row in target_rows}
    teacher_improvement = _float_or_none(distillation_summary.get("dense_teacher_ce_improvement"))
    best_sparse_r2 = max(
        (_float_or_none(row.get("teacher_residual_r2")) or -math.inf for row in sparse_rows if row.get("present")),
        default=-math.inf,
    )
    null_mses = [
        _float_or_none(row.get("teacher_residual_mse"))
        for row in nulls
        if _float_or_none(row.get("teacher_residual_mse")) is not None
    ]
    primary_mse = _float_or_none(primary.get("teacher_residual_mse"))
    return [
        _criterion(
            "strategy_review_pivot_consumed",
            strategy_review.get("status") == "read"
            and "dense-teacher columnability" in str(strategy_review.get("recommended_next_action", "")).lower(),
            "latest review must select the dense-teacher columnability pivot",
            strategy_review.get("recommended_next_action", ""),
            "strategy review missing or does not select the dense-teacher columnability pivot",
        ),
        _criterion(
            "branch_inventory_selected_pregate",
            branch_inventory.get("status") == "pass"
            and branch_inventory.get("selected_next_action") == "start_dense_teacher_columnability_pregate_before_gpu",
            "mechanism inventory must recover closed branches and select this pregate",
            {
                "status": branch_inventory.get("status"),
                "selected_next_action": branch_inventory.get("selected_next_action"),
            },
            "mechanism branch inventory did not select this pregate",
        ),
        _criterion(
            "teacher_residual_targets_collected",
            all(row["present"] and row["loadable"] and row["finite"] for row in target_rows)
            and (target_shapes.get("base_hidden") == target_shapes.get("teacher_hidden_residual"))
            and (target_shapes.get("base_logits") == target_shapes.get("teacher_logit_residual"))
            and all(float(row["mean_l2"] or 0.0) > 0.0 for row in target_rows if row["tensor"].startswith("teacher_")),
            "base/teacher hidden and logit residual tensors must be loadable, finite, shape-matched, and nonzero",
            target_rows,
            "teacher residual target tensors are missing, corrupt, nonfinite, zero, or shape-incompatible",
        ),
        _criterion(
            "dense_teacher_effect_size_nontrivial",
            teacher_improvement is not None and teacher_improvement >= min_teacher_ce_improvement,
            f"dense teacher CE improvement must be >= {min_teacher_ce_improvement}",
            teacher_improvement,
            "dense teacher correction is too small to justify a columnability test",
        ),
        _criterion(
            "required_null_controls_present",
            all(row.get("present") for row in nulls),
            "token/position, random, fixed, shuffled-feature, and shuffled-target null rows must be present",
            {row.get("arm"): row.get("present") for row in nulls},
            "one or more required null/control arms is missing",
        ),
        _criterion(
            "norm_budget_accounting_present",
            norm_budget.get("present")
            and norm_budget.get("residual_norm_budget") is not None
            and norm_budget.get("residual_norm_budget_error") is not None
            and norm_budget.get("residual_norm_budget_overuse") is not None
            and (_float_or_none(norm_budget.get("residual_norm_ratio")) or 0.0) >= min_norm_utilization,
            f"norm-budgeted sparse arm must report budget/error/overuse and use >= {min_norm_utilization} of teacher norm",
            norm_budget,
            "norm-budgeted arm lacks accounting or collapses to near-zero residual norm",
        ),
        _criterion(
            "sparse_columns_reconstruct_teacher_residual",
            best_sparse_r2 >= min_sparse_teacher_residual_r2,
            f"best sparse student teacher-residual R2 must be >= {min_sparse_teacher_residual_r2}",
            {"best_sparse_teacher_residual_r2": best_sparse_r2},
            "sparse column dictionary does not reconstruct nontrivial dense-teacher residual variance",
        ),
        _criterion(
            "learned_router_beats_null_targets",
            primary_mse is not None and null_mses and all(primary_mse < value for value in null_mses),
            "primary learned top-k2 sparse student must beat all nulls on teacher-residual MSE",
            {"primary_teacher_residual_mse": primary_mse, "null_teacher_residual_mses": null_mses},
            "learned support/router result does not beat every null on teacher-residual target reconstruction",
        ),
        _criterion(
            "interference_budget_plausible",
            (_float_or_none(primary.get("functional_churn")) or math.inf) <= max_functional_churn
            and (_float_or_none(primary.get("commutator_norm")) or math.inf) <= 0.10,
            f"primary sparse student must keep functional churn <= {max_functional_churn} and commutator norm <= 0.10",
            {
                "functional_churn": primary.get("functional_churn"),
                "commutator_norm": primary.get("commutator_norm"),
            },
            "sparse student reconstruction is too high-churn or lacks commutator cleanliness",
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


def _find_arm(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    for row in rows:
        if row.get("arm") == arm and abs((_float_or_none(row.get("teacher_scale")) or 1.0) - 1.0) <= 1e-12:
            return row
    return {}


def _arm_role(arm: str) -> str:
    if arm == "parameter_matched_causal_mlp_control":
        return "dense_teacher"
    if arm in SPARSE_STUDENT_ARMS:
        return "sparse_student"
    return "null_or_control"


def _direction_shift(strategy_review: dict[str, Any]) -> dict[str, Any]:
    notify_ben = str(strategy_review.get("notify_ben", "")).lower() == "true"
    major = str(strategy_review.get("strategic_change_level", "")).lower() == "major"
    return {
        "strategic_change_level": strategy_review.get("strategic_change_level", ""),
        "ben_should_be_notified": notify_ben or major,
        "recommendation_disposition": "accepted",
        "note": "Major dense-teacher columnability pivot recorded; Ben should be notified."
        if notify_ben or major
        else "No major notification flag in latest review.",
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    result = {
        "path": str(path),
        "status": "missing",
        "strategic_change_level": "",
        "notify_ben": "",
        "recommended_next_action": "",
        "verdict": "",
    }
    if not path.is_file():
        return result
    result["status"] = "read"
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in result:
            result[key] = value
    return result


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "target_tensors.csv", summary["target_tensors"])
    _write_csv(out_dir / "arm_rows.csv", summary["arm_rows"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _notes(summary: dict[str, Any]) -> str:
    failed = ", ".join(row["criterion"] for row in summary["failures"]) or "none"
    return "\n".join(
        [
            "# Dense Teacher Columnability Pregate",
            "",
            f"- Status: {summary['status']}",
            f"- Scientific gate: {summary['scientific_gate']}",
            f"- Decision: {summary['decision']}",
            f"- GPU: {summary['advance_to_gpu_validation']}",
            f"- Failed criteria: {failed}",
            f"- Next step: {summary['selected_next_step']}",
            "",
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _list(value: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in value] if isinstance(value, list) else []


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _csv_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-inventory", type=Path, default=DEFAULT_BRANCH_INVENTORY)
    parser.add_argument("--distillation-dir", type=Path, default=DEFAULT_DISTILLATION_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_dense_teacher_columnability_pregate(
        branch_inventory_path=args.branch_inventory,
        distillation_dir=args.distillation_dir,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "scientific_gate": summary["scientific_gate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
