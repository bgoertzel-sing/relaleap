"""Extract low-churn-MLP teacher rows for the sparse-factorization ceiling."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_DESIGN_DIR = Path("results/reports/low_churn_mlp_sparse_factorization_ceiling_design")
DEFAULT_LOW_CHURN_PILOT_DIR = Path("results/reports/low_churn_mlp_residual_control_pilot")
DEFAULT_OUT_DIR = Path("results/reports/low_churn_mlp_sparse_factorization_ceiling_extractor")

LOW_CHURN_ARM = "low_churn_mlp_residual_control"
SHUFFLED_NULL_ARM = "low_churn_mlp_shuffled_target_null"

NEXT_ACTION = "implement_low_churn_mlp_sparse_factorization_ceiling_training_harness"
REPAIR_ACTION = "repair_low_churn_mlp_sparse_factorization_ceiling_extractor_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "teacher_residual_rows.csv",
    "teacher_budget_rows.csv",
    "factorization_schema.csv",
    "support_arm_schema.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_low_churn_mlp_sparse_factorization_ceiling_extractor(
    *,
    design_dir: Path = DEFAULT_DESIGN_DIR,
    low_churn_pilot_dir: Path = DEFAULT_LOW_CHURN_PILOT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write read-only teacher rows and schema for the sparse ceiling."""

    start = time.time()
    design_summary = _read_json(design_dir / "summary.json")
    pilot_summary = _read_json(low_churn_pilot_dir / "summary.json")
    design_arms = _read_csv(design_dir / "support_arms.csv")
    design_observables = _read_csv(design_dir / "observable_rows.csv")
    arm_rows = _read_csv(low_churn_pilot_dir / "arm_metrics.csv")
    token_rows = _read_csv(low_churn_pilot_dir / "per_token_metrics.csv")

    source_rows = _source_rows(
        design_dir=design_dir,
        low_churn_pilot_dir=low_churn_pilot_dir,
        design_summary=design_summary,
        pilot_summary=pilot_summary,
        design_arms=design_arms,
        design_observables=design_observables,
        arm_rows=arm_rows,
        token_rows=token_rows,
    )
    teacher_rows = _teacher_residual_rows(token_rows)
    budget_rows = _teacher_budget_rows(arm_rows, pilot_summary)
    schema_rows = _factorization_schema_rows(design_observables)
    support_schema = _support_arm_schema_rows(design_arms)
    gate_rows = _gate_rows(
        design_summary=design_summary,
        pilot_summary=pilot_summary,
        source_rows=source_rows,
        teacher_rows=teacher_rows,
        budget_rows=budget_rows,
        schema_rows=schema_rows,
        support_schema=support_schema,
    )
    failures = [row for row in gate_rows if not row["passed"]]
    status = "pass" if not failures else "fail"
    summary = {
        "status": status,
        "decision": (
            "low_churn_mlp_sparse_factorization_ceiling_extractor_recorded"
            if status == "pass"
            else "low_churn_mlp_sparse_factorization_ceiling_extractor_failed_closed"
        ),
        "claim_status": "teacher_rows_and_schema_only_no_sparse_factorization_claim",
        "selected_next_action": NEXT_ACTION if status == "pass" else REPAIR_ACTION,
        "selected_next_step": (
            "implement the bounded sparse-factorization training harness using these teacher rows and schema"
            if status == "pass"
            else "repair missing design or low-churn pilot artifacts before training harness work"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "training_executed": False,
        "backend_policy": "local read-only extraction only; GPU validation remains blocked until sparse factorization gates exist",
        "source_rows": source_rows,
        "teacher_residual_row_count": len(teacher_rows),
        "heldout_teacher_residual_row_count": sum(1 for row in teacher_rows if row["split"] == "heldout"),
        "teacher_budget_row_count": len(budget_rows),
        "factorization_schema_row_count": len(schema_rows),
        "support_arm_schema_row_count": len(support_schema),
        "gate_criteria": gate_rows,
        "failures": failures,
        "design_dir": str(design_dir),
        "low_churn_pilot_dir": str(low_churn_pilot_dir),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(
        out_dir=out_dir,
        summary=summary,
        source_rows=source_rows,
        teacher_rows=teacher_rows,
        budget_rows=budget_rows,
        schema_rows=schema_rows,
        support_schema=support_schema,
        gate_rows=gate_rows,
    )
    return summary


def _source_rows(
    *,
    design_dir: Path,
    low_churn_pilot_dir: Path,
    design_summary: dict[str, Any],
    pilot_summary: dict[str, Any],
    design_arms: list[dict[str, str]],
    design_observables: list[dict[str, str]],
    arm_rows: list[dict[str, str]],
    token_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    return [
        {
            "source": "low_churn_sparse_factorization_design",
            "path": str(design_dir / "summary.json"),
            "present": (design_dir / "summary.json").is_file(),
            "status": design_summary.get("status", ""),
            "decision": design_summary.get("decision", ""),
            "row_count": 1 if design_summary else 0,
        },
        {
            "source": "low_churn_sparse_factorization_support_arms",
            "path": str(design_dir / "support_arms.csv"),
            "present": (design_dir / "support_arms.csv").is_file(),
            "status": "read" if design_arms else "missing_or_empty",
            "decision": "",
            "row_count": len(design_arms),
        },
        {
            "source": "low_churn_sparse_factorization_observables",
            "path": str(design_dir / "observable_rows.csv"),
            "present": (design_dir / "observable_rows.csv").is_file(),
            "status": "read" if design_observables else "missing_or_empty",
            "decision": "",
            "row_count": len(design_observables),
        },
        {
            "source": "low_churn_mlp_residual_control_pilot",
            "path": str(low_churn_pilot_dir / "summary.json"),
            "present": (low_churn_pilot_dir / "summary.json").is_file(),
            "status": pilot_summary.get("status", ""),
            "decision": pilot_summary.get("decision", ""),
            "row_count": 1 if pilot_summary else 0,
        },
        {
            "source": "low_churn_mlp_arm_metrics",
            "path": str(low_churn_pilot_dir / "arm_metrics.csv"),
            "present": (low_churn_pilot_dir / "arm_metrics.csv").is_file(),
            "status": "read" if arm_rows else "missing_or_empty",
            "decision": "",
            "row_count": len(arm_rows),
        },
        {
            "source": "low_churn_mlp_per_token_metrics",
            "path": str(low_churn_pilot_dir / "per_token_metrics.csv"),
            "present": (low_churn_pilot_dir / "per_token_metrics.csv").is_file(),
            "status": "read" if token_rows else "missing_or_empty",
            "decision": "",
            "row_count": len(token_rows),
        },
    ]


def _teacher_residual_rows(token_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in token_rows:
        if row.get("arm") != LOW_CHURN_ARM:
            continue
        rows.append(
            {
                "teacher_arm": LOW_CHURN_ARM,
                "teacher_row_id": f"{LOW_CHURN_ARM}:{row.get('split', '')}:{row.get('token_index', '')}",
                "token_index": row.get("token_index", ""),
                "split": row.get("split", ""),
                "base_ce_loss": _float(row.get("base_ce_loss")),
                "teacher_ce_loss": _float(row.get("ce_loss")),
                "teacher_delta_vs_base_ce": _float(row.get("delta_vs_base_ce")),
                "teacher_residual_update_l2": _float(row.get("residual_update_l2")),
                "teacher_logit_mse_vs_base": _float(row.get("logit_mse_vs_base")),
                "teacher_anchor_kl_vs_base": _float(row.get("anchor_kl_vs_base")),
                "teacher_prediction_changed_vs_base": _boolish(row.get("prediction_changed_vs_base")),
                "raw_teacher_vector_available": False,
                "raw_intervention_available": _boolish(row.get("raw_intervention_available")),
                "extraction_role": "teacher_residual_proxy_row",
            }
        )
    return rows


def _teacher_budget_rows(arm_rows: list[dict[str, str]], pilot_summary: dict[str, Any]) -> list[dict[str, Any]]:
    by_arm = {row.get("arm", ""): row for row in arm_rows}
    teacher = by_arm.get(LOW_CHURN_ARM, {})
    shuffled = by_arm.get(SHUFFLED_NULL_ARM, {})
    budgets = pilot_summary.get("budgets", {})
    rows = [
        _budget_row("teacher_heldout_ce_loss", teacher.get("heldout_ce_loss"), "low-churn teacher quality reference"),
        _budget_row("teacher_heldout_residual_update_l2", teacher.get("heldout_residual_update_l2"), "teacher norm budget target"),
        _budget_row("teacher_heldout_anchor_kl_vs_base", teacher.get("heldout_anchor_kl_vs_base"), "teacher anchor drift reference"),
        _budget_row("teacher_heldout_prediction_flip_rate", teacher.get("heldout_prediction_flip_rate"), "teacher churn reference"),
        _budget_row("teacher_active_params", teacher.get("active_params"), "active parameter budget"),
        _budget_row("teacher_stored_params", teacher.get("stored_params"), "stored parameter budget"),
        _budget_row("shuffled_null_heldout_ce_loss", shuffled.get("heldout_ce_loss"), "misaligned teacher null quality"),
        _budget_row("dense24_residual_l2_ceiling", budgets.get("dense24_residual_l2_ceiling"), "dense24 norm ceiling from pilot"),
        _budget_row("dense24_flip_churn_ceiling", budgets.get("dense24_flip_churn_ceiling"), "dense24 churn ceiling from pilot"),
        _budget_row("dense24_anchor_logit_mse_ceiling", budgets.get("dense24_anchor_logit_mse_ceiling"), "dense24 anchor-drift ceiling from pilot"),
    ]
    return rows


def _budget_row(metric: str, value: Any, role: str) -> dict[str, Any]:
    return {"metric": metric, "value": _float(value), "role": role}


def _factorization_schema_rows(observable_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    metric_names = {row.get("metric", "") for row in observable_rows}
    required = [
        ("teacher_residual_reconstruction_mse", "quality", "required" in metric_names or "teacher_residual_reconstruction_mse" in metric_names),
        ("teacher_residual_reconstruction_r2", "quality", "teacher_residual_reconstruction_mse" in metric_names),
        ("teacher_gap_closure_fraction", "quality", "teacher_gap_closure_fraction" in metric_names),
        ("heldout_ce_transfer", "quality", "heldout_ce_transfer" in metric_names),
        ("oracle_support_regret", "support", "oracle_support_regret" in metric_names),
        ("support_entropy_and_load", "support", "support_entropy_and_load" in metric_names),
        ("functional_churn_kl_and_flip_rate", "interference", "functional_churn_kl_and_flip_rate" in metric_names),
        ("anchor_kl", "interference", "anchor_kl" in metric_names),
        ("finite_update_commutator", "interference", "finite_update_commutator" in metric_names),
        ("intervention_fingerprint_specificity", "causal", "intervention_fingerprint_specificity" in metric_names),
    ]
    return [
        {
            "field": field,
            "family": family,
            "required_for_training_harness": True,
            "present_in_design": bool(present),
            "source": "design_observable_rows",
        }
        for field, family, present in required
    ]


def _support_arm_schema_rows(design_arms: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in design_arms:
        rows.append(
            {
                "arm": row.get("arm", ""),
                "support_type": row.get("support_type", ""),
                "trainable": _boolish(row.get("trainable")),
                "budget_match": row.get("budget_match", ""),
                "required_splits": row.get("required_splits", ""),
                "role": row.get("role", ""),
                "training_rows_present": False,
                "ready_for_gpu_validation": False,
            }
        )
    return rows


def _gate_rows(
    *,
    design_summary: dict[str, Any],
    pilot_summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    teacher_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    schema_rows: list[dict[str, Any]],
    support_schema: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_ok = all(row["present"] and row["row_count"] > 0 for row in source_rows)
    required_budget_metrics = {
        "teacher_heldout_ce_loss",
        "teacher_heldout_residual_update_l2",
        "teacher_active_params",
        "teacher_stored_params",
        "shuffled_null_heldout_ce_loss",
    }
    budget_by_metric = {row["metric"]: row for row in budget_rows}
    missing_budgets = [
        metric for metric in sorted(required_budget_metrics)
        if budget_by_metric.get(metric, {}).get("value") is None
    ]
    support_arms = {row["arm"] for row in support_schema}
    required_arms = {
        "oracle_support_sparse_ceiling",
        "learned_router_sparse_factorization",
        "token_position_router_sparse_factorization",
        "frequency_support_router_sparse_factorization",
        "random_fixed_support_sparse_factorization",
        "route_scrambled_same_values",
        "shuffled_teacher_residual_sparse_factorization",
    }
    missing_schema_fields = [
        row["field"] for row in schema_rows
        if row["required_for_training_harness"] and not row["present_in_design"]
    ]
    return [
        _criterion(
            "design_selected_extractor",
            design_summary.get("status") == "pass"
            and design_summary.get("selected_next_action") == "implement_low_churn_mlp_sparse_factorization_ceiling_extractor",
            design_summary.get("selected_next_action"),
            "design artifact must select the extractor",
        ),
        _criterion(
            "low_churn_pilot_passed",
            pilot_summary.get("status") == "pass"
            and pilot_summary.get("decision") == "low_churn_mlp_residual_control_pilot_completed",
            pilot_summary.get("decision"),
            "low-churn pilot must be the teacher/control source",
        ),
        _criterion("required_sources_present", source_ok, source_rows, "all required source artifacts must be present and nonempty"),
        _criterion(
            "teacher_proxy_rows_available",
            len(teacher_rows) > 0 and any(row["split"] == "heldout" for row in teacher_rows),
            {"row_count": len(teacher_rows), "heldout_count": sum(1 for row in teacher_rows if row["split"] == "heldout")},
            "low-churn per-token teacher rows must include heldout rows",
        ),
        _criterion(
            "teacher_budget_rows_complete",
            not missing_budgets,
            missing_budgets,
            "teacher CE/norm/parameter and shuffled-null budget rows are required",
        ),
        _criterion(
            "support_arm_schema_complete",
            required_arms.issubset(support_arms),
            sorted(required_arms - support_arms),
            "oracle, learned, shortcut, random, scrambled, and shuffled-teacher support arms are required",
        ),
        _criterion(
            "factorization_schema_complete",
            not missing_schema_fields,
            missing_schema_fields,
            "quality, support, interference, and causal schema fields must be present in the design",
        ),
        _criterion(
            "read_only_no_training_no_gpu",
            True,
            {"training_executed": False, "advance_to_gpu_validation": False},
            "extractor must stay read-only and keep GPU blocked",
        ),
    ]


def _criterion(criterion: str, passed: bool, actual: Any, threshold: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "actual": actual,
        "threshold": threshold,
        "failure_reason": "" if passed else threshold,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_artifacts(
    *,
    out_dir: Path,
    summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    teacher_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    schema_rows: list[dict[str, Any]],
    support_schema: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", source_rows)
    _write_csv(out_dir / "teacher_residual_rows.csv", teacher_rows)
    _write_csv(out_dir / "teacher_budget_rows.csv", budget_rows)
    _write_csv(out_dir / "factorization_schema.csv", schema_rows)
    _write_csv(out_dir / "support_arm_schema.csv", support_schema)
    _write_csv(out_dir / "gate_criteria.csv", gate_rows)
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"], lineterminator="\n")
        writer.writeheader()
        for row in rows or [{"status": "missing"}]:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in writer.fieldnames or []})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Low-Churn MLP Sparse-Factorization Ceiling Extractor",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Selected action: `{summary['selected_next_action']}`",
            f"- Teacher residual proxy rows: `{summary['teacher_residual_row_count']}`",
            f"- Heldout teacher proxy rows: `{summary['heldout_teacher_residual_row_count']}`",
            f"- Training executed: `{summary['training_executed']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            "",
            "This artifact is a read-only bridge from the low-churn MLP teacher/control pilot to the sparse-factorization ceiling harness. It does not claim sparse factorization and keeps GPU validation blocked.",
            "",
        ]
    )


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN_DIR)
    parser.add_argument("--low-churn-pilot-dir", type=Path, default=DEFAULT_LOW_CHURN_PILOT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_low_churn_mlp_sparse_factorization_ceiling_extractor(
        design_dir=args.design_dir,
        low_churn_pilot_dir=args.low_churn_pilot_dir,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "selected_next_action": summary["selected_next_action"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
