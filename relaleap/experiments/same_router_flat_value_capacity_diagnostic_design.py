"""Design the same-router flat-value-capacity diagnostic after sparse closeout."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_CLOSEOUT = Path("results/reports/learned_router_sparse_value_closeout/summary.json")
DEFAULT_SYNTHETIC_DIR = Path("results/reports/synthetic_mechanism_causal_modularity")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/same_router_flat_value_capacity_diagnostic_design")

SELECTED_ACTION = "implement_same_router_flat_value_capacity_diagnostic_locally"
REPAIR_ACTION = "repair_same_router_flat_value_capacity_design_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "diagnostic_design.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_same_router_flat_value_capacity_diagnostic_design(
    *,
    closeout_path: Path = DEFAULT_CLOSEOUT,
    synthetic_dir: Path = DEFAULT_SYNTHETIC_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Record a fail-closed local diagnostic design without launching GPU work."""

    start = time.time()
    closeout = _read_json(closeout_path)
    arm_metrics_path = synthetic_dir / "arm_metrics.csv"
    value_capacity_path = synthetic_dir / "value_capacity_core_periphery_diagnostic.csv"
    commutator_path = synthetic_dir / "commutator_rows.csv"
    forgetting_path = synthetic_dir / "forgetting_rows.csv"
    budget_path = synthetic_dir / "residual_budget_accounting.csv"
    arm_metrics = _read_csv(arm_metrics_path)
    value_capacity_rows = _read_csv(value_capacity_path)
    commutator_rows = _read_csv(commutator_path)
    forgetting_rows = _read_csv(forgetting_path)
    budget_rows = _read_csv(budget_path)
    strategy = _strategy_review(strategy_review_path)

    source_rows = [
        _source_json("learned_router_sparse_value_closeout", closeout_path, closeout),
        _source_csv("synthetic_arm_metrics", arm_metrics_path, arm_metrics),
        _source_csv("synthetic_value_capacity_core_periphery_diagnostic", value_capacity_path, value_capacity_rows),
        _source_csv("synthetic_commutator_rows", commutator_path, commutator_rows),
        _source_csv("synthetic_forgetting_rows", forgetting_path, forgetting_rows),
        _source_csv("synthetic_residual_budget_accounting", budget_path, budget_rows),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}"
            ),
            "row_count": "",
        },
    ]
    evidence = _evidence(closeout, arm_metrics, value_capacity_rows, commutator_rows, forgetting_rows)
    design_rows = _design_rows(evidence)
    gate_rows = _gate_rows(closeout, evidence, source_rows)
    failures = [row for row in gate_rows if not row["passed"]]
    status = "pass" if not failures else "fail"
    selected_action = SELECTED_ACTION if status == "pass" else REPAIR_ACTION
    selected_next_step = (
        "implement the local same-router flat-value-capacity diagnostic with null, dense/MLP, budget, and oracle-regret gates"
        if status == "pass"
        else "repair same-router flat-value-capacity diagnostic design source artifacts"
    )
    summary = {
        "status": status,
        "decision": (
            "same_router_flat_value_capacity_diagnostic_design_recorded"
            if status == "pass"
            else "same_router_flat_value_capacity_diagnostic_design_failed_closed"
        ),
        "claim_status": (
            "design_only_flat_value_capacity_diagnostic_not_yet_evidence"
            if status == "pass"
            else "source_artifacts_incomplete_or_closeout_not_selected"
        ),
        "selected_next_action": selected_action,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "source_rows": source_rows,
        "evidence": evidence,
        "diagnostic_design": design_rows,
        "gate_criteria": gate_rows,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "failures": failures,
        "rationale": (
            "The sparse learned-router branch closed because the same-router flat value control was stronger. "
            "The next local step should isolate whether that is generic value capacity, support-policy failure, "
            "or an interference/budget tradeoff, before any GPU validation."
        ),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _evidence(
    closeout: dict[str, Any],
    arm_metrics: list[dict[str, str]],
    value_capacity_rows: list[dict[str, str]],
    commutator_rows: list[dict[str, str]],
    forgetting_rows: list[dict[str, str]],
) -> dict[str, Any]:
    by_arm = {row.get("arm", ""): row for row in arm_metrics}
    sparse = by_arm.get("promoted_contextual_topk2", {})
    flat = by_arm.get("flat_column_value_mlp_topk2") or by_arm.get("flat_column_value_mlp_anchor_topk2") or {}
    token = by_arm.get("token_position_router_topk2", {})
    dense = by_arm.get("dense_rank_norm_matched", {})
    low_churn = by_arm.get("low_churn_mlp_active_matched", {})
    oracle_rows = [row for row in value_capacity_rows if row.get("branch") == "stored_value_capacity_upper_bound"]
    active_rows = [row for row in value_capacity_rows if row.get("branch") == "active_value_capacity_control"]
    return {
        "closeout_selected_next_action": closeout.get("selected_next_action"),
        "closeout_claim_status": closeout.get("claim_status"),
        "closeout_flat_control_ce_gain": _dig(closeout, "evidence", "flat_control_ce_gain"),
        "sparse_holdout_ce": _float_or_none(sparse.get("holdout_ce")),
        "flat_holdout_ce": _float_or_none(flat.get("holdout_ce")),
        "token_position_holdout_ce": _float_or_none(token.get("holdout_ce")),
        "dense_holdout_ce": _float_or_none(dense.get("holdout_ce")),
        "low_churn_holdout_ce": _float_or_none(low_churn.get("holdout_ce")),
        "flat_minus_sparse_ce": _delta(_float_or_none(flat.get("holdout_ce")), _float_or_none(sparse.get("holdout_ce"))),
        "token_position_minus_flat_ce": _delta(_float_or_none(token.get("holdout_ce")), _float_or_none(flat.get("holdout_ce"))),
        "flat_residual_l2": _float_or_none(flat.get("residual_l2")),
        "sparse_residual_l2": _float_or_none(sparse.get("residual_l2")),
        "flat_mean_commutator_l2": _mean_metric(commutator_rows, flat.get("arm", ""), "finite_update_commutator_l2"),
        "sparse_mean_commutator_l2": _mean_metric(commutator_rows, "promoted_contextual_topk2", "finite_update_commutator_l2"),
        "flat_mean_abs_functional_churn": _mean_abs_metric(forgetting_rows, flat.get("arm", ""), "functional_churn"),
        "sparse_mean_abs_functional_churn": _mean_abs_metric(forgetting_rows, "promoted_contextual_topk2", "functional_churn"),
        "value_capacity_active_rows": len(active_rows),
        "value_capacity_oracle_rows": len(oracle_rows),
        "flat_arm": flat.get("arm", ""),
    }


def _design_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _design(
            "same_router_flat_value_primary",
            evidence["flat_arm"] or "flat_column_value_mlp_topk2",
            "Test whether same support policy plus higher value capacity explains sparse branch failure.",
            "holdout_ce_vs_promoted_contextual_topk2",
            evidence["flat_minus_sparse_ce"],
            "must beat sparse by >=0.005 CE while satisfying nonworse norm/churn/commutator budgets",
        ),
        _design(
            "support_policy_nulls",
            "token_position_router_topk2;random_support_topk2;fixed_support_topk2",
            "Separate value capacity from support-policy quality.",
            "flat_ce_gain_vs_nulls",
            evidence["token_position_minus_flat_ce"],
            "flat value capacity must beat token/position and support nulls, or close as value-only capacity",
        ),
        _design(
            "dense_and_mlp_controls",
            "dense_rank_norm_matched;low_churn_mlp_active_matched",
            "Check whether flat value capacity is merely a dense/MLP capacity effect.",
            "flat_ce_vs_dense_and_low_churn_controls",
            f"dense={evidence['dense_holdout_ce']}; low_churn={evidence['low_churn_holdout_ce']}",
            "do not promote if dense/rank/norm or low-churn MLP controls are stronger at comparable budget",
        ),
        _design(
            "interference_budgets",
            "residual_norm;functional_churn;finite_update_commutator",
            "Prevent CE-only interpretation of value capacity.",
            "flat_budget_vs_sparse_and_token_position",
            (
                f"flat_norm={evidence['flat_residual_l2']}; flat_churn={evidence['flat_mean_abs_functional_churn']}; "
                f"flat_commutator={evidence['flat_mean_commutator_l2']}"
            ),
            "flat value diagnostic must emit residual norm, churn, and commutator rows and fail closed when worse",
        ),
        _design(
            "oracle_regret_strata",
            "value_capacity_core_periphery_diagnostic",
            "Connect value-capacity wins to oracle-support headroom instead of average CE alone.",
            "oracle_gap_recovery_by_rule_position",
            (
                f"active_rows={evidence['value_capacity_active_rows']}; "
                f"oracle_rows={evidence['value_capacity_oracle_rows']}"
            ),
            "diagnostic must report oracle-regret recovery by rule/position and support-frequency strata",
        ),
    ]


def _design(role: str, controls: str, question: str, metric: str, current_value: Any, gate: str) -> dict[str, Any]:
    return {
        "diagnostic_role": role,
        "required_controls": controls,
        "scientific_question": question,
        "primary_metric": metric,
        "current_source_value": current_value,
        "gate_requirement": gate,
        "requires_gpu_now": False,
        "promotion_allowed": False,
    }


def _gate_rows(
    closeout: dict[str, Any],
    evidence: dict[str, Any],
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _criterion(
            "closeout_selected_flat_value_diagnostic",
            closeout.get("selected_next_action") == "design_same_router_flat_value_capacity_diagnostic",
            closeout.get("selected_next_action"),
            "learned-router sparse-value closeout did not select the flat-value diagnostic",
        ),
        _criterion(
            "required_sources_present",
            all(row["present"] for row in source_rows if row["source"] != "strategy_review"),
            {row["source"]: row["present"] for row in source_rows},
            "one or more required source artifacts are missing",
        ),
        _criterion(
            "flat_control_stronger_source_signal_present",
            evidence["flat_minus_sparse_ce"] is not None and evidence["flat_minus_sparse_ce"] < 0.0,
            evidence["flat_minus_sparse_ce"],
            "source artifacts do not show the flat value control beating sparse",
        ),
        _criterion(
            "value_capacity_rows_available",
            evidence["value_capacity_active_rows"] > 0 and evidence["value_capacity_oracle_rows"] > 0,
            {
                "active_rows": evidence["value_capacity_active_rows"],
                "oracle_rows": evidence["value_capacity_oracle_rows"],
            },
            "value-capacity diagnostic rows are missing",
        ),
        _criterion(
            "gpu_blocked_by_design",
            True,
            "requires_gpu_now=False; advance_to_gpu_validation=False",
            "design must not request GPU validation",
        ),
    ]


def _criterion(criterion: str, passed: bool, actual: Any, failure_reason: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "actual": actual,
        "failure_reason": "" if passed else failure_reason,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _source_json(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(payload),
        "status": payload.get("status", "missing"),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
        "row_count": "",
    }


def _source_csv(source: str, path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(rows),
        "status": "present" if rows else "missing",
        "decision": "",
        "claim_status": "",
        "row_count": len(rows),
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "present": False,
            "strategic_change_level": "minor",
            "notify_ben": False,
            "recommended_next_action": "",
            "verdict": "",
        }
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:12]:
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    return {
        "present": True,
        "strategic_change_level": header.get("strategic_change_level", "minor"),
        "notify_ben": header.get("notify_ben", "false").lower() == "true",
        "recommended_next_action": header.get("recommended_next_action", ""),
        "verdict": header.get("verdict", ""),
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No external strategy review was present; proceeded from local closeout artifacts."
    return (
        "Read the latest external review. Its hidden-classifier fail-closed recommendation remains satisfied; "
        "this design follows the downstream local flat-value diagnostic selected by closeout artifacts."
    )


def _dig(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _mean_metric(rows: list[dict[str, str]], arm: str, key: str) -> float | None:
    values = [_float_or_none(row.get(key)) for row in rows if row.get("arm") == arm]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _mean_abs_metric(rows: list[dict[str, str]], arm: str, key: str) -> float | None:
    values = [_float_or_none(row.get(key)) for row in rows if row.get("arm") == arm]
    values = [abs(value) for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "diagnostic_design.csv", summary["diagnostic_design"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    notes = [
        "# Same-Router Flat-Value-Capacity Diagnostic Design",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Advance to GPU validation: `{summary['advance_to_gpu_validation']}`",
        f"- Flat minus sparse CE: `{summary['evidence']['flat_minus_sparse_ce']}`",
        f"- Strategy review handling: {summary['strategy_review_handling']}",
        "",
        (
            "This is a design-only local contract. GPU validation remains blocked until an implemented diagnostic "
            "separates same-router value capacity from support-policy quality and passes null, dense/MLP, "
            "residual-norm, functional-churn, commutator, and oracle-regret gates."
        ),
    ]
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closeout", type=Path, default=DEFAULT_CLOSEOUT)
    parser.add_argument("--synthetic-dir", type=Path, default=DEFAULT_SYNTHETIC_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_same_router_flat_value_capacity_diagnostic_design(
        closeout_path=args.closeout,
        synthetic_dir=args.synthetic_dir,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
