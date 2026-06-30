"""Pregate the learned-router non-PC sparse-value branch after hidden-head closeout."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_BRANCH_SELECTOR = Path("results/reports/hidden_support_classifier_branch_selector/summary.json")
DEFAULT_SYNTHETIC_DIR = Path("results/reports/synthetic_mechanism_causal_modularity")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/learned_router_sparse_value_pregate")

RETURN_LEARNED_ROUTER_ACTION = "return_to_learned_router_non_pc_sparse_value_branch"
REPAIR_SOURCES_ACTION = "repair_learned_router_sparse_value_pregate_sources"
NEXT_CLOSEOUT_ACTION = "close_or_redirect_learned_router_sparse_value_branch_locally"
REPEAT_LOCAL_ACTION = "repeat_learned_router_sparse_value_pregate_on_adjacent_seed"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "pregate_rows.csv",
    "notes.md",
)


def run_learned_router_sparse_value_pregate(
    *,
    branch_selector_path: Path = DEFAULT_BRANCH_SELECTOR,
    synthetic_dir: Path = DEFAULT_SYNTHETIC_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Consume existing local artifacts and fail-close the learned-router sparse branch."""

    start = time.time()
    branch_selector = _read_json(branch_selector_path)
    arm_metrics_path = synthetic_dir / "arm_metrics.csv"
    residual_budget_path = synthetic_dir / "residual_budget_accounting.csv"
    commutator_path = synthetic_dir / "commutator_rows.csv"
    forgetting_path = synthetic_dir / "forgetting_rows.csv"
    arm_metrics = _read_csv(arm_metrics_path)
    residual_budget = _read_csv(residual_budget_path)
    commutator_rows = _read_csv(commutator_path)
    forgetting_rows = _read_csv(forgetting_path)
    strategy = _strategy_review(strategy_review_path)

    source_rows = [
        _source_json("hidden_support_classifier_branch_selector", branch_selector_path, branch_selector),
        _source_csv("synthetic_arm_metrics", arm_metrics_path, arm_metrics),
        _source_csv("synthetic_residual_budget_accounting", residual_budget_path, residual_budget),
        _source_csv("synthetic_commutator_rows", commutator_path, commutator_rows),
        _source_csv("synthetic_forgetting_rows", forgetting_path, forgetting_rows),
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
        },
    ]
    failures = _source_failures(source_rows)
    pregate_rows = _pregate_rows(
        branch_selector=branch_selector,
        arm_metrics=arm_metrics,
        residual_budget=residual_budget,
        commutator_rows=commutator_rows,
        forgetting_rows=forgetting_rows,
        source_failures=failures,
    )
    primary = next((row for row in pregate_rows if row.get("selected") is True), {})
    pregate_passes = primary.get("pregate_passes") is True
    selected_next_action = (
        REPAIR_SOURCES_ACTION
        if failures
        else REPEAT_LOCAL_ACTION
        if pregate_passes
        else NEXT_CLOSEOUT_ACTION
    )
    selected_next_step = (
        "repair learned-router sparse-value pregate source artifacts"
        if failures
        else "repeat learned-router non-PC sparse-value pregate on an adjacent seed before any GPU validation"
        if pregate_passes
        else "close or redirect the learned-router non-PC sparse-value branch locally before any GPU validation"
    )
    status = "fail" if failures else "pass"
    decision = (
        "learned_router_sparse_value_pregate_failed_closed"
        if failures
        else "learned_router_sparse_value_pregate_passed_repeat_before_gpu"
        if pregate_passes
        else "learned_router_sparse_value_pregate_local_gpu_blocked"
    )
    claim_status = (
        "source_artifacts_incomplete"
        if failures
        else "learned_router_sparse_value_has_local_signal_needs_repeat"
        if pregate_passes
        else "learned_router_sparse_value_blocked_by_null_or_interference_controls"
    )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_action": selected_next_action,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "source_rows": source_rows,
        "pregate_row_count": len(pregate_rows),
        "pregate_primary_result": _primary_summary(primary),
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "failures": failures,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, source_rows, pregate_rows)
    return summary


def _pregate_rows(
    *,
    branch_selector: dict[str, Any],
    arm_metrics: list[dict[str, str]],
    residual_budget: list[dict[str, str]],
    commutator_rows: list[dict[str, str]],
    forgetting_rows: list[dict[str, str]],
    source_failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if source_failures:
        return [
            {
                "pregate_name": "learned_router_sparse_value_pregate",
                "pregate_role": "source_repair",
                "arm": "",
                "selected": True,
                "implemented_in_current_packet": False,
                "source_branch_selector_action": branch_selector.get("selected_next_action", ""),
                "pregate_passes": False,
                "failure_reasons": "source_artifacts_incomplete",
                "requires_gpu_now": False,
                "promotion_allowed": False,
                "advance_to_gpu_validation": False,
                "selected_next_experiment": REPAIR_SOURCES_ACTION,
            }
        ]

    by_arm = {row.get("arm", ""): row for row in arm_metrics}
    budget_by_arm = {row.get("arm", ""): row for row in residual_budget}
    primary = by_arm.get("promoted_contextual_topk2")
    token_position = by_arm.get("token_position_router_topk2")
    random_support = by_arm.get("random_support_topk2")
    fixed_support = by_arm.get("fixed_support_topk2")
    flat_value = by_arm.get("flat_column_value_mlp_topk2") or by_arm.get("flat_column_value_mlp_anchor_topk2")
    dense = by_arm.get("dense_rank_norm_matched")
    low_churn = by_arm.get("low_churn_mlp_active_matched")
    intervention_sparse = by_arm.get("intervention_trained_sparse_topk2")
    stored_upper = _best_ce_row(
        [
            row
            for row in arm_metrics
            if row.get("control_budget_role") == "stored_parameter_matched_dense_mlp_upper_bound"
        ]
    )
    if primary is None:
        return [
            {
                "pregate_name": "learned_router_sparse_value_pregate",
                "pregate_role": "source_repair",
                "arm": "promoted_contextual_topk2",
                "selected": True,
                "implemented_in_current_packet": False,
                "source_branch_selector_action": branch_selector.get("selected_next_action", ""),
                "pregate_passes": False,
                "failure_reasons": "missing_promoted_contextual_topk2_arm",
                "requires_gpu_now": False,
                "promotion_allowed": False,
                "advance_to_gpu_validation": False,
                "selected_next_experiment": REPAIR_SOURCES_ACTION,
            }
        ]

    primary_ce = _float_or_none(primary.get("holdout_ce"))
    token_ce = _float_or_none(token_position.get("holdout_ce")) if token_position else None
    random_ce = _float_or_none(random_support.get("holdout_ce")) if random_support else None
    fixed_ce = _float_or_none(fixed_support.get("holdout_ce")) if fixed_support else None
    flat_ce = _float_or_none(flat_value.get("holdout_ce")) if flat_value else None
    dense_ce = _float_or_none(dense.get("holdout_ce")) if dense else None
    low_churn_ce = _float_or_none(low_churn.get("holdout_ce")) if low_churn else None
    intervention_ce = _float_or_none(intervention_sparse.get("holdout_ce")) if intervention_sparse else None
    stored_ce = _float_or_none(stored_upper.get("holdout_ce")) if stored_upper else None

    primary_commutator = _mean_metric(commutator_rows, "promoted_contextual_topk2", "finite_update_commutator_l2")
    token_commutator = _mean_metric(commutator_rows, "token_position_router_topk2", "finite_update_commutator_l2")
    dense_commutator = _mean_metric(commutator_rows, "dense_rank_norm_matched", "finite_update_commutator_l2")
    primary_churn = _mean_abs_metric(forgetting_rows, "promoted_contextual_topk2", "functional_churn")
    token_churn = _mean_abs_metric(forgetting_rows, "token_position_router_topk2", "functional_churn")
    dense_churn = _mean_abs_metric(forgetting_rows, "dense_rank_norm_matched", "functional_churn")
    primary_norm = _float_or_none(primary.get("residual_l2"))
    token_norm = _float_or_none(token_position.get("residual_l2")) if token_position else None
    dense_norm = _float_or_none(dense.get("residual_l2")) if dense else None

    branch_selected = branch_selector.get("selected_next_action") == RETURN_LEARNED_ROUTER_ACTION
    token_null_ok = _gain(token_ce, primary_ce) is not None and _gain(token_ce, primary_ce) >= 0.005
    random_null_ok = _gain(random_ce, primary_ce) is not None and _gain(random_ce, primary_ce) >= 0.01
    fixed_null_ok = _gain(fixed_ce, primary_ce) is not None and _gain(fixed_ce, primary_ce) >= 0.005
    intervention_sparse_ok = _gain(intervention_ce, primary_ce) is not None and _gain(intervention_ce, primary_ce) >= -0.005
    flat_control_ok = _gain(flat_ce, primary_ce) is None or _gain(flat_ce, primary_ce) >= -0.005
    dense_control_ok = _gain(dense_ce, primary_ce) is None or _gain(dense_ce, primary_ce) >= -0.005
    low_churn_control_ok = _gain(low_churn_ce, primary_ce) is None or _gain(low_churn_ce, primary_ce) >= -0.005
    stored_upper_bound_blocks_promotion = bool(
        stored_ce is not None and primary_ce is not None and stored_ce + 0.05 < primary_ce
    )
    norm_budget_ok = bool(
        primary_norm is not None
        and token_norm is not None
        and primary_norm <= token_norm * 1.05
    )
    commutator_budget_ok = bool(
        primary_commutator is not None
        and token_commutator is not None
        and primary_commutator <= token_commutator * 1.10
        and (dense_commutator is None or primary_commutator <= max(dense_commutator * 1.10, token_commutator * 1.10))
    )
    churn_budget_ok = bool(
        primary_churn is not None
        and token_churn is not None
        and primary_churn <= token_churn * 1.10
        and (dense_churn is None or primary_churn <= max(dense_churn * 1.10, token_churn * 1.10))
    )
    budget_available = primary_commutator is not None and primary_churn is not None and primary_norm is not None
    pregate_passes = bool(
        branch_selected
        and token_null_ok
        and random_null_ok
        and fixed_null_ok
        and intervention_sparse_ok
        and flat_control_ok
        and dense_control_ok
        and low_churn_control_ok
        and norm_budget_ok
        and commutator_budget_ok
        and churn_budget_ok
        and not stored_upper_bound_blocks_promotion
    )
    failures = []
    if not branch_selected:
        failures.append("hidden_branch_selector_did_not_select_learned_router_sparse_value")
    if not token_null_ok:
        failures.append("token_position_null_too_close_or_stronger")
    if not random_null_ok:
        failures.append("random_support_null_not_cleared")
    if not fixed_null_ok:
        failures.append("fixed_support_null_not_cleared")
    if not intervention_sparse_ok:
        failures.append("intervention_trained_sparse_reference_stronger")
    if not flat_control_ok:
        failures.append("same_router_flat_value_control_stronger")
    if not dense_control_ok:
        failures.append("dense_rank_norm_control_stronger")
    if not low_churn_control_ok:
        failures.append("low_churn_mlp_control_stronger")
    if not norm_budget_ok:
        failures.append("residual_norm_budget_failed")
    if not commutator_budget_ok:
        failures.append("finite_update_commutator_budget_failed")
    if not churn_budget_ok:
        failures.append("functional_churn_budget_failed")
    if stored_upper_bound_blocks_promotion:
        failures.append("stored_parameter_dense_upper_bound_still_blocks_promotion")

    common = {
        "pregate_name": "learned_router_sparse_value_pregate",
        "source_branch_selector_action": branch_selector.get("selected_next_action", ""),
        "primary_arm": "promoted_contextual_topk2",
        "primary_holdout_ce": primary_ce,
        "primary_residual_l2": primary_norm,
        "primary_mean_commutator_l2": primary_commutator,
        "primary_mean_abs_functional_churn": primary_churn,
        "primary_flop_proxy_per_token": _float_or_none(
            budget_by_arm.get("promoted_contextual_topk2", {}).get("flop_proxy_per_token")
        ),
        "token_position_ce_gain": _gain(token_ce, primary_ce),
        "random_support_ce_gain": _gain(random_ce, primary_ce),
        "fixed_support_ce_gain": _gain(fixed_ce, primary_ce),
        "intervention_sparse_ce_gain": _gain(intervention_ce, primary_ce),
        "flat_control_ce_gain": _gain(flat_ce, primary_ce),
        "dense_rank_norm_ce_gain": _gain(dense_ce, primary_ce),
        "low_churn_mlp_ce_gain": _gain(low_churn_ce, primary_ce),
        "stored_upper_bound_ce_gain": _gain(stored_ce, primary_ce),
        "budget_evidence_available": budget_available,
        "branch_selector_ok": branch_selected,
        "token_position_null_ok": token_null_ok,
        "random_support_null_ok": random_null_ok,
        "fixed_support_null_ok": fixed_null_ok,
        "intervention_sparse_reference_ok": intervention_sparse_ok,
        "flat_control_ok": flat_control_ok,
        "dense_control_ok": dense_control_ok,
        "low_churn_control_ok": low_churn_control_ok,
        "stored_upper_bound_blocks_promotion": stored_upper_bound_blocks_promotion,
        "norm_budget_ok": norm_budget_ok,
        "commutator_budget_ok": commutator_budget_ok,
        "functional_churn_budget_ok": churn_budget_ok,
        "pregate_passes": pregate_passes,
        "failure_reasons": ";".join(failures),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "selected_next_experiment": REPEAT_LOCAL_ACTION if pregate_passes else NEXT_CLOSEOUT_ACTION,
        "interpretation": (
            "Local learned-router/non-PC sparse-value pregate after hidden-classifier closeout. "
            "The branch must beat token/position, random, fixed-support, flat-value, dense, and low-churn controls "
            "while preserving norm, functional-churn, and finite-update commutator budgets before any GPU validation."
        ),
    }
    controls = (
        ("primary_learned_router_sparse_value", "promoted_contextual_topk2", "primary", True),
        ("causal_token_position_null", "token_position_router_topk2", "router_null", False),
        ("random_support_null", "random_support_topk2", "support_null", False),
        ("fixed_support_null", "fixed_support_topk2", "support_null", False),
        ("intervention_trained_sparse_reference", "intervention_trained_sparse_topk2", "sparse_reference", False),
        ("same_router_flat_value_control", flat_value.get("arm", "") if flat_value else "flat_column_value_mlp_topk2", "flat_value_control", False),
        ("dense_rank_norm_control", "dense_rank_norm_matched", "dense_control", False),
        ("low_churn_mlp_control", "low_churn_mlp_active_matched", "mlp_control", False),
        ("stored_parameter_dense_upper_bound", stored_upper.get("arm", "") if stored_upper else "", "stored_upper_bound", False),
    )
    rows = []
    for role, arm, family, selected in controls:
        source = by_arm.get(arm, {})
        rows.append(
            {
                **common,
                "pregate_role": role,
                "arm": arm,
                "control_family": family,
                "selected": selected,
                "implemented_in_current_packet": bool(source),
                "source_holdout_ce": _float_or_none(source.get("holdout_ce")),
                "source_ce_minus_primary_ce": _gain(primary_ce, _float_or_none(source.get("holdout_ce"))),
                "source_residual_l2": _float_or_none(source.get("residual_l2")),
                "source_mean_commutator_l2": _mean_metric(commutator_rows, arm, "finite_update_commutator_l2"),
                "source_mean_abs_functional_churn": _mean_abs_metric(forgetting_rows, arm, "functional_churn"),
                "source_flop_proxy_per_token": _float_or_none(budget_by_arm.get(arm, {}).get("flop_proxy_per_token")),
            }
        )
    return rows


def _primary_summary(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {
            "row_count": 0,
            "pregate_passes": False,
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "advance_to_gpu_validation": False,
        }
    return {
        "primary_arm": row.get("primary_arm", ""),
        "primary_holdout_ce": row.get("primary_holdout_ce"),
        "token_position_ce_gain": row.get("token_position_ce_gain"),
        "random_support_ce_gain": row.get("random_support_ce_gain"),
        "fixed_support_ce_gain": row.get("fixed_support_ce_gain"),
        "flat_control_ce_gain": row.get("flat_control_ce_gain"),
        "dense_rank_norm_ce_gain": row.get("dense_rank_norm_ce_gain"),
        "low_churn_mlp_ce_gain": row.get("low_churn_mlp_ce_gain"),
        "stored_upper_bound_ce_gain": row.get("stored_upper_bound_ce_gain"),
        "branch_selector_ok": row.get("branch_selector_ok") is True,
        "token_position_null_ok": row.get("token_position_null_ok") is True,
        "random_support_null_ok": row.get("random_support_null_ok") is True,
        "fixed_support_null_ok": row.get("fixed_support_null_ok") is True,
        "flat_control_ok": row.get("flat_control_ok") is True,
        "dense_control_ok": row.get("dense_control_ok") is True,
        "low_churn_control_ok": row.get("low_churn_control_ok") is True,
        "stored_upper_bound_blocks_promotion": row.get("stored_upper_bound_blocks_promotion") is True,
        "norm_budget_ok": row.get("norm_budget_ok") is True,
        "commutator_budget_ok": row.get("commutator_budget_ok") is True,
        "functional_churn_budget_ok": row.get("functional_churn_budget_ok") is True,
        "pregate_passes": row.get("pregate_passes") is True,
        "failure_reasons": row.get("failure_reasons", ""),
        "selected_next_experiment": row.get("selected_next_experiment", ""),
        "requires_gpu_now": row.get("requires_gpu_now") is True,
        "promotion_allowed": row.get("promotion_allowed") is True,
        "advance_to_gpu_validation": row.get("advance_to_gpu_validation") is True,
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


def _source_failures(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = {
        "hidden_support_classifier_branch_selector",
        "synthetic_arm_metrics",
        "synthetic_residual_budget_accounting",
        "synthetic_commutator_rows",
        "synthetic_forgetting_rows",
    }
    return [
        {"source": row["source"], "path": row["path"], "reason": "missing_required_source"}
        for row in source_rows
        if row["source"] in required and not row["present"]
    ]


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gain(reference_ce: float | None, candidate_ce: float | None) -> float | None:
    if reference_ce is None or candidate_ce is None:
        return None
    return reference_ce - candidate_ce


def _best_ce_row(rows: list[dict[str, str]]) -> dict[str, str]:
    present = [row for row in rows if _float_or_none(row.get("holdout_ce")) is not None]
    if not present:
        return {}
    return min(present, key=lambda row: _float_or_none(row.get("holdout_ce")) or float("inf"))


def _mean_metric(rows: list[dict[str, str]], arm: str, key: str) -> float | None:
    values = [_float_or_none(row.get(key)) for row in rows if row.get("arm") == arm]
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def _mean_abs_metric(rows: list[dict[str, str]], arm: str, key: str) -> float | None:
    values = [_float_or_none(row.get(key)) for row in rows if row.get("arm") == arm]
    present = [abs(value) for value in values if value is not None]
    return mean(present) if present else None


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
        return "No external strategy review was present; proceeded with local fail-closed source artifacts."
    return (
        "Read the latest external review. Its no-RunPod hidden-classifier gate/report recommendation was already "
        "implemented in the prior run, so this run follows the recorded branch-selector return to learned-router "
        "non-PC sparse-value work while keeping GPU validation blocked."
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    pregate_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", source_rows)
    _write_csv(out_dir / "pregate_rows.csv", pregate_rows)
    primary = summary["pregate_primary_result"]
    notes = [
        "# Learned-Router Sparse-Value Pregate",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Primary arm: `{primary.get('primary_arm', '')}`",
        f"- Token/position CE gain: `{primary.get('token_position_ce_gain')}`",
        f"- Random-support CE gain: `{primary.get('random_support_ce_gain')}`",
        f"- Fixed-support CE gain: `{primary.get('fixed_support_ce_gain')}`",
        f"- Flat-control CE gain: `{primary.get('flat_control_ce_gain')}`",
        f"- Dense-control CE gain: `{primary.get('dense_rank_norm_ce_gain')}`",
        f"- Low-churn MLP CE gain: `{primary.get('low_churn_mlp_ce_gain')}`",
        f"- Pregate passes: `{primary.get('pregate_passes')}`",
        f"- Failure reasons: `{primary.get('failure_reasons', '')}`",
        "",
        "RunPod and promotion remain blocked unless this local branch clears null/control and interference gates.",
    ]
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-selector", type=Path, default=DEFAULT_BRANCH_SELECTOR)
    parser.add_argument("--synthetic-dir", type=Path, default=DEFAULT_SYNTHETIC_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_learned_router_sparse_value_pregate(
        branch_selector_path=args.branch_selector,
        synthetic_dir=args.synthetic_dir,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
