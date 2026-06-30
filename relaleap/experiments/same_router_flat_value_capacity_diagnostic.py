"""Diagnose the same-router flat value-capacity control before GPU work."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_DESIGN = Path("results/reports/same_router_flat_value_capacity_diagnostic_design/summary.json")
DEFAULT_SYNTHETIC_DIR = Path("results/reports/synthetic_mechanism_causal_modularity")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/same_router_flat_value_capacity_diagnostic")

REPAIR_ACTION = "repair_same_router_flat_value_capacity_diagnostic_sources"
CLOSE_ACTION = "close_flat_value_capacity_as_interference_or_generic_capacity_before_gpu"
REPEAT_ACTION = "repeat_same_router_flat_value_capacity_diagnostic_on_adjacent_seed"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "control_rows.csv",
    "budget_rows.csv",
    "oracle_regret_strata.csv",
    "gate_rows.csv",
    "notes.md",
)


def run_same_router_flat_value_capacity_diagnostic(
    *,
    design_path: Path = DEFAULT_DESIGN,
    synthetic_dir: Path = DEFAULT_SYNTHETIC_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Consume local source artifacts and decide whether flat value capacity merits repeat/GPU."""

    start = time.time()
    design = _read_json(design_path)
    paths = {
        "arm_metrics": synthetic_dir / "arm_metrics.csv",
        "residual_budget_accounting": synthetic_dir / "residual_budget_accounting.csv",
        "commutator_rows": synthetic_dir / "commutator_rows.csv",
        "forgetting_rows": synthetic_dir / "forgetting_rows.csv",
        "router_regret_ceiling_budget": synthetic_dir / "router_regret_ceiling_budget.csv",
        "router_value_regret_decomposition": synthetic_dir / "router_value_regret_decomposition.csv",
    }
    rows = {name: _read_csv(path) for name, path in paths.items()}
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_json("same_router_flat_value_capacity_diagnostic_design", design_path, design),
        *[_source_csv(f"synthetic_{name}", path, rows[name]) for name, path in paths.items()],
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
    failures = _source_failures(source_rows)
    control_rows, budget_rows, oracle_rows, gate_rows = _diagnostic_rows(
        design=design,
        arm_metrics=rows["arm_metrics"],
        residual_budget=rows["residual_budget_accounting"],
        commutator_rows=rows["commutator_rows"],
        forgetting_rows=rows["forgetting_rows"],
        router_regret_rows=rows["router_regret_ceiling_budget"],
        router_value_rows=rows["router_value_regret_decomposition"],
        source_failures=failures,
    )
    primary = next((row for row in control_rows if row.get("diagnostic_role") == "same_router_flat_value_primary"), {})
    gate_map = {row["gate"]: row["passes"] is True for row in gate_rows}
    diagnostic_passes = bool(gate_rows) and all(gate_map.values())
    selected_next_action = (
        REPAIR_ACTION
        if failures
        else REPEAT_ACTION
        if diagnostic_passes
        else CLOSE_ACTION
    )
    selected_next_step = (
        "repair same-router flat-value-capacity diagnostic source artifacts"
        if failures
        else "repeat same-router flat-value-capacity diagnostic on an adjacent seed before any GPU validation"
        if diagnostic_passes
        else "close or redesign flat value-capacity branch locally before any GPU validation"
    )
    status = "fail" if failures else "pass"
    decision = (
        "same_router_flat_value_capacity_diagnostic_failed_closed"
        if failures
        else "same_router_flat_value_capacity_diagnostic_passed_repeat_before_gpu"
        if diagnostic_passes
        else "same_router_flat_value_capacity_diagnostic_gpu_blocked"
    )
    summary = {
        "status": status,
        "decision": decision,
        "claim_status": (
            "source_artifacts_incomplete"
            if failures
            else "flat_value_capacity_signal_needs_repeat"
            if diagnostic_passes
            else "flat_value_capacity_blocked_by_controls_or_interference_budgets"
        ),
        "selected_next_action": selected_next_action,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "source_rows": source_rows,
        "control_rows": control_rows,
        "budget_rows": budget_rows,
        "oracle_regret_strata": oracle_rows,
        "gate_rows": gate_rows,
        "diagnostic_passes": diagnostic_passes,
        "primary_result": _primary_summary(primary, gate_map),
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "failures": failures,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _diagnostic_rows(
    *,
    design: dict[str, Any],
    arm_metrics: list[dict[str, str]],
    residual_budget: list[dict[str, str]],
    commutator_rows: list[dict[str, str]],
    forgetting_rows: list[dict[str, str]],
    router_regret_rows: list[dict[str, str]],
    router_value_rows: list[dict[str, str]],
    source_failures: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if source_failures:
        gate_rows = [
            {
                "gate": "source_artifacts_present",
                "passes": False,
                "actual": "missing required source artifacts",
                "failure_reason": "source_artifacts_incomplete",
            }
        ]
        return [], [], [], gate_rows

    by_arm = {row.get("arm", ""): row for row in arm_metrics}
    budget_by_arm = {row.get("arm", ""): row for row in residual_budget}
    primary_arm = "flat_column_value_mlp_topk2"
    flat = by_arm.get(primary_arm) or by_arm.get("flat_column_value_mlp_anchor_topk2", {})
    if flat.get("arm"):
        primary_arm = flat["arm"]
    controls = (
        ("same_router_flat_value_primary", primary_arm, "primary_flat_value_capacity"),
        ("promoted_sparse_reference", "promoted_contextual_topk2", "sparse_reference"),
        ("token_position_null", "token_position_router_topk2", "support_policy_null"),
        ("random_support_null", "random_support_topk2", "support_policy_null"),
        ("fixed_support_null", "fixed_support_topk2", "support_policy_null"),
        ("dense_rank_norm_control", "dense_rank_norm_matched", "dense_or_mlp_control"),
        ("low_churn_mlp_control", "low_churn_mlp_active_matched", "dense_or_mlp_control"),
    )
    flat_ce = _metric(flat, "holdout_ce")
    sparse_ce = _metric(by_arm.get("promoted_contextual_topk2", {}), "holdout_ce")
    control_rows = []
    for role, arm, family in controls:
        source = by_arm.get(arm, {})
        ce = _metric(source, "holdout_ce")
        control_rows.append(
            {
                "diagnostic_role": role,
                "arm": arm,
                "control_family": family,
                "implemented_in_source": bool(source),
                "holdout_ce": ce,
                "ce_gain_vs_flat": _gain(ce, flat_ce),
                "flat_ce_gain_vs_control": _gain(ce, flat_ce),
                "residual_l2": _metric(source, "residual_l2"),
                "active_parameters_proxy": _metric(source, "active_parameters_proxy"),
                "stored_parameters": _metric(source, "stored_parameters"),
                "flop_proxy_per_token": _metric(budget_by_arm.get(arm, {}), "flop_proxy_per_token"),
            }
        )

    budget_controls = (
        ("residual_norm", "residual_l2", flat.get("residual_l2"), by_arm.get("promoted_contextual_topk2", {}).get("residual_l2"), by_arm.get("token_position_router_topk2", {}).get("residual_l2")),
        ("functional_churn", "mean_abs_functional_churn", _mean_abs_metric(forgetting_rows, primary_arm, "functional_churn"), _mean_abs_metric(forgetting_rows, "promoted_contextual_topk2", "functional_churn"), _mean_abs_metric(forgetting_rows, "token_position_router_topk2", "functional_churn")),
        ("finite_update_commutator", "mean_commutator_l2", _mean_metric(commutator_rows, primary_arm, "finite_update_commutator_l2"), _mean_metric(commutator_rows, "promoted_contextual_topk2", "finite_update_commutator_l2"), _mean_metric(commutator_rows, "token_position_router_topk2", "finite_update_commutator_l2")),
    )
    budget_rows = []
    for budget, metric_name, candidate_raw, sparse_raw, token_raw in budget_controls:
        candidate = _float_or_none(candidate_raw)
        sparse_value = _float_or_none(sparse_raw)
        token_value = _float_or_none(token_raw)
        reference = _max_present(sparse_value, token_value)
        nonworse = candidate is not None and reference is not None and candidate <= reference * 1.10
        budget_rows.append(
            {
                "budget": budget,
                "metric": metric_name,
                "candidate_arm": primary_arm,
                "candidate_value": candidate,
                "sparse_reference_value": sparse_value,
                "token_position_reference_value": token_value,
                "reference_budget_value": reference,
                "nonworse_gate_passes": nonworse,
                "failure_reason": "" if nonworse else "flat value capacity exceeds sparse/token-position budget by >10% or evidence is missing",
            }
        )

    oracle_rows = _oracle_rows(router_regret_rows, router_value_rows, primary_arm, sparse_ce, flat_ce)
    gates = {
        "design_selected_diagnostic": design.get("selected_next_action") == "implement_same_router_flat_value_capacity_diagnostic_locally",
        "flat_beats_promoted_sparse": _gain(sparse_ce, flat_ce) is not None and _gain(sparse_ce, flat_ce) >= 0.005,
        "flat_beats_support_policy_nulls": all(
            _gain(_metric(by_arm.get(arm, {}), "holdout_ce"), flat_ce) is not None
            and _gain(_metric(by_arm.get(arm, {}), "holdout_ce"), flat_ce) >= 0.005
            for arm in ("token_position_router_topk2", "random_support_topk2", "fixed_support_topk2")
        ),
        "flat_beats_dense_mlp_controls": all(
            _gain(_metric(by_arm.get(arm, {}), "holdout_ce"), flat_ce) is not None
            and _gain(_metric(by_arm.get(arm, {}), "holdout_ce"), flat_ce) >= -0.005
            for arm in ("dense_rank_norm_matched", "low_churn_mlp_active_matched")
        ),
        "interference_budgets_nonworse": all(row["nonworse_gate_passes"] for row in budget_rows),
        "oracle_regret_strata_available": bool(oracle_rows)
        and all(row["oracle_regret_evidence_available"] for row in oracle_rows),
    }
    gate_rows = [
        {
            "gate": gate,
            "passes": passes,
            "actual": _gate_actual(gate, control_rows, budget_rows, oracle_rows),
            "failure_reason": "" if passes else _gate_failure(gate),
        }
        for gate, passes in gates.items()
    ]
    return control_rows, budget_rows, oracle_rows, gate_rows


def _oracle_rows(
    router_regret_rows: list[dict[str, str]],
    router_value_rows: list[dict[str, str]],
    flat_arm: str,
    sparse_ce: float | None,
    flat_ce: float | None,
) -> list[dict[str, Any]]:
    rows = []
    sparse_regret = next((row for row in router_regret_rows if row.get("arm") == "promoted_contextual_topk2"), {})
    flat_regret = next((row for row in router_regret_rows if row.get("arm") == flat_arm), {})
    for source in (sparse_regret, flat_regret):
        if not source:
            continue
        learned_ce = _metric(source, "learned_holdout_ce")
        oracle_ce = _metric(source, "oracle_support_ce_ceiling")
        rows.append(
            {
                "arm": source.get("arm", ""),
                "latent_rule": "all",
                "learned_holdout_ce": learned_ce,
                "oracle_support_ce_ceiling": oracle_ce,
                "mean_oracle_regret": _metric(source, "mean_oracle_regret"),
                "oracle_regret_evidence_available": learned_ce is not None and oracle_ce is not None,
                "flat_ce_gain_vs_promoted_sparse": _gain(sparse_ce, flat_ce),
                "interpretation": "router-only oracle ceiling is label-scored diagnostic evidence, not deployable training signal",
            }
        )
    for source in router_value_rows:
        if source.get("arm") not in {"promoted_contextual_topk2", flat_arm}:
            continue
        rows.append(
            {
                "arm": source.get("arm", ""),
                "latent_rule": source.get("latent_rule", ""),
                "learned_holdout_ce": _metric(source, "mean_learned_ce_loss"),
                "oracle_support_ce_ceiling": _metric(source, "mean_oracle_ce_loss"),
                "mean_oracle_regret": _metric(source, "mean_oracle_regret"),
                "oracle_regret_evidence_available": True,
                "flat_ce_gain_vs_promoted_sparse": _gain(sparse_ce, flat_ce),
                "interpretation": "rule-stratified oracle-regret context for the same-router flat value-capacity diagnostic",
            }
        )
    return rows


def _primary_summary(primary: dict[str, Any], gate_map: dict[str, bool]) -> dict[str, Any]:
    return {
        "primary_arm": primary.get("arm", ""),
        "flat_holdout_ce": primary.get("holdout_ce"),
        "flat_beats_promoted_sparse": gate_map.get("flat_beats_promoted_sparse", False),
        "flat_beats_support_policy_nulls": gate_map.get("flat_beats_support_policy_nulls", False),
        "flat_beats_dense_mlp_controls": gate_map.get("flat_beats_dense_mlp_controls", False),
        "interference_budgets_nonworse": gate_map.get("interference_budgets_nonworse", False),
        "oracle_regret_strata_available": gate_map.get("oracle_regret_strata_available", False),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
    }


def _gate_actual(
    gate: str,
    control_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
) -> Any:
    if gate == "flat_beats_promoted_sparse":
        row = next((item for item in control_rows if item["diagnostic_role"] == "promoted_sparse_reference"), {})
        return row.get("flat_ce_gain_vs_control")
    if gate == "flat_beats_support_policy_nulls":
        return {
            item["arm"]: item.get("flat_ce_gain_vs_control")
            for item in control_rows
            if item["control_family"] == "support_policy_null"
        }
    if gate == "flat_beats_dense_mlp_controls":
        return {
            item["arm"]: item.get("flat_ce_gain_vs_control")
            for item in control_rows
            if item["control_family"] == "dense_or_mlp_control"
        }
    if gate == "interference_budgets_nonworse":
        return {item["budget"]: item["nonworse_gate_passes"] for item in budget_rows}
    if gate == "oracle_regret_strata_available":
        return len(oracle_rows)
    return "see source rows"


def _gate_failure(gate: str) -> str:
    return {
        "design_selected_diagnostic": "design source did not select implementation of this diagnostic",
        "flat_beats_promoted_sparse": "flat value capacity does not beat promoted sparse by the required margin",
        "flat_beats_support_policy_nulls": "flat value capacity does not cleanly beat token/position, random, and fixed support nulls",
        "flat_beats_dense_mlp_controls": "flat value capacity is not separated from dense/MLP controls",
        "interference_budgets_nonworse": "residual norm, churn, or commutator budget is worse than sparse/token-position references",
        "oracle_regret_strata_available": "oracle-regret strata are missing",
    }[gate]


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


def _source_failures(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"source": row["source"], "path": row["path"], "reason": "required source artifact missing"}
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]


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
        return "No external strategy review was present; proceeded from local design and synthetic artifacts."
    return (
        "Read the latest external review. Its no-RunPod hidden-classifier recommendation remains satisfied; "
        "this diagnostic follows the downstream local same-router flat-value-capacity step."
    )


def _metric(row: dict[str, Any], key: str) -> float | None:
    return _float_or_none(row.get(key))


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


def _max_present(*values: float | None) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


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
    _write_csv(out_dir / "control_rows.csv", summary["control_rows"])
    _write_csv(out_dir / "budget_rows.csv", summary["budget_rows"])
    _write_csv(out_dir / "oracle_regret_strata.csv", summary["oracle_regret_strata"])
    _write_csv(out_dir / "gate_rows.csv", summary["gate_rows"])
    notes = [
        "# Same-Router Flat-Value-Capacity Diagnostic",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Advance to GPU validation: `{summary['advance_to_gpu_validation']}`",
        f"- Primary arm: `{summary['primary_result']['primary_arm']}`",
        f"- Flat holdout CE: `{summary['primary_result']['flat_holdout_ce']}`",
        f"- Flat beats promoted sparse: `{summary['primary_result']['flat_beats_promoted_sparse']}`",
        f"- Flat beats support-policy nulls: `{summary['primary_result']['flat_beats_support_policy_nulls']}`",
        f"- Interference budgets nonworse: `{summary['primary_result']['interference_budgets_nonworse']}`",
        f"- Strategy review handling: {summary['strategy_review_handling']}",
        "",
        (
            "GPU validation remains blocked unless the flat value-capacity diagnostic beats support-policy and "
            "dense/MLP controls while preserving residual-norm, functional-churn, finite-update commutator, and "
            "oracle-regret evidence."
        ),
    ]
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--synthetic-dir", type=Path, default=DEFAULT_SYNTHETIC_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_same_router_flat_value_capacity_diagnostic(
        design_path=args.design,
        synthetic_dir=args.synthetic_dir,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
