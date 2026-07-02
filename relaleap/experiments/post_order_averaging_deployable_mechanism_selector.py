"""Select the next deployable mechanism after order-averaging closeout."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_ORDER_CLOSEOUT = Path("results/reports/token_larger_promoted_topk2_order_averaging_closeout/summary.json")
DEFAULT_VALUE_PENALTY = Path("results/audits/token_larger_promoted_topk2_commutator_value_penalty_probe/summary.json")
DEFAULT_FLAT_CLOSEOUT = Path("results/reports/same_router_flat_value_commutator_mitigation_closeout/summary.json")
DEFAULT_MULTISITE_CLOSEOUT = Path("results/reports/multisite_continual_pc_core_periphery_closeout/summary.json")
DEFAULT_MECHANISM_INVENTORY = Path("results/reports/mechanism_source_inventory/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/post_order_averaging_deployable_mechanism_selector")

SELECTED_ACTION = "design_deployable_commutator_regularized_sparse_update_pregate"
REPAIR_ACTION = "repair_post_order_averaging_selector_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "decision_matrix.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_post_order_averaging_deployable_mechanism_selector(
    *,
    order_closeout_path: Path = DEFAULT_ORDER_CLOSEOUT,
    value_penalty_path: Path = DEFAULT_VALUE_PENALTY,
    flat_closeout_path: Path = DEFAULT_FLAT_CLOSEOUT,
    multisite_closeout_path: Path = DEFAULT_MULTISITE_CLOSEOUT,
    mechanism_inventory_path: Path = DEFAULT_MECHANISM_INVENTORY,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Pick one non-duplicative local mechanism branch and keep GPU blocked."""

    start = time.time()
    order_closeout = _read_json(order_closeout_path)
    value_penalty = _read_json(value_penalty_path)
    flat_closeout = _read_json(flat_closeout_path)
    multisite_closeout = _read_json(multisite_closeout_path)
    mechanism_inventory = _read_json(mechanism_inventory_path)
    strategy = _strategy_review(strategy_review_path)

    source_rows = [
        _source_row("order_averaging_closeout", order_closeout_path, order_closeout),
        _source_row("commutator_value_penalty_probe", value_penalty_path, value_penalty),
        _source_row("flat_value_commutator_closeout", flat_closeout_path, flat_closeout),
        _source_row("multisite_pc_core_periphery_closeout", multisite_closeout_path, multisite_closeout),
        _source_row("mechanism_source_inventory", mechanism_inventory_path, mechanism_inventory),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}; verdict={strategy['verdict']}"
            ),
            "selected_next_action": "",
        },
    ]
    evidence = _evidence(
        order_closeout,
        value_penalty,
        flat_closeout,
        multisite_closeout,
        mechanism_inventory,
        strategy,
    )
    failures = _failures(source_rows, evidence)
    decision_matrix = _decision_matrix(evidence)
    candidate_actions = _candidate_actions(failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "post_order_averaging_selector_failed_closed"
        claim_status = "post_order_averaging_selector_sources_incomplete"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair missing or inconsistent selector source artifacts before starting a branch"
        rationale = "Required selector inputs are missing or inconsistent, so no new mechanism branch is selected."
    else:
        selected_row = selected[0]
        status = "pass"
        decision = "post_order_averaging_deployable_mechanism_selected"
        claim_status = selected_row["claim_status"]
        selected_next_action = selected_row["candidate_action"]
        selected_next_step = selected_row["next_step"]
        rationale = selected_row["reason"]

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_action": selected_next_action,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local selector only; RunPod and Colab remain blocked",
        "source_rows": source_rows,
        "evidence": evidence,
        "decision_matrix": decision_matrix,
        "candidate_actions": candidate_actions,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "failures": failures,
        "rationale": rationale,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _evidence(
    order_closeout: dict[str, Any],
    value_penalty: dict[str, Any],
    flat_closeout: dict[str, Any],
    multisite_closeout: dict[str, Any],
    mechanism_inventory: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    order_evidence = order_closeout.get("evidence", {})
    value_metrics = value_penalty.get("metrics", {})
    return {
        "order_closeout_status": order_closeout.get("status"),
        "order_closeout_decision": order_closeout.get("decision"),
        "order_closeout_selected_next_action": order_closeout.get("selected_next_action"),
        "order_closeout_requires_gpu_now": order_closeout.get("requires_gpu_now"),
        "order_closeout_advance_to_gpu_validation": order_closeout.get("advance_to_gpu_validation"),
        "order_averaging_ratio": order_evidence.get("order_average_ratio"),
        "flat_value_order_averaging_control_present": order_evidence.get("flat_value_order_averaging_control_present"),
        "value_penalty_decision": value_penalty.get("decision"),
        "value_penalty_best_reduction_fraction": value_metrics.get("best_penalty_reduction_fraction"),
        "flat_value_closeout_decision": flat_closeout.get("decision"),
        "flat_value_selected_next_action": flat_closeout.get("selected_next_action"),
        "multisite_decision": multisite_closeout.get("decision"),
        "multisite_selected_next_action": multisite_closeout.get("selected_next_action"),
        "mechanism_inventory_selected_next_action": mechanism_inventory.get("selected_next_action"),
        "mechanism_inventory_claim_status": mechanism_inventory.get("claim_status"),
        "strategy_recommended_next_action": strategy["recommended_next_action"],
        "strategy_verdict": strategy["verdict"],
        "ben_notification_required": strategy["ben_notification_required"],
    }


def _failures(source_rows: list[dict[str, Any]], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows[:5]:
        if not row["present"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "source_artifact",
                    "expected": "file exists",
                    "actual": "missing",
                    "path": row["path"],
                }
            )
    expected = {
        "order_closeout_status": "pass",
        "order_closeout_decision": "promoted_topk2_order_averaging_closed_no_gpu",
        "order_closeout_selected_next_action": "close_order_averaging_before_gpu",
        "order_closeout_requires_gpu_now": False,
        "order_closeout_advance_to_gpu_validation": False,
        "value_penalty_decision": "commutator_value_penalty_not_established",
    }
    for field, expected_value in expected.items():
        if evidence.get(field) != expected_value:
            failures.append(
                {
                    "source": "evidence",
                    "field": field,
                    "expected": expected_value,
                    "actual": evidence.get(field),
                }
            )
    return failures


def _decision_matrix(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "criterion": "order_averaging_headroom_but_nondeployable",
            "passed": True,
            "evidence": (
                f"ratio={evidence.get('order_averaging_ratio')}; "
                f"flat_control_present={evidence.get('flat_value_order_averaging_control_present')}"
            ),
            "interpretation": "finite-update commutator headroom exists, but explicit order averaging cannot be promoted.",
        },
        {
            "criterion": "simple_value_penalty_already_failed",
            "passed": evidence.get("value_penalty_decision") == "commutator_value_penalty_not_established",
            "evidence": (
                f"decision={evidence.get('value_penalty_decision')}; "
                f"best_reduction={evidence.get('value_penalty_best_reduction_fraction')}"
            ),
            "interpretation": "do not duplicate the simple value-side penalty branch.",
        },
        {
            "criterion": "flat_value_capacity_not_sparse_specific",
            "passed": "flat_value" in str(evidence.get("flat_value_closeout_decision")),
            "evidence": (
                f"decision={evidence.get('flat_value_closeout_decision')}; "
                f"action={evidence.get('flat_value_selected_next_action')}"
            ),
            "interpretation": "generic flat-value capacity is a control/null, not the next sparse mechanism claim.",
        },
        {
            "criterion": "pc_core_periphery_retune_blocked",
            "passed": evidence.get("multisite_decision") == "multisite_pc_core_periphery_branch_closed",
            "evidence": (
                f"decision={evidence.get('multisite_decision')}; "
                f"action={evidence.get('multisite_selected_next_action')}"
            ),
            "interpretation": "do not retune the failed multi-site PC/core-periphery branch as the immediate next step.",
        },
        {
            "criterion": "gpu_validation_blocked",
            "passed": True,
            "evidence": "requires_gpu_now=false; promotion_allowed=false; advance_to_gpu_validation=false",
            "interpretation": "the next branch must be a local CPU pregate/design before any backend validation.",
        },
    ]


def _candidate_actions(failures: list[dict[str, Any]]) -> list[dict[str, str]]:
    if failures:
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "required selector sources are missing or inconsistent",
                "repair selector sources before choosing another branch",
                "source_repair_required",
            )
        ]
    return [
        _candidate(
            SELECTED_ACTION,
            "selected",
            (
                "explicit order averaging identifies finite-update commutator headroom but is nondeployable; "
                "the least duplicative next local branch is a deployable commutator-regularized sparse update "
                "pregate with dense/flat/random-support/no-update controls"
            ),
            (
                "design a local deployable commutator-regularized sparse update pregate with CE, residual-norm, "
                "parameter/logit commutator, forgetting, support-overlap, dense/flat/random-support/no-update, "
                "and nondeployable order-averaging upper-bound controls"
            ),
            "deployable_commutator_regularized_sparse_update_pregate_selected_no_gpu",
        ),
        _candidate(
            "rerun_explicit_order_averaging_probe",
            "rejected",
            "that probe and closeout already exist and classify the branch as nondeployable diagnostic only",
            "do not duplicate completed order-averaging work",
            "duplicate_order_averaging_rejected",
        ),
        _candidate(
            "retune_commutator_value_penalty",
            "rejected",
            "simple value penalties already failed the commutator gate and do not address update-rule deployability",
            "do not retune value penalties without a new selector",
            "value_penalty_retune_rejected",
        ),
        _candidate(
            "reopen_multisite_pc_core_periphery",
            "rejected",
            "the trained local multi-site branch failed dense/MLP, churn, retention, commutator, and null gates",
            "do not retune PC/core-periphery before a materially different selector",
            "pc_core_periphery_reopen_rejected",
        ),
        _candidate(
            "run_runpod_or_colab_validation",
            "rejected",
            "all consumed artifacts block GPU validation and promotion",
            "keep RunPod and Colab unused for this selector",
            "gpu_validation_blocked",
        ),
    ]


def _candidate(
    candidate_action: str,
    disposition: str,
    reason: str,
    next_step: str,
    claim_status: str,
) -> dict[str, str]:
    return {
        "candidate_action": candidate_action,
        "disposition": disposition,
        "reason": reason,
        "next_step": next_step,
        "claim_status": claim_status,
    }


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status") if payload else "missing",
        "decision": payload.get("decision") if payload else "",
        "claim_status": payload.get("claim_status") if payload else "",
        "selected_next_action": payload.get("selected_next_action") or payload.get("next_step") or "",
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _strategy_review(path: Path) -> dict[str, Any]:
    fields = {
        "strategic_change_level": "missing",
        "notify_ben": False,
        "recommended_next_action": "",
        "verdict": "missing",
    }
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key in fields:
                fields[key] = value
    notify = str(fields["notify_ben"]).lower() == "true"
    strategic_change = str(fields["strategic_change_level"])
    return {
        "path": str(path),
        "present": path.is_file(),
        "strategic_change_level": strategic_change,
        "notify_ben": notify,
        "ben_notification_required": notify or strategic_change == "major",
        "recommended_next_action": str(fields["recommended_next_action"]),
        "verdict": str(fields["verdict"]),
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if strategy["ben_notification_required"]:
        return (
            "Read the external review and preserved its notification requirement. "
            "This selector records that Ben should be notified before treating the direction shift as settled."
        )
    return (
        "Read the external review. The order-averaging recommendation was already completed and closed; "
        "this selector follows its scientifically sensible commutator-mechanism direction while avoiding duplicate work."
    )


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "summary.json", summary)
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "decision_matrix.csv", summary["decision_matrix"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Post-Order-Averaging Deployable Mechanism Selector",
            "",
            f"- status: {summary['status']}",
            f"- decision: {summary['decision']}",
            f"- selected_next_action: {summary['selected_next_action']}",
            f"- selected_next_step: {summary['selected_next_step']}",
            "- GPU validation remains blocked: requires_gpu_now=false, promotion_allowed=false, advance_to_gpu_validation=false.",
            f"- rationale: {summary['rationale']}",
            f"- strategy_review_handling: {summary['strategy_review_handling']}",
            "",
        ]
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_post_order_averaging_deployable_mechanism_selector(out_dir=args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
