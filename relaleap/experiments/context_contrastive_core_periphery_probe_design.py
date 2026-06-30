"""Design a fail-closed context-contrastive core/periphery probe."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_BRANCH_SELECTOR = Path("results/reports/post_low_churn_mlp_branch_selector/summary.json")
DEFAULT_CORE_CLOSEOUT = Path("results/reports/core_periphery_negative_evidence_closeout/summary.json")
DEFAULT_LOW_CHURN_PILOT = Path("results/reports/low_churn_mlp_residual_control_pilot/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/context_contrastive_core_periphery_probe_design")

SELECTED_NEXT_ACTION = "implement_context_contrastive_core_periphery_probe_locally"
REPAIR_ACTION = "repair_context_contrastive_core_periphery_design_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "mechanism_contract.csv",
    "control_matrix.csv",
    "gate_criteria.csv",
    "candidate_actions.csv",
    "notes.md",
)

REQUIRED_MECHANISM_FIELDS = (
    "context_contrastive_assignment",
    "protected_core_update_rule",
    "plastic_periphery_update_rule",
    "cross_context_consolidation",
    "periphery_specificity_penalty",
    "periphery_first_pruning",
    "causal_router_interface",
    "evaluation_leakage_guard",
)

REQUIRED_CONTROLS = (
    "current_demoted_core_periphery_mechanism",
    "promoted_sparse_contextual_router",
    "low_churn_mlp_residual_control",
    "dense_rank_norm_residual",
    "token_position_only_context_null",
    "shuffled_context_labels_null",
    "frequency_support_router_null",
    "no_core_ablation",
    "no_periphery_ablation",
    "equal_plasticity_ablation",
)

REQUIRED_GATES = (
    "source_selector_explicitly_selects_context_contrastive_design",
    "current_core_periphery_attempt_demoted",
    "low_churn_control_has_no_advancement_row",
    "mechanism_contract_complete",
    "control_matrix_complete",
    "local_only_before_gpu",
)


def run_context_contrastive_core_periphery_probe_design(
    *,
    branch_selector_path: Path = DEFAULT_BRANCH_SELECTOR,
    core_closeout_path: Path = DEFAULT_CORE_CLOSEOUT,
    low_churn_pilot_path: Path = DEFAULT_LOW_CHURN_PILOT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a local design contract for the next non-duplicative mechanism probe."""

    start = time.time()
    branch_selector = _read_json(branch_selector_path)
    core_closeout = _read_json(core_closeout_path)
    low_churn = _read_json(low_churn_pilot_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("post_low_churn_mlp_branch_selector", branch_selector_path, branch_selector),
        _source_row("core_periphery_negative_evidence_closeout", core_closeout_path, core_closeout),
        _source_row("low_churn_mlp_residual_control_pilot", low_churn_pilot_path, low_churn),
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
    mechanism_contract = _mechanism_contract()
    control_matrix = _control_matrix()
    gate_criteria = _gate_criteria(branch_selector, core_closeout, low_churn, mechanism_contract, control_matrix)
    failures = [row for row in gate_criteria if not row["passed"]]
    candidate_actions = _candidate_actions(failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "context_contrastive_core_periphery_probe_design_failed_closed"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair missing source, mechanism, control, or gate fields"
        claim_status = "context_contrastive_design_sources_or_contract_incomplete"
        rationale = "The design contract cannot safely open a new local branch until all fail-closed sources and controls are present."
    else:
        status = "pass"
        decision = "context_contrastive_core_periphery_probe_design_recorded"
        selected_row = selected[0]
        selected_next_action = selected_row["candidate_action"]
        selected_next_step = selected_row["next_step"]
        claim_status = selected_row["claim_status"]
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
        "backend_policy": "local design only; RunPod and Colab remain blocked until a local probe clears all gates",
        "source_rows": source_rows,
        "mechanism_contract": mechanism_contract,
        "control_matrix": control_matrix,
        "gate_criteria": gate_criteria,
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


def _mechanism_contract() -> list[dict[str, Any]]:
    return [
        {
            "field": "context_contrastive_assignment",
            "specification": (
                "learn core versus periphery roles from paired same-token contexts: core units are rewarded for "
                "cross-context reusable corrections, while periphery units are rewarded only for context-specific residual gains"
            ),
            "required": True,
        },
        {
            "field": "protected_core_update_rule",
            "specification": "lower learning rate, anchor retention penalty, and consolidation bonus for corrections that transfer across contexts",
            "required": True,
        },
        {
            "field": "plastic_periphery_update_rule",
            "specification": "higher learning rate plus sparsity and specificity pressure; no retention credit unless heldout causal utility is positive",
            "required": True,
        },
        {
            "field": "cross_context_consolidation",
            "specification": "promote repeated useful periphery corrections into protected core statistics only when off-target leakage and churn are nonworse",
            "required": True,
        },
        {
            "field": "periphery_specificity_penalty",
            "specification": "penalize peripheral residual energy on paired contexts where the same correction is not useful",
            "required": True,
        },
        {
            "field": "periphery_first_pruning",
            "specification": "rank periphery units by heldout necessity, sufficiency, leakage, and churn before pruning any protected core unit",
            "required": True,
        },
        {
            "field": "causal_router_interface",
            "specification": "use prefix-safe hidden/token/position/router-margin features only; full-context teacher signals are training labels, not deployable inputs",
            "required": True,
        },
        {
            "field": "evaluation_leakage_guard",
            "specification": "evaluation routing, gates, and pruning decisions cannot consume labels, future tokens, next_hidden, next_delta, or oracle support losses",
            "required": True,
        },
    ]


def _control_matrix() -> list[dict[str, Any]]:
    purposes = {
        "current_demoted_core_periphery_mechanism": "proves the new probe is not a rerun of the closed branch",
        "promoted_sparse_contextual_router": "current sparse support-routing baseline",
        "low_churn_mlp_residual_control": "dense/MLP matched-control guardrail with low churn",
        "dense_rank_norm_residual": "rank/norm-matched dense capacity comparator",
        "token_position_only_context_null": "tests whether context contrast is just token/position decoding",
        "shuffled_context_labels_null": "tests whether contrastive labels carry real paired-context structure",
        "frequency_support_router_null": "controls for support-frequency effects",
        "no_core_ablation": "tests whether protected core is necessary",
        "no_periphery_ablation": "tests whether plastic periphery is necessary",
        "equal_plasticity_ablation": "tests whether the core/periphery plasticity split matters",
    }
    return [
        {
            "control": name,
            "purpose": purposes[name],
            "matched_dimensions": "seed, train_tokens, active_params, active_compute, residual_l2_budget, support_width",
            "required": True,
        }
        for name in REQUIRED_CONTROLS
    ]


def _gate_criteria(
    branch_selector: dict[str, Any],
    core_closeout: dict[str, Any],
    low_churn: dict[str, Any],
    mechanism_contract: list[dict[str, Any]],
    control_matrix: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mechanism_fields = {str(row["field"]) for row in mechanism_contract if row.get("required") is True}
    controls = {str(row["control"]) for row in control_matrix if row.get("required") is True}
    return [
        _criterion(
            "source_selector_explicitly_selects_context_contrastive_design",
            branch_selector.get("status") == "pass"
            and branch_selector.get("selected_next_action") == "design_context_contrastive_core_periphery_probe_before_gpu",
            branch_selector.get("selected_next_action"),
            "post-low-churn selector must choose the context-contrastive design step",
        ),
        _criterion(
            "current_core_periphery_attempt_demoted",
            core_closeout.get("status") == "pass"
            and core_closeout.get("selected_next_action") == "demote_current_core_periphery_mechanism_to_diagnostic_status",
            core_closeout.get("selected_next_action"),
            "the prior core/periphery branch must be explicitly closed before opening a redesign",
        ),
        _criterion(
            "low_churn_control_has_no_advancement_row",
            low_churn.get("status") == "pass"
            and low_churn.get("advancement_row_count") == 0
            and low_churn.get("advance_to_gpu_validation") is False,
            {
                "advancement_row_count": low_churn.get("advancement_row_count"),
                "advance_to_gpu_validation": low_churn.get("advance_to_gpu_validation"),
            },
            "low-churn MLP control must not already require GPU validation",
        ),
        _criterion(
            "mechanism_contract_complete",
            set(REQUIRED_MECHANISM_FIELDS).issubset(mechanism_fields),
            sorted(mechanism_fields),
            "all required mechanism fields must be specified",
        ),
        _criterion(
            "control_matrix_complete",
            set(REQUIRED_CONTROLS).issubset(controls),
            sorted(controls),
            "all required controls and nulls must be specified",
        ),
        _criterion(
            "local_only_before_gpu",
            True,
            "requires_gpu_now=false; promotion_allowed=false; advance_to_gpu_validation=false",
            "this design may only select a local CPU probe implementation",
        ),
    ]


def _criterion(criterion: str, passed: bool, observed: Any, requirement: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "observed": observed,
        "requirement": requirement,
        "failure_reason": "" if passed else f"failed requirement: {requirement}",
    }


def _candidate_actions(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if failures:
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "one or more required sources or contract fields are missing",
                "repair the context-contrastive core/periphery design sources before implementing a probe",
                "context_contrastive_design_repair_required",
            ),
            _candidate(
                SELECTED_NEXT_ACTION,
                "blocked",
                "the local probe cannot start from an incomplete fail-closed design",
                "rerun after repair",
                "blocked_by_incomplete_design_contract",
            ),
        ]
    return [
        _candidate(
            SELECTED_NEXT_ACTION,
            "selected",
            (
                "the selector chose a new context-contrastive core/periphery design, the prior core/periphery "
                "branch is demoted, and the low-churn MLP control has no advancement row; implement a bounded "
                "local probe with strict null, dense/MLP, pruning, retention, and commutator gates"
            ),
            "implement the bounded local context-contrastive core/periphery probe; keep GPU validation blocked",
            "context_contrastive_probe_design_ready_no_gpu",
        ),
        _candidate(
            "run_runpod_validation",
            "rejected",
            "no local probe evidence exists yet",
            "wait until the local probe clears all gates",
            "gpu_validation_rejected_until_local_probe_passes",
        ),
        _candidate(
            "repeat_demoted_core_periphery_branch",
            "rejected",
            "the previous core/periphery branch is already demoted by local evidence",
            "do not duplicate the closed branch",
            "duplicate_closed_branch_rejected",
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
        "present": bool(payload),
        "status": payload.get("status", "missing") if payload else "missing",
        "decision": payload.get("decision", "") if payload else "",
        "claim_status": payload.get("claim_status", "") if payload else "",
        "selected_next_action": payload.get("selected_next_action", "") if payload else "",
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    notify = fields.get("notify_ben", "false").lower() == "true"
    level = fields.get("strategic_change_level", "missing")
    return {
        "present": bool(text),
        "strategic_change_level": level,
        "notify_ben": notify,
        "ben_notification_required": notify or level == "major",
        "recommended_next_action": fields.get("recommended_next_action", ""),
        "verdict": fields.get("verdict", ""),
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No external review was present; the local selector and fail-closed sources drive the design."
    return (
        "Read the latest external review. Its no-RunPod hidden-classifier recommendation is already satisfied "
        "by prior artifacts; this design accepts the no-GPU/fail-closed part and records the downstream local branch."
    )


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "mechanism_contract.csv", summary["mechanism_contract"])
    _write_csv(out_dir / "control_matrix.csv", summary["control_matrix"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Context-Contrastive Core/Periphery Probe Design",
            "",
            f"Status: `{summary['status']}`",
            f"Decision: `{summary['decision']}`",
            f"Claim status: `{summary['claim_status']}`",
            f"Selected next action: `{summary['selected_next_action']}`",
            "",
            "GPU validation remains blocked. This artifact is a local design contract, not training evidence.",
            "",
            "The probe must beat the promoted sparse router and dense/MLP controls while passing retention, "
            "periphery-first pruning, residual-norm, functional-churn, and finite-update commutator gates.",
            "",
            f"Strategy review handling: {summary['strategy_review_handling']}",
            "",
        ]
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-selector", type=Path, default=DEFAULT_BRANCH_SELECTOR)
    parser.add_argument("--core-closeout", type=Path, default=DEFAULT_CORE_CLOSEOUT)
    parser.add_argument("--low-churn-pilot", type=Path, default=DEFAULT_LOW_CHURN_PILOT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_context_contrastive_core_periphery_probe_design(
        branch_selector_path=args.branch_selector,
        core_closeout_path=args.core_closeout,
        low_churn_pilot_path=args.low_churn_pilot,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
