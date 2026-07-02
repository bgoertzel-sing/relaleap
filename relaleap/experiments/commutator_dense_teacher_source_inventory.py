"""Inventory commutator and dense-teacher sources before another local branch."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_BRANCH_SELECTOR = Path("results/reports/mechanism_factorized_cl_branch_selector/summary.json")
DEFAULT_DENSE_MLP_SYNTHESIS = Path("results/reports/dense_mlp_control_synthesis/summary.json")
DEFAULT_DENSE_TEACHER_DISTILLATION_CLOSEOUT = Path(
    "results/reports/dense_teacher_residual_distillation_closeout/summary.json"
)
DEFAULT_PAIR_COMPOSER_CLOSEOUT = Path("results/reports/dense_teacher_pair_composer_closeout/summary.json")
DEFAULT_FLAT_VALUE_COMMUTATOR_CLOSEOUT = Path(
    "results/reports/same_router_flat_value_commutator_mitigation_closeout/summary.json"
)
DEFAULT_TOPK2_POST_FINITE_UPDATE_CLOSEOUT = Path(
    "results/reports/token_larger_promoted_topk2_post_finite_update_closeout/summary.json"
)
DEFAULT_TOPK2_MITIGATION_SELECTOR = Path(
    "results/reports/token_larger_promoted_topk2_mitigation_branch_selector/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/commutator_dense_teacher_source_inventory")

REPAIR_ACTION = "repair_commutator_dense_teacher_source_inventory_sources"
ORDER_AVERAGING_ACTION = "run_explicit_order_averaging_mitigation_probe_locally"
GPU_ACTION = "launch_gpu_validation_for_commutator_or_dense_teacher_branch"
DENSE_TEACHER_ACTION = "reopen_dense_teacher_residual_distillation_or_pair_composer"
NEW_TRAINING_ACTION = "start_new_sparse_or_dense_teacher_training_branch"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "inventory_rows.csv",
    "duplicate_work_rows.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_commutator_dense_teacher_source_inventory(
    *,
    branch_selector_path: Path = DEFAULT_BRANCH_SELECTOR,
    dense_mlp_synthesis_path: Path = DEFAULT_DENSE_MLP_SYNTHESIS,
    dense_teacher_distillation_closeout_path: Path = DEFAULT_DENSE_TEACHER_DISTILLATION_CLOSEOUT,
    pair_composer_closeout_path: Path = DEFAULT_PAIR_COMPOSER_CLOSEOUT,
    flat_value_commutator_closeout_path: Path = DEFAULT_FLAT_VALUE_COMMUTATOR_CLOSEOUT,
    topk2_post_finite_update_closeout_path: Path = DEFAULT_TOPK2_POST_FINITE_UPDATE_CLOSEOUT,
    topk2_mitigation_selector_path: Path = DEFAULT_TOPK2_MITIGATION_SELECTOR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed inventory and pick one non-GPU next step."""

    start = time.time()
    sources = {
        "mechanism_factorized_cl_branch_selector": _read_json(branch_selector_path),
        "dense_mlp_control_synthesis": _read_json(dense_mlp_synthesis_path),
        "dense_teacher_residual_distillation_closeout": _read_json(dense_teacher_distillation_closeout_path),
        "dense_teacher_pair_composer_closeout": _read_json(pair_composer_closeout_path),
        "same_router_flat_value_commutator_mitigation_closeout": _read_json(
            flat_value_commutator_closeout_path
        ),
        "token_larger_promoted_topk2_post_finite_update_closeout": _read_json(
            topk2_post_finite_update_closeout_path
        ),
        "token_larger_promoted_topk2_mitigation_branch_selector": _read_json(
            topk2_mitigation_selector_path
        ),
    }
    paths = {
        "mechanism_factorized_cl_branch_selector": branch_selector_path,
        "dense_mlp_control_synthesis": dense_mlp_synthesis_path,
        "dense_teacher_residual_distillation_closeout": dense_teacher_distillation_closeout_path,
        "dense_teacher_pair_composer_closeout": pair_composer_closeout_path,
        "same_router_flat_value_commutator_mitigation_closeout": flat_value_commutator_closeout_path,
        "token_larger_promoted_topk2_post_finite_update_closeout": topk2_post_finite_update_closeout_path,
        "token_larger_promoted_topk2_mitigation_branch_selector": topk2_mitigation_selector_path,
    }
    strategy = _strategy_review(strategy_review_path)
    source_rows = [_source_row(name, paths[name], payload) for name, payload in sources.items()]
    source_rows.append(
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
            "selected_next_step": "",
        }
    )
    failures = _source_failures(source_rows, sources)
    inventory_rows = _inventory_rows(sources, strategy)
    duplicate_rows = _duplicate_work_rows(sources)
    candidate_actions = _candidate_actions(failures, inventory_rows, duplicate_rows)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "commutator_dense_teacher_source_inventory_failed_closed"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair missing or inconsistent commutator/dense-teacher source reports"
        claim_status = "commutator_dense_teacher_inventory_sources_incomplete"
        rationale = "Required source reports are missing or do not match the current handoff, so no branch is selected."
    else:
        selected_row = selected[0]
        status = "pass"
        decision = "commutator_dense_teacher_source_inventory_recorded"
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
        "training_executed": False,
        "backend_policy": "local source inventory only; RunPod and Colab remain blocked",
        "source_rows": source_rows,
        "inventory_rows": inventory_rows,
        "duplicate_work_rows": duplicate_rows,
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


def _inventory_rows(sources: dict[str, dict[str, Any]], strategy: dict[str, Any]) -> list[dict[str, Any]]:
    dense_synthesis = sources["dense_mlp_control_synthesis"]
    distill = sources["dense_teacher_residual_distillation_closeout"]
    pair = sources["dense_teacher_pair_composer_closeout"]
    flat_comm = sources["same_router_flat_value_commutator_mitigation_closeout"]
    post_finite = sources["token_larger_promoted_topk2_post_finite_update_closeout"]
    mitigation_selector = sources["token_larger_promoted_topk2_mitigation_branch_selector"]
    branch = sources["mechanism_factorized_cl_branch_selector"]
    return [
        _inventory(
            "handoff_requires_inventory",
            branch.get("selected_next_action") == "pivot_to_commutator_dense_teacher_source_inventory",
            "required",
            branch.get("selected_next_action", ""),
            "previous selector explicitly handed off to this inventory",
        ),
        _inventory(
            "dense_teacher_pair_composer_closed_by_dense_mlp_controls",
            pair.get("claim_status") == "pair_composer_closed_dense_mlp_controls_dominate_no_gpu",
            "dense_teacher",
            pair.get("decision", ""),
            "pair-composer signal is already closed as sparse-column evidence",
        ),
        _inventory(
            "dense_teacher_distillation_closed_negative",
            distill.get("claim_status") == "dense_teacher_distillation_negative_closeout_no_gpu",
            "dense_teacher",
            distill.get("decision", ""),
            "dense-teacher residual distillation is local negative evidence",
        ),
        _inventory(
            "dense_mlp_synthesis_redirect_already_completed",
            dense_synthesis.get("status") == "pass" and dense_synthesis.get("requires_gpu_now") is False,
            "dense_teacher",
            dense_synthesis.get("selected_next_action", ""),
            "dense/MLP dominance synthesis has already selected a local sparse-interference direction",
        ),
        _inventory(
            "flat_value_commutator_closed_as_generic_capacity",
            flat_comm.get("claim_status") == "flat_value_capacity_closed_as_generic_capacity",
            "commutator",
            flat_comm.get("decision", ""),
            "flat-value commutator mitigation did not establish a sparse mechanism",
        ),
        _inventory(
            "topk2_finite_update_fields_already_promoted_to_control_matrix",
            post_finite.get("selected_next_action")
            == "extend_causal_fingerprint_control_matrix_with_finite_update_fields",
            "commutator",
            post_finite.get("decision", ""),
            "finite-update evidence was already routed into a no-training control-matrix extension",
        ),
        _inventory(
            "explicit_order_averaging_selected_but_not_yet_run_here",
            mitigation_selector.get("selected_next_action") == "explicit_order_averaging_mitigation_probe",
            "commutator",
            mitigation_selector.get("decision", ""),
            "the existing mitigation selector identifies explicit order averaging as the next bounded local probe",
        ),
        _inventory(
            "latest_strategy_review_does_not_override_local_pivot",
            not strategy.get("ben_notification_required", False),
            "strategy",
            strategy.get("recommended_next_action", ""),
            "latest review is minor/no-notify and its scale-constrained recommendation is already satisfied",
        ),
    ]


def _duplicate_work_rows(sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _duplicate(
            DENSE_TEACHER_ACTION,
            True,
            "rejected",
            "pair-composer and dense-teacher distillation closeouts are already negative versus dense/MLP or null controls",
            {
                "pair_composer": sources["dense_teacher_pair_composer_closeout"].get("decision", ""),
                "distillation": sources["dense_teacher_residual_distillation_closeout"].get("decision", ""),
            },
        ),
        _duplicate(
            NEW_TRAINING_ACTION,
            True,
            "deferred",
            "the current handoff asks for a source inventory before new training; a smaller no-training commutator probe is already selected",
            sources["mechanism_factorized_cl_branch_selector"].get("selected_next_step", ""),
        ),
        _duplicate(
            "repeat_flat_value_commutator_mitigation",
            True,
            "rejected",
            "flat-value commutator mitigation is already closed as generic capacity",
            sources["same_router_flat_value_commutator_mitigation_closeout"].get("decision", ""),
        ),
        _duplicate(
            GPU_ACTION,
            True,
            "rejected",
            "no source report requires GPU validation and promotion is not allowed",
            "requires_gpu_now=false",
        ),
    ]


def _candidate_actions(
    failures: list[dict[str, Any]],
    inventory_rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if failures:
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "required source inventory inputs are missing or inconsistent",
                "repair local source reports before choosing another commutator or dense-teacher branch",
                "source_repair_required",
            )
        ]

    rows = {row["criterion"]: row for row in inventory_rows}
    dense_closed = (
        rows["dense_teacher_pair_composer_closed_by_dense_mlp_controls"]["passed"]
        and rows["dense_teacher_distillation_closed_negative"]["passed"]
        and rows["dense_mlp_synthesis_redirect_already_completed"]["passed"]
    )
    commutator_ready = (
        rows["flat_value_commutator_closed_as_generic_capacity"]["passed"]
        and rows["topk2_finite_update_fields_already_promoted_to_control_matrix"]["passed"]
        and rows["explicit_order_averaging_selected_but_not_yet_run_here"]["passed"]
    )
    if rows["handoff_requires_inventory"]["passed"] and dense_closed and commutator_ready:
        return [
            _candidate(
                ORDER_AVERAGING_ACTION,
                "selected",
                (
                    "dense-teacher branches are already closed by dense/MLP or null controls, while "
                    "finite-update evidence has an existing local mitigation selector; the smallest "
                    "non-duplicative next step is the explicit order-averaging mitigation probe"
                ),
                "run `python -m relaleap.experiments.promoted_topk2_explicit_order_averaging_mitigation_probe` locally; keep GPU blocked",
                "commutator_inventory_selects_order_averaging_probe_no_gpu",
            ),
            _candidate(
                DENSE_TEACHER_ACTION,
                "rejected",
                _duplicate_reason(duplicate_rows, DENSE_TEACHER_ACTION),
                "do not reopen dense-teacher branches without new headroom evidence",
                "dense_teacher_reopen_rejected",
            ),
            _candidate(
                NEW_TRAINING_ACTION,
                "deferred",
                _duplicate_reason(duplicate_rows, NEW_TRAINING_ACTION),
                "defer new training until the no-training commutator mitigation probe is recorded",
                "new_training_deferred_by_inventory",
            ),
            _candidate(
                GPU_ACTION,
                "rejected",
                _duplicate_reason(duplicate_rows, GPU_ACTION),
                "do not use RunPod or Colab for this inventory result",
                "gpu_validation_rejected_by_inventory",
            ),
        ]

    return [
        _candidate(
            REPAIR_ACTION,
            "selected",
            "source reports are present but do not encode a coherent dense-teacher-closed plus commutator-ready state",
            "inspect source summaries and regenerate this inventory after resolving stale evidence",
            "branch_state_inconsistent",
        )
    ]


def _source_failures(source_rows: list[dict[str, Any]], sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    failures = [
        {"source": row["source"], "reason": "missing_required_source", "path": row["path"]}
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]
    branch = sources["mechanism_factorized_cl_branch_selector"]
    if branch and branch.get("selected_next_action") != "pivot_to_commutator_dense_teacher_source_inventory":
        failures.append(
            {
                "source": "mechanism_factorized_cl_branch_selector",
                "reason": "handoff_does_not_select_this_inventory",
                "path": str(DEFAULT_BRANCH_SELECTOR),
            }
        )
    return failures


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(payload),
        "status": payload.get("status") if payload else "missing",
        "decision": payload.get("decision") if payload else "",
        "claim_status": payload.get("claim_status") if payload else "",
        "selected_next_action": payload.get("selected_next_action", "") if payload else "",
        "selected_next_step": payload.get("selected_next_step", "") if payload else "",
    }


def _inventory(
    criterion: str,
    passed: bool,
    evidence_family: str,
    observed: Any,
    interpretation: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "evidence_family": evidence_family,
        "observed": observed,
        "interpretation": interpretation,
    }


def _duplicate(
    action: str,
    duplicate_or_blocked: bool,
    disposition: str,
    reason: str,
    source_decision: Any,
) -> dict[str, Any]:
    return {
        "candidate_action": action,
        "duplicate_or_blocked": bool(duplicate_or_blocked),
        "disposition": disposition if duplicate_or_blocked else "not_applicable",
        "reason": reason,
        "source_decision": source_decision,
    }


def _candidate(
    action: str,
    disposition: str,
    reason: str,
    next_step: str,
    claim_status: str,
) -> dict[str, str]:
    return {
        "candidate_action": action,
        "disposition": disposition,
        "reason": reason,
        "next_step": next_step,
        "claim_status": claim_status,
    }


def _duplicate_reason(rows: list[dict[str, Any]], action: str) -> str:
    return str(next((row["reason"] for row in rows if row["candidate_action"] == action), ""))


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    values: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"strategic_change_level", "notify_ben", "recommended_next_action", "verdict"}:
            values[key] = value.strip()
    return {
        "present": path.is_file(),
        "strategic_change_level": values.get("strategic_change_level", "missing"),
        "notify_ben": values.get("notify_ben", "false"),
        "recommended_next_action": values.get("recommended_next_action", ""),
        "verdict": values.get("verdict", ""),
        "ben_notification_required": values.get("notify_ben", "false").lower() == "true"
        or values.get("strategic_change_level") == "major",
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if strategy.get("ben_notification_required"):
        return (
            "Read the latest external review; it requests Ben notification or a major shift, "
            "so this inventory records that condition and still keeps GPU/default changes blocked."
        )
    return (
        "Read the latest external review. Its scale-constrained recommendation has already been "
        "implemented and closed by the current source chain; no GPT-5.5-Pro recommendation is rejected."
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "inventory_rows.csv", summary["inventory_rows"])
    _write_csv(out_dir / "duplicate_work_rows.csv", summary["duplicate_work_rows"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Commutator/Dense-Teacher Source Inventory",
            "",
            f"- Status: {summary['status']}",
            f"- Decision: {summary['decision']}",
            f"- Claim status: {summary['claim_status']}",
            f"- Selected next action: {summary['selected_next_action']}",
            f"- Selected next step: {summary['selected_next_step']}",
            "- GPU validation remains blocked: requires_gpu_now=false, promotion_allowed=false, advance_to_gpu_validation=false.",
            f"- Strategy review handling: {summary['strategy_review_handling']}",
            "",
            "## Rationale",
            "",
            str(summary["rationale"]),
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
    parser.add_argument("--branch-selector", type=Path, default=DEFAULT_BRANCH_SELECTOR)
    parser.add_argument("--dense-mlp-synthesis", type=Path, default=DEFAULT_DENSE_MLP_SYNTHESIS)
    parser.add_argument(
        "--dense-teacher-distillation-closeout",
        type=Path,
        default=DEFAULT_DENSE_TEACHER_DISTILLATION_CLOSEOUT,
    )
    parser.add_argument("--pair-composer-closeout", type=Path, default=DEFAULT_PAIR_COMPOSER_CLOSEOUT)
    parser.add_argument(
        "--flat-value-commutator-closeout",
        type=Path,
        default=DEFAULT_FLAT_VALUE_COMMUTATOR_CLOSEOUT,
    )
    parser.add_argument(
        "--topk2-post-finite-update-closeout",
        type=Path,
        default=DEFAULT_TOPK2_POST_FINITE_UPDATE_CLOSEOUT,
    )
    parser.add_argument(
        "--topk2-mitigation-selector",
        type=Path,
        default=DEFAULT_TOPK2_MITIGATION_SELECTOR,
    )
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_commutator_dense_teacher_source_inventory(
        branch_selector_path=args.branch_selector,
        dense_mlp_synthesis_path=args.dense_mlp_synthesis,
        dense_teacher_distillation_closeout_path=args.dense_teacher_distillation_closeout,
        pair_composer_closeout_path=args.pair_composer_closeout,
        flat_value_commutator_closeout_path=args.flat_value_commutator_closeout,
        topk2_post_finite_update_closeout_path=args.topk2_post_finite_update_closeout,
        topk2_mitigation_selector_path=args.topk2_mitigation_selector,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
