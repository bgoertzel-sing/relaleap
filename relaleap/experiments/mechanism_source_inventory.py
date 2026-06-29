"""Inventory local mechanism evidence before opening another branch."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_RECONCILIATION = Path("results/reports/post_negative_loop_reconciliation/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/mechanism_source_inventory")

SELECTED_NEXT_ACTION = "run_acsr_broader_mechanism_gate_with_existing_local_packets"
STRATEGY_REFRESH_ACTION = "request_strategy_review_before_new_mechanism_branch"
REPAIR_ACTION = "repair_mechanism_source_inventory_inputs"

REQUIRED_SOURCES: tuple[tuple[str, Path], ...] = (
    ("post_negative_loop_reconciliation", DEFAULT_RECONCILIATION),
    ("core_periphery_negative_closeout", Path("results/reports/core_periphery_negative_evidence_closeout/summary.json")),
    (
        "dense_teacher_pair_composer_closeout",
        Path("results/reports/dense_teacher_pair_composer_pregate_closeout/summary.json"),
    ),
    ("low_churn_mlp_control_pilot", Path("results/reports/low_churn_mlp_residual_control_pilot/summary.json")),
    (
        "sparse_dense_mlp_matched_intervention_decision",
        Path("results/reports/sparse_dense_mlp_matched_intervention_decision/summary.json"),
    ),
)

OPTIONAL_SOURCES: tuple[tuple[str, Path], ...] = (
    ("acsr_broader_mechanism_gate", Path("results/audits/acsr_broader_mechanism_gate_local/summary.json")),
    ("acsr_common_causal_residual_benchmark", Path("results/reports/acsr_common_causal_residual_benchmark/summary.json")),
    ("acsr_dense_residual_transfer_control", Path("results/reports/acsr_dense_residual_transfer_control/summary.json")),
    ("acsr_post_negative_branch_selector", Path("results/reports/acsr_post_negative_branch_selector/summary.json")),
    (
        "mechanism_factorized_continual_learning_repeat",
        Path("results/reports/mechanism_factorized_continual_learning_repeat/summary.json"),
    ),
    ("acsr_finite_update_commutator_assay", Path("results/reports/acsr_finite_update_commutator_assay/summary.json")),
    (
        "dense_teacher_residual_distillation_comparison",
        Path("results/audits/token_larger_dense_teacher_residual_distillation_comparison/summary.json"),
    ),
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_inventory.csv",
    "evidence_gap_rows.csv",
    "duplicate_work_rows.csv",
    "candidate_actions.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_mechanism_source_inventory(
    *,
    reconciliation_path: Path = DEFAULT_RECONCILIATION,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
    source_root: Path = Path("."),
    urgent_review_status: str = "not_run",
) -> dict[str, Any]:
    """Write a fail-closed inventory of local mechanism evidence and gaps."""

    start = time.time()
    required_sources = _source_specs_with_reconciliation(reconciliation_path, source_root)
    optional_sources = tuple((name, _resolve_source_path(path, source_root)) for name, path in OPTIONAL_SOURCES)
    source_rows = [_source_row(name, path, required=True) for name, path in required_sources]
    source_rows.extend(_source_row(name, path, required=False) for name, path in optional_sources)
    sources = {row["source"]: _read_json(Path(row["path"])) for row in source_rows}
    review = _strategy_review(strategy_review_path)
    gap_rows = _evidence_gap_rows(source_rows, sources)
    duplicate_rows = _duplicate_work_rows(sources, review)
    criteria = _gate_criteria(source_rows, sources, gap_rows)
    hard_failures = [row for row in criteria if row["severity"] == "hard" and not row["passed"]]
    candidate_actions = _candidate_actions(hard_failures, gap_rows)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if hard_failures or len(selected) != 1:
        status = "fail"
        decision = "mechanism_source_inventory_failed_closed"
        selected_next_action = REPAIR_ACTION
        next_step = "repair missing required local source reports before selecting another mechanism step"
        claim_status = "mechanism_source_inventory_inputs_incomplete"
        rationale = "The inventory cannot safely select a non-duplicative step from missing or incoherent required source reports."
    else:
        status = "pass"
        decision = "mechanism_source_inventory_recorded"
        selected_next_action = selected[0]["candidate_action"]
        next_step = selected[0]["next_step"]
        claim_status = selected[0]["claim_status"]
        rationale = selected[0]["reason"]

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_action": selected_next_action,
        "next_step": next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "backend_policy": "local source inventory only; RunPod/Colab validation remains blocked",
        "urgent_review_status": urgent_review_status,
        "source_inventory": source_rows,
        "evidence_gap_rows": gap_rows,
        "duplicate_work_rows": duplicate_rows,
        "candidate_actions": candidate_actions,
        "gate_criteria": criteria,
        "strategy_review": review,
        "strategy_response": _strategy_response(review, urgent_review_status),
        "failures": hard_failures,
        "rationale": rationale,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _source_specs_with_reconciliation(
    reconciliation_path: Path,
    source_root: Path,
) -> tuple[tuple[str, Path], ...]:
    return (("post_negative_loop_reconciliation", reconciliation_path),) + tuple(
        (name, _resolve_source_path(path, source_root)) for name, path in REQUIRED_SOURCES[1:]
    )


def _resolve_source_path(path: Path, source_root: Path) -> Path:
    return path if path.is_absolute() else source_root / path


def _source_row(source: str, path: Path, *, required: bool) -> dict[str, Any]:
    payload = _read_json(path)
    return {
        "source": source,
        "path": str(path),
        "required": required,
        "present": path.is_file(),
        "status": payload.get("status") if payload else "missing",
        "decision": payload.get("decision") if payload else "",
        "claim_status": payload.get("claim_status") if payload else "",
        "selected_next_action": payload.get("selected_next_action", ""),
        "selected_next_step": payload.get("selected_next_step", ""),
        "scientific_gate": payload.get("scientific_gate", ""),
        "requires_gpu_now": payload.get("requires_gpu_now", ""),
        "promotion_allowed": payload.get("promotion_allowed", ""),
    }


def _evidence_gap_rows(source_rows: list[dict[str, Any]], sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    broader_present = _present(source_rows, "acsr_broader_mechanism_gate")
    broader = sources["acsr_broader_mechanism_gate"]
    return [
        _gap(
            "broader_mechanism_gate_missing",
            not broader_present,
            SELECTED_NEXT_ACTION,
            "Existing local packets have a command-driven ACSR broader mechanism gate module, but no gate artifact is present yet.",
            "local_synthesis_no_new_training",
        ),
        _gap(
            "broader_mechanism_gate_completed_failed",
            broader_present and broader.get("status") == "fail",
            STRATEGY_REFRESH_ACTION,
            "The existing-packet ACSR broader mechanism gate is already present and failed, so the remaining non-duplicative step is an external strategy refresh before another mechanism branch.",
            "all_local_mechanism_gates_closed",
        ),
        _gap(
            "common_causal_residual_benchmark_failed",
            sources["acsr_common_causal_residual_benchmark"].get("status") == "fail",
            "do_not_use_common_benchmark_as_sparse_identity_evidence",
            "The common causal residual benchmark failed to separate sparse support-specific effects from dense controls.",
            "blocked_claim",
        ),
        _gap(
            "dense_transfer_control_failed",
            sources["acsr_dense_residual_transfer_control"].get("status") == "fail",
            "require_stricter_mechanism_benchmark_before_transfer_claim",
            "Dense transfer controls block an ACSR transfer claim until a stricter mechanism gate passes.",
            "blocked_claim",
        ),
        _gap(
            "low_churn_mlp_control_blocked",
            sources["low_churn_mlp_control_pilot"].get("scientific_gate") == "blocked",
            "do_not_repeat_low_churn_mlp_control",
            "The low-churn MLP matched control already ran and produced no budgeted advancement claim.",
            "completed_negative_control",
        ),
        _gap(
            "sparse_dense_mlp_matched_decision_blocked",
            sources["sparse_dense_mlp_matched_intervention_decision"].get("scientific_gate") == "blocked",
            "do_not_repeat_matched_sparse_dense_mlp_decision",
            "The matched sparse/dense/MLP intervention decision already blocks a decisive challenger claim.",
            "completed_negative_control",
        ),
    ]


def _duplicate_work_rows(sources: dict[str, dict[str, Any]], review: dict[str, Any]) -> list[dict[str, Any]]:
    review_action = str(review.get("recommended_next_action", "")).lower()
    return [
        _duplicate(
            "latest_review_low_churn_recommendation",
            "low-churn" in review_action or "low churn" in review_action,
            "defer_as_completed",
            "The low-churn pilot is already present and blocked locally.",
            sources["low_churn_mlp_control_pilot"].get("decision", ""),
        ),
        _duplicate(
            "rerun_core_periphery_current_mechanism",
            sources["core_periphery_negative_closeout"].get("status") == "pass",
            "reject_duplicate_closed_branch",
            "The current core/periphery mechanism has a negative closeout and should not be rerun as the next step.",
            sources["core_periphery_negative_closeout"].get("selected_next_action", ""),
        ),
        _duplicate(
            "rerun_pair_composer_pregate",
            sources["dense_teacher_pair_composer_closeout"].get("status") == "pass",
            "reject_duplicate_closed_branch",
            "The pair-composer pregate is negative versus shuffled-pair null and already redirects away.",
            sources["dense_teacher_pair_composer_closeout"].get("selected_next_action", ""),
        ),
        _duplicate(
            "run_gpu_validation_now",
            True,
            "reject_local_gates_block_gpu",
            "No inventory source currently requires GPU validation; RunPod/Colab remain blocked.",
            "requires_gpu_now=false",
        ),
    ]


def _gate_criteria(
    source_rows: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    gap_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    required_rows = [row for row in source_rows if row["required"]]
    selected_gaps = [
        row
        for row in gap_rows
        if row["candidate_action"] in {SELECTED_NEXT_ACTION, STRATEGY_REFRESH_ACTION}
        and row["gap_present"]
    ]
    return [
        _criterion(
            "required_sources_present",
            all(row["present"] for row in required_rows),
            "hard",
            "reconciliation, core/periphery, pair-composer, low-churn, and matched sparse/dense/MLP summaries exist",
            [row["path"] for row in required_rows if row["present"]],
            "missing required local source report",
        ),
        _criterion(
            "required_sources_runtime_interpretable",
            all(row["status"] == "pass" for row in required_rows),
            "hard",
            "required source reports passed runtime/report gates",
            {row["source"]: row["status"] for row in required_rows},
            "at least one required report failed or is missing",
        ),
        _criterion(
            "reconciliation_selected_inventory",
            sources["post_negative_loop_reconciliation"].get("selected_next_action")
            == "run_local_mechanism_source_inventory_before_new_branch",
            "hard",
            "the previous bounded run explicitly selected this inventory step",
            sources["post_negative_loop_reconciliation"].get("selected_next_action"),
            "inventory is not the selected handoff from reconciliation",
        ),
        _criterion(
            "nonduplicative_local_gap_identified",
            len(selected_gaps) == 1,
            "claim",
            "exactly one non-duplicative local gap or strategy-refresh checkpoint should be selected",
            selected_gaps,
            "no unique local gap found",
        ),
    ]


def _candidate_actions(failures: list[dict[str, Any]], gap_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if failures:
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "required source inventory inputs are missing or failed",
                "repair local source reports before choosing another branch",
                "source_repair_required",
            )
        ]
    selected_gap = next(
        row
        for row in gap_rows
        if row["candidate_action"] in {SELECTED_NEXT_ACTION, STRATEGY_REFRESH_ACTION}
        and row["gap_present"]
    )
    if selected_gap["candidate_action"] == STRATEGY_REFRESH_ACTION:
        return [
            _candidate(
                STRATEGY_REFRESH_ACTION,
                "selected",
                selected_gap["rationale"],
                "refresh external strategy review or ask Ben for direction before implementing a new mechanism branch; keep GPU validation blocked",
                "mechanism_inventory_all_local_gates_closed_strategy_needed",
            ),
            _candidate(
                SELECTED_NEXT_ACTION,
                "rejected",
                "the broader ACSR mechanism gate artifact already exists and failed",
                "do not duplicate the completed broader gate",
                "duplicate_broader_gate_rejected",
            ),
            _candidate(
                "run_runpod_or_colab_validation",
                "rejected",
                "all relevant local mechanism gates are blocked or negative",
                "keep GPU validation blocked",
                "gpu_validation_rejected_by_local_inventory",
            ),
        ]
    return [
        _candidate(
            SELECTED_NEXT_ACTION,
            "selected",
            selected_gap["rationale"],
            "run `python -m relaleap.experiments.acsr_broader_mechanism_gate` locally using existing packets; do not start GPU validation",
            "mechanism_inventory_selects_existing_packet_gate_no_gpu",
        ),
        _candidate(
            "repeat_low_churn_mlp_or_matched_dense_controls",
            "rejected",
            "low-churn and matched sparse/dense/MLP control reports are already completed and locally blocked",
            "avoid duplicating completed dense-control work",
            "duplicate_dense_control_rejected",
        ),
        _candidate(
            "implement_new_core_periphery_mechanism",
            "deferred",
            "the inventory found a smaller existing-packet synthesis gate before a new implementation branch",
            "defer new mechanism implementation until the broader gate is recorded",
            "new_branch_deferred_until_inventory_gap_closed",
        ),
        _candidate(
            "run_runpod_or_colab_validation",
            "rejected",
            "local source reports do not require GPU and promotion is not allowed",
            "keep GPU validation blocked",
            "gpu_validation_rejected_by_local_inventory",
        ),
    ]


def _gap(
    gap: str,
    gap_present: bool,
    candidate_action: str,
    rationale: str,
    gap_type: str,
) -> dict[str, Any]:
    return {
        "gap": gap,
        "gap_present": bool(gap_present),
        "candidate_action": candidate_action if gap_present else "",
        "gap_type": gap_type,
        "rationale": rationale,
    }


def _duplicate(
    item: str,
    duplicate_or_blocked: bool,
    disposition: str,
    reason: str,
    source_decision: str,
) -> dict[str, Any]:
    return {
        "item": item,
        "duplicate_or_blocked": bool(duplicate_or_blocked),
        "disposition": disposition if duplicate_or_blocked else "not_applicable",
        "reason": reason,
        "source_decision": source_decision,
    }


def _criterion(
    criterion: str,
    passed: bool,
    severity: str,
    requirement: str,
    observed: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "severity": severity,
        "requirement": requirement,
        "observed": observed,
        "failure_reason": "" if passed else failure_reason,
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


def _present(source_rows: list[dict[str, Any]], source: str) -> bool:
    return any(row["source"] == source and row["present"] for row in source_rows)


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


def _strategy_response(review: dict[str, Any], urgent_review_status: str) -> dict[str, Any]:
    recommendation = str(review.get("recommended_next_action", ""))
    low_churn_already_done = "low-churn" in recommendation.lower() or "low churn" in recommendation.lower()
    if low_churn_already_done:
        disposition = "deferred_as_already_completed"
        reason = "The latest review's low-churn MLP pilot recommendation has already been implemented and blocked locally."
    else:
        disposition = "recorded"
        reason = "The latest review does not override the local source-inventory handoff."
    return {
        "review_recommendation": recommendation,
        "urgent_review_status": urgent_review_status,
        "disposition": disposition,
        "ben_should_be_notified": bool(review.get("ben_notification_required")),
        "reason": reason,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_inventory.csv", summary["source_inventory"])
    _write_csv(out_dir / "evidence_gap_rows.csv", summary["evidence_gap_rows"])
    _write_csv(out_dir / "duplicate_work_rows.csv", summary["duplicate_work_rows"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Mechanism Source Inventory",
            "",
            f"- Status: `{summary['status']}`.",
            f"- Decision: `{summary['decision']}`.",
            f"- Selected next action: `{summary['selected_next_action']}`.",
            f"- GPU required now: `{summary['requires_gpu_now']}`.",
            f"- Promotion allowed: `{summary['promotion_allowed']}`.",
            "",
            summary["rationale"],
            "",
        ]
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconciliation", type=Path, default=DEFAULT_RECONCILIATION)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--urgent-review-status", default="not_run")
    args = parser.parse_args(argv)
    summary = run_mechanism_source_inventory(
        reconciliation_path=args.reconciliation,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
        urgent_review_status=args.urgent_review_status,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
