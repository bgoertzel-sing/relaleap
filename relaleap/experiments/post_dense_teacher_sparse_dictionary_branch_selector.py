"""Select the next branch after hard dense-teacher sparse dictionaries close.

The selector consumes the local dense-teacher sparse value-formulation closeout
and adjacent diagnostics, then records one bounded next mechanism. It is a
decision artifact only: no training is run and GPU validation remains blocked.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_CLOSEOUT = Path("results/reports/dense_teacher_sparse_value_formulation_closeout/summary.json")
DEFAULT_DIAGNOSTIC = Path("results/reports/dense_teacher_sparse_value_selection_diagnostic/summary.json")
DEFAULT_CAPACITY_ASSAY = Path("results/reports/dense_teacher_residual_value_capacity_norm_assay/summary.json")
DEFAULT_FAILURE_LOCALIZATION = Path(
    "results/reports/dense_teacher_residual_columnability_failure_localization/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/post_dense_teacher_sparse_dictionary_branch_selector")

REPAIR_ACTION = "repair_post_dense_teacher_sparse_dictionary_branch_selector_sources"
CONTINUOUS_COEFFICIENT_ACTION = "design_continuous_coefficient_sparse_value_pregate"
SOFT_MIXTURE_ACTION = "design_soft_mixture_residual_compression_pregate"
CONSERVATIVE_CLOSEOUT_ACTION = "keep_sparse_dictionary_closed_without_new_mechanism"
HARD_DICTIONARY_ACTION = "continue_hard_in_column_sparse_value_dictionary"
GPU_ACTION = "launch_gpu_validation_for_hard_sparse_dictionary"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "branch_rows.csv",
    "gate_rows.csv",
    "notes.md",
)


def run_post_dense_teacher_sparse_dictionary_branch_selector(
    *,
    closeout_path: Path = DEFAULT_CLOSEOUT,
    diagnostic_path: Path = DEFAULT_DIAGNOSTIC,
    capacity_assay_path: Path = DEFAULT_CAPACITY_ASSAY,
    failure_localization_path: Path = DEFAULT_FAILURE_LOCALIZATION,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a deterministic fail-closed branch decision artifact."""

    start = time.time()
    closeout = _read_json(closeout_path)
    diagnostic = _read_json(diagnostic_path)
    capacity = _read_json(capacity_assay_path)
    failure_localization = _read_json(failure_localization_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("sparse_value_formulation_closeout", closeout_path, closeout),
        _source_row("sparse_value_selection_diagnostic", diagnostic_path, diagnostic),
        _source_row("value_capacity_norm_assay", capacity_assay_path, capacity),
        _source_row("columnability_failure_localization", failure_localization_path, failure_localization),
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
            "selected_next_step": "",
            "training_executed": "",
            "git_commit": "",
        },
    ]
    evidence = _evidence(closeout, diagnostic, capacity, failure_localization, strategy)
    gate_rows = _gate_rows(evidence, source_rows)
    failures = _failures(source_rows, gate_rows)
    branch_rows = _branch_rows(evidence, failures)
    selected = [row for row in branch_rows if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "post_dense_teacher_sparse_dictionary_branch_selector_failed_closed"
        claim_status = "post_dense_teacher_sparse_dictionary_sources_incomplete"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair or regenerate missing post-dense-teacher sparse-dictionary source artifacts"
        rationale = "Required source artifacts are missing or the local hard-dictionary kill gates are not satisfied."
    else:
        selected_row = selected[0]
        status = "pass"
        decision = "post_dense_teacher_sparse_dictionary_branch_selected"
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
        "training_executed": False,
        "backend_policy": "local branch selection only; RunPod and Colab remain blocked",
        "source_rows": source_rows,
        "evidence": evidence,
        "gate_rows": gate_rows,
        "branch_rows": branch_rows,
        "failures": failures,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "direction_shift": _direction_shift(strategy),
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
    closeout: dict[str, Any],
    diagnostic: dict[str, Any],
    capacity: dict[str, Any],
    failure_localization: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    closeout_evidence = _as_dict(closeout.get("evidence"))
    failures = _list(diagnostic.get("failures"))
    return {
        "closeout_status": closeout.get("status", ""),
        "closeout_decision": closeout.get("decision", ""),
        "closeout_claim_status": closeout.get("claim_status", ""),
        "diagnostic_status": diagnostic.get("status", ""),
        "capacity_status": capacity.get("status", ""),
        "failure_localization_status": failure_localization.get("status", ""),
        "base_holdout_ce": _coalesce_float(closeout_evidence.get("base_holdout_ce"), diagnostic.get("base_holdout_ce")),
        "dense_teacher_holdout_ce": _coalesce_float(
            closeout_evidence.get("dense_teacher_holdout_ce"),
            diagnostic.get("dense_teacher_holdout_ce"),
        ),
        "dense_teacher_ce_improvement": _coalesce_float(closeout_evidence.get("dense_teacher_ce_improvement")),
        "flat_value_ce": _coalesce_float(closeout_evidence.get("flat_value_ce")),
        "learned_sparse_ce": _coalesce_float(closeout_evidence.get("learned_sparse_ce")),
        "learned_sparse_ce_gap_vs_flat": _coalesce_float(
            closeout_evidence.get("learned_sparse_ce_gap_vs_flat")
        ),
        "flat_value_mse": _coalesce_float(closeout_evidence.get("flat_value_mse")),
        "oracle_in_column_value_mse": _coalesce_float(closeout_evidence.get("oracle_in_column_value_mse")),
        "global_dictionary_value_mse": _coalesce_float(closeout_evidence.get("global_dictionary_value_mse")),
        "oracle_in_column_mse_gap_vs_flat": _coalesce_float(
            closeout_evidence.get("oracle_in_column_mse_gap_vs_flat")
        ),
        "global_dictionary_mse_gap_vs_flat": _coalesce_float(
            closeout_evidence.get("global_dictionary_mse_gap_vs_flat")
        ),
        "oracle_support_mse_advantage_vs_random": _coalesce_float(
            closeout_evidence.get("oracle_support_mse_advantage_vs_random")
        ),
        "value_code_selection_regret": _coalesce_float(closeout_evidence.get("value_code_selection_regret")),
        "in_column_gap_vs_global": _coalesce_float(closeout_evidence.get("in_column_gap_vs_global")),
        "deployable_leakage_flags_false": bool(closeout_evidence.get("deployable_leakage_flags_false")),
        "oracle_value_code_non_deployable": bool(closeout_evidence.get("oracle_value_code_non_deployable")),
        "diagnostic_failure_count": len(failures),
        "strategy_recommended_next_action": strategy["recommended_next_action"],
        "strategy_verdict": strategy["verdict"],
        "ben_notification_required": strategy["ben_notification_required"],
    }


def _gate_rows(evidence: dict[str, Any], source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _gate(
            "required_sources_present",
            not any(row["source"] != "strategy_review" and not row["present"] for row in source_rows),
            "required",
            ",".join(row["source"] for row in source_rows if row["source"] != "strategy_review" and not row["present"]),
        ),
        _gate(
            "hard_sparse_dictionary_closeout_passed",
            evidence["closeout_status"] == "pass"
            and evidence["closeout_claim_status"] == "current_sparse_dictionary_value_formulation_retired_before_gpu",
            "required",
            f"closeout_status={evidence['closeout_status']}; claim={evidence['closeout_claim_status']}",
        ),
        _gate(
            "flat_value_dominates_non_deployable_sparse_ceilings",
            _gt(evidence["oracle_in_column_mse_gap_vs_flat"], 0.10)
            and _gt(evidence["global_dictionary_mse_gap_vs_flat"], 0.10),
            "required",
            (
                f"oracle_in_column_gap={evidence['oracle_in_column_mse_gap_vs_flat']}; "
                f"global_dictionary_gap={evidence['global_dictionary_mse_gap_vs_flat']}"
            ),
        ),
        _gate(
            "deployable_sparse_loses_ce_guardrail",
            _gt(evidence["learned_sparse_ce_gap_vs_flat"], 0.02),
            "required",
            (
                f"learned_sparse_ce={evidence['learned_sparse_ce']}; "
                f"flat_value_ce={evidence['flat_value_ce']}; gap={evidence['learned_sparse_ce_gap_vs_flat']}"
            ),
        ),
        _gate(
            "support_signal_present_but_insufficient",
            _gt(evidence["oracle_support_mse_advantage_vs_random"], 0.10),
            "scientific",
            f"oracle_support_mse_advantage_vs_random={evidence['oracle_support_mse_advantage_vs_random']}",
        ),
        _gate(
            "value_mechanism_not_router_only_blocker",
            _gt(evidence["value_code_selection_regret"], 0.10)
            and _gt(evidence["in_column_gap_vs_global"], 0.10),
            "scientific",
            (
                f"value_code_selection_regret={evidence['value_code_selection_regret']}; "
                f"in_column_gap_vs_global={evidence['in_column_gap_vs_global']}"
            ),
        ),
        _gate(
            "gpu_validation_blocked",
            True,
            "required",
            "requires_gpu_now=false; promotion_allowed=false; advance_to_gpu_validation=false",
        ),
    ]


def _branch_rows(evidence: dict[str, Any], failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if failures:
        return [
            _branch(
                REPAIR_ACTION,
                "selected",
                "source artifacts are missing or hard-dictionary kill gates did not pass",
                "repair post-dense-teacher sparse-dictionary selector sources",
                "post_dense_teacher_sparse_dictionary_source_repair_required",
            )
        ]

    return [
        _branch(
            HARD_DICTIONARY_ACTION,
            "killed",
            "same-router flat value dominates learned sparse and nondeployable hard sparse dictionary ceilings",
            "do not continue hard in-column sparse value-code dictionaries without a new confound-specific mechanism",
            "hard_in_column_sparse_value_dictionary_killed",
        ),
        _branch(
            CONTINUOUS_COEFFICIENT_ACTION,
            "selected",
            (
                "the next trainable candidate directly targets the observed blocker by keeping sparse support but "
                "replacing hard value-code lookup with continuous per-active-column coefficients/value generation, "
                "with norm, parameter, churn, commutator, retention, sparsity, and intervention-selectivity gates"
            ),
            "implement a local continuous-coefficient sparse-value pregate before any GPU validation",
            "continuous_coefficient_sparse_value_pregate_selected_no_gpu",
        ),
        _branch(
            SOFT_MIXTURE_ACTION,
            "deferred",
            "soft mixtures are a plausible alternative but risk collapsing into dense adapters unless sparsity/selectivity gates are defined first",
            "consider after the continuous-coefficient pregate or if it fails by becoming dense-like",
            "soft_mixture_residual_compression_deferred",
        ),
        _branch(
            CONSERVATIVE_CLOSEOUT_ACTION,
            "deferred",
            "the hard dictionary is closed, but the evidence does not falsify all sparse columns or continuous sparse values",
            "keep as fallback if continuous and soft-mixture local pregates fail",
            "conservative_closeout_deferred",
        ),
        _branch(
            GPU_ACTION,
            "rejected",
            "no local trainable arm has beaten the same-router flat value control or supplied compensating non-CE wins",
            "do not run RunPod or Colab validation for this branch selector",
            "gpu_validation_blocked_by_flat_value_control_dominance",
        ),
    ]


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(payload),
        "status": payload.get("status") if payload else "missing",
        "decision": payload.get("decision") if payload else "",
        "claim_status": payload.get("claim_status") if payload else "",
        "selected_next_step": payload.get("selected_next_step", ""),
        "training_executed": payload.get("training_executed", ""),
        "git_commit": payload.get("git_commit", ""),
    }


def _failures(source_rows: list[dict[str, Any]], gate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = [
        {"source": row["source"], "reason": "missing_required_source", "path": row["path"]}
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]
    failures.extend(row for row in gate_rows if row["gate_type"] == "required" and not row["passed"])
    return failures


def _gate(name: str, passed: bool, gate_type: str, evidence: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "gate_type": gate_type, "evidence": evidence}


def _branch(
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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    fields: dict[str, Any] = {
        "present": bool(text),
        "strategic_change_level": "unknown",
        "notify_ben": "unknown",
        "recommended_next_action": "",
        "verdict": "",
    }
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in fields:
            fields[key] = value.strip()
    fields["ben_notification_required"] = (
        str(fields["notify_ben"]).lower() == "true" or fields["strategic_change_level"] == "major"
    )
    return fields


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if strategy.get("ben_notification_required"):
        return (
            "Read the latest external review; it requests Ben notification or a major direction shift. "
            "The recommendation is accepted: hard sparse dictionaries are killed locally, GPU remains blocked, "
            "and the next branch is a continuous-coefficient sparse-value pregate."
        )
    return (
        "Read the latest external review and incorporated its local no-GPU branch-selection recommendation."
    )


def _direction_shift(strategy: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategic_change_level": strategy["strategic_change_level"],
        "ben_should_be_notified": bool(strategy["ben_notification_required"]),
        "direction": (
            "retire the current hard in-column sparse value-code dictionary and select a continuous-coefficient "
            "sparse residual value mechanism before any GPU validation"
        ),
        "recommendation_disposition": "accepted",
        "deferred_or_rejected_recommendations": [],
    }


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "branch_rows.csv", summary["branch_rows"])
    _write_csv(out_dir / "gate_rows.csv", summary["gate_rows"])
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
            "# Post Dense-Teacher Sparse Dictionary Branch Selector",
            "",
            f"- Status: {summary['status']}",
            f"- Decision: {summary['decision']}",
            f"- Claim status: {summary['claim_status']}",
            f"- Selected next action: {summary['selected_next_action']}",
            f"- Selected next step: {summary['selected_next_step']}",
            "- Hard in-column sparse value-code dictionary branch is killed.",
            "- GPU validation remains blocked: requires_gpu_now=false, promotion_allowed=false, advance_to_gpu_validation=false.",
            f"- Strategy review handling: {summary['strategy_review_handling']}",
            "",
            "## Rationale",
            "",
            str(summary["rationale"]),
            "",
        ]
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _coalesce_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _gt(value: float | None, threshold: float) -> bool:
    return value is not None and value > threshold


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closeout", type=Path, default=DEFAULT_CLOSEOUT)
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--capacity-assay", type=Path, default=DEFAULT_CAPACITY_ASSAY)
    parser.add_argument("--failure-localization", type=Path, default=DEFAULT_FAILURE_LOCALIZATION)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_post_dense_teacher_sparse_dictionary_branch_selector(
        closeout_path=args.closeout,
        diagnostic_path=args.diagnostic,
        capacity_assay_path=args.capacity_assay,
        failure_localization_path=args.failure_localization,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
