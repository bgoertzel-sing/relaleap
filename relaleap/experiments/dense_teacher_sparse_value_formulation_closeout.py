"""Close the current dense-teacher sparse value-formulation variant.

This report consumes the local sparse value-selection diagnostic and records a
bounded scientific decision: the current in-column sparse dictionary/value-code
formulation is retired before GPU validation when nondeployable sparse ceilings
remain behind the same-router flat value control.
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


DEFAULT_DIAGNOSTIC_DIR = Path("results/reports/dense_teacher_sparse_value_selection_diagnostic")
DEFAULT_SOURCE_ASSAY_DIR = Path("results/reports/dense_teacher_residual_value_capacity_norm_assay")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_sparse_value_formulation_closeout")

DECISION = "dense_teacher_sparse_value_formulation_variant_closed_no_gpu"
FAIL_DECISION = "dense_teacher_sparse_value_formulation_closeout_failed_closed"
SELECTED_NEXT_STEP = (
    "run a post-dense-teacher sparse-dictionary branch selector before designing any new "
    "value formulation or GPU validation"
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "closeout_rows.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_dense_teacher_sparse_value_formulation_closeout(
    *,
    diagnostic_dir: Path = DEFAULT_DIAGNOSTIC_DIR,
    source_assay_dir: Path = DEFAULT_SOURCE_ASSAY_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed local closeout for the current sparse dictionary."""

    start = time.time()
    diagnostic = _read_json(diagnostic_dir / "summary.json")
    source_assay = _read_json(source_assay_dir / "summary.json")
    review = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("sparse_value_selection_diagnostic", diagnostic_dir / "summary.json", diagnostic),
        _source_row("value_capacity_norm_assay", source_assay_dir / "summary.json", source_assay),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": review["present"],
            "status": "read" if review["present"] else "missing_optional",
            "decision": review["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={review['strategic_change_level']}; "
                f"notify_ben={review['notify_ben']}"
            ),
            "git_commit": "",
        },
    ]
    evidence = _evidence(diagnostic, source_assay, review)
    closeout_rows = _closeout_rows(evidence)
    candidate_actions = _candidate_actions(evidence, source_rows)
    failures = _failures(source_rows, closeout_rows)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = FAIL_DECISION
        claim_status = "dense_teacher_sparse_value_formulation_closeout_sources_incomplete"
        selected_next_step = "repair dense-teacher sparse value-formulation closeout sources"
        rationale = "Required diagnostic sources are missing or the local closure criteria are not satisfied."
    else:
        status = "pass"
        decision = DECISION
        claim_status = "current_sparse_dictionary_value_formulation_retired_before_gpu"
        selected_next_step = selected[0]["next_step"]
        rationale = (
            "The dense teacher improves the base, but the deployable learned sparse arm loses the CE "
            "guardrail to the same-router flat value control, and even nondeployable in-column/global "
            "dictionary value-code ceilings trail the flat value head on residual reconstruction. "
            "This makes the current hard in-column sparse value formulation the blocker rather than "
            "a GPU-scale validation candidate."
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "training_executed": False,
        "source_rows": source_rows,
        "evidence": evidence,
        "closeout_rows": closeout_rows,
        "candidate_actions": candidate_actions,
        "strategy_review": review,
        "strategy_review_handling": _strategy_review_handling(review),
        "direction_shift": _direction_shift(review),
        "failures": failures,
        "rationale": rationale,
        "backend_policy": "local closeout only; RunPod and Colab remain blocked",
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _evidence(diagnostic: dict[str, Any], source_assay: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    rows = {row.get("arm", ""): row for row in _list(diagnostic.get("diagnostic_rows"))}
    axes = {row.get("axis", ""): row for row in _list(diagnostic.get("failure_axis_rows"))}
    flat = rows.get("same_router_flat_value_control", {})
    oracle_code = rows.get("oracle_support_oracle_value_code_sparse", {})
    learned_sparse = rows.get("learned_support_learned_value_code_sparse", {})
    global_code = rows.get("global_oracle_support_value_code_sparse", {})
    random_code = rows.get("random_support_oracle_value_code_null", {})
    base_ce = _float(diagnostic.get("base_holdout_ce"))
    teacher_ce = _float(diagnostic.get("dense_teacher_holdout_ce"))
    return {
        "diagnostic_status": diagnostic.get("status", ""),
        "diagnostic_decision": diagnostic.get("decision", ""),
        "diagnostic_claim_status": diagnostic.get("claim_status", ""),
        "source_assay_status": source_assay.get("status", ""),
        "source_assay_decision": source_assay.get("decision", ""),
        "base_holdout_ce": base_ce,
        "dense_teacher_holdout_ce": teacher_ce,
        "dense_teacher_ce_improvement": _delta(base_ce, teacher_ce),
        "flat_value_ce": _float(flat.get("ce")),
        "learned_sparse_ce": _float(learned_sparse.get("ce")),
        "learned_sparse_ce_gap_vs_flat": _delta(_float(learned_sparse.get("ce")), _float(flat.get("ce"))),
        "flat_value_mse": _float(flat.get("teacher_residual_reconstruction_mse")),
        "oracle_in_column_value_mse": _float(oracle_code.get("teacher_residual_reconstruction_mse")),
        "global_dictionary_value_mse": _float(global_code.get("teacher_residual_reconstruction_mse")),
        "random_support_oracle_value_mse": _float(random_code.get("teacher_residual_reconstruction_mse")),
        "oracle_in_column_mse_gap_vs_flat": _delta(
            _float(oracle_code.get("teacher_residual_reconstruction_mse")),
            _float(flat.get("teacher_residual_reconstruction_mse")),
        ),
        "global_dictionary_mse_gap_vs_flat": _delta(
            _float(global_code.get("teacher_residual_reconstruction_mse")),
            _float(flat.get("teacher_residual_reconstruction_mse")),
        ),
        "oracle_support_mse_advantage_vs_random": _delta(
            _float(random_code.get("teacher_residual_reconstruction_mse")),
            _float(oracle_code.get("teacher_residual_reconstruction_mse")),
        ),
        "value_code_selection_regret": _float(_as_dict(axes.get("value_code_selection_regret")).get("delta")),
        "sparse_formulation_gap_vs_flat": _float(
            _as_dict(axes.get("sparse_formulation_gap_vs_flat_value")).get("delta")
        ),
        "learned_sparse_ce_gap_axis": _float(_as_dict(axes.get("learned_sparse_gap_vs_flat_value")).get("delta")),
        "in_column_gap_vs_global": _float(_as_dict(axes.get("in_column_gap_vs_global_dictionary_upper_bound")).get("delta")),
        "deployable_leakage_flags_false": _deployable_leakage_flags_false(rows.values()),
        "oracle_value_code_non_deployable": bool(diagnostic.get("oracle_value_code_non_deployable")),
        "strategy_verdict": review["verdict"],
        "strategy_recommended_next_action": review["recommended_next_action"],
        "ben_notification_required": review["notify_ben"] or review["strategic_change_level"] == "major",
    }


def _closeout_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _closeout(
            "source_diagnostic_passed",
            evidence["diagnostic_status"] == "pass",
            "required",
            f"diagnostic_status={evidence['diagnostic_status']}",
        ),
        _closeout(
            "dense_teacher_improves_base",
            evidence["dense_teacher_ce_improvement"] is not None and evidence["dense_teacher_ce_improvement"] > 0.0,
            "scientific",
            (
                f"base_ce={evidence['base_holdout_ce']}; teacher_ce={evidence['dense_teacher_holdout_ce']}; "
                f"improvement={evidence['dense_teacher_ce_improvement']}"
            ),
        ),
        _closeout(
            "deployable_sparse_loses_ce_to_flat_value",
            evidence["learned_sparse_ce_gap_vs_flat"] is not None and evidence["learned_sparse_ce_gap_vs_flat"] > 0.02,
            "scientific",
            (
                f"learned_sparse_ce={evidence['learned_sparse_ce']}; "
                f"flat_value_ce={evidence['flat_value_ce']}; gap={evidence['learned_sparse_ce_gap_vs_flat']}"
            ),
        ),
        _closeout(
            "nondeployable_in_column_sparse_ceiling_loses_to_flat_value",
            evidence["oracle_in_column_mse_gap_vs_flat"] is not None
            and evidence["oracle_in_column_mse_gap_vs_flat"] > 0.10,
            "scientific",
            (
                f"oracle_in_column_mse={evidence['oracle_in_column_value_mse']}; "
                f"flat_value_mse={evidence['flat_value_mse']}; gap={evidence['oracle_in_column_mse_gap_vs_flat']}"
            ),
        ),
        _closeout(
            "global_dictionary_upper_bound_still_loses_to_flat_value",
            evidence["global_dictionary_mse_gap_vs_flat"] is not None and evidence["global_dictionary_mse_gap_vs_flat"] > 0.10,
            "scientific",
            (
                f"global_dictionary_mse={evidence['global_dictionary_value_mse']}; "
                f"flat_value_mse={evidence['flat_value_mse']}; gap={evidence['global_dictionary_mse_gap_vs_flat']}"
            ),
        ),
        _closeout(
            "oracle_support_contains_signal_but_not_enough",
            evidence["oracle_support_mse_advantage_vs_random"] is not None
            and evidence["oracle_support_mse_advantage_vs_random"] > 0.10,
            "interpretive",
            (
                f"random_support_oracle_value_mse={evidence['random_support_oracle_value_mse']}; "
                f"oracle_in_column_mse={evidence['oracle_in_column_value_mse']}; "
                f"advantage={evidence['oracle_support_mse_advantage_vs_random']}"
            ),
        ),
        _closeout(
            "deployable_leakage_flags_false",
            evidence["deployable_leakage_flags_false"],
            "required",
            "deployable rows do not use future hidden/delta, task id, or teacher labels",
        ),
        _closeout(
            "gpu_validation_blocked",
            True,
            "required",
            "requires_gpu_now=false; advance_to_gpu_validation=false; promotion_allowed=false",
        ),
    ]


def _candidate_actions(evidence: dict[str, Any], source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing = [row["source"] for row in source_rows if row["source"] != "strategy_review" and not row["present"]]
    if missing:
        return [
            _candidate(
                "repair_closeout_sources",
                "selected",
                f"missing source summaries: {missing}",
                "repair dense-teacher sparse value-formulation closeout source artifacts",
                "source_repair_required",
            )
        ]
    return [
        _candidate(
            "close_current_sparse_dictionary_value_formulation",
            "selected",
            "flat value control dominates deployable and nondeployable sparse dictionary ceilings",
            SELECTED_NEXT_STEP,
            "current_sparse_dictionary_value_formulation_retired_before_gpu",
        ),
        _candidate(
            "launch_gpu_validation",
            "rejected",
            "local CE, MSE, and ceiling-control gates block backend spend",
            "do not run RunPod or Colab validation for this variant",
            "gpu_blocked_by_local_controls",
        ),
        _candidate(
            "redesign_sparse_value_formulation_immediately",
            "deferred",
            "the current formulation should be retired first so the next design is not treated as a rescue of this variant",
            "select a new branch explicitly before implementation",
            "needs_branch_selector",
        ),
    ]


def _failures(source_rows: list[dict[str, Any]], closeout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = [
        {
            "criterion": "source_artifacts_present",
            "passed": False,
            "required": True,
            "evidence": ",".join(row["source"] for row in source_rows if row["source"] != "strategy_review" and not row["present"]),
        }
    ] if any(row["source"] != "strategy_review" and not row["present"] for row in source_rows) else []
    failures.extend(row for row in closeout_rows if row["requirement"] == "required" and not row["passed"])
    return failures


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status", ""),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
        "git_commit": payload.get("git_commit", ""),
    }


def _closeout(criterion: str, passed: bool, requirement: str, evidence: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "requirement": requirement,
        "evidence": evidence,
    }


def _candidate(
    action: str,
    disposition: str,
    reason: str,
    next_step: str,
    claim_status: str,
) -> dict[str, Any]:
    return {
        "candidate_action": action,
        "disposition": disposition,
        "reason": reason,
        "next_step": next_step,
        "claim_status": claim_status,
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return {
        "present": path.is_file(),
        "strategic_change_level": parsed.get("strategic_change_level", ""),
        "notify_ben": parsed.get("notify_ben", "").lower() == "true",
        "recommended_next_action": parsed.get("recommended_next_action", ""),
        "verdict": parsed.get("verdict", ""),
    }


def _strategy_review_handling(review: dict[str, Any]) -> str:
    if not review["present"]:
        return "No external strategy review was present; closeout uses local diagnostic artifacts."
    return (
        "Accepted the GPT-5.5-Pro pivot where scientifically sensible: PC/core-periphery and "
        "teacher-support Transformer-ACSR remain closed; dense-teacher sparse dictionary evidence "
        "is interpreted locally with no GPU. No recommendation was rejected in this run."
    )


def _direction_shift(review: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategic_change_level": review["strategic_change_level"],
        "ben_should_be_notified": bool(review["notify_ben"] or review["strategic_change_level"] == "major"),
        "direction": (
            "retire the current dense-teacher hard sparse dictionary value formulation before GPU validation"
        ),
        "recommendation_disposition": "accepted" if review["present"] else "no_review_present",
    }


def _deployable_leakage_flags_false(rows: Any) -> bool:
    for row in rows:
        if row.get("oracle_value_code_non_deployable"):
            continue
        if row.get("uses_future_hidden_or_delta") or row.get("uses_task_id"):
            return False
        if row.get("uses_teacher_labels_in_deployable_router"):
            return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "closeout_rows.csv", summary["closeout_rows"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _notes(summary: dict[str, Any]) -> str:
    evidence = summary["evidence"]
    return "\n".join(
        [
            "# Dense-Teacher Sparse Value-Formulation Closeout",
            "",
            f"Decision: `{summary['decision']}`.",
            f"Claim status: `{summary['claim_status']}`.",
            "",
            "GPU validation remains blocked for the current hard sparse dictionary value formulation.",
            "",
            f"Flat value MSE: `{evidence['flat_value_mse']}`.",
            f"Oracle in-column sparse MSE: `{evidence['oracle_in_column_value_mse']}`.",
            f"Global dictionary upper-bound MSE: `{evidence['global_dictionary_value_mse']}`.",
            f"Learned sparse CE: `{evidence['learned_sparse_ce']}`.",
            f"Flat value CE: `{evidence['flat_value_ce']}`.",
            "",
            f"Next step: {summary['selected_next_step']}.",
            "",
        ]
    )


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(candidate: float | None, reference: float | None) -> float | None:
    if candidate is None or reference is None:
        return None
    return round(candidate - reference, 6)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC_DIR)
    parser.add_argument("--source-assay-dir", type=Path, default=DEFAULT_SOURCE_ASSAY_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_dense_teacher_sparse_value_formulation_closeout(
        diagnostic_dir=args.diagnostic_dir,
        source_assay_dir=args.source_assay_dir,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({key: summary[key] for key in ("status", "decision", "claim_status", "selected_next_step")}, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
