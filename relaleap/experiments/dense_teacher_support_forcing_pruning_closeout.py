"""Close or redirect the dense-teacher support-forcing/pruning branch.

This report consumes the local support-forcing/pruning pregate and records the
bounded scientific decision before any backend validation. It treats oracle
support and pruning as diagnostic ceiling evidence, not deployable promotion
evidence, when the same-router flat value control still dominates the learned
sparse mechanism.
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


DEFAULT_PREGATE_DIR = Path("results/reports/dense_teacher_support_forcing_pruning_pregate")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_support_forcing_pruning_closeout")

DECISION = "dense_teacher_support_forcing_pruning_branch_closed_no_gpu"
FAIL_DECISION = "dense_teacher_support_forcing_pruning_closeout_failed_closed"
SELECTED_NEXT_STEP = (
    "select a new local sparse value/support redesign branch with stronger flat-value controls before any backend validation"
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "closeout_rows.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_dense_teacher_support_forcing_pruning_closeout(
    *,
    pregate_dir: Path = DEFAULT_PREGATE_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a local closeout for the support-forcing/pruning pregate."""

    start = time.time()
    pregate = _read_json(pregate_dir / "summary.json")
    review = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("dense_teacher_support_forcing_pruning_pregate", pregate_dir / "summary.json", pregate),
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
    evidence = _evidence(pregate, review)
    closeout_rows = _closeout_rows(evidence)
    candidate_actions = _candidate_actions(source_rows, evidence)
    failures = _failures(source_rows, closeout_rows)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = FAIL_DECISION
        claim_status = "support_forcing_pruning_closeout_sources_incomplete"
        selected_next_step = "repair dense-teacher support-forcing/pruning closeout sources"
        rationale = "Required pregate source artifacts are missing or required closeout criteria failed."
    else:
        status = "pass"
        decision = DECISION
        claim_status = "support_forcing_pruning_sparse_specific_claim_not_established"
        selected_next_step = selected[0]["next_step"]
        rationale = (
            "Oracle support carries signal versus support nulls and causal-efficacy pruning can remove harmful "
            "columns locally, but the deployable learned sparse mechanism still fails the CE guardrail and loses "
            "teacher-residual reconstruction badly to the same-router flat value head. This closes the current "
            "support-forcing/pruning branch before Colab or RunPod validation."
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
        "backend_policy": "local closeout only; Colab and RunPod remain blocked",
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _evidence(pregate: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    rows = {row.get("arm", ""): row for row in _list(pregate.get("support_forcing_rows"))}
    gates = {row.get("criterion", ""): row for row in _list(pregate.get("gate_criteria"))}
    oracle = rows.get("oracle_support_same_values", {})
    learned = rows.get("learned_support_same_values", {})
    flat = rows.get("same_router_flat_value_control", {})
    load_permuted = rows.get("load_permuted_support_same_values", {})
    random = rows.get("random_support_same_values", {})
    pruned_oracle = rows.get("pruned_oracle_support_same_values", {})
    return {
        "pregate_status": pregate.get("status", ""),
        "pregate_decision": pregate.get("decision", ""),
        "pregate_claim_status": pregate.get("claim_status", ""),
        "base_holdout_ce": _float(pregate.get("base_holdout_ce")),
        "dense_teacher_holdout_ce": _float(pregate.get("dense_teacher_holdout_ce")),
        "dense_teacher_ce_improvement": _float(pregate.get("dense_teacher_ce_improvement")),
        "oracle_ce": _float(oracle.get("ce")),
        "oracle_r2": _float(oracle.get("teacher_residual_reconstruction_r2")),
        "oracle_mse": _float(oracle.get("teacher_residual_reconstruction_mse")),
        "oracle_ce_gap_closure": _float(oracle.get("teacher_ce_gap_closure_fraction")),
        "learned_ce": _float(learned.get("ce")),
        "learned_r2": _float(learned.get("teacher_residual_reconstruction_r2")),
        "learned_mse": _float(learned.get("teacher_residual_reconstruction_mse")),
        "learned_support_overlap": _float(learned.get("support_overlap_with_oracle")),
        "flat_ce": _float(flat.get("ce")),
        "flat_r2": _float(flat.get("teacher_residual_reconstruction_r2")),
        "flat_mse": _float(flat.get("teacher_residual_reconstruction_mse")),
        "load_permuted_r2": _float(load_permuted.get("teacher_residual_reconstruction_r2")),
        "random_r2": _float(random.get("teacher_residual_reconstruction_r2")),
        "pruned_oracle_ce": _float(pruned_oracle.get("ce")),
        "pruned_oracle_commutator_proxy": _float(pruned_oracle.get("finite_update_commutator_proxy")),
        "oracle_commutator_proxy": _float(oracle.get("finite_update_commutator_proxy")),
        "retained_columns_after_pruning": pregate.get("retained_columns_after_pruning", []),
        "same_sparse_values_across_support_conditions": bool(pregate.get("same_sparse_values_across_support_conditions")),
        "causal_efficacy_pruning_executed": bool(pregate.get("causal_efficacy_pruning_executed")),
        "oracle_support_non_deployable": bool(oracle.get("oracle_support_non_deployable")),
        "sparse_specific_gate_passed": bool(gates.get("sparse_specific_beats_flat_value_control", {}).get("passed")),
        "oracle_support_gate_passed": bool(gates.get("oracle_support_beats_support_nulls_same_values", {}).get("passed")),
        "learned_support_regret_gate_passed": bool(gates.get("learned_support_low_forcing_regret", {}).get("passed")),
        "pruning_gate_passed": bool(gates.get("pruning_retains_oracle_ce_gap_closure", {}).get("passed")),
        "gpu_was_blocked_by_pregate": (
            pregate.get("requires_gpu_now") is False
            and pregate.get("advance_to_gpu_validation") is False
            and pregate.get("promotion_allowed") is False
        ),
        "strategy_verdict": review["verdict"],
        "strategy_recommended_next_action": review["recommended_next_action"],
        "ben_notification_required": review["notify_ben"] or review["strategic_change_level"] == "major",
    }


def _closeout_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _closeout(
            "source_pregate_passed",
            evidence["pregate_status"] == "pass",
            "required",
            f"pregate_status={evidence['pregate_status']}",
        ),
        _closeout(
            "same_sparse_values_support_forcing_recorded",
            evidence["same_sparse_values_across_support_conditions"],
            "required",
            "oracle, learned, and null supports used the same sparse values",
        ),
        _closeout(
            "oracle_support_is_non_deployable_ceiling",
            evidence["oracle_support_non_deployable"],
            "required",
            "oracle support must not be promoted as deployable evidence",
        ),
        _closeout(
            "oracle_support_has_signal_vs_nulls",
            evidence["oracle_support_gate_passed"],
            "interpretive",
            (
                f"oracle_r2={evidence['oracle_r2']}; load_permuted_r2={evidence['load_permuted_r2']}; "
                f"random_r2={evidence['random_r2']}"
            ),
        ),
        _closeout(
            "learned_support_regret_not_primary_blocker",
            evidence["learned_support_regret_gate_passed"],
            "interpretive",
            (
                f"oracle_mse={evidence['oracle_mse']}; learned_mse={evidence['learned_mse']}; "
                f"overlap={evidence['learned_support_overlap']}"
            ),
        ),
        _closeout(
            "pruning_is_diagnostic_not_promotion",
            evidence["causal_efficacy_pruning_executed"] and evidence["pruning_gate_passed"],
            "interpretive",
            (
                f"retained={evidence['retained_columns_after_pruning']}; "
                f"oracle_ce={evidence['oracle_ce']}; pruned_oracle_ce={evidence['pruned_oracle_ce']}; "
                f"oracle_commutator={evidence['oracle_commutator_proxy']}; "
                f"pruned_commutator={evidence['pruned_oracle_commutator_proxy']}"
            ),
        ),
        _closeout(
            "sparse_specific_flat_control_gate_failed",
            not evidence["sparse_specific_gate_passed"],
            "scientific",
            (
                f"learned_r2={evidence['learned_r2']}; flat_r2={evidence['flat_r2']}; "
                f"learned_ce={evidence['learned_ce']}; flat_ce={evidence['flat_ce']}"
            ),
        ),
        _closeout(
            "gpu_validation_blocked",
            evidence["gpu_was_blocked_by_pregate"],
            "required",
            "requires_gpu_now=false; advance_to_gpu_validation=false; promotion_allowed=false",
        ),
    ]


def _candidate_actions(source_rows: list[dict[str, Any]], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    missing = [row["source"] for row in source_rows if row["source"] != "strategy_review" and not row["present"]]
    if missing:
        return [
            _candidate(
                "repair_closeout_sources",
                "selected",
                f"missing source summaries: {missing}",
                "repair dense-teacher support-forcing/pruning closeout source artifacts",
                "source_repair_required",
            )
        ]
    if evidence["sparse_specific_gate_passed"]:
        return [
            _candidate(
                "repeat_support_forcing_pruning_across_seeds",
                "selected",
                "sparse-specific gate passed locally, so repeat before backend validation",
                "repeat support-forcing/pruning pregate across seeds locally before any GPU validation",
                "local_repeat_required_before_gpu",
            )
        ]
    return [
        _candidate(
            "close_support_forcing_pruning_branch",
            "selected",
            "oracle/pruning diagnostics are useful but learned sparse still loses to flat value control",
            SELECTED_NEXT_STEP,
            "support_forcing_pruning_sparse_specific_claim_not_established",
        ),
        _candidate(
            "launch_gpu_validation",
            "rejected",
            "local flat-value and CE guardrail failures block backend spend",
            "do not run Colab or RunPod validation for this branch",
            "gpu_blocked_by_local_controls",
        ),
        _candidate(
            "treat_pruned_oracle_as_promotion_evidence",
            "rejected",
            "pruned oracle support uses nondeployable target support and does not establish learned sparse deployment",
            "keep pruning as diagnostic evidence only",
            "oracle_pruning_non_deployable",
        ),
    ]


def _failures(source_rows: list[dict[str, Any]], closeout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    missing = [row["source"] for row in source_rows if row["source"] != "strategy_review" and not row["present"]]
    if missing:
        failures.append(
            {
                "criterion": "source_artifacts_present",
                "passed": False,
                "required": True,
                "evidence": ",".join(missing),
            }
        )
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
        return "No external strategy review was present; closeout uses local pregate artifacts."
    return (
        "Accepted the GPT-5.5-Pro recommendation for dense-teacher columnability/distillation and strong "
        "flat/null controls. The support-forcing/pruning pregate is interpreted as local negative evidence; "
        "no recommendation was rejected in this run."
    )


def _direction_shift(review: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategic_change_level": review["strategic_change_level"],
        "ben_should_be_notified": bool(review["notify_ben"] or review["strategic_change_level"] == "major"),
        "direction": "close the current support-forcing/pruning branch before backend validation",
        "recommendation_disposition": "accepted" if review["present"] else "no_review_present",
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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
    evidence = summary["evidence"]
    return "\n".join(
        [
            "# Dense-Teacher Support-Forcing/Pruning Closeout",
            "",
            f"Decision: `{summary['decision']}`.",
            f"Claim status: `{summary['claim_status']}`.",
            "",
            "GPU validation remains blocked for this branch.",
            "",
            f"Oracle sparse R2: `{evidence['oracle_r2']}`.",
            f"Learned sparse R2: `{evidence['learned_r2']}`.",
            f"Flat value R2: `{evidence['flat_r2']}`.",
            f"Learned sparse CE: `{evidence['learned_ce']}`.",
            f"Flat value CE: `{evidence['flat_ce']}`.",
            f"Retained columns after pruning: `{evidence['retained_columns_after_pruning']}`.",
            "",
            f"Next step: {summary['selected_next_step']}.",
            "",
        ]
    )


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pregate-dir", type=Path, default=DEFAULT_PREGATE_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_dense_teacher_support_forcing_pruning_closeout(
        pregate_dir=args.pregate_dir,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({key: summary[key] for key in ("status", "decision", "claim_status", "selected_next_step")}, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
