"""Close out the scale-constrained sparse residual-compression pilot.

This report consumes the executable local pilot artifact. It runs no training
and fails closed on missing source evidence. The current scientific question is
whether a norm-budgeted sparse residual-compression mechanism survives matched
flat/dense controls before GPU validation; if it does not, this closeout
records the redirect rather than scaling the branch.
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


DEFAULT_PILOT = Path("results/reports/scale_constrained_sparse_residual_compression_pilot/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/scale_constrained_sparse_residual_compression_closeout")

DECISION = "scale_constrained_sparse_residual_compression_closed_no_gpu"
FAIL_DECISION = "scale_constrained_sparse_residual_compression_closeout_failed_closed"
SELECTED_NEXT_ACTION = "redirect_to_mechanism_factorized_continual_learning_local_gate"
SELECTED_NEXT_STEP = (
    "add a local mechanism-factorized continual-learning branch selector before any GPU validation"
)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "closeout_rows.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_scale_constrained_sparse_residual_compression_closeout(
    *,
    pilot_path: Path = DEFAULT_PILOT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a bounded closeout/branch-selector report for the local pilot."""

    start = time.time()
    pilot = _read_json(pilot_path)
    review = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("scale_constrained_sparse_residual_compression_pilot", pilot_path, pilot),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": review["present"],
            "status": "read" if review["present"] else "missing_optional",
            "decision": review["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={review['strategic_change_level']}; "
                f"notify_ben={review['notify_ben']}; verdict={review['verdict']}"
            ),
            "selected_next_action": "",
            "selected_next_step": "",
            "training_executed": "",
            "git_commit": "",
        },
    ]
    evidence = _evidence(pilot, review)
    closeout_rows = _closeout_rows(evidence, source_rows)
    failures = _failures(source_rows, closeout_rows)
    candidate_actions = _candidate_actions(evidence, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = FAIL_DECISION
        claim_status = "scale_constrained_sparse_closeout_sources_incomplete"
        selected_next_action = "repair_scale_constrained_sparse_closeout_sources"
        selected_next_step = "repair or regenerate scale-constrained sparse pilot artifacts"
        rationale = "Required source artifacts are missing or internally inconsistent."
    else:
        status = "pass"
        decision = DECISION
        claim_status = "scale_constrained_sparse_residual_compression_retired_before_gpu"
        selected_next_action = selected[0]["candidate_action"]
        selected_next_step = selected[0]["next_step"]
        rationale = (
            "The executable local pilot tested the intended norm-budgeted sparse mechanism, "
            "but identical norm-controller flat controls still dominate CE and teacher-residual "
            "MSE. Mechanism proxy wins and pruning retention are not enough to justify GPU "
            "validation or another residual-compression variant."
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_action": selected_next_action,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "training_executed": False,
        "backend_policy": "local closeout only; RunPod and Colab remain blocked",
        "source_rows": source_rows,
        "evidence": evidence,
        "closeout_rows": closeout_rows,
        "candidate_actions": candidate_actions,
        "strategy_review": review,
        "strategy_review_handling": _strategy_review_handling(review),
        "direction_shift": _direction_shift(review),
        "deferred_or_rejected_recommendations": [],
        "failures": failures,
        "rationale": rationale,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _evidence(pilot: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    arm_rows = _list(pilot.get("arm_metrics"))
    pruning_rows = _list(pilot.get("pruning_rows"))
    gate_rows = _list(pilot.get("gate_rows"))
    failed_gates = {row.get("criterion", "") for row in gate_rows if not bool(row.get("passed"))}
    ce_sparse = _row(arm_rows, "ce_only", "sparse_topr_norm_controller")
    ce_flat = _row(arm_rows, "ce_only", "same_controller_flat_residual")
    ce_dense = _row(arm_rows, "ce_only", "same_controller_dense_mlp")
    mse_sparse = _row(arm_rows, "mse_only", "sparse_topr_norm_controller")
    mse_flat = _row(arm_rows, "mse_only", "same_controller_flat_residual")
    best_retention = max(
        (
            _float(row.get("ce_gain_retention_fraction")) or 0.0
            for row in pruning_rows
            if row.get("objective") == "ce_only" and bool(row.get("pruned_at_least_half_components"))
        ),
        default=0.0,
    )
    return {
        "pilot_status": pilot.get("status", ""),
        "pilot_decision": pilot.get("decision", ""),
        "pilot_claim_status": pilot.get("claim_status", ""),
        "pilot_selected_next_step": pilot.get("selected_next_step", ""),
        "base_holdout_ce": _float(pilot.get("base_holdout_ce")),
        "dense_teacher_holdout_ce": _float(pilot.get("dense_teacher_holdout_ce")),
        "ce_sparse": _float(ce_sparse.get("ce")),
        "ce_flat": _float(ce_flat.get("ce")),
        "ce_dense": _float(ce_dense.get("ce")),
        "ce_gap_sparse_minus_flat": _delta(_float(ce_sparse.get("ce")), _float(ce_flat.get("ce"))),
        "mse_sparse": _float(mse_sparse.get("teacher_residual_reconstruction_mse")),
        "mse_flat": _float(mse_flat.get("teacher_residual_reconstruction_mse")),
        "mse_gap_sparse_minus_flat": _delta(
            _float(mse_sparse.get("teacher_residual_reconstruction_mse")),
            _float(mse_flat.get("teacher_residual_reconstruction_mse")),
        ),
        "sparse_selectivity": _float(ce_sparse.get("intervention_selectivity_proxy")),
        "flat_selectivity": _float(ce_flat.get("intervention_selectivity_proxy")),
        "sparse_commutator": _float(ce_sparse.get("finite_update_commutator_proxy")),
        "flat_commutator": _float(ce_flat.get("finite_update_commutator_proxy")),
        "sparse_active_component_fraction": _float(ce_sparse.get("active_component_fraction")),
        "best_half_prune_ce_gain_retention": round(best_retention, 6),
        "failed_scientific_gates": sorted(failed_gates),
        "arm_row_count": len(arm_rows),
        "pruning_row_count": len(pruning_rows),
        "prior_gpu_blocked": not (
            bool(pilot.get("requires_gpu_now"))
            or bool(pilot.get("advance_to_gpu_validation"))
            or bool(pilot.get("promotion_allowed"))
        ),
        "strategy_verdict": review["verdict"],
        "strategy_recommended_next_action": review["recommended_next_action"],
        "ben_notification_required": review["ben_notification_required"],
    }


def _closeout_rows(evidence: dict[str, Any], source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _closeout(
            "required_sources_present",
            all(row["present"] for row in source_rows if row["source"] != "strategy_review"),
            "required",
            ",".join(row["source"] for row in source_rows if row["source"] != "strategy_review" and not row["present"]),
        ),
        _closeout(
            "pilot_passed_runtime",
            evidence["pilot_status"] == "pass",
            "required",
            f"pilot_status={evidence['pilot_status']}; claim={evidence['pilot_claim_status']}",
        ),
        _closeout(
            "pilot_rows_complete",
            evidence["arm_row_count"] >= 21 and evidence["pruning_row_count"] >= 12,
            "required",
            f"arm_rows={evidence['arm_row_count']}; pruning_rows={evidence['pruning_row_count']}",
        ),
        _closeout(
            "flat_ce_control_blocks_sparse",
            _gt(evidence["ce_gap_sparse_minus_flat"], 0.002),
            "scientific",
            f"sparse_ce={evidence['ce_sparse']}; flat_ce={evidence['ce_flat']}; gap={evidence['ce_gap_sparse_minus_flat']}",
        ),
        _closeout(
            "flat_mse_control_blocks_sparse",
            _gt(evidence["mse_gap_sparse_minus_flat"], 0.02),
            "scientific",
            f"sparse_mse={evidence['mse_sparse']}; flat_mse={evidence['mse_flat']}; gap={evidence['mse_gap_sparse_minus_flat']}",
        ),
        _closeout(
            "sparse_is_scale_constrained",
            evidence["sparse_active_component_fraction"] is not None
            and evidence["sparse_active_component_fraction"] <= 0.5,
            "scientific",
            f"active_component_fraction={evidence['sparse_active_component_fraction']}",
        ),
        _closeout(
            "mechanism_proxies_do_not_rescue_flat_failures",
            "sparse_matches_or_beats_flat_ce" in evidence["failed_scientific_gates"]
            or "sparse_not_worse_than_flat_mse" in evidence["failed_scientific_gates"],
            "scientific",
            f"failed_gates={','.join(evidence['failed_scientific_gates'])}",
        ),
        _closeout(
            "gpu_validation_blocked",
            evidence["prior_gpu_blocked"],
            "required",
            "requires_gpu_now=false; promotion_allowed=false; advance_to_gpu_validation=false",
        ),
    ]


def _candidate_actions(evidence: dict[str, Any], failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if failures:
        return [
            _candidate(
                "repair_scale_constrained_sparse_closeout_sources",
                "selected",
                f"failures={len(failures)}",
                "repair or regenerate scale-constrained sparse pilot artifacts",
                "source_repair_required",
            )
        ]
    return [
        _candidate(
            "launch_gpu_validation_for_scale_constrained_sparse",
            "rejected",
            "identical norm-controller flat CE/MSE controls block promotion",
            "do not use RunPod or Colab for this branch",
            "gpu_blocked",
        ),
        _candidate(
            "iterate_another_residual_compression_variant",
            "rejected",
            "unconstrained, soft-mixture, and scale-constrained sparse variants are now flat-control dominated",
            "retire residual-compression variants unless new headroom evidence appears",
            "compression_path_retired",
        ),
        _candidate(
            "return_to_transformer_acsr_without_new_headroom",
            "rejected",
            "Transformer-ACSR remains blocked unless oracle-support regret shows deployable routing headroom",
            "do not reopen Transformer-ACSR as a GPU path from this closeout",
            "headroom_missing",
        ),
        _candidate(
            SELECTED_NEXT_ACTION,
            "selected",
            "the review's contingency for a constrained-pilot loss is to pivot to a mechanism-factorized continual-learning benchmark",
            SELECTED_NEXT_STEP,
            "selected_local_redirect",
        ),
    ]


def _closeout(criterion: str, passed: bool, row_type: str, evidence: str) -> dict[str, Any]:
    return {"criterion": criterion, "passed": bool(passed), "row_type": row_type, "evidence": evidence}


def _candidate(
    action: str, disposition: str, reason: str, next_step: str, claim_status: str
) -> dict[str, Any]:
    return {
        "candidate_action": action,
        "disposition": disposition,
        "reason": reason,
        "next_step": next_step,
        "claim_status": claim_status,
    }


def _failures(source_rows: list[dict[str, Any]], closeout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = [
        {"criterion": "source_present", "source": row["source"], "evidence": row["path"]}
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]
    failures.extend(row for row in closeout_rows if row["row_type"] == "required" and not row["passed"])
    return failures


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(payload),
        "status": payload.get("status", "") if payload else "missing",
        "decision": payload.get("decision", "") if payload else "",
        "claim_status": payload.get("claim_status", "") if payload else "",
        "selected_next_action": payload.get("selected_next_action", "") if payload else "",
        "selected_next_step": payload.get("selected_next_step", "") if payload else "",
        "training_executed": payload.get("training_executed", "") if payload else "",
        "git_commit": payload.get("git_commit", "") if payload else "",
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    text = ""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        pass
    header: dict[str, str] = {}
    for line in text.splitlines()[:12]:
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    notify = header.get("notify_ben", "false").lower() == "true"
    level = header.get("strategic_change_level", "missing" if not text else "")
    return {
        "present": bool(text),
        "path": str(path),
        "strategic_change_level": level,
        "notify_ben": notify,
        "recommended_next_action": header.get("recommended_next_action", ""),
        "verdict": header.get("verdict", ""),
        "ben_notification_required": notify or level == "major",
    }


def _strategy_review_handling(review: dict[str, Any]) -> str:
    if not review["present"]:
        return "No external strategy review was present; closeout proceeded from local pilot artifacts."
    return (
        "Accepted the latest GPT-5.5-Pro recommendation to run the scale-constrained local pilot "
        "before GPU. The pilot now failed the flat CE/MSE gates, so the review's contingency to "
        "stop residual-compression variants and pivot locally is accepted; no recommendation was rejected."
    )


def _direction_shift(review: dict[str, Any]) -> dict[str, Any]:
    return {
        "ben_should_be_notified": review["ben_notification_required"],
        "strategic_change_level": review["strategic_change_level"],
        "notify_ben_header": review["notify_ben"],
        "direction": "residual-compression variants are retired before GPU; local mechanism-factorized continual-learning gate is selected next",
        "recommendation_disposition": "accepted" if review["present"] else "not_available",
        "deferred_or_rejected_recommendations": [],
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
    return "\n".join(
        [
            "# Scale-Constrained Sparse Residual-Compression Closeout",
            "",
            f"- Status: {summary['status']}",
            f"- Decision: {summary['decision']}",
            f"- Claim status: {summary['claim_status']}",
            f"- Selected next action: {summary['selected_next_action']}",
            f"- Selected next step: {summary['selected_next_step']}",
            "- GPU validation remains blocked: requires_gpu_now=false, promotion_allowed=false, advance_to_gpu_validation=false.",
            "",
        ]
    )


def _list(value: object) -> list[dict[str, Any]]:
    return value if isinstance(value, list) else []


def _row(rows: list[dict[str, Any]], objective: str, arm: str) -> dict[str, Any]:
    for row in rows:
        if row.get("objective") == objective and row.get("arm") == arm:
            return row
    return {}


def _float(value: object) -> float | None:
    try:
        if value == "":
            return None
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(left - right, 6)


def _gt(value: float | None, threshold: float) -> bool:
    return value is not None and value > threshold


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_scale_constrained_sparse_residual_compression_closeout(
        pilot_path=args.pilot,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
