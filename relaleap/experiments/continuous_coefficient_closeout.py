"""Close the unconstrained continuous-coefficient sparse-value branch.

This report consumes the continuous-coefficient pregate and CE/MSE
adjudicator artifacts. It is a decision artifact only: no training is run, and
GPU validation remains blocked when same-objective flat controls or dense-like
coefficients explain the local CE signal.
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


DEFAULT_PREGATE = Path("results/reports/continuous_coefficient_sparse_value_pregate/summary.json")
DEFAULT_ADJUDICATOR = Path("results/reports/continuous_coefficient_ce_mse_discordance_adjudicator/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/continuous_coefficient_closeout")

DECISION = "continuous_coefficient_branch_closed_no_gpu"
FAIL_DECISION = "continuous_coefficient_closeout_failed_closed"
SELECTED_NEXT_ACTION = "design_prunable_soft_mixture_residual_compression_pregate"
SELECTED_NEXT_STEP = (
    "design a local prunable soft-mixture residual compression pregate with same-objective flat/dense controls before GPU"
)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "closeout_rows.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_continuous_coefficient_closeout(
    *,
    pregate_path: Path = DEFAULT_PREGATE,
    adjudicator_path: Path = DEFAULT_ADJUDICATOR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed closeout for unconstrained continuous coefficients."""

    start = time.time()
    pregate = _read_json(pregate_path)
    adjudicator = _read_json(adjudicator_path)
    review = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("continuous_coefficient_sparse_value_pregate", pregate_path, pregate),
        _source_row("continuous_coefficient_ce_mse_discordance_adjudicator", adjudicator_path, adjudicator),
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
            "selected_next_step": "",
            "training_executed": "",
            "git_commit": "",
        },
    ]
    evidence = _evidence(pregate, adjudicator, review)
    closeout_rows = _closeout_rows(evidence, source_rows)
    failures = _failures(source_rows, closeout_rows)
    candidate_actions = _candidate_actions(evidence, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = FAIL_DECISION
        claim_status = "continuous_coefficient_closeout_sources_incomplete"
        selected_next_action = "repair_continuous_coefficient_closeout_sources"
        selected_next_step = "repair or regenerate continuous-coefficient pregate/adjudicator source artifacts"
        rationale = "Required source artifacts are missing or the flat-control/dense-coefficient closure criteria are not satisfied."
    else:
        status = "pass"
        decision = DECISION
        claim_status = "unconstrained_continuous_coefficients_retired_before_gpu"
        selected_next_action = selected[0]["candidate_action"]
        selected_next_step = selected[0]["next_step"]
        rationale = (
            "The pregate's CE-positive result is non-promotional because it loses teacher-residual MSE to "
            "the same-router flat value control, coefficients remain dense-like, and the adjudicator shows "
            "same-objective flat controls block objective-parity promotion. The branch should close before "
            "GPU validation rather than be scaled."
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
        "failures": failures,
        "rationale": rationale,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _evidence(pregate: dict[str, Any], adjudicator: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    pregate_arms = {row.get("arm", ""): row for row in _list(pregate.get("arm_metrics"))}
    continuous = pregate_arms.get("continuous_coeff_learned_support", {})
    flat = pregate_arms.get("same_router_flat_value_control", {})
    oracle = pregate_arms.get("continuous_coeff_oracle_support_ceiling", {})
    adjudication_rows = _adjudication_rows(adjudicator)
    gate_rows = _list(adjudicator.get("gate_rows"))
    failed_gates = {row.get("criterion", "") for row in gate_rows if not bool(row.get("passed"))}
    ce_pair = _row_pair(adjudication_rows, "ce")
    mse_pair = _row_pair(adjudication_rows, "mse")
    combined_pair = _row_pair(adjudication_rows, "ce_mse")
    coeff_near_zero = _min_coeff_near_zero(adjudicator, adjudication_rows, pregate)
    return {
        "pregate_status": pregate.get("status", ""),
        "pregate_decision": pregate.get("decision", ""),
        "pregate_claim_status": pregate.get("claim_status", ""),
        "adjudicator_status": adjudicator.get("status", ""),
        "adjudicator_decision": adjudicator.get("decision", ""),
        "adjudicator_claim_status": adjudicator.get("claim_status", ""),
        "base_holdout_ce": _float(_coalesce(pregate.get("base_holdout_ce"), adjudicator.get("base_holdout_ce"))),
        "dense_teacher_holdout_ce": _float(
            _coalesce(pregate.get("dense_teacher_holdout_ce"), adjudicator.get("dense_teacher_holdout_ce"))
        ),
        "continuous_pregate_ce": _float(continuous.get("ce")),
        "flat_pregate_ce": _float(flat.get("ce")),
        "oracle_pregate_ce": _float(oracle.get("ce")),
        "continuous_pregate_mse": _float(continuous.get("teacher_residual_reconstruction_mse")),
        "flat_pregate_mse": _float(flat.get("teacher_residual_reconstruction_mse")),
        "oracle_pregate_mse": _float(oracle.get("teacher_residual_reconstruction_mse")),
        "pregate_ce_gap_vs_flat": _delta(_float(continuous.get("ce")), _float(flat.get("ce"))),
        "pregate_mse_gap_vs_flat": _delta(
            _float(continuous.get("teacher_residual_reconstruction_mse")),
            _float(flat.get("teacher_residual_reconstruction_mse")),
        ),
        "pregate_ce_mse_discordant": bool(pregate.get("ce_mse_discordant"))
        or pregate.get("claim_status") == "continuous_coeff_ce_mse_discordant_no_promotion",
        "adjudicator_same_objective_flat_controls_present": bool(
            adjudicator.get("same_objective_flat_controls_present", True)
        ),
        "adjudicator_ce_continuous": _float(ce_pair[0].get("ce")) if ce_pair else None,
        "adjudicator_ce_flat": _float(ce_pair[1].get("ce")) if ce_pair else None,
        "adjudicator_mse_continuous": _float(mse_pair[0].get("teacher_residual_reconstruction_mse")) if mse_pair else None,
        "adjudicator_mse_flat": _float(mse_pair[1].get("teacher_residual_reconstruction_mse")) if mse_pair else None,
        "adjudicator_combined_continuous_ce": _float(combined_pair[0].get("ce")) if combined_pair else None,
        "adjudicator_combined_flat_ce": _float(combined_pair[1].get("ce")) if combined_pair else None,
        "adjudicator_combined_continuous_mse": _float(
            combined_pair[0].get("teacher_residual_reconstruction_mse")
        )
        if combined_pair
        else None,
        "adjudicator_combined_flat_mse": _float(combined_pair[1].get("teacher_residual_reconstruction_mse"))
        if combined_pair
        else None,
        "adjudicator_failed_gates": sorted(failed_gates),
        "coeff_near_zero_fraction_min": coeff_near_zero,
        "dense_like_coefficients": coeff_near_zero is not None and coeff_near_zero < 0.05,
        "oracle_support_non_deployable": bool(oracle.get("oracle_support_non_deployable", True)),
        "requires_gpu_now": bool(pregate.get("requires_gpu_now")) or bool(adjudicator.get("requires_gpu_now")),
        "advance_to_gpu_validation": bool(pregate.get("advance_to_gpu_validation"))
        or bool(adjudicator.get("advance_to_gpu_validation")),
        "promotion_allowed": bool(pregate.get("promotion_allowed")) or bool(adjudicator.get("promotion_allowed")),
        "strategy_verdict": review["verdict"],
        "strategy_recommended_next_action": review["recommended_next_action"],
        "ben_notification_required": review["ben_notification_required"],
    }


def _adjudication_rows(adjudicator: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _list(adjudicator.get("objective_rows"))
    if rows:
        return rows
    return _list(adjudicator.get("adjudication_rows"))


def _row_pair(rows: list[dict[str, Any]], objective: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    continuous: dict[str, Any] | None = None
    flat: dict[str, Any] | None = None
    for row in rows:
        if _objective_key(row.get("objective", "")) != objective:
            continue
        variant = row.get("variant", row.get("residual_variant", ""))
        if variant not in ("norm_matched", ""):
            continue
        family = row.get("family", "")
        arm = row.get("arm", "")
        if family == "continuous_coeff" or arm in {f"{objective}_continuous_coeff_norm_matched", "continuous_coeff_learned_support"}:
            continuous = row
        if family == "same_router_flat" or arm in {f"{objective}_same_router_flat_norm_matched", "same_objective_flat_value_control"}:
            flat = row
    if continuous is not None and flat is not None:
        return continuous, flat
    return None


def _objective_key(value: object) -> str:
    text = str(value)
    if text in {"ce_only", "ce"}:
        return "ce"
    if text in {"mse_only", "mse"}:
        return "mse"
    if text in {"ce_mse_combined", "ce_mse", "combined"}:
        return "ce_mse"
    return text


def _min_coeff_near_zero(
    adjudicator: dict[str, Any], adjudication_rows: list[dict[str, Any]], pregate: dict[str, Any]
) -> float | None:
    values = []
    for row in _list(adjudicator.get("coefficient_rows")) + adjudication_rows + _list(pregate.get("coefficient_rows")):
        value = _float(row.get("coeff_near_zero_fraction", row.get("coefficient_near_zero_fraction")))
        if value is not None:
            values.append(value)
    return min(values) if values else None


def _closeout_rows(evidence: dict[str, Any], source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _closeout(
            "required_sources_present",
            not any(row["source"] != "strategy_review" and not row["present"] for row in source_rows),
            "required",
            ",".join(row["source"] for row in source_rows if row["source"] != "strategy_review" and not row["present"]),
        ),
        _closeout(
            "pregate_was_ce_mse_discordant",
            evidence["pregate_status"] == "pass" and evidence["pregate_ce_mse_discordant"],
            "required",
            f"pregate_status={evidence['pregate_status']}; claim={evidence['pregate_claim_status']}",
        ),
        _closeout(
            "adjudicator_passed_runtime",
            evidence["adjudicator_status"] == "pass",
            "required",
            f"adjudicator_status={evidence['adjudicator_status']}; claim={evidence['adjudicator_claim_status']}",
        ),
        _closeout(
            "same_objective_flat_controls_present",
            evidence["adjudicator_same_objective_flat_controls_present"],
            "required",
            "adjudicator includes same-objective flat controls",
        ),
        _closeout(
            "pregate_flat_mse_dominates_continuous",
            _gt(evidence["pregate_mse_gap_vs_flat"], 0.10),
            "scientific",
            (
                f"continuous_mse={evidence['continuous_pregate_mse']}; "
                f"flat_mse={evidence['flat_pregate_mse']}; gap={evidence['pregate_mse_gap_vs_flat']}"
            ),
        ),
        _closeout(
            "adjudicator_ce_objective_not_promotion_safe",
            not _lt(evidence["adjudicator_ce_continuous"], evidence["adjudicator_ce_flat"], margin=0.002),
            "scientific",
            f"continuous_ce={evidence['adjudicator_ce_continuous']}; flat_ce={evidence['adjudicator_ce_flat']}",
        ),
        _closeout(
            "adjudicator_mse_objective_flat_dominates",
            _gt(_delta(evidence["adjudicator_mse_continuous"], evidence["adjudicator_mse_flat"]), 0.10),
            "scientific",
            f"continuous_mse={evidence['adjudicator_mse_continuous']}; flat_mse={evidence['adjudicator_mse_flat']}",
        ),
        _closeout(
            "combined_objective_not_clean_continuous_win",
            not (
                _lt(evidence["adjudicator_combined_continuous_ce"], evidence["adjudicator_combined_flat_ce"], margin=0.002)
                and _lt(
                    evidence["adjudicator_combined_continuous_mse"],
                    evidence["adjudicator_combined_flat_mse"],
                    margin=-0.10,
                )
            ),
            "scientific",
            (
                f"continuous_ce={evidence['adjudicator_combined_continuous_ce']}; "
                f"flat_ce={evidence['adjudicator_combined_flat_ce']}; "
                f"continuous_mse={evidence['adjudicator_combined_continuous_mse']}; "
                f"flat_mse={evidence['adjudicator_combined_flat_mse']}"
            ),
        ),
        _closeout(
            "coefficients_dense_like",
            evidence["dense_like_coefficients"],
            "scientific",
            f"min_coeff_near_zero_fraction={evidence['coeff_near_zero_fraction_min']}",
        ),
        _closeout(
            "gpu_validation_blocked",
            not evidence["requires_gpu_now"]
            and not evidence["advance_to_gpu_validation"]
            and not evidence["promotion_allowed"],
            "required",
            (
                f"requires_gpu_now={evidence['requires_gpu_now']}; "
                f"advance_to_gpu_validation={evidence['advance_to_gpu_validation']}; "
                f"promotion_allowed={evidence['promotion_allowed']}"
            ),
        ),
    ]


def _candidate_actions(evidence: dict[str, Any], failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if failures:
        return [
            _candidate(
                "repair_continuous_coefficient_closeout_sources",
                "selected",
                f"failures={len(failures)}",
                "repair or regenerate continuous-coefficient source artifacts",
                "source_repair_required",
            )
        ]
    return [
        _candidate(
            "continue_unconstrained_continuous_coefficients",
            "rejected",
            "same-objective flat controls and dense coefficients block promotion",
            "do not continue unconstrained continuous coefficients",
            "blocked_by_flat_control_and_dense_coefficients",
        ),
        _candidate(
            "launch_gpu_validation_for_continuous_coefficients",
            "rejected",
            "local objective-parity and sparsity gates failed",
            "do not use RunPod or Colab for this branch",
            "gpu_blocked",
        ),
        _candidate(
            "redesign_sparse_scale_constrained_coefficients",
            "deferred",
            "could be revisited, but current evidence points to dense value generation hidden behind sparse support",
            "keep as fallback after a stronger compression baseline",
            "fallback_only",
        ),
        _candidate(
            SELECTED_NEXT_ACTION,
            "selected",
            "unconstrained coefficients behave like a dense adapter; next local mechanism should test prunable compression directly",
            SELECTED_NEXT_STEP,
            "selected_local_mechanism",
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
        return "No strategy review was present; closeout proceeded from local source artifacts only."
    return (
        "Accepted the latest GPT-5.5-Pro recommendation to keep the CE/MSE discordance non-promotional, "
        "block GPU validation, and close/adjudicate continuous coefficients locally before choosing a new mechanism."
    )


def _direction_shift(review: dict[str, Any]) -> dict[str, Any]:
    return {
        "ben_should_be_notified": review["ben_notification_required"],
        "strategic_change_level": review["strategic_change_level"],
        "notify_ben_header": review["notify_ben"],
        "direction": "unconstrained continuous coefficients are retired; local prunable soft-mixture compression is selected next",
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
            "# Continuous-Coefficient Closeout",
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


def _float(value: object) -> float | None:
    try:
        if value == "":
            return None
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _coalesce(*values: object) -> object:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(left - right, 6)


def _gt(value: float | None, threshold: float) -> bool:
    return value is not None and value > threshold


def _lt(left: float | None, right: float | None, *, margin: float) -> bool:
    if left is None or right is None:
        return False
    return left + margin < right


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pregate", type=Path, default=DEFAULT_PREGATE)
    parser.add_argument("--adjudicator", type=Path, default=DEFAULT_ADJUDICATOR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_continuous_coefficient_closeout(
        pregate_path=args.pregate,
        adjudicator_path=args.adjudicator,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {key: summary[key] for key in ("status", "decision", "claim_status", "selected_next_step")},
            indent=2,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
