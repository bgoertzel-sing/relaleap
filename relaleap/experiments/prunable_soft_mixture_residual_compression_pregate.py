"""Design the prunable soft-mixture residual-compression pregate.

This is a local design artifact, not a training run. It consumes the
continuous-coefficient closeout and records the exact no-GPU contract for the
next bounded pilot: train soft residual mixtures only with same-objective
flat/dense controls, explicit pruning sweeps, and fail-closed promotion gates.
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


DEFAULT_CLOSEOUT = Path("results/reports/continuous_coefficient_closeout/summary.json")
DEFAULT_ADJUDICATOR = Path("results/reports/continuous_coefficient_ce_mse_discordance_adjudicator/summary.json")
DEFAULT_PREGATE = Path("results/reports/continuous_coefficient_sparse_value_pregate/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/prunable_soft_mixture_residual_compression_pregate")

DECISION = "prunable_soft_mixture_residual_compression_pregate_recorded"
FAIL_DECISION = "prunable_soft_mixture_residual_compression_pregate_failed_closed"
SELECTED_NEXT_ACTION = "implement_prunable_soft_mixture_residual_compression_pilot"
REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "design_constraints.csv",
    "pregate_arms.csv",
    "advancement_gates.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_prunable_soft_mixture_residual_compression_pregate(
    *,
    closeout_path: Path = DEFAULT_CLOSEOUT,
    adjudicator_path: Path = DEFAULT_ADJUDICATOR,
    pregate_path: Path = DEFAULT_PREGATE,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Record a fail-closed local pregate for soft residual compression."""

    start = time.time()
    closeout = _read_json(closeout_path)
    adjudicator = _read_json(adjudicator_path)
    pregate = _read_json(pregate_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source("continuous_coefficient_closeout", closeout_path, closeout),
        _source("continuous_coefficient_ce_mse_discordance_adjudicator", adjudicator_path, adjudicator),
        _source("continuous_coefficient_sparse_value_pregate", pregate_path, pregate),
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
            "training_executed": "",
        },
    ]
    evidence = _evidence(closeout, adjudicator, pregate, strategy)
    design_constraints = _design_constraints(evidence)
    pregate_arms = _pregate_arms(evidence)
    advancement_gates = _advancement_gates(evidence)
    failures = _failures(source_rows, advancement_gates)
    status = "pass" if not failures else "fail"
    selected_next_action = SELECTED_NEXT_ACTION if status == "pass" else "repair_soft_mixture_pregate_sources"
    selected_next_step = (
        "implement a local prunable soft-mixture residual-compression pilot with same-objective flat/dense controls"
        if status == "pass"
        else "repair continuous-coefficient closeout/adjudicator artifacts before soft-mixture implementation"
    )
    summary = {
        "status": status,
        "decision": DECISION if status == "pass" else FAIL_DECISION,
        "claim_status": "design_only_prunable_soft_mixture_no_gpu_claim",
        "selected_next_action": selected_next_action,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "training_executed": False,
        "backend_policy": "local pregate design only; RunPod and Colab remain blocked",
        "source_rows": source_rows,
        "evidence": evidence,
        "design_constraints": design_constraints,
        "pregate_arms": pregate_arms,
        "advancement_gates": advancement_gates,
        "candidate_actions": _candidate_actions(selected_next_action),
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "direction_shift": _direction_shift(strategy),
        "deferred_or_rejected_recommendations": [],
        "failures": failures,
        "rationale": _rationale(status, evidence),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _evidence(
    closeout: dict[str, Any],
    adjudicator: dict[str, Any],
    pregate: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    closeout_evidence = closeout.get("evidence", {}) if isinstance(closeout.get("evidence"), dict) else {}
    failed_gates = closeout_evidence.get("adjudicator_failed_gates", [])
    return {
        "closeout_status": closeout.get("status", ""),
        "closeout_decision": closeout.get("decision", ""),
        "closeout_claim_status": closeout.get("claim_status", ""),
        "closeout_selected_next_action": closeout.get("selected_next_action", ""),
        "closeout_selected_next_step": closeout.get("selected_next_step", ""),
        "adjudicator_status": adjudicator.get("status", ""),
        "adjudicator_claim_status": adjudicator.get("claim_status", ""),
        "pregate_status": pregate.get("status", ""),
        "pregate_claim_status": pregate.get("claim_status", ""),
        "continuous_pregate_ce": _float(closeout_evidence.get("continuous_pregate_ce")),
        "flat_pregate_ce": _float(closeout_evidence.get("flat_pregate_ce")),
        "continuous_pregate_mse": _float(closeout_evidence.get("continuous_pregate_mse")),
        "flat_pregate_mse": _float(closeout_evidence.get("flat_pregate_mse")),
        "adjudicator_ce_continuous": _float(closeout_evidence.get("adjudicator_ce_continuous")),
        "adjudicator_ce_flat": _float(closeout_evidence.get("adjudicator_ce_flat")),
        "adjudicator_mse_continuous": _float(closeout_evidence.get("adjudicator_mse_continuous")),
        "adjudicator_mse_flat": _float(closeout_evidence.get("adjudicator_mse_flat")),
        "coeff_near_zero_fraction_min": _float(closeout_evidence.get("coeff_near_zero_fraction_min")),
        "dense_like_coefficients": bool(closeout_evidence.get("dense_like_coefficients")),
        "adjudicator_failed_gates": failed_gates if isinstance(failed_gates, list) else [],
        "prior_gpu_blocked": not (
            bool(closeout.get("requires_gpu_now"))
            or bool(closeout.get("advance_to_gpu_validation"))
            or bool(closeout.get("promotion_allowed"))
        ),
        "strategy_verdict": strategy["verdict"],
        "strategy_recommended_next_action": strategy["recommended_next_action"],
        "ben_notification_required": strategy["ben_notification_required"],
    }


def _design_constraints(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _constraint(
            "same_objective_controls",
            "mandatory",
            "Every soft-mixture candidate must be trained under CE-only, MSE-only, and CE+MSE objectives with matched flat and dense controls.",
            "continuous coefficients were objective-parity and flat-control confounded",
        ),
        _constraint(
            "soft_before_hard_support",
            "mandatory",
            "Use differentiable soft mixture weights during training, then run post-training pruning sweeps before any sparse claim.",
            "hard sparse dictionaries failed and unconstrained coefficients became dense-like",
        ),
        _constraint(
            "explicit_pruning_axis",
            "mandatory",
            "Report unpruned, entropy/L1 regularized, top-r pruned, and threshold-pruned variants from the same trained student.",
            "the next claim is compression and pruning tolerance, not raw CE",
        ),
        _constraint(
            "flat_dense_norm_budget",
            "mandatory",
            "Match residual norm and active parameter/FLOP budgets against same-router flat and dense-teacher residual controls.",
            "scale and capacity confounds remain plausible explanations",
        ),
        _constraint(
            "mechanism_observables",
            "mandatory",
            "Emit mixture entropy, effective component count, prune-retention curve, intervention selectivity, churn, and commutator proxies.",
            "CE/perplexity are guardrails rather than the main scientific signal",
        ),
        _constraint(
            "gpu_block",
            "mandatory",
            "Do not use RunPod or Colab until the local pilot beats flat/dense controls on the advancement gates.",
            "current branch is design-only and prior GPU validation is scientifically blocked",
        ),
    ]


def _pregate_arms(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _arm(
            "dense_teacher_residual_reference",
            "dense_control",
            False,
            "CE/MSE teacher residual endpoint; nondeployable reference only",
            "base_ce; teacher_ce; teacher_residual_mse=0; residual_norm",
        ),
        _arm(
            "same_objective_flat_value_control",
            "flat_control",
            True,
            "matched flat residual head trained under each objective",
            "CE; teacher_residual_mse; residual_norm; churn; commutator_proxy",
        ),
        _arm(
            "soft_mixture_residual_unpruned",
            "soft_mixture_candidate",
            True,
            "dense soft mixture with temperature schedule but no hard pruning",
            "CE; MSE; mixture_entropy; effective_component_count; support_load_entropy",
        ),
        _arm(
            "prunable_soft_mixture_entropy_l1",
            "soft_mixture_candidate",
            True,
            "soft mixture with entropy/L1 pressure and residual norm controller",
            "same metrics plus coefficient_near_zero_fraction and prune curve",
        ),
        _arm(
            "pruned_soft_mixture_topr_sweep",
            "posthoc_pruned_candidate",
            False,
            "top-r and threshold pruning sweep derived from the trained soft mixture",
            "CE_retention; MSE_retention; intervention_selectivity_retention; active_component_fraction",
        ),
        _arm(
            "shuffled_target_soft_mixture_null",
            "null_control",
            True,
            "same architecture trained on shuffled residual targets",
            "same metrics as candidate; must not pass mechanism gates",
        ),
        _arm(
            "scale_only_residual_null",
            "null_control",
            True,
            "single scalar residual scale control matched to teacher norm",
            "CE; MSE; residual_norm; verifies gains are not amplitude-only",
        ),
    ]


def _advancement_gates(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _gate(
            "closeout_selected_soft_mixture_pregate",
            evidence["closeout_status"] == "pass"
            and evidence["closeout_selected_next_action"] == "design_prunable_soft_mixture_residual_compression_pregate",
            "required",
            "continuous closeout must explicitly select this pregate",
            evidence["closeout_selected_next_action"],
        ),
        _gate(
            "adjudicator_and_pregate_sources_passed",
            evidence["adjudicator_status"] == "pass" and evidence["pregate_status"] == "pass",
            "required",
            "source adjudicator and continuous pregate must pass",
            f"adjudicator={evidence['adjudicator_status']}; pregate={evidence['pregate_status']}",
        ),
        _gate(
            "prior_gpu_validation_blocked",
            evidence["prior_gpu_blocked"],
            "required",
            "this report must not reopen GPU validation",
            f"prior_gpu_blocked={evidence['prior_gpu_blocked']}",
        ),
        _gate(
            "soft_mixture_must_beat_flat_ce_same_objective",
            False,
            "future_scientific",
            "pilot advancement requires prunable soft mixture CE < same-objective flat CE by >=0.002 after norm matching",
            "not evaluated in design-only pregate",
        ),
        _gate(
            "soft_mixture_must_not_lose_flat_mse",
            False,
            "future_scientific",
            "pilot advancement requires teacher-residual MSE <= same-objective flat MSE after norm matching",
            "not evaluated in design-only pregate",
        ),
        _gate(
            "pruning_retains_function",
            False,
            "future_scientific",
            "pilot advancement requires >=80% CE gain retention after pruning at least half of active mixture components",
            "not evaluated in design-only pregate",
        ),
        _gate(
            "mechanism_metrics_improve",
            False,
            "future_scientific",
            "pilot advancement requires better intervention selectivity, churn, or commutator proxy than flat/dense controls",
            "not evaluated in design-only pregate",
        ),
    ]


def _candidate_actions(selected: str) -> list[dict[str, str]]:
    rows = []
    if selected == "repair_soft_mixture_pregate_sources":
        rows.append(
            _candidate(
                "repair_soft_mixture_pregate_sources",
                "selected",
                "Required closeout/adjudicator sources are missing or inconsistent.",
                "repair source artifacts before implementation",
            )
        )
    rows.extend(
        [
            _candidate(
                SELECTED_NEXT_ACTION,
                "selected" if selected == SELECTED_NEXT_ACTION else "blocked",
                "The local design contract is complete and keeps GPU validation blocked.",
                "implement the local pilot with the recorded arms and gates",
            ),
            _candidate(
                "run_runpod_or_colab_validation",
                "rejected",
                "No prunable soft-mixture pilot has passed local flat/dense controls.",
                "keep GPU blocked",
            ),
            _candidate(
                "reopen_unconstrained_continuous_coefficients",
                "rejected",
                "Prior closeout retired unconstrained coefficients due to flat-control and dense-like usage failures.",
                "do not reopen without sparse/scale constraints",
            ),
            _candidate(
                "sparse_scale_constrained_coefficients",
                "deferred",
                "May become useful after a stronger soft-mixture compression baseline is established.",
                "keep as fallback",
            ),
        ]
    )
    return rows


def _failures(source_rows: list[dict[str, Any]], gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = [
        {"criterion": "source_present", "source": row["source"], "evidence": row["path"]}
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]
    failures.extend(row for row in gates if row["gate_type"] == "required" and not row["passed"])
    return failures


def _rationale(status: str, evidence: dict[str, Any]) -> str:
    if status != "pass":
        return "The soft-mixture pregate fails closed because required continuous-coefficient source artifacts are missing or inconsistent."
    return (
        "Unconstrained continuous coefficients are closed before GPU because the CE signal was flat-control/objective "
        "confounded, teacher-MSE parity failed, and coefficient use was dense-like. The next local mechanism should "
        "therefore test prunable soft residual mixtures directly, with same-objective flat/dense controls and pruning "
        "retention gates before any scaling."
    )


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    level = _header_value(text, "strategic_change_level") or ("missing" if not text else "")
    notify = (_header_value(text, "notify_ben") or "false").lower() == "true"
    return {
        "path": str(path),
        "present": bool(text),
        "strategic_change_level": level,
        "notify_ben": notify,
        "recommended_next_action": _header_value(text, "recommended_next_action"),
        "verdict": _header_value(text, "verdict"),
        "ben_notification_required": notify or level == "major",
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No strategy review was present; the pregate proceeded from local closeout artifacts only."
    return (
        "Accepted the latest GPT-5.5-Pro FIX recommendation as already satisfied by the continuous-coefficient "
        "adjudicator and closeout; this report continues the selected no-GPU soft-mixture pregate step."
    )


def _direction_shift(strategy: dict[str, Any]) -> dict[str, Any]:
    return {
        "ben_should_be_notified": strategy["ben_notification_required"],
        "strategic_change_level": strategy["strategic_change_level"],
        "notify_ben_header": strategy["notify_ben"],
        "direction": "local prunable soft-mixture residual compression pregate; no GPU validation",
        "recommendation_disposition": "accepted" if strategy["present"] else "not_available",
    }


def _source(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file() and bool(payload),
        "status": payload.get("status", "missing" if not path.is_file() else ""),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
        "selected_next_action": payload.get("selected_next_action", ""),
        "selected_next_step": payload.get("selected_next_step", ""),
        "training_executed": payload.get("training_executed", ""),
    }


def _constraint(name: str, priority: str, requirement: str, rationale: str) -> dict[str, str]:
    return {"constraint": name, "priority": priority, "requirement": requirement, "rationale": rationale}


def _arm(arm: str, family: str, trainable: bool, role: str, required_outputs: str) -> dict[str, Any]:
    return {
        "arm": arm,
        "family": family,
        "trainable": trainable,
        "role": role,
        "required_outputs": required_outputs,
    }


def _gate(criterion: str, passed: bool, gate_type: str, threshold: str, actual: Any) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "gate_type": gate_type,
        "threshold": threshold,
        "actual": actual,
        "failure_reason": "" if passed or gate_type == "future_scientific" else "required pregate condition failed",
    }


def _candidate(action: str, disposition: str, reason: str, next_step: str) -> dict[str, str]:
    return {"candidate_action": action, "disposition": disposition, "reason": reason, "next_step": next_step}


def _header_value(text: str, key: str) -> str:
    prefix = f"{key}:"
    for line in text.splitlines()[:20]:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "design_constraints.csv", summary["design_constraints"])
    _write_csv(out_dir / "pregate_arms.csv", summary["pregate_arms"])
    _write_csv(out_dir / "advancement_gates.csv", summary["advancement_gates"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _notes(summary: dict[str, Any]) -> str:
    lines = [
        "# Prunable Soft-Mixture Residual-Compression Pregate",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Selected next step: `{summary['selected_next_step']}`",
        "- GPU validation remains blocked.",
        "",
        summary["rationale"],
    ]
    if summary["failures"]:
        lines.extend(["", "## Failed Criteria"])
        lines.extend(f"- `{row['criterion']}`" for row in summary["failures"])
    return "\n".join(lines) + "\n"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closeout", type=Path, default=DEFAULT_CLOSEOUT)
    parser.add_argument("--adjudicator", type=Path, default=DEFAULT_ADJUDICATOR)
    parser.add_argument("--pregate", type=Path, default=DEFAULT_PREGATE)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_prunable_soft_mixture_residual_compression_pregate(
        closeout_path=args.closeout,
        adjudicator_path=args.adjudicator,
        pregate_path=args.pregate,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
