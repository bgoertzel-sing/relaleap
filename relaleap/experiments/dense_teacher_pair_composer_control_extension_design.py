"""Design the local control extension for dense-teacher pair composition."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_TRUTH_AUDIT = Path("results/reports/dense_teacher_pair_composer_truth_audit/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_pair_composer_control_extension_design")

SELECTED_ACTION = "implement_pair_composer_control_extension_probe_locally"
REPAIR_ACTION = "repair_pair_composer_control_extension_sources"
TRUTH_AUDIT_EXTENSION_ACTION = "extend_pair_composer_truth_audit_controls_before_gpu"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "control_extension_contract.csv",
    "probe_arms.csv",
    "gate_criteria.csv",
    "candidate_actions.csv",
    "notes.md",
)

REQUIRED_CONTROL_FIELDS = (
    "learned_causal_pair_router",
    "delayed_misaligned_leakage_sentinels",
    "norm_utilization_and_direction_error",
    "functional_churn_and_retention",
    "finite_update_commutator",
    "matched_dense_mlp_interference_controls",
)

REQUIRED_PROBE_ARMS = (
    "oracle_pair_composer_reference",
    "learned_causal_pair_router",
    "delayed_pair_target_null",
    "misaligned_support_pair_null",
    "token_position_pair_router_null",
    "feature_count_shuffled_pair_null",
    "same_param_independent_value_composer",
    "rank_norm_matched_dense_residual",
    "matched_mlp_residual",
)


def run_dense_teacher_pair_composer_control_extension_design(
    *,
    truth_audit_path: Path = DEFAULT_TRUTH_AUDIT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed local probe contract for the missing pair-composer controls."""

    start = time.time()
    truth_audit = _read_json(truth_audit_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_json("dense_teacher_pair_composer_truth_audit", truth_audit_path, truth_audit),
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
    evidence = _evidence(truth_audit, strategy)
    control_contract = _control_contract()
    probe_arms = _probe_arms()
    gate_criteria = _gate_criteria(evidence, source_rows, control_contract, probe_arms)
    failures = [row for row in gate_criteria if row["passed"] is False]
    candidate_actions = _candidate_actions(failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "dense_teacher_pair_composer_control_extension_design_failed_closed"
        claim_status = "pair_composer_control_extension_sources_or_contract_incomplete"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair pair-composer truth-audit source or control-extension contract"
        rationale = "The control-extension design cannot safely open a local probe until its source and required controls are coherent."
    else:
        status = "pass"
        selected_row = selected[0]
        decision = "dense_teacher_pair_composer_control_extension_design_recorded"
        claim_status = "design_only_pair_composer_controls_not_yet_evidence"
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
        "backend_policy": "local design/probe only; RunPod and Colab remain blocked",
        "source_rows": source_rows,
        "evidence": evidence,
        "control_extension_contract": control_contract,
        "probe_arms": probe_arms,
        "gate_criteria": gate_criteria,
        "candidate_actions": candidate_actions,
        "strategy_review": strategy,
        "strategy_response": _strategy_response(strategy),
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


def _evidence(truth_audit: dict[str, Any], strategy: dict[str, Any]) -> dict[str, Any]:
    pair_metrics = truth_audit.get("pair_metrics", {}) if isinstance(truth_audit.get("pair_metrics"), dict) else {}
    return {
        "truth_audit_status": truth_audit.get("status"),
        "truth_audit_decision": truth_audit.get("decision"),
        "truth_audit_claim_status": truth_audit.get("claim_status"),
        "truth_audit_selected_next_action": truth_audit.get("selected_next_action"),
        "truth_audit_advance_to_gpu_validation": truth_audit.get("advance_to_gpu_validation"),
        "truth_audit_promotion_allowed": truth_audit.get("promotion_allowed"),
        "pair_beats_independent": pair_metrics.get("pair_beats_independent"),
        "pair_beats_feature_count_null": pair_metrics.get("pair_beats_feature_count_null"),
        "pair_vs_independent_holdout_ce_gain": _float_or_none(pair_metrics.get("pair_vs_independent_holdout_ce_gain")),
        "pair_vs_feature_count_null_holdout_ce_gain": _float_or_none(pair_metrics.get("pair_vs_feature_count_null_holdout_ce_gain")),
        "pair_train_holdout_ce_gap": _float_or_none(pair_metrics.get("pair_train_holdout_ce_gap")),
        "holdout_token_count": _int_or_none(pair_metrics.get("holdout_token_count")),
        "strategy_recommended_next_action": strategy["recommended_next_action"],
        "ben_notification_required": strategy["ben_notification_required"],
    }


def _control_contract() -> list[dict[str, Any]]:
    return [
        {
            "control_field": "learned_causal_pair_router",
            "specification": (
                "train a strictly causal pair router from prefix-safe token, position, hidden, and router-margin "
                "features; oracle support and support-loss tables are labels only"
            ),
            "required_measurements": "oracle-pair regret;same-student forced CE;support-pair accuracy;calibrated margin",
            "failure_closes": "oracle pair-composer remains nondeployable and GPU-blocked",
            "required": True,
        },
        {
            "control_field": "delayed_misaligned_leakage_sentinels",
            "specification": (
                "rerun router/composer scoring with delayed teacher targets, misaligned support pairs, and "
                "frequency-preserving pair permutations that must fail"
            ),
            "required_measurements": "sentinel CE;sentinel R2;sentinel support overlap;pass/fail polarity",
            "failure_closes": "positive pair signal is treated as leakage-prone local artifact",
            "required": True,
        },
        {
            "control_field": "norm_utilization_and_direction_error",
            "specification": "record residual norm ratio, clipped norm budget, and hidden/logit direction error for every arm",
            "required_measurements": "residual_norm_ratio;residual_direction_error;budget_violation_rate",
            "failure_closes": "pair composer may be dense-capacity scaling rather than sparse residual structure",
            "required": True,
        },
        {
            "control_field": "functional_churn_and_retention",
            "specification": "measure support/value churn and reused-context retention against learned-router and dense/MLP controls",
            "required_measurements": "support_churn;functional_churn_kl;retention_delta;reused_context_ce",
            "failure_closes": "no low-interference or reusable-column claim",
            "required": True,
        },
        {
            "control_field": "finite_update_commutator",
            "specification": "evaluate finite update-order sensitivity for pair-composer updates at matched residual norm",
            "required_measurements": "commutator_norm;commutator_ratio;order_ce_gap;matched_norm_ratio",
            "failure_closes": "no continual-learning or low-interference mechanism claim",
            "required": True,
        },
        {
            "control_field": "matched_dense_mlp_interference_controls",
            "specification": "compare against same-active-parameter independent values, rank/norm dense residual, and matched MLP residual",
            "required_measurements": "same_param_ce;rank_norm_ce;mlp_ce;churn;commutator;retention",
            "failure_closes": "pair-composer win is generic capacity rather than sparse pair-conditioned structure",
            "required": True,
        },
    ]


def _probe_arms() -> list[dict[str, Any]]:
    roles = {
        "oracle_pair_composer_reference": "positive local reference; not deployable",
        "learned_causal_pair_router": "deployability test for pair support selection",
        "delayed_pair_target_null": "leakage sentinel that must fail",
        "misaligned_support_pair_null": "support-pair leakage/interference sentinel that must fail",
        "token_position_pair_router_null": "tests shortcut pair routing from token/position only",
        "feature_count_shuffled_pair_null": "existing null retained as regression guard",
        "same_param_independent_value_composer": "tests pair interaction versus additive columns at matched size",
        "rank_norm_matched_dense_residual": "dense capacity and residual-norm control",
        "matched_mlp_residual": "non-columnar learned control at matched active budget",
    }
    return [
        {
            "arm": arm,
            "role": roles[arm],
            "required_outputs": "summary row;per-token rows;norm/churn/retention/commutator rows",
            "required": True,
        }
        for arm in REQUIRED_PROBE_ARMS
    ]


def _gate_criteria(
    evidence: dict[str, Any],
    source_rows: list[dict[str, Any]],
    control_contract: list[dict[str, Any]],
    probe_arms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    required_sources_present = all(row["present"] for row in source_rows if row["source"] != "strategy_review")
    control_fields = {str(row["control_field"]) for row in control_contract if row.get("required") is True}
    probe_arm_names = {str(row["arm"]) for row in probe_arms if row.get("required") is True}
    truth_audit_selects_extension = (
        evidence["truth_audit_status"] == "pass"
        and evidence["truth_audit_selected_next_action"] == TRUTH_AUDIT_EXTENSION_ACTION
    )
    positive_pair_signal_preserved = (
        evidence["pair_beats_independent"] is True
        and evidence["pair_beats_feature_count_null"] is True
        and (evidence["pair_vs_independent_holdout_ce_gain"] or 0.0) > 0.0
        and (evidence["pair_vs_feature_count_null_holdout_ce_gain"] or 0.0) > 0.0
    )
    gpu_still_blocked = (
        evidence["truth_audit_advance_to_gpu_validation"] is False
        and evidence["truth_audit_promotion_allowed"] is False
    )
    return [
        _criterion("required_sources_present", required_sources_present, "truth-audit source must exist"),
        _criterion(
            "truth_audit_selects_control_extension",
            truth_audit_selects_extension,
            "truth audit must select local control extension before GPU",
        ),
        _criterion(
            "positive_pair_signal_preserved",
            positive_pair_signal_preserved,
            "pair signal must still beat independent and feature-count null controls",
        ),
        _criterion(
            "control_contract_complete",
            set(REQUIRED_CONTROL_FIELDS).issubset(control_fields),
            "all required missing-control fields must be specified",
        ),
        _criterion(
            "probe_arms_complete",
            set(REQUIRED_PROBE_ARMS).issubset(probe_arm_names),
            "all required local probe arms must be specified",
        ),
        _criterion("gpu_validation_still_blocked", gpu_still_blocked, "truth audit must still block GPU and promotion"),
        _criterion(
            "strategy_review_no_gpu_respected",
            "before any gpu" in str(evidence["strategy_recommended_next_action"]).lower()
            or "no gpu" in str(evidence["strategy_recommended_next_action"]).lower()
            or not evidence["ben_notification_required"],
            "external strategy review must not force backend validation",
        ),
    ]


def _candidate_actions(failures: list[dict[str, Any]]) -> list[dict[str, str]]:
    if failures:
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "required source or control-extension contract fields are incomplete",
                "repair or rerun pair-composer truth audit before implementing controls",
                "source_repair_required",
            )
        ]
    return [
        _candidate(
            SELECTED_ACTION,
            "selected",
            "the local pair-composer signal is coherent, but deployable routing, leakage, norm, churn, retention, commutator, and dense/MLP controls remain unmeasured",
            "implement the local pair-composer control-extension probe; keep RunPod blocked",
            "pair_composer_control_extension_design_ready_no_gpu",
        ),
        _candidate(
            "run_gpu_pair_composer_validation",
            "rejected",
            "backend validation is scientifically premature until the local control-extension probe passes",
            "do not use RunPod or Colab for this branch yet",
            "gpu_validation_blocked",
        ),
    ]


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "control_extension_contract.csv", summary["control_extension_contract"])
    _write_csv(out_dir / "probe_arms.csv", summary["probe_arms"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _notes(summary: dict[str, Any]) -> str:
    evidence = summary["evidence"]
    return "\n".join(
        [
            "# Dense-Teacher Pair-Composer Control Extension Design",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Claim status: `{summary['claim_status']}`",
            f"- Selected next action: `{summary['selected_next_action']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            f"- Advance to GPU validation: `{summary['advance_to_gpu_validation']}`",
            "",
            "## Source Signal",
            "",
            f"- Pair vs independent CE gain: `{evidence['pair_vs_independent_holdout_ce_gain']}`",
            f"- Pair vs feature-count null CE gain: `{evidence['pair_vs_feature_count_null_holdout_ce_gain']}`",
            f"- Pair train/holdout CE gap: `{evidence['pair_train_holdout_ce_gap']}`",
            f"- Holdout token count: `{evidence['holdout_token_count']}`",
            "",
            "## Interpretation",
            "",
            summary["rationale"],
            "",
            "GPU validation remains blocked. This artifact is a local probe contract, not new mechanism evidence.",
            "",
            "## Next Step",
            "",
            summary["selected_next_step"],
            "",
        ]
    )


def _criterion(name: str, passed: bool, requirement: str) -> dict[str, Any]:
    return {
        "criterion": name,
        "passed": bool(passed),
        "requirement": requirement,
        "failure_reason": "" if passed else requirement,
    }


def _candidate(action: str, disposition: str, reason: str, next_step: str, claim_status: str) -> dict[str, str]:
    return {
        "candidate_action": action,
        "disposition": disposition,
        "reason": reason,
        "next_step": next_step,
        "claim_status": claim_status,
    }


def _source_json(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(payload),
        "status": payload.get("status", "missing"),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
    }


def _strategy_response(strategy: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommendation_disposition": "accepted",
        "ben_should_be_notified": strategy["ben_notification_required"],
        "reason": (
            "The review recommends local pair-interaction controls before GPU. This design records the "
            "control-extension contract and keeps backend validation blocked."
        ),
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "present": False,
            "strategic_change_level": "missing",
            "notify_ben": "false",
            "ben_notification_required": False,
            "recommended_next_action": "",
            "verdict": "",
        }
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"strategic_change_level", "notify_ben", "recommended_next_action", "verdict"}:
            fields[key] = value.strip()
    notify = fields.get("notify_ben", "false").lower() == "true"
    major = fields.get("strategic_change_level", "").lower() == "major"
    return {
        "present": True,
        "strategic_change_level": fields.get("strategic_change_level", ""),
        "notify_ben": fields.get("notify_ben", "false"),
        "ben_notification_required": notify or major,
        "recommended_next_action": fields.get("recommended_next_action", ""),
        "verdict": fields.get("verdict", ""),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


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
            writer.writerow(row)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-audit", type=Path, default=DEFAULT_TRUTH_AUDIT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_dense_teacher_pair_composer_control_extension_design(
        truth_audit_path=args.truth_audit,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({key: summary[key] for key in ("status", "decision", "selected_next_action", "advance_to_gpu_validation")}, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
