"""Audit the local dense-teacher pair-composer signal before any GPU work."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_FAILURE_LOCALIZATION = Path("results/reports/dense_teacher_failure_localization/summary.json")
DEFAULT_CLOSEOUT = Path("results/reports/dense_teacher_pair_composer_pregate_closeout/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_pair_composer_truth_audit")

AUDIT_DECISION_GPU_BLOCKED = "dense_teacher_pair_composer_truth_audit_gpu_blocked"
AUDIT_DECISION_SOURCES_INCOMPLETE = "dense_teacher_pair_composer_truth_audit_sources_incomplete"
ADVANCE_ACTION = "advance_pair_composer_to_local_mechanism_probe"
CONTROL_EXTENSION_ACTION = "extend_pair_composer_truth_audit_controls_before_gpu"
REPAIR_ACTION = "repair_pair_composer_truth_audit_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "pair_metrics.csv",
    "gate_criteria.csv",
    "control_matrix.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_dense_teacher_pair_composer_truth_audit(
    *,
    failure_localization_path: Path = DEFAULT_FAILURE_LOCALIZATION,
    closeout_path: Path = DEFAULT_CLOSEOUT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed truth audit over the current pair-composer artifacts."""

    start = time.time()
    localization = _read_json(failure_localization_path)
    closeout = _read_json(closeout_path)
    strategy = _strategy_review(strategy_review_path)
    sources = [
        _source_row("dense_teacher_failure_localization", failure_localization_path, localization),
        _source_row("dense_teacher_pair_composer_pregate_closeout", closeout_path, closeout),
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
    source_failures = _source_failures(sources)
    pair_metrics = _pair_metrics(localization)
    control_matrix = _control_matrix(localization, closeout, pair_metrics)
    criteria = _criteria(sources, pair_metrics, control_matrix, closeout, strategy)
    failures = source_failures + [row for row in criteria if not row["passed"]]
    candidate_actions = _candidate_actions(source_failures, criteria)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if source_failures or len(selected) != 1:
        status = "fail"
        decision = AUDIT_DECISION_SOURCES_INCOMPLETE
        claim_status = "pair_composer_truth_audit_source_artifacts_incomplete"
        selected_next_action = REPAIR_ACTION
        next_step = "repair or regenerate pair-composer pregate artifacts before interpreting the truth audit"
        advance_to_gpu_validation = False
        rationale = "The truth audit cannot interpret the pair-composer signal until required source artifacts are present."
    else:
        status = "pass"
        decision = AUDIT_DECISION_GPU_BLOCKED
        selected_next_action = selected[0]["candidate_action"]
        next_step = selected[0]["next_step"]
        advance_to_gpu_validation = selected_next_action == ADVANCE_ACTION
        if advance_to_gpu_validation:
            claim_status = "pair_composer_truth_audit_local_gates_cleared_no_gpu_run_yet"
            rationale = (
                "The current local artifact clears the audited pair-composer controls. "
                "A separate bounded local mechanism probe should still precede any backend promotion claim."
            )
        else:
            claim_status = "pair_composer_positive_signal_controls_recorded_but_not_cleared"
            rationale = (
                "The artifact confirms a local oracle-support pair-composer CE/R2 signal against independent and "
                "feature-count null controls. It remains GPU-blocked because the current artifact does not include "
                "deployable pair routing, leakage sentinels beyond split/true-decoder accounting, norm utilization, "
                "functional churn, retention, finite-update commutator, or matched dense/MLP interference controls."
            )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_action": selected_next_action,
        "next_step": next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": advance_to_gpu_validation,
        "backend_policy": "RunPod/Colab blocked; this is a local artifact truth audit.",
        "source_rows": sources,
        "pair_metrics": pair_metrics,
        "control_matrix": control_matrix,
        "criteria": criteria,
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


def _pair_metrics(localization: dict[str, Any]) -> dict[str, Any]:
    rows = _as_list(localization.get("pair_composer_pregate_rows"))
    by_key = {
        (str(row.get("arm")), str(row.get("split"))): row
        for row in rows
        if isinstance(row, dict)
    }
    pair_train = by_key.get(("oracle_support_gated_value_pair_composer", "train"), {})
    pair_holdout = by_key.get(("oracle_support_gated_value_pair_composer", "holdout"), {})
    independent_train = by_key.get(("retrained_oracle_support_values", "train"), {})
    independent_holdout = by_key.get(("retrained_oracle_support_values", "holdout"), {})
    null_train = by_key.get(("feature_count_matched_shuffled_pair_null", "train"), {})
    null_holdout = by_key.get(("feature_count_matched_shuffled_pair_null", "holdout"), {})

    pair_holdout_ce = _float(pair_holdout.get("true_decoder_ce_loss"))
    independent_holdout_ce = _float(independent_holdout.get("true_decoder_ce_loss"))
    null_holdout_ce = _float(null_holdout.get("true_decoder_ce_loss"))
    pair_train_ce = _float(pair_train.get("true_decoder_ce_loss"))
    independent_train_ce = _float(independent_train.get("true_decoder_ce_loss"))
    null_train_ce = _float(null_train.get("true_decoder_ce_loss"))
    pair_feature_count = _int(pair_holdout.get("feature_count"))
    null_feature_count = _int(null_holdout.get("feature_count"))

    return {
        "split_seed": _int(pair_holdout.get("split_seed")),
        "holdout_token_count": _int(pair_holdout.get("token_count")),
        "train_token_count": _int(pair_train.get("token_count")),
        "pair_train_true_decoder_ce_loss": pair_train_ce,
        "pair_holdout_true_decoder_ce_loss": pair_holdout_ce,
        "independent_train_true_decoder_ce_loss": independent_train_ce,
        "independent_holdout_true_decoder_ce_loss": independent_holdout_ce,
        "feature_count_null_train_true_decoder_ce_loss": null_train_ce,
        "feature_count_null_holdout_true_decoder_ce_loss": null_holdout_ce,
        "pair_vs_independent_holdout_ce_gain": _diff(independent_holdout_ce, pair_holdout_ce),
        "pair_vs_feature_count_null_holdout_ce_gain": _diff(null_holdout_ce, pair_holdout_ce),
        "pair_train_holdout_ce_gap": _diff(pair_holdout_ce, pair_train_ce),
        "null_train_holdout_ce_gap": _diff(null_holdout_ce, null_train_ce),
        "pair_holdout_logit_r2": _float(pair_holdout.get("teacher_logit_residual_r2")),
        "pair_holdout_hidden_r2": _float(pair_holdout.get("teacher_hidden_residual_r2")),
        "pair_feature_count": pair_feature_count,
        "feature_count_null_feature_count": null_feature_count,
        "feature_count_matched": pair_feature_count is not None and pair_feature_count == null_feature_count,
        "pair_beats_independent": _bool(pair_holdout.get("beats_independent_holdout_ce"))
        if pair_holdout.get("beats_independent_holdout_ce") != ""
        else _lt(pair_holdout_ce, independent_holdout_ce),
        "pair_beats_feature_count_null": _bool(pair_holdout.get("beats_shuffled_pair_null_holdout_ce"))
        if pair_holdout.get("beats_shuffled_pair_null_holdout_ce") != ""
        else _lt(pair_holdout_ce, null_holdout_ce),
        "uses_true_frozen_decoder_for_ce": _bool(pair_holdout.get("uses_true_frozen_decoder_for_ce")),
        "train_holdout_split_recorded": _bool(pair_holdout.get("train_holdout_split_recorded")),
    }


def _control_matrix(
    localization: dict[str, Any],
    closeout: dict[str, Any],
    pair_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _control(
            "true_decoder_split_accounting",
            "passed" if pair_metrics["uses_true_frozen_decoder_for_ce"] and pair_metrics["train_holdout_split_recorded"] else "failed",
            "pair-composer row records true frozen-decoder CE and train/holdout split",
            {
                "uses_true_frozen_decoder_for_ce": pair_metrics["uses_true_frozen_decoder_for_ce"],
                "train_holdout_split_recorded": pair_metrics["train_holdout_split_recorded"],
            },
        ),
        _control(
            "feature_count_matched_shuffled_pair_null",
            "passed" if pair_metrics["pair_beats_feature_count_null"] and pair_metrics["feature_count_matched"] else "failed",
            "pair composer must beat a feature-count-matched shuffled-pair null",
            {
                "pair_feature_count": pair_metrics["pair_feature_count"],
                "feature_count_null_feature_count": pair_metrics["feature_count_null_feature_count"],
                "ce_gain": pair_metrics["pair_vs_feature_count_null_holdout_ce_gain"],
            },
        ),
        _control(
            "independent_value_control",
            "passed" if pair_metrics["pair_beats_independent"] else "failed",
            "pair composer must beat independent oracle-support values",
            {"ce_gain": pair_metrics["pair_vs_independent_holdout_ce_gain"]},
        ),
        _control(
            "deployable_pair_router",
            "blocked",
            "current positive uses oracle support; no learned causal pair router is audited",
            {"support_source": "oracle_support"},
        ),
        _control(
            "leakage_sentinels",
            "blocked",
            "split and true-decoder accounting exist, but delayed/misaligned/support-input leakage sentinels are absent",
            {"available_sentinels": ["train_holdout_split", "true_frozen_decoder_ce"]},
        ),
        _control(
            "norm_utilization",
            "blocked",
            "pair-composer artifact has CE/R2 but no residual norm-utilization or direction-error row",
            {"residual_norm_ratio": localization.get("pair_composer_residual_norm_ratio")},
        ),
        _control(
            "functional_churn",
            "blocked",
            "pair-composer artifact has no functional churn measurement",
            {"functional_churn": localization.get("pair_composer_functional_churn")},
        ),
        _control(
            "retention",
            "blocked",
            "pair-composer artifact has no task-free retention/reused-context measurement",
            {"retention_metric": localization.get("pair_composer_retention")},
        ),
        _control(
            "finite_update_commutator",
            "blocked",
            "pair-composer artifact has no finite-update order/commutator measurement",
            {"commutator_norm": localization.get("pair_composer_commutator_norm")},
        ),
        _control(
            "matched_dense_mlp_interference_controls",
            "blocked",
            "current closeout requires a truth audit before dense/MLP interference controls can be interpreted for pair composition",
            {"closeout_selected_next_action": closeout.get("selected_next_action")},
        ),
    ]


def _criteria(
    sources: list[dict[str, Any]],
    pair_metrics: dict[str, Any],
    control_matrix: list[dict[str, Any]],
    closeout: dict[str, Any],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    source_present = all(row["present"] for row in sources if row["source"] != "strategy_review")
    blocked_controls = [row["control"] for row in control_matrix if row["status"] == "blocked"]
    return [
        _criterion("required_sources_present", source_present, "failure-localization and closeout summaries must exist", source_present),
        _criterion(
            "closeout_selected_truth_audit",
            closeout.get("selected_next_action") == "run_local_pair_composer_truth_audit_before_gpu",
            "closeout must select the local truth audit before GPU",
            closeout.get("selected_next_action"),
        ),
        _criterion(
            "true_decoder_train_holdout_split",
            pair_metrics["uses_true_frozen_decoder_for_ce"] is True and pair_metrics["train_holdout_split_recorded"] is True,
            "pair rows must use true frozen-decoder CE and a train/holdout split",
            {
                "uses_true_frozen_decoder_for_ce": pair_metrics["uses_true_frozen_decoder_for_ce"],
                "train_holdout_split_recorded": pair_metrics["train_holdout_split_recorded"],
            },
        ),
        _criterion(
            "pair_beats_independent_holdout_ce",
            pair_metrics["pair_beats_independent"] is True and _positive(pair_metrics["pair_vs_independent_holdout_ce_gain"]),
            "pair-composer holdout CE must beat independent oracle-support values",
            pair_metrics["pair_vs_independent_holdout_ce_gain"],
        ),
        _criterion(
            "pair_beats_feature_count_null_holdout_ce",
            pair_metrics["pair_beats_feature_count_null"] is True
            and pair_metrics["feature_count_matched"] is True
            and _positive(pair_metrics["pair_vs_feature_count_null_holdout_ce_gain"]),
            "pair-composer holdout CE must beat a feature-count-matched shuffled-pair null",
            pair_metrics["pair_vs_feature_count_null_holdout_ce_gain"],
        ),
        _criterion(
            "train_holdout_gap_not_explosive",
            pair_metrics["pair_train_holdout_ce_gap"] is not None
            and pair_metrics["pair_train_holdout_ce_gap"] <= 1.0,
            "pair-composer CE train/holdout gap should stay within the local pregate budget <= 1.0",
            pair_metrics["pair_train_holdout_ce_gap"],
        ),
        _criterion(
            "nontrivial_residual_reconstruction",
            (pair_metrics["pair_holdout_hidden_r2"] or 0.0) >= 0.5
            and (pair_metrics["pair_holdout_logit_r2"] or 0.0) >= 0.5,
            "pair composer should have nontrivial hidden and logit residual R2 on holdout",
            {
                "hidden_r2": pair_metrics["pair_holdout_hidden_r2"],
                "logit_r2": pair_metrics["pair_holdout_logit_r2"],
            },
        ),
        _criterion(
            "mechanism_controls_complete_for_gpu",
            not blocked_controls,
            "deployable-router, leakage, norm, churn, retention, commutator, and dense/MLP controls must be present before GPU",
            blocked_controls,
        ),
        _criterion(
            "strategy_review_no_gpu_respected",
            "before any gpu" in str(strategy.get("recommended_next_action", "")).lower()
            or str(strategy.get("verdict", "")).upper() == "FIX",
            "external review recommendation must be incorporated",
            strategy.get("recommended_next_action"),
        ),
    ]


def _candidate_actions(
    source_failures: list[dict[str, Any]],
    criteria: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if source_failures:
        return [
            _candidate(REPAIR_ACTION, "selected", "required pair-composer source artifacts are missing", "repair or regenerate source artifacts", "source_repair_required"),
        ]
    control_complete = _criterion_passed(criteria, "mechanism_controls_complete_for_gpu")
    positive_signal = all(
        _criterion_passed(criteria, name)
        for name in (
            "true_decoder_train_holdout_split",
            "pair_beats_independent_holdout_ce",
            "pair_beats_feature_count_null_holdout_ce",
            "train_holdout_gap_not_explosive",
            "nontrivial_residual_reconstruction",
        )
    )
    if positive_signal and control_complete:
        return [
            _candidate(
                ADVANCE_ACTION,
                "selected",
                "all local truth-audit controls are present and pass",
                "run one local pair-composer mechanism probe before considering RunPod",
                "local_truth_audit_cleared",
            )
        ]
    return [
        _candidate(
            CONTROL_EXTENSION_ACTION,
            "selected",
            "the positive CE/R2 signal is real locally, but required mechanism and interference controls are absent",
            "extend the local pair-composer artifact with learned causal routing, leakage sentinels, norm utilization, churn, retention, finite-update commutator, and matched dense/MLP controls",
            "local_pair_composer_controls_missing_gpu_blocked",
        ),
        _candidate(
            "run_gpu_pair_composer_validation",
            "rejected",
            "RunPod/Colab validation would amplify an oracle-support artifact without deployable routing or interference controls",
            "keep backend validation blocked",
            "gpu_validation_blocked",
        ),
        _candidate(
            "promote_pair_composer_sparse_column_claim",
            "rejected",
            "pair-conditioned oracle-support value composition is not yet causal separability or reusable sparse-column evidence",
            "require the missing local controls first",
            "promotion_blocked",
        ),
    ]


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "pair_metrics.csv", [_flatten(summary["pair_metrics"])])
    _write_csv(out_dir / "gate_criteria.csv", summary["criteria"])
    _write_csv(out_dir / "control_matrix.csv", summary["control_matrix"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _notes(summary: dict[str, Any]) -> str:
    metrics = summary["pair_metrics"]
    return "\n".join(
        [
            "# Dense-Teacher Pair-Composer Truth Audit",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Claim status: `{summary['claim_status']}`",
            f"- Selected next action: `{summary['selected_next_action']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            f"- Advance to GPU validation: `{summary['advance_to_gpu_validation']}`",
            "",
            "## Local Signal",
            "",
            f"- Pair holdout true-decoder CE: `{metrics['pair_holdout_true_decoder_ce_loss']}`",
            f"- Independent holdout true-decoder CE: `{metrics['independent_holdout_true_decoder_ce_loss']}`",
            f"- Feature-count null holdout true-decoder CE: `{metrics['feature_count_null_holdout_true_decoder_ce_loss']}`",
            f"- Pair vs independent CE gain: `{metrics['pair_vs_independent_holdout_ce_gain']}`",
            f"- Pair vs feature-count null CE gain: `{metrics['pair_vs_feature_count_null_holdout_ce_gain']}`",
            f"- Pair train/holdout CE gap: `{metrics['pair_train_holdout_ce_gap']}`",
            "",
            "## Interpretation",
            "",
            summary["rationale"],
            "",
            "## Next Step",
            "",
            summary["next_step"],
            "",
        ]
    )


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status") if path.is_file() else "missing",
        "decision": payload.get("decision"),
        "claim_status": payload.get("claim_status"),
    }


def _source_failures(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"criterion": f"{row['source']}_present", "passed": False, "actual": row["path"], "threshold": "source artifact exists"}
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]


def _control(control: str, status: str, interpretation: str, observed: dict[str, Any]) -> dict[str, Any]:
    return {
        "control": control,
        "status": status,
        "interpretation": interpretation,
        "observed": json.dumps(observed, sort_keys=True),
    }


def _criterion(criterion: str, passed: bool, threshold: str, actual: Any) -> dict[str, Any]:
    return {"criterion": criterion, "passed": bool(passed), "threshold": threshold, "actual": actual}


def _candidate(action: str, disposition: str, reason: str, next_step: str, claim_status: str) -> dict[str, str]:
    return {
        "candidate_action": action,
        "disposition": disposition,
        "reason": reason,
        "next_step": next_step,
        "claim_status": claim_status,
    }


def _criterion_passed(criteria: list[dict[str, Any]], name: str) -> bool:
    return any(row["criterion"] == name and row["passed"] for row in criteria)


def _strategy_response(strategy: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommendation_disposition": "accepted",
        "ben_should_be_notified": strategy["ben_notification_required"],
        "reason": (
            "The review recommended a local true-decoder pair-interaction audit with leakage/null/interference "
            "gates before GPU. This command implements that audit and keeps backend validation blocked where "
            "the current artifact lacks required controls."
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


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _lt(left: float | None, right: float | None) -> bool | None:
    if left is None or right is None:
        return None
    return left < right


def _diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _positive(value: float | None) -> bool:
    return value is not None and value > 0.0


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in row.items()
    }


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
    parser.add_argument("--failure-localization", type=Path, default=DEFAULT_FAILURE_LOCALIZATION)
    parser.add_argument("--closeout", type=Path, default=DEFAULT_CLOSEOUT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_dense_teacher_pair_composer_truth_audit(
        failure_localization_path=args.failure_localization,
        closeout_path=args.closeout,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({key: summary[key] for key in ("status", "decision", "selected_next_action", "advance_to_gpu_validation")}, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
