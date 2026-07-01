"""Close out the dense-teacher pair-composer pregate after local null controls."""

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
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_pair_composer_pregate_closeout")

CLOSEOUT_DECISION = "dense_teacher_pair_composer_pregate_closed_negative"
POSITIVE_AUDIT_DECISION = "dense_teacher_pair_composer_pregate_positive_truth_audit_selected"
FAILED_CLOSED_DECISION = "dense_teacher_pair_composer_pregate_closeout_failed_closed"
NEXT_ACTION = "redirect_to_core_periphery_predictive_coding_column_design"
TRUTH_AUDIT_ACTION = "run_local_pair_composer_truth_audit_before_gpu"
REPAIR_ACTION = "repair_dense_teacher_pair_composer_pregate_source_artifact"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "gate_criteria.csv",
    "candidate_actions.csv",
    "pregate_metrics.csv",
    "notes.md",
)


def run_dense_teacher_pair_composer_pregate_closeout(
    *,
    failure_localization_path: Path = DEFAULT_FAILURE_LOCALIZATION,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Record the negative local pregate and select the next bounded branch."""

    start = time.time()
    localization = _read_json(failure_localization_path)
    strategy = _strategy_review(strategy_review_path)
    metrics = _pregate_metrics(localization)
    evidence = _evidence(localization, strategy, metrics)
    source_rows = [
        _source_row("dense_teacher_failure_localization", failure_localization_path, localization),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "present" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}"
            ),
        },
    ]
    criteria = _criteria(source_rows, evidence)
    failures = [row for row in criteria if not row["passed"]]
    candidate_actions = _candidate_actions(evidence, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = FAILED_CLOSED_DECISION
        claim_status = "pair_composer_closeout_source_evidence_incomplete"
        selected_next_action = REPAIR_ACTION
        next_step = "repair or rerun dense_teacher_failure_localization before branch selection"
        rationale = "The closeout cannot interpret the pair-composer pregate until the source artifact is coherent."
    else:
        status = "pass"
        selected_next_action = selected[0]["candidate_action"]
        if selected_next_action == TRUTH_AUDIT_ACTION:
            decision = POSITIVE_AUDIT_DECISION
            claim_status = "dense_teacher_pair_composer_positive_local_signal_truth_audit_needed"
            next_step = (
                "run a local true-decoder pair-interaction audit with leakage, null, norm, churn, "
                "and commutator gates before any GPU validation"
            )
            rationale = (
                "The local train/holdout pregate uses the exported frozen decoder and records a real split. "
                "The pair composer improves heldout true-decoder CE over independent oracle-support values "
                "and the feature-count-matched shuffled-pair null, so the earlier negative closeout polarity "
                "was stale. This is still a small oracle-support local signal, not a promotion or GPU claim."
            )
        else:
            decision = CLOSEOUT_DECISION
            claim_status = "dense_teacher_pair_composer_negative_local_evidence_no_gpu"
            next_step = (
                "resume the local core/periphery PC column design and pilot path; keep dense-teacher "
                "pair composition as negative diagnostic context, not GPU validation evidence"
            )
            rationale = (
                "The local train/holdout pregate uses the exported frozen decoder and records a real split. "
                "The pair composer improves heldout true-decoder CE over independent oracle-support values, "
                "but loses to the feature-count-matched shuffled-pair null. That blocks a sparse-column "
                "pair-composer claim and makes GPU validation low value."
            )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_action": selected_next_action,
        "next_step": next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "backend_policy": "RunPod/Colab blocked; this is a local closeout after true-decoder pregate controls.",
        "source_rows": source_rows,
        "evidence": evidence,
        "criteria": criteria,
        "candidate_actions": candidate_actions,
        "pregate_metrics": metrics,
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


def _evidence(
    localization: dict[str, Any],
    strategy: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "localization_status": localization.get("status"),
        "localization_decision": localization.get("decision"),
        "localization_claim_status": localization.get("claim_status"),
        "localization_selected_next_step": localization.get("selected_next_step"),
        "composer_train_holdout_split_recorded": localization.get("composer_train_holdout_split_recorded"),
        "composer_uses_true_frozen_decoder_for_ce": localization.get("composer_uses_true_frozen_decoder_for_ce"),
        "composer_ce_metric_path": localization.get("composer_ce_metric_path"),
        "no_gpu_pregate_status": localization.get("no_gpu_pregate_status"),
        "composer_validation_blocker": localization.get("composer_validation_blocker"),
        "pair_beats_independent": metrics.get("pair_beats_independent"),
        "pair_beats_feature_count_null": metrics.get("pair_beats_feature_count_null"),
        "pair_holdout_true_decoder_ce_loss": metrics.get("pair_holdout_true_decoder_ce_loss"),
        "independent_holdout_true_decoder_ce_loss": metrics.get("independent_holdout_true_decoder_ce_loss"),
        "feature_count_null_holdout_true_decoder_ce_loss": metrics.get(
            "feature_count_null_holdout_true_decoder_ce_loss"
        ),
        "strategy_verdict": strategy.get("verdict"),
        "strategy_recommended_next_action": strategy.get("recommended_next_action"),
        "ben_notification_required": strategy.get("ben_notification_required"),
    }


def _criteria(source_rows: list[dict[str, Any]], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    source_present = source_rows[0]["present"]
    split_recorded = evidence["composer_train_holdout_split_recorded"] is True
    true_decoder_recorded = evidence["composer_uses_true_frozen_decoder_for_ce"] is True
    null_result_recorded = evidence["pair_beats_feature_count_null"] in (True, False)
    independent_helped = evidence["pair_beats_independent"] is True
    return [
        _criterion(
            "failure_localization_source_present",
            source_present,
            "dense_teacher_failure_localization summary exists",
            source_present,
        ),
        _criterion(
            "localization_evaluator_completed",
            evidence["localization_status"] == "pass"
            and evidence["localization_decision"] == "dense_teacher_failure_localization_evaluator_recorded",
            "source evaluator passed and recorded all local rows",
            f"{evidence['localization_status']}; {evidence['localization_decision']}",
        ),
        _criterion(
            "true_decoder_train_holdout_pregate_recorded",
            split_recorded and true_decoder_recorded,
            "pair composer uses train/holdout split and true frozen decoder CE",
            f"split={split_recorded}; true_decoder={true_decoder_recorded}",
        ),
        _criterion(
            "pair_composer_improves_independent_control",
            independent_helped,
            "pair composer should at least improve over fixed-support independent values",
            independent_helped,
        ),
        _criterion(
            "feature_count_matched_null_result_recorded",
            null_result_recorded,
            "pair composer must record whether it beats the feature-count-matched shuffled-pair null",
            evidence["pair_beats_feature_count_null"],
        ),
        _criterion(
            "strategy_review_local_pregate_respected",
            "pregate" in str(evidence["strategy_recommended_next_action"]).lower()
            or str(evidence["strategy_verdict"]).upper() == "FIX",
            "external review asked for local pregate before GPU",
            evidence["strategy_recommended_next_action"],
        ),
    ]


def _candidate_actions(
    evidence: dict[str, Any],
    failures: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if failures:
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "required pregate source evidence is missing or contradictory",
                "repair or rerun dense_teacher_failure_localization",
                "source_repair_required_no_scientific_branch_selection",
            ),
            _candidate(
                NEXT_ACTION,
                "blocked",
                "cannot redirect from an incoherent pregate artifact",
                "rerun after source repair",
                "blocked_pending_source_repair",
            ),
        ]
    if evidence.get("pair_beats_feature_count_null") is True:
        return [
            _candidate(
                TRUTH_AUDIT_ACTION,
                "selected",
                "pair composer beats both independent oracle-support values and the feature-count-matched shuffled-pair null under heldout true-decoder CE",
                "add a local pair-composer truth audit with leakage/null/interference/norm gates; keep RunPod and Colab blocked",
                "positive_pair_composer_local_signal_truth_audit_selected_no_gpu",
            ),
            _candidate(
                "run_gpu_pair_composer_validation",
                "rejected",
                "the positive is still a small oracle-support local signal without leakage, learned-router, churn, commutator, or retention gates",
                "do not use RunPod or Colab before the truth audit",
                "gpu_validation_blocked_pending_truth_audit",
            ),
            _candidate(
                "promote_pair_composer_sparse_column_claim",
                "rejected",
                "oracle pair-composer evidence does not establish deployable routing or low-interference sparse columns",
                "require the local truth audit before any mechanism claim",
                "sparse_pair_composer_claim_not_established",
            ),
        ]
    return [
        _candidate(
            NEXT_ACTION,
            "selected",
            "pair composer loses to the feature-count-matched null under heldout true-decoder CE",
            "resume local core/periphery PC column design and pilot path before any GPU work",
            "core_periphery_pc_design_selected_after_negative_pair_composer_pregate",
        ),
        _candidate(
            "run_gpu_pair_composer_validation",
            "rejected",
            "GPU validation requires beating independent and feature-count-matched null controls locally",
            "do not use RunPod or Colab for this dense-teacher pair-composer branch",
            "gpu_validation_blocked_by_local_null",
        ),
        _candidate(
            "promote_pair_composer_sparse_column_claim",
            "rejected",
            "the observed pair-composer gain is not specific against a matched shuffled-pair feature null",
            "keep pair composer as a negative localization diagnostic",
            "sparse_pair_composer_claim_not_established",
        ),
    ]


def _pregate_metrics(localization: dict[str, Any]) -> dict[str, Any]:
    rows = localization.get("pair_composer_pregate_rows")
    if not isinstance(rows, list):
        rows = []
    holdout_by_arm = {
        str(row.get("arm")): row
        for row in rows
        if isinstance(row, dict) and row.get("split") == "holdout"
    }
    independent = holdout_by_arm.get("retrained_oracle_support_values", {})
    pair = holdout_by_arm.get("oracle_support_gated_value_pair_composer", {})
    null = holdout_by_arm.get("feature_count_matched_shuffled_pair_null", {})
    return {
        "independent_holdout_true_decoder_ce_loss": _float_or_none(
            independent.get("true_decoder_ce_loss")
        ),
        "pair_holdout_true_decoder_ce_loss": _float_or_none(pair.get("true_decoder_ce_loss")),
        "feature_count_null_holdout_true_decoder_ce_loss": _float_or_none(
            null.get("true_decoder_ce_loss")
        ),
        "pair_beats_independent": _bool_or_none(pair.get("beats_independent_holdout_ce")),
        "pair_beats_feature_count_null": _bool_or_none(
            pair.get("beats_shuffled_pair_null_holdout_ce")
        ),
        "pair_feature_count": _int_or_none(pair.get("feature_count")),
        "null_feature_count": _int_or_none(null.get("feature_count")),
        "split_seed": _int_or_none(pair.get("split_seed")),
        "holdout_token_count": _int_or_none(pair.get("token_count")),
        "pair_holdout_logit_r2": _float_or_none(pair.get("teacher_logit_residual_r2")),
        "pair_holdout_hidden_r2": _float_or_none(pair.get("teacher_hidden_residual_r2")),
    }


def _strategy_response(strategy: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommendation_disposition": "accepted",
        "reason": (
            "The review recommended a local decoder-exported train/holdout pregate before GPU. "
            "That pregate is now recorded and interpreted through matched-null controls, so the "
            "block-GPU direction is carried forward until a local truth audit clears leakage, null, "
            "norm, churn, and commutator gates."
        ),
        "ben_should_be_notified": bool(strategy.get("ben_notification_required")),
    }


def _criterion(criterion: str, passed: bool, threshold: str, actual: Any) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": actual,
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


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status") if path.is_file() else "missing",
        "decision": payload.get("decision"),
        "claim_status": payload.get("claim_status"),
    }


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        out_dir / "source_rows.csv",
        ["source", "path", "present", "status", "decision", "claim_status"],
        summary["source_rows"],
    )
    _write_csv(
        out_dir / "gate_criteria.csv",
        ["criterion", "passed", "threshold", "actual"],
        summary["criteria"],
    )
    _write_csv(
        out_dir / "candidate_actions.csv",
        ["candidate_action", "disposition", "reason", "next_step", "claim_status"],
        summary["candidate_actions"],
    )
    _write_csv(
        out_dir / "pregate_metrics.csv",
        ["metric", "value"],
        [{"metric": key, "value": value} for key, value in summary["pregate_metrics"].items()],
    )
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Dense Teacher Pair-Composer Pregate Closeout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Next step: {summary['next_step']}",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Ben notification: `{summary['strategy_response']['ben_should_be_notified']}`",
        "",
        summary["rationale"],
        "",
        "Key heldout CE values:",
        f"- Independent oracle-support values: `{summary['pregate_metrics'].get('independent_holdout_true_decoder_ce_loss')}`",
        f"- Pair composer: `{summary['pregate_metrics'].get('pair_holdout_true_decoder_ce_loss')}`",
        f"- Feature-count shuffled-pair null: `{summary['pregate_metrics'].get('feature_count_null_holdout_true_decoder_ce_loss')}`",
        "",
        "This artifact blocks RunPod/Colab validation for the dense-teacher pair-composer branch.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _strategy_review(path: Path) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "present": path.is_file(),
        "strategic_change_level": None,
        "notify_ben": None,
        "recommended_next_action": None,
        "verdict": None,
    }
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key in fields:
                fields[key] = value
    fields["ben_notification_required"] = (
        str(fields.get("notify_ben")).lower() == "true"
        or fields.get("strategic_change_level") == "major"
    )
    return fields


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-localization", type=Path, default=DEFAULT_FAILURE_LOCALIZATION)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = run_dense_teacher_pair_composer_pregate_closeout(
        failure_localization_path=args.failure_localization,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
