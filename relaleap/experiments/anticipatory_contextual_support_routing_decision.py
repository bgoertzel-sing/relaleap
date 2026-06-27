"""Post-RunPod decision report for anticipatory contextual support routing."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_LOCAL_SYNTHESIS = Path(
    "results/reports/token_larger_anticipatory_contextual_support_routing_synthesis/summary.json"
)
DEFAULT_RUNPOD_SYNTHESIS = Path(
    "results/reports/runpod_token_larger_anticipatory_contextual_support_routing_synthesis_local_check/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_anticipatory_contextual_support_routing_post_runpod_decision"
)

ACSR_POST_RUNPOD_CANDIDATE_RECORDED = "acsr_post_runpod_candidate_recorded"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
NEXT_STRONGER_NON_CE_CONTROL = "acsr_same_student_cross_context_retention_churn_probe"


def run_anticipatory_contextual_support_routing_decision(
    *,
    local_synthesis_path: Path = DEFAULT_LOCAL_SYNTHESIS,
    runpod_synthesis_path: Path = DEFAULT_RUNPOD_SYNTHESIS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Consume local and RunPod ACSR synthesis packets and select the next control."""

    start = time.time()
    local_synthesis = _read_json_object(local_synthesis_path)
    runpod_synthesis = _read_json_object(runpod_synthesis_path)
    strategy_review = _strategy_review(strategy_review_path)

    source_rows = [
        _source_row("local_two_seed_synthesis", local_synthesis_path, local_synthesis),
        _source_row("runpod_two_seed_synthesis_local_check", runpod_synthesis_path, runpod_synthesis),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy_review["present"],
            "status": "present" if strategy_review["present"] else "missing_optional",
            "decision": strategy_review["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy_review['strategic_change_level']}; "
                f"notify_ben={strategy_review['notify_ben']}"
            ),
        },
    ]
    synthesis_rows = [
        _synthesis_row("local", local_synthesis),
        _synthesis_row("runpod", runpod_synthesis),
    ]
    gate_status = _gate_status(synthesis_rows)
    failures = _failures(source_rows, synthesis_rows, gate_status)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        claim_status = "acsr_post_runpod_evidence_not_interpretable"
        selected_next_step = "repair_missing_or_failed_acsr_synthesis_sources"
        rationale = (
            "The post-RunPod ACSR decision cannot be closed because required "
            "local or fetched RunPod synthesis artifacts are missing, failed, "
            "or do not pass the replicated candidate gate."
        )
    else:
        status = "pass"
        decision = ACSR_POST_RUNPOD_CANDIDATE_RECORDED
        claim_status = "acsr_replicated_candidate_not_promoted"
        selected_next_step = NEXT_STRONGER_NON_CE_CONTROL
        rationale = (
            "Local and fetched RunPod ACSR two-seed packets both pass the "
            "leakage, shuffled-feature, token/position, same-student, and "
            "causal-regret gates with closely matching aggregate deltas. This "
            "records ACSR as a replicated candidate branch, not a default router. "
            "The next bounded step is one stronger non-CE control: same-student "
            "cross-context retention/churn, so support choices are tested through "
            "identical learned values after context transfer rather than only on "
            "the fixed smoke batch."
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "next_command": None,
        "claim_statuses": {
            "acsr_mlp_predicted_future": claim_status,
            "full_context_contextual_topk2_teacher": "nondeployable_oracle_diagnostic_only",
            "promoted_default_router": "blocked_pending_stronger_non_ce_controls",
            "causal_mechanism_claim": "blocked_pending_cross_context_retention_churn",
        },
        "strategy_review": strategy_review,
        "gate_status": gate_status,
        "synthesis_rows": synthesis_rows,
        "source_rows": source_rows,
        "failures": failures,
        "rationale": rationale,
        "deferred_or_rejected_recommendations": _deferred_recommendations(strategy_review),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "synthesis_metrics_csv": str(out_dir / "synthesis_metrics.csv"),
            "gate_criteria_csv": str(out_dir / "gate_criteria.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        out_dir / "source_rows.csv",
        ["source", "path", "present", "status", "decision", "claim_status"],
        source_rows,
    )
    _write_csv(
        out_dir / "synthesis_metrics.csv",
        [
            "backend",
            "status",
            "decision",
            "claim_status",
            "packet_count",
            "all_required_gates_pass",
            "all_same_student_beats_token_position",
            "all_same_student_beats_shuffled",
            "all_teacher_churn_below_token_position",
            "all_teacher_churn_below_shuffled",
            "mean_acsr_minus_causal_ce_loss",
            "mean_acsr_minus_teacher_ce_loss",
            "mean_acsr_minus_token_position_ce_loss",
            "mean_acsr_minus_shuffled_ce_loss",
            "mean_acsr_minus_causal_regret",
            "mean_mlp_predictor_r2",
            "mean_token_position_predictor_r2",
            "mean_acsr_teacher_support_churn",
            "mean_token_position_teacher_support_churn",
            "mean_shuffled_teacher_support_churn",
        ],
        synthesis_rows,
    )
    _write_csv(
        out_dir / "gate_criteria.csv",
        ["criterion", "passed", "threshold", "actual"],
        gate_status["criteria"],
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _source_row(source: str, path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": packet.get("status") if packet else "missing",
        "decision": packet.get("decision") if packet else None,
        "claim_status": packet.get("claim_status") if packet else None,
    }


def _synthesis_row(backend: str, packet: dict[str, Any]) -> dict[str, Any]:
    aggregates = packet.get("aggregates", {}) if isinstance(packet.get("aggregates"), dict) else {}
    return {
        "backend": backend,
        "status": packet.get("status"),
        "decision": packet.get("decision"),
        "claim_status": packet.get("claim_status"),
        "packet_count": _number(packet.get("packet_count")),
        "all_required_gates_pass": _bool(aggregates.get("all_required_gates_pass")),
        "all_same_student_beats_token_position": _bool(
            aggregates.get("all_same_student_beats_token_position")
        ),
        "all_same_student_beats_shuffled": _bool(
            aggregates.get("all_same_student_beats_shuffled")
        ),
        "all_teacher_churn_below_token_position": _bool(
            aggregates.get("all_teacher_churn_below_token_position")
        ),
        "all_teacher_churn_below_shuffled": _bool(
            aggregates.get("all_teacher_churn_below_shuffled")
        ),
        "mean_acsr_minus_causal_ce_loss": _number(
            aggregates.get("mean_acsr_minus_causal_ce_loss")
        ),
        "mean_acsr_minus_teacher_ce_loss": _number(
            aggregates.get("mean_acsr_minus_teacher_ce_loss")
        ),
        "mean_acsr_minus_token_position_ce_loss": _number(
            aggregates.get("mean_acsr_minus_token_position_ce_loss")
        ),
        "mean_acsr_minus_shuffled_ce_loss": _number(
            aggregates.get("mean_acsr_minus_shuffled_ce_loss")
        ),
        "mean_acsr_minus_causal_regret": _number(
            aggregates.get("mean_acsr_minus_causal_regret")
        ),
        "mean_mlp_predictor_r2": _number(aggregates.get("mean_mlp_predictor_r2")),
        "mean_token_position_predictor_r2": _number(
            aggregates.get("mean_token_position_predictor_r2")
        ),
        "mean_acsr_teacher_support_churn": _number(
            aggregates.get("mean_acsr_teacher_support_churn")
        ),
        "mean_token_position_teacher_support_churn": _number(
            aggregates.get("mean_token_position_teacher_support_churn")
        ),
        "mean_shuffled_teacher_support_churn": _number(
            aggregates.get("mean_shuffled_teacher_support_churn")
        ),
    }


def _gate_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    criteria = [
        _criterion(
            "both_synthesis_packets_pass",
            all(row.get("status") == "pass" for row in rows),
            "local and runpod synthesis status pass",
            [row.get("status") for row in rows],
        ),
        _criterion(
            "both_record_two_seed_acsr_synthesis",
            all(row.get("decision") == "acsr_two_seed_local_synthesis_recorded" for row in rows),
            "decision == acsr_two_seed_local_synthesis_recorded",
            [row.get("decision") for row in rows],
        ),
        _criterion(
            "required_gates_replicate",
            all(row.get("all_required_gates_pass") is True for row in rows),
            "all required ACSR gates pass",
            [row.get("all_required_gates_pass") for row in rows],
        ),
        _criterion(
            "same_student_nulls_replicate",
            all(
                row.get("all_same_student_beats_token_position") is True
                and row.get("all_same_student_beats_shuffled") is True
                for row in rows
            ),
            "ACSR support beats token/position and shuffled supports through same values",
            [
                (
                    row.get("all_same_student_beats_token_position"),
                    row.get("all_same_student_beats_shuffled"),
                )
                for row in rows
            ],
        ),
        _criterion(
            "teacher_reference_churn_advantage_replicates",
            all(
                row.get("all_teacher_churn_below_token_position") is True
                and row.get("all_teacher_churn_below_shuffled") is True
                for row in rows
            ),
            "ACSR fixed-teacher churn below token/position and shuffled controls",
            [
                (
                    row.get("all_teacher_churn_below_token_position"),
                    row.get("all_teacher_churn_below_shuffled"),
                )
                for row in rows
            ],
        ),
        _criterion(
            "acsr_ce_and_regret_not_worse_replicates",
            all(
                _lt(row.get("mean_acsr_minus_causal_ce_loss"), 0.0)
                and _le(row.get("mean_acsr_minus_causal_regret"), 0.0)
                for row in rows
            ),
            "mean ACSR minus causal CE < 0 and regret <= 0",
            [
                (
                    row.get("mean_acsr_minus_causal_ce_loss"),
                    row.get("mean_acsr_minus_causal_regret"),
                )
                for row in rows
            ],
        ),
    ]
    return {
        "passes_post_runpod_candidate_gate": all(row["passed"] for row in criteria),
        "criteria": criteria,
    }


def _failures(
    source_rows: list[dict[str, Any]],
    synthesis_rows: list[dict[str, Any]],
    gate_status: dict[str, Any],
) -> list[dict[str, Any]]:
    failures = []
    for row in source_rows:
        if row["source"] != "strategy_review" and not row["present"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "source_artifact",
                    "expected": "file exists",
                    "actual": "missing",
                    "path": row["path"],
                }
            )
    for row in synthesis_rows:
        if row.get("status") != "pass":
            failures.append(
                {
                    "source": row["backend"],
                    "field": "status",
                    "expected": "pass",
                    "actual": row.get("status"),
                }
            )
        if row.get("packet_count") is None or row.get("packet_count") < 2:
            failures.append(
                {
                    "source": row["backend"],
                    "field": "packet_count",
                    "expected": ">= 2",
                    "actual": row.get("packet_count"),
                }
            )
    for criterion in gate_status["criteria"]:
        if not criterion["passed"]:
            failures.append(
                {
                    "source": "post_runpod_candidate_gate",
                    "field": criterion["criterion"],
                    "expected": criterion["threshold"],
                    "actual": criterion["actual"],
                }
            )
    return failures


def _deferred_recommendations(strategy_review: dict[str, Any]) -> list[dict[str, Any]]:
    recommended = strategy_review.get("recommended_next_action")
    if not recommended:
        return []
    if "local CPU ACSR smoke pilot" in recommended:
        return [
            {
                "recommendation": recommended,
                "status": "accepted_already_satisfied",
                "reason": (
                    "The local fail-closed ACSR pilot and subsequent RunPod "
                    "replication already exist, so the decision report advances "
                    "to the status-file next step."
                ),
            }
        ]
    return []


def _criterion(criterion: str, passed: bool, threshold: str, actual: Any) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": actual,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "present": False,
            "strategic_change_level": None,
            "notify_ben": None,
            "ben_notification_required": False,
            "recommended_next_action": None,
        }
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"strategic_change_level", "notify_ben", "recommended_next_action"}:
            values[key] = value.strip()
    notify_ben = values.get("notify_ben", "").lower() == "true"
    strategic_change_level = values.get("strategic_change_level")
    return {
        "present": True,
        "strategic_change_level": strategic_change_level,
        "notify_ben": notify_ben,
        "ben_notification_required": notify_ben or strategic_change_level == "major",
        "recommended_next_action": values.get("recommended_next_action"),
    }


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ACSR Post-RunPod Decision",
        "",
        f"Status: {summary['status']}",
        f"Decision: {summary['decision']}",
        f"Claim status: {summary['claim_status']}",
        f"Selected next step: {summary['selected_next_step']}",
        "",
        "## Rationale",
        "",
        summary["rationale"],
        "",
        "## Interpretation",
        "",
        "- ACSR is recorded as a replicated candidate branch.",
        "- The full-context teacher remains nondeployable oracle evidence only.",
        "- Default-router promotion remains blocked pending stronger non-CE controls.",
        "",
    ]
    if summary["strategy_review"]["ben_notification_required"]:
        lines.extend(
            [
                "## Ben Notification",
                "",
                "The latest strategy review requests Ben notification.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _lt(left: Any, right: Any) -> bool:
    left_float = _number(left)
    right_float = _number(right)
    return left_float is not None and right_float is not None and left_float < right_float


def _le(left: Any, right: Any) -> bool:
    left_float = _number(left)
    right_float = _number(right)
    return left_float is not None and right_float is not None and left_float <= right_float


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-synthesis", type=Path, default=DEFAULT_LOCAL_SYNTHESIS)
    parser.add_argument("--runpod-synthesis", type=Path, default=DEFAULT_RUNPOD_SYNTHESIS)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_anticipatory_contextual_support_routing_decision(
        local_synthesis_path=args.local_synthesis,
        runpod_synthesis_path=args.runpod_synthesis,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
