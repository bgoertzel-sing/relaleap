"""Cross-context retention/churn probe for anticipatory contextual support routing."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_AUDIT_DIRS = (
    Path("results/audits/token_larger_anticipatory_contextual_support_routing"),
    Path("results/audits/token_larger_anticipatory_contextual_support_routing_seed2"),
    Path("results/runpod_fetch/audits/runpod_token_larger_anticipatory_contextual_support_routing"),
    Path("results/runpod_fetch/audits/runpod_token_larger_anticipatory_contextual_support_routing_seed2"),
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_anticipatory_contextual_support_routing_retention_churn_probe"
)

ACSR_CROSS_CONTEXT_RETENTION_CHURN_SUPPORTED = (
    "acsr_same_student_cross_context_retention_churn_supported_not_promoted"
)
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
PRIMARY_VARIANT = "acsr_mlp_predicted_future"
CONTROL_VARIANTS = (
    "token_position_only_predicted_features",
    "shuffled_predicted_features",
)
REQUIRED_FILES = (
    "summary.json",
    "same_student_metrics.csv",
    "retention_churn_metrics.csv",
)


def run_acsr_same_student_cross_context_retention_churn_probe(
    *,
    audit_dirs: tuple[Path, ...] = DEFAULT_AUDIT_DIRS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Gate ACSR's same-student support advantage after context transfer."""

    start = time.time()
    packet_rows = [
        _packet_row(index=index + 1, audit_dir=path)
        for index, path in enumerate(audit_dirs)
    ]
    comparison_rows = _comparison_rows(packet_rows)
    aggregate_rows = _aggregate_rows(comparison_rows)
    criteria = _criteria(packet_rows, comparison_rows)
    failures = _failures(packet_rows, criteria)
    strategy_review = _strategy_review(strategy_review_path)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        claim_status = "acsr_cross_context_retention_churn_not_interpretable"
        selected_next_step = "repair_or_extend_acsr_retention_churn_sources"
        rationale = (
            "The cross-context ACSR retention/churn probe cannot support a "
            "mechanism claim because required packet artifacts are missing, "
            "failed, or do not show ACSR beating token/position and shuffled "
            "same-student controls after the second-context update."
        )
    else:
        status = "pass"
        decision = ACSR_CROSS_CONTEXT_RETENTION_CHURN_SUPPORTED
        claim_status = "stronger_non_ce_acsr_control_supported_not_promoted"
        selected_next_step = (
            "design a no-default-promotion causal-mechanism gate that adds "
            "oracle-support regret and functional-churn probes on held-out contexts"
        )
        rationale = (
            "Across local and fetched RunPod ACSR packets, the MLP ACSR support "
            "choice beats token/position and shuffled null supports through the "
            "same learned values, and after a second-context update it has lower "
            "anchor logit MSE and lower or equal support churn than those nulls. "
            "This strengthens ACSR as a candidate branch while preserving the "
            "default-router and causal-mechanism blocks."
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "claim_statuses": {
            "acsr_mlp_predicted_future": claim_status,
            "token_position_only_predicted_features": "null_control",
            "shuffled_predicted_features": "null_control",
            "promoted_default_router": "blocked_pending_heldout_oracle_regret_functional_churn",
            "causal_mechanism_claim": "blocked_pending_heldout_context_interventions",
        },
        "strategy_review": strategy_review,
        "packet_rows": [_public_packet_row(row) for row in packet_rows],
        "comparison_rows": comparison_rows,
        "aggregate_rows": aggregate_rows,
        "gate_status": {
            "passes_cross_context_retention_churn_gate": not failures,
            "criteria": criteria,
        },
        "failures": failures,
        "rationale": rationale,
        "deferred_or_rejected_recommendations": _deferred_recommendations(strategy_review),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_packets_csv": str(out_dir / "source_packets.csv"),
            "same_student_cross_context_metrics_csv": str(
                out_dir / "same_student_cross_context_metrics.csv"
            ),
            "aggregate_metrics_csv": str(out_dir / "aggregate_metrics.csv"),
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
        out_dir / "source_packets.csv",
        [
            "packet",
            "audit_dir",
            "present",
            "status",
            "decision",
            "claim_status",
            "required_files_present",
            "same_student_rows",
            "retention_rows",
        ],
        packet_rows,
    )
    _write_csv(
        out_dir / "same_student_cross_context_metrics.csv",
        [
            "packet",
            "control_variant",
            "same_student_acsr_minus_control_ce_loss",
            "anchor_support_churn_after_transfer_acsr",
            "anchor_support_churn_after_transfer_control",
            "acsr_minus_control_anchor_support_churn",
            "anchor_logit_mse_after_transfer_acsr",
            "anchor_logit_mse_after_transfer_control",
            "acsr_minus_control_anchor_logit_mse",
            "transfer_ce_improvement_acsr",
            "transfer_ce_improvement_control",
            "acsr_minus_control_transfer_ce_improvement",
            "teacher_support_churn_acsr",
            "teacher_support_churn_control",
            "acsr_minus_control_teacher_support_churn",
            "passes_same_student_ce",
            "passes_anchor_support_churn",
            "passes_anchor_logit_mse",
            "passes_teacher_reference_churn",
        ],
        comparison_rows,
    )
    _write_csv(
        out_dir / "aggregate_metrics.csv",
        [
            "control_variant",
            "packet_count",
            "mean_same_student_acsr_minus_control_ce_loss",
            "mean_acsr_minus_control_anchor_support_churn",
            "mean_acsr_minus_control_anchor_logit_mse",
            "mean_acsr_minus_control_transfer_ce_improvement",
            "mean_acsr_minus_control_teacher_support_churn",
            "all_gates_pass",
        ],
        aggregate_rows,
    )
    _write_csv(
        out_dir / "gate_criteria.csv",
        ["criterion", "passed", "threshold", "actual"],
        criteria,
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _public_packet_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not key.startswith("_")
    }


def _packet_row(index: int, audit_dir: Path) -> dict[str, Any]:
    summary = _read_json_object(audit_dir / "summary.json")
    same_student_rows = _read_csv_rows(audit_dir / "same_student_metrics.csv")
    retention_rows = _read_csv_rows(audit_dir / "retention_churn_metrics.csv")
    return {
        "packet": f"packet{index}",
        "audit_dir": str(audit_dir),
        "present": audit_dir.is_dir(),
        "status": summary.get("status"),
        "decision": summary.get("decision"),
        "claim_status": summary.get("claim_status"),
        "required_files_present": all((audit_dir / name).is_file() for name in REQUIRED_FILES),
        "same_student_rows": len(same_student_rows),
        "retention_rows": len(retention_rows),
        "_same_student": same_student_rows,
        "_retention": retention_rows,
    }


def _comparison_rows(packet_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for packet in packet_rows:
        same_student = {
            row.get("comparison"): row for row in packet.get("_same_student", [])
        }
        retention = {
            (row.get("phase"), row.get("variant")): row
            for row in packet.get("_retention", [])
        }
        acsr_transfer = retention.get(("second_context_transfer", PRIMARY_VARIANT), {})
        acsr_teacher = retention.get(("fixed_context_teacher_reference", PRIMARY_VARIANT), {})
        for control in CONTROL_VARIANTS:
            control_transfer = retention.get(("second_context_transfer", control), {})
            control_teacher = retention.get(("fixed_context_teacher_reference", control), {})
            same_student_key = f"{PRIMARY_VARIANT}_support_vs_{control}"
            row = {
                "packet": packet["packet"],
                "control_variant": control,
                "same_student_acsr_minus_control_ce_loss": _number(
                    same_student.get(same_student_key, {}).get("acsr_minus_control_ce_loss")
                ),
                "anchor_support_churn_after_transfer_acsr": _number(
                    acsr_transfer.get("anchor_support_churn_after_transfer")
                ),
                "anchor_support_churn_after_transfer_control": _number(
                    control_transfer.get("anchor_support_churn_after_transfer")
                ),
                "anchor_logit_mse_after_transfer_acsr": _number(
                    acsr_transfer.get("anchor_logit_mse_after_transfer")
                ),
                "anchor_logit_mse_after_transfer_control": _number(
                    control_transfer.get("anchor_logit_mse_after_transfer")
                ),
                "transfer_ce_improvement_acsr": _number(
                    acsr_transfer.get("transfer_ce_improvement")
                ),
                "transfer_ce_improvement_control": _number(
                    control_transfer.get("transfer_ce_improvement")
                ),
                "teacher_support_churn_acsr": _number(
                    acsr_teacher.get("teacher_support_churn")
                ),
                "teacher_support_churn_control": _number(
                    control_teacher.get("teacher_support_churn")
                ),
            }
            row["acsr_minus_control_anchor_support_churn"] = _delta(
                row["anchor_support_churn_after_transfer_acsr"],
                row["anchor_support_churn_after_transfer_control"],
            )
            row["acsr_minus_control_anchor_logit_mse"] = _delta(
                row["anchor_logit_mse_after_transfer_acsr"],
                row["anchor_logit_mse_after_transfer_control"],
            )
            row["acsr_minus_control_transfer_ce_improvement"] = _delta(
                row["transfer_ce_improvement_acsr"],
                row["transfer_ce_improvement_control"],
            )
            row["acsr_minus_control_teacher_support_churn"] = _delta(
                row["teacher_support_churn_acsr"],
                row["teacher_support_churn_control"],
            )
            row["passes_same_student_ce"] = _lt(
                row["same_student_acsr_minus_control_ce_loss"], 0.0
            )
            row["passes_anchor_support_churn"] = _le(
                row["acsr_minus_control_anchor_support_churn"], 0.0
            )
            row["passes_anchor_logit_mse"] = _lt(
                row["acsr_minus_control_anchor_logit_mse"], 0.0
            )
            row["passes_teacher_reference_churn"] = _lt(
                row["acsr_minus_control_teacher_support_churn"], 0.0
            )
            rows.append(row)
    return rows


def _aggregate_rows(comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for control in CONTROL_VARIANTS:
        control_rows = [
            row for row in comparison_rows if row.get("control_variant") == control
        ]
        rows.append(
            {
                "control_variant": control,
                "packet_count": len(control_rows),
                "mean_same_student_acsr_minus_control_ce_loss": _mean_key(
                    control_rows, "same_student_acsr_minus_control_ce_loss"
                ),
                "mean_acsr_minus_control_anchor_support_churn": _mean_key(
                    control_rows, "acsr_minus_control_anchor_support_churn"
                ),
                "mean_acsr_minus_control_anchor_logit_mse": _mean_key(
                    control_rows, "acsr_minus_control_anchor_logit_mse"
                ),
                "mean_acsr_minus_control_transfer_ce_improvement": _mean_key(
                    control_rows, "acsr_minus_control_transfer_ce_improvement"
                ),
                "mean_acsr_minus_control_teacher_support_churn": _mean_key(
                    control_rows, "acsr_minus_control_teacher_support_churn"
                ),
                "all_gates_pass": bool(control_rows)
                and all(
                    row["passes_same_student_ce"]
                    and row["passes_anchor_support_churn"]
                    and row["passes_anchor_logit_mse"]
                    and row["passes_teacher_reference_churn"]
                    for row in control_rows
                ),
            }
        )
    return rows


def _criteria(
    packet_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_comparisons = len(packet_rows) * len(CONTROL_VARIANTS)
    return [
        _criterion(
            "all_packet_artifacts_present",
            all(row["present"] and row["required_files_present"] for row in packet_rows),
            "each packet has summary, same_student_metrics, and retention_churn_metrics",
            [
                (row["packet"], row["present"], row["required_files_present"])
                for row in packet_rows
            ],
        ),
        _criterion(
            "all_packets_pass",
            all(row.get("status") == "pass" for row in packet_rows),
            "summary.status == pass for each packet",
            [(row["packet"], row.get("status")) for row in packet_rows],
        ),
        _criterion(
            "all_expected_control_comparisons_present",
            len(comparison_rows) == expected_comparisons
            and all(
                row["same_student_acsr_minus_control_ce_loss"] is not None
                and row["anchor_support_churn_after_transfer_acsr"] is not None
                and row["anchor_support_churn_after_transfer_control"] is not None
                and row["anchor_logit_mse_after_transfer_acsr"] is not None
                and row["anchor_logit_mse_after_transfer_control"] is not None
                and row["teacher_support_churn_acsr"] is not None
                and row["teacher_support_churn_control"] is not None
                for row in comparison_rows
            ),
            f"{expected_comparisons} complete ACSR-vs-null rows",
            len(comparison_rows),
        ),
        _criterion(
            "same_student_support_ce_beats_nulls",
            all(row["passes_same_student_ce"] for row in comparison_rows),
            "ACSR minus control forced CE < 0 for every packet/control",
            [
                (
                    row["packet"],
                    row["control_variant"],
                    row["same_student_acsr_minus_control_ce_loss"],
                )
                for row in comparison_rows
            ],
        ),
        _criterion(
            "cross_context_anchor_support_churn_not_worse",
            all(row["passes_anchor_support_churn"] for row in comparison_rows),
            "ACSR minus control post-transfer anchor support churn <= 0",
            [
                (
                    row["packet"],
                    row["control_variant"],
                    row["acsr_minus_control_anchor_support_churn"],
                )
                for row in comparison_rows
            ],
        ),
        _criterion(
            "cross_context_anchor_logit_mse_better",
            all(row["passes_anchor_logit_mse"] for row in comparison_rows),
            "ACSR minus control post-transfer anchor logit MSE < 0",
            [
                (
                    row["packet"],
                    row["control_variant"],
                    row["acsr_minus_control_anchor_logit_mse"],
                )
                for row in comparison_rows
            ],
        ),
        _criterion(
            "fixed_teacher_reference_churn_better",
            all(row["passes_teacher_reference_churn"] for row in comparison_rows),
            "ACSR minus control teacher-reference support churn < 0",
            [
                (
                    row["packet"],
                    row["control_variant"],
                    row["acsr_minus_control_teacher_support_churn"],
                )
                for row in comparison_rows
            ],
        ),
    ]


def _failures(
    packet_rows: list[dict[str, Any]], criteria: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    failures = []
    for row in packet_rows:
        if not row["present"] or not row["required_files_present"]:
            failures.append(
                {
                    "source": row["packet"],
                    "field": "required_artifacts",
                    "expected": list(REQUIRED_FILES),
                    "actual": row["audit_dir"],
                }
            )
        if row.get("status") != "pass":
            failures.append(
                {
                    "source": row["packet"],
                    "field": "status",
                    "expected": "pass",
                    "actual": row.get("status"),
                }
            )
    for criterion in criteria:
        if not criterion["passed"]:
            failures.append(
                {
                    "source": "cross_context_retention_churn_gate",
                    "field": criterion["criterion"],
                    "expected": criterion["threshold"],
                    "actual": criterion["actual"],
                }
            )
    return failures


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
        if key.strip() in {"strategic_change_level", "notify_ben", "recommended_next_action"}:
            values[key.strip()] = value.strip()
    notify_ben = values.get("notify_ben", "").lower() == "true"
    strategic_change_level = values.get("strategic_change_level")
    return {
        "present": True,
        "strategic_change_level": strategic_change_level,
        "notify_ben": notify_ben,
        "ben_notification_required": notify_ben or strategic_change_level == "major",
        "recommended_next_action": values.get("recommended_next_action"),
    }


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
                    "The local ACSR pilot and fetched RunPod replication already "
                    "exist; this probe advances the status-file next non-CE control."
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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ACSR Cross-Context Retention/Churn Probe",
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
        "## Boundaries",
        "",
        "- This is a stronger non-CE control, not default-router promotion.",
        "- Full causal mechanism claims remain blocked pending held-out oracle-regret and functional-churn interventions.",
        "- The full-context teacher remains nondeployable oracle diagnostic evidence.",
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


def _mean_key(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_number(row.get(key)) for row in rows]
    numbers = [value for value in values if value is not None]
    return mean(numbers) if numbers else None


def _delta(left: Any, right: Any) -> float | None:
    left_float = _number(left)
    right_float = _number(right)
    if left_float is None or right_float is None:
        return None
    return left_float - right_float


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
    parser.add_argument("--audit-dir", action="append", type=Path, dest="audit_dirs")
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    audit_dirs = tuple(args.audit_dirs) if args.audit_dirs else DEFAULT_AUDIT_DIRS
    summary = run_acsr_same_student_cross_context_retention_churn_probe(
        audit_dirs=audit_dirs,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
