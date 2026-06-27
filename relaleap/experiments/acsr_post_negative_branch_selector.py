"""Post-negative ACSR synthesis and branch selector."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_COMMON_BENCHMARK = Path("results/reports/acsr_common_causal_residual_benchmark/summary.json")
DEFAULT_DENSE_TEACHER = Path(
    "results/audits/token_larger_dense_teacher_residual_distillation_comparison/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/acsr_post_negative_branch_selector")

TASK_FREE_RETENTION_ACTION = "task_free_continual_learning_retention_assay"
FINITE_UPDATE_COMMUTATOR_ACTION = "finite_update_commutator_reduction_assay"
RETIRE_ACSR_ACTION = "retire_acsr_promotion_in_favor_of_dense_residual_controls"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_acsr_post_negative_branch_selector(
    *,
    common_benchmark_path: Path = DEFAULT_COMMON_BENCHMARK,
    dense_teacher_path: Path = DEFAULT_DENSE_TEACHER,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Select one non-GPU branch after negative ACSR dense-control evidence."""

    start = time.time()
    common = _read_json(common_benchmark_path)
    dense_teacher = _read_json(dense_teacher_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("common_sparse_vs_dense_benchmark", common_benchmark_path, common),
        _source_row("dense_teacher_distillation_pilot", dense_teacher_path, dense_teacher),
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
    evidence = _evidence_snapshot(common, dense_teacher, strategy)
    failures = _failures(source_rows)
    candidate_actions = _candidate_actions(evidence, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "acsr_post_negative_branch_selection_failed_closed"
        selected_next_action = None
        next_step = "repair missing post-negative ACSR source artifacts"
        rationale = (
            "The selector could not make a bounded branch choice because required "
            "common-benchmark or dense-teacher artifacts were missing."
        )
    else:
        status = "pass"
        decision = "acsr_post_negative_branch_selected"
        selected_next_action = selected[0]["candidate_action"]
        next_step = selected[0]["next_step"]
        rationale = selected[0]["reason"]

    summary = {
        "status": status,
        "decision": decision,
        "selected_next_action": selected_next_action,
        "next_step": next_step,
        "claim_statuses": {
            "acsr_sparse_support_identity": "not_supported_by_current_dense_controls",
            "acsr_default_router_promotion": "retired_unless_future_common_dense_controls_reverse",
            "dense_residual_controls": "active_comparison_baseline",
            "ben_notification": "required" if strategy["ben_notification_required"] else "not_required",
        },
        "candidate_actions": candidate_actions,
        "source_rows": source_rows,
        "evidence": evidence,
        "strategy_review": strategy,
        "failures": failures,
        "rationale": rationale,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, source_rows, candidate_actions)
    return summary


def _evidence_snapshot(
    common: dict[str, Any],
    dense_teacher: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    common_failures = common.get("failures") if isinstance(common.get("failures"), list) else []
    dense_failures = dense_teacher.get("failures") if isinstance(dense_teacher.get("failures"), list) else []
    return {
        "common_benchmark_status": common.get("status"),
        "common_benchmark_decision": common.get("decision"),
        "common_claim_status": common.get("claim_status"),
        "common_selected_next_step": common.get("selected_next_step"),
        "common_sparse_beats_dense": not any(
            row.get("criterion") == "sparse_beats_causal_dense" and not row.get("passed", False)
            for row in common_failures
        ),
        "dense_teacher_status": dense_teacher.get("status"),
        "dense_teacher_decision": dense_teacher.get("decision"),
        "dense_teacher_claim_status": dense_teacher.get("claim_status"),
        "dense_teacher_selected_next_step": dense_teacher.get("selected_next_step"),
        "dense_teacher_ce_loss": _float_or_none(dense_teacher.get("dense_teacher_ce_loss")),
        "acsr_student_ce_loss": _variant_metric(
            dense_teacher,
            "acsr_predicted_future_support",
            "ce_loss",
        ),
        "acsr_compresses_dense_teacher": not any(
            row.get("criterion") == "acsr_ce_not_worse_than_teacher_by_large_margin"
            and not row.get("passed", False)
            for row in dense_failures
        ),
        "strategy_verdict": strategy.get("verdict"),
        "strategy_recommended_next_action": strategy.get("recommended_next_action"),
        "strategy_major_pivot": strategy.get("strategic_change_level") == "major",
        "ben_notification_required": strategy.get("ben_notification_required"),
    }


def _candidate_actions(
    evidence: dict[str, Any],
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if failures:
        return [
            _candidate(
                TASK_FREE_RETENTION_ACTION,
                "blocked",
                "required post-negative source artifacts are missing",
                "repair missing post-negative ACSR source artifacts",
            ),
            _candidate(
                FINITE_UPDATE_COMMUTATOR_ACTION,
                "blocked",
                "required post-negative source artifacts are missing",
                "repair missing post-negative ACSR source artifacts",
            ),
            _candidate(
                RETIRE_ACSR_ACTION,
                "blocked",
                "required post-negative source artifacts are missing",
                "repair missing post-negative ACSR source artifacts",
            ),
        ]

    common_supported = evidence["common_benchmark_status"] == "pass" and evidence["common_sparse_beats_dense"]
    dense_teacher_supported = (
        evidence["dense_teacher_status"] == "pass" and evidence["acsr_compresses_dense_teacher"]
    )
    if common_supported and dense_teacher_supported:
        return [
            _candidate(
                TASK_FREE_RETENTION_ACTION,
                "selected",
                "sparse support survived dense controls and dense-teacher compression, so retention is the next mechanism assay",
                "implement a task-free continual-learning retention assay with dense residual controls",
            ),
            _candidate(
                FINITE_UPDATE_COMMUTATOR_ACTION,
                "deferred",
                "retention should first test whether the supported sparse mechanism is reusable across tasks",
                "run after retention if interference remains the active blocker",
            ),
            _candidate(
                RETIRE_ACSR_ACTION,
                "rejected",
                "current artifacts would support further sparse-mechanism testing",
                "not applicable unless dense controls reverse the support",
            ),
        ]
    if common_supported and not dense_teacher_supported:
        return [
            _candidate(
                TASK_FREE_RETENTION_ACTION,
                "deferred",
                "dense-teacher compression failed, so cross-task reuse is premature",
                "run only if commutator evidence restores a sparse mechanism claim",
            ),
            _candidate(
                FINITE_UPDATE_COMMUTATOR_ACTION,
                "selected",
                "common sparse-vs-dense evidence is not fully negative but dense-teacher compression failed; finite-update interference is the smaller next mechanism assay",
                "implement a finite-update commutator reduction assay with causal dense controls",
            ),
            _candidate(
                RETIRE_ACSR_ACTION,
                "deferred",
                "one sparse-vs-dense gate remains supportive",
                "revisit after finite-update commutator evidence",
            ),
        ]
    return [
        _candidate(
            TASK_FREE_RETENTION_ACTION,
            "rejected",
            "task-free retention would test reuse before sparse support identity beats dense controls",
            "not the next bounded branch under negative dense-control evidence",
        ),
        _candidate(
            FINITE_UPDATE_COMMUTATOR_ACTION,
            "rejected",
            "finite-update mitigation would optimize a sparse mechanism that the common dense control currently explains better",
            "not the next bounded branch under negative dense-control evidence",
        ),
        _candidate(
            RETIRE_ACSR_ACTION,
            "selected",
            "both the common sparse-vs-dense benchmark and dense-teacher pilot failed their sparse-support gates; ACSR promotion should be retired unless a future common dense-control benchmark reverses this",
            "write the next experiment against dense residual controls rather than ACSR/default-router promotion",
        ),
    ]


def _candidate(action: str, disposition: str, reason: str, next_step: str) -> dict[str, str]:
    return {
        "candidate_action": action,
        "disposition": disposition,
        "reason": reason,
        "next_step": next_step,
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


def _failures(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows[:2]:
        if not row["present"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "source_artifact",
                    "failure_reason": f"{row['path']} is missing",
                }
            )
    return failures


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    candidate_actions: list[dict[str, Any]],
) -> None:
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
        out_dir / "candidate_actions.csv",
        ["candidate_action", "disposition", "reason", "next_step"],
        candidate_actions,
    )
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ACSR Post-Negative Branch Selector",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Next step: {summary['next_step']}",
        f"- Ben notification: `{summary['claim_statuses']['ben_notification']}`",
        "",
        summary["rationale"],
        "",
        "This report is local and non-GPU. It consumes the common sparse-vs-dense "
        "benchmark and dense-teacher distillation pilot, then chooses exactly one "
        "post-negative branch.",
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
            if key == "verdict":
                fields["verdict"] = value
    notify = str(fields.get("notify_ben")).lower() == "true"
    fields["ben_notification_required"] = notify or fields.get("strategic_change_level") == "major"
    return fields


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _variant_metric(payload: dict[str, Any], variant: str, metric: str) -> float | None:
    for row in payload.get("variant_rows", []):
        if isinstance(row, dict) and row.get("variant") == variant:
            return _float_or_none(row.get(metric))
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-benchmark", type=Path, default=DEFAULT_COMMON_BENCHMARK)
    parser.add_argument("--dense-teacher", type=Path, default=DEFAULT_DENSE_TEACHER)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_acsr_post_negative_branch_selector(
        common_benchmark_path=args.common_benchmark,
        dense_teacher_path=args.dense_teacher,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
