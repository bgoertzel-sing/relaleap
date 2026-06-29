"""Reconcile post-negative branch handoffs without starting GPU validation."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_CORE_CLOSEOUT = Path("results/reports/core_periphery_negative_evidence_closeout/summary.json")
DEFAULT_PAIR_CLOSEOUT = Path("results/reports/dense_teacher_pair_composer_pregate_closeout/summary.json")
DEFAULT_LOW_CHURN = Path("results/reports/low_churn_mlp_residual_control_pilot/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/post_negative_loop_reconciliation")

NEXT_ACTION = "run_local_mechanism_source_inventory_before_new_branch"
REPAIR_ACTION = "repair_post_negative_reconciliation_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "loop_edges.csv",
    "candidate_actions.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_post_negative_loop_reconciliation(
    *,
    core_closeout_path: Path = DEFAULT_CORE_CLOSEOUT,
    pair_closeout_path: Path = DEFAULT_PAIR_CLOSEOUT,
    low_churn_path: Path = DEFAULT_LOW_CHURN,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    urgent_review_status: str = "not_run",
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Select one conservative local step when negative branch reports loop."""

    start = time.time()
    core = _read_json(core_closeout_path)
    pair = _read_json(pair_closeout_path)
    low_churn = _read_json(low_churn_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("core_periphery_negative_closeout", core_closeout_path, core),
        _source_row("dense_teacher_pair_composer_closeout", pair_closeout_path, pair),
        _source_row("low_churn_mlp_control_pilot", low_churn_path, low_churn),
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
    evidence = _evidence(core, pair, low_churn, strategy, urgent_review_status)
    loop_edges = _loop_edges(evidence)
    criteria = _criteria(source_rows, evidence, loop_edges)
    failures = [row for row in criteria if not row["passed"] and row["severity"] == "hard"]
    candidate_actions = _candidate_actions(failures, evidence)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "post_negative_loop_reconciliation_failed_closed"
        selected_next_action = REPAIR_ACTION
        next_step = "repair missing post-negative source artifacts before choosing another branch"
        claim_status = "post_negative_reconciliation_sources_incomplete"
        rationale = "The reconciliation report cannot choose a branch from missing or incoherent source artifacts."
    else:
        status = "pass"
        decision = "post_negative_loop_reconciliation_recorded"
        selected_next_action = selected[0]["candidate_action"]
        next_step = selected[0]["next_step"]
        claim_status = selected[0]["claim_status"]
        rationale = selected[0]["reason"]

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_action": selected_next_action,
        "next_step": next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "backend_policy": "local reconciliation only; RunPod/Colab remain blocked",
        "urgent_review_status": urgent_review_status,
        "source_rows": source_rows,
        "evidence": evidence,
        "loop_edges": loop_edges,
        "gate_criteria": criteria,
        "candidate_actions": candidate_actions,
        "strategy_review": strategy,
        "strategy_response": _strategy_response(strategy, urgent_review_status),
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
    core: dict[str, Any],
    pair: dict[str, Any],
    low_churn: dict[str, Any],
    strategy: dict[str, Any],
    urgent_review_status: str,
) -> dict[str, Any]:
    return {
        "core_status": core.get("status"),
        "core_selected_next_action": core.get("selected_next_action"),
        "core_next_step": core.get("next_step"),
        "pair_status": pair.get("status"),
        "pair_selected_next_action": pair.get("selected_next_action"),
        "pair_next_step": pair.get("next_step"),
        "low_churn_status": low_churn.get("status"),
        "low_churn_scientific_gate": low_churn.get("scientific_gate"),
        "low_churn_selected_next_step": low_churn.get("selected_next_step"),
        "strategy_recommended_next_action": strategy.get("recommended_next_action"),
        "strategy_verdict": strategy.get("verdict"),
        "ben_notification_required": strategy.get("ben_notification_required"),
        "urgent_review_status": urgent_review_status,
    }


def _loop_edges(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _edge(
            "pair_composer_closeout_to_core_periphery",
            evidence["pair_selected_next_action"] == "redirect_to_core_periphery_predictive_coding_column_design",
            evidence["pair_next_step"],
            "pair-composer branch is locally negative and redirects to core/periphery",
        ),
        _edge(
            "core_periphery_closeout_to_dense_controls",
            evidence["core_selected_next_action"] == "demote_current_core_periphery_mechanism_to_diagnostic_status",
            evidence["core_next_step"],
            "current core/periphery mechanism is locally negative and redirects away from itself",
        ),
        _edge(
            "low_churn_control_blocks_gpu",
            evidence["low_churn_scientific_gate"] == "blocked",
            evidence["low_churn_selected_next_step"],
            "latest matched MLP control pilot blocks GPU advancement",
        ),
    ]


def _criteria(
    source_rows: list[dict[str, Any]],
    evidence: dict[str, Any],
    loop_edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _criterion(
            "required_negative_sources_present",
            all(row["present"] for row in source_rows[:3]),
            "hard",
            "core closeout, pair-composer closeout, and low-churn pilot summaries exist",
            [row["path"] for row in source_rows[:3] if row["present"]],
            "missing required post-negative source",
        ),
        _criterion(
            "required_negative_sources_passed_runtime",
            all(
                evidence[key] == "pass"
                for key in ("core_status", "pair_status", "low_churn_status")
            ),
            "hard",
            "required source reports passed runtime/artifact gates",
            {
                "core": evidence["core_status"],
                "pair": evidence["pair_status"],
                "low_churn": evidence["low_churn_status"],
            },
            "at least one required source is not runtime-interpretable",
        ),
        _criterion(
            "negative_loop_detected",
            all(row["present"] for row in loop_edges),
            "claim",
            "existing branch reports would otherwise send automation between closed local branches",
            loop_edges,
            "no loop detected; follow the latest selected source action instead",
        ),
        _criterion(
            "urgent_review_unavailable_or_nonblocking",
            evidence["urgent_review_status"] in {"not_run", "timeout", "failed", "completed"},
            "hard",
            "urgent strategy-review status is explicitly recorded",
            evidence["urgent_review_status"],
            "urgent review status not recorded",
        ),
    ]


def _candidate_actions(failures: list[dict[str, Any]], evidence: dict[str, Any]) -> list[dict[str, str]]:
    if failures:
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "required reconciliation sources are missing or failed",
                "repair the missing post-negative source artifacts",
                "source_repair_required",
            ),
            _candidate(
                NEXT_ACTION,
                "blocked",
                "cannot run inventory from incoherent branch sources",
                "rerun after source repair",
                "source_repair_required",
            ),
        ]
    return [
        _candidate(
            NEXT_ACTION,
            "selected",
            "current sparse/core-periphery/pair-composer and low-churn dense-control reports are all local-negative or GPU-blocking, and the urgent review timed out",
            "write a local mechanism source-inventory report that lists remaining non-duplicative evidence gaps before any new mechanism implementation or GPU validation",
            "negative_loop_reconciled_no_gpu_or_promotion",
        ),
        _candidate(
            "rerun_core_periphery_or_pair_composer",
            "rejected",
            "both branches already have command-driven negative closeouts",
            "do not duplicate completed local work",
            "duplicate_closed_branch_rejected",
        ),
        _candidate(
            "run_runpod_validation",
            "rejected",
            "no source report currently requires GPU and local gates are blocked",
            "keep RunPod unused for this automation run",
            "gpu_validation_blocked_by_local_gates",
        ),
        _candidate(
            "follow_latest_review_low_churn_again",
            "rejected" if "low-churn" in str(evidence["strategy_recommended_next_action"]).lower() else "deferred",
            "the low-churn MLP pilot has already been implemented and blocks advancement",
            "do not duplicate the completed low-churn pilot",
            "low_churn_recommendation_already_satisfied",
        ),
    ]


def _strategy_response(strategy: dict[str, Any], urgent_review_status: str) -> dict[str, Any]:
    return {
        "review_recommendation": strategy.get("recommended_next_action"),
        "urgent_review_status": urgent_review_status,
        "disposition": "deferred_after_timeout" if urgent_review_status == "timeout" else "recorded",
        "ben_should_be_notified": bool(strategy.get("ben_notification_required")),
        "reason": (
            "The urgent review did not return within the bounded run, so this report records "
            "the timeout and chooses a conservative local inventory step rather than a new "
            "scientific branch or GPU validation."
        ),
    }


def _criterion(
    criterion: str,
    passed: bool,
    severity: str,
    threshold: str,
    actual: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "severity": severity,
        "threshold": threshold,
        "actual": actual,
        "failure_reason": "" if passed else failure_reason,
    }


def _edge(edge: str, present: bool, next_step: Any, interpretation: str) -> dict[str, Any]:
    return {
        "edge": edge,
        "present": bool(present),
        "next_step": next_step,
        "interpretation": interpretation,
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
        "status": payload.get("status") if payload else "missing",
        "decision": payload.get("decision") if payload else "",
        "claim_status": payload.get("claim_status") if payload else "",
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    values: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"strategic_change_level", "notify_ben", "recommended_next_action", "verdict"}:
            values[key] = value.strip()
    return {
        "present": path.is_file(),
        "strategic_change_level": values.get("strategic_change_level", "missing"),
        "notify_ben": values.get("notify_ben", "false"),
        "recommended_next_action": values.get("recommended_next_action", ""),
        "verdict": values.get("verdict", ""),
        "ben_notification_required": values.get("notify_ben", "false").lower() == "true"
        or values.get("strategic_change_level") == "major",
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "loop_edges.csv", summary["loop_edges"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _notes(summary: dict[str, Any]) -> str:
    selected = summary["selected_next_action"]
    return "\n".join(
        [
            "# Post-Negative Loop Reconciliation",
            "",
            f"- Status: `{summary['status']}`.",
            f"- Decision: `{summary['decision']}`.",
            f"- Selected next action: `{selected}`.",
            f"- GPU required now: `{summary['requires_gpu_now']}`.",
            f"- Urgent review status: `{summary['urgent_review_status']}`.",
            "",
            summary["rationale"],
            "",
        ]
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-closeout", type=Path, default=DEFAULT_CORE_CLOSEOUT)
    parser.add_argument("--pair-closeout", type=Path, default=DEFAULT_PAIR_CLOSEOUT)
    parser.add_argument("--low-churn", type=Path, default=DEFAULT_LOW_CHURN)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--urgent-review-status", default="not_run")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_post_negative_loop_reconciliation(
        core_closeout_path=args.core_closeout,
        pair_closeout_path=args.pair_closeout,
        low_churn_path=args.low_churn,
        strategy_review_path=args.strategy_review,
        urgent_review_status=args.urgent_review_status,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
