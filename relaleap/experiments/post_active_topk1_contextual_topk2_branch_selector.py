"""Reconcile active top-k-1 diagnostics before returning to contextual top-k-2."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_ACTIVE_TOPK1_SYNTHESIS = Path(
    "results/reports/token_larger_active_topk1_causal_retention_synthesis/summary.json"
)
DEFAULT_CONTEXTUAL_FAILURE = Path(
    "results/reports/token_larger_contextual_router_regret_churn_failure_inspection/summary.json"
)
DEFAULT_TOPK2_VALUE_CLOSEOUT = Path(
    "results/reports/token_larger_promoted_topk2_post_localization_closeout/summary.json"
)
DEFAULT_TOPK2_RETURN = Path(
    "results/reports/token_larger_contextual_topk2_support_routing_return/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/post_active_topk1_contextual_topk2_branch_selector"
)

SELECTED_DECISION = "post_active_topk1_contextual_topk2_branch_selected"
FAILED_DECISION = "post_active_topk1_contextual_topk2_branch_failed_closed"
SELECTED_ACTION = "design_support_quality_preserving_contextual_topk2_pregate"
REPAIR_ACTION = "repair_post_active_topk1_contextual_topk2_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "candidate_actions.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_post_active_topk1_contextual_topk2_branch_selector(
    *,
    active_topk1_synthesis_path: Path = DEFAULT_ACTIVE_TOPK1_SYNTHESIS,
    contextual_failure_path: Path = DEFAULT_CONTEXTUAL_FAILURE,
    topk2_value_closeout_path: Path = DEFAULT_TOPK2_VALUE_CLOSEOUT,
    topk2_return_path: Path = DEFAULT_TOPK2_RETURN,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Select one local post-top-k-1 step without promoting failed branches."""

    start = time.time()
    active = _read_json(active_topk1_synthesis_path)
    contextual = _read_json(contextual_failure_path)
    value_closeout = _read_json(topk2_value_closeout_path)
    topk2_return = _read_json(topk2_return_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("active_topk1_causal_retention_synthesis", active_topk1_synthesis_path, active),
        _source_row("contextual_router_regret_churn_failure_inspection", contextual_failure_path, contextual),
        _source_row("promoted_topk2_value_router_family_closeout", topk2_value_closeout_path, value_closeout),
        _source_row("contextual_topk2_support_routing_return", topk2_return_path, topk2_return),
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
        },
    ]
    evidence = _evidence(active, contextual, value_closeout, topk2_return, strategy)
    criteria = _criteria(source_rows, evidence)
    failures = [row for row in criteria if not row["passed"]]
    selected_ok = not failures
    selected_action = SELECTED_ACTION if selected_ok else REPAIR_ACTION
    candidate_actions = _candidate_actions(selected_action)

    if selected_ok:
        status = "pass"
        decision = SELECTED_DECISION
        claim_status = "topk2_main_loop_selected_for_local_support_quality_redesign"
        selected_next_step = (
            "implement a local support-quality-preserving contextual top-k-2 pregate "
            "that keeps active top-k-1 as a retention/churn control, rejects simple "
            "value/router mitigations as closed, and gates any GPU work on improved "
            "oracle-support regret plus functional churn versus the linear top-k-2 control"
        )
        rationale = (
            "Active top-k-1 is useful only as a low-churn diagnostic bracket because "
            "its deployable gate failed. The causal-feature-safe contextual top-k-2 "
            "router wins CE but fails support-quality gates, while the top-k-2 "
            "value/router mitigation family is closed. The least duplicative next "
            "step is therefore a local pregate for a support-quality-preserving "
            "contextual top-k-2 redesign, not GPU validation or another closeout."
        )
    else:
        status = "fail"
        decision = FAILED_DECISION
        claim_status = "post_active_topk1_contextual_topk2_sources_incomplete"
        selected_next_step = "repair missing or contradictory source artifacts before selecting a branch"
        rationale = "Required handoff evidence is missing or contradictory."

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_action": selected_action,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local report only; Colab/RunPod remain blocked until a local support-quality gate passes",
        "source_rows": source_rows,
        "evidence": evidence,
        "gate_criteria": criteria,
        "candidate_actions": candidate_actions,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
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
    active: dict[str, Any],
    contextual: dict[str, Any],
    value_closeout: dict[str, Any],
    topk2_return: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    active_evidence = active.get("evidence", {}) if isinstance(active.get("evidence"), dict) else {}
    active_signals = (
        active_evidence.get("source_signals", {})
        if isinstance(active_evidence.get("source_signals"), dict)
        else {}
    )
    contextual_evidence = (
        contextual.get("evidence", {}) if isinstance(contextual.get("evidence"), dict) else {}
    )
    return {
        "active_topk1_status": active.get("status"),
        "active_topk1_decision": active.get("decision"),
        "active_topk1_claim_status": active.get("claim_status"),
        "active_topk1_deployable_gate_failed": bool(
            active_evidence.get("deployable_context_gate_failed")
            or active_signals.get("deployable_gate_passes_pre_registered_criteria") is False
        ),
        "active_topk1_local_bracket_supported": bool(
            active_evidence.get("local_retention_churn_bracket_supported")
            or active_signals.get("retention_branch_supported")
        ),
        "contextual_failure_status": contextual.get("status"),
        "contextual_failure_decision": contextual.get("decision"),
        "contextual_claim_status": contextual.get("claim_status"),
        "contextual_ce_win": bool(
            contextual_evidence.get("all_folds_causal_ce_beats_linear")
        ),
        "contextual_regret_worse": bool(
            contextual_evidence.get("all_folds_causal_regret_worse_than_linear")
        ),
        "contextual_churn_worse": bool(
            contextual_evidence.get("all_folds_causal_churn_worse_than_linear")
        ),
        "topk2_value_closeout_status": value_closeout.get("status"),
        "topk2_value_closeout_decision": value_closeout.get("decision"),
        "topk2_value_claim_status": value_closeout.get("claim_status"),
        "topk2_return_status": topk2_return.get("status"),
        "topk2_return_decision": topk2_return.get("decision"),
        "topk2_return_selected_action": topk2_return.get("selected_next_action"),
        "strategy_verdict": strategy.get("verdict"),
        "strategy_recommended_next_action": strategy.get("recommended_next_action"),
        "ben_notification_required": strategy.get("ben_notification_required"),
    }


def _criteria(source_rows: list[dict[str, Any]], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    required_present = all(row["present"] for row in source_rows[:4])
    active_demoted = (
        evidence["active_topk1_status"] == "pass"
        and evidence["active_topk1_decision"] == "causal_retention_claim_blocked_by_deployable_gate"
        and evidence["active_topk1_deployable_gate_failed"]
    )
    contextual_failure_recorded = (
        evidence["contextual_failure_status"] == "pass"
        and evidence["contextual_failure_decision"]
        == "contextual_router_regret_churn_failure_inspection_recorded"
        and evidence["contextual_ce_win"]
        and evidence["contextual_regret_worse"]
        and evidence["contextual_churn_worse"]
    )
    topk2_value_closed = (
        evidence["topk2_value_closeout_status"] == "pass"
        and evidence["topk2_value_closeout_decision"]
        in {
            "promoted_topk2_value_router_family_closed",
            "promoted_topk2_mitigation_closeout_no_promotion",
        }
    )
    return_report_present = (
        evidence["topk2_return_status"] == "pass"
        and evidence["topk2_return_decision"] == "contextual_topk2_support_routing_return_selected"
    )
    return [
        _criterion(
            "required_source_artifacts_present",
            required_present,
            "active top-k-1, contextual failure, top-k-2 closeout, and return selector summaries present",
            {row["source"]: row["present"] for row in source_rows[:4]},
        ),
        _criterion(
            "active_topk1_demoted_to_diagnostic_bracket",
            active_demoted,
            "active top-k-1 broad causal-retention claim blocked by deployable gate",
            {
                "decision": evidence["active_topk1_decision"],
                "deployable_gate_failed": evidence["active_topk1_deployable_gate_failed"],
            },
        ),
        _criterion(
            "contextual_router_failure_mode_recorded",
            contextual_failure_recorded,
            "contextual top-k-2 CE win coexists with worse oracle regret and churn",
            {
                "decision": evidence["contextual_failure_decision"],
                "ce_win": evidence["contextual_ce_win"],
                "regret_worse": evidence["contextual_regret_worse"],
                "churn_worse": evidence["contextual_churn_worse"],
            },
        ),
        _criterion(
            "topk2_value_router_mitigations_closed",
            topk2_value_closed,
            "top-k-2 value/router mitigation family closed without promotion",
            evidence["topk2_value_closeout_decision"],
        ),
        _criterion(
            "contextual_topk2_return_artifact_present",
            return_report_present,
            "contextual top-k-2 return selector exists and passed",
            {
                "decision": evidence["topk2_return_decision"],
                "selected_action": evidence["topk2_return_selected_action"],
            },
        ),
    ]


def _candidate_actions(selected_action: str) -> list[dict[str, Any]]:
    candidates = [
        {
            "candidate_action": SELECTED_ACTION,
            "disposition": "selected" if selected_action == SELECTED_ACTION else "blocked",
            "reason": (
                "the handoff evidence supports a local top-k-2 redesign focused on "
                "oracle-regret and functional-churn quality, with active top-k-1 as control"
            ),
        },
        {
            "candidate_action": "promote_active_topk1_causal_retention",
            "disposition": "disqualified",
            "reason": "deployable context gate failed, so singleton efficacy remains diagnostic only",
        },
        {
            "candidate_action": "rerun_topk2_value_router_mitigation_family",
            "disposition": "disqualified",
            "reason": "post-localization closeout already closed this family without promotion",
        },
        {
            "candidate_action": "use_colab_or_runpod_gpu_validation",
            "disposition": "blocked",
            "reason": "no local support-quality gate has passed for the selected redesign",
        },
    ]
    if selected_action == REPAIR_ACTION:
        candidates.insert(
            0,
            {
                "candidate_action": REPAIR_ACTION,
                "disposition": "selected",
                "reason": "required source summaries are missing or contradictory",
            },
        )
    return candidates


def _criterion(name: str, passed: bool, expected: object, actual: object) -> dict[str, Any]:
    return {"criterion": name, "passed": bool(passed), "expected": expected, "actual": actual}


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status", "missing"),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
    }


def _read_json(path: Path) -> dict[str, Any]:
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
            "strategic_change_level": "",
            "notify_ben": False,
            "ben_notification_required": False,
            "verdict": "",
            "recommended_next_action": "",
        }
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:12]:
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    notify = header.get("notify_ben", "").lower() == "true"
    level = header.get("strategic_change_level", "")
    return {
        "present": True,
        "strategic_change_level": level,
        "notify_ben": notify,
        "ben_notification_required": notify or level.lower() == "major",
        "verdict": header.get("verdict", ""),
        "recommended_next_action": header.get("recommended_next_action", ""),
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "no external strategy review present; no recommendation handled"
    if strategy["ben_notification_required"]:
        return "review requires Ben notification; direction shift preserved in status"
    return (
        "latest review recommendation was already implemented and closed by local "
        "flat-control evidence; no new recommendation is rejected or deferred"
    )


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
        out_dir / "candidate_actions.csv",
        ["candidate_action", "disposition", "reason"],
        summary["candidate_actions"],
    )
    _write_csv(
        out_dir / "gate_criteria.csv",
        ["criterion", "passed", "expected", "actual"],
        summary["gate_criteria"],
    )
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Post Active Top-k-1 Contextual Top-k-2 Branch Selector",
        "",
        f"- status: {summary['status']}",
        f"- decision: {summary['decision']}",
        f"- claim_status: {summary['claim_status']}",
        f"- selected_next_action: {summary['selected_next_action']}",
        f"- requires_gpu_now: {summary['requires_gpu_now']}",
        f"- promotion_allowed: {summary['promotion_allowed']}",
        "",
        "## Rationale",
        "",
        summary["rationale"],
        "",
        "## Next Step",
        "",
        summary["selected_next_step"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_post_active_topk1_contextual_topk2_branch_selector(out_dir=args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
