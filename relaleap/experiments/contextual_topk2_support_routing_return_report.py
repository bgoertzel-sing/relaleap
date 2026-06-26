"""Return selector for the contextual top-k-2 support-routing loop."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.causal_audit_coverage_report import (
    EXISTING_ARTIFACTS_SUFFICIENT,
    RANK_MATCHED_TOPK1_ACTIVE_POST_STOP,
)
from relaleap.experiments.promoted_topk2_finite_update_order_control_audit import (
    FINITE_UPDATE_ORDER_SENSITIVITY_CE_BOUNDED,
)


DEFAULT_PROMOTION_REPORT = Path(
    "results/reports/contextual_support_router_promotion_gate_satisfaction/decision_report.json"
)
DEFAULT_POST_PROMOTION_REPORT = Path(
    "results/reports/post_promotion_support_wide_promoted_default/decision_report.json"
)
DEFAULT_COVERAGE_REPORT = Path(
    "results/reports/token_larger_causal_audit_coverage/decision_report.json"
)
DEFAULT_FINITE_UPDATE_REPORT = Path(
    "results/reports/token_larger_promoted_topk2_finite_update_order_control_audit/summary.json"
)
DEFAULT_GATE_SUPPRESSION_AUDIT = Path(
    "results/audits/token_larger_active_topk1_context_gate_suppression_calibration/summary.json"
)
DEFAULT_POST_STOP_REPORT = Path(
    "results/reports/token_larger_post_stop_causal_bracket_decision/decision_report.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_contextual_topk2_support_routing_return"
)

SUPPORT_ROUTING_RETURN_SELECTED = "contextual_topk2_support_routing_return_selected"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
SELECTED_NEXT_ACTION = "contextual_router_shortcut_ablation"

_CLAIM_STATUSES = {
    "contextual_topk2_router": "promoted_operational_default_train_time_support_selection",
    "topk2_causal_cooperation": "blocked_pending_new_identified_controls",
    "topk1_singleton_reuse": "diagnostic_only_not_deployable",
    "hep_settling_improvement": "not_supported_post_promotion_alpha0_best",
}

_CANDIDATE_ACTIONS = (
    {
        "candidate_action": "matched_causal_control_intervention_matrix",
        "disposition": "already_satisfied_or_superseded",
        "reason": "coverage and post-stop reports already consume the matched top-k-2/top-k-1/random/dense control ladder and keep top-k-2 causal cooperation blocked",
    },
    {
        "candidate_action": SELECTED_NEXT_ACTION,
        "disposition": "selected",
        "reason": "tests whether contextual routing gains survive hidden-only, position-only, context-only, and full-context router feature ablations without duplicating completed causal-control reports",
    },
    {
        "candidate_action": "additional_hep_alpha_or_objective_sweep",
        "disposition": "disqualified",
        "reason": "post-promotion promoted-default evidence has alpha 0 best in all checked runs",
    },
    {
        "candidate_action": "deployable_topk1_singleton_reuse",
        "disposition": "disqualified",
        "reason": "context-gate suppression calibration failed pre-registered deployability criteria",
    },
)


def run_contextual_topk2_support_routing_return_report(
    *,
    promotion_report_path: Path = DEFAULT_PROMOTION_REPORT,
    post_promotion_report_path: Path = DEFAULT_POST_PROMOTION_REPORT,
    coverage_report_path: Path = DEFAULT_COVERAGE_REPORT,
    finite_update_report_path: Path = DEFAULT_FINITE_UPDATE_REPORT,
    gate_suppression_audit_path: Path = DEFAULT_GATE_SUPPRESSION_AUDIT,
    post_stop_report_path: Path = DEFAULT_POST_STOP_REPORT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Select the next non-duplicative architecture-loop diagnostic."""

    start = time.time()
    promotion = _read_json_object(promotion_report_path)
    post_promotion = _read_json_object(post_promotion_report_path)
    coverage = _read_json_object(coverage_report_path)
    finite_update = _read_json_object(finite_update_report_path)
    gate_suppression = _read_json_object(gate_suppression_audit_path)
    post_stop = _read_json_object(post_stop_report_path)
    strategy_review = _strategy_review(strategy_review_path)

    source_rows = [
        _source_row("contextual_router_promotion_gate", promotion_report_path, promotion),
        _source_row("post_promotion_promoted_default", post_promotion_report_path, post_promotion),
        _source_row("causal_audit_coverage", coverage_report_path, coverage),
        _source_row("finite_update_order_control", finite_update_report_path, finite_update),
        _source_row("topk1_gate_suppression_calibration", gate_suppression_audit_path, gate_suppression),
        _source_row("post_stop_causal_bracket", post_stop_report_path, post_stop),
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
    evidence = _evidence_snapshot(
        promotion=promotion,
        post_promotion=post_promotion,
        coverage=coverage,
        finite_update=finite_update,
        gate_suppression=gate_suppression,
        post_stop=post_stop,
    )
    failures = _failures(
        source_rows=source_rows,
        promotion=promotion,
        post_promotion=post_promotion,
        coverage=coverage,
        finite_update=finite_update,
        gate_suppression=gate_suppression,
        post_stop=post_stop,
        evidence=evidence,
    )

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        selected_next_action = None
        next_command = None
        next_step = "repair_missing_or_inconsistent_contextual_topk2_return_sources"
        rationale = (
            "The return selector cannot choose a bounded architecture-loop "
            "diagnostic because required source evidence is missing or inconsistent."
        )
    else:
        status = "pass"
        decision = SUPPORT_ROUTING_RETURN_SELECTED
        selected_next_action = SELECTED_NEXT_ACTION
        next_command = (
            "python -m relaleap.experiments.contextual_router_shortcut_ablation "
            "--config configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml "
            "--out results/audits/token_larger_contextual_router_shortcut_ablation"
        )
        next_step = (
            "implement and run the command-driven contextual-router shortcut "
            "ablation on the token-larger promoted-default support-wide setting"
        )
        rationale = (
            "The contextual top-k-2 router remains the operational default, but the "
            "matched causal-control ladder and post-stop bracket already block a "
            "top-k-2 causal-cooperation claim. The next non-duplicative architecture "
            "question is whether the contextual router's gain depends on full "
            "contextual features or mostly on shallow position/context shortcuts."
        )

    summary = {
        "status": status,
        "decision": decision,
        "selected_next_action": selected_next_action,
        "next_command": next_command,
        "next_step": next_step,
        "claim_statuses": dict(_CLAIM_STATUSES),
        "candidate_actions": list(_CANDIDATE_ACTIONS),
        "source_rows": source_rows,
        "evidence": evidence,
        "strategy_review": strategy_review,
        "failures": failures,
        "rationale": rationale,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "candidate_actions_csv": str(out_dir / "candidate_actions.csv"),
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
        out_dir / "candidate_actions.csv",
        ["candidate_action", "disposition", "reason"],
        list(_CANDIDATE_ACTIONS),
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _evidence_snapshot(
    *,
    promotion: dict[str, Any],
    post_promotion: dict[str, Any],
    coverage: dict[str, Any],
    finite_update: dict[str, Any],
    gate_suppression: dict[str, Any],
    post_stop: dict[str, Any],
) -> dict[str, Any]:
    post_evidence = post_promotion.get("evidence", {})
    coverage_evidence = coverage.get("coverage", {})
    finite_metrics = finite_update.get("metrics", {})
    gate_metrics = (gate_suppression.get("evidence", {}) or {}).get("metrics", {})
    return {
        "contextual_router_promoted": bool(
            promotion.get("decision") == "satisfy_contextual_support_router_promotion_or_repeat_gate"
            and promotion.get("status") == "pass"
        ),
        "promoted_default_confirmed": bool(
            post_promotion.get("promoted_support_router_default_confirmed")
            and post_promotion.get("status") == "pass"
        ),
        "post_promotion_alpha0_best_run_count": post_evidence.get("alpha0_best_run_count"),
        "post_promotion_run_count": post_evidence.get("run_count"),
        "post_promotion_accepted_nonzero_hep_run_count": post_evidence.get(
            "accepted_nonzero_hep_run_count"
        ),
        "causal_coverage_decision": coverage.get("decision"),
        "coverage_missing_fields": coverage_evidence.get(
            "missing_fields_for_deconfounded_no_training_audit"
        ),
        "coverage_missing_controls": coverage_evidence.get(
            "missing_controls_for_deconfounded_matrix"
        ),
        "post_stop_rank_matched_topk1_active": bool(
            post_stop.get("rank_matched_topk1_default_causal_audit_bracket")
        ),
        "post_stop_topk2_claim_supported": bool(
            post_stop.get("topk2_causal_cooperation_claim_supported")
        ),
        "topk2_ce_deficit_vs_rank_matched_topk1": coverage_evidence.get(
            "topk2_ce_deficit_vs_rank_matched_topk1"
        ),
        "finite_update_decision": finite_update.get("decision"),
        "topk2_mean_commutator_anchor_logit_mse": finite_metrics.get(
            "topk2_mean_commutator_anchor_logit_mse"
        ),
        "topk2_mean_commutator_anchor_residual_stream_l2": finite_metrics.get(
            "topk2_mean_commutator_anchor_residual_stream_l2"
        ),
        "topk2_mean_commutator_anchor_support_churn": finite_metrics.get(
            "topk2_mean_commutator_anchor_support_churn"
        ),
        "topk1_gate_suppression_decision": gate_suppression.get("decision"),
        "topk1_gate_retained_gain_fraction": gate_metrics.get(
            "deployable_retained_gain_fraction"
        ),
        "topk1_gate_offcontext_harm_suppression_fraction": gate_metrics.get(
            "deployable_offcontext_harm_suppression_fraction"
        ),
    }


def _failures(
    *,
    source_rows: list[dict[str, Any]],
    promotion: dict[str, Any],
    post_promotion: dict[str, Any],
    coverage: dict[str, Any],
    finite_update: dict[str, Any],
    gate_suppression: dict[str, Any],
    post_stop: dict[str, Any],
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows[:6]:
        if not row["present"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "source_artifact",
                    "expected": "file exists",
                    "actual": "missing",
                    "path": row["path"],
                }
            )
    expectations = (
        (
            "contextual_router_promotion_gate",
            promotion,
            "decision",
            "satisfy_contextual_support_router_promotion_or_repeat_gate",
        ),
        (
            "post_promotion_promoted_default",
            post_promotion,
            "decision",
            "confirm_post_promotion_support_wide_promoted_default",
        ),
        (
            "finite_update_order_control",
            finite_update,
            "decision",
            FINITE_UPDATE_ORDER_SENSITIVITY_CE_BOUNDED,
        ),
        (
            "topk1_gate_suppression_calibration",
            gate_suppression,
            "decision",
            "deployable_context_gate_suppression_calibration_failed",
        ),
        (
            "post_stop_causal_bracket",
            post_stop,
            "decision",
            "select_post_stop_rank_matched_topk1_causal_bracket",
        ),
    )
    for source, packet, field, expected in expectations:
        if packet.get(field) != expected:
            failures.append(
                {
                    "source": source,
                    "field": field,
                    "expected": expected,
                    "actual": packet.get(field),
                }
            )
    if coverage.get("decision") not in {
        EXISTING_ARTIFACTS_SUFFICIENT,
        RANK_MATCHED_TOPK1_ACTIVE_POST_STOP,
    }:
        failures.append(
            {
                "source": "causal_audit_coverage",
                "field": "decision",
                "expected": f"{EXISTING_ARTIFACTS_SUFFICIENT} or {RANK_MATCHED_TOPK1_ACTIVE_POST_STOP}",
                "actual": coverage.get("decision"),
            }
        )
    if not evidence["contextual_router_promoted"]:
        failures.append(
            {
                "source": "contextual_router_promotion_gate",
                "field": "contextual_router_promoted",
                "expected": True,
                "actual": evidence["contextual_router_promoted"],
            }
        )
    if not evidence["promoted_default_confirmed"]:
        failures.append(
            {
                "source": "post_promotion_promoted_default",
                "field": "promoted_default_confirmed",
                "expected": True,
                "actual": evidence["promoted_default_confirmed"],
            }
        )
    if evidence["post_promotion_alpha0_best_run_count"] != evidence["post_promotion_run_count"]:
        failures.append(
            {
                "source": "post_promotion_promoted_default",
                "field": "alpha0_best_run_count",
                "expected": evidence["post_promotion_run_count"],
                "actual": evidence["post_promotion_alpha0_best_run_count"],
            }
        )
    if evidence["coverage_missing_fields"] not in ([], None):
        failures.append(
            {
                "source": "causal_audit_coverage",
                "field": "missing_fields_for_deconfounded_no_training_audit",
                "expected": [],
                "actual": evidence["coverage_missing_fields"],
            }
        )
    if evidence["coverage_missing_controls"] not in ([], None):
        failures.append(
            {
                "source": "causal_audit_coverage",
                "field": "missing_controls_for_deconfounded_matrix",
                "expected": [],
                "actual": evidence["coverage_missing_controls"],
            }
        )
    if not evidence["post_stop_rank_matched_topk1_active"]:
        failures.append(
            {
                "source": "post_stop_causal_bracket",
                "field": "rank_matched_topk1_default_causal_audit_bracket",
                "expected": True,
                "actual": evidence["post_stop_rank_matched_topk1_active"],
            }
        )
    if evidence["post_stop_topk2_claim_supported"]:
        failures.append(
            {
                "source": "post_stop_causal_bracket",
                "field": "topk2_causal_cooperation_claim_supported",
                "expected": False,
                "actual": True,
            }
        )
    return failures


def _source_row(source: str, path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": packet.get("status"),
        "decision": packet.get("decision"),
        "claim_status": packet.get("claim_status") or packet.get("claim_policy"),
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "strategic_change_level": None,
            "notify_ben": None,
            "recommended_next_action": None,
            "incorporation": "optional review not present",
            "ben_notification_required": False,
        }
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:12]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() in {
            "strategic_change_level",
            "notify_ben",
            "recommended_next_action",
        }:
            header[key.strip()] = value.strip()
    notify_ben = _bool_or_none(header.get("notify_ben"))
    major = header.get("strategic_change_level") == "major"
    return {
        "path": str(path),
        "present": True,
        "strategic_change_level": header.get("strategic_change_level"),
        "notify_ben": notify_ben,
        "recommended_next_action": header.get("recommended_next_action"),
        "incorporation": (
            "accepted the recommendation to keep top-k-1 singleton reuse "
            "diagnostic-only and require causal-control context; the matched "
            "causal-control ladder is already present/post-stop, so the selected "
            "non-duplicative architecture-loop diagnostic is the contextual-router "
            "shortcut ablation"
        ),
        "ben_notification_required": bool(notify_ben) or major,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _bool_or_none(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
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


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    evidence = summary["evidence"]
    lines = [
        "# Contextual Top-k-2 Support-Routing Return Report",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Next command: `{summary['next_command']}`",
        "",
        "## Evidence",
        "",
        f"- Contextual router promoted: `{evidence['contextual_router_promoted']}`",
        f"- Promoted default confirmed: `{evidence['promoted_default_confirmed']}`",
        f"- Post-promotion alpha-0 best runs: `{evidence['post_promotion_alpha0_best_run_count']}` / `{evidence['post_promotion_run_count']}`",
        f"- Causal coverage decision: `{evidence['causal_coverage_decision']}`",
        f"- Post-stop top-k-1 active causal bracket: `{evidence['post_stop_rank_matched_topk1_active']}`",
        f"- Top-k-2 CE deficit vs rank-matched top-k-1: `{evidence['topk2_ce_deficit_vs_rank_matched_topk1']}`",
        f"- Finite-update decision: `{evidence['finite_update_decision']}`",
        f"- Top-k-1 gate-suppression decision: `{evidence['topk1_gate_suppression_decision']}`",
        "",
        "## Rationale",
        "",
        summary["rationale"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion-report", type=Path, default=DEFAULT_PROMOTION_REPORT)
    parser.add_argument(
        "--post-promotion-report", type=Path, default=DEFAULT_POST_PROMOTION_REPORT
    )
    parser.add_argument("--coverage-report", type=Path, default=DEFAULT_COVERAGE_REPORT)
    parser.add_argument(
        "--finite-update-report", type=Path, default=DEFAULT_FINITE_UPDATE_REPORT
    )
    parser.add_argument(
        "--gate-suppression-audit", type=Path, default=DEFAULT_GATE_SUPPRESSION_AUDIT
    )
    parser.add_argument("--post-stop-report", type=Path, default=DEFAULT_POST_STOP_REPORT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_contextual_topk2_support_routing_return_report(
        promotion_report_path=args.promotion_report,
        post_promotion_report_path=args.post_promotion_report,
        coverage_report_path=args.coverage_report,
        finite_update_report_path=args.finite_update_report,
        gate_suppression_audit_path=args.gate_suppression_audit,
        post_stop_report_path=args.post_stop_report,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
