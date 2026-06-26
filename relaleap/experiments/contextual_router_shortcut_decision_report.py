"""Decision report after the contextual-router shortcut ablation."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_SHORTCUT_AUDIT = Path(
    "results/audits/token_larger_contextual_router_shortcut_ablation/summary.json"
)
DEFAULT_SUPPORT_SELECTION_REPORT = Path(
    "results/reports/token_larger_promoted_topk2_support_selection_quality_audit/summary.json"
)
DEFAULT_FUNCTIONAL_CHURN_REPORT = Path(
    "results/reports/token_larger_promoted_topk2_functional_churn_control_audit/summary.json"
)
DEFAULT_FINITE_UPDATE_REPORT = Path(
    "results/reports/token_larger_promoted_topk2_finite_update_order_control_audit/summary.json"
)
DEFAULT_VALUE_MITIGATION_AUDIT = Path(
    "results/audits/token_larger_promoted_topk2_value_mitigation_gate/summary.json"
)
DEFAULT_LOW_RANK_VALUE_AUDIT = Path(
    "results/audits/token_larger_promoted_topk2_low_rank_value_gate/summary.json"
)
DEFAULT_TOPK1_GATE_AUDIT = Path(
    "results/audits/token_larger_active_topk1_context_gate_suppression_calibration/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_contextual_router_shortcut_decision"
)

SHORTCUT_DECISION_SELECTED = "contextual_router_shortcut_decision_selected"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
SELECTED_NEXT_ACTION = "commutator_aware_value_penalty_probe"
NEXT_COMMAND = (
    "python -m relaleap.experiments.promoted_topk2_commutator_value_penalty_probe "
    "--config configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml "
    "--out results/audits/token_larger_promoted_topk2_commutator_value_penalty_probe"
)

_CLAIM_STATUSES = {
    "contextual_topk2_router": "promoted_operational_default_train_time_support_selection",
    "contextual_shortcut_risk": "bounded_in_fixed_batch_but_not_a_generalization_claim",
    "topk2_causal_cooperation": "not_supported_pending_commutator_cleanliness",
    "topk2_finite_update_interference": "ce_bounded_but_residual_logit_order_sensitivity_material",
    "topk1_singleton_reuse": "diagnostic_only_not_deployable",
    "hep_settling_improvement": "not_supported_post_promotion_alpha0_best",
}


def run_contextual_router_shortcut_decision_report(
    *,
    shortcut_audit_path: Path = DEFAULT_SHORTCUT_AUDIT,
    support_selection_report_path: Path = DEFAULT_SUPPORT_SELECTION_REPORT,
    functional_churn_report_path: Path = DEFAULT_FUNCTIONAL_CHURN_REPORT,
    finite_update_report_path: Path = DEFAULT_FINITE_UPDATE_REPORT,
    value_mitigation_audit_path: Path = DEFAULT_VALUE_MITIGATION_AUDIT,
    low_rank_value_audit_path: Path = DEFAULT_LOW_RANK_VALUE_AUDIT,
    topk1_gate_audit_path: Path = DEFAULT_TOPK1_GATE_AUDIT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Consume shortcut-ablation evidence and select one non-duplicative next step."""

    start = time.time()
    shortcut = _read_json_object(shortcut_audit_path)
    support_selection = _read_json_object(support_selection_report_path)
    functional_churn = _read_json_object(functional_churn_report_path)
    finite_update = _read_json_object(finite_update_report_path)
    value_mitigation = _read_json_object(value_mitigation_audit_path)
    low_rank_value = _read_json_object(low_rank_value_audit_path)
    topk1_gate = _read_json_object(topk1_gate_audit_path)
    strategy_review = _strategy_review(strategy_review_path)

    source_rows = [
        _source_row("contextual_router_shortcut_ablation", shortcut_audit_path, shortcut),
        _source_row(
            "promoted_topk2_support_selection_quality",
            support_selection_report_path,
            support_selection,
        ),
        _source_row("functional_churn_control", functional_churn_report_path, functional_churn),
        _source_row("finite_update_order_control", finite_update_report_path, finite_update),
        _source_row("simple_value_mitigation_gate", value_mitigation_audit_path, value_mitigation),
        _source_row("low_rank_value_gate", low_rank_value_audit_path, low_rank_value),
        _source_row("topk1_context_gate_suppression", topk1_gate_audit_path, topk1_gate),
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
        shortcut=shortcut,
        support_selection=support_selection,
        functional_churn=functional_churn,
        finite_update=finite_update,
        value_mitigation=value_mitigation,
        low_rank_value=low_rank_value,
        topk1_gate=topk1_gate,
    )
    failures = _failures(source_rows, evidence)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        selected_next_action = None
        next_command = None
        missing_artifacts = [
            failure for failure in failures if failure.get("field") == "source_artifact"
        ]
        rationale = (
            "The shortcut decision selector cannot choose the next promoted top-k-2 "
            "diagnostic because one or more required source packets are missing or "
            "inconsistent."
        )
    else:
        status = "pass"
        decision = SHORTCUT_DECISION_SELECTED
        selected_next_action = SELECTED_NEXT_ACTION
        next_command = NEXT_COMMAND
        missing_artifacts = []
        rationale = (
            "The shortcut ablation does not support a pure position shortcut: "
            "full-context features are best, while position-only is strongly harmful. "
            "Because the remaining fixed-batch router-oracle gap is tiny, this is a "
            "conservative shortcut-risk diagnostic rather than a causal-cooperation "
            "claim. Existing support-selection, load-balance, simple value-mitigation, "
            "and low-rank value reports already close their branches. The remaining "
            "non-duplicative top-k-2 risk is finite-update residual/logit order "
            "sensitivity, so the next bounded local diagnostic should test a "
            "commutator-aware value-update penalty."
        )

    candidate_actions = _candidate_actions(selected_next_action)
    summary = {
        "status": status,
        "decision": decision,
        "selected_next_action": selected_next_action,
        "next_command": next_command,
        "next_step": (
            "implement and run the command-driven commutator-aware value-update "
            "penalty probe"
            if status == "pass"
            else "repair missing or inconsistent shortcut-decision source artifacts"
        ),
        "claim_statuses": dict(_CLAIM_STATUSES),
        "candidate_actions": candidate_actions,
        "source_rows": source_rows,
        "evidence": evidence,
        "strategy_review": strategy_review,
        "missing_artifacts": missing_artifacts,
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
        candidate_actions,
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _evidence_snapshot(
    *,
    shortcut: dict[str, Any],
    support_selection: dict[str, Any],
    functional_churn: dict[str, Any],
    finite_update: dict[str, Any],
    value_mitigation: dict[str, Any],
    low_rank_value: dict[str, Any],
    topk1_gate: dict[str, Any],
) -> dict[str, Any]:
    ablation = shortcut.get("ablation", {}) if isinstance(shortcut.get("ablation"), dict) else {}
    variants = ablation.get("variants", {}) if isinstance(ablation.get("variants"), dict) else {}
    finite_metrics = finite_update.get("metrics", {})
    topk1_metrics = (topk1_gate.get("evidence", {}) or {}).get("metrics", {})
    return {
        "shortcut_decision": shortcut.get("decision"),
        "shortcut_selected_variant": ablation.get("selected_variant"),
        "shortcut_interpretation": ablation.get("shortcut_interpretation"),
        "shortcut_router_oracle_gap": ablation.get("router_oracle_gap"),
        "full_context_holdout_recovery": _holdout_recovery(variants, "full_context"),
        "position_only_holdout_recovery": _holdout_recovery(variants, "position_only"),
        "context_only_holdout_recovery": _holdout_recovery(variants, "context_only"),
        "support_selection_decision": support_selection.get("decision"),
        "functional_churn_decision": functional_churn.get("decision"),
        "finite_update_decision": finite_update.get("decision"),
        "topk2_mean_commutator_anchor_logit_mse": finite_metrics.get(
            "topk2_mean_commutator_anchor_logit_mse"
        ),
        "topk2_mean_commutator_anchor_residual_stream_l2": finite_metrics.get(
            "topk2_mean_commutator_anchor_residual_stream_l2"
        ),
        "simple_value_mitigation_decision": value_mitigation.get("decision"),
        "low_rank_value_decision": low_rank_value.get("decision"),
        "topk1_gate_suppression_decision": topk1_gate.get("decision"),
        "topk1_gate_retained_gain_fraction": topk1_metrics.get(
            "deployable_retained_gain_fraction"
        ),
        "topk1_gate_offcontext_harm_suppression_fraction": topk1_metrics.get(
            "deployable_offcontext_harm_suppression_fraction"
        ),
    }


def _failures(
    source_rows: list[dict[str, Any]],
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows[:7]:
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
    expected = {
        "shortcut_decision": "contextual_router_shortcut_ablation_completed",
        "shortcut_selected_variant": "full_context",
        "shortcut_interpretation": "full_context_features_best_supported",
        "support_selection_decision": "promoted_topk2_support_selection_quality_established",
        "functional_churn_decision": (
            "support_identity_churn_functional_impact_bounded_with_commutator_risk"
        ),
        "finite_update_decision": (
            "finite_update_order_sensitivity_ce_bounded_but_residual_material"
        ),
        "simple_value_mitigation_decision": "value_mitigation_not_established",
        "low_rank_value_decision": "low_rank_value_not_established",
        "topk1_gate_suppression_decision": (
            "deployable_context_gate_suppression_calibration_failed"
        ),
    }
    for field, expected_value in expected.items():
        if evidence.get(field) != expected_value:
            failures.append(
                {
                    "source": "evidence_snapshot",
                    "field": field,
                    "expected": expected_value,
                    "actual": evidence.get(field),
                }
            )
    if not _positive(evidence.get("shortcut_router_oracle_gap")):
        failures.append(
            {
                "source": "contextual_router_shortcut_ablation",
                "field": "shortcut_router_oracle_gap",
                "expected": "positive finite gap",
                "actual": evidence.get("shortcut_router_oracle_gap"),
            }
        )
    return failures


def _candidate_actions(selected_next_action: str | None) -> list[dict[str, str]]:
    return [
        {
            "candidate_action": "rerun_contextual_router_shortcut_ablation",
            "disposition": "disqualified",
            "reason": "completed artifact already exists and gives a conservative fixed-batch shortcut interpretation",
        },
        {
            "candidate_action": "promote_contextual_shortcut_claim",
            "disposition": "disqualified",
            "reason": "the router-oracle gap is tiny and fixed-batch feature probes are not causal cooperation or broad generalization evidence",
        },
        {
            "candidate_action": "matched_causal_control_intervention_matrix",
            "disposition": "already_satisfied_or_superseded",
            "reason": "existing coverage and post-stop bracket reports keep top-k-2 causal cooperation blocked and top-k-1 diagnostic-only",
        },
        {
            "candidate_action": "simple_value_scaling_or_low_rank_repeat",
            "disposition": "disqualified",
            "reason": "simple value mitigation and low-rank value gates have both failed under their current criteria",
        },
        {
            "candidate_action": SELECTED_NEXT_ACTION,
            "disposition": "selected" if selected_next_action == SELECTED_NEXT_ACTION else "pending",
            "reason": "targets the remaining material finite-update residual/logit order-sensitivity risk without changing router policy",
        },
    ]


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
            "accepted the recommendation to keep contextual top-k-2 as the "
            "operational default while withholding causal-cooperation claims; "
            "after the shortcut ablation and completed causal/control closeouts, "
            "the selected non-duplicative step is a commutator-aware value penalty probe"
        ),
        "ben_notification_required": bool(notify_ben) or major,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _holdout_recovery(variants: dict[str, Any], variant: str) -> float | None:
    packet = variants.get(variant)
    if not isinstance(packet, dict):
        return None
    holdout = packet.get("holdout")
    if not isinstance(holdout, dict):
        return None
    return _float_or_none(holdout.get("intervention_oracle_gap_recovery_fraction"))


def _positive(value: Any) -> bool:
    number = _float_or_none(value)
    return number is not None and number > 0.0


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        "# Contextual Router Shortcut Decision",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Next command: `{summary['next_command']}`",
        "",
        "## Shortcut Evidence",
        "",
        f"- Selected variant: `{evidence['shortcut_selected_variant']}`",
        f"- Interpretation: `{evidence['shortcut_interpretation']}`",
        f"- Router-oracle gap: `{evidence['shortcut_router_oracle_gap']}`",
        f"- Full-context holdout recovery: `{evidence['full_context_holdout_recovery']}`",
        f"- Position-only holdout recovery: `{evidence['position_only_holdout_recovery']}`",
        "",
        "## Rationale",
        "",
        summary["rationale"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortcut-audit", type=Path, default=DEFAULT_SHORTCUT_AUDIT)
    parser.add_argument(
        "--support-selection-report", type=Path, default=DEFAULT_SUPPORT_SELECTION_REPORT
    )
    parser.add_argument(
        "--functional-churn-report", type=Path, default=DEFAULT_FUNCTIONAL_CHURN_REPORT
    )
    parser.add_argument(
        "--finite-update-report", type=Path, default=DEFAULT_FINITE_UPDATE_REPORT
    )
    parser.add_argument(
        "--value-mitigation-audit", type=Path, default=DEFAULT_VALUE_MITIGATION_AUDIT
    )
    parser.add_argument(
        "--low-rank-value-audit", type=Path, default=DEFAULT_LOW_RANK_VALUE_AUDIT
    )
    parser.add_argument("--topk1-gate-audit", type=Path, default=DEFAULT_TOPK1_GATE_AUDIT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_contextual_router_shortcut_decision_report(
        shortcut_audit_path=args.shortcut_audit,
        support_selection_report_path=args.support_selection_report,
        functional_churn_report_path=args.functional_churn_report,
        finite_update_report_path=args.finite_update_report,
        value_mitigation_audit_path=args.value_mitigation_audit,
        low_rank_value_audit_path=args.low_rank_value_audit,
        topk1_gate_audit_path=args.topk1_gate_audit,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
