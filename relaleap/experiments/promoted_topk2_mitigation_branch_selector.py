"""Select the next promoted contextual top-k-2 mitigation branch."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_SHORTCUT_DECISION = Path(
    "results/reports/token_larger_contextual_router_shortcut_decision/summary.json"
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
DEFAULT_COMMUTATOR_VALUE_PENALTY_AUDIT = Path(
    "results/audits/token_larger_promoted_topk2_commutator_value_penalty_probe/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_promoted_topk2_mitigation_branch_selector"
)

MITIGATION_BRANCH_SELECTED = "promoted_topk2_mitigation_branch_selected"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
ROUTER_POLICY_ACTION = "router_policy_mitigation_probe"
ORDER_AVERAGING_ACTION = "explicit_order_averaging_mitigation_probe"
ORDER_AVERAGING_COMMAND = (
    "python -m relaleap.experiments.promoted_topk2_explicit_order_averaging_mitigation_probe "
    "--out results/audits/token_larger_promoted_topk2_explicit_order_averaging_mitigation_probe"
)
ROUTER_POLICY_COMMAND = (
    "python -m relaleap.experiments.promoted_topk2_router_policy_mitigation_probe "
    "--config configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml "
    "--out results/audits/token_larger_promoted_topk2_router_policy_mitigation_probe"
)


def run_promoted_topk2_mitigation_branch_selector(
    *,
    shortcut_decision_path: Path = DEFAULT_SHORTCUT_DECISION,
    finite_update_report_path: Path = DEFAULT_FINITE_UPDATE_REPORT,
    value_mitigation_audit_path: Path = DEFAULT_VALUE_MITIGATION_AUDIT,
    low_rank_value_audit_path: Path = DEFAULT_LOW_RANK_VALUE_AUDIT,
    commutator_value_penalty_audit_path: Path = DEFAULT_COMMUTATOR_VALUE_PENALTY_AUDIT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
    order_averaging_ratio_gate: float = 0.5,
    order_averaging_ce_delta_gate: float = 0.0,
) -> dict[str, Any]:
    """Choose exactly one post-value-mitigation branch without rerunning training."""

    start = time.time()
    shortcut = _read_json_object(shortcut_decision_path)
    finite_update = _read_json_object(finite_update_report_path)
    value_mitigation = _read_json_object(value_mitigation_audit_path)
    low_rank_value = _read_json_object(low_rank_value_audit_path)
    commutator_value_penalty = _read_json_object(commutator_value_penalty_audit_path)
    strategy_review = _strategy_review(strategy_review_path)

    source_rows = [
        _source_row("contextual_router_shortcut_decision", shortcut_decision_path, shortcut),
        _source_row("finite_update_order_control", finite_update_report_path, finite_update),
        _source_row("simple_value_mitigation_gate", value_mitigation_audit_path, value_mitigation),
        _source_row("low_rank_value_gate", low_rank_value_audit_path, low_rank_value),
        _source_row(
            "commutator_value_penalty_probe",
            commutator_value_penalty_audit_path,
            commutator_value_penalty,
        ),
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
        finite_update=finite_update,
        value_mitigation=value_mitigation,
        low_rank_value=low_rank_value,
        commutator_value_penalty=commutator_value_penalty,
    )
    thresholds = {
        "order_averaging_ratio_gate": order_averaging_ratio_gate,
        "order_averaging_ce_delta_gate": order_averaging_ce_delta_gate,
    }
    failures = _failures(source_rows, evidence)
    candidate_actions = _candidate_actions(
        evidence,
        order_averaging_ratio_gate=order_averaging_ratio_gate,
        order_averaging_ce_delta_gate=order_averaging_ce_delta_gate,
    )

    selected = [row for row in candidate_actions if row["disposition"] == "selected"]
    if failures or len(selected) != 1:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        selected_next_action = None
        next_command = None
        rationale = (
            "The mitigation selector cannot choose exactly one branch because "
            "required source packets are missing/inconsistent or the candidate "
            "dispositions are ambiguous."
        )
        next_step = "repair mitigation-branch selector source artifacts"
    else:
        status = "pass"
        decision = MITIGATION_BRANCH_SELECTED
        selected_next_action = selected[0]["candidate_action"]
        next_command = (
            ORDER_AVERAGING_COMMAND
            if selected_next_action == ORDER_AVERAGING_ACTION
            else ROUTER_POLICY_COMMAND
        )
        next_step = f"implement and run `{selected_next_action}`"
        rationale = selected[0]["reason"]

    summary = {
        "status": status,
        "decision": decision,
        "selected_next_action": selected_next_action,
        "next_command": next_command,
        "next_step": next_step,
        "claim_statuses": {
            "contextual_topk2_router": "promoted_operational_default_train_time_support_selection",
            "topk2_causal_cooperation": "not_supported_pending_commutator_cleanliness",
            "topk2_finite_update_interference": "unresolved_after_value_side_mitigation_failures",
            "topk1_singleton_reuse": "diagnostic_only_not_deployable",
            "order_averaging": "candidate_mitigation_not_promoted",
        },
        "candidate_actions": candidate_actions,
        "source_rows": source_rows,
        "evidence": evidence,
        "thresholds": thresholds,
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
        candidate_actions,
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _evidence_snapshot(
    *,
    shortcut: dict[str, Any],
    finite_update: dict[str, Any],
    value_mitigation: dict[str, Any],
    low_rank_value: dict[str, Any],
    commutator_value_penalty: dict[str, Any],
) -> dict[str, Any]:
    finite_metrics = finite_update.get("metrics", {})
    value_metrics = value_mitigation.get("metrics", {})
    low_rank_metrics = low_rank_value.get("metrics", {})
    penalty_metrics = commutator_value_penalty.get("metrics", {})
    return {
        "shortcut_selected_next_action": shortcut.get("selected_next_action"),
        "shortcut_decision": shortcut.get("decision"),
        "finite_update_decision": finite_update.get("decision"),
        "topk2_mean_commutator_anchor_logit_mse": _float_or_none(
            finite_metrics.get("topk2_mean_commutator_anchor_logit_mse")
        ),
        "topk2_mean_commutator_anchor_support_churn": _float_or_none(
            finite_metrics.get("topk2_mean_commutator_anchor_support_churn")
        ),
        "topk2_order_averaged_to_commutator_anchor_logit_mse_ratio": _float_or_none(
            finite_metrics.get("topk2_order_averaged_to_commutator_anchor_logit_mse_ratio")
        ),
        "topk2_mean_order_averaged_anchor_ce_delta_vs_best_order": _float_or_none(
            finite_metrics.get("topk2_mean_order_averaged_anchor_ce_delta_vs_best_order")
        ),
        "topk2_mean_order_averaged_anchor_logit_mse_to_forward": _float_or_none(
            finite_metrics.get("topk2_mean_order_averaged_anchor_logit_mse_to_forward")
        ),
        "value_mitigation_decision": value_mitigation.get("decision"),
        "value_mitigation_best_reduction_fraction": _float_or_none(
            value_metrics.get("best_value_mitigation_reduction_fraction")
        ),
        "low_rank_value_decision": low_rank_value.get("decision"),
        "low_rank_best_reduction_fraction": _float_or_none(
            low_rank_metrics.get("best_low_rank_reduction_fraction")
        ),
        "commutator_value_penalty_decision": commutator_value_penalty.get("decision"),
        "commutator_value_penalty_best_reduction_fraction": _float_or_none(
            penalty_metrics.get("best_penalty_reduction_fraction")
        ),
        "commutator_value_penalty_best_transfer_retention_fraction": _float_or_none(
            penalty_metrics.get("best_penalty_transfer_retention_fraction")
        ),
    }


def _failures(
    source_rows: list[dict[str, Any]],
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows[:5]:
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
        "shortcut_selected_next_action": "commutator_aware_value_penalty_probe",
        "finite_update_decision": "finite_update_order_sensitivity_ce_bounded_but_residual_material",
        "value_mitigation_decision": "value_mitigation_not_established",
        "low_rank_value_decision": "low_rank_value_not_established",
        "commutator_value_penalty_decision": "commutator_value_penalty_not_established",
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
    required_numeric = (
        "topk2_order_averaged_to_commutator_anchor_logit_mse_ratio",
        "topk2_mean_order_averaged_anchor_ce_delta_vs_best_order",
        "topk2_mean_commutator_anchor_support_churn",
        "commutator_value_penalty_best_reduction_fraction",
    )
    for field in required_numeric:
        if evidence.get(field) is None:
            failures.append(
                {
                    "source": "evidence_snapshot",
                    "field": field,
                    "expected": "numeric value",
                    "actual": None,
                }
            )
    return failures


def _candidate_actions(
    evidence: dict[str, Any],
    *,
    order_averaging_ratio_gate: float,
    order_averaging_ce_delta_gate: float,
) -> list[dict[str, str]]:
    ratio = evidence.get("topk2_order_averaged_to_commutator_anchor_logit_mse_ratio")
    ce_delta = evidence.get("topk2_mean_order_averaged_anchor_ce_delta_vs_best_order")
    support_churn = evidence.get("topk2_mean_commutator_anchor_support_churn")
    order_promising = (
        ratio is not None
        and ratio <= order_averaging_ratio_gate
        and ce_delta is not None
        and ce_delta <= order_averaging_ce_delta_gate
    )
    value_side_failed = evidence.get("commutator_value_penalty_decision") == (
        "commutator_value_penalty_not_established"
    )
    if value_side_failed and order_promising:
        order_disposition = "selected"
        router_disposition = "deferred"
        order_reason = (
            "value-side gates failed, while existing finite-update evidence shows "
            f"order averaging at {ratio} of the forward/reverse logit-MSE "
            "commutator with non-worse anchor CE versus the best endpoint; select "
            "the explicit order-averaging branch as a mitigation candidate, not a "
            "causal-cooperation claim"
        )
        router_reason = (
            "router-policy changes remain scientifically plausible because support "
            f"churn is {support_churn}, but they are deferred behind the already "
            "observed order-averaging commutator reduction signal"
        )
    elif value_side_failed:
        order_disposition = "disqualified"
        router_disposition = "selected"
        order_reason = (
            "the finite-update packet does not show enough order-averaging "
            "commutator reduction under the current fail-closed selector gate"
        )
        router_reason = (
            "value-side mitigations failed and order averaging lacks sufficient "
            "source-artifact support, so the next branch should change router "
            "policy while preserving the promoted top-k-2 controls"
        )
    else:
        order_disposition = "pending"
        router_disposition = "pending"
        order_reason = "waiting for value-side mitigation closeout"
        router_reason = "waiting for value-side mitigation closeout"
    return [
        {
            "candidate_action": ROUTER_POLICY_ACTION,
            "disposition": router_disposition,
            "reason": router_reason,
        },
        {
            "candidate_action": ORDER_AVERAGING_ACTION,
            "disposition": order_disposition,
            "reason": order_reason,
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
            "after simple value, low-rank value, and commutator-penalty branches "
            "failed, this selector chooses the next bounded mitigation branch"
        ),
        "ben_notification_required": bool(notify_ben) or major,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


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
        "# Promoted Top-k-2 Mitigation Branch Selector",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Next command: `{summary['next_command']}`",
        "",
        "## Evidence",
        "",
        "- Value-side decisions: "
        f"`{evidence['value_mitigation_decision']}`, "
        f"`{evidence['low_rank_value_decision']}`, "
        f"`{evidence['commutator_value_penalty_decision']}`",
        "- Order-averaging logit-MSE ratio: "
        f"`{evidence['topk2_order_averaged_to_commutator_anchor_logit_mse_ratio']}`",
        "- Order-averaging CE delta versus best endpoint: "
        f"`{evidence['topk2_mean_order_averaged_anchor_ce_delta_vs_best_order']}`",
        "",
        "## Rationale",
        "",
        summary["rationale"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortcut-decision", type=Path, default=DEFAULT_SHORTCUT_DECISION)
    parser.add_argument(
        "--finite-update-report", type=Path, default=DEFAULT_FINITE_UPDATE_REPORT
    )
    parser.add_argument(
        "--value-mitigation-audit", type=Path, default=DEFAULT_VALUE_MITIGATION_AUDIT
    )
    parser.add_argument(
        "--low-rank-value-audit", type=Path, default=DEFAULT_LOW_RANK_VALUE_AUDIT
    )
    parser.add_argument(
        "--commutator-value-penalty-audit",
        type=Path,
        default=DEFAULT_COMMUTATOR_VALUE_PENALTY_AUDIT,
    )
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_promoted_topk2_mitigation_branch_selector(
        shortcut_decision_path=args.shortcut_decision,
        finite_update_report_path=args.finite_update_report,
        value_mitigation_audit_path=args.value_mitigation_audit,
        low_rank_value_audit_path=args.low_rank_value_audit,
        commutator_value_penalty_audit_path=args.commutator_value_penalty_audit,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
