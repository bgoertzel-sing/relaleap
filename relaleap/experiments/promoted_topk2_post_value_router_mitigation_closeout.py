"""Close out promoted contextual top-k-2 mitigation probes."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_promoted_topk2_post_value_router_mitigation_closeout"
)
DEFAULT_ROUTER_POLICY_PROBE = Path(
    "results/audits/token_larger_promoted_topk2_router_policy_mitigation_probe/summary.json"
)
DEFAULT_BRANCH_SELECTOR = Path(
    "results/reports/token_larger_promoted_topk2_mitigation_branch_selector/summary.json"
)
DEFAULT_ORDER_AVERAGING_PROBE = Path(
    "results/audits/token_larger_promoted_topk2_explicit_order_averaging_mitigation_probe/summary.json"
)
DEFAULT_UPDATE_DECOMPOSITION_AUDIT = Path(
    "results/audits/token_larger_promoted_topk2_update_decomposition_audit/summary.json"
)
DEFAULT_VALUE_MITIGATION_GATE = Path(
    "results/audits/token_larger_promoted_topk2_value_mitigation_gate/summary.json"
)
DEFAULT_LOW_RANK_VALUE_GATE = Path(
    "results/audits/token_larger_promoted_topk2_low_rank_value_gate/summary.json"
)
DEFAULT_COMMUTATOR_VALUE_PENALTY_PROBE = Path(
    "results/audits/token_larger_promoted_topk2_commutator_value_penalty_probe/summary.json"
)
DEFAULT_FINITE_UPDATE_REPORT = Path(
    "results/reports/token_larger_promoted_topk2_finite_update_order_control_audit/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")

MITIGATION_CLOSEOUT_NO_PROMOTION = "promoted_topk2_mitigation_closeout_no_promotion"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def run_promoted_topk2_post_value_router_mitigation_closeout(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    router_policy_probe_path: Path = DEFAULT_ROUTER_POLICY_PROBE,
    branch_selector_path: Path = DEFAULT_BRANCH_SELECTOR,
    order_averaging_probe_path: Path = DEFAULT_ORDER_AVERAGING_PROBE,
    update_decomposition_audit_path: Path = DEFAULT_UPDATE_DECOMPOSITION_AUDIT,
    value_mitigation_gate_path: Path = DEFAULT_VALUE_MITIGATION_GATE,
    low_rank_value_gate_path: Path = DEFAULT_LOW_RANK_VALUE_GATE,
    commutator_value_penalty_probe_path: Path = DEFAULT_COMMUTATOR_VALUE_PENALTY_PROBE,
    finite_update_report_path: Path = DEFAULT_FINITE_UPDATE_REPORT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
) -> dict[str, Any]:
    """Summarize completed top-k-2 mitigation branches and pick one next step."""

    start = time.time()
    packets = {
        "router_policy_probe": _read_json_object(router_policy_probe_path),
        "branch_selector": _read_json_object(branch_selector_path),
        "order_averaging_probe": _read_json_object(order_averaging_probe_path),
        "update_decomposition_audit": _read_json_object(update_decomposition_audit_path),
        "simple_value_mitigation_gate": _read_json_object(value_mitigation_gate_path),
        "low_rank_value_gate": _read_json_object(low_rank_value_gate_path),
        "commutator_value_penalty_probe": _read_json_object(
            commutator_value_penalty_probe_path
        ),
        "finite_update_order_control": _read_json_object(finite_update_report_path),
    }
    paths = {
        "router_policy_probe": router_policy_probe_path,
        "branch_selector": branch_selector_path,
        "order_averaging_probe": order_averaging_probe_path,
        "update_decomposition_audit": update_decomposition_audit_path,
        "simple_value_mitigation_gate": value_mitigation_gate_path,
        "low_rank_value_gate": low_rank_value_gate_path,
        "commutator_value_penalty_probe": commutator_value_penalty_probe_path,
        "finite_update_order_control": finite_update_report_path,
    }
    strategy_review = _strategy_review(strategy_review_path)
    source_rows = [_source_row(name, paths[name], packets[name]) for name in paths]
    source_rows.append(
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
        }
    )
    evidence = _evidence_snapshot(packets)
    closeout_rows = _closeout_rows(evidence)
    failures = _failures(source_rows, evidence)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        selected_next_action = None
        next_command = None
        next_step = "repair missing or inconsistent mitigation closeout artifacts"
        rationale = (
            "The closeout cannot be interpreted because required source artifacts "
            "or expected completed-branch decisions are missing."
        )
    else:
        status = "pass"
        decision = MITIGATION_CLOSEOUT_NO_PROMOTION
        selected_next_action = "pairwise_value_interaction_localization_audit"
        next_command = None
        next_step = (
            "implement a no-training pairwise value-interaction localization audit "
            "over the promoted top-k-2 checkpoint before proposing another "
            "mitigation family"
        )
        rationale = (
            "Router-policy pinning/freezing did not reduce the commutator, simple "
            "value clipping/scaling did not clear the gate, low-rank value updates "
            "did not clear the gate, and commutator-aware value penalties did not "
            "clear the gate. Explicit order averaging remains diagnostic-only. "
            "Because the update decomposition is value-dominated, the next "
            "non-duplicative step is to localize pairwise fixed-value interactions "
            "before designing another trainable mitigation."
        )

    summary = {
        "status": status,
        "decision": decision,
        "selected_next_action": selected_next_action,
        "next_command": next_command,
        "next_step": next_step,
        "claim_statuses": {
            "contextual_topk2_router": "promoted_operational_default_train_time_support_selection",
            "router_policy_mitigation": "not_established",
            "simple_value_mitigation": "not_established",
            "low_rank_value_mitigation": "not_established",
            "commutator_value_penalty": "not_established",
            "order_averaging": "diagnostic_only_not_promoted",
            "topk2_causal_cooperation": "not_supported",
            "topk2_finite_update_interference": "unresolved",
        },
        "source_rows": source_rows,
        "closeout_rows": closeout_rows,
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
            "closeout_rows_csv": str(out_dir / "closeout_rows.csv"),
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
        out_dir / "closeout_rows.csv",
        ["branch", "decision", "key_metric", "key_value", "disposition"],
        closeout_rows,
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _evidence_snapshot(packets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    finite_metrics = packets["finite_update_order_control"].get("metrics", {})
    decomposition_metrics = packets["update_decomposition_audit"].get("metrics", {})
    value_metrics = packets["simple_value_mitigation_gate"].get("metrics", {})
    low_rank_metrics = packets["low_rank_value_gate"].get("metrics", {})
    penalty_metrics = packets["commutator_value_penalty_probe"].get("metrics", {})
    router_rows = packets["router_policy_probe"].get("router_policy_rows", [])
    router_row = _best_router_policy_row(router_rows)
    return {
        "router_policy_decision": packets["router_policy_probe"].get("decision"),
        "router_policy_reduction_fraction": _float_or_none(
            router_row.get("commutator_anchor_logit_mse_reduction_fraction")
        ),
        "branch_selector_decision": packets["branch_selector"].get("decision"),
        "branch_selector_selected_next_action": packets["branch_selector"].get(
            "selected_next_action"
        ),
        "order_averaging_decision": packets["order_averaging_probe"].get("decision"),
        "finite_update_decision": packets["finite_update_order_control"].get("decision"),
        "topk2_order_averaged_to_commutator_anchor_logit_mse_ratio": _float_or_none(
            finite_metrics.get("topk2_order_averaged_to_commutator_anchor_logit_mse_ratio")
        ),
        "topk2_mean_order_averaged_anchor_ce_delta_vs_best_order": _float_or_none(
            finite_metrics.get("topk2_mean_order_averaged_anchor_ce_delta_vs_best_order")
        ),
        "topk2_mean_commutator_anchor_support_churn": _float_or_none(
            finite_metrics.get("topk2_mean_commutator_anchor_support_churn")
        ),
        "update_decomposition_decision": packets["update_decomposition_audit"].get(
            "decision"
        ),
        "value_only_fraction_of_full": _float_or_none(
            decomposition_metrics.get("value_only_fraction_of_full")
        ),
        "router_only_fraction_of_full": _float_or_none(
            decomposition_metrics.get("router_only_fraction_of_full")
        ),
        "simple_value_mitigation_decision": packets["simple_value_mitigation_gate"].get(
            "decision"
        ),
        "simple_value_best_reduction_fraction": _float_or_none(
            value_metrics.get("best_value_mitigation_reduction_fraction")
        ),
        "low_rank_value_decision": packets["low_rank_value_gate"].get("decision"),
        "low_rank_best_reduction_fraction": _float_or_none(
            low_rank_metrics.get("best_low_rank_reduction_fraction")
        ),
        "commutator_value_penalty_decision": packets[
            "commutator_value_penalty_probe"
        ].get("decision"),
        "commutator_value_penalty_best_reduction_fraction": _float_or_none(
            penalty_metrics.get("best_penalty_reduction_fraction")
        ),
    }


def _closeout_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "branch": "router_policy",
            "decision": evidence["router_policy_decision"],
            "key_metric": "commutator_anchor_logit_mse_reduction_fraction",
            "key_value": evidence["router_policy_reduction_fraction"],
            "disposition": "closed_not_established",
        },
        {
            "branch": "simple_value_update",
            "decision": evidence["simple_value_mitigation_decision"],
            "key_metric": "best_value_mitigation_reduction_fraction",
            "key_value": evidence["simple_value_best_reduction_fraction"],
            "disposition": "closed_not_established",
        },
        {
            "branch": "low_rank_value_update",
            "decision": evidence["low_rank_value_decision"],
            "key_metric": "best_low_rank_reduction_fraction",
            "key_value": evidence["low_rank_best_reduction_fraction"],
            "disposition": "closed_not_established",
        },
        {
            "branch": "commutator_value_penalty",
            "decision": evidence["commutator_value_penalty_decision"],
            "key_metric": "best_penalty_reduction_fraction",
            "key_value": evidence["commutator_value_penalty_best_reduction_fraction"],
            "disposition": "closed_not_established",
        },
        {
            "branch": "explicit_order_averaging",
            "decision": evidence["order_averaging_decision"],
            "key_metric": "order_averaged_to_commutator_anchor_logit_mse_ratio",
            "key_value": evidence[
                "topk2_order_averaged_to_commutator_anchor_logit_mse_ratio"
            ],
            "disposition": "diagnostic_only_not_promoted",
        },
        {
            "branch": "update_decomposition",
            "decision": evidence["update_decomposition_decision"],
            "key_metric": "value_only_fraction_of_full",
            "key_value": evidence["value_only_fraction_of_full"],
            "disposition": "mechanism_prior_for_next_audit",
        },
    ]


def _best_router_policy_row(rows: Any) -> dict[str, Any]:
    if not isinstance(rows, list):
        return {}
    candidates = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("variant") != "dynamic_contextual_topk2"
        and _float_or_none(
            row.get("commutator_anchor_logit_mse_reduction_fraction")
        )
        is not None
    ]
    if not candidates:
        candidates = [
            row
            for row in rows
            if isinstance(row, dict)
            and _float_or_none(
                row.get("commutator_anchor_logit_mse_reduction_fraction")
            )
            is not None
        ]
    if not candidates:
        return {}
    return max(
        candidates,
        key=lambda row: _float_or_none(
            row.get("commutator_anchor_logit_mse_reduction_fraction")
        )
        or float("-inf"),
    )


def _failures(
    source_rows: list[dict[str, Any]],
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows[:8]:
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
        "router_policy_decision": "value_composition_prioritized_over_router_policy",
        "branch_selector_decision": "promoted_topk2_mitigation_branch_selected",
        "branch_selector_selected_next_action": "explicit_order_averaging_mitigation_probe",
        "order_averaging_decision": "explicit_order_averaging_diagnostic_candidate_not_promoted",
        "finite_update_decision": "finite_update_order_sensitivity_ce_bounded_but_residual_material",
        "update_decomposition_decision": "value_update_dominated_order_sensitivity",
        "simple_value_mitigation_decision": "value_mitigation_not_established",
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
    for field in (
        "router_policy_reduction_fraction",
        "topk2_order_averaged_to_commutator_anchor_logit_mse_ratio",
        "value_only_fraction_of_full",
        "router_only_fraction_of_full",
        "simple_value_best_reduction_fraction",
        "low_rank_best_reduction_fraction",
        "commutator_value_penalty_best_reduction_fraction",
    ):
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
            "operational default, withhold causal-cooperation claims, and close "
            "out failed router/value mitigation branches before proposing another "
            "mitigation family"
        ),
        "ben_notification_required": bool(notify_ben) or major,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
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
        "# Promoted Top-k-2 Post-Value/Router Mitigation Closeout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Next command: `{summary['next_command']}`",
        "",
        "## Evidence",
        "",
        "- Router-policy reduction fraction: "
        f"`{evidence['router_policy_reduction_fraction']}`",
        "- Simple value best reduction fraction: "
        f"`{evidence['simple_value_best_reduction_fraction']}`",
        "- Low-rank value best reduction fraction: "
        f"`{evidence['low_rank_best_reduction_fraction']}`",
        "- Commutator value-penalty best reduction fraction: "
        f"`{evidence['commutator_value_penalty_best_reduction_fraction']}`",
        "- Order-averaging logit-MSE ratio: "
        f"`{evidence['topk2_order_averaged_to_commutator_anchor_logit_mse_ratio']}`",
        "- Value-only/full commutator fraction: "
        f"`{evidence['value_only_fraction_of_full']}`",
        "",
        "## Rationale",
        "",
        summary["rationale"],
        "",
        "## Next Step",
        "",
        summary["next_step"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--router-policy-probe", type=Path, default=DEFAULT_ROUTER_POLICY_PROBE)
    parser.add_argument("--branch-selector", type=Path, default=DEFAULT_BRANCH_SELECTOR)
    parser.add_argument("--order-averaging-probe", type=Path, default=DEFAULT_ORDER_AVERAGING_PROBE)
    parser.add_argument(
        "--update-decomposition-audit",
        type=Path,
        default=DEFAULT_UPDATE_DECOMPOSITION_AUDIT,
    )
    parser.add_argument("--value-mitigation-gate", type=Path, default=DEFAULT_VALUE_MITIGATION_GATE)
    parser.add_argument("--low-rank-value-gate", type=Path, default=DEFAULT_LOW_RANK_VALUE_GATE)
    parser.add_argument(
        "--commutator-value-penalty-probe",
        type=Path,
        default=DEFAULT_COMMUTATOR_VALUE_PENALTY_PROBE,
    )
    parser.add_argument("--finite-update-report", type=Path, default=DEFAULT_FINITE_UPDATE_REPORT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    args = parser.parse_args(argv)
    summary = run_promoted_topk2_post_value_router_mitigation_closeout(
        out_dir=args.out,
        router_policy_probe_path=args.router_policy_probe,
        branch_selector_path=args.branch_selector,
        order_averaging_probe_path=args.order_averaging_probe,
        update_decomposition_audit_path=args.update_decomposition_audit,
        value_mitigation_gate_path=args.value_mitigation_gate,
        low_rank_value_gate_path=args.low_rank_value_gate,
        commutator_value_penalty_probe_path=args.commutator_value_penalty_probe,
        finite_update_report_path=args.finite_update_report,
        strategy_review_path=args.strategy_review,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
