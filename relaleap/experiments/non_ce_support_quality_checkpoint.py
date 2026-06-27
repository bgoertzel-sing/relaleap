"""Select the next non-CE support-quality step after completed closeouts."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_CAUSAL_ROUTER_CLOSEOUT = Path(
    "results/reports/token_larger_causal_contextual_router_post_stratified_null_closeout/summary.json"
)
DEFAULT_POST_LOCALIZATION_CLOSEOUT = Path(
    "results/reports/token_larger_promoted_topk2_post_localization_closeout/summary.json"
)
DEFAULT_RETENTION_SYNTHESIS = Path(
    "results/reports/token_larger_active_topk1_causal_retention_synthesis/summary.json"
)
DEFAULT_SHORTCUT_DECISION = Path(
    "results/reports/token_larger_contextual_router_shortcut_decision/summary.json"
)
DEFAULT_COMMUTATOR_VALUE_PENALTY = Path(
    "results/audits/token_larger_promoted_topk2_commutator_value_penalty_probe/summary.json"
)
DEFAULT_SUPPORT_FREQUENCY_BLOCKER = Path(
    "results/reports/token_larger_support_frequency_blocker_diagnostic/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/token_larger_non_ce_support_quality_checkpoint")


CHECKPOINT_SELECTED = "non_ce_support_quality_checkpoint_selected"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
SELECTED_NEXT_ACTION = "router_value_disentanglement_audit_design"


def run_non_ce_support_quality_checkpoint(
    *,
    causal_router_closeout_path: Path = DEFAULT_CAUSAL_ROUTER_CLOSEOUT,
    post_localization_closeout_path: Path = DEFAULT_POST_LOCALIZATION_CLOSEOUT,
    retention_synthesis_path: Path = DEFAULT_RETENTION_SYNTHESIS,
    shortcut_decision_path: Path = DEFAULT_SHORTCUT_DECISION,
    commutator_value_penalty_path: Path = DEFAULT_COMMUTATOR_VALUE_PENALTY,
    support_frequency_blocker_path: Path = DEFAULT_SUPPORT_FREQUENCY_BLOCKER,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a bounded checkpoint over completed non-CE support-quality evidence."""

    start = time.time()
    sources = {
        "causal_router_post_stratified_null_closeout": (
            causal_router_closeout_path,
            _read_json_object(causal_router_closeout_path),
        ),
        "post_localization_closeout": (
            post_localization_closeout_path,
            _read_json_object(post_localization_closeout_path),
        ),
        "active_topk1_causal_retention_synthesis": (
            retention_synthesis_path,
            _read_json_object(retention_synthesis_path),
        ),
        "contextual_router_shortcut_decision": (
            shortcut_decision_path,
            _read_json_object(shortcut_decision_path),
        ),
        "commutator_value_penalty_probe": (
            commutator_value_penalty_path,
            _read_json_object(commutator_value_penalty_path),
        ),
        "support_frequency_blocker_diagnostic": (
            support_frequency_blocker_path,
            _read_json_object(support_frequency_blocker_path),
        ),
    }
    strategy_review = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row(name, path, packet) for name, (path, packet) in sources.items()
    ]
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
    evidence = _evidence({name: packet for name, (_, packet) in sources.items()})
    failures = _failures(source_rows, evidence)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        selected_next_action = None
        next_step = "repair missing or inconsistent non-CE checkpoint sources"
        rationale = (
            "The checkpoint cannot select a next action because at least one "
            "required command-generated source is missing, failing, or inconsistent."
        )
    else:
        status = "pass"
        decision = CHECKPOINT_SELECTED
        selected_next_action = SELECTED_NEXT_ACTION
        next_step = (
            "design one no-training router/value disentanglement audit that compares "
            "same learned values under alternate supports/routers and same supports "
            "under alternate value paths before any new mitigation, distillation, or "
            "GPU repeat"
        )
        rationale = (
            "The completed closeouts block causal-router promotion, teacher-support "
            "distillation, broad top-k-1 causal-retention claims, hub-pair mitigation, "
            "and order-averaging promotion. The contextual top-k-2 router remains an "
            "operational CE/support-diversity baseline, but commutator/value penalties "
            "and support-frequency denominators have not resolved support quality. "
            "The next non-duplicative local question is therefore router/value "
            "disentanglement rather than another CE sweep or mitigation branch."
        )

    candidate_actions = _candidate_actions(selected_next_action)
    summary = {
        "status": status,
        "decision": decision,
        "selected_next_action": selected_next_action,
        "next_step": next_step,
        "claim_statuses": {
            "causal_contextual_router": "ce_baseline_only_not_promoted",
            "teacher_support_distillation": "closed_not_promoted",
            "active_topk1": "retention_churn_control_not_deployable_causal_claim",
            "contextual_topk2_router": "operational_default_support_routing_baseline",
            "topk2_causal_cooperation": "not_supported",
            "hub_pair_mitigation": "deferred_rejected_diffuse_localization",
            "order_averaging": "diagnostic_only_not_promoted",
            "finite_update_interference": "unresolved",
        },
        "source_rows": source_rows,
        "candidate_actions": candidate_actions,
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
        candidate_actions,
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _evidence(packets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    causal_closeout = packets["causal_router_post_stratified_null_closeout"]
    localization = packets["post_localization_closeout"]
    retention = packets["active_topk1_causal_retention_synthesis"]
    shortcut = packets["contextual_router_shortcut_decision"]
    penalty = packets["commutator_value_penalty_probe"]
    frequency = packets["support_frequency_blocker_diagnostic"]
    return {
        "causal_router_closeout_decision": causal_closeout.get("decision"),
        "causal_router_claim_status": causal_closeout.get("claim_status"),
        "post_localization_decision": localization.get("decision"),
        "post_localization_pairwise_status": (
            localization.get("metrics", {}) or {}
        ).get("pairwise_localization_decision"),
        "retention_synthesis_decision": retention.get("decision"),
        "retention_causal_claim_supported": (
            retention.get("signals", {}) or {}
        ).get("causal_retention_claim_supported"),
        "shortcut_selected_next_action": shortcut.get("selected_next_action"),
        "shortcut_topk2_causal_status": (
            shortcut.get("claim_statuses", {}) or {}
        ).get("topk2_causal_cooperation"),
        "commutator_value_penalty_decision": penalty.get("decision"),
        "commutator_best_reduction_fraction": (penalty.get("metrics", {}) or {}).get(
            "best_penalty_reduction_fraction"
        ),
        "support_frequency_decision": frequency.get("decision"),
        "support_frequency_claim_bearing": (
            frequency.get("evidence", {}) or {}
        ).get("claim_bearing"),
    }


def _failures(
    source_rows: list[dict[str, Any]],
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows[:-1]:
        if not row["present"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "summary_json",
                    "expected": "file exists",
                    "actual": "missing",
                    "path": row["path"],
                }
            )
        elif row["status"] not in {"pass", "ok"}:
            failures.append(
                {
                    "source": row["source"],
                    "field": "status",
                    "expected": "pass or ok",
                    "actual": row["status"],
                }
            )
    expected = {
        "causal_router_closeout_decision": "causal_contextual_router_distillation_branch_closed_no_promotion",
        "post_localization_decision": "promoted_topk2_value_router_family_closed",
        "post_localization_pairwise_status": "pairwise_value_interaction_diffuse",
        "retention_synthesis_decision": "causal_retention_claim_blocked_by_deployable_gate",
        "commutator_value_penalty_decision": "commutator_value_penalty_not_established",
        "support_frequency_decision": "support_frequency_percentile_claim_remains_blocked_by_support_count_caliper",
    }
    for field, expected_value in expected.items():
        if evidence.get(field) != expected_value:
            failures.append(
                {
                    "source": "evidence",
                    "field": field,
                    "expected": expected_value,
                    "actual": evidence.get(field),
                }
            )
    if evidence.get("retention_causal_claim_supported") is not False:
        failures.append(
            {
                "source": "active_topk1_causal_retention_synthesis",
                "field": "retention_causal_claim_supported",
                "expected": False,
                "actual": evidence.get("retention_causal_claim_supported"),
            }
        )
    if evidence.get("support_frequency_claim_bearing") is not False:
        failures.append(
            {
                "source": "support_frequency_blocker_diagnostic",
                "field": "claim_bearing",
                "expected": False,
                "actual": evidence.get("support_frequency_claim_bearing"),
            }
        )
    return failures


def _candidate_actions(selected_next_action: str | None) -> list[dict[str, Any]]:
    return [
        {
            "candidate_action": "repeat_completed_retention_gate",
            "disposition": "disqualified",
            "reason": "same-student and retention/churn artifacts are already complete and synthesized",
        },
        {
            "candidate_action": "hub_pair_or_order_averaging_mitigation",
            "disposition": "deferred_rejected",
            "reason": "current localization is diffuse and the strategic review explicitly warns against this branch",
        },
        {
            "candidate_action": "new_gpu_repeat",
            "disposition": "disqualified",
            "reason": "the selected question is a local no-training design step and does not require RunPod validation yet",
        },
        {
            "candidate_action": SELECTED_NEXT_ACTION,
            "disposition": "selected" if selected_next_action == SELECTED_NEXT_ACTION else "blocked",
            "reason": "separates support selection from value redundancy using existing artifacts before new mitigation or distillation work",
        },
    ]


def _source_row(name: str, path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": name,
        "path": str(path),
        "present": path.is_file(),
        "status": packet.get("status"),
        "decision": packet.get("decision"),
        "claim_status": packet.get("claim_status"),
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "present": False,
            "path": str(path),
            "strategic_change_level": None,
            "notify_ben": None,
            "ben_notification_required": False,
            "recommended_next_action": None,
            "incorporation": "missing optional review; proceeded from command artifacts",
        }
    headers: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()
    notify_ben = headers.get("notify_ben", "").lower() == "true"
    strategic_change_level = headers.get("strategic_change_level")
    return {
        "present": True,
        "path": str(path),
        "strategic_change_level": strategic_change_level,
        "notify_ben": notify_ben,
        "ben_notification_required": notify_ben or strategic_change_level == "major",
        "recommended_next_action": headers.get("recommended_next_action"),
        "incorporation": (
            "accepted: recorded the no-training closeout and retention/churn gate as "
            "complete, deferred hub-pair/order-averaging mitigation, and selected "
            "router/value disentanglement as the next local support-quality design"
        ),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Non-CE Support-Quality Checkpoint",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        "",
        summary["rationale"],
        "",
        f"Next step: {summary['next_step']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_non_ce_support_quality_checkpoint(out_dir=args.out)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "selected_next_action": summary["selected_next_action"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
