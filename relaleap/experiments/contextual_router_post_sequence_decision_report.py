"""Post-sequence decision report for the causal-feature-safe contextual router."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_SEQUENCE_CLOSEOUT = Path(
    "results/reports/token_larger_contextual_router_sequence_kfold_backend_closeout/summary.json"
)
DEFAULT_LOCAL_SUPPORT_AUDIT = Path(
    "results/audits/local_causal_contextual_router_support_audit/summary.json"
)
DEFAULT_RUNPOD_SUPPORT_AUDIT = Path(
    "results/runpod_fetch/audits/runpod_token_larger_causal_contextual_router_support_audit/summary.json"
)
DEFAULT_DECONFOUNDED_INTERVENTION_AUDIT = Path(
    "results/audits/token_larger_topk2_vs_rank_matched_topk1_deconfounded_intervention/summary.json"
)
DEFAULT_CAUSAL_COVERAGE_REPORT = Path(
    "results/reports/token_larger_causal_audit_coverage/decision_report.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_contextual_router_post_sequence_decision"
)

POST_SEQUENCE_DECISION_RECORDED = "contextual_router_post_sequence_decision_recorded"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
SUPPORT_AUDIT_BLOCKED = "causal_contextual_router_support_audit_blocks_promotion"
SEQUENCE_VALIDATED = "sequence_kfold_backend_validated"
DECONFOUNDED_TOPK2_BLOCKED = "topk2_comparative_causal_cooperation_not_supported"
CAUSAL_COVERAGE_TOPK1_ACTIVE = "rank_matched_topk1_active_post_stop_bracket"


def run_contextual_router_post_sequence_decision_report(
    *,
    sequence_closeout_path: Path = DEFAULT_SEQUENCE_CLOSEOUT,
    local_support_audit_path: Path = DEFAULT_LOCAL_SUPPORT_AUDIT,
    runpod_support_audit_path: Path = DEFAULT_RUNPOD_SUPPORT_AUDIT,
    deconfounded_intervention_audit_path: Path = DEFAULT_DECONFOUNDED_INTERVENTION_AUDIT,
    causal_coverage_report_path: Path = DEFAULT_CAUSAL_COVERAGE_REPORT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Consume post-sequence evidence and select one conservative next step."""

    start = time.time()
    sequence = _read_json_object(sequence_closeout_path)
    local_support = _read_json_object(local_support_audit_path)
    runpod_support = _read_json_object(runpod_support_audit_path)
    deconfounded = _read_json_object(deconfounded_intervention_audit_path)
    coverage = _read_json_object(causal_coverage_report_path)
    strategy_review = _strategy_review(strategy_review_path)

    source_rows = [
        _source_row("sequence_kfold_backend_closeout", sequence_closeout_path, sequence),
        _source_row("local_causal_support_audit", local_support_audit_path, local_support),
        _source_row("runpod_causal_support_audit", runpod_support_audit_path, runpod_support),
        _source_row(
            "deconfounded_topk2_vs_rank_matched_topk1",
            deconfounded_intervention_audit_path,
            deconfounded,
        ),
        _source_row("causal_audit_coverage", causal_coverage_report_path, coverage),
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
    evidence = _evidence(
        sequence=sequence,
        local_support=local_support,
        runpod_support=runpod_support,
        deconfounded=deconfounded,
        coverage=coverage,
    )
    candidate_actions = _candidate_actions(evidence)
    failures = _failures(source_rows, evidence)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        claim_status = "post_sequence_decision_uninterpretable"
        selected_next_step = "repair_missing_or_inconsistent_post_sequence_sources"
        next_command = None
        rationale = (
            "The post-sequence decision report cannot be interpreted because a "
            "required sequence, support-quality, or deconfounded causal source is "
            "missing, failed, or inconsistent."
        )
    else:
        status = "pass"
        decision = POST_SEQUENCE_DECISION_RECORDED
        claim_status = "causal_feature_safe_router_not_promoted_support_quality_blocked"
        selected_next_step = (
            "run a bounded oracle-regret and functional-churn failure inspection "
            "for the causal-feature-safe contextual router"
        )
        next_command = None
        rationale = (
            "The causal-feature-safe contextual top-k-2 router has backend-stable "
            "sequence-heldout CE evidence versus the linear top-k-2 control, but "
            "both local and RunPod support audits block promotion: oracle-support "
            "regret and functional churn are worse than the linear control. The "
            "existing deconfounded top-k-2 audit also keeps the broader causal "
            "cooperation claim blocked and leaves rank-matched top-k-1 as the "
            "active causal bracket. The branch should therefore stop promotion "
            "work and inspect the support-regret/churn failure before any new "
            "distillation, default change, or GPU repeat."
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "next_command": next_command,
        "claim_statuses": {
            "causal_feature_safe_contextual_topk2": claim_status,
            "full_context_contextual_topk2": "nondeployable_oracle_diagnostic_only",
            "promoted_default_contextual_topk2": "operational_default_not_causal_mechanism_claim",
            "topk2_causal_cooperation": "blocked_by_deconfounded_rank_matched_topk1_controls",
            "rank_matched_topk1": "active_local_causal_audit_bracket",
            "causal_router_support_label_distillation": "frozen_pending_functional_gate",
        },
        "source_rows": source_rows,
        "evidence": evidence,
        "candidate_actions": candidate_actions,
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


def _evidence(
    *,
    sequence: dict[str, Any],
    local_support: dict[str, Any],
    runpod_support: dict[str, Any],
    deconfounded: dict[str, Any],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    local_causal = _aggregate(local_support, "causal_contextual_topk2")
    local_linear = _aggregate(local_support, "linear_topk2")
    runpod_causal = _aggregate(runpod_support, "causal_contextual_topk2")
    runpod_linear = _aggregate(runpod_support, "linear_topk2")
    deconfounded_evidence = (
        deconfounded.get("evidence", {})
        if isinstance(deconfounded.get("evidence"), dict)
        else {}
    )
    coverage_packet = (
        coverage.get("coverage", {}) if isinstance(coverage.get("coverage"), dict) else {}
    )
    return {
        "sequence_decision": sequence.get("decision"),
        "sequence_claim_status": sequence.get("claim_status"),
        "sequence_causal_beats_linear_both_backends": _nested(
            sequence, "evidence", "causal_contextual_beats_linear_both_backends"
        ),
        "sequence_full_context_beats_causal_both_backends": _nested(
            sequence, "evidence", "full_context_beats_causal_contextual_both_backends"
        ),
        "local_support_decision": local_support.get("decision"),
        "runpod_support_decision": runpod_support.get("decision"),
        "local_causal_ce_delta_vs_linear": local_causal.get(
            "mean_router_loss_delta_vs_linear"
        ),
        "runpod_causal_ce_delta_vs_linear": runpod_causal.get(
            "mean_router_loss_delta_vs_linear"
        ),
        "local_causal_oracle_regret": local_causal.get("mean_oracle_support_regret"),
        "local_linear_oracle_regret": local_linear.get("mean_oracle_support_regret"),
        "runpod_causal_oracle_regret": runpod_causal.get(
            "mean_oracle_support_regret"
        ),
        "runpod_linear_oracle_regret": runpod_linear.get("mean_oracle_support_regret"),
        "local_causal_functional_churn": local_causal.get(
            "mean_functional_churn_logit_l1"
        ),
        "local_linear_functional_churn": local_linear.get(
            "mean_functional_churn_logit_l1"
        ),
        "runpod_causal_functional_churn": runpod_causal.get(
            "mean_functional_churn_logit_l1"
        ),
        "runpod_linear_functional_churn": runpod_linear.get(
            "mean_functional_churn_logit_l1"
        ),
        "local_support_failures": _nested(local_support, "audit", "failures") or [],
        "runpod_support_failures": _nested(runpod_support, "audit", "failures") or [],
        "deconfounded_decision": deconfounded.get("decision"),
        "deconfounded_topk2_pair_synergy_positive_strata_fraction": (
            deconfounded_evidence.get(
                "deconfounded_topk2_pair_synergy_positive_strata_fraction"
            )
        ),
        "deconfounded_topk2_incremental_pair_gain_positive_strata_fraction": (
            deconfounded_evidence.get(
                "topk2_incremental_pair_gain_positive_strata_fraction"
            )
        ),
        "deconfounded_topk2_fixed_support_cleaner_strata_fraction": (
            deconfounded_evidence.get("topk2_fixed_support_cleaner_strata_fraction")
        ),
        "deconfounded_topk2_functional_churn_cleaner_strata_fraction": (
            deconfounded_evidence.get(
                "topk2_functional_churn_cleaner_strata_fraction"
            )
        ),
        "causal_coverage_decision": coverage.get("decision"),
        "causal_coverage_rank_matched_topk1_active": coverage_packet.get(
            "post_stop_rank_matched_topk1_active"
        ),
        "causal_coverage_topk2_claim_supported": coverage_packet.get(
            "post_stop_topk2_claim_supported"
        ),
    }


def _candidate_actions(evidence: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "candidate_action": "promote_causal_feature_safe_contextual_router",
            "disposition": "rejected",
            "reason": (
                "sequence CE is backend-stable, but local and RunPod support audits "
                "block promotion on oracle-regret and functional-churn gates"
            ),
        },
        {
            "candidate_action": "rerun_sequence_kfold_on_gpu",
            "disposition": "rejected",
            "reason": "RunPod sequence closeout already validates the local K-fold result",
        },
        {
            "candidate_action": "resume_support_label_distillation",
            "disposition": "rejected",
            "reason": "distillation remains frozen until same-student functional gates pass",
        },
        {
            "candidate_action": "revive_topk2_causal_cooperation_claim",
            "disposition": "rejected",
            "reason": (
                "deconfounded audit decision is "
                f"{evidence.get('deconfounded_decision')}"
            ),
        },
        {
            "candidate_action": "inspect_causal_router_oracle_regret_and_churn_failure",
            "disposition": "selected",
            "reason": (
                "the causal router wins CE over linear but loses the support-quality "
                "gates in both local and RunPod artifacts"
            ),
        },
    ]


def _failures(
    source_rows: list[dict[str, Any]], evidence: dict[str, Any]
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
        "sequence_decision": SEQUENCE_VALIDATED,
        "local_support_decision": SUPPORT_AUDIT_BLOCKED,
        "runpod_support_decision": SUPPORT_AUDIT_BLOCKED,
        "deconfounded_decision": DECONFOUNDED_TOPK2_BLOCKED,
        "causal_coverage_decision": CAUSAL_COVERAGE_TOPK1_ACTIVE,
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
    boolean_expected = {
        "sequence_causal_beats_linear_both_backends": True,
        "sequence_full_context_beats_causal_both_backends": True,
        "causal_coverage_rank_matched_topk1_active": True,
        "causal_coverage_topk2_claim_supported": False,
    }
    for field, expected_value in boolean_expected.items():
        if evidence.get(field) is not expected_value:
            failures.append(
                {
                    "source": "evidence",
                    "field": field,
                    "expected": expected_value,
                    "actual": evidence.get(field),
                }
            )
    return failures


def _aggregate(packet: dict[str, Any], control: str) -> dict[str, Any]:
    aggregate = _nested(packet, "audit", "aggregate_metrics")
    if not isinstance(aggregate, dict):
        return {}
    row = aggregate.get(control, {})
    return row if isinstance(row, dict) else {}


def _nested(packet: dict[str, Any], *keys: str) -> Any:
    current: Any = packet
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _source_row(source: str, path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": packet.get("status"),
        "decision": packet.get("decision"),
        "claim_status": packet.get("claim_status"),
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "strategic_change_level": None,
            "notify_ben": None,
            "recommended_next_action": None,
            "ben_notification_required": False,
            "incorporation": "optional review not present",
        }
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:12]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"strategic_change_level", "notify_ben", "recommended_next_action"}:
            header[key] = value.strip()
    notify_ben = _bool_or_none(header.get("notify_ben"))
    major = header.get("strategic_change_level") == "major"
    return {
        "path": str(path),
        "present": True,
        "strategic_change_level": header.get("strategic_change_level"),
        "notify_ben": notify_ben,
        "recommended_next_action": header.get("recommended_next_action"),
        "ben_notification_required": bool(notify_ben) or major,
        "incorporation": (
            "accepted: the review requested sequence-heldout causal-feature "
            "ablation and non-CE controls before any new distillation. This "
            "decision consumes the completed sequence closeout, support-quality "
            "audits, and deconfounded rank controls, then blocks promotion."
        ),
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
        "# Contextual Router Post-sequence Decision",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next step: `{summary['selected_next_step']}`",
        f"- Ben notification required: `{summary['strategy_review']['ben_notification_required']}`",
        "",
        "## Key Evidence",
        "",
        f"- Sequence backend closeout: `{evidence['sequence_decision']}`",
        f"- Local support audit: `{evidence['local_support_decision']}`",
        f"- RunPod support audit: `{evidence['runpod_support_decision']}`",
        f"- Deconfounded top-k-2 audit: `{evidence['deconfounded_decision']}`",
        f"- Causal coverage report: `{evidence['causal_coverage_decision']}`",
        f"- Local causal/linear oracle regret: `{evidence['local_causal_oracle_regret']}` / `{evidence['local_linear_oracle_regret']}`",
        f"- RunPod causal/linear oracle regret: `{evidence['runpod_causal_oracle_regret']}` / `{evidence['runpod_linear_oracle_regret']}`",
        f"- Local causal/linear functional churn: `{evidence['local_causal_functional_churn']}` / `{evidence['local_linear_functional_churn']}`",
        f"- RunPod causal/linear functional churn: `{evidence['runpod_causal_functional_churn']}` / `{evidence['runpod_linear_functional_churn']}`",
        "",
        "## Rationale",
        "",
        summary["rationale"],
        "",
        "## Candidate Actions",
    ]
    for row in summary["candidate_actions"]:
        lines.append(
            f"- {row['candidate_action']}: `{row['disposition']}` - {row['reason']}"
        )
    if summary["failures"]:
        lines.extend(["", "## Failures"])
        for failure in summary["failures"]:
            lines.append(f"- `{failure}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-closeout", type=Path, default=DEFAULT_SEQUENCE_CLOSEOUT)
    parser.add_argument("--local-support-audit", type=Path, default=DEFAULT_LOCAL_SUPPORT_AUDIT)
    parser.add_argument("--runpod-support-audit", type=Path, default=DEFAULT_RUNPOD_SUPPORT_AUDIT)
    parser.add_argument(
        "--deconfounded-intervention-audit",
        type=Path,
        default=DEFAULT_DECONFOUNDED_INTERVENTION_AUDIT,
    )
    parser.add_argument("--causal-coverage-report", type=Path, default=DEFAULT_CAUSAL_COVERAGE_REPORT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_contextual_router_post_sequence_decision_report(
        sequence_closeout_path=args.sequence_closeout,
        local_support_audit_path=args.local_support_audit,
        runpod_support_audit_path=args.runpod_support_audit,
        deconfounded_intervention_audit_path=args.deconfounded_intervention_audit,
        causal_coverage_report_path=args.causal_coverage_report,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "claim_status": summary["claim_status"],
                "selected_next_step": summary["selected_next_step"],
                "failures": summary["failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
