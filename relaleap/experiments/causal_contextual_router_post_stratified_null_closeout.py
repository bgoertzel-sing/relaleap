"""Close out causal-contextual router claims after stratified-null reversal."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_STRATIFIED_NULL_DIR = Path(
    "results/reports/token_larger_causal_contextual_router_stratified_null_reversal"
)
DEFAULT_SAME_STUDENT_DIR = Path(
    "results/reports/token_larger_causal_contextual_router_same_student_intervention_matrix"
)
DEFAULT_DISCRIMINATIVE_DIR = Path(
    "results/reports/token_larger_causal_contextual_router_discriminative_mechanism_audit"
)
DEFAULT_SUPPORT_AUDIT_DIR = Path(
    "results/audits/local_causal_contextual_router_support_audit"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_causal_contextual_router_post_stratified_null_closeout"
)

POST_STRATIFIED_NULL_CLOSEOUT = (
    "causal_contextual_router_distillation_branch_closed_no_promotion"
)
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def run_causal_contextual_router_post_stratified_null_closeout(
    *,
    stratified_null_dir: Path = DEFAULT_STRATIFIED_NULL_DIR,
    same_student_dir: Path = DEFAULT_SAME_STUDENT_DIR,
    discriminative_dir: Path = DEFAULT_DISCRIMINATIVE_DIR,
    support_audit_dir: Path = DEFAULT_SUPPORT_AUDIT_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Synthesize the stronger null controls into one branch decision."""

    start = time.time()
    packets = {
        "stratified_null_reversal": _read_json_object(stratified_null_dir / "summary.json"),
        "same_student_intervention_matrix": _read_json_object(same_student_dir / "summary.json"),
        "discriminative_mechanism_audit": _read_json_object(discriminative_dir / "summary.json"),
        "causal_support_audit": _read_json_object(support_audit_dir / "summary.json"),
    }
    paths = {
        "stratified_null_reversal": stratified_null_dir / "summary.json",
        "same_student_intervention_matrix": same_student_dir / "summary.json",
        "discriminative_mechanism_audit": discriminative_dir / "summary.json",
        "causal_support_audit": support_audit_dir / "summary.json",
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
    metrics = _metrics(packets)
    closure_rows = _closure_rows(metrics)
    failures = _failures(source_rows, packets, metrics)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        claim_status = INSUFFICIENT_EVIDENCE
        rationale = (
            "The post-stratified-null closeout cannot be interpreted because a "
            "required source artifact is missing, failing, or inconsistent with "
            "the strengthened-null evidence chain."
        )
        selected_next_step = "repair_missing_or_inconsistent_stratified_null_closeout_sources"
    else:
        status = "pass"
        decision = POST_STRATIFIED_NULL_CLOSEOUT
        claim_status = "causal_router_ce_baseline_only_support_mechanism_not_established"
        rationale = (
            "The causal contextual router remains a useful CE/support-diversity "
            "baseline, but the support-quality audit blocks promotion, the "
            "same-student matrix shows teacher support is not functionally better "
            "than the token/position null, and the stratified-null reversal "
            "supersedes the earlier discriminative-mechanism claim. Default "
            "promotion and distillation-mechanism claims therefore stay blocked."
        )
        selected_next_step = (
            "return_to_non_distillation_architecture_loop_with_causal_router_as_ce_baseline"
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "claim_statuses": {
            "contextual_mlp_full_context": "nondeployable_oracle_diagnostic_only",
            "contextual_mlp_causal": "ce_supported_not_promoted",
            "causal_router_distillation": "closed_not_functionally_established",
            "teacher_exact_pair_agreement": "insufficient_without_functional_null_margin",
            "topk2_causal_cooperation": "not_supported",
        },
        "source_rows": source_rows,
        "closure_rows": closure_rows,
        "metrics": metrics,
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
            "closure_rows_csv": str(out_dir / "closure_rows.csv"),
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
        out_dir / "closure_rows.csv",
        ["branch", "source_decision", "key_metric", "key_value", "disposition"],
        closure_rows,
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _metrics(packets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stratified = packets["stratified_null_reversal"]
    same_student = packets["same_student_intervention_matrix"]
    support = packets["causal_support_audit"]
    support_audit = support.get("audit", {})
    support_aggregates = support_audit.get("aggregate_metrics", {})
    causal = support_aggregates.get("causal_contextual_topk2", {})
    linear = support_aggregates.get("linear_topk2", {})
    return {
        "stratified_null_decision": stratified.get("decision"),
        "stratified_null_passes_reversal_gate": stratified.get("gate_status", {}).get(
            "passes_reversal_gate"
        ),
        "stratified_null_seed_count": len(stratified.get("seed_rows", [])),
        "same_student_decision": same_student.get("decision"),
        "teacher_minus_token_position_null_gain_all_tokens": same_student.get(
            "key_metrics", {}
        ).get("teacher_minus_token_position_null_gain_all_tokens"),
        "teacher_forced_gain_all_tokens": same_student.get("key_metrics", {}).get(
            "teacher_forced_gain_all_tokens"
        ),
        "discriminative_decision": packets["discriminative_mechanism_audit"].get(
            "decision"
        ),
        "support_audit_decision": support.get("decision"),
        "causal_router_loss_delta_vs_linear": causal.get(
            "mean_router_loss_delta_vs_linear"
        ),
        "causal_oracle_regret_delta_vs_linear": causal.get(
            "mean_oracle_regret_delta_vs_linear"
        ),
        "causal_functional_churn_delta_vs_linear": causal.get(
            "mean_functional_churn_delta_vs_linear"
        ),
        "linear_oracle_support_regret": linear.get("mean_oracle_support_regret"),
        "causal_oracle_support_regret": causal.get("mean_oracle_support_regret"),
    }


def _closure_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "branch": "causal_contextual_router_support_quality",
            "source_decision": metrics["support_audit_decision"],
            "key_metric": "causal_oracle_regret_delta_vs_linear",
            "key_value": metrics["causal_oracle_regret_delta_vs_linear"],
            "disposition": "blocks_default_promotion",
        },
        {
            "branch": "same_student_teacher_support_intervention",
            "source_decision": metrics["same_student_decision"],
            "key_metric": "teacher_minus_token_position_null_gain_all_tokens",
            "key_value": metrics["teacher_minus_token_position_null_gain_all_tokens"],
            "disposition": "blocks_functional_teacher_support_claim",
        },
        {
            "branch": "distillation_discriminative_mechanism",
            "source_decision": metrics["discriminative_decision"],
            "key_metric": "stratified_null_passes_reversal_gate",
            "key_value": metrics["stratified_null_passes_reversal_gate"],
            "disposition": "superseded_by_stratified_null",
        },
    ]


def _failures(
    source_rows: list[dict[str, Any]],
    packets: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows[:4]:
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
        elif row["status"] != "pass":
            failures.append(
                {
                    "source": row["source"],
                    "field": "status",
                    "expected": "pass",
                    "actual": row["status"],
                }
            )
    expectations = (
        (
            "stratified_null_reversal",
            "decision",
            "prior_distillation_mechanism_claim_superseded_by_stratified_null",
        ),
        (
            "same_student_intervention_matrix",
            "decision",
            "same_student_token_position_null_discriminator_blocks_claim",
        ),
        (
            "causal_support_audit",
            "decision",
            "causal_contextual_router_support_audit_blocks_promotion",
        ),
    )
    for source, field, expected in expectations:
        actual = packets[source].get(field)
        if actual != expected:
            failures.append(
                {
                    "source": source,
                    "field": field,
                    "expected": expected,
                    "actual": actual,
                }
            )
    if metrics["stratified_null_passes_reversal_gate"] is not True:
        failures.append(
            {
                "source": "stratified_null_reversal",
                "field": "passes_reversal_gate",
                "expected": True,
                "actual": metrics["stratified_null_passes_reversal_gate"],
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
    header = {}
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
        "ben_notification_required": bool(notify_ben) or major,
        "incorporation": (
            "accepted where applicable: this closeout follows the latest review's "
            "preference for non-CE retention/churn and same-student controls, and "
            "records that the existing stronger token/position nulls block causal "
            "router distillation/default-promotion claims"
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
    lines = [
        "# Causal Contextual Router Post-Stratified-Null Closeout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next step: `{summary['selected_next_step']}`",
        f"- Rationale: {summary['rationale']}",
        "",
        "## Strategy Review",
        "",
        f"- Present: `{summary['strategy_review']['present']}`",
        f"- Strategic change level: `{summary['strategy_review']['strategic_change_level']}`",
        f"- Notify Ben: `{summary['strategy_review']['notify_ben']}`",
        f"- Incorporation: {summary['strategy_review']['incorporation']}",
    ]
    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in summary["failures"]:
            lines.append(
                f"- `{failure.get('source')}` `{failure.get('field')}`: "
                f"expected `{failure.get('expected')}`, actual `{failure.get('actual')}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Close causal-contextual router distillation/default-promotion claims "
            "after same-student and token/position stratified-null controls."
        )
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stratified-null-dir", type=Path, default=DEFAULT_STRATIFIED_NULL_DIR)
    parser.add_argument("--same-student-dir", type=Path, default=DEFAULT_SAME_STUDENT_DIR)
    parser.add_argument("--discriminative-dir", type=Path, default=DEFAULT_DISCRIMINATIVE_DIR)
    parser.add_argument("--support-audit-dir", type=Path, default=DEFAULT_SUPPORT_AUDIT_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    args = parser.parse_args()
    summary = run_causal_contextual_router_post_stratified_null_closeout(
        out_dir=args.out,
        stratified_null_dir=args.stratified_null_dir,
        same_student_dir=args.same_student_dir,
        discriminative_dir=args.discriminative_dir,
        support_audit_dir=args.support_audit_dir,
        strategy_review_path=args.strategy_review,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
