"""Local pivot audit from ACSR anticipation to causal support routing."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_ACSR_CAPACITY_AUDIT = Path(
    "results/audits/acsr_causal_router_capacity_audit_local/summary.json"
)
DEFAULT_STRATIFIED_NULL_REPORT = Path(
    "results/reports/token_larger_causal_contextual_router_stratified_null_reversal/summary.json"
)
DEFAULT_SAME_STUDENT_REPORT = Path(
    "results/reports/token_larger_causal_contextual_router_same_student_intervention_matrix/summary.json"
)
DEFAULT_CONDITIONAL_RESAMPLE_REPORT = Path(
    "results/reports/token_larger_causal_contextual_router_conditional_permutation_resample_matrix/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/acsr_causal_support_router_pivot_audit")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_reports.csv",
    "gate_criteria.csv",
    "mechanism_evidence.csv",
    "notes.md",
)


def run_acsr_causal_support_router_pivot_audit(
    *,
    acsr_capacity_audit: Path = DEFAULT_ACSR_CAPACITY_AUDIT,
    stratified_null_report: Path = DEFAULT_STRATIFIED_NULL_REPORT,
    same_student_report: Path = DEFAULT_SAME_STUDENT_REPORT,
    conditional_resample_report: Path = DEFAULT_CONDITIONAL_RESAMPLE_REPORT,
    strategy_review: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Gate the post-ACSR pivot using existing local mechanism artifacts."""

    start = time.time()
    sources = {
        "acsr_capacity_audit": _read_json_object(acsr_capacity_audit),
        "stratified_null_report": _read_json_object(stratified_null_report),
        "same_student_report": _read_json_object(same_student_report),
        "conditional_resample_report": _read_json_object(conditional_resample_report),
    }
    source_paths = {
        "acsr_capacity_audit": acsr_capacity_audit,
        "stratified_null_report": stratified_null_report,
        "same_student_report": same_student_report,
        "conditional_resample_report": conditional_resample_report,
    }
    review = _strategy_review_notes(strategy_review)
    source_rows = _source_rows(sources, source_paths)
    evidence_rows = _evidence_rows(sources)
    gate = _gate_criteria(sources, review)
    failures = [
        {"gate": row["criterion"], "reason": row["failure_reason"]}
        for row in gate
        if not row["passed"]
    ]

    status = "pass" if not failures else "fail"
    functional_supported = _criterion(gate, "functional_token_position_null_support")[
        "passed"
    ]
    summary = {
        "status": status,
        "decision": (
            "causal_support_router_pivot_audit_passed"
            if status == "pass"
            else "causal_support_router_pivot_audit_failed_closed"
        ),
        "claim_status": (
            "direct_causal_support_router_mechanism_supported_not_promoted"
            if functional_supported and status == "pass"
            else "direct_causal_support_router_mechanism_not_established"
        ),
        "selected_next_step": (
            "design a capacity-matched causal support-router functional intervention with token-position stratified null and dual-student value deconfounding"
        ),
        "strategy_review": review,
        "direction_shift": _direction_shift(review),
        "source_reports": source_rows,
        "mechanism_evidence": evidence_rows,
        "gate_criteria": gate,
        "failures": failures,
        "interpretation": _interpretation(sources),
        "claim_boundaries": {
            "supported": [
                "ACSR-as-anticipation is blocked by a parameter-matched causal MLP control",
                "existing causal support-router artifacts are suitable local sources for the pivot audit",
            ],
            "not_supported": [
                "ACSR default promotion",
                "predicted future-context features as necessary support-routing variables",
                "direct causal support-router functional mechanism under token-position null",
                "dual-student value/support deconfounding for the causal-router pivot",
            ],
        },
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _source_rows(
    sources: dict[str, dict[str, Any]],
    source_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    rows = []
    for name, source in sources.items():
        rows.append(
            {
                "source": name,
                "path": str(source_paths[name]),
                "present": bool(source),
                "status": source.get("status", "missing"),
                "decision": source.get("decision", ""),
                "claim_status": source.get("claim_status", ""),
                "git_commit": source.get("git_commit", ""),
            }
        )
    return rows


def _evidence_rows(sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    acsr = sources["acsr_capacity_audit"].get("aggregate_metrics", {})
    same_student = sources["same_student_report"].get("key_metrics", {})
    conditional = sources["conditional_resample_report"].get("key_metrics", {})
    return [
        {
            "evidence": "acsr_vs_parameter_matched_causal_mlp",
            "status": sources["acsr_capacity_audit"].get("claim_status", ""),
            "metric": "mean_acsr_minus_parameter_matched_ce_loss",
            "value": acsr.get("mean_acsr_minus_parameter_matched_ce_loss", ""),
            "interpretation": "positive means ACSR is worse than the direct causal router control",
        },
        {
            "evidence": "acsr_support_redundancy",
            "status": "available" if acsr.get("support_agreement_available") else "missing",
            "metric": "mean_high_support_match_acsr_minus_parameter_matched_ce_loss",
            "value": acsr.get("mean_high_support_match_acsr_minus_parameter_matched_ce_loss", ""),
            "interpretation": "high support agreement makes predicted future features redundant in current packets",
        },
        {
            "evidence": "same_student_token_position_null",
            "status": sources["same_student_report"].get("claim_status", ""),
            "metric": "teacher_minus_token_position_null_gain_all_tokens",
            "value": same_student.get("teacher_minus_token_position_null_gain_all_tokens", ""),
            "interpretation": "positive would favor teacher support over the token-position null through the same values",
        },
        {
            "evidence": "conditional_assignment_signal",
            "status": sources["conditional_resample_report"].get("claim_status", ""),
            "metric": "student_exact_agreement_effect_vs_null_mean",
            "value": conditional.get("student_exact_agreement_effect_vs_null_mean", ""),
            "interpretation": "assignment signal alone is not functional support usefulness",
        },
    ]


def _gate_criteria(
    sources: dict[str, dict[str, Any]],
    review: dict[str, Any],
) -> list[dict[str, Any]]:
    acsr_status = sources["acsr_capacity_audit"].get("claim_status", "")
    stratified_status = sources["stratified_null_report"].get("claim_status", "")
    same_student_status = sources["same_student_report"].get("claim_status", "")
    conditional_status = sources["conditional_resample_report"].get("claim_status", "")
    same_student_metrics = sources["same_student_report"].get("key_metrics", {})
    teacher_minus_null = _number(
        same_student_metrics.get("teacher_minus_token_position_null_gain_all_tokens")
    )
    same_student_functional_supported = (
        teacher_minus_null is not None
        and teacher_minus_null > 0.0
        and "not_established" not in same_student_status
        and "blocks_claim" not in same_student_status
    )
    rows = [
        _criterion_row(
            "strategy_review_consumed",
            review.get("status") == "read",
            "latest strategy review is read before selecting the pivot step",
            {
                "status": review.get("status"),
                "strategic_change_level": review.get("strategic_change_level"),
                "notify_ben": review.get("notify_ben"),
            },
            "latest GPT-5.5-Pro strategy review was not consumed",
        ),
        _criterion_row(
            "acsr_anticipation_blocked",
            acsr_status == "acsr_as_anticipation_blocked_by_capacity_matched_causal_router",
            "ACSR capacity audit blocks the anticipation-specific claim",
            acsr_status,
            "ACSR capacity audit has not blocked the anticipation claim",
        ),
        _criterion_row(
            "causal_router_stratified_null_report_present",
            bool(stratified_status),
            "stratified token-position null report exists",
            stratified_status,
            "missing stratified token-position null report",
        ),
        _criterion_row(
            "same_student_token_position_matrix_present",
            bool(same_student_status),
            "same-student intervention matrix exists",
            same_student_status,
            "missing same-student token-position intervention matrix",
        ),
        _criterion_row(
            "assignment_signal_not_overread",
            "functional_claim_blocked" in conditional_status
            or "not_established" in conditional_status,
            "conditional assignment report blocks functional overclaim",
            conditional_status,
            "conditional assignment report does not bound functional claims",
        ),
        _criterion_row(
            "functional_token_position_null_support",
            same_student_functional_supported,
            "same-student source report supports teacher support over token-position null in forced loss",
            {
                "claim_status": same_student_status,
                "teacher_minus_token_position_null_gain_all_tokens": teacher_minus_null,
            },
            "same-student source report does not establish functional support over token-position null",
        ),
    ]
    return rows


def _criterion_row(
    criterion: str,
    passed: bool,
    threshold: str,
    actual: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": passed,
        "threshold": threshold,
        "actual": actual,
        "failure_reason": "" if passed else failure_reason,
    }


def _criterion(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in rows:
        if row["criterion"] == name:
            return row
    raise KeyError(name)


def _interpretation(sources: dict[str, dict[str, Any]]) -> str:
    same_student_metrics = sources["same_student_report"].get("key_metrics", {})
    teacher_minus_null = same_student_metrics.get(
        "teacher_minus_token_position_null_gain_all_tokens"
    )
    return (
        "The pivot is scientifically sensible, but the current local causal "
        "support-router mechanism remains fail-closed: ACSR does not beat the "
        "parameter-matched causal MLP control, and existing same-student evidence "
        f"does not show functional teacher-support gain over the token-position "
        f"null (teacher-minus-null gain={teacher_minus_null})."
    )


def _strategy_review_notes(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "status": "missing",
            "recommendation_accepted": False,
        }
    notes: dict[str, Any] = {
        "path": str(path),
        "status": "read",
        "recommendation_accepted": True,
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {
            "strategic_change_level",
            "notify_ben",
            "recommended_next_action",
            "verdict",
        }:
            notes[key] = value.strip()
    return notes


def _direction_shift(review: dict[str, Any]) -> str:
    if review.get("strategic_change_level") == "major" or review.get("notify_ben") == "true":
        return (
            "Major GPT-5.5-Pro pivot remains accepted: freeze ACSR promotion/GPU "
            "repeats and audit direct causal support routing locally. Ben should "
            f"be notified: {review.get('notify_ben')}."
        )
    return "No major strategic direction shift recorded by the latest review."


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        out_dir / "source_reports.csv",
        ["source", "path", "present", "status", "decision", "claim_status", "git_commit"],
        summary["source_reports"],
    )
    _write_csv(
        out_dir / "gate_criteria.csv",
        ["criterion", "passed", "threshold", "actual", "failure_reason"],
        summary["gate_criteria"],
    )
    _write_csv(
        out_dir / "mechanism_evidence.csv",
        ["evidence", "status", "metric", "value", "interpretation"],
        summary["mechanism_evidence"],
    )
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ACSR Causal Support Router Pivot Audit",
        "",
        f"Status: `{summary['status']}`",
        f"Claim status: `{summary['claim_status']}`",
        "",
        summary["interpretation"],
        "",
        f"Selected next step: {summary['selected_next_step']}",
        "",
        summary["direction_shift"],
        "",
        "## Gate Criteria",
    ]
    for row in summary["gate_criteria"]:
        mark = "pass" if row["passed"] else "fail"
        lines.append(f"- {row['criterion']}: {mark} ({row['actual']})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acsr-capacity-audit", type=Path, default=DEFAULT_ACSR_CAPACITY_AUDIT)
    parser.add_argument("--stratified-null-report", type=Path, default=DEFAULT_STRATIFIED_NULL_REPORT)
    parser.add_argument("--same-student-report", type=Path, default=DEFAULT_SAME_STUDENT_REPORT)
    parser.add_argument(
        "--conditional-resample-report",
        type=Path,
        default=DEFAULT_CONDITIONAL_RESAMPLE_REPORT,
    )
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_acsr_causal_support_router_pivot_audit(
        acsr_capacity_audit=args.acsr_capacity_audit,
        stratified_null_report=args.stratified_null_report,
        same_student_report=args.same_student_report,
        conditional_resample_report=args.conditional_resample_report,
        strategy_review=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
