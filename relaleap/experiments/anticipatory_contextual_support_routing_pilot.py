"""Fail-closed contract wrapper for the existing ACSR pilot artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_AUDIT_DIR = Path("results/audits/token_larger_anticipatory_contextual_support_routing")
DEFAULT_DENSE_SYNTHESIS = Path("results/reports/acsr_dense_rank_norm_synthesis/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/anticipatory_contextual_support_routing_pilot_contract")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)

REQUIRED_PACKET_FILES = (
    "summary.json",
    "router_metrics.csv",
    "same_student_metrics.csv",
    "feature_perturbation.csv",
    "retention_churn_metrics.csv",
    "parameter_counts.csv",
)

REQUIRED_ROUTER_VARIANTS = (
    "full_context_contextual_topk2_teacher",
    "causal_feature_safe_contextual_topk2",
    "acsr_mlp_predicted_future",
    "shuffled_predicted_features",
    "token_position_only_predicted_features",
    "random_fixed_topk2",
    "rank_matched_contextual_topk1",
    "parameter_matched_causal_mlp_control",
)


def run_anticipatory_contextual_support_routing_pilot_contract(
    *,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    dense_synthesis_path: Path = DEFAULT_DENSE_SYNTHESIS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Validate that the reviewed ACSR pilot contract is already artifact-backed."""

    start = time.time()
    packet_summary = _read_json(audit_dir / "summary.json")
    router_rows = _read_csv(audit_dir / "router_metrics.csv")
    same_student_rows = _read_csv(audit_dir / "same_student_metrics.csv")
    perturbation_rows = _read_csv(audit_dir / "feature_perturbation.csv")
    retention_rows = _read_csv(audit_dir / "retention_churn_metrics.csv")
    parameter_rows = _read_csv(audit_dir / "parameter_counts.csv")
    dense_synthesis = _read_json(dense_synthesis_path)
    strategy = _strategy_review(strategy_review_path)

    metrics = _metrics(
        audit_dir=audit_dir,
        packet_summary=packet_summary,
        router_rows=router_rows,
        same_student_rows=same_student_rows,
        perturbation_rows=perturbation_rows,
        retention_rows=retention_rows,
        parameter_rows=parameter_rows,
        dense_synthesis=dense_synthesis,
    )
    gate_rows = _gate_rows(
        audit_dir=audit_dir,
        packet_summary=packet_summary,
        router_rows=router_rows,
        same_student_rows=same_student_rows,
        perturbation_rows=perturbation_rows,
        retention_rows=retention_rows,
        parameter_rows=parameter_rows,
        dense_synthesis=dense_synthesis,
        strategy=strategy,
    )
    hard_failures = [row for row in gate_rows if not row["passed"] and row["severity"] == "hard"]
    status = "fail" if hard_failures else "pass"
    decision = (
        "anticipatory_contextual_support_routing_pilot_contract_recorded"
        if status == "pass"
        else "anticipatory_contextual_support_routing_pilot_contract_failed_closed"
    )
    claim_status = (
        "acsr_pilot_artifact_contract_satisfied_not_promoted"
        if status == "pass"
        else "acsr_pilot_artifact_contract_incomplete"
    )
    selected_next_step = (
        "keep ACSR local and require sparse-vs-dense mechanism separation before GPU validation"
        if status == "pass"
        else "repair ACSR pilot packet before any interpretation"
    )
    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "backend_policy": "local contract validation only; no RunPod/Colab action selected",
        "audit_dir": str(audit_dir),
        "dense_synthesis_path": str(dense_synthesis_path),
        "required_router_variants": list(REQUIRED_ROUTER_VARIANTS),
        "metrics": metrics,
        "gate_criteria": gate_rows,
        "failures": hard_failures,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "direction_shift": {
            "strategic_change_level": strategy["strategic_change_level"],
            "notify_ben": strategy["notify_ben"],
            "ben_should_be_notified": strategy["ben_notification_required"],
            "record": (
                "Latest review is major/notify_ben=true; Ben should be notified that "
                "the automation is honoring the ACSR pivot while treating the literal "
                "pilot request as already satisfied by existing command-generated artifacts."
                if strategy["ben_notification_required"]
                else "No Ben notification requested by latest review header."
            ),
        },
        "rationale": (
            "The expected ACSR pilot module name now has a command-driven contract. "
            "It validates the existing source-of-truth ACSR pilot packet and the dense "
            "rank/norm synthesis instead of duplicating training. Promotion remains "
            "blocked until sparse support evidence separates from dense controls."
        ),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _metrics(
    *,
    audit_dir: Path,
    packet_summary: dict[str, Any],
    router_rows: list[dict[str, str]],
    same_student_rows: list[dict[str, str]],
    perturbation_rows: list[dict[str, str]],
    retention_rows: list[dict[str, str]],
    parameter_rows: list[dict[str, str]],
    dense_synthesis: dict[str, Any],
) -> list[dict[str, Any]]:
    by_variant = {row.get("variant", ""): row for row in router_rows}
    acsr = by_variant.get("acsr_mlp_predicted_future", {})
    causal = by_variant.get("causal_feature_safe_contextual_topk2", {})
    shuffled = by_variant.get("shuffled_predicted_features", {})
    token = by_variant.get("token_position_only_predicted_features", {})
    parameter = by_variant.get("parameter_matched_causal_mlp_control", {})
    return [
        {
            "metric": "packet_status",
            "value": packet_summary.get("status"),
            "interpretation": "source ACSR packet status",
        },
        {
            "metric": "acsr_minus_causal_oracle_regret",
            "value": _delta(acsr.get("oracle_regret"), causal.get("oracle_regret")),
            "interpretation": "negative favors predicted-feature ACSR over causal-feature-safe baseline",
        },
        {
            "metric": "acsr_minus_token_position_oracle_regret",
            "value": _delta(acsr.get("oracle_regret"), token.get("oracle_regret")),
            "interpretation": "negative favors ACSR over token/position null",
        },
        {
            "metric": "acsr_minus_shuffled_oracle_regret",
            "value": _delta(acsr.get("oracle_regret"), shuffled.get("oracle_regret")),
            "interpretation": "negative favors ACSR over shuffled predicted-feature null",
        },
        {
            "metric": "acsr_minus_parameter_matched_control_ce",
            "value": _delta(acsr.get("ce_loss"), parameter.get("ce_loss")),
            "interpretation": "positive means parameter-matched causal control slightly beats ACSR",
        },
        {
            "metric": "same_student_comparison_count",
            "value": len(same_student_rows),
            "interpretation": "support interventions through the same residual values",
        },
        {
            "metric": "feature_perturbation_count",
            "value": len(perturbation_rows),
            "interpretation": "future-perturbation leakage checks",
        },
        {
            "metric": "retention_churn_row_count",
            "value": len(retention_rows),
            "interpretation": "retention/churn rows available in packet",
        },
        {
            "metric": "parameter_matched_control_available",
            "value": any(
                row.get("component") == "parameter_matched_causal_mlp_control"
                and row.get("status") == "available"
                for row in parameter_rows
            ),
            "interpretation": "rank/control accounting inside the ACSR packet",
        },
        {
            "metric": "dense_rank_norm_claim_status",
            "value": dense_synthesis.get("claim_status"),
            "interpretation": "external dense rank/norm synthesis remains the promotion blocker",
        },
        {
            "metric": "audit_dir",
            "value": str(audit_dir),
            "interpretation": "source packet directory",
        },
    ]


def _gate_rows(
    *,
    audit_dir: Path,
    packet_summary: dict[str, Any],
    router_rows: list[dict[str, str]],
    same_student_rows: list[dict[str, str]],
    perturbation_rows: list[dict[str, str]],
    retention_rows: list[dict[str, str]],
    parameter_rows: list[dict[str, str]],
    dense_synthesis: dict[str, Any],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    present_files = [name for name in REQUIRED_PACKET_FILES if (audit_dir / name).is_file()]
    missing_files = sorted(set(REQUIRED_PACKET_FILES) - set(present_files))
    variants = {row.get("variant", "") for row in router_rows}
    missing_variants = sorted(set(REQUIRED_ROUTER_VARIANTS) - variants)
    same_student_comparisons = {row.get("comparison", "") for row in same_student_rows}
    perturbation_pass = any(
        row.get("control_type") == "future_perturbation_negative"
        and str(row.get("passed", "")).lower() == "true"
        for row in perturbation_rows
    )
    leaky_positive = any(
        row.get("control_type") == "leaky_future_positive"
        and str(row.get("passed", "")).lower() == "true"
        for row in perturbation_rows
    )
    parameter_control = any(
        row.get("component") == "parameter_matched_causal_mlp_control"
        and row.get("status") == "available"
        for row in parameter_rows
    )
    return [
        _criterion(
            "strategy_review_consumed",
            strategy["present"],
            "hard",
            "latest external strategy review exists and was parsed",
            strategy["recommended_next_action"],
            "missing strategy review",
        ),
        _criterion(
            "packet_files_present",
            not missing_files,
            "hard",
            "ACSR packet contains required artifact files",
            present_files,
            f"missing packet files: {missing_files}",
        ),
        _criterion(
            "packet_status_pass",
            packet_summary.get("status") == "pass",
            "hard",
            "ACSR packet summary passed",
            packet_summary.get("status"),
            "ACSR packet did not pass",
        ),
        _criterion(
            "required_router_arms_present",
            not missing_variants,
            "hard",
            "teacher/current/predicted/null/random/rank/control arms are present",
            sorted(variants),
            f"missing router variants: {missing_variants}",
        ),
        _criterion(
            "same_student_null_controls_present",
            {
                "acsr_mlp_predicted_future_support_vs_shuffled_predicted_features",
                "acsr_mlp_predicted_future_support_vs_token_position_only_predicted_features",
            }.issubset(same_student_comparisons),
            "hard",
            "same-student ACSR vs shuffled and token-position forced-support controls exist",
            sorted(same_student_comparisons),
            "missing same-student shuffled or token-position control",
        ),
        _criterion(
            "future_perturbation_controls_present",
            perturbation_pass and leaky_positive,
            "hard",
            "future negative perturbation and leaky positive controls exist",
            {"negative": perturbation_pass, "positive": leaky_positive},
            "missing leakage perturbation controls",
        ),
        _criterion(
            "retention_churn_available",
            bool(retention_rows),
            "hard",
            "retention/churn rows exist",
            len(retention_rows),
            "missing retention/churn rows",
        ),
        _criterion(
            "parameter_matched_causal_control_available",
            parameter_control,
            "hard",
            "parameter-matched causal MLP control is available",
            parameter_control,
            "missing parameter-matched causal control",
        ),
        _criterion(
            "dense_rank_norm_synthesis_available",
            dense_synthesis.get("status") == "pass"
            and dense_synthesis.get("decision")
            == "acsr_sparse_support_claim_blocked_by_dense_rank_norm_controls",
            "hard",
            "dense rank/norm synthesis is available as the sparse-claim blocker",
            {
                "status": dense_synthesis.get("status"),
                "decision": dense_synthesis.get("decision"),
            },
            "dense rank/norm synthesis missing or not in expected blocking state",
        ),
    ]


def _criterion(
    criterion: str,
    passed: bool,
    severity: str,
    requirement: str,
    observed: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "severity": severity,
        "requirement": requirement,
        "observed": observed,
        "failure_reason": "" if passed else failure_reason,
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "present": path.is_file(),
        "strategic_change_level": "",
        "notify_ben": "",
        "recommended_next_action": "",
        "verdict": "",
        "ben_notification_required": False,
    }
    if not path.is_file():
        return fields
    for line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in fields:
            fields[key] = value.strip()
    fields["ben_notification_required"] = (
        str(fields["notify_ben"]).lower() == "true"
        or fields["strategic_change_level"] == "major"
    )
    return fields


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "Strategy review missing; contract fails closed."
    if strategy["ben_notification_required"]:
        return (
            "Accepted major ACSR pivot/no-GPU recommendation. Deferred literal new "
            "training because the source-of-truth ACSR pilot artifacts already exist; "
            "Ben should be notified per review header."
        )
    return "Accepted latest local ACSR recommendation without GPU escalation."


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _delta(left: Any, right: Any) -> float | None:
    try:
        return float(left) - float(right)
    except (TypeError, ValueError):
        return None


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "metrics.csv", summary["metrics"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Anticipatory Contextual Support Routing Pilot Contract",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Audit dir: `{summary['audit_dir']}`",
        f"- Dense synthesis: `{summary['dense_synthesis_path']}`",
        f"- Strategy review level: `{summary['strategy_review']['strategic_change_level']}`",
        f"- Notify Ben: `{summary['strategy_review']['notify_ben']}`",
        "",
        str(summary["rationale"]),
        "",
        "## Next Step",
        "",
        str(summary["selected_next_step"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--dense-synthesis", type=Path, default=DEFAULT_DENSE_SYNTHESIS)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_anticipatory_contextual_support_routing_pilot_contract(
        audit_dir=args.audit_dir,
        dense_synthesis_path=args.dense_synthesis,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
