"""Held-out oracle-regret and functional-churn gate for ACSR."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_AUDIT_DIRS = (
    Path("results/audits/token_larger_anticipatory_contextual_support_routing"),
    Path("results/audits/token_larger_anticipatory_contextual_support_routing_seed2"),
    Path("results/runpod_fetch/audits/runpod_token_larger_anticipatory_contextual_support_routing"),
    Path("results/runpod_fetch/audits/runpod_token_larger_anticipatory_contextual_support_routing_seed2"),
)
DEFAULT_PREVIOUS_PROBE = Path(
    "results/reports/token_larger_anticipatory_contextual_support_routing_retention_churn_probe/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_anticipatory_contextual_support_routing_causal_mechanism_gate"
)

PRIMARY_VARIANT = "acsr_mlp_predicted_future"
CAUSAL_BASELINE = "causal_feature_safe_contextual_topk2"
CONTROL_VARIANTS = (
    "token_position_only_predicted_features",
    "shuffled_predicted_features",
)
REQUIRED_PACKET_FILES = (
    "summary.json",
    "router_metrics.csv",
    "retention_churn_metrics.csv",
    "feature_perturbation.csv",
)
ACSR_CAUSAL_MECHANISM_SUPPORTED_NOT_PROMOTED = (
    "acsr_heldout_oracle_regret_functional_churn_supported_not_promoted"
)
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def run_acsr_causal_mechanism_gate(
    *,
    audit_dirs: tuple[Path, ...] = DEFAULT_AUDIT_DIRS,
    previous_probe_path: Path = DEFAULT_PREVIOUS_PROBE,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Summarize ACSR oracle-regret and functional-churn gates from packets."""

    start = time.time()
    packet_rows = [
        _packet_row(index=index + 1, audit_dir=path)
        for index, path in enumerate(audit_dirs)
    ]
    criteria = _criteria(packet_rows, previous_probe_path)
    failures = _failures(packet_rows, criteria)
    aggregate_rows = _aggregate_rows(packet_rows)
    strategy_review = _strategy_review(strategy_review_path)
    previous_probe = _previous_probe(previous_probe_path)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        claim_status = "acsr_causal_mechanism_gate_not_interpretable"
        selected_next_step = (
            "repair or extend ACSR packets with explicit held-out oracle-regret "
            "and functional-churn fields"
        )
        rationale = (
            "The causal-mechanism gate failed closed because at least one "
            "packet was missing, failed its own smoke gates, or did not show "
            "ACSR matching or beating the causal/token-position/null controls "
            "on oracle regret and functional churn."
        )
    else:
        status = "pass"
        decision = ACSR_CAUSAL_MECHANISM_SUPPORTED_NOT_PROMOTED
        claim_status = "heldout_non_ce_acsr_mechanism_evidence_supported_not_promoted"
        selected_next_step = (
            "run a bounded no-default-promotion dense-teacher residual "
            "distillation comparison against ACSR and the promoted contextual router"
        )
        rationale = (
            "Across local and fetched RunPod packets, ACSR closes the "
            "causal-router oracle-regret gap, beats token/position and shuffled "
            "controls on fixed-context oracle regret, and keeps lower functional "
            "churn after the second-context update. The result supports ACSR as "
            "a mechanism candidate, but it is still not a default-router "
            "promotion because the evidence comes from small held-out packet "
            "contexts rather than a broader benchmark."
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "claim_statuses": {
            PRIMARY_VARIANT: claim_status,
            "promoted_default_router": "blocked_pending_broader_benchmark",
            "causal_mechanism_claim": claim_status,
            "dense_teacher_distillation": "recommended_next_comparison",
        },
        "strategy_review": strategy_review,
        "previous_probe": previous_probe,
        "packet_rows": [_public_packet_row(row) for row in packet_rows],
        "aggregate_rows": aggregate_rows,
        "gate_status": {
            "passes_heldout_oracle_regret_functional_churn_gate": not failures,
            "criteria": criteria,
        },
        "failures": failures,
        "rationale": rationale,
        "deferred_or_rejected_recommendations": _deferred_recommendations(strategy_review),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "packet_gate_metrics_csv": str(out_dir / "packet_gate_metrics.csv"),
            "aggregate_gate_metrics_csv": str(out_dir / "aggregate_gate_metrics.csv"),
            "gate_criteria_csv": str(out_dir / "gate_criteria.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        out_dir / "packet_gate_metrics.csv",
        [
            "packet",
            "audit_dir",
            "present",
            "status",
            "future_perturbation_invariance",
            "acsr_oracle_regret",
            "causal_oracle_regret",
            "token_position_oracle_regret",
            "shuffled_oracle_regret",
            "acsr_minus_causal_oracle_regret",
            "acsr_minus_token_position_oracle_regret",
            "acsr_minus_shuffled_oracle_regret",
            "acsr_anchor_logit_mse_after_transfer",
            "token_position_anchor_logit_mse_after_transfer",
            "shuffled_anchor_logit_mse_after_transfer",
            "acsr_teacher_logit_mse",
            "token_position_teacher_logit_mse",
            "shuffled_teacher_logit_mse",
            "passes_oracle_regret_gate",
            "passes_functional_churn_gate",
        ],
        [_public_packet_row(row) for row in packet_rows],
    )
    _write_csv(
        out_dir / "aggregate_gate_metrics.csv",
        [
            "metric",
            "packet_count",
            "mean",
            "maximum",
            "all_pass",
        ],
        aggregate_rows,
    )
    _write_csv(
        out_dir / "gate_criteria.csv",
        ["criterion", "passed", "threshold", "actual"],
        criteria,
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _packet_row(index: int, audit_dir: Path) -> dict[str, Any]:
    summary = _read_json_object(audit_dir / "summary.json")
    router = {row.get("variant"): row for row in _read_csv_rows(audit_dir / "router_metrics.csv")}
    retention = {
        (row.get("phase"), row.get("variant")): row
        for row in _read_csv_rows(audit_dir / "retention_churn_metrics.csv")
    }
    perturbation = _read_csv_rows(audit_dir / "feature_perturbation.csv")

    acsr_router = router.get(PRIMARY_VARIANT, {})
    causal_router = router.get(CAUSAL_BASELINE, {})
    token_router = router.get("token_position_only_predicted_features", {})
    shuffled_router = router.get("shuffled_predicted_features", {})
    acsr_transfer = retention.get(("second_context_transfer", PRIMARY_VARIANT), {})
    token_transfer = retention.get(
        ("second_context_transfer", "token_position_only_predicted_features"), {}
    )
    shuffled_transfer = retention.get(
        ("second_context_transfer", "shuffled_predicted_features"), {}
    )
    acsr_teacher = retention.get(("fixed_context_teacher_reference", PRIMARY_VARIANT), {})
    token_teacher = retention.get(
        ("fixed_context_teacher_reference", "token_position_only_predicted_features"), {}
    )
    shuffled_teacher = retention.get(
        ("fixed_context_teacher_reference", "shuffled_predicted_features"), {}
    )

    row = {
        "packet": f"packet{index}",
        "audit_dir": str(audit_dir),
        "present": audit_dir.is_dir(),
        "status": summary.get("status"),
        "decision": summary.get("decision"),
        "required_files_present": all((audit_dir / name).is_file() for name in REQUIRED_PACKET_FILES),
        "future_perturbation_invariance": _perturbation_passed(summary, perturbation),
        "acsr_oracle_regret": _number(acsr_router.get("oracle_regret")),
        "causal_oracle_regret": _number(causal_router.get("oracle_regret")),
        "token_position_oracle_regret": _number(token_router.get("oracle_regret")),
        "shuffled_oracle_regret": _number(shuffled_router.get("oracle_regret")),
        "acsr_anchor_logit_mse_after_transfer": _number(
            acsr_transfer.get("anchor_logit_mse_after_transfer")
        ),
        "token_position_anchor_logit_mse_after_transfer": _number(
            token_transfer.get("anchor_logit_mse_after_transfer")
        ),
        "shuffled_anchor_logit_mse_after_transfer": _number(
            shuffled_transfer.get("anchor_logit_mse_after_transfer")
        ),
        "acsr_teacher_logit_mse": _number(acsr_teacher.get("teacher_logit_mse")),
        "token_position_teacher_logit_mse": _number(token_teacher.get("teacher_logit_mse")),
        "shuffled_teacher_logit_mse": _number(shuffled_teacher.get("teacher_logit_mse")),
    }
    row["acsr_minus_causal_oracle_regret"] = _delta(
        row["acsr_oracle_regret"], row["causal_oracle_regret"]
    )
    row["acsr_minus_token_position_oracle_regret"] = _delta(
        row["acsr_oracle_regret"], row["token_position_oracle_regret"]
    )
    row["acsr_minus_shuffled_oracle_regret"] = _delta(
        row["acsr_oracle_regret"], row["shuffled_oracle_regret"]
    )
    row["passes_oracle_regret_gate"] = all(
        value is not None and value <= 0.0
        for value in (
            row["acsr_minus_causal_oracle_regret"],
            row["acsr_minus_token_position_oracle_regret"],
            row["acsr_minus_shuffled_oracle_regret"],
        )
    )
    row["passes_functional_churn_gate"] = all(
        _le(row[left], row[right])
        for left, right in (
            (
                "acsr_anchor_logit_mse_after_transfer",
                "token_position_anchor_logit_mse_after_transfer",
            ),
            (
                "acsr_anchor_logit_mse_after_transfer",
                "shuffled_anchor_logit_mse_after_transfer",
            ),
            ("acsr_teacher_logit_mse", "token_position_teacher_logit_mse"),
            ("acsr_teacher_logit_mse", "shuffled_teacher_logit_mse"),
        )
    )
    return row


def _criteria(packet_rows: list[dict[str, Any]], previous_probe_path: Path) -> list[dict[str, Any]]:
    present_packets = [row for row in packet_rows if row["present"]]
    previous_probe = _previous_probe(previous_probe_path)
    return [
        {
            "criterion": "all_requested_packets_present",
            "passed": len(present_packets) == len(packet_rows) and bool(packet_rows),
            "threshold": f"{len(packet_rows)} packets",
            "actual": str(len(present_packets)),
        },
        {
            "criterion": "all_packet_smoke_gates_pass",
            "passed": all(row["status"] == "pass" for row in packet_rows),
            "threshold": "status == pass for every packet",
            "actual": ",".join(str(row["status"]) for row in packet_rows),
        },
        {
            "criterion": "future_perturbation_invariance_passes",
            "passed": all(bool(row["future_perturbation_invariance"]) for row in packet_rows),
            "threshold": "true for every packet",
            "actual": ",".join(str(row["future_perturbation_invariance"]) for row in packet_rows),
        },
        {
            "criterion": "acsr_oracle_regret_not_worse_than_causal_and_nulls",
            "passed": all(bool(row["passes_oracle_regret_gate"]) for row in packet_rows),
            "threshold": "acsr - control <= 0 for causal, token-position, shuffled",
            "actual": ",".join(str(row["passes_oracle_regret_gate"]) for row in packet_rows),
        },
        {
            "criterion": "acsr_functional_churn_not_worse_than_nulls",
            "passed": all(bool(row["passes_functional_churn_gate"]) for row in packet_rows),
            "threshold": "anchor and teacher logit MSE <= token-position and shuffled",
            "actual": ",".join(str(row["passes_functional_churn_gate"]) for row in packet_rows),
        },
        {
            "criterion": "previous_same_student_retention_churn_gate_passed",
            "passed": bool(
                previous_probe.get("gate_status", {}).get(
                    "passes_cross_context_retention_churn_gate"
                )
            ),
            "threshold": "previous retention/churn probe passed",
            "actual": str(previous_probe.get("decision")),
        },
    ]


def _failures(
    packet_rows: list[dict[str, Any]], criteria: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for criterion in criteria:
        if not criterion["passed"]:
            failures.append(
                {
                    "field": criterion["criterion"],
                    "reason": "gate criterion failed",
                    "actual": criterion["actual"],
                    "threshold": criterion["threshold"],
                }
            )
    for row in packet_rows:
        if not row["required_files_present"]:
            failures.append(
                {
                    "packet": row["packet"],
                    "field": "required_files_present",
                    "reason": "packet is missing required ACSR artifacts",
                }
            )
    return failures


def _aggregate_rows(packet_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for field in (
        "acsr_minus_causal_oracle_regret",
        "acsr_minus_token_position_oracle_regret",
        "acsr_minus_shuffled_oracle_regret",
        "acsr_anchor_logit_mse_after_transfer",
        "acsr_teacher_logit_mse",
    ):
        values = [row[field] for row in packet_rows if row.get(field) is not None]
        rows.append(
            {
                "metric": field,
                "packet_count": len(values),
                "mean": mean(values) if values else "",
                "maximum": max(values) if values else "",
                "all_pass": all(value <= 0.0 for value in values)
                if field.startswith("acsr_minus_")
                else bool(values),
            }
        )
    return rows


def _public_packet_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _previous_probe(path: Path) -> dict[str, Any]:
    return _read_json_object(path)


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "present": False,
            "path": str(path),
            "strategic_change_level": None,
            "notify_ben": None,
            "recommendation_status": "not_available",
        }
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    return {
        "present": True,
        "path": str(path),
        "strategic_change_level": header.get("strategic_change_level"),
        "notify_ben": header.get("notify_ben"),
        "recommended_next_action": header.get("recommended_next_action"),
        "recommendation_status": "accepted_as_acsr_local_cpu_gate_continuation",
    }


def _deferred_recommendations(strategy_review: dict[str, Any]) -> list[dict[str, Any]]:
    if not strategy_review.get("present"):
        return []
    return [
        {
            "recommendation": strategy_review.get("recommended_next_action"),
            "status": "accepted_already_satisfied_and_extended",
            "reason": (
                "The local ACSR smoke and retention/churn gates already exist; "
                "this run extends the sensible recommendation with the named "
                "oracle-regret and functional-churn gate before any default "
                "promotion."
            ),
        }
    ]


def _perturbation_passed(summary: dict[str, Any], rows: list[dict[str, str]]) -> bool:
    gate_value = summary.get("gates", {}).get("future_perturbation_invariance")
    if gate_value is not None:
        return bool(gate_value)
    if not rows:
        return False
    return all(str(row.get("passed")).lower() == "true" for row in rows)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ACSR Causal Mechanism Gate",
        "",
        f"Status: {summary['status']}",
        f"Decision: {summary['decision']}",
        f"Claim status: {summary['claim_status']}",
        "",
        summary["rationale"],
        "",
        "This is a no-default-promotion gate. A broader benchmark is still "
        "required before changing the promoted router policy.",
        "",
        f"Next step: {summary['selected_next_step']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _le(left: float | None, right: float | None) -> bool:
    return left is not None and right is not None and left <= right


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir",
        action="append",
        type=Path,
        dest="audit_dirs",
        help="ACSR audit directory to consume; may be passed multiple times.",
    )
    parser.add_argument(
        "--previous-probe",
        type=Path,
        default=DEFAULT_PREVIOUS_PROBE,
        help="Existing same-student retention/churn probe summary.",
    )
    parser.add_argument(
        "--strategy-review",
        type=Path,
        default=DEFAULT_STRATEGY_REVIEW,
        help="External strategic review to record.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    audit_dirs = tuple(args.audit_dirs) if args.audit_dirs else DEFAULT_AUDIT_DIRS
    summary = run_acsr_causal_mechanism_gate(
        audit_dirs=audit_dirs,
        previous_probe_path=args.previous_probe,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, indent=2))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
