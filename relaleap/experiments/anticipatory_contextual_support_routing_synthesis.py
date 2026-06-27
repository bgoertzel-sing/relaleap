"""Two-seed synthesis gate for anticipatory contextual support routing."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_AUDIT_DIRS = (
    Path("results/audits/token_larger_anticipatory_contextual_support_routing"),
    Path("results/audits/token_larger_anticipatory_contextual_support_routing_seed2"),
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_anticipatory_contextual_support_routing_synthesis"
)

ACSR_LOCAL_SYNTHESIS_RECORDED = "acsr_two_seed_local_synthesis_recorded"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
RUNPOD_REPLICATION_WARRANTED = "runpod_replication_warranted"
LOCAL_REPAIR_REQUIRED = "local_repair_required"

REQUIRED_GATES = (
    "future_perturbation_invariance",
    "acsr_beats_shuffled_ce",
    "acsr_beats_token_position_ce",
    "acsr_does_not_worsen_causal_regret",
)
REQUIRED_FILES = (
    "summary.json",
    "predictor_metrics.csv",
    "router_metrics.csv",
    "same_student_metrics.csv",
    "feature_perturbation.csv",
    "retention_churn_metrics.csv",
    "notes.md",
)


def run_anticipatory_contextual_support_routing_synthesis(
    *,
    audit_dirs: tuple[Path, ...] = DEFAULT_AUDIT_DIRS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Aggregate completed local ACSR smoke packets into a replication gate."""

    start = time.time()
    packet_rows = [_packet_row(index, path) for index, path in enumerate(audit_dirs, 1)]
    failures = [failure for row in packet_rows for failure in _packet_failures(row)]
    aggregates = _aggregate_packet_rows(packet_rows)
    strategy_review = _strategy_review(strategy_review_path)
    backend = os.environ.get("RELALEAP_GPU_BACKEND", "unset")

    consistent = len(packet_rows) >= 2 and not failures and aggregates[
        "all_required_gates_pass"
    ]
    if consistent:
        status = "pass"
        decision = ACSR_LOCAL_SYNTHESIS_RECORDED
        claim_status = "local_acsr_controls_consistently_discriminative"
        replication_gate = RUNPOD_REPLICATION_WARRANTED
        if backend == "runpod":
            next_step = (
                "run RunPod ACSR replication for the two token-larger seed configs, "
                "then fetch artifacts and run the same local synthesis/checks"
            )
            next_command = (
                "./.venv-conda/bin/python tools/runpod_ssh_runner.py bootstrap && "
                "./.venv-conda/bin/python tools/runpod_ssh_runner.py run --command "
                "'python -m relaleap.experiments.anticipatory_contextual_support_routing "
                "--config configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml "
                "--out results/audits/runpod_token_larger_anticipatory_contextual_support_routing && "
                "python -m relaleap.experiments.anticipatory_contextual_support_routing "
                "--config configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate_seed2.yaml "
                "--out results/audits/runpod_token_larger_anticipatory_contextual_support_routing_seed2' && "
                "./.venv-conda/bin/python tools/runpod_ssh_runner.py fetch"
            )
        else:
            next_step = (
                "replicate ACSR on the configured GPU backend before making any "
                "default-router or causal-mechanism claim"
            )
            next_command = None
        rationale = (
            "Both local token-larger ACSR smoke packets pass the fail-closed "
            "leakage, shuffled-feature, token/position-only, and regret gates. "
            "ACSR closes the causal-router to full-context teacher CE gap in both "
            "packets, beats the null supports through same-student values, and "
            "keeps fixed-teacher churn well below shuffled and token/position "
            "controls. This warrants backend replication, not promotion."
        )
    else:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        claim_status = "local_acsr_synthesis_not_interpretable"
        replication_gate = LOCAL_REPAIR_REQUIRED
        next_step = "repair missing or failing local ACSR smoke artifacts"
        next_command = None
        rationale = (
            "The local ACSR packets are missing, failed, or not consistently "
            "discriminative enough to justify backend replication."
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "replication_gate": replication_gate,
        "selected_next_step": next_step,
        "next_command": next_command,
        "gpu_backend": backend,
        "packet_count": len(packet_rows),
        "required_gates": list(REQUIRED_GATES),
        "packet_rows": packet_rows,
        "aggregates": aggregates,
        "strategy_review": strategy_review,
        "failures": failures,
        "rationale": rationale,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "packet_metrics_csv": str(out_dir / "packet_metrics.csv"),
            "gate_metrics_csv": str(out_dir / "gate_metrics.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_packet_metrics(out_dir / "packet_metrics.csv", packet_rows)
    _write_gate_metrics(out_dir / "gate_metrics.csv", packet_rows)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _packet_row(index: int, audit_dir: Path) -> dict[str, Any]:
    summary_path = audit_dir / "summary.json"
    summary = _read_json_object(summary_path)
    router_rows = _read_csv_by_key(audit_dir / "router_metrics.csv", "variant")
    same_student_rows = _read_csv_rows(audit_dir / "same_student_metrics.csv")
    retention_rows = _read_csv_rows(audit_dir / "retention_churn_metrics.csv")
    perturbation_rows = _read_csv_rows(audit_dir / "feature_perturbation.csv")
    predictor_rows = _read_csv_by_key(audit_dir / "predictor_metrics.csv", "predictor")

    acsr = router_rows.get("acsr_mlp_predicted_future", {})
    shuffled = router_rows.get("shuffled_predicted_features", {})
    token_position = router_rows.get("token_position_only_predicted_features", {})
    causal = router_rows.get("causal_feature_safe_contextual_topk2", {})
    teacher = router_rows.get("full_context_contextual_topk2_teacher", {})
    mlp_predictor = predictor_rows.get("mlp_causal", {})
    token_position_predictor = predictor_rows.get("token_position_only", {})

    same_student_token = _same_student_delta(
        same_student_rows,
        "acsr_mlp_predicted_future_support_vs_token_position_only_predicted_features",
    )
    same_student_shuffled = _same_student_delta(
        same_student_rows,
        "acsr_mlp_predicted_future_support_vs_shuffled_predicted_features",
    )
    acsr_retention = _retention_row(retention_rows, "second_context_transfer", "acsr_mlp_predicted_future")
    shuffled_retention = _retention_row(retention_rows, "second_context_transfer", "shuffled_predicted_features")
    token_retention = _retention_row(retention_rows, "second_context_transfer", "token_position_only_predicted_features")
    acsr_teacher = _retention_row(retention_rows, "fixed_context_teacher_reference", "acsr_mlp_predicted_future")
    shuffled_teacher = _retention_row(retention_rows, "fixed_context_teacher_reference", "shuffled_predicted_features")
    token_teacher = _retention_row(retention_rows, "fixed_context_teacher_reference", "token_position_only_predicted_features")

    row: dict[str, Any] = {
        "packet": f"seed{index}",
        "audit_dir": str(audit_dir),
        "summary_path": str(summary_path),
        "summary_present": summary_path.is_file(),
        "status": summary.get("status"),
        "decision": summary.get("decision"),
        "config_path": summary.get("config_path"),
        "train_steps": summary.get("train_steps"),
        "predictor_steps": summary.get("predictor_steps"),
        "all_required_files_present": all((audit_dir / name).is_file() for name in REQUIRED_FILES),
        "future_perturbation_rows_pass": all(_bool(row.get("passed")) for row in perturbation_rows),
        "mlp_predictor_r2": _float_or_none(mlp_predictor.get("holdout_r2")),
        "token_position_predictor_r2": _float_or_none(
            token_position_predictor.get("holdout_r2")
        ),
        "acsr_ce_loss": _float_or_none(acsr.get("ce_loss")),
        "causal_ce_loss": _float_or_none(causal.get("ce_loss")),
        "teacher_ce_loss": _float_or_none(teacher.get("ce_loss")),
        "shuffled_ce_loss": _float_or_none(shuffled.get("ce_loss")),
        "token_position_ce_loss": _float_or_none(token_position.get("ce_loss")),
        "acsr_oracle_regret": _float_or_none(acsr.get("oracle_regret")),
        "causal_oracle_regret": _float_or_none(causal.get("oracle_regret")),
        "shuffled_oracle_regret": _float_or_none(shuffled.get("oracle_regret")),
        "token_position_oracle_regret": _float_or_none(token_position.get("oracle_regret")),
        "same_student_delta_vs_token_position": same_student_token,
        "same_student_delta_vs_shuffled": same_student_shuffled,
        "acsr_anchor_support_churn_after_transfer": _float_or_none(
            acsr_retention.get("anchor_support_churn_after_transfer")
        ),
        "shuffled_anchor_support_churn_after_transfer": _float_or_none(
            shuffled_retention.get("anchor_support_churn_after_transfer")
        ),
        "token_position_anchor_support_churn_after_transfer": _float_or_none(
            token_retention.get("anchor_support_churn_after_transfer")
        ),
        "acsr_anchor_logit_mse_after_transfer": _float_or_none(
            acsr_retention.get("anchor_logit_mse_after_transfer")
        ),
        "shuffled_anchor_logit_mse_after_transfer": _float_or_none(
            shuffled_retention.get("anchor_logit_mse_after_transfer")
        ),
        "token_position_anchor_logit_mse_after_transfer": _float_or_none(
            token_retention.get("anchor_logit_mse_after_transfer")
        ),
        "acsr_teacher_support_churn": _float_or_none(acsr_teacher.get("teacher_support_churn")),
        "shuffled_teacher_support_churn": _float_or_none(
            shuffled_teacher.get("teacher_support_churn")
        ),
        "token_position_teacher_support_churn": _float_or_none(
            token_teacher.get("teacher_support_churn")
        ),
        "summary_failures": summary.get("failures", []),
    }
    for gate in REQUIRED_GATES:
        row[gate] = bool(summary.get("gates", {}).get(gate))
    row["acsr_minus_causal_ce_loss"] = _delta(row["acsr_ce_loss"], row["causal_ce_loss"])
    row["acsr_minus_teacher_ce_loss"] = _delta(row["acsr_ce_loss"], row["teacher_ce_loss"])
    row["acsr_minus_token_position_ce_loss"] = _delta(
        row["acsr_ce_loss"], row["token_position_ce_loss"]
    )
    row["acsr_minus_shuffled_ce_loss"] = _delta(row["acsr_ce_loss"], row["shuffled_ce_loss"])
    row["acsr_minus_causal_regret"] = _delta(
        row["acsr_oracle_regret"], row["causal_oracle_regret"]
    )
    row["acsr_teacher_churn_advantage_vs_token_position"] = _delta(
        row["token_position_teacher_support_churn"], row["acsr_teacher_support_churn"]
    )
    row["acsr_teacher_churn_advantage_vs_shuffled"] = _delta(
        row["shuffled_teacher_support_churn"], row["acsr_teacher_support_churn"]
    )
    return row


def _packet_failures(row: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    if not row["summary_present"]:
        return [
            {
                "packet": row["packet"],
                "field": "summary_json",
                "expected": "file exists",
                "actual": "missing",
                "path": row["summary_path"],
            }
        ]
    if row["status"] != "pass":
        failures.append({"packet": row["packet"], "field": "status", "expected": "pass", "actual": row["status"]})
    if row["decision"] != "anticipatory_contextual_support_routing_smoke_completed":
        failures.append(
            {
                "packet": row["packet"],
                "field": "decision",
                "expected": "anticipatory_contextual_support_routing_smoke_completed",
                "actual": row["decision"],
            }
        )
    if not row["all_required_files_present"]:
        failures.append(
            {
                "packet": row["packet"],
                "field": "required_artifacts",
                "expected": list(REQUIRED_FILES),
                "actual": "one_or_more_missing",
            }
        )
    if not row["future_perturbation_rows_pass"]:
        failures.append(
            {
                "packet": row["packet"],
                "field": "feature_perturbation_csv.passed",
                "expected": True,
                "actual": False,
            }
        )
    for gate in REQUIRED_GATES:
        if not row[gate]:
            failures.append(
                {
                    "packet": row["packet"],
                    "field": f"gates.{gate}",
                    "expected": True,
                    "actual": row[gate],
                }
            )
    for metric in (
        "acsr_ce_loss",
        "causal_ce_loss",
        "teacher_ce_loss",
        "shuffled_ce_loss",
        "token_position_ce_loss",
        "acsr_oracle_regret",
        "same_student_delta_vs_token_position",
        "same_student_delta_vs_shuffled",
        "acsr_teacher_support_churn",
        "shuffled_teacher_support_churn",
        "token_position_teacher_support_churn",
    ):
        if row.get(metric) is None:
            failures.append(
                {
                    "packet": row["packet"],
                    "field": metric,
                    "expected": "numeric metric",
                    "actual": None,
                }
            )
    return failures


def _aggregate_packet_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all_packets_pass": all(row.get("status") == "pass" for row in rows),
        "all_required_artifacts_present": all(row.get("all_required_files_present") for row in rows),
        "all_required_gates_pass": all(
            all(row.get(gate) is True for gate in REQUIRED_GATES) for row in rows
        ),
        "all_perturbation_rows_pass": all(
            row.get("future_perturbation_rows_pass") for row in rows
        ),
        "all_acsr_ce_beats_causal": all(
            _lt(row.get("acsr_ce_loss"), row.get("causal_ce_loss")) for row in rows
        ),
        "all_acsr_ce_beats_token_position": all(
            _lt(row.get("acsr_ce_loss"), row.get("token_position_ce_loss"))
            for row in rows
        ),
        "all_acsr_ce_beats_shuffled": all(
            _lt(row.get("acsr_ce_loss"), row.get("shuffled_ce_loss")) for row in rows
        ),
        "all_acsr_regret_not_worse_than_causal": all(
            _le(row.get("acsr_oracle_regret"), row.get("causal_oracle_regret"))
            for row in rows
        ),
        "all_same_student_beats_token_position": all(
            _lt(row.get("same_student_delta_vs_token_position"), 0.0) for row in rows
        ),
        "all_same_student_beats_shuffled": all(
            _lt(row.get("same_student_delta_vs_shuffled"), 0.0) for row in rows
        ),
        "all_teacher_churn_below_token_position": all(
            _lt(row.get("acsr_teacher_support_churn"), row.get("token_position_teacher_support_churn"))
            for row in rows
        ),
        "all_teacher_churn_below_shuffled": all(
            _lt(row.get("acsr_teacher_support_churn"), row.get("shuffled_teacher_support_churn"))
            for row in rows
        ),
        "mean_acsr_minus_causal_ce_loss": _mean_field(rows, "acsr_minus_causal_ce_loss"),
        "mean_acsr_minus_teacher_ce_loss": _mean_field(rows, "acsr_minus_teacher_ce_loss"),
        "mean_acsr_minus_token_position_ce_loss": _mean_field(
            rows, "acsr_minus_token_position_ce_loss"
        ),
        "mean_acsr_minus_shuffled_ce_loss": _mean_field(rows, "acsr_minus_shuffled_ce_loss"),
        "mean_acsr_minus_causal_regret": _mean_field(rows, "acsr_minus_causal_regret"),
        "mean_mlp_predictor_r2": _mean_field(rows, "mlp_predictor_r2"),
        "mean_token_position_predictor_r2": _mean_field(
            rows, "token_position_predictor_r2"
        ),
        "mean_acsr_teacher_support_churn": _mean_field(rows, "acsr_teacher_support_churn"),
        "mean_token_position_teacher_support_churn": _mean_field(
            rows, "token_position_teacher_support_churn"
        ),
        "mean_shuffled_teacher_support_churn": _mean_field(
            rows, "shuffled_teacher_support_churn"
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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_csv_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    return {str(row.get(key, "")): row for row in _read_csv_rows(path) if row.get(key)}


def _retention_row(rows: list[dict[str, str]], phase: str, variant: str) -> dict[str, str]:
    for row in rows:
        if row.get("phase") == phase and row.get("variant") == variant:
            return row
    return {}


def _same_student_delta(rows: list[dict[str, str]], comparison: str) -> float | None:
    for row in rows:
        if row.get("comparison") == comparison:
            return _float_or_none(row.get("acsr_minus_control_ce_loss"))
    return None


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "present": False,
            "strategic_change_level": None,
            "notify_ben": None,
            "ben_notification_required": False,
            "recommended_next_action": None,
        }
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"strategic_change_level", "notify_ben", "recommended_next_action"}:
            values[key] = value.strip()
    notify_ben = values.get("notify_ben", "").lower() == "true"
    strategic_change_level = values.get("strategic_change_level")
    return {
        "present": True,
        "strategic_change_level": strategic_change_level,
        "notify_ben": notify_ben,
        "ben_notification_required": notify_ben or strategic_change_level == "major",
        "recommended_next_action": values.get("recommended_next_action"),
    }


def _write_packet_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "packet",
        "audit_dir",
        "status",
        "decision",
        "config_path",
        "mlp_predictor_r2",
        "token_position_predictor_r2",
        "acsr_ce_loss",
        "causal_ce_loss",
        "teacher_ce_loss",
        "shuffled_ce_loss",
        "token_position_ce_loss",
        "acsr_minus_causal_ce_loss",
        "acsr_minus_teacher_ce_loss",
        "acsr_minus_token_position_ce_loss",
        "acsr_minus_shuffled_ce_loss",
        "acsr_oracle_regret",
        "causal_oracle_regret",
        "acsr_minus_causal_regret",
        "same_student_delta_vs_token_position",
        "same_student_delta_vs_shuffled",
        "acsr_anchor_support_churn_after_transfer",
        "shuffled_anchor_support_churn_after_transfer",
        "token_position_anchor_support_churn_after_transfer",
        "acsr_anchor_logit_mse_after_transfer",
        "shuffled_anchor_logit_mse_after_transfer",
        "token_position_anchor_logit_mse_after_transfer",
        "acsr_teacher_support_churn",
        "shuffled_teacher_support_churn",
        "token_position_teacher_support_churn",
    ]
    _write_csv(path, fields, rows)


def _write_gate_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "packet",
        "all_required_files_present",
        "future_perturbation_rows_pass",
        *REQUIRED_GATES,
    ]
    _write_csv(path, fields, rows)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ACSR Two-Seed Local Synthesis",
        "",
        f"Status: {summary['status']}",
        f"Decision: {summary['decision']}",
        f"Replication gate: {summary['replication_gate']}",
        f"GPU backend: {summary['gpu_backend']}",
        "",
        "## Rationale",
        "",
        summary["rationale"],
        "",
        "## Next Step",
        "",
        summary["selected_next_step"],
        "",
    ]
    if summary["next_command"]:
        lines.extend(["## Command", "", f"`{summary['next_command']}`", ""])
    if summary["strategy_review"]["ben_notification_required"]:
        lines.extend(
            [
                "## Ben Notification",
                "",
                "The latest strategy review requests Ben notification.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def _delta(left: Any, right: Any) -> float | None:
    left_float = _float_or_none(left)
    right_float = _float_or_none(right)
    if left_float is None or right_float is None:
        return None
    return left_float - right_float


def _lt(left: Any, right: Any) -> bool:
    left_float = _float_or_none(left)
    right_float = _float_or_none(right)
    return left_float is not None and right_float is not None and left_float < right_float


def _le(left: Any, right: Any) -> bool:
    left_float = _float_or_none(left)
    right_float = _float_or_none(right)
    return left_float is not None and right_float is not None and left_float <= right_float


def _mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [
        value
        for value in (_float_or_none(row.get(field)) for row in rows)
        if value is not None
    ]
    return mean(values) if values else None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir",
        action="append",
        type=Path,
        dest="audit_dirs",
        help="Completed ACSR audit directory. Repeat for multiple seeds.",
    )
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_anticipatory_contextual_support_routing_synthesis(
        audit_dirs=tuple(args.audit_dirs) if args.audit_dirs else DEFAULT_AUDIT_DIRS,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
