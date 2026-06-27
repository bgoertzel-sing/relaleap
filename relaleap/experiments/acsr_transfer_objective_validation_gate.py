"""Local-plus-RunPod validation gate for the ACSR transfer-objective probe."""

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


DEFAULT_SOURCE_DIRS = (
    Path("results/audits/acsr_transfer_objective_probe"),
    Path("results/audits/acsr_transfer_objective_probe_seed2"),
    Path("results/runpod_fetch/audits/runpod_acsr_transfer_objective_probe"),
    Path("results/runpod_fetch/audits/runpod_acsr_transfer_objective_probe_seed2"),
)
DEFAULT_OUT_DIR = Path("results/reports/acsr_transfer_objective_validation_gate")
REQUIRED_SOURCE_ARTIFACTS = (
    "summary.json",
    "metrics.csv",
    "arm_metrics.csv",
    "per_token_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)
REQUIRED_OUTPUT_ARTIFACTS = (
    "summary.json",
    "source_packets.csv",
    "primary_metrics_by_packet.csv",
    "aggregate_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)
EXPECTED_BACKENDS = ("local", "runpod")
EXPECTED_SEEDS = ("seed1", "seed2")
OWN_CE_GUARDRAIL = 0.02


def run_acsr_transfer_objective_validation_gate(
    *,
    source_dirs: tuple[Path, ...] = DEFAULT_SOURCE_DIRS,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Synthesize local and RunPod transfer-objective probe summaries."""

    start = time.time()
    packets = [_load_packet(path) for path in source_dirs]
    packet_rows = [_source_packet_row(packet) for packet in packets]
    metric_rows = _primary_metric_rows(packets)
    aggregate = _aggregate_metrics(metric_rows)
    gate_rows = _gate_rows(packet_rows, metric_rows, aggregate)
    failures = [
        {"gate": row["criterion"], "reason": row["failure_reason"]}
        for row in gate_rows
        if not row["passed"]
    ]
    status = "pass" if not failures else "fail"
    summary = {
        "status": status,
        "decision": (
            "acsr_transfer_objective_validation_gate_passed"
            if status == "pass"
            else "acsr_transfer_objective_validation_gate_failed_closed"
        ),
        "claim_status": (
            "cross_backend_transfer_objective_supported_not_promoted"
            if status == "pass"
            else "cross_backend_transfer_objective_not_supported"
        ),
        "selected_next_step": (
            "continue to stronger held-out ACSR transfer controls before any mechanism or default-router claim"
            if status == "pass"
            else "stop the ACSR transfer-objective branch and inspect failed guardrails"
        ),
        "source_dirs": [str(path) for path in source_dirs],
        "source_packets": packet_rows,
        "primary_metrics_by_packet": metric_rows,
        "aggregate_metrics": aggregate,
        "gate_criteria": gate_rows,
        "failures": failures,
        "claim_boundaries": {
            "supported": [
                "the low-step margin-aware transfer objective passed two local and two RunPod seed/backend probe packets",
                "partner-through-values CE and oracle regret improved against direct and required null arms in the available packets",
                "own-value CE damage stayed within the current guardrail in the available packets",
            ],
            "not_supported": [
                "ACSR-as-anticipation",
                "default ACSR or support-router promotion",
                "a mechanism claim before stronger held-out, residual-norm, and rank/FLOP-matched controls",
            ],
        },
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            name.replace(".", "_"): str(out_dir / name)
            for name in REQUIRED_OUTPUT_ARTIFACTS
        },
    }
    _write_artifacts(out_dir, summary)
    return summary


def _load_packet(source_dir: Path) -> dict[str, Any]:
    missing = [
        name for name in REQUIRED_SOURCE_ARTIFACTS if not (source_dir / name).is_file()
    ]
    summary = _read_json(source_dir / "summary.json")
    gate_rows = _read_csv(source_dir / "gate_criteria.csv")
    return {
        "source_dir": str(source_dir),
        "backend": _backend_label(source_dir),
        "seed": _seed_label(source_dir, summary),
        "missing_artifacts": missing,
        "summary": summary,
        "gate_rows": gate_rows,
    }


def _source_packet_row(packet: dict[str, Any]) -> dict[str, Any]:
    summary = packet["summary"]
    individual_failures = [
        row.get("criterion", "unknown")
        for row in packet["gate_rows"]
        if not _bool_value(row.get("passed"))
    ]
    return {
        "source_dir": packet["source_dir"],
        "backend": packet["backend"],
        "seed": packet["seed"],
        "loaded": not packet["missing_artifacts"] and bool(summary),
        "status": summary.get("status", "missing"),
        "decision": summary.get("decision", ""),
        "claim_status": summary.get("claim_status", ""),
        "missing_artifacts": ";".join(packet["missing_artifacts"]),
        "gate_row_count": len(packet["gate_rows"]),
        "individual_gate_failures": ";".join(individual_failures),
        "git_commit": summary.get("git_commit", ""),
        "platform": summary.get("platform", ""),
    }


def _primary_metric_rows(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for packet in packets:
        summary = packet["summary"]
        metrics = summary.get("primary_metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        row = {
            "source_dir": packet["source_dir"],
            "backend": packet["backend"],
            "seed": packet["seed"],
            "status": summary.get("status", "missing"),
            "partner_transfer_minus_direct_ce": _number(
                metrics.get("partner_transfer_minus_direct_ce")
            ),
            "partner_transfer_minus_token_position_ce": _number(
                metrics.get("partner_transfer_minus_token_position_ce")
            ),
            "partner_transfer_minus_random_ce": _number(
                metrics.get("partner_transfer_minus_random_ce")
            ),
            "own_transfer_minus_direct_ce": _number(
                metrics.get("own_transfer_minus_direct_ce")
            ),
            "partner_transfer_oracle_regret": _number(
                metrics.get("partner_transfer_oracle_regret")
            ),
            "partner_direct_oracle_regret": _number(
                metrics.get("partner_direct_oracle_regret")
            ),
            "partner_transfer_residual_norm_normalized_delta_vs_direct": _number(
                metrics.get("partner_transfer_residual_norm_normalized_delta_vs_direct")
            ),
            "transfer_support_jaccard_with_direct": _number(
                metrics.get("transfer_support_jaccard_with_direct")
            ),
        }
        row["partner_oracle_regret_delta_vs_direct"] = _delta(
            row["partner_transfer_oracle_regret"],
            row["partner_direct_oracle_regret"],
        )
        rows.append(row)
    return rows


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "packet_count": len(rows),
        "loaded_backend_seed_cells": sorted(
            {
                f"{row.get('backend')}:{row.get('seed')}"
                for row in rows
                if row.get("status") == "pass"
            }
        ),
        "mean_partner_transfer_minus_direct_ce": _mean_key(
            rows, "partner_transfer_minus_direct_ce"
        ),
        "mean_partner_transfer_minus_token_position_ce": _mean_key(
            rows, "partner_transfer_minus_token_position_ce"
        ),
        "mean_partner_transfer_minus_random_ce": _mean_key(
            rows, "partner_transfer_minus_random_ce"
        ),
        "mean_own_transfer_minus_direct_ce": _mean_key(
            rows, "own_transfer_minus_direct_ce"
        ),
        "max_own_transfer_minus_direct_ce": _max_key(
            rows, "own_transfer_minus_direct_ce"
        ),
        "mean_partner_oracle_regret_delta_vs_direct": _mean_key(
            rows, "partner_oracle_regret_delta_vs_direct"
        ),
        "mean_residual_norm_normalized_delta_vs_direct": _mean_key(
            rows, "partner_transfer_residual_norm_normalized_delta_vs_direct"
        ),
        "min_transfer_support_jaccard_with_direct": _min_key(
            rows, "transfer_support_jaccard_with_direct"
        ),
        "all_partner_beats_direct": _all_lt(rows, "partner_transfer_minus_direct_ce", 0.0),
        "all_partner_beats_token_position_null": _all_lt(
            rows, "partner_transfer_minus_token_position_ce", 0.0
        ),
        "all_partner_beats_random_null": _all_lt(
            rows, "partner_transfer_minus_random_ce", 0.0
        ),
        "all_partner_oracle_regret_improves": _all_lt(
            rows, "partner_oracle_regret_delta_vs_direct", 0.0
        ),
        "all_own_ce_within_guardrail": _all_le(
            rows, "own_transfer_minus_direct_ce", OWN_CE_GUARDRAIL
        ),
        "all_residual_norm_normalized_metrics_present": all(
            isinstance(row.get("partner_transfer_residual_norm_normalized_delta_vs_direct"), float)
            for row in rows
        ),
    }


def _gate_rows(
    packet_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_cells = {f"{backend}:{seed}" for backend in EXPECTED_BACKENDS for seed in EXPECTED_SEEDS}
    loaded_cells = set(aggregate["loaded_backend_seed_cells"])
    missing_cells = sorted(expected_cells - loaded_cells)
    return [
        _criterion(
            "required_source_artifacts_present",
            all(row["loaded"] for row in packet_rows),
            "all four source packets expose summary, metrics, arm, per-token, gate, and notes artifacts",
            ";".join(row["source_dir"] for row in packet_rows if not row["loaded"]) or "all_present",
            "one or more source packet artifacts are missing",
        ),
        _criterion(
            "source_status_pass",
            all(row["status"] == "pass" for row in packet_rows),
            "all source summaries report status pass",
            ";".join(f"{row['backend']}:{row['seed']}={row['status']}" for row in packet_rows),
            "one or more source summaries did not pass",
        ),
        _criterion(
            "individual_probe_gates_passed",
            all(not row["individual_gate_failures"] for row in packet_rows),
            "all individual source gate rows pass",
            ";".join(row["individual_gate_failures"] for row in packet_rows if row["individual_gate_failures"]) or "all_passed",
            "one or more source probe gate criteria failed",
        ),
        _criterion(
            "backend_seed_coverage_complete",
            not missing_cells,
            "local and RunPod seed1/seed2 packets are present",
            ";".join(sorted(loaded_cells)),
            f"missing backend/seed cells: {';'.join(missing_cells)}",
        ),
        _criterion(
            "partner_transfer_beats_direct_all_packets",
            aggregate["all_partner_beats_direct"],
            "transfer objective improves partner-through-values CE versus direct causal MLP in every packet",
            aggregate["mean_partner_transfer_minus_direct_ce"],
            "transfer objective did not beat direct causal MLP in every packet",
        ),
        _criterion(
            "partner_transfer_beats_token_position_null_all_packets",
            aggregate["all_partner_beats_token_position_null"],
            "transfer objective improves partner-through-values CE versus token-position trained null in every packet",
            aggregate["mean_partner_transfer_minus_token_position_ce"],
            "transfer objective did not beat token-position null in every packet",
        ),
        _criterion(
            "partner_transfer_beats_random_null_all_packets",
            aggregate["all_partner_beats_random_null"],
            "transfer objective improves partner-through-values CE versus frequency-random support null in every packet",
            aggregate["mean_partner_transfer_minus_random_ce"],
            "transfer objective did not beat random-frequency support null in every packet",
        ),
        _criterion(
            "partner_oracle_regret_improves_all_packets",
            aggregate["all_partner_oracle_regret_improves"],
            "partner oracle regret is lower for transfer objective than direct causal MLP in every packet",
            aggregate["mean_partner_oracle_regret_delta_vs_direct"],
            "partner oracle regret did not improve in every packet",
        ),
        _criterion(
            "own_ce_guardrail_all_packets",
            aggregate["all_own_ce_within_guardrail"],
            f"own-value CE worsens by no more than {OWN_CE_GUARDRAIL} versus direct causal MLP in every packet",
            aggregate["max_own_transfer_minus_direct_ce"],
            "own-value CE damage exceeded guardrail",
        ),
        _criterion(
            "residual_norm_normalized_metrics_present",
            aggregate["all_residual_norm_normalized_metrics_present"],
            "residual-norm-normalized transfer metric is present in every packet",
            aggregate["mean_residual_norm_normalized_delta_vs_direct"],
            "residual-norm-normalized metric is missing in one or more packets",
        ),
    ]


def _backend_label(source_dir: Path) -> str:
    return "runpod" if "runpod" in str(source_dir).lower() else "local"


def _seed_label(source_dir: Path, summary: dict[str, Any]) -> str:
    text = f"{source_dir} {summary.get('config_path', '')}".lower()
    if "seed2" in text or "seed_2" in text:
        return "seed2"
    return "seed1"


def _criterion(
    criterion: str,
    passed: bool,
    threshold: str,
    actual: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": actual,
        "failure_reason": "" if passed else failure_reason,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_packets.csv", summary["source_packets"])
    _write_csv(
        out_dir / "primary_metrics_by_packet.csv",
        summary["primary_metrics_by_packet"],
    )
    _write_csv(
        out_dir / "aggregate_metrics.csv",
        [{"metric": key, "value": value} for key, value in summary["aggregate_metrics"].items()],
    )
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [{"status": "missing"}]
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["aggregate_metrics"]
    lines = [
        "# ACSR Transfer Objective Validation Gate",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Mean partner delta vs direct: `{metrics['mean_partner_transfer_minus_direct_ce']}`",
        f"- Mean partner delta vs token-position null: `{metrics['mean_partner_transfer_minus_token_position_ce']}`",
        f"- Mean partner delta vs random null: `{metrics['mean_partner_transfer_minus_random_ce']}`",
        f"- Max own CE delta vs direct: `{metrics['max_own_transfer_minus_direct_ce']}`",
        f"- Mean partner oracle-regret delta vs direct: `{metrics['mean_partner_oracle_regret_delta_vs_direct']}`",
        "",
        "This gate consumes existing command-generated local and fetched RunPod "
        "transfer-objective probe artifacts. It does not train, launch pods, or "
        "promote ACSR. Passing means the branch may continue to stronger held-out "
        "controls before any mechanism or default-router claim.",
    ]
    if summary["failures"]:
        lines.extend(["", "## Fail-Closed Reasons"])
        for failure in summary["failures"]:
            lines.append(f"- `{failure['gate']}`: {failure['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _number(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(left: Any, right: Any) -> float | None:
    if not isinstance(left, float) or not isinstance(right, float):
        return None
    return left - right


def _mean_key(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), float)]
    return float(mean(values)) if values else None


def _min_key(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), float)]
    return min(values) if values else None


def _max_key(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), float)]
    return max(values) if values else None


def _all_lt(rows: list[dict[str, Any]], key: str, threshold: float) -> bool:
    return bool(rows) and all(isinstance(row.get(key), float) and row[key] < threshold for row in rows)


def _all_le(rows: list[dict[str, Any]], key: str, threshold: float) -> bool:
    return bool(rows) and all(isinstance(row.get(key), float) and row[key] <= threshold for row in rows)


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


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
    parser.add_argument("--source-dir", type=Path, action="append", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_acsr_transfer_objective_validation_gate(
        source_dirs=tuple(args.source_dir) if args.source_dir else DEFAULT_SOURCE_DIRS,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
