"""Held-out context controls for the ACSR transfer-objective probe."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_SOURCE_DIRS = (
    Path("results/audits/acsr_transfer_objective_probe"),
    Path("results/audits/acsr_transfer_objective_probe_seed2"),
    Path("results/runpod_fetch/audits/runpod_acsr_transfer_objective_probe"),
    Path("results/runpod_fetch/audits/runpod_acsr_transfer_objective_probe_seed2"),
)
DEFAULT_OUT_DIR = Path("results/reports/acsr_transfer_objective_heldout_control_gate")
REQUIRED_SOURCE_ARTIFACTS = (
    "summary.json",
    "arm_metrics.csv",
    "per_token_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)
REQUIRED_OUTPUT_ARTIFACTS = (
    "summary.json",
    "source_packets.csv",
    "context_split_metrics.csv",
    "residual_norm_bin_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)
EXPECTED_BACKENDS = ("local", "runpod")
EXPECTED_SEEDS = ("seed1", "seed2")
OWN_CE_GUARDRAIL = 0.02


def run_acsr_transfer_objective_heldout_control_gate(
    *,
    source_dirs: tuple[Path, ...] = DEFAULT_SOURCE_DIRS,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Test existing transfer-objective packets on held-out positions and norm bins."""

    start = time.time()
    packets = [_load_packet(path) for path in source_dirs]
    source_rows = [_source_packet_row(packet) for packet in packets]
    context_rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []
    for packet in packets:
        context_rows.extend(_context_split_rows(packet))
        norm_rows.extend(_residual_norm_bin_rows(packet))
    aggregate = _aggregate(source_rows, context_rows, norm_rows)
    gate_rows = _gate_rows(source_rows, aggregate)
    failures = [
        {"gate": row["criterion"], "reason": row["failure_reason"]}
        for row in gate_rows
        if not row["passed"]
    ]
    status = "pass" if not failures else "fail"
    summary = {
        "status": status,
        "decision": (
            "acsr_transfer_objective_heldout_control_gate_passed"
            if status == "pass"
            else "acsr_transfer_objective_heldout_control_gate_failed_closed"
        ),
        "claim_status": (
            "heldout_transfer_controls_supported_not_promoted"
            if status == "pass"
            else "heldout_transfer_controls_not_supported"
        ),
        "selected_next_step": (
            "add a true rank-or-FLOP matched dense residual/control probe before any ACSR mechanism claim"
            if status == "pass"
            else "stop the ACSR transfer-objective branch and inspect held-out or residual-norm failures"
        ),
        "source_dirs": [str(path) for path in source_dirs],
        "source_packets": source_rows,
        "context_split_metrics": context_rows,
        "residual_norm_bin_metrics": norm_rows,
        "aggregate_metrics": aggregate,
        "gate_criteria": gate_rows,
        "failures": failures,
        "claim_boundaries": {
            "supported": [
                "existing local and fetched RunPod transfer-objective packets retain partner-through-values gains on held-out positions",
                "held-out gains beat the direct causal MLP, token-position trained null, and frequency-random support null in every packet",
                "held-out own-value CE damage remains within the current guardrail",
                "held-out partner gains are present in both low and high direct-residual-norm bins",
            ],
            "not_supported": [
                "ACSR-as-anticipation",
                "default router promotion",
                "a mechanism claim before true rank/FLOP-matched dense residual or adapter controls",
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
    return {
        "source_dir": str(source_dir),
        "backend": _backend_label(source_dir),
        "seed": _seed_label(source_dir, summary),
        "missing_artifacts": missing,
        "summary": summary,
        "per_token_rows": _read_csv(source_dir / "per_token_metrics.csv"),
        "gate_rows": _read_csv(source_dir / "gate_criteria.csv"),
    }


def _source_packet_row(packet: dict[str, Any]) -> dict[str, Any]:
    summary = packet["summary"]
    individual_failures = [
        row.get("criterion", "unknown")
        for row in packet["gate_rows"]
        if not _bool_value(row.get("passed"))
    ]
    token_count = _tokens_per_arm(packet)
    return {
        "source_dir": packet["source_dir"],
        "backend": packet["backend"],
        "seed": packet["seed"],
        "loaded": not packet["missing_artifacts"] and bool(summary) and token_count > 0,
        "status": summary.get("status", "missing"),
        "decision": summary.get("decision", ""),
        "missing_artifacts": ";".join(packet["missing_artifacts"]),
        "individual_gate_failures": ";".join(individual_failures),
        "tokens_per_arm": token_count,
        "seq_len_minus_one": _seq_len_minus_one(packet),
        "train_position_cutoff": _train_position_cutoff(packet),
        "git_commit": summary.get("git_commit", ""),
        "platform": summary.get("platform", ""),
    }


def _context_split_rows(packet: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seq_len_minus_one = _seq_len_minus_one(packet)
    train_cutoff = _train_position_cutoff(packet)
    if seq_len_minus_one <= 0 or train_cutoff <= 0:
        return rows
    for split_name, include_position in (
        ("train_positions", lambda position: position < train_cutoff),
        ("heldout_positions", lambda position: position >= train_cutoff),
    ):
        for value_path, baseline_arm in (
            ("partner_values", "direct_causal_mlp_baseline"),
            ("partner_values", "token_position_only_transfer_null"),
            ("partner_values", "random_frequency_support_null"),
            ("own_values", "direct_causal_mlp_baseline"),
        ):
            deltas = _paired_ce_deltas(
                packet,
                value_path=value_path,
                left_arm="transfer_objective_router",
                right_arm=baseline_arm,
                include_position=include_position,
                seq_len_minus_one=seq_len_minus_one,
            )
            rows.append(
                {
                    "source_dir": packet["source_dir"],
                    "backend": packet["backend"],
                    "seed": packet["seed"],
                    "context_split": split_name,
                    "value_path": value_path,
                    "comparison": f"transfer_objective_router_minus_{baseline_arm}",
                    "mean_ce_delta": _mean(deltas),
                    "token_count": len(deltas),
                }
            )
    return rows


def _residual_norm_bin_rows(packet: dict[str, Any]) -> list[dict[str, Any]]:
    seq_len_minus_one = _seq_len_minus_one(packet)
    train_cutoff = _train_position_cutoff(packet)
    if seq_len_minus_one <= 0 or train_cutoff <= 0:
        return []
    direct_rows = _rows_by_index(
        packet,
        value_path="partner_values",
        arm="direct_causal_mlp_baseline",
    )
    heldout_norms = [
        _float(row.get("residual_update_l2"))
        for index, row in direct_rows.items()
        if index % seq_len_minus_one >= train_cutoff
    ]
    heldout_norms = [value for value in heldout_norms if value is not None]
    if not heldout_norms:
        return []
    threshold = float(median(heldout_norms))
    rows = []
    for bin_name, predicate in (
        ("low_direct_residual_norm", lambda value: value <= threshold),
        ("high_direct_residual_norm", lambda value: value > threshold),
    ):
        deltas = _paired_ce_deltas(
            packet,
            value_path="partner_values",
            left_arm="transfer_objective_router",
            right_arm="direct_causal_mlp_baseline",
            include_position=lambda position: position >= train_cutoff,
            seq_len_minus_one=seq_len_minus_one,
            include_index=lambda index: predicate(
                _float(direct_rows[index].get("residual_update_l2")) or 0.0
            ),
        )
        rows.append(
            {
                "source_dir": packet["source_dir"],
                "backend": packet["backend"],
                "seed": packet["seed"],
                "context_split": "heldout_positions",
                "value_path": "partner_values",
                "comparison": "transfer_objective_router_minus_direct_causal_mlp_baseline",
                "residual_norm_bin": bin_name,
                "direct_residual_norm_median": threshold,
                "mean_ce_delta": _mean(deltas),
                "token_count": len(deltas),
            }
        )
    return rows


def _paired_ce_deltas(
    packet: dict[str, Any],
    *,
    value_path: str,
    left_arm: str,
    right_arm: str,
    include_position: Any,
    seq_len_minus_one: int,
    include_index: Any | None = None,
) -> list[float]:
    left_rows = _rows_by_index(packet, value_path=value_path, arm=left_arm)
    right_rows = _rows_by_index(packet, value_path=value_path, arm=right_arm)
    deltas = []
    for index, left in left_rows.items():
        right = right_rows.get(index)
        if right is None:
            continue
        position = index % seq_len_minus_one
        if not include_position(position):
            continue
        if include_index is not None and not include_index(index):
            continue
        left_loss = _float(left.get("ce_loss"))
        right_loss = _float(right.get("ce_loss"))
        if left_loss is None or right_loss is None:
            continue
        deltas.append(left_loss - right_loss)
    return deltas


def _rows_by_index(packet: dict[str, Any], *, value_path: str, arm: str) -> dict[int, dict[str, Any]]:
    rows = {}
    for row in packet["per_token_rows"]:
        if row.get("value_path") != value_path or row.get("arm") != arm:
            continue
        try:
            rows[int(row["token_index"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def _aggregate(
    source_rows: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
    norm_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    heldout_partner_direct = _select(
        context_rows,
        context_split="heldout_positions",
        value_path="partner_values",
        comparison="transfer_objective_router_minus_direct_causal_mlp_baseline",
    )
    heldout_partner_token = _select(
        context_rows,
        context_split="heldout_positions",
        value_path="partner_values",
        comparison="transfer_objective_router_minus_token_position_only_transfer_null",
    )
    heldout_partner_random = _select(
        context_rows,
        context_split="heldout_positions",
        value_path="partner_values",
        comparison="transfer_objective_router_minus_random_frequency_support_null",
    )
    heldout_own_direct = _select(
        context_rows,
        context_split="heldout_positions",
        value_path="own_values",
        comparison="transfer_objective_router_minus_direct_causal_mlp_baseline",
    )
    low_norm = _select(norm_rows, residual_norm_bin="low_direct_residual_norm")
    high_norm = _select(norm_rows, residual_norm_bin="high_direct_residual_norm")
    return {
        "packet_count": len(source_rows),
        "loaded_backend_seed_cells": sorted(
            {
                f"{row.get('backend')}:{row.get('seed')}"
                for row in source_rows
                if row.get("loaded") and row.get("status") == "pass"
            }
        ),
        "mean_heldout_partner_transfer_minus_direct_ce": _mean_values(heldout_partner_direct),
        "mean_heldout_partner_transfer_minus_token_position_ce": _mean_values(heldout_partner_token),
        "mean_heldout_partner_transfer_minus_random_ce": _mean_values(heldout_partner_random),
        "mean_heldout_own_transfer_minus_direct_ce": _mean_values(heldout_own_direct),
        "max_heldout_own_transfer_minus_direct_ce": _max_values(heldout_own_direct),
        "mean_low_norm_heldout_partner_transfer_minus_direct_ce": _mean_values(low_norm),
        "mean_high_norm_heldout_partner_transfer_minus_direct_ce": _mean_values(high_norm),
        "all_heldout_partner_beats_direct": _all_lt(heldout_partner_direct, 0.0),
        "all_heldout_partner_beats_token_position_null": _all_lt(heldout_partner_token, 0.0),
        "all_heldout_partner_beats_random_null": _all_lt(heldout_partner_random, 0.0),
        "all_heldout_own_ce_within_guardrail": _all_le(heldout_own_direct, OWN_CE_GUARDRAIL),
        "all_low_norm_heldout_partner_beats_direct": _all_lt(low_norm, 0.0),
        "all_high_norm_heldout_partner_beats_direct": _all_lt(high_norm, 0.0),
        "heldout_context_rows": len(heldout_partner_direct),
        "residual_norm_bin_rows": len(norm_rows),
    }


def _gate_rows(
    source_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_cells = {
        f"{backend}:{seed}" for backend in EXPECTED_BACKENDS for seed in EXPECTED_SEEDS
    }
    loaded_cells = set(aggregate["loaded_backend_seed_cells"])
    missing_cells = sorted(expected_cells - loaded_cells)
    return [
        _criterion(
            "required_source_artifacts_present",
            all(row["loaded"] for row in source_rows),
            "all source packets expose summary, arm, per-token, gate, and notes artifacts",
            ";".join(row["source_dir"] for row in source_rows if not row["loaded"]) or "all_present",
            "one or more source packet artifacts are missing or empty",
        ),
        _criterion(
            "backend_seed_coverage_complete",
            not missing_cells,
            "local and RunPod seed1/seed2 packets are present",
            ";".join(sorted(loaded_cells)),
            f"missing backend/seed cells: {';'.join(missing_cells)}",
        ),
        _criterion(
            "source_status_pass",
            all(row["status"] == "pass" for row in source_rows),
            "all source summaries report status pass",
            ";".join(f"{row['backend']}:{row['seed']}={row['status']}" for row in source_rows),
            "one or more source summaries did not pass",
        ),
        _criterion(
            "individual_probe_gates_passed",
            all(not row["individual_gate_failures"] for row in source_rows),
            "all individual source gate rows pass",
            ";".join(row["individual_gate_failures"] for row in source_rows if row["individual_gate_failures"]) or "all_passed",
            "one or more source probe gate criteria failed",
        ),
        _criterion(
            "heldout_partner_transfer_beats_direct_all_packets",
            aggregate["all_heldout_partner_beats_direct"],
            "held-out partner-through-values CE beats direct causal MLP in every packet",
            aggregate["mean_heldout_partner_transfer_minus_direct_ce"],
            "held-out transfer objective did not beat direct causal MLP in every packet",
        ),
        _criterion(
            "heldout_partner_transfer_beats_token_position_null_all_packets",
            aggregate["all_heldout_partner_beats_token_position_null"],
            "held-out partner-through-values CE beats token-position trained null in every packet",
            aggregate["mean_heldout_partner_transfer_minus_token_position_ce"],
            "held-out transfer objective did not beat token-position null in every packet",
        ),
        _criterion(
            "heldout_partner_transfer_beats_random_null_all_packets",
            aggregate["all_heldout_partner_beats_random_null"],
            "held-out partner-through-values CE beats frequency-random support null in every packet",
            aggregate["mean_heldout_partner_transfer_minus_random_ce"],
            "held-out transfer objective did not beat random-frequency support null in every packet",
        ),
        _criterion(
            "heldout_own_ce_guardrail_all_packets",
            aggregate["all_heldout_own_ce_within_guardrail"],
            f"held-out own-value CE worsens by no more than {OWN_CE_GUARDRAIL} versus direct causal MLP",
            aggregate["max_heldout_own_transfer_minus_direct_ce"],
            "held-out own-value CE damage exceeded guardrail",
        ),
        _criterion(
            "heldout_low_norm_partner_transfer_beats_direct_all_packets",
            aggregate["all_low_norm_heldout_partner_beats_direct"],
            "held-out partner gain versus direct survives low direct-residual-norm bin",
            aggregate["mean_low_norm_heldout_partner_transfer_minus_direct_ce"],
            "held-out gain appears absent in low direct-residual-norm bin",
        ),
        _criterion(
            "heldout_high_norm_partner_transfer_beats_direct_all_packets",
            aggregate["all_high_norm_heldout_partner_beats_direct"],
            "held-out partner gain versus direct survives high direct-residual-norm bin",
            aggregate["mean_high_norm_heldout_partner_transfer_minus_direct_ce"],
            "held-out gain appears absent in high direct-residual-norm bin",
        ),
    ]


def _tokens_per_arm(packet: dict[str, Any]) -> int:
    return len(
        _rows_by_index(
            packet,
            value_path="partner_values",
            arm="direct_causal_mlp_baseline",
        )
    )


def _seq_len_minus_one(packet: dict[str, Any]) -> int:
    count = _tokens_per_arm(packet)
    if count <= 0:
        return 0
    # The command harness uses batch_size=4 for this probe.
    if count % 4 == 0:
        return count // 4
    return count


def _train_position_cutoff(packet: dict[str, Any]) -> int:
    seq_len_minus_one = _seq_len_minus_one(packet)
    if seq_len_minus_one <= 0:
        return 0
    seq_len = seq_len_minus_one + 1
    split = max(1, seq_len // 2)
    return max(1, split - 1)


def _select(rows: list[dict[str, Any]], **filters: str) -> list[float]:
    values = []
    for row in rows:
        if any(row.get(key) != value for key, value in filters.items()):
            continue
        value = _float(row.get("mean_ce_delta"))
        if value is not None:
            values.append(value)
    return values


def _mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def _mean_values(values: list[float]) -> float | None:
    return _mean(values)


def _max_values(values: list[float]) -> float | None:
    return max(values) if values else None


def _all_lt(values: list[float], threshold: float) -> bool:
    return bool(values) and all(value < threshold for value in values)


def _all_le(values: list[float], threshold: float) -> bool:
    return bool(values) and all(value <= threshold for value in values)


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
    _write_csv(out_dir / "context_split_metrics.csv", summary["context_split_metrics"])
    _write_csv(out_dir / "residual_norm_bin_metrics.csv", summary["residual_norm_bin_metrics"])
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
        "# ACSR Transfer Objective Held-Out Control Gate",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Mean held-out partner delta vs direct: `{metrics['mean_heldout_partner_transfer_minus_direct_ce']}`",
        f"- Mean held-out partner delta vs token-position null: `{metrics['mean_heldout_partner_transfer_minus_token_position_ce']}`",
        f"- Mean held-out partner delta vs random null: `{metrics['mean_heldout_partner_transfer_minus_random_ce']}`",
        f"- Max held-out own CE delta vs direct: `{metrics['max_heldout_own_transfer_minus_direct_ce']}`",
        f"- Mean low-norm held-out partner delta vs direct: `{metrics['mean_low_norm_heldout_partner_transfer_minus_direct_ce']}`",
        f"- Mean high-norm held-out partner delta vs direct: `{metrics['mean_high_norm_heldout_partner_transfer_minus_direct_ce']}`",
        "",
        "This no-training gate consumes existing command-generated local and fetched "
        "RunPod transfer-objective probe artifacts. It treats positions from the "
        "second half of each sequence as held-out relative to the router objective "
        "fit, then checks direct, token-position, random-support, own-CE, and "
        "direct-residual-norm-bin controls. Passing does not promote ACSR.",
    ]
    if summary["failures"]:
        lines.extend(["", "## Fail-Closed Reasons"])
        for failure in summary["failures"]:
            lines.append(f"- `{failure['gate']}`: {failure['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    summary = run_acsr_transfer_objective_heldout_control_gate(
        source_dirs=tuple(args.source_dir) if args.source_dir else DEFAULT_SOURCE_DIRS,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
