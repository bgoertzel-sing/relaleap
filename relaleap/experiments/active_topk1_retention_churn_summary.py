"""Summarize seed stability for the active top-k-1 retention/churn probes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from relaleap.experiments.active_topk1_retention_churn_probe import (
    ACTIVE_TOPK1_RETENTION_CHURN_PROBE_ESTABLISHED,
)


DEFAULT_PROBE_DIRS = (
    Path("results/audits/token_larger_active_rank_matched_topk1_retention_churn_probe"),
    Path(
        "results/audits/token_larger_active_rank_matched_topk1_retention_churn_probe_seed2"
    ),
)
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_active_rank_matched_topk1_retention_churn_stability"
)

ACTIVE_TOPK1_RETENTION_CHURN_STABLE = (
    "active_topk1_retention_churn_stable_across_local_seeds"
)
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

_REQUIRED_SIGNALS = (
    "required_variants_present",
    "topk1_support_churn_lower_than_topk2",
    "topk1_logit_churn_not_higher_than_topk2",
    "topk1_transfer_improvement_at_least_topk2",
    "source_singleton_gain_still_negative",
)
_METRIC_FIELDS = (
    "topk1_anchor_support_churn_after_transfer",
    "topk2_anchor_support_churn_after_transfer",
    "topk1_anchor_logit_mse_drift",
    "topk2_anchor_logit_mse_drift",
    "topk1_transfer_ce_improvement",
    "topk2_transfer_ce_improvement",
    "dense_transfer_ce_improvement",
    "source_topk1_singleton_gain_mean",
    "source_context_level_topk1_singleton_gain_mean",
)


def summarize_active_topk1_retention_churn(
    *,
    probe_dirs: tuple[Path, ...] = DEFAULT_PROBE_DIRS,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Aggregate completed probe packets into a bounded stability decision."""

    rows = [_probe_row(index, path) for index, path in enumerate(probe_dirs, start=1)]
    failures = [
        failure
        for row in rows
        for failure in _row_failures(row)
    ]
    aggregates = _aggregate_rows(rows)
    enough_packets = len(rows) >= 2
    stable = enough_packets and not failures and aggregates["all_required_signals_pass"]

    if stable:
        status = "pass"
        decision = ACTIVE_TOPK1_RETENTION_CHURN_STABLE
        rationale = (
            "Both local seed packets pass the active top-k-1 retention/churn probe. "
            "Rank-matched contextual top-k-1 has much lower support churn than the "
            "promoted top-k-2 reference in each packet, no higher logit churn, and "
            "at least comparable transfer CE improvement. This establishes a local "
            "retention/churn stability bracket, while preserving the negative "
            "singleton-gain caveat from the source separability packet."
        )
        next_step = (
            "decide whether the local retention/churn stability bracket warrants "
            "a targeted Colab/GPU replication or should remain local support for "
            "the active rank-matched top-k-1 causal bracket"
        )
    else:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        rationale = (
            "The completed probe packets do not yet establish seed-stable active "
            "top-k-1 retention/churn evidence."
        )
        next_step = (
            "repair or regenerate the missing or failing active top-k-1 "
            "retention/churn probe packets before considering replication"
        )

    summary = {
        "status": status,
        "decision": decision,
        "probe_dirs": [str(path) for path in probe_dirs],
        "packet_count": len(rows),
        "required_signals": list(_REQUIRED_SIGNALS),
        "aggregates": aggregates,
        "probe_rows": rows,
        "failures": failures,
        "rationale": rationale,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "probe_metrics_csv": str(out_dir / "probe_metrics.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_probe_metrics_csv(out_dir / "probe_metrics.csv", rows)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _probe_row(index: int, path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    summary = _read_json_object(summary_path)
    evidence = summary.get("evidence", {})
    metrics = evidence.get("metrics", {}) if isinstance(evidence, dict) else {}
    signals = evidence.get("signals", {}) if isinstance(evidence, dict) else {}
    row: dict[str, Any] = {
        "packet": f"seed{index}",
        "probe_dir": str(path),
        "summary_path": str(summary_path),
        "summary_present": summary_path.is_file(),
        "status": summary.get("status"),
        "decision": summary.get("decision"),
        "config_path": summary.get("config_path"),
    }
    for field in _METRIC_FIELDS:
        row[field] = _float_or_none(metrics.get(field))
    for signal in _REQUIRED_SIGNALS:
        row[signal] = bool(signals.get(signal))
    topk1_churn = row["topk1_anchor_support_churn_after_transfer"]
    topk2_churn = row["topk2_anchor_support_churn_after_transfer"]
    topk1_logit = row["topk1_anchor_logit_mse_drift"]
    topk2_logit = row["topk2_anchor_logit_mse_drift"]
    topk1_transfer = row["topk1_transfer_ce_improvement"]
    topk2_transfer = row["topk2_transfer_ce_improvement"]
    row["support_churn_advantage"] = (
        topk2_churn - topk1_churn
        if topk1_churn is not None and topk2_churn is not None
        else None
    )
    row["logit_churn_advantage"] = (
        topk2_logit - topk1_logit
        if topk1_logit is not None and topk2_logit is not None
        else None
    )
    row["transfer_improvement_advantage"] = (
        topk1_transfer - topk2_transfer
        if topk1_transfer is not None and topk2_transfer is not None
        else None
    )
    return row


def _row_failures(row: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    if not row["summary_present"]:
        failures.append(
            {
                "packet": row["packet"],
                "field": "summary_json",
                "expected": "file exists",
                "actual": "missing",
                "path": row["summary_path"],
            }
        )
        return failures
    if row["status"] != "pass":
        failures.append(
            {
                "packet": row["packet"],
                "field": "status",
                "expected": "pass",
                "actual": row["status"],
            }
        )
    if row["decision"] != ACTIVE_TOPK1_RETENTION_CHURN_PROBE_ESTABLISHED:
        failures.append(
            {
                "packet": row["packet"],
                "field": "decision",
                "expected": ACTIVE_TOPK1_RETENTION_CHURN_PROBE_ESTABLISHED,
                "actual": row["decision"],
            }
        )
    for signal in _REQUIRED_SIGNALS:
        if not row[signal]:
            failures.append(
                {
                    "packet": row["packet"],
                    "field": f"signals.{signal}",
                    "expected": True,
                    "actual": row[signal],
                }
            )
    return failures


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {}
    for field in (
        *_METRIC_FIELDS,
        "support_churn_advantage",
        "logit_churn_advantage",
        "transfer_improvement_advantage",
    ):
        values[field] = [
            value
            for value in (row.get(field) for row in rows)
            if isinstance(value, float)
        ]
    return {
        "all_required_signals_pass": all(
            all(bool(row.get(signal)) for signal in _REQUIRED_SIGNALS)
            for row in rows
        ),
        "all_packets_pass": all(row.get("status") == "pass" for row in rows),
        "topk1_support_churn_lower_than_topk2_all_packets": all(
            bool(row.get("topk1_support_churn_lower_than_topk2")) for row in rows
        ),
        "topk1_logit_churn_not_higher_all_packets": all(
            bool(row.get("topk1_logit_churn_not_higher_than_topk2")) for row in rows
        ),
        "topk1_transfer_improvement_at_least_topk2_all_packets": all(
            bool(row.get("topk1_transfer_improvement_at_least_topk2"))
            for row in rows
        ),
        "source_singleton_gain_negative_all_packets": all(
            bool(row.get("source_singleton_gain_still_negative")) for row in rows
        ),
        "mean_topk1_support_churn": _mean_or_none(
            values["topk1_anchor_support_churn_after_transfer"]
        ),
        "mean_topk2_support_churn": _mean_or_none(
            values["topk2_anchor_support_churn_after_transfer"]
        ),
        "min_support_churn_advantage": _min_or_none(values["support_churn_advantage"]),
        "mean_support_churn_advantage": _mean_or_none(
            values["support_churn_advantage"]
        ),
        "min_logit_churn_advantage": _min_or_none(values["logit_churn_advantage"]),
        "mean_logit_churn_advantage": _mean_or_none(values["logit_churn_advantage"]),
        "min_transfer_improvement_advantage": _min_or_none(
            values["transfer_improvement_advantage"]
        ),
        "mean_transfer_improvement_advantage": _mean_or_none(
            values["transfer_improvement_advantage"]
        ),
    }


def _write_probe_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "packet",
        "probe_dir",
        "status",
        "decision",
        "config_path",
        *_REQUIRED_SIGNALS,
        *_METRIC_FIELDS,
        "support_churn_advantage",
        "logit_churn_advantage",
        "transfer_improvement_advantage",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    aggregates = summary["aggregates"]
    lines = [
        "# Active Top-k-1 Retention/Churn Stability",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Packet count: `{summary['packet_count']}`",
        "- Mean top-k-1 support churn: "
        f"`{aggregates['mean_topk1_support_churn']}`",
        "- Mean top-k-2 support churn: "
        f"`{aggregates['mean_topk2_support_churn']}`",
        "- Minimum support-churn advantage: "
        f"`{aggregates['min_support_churn_advantage']}`",
        "- Minimum logit-churn advantage: "
        f"`{aggregates['min_logit_churn_advantage']}`",
        "- Minimum transfer-improvement advantage: "
        f"`{aggregates['min_transfer_improvement_advantage']}`",
        "",
        "## Rationale",
        "",
        summary["rationale"],
        "",
        "## Caveat",
        "",
        "The source separability packet still reports negative top-k-1 singleton "
        "gain, so this summary should be read as retention/churn stability "
        "evidence for the active bracket, not as a singleton causal "
        "separability claim.",
        "",
        "## Next Step",
        "",
        summary["next_step"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _min_or_none(values: list[float]) -> float | None:
    return min(values) if values else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-dir",
        type=Path,
        action="append",
        dest="probe_dirs",
        help="Completed active top-k-1 retention/churn probe directory.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    probe_dirs = tuple(args.probe_dirs) if args.probe_dirs else DEFAULT_PROBE_DIRS
    summary = summarize_active_topk1_retention_churn(
        probe_dirs=probe_dirs,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "aggregates": summary["aggregates"],
                "failures": summary["failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
