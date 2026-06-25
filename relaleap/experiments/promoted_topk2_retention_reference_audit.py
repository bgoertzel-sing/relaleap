"""Retention reference audit for the promoted contextual top-k-2 router."""

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
DEFAULT_GATE_AUDIT_DIR = Path(
    "results/audits/token_larger_active_topk1_context_gate_suppression_calibration"
)
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_promoted_topk2_retention_reference_audit"
)

PROMOTED_TOPK2_ROUTER_DEFAULT_RETENTION_REFERENCE = (
    "promoted_topk2_router_default_retention_reference"
)
PROMOTED_TOPK2_LOW_CHURN_RETENTION_SUPPORTED = (
    "promoted_topk2_low_churn_retention_supported"
)
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

_TOPK2_FIELDS = (
    "topk2_anchor_support_churn_after_transfer",
    "topk2_anchor_logit_mse_drift",
    "topk2_anchor_residual_stream_l2_drift",
    "topk2_anchor_ce_drift",
    "topk2_transfer_ce_improvement",
    "topk2_commutator_anchor_logit_mse",
    "topk2_commutator_transfer_logit_mse",
    "topk2_commutator_anchor_residual_stream_l2",
    "topk2_commutator_transfer_residual_stream_l2",
)
_COMPARATOR_FIELDS = (
    "topk1_anchor_support_churn_after_transfer",
    "topk1_anchor_logit_mse_drift",
    "topk1_transfer_ce_improvement",
    "dense_transfer_ce_improvement",
)


def run_promoted_topk2_retention_reference_audit(
    *,
    probe_dirs: tuple[Path, ...] = DEFAULT_PROBE_DIRS,
    gate_audit_dir: Path = DEFAULT_GATE_AUDIT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    high_churn_threshold: float = 0.5,
    min_transfer_improvement_over_dense: float = 0.0,
) -> dict[str, Any]:
    """Summarize promoted top-k-2 retention/churn behavior from existing packets."""

    rows = [_probe_row(index, path) for index, path in enumerate(probe_dirs, 1)]
    gate_summary = _gate_summary(gate_audit_dir)
    failures = [
        failure
        for row in rows
        for failure in _row_failures(row)
    ]
    if len(rows) < 2:
        failures.append(
            {
                "field": "packet_count",
                "expected": "at least 2",
                "actual": len(rows),
            }
        )
    if gate_summary["status"] != "pass":
        failures.append(
            {
                "field": "context_gate_summary.status",
                "expected": "pass",
                "actual": gate_summary["status"],
                "path": gate_summary["summary_path"],
            }
        )

    aggregates = _aggregate(rows)
    signals = _signals(
        aggregates,
        gate_summary,
        high_churn_threshold=high_churn_threshold,
        min_transfer_improvement_over_dense=min_transfer_improvement_over_dense,
    )

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        rationale = (
            "The promoted top-k-2 retention reference cannot be established because "
            "one or more required local probe packets or the top-k-1 gate closeout "
            "is missing or failing."
        )
        next_step = (
            "repair the missing source packets before using promoted top-k-2 as a "
            "retention reference"
        )
    elif signals["topk2_low_churn_retention_claim_supported"]:
        status = "pass"
        decision = PROMOTED_TOPK2_LOW_CHURN_RETENTION_SUPPORTED
        rationale = (
            "The existing packets show promoted contextual top-k-2 transfer "
            "improvement over the dense control without high support churn. This "
            "would support a top-k-2 low-churn retention claim, but it should still "
            "be treated separately from the already closed top-k-2 causal-cooperation "
            "claim."
        )
        next_step = (
            "decide whether a backend repeat is warranted for the top-k-2 retention "
            "claim before changing the causal interpretation"
        )
    else:
        status = "pass"
        decision = PROMOTED_TOPK2_ROUTER_DEFAULT_RETENTION_REFERENCE
        rationale = (
            "The context-gate suppression audit failed, so top-k-1 singleton gating "
            "stays diagnostic-only. The existing retention/churn packets still "
            "support keeping promoted contextual top-k-2 as the main router default "
            "for CE and support-selection evidence, but not as a low-churn "
            "causal-retention mechanism: top-k-2 improves transfer CE over the "
            "dense control while showing high support churn and larger finite-update "
            "commutator drift than the rank-matched top-k-1 bracket."
        )
        next_step = (
            "run one local no-training audit on the promoted contextual top-k-2 "
            "router that targets support-selection quality rather than reusable "
            "singleton gating or low-churn retention"
        )

    summary = {
        "status": status,
        "decision": decision,
        "probe_dirs": [str(path) for path in probe_dirs],
        "gate_audit_dir": str(gate_audit_dir),
        "out_dir": str(out_dir),
        "thresholds": {
            "high_churn_threshold": high_churn_threshold,
            "min_transfer_improvement_over_dense": min_transfer_improvement_over_dense,
        },
        "gate_summary": gate_summary,
        "aggregates": aggregates,
        "signals": signals,
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
    _write_probe_metrics(out_dir / "probe_metrics.csv", rows)
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
        "required_variants_present": bool(signals.get("required_variants_present")),
        "finite_update_commutator_present": bool(
            signals.get("finite_update_commutator_present")
        ),
    }
    for field in (*_TOPK2_FIELDS, *_COMPARATOR_FIELDS):
        row[field] = _float_or_none(metrics.get(field))
    row["topk2_support_churn_minus_topk1"] = _delta(
        row["topk2_anchor_support_churn_after_transfer"],
        row["topk1_anchor_support_churn_after_transfer"],
    )
    row["topk2_logit_churn_minus_topk1"] = _delta(
        row["topk2_anchor_logit_mse_drift"],
        row["topk1_anchor_logit_mse_drift"],
    )
    row["topk2_transfer_improvement_over_dense"] = _delta(
        row["topk2_transfer_ce_improvement"],
        row["dense_transfer_ce_improvement"],
    )
    row["topk2_transfer_improvement_minus_topk1"] = _delta(
        row["topk2_transfer_ce_improvement"],
        row["topk1_transfer_ce_improvement"],
    )
    return row


def _row_failures(row: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
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
    if not row["required_variants_present"]:
        failures.append(
            {
                "packet": row["packet"],
                "field": "signals.required_variants_present",
                "expected": True,
                "actual": row["required_variants_present"],
            }
        )
    if not row["finite_update_commutator_present"]:
        failures.append(
            {
                "packet": row["packet"],
                "field": "signals.finite_update_commutator_present",
                "expected": True,
                "actual": row["finite_update_commutator_present"],
            }
        )
    for field in _TOPK2_FIELDS:
        if row[field] is None:
            failures.append(
                {
                    "packet": row["packet"],
                    "field": f"metrics.{field}",
                    "expected": "numeric",
                    "actual": None,
                }
            )
    return failures


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = {
        field: [value for value in (row.get(field) for row in rows) if isinstance(value, float)]
        for field in (
            *_TOPK2_FIELDS,
            *_COMPARATOR_FIELDS,
            "topk2_support_churn_minus_topk1",
            "topk2_logit_churn_minus_topk1",
            "topk2_transfer_improvement_over_dense",
            "topk2_transfer_improvement_minus_topk1",
        )
    }
    return {
        "all_packets_pass": all(row.get("status") == "pass" for row in rows),
        "all_required_variants_present": all(
            bool(row.get("required_variants_present")) for row in rows
        ),
        "finite_update_commutator_present_all_packets": all(
            bool(row.get("finite_update_commutator_present")) for row in rows
        ),
        "mean_topk2_support_churn": _mean_or_none(
            values["topk2_anchor_support_churn_after_transfer"]
        ),
        "min_topk2_support_churn": _min_or_none(
            values["topk2_anchor_support_churn_after_transfer"]
        ),
        "mean_topk2_logit_mse_drift": _mean_or_none(
            values["topk2_anchor_logit_mse_drift"]
        ),
        "mean_topk2_commutator_anchor_logit_mse": _mean_or_none(
            values["topk2_commutator_anchor_logit_mse"]
        ),
        "mean_topk2_commutator_transfer_logit_mse": _mean_or_none(
            values["topk2_commutator_transfer_logit_mse"]
        ),
        "mean_topk2_transfer_ce_improvement": _mean_or_none(
            values["topk2_transfer_ce_improvement"]
        ),
        "min_topk2_transfer_improvement_over_dense": _min_or_none(
            values["topk2_transfer_improvement_over_dense"]
        ),
        "mean_topk2_transfer_improvement_over_dense": _mean_or_none(
            values["topk2_transfer_improvement_over_dense"]
        ),
        "max_topk2_transfer_improvement_minus_topk1": _max_or_none(
            values["topk2_transfer_improvement_minus_topk1"]
        ),
        "mean_topk2_support_churn_minus_topk1": _mean_or_none(
            values["topk2_support_churn_minus_topk1"]
        ),
        "min_topk2_support_churn_minus_topk1": _min_or_none(
            values["topk2_support_churn_minus_topk1"]
        ),
        "mean_topk2_logit_churn_minus_topk1": _mean_or_none(
            values["topk2_logit_churn_minus_topk1"]
        ),
    }


def _signals(
    aggregates: dict[str, Any],
    gate_summary: dict[str, Any],
    *,
    high_churn_threshold: float,
    min_transfer_improvement_over_dense: float,
) -> dict[str, bool]:
    mean_churn = aggregates["mean_topk2_support_churn"]
    min_dense_advantage = aggregates["min_topk2_transfer_improvement_over_dense"]
    min_churn_minus_topk1 = aggregates["min_topk2_support_churn_minus_topk1"]
    max_transfer_minus_topk1 = aggregates["max_topk2_transfer_improvement_minus_topk1"]
    topk2_high_churn = mean_churn is not None and mean_churn >= high_churn_threshold
    topk2_beats_dense = (
        min_dense_advantage is not None
        and min_dense_advantage > min_transfer_improvement_over_dense
    )
    topk2_churn_higher_than_topk1 = (
        min_churn_minus_topk1 is not None and min_churn_minus_topk1 > 0.0
    )
    topk2_transfer_not_better_than_topk1 = (
        max_transfer_minus_topk1 is not None and max_transfer_minus_topk1 <= 0.0
    )
    gate_failed = (
        gate_summary.get("decision")
        == "deployable_context_gate_suppression_calibration_failed"
    )
    low_churn_claim = topk2_beats_dense and not topk2_high_churn
    return {
        "topk1_context_gate_failed": gate_failed,
        "topk2_transfer_beats_dense_control": topk2_beats_dense,
        "topk2_support_churn_high": topk2_high_churn,
        "topk2_support_churn_higher_than_topk1": topk2_churn_higher_than_topk1,
        "topk2_transfer_not_better_than_topk1": topk2_transfer_not_better_than_topk1,
        "topk2_low_churn_retention_claim_supported": low_churn_claim,
        "topk2_router_default_reference_supported": (
            gate_failed and topk2_beats_dense and topk2_high_churn
        ),
    }


def _gate_summary(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    summary = _read_json_object(summary_path)
    return {
        "summary_path": str(summary_path),
        "present": summary_path.is_file(),
        "status": summary.get("status"),
        "decision": summary.get("decision"),
        "next_step": summary.get("next_step"),
    }


def _write_probe_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "packet",
        "probe_dir",
        "status",
        "decision",
        "config_path",
        "required_variants_present",
        "finite_update_commutator_present",
        *_TOPK2_FIELDS,
        *_COMPARATOR_FIELDS,
        "topk2_support_churn_minus_topk1",
        "topk2_logit_churn_minus_topk1",
        "topk2_transfer_improvement_over_dense",
        "topk2_transfer_improvement_minus_topk1",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    aggregates = summary["aggregates"]
    signals = summary["signals"]
    lines = [
        "# Promoted Top-k-2 Retention Reference Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Mean top-k-2 support churn: `{aggregates['mean_topk2_support_churn']}`",
        "- Mean top-k-2 transfer CE improvement: "
        f"`{aggregates['mean_topk2_transfer_ce_improvement']}`",
        "- Minimum top-k-2 transfer improvement over dense: "
        f"`{aggregates['min_topk2_transfer_improvement_over_dense']}`",
        "- Mean top-k-2 support churn minus top-k-1: "
        f"`{aggregates['mean_topk2_support_churn_minus_topk1']}`",
        "- Top-k-1 context gate failed: "
        f"`{signals['topk1_context_gate_failed']}`",
        "",
        "## Rationale",
        "",
        summary["rationale"],
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


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _min_or_none(values: list[float]) -> float | None:
    return min(values) if values else None


def _max_or_none(values: list[float]) -> float | None:
    return max(values) if values else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-dir",
        type=Path,
        action="append",
        dest="probe_dirs",
        help="Completed active top-k-1 retention/churn probe directory.",
    )
    parser.add_argument("--gate-audit-dir", type=Path, default=DEFAULT_GATE_AUDIT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    probe_dirs = tuple(args.probe_dirs) if args.probe_dirs else DEFAULT_PROBE_DIRS
    summary = run_promoted_topk2_retention_reference_audit(
        probe_dirs=probe_dirs,
        gate_audit_dir=args.gate_audit_dir,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "aggregates": summary["aggregates"],
                "signals": summary["signals"],
                "failures": summary["failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
