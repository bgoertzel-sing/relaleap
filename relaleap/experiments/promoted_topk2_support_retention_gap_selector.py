"""Select the next promoted top-k-2 support-retention gap audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_RETENTION_REFERENCE_DIR = Path(
    "results/reports/token_larger_promoted_topk2_retention_reference_audit"
)
DEFAULT_SUPPORT_QUALITY_DIR = Path(
    "results/reports/token_larger_promoted_topk2_support_selection_quality_audit"
)
DEFAULT_LOAD_BALANCE_CLOSEOUT_DIR = Path(
    "results/reports/token_larger_promoted_topk2_load_balance_closeout"
)
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_promoted_topk2_support_retention_gap_selector"
)

SELECT_FUNCTIONAL_CHURN_CONTROLS = "select_functional_churn_controls"
SELECT_RESIDUAL_SUM_NORMALIZATION_CONTROLS = (
    "select_residual_sum_normalization_controls"
)
SELECT_BOUNDED_BACKEND_REPEAT = "select_bounded_backend_repeat"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def run_promoted_topk2_support_retention_gap_selector(
    *,
    retention_reference_dir: Path = DEFAULT_RETENTION_REFERENCE_DIR,
    support_quality_dir: Path = DEFAULT_SUPPORT_QUALITY_DIR,
    load_balance_closeout_dir: Path = DEFAULT_LOAD_BALANCE_CLOSEOUT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    high_support_churn_threshold: float = 0.5,
    low_logit_churn_gap_threshold: float = 0.05,
    high_commutator_ratio_threshold: float = 5.0,
    residual_drift_ratio_threshold: float = 1.25,
) -> dict[str, Any]:
    """Choose one no-training next step from existing validated packets."""

    retention = _read_json_object(retention_reference_dir / "summary.json")
    support_quality = _read_json_object(support_quality_dir / "summary.json")
    load_balance = _read_json_object(load_balance_closeout_dir / "summary.json")
    source_rows = [
        _source_row("retention_reference", retention_reference_dir / "summary.json"),
        _source_row("support_selection_quality", support_quality_dir / "summary.json"),
        _source_row("load_balance_closeout", load_balance_closeout_dir / "summary.json"),
    ]
    metrics = _metrics(retention, support_quality, load_balance)
    signals = _signals(
        metrics,
        retention,
        support_quality,
        load_balance,
        high_support_churn_threshold=high_support_churn_threshold,
        low_logit_churn_gap_threshold=low_logit_churn_gap_threshold,
        high_commutator_ratio_threshold=high_commutator_ratio_threshold,
        residual_drift_ratio_threshold=residual_drift_ratio_threshold,
    )
    failures = _failures(source_rows, metrics, retention, support_quality, load_balance)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        rationale = (
            "The support-retention gap selector cannot be interpreted because a "
            "required source packet is missing, failing, or lacks the retention "
            "or support-quality metrics needed for the local no-training choice."
        )
        next_step = "repair_missing_support_retention_selector_source_packets"
    elif (
        signals["support_selection_good"]
        and signals["load_balance_closed"]
        and signals["support_churn_high"]
        and signals["logit_churn_gap_low"]
        and signals["commutator_gap_high"]
    ):
        status = "pass"
        decision = SELECT_FUNCTIONAL_CHURN_CONTROLS
        rationale = (
            "The promoted contextual top-k-2 router still looks like the right CE "
            "and support-selection reference, but its high support churn is not "
            "matched by a large extra logit-drift gap. The largest unresolved "
            "gap is whether support churn is functionally meaningful or mostly "
            "an interchangeable-support bookkeeping effect, especially given the "
            "large top-k-2 finite-update commutator gap. The next local audit "
            "should therefore target functional churn directly before spending "
            "GPU time or changing router defaults."
        )
        next_step = (
            "run a local no-training promoted top-k-2 functional-churn control "
            "audit that measures support-set churn against logit, residual, and "
            "CE churn under matched contexts and finite-update order controls"
        )
    elif signals["residual_drift_gap_high"]:
        status = "pass"
        decision = SELECT_RESIDUAL_SUM_NORMALIZATION_CONTROLS
        rationale = (
            "The current packets suggest the main unresolved top-k-2 gap is "
            "residual-stream magnitude rather than support selection or load "
            "balance. A residual-sum normalization control is the most focused "
            "local next step."
        )
        next_step = (
            "run a local no-training residual-sum normalization control audit "
            "for promoted top-k-2 support interventions"
        )
    else:
        status = "pass"
        decision = SELECT_BOUNDED_BACKEND_REPEAT
        rationale = (
            "The existing local packets do not isolate a clear support-retention "
            "mechanism gap, so the next coherent step is a bounded backend repeat "
            "of the currently selected evidence before adding new controls."
        )
        next_step = (
            "run one bounded backend repeat of the promoted top-k-2 retention "
            "and support-selection packet, then re-run this selector locally"
        )

    summary = {
        "status": status,
        "decision": decision,
        "out_dir": str(out_dir),
        "source_rows": source_rows,
        "thresholds": {
            "high_support_churn_threshold": high_support_churn_threshold,
            "low_logit_churn_gap_threshold": low_logit_churn_gap_threshold,
            "high_commutator_ratio_threshold": high_commutator_ratio_threshold,
            "residual_drift_ratio_threshold": residual_drift_ratio_threshold,
        },
        "metrics": metrics,
        "signals": signals,
        "failures": failures,
        "rationale": rationale,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_source_rows(out_dir / "source_rows.csv", source_rows)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _metrics(
    retention: dict[str, Any],
    support_quality: dict[str, Any],
    load_balance: dict[str, Any],
) -> dict[str, Any]:
    aggregates = (
        retention.get("aggregates")
        if isinstance(retention.get("aggregates"), dict)
        else {}
    )
    support_metrics = (
        support_quality.get("metrics")
        if isinstance(support_quality.get("metrics"), dict)
        else {}
    )
    load_metrics = (
        load_balance.get("metrics")
        if isinstance(load_balance.get("metrics"), dict)
        else {}
    )
    mean_topk2_commutator = _float_or_none(
        aggregates.get("mean_topk2_commutator_anchor_logit_mse")
    )
    probe_rows = (
        retention.get("probe_rows")
        if isinstance(retention.get("probe_rows"), list)
        else []
    )
    source_probe_metrics = _source_probe_metrics(retention)
    topk1_commutators = [
        _float_or_none(row.get("topk1_commutator_anchor_logit_mse"))
        for row in probe_rows
        if isinstance(row, dict)
    ] + [
        _float_or_none(row.get("topk1_commutator_anchor_logit_mse"))
        for row in source_probe_metrics
    ]
    topk1_residual_drifts = [
        _float_or_none(row.get("topk1_anchor_residual_stream_l2_drift"))
        for row in probe_rows
        if isinstance(row, dict)
    ] + [
        _float_or_none(row.get("topk1_anchor_residual_stream_l2_drift"))
        for row in source_probe_metrics
    ]
    topk2_residual_drifts = [
        _float_or_none(row.get("topk2_anchor_residual_stream_l2_drift"))
        for row in probe_rows
        if isinstance(row, dict)
    ]
    mean_topk1_commutator = _mean_or_none(
        [value for value in topk1_commutators if value is not None]
    )
    mean_topk1_residual_drift = _mean_or_none(
        [value for value in topk1_residual_drifts if value is not None]
    )
    mean_topk2_residual_drift = _mean_or_none(
        [value for value in topk2_residual_drifts if value is not None]
    )
    return {
        "retention_decision": retention.get("decision"),
        "support_quality_decision": support_quality.get("decision"),
        "load_balance_decision": load_balance.get("decision"),
        "mean_topk2_support_churn": _float_or_none(
            aggregates.get("mean_topk2_support_churn")
        ),
        "mean_topk2_support_churn_minus_topk1": _float_or_none(
            aggregates.get("mean_topk2_support_churn_minus_topk1")
        ),
        "mean_topk2_logit_churn_minus_topk1": _float_or_none(
            aggregates.get("mean_topk2_logit_churn_minus_topk1")
        ),
        "mean_topk2_commutator_anchor_logit_mse": mean_topk2_commutator,
        "mean_topk1_commutator_anchor_logit_mse": mean_topk1_commutator,
        "topk2_to_topk1_commutator_ratio": _ratio(
            mean_topk2_commutator,
            mean_topk1_commutator,
        ),
        "mean_topk2_residual_stream_l2_drift": mean_topk2_residual_drift,
        "mean_topk1_residual_stream_l2_drift": mean_topk1_residual_drift,
        "topk2_to_topk1_residual_drift_ratio": _ratio(
            mean_topk2_residual_drift,
            mean_topk1_residual_drift,
        ),
        "oracle_support_regret": _float_or_none(
            support_metrics.get("oracle_support_regret")
        ),
        "oracle_support_regret_positive_fraction": _float_or_none(
            support_metrics.get("oracle_support_regret_positive_fraction")
        ),
        "router_improvement_over_best_global_fixed_support": _float_or_none(
            support_metrics.get("router_improvement_over_best_global_fixed_support")
        ),
        "min_used_column_gain": _float_or_none(load_metrics.get("min_used_column_gain")),
        "max_mean_abs_force_delta_gain": _float_or_none(
            load_metrics.get("max_mean_abs_force_delta_gain")
        ),
    }


def _source_probe_metrics(retention: dict[str, Any]) -> list[dict[str, Any]]:
    probe_dirs = (
        retention.get("probe_dirs")
        if isinstance(retention.get("probe_dirs"), list)
        else []
    )
    rows: list[dict[str, Any]] = []
    for value in probe_dirs:
        if not isinstance(value, str):
            continue
        summary = _read_json_object(Path(value) / "summary.json")
        evidence = summary.get("evidence") if isinstance(summary.get("evidence"), dict) else {}
        metrics = evidence.get("metrics") if isinstance(evidence.get("metrics"), dict) else {}
        rows.append(metrics)
    return rows


def _signals(
    metrics: dict[str, Any],
    retention: dict[str, Any],
    support_quality: dict[str, Any],
    load_balance: dict[str, Any],
    *,
    high_support_churn_threshold: float,
    low_logit_churn_gap_threshold: float,
    high_commutator_ratio_threshold: float,
    residual_drift_ratio_threshold: float,
) -> dict[str, bool]:
    support_churn = metrics["mean_topk2_support_churn"]
    logit_gap = metrics["mean_topk2_logit_churn_minus_topk1"]
    commutator_ratio = metrics["topk2_to_topk1_commutator_ratio"]
    residual_ratio = metrics["topk2_to_topk1_residual_drift_ratio"]
    return {
        "retention_reference_established": retention.get("decision")
        == "promoted_topk2_router_default_retention_reference",
        "support_selection_good": support_quality.get("decision")
        == "promoted_topk2_support_selection_quality_established",
        "load_balance_closed": load_balance.get("decision")
        == "keep_load_balance_opt_in_branch_closed",
        "support_churn_high": (
            support_churn is not None and support_churn >= high_support_churn_threshold
        ),
        "logit_churn_gap_low": (
            logit_gap is not None and abs(logit_gap) <= low_logit_churn_gap_threshold
        ),
        "commutator_gap_high": (
            commutator_ratio is not None
            and commutator_ratio >= high_commutator_ratio_threshold
        ),
        "residual_drift_gap_high": (
            residual_ratio is not None and residual_ratio >= residual_drift_ratio_threshold
        ),
    }


def _failures(
    source_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    retention: dict[str, Any],
    support_quality: dict[str, Any],
    load_balance: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    expected = {
        "retention_reference": ("pass", "promoted_topk2_router_default_retention_reference"),
        "support_selection_quality": (
            "pass",
            "promoted_topk2_support_selection_quality_established",
        ),
        "load_balance_closeout": ("pass", "keep_load_balance_opt_in_branch_closed"),
    }
    for row in source_rows:
        status, decision = expected[row["source"]]
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
            continue
        if row["status"] != status:
            failures.append(
                {
                    "source": row["source"],
                    "field": "status",
                    "expected": status,
                    "actual": row["status"],
                }
            )
        if row["decision"] != decision:
            failures.append(
                {
                    "source": row["source"],
                    "field": "decision",
                    "expected": decision,
                    "actual": row["decision"],
                }
            )
    required_metrics = (
        "mean_topk2_support_churn",
        "mean_topk2_logit_churn_minus_topk1",
        "topk2_to_topk1_commutator_ratio",
        "topk2_to_topk1_residual_drift_ratio",
        "oracle_support_regret",
        "router_improvement_over_best_global_fixed_support",
    )
    for field in required_metrics:
        if metrics.get(field) is None:
            failures.append(
                {"field": f"metrics.{field}", "expected": "numeric", "actual": None}
            )
    return failures


def _source_row(source: str, path: Path) -> dict[str, Any]:
    value = _read_json_object(path)
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": value.get("status"),
        "decision": value.get("decision"),
    }


def _write_source_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["source", "path", "present", "status", "decision"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    lines = [
        "# Promoted Top-k-2 Support-Retention Gap Selector",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Mean top-k-2 support churn: `{metrics['mean_topk2_support_churn']}`",
        "- Mean top-k-2 logit-churn minus top-k-1: "
        f"`{metrics['mean_topk2_logit_churn_minus_topk1']}`",
        "- Top-k-2/top-k-1 commutator ratio: "
        f"`{metrics['topk2_to_topk1_commutator_ratio']}`",
        "- Top-k-2/top-k-1 residual-drift ratio: "
        f"`{metrics['topk2_to_topk1_residual_drift_ratio']}`",
        f"- Oracle support regret: `{metrics['oracle_support_regret']}`",
        "",
        "## Signals",
        "",
    ]
    for key, value in summary["signals"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
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
    )
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
    return sum(values) / len(values) if values else None


def _ratio(left: float | None, right: float | None) -> float | None:
    if left is None or right in (None, 0.0):
        return None
    return left / right


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_promoted_topk2_support_retention_gap_selector(out_dir=args.out)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "metrics": summary["metrics"],
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
