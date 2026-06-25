"""Functional-churn control audit for promoted contextual top-k-2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from relaleap.experiments.promoted_topk2_support_retention_gap_selector import (
    SELECT_FUNCTIONAL_CHURN_CONTROLS,
)


DEFAULT_SELECTOR_DIR = Path(
    "results/reports/token_larger_promoted_topk2_support_retention_gap_selector"
)
DEFAULT_RETENTION_REFERENCE_DIR = Path(
    "results/reports/token_larger_promoted_topk2_retention_reference_audit"
)
DEFAULT_FINGERPRINT_DIR = Path(
    "results/audits/token_larger_support_wide_promoted_default_causal_column_fingerprint_stability_topk1"
)
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_promoted_topk2_functional_churn_control_audit"
)

FUNCTIONAL_CHURN_BOUNDED_WITH_COMMUTATOR_RISK = (
    "support_identity_churn_functional_impact_bounded_with_commutator_risk"
)
FUNCTIONAL_CHURN_FUNCTIONALLY_MEANINGFUL = (
    "support_identity_churn_functionally_meaningful"
)
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def run_promoted_topk2_functional_churn_control_audit(
    *,
    selector_dir: Path = DEFAULT_SELECTOR_DIR,
    retention_reference_dir: Path = DEFAULT_RETENTION_REFERENCE_DIR,
    fingerprint_dir: Path = DEFAULT_FINGERPRINT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    high_support_churn_gap_threshold: float = 0.5,
    low_logit_churn_gap_threshold: float = 0.05,
    low_residual_drift_ratio_threshold: float = 1.1,
    low_ce_drift_gap_threshold: float = 0.05,
    high_commutator_ratio_threshold: float = 5.0,
) -> dict[str, Any]:
    """Audit whether high top-k-2 support churn corresponds to functional churn."""

    selector = _read_json_object(selector_dir / "summary.json")
    retention = _read_json_object(retention_reference_dir / "summary.json")
    fingerprint = _read_json_object(fingerprint_dir / "summary.json")
    source_rows = [
        _source_row("gap_selector", selector_dir / "summary.json"),
        _source_row("retention_reference", retention_reference_dir / "summary.json"),
        _source_row("causal_fingerprint", fingerprint_dir / "summary.json"),
    ]
    packet_rows = _packet_rows(retention)
    fingerprint_rows = _fingerprint_rows(fingerprint)
    metrics = _metrics(packet_rows, fingerprint_rows)
    signals = _signals(
        selector,
        metrics,
        high_support_churn_gap_threshold=high_support_churn_gap_threshold,
        low_logit_churn_gap_threshold=low_logit_churn_gap_threshold,
        low_residual_drift_ratio_threshold=low_residual_drift_ratio_threshold,
        low_ce_drift_gap_threshold=low_ce_drift_gap_threshold,
        high_commutator_ratio_threshold=high_commutator_ratio_threshold,
    )
    failures = _failures(source_rows, packet_rows, fingerprint_rows, metrics, selector)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        rationale = (
            "The promoted top-k-2 functional-churn control audit cannot be "
            "interpreted because a required source packet is missing, failing, "
            "or lacks matched churn and finite-update metrics."
        )
        next_step = "repair_missing_promoted_topk2_functional_churn_sources"
    elif (
        signals["support_churn_gap_high"]
        and signals["logit_churn_gap_low"]
        and signals["residual_drift_ratio_low"]
        and signals["ce_drift_gap_low"]
        and signals["finite_update_commutator_risk_high"]
    ):
        status = "pass"
        decision = FUNCTIONAL_CHURN_BOUNDED_WITH_COMMUTATOR_RISK
        rationale = (
            "Across the existing matched retention packets, promoted top-k-2 has "
            "much higher support identity churn than rank-matched contextual "
            "top-k-1, but the extra logit, residual, and CE churn are small. "
            "That makes raw support-set churn look mostly like an "
            "interchangeable-support bookkeeping signal rather than direct "
            "functional harm. The unresolved risk is finite-update order "
            "sensitivity: the top-k-2/top-k-1 commutator gap remains large."
        )
        next_step = (
            "run a local no-training promoted top-k-2 finite-update order-control "
            "audit that separates support identity churn from commutator-driven "
            "logit and residual drift"
        )
    else:
        status = "pass"
        decision = FUNCTIONAL_CHURN_FUNCTIONALLY_MEANINGFUL
        rationale = (
            "The current matched packets do not bound top-k-2 functional churn: "
            "support identity churn is accompanied by enough logit, residual, CE, "
            "or previous-support drift to treat it as functionally meaningful."
        )
        next_step = (
            "keep promoted contextual top-k-2 as a CE/support-selection reference "
            "only and run a bounded local mitigation selector before any backend "
            "repeat"
        )

    summary = {
        "status": status,
        "decision": decision,
        "out_dir": str(out_dir),
        "source_rows": source_rows,
        "thresholds": {
            "high_support_churn_gap_threshold": high_support_churn_gap_threshold,
            "low_logit_churn_gap_threshold": low_logit_churn_gap_threshold,
            "low_residual_drift_ratio_threshold": low_residual_drift_ratio_threshold,
            "low_ce_drift_gap_threshold": low_ce_drift_gap_threshold,
            "high_commutator_ratio_threshold": high_commutator_ratio_threshold,
        },
        "packet_rows": packet_rows,
        "fingerprint_functional_churn_rows": fingerprint_rows,
        "metrics": metrics,
        "signals": signals,
        "failures": failures,
        "rationale": rationale,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "packet_metrics_csv": str(out_dir / "packet_metrics.csv"),
            "fingerprint_functional_churn_csv": str(
                out_dir / "fingerprint_functional_churn.csv"
            ),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_rows.csv", source_rows)
    _write_csv(out_dir / "packet_metrics.csv", packet_rows)
    _write_csv(out_dir / "fingerprint_functional_churn.csv", fingerprint_rows)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _packet_rows(retention: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in retention.get("probe_rows", []):
        if not isinstance(source, dict):
            continue
        topk2_churn = _float_or_none(source.get("topk2_anchor_support_churn_after_transfer"))
        topk1_churn = _float_or_none(source.get("topk1_anchor_support_churn_after_transfer"))
        topk2_logit = _float_or_none(source.get("topk2_anchor_logit_mse_drift"))
        topk1_logit = _float_or_none(source.get("topk1_anchor_logit_mse_drift"))
        topk2_residual = _float_or_none(source.get("topk2_anchor_residual_stream_l2_drift"))
        topk1_residual = _source_topk1_residual(source)
        topk2_ce = _float_or_none(source.get("topk2_anchor_ce_drift"))
        topk1_ce = _source_topk1_ce(source)
        topk2_commutator = _float_or_none(source.get("topk2_commutator_anchor_logit_mse"))
        topk1_commutator = _source_topk1_commutator(source)
        row = {
            "packet": source.get("packet"),
            "config_path": source.get("config_path"),
            "topk2_support_churn": topk2_churn,
            "topk1_support_churn": topk1_churn,
            "support_churn_gap": _delta(topk2_churn, topk1_churn),
            "topk2_logit_mse_drift": topk2_logit,
            "topk1_logit_mse_drift": topk1_logit,
            "logit_churn_gap": _delta(topk2_logit, topk1_logit),
            "topk2_residual_l2_drift": topk2_residual,
            "topk1_residual_l2_drift": topk1_residual,
            "residual_drift_gap": _delta(topk2_residual, topk1_residual),
            "residual_drift_ratio": _ratio(topk2_residual, topk1_residual),
            "topk2_ce_drift_abs": abs(topk2_ce) if topk2_ce is not None else None,
            "topk1_ce_drift_abs": abs(topk1_ce) if topk1_ce is not None else None,
            "ce_drift_abs_gap": _delta(
                abs(topk2_ce) if topk2_ce is not None else None,
                abs(topk1_ce) if topk1_ce is not None else None,
            ),
            "topk2_commutator_logit_mse": topk2_commutator,
            "topk1_commutator_logit_mse": topk1_commutator,
            "commutator_logit_mse_ratio": _ratio(topk2_commutator, topk1_commutator),
        }
        rows.append(row)
    return rows


def _source_topk1_residual(row: dict[str, Any]) -> float | None:
    value = _float_or_none(row.get("topk1_anchor_residual_stream_l2_drift"))
    if value is not None:
        return value
    summary = _read_json_object(Path(str(row.get("probe_dir", ""))) / "summary.json")
    metrics = _evidence_metrics(summary)
    return _float_or_none(metrics.get("topk1_anchor_residual_stream_l2_drift"))


def _source_topk1_ce(row: dict[str, Any]) -> float | None:
    value = _float_or_none(row.get("topk1_anchor_ce_drift"))
    if value is not None:
        return value
    summary = _read_json_object(Path(str(row.get("probe_dir", ""))) / "summary.json")
    metrics = _evidence_metrics(summary)
    return _float_or_none(metrics.get("topk1_anchor_ce_drift"))


def _source_topk1_commutator(row: dict[str, Any]) -> float | None:
    value = _float_or_none(row.get("topk1_commutator_anchor_logit_mse"))
    if value is not None:
        return value
    summary = _read_json_object(Path(str(row.get("probe_dir", ""))) / "summary.json")
    metrics = _evidence_metrics(summary)
    return _float_or_none(metrics.get("topk1_commutator_anchor_logit_mse"))


def _evidence_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    evidence = summary.get("evidence")
    if not isinstance(evidence, dict):
        return {}
    metrics = evidence.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _fingerprint_rows(fingerprint: dict[str, Any]) -> list[dict[str, Any]]:
    audit = fingerprint.get("audit")
    churn = audit.get("functional_churn") if isinstance(audit, dict) else []
    rows: list[dict[str, Any]] = []
    for row in churn if isinstance(churn, list) else []:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "variant": row.get("variant"),
                "load_balance_weight": _float_or_none(row.get("load_balance_weight")),
                "adjacent_support_identity_churn_fraction": _float_or_none(
                    row.get("adjacent_support_identity_churn_fraction")
                ),
                "previous_support_changed_logit_mse_mean": _float_or_none(
                    row.get("previous_support_changed_logit_mse_mean")
                ),
                "previous_support_changed_residual_l2_mean": _float_or_none(
                    row.get("previous_support_changed_residual_l2_mean")
                ),
            }
        )
    return rows


def _metrics(
    packet_rows: list[dict[str, Any]],
    fingerprint_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    topk2_fingerprint = _first(
        row for row in fingerprint_rows if row.get("variant") == "baseline"
    )
    topk1_fingerprint = _first(
        row
        for row in fingerprint_rows
        if row.get("variant") == "rank_matched_topk1_contextual"
    )
    return {
        "packet_count": len(packet_rows),
        "mean_support_churn_gap": _mean_field(packet_rows, "support_churn_gap"),
        "mean_logit_churn_gap": _mean_field(packet_rows, "logit_churn_gap"),
        "mean_residual_drift_ratio": _mean_field(packet_rows, "residual_drift_ratio"),
        "mean_ce_drift_abs_gap": _mean_field(packet_rows, "ce_drift_abs_gap"),
        "mean_commutator_logit_mse_ratio": _mean_field(
            packet_rows, "commutator_logit_mse_ratio"
        ),
        "topk2_previous_support_changed_logit_mse_mean": (
            topk2_fingerprint or {}
        ).get("previous_support_changed_logit_mse_mean"),
        "topk1_previous_support_changed_logit_mse_mean": (
            topk1_fingerprint or {}
        ).get("previous_support_changed_logit_mse_mean"),
        "topk2_previous_support_changed_residual_l2_mean": (
            topk2_fingerprint or {}
        ).get("previous_support_changed_residual_l2_mean"),
        "topk1_previous_support_changed_residual_l2_mean": (
            topk1_fingerprint or {}
        ).get("previous_support_changed_residual_l2_mean"),
    }


def _signals(
    selector: dict[str, Any],
    metrics: dict[str, Any],
    *,
    high_support_churn_gap_threshold: float,
    low_logit_churn_gap_threshold: float,
    low_residual_drift_ratio_threshold: float,
    low_ce_drift_gap_threshold: float,
    high_commutator_ratio_threshold: float,
) -> dict[str, bool]:
    return {
        "selector_selected_functional_churn_controls": selector.get("decision")
        == SELECT_FUNCTIONAL_CHURN_CONTROLS,
        "support_churn_gap_high": _at_least(
            metrics["mean_support_churn_gap"], high_support_churn_gap_threshold
        ),
        "logit_churn_gap_low": _abs_at_most(
            metrics["mean_logit_churn_gap"], low_logit_churn_gap_threshold
        ),
        "residual_drift_ratio_low": _at_most(
            metrics["mean_residual_drift_ratio"], low_residual_drift_ratio_threshold
        ),
        "ce_drift_gap_low": _abs_at_most(
            metrics["mean_ce_drift_abs_gap"], low_ce_drift_gap_threshold
        ),
        "finite_update_commutator_risk_high": _at_least(
            metrics["mean_commutator_logit_mse_ratio"],
            high_commutator_ratio_threshold,
        ),
    }


def _failures(
    source_rows: list[dict[str, Any]],
    packet_rows: list[dict[str, Any]],
    fingerprint_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    selector: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    expected = {
        "gap_selector": {"pass"},
        "retention_reference": {"pass"},
        "causal_fingerprint": {"ok", "pass"},
    }
    for row in source_rows:
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
        if row["status"] not in expected[row["source"]]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "status",
                    "expected": sorted(expected[row["source"]]),
                    "actual": row["status"],
                }
            )
    if selector.get("decision") != SELECT_FUNCTIONAL_CHURN_CONTROLS:
        failures.append(
            {
                "source": "gap_selector",
                "field": "decision",
                "expected": SELECT_FUNCTIONAL_CHURN_CONTROLS,
                "actual": selector.get("decision"),
            }
        )
    if len(packet_rows) < 2:
        failures.append(
            {"field": "packet_rows", "expected": "at least 2", "actual": len(packet_rows)}
        )
    if len(fingerprint_rows) < 2:
        failures.append(
            {
                "field": "fingerprint_functional_churn_rows",
                "expected": "at least 2",
                "actual": len(fingerprint_rows),
            }
        )
    required = (
        "mean_support_churn_gap",
        "mean_logit_churn_gap",
        "mean_residual_drift_ratio",
        "mean_ce_drift_abs_gap",
        "mean_commutator_logit_mse_ratio",
    )
    for field in required:
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    lines = [
        "# Promoted Top-k-2 Functional-Churn Control Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Mean support-churn gap: `{metrics['mean_support_churn_gap']}`",
        f"- Mean logit-churn gap: `{metrics['mean_logit_churn_gap']}`",
        f"- Mean residual-drift ratio: `{metrics['mean_residual_drift_ratio']}`",
        f"- Mean CE-drift absolute gap: `{metrics['mean_ce_drift_abs_gap']}`",
        "- Mean finite-update commutator logit-MSE ratio: "
        f"`{metrics['mean_commutator_logit_mse_ratio']}`",
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


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _ratio(left: float | None, right: float | None) -> float | None:
    if left is None or right in (None, 0.0):
        return None
    return left / right


def _mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [_float_or_none(row.get(field)) for row in rows]
    numeric = [value for value in values if value is not None]
    return mean(numeric) if numeric else None


def _first(values: Any) -> Any | None:
    for value in values:
        return value
    return None


def _at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def _at_most(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold


def _abs_at_most(value: float | None, threshold: float) -> bool:
    return value is not None and abs(value) <= threshold


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_promoted_topk2_functional_churn_control_audit(out_dir=args.out)
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
