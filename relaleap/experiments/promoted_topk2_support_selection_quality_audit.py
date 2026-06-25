"""Support-selection quality audit for the promoted contextual top-k-2 router."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_EXHAUSTIVE_AUDIT_DIR = Path(
    "results/audits/token_larger_support_wide_promoted_default_exhaustive_support"
)
DEFAULT_EXHAUSTIVE_REPORT_DIR = Path(
    "results/reports/token_larger_support_wide_promoted_default_exhaustive_support_audit"
)
DEFAULT_RETENTION_REFERENCE_DIR = Path(
    "results/reports/token_larger_promoted_topk2_retention_reference_audit"
)
DEFAULT_GATE_AUDIT_DIR = Path(
    "results/audits/token_larger_active_topk1_context_gate_suppression_calibration"
)
DEFAULT_COLUMN_REDUNDANCY_DIR = Path(
    "results/audits/token_larger_support_wide_promoted_default_column_redundancy"
)
DEFAULT_DEAD_COLUMN_PROBE_DIR = Path(
    "results/audits/token_larger_support_wide_promoted_default_dead_column_probe_low_weight_bracket"
)
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_promoted_topk2_support_selection_quality_audit"
)

PROMOTED_TOPK2_SUPPORT_SELECTION_QUALITY_ESTABLISHED = (
    "promoted_topk2_support_selection_quality_established"
)
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def run_promoted_topk2_support_selection_quality_audit(
    *,
    exhaustive_audit_dir: Path = DEFAULT_EXHAUSTIVE_AUDIT_DIR,
    exhaustive_report_dir: Path = DEFAULT_EXHAUSTIVE_REPORT_DIR,
    retention_reference_dir: Path = DEFAULT_RETENTION_REFERENCE_DIR,
    gate_audit_dir: Path = DEFAULT_GATE_AUDIT_DIR,
    column_redundancy_dir: Path = DEFAULT_COLUMN_REDUNDANCY_DIR,
    dead_column_probe_dir: Path = DEFAULT_DEAD_COLUMN_PROBE_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    small_oracle_regret_ce: float = 0.01,
    low_positive_regret_fraction: float = 0.10,
) -> dict[str, Any]:
    """Summarize existing top-k-2 oracle-regret evidence without retraining."""

    exhaustive = _read_json_object(exhaustive_audit_dir / "summary.json")
    exhaustive_report = _read_json_object(exhaustive_report_dir / "decision_report.json")
    retention = _read_json_object(retention_reference_dir / "summary.json")
    gate = _read_json_object(gate_audit_dir / "summary.json")
    column_redundancy = _read_json_object(column_redundancy_dir / "summary.json")
    dead_probe = _read_json_object(dead_column_probe_dir / "summary.json")

    source_rows = [
        _source_row("exhaustive_support_audit", exhaustive_audit_dir / "summary.json", exhaustive),
        _source_row(
            "exhaustive_support_decision",
            exhaustive_report_dir / "decision_report.json",
            exhaustive_report,
        ),
        _source_row(
            "promoted_topk2_retention_reference",
            retention_reference_dir / "summary.json",
            retention,
        ),
        _source_row("topk1_context_gate_audit", gate_audit_dir / "summary.json", gate),
        _source_row("column_redundancy", column_redundancy_dir / "summary.json", column_redundancy),
        _source_row("dead_column_probe", dead_column_probe_dir / "summary.json", dead_probe),
    ]
    metrics = _metrics(exhaustive)
    signals = _signals(
        metrics,
        retention,
        gate,
        dead_probe,
        small_oracle_regret_ce=small_oracle_regret_ce,
        low_positive_regret_fraction=low_positive_regret_fraction,
    )
    failures = _failures(source_rows, metrics, retention, gate)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        rationale = (
            "The promoted top-k-2 support-selection audit cannot be interpreted "
            "because a required source packet is missing, failing, or lacks the "
            "oracle-regret fields needed for a no-training closeout."
        )
        next_step = "repair_missing_support_selection_source_packets"
    else:
        status = "pass"
        decision = PROMOTED_TOPK2_SUPPORT_SELECTION_QUALITY_ESTABLISHED
        rationale = (
            "The promoted contextual top-k-2 router has small fixed-batch "
            "per-token oracle-support regret and beats every global fixed pair by "
            "a wide CE margin, so the active top-k-2 issue is not broad "
            "support-selection failure. The oracle-target contextual selector is "
            "an upper bound, while the deployable contextual support head recovers "
            "only a small positive share of the remaining gap. Combined with the "
            "failed top-k-1 gate and high top-k-2 churn, this supports keeping "
            "top-k-2 as the router-default/reference line, not a low-churn "
            "retention or singleton-reuse claim."
        )
        next_step = (
            "run one local no-training load-balance closeout over the existing "
            "low-weight dead-column probe packets before considering any "
            "support-router default change"
        )

    summary = {
        "status": status,
        "decision": decision,
        "out_dir": str(out_dir),
        "source_rows": source_rows,
        "thresholds": {
            "small_oracle_regret_ce": small_oracle_regret_ce,
            "low_positive_regret_fraction": low_positive_regret_fraction,
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


def _metrics(exhaustive: dict[str, Any]) -> dict[str, Any]:
    audit = exhaustive.get("audit", {}) if isinstance(exhaustive.get("audit"), dict) else {}
    support_audit = (
        audit.get("support_audit", {}) if isinstance(audit.get("support_audit"), dict) else {}
    )
    contextual_target = _nested(audit, "router_oracle_target_contextual_diagnostic", "holdout")
    linear_target = _nested(audit, "router_oracle_target_diagnostic", "holdout")
    nonlinear_target = _nested(audit, "router_oracle_target_nonlinear_diagnostic", "holdout")
    contextual_head = _nested(audit, "contextual_router_support_head", "holdout")
    contextual_intervention = _nested(audit, "contextual_router_support_intervention", "holdout")
    router_loss = _float_or_none(audit.get("router_loss"))
    best_fixed_loss = _float_or_none(audit.get("best_global_fixed_support_loss"))
    oracle_loss = _float_or_none(audit.get("oracle_loss"))
    return {
        "config_path": audit.get("config_path") or exhaustive.get("config_path"),
        "dataset": audit.get("dataset"),
        "num_columns": _int_or_none(audit.get("num_columns")),
        "top_k": _int_or_none(audit.get("top_k")),
        "support_router": audit.get("support_router"),
        "router_loss": router_loss,
        "oracle_loss": oracle_loss,
        "router_oracle_gap": _delta(router_loss, oracle_loss),
        "oracle_support_regret": _float_or_none(audit.get("oracle_support_regret")),
        "oracle_support_regret_positive_fraction": _float_or_none(
            audit.get("oracle_support_regret_positive_fraction")
        ),
        "best_global_fixed_support": audit.get("best_global_fixed_support"),
        "best_global_fixed_support_loss": best_fixed_loss,
        "router_improvement_over_best_global_fixed_support": _delta(best_fixed_loss, router_loss),
        "dominant_router_support": audit.get("dominant_router_support"),
        "dominant_router_support_regret": _float_or_none(
            audit.get("dominant_router_support_regret")
        ),
        "used_columns": _int_or_none(support_audit.get("used_columns")),
        "dead_columns": _int_or_none(support_audit.get("dead_columns")),
        "unique_support_sets": _int_or_none(support_audit.get("unique_support_sets")),
        "contextual_oracle_target_holdout_accuracy": _float_or_none(
            contextual_target.get("oracle_target_accuracy")
        ),
        "contextual_oracle_target_holdout_gap_recovery": _float_or_none(
            contextual_target.get("oracle_gap_recovery_fraction")
        ),
        "contextual_oracle_target_holdout_minus_router_loss": _float_or_none(
            contextual_target.get("selector_minus_router_loss")
        ),
        "linear_oracle_target_holdout_gap_recovery": _float_or_none(
            linear_target.get("oracle_gap_recovery_fraction")
        ),
        "nonlinear_oracle_target_holdout_gap_recovery": _float_or_none(
            nonlinear_target.get("oracle_gap_recovery_fraction")
        ),
        "contextual_support_head_holdout_gap_recovery": _float_or_none(
            contextual_head.get("oracle_gap_recovery_fraction")
        ),
        "contextual_support_head_holdout_minus_router_loss": _float_or_none(
            contextual_head.get("intervention_minus_router_loss")
        ),
        "contextual_support_intervention_holdout_gap_recovery": _float_or_none(
            contextual_intervention.get("oracle_gap_recovery_fraction")
        ),
        "contextual_support_intervention_holdout_minus_router_loss": _float_or_none(
            contextual_intervention.get("intervention_minus_router_loss")
        ),
    }


def _signals(
    metrics: dict[str, Any],
    retention: dict[str, Any],
    gate: dict[str, Any],
    dead_probe: dict[str, Any],
    *,
    small_oracle_regret_ce: float,
    low_positive_regret_fraction: float,
) -> dict[str, bool]:
    regret = metrics["oracle_support_regret"]
    positive_fraction = metrics["oracle_support_regret_positive_fraction"]
    fixed_improvement = metrics["router_improvement_over_best_global_fixed_support"]
    head_recovery = metrics["contextual_support_head_holdout_gap_recovery"]
    oracle_target_recovery = metrics["contextual_oracle_target_holdout_gap_recovery"]
    dead_decision = _nested(dead_probe, "probe", "decision")
    return {
        "topk1_context_gate_failed": gate.get("decision")
        == "deployable_context_gate_suppression_calibration_failed",
        "topk2_retention_reference_established": retention.get("decision")
        == "promoted_topk2_router_default_retention_reference",
        "oracle_regret_small": regret is not None and regret <= small_oracle_regret_ce,
        "positive_regret_fraction_low": (
            positive_fraction is not None
            and positive_fraction <= low_positive_regret_fraction
        ),
        "router_beats_best_global_fixed_pair": (
            fixed_improvement is not None and fixed_improvement > 0.0
        ),
        "oracle_contextual_selector_is_upper_bound": (
            oracle_target_recovery is not None and oracle_target_recovery >= 0.95
        ),
        "deployable_contextual_support_head_positive_but_small": (
            head_recovery is not None and 0.0 < head_recovery < 0.5
        ),
        "dead_columns_recruited_without_ce_hurt": dead_decision.get("status")
        == "recruited_without_ce_hurt",
    }


def _failures(
    source_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    retention: dict[str, Any],
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    required = {
        "exhaustive_support_audit": {"status": "ok"},
        "exhaustive_support_decision": {"status": "pass"},
        "promoted_topk2_retention_reference": {"status": "pass"},
        "topk1_context_gate_audit": {"status": "pass"},
    }
    for row in source_rows:
        expected = required.get(row["source"])
        if expected is None:
            continue
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
        if row["status"] != expected["status"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "status",
                    "expected": expected["status"],
                    "actual": row["status"],
                    "path": row["path"],
                }
            )
    for field in (
        "router_loss",
        "oracle_loss",
        "oracle_support_regret",
        "oracle_support_regret_positive_fraction",
        "best_global_fixed_support_loss",
        "contextual_support_head_holdout_gap_recovery",
        "contextual_oracle_target_holdout_gap_recovery",
    ):
        if metrics.get(field) is None:
            failures.append({"field": f"metrics.{field}", "expected": "numeric", "actual": None})
    if retention.get("decision") != "promoted_topk2_router_default_retention_reference":
        failures.append(
            {
                "source": "promoted_topk2_retention_reference",
                "field": "decision",
                "expected": "promoted_topk2_router_default_retention_reference",
                "actual": retention.get("decision"),
            }
        )
    if gate.get("decision") != "deployable_context_gate_suppression_calibration_failed":
        failures.append(
            {
                "source": "topk1_context_gate_audit",
                "field": "decision",
                "expected": "deployable_context_gate_suppression_calibration_failed",
                "actual": gate.get("decision"),
            }
        )
    return failures


def _source_row(source: str, path: Path, value: dict[str, Any]) -> dict[str, Any]:
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
    signals = summary["signals"]
    lines = [
        "# Promoted Top-k-2 Support Selection Quality Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Router loss: `{metrics['router_loss']}`",
        f"- Oracle loss: `{metrics['oracle_loss']}`",
        f"- Oracle support regret: `{metrics['oracle_support_regret']}`",
        "- Oracle support positive-regret fraction: "
        f"`{metrics['oracle_support_regret_positive_fraction']}`",
        "- Router improvement over best global fixed pair: "
        f"`{metrics['router_improvement_over_best_global_fixed_support']}`",
        "- Contextual oracle-target holdout gap recovery: "
        f"`{metrics['contextual_oracle_target_holdout_gap_recovery']}`",
        "- Deployable contextual support-head holdout gap recovery: "
        f"`{metrics['contextual_support_head_holdout_gap_recovery']}`",
        f"- Dead columns recruited without CE hurt: `{signals['dead_columns_recruited_without_ce_hurt']}`",
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


def _nested(value: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exhaustive-audit-dir", type=Path, default=DEFAULT_EXHAUSTIVE_AUDIT_DIR)
    parser.add_argument("--exhaustive-report-dir", type=Path, default=DEFAULT_EXHAUSTIVE_REPORT_DIR)
    parser.add_argument("--retention-reference-dir", type=Path, default=DEFAULT_RETENTION_REFERENCE_DIR)
    parser.add_argument("--gate-audit-dir", type=Path, default=DEFAULT_GATE_AUDIT_DIR)
    parser.add_argument("--column-redundancy-dir", type=Path, default=DEFAULT_COLUMN_REDUNDANCY_DIR)
    parser.add_argument("--dead-column-probe-dir", type=Path, default=DEFAULT_DEAD_COLUMN_PROBE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_promoted_topk2_support_selection_quality_audit(
        exhaustive_audit_dir=args.exhaustive_audit_dir,
        exhaustive_report_dir=args.exhaustive_report_dir,
        retention_reference_dir=args.retention_reference_dir,
        gate_audit_dir=args.gate_audit_dir,
        column_redundancy_dir=args.column_redundancy_dir,
        dead_column_probe_dir=args.dead_column_probe_dir,
        out_dir=args.out,
    )
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
