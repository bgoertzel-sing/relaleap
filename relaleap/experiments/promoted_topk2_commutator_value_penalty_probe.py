"""Commutator-aware value penalty probe for promoted contextual top-k-2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from relaleap.experiments.retention_churn_microtest import DEFAULT_CONFIG
from relaleap.experiments.retention_churn_microtest import run_retention_churn_microtest


DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_promoted_topk2_commutator_value_penalty_probe"
)

COMMUTATOR_VALUE_PENALTY_CANDIDATE_FOUND = (
    "commutator_value_penalty_candidate_found"
)
COMMUTATOR_VALUE_PENALTY_NOT_ESTABLISHED = (
    "commutator_value_penalty_not_established"
)
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

_BASELINE_VARIANT = "promoted_contextual_topk2"
_CONTROL_VARIANTS = (
    "rank_matched_contextual_topk1",
    "random_fixed_topk2",
    "norm_matched_dense_active_rank",
)
_PENALTY_VARIANTS = (
    "commutator_value_penalty_w010_contextual_topk2",
    "commutator_value_penalty_w100_contextual_topk2",
)


def run_promoted_topk2_commutator_value_penalty_probe(
    *,
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path = DEFAULT_OUT_DIR,
    commutator_reduction_fraction: float = 0.5,
    transfer_retention_fraction: float = 0.8,
    support_usage_retention_fraction: float = 0.8,
    anchor_ce_drift_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Run and gate residual-change penalty variants plus existing controls."""

    out_dir.mkdir(parents=True, exist_ok=True)
    microtest = run_retention_churn_microtest(
        config_path,
        out_dir,
        include_commutator_value_penalty_variants=True,
    )
    variant_rows = [
        row
        for row in microtest.get("audit", {}).get("variants", [])
        if isinstance(row, dict)
    ]
    variants = {str(row.get("variant")): row for row in variant_rows}
    penalty_rows = [
        _penalty_row(variants, name) for name in _PENALTY_VARIANTS if name in variants
    ]
    thresholds = {
        "commutator_reduction_fraction": commutator_reduction_fraction,
        "transfer_retention_fraction": transfer_retention_fraction,
        "support_usage_retention_fraction": support_usage_retention_fraction,
        "anchor_ce_drift_tolerance": anchor_ce_drift_tolerance,
    }
    failures = _failures(microtest, variants)
    metrics = _metrics(variants, penalty_rows)
    qualifying_rows = [
        row
        for row in penalty_rows
        if _qualifies(
            row,
            commutator_reduction_fraction=commutator_reduction_fraction,
            transfer_retention_fraction=transfer_retention_fraction,
            support_usage_retention_fraction=support_usage_retention_fraction,
            anchor_ce_drift_tolerance=anchor_ce_drift_tolerance,
        )
    ]

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        rationale = (
            "The commutator-aware value penalty probe cannot be interpreted "
            "because the microtest did not produce the promoted top-k-2 "
            "baseline, all controls, and both penalty rows with numeric gate "
            "metrics."
        )
        next_step = "repair the commutator-aware value penalty source artifact"
    elif qualifying_rows:
        status = "pass"
        decision = COMMUTATOR_VALUE_PENALTY_CANDIDATE_FOUND
        best = min(
            qualifying_rows,
            key=lambda row: float(row["commutator_anchor_logit_mse"]),
        )
        rationale = (
            f"`{best['variant']}` materially reduced absolute anchor "
            "commutator logit MSE while retaining transfer improvement, "
            "support usage, and anchor CE drift within the gate."
        )
        next_step = (
            "validate the qualifying commutator-aware value penalty candidate "
            "on RunPod with the same controls before treating it as evidence"
        )
    else:
        status = "pass"
        decision = COMMUTATOR_VALUE_PENALTY_NOT_ESTABLISHED
        rationale = (
            "The bounded residual-change penalty variants did not reduce "
            "absolute top-k-2 commutator logit MSE enough while preserving "
            "transfer improvement, support usage, and anchor CE drift. This "
            "keeps finite-update interference unresolved under the current "
            "local gate."
        )
        next_step = (
            "select a router-policy or explicit order-averaging mitigation "
            "only after recording that simple value penalties did not clear "
            "the commutator gate"
        )

    summary = {
        "status": status,
        "decision": decision,
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "source_rows": [
            {
                "source": "retention_churn_microtest",
                "path": str(out_dir / "variant_metrics.csv"),
                "present": (out_dir / "variant_metrics.csv").is_file(),
                "status": microtest.get("status"),
                "variant_count": len(variant_rows),
            }
        ],
        "thresholds": thresholds,
        "control_variants": list(_CONTROL_VARIANTS),
        "commutator_value_penalty_variants": list(_PENALTY_VARIANTS),
        "metrics": metrics,
        "commutator_value_penalty_rows": penalty_rows,
        "failures": failures,
        "rationale": rationale,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "commutator_value_penalty_rows_csv": str(
                out_dir / "commutator_value_penalty_rows.csv"
            ),
            "variant_metrics_csv": str(out_dir / "variant_metrics.csv"),
            "phase_metrics_csv": str(out_dir / "phase_metrics.csv"),
            "per_token_commutator_csv": str(out_dir / "per_token_commutator.csv"),
            "notes_md": str(out_dir / "commutator_value_penalty_notes.md"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "commutator_value_penalty_rows.csv", penalty_rows)
    _write_notes(out_dir / "commutator_value_penalty_notes.md", summary)
    return summary


def _penalty_row(variants: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    baseline = variants.get(_BASELINE_VARIANT, {})
    current = variants.get(name, {})
    return {
        "variant": name,
        "baseline_variant": _BASELINE_VARIANT,
        "commutator_value_penalty_weight": _float_or_none(
            current.get("commutator_value_penalty_weight")
        ),
        "commutator_anchor_logit_mse": _float_or_none(
            current.get("commutator_anchor_logit_mse")
        ),
        "baseline_commutator_anchor_logit_mse": _float_or_none(
            baseline.get("commutator_anchor_logit_mse")
        ),
        "commutator_anchor_logit_mse_reduction_fraction": _fractional_reduction(
            baseline.get("commutator_anchor_logit_mse"),
            current.get("commutator_anchor_logit_mse"),
        ),
        "commutator_transfer_logit_mse": _float_or_none(
            current.get("commutator_transfer_logit_mse")
        ),
        "transfer_ce_improvement": _float_or_none(current.get("transfer_ce_improvement")),
        "baseline_transfer_ce_improvement": _float_or_none(
            baseline.get("transfer_ce_improvement")
        ),
        "transfer_retention_fraction": _ratio(
            current.get("transfer_ce_improvement"),
            baseline.get("transfer_ce_improvement"),
        ),
        "anchor_used_columns_after_transfer": _float_or_none(
            current.get("anchor_used_columns_after_transfer")
        ),
        "baseline_anchor_used_columns_after_transfer": _float_or_none(
            baseline.get("anchor_used_columns_after_transfer")
        ),
        "support_usage_retention_fraction": _ratio(
            current.get("anchor_used_columns_after_transfer"),
            baseline.get("anchor_used_columns_after_transfer"),
        ),
        "anchor_ce_drift": _float_or_none(current.get("anchor_ce_drift")),
        "baseline_anchor_ce_drift": _float_or_none(baseline.get("anchor_ce_drift")),
        "anchor_support_churn_after_transfer": _float_or_none(
            current.get("anchor_support_churn_after_transfer")
        ),
        "commutator_anchor_support_churn": _float_or_none(
            current.get("commutator_anchor_support_churn")
        ),
        "commutator_anchor_residual_stream_l2": _float_or_none(
            current.get("commutator_anchor_residual_stream_l2")
        ),
    }


def _metrics(
    variants: dict[str, dict[str, Any]],
    penalty_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = variants.get(_BASELINE_VARIANT, {})
    controls = [variants[name] for name in _CONTROL_VARIANTS if name in variants]
    commutators = [
        float(row["commutator_anchor_logit_mse"])
        for row in penalty_rows
        if row["commutator_anchor_logit_mse"] is not None
    ]
    reductions = [
        float(row["commutator_anchor_logit_mse_reduction_fraction"])
        for row in penalty_rows
        if row["commutator_anchor_logit_mse_reduction_fraction"] is not None
    ]
    transfer_fractions = [
        float(row["transfer_retention_fraction"])
        for row in penalty_rows
        if row["transfer_retention_fraction"] is not None
    ]
    return {
        "baseline_commutator_anchor_logit_mse": _float_or_none(
            baseline.get("commutator_anchor_logit_mse")
        ),
        "baseline_transfer_ce_improvement": _float_or_none(
            baseline.get("transfer_ce_improvement")
        ),
        "baseline_anchor_used_columns_after_transfer": _float_or_none(
            baseline.get("anchor_used_columns_after_transfer")
        ),
        "baseline_anchor_ce_drift": _float_or_none(baseline.get("anchor_ce_drift")),
        "control_count": len(controls),
        "commutator_value_penalty_count": len(penalty_rows),
        "best_penalty_commutator_anchor_logit_mse": (
            min(commutators) if commutators else None
        ),
        "best_penalty_reduction_fraction": max(reductions) if reductions else None,
        "best_penalty_transfer_retention_fraction": (
            max(transfer_fractions) if transfer_fractions else None
        ),
    }


def _qualifies(
    row: dict[str, Any],
    *,
    commutator_reduction_fraction: float,
    transfer_retention_fraction: float,
    support_usage_retention_fraction: float,
    anchor_ce_drift_tolerance: float,
) -> bool:
    anchor_ce_drift = _float_or_none(row.get("anchor_ce_drift"))
    return (
        _at_least(
            row.get("commutator_anchor_logit_mse_reduction_fraction"),
            commutator_reduction_fraction,
        )
        and _at_least(row.get("transfer_retention_fraction"), transfer_retention_fraction)
        and _at_least(
            row.get("support_usage_retention_fraction"),
            support_usage_retention_fraction,
        )
        and anchor_ce_drift is not None
        and abs(anchor_ce_drift) <= anchor_ce_drift_tolerance
    )


def _failures(
    microtest: dict[str, Any],
    variants: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    failures = []
    if microtest.get("status") != "ok":
        failures.append(
            {"field": "microtest.status", "expected": "ok", "actual": microtest.get("status")}
        )
    for name in (_BASELINE_VARIANT, *_CONTROL_VARIANTS, *_PENALTY_VARIANTS):
        if name not in variants:
            failures.append({"field": "variant", "expected": name, "actual": "missing"})
    for name in (_BASELINE_VARIANT, *_PENALTY_VARIANTS):
        row = variants.get(name, {})
        for field in (
            "commutator_anchor_logit_mse",
            "transfer_ce_improvement",
            "anchor_used_columns_after_transfer",
            "anchor_ce_drift",
        ):
            if _float_or_none(row.get(field)) is None:
                failures.append(
                    {
                        "field": f"{name}.{field}",
                        "expected": "numeric",
                        "actual": row.get(field),
                    }
                )
    return failures


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    num = _float_or_none(numerator)
    den = _float_or_none(denominator)
    if num is None or den is None or abs(den) <= 1e-12:
        return None
    return num / den


def _fractional_reduction(baseline: Any, current: Any) -> float | None:
    base = _float_or_none(baseline)
    value = _float_or_none(current)
    if base is None or value is None or abs(base) <= 1e-12:
        return None
    return (base - value) / base


def _at_least(value: Any, threshold: float) -> bool:
    numeric = _float_or_none(value)
    return numeric is not None and numeric >= threshold


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Promoted Top-k-2 Commutator Value Penalty Probe",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Config: `{summary['config_path']}`",
        "- Baseline anchor commutator logit MSE: "
        f"`{summary['metrics']['baseline_commutator_anchor_logit_mse']}`",
        "- Best penalty reduction fraction: "
        f"`{summary['metrics']['best_penalty_reduction_fraction']}`",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_promoted_topk2_commutator_value_penalty_probe(
        config_path=args.config,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "metrics": summary["metrics"],
                "next_step": summary["next_step"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
