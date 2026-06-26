"""Low-rank value-update gate for promoted contextual top-k-2 order risk."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from relaleap.experiments.retention_churn_microtest import DEFAULT_CONFIG
from relaleap.experiments.retention_churn_microtest import run_retention_churn_microtest


DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_promoted_topk2_low_rank_value_gate"
)

LOW_RANK_VALUE_CANDIDATE_FOUND = "low_rank_value_candidate_found"
LOW_RANK_VALUE_NOT_ESTABLISHED = "low_rank_value_not_established"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

_BASELINE_VARIANT = "promoted_contextual_topk2"
_CONTROL_VARIANTS = (
    "rank_matched_contextual_topk1",
    "random_fixed_topk2",
    "norm_matched_dense_active_rank",
)
_LOW_RANK_VARIANTS = (
    "value_gradient_rank1_contextual_topk2",
    "value_gradient_rank2_contextual_topk2",
)


def run_promoted_topk2_low_rank_value_gate(
    *,
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path = DEFAULT_OUT_DIR,
    commutator_reduction_fraction: float = 0.5,
    transfer_retention_fraction: float = 0.8,
    support_usage_retention_fraction: float = 0.8,
) -> dict[str, Any]:
    """Run and interpret low-rank value-gradient variants plus controls."""

    out_dir.mkdir(parents=True, exist_ok=True)
    microtest = run_retention_churn_microtest(
        config_path,
        out_dir,
        include_low_rank_value_variants=True,
    )
    variant_rows = [
        row
        for row in microtest.get("audit", {}).get("variants", [])
        if isinstance(row, dict)
    ]
    variants = {str(row.get("variant")): row for row in variant_rows}
    low_rank_rows = [
        _low_rank_row(variants, name) for name in _LOW_RANK_VARIANTS if name in variants
    ]
    thresholds = {
        "commutator_reduction_fraction": commutator_reduction_fraction,
        "transfer_retention_fraction": transfer_retention_fraction,
        "support_usage_retention_fraction": support_usage_retention_fraction,
    }
    failures = _failures(microtest, variants)
    metrics = _metrics(variants, low_rank_rows)
    qualifying_rows = [
        row
        for row in low_rank_rows
        if _qualifies(
            row,
            commutator_reduction_fraction=commutator_reduction_fraction,
            transfer_retention_fraction=transfer_retention_fraction,
            support_usage_retention_fraction=support_usage_retention_fraction,
        )
    ]

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        rationale = (
            "The low-rank value-update gate cannot be interpreted because "
            "the microtest did not produce the promoted top-k-2 baseline, all "
            "controls, and both low-rank value rows with numeric gate metrics."
        )
        next_step = "repair the low-rank value-update gate source artifact"
    elif qualifying_rows:
        status = "pass"
        decision = LOW_RANK_VALUE_CANDIDATE_FOUND
        best = min(
            qualifying_rows,
            key=lambda row: float(row["commutator_anchor_logit_mse"]),
        )
        rationale = (
            f"`{best['variant']}` materially reduced absolute anchor "
            "commutator logit MSE while retaining transfer improvement and "
            "support usage relative to promoted contextual top-k-2."
        )
        next_step = (
            "validate the qualifying low-rank value-update candidate on RunPod "
            "with the same controls before treating it as evidence"
        )
    else:
        status = "pass"
        decision = LOW_RANK_VALUE_NOT_ESTABLISHED
        rationale = (
            "The bounded low-rank value-gradient variants did not reduce "
            "absolute top-k-2 commutator logit MSE enough while preserving "
            "transfer improvement and support usage. This rejects these "
            "low-rank value-update projections as a promoted fix under the "
            "current gate."
        )
        next_step = (
            "return to the commutator-aware penalty branch or a router-policy "
            "mitigation only after recording why low-rank value updates failed"
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
        "low_rank_variants": list(_LOW_RANK_VARIANTS),
        "metrics": metrics,
        "low_rank_rows": low_rank_rows,
        "failures": failures,
        "rationale": rationale,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "low_rank_rows_csv": str(out_dir / "low_rank_rows.csv"),
            "variant_metrics_csv": str(out_dir / "variant_metrics.csv"),
            "phase_metrics_csv": str(out_dir / "phase_metrics.csv"),
            "notes_md": str(out_dir / "low_rank_value_notes.md"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "low_rank_rows.csv", low_rank_rows)
    _write_notes(out_dir / "low_rank_value_notes.md", summary)
    return summary


def _low_rank_row(variants: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    baseline = variants.get(_BASELINE_VARIANT, {})
    current = variants.get(name, {})
    return {
        "variant": name,
        "baseline_variant": _BASELINE_VARIANT,
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
        "anchor_support_churn_after_transfer": _float_or_none(
            current.get("anchor_support_churn_after_transfer")
        ),
        "commutator_anchor_support_churn": _float_or_none(
            current.get("commutator_anchor_support_churn")
        ),
        "value_gradient_low_rank": current.get("value_gradient_low_rank"),
    }


def _metrics(
    variants: dict[str, dict[str, Any]],
    low_rank_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = variants.get(_BASELINE_VARIANT, {})
    controls = [variants[name] for name in _CONTROL_VARIANTS if name in variants]
    commutators = [
        float(row["commutator_anchor_logit_mse"])
        for row in low_rank_rows
        if row["commutator_anchor_logit_mse"] is not None
    ]
    reductions = [
        float(row["commutator_anchor_logit_mse_reduction_fraction"])
        for row in low_rank_rows
        if row["commutator_anchor_logit_mse_reduction_fraction"] is not None
    ]
    transfer_fractions = [
        float(row["transfer_retention_fraction"])
        for row in low_rank_rows
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
        "control_count": len(controls),
        "low_rank_count": len(low_rank_rows),
        "best_low_rank_commutator_anchor_logit_mse": (
            min(commutators) if commutators else None
        ),
        "best_low_rank_reduction_fraction": max(reductions) if reductions else None,
        "best_low_rank_transfer_retention_fraction": (
            max(transfer_fractions) if transfer_fractions else None
        ),
    }


def _qualifies(
    row: dict[str, Any],
    *,
    commutator_reduction_fraction: float,
    transfer_retention_fraction: float,
    support_usage_retention_fraction: float,
) -> bool:
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
    for name in (_BASELINE_VARIANT, *_CONTROL_VARIANTS, *_LOW_RANK_VARIANTS):
        if name not in variants:
            failures.append({"field": "variant", "expected": name, "actual": "missing"})
    for name in (_BASELINE_VARIANT, *_LOW_RANK_VARIANTS):
        row = variants.get(name, {})
        for field in (
            "commutator_anchor_logit_mse",
            "transfer_ce_improvement",
            "anchor_used_columns_after_transfer",
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
        "# Promoted Top-k-2 Low-rank Value Gate",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Config: `{summary['config_path']}`",
        "- Baseline anchor commutator logit MSE: "
        f"`{summary['metrics']['baseline_commutator_anchor_logit_mse']}`",
        "- Best low-rank reduction fraction: "
        f"`{summary['metrics']['best_low_rank_reduction_fraction']}`",
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
    summary = run_promoted_topk2_low_rank_value_gate(
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
