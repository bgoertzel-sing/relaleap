"""Decompose promoted top-k-2 finite-update order sensitivity by update group."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from relaleap.experiments.retention_churn_microtest import DEFAULT_CONFIG
from relaleap.experiments.retention_churn_microtest import run_retention_churn_microtest


DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_promoted_topk2_update_decomposition_audit"
)

VALUE_UPDATE_DOMINATED = "value_update_dominated_order_sensitivity"
ROUTER_UPDATE_DOMINATED = "router_update_dominated_order_sensitivity"
MIXED_UPDATE_SENSITIVITY = "mixed_update_order_sensitivity"
DECOMPOSITION_INSUFFICIENT = "decomposition_insufficient_evidence"

_BASELINE_VARIANT = "promoted_contextual_topk2"
_DECOMPOSITION_VARIANTS = (
    "router_only_transfer_topk2",
    "value_only_transfer_topk2",
)


def run_promoted_topk2_update_decomposition_audit(
    *,
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path = DEFAULT_OUT_DIR,
    material_fraction: float = 0.5,
) -> dict[str, Any]:
    """Run the retention microtest with router-only/value-only transfer rows."""

    out_dir.mkdir(parents=True, exist_ok=True)
    microtest = run_retention_churn_microtest(
        config_path,
        out_dir,
        include_decomposition_variants=True,
    )
    variant_rows = [
        row
        for row in microtest.get("audit", {}).get("variants", [])
        if isinstance(row, dict)
    ]
    variants = {str(row.get("variant")): row for row in variant_rows}
    decomposition_rows = [
        _decomposition_row(variants, name)
        for name in _DECOMPOSITION_VARIANTS
        if name in variants
    ]
    failures = _failures(microtest, variants)
    metrics = _metrics(variants, decomposition_rows)
    if failures:
        status = "fail"
        decision = DECOMPOSITION_INSUFFICIENT
        rationale = (
            "The update decomposition audit cannot be interpreted because the "
            "retention microtest did not produce the promoted top-k-2 baseline "
            "and both decomposition rows with numeric commutator metrics."
        )
        next_step = "repair the promoted top-k-2 update decomposition audit"
    else:
        status = "pass"
        decision = _decision(metrics, material_fraction)
        rationale = _rationale(decision, metrics, material_fraction)
        next_step = _next_step(decision)

    summary = {
        "status": status,
        "decision": decision,
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "thresholds": {
            "material_fraction_of_full_commutator": material_fraction,
        },
        "source_rows": [
            {
                "source": "retention_churn_microtest",
                "path": str(out_dir / "variant_metrics.csv"),
                "present": (out_dir / "variant_metrics.csv").is_file(),
                "status": microtest.get("status"),
                "variant_count": len(variant_rows),
            }
        ],
        "decomposition_variants": list(_DECOMPOSITION_VARIANTS),
        "metrics": metrics,
        "decomposition_rows": decomposition_rows,
        "failures": failures,
        "rationale": rationale,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "decomposition_rows_csv": str(out_dir / "decomposition_rows.csv"),
            "variant_metrics_csv": str(out_dir / "variant_metrics.csv"),
            "phase_metrics_csv": str(out_dir / "phase_metrics.csv"),
            "notes_md": str(out_dir / "decomposition_notes.md"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "decomposition_rows.csv", decomposition_rows)
    _write_notes(out_dir / "decomposition_notes.md", summary)
    return summary


def _decomposition_row(
    variants: dict[str, dict[str, Any]],
    name: str,
) -> dict[str, Any]:
    baseline = variants.get(_BASELINE_VARIANT, {})
    current = variants.get(name, {})
    return {
        "variant": name,
        "baseline_variant": _BASELINE_VARIANT,
        "transfer_update_group": current.get("transfer_update_group"),
        "commutator_anchor_logit_mse": _float_or_none(
            current.get("commutator_anchor_logit_mse")
        ),
        "baseline_commutator_anchor_logit_mse": _float_or_none(
            baseline.get("commutator_anchor_logit_mse")
        ),
        "commutator_anchor_fraction_of_full": _ratio(
            current.get("commutator_anchor_logit_mse"),
            baseline.get("commutator_anchor_logit_mse"),
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
        "anchor_ce_drift": _float_or_none(current.get("anchor_ce_drift")),
        "anchor_support_churn_after_transfer": _float_or_none(
            current.get("anchor_support_churn_after_transfer")
        ),
        "commutator_anchor_support_churn": _float_or_none(
            current.get("commutator_anchor_support_churn")
        ),
        "anchor_used_columns_after_transfer": _float_or_none(
            current.get("anchor_used_columns_after_transfer")
        ),
    }


def _metrics(
    variants: dict[str, dict[str, Any]],
    decomposition_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = variants.get(_BASELINE_VARIANT, {})
    rows_by_group = {
        str(row.get("transfer_update_group")): row for row in decomposition_rows
    }
    router = rows_by_group.get("router_only", {})
    value = rows_by_group.get("value_only", {})
    return {
        "baseline_commutator_anchor_logit_mse": _float_or_none(
            baseline.get("commutator_anchor_logit_mse")
        ),
        "baseline_transfer_ce_improvement": _float_or_none(
            baseline.get("transfer_ce_improvement")
        ),
        "router_only_commutator_anchor_logit_mse": _float_or_none(
            router.get("commutator_anchor_logit_mse")
        ),
        "router_only_fraction_of_full": _float_or_none(
            router.get("commutator_anchor_fraction_of_full")
        ),
        "router_only_transfer_retention_fraction": _float_or_none(
            router.get("transfer_retention_fraction")
        ),
        "value_only_commutator_anchor_logit_mse": _float_or_none(
            value.get("commutator_anchor_logit_mse")
        ),
        "value_only_fraction_of_full": _float_or_none(
            value.get("commutator_anchor_fraction_of_full")
        ),
        "value_only_transfer_retention_fraction": _float_or_none(
            value.get("transfer_retention_fraction")
        ),
    }


def _decision(metrics: dict[str, Any], material_fraction: float) -> str:
    router_fraction = _float_or_none(metrics.get("router_only_fraction_of_full"))
    value_fraction = _float_or_none(metrics.get("value_only_fraction_of_full"))
    if router_fraction is None or value_fraction is None:
        return DECOMPOSITION_INSUFFICIENT
    router_material = router_fraction >= material_fraction
    value_material = value_fraction >= material_fraction
    if value_material and not router_material:
        return VALUE_UPDATE_DOMINATED
    if router_material and not value_material:
        return ROUTER_UPDATE_DOMINATED
    return MIXED_UPDATE_SENSITIVITY


def _rationale(
    decision: str,
    metrics: dict[str, Any],
    material_fraction: float,
) -> str:
    router_fraction = metrics.get("router_only_fraction_of_full")
    value_fraction = metrics.get("value_only_fraction_of_full")
    if decision == VALUE_UPDATE_DOMINATED:
        return (
            "Value-only transfer updates preserve a material share of the full "
            f"commutator signal while router-only transfer updates fall below "
            f"the `{material_fraction}` fraction threshold."
        )
    if decision == ROUTER_UPDATE_DOMINATED:
        return (
            "Router-only transfer updates preserve a material share of the full "
            f"commutator signal while value-only transfer updates fall below "
            f"the `{material_fraction}` fraction threshold."
        )
    if decision == MIXED_UPDATE_SENSITIVITY:
        return (
            "Router-only and value-only transfer rows do not isolate a single "
            "dominant source under the current threshold. Router/full fraction "
            f"is `{router_fraction}` and value/full fraction is `{value_fraction}`."
        )
    return "Required decomposition metrics are missing or nonnumeric."


def _next_step(decision: str) -> str:
    if decision == VALUE_UPDATE_DOMINATED:
        return "test value-update regularization or lower-rank value updates before changing router policy"
    if decision == ROUTER_UPDATE_DOMINATED:
        return "test router-logit anchoring or router distillation before changing value updates"
    if decision == MIXED_UPDATE_SENSITIVITY:
        return "run one RunPod validation of the decomposition audit before selecting a mitigation family"
    return "repair missing decomposition source artifacts"


def _failures(
    microtest: dict[str, Any],
    variants: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    failures = []
    if microtest.get("status") != "ok":
        failures.append(
            {"field": "microtest.status", "expected": "ok", "actual": microtest.get("status")}
        )
    for name in (_BASELINE_VARIANT, *_DECOMPOSITION_VARIANTS):
        if name not in variants:
            failures.append({"field": "variant", "expected": name, "actual": "missing"})
    for name in (_BASELINE_VARIANT, *_DECOMPOSITION_VARIANTS):
        row = variants.get(name, {})
        for field in ("commutator_anchor_logit_mse", "transfer_ce_improvement"):
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
        "# Promoted Top-k-2 Update Decomposition Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Config: `{summary['config_path']}`",
        "- Full anchor commutator logit MSE: "
        f"`{summary['metrics']['baseline_commutator_anchor_logit_mse']}`",
        "- Router-only fraction of full: "
        f"`{summary['metrics']['router_only_fraction_of_full']}`",
        "- Value-only fraction of full: "
        f"`{summary['metrics']['value_only_fraction_of_full']}`",
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
    parser.add_argument("--material-fraction", type=float, default=0.5)
    args = parser.parse_args(argv)
    summary = run_promoted_topk2_update_decomposition_audit(
        config_path=args.config,
        out_dir=args.out,
        material_fraction=args.material_fraction,
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
