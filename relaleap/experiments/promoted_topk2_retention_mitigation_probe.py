"""Bounded mitigation probe for promoted contextual top-k-2 retention risk."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from relaleap.experiments.retention_churn_microtest import DEFAULT_CONFIG
from relaleap.experiments.retention_churn_microtest import run_retention_churn_microtest


DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_promoted_topk2_retention_mitigation_probe"
)

RETENTION_MITIGATION_CANDIDATE_FOUND = "retention_mitigation_candidate_found"
RETENTION_MITIGATION_NOT_ESTABLISHED = "retention_mitigation_not_established"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

_BASELINE_VARIANT = "promoted_contextual_topk2"
_CONTROL_VARIANTS = (
    "rank_matched_contextual_topk1",
    "random_fixed_topk2",
    "norm_matched_dense_active_rank",
)
_MITIGATION_VARIANTS = (
    "router_frozen_transfer_topk2",
    "gradient_clipped_contextual_topk2",
)


def run_promoted_topk2_retention_mitigation_probe(
    *,
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path = DEFAULT_OUT_DIR,
    commutator_reduction_fraction: float = 0.5,
    transfer_retention_fraction: float = 0.8,
    support_usage_retention_fraction: float = 0.8,
) -> dict[str, Any]:
    """Run and interpret a small mitigation matrix over the retention microtest."""

    out_dir.mkdir(parents=True, exist_ok=True)
    microtest = run_retention_churn_microtest(
        config_path,
        out_dir,
        include_mitigation_variants=True,
    )
    variant_rows = [
        row
        for row in microtest.get("audit", {}).get("variants", [])
        if isinstance(row, dict)
    ]
    variants = {str(row.get("variant")): row for row in variant_rows}
    mitigation_rows = [
        _mitigation_row(variants, name)
        for name in _MITIGATION_VARIANTS
        if name in variants
    ]
    source_rows = [
        {
            "source": "retention_churn_microtest",
            "path": str(out_dir / "variant_metrics.csv"),
            "present": (out_dir / "variant_metrics.csv").is_file(),
            "status": microtest.get("status"),
            "variant_count": len(variant_rows),
        }
    ]
    metrics = _metrics(variants, mitigation_rows)
    thresholds = {
        "commutator_reduction_fraction": commutator_reduction_fraction,
        "transfer_retention_fraction": transfer_retention_fraction,
        "support_usage_retention_fraction": support_usage_retention_fraction,
    }
    failures = _failures(microtest, variants)
    qualifying_rows = [
        row
        for row in mitigation_rows
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
            "The mitigation probe cannot be interpreted because the extended "
            "retention microtest did not produce the baseline, controls, and "
            "mitigation rows required by the gate."
        )
        next_step = "repair the retention mitigation probe source artifact"
    elif qualifying_rows:
        status = "pass"
        decision = RETENTION_MITIGATION_CANDIDATE_FOUND
        best = min(
            qualifying_rows,
            key=lambda row: float(row["commutator_anchor_logit_mse"]),
        )
        rationale = (
            f"`{best['variant']}` reduced absolute anchor commutator logit MSE "
            "by the required fraction while retaining transfer improvement and "
            "support usage relative to promoted contextual top-k-2 in this "
            "bounded local probe."
        )
        next_step = (
            "validate the qualifying retention mitigation candidate on RunPod "
            "with the same controls before treating it as scientific evidence"
        )
    else:
        status = "pass"
        decision = RETENTION_MITIGATION_NOT_ESTABLISHED
        rationale = (
            "Neither bounded mitigation variant reduced absolute top-k-2 "
            "commutator logit MSE enough while preserving transfer improvement "
            "and support usage. The probe does not yet support router-stability "
            "or simple update-clipping as a promoted retention fix."
        )
        next_step = (
            "run a decomposition audit separating router-only, value-only, and "
            "full updates for the promoted contextual top-k-2 retention packet"
        )

    summary = {
        "status": status,
        "decision": decision,
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "source_rows": source_rows,
        "thresholds": thresholds,
        "control_variants": list(_CONTROL_VARIANTS),
        "mitigation_variants": list(_MITIGATION_VARIANTS),
        "metrics": metrics,
        "mitigation_rows": mitigation_rows,
        "failures": failures,
        "rationale": rationale,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "mitigation_rows_csv": str(out_dir / "mitigation_rows.csv"),
            "variant_metrics_csv": str(out_dir / "variant_metrics.csv"),
            "phase_metrics_csv": str(out_dir / "phase_metrics.csv"),
            "notes_md": str(out_dir / "mitigation_notes.md"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "mitigation_rows.csv", mitigation_rows)
    _write_notes(out_dir / "mitigation_notes.md", summary)
    return summary


def _mitigation_row(variants: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
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
        "anchor_support_churn_after_transfer": _float_or_none(
            current.get("anchor_support_churn_after_transfer")
        ),
        "commutator_anchor_support_churn": _float_or_none(
            current.get("commutator_anchor_support_churn")
        ),
        "freeze_router_during_transfer": current.get("freeze_router_during_transfer"),
        "gradient_clip_norm": current.get("gradient_clip_norm"),
    }


def _metrics(
    variants: dict[str, dict[str, Any]],
    mitigation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = variants.get(_BASELINE_VARIANT, {})
    controls = [variants[name] for name in _CONTROL_VARIANTS if name in variants]
    mitigation_commutators = [
        float(row["commutator_anchor_logit_mse"])
        for row in mitigation_rows
        if row["commutator_anchor_logit_mse"] is not None
    ]
    mitigation_reductions = [
        float(row["commutator_anchor_logit_mse_reduction_fraction"])
        for row in mitigation_rows
        if row["commutator_anchor_logit_mse_reduction_fraction"] is not None
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
        "mitigation_count": len(mitigation_rows),
        "best_mitigation_commutator_anchor_logit_mse": (
            min(mitigation_commutators) if mitigation_commutators else None
        ),
        "best_mitigation_reduction_fraction": (
            max(mitigation_reductions) if mitigation_reductions else None
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
    for name in (_BASELINE_VARIANT, *_CONTROL_VARIANTS, *_MITIGATION_VARIANTS):
        if name not in variants:
            failures.append(
                {"field": "variant", "expected": name, "actual": "missing"}
            )
    for name in (_BASELINE_VARIANT, *_MITIGATION_VARIANTS):
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
        "# Promoted Top-k-2 Retention Mitigation Probe",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Config: `{summary['config_path']}`",
        "- Baseline anchor commutator logit MSE: "
        f"`{summary['metrics']['baseline_commutator_anchor_logit_mse']}`",
        "- Best mitigation reduction fraction: "
        f"`{summary['metrics']['best_mitigation_reduction_fraction']}`",
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
    summary = run_promoted_topk2_retention_mitigation_probe(
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
