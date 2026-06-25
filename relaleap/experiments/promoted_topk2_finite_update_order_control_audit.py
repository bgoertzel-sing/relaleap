"""Finite-update order-control audit for promoted contextual top-k-2."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from relaleap.experiments.active_topk1_retention_churn_probe import (
    ACTIVE_TOPK1_RETENTION_CHURN_PROBE_ESTABLISHED,
)
from relaleap.experiments.promoted_topk2_functional_churn_control_audit import (
    FUNCTIONAL_CHURN_BOUNDED_WITH_COMMUTATOR_RISK,
)


DEFAULT_FUNCTIONAL_CHURN_DIR = Path(
    "results/reports/token_larger_promoted_topk2_functional_churn_control_audit"
)
DEFAULT_PROBE_DIRS = (
    Path("results/audits/token_larger_active_rank_matched_topk1_retention_churn_probe"),
    Path(
        "results/audits/token_larger_active_rank_matched_topk1_retention_churn_probe_seed2"
    ),
)
DEFAULT_FINGERPRINT_DIR = Path(
    "results/audits/token_larger_support_wide_promoted_default_causal_column_fingerprint_stability_topk1"
)
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_promoted_topk2_finite_update_order_control_audit"
)

FINITE_UPDATE_ORDER_SENSITIVITY_CE_BOUNDED = (
    "finite_update_order_sensitivity_ce_bounded_but_residual_material"
)
FINITE_UPDATE_ORDER_SENSITIVITY_FUNCTIONAL_HARM = (
    "finite_update_order_sensitivity_functional_harm"
)
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def run_promoted_topk2_finite_update_order_control_audit(
    *,
    functional_churn_dir: Path = DEFAULT_FUNCTIONAL_CHURN_DIR,
    probe_dirs: tuple[Path, ...] = DEFAULT_PROBE_DIRS,
    fingerprint_dir: Path = DEFAULT_FINGERPRINT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    low_ce_abs_delta_threshold: float = 0.05,
    material_logit_mse_threshold: float = 0.1,
    material_residual_l2_threshold: float = 3.0,
    high_support_churn_threshold: float = 0.5,
) -> dict[str, Any]:
    """Separate finite-update commutator harm from support identity churn."""

    functional_summary = _read_json_object(functional_churn_dir / "summary.json")
    packet_rows = [
        _packet_row(index, path) for index, path in enumerate(probe_dirs, start=1)
    ]
    variant_rows = [
        variant
        for packet in packet_rows
        for variant in packet.pop("variant_rows", [])
    ]
    token_rows = _read_csv_dicts(fingerprint_dir / "per_token_pair_interventions.csv")
    baseline_token_rows = [
        row
        for row in token_rows
        if row.get("variant") == "baseline"
        and row.get("intervention") == "fixed_dominant_router_support"
    ]
    token_strata_rows = _token_strata_rows(baseline_token_rows)
    token_correlation_rows = _token_correlation_rows(baseline_token_rows)
    source_rows = [
        _source_row(
            "functional_churn_control",
            functional_churn_dir / "summary.json",
            functional_summary,
        ),
        *[
            _source_row(f"retention_churn_probe_seed{index}", path / "summary.json")
            for index, path in enumerate(probe_dirs, start=1)
        ],
        _source_row(
            "causal_fingerprint_per_token",
            fingerprint_dir / "per_token_pair_interventions.csv",
        ),
    ]
    metrics = _metrics(variant_rows, token_strata_rows, token_correlation_rows)
    signals = {
        "functional_churn_control_bounded_with_commutator_risk": (
            functional_summary.get("decision")
            == FUNCTIONAL_CHURN_BOUNDED_WITH_COMMUTATOR_RISK
        ),
        "topk2_absolute_commutator_ce_bounded": _at_most(
            metrics.get("topk2_mean_commutator_anchor_ce_abs_delta"),
            low_ce_abs_delta_threshold,
        )
        and _at_most(
            metrics.get("topk2_mean_commutator_transfer_ce_abs_delta"),
            low_ce_abs_delta_threshold,
        ),
        "topk2_absolute_commutator_logit_material": _at_least(
            metrics.get("topk2_mean_commutator_anchor_logit_mse"),
            material_logit_mse_threshold,
        ),
        "topk2_absolute_commutator_residual_material": _at_least(
            metrics.get("topk2_mean_commutator_anchor_residual_stream_l2"),
            material_residual_l2_threshold,
        ),
        "topk2_commutator_support_churn_high": _at_least(
            metrics.get("topk2_mean_commutator_anchor_support_churn"),
            high_support_churn_threshold,
        ),
        "topk2_exceeds_topk1_commutator_logit": _positive(
            metrics.get("topk2_minus_topk1_mean_commutator_anchor_logit_mse")
        ),
        "topk2_exceeds_dense_commutator_logit": _positive(
            metrics.get("topk2_minus_dense_mean_commutator_anchor_logit_mse")
        ),
        "per_token_strata_available": bool(token_strata_rows),
        "per_token_commutator_ce_kl_available": False,
    }
    failures = _failures(
        functional_summary=functional_summary,
        source_rows=source_rows,
        packet_rows=packet_rows,
        variant_rows=variant_rows,
        token_strata_rows=token_strata_rows,
    )

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        rationale = (
            "The finite-update order-control audit cannot be interpreted because "
            "a required source packet, variant row, or per-token support-control "
            "artifact is missing or failing."
        )
        next_step = "repair missing finite-update order-control source artifacts"
    elif signals["topk2_absolute_commutator_ce_bounded"]:
        status = "pass"
        decision = FINITE_UPDATE_ORDER_SENSITIVITY_CE_BOUNDED
        rationale = (
            "The promoted top-k-2 finite-update commutator ratio is not just a "
            "small-denominator artifact: absolute top-k-2 logit MSE and residual "
            "stream L2 are materially above the rank-matched top-k-1 and dense "
            "controls. However, anchor/transfer CE absolute deltas remain below "
            "the guardrail across both packets. Current evidence therefore "
            "supports a bounded functional-harm interpretation, while keeping "
            "finite-update residual/order sensitivity as a real risk."
        )
        next_step = (
            "run a fresh task-free continual-learning anchor-retention matrix with "
            "promoted top-k-2, rank-matched top-k-1, random fixed top-k-2, and "
            "dense rank/FLOP-matched residual controls"
        )
    else:
        status = "pass"
        decision = FINITE_UPDATE_ORDER_SENSITIVITY_FUNCTIONAL_HARM
        rationale = (
            "Absolute finite-update CE movement is not bounded by the current "
            "guardrail, so the large commutator ratio should be treated as "
            "functional interference rather than support bookkeeping."
        )
        next_step = (
            "run a bounded local finite-update stabilization probe before making "
            "any causal-retention claim"
        )

    summary = {
        "status": status,
        "decision": decision,
        "out_dir": str(out_dir),
        "source_rows": source_rows,
        "thresholds": {
            "low_ce_abs_delta_threshold": low_ce_abs_delta_threshold,
            "material_logit_mse_threshold": material_logit_mse_threshold,
            "material_residual_l2_threshold": material_residual_l2_threshold,
            "high_support_churn_threshold": high_support_churn_threshold,
        },
        "packet_rows": packet_rows,
        "variant_rows": variant_rows,
        "token_strata_rows": token_strata_rows,
        "token_correlation_rows": token_correlation_rows,
        "metrics": metrics,
        "signals": signals,
        "source_limitations": [
            "Existing artifacts do not expose per-token finite-update commutator CE.",
            "Existing artifacts do not expose finite-update KL deltas.",
            "Fixed pre-update/post-update support replay rows are not present; "
            "previous-support controls are available only from the fingerprint "
            "functional-churn packet.",
        ],
        "failures": failures,
        "rationale": rationale,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "packet_metrics_csv": str(out_dir / "packet_metrics.csv"),
            "variant_commutator_csv": str(out_dir / "variant_commutator.csv"),
            "token_strata_csv": str(out_dir / "token_strata.csv"),
            "token_correlations_csv": str(out_dir / "token_correlations.csv"),
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
    _write_csv(out_dir / "variant_commutator.csv", variant_rows)
    _write_csv(out_dir / "token_strata.csv", token_strata_rows)
    _write_csv(out_dir / "token_correlations.csv", token_correlation_rows)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _packet_row(index: int, path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    summary = _read_json_object(summary_path)
    variants = summary.get("audit", {}).get("variants", [])
    variant_rows = [
        _variant_row(index, path, variant)
        for variant in variants
        if isinstance(variant, dict)
    ]
    return {
        "packet": f"seed{index}",
        "probe_dir": str(path),
        "summary_path": str(summary_path),
        "summary_present": summary_path.is_file(),
        "status": summary.get("status"),
        "decision": summary.get("decision"),
        "config_path": summary.get("config_path"),
        "variant_count": len(variant_rows),
        "variant_rows": variant_rows,
    }


def _variant_row(index: int, path: Path, variant: dict[str, Any]) -> dict[str, Any]:
    row = {
        "packet": f"seed{index}",
        "probe_dir": str(path),
        "variant": variant.get("variant"),
        "kind": variant.get("kind"),
        "top_k": _float_or_none(variant.get("top_k")),
        "num_columns": _float_or_none(variant.get("num_columns")),
        "active_parameters_proxy": _float_or_none(
            variant.get("active_parameters_proxy")
        ),
        "anchor_ce_drift": _float_or_none(variant.get("anchor_ce_drift")),
        "anchor_logit_mse_drift": _float_or_none(
            variant.get("anchor_logit_mse_drift")
        ),
        "anchor_residual_stream_l2_drift": _float_or_none(
            variant.get("anchor_residual_stream_l2_drift")
        ),
        "anchor_support_churn_after_transfer": _float_or_none(
            variant.get("anchor_support_churn_after_transfer")
        ),
        "commutator_anchor_ce_abs_delta": _float_or_none(
            variant.get("commutator_anchor_ce_abs_delta")
        ),
        "commutator_transfer_ce_abs_delta": _float_or_none(
            variant.get("commutator_transfer_ce_abs_delta")
        ),
        "commutator_anchor_logit_mse": _float_or_none(
            variant.get("commutator_anchor_logit_mse")
        ),
        "commutator_transfer_logit_mse": _float_or_none(
            variant.get("commutator_transfer_logit_mse")
        ),
        "commutator_anchor_residual_stream_l2": _float_or_none(
            variant.get("commutator_anchor_residual_stream_l2")
        ),
        "commutator_transfer_residual_stream_l2": _float_or_none(
            variant.get("commutator_transfer_residual_stream_l2")
        ),
        "commutator_anchor_support_churn": _float_or_none(
            variant.get("commutator_anchor_support_churn")
        ),
        "commutator_transfer_support_churn": _float_or_none(
            variant.get("commutator_transfer_support_churn")
        ),
    }
    return row


def _token_strata_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    strata = [
        ("all", lambda row: "all"),
        ("position_bin", lambda row: row.get("position_bin", "")),
        ("token_class", lambda row: row.get("token_class", "")),
        ("residual_norm_bin", lambda row: row.get("residual_norm_bin", "")),
        ("residual_gain_bin", lambda row: row.get("residual_gain_bin", "")),
    ]
    out: list[dict[str, Any]] = []
    for stratum, key_fn in strata:
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(key_fn(row) or "missing", []).append(row)
        for value, group in sorted(grouped.items()):
            out.append(
                {
                    "stratum": stratum,
                    "value": value,
                    "row_count": len(group),
                    "mean_fixed_support_logit_mse": _mean_csv_field(
                        group, "fixed_support_logit_mse"
                    ),
                    "mean_fixed_support_residual_stream_l2_delta": _mean_csv_field(
                        group, "fixed_support_residual_stream_l2_delta"
                    ),
                    "mean_fixed_support_loss_delta": _mean_csv_field(
                        group, "fixed_support_loss_delta"
                    ),
                    "mean_residual_norm": _mean_csv_field(group, "residual_norm"),
                    "mean_residual_gain": _mean_csv_field(group, "residual_gain"),
                    "mean_pair_synergy": _mean_csv_field(group, "pair_synergy"),
                    "mean_router_support_count": _mean_csv_field(
                        group, "router_support_count"
                    ),
                }
            )
    return out


def _token_correlation_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    left_fields = ("fixed_support_logit_mse", "fixed_support_residual_stream_l2_delta")
    right_fields = (
        "residual_norm",
        "residual_gain",
        "pair_synergy",
        "router_support_count",
        "fixed_support_loss_delta",
    )
    out = []
    for left in left_fields:
        for right in right_fields:
            pairs = [
                (_float_or_none(row.get(left)), _float_or_none(row.get(right)))
                for row in rows
            ]
            numeric_pairs = [
                (x, y) for x, y in pairs if x is not None and y is not None
            ]
            out.append(
                {
                    "left": left,
                    "right": right,
                    "row_count": len(numeric_pairs),
                    "pearson": _pearson(numeric_pairs),
                }
            )
    return out


def _metrics(
    variant_rows: list[dict[str, Any]],
    token_strata_rows: list[dict[str, Any]],
    token_correlation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    topk2 = _rows_for_variant(variant_rows, "promoted_contextual_topk2")
    topk1 = _rows_for_variant(variant_rows, "rank_matched_contextual_topk1")
    dense = _rows_for_variant(variant_rows, "norm_matched_dense_active_rank")
    topk2_logit = _mean_field(topk2, "commutator_anchor_logit_mse")
    topk1_logit = _mean_field(topk1, "commutator_anchor_logit_mse")
    dense_logit = _mean_field(dense, "commutator_anchor_logit_mse")
    all_stratum = _first(row for row in token_strata_rows if row["stratum"] == "all")
    return {
        "packet_count": len({row.get("packet") for row in variant_rows}),
        "variant_row_count": len(variant_rows),
        "topk2_mean_commutator_anchor_ce_abs_delta": _mean_field(
            topk2, "commutator_anchor_ce_abs_delta"
        ),
        "topk2_mean_commutator_transfer_ce_abs_delta": _mean_field(
            topk2, "commutator_transfer_ce_abs_delta"
        ),
        "topk1_mean_commutator_anchor_ce_abs_delta": _mean_field(
            topk1, "commutator_anchor_ce_abs_delta"
        ),
        "dense_mean_commutator_anchor_ce_abs_delta": _mean_field(
            dense, "commutator_anchor_ce_abs_delta"
        ),
        "topk2_mean_commutator_anchor_logit_mse": topk2_logit,
        "topk1_mean_commutator_anchor_logit_mse": topk1_logit,
        "dense_mean_commutator_anchor_logit_mse": dense_logit,
        "topk2_minus_topk1_mean_commutator_anchor_logit_mse": _delta(
            topk2_logit, topk1_logit
        ),
        "topk2_minus_dense_mean_commutator_anchor_logit_mse": _delta(
            topk2_logit, dense_logit
        ),
        "topk2_to_topk1_mean_commutator_anchor_logit_mse_ratio": _ratio(
            topk2_logit, topk1_logit
        ),
        "topk2_to_dense_mean_commutator_anchor_logit_mse_ratio": _ratio(
            topk2_logit, dense_logit
        ),
        "topk2_mean_commutator_anchor_residual_stream_l2": _mean_field(
            topk2, "commutator_anchor_residual_stream_l2"
        ),
        "topk1_mean_commutator_anchor_residual_stream_l2": _mean_field(
            topk1, "commutator_anchor_residual_stream_l2"
        ),
        "dense_mean_commutator_anchor_residual_stream_l2": _mean_field(
            dense, "commutator_anchor_residual_stream_l2"
        ),
        "topk2_mean_commutator_anchor_support_churn": _mean_field(
            topk2, "commutator_anchor_support_churn"
        ),
        "topk1_mean_commutator_anchor_support_churn": _mean_field(
            topk1, "commutator_anchor_support_churn"
        ),
        "token_strata_row_count": len(token_strata_rows),
        "token_correlation_row_count": len(token_correlation_rows),
        "per_token_fixed_support_logit_mse_mean": (
            all_stratum or {}
        ).get("mean_fixed_support_logit_mse"),
        "per_token_fixed_support_residual_l2_delta_mean": (
            all_stratum or {}
        ).get("mean_fixed_support_residual_stream_l2_delta"),
    }


def _failures(
    *,
    functional_summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    packet_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    token_strata_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows:
        if not row["present"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "artifact",
                    "expected": "present",
                    "actual": "missing",
                    "path": row["path"],
                }
            )
    if functional_summary.get("decision") != FUNCTIONAL_CHURN_BOUNDED_WITH_COMMUTATOR_RISK:
        failures.append(
            {
                "source": "functional_churn_control",
                "field": "decision",
                "expected": FUNCTIONAL_CHURN_BOUNDED_WITH_COMMUTATOR_RISK,
                "actual": functional_summary.get("decision"),
            }
        )
    for row in packet_rows:
        if row.get("status") != "pass":
            failures.append(
                {
                    "source": row.get("packet"),
                    "field": "status",
                    "expected": "pass",
                    "actual": row.get("status"),
                }
            )
        if row.get("decision") != ACTIVE_TOPK1_RETENTION_CHURN_PROBE_ESTABLISHED:
            failures.append(
                {
                    "source": row.get("packet"),
                    "field": "decision",
                    "expected": ACTIVE_TOPK1_RETENTION_CHURN_PROBE_ESTABLISHED,
                    "actual": row.get("decision"),
                }
            )
    required_variants = {
        "promoted_contextual_topk2",
        "rank_matched_contextual_topk1",
        "norm_matched_dense_active_rank",
    }
    for packet in {row.get("packet") for row in variant_rows}:
        present = {
            row.get("variant") for row in variant_rows if row.get("packet") == packet
        }
        missing = sorted(required_variants - present)
        if missing:
            failures.append(
                {
                    "source": packet,
                    "field": "variant_rows",
                    "expected": sorted(required_variants),
                    "actual_missing": missing,
                }
            )
    numeric_fields = (
        "commutator_anchor_ce_abs_delta",
        "commutator_transfer_ce_abs_delta",
        "commutator_anchor_logit_mse",
        "commutator_anchor_residual_stream_l2",
    )
    for row in variant_rows:
        for field in numeric_fields:
            if row.get(field) is None:
                failures.append(
                    {
                        "source": row.get("packet"),
                        "variant": row.get("variant"),
                        "field": field,
                        "expected": "numeric",
                        "actual": None,
                    }
                )
    if not token_strata_rows:
        failures.append(
            {
                "source": "causal_fingerprint_per_token",
                "field": "token_strata_rows",
                "expected": "at least one",
                "actual": 0,
            }
        )
    return failures


def _source_row(
    source: str, path: Path, value: dict[str, Any] | None = None
) -> dict[str, Any]:
    loaded = value if value is not None else (
        _read_json_object(path) if path.suffix == ".json" else {}
    )
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": loaded.get("status"),
        "decision": loaded.get("decision"),
    }


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    lines = [
        "# Promoted Top-k-2 Finite-Update Order-Control Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        "- Top-k-2 mean anchor commutator CE abs delta: "
        f"`{metrics['topk2_mean_commutator_anchor_ce_abs_delta']}`",
        "- Top-k-2 mean anchor commutator logit MSE: "
        f"`{metrics['topk2_mean_commutator_anchor_logit_mse']}`",
        "- Top-k-2/top-k-1 anchor commutator logit-MSE ratio: "
        f"`{metrics['topk2_to_topk1_mean_commutator_anchor_logit_mse_ratio']}`",
        "- Top-k-2/dense anchor commutator logit-MSE ratio: "
        f"`{metrics['topk2_to_dense_mean_commutator_anchor_logit_mse_ratio']}`",
        "- Top-k-2 mean anchor commutator residual-stream L2: "
        f"`{metrics['topk2_mean_commutator_anchor_residual_stream_l2']}`",
        "- Top-k-2 mean anchor commutator support churn: "
        f"`{metrics['topk2_mean_commutator_anchor_support_churn']}`",
        "",
        "## Signals",
        "",
    ]
    for key, value in summary["signals"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Source Limitations", ""])
    for limitation in summary["source_limitations"]:
        lines.append(f"- {limitation}")
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


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _rows_for_variant(
    rows: list[dict[str, Any]], variant: str
) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("variant") == variant]


def _mean_csv_field(rows: list[dict[str, str]], field: str) -> float | None:
    return _mean_values(_float_or_none(row.get(field)) for row in rows)


def _mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    return _mean_values(_float_or_none(row.get(field)) for row in rows)


def _mean_values(values: Any) -> float | None:
    numeric = [value for value in values if value is not None]
    return mean(numeric) if numeric else None


def _pearson(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_var * y_var)
    return numerator / denominator if denominator else None


def _first(values: Any) -> Any | None:
    for value in values:
        return value
    return None


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


def _at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def _at_most(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold


def _positive(value: float | None) -> bool:
    return value is not None and value > 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_promoted_topk2_finite_update_order_control_audit(out_dir=args.out)
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
