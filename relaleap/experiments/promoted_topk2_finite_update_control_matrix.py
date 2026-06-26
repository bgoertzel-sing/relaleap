"""No-training finite-update causal fingerprint/control matrix."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path
from typing import Any, Callable


DEFAULT_FINITE_UPDATE_REPORT_DIR = Path(
    "results/reports/token_larger_promoted_topk2_finite_update_order_control_audit"
)
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_promoted_topk2_finite_update_control_matrix"
)

FINITE_UPDATE_CONTROL_MATRIX_READY = "finite_update_control_matrix_ready"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

ROLE_BY_VARIANT = {
    "promoted_contextual_topk2": "promoted_contextual_topk2",
    "rank_matched_contextual_topk1": "rank_matched_contextual_topk1",
    "random_fixed_topk2": "random_fixed_topk2",
    "norm_matched_dense_active_rank": "dense_active_rank",
}
REQUIRED_VARIANTS = tuple(ROLE_BY_VARIANT)
REQUIRED_FIELDS = {
    "variant",
    "split",
    "batch_index",
    "position_index",
    "position_bin",
    "target_token",
    "token_class",
    "forward_ce",
    "reverse_ce",
    "ce_delta_forward_minus_reverse",
    "ce_abs_delta",
    "symmetric_kl",
    "logit_mse",
    "residual_delta_l2",
    "residual_norm",
    "residual_norm_bin",
    "residual_delta_l2_bin",
    "support_churn",
    "forward_support",
    "reverse_support",
}
NUMERIC_FIELDS = (
    "forward_ce",
    "reverse_ce",
    "ce_delta_forward_minus_reverse",
    "ce_abs_delta",
    "symmetric_kl",
    "logit_mse",
    "residual_delta_l2",
    "residual_norm",
)


def run_promoted_topk2_finite_update_control_matrix(
    *,
    finite_update_report_dir: Path = DEFAULT_FINITE_UPDATE_REPORT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    high_support_churn_threshold: float = 0.5,
) -> dict[str, Any]:
    """Build a fail-closed per-token finite-update control matrix."""

    start = time.time()
    summary_path = finite_update_report_dir / "summary.json"
    failures: list[dict[str, Any]] = []
    source_summary = _read_json_object(summary_path)
    if not summary_path.is_file():
        failures.append(
            {
                "field": "finite_update_summary",
                "expected": str(summary_path),
                "actual": "missing",
            }
        )

    source_rows = _source_rows(source_summary)
    per_token_rows: list[dict[str, Any]] = []
    for source in source_rows:
        path = Path(str(source["probe_dir"])) / "per_token_commutator.csv"
        if not path.is_file():
            source["per_token_commutator_present"] = False
            continue
        rows, missing_fields = _read_per_token_rows(path)
        source["per_token_commutator_present"] = True
        source["per_token_row_count"] = len(rows)
        if missing_fields:
            failures.append(
                {
                    "field": "per_token_commutator_fields",
                    "source": source["packet"],
                    "path": str(path),
                    "missing": missing_fields,
                }
            )
        for row in rows:
            variant = str(row.get("variant", ""))
            enriched = dict(row)
            enriched["packet"] = source["packet"]
            enriched["probe_dir"] = source["probe_dir"]
            enriched["matrix_role"] = ROLE_BY_VARIANT.get(variant, "unmapped")
            enriched["support_transition"] = (
                f"{row.get('forward_support', '')}->{row.get('reverse_support', '')}"
            )
            per_token_rows.append(enriched)

    variants_present = {str(row.get("variant")) for row in per_token_rows}
    for variant in REQUIRED_VARIANTS:
        if variant not in variants_present:
            failures.append(
                {
                    "field": "required_variant",
                    "expected": variant,
                    "actual": "missing",
                }
            )
    if not per_token_rows:
        failures.append(
            {
                "field": "per_token_commutator_rows",
                "expected": "at least one row",
                "actual": 0,
            }
        )

    matrix_rows = _matrix_rows(per_token_rows)
    strata_rows = _strata_rows(per_token_rows)
    metrics = _metrics(matrix_rows, strata_rows)
    signals = {
        "all_required_variants_present": not any(
            failure["field"] == "required_variant" for failure in failures
        ),
        "required_per_token_fields_present": not any(
            failure["field"] == "per_token_commutator_fields"
            for failure in failures
        ),
        "topk2_support_churn_high": _at_least(
            metrics.get("topk2_support_churn_fraction"),
            high_support_churn_threshold,
        ),
        "topk2_logit_mse_exceeds_topk1": _positive(
            _delta(
                metrics.get("topk2_mean_logit_mse"),
                metrics.get("topk1_mean_logit_mse"),
            )
        ),
        "topk2_symmetric_kl_exceeds_topk1": _positive(
            _delta(
                metrics.get("topk2_mean_symmetric_kl"),
                metrics.get("topk1_mean_symmetric_kl"),
            )
        ),
        "topk2_ce_abs_delta_below_random_fixed_topk2": _negative(
            _delta(
                metrics.get("topk2_mean_ce_abs_delta"),
                metrics.get("random_fixed_topk2_mean_ce_abs_delta"),
            )
        ),
        "dense_control_has_per_token_rows": metrics.get(
            "dense_active_rank_row_count"
        )
        is not None,
    }

    status = "fail" if failures else "pass"
    decision = (
        INSUFFICIENT_EVIDENCE if failures else FINITE_UPDATE_CONTROL_MATRIX_READY
    )
    next_step = (
        "repair missing per-token finite-update control artifacts"
        if failures
        else (
            "use this matrix to extend the causal fingerprint/control audit; "
            "top-k-2 causal-cooperation language remains blocked until finite-update "
            "risk is matched against functional benefit under these strata"
        )
    )
    summary = {
        "status": status,
        "decision": decision,
        "finite_update_report_dir": str(finite_update_report_dir),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "thresholds": {
            "high_support_churn_threshold": high_support_churn_threshold,
        },
        "source_rows": source_rows,
        "metrics": metrics,
        "signals": signals,
        "failures": failures,
        "matrix_rows": matrix_rows,
        "strata_row_count": len(strata_rows),
        "claim_gate": (
            "matrix_input_only_not_causal_cooperation_evidence"
        ),
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "finite_update_control_matrix_csv": str(
                out_dir / "finite_update_control_matrix.csv"
            ),
            "finite_update_control_strata_csv": str(
                out_dir / "finite_update_control_strata.csv"
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
    _write_csv(out_dir / "finite_update_control_matrix.csv", matrix_rows)
    _write_csv(out_dir / "finite_update_control_strata.csv", strata_rows)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _source_rows(source_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for packet in [
        *source_summary.get("packet_rows", []),
        *source_summary.get("microtest_packet_rows", []),
    ]:
        if not isinstance(packet, dict):
            continue
        rows.append(
            {
                "packet": packet.get("packet"),
                "probe_dir": packet.get("probe_dir"),
                "status": packet.get("status"),
                "decision": packet.get("decision"),
                "config_path": packet.get("config_path"),
            }
        )
    return rows


def _read_per_token_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_FIELDS - set(reader.fieldnames or []))
        return list(reader), missing


def _matrix_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for variant in REQUIRED_VARIANTS:
        role = ROLE_BY_VARIANT[variant]
        variant_rows = [row for row in rows if row.get("variant") == variant]
        for split in ["all", *sorted({str(row.get("split")) for row in variant_rows})]:
            group = (
                variant_rows
                if split == "all"
                else [row for row in variant_rows if row.get("split") == split]
            )
            if not group:
                continue
            out.append(_aggregate_row(group, matrix_role=role, variant=variant, split=split))
    return out


def _strata_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strata: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
        ("all", lambda row: "all"),
        ("matrix_role", lambda row: str(row.get("matrix_role", ""))),
        ("variant", lambda row: str(row.get("variant", ""))),
        ("split", lambda row: str(row.get("split", ""))),
        ("position_bin", lambda row: str(row.get("position_bin", ""))),
        ("token_class", lambda row: str(row.get("token_class", ""))),
        ("residual_norm_bin", lambda row: str(row.get("residual_norm_bin", ""))),
        (
            "residual_delta_l2_bin",
            lambda row: str(row.get("residual_delta_l2_bin", "")),
        ),
        ("support_churn", lambda row: str(row.get("support_churn", ""))),
        ("forward_support", lambda row: str(row.get("forward_support", ""))),
        ("reverse_support", lambda row: str(row.get("reverse_support", ""))),
        ("support_transition", lambda row: str(row.get("support_transition", ""))),
        (
            "matrix_role_x_position_bin",
            lambda row: f"{row.get('matrix_role', '')}|{row.get('position_bin', '')}",
        ),
        (
            "matrix_role_x_support_churn",
            lambda row: f"{row.get('matrix_role', '')}|{row.get('support_churn', '')}",
        ),
        (
            "matrix_role_x_residual_delta_l2_bin",
            lambda row: (
                f"{row.get('matrix_role', '')}|"
                f"{row.get('residual_delta_l2_bin', '')}"
            ),
        ),
    ]
    out = []
    for stratum, key_fn in strata:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(key_fn(row) or "missing", []).append(row)
        for value, group in sorted(groups.items()):
            out.append(_aggregate_row(group, stratum=stratum, value=value))
    return out


def _aggregate_row(
    rows: list[dict[str, Any]],
    *,
    matrix_role: str | None = None,
    variant: str | None = None,
    split: str | None = None,
    stratum: str | None = None,
    value: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "row_count": len(rows),
        "source_packet_count": len({row.get("packet") for row in rows}),
        "unique_forward_support_count": len(
            {row.get("forward_support") for row in rows if row.get("forward_support")}
        ),
        "unique_reverse_support_count": len(
            {row.get("reverse_support") for row in rows if row.get("reverse_support")}
        ),
        "unique_support_transition_count": len(
            {
                row.get("support_transition")
                for row in rows
                if row.get("support_transition")
            }
        ),
        "support_churn_fraction": _true_fraction(rows, "support_churn"),
    }
    if matrix_role is not None:
        row.update({"matrix_role": matrix_role, "variant": variant, "split": split})
    if stratum is not None:
        row.update({"stratum": stratum, "value": value})
    for field in NUMERIC_FIELDS:
        row[f"mean_{field}"] = _mean_csv_field(rows, field)
    return row


def _metrics(
    matrix_rows: list[dict[str, Any]],
    strata_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_role = {
        row.get("matrix_role"): row
        for row in matrix_rows
        if row.get("split") == "all"
    }
    topk2 = by_role.get("promoted_contextual_topk2", {})
    topk1 = by_role.get("rank_matched_contextual_topk1", {})
    random_topk2 = by_role.get("random_fixed_topk2", {})
    dense = by_role.get("dense_active_rank", {})
    return {
        "matrix_row_count": len(matrix_rows),
        "strata_row_count": len(strata_rows),
        "topk2_row_count": topk2.get("row_count"),
        "topk1_row_count": topk1.get("row_count"),
        "random_fixed_topk2_row_count": random_topk2.get("row_count"),
        "dense_active_rank_row_count": dense.get("row_count"),
        "topk2_mean_ce_abs_delta": topk2.get("mean_ce_abs_delta"),
        "topk1_mean_ce_abs_delta": topk1.get("mean_ce_abs_delta"),
        "random_fixed_topk2_mean_ce_abs_delta": random_topk2.get(
            "mean_ce_abs_delta"
        ),
        "dense_active_rank_mean_ce_abs_delta": dense.get("mean_ce_abs_delta"),
        "topk2_mean_logit_mse": topk2.get("mean_logit_mse"),
        "topk1_mean_logit_mse": topk1.get("mean_logit_mse"),
        "random_fixed_topk2_mean_logit_mse": random_topk2.get("mean_logit_mse"),
        "dense_active_rank_mean_logit_mse": dense.get("mean_logit_mse"),
        "topk2_mean_symmetric_kl": topk2.get("mean_symmetric_kl"),
        "topk1_mean_symmetric_kl": topk1.get("mean_symmetric_kl"),
        "topk2_mean_residual_delta_l2": topk2.get("mean_residual_delta_l2"),
        "topk1_mean_residual_delta_l2": topk1.get("mean_residual_delta_l2"),
        "topk2_support_churn_fraction": topk2.get("support_churn_fraction"),
        "topk1_support_churn_fraction": topk1.get("support_churn_fraction"),
        "topk2_unique_support_transition_count": topk2.get(
            "unique_support_transition_count"
        ),
        "topk2_minus_topk1_logit_mse": _delta(
            topk2.get("mean_logit_mse"),
            topk1.get("mean_logit_mse"),
        ),
        "topk2_minus_dense_logit_mse": _delta(
            topk2.get("mean_logit_mse"),
            dense.get("mean_logit_mse"),
        ),
        "topk2_minus_random_fixed_topk2_logit_mse": _delta(
            topk2.get("mean_logit_mse"),
            random_topk2.get("mean_logit_mse"),
        ),
    }


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Promoted top-k-2 finite-update control matrix",
        "",
        f"Status: `{summary['status']}`",
        f"Decision: `{summary['decision']}`",
        "",
        "This is a no-training artifact audit. It aggregates raw per-token "
        "forward-vs-reverse finite-update commutator rows across promoted "
        "contextual top-k-2 and the rank-matched top-k-1, random fixed top-k-2, "
        "and dense active-rank controls.",
        "",
        f"Claim gate: `{summary['claim_gate']}`",
        f"Next step: {summary['next_step']}",
    ]
    if summary["failures"]:
        lines.extend(["", "## Failures"])
        lines.extend(f"- `{failure}`" for failure in summary["failures"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _mean_csv_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [
        value
        for value in (_float_or_none(row.get(field)) for row in rows)
        if value is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _true_fraction(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [str(row.get(field, "")).strip().lower() for row in rows]
    present = [value for value in values if value in {"true", "false"}]
    if not present:
        return None
    return sum(value == "true" for value in present) / len(present)


def _delta(left: Any, right: Any) -> float | None:
    left_float = _float_or_none(left)
    right_float = _float_or_none(right)
    if left_float is None or right_float is None:
        return None
    return left_float - right_float


def _at_least(value: Any, threshold: float) -> bool:
    numeric = _float_or_none(value)
    return numeric is not None and numeric >= threshold


def _positive(value: Any) -> bool:
    numeric = _float_or_none(value)
    return numeric is not None and numeric > 0.0


def _negative(value: Any) -> bool:
    numeric = _float_or_none(value)
    return numeric is not None and numeric < 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finite-update-report-dir",
        type=Path,
        default=DEFAULT_FINITE_UPDATE_REPORT_DIR,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_promoted_topk2_finite_update_control_matrix(
        finite_update_report_dir=args.finite_update_report_dir,
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
