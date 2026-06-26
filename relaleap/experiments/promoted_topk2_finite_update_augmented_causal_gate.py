"""Finite-update-augmented top-k-2 causal-cooperation gate."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from relaleap.experiments.promoted_topk2_finite_update_control_matrix import (
    FINITE_UPDATE_CONTROL_MATRIX_READY,
    ROLE_BY_VARIANT,
)


DEFAULT_DECONFOUNDED_DIR = Path(
    "results/audits/token_larger_topk2_vs_rank_matched_topk1_deconfounded_intervention"
)
DEFAULT_FINITE_UPDATE_MATRIX_DIR = Path(
    "results/reports/token_larger_promoted_topk2_finite_update_control_matrix"
)
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_promoted_topk2_finite_update_augmented_causal_gate"
)

SUPPORTED = "finite_update_augmented_topk2_causal_cooperation_supported"
BLOCKED = "finite_update_augmented_topk2_causal_cooperation_blocked"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

REQUIRED_ROLES = (
    "promoted_contextual_topk2",
    "rank_matched_contextual_topk1",
    "random_fixed_topk2",
    "dense_active_rank",
)
FINITE_REQUIRED_FIELDS = {
    "variant",
    "position_bin",
    "token_class",
    "residual_norm_bin",
    "ce_abs_delta",
    "logit_mse",
    "symmetric_kl",
    "residual_delta_l2",
    "support_churn",
}
DECONFOUNDED_REQUIRED_FIELDS = {
    "position_bin",
    "token_class",
    "residual_norm_bin",
    "residual_gain_bin",
    "support_count_bin",
    "matched_exact_context_count",
    "topk2_incremental_pair_gain_minus_topk1_singleton",
    "topk2_fixed_delta_minus_topk1",
    "topk2_logit_mse_minus_topk1",
    "topk2_residual_stream_l2_delta_minus_topk1",
}
KEY_FIELDS = ("position_bin", "token_class", "residual_norm_bin")


def run_promoted_topk2_finite_update_augmented_causal_gate(
    *,
    deconfounded_dir: Path = DEFAULT_DECONFOUNDED_DIR,
    finite_update_matrix_dir: Path = DEFAULT_FINITE_UPDATE_MATRIX_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    min_positive_benefit_fraction: float = 0.8,
    max_topk2_vs_topk1_logit_mse_delta: float = 0.0,
    max_topk2_vs_dense_logit_mse_delta: float = 0.0,
    max_topk2_support_churn_fraction: float = 0.5,
) -> dict[str, Any]:
    """Join functional-benefit strata to finite-update risk controls."""

    start = time.time()
    failures: list[dict[str, Any]] = []
    deconf_summary_path = deconfounded_dir / "summary.json"
    deconf_strata_path = deconfounded_dir / "matched_deconfounded_strata.csv"
    finite_summary_path = finite_update_matrix_dir / "summary.json"
    finite_source_path = finite_update_matrix_dir / "source_rows.csv"

    for field, path in (
        ("deconfounded_summary_json", deconf_summary_path),
        ("matched_deconfounded_strata_csv", deconf_strata_path),
        ("finite_update_matrix_summary_json", finite_summary_path),
        ("finite_update_source_rows_csv", finite_source_path),
    ):
        if not path.is_file():
            failures.append({"field": field, "expected": str(path), "actual": "missing"})

    deconf_summary = _read_json_object(deconf_summary_path)
    finite_summary = _read_json_object(finite_summary_path)
    if deconf_summary and deconf_summary.get("status") != "pass":
        failures.append(
            {
                "field": "deconfounded_status",
                "expected": "pass",
                "actual": deconf_summary.get("status"),
            }
        )
    if finite_summary and finite_summary.get("decision") != FINITE_UPDATE_CONTROL_MATRIX_READY:
        failures.append(
            {
                "field": "finite_update_matrix_decision",
                "expected": FINITE_UPDATE_CONTROL_MATRIX_READY,
                "actual": finite_summary.get("decision"),
            }
        )

    deconf_rows = _read_csv_rows(deconf_strata_path)
    finite_source_rows = _read_csv_rows(finite_source_path)
    if deconf_rows:
        missing = sorted(DECONFOUNDED_REQUIRED_FIELDS - set(deconf_rows[0]))
        if missing:
            failures.append(
                {
                    "field": "matched_deconfounded_strata_fields",
                    "missing": missing,
                }
            )
    else:
        failures.append(
            {
                "field": "matched_deconfounded_strata_rows",
                "expected": "at least one row",
                "actual": 0,
            }
        )

    finite_rows, finite_sources = _read_finite_rows(finite_source_rows)
    if finite_rows:
        missing = sorted(FINITE_REQUIRED_FIELDS - set(finite_rows[0]))
        if missing:
            failures.append(
                {"field": "finite_update_per_token_fields", "missing": missing}
            )
    else:
        failures.append(
            {
                "field": "finite_update_per_token_rows",
                "expected": "at least one raw per-token commutator row",
                "actual": 0,
            }
        )

    role_counts = _role_counts(finite_rows)
    for role in REQUIRED_ROLES:
        if role_counts.get(role, 0) <= 0:
            failures.append(
                {
                    "field": "finite_update_required_role",
                    "expected": role,
                    "actual": "missing",
                }
            )

    risk_rows = _risk_control_rows(finite_rows)
    augmented_rows = _augmented_rows(deconf_rows, risk_rows)
    if not failures and not augmented_rows:
        failures.append(
            {
                "field": "augmented_join_rows",
                "expected": "at least one deconfounded stratum with finite-update controls",
                "actual": 0,
            }
        )

    metrics = _gate_metrics(
        deconf_summary,
        finite_summary,
        augmented_rows,
        min_positive_benefit_fraction=min_positive_benefit_fraction,
        max_topk2_vs_topk1_logit_mse_delta=max_topk2_vs_topk1_logit_mse_delta,
        max_topk2_vs_dense_logit_mse_delta=max_topk2_vs_dense_logit_mse_delta,
        max_topk2_support_churn_fraction=max_topk2_support_churn_fraction,
    )
    signals = metrics["signals"]
    claim_gate_passed = (
        not failures
        and signals["deconfounded_original_decision_supports_topk2"]
        and signals["benefit_fraction_gate_passed"]
        and signals["cleanliness_fraction_gate_passed"]
        and signals["topk2_logit_mse_not_worse_than_topk1"]
        and signals["topk2_logit_mse_not_worse_than_dense"]
        and signals["topk2_support_churn_not_high"]
        and signals["topk2_not_worse_than_random_fixed_on_logit_mse"]
    )
    status = "fail" if failures else "pass"
    decision = (
        INSUFFICIENT_EVIDENCE
        if failures
        else SUPPORTED
        if claim_gate_passed
        else BLOCKED
    )
    if failures:
        next_step = "repair missing source artifacts before interpreting the joint gate"
    elif claim_gate_passed:
        next_step = (
            "only then consider a small targeted finite-update mitigation; keep CE "
            "guardrails and preserve the rank-matched top-k-1/random/dense controls"
        )
    else:
        next_step = (
            "keep top-k-2 causal-cooperation claims blocked; return to active "
            "rank-matched top-k-1 controls, retention/churn, or matched deconfounding"
        )

    summary = {
        "status": status,
        "decision": decision,
        "deconfounded_dir": str(deconfounded_dir),
        "finite_update_matrix_dir": str(finite_update_matrix_dir),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "thresholds": {
            "min_positive_benefit_fraction": min_positive_benefit_fraction,
            "max_topk2_vs_topk1_logit_mse_delta": max_topk2_vs_topk1_logit_mse_delta,
            "max_topk2_vs_dense_logit_mse_delta": max_topk2_vs_dense_logit_mse_delta,
            "max_topk2_support_churn_fraction": max_topk2_support_churn_fraction,
        },
        "source_artifacts": {
            "deconfounded_summary_json": str(deconf_summary_path),
            "matched_deconfounded_strata_csv": str(deconf_strata_path),
            "finite_update_matrix_summary_json": str(finite_summary_path),
            "finite_update_source_rows_csv": str(finite_source_path),
        },
        "finite_sources": finite_sources,
        "role_counts": role_counts,
        "metrics": metrics["values"],
        "signals": signals,
        "failures": failures,
        "claim_gate": (
            "causal_cooperation_blocked_unless_functional_benefit_survives_"
            "finite_update_risk_controls"
        ),
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "finite_update_risk_controls_csv": str(
                out_dir / "finite_update_risk_controls.csv"
            ),
            "augmented_deconfounded_strata_csv": str(
                out_dir / "augmented_deconfounded_strata.csv"
            ),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "finite_update_risk_controls.csv", risk_rows)
    _write_csv(out_dir / "augmented_deconfounded_strata.csv", augmented_rows)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _read_finite_rows(source_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for source in source_rows:
        if str(source.get("per_token_commutator_present", "")).lower() != "true":
            continue
        path = Path(str(source.get("probe_dir", ""))) / "per_token_commutator.csv"
        present = path.is_file()
        source_record = dict(source)
        source_record["path"] = str(path)
        source_record["path_present"] = present
        if not present:
            sources.append(source_record)
            continue
        source_rows_from_csv = _read_csv_rows(path)
        source_record["read_row_count"] = len(source_rows_from_csv)
        sources.append(source_record)
        for row in source_rows_from_csv:
            variant = str(row.get("variant", ""))
            enriched = dict(row)
            enriched["matrix_role"] = ROLE_BY_VARIANT.get(variant, "unmapped")
            enriched["source_packet"] = source.get("packet", "")
            rows.append(enriched)
    return rows, sources


def _risk_control_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        role = str(row.get("matrix_role", ""))
        if role not in REQUIRED_ROLES:
            continue
        key = (
            str(row.get("position_bin", "")),
            str(row.get("token_class", "")),
            str(row.get("residual_norm_bin", "")),
            role,
        )
        buckets[key].append(row)

    out: list[dict[str, Any]] = []
    for key, group in sorted(buckets.items()):
        position_bin, token_class, residual_norm_bin, role = key
        out.append(
            {
                "position_bin": position_bin,
                "token_class": token_class,
                "residual_norm_bin": residual_norm_bin,
                "matrix_role": role,
                "row_count": len(group),
                "mean_ce_abs_delta": _mean(group, "ce_abs_delta"),
                "mean_logit_mse": _mean(group, "logit_mse"),
                "mean_symmetric_kl": _mean(group, "symmetric_kl"),
                "mean_residual_delta_l2": _mean(group, "residual_delta_l2"),
                "support_churn_fraction": _true_fraction(group, "support_churn"),
            }
        )
    return out


def _augmented_rows(
    deconf_rows: list[dict[str, str]],
    risk_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key_role = {
        (
            str(row.get("position_bin", "")),
            str(row.get("token_class", "")),
            str(row.get("residual_norm_bin", "")),
            str(row.get("matrix_role", "")),
        ): row
        for row in risk_rows
    }
    out: list[dict[str, Any]] = []
    for row in deconf_rows:
        base_key = tuple(str(row.get(field, "")) for field in KEY_FIELDS)
        controls = {
            role: by_key_role.get((*base_key, role), {}) for role in REQUIRED_ROLES
        }
        if not all(controls.values()):
            continue
        topk2 = controls["promoted_contextual_topk2"]
        topk1 = controls["rank_matched_contextual_topk1"]
        random_topk2 = controls["random_fixed_topk2"]
        dense = controls["dense_active_rank"]
        augmented = dict(row)
        augmented.update(
            {
                "finite_topk2_row_count": topk2.get("row_count"),
                "finite_topk1_row_count": topk1.get("row_count"),
                "finite_random_fixed_topk2_row_count": random_topk2.get("row_count"),
                "finite_dense_active_rank_row_count": dense.get("row_count"),
                "finite_topk2_mean_logit_mse": topk2.get("mean_logit_mse"),
                "finite_topk1_mean_logit_mse": topk1.get("mean_logit_mse"),
                "finite_random_fixed_topk2_mean_logit_mse": random_topk2.get(
                    "mean_logit_mse"
                ),
                "finite_dense_active_rank_mean_logit_mse": dense.get(
                    "mean_logit_mse"
                ),
                "finite_topk2_minus_topk1_logit_mse": _difference(
                    topk2.get("mean_logit_mse"),
                    topk1.get("mean_logit_mse"),
                ),
                "finite_topk2_minus_random_fixed_topk2_logit_mse": _difference(
                    topk2.get("mean_logit_mse"),
                    random_topk2.get("mean_logit_mse"),
                ),
                "finite_topk2_minus_dense_logit_mse": _difference(
                    topk2.get("mean_logit_mse"),
                    dense.get("mean_logit_mse"),
                ),
                "finite_topk2_support_churn_fraction": topk2.get(
                    "support_churn_fraction"
                ),
                "finite_topk1_support_churn_fraction": topk1.get(
                    "support_churn_fraction"
                ),
                "finite_topk2_mean_ce_abs_delta": topk2.get("mean_ce_abs_delta"),
                "finite_topk1_mean_ce_abs_delta": topk1.get("mean_ce_abs_delta"),
                "finite_topk2_mean_residual_delta_l2": topk2.get(
                    "mean_residual_delta_l2"
                ),
                "finite_topk1_mean_residual_delta_l2": topk1.get(
                    "mean_residual_delta_l2"
                ),
            }
        )
        out.append(augmented)
    return out


def _gate_metrics(
    deconf_summary: dict[str, Any],
    finite_summary: dict[str, Any],
    augmented_rows: list[dict[str, Any]],
    *,
    min_positive_benefit_fraction: float,
    max_topk2_vs_topk1_logit_mse_delta: float,
    max_topk2_vs_dense_logit_mse_delta: float,
    max_topk2_support_churn_fraction: float,
) -> dict[str, dict[str, Any]]:
    evidence = deconf_summary.get("evidence", {})
    original_decision = deconf_summary.get("decision")
    finite_matrix_metrics = finite_summary.get("metrics", {})
    values = {
        "deconfounded_original_decision": original_decision,
        "deconfounded_topk2_incremental_pair_gain_positive_strata_fraction": evidence.get(
            "topk2_incremental_pair_gain_positive_strata_fraction"
        ),
        "deconfounded_topk2_fixed_support_cleaner_strata_fraction": evidence.get(
            "topk2_fixed_support_cleaner_strata_fraction"
        ),
        "deconfounded_topk2_functional_churn_cleaner_strata_fraction": evidence.get(
            "topk2_functional_churn_cleaner_strata_fraction"
        ),
        "deconfounded_ce_guardrail_passed": evidence.get("ce_guardrail_passed"),
        "augmented_strata_count": len(augmented_rows),
        "augmented_matched_exact_context_count": sum(
            int(_optional_float(row.get("matched_exact_context_count")) or 0)
            for row in augmented_rows
        ),
        "augmented_positive_benefit_fraction": _fraction(
            augmented_rows,
            "topk2_incremental_pair_gain_minus_topk1_singleton",
            lambda value: value > 0.0,
        ),
        "augmented_fixed_cleaner_fraction": _fraction(
            augmented_rows,
            "topk2_fixed_delta_minus_topk1",
            lambda value: value < 0.0,
        ),
        "augmented_functional_churn_cleaner_fraction": _fraction(
            augmented_rows,
            "topk2_residual_stream_l2_delta_minus_topk1",
            lambda value: value < 0.0,
        ),
        "augmented_mean_topk2_minus_topk1_finite_logit_mse": _mean(
            augmented_rows,
            "finite_topk2_minus_topk1_logit_mse",
        ),
        "augmented_mean_topk2_minus_random_fixed_finite_logit_mse": _mean(
            augmented_rows,
            "finite_topk2_minus_random_fixed_topk2_logit_mse",
        ),
        "augmented_mean_topk2_minus_dense_finite_logit_mse": _mean(
            augmented_rows,
            "finite_topk2_minus_dense_logit_mse",
        ),
        "augmented_mean_topk2_finite_support_churn_fraction": _mean(
            augmented_rows,
            "finite_topk2_support_churn_fraction",
        ),
        "matrix_topk2_minus_topk1_logit_mse": finite_matrix_metrics.get(
            "topk2_minus_topk1_logit_mse"
        ),
        "matrix_topk2_minus_dense_logit_mse": finite_matrix_metrics.get(
            "topk2_minus_dense_logit_mse"
        ),
        "matrix_topk2_minus_random_fixed_topk2_logit_mse": finite_matrix_metrics.get(
            "topk2_minus_random_fixed_topk2_logit_mse"
        ),
        "matrix_topk2_support_churn_fraction": finite_matrix_metrics.get(
            "topk2_support_churn_fraction"
        ),
    }
    signals = {
        "deconfounded_original_decision_supports_topk2": original_decision
        == "topk2_causal_metrics_survive_deconfounding_with_ce_guardrail",
        "ce_guardrail_passed": bool(values["deconfounded_ce_guardrail_passed"]),
        "benefit_fraction_gate_passed": _at_least(
            values["augmented_positive_benefit_fraction"],
            min_positive_benefit_fraction,
        ),
        "cleanliness_fraction_gate_passed": _at_least(
            values["augmented_fixed_cleaner_fraction"],
            min_positive_benefit_fraction,
        )
        and _at_least(
            values["augmented_functional_churn_cleaner_fraction"],
            min_positive_benefit_fraction,
        ),
        "topk2_logit_mse_not_worse_than_topk1": _at_most(
            values["augmented_mean_topk2_minus_topk1_finite_logit_mse"],
            max_topk2_vs_topk1_logit_mse_delta,
        ),
        "topk2_logit_mse_not_worse_than_dense": _at_most(
            values["augmented_mean_topk2_minus_dense_finite_logit_mse"],
            max_topk2_vs_dense_logit_mse_delta,
        ),
        "topk2_not_worse_than_random_fixed_on_logit_mse": _at_most(
            values["augmented_mean_topk2_minus_random_fixed_finite_logit_mse"],
            0.0,
        ),
        "topk2_support_churn_not_high": _at_most(
            values["augmented_mean_topk2_finite_support_churn_fraction"],
            max_topk2_support_churn_fraction,
        ),
    }
    return {"values": values, "signals": signals}


def _role_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("matrix_role", ""))] += 1
    return dict(counts)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    lines = [
        "# Finite-update-augmented causal gate",
        "",
        f"Status: `{summary['status']}`",
        f"Decision: `{summary['decision']}`",
        "",
        "This no-training gate joins deconfounded functional-benefit strata to "
        "finite-update order-sensitivity controls. It is a claim gate, not a new "
        "training result.",
        "",
        f"Claim gate: `{summary['claim_gate']}`",
        f"Augmented strata: `{metrics.get('augmented_strata_count')}`",
        f"Positive benefit fraction: `{metrics.get('augmented_positive_benefit_fraction')}`",
        f"Fixed-support cleaner fraction: `{metrics.get('augmented_fixed_cleaner_fraction')}`",
        f"Top-k-2 minus top-k-1 finite logit MSE: `{metrics.get('augmented_mean_topk2_minus_topk1_finite_logit_mse')}`",
        f"Top-k-2 finite support churn: `{metrics.get('augmented_mean_topk2_finite_support_churn_fraction')}`",
        f"Next step: {summary['next_step']}",
    ]
    if summary["failures"]:
        lines.extend(["", "## Failures"])
        lines.extend(f"- `{failure}`" for failure in summary["failures"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [
        value
        for value in (_optional_float(row.get(field)) for row in rows)
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


def _fraction(
    rows: list[dict[str, Any]],
    field: str,
    predicate: Any,
) -> float | None:
    values = [
        value
        for value in (_optional_float(row.get(field)) for row in rows)
        if value is not None
    ]
    if not values:
        return None
    return sum(1 for value in values if predicate(value)) / len(values)


def _difference(left: Any, right: Any) -> float | None:
    left_float = _optional_float(left)
    right_float = _optional_float(right)
    if left_float is None or right_float is None:
        return None
    return left_float - right_float


def _at_least(value: Any, threshold: float) -> bool:
    numeric = _optional_float(value)
    return numeric is not None and numeric >= threshold


def _at_most(value: Any, threshold: float) -> bool:
    numeric = _optional_float(value)
    return numeric is not None and numeric <= threshold


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deconfounded-dir", type=Path, default=DEFAULT_DECONFOUNDED_DIR)
    parser.add_argument(
        "--finite-update-matrix-dir",
        type=Path,
        default=DEFAULT_FINITE_UPDATE_MATRIX_DIR,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_promoted_topk2_finite_update_augmented_causal_gate(
        deconfounded_dir=args.deconfounded_dir,
        finite_update_matrix_dir=args.finite_update_matrix_dir,
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
