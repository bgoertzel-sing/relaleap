"""Post-hoc matched-strata intervention audit for causal fingerprint artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_AUDIT_DIR = Path(
    "results/audits/token_larger_support_wide_promoted_default_causal_column_fingerprint_stability_topk1"
)
DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_rank_matched_topk1_vs_topk2_matched_strata_intervention"
)
TOPK2_VARIANT = "baseline"
TOPK1_VARIANT = "rank_matched_topk1_contextual"


def run_matched_strata_intervention_audit(
    audit_dir: Path,
    out_dir: Path,
    *,
    topk2_variant: str = TOPK2_VARIANT,
    topk1_variant: str = TOPK1_VARIANT,
) -> dict[str, Any]:
    """Compare top-k-2 and rank-matched top-k-1 intervention rows by strata."""

    start = time.time()
    failures: list[dict[str, Any]] = []
    summary_path = audit_dir / "summary.json"
    pair_path = audit_dir / "pair_interventions.csv"
    if not summary_path.is_file():
        failures.append({"field": "summary_json", "expected": str(summary_path)})
    if not pair_path.is_file():
        failures.append(
            {"field": "pair_interventions_csv", "expected": str(pair_path)}
        )

    source_summary: dict[str, Any] = {}
    pair_rows: list[dict[str, str]] = []
    if not failures:
        source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with pair_path.open(newline="", encoding="utf-8") as handle:
            pair_rows = list(csv.DictReader(handle))

    source_variants = {
        str(row.get("variant"))
        for row in source_summary.get("audit", {}).get("variants", [])
    }
    if source_summary and topk2_variant not in source_variants:
        failures.append(
            {"field": "variants", "expected": topk2_variant, "actual": sorted(source_variants)}
        )
    if source_summary and topk1_variant not in source_variants:
        failures.append(
            {"field": "variants", "expected": topk1_variant, "actual": sorted(source_variants)}
        )

    topk2_rows = [
        row
        for row in pair_rows
        if row.get("variant") == topk2_variant
        and row.get("intervention") == "fixed_dominant_router_support"
    ]
    topk1_rows = [
        row
        for row in pair_rows
        if row.get("variant") == topk1_variant
        and row.get("intervention") == "fixed_dominant_router_singleton"
    ]
    if pair_rows and not topk2_rows:
        failures.append(
            {
                "field": "topk2_interventions",
                "expected": "fixed_dominant_router_support rows",
            }
        )
    if pair_rows and not topk1_rows:
        failures.append(
            {
                "field": "topk1_interventions",
                "expected": "fixed_dominant_router_singleton rows",
            }
        )

    matched_rows: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {
        "failures": failures,
        "source_audit_dir": str(audit_dir),
        "topk2_variant": topk2_variant,
        "topk1_variant": topk1_variant,
    }
    status = "fail" if failures else "pass"
    decision = "insufficient_evidence"

    if not failures:
        matched_rows = _matched_strata_rows(topk2_rows, topk1_rows)
        if not matched_rows:
            failures.append(
                {
                    "field": "matched_strata",
                    "expected": "at least one shared position_bin/token_class stratum",
                }
            )
            status = "fail"
        else:
            evidence.update(_matched_strata_evidence(source_summary, matched_rows))
            decision = _matched_strata_decision(evidence)

    if failures:
        evidence["failures"] = failures

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        out_dir / "matched_strata.csv",
        _MATCHED_STRATA_FIELDNAMES,
        matched_rows,
    )
    summary = {
        "status": status,
        "decision": decision,
        "audit_dir": str(audit_dir),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "evidence": evidence,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "matched_strata_csv": str(out_dir / "matched_strata.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _matched_strata_rows(
    topk2_rows: list[dict[str, str]],
    topk1_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    topk2_by_stratum = _aggregate_by_stratum(topk2_rows)
    topk1_by_stratum = _aggregate_by_stratum(topk1_rows)
    matched = []
    for stratum in sorted(set(topk2_by_stratum) & set(topk1_by_stratum)):
        position_bin, token_class = stratum
        topk2 = topk2_by_stratum[stratum]
        topk1 = topk1_by_stratum[stratum]
        topk2_fixed_delta = topk2["fixed_support_loss_delta_mean"]
        topk1_fixed_delta = topk1["fixed_support_loss_delta_mean"]
        topk2_gain = topk2["pair_gain_mean"]
        topk1_gain = topk1["singleton_left_gain_mean"]
        matched.append(
            {
                "position_bin": position_bin,
                "token_class": token_class,
                "topk2_row_count": topk2["row_count"],
                "topk1_row_count": topk1["row_count"],
                "topk2_router_loss_mean": topk2["router_loss_mean"],
                "topk1_router_loss_mean": topk1["router_loss_mean"],
                "topk2_router_loss_minus_topk1": (
                    topk2["router_loss_mean"] - topk1["router_loss_mean"]
                ),
                "topk2_fixed_support_loss_delta_mean": topk2_fixed_delta,
                "topk1_fixed_support_loss_delta_mean": topk1_fixed_delta,
                "topk2_fixed_delta_minus_topk1": (
                    topk2_fixed_delta - topk1_fixed_delta
                ),
                "topk2_pair_gain_mean": topk2_gain,
                "topk1_singleton_gain_mean": topk1_gain,
                "topk2_pair_gain_minus_topk1_singleton": topk2_gain - topk1_gain,
                "topk2_pair_synergy_mean": topk2["pair_synergy_mean"],
                "topk2_pair_synergy_positive_fraction": topk2[
                    "pair_synergy_positive_fraction"
                ],
                "topk2_pair_value_cosine_mean": topk2["pair_value_cosine_mean"],
                "topk1_singleton_value_norm_mean": topk1["pair_value_cosine_mean"],
            }
        )
    return matched


def _aggregate_by_stratum(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row.get("position_bin")), str(row.get("token_class")))].append(
            row
        )
    return {
        stratum: {
            "row_count": len(stratum_rows),
            "router_loss_mean": _mean_field(stratum_rows, "router_loss"),
            "fixed_support_loss_delta_mean": _mean_field(
                stratum_rows,
                "fixed_support_loss_delta",
            ),
            "pair_gain_mean": _mean_field(stratum_rows, "pair_gain"),
            "singleton_left_gain_mean": _mean_field(
                stratum_rows,
                "singleton_left_gain",
            ),
            "pair_synergy_mean": _mean_field(stratum_rows, "pair_synergy"),
            "pair_synergy_positive_fraction": _positive_fraction(
                stratum_rows,
                "pair_synergy",
            ),
            "pair_value_cosine_mean": _mean_field(stratum_rows, "pair_value_cosine"),
        }
        for stratum, stratum_rows in buckets.items()
    }


def _matched_strata_evidence(
    source_summary: dict[str, Any],
    matched_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source_variants = {
        row["variant"]: row
        for row in source_summary.get("audit", {}).get("variants", [])
        if "variant" in row
    }
    churn = {
        row["variant"]: row
        for row in source_summary.get("audit", {}).get("functional_churn", [])
        if "variant" in row
    }
    topk2_variant = source_variants.get(TOPK2_VARIANT, {})
    topk1_variant = source_variants.get(TOPK1_VARIANT, {})
    topk2_router_ce = _optional_float(topk2_variant.get("alpha0_ce_loss"))
    topk1_router_ce = _optional_float(topk1_variant.get("alpha0_ce_loss"))
    topk2_churn = _optional_float(
        churn.get(TOPK2_VARIANT, {}).get("previous_support_changed_logit_mse_mean")
    )
    topk1_churn = _optional_float(
        churn.get(TOPK1_VARIANT, {}).get("previous_support_changed_logit_mse_mean")
    )
    topk2_synergies = [
        row["topk2_pair_synergy_mean"]
        for row in matched_rows
        if row["topk2_pair_synergy_mean"] is not None
    ]
    fixed_delta_differences = [
        row["topk2_fixed_delta_minus_topk1"]
        for row in matched_rows
        if row["topk2_fixed_delta_minus_topk1"] is not None
    ]
    return {
        "matched_strata_count": len(matched_rows),
        "matched_strata": [
            f"{row['position_bin']}|{row['token_class']}" for row in matched_rows
        ],
        "topk2_alpha0_ce_loss": topk2_router_ce,
        "topk1_alpha0_ce_loss": topk1_router_ce,
        "topk2_router_ce_better_than_topk1": (
            topk2_router_ce is not None
            and topk1_router_ce is not None
            and topk2_router_ce < topk1_router_ce
        ),
        "rank_matched_topk1_router_ce_better_than_topk2": (
            topk2_router_ce is not None
            and topk1_router_ce is not None
            and topk1_router_ce < topk2_router_ce
        ),
        "topk2_pair_synergy_mean_across_strata": _mean(topk2_synergies),
        "topk2_pair_synergy_positive_strata_fraction": _fraction(
            value is not None and value > 0.0 for value in topk2_synergies
        ),
        "topk2_fixed_delta_minus_topk1_mean_across_strata": _mean(
            fixed_delta_differences
        ),
        "topk2_fixed_support_cleaner_than_topk1_strata_fraction": _fraction(
            value is not None and value < 0.0 for value in fixed_delta_differences
        ),
        "topk2_changed_support_logit_mse": topk2_churn,
        "topk1_changed_support_logit_mse": topk1_churn,
        "topk2_functional_churn_cleaner_than_topk1": (
            topk2_churn is not None
            and topk1_churn is not None
            and topk2_churn < topk1_churn
        ),
    }


def _matched_strata_decision(evidence: dict[str, Any]) -> str:
    supports_topk2 = (
        evidence["topk2_router_ce_better_than_topk1"]
        and evidence["topk2_pair_synergy_mean_across_strata"] is not None
        and evidence["topk2_pair_synergy_mean_across_strata"] > 0.0
        and evidence["topk2_pair_synergy_positive_strata_fraction"] >= 0.8
        and evidence["topk2_fixed_support_cleaner_than_topk1_strata_fraction"] >= 0.8
        and evidence["topk2_functional_churn_cleaner_than_topk1"]
    )
    if supports_topk2:
        return "topk2_cooperation_supported_by_matched_strata"
    if evidence["rank_matched_topk1_router_ce_better_than_topk2"]:
        return "prefer_rank_matched_topk1_for_causal_audits"
    return "topk2_cooperation_not_supported_by_matched_strata"


def _mean_field(rows: list[dict[str, str]], field: str) -> float | None:
    return _mean([value for row in rows if (value := _optional_float(row.get(field))) is not None])


def _positive_fraction(rows: list[dict[str, str]], field: str) -> float | None:
    values = [_optional_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return _fraction(value > 0.0 for value in values)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _fraction(values: Any) -> float | None:
    materialized = list(values)
    if not materialized:
        return None
    return float(sum(1 for value in materialized if value) / len(materialized))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    evidence = summary["evidence"]
    lines = [
        "# Matched-Strata Intervention Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Source audit: `{summary['audit_dir']}`",
    ]
    if summary["status"] == "pass":
        lines.extend(
            [
                f"- Matched strata: `{evidence['matched_strata_count']}`",
                f"- Top-k-2 alpha-0 CE: `{evidence['topk2_alpha0_ce_loss']}`",
                f"- Rank-matched top-k-1 alpha-0 CE: `{evidence['topk1_alpha0_ce_loss']}`",
                f"- Top-k-2 mean pair synergy across strata: `{evidence['topk2_pair_synergy_mean_across_strata']}`",
                f"- Top-k-2 fixed-delta minus top-k-1 mean across strata: `{evidence['topk2_fixed_delta_minus_topk1_mean_across_strata']}`",
                f"- Top-k-2 changed-support logit MSE: `{evidence['topk2_changed_support_logit_mse']}`",
                f"- Rank-matched top-k-1 changed-support logit MSE: `{evidence['topk1_changed_support_logit_mse']}`",
            ]
        )
    else:
        for failure in evidence.get("failures", []):
            lines.append(f"- Missing `{failure['field']}`: expected `{failure.get('expected')}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


_MATCHED_STRATA_FIELDNAMES = [
    "position_bin",
    "token_class",
    "topk2_row_count",
    "topk1_row_count",
    "topk2_router_loss_mean",
    "topk1_router_loss_mean",
    "topk2_router_loss_minus_topk1",
    "topk2_fixed_support_loss_delta_mean",
    "topk1_fixed_support_loss_delta_mean",
    "topk2_fixed_delta_minus_topk1",
    "topk2_pair_gain_mean",
    "topk1_singleton_gain_mean",
    "topk2_pair_gain_minus_topk1_singleton",
    "topk2_pair_synergy_mean",
    "topk2_pair_synergy_positive_fraction",
    "topk2_pair_value_cosine_mean",
    "topk1_singleton_value_norm_mean",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--topk2-variant", default=TOPK2_VARIANT)
    parser.add_argument("--topk1-variant", default=TOPK1_VARIANT)
    args = parser.parse_args(argv)
    summary = run_matched_strata_intervention_audit(
        args.audit_dir,
        args.out,
        topk2_variant=args.topk2_variant,
        topk1_variant=args.topk1_variant,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "evidence": summary["evidence"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
