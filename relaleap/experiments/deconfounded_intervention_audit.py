"""No-training deconfounded intervention audit for causal fingerprint artifacts."""

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
DEFAULT_MATCHED_STRATA_DIR = Path(
    "results/audits/token_larger_rank_matched_topk1_vs_topk2_matched_strata_intervention"
)
DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_topk2_vs_rank_matched_topk1_deconfounded_intervention"
)
TOPK2_VARIANT = "baseline"
TOPK1_VARIANT = "rank_matched_topk1_contextual"
TOPK2_INTERVENTION = "fixed_dominant_router_support"
TOPK1_INTERVENTION = "fixed_dominant_router_singleton"
CE_GUARDRAIL_TOLERANCE = 0.05
CONTEXT_FIELDS = ("batch_index", "position_index", "token_index", "target_token")


def run_deconfounded_intervention_audit(
    audit_dir: Path,
    out_dir: Path,
    *,
    matched_strata_dir: Path = DEFAULT_MATCHED_STRATA_DIR,
    topk2_variant: str = TOPK2_VARIANT,
    topk1_variant: str = TOPK1_VARIANT,
    ce_guardrail_tolerance: float = CE_GUARDRAIL_TOLERANCE,
    min_rows_per_side: int = 1,
) -> dict[str, Any]:
    """Compare top-k-2 and rank-matched top-k-1 after per-token stratum matching."""

    start = time.time()
    failures: list[dict[str, Any]] = []
    summary_path = audit_dir / "summary.json"
    per_token_path = audit_dir / "per_token_pair_interventions.csv"
    pair_path = audit_dir / "pair_interventions.csv"
    for field, path in (
        ("summary_json", summary_path),
        ("per_token_pair_interventions_csv", per_token_path),
        ("pair_interventions_csv", pair_path),
    ):
        if not path.is_file():
            failures.append({"field": field, "expected": str(path)})

    source_summary: dict[str, Any] = {}
    per_token_rows: list[dict[str, str]] = []
    pair_rows: list[dict[str, str]] = []
    if not failures:
        source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with per_token_path.open(newline="", encoding="utf-8") as handle:
            per_token_rows = list(csv.DictReader(handle))
        with pair_path.open(newline="", encoding="utf-8") as handle:
            pair_rows = list(csv.DictReader(handle))

    topk2_rows = [
        row
        for row in per_token_rows
        if row.get("variant") == topk2_variant
        and row.get("intervention") == TOPK2_INTERVENTION
    ]
    topk1_rows = [
        row
        for row in per_token_rows
        if row.get("variant") == topk1_variant
        and row.get("intervention") == TOPK1_INTERVENTION
    ]
    if per_token_rows and not topk2_rows:
        failures.append(
            {
                "field": "topk2_per_token_interventions",
                "expected": f"{topk2_variant}/{TOPK2_INTERVENTION} rows",
            }
        )
    if per_token_rows and not topk1_rows:
        failures.append(
            {
                "field": "topk1_per_token_interventions",
                "expected": f"{topk1_variant}/{TOPK1_INTERVENTION} rows",
            }
        )

    required_fields = {
        *CONTEXT_FIELDS,
        "position_bin",
        "token_class",
        "router_support_count",
        "router_loss",
        "pair_gain",
        "singleton_left_gain",
        "fixed_support_loss_delta",
        "fixed_support_logit_mse",
        "fixed_support_residual_stream_l2_delta",
        "residual_norm_bin",
        "residual_gain_bin",
        "active_rank_proxy",
    }
    if per_token_rows:
        fields = set(per_token_rows[0])
        missing_fields = sorted(required_fields - fields)
        if missing_fields:
            failures.append(
                {
                    "field": "per_token_matching_fields",
                    "expected": sorted(required_fields),
                    "actual_missing": missing_fields,
                }
            )

    status = "fail" if failures else "pass"
    decision = "insufficient_evidence"
    matched_rows: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {
        "failures": failures,
        "source_audit_dir": str(audit_dir),
        "matched_strata_dir": str(matched_strata_dir),
        "topk2_variant": topk2_variant,
        "topk1_variant": topk1_variant,
        "ce_guardrail_tolerance": ce_guardrail_tolerance,
        "min_rows_per_side": min_rows_per_side,
        "active_rank_matching_note": (
            "active_rank_proxy is reported as a bracket dimension, not exact-matched, "
            "because promoted top-k-2 and rank-matched top-k-1 have structurally "
            "different active-rank proxies in this artifact."
        ),
        "exact_context_matching_note": (
            "top-k-2 and rank-matched top-k-1 rows are first paired by exact "
            "batch_index/position_index/token_index/target_token context, then "
            "summarized inside matched position/token/residual/support-count strata."
        ),
    }

    if not failures:
        support_count_bins = _support_count_bins(topk2_rows + topk1_rows)
        (
            matched_rows,
            paired_context_rows,
            context_matching_evidence,
        ) = _matched_deconfounded_rows(
            topk2_rows,
            topk1_rows,
            support_count_bins=support_count_bins,
            min_rows_per_side=min_rows_per_side,
        )
        if not matched_rows:
            failures.append(
                {
                    "field": "matched_deconfounded_strata",
                    "expected": "at least one shared position/token/residual/support stratum",
                }
            )
            status = "fail"
        else:
            evidence.update(
                _deconfounded_evidence(
                    source_summary,
                    pair_rows,
                    matched_rows,
                    topk2_variant=topk2_variant,
                    topk1_variant=topk1_variant,
                    ce_guardrail_tolerance=ce_guardrail_tolerance,
                )
            )
            evidence.update(context_matching_evidence)
            decision = _deconfounded_decision(evidence)
    else:
        paired_context_rows = []

    if failures:
        evidence["failures"] = failures

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "matched_deconfounded_strata.csv", _FIELDNAMES, matched_rows)
    _write_csv(
        out_dir / "paired_exact_context_deltas.csv",
        _CONTEXT_FIELDNAMES,
        paired_context_rows,
    )
    summary = {
        "status": status,
        "decision": decision,
        "audit_dir": str(audit_dir),
        "matched_strata_dir": str(matched_strata_dir),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "evidence": evidence,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "matched_deconfounded_strata_csv": str(
                out_dir / "matched_deconfounded_strata.csv"
            ),
            "paired_exact_context_deltas_csv": str(
                out_dir / "paired_exact_context_deltas.csv"
            ),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _matched_deconfounded_rows(
    topk2_rows: list[dict[str, str]],
    topk1_rows: list[dict[str, str]],
    *,
    support_count_bins: dict[int, str],
    min_rows_per_side: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    topk2_by_context = _aggregate_by_context_stratum(topk2_rows, support_count_bins)
    topk1_by_context = _aggregate_by_context_stratum(topk1_rows, support_count_bins)
    shared_context_strata = sorted(set(topk2_by_context) & set(topk1_by_context))
    topk2_contexts = {context for context, _stratum in topk2_by_context}
    topk1_contexts = {context for context, _stratum in topk1_by_context}
    matched_contexts = {context for context, _stratum in shared_context_strata}
    paired_context_stratum_rows = _paired_context_stratum_rows(
        shared_context_strata,
        topk2_by_context,
        topk1_by_context,
    )
    paired_context_rows = _aggregate_paired_context_rows(paired_context_stratum_rows)
    topk2_by_stratum = _aggregate_matched_contexts_by_stratum(
        {
            context_stratum: topk2_by_context[context_stratum]
            for context_stratum in shared_context_strata
        }
    )
    topk1_by_stratum = _aggregate_matched_contexts_by_stratum(
        {
            context_stratum: topk1_by_context[context_stratum]
            for context_stratum in shared_context_strata
        }
    )
    rows: list[dict[str, Any]] = []
    for stratum in sorted(set(topk2_by_stratum) & set(topk1_by_stratum)):
        topk2 = topk2_by_stratum[stratum]
        topk1 = topk1_by_stratum[stratum]
        if (
            topk2["row_count"] < min_rows_per_side
            or topk1["row_count"] < min_rows_per_side
        ):
            continue
        (
            position_bin,
            token_class,
            residual_norm_bin,
            residual_gain_bin,
            support_count_bin,
        ) = stratum
        row = {
            "position_bin": position_bin,
            "token_class": token_class,
            "residual_norm_bin": residual_norm_bin,
            "residual_gain_bin": residual_gain_bin,
            "support_count_bin": support_count_bin,
            "topk2_active_rank_proxy": topk2["active_rank_proxy"],
            "topk1_active_rank_proxy": topk1["active_rank_proxy"],
            "matched_exact_context_count": topk2["context_count"],
            "topk2_row_count": topk2["row_count"],
            "topk1_row_count": topk1["row_count"],
            "topk2_router_support_count_mean": topk2["router_support_count_mean"],
            "topk1_router_support_count_mean": topk1["router_support_count_mean"],
            "topk2_router_loss_mean": topk2["router_loss_mean"],
            "topk1_router_loss_mean": topk1["router_loss_mean"],
            "topk2_pair_gain_mean": topk2["pair_gain_mean"],
            "topk1_singleton_gain_mean": topk1["singleton_left_gain_mean"],
            "topk2_incremental_pair_gain_minus_topk1_singleton": _difference(
                topk2["pair_gain_mean"],
                topk1["singleton_left_gain_mean"],
            ),
            "topk2_fixed_support_loss_delta_mean": topk2[
                "fixed_support_loss_delta_mean"
            ],
            "topk1_fixed_support_loss_delta_mean": topk1[
                "fixed_support_loss_delta_mean"
            ],
            "topk2_fixed_delta_minus_topk1": (
                topk2["fixed_support_loss_delta_mean"]
                - topk1["fixed_support_loss_delta_mean"]
            ),
            "topk2_fixed_support_logit_mse_mean": topk2[
                "fixed_support_logit_mse_mean"
            ],
            "topk1_fixed_support_logit_mse_mean": topk1[
                "fixed_support_logit_mse_mean"
            ],
            "topk2_logit_mse_minus_topk1": (
                topk2["fixed_support_logit_mse_mean"]
                - topk1["fixed_support_logit_mse_mean"]
            ),
            "topk2_residual_stream_l2_delta_mean": topk2[
                "fixed_support_residual_stream_l2_delta_mean"
            ],
            "topk1_residual_stream_l2_delta_mean": topk1[
                "fixed_support_residual_stream_l2_delta_mean"
            ],
            "topk2_residual_stream_l2_delta_minus_topk1": (
                topk2["fixed_support_residual_stream_l2_delta_mean"]
                - topk1["fixed_support_residual_stream_l2_delta_mean"]
            ),
            "topk2_pair_synergy_mean": topk2["pair_synergy_mean"],
        }
        rows.append(row)
    context_evidence = {
        "topk2_exact_context_count": len(topk2_contexts),
        "topk1_exact_context_count": len(topk1_contexts),
        "shared_exact_context_count_before_stratum_match": len(
            topk2_contexts & topk1_contexts
        ),
        "matched_exact_context_count": len(matched_contexts),
        "unmatched_topk2_context_count": len(topk2_contexts - matched_contexts),
        "unmatched_topk1_context_count": len(topk1_contexts - matched_contexts),
        "matched_topk2_context_fraction": (
            len(matched_contexts) / len(topk2_contexts) if topk2_contexts else None
        ),
        "matched_topk1_context_fraction": (
            len(matched_contexts) / len(topk1_contexts) if topk1_contexts else None
        ),
    }
    return rows, paired_context_rows, context_evidence


def _paired_context_stratum_rows(
    shared_context_strata: list[
        tuple[tuple[str, str, str, str], tuple[str, str, str, str, str]]
    ],
    topk2_by_context: dict[
        tuple[tuple[str, str, str, str], tuple[str, str, str, str, str]],
        dict[str, Any],
    ],
    topk1_by_context: dict[
        tuple[tuple[str, str, str, str], tuple[str, str, str, str, str]],
        dict[str, Any],
    ],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for context_stratum in shared_context_strata:
        context, stratum = context_stratum
        topk2 = topk2_by_context[context_stratum]
        topk1 = topk1_by_context[context_stratum]
        (
            position_bin,
            token_class,
            residual_norm_bin,
            residual_gain_bin,
            support_count_bin,
        ) = stratum
        row = {
            "batch_index": context[0],
            "position_index": context[1],
            "token_index": context[2],
            "target_token": context[3],
            "position_bin": position_bin,
            "token_class": token_class,
            "residual_norm_bin": residual_norm_bin,
            "residual_gain_bin": residual_gain_bin,
            "support_count_bin": support_count_bin,
            "topk2_row_count": topk2["row_count"],
            "topk1_row_count": topk1["row_count"],
            "topk2_router_loss_mean": topk2["router_loss_mean"],
            "topk1_router_loss_mean": topk1["router_loss_mean"],
            "topk2_pair_gain_mean": topk2["pair_gain_mean"],
            "topk1_singleton_gain_mean": topk1["singleton_left_gain_mean"],
            "topk2_incremental_pair_gain_minus_topk1_singleton": _difference(
                topk2["pair_gain_mean"],
                topk1["singleton_left_gain_mean"],
            ),
            "topk2_fixed_delta_minus_topk1": _difference(
                topk2["fixed_support_loss_delta_mean"],
                topk1["fixed_support_loss_delta_mean"],
            ),
            "topk2_logit_mse_minus_topk1": _difference(
                topk2["fixed_support_logit_mse_mean"],
                topk1["fixed_support_logit_mse_mean"],
            ),
            "topk2_residual_stream_l2_delta_minus_topk1": _difference(
                topk2["fixed_support_residual_stream_l2_delta_mean"],
                topk1["fixed_support_residual_stream_l2_delta_mean"],
            ),
            "topk2_pair_synergy_mean": topk2["pair_synergy_mean"],
        }
        rows.append(row)
    return rows


def _aggregate_paired_context_rows(
    paired_context_stratum_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paired_context_stratum_rows:
        context = (
            str(row["batch_index"]),
            str(row["position_index"]),
            str(row["token_index"]),
            str(row["target_token"]),
        )
        buckets[context].append(row)
    rows: list[dict[str, Any]] = []
    for context, context_rows in sorted(buckets.items()):
        rows.append(
            {
                "batch_index": context[0],
                "position_index": context[1],
                "token_index": context[2],
                "target_token": context[3],
                "matched_context_stratum_count": len(context_rows),
                "position_bins": _joined_stats_values(context_rows, "position_bin"),
                "token_classes": _joined_stats_values(context_rows, "token_class"),
                "residual_norm_bins": _joined_stats_values(
                    context_rows, "residual_norm_bin"
                ),
                "residual_gain_bins": _joined_stats_values(
                    context_rows, "residual_gain_bin"
                ),
                "support_count_bins": _joined_stats_values(
                    context_rows, "support_count_bin"
                ),
                "topk2_row_count": sum(int(row["topk2_row_count"]) for row in context_rows),
                "topk1_row_count": sum(int(row["topk1_row_count"]) for row in context_rows),
                "topk2_router_loss_mean": _mean_stats(
                    context_rows, "topk2_router_loss_mean"
                ),
                "topk1_router_loss_mean": _mean_stats(
                    context_rows, "topk1_router_loss_mean"
                ),
                "topk2_pair_gain_mean": _mean_stats(
                    context_rows, "topk2_pair_gain_mean"
                ),
                "topk1_singleton_gain_mean": _mean_stats(
                    context_rows, "topk1_singleton_gain_mean"
                ),
                "topk2_incremental_pair_gain_minus_topk1_singleton": _mean_stats(
                    context_rows,
                    "topk2_incremental_pair_gain_minus_topk1_singleton",
                ),
                "topk2_fixed_delta_minus_topk1": _mean_stats(
                    context_rows, "topk2_fixed_delta_minus_topk1"
                ),
                "topk2_logit_mse_minus_topk1": _mean_stats(
                    context_rows, "topk2_logit_mse_minus_topk1"
                ),
                "topk2_residual_stream_l2_delta_minus_topk1": _mean_stats(
                    context_rows, "topk2_residual_stream_l2_delta_minus_topk1"
                ),
                "topk2_pair_synergy_mean": _mean_stats(
                    context_rows, "topk2_pair_synergy_mean"
                ),
            }
        )
    return rows


def _aggregate_by_context_stratum(
    rows: list[dict[str, str]],
    support_count_bins: dict[int, str],
) -> dict[tuple[tuple[str, str, str, str], tuple[str, str, str, str, str]], dict[str, Any]]:
    buckets: dict[
        tuple[tuple[str, str, str, str], tuple[str, str, str, str, str]],
        list[dict[str, str]],
    ] = defaultdict(list)
    for row in rows:
        buckets[(_context_key(row), _stratum_key(row, support_count_bins))].append(row)
    return {
        context_stratum: _stats_for_rows(context_rows)
        for context_stratum, context_rows in buckets.items()
    }


def _aggregate_matched_contexts_by_stratum(
    context_stats: dict[
        tuple[tuple[str, str, str, str], tuple[str, str, str, str, str]],
        dict[str, Any],
    ]
) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for (_context, stratum), stats in context_stats.items():
        buckets[stratum].append(stats)
    return {
        stratum: {
            "context_count": len(stats_rows),
            "row_count": sum(int(row["row_count"]) for row in stats_rows),
            "active_rank_proxy": ",".join(
                sorted(
                    {
                        value
                        for row in stats_rows
                        for value in str(row["active_rank_proxy"]).split(",")
                        if value
                    }
                )
            ),
            "router_support_count_mean": _mean_stats(
                stats_rows, "router_support_count_mean"
            ),
            "router_loss_mean": _mean_stats(stats_rows, "router_loss_mean"),
            "pair_gain_mean": _mean_stats(stats_rows, "pair_gain_mean"),
            "singleton_left_gain_mean": _mean_stats(
                stats_rows, "singleton_left_gain_mean"
            ),
            "fixed_support_loss_delta_mean": _mean_stats(
                stats_rows, "fixed_support_loss_delta_mean"
            ),
            "fixed_support_logit_mse_mean": _mean_stats(
                stats_rows, "fixed_support_logit_mse_mean"
            ),
            "fixed_support_residual_stream_l2_delta_mean": _mean_stats(
                stats_rows, "fixed_support_residual_stream_l2_delta_mean"
            ),
            "pair_synergy_mean": _mean_stats(stats_rows, "pair_synergy_mean"),
        }
        for stratum, stats_rows in buckets.items()
    }


def _stats_for_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "active_rank_proxy": _joined_values(rows, "active_rank_proxy"),
        "router_support_count_mean": _mean_field(rows, "router_support_count"),
        "router_loss_mean": _mean_field(rows, "router_loss"),
        "pair_gain_mean": _mean_field(rows, "pair_gain"),
        "singleton_left_gain_mean": _mean_field(rows, "singleton_left_gain"),
        "fixed_support_loss_delta_mean": _mean_field(
            rows, "fixed_support_loss_delta"
        ),
        "fixed_support_logit_mse_mean": _mean_field(
            rows, "fixed_support_logit_mse"
        ),
        "fixed_support_residual_stream_l2_delta_mean": _mean_field(
            rows, "fixed_support_residual_stream_l2_delta"
        ),
        "pair_synergy_mean": _mean_field(rows, "pair_synergy"),
    }


def _context_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return tuple(str(row.get(field)) for field in CONTEXT_FIELDS)  # type: ignore[return-value]


def _stratum_key(
    row: dict[str, str], support_count_bins: dict[int, str]
) -> tuple[str, str, str, str, str]:
    support_count = int(_optional_float(row.get("router_support_count")) or 0)
    return (
        str(row.get("position_bin")),
        str(row.get("token_class")),
        str(row.get("residual_norm_bin")),
        str(row.get("residual_gain_bin")),
        support_count_bins.get(support_count, "unknown"),
    )


def _deconfounded_evidence(
    source_summary: dict[str, Any],
    pair_rows: list[dict[str, str]],
    matched_rows: list[dict[str, Any]],
    *,
    topk2_variant: str,
    topk1_variant: str,
    ce_guardrail_tolerance: float,
) -> dict[str, Any]:
    variants = {
        row["variant"]: row
        for row in source_summary.get("audit", {}).get("variants", [])
        if "variant" in row
    }
    topk2_ce = _optional_float(variants.get(topk2_variant, {}).get("alpha0_ce_loss"))
    topk1_ce = _optional_float(variants.get(topk1_variant, {}).get("alpha0_ce_loss"))
    ce_deficit = (
        topk2_ce - topk1_ce
        if topk2_ce is not None and topk1_ce is not None
        else None
    )
    fixed_delta_differences = [
        row["topk2_fixed_delta_minus_topk1"]
        for row in matched_rows
        if row["topk2_fixed_delta_minus_topk1"] is not None
    ]
    logit_mse_differences = [
        row["topk2_logit_mse_minus_topk1"]
        for row in matched_rows
        if row["topk2_logit_mse_minus_topk1"] is not None
    ]
    residual_l2_differences = [
        row["topk2_residual_stream_l2_delta_minus_topk1"]
        for row in matched_rows
        if row["topk2_residual_stream_l2_delta_minus_topk1"] is not None
    ]
    incremental_pair_gain_differences = [
        row["topk2_incremental_pair_gain_minus_topk1_singleton"]
        for row in matched_rows
        if row["topk2_incremental_pair_gain_minus_topk1_singleton"] is not None
    ]
    per_token_synergies = [
        row["topk2_pair_synergy_mean"]
        for row in matched_rows
        if row["topk2_pair_synergy_mean"] is not None
    ]
    coarse_synergies = [
        _optional_float(row.get("pair_synergy"))
        for row in pair_rows
        if row.get("variant") == topk2_variant
        and row.get("intervention") == TOPK2_INTERVENTION
    ]
    coarse_synergies = [value for value in coarse_synergies if value is not None]
    return {
        "matched_deconfounded_strata_count": len(matched_rows),
        "matched_exact_context_count": sum(
            int(row["matched_exact_context_count"]) for row in matched_rows
        ),
        "matched_deconfounded_tokens_topk2": sum(
            int(row["topk2_row_count"]) for row in matched_rows
        ),
        "matched_deconfounded_tokens_topk1": sum(
            int(row["topk1_row_count"]) for row in matched_rows
        ),
        "matched_dimensions": [
            "position_bin",
            "token_class",
            "residual_norm_bin",
            "residual_gain_bin",
            "support_count_bin",
        ],
        "reported_bracket_dimensions": ["active_rank_proxy"],
        "topk2_alpha0_ce_loss": topk2_ce,
        "topk1_alpha0_ce_loss": topk1_ce,
        "topk2_ce_deficit_vs_topk1": ce_deficit,
        "ce_guardrail_passed": (
            ce_deficit is not None and ce_deficit <= ce_guardrail_tolerance
        ),
        "topk2_fixed_delta_minus_topk1_mean": _mean(fixed_delta_differences),
        "topk2_fixed_support_cleaner_strata_fraction": _fraction(
            value < 0.0 for value in fixed_delta_differences
        ),
        "topk2_logit_mse_minus_topk1_mean": _mean(logit_mse_differences),
        "topk2_functional_churn_cleaner_strata_fraction": _fraction(
            value < 0.0 for value in logit_mse_differences
        ),
        "topk2_residual_l2_delta_minus_topk1_mean": _mean(residual_l2_differences),
        "topk2_incremental_pair_gain_minus_topk1_singleton_mean": _mean(
            incremental_pair_gain_differences
        ),
        "topk2_incremental_pair_gain_positive_strata_fraction": _fraction(
            value > 0.0 for value in incremental_pair_gain_differences
        ),
        "coarse_topk2_pair_synergy_mean": _mean(coarse_synergies),
        "coarse_topk2_pair_synergy_positive_fraction": _fraction(
            value > 0.0 for value in coarse_synergies
        ),
        "per_token_pair_synergy_available": bool(per_token_synergies),
        "deconfounded_topk2_pair_synergy_mean": _mean(per_token_synergies),
        "deconfounded_topk2_pair_synergy_positive_strata_fraction": _fraction(
            value > 0.0 for value in per_token_synergies
        ),
    }


def _deconfounded_decision(evidence: dict[str, Any]) -> str:
    causal_metrics_pass = (
        evidence["topk2_fixed_support_cleaner_strata_fraction"] is not None
        and evidence["topk2_fixed_support_cleaner_strata_fraction"] >= 0.8
        and evidence["topk2_functional_churn_cleaner_strata_fraction"] is not None
        and evidence["topk2_functional_churn_cleaner_strata_fraction"] >= 0.8
    )
    if causal_metrics_pass and evidence["ce_guardrail_passed"]:
        return "topk2_causal_metrics_survive_deconfounding_with_ce_guardrail"
    if causal_metrics_pass:
        return "topk2_causal_metrics_survive_deconfounding_but_ce_guardrail_fails"
    synergy_survives = (
        evidence["per_token_pair_synergy_available"]
        and evidence["deconfounded_topk2_pair_synergy_positive_strata_fraction"] is not None
        and evidence["deconfounded_topk2_pair_synergy_positive_strata_fraction"] >= 0.8
        and evidence["topk2_incremental_pair_gain_positive_strata_fraction"] is not None
        and evidence["topk2_incremental_pair_gain_positive_strata_fraction"] >= 0.8
    )
    if synergy_survives and evidence["ce_guardrail_passed"]:
        return "topk2_pair_synergy_survives_deconfounding_but_cleanliness_bar_fails"
    if evidence["ce_guardrail_passed"]:
        return "topk2_comparative_causal_cooperation_not_supported"
    return "rank_matched_topk1_remains_cleaner_causal_audit_bracket"


def _support_count_bins(rows: list[dict[str, str]]) -> dict[int, str]:
    counts = sorted(
        {
            int(value)
            for row in rows
            if (value := _optional_float(row.get("router_support_count"))) is not None
        }
    )
    if not counts:
        return {}
    low_index = min(len(counts) - 1, len(counts) // 3)
    high_index = min(len(counts) - 1, (2 * len(counts)) // 3)
    low = counts[low_index]
    high = counts[high_index]
    bins: dict[int, str] = {}
    for count in counts:
        if count <= low:
            bins[count] = "low"
        elif count <= high:
            bins[count] = "mid"
        else:
            bins[count] = "high"
    return bins


def _mean_field(rows: list[dict[str, str]], field: str) -> float | None:
    return _mean(
        [value for row in rows if (value := _optional_float(row.get(field))) is not None]
    )


def _mean_stats(rows: list[dict[str, Any]], field: str) -> float | None:
    return _mean([row[field] for row in rows if row.get(field) is not None])


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _joined_values(rows: list[dict[str, str]], field: str) -> str:
    return ",".join(sorted({str(row.get(field)) for row in rows if row.get(field) != ""}))


def _joined_stats_values(rows: list[dict[str, Any]], field: str) -> str:
    return ",".join(sorted({str(row.get(field)) for row in rows if row.get(field) != ""}))


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
        "# Deconfounded Intervention Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Source audit: `{summary['audit_dir']}`",
        f"- CE guardrail tolerance: `{evidence['ce_guardrail_tolerance']}`",
    ]
    if summary["status"] == "pass":
        lines.extend(
            [
                f"- Matched deconfounded strata: `{evidence['matched_deconfounded_strata_count']}`",
                f"- Exact matched token contexts: `{evidence['matched_exact_context_count']}`",
                f"- Top-k-2 exact contexts: `{evidence['topk2_exact_context_count']}`",
                f"- Rank-matched top-k-1 exact contexts: `{evidence['topk1_exact_context_count']}`",
                f"- Unmatched top-k-2 contexts: `{evidence['unmatched_topk2_context_count']}`",
                f"- Unmatched rank-matched top-k-1 contexts: `{evidence['unmatched_topk1_context_count']}`",
                f"- Top-k-2 alpha-0 CE: `{evidence['topk2_alpha0_ce_loss']}`",
                f"- Rank-matched top-k-1 alpha-0 CE: `{evidence['topk1_alpha0_ce_loss']}`",
                f"- Top-k-2 CE deficit: `{evidence['topk2_ce_deficit_vs_topk1']}`",
                f"- CE guardrail passed: `{evidence['ce_guardrail_passed']}`",
                f"- Top-k-2 fixed delta minus top-k-1 mean: `{evidence['topk2_fixed_delta_minus_topk1_mean']}`",
                f"- Top-k-2 fixed-support cleaner strata fraction: `{evidence['topk2_fixed_support_cleaner_strata_fraction']}`",
                f"- Top-k-2 logit MSE minus top-k-1 mean: `{evidence['topk2_logit_mse_minus_topk1_mean']}`",
                f"- Top-k-2 functional-churn cleaner strata fraction: `{evidence['topk2_functional_churn_cleaner_strata_fraction']}`",
                f"- Top-k-2 incremental pair gain minus top-k-1 singleton mean: `{evidence['topk2_incremental_pair_gain_minus_topk1_singleton_mean']}`",
                f"- Top-k-2 incremental pair-gain positive strata fraction: `{evidence['topk2_incremental_pair_gain_positive_strata_fraction']}`",
                f"- Coarse top-k-2 pair synergy mean: `{evidence['coarse_topk2_pair_synergy_mean']}`",
                f"- Per-token pair synergy available: `{evidence['per_token_pair_synergy_available']}`",
                f"- Deconfounded top-k-2 pair synergy mean: `{evidence['deconfounded_topk2_pair_synergy_mean']}`",
                f"- Deconfounded top-k-2 pair synergy positive strata fraction: `{evidence['deconfounded_topk2_pair_synergy_positive_strata_fraction']}`",
                f"- Active-rank note: {evidence['active_rank_matching_note']}",
                f"- Exact-context note: {evidence['exact_context_matching_note']}",
            ]
        )
    else:
        for failure in evidence.get("failures", []):
            lines.append(f"- Missing `{failure['field']}`: expected `{failure.get('expected')}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


_FIELDNAMES = [
    "position_bin",
    "token_class",
    "residual_norm_bin",
    "residual_gain_bin",
    "support_count_bin",
    "topk2_active_rank_proxy",
    "topk1_active_rank_proxy",
    "matched_exact_context_count",
    "topk2_row_count",
    "topk1_row_count",
    "topk2_router_support_count_mean",
    "topk1_router_support_count_mean",
    "topk2_router_loss_mean",
    "topk1_router_loss_mean",
    "topk2_pair_gain_mean",
    "topk1_singleton_gain_mean",
    "topk2_incremental_pair_gain_minus_topk1_singleton",
    "topk2_fixed_support_loss_delta_mean",
    "topk1_fixed_support_loss_delta_mean",
    "topk2_fixed_delta_minus_topk1",
    "topk2_fixed_support_logit_mse_mean",
    "topk1_fixed_support_logit_mse_mean",
    "topk2_logit_mse_minus_topk1",
    "topk2_residual_stream_l2_delta_mean",
    "topk1_residual_stream_l2_delta_mean",
    "topk2_residual_stream_l2_delta_minus_topk1",
    "topk2_pair_synergy_mean",
]

_CONTEXT_FIELDNAMES = [
    "batch_index",
    "position_index",
    "token_index",
    "target_token",
    "matched_context_stratum_count",
    "position_bins",
    "token_classes",
    "residual_norm_bins",
    "residual_gain_bins",
    "support_count_bins",
    "topk2_row_count",
    "topk1_row_count",
    "topk2_router_loss_mean",
    "topk1_router_loss_mean",
    "topk2_pair_gain_mean",
    "topk1_singleton_gain_mean",
    "topk2_incremental_pair_gain_minus_topk1_singleton",
    "topk2_fixed_delta_minus_topk1",
    "topk2_logit_mse_minus_topk1",
    "topk2_residual_stream_l2_delta_minus_topk1",
    "topk2_pair_synergy_mean",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--matched-strata-dir", type=Path, default=DEFAULT_MATCHED_STRATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--topk2-variant", default=TOPK2_VARIANT)
    parser.add_argument("--topk1-variant", default=TOPK1_VARIANT)
    parser.add_argument("--ce-guardrail-tolerance", type=float, default=CE_GUARDRAIL_TOLERANCE)
    parser.add_argument("--min-rows-per-side", type=int, default=1)
    args = parser.parse_args(argv)
    summary = run_deconfounded_intervention_audit(
        args.audit_dir,
        args.out,
        matched_strata_dir=args.matched_strata_dir,
        topk2_variant=args.topk2_variant,
        topk1_variant=args.topk1_variant,
        ce_guardrail_tolerance=args.ce_guardrail_tolerance,
        min_rows_per_side=args.min_rows_per_side,
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
