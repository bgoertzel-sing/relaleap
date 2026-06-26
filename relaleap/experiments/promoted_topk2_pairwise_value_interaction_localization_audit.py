"""Localize promoted top-k-2 pairwise value interactions from source artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_promoted_topk2_pairwise_value_interaction_localization_audit"
)
DEFAULT_FINGERPRINT_DIR = Path(
    "results/audits/token_larger_support_wide_promoted_default_causal_column_fingerprint_stability_topk1"
)
DEFAULT_UPDATE_DECOMPOSITION_AUDIT = Path(
    "results/audits/token_larger_promoted_topk2_update_decomposition_audit/summary.json"
)
DEFAULT_FINITE_UPDATE_REPORT = Path(
    "results/reports/token_larger_promoted_topk2_finite_update_order_control_audit/summary.json"
)
DEFAULT_CLOSEOUT_REPORT = Path(
    "results/reports/token_larger_promoted_topk2_post_value_router_mitigation_closeout/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")

PAIRWISE_VALUE_INTERACTION_LOCALIZED = "pairwise_value_interaction_localized_hub_family"
PAIRWISE_VALUE_INTERACTION_DIFFUSE = "pairwise_value_interaction_diffuse"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def run_promoted_topk2_pairwise_value_interaction_localization_audit(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    fingerprint_dir: Path = DEFAULT_FINGERPRINT_DIR,
    update_decomposition_audit_path: Path = DEFAULT_UPDATE_DECOMPOSITION_AUDIT,
    finite_update_report_path: Path = DEFAULT_FINITE_UPDATE_REPORT,
    closeout_report_path: Path = DEFAULT_CLOSEOUT_REPORT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    top3_pair_share_threshold: float = 0.35,
    hub_share_threshold: float = 0.60,
) -> dict[str, Any]:
    """Summarize where fixed-value pair interactions concentrate."""

    start = time.time()
    per_token_path = fingerprint_dir / "per_token_pair_interventions.csv"
    pair_path = fingerprint_dir / "pair_interventions.csv"
    column_path = fingerprint_dir / "column_fingerprints.csv"
    frequency_control_path = fingerprint_dir / "support_frequency_candidate_controls.csv"
    update_decomposition = _read_json_object(update_decomposition_audit_path)
    finite_update = _read_json_object(finite_update_report_path)
    closeout = _read_json_object(closeout_report_path)
    strategy_review = _strategy_review(strategy_review_path)

    per_token_rows = _read_csv_dicts(per_token_path)
    pair_rows = _read_csv_dicts(pair_path)
    column_rows = _read_csv_dicts(column_path)
    frequency_control_rows = _read_csv_dicts(frequency_control_path)
    token_rows = _filter_pair_rows(per_token_rows)
    aggregate_pair_rows = _filter_pair_rows(
        pair_rows,
        require_all_position_token=True,
    )
    column_lookup = {
        int(row["column"]): row
        for row in column_rows
        if row.get("variant") == "baseline"
        and _int_or_none(row.get("column")) is not None
    }
    pair_lookup = {row.get("support"): row for row in aggregate_pair_rows}
    localization_rows = _localization_rows(token_rows, pair_lookup)
    column_localization_rows = _column_localization_rows(
        localization_rows,
        column_lookup,
    )
    stratum_rows = _stratum_rows(token_rows)
    split_rows = _split_rows(token_rows)
    frequency_control = _frequency_control(frequency_control_rows, localization_rows)
    metrics = _metrics(
        localization_rows,
        column_localization_rows,
        update_decomposition,
        finite_update,
        frequency_control,
        split_rows,
    )
    source_rows = [
        _source_row("per_token_pair_interventions", per_token_path, bool(per_token_rows)),
        _source_row("pair_interventions", pair_path, bool(pair_rows)),
        _source_row("column_fingerprints", column_path, bool(column_rows)),
        {
            "source": "support_frequency_candidate_controls",
            "path": str(frequency_control_path),
            "present": frequency_control_path.is_file() and bool(frequency_control_rows),
            "status": (
                "present"
                if frequency_control_path.is_file() and frequency_control_rows
                else "missing_optional"
            ),
            "decision": frequency_control.get("status"),
            "row_count": len(frequency_control_rows) if frequency_control_rows else "",
        },
        _source_row(
            "update_decomposition_audit",
            update_decomposition_audit_path,
            bool(update_decomposition),
            update_decomposition,
        ),
        _source_row(
            "finite_update_order_control",
            finite_update_report_path,
            bool(finite_update),
            finite_update,
        ),
        _source_row(
            "post_value_router_mitigation_closeout",
            closeout_report_path,
            bool(closeout),
            closeout,
        ),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy_review["present"],
            "status": "present" if strategy_review["present"] else "missing_optional",
            "decision": strategy_review["recommended_next_action"],
            "row_count": "",
        },
    ]
    failures = _failures(
        source_rows,
        token_rows,
        localization_rows,
        update_decomposition,
        closeout,
        metrics,
    )
    localization_status = _localization_status(metrics, failures)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        rationale = (
            "The localization audit cannot be interpreted because required "
            "fingerprint, decomposition, or closeout source artifacts are "
            "missing or inconsistent."
        )
        next_step = "repair missing pairwise value-interaction localization source artifacts"
    else:
        status = "pass"
        if localization_status == "hub_localized":
            decision = PAIRWISE_VALUE_INTERACTION_LOCALIZED
            rationale = (
                "The value-dominated top-k-2 interference signal is concentrated "
                "in a small fixed-support family whose leading pairs share a "
                "dominant column. Frequency and split controls do not explain "
                "away the concentration, so the next mitigation can target "
                "pairwise value composition around that hub instead of reopening "
                "router-policy pinning."
            )
            next_step = (
                "design a no-promotion column-hub pairwise value-composition "
                "mitigation candidate, then evaluate it with the existing "
                "commutator/CE/residual-norm gates"
            )
        elif localization_status == "frequency_explained":
            decision = PAIRWISE_VALUE_INTERACTION_DIFFUSE
            rationale = (
                "The apparent pairwise value-interaction concentration tracks "
                "support-frequency structure strongly enough that a hub-pair "
                "mitigation would be underjustified."
            )
            next_step = (
                "return to matched support-frequency controls before proposing "
                "another value-composition mitigation"
            )
        elif localization_status == "undetermined":
            decision = INSUFFICIENT_EVIDENCE
            rationale = (
                "The pairwise concentration signal exists, but the available "
                "frequency/null or discovery-confirmation controls are not strong "
                "enough to authorize a new mitigation family."
            )
            next_step = (
                "extend the no-training localization controls with usable "
                "frequency-matched or permutation null denominators"
            )
        else:
            decision = PAIRWISE_VALUE_INTERACTION_DIFFUSE
            rationale = (
                "The fixed-support value interaction signal is not sufficiently "
                "localized under the current thresholds and controls, so a "
                "narrow hub-pair penalty would be underjustified."
            )
            next_step = (
                "close the current value/router mitigation family and resume "
                "matched deconfounding plus retention/churn evidence"
            )

    summary = {
        "status": status,
        "decision": decision,
        "thresholds": {
            "top3_pair_abs_synergy_share": top3_pair_share_threshold,
            "dominant_column_abs_synergy_share": hub_share_threshold,
            "top_pair_null_percentile": 0.95,
            "discovery_confirmation_top3_overlap": 0.5,
            "frequency_explained_abs_correlation": 0.75,
        },
        "localization_status": localization_status,
        "metrics": metrics,
        "claim_statuses": {
            "contextual_topk2_router": "operational_default_train_time_support_selection",
            "topk2_causal_cooperation": "not_supported",
            "pairwise_value_interaction": (
                "localized_for_mitigation_design"
                if decision == PAIRWISE_VALUE_INTERACTION_LOCALIZED
                else "not_localized"
            ),
            "router_policy_mitigation": "closed_not_established",
            "order_averaging": "diagnostic_only_not_promoted",
        },
        "source_rows": source_rows,
        "localization_rows": localization_rows,
        "column_localization_rows": column_localization_rows,
        "stratum_rows": stratum_rows,
        "split_rows": split_rows,
        "frequency_control": frequency_control,
        "strategy_review": strategy_review,
        "failures": failures,
        "rationale": rationale,
        "next_step": next_step,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "localization_rows_csv": str(out_dir / "localization_rows.csv"),
            "column_localization_rows_csv": str(
                out_dir / "column_localization_rows.csv"
            ),
            "stratum_rows_csv": str(out_dir / "stratum_rows.csv"),
            "split_rows_csv": str(out_dir / "split_rows.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        out_dir / "source_rows.csv",
        ["source", "path", "present", "status", "decision", "row_count"],
        source_rows,
    )
    _write_csv(
        out_dir / "localization_rows.csv",
        [
            "support",
            "row_count",
            "columns",
            "mean_pair_synergy",
            "mean_abs_pair_synergy",
            "abs_pair_synergy_share",
            "mean_pair_gain",
            "mean_singleton_left_gain",
            "mean_singleton_right_gain",
            "mean_fixed_support_loss_delta",
            "mean_fixed_support_logit_mse",
            "mean_fixed_support_residual_stream_l2_delta",
            "mean_residual_norm",
            "router_support_count",
            "pair_value_cosine",
        ],
        localization_rows,
    )
    _write_csv(
        out_dir / "column_localization_rows.csv",
        [
            "column",
            "pair_count",
            "abs_pair_synergy_share",
            "mean_pair_synergy",
            "mean_abs_pair_synergy",
            "router_support_count",
            "router_support_fraction",
            "column_value_norm",
            "force_loss_delta",
            "ablate_loss_delta",
        ],
        column_localization_rows,
    )
    _write_csv(
        out_dir / "stratum_rows.csv",
        [
            "stratum",
            "value",
            "row_count",
            "support_count",
            "mean_pair_synergy",
            "mean_abs_pair_synergy",
            "mean_fixed_support_loss_delta",
            "mean_fixed_support_logit_mse",
            "mean_residual_norm",
        ],
        stratum_rows,
    )
    _write_csv(
        out_dir / "split_rows.csv",
        [
            "split",
            "row_count",
            "support_count",
            "top_support",
            "top3_supports",
            "top3_pair_abs_synergy_share",
        ],
        split_rows,
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _filter_pair_rows(
    rows: list[dict[str, str]],
    *,
    require_all_position_token: bool = False,
) -> list[dict[str, str]]:
    filtered = [
        row
        for row in rows
        if row.get("variant") == "baseline"
        and row.get("intervention") == "fixed_dominant_router_support"
        and _float_or_none(row.get("pair_synergy")) is not None
    ]
    if require_all_position_token:
        filtered = [
            row
            for row in filtered
            if row.get("position_bin") == "all" and row.get("token_class") == "all"
        ]
    return filtered


def _localization_rows(
    token_rows: list[dict[str, str]],
    pair_lookup: dict[str | None, dict[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in token_rows:
        grouped[row.get("support", "")].append(row)
    raw_rows: list[dict[str, Any]] = []
    total_abs = 0.0
    for support, rows in grouped.items():
        abs_sum = sum(abs(_float_or_none(row.get("pair_synergy")) or 0.0) for row in rows)
        total_abs += abs_sum
        pair_row = pair_lookup.get(support, {})
        raw_rows.append(
            {
                "support": support,
                "row_count": len(rows),
                "columns": support,
                "mean_pair_synergy": _mean_field(rows, "pair_synergy"),
                "mean_abs_pair_synergy": mean(
                    abs(_float_or_none(row.get("pair_synergy")) or 0.0)
                    for row in rows
                ),
                "_abs_pair_synergy_sum": abs_sum,
                "mean_pair_gain": _mean_field(rows, "pair_gain"),
                "mean_singleton_left_gain": _mean_field(rows, "singleton_left_gain"),
                "mean_singleton_right_gain": _mean_field(rows, "singleton_right_gain"),
                "mean_fixed_support_loss_delta": _mean_field(
                    rows, "fixed_support_loss_delta"
                ),
                "mean_fixed_support_logit_mse": _mean_field(
                    rows, "fixed_support_logit_mse"
                ),
                "mean_fixed_support_residual_stream_l2_delta": _mean_field(
                    rows, "fixed_support_residual_stream_l2_delta"
                ),
                "mean_residual_norm": _mean_field(rows, "residual_norm"),
                "router_support_count": _float_or_none(
                    pair_row.get("router_support_count")
                ),
                "pair_value_cosine": _float_or_none(pair_row.get("pair_value_cosine")),
            }
        )
    for row in raw_rows:
        row["abs_pair_synergy_share"] = (
            row["_abs_pair_synergy_sum"] / total_abs if total_abs > 0 else None
        )
        del row["_abs_pair_synergy_sum"]
    return sorted(
        raw_rows,
        key=lambda row: row.get("mean_abs_pair_synergy") or 0.0,
        reverse=True,
    )


def _column_localization_rows(
    localization_rows: list[dict[str, Any]],
    column_lookup: dict[int, dict[str, str]],
) -> list[dict[str, Any]]:
    totals: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "pair_count": 0,
            "abs_share": 0.0,
            "pair_synergies": [],
            "abs_pair_synergies": [],
        }
    )
    for row in localization_rows:
        columns = [
            column
            for column in (_int_or_none(part) for part in str(row["support"]).split(","))
            if column is not None
        ]
        for column in columns:
            totals[column]["pair_count"] += 1
            totals[column]["abs_share"] += row.get("abs_pair_synergy_share") or 0.0
            totals[column]["pair_synergies"].append(row.get("mean_pair_synergy"))
            totals[column]["abs_pair_synergies"].append(row.get("mean_abs_pair_synergy"))
    rows = []
    for column, values in totals.items():
        column_row = column_lookup.get(column, {})
        rows.append(
            {
                "column": column,
                "pair_count": values["pair_count"],
                "abs_pair_synergy_share": values["abs_share"],
                "mean_pair_synergy": _mean_numbers(values["pair_synergies"]),
                "mean_abs_pair_synergy": _mean_numbers(values["abs_pair_synergies"]),
                "router_support_count": _float_or_none(
                    column_row.get("router_support_count")
                ),
                "router_support_fraction": _float_or_none(
                    column_row.get("router_support_fraction")
                ),
                "column_value_norm": _float_or_none(column_row.get("column_value_norm")),
                "force_loss_delta": _float_or_none(column_row.get("force_loss_delta")),
                "ablate_loss_delta": _float_or_none(column_row.get("ablate_loss_delta")),
            }
        )
    return sorted(
        rows,
        key=lambda row: row.get("abs_pair_synergy_share") or 0.0,
        reverse=True,
    )


def _stratum_rows(token_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in token_rows:
        for field in ("position_bin", "token_class", "residual_norm_bin", "residual_gain_bin"):
            value = row.get(field)
            if value:
                strata[(field, value)].append(row)
    out = []
    for (field, value), rows in sorted(strata.items()):
        out.append(
            {
                "stratum": field,
                "value": value,
                "row_count": len(rows),
                "support_count": len({row.get("support") for row in rows}),
                "mean_pair_synergy": _mean_field(rows, "pair_synergy"),
                "mean_abs_pair_synergy": mean(
                    abs(_float_or_none(row.get("pair_synergy")) or 0.0)
                    for row in rows
                ),
                "mean_fixed_support_loss_delta": _mean_field(
                    rows, "fixed_support_loss_delta"
                ),
                "mean_fixed_support_logit_mse": _mean_field(
                    rows, "fixed_support_logit_mse"
                ),
                "mean_residual_norm": _mean_field(rows, "residual_norm"),
            }
        )
    return out


def _split_rows(token_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for split, split_token_rows in (
        ("even_position", [row for row in token_rows if row.get("position_bin") == "even"]),
        ("odd_position", [row for row in token_rows if row.get("position_bin") == "odd"]),
    ):
        localization_rows = _localization_rows(split_token_rows, {})
        top3 = localization_rows[:3]
        rows.append(
            {
                "split": split,
                "row_count": len(split_token_rows),
                "support_count": len(localization_rows),
                "top_support": top3[0]["support"] if top3 else None,
                "top3_supports": ";".join(str(row["support"]) for row in top3),
                "top3_pair_abs_synergy_share": sum(
                    row.get("abs_pair_synergy_share") or 0.0 for row in top3
                )
                if top3
                else None,
            }
        )
    return rows


def _frequency_control(
    rows: list[dict[str, str]],
    localization_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {
            "status": "missing_optional",
            "row_count": 0,
            "primary_denominator_count": 0,
            "candidate_abs_synergy_count": 0,
            "top_support_null_percentile": None,
            "median_support_null_percentile": None,
        }
    candidates = [
        abs(_float_or_none(row.get("candidate_pair_synergy")) or 0.0)
        for row in rows
        if _float_or_none(row.get("candidate_pair_synergy")) is not None
    ]
    primary_candidates = [
        abs(_float_or_none(row.get("candidate_pair_synergy")) or 0.0)
        for row in rows
        if row.get("included_in_primary_percentile_denominator") == "True"
        and _float_or_none(row.get("candidate_pair_synergy")) is not None
    ]
    denominator = primary_candidates or candidates
    percentiles = []
    for row in localization_rows:
        abs_synergy = row.get("mean_abs_pair_synergy")
        if abs_synergy is not None and denominator:
            percentiles.append(_percentile_rank(denominator, abs_synergy))
    status = "primary_frequency_matched"
    if not primary_candidates:
        status = "permutation_null_only_frequency_match_unavailable"
    return {
        "status": status,
        "row_count": len(rows),
        "primary_denominator_count": len(primary_candidates),
        "candidate_abs_synergy_count": len(candidates),
        "top_support_null_percentile": percentiles[0] if percentiles else None,
        "median_support_null_percentile": _median(percentiles),
    }


def _metrics(
    localization_rows: list[dict[str, Any]],
    column_localization_rows: list[dict[str, Any]],
    update_decomposition: dict[str, Any],
    finite_update: dict[str, Any],
    frequency_control: dict[str, Any],
    split_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    top_row = localization_rows[0] if localization_rows else {}
    top3_share = sum(
        row.get("abs_pair_synergy_share") or 0.0 for row in localization_rows[:3]
    )
    update_metrics = update_decomposition.get("metrics", {})
    finite_metrics = finite_update.get("metrics", {})
    dominant_column = column_localization_rows[0] if column_localization_rows else {}
    shares = [
        row.get("abs_pair_synergy_share") or 0.0
        for row in localization_rows
        if row.get("abs_pair_synergy_share") is not None
    ]
    top5_share = sum(shares[:5])
    router_counts = [row.get("router_support_count") for row in localization_rows]
    abs_synergies = [row.get("mean_abs_pair_synergy") for row in localization_rows]
    split_top3_sets = [
        {
            support
            for support in str(row.get("top3_supports") or "").split(";")
            if support
        }
        for row in split_rows
    ]
    split_overlap = None
    if len(split_top3_sets) == 2 and split_top3_sets[0] and split_top3_sets[1]:
        split_overlap = len(split_top3_sets[0] & split_top3_sets[1]) / min(
            len(split_top3_sets[0]), len(split_top3_sets[1])
        )
    return {
        "per_token_row_count": sum(row.get("row_count") or 0 for row in localization_rows),
        "support_pair_count": len(localization_rows),
        "top_support": top_row.get("support"),
        "top_support_mean_pair_synergy": top_row.get("mean_pair_synergy"),
        "top_support_mean_abs_pair_synergy": top_row.get("mean_abs_pair_synergy"),
        "top_pair_abs_synergy_share": top_row.get("abs_pair_synergy_share"),
        "top3_pair_abs_synergy_share": top3_share if localization_rows else None,
        "top5_pair_abs_synergy_share": top5_share if localization_rows else None,
        "pair_abs_synergy_hhi": sum(share * share for share in shares)
        if shares
        else None,
        "pair_abs_synergy_gini": _gini(shares) if shares else None,
        "dominant_column": dominant_column.get("column"),
        "dominant_column_abs_synergy_share": dominant_column.get(
            "abs_pair_synergy_share"
        ),
        "dominant_column_pair_count": dominant_column.get("pair_count"),
        "value_only_fraction_of_full": _float_or_none(
            update_metrics.get("value_only_fraction_of_full")
        ),
        "router_only_fraction_of_full": _float_or_none(
            update_metrics.get("router_only_fraction_of_full")
        ),
        "topk2_mean_commutator_anchor_support_churn": _float_or_none(
            finite_metrics.get("topk2_mean_commutator_anchor_support_churn")
        ),
        "topk2_mean_commutator_anchor_logit_mse": _float_or_none(
            finite_metrics.get("topk2_mean_commutator_anchor_logit_mse")
        ),
        "topk2_mean_commutator_anchor_residual_stream_l2": _float_or_none(
            finite_metrics.get("topk2_mean_commutator_anchor_residual_stream_l2")
        ),
        "router_support_count_abs_synergy_correlation": _pearson(
            router_counts,
            abs_synergies,
        ),
        "discovery_confirmation_top3_overlap": split_overlap,
        "frequency_control_status": frequency_control.get("status"),
        "frequency_control_primary_denominator_count": frequency_control.get(
            "primary_denominator_count"
        ),
        "top_support_null_percentile": frequency_control.get(
            "top_support_null_percentile"
        ),
        "median_support_null_percentile": frequency_control.get(
            "median_support_null_percentile"
        ),
    }


def _localization_status(
    metrics: dict[str, Any],
    failures: list[dict[str, Any]],
) -> str:
    if failures:
        return "undetermined"
    has_primary_frequency_control = (
        (metrics.get("frequency_control_primary_denominator_count") or 0) > 0
    )
    frequency_corr = abs(
        _float_or_none(metrics.get("router_support_count_abs_synergy_correlation"))
        or 0.0
    )
    localized_pair = _at_least(metrics.get("top3_pair_abs_synergy_share"), 0.35)
    localized_hub = _at_least(metrics.get("dominant_column_abs_synergy_share"), 0.60)
    null_survives = _at_least(metrics.get("top_support_null_percentile"), 0.95)
    split_survives = _at_least(metrics.get("discovery_confirmation_top3_overlap"), 0.5)
    if has_primary_frequency_control and frequency_corr >= 0.75 and not null_survives:
        return "frequency_explained"
    if localized_pair and localized_hub and null_survives and split_survives:
        return "hub_localized"
    if localized_pair and localized_hub and (
        metrics.get("top_support_null_percentile") is None
        or metrics.get("discovery_confirmation_top3_overlap") is None
    ):
        return "undetermined"
    return "diffuse"


def _failures(
    source_rows: list[dict[str, Any]],
    token_rows: list[dict[str, str]],
    localization_rows: list[dict[str, Any]],
    update_decomposition: dict[str, Any],
    closeout: dict[str, Any],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    failures = []
    required_source_names = {
        "per_token_pair_interventions",
        "pair_interventions",
        "column_fingerprints",
        "update_decomposition_audit",
        "finite_update_order_control",
        "post_value_router_mitigation_closeout",
    }
    required_sources = [
        row for row in source_rows if row.get("source") in required_source_names
    ]
    for row in required_sources:
        if not row["present"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "source_artifact",
                    "expected": "present",
                    "actual": "missing",
                    "path": row["path"],
                }
            )
    if not token_rows:
        failures.append(
            {
                "source": "per_token_pair_interventions",
                "field": "baseline fixed_dominant_router_support rows",
                "expected": "nonempty",
                "actual": 0,
            }
        )
    if len(localization_rows) < 2:
        failures.append(
            {
                "source": "localization_rows",
                "field": "support_pair_count",
                "expected": ">= 2",
                "actual": len(localization_rows),
            }
        )
    if update_decomposition.get("decision") != "value_update_dominated_order_sensitivity":
        failures.append(
            {
                "source": "update_decomposition_audit",
                "field": "decision",
                "expected": "value_update_dominated_order_sensitivity",
                "actual": update_decomposition.get("decision"),
            }
        )
    if (
        closeout.get("selected_next_action")
        != "pairwise_value_interaction_localization_audit"
    ):
        failures.append(
            {
                "source": "post_value_router_mitigation_closeout",
                "field": "selected_next_action",
                "expected": "pairwise_value_interaction_localization_audit",
                "actual": closeout.get("selected_next_action"),
            }
        )
    for field in (
        "top3_pair_abs_synergy_share",
        "dominant_column_abs_synergy_share",
        "value_only_fraction_of_full",
        "router_only_fraction_of_full",
    ):
        if metrics.get(field) is None:
            failures.append(
                {
                    "source": "metrics",
                    "field": field,
                    "expected": "numeric",
                    "actual": None,
                }
            )
    return failures


def _source_row(
    source: str,
    path: Path,
    present: bool,
    packet: dict[str, Any] | None = None,
) -> dict[str, Any]:
    packet = packet or {}
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file() and present,
        "status": packet.get("status") or ("present" if path.is_file() else "missing"),
        "decision": packet.get("decision") or packet.get("selected_next_action"),
        "row_count": "",
    }


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "strategic_change_level": None,
            "notify_ben": None,
            "recommended_next_action": None,
            "incorporation": "optional review not present",
            "ben_notification_required": False,
        }
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:12]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() in {
            "strategic_change_level",
            "notify_ben",
            "recommended_next_action",
        }:
            header[key.strip()] = value.strip()
    notify_ben = _bool_or_none(header.get("notify_ben"))
    major = header.get("strategic_change_level") == "major"
    return {
        "path": str(path),
        "present": True,
        "strategic_change_level": header.get("strategic_change_level"),
        "notify_ben": notify_ben,
        "recommended_next_action": header.get("recommended_next_action"),
        "incorporation": (
            "accepted the recommendation to keep contextual top-k-2 as an "
            "operational default while withholding causal-cooperation claims; "
            "this audit sharpens the value-composition branch before any new "
            "training"
        ),
        "ben_notification_required": bool(notify_ben) or major,
    }


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
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def _mean_field(rows: list[dict[str, str]], field: str) -> float | None:
    return _mean_numbers(_float_or_none(row.get(field)) for row in rows)


def _mean_numbers(values: Any) -> float | None:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return mean(numeric)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _percentile_rank(values: list[float], value: float) -> float | None:
    if not values:
        return None
    return sum(1 for candidate in values if candidate <= value) / len(values)


def _pearson(xs: list[Any], ys: list[Any]) -> float | None:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if x is not None and y is not None
    ]
    if len(pairs) < 2:
        return None
    mean_x = mean(x for x, _ in pairs)
    mean_y = mean(y for _, y in pairs)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denom_x = sum((x - mean_x) ** 2 for x, _ in pairs) ** 0.5
    denom_y = sum((y - mean_y) ** 2 for _, y in pairs) ** 0.5
    if denom_x == 0.0 or denom_y == 0.0:
        return None
    return numerator / (denom_x * denom_y)


def _gini(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(value for value in values if value >= 0.0)
    if not ordered:
        return None
    total = sum(ordered)
    if total == 0.0:
        return 0.0
    n = len(ordered)
    weighted_sum = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n


def _at_least(value: Any, threshold: float) -> bool:
    numeric = _float_or_none(value)
    return numeric is not None and numeric >= threshold


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    lines = [
        "# Promoted Top-k-2 Pairwise Value-Interaction Localization Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Top support: `{metrics['top_support']}`",
        f"- Top-3 pair absolute-synergy share: `{metrics['top3_pair_abs_synergy_share']}`",
        f"- Dominant column: `{metrics['dominant_column']}`",
        "- Dominant-column absolute-synergy share: "
        f"`{metrics['dominant_column_abs_synergy_share']}`",
        "- Value-only/full commutator fraction: "
        f"`{metrics['value_only_fraction_of_full']}`",
        "",
        "## Interpretation",
        "",
        summary["rationale"],
        "",
        "## Claim Policy",
        "",
        "This is a mitigation-design localization audit only. It does not "
        "promote top-k-2 causal-cooperation claims.",
        "",
        "## Next Step",
        "",
        summary["next_step"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fingerprint-dir", type=Path, default=DEFAULT_FINGERPRINT_DIR)
    parser.add_argument(
        "--update-decomposition-audit",
        type=Path,
        default=DEFAULT_UPDATE_DECOMPOSITION_AUDIT,
    )
    parser.add_argument("--finite-update-report", type=Path, default=DEFAULT_FINITE_UPDATE_REPORT)
    parser.add_argument("--closeout-report", type=Path, default=DEFAULT_CLOSEOUT_REPORT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--top3-pair-share-threshold", type=float, default=0.35)
    parser.add_argument("--hub-share-threshold", type=float, default=0.60)
    args = parser.parse_args(argv)
    summary = run_promoted_topk2_pairwise_value_interaction_localization_audit(
        out_dir=args.out,
        fingerprint_dir=args.fingerprint_dir,
        update_decomposition_audit_path=args.update_decomposition_audit,
        finite_update_report_path=args.finite_update_report,
        closeout_report_path=args.closeout_report,
        strategy_review_path=args.strategy_review,
        top3_pair_share_threshold=args.top3_pair_share_threshold,
        hub_share_threshold=args.hub_share_threshold,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "localization_status": summary["localization_status"],
                "metrics": summary["metrics"],
                "next_step": summary["next_step"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
