"""Null-controlled pair-synergy audit for causal fingerprint artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from relaleap.experiments.deconfounded_intervention_audit import (
    CE_GUARDRAIL_TOLERANCE,
    DEFAULT_AUDIT_DIR,
    DEFAULT_OUT_DIR as DEFAULT_DECONFOUNDED_DIR,
    TOPK1_VARIANT,
    TOPK2_INTERVENTION,
    TOPK2_VARIANT,
)


DEFAULT_OUT_DIR = Path("results/audits/token_larger_topk2_causal_synergy_null_audit")
CONTROL_INTERVENTION = "fixed_best_support_swap"


def run_causal_synergy_null_audit(
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    deconfounded_dir: Path = DEFAULT_DECONFOUNDED_DIR,
    topk2_variant: str = TOPK2_VARIANT,
    topk1_variant: str = TOPK1_VARIANT,
    observed_intervention: str = TOPK2_INTERVENTION,
    control_intervention: str = CONTROL_INTERVENTION,
    bootstrap_samples: int = 1000,
    seed: int = 17,
    ci_level: float = 0.95,
    ce_guardrail_tolerance: float = CE_GUARDRAIL_TOLERANCE,
    cleaner_fraction_threshold: float = 0.8,
) -> dict[str, Any]:
    """Compare observed top-k-2 pair synergy with matched null/control estimates."""

    start = time.time()
    failures: list[dict[str, Any]] = []
    per_token_path = audit_dir / "per_token_pair_interventions.csv"
    deconfounded_summary_path = deconfounded_dir / "summary.json"
    deconfounded_strata_path = deconfounded_dir / "matched_deconfounded_strata.csv"
    for field, path in (
        ("per_token_pair_interventions_csv", per_token_path),
        ("deconfounded_summary_json", deconfounded_summary_path),
        ("matched_deconfounded_strata_csv", deconfounded_strata_path),
    ):
        if not path.is_file():
            failures.append({"field": field, "expected": str(path)})

    per_token_rows: list[dict[str, str]] = []
    deconfounded_summary: dict[str, Any] = {}
    matched_keys: set[tuple[str, str, str, str, str]] = set()
    if not failures:
        with per_token_path.open(newline="", encoding="utf-8") as handle:
            per_token_rows = list(csv.DictReader(handle))
        deconfounded_summary = json.loads(
            deconfounded_summary_path.read_text(encoding="utf-8")
        )
        matched_keys = _read_matched_keys(deconfounded_strata_path)

    required_fields = {
        "variant",
        "intervention",
        "position_bin",
        "token_class",
        "router_support_count",
        "pair_gain",
        "singleton_left_gain",
        "singleton_right_gain",
        "pair_synergy",
        "residual_norm_bin",
        "residual_gain_bin",
    }
    if per_token_rows:
        missing_fields = sorted(required_fields - set(per_token_rows[0]))
        if missing_fields:
            failures.append(
                {
                    "field": "per_token_pair_synergy_fields",
                    "expected": sorted(required_fields),
                    "actual_missing": missing_fields,
                }
            )

    evidence: dict[str, Any] = {
        "failures": failures,
        "source_audit_dir": str(audit_dir),
        "deconfounded_dir": str(deconfounded_dir),
        "topk2_variant": topk2_variant,
        "topk1_variant": topk1_variant,
        "observed_intervention": observed_intervention,
        "control_intervention": control_intervention,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "ci_level": ci_level,
        "ce_guardrail_tolerance": ce_guardrail_tolerance,
        "cleaner_fraction_threshold": cleaner_fraction_threshold,
        "matched_dimensions": [
            "position_bin",
            "token_class",
            "residual_norm_bin",
            "residual_gain_bin",
            "support_count_bin",
        ],
        "null_controls": [
            "stratified sign-flip null centered at zero synergy",
            f"artifact control intervention `{control_intervention}` when present",
        ],
    }
    status = "fail" if failures else "pass"
    decision = "insufficient_evidence"
    stratum_rows: list[dict[str, Any]] = []

    if not failures:
        support_count_bins = _support_count_bins(per_token_rows)
        observed = _stratum_stats(
            per_token_rows,
            variant=topk2_variant,
            intervention=observed_intervention,
            support_count_bins=support_count_bins,
            matched_keys=matched_keys,
        )
        control = _stratum_stats(
            per_token_rows,
            variant=topk2_variant,
            intervention=control_intervention,
            support_count_bins=support_count_bins,
            matched_keys=matched_keys,
        )
        shared_keys = sorted(set(observed) & matched_keys)
        if not shared_keys:
            failures.append(
                {
                    "field": "observed_matched_pair_synergy_strata",
                    "expected": "at least one observed top-k-2 stratum shared with the deconfounded audit",
                }
            )
            status = "fail"
        else:
            control_keys = sorted(set(observed) & set(control) & matched_keys)
            stratum_rows = _summary_rows(shared_keys, control_keys, observed, control)
            evidence.update(
                _null_evidence(
                    shared_keys,
                    control_keys,
                    observed,
                    control,
                    deconfounded_summary,
                    control_intervention=control_intervention,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                    ci_level=ci_level,
                    ce_guardrail_tolerance=ce_guardrail_tolerance,
                    cleaner_fraction_threshold=cleaner_fraction_threshold,
                )
            )
            decision = _decision(evidence)

    if failures:
        evidence["failures"] = failures

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "matched_synergy_null_strata.csv", _FIELDNAMES, stratum_rows)
    summary = {
        "status": status,
        "decision": decision,
        "audit_dir": str(audit_dir),
        "deconfounded_dir": str(deconfounded_dir),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "evidence": evidence,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "matched_synergy_null_strata_csv": str(
                out_dir / "matched_synergy_null_strata.csv"
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


def _stratum_stats(
    rows: list[dict[str, str]],
    *,
    variant: str,
    intervention: str,
    support_count_bins: dict[int, str],
    matched_keys: set[tuple[str, str, str, str, str]],
) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(
        list
    )
    for row in rows:
        if row.get("variant") != variant or row.get("intervention") != intervention:
            continue
        support_count = _matching_router_support_count(row)
        key = (
            str(row.get("position_bin")),
            str(row.get("token_class")),
            str(row.get("residual_norm_bin")),
            str(row.get("residual_gain_bin")),
            support_count_bins.get(support_count, "unknown"),
        )
        if matched_keys and key not in matched_keys:
            continue
        buckets[key].append(row)
    stats: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for key, stratum_rows in buckets.items():
        synergies = _field_values(stratum_rows, "pair_synergy")
        if not synergies:
            continue
        stats[key] = {
            "row_count": len(synergies),
            "pair_synergy_mean": _mean(synergies),
            "pair_gain_mean": _mean(_field_values(stratum_rows, "pair_gain")),
            "singleton_left_gain_mean": _mean(
                _field_values(stratum_rows, "singleton_left_gain")
            ),
            "singleton_right_gain_mean": _mean(
                _field_values(stratum_rows, "singleton_right_gain")
            ),
            "router_support_count_mean": _mean(
                _field_values(stratum_rows, "router_support_count")
            ),
            "support_count_difference_mean": _mean(
                _field_values(stratum_rows, "support_count_difference")
            ),
            "support_count_difference_abs_mean": _mean(
                [
                    abs(value)
                    for value in _field_values(
                        stratum_rows,
                        "support_count_difference",
                    )
                ]
            ),
            "fixed_support_loss_difference_mean": _mean(
                _field_values(stratum_rows, "fixed_support_loss_difference")
            ),
            "fixed_support_loss_difference_abs_mean": _mean(
                [
                    abs(value)
                    for value in _field_values(
                        stratum_rows,
                        "fixed_support_loss_difference",
                    )
                ]
            ),
            "anchor_support_count": len(
                {
                    row.get("anchor_support")
                    for row in stratum_rows
                    if row.get("anchor_support")
                }
            ),
            "control_support_count": len(
                {
                    row.get("control_support") or row.get("support")
                    for row in stratum_rows
                    if row.get("control_support") or row.get("support")
                }
            ),
            "control_exact_support_count_candidate_count_mean": _mean(
                _field_values(stratum_rows, "control_exact_support_count_candidate_count")
            ),
            "control_near_support_count_candidate_count_mean": _mean(
                _field_values(stratum_rows, "control_near_support_count_candidate_count")
            ),
            "control_min_support_count_abs_difference_available_mean": _mean(
                _field_values(
                    stratum_rows,
                    "control_min_support_count_abs_difference_available",
                )
            ),
            "control_match_status_counts": _value_counts(
                row.get("control_match_status") for row in stratum_rows
            ),
        }
    return stats


def _matching_router_support_count(row: dict[str, str]) -> int:
    if row.get("anchor_support"):
        anchor_count = _optional_float(row.get("anchor_router_support_count"))
        if anchor_count is not None:
            return int(anchor_count)
    return int(_optional_float(row.get("router_support_count")) or 0)


def _null_evidence(
    observed_keys: list[tuple[str, str, str, str, str]],
    control_keys: list[tuple[str, str, str, str, str]],
    observed: dict[tuple[str, str, str, str, str], dict[str, Any]],
    control: dict[tuple[str, str, str, str, str], dict[str, Any]],
    deconfounded_summary: dict[str, Any],
    *,
    control_intervention: str,
    bootstrap_samples: int,
    seed: int,
    ci_level: float,
    ce_guardrail_tolerance: float,
    cleaner_fraction_threshold: float,
) -> dict[str, Any]:
    rng = random.Random(seed)
    observed_values = [observed[key]["pair_synergy_mean"] for key in observed_keys]
    observed_bootstrap = _bootstrap_means(observed_values, bootstrap_samples, rng)
    sign_flip_null = _sign_flip_null(observed_values, bootstrap_samples, rng)
    sign_flip_p_value = _one_sided_p_value(sign_flip_null, _mean(observed_values))
    control_available = bool(control_keys)
    control_values = [control[key]["pair_synergy_mean"] for key in control_keys]
    observed_control_values = [
        observed[key]["pair_synergy_mean"] for key in control_keys
    ]
    observed_minus_control = [
        obs - ctl for obs, ctl in zip(observed_control_values, control_values)
    ]
    observed_minus_control_bootstrap = _bootstrap_means(
        observed_minus_control, bootstrap_samples, rng
    )
    control_match_stats = _control_match_stats(
        [control[key] for key in control_keys]
    )
    deconfounded_evidence = deconfounded_summary.get("evidence", {})
    ce_deficit = deconfounded_evidence.get("topk2_ce_deficit_vs_topk1")
    fixed_cleaner = deconfounded_evidence.get(
        "topk2_fixed_support_cleaner_strata_fraction"
    )
    churn_cleaner = deconfounded_evidence.get(
        "topk2_functional_churn_cleaner_strata_fraction"
    )
    observed_ci = _ci(observed_bootstrap, ci_level)
    sign_flip_ci = _ci(sign_flip_null, ci_level)
    minus_control_ci = _ci(observed_minus_control_bootstrap, ci_level)
    sign_flip_synergy_supported = (
        observed_ci[0] is not None
        and observed_ci[0] > 0.0
        and sign_flip_p_value is not None
        and sign_flip_p_value <= 0.05
    )
    artifact_control_supported = (
        not control_available
        or (minus_control_ci[0] is not None and minus_control_ci[0] > 0.0)
    )
    pair_synergy_supported = sign_flip_synergy_supported and artifact_control_supported
    cleaner_supported = (
        ce_deficit is not None
        and ce_deficit <= ce_guardrail_tolerance
        and fixed_cleaner is not None
        and fixed_cleaner >= cleaner_fraction_threshold
        and churn_cleaner is not None
        and churn_cleaner >= cleaner_fraction_threshold
    )
    return {
        "matched_observed_strata_count": len(observed_keys),
        "matched_observed_tokens": sum(observed[key]["row_count"] for key in observed_keys),
        "observed_deconfounded_pair_synergy_mean": _mean(observed_values),
        "observed_deconfounded_pair_synergy_ci": list(observed_ci),
        "observed_positive_strata_fraction": _fraction(value > 0.0 for value in observed_values),
        "sign_flip_null_mean": _mean(sign_flip_null),
        "sign_flip_null_ci": list(sign_flip_ci),
        "sign_flip_one_sided_p_value": sign_flip_p_value,
        "control_available": control_available,
        "control_matched_strata_count": len(control_keys),
        "control_pair_synergy_mean": _mean(control_values),
        "observed_minus_control_synergy_mean": _mean(observed_minus_control),
        "observed_minus_control_synergy_ci": list(minus_control_ci),
        "control_match_diagnostics": {
            **control_match_stats,
            "matched_strata_overlap_fraction": (
                len(control_keys) / len(observed_keys) if observed_keys else None
            ),
        },
        "sign_flip_synergy_supported": sign_flip_synergy_supported,
        "artifact_control_supported": artifact_control_supported,
        "topk2_ce_deficit_vs_topk1": ce_deficit,
        "ce_guardrail_passed": ce_deficit is not None and ce_deficit <= ce_guardrail_tolerance,
        "topk2_fixed_support_cleaner_strata_fraction": fixed_cleaner,
        "topk2_functional_churn_cleaner_strata_fraction": churn_cleaner,
        "pair_synergy_supported": pair_synergy_supported,
        "cleaner_causal_bracket_supported": cleaner_supported,
        "control_note": (
            f"`{control_intervention}` is an artifact-level matched control, not a retraining control."
        ),
    }


def _decision(evidence: dict[str, Any]) -> str:
    if evidence["pair_synergy_supported"] and evidence["cleaner_causal_bracket_supported"]:
        return "pair_synergy_and_cleaner_bracket_supported_against_local_nulls"
    if evidence["pair_synergy_supported"]:
        return "pair_synergy_supported_against_local_nulls_but_cleaner_bracket_fails"
    if evidence["control_available"]:
        return "pair_synergy_not_supported_against_local_control_null"
    return "pair_synergy_requires_artifact_random_pair_controls"


def _summary_rows(
    observed_keys: list[tuple[str, str, str, str, str]],
    control_keys: list[tuple[str, str, str, str, str]],
    observed: dict[tuple[str, str, str, str, str], dict[str, Any]],
    control: dict[tuple[str, str, str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    control_key_set = set(control_keys)
    rows: list[dict[str, Any]] = []
    for key in observed_keys:
        obs = observed[key]
        ctl = control.get(key, {})
        row = {
            "position_bin": key[0],
            "token_class": key[1],
            "residual_norm_bin": key[2],
            "residual_gain_bin": key[3],
            "support_count_bin": key[4],
            "observed_row_count": obs["row_count"],
            "observed_pair_synergy_mean": obs["pair_synergy_mean"],
            "observed_pair_gain_mean": obs["pair_gain_mean"],
            "observed_singleton_left_gain_mean": obs["singleton_left_gain_mean"],
            "observed_singleton_right_gain_mean": obs["singleton_right_gain_mean"],
            "control_present": key in control_key_set,
            "control_row_count": ctl.get("row_count"),
            "control_pair_synergy_mean": ctl.get("pair_synergy_mean"),
            "control_support_count_difference_mean": ctl.get(
                "support_count_difference_mean"
            ),
            "control_support_count_difference_abs_mean": ctl.get(
                "support_count_difference_abs_mean"
            ),
            "control_fixed_support_loss_difference_mean": ctl.get(
                "fixed_support_loss_difference_mean"
            ),
            "control_fixed_support_loss_difference_abs_mean": ctl.get(
                "fixed_support_loss_difference_abs_mean"
            ),
            "control_exact_support_count_candidate_count_mean": ctl.get(
                "control_exact_support_count_candidate_count_mean"
            ),
            "control_near_support_count_candidate_count_mean": ctl.get(
                "control_near_support_count_candidate_count_mean"
            ),
            "control_min_support_count_abs_difference_available_mean": ctl.get(
                "control_min_support_count_abs_difference_available_mean"
            ),
            "control_match_status_counts": json.dumps(
                ctl.get("control_match_status_counts"),
                sort_keys=True,
            )
            if ctl.get("control_match_status_counts") is not None
            else None,
            "control_anchor_support_count": ctl.get("anchor_support_count"),
            "control_support_count": ctl.get("control_support_count"),
            "observed_minus_control_pair_synergy": (
                obs["pair_synergy_mean"] - ctl["pair_synergy_mean"]
                if key in control_key_set
                else None
            ),
        }
        rows.append(row)
    return rows


def _control_match_stats(control_stats: list[dict[str, Any]]) -> dict[str, Any]:
    if not control_stats:
        return {
            "support_count_difference_mean": None,
            "support_count_difference_abs_mean": None,
            "fixed_support_loss_difference_mean": None,
            "fixed_support_loss_difference_abs_mean": None,
            "matched_anchor_support_count": 0,
            "matched_control_support_count": 0,
            "control_exact_support_count_candidate_count_mean": None,
            "control_near_support_count_candidate_count_mean": None,
            "control_min_support_count_abs_difference_available_mean": None,
            "control_match_status_counts": {},
        }
    return {
        "support_count_difference_mean": _mean(
            [
                value
                for stat in control_stats
                if (value := stat.get("support_count_difference_mean")) is not None
            ]
        ),
        "support_count_difference_abs_mean": _mean(
            [
                value
                for stat in control_stats
                if (value := stat.get("support_count_difference_abs_mean")) is not None
            ]
        ),
        "fixed_support_loss_difference_mean": _mean(
            [
                value
                for stat in control_stats
                if (value := stat.get("fixed_support_loss_difference_mean")) is not None
            ]
        ),
        "fixed_support_loss_difference_abs_mean": _mean(
            [
                value
                for stat in control_stats
                if (value := stat.get("fixed_support_loss_difference_abs_mean")) is not None
            ]
        ),
        "matched_anchor_support_count": sum(
            int(stat.get("anchor_support_count") or 0) for stat in control_stats
        ),
        "matched_control_support_count": sum(
            int(stat.get("control_support_count") or 0) for stat in control_stats
        ),
        "control_exact_support_count_candidate_count_mean": _mean(
            [
                value
                for stat in control_stats
                if (
                    value := stat.get(
                        "control_exact_support_count_candidate_count_mean"
                    )
                )
                is not None
            ]
        ),
        "control_near_support_count_candidate_count_mean": _mean(
            [
                value
                for stat in control_stats
                if (
                    value := stat.get(
                        "control_near_support_count_candidate_count_mean"
                    )
                )
                is not None
            ]
        ),
        "control_min_support_count_abs_difference_available_mean": _mean(
            [
                value
                for stat in control_stats
                if (
                    value := stat.get(
                        "control_min_support_count_abs_difference_available_mean"
                    )
                )
                is not None
            ]
        ),
        "control_match_status_counts": _merge_counts(
            stat.get("control_match_status_counts", {}) for stat in control_stats
        ),
    }


def _read_matched_keys(path: Path) -> set[tuple[str, str, str, str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (
                row["position_bin"],
                row["token_class"],
                row["residual_norm_bin"],
                row["residual_gain_bin"],
                row["support_count_bin"],
            )
            for row in csv.DictReader(handle)
        }


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
    low = counts[min(len(counts) - 1, len(counts) // 3)]
    high = counts[min(len(counts) - 1, (2 * len(counts)) // 3)]
    bins: dict[int, str] = {}
    for count in counts:
        if count <= low:
            bins[count] = "low"
        elif count <= high:
            bins[count] = "mid"
        else:
            bins[count] = "high"
    return bins


def _bootstrap_means(
    values: list[float],
    samples: int,
    rng: random.Random,
) -> list[float]:
    if not values:
        return []
    return [
        _mean([values[rng.randrange(len(values))] for _ in values])
        for _ in range(samples)
    ]


def _sign_flip_null(
    values: list[float],
    samples: int,
    rng: random.Random,
) -> list[float]:
    if not values:
        return []
    return [
        _mean([value * (-1.0 if rng.random() < 0.5 else 1.0) for value in values])
        for _ in range(samples)
    ]


def _one_sided_p_value(null_values: list[float], observed_value: float | None) -> float | None:
    if observed_value is None or not null_values:
        return None
    exceedances = sum(1 for value in null_values if value >= observed_value)
    return float((exceedances + 1) / (len(null_values) + 1))


def _ci(values: list[float], ci_level: float) -> tuple[float | None, float | None]:
    if not values:
        return (None, None)
    sorted_values = sorted(values)
    tail = max(0.0, min(1.0, (1.0 - ci_level) / 2.0))
    low_index = min(len(sorted_values) - 1, max(0, int(math.floor(tail * len(sorted_values)))))
    high_index = min(
        len(sorted_values) - 1,
        max(0, int(math.ceil((1.0 - tail) * len(sorted_values))) - 1),
    )
    return (float(sorted_values[low_index]), float(sorted_values[high_index]))


def _field_values(rows: list[dict[str, str]], field: str) -> list[float]:
    return [
        value
        for row in rows
        if (value := _optional_float(row.get(field))) is not None
    ]


def _value_counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None or value == "":
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _merge_counts(counts_list: Any) -> dict[str, int]:
    merged: dict[str, int] = {}
    for counts in counts_list:
        if not isinstance(counts, dict):
            continue
        for key, value in counts.items():
            merged[str(key)] = merged.get(str(key), 0) + int(value)
    return dict(sorted(merged.items()))


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
        "# Causal Synergy Null Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Source audit: `{summary['audit_dir']}`",
        f"- Deconfounded audit: `{summary['deconfounded_dir']}`",
    ]
    if summary["status"] == "pass":
        lines.extend(
            [
                f"- Observed matched strata: `{evidence['matched_observed_strata_count']}`",
                f"- Observed synergy mean: `{evidence['observed_deconfounded_pair_synergy_mean']}`",
                f"- Observed synergy CI: `{evidence['observed_deconfounded_pair_synergy_ci']}`",
                f"- Sign-flip null mean: `{evidence['sign_flip_null_mean']}`",
                f"- Sign-flip one-sided p-value: `{evidence['sign_flip_one_sided_p_value']}`",
                f"- Control available: `{evidence['control_available']}`",
                f"- Control synergy mean: `{evidence['control_pair_synergy_mean']}`",
                f"- Observed-minus-control synergy mean: `{evidence['observed_minus_control_synergy_mean']}`",
                f"- Observed-minus-control CI: `{evidence['observed_minus_control_synergy_ci']}`",
                f"- Control match diagnostics: `{evidence['control_match_diagnostics']}`",
                f"- Pair synergy supported: `{evidence['pair_synergy_supported']}`",
                f"- Cleaner causal bracket supported: `{evidence['cleaner_causal_bracket_supported']}`",
                f"- Control note: {evidence['control_note']}",
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
    "observed_row_count",
    "observed_pair_synergy_mean",
    "observed_pair_gain_mean",
    "observed_singleton_left_gain_mean",
    "observed_singleton_right_gain_mean",
    "control_present",
    "control_row_count",
    "control_pair_synergy_mean",
    "control_support_count_difference_mean",
    "control_support_count_difference_abs_mean",
    "control_fixed_support_loss_difference_mean",
    "control_fixed_support_loss_difference_abs_mean",
    "control_exact_support_count_candidate_count_mean",
    "control_near_support_count_candidate_count_mean",
    "control_min_support_count_abs_difference_available_mean",
    "control_match_status_counts",
    "control_anchor_support_count",
    "control_support_count",
    "observed_minus_control_pair_synergy",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--deconfounded-dir", type=Path, default=DEFAULT_DECONFOUNDED_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--topk2-variant", default=TOPK2_VARIANT)
    parser.add_argument("--topk1-variant", default=TOPK1_VARIANT)
    parser.add_argument("--observed-intervention", default=TOPK2_INTERVENTION)
    parser.add_argument("--control-intervention", default=CONTROL_INTERVENTION)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--ce-guardrail-tolerance", type=float, default=CE_GUARDRAIL_TOLERANCE)
    parser.add_argument("--cleaner-fraction-threshold", type=float, default=0.8)
    args = parser.parse_args(argv)
    summary = run_causal_synergy_null_audit(
        args.audit_dir,
        args.out,
        deconfounded_dir=args.deconfounded_dir,
        topk2_variant=args.topk2_variant,
        topk1_variant=args.topk1_variant,
        observed_intervention=args.observed_intervention,
        control_intervention=args.control_intervention,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        ci_level=args.ci_level,
        ce_guardrail_tolerance=args.ce_guardrail_tolerance,
        cleaner_fraction_threshold=args.cleaner_fraction_threshold,
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
