"""Context-gate suppression calibration audit for active top-k-1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from relaleap.experiments.active_topk1_post_decomposition_decision_report import (
    BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED,
    COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS,
)
from relaleap.experiments.active_topk1_runpod_post_decomposition_closeout_report import (
    RUNPOD_POST_DECOMPOSITION_VALIDATED,
)


DEFAULT_INTERFERENCE_DIR = Path(
    "results/audits/token_larger_active_topk1_context_conditioned_singleton_interference"
)
DEFAULT_CLOSEOUT_DIR = Path(
    "results/reports/token_larger_active_topk1_runpod_post_decomposition_closeout"
)
DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_active_topk1_context_gate_suppression_calibration"
)

CONTEXT_GATE_SUPPRESSION_CALIBRATION_AUDIT_ESTABLISHED = (
    "context_gate_suppression_calibration_audit_established"
)
CONTEXT_GATE_SUPPRESSION_CALIBRATION_PASSED = (
    "deployable_context_gate_suppression_calibration_passed"
)
CONTEXT_GATE_SUPPRESSION_CALIBRATION_FAILED = (
    "deployable_context_gate_suppression_calibration_failed"
)
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

GATE_FIELDS = ("position_bin", "token_class", "residual_norm_bin", "residual_gain_bin")
CONTEXT_FIELDS = ("batch_index", "position_index", "token_index", "target_token")
REQUIRED_CONTEXT_FIELDS = {
    *CONTEXT_FIELDS,
    *GATE_FIELDS,
    "own_context_singleton_gain",
    "off_context_singleton_gain",
    "off_context_singleton_harm",
    "topk2_reference_gain",
    "random_singleton_gain",
    "exhaustive_singleton_gain",
    "has_selected_context",
    "has_offcontext_match",
    "has_topk2_reference",
    "has_random_control",
    "has_exhaustive_control",
}
REQUIRED_ARTIFACTS = (
    "summary.json",
    "singleton_interference_by_context.csv",
    "singleton_interference_by_stratum.csv",
    "context_gate_holdout.csv",
    "notes.md",
)
PASS_RETAINED_GAIN_FRACTION = 0.8
PASS_RANDOM_ADVANTAGE = 0.05
PASS_HARM_SUPPRESSION_FRACTION = 0.5


def run_active_topk1_context_gate_suppression_calibration_audit(
    *,
    interference_dir: Path = DEFAULT_INTERFERENCE_DIR,
    closeout_dir: Path = DEFAULT_CLOSEOUT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Run a no-training deployable gate calibration audit on interference CSVs."""

    start = time.time()
    interference = _read_json_object(interference_dir / "summary.json")
    closeout = _read_json_object(closeout_dir / "summary.json")
    context_rows = _read_csv_rows(interference_dir / "singleton_interference_by_context.csv")
    previous_gate_rows = _read_csv_rows(interference_dir / "context_gate_holdout.csv")
    source_rows = _source_rows(interference_dir, closeout_dir, interference, closeout)
    failures = _source_failures(
        source_rows=source_rows,
        interference=interference,
        closeout=closeout,
        context_rows=context_rows,
        previous_gate_rows=previous_gate_rows,
    )

    policy_rows: list[dict[str, Any]] = []
    stratum_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        claim_status = INSUFFICIENT_EVIDENCE
        selected_policy = None
        metrics = {}
        signals = {}
        rationale = (
            "The context-gate suppression calibration audit could not run because "
            "required validated interference artifacts, controls, or fields are "
            "missing or inconsistent."
        )
        next_step = "repair_missing_context_gate_suppression_calibration_sources"
    else:
        rows = [_normalize_context_row(row) for row in context_rows]
        policy_rows, stratum_rows = _cross_validated_policy_rows(rows, previous_gate_rows)
        bootstrap_rows = _bootstrap_rows(rows)
        metrics = _summary_metrics(policy_rows, rows)
        signals = _signals(metrics)
        status = "pass"
        decision = (
            CONTEXT_GATE_SUPPRESSION_CALIBRATION_PASSED
            if signals["deployable_gate_passes_pre_registered_criteria"]
            else CONTEXT_GATE_SUPPRESSION_CALIBRATION_FAILED
        )
        claim_status = COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS
        selected_policy = "deployable_calibrated_stratum_gate"
        if signals["deployable_gate_passes_pre_registered_criteria"]:
            rationale = (
                "The deployable stratum gate keeps positive held-out singleton "
                "gain, suppresses off-context harm relative to ungated reuse, and "
                "beats coverage-matched random under the pre-registered criteria."
            )
            next_step = (
                "implement a small trainable deployable context gate and compare it "
                "against contextual top-k-2, rank-matched top-k-1, and random controls"
            )
        else:
            rationale = (
                "The no-training deployable stratum gate does not satisfy the "
                "pre-registered retained-gain, harm-suppression, and random-control "
                "criteria, so the active top-k-1 packet remains an interpretability "
                "diagnostic rather than a promotable reusable singleton mechanism."
            )
            next_step = (
                "keep top-k-1 singletons diagnostic-only and return the main "
                "architecture evidence loop to contextual top-k-2 support routing"
            )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "claim_policy": BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED,
        "selected_policy": selected_policy,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "source_rows": source_rows,
        "evidence": {
            "metrics": metrics,
            "signals": signals,
            "pass_criteria": {
                "holdout_net_gain": "> 0.0",
                "gain_minus_ungated": "> 0.0",
                "coverage_matched_random_advantage": f">= {PASS_RANDOM_ADVANTAGE}",
                "retained_gain_fraction": f">= {PASS_RETAINED_GAIN_FRACTION}",
                "offcontext_harm_suppression_fraction": (
                    f">= {PASS_HARM_SUPPRESSION_FRACTION}"
                ),
            },
            "provenance": {
                "interference_dir": str(interference_dir),
                "closeout_dir": str(closeout_dir),
                "context_csv_sha256": _sha256(
                    interference_dir / "singleton_interference_by_context.csv"
                ),
                "previous_gate_csv_sha256": _sha256(
                    interference_dir / "context_gate_holdout.csv"
                ),
                "closeout_summary_sha256": _sha256(closeout_dir / "summary.json"),
                "context_key_fields": list(CONTEXT_FIELDS),
                "deployable_gate_features": list(GATE_FIELDS),
                "cross_validation": "3 deterministic folds by position_index modulo 3",
                "oracle_policy_note": (
                    "oracle_positive_gain_gate uses holdout realized singleton gain "
                    "and is reported only as a non-deployable upper bound"
                ),
            },
        },
        "failures": failures,
        "rationale": rationale,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "policy_metrics_csv": str(out_dir / "policy_metrics.csv"),
            "stratum_decisions_csv": str(out_dir / "stratum_decisions.csv"),
            "bootstrap_intervals_csv": str(out_dir / "bootstrap_intervals.csv"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "policy_metrics.csv", _POLICY_FIELDS, policy_rows)
    _write_csv(out_dir / "stratum_decisions.csv", _STRATUM_FIELDS, stratum_rows)
    _write_csv(out_dir / "bootstrap_intervals.csv", _BOOTSTRAP_FIELDS, bootstrap_rows)
    _write_csv(
        out_dir / "source_rows.csv",
        ["source", "path", "present", "status", "decision", "sha256"],
        source_rows,
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _source_rows(
    interference_dir: Path,
    closeout_dir: Path,
    interference: dict[str, Any],
    closeout: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for source, path, payload in [
        ("interference_summary", interference_dir / "summary.json", interference),
        ("runpod_closeout", closeout_dir / "summary.json", closeout),
    ]:
        rows.append(
            {
                "source": source,
                "path": str(path),
                "present": path.is_file(),
                "status": payload.get("status"),
                "decision": payload.get("decision"),
                "sha256": _sha256(path) if path.is_file() else None,
            }
        )
    for name in REQUIRED_ARTIFACTS[1:]:
        path = interference_dir / name
        rows.append(
            {
                "source": "interference_artifact",
                "path": str(path),
                "present": path.is_file(),
                "status": "present" if path.is_file() else "missing",
                "decision": "",
                "sha256": _sha256(path) if path.is_file() else None,
            }
        )
    return rows


def _source_failures(
    *,
    source_rows: list[dict[str, Any]],
    interference: dict[str, Any],
    closeout: dict[str, Any],
    context_rows: list[dict[str, str]],
    previous_gate_rows: list[dict[str, str]],
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
    if interference.get("status") != "pass":
        failures.append(
            {
                "source": "interference_summary",
                "field": "status",
                "expected": "pass",
                "actual": interference.get("status"),
            }
        )
    if closeout.get("status") != "pass" or closeout.get("decision") != RUNPOD_POST_DECOMPOSITION_VALIDATED:
        failures.append(
            {
                "source": "runpod_closeout",
                "field": "decision",
                "expected": f"pass/{RUNPOD_POST_DECOMPOSITION_VALIDATED}",
                "actual": {
                    "status": closeout.get("status"),
                    "decision": closeout.get("decision"),
                },
            }
        )
    signals = interference.get("evidence", {}).get("signals", {})
    for field in (
        "own_context_singleton_gain_positive",
        "offcontext_singleton_interference_present",
        "matched_topk2_reference_present",
        "random_control_present",
        "exhaustive_control_present",
    ):
        if signals.get(field) is not True:
            failures.append(
                {
                    "source": "interference_summary",
                    "field": field,
                    "expected": True,
                    "actual": signals.get(field),
                }
            )
    if not context_rows:
        failures.append(
            {
                "source": "interference_context_csv",
                "field": "rows",
                "expected": "> 0",
                "actual": len(context_rows),
            }
        )
    elif missing := sorted(REQUIRED_CONTEXT_FIELDS - set(context_rows[0])):
        failures.append(
            {
                "source": "interference_context_csv",
                "field": "required_fields",
                "expected": sorted(REQUIRED_CONTEXT_FIELDS),
                "actual_missing": missing,
            }
        )
    if not previous_gate_rows:
        failures.append(
            {
                "source": "previous_context_gate_csv",
                "field": "rows",
                "expected": "> 0",
                "actual": len(previous_gate_rows),
            }
        )
    return failures


def _cross_validated_policy_rows(
    rows: list[dict[str, Any]],
    previous_gate_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    folds = sorted({_fold(row) for row in rows})
    policy_parts: dict[str, list[dict[str, float]]] = defaultdict(list)
    stratum_decisions: list[dict[str, Any]] = []
    previous_active = _previous_gate_active(previous_gate_rows)
    for fold in folds:
        train = [row for row in rows if _fold(row) != fold]
        holdout = [row for row in rows if _fold(row) == fold]
        calibrated_active, threshold, train_score = _calibrated_gate(train)
        random_active = _coverage_matched_random_keys(holdout, len(calibrated_active))
        policies = {
            "ungated_forced_singleton_reuse": { _gate_key(row) for row in holdout },
            "previous_context_gate": previous_active,
            "coverage_matched_random_gate": random_active,
            "oracle_positive_gain_gate": { _gate_key(row) for row in holdout if _own_gain(row) > 0.0 },
            "deployable_calibrated_stratum_gate": calibrated_active,
        }
        for policy, active in policies.items():
            policy_parts[policy].append(_policy_metrics(policy, holdout, active, fold))
        for key, stats in sorted(_stratum_stats(train).items()):
            stratum_decisions.append(
                {
                    "fold": fold,
                    "position_bin": key[0],
                    "token_class": key[1],
                    "residual_norm_bin": key[2],
                    "residual_gain_bin": key[3],
                    "train_context_count": stats["context_count"],
                    "train_selected_context_count": stats["selected_count"],
                    "train_net_score": stats["net_score"],
                    "train_retained_gain": stats["retained_gain"],
                    "train_offcontext_harm": stats["offcontext_harm"],
                    "selected_threshold": threshold,
                    "selected_train_objective": train_score,
                    "deployable_gate_active": key in calibrated_active,
                }
            )
    policy_rows = [_combine_policy_parts(policy, parts) for policy, parts in sorted(policy_parts.items())]
    topk2 = _reference_metrics("topk2_reference", rows, "topk2_reference_gain")
    random_singleton = _reference_metrics("random_singleton_control", rows, "random_singleton_gain")
    exhaustive = _reference_metrics("exhaustive_singleton_upper_bound", rows, "exhaustive_singleton_gain")
    return policy_rows + [topk2, random_singleton, exhaustive], stratum_decisions


def _calibrated_gate(train: list[dict[str, Any]]) -> tuple[set[tuple[str, str, str, str]], float, float]:
    stats = _stratum_stats(train)
    candidates = sorted({0.0, *[float(row["net_score"]) for row in stats.values()]})
    best_active: set[tuple[str, str, str, str]] = set()
    best_threshold = 0.0
    best_objective = float("-inf")
    for threshold in candidates:
        active = {
            key
            for key, row in stats.items()
            if row["selected_count"] > 0 and row["net_score"] >= threshold
        }
        metrics = _policy_metrics("candidate", train, active, fold=-1)
        objective = (
            metrics["holdout_net_gain"]
            - metrics["offcontext_harm_after_gate"]
            + 0.1 * metrics["retained_gain_fraction"]
        )
        if objective > best_objective:
            best_objective = objective
            best_threshold = threshold
            best_active = active
    return best_active, best_threshold, best_objective


def _stratum_stats(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, float]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_gate_key(row)].append(row)
    stats = {}
    for key, values in grouped.items():
        selected = [row for row in values if row["has_selected_context"]]
        retained_gain = _mean([_own_gain(row) for row in selected])
        harm = _mean([_harm(row) for row in values if row["has_offcontext_match"]])
        stats[key] = {
            "context_count": len(values),
            "selected_count": len(selected),
            "retained_gain": retained_gain or 0.0,
            "offcontext_harm": harm or 0.0,
            "net_score": (retained_gain or 0.0) - (harm or 0.0),
        }
    return stats


def _policy_metrics(
    policy: str,
    rows: list[dict[str, Any]],
    active: set[tuple[str, str, str, str]],
    fold: int,
) -> dict[str, Any]:
    selected_rows = [row for row in rows if row["has_selected_context"]]
    active_rows = [row for row in rows if _gate_key(row) in active]
    active_selected = [row for row in selected_rows if _gate_key(row) in active]
    ungated_harm = _mean([_harm(row) for row in rows if row["has_offcontext_match"]]) or 0.0
    harm_after_gate = _mean(
        [_harm(row) if _gate_key(row) in active else 0.0 for row in rows if row["has_offcontext_match"]]
    ) or 0.0
    all_own_gain = sum(max(_own_gain(row), 0.0) for row in selected_rows)
    accepted_own_gain = sum(max(_own_gain(row), 0.0) for row in active_selected)
    net_values = [
        _own_gain(row) if row["has_selected_context"] and _gate_key(row) in active else 0.0
        for row in rows
    ]
    return {
        "policy": policy,
        "fold": fold,
        "context_count": len(rows),
        "active_context_count": len(active_rows),
        "accepted_selected_context_count": len(active_selected),
        "accepted_coverage": _safe_div(len(active_rows), len(rows)),
        "selected_coverage": _safe_div(len(active_selected), len(selected_rows)),
        "holdout_net_gain": _mean(net_values) or 0.0,
        "mean_gain_on_accepted_selected_contexts": _mean([_own_gain(row) for row in active_selected]),
        "offcontext_harm_after_gate": harm_after_gate,
        "offcontext_harm_suppression_fraction": _safe_div(ungated_harm - harm_after_gate, ungated_harm),
        "retained_gain_fraction": _safe_div(accepted_own_gain, all_own_gain),
        "tail_net_gain_p10": _quantile(net_values, 0.10),
        "tail_net_gain_p05": _quantile(net_values, 0.05),
        "worst_net_gain": min(net_values) if net_values else None,
    }


def _combine_policy_parts(policy: str, parts: list[dict[str, Any]]) -> dict[str, Any]:
    weighted_fields = (
        "accepted_coverage",
        "selected_coverage",
        "holdout_net_gain",
        "offcontext_harm_after_gate",
        "offcontext_harm_suppression_fraction",
        "tail_net_gain_p10",
        "tail_net_gain_p05",
        "worst_net_gain",
    )
    total_context = sum(int(part["context_count"]) for part in parts)
    total_active = sum(int(part["active_context_count"]) for part in parts)
    total_selected = sum(int(part["accepted_selected_context_count"]) for part in parts)
    row = {
        "policy": policy,
        "fold": "all",
        "context_count": total_context,
        "active_context_count": total_active,
        "accepted_selected_context_count": total_selected,
        "mean_gain_on_accepted_selected_contexts": _mean(
            [
                part["mean_gain_on_accepted_selected_contexts"]
                for part in parts
                if part["mean_gain_on_accepted_selected_contexts"] is not None
            ]
        ),
        "retained_gain_fraction": _mean([part["retained_gain_fraction"] for part in parts]),
    }
    for field in weighted_fields:
        row[field] = _weighted_mean(parts, field)
    return row


def _reference_metrics(policy: str, rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [_float_or_none(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return {
        "policy": policy,
        "fold": "reference",
        "context_count": len(values),
        "active_context_count": len(values),
        "accepted_selected_context_count": "",
        "accepted_coverage": 1.0 if values else 0.0,
        "selected_coverage": "",
        "holdout_net_gain": _mean(values),
        "mean_gain_on_accepted_selected_contexts": _mean(values),
        "offcontext_harm_after_gate": "",
        "offcontext_harm_suppression_fraction": "",
        "retained_gain_fraction": "",
        "tail_net_gain_p10": _quantile(values, 0.10),
        "tail_net_gain_p05": _quantile(values, 0.05),
        "worst_net_gain": min(values) if values else None,
    }


def _bootstrap_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active, _, _ = _calibrated_gate(rows)
    per_context = [
        _own_gain(row) if row["has_selected_context"] and _gate_key(row) in active else 0.0
        for row in rows
    ]
    if not per_context:
        return []
    estimates = []
    for iteration in range(200):
        sample = [
            per_context[(iteration * 37 + index * 17) % len(per_context)]
            for index in range(len(per_context))
        ]
        estimates.append(mean(sample))
    estimates = sorted(estimates)
    return [
        {
            "policy": "deployable_calibrated_stratum_gate",
            "metric": "holdout_net_gain",
            "estimate": mean(per_context),
            "ci05": _percentile(estimates, 0.05),
            "ci95": _percentile(estimates, 0.95),
            "bootstrap_replicates": len(estimates),
            "grouping": "context",
        }
    ]


def _summary_metrics(policy_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy = {row["policy"]: row for row in policy_rows}
    deployable = by_policy["deployable_calibrated_stratum_gate"]
    ungated = by_policy["ungated_forced_singleton_reuse"]
    random_gate = by_policy["coverage_matched_random_gate"]
    oracle = by_policy["oracle_positive_gain_gate"]
    topk2 = by_policy["topk2_reference"]
    return {
        "context_count": len(rows),
        "selected_context_count": sum(1 for row in rows if row["has_selected_context"]),
        "deployable_holdout_net_gain": deployable["holdout_net_gain"],
        "deployable_gain_minus_ungated": deployable["holdout_net_gain"] - ungated["holdout_net_gain"],
        "deployable_gain_minus_coverage_matched_random": (
            deployable["holdout_net_gain"] - random_gate["holdout_net_gain"]
        ),
        "deployable_retained_gain_fraction": deployable["retained_gain_fraction"],
        "deployable_offcontext_harm_after_gate": deployable["offcontext_harm_after_gate"],
        "deployable_offcontext_harm_suppression_fraction": (
            deployable["offcontext_harm_suppression_fraction"]
        ),
        "deployable_accepted_coverage": deployable["accepted_coverage"],
        "deployable_selected_coverage": deployable["selected_coverage"],
        "deployable_tail_net_gain_p05": deployable["tail_net_gain_p05"],
        "deployable_worst_net_gain": deployable["worst_net_gain"],
        "ungated_holdout_net_gain": ungated["holdout_net_gain"],
        "ungated_offcontext_harm": ungated["offcontext_harm_after_gate"],
        "coverage_matched_random_holdout_net_gain": random_gate["holdout_net_gain"],
        "oracle_holdout_net_gain": oracle["holdout_net_gain"],
        "topk2_reference_gain_mean": topk2["holdout_net_gain"],
    }


def _signals(metrics: dict[str, Any]) -> dict[str, bool]:
    gain_positive = _gt(metrics["deployable_holdout_net_gain"], 0.0)
    improves_ungated = _gt(metrics["deployable_gain_minus_ungated"], 0.0)
    beats_random = _gte(
        metrics["deployable_gain_minus_coverage_matched_random"], PASS_RANDOM_ADVANTAGE
    )
    retains_gain = _gte(
        metrics["deployable_retained_gain_fraction"], PASS_RETAINED_GAIN_FRACTION
    )
    suppresses_harm = _gte(
        metrics["deployable_offcontext_harm_suppression_fraction"],
        PASS_HARM_SUPPRESSION_FRACTION,
    )
    return {
        "deployable_holdout_net_gain_positive": gain_positive,
        "deployable_improves_over_ungated": improves_ungated,
        "deployable_beats_coverage_matched_random": beats_random,
        "deployable_retains_enough_own_context_gain": retains_gain,
        "deployable_suppresses_offcontext_harm": suppresses_harm,
        "oracle_upper_bound_positive": _gt(metrics["oracle_holdout_net_gain"], 0.0),
        "topk2_reference_present": metrics["topk2_reference_gain_mean"] is not None,
        "deployable_gate_passes_pre_registered_criteria": (
            gain_positive and improves_ungated and beats_random and retains_gain and suppresses_harm
        ),
    }


def _normalize_context_row(row: dict[str, str]) -> dict[str, Any]:
    normalized: dict[str, Any] = dict(row)
    for field in (
        "own_context_singleton_gain",
        "off_context_singleton_gain",
        "off_context_singleton_harm",
        "topk2_reference_gain",
        "random_singleton_gain",
        "exhaustive_singleton_gain",
    ):
        normalized[field] = _float_or_none(row.get(field))
    for field in (
        "has_selected_context",
        "has_offcontext_match",
        "has_topk2_reference",
        "has_random_control",
        "has_exhaustive_control",
    ):
        normalized[field] = _bool_value(row.get(field))
    normalized["position_index"] = _int_or_zero(row.get("position_index"))
    return normalized


def _previous_gate_active(rows: list[dict[str, str]]) -> set[tuple[str, str, str, str]]:
    return {
        tuple(str(row.get(field, "")) for field in GATE_FIELDS)
        for row in rows
        if _bool_value(row.get("gate_active"))
    }


def _coverage_matched_random_keys(
    rows: list[dict[str, Any]], target_count: int
) -> set[tuple[str, str, str, str]]:
    keys = sorted({_gate_key(row) for row in rows})
    if not keys or target_count <= 0:
        return set()
    target = min(len(keys), target_count)
    ranked = sorted(keys, key=lambda key: hashlib.sha256("|".join(key).encode("utf-8")).hexdigest())
    return set(ranked[:target])


def _fold(row: dict[str, Any]) -> int:
    return _int_or_zero(row.get("position_index")) % 3


def _gate_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return tuple(str(row.get(field, "")) for field in GATE_FIELDS)


def _own_gain(row: dict[str, Any]) -> float:
    return _float_or_none(row.get("own_context_singleton_gain")) or 0.0


def _harm(row: dict[str, Any]) -> float:
    value = _float_or_none(row.get("off_context_singleton_harm"))
    return value if value is not None else 0.0


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary.get("evidence", {}).get("metrics", {})
    signals = summary.get("evidence", {}).get("signals", {})
    lines = [
        "# Active top-k-1 context-gate suppression calibration audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Claim policy: `{summary['claim_policy']}`",
        f"- Deployable holdout net gain: `{metrics.get('deployable_holdout_net_gain')}`",
        f"- Deployable gain minus ungated: `{metrics.get('deployable_gain_minus_ungated')}`",
        f"- Deployable gain minus coverage-matched random: `{metrics.get('deployable_gain_minus_coverage_matched_random')}`",
        f"- Retained own-context gain fraction: `{metrics.get('deployable_retained_gain_fraction')}`",
        f"- Off-context harm suppression fraction: `{metrics.get('deployable_offcontext_harm_suppression_fraction')}`",
        f"- Pass criteria satisfied: `{signals.get('deployable_gate_passes_pre_registered_criteria')}`",
        "",
        summary["rationale"],
        "",
        f"Next step: {summary['next_step']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _int_or_zero(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if isinstance(value, (int, float))]
    return mean(clean) if clean else None


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _weighted_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    weighted = [
        (float(row[field]), int(row["context_count"]))
        for row in rows
        if isinstance(row.get(field), (int, float)) and int(row["context_count"]) > 0
    ]
    total = sum(weight for _, weight in weighted)
    if not total:
        return None
    return sum(value * weight for value, weight in weighted) / total


def _quantile(values: Iterable[float], q: float) -> float | None:
    clean = sorted(value for value in values if isinstance(value, (int, float)))
    if not clean:
        return None
    index = min(len(clean) - 1, max(0, int(round(q * (len(clean) - 1)))))
    return clean[index]


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[index]


def _gt(left: Any, right: float) -> bool:
    return isinstance(left, (int, float)) and left > right


def _gte(left: Any, right: float) -> bool:
    return isinstance(left, (int, float)) and left >= right


_POLICY_FIELDS = [
    "policy",
    "fold",
    "context_count",
    "active_context_count",
    "accepted_selected_context_count",
    "accepted_coverage",
    "selected_coverage",
    "holdout_net_gain",
    "mean_gain_on_accepted_selected_contexts",
    "offcontext_harm_after_gate",
    "offcontext_harm_suppression_fraction",
    "retained_gain_fraction",
    "tail_net_gain_p10",
    "tail_net_gain_p05",
    "worst_net_gain",
]
_STRATUM_FIELDS = [
    "fold",
    "position_bin",
    "token_class",
    "residual_norm_bin",
    "residual_gain_bin",
    "train_context_count",
    "train_selected_context_count",
    "train_net_score",
    "train_retained_gain",
    "train_offcontext_harm",
    "selected_threshold",
    "selected_train_objective",
    "deployable_gate_active",
]
_BOOTSTRAP_FIELDS = [
    "policy",
    "metric",
    "estimate",
    "ci05",
    "ci95",
    "bootstrap_replicates",
    "grouping",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interference-dir", type=Path, default=DEFAULT_INTERFERENCE_DIR)
    parser.add_argument("--closeout-dir", type=Path, default=DEFAULT_CLOSEOUT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_active_topk1_context_gate_suppression_calibration_audit(
        interference_dir=args.interference_dir,
        closeout_dir=args.closeout_dir,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
