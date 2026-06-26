"""Retention synthesis gate for the promoted contextual top-k-2 router."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from relaleap.experiments.promoted_topk2_finite_update_order_control_audit import (
    FINITE_UPDATE_ORDER_SENSITIVITY_CE_BOUNDED,
)
from relaleap.experiments.promoted_topk2_functional_churn_control_audit import (
    FUNCTIONAL_CHURN_BOUNDED_WITH_COMMUTATOR_RISK,
)
from relaleap.experiments.promoted_topk2_support_selection_quality_audit import (
    PROMOTED_TOPK2_SUPPORT_SELECTION_QUALITY_ESTABLISHED,
)


DEFAULT_MICROTEST_DIRS = (
    Path(
        "results/runpod_fetch/audits/"
        "runpod_token_larger_task_free_anchor_retention_matrix_20260626"
    ),
    Path("results/runpod_fetch/audits/runpod_token_larger_retention_churn_microtest_seed2"),
)
DEFAULT_FINITE_UPDATE_DIR = Path(
    "results/reports/token_larger_promoted_topk2_finite_update_order_control_audit"
)
DEFAULT_FUNCTIONAL_CHURN_DIR = Path(
    "results/reports/token_larger_promoted_topk2_functional_churn_control_audit"
)
DEFAULT_SUPPORT_SELECTION_DIR = Path(
    "results/reports/token_larger_promoted_topk2_support_selection_quality_audit"
)
DEFAULT_DECONFOUNDED_DIR = Path(
    "results/audits/token_larger_topk2_vs_rank_matched_topk1_deconfounded_intervention"
)
DEFAULT_CONTEXT_GATE_DIR = Path(
    "results/audits/token_larger_active_topk1_context_gate_suppression_calibration"
)
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_promoted_topk2_retention_synthesis_gate"
)

RETENTION_SEPARABILITY_RISK_MITIGATION_RECOMMENDED = (
    "retention_separability_risk_mitigation_recommended"
)
CONTEXTUAL_TOPK2_ROUTER_DEFAULT_TOPK1_DIAGNOSTIC = (
    "contextual_topk2_router_default_topk1_diagnostic"
)
ANOTHER_RETENTION_SEED_RECOMMENDED = "another_retention_seed_recommended"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
TOPK1_GATE_FAILED = "deployable_context_gate_suppression_calibration_failed"

_REQUIRED_VARIANTS = (
    "promoted_contextual_topk2",
    "rank_matched_contextual_topk1",
    "random_fixed_topk2",
    "norm_matched_dense_active_rank",
)


def run_promoted_topk2_retention_synthesis_gate(
    *,
    microtest_dirs: tuple[Path, ...] = DEFAULT_MICROTEST_DIRS,
    finite_update_dir: Path = DEFAULT_FINITE_UPDATE_DIR,
    functional_churn_dir: Path = DEFAULT_FUNCTIONAL_CHURN_DIR,
    support_selection_dir: Path = DEFAULT_SUPPORT_SELECTION_DIR,
    deconfounded_dir: Path = DEFAULT_DECONFOUNDED_DIR,
    context_gate_dir: Path | None = DEFAULT_CONTEXT_GATE_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    high_support_churn_threshold: float = 0.5,
    commutator_ratio_threshold: float = 10.0,
    low_topk1_support_churn_threshold: float = 0.05,
) -> dict[str, Any]:
    """Choose the next top-k-2 retention action from completed source packets."""

    finite_update = _read_json_object(finite_update_dir / "summary.json")
    functional_churn = _read_json_object(functional_churn_dir / "summary.json")
    support_selection = _read_json_object(support_selection_dir / "summary.json")
    deconfounded = _read_json_object(deconfounded_dir / "summary.json")
    context_gate = (
        _read_json_object(context_gate_dir / "summary.json")
        if context_gate_dir is not None and (context_gate_dir / "summary.json").is_file()
        else {}
    )
    microtest_rows = [
        _microtest_row(index, path) for index, path in enumerate(microtest_dirs, start=1)
    ]
    source_rows = [
        *[
            _source_row(f"retention_microtest_seed{index}", path / "summary.json")
            for index, path in enumerate(microtest_dirs, start=1)
        ],
        _source_row("finite_update_order_control", finite_update_dir / "summary.json", finite_update),
        _source_row("functional_churn_control", functional_churn_dir / "summary.json", functional_churn),
        _source_row("support_selection_quality", support_selection_dir / "summary.json", support_selection),
        _source_row("deconfounded_intervention", deconfounded_dir / "summary.json", deconfounded),
        *(
            [
                _source_row(
                    "context_gate_suppression_calibration",
                    context_gate_dir / "summary.json",
                    context_gate,
                )
            ]
            if context_gate
            else []
        ),
    ]
    metrics = _metrics(microtest_rows, finite_update, support_selection, deconfounded)
    signals = _signals(
        metrics,
        finite_update,
        functional_churn,
        support_selection,
        deconfounded,
        context_gate,
        high_support_churn_threshold=high_support_churn_threshold,
        commutator_ratio_threshold=commutator_ratio_threshold,
        low_topk1_support_churn_threshold=low_topk1_support_churn_threshold,
    )
    failures = _failures(source_rows, microtest_rows, finite_update, functional_churn, support_selection)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        rationale = (
            "The retention synthesis gate cannot choose a research action because "
            "a required RunPod retention packet or no-training source report is "
            "missing, failing, or lacks required variant metrics."
        )
        next_step = "repair_missing_retention_synthesis_source_packets"
    elif (
        signals["topk2_transfer_beats_random_and_dense"]
        and signals["topk2_support_selection_quality_established"]
        and signals["topk2_high_support_churn_replicated"]
        and signals["topk2_commutator_risk_replicated"]
        and signals["topk1_low_churn_replicated"]
        and signals["topk1_transfer_competitive"]
        and signals["topk1_context_gate_failed"]
    ):
        status = "pass"
        decision = CONTEXTUAL_TOPK2_ROUTER_DEFAULT_TOPK1_DIAGNOSTIC
        rationale = (
            "The newest fetched RunPod anchor-retention matrix repeats the same "
            "tradeoff: rank-matched contextual top-k-1 is cleaner on support churn "
            "and finite-update commutators and is transfer-competitive, but the "
            "deployable context-gate suppression audit failed. That blocks a "
            "scientific shift to top-k-1 as a reusable singleton mechanism. "
            "Promoted contextual top-k-2 should remain the router default for CE "
            "and support-selection evidence while top-k-1 stays a diagnostic "
            "retention bracket. The next non-duplicative step is to probe "
            "finite-update order symmetrization rather than adding more low-rank "
            "value or top-k-1 singleton-gate variants."
        )
        next_step = (
            "run a local no-training finite-update order-symmetrization audit for "
            "promoted contextual top-k-2, retaining rank-matched top-k-1, random "
            "fixed top-k-2, and dense active-rank controls"
        )
    elif (
        signals["topk2_transfer_beats_random_and_dense"]
        and signals["topk2_support_selection_quality_established"]
        and signals["topk2_high_support_churn_replicated"]
        and signals["topk2_commutator_risk_replicated"]
        and signals["topk1_low_churn_replicated"]
        and signals["topk2_causal_cooperation_not_supported"]
    ):
        status = "pass"
        decision = RETENTION_SEPARABILITY_RISK_MITIGATION_RECOMMENDED
        rationale = (
            "Across the two fetched RunPod task-free retention packets, promoted "
            "contextual top-k-2 remains a strong train-time transfer router versus "
            "random fixed top-k-2 and dense active-rank controls, while support "
            "selection quality is already established by low oracle-support regret. "
            "The same packets replicate high support churn and a much larger "
            "finite-update commutator than rank-matched top-k-1, and the "
            "deconfounded causal packet does not support broad top-k-2 causal "
            "cooperation. Another seed would mostly measure stability of a known "
            "risk; the higher-information next step is a bounded support-stability "
            "or finite-update mitigation experiment."
        )
        next_step = (
            "run a bounded support-stability or finite-update mitigation probe "
            "against promoted contextual top-k-2, with rank-matched top-k-1, "
            "random fixed top-k-2, and dense active-rank controls retained"
        )
    else:
        status = "pass"
        decision = ANOTHER_RETENTION_SEED_RECOMMENDED
        rationale = (
            "The current synthesis does not yet show a replicated top-k-2 "
            "retention/separability risk strong enough to spend the next run on "
            "mitigation. A further fresh retention seed is the more conservative "
            "next evidence step."
        )
        next_step = (
            "run one more fresh task-free retention seed before choosing a "
            "mitigation mechanism"
        )

    summary = {
        "status": status,
        "decision": decision,
        "out_dir": str(out_dir),
        "source_rows": source_rows,
        "thresholds": {
            "high_support_churn_threshold": high_support_churn_threshold,
            "commutator_ratio_threshold": commutator_ratio_threshold,
            "low_topk1_support_churn_threshold": low_topk1_support_churn_threshold,
        },
        "microtest_rows": microtest_rows,
        "metrics": metrics,
        "signals": signals,
        "failures": failures,
        "rationale": rationale,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "retention_seed_metrics_csv": str(out_dir / "retention_seed_metrics.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_rows.csv", source_rows)
    _write_csv(out_dir / "retention_seed_metrics.csv", microtest_rows)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _microtest_row(index: int, path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    summary = _read_json_object(summary_path)
    variants = {
        str(row.get("variant")): row
        for row in summary.get("audit", {}).get("variants", [])
        if isinstance(row, dict)
    }
    row: dict[str, Any] = {
        "seed": index,
        "microtest_dir": str(path),
        "summary_path": str(summary_path),
        "summary_present": summary_path.is_file(),
        "status": summary.get("status"),
        "config_path": summary.get("config_path"),
        "device": summary.get("device"),
        "cuda_device_name": summary.get("cuda_device_name"),
        "required_variants_present": all(name in variants for name in _REQUIRED_VARIANTS),
    }
    for variant in _REQUIRED_VARIANTS:
        prefix = _variant_prefix(variant)
        source = variants.get(variant, {})
        row[f"{prefix}_transfer_ce_improvement"] = _float_or_none(
            source.get("transfer_ce_improvement")
        )
        row[f"{prefix}_anchor_ce_drift"] = _float_or_none(source.get("anchor_ce_drift"))
        row[f"{prefix}_anchor_support_churn_after_transfer"] = _float_or_none(
            source.get("anchor_support_churn_after_transfer")
        )
        row[f"{prefix}_commutator_anchor_logit_mse"] = _float_or_none(
            source.get("commutator_anchor_logit_mse")
        )
        row[f"{prefix}_commutator_anchor_ce_abs_delta"] = _float_or_none(
            source.get("commutator_anchor_ce_abs_delta")
        )
        row[f"{prefix}_commutator_anchor_residual_stream_l2"] = _float_or_none(
            source.get("commutator_anchor_residual_stream_l2")
        )
    row["topk2_transfer_advantage_vs_random_fixed_topk2"] = _delta(
        row["topk2_transfer_ce_improvement"],
        row["random_fixed_topk2_transfer_ce_improvement"],
    )
    row["topk2_transfer_advantage_vs_dense"] = _delta(
        row["topk2_transfer_ce_improvement"],
        row["dense_transfer_ce_improvement"],
    )
    row["topk2_minus_topk1_support_churn"] = _delta(
        row["topk2_anchor_support_churn_after_transfer"],
        row["topk1_anchor_support_churn_after_transfer"],
    )
    row["topk2_to_topk1_commutator_anchor_logit_mse_ratio"] = _ratio(
        row["topk2_commutator_anchor_logit_mse"],
        row["topk1_commutator_anchor_logit_mse"],
    )
    return row


def _metrics(
    microtest_rows: list[dict[str, Any]],
    finite_update: dict[str, Any],
    support_selection: dict[str, Any],
    deconfounded: dict[str, Any],
) -> dict[str, Any]:
    deconfounded_evidence = (
        deconfounded.get("evidence", {}) if isinstance(deconfounded.get("evidence"), dict) else {}
    )
    support_metrics = (
        support_selection.get("metrics", {})
        if isinstance(support_selection.get("metrics"), dict)
        else {}
    )
    finite_metrics = (
        finite_update.get("metrics", {})
        if isinstance(finite_update.get("metrics"), dict)
        else {}
    )
    return {
        "retention_packet_count": len(microtest_rows),
        "mean_topk2_transfer_ce_improvement": _mean_field(
            microtest_rows, "topk2_transfer_ce_improvement"
        ),
        "mean_topk1_transfer_ce_improvement": _mean_field(
            microtest_rows, "topk1_transfer_ce_improvement"
        ),
        "mean_topk2_transfer_improvement_minus_topk1": _mean_delta_fields(
            microtest_rows,
            "topk2_transfer_ce_improvement",
            "topk1_transfer_ce_improvement",
        ),
        "mean_random_fixed_topk2_transfer_ce_improvement": _mean_field(
            microtest_rows, "random_fixed_topk2_transfer_ce_improvement"
        ),
        "mean_dense_transfer_ce_improvement": _mean_field(
            microtest_rows, "dense_transfer_ce_improvement"
        ),
        "min_topk2_transfer_advantage_vs_random_fixed_topk2": _min_field(
            microtest_rows, "topk2_transfer_advantage_vs_random_fixed_topk2"
        ),
        "min_topk2_transfer_advantage_vs_dense": _min_field(
            microtest_rows, "topk2_transfer_advantage_vs_dense"
        ),
        "mean_topk2_support_churn_after_transfer": _mean_field(
            microtest_rows, "topk2_anchor_support_churn_after_transfer"
        ),
        "mean_topk1_support_churn_after_transfer": _mean_field(
            microtest_rows, "topk1_anchor_support_churn_after_transfer"
        ),
        "min_topk2_minus_topk1_support_churn": _min_field(
            microtest_rows, "topk2_minus_topk1_support_churn"
        ),
        "mean_topk2_commutator_anchor_logit_mse": _mean_field(
            microtest_rows, "topk2_commutator_anchor_logit_mse"
        ),
        "mean_topk1_commutator_anchor_logit_mse": _mean_field(
            microtest_rows, "topk1_commutator_anchor_logit_mse"
        ),
        "min_topk2_to_topk1_commutator_anchor_logit_mse_ratio": _min_field(
            microtest_rows, "topk2_to_topk1_commutator_anchor_logit_mse_ratio"
        ),
        "mean_topk2_commutator_anchor_ce_abs_delta": _mean_field(
            microtest_rows, "topk2_commutator_anchor_ce_abs_delta"
        ),
        "finite_update_topk2_to_topk1_commutator_ratio": finite_metrics.get(
            "topk2_to_topk1_mean_commutator_anchor_logit_mse_ratio"
        ),
        "oracle_support_regret": support_metrics.get("oracle_support_regret"),
        "oracle_support_regret_positive_fraction": support_metrics.get(
            "oracle_support_regret_positive_fraction"
        ),
        "topk2_ce_deficit_vs_topk1": deconfounded_evidence.get(
            "topk2_ce_deficit_vs_topk1"
        ),
        "topk2_incremental_pair_gain_positive_strata_fraction": deconfounded_evidence.get(
            "topk2_incremental_pair_gain_positive_strata_fraction"
        ),
        "topk2_fixed_support_cleaner_strata_fraction": deconfounded_evidence.get(
            "topk2_fixed_support_cleaner_strata_fraction"
        ),
        "topk2_functional_churn_cleaner_strata_fraction": deconfounded_evidence.get(
            "topk2_functional_churn_cleaner_strata_fraction"
        ),
    }


def _signals(
    metrics: dict[str, Any],
    finite_update: dict[str, Any],
    functional_churn: dict[str, Any],
    support_selection: dict[str, Any],
    deconfounded: dict[str, Any],
    context_gate: dict[str, Any],
    *,
    high_support_churn_threshold: float,
    commutator_ratio_threshold: float,
    low_topk1_support_churn_threshold: float,
) -> dict[str, bool]:
    return {
        "two_retention_packets_present": metrics["retention_packet_count"] >= 2,
        "topk2_transfer_beats_random_and_dense": _positive(
            metrics.get("min_topk2_transfer_advantage_vs_random_fixed_topk2")
        )
        and _positive(metrics.get("min_topk2_transfer_advantage_vs_dense")),
        "topk1_transfer_competitive": _at_most(
            metrics.get("mean_topk2_transfer_improvement_minus_topk1"), 0.02
        ),
        "topk2_high_support_churn_replicated": _at_least(
            metrics.get("mean_topk2_support_churn_after_transfer"),
            high_support_churn_threshold,
        )
        and _at_least(metrics.get("min_topk2_minus_topk1_support_churn"), high_support_churn_threshold),
        "topk1_low_churn_replicated": _at_most(
            metrics.get("mean_topk1_support_churn_after_transfer"),
            low_topk1_support_churn_threshold,
        ),
        "topk2_commutator_risk_replicated": _at_least(
            metrics.get("min_topk2_to_topk1_commutator_anchor_logit_mse_ratio"),
            commutator_ratio_threshold,
        ),
        "finite_update_ce_bounded_but_material": finite_update.get("decision")
        == FINITE_UPDATE_ORDER_SENSITIVITY_CE_BOUNDED,
        "functional_churn_bounded_with_commutator_risk": functional_churn.get("decision")
        == FUNCTIONAL_CHURN_BOUNDED_WITH_COMMUTATOR_RISK,
        "topk2_support_selection_quality_established": support_selection.get("decision")
        == PROMOTED_TOPK2_SUPPORT_SELECTION_QUALITY_ESTABLISHED,
        "topk2_causal_cooperation_not_supported": deconfounded.get("decision")
        == "topk2_comparative_causal_cooperation_not_supported",
        "topk1_context_gate_failed": context_gate.get("decision") == TOPK1_GATE_FAILED,
    }


def _failures(
    source_rows: list[dict[str, Any]],
    microtest_rows: list[dict[str, Any]],
    finite_update: dict[str, Any],
    functional_churn: dict[str, Any],
    support_selection: dict[str, Any],
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
    for row in microtest_rows:
        if row.get("status") != "ok":
            failures.append(
                {
                    "source": f"retention_microtest_seed{row.get('seed')}",
                    "field": "status",
                    "expected": "ok",
                    "actual": row.get("status"),
                }
            )
        if not row.get("required_variants_present"):
            failures.append(
                {
                    "source": f"retention_microtest_seed{row.get('seed')}",
                    "field": "required_variants",
                    "expected": list(_REQUIRED_VARIANTS),
                    "actual": "missing",
                }
            )
        for field in (
            "topk2_transfer_ce_improvement",
            "random_fixed_topk2_transfer_ce_improvement",
            "dense_transfer_ce_improvement",
            "topk2_anchor_support_churn_after_transfer",
            "topk1_anchor_support_churn_after_transfer",
            "topk2_commutator_anchor_logit_mse",
            "topk1_commutator_anchor_logit_mse",
        ):
            if row.get(field) is None:
                failures.append(
                    {
                        "source": f"retention_microtest_seed{row.get('seed')}",
                        "field": field,
                        "expected": "numeric",
                        "actual": None,
                    }
                )
    if len(microtest_rows) < 2:
        failures.append(
            {
                "source": "retention_microtests",
                "field": "packet_count",
                "expected": "at least 2",
                "actual": len(microtest_rows),
            }
        )
    expected_decisions = (
        (
            "finite_update_order_control",
            finite_update.get("decision"),
            FINITE_UPDATE_ORDER_SENSITIVITY_CE_BOUNDED,
        ),
        (
            "functional_churn_control",
            functional_churn.get("decision"),
            FUNCTIONAL_CHURN_BOUNDED_WITH_COMMUTATOR_RISK,
        ),
        (
            "support_selection_quality",
            support_selection.get("decision"),
            PROMOTED_TOPK2_SUPPORT_SELECTION_QUALITY_ESTABLISHED,
        ),
    )
    for source, actual, expected in expected_decisions:
        if actual != expected:
            failures.append(
                {
                    "source": source,
                    "field": "decision",
                    "expected": expected,
                    "actual": actual,
                }
            )
    return failures


def _source_row(source: str, path: Path, value: dict[str, Any] | None = None) -> dict[str, Any]:
    loaded = value if value is not None else _read_json_object(path)
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
        "# Promoted Top-k-2 Retention Synthesis Gate",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        "- Mean promoted top-k-2 transfer CE improvement: "
        f"`{metrics['mean_topk2_transfer_ce_improvement']}`",
        "- Mean random-fixed top-k-2 transfer CE improvement: "
        f"`{metrics['mean_random_fixed_topk2_transfer_ce_improvement']}`",
        "- Mean dense transfer CE improvement: "
        f"`{metrics['mean_dense_transfer_ce_improvement']}`",
        "- Mean promoted top-k-2 support churn after transfer: "
        f"`{metrics['mean_topk2_support_churn_after_transfer']}`",
        "- Mean rank-matched top-k-1 support churn after transfer: "
        f"`{metrics['mean_topk1_support_churn_after_transfer']}`",
        "- Minimum top-k-2/top-k-1 commutator logit-MSE ratio: "
        f"`{metrics['min_topk2_to_topk1_commutator_anchor_logit_mse_ratio']}`",
        "- Oracle support regret: "
        f"`{metrics['oracle_support_regret']}`",
        "",
        summary["rationale"],
        "",
        f"Next step: {summary['next_step']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _variant_prefix(variant: str) -> str:
    return {
        "promoted_contextual_topk2": "topk2",
        "rank_matched_contextual_topk1": "topk1",
        "random_fixed_topk2": "random_fixed_topk2",
        "norm_matched_dense_active_rank": "dense",
    }[variant]


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _float_or_none(value: Any) -> float | None:
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [_float_or_none(row.get(field)) for row in rows]
    numeric = [value for value in values if value is not None]
    return mean(numeric) if numeric else None


def _mean_delta_fields(rows: list[dict[str, Any]], left: str, right: str) -> float | None:
    values = [_delta(row.get(left), row.get(right)) for row in rows]
    numeric = [value for value in values if value is not None]
    return mean(numeric) if numeric else None


def _min_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [_float_or_none(row.get(field)) for row in rows]
    numeric = [value for value in values if value is not None]
    return min(numeric) if numeric else None


def _delta(left: Any, right: Any) -> float | None:
    left_float = _float_or_none(left)
    right_float = _float_or_none(right)
    if left_float is None or right_float is None:
        return None
    return left_float - right_float


def _ratio(numerator: Any, denominator: Any) -> float | None:
    numerator_float = _float_or_none(numerator)
    denominator_float = _float_or_none(denominator)
    if numerator_float is None or denominator_float in (None, 0.0):
        return None
    return numerator_float / denominator_float


def _positive(value: Any) -> bool:
    numeric = _float_or_none(value)
    return numeric is not None and numeric > 0.0


def _at_least(value: Any, threshold: float) -> bool:
    numeric = _float_or_none(value)
    return numeric is not None and numeric >= threshold


def _at_most(value: Any, threshold: float) -> bool:
    numeric = _float_or_none(value)
    return numeric is not None and numeric <= threshold


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthesize promoted top-k-2 retention evidence and select one next action."
    )
    parser.add_argument(
        "--microtest-dir",
        action="append",
        type=Path,
        dest="microtest_dirs",
        help="Retention microtest directory. Repeat for multiple seeds.",
    )
    parser.add_argument("--finite-update-dir", type=Path, default=DEFAULT_FINITE_UPDATE_DIR)
    parser.add_argument("--functional-churn-dir", type=Path, default=DEFAULT_FUNCTIONAL_CHURN_DIR)
    parser.add_argument("--support-selection-dir", type=Path, default=DEFAULT_SUPPORT_SELECTION_DIR)
    parser.add_argument("--deconfounded-dir", type=Path, default=DEFAULT_DECONFOUNDED_DIR)
    parser.add_argument("--context-gate-dir", type=Path, default=DEFAULT_CONTEXT_GATE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_promoted_topk2_retention_synthesis_gate(
        microtest_dirs=tuple(args.microtest_dirs or DEFAULT_MICROTEST_DIRS),
        finite_update_dir=args.finite_update_dir,
        functional_churn_dir=args.functional_churn_dir,
        support_selection_dir=args.support_selection_dir,
        deconfounded_dir=args.deconfounded_dir,
        context_gate_dir=args.context_gate_dir,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
