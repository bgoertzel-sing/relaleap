"""Causal-adequacy matrix for promoted contextual top-k-2 routing."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_RETENTION_SYNTHESIS_DIR = Path(
    "results/reports/token_larger_promoted_topk2_retention_synthesis_gate"
)
DEFAULT_FINITE_UPDATE_MATRIX_DIR = Path(
    "results/reports/token_larger_promoted_topk2_finite_update_control_matrix"
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
DEFAULT_TOPK1_CAUSAL_RETENTION_DIR = Path(
    "results/reports/token_larger_active_topk1_causal_retention_synthesis"
)
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_promoted_topk2_causal_adequacy_matrix"
)

PREDICTIVE_DEFAULT_CAUSAL_ADEQUACY_NOT_ESTABLISHED = (
    "predictive_default_causal_adequacy_not_established"
)
PROMOTED_TOPK2_CAUSAL_ADEQUACY_SUPPORTED = (
    "promoted_topk2_causal_adequacy_supported"
)
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

REQUIRED_SOURCES = {
    "retention_synthesis": (
        DEFAULT_RETENTION_SYNTHESIS_DIR,
        "summary.json",
        "pass",
        "contextual_topk2_router_default_topk1_diagnostic",
    ),
    "finite_update_control_matrix": (
        DEFAULT_FINITE_UPDATE_MATRIX_DIR,
        "summary.json",
        "pass",
        "finite_update_control_matrix_ready",
    ),
    "functional_churn_control": (
        DEFAULT_FUNCTIONAL_CHURN_DIR,
        "summary.json",
        "pass",
        "support_identity_churn_functional_impact_bounded_with_commutator_risk",
    ),
    "support_selection_quality": (
        DEFAULT_SUPPORT_SELECTION_DIR,
        "summary.json",
        "pass",
        "promoted_topk2_support_selection_quality_established",
    ),
    "deconfounded_intervention": (
        DEFAULT_DECONFOUNDED_DIR,
        "summary.json",
        "pass",
        "topk2_comparative_causal_cooperation_not_supported",
    ),
    "active_topk1_causal_retention": (
        DEFAULT_TOPK1_CAUSAL_RETENTION_DIR,
        "summary.json",
        "pass",
        "causal_retention_claim_blocked_by_deployable_gate",
    ),
}


def run_promoted_topk2_causal_adequacy_matrix(
    *,
    retention_synthesis_dir: Path = DEFAULT_RETENTION_SYNTHESIS_DIR,
    finite_update_matrix_dir: Path = DEFAULT_FINITE_UPDATE_MATRIX_DIR,
    functional_churn_dir: Path = DEFAULT_FUNCTIONAL_CHURN_DIR,
    support_selection_dir: Path = DEFAULT_SUPPORT_SELECTION_DIR,
    deconfounded_dir: Path = DEFAULT_DECONFOUNDED_DIR,
    topk1_causal_retention_dir: Path = DEFAULT_TOPK1_CAUSAL_RETENTION_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    ce_guardrail_tolerance: float = 0.05,
    low_support_churn_threshold: float = 0.05,
    high_support_churn_threshold: float = 0.50,
    topk2_required_cleaner_strata_fraction: float = 0.80,
) -> dict[str, Any]:
    """Assemble the matched-control causal-adequacy gate without retraining."""

    start = time.time()
    source_dirs = {
        "retention_synthesis": retention_synthesis_dir,
        "finite_update_control_matrix": finite_update_matrix_dir,
        "functional_churn_control": functional_churn_dir,
        "support_selection_quality": support_selection_dir,
        "deconfounded_intervention": deconfounded_dir,
        "active_topk1_causal_retention": topk1_causal_retention_dir,
    }
    packets = {
        name: _read_json_object(path / "summary.json")
        for name, path in source_dirs.items()
    }
    source_rows = _source_rows(source_dirs, packets)
    matrix_rows = _matrix_rows(packets)
    metrics = _metrics(packets, matrix_rows)
    signals = _signals(
        metrics,
        packets,
        ce_guardrail_tolerance=ce_guardrail_tolerance,
        low_support_churn_threshold=low_support_churn_threshold,
        high_support_churn_threshold=high_support_churn_threshold,
        topk2_required_cleaner_strata_fraction=topk2_required_cleaner_strata_fraction,
    )
    failures = _failures(source_rows, matrix_rows, metrics)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        interpretation = (
            "The causal-adequacy matrix cannot be interpreted because one or "
            "more required command-generated source packets is missing, failing, "
            "or lacks the matched-control metrics needed for the gate."
        )
        next_step = "repair_missing_promoted_topk2_causal_adequacy_sources"
    elif signals["promoted_topk2_causal_adequacy_supported"]:
        status = "pass"
        decision = PROMOTED_TOPK2_CAUSAL_ADEQUACY_SUPPORTED
        interpretation = (
            "Promoted contextual top-k-2 passes the matched-control causal "
            "adequacy gate: CE is within the guardrail, oracle regret is low, "
            "retention/functional churn and finite-update controls are clean, "
            "and deconfounded intervention strata support reusable pair gains."
        )
        next_step = "run one backend-stable repeat before broadening the causal claim"
    else:
        status = "pass"
        decision = PREDICTIVE_DEFAULT_CAUSAL_ADEQUACY_NOT_ESTABLISHED
        interpretation = (
            "Promoted contextual top-k-2 remains the predictive support-routing "
            "default because it beats random fixed top-k-2 and dense active-rank "
            "controls on transfer and has small oracle-support regret. It does "
            "not pass the stronger causal-adequacy gate: support churn and "
            "finite-update logit/residual risk are high versus rank-matched "
            "top-k-1, and deconfounded intervention strata do not clear the "
            "pre-registered causal-cooperation threshold. Rank-matched top-k-1 "
            "therefore stays a retention/churn control, not the default router."
        )
        next_step = (
            "run the already-selected local no-training finite-update "
            "order-symmetrization audit for promoted contextual top-k-2"
        )

    summary = {
        "status": status,
        "decision": decision,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "thresholds": {
            "ce_guardrail_tolerance": ce_guardrail_tolerance,
            "low_support_churn_threshold": low_support_churn_threshold,
            "high_support_churn_threshold": high_support_churn_threshold,
            "topk2_required_cleaner_strata_fraction": topk2_required_cleaner_strata_fraction,
        },
        "source_rows": source_rows,
        "matrix_rows": matrix_rows,
        "metrics": metrics,
        "signals": signals,
        "failures": failures,
        "interpretation": interpretation,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "causal_adequacy_matrix_csv": str(out_dir / "causal_adequacy_matrix.csv"),
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
    _write_csv(out_dir / "causal_adequacy_matrix.csv", matrix_rows)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _source_rows(
    source_dirs: dict[str, Path], packets: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for source, path in source_dirs.items():
        expected = REQUIRED_SOURCES[source]
        packet = packets[source]
        summary_path = path / "summary.json"
        rows.append(
            {
                "source": source,
                "path": str(summary_path),
                "present": summary_path.is_file(),
                "status": packet.get("status"),
                "expected_status": expected[2],
                "decision": packet.get("decision"),
                "expected_decision": expected[3],
            }
        )
    return rows


def _matrix_rows(packets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    retention = packets["retention_synthesis"]
    finite = packets["finite_update_control_matrix"]
    deconfounded = packets["deconfounded_intervention"].get("evidence", {})
    support = packets["support_selection_quality"].get("metrics", {})
    active_topk1 = packets["active_topk1_causal_retention"].get("evidence", {}).get(
        "metrics", {}
    )
    retention_metrics = retention.get("metrics", {})
    finite_metrics = finite.get("metrics", {})
    rows = [
        {
            "variant": "promoted_contextual_topk2",
            "role": "promoted_default",
            "mean_transfer_ce_improvement": retention_metrics.get(
                "mean_topk2_transfer_ce_improvement"
            ),
            "support_churn": retention_metrics.get(
                "mean_topk2_support_churn_after_transfer"
            ),
            "finite_update_logit_mse": finite_metrics.get("topk2_mean_logit_mse"),
            "finite_update_ce_abs_delta": finite_metrics.get(
                "topk2_mean_ce_abs_delta"
            ),
            "finite_update_residual_delta_l2": finite_metrics.get(
                "topk2_mean_residual_delta_l2"
            ),
            "oracle_support_regret": support.get("oracle_support_regret"),
            "oracle_support_regret_positive_fraction": support.get(
                "oracle_support_regret_positive_fraction"
            ),
            "intervention_cleaner_strata_fraction": deconfounded.get(
                "topk2_fixed_support_cleaner_strata_fraction"
            ),
            "functional_churn_cleaner_strata_fraction": deconfounded.get(
                "topk2_functional_churn_cleaner_strata_fraction"
            ),
            "incremental_pair_gain_positive_strata_fraction": deconfounded.get(
                "topk2_incremental_pair_gain_positive_strata_fraction"
            ),
            "ce_loss": deconfounded.get("topk2_alpha0_ce_loss"),
            "ce_deficit_vs_topk1": deconfounded.get("topk2_ce_deficit_vs_topk1"),
            "claim": "predictive_default_candidate",
        },
        {
            "variant": "rank_matched_contextual_topk1",
            "role": "retention_churn_control",
            "mean_transfer_ce_improvement": retention_metrics.get(
                "mean_topk1_transfer_ce_improvement"
            ),
            "support_churn": retention_metrics.get(
                "mean_topk1_support_churn_after_transfer"
            ),
            "finite_update_logit_mse": finite_metrics.get("topk1_mean_logit_mse"),
            "finite_update_ce_abs_delta": finite_metrics.get(
                "topk1_mean_ce_abs_delta"
            ),
            "finite_update_residual_delta_l2": finite_metrics.get(
                "topk1_mean_residual_delta_l2"
            ),
            "ce_loss": deconfounded.get("topk1_alpha0_ce_loss"),
            "selected_singleton_gain_mean": active_topk1.get(
                "selected_singleton_gain_mean"
            ),
            "deployable_gain_minus_ungated": active_topk1.get(
                "deployable_gain_minus_ungated"
            ),
            "deployable_offcontext_harm_suppression_fraction": active_topk1.get(
                "deployable_offcontext_harm_suppression_fraction"
            ),
            "claim": "local_retention_control_only",
        },
        {
            "variant": "random_fixed_topk2",
            "role": "random_topk2_control",
            "mean_transfer_ce_improvement": retention_metrics.get(
                "mean_random_fixed_topk2_transfer_ce_improvement"
            ),
            "finite_update_logit_mse": finite_metrics.get(
                "random_fixed_topk2_mean_logit_mse"
            ),
            "finite_update_ce_abs_delta": finite_metrics.get(
                "random_fixed_topk2_mean_ce_abs_delta"
            ),
            "claim": "negative_support_selection_control",
        },
        {
            "variant": "norm_matched_dense_active_rank",
            "role": "dense_active_rank_control",
            "mean_transfer_ce_improvement": retention_metrics.get(
                "mean_dense_transfer_ce_improvement"
            ),
            "finite_update_logit_mse": finite_metrics.get(
                "dense_active_rank_mean_logit_mse"
            ),
            "finite_update_ce_abs_delta": finite_metrics.get(
                "dense_active_rank_mean_ce_abs_delta"
            ),
            "claim": "rank_matched_dense_control",
        },
    ]
    return rows


def _metrics(
    packets: dict[str, dict[str, Any]], matrix_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_variant = {row["variant"]: row for row in matrix_rows}
    topk2 = by_variant.get("promoted_contextual_topk2", {})
    topk1 = by_variant.get("rank_matched_contextual_topk1", {})
    random_topk2 = by_variant.get("random_fixed_topk2", {})
    dense = by_variant.get("norm_matched_dense_active_rank", {})
    deconfounded = packets["deconfounded_intervention"].get("evidence", {})
    active_topk1 = packets["active_topk1_causal_retention"].get("evidence", {}).get(
        "metrics", {}
    )
    return {
        "matrix_variant_count": len(matrix_rows),
        "topk2_transfer_advantage_vs_random_fixed_topk2": _delta(
            topk2.get("mean_transfer_ce_improvement"),
            random_topk2.get("mean_transfer_ce_improvement"),
        ),
        "topk2_transfer_advantage_vs_dense": _delta(
            topk2.get("mean_transfer_ce_improvement"),
            dense.get("mean_transfer_ce_improvement"),
        ),
        "topk2_transfer_delta_vs_topk1": _delta(
            topk2.get("mean_transfer_ce_improvement"),
            topk1.get("mean_transfer_ce_improvement"),
        ),
        "topk2_support_churn": topk2.get("support_churn"),
        "topk1_support_churn": topk1.get("support_churn"),
        "topk2_minus_topk1_support_churn": _delta(
            topk2.get("support_churn"), topk1.get("support_churn")
        ),
        "topk2_finite_update_logit_mse": topk2.get("finite_update_logit_mse"),
        "topk1_finite_update_logit_mse": topk1.get("finite_update_logit_mse"),
        "topk2_to_topk1_finite_update_logit_mse_ratio": _ratio(
            topk2.get("finite_update_logit_mse"),
            topk1.get("finite_update_logit_mse"),
        ),
        "topk2_to_dense_finite_update_logit_mse_ratio": _ratio(
            topk2.get("finite_update_logit_mse"),
            dense.get("finite_update_logit_mse"),
        ),
        "topk2_ce_deficit_vs_topk1": deconfounded.get("topk2_ce_deficit_vs_topk1"),
        "topk2_fixed_support_cleaner_strata_fraction": deconfounded.get(
            "topk2_fixed_support_cleaner_strata_fraction"
        ),
        "topk2_functional_churn_cleaner_strata_fraction": deconfounded.get(
            "topk2_functional_churn_cleaner_strata_fraction"
        ),
        "topk2_incremental_pair_gain_positive_strata_fraction": deconfounded.get(
            "topk2_incremental_pair_gain_positive_strata_fraction"
        ),
        "oracle_support_regret": topk2.get("oracle_support_regret"),
        "oracle_support_regret_positive_fraction": topk2.get(
            "oracle_support_regret_positive_fraction"
        ),
        "topk1_deployable_gain_minus_ungated": active_topk1.get(
            "deployable_gain_minus_ungated"
        ),
        "topk1_deployable_offcontext_harm_suppression_fraction": active_topk1.get(
            "deployable_offcontext_harm_suppression_fraction"
        ),
    }


def _signals(
    metrics: dict[str, Any],
    packets: dict[str, dict[str, Any]],
    *,
    ce_guardrail_tolerance: float,
    low_support_churn_threshold: float,
    high_support_churn_threshold: float,
    topk2_required_cleaner_strata_fraction: float,
) -> dict[str, bool]:
    ce_guardrail = _at_most(metrics.get("topk2_ce_deficit_vs_topk1"), ce_guardrail_tolerance)
    oracle_regret_low = _at_most(metrics.get("oracle_support_regret"), 0.01)
    predictive_control_win = _positive(
        metrics.get("topk2_transfer_advantage_vs_random_fixed_topk2")
    ) and _positive(metrics.get("topk2_transfer_advantage_vs_dense"))
    topk1_cleaner_retention = _at_most(
        metrics.get("topk1_support_churn"), low_support_churn_threshold
    ) and _at_least(metrics.get("topk2_support_churn"), high_support_churn_threshold)
    finite_update_risk_high = _at_least(
        metrics.get("topk2_to_topk1_finite_update_logit_mse_ratio"), 5.0
    )
    intervention_gate = (
        _at_least(
            metrics.get("topk2_fixed_support_cleaner_strata_fraction"),
            topk2_required_cleaner_strata_fraction,
        )
        and _at_least(
            metrics.get("topk2_functional_churn_cleaner_strata_fraction"),
            topk2_required_cleaner_strata_fraction,
        )
        and _at_least(
            metrics.get("topk2_incremental_pair_gain_positive_strata_fraction"),
            topk2_required_cleaner_strata_fraction,
        )
    )
    topk1_deployable_gate_failed = (
        packets["active_topk1_causal_retention"].get("decision")
        == "causal_retention_claim_blocked_by_deployable_gate"
    )
    return {
        "ce_guardrail_passed": ce_guardrail,
        "oracle_support_regret_low": oracle_regret_low,
        "topk2_predictive_control_win": predictive_control_win,
        "topk1_cleaner_retention_control": topk1_cleaner_retention,
        "finite_update_risk_high_vs_topk1": finite_update_risk_high,
        "intervention_cleanliness_gate_passed": intervention_gate,
        "topk1_deployable_gate_failed": topk1_deployable_gate_failed,
        "promoted_topk2_causal_adequacy_supported": (
            ce_guardrail
            and oracle_regret_low
            and predictive_control_win
            and not topk1_cleaner_retention
            and not finite_update_risk_high
            and intervention_gate
        ),
    }


def _failures(
    source_rows: list[dict[str, Any]],
    matrix_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows:
        if not row["present"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "summary_json",
                    "expected": "present",
                    "actual": "missing",
                    "path": row["path"],
                }
            )
            continue
        if row["status"] != row["expected_status"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "status",
                    "expected": row["expected_status"],
                    "actual": row["status"],
                }
            )
        if row["decision"] != row["expected_decision"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "decision",
                    "expected": row["expected_decision"],
                    "actual": row["decision"],
                }
            )
    expected_variants = {
        "promoted_contextual_topk2",
        "rank_matched_contextual_topk1",
        "random_fixed_topk2",
        "norm_matched_dense_active_rank",
    }
    actual_variants = {str(row.get("variant")) for row in matrix_rows}
    for variant in sorted(expected_variants - actual_variants):
        failures.append(
            {
                "source": "causal_adequacy_matrix",
                "field": "variant",
                "expected": variant,
                "actual": "missing",
            }
        )
    for field in (
        "topk2_transfer_advantage_vs_random_fixed_topk2",
        "topk2_transfer_advantage_vs_dense",
        "topk2_support_churn",
        "topk1_support_churn",
        "topk2_to_topk1_finite_update_logit_mse_ratio",
        "topk2_ce_deficit_vs_topk1",
        "topk2_fixed_support_cleaner_strata_fraction",
        "topk2_functional_churn_cleaner_strata_fraction",
        "topk2_incremental_pair_gain_positive_strata_fraction",
        "oracle_support_regret",
    ):
        if metrics.get(field) is None:
            failures.append(
                {
                    "source": "causal_adequacy_metrics",
                    "field": field,
                    "expected": "numeric",
                    "actual": None,
                }
            )
    return failures


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    lines = [
        "# Promoted Top-k-2 Causal Adequacy Matrix",
        "",
        f"Status: `{summary['status']}`",
        f"Decision: `{summary['decision']}`",
        "",
        "This is a no-training synthesis over command-generated artifacts. It "
        "compares promoted contextual top-k-2 against rank-matched contextual "
        "top-k-1, random fixed top-k-2, and dense active-rank controls.",
        "",
        "- Top-k-2 transfer advantage vs random fixed top-k-2: "
        f"`{metrics['topk2_transfer_advantage_vs_random_fixed_topk2']}`",
        "- Top-k-2 transfer advantage vs dense active-rank: "
        f"`{metrics['topk2_transfer_advantage_vs_dense']}`",
        "- Top-k-2 CE deficit vs top-k-1: "
        f"`{metrics['topk2_ce_deficit_vs_topk1']}`",
        "- Top-k-2 minus top-k-1 support churn: "
        f"`{metrics['topk2_minus_topk1_support_churn']}`",
        "- Top-k-2/top-k-1 finite-update logit-MSE ratio: "
        f"`{metrics['topk2_to_topk1_finite_update_logit_mse_ratio']}`",
        "- Oracle support regret: "
        f"`{metrics['oracle_support_regret']}`",
        "",
        summary["interpretation"],
        "",
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _at_most(value: Any, threshold: float) -> bool:
    numeric = _float_or_none(value)
    return numeric is not None and numeric <= threshold


def _at_least(value: Any, threshold: float) -> bool:
    numeric = _float_or_none(value)
    return numeric is not None and numeric >= threshold


def _positive(value: Any) -> bool:
    numeric = _float_or_none(value)
    return numeric is not None and numeric > 0.0


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention-synthesis-dir", type=Path, default=DEFAULT_RETENTION_SYNTHESIS_DIR)
    parser.add_argument("--finite-update-matrix-dir", type=Path, default=DEFAULT_FINITE_UPDATE_MATRIX_DIR)
    parser.add_argument("--functional-churn-dir", type=Path, default=DEFAULT_FUNCTIONAL_CHURN_DIR)
    parser.add_argument("--support-selection-dir", type=Path, default=DEFAULT_SUPPORT_SELECTION_DIR)
    parser.add_argument("--deconfounded-dir", type=Path, default=DEFAULT_DECONFOUNDED_DIR)
    parser.add_argument("--topk1-causal-retention-dir", type=Path, default=DEFAULT_TOPK1_CAUSAL_RETENTION_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_promoted_topk2_causal_adequacy_matrix(
        retention_synthesis_dir=args.retention_synthesis_dir,
        finite_update_matrix_dir=args.finite_update_matrix_dir,
        functional_churn_dir=args.functional_churn_dir,
        support_selection_dir=args.support_selection_dir,
        deconfounded_dir=args.deconfounded_dir,
        topk1_causal_retention_dir=args.topk1_causal_retention_dir,
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
