"""Fail-closed explicit order-averaging mitigation probe for promoted top-k-2."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_BRANCH_SELECTOR = Path(
    "results/reports/token_larger_promoted_topk2_mitigation_branch_selector/summary.json"
)
DEFAULT_FINITE_UPDATE_REPORT = Path(
    "results/reports/token_larger_promoted_topk2_finite_update_order_control_audit/summary.json"
)
DEFAULT_CONTROL_MATRIX = Path(
    "results/reports/token_larger_promoted_topk2_finite_update_control_matrix/summary.json"
)
DEFAULT_FLAT_VALUE_REPORT = Path(
    "results/reports/same_router_flat_value_commutator_mitigation_probe/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_promoted_topk2_explicit_order_averaging_mitigation_probe"
)

ORDER_AVERAGING_DIAGNOSTIC_CANDIDATE = (
    "explicit_order_averaging_diagnostic_candidate_not_promoted"
)
ORDER_AVERAGING_NOT_ESTABLISHED = "explicit_order_averaging_not_established"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

NEXT_ROUTER_POLICY_COMMAND = (
    "python -m relaleap.experiments.promoted_topk2_router_policy_mitigation_probe "
    "--config configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml "
    "--out results/audits/token_larger_promoted_topk2_router_policy_mitigation_probe"
)


def run_promoted_topk2_explicit_order_averaging_mitigation_probe(
    *,
    branch_selector_path: Path = DEFAULT_BRANCH_SELECTOR,
    finite_update_report_path: Path = DEFAULT_FINITE_UPDATE_REPORT,
    control_matrix_path: Path = DEFAULT_CONTROL_MATRIX,
    flat_value_report_path: Path = DEFAULT_FLAT_VALUE_REPORT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
    order_averaging_ratio_gate: float = 0.5,
    order_averaging_ce_delta_gate: float = 0.0,
) -> dict[str, Any]:
    """Gate explicit order averaging as a diagnostic, not a promoted path."""

    start = time.time()
    branch_selector = _read_json_object(branch_selector_path)
    finite_update = _read_json_object(finite_update_report_path)
    control_matrix = _read_json_object(control_matrix_path)
    flat_value_report = _read_json_object(flat_value_report_path)
    strategy_review = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("mitigation_branch_selector", branch_selector_path, branch_selector),
        _source_row("finite_update_order_control", finite_update_report_path, finite_update),
        _source_row("finite_update_control_matrix", control_matrix_path, control_matrix),
        _source_row("flat_value_commutator_mitigation", flat_value_report_path, flat_value_report),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy_review["present"],
            "status": "present" if strategy_review["present"] else "missing_optional",
            "decision": strategy_review["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy_review['strategic_change_level']}; "
                f"notify_ben={strategy_review['notify_ben']}"
            ),
        },
    ]
    evidence = _evidence_snapshot(branch_selector, finite_update)
    thresholds = {
        "order_averaging_ratio_gate": order_averaging_ratio_gate,
        "order_averaging_ce_delta_gate": order_averaging_ce_delta_gate,
    }
    control_rows = _control_rows(control_matrix, flat_value_report, evidence)
    gate_rows = _gate_rows(control_rows, evidence, thresholds)
    failures = _failures(source_rows, evidence, control_matrix)
    order_rows = [_order_averaging_row(evidence, thresholds)]

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        selected_next_action = None
        next_command = None
        next_step = "repair missing or inconsistent order-averaging source artifacts"
        rationale = (
            "The explicit order-averaging probe cannot be interpreted because "
            "required source artifacts or numeric finite-update fields are missing."
        )
    elif order_rows[0]["passes_diagnostic_gate"]:
        status = "pass"
        decision = ORDER_AVERAGING_DIAGNOSTIC_CANDIDATE
        selected_next_action = "record_order_averaging_matched_control_closeout_no_gpu"
        next_command = None
        next_step = (
            "record order averaging as a nondeployable diagnostic upper bound; "
            "do not promote top-k-2 causal cooperation or launch GPU until a "
            "deployable sparse mechanism beats dense, flat, random-support, and "
            "no-update controls under matched budgets"
        )
        rationale = (
            "Explicit forward/reverse order averaging reduces the observed top-k-2 "
            "commutator logit-MSE proxy and does not worsen anchor CE versus the "
            "best endpoint in the existing finite-update packet. Per the current "
            "strategic review and matched-control rows, this is recorded only as "
            "a diagnostic upper bound: the operation averages both orders, is not "
            "deployable as a finite update rule, has no measured flat-value "
            "order-averaging counterpart, and does not establish sparse causal "
            "column cooperation beyond generic smoothing/null controls."
        )
    else:
        status = "pass"
        decision = ORDER_AVERAGING_NOT_ESTABLISHED
        selected_next_action = "router_policy_mitigation_probe"
        next_command = NEXT_ROUTER_POLICY_COMMAND
        next_step = "close order averaging as unestablished and use non-router local mechanism selectors"
        rationale = (
            "The finite-update packet does not clear the explicit order-averaging "
            "diagnostic gate, so the next mitigation branch should target router "
            "policy directly."
        )

    summary = {
        "status": status,
        "decision": decision,
        "selected_next_action": selected_next_action,
        "next_command": next_command,
        "next_step": next_step,
        "claim_statuses": {
            "contextual_topk2_router": "promoted_operational_default_train_time_support_selection",
            "order_averaging": "diagnostic_upper_bound_not_deployable_not_promoted",
            "topk2_causal_cooperation": "not_supported",
            "topk2_finite_update_interference": "mitigated_only_by_non_deployable_order_average",
            "topk1_singleton_reuse": "diagnostic_only_not_deployable",
        },
        "source_rows": source_rows,
        "evidence": evidence,
        "thresholds": thresholds,
        "order_averaging_rows": order_rows,
        "control_rows": control_rows,
        "gate_rows": gate_rows,
        "strategy_review": strategy_review,
        "failures": failures,
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "rationale": rationale,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "order_averaging_rows_csv": str(out_dir / "order_averaging_rows.csv"),
            "control_rows_csv": str(out_dir / "control_rows.csv"),
            "gate_rows_csv": str(out_dir / "gate_rows.csv"),
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
        ["source", "path", "present", "status", "decision", "claim_status"],
        source_rows,
    )
    _write_csv(out_dir / "order_averaging_rows.csv", order_rows)
    _write_csv(out_dir / "control_rows.csv", control_rows)
    _write_csv(out_dir / "gate_rows.csv", gate_rows)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _evidence_snapshot(
    branch_selector: dict[str, Any],
    finite_update: dict[str, Any],
) -> dict[str, Any]:
    metrics = finite_update.get("metrics", {})
    signals = finite_update.get("signals", {})
    return {
        "branch_selector_decision": branch_selector.get("decision"),
        "branch_selector_selected_next_action": branch_selector.get("selected_next_action"),
        "finite_update_decision": finite_update.get("decision"),
        "order_averaged_rows_available": bool(signals.get("order_averaged_rows_available")),
        "topk2_mean_commutator_anchor_logit_mse": _float_or_none(
            metrics.get("topk2_mean_commutator_anchor_logit_mse")
        ),
        "topk2_mean_order_averaged_anchor_logit_mse_to_forward": _float_or_none(
            metrics.get("topk2_mean_order_averaged_anchor_logit_mse_to_forward")
        ),
        "topk2_order_averaged_to_commutator_anchor_logit_mse_ratio": _float_or_none(
            metrics.get("topk2_order_averaged_to_commutator_anchor_logit_mse_ratio")
        ),
        "topk2_mean_order_averaged_anchor_ce_delta_vs_best_order": _float_or_none(
            metrics.get("topk2_mean_order_averaged_anchor_ce_delta_vs_best_order")
        ),
        "topk2_mean_order_averaged_anchor_ce_delta_vs_forward": _float_or_none(
            metrics.get("topk2_mean_order_averaged_anchor_ce_delta_vs_forward")
        ),
        "topk2_mean_same_order_ensemble_anchor_ce_delta_vs_best_endpoint": _float_or_none(
            metrics.get("topk2_mean_same_order_ensemble_anchor_ce_delta_vs_best_endpoint")
        ),
        "topk2_same_order_identical_anchor_logit_mse_to_commutator_ratio": _float_or_none(
            metrics.get("topk2_same_order_identical_anchor_logit_mse_to_commutator_ratio")
        ),
    }


def _failures(
    source_rows: list[dict[str, Any]],
    evidence: dict[str, Any],
    control_matrix: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows[:3]:
        if not row["present"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "source_artifact",
                    "expected": "file exists",
                    "actual": "missing",
                    "path": row["path"],
                }
            )
    expected = {
        "branch_selector_decision": "promoted_topk2_mitigation_branch_selected",
        "branch_selector_selected_next_action": "explicit_order_averaging_mitigation_probe",
        "finite_update_decision": "finite_update_order_sensitivity_ce_bounded_but_residual_material",
    }
    for field, expected_value in expected.items():
        if evidence.get(field) != expected_value:
            failures.append(
                {
                    "source": "evidence_snapshot",
                    "field": field,
                    "expected": expected_value,
                    "actual": evidence.get(field),
                }
            )
    for field in (
        "topk2_mean_commutator_anchor_logit_mse",
        "topk2_mean_order_averaged_anchor_logit_mse_to_forward",
        "topk2_order_averaged_to_commutator_anchor_logit_mse_ratio",
        "topk2_mean_order_averaged_anchor_ce_delta_vs_best_order",
    ):
        if evidence.get(field) is None:
            failures.append(
                {
                    "source": "evidence_snapshot",
                    "field": field,
                    "expected": "numeric value",
                    "actual": None,
                }
            )
    matrix_rows = control_matrix.get("matrix_rows")
    matrix_variants = {
        row.get("variant")
        for row in matrix_rows
        if isinstance(row, dict) and row.get("split") == "all"
    } if isinstance(matrix_rows, list) else set()
    for variant in (
        "promoted_contextual_topk2",
        "norm_matched_dense_active_rank",
        "random_fixed_topk2",
    ):
        if variant not in matrix_variants:
            failures.append(
                {
                    "source": "finite_update_control_matrix",
                    "field": "matrix_rows",
                    "expected": f"all-split row for {variant}",
                    "actual": sorted(matrix_variants),
                }
            )
    return failures


def _order_averaging_row(
    evidence: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    ratio = evidence.get("topk2_order_averaged_to_commutator_anchor_logit_mse_ratio")
    ce_delta = evidence.get("topk2_mean_order_averaged_anchor_ce_delta_vs_best_order")
    passes = (
        ratio is not None
        and ce_delta is not None
        and ratio <= thresholds["order_averaging_ratio_gate"]
        and ce_delta <= thresholds["order_averaging_ce_delta_gate"]
    )
    return {
        "variant": "explicit_forward_reverse_order_average",
        "role": "sparse_order_average_diagnostic",
        "deployability": "nondeployable_diagnostic_uses_both_update_orders",
        "commutator_anchor_logit_mse": evidence.get(
            "topk2_mean_commutator_anchor_logit_mse"
        ),
        "order_averaged_anchor_logit_mse_to_forward": evidence.get(
            "topk2_mean_order_averaged_anchor_logit_mse_to_forward"
        ),
        "order_averaged_to_commutator_anchor_logit_mse_ratio": ratio,
        "order_averaged_anchor_ce_delta_vs_best_order": ce_delta,
        "order_averaged_anchor_ce_delta_vs_forward": evidence.get(
            "topk2_mean_order_averaged_anchor_ce_delta_vs_forward"
        ),
        "same_order_ensemble_anchor_ce_delta_vs_best_endpoint": evidence.get(
            "topk2_mean_same_order_ensemble_anchor_ce_delta_vs_best_endpoint"
        ),
        "same_order_identical_anchor_logit_mse_to_commutator_ratio": evidence.get(
            "topk2_same_order_identical_anchor_logit_mse_to_commutator_ratio"
        ),
        "passes_diagnostic_gate": passes,
        "claim_status": "diagnostic_upper_bound_not_promoted",
    }


def _control_rows(
    control_matrix: dict[str, Any],
    flat_value_report: dict[str, Any],
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matrix_rows = control_matrix.get("matrix_rows", [])
    by_variant = {
        row.get("variant"): row
        for row in matrix_rows
        if isinstance(row, dict) and row.get("split") == "all"
    }
    for role, variant, claim in (
        (
            "ordinary_sparse_topk2",
            "promoted_contextual_topk2",
            "active sparse baseline has high order sensitivity and support churn",
        ),
        (
            "dense_control",
            "norm_matched_dense_active_rank",
            "dense active-rank control is required for generic smoothing comparisons",
        ),
        (
            "random_support_control",
            "random_fixed_topk2",
            "random-support sparse null is required for support-policy comparisons",
        ),
        (
            "rank_matched_topk1_control",
            "rank_matched_contextual_topk1",
            "top-k1 retention bracket remains diagnostic only",
        ),
    ):
        source = by_variant.get(variant, {})
        rows.append(
            {
                "role": role,
                "variant": variant,
                "source": "finite_update_control_matrix",
                "measured": bool(source),
                "mean_logit_mse": _float_or_none(source.get("mean_logit_mse")),
                "mean_ce_abs_delta": _float_or_none(source.get("mean_ce_abs_delta")),
                "mean_residual_delta_l2": _float_or_none(source.get("mean_residual_delta_l2")),
                "mean_symmetric_kl": _float_or_none(source.get("mean_symmetric_kl")),
                "support_churn_fraction": _float_or_none(source.get("support_churn_fraction")),
                "row_count": source.get("row_count"),
                "claim_status": claim,
            }
        )
    order_avg_mse = evidence.get("topk2_mean_order_averaged_anchor_logit_mse_to_forward")
    dense_mse = rows[1]["mean_logit_mse"]
    random_mse = rows[2]["mean_logit_mse"]
    rows.append(
        {
            "role": "sparse_order_average_diagnostic",
            "variant": "explicit_forward_reverse_order_average",
            "source": "finite_update_order_control",
            "measured": order_avg_mse is not None,
            "mean_logit_mse": order_avg_mse,
            "mean_ce_abs_delta": evidence.get("topk2_mean_order_averaged_anchor_ce_delta_vs_best_order"),
            "mean_residual_delta_l2": None,
            "mean_symmetric_kl": None,
            "support_churn_fraction": None,
            "row_count": None,
            "beats_dense_control_on_logit_mse": (
                order_avg_mse is not None and dense_mse is not None and order_avg_mse <= dense_mse
            ),
            "beats_random_support_on_logit_mse": (
                order_avg_mse is not None and random_mse is not None and order_avg_mse <= random_mse
            ),
            "claim_status": "nondeployable diagnostic upper bound, not sparse-mechanism evidence",
        }
    )
    flat_missing_count = flat_value_report.get("missing_required_variant_count")
    rows.append(
        {
            "role": "flat_value_control",
            "variant": "flat_value_order_averaged_updates",
            "source": "same_router_flat_value_commutator_mitigation_probe",
            "measured": False,
            "mean_logit_mse": None,
            "mean_ce_abs_delta": None,
            "mean_residual_delta_l2": None,
            "mean_symmetric_kl": None,
            "support_churn_fraction": None,
            "row_count": None,
            "missing_required_variant_count": flat_missing_count,
            "claim_status": "flat order-averaging control missing, so sparse-specific claim fails closed",
        }
    )
    rows.append(
        {
            "role": "no_update_null",
            "variant": "no_update",
            "source": "defined_null",
            "measured": True,
            "mean_logit_mse": 0.0,
            "mean_ce_abs_delta": 0.0,
            "mean_residual_delta_l2": 0.0,
            "mean_symmetric_kl": 0.0,
            "support_churn_fraction": 0.0,
            "row_count": None,
            "claim_status": "zero commutator but zero learning; guardrail null, not a mitigation win",
        }
    )
    return rows


def _gate_rows(
    control_rows: list[dict[str, Any]],
    evidence: dict[str, Any],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    by_role = {row["role"]: row for row in control_rows}
    order_avg = by_role.get("sparse_order_average_diagnostic", {})
    dense = by_role.get("dense_control", {})
    random = by_role.get("random_support_control", {})
    flat = by_role.get("flat_value_control", {})
    ratio = evidence.get("topk2_order_averaged_to_commutator_anchor_logit_mse_ratio")
    ce_delta = evidence.get("topk2_mean_order_averaged_anchor_ce_delta_vs_best_order")
    order_mse = order_avg.get("mean_logit_mse")
    return [
        {
            "gate": "sparse_order_average_reduces_commutator_with_ce_guardrail",
            "passes": bool(
                ratio is not None
                and ce_delta is not None
                and ratio <= thresholds["order_averaging_ratio_gate"]
                and ce_delta <= thresholds["order_averaging_ce_delta_gate"]
            ),
            "interpretation": "diagnostic mitigation exists but is not deployable",
        },
        {
            "gate": "dense_control_present_and_matched",
            "passes": bool(dense.get("measured")),
            "interpretation": "dense active-rank control is available from the finite-update matrix",
        },
        {
            "gate": "random_support_control_present",
            "passes": bool(random.get("measured")),
            "interpretation": "random fixed top-k2 support null is available",
        },
        {
            "gate": "order_average_beats_dense_and_random_on_logit_mse",
            "passes": bool(
                order_mse is not None
                and dense.get("mean_logit_mse") is not None
                and random.get("mean_logit_mse") is not None
                and order_mse <= dense["mean_logit_mse"]
                and order_mse <= random["mean_logit_mse"]
            ),
            "interpretation": "narrow diagnostic comparison only; no forgetting or deployability claim",
        },
        {
            "gate": "flat_value_order_averaging_control_present",
            "passes": bool(flat.get("measured")),
            "interpretation": "fails closed because the flat-value order-averaging row is missing",
        },
        {
            "gate": "promotion_or_gpu_allowed",
            "passes": False,
            "interpretation": "blocked: order averaging is a nondeployable upper bound and no-update has zero commutator by construction",
        },
    ]


def _source_row(source: str, path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": packet.get("status"),
        "decision": packet.get("decision"),
        "claim_status": packet.get("claim_status") or packet.get("claim_policy"),
    }


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
            "accepted the recommendation to treat explicit order averaging as a "
            "local commutator diagnostic with dense, flat, random-support, and "
            "no-update controls; promotion and GPU validation remain blocked"
        ),
        "ben_notification_required": bool(notify_ben) or major,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
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


def _write_csv(
    path: Path,
    fieldnames_or_rows: list[str] | list[dict[str, Any]],
    rows: list[dict[str, Any]] | None = None,
) -> None:
    if rows is None:
        rows = fieldnames_or_rows  # type: ignore[assignment]
        fieldnames = sorted({field for row in rows for field in row})
    else:
        fieldnames = fieldnames_or_rows  # type: ignore[assignment]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    evidence = summary["evidence"]
    lines = [
        "# Promoted Top-k-2 Explicit Order-Averaging Mitigation Probe",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Next command: `{summary['next_command']}`",
        "",
        "## Evidence",
        "",
        "- Order-averaged/logit-commutator ratio: "
        f"`{evidence['topk2_order_averaged_to_commutator_anchor_logit_mse_ratio']}`",
        "- Order-averaged CE delta versus best endpoint: "
        f"`{evidence['topk2_mean_order_averaged_anchor_ce_delta_vs_best_order']}`",
        "- Claim status: `diagnostic_only_not_promoted`",
        "- GPU validation: `blocked`",
        "- Promotion: `blocked`",
        "",
        "## Matched Controls",
        "",
        "- Dense, random-support, rank-matched top-k1, flat-value, and no-update "
        "control rows are written to `control_rows.csv`.",
        "- Gate rows are written to `gate_rows.csv`; flat-value order averaging "
        "fails closed because no measured flat order-averaged row exists.",
        "",
        "## Rationale",
        "",
        summary["rationale"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-selector", type=Path, default=DEFAULT_BRANCH_SELECTOR)
    parser.add_argument(
        "--finite-update-report", type=Path, default=DEFAULT_FINITE_UPDATE_REPORT
    )
    parser.add_argument("--control-matrix", type=Path, default=DEFAULT_CONTROL_MATRIX)
    parser.add_argument("--flat-value-report", type=Path, default=DEFAULT_FLAT_VALUE_REPORT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--order-averaging-ratio-gate", type=float, default=0.5)
    parser.add_argument("--order-averaging-ce-delta-gate", type=float, default=0.0)
    args = parser.parse_args(argv)
    summary = run_promoted_topk2_explicit_order_averaging_mitigation_probe(
        branch_selector_path=args.branch_selector,
        finite_update_report_path=args.finite_update_report,
        control_matrix_path=args.control_matrix,
        flat_value_report_path=args.flat_value_report,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
        order_averaging_ratio_gate=args.order_averaging_ratio_gate,
        order_averaging_ce_delta_gate=args.order_averaging_ce_delta_gate,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
