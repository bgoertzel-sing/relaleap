"""Router-policy mitigation gate for promoted contextual top-k-2 order risk."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path(
    "configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml"
)
DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_promoted_topk2_router_policy_mitigation_probe"
)
DEFAULT_ORDER_AVERAGING_PROBE = Path(
    "results/audits/token_larger_promoted_topk2_explicit_order_averaging_mitigation_probe/summary.json"
)
DEFAULT_RETENTION_MITIGATION_PROBE = Path(
    "results/audits/token_larger_promoted_topk2_retention_mitigation_probe/summary.json"
)
DEFAULT_UPDATE_DECOMPOSITION_AUDIT = Path(
    "results/audits/token_larger_promoted_topk2_update_decomposition_audit/summary.json"
)
DEFAULT_VALUE_MITIGATION_GATE = Path(
    "results/audits/token_larger_promoted_topk2_value_mitigation_gate/summary.json"
)
DEFAULT_COMMUTATOR_VALUE_PENALTY_PROBE = Path(
    "results/audits/token_larger_promoted_topk2_commutator_value_penalty_probe/summary.json"
)
DEFAULT_FINITE_UPDATE_REPORT = Path(
    "results/reports/token_larger_promoted_topk2_finite_update_order_control_audit/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")

ROUTER_POLICY_MITIGATION_NOT_ESTABLISHED = "router_policy_mitigation_not_established"
ROUTER_POLICY_MITIGATION_CANDIDATE_FOUND = "router_policy_mitigation_candidate_found"
VALUE_COMPOSITION_PRIORITIZED = "value_composition_prioritized_over_router_policy"
VALUE_COMPOSITION_PENDING_PENALTY_PROBE = (
    "value_composition_prioritized_pending_penalty_probe"
)
SCALE_CONTROL_PRIORITIZED = "residual_scale_control_prioritized_over_router_policy"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

_PINNED_VARIANT = "router_frozen_transfer_topk2"
_VALUE_DOMINATED_DECISION = "value_update_dominated_order_sensitivity"
_ORDER_DIAGNOSTIC_DECISION = "explicit_order_averaging_diagnostic_candidate_not_promoted"


def run_promoted_topk2_router_policy_mitigation_probe(
    *,
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path = DEFAULT_OUT_DIR,
    order_averaging_probe_path: Path = DEFAULT_ORDER_AVERAGING_PROBE,
    retention_mitigation_probe_path: Path = DEFAULT_RETENTION_MITIGATION_PROBE,
    update_decomposition_audit_path: Path = DEFAULT_UPDATE_DECOMPOSITION_AUDIT,
    value_mitigation_gate_path: Path = DEFAULT_VALUE_MITIGATION_GATE,
    commutator_value_penalty_probe_path: Path = DEFAULT_COMMUTATOR_VALUE_PENALTY_PROBE,
    finite_update_report_path: Path = DEFAULT_FINITE_UPDATE_REPORT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    commutator_reduction_fraction: float = 0.5,
    transfer_retention_fraction: float = 0.8,
    support_usage_retention_fraction: float = 0.8,
    scale_ratio_gate: float = 2.0,
) -> dict[str, Any]:
    """Interpret existing pinned/sticky/router-policy evidence fail-closed."""

    start = time.time()
    order_probe = _read_json_object(order_averaging_probe_path)
    retention_probe = _read_json_object(retention_mitigation_probe_path)
    decomposition = _read_json_object(update_decomposition_audit_path)
    value_gate = _read_json_object(value_mitigation_gate_path)
    commutator_value_penalty = _read_json_object(commutator_value_penalty_probe_path)
    finite_update = _read_json_object(finite_update_report_path)
    strategy_review = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("explicit_order_averaging_probe", order_averaging_probe_path, order_probe),
        _source_row("retention_mitigation_probe", retention_mitigation_probe_path, retention_probe),
        _source_row("update_decomposition_audit", update_decomposition_audit_path, decomposition),
        _source_row("value_mitigation_gate", value_mitigation_gate_path, value_gate),
        _source_row("finite_update_order_control", finite_update_report_path, finite_update),
        _source_row(
            "commutator_value_penalty_probe_optional",
            commutator_value_penalty_probe_path,
            commutator_value_penalty,
        ),
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
    thresholds = {
        "commutator_reduction_fraction": commutator_reduction_fraction,
        "transfer_retention_fraction": transfer_retention_fraction,
        "support_usage_retention_fraction": support_usage_retention_fraction,
        "scale_ratio_gate": scale_ratio_gate,
    }
    evidence = _evidence_snapshot(
        order_probe=order_probe,
        retention_probe=retention_probe,
        decomposition=decomposition,
        value_gate=value_gate,
        commutator_value_penalty=commutator_value_penalty,
        finite_update=finite_update,
    )
    policy_rows = [_router_policy_row(retention_probe, thresholds)]
    interpretation_rows = _interpretation_rows(evidence, policy_rows[0], thresholds)
    failures = _failures(source_rows, evidence, policy_rows[0])

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        selected_next_action = None
        next_command = None
        next_step = "repair missing router-policy mitigation source artifacts"
        rationale = (
            "The router-policy mitigation probe cannot be interpreted because "
            "required source artifacts or paired numeric gate fields are missing."
        )
    else:
        status = "pass"
        decision = _decision(evidence, policy_rows[0], thresholds)
        selected_next_action, next_command, next_step = _next_action(decision)
        rationale = _rationale(decision, evidence, policy_rows[0], thresholds)

    summary = {
        "status": status,
        "decision": decision,
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "selected_next_action": selected_next_action,
        "next_command": next_command,
        "next_step": next_step,
        "claim_statuses": {
            "contextual_topk2_router": "promoted_operational_default_train_time_support_selection",
            "router_policy_mitigation": (
                "candidate_not_promoted"
                if decision == ROUTER_POLICY_MITIGATION_CANDIDATE_FOUND
                else "not_established"
            ),
            "order_averaging": "diagnostic_only_not_promoted",
            "topk2_causal_cooperation": "not_supported",
            "topk2_finite_update_interference": "unresolved",
        },
        "thresholds": thresholds,
        "source_rows": source_rows,
        "evidence": evidence,
        "router_policy_rows": policy_rows,
        "interpretation_rows": interpretation_rows,
        "strategy_review": strategy_review,
        "failures": failures,
        "rationale": rationale,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "router_policy_rows_csv": str(out_dir / "router_policy_rows.csv"),
            "interpretation_rows_csv": str(out_dir / "interpretation_rows.csv"),
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
    _write_csv(out_dir / "router_policy_rows.csv", policy_rows)
    _write_csv(out_dir / "interpretation_rows.csv", interpretation_rows)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _evidence_snapshot(
    *,
    order_probe: dict[str, Any],
    retention_probe: dict[str, Any],
    decomposition: dict[str, Any],
    value_gate: dict[str, Any],
    commutator_value_penalty: dict[str, Any],
    finite_update: dict[str, Any],
) -> dict[str, Any]:
    finite_metrics = finite_update.get("metrics", {})
    decomposition_metrics = decomposition.get("metrics", {})
    value_metrics = value_gate.get("metrics", {})
    return {
        "order_averaging_decision": order_probe.get("decision"),
        "retention_mitigation_decision": retention_probe.get("decision"),
        "update_decomposition_decision": decomposition.get("decision"),
        "value_mitigation_decision": value_gate.get("decision"),
        "commutator_value_penalty_decision": commutator_value_penalty.get("decision"),
        "finite_update_decision": finite_update.get("decision"),
        "topk2_commutator_anchor_logit_mse": _float_or_none(
            finite_metrics.get("topk2_mean_commutator_anchor_logit_mse")
        ),
        "topk2_commutator_anchor_residual_stream_l2": _float_or_none(
            finite_metrics.get("topk2_mean_commutator_anchor_residual_stream_l2")
        ),
        "dense_commutator_anchor_residual_stream_l2": _float_or_none(
            finite_metrics.get("dense_mean_commutator_anchor_residual_stream_l2")
        ),
        "topk1_commutator_anchor_residual_stream_l2": _float_or_none(
            finite_metrics.get("topk1_mean_commutator_anchor_residual_stream_l2")
        ),
        "random_fixed_topk2_commutator_anchor_residual_stream_l2": _float_or_none(
            finite_metrics.get("random_fixed_topk2_mean_commutator_anchor_residual_stream_l2")
        ),
        "topk2_commutator_anchor_support_churn": _float_or_none(
            finite_metrics.get("topk2_mean_commutator_anchor_support_churn")
        ),
        "topk2_to_dense_commutator_logit_mse_ratio": _float_or_none(
            finite_metrics.get("topk2_to_dense_mean_commutator_anchor_logit_mse_ratio")
        ),
        "topk2_to_topk1_commutator_logit_mse_ratio": _float_or_none(
            finite_metrics.get("topk2_to_topk1_mean_commutator_anchor_logit_mse_ratio")
        ),
        "value_only_fraction_of_full": _float_or_none(
            decomposition_metrics.get("value_only_fraction_of_full")
        ),
        "router_only_fraction_of_full": _float_or_none(
            decomposition_metrics.get("router_only_fraction_of_full")
        ),
        "best_value_mitigation_reduction_fraction": _float_or_none(
            value_metrics.get("best_value_mitigation_reduction_fraction")
        ),
    }


def _router_policy_row(
    retention_probe: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    rows = [
        row
        for row in retention_probe.get("mitigation_rows", [])
        if isinstance(row, dict) and row.get("variant") == _PINNED_VARIANT
    ]
    row = rows[0] if rows else {}
    reduction = _float_or_none(row.get("commutator_anchor_logit_mse_reduction_fraction"))
    transfer_retention = _float_or_none(row.get("transfer_retention_fraction"))
    support_retention = _float_or_none(row.get("support_usage_retention_fraction"))
    qualifies = (
        reduction is not None
        and transfer_retention is not None
        and support_retention is not None
        and reduction >= thresholds["commutator_reduction_fraction"]
        and transfer_retention >= thresholds["transfer_retention_fraction"]
        and support_retention >= thresholds["support_usage_retention_fraction"]
    )
    return {
        "variant": _PINNED_VARIANT,
        "policy_family": "pinned_support_or_router_frozen_transfer",
        "commutator_anchor_logit_mse": _float_or_none(row.get("commutator_anchor_logit_mse")),
        "baseline_commutator_anchor_logit_mse": _float_or_none(
            row.get("baseline_commutator_anchor_logit_mse")
        ),
        "commutator_anchor_logit_mse_reduction_fraction": reduction,
        "transfer_retention_fraction": transfer_retention,
        "support_usage_retention_fraction": support_retention,
        "anchor_support_churn_after_transfer": _float_or_none(
            row.get("anchor_support_churn_after_transfer")
        ),
        "commutator_anchor_support_churn": _float_or_none(
            row.get("commutator_anchor_support_churn")
        ),
        "passes_router_policy_gate": qualifies,
        "claim_status": "candidate_not_promoted" if qualifies else "not_established",
    }


def _interpretation_rows(
    evidence: dict[str, Any],
    policy_row: dict[str, Any],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    topk2_l2 = evidence.get("topk2_commutator_anchor_residual_stream_l2")
    dense_l2 = evidence.get("dense_commutator_anchor_residual_stream_l2")
    topk1_l2 = evidence.get("topk1_commutator_anchor_residual_stream_l2")
    scale_ratio_dense = _ratio(topk2_l2, dense_l2)
    scale_ratio_topk1 = _ratio(topk2_l2, topk1_l2)
    return [
        {
            "hypothesis": "router_policy_instability",
            "primary_signal": "pinned/router-frozen intervention",
            "metric": "commutator_anchor_logit_mse_reduction_fraction",
            "value": policy_row.get("commutator_anchor_logit_mse_reduction_fraction"),
            "threshold": thresholds["commutator_reduction_fraction"],
            "supported": bool(policy_row.get("passes_router_policy_gate")),
            "interpretation": (
                "router policy remains a trainable mitigation candidate"
                if policy_row.get("passes_router_policy_gate")
                else "pinned/router-frozen policy did not materially reduce commutator"
            ),
        },
        {
            "hypothesis": "residual_scale_confound",
            "primary_signal": "promoted top-k-2 residual L2 versus controls",
            "metric": "topk2_to_dense_residual_l2_ratio",
            "value": scale_ratio_dense,
            "threshold": thresholds["scale_ratio_gate"],
            "supported": scale_ratio_dense is not None
            and scale_ratio_dense >= thresholds["scale_ratio_gate"],
            "interpretation": (
                "residual scale is a material confound"
                if scale_ratio_dense is not None
                and scale_ratio_dense >= thresholds["scale_ratio_gate"]
                else "residual scale alone is not isolated by this gate"
            ),
        },
        {
            "hypothesis": "active_rank_scale_confound",
            "primary_signal": "promoted top-k-2 residual L2 versus rank-matched top-k-1",
            "metric": "topk2_to_topk1_residual_l2_ratio",
            "value": scale_ratio_topk1,
            "threshold": thresholds["scale_ratio_gate"],
            "supported": scale_ratio_topk1 is not None
            and scale_ratio_topk1 >= thresholds["scale_ratio_gate"],
            "interpretation": (
                "rank/top-k support width is entangled with residual scale"
                if scale_ratio_topk1 is not None
                and scale_ratio_topk1 >= thresholds["scale_ratio_gate"]
                else "rank/top-k residual-scale entanglement is below this gate"
            ),
        },
        {
            "hypothesis": "fixed_value_composition",
            "primary_signal": "value-only update decomposition",
            "metric": "value_only_fraction_of_full",
            "value": evidence.get("value_only_fraction_of_full"),
            "threshold": 0.5,
            "supported": _at_least(evidence.get("value_only_fraction_of_full"), 0.5),
            "interpretation": (
                "fixed/value update composition remains the leading mechanism"
                if _at_least(evidence.get("value_only_fraction_of_full"), 0.5)
                else "value-only row does not dominate the full commutator"
            ),
        },
    ]


def _decision(
    evidence: dict[str, Any],
    policy_row: dict[str, Any],
    thresholds: dict[str, float],
) -> str:
    if bool(policy_row.get("passes_router_policy_gate")):
        return ROUTER_POLICY_MITIGATION_CANDIDATE_FOUND
    if _at_least(evidence.get("value_only_fraction_of_full"), 0.5):
        if evidence.get("commutator_value_penalty_decision") is None:
            return VALUE_COMPOSITION_PENDING_PENALTY_PROBE
        return VALUE_COMPOSITION_PRIORITIZED
    topk2_l2 = evidence.get("topk2_commutator_anchor_residual_stream_l2")
    dense_l2 = evidence.get("dense_commutator_anchor_residual_stream_l2")
    if _at_least(_ratio(topk2_l2, dense_l2), thresholds["scale_ratio_gate"]):
        return SCALE_CONTROL_PRIORITIZED
    return ROUTER_POLICY_MITIGATION_NOT_ESTABLISHED


def _next_action(decision: str) -> tuple[str | None, str | None, str]:
    if decision == ROUTER_POLICY_MITIGATION_CANDIDATE_FOUND:
        return (
            "router_stability_regularizer_training_candidate",
            None,
            "validate a bounded router-stability or support-churn regularizer on RunPod before any promotion",
        )
    if decision == VALUE_COMPOSITION_PRIORITIZED:
        return (
            "post_value_router_mitigation_closeout_report",
            None,
            "write a no-training closeout that records router-policy and value-penalty failures before choosing any new mitigation family",
        )
    if decision == VALUE_COMPOSITION_PENDING_PENALTY_PROBE:
        command = (
            "python -m relaleap.experiments.promoted_topk2_commutator_value_penalty_probe "
            "--config configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml "
            "--out results/audits/token_larger_promoted_topk2_commutator_value_penalty_probe"
        )
        return (
            "commutator_value_penalty_probe",
            command,
            "test a commutator-aware value penalty before spending GPU time on router-policy training",
        )
    if decision == SCALE_CONTROL_PRIORITIZED:
        return (
            "residual_norm_matched_intervention_probe",
            None,
            "build a residual-norm-matched top-k-2 finite-update control before interpreting router-policy changes",
        )
    return (
        "no_router_policy_candidate",
        None,
        "keep contextual top-k-2 as operational default but do not pursue router-policy mitigation from this evidence",
    )


def _rationale(
    decision: str,
    evidence: dict[str, Any],
    policy_row: dict[str, Any],
    thresholds: dict[str, float],
) -> str:
    reduction = policy_row.get("commutator_anchor_logit_mse_reduction_fraction")
    value_fraction = evidence.get("value_only_fraction_of_full")
    router_fraction = evidence.get("router_only_fraction_of_full")
    if decision == ROUTER_POLICY_MITIGATION_CANDIDATE_FOUND:
        return (
            "The pinned/router-frozen intervention materially reduced anchor "
            "commutator logit MSE while retaining transfer improvement and support "
            "usage. This selects a router-stability candidate, but it remains "
            "unpromoted until GPU validation and matched controls pass."
        )
    if decision == VALUE_COMPOSITION_PRIORITIZED:
        return (
            "Pinned/router-frozen policy did not clear the preregistered router "
            f"gate: reduction `{reduction}` versus required "
            f"`{thresholds['commutator_reduction_fraction']}`. The update "
            f"decomposition is value-dominated instead: value/full fraction "
            f"`{value_fraction}` versus router/full fraction `{router_fraction}`. "
            "The downstream commutator-aware value-penalty probe is already "
            "present, so the non-duplicative next step is a no-training closeout "
            "rather than rerunning that completed branch."
        )
    if decision == VALUE_COMPOSITION_PENDING_PENALTY_PROBE:
        return (
            "Pinned/router-frozen policy did not clear the preregistered router "
            f"gate: reduction `{reduction}` versus required "
            f"`{thresholds['commutator_reduction_fraction']}`. The update "
            f"decomposition is value-dominated instead: value/full fraction "
            f"`{value_fraction}` versus router/full fraction `{router_fraction}`. "
            "The scientifically coherent next branch is the commutator-aware "
            "value-penalty probe, not router-policy promotion."
        )
    if decision == SCALE_CONTROL_PRIORITIZED:
        return (
            "The pinned/router-frozen policy did not clear the mitigation gate, "
            "and promoted top-k-2 residual stream L2 remains large relative to "
            "norm-matched controls. Treat apparent mitigations as scale controls "
            "until a norm-matched intervention separates scale from mechanism."
        )
    return (
        "The existing paired source artifacts do not establish a router-policy "
        "mitigation candidate. Contextual top-k-2 remains the operational support "
        "router default, but finite-update causal-cooperation claims stay blocked."
    )


def _failures(
    source_rows: list[dict[str, Any]],
    evidence: dict[str, Any],
    policy_row: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows[:5]:
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
        "order_averaging_decision": _ORDER_DIAGNOSTIC_DECISION,
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
    if policy_row.get("commutator_anchor_logit_mse_reduction_fraction") is None:
        failures.append(
            {
                "source": "retention_mitigation_probe",
                "field": f"{_PINNED_VARIANT}.commutator_anchor_logit_mse_reduction_fraction",
                "expected": "numeric",
                "actual": None,
            }
        )
    for field in (
        "topk2_commutator_anchor_logit_mse",
        "topk2_commutator_anchor_residual_stream_l2",
        "dense_commutator_anchor_residual_stream_l2",
        "value_only_fraction_of_full",
        "router_only_fraction_of_full",
    ):
        if evidence.get(field) is None:
            failures.append(
                {
                    "source": "evidence_snapshot",
                    "field": field,
                    "expected": "numeric",
                    "actual": None,
                }
            )
    if (
        evidence.get("update_decomposition_decision") != _VALUE_DOMINATED_DECISION
        and evidence.get("update_decomposition_decision") is None
    ):
        failures.append(
            {
                "source": "update_decomposition_audit",
                "field": "decision",
                "expected": "present decision",
                "actual": None,
            }
        )
    return failures


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
            "accepted the recommendation to make router-policy mitigation a "
            "paired intervention/decomposition gate; router changes are not "
            "promoted when pinned/router-frozen support fails and value or scale "
            "confounds dominate"
        ),
        "ben_notification_required": bool(notify_ben) or major,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    num = _float_or_none(numerator)
    den = _float_or_none(denominator)
    if num is None or den is None or abs(den) <= 1e-12:
        return None
    return num / den


def _at_least(value: Any, threshold: float) -> bool:
    numeric = _float_or_none(value)
    return numeric is not None and numeric >= threshold


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
    router_row = summary["router_policy_rows"][0]
    lines = [
        "# Promoted Top-k-2 Router-Policy Mitigation Probe",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Next command: `{summary['next_command']}`",
        "- Pinned/router-frozen reduction fraction: "
        f"`{router_row['commutator_anchor_logit_mse_reduction_fraction']}`",
        "- Value-only/full commutator fraction: "
        f"`{evidence['value_only_fraction_of_full']}`",
        "- Router-only/full commutator fraction: "
        f"`{evidence['router_only_fraction_of_full']}`",
        "- Top-k-2 residual L2: "
        f"`{evidence['topk2_commutator_anchor_residual_stream_l2']}`",
        "- Dense residual L2: "
        f"`{evidence['dense_commutator_anchor_residual_stream_l2']}`",
        "",
        "## Rationale",
        "",
        summary["rationale"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--order-averaging-probe", type=Path, default=DEFAULT_ORDER_AVERAGING_PROBE)
    parser.add_argument(
        "--retention-mitigation-probe",
        type=Path,
        default=DEFAULT_RETENTION_MITIGATION_PROBE,
    )
    parser.add_argument(
        "--update-decomposition-audit",
        type=Path,
        default=DEFAULT_UPDATE_DECOMPOSITION_AUDIT,
    )
    parser.add_argument("--value-mitigation-gate", type=Path, default=DEFAULT_VALUE_MITIGATION_GATE)
    parser.add_argument("--finite-update-report", type=Path, default=DEFAULT_FINITE_UPDATE_REPORT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--commutator-reduction-fraction", type=float, default=0.5)
    parser.add_argument("--transfer-retention-fraction", type=float, default=0.8)
    parser.add_argument("--support-usage-retention-fraction", type=float, default=0.8)
    parser.add_argument("--scale-ratio-gate", type=float, default=2.0)
    args = parser.parse_args(argv)
    summary = run_promoted_topk2_router_policy_mitigation_probe(
        config_path=args.config,
        out_dir=args.out,
        order_averaging_probe_path=args.order_averaging_probe,
        retention_mitigation_probe_path=args.retention_mitigation_probe,
        update_decomposition_audit_path=args.update_decomposition_audit,
        value_mitigation_gate_path=args.value_mitigation_gate,
        finite_update_report_path=args.finite_update_report,
        strategy_review_path=args.strategy_review,
        commutator_reduction_fraction=args.commutator_reduction_fraction,
        transfer_retention_fraction=args.transfer_retention_fraction,
        support_usage_retention_fraction=args.support_usage_retention_fraction,
        scale_ratio_gate=args.scale_ratio_gate,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
