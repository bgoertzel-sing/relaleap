"""Route-only contextual top-k-2 support-quality pregate pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_LOCAL_SUPPORT_AUDIT = Path("results/audits/local_causal_contextual_router_support_audit")
DEFAULT_RUNPOD_SUPPORT_AUDIT = Path(
    "results/runpod_fetch/audits/runpod_token_larger_causal_contextual_router_support_audit"
)
DEFAULT_ACTIVE_TOPK1_SYNTHESIS = Path(
    "results/reports/token_larger_active_topk1_causal_retention_synthesis/summary.json"
)
DEFAULT_BRANCH_SELECTOR = Path(
    "results/reports/post_active_topk1_contextual_topk2_branch_selector/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/audits/contextual_topk2_support_quality_pregate_pilot")

PILOT_RECORDED = "contextual_topk2_support_quality_pregate_pilot_recorded"
INSUFFICIENT_EVIDENCE = "contextual_topk2_support_quality_pregate_sources_incomplete"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "arm_metrics.csv",
    "fold_policy_rows.csv",
    "per_token_policy_rows.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_contextual_topk2_support_quality_pregate_pilot(
    *,
    local_support_audit_dir: Path = DEFAULT_LOCAL_SUPPORT_AUDIT,
    runpod_support_audit_dir: Path = DEFAULT_RUNPOD_SUPPORT_AUDIT,
    active_topk1_synthesis_path: Path = DEFAULT_ACTIVE_TOPK1_SYNTHESIS,
    branch_selector_path: Path = DEFAULT_BRANCH_SELECTOR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
    regret_margin: float = 0.002,
    churn_margin: float = 0.0,
    support_switch_penalty: float = 0.004,
    ce_guardrail_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Run a fail-closed route-only pregate over existing support-audit rows."""

    start = time.time()
    local = _load_support_packet("local", local_support_audit_dir)
    runpod = _load_support_packet("runpod", runpod_support_audit_dir)
    active = _read_json(active_topk1_synthesis_path)
    branch_selector = _read_json(branch_selector_path)
    strategy = _strategy_review(strategy_review_path)

    source_rows = [
        _source_row("local_support_audit", local_support_audit_dir / "summary.json", local["summary"]),
        _source_row("runpod_support_audit", runpod_support_audit_dir / "summary.json", runpod["summary"]),
        _source_row("active_topk1_causal_retention_synthesis", active_topk1_synthesis_path, active),
        _source_row("post_active_topk1_contextual_topk2_branch_selector", branch_selector_path, branch_selector),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}; verdict={strategy['verdict']}"
            ),
            "sha256": _file_sha256(strategy_review_path),
        },
    ]
    source_failures = _source_failures(source_rows, branch_selector)
    fold_policy_rows = []
    per_token_policy_rows = []
    arm_rows = []
    if not source_failures:
        fold_policy_rows = _fold_policy_rows(
            [local, runpod],
            regret_margin=regret_margin,
            churn_margin=churn_margin,
        )
        per_token_policy_rows = _per_token_policy_rows(
            [local, runpod],
            regret_margin=regret_margin,
            support_switch_penalty=support_switch_penalty,
        )
        arm_rows = _arm_rows([local, runpod], fold_policy_rows, per_token_policy_rows, active)
    gates = _gate_criteria(
        source_failures,
        arm_rows,
        branch_selector,
        strategy,
        ce_guardrail_tolerance=ce_guardrail_tolerance,
    )
    failures = source_failures + [row for row in gates if not row["passed"]]

    if source_failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        claim_status = "route_only_pregate_uninterpretable"
        selected_next_action = "repair_contextual_topk2_support_quality_sources"
        rationale = (
            "The route-only pregate cannot be interpreted because required source "
            "artifacts are missing, failing, or no longer point to this local pilot."
        )
    else:
        status = "pass"
        decision = PILOT_RECORDED
        promotion_gate_passed = all(row["passed"] for row in gates)
        if promotion_gate_passed:
            claim_status = "route_only_contextual_topk2_support_quality_gate_passed"
            selected_next_action = "consider_bounded_gpu_validation_after_artifact_checks"
            rationale = (
                "The frozen-value route-only pregate improved support-quality and "
                "CE guardrails against linear while beating null controls. GPU "
                "validation may be considered only after the same artifact checks."
            )
        else:
            claim_status = "route_only_contextual_topk2_support_quality_gate_failed_no_gpu"
            selected_next_action = "defer_contextual_router_gpu_validation_and_keep_linear_topk2_control"
            rationale = (
                "The executable fold-level pregate is conservative enough to avoid "
                "the contextual router's worse regret/churn rows, but collapses to "
                "the linear support policy. The per-token one-swap label arm recovers "
                "some local oracle-regret headroom; the churn-aware variant tests "
                "whether that headroom can be retained under a switching penalty. "
                "This remains local route-only evidence, not promotion or GPU evidence."
            )

    evidence = _evidence(arm_rows, fold_policy_rows, active, strategy)
    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_action": selected_next_action,
        "selected_next_step": _selected_next_step(status, failures),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local route-only artifact/test work; Colab/RunPod remain blocked unless gates pass",
        "value_freeze_status": "frozen_existing_topk2_values_route_only_support_policy",
        "feature_safety": "uses causal_contextual_topk2 and linear_topk2 audit rows only; full_context rows are headroom controls",
        "training_executed": False,
        "route_policy": {
            "policy": "linear support unless contextual one-swap is predicted to beat linear by regret and churn margins",
            "per_token_policy": "linear support unless the linear-row one-swap label reduces oracle regret by the calibrated margin",
            "churn_aware_per_token_policy": (
                "linear support unless the one-swap label clears the regret margin plus "
                "a support-switch penalty when it would change the previous selected support"
            ),
            "regret_margin": regret_margin,
            "churn_margin": churn_margin,
            "support_switch_penalty": support_switch_penalty,
            "ce_guardrail_tolerance": ce_guardrail_tolerance,
        },
        "source_rows": source_rows,
        "evidence": evidence,
        "gate_criteria": gates,
        "failures": failures,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "rationale": rationale,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, arm_rows, fold_policy_rows, per_token_policy_rows, gates)
    return summary


def _load_support_packet(backend: str, audit_dir: Path) -> dict[str, Any]:
    return {
        "backend": backend,
        "dir": audit_dir,
        "summary": _read_json(audit_dir / "summary.json"),
        "fold_metrics": _read_csv(audit_dir / "fold_metrics.csv"),
        "aggregate_metrics": _read_csv(audit_dir / "aggregate_metrics.csv"),
        "per_token_support_labels": _read_csv(audit_dir / "per_token_support_labels.csv"),
    }


def _fold_policy_rows(
    packets: list[dict[str, Any]], *, regret_margin: float, churn_margin: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for packet in packets:
        by_fold: dict[str, dict[str, dict[str, str]]] = {}
        for row in packet["fold_metrics"]:
            control = row.get("control", "")
            if control in {"causal_contextual_topk2", "linear_topk2", "full_context_oracle_topk2"}:
                by_fold.setdefault(row.get("fold", ""), {})[control] = row
        for fold in sorted(by_fold, key=lambda item: int(item) if item.isdigit() else item):
            controls = by_fold[fold]
            causal = controls.get("causal_contextual_topk2", {})
            linear = controls.get("linear_topk2", {})
            if not causal or not linear:
                continue
            contextual_regret_delta = _float(causal.get("oracle_support_regret")) - _float(
                linear.get("oracle_support_regret")
            )
            contextual_churn_delta = _float(causal.get("functional_churn_logit_l1")) - _float(
                linear.get("functional_churn_logit_l1")
            )
            contextual_ce_delta = _float(causal.get("router_loss")) - _float(
                linear.get("router_loss")
            )
            accept_contextual = (
                contextual_regret_delta <= -regret_margin
                and contextual_churn_delta <= churn_margin
            )
            selected = causal if accept_contextual else linear
            selected_source = "causal_contextual_topk2_one_swap" if accept_contextual else "linear_topk2_hysteresis"
            rows.append(
                {
                    "backend": packet["backend"],
                    "fold": fold,
                    "selected_source": selected_source,
                    "accepted_contextual_swap": accept_contextual,
                    "contextual_minus_linear_router_loss": contextual_ce_delta,
                    "contextual_minus_linear_oracle_regret": contextual_regret_delta,
                    "contextual_minus_linear_functional_churn": contextual_churn_delta,
                    "pregated_router_loss": _float(selected.get("router_loss")),
                    "pregated_oracle_loss": _float(selected.get("oracle_loss")),
                    "pregated_oracle_support_regret": _float(selected.get("oracle_support_regret")),
                    "pregated_functional_churn_logit_l1": _float(
                        selected.get("functional_churn_logit_l1")
                    ),
                    "pregated_support_change_fraction": _float(
                        selected.get("support_change_fraction")
                    ),
                    "pregated_used_columns": _float(selected.get("used_columns")),
                    "pregated_unique_support_sets": _float(selected.get("unique_support_sets")),
                }
            )
    return rows


def _arm_rows(
    packets: list[dict[str, Any]],
    fold_policy_rows: list[dict[str, Any]],
    per_token_policy_rows: list[dict[str, Any]],
    active: dict[str, Any],
) -> list[dict[str, Any]]:
    base_rows: list[dict[str, Any]] = []
    for packet in packets:
        by_control: dict[str, list[dict[str, str]]] = {}
        for row in packet["fold_metrics"]:
            by_control.setdefault(row.get("control", ""), []).append(row)
        for control in ("linear_topk2", "causal_contextual_topk2", "full_context_oracle_topk2"):
            rows = by_control.get(control, [])
            if rows:
                base_rows.append(_summarize_control(packet["backend"], control, rows))
        base_rows.extend(_null_control_rows(packet["backend"], by_control))

    by_backend_policy = {}
    for row in fold_policy_rows:
        by_backend_policy.setdefault(row["backend"], []).append(row)
    for backend, rows in by_backend_policy.items():
        base_rows.append(
            {
                "backend": backend,
                "arm": "pregated_contextual_topk2_route_only",
                "role": "candidate",
                "feature_safety": "causal_prefix_safe",
                "value_status": "frozen",
                "mean_router_loss": _mean(row["pregated_router_loss"] for row in rows),
                "mean_oracle_loss": _mean(row["pregated_oracle_loss"] for row in rows),
                "mean_oracle_support_regret": _mean(
                    row["pregated_oracle_support_regret"] for row in rows
                ),
                "p90_oracle_support_regret": _percentile(
                    [row["pregated_oracle_support_regret"] for row in rows], 0.9
                ),
                "mean_functional_churn_logit_l1": _mean(
                    row["pregated_functional_churn_logit_l1"] for row in rows
                ),
                "mean_support_change_fraction": _mean(
                    row["pregated_support_change_fraction"] for row in rows
                ),
                "mean_used_columns": _mean(row["pregated_used_columns"] for row in rows),
                "mean_unique_support_sets": _mean(
                    row["pregated_unique_support_sets"] for row in rows
                ),
                "accepted_contextual_swap_fraction": _mean(
                    1.0 if row["accepted_contextual_swap"] else 0.0 for row in rows
                ),
                "same_student_intervention_proxy": _same_student_proxy(active),
            }
        )
    by_backend_token_policy: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in per_token_policy_rows:
        by_backend_token_policy.setdefault((row["backend"], row["policy_arm"]), []).append(row)
    for (backend, policy_arm), rows in by_backend_token_policy.items():
        base_rows.append(
            {
                "backend": backend,
                "arm": policy_arm,
                "role": "candidate",
                "feature_safety": "causal_prefix_safe_label_pilot",
                "value_status": "frozen",
                "mean_router_loss": _mean(row["selected_support_loss"] for row in rows),
                "mean_oracle_loss": _mean(row["oracle_support_loss"] for row in rows),
                "mean_oracle_support_regret": _mean(row["selected_oracle_support_regret"] for row in rows),
                "p90_oracle_support_regret": _percentile(
                    [row["selected_oracle_support_regret"] for row in rows], 0.9
                ),
                "mean_functional_churn_logit_l1": _support_churn_proxy(rows, "selected_support"),
                "mean_support_change_fraction": _support_churn_proxy(rows, "selected_support"),
                "mean_used_columns": _mean(row["selected_support_width"] for row in rows),
                "mean_unique_support_sets": len({row["selected_support"] for row in rows}),
                "accepted_contextual_swap_fraction": _mean(
                    1.0 if row["accepted_one_swap"] else 0.0 for row in rows
                ),
                "same_student_intervention_proxy": _same_student_proxy(active),
            }
        )
    return base_rows


def _per_token_policy_rows(
    packets: list[dict[str, Any]], *, regret_margin: float, support_switch_penalty: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for packet in packets:
        labels = packet.get("per_token_support_labels", [])
        if not labels:
            continue
        by_key: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = {}
        for row in labels:
            key = (
                row.get("fold", ""),
                row.get("sequence_index", ""),
                row.get("position_index", ""),
                row.get("target_token", ""),
            )
            by_key.setdefault(key, {})[row.get("control", "")] = row
        sorted_items = sorted(
            by_key.items(),
            key=lambda item: tuple(_sort_key(part) for part in item[0]),
        )
        for policy_arm in (
            "per_token_one_swap_route_only",
            "churn_aware_per_token_one_swap_route_only",
        ):
            previous_by_fold: dict[str, str] = {}
            for key, controls in sorted_items:
                linear = controls.get("linear_topk2")
                contextual = controls.get("causal_contextual_topk2")
                oracle = controls.get("full_context_oracle_topk2")
                if not linear:
                    continue
                actual_support = linear.get("actual_support", "")
                swap_support = linear.get("best_one_swap_support", "")
                previous_support = previous_by_fold.get(key[0])
                actual_regret = _float(linear.get("oracle_support_regret"))
                swap_regret = _float(linear.get("best_one_swap_regret"))
                regret_reduction = actual_regret - swap_regret
                switch_penalty_applied = (
                    policy_arm == "churn_aware_per_token_one_swap_route_only"
                    and previous_support is not None
                    and swap_support != previous_support
                )
                required_reduction = regret_margin + (
                    support_switch_penalty if switch_penalty_applied else 0.0
                )
                accepted = (
                    linear.get("best_one_swap_improves_actual") == "True"
                    and regret_reduction >= required_reduction
                )
                selected_support = swap_support if accepted else actual_support
                selected_loss = _float(
                    linear.get("best_one_swap_support_loss")
                    if accepted
                    else linear.get("actual_support_loss")
                )
                selected_regret = swap_regret if accepted else actual_regret
                rows.append(
                    {
                        "backend": packet["backend"],
                        "policy_arm": policy_arm,
                        "fold": key[0],
                        "sequence_index": key[1],
                        "position_index": key[2],
                        "target_token": key[3],
                        "selected_source": (
                            "linear_one_swap_churn_aware"
                            if accepted and policy_arm == "churn_aware_per_token_one_swap_route_only"
                            else "linear_one_swap_label"
                            if accepted
                            else "linear_actual_hysteresis"
                        ),
                        "accepted_one_swap": accepted,
                        "switch_penalty_applied": switch_penalty_applied,
                        "required_regret_reduction": required_reduction,
                        "previous_selected_support": previous_support or "",
                        "linear_actual_support": actual_support,
                        "selected_support": selected_support or "",
                        "oracle_support": linear.get("oracle_support", ""),
                        "contextual_actual_support": contextual.get("actual_support", "") if contextual else "",
                        "headroom_actual_support": oracle.get("actual_support", "") if oracle else "",
                        "linear_actual_support_loss": _float(linear.get("actual_support_loss")),
                        "selected_support_loss": selected_loss,
                        "oracle_support_loss": _float(linear.get("oracle_support_loss")),
                        "linear_oracle_support_regret": actual_regret,
                        "selected_oracle_support_regret": selected_regret,
                        "regret_reduction_vs_linear": regret_reduction if accepted else 0.0,
                        "contextual_oracle_support_regret": _float(contextual.get("oracle_support_regret")) if contextual else "",
                        "selected_matches_oracle": selected_support == linear.get("oracle_support"),
                        "selected_support_width": len([part for part in (selected_support or "").split(",") if part != ""]),
                    }
                )
                previous_by_fold[key[0]] = selected_support or ""
    return rows


def _summarize_control(backend: str, control: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    role = {
        "linear_topk2": "strong_control",
        "causal_contextual_topk2": "current_contextual_control",
        "full_context_oracle_topk2": "nondeployable_headroom",
    }[control]
    feature_safety = "nondeployable_future_context" if control == "full_context_oracle_topk2" else "causal_prefix_safe"
    return {
        "backend": backend,
        "arm": control,
        "role": role,
        "feature_safety": feature_safety,
        "value_status": "frozen",
        "mean_router_loss": _mean(_float(row.get("router_loss")) for row in rows),
        "mean_oracle_loss": _mean(_float(row.get("oracle_loss")) for row in rows),
        "mean_oracle_support_regret": _mean(
            _float(row.get("oracle_support_regret")) for row in rows
        ),
        "p90_oracle_support_regret": _percentile(
            [_float(row.get("oracle_support_regret")) for row in rows], 0.9
        ),
        "mean_functional_churn_logit_l1": _mean(
            _float(row.get("functional_churn_logit_l1")) for row in rows
        ),
        "mean_support_change_fraction": _mean(
            _float(row.get("support_change_fraction")) for row in rows
        ),
        "mean_used_columns": _mean(_float(row.get("used_columns")) for row in rows),
        "mean_unique_support_sets": _mean(_float(row.get("unique_support_sets")) for row in rows),
        "accepted_contextual_swap_fraction": "",
        "same_student_intervention_proxy": "",
    }


def _null_control_rows(backend: str, by_control: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    causal = by_control.get("causal_contextual_topk2", [])
    linear = by_control.get("linear_topk2", [])
    rows: list[dict[str, Any]] = []
    if causal:
        rows.append(_variant_row(backend, "shuffled_contextual_support_labels", "shuffled_null", causal, "shuffled_support_loss"))
        rows.append(_variant_row(backend, "random_contextual_support", "random_null", causal, "random_support_loss"))
        rows.append(_variant_row(backend, "frequency_contextual_support", "frequency_null", causal, "dominant_fixed_support_loss"))
    if linear:
        rows.append(_variant_row(backend, "token_position_only_linear_proxy", "token_position_control", linear, "router_loss"))
    return rows


def _variant_row(
    backend: str, arm: str, role: str, source_rows: list[dict[str, str]], loss_key: str
) -> dict[str, Any]:
    return {
        "backend": backend,
        "arm": arm,
        "role": role,
        "feature_safety": "causal_or_null",
        "value_status": "frozen",
        "mean_router_loss": _mean(_float(row.get(loss_key)) for row in source_rows),
        "mean_oracle_loss": "",
        "mean_oracle_support_regret": "",
        "p90_oracle_support_regret": "",
        "mean_functional_churn_logit_l1": "",
        "mean_support_change_fraction": "",
        "mean_used_columns": "",
        "mean_unique_support_sets": "",
        "accepted_contextual_swap_fraction": "",
        "same_student_intervention_proxy": "",
    }


def _gate_criteria(
    source_failures: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    branch_selector: dict[str, Any],
    strategy: dict[str, Any],
    *,
    ce_guardrail_tolerance: float,
) -> list[dict[str, Any]]:
    if source_failures:
        return [
            _criterion(
                "required_source_artifacts_present",
                False,
                "support audits, active top-k-1 synthesis, and branch selector must be present",
                [failure.get("criterion") for failure in source_failures],
            )
        ]
    by_backend = _rows_by_backend_arm(arm_rows)
    backend_rows = []
    for backend, arms in sorted(by_backend.items()):
        candidate = arms.get("pregated_contextual_topk2_route_only", {})
        per_token_candidate = arms.get("per_token_one_swap_route_only", {})
        churn_aware_per_token_candidate = arms.get(
            "churn_aware_per_token_one_swap_route_only", {}
        )
        linear = arms.get("linear_topk2", {})
        contextual = arms.get("causal_contextual_topk2", {})
        shuffled = arms.get("shuffled_contextual_support_labels", {})
        random = arms.get("random_contextual_support", {})
        token_position = arms.get("token_position_only_linear_proxy", {})
        backend_rows.append(
            {
                "backend": backend,
                "candidate_regret_delta_vs_linear": _delta(
                    candidate.get("mean_oracle_support_regret"),
                    linear.get("mean_oracle_support_regret"),
                ),
                "candidate_churn_delta_vs_linear": _delta(
                    candidate.get("mean_functional_churn_logit_l1"),
                    linear.get("mean_functional_churn_logit_l1"),
                ),
                "candidate_loss_delta_vs_linear": _delta(
                    candidate.get("mean_router_loss"), linear.get("mean_router_loss")
                ),
                "candidate_loss_delta_vs_contextual": _delta(
                    candidate.get("mean_router_loss"), contextual.get("mean_router_loss")
                ),
                "candidate_loss_minus_shuffled": _delta(
                    candidate.get("mean_router_loss"), shuffled.get("mean_router_loss")
                ),
                "candidate_loss_minus_random": _delta(
                    candidate.get("mean_router_loss"), random.get("mean_router_loss")
                ),
                "candidate_loss_minus_token_position": _delta(
                    candidate.get("mean_router_loss"), token_position.get("mean_router_loss")
                ),
                "accepted_contextual_swap_fraction": candidate.get(
                    "accepted_contextual_swap_fraction"
                ),
                "per_token_candidate_regret_delta_vs_linear": _delta(
                    per_token_candidate.get("mean_oracle_support_regret"),
                    linear.get("mean_oracle_support_regret"),
                ),
                "per_token_candidate_churn_delta_vs_linear": _delta(
                    per_token_candidate.get("mean_support_change_fraction"),
                    linear.get("mean_support_change_fraction"),
                ),
                "per_token_candidate_swap_fraction": per_token_candidate.get(
                    "accepted_contextual_swap_fraction"
                ),
                "churn_aware_per_token_candidate_regret_delta_vs_linear": _delta(
                    churn_aware_per_token_candidate.get("mean_oracle_support_regret"),
                    linear.get("mean_oracle_support_regret"),
                ),
                "churn_aware_per_token_candidate_churn_delta_vs_linear": _delta(
                    churn_aware_per_token_candidate.get("mean_support_change_fraction"),
                    linear.get("mean_support_change_fraction"),
                ),
                "churn_aware_per_token_candidate_swap_fraction": churn_aware_per_token_candidate.get(
                    "accepted_contextual_swap_fraction"
                ),
            }
        )
    return [
        _criterion(
            "branch_selector_requested_local_support_quality_pilot",
            branch_selector.get("selected_next_action")
            == "design_support_quality_preserving_contextual_topk2_pregate",
            "latest branch selector should send the loop to a local contextual top-k-2 support-quality pregate",
            branch_selector.get("selected_next_action"),
        ),
        _criterion(
            "strategy_review_recommendation_incorporated",
            bool(
                strategy.get("present")
                and "support-quality-preserving contextual top-k-2" in strategy.get("recommended_next_action", "")
            ),
            "latest GPT-5.5-Pro review asks for an executable local support-quality-preserving contextual top-k-2 route-only pilot",
            strategy.get("recommended_next_action"),
        ),
        _criterion(
            "candidate_reduces_mean_oracle_regret_vs_linear",
            all(row["candidate_regret_delta_vs_linear"] is not None and row["candidate_regret_delta_vs_linear"] < 0 for row in backend_rows),
            "candidate mean oracle-support regret must be lower than linear top-k-2 on every backend",
            backend_rows,
        ),
        _criterion(
            "candidate_reduces_functional_churn_vs_linear",
            all(row["candidate_churn_delta_vs_linear"] is not None and row["candidate_churn_delta_vs_linear"] < 0 for row in backend_rows),
            "candidate functional churn must be lower than linear top-k-2 on every backend",
            backend_rows,
        ),
        _criterion(
            "candidate_preserves_ce_guardrail_vs_linear",
            all(
                row["candidate_loss_delta_vs_linear"] is not None
                and row["candidate_loss_delta_vs_linear"] <= ce_guardrail_tolerance
                for row in backend_rows
            ),
            "candidate CE/loss must not be materially worse than linear top-k-2",
            backend_rows,
        ),
        _criterion(
            "candidate_beats_null_controls",
            all(
                row["candidate_loss_minus_shuffled"] is not None
                and row["candidate_loss_minus_shuffled"] < 0
                and row["candidate_loss_minus_random"] is not None
                and row["candidate_loss_minus_random"] < 0
                and row["candidate_loss_minus_token_position"] is not None
                and row["candidate_loss_minus_token_position"] <= ce_guardrail_tolerance
                for row in backend_rows
            ),
            "candidate must beat shuffled/random nulls and avoid losing to token-position/linear proxy",
            backend_rows,
        ),
        _criterion(
            "per_token_one_swap_reduces_oracle_regret_vs_linear",
            any(
                row["per_token_candidate_regret_delta_vs_linear"] is not None
                and row["per_token_candidate_regret_delta_vs_linear"] < 0
                for row in backend_rows
            ),
            "per-token one-swap labels should recover support-quality headroom on at least one source",
            backend_rows,
        ),
        _criterion(
            "per_token_one_swap_does_not_increase_support_churn_vs_linear",
            all(
                row["per_token_candidate_churn_delta_vs_linear"] is not None
                and row["per_token_candidate_churn_delta_vs_linear"] <= 0
                for row in backend_rows
                if row["per_token_candidate_regret_delta_vs_linear"] is not None
            ),
            "per-token one-swap support-quality gains must not come from higher support churn",
            backend_rows,
        ),
        _criterion(
            "churn_aware_per_token_one_swap_reduces_oracle_regret_vs_linear",
            any(
                row["churn_aware_per_token_candidate_regret_delta_vs_linear"] is not None
                and row["churn_aware_per_token_candidate_regret_delta_vs_linear"] < 0
                for row in backend_rows
            ),
            "churn-aware per-token one-swap labels should retain support-quality headroom on at least one source",
            backend_rows,
        ),
        _criterion(
            "churn_aware_per_token_one_swap_does_not_increase_support_churn_vs_linear",
            all(
                row["churn_aware_per_token_candidate_churn_delta_vs_linear"] is not None
                and row["churn_aware_per_token_candidate_churn_delta_vs_linear"] <= 0
                for row in backend_rows
                if row["churn_aware_per_token_candidate_regret_delta_vs_linear"] is not None
            ),
            "churn-aware per-token one-swap gains must not raise the support-churn proxy",
            backend_rows,
        ),
        _criterion(
            "gpu_validation_remains_blocked_until_all_support_quality_gates_pass",
            True,
            "this local pilot is not itself GPU evidence; GPU remains blocked unless all above gates pass in a future run",
            "advance_to_gpu_validation=false",
        ),
    ]


def _evidence(
    arm_rows: list[dict[str, Any]],
    fold_policy_rows: list[dict[str, Any]],
    active: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    by_backend = _rows_by_backend_arm(arm_rows)
    backend_summaries = {}
    for backend, arms in by_backend.items():
        candidate = arms.get("pregated_contextual_topk2_route_only", {})
        per_token_candidate = arms.get("per_token_one_swap_route_only", {})
        churn_aware_per_token_candidate = arms.get(
            "churn_aware_per_token_one_swap_route_only", {}
        )
        linear = arms.get("linear_topk2", {})
        contextual = arms.get("causal_contextual_topk2", {})
        backend_summaries[backend] = {
            "candidate_mean_router_loss": candidate.get("mean_router_loss"),
            "linear_mean_router_loss": linear.get("mean_router_loss"),
            "contextual_mean_router_loss": contextual.get("mean_router_loss"),
            "candidate_minus_linear_router_loss": _delta(
                candidate.get("mean_router_loss"), linear.get("mean_router_loss")
            ),
            "candidate_minus_contextual_router_loss": _delta(
                candidate.get("mean_router_loss"), contextual.get("mean_router_loss")
            ),
            "candidate_mean_oracle_support_regret": candidate.get(
                "mean_oracle_support_regret"
            ),
            "linear_mean_oracle_support_regret": linear.get("mean_oracle_support_regret"),
            "contextual_mean_oracle_support_regret": contextual.get(
                "mean_oracle_support_regret"
            ),
            "candidate_minus_linear_oracle_regret": _delta(
                candidate.get("mean_oracle_support_regret"),
                linear.get("mean_oracle_support_regret"),
            ),
            "candidate_mean_functional_churn": candidate.get(
                "mean_functional_churn_logit_l1"
            ),
            "linear_mean_functional_churn": linear.get("mean_functional_churn_logit_l1"),
            "contextual_mean_functional_churn": contextual.get(
                "mean_functional_churn_logit_l1"
            ),
            "candidate_minus_linear_functional_churn": _delta(
                candidate.get("mean_functional_churn_logit_l1"),
                linear.get("mean_functional_churn_logit_l1"),
            ),
            "accepted_contextual_swap_fraction": candidate.get(
                "accepted_contextual_swap_fraction"
            ),
            "per_token_candidate_mean_router_loss": per_token_candidate.get("mean_router_loss"),
            "per_token_candidate_mean_oracle_support_regret": per_token_candidate.get(
                "mean_oracle_support_regret"
            ),
            "per_token_candidate_minus_linear_oracle_regret": _delta(
                per_token_candidate.get("mean_oracle_support_regret"),
                linear.get("mean_oracle_support_regret"),
            ),
            "per_token_candidate_support_churn_proxy": per_token_candidate.get(
                "mean_support_change_fraction"
            ),
            "per_token_candidate_minus_linear_support_churn_proxy": _delta(
                per_token_candidate.get("mean_support_change_fraction"),
                linear.get("mean_support_change_fraction"),
            ),
            "per_token_candidate_accepted_one_swap_fraction": per_token_candidate.get(
                "accepted_contextual_swap_fraction"
            ),
            "churn_aware_per_token_candidate_mean_router_loss": churn_aware_per_token_candidate.get("mean_router_loss"),
            "churn_aware_per_token_candidate_mean_oracle_support_regret": churn_aware_per_token_candidate.get(
                "mean_oracle_support_regret"
            ),
            "churn_aware_per_token_candidate_minus_linear_oracle_regret": _delta(
                churn_aware_per_token_candidate.get("mean_oracle_support_regret"),
                linear.get("mean_oracle_support_regret"),
            ),
            "churn_aware_per_token_candidate_support_churn_proxy": churn_aware_per_token_candidate.get(
                "mean_support_change_fraction"
            ),
            "churn_aware_per_token_candidate_minus_linear_support_churn_proxy": _delta(
                churn_aware_per_token_candidate.get("mean_support_change_fraction"),
                linear.get("mean_support_change_fraction"),
            ),
            "churn_aware_per_token_candidate_accepted_one_swap_fraction": churn_aware_per_token_candidate.get(
                "accepted_contextual_swap_fraction"
            ),
        }
    active_evidence = active.get("evidence", {}) if isinstance(active.get("evidence"), dict) else {}
    return {
        "backend_summaries": backend_summaries,
        "fold_count": len(fold_policy_rows),
        "all_values_frozen": bool(arm_rows) and all(row.get("value_status") == "frozen" for row in arm_rows),
        "training_executed": False,
        "active_topk1_control_claim_status": active.get("claim_status"),
        "active_topk1_local_retention_bracket_supported": bool(
            active_evidence.get("local_retention_churn_bracket_supported")
            or active.get("signals", {}).get("local_retention_churn_bracket_supported")
        ),
        "strategy_recommended_next_action": strategy.get("recommended_next_action"),
        "interpretation": (
            "route-only one-swap/hysteresis candidate is executable but fails "
            "promotion unless it improves regret and churn rather than collapsing "
            "to the linear support policy"
        ),
    }


def _source_failures(source_rows: list[dict[str, Any]], branch_selector: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows[:4]:
        if not row["present"]:
            failures.append(_failure(row["source"], "source_artifact", "present readable source", row["path"]))
    for row in source_rows[:4]:
        if row["present"] and row.get("status") not in {"pass", "ok"}:
            failures.append(_failure(row["source"], "status", "pass or ok", row.get("status")))
    if branch_selector and branch_selector.get("selected_next_action") != "design_support_quality_preserving_contextual_topk2_pregate":
        failures.append(
            _failure(
                "post_active_topk1_contextual_topk2_branch_selector",
                "selected_next_action",
                "design_support_quality_preserving_contextual_topk2_pregate",
                branch_selector.get("selected_next_action"),
            )
        )
    return failures


def _rows_by_backend_arm(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(str(row.get("backend")), {})[str(row.get("arm"))] = row
    return result


def _selected_next_step(status: str, failures: list[dict[str, Any]]) -> str:
    if status == "fail":
        return "repair missing contextual top-k-2 support-quality pregate sources"
    if failures:
        return (
            "keep GPU validation blocked; add same-student forced-support intervention "
            "rows for the churn-aware per-token one-swap arm and recheck selectivity "
            "against linear, token-position, shuffled, and random controls"
        )
    return "run local artifact checks, then consider the configured GPU backend for validation"


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    header: dict[str, str] = {}
    for line in text.splitlines()[:10]:
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    notify = header.get("notify_ben", "false").lower() == "true"
    return {
        "present": path.is_file(),
        "strategic_change_level": header.get("strategic_change_level", ""),
        "notify_ben": notify,
        "ben_notification_required": notify
        or header.get("strategic_change_level", "").lower() == "major",
        "recommended_next_action": header.get("recommended_next_action", ""),
        "verdict": header.get("verdict", ""),
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy.get("present"):
        return "no latest review found; continued from automation status and local artifacts"
    if strategy.get("ben_notification_required"):
        return (
            "latest review requests Ben notification or a major shift; this run records "
            "that direction before making only local non-GPU changes"
        )
    return (
        "latest GPT-5.5-Pro recommendation accepted: implemented an executable local "
        "support-quality-preserving contextual top-k-2 route-only pilot before any GPU"
    )


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(payload),
        "status": payload.get("status", "missing") if payload else "missing",
        "decision": payload.get("decision", "") if payload else "",
        "claim_status": payload.get("claim_status", "") if payload else "",
        "sha256": _file_sha256(path),
    }


def _criterion(criterion: str, passed: bool, expected: str, actual: Any) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "expected": expected,
        "actual": actual,
    }


def _failure(source: str, field: str, expected: Any, actual: Any) -> dict[str, Any]:
    return {
        "source": source,
        "field": field,
        "criterion": f"{source}.{field}",
        "expected": expected,
        "actual": actual,
        "passed": False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    arm_rows: list[dict[str, Any]],
    fold_policy_rows: list[dict[str, Any]],
    per_token_policy_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        out_dir / "source_rows.csv",
        ["source", "path", "present", "status", "decision", "claim_status", "sha256"],
        summary["source_rows"],
    )
    _write_csv(
        out_dir / "arm_metrics.csv",
        [
            "backend",
            "arm",
            "role",
            "feature_safety",
            "value_status",
            "mean_router_loss",
            "mean_oracle_loss",
            "mean_oracle_support_regret",
            "p90_oracle_support_regret",
            "mean_functional_churn_logit_l1",
            "mean_support_change_fraction",
            "mean_used_columns",
            "mean_unique_support_sets",
            "accepted_contextual_swap_fraction",
            "same_student_intervention_proxy",
        ],
        arm_rows,
    )
    _write_csv(
        out_dir / "fold_policy_rows.csv",
        [
            "backend",
            "fold",
            "selected_source",
            "accepted_contextual_swap",
            "contextual_minus_linear_router_loss",
            "contextual_minus_linear_oracle_regret",
            "contextual_minus_linear_functional_churn",
            "pregated_router_loss",
            "pregated_oracle_loss",
            "pregated_oracle_support_regret",
            "pregated_functional_churn_logit_l1",
            "pregated_support_change_fraction",
            "pregated_used_columns",
            "pregated_unique_support_sets",
        ],
        fold_policy_rows,
    )
    _write_csv(
        out_dir / "per_token_policy_rows.csv",
        [
            "backend",
            "policy_arm",
            "fold",
            "sequence_index",
            "position_index",
            "target_token",
            "selected_source",
            "accepted_one_swap",
            "switch_penalty_applied",
            "required_regret_reduction",
            "previous_selected_support",
            "linear_actual_support",
            "selected_support",
            "oracle_support",
            "contextual_actual_support",
            "headroom_actual_support",
            "linear_actual_support_loss",
            "selected_support_loss",
            "oracle_support_loss",
            "linear_oracle_support_regret",
            "selected_oracle_support_regret",
            "regret_reduction_vs_linear",
            "contextual_oracle_support_regret",
            "selected_matches_oracle",
            "selected_support_width",
        ],
        per_token_policy_rows,
    )
    _write_csv(
        out_dir / "gate_criteria.csv",
        ["criterion", "passed", "expected", "actual"],
        [
            {
                **row,
                "actual": json.dumps(row.get("actual"), sort_keys=True),
            }
            for row in gate_rows
        ],
    )
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Contextual Top-k-2 Support-Quality Pregate Pilot",
        "",
        f"- Status: {summary['status']}",
        f"- Decision: {summary['decision']}",
        f"- Claim status: {summary['claim_status']}",
        f"- Selected next action: {summary['selected_next_action']}",
        f"- Requires GPU now: {summary['requires_gpu_now']}",
        f"- Promotion allowed: {summary['promotion_allowed']}",
        "",
        summary["rationale"],
        "",
        "## Gate Criteria",
    ]
    for row in summary["gate_criteria"]:
        mark = "PASS" if row["passed"] else "FAIL"
        lines.append(f"- {mark}: {row['criterion']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_student_proxy(active: dict[str, Any]) -> Any:
    evidence = active.get("evidence", {}) if isinstance(active.get("evidence"), dict) else {}
    metrics = evidence.get("metrics", {}) if isinstance(evidence.get("metrics"), dict) else {}
    return metrics.get("deployable_gain_minus_ungated", "")


def _delta(left: Any, right: Any) -> float | None:
    left_float = _optional_float(left)
    right_float = _optional_float(right)
    if left_float is None or right_float is None:
        return None
    return left_float - right_float


def _float(value: Any) -> float:
    parsed = _optional_float(value)
    return 0.0 if parsed is None else parsed


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Any) -> float:
    vals = [float(value) for value in values if value not in (None, "")]
    return sum(vals) / len(vals) if vals else 0.0


def _percentile(values: list[float], q: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    index = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * q))))
    return vals[index]


def _support_churn_proxy(rows: list[dict[str, Any]], support_key: str) -> float:
    changed: list[bool] = []
    by_fold: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_fold.setdefault(str(row.get("fold", "")), []).append(row)
    for fold_rows in by_fold.values():
        previous = None
        for row in sorted(
            fold_rows,
            key=lambda item: (
                _sort_key(str(item.get("sequence_index", ""))),
                _sort_key(str(item.get("position_index", ""))),
                _sort_key(str(item.get("target_token", ""))),
            ),
        ):
            current = row.get(support_key, "")
            if previous is not None:
                changed.append(current != previous)
            previous = current
    return _mean(1.0 if value else 0.0 for value in changed)


def _sort_key(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-support-audit-dir", type=Path, default=DEFAULT_LOCAL_SUPPORT_AUDIT)
    parser.add_argument("--runpod-support-audit-dir", type=Path, default=DEFAULT_RUNPOD_SUPPORT_AUDIT)
    parser.add_argument("--active-topk1-synthesis", type=Path, default=DEFAULT_ACTIVE_TOPK1_SYNTHESIS)
    parser.add_argument("--branch-selector", type=Path, default=DEFAULT_BRANCH_SELECTOR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--support-switch-penalty", type=float, default=0.004)
    args = parser.parse_args()
    summary = run_contextual_topk2_support_quality_pregate_pilot(
        local_support_audit_dir=args.local_support_audit_dir,
        runpod_support_audit_dir=args.runpod_support_audit_dir,
        active_topk1_synthesis_path=args.active_topk1_synthesis,
        branch_selector_path=args.branch_selector,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
        support_switch_penalty=args.support_switch_penalty,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
