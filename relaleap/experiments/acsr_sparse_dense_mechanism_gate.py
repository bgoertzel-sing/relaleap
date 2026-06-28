"""Fail-closed sparse-vs-dense ACSR mechanism separation gate."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_ACSR_DIR = Path("results/audits/token_larger_anticipatory_contextual_support_routing")
DEFAULT_DENSE_MATRIX_DIR = Path("results/reports/dense_residual_rank_norm_matrix")
DEFAULT_DENSE_SYNTHESIS = Path("results/reports/acsr_dense_rank_norm_synthesis/summary.json")
DEFAULT_DENSE_OBSERVABLES_DIR = Path("results/reports/acsr_dense_mechanism_observables")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/acsr_sparse_dense_mechanism_gate")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "mechanism_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)

REQUIRED_ACSR_FILES = (
    "summary.json",
    "router_metrics.csv",
    "same_student_metrics.csv",
    "retention_churn_metrics.csv",
    "feature_perturbation.csv",
    "parameter_counts.csv",
)

REQUIRED_DENSE_FILES = (
    "summary.json",
    "matrix_metrics.csv",
    "per_token_metrics.csv",
    "rank_summary.csv",
)

REQUIRED_SPARSE_ARMS = (
    "acsr_mlp_predicted_future",
    "causal_feature_safe_contextual_topk2",
    "full_context_contextual_topk2_teacher",
)

REQUIRED_NULL_ARMS = (
    "shuffled_predicted_features",
    "token_position_only_predicted_features",
    "random_fixed_topk2",
)

REQUIRED_DENSE_RANKS = (16, 24)

MECHANISM_FIELDS = (
    "anchor_kl_or_logit_mse",
    "functional_churn",
    "retention_or_forgetting",
    "intervention_fingerprint_purity",
)


def run_acsr_sparse_dense_mechanism_gate(
    *,
    acsr_dir: Path = DEFAULT_ACSR_DIR,
    dense_matrix_dir: Path = DEFAULT_DENSE_MATRIX_DIR,
    dense_synthesis_path: Path = DEFAULT_DENSE_SYNTHESIS,
    dense_observables_dir: Path = DEFAULT_DENSE_OBSERVABLES_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Summarize available sparse/dense mechanism observables and refuse weak claims."""

    start = time.time()
    acsr_summary = _read_json(acsr_dir / "summary.json")
    router_rows = _read_csv(acsr_dir / "router_metrics.csv")
    same_student_rows = _read_csv(acsr_dir / "same_student_metrics.csv")
    retention_rows = _read_csv(acsr_dir / "retention_churn_metrics.csv")
    perturbation_rows = _read_csv(acsr_dir / "feature_perturbation.csv")
    parameter_rows = _read_csv(acsr_dir / "parameter_counts.csv")
    dense_summary = _read_json(dense_matrix_dir / "summary.json")
    dense_matrix_rows = _read_csv(dense_matrix_dir / "matrix_metrics.csv")
    dense_rank_rows = _read_csv(dense_matrix_dir / "rank_summary.csv")
    dense_observable_rows = _read_csv(dense_observables_dir / "dense_mechanism_observables.csv")
    control_observable_rows = _read_csv(dense_observables_dir / "control_mechanism_observables.csv")
    dense_synthesis = _read_json(dense_synthesis_path)
    strategy = _strategy_review(strategy_review_path)

    source_rows = _source_gate_rows(
        acsr_dir=acsr_dir,
        dense_matrix_dir=dense_matrix_dir,
        dense_synthesis_path=dense_synthesis_path,
        acsr_summary=acsr_summary,
        dense_summary=dense_summary,
        dense_synthesis=dense_synthesis,
    )
    metrics = _mechanism_rows(
        router_rows=router_rows,
        same_student_rows=same_student_rows,
        retention_rows=retention_rows,
        perturbation_rows=perturbation_rows,
        parameter_rows=parameter_rows,
        dense_matrix_rows=dense_matrix_rows,
        dense_rank_rows=dense_rank_rows,
        dense_observable_rows=dense_observable_rows,
        control_observable_rows=control_observable_rows,
    )
    gate_rows = source_rows + _comparison_gate_rows(metrics, router_rows, dense_rank_rows, strategy)
    hard_source_failures = [row for row in source_rows if not row["passed"]]
    blocking_failures = [row for row in gate_rows if not row["passed"]]

    if hard_source_failures:
        status = "fail"
        decision = "acsr_sparse_dense_mechanism_gate_failed_closed"
        claim_status = "source_artifacts_missing_or_failed"
        selected_next_step = "repair missing ACSR or dense rank/norm source artifacts"
    else:
        status = "pass"
        if blocking_failures:
            decision = "acsr_sparse_dense_mechanism_gate_blocked"
            if _has_missing_observable_failure(blocking_failures):
                claim_status = "sparse_mechanism_claim_blocked_by_observable_gap"
                selected_next_step = (
                    "repair missing local control observables for anchor KL/logit MSE, "
                    "functional churn, retention, and intervention fingerprints"
                )
            else:
                claim_status = "sparse_mechanism_claim_blocked_by_matched_controls"
                selected_next_step = (
                    "run one local mechanism-stratified sparse-vs-control follow-up or demote "
                    "ACSR sparse columns to diagnostics if matched controls continue to win"
                )
        else:
            decision = "acsr_sparse_dense_mechanism_gate_passed"
            claim_status = "sparse_mechanism_separates_from_dense_rank16_24_controls"
            selected_next_step = "repeat the sparse-vs-dense mechanism gate on a held-out local packet"

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "backend_policy": "local artifact gate only; dense mechanism observables must pass before GPU validation",
        "source_dirs": {
            "acsr": str(acsr_dir),
            "dense_matrix": str(dense_matrix_dir),
            "dense_synthesis": str(dense_synthesis_path),
            "dense_observables": str(dense_observables_dir),
        },
        "required_mechanism_fields": list(MECHANISM_FIELDS),
        "required_dense_ranks": list(REQUIRED_DENSE_RANKS),
        "mechanism_metrics": metrics,
        "gate_criteria": gate_rows,
        "failures": blocking_failures,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "promotion_allowed": status == "pass" and not blocking_failures,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, metrics, gate_rows)
    return summary


def _source_gate_rows(
    *,
    acsr_dir: Path,
    dense_matrix_dir: Path,
    dense_synthesis_path: Path,
    acsr_summary: dict[str, Any],
    dense_summary: dict[str, Any],
    dense_synthesis: dict[str, Any],
) -> list[dict[str, Any]]:
    missing_acsr = [name for name in REQUIRED_ACSR_FILES if not (acsr_dir / name).is_file()]
    missing_dense = [name for name in REQUIRED_DENSE_FILES if not (dense_matrix_dir / name).is_file()]
    return [
        _criterion(
            "acsr_packet_files_present",
            not missing_acsr,
            "ACSR packet has required local artifacts",
            {"path": str(acsr_dir), "missing": missing_acsr},
            f"missing ACSR packet files: {missing_acsr}",
        ),
        _criterion(
            "acsr_packet_passed",
            acsr_summary.get("status") == "pass",
            "ACSR source packet status is pass",
            acsr_summary.get("status"),
            "ACSR source packet is missing or not passing",
        ),
        _criterion(
            "dense_matrix_files_present",
            not missing_dense,
            "dense rank/norm matrix has required artifacts",
            {"path": str(dense_matrix_dir), "missing": missing_dense},
            f"missing dense matrix files: {missing_dense}",
        ),
        _criterion(
            "dense_matrix_passed",
            dense_summary.get("status") == "pass",
            "dense rank/norm matrix status is pass",
            dense_summary.get("status"),
            "dense rank/norm matrix is missing or not passing",
        ),
        _criterion(
            "dense_synthesis_present",
            dense_synthesis_path.is_file() and bool(dense_synthesis.get("decision")),
            "dense synthesis summary exists",
            {"path": str(dense_synthesis_path), "decision": dense_synthesis.get("decision")},
            "dense synthesis summary missing",
        ),
    ]


def _mechanism_rows(
    *,
    router_rows: list[dict[str, str]],
    same_student_rows: list[dict[str, str]],
    retention_rows: list[dict[str, str]],
    perturbation_rows: list[dict[str, str]],
    parameter_rows: list[dict[str, str]],
    dense_matrix_rows: list[dict[str, str]],
    dense_rank_rows: list[dict[str, str]],
    dense_observable_rows: list[dict[str, str]],
    control_observable_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sparse_variants = set(REQUIRED_SPARSE_ARMS) | set(REQUIRED_NULL_ARMS) | {
        "parameter_matched_causal_mlp_control"
    }
    for router_row in router_rows:
        variant = router_row.get("variant", "")
        if variant not in sparse_variants:
            continue
        control_observables = _control_observable_row(control_observable_rows, variant)
        rows.append(
            _arm_row(
                arm=variant,
                family=_family(variant),
                ce_loss=_float_or_blank(router_row.get("ce_loss")),
                oracle_regret=_float_or_blank(router_row.get("oracle_regret")),
                residual_l2=_float_or_blank(control_observables.get("residual_l2")) if variant == "parameter_matched_causal_mlp_control" else "",
                active_rank_or_topk=_int_or_blank(router_row.get("top_k")),
                active_params=_active_params(variant, parameter_rows),
                anchor_kl_or_logit_mse=_coalesce_float(
                    _retention_value(retention_rows, variant, "anchor_logit_mse_after_transfer"),
                    control_observables.get("anchor_kl_or_logit_mse"),
                ),
                functional_churn=_coalesce_float(
                    _retention_value(retention_rows, variant, "anchor_support_churn_after_transfer"),
                    control_observables.get("functional_churn"),
                ),
                retention_or_forgetting=_coalesce_float(
                    _retention_value(retention_rows, variant, "anchor_ce_drift"),
                    control_observables.get("retention_or_forgetting"),
                ),
                intervention_fingerprint_purity=_coalesce_float(
                    _intervention_proxy(variant, same_student_rows, perturbation_rows),
                    control_observables.get("intervention_fingerprint_purity"),
                ),
            )
        )
    for rank in REQUIRED_DENSE_RANKS:
        best = _dense_best_rank_row(dense_rank_rows, dense_matrix_rows, rank)
        observables = _dense_observable_row(dense_observable_rows, rank)
        rows.append(
            _arm_row(
                arm=f"dense_rank{rank}_best_norm",
                family="dense_rank_control",
                ce_loss=_float_or_blank(best.get("heldout_ce_loss")),
                oracle_regret="",
                residual_l2=_float_or_blank(best.get("heldout_residual_update_l2")),
                active_rank_or_topk=rank,
                active_params=_int_or_blank(best.get("active_params_proxy")),
                anchor_kl_or_logit_mse=_coalesce_float(
                    best.get("anchor_kl_or_logit_mse"), observables.get("anchor_kl_or_logit_mse")
                ),
                functional_churn=_coalesce_float(
                    best.get("functional_churn"), observables.get("functional_churn")
                ),
                retention_or_forgetting=_coalesce_float(
                    best.get("retention_or_forgetting"), observables.get("retention_or_forgetting")
                ),
                intervention_fingerprint_purity=_float_or_blank(
                    best.get("intervention_fingerprint_purity")
                ) if best.get("intervention_fingerprint_purity", "") not in ("", None) else _float_or_blank(
                    observables.get("intervention_fingerprint_purity")
                ),
                delta_vs_sparse=_float_or_blank(best.get("heldout_delta_minus_sparse_topk2")),
            )
        )
    return rows


def _arm_row(
    *,
    arm: str,
    family: str,
    ce_loss: Any,
    oracle_regret: Any,
    residual_l2: Any,
    active_rank_or_topk: Any,
    active_params: Any,
    anchor_kl_or_logit_mse: Any,
    functional_churn: Any,
    retention_or_forgetting: Any,
    intervention_fingerprint_purity: Any,
    delta_vs_sparse: Any = "",
) -> dict[str, Any]:
    values = {
        "anchor_kl_or_logit_mse": anchor_kl_or_logit_mse,
        "functional_churn": functional_churn,
        "retention_or_forgetting": retention_or_forgetting,
        "intervention_fingerprint_purity": intervention_fingerprint_purity,
    }
    missing = [field for field, value in values.items() if value == ""]
    return {
        "arm": arm,
        "family": family,
        "ce_loss": ce_loss,
        "oracle_regret": oracle_regret,
        "residual_l2": residual_l2,
        "active_rank_or_topk": active_rank_or_topk,
        "active_params": active_params,
        "anchor_kl_or_logit_mse": anchor_kl_or_logit_mse,
        "functional_churn": functional_churn,
        "retention_or_forgetting": retention_or_forgetting,
        "intervention_fingerprint_purity": intervention_fingerprint_purity,
        "delta_vs_sparse": delta_vs_sparse,
        "mechanism_fields_present": not missing,
        "missing_mechanism_fields": ";".join(missing),
    }


def _comparison_gate_rows(
    metrics: list[dict[str, Any]],
    router_rows: list[dict[str, str]],
    dense_rank_rows: list[dict[str, str]],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    arms = {row["arm"]: row for row in metrics}
    sparse_present = all(arm in arms for arm in REQUIRED_SPARSE_ARMS)
    null_present = all(arm in arms for arm in REQUIRED_NULL_ARMS)
    dense_present = all(f"dense_rank{rank}_best_norm" in arms for rank in REQUIRED_DENSE_RANKS)
    control_present = "parameter_matched_causal_mlp_control" in arms
    control_mechanism_ready = control_present and arms["parameter_matched_causal_mlp_control"][
        "mechanism_fields_present"
    ]
    dense_mechanism_ready = dense_present and all(
        arms[f"dense_rank{rank}_best_norm"]["mechanism_fields_present"]
        for rank in REQUIRED_DENSE_RANKS
    )
    sparse_mechanism_ready = sparse_present and arms["acsr_mlp_predicted_future"][
        "mechanism_fields_present"
    ]
    dense_rank_blocker = [
        row for row in dense_rank_rows
        if int(_float(row.get("rank")) or 0) in REQUIRED_DENSE_RANKS
        and _bool(row.get("beats_sparse_topk2"))
    ]
    null_ce_failures = _null_failures(router_rows)
    mechanism_separation = _sparse_dense_mechanism_separation(arms)
    control_separation = _sparse_control_mechanism_separation(arms)
    return [
        _criterion(
            "strategy_review_consumed",
            strategy["present"],
            "latest external strategy review was parsed",
            strategy.get("recommended_next_action"),
            "strategy review absent",
        ),
        _criterion(
            "required_sparse_arms_present",
            sparse_present,
            "ACSR teacher/current/predicted sparse arms are present",
            sorted(arm for arm in arms if arms[arm]["family"].startswith("sparse")),
            "missing required sparse ACSR arms",
        ),
        _criterion(
            "required_null_arms_present",
            null_present,
            "shuffled, token-position, and random/frequency null arms are present",
            sorted(arm for arm in arms if arms[arm]["family"] == "null_support"),
            "missing null support arms",
        ),
        _criterion(
            "dense_rank16_24_present",
            dense_present,
            "dense rank 16 and rank 24 controls are present",
            sorted(arm for arm in arms if arms[arm]["family"] == "dense_rank_control"),
            "missing dense rank 16/24 controls",
        ),
        _criterion(
            "ce_norm_rank_accounting_present",
            dense_present and sparse_present,
            "sparse and dense rows expose CE plus rank/top-k accounting",
            {
                arm: {
                    "ce_loss": arms[arm].get("ce_loss"),
                    "residual_l2": arms[arm].get("residual_l2"),
                    "active_rank_or_topk": arms[arm].get("active_rank_or_topk"),
                }
                for arm in arms
                if arm == "acsr_mlp_predicted_future" or arm.startswith("dense_rank")
            },
            "CE/rank/norm accounting is incomplete",
        ),
        _criterion(
            "dense_mechanism_observables_present",
            dense_mechanism_ready,
            "dense rank 16/24 rows include anchor KL/logit MSE, churn, retention, and fingerprint fields",
            {
                arm: arms[arm]["missing_mechanism_fields"]
                for arm in arms
                if arm.startswith("dense_rank")
            },
            "dense controls lack required non-CE mechanism observables",
        ),
        _criterion(
            "parameter_matched_control_mechanism_observables_present",
            control_mechanism_ready,
            "parameter-matched causal MLP row includes anchor KL/logit MSE, churn, retention, and fingerprint fields",
            arms.get("parameter_matched_causal_mlp_control", {}).get("missing_mechanism_fields"),
            "parameter-matched causal MLP lacks required non-CE mechanism observables",
        ),
        _criterion(
            "sparse_mechanism_observables_present",
            sparse_mechanism_ready,
            "ACSR sparse row includes anchor/churn/retention/fingerprint fields",
            arms.get("acsr_mlp_predicted_future", {}).get("missing_mechanism_fields"),
            "ACSR sparse row lacks required mechanism observables",
        ),
        _criterion(
            "null_supports_fail",
            null_present and not null_ce_failures,
            "null supports are worse than ACSR on available CE rows",
            null_ce_failures,
            "one or more null supports do not fail against ACSR",
        ),
        _criterion(
            "dense_rank16_24_do_not_match_or_beat_sparse",
            not dense_rank_blocker,
            "dense rank 16/24 controls do not match or beat sparse contextual top-k2",
            [
                {
                    "rank": row.get("rank"),
                    "best_delta_minus_sparse_topk2": row.get("best_delta_minus_sparse_topk2"),
                }
                for row in dense_rank_blocker
            ],
            "dense rank 16/24 controls already match or beat sparse CE",
        ),
        _criterion(
            "sparse_beats_dense_on_required_mechanism_fields",
            bool(mechanism_separation["passed"]),
            "ACSR must beat best dense rank16/24 on matched non-CE mechanism fields",
            mechanism_separation,
            "sparse-vs-dense mechanism separation is not established",
        ),
        _criterion(
            "sparse_beats_parameter_matched_control_on_required_mechanism_fields",
            bool(control_separation["passed"]),
            "ACSR must beat the parameter-matched causal MLP control on matched non-CE mechanism fields",
            control_separation,
            "sparse-vs-parameter-matched causal MLP mechanism separation is not established",
        ),
    ]


def _dense_best_rank_row(
    rank_rows: list[dict[str, str]],
    matrix_rows: list[dict[str, str]],
    rank: int,
) -> dict[str, str]:
    rank_row = next((row for row in rank_rows if int(_float(row.get("rank")) or 0) == rank), {})
    best_arm = rank_row.get("best_arm", "")
    matrix = next((row for row in matrix_rows if row.get("arm") == best_arm), {})
    merged = dict(rank_row)
    merged.update(matrix)
    return merged


def _dense_observable_row(rows: list[dict[str, str]], rank: int) -> dict[str, str]:
    target_arm = f"dense_rank{rank}_best_norm"
    return next(
        (
            row
            for row in rows
            if row.get("arm") == target_arm or int(_float(row.get("rank")) or 0) == rank
        ),
        {},
    )


def _control_observable_row(rows: list[dict[str, str]], arm: str) -> dict[str, str]:
    return next((row for row in rows if row.get("arm") == arm), {})


def _has_missing_observable_failure(rows: list[dict[str, Any]]) -> bool:
    missing_criteria = {
        "dense_mechanism_observables_present",
        "parameter_matched_control_mechanism_observables_present",
        "sparse_mechanism_observables_present",
    }
    return any(row.get("criterion") in missing_criteria for row in rows)


def _sparse_dense_mechanism_separation(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sparse = arms.get("acsr_mlp_predicted_future", {})
    dense = [
        arms.get(f"dense_rank{rank}_best_norm", {})
        for rank in REQUIRED_DENSE_RANKS
    ]
    if not sparse or any(not row for row in dense):
        return {"passed": False, "reason": "missing sparse or dense arms"}
    if not sparse.get("mechanism_fields_present") or any(not row.get("mechanism_fields_present") for row in dense):
        return {"passed": False, "reason": "missing sparse or dense mechanism fields"}
    lower_better = ("anchor_kl_or_logit_mse", "functional_churn", "retention_or_forgetting")
    comparisons: dict[str, Any] = {}
    passed = True
    for field in lower_better:
        sparse_value = _float(sparse.get(field))
        dense_best = min(
            (_float(row.get(field)) for row in dense if _float(row.get(field)) is not None),
            default=None,
        )
        field_passed = sparse_value is not None and dense_best is not None and sparse_value < dense_best
        comparisons[field] = {
            "sparse": sparse_value,
            "best_dense_rank16_24": dense_best,
            "sparse_better": field_passed,
            "direction": "lower_is_better",
        }
        passed = passed and field_passed
    sparse_purity = _float(sparse.get("intervention_fingerprint_purity"))
    dense_purity = max(
        (_float(row.get("intervention_fingerprint_purity")) for row in dense if _float(row.get("intervention_fingerprint_purity")) is not None),
        default=None,
    )
    purity_passed = sparse_purity is not None and dense_purity is not None and sparse_purity > dense_purity
    comparisons["intervention_fingerprint_purity"] = {
        "sparse": sparse_purity,
        "best_dense_rank16_24": dense_purity,
        "sparse_better": purity_passed,
        "direction": "higher_is_better",
    }
    return {"passed": passed and purity_passed, "comparisons": comparisons}


def _sparse_control_mechanism_separation(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sparse = arms.get("acsr_mlp_predicted_future", {})
    control = arms.get("parameter_matched_causal_mlp_control", {})
    if not sparse or not control:
        return {"passed": False, "reason": "missing sparse or parameter-matched control arm"}
    if not sparse.get("mechanism_fields_present") or not control.get("mechanism_fields_present"):
        return {"passed": False, "reason": "missing sparse or control mechanism fields"}
    lower_better = ("anchor_kl_or_logit_mse", "functional_churn", "retention_or_forgetting")
    comparisons: dict[str, Any] = {}
    passed = True
    for field in lower_better:
        sparse_value = _float(sparse.get(field))
        control_value = _float(control.get(field))
        field_passed = sparse_value is not None and control_value is not None and sparse_value < control_value
        comparisons[field] = {
            "sparse": sparse_value,
            "parameter_matched_causal_mlp": control_value,
            "sparse_better": field_passed,
            "direction": "lower_is_better",
        }
        passed = passed and field_passed
    sparse_purity = _float(sparse.get("intervention_fingerprint_purity"))
    control_purity = _float(control.get("intervention_fingerprint_purity"))
    purity_passed = sparse_purity is not None and control_purity is not None and sparse_purity > control_purity
    comparisons["intervention_fingerprint_purity"] = {
        "sparse": sparse_purity,
        "parameter_matched_causal_mlp": control_purity,
        "sparse_better": purity_passed,
        "direction": "higher_is_better",
    }
    return {"passed": passed and purity_passed, "comparisons": comparisons}


def _retention_value(rows: list[dict[str, str]], variant: str, field: str) -> Any:
    for row in rows:
        if row.get("variant") == variant and row.get(field, "") not in ("", None):
            return _float_or_blank(row.get(field))
    return ""


def _intervention_proxy(
    variant: str,
    same_student_rows: list[dict[str, str]],
    perturbation_rows: list[dict[str, str]],
) -> Any:
    if variant == "acsr_mlp_predicted_future":
        negative = any(
            row.get("control_type") == "future_perturbation_negative"
            and _bool(row.get("passed"))
            for row in perturbation_rows
        )
        same_student = any(
            row.get("comparison") == "acsr_mlp_predicted_future_support_vs_shuffled_predicted_features"
            and _float(row.get("acsr_minus_control_ce_loss")) is not None
            and (_float(row.get("acsr_minus_control_ce_loss")) or 0.0) < 0.0
            for row in same_student_rows
        )
        return 1.0 if negative and same_student else ""
    if variant in REQUIRED_NULL_ARMS:
        return 0.0
    return ""


def _active_params(variant: str, rows: list[dict[str, str]]) -> Any:
    component = {
        "acsr_mlp_predicted_future": "acsr_mlp_predictor_plus_contextual_router",
        "parameter_matched_causal_mlp_control": "parameter_matched_causal_mlp_control",
    }.get(variant, "residual_columns")
    for row in rows:
        if row.get("component") == component:
            return _int_or_blank(row.get("active_parameter_count"))
    return ""


def _null_failures(router_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_variant = {row.get("variant", ""): row for row in router_rows}
    acsr_ce = _float(by_variant.get("acsr_mlp_predicted_future", {}).get("ce_loss"))
    failures = []
    for null in REQUIRED_NULL_ARMS:
        null_ce = _float(by_variant.get(null, {}).get("ce_loss"))
        if acsr_ce is None or null_ce is None or null_ce <= acsr_ce:
            failures.append({"null_arm": null, "acsr_ce": acsr_ce, "null_ce": null_ce})
    return failures


def _family(variant: str) -> str:
    if variant in REQUIRED_NULL_ARMS:
        return "null_support"
    if variant == "parameter_matched_causal_mlp_control":
        return "parameter_matched_dense_control"
    if variant in REQUIRED_SPARSE_ARMS:
        return "sparse_support"
    return "other"


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "present": False,
            "strategic_change_level": None,
            "notify_ben": None,
            "ben_notification_required": False,
            "recommended_next_action": None,
            "verdict": None,
        }
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() in {
            "strategic_change_level",
            "notify_ben",
            "recommended_next_action",
            "verdict",
        }:
            values[key.strip()] = value.strip()
    notify_ben = values.get("notify_ben", "").lower() == "true"
    return {
        "present": True,
        "strategic_change_level": values.get("strategic_change_level"),
        "notify_ben": notify_ben,
        "ben_notification_required": notify_ben or values.get("strategic_change_level") == "major",
        "recommended_next_action": values.get("recommended_next_action"),
        "verdict": values.get("verdict"),
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No external strategy review was available; gate still fails closed on local artifacts."
    return (
        "Accepted latest review recommendation to replace artifact-contract looping with a "
        "local sparse-vs-dense mechanism-separation gate. No GPU validation selected."
    )


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    metrics: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "mechanism_metrics.csv", metrics)
    _write_csv(out_dir / "gate_criteria.csv", gate_rows)
    lines = [
        "# ACSR Sparse-Dense Mechanism Gate",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Promotion allowed: `{summary['promotion_allowed']}`",
        f"- Next step: {summary['selected_next_step']}",
        "",
        "This gate compares the current sparse ACSR packet with dense rank-16/24 controls. "
        "It treats CE and residual norm as guardrails and requires dense controls to expose "
        "the same non-CE mechanism fields before a sparse mechanism claim can advance.",
    ]
    if summary["failures"]:
        lines.extend(["", "## Blocking Gates"])
        lines.extend(
            f"- `{row['criterion']}`: {row['failure_reason']}" for row in summary["failures"]
        )
    (out_dir / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _criterion(
    criterion: str,
    passed: bool,
    threshold: Any,
    actual: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": actual,
        "failure_reason": "" if passed else failure_reason,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_blank(value: Any) -> float | str:
    parsed = _float(value)
    return "" if parsed is None else parsed


def _coalesce_float(*values: Any) -> float | str:
    for value in values:
        parsed = _float(value)
        if parsed is not None:
            return parsed
    return ""


def _int_or_blank(value: Any) -> int | str:
    parsed = _float(value)
    return "" if parsed is None else int(parsed)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acsr-dir", type=Path, default=DEFAULT_ACSR_DIR)
    parser.add_argument("--dense-matrix-dir", type=Path, default=DEFAULT_DENSE_MATRIX_DIR)
    parser.add_argument("--dense-synthesis", type=Path, default=DEFAULT_DENSE_SYNTHESIS)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_acsr_sparse_dense_mechanism_gate(
        acsr_dir=args.acsr_dir,
        dense_matrix_dir=args.dense_matrix_dir,
        dense_synthesis_path=args.dense_synthesis,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "claim_status": summary["claim_status"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
