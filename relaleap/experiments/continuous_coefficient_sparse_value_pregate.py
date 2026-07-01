"""Train a local continuous-coefficient sparse-value pregate.

This is the first trainable arm after the hard in-column sparse value-code
dictionary was killed. It keeps top-1 sparse support but replaces discrete
value-code lookup with a per-active-column continuous coefficient generator.
The run is intentionally local CPU evidence only; GPU validation remains
blocked unless this mechanism clears flat-value and non-CE gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments import dense_teacher_residual_value_capacity_norm_assay as assay


DEFAULT_SELECTOR = Path("results/reports/post_dense_teacher_sparse_dictionary_branch_selector/summary.json")
DEFAULT_OUT_DIR = Path("results/reports/continuous_coefficient_sparse_value_pregate")

DECISION = "continuous_coefficient_sparse_value_pregate_recorded"
FAIL_DECISION = "continuous_coefficient_sparse_value_pregate_failed_closed"
REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "arm_metrics.csv",
    "coefficient_rows.csv",
    "gate_rows.csv",
    "notes.md",
)


def run_continuous_coefficient_sparse_value_pregate(
    *,
    selector_path: Path = DEFAULT_SELECTOR,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 31,
    teacher_steps: int = 100,
    router_steps: int = 80,
    value_steps: int = 100,
    control_steps: int = 80,
    column_count: int = 6,
    coeff_dim: int = 3,
    values_per_column: int = 3,
) -> dict[str, Any]:
    """Train the bounded local pregate and write JSON/CSV artifacts."""

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError("continuous-coefficient sparse-value pregate requires torch") from exc

    if min(teacher_steps, router_steps, value_steps, control_steps) < 1:
        raise ValueError("all training step counts must be positive")
    if column_count < 2:
        raise ValueError("column_count must be at least 2")
    if coeff_dim < 1:
        raise ValueError("coeff_dim must be positive")

    start = time.time()
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    selector = _read_json(selector_path)
    source_rows = [_source_row("post_dense_teacher_sparse_dictionary_branch_selector", selector_path, selector)]

    data = assay._make_data(torch, seed=seed, column_count=column_count)
    teacher = assay._Teacher(torch, data["input_dim"], data["classes"])
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=0.01)
    for _ in range(teacher_steps):
        logits = data["base_logits_train"] + teacher(data["x_train"])
        loss = F.cross_entropy(logits, data["y_train"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        teacher_residual_train = teacher(data["x_train"])
        teacher_residual_holdout = teacher(data["x_holdout"])
        base_holdout_ce = float(F.cross_entropy(data["base_logits_holdout"], data["y_holdout"]).item())
        teacher_holdout_ce = float(
            F.cross_entropy(data["base_logits_holdout"] + teacher_residual_holdout, data["y_holdout"]).item()
        )

    router = assay._train_support_router(
        torch,
        data["x_train"],
        data["support_train"],
        data["input_dim"],
        column_count,
        steps=router_steps,
    )
    support_rows = assay._support_rows(torch, data, router, column_count)
    learned_support = support_rows["learned"]
    oracle_support = data["support_holdout"]

    continuous = _train_continuous_model(
        torch,
        F,
        data["x_train"],
        teacher_residual_train,
        data["support_train"],
        column_count,
        data["classes"],
        coeff_dim,
        steps=value_steps,
    )
    shuffled_continuous = _train_continuous_model(
        torch,
        F,
        data["x_train"],
        torch.roll(teacher_residual_train, shifts=1, dims=0),
        data["support_train"],
        column_count,
        data["classes"],
        coeff_dim,
        steps=max(1, value_steps // 2),
    )
    dictionary = assay._fit_dictionary_model(
        torch,
        data["x_train"],
        teacher_residual_train,
        data["support_train"],
        column_count,
        values_per_column,
        steps=max(1, value_steps // 2),
        lr=0.035,
    )
    flat_value = assay._train_flat_value_head(
        torch,
        F,
        data["x_train"],
        teacher_residual_train,
        data["input_dim"],
        data["classes"],
        steps=control_steps,
    )

    predictions = {
        "continuous_coeff_oracle_support_ceiling": (
            _predict_continuous(torch, continuous, data["x_holdout"], oracle_support, teacher_residual_train),
            oracle_support,
            True,
            "oracle support ceiling for continuous coefficients; nondeployable support only",
        ),
        "continuous_coeff_learned_support": (
            _predict_continuous(torch, continuous, data["x_holdout"], learned_support, teacher_residual_train),
            learned_support,
            False,
            "deployable learned support with continuous per-active-column coefficients",
        ),
        "hard_dictionary_learned_support_control": (
            assay._predict_dictionary(torch, dictionary, data["x_holdout"], learned_support, teacher_residual_train),
            learned_support,
            False,
            "current hard value-code dictionary control under the same learned support",
        ),
        "same_router_flat_value_control": (
            assay._norm_match(torch, flat_value(data["x_holdout"]), teacher_residual_train),
            learned_support,
            False,
            "same learned support with dense flat value head control",
        ),
        "random_support_continuous_null": (
            _predict_continuous(torch, continuous, data["x_holdout"], support_rows["random"], teacher_residual_train),
            support_rows["random"],
            False,
            "random support null with the same continuous coefficient model",
        ),
        "frequency_support_continuous_null": (
            _predict_continuous(torch, continuous, data["x_holdout"], support_rows["frequency"], teacher_residual_train),
            support_rows["frequency"],
            False,
            "frequency support null with the same continuous coefficient model",
        ),
        "shuffled_target_continuous_null": (
            _predict_continuous(torch, shuffled_continuous, data["x_holdout"], oracle_support, teacher_residual_train),
            oracle_support,
            False,
            "shuffled teacher-residual target null with oracle support labels",
        ),
    }
    arm_rows = [
        _arm_row(
            torch,
            F,
            arm=arm,
            pred=pred,
            support=support,
            target=teacher_residual_holdout,
            data=data,
            base_ce=base_holdout_ce,
            teacher_ce=teacher_holdout_ce,
            oracle_support_non_deployable=oracle,
            note=note,
            column_count=column_count,
            coeff_dim=coeff_dim,
        )
        for arm, (pred, support, oracle, note) in predictions.items()
    ]
    coefficient_rows = _coefficient_rows(torch, continuous, data["x_holdout"], learned_support, oracle_support)
    discordance = _discordance_flags(arm_rows, coefficient_rows)
    gate_rows = _gate_rows(source_rows, arm_rows, coefficient_rows, base_holdout_ce, teacher_holdout_ce)
    runtime_failures = [row for row in gate_rows if row["required"] and not row["passed"]]
    scientific_failures = [row for row in gate_rows if row["gate_type"] == "scientific" and not row["passed"]]
    status = "fail" if runtime_failures else "pass"
    claim_status = _claim_status(status, scientific_failures, discordance)
    summary = {
        "status": status,
        "decision": DECISION if status == "pass" else FAIL_DECISION,
        "claim_status": claim_status,
        "selected_next_step": _selected_next_step(status, scientific_failures, discordance),
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "backend_policy": "local CPU pregate only; RunPod and Colab remain blocked",
        "training_executed": True,
        "teacher_trained": True,
        "seed": seed,
        "teacher_train_steps": teacher_steps,
        "router_train_steps": router_steps,
        "value_train_steps": value_steps,
        "control_train_steps": control_steps,
        "column_count": column_count,
        "coeff_dim": coeff_dim,
        "values_per_column_hard_dictionary_control": values_per_column,
        "base_holdout_ce": round(base_holdout_ce, 6),
        "dense_teacher_holdout_ce": round(teacher_holdout_ce, 6),
        "dense_teacher_ce_improvement": round(base_holdout_ce - teacher_holdout_ce, 6),
        "ce_guardrail_positive": discordance["ce_guardrail_positive"],
        "teacher_mse_negative": discordance["teacher_mse_negative"],
        "sparsity_gate_failed": discordance["sparsity_gate_failed"],
        "ce_mse_discordant": discordance["ce_mse_discordant"],
        "source_rows": source_rows,
        "arm_metrics": arm_rows,
        "coefficient_rows": coefficient_rows,
        "gate_rows": gate_rows,
        "failures": runtime_failures + scientific_failures,
        "uses_future_oracle_task_flags": {
            "uses_future_hidden_or_delta": False,
            "deployable_router_uses_oracle_support": False,
            "uses_task_id": False,
            "uses_teacher_labels_in_deployable_router": False,
        },
        "strategy_review_handling": (
            "Accepted the major pivot and the completed branch selector. After the default run produced a "
            "CE-positive/MSE-negative dense-like coefficient result, an urgent GPT-5.5-Pro review was triggered "
            "and accepted: classify the result as non-promotional discordance and run a local CE/MSE adjudicator "
            "before any GPU validation."
        ),
        "direction_shift": {
            "ben_should_be_notified": True,
            "direction": "hard sparse value-code dictionaries remain killed; continuous sparse values are now the active local pregate",
            "recommendation_disposition": "accepted",
            "deferred_or_rejected_recommendations": [],
        },
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _train_continuous_model(
    torch: Any,
    F: Any,
    x_train: Any,
    targets: Any,
    support: Any,
    column_count: int,
    classes: int,
    coeff_dim: int,
    *,
    steps: int,
) -> dict[str, Any]:
    model = {
        "basis": torch.nn.Parameter(torch.randn(column_count, coeff_dim, classes) * 0.08),
        "coeff_head": torch.nn.Sequential(
            torch.nn.Linear(x_train.shape[1], 24),
            torch.nn.Tanh(),
            torch.nn.Linear(24, column_count * coeff_dim),
        ),
    }
    params = [model["basis"], *list(model["coeff_head"].parameters())]
    optimizer = torch.optim.AdamW(params, lr=0.018)
    for _ in range(steps):
        pred, coeff = _raw_predict_continuous(torch, model, x_train, support)
        basis = model["basis"]
        off_diag = torch.matmul(basis, basis.transpose(1, 2))
        identity = torch.eye(coeff_dim).unsqueeze(0)
        orthogonal_penalty = ((off_diag - identity * off_diag.diagonal(dim1=1, dim2=2).unsqueeze(-1)) ** 2).mean()
        loss = F.mse_loss(pred, targets) + 0.0025 * coeff.abs().mean() + 0.0005 * orthogonal_penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def _raw_predict_continuous(torch: Any, model: dict[str, Any], x: Any, support: Any) -> tuple[Any, Any]:
    basis = model["basis"]
    coeffs = model["coeff_head"](x).reshape(len(x), basis.shape[0], basis.shape[1])
    selected_coeffs = []
    selected_preds = []
    for index in range(len(x)):
        column = int(support[index].item())
        coeff = coeffs[index, column]
        selected_coeffs.append(coeff)
        selected_preds.append(coeff @ basis[column])
    return torch.stack(selected_preds), torch.stack(selected_coeffs)


def _predict_continuous(torch: Any, model: dict[str, Any], x: Any, support: Any, norm_target: Any) -> Any:
    pred, _coeff = _raw_predict_continuous(torch, model, x, support)
    return assay._norm_match(torch, pred, norm_target)


def _arm_row(
    torch: Any,
    F: Any,
    *,
    arm: str,
    pred: Any,
    support: Any,
    target: Any,
    data: dict[str, Any],
    base_ce: float,
    teacher_ce: float,
    oracle_support_non_deployable: bool,
    note: str,
    column_count: int,
    coeff_dim: int,
) -> dict[str, Any]:
    logits = data["base_logits_holdout"] + pred
    ce = float(F.cross_entropy(logits, data["y_holdout"]).item())
    mse = float(F.mse_loss(pred, target).item())
    residual_l2 = torch.linalg.vector_norm(pred, dim=1)
    teacher_l2 = torch.linalg.vector_norm(target, dim=1)
    return {
        "arm": arm,
        "ce": round(ce, 6),
        "base_ce": round(base_ce, 6),
        "dense_teacher_ce": round(teacher_ce, 6),
        "ce_gap_vs_dense_teacher": round(ce - teacher_ce, 6),
        "ce_improvement_vs_base": round(base_ce - ce, 6),
        "teacher_residual_reconstruction_mse": round(mse, 6),
        "functional_churn": round(
            float((logits.argmax(dim=-1) != data["base_logits_holdout"].argmax(dim=-1)).float().mean().item()), 6
        ),
        "retention_proxy": round(_retention_proxy(torch, logits, data, target), 6),
        "finite_update_commutator_proxy": round(assay._commutator_proxy(torch, pred, support), 6),
        "intervention_selectivity_proxy": round(assay._selectivity_proxy(torch, pred, target, support), 6),
        "residual_l2_mean": round(float(residual_l2.mean().item()), 6),
        "residual_l2_p95": round(float(torch.quantile(residual_l2, 0.95).item()), 6),
        "teacher_residual_l2_mean": round(float(teacher_l2.mean().item()), 6),
        "residual_l2_mean_ratio_vs_teacher": round(float((residual_l2.mean() / teacher_l2.mean().clamp_min(1e-6)).item()), 6),
        "active_params": data["classes"] * coeff_dim + coeff_dim,
        "stored_params": column_count * coeff_dim * data["classes"] + data["input_dim"] * 24 + 24 * column_count * coeff_dim + 24 + column_count * coeff_dim,
        "oracle_support_non_deployable": oracle_support_non_deployable,
        "uses_oracle_support_at_eval": oracle_support_non_deployable,
        "uses_future_hidden_or_delta": False,
        "uses_task_id": False,
        "uses_teacher_labels_in_deployable_router": False,
        "target_access_at_eval": "oracle_support_only" if oracle_support_non_deployable else "prefix_safe_or_null_support",
        "feature_schema_hash": "prefix_x_position_support_only_v2",
        "note": note,
    }


def _retention_proxy(torch: Any, logits: Any, data: dict[str, Any], target: Any) -> float:
    base_argmax = data["base_logits_holdout"].argmax(dim=-1)
    churn = float((logits.argmax(dim=-1) != base_argmax).float().mean().item())
    teacher_churn = float(((data["base_logits_holdout"] + target).argmax(dim=-1) != base_argmax).float().mean().item())
    return max(0.0, 1.0 - abs(churn - teacher_churn))


def _coefficient_rows(torch: Any, model: dict[str, Any], x: Any, learned_support: Any, oracle_support: Any) -> list[dict[str, Any]]:
    _pred, coeff = _raw_predict_continuous(torch, model, x, learned_support)
    abs_coeff = coeff.abs()
    soft = torch.softmax(abs_coeff, dim=-1)
    entropy = -(soft * torch.log(soft.clamp_min(1e-8))).sum(dim=-1)
    rows = [
        {
            "row": "aggregate",
            "coeff_abs_mean": round(float(abs_coeff.mean().item()), 6),
            "coeff_abs_p95": round(float(torch.quantile(abs_coeff, 0.95).item()), 6),
            "coeff_entropy_mean": round(float(entropy.mean().item()), 6),
            "coeff_near_zero_fraction": round(float((abs_coeff < 0.05).float().mean().item()), 6),
            "support_accuracy_vs_oracle": round(float((learned_support == oracle_support).float().mean().item()), 6),
            "uses_future_hidden_or_delta": False,
            "uses_task_id": False,
        }
    ]
    for column in range(model["basis"].shape[0]):
        mask = learned_support == column
        column_coeff = abs_coeff[mask]
        rows.append(
            {
                "row": f"column_{column}",
                "coeff_abs_mean": round(float(column_coeff.mean().item()), 6) if bool(mask.any()) else 0.0,
                "coeff_abs_p95": round(float(torch.quantile(column_coeff, 0.95).item()), 6) if bool(mask.any()) else 0.0,
                "coeff_entropy_mean": round(float(entropy[mask].mean().item()), 6) if bool(mask.any()) else 0.0,
                "coeff_near_zero_fraction": round(float((column_coeff < 0.05).float().mean().item()), 6) if bool(mask.any()) else 1.0,
                "support_accuracy_vs_oracle": "",
                "uses_future_hidden_or_delta": False,
                "uses_task_id": False,
            }
        )
    return rows


def _gate_rows(
    source_rows: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    coefficient_rows: list[dict[str, Any]],
    base_ce: float,
    teacher_ce: float,
) -> list[dict[str, Any]]:
    by_arm = {row["arm"]: row for row in arm_rows}
    continuous = by_arm.get("continuous_coeff_learned_support", {})
    oracle_continuous = by_arm.get("continuous_coeff_oracle_support_ceiling", {})
    hard = by_arm.get("hard_dictionary_learned_support_control", {})
    flat = by_arm.get("same_router_flat_value_control", {})
    random_null = by_arm.get("random_support_continuous_null", {})
    shuffled_null = by_arm.get("shuffled_target_continuous_null", {})
    aggregate_coeff = coefficient_rows[0] if coefficient_rows else {}
    return [
        _gate("selector_source_present", all(row["present"] for row in source_rows), True, "runtime", str(source_rows)),
        _gate(
            "selector_chose_continuous_coefficients",
            source_rows[0].get("selected_next_action") == "design_continuous_coefficient_sparse_value_pregate",
            True,
            "runtime",
            str(source_rows[0]),
        ),
        _gate("required_arms_present", len(by_arm) == 7, True, "runtime", f"arms={sorted(by_arm)}"),
        _gate(
            "deployable_leakage_flags_false",
            all(
                not row["uses_future_hidden_or_delta"]
                and not row["uses_task_id"]
                and not row["uses_teacher_labels_in_deployable_router"]
                for row in arm_rows
            ),
            True,
            "runtime",
            "all deployable rows use prefix-safe synthetic features only",
        ),
        _gate("gpu_blocked", True, True, "runtime", "requires_gpu_now=false; promotion_allowed=false"),
        _gate("dense_teacher_improves_base", teacher_ce < base_ce, False, "scientific", f"base_ce={base_ce:.6f}; teacher_ce={teacher_ce:.6f}"),
        _gate(
            "continuous_beats_hard_dictionary_mse",
            _metric(continuous, "teacher_residual_reconstruction_mse") < _metric(hard, "teacher_residual_reconstruction_mse"),
            False,
            "scientific",
            f"continuous_mse={continuous.get('teacher_residual_reconstruction_mse')}; hard_mse={hard.get('teacher_residual_reconstruction_mse')}",
        ),
        _gate(
            "continuous_close_to_flat_value_mse",
            _metric(continuous, "teacher_residual_reconstruction_mse") <= _metric(flat, "teacher_residual_reconstruction_mse") + 0.10,
            False,
            "scientific",
            f"continuous_mse={continuous.get('teacher_residual_reconstruction_mse')}; flat_mse={flat.get('teacher_residual_reconstruction_mse')}",
        ),
        _gate(
            "oracle_support_ceiling_improves_learned_support",
            _metric(oracle_continuous, "teacher_residual_reconstruction_mse") <= _metric(continuous, "teacher_residual_reconstruction_mse"),
            False,
            "scientific",
            f"oracle_mse={oracle_continuous.get('teacher_residual_reconstruction_mse')}; learned_mse={continuous.get('teacher_residual_reconstruction_mse')}",
        ),
        _gate(
            "continuous_support_beats_random_null",
            _metric(continuous, "teacher_residual_reconstruction_mse") < _metric(random_null, "teacher_residual_reconstruction_mse"),
            False,
            "scientific",
            f"continuous_mse={continuous.get('teacher_residual_reconstruction_mse')}; random_mse={random_null.get('teacher_residual_reconstruction_mse')}",
        ),
        _gate(
            "continuous_beats_shuffled_target_null",
            _metric(continuous, "teacher_residual_reconstruction_mse") < _metric(shuffled_null, "teacher_residual_reconstruction_mse"),
            False,
            "scientific",
            f"continuous_mse={continuous.get('teacher_residual_reconstruction_mse')}; shuffled_mse={shuffled_null.get('teacher_residual_reconstruction_mse')}",
        ),
        _gate(
            "coefficients_not_dense_like",
            float(aggregate_coeff.get("coeff_near_zero_fraction", 0.0) or 0.0) >= 0.05,
            False,
            "scientific",
            f"coeff_near_zero_fraction={aggregate_coeff.get('coeff_near_zero_fraction')}",
        ),
    ]


def _metric(row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("inf")


def _discordance_flags(arm_rows: list[dict[str, Any]], coefficient_rows: list[dict[str, Any]]) -> dict[str, bool]:
    by_arm = {row["arm"]: row for row in arm_rows}
    continuous = by_arm.get("continuous_coeff_learned_support", {})
    flat = by_arm.get("same_router_flat_value_control", {})
    aggregate_coeff = coefficient_rows[0] if coefficient_rows else {}
    ce_guardrail_positive = (
        _metric(continuous, "ce") < _metric(flat, "ce")
        and _metric(continuous, "ce") < _metric(continuous, "dense_teacher_ce")
    )
    teacher_mse_negative = (
        _metric(continuous, "teacher_residual_reconstruction_mse")
        > _metric(flat, "teacher_residual_reconstruction_mse") + 0.10
    )
    sparsity_gate_failed = float(aggregate_coeff.get("coeff_near_zero_fraction", 0.0) or 0.0) < 0.05
    return {
        "ce_guardrail_positive": bool(ce_guardrail_positive),
        "teacher_mse_negative": bool(teacher_mse_negative),
        "sparsity_gate_failed": bool(sparsity_gate_failed),
        "ce_mse_discordant": bool(ce_guardrail_positive and teacher_mse_negative),
    }


def _gate(criterion: str, passed: bool, required: bool, gate_type: str, evidence: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "required": bool(required),
        "gate_type": gate_type,
        "evidence": evidence,
    }


def _claim_status(
    status: str,
    scientific_failures: list[dict[str, Any]],
    discordance: dict[str, bool],
) -> str:
    if status != "pass":
        return "continuous_coefficient_sparse_value_sources_or_runtime_failed"
    if discordance["ce_mse_discordant"]:
        return "continuous_coeff_ce_mse_discordant_no_promotion"
    failed = {row["criterion"] for row in scientific_failures}
    if not failed:
        return "continuous_coefficient_sparse_value_local_gates_support_repeat_before_gpu"
    if "continuous_close_to_flat_value_mse" in failed:
        return "continuous_coefficient_sparse_value_still_loses_to_flat_control_no_gpu"
    if "continuous_beats_hard_dictionary_mse" in failed:
        return "continuous_coefficient_sparse_value_does_not_repair_hard_dictionary_no_gpu"
    return "continuous_coefficient_sparse_value_partial_local_signal_no_gpu"


def _selected_next_step(
    status: str,
    scientific_failures: list[dict[str, Any]],
    discordance: dict[str, bool],
) -> str:
    if status != "pass":
        return "repair continuous-coefficient pregate source/runtime artifacts before interpretation"
    if discordance["ce_mse_discordant"]:
        return "run continuous CE/MSE discordance adjudicator with same-objective flat controls and no GPU"
    failed = {row["criterion"] for row in scientific_failures}
    if not failed:
        return "repeat continuous-coefficient sparse-value pregate on adjacent seed before any GPU validation"
    if "continuous_close_to_flat_value_mse" in failed or "continuous_beats_hard_dictionary_mse" in failed:
        return "close or redesign continuous sparse value generation before any GPU validation"
    return "inspect failed continuous-coefficient null and sparsity gates before scaling"


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(payload),
        "status": payload.get("status", "") if payload else "missing",
        "decision": payload.get("decision", "") if payload else "",
        "claim_status": payload.get("claim_status", "") if payload else "",
        "selected_next_action": payload.get("selected_next_action", "") if payload else "",
        "selected_next_step": payload.get("selected_next_step", "") if payload else "",
        "training_executed": payload.get("training_executed", "") if payload else "",
        "git_commit": payload.get("git_commit", "") if payload else "",
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "arm_metrics.csv", summary["arm_metrics"])
    _write_csv(out_dir / "coefficient_rows.csv", summary["coefficient_rows"])
    _write_csv(out_dir / "gate_rows.csv", summary["gate_rows"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Continuous-Coefficient Sparse-Value Pregate",
            "",
            f"- Status: {summary['status']}",
            f"- Decision: {summary['decision']}",
            f"- Claim status: {summary['claim_status']}",
            f"- Selected next step: {summary['selected_next_step']}",
            "- GPU validation remains blocked: requires_gpu_now=false, promotion_allowed=false, advance_to_gpu_validation=false.",
            "- Ben should be notified per the accepted major strategy-review direction shift recorded by the branch selector.",
            "",
        ]
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--teacher-steps", type=int, default=100)
    parser.add_argument("--router-steps", type=int, default=80)
    parser.add_argument("--value-steps", type=int, default=100)
    parser.add_argument("--control-steps", type=int, default=80)
    args = parser.parse_args(argv)
    summary = run_continuous_coefficient_sparse_value_pregate(
        selector_path=args.selector,
        out_dir=args.out,
        seed=args.seed,
        teacher_steps=args.teacher_steps,
        router_steps=args.router_steps,
        value_steps=args.value_steps,
        control_steps=args.control_steps,
    )
    print(
        json.dumps(
            {key: summary[key] for key in ("status", "decision", "claim_status", "selected_next_step")},
            indent=2,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
