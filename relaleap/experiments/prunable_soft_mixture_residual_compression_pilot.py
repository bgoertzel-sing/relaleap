"""Run a local prunable soft-mixture residual-compression pilot.

This is the trainable follow-up to the design pregate.  It stays local CPU
only and compares soft residual mixtures against same-objective flat controls,
then evaluates whether post-training pruning preserves any CE gain without
losing the dense teacher residual target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments import dense_teacher_residual_value_capacity_norm_assay as assay


DEFAULT_PREGATE = Path("results/reports/prunable_soft_mixture_residual_compression_pregate/summary.json")
DEFAULT_OUT_DIR = Path("results/reports/prunable_soft_mixture_residual_compression_pilot")

DECISION = "prunable_soft_mixture_residual_compression_pilot_recorded"
FAIL_DECISION = "prunable_soft_mixture_residual_compression_pilot_failed_closed"
OBJECTIVES = ("ce_only", "mse_only", "ce_mse_combined")
VARIANTS = ("raw", "norm_matched")
PRUNE_RULES = ("top1", "top2", "top4", "threshold_0p10", "threshold_0p20")
REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "objective_rows.csv",
    "mixture_rows.csv",
    "pruning_rows.csv",
    "gate_rows.csv",
    "notes.md",
)


def run_prunable_soft_mixture_residual_compression_pilot(
    *,
    pregate_path: Path = DEFAULT_PREGATE,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 31,
    teacher_steps: int = 100,
    value_steps: int = 120,
    control_steps: int = 120,
    component_count: int = 8,
) -> dict[str, Any]:
    """Train the bounded local pilot and write report artifacts."""

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError("prunable soft-mixture pilot requires torch") from exc

    if min(teacher_steps, value_steps, control_steps) < 1:
        raise ValueError("all training step counts must be positive")
    if component_count < 2:
        raise ValueError("component_count must be at least 2")

    start = time.time()
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    pregate = _read_json(pregate_path)
    source_rows = [_source_row(pregate_path, pregate)]

    data = assay._make_data(torch, seed=seed, column_count=6)
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

    objective_rows: list[dict[str, Any]] = []
    mixture_rows: list[dict[str, Any]] = []
    pruning_rows: list[dict[str, Any]] = []
    for objective in OBJECTIVES:
        flat = _train_flat_objective(
            torch,
            F,
            data,
            teacher_residual_train,
            objective=objective,
            steps=control_steps,
        )
        soft = _train_soft_mixture(
            torch,
            F,
            data,
            teacher_residual_train,
            objective=objective,
            component_count=component_count,
            steps=value_steps,
            entropy_weight=0.0,
            l1_weight=0.0,
        )
        sparse_soft = _train_soft_mixture(
            torch,
            F,
            data,
            teacher_residual_train,
            objective=objective,
            component_count=component_count,
            steps=value_steps,
            entropy_weight=0.004,
            l1_weight=0.001,
        )
        families = {
            "same_objective_flat": (flat(data["x_holdout"]), None, "flat control"),
            "soft_mixture_unpruned": (
                _predict_soft_mixture(torch, soft, data["x_holdout"])[0],
                _predict_soft_mixture(torch, soft, data["x_holdout"])[1],
                "unpruned soft mixture candidate",
            ),
            "prunable_soft_mixture_entropy_l1": (
                _predict_soft_mixture(torch, sparse_soft, data["x_holdout"])[0],
                _predict_soft_mixture(torch, sparse_soft, data["x_holdout"])[1],
                "entropy/L1 prunable soft mixture candidate",
            ),
            "scale_only_residual_null": (
                teacher_residual_train.mean(dim=0, keepdim=True).repeat(len(data["x_holdout"]), 1)
                * (
                    torch.linalg.vector_norm(flat(data["x_holdout"]), dim=1).mean()
                    / torch.linalg.vector_norm(teacher_residual_train.mean(dim=0, keepdim=True), dim=1).mean().clamp_min(1e-6)
                ),
                None,
                "train-mean residual direction with scalar norm control",
            ),
        }
        shuffled = _train_soft_mixture(
            torch,
            F,
            data,
            torch.roll(teacher_residual_train, shifts=1, dims=0),
            objective="mse_only",
            component_count=component_count,
            steps=max(1, value_steps // 2),
            entropy_weight=0.004,
            l1_weight=0.001,
        )
        shuffled_pred, shuffled_weights = _predict_soft_mixture(torch, shuffled, data["x_holdout"])
        families["shuffled_target_soft_mixture_null"] = (
            shuffled_pred,
            shuffled_weights,
            "shuffled teacher-residual target null",
        )
        for family, (pred, weights, note) in families.items():
            for variant, adjusted in _scaled_predictions(torch, pred, teacher_residual_train).items():
                objective_rows.append(
                    _objective_row(
                        torch,
                        F,
                        objective=objective,
                        family=family,
                        variant=variant,
                        pred=adjusted,
                        data=data,
                        target=teacher_residual_holdout,
                        base_ce=base_holdout_ce,
                        teacher_ce=teacher_holdout_ce,
                        component_count=component_count,
                        note=note,
                    )
                )
            if weights is not None:
                mixture_rows.append(_mixture_row(torch, objective, family, weights))
        for model_family, model in (
            ("soft_mixture_unpruned", soft),
            ("prunable_soft_mixture_entropy_l1", sparse_soft),
        ):
            pruning_rows.extend(
                _pruning_rows(
                    torch,
                    F,
                    objective=objective,
                    family=model_family,
                    model=model,
                    data=data,
                    target=teacher_residual_holdout,
                    norm_target=teacher_residual_train,
                    component_count=component_count,
                    base_ce=base_holdout_ce,
                )
            )

    gate_rows = _gate_rows(
        source_rows,
        objective_rows,
        mixture_rows,
        pruning_rows,
        base_holdout_ce,
        teacher_holdout_ce,
        component_count,
    )
    runtime_failures = [row for row in gate_rows if row["required"] and not row["passed"]]
    scientific_failures = [row for row in gate_rows if row["gate_type"] == "scientific" and not row["passed"]]
    status = "fail" if runtime_failures else "pass"
    claim_status = _claim_status(status, scientific_failures)
    summary = {
        "status": status,
        "decision": DECISION if status == "pass" else FAIL_DECISION,
        "claim_status": claim_status,
        "selected_next_step": _selected_next_step(status, scientific_failures),
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "backend_policy": "local CPU pilot only; RunPod and Colab remain blocked",
        "training_executed": True,
        "teacher_trained": True,
        "seed": seed,
        "teacher_train_steps": teacher_steps,
        "value_train_steps": value_steps,
        "control_train_steps": control_steps,
        "component_count": component_count,
        "objectives": list(OBJECTIVES),
        "variants": list(VARIANTS),
        "prune_rules": list(PRUNE_RULES),
        "base_holdout_ce": round(base_holdout_ce, 6),
        "dense_teacher_holdout_ce": round(teacher_holdout_ce, 6),
        "source_rows": source_rows,
        "objective_rows": objective_rows,
        "mixture_rows": mixture_rows,
        "pruning_rows": pruning_rows,
        "gate_rows": gate_rows,
        "failures": runtime_failures + scientific_failures,
        "strategy_review_handling": (
            "Accepted the latest GPT-5.5-Pro recommendation as already satisfied by the "
            "continuous-coefficient adjudicator/closeout, and executed the selected local "
            "soft-mixture pilot without GPU validation."
        ),
        "deferred_or_rejected_recommendations": [],
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _train_flat_objective(
    torch: Any,
    F: Any,
    data: dict[str, Any],
    targets: Any,
    *,
    objective: str,
    steps: int,
) -> Any:
    model = torch.nn.Sequential(torch.nn.Linear(data["input_dim"], 24), torch.nn.Tanh(), torch.nn.Linear(24, data["classes"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(steps):
        pred = model(data["x_train"])
        loss = _objective_loss(F, objective, pred, targets, data["base_logits_train"], data["y_train"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def _train_soft_mixture(
    torch: Any,
    F: Any,
    data: dict[str, Any],
    targets: Any,
    *,
    objective: str,
    component_count: int,
    steps: int,
    entropy_weight: float,
    l1_weight: float,
) -> dict[str, Any]:
    model = {
        "components": torch.nn.Parameter(torch.randn(component_count, data["classes"]) * 0.08),
        "weight_head": torch.nn.Sequential(
            torch.nn.Linear(data["input_dim"], 24),
            torch.nn.Tanh(),
            torch.nn.Linear(24, component_count),
        ),
        "temperature": 0.75,
    }
    params = [model["components"], *list(model["weight_head"].parameters())]
    optimizer = torch.optim.AdamW(params, lr=0.014)
    for _ in range(steps):
        pred, weights = _predict_soft_mixture(torch, model, data["x_train"])
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1).mean()
        loss = _objective_loss(F, objective, pred, targets, data["base_logits_train"], data["y_train"])
        loss = loss + entropy_weight * entropy + l1_weight * model["components"].abs().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def _predict_soft_mixture(torch: Any, model: dict[str, Any], x: Any) -> tuple[Any, Any]:
    logits = model["weight_head"](x) / float(model["temperature"])
    weights = torch.softmax(logits, dim=-1)
    pred = weights @ model["components"]
    return pred, weights


def _objective_loss(F: Any, objective: str, pred: Any, target: Any, base_logits: Any, y: Any) -> Any:
    ce = F.cross_entropy(base_logits + pred, y)
    mse = F.mse_loss(pred, target)
    if objective == "ce_only":
        return ce
    if objective == "mse_only":
        return mse
    if objective == "ce_mse_combined":
        return ce + 0.25 * mse
    raise ValueError(f"unknown objective: {objective}")


def _scaled_predictions(torch: Any, pred: Any, target_train: Any) -> dict[str, Any]:
    return {
        "raw": pred,
        "norm_matched": assay._norm_match(torch, pred, target_train),
    }


def _objective_row(
    torch: Any,
    F: Any,
    *,
    objective: str,
    family: str,
    variant: str,
    pred: Any,
    data: dict[str, Any],
    target: Any,
    base_ce: float,
    teacher_ce: float,
    component_count: int,
    note: str,
) -> dict[str, Any]:
    logits = data["base_logits_holdout"] + pred
    ce = float(F.cross_entropy(logits, data["y_holdout"]).item())
    mse = float(F.mse_loss(pred, target).item())
    residual_l2 = torch.linalg.vector_norm(pred, dim=1)
    teacher_l2 = torch.linalg.vector_norm(target, dim=1)
    support_proxy = pred.abs().argmax(dim=-1) % component_count
    return {
        "objective": objective,
        "family": family,
        "variant": variant,
        "arm": f"{objective}_{family}_{variant}",
        "ce": round(ce, 6),
        "base_ce": round(base_ce, 6),
        "dense_teacher_ce": round(teacher_ce, 6),
        "ce_gap_vs_flat": "",
        "ce_improvement_vs_base": round(base_ce - ce, 6),
        "teacher_residual_reconstruction_mse": round(mse, 6),
        "functional_churn": round(
            float((logits.argmax(dim=-1) != data["base_logits_holdout"].argmax(dim=-1)).float().mean().item()), 6
        ),
        "finite_update_commutator_proxy": round(assay._commutator_proxy(torch, pred, support_proxy), 6),
        "intervention_selectivity_proxy": round(assay._selectivity_proxy(torch, pred, target, support_proxy), 6),
        "residual_l2_mean": round(float(residual_l2.mean().item()), 6),
        "teacher_residual_l2_mean": round(float(teacher_l2.mean().item()), 6),
        "residual_l2_mean_ratio_vs_teacher": round(float((residual_l2.mean() / teacher_l2.mean().clamp_min(1e-6)).item()), 6),
        "active_params": data["classes"] if family == "same_objective_flat" else component_count + data["classes"],
        "stored_params": (
            data["input_dim"] * 24 + 24 * data["classes"] + 24 + data["classes"]
            if family == "same_objective_flat"
            else component_count * data["classes"] + data["input_dim"] * 24 + 24 * component_count + 24 + component_count
        ),
        "oracle_support_non_deployable": False,
        "uses_future_hidden_or_delta": False,
        "uses_task_id": False,
        "uses_teacher_labels_in_deployable_router": False,
        "target_access_at_eval": "prefix_safe_synthetic_features",
        "feature_schema_hash": "prefix_x_soft_mixture_v1",
        "note": note,
    }


def _mixture_row(torch: Any, objective: str, family: str, weights: Any) -> dict[str, Any]:
    entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1)
    effective = torch.exp(entropy)
    return {
        "objective": objective,
        "family": family,
        "mixture_entropy_mean": round(float(entropy.mean().item()), 6),
        "effective_component_count_mean": round(float(effective.mean().item()), 6),
        "max_weight_mean": round(float(weights.max(dim=-1).values.mean().item()), 6),
        "weight_near_zero_fraction": round(float((weights < 0.05).float().mean().item()), 6),
        "active_component_fraction_gt_0p05": round(float((weights > 0.05).any(dim=0).float().mean().item()), 6),
    }


def _pruning_rows(
    torch: Any,
    F: Any,
    *,
    objective: str,
    family: str,
    model: dict[str, Any],
    data: dict[str, Any],
    target: Any,
    norm_target: Any,
    component_count: int,
    base_ce: float,
) -> list[dict[str, Any]]:
    pred, weights = _predict_soft_mixture(torch, model, data["x_holdout"])
    norm_pred = assay._norm_match(torch, pred, norm_target)
    unpruned_ce = float(F.cross_entropy(data["base_logits_holdout"] + norm_pred, data["y_holdout"]).item())
    unpruned_gain = max(0.0, base_ce - unpruned_ce)
    rows = []
    for rule in PRUNE_RULES:
        pruned_weights = _apply_prune_rule(torch, weights, rule)
        pruned_pred = assay._norm_match(torch, pruned_weights @ model["components"], norm_target)
        ce = float(F.cross_entropy(data["base_logits_holdout"] + pruned_pred, data["y_holdout"]).item())
        mse = float(F.mse_loss(pruned_pred, target).item())
        active_fraction = float((pruned_weights > 0).float().sum(dim=-1).mean().item() / component_count)
        gain = max(0.0, base_ce - ce)
        rows.append(
            {
                "objective": objective,
                "family": family,
                "prune_rule": rule,
                "ce": round(ce, 6),
                "teacher_residual_reconstruction_mse": round(mse, 6),
                "active_component_fraction": round(active_fraction, 6),
                "ce_gain_retention_fraction": round(gain / unpruned_gain, 6) if unpruned_gain > 1e-8 else 0.0,
                "pruned_at_least_half_components": active_fraction <= 0.5,
            }
        )
    return rows


def _apply_prune_rule(torch: Any, weights: Any, rule: str) -> Any:
    if rule.startswith("top"):
        keep = int(rule.replace("top", ""))
        keep = max(1, min(keep, weights.shape[1]))
        threshold = torch.topk(weights, keep, dim=-1).values[:, -1].unsqueeze(-1)
        pruned = torch.where(weights >= threshold, weights, torch.zeros_like(weights))
    else:
        threshold = float(rule.replace("threshold_", "").replace("p", "."))
        pruned = torch.where(weights >= threshold, weights, torch.zeros_like(weights))
    denom = pruned.sum(dim=-1, keepdim=True)
    fallback = torch.nn.functional.one_hot(weights.argmax(dim=-1), num_classes=weights.shape[1]).to(weights.dtype)
    pruned = torch.where(denom > 0, pruned / denom.clamp_min(1e-8), fallback)
    return pruned


def _gate_rows(
    source_rows: list[dict[str, Any]],
    objective_rows: list[dict[str, Any]],
    mixture_rows: list[dict[str, Any]],
    pruning_rows: list[dict[str, Any]],
    base_ce: float,
    teacher_ce: float,
    component_count: int,
) -> list[dict[str, Any]]:
    rows = {(row["objective"], row["family"], row["variant"]): row for row in objective_rows}
    mix = {(row["objective"], row["family"]): row for row in mixture_rows}
    ce_soft = rows.get(("ce_only", "prunable_soft_mixture_entropy_l1", "norm_matched"), {})
    ce_flat = rows.get(("ce_only", "same_objective_flat", "norm_matched"), {})
    mse_soft = rows.get(("mse_only", "prunable_soft_mixture_entropy_l1", "norm_matched"), {})
    mse_flat = rows.get(("mse_only", "same_objective_flat", "norm_matched"), {})
    soft_ce = _metric(ce_soft, "ce")
    flat_ce = _metric(ce_flat, "ce")
    soft_mse = _metric(mse_soft, "teacher_residual_reconstruction_mse")
    flat_mse = _metric(mse_flat, "teacher_residual_reconstruction_mse")
    prune_candidates = [
        row
        for row in pruning_rows
        if row["objective"] == "ce_only"
        and row["family"] == "prunable_soft_mixture_entropy_l1"
        and row["pruned_at_least_half_components"]
    ]
    best_prune_retention = max((_metric(row, "ce_gain_retention_fraction") for row in prune_candidates), default=0.0)
    soft_mix = mix.get(("ce_only", "prunable_soft_mixture_entropy_l1"), {})
    flat_selectivity = _metric(ce_flat, "intervention_selectivity_proxy")
    soft_selectivity = _metric(ce_soft, "intervention_selectivity_proxy")
    flat_commutator = _metric(ce_flat, "finite_update_commutator_proxy")
    soft_commutator = _metric(ce_soft, "finite_update_commutator_proxy")
    required = {
        (objective, family, variant)
        for objective in OBJECTIVES
        for family in (
            "same_objective_flat",
            "soft_mixture_unpruned",
            "prunable_soft_mixture_entropy_l1",
            "scale_only_residual_null",
            "shuffled_target_soft_mixture_null",
        )
        for variant in VARIANTS
    }
    present = set(rows)
    return [
        _gate("pregate_source_present", all(row["present"] for row in source_rows), True, "runtime", str(source_rows)),
        _gate(
            "pregate_selected_pilot",
            source_rows[0].get("selected_next_action") == "implement_prunable_soft_mixture_residual_compression_pilot",
            True,
            "runtime",
            str(source_rows[0]),
        ),
        _gate("required_objective_rows_present", required.issubset(present), True, "runtime", f"rows={len(objective_rows)}"),
        _gate("mixture_rows_present", len(mixture_rows) >= len(OBJECTIVES) * 3, True, "runtime", f"rows={len(mixture_rows)}"),
        _gate("pruning_rows_present", len(pruning_rows) >= len(OBJECTIVES) * 2 * len(PRUNE_RULES), True, "runtime", f"rows={len(pruning_rows)}"),
        _gate("gpu_blocked", True, True, "runtime", "requires_gpu_now=false; promotion_allowed=false"),
        _gate(
            "deployable_leakage_flags_false",
            all(
                not row["uses_future_hidden_or_delta"]
                and not row["uses_task_id"]
                and not row["uses_teacher_labels_in_deployable_router"]
                for row in objective_rows
            ),
            True,
            "runtime",
            "no deployable row uses future hidden/delta, task id, or teacher labels",
        ),
        _gate("dense_teacher_improves_base", teacher_ce < base_ce, False, "scientific", f"base_ce={base_ce:.6f}; teacher_ce={teacher_ce:.6f}"),
        _gate(
            "soft_mixture_beats_flat_ce_same_objective",
            soft_ce + 0.002 < flat_ce,
            False,
            "scientific",
            f"soft_ce={ce_soft.get('ce')}; flat_ce={ce_flat.get('ce')}",
        ),
        _gate(
            "soft_mixture_not_worse_than_flat_mse",
            soft_mse <= flat_mse + 0.02,
            False,
            "scientific",
            f"soft_mse={mse_soft.get('teacher_residual_reconstruction_mse')}; flat_mse={mse_flat.get('teacher_residual_reconstruction_mse')}",
        ),
        _gate(
            "pruning_retains_function_after_halving",
            best_prune_retention >= 0.80,
            False,
            "scientific",
            f"best_retention={best_prune_retention:.6f}",
        ),
        _gate(
            "mixture_is_prunable_not_dense",
            _metric(soft_mix, "weight_near_zero_fraction") >= 0.25
            and _metric(soft_mix, "effective_component_count_mean") <= 0.75 * component_count,
            False,
            "scientific",
            f"mixture={soft_mix}",
        ),
        _gate(
            "mechanism_proxy_improves_over_flat",
            soft_selectivity > flat_selectivity or soft_commutator < flat_commutator,
            False,
            "scientific",
            f"soft_selectivity={soft_selectivity}; flat_selectivity={flat_selectivity}; soft_commutator={soft_commutator}; flat_commutator={flat_commutator}",
        ),
    ]


def _gate(criterion: str, passed: bool, required: bool, gate_type: str, evidence: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "required": bool(required),
        "gate_type": gate_type,
        "evidence": evidence,
    }


def _metric(row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("inf")


def _claim_status(status: str, scientific_failures: list[dict[str, Any]]) -> str:
    if status != "pass":
        return "prunable_soft_mixture_pilot_runtime_failed"
    failed = {row["criterion"] for row in scientific_failures}
    if not failed:
        return "prunable_soft_mixture_local_gates_support_repeat_before_gpu"
    if "soft_mixture_beats_flat_ce_same_objective" in failed:
        return "prunable_soft_mixture_flat_ce_control_blocks_gpu"
    if "soft_mixture_not_worse_than_flat_mse" in failed:
        return "prunable_soft_mixture_teacher_mse_control_blocks_gpu"
    if "pruning_retains_function_after_halving" in failed:
        return "prunable_soft_mixture_pruning_retention_blocks_gpu"
    return "prunable_soft_mixture_partial_local_signal_no_gpu"


def _selected_next_step(status: str, scientific_failures: list[dict[str, Any]]) -> str:
    if status != "pass":
        return "repair prunable soft-mixture pilot source/runtime artifacts before interpretation"
    failed = {row["criterion"] for row in scientific_failures}
    if not failed:
        return "repeat prunable soft-mixture pilot on an adjacent seed before any GPU validation"
    if "soft_mixture_beats_flat_ce_same_objective" in failed or "soft_mixture_not_worse_than_flat_mse" in failed:
        return "close or redesign soft-mixture compression before GPU; same-objective flat controls still dominate"
    if "pruning_retains_function_after_halving" in failed:
        return "redesign soft-mixture sparsity pressure and pruning schedule before GPU"
    return "inspect soft-mixture mechanism-proxy failures before any GPU validation"


def _source_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "prunable_soft_mixture_residual_compression_pregate",
        "path": str(path),
        "present": bool(payload),
        "status": payload.get("status", "") if payload else "missing",
        "decision": payload.get("decision", "") if payload else "",
        "claim_status": payload.get("claim_status", "") if payload else "",
        "selected_next_action": payload.get("selected_next_action", "") if payload else "",
        "selected_next_step": payload.get("selected_next_step", "") if payload else "",
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
    _write_csv(out_dir / "objective_rows.csv", summary["objective_rows"])
    _write_csv(out_dir / "mixture_rows.csv", summary["mixture_rows"])
    _write_csv(out_dir / "pruning_rows.csv", summary["pruning_rows"])
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
    lines = [
        "# Prunable Soft-Mixture Residual-Compression Pilot",
        "",
        f"- Status: {summary['status']}",
        f"- Decision: {summary['decision']}",
        f"- Claim status: {summary['claim_status']}",
        f"- Selected next step: {summary['selected_next_step']}",
        "- Same-objective flat controls, norm matching, pruning sweeps, and null rows are included.",
        "- GPU validation remains blocked: requires_gpu_now=false, promotion_allowed=false, advance_to_gpu_validation=false.",
        "",
    ]
    if summary["failures"]:
        lines.extend(["## Failed Criteria", ""])
        lines.extend(f"- `{row['criterion']}`: {row['evidence']}" for row in summary["failures"])
    return "\n".join(lines) + "\n"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pregate", type=Path, default=DEFAULT_PREGATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--teacher-steps", type=int, default=100)
    parser.add_argument("--value-steps", type=int, default=120)
    parser.add_argument("--control-steps", type=int, default=120)
    parser.add_argument("--component-count", type=int, default=8)
    args = parser.parse_args(argv)
    summary = run_prunable_soft_mixture_residual_compression_pilot(
        pregate_path=args.pregate,
        out_dir=args.out,
        seed=args.seed,
        teacher_steps=args.teacher_steps,
        value_steps=args.value_steps,
        control_steps=args.control_steps,
        component_count=args.component_count,
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
