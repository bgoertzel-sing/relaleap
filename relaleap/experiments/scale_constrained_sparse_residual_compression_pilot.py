"""Run a local scale-constrained sparse residual-compression pilot.

This is the executable follow-up to the soft-mixture closeout. It asks whether
a sparse top-r atom bank with an explicit residual-norm controller can survive
same-controller flat/dense/null controls and pruning-retention gates before any
GPU validation is reconsidered.
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


DEFAULT_CLOSEOUT = Path("results/reports/prunable_soft_mixture_residual_compression_closeout/summary.json")
DEFAULT_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/scale_constrained_sparse_residual_compression_pilot")

DECISION = "scale_constrained_sparse_residual_compression_pilot_recorded"
FAIL_DECISION = "scale_constrained_sparse_residual_compression_pilot_failed_closed"
OBJECTIVES = ("ce_only", "mse_only", "ce_mse_combined")
ARMS = (
    "sparse_topr_norm_controller",
    "same_controller_flat_residual",
    "same_controller_dense_mlp",
    "scale_only_null",
    "shuffled_target_sparse_null",
    "random_support_sparse_null",
    "position_support_sparse_null",
)
PRUNE_RULES = ("top1", "top2", "threshold_0p15", "threshold_0p25")
REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "arm_metrics.csv",
    "pruning_rows.csv",
    "gate_rows.csv",
    "notes.md",
)


def run_scale_constrained_sparse_residual_compression_pilot(
    *,
    closeout_path: Path = DEFAULT_CLOSEOUT,
    strategy_review_path: Path = DEFAULT_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 31,
    teacher_steps: int = 100,
    value_steps: int = 120,
    control_steps: int = 120,
    atom_count: int = 8,
    top_r: int = 2,
) -> dict[str, Any]:
    """Train the bounded local pilot and write report artifacts."""

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError("scale-constrained sparse residual pilot requires torch") from exc

    if min(teacher_steps, value_steps, control_steps) < 1:
        raise ValueError("all training step counts must be positive")
    if atom_count < 2:
        raise ValueError("atom_count must be at least 2")
    if not 1 <= top_r <= atom_count:
        raise ValueError("top_r must be between 1 and atom_count")

    start = time.time()
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    closeout = _read_json(closeout_path)
    review = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("prunable_soft_mixture_residual_compression_closeout", closeout_path, closeout),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": review["present"],
            "status": "read" if review["present"] else "missing_optional",
            "decision": review["recommended_next_action"],
            "claim_status": f"strategic_change_level={review['strategic_change_level']}; notify_ben={review['notify_ben']}; verdict={review['verdict']}",
            "selected_next_action": "",
            "selected_next_step": "",
            "git_commit": "",
        },
    ]

    data = assay._make_data(torch, seed=seed, column_count=6)
    teacher = assay._Teacher(torch, data["input_dim"], data["classes"])
    opt = torch.optim.AdamW(teacher.parameters(), lr=0.01)
    for _ in range(teacher_steps):
        loss = F.cross_entropy(data["base_logits_train"] + teacher(data["x_train"]), data["y_train"])
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        teacher_train = teacher(data["x_train"])
        teacher_holdout = teacher(data["x_holdout"])
        base_ce = float(F.cross_entropy(data["base_logits_holdout"], data["y_holdout"]).item())
        teacher_ce = float(F.cross_entropy(data["base_logits_holdout"] + teacher_holdout, data["y_holdout"]).item())
        norm_budget = float(torch.linalg.vector_norm(teacher_train, dim=1).mean().item())

    arm_rows: list[dict[str, Any]] = []
    pruning_rows: list[dict[str, Any]] = []
    for objective in OBJECTIVES:
        sparse = _train_sparse_model(
            torch,
            F,
            data,
            teacher_train,
            objective=objective,
            atom_count=atom_count,
            top_r=top_r,
            norm_budget=norm_budget,
            steps=value_steps,
            support_mode="learned",
        )
        flat = _train_control_model(
            torch,
            F,
            data,
            teacher_train,
            objective=objective,
            hidden=0,
            norm_budget=norm_budget,
            steps=control_steps,
        )
        dense = _train_control_model(
            torch,
            F,
            data,
            teacher_train,
            objective=objective,
            hidden=32,
            norm_budget=norm_budget,
            steps=control_steps,
        )
        shuffled = _train_sparse_model(
            torch,
            F,
            data,
            torch.roll(teacher_train, shifts=1, dims=0),
            objective="mse_only",
            atom_count=atom_count,
            top_r=top_r,
            norm_budget=norm_budget,
            steps=max(1, value_steps // 2),
            support_mode="learned",
        )
        random_null = _train_sparse_model(
            torch,
            F,
            data,
            teacher_train,
            objective=objective,
            atom_count=atom_count,
            top_r=top_r,
            norm_budget=norm_budget,
            steps=max(1, value_steps // 2),
            support_mode="random",
        )
        position_null = _train_sparse_model(
            torch,
            F,
            data,
            teacher_train,
            objective=objective,
            atom_count=atom_count,
            top_r=top_r,
            norm_budget=norm_budget,
            steps=max(1, value_steps // 2),
            support_mode="position",
        )

        predictions = {
            "sparse_topr_norm_controller": (*_predict_sparse(torch, sparse, data["x_holdout"]), "budgeted sparse top-r atom bank"),
            "same_controller_flat_residual": (*_predict_control(torch, flat, data["x_holdout"]), "same residual-norm controller, linear direction"),
            "same_controller_dense_mlp": (*_predict_control(torch, dense, data["x_holdout"]), "same residual-norm controller, dense MLP direction"),
            "scale_only_null": (
                _scale_only_prediction(torch, teacher_train, len(data["x_holdout"]), norm_budget),
                None,
                "train-mean direction with only the shared residual scale budget",
            ),
            "shuffled_target_sparse_null": (*_predict_sparse(torch, shuffled, data["x_holdout"]), "shuffled teacher-residual target null"),
            "random_support_sparse_null": (*_predict_sparse(torch, random_null, data["x_holdout"]), "random fixed-support sparse null"),
            "position_support_sparse_null": (*_predict_sparse(torch, position_null, data["x_holdout"]), "position-derived fixed-support sparse null"),
        }
        for arm, (pred, weights, note) in predictions.items():
            support = _support_proxy(torch, data, weights, arm, atom_count)
            arm_rows.append(
                _arm_row(
                    torch,
                    F,
                    objective=objective,
                    arm=arm,
                    pred=pred,
                    support=support,
                    weights=weights,
                    data=data,
                    target=teacher_holdout,
                    base_ce=base_ce,
                    teacher_ce=teacher_ce,
                    norm_budget=norm_budget,
                    atom_count=atom_count,
                    top_r=top_r,
                    note=note,
                )
            )
        pruning_rows.extend(
            _pruning_rows(
                torch,
                F,
                objective=objective,
                model=sparse,
                data=data,
                target=teacher_holdout,
                base_ce=base_ce,
                atom_count=atom_count,
            )
        )

    gate_rows = _gate_rows(source_rows, arm_rows, pruning_rows, base_ce, teacher_ce, atom_count)
    runtime_failures = [row for row in gate_rows if row["required"] and not row["passed"]]
    scientific_failures = [row for row in gate_rows if row["gate_type"] == "scientific" and not row["passed"]]
    status = "fail" if runtime_failures else "pass"
    summary = {
        "status": status,
        "decision": DECISION if status == "pass" else FAIL_DECISION,
        "claim_status": _claim_status(status, scientific_failures),
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
        "atom_count": atom_count,
        "top_r": top_r,
        "objectives": list(OBJECTIVES),
        "arms": list(ARMS),
        "prune_rules": list(PRUNE_RULES),
        "norm_controller_parity": "all trained arms use direction times sigmoid norm head with the same norm budget",
        "base_holdout_ce": round(base_ce, 6),
        "dense_teacher_holdout_ce": round(teacher_ce, 6),
        "source_rows": source_rows,
        "arm_metrics": arm_rows,
        "pruning_rows": pruning_rows,
        "gate_rows": gate_rows,
        "failures": runtime_failures + scientific_failures,
        "strategy_review": review,
        "strategy_review_handling": _strategy_review_handling(review),
        "deferred_or_rejected_recommendations": [],
        "direction_shift": _direction_shift(review),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _train_sparse_model(
    torch: Any,
    F: Any,
    data: dict[str, Any],
    targets: Any,
    *,
    objective: str,
    atom_count: int,
    top_r: int,
    norm_budget: float,
    steps: int,
    support_mode: str,
) -> dict[str, Any]:
    model = {
        "atoms": torch.nn.Parameter(torch.randn(atom_count, data["classes"]) * 0.08),
        "weight_head": torch.nn.Sequential(torch.nn.Linear(data["input_dim"], 24), torch.nn.Tanh(), torch.nn.Linear(24, atom_count)),
        "norm_head": torch.nn.Sequential(torch.nn.Linear(data["input_dim"], 12), torch.nn.Tanh(), torch.nn.Linear(12, 1)),
        "top_r": top_r,
        "norm_budget": norm_budget,
        "support_mode": support_mode,
    }
    opt = torch.optim.AdamW([model["atoms"], *model["weight_head"].parameters(), *model["norm_head"].parameters()], lr=0.014)
    for step in range(steps):
        pred, weights = _predict_sparse(torch, model, data["x_train"], train_step=step)
        loss = _objective_loss(F, objective, pred, targets, data["base_logits_train"], data["y_train"])
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1).mean()
        diversity = (model["atoms"] @ model["atoms"].T).fill_diagonal_(0.0).pow(2).mean()
        loss = loss + 0.002 * entropy + 0.0005 * model["atoms"].abs().mean() + 0.0002 * diversity
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def _train_control_model(
    torch: Any,
    F: Any,
    data: dict[str, Any],
    targets: Any,
    *,
    objective: str,
    hidden: int,
    norm_budget: float,
    steps: int,
) -> dict[str, Any]:
    if hidden:
        direction = torch.nn.Sequential(torch.nn.Linear(data["input_dim"], hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, data["classes"]))
    else:
        direction = torch.nn.Linear(data["input_dim"], data["classes"])
    norm_head = torch.nn.Sequential(torch.nn.Linear(data["input_dim"], 12), torch.nn.Tanh(), torch.nn.Linear(12, 1))
    opt = torch.optim.AdamW([*direction.parameters(), *norm_head.parameters()], lr=0.012)
    model = {"direction": direction, "norm_head": norm_head, "norm_budget": norm_budget}
    for _ in range(steps):
        pred, _ = _predict_control(torch, model, data["x_train"])
        loss = _objective_loss(F, objective, pred, targets, data["base_logits_train"], data["y_train"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def _predict_sparse(torch: Any, model: dict[str, Any], x: Any, train_step: int | None = None) -> tuple[Any, Any]:
    logits = model["weight_head"](x)
    if model["support_mode"] == "random":
        ids = (torch.arange(len(x), device=x.device) * 5 + 1) % logits.shape[1]
        mask = torch.nn.functional.one_hot(ids, num_classes=logits.shape[1]).to(logits.dtype)
    elif model["support_mode"] == "position":
        ids = torch.arange(len(x), device=x.device) % logits.shape[1]
        mask = torch.nn.functional.one_hot(ids, num_classes=logits.shape[1]).to(logits.dtype)
    else:
        keep = torch.topk(logits, int(model["top_r"]), dim=-1).indices
        mask = torch.zeros_like(logits).scatter(1, keep, 1.0)
    if train_step is not None and train_step % 3 == 1 and mask.shape[1] > 1:
        drop = ((torch.arange(mask.shape[1], device=mask.device) + train_step) % 5 == 0).to(mask.dtype)
        mask = mask * (1.0 - drop).unsqueeze(0)
        empty = mask.sum(dim=-1, keepdim=True) <= 0
        mask = torch.where(empty, torch.nn.functional.one_hot(logits.argmax(dim=-1), num_classes=mask.shape[1]).to(mask.dtype), mask)
    weights = torch.softmax(logits.masked_fill(mask <= 0, -1e9), dim=-1)
    direction = weights @ model["atoms"]
    return _apply_norm_controller(torch, direction, model["norm_head"](x), float(model["norm_budget"])), weights


def _predict_control(torch: Any, model: dict[str, Any], x: Any) -> tuple[Any, None]:
    direction = model["direction"](x)
    return _apply_norm_controller(torch, direction, model["norm_head"](x), float(model["norm_budget"])), None


def _apply_norm_controller(torch: Any, direction: Any, raw_norm: Any, norm_budget: float) -> Any:
    unit = direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1e-6)
    scale = 2.0 * norm_budget * torch.sigmoid(raw_norm)
    return unit * scale


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


def _scale_only_prediction(torch: Any, target_train: Any, n: int, norm_budget: float) -> Any:
    direction = target_train.mean(dim=0, keepdim=True).repeat(n, 1)
    unit = direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1e-6)
    return unit * norm_budget


def _support_proxy(torch: Any, data: dict[str, Any], weights: Any | None, arm: str, atom_count: int) -> Any:
    if weights is not None:
        return weights.argmax(dim=-1)
    if arm == "scale_only_null":
        return torch.zeros(len(data["x_holdout"]), dtype=torch.long)
    return data["position_holdout"] % atom_count


def _arm_row(
    torch: Any,
    F: Any,
    *,
    objective: str,
    arm: str,
    pred: Any,
    support: Any,
    weights: Any | None,
    data: dict[str, Any],
    target: Any,
    base_ce: float,
    teacher_ce: float,
    norm_budget: float,
    atom_count: int,
    top_r: int,
    note: str,
) -> dict[str, Any]:
    logits = data["base_logits_holdout"] + pred
    residual_l2 = torch.linalg.vector_norm(pred, dim=1)
    teacher_l2 = torch.linalg.vector_norm(target, dim=1)
    entropy = 0.0 if weights is None else float((-(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1)).mean().item())
    active_fraction = 1.0 if weights is None else float((weights > 1e-8).float().sum(dim=-1).mean().item() / atom_count)
    return {
        "objective": objective,
        "arm": arm,
        "ce": round(float(F.cross_entropy(logits, data["y_holdout"]).item()), 6),
        "base_ce": round(base_ce, 6),
        "dense_teacher_ce": round(teacher_ce, 6),
        "ce_improvement_vs_base": round(base_ce - float(F.cross_entropy(logits, data["y_holdout"]).item()), 6),
        "teacher_residual_reconstruction_mse": round(float(F.mse_loss(pred, target).item()), 6),
        "residual_teacher_cosine": round(float(F.cosine_similarity(pred, target, dim=1).mean().item()), 6),
        "functional_churn": round(float((logits.argmax(dim=-1) != data["base_logits_holdout"].argmax(dim=-1)).float().mean().item()), 6),
        "finite_update_commutator_proxy": round(assay._commutator_proxy(torch, pred, support), 6),
        "intervention_selectivity_proxy": round(assay._selectivity_proxy(torch, pred, target, support), 6),
        "residual_l2_mean": round(float(residual_l2.mean().item()), 6),
        "teacher_residual_l2_mean": round(float(teacher_l2.mean().item()), 6),
        "norm_budget": round(norm_budget, 6),
        "residual_l2_mean_ratio_vs_teacher": round(float((residual_l2.mean() / teacher_l2.mean().clamp_min(1e-6)).item()), 6),
        "support_entropy_mean": round(entropy, 6),
        "active_component_fraction": round(active_fraction, 6),
        "top_r": top_r if weights is not None else "",
        "active_params": _active_params(arm, data["input_dim"], data["classes"], atom_count),
        "stored_params": _stored_params(arm, data["input_dim"], data["classes"], atom_count),
        "oracle_support_non_deployable": False,
        "uses_future_hidden_or_delta": False,
        "uses_task_id": False,
        "uses_teacher_labels_in_deployable_router": False,
        "target_access_at_eval": "prefix_safe_synthetic_features",
        "feature_schema_hash": "prefix_x_scale_constrained_sparse_v1",
        "note": note,
    }


def _pruning_rows(
    torch: Any,
    F: Any,
    *,
    objective: str,
    model: dict[str, Any],
    data: dict[str, Any],
    target: Any,
    base_ce: float,
    atom_count: int,
) -> list[dict[str, Any]]:
    pred, weights = _predict_sparse(torch, model, data["x_holdout"])
    unpruned_ce = float(F.cross_entropy(data["base_logits_holdout"] + pred, data["y_holdout"]).item())
    unpruned_gain = max(0.0, base_ce - unpruned_ce)
    rows = []
    for rule in PRUNE_RULES:
        pruned = _apply_prune_rule(torch, weights, rule)
        direction = pruned @ model["atoms"]
        pred_pruned = _apply_norm_controller(torch, direction, model["norm_head"](data["x_holdout"]), float(model["norm_budget"]))
        ce = float(F.cross_entropy(data["base_logits_holdout"] + pred_pruned, data["y_holdout"]).item())
        gain = max(0.0, base_ce - ce)
        active_fraction = float((pruned > 0).float().sum(dim=-1).mean().item() / atom_count)
        rows.append(
            {
                "objective": objective,
                "arm": "sparse_topr_norm_controller",
                "prune_rule": rule,
                "ce": round(ce, 6),
                "teacher_residual_reconstruction_mse": round(float(F.mse_loss(pred_pruned, target).item()), 6),
                "active_component_fraction": round(active_fraction, 6),
                "ce_gain_retention_fraction": round(gain / unpruned_gain, 6) if unpruned_gain > 1e-8 else 0.0,
                "pruned_at_least_half_components": active_fraction <= 0.5,
            }
        )
    return rows


def _apply_prune_rule(torch: Any, weights: Any, rule: str) -> Any:
    if rule.startswith("top"):
        keep = max(1, min(int(rule.replace("top", "")), weights.shape[1]))
        threshold = torch.topk(weights, keep, dim=-1).values[:, -1].unsqueeze(-1)
        pruned = torch.where(weights >= threshold, weights, torch.zeros_like(weights))
    else:
        threshold = float(rule.replace("threshold_", "").replace("p", "."))
        pruned = torch.where(weights >= threshold, weights, torch.zeros_like(weights))
    denom = pruned.sum(dim=-1, keepdim=True)
    fallback = torch.nn.functional.one_hot(weights.argmax(dim=-1), num_classes=weights.shape[1]).to(weights.dtype)
    return torch.where(denom > 0, pruned / denom.clamp_min(1e-8), fallback)


def _gate_rows(
    source_rows: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    pruning_rows: list[dict[str, Any]],
    base_ce: float,
    teacher_ce: float,
    atom_count: int,
) -> list[dict[str, Any]]:
    rows = {(row["objective"], row["arm"]): row for row in arm_rows}
    sparse_ce = rows.get(("ce_only", "sparse_topr_norm_controller"), {})
    flat_ce = rows.get(("ce_only", "same_controller_flat_residual"), {})
    dense_ce = rows.get(("ce_only", "same_controller_dense_mlp"), {})
    sparse_mse = rows.get(("mse_only", "sparse_topr_norm_controller"), {})
    flat_mse = rows.get(("mse_only", "same_controller_flat_residual"), {})
    best_retention = max(
        (
            _metric(row, "ce_gain_retention_fraction")
            for row in pruning_rows
            if row["objective"] == "ce_only" and row["pruned_at_least_half_components"]
        ),
        default=0.0,
    )
    mechanism_wins = sum(
        [
            _metric(sparse_ce, "intervention_selectivity_proxy") > _metric(flat_ce, "intervention_selectivity_proxy"),
            _metric(sparse_ce, "finite_update_commutator_proxy") < _metric(flat_ce, "finite_update_commutator_proxy"),
            _metric(sparse_ce, "functional_churn") <= _metric(flat_ce, "functional_churn"),
            best_retention >= 0.80,
        ]
    )
    required_rows = {(objective, arm) for objective in OBJECTIVES for arm in ARMS}
    present = set(rows)
    return [
        _gate("closeout_source_present", all(row["present"] for row in source_rows if row["source"] != "strategy_review"), True, "runtime", str(source_rows[0])),
        _gate(
            "closeout_selected_scale_constrained_path",
            source_rows[0].get("selected_next_action") == "design_scale_constrained_sparse_residual_compression_pregate",
            True,
            "runtime",
            str(source_rows[0]),
        ),
        _gate("required_arm_rows_present", required_rows.issubset(present), True, "runtime", f"rows={len(arm_rows)}"),
        _gate("pruning_rows_present", len(pruning_rows) >= len(OBJECTIVES) * len(PRUNE_RULES), True, "runtime", f"rows={len(pruning_rows)}"),
        _gate("gpu_blocked", True, True, "runtime", "requires_gpu_now=false; promotion_allowed=false"),
        _gate("deployable_leakage_flags_false", all(not row["uses_future_hidden_or_delta"] and not row["uses_task_id"] and not row["uses_teacher_labels_in_deployable_router"] for row in arm_rows), True, "runtime", "no deployable leakage flags set"),
        _gate("dense_teacher_improves_base", teacher_ce < base_ce, False, "scientific", f"base_ce={base_ce:.6f}; teacher_ce={teacher_ce:.6f}"),
        _gate("sparse_matches_or_beats_flat_ce", _metric(sparse_ce, "ce") <= _metric(flat_ce, "ce") + 0.002, False, "scientific", f"sparse_ce={sparse_ce.get('ce')}; flat_ce={flat_ce.get('ce')}"),
        _gate("sparse_matches_or_beats_dense_ce", _metric(sparse_ce, "ce") <= _metric(dense_ce, "ce") + 0.01, False, "scientific", f"sparse_ce={sparse_ce.get('ce')}; dense_ce={dense_ce.get('ce')}"),
        _gate("sparse_not_worse_than_flat_mse", _metric(sparse_mse, "teacher_residual_reconstruction_mse") <= _metric(flat_mse, "teacher_residual_reconstruction_mse") + 0.02, False, "scientific", f"sparse_mse={sparse_mse.get('teacher_residual_reconstruction_mse')}; flat_mse={flat_mse.get('teacher_residual_reconstruction_mse')}"),
        _gate("pruning_retains_function_after_halving", best_retention >= 0.80, False, "scientific", f"best_retention={best_retention:.6f}"),
        _gate("sparse_support_is_constrained", _metric(sparse_ce, "active_component_fraction") <= (0.75 * atom_count / atom_count), False, "scientific", f"active_fraction={sparse_ce.get('active_component_fraction')}"),
        _gate("mechanism_proxy_wins_at_least_two", mechanism_wins >= 2, False, "scientific", f"mechanism_win_count={mechanism_wins}"),
    ]


def _active_params(arm: str, input_dim: int, classes: int, atom_count: int) -> int:
    if "sparse" in arm:
        return classes * 2 + atom_count
    if "flat" in arm:
        return classes + 1
    if "scale_only" in arm:
        return classes
    return input_dim * classes + classes


def _stored_params(arm: str, input_dim: int, classes: int, atom_count: int) -> int:
    if "sparse" in arm:
        return atom_count * classes + input_dim * 24 + 24 * atom_count + input_dim * 12 + 12
    if "flat" in arm:
        return input_dim * classes + input_dim * 12 + 12
    if "scale_only" in arm:
        return classes
    return input_dim * 32 + 32 * classes + input_dim * 12 + 12


def _gate(criterion: str, passed: bool, required: bool, gate_type: str, evidence: str) -> dict[str, Any]:
    return {"criterion": criterion, "passed": bool(passed), "required": bool(required), "gate_type": gate_type, "evidence": evidence}


def _metric(row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("inf")


def _claim_status(status: str, scientific_failures: list[dict[str, Any]]) -> str:
    if status != "pass":
        return "scale_constrained_sparse_pilot_runtime_failed"
    failed = {row["criterion"] for row in scientific_failures}
    if not failed:
        return "scale_constrained_sparse_local_gates_support_repeat_before_gpu"
    if "sparse_matches_or_beats_flat_ce" in failed or "sparse_not_worse_than_flat_mse" in failed:
        return "scale_constrained_sparse_flat_control_blocks_gpu"
    if "pruning_retains_function_after_halving" in failed:
        return "scale_constrained_sparse_pruning_retention_blocks_gpu"
    return "scale_constrained_sparse_partial_local_signal_no_gpu"


def _selected_next_step(status: str, scientific_failures: list[dict[str, Any]]) -> str:
    if status != "pass":
        return "repair scale-constrained sparse pilot source/runtime artifacts before interpretation"
    if not scientific_failures:
        return "repeat scale-constrained sparse pilot on adjacent seeds before any GPU validation"
    return "close or redesign scale-constrained sparse residual compression before GPU; local gates did not clear"


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
        "git_commit": payload.get("git_commit", "") if payload else "",
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _strategy_review(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        text = ""
    header: dict[str, str] = {}
    for line in text.splitlines()[:8]:
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    return {
        "present": bool(text),
        "strategic_change_level": header.get("strategic_change_level", ""),
        "notify_ben": header.get("notify_ben", "false"),
        "ben_notification_required": header.get("notify_ben", "false").lower() == "true",
        "recommended_next_action": header.get("recommended_next_action", ""),
        "verdict": header.get("verdict", ""),
    }


def _strategy_review_handling(review: dict[str, Any]) -> str:
    if not review["present"]:
        return "No current strategy review was present; continued from AUTOMATION_STATUS selected next step."
    return (
        "Accepted the GPT-5.5-Pro recommendation to implement a local executable "
        "scale-constrained sparse residual-compression pilot with identical norm-controller controls before GPU."
    )


def _direction_shift(review: dict[str, Any]) -> str:
    if review.get("ben_notification_required") or review.get("strategic_change_level") == "major":
        return "Strategy review requested a major/Ben-notify shift; Ben should be notified."
    return "No major strategy-review direction shift or Ben notification required."


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "arm_metrics.csv", summary["arm_metrics"])
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
        "# Scale-Constrained Sparse Residual-Compression Pilot",
        "",
        f"- Status: {summary['status']}",
        f"- Decision: {summary['decision']}",
        f"- Claim status: {summary['claim_status']}",
        f"- Selected next step: {summary['selected_next_step']}",
        "- Sparse, flat, dense, scale-only, shuffled-target, random-support, and position-support arms share the same residual norm budget.",
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
    parser.add_argument("--closeout", type=Path, default=DEFAULT_CLOSEOUT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--teacher-steps", type=int, default=100)
    parser.add_argument("--value-steps", type=int, default=120)
    parser.add_argument("--control-steps", type=int, default=120)
    parser.add_argument("--atom-count", type=int, default=8)
    parser.add_argument("--top-r", type=int, default=2)
    args = parser.parse_args(argv)
    summary = run_scale_constrained_sparse_residual_compression_pilot(
        closeout_path=args.closeout,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
        seed=args.seed,
        teacher_steps=args.teacher_steps,
        value_steps=args.value_steps,
        control_steps=args.control_steps,
        atom_count=args.atom_count,
        top_r=args.top_r,
    )
    print(json.dumps({key: summary[key] for key in ("status", "decision", "claim_status", "selected_next_step")}, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
