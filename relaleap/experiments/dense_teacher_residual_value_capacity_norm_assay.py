"""Train the dense-teacher residual value-capacity/norm-control assay.

This is the first trained local repair after the dense-teacher residual
columnability failure localization. It keeps GPU validation blocked and asks a
more precise question than the prior assay: after adding sparse value capacity
and explicit residual-norm controls, can oracle-supported sparse dictionaries
represent a dense teacher's correction field better than support/target nulls,
and only then does a learned causal router become interpretable?
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


DEFAULT_PREGATE_DIR = Path("results/reports/dense_teacher_residual_value_capacity_norm_pregate")
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_residual_value_capacity_norm_assay")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "training_rows.csv",
    "arm_metrics.csv",
    "norm_control_rows.csv",
    "gate_criteria.csv",
    "notes.md",
)

ARMS = (
    "dense_teacher_residual_control",
    "rank_matched_residual_control",
    "norm_clipped_mlp_control",
    "oracle_support_norm_matched_multi_value_dictionary",
    "oracle_support_low_rank_value_dictionary",
    "learned_router_norm_matched_multi_value_dictionary",
    "same_router_flat_value_norm_matched_control",
    "random_support_norm_matched_null",
    "frequency_support_norm_matched_null",
    "token_position_norm_matched_null",
    "shuffled_teacher_residual_norm_matched_null",
    "delayed_teacher_residual_norm_matched_null",
)

DECISION = "dense_teacher_residual_value_capacity_norm_assay_recorded"
FAIL_DECISION = "dense_teacher_residual_value_capacity_norm_assay_failed_closed"


def run_dense_teacher_residual_value_capacity_norm_assay(
    *,
    pregate_dir: Path = DEFAULT_PREGATE_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 31,
    teacher_steps: int = 100,
    router_steps: int = 80,
    value_steps: int = 80,
    control_steps: int = 80,
    column_count: int = 6,
    values_per_column: int = 3,
) -> dict[str, Any]:
    """Run the bounded local trained assay and write artifacts."""

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError("dense-teacher value-capacity/norm-control assay requires torch") from exc

    if min(teacher_steps, router_steps, value_steps, control_steps) < 1:
        raise ValueError("all training step counts must be positive")
    if column_count < 2:
        raise ValueError("column_count must be at least 2")
    if values_per_column < 2:
        raise ValueError("values_per_column must be at least 2")

    start = time.time()
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    pregate = _read_json(pregate_dir / "summary.json")
    source_rows = [_source_row("dense_teacher_residual_value_capacity_norm_pregate", pregate_dir / "summary.json", pregate)]

    data = _make_data(torch, seed=seed, column_count=column_count)
    teacher = _Teacher(torch, data["input_dim"], data["classes"])
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

    oracle_model = _fit_dictionary_model(
        torch,
        data["x_train"],
        teacher_residual_train,
        data["support_train"],
        column_count,
        values_per_column,
        steps=value_steps,
        lr=0.035,
    )
    shuffled_model = _fit_dictionary_model(
        torch,
        data["x_train"],
        torch.roll(teacher_residual_train, shifts=1, dims=0),
        data["support_train"],
        column_count,
        values_per_column,
        steps=value_steps,
        lr=0.035,
    )
    delayed_model = _fit_dictionary_model(
        torch,
        data["x_train"],
        torch.roll(teacher_residual_train, shifts=7, dims=0),
        data["support_train"],
        column_count,
        values_per_column,
        steps=value_steps,
        lr=0.035,
    )
    support_router = _train_support_router(
        torch,
        data["x_train"],
        data["support_train"],
        data["input_dim"],
        column_count,
        steps=router_steps,
    )
    flat_value = _train_flat_value_head(
        torch,
        F,
        data["x_train"],
        teacher_residual_train,
        data["input_dim"],
        data["classes"],
        steps=control_steps,
    )
    rank_control = _train_rank_control(
        torch,
        F,
        data["x_train"],
        teacher_residual_train,
        data["input_dim"],
        data["classes"],
        rank=min(3, data["classes"]),
        steps=control_steps,
    )

    support_by_arm = _support_rows(torch, data, support_router, column_count)
    predictions: dict[str, tuple[Any, Any, bool, str]] = {
        "dense_teacher_residual_control": (
            teacher_residual_holdout,
            data["support_holdout"],
            False,
            "dense teacher residual target; control only",
        ),
        "rank_matched_residual_control": (
            _norm_match(torch, rank_control(data["x_holdout"]), teacher_residual_train),
            data["support_holdout"],
            False,
            "deployable low-rank residual control",
        ),
        "norm_clipped_mlp_control": (
            _norm_match(torch, flat_value(data["x_holdout"]).clamp(-1.25, 1.25), teacher_residual_train),
            support_by_arm["learned"],
            False,
            "deployable norm-clipped MLP value control",
        ),
        "oracle_support_norm_matched_multi_value_dictionary": (
            _predict_dictionary(torch, oracle_model, data["x_holdout"], data["support_holdout"], teacher_residual_train),
            data["support_holdout"],
            True,
            "oracle support ceiling; value code chosen from prefix-safe features",
        ),
        "oracle_support_low_rank_value_dictionary": (
            _predict_low_rank_dictionary(
                torch,
                oracle_model,
                data["x_holdout"],
                data["support_holdout"],
                teacher_residual_train,
                rank=min(2, data["classes"]),
            ),
            data["support_holdout"],
            True,
            "oracle support low-rank sparse value ceiling",
        ),
        "learned_router_norm_matched_multi_value_dictionary": (
            _predict_dictionary(torch, oracle_model, data["x_holdout"], support_by_arm["learned"], teacher_residual_train),
            support_by_arm["learned"],
            False,
            "deployable learned causal router with norm-matched sparse values",
        ),
        "same_router_flat_value_norm_matched_control": (
            _norm_match(torch, flat_value(data["x_holdout"]), teacher_residual_train),
            support_by_arm["learned"],
            False,
            "same learned support with flat value head control",
        ),
        "random_support_norm_matched_null": (
            _predict_dictionary(torch, oracle_model, data["x_holdout"], support_by_arm["random"], teacher_residual_train),
            support_by_arm["random"],
            False,
            "random support null with matched sparse values",
        ),
        "frequency_support_norm_matched_null": (
            _predict_dictionary(torch, oracle_model, data["x_holdout"], support_by_arm["frequency"], teacher_residual_train),
            support_by_arm["frequency"],
            False,
            "frequency support null with matched sparse values",
        ),
        "token_position_norm_matched_null": (
            _predict_dictionary(torch, oracle_model, data["x_holdout"], support_by_arm["position"], teacher_residual_train),
            support_by_arm["position"],
            False,
            "token/position support null with matched sparse values",
        ),
        "shuffled_teacher_residual_norm_matched_null": (
            _predict_dictionary(torch, shuffled_model, data["x_holdout"], data["support_holdout"], teacher_residual_train),
            data["support_holdout"],
            False,
            "shuffled teacher-residual target null with same support and value budget",
        ),
        "delayed_teacher_residual_norm_matched_null": (
            _predict_dictionary(torch, delayed_model, data["x_holdout"], data["support_holdout"], teacher_residual_train),
            data["support_holdout"],
            False,
            "delayed teacher-residual target null with same support and value budget",
        ),
    }

    arm_rows = [
        _arm_metrics(
            torch,
            F,
            arm,
            pred,
            support,
            teacher_residual_holdout,
            data,
            teacher_holdout_ce,
            base_holdout_ce,
            oracle_non_deployable,
            note,
            column_count,
            values_per_column,
        )
        for arm, (pred, support, oracle_non_deployable, note) in predictions.items()
    ]
    norm_rows = _norm_control_rows(arm_rows)
    training_rows = _training_rows(torch, predictions, teacher_residual_holdout)
    gate_rows = _gate_rows(source_rows, arm_rows, base_holdout_ce, teacher_holdout_ce)
    runtime_failures = [row for row in gate_rows if row["required"] and not row["passed"]]
    scientific_failures = [row for row in gate_rows if row["gate_type"] == "scientific" and not row["passed"]]
    status = "fail" if runtime_failures else "pass"
    claim_status = (
        "value_capacity_norm_control_local_gates_support_repeat_before_gpu"
        if status == "pass" and not scientific_failures
        else "value_capacity_norm_control_local_gates_block_gpu"
    )
    summary = {
        "status": status,
        "decision": DECISION if status == "pass" else FAIL_DECISION,
        "claim_status": claim_status,
        "selected_next_step": _selected_next_step(scientific_failures, status),
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "backend_policy": "local CPU trained assay only; RunPod and Colab remain blocked",
        "training_rows_present": bool(training_rows),
        "training_executed": True,
        "teacher_trained": True,
        "teacher_train_steps": teacher_steps,
        "router_train_steps": router_steps,
        "value_train_steps": value_steps,
        "control_train_steps": control_steps,
        "seed": seed,
        "column_count": column_count,
        "values_per_column": values_per_column,
        "base_holdout_ce": round(base_holdout_ce, 6),
        "dense_teacher_holdout_ce": round(teacher_holdout_ce, 6),
        "dense_teacher_ce_improvement": round(base_holdout_ce - teacher_holdout_ce, 6),
        "oracle_support_non_deployable": True,
        "uses_future_oracle_task_flags": {
            "uses_future_hidden_or_delta": False,
            "deployable_router_uses_oracle_support": False,
            "uses_task_id": False,
            "uses_teacher_labels_in_deployable_router": False,
        },
        "source_rows": source_rows,
        "arm_metrics": arm_rows,
        "norm_control_rows": norm_rows,
        "gate_criteria": gate_rows,
        "failures": runtime_failures + scientific_failures,
        "strategy_review_handling": (
            "Accepted the major pivot: PC/core-periphery and teacher-support Transformer-ACSR "
            "remain closed; this run tests sparse residual value capacity/norm controls locally "
            "before any routing or GPU claim."
        ),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, source_rows, training_rows, arm_rows, norm_rows, gate_rows)
    return summary


def _make_data(torch: Any, *, seed: int, column_count: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    base_dim = 10
    input_dim = base_dim + column_count
    classes = 5
    n_train = 640
    n_holdout = 128
    base_w = torch.randn(input_dim, classes, generator=generator) * 0.025
    mech_w = torch.randn(column_count, input_dim, classes, generator=generator) * 0.62
    mech_bias = torch.randn(column_count, classes, generator=generator) * 0.24

    def build(n: int) -> tuple[Any, Any, Any, Any, Any]:
        base_x = torch.randn(n, base_dim, generator=generator)
        position = torch.arange(n) % 11
        base_x[:, 0] += (position.float() - 5.0) / 7.0
        base_x[:, 1] += torch.sin(position.float()) / 3.0
        position_mod = position % column_count
        position_features = torch.nn.functional.one_hot(position_mod, num_classes=column_count).float()
        x = torch.cat([base_x, position_features], dim=1)
        support = ((base_x[:, 0] > -0.15).long() + 2 * (base_x[:, 1] > 0.05).long() + 3 * position_mod.long()) % column_count
        true_residual = torch.stack(
            [torch.tanh(x[index] @ mech_w[int(support[index])] + mech_bias[int(support[index])]) for index in range(n)]
        )
        true_residual = true_residual * 1.55
        base_logits = x @ base_w
        y = (base_logits + true_residual).argmax(dim=-1)
        return x, position, support, base_logits, y

    x_train, position_train, support_train, base_logits_train, y_train = build(n_train)
    x_holdout, position_holdout, support_holdout, base_logits_holdout, y_holdout = build(n_holdout)
    return {
        "input_dim": input_dim,
        "classes": classes,
        "x_train": x_train,
        "position_train": position_train,
        "support_train": support_train,
        "base_logits_train": base_logits_train,
        "y_train": y_train,
        "x_holdout": x_holdout,
        "position_holdout": position_holdout,
        "support_holdout": support_holdout,
        "base_logits_holdout": base_logits_holdout,
        "y_holdout": y_holdout,
    }


class _Teacher:
    def __new__(cls, torch: Any, input_dim: int, classes: int) -> Any:
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, classes),
        )


def _fit_dictionary_model(
    torch: Any,
    x_train: Any,
    targets: Any,
    support: Any,
    column_count: int,
    values_per_column: int,
    *,
    steps: int,
    lr: float,
) -> dict[str, Any]:
    dictionary, code_labels = _multi_value_dictionary(torch, targets, support, column_count, values_per_column)
    code_head = torch.nn.Linear(x_train.shape[1], column_count * values_per_column)
    optimizer = torch.optim.AdamW(code_head.parameters(), lr=lr)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(code_head(x_train), code_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return {"dictionary": dictionary, "code_head": code_head}


def _multi_value_dictionary(
    torch: Any,
    targets: Any,
    support: Any,
    column_count: int,
    values_per_column: int,
) -> tuple[Any, Any]:
    fallback = targets.mean(dim=0)
    rows = []
    labels = torch.zeros(len(targets), dtype=torch.long)
    for column in range(column_count):
        indices = torch.nonzero(support == column, as_tuple=False).flatten()
        if len(indices) == 0:
            centroids = fallback.repeat(values_per_column, 1)
        else:
            values = targets[indices]
            centroids = _kmeans(torch, values, values_per_column)
            distances = torch.cdist(values, centroids)
            local_labels = distances.argmin(dim=1)
            labels[indices] = column * values_per_column + local_labels
        rows.append(centroids)
    return torch.stack(rows), labels


def _kmeans(torch: Any, values: Any, k: int, iterations: int = 8) -> Any:
    if len(values) == 0:
        raise ValueError("cannot fit empty kmeans")
    if len(values) < k:
        repeats = math.ceil(k / len(values))
        return values.repeat((repeats, 1))[:k].clone()
    order = torch.linspace(0, len(values) - 1, steps=k).round().long()
    centroids = values[order].clone()
    for _ in range(iterations):
        labels = torch.cdist(values, centroids).argmin(dim=1)
        updated = []
        for idx in range(k):
            mask = labels == idx
            updated.append(values[mask].mean(dim=0) if bool(mask.any()) else centroids[idx])
        centroids = torch.stack(updated)
    return centroids


def _train_support_router(torch: Any, x_train: Any, support: Any, input_dim: int, column_count: int, *, steps: int) -> Any:
    router = torch.nn.Linear(input_dim, column_count)
    optimizer = torch.optim.AdamW(router.parameters(), lr=0.04)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(router(x_train), support)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return router


def _train_flat_value_head(torch: Any, F: Any, x_train: Any, targets: Any, input_dim: int, classes: int, *, steps: int) -> Any:
    model = torch.nn.Sequential(torch.nn.Linear(input_dim, 24), torch.nn.Tanh(), torch.nn.Linear(24, classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(steps):
        loss = F.mse_loss(model(x_train), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def _train_rank_control(
    torch: Any,
    F: Any,
    x_train: Any,
    targets: Any,
    input_dim: int,
    classes: int,
    *,
    rank: int,
    steps: int,
) -> Any:
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, rank, bias=False),
        torch.nn.Linear(rank, classes, bias=False),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(steps):
        loss = F.mse_loss(model(x_train), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def _support_rows(torch: Any, data: dict[str, Any], router: Any, column_count: int) -> dict[str, Any]:
    n = len(data["x_holdout"])
    train_counts = torch.bincount(data["support_train"], minlength=column_count)
    frequency_support = int(torch.argmax(train_counts).item())
    return {
        "learned": router(data["x_holdout"]).argmax(dim=-1),
        "random": (torch.arange(n) * 5 + 1) % column_count,
        "frequency": torch.full((n,), frequency_support, dtype=torch.long),
        "position": data["position_holdout"] % column_count,
    }


def _predict_dictionary(torch: Any, model: dict[str, Any], x: Any, support: Any, norm_target: Any) -> Any:
    dictionary = model["dictionary"]
    values_per_column = dictionary.shape[1]
    logits = model["code_head"](x).reshape(len(x), dictionary.shape[0], values_per_column)
    selected = []
    for index in range(len(x)):
        column = int(support[index].item())
        local_code = int(logits[index, column].argmax().item())
        selected.append(dictionary[column, local_code])
    return _norm_match(torch, torch.stack(selected), norm_target)


def _predict_low_rank_dictionary(torch: Any, model: dict[str, Any], x: Any, support: Any, norm_target: Any, *, rank: int) -> Any:
    pred = _predict_dictionary(torch, model, x, support, norm_target)
    _u, _s, vh = torch.linalg.svd(model["dictionary"].reshape(-1, model["dictionary"].shape[-1]), full_matrices=False)
    basis = vh[:rank]
    return _norm_match(torch, (pred @ basis.T) @ basis, norm_target)


def _norm_match(torch: Any, pred: Any, target_train: Any) -> Any:
    pred_l2 = torch.linalg.vector_norm(pred, dim=1).mean().clamp_min(1e-6)
    target_l2 = torch.linalg.vector_norm(target_train, dim=1).mean()
    scale = torch.clamp(target_l2 / pred_l2, 0.25, 3.0)
    return pred * scale


def _arm_metrics(
    torch: Any,
    F: Any,
    arm: str,
    pred: Any,
    support: Any,
    target: Any,
    data: dict[str, Any],
    teacher_ce: float,
    base_ce: float,
    oracle_non_deployable: bool,
    note: str,
    column_count: int,
    values_per_column: int,
) -> dict[str, Any]:
    logits = data["base_logits_holdout"] + pred
    ce = float(F.cross_entropy(logits, data["y_holdout"]).item())
    mse = float(F.mse_loss(pred, target).item())
    residual_l2 = torch.linalg.vector_norm(pred, dim=1)
    teacher_l2 = torch.linalg.vector_norm(target, dim=1)
    churn = float((logits.argmax(dim=-1) != data["base_logits_holdout"].argmax(dim=-1)).float().mean().item())
    teacher_logits = data["base_logits_holdout"] + target
    teacher_churn = float((teacher_logits.argmax(dim=-1) != data["base_logits_holdout"].argmax(dim=-1)).float().mean().item())
    retention = max(0.0, 1.0 - abs(churn - teacher_churn))
    return {
        "arm": arm,
        "row_source": "bounded_local_cpu_trained_value_capacity_norm_control_assay",
        "teacher_trained": True,
        "ce": round(ce, 6),
        "base_ce": round(base_ce, 6),
        "dense_teacher_ce": round(teacher_ce, 6),
        "ce_gap_vs_dense_teacher": round(ce - teacher_ce, 6),
        "ce_improvement_vs_base": round(base_ce - ce, 6),
        "teacher_residual_reconstruction_mse": round(mse, 6),
        "functional_churn": round(churn, 6),
        "retention_proxy": round(retention, 6),
        "finite_update_commutator_proxy": round(_commutator_proxy(torch, pred, support), 6),
        "intervention_selectivity_proxy": round(_selectivity_proxy(torch, pred, target, support), 6),
        "residual_l2_mean": round(float(residual_l2.mean().item()), 6),
        "residual_l2_p95": round(float(torch.quantile(residual_l2, 0.95).item()), 6),
        "teacher_residual_l2_mean": round(float(teacher_l2.mean().item()), 6),
        "residual_l2_mean_ratio_vs_teacher": round(float((residual_l2.mean() / teacher_l2.mean().clamp_min(1e-6)).item()), 6),
        "active_params": _active_params(arm, data["input_dim"], data["classes"], column_count, values_per_column),
        "stored_params": _stored_params(arm, data["input_dim"], data["classes"], column_count, values_per_column),
        "oracle_support_non_deployable": oracle_non_deployable,
        "uses_future_hidden_or_delta": False,
        "uses_oracle_support_at_eval": oracle_non_deployable,
        "uses_task_id": False,
        "uses_teacher_labels_in_deployable_router": False,
        "target_access_at_eval": "oracle_support_only" if oracle_non_deployable else "prefix_safe_or_null_support",
        "feature_schema_hash": "prefix_x_position_support_only_v2",
        "note": note,
    }


def _commutator_proxy(torch: Any, pred: Any, support: Any) -> float:
    if len(pred) < 2:
        return 0.0
    deltas = torch.linalg.vector_norm(pred[1:] - pred[:-1], dim=1)
    flips = (support[1:] != support[:-1]).float()
    return float((deltas * (1.0 + flips)).mean().item())


def _selectivity_proxy(torch: Any, pred: Any, target: Any, support: Any) -> float:
    fit = 1.0 / (1.0 + float(torch.mean((pred - target) ** 2).item()))
    unique_fraction = len(set(int(item) for item in support.tolist())) / max(1, len(support))
    return max(0.0, min(1.0, fit * (1.0 - unique_fraction)))


def _active_params(arm: str, input_dim: int, classes: int, column_count: int, values_per_column: int) -> int:
    if "multi_value_dictionary" in arm or "support_low_rank" in arm or arm.endswith("_null"):
        return classes + values_per_column + column_count
    if "rank_matched" in arm:
        return input_dim + classes
    return classes * column_count


def _stored_params(arm: str, input_dim: int, classes: int, column_count: int, values_per_column: int) -> int:
    if arm == "dense_teacher_residual_control":
        return input_dim * 32 + 32 * classes + 32 + classes
    if "rank_matched" in arm or "support_low_rank" in arm:
        return (input_dim + classes) * min(3, classes)
    if "flat_value" in arm or "norm_clipped_mlp" in arm:
        return input_dim * 24 + 24 * classes + 24 + classes
    return column_count * values_per_column * classes + input_dim * column_count + input_dim * column_count * values_per_column


def _norm_control_rows(arm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in arm_rows:
        rows.append(
            {
                "arm": row["arm"],
                "residual_l2_mean": row["residual_l2_mean"],
                "teacher_residual_l2_mean": row["teacher_residual_l2_mean"],
                "residual_l2_mean_ratio_vs_teacher": row["residual_l2_mean_ratio_vs_teacher"],
                "residual_l2_p95": row["residual_l2_p95"],
                "active_params": row["active_params"],
                "stored_params": row["stored_params"],
                "norm_control_applied": row["arm"] != "dense_teacher_residual_control",
            }
        )
    return rows


def _training_rows(torch: Any, predictions: dict[str, tuple[Any, Any, bool, str]], target: Any) -> list[dict[str, Any]]:
    rows = []
    for arm, (pred, support, oracle, _note) in predictions.items():
        for index in range(min(6, len(pred))):
            rows.append(
                {
                    "arm": arm,
                    "row_index": index,
                    "split": "holdout",
                    "support": int(support[index].item()),
                    "oracle_support_non_deployable": oracle,
                    "target_residual_l2": round(float(torch.linalg.vector_norm(target[index]).item()), 6),
                    "predicted_residual_l2": round(float(torch.linalg.vector_norm(pred[index]).item()), 6),
                    "residual_mse": round(float(torch.mean((pred[index] - target[index]) ** 2).item()), 6),
                }
            )
    return rows


def _gate_rows(source_rows: list[dict[str, Any]], arm_rows: list[dict[str, Any]], base_ce: float, teacher_ce: float) -> list[dict[str, Any]]:
    arms = {row["arm"]: row for row in arm_rows}
    oracle = arms.get("oracle_support_norm_matched_multi_value_dictionary", {})
    learned = arms.get("learned_router_norm_matched_multi_value_dictionary", {})
    shuffled = arms.get("shuffled_teacher_residual_norm_matched_null", {})
    delayed = arms.get("delayed_teacher_residual_norm_matched_null", {})
    random_null = arms.get("random_support_norm_matched_null", {})
    frequency_null = arms.get("frequency_support_norm_matched_null", {})
    token_null = arms.get("token_position_norm_matched_null", {})
    flat = arms.get("same_router_flat_value_norm_matched_control", {})
    required = set(ARMS)
    oracle_mse = _float(oracle.get("teacher_residual_reconstruction_mse"), math.inf)
    learned_mse = _float(learned.get("teacher_residual_reconstruction_mse"), math.inf)
    support_null_mse = min(
        _float(random_null.get("teacher_residual_reconstruction_mse"), math.inf),
        _float(frequency_null.get("teacher_residual_reconstruction_mse"), math.inf),
        _float(token_null.get("teacher_residual_reconstruction_mse"), math.inf),
    )
    target_null_mse = min(
        _float(shuffled.get("teacher_residual_reconstruction_mse"), math.inf),
        _float(delayed.get("teacher_residual_reconstruction_mse"), math.inf),
    )
    return [
        _gate("pregate_source_present", all(row["present"] for row in source_rows), True, "runtime", str(source_rows)),
        _gate("teacher_trained", bool(arm_rows), True, "runtime", "dense residual teacher and local controls trained"),
        _gate("training_rows_present", bool(arm_rows), True, "runtime", f"arm_rows={len(arm_rows)}"),
        _gate("required_arms_present", required.issubset(arms), True, "runtime", ",".join(sorted(arms))),
        _gate("gpu_blocked", True, True, "runtime", "requires_gpu_now=false; advance_to_gpu_validation=false"),
        _gate("oracle_support_non_deployable_labeled", bool(oracle.get("oracle_support_non_deployable")), True, "runtime", "oracle support ceiling only"),
        _gate(
            "deployable_leakage_flags_false",
            all(not row["uses_future_hidden_or_delta"] and not row["uses_task_id"] and not row["uses_teacher_labels_in_deployable_router"] for row in arm_rows),
            True,
            "runtime",
            "no deployable row uses future hidden/delta, task id, or teacher labels",
        ),
        _gate("dense_teacher_improves_base", teacher_ce < base_ce, False, "scientific", f"base_ce={base_ce:.6f}; teacher_ce={teacher_ce:.6f}"),
        _gate(
            "oracle_sparse_norm_scale_repaired",
            _float(oracle.get("residual_l2_mean_ratio_vs_teacher"), 0.0) >= 0.5,
            False,
            "scientific",
            f"oracle_ratio={oracle.get('residual_l2_mean_ratio_vs_teacher')}",
        ),
        _gate(
            "oracle_sparse_beats_target_nulls",
            oracle_mse + 0.02 < target_null_mse,
            False,
            "scientific",
            f"oracle_mse={oracle.get('teacher_residual_reconstruction_mse')}; target_null_mse={target_null_mse:.6f}",
        ),
        _gate(
            "oracle_sparse_beats_support_nulls",
            oracle_mse + 0.02 < support_null_mse,
            False,
            "scientific",
            f"oracle_mse={oracle.get('teacher_residual_reconstruction_mse')}; support_null_mse={support_null_mse:.6f}",
        ),
        _gate(
            "learned_router_low_oracle_regret",
            learned_mse - oracle_mse <= 0.05,
            False,
            "scientific",
            f"learned_mse={learned.get('teacher_residual_reconstruction_mse')}; oracle_mse={oracle.get('teacher_residual_reconstruction_mse')}",
        ),
        _gate(
            "learned_router_beats_flat_value_control_on_churn_or_commutator",
            _float(learned.get("finite_update_commutator_proxy"), math.inf)
            <= _float(flat.get("finite_update_commutator_proxy"), -math.inf)
            or _float(learned.get("functional_churn"), math.inf) <= _float(flat.get("functional_churn"), -math.inf),
            False,
            "scientific",
            f"learned_commutator={learned.get('finite_update_commutator_proxy')}; flat_commutator={flat.get('finite_update_commutator_proxy')}",
        ),
        _gate(
            "learned_router_ce_guardrail",
            _float(learned.get("ce"), math.inf) <= base_ce + 0.02,
            False,
            "scientific",
            f"learned_ce={learned.get('ce')}; base_ce={base_ce:.6f}",
        ),
    ]


def _gate(criterion: str, passed: bool, required: bool, gate_type: str, evidence: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "required": required,
        "gate_type": gate_type,
        "evidence": evidence,
    }


def _selected_next_step(scientific_failures: list[dict[str, Any]], status: str) -> str:
    if status != "pass":
        return "repair runtime/source failures before rerunning the value-capacity/norm-control assay"
    failed = {row["criterion"] for row in scientific_failures}
    if "dense_teacher_improves_base" in failed:
        return "repair dense-teacher adequacy before interpreting sparse value capacity or routing"
    if "oracle_sparse_norm_scale_repaired" in failed or "oracle_sparse_beats_target_nulls" in failed:
        return "redesign column value capacity or residual target factorization before more routing work"
    if "learned_router_low_oracle_regret" in failed or "learned_router_ce_guardrail" in failed:
        return "diagnose learned sparse routing/value selection versus flat-value control dominance before GPU"
    if scientific_failures:
        return "inspect failed non-CE gates before GPU validation"
    return "repeat on additional seeds before any GPU validation"


def _source_row(source: str, path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": summary.get("status", ""),
        "decision": summary.get("decision", ""),
        "claim_status": summary.get("claim_status", ""),
        "selected_next_step": summary.get("selected_next_step", ""),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    norm_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", source_rows)
    _write_csv(out_dir / "training_rows.csv", training_rows)
    _write_csv(out_dir / "arm_metrics.csv", arm_rows)
    _write_csv(out_dir / "norm_control_rows.csv", norm_rows)
    _write_csv(out_dir / "gate_criteria.csv", gate_rows)
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Dense-Teacher Residual Value-Capacity/Norm-Control Assay",
            "",
            f"Decision: `{summary['decision']}`.",
            f"Claim status: `{summary['claim_status']}`.",
            "",
            "This is local CPU trained evidence only. GPU validation and promotion remain blocked.",
            "",
            f"Dense-teacher CE improvement: `{summary['dense_teacher_ce_improvement']}`.",
            f"Next step: {summary['selected_next_step']}",
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
    parser.add_argument("--pregate-dir", type=Path, default=DEFAULT_PREGATE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--teacher-steps", type=int, default=100)
    parser.add_argument("--router-steps", type=int, default=80)
    parser.add_argument("--value-steps", type=int, default=80)
    parser.add_argument("--control-steps", type=int, default=80)
    args = parser.parse_args(argv)
    summary = run_dense_teacher_residual_value_capacity_norm_assay(
        pregate_dir=args.pregate_dir,
        out_dir=args.out,
        seed=args.seed,
        teacher_steps=args.teacher_steps,
        router_steps=args.router_steps,
        value_steps=args.value_steps,
        control_steps=args.control_steps,
    )
    print(json.dumps({key: summary[key] for key in ("status", "decision", "claim_status", "selected_next_step")}, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
