"""Run the local pilot contract for the orthogonalized sparse core/periphery branch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PREGATE_DIR = Path("results/reports/orthogonalized_sparse_core_periphery_interference_pregate")
DEFAULT_OUT_DIR = Path("results/reports/orthogonalized_sparse_core_periphery_interference_pilot")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "arm_metrics.csv",
    "observable_gates.csv",
    "matched_control_matrix.csv",
    "leakage_null_matrix.csv",
    "notes.md",
)

CANDIDATE_ARM = "orthogonalized_sparse_additive_core_periphery"
NEXT_ACTION = "inspect_trained_local_cpu_pilot_observable_failures_before_gpu"
REPAIR_ACTION = "repair_orthogonalized_sparse_core_periphery_pilot_sources"
TRAINING_NOT_IMPLEMENTED = "training_not_implemented_yet"


def run_orthogonalized_sparse_core_periphery_interference_pilot(
    *,
    pregate_dir: Path = DEFAULT_PREGATE_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    schema_only: bool = False,
) -> dict[str, Any]:
    """Write pilot artifacts, failing closed until real training rows are implemented."""

    start = time.time()
    pregate_summary_path = pregate_dir / "summary.json"
    pregate = _read_json(pregate_summary_path)
    source_rows = [_source_row("orthogonalized_sparse_core_periphery_interference_pregate", pregate_summary_path, pregate)]
    preflight = _preflight_rows(pregate_dir, pregate)
    preflight_passed = all(row["passed"] for row in preflight)
    runtime_error = ""
    arm_rows: list[dict[str, Any]] = []
    if preflight_passed:
        if schema_only:
            arm_rows = _synthetic_arm_rows()
        else:
            try:
                arm_rows = _trained_arm_rows()
            except Exception as exc:  # pragma: no cover - depends on torch runtime
                runtime_error = f"{type(exc).__name__}: {exc}"
    control_rows = _matched_control_rows(arm_rows)
    null_rows = _leakage_null_rows(arm_rows)
    observable_rows = _observable_rows(arm_rows)
    artifact_rows = _artifact_gate_rows(arm_rows, control_rows, null_rows)
    training_rows = _training_gate_rows(
        schema_only=schema_only,
        preflight_passed=preflight_passed,
        arm_rows=arm_rows,
        runtime_error=runtime_error,
    )
    failures = [row for row in preflight + training_rows + artifact_rows if not row["passed"]]
    scientific_failures = [row for row in observable_rows if not row["passed"]]
    status = "pass" if not failures else "fail"
    scientific_gate = "blocked" if scientific_failures or status != "pass" else "advances_local_review_only"
    source_failed = any(not row["passed"] for row in preflight)
    selected_next_action = REPAIR_ACTION if source_failed else NEXT_ACTION
    training_rows_present = bool(arm_rows) and not schema_only and not runtime_error
    summary = {
        "status": status,
        "decision": (
            "orthogonalized_sparse_core_periphery_interference_pilot_recorded"
            if status == "pass"
            else "orthogonalized_sparse_core_periphery_interference_pilot_failed_closed"
        ),
        "claim_status": (
            "deterministic_schema_pilot_blocks_gpu_until_real_training_rows_clear_dense_mlp_gates"
            if status == "pass" and schema_only
            else "bounded_local_cpu_training_rows_recorded_no_gpu_claim"
            if status == "pass" and training_rows_present
            else "training_runtime_or_artifact_contract_failed"
            if not source_failed
            else "pilot_sources_or_artifact_contract_incomplete"
        ),
        "scientific_gate": scientific_gate,
        "selected_next_action": selected_next_action,
        "selected_next_step": (
            "inspect the trained local CPU pilot's failed observable gates and decide whether to redesign or close this branch before GPU validation"
            if training_rows_present
            else "replace deterministic synthetic pilot rows with a bounded local CPU training pilot using the same artifacts and gates"
            if not source_failed
            else "repair the missing pregate source before running the pilot"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local CPU schema/pilot work only; RunPod and Colab remain blocked",
        "pregate_dir": str(pregate_dir),
        "out_dir": str(out_dir),
        "source_rows": source_rows,
        "arm_metrics": arm_rows,
        "matched_control_matrix": control_rows,
        "leakage_null_matrix": null_rows,
        "observable_gates": observable_rows,
        "gate_criteria": preflight + training_rows + artifact_rows + observable_rows,
        "failures": failures,
        "scientific_failures": scientific_failures,
        "candidate_arm": CANDIDATE_ARM,
        "arm_count": len(arm_rows),
        "matched_control_row_count": len(control_rows),
        "leakage_null_row_count": len(null_rows),
        "observable_gate_count": len(observable_rows),
        "schema_only": schema_only,
        "synthetic_rows_only": schema_only and bool(arm_rows),
        "training_rows_present": training_rows_present,
        "training_status": "schema_only" if schema_only else "trained_local_cpu_rows" if training_rows_present else TRAINING_NOT_IMPLEMENTED,
        "runtime_error": runtime_error,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _preflight_rows(pregate_dir: Path, pregate: dict[str, Any]) -> list[dict[str, Any]]:
    required_files = (
        "summary.json",
        "mechanism_arms.csv",
        "matched_controls.csv",
        "observable_gates.csv",
        "leakage_nulls.csv",
    )
    return [
        _criterion(
            "pregate_summary_selected_pilot",
            pregate.get("status") == "pass"
            and pregate.get("selected_next_action") == "implement_local_orthogonalized_sparse_core_periphery_interference_pilot"
            and pregate.get("requires_gpu_now") is False
            and pregate.get("advance_to_gpu_validation") is False,
            pregate.get("selected_next_action", ""),
            "pregate must pass and select this local CPU pilot with GPU blocked",
        ),
        _criterion(
            "pregate_artifacts_present",
            all((pregate_dir / name).is_file() for name in required_files),
            [name for name in required_files if (pregate_dir / name).is_file()],
            "pregate summary, arms, controls, observable gates, and leakage nulls must exist",
        ),
    ]


def _synthetic_arm_rows() -> list[dict[str, Any]]:
    rows = [
        _arm(CANDIDATE_ARM, "sparse_mechanism_candidate", 0.48, 1.00, 1.12, 384, 1536, 0.19, 0.88, 0.010, 0.62, 0.58, 0.08),
        _arm("orthogonalized_sparse_no_norm_controller_ablation", "mechanism_ablation", 0.62, 1.44, 1.76, 384, 1536, 0.28, 0.82, 0.017, 0.44, 0.42, 0.04),
        _arm("orthogonalized_sparse_no_core_protection_ablation", "mechanism_ablation", 0.50, 1.03, 1.17, 384, 1536, 0.30, 0.78, 0.019, 0.57, 0.50, -0.01),
        _arm("orthogonalized_sparse_no_update_masks_ablation", "mechanism_ablation", 0.47, 1.02, 1.18, 384, 1536, 0.34, 0.75, 0.024, 0.59, 0.51, 0.01),
        _arm("dense_ridge_residual", "matched_control", 0.43, 1.00, 1.10, 384, 384, 0.31, 0.79, 0.020, 0.20, 0.12, 0.0),
        _arm("random_feature_mlp_residual", "matched_control", 0.39, 1.01, 1.11, 384, 1536, 0.29, 0.80, 0.018, 0.24, 0.14, 0.0),
        _arm("low_rank_residual", "matched_control", 0.45, 0.99, 1.09, 256, 768, 0.27, 0.81, 0.016, 0.18, 0.11, 0.0),
        _arm("same_router_flat_value_mlp", "matched_control", 0.41, 1.00, 1.12, 384, 1536, 0.26, 0.82, 0.015, 0.28, 0.16, 0.0),
        _arm("random_sparse_columns", "leakage_or_null_control", 0.91, 1.00, 1.14, 384, 1536, 0.25, 0.83, 0.014, 0.07, 0.05, 0.0),
        _arm("frequency_matched_sparse_router", "leakage_or_null_control", 0.74, 1.00, 1.13, 384, 1536, 0.24, 0.84, 0.013, 0.12, 0.09, 0.0),
        _arm("token_position_only_router", "leakage_or_null_control", 0.86, 1.00, 1.12, 384, 1536, 0.23, 0.84, 0.012, 0.08, 0.07, 0.0),
        _arm("shuffled_teacher_residual_targets", "leakage_or_null_control", 1.02, 1.00, 1.11, 384, 1536, 0.21, 0.86, 0.011, 0.03, 0.02, 0.0),
        _arm("delayed_teacher_residual_targets", "leakage_or_null_control", 0.88, 1.00, 1.11, 384, 1536, 0.22, 0.85, 0.012, 0.06, 0.04, 0.0),
    ]
    best_control_ce = min(row["ce"] for row in rows if row["family"] == "matched_control")
    for row in rows:
        row["ce_delta_vs_best_matched_dense_mlp"] = round(row["ce"] - best_control_ce, 6)
        row["feature_schema_hash"] = _hash_text("token_id,position_id,prefix_hidden_only")
        row["uses_future_hidden_or_delta"] = False
        row["uses_teacher_residual_or_logits_at_eval"] = False
        row["uses_oracle_support_at_eval"] = False
    return rows


@dataclass(frozen=True)
class _ArmSpec:
    arm: str
    family: str
    model_kind: str
    orthogonalize: bool = False
    norm_controller: bool = False
    core_protection: bool = False
    update_masks: bool = False
    shuffled_targets: bool = False
    delayed_targets: bool = False
    token_position_only: bool = False
    random_support: bool = False
    frequency_support: bool = False


def _trained_arm_rows() -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(11)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    data = _synthetic_training_stream(torch, F)
    rows: list[dict[str, Any]] = []
    for spec in _arm_specs():
        row = _train_and_evaluate_arm(torch, F, data, spec)
        row["feature_schema_hash"] = _hash_text("synthetic_prefix_features,token_id,position_id")
        row["uses_future_hidden_or_delta"] = False
        row["uses_teacher_residual_or_logits_at_eval"] = False
        row["uses_oracle_support_at_eval"] = False
        rows.append(row)
    best_control_ce = min(row["ce"] for row in rows if row["family"] == "matched_control")
    for row in rows:
        row["ce_delta_vs_best_matched_dense_mlp"] = round(row["ce"] - best_control_ce, 6)
    return rows


def _synthetic_training_stream(torch: Any, F: Any) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(7)
    n = 192
    input_dim = 12
    classes = 5
    x = torch.randn(n, input_dim, generator=generator)
    token_id = torch.arange(n) % 8
    position_id = torch.arange(n) % 6
    x[:, 0] += (token_id.float() - 3.5) / 4.0
    x[:, 1] += (position_id.float() - 2.5) / 3.0
    base_w = torch.randn(input_dim, classes, generator=generator) * 0.22
    base_logits = x @ base_w
    support = ((x[:, 0] > 0).long() + 2 * (x[:, 1] > 0).long() + 4 * (x[:, 2] > 0).long()) % 6
    values = torch.tensor(
        [
            [0.75, -0.40, 0.10, -0.25, -0.20],
            [-0.15, 0.70, -0.35, -0.10, -0.10],
            [-0.10, -0.25, 0.78, -0.23, -0.20],
            [-0.30, -0.15, -0.15, 0.72, -0.12],
            [-0.18, -0.18, -0.16, -0.16, 0.68],
            [0.38, -0.30, 0.34, -0.22, -0.20],
        ],
        dtype=x.dtype,
    )
    teacher_residual = values[support] + 0.10 * torch.sin(x[:, :classes])
    teacher_logits = base_logits + teacher_residual
    labels = teacher_logits.argmax(dim=-1)
    train = torch.arange(0, 96)
    task_a = torch.arange(96, 144)
    task_b = torch.arange(144, 192)
    eval_idx = torch.arange(96, 192)
    return {
        "x": x,
        "token_id": token_id,
        "position_id": position_id,
        "base_logits": base_logits,
        "teacher_residual": teacher_residual,
        "teacher_logits": teacher_logits,
        "labels": labels,
        "support": support,
        "train": train,
        "task_a": task_a,
        "task_b": task_b,
        "eval": eval_idx,
        "input_dim": input_dim,
        "classes": classes,
    }


def _arm_specs() -> list[_ArmSpec]:
    return [
        _ArmSpec(CANDIDATE_ARM, "sparse_mechanism_candidate", "sparse", True, True, True, True),
        _ArmSpec("orthogonalized_sparse_no_norm_controller_ablation", "mechanism_ablation", "sparse", True, False, True, True),
        _ArmSpec("orthogonalized_sparse_no_core_protection_ablation", "mechanism_ablation", "sparse", True, True, False, True),
        _ArmSpec("orthogonalized_sparse_no_update_masks_ablation", "mechanism_ablation", "sparse", True, True, True, False),
        _ArmSpec("dense_ridge_residual", "matched_control", "linear"),
        _ArmSpec("random_feature_mlp_residual", "matched_control", "mlp"),
        _ArmSpec("low_rank_residual", "matched_control", "low_rank"),
        _ArmSpec("same_router_flat_value_mlp", "matched_control", "flat_sparse"),
        _ArmSpec("random_sparse_columns", "leakage_or_null_control", "sparse", random_support=True),
        _ArmSpec("frequency_matched_sparse_router", "leakage_or_null_control", "sparse", frequency_support=True),
        _ArmSpec("token_position_only_router", "leakage_or_null_control", "linear", token_position_only=True),
        _ArmSpec("shuffled_teacher_residual_targets", "leakage_or_null_control", "linear", shuffled_targets=True),
        _ArmSpec("delayed_teacher_residual_targets", "leakage_or_null_control", "linear", delayed_targets=True),
    ]


def _train_and_evaluate_arm(torch: Any, F: Any, data: dict[str, Any], spec: _ArmSpec) -> dict[str, Any]:
    model = _make_model(torch, data, spec)
    optimizer = torch.optim.AdamW(_parameter_groups(model, spec), lr=0.04, weight_decay=1e-4)
    train_idx = data["train"]
    target = data["labels"].clone()
    residual_target = data["teacher_residual"].clone()
    if spec.shuffled_targets:
        target[train_idx] = target[train_idx][torch.randperm(train_idx.numel())]
        residual_target[train_idx] = residual_target[train_idx][torch.randperm(train_idx.numel())]
    if spec.delayed_targets:
        target[train_idx] = target[train_idx.roll(1)]
        residual_target[train_idx] = residual_target[train_idx.roll(1)]
    for _ in range(36):
        optimizer.zero_grad(set_to_none=True)
        residual = _model_residual(torch, model, data, spec)
        logits = data["base_logits"] + residual
        ce = F.cross_entropy(logits[train_idx], target[train_idx])
        mse = F.mse_loss(residual[train_idx], residual_target[train_idx])
        penalty = _model_penalty(torch, model, spec)
        loss = ce + 0.35 * mse + penalty
        loss.backward()
        optimizer.step()

    eval_idx = data["eval"]
    with torch.no_grad():
        residual = _model_residual(torch, model, data, spec)
        logits = data["base_logits"] + residual
        ce = float(F.cross_entropy(logits[eval_idx], data["labels"][eval_idx]).item())
        residual_eval = residual[eval_idx]
        residual_l2 = residual_eval.norm(dim=-1)
        base_pred = data["base_logits"][eval_idx].argmax(dim=-1)
        pred = logits[eval_idx].argmax(dim=-1)
        churn = float((pred != base_pred).float().mean().item())
        selectivity = _intervention_selectivity(torch, data, model, spec)
        reuse = _context_reuse_score(torch, data, logits)
        prune = _pruning_delta(torch, F, data, model, spec)
    retention = _retention_after_sequential_updates(torch, F, data, spec)
    commutator = _finite_update_commutator(torch, F, data, spec)
    active_params, stored_params = _param_counts(model, spec)
    return {
        "arm": spec.arm,
        "family": spec.family,
        "row_source": "bounded_local_cpu_trained_synthetic_mechanism_stream",
        "ce": round(ce, 6),
        "residual_l2_mean": round(float(residual_l2.mean().item()), 6),
        "residual_l2_p95": round(float(torch.quantile(residual_l2, 0.95).item()), 6),
        "active_params": active_params,
        "stored_params": stored_params,
        "functional_churn_flip_rate": round(churn, 6),
        "retention_after_sequential_updates": round(retention, 6),
        "finite_update_commutator_symmetric_kl": round(commutator, 6),
        "intervention_selectivity": round(selectivity, 6),
        "context_reuse_score": round(reuse, 6),
        "periphery_first_pruning_delta": round(prune, 6),
    }


def _make_model(torch: Any, data: dict[str, Any], spec: _ArmSpec) -> Any:
    if spec.model_kind in {"sparse", "flat_sparse"}:
        return {
            "router": torch.nn.Linear(data["input_dim"], 6),
            "core": torch.nn.Parameter(torch.randn(6, data["classes"]) * 0.04),
            "periphery": torch.nn.Parameter(torch.randn(6, data["classes"]) * 0.04),
            "norm": torch.nn.Linear(data["input_dim"], 1),
        }
    if spec.model_kind == "linear":
        dim = 2 if spec.token_position_only else data["input_dim"]
        return torch.nn.Linear(dim, data["classes"], bias=False)
    if spec.model_kind == "low_rank":
        return torch.nn.Sequential(
            torch.nn.Linear(data["input_dim"], 3, bias=False),
            torch.nn.Linear(3, data["classes"], bias=False),
        )
    return torch.nn.Sequential(
        torch.nn.Linear(data["input_dim"], 10),
        torch.nn.Tanh(),
        torch.nn.Linear(10, data["classes"]),
    )


def _parameter_groups(model: Any, spec: _ArmSpec) -> Any:
    if not isinstance(model, dict):
        return model.parameters()
    core_lr = 0.008 if spec.core_protection else 0.04
    return [
        {"params": model["router"].parameters(), "lr": 0.04},
        {"params": [model["core"]], "lr": core_lr},
        {"params": [model["periphery"]], "lr": 0.04},
        {"params": model["norm"].parameters(), "lr": 0.04},
    ]


def _model_residual(torch: Any, model: Any, data: dict[str, Any], spec: _ArmSpec) -> Any:
    if not isinstance(model, dict):
        if spec.token_position_only:
            x = torch.stack([data["token_id"].float() / 7.0, data["position_id"].float() / 5.0], dim=-1)
        else:
            x = data["x"]
        return model(x)
    x = data["x"]
    router_logits = model["router"](x)
    if spec.random_support:
        router_logits = torch.randn_like(router_logits)
    elif spec.frequency_support:
        frequent = int(data["support"].bincount().argmax().item())
        router_logits = torch.nn.functional.one_hot(torch.full_like(data["support"], frequent), 6).float() * 4.0
    weights = torch.softmax(router_logits, dim=-1)
    values = model["core"] + model["periphery"]
    if spec.update_masks:
        mask = (torch.arange(values.shape[1], device=values.device).view(1, -1) + torch.arange(values.shape[0], device=values.device).view(-1, 1)) % 2
        values = model["core"] + model["periphery"] * (0.65 + 0.35 * mask.float())
    residual = weights @ values
    if spec.norm_controller:
        residual = residual * (0.55 + torch.sigmoid(model["norm"](x)))
    return residual


def _model_penalty(torch: Any, model: Any, spec: _ArmSpec) -> Any:
    if not isinstance(model, dict):
        return 0.0
    penalty = 0.0
    if spec.orthogonalize:
        values = model["core"] + model["periphery"]
        normed = values / values.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        gram = normed @ normed.t()
        eye = torch.eye(gram.shape[0], device=gram.device)
        penalty = penalty + 0.04 * ((gram - eye) ** 2).mean()
    if spec.core_protection:
        penalty = penalty + 0.02 * model["core"].pow(2).mean()
    return penalty


def _retention_after_sequential_updates(torch: Any, F: Any, data: dict[str, Any], spec: _ArmSpec) -> float:
    model = _make_model(torch, data, spec)
    opt = torch.optim.AdamW(_parameter_groups(model, spec), lr=0.035)
    _train_indices(torch, F, model, opt, data, spec, data["task_a"], steps=12)
    with torch.no_grad():
        before = (data["base_logits"] + _model_residual(torch, model, data, spec))[data["task_a"]].argmax(dim=-1)
    _train_indices(torch, F, model, opt, data, spec, data["task_b"], steps=12)
    with torch.no_grad():
        after = (data["base_logits"] + _model_residual(torch, model, data, spec))[data["task_a"]].argmax(dim=-1)
    return float((before == after).float().mean().item())


def _finite_update_commutator(torch: Any, F: Any, data: dict[str, Any], spec: _ArmSpec) -> float:
    ab = _make_model(torch, data, spec)
    ba = _make_model(torch, data, spec)
    ba.load_state_dict(ab.state_dict()) if not isinstance(ab, dict) else _copy_sparse_state(torch, ab, ba)
    opt_ab = torch.optim.AdamW(_parameter_groups(ab, spec), lr=0.035)
    opt_ba = torch.optim.AdamW(_parameter_groups(ba, spec), lr=0.035)
    _train_indices(torch, F, ab, opt_ab, data, spec, data["task_a"], steps=8)
    _train_indices(torch, F, ab, opt_ab, data, spec, data["task_b"], steps=8)
    _train_indices(torch, F, ba, opt_ba, data, spec, data["task_b"], steps=8)
    _train_indices(torch, F, ba, opt_ba, data, spec, data["task_a"], steps=8)
    with torch.no_grad():
        p = torch.log_softmax((data["base_logits"] + _model_residual(torch, ab, data, spec))[data["eval"]], dim=-1)
        q = torch.log_softmax((data["base_logits"] + _model_residual(torch, ba, data, spec))[data["eval"]], dim=-1)
        pp = p.exp()
        qq = q.exp()
        kl = 0.5 * ((pp * (p - q)).sum(dim=-1).mean() + (qq * (q - p)).sum(dim=-1).mean())
    return float(kl.item())


def _copy_sparse_state(torch: Any, src: dict[str, Any], dst: dict[str, Any]) -> None:
    dst["router"].load_state_dict(src["router"].state_dict())
    dst["norm"].load_state_dict(src["norm"].state_dict())
    with torch.no_grad():
        dst["core"].copy_(src["core"])
        dst["periphery"].copy_(src["periphery"])


def _train_indices(torch: Any, F: Any, model: Any, optimizer: Any, data: dict[str, Any], spec: _ArmSpec, idx: Any, *, steps: int) -> None:
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = data["base_logits"] + _model_residual(torch, model, data, spec)
        loss = F.cross_entropy(logits[idx], data["labels"][idx]) + _model_penalty(torch, model, spec)
        loss.backward()
        optimizer.step()


def _intervention_selectivity(torch: Any, data: dict[str, Any], model: Any, spec: _ArmSpec) -> float:
    if not isinstance(model, dict):
        return 0.15
    with torch.no_grad():
        residual = _model_residual(torch, model, data, spec)
        selectivities = []
        for support_id in range(6):
            mask = data["support"] == support_id
            if not bool(mask.any()):
                continue
            perturbed = residual.clone()
            perturbed[mask] -= model["periphery"][support_id]
            in_delta = (residual[mask] - perturbed[mask]).norm(dim=-1).mean()
            out_delta = (residual[~mask] - perturbed[~mask]).norm(dim=-1).mean()
            selectivities.append(float((in_delta / (in_delta + out_delta + 1e-6)).item()))
    return sum(selectivities) / max(1, len(selectivities))


def _context_reuse_score(torch: Any, data: dict[str, Any], logits: Any) -> float:
    with torch.no_grad():
        pred = logits[data["eval"]].argmax(dim=-1)
        labels = data["labels"][data["eval"]]
        agreement = (pred == labels).float()
        support = data["support"][data["eval"]]
        per_support = [agreement[support == idx].mean().item() for idx in range(6) if bool((support == idx).any())]
    return float(sum(per_support) / max(1, len(per_support)))


def _pruning_delta(torch: Any, F: Any, data: dict[str, Any], model: Any, spec: _ArmSpec) -> float:
    if not isinstance(model, dict):
        return 0.0
    eval_idx = data["eval"]
    with torch.no_grad():
        base_values = model["core"] + model["periphery"]
        weights = torch.softmax(model["router"](data["x"]), dim=-1)
        if spec.norm_controller:
            scale = 0.55 + torch.sigmoid(model["norm"](data["x"]))
        else:
            scale = 1.0
        core_only = weights @ model["core"] * scale
        periphery_only = weights @ model["periphery"] * scale
        full = weights @ base_values * scale
        full_ce = F.cross_entropy((data["base_logits"] + full)[eval_idx], data["labels"][eval_idx])
        prune_periphery_ce = F.cross_entropy((data["base_logits"] + core_only)[eval_idx], data["labels"][eval_idx])
        prune_core_ce = F.cross_entropy((data["base_logits"] + periphery_only)[eval_idx], data["labels"][eval_idx])
    return float((prune_periphery_ce - full_ce - (prune_core_ce - full_ce)).item())


def _param_counts(model: Any, spec: _ArmSpec) -> tuple[int, int]:
    if isinstance(model, dict):
        stored = sum(parameter.numel() for module in (model["router"], model["norm"]) for parameter in module.parameters())
        stored += model["core"].numel() + model["periphery"].numel()
        return stored, stored
    stored = sum(parameter.numel() for parameter in model.parameters())
    return stored, stored


def _arm(
    arm: str,
    family: str,
    ce: float,
    residual_l2_mean: float,
    residual_l2_p95: float,
    active_params: int,
    stored_params: int,
    functional_churn_flip_rate: float,
    retention_after_updates: float,
    finite_update_commutator_symmetric_kl: float,
    intervention_selectivity: float,
    context_reuse_score: float,
    periphery_first_pruning_delta: float,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "family": family,
        "row_source": "deterministic_tiny_synthetic_schema_pilot",
        "ce": ce,
        "residual_l2_mean": residual_l2_mean,
        "residual_l2_p95": residual_l2_p95,
        "active_params": active_params,
        "stored_params": stored_params,
        "functional_churn_flip_rate": functional_churn_flip_rate,
        "retention_after_sequential_updates": retention_after_updates,
        "finite_update_commutator_symmetric_kl": finite_update_commutator_symmetric_kl,
        "intervention_selectivity": intervention_selectivity,
        "context_reuse_score": context_reuse_score,
        "periphery_first_pruning_delta": periphery_first_pruning_delta,
    }


def _matched_control_rows(arm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate = _row_for(arm_rows, CANDIDATE_ARM)
    return [
        {
            "control": row["arm"],
            "candidate_ce_delta": round(candidate.get("ce", 0.0) - row["ce"], 6),
            "candidate_churn_delta": round(candidate.get("functional_churn_flip_rate", 0.0) - row["functional_churn_flip_rate"], 6),
            "candidate_retention_delta": round(candidate.get("retention_after_sequential_updates", 0.0) - row["retention_after_sequential_updates"], 6),
            "candidate_commutator_delta": round(candidate.get("finite_update_commutator_symmetric_kl", 0.0) - row["finite_update_commutator_symmetric_kl"], 6),
            "matched_residual_l2": abs(candidate.get("residual_l2_mean", 0.0) - row["residual_l2_mean"]) <= 0.02,
            "matched_active_params": candidate.get("active_params") == row["active_params"],
            "matched_stored_params": candidate.get("stored_params") == row["stored_params"],
        }
        for row in arm_rows
        if row.get("family") == "matched_control"
    ]


def _leakage_null_rows(arm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate = _row_for(arm_rows, CANDIDATE_ARM)
    return [
        {
            "null_control": row["arm"],
            "candidate_ce_better_by": round(row["ce"] - candidate.get("ce", 0.0), 6),
            "candidate_selectivity_better_by": round(candidate.get("intervention_selectivity", 0.0) - row["intervention_selectivity"], 6),
            "null_matches_candidate_within_tolerance": abs(row["ce"] - candidate.get("ce", 0.0)) <= 0.03,
            "uses_nondeployable_features": bool(
                row.get("uses_future_hidden_or_delta")
                or row.get("uses_teacher_residual_or_logits_at_eval")
                or row.get("uses_oracle_support_at_eval")
            ),
        }
        for row in arm_rows
        if row.get("family") == "leakage_or_null_control"
    ]


def _observable_rows(arm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not arm_rows:
        return []
    candidate = _row_for(arm_rows, CANDIDATE_ARM)
    dense_mlp = [row for row in arm_rows if row["arm"] in {"dense_ridge_residual", "random_feature_mlp_residual"}]
    ablations = [row for row in arm_rows if row.get("family") == "mechanism_ablation"]
    nulls = [row for row in arm_rows if row.get("family") == "leakage_or_null_control"]
    best_dense_mlp_ce = min(row["ce"] for row in dense_mlp)
    ce_threshold = best_dense_mlp_ce + min(0.05, best_dense_mlp_ce * 0.05)
    return [
        _criterion("ce_guardrail", candidate["ce"] <= ce_threshold, candidate["ce"], f"candidate CE must be <= {ce_threshold:.6f}"),
        _criterion("residual_l2_budget", all(abs(candidate["residual_l2_mean"] - row["residual_l2_mean"]) <= 0.02 for row in dense_mlp), candidate["residual_l2_mean"], "candidate and dense/MLP controls must match residual L2 mean within 0.02"),
        _criterion("active_and_stored_params", all(row["active_params"] == candidate["active_params"] for row in dense_mlp), candidate["active_params"], "candidate must match dense/MLP active params in this schema pilot"),
        _criterion("functional_churn_flip_rate", candidate["functional_churn_flip_rate"] <= min(row["functional_churn_flip_rate"] for row in dense_mlp), candidate["functional_churn_flip_rate"], "candidate churn must beat or match best dense/MLP"),
        _criterion("retention_after_sequential_updates", candidate["retention_after_sequential_updates"] >= max(row["retention_after_sequential_updates"] for row in dense_mlp), candidate["retention_after_sequential_updates"], "candidate retention must beat or match best dense/MLP"),
        _criterion("finite_update_commutator_symmetric_kl", candidate["finite_update_commutator_symmetric_kl"] <= min(row["finite_update_commutator_symmetric_kl"] for row in dense_mlp), candidate["finite_update_commutator_symmetric_kl"], "candidate commutator must beat or match best dense/MLP"),
        _criterion("intervention_selectivity", candidate["intervention_selectivity"] >= 0.50, candidate["intervention_selectivity"], "candidate selectivity must be at least 0.50"),
        _criterion("context_reuse_score", candidate["context_reuse_score"] >= 0.50, candidate["context_reuse_score"], "candidate reuse score must be at least 0.50"),
        _criterion("periphery_first_pruning_delta", candidate["periphery_first_pruning_delta"] > max(row["periphery_first_pruning_delta"] for row in ablations), candidate["periphery_first_pruning_delta"], "full candidate must have stronger periphery-first pruning signal than ablations"),
        _criterion("null_control_rejection", all(row["ce"] - candidate["ce"] > 0.03 for row in nulls), candidate["ce"], "all leakage/null controls must be worse than candidate by >0.03 CE"),
        _criterion("deployable_feature_schema", not any(row.get("uses_future_hidden_or_delta") or row.get("uses_teacher_residual_or_logits_at_eval") or row.get("uses_oracle_support_at_eval") for row in arm_rows), candidate["feature_schema_hash"], "no evaluation row may use future hidden/delta, teacher residual/logits, or oracle supports"),
    ]


def _artifact_gate_rows(
    arm_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    null_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _criterion("candidate_and_ablations_present", {CANDIDATE_ARM, "orthogonalized_sparse_no_norm_controller_ablation", "orthogonalized_sparse_no_core_protection_ablation", "orthogonalized_sparse_no_update_masks_ablation"}.issubset({row["arm"] for row in arm_rows}), len(arm_rows), "candidate plus norm/core/mask ablations are required"),
        _criterion("matched_controls_present", len(control_rows) >= 4, len(control_rows), "dense, MLP, low-rank, and flat controls are required"),
        _criterion("leakage_nulls_present", len(null_rows) >= 5, len(null_rows), "token-position, shuffled, delayed, random, and frequency null rows are required"),
    ]


def _training_gate_rows(
    *,
    schema_only: bool,
    preflight_passed: bool,
    arm_rows: list[dict[str, Any]],
    runtime_error: str,
) -> list[dict[str, Any]]:
    if not preflight_passed:
        return []
    if schema_only:
        passed = bool(arm_rows)
        actual = "schema_only" if passed else "schema_only_rows_missing"
    else:
        passed = bool(arm_rows) and not runtime_error and all(
            row.get("row_source") == "bounded_local_cpu_trained_synthetic_mechanism_stream" for row in arm_rows
        )
        actual = "trained_local_cpu_rows" if passed else runtime_error or TRAINING_NOT_IMPLEMENTED
    return [
        _criterion(
            "real_training_rows_present",
            passed,
            actual,
            "default pilot runs must use real training rows; pass --schema-only for deterministic contract rows",
        )
    ]


def _row_for(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    for row in rows:
        if row.get("arm") == arm:
            return row
    return {}


def _criterion(criterion: str, passed: bool, actual: Any, threshold: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "actual": actual,
        "threshold": threshold,
        "failure_reason": "" if passed else threshold,
    }


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file() and bool(payload),
        "sha256": _file_sha256(path),
        "mtime": _mtime(path),
        "status": payload.get("status", "missing" if not path.is_file() else ""),
        "decision": payload.get("decision", ""),
        "selected_next_action": payload.get("selected_next_action", ""),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "arm_metrics.csv", summary["arm_metrics"])
    _write_csv(out_dir / "observable_gates.csv", summary["observable_gates"])
    _write_csv(out_dir / "matched_control_matrix.csv", summary["matched_control_matrix"])
    _write_csv(out_dir / "leakage_null_matrix.csv", summary["leakage_null_matrix"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"], lineterminator="\n")
        writer.writeheader()
        for row in rows or [{"status": "missing"}]:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in writer.fieldnames or []})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _notes(summary: dict[str, Any]) -> str:
    if summary["schema_only"]:
        evidence_note = "GPU validation remains blocked. These rows are a deterministic local schema pilot, not training evidence."
    elif summary["training_rows_present"]:
        evidence_note = "GPU validation remains blocked. These are bounded local CPU training rows; promotion still requires observable gates against dense/MLP/null controls."
    else:
        evidence_note = "GPU validation remains blocked. Default pilot execution fails closed because real training rows are not implemented yet."
    return "\n".join(
        [
            "# Orthogonalized Sparse Core/Periphery Interference Pilot",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Scientific gate: `{summary['scientific_gate']}`",
            f"- Selected action: `{summary['selected_next_action']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            f"- Schema only: `{summary['schema_only']}`",
            f"- Synthetic rows only: `{summary['synthetic_rows_only']}`",
            f"- Training status: `{summary['training_status']}`",
            "",
            evidence_note,
            "",
        ]
    )


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mtime(path: Path) -> str:
    if not path.is_file():
        return ""
    return str(path.stat().st_mtime)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pregate-dir", type=Path, default=DEFAULT_PREGATE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Emit deterministic schema rows for contract checks instead of requiring real training rows.",
    )
    args = parser.parse_args()
    summary = run_orthogonalized_sparse_core_periphery_interference_pilot(
        pregate_dir=args.pregate_dir,
        out_dir=args.out,
        schema_only=args.schema_only,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "scientific_gate": summary["scientific_gate"],
                "selected_next_action": summary["selected_next_action"],
                "requires_gpu_now": summary["requires_gpu_now"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
