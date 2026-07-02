"""Probe deployable imitation of the oracle sparse-coding basis.

The preceding oracle sparse-coding feasibility assay showed that the teacher
residual is locally columnable under nondeployable top-k coefficients, but that
the deployable linear router/scalar head retained too little oracle gain. This
bounded CPU probe tests whether a stronger prefix-safe joint MLP imitation
materially closes that regret before any GPU validation.
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

from relaleap.experiments.dense_teacher_oracle_sparse_coding_feasibility import (
    _decode_sparse,
    _deterministic_random_mask,
    _mask_to_support,
    _orthogonal_basis,
    _read_json,
    _review_field,
    _selected_next_step as _oracle_selected_next_step,
    _topk_mask,
    _train_coeff_head,
    _train_mask_router,
)
from relaleap.experiments.dense_teacher_residual_value_capacity_norm_assay import (
    _Teacher,
    _arm_metrics,
    _make_data,
    _norm_match,
    _source_row,
    _train_flat_value_head,
)


DEFAULT_ORACLE_FEASIBILITY = Path("results/reports/dense_teacher_oracle_sparse_coding_feasibility/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_deployable_sparse_coding_imitation_probe")

DECISION = "dense_teacher_deployable_sparse_coding_imitation_probe_recorded"
FAIL_DECISION = "dense_teacher_deployable_sparse_coding_imitation_probe_failed_closed"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "imitation_rows.csv",
    "gate_criteria.csv",
    "notes.md",
)

ARMS = (
    "oracle_topk_orthogonal_sparse_coding",
    "oracle_support_learned_combo_coeff_sparse_coding",
    "learned_combo_support_oracle_coeff_sparse_coding",
    "baseline_linear_router_scalar_imitation",
    "enhanced_joint_mlp_router_scalar_imitation",
    "combo_mlp_router_scalar_imitation",
    "same_router_flat_value_control",
    "random_topk_sparse_coding_null",
    "no_update_control",
)


def run_dense_teacher_deployable_sparse_coding_imitation_probe(
    *,
    oracle_feasibility_path: Path = DEFAULT_ORACLE_FEASIBILITY,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 31,
    teacher_steps: int = 100,
    baseline_steps: int = 80,
    enhanced_steps: int = 200,
    combo_steps: int = 260,
    control_steps: int = 80,
    basis_size: int = 8,
    top_k: int = 2,
    hidden_dim: int = 48,
    data_column_count: int = 6,
) -> dict[str, Any]:
    """Run the bounded local deployable-imitation probe and write artifacts."""

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError("dense-teacher deployable sparse-coding imitation probe requires torch") from exc

    if min(teacher_steps, baseline_steps, enhanced_steps, combo_steps, control_steps) < 1:
        raise ValueError("all training step counts must be positive")
    if basis_size < 2:
        raise ValueError("basis_size must be at least 2")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if hidden_dim < 4:
        raise ValueError("hidden_dim must be at least 4")

    start = time.time()
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    oracle_feasibility = _read_json(oracle_feasibility_path)
    review_text = strategy_review_path.read_text(encoding="utf-8") if strategy_review_path.is_file() else ""
    source_rows = [
        _source_row("dense_teacher_oracle_sparse_coding_feasibility", oracle_feasibility_path, oracle_feasibility),
        {
            "source": "gpt_5_5_pro_strategy_review",
            "path": str(strategy_review_path),
            "present": strategy_review_path.is_file(),
            "status": "present" if strategy_review_path.is_file() else "missing",
            "decision": _review_field(review_text, "verdict"),
            "claim_status": _review_field(review_text, "strategic_change_level"),
            "selected_next_step": _review_field(review_text, "recommended_next_action"),
        },
    ]

    data = _make_data(torch, seed=seed, column_count=data_column_count)
    effective_basis_size = min(basis_size, data["classes"])
    if top_k > effective_basis_size:
        raise ValueError("top_k must not exceed the effective residual basis size")

    teacher = _Teacher(torch, data["input_dim"], data["classes"])
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=0.01)
    for _ in range(teacher_steps):
        loss = F.cross_entropy(data["base_logits_train"] + teacher(data["x_train"]), data["y_train"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        teacher_train = teacher(data["x_train"])
        teacher_holdout = teacher(data["x_holdout"])
        base_holdout_ce = float(F.cross_entropy(data["base_logits_holdout"], data["y_holdout"]).item())
        teacher_holdout_ce = float(F.cross_entropy(data["base_logits_holdout"] + teacher_holdout, data["y_holdout"]).item())

    mean, basis, train_coeff, holdout_coeff, _spectrum_rows = _orthogonal_basis(
        torch,
        teacher_train,
        teacher_holdout,
        effective_basis_size,
    )
    train_oracle_mask = _topk_mask(torch, train_coeff, top_k)
    oracle_mask = _topk_mask(torch, holdout_coeff, top_k)
    random_mask = _deterministic_random_mask(torch, len(data["x_holdout"]), effective_basis_size, top_k)

    baseline_router = _train_mask_router(
        torch,
        data["x_train"],
        train_oracle_mask,
        data["input_dim"],
        effective_basis_size,
        steps=baseline_steps,
    )
    baseline_coeff = _train_coeff_head(
        torch,
        F,
        data["x_train"],
        train_coeff,
        data["input_dim"],
        effective_basis_size,
        steps=baseline_steps,
    )
    enhanced = _train_joint_mlp_imitation(
        torch,
        F,
        data["x_train"],
        train_coeff,
        train_oracle_mask,
        data["input_dim"],
        effective_basis_size,
        hidden_dim=hidden_dim,
        steps=enhanced_steps,
    )
    combo_model, support_combos = _train_combo_mlp_imitation(
        torch,
        F,
        data["x_train"],
        train_coeff,
        train_oracle_mask,
        data["input_dim"],
        effective_basis_size,
        top_k,
        hidden_dim=hidden_dim,
        steps=combo_steps,
    )
    flat_value = _train_flat_value_head(
        torch,
        F,
        data["x_train"],
        teacher_train,
        data["input_dim"],
        data["classes"],
        steps=control_steps,
    )

    baseline_logits = baseline_router(data["x_holdout"])
    baseline_mask = _topk_mask(torch, baseline_logits, top_k)
    baseline_pred = _decode_sparse(mean, basis, baseline_coeff(data["x_holdout"]), baseline_mask)
    enhanced_logits, enhanced_coeff = _joint_outputs(enhanced, data["x_holdout"], effective_basis_size)
    enhanced_mask = _topk_mask(torch, enhanced_logits, top_k)
    enhanced_pred = _decode_sparse(mean, basis, enhanced_coeff, enhanced_mask)
    combo_logits, combo_coeff = _combo_outputs(combo_model, data["x_holdout"], effective_basis_size, len(support_combos))
    combo_mask = _combo_mask_from_logits(torch, combo_logits, support_combos, effective_basis_size)
    combo_pred = _decode_sparse(mean, basis, combo_coeff, combo_mask)
    oracle_support_learned_coeff_pred = _decode_sparse(mean, basis, combo_coeff, oracle_mask)
    learned_support_oracle_coeff_pred = _decode_sparse(mean, basis, holdout_coeff, combo_mask)
    zero_support = torch.zeros(len(data["x_holdout"]), dtype=torch.long)

    arms: dict[str, tuple[Any, Any, bool, str]] = {
        "oracle_topk_orthogonal_sparse_coding": (
            _decode_sparse(mean, basis, holdout_coeff, oracle_mask),
            _mask_to_support(torch, oracle_mask),
            True,
            "nondeployable oracle top-k sparse-coding ceiling",
        ),
        "oracle_support_learned_combo_coeff_sparse_coding": (
            oracle_support_learned_coeff_pred,
            _mask_to_support(torch, oracle_mask),
            True,
            "diagnostic cross: oracle support with deployable combo-learned scalar coefficients",
        ),
        "learned_combo_support_oracle_coeff_sparse_coding": (
            learned_support_oracle_coeff_pred,
            _mask_to_support(torch, combo_mask),
            True,
            "diagnostic cross: deployable combo support with nondeployable oracle scalar coefficients",
        ),
        "baseline_linear_router_scalar_imitation": (
            baseline_pred,
            _mask_to_support(torch, baseline_mask),
            False,
            "previous linear top-k mask router plus scalar coefficient head",
        ),
        "enhanced_joint_mlp_router_scalar_imitation": (
            enhanced_pred,
            _mask_to_support(torch, enhanced_mask),
            False,
            "prefix-safe joint MLP predicting top-k mask logits and scalar coefficients",
        ),
        "combo_mlp_router_scalar_imitation": (
            combo_pred,
            _mask_to_support(torch, combo_mask),
            False,
            "prefix-safe MLP predicting a structured top-k support combination and scalar coefficients",
        ),
        "same_router_flat_value_control": (
            _norm_match(torch, flat_value(data["x_holdout"]), teacher_train),
            _mask_to_support(torch, combo_mask),
            False,
            "combo-router support summarized with a flat value head control",
        ),
        "random_topk_sparse_coding_null": (
            _decode_sparse(mean, basis, holdout_coeff, random_mask),
            _mask_to_support(torch, random_mask),
            False,
            "random top-k basis mask null with oracle coefficients",
        ),
        "no_update_control": (
            torch.zeros_like(teacher_holdout),
            zero_support,
            False,
            "zero residual update control",
        ),
    }

    imitation_rows = [
        _with_imitation_fields(
            _arm_metrics(
                torch,
                F,
                arm,
                pred,
                support,
                teacher_holdout,
                data,
                teacher_holdout_ce,
                base_holdout_ce,
                oracle,
                note,
                effective_basis_size,
                top_k,
            ),
            torch,
            mask=_mask_for_arm(arm, oracle_mask, baseline_mask, enhanced_mask, combo_mask, random_mask),
            oracle_mask=oracle_mask,
            coeff=_coeff_for_arm(arm, holdout_coeff, baseline_coeff(data["x_holdout"]), enhanced_coeff, combo_coeff),
            oracle_coeff=holdout_coeff,
            no_update_r2=None,
        )
        for arm, (pred, support, oracle, note) in arms.items()
    ]
    _fill_gain_fields(imitation_rows)

    gate_rows = _gate_rows(source_rows, imitation_rows)
    runtime_failures = [row for row in gate_rows if row["required"] and not row["passed"]]
    scientific_failures = [row for row in gate_rows if row["gate_type"] == "scientific" and not row["passed"]]
    status = "fail" if runtime_failures else "pass"
    claim_status = _claim_status(status, scientific_failures, imitation_rows)
    summary = {
        "status": status,
        "decision": DECISION if status == "pass" else FAIL_DECISION,
        "claim_status": claim_status,
        "selected_next_step": _selected_next_step(status, scientific_failures, imitation_rows),
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "backend_policy": "local CPU deployable imitation probe only; RunPod and Colab remain blocked",
        "training_executed": True,
        "teacher_trained": True,
        "seed": seed,
        "teacher_train_steps": teacher_steps,
        "baseline_train_steps": baseline_steps,
        "enhanced_train_steps": enhanced_steps,
        "combo_train_steps": combo_steps,
        "control_train_steps": control_steps,
        "basis_size": effective_basis_size,
        "top_k": top_k,
        "hidden_dim": hidden_dim,
        "base_holdout_ce": round(base_holdout_ce, 6),
        "dense_teacher_holdout_ce": round(teacher_holdout_ce, 6),
        "source_rows": source_rows,
        "imitation_rows": imitation_rows,
        "gate_criteria": gate_rows,
        "failures": runtime_failures + scientific_failures,
        "strategy_review_handling": (
            "Accepted the major GPT-5.5-Pro pivot and the prior oracle-feasibility result. "
            "This run only tests deployable router/scalar imitation locally; GPU remains blocked."
        ),
        "ben_notification_recommended": _review_field(review_text, "notify_ben").lower() == "true",
        "strategic_change_level": _review_field(review_text, "strategic_change_level"),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _train_joint_mlp_imitation(
    torch: Any,
    F: Any,
    x_train: Any,
    coeff: Any,
    mask: Any,
    input_dim: int,
    basis_size: int,
    *,
    hidden_dim: int,
    steps: int,
) -> Any:
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 2 * basis_size),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    pos_weight = ((1.0 - mask.mean(dim=0)).clamp_min(0.05) / mask.mean(dim=0).clamp_min(0.05)).detach()
    for _ in range(steps):
        logits, pred_coeff = _joint_outputs(model, x_train, basis_size)
        mask_loss = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=pos_weight)
        coeff_loss = F.mse_loss(pred_coeff, coeff)
        active_coeff_loss = F.mse_loss(pred_coeff * mask, coeff * mask)
        loss = mask_loss + coeff_loss + 0.5 * active_coeff_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def _train_combo_mlp_imitation(
    torch: Any,
    F: Any,
    x_train: Any,
    coeff: Any,
    mask: Any,
    input_dim: int,
    basis_size: int,
    top_k: int,
    *,
    hidden_dim: int,
    steps: int,
) -> tuple[Any, list[tuple[int, ...]]]:
    combos = _support_combinations(basis_size, top_k)
    labels = _combo_labels(torch, mask, combos)
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, len(combos) + basis_size),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.015)
    for _ in range(steps):
        combo_logits, pred_coeff = _combo_outputs(model, x_train, basis_size, len(combos))
        combo_loss = F.cross_entropy(combo_logits, labels)
        coeff_loss = F.mse_loss(pred_coeff, coeff)
        active_coeff_loss = F.mse_loss(pred_coeff * mask, coeff * mask)
        loss = combo_loss + 0.75 * coeff_loss + 0.75 * active_coeff_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, combos


def _joint_outputs(model: Any, x: Any, basis_size: int) -> tuple[Any, Any]:
    out = model(x)
    return out[:, :basis_size], out[:, basis_size:]


def _combo_outputs(model: Any, x: Any, basis_size: int, combo_count: int) -> tuple[Any, Any]:
    out = model(x)
    return out[:, :combo_count], out[:, combo_count : combo_count + basis_size]


def _support_combinations(basis_size: int, top_k: int) -> list[tuple[int, ...]]:
    combos: list[tuple[int, ...]] = []

    def build(start: int, chosen: list[int]) -> None:
        if len(chosen) == top_k:
            combos.append(tuple(chosen))
            return
        remaining = top_k - len(chosen)
        for idx in range(start, basis_size - remaining + 1):
            build(idx + 1, chosen + [idx])

    build(0, [])
    return combos


def _combo_labels(torch: Any, mask: Any, combos: list[tuple[int, ...]]) -> Any:
    lookup = {combo: index for index, combo in enumerate(combos)}
    labels = []
    for row in mask:
        combo = tuple(sorted(int(idx) for idx in torch.nonzero(row > 0.0, as_tuple=False).flatten().tolist()))
        labels.append(lookup[combo])
    return torch.tensor(labels, dtype=torch.long, device=mask.device)


def _combo_mask_from_logits(torch: Any, logits: Any, combos: list[tuple[int, ...]], basis_size: int) -> Any:
    labels = logits.argmax(dim=1).tolist()
    mask = torch.zeros(logits.shape[0], basis_size, device=logits.device)
    for row, label in enumerate(labels):
        for idx in combos[int(label)]:
            mask[row, idx] = 1.0
    return mask


def _mask_for_arm(arm: str, oracle_mask: Any, baseline_mask: Any, enhanced_mask: Any, combo_mask: Any, random_mask: Any) -> Any:
    if arm in {"oracle_topk_orthogonal_sparse_coding", "oracle_support_learned_combo_coeff_sparse_coding"}:
        return oracle_mask
    if arm == "learned_combo_support_oracle_coeff_sparse_coding":
        return combo_mask
    if arm == "baseline_linear_router_scalar_imitation":
        return baseline_mask
    if arm == "enhanced_joint_mlp_router_scalar_imitation":
        return enhanced_mask
    if arm in {"combo_mlp_router_scalar_imitation", "same_router_flat_value_control"}:
        return combo_mask
    if arm == "random_topk_sparse_coding_null":
        return random_mask
    return oracle_mask * 0.0


def _coeff_for_arm(arm: str, oracle_coeff: Any, baseline_coeff: Any, enhanced_coeff: Any, combo_coeff: Any) -> Any:
    if arm in {"oracle_topk_orthogonal_sparse_coding", "learned_combo_support_oracle_coeff_sparse_coding", "random_topk_sparse_coding_null"}:
        return oracle_coeff
    if arm == "baseline_linear_router_scalar_imitation":
        return baseline_coeff
    if arm == "enhanced_joint_mlp_router_scalar_imitation":
        return enhanced_coeff
    if arm in {"combo_mlp_router_scalar_imitation", "oracle_support_learned_combo_coeff_sparse_coding", "same_router_flat_value_control"}:
        return combo_coeff
    return oracle_coeff * 0.0


def _with_imitation_fields(
    row: dict[str, Any],
    torch: Any,
    *,
    mask: Any,
    oracle_mask: Any,
    coeff: Any,
    oracle_coeff: Any,
    no_update_r2: float | None,
) -> dict[str, Any]:
    exact_overlap = float(((mask > 0.0) == (oracle_mask > 0.0)).float().mean().item())
    selected_overlap = float(((mask > 0.0) & (oracle_mask > 0.0)).float().sum(dim=1).mean().item() / max(1, int(oracle_mask.sum(dim=1).max().item())))
    coeff_error = (coeff - oracle_coeff) ** 2
    active_error = coeff_error * oracle_mask
    active_denom = oracle_mask.sum().clamp_min(1.0)
    coeff_cosine = torch.nn.functional.cosine_similarity(coeff, oracle_coeff, dim=1).mean()
    row["oracle_mask_exact_cell_overlap"] = round(exact_overlap, 6)
    row["oracle_selected_component_overlap"] = round(selected_overlap, 6)
    row["coefficient_mse_vs_oracle"] = round(float(coeff_error.mean().item()), 6)
    row["oracle_active_coefficient_mse"] = round(float(active_error.sum().item() / active_denom.item()), 6)
    row["coefficient_cosine_vs_oracle"] = round(float(coeff_cosine.item()), 6)
    row["oracle_gain_retained_fraction"] = ""
    row["no_update_relative_r2_gain"] = ""
    return row


def _fill_gain_fields(rows: list[dict[str, Any]]) -> None:
    by_arm = {row["arm"]: row for row in rows}
    no_update_r2 = _float(by_arm.get("no_update_control", {}).get("teacher_residual_reconstruction_r2"), 0.0)
    oracle_r2 = _float(by_arm.get("oracle_topk_orthogonal_sparse_coding", {}).get("teacher_residual_reconstruction_r2"), no_update_r2)
    oracle_gain = oracle_r2 - no_update_r2
    for row in rows:
        r2 = _float(row.get("teacher_residual_reconstruction_r2"), no_update_r2)
        gain = r2 - no_update_r2
        row["no_update_relative_r2_gain"] = round(gain, 6)
        row["oracle_gain_retained_fraction"] = round(gain / oracle_gain, 6) if oracle_gain > 0.0 else ""


def _gate_rows(source_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_arm = {row["arm"]: row for row in rows}
    oracle = by_arm.get("oracle_topk_orthogonal_sparse_coding", {})
    oracle_support_learned_coeff = by_arm.get("oracle_support_learned_combo_coeff_sparse_coding", {})
    learned_support_oracle_coeff = by_arm.get("learned_combo_support_oracle_coeff_sparse_coding", {})
    baseline = by_arm.get("baseline_linear_router_scalar_imitation", {})
    enhanced = by_arm.get("enhanced_joint_mlp_router_scalar_imitation", {})
    combo = by_arm.get("combo_mlp_router_scalar_imitation", {})
    flat = by_arm.get("same_router_flat_value_control", {})
    random_null = by_arm.get("random_topk_sparse_coding_null", {})
    no_update = by_arm.get("no_update_control", {})
    enhanced_retention = _float(enhanced.get("oracle_gain_retained_fraction"), -math.inf)
    combo_retention = _float(combo.get("oracle_gain_retained_fraction"), -math.inf)
    baseline_retention = _float(baseline.get("oracle_gain_retained_fraction"), -math.inf)
    oracle_support_learned_coeff_retention = _float(oracle_support_learned_coeff.get("oracle_gain_retained_fraction"), -math.inf)
    learned_support_oracle_coeff_retention = _float(learned_support_oracle_coeff.get("oracle_gain_retained_fraction"), -math.inf)
    best_deployable_retention = max(enhanced_retention, combo_retention)
    best_deployable_r2 = max(
        _float(enhanced.get("teacher_residual_reconstruction_r2"), -math.inf),
        _float(combo.get("teacher_residual_reconstruction_r2"), -math.inf),
    )
    best_deployable_ce = min(_float(enhanced.get("ce"), math.inf), _float(combo.get("ce"), math.inf))
    return [
        _gate("oracle_feasibility_source_present", bool(source_rows[0].get("present")), True, "runtime", str(source_rows[0])),
        _gate("oracle_feasibility_selected_router_imitation", "router/scalar imitation" in str(source_rows[0].get("selected_next_step", "")), True, "runtime", str(source_rows[0].get("selected_next_step", ""))),
        _gate("strategy_review_present", bool(source_rows[1].get("present")), True, "runtime", str(source_rows[1])),
        _gate("required_arms_present", set(ARMS).issubset(by_arm), True, "runtime", ",".join(sorted(by_arm))),
        _gate("gpu_blocked", True, True, "runtime", "requires_gpu_now=false; advance_to_gpu_validation=false"),
        _gate("oracle_sparse_still_feasible", _float(oracle.get("teacher_residual_reconstruction_r2"), -math.inf) >= 0.5, False, "scientific", f"oracle_r2={oracle.get('teacher_residual_reconstruction_r2')}"),
        _gate("enhanced_improves_linear_imitation", enhanced_retention > baseline_retention + 0.05, False, "scientific", f"enhanced_retention={enhanced_retention:.6f}; baseline_retention={baseline_retention:.6f}"),
        _gate("combo_improves_linear_imitation", combo_retention > baseline_retention + 0.05, False, "scientific", f"combo_retention={combo_retention:.6f}; baseline_retention={baseline_retention:.6f}"),
        _gate("enhanced_beats_random_null", _float(enhanced.get("teacher_residual_reconstruction_r2"), -math.inf) > _float(random_null.get("teacher_residual_reconstruction_r2"), -math.inf) + 0.05, False, "scientific", f"enhanced_r2={enhanced.get('teacher_residual_reconstruction_r2')}; random_r2={random_null.get('teacher_residual_reconstruction_r2')}"),
        _gate("combo_beats_random_null", _float(combo.get("teacher_residual_reconstruction_r2"), -math.inf) > _float(random_null.get("teacher_residual_reconstruction_r2"), -math.inf) + 0.05, False, "scientific", f"combo_r2={combo.get('teacher_residual_reconstruction_r2')}; random_r2={random_null.get('teacher_residual_reconstruction_r2')}"),
        _gate("best_deployable_retains_oracle_gain", best_deployable_retention >= 0.8, False, "scientific", f"best_retention={best_deployable_retention:.6f}; enhanced={enhanced_retention:.6f}; combo={combo_retention:.6f}; required=0.800000"),
        _gate("best_deployable_near_flat_control", best_deployable_r2 >= _float(flat.get("teacher_residual_reconstruction_r2"), -math.inf) - 0.1, False, "scientific", f"best_deployable_r2={best_deployable_r2:.6f}; flat_r2={flat.get('teacher_residual_reconstruction_r2')}"),
        _gate("best_deployable_beats_no_update_ce", best_deployable_ce < _float(no_update.get("ce"), -math.inf), False, "scientific", f"best_deployable_ce={best_deployable_ce:.6f}; no_update_ce={no_update.get('ce')}"),
        _gate("oracle_support_learned_coeff_retains_oracle_gain", oracle_support_learned_coeff_retention >= 0.8, False, "scientific", f"retention={oracle_support_learned_coeff_retention:.6f}; coefficient_mse={oracle_support_learned_coeff.get('coefficient_mse_vs_oracle')}; required=0.800000"),
        _gate("learned_support_oracle_coeff_retains_oracle_gain", learned_support_oracle_coeff_retention >= 0.8, False, "scientific", f"retention={learned_support_oracle_coeff_retention:.6f}; selected_overlap={learned_support_oracle_coeff.get('oracle_selected_component_overlap')}; required=0.800000"),
    ]


def _claim_status(status: str, failures: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    if status != "pass":
        return "deployable_sparse_coding_imitation_runtime_failed_closed"
    failed = {row["criterion"] for row in failures}
    if "oracle_support_learned_coeff_retains_oracle_gain" in failed and "learned_support_oracle_coeff_retains_oracle_gain" in failed:
        return "support_and_coefficient_bottlenecks_block_gpu"
    if "oracle_support_learned_coeff_retains_oracle_gain" in failed:
        return "coefficient_value_bottleneck_blocks_gpu"
    if "learned_support_oracle_coeff_retains_oracle_gain" in failed:
        return "support_routing_bottleneck_blocks_gpu"
    if not failures:
        return "deployable_sparse_coding_imitation_clears_local_gates_no_gpu_yet"
    if "best_deployable_retains_oracle_gain" in failed:
        return "support_coefficient_interaction_bottleneck_blocks_gpu"
    return "deployable_sparse_coding_imitation_local_gates_block_gpu"


def _selected_next_step(status: str, failures: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    if status != "pass":
        return "repair deployable sparse-coding imitation probe runtime artifacts before interpretation"
    failed = {row["criterion"] for row in failures}
    if "oracle_support_learned_coeff_retains_oracle_gain" in failed and "learned_support_oracle_coeff_retains_oracle_gain" in failed:
        return "decompose support and coefficient failures with a richer support-conditioned sparse value model before GPU"
    if "oracle_support_learned_coeff_retains_oracle_gain" in failed:
        return "improve sparse coefficient/value model under oracle support before GPU"
    if "learned_support_oracle_coeff_retains_oracle_gain" in failed:
        return "improve deployable support routing for the oracle sparse-coding basis before GPU"
    if "best_deployable_retains_oracle_gain" in failed:
        return "train a support-conditioned sparse coefficient/value head for the combo support before GPU"
    if "best_deployable_near_flat_control" in failed:
        return "inspect flat-control gap before any backend validation"
    if failures:
        return "inspect deployable sparse-coding guardrail failures before any backend validation"
    return "prepare a local multi-seed deployable sparse-coding imitation confirmation before considering GPU"


def _gate(criterion: str, passed: bool, required: bool, gate_type: str, evidence: str) -> dict[str, Any]:
    return {"criterion": criterion, "passed": bool(passed), "required": required, "gate_type": gate_type, "evidence": evidence}


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "imitation_rows.csv", summary["imitation_rows"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
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
            "# Dense-Teacher Deployable Sparse-Coding Imitation Probe",
            "",
            f"Decision: `{summary['decision']}`.",
            f"Claim status: `{summary['claim_status']}`.",
            "",
            "This is local CPU evidence only. GPU validation remains blocked.",
            f"Ben notification recommended by strategy review: `{summary['ben_notification_recommended']}`.",
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
    parser.add_argument("--oracle-feasibility", type=Path, default=DEFAULT_ORACLE_FEASIBILITY)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--teacher-steps", type=int, default=100)
    parser.add_argument("--baseline-steps", type=int, default=80)
    parser.add_argument("--enhanced-steps", type=int, default=200)
    parser.add_argument("--combo-steps", type=int, default=260)
    parser.add_argument("--control-steps", type=int, default=80)
    parser.add_argument("--basis-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--data-column-count", type=int, default=6)
    args = parser.parse_args(argv)
    summary = run_dense_teacher_deployable_sparse_coding_imitation_probe(
        oracle_feasibility_path=args.oracle_feasibility,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
        seed=args.seed,
        teacher_steps=args.teacher_steps,
        baseline_steps=args.baseline_steps,
        enhanced_steps=args.enhanced_steps,
        combo_steps=args.combo_steps,
        control_steps=args.control_steps,
        basis_size=args.basis_size,
        top_k=args.top_k,
        hidden_dim=args.hidden_dim,
        data_column_count=args.data_column_count,
    )
    print(json.dumps({key: summary[key] for key in ("status", "decision", "claim_status", "selected_next_step")}, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
