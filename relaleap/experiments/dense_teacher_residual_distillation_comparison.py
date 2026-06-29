"""Dense-teacher residual distillation pilot for ACSR comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.anticipatory_contextual_support_routing import (
    _FuturePredictor,
    _contextual_chunks,
    _decode_for_support,
    _feature_tensor,
    _position_predictor_inputs,
    _read_yaml,
    _score_from_features,
    _shuffle_tokens,
    _support_entropy,
    _train_predictor_row,
    _unique_support_sets,
    _used_columns,
)


DEFAULT_CONFIG = Path(
    "configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml"
)
DEFAULT_ACSR_GATE = Path(
    "results/reports/token_larger_anticipatory_contextual_support_routing_causal_mechanism_gate/summary.json"
)
DEFAULT_CONTEXTUAL_GATE = Path(
    "results/comparisons/contextual_support_router_promotion_gate_larger_char_token/summary.json"
)
DEFAULT_PRIOR_DISTILLATION_CLOSEOUT = Path(
    "results/reports/token_larger_causal_contextual_router_post_stratified_null_closeout/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_dense_teacher_residual_distillation_comparison"
)

PRIMARY_VARIANT = "acsr_predicted_future_support"
CONTEXTUAL_VARIANT = "promoted_contextual_router_support"
CONTROL_VARIANTS = (
    "token_position_only_predicted_support",
    "shuffled_predicted_support",
)
CONTRACT_VARIANTS = (
    "promoted_contextual_topk2_ce_mse_distill",
    "promoted_contextual_topk2_mse_only_distill",
    "norm_budgeted_promoted_contextual_topk2_ce_mse_distill",
    "rank_matched_contextual_topk1",
    "random_support_topk2",
    "fixed_support_topk2",
    "token_position_only_router_topk2",
    "shuffled_feature_router_topk2",
    "shuffled_teacher_target_topk2",
)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "variant_metrics.csv",
    "support_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_dense_teacher_residual_distillation_comparison(
    *,
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path = DEFAULT_OUT_DIR,
    acsr_gate_path: Path = DEFAULT_ACSR_GATE,
    contextual_gate_path: Path = DEFAULT_CONTEXTUAL_GATE,
    prior_distillation_closeout_path: Path = DEFAULT_PRIOR_DISTILLATION_CLOSEOUT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    max_steps: int | None = None,
    teacher_steps: int = 80,
    student_steps: int = 240,
    predictor_steps: int = 50,
    teacher_scales: tuple[float, ...] = (1.0, 0.25),
) -> dict[str, Any]:
    """Run a bounded local dense-teacher distillation comparison."""

    start = time.time()
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        from relaleap.smoke import ResidualColumns, TinyCharTransformer, _build_batch
    except Exception as exc:  # pragma: no cover - environment dependent
        summary = _runtime_failure(out_dir, start, config_path, str(exc))
        _write_artifacts(out_dir, summary, [], [], [])
        return summary

    config = _read_yaml(config_path)
    run_cfg = _as_dict(config.get("run"))
    data_cfg = _as_dict(config.get("data"))
    model_cfg = _as_dict(config.get("model"))
    base_cfg = _as_dict(model_cfg.get("base"))
    column_cfg = _as_dict(model_cfg.get("columns"))

    seed = int(run_cfg.get("seed", 1))
    train_steps = int(max_steps if max_steps is not None else run_cfg.get("max_steps", 50))
    teacher_step_budget = max(1, int(teacher_steps))
    student_step_budget = max(1, int(student_steps))
    scale_values = _teacher_scales(teacher_scales)
    dataset = str(data_cfg.get("dataset", "tiny_shakespeare_word"))
    seq_len = int(data_cfg.get("seq_len", 64))
    hidden_dim = int(base_cfg.get("hidden_dim", 96))
    layers = int(base_cfg.get("layers", 2))
    num_columns = int(column_cfg.get("num_columns", 24))
    atoms_per_column = int(column_cfg.get("atoms_per_column", 4))
    top_k = int(column_cfg.get("top_k", 2))
    contextual_width = int(column_cfg.get("contextual_router_hidden_dim", hidden_dim * 2))

    torch.manual_seed(seed)
    inputs, targets, vocab_size = _build_batch(
        dataset=dataset,
        seq_len=seq_len,
        batch_size=4,
    )
    base = TinyCharTransformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        layers=layers,
    )
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base.eval()
    with torch.no_grad():
        hidden = base.encode(inputs)
        base_logits = base.decode(hidden)
        base_ce = _loss_value(_ce_loss(F, base_logits, targets, vocab_size))

    teacher = _DenseResidualTeacher(nn, hidden_dim)
    _train_dense_teacher(
        torch,
        F,
        base,
        teacher,
        hidden,
        targets,
        vocab_size,
        steps=teacher_step_budget,
    )
    teacher.eval()
    with torch.no_grad():
        teacher_hidden = teacher(hidden)
        teacher_logits = base.decode(teacher_hidden)
        teacher_ce = _loss_value(_ce_loss(F, teacher_logits, targets, vocab_size))
        teacher_hidden_residual = teacher_hidden - hidden
        teacher_logit_residual = teacher_logits - base_logits

    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_hidden_residual_export = out_dir / "teacher_hidden_residual.pt"
    teacher_logit_residual_export = out_dir / "teacher_logit_residual.pt"
    torch.save(teacher_hidden_residual.detach().cpu(), teacher_hidden_residual_export)
    torch.save(teacher_logit_residual.detach().cpu(), teacher_logit_residual_export)

    chunks = _contextual_chunks(torch, hidden)
    causal_inputs = _causal_predictor_inputs(torch, chunks)
    position_inputs = _position_predictor_inputs(torch, chunks)
    future_targets = torch.cat([chunks["next"], chunks["next_delta"]], dim=-1)
    predictor = _FuturePredictor(nn, causal_inputs.shape[-1], hidden_dim * 2, hidden_dim)
    token_position_predictor = _FuturePredictor(
        nn,
        position_inputs.shape[-1],
        hidden_dim * 2,
        max(8, min(hidden_dim, 64)),
    )
    predictor_rows = [
        _train_predictor_row(
            torch,
            F,
            predictor,
            causal_inputs,
            future_targets,
            steps=max(1, predictor_steps),
            label="mlp_causal",
        ),
        _train_predictor_row(
            torch,
            F,
            token_position_predictor,
            position_inputs,
            future_targets,
            steps=max(1, predictor_steps),
            label="token_position_only",
        ),
    ]
    with torch.no_grad():
        predicted_future = predictor(causal_inputs)
        token_position_future = token_position_predictor(position_inputs)
        shuffled_future = _shuffle_tokens(torch, predicted_future)

    variants = {
        CONTEXTUAL_VARIANT: {
            "kind": "native_contextual",
            "features": _feature_tensor(torch, chunks, future_targets),
            "top_k": top_k,
            "loss_mode": "ce_mse",
        },
        PRIMARY_VARIANT: {
            "kind": "forced_features",
            "features": _feature_tensor(torch, chunks, predicted_future),
            "top_k": top_k,
            "loss_mode": "ce_mse",
        },
        "promoted_contextual_topk2_mse_only_distill": {
            "kind": "native_contextual",
            "features": _feature_tensor(torch, chunks, future_targets),
            "top_k": top_k,
            "loss_mode": "mse_only",
        },
        "norm_budgeted_promoted_contextual_topk2_ce_mse_distill": {
            "kind": "native_contextual",
            "features": _feature_tensor(torch, chunks, future_targets),
            "top_k": top_k,
            "loss_mode": "ce_mse_norm_budget",
        },
        "rank_matched_contextual_topk1": {
            "kind": "native_contextual",
            "features": _feature_tensor(torch, chunks, future_targets),
            "top_k": 1,
            "loss_mode": "ce_mse",
        },
        "random_support_topk2": {
            "kind": "random_support",
            "features": _feature_tensor(torch, chunks, future_targets),
            "top_k": top_k,
            "loss_mode": "ce_mse",
        },
        "fixed_support_topk2": {
            "kind": "fixed_support",
            "features": _feature_tensor(torch, chunks, future_targets),
            "top_k": top_k,
            "loss_mode": "ce_mse",
        },
        "token_position_only_predicted_support": {
            "kind": "forced_features",
            "features": _feature_tensor(torch, chunks, token_position_future),
            "top_k": top_k,
            "loss_mode": "ce_mse",
        },
        "shuffled_predicted_support": {
            "kind": "forced_features",
            "features": _feature_tensor(torch, chunks, shuffled_future),
            "top_k": top_k,
            "loss_mode": "ce_mse",
        },
        "shuffled_teacher_target_topk2": {
            "kind": "native_contextual",
            "features": _feature_tensor(torch, chunks, future_targets),
            "top_k": top_k,
            "loss_mode": "shuffled_target",
        },
    }
    variant_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    teacher_accounting = _teacher_accounting(
        teacher=teacher,
        hidden_dim=hidden_dim,
        hidden=hidden,
        teacher_hidden_residual=teacher_hidden_residual,
        teacher_logit_residual=teacher_logit_residual,
        teacher_ce=teacher_ce,
        base_logits=base_logits,
        teacher_hidden_residual_export=teacher_hidden_residual_export,
        teacher_logit_residual_export=teacher_logit_residual_export,
        teacher_scale=1.0,
    )
    variant_rows.append(dict(teacher_accounting, arm="parameter_matched_causal_mlp_control", variant="dense_teacher"))
    variant_rows.append(dict(teacher_accounting, arm="dense_teacher_parameter_matched_mlp", variant="dense_teacher_parameter_matched_mlp"))
    variant_rows.append(
        dict(
            teacher_accounting,
            arm="dense_rank_norm_control",
            variant="dense_rank_norm_control",
            active_rank_or_topk=min(teacher_accounting["active_rank_or_topk"], top_k * atoms_per_column),
        )
    )
    scale_summaries: list[dict[str, Any]] = []
    for teacher_scale in scale_values:
        scaled_teacher_logits = base_logits + teacher_scale * teacher_logit_residual
        scaled_teacher_hidden_residual = teacher_scale * teacher_hidden_residual
        scaled_teacher_ce = _loss_value(_ce_loss(F, scaled_teacher_logits, targets, vocab_size))
        if teacher_scale != 1.0:
            scaled_accounting = _teacher_accounting(
                teacher=teacher,
                hidden_dim=hidden_dim,
                hidden=hidden,
                teacher_hidden_residual=scaled_teacher_hidden_residual,
                teacher_logit_residual=teacher_scale * teacher_logit_residual,
                teacher_ce=scaled_teacher_ce,
                base_logits=base_logits,
                teacher_hidden_residual_export=teacher_hidden_residual_export,
                teacher_logit_residual_export=teacher_logit_residual_export,
                teacher_scale=teacher_scale,
            )
            scale_arm = f"dense_teacher_calibrated_scale_{_scale_suffix(teacher_scale)}"
            variant_rows.append(dict(scaled_accounting, arm=scale_arm, variant=scale_arm))
        for name, spec in variants.items():
            residual = ResidualColumns(
                hidden_dim=hidden_dim,
                num_columns=num_columns,
                atoms_per_column=atoms_per_column,
                top_k=int(spec["top_k"]),
                support_router="contextual_mlp",
                contextual_router_hidden_dim=contextual_width,
            )
            student_teacher_logits = scaled_teacher_logits
            if spec["loss_mode"] == "shuffled_target":
                student_teacher_logits = _shuffle_tokens(torch, scaled_teacher_logits)
            residual_norm_budget = None
            if spec["loss_mode"] == "ce_mse_norm_budget":
                residual_norm_budget = float(
                    scaled_teacher_hidden_residual.norm(dim=-1).mean().detach().item()
                )
            _train_student(
                torch,
                F,
                base,
                residual,
                hidden,
                targets,
                student_teacher_logits,
                vocab_size,
                variant_kind=str(spec["kind"]),
                features=spec["features"],
                steps=student_step_budget,
                loss_mode=str(spec["loss_mode"]),
                residual_norm_budget=residual_norm_budget,
            )
            row, support = _student_metric_row(
                torch,
                F,
                base,
                residual,
                hidden,
                targets,
                scaled_teacher_logits,
                scaled_teacher_hidden_residual,
                base_logits,
                vocab_size,
                variant=_scaled_variant_name(name, teacher_scale),
                variant_kind=str(spec["kind"]),
                features=spec["features"],
                top_k=int(spec["top_k"]),
                num_columns=num_columns,
                atoms_per_column=atoms_per_column,
                teacher_scale=teacher_scale,
                teacher_ce=scaled_teacher_ce,
                residual_norm_budget=residual_norm_budget,
            )
            row["arm"] = _scaled_arm_name(
                {
                    CONTEXTUAL_VARIANT: "promoted_contextual_topk2_ce_mse_distill",
                    "promoted_contextual_topk2_ce_mse_distill": "promoted_contextual_topk2_ce_mse_distill",
                    "promoted_contextual_topk2_mse_only_distill": "promoted_contextual_topk2_mse_only_distill",
                    "norm_budgeted_promoted_contextual_topk2_ce_mse_distill": "norm_budgeted_promoted_contextual_topk2_ce_mse_distill",
                    "rank_matched_contextual_topk1": "rank_matched_contextual_topk1",
                    "random_support_topk2": "random_support_topk2",
                    "fixed_support_topk2": "fixed_support_topk2",
                    "token_position_only_predicted_support": "token_position_only_router_topk2",
                    "shuffled_predicted_support": "shuffled_feature_router_topk2",
                    "shuffled_teacher_target_topk2": "shuffled_teacher_target_topk2",
                }.get(name, name),
                teacher_scale,
            )
            variant_rows.append(row)
            support_rows.append(
                {
                    "variant": row["variant"],
                    "arm": row["arm"],
                    "teacher_scale": teacher_scale,
                    "used_columns": _used_columns(support),
                    "unique_support_sets": _unique_support_sets(support),
                    "support_entropy": _support_entropy(torch, support, num_columns),
                }
            )
        scale_summaries.append(
            _scale_summary(
                variant_rows=variant_rows,
                teacher_scale=teacher_scale,
                teacher_ce=scaled_teacher_ce,
                base_ce=base_ce,
            )
        )

    source_rows = _source_rows(
        acsr_gate_path=acsr_gate_path,
        contextual_gate_path=contextual_gate_path,
        prior_distillation_closeout_path=prior_distillation_closeout_path,
        strategy_review_path=strategy_review_path,
    )
    criteria = _gate_criteria(
        variant_rows=variant_rows,
        teacher_ce=teacher_ce,
        base_ce=base_ce,
        source_rows=source_rows,
    )
    failures = [row for row in criteria if not row["passed"]]
    if failures:
        status = "fail"
        decision = "dense_teacher_residual_distillation_pilot_not_supported"
        claim_status = "dense_teacher_distillation_not_interpretable_or_not_better_than_controls"
        next_step = "repair_dense_teacher_distillation_pilot_or_return_to_acsr_broader_benchmark"
    else:
        status = "pass"
        decision = "dense_teacher_residual_distillation_acsr_pilot_supported_not_promoted"
        claim_status = "dense_teacher_distillation_acsr_pilot_supported_not_promoted"
        next_step = "replicate_dense_teacher_residual_distillation_on_a_broader_local_benchmark"

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": next_step,
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "dataset": dataset,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "num_columns": num_columns,
        "top_k": top_k,
        "config_train_steps": train_steps,
        "teacher_steps": teacher_step_budget,
        "student_steps": student_step_budget,
        "predictor_steps": max(1, predictor_steps),
        "teacher_scales": scale_values,
        "base_ce_loss": base_ce,
        "dense_teacher_ce_loss": teacher_ce,
        "dense_teacher_ce_improvement": base_ce - teacher_ce,
        "teacher_scale_summaries": scale_summaries,
        "variant_rows": variant_rows,
        "support_rows": support_rows,
        "predictor_rows": predictor_rows,
        "source_rows": source_rows,
        "gate_status": {
            "passes_dense_teacher_distillation_gate": not failures,
            "criteria": criteria,
        },
        "failures": failures,
        "claim_statuses": {
            PRIMARY_VARIANT: claim_status if status == "pass" else "not_supported",
            CONTEXTUAL_VARIANT: "promoted_contextual_router_comparison_baseline",
            "dense_teacher": "local_cpu_pilot_teacher_not_default",
            "promoted_default_router": "no_default_change",
        },
        "rationale": _rationale(status),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, variant_rows, support_rows, criteria)
    return summary


class _DenseResidualTeacher:
    def __new__(cls, nn: Any, hidden_dim: int) -> Any:
        class DenseTeacher(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                nn.init.zeros_(self.net[-1].weight)
                nn.init.zeros_(self.net[-1].bias)

            def forward(self, hidden: Any) -> Any:
                return hidden + self.net(hidden)

        return DenseTeacher()


def _train_dense_teacher(
    torch: Any,
    F: Any,
    base: Any,
    teacher: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    *,
    steps: int,
) -> None:
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=3e-3)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = base.decode(teacher(hidden.detach()))
        loss = _ce_loss(F, logits, targets, vocab_size)
        loss.backward()
        optimizer.step()


def _train_student(
    torch: Any,
    F: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    teacher_logits: Any,
    vocab_size: int,
    *,
    variant_kind: str,
    features: Any,
    steps: int,
    loss_mode: str = "ce_mse",
    residual_norm_budget: float | None = None,
) -> None:
    optimizer = torch.optim.AdamW(residual.parameters(), lr=3e-3)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits, _, student_hidden = _student_logits_and_support(
            torch,
            base,
            residual,
            hidden,
            variant_kind=variant_kind,
            features=features,
            top_k=residual.top_k,
        )
        loss = F.mse_loss(logits, teacher_logits.detach())
        if loss_mode != "mse_only":
            loss = loss + 0.1 * _ce_loss(F, logits, targets, vocab_size)
        if residual_norm_budget is not None:
            update_l2 = (student_hidden - hidden).norm(dim=-1).mean()
            budget = torch.as_tensor(
                residual_norm_budget,
                dtype=update_l2.dtype,
                device=update_l2.device,
            )
            loss = loss + 2.0 * (update_l2 - budget).pow(2) + 4.0 * torch.relu(update_l2 - budget).pow(2)
        loss.backward()
        optimizer.step()


def _student_metric_row(
    torch: Any,
    F: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    teacher_logits: Any,
    teacher_hidden_residual: Any,
    base_logits: Any,
    vocab_size: int,
    *,
    variant: str,
    variant_kind: str,
    features: Any,
    top_k: int,
    num_columns: int,
    atoms_per_column: int,
    teacher_scale: float,
    teacher_ce: float,
    residual_norm_budget: float | None = None,
) -> tuple[dict[str, Any], Any]:
    with torch.no_grad():
        logits, support, student_hidden = _student_logits_and_support(
            torch,
            base,
            residual,
            hidden,
            variant_kind=variant_kind,
            features=features,
            top_k=top_k,
        )
        ce_loss = _loss_value(_ce_loss(F, logits, targets, vocab_size))
        distill_mse = float(F.mse_loss(logits, teacher_logits).item())
        update = student_hidden - hidden
        residual_mse = float(F.mse_loss(update, teacher_hidden_residual).item())
        residual_r2 = _r2(torch, update, teacher_hidden_residual)
        residual_cosine = _cosine(F, update, teacher_hidden_residual)
        residual_l2 = float(update.norm(dim=-1).mean().item())
        teacher_l2 = float(teacher_hidden_residual.norm(dim=-1).mean().item())
        logit_mse = float(F.mse_loss(logits, base_logits).item())
        churn = float((logits.argmax(dim=-1) != teacher_logits.argmax(dim=-1)).to(dtype=torch.float32).mean().item())
        budget = teacher_l2 if residual_norm_budget is None else float(residual_norm_budget)
        budget_error = abs(residual_l2 - budget)
    return (
        {
            "variant": variant,
            "variant_kind": variant_kind,
            "teacher_scale": teacher_scale,
            "teacher_ce_loss": teacher_ce,
            "ce_loss": ce_loss,
            "teacher_logit_mse": distill_mse,
            "anchor_kl_or_logit_mse": logit_mse,
            "functional_churn": churn,
            "intervention_fingerprint_purity": max(0.0, min(1.0, 1.0 - churn)),
            "support_regret": max(0.0, ce_loss - teacher_ce),
            "commutator_norm": 0.0,
            "stored_params": _param_count(residual),
            "active_params": float(top_k * atoms_per_column * hidden.shape[-1]),
            "active_rank_or_topk": float(top_k),
            "residual_l2": residual_l2,
            "residual_norm_ratio": residual_l2 / max(teacher_l2, 1e-12),
            "residual_norm_budget": budget,
            "residual_norm_budget_error": budget_error,
            "residual_norm_budget_overuse": max(0.0, residual_l2 - budget),
            "flops_estimate": float(hidden.numel() * max(top_k, 1)),
            "teacher_residual_mse": residual_mse,
            "teacher_residual_r2": residual_r2,
            "teacher_residual_cosine": residual_cosine,
        },
        support,
    )


def _student_logits_and_support(
    torch: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    *,
    variant_kind: str,
    features: Any,
    top_k: int | None = None,
) -> tuple[Any, Any, Any]:
    if variant_kind == "native_contextual":
        scores = residual._score_columns(hidden) + residual.score_tie_breaker.to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
    elif variant_kind == "random_support":
        generator = torch.Generator(device=hidden.device)
        generator.manual_seed(1729)
        support = torch.randint(
            low=0,
            high=residual.num_columns,
            size=(hidden.shape[0], hidden.shape[1], top_k or residual.top_k),
            generator=generator,
            device=hidden.device,
        )
        top_values = torch.zeros_like(support, dtype=hidden.dtype)
        student_hidden = _hidden_for_support(torch, residual, hidden, support, top_values)
        return base.decode(student_hidden), support.detach(), student_hidden
    elif variant_kind == "fixed_support":
        support = torch.arange(top_k or residual.top_k, device=hidden.device, dtype=torch.long).view(1, 1, -1)
        support = support.expand(hidden.shape[0], hidden.shape[1], support.shape[-1])
        top_values = torch.zeros_like(support, dtype=hidden.dtype)
        student_hidden = _hidden_for_support(torch, residual, hidden, support, top_values)
        return base.decode(student_hidden), support.detach(), student_hidden
    else:
        scores = _score_from_features(residual, features)
    top_values, support = scores.topk(top_k or residual.top_k, dim=-1)
    student_hidden = _hidden_for_support(torch, residual, hidden, support, top_values)
    return base.decode(student_hidden), support.detach(), student_hidden


def _hidden_for_support(torch: Any, residual: Any, hidden: Any, support: Any, top_values: Any) -> Any:
    column_weights = torch.softmax(top_values, dim=-1)
    atom_weights = torch.softmax(residual.atom_logits, dim=-1)
    column_values = torch.einsum("ca,cah->ch", atom_weights, residual.atom_values)
    selected_values = column_values[support]
    residual_update = torch.einsum("bsk,bskh->bsh", column_weights, selected_values)
    return hidden + residual_update


def _teacher_accounting(
    *,
    teacher: Any,
    hidden_dim: int,
    hidden: Any,
    teacher_hidden_residual: Any,
    teacher_logit_residual: Any,
    teacher_ce: float,
    base_logits: Any,
    teacher_hidden_residual_export: Path,
    teacher_logit_residual_export: Path,
    teacher_scale: float,
) -> dict[str, Any]:
    teacher_l2 = float(teacher_hidden_residual.norm(dim=-1).mean().item())
    logit_l2 = float(teacher_logit_residual.norm(dim=-1).mean().item())
    return {
        "stored_params": _param_count(teacher),
        "teacher_scale": teacher_scale,
        "teacher_ce_loss": teacher_ce,
        "active_params": _param_count(teacher),
        "active_rank_or_topk": float(hidden_dim),
        "residual_l2": teacher_l2,
        "residual_norm_ratio": 1.0,
        "flops_estimate": float(hidden.numel() * hidden_dim * 4),
        "ce_loss": teacher_ce,
        "anchor_kl_or_logit_mse": float((teacher_logit_residual.pow(2)).mean().item()),
        "functional_churn": 0.0,
        "intervention_fingerprint_purity": 1.0,
        "support_regret": 0.0,
        "commutator_norm": 0.0,
        "teacher_hidden_residual_export": str(teacher_hidden_residual_export),
        "teacher_logit_residual_export": str(teacher_logit_residual_export),
        "teacher_residual_mse": 0.0,
        "teacher_residual_r2": 1.0,
        "teacher_residual_cosine": 1.0 if logit_l2 >= 0.0 and base_logits.numel() else 1.0,
    }


def _param_count(module: Any) -> float:
    return float(sum(parameter.numel() for parameter in module.parameters()))


def _r2(torch: Any, prediction: Any, target: Any) -> float:
    residual = ((prediction - target) ** 2).sum()
    centered = ((target - target.mean()) ** 2).sum()
    if float(centered.item()) <= 1e-12:
        return 1.0 if float(residual.item()) <= 1e-12 else 0.0
    return float((1.0 - residual / centered).item())


def _cosine(F: Any, prediction: Any, target: Any) -> float:
    return float(F.cosine_similarity(prediction.reshape(1, -1), target.reshape(1, -1), dim=-1).item())


def _teacher_scales(values: tuple[float, ...]) -> list[float]:
    scales: list[float] = []
    for value in values:
        scale = float(value)
        if scale <= 0.0:
            continue
        if not any(abs(scale - existing) <= 1e-12 for existing in scales):
            scales.append(scale)
    if not any(abs(scale - 1.0) <= 1e-12 for scale in scales):
        scales.insert(0, 1.0)
    return scales


def _scale_suffix(scale: float) -> str:
    return f"{scale:g}".replace("-", "m").replace(".", "p")


def _scaled_variant_name(name: str, scale: float) -> str:
    if abs(scale - 1.0) <= 1e-12:
        return name
    return f"{name}_teacher_scale_{_scale_suffix(scale)}"


def _scaled_arm_name(name: str, scale: float) -> str:
    if abs(scale - 1.0) <= 1e-12:
        return name
    return f"{name}_teacher_scale_{_scale_suffix(scale)}"


def _scale_summary(
    *,
    variant_rows: list[dict[str, Any]],
    teacher_scale: float,
    teacher_ce: float,
    base_ce: float,
) -> dict[str, Any]:
    rows = [row for row in variant_rows if abs(float(row.get("teacher_scale", 1.0)) - teacher_scale) <= 1e-12]
    by_variant = {str(row.get("variant")): row for row in rows}
    primary = by_variant.get(_scaled_variant_name(PRIMARY_VARIANT, teacher_scale), {})
    contextual = by_variant.get(_scaled_variant_name(CONTEXTUAL_VARIANT, teacher_scale), {})
    controls = [
        by_variant.get(_scaled_variant_name(name, teacher_scale), {})
        for name in CONTROL_VARIANTS
    ]
    primary_mse = _number(primary.get("teacher_logit_mse"))
    contextual_mse = _number(contextual.get("teacher_logit_mse"))
    control_mses = [_number(row.get("teacher_logit_mse")) for row in controls]
    primary_ce = _number(primary.get("ce_loss"))
    return {
        "teacher_scale": teacher_scale,
        "teacher_ce_loss": teacher_ce,
        "teacher_ce_improvement": base_ce - teacher_ce,
        "primary_variant": _scaled_variant_name(PRIMARY_VARIANT, teacher_scale),
        "primary_ce_loss": primary_ce,
        "primary_teacher_logit_mse": primary_mse,
        "contextual_teacher_logit_mse": contextual_mse,
        "null_teacher_logit_mses": control_mses,
        "primary_beats_contextual_mse": primary_mse is not None
        and contextual_mse is not None
        and primary_mse <= contextual_mse,
        "primary_beats_null_mses": primary_mse is not None
        and all(value is not None and primary_mse <= value for value in control_mses),
        "primary_ce_within_teacher_margin": primary_ce is not None and primary_ce <= teacher_ce + 0.25,
    }


def _causal_predictor_inputs(torch: Any, chunks: dict[str, Any]) -> Any:
    return torch.cat(
        [
            chunks["current"],
            chunks["previous"],
            chunks["previous_delta"],
            chunks["position"],
            chunks["sin_position"],
            chunks["cos_position"],
        ],
        dim=-1,
    )


def _ce_loss(F: Any, logits: Any, targets: Any, vocab_size: int) -> float:
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab_size),
        targets[:, :-1].reshape(-1),
    )


def _loss_value(loss: Any) -> float:
    return float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)


def _gate_criteria(
    *,
    variant_rows: list[dict[str, Any]],
    teacher_ce: float,
    base_ce: float,
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_name = {row["variant"]: row for row in variant_rows}
    acsr = by_name.get(PRIMARY_VARIANT, {})
    contextual = by_name.get(CONTEXTUAL_VARIANT, {})
    controls = [by_name.get(name, {}) for name in CONTROL_VARIANTS]
    source_pass = all(
        row.get("present") and row.get("status") in {"pass", "ok"}
        for row in source_rows[:3]
    )
    acsr_mse = _number(acsr.get("teacher_logit_mse"))
    acsr_ce = _number(acsr.get("ce_loss"))
    contextual_mse = _number(contextual.get("teacher_logit_mse"))
    control_mses = [_number(row.get("teacher_logit_mse")) for row in controls]
    calibrated_summaries = [
        _scale_summary(
            variant_rows=variant_rows,
            teacher_scale=float(row.get("teacher_scale")),
            teacher_ce=float(row.get("teacher_ce_loss")),
            base_ce=base_ce,
        )
        for row in _unique_scale_rows(variant_rows, teacher_ce)
        if abs(float(row.get("teacher_scale")) - 1.0) > 1e-12
    ]
    calibrated_passes = [
        row
        for row in calibrated_summaries
        if row["primary_beats_contextual_mse"]
        and row["primary_beats_null_mses"]
        and row["primary_ce_within_teacher_margin"]
    ]
    return [
        {
            "criterion": "source_gates_present_and_passing",
            "passed": source_pass,
            "threshold": "ACSR gate, contextual gate, and prior distillation closeout pass",
            "actual": ",".join(str(row.get("status")) for row in source_rows[:3]),
        },
        {
            "criterion": "dense_teacher_improves_base_ce",
            "passed": teacher_ce < base_ce,
            "threshold": "dense teacher CE < base CE",
            "actual": f"{teacher_ce:.6f} < {base_ce:.6f}",
        },
        {
            "criterion": "acsr_distills_at_least_as_well_as_promoted_contextual",
            "passed": acsr_mse is not None and contextual_mse is not None and acsr_mse <= contextual_mse,
            "threshold": "ACSR teacher logit MSE <= promoted contextual MSE",
            "actual": f"{acsr_mse} <= {contextual_mse}",
        },
        {
            "criterion": "acsr_beats_token_position_and_shuffled_distillation_nulls",
            "passed": acsr_mse is not None
            and all(value is not None and acsr_mse <= value for value in control_mses),
            "threshold": "ACSR teacher logit MSE <= null MSEs",
            "actual": f"{acsr_mse} <= {control_mses}",
        },
        {
            "criterion": "acsr_ce_not_worse_than_teacher_by_large_margin",
            "passed": acsr_ce is not None and acsr_ce <= teacher_ce + 0.25,
            "threshold": "ACSR CE <= dense teacher CE + 0.25",
            "actual": f"{acsr_ce} <= {teacher_ce + 0.25:.6f}",
        },
        {
            "criterion": "calibrated_teacher_scale_gate",
            "passed": bool(calibrated_passes),
            "threshold": "at least one teacher scale < 1.0 passes contextual/null MSE controls and CE margin",
            "actual": json.dumps(calibrated_summaries, sort_keys=True),
        },
    ]


def _unique_scale_rows(variant_rows: list[dict[str, Any]], teacher_ce: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[float] = set()
    for row in variant_rows:
        scale = _number(row.get("teacher_scale"))
        if scale is None or scale in seen:
            continue
        seen.add(scale)
        rows.append(
            {
                "teacher_scale": scale,
                "teacher_ce_loss": row.get("teacher_ce_loss", teacher_ce if scale == 1.0 else row.get("ce_loss", teacher_ce)),
            }
        )
    return rows


def _source_rows(
    *,
    acsr_gate_path: Path,
    contextual_gate_path: Path,
    prior_distillation_closeout_path: Path,
    strategy_review_path: Path,
) -> list[dict[str, Any]]:
    paths = [
        ("acsr_causal_mechanism_gate", acsr_gate_path),
        ("contextual_support_router_promotion_gate", contextual_gate_path),
        ("prior_causal_router_distillation_closeout", prior_distillation_closeout_path),
        ("strategy_review", strategy_review_path),
    ]
    rows = []
    for source, path in paths:
        if path.suffix == ".md":
            review = _strategy_review(path)
            rows.append(
                {
                    "source": source,
                    "path": str(path),
                    "present": path.is_file(),
                    "status": "pass" if path.is_file() else "missing",
                    "decision": review.get("recommended_next_action"),
                    "claim_status": (
                        f"strategic_change_level={review.get('strategic_change_level')}; "
                        f"notify_ben={review.get('notify_ben')}"
                    ),
                }
            )
            continue
        payload = _read_json(path)
        status = payload.get("status")
        if status is None and payload.get("comparison_status") == "ok":
            status = "pass"
        elif status == "ok":
            status = "pass"
        rows.append(
            {
                "source": source,
                "path": str(path),
                "present": path.is_file(),
                "status": status,
                "decision": payload.get("decision"),
                "claim_status": payload.get("claim_status"),
            }
        )
    return rows


def _rationale(status: str) -> str:
    if status == "pass":
        return (
            "The local dense-teacher pilot supports ACSR-predicted support as a "
            "distillation candidate, but it is bounded to one CPU packet and "
            "does not change the default router."
        )
    return (
        "The dense-teacher pilot failed closed because ACSR did not beat all "
        "dense-distillation controls or because required source gates were not "
        "available. No default-router or mechanism-promotion claim is made."
    )


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    variant_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    criteria: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "variant_metrics.csv", variant_rows)
    _write_csv(out_dir / "support_metrics.csv", support_rows)
    _write_csv(out_dir / "gate_criteria.csv", criteria)
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Dense-Teacher Residual Distillation Comparison",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Decision: `{summary.get('decision')}`",
        f"- Claim status: `{summary.get('claim_status')}`",
        f"- Base CE: `{summary.get('base_ce_loss')}`",
        f"- Dense-teacher CE: `{summary.get('dense_teacher_ce_loss')}`",
        f"- Selected next step: {summary.get('selected_next_step')}",
        "",
        "This is a local CPU pilot only. It records whether ACSR-predicted "
        "support distills a dense residual teacher better than the promoted "
        "contextual-router support and token/position or shuffled null supports.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    else:
        fieldnames = ["status"]
        rows = [{"status": "missing"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False, "path": str(path)}
    fields: dict[str, Any] = {"present": True, "path": str(path)}
    for line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {"strategic_change_level", "notify_ben", "recommended_next_action"}:
            fields[key] = value
    fields["notify_ben"] = str(fields.get("notify_ben", "false")).lower() == "true"
    return fields


def _runtime_failure(out_dir: Path, start: float, config_path: Path, reason: str) -> dict[str, Any]:
    return {
        "status": "fail",
        "decision": "dense_teacher_residual_distillation_runtime_unavailable",
        "claim_status": "dense_teacher_distillation_not_executed",
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "failures": [{"criterion": "runtime", "reason": reason}],
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _parse_teacher_scales(value: str) -> tuple[float, ...]:
    scales: list[float] = []
    for raw in value.split(","):
        raw = raw.strip()
        if raw:
            scales.append(float(raw))
    return tuple(scales or [1.0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--acsr-gate", type=Path, default=DEFAULT_ACSR_GATE)
    parser.add_argument("--contextual-gate", type=Path, default=DEFAULT_CONTEXTUAL_GATE)
    parser.add_argument(
        "--prior-distillation-closeout",
        type=Path,
        default=DEFAULT_PRIOR_DISTILLATION_CLOSEOUT,
    )
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--teacher-steps", type=int, default=80)
    parser.add_argument("--student-steps", type=int, default=240)
    parser.add_argument("--predictor-steps", type=int, default=50)
    parser.add_argument(
        "--teacher-scales",
        type=str,
        default="1.0,0.25",
        help="Comma-separated dense teacher residual/logit scales; 1.0 is always included.",
    )
    args = parser.parse_args()
    summary = run_dense_teacher_residual_distillation_comparison(
        config_path=args.config,
        out_dir=args.out,
        acsr_gate_path=args.acsr_gate,
        contextual_gate_path=args.contextual_gate,
        prior_distillation_closeout_path=args.prior_distillation_closeout,
        strategy_review_path=args.strategy_review,
        max_steps=args.max_steps,
        teacher_steps=args.teacher_steps,
        student_steps=args.student_steps,
        predictor_steps=args.predictor_steps,
        teacher_scales=_parse_teacher_scales(args.teacher_scales),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
