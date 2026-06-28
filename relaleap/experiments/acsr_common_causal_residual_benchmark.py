"""Common-baseline sparse-vs-dense causal residual benchmark for ACSR."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from relaleap.experiments.acsr_dense_residual_transfer_control import (
    DEFAULT_SOURCE_PROBE,
    _LowRankCausalDenseAdapter,
    _criterion,
    _float_or_none,
    _heldout_mask,
    _parameter_count,
    _per_token_ce,
)
from relaleap.experiments.acsr_transfer_objective_probe import DEFAULT_CONFIG
from relaleap.experiments.anticipatory_contextual_support_routing import (
    _causal_predictor_inputs,
    _contextual_chunks,
    _frequency_matched_random_support,
    _position_predictor_inputs,
    _support_entropy,
    _support_eval_metrics,
    _support_jaccard,
)


DEFAULT_OUT_DIR = Path("results/reports/acsr_common_causal_residual_benchmark")
REQUIRED_ARTIFACTS = (
    "summary.json",
    "arm_metrics.csv",
    "per_token_metrics.csv",
    "norm_sweep.csv",
    "intervention_fingerprints.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_acsr_common_causal_residual_benchmark(
    *,
    source_probe_dir: Path = DEFAULT_SOURCE_PROBE,
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path = DEFAULT_OUT_DIR,
    train_steps: int = 12,
    dense_steps: int = 80,
) -> dict[str, Any]:
    """Train/evaluate sparse and dense arms on one local held-out packet."""

    start = time.time()
    source_summary = _read_json(source_probe_dir / "summary.json")
    source_rows = _read_csv(source_probe_dir / "per_token_metrics.csv")
    preflight = _preflight_rows(source_probe_dir, source_summary, source_rows, config_path)
    if any(not row["passed"] for row in preflight):
        summary = _summary(
            status="fail",
            decision="acsr_common_causal_residual_benchmark_failed_closed",
            claim_status="benchmark_not_run",
            start=start,
            source_probe_dir=source_probe_dir,
            config_path=config_path,
            train_steps=train_steps,
            dense_steps=dense_steps,
            arm_rows=[],
            per_token_rows=[],
            norm_rows=[],
            fingerprint_rows=[],
            gate_rows=preflight,
            out_dir=out_dir,
        )
        _write_artifacts(out_dir, summary, [], [], [], [], preflight)
        return summary

    try:
        arm_rows, per_token_rows, norm_rows, fingerprint_rows = _run_benchmark(
            config_path=config_path,
            train_steps=train_steps,
            dense_steps=dense_steps,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        gate_rows = preflight + [
            _criterion(
                "benchmark_runtime",
                False,
                "common benchmark training/evaluation completes",
                str(exc),
                "common sparse-vs-dense benchmark could not run",
            )
        ]
        summary = _summary(
            status="fail",
            decision="acsr_common_causal_residual_benchmark_failed_closed",
            claim_status="benchmark_runtime_failed",
            start=start,
            source_probe_dir=source_probe_dir,
            config_path=config_path,
            train_steps=train_steps,
            dense_steps=dense_steps,
            arm_rows=[],
            per_token_rows=[],
            norm_rows=[],
            fingerprint_rows=[],
            gate_rows=gate_rows,
            out_dir=out_dir,
        )
        _write_artifacts(out_dir, summary, [], [], [], [], gate_rows)
        return summary

    gate_rows = preflight + _benchmark_gate_rows(arm_rows, fingerprint_rows)
    status = "pass" if all(row["passed"] for row in gate_rows) else "fail"
    summary = _summary(
        status=status,
        decision=(
            "acsr_common_causal_residual_benchmark_supported"
            if status == "pass"
            else "acsr_common_causal_residual_benchmark_failed_gate"
        ),
        claim_status=(
            "sparse_support_specific_effect_survives_common_dense_controls_not_promoted"
            if status == "pass"
            else "sparse_support_specific_effect_not_separated_from_common_dense_controls"
        ),
        start=start,
        source_probe_dir=source_probe_dir,
        config_path=config_path,
        train_steps=train_steps,
        dense_steps=dense_steps,
        arm_rows=arm_rows,
        per_token_rows=per_token_rows,
        norm_rows=norm_rows,
        fingerprint_rows=fingerprint_rows,
        gate_rows=gate_rows,
        out_dir=out_dir,
    )
    _write_artifacts(
        out_dir,
        summary,
        arm_rows,
        per_token_rows,
        norm_rows,
        fingerprint_rows,
        gate_rows,
    )
    return summary


def _run_benchmark(
    *,
    config_path: Path,
    train_steps: int,
    dense_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import torch
    import torch.nn.functional as F

    from relaleap.smoke import ResidualColumns, TinyCharTransformer, _build_batch, _residual_loss

    nn = __import__("torch.nn").nn
    config = _read_yaml(config_path)
    run_cfg = _as_dict(config.get("run"))
    data_cfg = _as_dict(config.get("data"))
    model_cfg = _as_dict(config.get("model"))
    base_cfg = _as_dict(model_cfg.get("base"))
    column_cfg = _as_dict(model_cfg.get("columns"))
    training_cfg = _as_dict(config.get("training"))

    seed = int(run_cfg.get("seed", 1))
    learning_rate = float(run_cfg.get("learning_rate", 1e-2))
    dataset = str(data_cfg.get("dataset", "tiny_shakespeare_word"))
    seq_len = int(data_cfg.get("seq_len", 64))
    hidden_dim = int(base_cfg.get("hidden_dim", 96))
    layers = int(base_cfg.get("layers", 2))
    num_columns = int(column_cfg.get("num_columns", 24))
    atoms_per_column = int(column_cfg.get("atoms_per_column", 4))
    contextual_width = int(column_cfg.get("contextual_router_hidden_dim", hidden_dim * 2))
    residual_objective = str(training_cfg.get("residual_objective", "supervised_ce"))

    torch.manual_seed(seed)
    inputs, targets, vocab_size = _build_batch(dataset=dataset, seq_len=seq_len, batch_size=4)
    base = TinyCharTransformer(vocab_size=vocab_size, seq_len=seq_len, hidden_dim=hidden_dim, layers=layers)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base.eval()

    topk2 = _train_residual(
        torch,
        base,
        ResidualColumns,
        inputs,
        targets,
        vocab_size,
        hidden_dim=hidden_dim,
        num_columns=num_columns,
        atoms_per_column=atoms_per_column,
        top_k=2,
        support_router="contextual_mlp",
        contextual_width=contextual_width,
        residual_objective=residual_objective,
        learning_rate=learning_rate,
        steps=train_steps,
    )
    topk1 = _train_residual(
        torch,
        base,
        ResidualColumns,
        inputs,
        targets,
        vocab_size,
        hidden_dim=hidden_dim,
        num_columns=num_columns,
        atoms_per_column=atoms_per_column,
        top_k=1,
        support_router="contextual_mlp",
        contextual_width=contextual_width,
        residual_objective=residual_objective,
        learning_rate=learning_rate,
        steps=train_steps,
    )

    with torch.no_grad():
        hidden = base.encode(inputs)
        base_logits = base.decode(hidden)
        base_losses = _per_token_ce(F, base_logits, targets, vocab_size)
        base_loss = float(base_losses.mean().item())
        mask = _heldout_mask(base_losses.numel(), int(hidden.shape[1] - 1))
        heldout_base = float(base_losses[mask].mean().item())
        chunks = _contextual_chunks(torch, hidden)
        causal_inputs = _causal_predictor_inputs(torch, chunks)
        position_inputs = _position_predictor_inputs(torch, chunks)
        topk2_scores = topk2._score_columns(hidden) + topk2.score_tie_breaker.to(
            device=hidden.device, dtype=hidden.dtype
        )
        topk1_scores = topk1._score_columns(hidden) + topk1.score_tie_breaker.to(
            device=hidden.device, dtype=hidden.dtype
        )
        topk2_support = topk2_scores.topk(2, dim=-1).indices
        topk1_support = topk1_scores.topk(1, dim=-1).indices

    sparse_value_params = int(num_columns * atoms_per_column * (hidden_dim + 1))
    contextual2 = _eval_sparse_arm(
        torch,
        F,
        base,
        topk2,
        hidden,
        targets,
        vocab_size,
        "sparse_contextual_topk2",
        topk2_support,
        topk2_scores,
        base_losses,
        base_loss,
        heldout_base,
        mask,
    )
    contextual1 = _eval_sparse_arm(
        torch,
        F,
        base,
        topk1,
        hidden,
        targets,
        vocab_size,
        "sparse_rank_matched_topk1",
        topk1_support,
        topk1_scores,
        base_losses,
        base_loss,
        heldout_base,
        mask,
    )
    target_l2 = max(0.05, float(contextual2["heldout_residual_update_l2"]))
    sparse_active_params = int(contextual2["active_params_proxy"])
    dense_causal, dense_causal_losses, dense_causal_l2 = _train_dense_arm(
        torch,
        F,
        nn,
        base,
        hidden,
        targets,
        vocab_size,
        causal_inputs,
        label="rank_flop_matched_causal_dense",
        target_parameter_count=sparse_value_params,
        steps=dense_steps,
        base_losses=base_losses,
        target_update_l2=target_l2,
        heldout_mask=mask,
    )
    teacher_update = dense_causal.get("residual_update_tensor")
    if teacher_update is None:
        teacher_update = torch.zeros_like(hidden)
    frequency_support = _frequency_matched_random_support(
        torch,
        topk2_support,
        num_columns=num_columns,
        seed=seed + 17,
    )
    frequency_support_topk1 = _frequency_matched_random_support(
        torch,
        topk1_support,
        num_columns=num_columns,
        seed=seed + 19,
    )
    shuffled_support = _shuffle_support(torch, topk2_support, seed=seed + 29)
    token_position_support = _token_position_support(torch, topk2_support, num_columns)
    oracle_support = _oracle_support_topk(torch, F, base, topk2, hidden, targets, vocab_size, topk2_scores, k=2)

    teacher_distilled = _train_sparse_teacher_distilled_arm(
        torch,
        F,
        base,
        ResidualColumns,
        hidden,
        targets,
        vocab_size,
        teacher_update,
        label="sparse_teacher_distilled_norm_topk2",
        hidden_dim=hidden_dim,
        num_columns=num_columns,
        atoms_per_column=atoms_per_column,
        contextual_width=contextual_width,
        learning_rate=learning_rate,
        steps=max(train_steps * 4, train_steps + 24),
        base_losses=base_losses,
        base_loss=base_loss,
        heldout_base=heldout_base,
        heldout_mask=mask,
    )
    teacher_distilled_target_norm = _train_sparse_teacher_distilled_arm(
        torch,
        F,
        base,
        ResidualColumns,
        hidden,
        targets,
        vocab_size,
        teacher_update,
        label="sparse_teacher_distilled_target_norm_topk2",
        hidden_dim=hidden_dim,
        num_columns=num_columns,
        atoms_per_column=atoms_per_column,
        contextual_width=contextual_width,
        learning_rate=learning_rate,
        steps=max(train_steps * 4, train_steps + 24),
        base_losses=base_losses,
        base_loss=base_loss,
        heldout_base=heldout_base,
        heldout_mask=mask,
        norm_loss_weight=1.0,
        posthoc_target_norm_scale=True,
    )
    teacher_distilled_oracle = _train_sparse_teacher_distilled_arm(
        torch,
        F,
        base,
        ResidualColumns,
        hidden,
        targets,
        vocab_size,
        teacher_update,
        label="sparse_teacher_distilled_oracle_support_topk2",
        hidden_dim=hidden_dim,
        num_columns=num_columns,
        atoms_per_column=atoms_per_column,
        contextual_width=contextual_width,
        learning_rate=learning_rate,
        steps=max(train_steps * 4, train_steps + 24),
        base_losses=base_losses,
        base_loss=base_loss,
        heldout_base=heldout_base,
        heldout_mask=mask,
        fixed_support=oracle_support,
        norm_loss_weight=1.0,
        posthoc_target_norm_scale=True,
    )
    teacher_distilled_soft = _train_sparse_teacher_distilled_soft_arm(
        torch,
        F,
        base,
        ResidualColumns,
        hidden,
        targets,
        vocab_size,
        teacher_update,
        label="sparse_teacher_distilled_soft_temperature_topk2",
        hidden_dim=hidden_dim,
        num_columns=num_columns,
        atoms_per_column=atoms_per_column,
        contextual_width=contextual_width,
        learning_rate=learning_rate,
        steps=max(train_steps * 4, train_steps + 24),
        base_losses=base_losses,
        base_loss=base_loss,
        heldout_base=heldout_base,
        heldout_mask=mask,
        temperature=0.5,
    )
    teacher_distilled_token_position = _train_sparse_teacher_distilled_arm(
        torch,
        F,
        base,
        ResidualColumns,
        hidden,
        targets,
        vocab_size,
        teacher_update,
        label="sparse_teacher_distilled_token_position_null",
        hidden_dim=hidden_dim,
        num_columns=num_columns,
        atoms_per_column=atoms_per_column,
        contextual_width=contextual_width,
        learning_rate=learning_rate,
        steps=max(train_steps * 4, train_steps + 24),
        base_losses=base_losses,
        base_loss=base_loss,
        heldout_base=heldout_base,
        heldout_mask=mask,
        fixed_support=token_position_support,
        norm_loss_weight=1.0,
        posthoc_target_norm_scale=True,
    )
    shuffled_teacher_update = _shuffle_teacher_update(torch, teacher_update, seed=seed + 43)
    shuffled_teacher_distilled = _train_sparse_teacher_distilled_arm(
        torch,
        F,
        base,
        ResidualColumns,
        hidden,
        targets,
        vocab_size,
        shuffled_teacher_update,
        label="sparse_teacher_distilled_shuffled_teacher_null",
        hidden_dim=hidden_dim,
        num_columns=num_columns,
        atoms_per_column=atoms_per_column,
        contextual_width=contextual_width,
        learning_rate=learning_rate,
        steps=max(train_steps * 4, train_steps + 24),
        base_losses=base_losses,
        base_loss=base_loss,
        heldout_base=heldout_base,
        heldout_mask=mask,
    )
    dense_ladder = _train_dense_ladder(
        torch,
        F,
        nn,
        base,
        hidden,
        targets,
        vocab_size,
        causal_inputs,
        sparse_active_params=sparse_active_params,
        steps=dense_steps,
        base_losses=base_losses,
        target_update_l2=target_l2,
        heldout_mask=mask,
    )
    dense_position, dense_position_losses, dense_position_l2 = _train_dense_arm(
        torch,
        F,
        nn,
        base,
        hidden,
        targets,
        vocab_size,
        position_inputs,
        label="rank_flop_matched_token_position_dense",
        target_parameter_count=sparse_value_params,
        steps=dense_steps,
        base_losses=base_losses,
        target_update_l2=target_l2,
        heldout_mask=mask,
    )
    shuffled_causal_inputs = _shuffle_feature_rows(torch, causal_inputs, seed=seed + 59)
    dense_shuffled_causal, dense_shuffled_causal_losses, dense_shuffled_causal_l2 = _train_dense_arm(
        torch,
        F,
        nn,
        base,
        hidden,
        targets,
        vocab_size,
        shuffled_causal_inputs,
        label="rank_flop_matched_shuffled_causal_feature_dense_null",
        target_parameter_count=sparse_value_params,
        steps=dense_steps,
        base_losses=base_losses,
        target_update_l2=target_l2,
        heldout_mask=mask,
    )
    ablated_causal_inputs = torch.zeros_like(causal_inputs)
    dense_ablated_context, dense_ablated_context_losses, dense_ablated_context_l2 = _train_dense_arm(
        torch,
        F,
        nn,
        base,
        hidden,
        targets,
        vocab_size,
        ablated_causal_inputs,
        label="rank_flop_matched_ablated_context_dense",
        target_parameter_count=sparse_value_params,
        steps=dense_steps,
        base_losses=base_losses,
        target_update_l2=target_l2,
        heldout_mask=mask,
    )
    frequency_matched_random_topk1 = _eval_sparse_arm(
        torch,
        F,
        base,
        topk1,
        hidden,
        targets,
        vocab_size,
        "sparse_frequency_matched_random_topk1",
        frequency_support_topk1,
        topk1_scores,
        base_losses,
        base_loss,
        heldout_base,
        mask,
    )
    sparse_arms = [
        contextual2,
        contextual1,
        teacher_distilled,
        teacher_distilled_target_norm,
        teacher_distilled_oracle,
        teacher_distilled_soft,
        teacher_distilled_token_position,
        shuffled_teacher_distilled,
        frequency_matched_random_topk1,
        _eval_sparse_arm(torch, F, base, topk2, hidden, targets, vocab_size, "sparse_frequency_matched_random", frequency_support, topk2_scores, base_losses, base_loss, heldout_base, mask),
        _eval_sparse_arm(torch, F, base, topk2, hidden, targets, vocab_size, "sparse_shuffled_support_marginals", shuffled_support, topk2_scores, base_losses, base_loss, heldout_base, mask),
        _eval_sparse_arm(torch, F, base, topk2, hidden, targets, vocab_size, "sparse_token_position_null", token_position_support, topk2_scores, base_losses, base_loss, heldout_base, mask),
        _eval_sparse_arm(torch, F, base, topk2, hidden, targets, vocab_size, "sparse_oracle_support", oracle_support, topk2_scores, base_losses, base_loss, heldout_base, mask),
    ]
    dense_ladder_rows = [row for row, _, _ in dense_ladder]
    arm_rows = (
        [_base_arm(base_loss, heldout_base)]
        + sparse_arms
        + [dense_causal]
        + dense_ladder_rows
        + [dense_position, dense_shuffled_causal, dense_ablated_context]
    )
    dense_loss_by_arm = {
        "rank_flop_matched_causal_dense": dense_causal_losses,
        "rank_flop_matched_token_position_dense": dense_position_losses,
        "rank_flop_matched_shuffled_causal_feature_dense_null": dense_shuffled_causal_losses,
        "rank_flop_matched_ablated_context_dense": dense_ablated_context_losses,
    }
    dense_l2_by_arm = {
        "rank_flop_matched_causal_dense": dense_causal_l2,
        "rank_flop_matched_token_position_dense": dense_position_l2,
        "rank_flop_matched_shuffled_causal_feature_dense_null": dense_shuffled_causal_l2,
        "rank_flop_matched_ablated_context_dense": dense_ablated_context_l2,
    }
    for row, losses, l2 in dense_ladder:
        dense_loss_by_arm[str(row["arm"])] = losses
        dense_l2_by_arm[str(row["arm"])] = l2
    per_token_rows = _per_token_rows(
        torch,
        base_losses,
        dict(
            {
            "sparse_contextual_topk2": contextual2["per_token_losses"],
            "sparse_rank_matched_topk1": contextual1["per_token_losses"],
            "sparse_teacher_distilled_norm_topk2": teacher_distilled["per_token_losses"],
            "sparse_teacher_distilled_target_norm_topk2": teacher_distilled_target_norm["per_token_losses"],
            "sparse_teacher_distilled_oracle_support_topk2": teacher_distilled_oracle["per_token_losses"],
            "sparse_teacher_distilled_soft_temperature_topk2": teacher_distilled_soft["per_token_losses"],
            "sparse_teacher_distilled_token_position_null": teacher_distilled_token_position["per_token_losses"],
            "sparse_teacher_distilled_shuffled_teacher_null": shuffled_teacher_distilled["per_token_losses"],
            "sparse_frequency_matched_random_topk1": frequency_matched_random_topk1["per_token_losses"],
            },
            **dense_loss_by_arm,
        ),
        dict(
            {
            "sparse_contextual_topk2": contextual2["residual_update_l2_per_token"],
            "sparse_rank_matched_topk1": contextual1["residual_update_l2_per_token"],
            "sparse_teacher_distilled_norm_topk2": teacher_distilled["residual_update_l2_per_token"],
            "sparse_teacher_distilled_target_norm_topk2": teacher_distilled_target_norm["residual_update_l2_per_token"],
            "sparse_teacher_distilled_oracle_support_topk2": teacher_distilled_oracle["residual_update_l2_per_token"],
            "sparse_teacher_distilled_soft_temperature_topk2": teacher_distilled_soft["residual_update_l2_per_token"],
            "sparse_teacher_distilled_token_position_null": teacher_distilled_token_position["residual_update_l2_per_token"],
            "sparse_teacher_distilled_shuffled_teacher_null": shuffled_teacher_distilled["residual_update_l2_per_token"],
            "sparse_frequency_matched_random_topk1": frequency_matched_random_topk1["residual_update_l2_per_token"],
            },
            **dense_l2_by_arm,
        ),
        mask,
    )
    norm_rows = _norm_sweep_rows(arm_rows)
    fingerprint_rows = _fingerprint_rows(
        torch,
        arm_rows,
        topk2_support=topk2_support,
        topk1_support=topk1_support,
        oracle_support=oracle_support,
        random_support=frequency_support,
    )
    return arm_rows, per_token_rows, norm_rows, fingerprint_rows


def _train_dense_ladder(
    torch: Any,
    F: Any,
    nn: Any,
    base: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    features: Any,
    *,
    sparse_active_params: int,
    steps: int,
    base_losses: Any,
    target_update_l2: float,
    heldout_mask: Any,
) -> list[tuple[dict[str, Any], Any, Any]]:
    """Train a small causal-dense bottleneck ladder around sparse active compute."""

    input_dim = int(features.shape[-1])
    hidden_dim = int(hidden.shape[-1])
    rank1_params = input_dim + hidden_dim
    max_rank = max(1, int(sparse_active_params * 2) // max(1, rank1_params))
    candidate_ranks = [0, 1, max_rank]
    ranks: list[int] = []
    for rank in candidate_ranks:
        if rank not in ranks:
            ranks.append(rank)
    rows: list[tuple[dict[str, Any], Any, Any]] = []
    for rank in ranks:
        label = f"dense_bottleneck_causal_rank{rank}"
        rows.append(
            _train_dense_arm(
                torch,
                F,
                nn,
                base,
                hidden,
                targets,
                vocab_size,
                features,
                label=label,
                target_parameter_count=sparse_active_params,
                steps=steps,
                base_losses=base_losses,
                target_update_l2=target_update_l2,
                heldout_mask=heldout_mask,
                rank_override=rank,
            )
        )
    return rows


def _train_residual(
    torch: Any,
    base: Any,
    residual_cls: Any,
    inputs: Any,
    targets: Any,
    vocab_size: int,
    *,
    hidden_dim: int,
    num_columns: int,
    atoms_per_column: int,
    top_k: int,
    support_router: str,
    contextual_width: int,
    residual_objective: str,
    learning_rate: float,
    steps: int,
) -> Any:
    from relaleap.smoke import _residual_loss

    residual = residual_cls(
        hidden_dim=hidden_dim,
        num_columns=num_columns,
        atoms_per_column=atoms_per_column,
        top_k=top_k,
        support_router=support_router,
        contextual_router_hidden_dim=contextual_width,
    )
    residual.train()
    optimizer = torch.optim.AdamW(residual.parameters(), lr=learning_rate)
    for _ in range(max(1, steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = _residual_loss(base, residual, inputs, targets, vocab_size, objective=residual_objective)
        loss.backward()
        optimizer.step()
    residual.eval()
    return residual


def _eval_sparse_arm(
    torch: Any,
    F: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    arm: str,
    support: Any,
    scores: Any,
    base_losses: Any,
    base_loss: float,
    heldout_base: float,
    heldout_mask: Any,
) -> dict[str, Any]:
    metrics = _support_eval_metrics(
        torch,
        F,
        base,
        residual,
        hidden,
        targets,
        vocab_size,
        support=support,
        target_scores=scores,
    )
    losses = metrics["per_token_losses"]
    l2 = metrics["residual_update_l2_per_token"]
    heldout_loss = float(losses[heldout_mask].mean().item())
    row = {
        "arm": arm,
        "family": "sparse",
        "top_k": int(support.shape[-1]),
        "all_ce_loss": metrics["ce_loss"],
        "heldout_ce_loss": heldout_loss,
        "delta_vs_base_ce": metrics["ce_loss"] - base_loss,
        "heldout_delta_vs_base_ce": heldout_loss - heldout_base,
        "residual_update_l2_mean": metrics["residual_update_l2_mean"],
        "heldout_residual_update_l2": float(l2[heldout_mask].mean().item()),
        "active_params_proxy": int(residual.atom_values.shape[-1] * support.shape[-1]),
        "stored_params_proxy": _parameter_count(residual),
        "flops_proxy": int(residual.atom_values.shape[-1] * support.shape[-1]),
        "support_entropy": _support_entropy(torch, support, residual.num_columns),
        "used_columns": len({int(v) for v in support.reshape(-1).detach().cpu().tolist()}),
        "unique_support_sets": len({tuple(int(v) for v in row) for row in support.reshape(-1, support.shape[-1]).detach().cpu().tolist()}),
        "per_token_losses": losses.detach(),
        "residual_update_l2_per_token": l2.detach(),
    }
    return row


def _train_sparse_teacher_distilled_arm(
    torch: Any,
    F: Any,
    base: Any,
    residual_cls: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    teacher_update: Any,
    *,
    label: str,
    hidden_dim: int,
    num_columns: int,
    atoms_per_column: int,
    contextual_width: int,
    learning_rate: float,
    steps: int,
    base_losses: Any,
    base_loss: float,
    heldout_base: float,
    heldout_mask: Any,
    fixed_support: Any | None = None,
    norm_loss_weight: float = 0.25,
    posthoc_target_norm_scale: bool = False,
) -> dict[str, Any]:
    residual = residual_cls(
        hidden_dim=hidden_dim,
        num_columns=num_columns,
        atoms_per_column=atoms_per_column,
        top_k=2,
        support_router="contextual_mlp",
        contextual_router_hidden_dim=contextual_width,
    )
    residual.train()
    optimizer = torch.optim.AdamW(residual.parameters(), lr=learning_rate)
    teacher = teacher_update.detach()
    teacher_direction = _normalize_update(F, teacher)
    teacher_l2 = teacher[:, :-1, :].reshape(-1, teacher.shape[-1]).norm(dim=-1)
    target_l2 = teacher_l2.mean().detach()
    for _ in range(max(1, steps)):
        optimizer.zero_grad(set_to_none=True)
        updated = residual(hidden.detach(), support_indices=fixed_support)
        update = updated - hidden.detach()
        logits = base.decode(updated)
        update_l2 = update[:, :-1, :].reshape(-1, update.shape[-1]).norm(dim=-1).mean()
        direction_loss = F.mse_loss(_normalize_update(F, update), teacher_direction)
        norm_loss = (update_l2 - target_l2) ** 2
        ce_loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, vocab_size),
            targets[:, :-1].reshape(-1),
        )
        loss = direction_loss + norm_loss_weight * norm_loss + 0.05 * ce_loss
        loss.backward()
        optimizer.step()
    residual.eval()
    with torch.no_grad():
        if posthoc_target_norm_scale:
            scored = residual(hidden.detach(), support_indices=fixed_support)
            scored_update = scored - hidden.detach()
            scored_l2 = scored_update[:, :-1, :].reshape(-1, scored_update.shape[-1]).norm(dim=-1).mean()
            if float(scored_l2.item()) > 1e-12:
                residual.atom_values.mul_(float((target_l2 / scored_l2).item()))
        scores = residual._score_columns(hidden) + residual.score_tie_breaker.to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        support = fixed_support if fixed_support is not None else scores.topk(2, dim=-1).indices
    row = _eval_sparse_arm(
        torch,
        F,
        base,
        residual,
        hidden,
        targets,
        vocab_size,
        label,
        support,
        scores,
        base_losses,
        base_loss,
        heldout_base,
        heldout_mask,
    )
    with torch.no_grad():
        residual_update = residual(hidden, support_indices=support) - hidden
        teacher_flat = teacher[:, :-1, :].reshape(-1, teacher.shape[-1])
        residual_flat = residual_update[:, :-1, :].reshape(-1, residual_update.shape[-1])
        teacher_l2_flat = teacher_flat.norm(dim=-1)
        residual_l2_flat = residual_flat.norm(dim=-1)
        cosine = F.cosine_similarity(residual_flat, teacher_flat, dim=-1, eps=1e-8)
        mse = ((residual_flat - teacher_flat) ** 2).mean(dim=-1)
    row.update(
        {
            "teacher_residual_cosine": float(cosine[heldout_mask].mean().item()),
            "teacher_residual_mse": float(mse[heldout_mask].mean().item()),
            "teacher_heldout_residual_update_l2": float(teacher_l2_flat[heldout_mask].mean().item()),
            "heldout_residual_l2_ratio_to_teacher": (
                ""
                if float(teacher_l2_flat[heldout_mask].mean().item()) <= 1e-12
                else float(residual_l2_flat[heldout_mask].mean().item())
                / float(teacher_l2_flat[heldout_mask].mean().item())
            ),
        }
    )
    return row


def _train_sparse_teacher_distilled_soft_arm(
    torch: Any,
    F: Any,
    base: Any,
    residual_cls: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    teacher_update: Any,
    *,
    label: str,
    hidden_dim: int,
    num_columns: int,
    atoms_per_column: int,
    contextual_width: int,
    learning_rate: float,
    steps: int,
    base_losses: Any,
    base_loss: float,
    heldout_base: float,
    heldout_mask: Any,
    temperature: float,
) -> dict[str, Any]:
    residual = residual_cls(
        hidden_dim=hidden_dim,
        num_columns=num_columns,
        atoms_per_column=atoms_per_column,
        top_k=2,
        support_router="contextual_mlp",
        contextual_router_hidden_dim=contextual_width,
    )
    residual.train()
    optimizer = torch.optim.AdamW(residual.parameters(), lr=learning_rate)
    teacher = teacher_update.detach()
    teacher_direction = _normalize_update(F, teacher)
    teacher_l2 = teacher[:, :-1, :].reshape(-1, teacher.shape[-1]).norm(dim=-1)
    target_l2 = teacher_l2.mean().detach()
    for _ in range(max(1, steps)):
        optimizer.zero_grad(set_to_none=True)
        updated, update = _soft_residual_update(torch, F, residual, hidden.detach(), temperature=temperature)
        logits = base.decode(updated)
        update_l2 = update[:, :-1, :].reshape(-1, update.shape[-1]).norm(dim=-1).mean()
        direction_loss = F.mse_loss(_normalize_update(F, update), teacher_direction)
        norm_loss = (update_l2 - target_l2) ** 2
        ce_loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, vocab_size),
            targets[:, :-1].reshape(-1),
        )
        loss = direction_loss + norm_loss + 0.05 * ce_loss
        loss.backward()
        optimizer.step()
    residual.eval()
    with torch.no_grad():
        updated, residual_update = _soft_residual_update(torch, F, residual, hidden, temperature=temperature)
        raw_l2 = residual_update[:, :-1, :].reshape(-1, residual_update.shape[-1]).norm(dim=-1).mean()
        if float(raw_l2.item()) > 1e-12:
            residual.atom_values.mul_(float((target_l2 / raw_l2).item()))
            updated, residual_update = _soft_residual_update(torch, F, residual, hidden, temperature=temperature)
        logits = base.decode(updated)
        losses = _per_token_ce(F, logits, targets, vocab_size)
        l2 = residual_update[:, :-1, :].reshape(-1, residual_update.shape[-1]).norm(dim=-1)
        heldout_loss = float(losses[heldout_mask].mean().item())
        teacher_flat = teacher[:, :-1, :].reshape(-1, teacher.shape[-1])
        residual_flat = residual_update[:, :-1, :].reshape(-1, residual_update.shape[-1])
        teacher_l2_flat = teacher_flat.norm(dim=-1)
        residual_l2_flat = residual_flat.norm(dim=-1)
        cosine = F.cosine_similarity(residual_flat, teacher_flat, dim=-1, eps=1e-8)
        mse = ((residual_flat - teacher_flat) ** 2).mean(dim=-1)
    return {
        "arm": label,
        "family": "sparse",
        "top_k": "soft_all",
        "temperature": temperature,
        "all_ce_loss": float(losses.mean().item()),
        "heldout_ce_loss": heldout_loss,
        "delta_vs_base_ce": float(losses.mean().item()) - base_loss,
        "heldout_delta_vs_base_ce": heldout_loss - heldout_base,
        "residual_update_l2_mean": float(l2.mean().item()),
        "heldout_residual_update_l2": float(l2[heldout_mask].mean().item()),
        "active_params_proxy": int(residual.atom_values.shape[-1] * residual.num_columns),
        "stored_params_proxy": _parameter_count(residual),
        "flops_proxy": int(residual.atom_values.shape[-1] * residual.num_columns),
        "support_entropy": "",
        "used_columns": residual.num_columns,
        "unique_support_sets": "soft_all",
        "teacher_residual_cosine": float(cosine[heldout_mask].mean().item()),
        "teacher_residual_mse": float(mse[heldout_mask].mean().item()),
        "teacher_heldout_residual_update_l2": float(teacher_l2_flat[heldout_mask].mean().item()),
        "heldout_residual_l2_ratio_to_teacher": (
            ""
            if float(teacher_l2_flat[heldout_mask].mean().item()) <= 1e-12
            else float(residual_l2_flat[heldout_mask].mean().item())
            / float(teacher_l2_flat[heldout_mask].mean().item())
        ),
        "per_token_losses": losses.detach(),
        "residual_update_l2_per_token": l2.detach(),
    }


def _soft_residual_update(
    torch: Any,
    F: Any,
    residual: Any,
    hidden: Any,
    *,
    temperature: float,
) -> tuple[Any, Any]:
    scores = residual._score_columns(hidden) + residual.score_tie_breaker.to(
        device=hidden.device,
        dtype=hidden.dtype,
    )
    column_weights = F.softmax(scores / max(temperature, 1e-6), dim=-1)
    atom_weights = F.softmax(residual.atom_logits, dim=-1)
    column_values = torch.einsum("ca,cah->ch", atom_weights, residual.atom_values)
    update = torch.einsum("bsc,ch->bsh", column_weights, column_values)
    return hidden + update, update


def _normalize_update(F: Any, update: Any) -> Any:
    return F.normalize(update[:, :-1, :], p=2, dim=-1, eps=1e-8)


def _shuffle_teacher_update(torch: Any, teacher_update: Any, *, seed: int) -> Any:
    generator = torch.Generator(device=teacher_update.device)
    generator.manual_seed(seed)
    shuffled = teacher_update.clone()
    flat = teacher_update[:, :-1, :].reshape(-1, teacher_update.shape[-1])
    order = torch.randperm(flat.shape[0], generator=generator, device=teacher_update.device)
    shuffled[:, :-1, :] = flat[order].view(
        teacher_update.shape[0],
        teacher_update.shape[1] - 1,
        teacher_update.shape[-1],
    )
    shuffled[:, -1:, :] = shuffled[:, -2:-1, :]
    return shuffled


def _shuffle_feature_rows(torch: Any, features: Any, *, seed: int) -> Any:
    generator = torch.Generator(device=features.device)
    generator.manual_seed(seed)
    shuffled = features.clone()
    flat = features.reshape(-1, features.shape[-1])
    order = torch.randperm(flat.shape[0], generator=generator, device=features.device)
    return flat[order].view_as(shuffled)


def _train_dense_arm(
    torch: Any,
    F: Any,
    nn: Any,
    base: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    features: Any,
    *,
    label: str,
    target_parameter_count: int,
    steps: int,
    base_losses: Any,
    target_update_l2: float,
    heldout_mask: Any,
    rank_override: int | None = None,
) -> tuple[dict[str, Any], Any, Any]:
    split = max(1, int(features.shape[1]) // 2)
    rank = (
        max(0, int(rank_override))
        if rank_override is not None
        else max(1, target_parameter_count // max(1, int(features.shape[-1]) + int(hidden.shape[-1])))
    )
    if rank == 0:
        with torch.no_grad():
            update = torch.zeros_like(hidden)
            logits = base.decode(hidden + update)
            losses = _per_token_ce(F, logits, targets, vocab_size)
            l2 = update[:, :-1, :].reshape(-1, update.shape[-1]).norm(dim=-1)
            heldout_loss = float(losses[heldout_mask].mean().item())
            base_loss = float(base_losses.mean().item())
            heldout_base = float(base_losses[heldout_mask].mean().item())
        return (
            {
                "arm": label,
                "family": "dense",
                "top_k": "",
                "rank": rank,
                "target_active_params_proxy": target_parameter_count,
                "posthoc_residual_norm_scale": 0.0,
                "raw_heldout_residual_update_l2": 0.0,
                "all_ce_loss": float(losses.mean().item()),
                "heldout_ce_loss": heldout_loss,
                "delta_vs_base_ce": float(losses.mean().item()) - base_loss,
                "heldout_delta_vs_base_ce": heldout_loss - heldout_base,
                "residual_update_l2_mean": float(l2.mean().item()),
                "heldout_residual_update_l2": float(l2[heldout_mask].mean().item()),
                "active_params_proxy": 0,
                "stored_params_proxy": 0,
                "flops_proxy": 0,
                "support_entropy": "",
                "used_columns": "",
                "unique_support_sets": "",
                "residual_update_tensor": update.detach(),
            },
            losses.detach(),
            l2.detach(),
        )
    adapter = _LowRankCausalDenseAdapter(nn, int(features.shape[-1]), int(hidden.shape[-1]), rank)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=3e-3)
    for _ in range(max(1, steps)):
        optimizer.zero_grad(set_to_none=True)
        update = adapter(features[:, :split, :])
        logits = base.decode(hidden[:, :split, :] + update)
        ce_loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab_size), targets[:, :split][:, :-1].reshape(-1))
        update_l2 = update[:, :-1, :].reshape(-1, update.shape[-1]).norm(dim=-1).mean()
        loss = ce_loss + (update_l2 - target_update_l2) ** 2
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        raw_update = adapter(features)
        raw_l2 = raw_update[:, :-1, :].reshape(-1, raw_update.shape[-1]).norm(dim=-1)
        heldout_raw_l2 = float(raw_l2[heldout_mask].mean().item())
        scale = 1.0
        if heldout_raw_l2 > 1e-12:
            scale = min(1.0, target_update_l2 / heldout_raw_l2)
        update = raw_update * scale
        logits = base.decode(hidden + update)
        losses = _per_token_ce(F, logits, targets, vocab_size)
        l2 = update[:, :-1, :].reshape(-1, update.shape[-1]).norm(dim=-1)
        heldout_loss = float(losses[heldout_mask].mean().item())
        base_loss = float(base_losses.mean().item())
        heldout_base = float(base_losses[heldout_mask].mean().item())
    return (
        {
            "arm": label,
            "family": "dense",
            "top_k": "",
            "rank": rank,
            "target_active_params_proxy": target_parameter_count,
            "posthoc_residual_norm_scale": scale,
            "raw_heldout_residual_update_l2": heldout_raw_l2,
            "all_ce_loss": float(losses.mean().item()),
            "heldout_ce_loss": heldout_loss,
            "delta_vs_base_ce": float(losses.mean().item()) - base_loss,
            "heldout_delta_vs_base_ce": heldout_loss - heldout_base,
            "residual_update_l2_mean": float(l2.mean().item()),
            "heldout_residual_update_l2": float(l2[heldout_mask].mean().item()),
            "active_params_proxy": _parameter_count(adapter),
            "stored_params_proxy": _parameter_count(adapter),
            "flops_proxy": _parameter_count(adapter),
            "support_entropy": "",
            "used_columns": "",
            "unique_support_sets": "",
            "residual_update_tensor": update.detach(),
        },
        losses.detach(),
        l2.detach(),
    )


def _oracle_support_topk(
    torch: Any,
    F: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    scores: Any,
    *,
    k: int,
) -> Any:
    candidates: list[tuple[int, ...]] = []
    if k == 1:
        candidates = [(idx,) for idx in range(residual.num_columns)]
    else:
        for left in range(residual.num_columns):
            for right in range(left + 1, residual.num_columns):
                candidates.append((left, right))
    losses = []
    supports = []
    for candidate in candidates:
        support = torch.empty(*hidden.shape[:2], k, dtype=torch.long, device=hidden.device)
        for idx, column in enumerate(candidate):
            support[..., idx] = column
        metrics = _support_eval_metrics(torch, F, base, residual, hidden, targets, vocab_size, support=support, target_scores=scores)
        losses.append(metrics["per_token_losses"])
        supports.append(support[:, :-1, :].reshape(-1, k))
    stacked = torch.stack(losses, dim=0)
    best = stacked.argmin(dim=0)
    flat_supports = torch.stack(supports, dim=0)
    best_flat = flat_supports[best, torch.arange(best.numel(), device=best.device)]
    output = torch.empty(*hidden.shape[:2], k, dtype=torch.long, device=hidden.device)
    output[:, :-1, :] = best_flat.view(hidden.shape[0], hidden.shape[1] - 1, k)
    output[:, -1:, :] = output[:, -2:-1, :]
    return output


def _shuffle_support(torch: Any, support: Any, *, seed: int) -> Any:
    generator = torch.Generator(device=support.device)
    generator.manual_seed(seed)
    flat = support[:, :-1, :].reshape(-1, support.shape[-1])
    order = torch.randperm(flat.shape[0], generator=generator, device=support.device)
    shuffled = torch.empty_like(support)
    shuffled[:, :-1, :] = flat[order].view(support.shape[0], support.shape[1] - 1, support.shape[-1])
    shuffled[:, -1:, :] = shuffled[:, -2:-1, :]
    return shuffled


def _token_position_support(torch: Any, like_support: Any, num_columns: int) -> Any:
    positions = torch.arange(like_support.shape[1], device=like_support.device).view(1, -1, 1)
    offsets = torch.arange(like_support.shape[-1], device=like_support.device).view(1, 1, -1)
    return ((positions + offsets) % num_columns).expand_as(like_support).long()


def _base_arm(base_loss: float, heldout_base: float) -> dict[str, Any]:
    return {
        "arm": "base_no_residual",
        "family": "base",
        "all_ce_loss": base_loss,
        "heldout_ce_loss": heldout_base,
        "delta_vs_base_ce": 0.0,
        "heldout_delta_vs_base_ce": 0.0,
        "residual_update_l2_mean": 0.0,
        "heldout_residual_update_l2": 0.0,
        "active_params_proxy": 0,
        "stored_params_proxy": 0,
        "flops_proxy": 0,
    }


def _per_token_rows(
    torch: Any,
    base_losses: Any,
    loss_by_arm: dict[str, Any],
    l2_by_arm: dict[str, Any],
    heldout_mask: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seq_len_minus_one = max(1, int(heldout_mask.numel() // 4))
    for arm, losses in loss_by_arm.items():
        for idx, loss in enumerate(losses.detach().cpu().tolist()):
            rows.append(
                {
                    "arm": arm,
                    "token_index": idx,
                    "position_index": idx % seq_len_minus_one,
                    "split": "heldout" if bool(heldout_mask[idx].item()) else "train",
                    "base_ce_loss": float(base_losses[idx].item()),
                    "ce_loss": float(loss),
                    "delta_vs_base_ce": float(loss - base_losses[idx].item()),
                    "residual_update_l2": float(l2_by_arm[arm][idx].item()),
                }
            )
    return rows


def _norm_sweep_rows(arm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    sparse_active = _float_or_none(
        next(
            (
                row.get("active_params_proxy")
                for row in arm_rows
                if row.get("arm") == "sparse_contextual_topk2"
            ),
            None,
        )
    )
    for row in arm_rows:
        if row.get("arm") == "base_no_residual":
            continue
        ce_delta = _float_or_none(row.get("heldout_delta_vs_base_ce"))
        l2 = _float_or_none(row.get("heldout_residual_update_l2"))
        active = _float_or_none(row.get("active_params_proxy"))
        flops = _float_or_none(row.get("flops_proxy"))
        rows.append(
            {
                "arm": row["arm"],
                "family": row.get("family", ""),
                "rank": row.get("rank", ""),
                "heldout_delta_vs_base_ce": ce_delta,
                "heldout_residual_update_l2": l2,
                "ce_gain_per_l2": "" if l2 is None or abs(l2) < 1e-12 or ce_delta is None else ce_delta / l2,
                "active_params_proxy": active,
                "flops_proxy": flops,
                "active_ratio_vs_sparse_topk2": (
                    ""
                    if active is None or sparse_active is None or sparse_active <= 0
                    else active / sparse_active
                ),
                "heldout_ce_delta_per_active_param": (
                    ""
                    if active is None or active <= 0 or ce_delta is None
                    else ce_delta / active
                ),
                "norm_bucket": _norm_bucket(l2),
            }
        )
    for row in rows:
        row_delta = _float_or_none(row.get("heldout_delta_vs_base_ce"))
        row_active = _float_or_none(row.get("active_params_proxy"))
        if row_delta is None or row_active is None:
            row["active_compute_pareto_front"] = False
            continue
        dominated = False
        for other in rows:
            other_delta = _float_or_none(other.get("heldout_delta_vs_base_ce"))
            other_active = _float_or_none(other.get("active_params_proxy"))
            if other_delta is None or other_active is None or other is row:
                continue
            if other_delta <= row_delta and other_active <= row_active and (
                other_delta < row_delta or other_active < row_active
            ):
                dominated = True
                break
        row["active_compute_pareto_front"] = not dominated
    return rows


def _fingerprint_rows(
    torch: Any,
    arm_rows: list[dict[str, Any]],
    *,
    topk2_support: Any,
    topk1_support: Any,
    oracle_support: Any,
    random_support: Any,
) -> list[dict[str, Any]]:
    by_name = {str(row.get("arm")): row for row in arm_rows}
    topk2 = by_name.get("sparse_contextual_topk2", {})
    oracle = by_name.get("sparse_oracle_support", {})
    random = by_name.get("sparse_frequency_matched_random", {})
    return [
        {
            "fingerprint": "oracle_regret",
            "arm": "sparse_contextual_topk2",
            "value": _maybe_subtract(topk2.get("heldout_ce_loss"), oracle.get("heldout_ce_loss")),
            "interpretation": "lower means learned support is closer to exhaustive support choice",
        },
        {
            "fingerprint": "random_support_damage",
            "arm": "sparse_contextual_topk2",
            "value": _maybe_subtract(random.get("heldout_ce_loss"), topk2.get("heldout_ce_loss")),
            "interpretation": "positive means selected support matters versus frequency-matched random support",
        },
        {
            "fingerprint": "topk1_topk2_support_jaccard",
            "arm": "sparse_rank_matched_topk1",
            "value": _support_jaccard(topk1_support.expand(*topk1_support.shape[:-1], topk2_support.shape[-1]), topk2_support),
            "interpretation": "rank bracket support overlap proxy",
        },
        {
            "fingerprint": "oracle_support_jaccard",
            "arm": "sparse_contextual_topk2",
            "value": _support_jaccard(oracle_support, topk2_support),
            "interpretation": "selected-support overlap with exhaustive support",
        },
        {
            "fingerprint": "random_support_jaccard",
            "arm": "sparse_frequency_matched_random",
            "value": _support_jaccard(random_support, topk2_support),
            "interpretation": "frequency-matched null overlap with selected support",
        },
    ]


def _benchmark_gate_rows(
    arm_rows: list[dict[str, Any]],
    fingerprint_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    arms = {str(row.get("arm")): row for row in arm_rows}
    required = {
        "base_no_residual",
        "sparse_contextual_topk2",
        "sparse_rank_matched_topk1",
        "sparse_teacher_distilled_norm_topk2",
        "sparse_teacher_distilled_target_norm_topk2",
        "sparse_teacher_distilled_oracle_support_topk2",
        "sparse_teacher_distilled_soft_temperature_topk2",
        "sparse_teacher_distilled_token_position_null",
        "sparse_teacher_distilled_shuffled_teacher_null",
        "rank_flop_matched_causal_dense",
        "rank_flop_matched_token_position_dense",
        "rank_flop_matched_shuffled_causal_feature_dense_null",
        "rank_flop_matched_ablated_context_dense",
        "sparse_frequency_matched_random_topk1",
        "sparse_frequency_matched_random",
        "sparse_shuffled_support_marginals",
        "sparse_token_position_null",
        "sparse_oracle_support",
    }
    sparse = arms.get("sparse_contextual_topk2", {})
    dense = arms.get("rank_flop_matched_causal_dense", {})
    token_dense = arms.get("rank_flop_matched_token_position_dense", {})
    teacher_distilled = arms.get("sparse_teacher_distilled_norm_topk2", {})
    target_norm_teacher = arms.get("sparse_teacher_distilled_target_norm_topk2", {})
    oracle_teacher = arms.get("sparse_teacher_distilled_oracle_support_topk2", {})
    soft_teacher = arms.get("sparse_teacher_distilled_soft_temperature_topk2", {})
    token_position_teacher = arms.get("sparse_teacher_distilled_token_position_null", {})
    shuffled_teacher = arms.get("sparse_teacher_distilled_shuffled_teacher_null", {})
    random = arms.get("sparse_frequency_matched_random", {})
    oracle = arms.get("sparse_oracle_support", {})
    sparse_delta = _float_or_none(sparse.get("heldout_delta_vs_base_ce"))
    dense_delta = _float_or_none(dense.get("heldout_delta_vs_base_ce"))
    teacher_distilled_delta = _float_or_none(teacher_distilled.get("heldout_delta_vs_base_ce"))
    target_norm_delta = _float_or_none(target_norm_teacher.get("heldout_delta_vs_base_ce"))
    oracle_teacher_delta = _float_or_none(oracle_teacher.get("heldout_delta_vs_base_ce"))
    soft_teacher_delta = _float_or_none(soft_teacher.get("heldout_delta_vs_base_ce"))
    token_position_teacher_delta = _float_or_none(token_position_teacher.get("heldout_delta_vs_base_ce"))
    shuffled_teacher_delta = _float_or_none(shuffled_teacher.get("heldout_delta_vs_base_ce"))
    teacher_distilled_mse = _float_or_none(teacher_distilled.get("teacher_residual_mse"))
    target_norm_mse = _float_or_none(target_norm_teacher.get("teacher_residual_mse"))
    oracle_teacher_mse = _float_or_none(oracle_teacher.get("teacher_residual_mse"))
    soft_teacher_mse = _float_or_none(soft_teacher.get("teacher_residual_mse"))
    token_position_teacher_mse = _float_or_none(token_position_teacher.get("teacher_residual_mse"))
    shuffled_teacher_mse = _float_or_none(shuffled_teacher.get("teacher_residual_mse"))
    hard_distill_best_mse = _min_present([teacher_distilled_mse, target_norm_mse, oracle_teacher_mse])
    token_dense_delta = _float_or_none(token_dense.get("heldout_delta_vs_base_ce"))
    sparse_l2 = _float_or_none(sparse.get("heldout_residual_update_l2"))
    dense_l2 = _float_or_none(dense.get("heldout_residual_update_l2"))
    active_compute = _active_compute_match_details(arm_rows, sparse_arm="sparse_contextual_topk2")
    random_damage = _maybe_subtract(random.get("heldout_ce_loss"), sparse.get("heldout_ce_loss"))
    oracle_regret = _maybe_subtract(sparse.get("heldout_ce_loss"), oracle.get("heldout_ce_loss"))
    return [
        _criterion("common_baselines_present", required.issubset(arms), "all required sparse/dense/null/oracle arms exist", sorted(arms), "missing one or more required common-baseline arms"),
        _criterion("dense_norm_matched", sparse_l2 is not None and dense_l2 is not None and dense_l2 <= max(0.25, sparse_l2 * 2.0), "causal dense held-out residual L2 is within 2x sparse top-k2 L2 with 0.25 floor", {"sparse_l2": sparse_l2, "dense_l2": dense_l2}, "causal dense control used too much residual norm"),
        _criterion("active_compute_matched_or_bracketed", bool(active_compute["matched_or_bracketed"]), "causal dense active params/FLOPs must be matched to or bracket sparse top-k2 before dense-vs-sparse claims", active_compute, "dense control is not active-param/FLOP matched or bracketed"),
        _criterion("causal_dense_beats_token_position_null", dense_delta is not None and token_dense_delta is not None and dense_delta < token_dense_delta, "causal dense should beat token-position dense null", {"causal_dense_delta": dense_delta, "token_position_dense_delta": token_dense_delta}, "causal dense did not beat token-position dense null"),
        _criterion("sparse_beats_causal_dense", sparse_delta is not None and dense_delta is not None and sparse_delta < dense_delta, "sparse top-k2 held-out CE delta must beat rank/FLOP-matched causal dense", {"sparse_delta": sparse_delta, "dense_delta": dense_delta}, "sparse top-k2 did not beat causal dense"),
        _criterion("teacher_distilled_sparse_beats_shuffled_teacher_null", teacher_distilled_mse is not None and shuffled_teacher_mse is not None and teacher_distilled_mse < shuffled_teacher_mse and teacher_distilled_delta is not None and shuffled_teacher_delta is not None and teacher_distilled_delta < shuffled_teacher_delta, "dense-teacher-distilled sparse arm should beat shuffled-teacher null on teacher residual MSE and held-out CE", {"teacher_distilled_mse": teacher_distilled_mse, "shuffled_teacher_mse": shuffled_teacher_mse, "teacher_distilled_delta": teacher_distilled_delta, "shuffled_teacher_delta": shuffled_teacher_delta}, "teacher-distilled sparse rescue did not beat shuffled-teacher null"),
        _criterion("target_norm_distill_beats_current_distill_mse", target_norm_mse is not None and teacher_distilled_mse is not None and target_norm_mse <= teacher_distilled_mse, "target-norm-scaled distillation should improve or match current sparse teacher MSE", {"target_norm_mse": target_norm_mse, "current_mse": teacher_distilled_mse, "target_norm_delta": target_norm_delta, "current_delta": teacher_distilled_delta}, "target-norm scaling did not improve the current distill MSE"),
        _criterion("oracle_support_distill_tests_discovery_bottleneck", oracle_teacher_mse is not None and target_norm_mse is not None and oracle_teacher_mse <= target_norm_mse, "oracle-support distill should be at least as good as learned-support target-norm distill if discovery is the bottleneck", {"oracle_support_mse": oracle_teacher_mse, "target_norm_mse": target_norm_mse, "oracle_support_delta": oracle_teacher_delta, "target_norm_delta": target_norm_delta}, "oracle-support distill did not improve on learned-support target-norm distill"),
        _criterion("soft_topk_distill_tests_representation_ceiling", soft_teacher_mse is not None and hard_distill_best_mse is not None and soft_teacher_mse <= hard_distill_best_mse, "soft/temperature sparse mixture should provide a representation ceiling no worse than hard sparse distill arms", {"soft_mse": soft_teacher_mse, "best_hard_sparse_distill_mse": hard_distill_best_mse, "current_mse": teacher_distilled_mse, "target_norm_mse": target_norm_mse, "oracle_support_mse": oracle_teacher_mse, "soft_delta": soft_teacher_delta}, "soft/temperature sparse mixture did not improve the hard sparse distill ceiling"),
        _criterion("teacher_distill_beats_token_position_null", target_norm_mse is not None and token_position_teacher_mse is not None and target_norm_mse < token_position_teacher_mse and target_norm_delta is not None and token_position_teacher_delta is not None and target_norm_delta < token_position_teacher_delta, "teacher-distilled learned support should beat token/position-only support null", {"target_norm_mse": target_norm_mse, "token_position_mse": token_position_teacher_mse, "target_norm_delta": target_norm_delta, "token_position_delta": token_position_teacher_delta}, "teacher-distilled learned support did not beat token/position-only support null"),
        _criterion("support_identity_matters", isinstance(random_damage, float) and random_damage > 0.0, "frequency-matched random support should hurt held-out CE versus selected sparse support", random_damage, "selected support was not better than frequency-matched random support"),
        _criterion("oracle_regret_nonnegative", isinstance(oracle_regret, float) and oracle_regret >= -1e-8, "exhaustive oracle support should not be worse than selected support", oracle_regret, "oracle support sanity check failed"),
        _criterion("intervention_fingerprints_present", len(fingerprint_rows) >= 5, "fingerprint rows include oracle/random/support-overlap observables", len(fingerprint_rows), "missing intervention fingerprint rows"),
    ]


def _active_compute_match_details(
    arm_rows: list[dict[str, Any]],
    *,
    sparse_arm: str,
    match_ratio: float = 2.0,
) -> dict[str, Any]:
    arms = {str(row.get("arm")): row for row in arm_rows}
    sparse_active = _float_or_none(arms.get(sparse_arm, {}).get("active_params_proxy"))
    dense_rows = [
        row
        for row in arm_rows
        if (
            str(row.get("family")) == "dense"
            or "dense" in str(row.get("arm"))
        )
        and _float_or_none(row.get("active_params_proxy")) is not None
    ]
    dense_active_values = [
        int(active)
        for row in dense_rows
        for active in [_float_or_none(row.get("active_params_proxy"))]
        if active is not None
    ]
    if sparse_active is None or sparse_active <= 0 or not dense_active_values:
        return {
            "sparse_arm": sparse_arm,
            "sparse_active_params_proxy": sparse_active,
            "dense_active_params_proxy": dense_active_values,
            "matched_or_bracketed": False,
            "reason": "missing sparse or dense active-param proxy",
        }
    lower = sparse_active / match_ratio
    upper = sparse_active * match_ratio
    matched = [value for value in dense_active_values if lower <= value <= upper]
    below = [value for value in dense_active_values if value <= sparse_active]
    above = [value for value in dense_active_values if value >= sparse_active]
    ratios = [value / sparse_active for value in dense_active_values]
    return {
        "sparse_arm": sparse_arm,
        "sparse_active_params_proxy": int(sparse_active),
        "dense_active_params_proxy": dense_active_values,
        "dense_to_sparse_active_ratios": ratios,
        "match_ratio": match_ratio,
        "matched_dense_count": len(matched),
        "bracketed_by_dense_ladder": bool(below and above),
        "matched_or_bracketed": bool(matched or (below and above)),
    }


def _preflight_rows(
    source_probe_dir: Path,
    source_summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    config_path: Path,
) -> list[dict[str, Any]]:
    return [
        _criterion("source_probe_present", bool(source_summary), "source transfer probe summary exists", str(source_probe_dir / "summary.json"), "missing source transfer probe summary"),
        _criterion("source_per_token_rows_present", bool(source_rows), "source transfer probe per-token metrics exist", str(source_probe_dir / "per_token_metrics.csv"), "missing source per-token metrics"),
        _criterion("config_present", config_path.is_file(), "benchmark config exists", str(config_path), "missing benchmark config"),
    ]


def _summary(
    *,
    status: str,
    decision: str,
    claim_status: str,
    start: float,
    source_probe_dir: Path,
    config_path: Path,
    train_steps: int,
    dense_steps: int,
    arm_rows: list[dict[str, Any]],
    per_token_rows: list[dict[str, Any]],
    norm_rows: list[dict[str, Any]],
    fingerprint_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    failures = [row for row in gate_rows if not row["passed"]]
    serial_arm_rows = [_serializable_row(row) for row in arm_rows]
    outcome_flags = _benchmark_outcome_flags(arm_rows)
    return {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "source_probe_dir": str(source_probe_dir),
        "config_path": str(config_path),
        "train_steps": train_steps,
        "dense_steps": dense_steps,
        "arm_count": len(arm_rows),
        "per_token_row_count": len(per_token_rows),
        "norm_sweep_row_count": len(norm_rows),
        "intervention_fingerprint_count": len(fingerprint_rows),
        "arm_metrics": serial_arm_rows,
        "benchmark_interpretation": outcome_flags,
        "gate_criteria": gate_rows,
        "failures": failures,
        "selected_next_step": _selected_next_step(status, failures),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "git_diff_hash": _git_diff_hash(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }


def _benchmark_outcome_flags(arm_rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms = {str(row.get("arm")): row for row in arm_rows}
    sparse = arms.get("sparse_contextual_topk2", {})
    sparse_delta = _float_or_none(sparse.get("heldout_delta_vs_base_ce"))
    sparse_active = _float_or_none(sparse.get("active_params_proxy"))
    dense = arms.get("rank_flop_matched_causal_dense", {})
    dense_delta = _float_or_none(dense.get("heldout_delta_vs_base_ce"))
    dense_l2 = _float_or_none(dense.get("heldout_residual_update_l2"))
    sparse_l2 = _float_or_none(sparse.get("heldout_residual_update_l2"))
    teacher_distilled = arms.get("sparse_teacher_distilled_norm_topk2", {})
    target_norm_teacher = arms.get("sparse_teacher_distilled_target_norm_topk2", {})
    oracle_teacher = arms.get("sparse_teacher_distilled_oracle_support_topk2", {})
    soft_teacher = arms.get("sparse_teacher_distilled_soft_temperature_topk2", {})
    token_position_teacher = arms.get("sparse_teacher_distilled_token_position_null", {})
    shuffled_teacher = arms.get("sparse_teacher_distilled_shuffled_teacher_null", {})
    teacher_distilled_delta = _float_or_none(teacher_distilled.get("heldout_delta_vs_base_ce"))
    target_norm_delta = _float_or_none(target_norm_teacher.get("heldout_delta_vs_base_ce"))
    oracle_teacher_delta = _float_or_none(oracle_teacher.get("heldout_delta_vs_base_ce"))
    soft_teacher_delta = _float_or_none(soft_teacher.get("heldout_delta_vs_base_ce"))
    token_position_teacher_delta = _float_or_none(token_position_teacher.get("heldout_delta_vs_base_ce"))
    shuffled_teacher_delta = _float_or_none(shuffled_teacher.get("heldout_delta_vs_base_ce"))
    teacher_distilled_mse = _float_or_none(teacher_distilled.get("teacher_residual_mse"))
    target_norm_mse = _float_or_none(target_norm_teacher.get("teacher_residual_mse"))
    oracle_teacher_mse = _float_or_none(oracle_teacher.get("teacher_residual_mse"))
    soft_teacher_mse = _float_or_none(soft_teacher.get("teacher_residual_mse"))
    token_position_teacher_mse = _float_or_none(token_position_teacher.get("teacher_residual_mse"))
    shuffled_teacher_mse = _float_or_none(shuffled_teacher.get("teacher_residual_mse"))
    hard_distill_mses = [
        ("current", teacher_distilled_mse),
        ("target_norm", target_norm_mse),
        ("oracle_support", oracle_teacher_mse),
    ]
    best_hard_distill = min(
        (
            {"arm": name, "teacher_residual_mse": value}
            for name, value in hard_distill_mses
            if value is not None
        ),
        key=lambda row: float(row["teacher_residual_mse"]),
        default={},
    )
    compute_matched_deltas = []
    if sparse_active is not None and sparse_active > 0:
        for row in arm_rows:
            if str(row.get("family")) != "dense":
                continue
            active = _float_or_none(row.get("active_params_proxy"))
            delta = _float_or_none(row.get("heldout_delta_vs_base_ce"))
            if active is None or delta is None:
                continue
            if sparse_active / 2.0 <= active <= sparse_active * 2.0:
                compute_matched_deltas.append({"arm": row.get("arm"), "active_params_proxy": active, "heldout_delta_vs_base_ce": delta})
    best_compute = min(
        compute_matched_deltas,
        key=lambda row: float(row["heldout_delta_vs_base_ce"]),
        default=None,
    )
    return {
        "dense_wins_l2_matched": (
            sparse_delta is not None
            and dense_delta is not None
            and sparse_l2 is not None
            and dense_l2 is not None
            and dense_l2 <= max(0.25, sparse_l2 * 2.0)
            and dense_delta < sparse_delta
        ),
        "dense_wins_compute_matched": (
            sparse_delta is not None
            and best_compute is not None
            and _float_or_none(best_compute.get("heldout_delta_vs_base_ce")) is not None
            and float(best_compute["heldout_delta_vs_base_ce"]) < sparse_delta
        ),
        "best_compute_matched_dense": best_compute or {},
        "compute_matched_dense_count": len(compute_matched_deltas),
        "teacher_distilled_sparse_beats_shuffled_teacher_null": (
            teacher_distilled_mse is not None
            and shuffled_teacher_mse is not None
            and teacher_distilled_delta is not None
            and shuffled_teacher_delta is not None
            and teacher_distilled_mse < shuffled_teacher_mse
            and teacher_distilled_delta < shuffled_teacher_delta
        ),
        "teacher_distilled_sparse_beats_default_sparse": (
            sparse_delta is not None
            and teacher_distilled_delta is not None
            and teacher_distilled_delta < sparse_delta
        ),
        "teacher_distilled_sparse_beats_l2_matched_dense": (
            dense_delta is not None
            and teacher_distilled_delta is not None
            and teacher_distilled_delta < dense_delta
        ),
        "teacher_distilled_gap_vs_default_sparse_ce_delta": _maybe_subtract(
            teacher_distilled_delta,
            sparse_delta,
        ),
        "teacher_distilled_gap_vs_l2_matched_dense_ce_delta": _maybe_subtract(
            teacher_distilled_delta,
            dense_delta,
        ),
        "teacher_distilled_mse_margin_vs_shuffled_teacher": _maybe_subtract(
            shuffled_teacher_mse,
            teacher_distilled_mse,
        ),
        "target_norm_distill_mse_margin_vs_current": _maybe_subtract(
            teacher_distilled_mse,
            target_norm_mse,
        ),
        "oracle_support_distill_mse_margin_vs_target_norm": _maybe_subtract(
            target_norm_mse,
            oracle_teacher_mse,
        ),
        "soft_topk_distill_mse_margin_vs_best_hard_sparse": _maybe_subtract(
            _float_or_none(best_hard_distill.get("teacher_residual_mse")),
            soft_teacher_mse,
        ),
        "target_norm_distill_gap_vs_default_sparse_ce_delta": _maybe_subtract(
            target_norm_delta,
            sparse_delta,
        ),
        "oracle_support_distill_gap_vs_default_sparse_ce_delta": _maybe_subtract(
            oracle_teacher_delta,
            sparse_delta,
        ),
        "soft_topk_distill_gap_vs_default_sparse_ce_delta": _maybe_subtract(
            soft_teacher_delta,
            sparse_delta,
        ),
        "target_norm_distill_beats_token_position_null": (
            target_norm_mse is not None
            and token_position_teacher_mse is not None
            and target_norm_delta is not None
            and token_position_teacher_delta is not None
            and target_norm_mse < token_position_teacher_mse
            and target_norm_delta < token_position_teacher_delta
        ),
        "best_hard_sparse_teacher_distill_by_mse": best_hard_distill,
        "columnability_gate_interpretation": _columnability_interpretation(
            default_sparse_delta=sparse_delta,
            dense_teacher_delta=dense_delta,
            current_mse=teacher_distilled_mse,
            target_norm_mse=target_norm_mse,
            oracle_support_mse=oracle_teacher_mse,
            soft_mse=soft_teacher_mse,
            token_position_mse=token_position_teacher_mse,
            shuffled_teacher_mse=shuffled_teacher_mse,
        ),
        "teacher_distilled_heldout_cosine_to_teacher": _float_or_none(
            teacher_distilled.get("teacher_residual_cosine")
        ),
        "teacher_distilled_heldout_norm_ratio_to_teacher": _float_or_none(
            teacher_distilled.get("heldout_residual_l2_ratio_to_teacher")
        ),
    }


def _selected_next_step(status: str, failures: list[dict[str, Any]]) -> str:
    if status == "pass":
        return "escalate the common benchmark to a seed-2 or RunPod repeat only if sparse beats dense"
    failed = {str(row.get("criterion")) for row in failures}
    if (
        "target_norm_distill_beats_current_distill_mse" in failed
        or "oracle_support_distill_tests_discovery_bottleneck" in failed
        or "soft_topk_distill_tests_representation_ceiling" in failed
        or "teacher_distill_beats_token_position_null" in failed
    ):
        return "synthesize the local columnability/discovery gate before any RunPod repeat or sparse-support identity claim"
    if "active_compute_matched_or_bracketed" in failed:
        return "repair local compute-matched dense bottleneck ladder and sparse teacher-distilled rescue controls"
    if "teacher_distilled_sparse_beats_shuffled_teacher_null" in failed:
        return "inspect sparse teacher-distillation objective and null construction before any RunPod repeat"
    if "sparse_beats_causal_dense" in failed:
        return "synthesize the negative sparse teacher-distilled rescue result and decide whether to retire sparse-support identity claims or test a stronger sparse rescue objective"
    return "inspect failed common-benchmark gate criteria and repair the lowest-level local artifact issue"


def _columnability_interpretation(
    *,
    default_sparse_delta: float | None,
    dense_teacher_delta: float | None,
    current_mse: float | None,
    target_norm_mse: float | None,
    oracle_support_mse: float | None,
    soft_mse: float | None,
    token_position_mse: float | None,
    shuffled_teacher_mse: float | None,
) -> str:
    hard_best = _min_present([current_mse, target_norm_mse, oracle_support_mse])
    if hard_best is None or soft_mse is None:
        return "columnability_gate_missing_required_teacher_mse"
    if shuffled_teacher_mse is not None and hard_best >= shuffled_teacher_mse:
        return "sparse_teacher_distill_not_separated_from_shuffled_teacher_null"
    if token_position_mse is not None and target_norm_mse is not None and target_norm_mse >= token_position_mse:
        return "learned_sparse_distill_not_separated_from_token_position_null"
    if oracle_support_mse is not None and target_norm_mse is not None and oracle_support_mse < target_norm_mse:
        return "support_discovery_bottleneck_candidate"
    if soft_mse < hard_best:
        return "hard_support_discretization_or_columnability_bottleneck_candidate"
    if (
        default_sparse_delta is not None
        and dense_teacher_delta is not None
        and default_sparse_delta > dense_teacher_delta
    ):
        return "sparse_columns_remain_below_dense_teacher_after_local_columnability_gate"
    return "sparse_columnability_not_rejected_by_local_gate"


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    arm_rows: list[dict[str, Any]],
    per_token_rows: list[dict[str, Any]],
    norm_rows: list[dict[str, Any]],
    fingerprint_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "arm_metrics.csv", [_serializable_row(row) for row in arm_rows])
    _write_csv(out_dir / "per_token_metrics.csv", per_token_rows)
    _write_csv(out_dir / "norm_sweep.csv", norm_rows)
    _write_csv(out_dir / "intervention_fingerprints.csv", fingerprint_rows)
    _write_csv(out_dir / "gate_criteria.csv", gate_rows)
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [{"status": "missing"}]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ACSR Common Causal Residual Benchmark",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Train steps: `{summary['train_steps']}`",
        f"- Dense steps: `{summary['dense_steps']}`",
        "",
        "This local benchmark compares sparse contextual supports, rank/FLOP-matched dense residuals, token-position nulls, frequency/shuffled support nulls, and an exhaustive oracle support arm on one common batch and held-out split.",
    ]
    if summary.get("failures"):
        lines.extend(["", "## Failures"])
        for row in summary["failures"]:
            lines.append(f"- `{row['criterion']}`: {row['failure_reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _serializable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"per_token_losses", "residual_update_l2_per_token", "residual_update_tensor"}
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _maybe_subtract(left: Any, right: Any) -> float | str:
    left_value = _float_or_none(left)
    right_value = _float_or_none(right)
    if left_value is None or right_value is None:
        return ""
    return left_value - right_value


def _min_present(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _norm_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.25:
        return "low"
    if value < 0.75:
        return "medium"
    return "high"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        return False


def _git_diff_hash() -> str:
    try:
        diff = subprocess.check_output(["git", "diff", "--binary"], stderr=subprocess.DEVNULL)
    except Exception:
        return ""
    if not diff:
        return ""
    import hashlib

    return hashlib.sha256(diff).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-probe-dir", type=Path, default=DEFAULT_SOURCE_PROBE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-steps", type=int, default=12)
    parser.add_argument("--dense-steps", type=int, default=80)
    args = parser.parse_args()
    summary = run_acsr_common_causal_residual_benchmark(
        source_probe_dir=args.source_probe_dir,
        config_path=args.config,
        out_dir=args.out,
        train_steps=args.train_steps,
        dense_steps=args.dense_steps,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
