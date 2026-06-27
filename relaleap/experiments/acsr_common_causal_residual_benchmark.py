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

    frequency_support = _frequency_matched_random_support(
        torch,
        topk2_support,
        num_columns=num_columns,
        seed=seed + 17,
    )
    shuffled_support = _shuffle_support(torch, topk2_support, seed=seed + 29)
    token_position_support = _token_position_support(torch, topk2_support, num_columns)
    oracle_support = _oracle_support_topk(torch, F, base, topk2, hidden, targets, vocab_size, topk2_scores, k=2)
    sparse_arms = [
        contextual2,
        contextual1,
        _eval_sparse_arm(torch, F, base, topk2, hidden, targets, vocab_size, "sparse_frequency_matched_random", frequency_support, topk2_scores, base_losses, base_loss, heldout_base, mask),
        _eval_sparse_arm(torch, F, base, topk2, hidden, targets, vocab_size, "sparse_shuffled_support_marginals", shuffled_support, topk2_scores, base_losses, base_loss, heldout_base, mask),
        _eval_sparse_arm(torch, F, base, topk2, hidden, targets, vocab_size, "sparse_token_position_null", token_position_support, topk2_scores, base_losses, base_loss, heldout_base, mask),
        _eval_sparse_arm(torch, F, base, topk2, hidden, targets, vocab_size, "sparse_oracle_support", oracle_support, topk2_scores, base_losses, base_loss, heldout_base, mask),
    ]
    arm_rows = [_base_arm(base_loss, heldout_base)] + sparse_arms + [dense_causal, dense_position]
    per_token_rows = _per_token_rows(
        torch,
        base_losses,
        {
            "sparse_contextual_topk2": contextual2["per_token_losses"],
            "sparse_rank_matched_topk1": contextual1["per_token_losses"],
            "rank_flop_matched_causal_dense": dense_causal_losses,
            "rank_flop_matched_token_position_dense": dense_position_losses,
        },
        {
            "sparse_contextual_topk2": contextual2["residual_update_l2_per_token"],
            "sparse_rank_matched_topk1": contextual1["residual_update_l2_per_token"],
            "rank_flop_matched_causal_dense": dense_causal_l2,
            "rank_flop_matched_token_position_dense": dense_position_l2,
        },
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
) -> tuple[dict[str, Any], Any, Any]:
    split = max(1, int(features.shape[1]) // 2)
    rank = max(1, target_parameter_count // max(1, int(features.shape[-1]) + int(hidden.shape[-1])))
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
    for row in arm_rows:
        if row.get("arm") == "base_no_residual":
            continue
        ce_delta = _float_or_none(row.get("heldout_delta_vs_base_ce"))
        l2 = _float_or_none(row.get("heldout_residual_update_l2"))
        rows.append(
            {
                "arm": row["arm"],
                "heldout_delta_vs_base_ce": ce_delta,
                "heldout_residual_update_l2": l2,
                "ce_gain_per_l2": "" if l2 is None or abs(l2) < 1e-12 or ce_delta is None else ce_delta / l2,
                "norm_bucket": _norm_bucket(l2),
            }
        )
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
        "rank_flop_matched_causal_dense",
        "rank_flop_matched_token_position_dense",
        "sparse_frequency_matched_random",
        "sparse_shuffled_support_marginals",
        "sparse_token_position_null",
        "sparse_oracle_support",
    }
    sparse = arms.get("sparse_contextual_topk2", {})
    dense = arms.get("rank_flop_matched_causal_dense", {})
    token_dense = arms.get("rank_flop_matched_token_position_dense", {})
    random = arms.get("sparse_frequency_matched_random", {})
    oracle = arms.get("sparse_oracle_support", {})
    sparse_delta = _float_or_none(sparse.get("heldout_delta_vs_base_ce"))
    dense_delta = _float_or_none(dense.get("heldout_delta_vs_base_ce"))
    token_dense_delta = _float_or_none(token_dense.get("heldout_delta_vs_base_ce"))
    sparse_l2 = _float_or_none(sparse.get("heldout_residual_update_l2"))
    dense_l2 = _float_or_none(dense.get("heldout_residual_update_l2"))
    random_damage = _maybe_subtract(random.get("heldout_ce_loss"), sparse.get("heldout_ce_loss"))
    oracle_regret = _maybe_subtract(sparse.get("heldout_ce_loss"), oracle.get("heldout_ce_loss"))
    return [
        _criterion("common_baselines_present", required.issubset(arms), "all required sparse/dense/null/oracle arms exist", sorted(arms), "missing one or more required common-baseline arms"),
        _criterion("dense_norm_matched", sparse_l2 is not None and dense_l2 is not None and dense_l2 <= max(0.25, sparse_l2 * 2.0), "causal dense held-out residual L2 is within 2x sparse top-k2 L2 with 0.25 floor", {"sparse_l2": sparse_l2, "dense_l2": dense_l2}, "causal dense control used too much residual norm"),
        _criterion("causal_dense_beats_token_position_null", dense_delta is not None and token_dense_delta is not None and dense_delta < token_dense_delta, "causal dense should beat token-position dense null", {"causal_dense_delta": dense_delta, "token_position_dense_delta": token_dense_delta}, "causal dense did not beat token-position dense null"),
        _criterion("sparse_beats_causal_dense", sparse_delta is not None and dense_delta is not None and sparse_delta < dense_delta, "sparse top-k2 held-out CE delta must beat rank/FLOP-matched causal dense", {"sparse_delta": sparse_delta, "dense_delta": dense_delta}, "sparse top-k2 did not beat causal dense"),
        _criterion("support_identity_matters", isinstance(random_damage, float) and random_damage > 0.0, "frequency-matched random support should hurt held-out CE versus selected sparse support", random_damage, "selected support was not better than frequency-matched random support"),
        _criterion("oracle_regret_nonnegative", isinstance(oracle_regret, float) and oracle_regret >= -1e-8, "exhaustive oracle support should not be worse than selected support", oracle_regret, "oracle support sanity check failed"),
        _criterion("intervention_fingerprints_present", len(fingerprint_rows) >= 5, "fingerprint rows include oracle/random/support-overlap observables", len(fingerprint_rows), "missing intervention fingerprint rows"),
    ]


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
        "gate_criteria": gate_rows,
        "failures": failures,
        "selected_next_step": (
            "escalate the common benchmark to a seed-2 or RunPod repeat only if sparse beats dense"
            if status == "pass"
            else "defer ACSR promotion and improve sparse mechanism controls or pivot to dense-teacher distillation"
        ),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }


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
        if key not in {"per_token_losses", "residual_update_l2_per_token"}
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
