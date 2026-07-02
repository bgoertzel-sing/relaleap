"""Support-quality audit for the causal contextual top-k-2 router."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.contextual_router_sequence_kfold_ablation import (
    _forward_with_feature_ablation,
)
from relaleap.experiments.run import _load_torch_info
from relaleap.experiments.run import _read_config
from relaleap.experiments.support_audit import _ce_loss
from relaleap.experiments.support_audit import _configured_residual_loss
from relaleap.experiments.support_audit import _score_for_support
from relaleap.experiments.support_audit import _support_key
from relaleap.experiments.support_audit import _token_losses
from relaleap.smoke import ResidualColumns
from relaleap.smoke import TinyCharTransformer
from relaleap.smoke import _build_batch
from relaleap.smoke import _residual_support_audit
from relaleap.smoke import _state_dict_delta


DEFAULT_CONFIG = Path(
    "configs/token_larger_support_wide_causal_contextual_router_hep_temporal_clipped_objective_gate.yaml"
)
DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_causal_contextual_router_support_audit"
)

CAUSAL_SUPPORT_AUDIT_PASSED = "causal_contextual_router_support_audit_passed"
CAUSAL_SUPPORT_AUDIT_BLOCKED = "causal_contextual_router_support_audit_blocks_promotion"


def run_causal_contextual_router_support_audit(
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    max_folds: int | None = None,
    random_seed: int = 1701,
) -> dict[str, Any]:
    """Train sequence-heldout folds and audit actual support quality."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("causal contextual router support audit requires torch") from exc

    start = time.time()
    config = _read_config(config_path)
    run_cfg = config.get("run", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    base_cfg = model_cfg.get("base", {})
    column_cfg = model_cfg.get("columns", {})
    training_cfg = config.get("training", {})

    seed = int(run_cfg.get("seed", 1))
    max_steps = int(run_cfg.get("max_steps", 10))
    learning_rate = float(run_cfg.get("learning_rate", 1e-2))
    experiment_id = str(run_cfg.get("experiment_id", config_path.stem))
    residual_objective = str(training_cfg.get("residual_objective", "supervised_ce"))
    dataset = str(data_cfg.get("dataset", "tiny_shakespeare_char"))
    seq_len = int(data_cfg.get("seq_len", 32))
    hidden_dim = int(base_cfg.get("hidden_dim", 32))
    layers = int(base_cfg.get("layers", 2))
    num_columns = int(column_cfg.get("num_columns", 8))
    atoms_per_column = int(column_cfg.get("atoms_per_column", 4))
    top_k = int(column_cfg.get("top_k", 1))
    support_router = str(column_cfg.get("support_router", "linear"))
    contextual_router_hidden_dim = int(
        column_cfg.get("contextual_router_hidden_dim", hidden_dim * 2)
    )
    if top_k != 2:
        raise ValueError("causal support audit expects model.columns.top_k: 2")
    if support_router != "contextual_mlp_causal":
        raise ValueError(
            "causal support audit expects model.columns.support_router: contextual_mlp_causal"
        )
    if max_folds is not None and max_folds < 1:
        raise ValueError("max_folds must be positive when set")

    torch.manual_seed(seed)
    inputs, targets, vocab_size = _build_batch(dataset=dataset, seq_len=seq_len, batch_size=4)
    fold_count = int(inputs.shape[0]) if max_folds is None else min(max_folds, int(inputs.shape[0]))
    all_pairs = list(itertools.combinations(range(num_columns), top_k))
    fold_rows: list[dict[str, Any]] = []
    support_count_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    per_token_label_rows: list[dict[str, Any]] = []
    final_delta: float | None = None
    final_support_audit: dict[str, Any] | None = None

    for fold_index in range(fold_count):
        train_indices = [index for index in range(int(inputs.shape[0])) if index != fold_index]
        train_inputs = inputs[train_indices]
        train_targets = targets[train_indices]
        holdout_inputs = inputs[fold_index : fold_index + 1]
        holdout_targets = targets[fold_index : fold_index + 1]
        for spec in _control_specs(contextual_router_hidden_dim):
            torch.manual_seed(seed)
            base = TinyCharTransformer(
                vocab_size=vocab_size,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                layers=layers,
            )
            base.eval()
            for parameter in base.parameters():
                parameter.requires_grad_(False)
            residual = ResidualColumns(
                hidden_dim=hidden_dim,
                num_columns=num_columns,
                atoms_per_column=atoms_per_column,
                top_k=spec["top_k"],
                support_router=spec["support_router"],
                contextual_router_hidden_dim=contextual_router_hidden_dim,
            )
            residual.train()
            before_residual = {
                key: value.detach().clone() for key, value in residual.state_dict().items()
            }
            optimizer = torch.optim.AdamW(residual.parameters(), lr=learning_rate)
            for _ in range(max_steps):
                optimizer.zero_grad(set_to_none=True)
                loss = _configured_residual_loss(
                    base,
                    residual,
                    train_inputs,
                    train_targets,
                    vocab_size,
                    residual_objective=residual_objective,
                    training_cfg=training_cfg,
                )
                loss.backward()
                optimizer.step()
            residual.eval()
            with torch.no_grad():
                hidden = base.encode(holdout_inputs)
                logits, support = _forward_with_feature_ablation(
                    base,
                    residual,
                    hidden,
                    feature_mask=None,
                )
                router_token_losses = _token_losses(logits, holdout_targets).reshape(-1)
                router_loss = float(router_token_losses.mean().item())
                empty_logits = base.decode(hidden)
                empty_loss = _ce_loss(empty_logits, holdout_targets, vocab_size)
                pair_rows = [
                    _score_for_support(
                        base,
                        residual,
                        hidden,
                        holdout_targets,
                        vocab_size,
                        support=pair,
                        empty_loss=empty_loss,
                        router_loss=router_loss,
                    )
                    for pair in all_pairs
                ]
                audit = _support_quality_metrics(
                    base=base,
                    residual=residual,
                    hidden=hidden,
                    targets=holdout_targets,
                    vocab_size=vocab_size,
                    support=support,
                    router_token_losses=router_token_losses,
                    router_loss=router_loss,
                    empty_loss=empty_loss,
                    pair_rows=pair_rows,
                    random_seed=random_seed + fold_index,
                )
            row = {
                "fold": fold_index,
                "heldout_sequence_index": fold_index,
                "control": spec["name"],
                "support_router": spec["support_router"],
                "top_k": spec["top_k"],
                "causal_feature_safe": spec["causal_feature_safe"],
                "positions": int(router_token_losses.numel()),
                **audit["metrics"],
            }
            fold_rows.append(row)
            control_rows.extend(
                {"fold": fold_index, "control": spec["name"], **item}
                for item in audit["controls"]
            )
            per_token_label_rows.extend(
                {"fold": fold_index, "control": spec["name"], **item}
                for item in audit["per_token_support_labels"]
            )
            support_count_rows.extend(
                {
                    "fold": fold_index,
                    "control": spec["name"],
                    "support": key,
                    "count": count,
                }
                for key, count in audit["support_counts"].items()
            )
            if spec["name"] == "causal_contextual_topk2":
                final_delta = _state_dict_delta(before_residual, residual)
                final_support_audit = _residual_support_audit(base, residual, holdout_inputs)

    aggregate_rows = _aggregate_rows(fold_rows)
    decision = _decision(aggregate_rows)
    summary = {
        "status": "pass",
        "decision": decision["decision"],
        "claim_status": decision["claim_status"],
        "selected_next_step": decision["next_step"],
        "experiment_id": f"{experiment_id}_support_audit",
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        **_load_torch_info(),
        "audit": {
            "dataset": dataset,
            "seq_len": seq_len,
            "batch_size": int(inputs.shape[0]),
            "fold_count": fold_count,
            "vocab_size": vocab_size,
            "training_steps": max_steps,
            "residual_objective": residual_objective,
            "num_columns": num_columns,
            "atoms_per_column": atoms_per_column,
            "top_k": top_k,
            "support_router": support_router,
            "support_set_count": len(all_pairs),
            "fold_metrics": fold_rows,
            "aggregate_metrics": {row["control"]: row for row in aggregate_rows},
            "gate_criteria": decision["criteria"],
            "failures": decision["failures"],
            "rationale": decision["rationale"],
            "per_token_support_label_rows": len(per_token_label_rows),
            "support_audit_last_causal_fold": final_support_audit,
            "residual_parameter_delta_last_causal_fold": final_delta,
        },
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "fold_metrics_csv": str(out_dir / "fold_metrics.csv"),
            "aggregate_metrics_csv": str(out_dir / "aggregate_metrics.csv"),
            "control_metrics_csv": str(out_dir / "control_metrics.csv"),
            "per_token_support_labels_csv": str(out_dir / "per_token_support_labels.csv"),
            "support_counts_csv": str(out_dir / "support_counts.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
        "git_commit": _git_commit(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "fold_metrics.csv", fold_rows)
    _write_csv(out_dir / "aggregate_metrics.csv", aggregate_rows)
    _write_csv(out_dir / "control_metrics.csv", control_rows)
    _write_csv(out_dir / "per_token_support_labels.csv", per_token_label_rows)
    _write_csv(out_dir / "support_counts.csv", support_count_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _control_specs(contextual_router_hidden_dim: int) -> list[dict[str, Any]]:
    del contextual_router_hidden_dim
    return [
        {
            "name": "causal_contextual_topk2",
            "support_router": "contextual_mlp_causal",
            "top_k": 2,
            "causal_feature_safe": True,
        },
        {
            "name": "linear_topk2",
            "support_router": "linear",
            "top_k": 2,
            "causal_feature_safe": True,
        },
        {
            "name": "full_context_oracle_topk2",
            "support_router": "contextual_mlp",
            "top_k": 2,
            "causal_feature_safe": False,
        },
    ]


def _support_quality_metrics(
    *,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    support: Any,
    router_token_losses: Any,
    router_loss: float,
    empty_loss: float,
    pair_rows: list[dict[str, Any]],
    random_seed: int,
) -> dict[str, Any]:
    import torch

    token_loss_matrix = torch.stack([row["_token_losses"].reshape(-1) for row in pair_rows], dim=1)
    oracle_losses, oracle_indices = token_loss_matrix.min(dim=1)
    oracle_loss = float(oracle_losses.mean().item())
    oracle_regret = router_token_losses - oracle_losses
    support_keys = [
        _support_key(tuple(sorted(int(value) for value in row)))
        for row in support.reshape(-1, support.shape[-1]).detach().cpu().tolist()
    ]
    support_counts: dict[str, int] = {}
    for key in support_keys:
        support_counts[key] = support_counts.get(key, 0) + 1
    dominant_key, dominant_count = max(support_counts.items(), key=lambda item: item[1])
    pair_index_by_key = {str(row["support_key"]): index for index, row in enumerate(pair_rows)}
    dominant_loss = float(pair_rows[pair_index_by_key[dominant_key]]["loss"])
    best_fixed_row = min(pair_rows, key=lambda row: float(row["loss"]))
    generator = torch.Generator(device=token_loss_matrix.device)
    generator.manual_seed(random_seed)
    random_indices = torch.randint(
        low=0,
        high=len(pair_rows),
        size=(token_loss_matrix.shape[0],),
        generator=generator,
        device=token_loss_matrix.device,
    )
    random_loss = float(
        token_loss_matrix[
            torch.arange(token_loss_matrix.shape[0], device=token_loss_matrix.device),
            random_indices,
        ].mean().item()
    )
    shuffled_support = _shuffled_support(support, seed=random_seed + 11)
    shuffled_logits = base.decode(residual(hidden, support_indices=shuffled_support))
    shuffled_loss = _ce_loss(shuffled_logits, targets, vocab_size)
    functional_churn = _functional_churn(base, residual, hidden, support)
    support_change_fraction = _support_change_fraction(support)
    entropy = _normalized_load_entropy(support, residual.num_columns)
    router_gap = router_loss - oracle_loss
    linear_gap_recovery_denominator = empty_loss - oracle_loss
    metrics = {
        "empty_loss": empty_loss,
        "router_loss": router_loss,
        "oracle_loss": oracle_loss,
        "oracle_support_regret": float(oracle_regret.mean().item()),
        "oracle_support_regret_positive_fraction": float(
            (oracle_regret > 0).to(dtype=torch.float32).mean().item()
        ),
        "router_oracle_gap": router_gap,
        "recovery_fraction_vs_empty": _fraction(empty_loss - router_loss, linear_gap_recovery_denominator),
        "best_global_fixed_support": str(best_fixed_row["support_key"]),
        "best_global_fixed_support_loss": float(best_fixed_row["loss"]),
        "dominant_fixed_support": dominant_key,
        "dominant_fixed_support_count": dominant_count,
        "dominant_fixed_support_loss": dominant_loss,
        "random_support_loss": random_loss,
        "shuffled_support_loss": shuffled_loss,
        "used_columns": _used_columns(support),
        "dead_columns": int(residual.num_columns) - _used_columns(support),
        "unique_support_sets": len(support_counts),
        "support_change_fraction": support_change_fraction,
        "functional_churn_logit_l1": functional_churn,
        "support_load_entropy": entropy,
    }
    controls = [
        {"variant": "actual_router", "loss": router_loss, "delta_from_router": 0.0},
        {
            "variant": "oracle_per_token",
            "loss": oracle_loss,
            "delta_from_router": oracle_loss - router_loss,
        },
        {
            "variant": "best_global_fixed_support",
            "support": str(best_fixed_row["support_key"]),
            "loss": float(best_fixed_row["loss"]),
            "delta_from_router": float(best_fixed_row["loss"]) - router_loss,
        },
        {
            "variant": "dominant_fixed_support",
            "support": dominant_key,
            "loss": dominant_loss,
            "delta_from_router": dominant_loss - router_loss,
        },
        {"variant": "random_support", "loss": random_loss, "delta_from_router": random_loss - router_loss},
        {
            "variant": "shuffled_router_support",
            "loss": shuffled_loss,
            "delta_from_router": shuffled_loss - router_loss,
        },
    ]
    per_token_support_labels = _per_token_support_label_rows(
        support=support,
        targets=targets,
        router_token_losses=router_token_losses,
        token_loss_matrix=token_loss_matrix,
        oracle_indices=oracle_indices,
        pair_rows=pair_rows,
    )
    return {
        "metrics": metrics,
        "controls": controls,
        "support_counts": support_counts,
        "per_token_support_labels": per_token_support_labels,
    }


def _per_token_support_label_rows(
    *,
    support: Any,
    targets: Any,
    router_token_losses: Any,
    token_loss_matrix: Any,
    oracle_indices: Any,
    pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return per-token oracle and one-swap support labels for route-only training."""

    pair_supports = [tuple(int(value) for value in row["support"]) for row in pair_rows]
    pair_keys = [str(row["support_key"]) for row in pair_rows]
    pair_index_by_key = {key: index for index, key in enumerate(pair_keys)}
    effective_seq_len = int(targets.shape[1]) - 1
    flat_targets = targets[:, :-1].reshape(-1).detach().cpu().tolist()
    actual_supports = support[:, :-1, :].reshape(-1, support.shape[-1]).detach().cpu().tolist()
    rows: list[dict[str, Any]] = []
    for flat_position, support_values in enumerate(actual_supports):
        actual_support = tuple(sorted(int(value) for value in support_values))
        actual_key = _support_key(actual_support)
        actual_index = pair_index_by_key.get(actual_key)
        oracle_index = int(oracle_indices[flat_position].detach().cpu().item())
        one_swap_index = _best_one_swap_index(
            token_loss_matrix=token_loss_matrix,
            flat_position=flat_position,
            pair_supports=pair_supports,
            actual_support=actual_support,
            fallback_index=actual_index,
        )
        router_loss = float(router_token_losses[flat_position].detach().cpu().item())
        actual_loss = (
            router_loss
            if actual_index is None
            else float(token_loss_matrix[flat_position, actual_index].detach().cpu().item())
        )
        oracle_loss = float(token_loss_matrix[flat_position, oracle_index].detach().cpu().item())
        one_swap_loss = (
            None
            if one_swap_index is None
            else float(token_loss_matrix[flat_position, one_swap_index].detach().cpu().item())
        )
        rows.append(
            {
                "flat_position": flat_position,
                "sequence_index": flat_position // effective_seq_len,
                "position_index": flat_position % effective_seq_len,
                "target_token": int(flat_targets[flat_position]),
                "actual_support": actual_key,
                "actual_support_loss": actual_loss,
                "router_support_loss": router_loss,
                "oracle_support": pair_keys[oracle_index],
                "oracle_support_loss": oracle_loss,
                "oracle_support_regret": router_loss - oracle_loss,
                "oracle_support_exact_match": actual_key == pair_keys[oracle_index],
                "best_one_swap_support": "" if one_swap_index is None else pair_keys[one_swap_index],
                "best_one_swap_support_loss": "" if one_swap_loss is None else one_swap_loss,
                "best_one_swap_regret": "" if one_swap_loss is None else one_swap_loss - oracle_loss,
                "best_one_swap_improves_actual": (
                    False if one_swap_loss is None else one_swap_loss < actual_loss
                ),
                "best_one_swap_gain_vs_actual": (
                    "" if one_swap_loss is None else actual_loss - one_swap_loss
                ),
                "one_swap_label_is_oracle": (
                    False if one_swap_index is None else pair_keys[one_swap_index] == pair_keys[oracle_index]
                ),
            }
        )
    return rows


def _best_one_swap_index(
    *,
    token_loss_matrix: Any,
    flat_position: int,
    pair_supports: list[tuple[int, ...]],
    actual_support: tuple[int, ...],
    fallback_index: int | None,
) -> int | None:
    selected = set(actual_support)
    candidates = [
        index
        for index, support in enumerate(pair_supports)
        if len(set(support).intersection(selected)) == len(actual_support) - 1
    ]
    if not candidates:
        return fallback_index
    return min(
        candidates,
        key=lambda index: float(token_loss_matrix[flat_position, index].detach().cpu().item()),
    )


def _shuffled_support(support: Any, *, seed: int) -> Any:
    import torch

    flat = support.reshape(-1, support.shape[-1])
    generator = torch.Generator(device=flat.device)
    generator.manual_seed(seed)
    permutation = torch.randperm(flat.shape[0], generator=generator, device=flat.device)
    return flat[permutation].reshape_as(support)


def _functional_churn(base: Any, residual: Any, hidden: Any, support: Any) -> float:
    import torch

    if support.shape[1] < 2:
        return 0.0
    previous_support = torch.cat([support[:, :1, :], support[:, :-1, :]], dim=1)
    current_logits = base.decode(residual(hidden, support_indices=support))
    previous_logits = base.decode(residual(hidden, support_indices=previous_support))
    delta = (current_logits[:, 1:, :] - previous_logits[:, 1:, :]).abs().mean()
    return float(delta.item())


def _support_change_fraction(support: Any) -> float:
    rows = [
        tuple(sorted(int(value) for value in row))
        for row in support.reshape(-1, support.shape[-1]).detach().cpu().tolist()
    ]
    if len(rows) <= 1:
        return 0.0
    changes = sum(1 for left, right in zip(rows, rows[1:]) if left != right)
    return changes / (len(rows) - 1)


def _normalized_load_entropy(support: Any, num_columns: int) -> float:
    counts = [0 for _ in range(num_columns)]
    for value in support.reshape(-1).detach().cpu().tolist():
        counts[int(value)] += 1
    total = sum(counts)
    if total <= 0 or num_columns <= 1:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(num_columns)


def _used_columns(support: Any) -> int:
    return len({int(value) for value in support.reshape(-1).detach().cpu().tolist()})


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    controls = sorted({str(row["control"]) for row in rows})
    numeric_fields = [
        "empty_loss",
        "router_loss",
        "oracle_loss",
        "oracle_support_regret",
        "oracle_support_regret_positive_fraction",
        "router_oracle_gap",
        "recovery_fraction_vs_empty",
        "best_global_fixed_support_loss",
        "dominant_fixed_support_loss",
        "random_support_loss",
        "shuffled_support_loss",
        "used_columns",
        "dead_columns",
        "unique_support_sets",
        "support_change_fraction",
        "functional_churn_logit_l1",
        "support_load_entropy",
    ]
    aggregates = []
    for control in controls:
        group = [row for row in rows if row["control"] == control]
        aggregate = {
            "control": control,
            "folds": len(group),
            "support_router": group[0]["support_router"],
            "top_k": group[0]["top_k"],
            "causal_feature_safe": group[0]["causal_feature_safe"],
        }
        for field in numeric_fields:
            aggregate[f"mean_{field}"] = _mean(group, field)
        aggregates.append(aggregate)
    by_control = {row["control"]: row for row in aggregates}
    causal = by_control.get("causal_contextual_topk2")
    linear = by_control.get("linear_topk2")
    full = by_control.get("full_context_oracle_topk2")
    if causal and linear:
        causal["mean_router_loss_delta_vs_linear"] = (
            causal["mean_router_loss"] - linear["mean_router_loss"]
        )
        causal["mean_oracle_regret_delta_vs_linear"] = (
            causal["mean_oracle_support_regret"] - linear["mean_oracle_support_regret"]
        )
        causal["mean_functional_churn_delta_vs_linear"] = (
            causal["mean_functional_churn_logit_l1"] - linear["mean_functional_churn_logit_l1"]
        )
    if causal and full:
        causal["mean_router_loss_delta_vs_full_context_oracle"] = (
            causal["mean_router_loss"] - full["mean_router_loss"]
        )
    return aggregates


def _decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_control = {row["control"]: row for row in rows}
    causal = by_control.get("causal_contextual_topk2", {})
    linear = by_control.get("linear_topk2", {})
    criteria = [
        _criterion(
            "causal_beats_linear_sequence_heldout_ce",
            _value(causal, "mean_router_loss") < _value(linear, "mean_router_loss"),
            "causal mean router CE < linear mean router CE",
            _delta_text(causal, linear, "mean_router_loss"),
        ),
        _criterion(
            "causal_reduces_oracle_support_regret_vs_linear",
            _value(causal, "mean_oracle_support_regret")
            < _value(linear, "mean_oracle_support_regret"),
            "causal mean oracle-support regret < linear mean oracle-support regret",
            _delta_text(causal, linear, "mean_oracle_support_regret"),
        ),
        _criterion(
            "causal_no_functional_churn_increase_vs_linear",
            _value(causal, "mean_functional_churn_logit_l1")
            <= _value(linear, "mean_functional_churn_logit_l1"),
            "causal mean functional churn <= linear mean functional churn",
            _delta_text(causal, linear, "mean_functional_churn_logit_l1"),
        ),
        _criterion(
            "causal_beats_random_and_dominant_controls",
            _value(causal, "mean_router_loss")
            < min(
                _value(causal, "mean_random_support_loss"),
                _value(causal, "mean_dominant_fixed_support_loss"),
            ),
            "causal router CE < random and dominant fixed support CE",
            {
                "router": causal.get("mean_router_loss"),
                "random": causal.get("mean_random_support_loss"),
                "dominant": causal.get("mean_dominant_fixed_support_loss"),
            },
        ),
    ]
    failures = [row for row in criteria if not row["passed"]]
    if failures:
        return {
            "decision": CAUSAL_SUPPORT_AUDIT_BLOCKED,
            "claim_status": "causal_contextual_router_ce_supported_support_quality_not_established",
            "next_step": "inspect causal router oracle-regret and churn failure before any default promotion",
            "criteria": criteria,
            "failures": failures,
            "rationale": (
                "The causal contextual router remains a strong CE router, but this "
                "audit does not establish support-quality promotion evidence because "
                "at least one oracle-regret, churn, or matched-control gate failed."
            ),
        }
    return {
        "decision": CAUSAL_SUPPORT_AUDIT_PASSED,
        "claim_status": "causal_contextual_router_support_quality_supported_not_promoted",
        "next_step": "run a backend repeat or broader dataset support audit before default promotion",
        "criteria": criteria,
        "failures": [],
        "rationale": (
            "The causal contextual router beats linear on CE while also reducing "
            "oracle-support regret and avoiding increased functional churn."
        ),
    }


def _criterion(name: str, passed: bool, threshold: Any, actual: Any) -> dict[str, Any]:
    return {"criterion": name, "passed": bool(passed), "threshold": threshold, "actual": actual}


def _value(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        return float("inf")
    return float(value)


def _delta_text(left: dict[str, Any], right: dict[str, Any], field: str) -> dict[str, Any]:
    left_value = left.get(field)
    right_value = right.get(field)
    return {
        "causal": left_value,
        "linear": right_value,
        "delta": None if left_value is None or right_value is None else left_value - right_value,
    }


def _fraction(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= 1e-12:
        return None
    return numerator / denominator


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / max(1, len(values))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    audit = summary["audit"]
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "Causal contextual-router support-quality audit.",
        "",
        f"- Config: `{summary['config_path']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Folds: `{audit['fold_count']}`",
        f"- Rationale: {audit['rationale']}",
        "",
        "## Gate Criteria",
    ]
    for row in audit["gate_criteria"]:
        lines.append(
            f"- {row['criterion']}: `{row['passed']}` "
            f"(threshold `{row['threshold']}`, actual `{row['actual']}`)"
        )
    lines.extend(["", "## Aggregate Metrics"])
    for row in sorted(audit["aggregate_metrics"].values(), key=lambda item: item["control"]):
        lines.append(
            "- "
            f"{row['control']}: CE `{row['mean_router_loss']}`, "
            f"oracle regret `{row['mean_oracle_support_regret']}`, "
            f"functional churn `{row['mean_functional_churn_logit_l1']}`"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-folds", type=int, default=None)
    args = parser.parse_args(argv)
    summary = run_causal_contextual_router_support_audit(
        args.config,
        args.out,
        max_folds=args.max_folds,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
