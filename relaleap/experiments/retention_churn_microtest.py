"""Local retention/churn microtest for promoted contextual-router controls."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from relaleap.experiments.run import _load_torch_info
from relaleap.experiments.run import _read_config
from relaleap.experiments.support_audit import _ce_loss
from relaleap.experiments.support_deconfounding import _support_audit_from_support
from relaleap.smoke import ResidualColumns
from relaleap.smoke import TinyCharTransformer
from relaleap.smoke import _build_batch
from relaleap.smoke import _state_dict_delta


DEFAULT_CONFIG = Path(
    "configs/token_larger_support_wide_hep_temporal_clipped_objective_gate.yaml"
)
DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_retention_churn_microtest"
)


@dataclass(frozen=True)
class _VariantSpec:
    name: str
    kind: str
    top_k: int
    num_columns: int
    atoms_per_column: int
    support_router: str
    contextual_router_hidden_dim: int
    dense_rank: int | None = None
    freeze_router_during_transfer: bool = False
    gradient_clip_norm: float | None = None
    value_gradient_clip_norm: float | None = None
    value_gradient_low_rank: int | None = None
    value_update_scale: float = 1.0
    anchor_update_group: str = "full"
    transfer_update_group: str = "full"


def run_retention_churn_microtest(
    config_path: Path,
    out_dir: Path,
    *,
    include_mitigation_variants: bool = False,
    include_decomposition_variants: bool = False,
    include_value_mitigation_variants: bool = False,
    include_low_rank_value_variants: bool = False,
) -> dict[str, Any]:
    """Train on slice A then slice B and measure anchor drift/churn."""

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("retention/churn microtest requires torch") from exc

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
    if residual_objective != "supervised_ce":
        raise ValueError("retention/churn microtest currently requires supervised_ce")
    if top_k != 2:
        raise ValueError("retention/churn microtest expects promoted top_k: 2")
    if support_router != "contextual_mlp":
        raise ValueError("retention/churn microtest expects contextual_mlp baseline")

    torch.manual_seed(seed)
    inputs, targets, vocab_size = _build_batch(
        dataset=dataset,
        seq_len=seq_len,
        batch_size=8,
    )
    anchor_inputs, transfer_inputs = inputs[:4], inputs[4:]
    anchor_targets, transfer_targets = targets[:4], targets[4:]
    base = TinyCharTransformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        layers=layers,
    )
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    anchor_hidden = base.encode(anchor_inputs).detach()
    transfer_hidden = base.encode(transfer_inputs).detach()
    empty_anchor_ce = _ce_loss(base.decode(anchor_hidden), anchor_targets, vocab_size)
    empty_transfer_ce = _ce_loss(base.decode(transfer_hidden), transfer_targets, vocab_size)

    active_rank = top_k * atoms_per_column
    specs = [
        _VariantSpec(
            name="promoted_contextual_topk2",
            kind="sparse",
            top_k=2,
            num_columns=num_columns,
            atoms_per_column=atoms_per_column,
            support_router="contextual_mlp",
            contextual_router_hidden_dim=contextual_router_hidden_dim,
        ),
        _VariantSpec(
            name="rank_matched_contextual_topk1",
            kind="sparse",
            top_k=1,
            num_columns=num_columns * top_k,
            atoms_per_column=atoms_per_column,
            support_router="contextual_mlp",
            contextual_router_hidden_dim=contextual_router_hidden_dim,
        ),
        _VariantSpec(
            name="random_fixed_topk2",
            kind="sparse_fixed",
            top_k=2,
            num_columns=num_columns,
            atoms_per_column=atoms_per_column,
            support_router="contextual_mlp",
            contextual_router_hidden_dim=contextual_router_hidden_dim,
        ),
        _VariantSpec(
            name="norm_matched_dense_active_rank",
            kind="dense",
            top_k=0,
            num_columns=0,
            atoms_per_column=0,
            support_router="none",
            contextual_router_hidden_dim=0,
            dense_rank=active_rank,
        ),
    ]
    if include_mitigation_variants:
        specs.extend(
            [
                _VariantSpec(
                    name="router_frozen_transfer_topk2",
                    kind="sparse",
                    top_k=2,
                    num_columns=num_columns,
                    atoms_per_column=atoms_per_column,
                    support_router="contextual_mlp",
                    contextual_router_hidden_dim=contextual_router_hidden_dim,
                    freeze_router_during_transfer=True,
                ),
                _VariantSpec(
                    name="gradient_clipped_contextual_topk2",
                    kind="sparse",
                    top_k=2,
                    num_columns=num_columns,
                    atoms_per_column=atoms_per_column,
                    support_router="contextual_mlp",
                    contextual_router_hidden_dim=contextual_router_hidden_dim,
                    gradient_clip_norm=0.25,
                ),
            ]
        )
    if include_decomposition_variants:
        specs.extend(
            [
                _VariantSpec(
                    name="router_only_transfer_topk2",
                    kind="sparse",
                    top_k=2,
                    num_columns=num_columns,
                    atoms_per_column=atoms_per_column,
                    support_router="contextual_mlp",
                    contextual_router_hidden_dim=contextual_router_hidden_dim,
                    transfer_update_group="router_only",
                ),
                _VariantSpec(
                    name="value_only_transfer_topk2",
                    kind="sparse",
                    top_k=2,
                    num_columns=num_columns,
                    atoms_per_column=atoms_per_column,
                    support_router="contextual_mlp",
                    contextual_router_hidden_dim=contextual_router_hidden_dim,
                    transfer_update_group="value_only",
                ),
            ]
        )
    if include_value_mitigation_variants:
        specs.extend(
            [
                _VariantSpec(
                    name="value_gradient_clipped_contextual_topk2",
                    kind="sparse",
                    top_k=2,
                    num_columns=num_columns,
                    atoms_per_column=atoms_per_column,
                    support_router="contextual_mlp",
                    contextual_router_hidden_dim=contextual_router_hidden_dim,
                    value_gradient_clip_norm=0.10,
                ),
                _VariantSpec(
                    name="value_update_scaled_contextual_topk2",
                    kind="sparse",
                    top_k=2,
                    num_columns=num_columns,
                    atoms_per_column=atoms_per_column,
                    support_router="contextual_mlp",
                    contextual_router_hidden_dim=contextual_router_hidden_dim,
                    value_update_scale=0.50,
                ),
            ]
        )
    if include_low_rank_value_variants:
        specs.extend(
            [
                _VariantSpec(
                    name="value_gradient_rank1_contextual_topk2",
                    kind="sparse",
                    top_k=2,
                    num_columns=num_columns,
                    atoms_per_column=atoms_per_column,
                    support_router="contextual_mlp",
                    contextual_router_hidden_dim=contextual_router_hidden_dim,
                    value_gradient_low_rank=1,
                ),
                _VariantSpec(
                    name="value_gradient_rank2_contextual_topk2",
                    kind="sparse",
                    top_k=2,
                    num_columns=num_columns,
                    atoms_per_column=atoms_per_column,
                    support_router="contextual_mlp",
                    contextual_router_hidden_dim=contextual_router_hidden_dim,
                    value_gradient_low_rank=2,
                ),
            ]
        )

    variant_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    per_token_commutator_rows: list[dict[str, Any]] = []
    token_classes = _target_token_classes(targets)
    promoted_anchor_residual_norm: float | None = None
    for offset, spec in enumerate(specs):
        torch.manual_seed(seed + 100 * offset)
        if spec.kind == "dense":
            if promoted_anchor_residual_norm is None:
                raise RuntimeError("dense norm-matched control requires sparse baseline first")
            adapter = nn.Sequential(
                nn.Linear(hidden_dim, int(spec.dense_rank or active_rank), bias=False),
                nn.Linear(int(spec.dense_rank or active_rank), hidden_dim, bias=False),
            )
            nn.init.normal_(adapter[0].weight, mean=0.0, std=0.02)
            nn.init.zeros_(adapter[1].weight)
            before = {key: value.detach().clone() for key, value in adapter.state_dict().items()}
            _train_dense(
                base=base,
                adapter=adapter,
                hidden=anchor_hidden,
                targets=anchor_targets,
                vocab_size=vocab_size,
                steps=max_steps,
                learning_rate=learning_rate,
            )
            anchor_raw = adapter(anchor_hidden)
            anchor_raw_norm = _residual_norm(anchor_raw)
            eval_scale = promoted_anchor_residual_norm / max(anchor_raw_norm, 1e-12)
            before_b = {key: value.detach().clone() for key, value in adapter.state_dict().items()}
            anchor_before = _dense_snapshot(
                base=base,
                adapter=adapter,
                hidden=anchor_hidden,
                targets=anchor_targets,
                vocab_size=vocab_size,
                eval_scale=eval_scale,
            )
            transfer_before = _dense_snapshot(
                base=base,
                adapter=adapter,
                hidden=transfer_hidden,
                targets=transfer_targets,
                vocab_size=vocab_size,
                eval_scale=eval_scale,
            )
            _train_dense(
                base=base,
                adapter=adapter,
                hidden=transfer_hidden,
                targets=transfer_targets,
                vocab_size=vocab_size,
                steps=max_steps,
                learning_rate=learning_rate,
            )
            anchor_after = _dense_snapshot(
                base=base,
                adapter=adapter,
                hidden=anchor_hidden,
                targets=anchor_targets,
                vocab_size=vocab_size,
                eval_scale=eval_scale,
            )
            transfer_after = _dense_snapshot(
                base=base,
                adapter=adapter,
                hidden=transfer_hidden,
                targets=transfer_targets,
                vocab_size=vocab_size,
                eval_scale=eval_scale,
            )
            stored_parameters = sum(p.numel() for p in adapter.parameters())
            parameter_delta_after_anchor = _state_dict_delta(before, adapter)
            parameter_delta_during_transfer = _state_dict_delta(before_b, adapter)
            support_columns = ""
            torch.manual_seed(seed + 100 * offset)
            reverse_adapter = nn.Sequential(
                nn.Linear(hidden_dim, int(spec.dense_rank or active_rank), bias=False),
                nn.Linear(int(spec.dense_rank or active_rank), hidden_dim, bias=False),
            )
            nn.init.normal_(reverse_adapter[0].weight, mean=0.0, std=0.02)
            nn.init.zeros_(reverse_adapter[1].weight)
            _train_dense(
                base=base,
                adapter=reverse_adapter,
                hidden=transfer_hidden,
                targets=transfer_targets,
                vocab_size=vocab_size,
                steps=max_steps,
                learning_rate=learning_rate,
            )
            _train_dense(
                base=base,
                adapter=reverse_adapter,
                hidden=anchor_hidden,
                targets=anchor_targets,
                vocab_size=vocab_size,
                steps=max_steps,
                learning_rate=learning_rate,
            )
            reverse_anchor_final = _dense_snapshot(
                base=base,
                adapter=reverse_adapter,
                hidden=anchor_hidden,
                targets=anchor_targets,
                vocab_size=vocab_size,
                eval_scale=eval_scale,
            )
            reverse_transfer_final = _dense_snapshot(
                base=base,
                adapter=reverse_adapter,
                hidden=transfer_hidden,
                targets=transfer_targets,
                vocab_size=vocab_size,
                eval_scale=eval_scale,
            )
            torch.manual_seed(seed + 10_000 + 100 * offset)
            same_order_adapter = nn.Sequential(
                nn.Linear(hidden_dim, int(spec.dense_rank or active_rank), bias=False),
                nn.Linear(int(spec.dense_rank or active_rank), hidden_dim, bias=False),
            )
            nn.init.normal_(same_order_adapter[0].weight, mean=0.0, std=0.02)
            nn.init.zeros_(same_order_adapter[1].weight)
            _train_dense(
                base=base,
                adapter=same_order_adapter,
                hidden=anchor_hidden,
                targets=anchor_targets,
                vocab_size=vocab_size,
                steps=max_steps,
                learning_rate=learning_rate,
            )
            _train_dense(
                base=base,
                adapter=same_order_adapter,
                hidden=transfer_hidden,
                targets=transfer_targets,
                vocab_size=vocab_size,
                steps=max_steps,
                learning_rate=learning_rate,
            )
            same_order_anchor_final = _dense_snapshot(
                base=base,
                adapter=same_order_adapter,
                hidden=anchor_hidden,
                targets=anchor_targets,
                vocab_size=vocab_size,
                eval_scale=eval_scale,
            )
            same_order_transfer_final = _dense_snapshot(
                base=base,
                adapter=same_order_adapter,
                hidden=transfer_hidden,
                targets=transfer_targets,
                vocab_size=vocab_size,
                eval_scale=eval_scale,
            )
            torch.manual_seed(seed + 100 * offset)
            identical_order_adapter = nn.Sequential(
                nn.Linear(hidden_dim, int(spec.dense_rank or active_rank), bias=False),
                nn.Linear(int(spec.dense_rank or active_rank), hidden_dim, bias=False),
            )
            nn.init.normal_(identical_order_adapter[0].weight, mean=0.0, std=0.02)
            nn.init.zeros_(identical_order_adapter[1].weight)
            _train_dense(
                base=base,
                adapter=identical_order_adapter,
                hidden=anchor_hidden,
                targets=anchor_targets,
                vocab_size=vocab_size,
                steps=max_steps,
                learning_rate=learning_rate,
            )
            _train_dense(
                base=base,
                adapter=identical_order_adapter,
                hidden=transfer_hidden,
                targets=transfer_targets,
                vocab_size=vocab_size,
                steps=max_steps,
                learning_rate=learning_rate,
            )
            identical_order_anchor_final = _dense_snapshot(
                base=base,
                adapter=identical_order_adapter,
                hidden=anchor_hidden,
                targets=anchor_targets,
                vocab_size=vocab_size,
                eval_scale=eval_scale,
            )
            identical_order_transfer_final = _dense_snapshot(
                base=base,
                adapter=identical_order_adapter,
                hidden=transfer_hidden,
                targets=transfer_targets,
                vocab_size=vocab_size,
                eval_scale=eval_scale,
            )
        else:
            residual = ResidualColumns(
                hidden_dim=hidden_dim,
                num_columns=spec.num_columns,
                atoms_per_column=spec.atoms_per_column,
                top_k=spec.top_k,
                support_router=spec.support_router,
                contextual_router_hidden_dim=spec.contextual_router_hidden_dim,
            )
            fixed_anchor_support = None
            fixed_transfer_support = None
            if spec.kind == "sparse_fixed":
                fixed_anchor_support = _random_fixed_support(
                    shape=anchor_inputs.shape,
                    num_columns=spec.num_columns,
                    top_k=spec.top_k,
                    seed=seed + 10_000 + offset,
                )
                fixed_transfer_support = _random_fixed_support(
                    shape=transfer_inputs.shape,
                    num_columns=spec.num_columns,
                    top_k=spec.top_k,
                    seed=seed + 20_000 + offset,
                )
            before = {key: value.detach().clone() for key, value in residual.state_dict().items()}
            _train_sparse(
                base=base,
                residual=residual,
                inputs=anchor_inputs,
                targets=anchor_targets,
                vocab_size=vocab_size,
                steps=max_steps,
                learning_rate=learning_rate,
                support_indices=fixed_anchor_support,
                gradient_clip_norm=spec.gradient_clip_norm,
                value_gradient_clip_norm=spec.value_gradient_clip_norm,
                value_gradient_low_rank=spec.value_gradient_low_rank,
                value_update_scale=spec.value_update_scale,
                update_group=spec.anchor_update_group,
            )
            before_b = {key: value.detach().clone() for key, value in residual.state_dict().items()}
            anchor_before = _sparse_snapshot(
                base=base,
                residual=residual,
                inputs=anchor_inputs,
                targets=anchor_targets,
                vocab_size=vocab_size,
                support_indices=fixed_anchor_support,
            )
            if spec.name == "promoted_contextual_topk2":
                promoted_anchor_residual_norm = float(anchor_before["residual_norm_mean"])
            transfer_before = _sparse_snapshot(
                base=base,
                residual=residual,
                inputs=transfer_inputs,
                targets=transfer_targets,
                vocab_size=vocab_size,
                support_indices=fixed_transfer_support,
            )
            torch.manual_seed(seed + 10_000 + 100 * offset)
            same_order_residual = ResidualColumns(
                hidden_dim=hidden_dim,
                num_columns=spec.num_columns,
                atoms_per_column=spec.atoms_per_column,
                top_k=spec.top_k,
                support_router=spec.support_router,
                contextual_router_hidden_dim=spec.contextual_router_hidden_dim,
            )
            _train_sparse(
                base=base,
                residual=same_order_residual,
                inputs=anchor_inputs,
                targets=anchor_targets,
                vocab_size=vocab_size,
                steps=max_steps,
                learning_rate=learning_rate,
                support_indices=fixed_anchor_support,
                gradient_clip_norm=spec.gradient_clip_norm,
                value_gradient_clip_norm=spec.value_gradient_clip_norm,
                value_gradient_low_rank=spec.value_gradient_low_rank,
                value_update_scale=spec.value_update_scale,
                update_group=spec.anchor_update_group,
            )
            _train_sparse(
                base=base,
                residual=same_order_residual,
                inputs=transfer_inputs,
                targets=transfer_targets,
                vocab_size=vocab_size,
                steps=max_steps,
                learning_rate=learning_rate,
                support_indices=fixed_transfer_support,
                freeze_router=spec.freeze_router_during_transfer,
                gradient_clip_norm=spec.gradient_clip_norm,
                value_gradient_clip_norm=spec.value_gradient_clip_norm,
                value_gradient_low_rank=spec.value_gradient_low_rank,
                value_update_scale=spec.value_update_scale,
                update_group=spec.transfer_update_group,
            )
            same_order_anchor_final = _sparse_snapshot(
                base=base,
                residual=same_order_residual,
                inputs=anchor_inputs,
                targets=anchor_targets,
                vocab_size=vocab_size,
                support_indices=fixed_anchor_support,
            )
            same_order_transfer_final = _sparse_snapshot(
                base=base,
                residual=same_order_residual,
                inputs=transfer_inputs,
                targets=transfer_targets,
                vocab_size=vocab_size,
                support_indices=fixed_transfer_support,
            )
            _train_sparse(
                base=base,
                residual=residual,
                inputs=transfer_inputs,
                targets=transfer_targets,
                vocab_size=vocab_size,
                steps=max_steps,
                learning_rate=learning_rate,
                support_indices=fixed_transfer_support,
                freeze_router=spec.freeze_router_during_transfer,
                gradient_clip_norm=spec.gradient_clip_norm,
                value_gradient_clip_norm=spec.value_gradient_clip_norm,
                value_gradient_low_rank=spec.value_gradient_low_rank,
                value_update_scale=spec.value_update_scale,
                update_group=spec.transfer_update_group,
            )
            anchor_after = _sparse_snapshot(
                base=base,
                residual=residual,
                inputs=anchor_inputs,
                targets=anchor_targets,
                vocab_size=vocab_size,
                support_indices=fixed_anchor_support,
            )
            transfer_after = _sparse_snapshot(
                base=base,
                residual=residual,
                inputs=transfer_inputs,
                targets=transfer_targets,
                vocab_size=vocab_size,
                support_indices=fixed_transfer_support,
            )
            stored_parameters = sum(p.numel() for p in residual.parameters())
            parameter_delta_after_anchor = _state_dict_delta(before, residual)
            parameter_delta_during_transfer = _state_dict_delta(before_b, residual)
            support_columns = spec.num_columns
            torch.manual_seed(seed + 100 * offset)
            reverse_residual = ResidualColumns(
                hidden_dim=hidden_dim,
                num_columns=spec.num_columns,
                atoms_per_column=spec.atoms_per_column,
                top_k=spec.top_k,
                support_router=spec.support_router,
                contextual_router_hidden_dim=spec.contextual_router_hidden_dim,
            )
            _train_sparse(
                base=base,
                residual=reverse_residual,
                inputs=transfer_inputs,
                targets=transfer_targets,
                vocab_size=vocab_size,
                steps=max_steps,
                learning_rate=learning_rate,
                support_indices=fixed_transfer_support,
                gradient_clip_norm=spec.gradient_clip_norm,
                value_gradient_clip_norm=spec.value_gradient_clip_norm,
                value_gradient_low_rank=spec.value_gradient_low_rank,
                value_update_scale=spec.value_update_scale,
                update_group=spec.anchor_update_group,
            )
            _train_sparse(
                base=base,
                residual=reverse_residual,
                inputs=anchor_inputs,
                targets=anchor_targets,
                vocab_size=vocab_size,
                steps=max_steps,
                learning_rate=learning_rate,
                support_indices=fixed_anchor_support,
                freeze_router=spec.freeze_router_during_transfer,
                gradient_clip_norm=spec.gradient_clip_norm,
                value_gradient_clip_norm=spec.value_gradient_clip_norm,
                value_gradient_low_rank=spec.value_gradient_low_rank,
                value_update_scale=spec.value_update_scale,
                update_group=spec.transfer_update_group,
            )
            reverse_anchor_final = _sparse_snapshot(
                base=base,
                residual=reverse_residual,
                inputs=anchor_inputs,
                targets=anchor_targets,
                vocab_size=vocab_size,
                support_indices=fixed_anchor_support,
            )
            reverse_transfer_final = _sparse_snapshot(
                base=base,
                residual=reverse_residual,
                inputs=transfer_inputs,
                targets=transfer_targets,
                vocab_size=vocab_size,
                support_indices=fixed_transfer_support,
            )
            torch.manual_seed(seed + 100 * offset)
            identical_order_residual = ResidualColumns(
                hidden_dim=hidden_dim,
                num_columns=spec.num_columns,
                atoms_per_column=spec.atoms_per_column,
                top_k=spec.top_k,
                support_router=spec.support_router,
                contextual_router_hidden_dim=spec.contextual_router_hidden_dim,
            )
            _train_sparse(
                base=base,
                residual=identical_order_residual,
                inputs=anchor_inputs,
                targets=anchor_targets,
                vocab_size=vocab_size,
                steps=max_steps,
                learning_rate=learning_rate,
                support_indices=fixed_anchor_support,
                gradient_clip_norm=spec.gradient_clip_norm,
                value_gradient_clip_norm=spec.value_gradient_clip_norm,
                value_gradient_low_rank=spec.value_gradient_low_rank,
                value_update_scale=spec.value_update_scale,
                update_group=spec.anchor_update_group,
            )
            _train_sparse(
                base=base,
                residual=identical_order_residual,
                inputs=transfer_inputs,
                targets=transfer_targets,
                vocab_size=vocab_size,
                steps=max_steps,
                learning_rate=learning_rate,
                support_indices=fixed_transfer_support,
                freeze_router=spec.freeze_router_during_transfer,
                gradient_clip_norm=spec.gradient_clip_norm,
                value_gradient_clip_norm=spec.value_gradient_clip_norm,
                value_gradient_low_rank=spec.value_gradient_low_rank,
                value_update_scale=spec.value_update_scale,
                update_group=spec.transfer_update_group,
            )
            identical_order_anchor_final = _sparse_snapshot(
                base=base,
                residual=identical_order_residual,
                inputs=anchor_inputs,
                targets=anchor_targets,
                vocab_size=vocab_size,
                support_indices=fixed_anchor_support,
            )
            identical_order_transfer_final = _sparse_snapshot(
                base=base,
                residual=identical_order_residual,
                inputs=transfer_inputs,
                targets=transfer_targets,
                vocab_size=vocab_size,
                support_indices=fixed_transfer_support,
            )

        phase_rows.extend(
            _phase_rows(
                variant=spec.name,
                anchor_before=anchor_before,
                anchor_after=anchor_after,
                transfer_before=transfer_before,
                transfer_after=transfer_after,
            )
        )
        order_average = _order_average_metrics(
            anchor_after=anchor_after,
            transfer_after=transfer_after,
            reverse_anchor_final=reverse_anchor_final,
            reverse_transfer_final=reverse_transfer_final,
            anchor_targets=anchor_targets,
            transfer_targets=transfer_targets,
            vocab_size=vocab_size,
        )
        same_order_ensemble = _same_order_ensemble_metrics(
            anchor_after=anchor_after,
            transfer_after=transfer_after,
            same_order_anchor_final=same_order_anchor_final,
            same_order_transfer_final=same_order_transfer_final,
            anchor_targets=anchor_targets,
            transfer_targets=transfer_targets,
            vocab_size=vocab_size,
        )
        same_order_sanity = _same_order_identical_replay_metrics(
            anchor_after=anchor_after,
            transfer_after=transfer_after,
            identical_order_anchor_final=identical_order_anchor_final,
            identical_order_transfer_final=identical_order_transfer_final,
        )
        per_token_commutator_rows.extend(
            _per_token_commutator_rows(
                variant=spec.name,
                split="anchor",
                forward=anchor_after,
                reverse=reverse_anchor_final,
                targets=anchor_targets,
                vocab_size=vocab_size,
                token_classes=token_classes,
            )
        )
        per_token_commutator_rows.extend(
            _per_token_commutator_rows(
                variant=spec.name,
                split="transfer",
                forward=transfer_after,
                reverse=reverse_transfer_final,
                targets=transfer_targets,
                vocab_size=vocab_size,
                token_classes=token_classes,
            )
        )
        row = {
            "variant": spec.name,
            "kind": spec.kind,
            "top_k": spec.top_k,
            "num_columns": support_columns,
            "stored_parameters": stored_parameters,
            "active_parameters_proxy": (
                2 * hidden_dim * int(spec.dense_rank or active_rank)
                if spec.kind == "dense"
                else spec.top_k * hidden_dim * spec.atoms_per_column
            ),
            "anchor_ce_before_transfer": anchor_before["ce_loss"],
            "anchor_ce_after_transfer": anchor_after["ce_loss"],
            "anchor_ce_drift": anchor_after["ce_loss"] - anchor_before["ce_loss"],
            "anchor_logit_mse_drift": _mse_delta(anchor_after["logits"], anchor_before["logits"]),
            "anchor_mean_abs_logit_drift": _mean_abs_delta(
                anchor_after["logits"],
                anchor_before["logits"],
            ),
            "anchor_residual_stream_l2_drift": _stream_l2_delta(
                anchor_after["residual_delta"],
                anchor_before["residual_delta"],
            ),
            "anchor_residual_norm_before_transfer": anchor_before["residual_norm_mean"],
            "anchor_residual_norm_after_transfer": anchor_after["residual_norm_mean"],
            "anchor_support_churn_after_transfer": _support_churn(
                anchor_before.get("support"),
                anchor_after.get("support"),
            ),
            "anchor_used_columns_before_transfer": anchor_before.get("used_columns", ""),
            "anchor_used_columns_after_transfer": anchor_after.get("used_columns", ""),
            "anchor_unique_support_sets_before_transfer": anchor_before.get(
                "unique_support_sets",
                "",
            ),
            "anchor_unique_support_sets_after_transfer": anchor_after.get(
                "unique_support_sets",
                "",
            ),
            "transfer_ce_before_transfer": transfer_before["ce_loss"],
            "transfer_ce_after_transfer": transfer_after["ce_loss"],
            "transfer_ce_improvement": (
                transfer_before["ce_loss"] - transfer_after["ce_loss"]
            ),
            "commutator_anchor_ce_abs_delta": abs(
                anchor_after["ce_loss"] - reverse_anchor_final["ce_loss"]
            ),
            "commutator_transfer_ce_abs_delta": abs(
                transfer_after["ce_loss"] - reverse_transfer_final["ce_loss"]
            ),
            "commutator_anchor_logit_mse": _mse_delta(
                anchor_after["logits"],
                reverse_anchor_final["logits"],
            ),
            "commutator_transfer_logit_mse": _mse_delta(
                transfer_after["logits"],
                reverse_transfer_final["logits"],
            ),
            "commutator_anchor_residual_stream_l2": _stream_l2_delta(
                anchor_after["residual_delta"],
                reverse_anchor_final["residual_delta"],
            ),
            "commutator_transfer_residual_stream_l2": _stream_l2_delta(
                transfer_after["residual_delta"],
                reverse_transfer_final["residual_delta"],
            ),
            "commutator_anchor_support_churn": _support_churn(
                anchor_after.get("support"),
                reverse_anchor_final.get("support"),
            ),
            "commutator_transfer_support_churn": _support_churn(
                transfer_after.get("support"),
                reverse_transfer_final.get("support"),
            ),
            **order_average,
            **same_order_ensemble,
            **same_order_sanity,
            "same_order_primary_seed": seed + 100 * offset,
            "same_order_identical_replay_seed": seed + 100 * offset,
            "same_order_independent_seed": seed + 10_000 + 100 * offset,
            "parameter_delta_after_anchor": parameter_delta_after_anchor,
            "parameter_delta_during_transfer": parameter_delta_during_transfer,
            "freeze_router_during_transfer": spec.freeze_router_during_transfer,
            "gradient_clip_norm": (
                "" if spec.gradient_clip_norm is None else spec.gradient_clip_norm
            ),
            "value_gradient_clip_norm": (
                ""
                if spec.value_gradient_clip_norm is None
                else spec.value_gradient_clip_norm
            ),
            "value_gradient_low_rank": (
                "" if spec.value_gradient_low_rank is None else spec.value_gradient_low_rank
            ),
            "value_update_scale": spec.value_update_scale,
            "anchor_update_group": spec.anchor_update_group,
            "transfer_update_group": spec.transfer_update_group,
        }
        variant_rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "variant_metrics.csv", variant_rows)
    _write_csv(out_dir / "phase_metrics.csv", phase_rows)
    _write_csv(out_dir / "per_token_commutator.csv", per_token_commutator_rows)
    summary = {
        "status": "ok",
        "experiment_id": f"{experiment_id}_retention_churn_microtest",
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        **_load_torch_info(),
        "audit": {
            "dataset": dataset,
            "seq_len": seq_len,
            "anchor_batch_size": int(anchor_inputs.shape[0]),
            "transfer_batch_size": int(transfer_inputs.shape[0]),
            "vocab_size": vocab_size,
            "training_steps_per_slice": max_steps,
            "residual_objective": residual_objective,
            "empty_anchor_ce": empty_anchor_ce,
            "empty_transfer_ce": empty_transfer_ce,
            "variants": variant_rows,
            "primary_outputs": [
                "anchor_ce_drift",
                "anchor_logit_mse_drift",
                "anchor_residual_stream_l2_drift",
                "anchor_support_churn_after_transfer",
                "transfer_ce_improvement",
                "commutator_anchor_logit_mse",
                "commutator_transfer_logit_mse",
                "commutator_anchor_residual_stream_l2",
                "commutator_transfer_residual_stream_l2",
                "order_averaged_anchor_ce_loss",
                "order_averaged_transfer_ce_loss",
                "order_averaged_anchor_logit_mse_to_forward",
                "order_averaged_transfer_logit_mse_to_forward",
                "same_order_ensemble_anchor_ce_loss",
                "same_order_ensemble_transfer_ce_loss",
                "same_order_ensemble_anchor_ce_delta_vs_best_endpoint",
                "same_order_ensemble_transfer_ce_delta_vs_best_endpoint",
                "same_order_identical_replay_nonperturbation_pass",
                "same_order_identical_anchor_logit_mse_to_primary",
                "same_order_identical_transfer_logit_mse_to_primary",
                "per_token_commutator_ce_delta",
                "per_token_commutator_symmetric_kl",
                "per_token_commutator_logit_mse",
                "per_token_commutator_residual_delta_l2",
            ],
            "include_mitigation_variants": include_mitigation_variants,
            "include_decomposition_variants": include_decomposition_variants,
            "include_value_mitigation_variants": include_value_mitigation_variants,
            "include_low_rank_value_variants": include_low_rank_value_variants,
        },
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "variant_metrics_csv": str(out_dir / "variant_metrics.csv"),
            "phase_metrics_csv": str(out_dir / "phase_metrics.csv"),
            "per_token_commutator_csv": str(out_dir / "per_token_commutator.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _train_sparse(
    *,
    base: Any,
    residual: Any,
    inputs: Any,
    targets: Any,
    vocab_size: int,
    steps: int,
    learning_rate: float,
    support_indices: Any | None = None,
    freeze_router: bool = False,
    gradient_clip_norm: float | None = None,
    value_gradient_clip_norm: float | None = None,
    value_gradient_low_rank: int | None = None,
    value_update_scale: float = 1.0,
    update_group: str = "full",
) -> None:
    import torch
    import torch.nn.functional as F

    residual.train()
    trainable_parameters = []
    value_parameters = []
    for name, parameter in residual.named_parameters():
        is_router = _is_router_parameter(name)
        if not is_router:
            value_parameters.append(parameter)
        if freeze_router and is_router:
            continue
        if update_group == "full":
            trainable_parameters.append(parameter)
        elif update_group == "router_only" and is_router:
            trainable_parameters.append(parameter)
        elif update_group == "value_only" and not is_router:
            trainable_parameters.append(parameter)
        elif update_group not in {"full", "router_only", "value_only"}:
            raise ValueError("update_group must be one of: full, router_only, value_only")
    if not trainable_parameters:
        raise ValueError(f"update_group selected no trainable parameters: {update_group}")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if support_indices is None:
            logits = base(inputs, residual_adapter=residual)
        else:
            hidden = base.encode(inputs)
            logits = base.decode(residual(hidden, support_indices=support_indices))
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, vocab_size),
            targets[:, :-1].reshape(-1),
        )
        loss.backward()
        if value_gradient_low_rank is not None:
            _project_value_gradients_low_rank(value_parameters, value_gradient_low_rank)
        if value_gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(value_parameters, value_gradient_clip_norm)
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, gradient_clip_norm)
        if value_update_scale != 1.0:
            for parameter in value_parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(value_update_scale)
        optimizer.step()
    residual.eval()


def _project_value_gradients_low_rank(value_parameters: list[Any], rank: int) -> None:
    if rank < 1:
        raise ValueError("value_gradient_low_rank must be positive when set")
    import torch

    for parameter in value_parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad
        if grad.ndim < 2:
            continue
        original_shape = grad.shape
        if grad.ndim == 3:
            matrices = grad.reshape(-1, original_shape[-2], original_shape[-1])
            projected = []
            for matrix in matrices:
                max_rank = min(rank, int(matrix.shape[0]), int(matrix.shape[1]))
                if max_rank >= min(int(matrix.shape[0]), int(matrix.shape[1])):
                    projected.append(matrix)
                    continue
                u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
                projected.append((u[:, :max_rank] * s[:max_rank]) @ vh[:max_rank, :])
            grad.copy_(torch.stack(projected, dim=0).reshape(original_shape))
        elif grad.ndim == 2:
            max_rank = min(rank, int(grad.shape[0]), int(grad.shape[1]))
            if max_rank < min(int(grad.shape[0]), int(grad.shape[1])):
                u, s, vh = torch.linalg.svd(grad, full_matrices=False)
                grad.copy_((u[:, :max_rank] * s[:max_rank]) @ vh[:max_rank, :])


def _is_router_parameter(name: str) -> bool:
    return name.startswith("column_scores.") or name.startswith(
        "contextual_column_scores."
    )


def _train_dense(
    *,
    base: Any,
    adapter: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    steps: int,
    learning_rate: float,
) -> None:
    import torch
    import torch.nn.functional as F

    adapter.train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = base.decode(hidden + adapter(hidden))
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, vocab_size),
            targets[:, :-1].reshape(-1),
        )
        loss.backward()
        optimizer.step()
    adapter.eval()


def _sparse_snapshot(
    *,
    base: Any,
    residual: Any,
    inputs: Any,
    targets: Any,
    vocab_size: int,
    support_indices: Any | None = None,
) -> dict[str, Any]:
    with __import__("torch").no_grad():
        hidden = base.encode(inputs)
        output_hidden, support = residual(
            hidden,
            support_indices=support_indices,
            return_support=True,
        )
        logits = base.decode(output_hidden)
        residual_delta = output_hidden - hidden
        support_audit = _support_audit_from_support(support, residual.num_columns)
    return {
        "ce_loss": _ce_loss(logits, targets, vocab_size),
        "logits": logits.detach().clone(),
        "residual_delta": residual_delta.detach().clone(),
        "residual_norm_mean": _residual_norm(residual_delta),
        "support": support.detach().clone(),
        "used_columns": support_audit["used_columns"],
        "unique_support_sets": support_audit["unique_support_sets"],
    }


def _random_fixed_support(
    *,
    shape: Any,
    num_columns: int,
    top_k: int,
    seed: int,
) -> Any:
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    scores = torch.rand(
        int(shape[0]),
        int(shape[1]),
        num_columns,
        generator=generator,
    )
    return scores.topk(top_k, dim=-1).indices


def _dense_snapshot(
    *,
    base: Any,
    adapter: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    eval_scale: float,
) -> dict[str, Any]:
    with __import__("torch").no_grad():
        residual_delta = adapter(hidden) * eval_scale
        logits = base.decode(hidden + residual_delta)
    return {
        "ce_loss": _ce_loss(logits, targets, vocab_size),
        "logits": logits.detach().clone(),
        "residual_delta": residual_delta.detach().clone(),
        "residual_norm_mean": _residual_norm(residual_delta),
    }


def _phase_rows(
    *,
    variant: str,
    anchor_before: dict[str, Any],
    anchor_after: dict[str, Any],
    transfer_before: dict[str, Any],
    transfer_after: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for split, phase, snapshot in (
        ("anchor", "after_anchor_training", anchor_before),
        ("anchor", "after_transfer_training", anchor_after),
        ("transfer", "before_transfer_training", transfer_before),
        ("transfer", "after_transfer_training", transfer_after),
    ):
        rows.append(
            {
                "variant": variant,
                "split": split,
                "phase": phase,
                "ce_loss": snapshot["ce_loss"],
                "residual_norm_mean": snapshot["residual_norm_mean"],
                "used_columns": snapshot.get("used_columns", ""),
                "unique_support_sets": snapshot.get("unique_support_sets", ""),
            }
        )
    return rows


def _order_average_metrics(
    *,
    anchor_after: dict[str, Any],
    transfer_after: dict[str, Any],
    reverse_anchor_final: dict[str, Any],
    reverse_transfer_final: dict[str, Any],
    anchor_targets: Any,
    transfer_targets: Any,
    vocab_size: int,
) -> dict[str, float]:
    anchor_logits = 0.5 * (
        anchor_after["logits"] + reverse_anchor_final["logits"]
    )
    transfer_logits = 0.5 * (
        transfer_after["logits"] + reverse_transfer_final["logits"]
    )
    anchor_residual = 0.5 * (
        anchor_after["residual_delta"] + reverse_anchor_final["residual_delta"]
    )
    transfer_residual = 0.5 * (
        transfer_after["residual_delta"] + reverse_transfer_final["residual_delta"]
    )
    anchor_forward_ce = float(anchor_after["ce_loss"])
    anchor_reverse_ce = float(reverse_anchor_final["ce_loss"])
    transfer_forward_ce = float(transfer_after["ce_loss"])
    transfer_reverse_ce = float(reverse_transfer_final["ce_loss"])
    anchor_average_ce = _ce_loss(anchor_logits, anchor_targets, vocab_size)
    transfer_average_ce = _ce_loss(transfer_logits, transfer_targets, vocab_size)
    return {
        "order_averaged_anchor_ce_loss": anchor_average_ce,
        "order_averaged_transfer_ce_loss": transfer_average_ce,
        "order_averaged_anchor_ce_delta_vs_forward": (
            anchor_average_ce - anchor_forward_ce
        ),
        "order_averaged_transfer_ce_delta_vs_forward": (
            transfer_average_ce - transfer_forward_ce
        ),
        "order_averaged_anchor_ce_delta_vs_best_order": (
            anchor_average_ce - min(anchor_forward_ce, anchor_reverse_ce)
        ),
        "order_averaged_transfer_ce_delta_vs_best_order": (
            transfer_average_ce - min(transfer_forward_ce, transfer_reverse_ce)
        ),
        "order_averaged_anchor_logit_mse_to_forward": _mse_delta(
            anchor_logits, anchor_after["logits"]
        ),
        "order_averaged_anchor_logit_mse_to_reverse": _mse_delta(
            anchor_logits, reverse_anchor_final["logits"]
        ),
        "order_averaged_transfer_logit_mse_to_forward": _mse_delta(
            transfer_logits, transfer_after["logits"]
        ),
        "order_averaged_transfer_logit_mse_to_reverse": _mse_delta(
            transfer_logits, reverse_transfer_final["logits"]
        ),
        "order_averaged_anchor_residual_stream_l2_to_forward": _stream_l2_delta(
            anchor_residual, anchor_after["residual_delta"]
        ),
        "order_averaged_anchor_residual_stream_l2_to_reverse": _stream_l2_delta(
            anchor_residual, reverse_anchor_final["residual_delta"]
        ),
        "order_averaged_transfer_residual_stream_l2_to_forward": _stream_l2_delta(
            transfer_residual, transfer_after["residual_delta"]
        ),
        "order_averaged_transfer_residual_stream_l2_to_reverse": _stream_l2_delta(
            transfer_residual, reverse_transfer_final["residual_delta"]
        ),
    }


def _same_order_ensemble_metrics(
    *,
    anchor_after: dict[str, Any],
    transfer_after: dict[str, Any],
    same_order_anchor_final: dict[str, Any],
    same_order_transfer_final: dict[str, Any],
    anchor_targets: Any,
    transfer_targets: Any,
    vocab_size: int,
) -> dict[str, float]:
    anchor_logits = 0.5 * (
        anchor_after["logits"] + same_order_anchor_final["logits"]
    )
    transfer_logits = 0.5 * (
        transfer_after["logits"] + same_order_transfer_final["logits"]
    )
    anchor_primary_ce = float(anchor_after["ce_loss"])
    anchor_repeat_ce = float(same_order_anchor_final["ce_loss"])
    transfer_primary_ce = float(transfer_after["ce_loss"])
    transfer_repeat_ce = float(same_order_transfer_final["ce_loss"])
    anchor_ensemble_ce = _ce_loss(anchor_logits, anchor_targets, vocab_size)
    transfer_ensemble_ce = _ce_loss(transfer_logits, transfer_targets, vocab_size)
    return {
        "same_order_ensemble_anchor_ce_loss": anchor_ensemble_ce,
        "same_order_ensemble_transfer_ce_loss": transfer_ensemble_ce,
        "same_order_ensemble_anchor_ce_delta_vs_primary": (
            anchor_ensemble_ce - anchor_primary_ce
        ),
        "same_order_ensemble_transfer_ce_delta_vs_primary": (
            transfer_ensemble_ce - transfer_primary_ce
        ),
        "same_order_ensemble_anchor_ce_delta_vs_best_endpoint": (
            anchor_ensemble_ce - min(anchor_primary_ce, anchor_repeat_ce)
        ),
        "same_order_ensemble_transfer_ce_delta_vs_best_endpoint": (
            transfer_ensemble_ce - min(transfer_primary_ce, transfer_repeat_ce)
        ),
        "same_order_ensemble_anchor_logit_mse_to_primary": _mse_delta(
            anchor_logits, anchor_after["logits"]
        ),
        "same_order_ensemble_transfer_logit_mse_to_primary": _mse_delta(
            transfer_logits, transfer_after["logits"]
        ),
    }


def _same_order_identical_replay_metrics(
    *,
    anchor_after: dict[str, Any],
    transfer_after: dict[str, Any],
    identical_order_anchor_final: dict[str, Any],
    identical_order_transfer_final: dict[str, Any],
) -> dict[str, float | bool | str]:
    anchor_ce_abs_delta = abs(
        float(identical_order_anchor_final["ce_loss"]) - float(anchor_after["ce_loss"])
    )
    transfer_ce_abs_delta = abs(
        float(identical_order_transfer_final["ce_loss"])
        - float(transfer_after["ce_loss"])
    )
    anchor_logit_mse = _mse_delta(
        identical_order_anchor_final["logits"], anchor_after["logits"]
    )
    transfer_logit_mse = _mse_delta(
        identical_order_transfer_final["logits"], transfer_after["logits"]
    )
    anchor_residual_l2 = _stream_l2_delta(
        identical_order_anchor_final["residual_delta"],
        anchor_after["residual_delta"],
    )
    transfer_residual_l2 = _stream_l2_delta(
        identical_order_transfer_final["residual_delta"],
        transfer_after["residual_delta"],
    )
    anchor_support_churn = _support_churn(
        identical_order_anchor_final.get("support"),
        anchor_after.get("support"),
    )
    transfer_support_churn = _support_churn(
        identical_order_transfer_final.get("support"),
        transfer_after.get("support"),
    )
    numeric_checks = [
        anchor_ce_abs_delta,
        transfer_ce_abs_delta,
        anchor_logit_mse,
        transfer_logit_mse,
        anchor_residual_l2,
        transfer_residual_l2,
    ]
    support_checks = [
        churn
        for churn in (anchor_support_churn, transfer_support_churn)
        if churn != ""
    ]
    return {
        "same_order_identical_anchor_ce_abs_delta_to_primary": anchor_ce_abs_delta,
        "same_order_identical_transfer_ce_abs_delta_to_primary": transfer_ce_abs_delta,
        "same_order_identical_anchor_logit_mse_to_primary": anchor_logit_mse,
        "same_order_identical_transfer_logit_mse_to_primary": transfer_logit_mse,
        "same_order_identical_anchor_residual_stream_l2_to_primary": (
            anchor_residual_l2
        ),
        "same_order_identical_transfer_residual_stream_l2_to_primary": (
            transfer_residual_l2
        ),
        "same_order_identical_anchor_support_churn_to_primary": anchor_support_churn,
        "same_order_identical_transfer_support_churn_to_primary": transfer_support_churn,
        "same_order_identical_replay_nonperturbation_pass": all(
            value <= 1e-12 for value in numeric_checks
        )
        and all(float(value) <= 1e-12 for value in support_checks),
    }


def _per_token_commutator_rows(
    *,
    variant: str,
    split: str,
    forward: dict[str, Any],
    reverse: dict[str, Any],
    targets: Any,
    vocab_size: int,
    token_classes: dict[int, str],
) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    forward_logits = forward["logits"][:, :-1, :]
    reverse_logits = reverse["logits"][:, :-1, :]
    target_tokens = targets[:, :-1]
    forward_ce = F.cross_entropy(
        forward_logits.reshape(-1, vocab_size),
        target_tokens.reshape(-1),
        reduction="none",
    ).reshape(target_tokens.shape)
    reverse_ce = F.cross_entropy(
        reverse_logits.reshape(-1, vocab_size),
        target_tokens.reshape(-1),
        reduction="none",
    ).reshape(target_tokens.shape)
    forward_log_probs = F.log_softmax(forward_logits, dim=-1)
    reverse_log_probs = F.log_softmax(reverse_logits, dim=-1)
    forward_probs = forward_log_probs.exp()
    reverse_probs = reverse_log_probs.exp()
    kl_forward_reverse = (
        forward_probs * (forward_log_probs - reverse_log_probs)
    ).sum(dim=-1)
    kl_reverse_forward = (
        reverse_probs * (reverse_log_probs - forward_log_probs)
    ).sum(dim=-1)
    symmetric_kl = 0.5 * (kl_forward_reverse + kl_reverse_forward)
    logit_mse = (forward_logits - reverse_logits).pow(2).mean(dim=-1)
    residual_delta_l2 = (
        forward["residual_delta"][:, :-1, :]
        - reverse["residual_delta"][:, :-1, :]
    ).norm(dim=-1)
    residual_norm = forward["residual_delta"][:, :-1, :].norm(dim=-1)
    residual_norm_thresholds = _tertile_thresholds(residual_norm)
    residual_delta_thresholds = _tertile_thresholds(residual_delta_l2)
    forward_support = forward.get("support")
    reverse_support = reverse.get("support")
    rows = []
    for batch_index in range(int(target_tokens.shape[0])):
        for position_index in range(int(target_tokens.shape[1])):
            token_id = int(target_tokens[batch_index, position_index].item())
            support_churn: bool | str = ""
            forward_support_key = ""
            reverse_support_key = ""
            if forward_support is not None and reverse_support is not None:
                left = forward_support[batch_index, position_index]
                right = reverse_support[batch_index, position_index]
                left_sorted = left.sort().values
                right_sorted = right.sort().values
                support_churn = bool((left_sorted != right_sorted).any().item())
                forward_support_key = _support_key(left_sorted)
                reverse_support_key = _support_key(right_sorted)
            row_residual_norm = float(residual_norm[batch_index, position_index].item())
            row_residual_delta = float(
                residual_delta_l2[batch_index, position_index].item()
            )
            rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "batch_index": batch_index,
                    "position_index": position_index,
                    "position_bin": (
                        "even" if int(position_index) % 2 == 0 else "odd"
                    ),
                    "target_token": token_id,
                    "token_class": token_classes.get(token_id, "unknown_target"),
                    "forward_ce": float(forward_ce[batch_index, position_index].item()),
                    "reverse_ce": float(reverse_ce[batch_index, position_index].item()),
                    "ce_delta_forward_minus_reverse": float(
                        (
                            forward_ce[batch_index, position_index]
                            - reverse_ce[batch_index, position_index]
                        ).item()
                    ),
                    "ce_abs_delta": float(
                        (
                            forward_ce[batch_index, position_index]
                            - reverse_ce[batch_index, position_index]
                        )
                        .abs()
                        .item()
                    ),
                    "symmetric_kl": float(
                        symmetric_kl[batch_index, position_index].item()
                    ),
                    "logit_mse": float(logit_mse[batch_index, position_index].item()),
                    "residual_delta_l2": row_residual_delta,
                    "residual_norm": row_residual_norm,
                    "residual_norm_bin": _value_bin(
                        row_residual_norm, residual_norm_thresholds
                    ),
                    "residual_delta_l2_bin": _value_bin(
                        row_residual_delta, residual_delta_thresholds
                    ),
                    "support_churn": support_churn,
                    "forward_support": forward_support_key,
                    "reverse_support": reverse_support_key,
                }
            )
    return rows


def _target_token_classes(targets: Any) -> dict[int, str]:
    import torch

    target_tokens = targets[:, :-1].detach()
    unique, counts = torch.unique(target_tokens.reshape(-1), return_counts=True)
    if int(unique.numel()) == 0:
        return {}
    ordered = sorted(
        (
            (int(count.item()), int(token.item()))
            for token, count in zip(unique.cpu(), counts.cpu())
        ),
        key=lambda item: (-item[0], item[1]),
    )
    split = max(1, len(ordered) // 2)
    classes: dict[int, str] = {}
    for _, token in ordered[:split]:
        classes[token] = "common_target"
    for _, token in ordered[split:]:
        classes[token] = "rare_target"
    return classes


def _tertile_thresholds(values: Any) -> tuple[float, float]:
    flat = values.detach().reshape(-1)
    if int(flat.numel()) == 0:
        return (0.0, 0.0)
    lower = float(flat.quantile(1.0 / 3.0).item())
    upper = float(flat.quantile(2.0 / 3.0).item())
    return (lower, upper)


def _value_bin(value: float, thresholds: tuple[float, float]) -> str:
    lower, upper = thresholds
    if value <= lower:
        return "low"
    if value <= upper:
        return "mid"
    return "high"


def _support_key(values: Any) -> str:
    return ",".join(str(int(value.item())) for value in values.detach().cpu().reshape(-1))


def _support_churn(left: Any | None, right: Any | None) -> float | str:
    if left is None or right is None:
        return ""
    left_sorted = left.sort(dim=-1).values
    right_sorted = right.sort(dim=-1).values
    return float((left_sorted != right_sorted).any(dim=-1).float().mean().item())


def _residual_norm(delta: Any) -> float:
    return float(delta.norm(dim=-1).mean().detach().item())


def _mse_delta(left: Any, right: Any) -> float:
    import torch
    import torch.nn.functional as F

    return float(F.mse_loss(left - right, torch.zeros_like(left)).item())


def _mean_abs_delta(left: Any, right: Any) -> float:
    return float((left - right).abs().mean().item())


def _stream_l2_delta(left: Any, right: Any) -> float:
    return float((left - right).norm(dim=-1).mean().item())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    audit = summary["audit"]
    variants = audit["variants"]
    lowest_drift = min(variants, key=lambda row: float(row["anchor_ce_drift"]))
    lines = [
        "# Retention/Churn Microtest",
        "",
        f"- Experiment: `{summary['experiment_id']}`",
        f"- Config: `{summary['config_path']}`",
        f"- Status: `{summary['status']}`",
        f"- Training steps per slice: `{audit['training_steps_per_slice']}`",
        f"- Lowest anchor CE drift: `{lowest_drift['variant']}` ({lowest_drift['anchor_ce_drift']})",
        "",
        "This local diagnostic trains each control on anchor slice A, continues on transfer slice B, and measures anchor CE/logit/residual drift plus sparse support churn.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--include-mitigation-variants",
        action="store_true",
        help="Also evaluate bounded router-freeze and update-clipping top-k-2 variants.",
    )
    parser.add_argument(
        "--include-decomposition-variants",
        action="store_true",
        help="Also evaluate full-anchor then router-only/value-only transfer top-k-2 variants.",
    )
    parser.add_argument(
        "--include-value-mitigation-variants",
        action="store_true",
        help="Also evaluate value-update clipping/scaling top-k-2 variants.",
    )
    parser.add_argument(
        "--include-low-rank-value-variants",
        action="store_true",
        help="Also evaluate low-rank projected value-gradient top-k-2 variants.",
    )
    args = parser.parse_args()
    summary = run_retention_churn_microtest(
        args.config,
        args.out,
        include_mitigation_variants=args.include_mitigation_variants,
        include_decomposition_variants=args.include_decomposition_variants,
        include_value_mitigation_variants=args.include_value_mitigation_variants,
        include_low_rank_value_variants=args.include_low_rank_value_variants,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "variants": summary["audit"]["variants"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
