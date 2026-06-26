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


def run_retention_churn_microtest(
    config_path: Path,
    out_dir: Path,
    *,
    include_mitigation_variants: bool = False,
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

    variant_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
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

        phase_rows.extend(
            _phase_rows(
                variant=spec.name,
                anchor_before=anchor_before,
                anchor_after=anchor_after,
                transfer_before=transfer_before,
                transfer_after=transfer_after,
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
            "parameter_delta_after_anchor": parameter_delta_after_anchor,
            "parameter_delta_during_transfer": parameter_delta_during_transfer,
            "freeze_router_during_transfer": spec.freeze_router_during_transfer,
            "gradient_clip_norm": (
                "" if spec.gradient_clip_norm is None else spec.gradient_clip_norm
            ),
        }
        variant_rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "variant_metrics.csv", variant_rows)
    _write_csv(out_dir / "phase_metrics.csv", phase_rows)
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
            ],
            "include_mitigation_variants": include_mitigation_variants,
        },
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "variant_metrics_csv": str(out_dir / "variant_metrics.csv"),
            "phase_metrics_csv": str(out_dir / "phase_metrics.csv"),
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
) -> None:
    import torch
    import torch.nn.functional as F

    residual.train()
    trainable_parameters = [
        parameter
        for name, parameter in residual.named_parameters()
        if not (freeze_router and _is_router_parameter(name))
    ]
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
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, gradient_clip_norm)
        optimizer.step()
    residual.eval()


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
    args = parser.parse_args()
    summary = run_retention_churn_microtest(
        args.config,
        args.out,
        include_mitigation_variants=args.include_mitigation_variants,
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
