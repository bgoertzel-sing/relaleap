"""K-fold sequence-heldout ablation for promoted contextual support routing."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.run import _load_torch_info
from relaleap.experiments.run import _read_config
from relaleap.experiments.support_audit import _ce_loss
from relaleap.experiments.support_audit import _configured_residual_loss
from relaleap.experiments.support_audit import _score_for_support
from relaleap.experiments.support_audit import _token_losses
from relaleap.smoke import ResidualColumns
from relaleap.smoke import TinyCharTransformer
from relaleap.smoke import _build_batch
from relaleap.smoke import _residual_support_audit
from relaleap.smoke import _state_dict_delta


DEFAULT_CONFIG = Path(
    "configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml"
)
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_contextual_router_sequence_kfold_ablation"
)


def run_contextual_router_sequence_kfold_ablation(
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    max_folds: int | None = None,
) -> dict[str, Any]:
    """Train sequence folds and compare actual-router causal feature views."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("sequence K-fold ablation requires torch") from exc

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
        raise ValueError("sequence K-fold ablation expects promoted top_k: 2")
    if support_router != "contextual_mlp":
        raise ValueError("sequence K-fold ablation expects support_router: contextual_mlp")
    if max_folds is not None and max_folds < 1:
        raise ValueError("max_folds must be positive when set")

    torch.manual_seed(seed)
    inputs, targets, vocab_size = _build_batch(
        dataset=dataset,
        seq_len=seq_len,
        batch_size=4,
    )
    fold_count = int(inputs.shape[0]) if max_folds is None else min(max_folds, int(inputs.shape[0]))
    fold_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    final_support_audit: dict[str, Any] | None = None
    final_delta: float | None = None

    for fold_index in range(fold_count):
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
        train_indices = [index for index in range(int(inputs.shape[0])) if index != fold_index]
        train_inputs = inputs[train_indices]
        train_targets = targets[train_indices]
        holdout_inputs = inputs[fold_index : fold_index + 1]
        holdout_targets = targets[fold_index : fold_index + 1]

        for control in _control_specs(
            hidden_dim=hidden_dim,
            promoted_top_k=top_k,
            contextual_router_hidden_dim=contextual_router_hidden_dim,
        ):
            residual = ResidualColumns(
                hidden_dim=hidden_dim,
                num_columns=num_columns,
                atoms_per_column=atoms_per_column,
                top_k=control["top_k"],
                support_router=control["support_router"],
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
                empty_logits = base.decode(hidden)
                empty_loss = _ce_loss(empty_logits, holdout_targets, vocab_size)
                oracle = _fold_oracle(
                    base,
                    residual,
                    hidden,
                    holdout_targets,
                    vocab_size,
                    empty_loss=empty_loss,
                    num_columns=num_columns,
                    top_k=control["top_k"],
                )
                if control["name"] == "promoted_contextual_topk2":
                    final_support_audit = _residual_support_audit(
                        base,
                        residual,
                        holdout_inputs,
                    )
                    final_delta = _state_dict_delta(before_residual, residual)
                variants = _feature_variants_for_router(control["support_router"])
                for variant_name, feature_mask, uses_future_context in variants:
                    logits, support = _forward_with_feature_ablation(
                        base,
                        residual,
                        hidden,
                        feature_mask=feature_mask,
                    )
                    token_losses = _token_losses(logits, holdout_targets).reshape(-1)
                    loss_value = float(token_losses.mean().item())
                    row = {
                        "fold": fold_index,
                        "heldout_sequence_index": fold_index,
                        "control": control["name"],
                        "variant": variant_name,
                        "support_router": control["support_router"],
                        "top_k": control["top_k"],
                        "uses_future_context": uses_future_context,
                        "causal_feature_safe": not uses_future_context,
                        "positions": int(token_losses.numel()),
                        "empty_loss": empty_loss,
                        "router_loss": loss_value,
                        "oracle_loss": oracle["oracle_loss"],
                        "router_oracle_gap": loss_value - oracle["oracle_loss"],
                        "router_minus_empty_loss": loss_value - empty_loss,
                        "oracle_regret_positive_fraction": oracle[
                            "oracle_regret_positive_fraction"
                        ],
                        "unique_support_sets": _unique_support_sets(support),
                        "used_columns": _used_columns(support),
                    }
                    fold_rows.append(row)
                    support_rows.extend(
                        _support_count_rows(
                            fold_index=fold_index,
                            control=control["name"],
                            variant=variant_name,
                            support=support,
                        )
                    )

    variant_rows = _aggregate_variant_rows(fold_rows)
    decision = _decision(variant_rows)
    summary = {
        "status": "ok",
        "decision": decision["decision"],
        "claim_status": decision["claim_status"],
        "experiment_id": f"{experiment_id}_sequence_kfold_ablation",
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        **_load_torch_info(),
        "ablation": {
            "dataset": dataset,
            "seq_len": seq_len,
            "batch_size": int(inputs.shape[0]),
            "fold_count": fold_count,
            "vocab_size": vocab_size,
            "training_steps": max_steps,
            "residual_objective": residual_objective,
            "num_columns": num_columns,
            "atoms_per_column": atoms_per_column,
            "promoted_top_k": top_k,
            "promoted_support_router": support_router,
            "contextual_router_hidden_dim": contextual_router_hidden_dim,
            "controls": [spec["name"] for spec in _control_specs(
                hidden_dim=hidden_dim,
                promoted_top_k=top_k,
                contextual_router_hidden_dim=contextual_router_hidden_dim,
            )],
            "variants": {
                row["variant_key"]: row for row in variant_rows
            },
            "key_comparisons": _key_comparisons(fold_rows, variant_rows),
            "decision_reason": decision["reason"],
            "future_context_material_loss_delta": decision[
                "future_context_material_loss_delta"
            ],
            "promoted_vs_linear_loss_delta": decision["promoted_vs_linear_loss_delta"],
            "causal_contextual_vs_linear_loss_delta": decision[
                "causal_contextual_vs_linear_loss_delta"
            ],
            "causal_contextual_vs_promoted_full_loss_delta": decision[
                "causal_contextual_vs_promoted_full_loss_delta"
            ],
            "support_audit_last_promoted_fold": final_support_audit,
            "residual_parameter_delta_last_promoted_fold": final_delta,
        },
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "fold_metrics_csv": str(out_dir / "fold_metrics.csv"),
            "variant_metrics_csv": str(out_dir / "variant_metrics.csv"),
            "support_counts_csv": str(out_dir / "support_counts.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
        "git_commit": _git_commit(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "fold_metrics.csv", fold_rows)
    _write_csv(out_dir / "variant_metrics.csv", variant_rows)
    _write_csv(out_dir / "support_counts.csv", support_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _control_specs(
    *,
    hidden_dim: int,
    promoted_top_k: int,
    contextual_router_hidden_dim: int,
) -> list[dict[str, Any]]:
    del hidden_dim, contextual_router_hidden_dim
    return [
        {
            "name": "promoted_contextual_topk2",
            "support_router": "contextual_mlp",
            "top_k": promoted_top_k,
        },
        {
            "name": "linear_topk2_control",
            "support_router": "linear",
            "top_k": promoted_top_k,
        },
        {
            "name": "causal_contextual_topk2",
            "support_router": "contextual_mlp_causal",
            "top_k": promoted_top_k,
        },
        {
            "name": "contextual_topk1_control",
            "support_router": "contextual_mlp",
            "top_k": 1,
        },
        {
            "name": "causal_contextual_topk1_control",
            "support_router": "contextual_mlp_causal",
            "top_k": 1,
        },
    ]


def _feature_variants_for_router(
    support_router: str,
) -> list[tuple[str, set[str] | None, bool]]:
    if support_router == "contextual_mlp":
        return _contextual_feature_variants()
    if support_router == "contextual_mlp_causal":
        return [
            ("actual_causal_context", None, False),
            (
                "causal_current_past_position",
                {"current", "previous", "previous_delta", "position"},
                False,
            ),
            (
                "current_past_no_position",
                {"current", "previous", "previous_delta"},
                False,
            ),
            ("current_hidden_only", {"current"}, False),
            ("position_only", {"position"}, False),
            ("past_context_only", {"previous", "previous_delta"}, False),
        ]
    return [("linear_actual", None, False)]


def _contextual_feature_variants() -> list[tuple[str, set[str] | None, bool]]:
    return [
        ("actual_full_context", None, True),
        (
            "causal_current_past_position",
            {"current", "previous", "previous_delta", "position"},
            False,
        ),
        ("current_hidden_only", {"current"}, False),
        ("position_only", {"position"}, False),
        ("past_context_only", {"previous", "previous_delta"}, False),
    ]


def _fold_oracle(
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    *,
    empty_loss: float,
    num_columns: int,
    top_k: int,
) -> dict[str, Any]:
    rows = [
        _score_for_support(
            base,
            residual,
            hidden,
            targets,
            vocab_size,
            support=support,
            empty_loss=empty_loss,
            router_loss=empty_loss,
        )
        for support in itertools.combinations(range(num_columns), top_k)
    ]
    import torch

    token_loss_matrix = torch.stack([row["_token_losses"] for row in rows], dim=1)
    oracle_losses = token_loss_matrix.min(dim=1).values.detach()
    empty_token_losses = _token_losses(base.decode(hidden), targets).reshape(-1).detach()
    regret = empty_token_losses - oracle_losses
    return {
        "oracle_loss": float(oracle_losses.mean().item()),
        "oracle_regret_positive_fraction": float(
            (regret > 0).to(dtype=oracle_losses.dtype).mean().item()
        ),
    }


def _forward_with_feature_ablation(
    base: Any,
    residual: Any,
    hidden: Any,
    *,
    feature_mask: set[str] | None,
) -> tuple[Any, Any]:
    if getattr(residual, "support_router", None) not in {
        "contextual_mlp",
        "contextual_mlp_causal",
    }:
        output, support = residual(hidden, return_support=True)
        return base.decode(output), support
    if feature_mask is None:
        output, support = residual(hidden, return_support=True)
        return base.decode(output), support
    import torch

    features = _masked_contextual_features(hidden, feature_mask)
    scores = residual.contextual_column_scores(features) + residual.score_tie_breaker.to(
        device=hidden.device,
        dtype=hidden.dtype,
    )
    top_values, top_indices = scores.topk(residual.top_k, dim=-1)
    column_weights = torch.softmax(top_values, dim=-1)
    atom_weights = torch.softmax(residual.atom_logits, dim=-1)
    column_values = torch.einsum("ca,cah->ch", atom_weights, residual.atom_values)
    selected_values = column_values[top_indices]
    residual_update = torch.einsum("bsk,bskh->bsh", column_weights, selected_values)
    return base.decode(hidden + residual_update), top_indices.detach()


def _masked_contextual_features(hidden: Any, feature_mask: set[str]) -> Any:
    import torch

    current = hidden.detach()
    previous = torch.cat([current[:, :1, :], current[:, :-1, :]], dim=1)
    next_hidden = torch.cat([current[:, 1:, :], current[:, -1:, :]], dim=1)
    seq_len = int(current.shape[1])
    if seq_len <= 1:
        normalized_position = torch.zeros(
            current.shape[0],
            seq_len,
            1,
            dtype=current.dtype,
            device=current.device,
        )
    else:
        normalized_position = torch.linspace(
            0.0,
            1.0,
            seq_len,
            dtype=current.dtype,
            device=current.device,
        ).view(1, seq_len, 1).expand(current.shape[0], seq_len, 1)
    angle = normalized_position * (2.0 * torch.pi)
    zero_hidden = torch.zeros_like(current)
    zero_pos = torch.zeros_like(normalized_position)
    chunks = [
        current if "current" in feature_mask else zero_hidden,
        previous if "previous" in feature_mask else zero_hidden,
        next_hidden if "next" in feature_mask else zero_hidden,
        current - previous if "previous_delta" in feature_mask else zero_hidden,
        next_hidden - current if "next_delta" in feature_mask else zero_hidden,
        normalized_position if "position" in feature_mask else zero_pos,
        torch.sin(angle) if "position" in feature_mask else zero_pos,
        torch.cos(angle) if "position" in feature_mask else zero_pos,
    ]
    return torch.cat(chunks, dim=-1)


def _unique_support_sets(support: Any) -> int:
    flattened = support.reshape(-1, support.shape[-1]).detach().cpu().tolist()
    return len({tuple(int(value) for value in row) for row in flattened})


def _used_columns(support: Any) -> int:
    return len({int(value) for value in support.reshape(-1).detach().cpu().tolist()})


def _support_count_rows(
    *,
    fold_index: int,
    control: str,
    variant: str,
    support: Any,
) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in support.reshape(-1, support.shape[-1]).detach().cpu().tolist():
        key = ",".join(str(int(value)) for value in row)
        counts[key] = counts.get(key, 0) + 1
    return [
        {
            "fold": fold_index,
            "control": control,
            "variant": variant,
            "support": key,
            "count": count,
        }
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _aggregate_variant_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["control"]), str(row["variant"])), []).append(row)
    aggregates: list[dict[str, Any]] = []
    for (control, variant), group in sorted(grouped.items()):
        mean_router = _mean(group, "router_loss")
        mean_oracle = _mean(group, "oracle_loss")
        aggregates.append(
            {
                "variant_key": f"{control}:{variant}",
                "control": control,
                "variant": variant,
                "folds": len(group),
                "support_router": group[0]["support_router"],
                "top_k": group[0]["top_k"],
                "uses_future_context": group[0]["uses_future_context"],
                "causal_feature_safe": group[0]["causal_feature_safe"],
                "mean_router_loss": mean_router,
                "mean_oracle_loss": mean_oracle,
                "mean_router_oracle_gap": mean_router - mean_oracle,
                "mean_router_minus_empty_loss": _mean(group, "router_minus_empty_loss"),
                "mean_unique_support_sets": _mean(group, "unique_support_sets"),
                "mean_used_columns": _mean(group, "used_columns"),
            }
        )
    return aggregates


def _key_comparisons(
    fold_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    fold_by_key: dict[str, dict[int, dict[str, Any]]] = {}
    for row in fold_rows:
        key = f"{row['control']}:{row['variant']}"
        fold_by_key.setdefault(key, {})[int(row["fold"])] = row
    aggregate_by_key = {row["variant_key"]: row for row in variant_rows}
    comparisons = {
        "causal_contextual_vs_linear": (
            "causal_contextual_topk2:actual_causal_context",
            "linear_topk2_control:linear_actual",
        ),
        "causal_contextual_vs_full_context_oracle_baseline": (
            "causal_contextual_topk2:actual_causal_context",
            "promoted_contextual_topk2:actual_full_context",
        ),
        "full_context_oracle_baseline_vs_linear": (
            "promoted_contextual_topk2:actual_full_context",
            "linear_topk2_control:linear_actual",
        ),
    }
    result: dict[str, Any] = {}
    for name, (left_key, right_key) in comparisons.items():
        left_folds = fold_by_key.get(left_key, {})
        right_folds = fold_by_key.get(right_key, {})
        fold_deltas = []
        for fold in sorted(set(left_folds) & set(right_folds)):
            left = left_folds[fold]
            right = right_folds[fold]
            fold_deltas.append(
                {
                    "fold": fold,
                    "left_variant_key": left_key,
                    "right_variant_key": right_key,
                    "loss_delta": float(left["router_loss"])
                    - float(right["router_loss"]),
                    "left_router_loss": float(left["router_loss"]),
                    "right_router_loss": float(right["router_loss"]),
                    "left_oracle_gap": float(left["router_oracle_gap"]),
                    "right_oracle_gap": float(right["router_oracle_gap"]),
                    "left_used_columns": int(left["used_columns"]),
                    "right_used_columns": int(right["used_columns"]),
                    "left_unique_support_sets": int(left["unique_support_sets"]),
                    "right_unique_support_sets": int(right["unique_support_sets"]),
                }
            )
        aggregate_left = aggregate_by_key.get(left_key, {})
        aggregate_right = aggregate_by_key.get(right_key, {})
        result[name] = {
            "left_variant_key": left_key,
            "right_variant_key": right_key,
            "mean_loss_delta": (
                float(aggregate_left["mean_router_loss"])
                - float(aggregate_right["mean_router_loss"])
                if aggregate_left and aggregate_right
                else None
            ),
            "fold_count": len(fold_deltas),
            "left_wins": sum(1 for row in fold_deltas if row["loss_delta"] < 0.0),
            "right_wins": sum(1 for row in fold_deltas if row["loss_delta"] > 0.0),
            "ties": sum(1 for row in fold_deltas if row["loss_delta"] == 0.0),
            "fold_deltas": fold_deltas,
        }
    return result


def _decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {row["variant_key"]: row for row in rows}
    full = by_key.get("promoted_contextual_topk2:actual_full_context")
    causal = by_key.get("promoted_contextual_topk2:causal_current_past_position")
    causal_trained = by_key.get("causal_contextual_topk2:actual_causal_context")
    linear = by_key.get("linear_topk2_control:linear_actual")
    future_delta = None
    promoted_vs_linear = None
    causal_vs_linear = None
    causal_vs_promoted_full = None
    if full is not None and causal is not None:
        future_delta = float(causal["mean_router_loss"]) - float(full["mean_router_loss"])
    if full is not None and linear is not None:
        promoted_vs_linear = float(full["mean_router_loss"]) - float(linear["mean_router_loss"])
    if causal_trained is not None and linear is not None:
        causal_vs_linear = float(causal_trained["mean_router_loss"]) - float(
            linear["mean_router_loss"]
        )
    if causal_trained is not None and full is not None:
        causal_vs_promoted_full = float(causal_trained["mean_router_loss"]) - float(
            full["mean_router_loss"]
        )
    if promoted_vs_linear is not None and promoted_vs_linear > 0.0:
        return {
            "decision": "promoted_contextual_router_sequence_holdout_underperforms_linear",
            "claim_status": "promoted_router_sequence_holdout_predictive_claim_blocked",
            "reason": "The promoted contextual top-k-2 router has higher mean sequence-heldout CE than the linear top-k-2 control.",
            "future_context_material_loss_delta": future_delta,
            "promoted_vs_linear_loss_delta": promoted_vs_linear,
            "causal_contextual_vs_linear_loss_delta": causal_vs_linear,
            "causal_contextual_vs_promoted_full_loss_delta": causal_vs_promoted_full,
        }
    if causal_vs_linear is not None and causal_vs_linear > 0.0:
        return {
            "decision": "causal_contextual_router_sequence_holdout_underperforms_linear",
            "claim_status": "causal_feature_safe_router_claim_blocked_by_linear_control",
            "reason": "The trained causal contextual top-k-2 router has higher mean sequence-heldout CE than the linear top-k-2 control.",
            "future_context_material_loss_delta": future_delta,
            "promoted_vs_linear_loss_delta": promoted_vs_linear,
            "causal_contextual_vs_linear_loss_delta": causal_vs_linear,
            "causal_contextual_vs_promoted_full_loss_delta": causal_vs_promoted_full,
        }
    if causal_vs_linear is not None and causal_vs_linear <= 0.0:
        return {
            "decision": "causal_contextual_router_sequence_holdout_candidate",
            "claim_status": "causal_feature_safe_router_local_sequence_holdout_supported",
            "reason": "The trained causal contextual top-k-2 router matches or beats the linear top-k-2 control on mean sequence-heldout CE.",
            "future_context_material_loss_delta": future_delta,
            "promoted_vs_linear_loss_delta": promoted_vs_linear,
            "causal_contextual_vs_linear_loss_delta": causal_vs_linear,
            "causal_contextual_vs_promoted_full_loss_delta": causal_vs_promoted_full,
        }
    if future_delta is not None and future_delta > 0.01:
        return {
            "decision": "future_context_features_material_for_promoted_router",
            "claim_status": "deployable_autoregressive_router_claim_blocked_by_future_context",
            "reason": "Removing future-token feature groups worsens mean sequence-heldout CE by more than 0.01.",
            "future_context_material_loss_delta": future_delta,
            "promoted_vs_linear_loss_delta": promoted_vs_linear,
            "causal_contextual_vs_linear_loss_delta": causal_vs_linear,
            "causal_contextual_vs_promoted_full_loss_delta": causal_vs_promoted_full,
        }
    return {
        "decision": "sequence_kfold_causal_feature_ablation_completed",
        "claim_status": "sequence_holdout_router_ablation_recorded",
        "reason": "The K-fold report records promoted-router sequence holdout versus causal-feature and linear/top-k controls without a fail-closed threshold breach.",
        "future_context_material_loss_delta": future_delta,
        "promoted_vs_linear_loss_delta": promoted_vs_linear,
        "causal_contextual_vs_linear_loss_delta": causal_vs_linear,
        "causal_contextual_vs_promoted_full_loss_delta": causal_vs_promoted_full,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(1, len(rows))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    ablation = summary["ablation"]
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "K-fold sequence-heldout causal-feature ablation for the promoted contextual router.",
        "",
        f"- Config: `{summary['config_path']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Folds: `{ablation['fold_count']}`",
        f"- Future-context loss delta: `{ablation['future_context_material_loss_delta']}`",
        f"- Promoted-vs-linear loss delta: `{ablation['promoted_vs_linear_loss_delta']}`",
        "- Causal-contextual-vs-linear loss delta: "
        f"`{ablation['causal_contextual_vs_linear_loss_delta']}`",
        "- Causal-contextual-vs-promoted-full loss delta: "
        f"`{ablation['causal_contextual_vs_promoted_full_loss_delta']}`",
        "",
        "## Key Fold Comparisons",
    ]
    for name, comparison in ablation["key_comparisons"].items():
        lines.append(
            "- "
            f"{name}: mean delta `{comparison['mean_loss_delta']}`, "
            f"left wins `{comparison['left_wins']}/{comparison['fold_count']}`"
        )
    lines.extend([
        "",
        "## Mean Heldout Loss",
    ])
    for row in sorted(
        ablation["variants"].values(),
        key=lambda item: float(item["mean_router_loss"]),
    ):
        lines.append(
            "- "
            f"{row['variant_key']}: `{row['mean_router_loss']}` "
            f"(oracle gap `{row['mean_router_oracle_gap']}`, "
            f"causal `{row['causal_feature_safe']}`)"
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
    summary = run_contextual_router_sequence_kfold_ablation(
        args.config,
        args.out,
        max_folds=args.max_folds,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
