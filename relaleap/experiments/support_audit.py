"""Exhaustive residual-column support audit for small RelaLeap runs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import platform
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.run import _load_torch_info
from relaleap.experiments.run import _read_config
from relaleap.smoke import ResidualColumns
from relaleap.smoke import SUPPORT_ROUTER_CHOICES
from relaleap.smoke import TinyCharTransformer
from relaleap.smoke import _build_batch
from relaleap.smoke import _residual_loss
from relaleap.smoke import _residual_support_audit
from relaleap.smoke import _state_dict_delta


DEFAULT_CONFIG = Path(
    "configs/char_validation_support_wide_hep_temporal_clipped_objective_gate.yaml"
)
DEFAULT_OUT_DIR = Path("results/audits/validation_support_wide_exhaustive_support")


def run_support_audit(config_path: Path, out_dir: Path) -> dict[str, Any]:
    """Train a configured residual adapter and exhaustively score support sets."""

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("support audit requires an importable torch install") from exc

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
        raise ValueError("the first exhaustive support audit expects model.columns.top_k: 2")
    if support_router not in SUPPORT_ROUTER_CHOICES:
        raise ValueError(
            "model.columns.support_router must be one of: "
            "linear, contextual_mlp, contextual_mlp_causal"
        )
    if contextual_router_hidden_dim < 1:
        raise ValueError("model.columns.contextual_router_hidden_dim must be positive")

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
    residual = ResidualColumns(
        hidden_dim=hidden_dim,
        num_columns=num_columns,
        atoms_per_column=atoms_per_column,
        top_k=top_k,
        support_router=support_router,
        contextual_router_hidden_dim=contextual_router_hidden_dim,
    )
    base.eval()
    residual.train()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    before_residual = {
        key: value.detach().clone() for key, value in residual.state_dict().items()
    }
    optimizer = torch.optim.AdamW(residual.parameters(), lr=learning_rate)
    for _ in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _configured_residual_loss(
            base,
            residual,
            inputs,
            targets,
            vocab_size,
            residual_objective=residual_objective,
            training_cfg=training_cfg,
        )
        loss.backward()
        optimizer.step()

    residual.eval()
    with torch.no_grad():
        hidden = base.encode(inputs)
        ordinary_hidden, router_support = residual(hidden, return_support=True)
        ordinary_logits = base.decode(ordinary_hidden)
        empty_logits = base.decode(hidden)
        router_loss = _ce_loss(ordinary_logits, targets, vocab_size)
        empty_loss = _ce_loss(empty_logits, targets, vocab_size)
        router_token_losses = _token_losses(ordinary_logits, targets)

        singleton_rows = [
            _score_for_support(
                base,
                residual,
                hidden,
                targets,
                vocab_size,
                support=(column,),
                empty_loss=empty_loss,
                router_loss=router_loss,
            )
            for column in range(num_columns)
        ]
        pair_rows = [
            _score_for_support(
                base,
                residual,
                hidden,
                targets,
                vocab_size,
                support=pair,
                empty_loss=empty_loss,
                router_loss=router_loss,
            )
            for pair in itertools.combinations(range(num_columns), top_k)
        ]

    singleton_gain = {
        tuple(row["support"]): float(row["gain_from_empty"]) for row in singleton_rows
    }
    for row in pair_rows:
        a, b = row["support"]
        row["pairwise_synergy"] = (
            float(row["gain_from_empty"])
            - singleton_gain[(a,)]
            - singleton_gain[(b,)]
        )

    best_pair = min(pair_rows, key=lambda row: float(row["loss"]))
    token_oracle = _token_oracle(router_token_losses, pair_rows)
    router_target = _router_oracle_target_diagnostic(
        hidden,
        router_token_losses,
        pair_rows,
        oracle_indices=token_oracle["_oracle_indices"],
        seed=seed,
    )
    dominant_router = _dominant_router_support(router_support)
    router_support_row = _find_support_row(pair_rows, dominant_router["support"])
    one_swap_rows = _one_swap_neighbors(pair_rows, dominant_router["support"])
    best_one_swap = min(one_swap_rows, key=lambda row: float(row["loss"])) if one_swap_rows else None
    global_fixed_support_gap = router_loss - float(best_pair["loss"])
    dominant_router_regret = (
        None if router_support_row is None else float(router_support_row["loss"]) - float(best_pair["loss"])
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    support_rows = sorted(singleton_rows + pair_rows, key=lambda row: (len(row["support"]), tuple(row["support"])))
    _write_support_losses(out_dir / "support_losses.csv", support_rows)
    _write_pairwise_synergy(out_dir / "pairwise_synergy.csv", pair_rows)
    _write_router_target_diagnostic(
        out_dir / "router_target_diagnostic.csv",
        router_target["splits"],
    )
    _write_router_target_diagnostic(
        out_dir / "router_target_nonlinear_diagnostic.csv",
        router_target["nonlinear_splits"],
    )
    _write_router_target_diagnostic(
        out_dir / "router_target_contextual_diagnostic.csv",
        router_target["contextual_splits"],
    )
    _write_router_target_diagnostic(
        out_dir / "router_target_contextual_sequence_diagnostic.csv",
        router_target["contextual_sequence_splits"],
    )
    contextual_intervention = _router_support_intervention(
        base,
        residual,
        hidden,
        targets,
        vocab_size,
        rows=pair_rows,
        predicted_indices=router_target["_contextual_predicted_indices"],
        train_mask=router_target["_train_mask"],
        router_token_losses=router_token_losses,
    )
    contextual_sequence_intervention = _router_support_intervention(
        base,
        residual,
        hidden,
        targets,
        vocab_size,
        rows=pair_rows,
        predicted_indices=router_target["_contextual_sequence_predicted_indices"],
        train_mask=router_target["_sequence_train_mask"],
        router_token_losses=router_token_losses,
        train_split_name="train_even_sequences",
        holdout_split_name="holdout_odd_sequences",
    )
    contextual_head = _contextual_router_support_head(
        base,
        residual,
        hidden,
        targets,
        vocab_size,
        rows=pair_rows,
        train_mask=router_target["_train_mask"],
        router_token_losses=router_token_losses,
        seed=seed,
    )
    contextual_sequence_head = _contextual_router_support_head(
        base,
        residual,
        hidden,
        targets,
        vocab_size,
        rows=pair_rows,
        train_mask=router_target["_sequence_train_mask"],
        router_token_losses=router_token_losses,
        seed=seed + 17,
        train_split_label="even full sequences",
        holdout_split_label="odd full sequences",
        train_split_name="train_even_sequences",
        holdout_split_name="holdout_odd_sequences",
    )
    _write_router_support_intervention(
        out_dir / "router_support_intervention.csv",
        contextual_intervention["splits"],
    )
    _write_router_support_intervention(
        out_dir / "contextual_router_support_head.csv",
        contextual_head["splits"],
    )
    _write_router_support_intervention(
        out_dir / "router_support_sequence_intervention.csv",
        contextual_sequence_intervention["splits"],
    )
    _write_router_support_intervention(
        out_dir / "contextual_router_support_sequence_head.csv",
        contextual_sequence_head["splits"],
    )
    summary = {
        "status": "ok",
        "experiment_id": f"{experiment_id}_exhaustive_support_audit",
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        **_load_torch_info(),
        "audit": {
            "dataset": dataset,
            "seq_len": seq_len,
            "batch_size": int(inputs.shape[0]),
            "vocab_size": vocab_size,
            "training_steps": max_steps,
            "residual_objective": residual_objective,
            "num_columns": num_columns,
            "top_k": top_k,
            "support_router": support_router,
            "contextual_router_hidden_dim": contextual_router_hidden_dim,
            "support_set_count": len(pair_rows),
            "singleton_count": len(singleton_rows),
            "empty_loss": empty_loss,
            "router_loss": router_loss,
            "oracle_loss": token_oracle["oracle_loss"],
            "oracle_support_regret": token_oracle["oracle_support_regret"],
            "oracle_support_regret_positive_fraction": token_oracle[
                "oracle_support_regret_positive_fraction"
            ],
            "oracle_support_counts": token_oracle["oracle_support_counts"],
            "best_global_fixed_support_loss": float(best_pair["loss"]),
            "best_global_fixed_support": _support_key(best_pair["support"]),
            "router_minus_best_global_fixed_support_loss": global_fixed_support_gap,
            "dominant_router_support": _support_key(dominant_router["support"]),
            "dominant_router_support_count": dominant_router["count"],
            "dominant_router_support_loss": (
                None if router_support_row is None else float(router_support_row["loss"])
            ),
            "dominant_router_support_regret": dominant_router_regret,
            "best_one_swap_support": (
                None if best_one_swap is None else _support_key(best_one_swap["support"])
            ),
            "best_one_swap_loss": (
                None if best_one_swap is None else float(best_one_swap["loss"])
            ),
            "best_one_swap_recovers_oracle_gap_fraction": _gap_recovery_fraction(
                router_loss=router_loss,
                oracle_loss=float(best_pair["loss"]),
                candidate_loss=None if best_one_swap is None else float(best_one_swap["loss"]),
            ),
            "loss_distribution": _loss_distribution(pair_rows),
            "top_supports_by_loss": _top_supports(pair_rows, key="loss", reverse=False),
            "top_supports_by_synergy": _top_supports(pair_rows, key="pairwise_synergy", reverse=True),
            "router_oracle_target_diagnostic": router_target["summary"],
            "router_oracle_target_nonlinear_diagnostic": router_target[
                "nonlinear_summary"
            ],
            "router_oracle_target_contextual_diagnostic": router_target[
                "contextual_summary"
            ],
            "router_oracle_target_contextual_sequence_diagnostic": router_target[
                "contextual_sequence_summary"
            ],
            "contextual_router_support_intervention": contextual_intervention[
                "summary"
            ],
            "contextual_router_support_head": contextual_head["summary"],
            "contextual_router_support_sequence_intervention": contextual_sequence_intervention[
                "summary"
            ],
            "contextual_router_support_sequence_head": contextual_sequence_head[
                "summary"
            ],
            "support_audit": _residual_support_audit(base, residual, inputs),
            "residual_parameter_delta": _state_dict_delta(before_residual, residual),
        },
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "support_losses_csv": str(out_dir / "support_losses.csv"),
            "pairwise_synergy_csv": str(out_dir / "pairwise_synergy.csv"),
            "router_target_diagnostic_csv": str(out_dir / "router_target_diagnostic.csv"),
            "router_target_nonlinear_diagnostic_csv": str(
                out_dir / "router_target_nonlinear_diagnostic.csv"
            ),
            "router_target_contextual_diagnostic_csv": str(
                out_dir / "router_target_contextual_diagnostic.csv"
            ),
            "router_target_contextual_sequence_diagnostic_csv": str(
                out_dir / "router_target_contextual_sequence_diagnostic.csv"
            ),
            "router_support_intervention_csv": str(
                out_dir / "router_support_intervention.csv"
            ),
            "contextual_router_support_head_csv": str(
                out_dir / "contextual_router_support_head.csv"
            ),
            "router_support_sequence_intervention_csv": str(
                out_dir / "router_support_sequence_intervention.csv"
            ),
            "contextual_router_support_sequence_head_csv": str(
                out_dir / "contextual_router_support_sequence_head.csv"
            ),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _configured_residual_loss(
    base: Any,
    residual: Any,
    inputs: Any,
    targets: Any,
    vocab_size: int,
    *,
    residual_objective: str,
    training_cfg: dict[str, Any],
) -> Any:
    return _residual_loss(
        base,
        residual,
        inputs,
        targets,
        vocab_size,
        objective=residual_objective,
        ce_anchor_weight=float(training_cfg.get("ce_anchor_weight", 0.1)),
        confidence_penalty_weight=float(
            training_cfg.get("confidence_penalty_weight", 0.01)
        ),
        margin_penalty_weight=float(training_cfg.get("margin_penalty_weight", 0.01)),
        target_logit_margin=float(training_cfg.get("target_logit_margin", 0.25)),
        label_smoothing_weight=float(training_cfg.get("label_smoothing_weight", 0.05)),
        focal_gamma=float(training_cfg.get("focal_gamma", 2.0)),
        temporal_consistency_weight=float(
            training_cfg.get("temporal_consistency_weight", 0.01)
        ),
    )


def _ce_loss(logits: Any, targets: Any, vocab_size: int) -> float:
    import torch.nn.functional as F

    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab_size),
        targets[:, :-1].reshape(-1),
    )
    return float(loss.detach().item())


def _token_losses(logits: Any, targets: Any) -> Any:
    import torch.nn.functional as F

    vocab_size = int(logits.shape[-1])
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab_size),
        targets[:, :-1].reshape(-1),
        reduction="none",
    ).detach()


def _score_for_support(
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    *,
    support: tuple[int, ...],
    empty_loss: float,
    router_loss: float,
) -> dict[str, Any]:
    import torch

    support_indices = torch.tensor(
        support,
        dtype=torch.long,
        device=hidden.device,
    ).view(1, 1, len(support)).expand(hidden.shape[0], hidden.shape[1], len(support))
    logits = base.decode(residual(hidden, support_indices=support_indices))
    token_losses = _token_losses(logits, targets)
    loss = _ce_loss(logits, targets, vocab_size)
    return {
        "support": support,
        "support_key": _support_key(support),
        "support_size": len(support),
        "loss": loss,
        "_token_losses": token_losses,
        "gain_from_empty": empty_loss - loss,
        "delta_from_router": loss - router_loss,
    }


def _token_oracle(router_token_losses: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    stacked = torch.stack([row["_token_losses"] for row in rows], dim=0)
    oracle_losses, oracle_indices = stacked.min(dim=0)
    regret = router_token_losses - oracle_losses
    support_counts: dict[str, int] = {}
    for index in oracle_indices.detach().cpu().tolist():
        support_key = str(rows[int(index)]["support_key"])
        support_counts[support_key] = support_counts.get(support_key, 0) + 1
    return {
        "oracle_loss": float(oracle_losses.mean().item()),
        "oracle_support_regret": float(regret.mean().item()),
        "oracle_support_regret_positive_fraction": float(
            (regret > 0).to(dtype=torch.float32).mean().item()
        ),
        "oracle_support_counts": dict(
            sorted(support_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "_oracle_indices": oracle_indices.detach(),
    }


def _router_oracle_target_diagnostic(
    hidden: Any,
    router_token_losses: Any,
    rows: list[dict[str, Any]],
    *,
    oracle_indices: Any,
    seed: int,
    steps: int = 200,
) -> dict[str, Any]:
    """Train tiny hidden-state probes to imitate per-token oracle supports."""

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed + 1009)
    features = hidden[:, :-1, :].reshape(-1, hidden.shape[-1]).detach()
    targets = oracle_indices.reshape(-1).detach().to(dtype=torch.long)
    token_loss_matrix = torch.stack([row["_token_losses"] for row in rows], dim=1)
    router_losses = router_token_losses.reshape(-1).detach()
    oracle_losses = token_loss_matrix.min(dim=1).values.detach()
    position_ids = torch.arange(features.shape[0], device=features.device)
    train_mask = position_ids.remainder(2) == 0
    holdout_mask = ~train_mask

    linear = _train_router_target_probe(
        nn.Linear(features.shape[-1], len(rows), bias=True).to(features.device),
        features=features,
        targets=targets,
        train_mask=train_mask,
        token_loss_matrix=token_loss_matrix,
        router_losses=router_losses,
        oracle_losses=oracle_losses,
        rows=rows,
        steps=steps,
        learning_rate=0.05,
        weight_decay=1e-4,
    )
    hidden_width = max(16, min(128, features.shape[-1] * 2))
    nonlinear = _train_router_target_probe(
        nn.Sequential(
            nn.LayerNorm(features.shape[-1]),
            nn.Linear(features.shape[-1], hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, len(rows)),
        ).to(features.device),
        features=features,
        targets=targets,
        train_mask=train_mask,
        token_loss_matrix=token_loss_matrix,
        router_losses=router_losses,
        oracle_losses=oracle_losses,
        rows=rows,
        steps=steps,
        learning_rate=0.01,
        weight_decay=1e-3,
    )
    contextual_features = _contextual_router_features(hidden)
    contextual_width = max(16, min(128, contextual_features.shape[-1]))
    contextual = _train_router_target_probe(
        nn.Sequential(
            nn.LayerNorm(contextual_features.shape[-1]),
            nn.Linear(contextual_features.shape[-1], contextual_width),
            nn.GELU(),
            nn.Linear(contextual_width, len(rows)),
        ).to(contextual_features.device),
        features=contextual_features,
        targets=targets,
        train_mask=train_mask,
        token_loss_matrix=token_loss_matrix,
        router_losses=router_losses,
        oracle_losses=oracle_losses,
        rows=rows,
        steps=steps,
        learning_rate=0.01,
        weight_decay=1e-3,
    )
    sequence_train_mask = _sequence_level_train_mask(hidden)
    contextual_sequence = _train_router_target_probe(
        nn.Sequential(
            nn.LayerNorm(contextual_features.shape[-1]),
            nn.Linear(contextual_features.shape[-1], contextual_width),
            nn.GELU(),
            nn.Linear(contextual_width, len(rows)),
        ).to(contextual_features.device),
        features=contextual_features,
        targets=targets,
        train_mask=sequence_train_mask,
        token_loss_matrix=token_loss_matrix,
        router_losses=router_losses,
        oracle_losses=oracle_losses,
        rows=rows,
        steps=steps,
        learning_rate=0.01,
        weight_decay=1e-3,
        train_split_name="train_even_sequences",
        holdout_split_name="holdout_odd_sequences",
    )
    linear_by_name = {row["split"]: row for row in linear["splits"]}
    nonlinear_by_name = {row["split"]: row for row in nonlinear["splits"]}
    contextual_by_name = {row["split"]: row for row in contextual["splits"]}
    contextual_sequence_by_name = {
        row["split"]: row for row in contextual_sequence["splits"]
    }
    return {
        "summary": {
            "selector": "linear_hidden_to_oracle_pair",
            "training_steps": steps,
            "train_split": "even flattened token positions",
            "holdout_split": "odd flattened token positions",
            "selected_support_counts": linear["selected_support_counts"],
            "all": linear_by_name["all"],
            "holdout": linear_by_name["holdout_odd_positions"],
        },
        "splits": linear["splits"],
        "nonlinear_summary": {
            "selector": "mlp_hidden_to_oracle_pair",
            "training_steps": steps,
            "hidden_width": hidden_width,
            "train_split": "even flattened token positions",
            "holdout_split": "odd flattened token positions",
            "selected_support_counts": nonlinear["selected_support_counts"],
            "all": nonlinear_by_name["all"],
            "holdout": nonlinear_by_name["holdout_odd_positions"],
        },
        "nonlinear_splits": nonlinear["splits"],
        "contextual_summary": {
            "selector": "mlp_contextual_hidden_to_oracle_pair",
            "training_steps": steps,
            "hidden_width": contextual_width,
            "features": [
                "current_hidden",
                "previous_hidden",
                "next_hidden",
                "previous_delta",
                "next_delta",
                "normalized_token_position",
                "position_sin",
                "position_cos",
            ],
            "train_split": "even flattened token positions",
            "holdout_split": "odd flattened token positions",
            "selected_support_counts": contextual["selected_support_counts"],
            "all": contextual_by_name["all"],
            "holdout": contextual_by_name["holdout_odd_positions"],
        },
        "contextual_splits": contextual["splits"],
        "contextual_sequence_summary": {
            "selector": "mlp_contextual_hidden_to_oracle_pair",
            "training_steps": steps,
            "hidden_width": contextual_width,
            "features": [
                "current_hidden",
                "previous_hidden",
                "next_hidden",
                "previous_delta",
                "next_delta",
                "normalized_token_position",
                "position_sin",
                "position_cos",
            ],
            "train_split": "even full sequences",
            "holdout_split": "odd full sequences",
            "selected_support_counts": contextual_sequence["selected_support_counts"],
            "all": contextual_sequence_by_name["all"],
            "holdout": contextual_sequence_by_name["holdout_odd_sequences"],
        },
        "contextual_sequence_splits": contextual_sequence["splits"],
        "_contextual_predicted_indices": contextual["_predicted_indices"],
        "_contextual_sequence_predicted_indices": contextual_sequence[
            "_predicted_indices"
        ],
        "_train_mask": train_mask,
        "_sequence_train_mask": sequence_train_mask,
    }


def _contextual_router_features(hidden: Any) -> Any:
    import torch

    current = hidden[:, :-1, :].detach()
    previous = torch.cat([current[:, :1, :], current[:, :-1, :]], dim=1)
    next_hidden = hidden[:, 1:, :].detach()
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
    context = torch.cat(
        [
            current,
            previous,
            next_hidden,
            current - previous,
            next_hidden - current,
            normalized_position,
            torch.sin(angle),
            torch.cos(angle),
        ],
        dim=-1,
    )
    return context.reshape(-1, context.shape[-1])


def _sequence_level_train_mask(hidden: Any) -> Any:
    import torch

    batch_size = int(hidden.shape[0])
    positions_per_sequence = max(0, int(hidden.shape[1]) - 1)
    sequence_ids = torch.arange(batch_size, device=hidden.device).view(batch_size, 1)
    sequence_ids = sequence_ids.expand(batch_size, positions_per_sequence).reshape(-1)
    return sequence_ids.remainder(2) == 0


def _train_router_target_probe(
    selector: Any,
    *,
    features: Any,
    targets: Any,
    train_mask: Any,
    token_loss_matrix: Any,
    router_losses: Any,
    oracle_losses: Any,
    rows: list[dict[str, Any]],
    steps: int,
    learning_rate: float,
    weight_decay: float,
    train_split_name: str = "train_even_positions",
    holdout_split_name: str = "holdout_odd_positions",
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    optimizer = torch.optim.AdamW(
        selector.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = selector(features[train_mask])
        loss = F.cross_entropy(logits, targets[train_mask])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = selector(features)
        predicted_indices = logits.argmax(dim=-1)
        selected_losses = token_loss_matrix[
            torch.arange(token_loss_matrix.shape[0], device=features.device),
            predicted_indices,
        ].detach()
        selected_counts: dict[str, int] = {}
        for index in predicted_indices.detach().cpu().tolist():
            key = str(rows[int(index)]["support_key"])
            selected_counts[key] = selected_counts.get(key, 0) + 1

    holdout_mask = ~train_mask
    splits = [
        _router_target_split(
            "all",
            torch.ones_like(train_mask, dtype=torch.bool),
            targets=targets,
            predicted_indices=predicted_indices,
            router_losses=router_losses,
            oracle_losses=oracle_losses,
            selected_losses=selected_losses,
        ),
        _router_target_split(
            train_split_name,
            train_mask,
            targets=targets,
            predicted_indices=predicted_indices,
            router_losses=router_losses,
            oracle_losses=oracle_losses,
            selected_losses=selected_losses,
        ),
        _router_target_split(
            holdout_split_name,
            holdout_mask,
            targets=targets,
            predicted_indices=predicted_indices,
            router_losses=router_losses,
            oracle_losses=oracle_losses,
            selected_losses=selected_losses,
        ),
    ]
    return {
        "splits": splits,
        "selected_support_counts": dict(
            sorted(selected_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "_predicted_indices": predicted_indices.detach(),
    }


def _router_support_intervention(
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    *,
    rows: list[dict[str, Any]],
    predicted_indices: Any,
    train_mask: Any,
    router_token_losses: Any,
    train_split_name: str = "train_even_positions",
    holdout_split_name: str = "holdout_odd_positions",
) -> dict[str, Any]:
    """Evaluate contextual selector supports as an actual residual intervention."""

    import torch

    flat_supports = [
        tuple(int(index) for index in rows[int(row_index)]["support"])
        for row_index in predicted_indices.detach().cpu().tolist()
    ]
    top_k = len(flat_supports[0])
    batch_size, seq_len = int(hidden.shape[0]), int(hidden.shape[1])
    support_indices = torch.zeros(
        batch_size,
        seq_len,
        top_k,
        dtype=torch.long,
        device=hidden.device,
    )
    selected = torch.tensor(
        flat_supports,
        dtype=torch.long,
        device=hidden.device,
    ).view(batch_size, seq_len - 1, top_k)
    support_indices[:, :-1, :] = selected
    support_indices[:, -1:, :] = selected[:, -1:, :]
    logits = base.decode(residual(hidden, support_indices=support_indices))
    token_losses = _token_losses(logits, targets).reshape(-1).detach()
    router_losses = router_token_losses.reshape(-1).detach()
    token_loss_matrix = torch.stack([row["_token_losses"] for row in rows], dim=1)
    oracle_losses = token_loss_matrix.min(dim=1).values.detach()
    router_loss = float(router_losses.mean().item())
    oracle_loss = float(oracle_losses.mean().item())
    router_gap = router_loss - oracle_loss
    all_mask = torch.ones_like(train_mask, dtype=torch.bool)
    splits = [
        _router_support_intervention_split(
            "all",
            all_mask,
            token_losses=token_losses,
            router_losses=router_losses,
            oracle_losses=oracle_losses,
        ),
        _router_support_intervention_split(
            train_split_name,
            train_mask,
            token_losses=token_losses,
            router_losses=router_losses,
            oracle_losses=oracle_losses,
        ),
        _router_support_intervention_split(
            holdout_split_name,
            ~train_mask,
            token_losses=token_losses,
            router_losses=router_losses,
            oracle_losses=oracle_losses,
        ),
    ]
    by_name = {row["split"]: row for row in splits}
    return {
        "summary": {
            "selector": "mlp_contextual_hidden_to_oracle_pair",
            "intervention": "per_token_predicted_support_indices",
            "train_split": train_split_name.replace("_", " "),
            "holdout_split": holdout_split_name.replace("_", " "),
            "router_loss": router_loss,
            "oracle_loss": oracle_loss,
            "router_oracle_gap": router_gap,
            "all": by_name["all"],
            "holdout": by_name[holdout_split_name],
        },
        "splits": splits,
    }


def _contextual_router_support_head(
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    *,
    rows: list[dict[str, Any]],
    train_mask: Any,
    router_token_losses: Any,
    seed: int,
    steps: int = 200,
    train_split_label: str = "even flattened token positions",
    holdout_split_label: str = "odd flattened token positions",
    train_split_name: str = "train_even_positions",
    holdout_split_name: str = "holdout_odd_positions",
) -> dict[str, Any]:
    """Train a contextual support head against fixed-batch support CE losses."""

    import torch
    import torch.nn as nn

    torch.manual_seed(seed + 2027)
    features = _contextual_router_features(hidden)
    token_loss_matrix = torch.stack([row["_token_losses"] for row in rows], dim=1)
    width = max(16, min(128, features.shape[-1]))
    selector = nn.Sequential(
        nn.LayerNorm(features.shape[-1]),
        nn.Linear(features.shape[-1], width),
        nn.GELU(),
        nn.Linear(width, len(rows)),
    ).to(features.device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=0.01, weight_decay=1e-3)
    temperature = 0.25
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        support_probs = torch.softmax(
            selector(features[train_mask]) / temperature,
            dim=-1,
        )
        expected_loss = (
            support_probs * token_loss_matrix[train_mask].detach()
        ).sum(dim=-1).mean()
        expected_loss.backward()
        optimizer.step()

    with torch.no_grad():
        predicted_indices = selector(features).argmax(dim=-1).detach()
    intervention = _router_support_intervention(
        base,
        residual,
        hidden,
        targets,
        vocab_size,
        rows=rows,
        predicted_indices=predicted_indices,
        train_mask=train_mask,
        router_token_losses=router_token_losses,
        train_split_name=train_split_name,
        holdout_split_name=holdout_split_name,
    )
    selected_counts: dict[str, int] = {}
    for index in predicted_indices.detach().cpu().tolist():
        key = str(rows[int(index)]["support_key"])
        selected_counts[key] = selected_counts.get(key, 0) + 1
    intervention["summary"] = {
        **intervention["summary"],
        "selector": "mlp_contextual_support_head_ce_minimizer",
        "training_steps": steps,
        "training_objective": "expected_fixed_batch_support_ce",
        "temperature": temperature,
        "hidden_width": width,
        "train_split": train_split_label,
        "holdout_split": holdout_split_label,
        "selected_support_counts": dict(
            sorted(selected_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
    }
    return intervention


def _router_support_intervention_split(
    split: str,
    mask: Any,
    *,
    token_losses: Any,
    router_losses: Any,
    oracle_losses: Any,
) -> dict[str, Any]:
    positions = int(mask.sum().item())
    if positions == 0:
        return {
            "split": split,
            "positions": 0,
            "router_loss": None,
            "oracle_loss": None,
            "intervention_loss": None,
            "intervention_minus_router_loss": None,
            "intervention_oracle_regret": None,
            "oracle_gap_recovery_fraction": None,
        }
    router_loss = float(router_losses[mask].mean().item())
    oracle_loss = float(oracle_losses[mask].mean().item())
    intervention_loss = float(token_losses[mask].mean().item())
    gap = router_loss - oracle_loss
    recovery = None if abs(gap) <= 1e-12 else (router_loss - intervention_loss) / gap
    return {
        "split": split,
        "positions": positions,
        "router_loss": router_loss,
        "oracle_loss": oracle_loss,
        "intervention_loss": intervention_loss,
        "intervention_minus_router_loss": intervention_loss - router_loss,
        "intervention_oracle_regret": intervention_loss - oracle_loss,
        "oracle_gap_recovery_fraction": recovery,
    }


def _router_target_split(
    split: str,
    mask: Any,
    *,
    targets: Any,
    predicted_indices: Any,
    router_losses: Any,
    oracle_losses: Any,
    selected_losses: Any,
) -> dict[str, Any]:
    selected_count = int(mask.sum().item())
    if selected_count == 0:
        return {
            "split": split,
            "positions": 0,
            "oracle_target_accuracy": 0.0,
            "router_loss": None,
            "oracle_loss": None,
            "selector_loss": None,
            "selector_minus_router_loss": None,
            "selector_oracle_regret": None,
            "oracle_gap_recovery_fraction": None,
        }
    router_loss = float(router_losses[mask].mean().item())
    oracle_loss = float(oracle_losses[mask].mean().item())
    selector_loss = float(selected_losses[mask].mean().item())
    gap = router_loss - oracle_loss
    recovery = None if abs(gap) <= 1e-12 else (router_loss - selector_loss) / gap
    return {
        "split": split,
        "positions": selected_count,
        "oracle_target_accuracy": float(
            (predicted_indices[mask] == targets[mask]).to(dtype=router_losses.dtype).mean().item()
        ),
        "router_loss": router_loss,
        "oracle_loss": oracle_loss,
        "selector_loss": selector_loss,
        "selector_minus_router_loss": selector_loss - router_loss,
        "selector_oracle_regret": selector_loss - oracle_loss,
        "oracle_gap_recovery_fraction": recovery,
    }


def _dominant_router_support(router_support: Any) -> dict[str, Any]:
    counts: dict[tuple[int, ...], int] = {}
    for row in router_support.reshape(-1, router_support.shape[-1]).detach().cpu().tolist():
        support = tuple(sorted(int(index) for index in row))
        counts[support] = counts.get(support, 0) + 1
    support, count = max(counts.items(), key=lambda item: (item[1], tuple(-x for x in item[0])))
    return {"support": support, "count": count}


def _find_support_row(
    rows: list[dict[str, Any]],
    support: tuple[int, ...],
) -> dict[str, Any] | None:
    normalized = tuple(sorted(support))
    for row in rows:
        if tuple(row["support"]) == normalized:
            return row
    return None


def _one_swap_neighbors(
    rows: list[dict[str, Any]],
    support: tuple[int, ...],
) -> list[dict[str, Any]]:
    selected = set(support)
    return [
        row
        for row in rows
        if len(set(row["support"]).intersection(selected)) == len(support) - 1
    ]


def _gap_recovery_fraction(
    *,
    router_loss: float,
    oracle_loss: float,
    candidate_loss: float | None,
) -> float | None:
    if candidate_loss is None:
        return None
    gap = router_loss - oracle_loss
    if abs(gap) <= 1e-12:
        return None
    return (router_loss - candidate_loss) / gap


def _loss_distribution(rows: list[dict[str, Any]]) -> dict[str, float]:
    losses = sorted(float(row["loss"]) for row in rows)
    return {
        "min": losses[0],
        "p25": _quantile(losses, 0.25),
        "median": _quantile(losses, 0.5),
        "p75": _quantile(losses, 0.75),
        "max": losses[-1],
    }


def _quantile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _top_supports(
    rows: list[dict[str, Any]],
    *,
    key: str,
    reverse: bool,
    limit: int = 10,
) -> list[dict[str, Any]]:
    selected = sorted(rows, key=lambda row: float(row[key]), reverse=reverse)[:limit]
    return [
        {
            "support": row["support_key"],
            "loss": float(row["loss"]),
            "gain_from_empty": float(row["gain_from_empty"]),
            "delta_from_router": float(row["delta_from_router"]),
            "pairwise_synergy": float(row.get("pairwise_synergy", 0.0)),
        }
        for row in selected
    ]


def _write_support_losses(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(
        path,
        rows,
        [
            "support_key",
            "support_size",
            "loss",
            "gain_from_empty",
            "delta_from_router",
            "pairwise_synergy",
        ],
    )


def _write_pairwise_synergy(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(
        path,
        sorted(rows, key=lambda row: float(row["pairwise_synergy"]), reverse=True),
        [
            "support_key",
            "loss",
            "gain_from_empty",
            "delta_from_router",
            "pairwise_synergy",
        ],
    )


def _write_router_target_diagnostic(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(
        path,
        rows,
        [
            "split",
            "positions",
            "oracle_target_accuracy",
            "router_loss",
            "oracle_loss",
            "selector_loss",
            "selector_minus_router_loss",
            "selector_oracle_regret",
            "oracle_gap_recovery_fraction",
        ],
    )


def _write_router_support_intervention(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(
        path,
        rows,
        [
            "split",
            "positions",
            "router_loss",
            "oracle_loss",
            "intervention_loss",
            "intervention_minus_router_loss",
            "intervention_oracle_regret",
            "oracle_gap_recovery_fraction",
        ],
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    audit = summary["audit"]
    path.write_text(
        "\n".join(
            [
                f"# {summary['experiment_id']}",
                "",
                "Exhaustive fixed-batch residual support audit.",
                "",
                f"- Config: `{summary['config_path']}`",
                f"- Router loss: `{audit['router_loss']:.8f}`",
        f"- Per-token oracle loss: `{audit['oracle_loss']:.8f}`",
        f"- Best global fixed support: `{audit['best_global_fixed_support']}`",
                f"- Oracle-support regret: `{audit['oracle_support_regret']:.8f}`",
                f"- Dominant router support: `{audit['dominant_router_support']}`",
                f"- Best one-swap support: `{audit['best_one_swap_support']}`",
                "- Router-target holdout gap recovery: "
                f"`{audit['router_oracle_target_diagnostic']['holdout']['oracle_gap_recovery_fraction']}`",
                "- Router-target nonlinear holdout gap recovery: "
                f"`{audit['router_oracle_target_nonlinear_diagnostic']['holdout']['oracle_gap_recovery_fraction']}`",
                "- Router-target contextual holdout gap recovery: "
                f"`{audit['router_oracle_target_contextual_diagnostic']['holdout']['oracle_gap_recovery_fraction']}`",
                "- Contextual support-intervention holdout gap recovery: "
                f"`{audit['contextual_router_support_intervention']['holdout']['oracle_gap_recovery_fraction']}`",
                "- Contextual support-head holdout gap recovery: "
                f"`{audit['contextual_router_support_head']['holdout']['oracle_gap_recovery_fraction']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _support_key(support: tuple[int, ...]) -> str:
    return ",".join(str(index) for index in support)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_support_audit(args.config, args.out)
    print(json.dumps(summary["audit"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
