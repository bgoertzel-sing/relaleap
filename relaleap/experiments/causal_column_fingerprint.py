"""Causal column fingerprint audit for residual-column variants."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.dead_column_probe import _column_values
from relaleap.experiments.dead_column_probe import _parse_weights
from relaleap.experiments.dead_column_probe import _router_load_balance_loss
from relaleap.experiments.dead_column_probe import _variant_name
from relaleap.experiments.run import _load_torch_info
from relaleap.experiments.run import _read_config
from relaleap.experiments.support_audit import _ce_loss
from relaleap.experiments.support_audit import _configured_residual_loss
from relaleap.experiments.support_audit import _score_for_support
from relaleap.experiments.support_audit import _support_key
from relaleap.smoke import ResidualColumns
from relaleap.smoke import TinyCharTransformer
from relaleap.smoke import _build_batch
from relaleap.smoke import _residual_support_audit


DEFAULT_CONFIG = Path(
    "configs/token_larger_support_wide_hep_temporal_clipped_objective_gate.yaml"
)
DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_support_wide_promoted_default_causal_column_fingerprint"
)
DEFAULT_LOAD_BALANCE_WEIGHTS = (0.0, 0.0125)
DEFAULT_MAX_PAIR_ROWS = 8


def run_causal_column_fingerprint(
    config_path: Path,
    out_dir: Path,
    *,
    load_balance_weights: list[float] | None = None,
    include_rank_matched_topk1: bool = False,
    max_pair_rows: int = DEFAULT_MAX_PAIR_ROWS,
) -> dict[str, Any]:
    """Train variants and measure ablate/force/swap intervention fingerprints."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("causal column fingerprint audit requires torch") from exc

    if max_pair_rows < 1:
        raise ValueError("max_pair_rows must be positive")
    weights = (
        list(DEFAULT_LOAD_BALANCE_WEIGHTS)
        if load_balance_weights is None
        else [float(weight) for weight in load_balance_weights]
    )
    if not weights:
        raise ValueError("at least one load-balance weight is required")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("load-balance weights must be non-negative")
    if 0.0 not in weights:
        weights = [0.0, *weights]

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
        raise ValueError("causal column fingerprint audit currently expects top_k: 2")

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
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)

    column_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    per_token_pair_rows: list[dict[str, Any]] = []
    variant_summaries: list[dict[str, Any]] = []
    stability_summaries: list[dict[str, Any]] = []
    functional_churn_summaries: list[dict[str, Any]] = []
    variant_specs = [
        {
            "variant": _variant_name(weight),
            "load_balance_weight": weight,
            "num_columns": num_columns,
            "top_k": top_k,
            "support_router": support_router,
            "contextual_router_hidden_dim": contextual_router_hidden_dim,
        }
        for weight in weights
    ]
    if include_rank_matched_topk1:
        variant_specs.append(
            {
                "variant": "rank_matched_topk1_contextual",
                "load_balance_weight": 0.0,
                "num_columns": num_columns * top_k,
                "top_k": 1,
                "support_router": "contextual_mlp",
                "contextual_router_hidden_dim": contextual_router_hidden_dim,
            }
        )
    for spec in variant_specs:
        residual = _train_residual_variant(
            base=base,
            inputs=inputs,
            targets=targets,
            vocab_size=vocab_size,
            seed=seed,
            hidden_dim=hidden_dim,
            num_columns=int(spec["num_columns"]),
            atoms_per_column=atoms_per_column,
            top_k=int(spec["top_k"]),
            support_router=str(spec["support_router"]),
            contextual_router_hidden_dim=int(spec["contextual_router_hidden_dim"]),
            max_steps=max_steps,
            learning_rate=learning_rate,
            residual_objective=residual_objective,
            training_cfg=training_cfg,
            load_balance_weight=float(spec["load_balance_weight"]),
        )
        residual.eval()
        with torch.no_grad():
            hidden = base.encode(inputs)
            router_hidden, router_support = residual(hidden, return_support=True)
            router_logits = base.decode(router_hidden)
            router_loss = _ce_loss(router_logits, targets, vocab_size)
            empty_loss = _ce_loss(base.decode(hidden), targets, vocab_size)
            values = _column_values(residual)
            scores = residual._score_columns(hidden)
            support_audit = _residual_support_audit(base, residual, inputs)

            column_rows.extend(
                _column_fingerprint_rows(
                    base=base,
                    residual=residual,
                    hidden=hidden,
                    targets=targets,
                    vocab_size=vocab_size,
                    variant=str(spec["variant"]),
                    load_balance_weight=float(spec["load_balance_weight"]),
                    router_hidden=router_hidden,
                    router_logits=router_logits,
                    router_loss=router_loss,
                    router_support=router_support,
                    scores=scores,
                    column_values=values,
                    column_counts=support_audit["column_counts"],
                )
            )
            stability_summaries.append(
                _column_stability_summary(
                    base=base,
                    residual=residual,
                    hidden=hidden,
                    targets=targets,
                    vocab_size=vocab_size,
                    variant=str(spec["variant"]),
                    load_balance_weight=float(spec["load_balance_weight"]),
                    router_hidden=router_hidden,
                    router_logits=router_logits,
                    router_support=router_support,
                )
            )
            functional_churn_summaries.append(
                _functional_churn_summary(
                    base=base,
                    residual=residual,
                    hidden=hidden,
                    variant=str(spec["variant"]),
                    load_balance_weight=float(spec["load_balance_weight"]),
                    router_hidden=router_hidden,
                    router_logits=router_logits,
                    router_support=router_support,
                )
            )
            if int(spec["top_k"]) == 2:
                fixed_rows = [
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
                    for pair in _all_pairs(int(spec["num_columns"]))
                ]
                selected_pairs = _selected_pair_interventions(
                    fixed_rows=fixed_rows,
                    router_support=router_support,
                    max_pair_rows=max_pair_rows,
                )
                pair_rows.extend(
                    _pair_fingerprint_rows(
                        base=base,
                        residual=residual,
                        hidden=hidden,
                        targets=targets,
                        vocab_size=vocab_size,
                        variant=str(spec["variant"]),
                        load_balance_weight=float(spec["load_balance_weight"]),
                        empty_loss=empty_loss,
                        router_hidden=router_hidden,
                        router_logits=router_logits,
                        router_loss=router_loss,
                        column_values=values,
                        selected_pairs=selected_pairs,
                    )
                )
                per_token_pair_rows.extend(
                    _per_token_pair_fingerprint_rows(
                        base=base,
                        residual=residual,
                        hidden=hidden,
                        targets=targets,
                        vocab_size=vocab_size,
                        variant=str(spec["variant"]),
                        load_balance_weight=float(spec["load_balance_weight"]),
                        router_hidden=router_hidden,
                        router_logits=router_logits,
                        router_support=router_support,
                        column_values=values,
                        selected_interventions=selected_pairs,
                    )
                )
            elif int(spec["top_k"]) == 1:
                selected_singletons = _selected_singleton_interventions(
                    hidden=hidden,
                    residual=residual,
                    base=base,
                    targets=targets,
                    vocab_size=vocab_size,
                    router_loss=router_loss,
                    router_support=router_support,
                    max_pair_rows=max_pair_rows,
                )
                pair_rows.extend(
                    _singleton_intervention_rows(
                        base=base,
                        residual=residual,
                        hidden=hidden,
                        targets=targets,
                        vocab_size=vocab_size,
                        variant=str(spec["variant"]),
                        load_balance_weight=float(spec["load_balance_weight"]),
                        empty_loss=empty_loss,
                        router_hidden=router_hidden,
                        router_logits=router_logits,
                        router_loss=router_loss,
                        column_values=values,
                        selected_singletons=selected_singletons,
                    )
                )
                per_token_pair_rows.extend(
                    _per_token_pair_fingerprint_rows(
                        base=base,
                        residual=residual,
                        hidden=hidden,
                        targets=targets,
                        vocab_size=vocab_size,
                        variant=str(spec["variant"]),
                        load_balance_weight=float(spec["load_balance_weight"]),
                        router_hidden=router_hidden,
                        router_logits=router_logits,
                        router_support=router_support,
                        column_values=values,
                        selected_interventions=selected_singletons,
                    )
                )
            variant_summaries.append(
                {
                    "variant": str(spec["variant"]),
                    "load_balance_weight": float(spec["load_balance_weight"]),
                    "num_columns": int(spec["num_columns"]),
                    "top_k": int(spec["top_k"]),
                    "support_router": str(spec["support_router"]),
                    "alpha0_ce_loss": router_loss,
                    "used_columns": support_audit["used_columns"],
                    "dead_columns": support_audit["dead_columns"],
                    "unique_support_sets": support_audit["unique_support_sets"],
                    "load_entropy": _load_entropy(support_audit["column_counts"]),
                    "mean_abs_ablate_loss_delta": _mean_abs(
                        [
                            row["ablate_loss_delta"]
                            for row in column_rows
                            if row["variant"] == str(spec["variant"])
                        ]
                    ),
                    "mean_abs_force_loss_delta": _mean_abs(
                        [
                            row["force_loss_delta"]
                            for row in column_rows
                            if row["variant"] == str(spec["variant"])
                        ]
                    ),
                    "max_abs_ablate_loss_delta": _max_abs(
                        [
                            row["ablate_loss_delta"]
                            for row in column_rows
                            if row["variant"] == str(spec["variant"])
                        ]
                    ),
                    "max_abs_force_loss_delta": _max_abs(
                        [
                            row["force_loss_delta"]
                            for row in column_rows
                            if row["variant"] == str(spec["variant"])
                        ]
                    ),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "column_fingerprints.csv", _COLUMN_FIELDNAMES, column_rows)
    _write_csv(out_dir / "pair_interventions.csv", _PAIR_FIELDNAMES, pair_rows)
    _write_csv(
        out_dir / "per_token_pair_interventions.csv",
        _PER_TOKEN_PAIR_FIELDNAMES,
        per_token_pair_rows,
    )
    summary = {
        "status": "ok",
        "experiment_id": f"{experiment_id}_causal_column_fingerprint",
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
            "atoms_per_column": atoms_per_column,
            "top_k": top_k,
            "support_router": support_router,
            "contextual_router_hidden_dim": contextual_router_hidden_dim,
            "load_balance_weights": weights,
            "include_rank_matched_topk1": include_rank_matched_topk1,
            "column_fingerprint_count": len(column_rows),
            "pair_intervention_count": len(pair_rows),
            "per_token_pair_intervention_count": len(per_token_pair_rows),
            "pair_intervention_counts": _intervention_counts(pair_rows),
            "per_token_pair_intervention_counts": _intervention_counts(
                per_token_pair_rows
            ),
            "variants": variant_summaries,
            "heldout_stability": stability_summaries,
            "functional_churn": functional_churn_summaries,
            "pair_synergy_sign_convention": (
                "pair_synergy = pair_gain - singleton_left_gain - "
                "singleton_right_gain, where gain = empty_loss - intervention_loss; "
                "positive values mean the fixed pair improves loss more than the "
                "sum of its singleton interventions under this loss-space diagnostic."
            ),
        },
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "column_fingerprints_csv": str(out_dir / "column_fingerprints.csv"),
            "pair_interventions_csv": str(out_dir / "pair_interventions.csv"),
            "per_token_pair_interventions_csv": str(
                out_dir / "per_token_pair_interventions.csv"
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


def _train_residual_variant(
    *,
    base: Any,
    inputs: Any,
    targets: Any,
    vocab_size: int,
    seed: int,
    hidden_dim: int,
    num_columns: int,
    atoms_per_column: int,
    top_k: int,
    support_router: str,
    contextual_router_hidden_dim: int,
    max_steps: int,
    learning_rate: float,
    residual_objective: str,
    training_cfg: dict[str, Any],
    load_balance_weight: float,
) -> Any:
    import torch

    torch.manual_seed(seed)
    residual = ResidualColumns(
        hidden_dim=hidden_dim,
        num_columns=num_columns,
        atoms_per_column=atoms_per_column,
        top_k=top_k,
        support_router=support_router,
        contextual_router_hidden_dim=contextual_router_hidden_dim,
    )
    optimizer = torch.optim.AdamW(residual.parameters(), lr=learning_rate)
    residual.train()
    for _ in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        objective_loss = _configured_residual_loss(
            base,
            residual,
            inputs,
            targets,
            vocab_size,
            residual_objective=residual_objective,
            training_cfg=training_cfg,
        )
        hidden = base.encode(inputs)
        load_balance_loss = _router_load_balance_loss(residual, hidden)
        loss = objective_loss + load_balance_weight * load_balance_loss
        loss.backward()
        optimizer.step()
    return residual


def _column_fingerprint_rows(
    *,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    variant: str,
    load_balance_weight: float,
    router_hidden: Any,
    router_logits: Any,
    router_loss: float,
    router_support: Any,
    scores: Any,
    column_values: Any,
    column_counts: list[int],
) -> list[dict[str, Any]]:
    rows = []
    total_slots = sum(int(count) for count in column_counts)
    for column in range(residual.num_columns):
        ablated_hidden = _ablate_column_hidden(
            hidden=hidden,
            residual=residual,
            router_support=router_support,
            column=column,
        )
        ablated_logits = base.decode(ablated_hidden)
        ablated_loss = _ce_loss(ablated_logits, targets, vocab_size)
        forced_hidden = hidden + column_values[column].view(1, 1, -1)
        forced_logits = base.decode(forced_hidden)
        forced_loss = _ce_loss(forced_logits, targets, vocab_size)
        rows.append(
            {
                "variant": variant,
                "load_balance_weight": load_balance_weight,
                "column": column,
                "router_support_count": int(column_counts[column]),
                "router_support_fraction": (
                    int(column_counts[column]) / total_slots if total_slots else 0.0
                ),
                "column_value_norm": float(column_values[column].norm().item()),
                "mean_router_score": float(scores[..., column].mean().item()),
                "ablate_loss": ablated_loss,
                "ablate_loss_delta": ablated_loss - router_loss,
                "ablate_logit_mse": _mse_delta(ablated_logits, router_logits),
                "ablate_residual_stream_l2_delta": _stream_l2_delta(
                    ablated_hidden,
                    router_hidden,
                ),
                "force_loss": forced_loss,
                "force_loss_delta": forced_loss - router_loss,
                "force_logit_mse": _mse_delta(forced_logits, router_logits),
                "force_residual_stream_l2_delta": _stream_l2_delta(
                    forced_hidden,
                    router_hidden,
                ),
            }
        )
    return rows


def _pair_fingerprint_rows(
    *,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    variant: str,
    load_balance_weight: float,
    empty_loss: float,
    router_hidden: Any,
    router_logits: Any,
    router_loss: float,
    column_values: Any,
    selected_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    empty_logits = base.decode(hidden)
    token_class_masks = _token_class_masks(targets)
    strata: list[tuple[str, str, list[int] | None, Any | None]] = [
        ("all", "all", None, None),
        (
            "even",
            "all",
            [index for index in range(int(hidden.shape[1]) - 1) if index % 2 == 0],
            None,
        ),
        (
            "odd",
            "all",
            [index for index in range(int(hidden.shape[1]) - 1) if index % 2 == 1],
            None,
        ),
    ]
    strata.extend(
        ("all", token_class, None, token_mask)
        for token_class, token_mask in token_class_masks
    )
    for selected in selected_pairs:
        support = tuple(int(part) for part in selected["support"])
        left_support = (support[0],)
        right_support = (support[1],)
        fixed_hidden = residual(hidden, support_indices=_fixed_support(hidden, support))
        fixed_logits = base.decode(fixed_hidden)
        fixed_loss = _ce_loss(fixed_logits, targets, vocab_size)
        left_hidden = residual(
            hidden,
            support_indices=_fixed_support(hidden, left_support),
        )
        right_hidden = residual(
            hidden,
            support_indices=_fixed_support(hidden, right_support),
        )
        left_logits = base.decode(left_hidden)
        right_logits = base.decode(right_hidden)
        for position_bin, token_class, position_indices, token_mask in strata:
            if position_indices is None and token_mask is None:
                row_empty_loss = empty_loss
                row_router_loss = router_loss
                row_pair_loss = fixed_loss
                row_left_loss = _ce_loss(left_logits, targets, vocab_size)
                row_right_loss = _ce_loss(right_logits, targets, vocab_size)
                logit_mse = _mse_delta(fixed_logits, router_logits)
                stream_l2 = _stream_l2_delta(fixed_hidden, router_hidden)
            else:
                row_empty_loss = _ce_loss_filtered(
                    empty_logits,
                    targets,
                    vocab_size,
                    position_indices=position_indices,
                    token_mask=token_mask,
                )
                row_router_loss = _ce_loss_filtered(
                    router_logits,
                    targets,
                    vocab_size,
                    position_indices=position_indices,
                    token_mask=token_mask,
                )
                row_pair_loss = _ce_loss_filtered(
                    fixed_logits,
                    targets,
                    vocab_size,
                    position_indices=position_indices,
                    token_mask=token_mask,
                )
                row_left_loss = _ce_loss_filtered(
                    left_logits,
                    targets,
                    vocab_size,
                    position_indices=position_indices,
                    token_mask=token_mask,
                )
                row_right_loss = _ce_loss_filtered(
                    right_logits,
                    targets,
                    vocab_size,
                    position_indices=position_indices,
                    token_mask=token_mask,
                )
                logit_mse = None
                stream_l2 = None
            pair_gain = row_empty_loss - row_pair_loss
            singleton_left_gain = row_empty_loss - row_left_loss
            singleton_right_gain = row_empty_loss - row_right_loss
            rows.append(
                {
                    "variant": variant,
                    "load_balance_weight": load_balance_weight,
                    "intervention": selected["intervention"],
                    "support": _support_key(support),
                    "position_bin": position_bin,
                    "token_class": token_class,
                    "router_support_count": selected["router_support_count"],
                    "empty_loss": row_empty_loss,
                    "router_loss": row_router_loss,
                    "singleton_left_loss": row_left_loss,
                    "singleton_right_loss": row_right_loss,
                    "pair_loss": row_pair_loss,
                    "singleton_left_gain": singleton_left_gain,
                    "singleton_right_gain": singleton_right_gain,
                    "pair_gain": pair_gain,
                    "pair_synergy": (
                        pair_gain - singleton_left_gain - singleton_right_gain
                    ),
                    "fixed_support_loss": row_pair_loss,
                    "fixed_support_loss_delta": row_pair_loss - row_router_loss,
                    "fixed_support_logit_mse": logit_mse,
                    "fixed_support_residual_stream_l2_delta": stream_l2,
                    "pair_value_cosine": _pair_value_cosine(column_values, support),
                }
            )
    return rows


def _singleton_intervention_rows(
    *,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    variant: str,
    load_balance_weight: float,
    empty_loss: float,
    router_hidden: Any,
    router_logits: Any,
    router_loss: float,
    column_values: Any,
    selected_singletons: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    empty_logits = base.decode(hidden)
    token_class_masks = _token_class_masks(targets)
    strata: list[tuple[str, str, list[int] | None, Any | None]] = [
        ("all", "all", None, None),
        (
            "even",
            "all",
            [index for index in range(int(hidden.shape[1]) - 1) if index % 2 == 0],
            None,
        ),
        (
            "odd",
            "all",
            [index for index in range(int(hidden.shape[1]) - 1) if index % 2 == 1],
            None,
        ),
    ]
    strata.extend(
        ("all", token_class, None, token_mask)
        for token_class, token_mask in token_class_masks
    )
    for selected in selected_singletons:
        support = (int(selected["support"][0]),)
        fixed_hidden = residual(hidden, support_indices=_fixed_support(hidden, support))
        fixed_logits = base.decode(fixed_hidden)
        fixed_loss = _ce_loss(fixed_logits, targets, vocab_size)
        for position_bin, token_class, position_indices, token_mask in strata:
            if position_indices is None and token_mask is None:
                row_empty_loss = empty_loss
                row_router_loss = router_loss
                row_singleton_loss = fixed_loss
                logit_mse = _mse_delta(fixed_logits, router_logits)
                stream_l2 = _stream_l2_delta(fixed_hidden, router_hidden)
            else:
                row_empty_loss = _ce_loss_filtered(
                    empty_logits,
                    targets,
                    vocab_size,
                    position_indices=position_indices,
                    token_mask=token_mask,
                )
                row_router_loss = _ce_loss_filtered(
                    router_logits,
                    targets,
                    vocab_size,
                    position_indices=position_indices,
                    token_mask=token_mask,
                )
                row_singleton_loss = _ce_loss_filtered(
                    fixed_logits,
                    targets,
                    vocab_size,
                    position_indices=position_indices,
                    token_mask=token_mask,
                )
                logit_mse = None
                stream_l2 = None
            singleton_gain = row_empty_loss - row_singleton_loss
            rows.append(
                {
                    "variant": variant,
                    "load_balance_weight": load_balance_weight,
                    "intervention": selected["intervention"],
                    "support": _support_key(support),
                    "position_bin": position_bin,
                    "token_class": token_class,
                    "router_support_count": selected["router_support_count"],
                    "empty_loss": row_empty_loss,
                    "router_loss": row_router_loss,
                    "singleton_left_loss": row_singleton_loss,
                    "singleton_right_loss": None,
                    "pair_loss": None,
                    "singleton_left_gain": singleton_gain,
                    "singleton_right_gain": None,
                    "pair_gain": None,
                    "pair_synergy": None,
                    "fixed_support_loss": row_singleton_loss,
                    "fixed_support_loss_delta": row_singleton_loss - row_router_loss,
                    "fixed_support_logit_mse": logit_mse,
                    "fixed_support_residual_stream_l2_delta": stream_l2,
                    "pair_value_cosine": _singleton_value_norm(column_values, support),
                }
            )
    return rows


def _per_token_pair_fingerprint_rows(
    *,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    variant: str,
    load_balance_weight: float,
    router_hidden: Any,
    router_logits: Any,
    router_support: Any,
    column_values: Any,
    selected_interventions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    import torch

    empty_logits = base.decode(hidden)
    empty_token_loss = _token_ce_losses(empty_logits, targets, vocab_size)
    router_token_loss = _token_ce_losses(router_logits, targets, vocab_size)
    target_tokens = targets[:, :-1].detach()
    token_classes = _token_class_lookup(targets)
    rows: list[dict[str, Any]] = []
    for selected in selected_interventions:
        support = tuple(int(part) for part in selected["support"])
        fixed_hidden = residual(hidden, support_indices=_fixed_support(hidden, support))
        fixed_logits = base.decode(fixed_hidden)
        fixed_token_loss = _token_ce_losses(fixed_logits, targets, vocab_size)
        left_token_loss = None
        right_token_loss = None
        if len(support) == 2:
            left_hidden = residual(
                hidden,
                support_indices=_fixed_support(hidden, (support[0],)),
            )
            right_hidden = residual(
                hidden,
                support_indices=_fixed_support(hidden, (support[1],)),
            )
            left_token_loss = _token_ce_losses(
                base.decode(left_hidden),
                targets,
                vocab_size,
            )
            right_token_loss = _token_ce_losses(
                base.decode(right_hidden),
                targets,
                vocab_size,
            )
        fixed_logit_mse = (fixed_logits[:, :-1, :] - router_logits[:, :-1, :]).pow(2).mean(
            dim=-1
        )
        fixed_stream_l2 = (fixed_hidden[:, :-1, :] - router_hidden[:, :-1, :]).norm(
            dim=-1
        )
        fixed_residual_norm = (fixed_hidden[:, :-1, :] - hidden[:, :-1, :]).norm(dim=-1)
        residual_gain = empty_token_loss - fixed_token_loss
        norm_bins = _tensor_quantile_bins(fixed_residual_norm)
        gain_bins = _tensor_quantile_bins(residual_gain)
        active_rank = _support_active_rank(column_values, support)
        support_key = _support_key(support)
        sorted_support = torch.tensor(
            sorted(support),
            dtype=torch.long,
            device=router_support.device,
        ).view(1, 1, len(support))
        router_support_matches_fixed = (
            torch.sort(router_support[:, :-1, :], dim=-1).values == sorted_support
        ).all(dim=-1)
        for batch_index in range(int(hidden.shape[0])):
            for position_index in range(int(hidden.shape[1]) - 1):
                token_id = int(target_tokens[batch_index, position_index].item())
                empty_loss = float(
                    empty_token_loss[batch_index, position_index].detach().item()
                )
                router_loss = float(
                    router_token_loss[batch_index, position_index].detach().item()
                )
                fixed_loss = float(
                    fixed_token_loss[batch_index, position_index].detach().item()
                )
                if left_token_loss is not None and right_token_loss is not None:
                    singleton_left_loss = float(
                        left_token_loss[batch_index, position_index].detach().item()
                    )
                    singleton_right_loss = float(
                        right_token_loss[batch_index, position_index].detach().item()
                    )
                    singleton_left_gain = empty_loss - singleton_left_loss
                    singleton_right_gain = empty_loss - singleton_right_loss
                    pair_gain = empty_loss - fixed_loss
                    pair_synergy = (
                        pair_gain - singleton_left_gain - singleton_right_gain
                    )
                else:
                    singleton_left_loss = fixed_loss
                    singleton_right_loss = None
                    singleton_left_gain = empty_loss - fixed_loss
                    singleton_right_gain = None
                    pair_gain = None
                    pair_synergy = None
                rows.append(
                    {
                        "variant": variant,
                        "load_balance_weight": load_balance_weight,
                        "intervention": selected["intervention"],
                        "support": support_key,
                        "batch_index": batch_index,
                        "position_index": position_index,
                        "token_index": batch_index * (int(hidden.shape[1]) - 1)
                        + position_index,
                        "target_token": token_id,
                        "position_bin": "even"
                        if position_index % 2 == 0
                        else "odd",
                        "token_class": token_classes.get(token_id, "other_target"),
                        "router_support_count": selected["router_support_count"],
                        "router_support_matches_fixed": bool(
                            router_support_matches_fixed[
                                batch_index,
                                position_index,
                            ].item()
                        ),
                        "empty_loss": empty_loss,
                        "router_loss": router_loss,
                        "singleton_left_loss": singleton_left_loss,
                        "singleton_right_loss": singleton_right_loss,
                        "singleton_left_gain": singleton_left_gain,
                        "singleton_right_gain": singleton_right_gain,
                        "pair_gain": pair_gain,
                        "pair_synergy": pair_synergy,
                        "fixed_support_loss": fixed_loss,
                        "fixed_support_loss_delta": fixed_loss - router_loss,
                        "fixed_support_logit_mse": float(
                            fixed_logit_mse[batch_index, position_index].detach().item()
                        ),
                        "fixed_support_residual_stream_l2_delta": float(
                            fixed_stream_l2[batch_index, position_index].detach().item()
                        ),
                        "residual_norm": float(
                            fixed_residual_norm[
                                batch_index,
                                position_index,
                            ]
                            .detach()
                            .item()
                        ),
                        "residual_norm_bin": norm_bins[batch_index][position_index],
                        "residual_gain": empty_loss - fixed_loss,
                        "residual_gain_bin": gain_bins[batch_index][position_index],
                        "active_rank_proxy": active_rank,
                    }
                )
    return rows


def _token_ce_losses(logits: Any, targets: Any, vocab_size: int) -> Any:
    import torch.nn.functional as F

    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab_size),
        targets[:, :-1].reshape(-1),
        reduction="none",
    ).reshape(logits.shape[0], logits.shape[1] - 1)


def _token_class_lookup(targets: Any) -> dict[int, str]:
    lookup: dict[int, str] = {}
    target_tokens = targets[:, :-1]
    for token_class, mask in _token_class_masks(targets):
        token_ids = sorted(
            {
                int(token.item())
                for token in target_tokens[
                    mask.to(device=target_tokens.device)
                ]
                .detach()
                .cpu()
            }
        )
        for token_id in token_ids:
            lookup[token_id] = token_class
    return lookup


def _tensor_quantile_bins(values: Any) -> list[list[str]]:
    import torch

    flat = values.detach().reshape(-1)
    if int(flat.numel()) == 0:
        return []
    low = torch.quantile(flat, 1.0 / 3.0)
    high = torch.quantile(flat, 2.0 / 3.0)
    labels = []
    for row in values.detach().cpu():
        label_row = []
        for value in row:
            if float(value.item()) <= float(low.item()):
                label_row.append("low")
            elif float(value.item()) <= float(high.item()):
                label_row.append("mid")
            else:
                label_row.append("high")
        labels.append(label_row)
    return labels


def _support_active_rank(column_values: Any, support: tuple[int, ...]) -> int:
    import torch

    if not support:
        return 0
    matrix = column_values[list(support)].detach()
    return int(torch.linalg.matrix_rank(matrix).item())


def _column_stability_summary(
    *,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    variant: str,
    load_balance_weight: float,
    router_hidden: Any,
    router_logits: Any,
    router_support: Any,
) -> dict[str, Any]:
    del router_hidden

    even_positions = [index for index in range(int(hidden.shape[1]) - 1) if index % 2 == 0]
    odd_positions = [index for index in range(int(hidden.shape[1]) - 1) if index % 2 == 1]
    first_batch = list(range(max(1, int(hidden.shape[0]) // 2)))
    second_batch = list(range(max(1, int(hidden.shape[0]) // 2), int(hidden.shape[0])))
    if not second_batch:
        second_batch = first_batch
    position_left = _ablate_delta_vector(
        base=base,
        residual=residual,
        hidden=hidden,
        targets=targets,
        vocab_size=vocab_size,
        router_logits=router_logits,
        router_support=router_support,
        position_indices=even_positions,
    )
    position_right = _ablate_delta_vector(
        base=base,
        residual=residual,
        hidden=hidden,
        targets=targets,
        vocab_size=vocab_size,
        router_logits=router_logits,
        router_support=router_support,
        position_indices=odd_positions,
    )
    batch_left = _ablate_delta_vector(
        base=base,
        residual=residual,
        hidden=hidden,
        targets=targets,
        vocab_size=vocab_size,
        router_logits=router_logits,
        router_support=router_support,
        batch_indices=first_batch,
    )
    batch_right = _ablate_delta_vector(
        base=base,
        residual=residual,
        hidden=hidden,
        targets=targets,
        vocab_size=vocab_size,
        router_logits=router_logits,
        router_support=router_support,
        batch_indices=second_batch,
    )
    return {
        "variant": variant,
        "load_balance_weight": load_balance_weight,
        "position_split": "even_vs_odd",
        "position_ablate_delta_correlation": _pearson(position_left, position_right),
        "position_mean_abs_ablate_delta_difference": _mean_abs_difference(
            position_left,
            position_right,
        ),
        "batch_split": "first_half_vs_second_half",
        "batch_ablate_delta_correlation": _pearson(batch_left, batch_right),
        "batch_mean_abs_ablate_delta_difference": _mean_abs_difference(
            batch_left,
            batch_right,
        ),
    }


def _functional_churn_summary(
    *,
    base: Any,
    residual: Any,
    hidden: Any,
    variant: str,
    load_balance_weight: float,
    router_hidden: Any,
    router_logits: Any,
    router_support: Any,
) -> dict[str, Any]:
    import torch

    previous_support = torch.cat(
        [router_support[:, :1, :], router_support[:, :-1, :]],
        dim=1,
    )
    previous_hidden = residual(hidden, support_indices=previous_support)
    previous_logits = base.decode(previous_hidden)
    sorted_support = torch.sort(router_support, dim=-1).values
    sorted_previous_support = torch.sort(previous_support, dim=-1).values
    changed = (
        sorted_support[:, 1:, :] != sorted_previous_support[:, 1:, :]
    ).any(dim=-1)
    retained = ~changed
    logit_mse = (previous_logits - router_logits).pow(2).mean(dim=-1)[:, 1:]
    stream_l2 = (previous_hidden - router_hidden).norm(dim=-1)[:, 1:]

    changed_logit = _masked_mean(logit_mse, changed)
    retained_logit = _masked_mean(logit_mse, retained)
    changed_stream = _masked_mean(stream_l2, changed)
    retained_stream = _masked_mean(stream_l2, retained)
    total_positions = int(changed.numel())
    changed_count = int(changed.sum().item())
    return {
        "variant": variant,
        "load_balance_weight": load_balance_weight,
        "adjacent_support_identity_churn_fraction": (
            changed_count / total_positions if total_positions else 0.0
        ),
        "adjacent_support_changed_count": changed_count,
        "adjacent_support_position_count": total_positions,
        "previous_support_changed_logit_mse_mean": changed_logit,
        "previous_support_retained_logit_mse_mean": retained_logit,
        "previous_support_changed_residual_l2_mean": changed_stream,
        "previous_support_retained_residual_l2_mean": retained_stream,
        "previous_support_changed_to_retained_logit_mse_ratio": _safe_ratio(
            changed_logit,
            retained_logit,
        ),
        "previous_support_changed_to_retained_residual_l2_ratio": _safe_ratio(
            changed_stream,
            retained_stream,
        ),
    }


def _masked_mean(values: Any, mask: Any) -> float | None:
    selected = values[mask]
    if int(selected.numel()) == 0:
        return None
    return float(selected.mean().detach().item())


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(float(denominator)) <= 1e-12:
        return None
    return float(numerator) / float(denominator)


def _ablate_delta_vector(
    *,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    router_logits: Any,
    router_support: Any,
    batch_indices: list[int] | None = None,
    position_indices: list[int] | None = None,
) -> list[float]:
    router_loss = _ce_loss_subset(
        router_logits,
        targets,
        vocab_size,
        batch_indices=batch_indices,
        position_indices=position_indices,
    )
    del vocab_size
    values = []
    for column in range(residual.num_columns):
        ablated_hidden = _ablate_column_hidden(
            hidden=hidden,
            residual=residual,
            router_support=router_support,
            column=column,
        )
        ablated_logits = base.decode(ablated_hidden)
        values.append(
            _ce_loss_subset(
                ablated_logits,
                targets,
                int(ablated_logits.shape[-1]),
                batch_indices=batch_indices,
                position_indices=position_indices,
            )
            - router_loss
        )
    return values


def _ce_loss_subset(
    logits: Any,
    targets: Any,
    vocab_size: int,
    *,
    batch_indices: list[int] | None = None,
    position_indices: list[int] | None = None,
) -> float:
    import torch
    import torch.nn.functional as F

    losses = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab_size),
        targets[:, :-1].reshape(-1),
        reduction="none",
    ).reshape(logits.shape[0], logits.shape[1] - 1)
    if batch_indices is not None:
        batch = torch.tensor(batch_indices, dtype=torch.long, device=logits.device)
        losses = losses.index_select(0, batch)
    if position_indices is not None:
        positions = [
            index
            for index in position_indices
            if 0 <= int(index) < int(logits.shape[1]) - 1
        ]
        if not positions:
            positions = list(range(int(logits.shape[1]) - 1))
        position = torch.tensor(positions, dtype=torch.long, device=logits.device)
        losses = losses.index_select(1, position)
    return float(losses.mean().detach().item())


def _ce_loss_filtered(
    logits: Any,
    targets: Any,
    vocab_size: int,
    *,
    position_indices: list[int] | None = None,
    token_mask: Any | None = None,
) -> float:
    import torch
    import torch.nn.functional as F

    losses = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab_size),
        targets[:, :-1].reshape(-1),
        reduction="none",
    ).reshape(logits.shape[0], logits.shape[1] - 1)
    mask = torch.ones_like(losses, dtype=torch.bool)
    if position_indices is not None:
        position_mask = torch.zeros_like(mask)
        positions = [
            int(index)
            for index in position_indices
            if 0 <= int(index) < int(logits.shape[1]) - 1
        ]
        if positions:
            position = torch.tensor(positions, dtype=torch.long, device=logits.device)
            position_mask.index_fill_(1, position, True)
            mask = mask & position_mask
    if token_mask is not None:
        mask = mask & token_mask.to(device=logits.device, dtype=torch.bool)
    selected = losses[mask]
    if selected.numel() == 0:
        return _ce_loss_subset(
            logits,
            targets,
            vocab_size,
            position_indices=position_indices,
        )
    return float(selected.mean().detach().item())


def _token_class_masks(targets: Any) -> list[tuple[str, Any]]:
    import torch

    target_tokens = targets[:, :-1].detach()
    unique, counts = torch.unique(target_tokens.reshape(-1), return_counts=True)
    if int(unique.numel()) < 2:
        return []
    ordered = sorted(
        (
            (int(count.item()), int(token.item()))
            for token, count in zip(unique.cpu(), counts.cpu())
        ),
        key=lambda item: (-item[0], item[1]),
    )
    split = max(1, len(ordered) // 2)
    classes = [
        ("common_target", [token for _, token in ordered[:split]]),
        ("rare_target", [token for _, token in ordered[split:]]),
    ]
    masks = []
    for name, token_ids in classes:
        if not token_ids:
            continue
        mask = torch.zeros_like(target_tokens, dtype=torch.bool)
        for token_id in token_ids:
            mask = mask | (target_tokens == int(token_id))
        if bool(mask.any().item()):
            masks.append((name, mask))
    return masks


def _ablate_column_hidden(
    *,
    hidden: Any,
    residual: Any,
    router_support: Any,
    column: int,
) -> Any:
    import torch
    import torch.nn.functional as F

    scores = residual._score_columns(hidden) + residual.score_tie_breaker.to(
        device=hidden.device,
        dtype=hidden.dtype,
    )
    top_values = scores.gather(dim=-1, index=router_support)
    weights = F.softmax(top_values, dim=-1)
    keep = (router_support != column).to(dtype=weights.dtype)
    weights = weights * keep
    atom_weights = F.softmax(residual.atom_logits, dim=-1)
    column_values = torch.einsum("ca,cah->ch", atom_weights, residual.atom_values)
    selected_values = column_values[router_support]
    residual_delta = torch.einsum("bsk,bskh->bsh", weights, selected_values)
    return hidden + residual_delta


def _fixed_support(hidden: Any, support: tuple[int, ...]) -> Any:
    import torch

    return torch.tensor(
        support,
        dtype=torch.long,
        device=hidden.device,
    ).view(1, 1, len(support)).expand(hidden.shape[0], hidden.shape[1], len(support))


def _all_pairs(num_columns: int) -> list[tuple[int, int]]:
    return [
        (left, right)
        for left in range(num_columns)
        for right in range(left + 1, num_columns)
    ]


def _selected_pair_interventions(
    *,
    fixed_rows: list[dict[str, Any]],
    router_support: Any,
    max_pair_rows: int,
) -> list[dict[str, Any]]:
    support_counts = _router_support_counts(router_support)
    rows_by_key = {str(row["support_key"]): row for row in fixed_rows}
    selected: dict[str, dict[str, Any]] = {}

    for key, count in list(support_counts.items())[:max_pair_rows]:
        if key in rows_by_key:
            selected[f"dominant_router_{len(selected) + 1}"] = {
                "intervention": "fixed_dominant_router_support",
                "support": rows_by_key[key]["support"],
                "router_support_count": count,
            }
    for row in sorted(fixed_rows, key=lambda item: float(item["loss"]))[:max_pair_rows]:
        key = str(row["support_key"])
        selected.setdefault(
            f"best_fixed_{key}",
            {
                "intervention": "fixed_best_support_swap",
                "support": row["support"],
                "router_support_count": support_counts.get(key, 0),
            },
        )
    anchors = list(selected.values())
    for row in _matched_control_pair_rows(
        fixed_rows=fixed_rows,
        support_counts=support_counts,
        anchors=anchors,
        selected=selected,
        max_pair_rows=max_pair_rows,
        intervention="fixed_support_frequency_matched_control",
        prefer_nonrouter=False,
        score_fn=lambda candidate, anchor: (
            abs(
                support_counts.get(str(candidate["support_key"]), 0)
                - int(anchor["router_support_count"])
            ),
            abs(float(candidate["loss"]) - _anchor_loss(anchor, rows_by_key)),
            str(candidate["support_key"]),
        ),
    ):
        key = str(row["support_key"])
        selected[f"support_frequency_control_{key}"] = {
            "intervention": "fixed_support_frequency_matched_control",
            "support": row["support"],
            "router_support_count": support_counts.get(key, 0),
        }
    for row in _matched_control_pair_rows(
        fixed_rows=fixed_rows,
        support_counts=support_counts,
        anchors=anchors,
        selected=selected,
        max_pair_rows=max_pair_rows,
        intervention="fixed_loss_matched_nonrouter_control",
        prefer_nonrouter=True,
        score_fn=lambda candidate, anchor: (
            abs(float(candidate["loss"]) - _anchor_loss(anchor, rows_by_key)),
            abs(
                support_counts.get(str(candidate["support_key"]), 0)
                - int(anchor["router_support_count"])
            ),
            str(candidate["support_key"]),
        ),
    ):
        key = str(row["support_key"])
        selected[f"loss_matched_control_{key}"] = {
            "intervention": "fixed_loss_matched_nonrouter_control",
            "support": row["support"],
            "router_support_count": support_counts.get(key, 0),
        }
    return list(selected.values())[: max_pair_rows * 10]


def _matched_control_pair_rows(
    *,
    fixed_rows: list[dict[str, Any]],
    support_counts: dict[str, int],
    anchors: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    max_pair_rows: int,
    intervention: str,
    prefer_nonrouter: bool,
    score_fn: Any,
) -> list[dict[str, Any]]:
    del intervention

    selected_supports = {
        _support_key(tuple(row["support"]))
        for row in selected.values()
        if not str(row.get("intervention", "")).endswith("_control")
    }
    control_rows: list[dict[str, Any]] = []
    control_keys: set[str] = set()
    candidates = [
        row for row in fixed_rows if str(row["support_key"]) not in selected_supports
    ]
    if prefer_nonrouter:
        nonrouter_candidates = [
            row
            for row in candidates
            if support_counts.get(str(row["support_key"]), 0) == 0
        ]
        if nonrouter_candidates:
            candidates = nonrouter_candidates
    target_count = max_pair_rows * 4
    anchor_index = 0
    while anchors and len(control_rows) < target_count:
        anchor = anchors[anchor_index % len(anchors)]
        ranked = sorted(candidates, key=lambda row: score_fn(row, anchor))
        for row in ranked:
            key = str(row["support_key"])
            if key in control_keys:
                continue
            control_keys.add(key)
            control_rows.append(row)
            break
        anchor_index += 1
        if anchor_index > len(anchors) + len(candidates):
            break
    return control_rows


def _anchor_loss(anchor: dict[str, Any], rows_by_key: dict[str, dict[str, Any]]) -> float:
    key = _support_key(tuple(int(part) for part in anchor["support"]))
    return float(rows_by_key[key]["loss"])


def _selected_singleton_interventions(
    *,
    hidden: Any,
    residual: Any,
    base: Any,
    targets: Any,
    vocab_size: int,
    router_loss: float,
    router_support: Any,
    max_pair_rows: int,
) -> list[dict[str, Any]]:
    support_counts = _router_support_counts(router_support)
    selected: dict[str, dict[str, Any]] = {}
    for key, count in list(support_counts.items())[:max_pair_rows]:
        support = tuple(int(part) for part in key.split(",") if part != "")
        if len(support) == 1:
            selected[f"dominant_router_{len(selected) + 1}"] = {
                "intervention": "fixed_dominant_router_singleton",
                "support": support,
                "router_support_count": count,
            }

    fixed_rows = []
    for column in range(residual.num_columns):
        support = (column,)
        fixed_hidden = residual(hidden, support_indices=_fixed_support(hidden, support))
        fixed_loss = _ce_loss(base.decode(fixed_hidden), targets, vocab_size)
        fixed_rows.append({"support": support, "loss_delta": fixed_loss - router_loss})
    for row in sorted(fixed_rows, key=lambda item: float(item["loss_delta"]))[
        :max_pair_rows
    ]:
        key = _support_key(row["support"])
        selected.setdefault(
            f"best_fixed_{key}",
            {
                "intervention": "fixed_best_singleton_swap",
                "support": row["support"],
                "router_support_count": support_counts.get(key, 0),
            },
        )
    return list(selected.values())[: max_pair_rows * 2]


def _router_support_counts(router_support: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pair in router_support.reshape(-1, router_support.shape[-1]).detach().cpu().tolist():
        key = _support_key(tuple(sorted(int(part) for part in pair)))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _pair_value_cosine(column_values: Any, support: tuple[int, ...]) -> float | None:
    if len(support) != 2:
        return None
    import torch.nn.functional as F

    left, right = support
    return float(
        F.cosine_similarity(
            column_values[left].view(1, -1),
            column_values[right].view(1, -1),
            dim=-1,
            eps=1e-12,
        ).item()
    )


def _singleton_value_norm(column_values: Any, support: tuple[int, ...]) -> float | None:
    if len(support) != 1:
        return None
    return float(column_values[support[0]].norm().item())


def _mse_delta(left: Any, right: Any) -> float:
    import torch
    import torch.nn.functional as F

    return float(F.mse_loss(left - right, torch.zeros_like(left)).item())


def _stream_l2_delta(left: Any, right: Any) -> float:
    return float((left - right).norm(dim=-1).mean().item())


def _load_entropy(column_counts: list[int]) -> float:
    total = sum(int(count) for count in column_counts)
    if total == 0:
        return 0.0
    return -sum(
        fraction * math.log(fraction)
        for fraction in (int(count) / total for count in column_counts)
        if fraction > 0.0
    )


def _mean_abs(values: list[float]) -> float | None:
    return None if not values else sum(abs(float(value)) for value in values) / len(values)


def _max_abs(values: list[float]) -> float | None:
    return None if not values else max(abs(float(value)) for value in values)


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered)
    )
    left_norm = math.sqrt(sum(value * value for value in left_centered))
    right_norm = math.sqrt(sum(value * value for value in right_centered))
    denominator = left_norm * right_norm
    if denominator <= 1e-12:
        return None
    return numerator / denominator


def _mean_abs_difference(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    return sum(
        abs(float(left_value) - float(right_value))
        for left_value, right_value in zip(left, right)
    ) / len(left)


def _intervention_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        intervention = str(row.get("intervention", ""))
        counts[intervention] = counts.get(intervention, 0) + 1
    return dict(sorted(counts.items()))


_COLUMN_FIELDNAMES = [
    "variant",
    "load_balance_weight",
    "column",
    "router_support_count",
    "router_support_fraction",
    "column_value_norm",
    "mean_router_score",
    "ablate_loss",
    "ablate_loss_delta",
    "ablate_logit_mse",
    "ablate_residual_stream_l2_delta",
    "force_loss",
    "force_loss_delta",
    "force_logit_mse",
    "force_residual_stream_l2_delta",
]

_PAIR_FIELDNAMES = [
    "variant",
    "load_balance_weight",
    "intervention",
    "support",
    "position_bin",
    "token_class",
    "router_support_count",
    "empty_loss",
    "router_loss",
    "singleton_left_loss",
    "singleton_right_loss",
    "pair_loss",
    "singleton_left_gain",
    "singleton_right_gain",
    "pair_gain",
    "pair_synergy",
    "fixed_support_loss",
    "fixed_support_loss_delta",
    "fixed_support_logit_mse",
    "fixed_support_residual_stream_l2_delta",
    "pair_value_cosine",
]

_PER_TOKEN_PAIR_FIELDNAMES = [
    "variant",
    "load_balance_weight",
    "intervention",
    "support",
    "batch_index",
    "position_index",
    "token_index",
    "target_token",
    "position_bin",
    "token_class",
    "router_support_count",
    "router_support_matches_fixed",
    "empty_loss",
    "router_loss",
    "singleton_left_loss",
    "singleton_right_loss",
    "singleton_left_gain",
    "singleton_right_gain",
    "pair_gain",
    "pair_synergy",
    "fixed_support_loss",
    "fixed_support_loss_delta",
    "fixed_support_logit_mse",
    "fixed_support_residual_stream_l2_delta",
    "residual_norm",
    "residual_norm_bin",
    "residual_gain",
    "residual_gain_bin",
    "active_rank_proxy",
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    audit = summary["audit"]
    lines = [
        "# Causal Column Fingerprint Audit",
        "",
        f"- Experiment: `{summary['experiment_id']}`",
        f"- Config: `{summary['config_path']}`",
        f"- Status: `{summary['status']}`",
        f"- Variants: `{', '.join(row['variant'] for row in audit['variants'])}`",
        f"- Column fingerprint rows: `{audit['column_fingerprint_count']}`",
        f"- Pair intervention rows: `{audit['pair_intervention_count']}`",
        f"- Per-token pair intervention rows: `{audit['per_token_pair_intervention_count']}`",
        f"- Pair synergy sign convention: {audit['pair_synergy_sign_convention']}",
        "",
        "## Variant Summary",
        "",
    ]
    for row in audit["variants"]:
        lines.append(
            "- "
            f"`{row['variant']}`: alpha-0 CE `{row['alpha0_ce_loss']}`, "
            f"used columns `{row['used_columns']}`, "
            f"unique support sets `{row['unique_support_sets']}`, "
            f"load entropy `{row['load_entropy']}`, "
            f"mean abs ablate delta `{row['mean_abs_ablate_loss_delta']}`, "
            f"mean abs force delta `{row['mean_abs_force_loss_delta']}`"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--load-balance-weights",
        type=_parse_weights,
        default=list(DEFAULT_LOAD_BALANCE_WEIGHTS),
        help="Comma-separated router load-balance weights.",
    )
    parser.add_argument(
        "--include-rank-matched-topk1",
        action="store_true",
        help="Also train a contextual top-k-1 fingerprint variant with num_columns * top_k columns.",
    )
    parser.add_argument("--max-pair-rows", type=int, default=DEFAULT_MAX_PAIR_ROWS)
    args = parser.parse_args(argv)
    summary = run_causal_column_fingerprint(
        args.config,
        args.out,
        load_balance_weights=args.load_balance_weights,
        include_rank_matched_topk1=args.include_rank_matched_topk1,
        max_pair_rows=args.max_pair_rows,
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
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
