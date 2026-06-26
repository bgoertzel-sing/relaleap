"""Contextual-router shortcut ablation for promoted top-k-2 support routing."""

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
DEFAULT_OUT_DIR = Path("results/audits/token_larger_contextual_router_shortcut_ablation")


def run_contextual_router_shortcut_ablation(
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    probe_steps: int = 200,
) -> dict[str, Any]:
    """Train feature-ablation support heads against fixed support CE losses."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("contextual-router shortcut ablation requires torch") from exc

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
        raise ValueError("contextual-router shortcut ablation expects top_k: 2")
    if support_router != "contextual_mlp":
        raise ValueError(
            "contextual-router shortcut ablation expects support_router: contextual_mlp"
        )
    if probe_steps < 1:
        raise ValueError("probe_steps must be positive")

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
        ordinary_hidden = residual(hidden)
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

    token_loss_matrix = torch.stack([row["_token_losses"] for row in pair_rows], dim=1)
    oracle_losses, oracle_indices = token_loss_matrix.min(dim=1)
    train_mask = _even_flat_position_mask(hidden)
    variant_rows = []
    variant_summaries: dict[str, Any] = {}
    for variant_name, features, feature_names in _feature_variants(hidden):
        result = _train_feature_ablation_support_head(
            base,
            residual,
            hidden,
            targets,
            vocab_size,
            rows=pair_rows,
            features=features,
            train_mask=train_mask,
            token_loss_matrix=token_loss_matrix,
            router_token_losses=router_token_losses,
            oracle_indices=oracle_indices,
            seed=seed,
            variant_name=variant_name,
            feature_names=feature_names,
            steps=probe_steps,
        )
        variant_rows.extend(result["rows"])
        variant_summaries[variant_name] = result["summary"]

    selected = _select_variant(variant_summaries)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_variant_metrics(out_dir / "variant_metrics.csv", variant_rows)
    summary = {
        "status": "ok",
        "decision": selected["decision"],
        "experiment_id": f"{experiment_id}_contextual_router_shortcut_ablation",
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        **_load_torch_info(),
        "ablation": {
            "dataset": dataset,
            "seq_len": seq_len,
            "batch_size": int(inputs.shape[0]),
            "vocab_size": vocab_size,
            "training_steps": max_steps,
            "probe_steps": probe_steps,
            "residual_objective": residual_objective,
            "num_columns": num_columns,
            "top_k": top_k,
            "support_router": support_router,
            "contextual_router_hidden_dim": contextual_router_hidden_dim,
            "support_set_count": len(pair_rows),
            "singleton_count": len(singleton_rows),
            "router_loss": router_loss,
            "empty_loss": empty_loss,
            "oracle_loss": float(oracle_losses.mean().item()),
            "router_oracle_gap": float(
                router_token_losses.reshape(-1).mean().item()
                - oracle_losses.mean().item()
            ),
            "selected_variant": selected["selected_variant"],
            "selected_variant_reason": selected["reason"],
            "shortcut_interpretation": selected["shortcut_interpretation"],
            "variants": variant_summaries,
            "support_audit": _residual_support_audit(base, residual, inputs),
            "residual_parameter_delta": _state_dict_delta(before_residual, residual),
        },
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "variant_metrics_csv": str(out_dir / "variant_metrics.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _feature_variants(hidden: Any) -> list[tuple[str, Any, list[str]]]:
    import torch

    current = hidden[:, :-1, :].detach()
    previous = torch.cat([current[:, :1, :], current[:, :-1, :]], dim=1)
    next_hidden = hidden[:, 1:, :].detach()
    seq_len = int(current.shape[1])
    normalized_position = _normalized_position(
        batch_size=int(current.shape[0]),
        seq_len=seq_len,
        dtype=current.dtype,
        device=current.device,
    )
    angle = normalized_position * (2.0 * torch.pi)
    position = torch.cat(
        [normalized_position, torch.sin(angle), torch.cos(angle)],
        dim=-1,
    )
    local_context = torch.cat(
        [previous, next_hidden, current - previous, next_hidden - current],
        dim=-1,
    )
    full_context = torch.cat([current, local_context, position], dim=-1)
    return [
        ("hidden_only", current.reshape(-1, current.shape[-1]), ["current_hidden"]),
        (
            "position_only",
            position.reshape(-1, position.shape[-1]),
            ["normalized_token_position", "position_sin", "position_cos"],
        ),
        (
            "context_only",
            local_context.reshape(-1, local_context.shape[-1]),
            ["previous_hidden", "next_hidden", "previous_delta", "next_delta"],
        ),
        (
            "full_context",
            full_context.reshape(-1, full_context.shape[-1]),
            [
                "current_hidden",
                "previous_hidden",
                "next_hidden",
                "previous_delta",
                "next_delta",
                "normalized_token_position",
                "position_sin",
                "position_cos",
            ],
        ),
    ]


def _normalized_position(
    *,
    batch_size: int,
    seq_len: int,
    dtype: Any,
    device: Any,
) -> Any:
    import torch

    if seq_len <= 1:
        return torch.zeros(batch_size, seq_len, 1, dtype=dtype, device=device)
    return torch.linspace(0.0, 1.0, seq_len, dtype=dtype, device=device).view(
        1, seq_len, 1
    ).expand(batch_size, seq_len, 1)


def _even_flat_position_mask(hidden: Any) -> Any:
    import torch

    flat_count = int(hidden.shape[0]) * (int(hidden.shape[1]) - 1)
    return torch.arange(flat_count, device=hidden.device).remainder(2) == 0


def _train_feature_ablation_support_head(
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    *,
    rows: list[dict[str, Any]],
    features: Any,
    train_mask: Any,
    token_loss_matrix: Any,
    router_token_losses: Any,
    oracle_indices: Any,
    seed: int,
    variant_name: str,
    feature_names: list[str],
    steps: int,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed + 3209 + len(feature_names) * 17)
    width = max(16, min(128, int(features.shape[-1]) * 2))
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
        support_probs = torch.softmax(selector(features[train_mask]) / temperature, dim=-1)
        expected_loss = (
            support_probs * token_loss_matrix[train_mask].detach()
        ).sum(dim=-1).mean()
        oracle_loss = F.cross_entropy(selector(features[train_mask]), oracle_indices[train_mask])
        loss = expected_loss + 0.05 * oracle_loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predicted_indices = selector(features).argmax(dim=-1).detach()
        selected_losses = token_loss_matrix[
            torch.arange(token_loss_matrix.shape[0], device=features.device),
            predicted_indices,
        ].detach()
    token_losses = _intervention_token_losses(
        base,
        residual,
        hidden,
        targets,
        vocab_size,
        rows=rows,
        predicted_indices=predicted_indices,
    )
    flat_router_losses = router_token_losses.reshape(-1).detach()
    oracle_losses = token_loss_matrix.min(dim=1).values.detach()
    split_rows = [
        _split_row(
            variant_name,
            "all",
            torch.ones_like(train_mask, dtype=torch.bool),
            predicted_indices=predicted_indices,
            oracle_indices=oracle_indices,
            selector_losses=selected_losses,
            intervention_losses=token_losses,
            router_losses=flat_router_losses,
            oracle_losses=oracle_losses,
        ),
        _split_row(
            variant_name,
            "train_even_positions",
            train_mask,
            predicted_indices=predicted_indices,
            oracle_indices=oracle_indices,
            selector_losses=selected_losses,
            intervention_losses=token_losses,
            router_losses=flat_router_losses,
            oracle_losses=oracle_losses,
        ),
        _split_row(
            variant_name,
            "holdout_odd_positions",
            ~train_mask,
            predicted_indices=predicted_indices,
            oracle_indices=oracle_indices,
            selector_losses=selected_losses,
            intervention_losses=token_losses,
            router_losses=flat_router_losses,
            oracle_losses=oracle_losses,
        ),
    ]
    selected_counts: dict[str, int] = {}
    for index in predicted_indices.detach().cpu().tolist():
        key = str(rows[int(index)]["support_key"])
        selected_counts[key] = selected_counts.get(key, 0) + 1
    by_split = {row["split"]: row for row in split_rows}
    return {
        "summary": {
            "variant": variant_name,
            "features": feature_names,
            "feature_dim": int(features.shape[-1]),
            "hidden_width": width,
            "training_steps": steps,
            "training_objective": "expected_fixed_batch_support_ce_plus_oracle_index_anchor",
            "temperature": temperature,
            "selected_support_counts": dict(
                sorted(selected_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
            "all": by_split["all"],
            "holdout": by_split["holdout_odd_positions"],
        },
        "rows": split_rows,
    }


def _intervention_token_losses(
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    *,
    rows: list[dict[str, Any]],
    predicted_indices: Any,
) -> Any:
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
    selected = torch.tensor(flat_supports, dtype=torch.long, device=hidden.device).view(
        batch_size,
        seq_len - 1,
        top_k,
    )
    support_indices[:, :-1, :] = selected
    support_indices[:, -1:, :] = selected[:, -1:, :]
    logits = base.decode(residual(hidden, support_indices=support_indices))
    return _token_losses(logits, targets).reshape(-1).detach()


def _split_row(
    variant: str,
    split: str,
    mask: Any,
    *,
    predicted_indices: Any,
    oracle_indices: Any,
    selector_losses: Any,
    intervention_losses: Any,
    router_losses: Any,
    oracle_losses: Any,
) -> dict[str, Any]:
    positions = int(mask.sum().item())
    if positions == 0:
        return {
            "variant": variant,
            "split": split,
            "positions": 0,
            "oracle_target_accuracy": 0.0,
            "selector_loss": None,
            "selector_minus_router_loss": None,
            "selector_oracle_gap_recovery_fraction": None,
            "intervention_loss": None,
            "intervention_minus_router_loss": None,
            "intervention_oracle_gap_recovery_fraction": None,
        }
    router_loss = float(router_losses[mask].mean().item())
    oracle_loss = float(oracle_losses[mask].mean().item())
    selector_loss = float(selector_losses[mask].mean().item())
    intervention_loss = float(intervention_losses[mask].mean().item())
    gap = router_loss - oracle_loss
    selector_recovery = None if abs(gap) <= 1e-12 else (router_loss - selector_loss) / gap
    intervention_recovery = (
        None if abs(gap) <= 1e-12 else (router_loss - intervention_loss) / gap
    )
    return {
        "variant": variant,
        "split": split,
        "positions": positions,
        "router_loss": router_loss,
        "oracle_loss": oracle_loss,
        "oracle_target_accuracy": float(
            (predicted_indices[mask] == oracle_indices[mask])
            .to(dtype=router_losses.dtype)
            .mean()
            .item()
        ),
        "selector_loss": selector_loss,
        "selector_minus_router_loss": selector_loss - router_loss,
        "selector_oracle_gap_recovery_fraction": selector_recovery,
        "intervention_loss": intervention_loss,
        "intervention_minus_router_loss": intervention_loss - router_loss,
        "intervention_oracle_gap_recovery_fraction": intervention_recovery,
    }


def _select_variant(variants: dict[str, Any]) -> dict[str, str | None]:
    holdouts = {
        name: variant["holdout"]["intervention_oracle_gap_recovery_fraction"]
        for name, variant in variants.items()
    }
    valid = {name: value for name, value in holdouts.items() if value is not None}
    if not valid:
        return {
            "decision": "contextual_router_shortcut_ablation_inconclusive",
            "selected_variant": None,
            "reason": "no variant has a nonzero router-oracle gap on holdout",
            "shortcut_interpretation": "inconclusive",
        }
    selected_variant, selected_value = max(valid.items(), key=lambda item: item[1])
    full = valid.get("full_context")
    position = valid.get("position_only")
    context = valid.get("context_only")
    if selected_variant == "position_only":
        interpretation = "position_shortcut_risk_high"
    elif full is not None and position is not None and position >= 0.8 * full:
        interpretation = "position_shortcut_risk_material"
    elif full is not None and context is not None and context >= 0.8 * full:
        interpretation = "local_context_without_current_hidden_explains_most_gain"
    elif selected_variant == "full_context":
        interpretation = "full_context_features_best_supported"
    else:
        interpretation = "non_full_feature_view_competitive"
    return {
        "decision": "contextual_router_shortcut_ablation_completed",
        "selected_variant": selected_variant,
        "reason": (
            f"{selected_variant} has the highest holdout realized intervention "
            f"oracle-gap recovery ({selected_value})"
        ),
        "shortcut_interpretation": interpretation,
    }


def _write_variant_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "variant",
        "split",
        "positions",
        "router_loss",
        "oracle_loss",
        "oracle_target_accuracy",
        "selector_loss",
        "selector_minus_router_loss",
        "selector_oracle_gap_recovery_fraction",
        "intervention_loss",
        "intervention_minus_router_loss",
        "intervention_oracle_gap_recovery_fraction",
    ]
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
        "Contextual-router shortcut ablation.",
        "",
        f"- Config: `{summary['config_path']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Router loss: `{ablation['router_loss']:.8f}`",
        f"- Oracle loss: `{ablation['oracle_loss']:.8f}`",
        f"- Selected variant: `{ablation['selected_variant']}`",
        f"- Interpretation: `{ablation['shortcut_interpretation']}`",
        "",
        "## Holdout Realized Intervention Recovery",
    ]
    for name, variant in ablation["variants"].items():
        holdout = variant["holdout"]
        lines.append(
            "- "
            f"{name}: `{holdout['intervention_oracle_gap_recovery_fraction']}` "
            f"(loss `{holdout['intervention_loss']}`)"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--probe-steps", type=int, default=200)
    args = parser.parse_args(argv)
    summary = run_contextual_router_shortcut_ablation(
        args.config,
        args.out,
        probe_steps=args.probe_steps,
    )
    print(json.dumps(summary["ablation"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
