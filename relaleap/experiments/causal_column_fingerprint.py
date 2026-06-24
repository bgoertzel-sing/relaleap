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
    variant_summaries: list[dict[str, Any]] = []
    for weight in weights:
        residual = _train_residual_variant(
            base=base,
            inputs=inputs,
            targets=targets,
            vocab_size=vocab_size,
            seed=seed,
            hidden_dim=hidden_dim,
            num_columns=num_columns,
            atoms_per_column=atoms_per_column,
            top_k=top_k,
            support_router=support_router,
            contextual_router_hidden_dim=contextual_router_hidden_dim,
            max_steps=max_steps,
            learning_rate=learning_rate,
            residual_objective=residual_objective,
            training_cfg=training_cfg,
            load_balance_weight=weight,
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
                    variant=_variant_name(weight),
                    load_balance_weight=weight,
                    router_hidden=router_hidden,
                    router_logits=router_logits,
                    router_loss=router_loss,
                    router_support=router_support,
                    scores=scores,
                    column_values=values,
                    column_counts=support_audit["column_counts"],
                )
            )
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
                for pair in _all_pairs(num_columns)
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
                    variant=_variant_name(weight),
                    load_balance_weight=weight,
                    router_hidden=router_hidden,
                    router_logits=router_logits,
                    router_loss=router_loss,
                    column_values=values,
                    selected_pairs=selected_pairs,
                )
            )
            variant_summaries.append(
                {
                    "variant": _variant_name(weight),
                    "load_balance_weight": weight,
                    "alpha0_ce_loss": router_loss,
                    "used_columns": support_audit["used_columns"],
                    "dead_columns": support_audit["dead_columns"],
                    "unique_support_sets": support_audit["unique_support_sets"],
                    "load_entropy": _load_entropy(support_audit["column_counts"]),
                    "mean_abs_ablate_loss_delta": _mean_abs(
                        [
                            row["ablate_loss_delta"]
                            for row in column_rows
                            if row["variant"] == _variant_name(weight)
                        ]
                    ),
                    "mean_abs_force_loss_delta": _mean_abs(
                        [
                            row["force_loss_delta"]
                            for row in column_rows
                            if row["variant"] == _variant_name(weight)
                        ]
                    ),
                    "max_abs_ablate_loss_delta": _max_abs(
                        [
                            row["ablate_loss_delta"]
                            for row in column_rows
                            if row["variant"] == _variant_name(weight)
                        ]
                    ),
                    "max_abs_force_loss_delta": _max_abs(
                        [
                            row["force_loss_delta"]
                            for row in column_rows
                            if row["variant"] == _variant_name(weight)
                        ]
                    ),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "column_fingerprints.csv", _COLUMN_FIELDNAMES, column_rows)
    _write_csv(out_dir / "pair_interventions.csv", _PAIR_FIELDNAMES, pair_rows)
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
            "column_fingerprint_count": len(column_rows),
            "pair_intervention_count": len(pair_rows),
            "variants": variant_summaries,
        },
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "column_fingerprints_csv": str(out_dir / "column_fingerprints.csv"),
            "pair_interventions_csv": str(out_dir / "pair_interventions.csv"),
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
    router_hidden: Any,
    router_logits: Any,
    router_loss: float,
    column_values: Any,
    selected_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for selected in selected_pairs:
        support = tuple(int(part) for part in selected["support"])
        fixed_hidden = residual(hidden, support_indices=_fixed_support(hidden, support))
        fixed_logits = base.decode(fixed_hidden)
        fixed_loss = _ce_loss(fixed_logits, targets, vocab_size)
        rows.append(
            {
                "variant": variant,
                "load_balance_weight": load_balance_weight,
                "intervention": selected["intervention"],
                "support": _support_key(support),
                "router_support_count": selected["router_support_count"],
                "fixed_support_loss": fixed_loss,
                "fixed_support_loss_delta": fixed_loss - router_loss,
                "fixed_support_logit_mse": _mse_delta(fixed_logits, router_logits),
                "fixed_support_residual_stream_l2_delta": _stream_l2_delta(
                    fixed_hidden,
                    router_hidden,
                ),
                "pair_value_cosine": _pair_value_cosine(column_values, support),
            }
        )
    return rows


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
    "router_support_count",
    "fixed_support_loss",
    "fixed_support_loss_delta",
    "fixed_support_logit_mse",
    "fixed_support_residual_stream_l2_delta",
    "pair_value_cosine",
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
    parser.add_argument("--max-pair-rows", type=int, default=DEFAULT_MAX_PAIR_ROWS)
    args = parser.parse_args(argv)
    summary = run_causal_column_fingerprint(
        args.config,
        args.out,
        load_balance_weights=args.load_balance_weights,
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
