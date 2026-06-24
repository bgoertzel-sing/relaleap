"""Support-selection deconfounding controls for promoted contextual routers."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from relaleap.experiments.run import _load_torch_info
from relaleap.experiments.run import _read_config
from relaleap.experiments.support_audit import _ce_loss
from relaleap.experiments.support_audit import _score_for_support
from relaleap.experiments.support_audit import _support_key
from relaleap.experiments.support_audit import _token_losses
from relaleap.smoke import ResidualColumns
from relaleap.smoke import TinyCharTransformer
from relaleap.smoke import _build_batch
from relaleap.smoke import _residual_support_audit
from relaleap.smoke import _state_dict_delta


DEFAULT_CONFIG = Path(
    "configs/char_validation_support_wide_hep_temporal_clipped_objective_gate.yaml"
)
DEFAULT_OUT_DIR = Path("results/audits/support_deconfounding_controls")


@dataclass(frozen=True)
class _VariantSpec:
    name: str
    kind: str
    top_k: int
    num_columns: int
    atoms_per_column: int
    support_router: str
    contextual_router_hidden_dim: int
    residual_scale: float = 1.0
    fixed_support: tuple[int, ...] | None = None
    dense_rank: int | None = None
    dense_norm_match: bool = False


def run_support_deconfounding(config_path: Path, out_dir: Path) -> dict[str, Any]:
    """Train matched sparse/dense controls and write deconfounding artifacts."""

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("support deconfounding audit requires torch") from exc

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
        raise ValueError("support deconfounding controls currently require supervised_ce")
    if top_k != 2:
        raise ValueError("support deconfounding controls expect promoted top_k: 2")
    if support_router != "contextual_mlp":
        raise ValueError("support deconfounding controls expect contextual_mlp baseline")

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
    hidden = base.encode(inputs).detach()
    empty_logits = base.decode(hidden)
    empty_loss = _ce_loss(empty_logits, targets, vocab_size)

    rank_matched_columns = max(1, num_columns * top_k)
    sparse_baseline_stored_parameters = sum(
        p.numel()
        for p in ResidualColumns(
            hidden_dim=hidden_dim,
            num_columns=num_columns,
            atoms_per_column=atoms_per_column,
            top_k=top_k,
            support_router=support_router,
            contextual_router_hidden_dim=contextual_router_hidden_dim,
        ).parameters()
    )
    active_rank = top_k * atoms_per_column
    stored_matched_rank = max(1, round(sparse_baseline_stored_parameters / (2 * hidden_dim)))
    specs = [
        _VariantSpec(
            name="learned_topk2_contextual",
            kind="sparse",
            top_k=2,
            num_columns=num_columns,
            atoms_per_column=atoms_per_column,
            support_router="contextual_mlp",
            contextual_router_hidden_dim=contextual_router_hidden_dim,
        ),
        _VariantSpec(
            name="rank_matched_topk1_contextual",
            kind="sparse",
            top_k=1,
            num_columns=rank_matched_columns,
            atoms_per_column=atoms_per_column,
            support_router="contextual_mlp",
            contextual_router_hidden_dim=contextual_router_hidden_dim,
        ),
        _VariantSpec(
            name="random_fixed_topk2",
            kind="sparse",
            top_k=2,
            num_columns=num_columns,
            atoms_per_column=atoms_per_column,
            support_router="contextual_mlp",
            contextual_router_hidden_dim=contextual_router_hidden_dim,
            fixed_support=_fixed_random_support(num_columns, seed),
        ),
        _VariantSpec(
            name="learned_topk2_scale_one_over_k",
            kind="sparse",
            top_k=2,
            num_columns=num_columns,
            atoms_per_column=atoms_per_column,
            support_router="contextual_mlp",
            contextual_router_hidden_dim=contextual_router_hidden_dim,
            residual_scale=1.0 / top_k,
        ),
        _VariantSpec(
            name="learned_topk2_scale_one_over_sqrt_k",
            kind="sparse",
            top_k=2,
            num_columns=num_columns,
            atoms_per_column=atoms_per_column,
            support_router="contextual_mlp",
            contextual_router_hidden_dim=contextual_router_hidden_dim,
            residual_scale=1.0 / math.sqrt(top_k),
        ),
        _VariantSpec(
            name="dense_rank_flop_matched_residual",
            kind="dense",
            top_k=0,
            num_columns=0,
            atoms_per_column=0,
            support_router="none",
            contextual_router_hidden_dim=0,
            dense_rank=active_rank,
        ),
        _VariantSpec(
            name="dense_rank_flop_matched_norm_matched",
            kind="dense",
            top_k=0,
            num_columns=0,
            atoms_per_column=0,
            support_router="none",
            contextual_router_hidden_dim=0,
            dense_rank=active_rank,
            dense_norm_match=True,
        ),
        _VariantSpec(
            name="dense_stored_parameter_matched_residual",
            kind="dense",
            top_k=0,
            num_columns=0,
            atoms_per_column=0,
            support_router="none",
            contextual_router_hidden_dim=0,
            dense_rank=stored_matched_rank,
        ),
    ]

    variant_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    learned_support = None
    learned_sparse_residual_norm: float | None = None
    for offset, spec in enumerate(specs):
        torch.manual_seed(seed + 100 * offset)
        if spec.kind == "dense":
            dense_rank = int(spec.dense_rank or active_rank)
            adapter = nn.Sequential(
                nn.Linear(hidden_dim, dense_rank, bias=False),
                nn.Linear(dense_rank, hidden_dim, bias=False),
            )
            nn.init.normal_(adapter[0].weight, mean=0.0, std=0.02)
            nn.init.zeros_(adapter[1].weight)
            before = {key: value.detach().clone() for key, value in adapter.state_dict().items()}
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate)
            for _ in range(max_steps):
                optimizer.zero_grad(set_to_none=True)
                logits = base.decode(hidden + adapter(hidden))
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, vocab_size),
                    targets[:, :-1].reshape(-1),
                )
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                raw_residual = adapter(hidden)
                raw_residual_norm = float(raw_residual.norm(dim=-1).mean().item())
                dense_eval_scale = 1.0
                if spec.dense_norm_match:
                    if learned_sparse_residual_norm is None:
                        raise RuntimeError("norm-matched dense control requires sparse baseline first")
                    dense_eval_scale = learned_sparse_residual_norm / max(
                        raw_residual_norm,
                        1e-12,
                    )
                output_hidden = hidden + raw_residual * dense_eval_scale
                logits = base.decode(output_hidden)
                loss_value = _ce_loss(logits, targets, vocab_size)
                residual_norm = float((output_hidden - hidden).norm(dim=-1).mean().item())
                stored_parameters = sum(p.numel() for p in adapter.parameters())
            variant_rows.append(
                {
                    **_variant_base_row(spec, loss_value, empty_loss, residual_norm),
                    "raw_residual_norm_mean": raw_residual_norm,
                    "norm_match_target": "" if not spec.dense_norm_match else learned_sparse_residual_norm,
                    "norm_match_scale": dense_eval_scale,
                    "support_margin_mean": "",
                    "used_columns": "",
                    "dead_columns": "",
                    "unique_support_sets": "",
                    "oracle_support_regret": "",
                    "stored_parameters": stored_parameters,
                    "active_parameters_proxy": 2 * hidden_dim * dense_rank,
                    "stored_parameter_ratio_to_sparse": stored_parameters
                    / sparse_baseline_stored_parameters,
                    "parameter_delta": _state_dict_delta(before, adapter),
                }
            )
            continue

        residual = ResidualColumns(
            hidden_dim=hidden_dim,
            num_columns=spec.num_columns,
            atoms_per_column=spec.atoms_per_column,
            top_k=spec.top_k,
            support_router=spec.support_router,
            contextual_router_hidden_dim=spec.contextual_router_hidden_dim,
        )
        before = {key: value.detach().clone() for key, value in residual.state_dict().items()}
        optimizer = torch.optim.AdamW(residual.parameters(), lr=learning_rate)
        fixed_support_indices = _fixed_support_indices(
            spec.fixed_support,
            hidden,
            spec.top_k,
        )
        for _ in range(max_steps):
            optimizer.zero_grad(set_to_none=True)
            output_hidden = _scaled_sparse_forward(
                residual,
                hidden,
                residual_scale=spec.residual_scale,
                support_indices=fixed_support_indices,
            )
            logits = base.decode(output_hidden)
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, vocab_size),
                targets[:, :-1].reshape(-1),
            )
            loss.backward()
            optimizer.step()
        residual.eval()
        with torch.no_grad():
            output_hidden, support = _scaled_sparse_forward(
                residual,
                hidden,
                residual_scale=spec.residual_scale,
                support_indices=fixed_support_indices,
                return_support=True,
            )
            logits = base.decode(output_hidden)
            loss_value = _ce_loss(logits, targets, vocab_size)
            residual_norm = float((output_hidden - hidden).norm(dim=-1).mean().item())
            scores = residual._score_columns(hidden)
            margins = _support_margins(scores, spec.top_k)
            support_audit = _support_audit_from_support(support, spec.num_columns)
            oracle_regret = ""
            if spec.top_k == 2 and spec.num_columns <= 24:
                pair_rows = [
                    _score_for_support(
                        base,
                        residual,
                        hidden,
                        targets,
                        vocab_size,
                        support=pair,
                        empty_loss=empty_loss,
                        router_loss=loss_value,
                    )
                    for pair in itertools.combinations(range(spec.num_columns), 2)
                ]
                token_losses = _token_losses(logits, targets)
                oracle_regret = _oracle_regret(token_losses, pair_rows)
            if spec.name == "learned_topk2_contextual":
                learned_support = support.detach().clone()
                learned_sparse_residual_norm = residual_norm
            support_rows.extend(
                _support_overlap_rows(
                    variant=spec.name,
                    support=support,
                    baseline_support=learned_support,
                )
            )
            intervention_rows.extend(
                _column_intervention_rows(
                    base=base,
                    residual=residual,
                    hidden=hidden,
                    targets=targets,
                    vocab_size=vocab_size,
                    variant=spec.name,
                    support_audit=support_audit,
                    router_loss=loss_value,
                    residual_scale=spec.residual_scale,
                )
            )
        variant_rows.append(
            {
                **_variant_base_row(spec, loss_value, empty_loss, residual_norm),
                "raw_residual_norm_mean": residual_norm,
                "norm_match_target": "",
                "norm_match_scale": 1.0,
                "support_margin_mean": float(margins["mean"]),
                "used_columns": support_audit["used_columns"],
                "dead_columns": support_audit["dead_columns"],
                "unique_support_sets": support_audit["unique_support_sets"],
                "oracle_support_regret": oracle_regret,
                "stored_parameters": sum(p.numel() for p in residual.parameters()),
                "active_parameters_proxy": spec.top_k * hidden_dim * spec.atoms_per_column,
                "stored_parameter_ratio_to_sparse": sum(p.numel() for p in residual.parameters())
                / sparse_baseline_stored_parameters,
                "parameter_delta": _state_dict_delta(before, residual),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "variant_metrics.csv", variant_rows)
    _write_csv(out_dir / "column_interventions.csv", intervention_rows)
    _write_csv(out_dir / "support_churn.csv", support_rows)
    summary = {
        "status": "ok",
        "experiment_id": f"{experiment_id}_support_deconfounding_controls",
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
            "baseline_num_columns": num_columns,
            "baseline_top_k": top_k,
            "variant_count": len(variant_rows),
            "variants": variant_rows,
            "primary_guardrail": "ce_loss",
            "primary_deconfounding_outputs": [
                "oracle_support_regret",
                "support_churn",
                "residual_norm_mean",
                "raw_residual_norm_mean",
                "norm_match_scale",
                "support_margin_mean",
                "column_intervention_loss_delta",
                "stored_parameters",
                "stored_parameter_ratio_to_sparse",
                "active_parameters_proxy",
            ],
        },
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "variant_metrics_csv": str(out_dir / "variant_metrics.csv"),
            "column_interventions_csv": str(out_dir / "column_interventions.csv"),
            "support_churn_csv": str(out_dir / "support_churn.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _fixed_random_support(num_columns: int, seed: int) -> tuple[int, int]:
    import random

    rng = random.Random(seed + 4049)
    return tuple(sorted(rng.sample(range(num_columns), 2)))  # type: ignore[return-value]


def _fixed_support_indices(
    support: tuple[int, ...] | None,
    hidden: Any,
    top_k: int,
) -> Any | None:
    if support is None:
        return None
    import torch

    return torch.tensor(support, dtype=torch.long, device=hidden.device).view(
        1,
        1,
        top_k,
    ).expand(hidden.shape[0], hidden.shape[1], top_k)


def _scaled_sparse_forward(
    residual: Any,
    hidden: Any,
    *,
    residual_scale: float,
    support_indices: Any | None,
    return_support: bool = False,
) -> Any:
    if return_support:
        output, support = residual(
            hidden,
            support_indices=support_indices,
            return_support=True,
        )
        return hidden + (output - hidden) * residual_scale, support
    output = residual(hidden, support_indices=support_indices)
    return hidden + (output - hidden) * residual_scale


def _variant_base_row(
    spec: _VariantSpec,
    ce_loss: float,
    empty_loss: float,
    residual_norm: float,
) -> dict[str, Any]:
    return {
        "variant": spec.name,
        "kind": spec.kind,
        "top_k": spec.top_k,
        "num_columns": spec.num_columns,
        "atoms_per_column": spec.atoms_per_column,
        "support_router": spec.support_router,
        "residual_scale": spec.residual_scale,
        "fixed_support": "" if spec.fixed_support is None else _support_key(spec.fixed_support),
        "ce_loss": ce_loss,
        "delta_from_empty_ce": ce_loss - empty_loss,
        "residual_norm_mean": residual_norm,
        "raw_residual_norm_mean": residual_norm,
        "norm_match_target": "",
        "norm_match_scale": 1.0,
    }


def _support_margins(scores: Any, top_k: int) -> dict[str, float]:
    sorted_scores = scores.sort(dim=-1, descending=True).values
    if top_k >= sorted_scores.shape[-1]:
        margin = sorted_scores[..., top_k - 1] - sorted_scores[..., top_k - 1]
    else:
        margin = sorted_scores[..., top_k - 1] - sorted_scores[..., top_k]
    return {"mean": float(margin.mean().detach().item())}


def _support_audit_from_support(support: Any, num_columns: int) -> dict[str, Any]:
    flat = support.reshape(-1, support.shape[-1]).detach().cpu().tolist()
    counts = [0 for _ in range(num_columns)]
    support_sets: dict[str, int] = {}
    for row in flat:
        key = _support_key(tuple(sorted(int(index) for index in row)))
        support_sets[key] = support_sets.get(key, 0) + 1
        for index in row:
            counts[int(index)] += 1
    used = sum(1 for count in counts if count > 0)
    return {
        "num_columns": num_columns,
        "top_k": int(support.shape[-1]),
        "column_counts": counts,
        "used_columns": used,
        "dead_columns": num_columns - used,
        "unique_support_sets": len(support_sets),
    }


def _oracle_regret(router_token_losses: Any, rows: list[dict[str, Any]]) -> float:
    import torch

    stacked = torch.stack([row["_token_losses"] for row in rows], dim=0)
    oracle_losses = stacked.min(dim=0).values
    return float((router_token_losses - oracle_losses).mean().item())


def _support_overlap_rows(
    *,
    variant: str,
    support: Any,
    baseline_support: Any | None,
) -> list[dict[str, Any]]:
    flat = support.reshape(-1, support.shape[-1]).detach().cpu().tolist()
    unique = len({_support_key(tuple(sorted(int(index) for index in row))) for row in flat})
    if baseline_support is None:
        churn = 0.0
        exact = 1.0
    else:
        base_flat = baseline_support.reshape(-1, baseline_support.shape[-1]).detach().cpu().tolist()
        overlap = []
        exact_matches = 0
        for row, base_row in zip(flat, base_flat):
            row_set = {int(index) for index in row}
            base_set = {int(index) for index in base_row}
            overlap.append(len(row_set & base_set) / max(len(row_set | base_set), 1))
            exact_matches += int(row_set == base_set)
        exact = exact_matches / len(flat) if flat else 0.0
        churn = 1.0 - (sum(overlap) / len(overlap) if overlap else 0.0)
    return [
        {
            "variant": variant,
            "comparison": "against_learned_topk2_contextual",
            "support_churn_jaccard": churn,
            "exact_support_match_fraction": exact,
            "unique_support_sets": unique,
        }
    ]


def _column_intervention_rows(
    *,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    variant: str,
    support_audit: dict[str, Any],
    router_loss: float,
    residual_scale: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    num_columns = int(support_audit["num_columns"])
    top_k = int(support_audit["top_k"])
    for column in range(num_columns):
        if top_k == 1:
            support = (column,)
        else:
            partner = 0 if column != 0 else 1
            support = tuple(sorted((column, partner)))
        forced = _fixed_support_indices(support, hidden, len(support))
        output = _scaled_sparse_forward(
            residual,
            hidden,
            residual_scale=residual_scale,
            support_indices=forced,
        )
        force_loss = _ce_loss(base.decode(output), targets, vocab_size)
        rows.append(
            {
                "variant": variant,
                "column": column,
                "support_count": support_audit["column_counts"][column],
                "forced_support": _support_key(support),
                "force_loss": force_loss,
                "force_loss_delta": force_loss - router_loss,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    audit = summary["audit"]
    best = min(audit["variants"], key=lambda row: float(row["ce_loss"]))
    path.write_text(
        "\n".join(
            [
                f"# {summary['experiment_id']}",
                "",
                f"Status: {summary['status']}",
                f"Config: {summary['config_path']}",
                f"Best CE guardrail variant: {best['variant']} ({best['ce_loss']})",
                "",
                "This audit treats CE as a guardrail and records support/regret/churn, residual norm, parameter, and intervention controls for promoted contextual-router deconfounding.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_support_deconfounding(args.config, args.out)
    print(json.dumps({"status": summary["status"], "out_dir": summary["out_dir"]}))


if __name__ == "__main__":
    main()
