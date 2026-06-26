"""Column redundancy and load-balance diagnostic for support-wide runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.run import _load_torch_info
from relaleap.experiments.run import _read_config
from relaleap.experiments.support_audit import _configured_residual_loss
from relaleap.smoke import ResidualColumns
from relaleap.smoke import SUPPORT_ROUTER_CHOICES
from relaleap.smoke import TinyCharTransformer
from relaleap.smoke import _build_batch
from relaleap.smoke import _residual_support_audit
from relaleap.smoke import _state_dict_delta


DEFAULT_CONFIG = Path(
    "configs/token_larger_support_wide_hep_temporal_clipped_objective_gate.yaml"
)
DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_support_wide_promoted_default_column_redundancy"
)


def run_column_redundancy(config_path: Path, out_dir: Path) -> dict[str, Any]:
    """Train a configured residual adapter and inspect column load/redundancy."""

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("column redundancy diagnostic requires torch") from exc

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
        _, support = residual(hidden, return_support=True)
        scores = residual._score_columns(hidden)
        atom_weights = F.softmax(residual.atom_logits, dim=-1)
        column_values = torch.einsum("ca,cah->ch", atom_weights, residual.atom_values)
        column_norms = column_values.norm(dim=-1)
        normalized_values = F.normalize(column_values, dim=-1, eps=1e-12)
        similarity = normalized_values @ normalized_values.T
        pairwise_distance = torch.cdist(column_values, column_values, p=2)

    support_audit = _residual_support_audit(base, residual, inputs)
    column_counts = [int(count) for count in support_audit["column_counts"]]
    total_slots = int(support_audit["total_support_slots"])
    support_rows = support.reshape(-1, support.shape[-1]).detach().cpu().tolist()
    co_selected_counts = _co_selected_counts(support_rows, num_columns)
    load_rows = _column_load_rows(
        column_counts=column_counts,
        total_slots=total_slots,
        column_norms=column_norms.detach().cpu().tolist(),
        scores=scores.detach().cpu(),
    )
    pair_rows = _column_pair_rows(
        column_counts=column_counts,
        similarity=similarity.detach().cpu(),
        pairwise_distance=pairwise_distance.detach().cpu(),
        co_selected_counts=co_selected_counts,
    )
    diagnostic = _diagnostic_summary(
        support_audit=support_audit,
        load_rows=load_rows,
        pair_rows=pair_rows,
        column_counts=column_counts,
        total_slots=total_slots,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "column_loads.csv", load_rows)
    _write_csv(out_dir / "column_pair_similarity.csv", pair_rows)
    summary = {
        "status": "ok",
        "experiment_id": f"{experiment_id}_column_redundancy",
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        **_load_torch_info(),
        "diagnostic": {
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
            "support_audit": support_audit,
            "residual_parameter_delta": _state_dict_delta(before_residual, residual),
            **diagnostic,
        },
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "column_loads_csv": str(out_dir / "column_loads.csv"),
            "column_pair_similarity_csv": str(out_dir / "column_pair_similarity.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _column_load_rows(
    *,
    column_counts: list[int],
    total_slots: int,
    column_norms: list[float],
    scores: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    flat_scores = scores.reshape(-1, scores.shape[-1])
    score_means = flat_scores.mean(dim=0).tolist()
    score_stds = flat_scores.std(dim=0, unbiased=False).tolist()
    score_maxes = flat_scores.max(dim=0).values.tolist()
    ranked_columns = sorted(
        range(len(column_counts)),
        key=lambda column: float(score_means[column]),
        reverse=True,
    )
    score_ranks = {
        column: rank + 1 for rank, column in enumerate(ranked_columns)
    }
    for column, count in enumerate(column_counts):
        rows.append(
            {
                "column": column,
                "support_count": count,
                "support_fraction": count / total_slots if total_slots else 0.0,
                "dead": count == 0,
                "value_norm": float(column_norms[column]),
                "score_mean": float(score_means[column]),
                "score_std": float(score_stds[column]),
                "score_max": float(score_maxes[column]),
                "score_mean_rank": score_ranks[column],
            }
        )
    return rows


def _co_selected_counts(support_rows: list[list[int]], num_columns: int) -> dict[tuple[int, int], int]:
    counts = {
        (column_a, column_b): 0
        for column_a in range(num_columns)
        for column_b in range(column_a + 1, num_columns)
    }
    for support in support_rows:
        normalized = sorted({int(column) for column in support})
        for left_index, column_a in enumerate(normalized):
            for column_b in normalized[left_index + 1 :]:
                counts[(column_a, column_b)] += 1
    return counts


def _column_pair_rows(
    *,
    column_counts: list[int],
    similarity: Any,
    pairwise_distance: Any,
    co_selected_counts: dict[tuple[int, int], int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    num_columns = len(column_counts)
    for column_a in range(num_columns):
        for column_b in range(column_a + 1, num_columns):
            rows.append(
                {
                    "column_a": column_a,
                    "column_b": column_b,
                    "cosine_similarity": float(similarity[column_a, column_b].item()),
                    "value_distance": float(pairwise_distance[column_a, column_b].item()),
                    "column_a_used": column_counts[column_a] > 0,
                    "column_b_used": column_counts[column_b] > 0,
                    "both_used": column_counts[column_a] > 0 and column_counts[column_b] > 0,
                    "co_selected_count": co_selected_counts[(column_a, column_b)],
                }
            )
    return rows


def _diagnostic_summary(
    *,
    support_audit: dict[str, Any],
    load_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    column_counts: list[int],
    total_slots: int,
) -> dict[str, Any]:
    fractions = [count / total_slots for count in column_counts] if total_slots else []
    entropy = -sum(fraction * math.log(fraction) for fraction in fractions if fraction > 0.0)
    used_columns = int(support_audit["used_columns"])
    num_columns = int(support_audit["num_columns"])
    effective_columns = 1.0 / sum(fraction * fraction for fraction in fractions) if fractions and sum(fraction * fraction for fraction in fractions) > 0.0 else 0.0
    normalized_entropy = entropy / math.log(num_columns) if num_columns > 1 else 0.0
    active_pairs = [row for row in pair_rows if row["both_used"]]
    dead_columns = [row for row in load_rows if row["dead"]]
    live_columns = [row for row in load_rows if not row["dead"]]
    high_similarity_pairs = [
        row for row in active_pairs if row["cosine_similarity"] >= 0.95
    ]
    nearest_live_for_dead = []
    for dead in dead_columns:
        column = int(dead["column"])
        candidates = [
            row
            for row in pair_rows
            if (
                (row["column_a"] == column and not load_rows[int(row["column_b"])]["dead"])
                or (row["column_b"] == column and not load_rows[int(row["column_a"])]["dead"])
            )
        ]
        if candidates:
            nearest_live_for_dead.append(
                max(candidates, key=lambda row: float(row["cosine_similarity"]))
            )
    strongest_similarity = (
        max(pair_rows, key=lambda row: float(row["cosine_similarity"]))
        if pair_rows
        else None
    )
    strongest_active_similarity = (
        max(active_pairs, key=lambda row: float(row["cosine_similarity"]))
        if active_pairs
        else None
    )
    most_loaded = max(load_rows, key=lambda row: int(row["support_count"])) if load_rows else None
    return {
        "used_columns": used_columns,
        "dead_columns": num_columns - used_columns,
        "load_entropy": entropy,
        "normalized_load_entropy": normalized_entropy,
        "effective_num_columns": effective_columns,
        "effective_column_fraction": effective_columns / num_columns if num_columns else 0.0,
        "max_column_fraction": support_audit.get("max_column_fraction", 0.0),
        "most_loaded_column": None if most_loaded is None else most_loaded["column"],
        "most_loaded_column_fraction": None if most_loaded is None else most_loaded["support_fraction"],
        "zero_value_columns": sum(1 for row in load_rows if float(row["value_norm"]) <= 1e-12),
        "live_column_mean_value_norm": _mean(
            [float(row["value_norm"]) for row in live_columns]
        ),
        "dead_column_mean_value_norm": _mean(
            [float(row["value_norm"]) for row in dead_columns]
        ),
        "active_pair_count": len(active_pairs),
        "high_similarity_active_pair_count": len(high_similarity_pairs),
        "max_active_pair_cosine_similarity": (
            None
            if strongest_active_similarity is None
            else strongest_active_similarity["cosine_similarity"]
        ),
        "strongest_similarity_pair": _pair_key(strongest_similarity),
        "strongest_active_similarity_pair": _pair_key(strongest_active_similarity),
        "dead_column_nearest_live_mean_cosine_similarity": _mean(
            [float(row["cosine_similarity"]) for row in nearest_live_for_dead]
        ),
    }


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _pair_key(row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    return f"{row['column_a']},{row['column_b']}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    diagnostic = summary["diagnostic"]
    path.write_text(
        "\n".join(
            [
                "# Column Redundancy Diagnostic",
                "",
                f"- Experiment: `{summary['experiment_id']}`",
                f"- Config: `{summary['config_path']}`",
                f"- Status: `{summary['status']}`",
                f"- Used columns: `{diagnostic['used_columns']}` of `{diagnostic['num_columns']}`",
                f"- Dead columns: `{diagnostic['dead_columns']}`",
                f"- Effective number of columns: `{diagnostic['effective_num_columns']}`",
                f"- Normalized load entropy: `{diagnostic['normalized_load_entropy']}`",
                f"- Max column fraction: `{diagnostic['max_column_fraction']}`",
                f"- Strongest active similarity pair: `{diagnostic['strongest_active_similarity_pair']}`",
                f"- High-similarity active pairs: `{diagnostic['high_similarity_active_pair_count']}`",
                f"- Dead-nearest-live mean cosine: `{diagnostic['dead_column_nearest_live_mean_cosine_similarity']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_column_redundancy(args.config, args.out)
    print(json.dumps(summary["diagnostic"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
