"""Dead-column recruitment probe for promoted support-wide runs."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.run import _load_torch_info
from relaleap.experiments.run import _read_config
from relaleap.experiments.support_audit import _ce_loss
from relaleap.experiments.support_audit import _configured_residual_loss
from relaleap.smoke import ResidualColumns
from relaleap.smoke import TinyCharTransformer
from relaleap.smoke import _build_batch
from relaleap.smoke import _evaluate_hep_alpha_sweep
from relaleap.smoke import _parse_hep_alpha_sweep
from relaleap.smoke import _parse_optional_float
from relaleap.smoke import _residual_support_audit
from relaleap.smoke import _state_dict_delta


DEFAULT_CONFIG = Path(
    "configs/token_larger_support_wide_hep_temporal_clipped_objective_gate.yaml"
)
DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_support_wide_promoted_default_dead_column_probe"
)
DEFAULT_LOAD_BALANCE_WEIGHTS = (0.0, 0.001, 0.01, 0.05)
DEFAULT_CE_TOLERANCE = 0.01


def run_dead_column_probe(
    config_path: Path,
    out_dir: Path,
    *,
    load_balance_weights: list[float] | None = None,
    ce_tolerance: float = DEFAULT_CE_TOLERANCE,
) -> dict[str, Any]:
    """Train baseline and load-balanced variants, then compare support usage."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("dead-column probe requires torch") from exc

    if ce_tolerance < 0.0:
        raise ValueError("ce_tolerance must be non-negative")
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
    inference_cfg = config.get("inference", {})

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
    pc_steps = int(inference_cfg.get("pc_steps", 1))
    hep_alpha = float(inference_cfg.get("hep_alpha", 0.0))
    hep_alphas = _parse_hep_alpha_sweep(inference_cfg, fallback_alpha=hep_alpha)
    hep_update_clip_norm = _parse_optional_float(
        inference_cfg.get("hep_update_clip_norm")
    )
    hep_settling_objective = str(
        inference_cfg.get("hep_settling_objective", "residual_adapter")
    )

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

    variant_rows = []
    for weight in weights:
        torch.manual_seed(seed)
        residual = ResidualColumns(
            hidden_dim=hidden_dim,
            num_columns=num_columns,
            atoms_per_column=atoms_per_column,
            top_k=top_k,
            support_router=support_router,
            contextual_router_hidden_dim=contextual_router_hidden_dim,
        )
        before_residual = {
            key: value.detach().clone() for key, value in residual.state_dict().items()
        }
        optimizer = torch.optim.AdamW(residual.parameters(), lr=learning_rate)
        residual.train()
        final_objective_loss = None
        final_load_balance_loss = None
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
            loss = objective_loss + weight * load_balance_loss
            loss.backward()
            optimizer.step()
            final_objective_loss = objective_loss.detach()
            final_load_balance_loss = load_balance_loss.detach()

        residual.eval()
        with torch.no_grad():
            hidden = base.encode(inputs)
            logits = base.decode(residual(hidden))
            alpha0_ce_loss = _ce_loss(logits, targets, vocab_size)
            support_audit = _residual_support_audit(base, residual, inputs)
        hep_rows = _evaluate_hep_alpha_sweep(
            base,
            residual,
            inputs,
            targets,
            vocab_size,
            pc_steps=pc_steps,
            alphas=hep_alphas,
            pinned_support=bool(column_cfg.get("pinned_support", False)),
            hep_update_clip_norm=hep_update_clip_norm,
            hep_settling_objective=hep_settling_objective,
        )
        best_hep = min(hep_rows, key=lambda row: float(row["loss"])) if hep_rows else None
        variant_rows.append(
            {
                "variant": _variant_name(weight),
                "load_balance_weight": weight,
                "alpha0_ce_loss": alpha0_ce_loss,
                "best_hep_alpha": None if best_hep is None else best_hep["alpha"],
                "best_hep_loss": None if best_hep is None else best_hep["loss"],
                "objective_loss": (
                    None if final_objective_loss is None else float(final_objective_loss.item())
                ),
                "load_balance_loss": (
                    None
                    if final_load_balance_loss is None
                    else float(final_load_balance_loss.item())
                ),
                "used_columns": support_audit["used_columns"],
                "dead_columns": support_audit["dead_columns"],
                "unique_support_sets": support_audit["unique_support_sets"],
                "max_column_fraction": support_audit["max_column_fraction"],
                "column_counts": support_audit["column_counts"],
                "support_set_counts": support_audit["support_set_counts"],
                "residual_parameter_delta": _state_dict_delta(
                    before_residual,
                    residual,
                ),
            }
        )

    baseline = _baseline_row(variant_rows)
    decision = _decision(
        baseline=baseline,
        variant_rows=variant_rows,
        ce_tolerance=ce_tolerance,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_variant_metrics(out_dir / "variant_metrics.csv", variant_rows, baseline)
    summary = {
        "status": "ok",
        "experiment_id": f"{experiment_id}_dead_column_probe",
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        **_load_torch_info(),
        "probe": {
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
            "ce_tolerance": ce_tolerance,
            "baseline_variant": baseline["variant"],
            "decision": decision,
            "variants": variant_rows,
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


def _router_load_balance_loss(residual: Any, hidden: Any) -> Any:
    import torch

    scores = residual._score_columns(hidden)
    probabilities = torch.softmax(scores, dim=-1)
    mean_probabilities = probabilities.reshape(-1, probabilities.shape[-1]).mean(dim=0)
    target = torch.full_like(mean_probabilities, 1.0 / mean_probabilities.numel())
    return mean_probabilities.numel() * (mean_probabilities - target).pow(2).sum()


def _variant_name(weight: float) -> str:
    return "baseline" if weight == 0.0 else f"load_balance_{weight:g}"


def _baseline_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if float(row["load_balance_weight"]) == 0.0:
            return row
    raise ValueError("baseline row missing")


def _decision(
    *,
    baseline: dict[str, Any],
    variant_rows: list[dict[str, Any]],
    ce_tolerance: float,
) -> dict[str, Any]:
    baseline_loss = float(baseline["alpha0_ce_loss"])
    baseline_used = int(baseline["used_columns"])
    candidates = [
        row
        for row in variant_rows
        if row is not baseline
        and float(row["alpha0_ce_loss"]) <= baseline_loss + ce_tolerance
        and int(row["used_columns"]) > baseline_used
    ]
    best_candidate = None
    if candidates:
        best_candidate = max(
            candidates,
            key=lambda row: (
                int(row["used_columns"]) - baseline_used,
                -float(row["alpha0_ce_loss"]),
            ),
        )
    return {
        "status": "recruited_without_ce_hurt" if best_candidate else "no_safe_recruitment",
        "criterion": (
            "variant must use more columns than baseline while alpha0 CE loss stays "
            f"within {ce_tolerance} of baseline"
        ),
        "baseline_alpha0_ce_loss": baseline_loss,
        "baseline_used_columns": baseline_used,
        "selected_variant": None if best_candidate is None else best_candidate["variant"],
        "selected_alpha0_ce_loss": (
            None if best_candidate is None else best_candidate["alpha0_ce_loss"]
        ),
        "selected_used_columns": (
            None if best_candidate is None else best_candidate["used_columns"]
        ),
        "safe_candidate_count": len(candidates),
    }


def _write_variant_metrics(
    path: Path,
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> None:
    fieldnames = [
        "variant",
        "load_balance_weight",
        "alpha0_ce_loss",
        "alpha0_ce_delta_from_baseline",
        "best_hep_alpha",
        "best_hep_loss",
        "used_columns",
        "used_column_delta_from_baseline",
        "dead_columns",
        "unique_support_sets",
        "max_column_fraction",
        "objective_loss",
        "load_balance_loss",
        "residual_parameter_delta",
    ]
    baseline_loss = float(baseline["alpha0_ce_loss"])
    baseline_used = int(baseline["used_columns"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "variant": row["variant"],
                    "load_balance_weight": row["load_balance_weight"],
                    "alpha0_ce_loss": row["alpha0_ce_loss"],
                    "alpha0_ce_delta_from_baseline": (
                        float(row["alpha0_ce_loss"]) - baseline_loss
                    ),
                    "best_hep_alpha": row["best_hep_alpha"],
                    "best_hep_loss": row["best_hep_loss"],
                    "used_columns": row["used_columns"],
                    "used_column_delta_from_baseline": (
                        int(row["used_columns"]) - baseline_used
                    ),
                    "dead_columns": row["dead_columns"],
                    "unique_support_sets": row["unique_support_sets"],
                    "max_column_fraction": row["max_column_fraction"],
                    "objective_loss": row["objective_loss"],
                    "load_balance_loss": row["load_balance_loss"],
                    "residual_parameter_delta": row["residual_parameter_delta"],
                }
            )


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    probe = summary["probe"]
    decision = probe["decision"]
    lines = [
        "# Dead-Column Recruitment Probe",
        "",
        f"- Experiment: `{summary['experiment_id']}`",
        f"- Config: `{summary['config_path']}`",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{decision['status']}`",
        f"- Baseline alpha-0 CE loss: `{decision['baseline_alpha0_ce_loss']}`",
        f"- Baseline used columns: `{decision['baseline_used_columns']}`",
        f"- CE tolerance: `{probe['ce_tolerance']}`",
        f"- Selected variant: `{decision['selected_variant']}`",
        "",
        "## Variants",
        "",
    ]
    for row in probe["variants"]:
        lines.append(
            "- "
            f"`{row['variant']}`: alpha-0 CE `{row['alpha0_ce_loss']}`, "
            f"used columns `{row['used_columns']}`, "
            f"dead columns `{row['dead_columns']}`, "
            f"unique support sets `{row['unique_support_sets']}`"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_weights(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


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
    parser.add_argument("--ce-tolerance", type=float, default=DEFAULT_CE_TOLERANCE)
    args = parser.parse_args(argv)
    summary = run_dead_column_probe(
        args.config,
        args.out,
        load_balance_weights=args.load_balance_weights,
        ce_tolerance=args.ce_tolerance,
    )
    print(json.dumps(summary["probe"]["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
