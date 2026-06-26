"""Support-stability regularization probe for causal contextual top-k-2 routing."""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.causal_contextual_router_support_audit import (
    DEFAULT_CONFIG,
    _aggregate_rows,
    _control_specs,
    _git_commit,
    _support_quality_metrics,
    _write_csv,
)
from relaleap.experiments.contextual_router_sequence_kfold_ablation import (
    _forward_with_feature_ablation,
)
from relaleap.experiments.run import _load_torch_info
from relaleap.experiments.run import _read_config
from relaleap.experiments.support_audit import _ce_loss
from relaleap.experiments.support_audit import _configured_residual_loss
from relaleap.experiments.support_audit import _score_for_support
from relaleap.smoke import ResidualColumns
from relaleap.smoke import TinyCharTransformer
from relaleap.smoke import _build_batch


DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_causal_contextual_router_regularization_probe"
)

REGULARIZATION_CANDIDATE_FOUND = "causal_router_regularization_candidate_found"
REGULARIZATION_NOT_ESTABLISHED = "causal_router_regularization_not_established"


def run_causal_contextual_router_regularization_probe(
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    max_folds: int | None = None,
    smooth_weights: tuple[float, ...] = (0.01, 0.05),
    ce_guardrail: float = 0.05,
    random_seed: int = 2606,
) -> dict[str, Any]:
    """Train causal folds with router-score smoothness and audit support quality."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("causal contextual router regularization probe requires torch") from exc

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
        raise ValueError("regularization probe expects model.columns.top_k: 2")
    if support_router != "contextual_mlp_causal":
        raise ValueError(
            "regularization probe expects model.columns.support_router: contextual_mlp_causal"
        )
    if max_folds is not None and max_folds < 1:
        raise ValueError("max_folds must be positive when set")
    if any(weight < 0.0 for weight in smooth_weights):
        raise ValueError("smooth_weights must be non-negative")

    torch.manual_seed(seed)
    inputs, targets, vocab_size = _build_batch(dataset=dataset, seq_len=seq_len, batch_size=4)
    fold_count = int(inputs.shape[0]) if max_folds is None else min(max_folds, int(inputs.shape[0]))
    all_pairs = list(itertools.combinations(range(num_columns), top_k))
    variants = _variant_specs(smooth_weights)
    fold_rows: list[dict[str, Any]] = []
    support_count_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []

    for fold_index in range(fold_count):
        train_indices = [index for index in range(int(inputs.shape[0])) if index != fold_index]
        train_inputs = inputs[train_indices]
        train_targets = targets[train_indices]
        holdout_inputs = inputs[fold_index : fold_index + 1]
        holdout_targets = targets[fold_index : fold_index + 1]
        for spec in variants:
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
            residual = ResidualColumns(
                hidden_dim=hidden_dim,
                num_columns=num_columns,
                atoms_per_column=atoms_per_column,
                top_k=spec["top_k"],
                support_router=spec["support_router"],
                contextual_router_hidden_dim=contextual_router_hidden_dim,
            )
            optimizer = torch.optim.AdamW(residual.parameters(), lr=learning_rate)
            residual.train()
            for _ in range(max_steps):
                optimizer.zero_grad(set_to_none=True)
                ce_loss = _configured_residual_loss(
                    base,
                    residual,
                    train_inputs,
                    train_targets,
                    vocab_size,
                    residual_objective=residual_objective,
                    training_cfg=training_cfg,
                )
                hidden = base.encode(train_inputs)
                loss = ce_loss + float(spec["score_smooth_weight"]) * _score_smoothness_loss(
                    residual,
                    hidden,
                )
                loss.backward()
                optimizer.step()

            residual.eval()
            with torch.no_grad():
                hidden = base.encode(holdout_inputs)
                logits, support = _forward_with_feature_ablation(
                    base,
                    residual,
                    hidden,
                    feature_mask=None,
                )
                router_token_losses = _token_losses(logits, holdout_targets).reshape(-1)
                router_loss = float(router_token_losses.mean().item())
                empty_logits = base.decode(hidden)
                empty_loss = _ce_loss(empty_logits, holdout_targets, vocab_size)
                pair_rows = [
                    _score_for_support(
                        base,
                        residual,
                        hidden,
                        holdout_targets,
                        vocab_size,
                        support=pair,
                        empty_loss=empty_loss,
                        router_loss=router_loss,
                    )
                    for pair in all_pairs
                ]
                audit = _support_quality_metrics(
                    base=base,
                    residual=residual,
                    hidden=hidden,
                    targets=holdout_targets,
                    vocab_size=vocab_size,
                    support=support,
                    router_token_losses=router_token_losses,
                    router_loss=router_loss,
                    empty_loss=empty_loss,
                    pair_rows=pair_rows,
                    random_seed=random_seed + fold_index,
                )
            fold_row = {
                "fold": fold_index,
                "heldout_sequence_index": fold_index,
                "control": spec["name"],
                "variant": spec["name"],
                "regularizer": spec["regularizer"],
                "score_smooth_weight": spec["score_smooth_weight"],
                "support_router": spec["support_router"],
                "top_k": spec["top_k"],
                "causal_feature_safe": spec["causal_feature_safe"],
                "positions": int(router_token_losses.numel()),
                **audit["metrics"],
            }
            fold_rows.append(fold_row)
            control_rows.extend(
                {"fold": fold_index, "control": spec["name"], **item}
                for item in audit["controls"]
            )
            support_count_rows.extend(
                {
                    "fold": fold_index,
                    "control": spec["name"],
                    "support": key,
                    "count": count,
                }
                for key, count in audit["support_counts"].items()
            )

    aggregate_rows = _aggregate_rows(fold_rows)
    variant_rows = _regularized_variant_rows(aggregate_rows, ce_guardrail=ce_guardrail)
    decision = _decision(variant_rows)
    summary = {
        "status": "pass",
        "decision": decision["decision"],
        "claim_status": decision["claim_status"],
        "selected_next_step": decision["next_step"],
        "experiment_id": f"{experiment_id}_regularization_probe",
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        **_load_torch_info(),
        "probe": {
            "dataset": dataset,
            "seq_len": seq_len,
            "batch_size": int(inputs.shape[0]),
            "fold_count": fold_count,
            "vocab_size": vocab_size,
            "training_steps": max_steps,
            "residual_objective": residual_objective,
            "num_columns": num_columns,
            "atoms_per_column": atoms_per_column,
            "top_k": top_k,
            "support_router": support_router,
            "ce_guardrail": ce_guardrail,
            "variants": variants,
            "fold_metrics": fold_rows,
            "aggregate_metrics": {row["control"]: row for row in aggregate_rows},
            "variant_gate_rows": variant_rows,
            "failures": decision["failures"],
            "rationale": decision["rationale"],
        },
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "fold_metrics_csv": str(out_dir / "fold_metrics.csv"),
            "aggregate_metrics_csv": str(out_dir / "aggregate_metrics.csv"),
            "variant_gate_csv": str(out_dir / "variant_gate.csv"),
            "control_metrics_csv": str(out_dir / "control_metrics.csv"),
            "support_counts_csv": str(out_dir / "support_counts.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
        "git_commit": _git_commit(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "fold_metrics.csv", fold_rows)
    _write_csv(out_dir / "aggregate_metrics.csv", aggregate_rows)
    _write_csv(out_dir / "variant_gate.csv", variant_rows)
    _write_csv(out_dir / "control_metrics.csv", control_rows)
    _write_csv(out_dir / "support_counts.csv", support_count_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _variant_specs(smooth_weights: tuple[float, ...]) -> list[dict[str, Any]]:
    specs = _control_specs(contextual_router_hidden_dim=0)
    out = [
        {**spec, "regularizer": "none", "score_smooth_weight": 0.0}
        for spec in specs
        if spec["name"] in {"causal_contextual_topk2", "linear_topk2"}
    ]
    for weight in smooth_weights:
        out.append(
            {
                "name": f"causal_contextual_score_smooth_{weight:g}",
                "support_router": "contextual_mlp_causal",
                "top_k": 2,
                "causal_feature_safe": True,
                "regularizer": "adjacent_router_score_distribution_l2",
                "score_smooth_weight": float(weight),
            }
        )
    return out


def _score_smoothness_loss(residual: Any, hidden: Any) -> Any:
    import torch.nn.functional as F

    scores = residual._score_columns(hidden)
    probabilities = F.softmax(scores, dim=-1)
    if probabilities.shape[1] < 2:
        return probabilities.sum() * 0.0
    delta = probabilities[:, 1:, :] - probabilities[:, :-1, :]
    return delta.pow(2).mean()


def _token_losses(logits: Any, targets: Any) -> Any:
    import torch.nn.functional as F

    vocab_size = int(logits.shape[-1])
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab_size),
        targets[:, :-1].reshape(-1),
        reduction="none",
    )


def _regularized_variant_rows(
    aggregate_rows: list[dict[str, Any]], *, ce_guardrail: float
) -> list[dict[str, Any]]:
    by_control = {row["control"]: row for row in aggregate_rows}
    baseline = by_control.get("causal_contextual_topk2", {})
    linear = by_control.get("linear_topk2", {})
    out = []
    for row in aggregate_rows:
        if row["control"] in {"causal_contextual_topk2", "linear_topk2"}:
            continue
        ce_delta = _delta(row, baseline, "mean_router_loss")
        regret_delta = _delta(row, baseline, "mean_oracle_support_regret")
        churn_delta = _delta(row, baseline, "mean_functional_churn_logit_l1")
        linear_ce_delta = _delta(row, linear, "mean_router_loss")
        qualifies = (
            ce_delta is not None
            and ce_delta <= ce_guardrail
            and regret_delta is not None
            and regret_delta < 0.0
            and churn_delta is not None
            and churn_delta < 0.0
            and linear_ce_delta is not None
            and linear_ce_delta < 0.0
        )
        out.append(
            {
                "variant": row["control"],
                "mean_router_loss": row.get("mean_router_loss"),
                "mean_router_loss_delta_vs_causal": ce_delta,
                "mean_router_loss_delta_vs_linear": linear_ce_delta,
                "mean_oracle_support_regret": row.get("mean_oracle_support_regret"),
                "mean_oracle_regret_delta_vs_causal": regret_delta,
                "mean_functional_churn_logit_l1": row.get("mean_functional_churn_logit_l1"),
                "mean_functional_churn_delta_vs_causal": churn_delta,
                "ce_guardrail": ce_guardrail,
                "passes_regularization_gate": qualifies,
            }
        )
    return out


def _decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("passes_regularization_gate")]
    if candidates:
        best = min(
            candidates,
            key=lambda row: float(row["mean_oracle_regret_delta_vs_causal"]),
        )
        return {
            "decision": REGULARIZATION_CANDIDATE_FOUND,
            "claim_status": "regularized_causal_router_candidate_not_promoted",
            "next_step": (
                "validate the selected causal-router regularization candidate on RunPod "
                "against the full causal support audit before any default change"
            ),
            "failures": [],
            "rationale": (
                f"Variant `{best['variant']}` reduced oracle-support regret and "
                "functional churn versus the unregularized causal router while "
                "staying inside the CE guardrail and still beating linear CE."
            ),
        }
    return {
        "decision": REGULARIZATION_NOT_ESTABLISHED,
        "claim_status": "causal_router_regularization_support_quality_not_established",
        "next_step": (
            "try a more targeted oracle-support or support-load regularizer, or "
            "return to causal-feature design before any default promotion"
        ),
        "failures": [row for row in rows if not row.get("passes_regularization_gate")],
        "rationale": (
            "No score-smoothness regularization variant jointly reduced oracle-support "
            "regret and functional churn versus the unregularized causal router while "
            "preserving the CE guardrail."
        ),
    }


def _delta(left: dict[str, Any], right: dict[str, Any], field: str) -> float | None:
    left_value = left.get(field)
    right_value = right.get(field)
    if left_value is None or right_value is None:
        return None
    return float(left_value) - float(right_value)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    probe = summary["probe"]
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "Causal contextual-router support-stability regularization probe.",
        "",
        f"- Config: `{summary['config_path']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Folds: `{probe['fold_count']}`",
        f"- Rationale: {probe['rationale']}",
        "",
        "## Variant Gate",
    ]
    for row in probe["variant_gate_rows"]:
        lines.append(
            "- "
            f"{row['variant']}: pass `{row['passes_regularization_gate']}`, "
            f"CE delta vs causal `{row['mean_router_loss_delta_vs_causal']}`, "
            f"oracle-regret delta `{row['mean_oracle_regret_delta_vs_causal']}`, "
            f"functional-churn delta `{row['mean_functional_churn_delta_vs_causal']}`"
        )
    lines.extend(["", "## Aggregate Metrics"])
    for row in sorted(probe["aggregate_metrics"].values(), key=lambda item: item["control"]):
        lines.append(
            "- "
            f"{row['control']}: CE `{row['mean_router_loss']}`, "
            f"oracle regret `{row['mean_oracle_support_regret']}`, "
            f"functional churn `{row['mean_functional_churn_logit_l1']}`"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--smooth-weight", type=float, action="append", dest="smooth_weights")
    parser.add_argument("--ce-guardrail", type=float, default=0.05)
    args = parser.parse_args(argv)
    smooth_weights = tuple(args.smooth_weights) if args.smooth_weights else (0.01, 0.05)
    summary = run_causal_contextual_router_regularization_probe(
        args.config,
        args.out,
        max_folds=args.max_folds,
        smooth_weights=smooth_weights,
        ce_guardrail=args.ce_guardrail,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
