"""Deployable causal-router distillation audit after oracle-target regularization."""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.causal_contextual_router_regularization_probe import (
    _oracle_target_support_loss,
    _token_losses,
)
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
    "results/audits/token_larger_causal_contextual_router_distillation_audit"
)

DISTILLATION_CANDIDATE_FOUND = "causal_router_distillation_candidate_found"
DISTILLATION_NOT_ESTABLISHED = "causal_router_distillation_not_established"


def run_causal_contextual_router_distillation_audit(
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    max_folds: int | None = None,
    teacher_oracle_weight: float = 0.05,
    distill_weights: tuple[float, ...] = (0.01, 0.05),
    ce_guardrail: float = 0.05,
    random_seed: int = 2801,
) -> dict[str, Any]:
    """Audit causal-only students distilled from an oracle-target teacher policy."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("causal contextual router distillation audit requires torch") from exc

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
        raise ValueError("distillation audit expects model.columns.top_k: 2")
    if support_router != "contextual_mlp_causal":
        raise ValueError(
            "distillation audit expects model.columns.support_router: contextual_mlp_causal"
        )
    if max_folds is not None and max_folds < 1:
        raise ValueError("max_folds must be positive when set")
    if teacher_oracle_weight <= 0.0:
        raise ValueError("teacher_oracle_weight must be positive")
    if any(weight <= 0.0 for weight in distill_weights):
        raise ValueError("distill_weights must be positive")

    torch.manual_seed(seed)
    inputs, targets, vocab_size = _build_batch(dataset=dataset, seq_len=seq_len, batch_size=4)
    fold_count = int(inputs.shape[0]) if max_folds is None else min(max_folds, int(inputs.shape[0]))
    all_pairs = list(itertools.combinations(range(num_columns), top_k))
    fold_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    support_count_rows: list[dict[str, Any]] = []

    for fold_index in range(fold_count):
        train_indices = [index for index in range(int(inputs.shape[0])) if index != fold_index]
        train_inputs = inputs[train_indices]
        train_targets = targets[train_indices]
        holdout_inputs = inputs[fold_index : fold_index + 1]
        holdout_targets = targets[fold_index : fold_index + 1]
        torch.manual_seed(seed)
        teacher_base, teacher_residual = _train_residual(
            vocab_size=vocab_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            layers=layers,
            num_columns=num_columns,
            atoms_per_column=atoms_per_column,
            top_k=top_k,
            support_router="contextual_mlp_causal",
            contextual_router_hidden_dim=contextual_router_hidden_dim,
            learning_rate=learning_rate,
            max_steps=max_steps,
            residual_objective=residual_objective,
            training_cfg=training_cfg,
            train_inputs=train_inputs,
            train_targets=train_targets,
            all_pairs=all_pairs,
            oracle_target_weight=teacher_oracle_weight,
        )
        teacher_residual.eval()
        with torch.no_grad():
            teacher_hidden = teacher_base.encode(train_inputs)
            _, teacher_train_support = teacher_residual(teacher_hidden, return_support=True)

        specs = _distillation_specs(
            distill_weights=distill_weights,
            teacher_oracle_weight=teacher_oracle_weight,
        )
        for spec in specs:
            torch.manual_seed(seed)
            if spec["variant_kind"] == "teacher_oracle_target":
                base, residual = _train_residual(
                    vocab_size=vocab_size,
                    seq_len=seq_len,
                    hidden_dim=hidden_dim,
                    layers=layers,
                    num_columns=num_columns,
                    atoms_per_column=atoms_per_column,
                    top_k=top_k,
                    support_router="contextual_mlp_causal",
                    contextual_router_hidden_dim=contextual_router_hidden_dim,
                    learning_rate=learning_rate,
                    max_steps=max_steps,
                    residual_objective=residual_objective,
                    training_cfg=training_cfg,
                    train_inputs=train_inputs,
                    train_targets=train_targets,
                    all_pairs=all_pairs,
                    oracle_target_weight=teacher_oracle_weight,
                )
            else:
                base, residual = _train_residual(
                    vocab_size=vocab_size,
                    seq_len=seq_len,
                    hidden_dim=hidden_dim,
                    layers=layers,
                    num_columns=num_columns,
                    atoms_per_column=atoms_per_column,
                    top_k=spec["top_k"],
                    support_router=spec["support_router"],
                    contextual_router_hidden_dim=contextual_router_hidden_dim,
                    learning_rate=learning_rate,
                    max_steps=max_steps,
                    residual_objective=residual_objective,
                    training_cfg=training_cfg,
                    train_inputs=train_inputs,
                    train_targets=train_targets,
                    all_pairs=all_pairs,
                    teacher_support=teacher_train_support,
                    distill_weight=spec["distill_weight"],
                )
            row_bundle = _evaluate_fold_variant(
                base=base,
                residual=residual,
                holdout_inputs=holdout_inputs,
                holdout_targets=holdout_targets,
                vocab_size=vocab_size,
                all_pairs=all_pairs,
                fold_index=fold_index,
                spec=spec,
                random_seed=random_seed,
            )
            fold_rows.append(row_bundle["fold_row"])
            control_rows.extend(row_bundle["control_rows"])
            support_count_rows.extend(row_bundle["support_count_rows"])

    aggregate_rows = _aggregate_rows(fold_rows)
    distillation_rows = _distillation_gate_rows(aggregate_rows, ce_guardrail=ce_guardrail)
    decision = _decision(distillation_rows)
    summary = {
        "status": "pass",
        "decision": decision["decision"],
        "claim_status": decision["claim_status"],
        "selected_next_step": decision["next_step"],
        "experiment_id": f"{experiment_id}_distillation_audit",
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        **_load_torch_info(),
        "audit": {
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
            "support_set_count": len(all_pairs),
            "teacher_oracle_weight": teacher_oracle_weight,
            "distill_weights": list(distill_weights),
            "ce_guardrail": ce_guardrail,
            "fold_metrics": fold_rows,
            "aggregate_metrics": {row["control"]: row for row in aggregate_rows},
            "distillation_gate_rows": distillation_rows,
            "failures": decision["failures"],
            "rationale": decision["rationale"],
        },
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "fold_metrics_csv": str(out_dir / "fold_metrics.csv"),
            "aggregate_metrics_csv": str(out_dir / "aggregate_metrics.csv"),
            "distillation_gate_csv": str(out_dir / "distillation_gate.csv"),
            "control_metrics_csv": str(out_dir / "control_metrics.csv"),
            "support_counts_csv": str(out_dir / "support_counts.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
        "git_commit": _git_commit(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "fold_metrics.csv", fold_rows)
    _write_csv(out_dir / "aggregate_metrics.csv", aggregate_rows)
    _write_csv(out_dir / "distillation_gate.csv", distillation_rows)
    _write_csv(out_dir / "control_metrics.csv", control_rows)
    _write_csv(out_dir / "support_counts.csv", support_count_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _make_base(*, vocab_size: int, seq_len: int, hidden_dim: int, layers: int) -> Any:
    base = TinyCharTransformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        layers=layers,
    )
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    return base


def _train_residual(
    *,
    vocab_size: int,
    seq_len: int,
    hidden_dim: int,
    layers: int,
    num_columns: int,
    atoms_per_column: int,
    top_k: int,
    support_router: str,
    contextual_router_hidden_dim: int,
    learning_rate: float,
    max_steps: int,
    residual_objective: str,
    training_cfg: dict[str, Any],
    train_inputs: Any,
    train_targets: Any,
    all_pairs: list[tuple[int, ...]],
    oracle_target_weight: float = 0.0,
    teacher_support: Any | None = None,
    distill_weight: float = 0.0,
) -> tuple[Any, Any]:
    import torch

    base = _make_base(
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
        loss = ce_loss
        if oracle_target_weight:
            loss = loss + oracle_target_weight * _oracle_target_support_loss(
                base=base,
                residual=residual,
                hidden=hidden,
                targets=train_targets,
                vocab_size=vocab_size,
                all_pairs=all_pairs,
            )
        if teacher_support is not None and distill_weight:
            loss = loss + distill_weight * _support_distillation_loss(
                residual=residual,
                hidden=hidden,
                teacher_support=teacher_support,
            )
        loss.backward()
        optimizer.step()
    residual.eval()
    return base, residual


def _support_distillation_loss(*, residual: Any, hidden: Any, teacher_support: Any) -> Any:
    import torch
    import torch.nn.functional as F

    target = torch.zeros(
        hidden.shape[0],
        hidden.shape[1],
        residual.num_columns,
        dtype=hidden.dtype,
        device=hidden.device,
    )
    target.scatter_(dim=-1, index=teacher_support.to(device=hidden.device), value=1.0)
    scores = residual._score_columns(hidden)
    return F.binary_cross_entropy_with_logits(
        scores[:, :-1, :].reshape(-1, residual.num_columns),
        target[:, :-1, :].reshape(-1, residual.num_columns),
    )


def _distillation_specs(
    *,
    distill_weights: tuple[float, ...],
    teacher_oracle_weight: float,
) -> list[dict[str, Any]]:
    specs = [
        {
            **spec,
            "variant_kind": "control",
            "teacher_oracle_weight": 0.0,
            "distill_weight": 0.0,
            "distills_from_teacher": False,
        }
        for spec in _control_specs(contextual_router_hidden_dim=0)
        if spec["name"] in {"causal_contextual_topk2", "linear_topk2"}
    ]
    specs.append(
        {
            "name": f"oracle_target_teacher_{teacher_oracle_weight:g}",
            "support_router": "contextual_mlp_causal",
            "top_k": 2,
            "causal_feature_safe": True,
            "variant_kind": "teacher_oracle_target",
            "teacher_oracle_weight": teacher_oracle_weight,
            "distill_weight": 0.0,
            "distills_from_teacher": False,
        }
    )
    for weight in distill_weights:
        specs.append(
            {
                "name": f"causal_distilled_from_oracle_target_{weight:g}",
                "support_router": "contextual_mlp_causal",
                "top_k": 2,
                "causal_feature_safe": True,
                "variant_kind": "distilled_student",
                "teacher_oracle_weight": teacher_oracle_weight,
                "distill_weight": float(weight),
                "distills_from_teacher": True,
            }
        )
    return specs


def _evaluate_fold_variant(
    *,
    base: Any,
    residual: Any,
    holdout_inputs: Any,
    holdout_targets: Any,
    vocab_size: int,
    all_pairs: list[tuple[int, ...]],
    fold_index: int,
    spec: dict[str, Any],
    random_seed: int,
) -> dict[str, Any]:
    with _no_grad():
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
        "variant_kind": spec["variant_kind"],
        "teacher_oracle_weight": spec["teacher_oracle_weight"],
        "distill_weight": spec["distill_weight"],
        "distills_from_teacher": spec["distills_from_teacher"],
        "support_router": spec["support_router"],
        "top_k": spec["top_k"],
        "causal_feature_safe": spec["causal_feature_safe"],
        "positions": int(router_token_losses.numel()),
        **audit["metrics"],
    }
    return {
        "fold_row": fold_row,
        "control_rows": [
            {"fold": fold_index, "control": spec["name"], **item}
            for item in audit["controls"]
        ],
        "support_count_rows": [
            {
                "fold": fold_index,
                "control": spec["name"],
                "support": key,
                "count": count,
            }
            for key, count in audit["support_counts"].items()
        ],
    }


class _no_grad:
    def __enter__(self) -> None:
        import torch

        self._context = torch.no_grad()
        self._context.__enter__()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool | None:
        return self._context.__exit__(exc_type, exc, tb)


def _distillation_gate_rows(
    aggregate_rows: list[dict[str, Any]], *, ce_guardrail: float
) -> list[dict[str, Any]]:
    by_control = {row["control"]: row for row in aggregate_rows}
    causal = by_control.get("causal_contextual_topk2", {})
    linear = by_control.get("linear_topk2", {})
    teacher = next(
        (row for row in aggregate_rows if row["control"].startswith("oracle_target_teacher_")),
        {},
    )
    rows = []
    for row in aggregate_rows:
        if not row["control"].startswith("causal_distilled_from_oracle_target_"):
            continue
        ce_delta = _delta(row, causal, "mean_router_loss")
        regret_delta = _delta(row, causal, "mean_oracle_support_regret")
        churn_delta = _delta(row, causal, "mean_functional_churn_logit_l1")
        linear_ce_delta = _delta(row, linear, "mean_router_loss")
        teacher_regret_delta = _delta(row, teacher, "mean_oracle_support_regret")
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
        rows.append(
            {
                "variant": row["control"],
                "mean_router_loss": row.get("mean_router_loss"),
                "mean_router_loss_delta_vs_causal": ce_delta,
                "mean_router_loss_delta_vs_linear": linear_ce_delta,
                "mean_oracle_support_regret": row.get("mean_oracle_support_regret"),
                "mean_oracle_regret_delta_vs_causal": regret_delta,
                "mean_oracle_regret_delta_vs_teacher": teacher_regret_delta,
                "mean_functional_churn_logit_l1": row.get("mean_functional_churn_logit_l1"),
                "mean_functional_churn_delta_vs_causal": churn_delta,
                "ce_guardrail": ce_guardrail,
                "passes_distillation_gate": qualifies,
            }
        )
    return rows


def _decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("passes_distillation_gate")]
    if candidates:
        best = min(candidates, key=lambda row: float(row["mean_oracle_regret_delta_vs_causal"]))
        return {
            "decision": DISTILLATION_CANDIDATE_FOUND,
            "claim_status": "deployable_distilled_causal_router_candidate_not_promoted",
            "next_step": (
                "repeat the distilled causal-router candidate on the full backend audit "
                "and inspect teacher-student support agreement before any default promotion"
            ),
            "failures": [],
            "rationale": (
                f"Variant `{best['variant']}` reduced oracle-support regret and "
                "functional churn versus the unregularized causal router while "
                "staying inside the CE guardrail and still beating linear CE."
            ),
        }
    return {
        "decision": DISTILLATION_NOT_ESTABLISHED,
        "claim_status": "deployable_distilled_causal_router_support_quality_not_established",
        "next_step": (
            "inspect whether teacher policy mismatch or student capacity prevents "
            "causal-only distillation before changing defaults"
        ),
        "failures": [row for row in rows if not row.get("passes_distillation_gate")],
        "rationale": (
            "No causal-only distilled student jointly reduced oracle-support regret "
            "and functional churn versus the unregularized causal router while "
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
    audit = summary["audit"]
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "Deployable causal-router distillation audit.",
        "",
        f"- Config: `{summary['config_path']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Folds: `{audit['fold_count']}`",
        f"- Teacher oracle-target weight: `{audit['teacher_oracle_weight']}`",
        f"- Rationale: {audit['rationale']}",
        "",
        "## Distillation Gate",
    ]
    for row in audit["distillation_gate_rows"]:
        lines.append(
            "- "
            f"{row['variant']}: pass `{row['passes_distillation_gate']}`, "
            f"CE delta vs causal `{row['mean_router_loss_delta_vs_causal']}`, "
            f"oracle-regret delta `{row['mean_oracle_regret_delta_vs_causal']}`, "
            f"functional-churn delta `{row['mean_functional_churn_delta_vs_causal']}`"
        )
    lines.extend(["", "## Aggregate Metrics"])
    for row in sorted(audit["aggregate_metrics"].values(), key=lambda item: item["control"]):
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
    parser.add_argument("--teacher-oracle-weight", type=float, default=0.05)
    parser.add_argument("--distill-weight", type=float, action="append", dest="distill_weights")
    parser.add_argument("--ce-guardrail", type=float, default=0.05)
    args = parser.parse_args(argv)
    distill_weights = tuple(args.distill_weights) if args.distill_weights else (0.01, 0.05)
    summary = run_causal_contextual_router_distillation_audit(
        args.config,
        args.out,
        max_folds=args.max_folds,
        teacher_oracle_weight=args.teacher_oracle_weight,
        distill_weights=distill_weights,
        ce_guardrail=args.ce_guardrail,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
