"""Teacher-student support agreement audit for causal-router distillation."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.causal_contextual_router_distillation_audit import (
    DEFAULT_CONFIG,
    DEFAULT_OUT_DIR as DEFAULT_DISTILLATION_AUDIT_DIR,
    _distillation_specs,
    _train_residual,
)
from relaleap.experiments.causal_contextual_router_support_audit import (
    _aggregate_rows,
    _git_commit,
    _normalized_load_entropy,
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
from relaleap.experiments.support_audit import _support_key
from relaleap.experiments.support_audit import _token_losses
from relaleap.smoke import ResidualColumns
from relaleap.smoke import TinyCharTransformer
from relaleap.smoke import _build_batch


DEFAULT_RUNPOD_DISTILLATION_AUDIT_DIR = Path(
    "results/runpod_fetch/audits/runpod_token_larger_causal_contextual_router_distillation_audit"
)
DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_causal_contextual_router_distillation_agreement"
)

AGREEMENT_SUPPORTED = "teacher_student_support_agreement_intervention_supported"
AGREEMENT_BLOCKED = "teacher_student_support_agreement_intervention_blocks_promotion"


def run_causal_contextual_router_distillation_agreement_audit(
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    audit_dir: Path = DEFAULT_DISTILLATION_AUDIT_DIR,
    runpod_audit_dir: Path | None = DEFAULT_RUNPOD_DISTILLATION_AUDIT_DIR,
    max_folds: int | None = None,
    teacher_oracle_weight: float = 0.05,
    student_distill_weight: float = 0.05,
    ce_guardrail: float = 0.05,
    random_seed: int = 3901,
    batch_size: int = 4,
    capture_hidden_future: bool = False,
    capture_train_hidden_future: bool = False,
) -> dict[str, Any]:
    """Rerun the bounded distillation setup with per-token support logging."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("distillation agreement audit requires torch") from exc

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
        raise ValueError("agreement audit expects model.columns.top_k: 2")
    if support_router != "contextual_mlp_causal":
        raise ValueError(
            "agreement audit expects model.columns.support_router: contextual_mlp_causal"
        )
    if max_folds is not None and max_folds < 1:
        raise ValueError("max_folds must be positive when set")
    if teacher_oracle_weight <= 0.0:
        raise ValueError("teacher_oracle_weight must be positive")
    if student_distill_weight <= 0.0:
        raise ValueError("student_distill_weight must be positive")
    if capture_train_hidden_future and not capture_hidden_future:
        raise ValueError("capture_train_hidden_future requires capture_hidden_future")
    if capture_train_hidden_future and max_folds != 1:
        raise ValueError(
            "capture_train_hidden_future requires max_folds=1 to avoid cross-fold leakage"
        )
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2")

    source_assessment = _source_artifact_assessment(audit_dir, runpod_audit_dir)
    torch.manual_seed(seed)
    inputs, targets, vocab_size = _build_batch(
        dataset=dataset,
        seq_len=seq_len,
        batch_size=batch_size,
    )
    fold_count = int(inputs.shape[0]) if max_folds is None else min(max_folds, int(inputs.shape[0]))
    all_pairs = list(itertools.combinations(range(num_columns), top_k))
    fold_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    null_control_rows: list[dict[str, Any]] = []
    null_sampling_rows: list[dict[str, Any]] = []
    per_token_rows: list[dict[str, Any]] = []
    hidden_future_rows: list[dict[str, Any]] = []
    exact_intervention_rows: list[dict[str, Any]] = []
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
        with torch.no_grad():
            teacher_train_hidden = teacher_base.encode(train_inputs)
            _, teacher_train_support = teacher_residual(
                teacher_train_hidden,
                return_support=True,
            )
        shuffled_teacher_train_support = _shuffle_support(
            teacher_train_support,
            seed=random_seed + 503 + fold_index,
        )
        frequency_matched_teacher_train_support = _frequency_matched_support(
            teacher_train_support,
            seed=random_seed + 907 + fold_index,
        )
        (
            token_position_frequency_matched_teacher_train_support,
            token_position_sampling_rows,
        ) = _token_position_frequency_matched_support(
            teacher_train_support,
            train_targets,
            seed=random_seed + 1301 + fold_index,
        )
        for row in token_position_sampling_rows:
            null_sampling_rows.append(
                {
                    "fold": fold_index,
                    "null_control": (
                        "causal_distilled_from_token_position_frequency_matched_teacher_"
                        f"{student_distill_weight:g}"
                    ),
                    **row,
                }
            )

        variants: dict[str, tuple[Any, Any, dict[str, Any]]] = {}
        specs = [
            spec
            for spec in _distillation_specs(
                distill_weights=(student_distill_weight,),
                teacher_oracle_weight=teacher_oracle_weight,
            )
            if spec["name"]
            in {
                "causal_contextual_topk2",
                "linear_topk2",
                f"causal_distilled_from_oracle_target_{student_distill_weight:g}",
            }
        ]
        for spec in specs:
            torch.manual_seed(seed)
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
                teacher_support=teacher_train_support if spec["distills_from_teacher"] else None,
                distill_weight=spec["distill_weight"],
            )
            variants[spec["name"]] = (base, residual, spec)
        null_specs = [
            (
                {
                    "name": f"causal_distilled_from_shuffled_teacher_{student_distill_weight:g}",
                    "support_router": "contextual_mlp_causal",
                    "top_k": 2,
                    "causal_feature_safe": True,
                    "variant_kind": "shuffled_teacher_null",
                    "teacher_oracle_weight": teacher_oracle_weight,
                    "distill_weight": float(student_distill_weight),
                    "distills_from_teacher": False,
                    "null_control": "shuffled_teacher",
                },
                shuffled_teacher_train_support,
            ),
            (
                {
                    "name": f"causal_distilled_from_frequency_matched_teacher_{student_distill_weight:g}",
                    "support_router": "contextual_mlp_causal",
                    "top_k": 2,
                    "causal_feature_safe": True,
                    "variant_kind": "frequency_matched_teacher_null",
                    "teacher_oracle_weight": teacher_oracle_weight,
                    "distill_weight": float(student_distill_weight),
                    "distills_from_teacher": False,
                    "null_control": "frequency_matched_teacher",
                },
                frequency_matched_teacher_train_support,
            ),
            (
                {
                    "name": (
                        "causal_distilled_from_token_position_frequency_matched_teacher_"
                        f"{student_distill_weight:g}"
                    ),
                    "support_router": "contextual_mlp_causal",
                    "top_k": 2,
                    "causal_feature_safe": True,
                    "variant_kind": "token_position_frequency_matched_teacher_null",
                    "teacher_oracle_weight": teacher_oracle_weight,
                    "distill_weight": float(student_distill_weight),
                    "distills_from_teacher": False,
                    "null_control": "token_position_frequency_matched_teacher",
                },
                token_position_frequency_matched_teacher_train_support,
            ),
        ]
        for spec, null_support in null_specs:
            torch.manual_seed(seed)
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
                teacher_support=null_support,
                distill_weight=spec["distill_weight"],
            )
            variants[spec["name"]] = (base, residual, spec)

        student_name = f"causal_distilled_from_oracle_target_{student_distill_weight:g}"
        student_base, student_residual, student_spec = variants[student_name]
        linear_base, linear_residual, _ = variants["linear_topk2"]
        causal_base, causal_residual, causal_spec = variants["causal_contextual_topk2"]
        null_names = [spec["name"] for spec, _ in null_specs]

        with torch.no_grad():
            student_hidden = student_base.encode(holdout_inputs)
            teacher_hidden = teacher_base.encode(holdout_inputs)
            linear_hidden = linear_base.encode(holdout_inputs)
            causal_hidden = causal_base.encode(holdout_inputs)
            teacher_logits, teacher_support = _forward_with_feature_ablation(
                teacher_base,
                teacher_residual,
                teacher_hidden,
                feature_mask=None,
            )
            teacher_scores = (
                teacher_residual._score_columns(teacher_hidden)
                + teacher_residual.score_tie_breaker.to(
                    device=teacher_hidden.device,
                    dtype=teacher_hidden.dtype,
                )
            )
            student_logits, student_support = _forward_with_feature_ablation(
                student_base,
                student_residual,
                student_hidden,
                feature_mask=None,
            )
            _, linear_support = _forward_with_feature_ablation(
                linear_base,
                linear_residual,
                linear_hidden,
                feature_mask=None,
            )
            causal_logits, causal_support = _forward_with_feature_ablation(
                causal_base,
                causal_residual,
                causal_hidden,
                feature_mask=None,
            )
            student_token_losses = _token_losses(student_logits, holdout_targets).reshape(-1)
            teacher_token_losses = _token_losses(teacher_logits, holdout_targets).reshape(-1)
            causal_token_losses = _token_losses(causal_logits, holdout_targets).reshape(-1)
            empty_logits = student_base.decode(student_hidden)
            empty_loss = _ce_loss(empty_logits, holdout_targets, vocab_size)
            student_router_loss = float(student_token_losses.mean().item())
            pair_rows = [
                _score_for_support(
                    student_base,
                    student_residual,
                    student_hidden,
                    holdout_targets,
                    vocab_size,
                    support=pair,
                    empty_loss=empty_loss,
                    router_loss=student_router_loss,
                )
                for pair in all_pairs
            ]
            pair_loss_matrix = torch.stack(
                [row["_token_losses"].reshape(-1) for row in pair_rows],
                dim=1,
            )
            oracle_losses, oracle_indices = pair_loss_matrix.min(dim=1)
            teacher_token_support = teacher_support[:, :-1, :]
            student_token_support = student_support[:, :-1, :]
            linear_token_support = linear_support[:, :-1, :]
            causal_token_support = causal_support[:, :-1, :]
            oracle_token_support = _support_from_pair_indices(
                oracle_indices,
                all_pairs,
                like=student_token_support,
            )
            oracle_support = _with_last_support(oracle_token_support)
            shuffled_support = _shuffle_support(student_support, seed=random_seed + fold_index)
            random_support = _random_support(
                all_pairs,
                like=student_support,
                seed=random_seed + 101 + fold_index,
            )
            forced_losses = {
                "student_router_support": student_token_losses,
                "teacher_support_forced_into_student": _forced_token_losses(
                    student_base,
                    student_residual,
                    student_hidden,
                    holdout_targets,
                    support=teacher_support,
                ),
                "oracle_best_support_for_student": oracle_losses,
                "linear_support_forced_into_student": _forced_token_losses(
                    student_base,
                    student_residual,
                    student_hidden,
                    holdout_targets,
                    support=linear_support,
                ),
                "marginal_shuffled_student_support": _forced_token_losses(
                    student_base,
                    student_residual,
                    student_hidden,
                    holdout_targets,
                    support=shuffled_support,
                ),
                "uniform_random_support": _forced_token_losses(
                    student_base,
                    student_residual,
                    student_hidden,
                    holdout_targets,
                    support=random_support,
                ),
            }
            null_eval_rows = []
            null_supports = {}
            token_position_null_support = None
            for null_name in null_names:
                null_base, null_residual, null_spec = variants[null_name]
                null_hidden = null_base.encode(holdout_inputs)
                null_logits, null_support = _forward_with_feature_ablation(
                    null_base,
                    null_residual,
                    null_hidden,
                    feature_mask=None,
                )
                null_token_support = null_support[:, :-1, :]
                null_token_losses = _token_losses(null_logits, holdout_targets).reshape(-1)
                null_supports[null_name] = null_token_support
                if (
                    null_spec.get("null_control")
                    == "token_position_frequency_matched_teacher"
                ):
                    token_position_null_support = null_support
                null_eval_rows.append(
                    _variant_fold_row(
                        fold_index=fold_index,
                        spec=null_spec,
                        support=null_token_support,
                        token_losses=null_token_losses,
                        oracle_losses=oracle_losses,
                        num_columns=num_columns,
                    )
                )
                null_control_rows.append(
                    _null_control_row(
                        fold_index=fold_index,
                        null_name=null_name,
                        null_spec=null_spec,
                        student_support=student_token_support,
                        null_support=null_token_support,
                        teacher_support=teacher_token_support,
                        student_losses=student_token_losses,
                        null_losses=null_token_losses,
                        student_oracle_losses=oracle_losses,
                        num_columns=num_columns,
                    )
                )
            if token_position_null_support is None:
                raise RuntimeError("token/position null support was not evaluated")
            token_position_null_token_support = token_position_null_support[:, :-1, :]
            forced_losses["token_position_null_support_forced_into_student"] = (
                _forced_token_losses(
                    student_base,
                    student_residual,
                    student_hidden,
                    holdout_targets,
                    support=token_position_null_support,
                )
            )
            if capture_hidden_future and capture_train_hidden_future:
                (
                    train_hidden_future_rows,
                    train_exact_intervention_rows,
                ) = _capture_split_hidden_future_and_interventions(
                    split="train",
                    fold_index=fold_index,
                    all_pairs=all_pairs,
                    student_base=student_base,
                    student_residual=student_residual,
                    student_inputs=train_inputs,
                    student_targets=train_targets,
                    teacher_base=teacher_base,
                    teacher_residual=teacher_residual,
                    token_position_null_support=token_position_frequency_matched_teacher_train_support[
                        :, :-1, :
                    ],
                    vocab_size=vocab_size,
                )
                hidden_future_rows.extend(train_hidden_future_rows)
                exact_intervention_rows.extend(train_exact_intervention_rows)

        fold_rows.extend(
            [
                _variant_fold_row(
                    fold_index=fold_index,
                    spec=student_spec,
                    support=student_token_support,
                    token_losses=student_token_losses,
                    oracle_losses=oracle_losses,
                    num_columns=num_columns,
                ),
                _variant_fold_row(
                    fold_index=fold_index,
                    spec=causal_spec,
                    support=causal_token_support,
                    token_losses=causal_token_losses,
                    oracle_losses=oracle_losses,
                    num_columns=num_columns,
                ),
                {
                    **_variant_fold_row(
                        fold_index=fold_index,
                        spec={
                            "name": f"oracle_target_teacher_{teacher_oracle_weight:g}",
                            "support_router": "contextual_mlp_causal",
                            "top_k": top_k,
                            "causal_feature_safe": True,
                        },
                        support=teacher_token_support,
                        token_losses=teacher_token_losses,
                        oracle_losses=oracle_losses,
                        num_columns=num_columns,
                    ),
                    "teacher_oracle_weight": teacher_oracle_weight,
                },
                *null_eval_rows,
            ]
        )
        agreement_rows.extend(
            _agreement_rows(
                fold_index=fold_index,
                teacher_support=teacher_token_support,
                student_support=student_token_support,
                oracle_support=oracle_token_support,
                linear_support=linear_token_support,
                random_seed=random_seed + fold_index,
                num_columns=num_columns,
            )
        )
        disagreement_mask = _support_neq(teacher_token_support, student_token_support).reshape(-1)
        high_regret_mask = (student_token_losses - oracle_losses) > 0.0
        intervention_rows.extend(
            _intervention_rows(
                fold_index=fold_index,
                losses_by_intervention=forced_losses,
                disagreement_mask=disagreement_mask,
                high_regret_mask=high_regret_mask,
            )
        )
        per_token_rows.extend(
            _per_token_rows(
                fold_index=fold_index,
                targets=holdout_targets[:, :-1],
                teacher_support=teacher_token_support,
                student_support=student_token_support,
                oracle_support=oracle_token_support,
                linear_support=linear_token_support,
                token_position_null_support=token_position_null_token_support,
                losses_by_intervention=forced_losses,
            )
        )
        if capture_hidden_future:
            hidden_future_rows.extend(
                _hidden_future_capture_rows(
                    split="heldout",
                    fold_index=fold_index,
                    current_hidden=student_hidden[:, :-1, :],
                    future_hidden=teacher_hidden[:, 1:, :],
                    targets=holdout_targets[:, :-1],
                    teacher_scores=teacher_scores[:, :-1, :],
                    teacher_support=teacher_token_support,
                    student_support=student_token_support,
                    oracle_support=oracle_token_support,
                    token_position_null_support=token_position_null_token_support,
                )
            )
            exact_intervention_rows.extend(
                _exact_same_student_intervention_rows(
                    split="heldout",
                    fold_index=fold_index,
                    all_pairs=all_pairs,
                    pair_rows=pair_rows,
                    student_losses=student_token_losses,
                    teacher_losses=forced_losses["teacher_support_forced_into_student"],
                    oracle_losses=oracle_losses,
                    teacher_support=teacher_token_support,
                    student_support=student_token_support,
                    oracle_support=oracle_token_support,
                )
            )
        support_count_rows.extend(
            _support_count_rows(fold_index, "teacher_support", teacher_token_support)
        )
        support_count_rows.extend(
            _support_count_rows(fold_index, "student_support", student_token_support)
        )
        support_count_rows.extend(
            _support_count_rows(fold_index, "oracle_support", oracle_token_support)
        )
        support_count_rows.extend(
            _support_count_rows(fold_index, "linear_support", linear_token_support)
        )
        support_count_rows.extend(
            _support_count_rows(
                fold_index,
                "token_position_null_support",
                token_position_null_token_support,
            )
        )
        for null_name, null_support in null_supports.items():
            support_count_rows.extend(
                _support_count_rows(fold_index, f"{null_name}_support", null_support)
            )

    aggregate_rows = _aggregate_rows(fold_rows)
    agreement_aggregates = _aggregate_agreement_rows(agreement_rows)
    intervention_aggregates = _aggregate_intervention_rows(intervention_rows)
    null_control_aggregates = _aggregate_null_control_rows(null_control_rows)
    decision = _decision(
        agreement_aggregates=agreement_aggregates,
        intervention_aggregates=intervention_aggregates,
        null_control_aggregates=null_control_aggregates,
        ce_guardrail=ce_guardrail,
    )
    summary = {
        "status": "pass",
        "decision": decision["decision"],
        "claim_status": decision["claim_status"],
        "selected_next_step": decision["next_step"],
        "experiment_id": f"{experiment_id}_distillation_agreement_audit",
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
            "student_distill_weight": student_distill_weight,
            "source_artifact_assessment": source_assessment,
            "fold_metrics": fold_rows,
            "aggregate_metrics": {row["control"]: row for row in aggregate_rows},
            "agreement_rows": agreement_rows,
            "agreement_aggregates": agreement_aggregates,
            "intervention_rows": intervention_rows,
            "intervention_aggregates": intervention_aggregates,
            "null_control_rows": null_control_rows,
            "null_control_aggregates": null_control_aggregates,
            "null_sampling_rows": null_sampling_rows,
            "null_sampling_aggregates": _aggregate_null_sampling_rows(null_sampling_rows),
            "hidden_future_capture": _hidden_future_capture_summary(
                capture_hidden_future=capture_hidden_future,
                capture_train_hidden_future=capture_train_hidden_future,
                hidden_future_rows=hidden_future_rows,
                exact_intervention_rows=exact_intervention_rows,
            ),
            "gate_criteria": decision["criteria"],
            "failures": decision["failures"],
            "rationale": decision["rationale"],
        },
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "fold_metrics_csv": str(out_dir / "fold_metrics.csv"),
            "aggregate_metrics_csv": str(out_dir / "aggregate_metrics.csv"),
            "agreement_metrics_csv": str(out_dir / "agreement_metrics.csv"),
            "intervention_metrics_csv": str(out_dir / "intervention_metrics.csv"),
            "null_control_metrics_csv": str(out_dir / "null_control_metrics.csv"),
            "null_sampling_diagnostics_csv": str(out_dir / "null_sampling_diagnostics.csv"),
            "per_token_supports_csv": str(out_dir / "per_token_supports.csv"),
            "hidden_future_rows_csv": str(out_dir / "hidden_future_rows.csv"),
            "intervention_rows_exact_csv": str(out_dir / "intervention_rows_exact.csv"),
            "support_counts_csv": str(out_dir / "support_counts.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
        "git_commit": _git_commit(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "fold_metrics.csv", fold_rows)
    _write_csv(out_dir / "aggregate_metrics.csv", aggregate_rows)
    _write_csv(out_dir / "agreement_metrics.csv", agreement_rows)
    _write_csv(out_dir / "intervention_metrics.csv", intervention_rows)
    _write_csv(out_dir / "null_control_metrics.csv", null_control_rows)
    _write_csv(out_dir / "null_sampling_diagnostics.csv", null_sampling_rows)
    _write_csv(out_dir / "per_token_supports.csv", per_token_rows)
    _write_csv(out_dir / "hidden_future_rows.csv", hidden_future_rows)
    _write_csv(out_dir / "intervention_rows_exact.csv", exact_intervention_rows)
    _write_csv(out_dir / "support_counts.csv", support_count_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _source_artifact_assessment(
    audit_dir: Path,
    runpod_audit_dir: Path | None,
) -> list[dict[str, Any]]:
    rows = []
    for label, path in (
        ("local_distillation_audit", audit_dir),
        ("runpod_distillation_audit", runpod_audit_dir),
    ):
        if path is None:
            continue
        per_token = path / "per_token_supports.csv"
        checkpoints = list(path.glob("*.pt")) + list(path.glob("*.pth"))
        rows.append(
            {
                "source": label,
                "path": str(path),
                "exists": path.exists(),
                "has_summary": (path / "summary.json").is_file(),
                "has_per_token_supports": per_token.is_file(),
                "checkpoint_count": len(checkpoints),
                "usable_without_rerun": per_token.is_file() or bool(checkpoints),
                "action": "reuse" if per_token.is_file() else "fail_closed_bounded_rerun",
            }
        )
    return rows


def _capture_split_hidden_future_and_interventions(
    *,
    split: str,
    fold_index: int,
    all_pairs: list[tuple[int, ...]],
    student_base: Any,
    student_residual: Any,
    student_inputs: Any,
    student_targets: Any,
    teacher_base: Any,
    teacher_residual: Any,
    token_position_null_support: Any,
    vocab_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    with torch.no_grad():
        student_hidden = student_base.encode(student_inputs)
        teacher_hidden = teacher_base.encode(student_inputs)
        student_logits, student_support = _forward_with_feature_ablation(
            student_base,
            student_residual,
            student_hidden,
            feature_mask=None,
        )
        _, teacher_support = _forward_with_feature_ablation(
            teacher_base,
            teacher_residual,
            teacher_hidden,
            feature_mask=None,
        )
        teacher_scores = (
            teacher_residual._score_columns(teacher_hidden)
            + teacher_residual.score_tie_breaker.to(
                device=teacher_hidden.device,
                dtype=teacher_hidden.dtype,
            )
        )
        student_token_losses = _token_losses(student_logits, student_targets).reshape(-1)
        empty_logits = student_base.decode(student_hidden)
        empty_loss = _ce_loss(empty_logits, student_targets, vocab_size)
        student_router_loss = float(student_token_losses.mean().item())
        pair_rows = [
            _score_for_support(
                student_base,
                student_residual,
                student_hidden,
                student_targets,
                vocab_size,
                support=pair,
                empty_loss=empty_loss,
                router_loss=student_router_loss,
            )
            for pair in all_pairs
        ]
        pair_loss_matrix = torch.stack(
            [row["_token_losses"].reshape(-1) for row in pair_rows],
            dim=1,
        )
        oracle_losses, oracle_indices = pair_loss_matrix.min(dim=1)
        student_token_support = student_support[:, :-1, :]
        teacher_token_support = teacher_support[:, :-1, :]
        oracle_token_support = _support_from_pair_indices(
            oracle_indices,
            all_pairs,
            like=student_token_support,
        )
        teacher_losses = _forced_token_losses(
            student_base,
            student_residual,
            student_hidden,
            student_targets,
            support=teacher_support,
        )
    hidden_rows = _hidden_future_capture_rows(
        split=split,
        fold_index=fold_index,
        current_hidden=student_hidden[:, :-1, :],
        future_hidden=teacher_hidden[:, 1:, :],
        targets=student_targets[:, :-1],
        teacher_scores=teacher_scores[:, :-1, :],
        teacher_support=teacher_token_support,
        student_support=student_token_support,
        oracle_support=oracle_token_support,
        token_position_null_support=token_position_null_support,
    )
    exact_rows = _exact_same_student_intervention_rows(
        split=split,
        fold_index=fold_index,
        all_pairs=all_pairs,
        pair_rows=pair_rows,
        student_losses=student_token_losses,
        teacher_losses=teacher_losses,
        oracle_losses=oracle_losses,
        teacher_support=teacher_token_support,
        student_support=student_token_support,
        oracle_support=oracle_token_support,
    )
    return hidden_rows, exact_rows


def _hidden_future_capture_rows(
    *,
    split: str,
    fold_index: int,
    current_hidden: Any,
    future_hidden: Any,
    targets: Any,
    teacher_scores: Any,
    teacher_support: Any,
    student_support: Any,
    oracle_support: Any,
    token_position_null_support: Any,
) -> list[dict[str, Any]]:
    import torch

    previous_hidden = torch.cat(
        [current_hidden[:, :1, :], current_hidden[:, :-1, :]],
        dim=1,
    )
    future_delta = future_hidden - current_hidden
    rows: list[dict[str, Any]] = []
    flat_position = 0
    for batch_index in range(int(current_hidden.shape[0])):
        for position_index in range(int(current_hidden.shape[1])):
            teacher_pair = _tensor_support_key(teacher_support[batch_index, position_index])
            student_pair = _tensor_support_key(student_support[batch_index, position_index])
            oracle_pair = _tensor_support_key(oracle_support[batch_index, position_index])
            null_pair = _tensor_support_key(
                token_position_null_support[batch_index, position_index]
            )
            rows.append(
                {
                    "fold": fold_index,
                    "split": split,
                    "sequence_id": f"fold{fold_index}_{split}_sequence{batch_index}",
                    "batch_index": batch_index,
                    "position_index": position_index,
                    "flat_position": flat_position,
                    "target_token_eval_only": int(targets[batch_index, position_index].item()),
                    "current_hidden_json": _tensor_json(
                        current_hidden[batch_index, position_index]
                    ),
                    "previous_hidden_json": _tensor_json(
                        previous_hidden[batch_index, position_index]
                    ),
                    "future_hidden_json": _tensor_json(
                        future_hidden[batch_index, position_index]
                    ),
                    "future_delta_json": _tensor_json(
                        future_delta[batch_index, position_index]
                    ),
                    "teacher_support_logits_json": _tensor_json(
                        teacher_scores[batch_index, position_index]
                    ),
                    "teacher_topk_support": teacher_pair,
                    "student_router_support": student_pair,
                    "oracle_support_eval_only": oracle_pair,
                    "token_position_null_support": null_pair,
                    "prefix_safe_fields": "current_hidden_json;previous_hidden_json;position_index",
                    "teacher_target_fields": (
                        "future_hidden_json;future_delta_json;"
                        "teacher_support_logits_json;teacher_topk_support"
                    ),
                    "forbidden_predictor_fields": (
                        "future_hidden_json;future_delta_json;teacher_support_logits_json;"
                        "teacher_topk_support;target_token_eval_only;oracle_support_eval_only"
                    ),
                    "router_teacher_provenance": "contextual_router_distillation_agreement_teacher",
                    "future_targets_nondeployable": True,
                }
            )
            flat_position += 1
    return rows


def _exact_same_student_intervention_rows(
    *,
    split: str,
    fold_index: int,
    all_pairs: list[tuple[int, ...]],
    pair_rows: list[dict[str, Any]],
    student_losses: Any,
    teacher_losses: Any,
    oracle_losses: Any,
    teacher_support: Any,
    student_support: Any,
    oracle_support: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    student_flat = student_losses.reshape(-1)
    teacher_flat = teacher_losses.reshape(-1)
    oracle_loss_flat = oracle_losses.reshape(-1)
    teacher_support_flat = teacher_support.reshape(-1, teacher_support.shape[-1])
    student_support_flat = student_support.reshape(-1, student_support.shape[-1])
    oracle_support_flat = oracle_support.reshape(-1, oracle_support.shape[-1])
    pair_loss_columns = [
        pair_row["_token_losses"].reshape(-1).detach().cpu().tolist()
        for pair_row in pair_rows
    ]
    positions_per_sequence = int(teacher_support.shape[1])
    for flat_position in range(int(student_flat.shape[0])):
        sequence_index = flat_position // positions_per_sequence
        position_index = flat_position % positions_per_sequence
        teacher_pair = _tensor_support_key(teacher_support_flat[flat_position])
        student_pair = _tensor_support_key(student_support_flat[flat_position])
        oracle_pair = _tensor_support_key(oracle_support_flat[flat_position])
        student_loss = float(student_flat[flat_position].item())
        oracle_loss = float(oracle_loss_flat[flat_position].item())
        for pair_index, pair in enumerate(all_pairs):
            pair_key = _support_key(pair)
            forced_loss = float(pair_loss_columns[pair_index][flat_position])
            rows.append(
                {
                    "fold": fold_index,
                    "split": split,
                    "sequence_id": f"fold{fold_index}_{split}_sequence{sequence_index}",
                    "position_index": position_index,
                    "flat_position": flat_position,
                    "forced_support_pair": pair_key,
                    "forced_support_pair_index": pair_index,
                    "forced_support_loss": forced_loss,
                    "student_router_support": student_pair,
                    "student_router_support_loss": student_loss,
                    "teacher_support": teacher_pair,
                    "teacher_support_forced_loss": float(
                        teacher_flat[flat_position].item()
                    ),
                    "oracle_support": oracle_pair,
                    "oracle_support_loss": oracle_loss,
                    "forced_minus_student_router_loss": forced_loss - student_loss,
                    "forced_minus_oracle_loss": forced_loss - oracle_loss,
                    "is_teacher_support_pair": pair_key == teacher_pair,
                    "is_student_router_support_pair": pair_key == student_pair,
                    "is_oracle_support_pair": pair_key == oracle_pair,
                    "row_family": "same_student_forced_support_exact_pair",
                }
            )
    return rows


def _hidden_future_capture_summary(
    *,
    capture_hidden_future: bool,
    capture_train_hidden_future: bool,
    hidden_future_rows: list[dict[str, Any]],
    exact_intervention_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    hidden_fields = {
        "current_hidden_json",
        "previous_hidden_json",
        "future_hidden_json",
        "future_delta_json",
        "teacher_support_logits_json",
    }
    intervention_fields = {
        "forced_support_pair",
        "forced_support_loss",
        "student_router_support_loss",
        "teacher_support_forced_loss",
        "oracle_support_loss",
    }
    hidden_schema_ok = bool(hidden_future_rows) and hidden_fields.issubset(
        hidden_future_rows[0]
    )
    intervention_schema_ok = bool(
        exact_intervention_rows
    ) and intervention_fields.issubset(exact_intervention_rows[0])
    hidden_split_counts = _split_counts(hidden_future_rows)
    intervention_split_counts = _split_counts(exact_intervention_rows)
    return {
        "enabled": capture_hidden_future,
        "train_capture_enabled": capture_train_hidden_future,
        "status": (
            "captured"
            if capture_hidden_future and hidden_schema_ok and intervention_schema_ok
            else "not_requested"
            if not capture_hidden_future
            else "failed_closed"
        ),
        "hidden_future_row_count": len(hidden_future_rows),
        "exact_intervention_row_count": len(exact_intervention_rows),
        "hidden_future_split_counts": hidden_split_counts,
        "exact_intervention_split_counts": intervention_split_counts,
        "train_sequence_count": _sequence_count(hidden_future_rows, "train"),
        "heldout_sequence_count": _sequence_count(hidden_future_rows, "heldout"),
        "split_coverage_available": (
            _sequence_count(hidden_future_rows, "train") > 0
            and _sequence_count(hidden_future_rows, "heldout") > 0
        ),
        "hidden_future_schema_ok": hidden_schema_ok,
        "exact_intervention_schema_ok": intervention_schema_ok,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "leakage_contract": (
            "current/previous hidden are prefix-safe inputs; future hidden, future delta, "
            "teacher logits/support, targets, and oracle labels are forbidden predictor inputs"
        ),
    }


def _split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        split = str(row.get("split", ""))
        counts[split] = counts.get(split, 0) + 1
    return counts


def _sequence_count(rows: list[dict[str, Any]], split: str) -> int:
    return len({row["sequence_id"] for row in rows if row.get("split") == split})


def _tensor_json(value: Any) -> str:
    return json.dumps(
        [round(float(item), 8) for item in value.detach().cpu().reshape(-1).tolist()],
        separators=(",", ":"),
    )


def _tensor_support_key(value: Any) -> str:
    return ",".join(str(int(item)) for item in value.detach().cpu().reshape(-1).tolist())


def _support_from_pair_indices(indices: Any, all_pairs: list[tuple[int, ...]], *, like: Any) -> Any:
    import torch

    flat = [all_pairs[int(index)] for index in indices.detach().cpu().tolist()]
    return torch.tensor(flat, dtype=like.dtype, device=like.device).reshape_as(like)


def _with_last_support(token_support: Any) -> Any:
    import torch

    return torch.cat([token_support, token_support[:, -1:, :]], dim=1)


def _shuffle_support(support: Any, *, seed: int) -> Any:
    import torch

    flat = support.reshape(-1, support.shape[-1])
    generator = torch.Generator(device=flat.device)
    generator.manual_seed(seed)
    permutation = torch.randperm(flat.shape[0], generator=generator, device=flat.device)
    return flat[permutation].reshape_as(support)


def _frequency_matched_support(support: Any, *, seed: int) -> Any:
    import torch

    flat = support.reshape(-1, support.shape[-1])
    generator = torch.Generator(device=flat.device)
    generator.manual_seed(seed)
    sample_indices = torch.randint(
        low=0,
        high=flat.shape[0],
        size=(flat.shape[0],),
        generator=generator,
        device=flat.device,
    )
    return flat[sample_indices].reshape_as(support)


def _token_position_frequency_matched_support(
    support: Any,
    targets: Any,
    *,
    seed: int,
) -> tuple[Any, list[dict[str, Any]]]:
    """Resample teacher support within target-token/position strata when possible."""

    import torch

    flat_support = support.reshape(-1, support.shape[-1])
    target_rows = targets.reshape(targets.shape[0], targets.shape[1])
    if target_rows.shape[:2] != support.shape[:2]:
        raise ValueError("targets must align with support batch/sequence dimensions")
    flat_targets = target_rows.reshape(-1).detach().cpu().tolist()
    positions = (
        torch.arange(support.shape[1], device=support.device)
        .repeat(support.shape[0])
        .detach()
        .cpu()
        .tolist()
    )
    token_position_groups: dict[tuple[int, int], list[int]] = {}
    token_groups: dict[int, list[int]] = {}
    for index, (target, position) in enumerate(zip(flat_targets, positions, strict=True)):
        token = int(target)
        pos = int(position)
        token_position_groups.setdefault((token, pos), []).append(index)
        token_groups.setdefault(token, []).append(index)

    generator = torch.Generator(device=flat_support.device)
    generator.manual_seed(seed)
    global_indices = list(range(flat_support.shape[0]))
    sampled_indices = []
    diagnostic_rows = []
    for index, (target, position) in enumerate(zip(flat_targets, positions, strict=True)):
        token = int(target)
        pos = int(position)
        candidates = token_position_groups.get((token, pos), [])
        sampling_mode = "target_position"
        if len(candidates) <= 1:
            candidates = token_groups.get(token, [])
            sampling_mode = "target_only"
        if len(candidates) <= 1:
            candidates = global_indices
            sampling_mode = "global"
        original_stratum_size = len(candidates)
        if len(candidates) > 1 and index in candidates:
            candidates = [candidate for candidate in candidates if candidate != index]
        choice = torch.randint(
            low=0,
            high=len(candidates),
            size=(1,),
            generator=generator,
            device=flat_support.device,
        )
        sampled_indices.append(candidates[int(choice.item())])
        diagnostic_rows.append(
            {
                "flat_position": index,
                "target_token": token,
                "position_index": pos,
                "sampling_mode": sampling_mode,
                "candidate_count": len(candidates),
                "original_stratum_size": original_stratum_size,
                "candidate_support_entropy": _candidate_support_entropy(
                    flat_support,
                    candidates,
                ),
            }
        )
    sample_tensor = torch.tensor(sampled_indices, dtype=torch.long, device=flat_support.device)
    return flat_support[sample_tensor].reshape_as(support), diagnostic_rows


def _candidate_support_entropy(flat_support: Any, candidates: list[int]) -> float:
    counts: dict[str, int] = {}
    for candidate in candidates:
        row = flat_support[candidate].detach().cpu().tolist()
        key = _support_key(tuple(sorted(int(value) for value in row)))
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(total)


def _random_support(all_pairs: list[tuple[int, ...]], *, like: Any, seed: int) -> Any:
    import torch

    generator = torch.Generator(device=like.device)
    generator.manual_seed(seed)
    indices = torch.randint(
        low=0,
        high=len(all_pairs),
        size=(like.reshape(-1, like.shape[-1]).shape[0],),
        generator=generator,
        device=like.device,
    )
    return _support_from_pair_indices(indices, all_pairs, like=like)


def _forced_token_losses(
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    *,
    support: Any,
) -> Any:
    logits = base.decode(residual(hidden, support_indices=support))
    return _token_losses(logits, targets).reshape(-1)


def _support_neq(left: Any, right: Any) -> Any:
    return (left.sort(dim=-1).values != right.sort(dim=-1).values).any(dim=-1)


def _variant_fold_row(
    *,
    fold_index: int,
    spec: dict[str, Any],
    support: Any,
    token_losses: Any,
    oracle_losses: Any,
    num_columns: int,
) -> dict[str, Any]:
    return {
        "fold": fold_index,
        "heldout_sequence_index": fold_index,
        "control": spec["name"],
        "support_router": spec["support_router"],
        "top_k": spec["top_k"],
        "causal_feature_safe": spec["causal_feature_safe"],
        "positions": int(token_losses.numel()),
        "empty_loss": None,
        "router_loss": float(token_losses.mean().item()),
        "oracle_loss": float(oracle_losses.mean().item()),
        "oracle_support_regret": float((token_losses - oracle_losses).mean().item()),
        "oracle_support_regret_positive_fraction": float(
            ((token_losses - oracle_losses) > 0.0).to(dtype=token_losses.dtype).mean().item()
        ),
        "router_oracle_gap": float((token_losses - oracle_losses).mean().item()),
        "recovery_fraction_vs_empty": None,
        "best_global_fixed_support_loss": None,
        "dominant_fixed_support_loss": None,
        "random_support_loss": None,
        "shuffled_support_loss": None,
        "used_columns": _used_columns(support),
        "dead_columns": num_columns - _used_columns(support),
        "unique_support_sets": len(_support_counts(support)),
        "support_change_fraction": _support_change_fraction(support),
        "functional_churn_logit_l1": None,
        "support_load_entropy": _normalized_load_entropy(support, num_columns),
    }


def _agreement_rows(
    *,
    fold_index: int,
    teacher_support: Any,
    student_support: Any,
    oracle_support: Any,
    linear_support: Any,
    random_seed: int,
    num_columns: int,
) -> list[dict[str, Any]]:
    shuffled_teacher = _shuffle_support(teacher_support, seed=random_seed + 17)
    rows = [
        _pair_agreement_row(
            fold_index,
            "student_vs_teacher",
            student_support,
            teacher_support,
            num_columns=num_columns,
        ),
        _pair_agreement_row(
            fold_index,
            "student_vs_oracle",
            student_support,
            oracle_support,
            num_columns=num_columns,
        ),
        _pair_agreement_row(
            fold_index,
            "student_vs_linear",
            student_support,
            linear_support,
            num_columns=num_columns,
        ),
        _pair_agreement_row(
            fold_index,
            "student_vs_shuffled_teacher_chance",
            student_support,
            shuffled_teacher,
            num_columns=num_columns,
        ),
    ]
    exact = rows[0]["exact_pair_agreement"]
    chance = rows[-1]["exact_pair_agreement"]
    rows[0]["exact_pair_agreement_lift_vs_shuffled_teacher"] = exact - chance
    return rows


def _pair_agreement_row(
    fold_index: int,
    comparison: str,
    left: Any,
    right: Any,
    *,
    num_columns: int,
) -> dict[str, Any]:
    import torch

    left_flat = left.reshape(-1, left.shape[-1]).sort(dim=-1).values
    right_flat = right.reshape(-1, right.shape[-1]).sort(dim=-1).values
    exact = (left_flat == right_flat).all(dim=-1).to(dtype=torch.float32)
    intersections = []
    for left_row, right_row in zip(left_flat.cpu().tolist(), right_flat.cpu().tolist()):
        intersections.append(len(set(int(value) for value in left_row) & set(int(value) for value in right_row)))
    intersection_tensor = torch.tensor(intersections, dtype=torch.float32)
    k = float(left.shape[-1])
    left_load = _column_load(left, num_columns)
    right_load = _column_load(right, num_columns)
    return {
        "fold": fold_index,
        "comparison": comparison,
        "positions": int(exact.numel()),
        "exact_pair_agreement": float(exact.mean().item()),
        "mean_intersection_size": float(intersection_tensor.mean().item()),
        "mean_jaccard": float((intersection_tensor / (2.0 * k - intersection_tensor)).mean().item()),
        "column_precision": float((intersection_tensor / k).mean().item()),
        "column_recall": float((intersection_tensor / k).mean().item()),
        "column_f1": float((intersection_tensor / k).mean().item()),
        "left_support_entropy": _normalized_load_entropy(left, num_columns),
        "right_support_entropy": _normalized_load_entropy(right, num_columns),
        "load_correlation": _pearson(left_load, right_load),
    }


def _intervention_rows(
    *,
    fold_index: int,
    losses_by_intervention: dict[str, Any],
    disagreement_mask: Any,
    high_regret_mask: Any,
) -> list[dict[str, Any]]:
    rows = []
    for mask_name, mask in {
        "all_tokens": None,
        "teacher_student_disagreement_tokens": disagreement_mask,
        "student_positive_oracle_regret_tokens": high_regret_mask,
    }.items():
        baseline = _masked_mean(losses_by_intervention["student_router_support"], mask)
        for intervention, losses in losses_by_intervention.items():
            rows.append(
                {
                    "fold": fold_index,
                    "token_subset": mask_name,
                    "intervention": intervention,
                    "positions": _masked_count(losses, mask),
                    "loss": _masked_mean(losses, mask),
                    "delta_vs_student_router_support": _masked_mean(losses, mask) - baseline,
                }
            )
    return rows


def _per_token_rows(
    *,
    fold_index: int,
    targets: Any,
    teacher_support: Any,
    student_support: Any,
    oracle_support: Any,
    linear_support: Any,
    token_position_null_support: Any,
    losses_by_intervention: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    target_flat = targets.reshape(-1).detach().cpu().tolist()
    supports = {
        "teacher_support": teacher_support,
        "student_support": student_support,
        "oracle_support": oracle_support,
        "linear_support": linear_support,
        "token_position_null_support": token_position_null_support,
    }
    support_keys = {
        name: [
            _support_key(tuple(sorted(int(value) for value in row)))
            for row in tensor.reshape(-1, tensor.shape[-1]).detach().cpu().tolist()
        ]
        for name, tensor in supports.items()
    }
    loss_values = {
        name: tensor.detach().cpu().tolist()
        for name, tensor in losses_by_intervention.items()
    }
    for index, target in enumerate(target_flat):
        row = {
            "fold": fold_index,
            "flat_position": index,
            "target_token": int(target),
            **{name: values[index] for name, values in support_keys.items()},
            **{f"{name}_loss": float(values[index]) for name, values in loss_values.items()},
        }
        row["teacher_student_exact_pair_match"] = (
            row["teacher_support"] == row["student_support"]
        )
        rows.append(row)
    return rows


def _support_count_rows(fold_index: int, source: str, support: Any) -> list[dict[str, Any]]:
    return [
        {"fold": fold_index, "source": source, "support": key, "count": count}
        for key, count in _support_counts(support).items()
    ]


def _null_control_row(
    *,
    fold_index: int,
    null_name: str,
    null_spec: dict[str, Any],
    student_support: Any,
    null_support: Any,
    teacher_support: Any,
    student_losses: Any,
    null_losses: Any,
    student_oracle_losses: Any,
    num_columns: int,
) -> dict[str, Any]:
    student_teacher = _pair_agreement_row(
        fold_index,
        "student_vs_teacher",
        student_support,
        teacher_support,
        num_columns=num_columns,
    )
    null_teacher = _pair_agreement_row(
        fold_index,
        f"{null_name}_vs_teacher",
        null_support,
        teacher_support,
        num_columns=num_columns,
    )
    student_regret = student_losses - student_oracle_losses
    null_regret = null_losses - student_oracle_losses
    return {
        "fold": fold_index,
        "null_control": null_name,
        "null_control_kind": null_spec["null_control"],
        "positions": int(student_losses.numel()),
        "student_router_loss": float(student_losses.mean().item()),
        "null_router_loss": float(null_losses.mean().item()),
        "student_minus_null_router_loss": float(
            (student_losses.mean() - null_losses.mean()).item()
        ),
        "student_oracle_regret": float(student_regret.mean().item()),
        "null_oracle_regret": float(null_regret.mean().item()),
        "student_minus_null_oracle_regret": float(
            (student_regret.mean() - null_regret.mean()).item()
        ),
        "student_teacher_exact_pair_agreement": student_teacher["exact_pair_agreement"],
        "null_teacher_exact_pair_agreement": null_teacher["exact_pair_agreement"],
        "student_minus_null_teacher_exact_pair_agreement": (
            student_teacher["exact_pair_agreement"] - null_teacher["exact_pair_agreement"]
        ),
        "student_teacher_mean_jaccard": student_teacher["mean_jaccard"],
        "null_teacher_mean_jaccard": null_teacher["mean_jaccard"],
        "student_minus_null_teacher_mean_jaccard": (
            student_teacher["mean_jaccard"] - null_teacher["mean_jaccard"]
        ),
        "student_support_entropy": _normalized_load_entropy(student_support, num_columns),
        "null_support_entropy": _normalized_load_entropy(null_support, num_columns),
    }


def _support_counts(support: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in support.reshape(-1, support.shape[-1]).detach().cpu().tolist():
        key = _support_key(tuple(sorted(int(value) for value in row)))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _support_change_fraction(support: Any) -> float:
    rows = [
        tuple(sorted(int(value) for value in row))
        for row in support.reshape(-1, support.shape[-1]).detach().cpu().tolist()
    ]
    if len(rows) <= 1:
        return 0.0
    changes = sum(1 for left, right in zip(rows, rows[1:]) if left != right)
    return changes / (len(rows) - 1)


def _used_columns(support: Any) -> int:
    return len({int(value) for value in support.reshape(-1).detach().cpu().tolist()})


def _column_load(support: Any, num_columns: int) -> list[float]:
    counts = [0.0 for _ in range(num_columns)]
    for value in support.reshape(-1).detach().cpu().tolist():
        counts[int(value)] += 1.0
    total = sum(counts)
    if total <= 0.0:
        return counts
    return [count / total for count in counts]


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((l - left_mean) * (r - right_mean) for l, r in zip(left, right))
    left_var = sum((l - left_mean) ** 2 for l in left)
    right_var = sum((r - right_mean) ** 2 for r in right)
    denominator = math.sqrt(left_var * right_var)
    if denominator <= 1e-12:
        return None
    return numerator / denominator


def _masked_mean(values: Any, mask: Any | None) -> float:
    if mask is None:
        selected = values
    else:
        selected = values[mask]
    if int(selected.numel()) == 0:
        return float("nan")
    return float(selected.mean().item())


def _masked_count(values: Any, mask: Any | None) -> int:
    if mask is None:
        return int(values.numel())
    return int(mask.to(dtype=values.dtype).sum().item())


def _aggregate_agreement_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        comparison: _mean_row(group, key_name="comparison", key_value=comparison)
        for comparison, group in _group_by(rows, "comparison").items()
    }


def _aggregate_intervention_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped = {}
    for row in rows:
        key = f"{row['token_subset']}::{row['intervention']}"
        grouped.setdefault(key, []).append(row)
    return {key: _mean_row(group, key_name="intervention_key", key_value=key) for key, group in grouped.items()}


def _aggregate_null_control_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        null_control: _mean_row(group, key_name="null_control", key_value=null_control)
        for null_control, group in _group_by(rows, "null_control").items()
    }


def _aggregate_null_sampling_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregates = {}
    for null_control, group in _group_by(rows, "null_control").items():
        total = len(group)
        by_mode = _group_by(group, "sampling_mode")
        aggregates[null_control] = {
            "null_control": null_control,
            "positions": total,
            "target_position_fraction": len(by_mode.get("target_position", [])) / total
            if total
            else 0.0,
            "target_only_fraction": len(by_mode.get("target_only", [])) / total
            if total
            else 0.0,
            "global_fraction": len(by_mode.get("global", [])) / total if total else 0.0,
            "mean_candidate_count": _mean_numeric(group, "candidate_count"),
            "mean_original_stratum_size": _mean_numeric(group, "original_stratum_size"),
            "mean_candidate_support_entropy": _mean_numeric(
                group,
                "candidate_support_entropy",
            ),
        }
    return aggregates


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


def _mean_row(rows: list[dict[str, Any]], *, key_name: str, key_value: str) -> dict[str, Any]:
    result: dict[str, Any] = {key_name: key_value, "folds": len(rows)}
    fields = sorted({field for row in rows for field in row})
    for field in fields:
        values = [row[field] for row in rows if isinstance(row.get(field), (int, float))]
        if values:
            result[f"mean_{field}"] = sum(float(value) for value in values) / len(values)
    return result


def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    return sum(values) / len(values)


def _decision(
    *,
    agreement_aggregates: dict[str, dict[str, Any]],
    intervention_aggregates: dict[str, dict[str, Any]],
    null_control_aggregates: dict[str, dict[str, Any]],
    ce_guardrail: float,
) -> dict[str, Any]:
    student_teacher = agreement_aggregates.get("student_vs_teacher", {})
    chance = agreement_aggregates.get("student_vs_shuffled_teacher_chance", {})
    teacher_forced = intervention_aggregates.get(
        "teacher_student_disagreement_tokens::teacher_support_forced_into_student",
        {},
    )
    student_router = intervention_aggregates.get(
        "teacher_student_disagreement_tokens::student_router_support",
        {},
    )
    oracle_forced = intervention_aggregates.get(
        "teacher_student_disagreement_tokens::oracle_best_support_for_student",
        {},
    )
    teacher_delta = teacher_forced.get("mean_delta_vs_student_router_support")
    oracle_delta = oracle_forced.get("mean_delta_vs_student_router_support")
    criteria = [
        _criterion(
            "student_teacher_pair_agreement_above_shuffled",
            _value(student_teacher, "mean_exact_pair_agreement")
            > _value(chance, "mean_exact_pair_agreement"),
            "student-vs-teacher exact agreement > shuffled-teacher chance",
            {
                "student_teacher": student_teacher.get("mean_exact_pair_agreement"),
                "shuffled_teacher": chance.get("mean_exact_pair_agreement"),
            },
        ),
        _criterion(
            "teacher_forced_not_better_on_disagreements",
            teacher_delta is not None and teacher_delta >= -ce_guardrail,
            f"teacher-forced disagreement CE delta >= -{ce_guardrail}",
            teacher_delta,
        ),
        _criterion(
            "oracle_intervention_still_finds_headroom",
            oracle_delta is not None and oracle_delta < 0.0,
            "oracle forced support beats student support on disagreement tokens",
            oracle_delta,
        ),
        _criterion(
            "disagreement_subset_nonempty",
            _value(student_router, "mean_positions") > 0.0,
            "teacher/student disagreement subset has positions",
            student_router.get("mean_positions"),
        ),
    ]
    for null_name, null_row in sorted(null_control_aggregates.items()):
        criteria.extend(
            [
                _criterion(
                    f"real_student_ce_beats_{null_name}",
                    _has_lt(null_row, "mean_student_minus_null_router_loss", 0.0),
                    "real teacher-distilled student CE < null-distilled student CE",
                    null_row.get("mean_student_minus_null_router_loss"),
                ),
                _criterion(
                    f"real_student_regret_beats_{null_name}",
                    _has_lt(null_row, "mean_student_minus_null_oracle_regret", 0.0),
                    "real teacher-distilled student oracle regret < null-distilled student oracle regret",
                    null_row.get("mean_student_minus_null_oracle_regret"),
                ),
                _criterion(
                    f"real_student_teacher_agreement_beats_{null_name}",
                    _has_gt(
                        null_row,
                        "mean_student_minus_null_teacher_exact_pair_agreement",
                        0.0,
                    ),
                    "real teacher-distilled student agreement > null-distilled student agreement",
                    null_row.get("mean_student_minus_null_teacher_exact_pair_agreement"),
                ),
            ]
        )
    failures = [row for row in criteria if not row["passed"]]
    if failures:
        return {
            "decision": AGREEMENT_BLOCKED,
            "claim_status": "distilled_causal_router_mechanism_not_established",
            "next_step": (
                "keep defaults blocked; inspect failed agreement or null-control criteria "
                "before broader promotion repeats"
            ),
            "criteria": criteria,
            "failures": failures,
            "rationale": (
                "The richer audit does not yet establish that the deployable student "
                "learned the teacher's useful support policy rather than an auxiliary "
                "regularization effect."
            ),
        }
    return {
        "decision": AGREEMENT_SUPPORTED,
        "claim_status": (
            "distilled_causal_router_support_mechanism_and_null_controls_supported_not_promoted"
        ),
        "next_step": (
            "repeat the real-vs-null causal-router distillation mechanism audit "
            "across broader folds or a second dataset before any promotion repeat"
        ),
        "criteria": criteria,
        "failures": [],
        "rationale": (
            "The student agrees with the teacher above chance, teacher-forced supports "
            "do not improve disagreement-token CE beyond the guardrail, and oracle "
            "forcing still exposes bounded support headroom. The real teacher-distilled "
            "student also beats shuffled, frequency-matched, and token/position-stratified "
            "frequency-matched null-distilled students on the bounded CE, regret, and "
            "teacher-agreement checks."
        ),
    }


def _criterion(name: str, passed: bool, threshold: Any, actual: Any) -> dict[str, Any]:
    return {"criterion": name, "passed": bool(passed), "threshold": threshold, "actual": actual}


def _value(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        return float("-inf")
    return float(value)


def _has_lt(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = row.get(key)
    return value is not None and float(value) < threshold


def _has_gt(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = row.get(key)
    return value is not None and float(value) > threshold


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    audit = summary["audit"]
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "Teacher-student support agreement and disagreement-intervention audit.",
        "",
        f"- Config: `{summary['config_path']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Folds: `{audit['fold_count']}`",
        f"- Teacher oracle-target weight: `{audit['teacher_oracle_weight']}`",
        f"- Student distill weight: `{audit['student_distill_weight']}`",
        f"- Rationale: {audit['rationale']}",
        "",
        "## Source Artifact Assessment",
    ]
    for row in audit["source_artifact_assessment"]:
        lines.append(
            "- "
            f"{row['source']}: per-token supports `{row['has_per_token_supports']}`, "
            f"checkpoints `{row['checkpoint_count']}`, action `{row['action']}`"
        )
    lines.extend(["", "## Gate Criteria"])
    for row in audit["gate_criteria"]:
        lines.append(
            f"- {row['criterion']}: `{row['passed']}` "
            f"(threshold `{row['threshold']}`, actual `{row['actual']}`)"
        )
    lines.extend(["", "## Agreement Aggregates"])
    for key, row in sorted(audit["agreement_aggregates"].items()):
        lines.append(
            "- "
            f"{key}: exact `{row.get('mean_exact_pair_agreement')}`, "
            f"Jaccard `{row.get('mean_mean_jaccard')}`, "
            f"load corr `{row.get('mean_load_correlation')}`"
        )
    lines.extend(["", "## Disagreement Interventions"])
    for key, row in sorted(audit["intervention_aggregates"].items()):
        if not key.startswith("teacher_student_disagreement_tokens::"):
            continue
        lines.append(
            "- "
            f"{key}: loss `{row.get('mean_loss')}`, "
            f"delta `{row.get('mean_delta_vs_student_router_support')}`"
        )
    lines.extend(["", "## Null Distillation Controls"])
    for key, row in sorted(audit["null_control_aggregates"].items()):
        lines.append(
            "- "
            f"{key}: student-null CE delta `{row.get('mean_student_minus_null_router_loss')}`, "
            f"student-null regret delta `{row.get('mean_student_minus_null_oracle_regret')}`, "
            f"student-null agreement delta "
            f"`{row.get('mean_student_minus_null_teacher_exact_pair_agreement')}`"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_DISTILLATION_AUDIT_DIR)
    parser.add_argument("--runpod-audit-dir", type=Path, default=DEFAULT_RUNPOD_DISTILLATION_AUDIT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--teacher-oracle-weight", type=float, default=0.05)
    parser.add_argument("--student-distill-weight", type=float, default=0.05)
    parser.add_argument("--ce-guardrail", type=float, default=0.05)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=(
            "batch sequences to build before fold splitting; with "
            "--capture-train-hidden-future and --max-folds 1 this scales train "
            "coverage without cross-fold split leakage"
        ),
    )
    parser.add_argument(
        "--capture-hidden-future",
        action="store_true",
        help="emit prefix hidden, future target, teacher-logit, and exact intervention rows",
    )
    parser.add_argument(
        "--capture-train-hidden-future",
        action="store_true",
        help=(
            "also emit train split hidden/future rows; requires --capture-hidden-future "
            "and --max-folds 1"
        ),
    )
    args = parser.parse_args(argv)
    summary = run_causal_contextual_router_distillation_agreement_audit(
        args.config,
        args.out,
        audit_dir=args.audit_dir,
        runpod_audit_dir=args.runpod_audit_dir,
        max_folds=args.max_folds,
        teacher_oracle_weight=args.teacher_oracle_weight,
        student_distill_weight=args.student_distill_weight,
        ce_guardrail=args.ce_guardrail,
        batch_size=args.batch_size,
        capture_hidden_future=args.capture_hidden_future,
        capture_train_hidden_future=args.capture_train_hidden_future,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
