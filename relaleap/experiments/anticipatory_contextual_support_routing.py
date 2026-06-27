"""Local smoke pilot for anticipatory contextual support routing."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path(
    "configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml"
)
DEFAULT_OUT_DIR = Path("results/audits/token_larger_anticipatory_contextual_support_routing")
REQUIRED_ARTIFACTS = [
    "summary.json",
    "predictor_metrics.csv",
    "router_metrics.csv",
    "same_student_metrics.csv",
    "support_agreement.csv",
    "feature_perturbation.csv",
    "sequence_heldout_metrics.csv",
    "per_sequence_paired_deltas.csv",
    "margin_fragility.csv",
    "parameter_counts.csv",
    "retention_churn_metrics.csv",
    "notes.md",
]


def run_anticipatory_contextual_support_routing(
    *,
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path = DEFAULT_OUT_DIR,
    max_steps: int | None = None,
    predictor_steps: int = 80,
) -> dict[str, Any]:
    """Run a bounded CPU ACSR pilot and write fail-closed artifacts."""

    start = time.time()
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        from relaleap.smoke import (
            ResidualColumns,
            TinyCharTransformer,
            _build_batch,
            _residual_loss,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        summary = _failure_summary(
            out_dir=out_dir,
            start=start,
            config_path=config_path,
            reason=f"torch_or_smoke_import_failed: {exc}",
        )
        _write_artifacts(
            out_dir,
            summary,
            predictor_rows=[],
            router_rows=[],
            same_student_rows=[],
            support_agreement_rows=[],
            perturbation_rows=[],
            sequence_rows=[],
            margin_rows=[],
            parameter_rows=[],
            retention_rows=[],
            per_sequence_rows=[],
        )
        return summary

    config = _read_yaml(config_path)
    run_cfg = _dict(config.get("run"))
    data_cfg = _dict(config.get("data"))
    model_cfg = _dict(config.get("model"))
    base_cfg = _dict(model_cfg.get("base"))
    column_cfg = _dict(model_cfg.get("columns"))
    training_cfg = _dict(config.get("training"))

    seed = int(run_cfg.get("seed", 1))
    train_steps = int(max_steps if max_steps is not None else run_cfg.get("max_steps", 50))
    learning_rate = float(run_cfg.get("learning_rate", 1e-2))
    dataset = str(data_cfg.get("dataset", "tiny_shakespeare_word"))
    seq_len = int(data_cfg.get("seq_len", 64))
    hidden_dim = int(base_cfg.get("hidden_dim", 96))
    layers = int(base_cfg.get("layers", 2))
    num_columns = int(column_cfg.get("num_columns", 24))
    atoms_per_column = int(column_cfg.get("atoms_per_column", 4))
    top_k = int(column_cfg.get("top_k", 2))
    contextual_router_hidden_dim = int(
        column_cfg.get("contextual_router_hidden_dim", hidden_dim * 2)
    )
    residual_objective = str(training_cfg.get("residual_objective", "supervised_ce"))

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
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base.eval()

    residual = ResidualColumns(
        hidden_dim=hidden_dim,
        num_columns=num_columns,
        atoms_per_column=atoms_per_column,
        top_k=top_k,
        support_router="contextual_mlp",
        contextual_router_hidden_dim=contextual_router_hidden_dim,
    )
    residual.train()
    optimizer = torch.optim.AdamW(residual.parameters(), lr=learning_rate)
    for _ in range(train_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _residual_loss(
            base,
            residual,
            inputs,
            targets,
            vocab_size,
            objective=residual_objective,
        )
        loss.backward()
        optimizer.step()
    residual.eval()

    with torch.no_grad():
        hidden = base.encode(inputs)
        chunks = _contextual_chunks(torch, hidden)
        causal_inputs = _causal_predictor_inputs(torch, chunks)
        position_inputs = _position_predictor_inputs(torch, chunks)
        targets_future = torch.cat([chunks["next"], chunks["next_delta"]], dim=-1)

    predictor = _FuturePredictor(nn, causal_inputs.shape[-1], hidden_dim * 2, hidden_dim)
    gru_predictor = _FutureGRUPredictor(
        nn,
        causal_inputs.shape[-1],
        hidden_dim * 2,
        hidden_dim,
    )
    token_position_predictor = _FuturePredictor(
        nn,
        position_inputs.shape[-1],
        hidden_dim * 2,
        max(8, min(hidden_dim, 64)),
    )
    acsr_predictor_parameter_count = _parameter_count(predictor)
    contextual_router_parameter_count = _parameter_count(residual.contextual_column_scores)
    parameter_matched_width = _matched_mlp_width(
        input_dim=causal_inputs.shape[-1],
        output_dim=num_columns,
        target_parameter_count=acsr_predictor_parameter_count
        + contextual_router_parameter_count,
    )
    parameter_matched_causal_score_mlp = _CausalScoreMLP(
        nn,
        causal_inputs.shape[-1],
        num_columns,
        parameter_matched_width,
    )
    predictor_rows = []
    predictor_rows.append(
        _train_predictor_row(
            torch,
            F,
            predictor,
            causal_inputs,
            targets_future,
            steps=predictor_steps,
            label="mlp_causal",
        )
    )
    predictor_rows.append(
        _train_predictor_row(
            torch,
            F,
            gru_predictor,
            causal_inputs,
            targets_future,
            steps=predictor_steps,
            label="gru_causal",
        )
    )
    predictor_rows.append(
        _train_predictor_row(
            torch,
            F,
            token_position_predictor,
            position_inputs,
            targets_future,
            steps=predictor_steps,
            label="token_position_only",
        )
    )
    parameter_matched_row = _train_causal_score_control_row(
        torch,
        F,
        parameter_matched_causal_score_mlp,
        causal_inputs,
        _score_from_features(residual, _feature_tensor(torch, chunks, targets_future)).detach(),
        steps=predictor_steps,
        label="parameter_matched_causal_mlp_control",
    )
    predictor_rows.append(parameter_matched_row)

    with torch.no_grad():
        predicted = predictor(causal_inputs)
        gru_predicted = gru_predictor(causal_inputs)
        token_position_predicted = token_position_predictor(position_inputs)
        mean_predicted = targets_future.mean(dim=(0, 1), keepdim=True).expand_as(
            targets_future
        )
        zero_predicted = torch.zeros_like(targets_future)
        shuffled_predicted = _shuffle_tokens(torch, predicted)

        variants = {
            "full_context_contextual_topk2_teacher": _feature_tensor(
                torch,
                chunks,
                targets_future,
            ),
            "causal_feature_safe_contextual_topk2": _feature_tensor(
                torch,
                chunks,
                zero_predicted,
            ),
            "acsr_mlp_predicted_future": _feature_tensor(torch, chunks, predicted),
            "acsr_gru_predicted_future": _feature_tensor(torch, chunks, gru_predicted),
            "shuffled_predicted_features": _feature_tensor(
                torch,
                chunks,
                shuffled_predicted,
            ),
            "token_position_only_predicted_features": _feature_tensor(
                torch,
                chunks,
                token_position_predicted,
            ),
            "mean_predicted_features": _feature_tensor(torch, chunks, mean_predicted),
            "zero_predicted_features": _feature_tensor(torch, chunks, zero_predicted),
        }
        score_rows = {
            name: _score_from_features(residual, features)
            for name, features in variants.items()
        }
        score_rows["parameter_matched_causal_mlp_control"] = (
            parameter_matched_causal_score_mlp(causal_inputs)
            + residual.score_tie_breaker.to(device=causal_inputs.device, dtype=causal_inputs.dtype)
        )
        router_rows = [
            _router_metric_row(
                torch,
                F,
                base,
                residual,
                hidden,
                targets,
                vocab_size,
                name,
                scores,
                top_k=top_k,
            )
            for name, scores in score_rows.items()
        ]
        router_rows.append(
            _linear_router_row(
                torch,
                F,
                base,
                residual,
                hidden,
                targets,
                vocab_size,
                top_k=top_k,
            )
        )
        router_rows.append(
            _router_metric_row(
                torch,
                F,
                base,
                residual,
                hidden,
                targets,
                vocab_size,
                "rank_matched_contextual_topk1",
                score_rows["full_context_contextual_topk2_teacher"],
                top_k=1,
            )
        )
        router_rows.append(
            _fixed_support_row(
                torch,
                F,
                base,
                residual,
                hidden,
                targets,
                vocab_size,
                name="random_fixed_topk2",
                seed=seed + 17,
            )
        )
        oracle = _oracle_pair_regret(
            torch,
            F,
            base,
            residual,
            hidden,
            targets,
            vocab_size,
            score_rows["full_context_contextual_topk2_teacher"],
            router_rows,
        )
        same_student_rows = _same_student_rows(
            torch,
            F,
            base,
            residual,
            hidden,
            targets,
            vocab_size,
            score_rows,
            top_k=top_k,
        )
        independent_residual = _train_independent_residual(
            torch,
            F,
            ResidualColumns,
            base,
            inputs,
            targets,
            vocab_size,
            hidden_dim=hidden_dim,
            num_columns=num_columns,
            atoms_per_column=atoms_per_column,
            top_k=top_k,
            contextual_router_hidden_dim=contextual_router_hidden_dim,
            learning_rate=learning_rate,
            train_steps=train_steps,
            residual_objective=residual_objective,
            seed=seed + 1009,
        )
        independent_score_rows = _independent_score_rows(
            torch,
            independent_residual,
            parameter_matched_causal_score_mlp,
            chunks,
            predicted,
            targets_future,
            causal_inputs,
        )
        same_student_rows.extend(
            _dual_student_cross_forcing_rows(
                torch,
                F,
                base,
                residual,
                independent_residual,
                hidden,
                targets,
                vocab_size,
                score_rows,
                independent_score_rows,
                top_k=top_k,
            )
        )
        support_agreement_rows = _support_agreement_rows(
            torch,
            score_rows,
            top_k=top_k,
        )
        perturbation_rows = _future_perturbation_rows(
            torch,
            residual,
            hidden,
            predictor,
            perturb_start=max(2, seq_len // 2),
        )
        perturbation_rows.extend(
            _leaky_future_positive_control_rows(
                torch,
                residual,
                hidden,
                perturb_start=max(2, seq_len // 2),
            )
        )
        sequence_rows = _sequence_heldout_rows(
            torch,
            F,
            base,
            residual,
            hidden,
            targets,
            vocab_size,
            score_rows,
            top_k=top_k,
            holdout_start=max(1, seq_len // 2),
        )
        per_sequence_rows = _per_sequence_paired_delta_rows(
            torch,
            F,
            base,
            residual,
            hidden,
            targets,
            vocab_size,
            score_rows,
            top_k=top_k,
            holdout_start=max(1, seq_len // 2),
        )
        margin_rows = _margin_fragility_rows(
            torch,
            residual,
            variants,
            score_rows,
            top_k=top_k,
            seed=seed + 101,
        )
        parameter_rows = _parameter_count_rows(
            residual=residual,
            predictor=predictor,
            gru_predictor=gru_predictor,
            token_position_predictor=token_position_predictor,
            parameter_matched_causal_score_mlp=parameter_matched_causal_score_mlp,
            hidden_dim=hidden_dim,
            num_columns=num_columns,
            top_k=top_k,
            acsr_predictor_parameter_count=acsr_predictor_parameter_count,
            contextual_router_parameter_count=contextual_router_parameter_count,
            parameter_matched_row=parameter_matched_row,
        )

    transfer_inputs, transfer_targets, transfer_vocab_size = _build_batch(
        dataset=dataset,
        seq_len=seq_len,
        batch_size=8,
    )
    if transfer_vocab_size != vocab_size:
        raise RuntimeError("transfer batch vocab size diverged from anchor batch")
    transfer_inputs = transfer_inputs[4:].detach().clone()
    transfer_targets = transfer_targets[4:].detach().clone()
    retention_rows = _second_context_retention_churn_rows(
        torch,
        F,
        ResidualColumns,
        base,
        residual,
        predictor,
        gru_predictor,
        token_position_predictor,
        inputs,
        targets,
        transfer_inputs,
        transfer_targets,
        vocab_size,
        learning_rate=learning_rate,
        transfer_steps=max(1, min(10, train_steps)),
        top_k=top_k,
        contextual_router_hidden_dim=contextual_router_hidden_dim,
        residual_objective=residual_objective,
    )
    with torch.no_grad():
        retention_rows.extend(
            _teacher_context_churn_rows(
                torch,
                F,
                base,
                residual,
                hidden,
                targets,
                vocab_size,
                score_rows,
                top_k=top_k,
            )
        )

    router_by_name = {row["variant"]: row for row in router_rows}
    failures = _gate_failures(router_by_name, predictor_rows, perturbation_rows)
    status = "pass" if not failures else "fail"
    acsr = router_by_name.get("acsr_mlp_predicted_future", {})
    causal = router_by_name.get("causal_feature_safe_contextual_topk2", {})
    teacher = router_by_name.get("full_context_contextual_topk2_teacher", {})
    summary = {
        "status": status,
        "decision": (
            "anticipatory_contextual_support_routing_smoke_completed"
            if status == "pass"
            else "anticipatory_contextual_support_routing_smoke_failed_gate"
        ),
        "claim_status": (
            "local_acsr_smoke_evidence_recorded"
            if status == "pass"
            else "local_acsr_smoke_not_promotable"
        ),
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "train_steps": train_steps,
        "predictor_steps": predictor_steps,
        "dataset": dataset,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "num_columns": num_columns,
        "top_k": top_k,
        "best_predictor": "mlp_causal",
        "primary_metrics": {
            "acsr_alpha0_ce_loss": acsr.get("ce_loss"),
            "causal_baseline_alpha0_ce_loss": causal.get("ce_loss"),
            "full_context_teacher_alpha0_ce_loss": teacher.get("ce_loss"),
            "acsr_minus_causal_ce_loss": _optional_delta(acsr, causal, "ce_loss"),
            "acsr_minus_teacher_ce_loss": _optional_delta(acsr, teacher, "ce_loss"),
            "acsr_oracle_regret": acsr.get("oracle_regret"),
        },
        "gates": {
            "future_perturbation_invariance": not any(
                not row["passed"]
                for row in perturbation_rows
                if row.get("control_type") != "leaky_future_positive"
            ),
            "leaky_future_positive_control": any(
                row.get("control_type") == "leaky_future_positive" and row["passed"]
                for row in perturbation_rows
            ),
            "sequence_heldout_available": bool(sequence_rows),
            "margin_fragility_available": bool(margin_rows),
            "per_sequence_paired_deltas_available": bool(per_sequence_rows),
            "parameter_matched_causal_control_available": any(
                row.get("component") == "parameter_matched_causal_mlp_control"
                and row.get("status") == "available"
                for row in parameter_rows
            ),
            "support_agreement_available": bool(support_agreement_rows),
            "dual_student_cross_forcing_available": any(
                row.get("forcing_type") == "dual_student_cross_forcing"
                and row.get("status") == "available"
                for row in same_student_rows
            ),
            "acsr_beats_shuffled_ce": _metric_lt(
                router_by_name,
                "acsr_mlp_predicted_future",
                "shuffled_predicted_features",
                "ce_loss",
            ),
            "acsr_beats_token_position_ce": _metric_lt(
                router_by_name,
                "acsr_mlp_predicted_future",
                "token_position_only_predicted_features",
                "ce_loss",
            ),
            "acsr_does_not_worsen_causal_regret": _metric_le(
                router_by_name,
                "acsr_mlp_predicted_future",
                "causal_feature_safe_contextual_topk2",
                "oracle_regret",
            ),
        },
        "failures": failures,
        "oracle_pair_regret": oracle,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(
        out_dir,
        summary,
        predictor_rows=predictor_rows,
        router_rows=router_rows,
        same_student_rows=same_student_rows,
        support_agreement_rows=support_agreement_rows,
        perturbation_rows=perturbation_rows,
        sequence_rows=sequence_rows,
        per_sequence_rows=per_sequence_rows,
        margin_rows=margin_rows,
        parameter_rows=parameter_rows,
        retention_rows=retention_rows,
    )
    return summary


def _contextual_chunks(torch: Any, hidden: Any) -> dict[str, Any]:
    current = hidden.detach()
    previous = torch.cat([current[:, :1, :], current[:, :-1, :]], dim=1)
    next_hidden = torch.cat([current[:, 1:, :], current[:, -1:, :]], dim=1)
    seq_len = int(current.shape[1])
    if seq_len <= 1:
        normalized_position = torch.zeros(
            current.shape[0], seq_len, 1, dtype=current.dtype, device=current.device
        )
    else:
        normalized_position = torch.linspace(
            0.0, 1.0, seq_len, dtype=current.dtype, device=current.device
        ).view(1, seq_len, 1).expand(current.shape[0], seq_len, 1)
    angle = normalized_position * (2.0 * torch.pi)
    return {
        "current": current,
        "previous": previous,
        "next": next_hidden,
        "previous_delta": current - previous,
        "next_delta": next_hidden - current,
        "position": normalized_position,
        "sin_position": torch.sin(angle),
        "cos_position": torch.cos(angle),
    }


def _causal_predictor_inputs(torch: Any, chunks: dict[str, Any]) -> Any:
    return torch.cat(
        [
            chunks["current"],
            chunks["previous"],
            chunks["previous_delta"],
            chunks["position"],
            chunks["sin_position"],
            chunks["cos_position"],
        ],
        dim=-1,
    )


def _position_predictor_inputs(torch: Any, chunks: dict[str, Any]) -> Any:
    return torch.cat(
        [chunks["position"], chunks["sin_position"], chunks["cos_position"]],
        dim=-1,
    )


def _feature_tensor(torch: Any, chunks: dict[str, Any], future_pair: Any) -> Any:
    hidden_dim = int(chunks["current"].shape[-1])
    return torch.cat(
        [
            chunks["current"],
            chunks["previous"],
            future_pair[..., :hidden_dim],
            chunks["previous_delta"],
            future_pair[..., hidden_dim:],
            chunks["position"],
            chunks["sin_position"],
            chunks["cos_position"],
        ],
        dim=-1,
    )


class _FuturePredictor:
    def __new__(
        cls,
        nn: Any,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
    ) -> Any:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )


class _FutureGRUPredictor:
    def __new__(
        cls,
        nn: Any,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
    ) -> Any:
        class _CausalGRUPredictor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_norm = nn.LayerNorm(input_dim)
                self.gru = nn.GRU(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    batch_first=True,
                )
                self.output = nn.Linear(hidden_dim, output_dim)

            def forward(self, inputs: Any) -> Any:
                normalized = self.input_norm(inputs)
                states, _ = self.gru(normalized)
                return self.output(states)

        return _CausalGRUPredictor()


class _CausalScoreMLP:
    def __new__(
        cls,
        nn: Any,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
    ) -> Any:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )


def _matched_mlp_width(
    *,
    input_dim: int,
    output_dim: int,
    target_parameter_count: int,
) -> int:
    best_width = 1
    best_gap = None
    for width in range(1, 2049):
        candidate = input_dim * 2 + input_dim * width + width + width * output_dim
        gap = abs(candidate - target_parameter_count)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_width = width
    return best_width


def _train_predictor_row(
    torch: Any,
    F: Any,
    predictor: Any,
    inputs: Any,
    targets: Any,
    *,
    steps: int,
    label: str,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=3e-3)
    split = max(1, int(inputs.shape[1]) // 2)
    train_x = inputs[:, :split, :]
    train_y = targets[:, :split, :]
    holdout_x = inputs[:, split:, :]
    holdout_y = targets[:, split:, :]
    for _ in range(max(1, steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(predictor(train_x), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = predictor(holdout_x)
        mse = F.mse_loss(pred, holdout_y)
        centered_target = holdout_y - holdout_y.mean()
        sst = centered_target.pow(2).mean().clamp_min(1e-12)
        r2 = 1.0 - mse / sst
        cosine = F.cosine_similarity(
            pred.reshape(-1, pred.shape[-1]),
            holdout_y.reshape(-1, holdout_y.shape[-1]),
            dim=-1,
        ).mean()
    return {
        "predictor": label,
        "train_positions": int(split),
        "holdout_positions": int(inputs.shape[1] - split),
        "holdout_mse": float(mse.item()),
        "holdout_r2": float(r2.item()),
        "holdout_cosine": float(cosine.item()),
    }


def _train_causal_score_control_row(
    torch: Any,
    F: Any,
    model: Any,
    inputs: Any,
    target_scores: Any,
    *,
    steps: int,
    label: str,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    split = max(1, int(inputs.shape[1]) // 2)
    train_x = inputs[:, :split, :]
    train_y = target_scores[:, :split, :]
    holdout_x = inputs[:, split:, :]
    holdout_y = target_scores[:, split:, :]
    for _ in range(max(1, steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = model(holdout_x)
        mse = F.mse_loss(pred, holdout_y)
        centered_target = holdout_y - holdout_y.mean()
        sst = centered_target.pow(2).mean().clamp_min(1e-12)
        r2 = 1.0 - mse / sst
        support_match = (
            pred.topk(min(2, pred.shape[-1]), dim=-1).indices
            == holdout_y.topk(min(2, holdout_y.shape[-1]), dim=-1).indices
        ).all(dim=-1).to(dtype=pred.dtype).mean()
    return {
        "predictor": label,
        "train_positions": int(split),
        "holdout_positions": int(inputs.shape[1] - split),
        "holdout_mse": float(mse.item()),
        "holdout_r2": float(r2.item()),
        "holdout_topk_support_match": float(support_match.item()),
    }


def _score_from_features(residual: Any, features: Any) -> Any:
    return residual.contextual_column_scores(features) + residual.score_tie_breaker.to(
        device=features.device, dtype=features.dtype
    )


def _router_metric_row(
    torch: Any,
    F: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    variant: str,
    scores: Any,
    *,
    top_k: int,
) -> dict[str, Any]:
    top_values, support = scores.topk(top_k, dim=-1)
    logits = _decode_for_support(torch, base, residual, hidden, support, top_values)
    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab_size), targets[:, :-1].reshape(-1))
    return {
        "variant": variant,
        "top_k": int(top_k),
        "ce_loss": float(loss.item()),
        "used_columns": _used_columns(support),
        "unique_support_sets": _unique_support_sets(support),
        "support_entropy": _support_entropy(torch, support, residual.num_columns),
        "mean_topk_margin": _mean_topk_margin(scores, top_k),
        "oracle_regret": "",
    }


def _linear_router_row(
    torch: Any,
    F: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    *,
    top_k: int,
) -> dict[str, Any]:
    scores = residual.column_scores(hidden) + residual.score_tie_breaker.to(
        device=hidden.device, dtype=hidden.dtype
    )
    return _router_metric_row(
        torch,
        F,
        base,
        residual,
        hidden,
        targets,
        vocab_size,
        "linear_topk2_same_values",
        scores,
        top_k=top_k,
    )


def _fixed_support_row(
    torch: Any,
    F: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    *,
    name: str,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=hidden.device)
    generator.manual_seed(seed)
    support = torch.randint(
        low=0,
        high=residual.num_columns,
        size=(hidden.shape[0], hidden.shape[1], residual.top_k),
        generator=generator,
        device=hidden.device,
    )
    scores = torch.zeros(
        hidden.shape[0],
        hidden.shape[1],
        residual.num_columns,
        dtype=hidden.dtype,
        device=hidden.device,
    )
    top_values = scores.gather(dim=-1, index=support)
    logits = _decode_for_support(torch, base, residual, hidden, support, top_values)
    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab_size), targets[:, :-1].reshape(-1))
    return {
        "variant": name,
        "top_k": int(residual.top_k),
        "ce_loss": float(loss.item()),
        "used_columns": _used_columns(support),
        "unique_support_sets": _unique_support_sets(support),
        "support_entropy": _support_entropy(torch, support, residual.num_columns),
        "mean_topk_margin": 0.0,
        "oracle_regret": "",
    }


def _decode_for_support(
    torch: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    support: Any,
    top_values: Any,
) -> Any:
    column_weights = torch.softmax(top_values, dim=-1)
    atom_weights = torch.softmax(residual.atom_logits, dim=-1)
    column_values = torch.einsum("ca,cah->ch", atom_weights, residual.atom_values)
    selected_values = column_values[support]
    residual_update = torch.einsum("bsk,bskh->bsh", column_weights, selected_values)
    return base.decode(hidden + residual_update)


def _oracle_pair_regret(
    torch: Any,
    F: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    reference_scores: Any,
    router_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    losses = []
    target_flat = targets[:, :-1].reshape(-1)
    for left in range(residual.num_columns):
        for right in range(left + 1, residual.num_columns):
            support = torch.empty(
                hidden.shape[0],
                hidden.shape[1],
                2,
                dtype=torch.long,
                device=hidden.device,
            )
            support[..., 0] = left
            support[..., 1] = right
            top_values = reference_scores.gather(dim=-1, index=support)
            logits = _decode_for_support(torch, base, residual, hidden, support, top_values)
            per_token = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, vocab_size),
                target_flat,
                reduction="none",
            )
            losses.append(per_token)
    if not losses:
        return {"evaluated_pairs": 0, "oracle_loss": None}
    pair_losses = torch.stack(losses, dim=0)
    oracle_losses = pair_losses.min(dim=0).values
    oracle_loss = float(oracle_losses.mean().item())
    for row in router_rows:
        if int(row.get("top_k", 0)) != 2:
            continue
        row["oracle_regret"] = float(float(row["ce_loss"]) - oracle_loss)
    return {
        "evaluated_pairs": len(losses),
        "oracle_loss": oracle_loss,
    }


def _same_student_rows(
    torch: Any,
    F: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    score_rows: dict[str, Any],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    rows = []
    for baseline_name in [
        "acsr_mlp_predicted_future",
        "acsr_gru_predicted_future",
    ]:
        baseline = _forced_support_loss(
            torch,
            F,
            base,
            residual,
            hidden,
            targets,
            vocab_size,
            score_rows[baseline_name],
            top_k=top_k,
        )
        for control in [
            "shuffled_predicted_features",
            "token_position_only_predicted_features",
            "mean_predicted_features",
            "zero_predicted_features",
            "parameter_matched_causal_mlp_control",
        ]:
            control_loss = _forced_support_loss(
                torch,
                F,
                base,
                residual,
                hidden,
                targets,
                vocab_size,
                score_rows[control],
                top_k=top_k,
            )
            rows.append(
                {
                    "forcing_type": "same_student",
                    "status": "available",
                    "comparison": f"{baseline_name}_support_vs_{control}",
                    "acsr_forced_ce_loss": baseline,
                    "control_forced_ce_loss": control_loss,
                    "acsr_minus_control_ce_loss": baseline - control_loss,
                }
            )
    return rows


def _train_independent_residual(
    torch: Any,
    F: Any,
    ResidualColumns: Any,
    base: Any,
    inputs: Any,
    targets: Any,
    vocab_size: int,
    *,
    hidden_dim: int,
    num_columns: int,
    atoms_per_column: int,
    top_k: int,
    contextual_router_hidden_dim: int,
    learning_rate: float,
    train_steps: int,
    residual_objective: str,
    seed: int,
) -> Any:
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        residual = ResidualColumns(
            hidden_dim=hidden_dim,
            num_columns=num_columns,
            atoms_per_column=atoms_per_column,
            top_k=top_k,
            support_router="contextual_mlp",
            contextual_router_hidden_dim=contextual_router_hidden_dim,
        )
    finally:
        torch.random.set_rng_state(rng_state)
    residual.train()
    optimizer = torch.optim.AdamW(residual.parameters(), lr=learning_rate)
    with torch.enable_grad():
        for _ in range(train_steps):
            optimizer.zero_grad(set_to_none=True)
            logits = base(inputs, residual_adapter=residual)
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, vocab_size),
                targets[:, :-1].reshape(-1),
            )
            if residual_objective != "supervised_ce":
                loss = loss + 0.0 * logits.mean()
            loss.backward()
            optimizer.step()
    residual.eval()
    return residual


def _independent_score_rows(
    torch: Any,
    residual: Any,
    parameter_matched_causal_score_mlp: Any,
    chunks: dict[str, Any],
    predicted: Any,
    targets_future: Any,
    causal_inputs: Any,
) -> dict[str, Any]:
    zero_predicted = torch.zeros_like(targets_future)
    return {
        "causal_feature_safe_contextual_topk2": _score_from_features(
            residual,
            _feature_tensor(torch, chunks, zero_predicted),
        ),
        "acsr_mlp_predicted_future": _score_from_features(
            residual,
            _feature_tensor(torch, chunks, predicted),
        ),
        "parameter_matched_causal_mlp_control": (
            parameter_matched_causal_score_mlp(causal_inputs)
            + residual.score_tie_breaker.to(device=causal_inputs.device, dtype=causal_inputs.dtype)
        ),
    }


def _dual_student_cross_forcing_rows(
    torch: Any,
    F: Any,
    base: Any,
    student_a: Any,
    student_b: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    scores_a: dict[str, Any],
    scores_b: dict[str, Any],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    rows = []
    primary = "acsr_mlp_predicted_future"
    control = "parameter_matched_causal_mlp_control"
    for target_student, target_residual, target_scores in [
        ("student_a_values", student_a, scores_a),
        ("student_b_values", student_b, scores_b),
    ]:
        acsr_loss = _loss_with_source_support(
            torch,
            F,
            base,
            target_residual,
            hidden,
            targets,
            vocab_size,
            source_scores=scores_a[primary],
            target_scores=target_scores[primary],
            top_k=top_k,
        )
        control_loss = _loss_with_source_support(
            torch,
            F,
            base,
            target_residual,
            hidden,
            targets,
            vocab_size,
            source_scores=scores_a[control],
            target_scores=target_scores[control],
            top_k=top_k,
        )
        rows.append(
            {
                "forcing_type": "dual_student_cross_forcing",
                "status": "available",
                "comparison": f"{primary}_support_vs_{control}",
                "target_student": target_student,
                "acsr_forced_ce_loss": acsr_loss,
                "control_forced_ce_loss": control_loss,
                "acsr_minus_control_ce_loss": acsr_loss - control_loss,
            }
        )
    return rows


def _loss_with_source_support(
    torch: Any,
    F: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    *,
    source_scores: Any,
    target_scores: Any,
    top_k: int,
) -> float:
    support = source_scores.topk(top_k, dim=-1).indices
    top_values = target_scores.gather(dim=-1, index=support)
    logits = _decode_for_support(torch, base, residual, hidden, support, top_values)
    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab_size), targets[:, :-1].reshape(-1))
    return float(loss.item())


def _support_agreement_rows(
    torch: Any,
    score_rows: dict[str, Any],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    primary = "acsr_mlp_predicted_future"
    primary_support = score_rows[primary].topk(top_k, dim=-1).indices
    rows = []
    for control in [
        "parameter_matched_causal_mlp_control",
        "shuffled_predicted_features",
        "token_position_only_predicted_features",
        "mean_predicted_features",
        "zero_predicted_features",
    ]:
        control_support = score_rows[control].topk(top_k, dim=-1).indices
        slot_match = (primary_support == control_support).to(dtype=score_rows[primary].dtype)
        set_match = (
            primary_support.sort(dim=-1).values == control_support.sort(dim=-1).values
        ).all(dim=-1)
        set_match_float = set_match.to(dtype=score_rows[primary].dtype).mean()
        rows.append(
            {
                "comparison": f"{primary}_support_vs_{control}",
                "status": "available",
                "top_k": int(top_k),
                "slot_match_fraction": float(slot_match.mean().item()),
                "set_match_fraction": float(set_match_float.item()),
                "changed_support_fraction": float(1.0 - set_match_float.item()),
            }
        )
    return rows


def _forced_support_loss(
    torch: Any,
    F: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    scores: Any,
    *,
    top_k: int,
) -> float:
    top_values, support = scores.topk(top_k, dim=-1)
    logits = _decode_for_support(torch, base, residual, hidden, support, top_values)
    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab_size), targets[:, :-1].reshape(-1))
    return float(loss.item())


def _future_perturbation_rows(
    torch: Any,
    residual: Any,
    hidden: Any,
    predictor: Any,
    *,
    perturb_start: int,
) -> list[dict[str, Any]]:
    perturbed = hidden.clone()
    perturbed[:, perturb_start:, :] = torch.flip(perturbed[:, perturb_start:, :], dims=[1])
    base_chunks = _contextual_chunks(torch, hidden)
    perturbed_chunks = _contextual_chunks(torch, perturbed)
    base_pred = predictor(_causal_predictor_inputs(torch, base_chunks))
    perturbed_pred = predictor(_causal_predictor_inputs(torch, perturbed_chunks))
    base_scores = _score_from_features(residual, _feature_tensor(torch, base_chunks, base_pred))
    perturbed_scores = _score_from_features(
        residual,
        _feature_tensor(torch, perturbed_chunks, perturbed_pred),
    )
    earlier = slice(0, perturb_start)
    score_delta = float(
        (base_scores[:, earlier, :] - perturbed_scores[:, earlier, :]).abs().max().item()
    )
    pred_delta = float(
        (base_pred[:, earlier, :] - perturbed_pred[:, earlier, :]).abs().max().item()
    )
    support_same = bool(
        torch.equal(
            base_scores[:, earlier, :].topk(residual.top_k, dim=-1).indices,
            perturbed_scores[:, earlier, :].topk(residual.top_k, dim=-1).indices,
        )
    )
    return [
        {
            "control_type": "future_perturbation_negative",
            "check": "future_positions_do_not_change_prefix_predictions_or_support",
            "perturb_start": int(perturb_start),
            "checked_prefix_positions": int(perturb_start),
            "max_predicted_feature_delta": pred_delta,
            "max_router_score_delta": score_delta,
            "support_unchanged": support_same,
            "passed": pred_delta <= 1e-8 and score_delta <= 1e-8 and support_same,
        }
    ]


def _leaky_future_positive_control_rows(
    torch: Any,
    residual: Any,
    hidden: Any,
    *,
    perturb_start: int,
) -> list[dict[str, Any]]:
    perturbed = hidden.clone()
    perturbed[:, perturb_start:, :] = torch.flip(perturbed[:, perturb_start:, :], dims=[1])
    base_chunks = _contextual_chunks(torch, hidden)
    perturbed_chunks = _contextual_chunks(torch, perturbed)
    base_future = torch.cat([base_chunks["next"], base_chunks["next_delta"]], dim=-1)
    perturbed_future = torch.cat(
        [perturbed_chunks["next"], perturbed_chunks["next_delta"]],
        dim=-1,
    )
    base_scores = _score_from_features(residual, _feature_tensor(torch, base_chunks, base_future))
    perturbed_scores = _score_from_features(
        residual,
        _feature_tensor(torch, perturbed_chunks, perturbed_future),
    )
    earlier = slice(0, perturb_start)
    score_delta = float(
        (base_scores[:, earlier, :] - perturbed_scores[:, earlier, :]).abs().max().item()
    )
    feature_delta = float(
        (base_future[:, earlier, :] - perturbed_future[:, earlier, :]).abs().max().item()
    )
    support_same = bool(
        torch.equal(
            base_scores[:, earlier, :].topk(residual.top_k, dim=-1).indices,
            perturbed_scores[:, earlier, :].topk(residual.top_k, dim=-1).indices,
        )
    )
    return [
        {
            "control_type": "leaky_future_positive",
            "check": "full_context_features_detect_future_perturbation",
            "perturb_start": int(perturb_start),
            "checked_prefix_positions": int(perturb_start),
            "max_predicted_feature_delta": feature_delta,
            "max_router_score_delta": score_delta,
            "support_unchanged": support_same,
            "passed": feature_delta > 1e-8 or score_delta > 1e-8 or not support_same,
        }
    ]


def _sequence_heldout_rows(
    torch: Any,
    F: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    score_rows: dict[str, Any],
    *,
    top_k: int,
    holdout_start: int,
) -> list[dict[str, Any]]:
    rows = []
    token_slice = slice(holdout_start, max(holdout_start + 1, hidden.shape[1] - 1))
    target_slice = targets[:, token_slice].reshape(-1)
    teacher_scores = score_rows["full_context_contextual_topk2_teacher"]
    oracle_loss = _oracle_loss_for_slice(
        torch,
        F,
        base,
        residual,
        hidden,
        target_slice,
        vocab_size,
        teacher_scores,
        token_slice=token_slice,
    )
    for variant, scores in score_rows.items():
        if variant == "full_context_contextual_topk2_teacher" or top_k != 2:
            variant_top_k = min(top_k, int(scores.shape[-1]))
        else:
            variant_top_k = top_k
        top_values, support = scores.topk(variant_top_k, dim=-1)
        logits = _decode_for_support(torch, base, residual, hidden, support, top_values)
        loss = F.cross_entropy(
            logits[:, token_slice, :].reshape(-1, vocab_size),
            target_slice,
        )
        rows.append(
            {
                "split": "sequence_suffix_holdout",
                "variant": variant,
                "top_k": int(variant_top_k),
                "holdout_start": int(holdout_start),
                "heldout_positions": int(logits[:, token_slice, :].shape[1]),
                "ce_loss": float(loss.item()),
                "oracle_loss": oracle_loss,
                "oracle_regret": float(loss.item()) - oracle_loss,
            }
        )
    return rows


def _per_sequence_paired_delta_rows(
    torch: Any,
    F: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    score_rows: dict[str, Any],
    *,
    top_k: int,
    holdout_start: int,
) -> list[dict[str, Any]]:
    primary = "acsr_mlp_predicted_future"
    control = "parameter_matched_causal_mlp_control"
    if primary not in score_rows or control not in score_rows or top_k != 2:
        return []
    token_slice = slice(holdout_start, max(holdout_start + 1, hidden.shape[1] - 1))
    oracle_loss_by_sequence = _oracle_loss_by_sequence_for_slice(
        torch,
        F,
        base,
        residual,
        hidden,
        targets,
        vocab_size,
        score_rows["full_context_contextual_topk2_teacher"],
        token_slice=token_slice,
    )
    acsr_loss_by_sequence = _loss_by_sequence_for_scores(
        torch,
        F,
        base,
        residual,
        hidden,
        targets,
        vocab_size,
        score_rows[primary],
        top_k=top_k,
        token_slice=token_slice,
    )
    control_loss_by_sequence = _loss_by_sequence_for_scores(
        torch,
        F,
        base,
        residual,
        hidden,
        targets,
        vocab_size,
        score_rows[control],
        top_k=top_k,
        token_slice=token_slice,
    )
    rows = []
    heldout_positions = int(hidden[:, token_slice, :].shape[1])
    for sequence_index in range(int(hidden.shape[0])):
        acsr_ce = float(acsr_loss_by_sequence[sequence_index].item())
        control_ce = float(control_loss_by_sequence[sequence_index].item())
        oracle_loss = float(oracle_loss_by_sequence[sequence_index].item())
        acsr_regret = acsr_ce - oracle_loss
        control_regret = control_ce - oracle_loss
        rows.append(
            {
                "split": "sequence_suffix_holdout",
                "sequence_index": int(sequence_index),
                "top_k": int(top_k),
                "holdout_start": int(holdout_start),
                "heldout_positions": heldout_positions,
                "acsr_ce_loss": acsr_ce,
                "parameter_matched_ce_loss": control_ce,
                "acsr_minus_parameter_matched_ce_loss": acsr_ce - control_ce,
                "oracle_loss": oracle_loss,
                "acsr_oracle_regret": acsr_regret,
                "parameter_matched_oracle_regret": control_regret,
                "acsr_minus_parameter_matched_oracle_regret": acsr_regret
                - control_regret,
                "acsr_strictly_better": acsr_ce < control_ce
                and acsr_regret < control_regret,
            }
        )
    return rows


def _loss_by_sequence_for_scores(
    torch: Any,
    F: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    scores: Any,
    *,
    top_k: int,
    token_slice: slice,
) -> Any:
    top_values, support = scores.topk(top_k, dim=-1)
    logits = _decode_for_support(torch, base, residual, hidden, support, top_values)
    per_token = F.cross_entropy(
        logits[:, token_slice, :].reshape(-1, vocab_size),
        targets[:, token_slice].reshape(-1),
        reduction="none",
    )
    return per_token.view(hidden.shape[0], -1).mean(dim=1)


def _oracle_loss_by_sequence_for_slice(
    torch: Any,
    F: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    reference_scores: Any,
    *,
    token_slice: slice,
) -> Any:
    losses = []
    for left in range(residual.num_columns):
        for right in range(left + 1, residual.num_columns):
            support = torch.empty(
                hidden.shape[0],
                hidden.shape[1],
                2,
                dtype=torch.long,
                device=hidden.device,
            )
            support[..., 0] = left
            support[..., 1] = right
            top_values = reference_scores.gather(dim=-1, index=support)
            logits = _decode_for_support(torch, base, residual, hidden, support, top_values)
            per_token = F.cross_entropy(
                logits[:, token_slice, :].reshape(-1, vocab_size),
                targets[:, token_slice].reshape(-1),
                reduction="none",
            )
            losses.append(per_token.view(hidden.shape[0], -1))
    return torch.stack(losses, dim=0).min(dim=0).values.mean(dim=1)


def _oracle_loss_for_slice(
    torch: Any,
    F: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    target_flat: Any,
    vocab_size: int,
    reference_scores: Any,
    *,
    token_slice: slice,
) -> float:
    losses = []
    for left in range(residual.num_columns):
        for right in range(left + 1, residual.num_columns):
            support = torch.empty(
                hidden.shape[0],
                hidden.shape[1],
                2,
                dtype=torch.long,
                device=hidden.device,
            )
            support[..., 0] = left
            support[..., 1] = right
            top_values = reference_scores.gather(dim=-1, index=support)
            logits = _decode_for_support(torch, base, residual, hidden, support, top_values)
            per_token = F.cross_entropy(
                logits[:, token_slice, :].reshape(-1, vocab_size),
                target_flat,
                reduction="none",
            )
            losses.append(per_token)
    pair_losses = torch.stack(losses, dim=0)
    return float(pair_losses.min(dim=0).values.mean().item())


def _margin_fragility_rows(
    torch: Any,
    residual: Any,
    variants: dict[str, Any],
    score_rows: dict[str, Any],
    *,
    top_k: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    generator = torch.Generator(device=next(iter(variants.values())).device)
    generator.manual_seed(seed)
    for variant, features in variants.items():
        scores = score_rows[variant]
        base_support = scores.topk(top_k, dim=-1).indices
        feature_scale = features.detach().std().clamp_min(1e-6)
        noise = torch.randn(
            features.shape,
            generator=generator,
            device=features.device,
            dtype=features.dtype,
        ) * (0.01 * feature_scale)
        noisy_scores = _score_from_features(residual, features + noise)
        noisy_support = noisy_scores.topk(top_k, dim=-1).indices
        margin = _topk_margin_tensor(scores, top_k)
        low_margin_threshold = torch.quantile(margin.reshape(-1), 0.25)
        low_margin_mask = margin <= low_margin_threshold
        support_changed = (base_support != noisy_support).any(dim=-1)
        if bool(low_margin_mask.any().item()):
            low_margin_flip_rate = float(
                support_changed[low_margin_mask].to(dtype=features.dtype).mean().item()
            )
        else:
            low_margin_flip_rate = 0.0
        rows.append(
            {
                "variant": variant,
                "top_k": int(top_k),
                "noise_kind": "gaussian_feature_noise",
                "noise_scale_fraction": 0.01,
                "mean_topk_margin": float(margin.mean().item()),
                "p25_topk_margin": float(low_margin_threshold.item()),
                "feature_noise_flip_rate": float(
                    support_changed.to(dtype=features.dtype).mean().item()
                ),
                "low_margin_feature_noise_flip_rate": low_margin_flip_rate,
            }
        )
    return rows


def _topk_margin_tensor(scores: Any, top_k: int) -> Any:
    if scores.shape[-1] <= top_k:
        return scores.new_zeros(scores.shape[:-1])
    sorted_scores = scores.sort(dim=-1, descending=True).values
    return sorted_scores[..., top_k - 1] - sorted_scores[..., top_k]


def _second_context_retention_churn_rows(
    torch: Any,
    F: Any,
    ResidualColumns: Any,
    base: Any,
    residual: Any,
    predictor: Any,
    gru_predictor: Any,
    token_position_predictor: Any,
    anchor_inputs: Any,
    anchor_targets: Any,
    transfer_inputs: Any,
    transfer_targets: Any,
    vocab_size: int,
    *,
    learning_rate: float,
    transfer_steps: int,
    top_k: int,
    contextual_router_hidden_dim: int,
    residual_objective: str,
) -> list[dict[str, Any]]:
    if residual_objective != "supervised_ce":
        return [
            {
                "phase": "second_context_transfer",
                "variant": "not_run",
                "reason": "second-context ACSR microtest currently requires supervised_ce",
            }
        ]

    transfer_residual = ResidualColumns(
        hidden_dim=int(residual.atom_values.shape[-1]),
        num_columns=int(residual.num_columns),
        atoms_per_column=int(residual.atom_logits.shape[-1]),
        top_k=int(residual.top_k),
        support_router="contextual_mlp",
        contextual_router_hidden_dim=contextual_router_hidden_dim,
    )
    transfer_residual.load_state_dict(deepcopy(residual.state_dict()))
    transfer_residual.train()
    optimizer = torch.optim.AdamW(transfer_residual.parameters(), lr=learning_rate)
    for _ in range(max(1, transfer_steps)):
        optimizer.zero_grad(set_to_none=True)
        logits = base(transfer_inputs, residual_adapter=transfer_residual)
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, vocab_size),
            transfer_targets[:, :-1].reshape(-1),
        )
        loss.backward()
        optimizer.step()
    transfer_residual.eval()

    with torch.no_grad():
        anchor_hidden = base.encode(anchor_inputs)
        transfer_hidden = base.encode(transfer_inputs)
        before_anchor_scores = _acsr_score_rows_for_hidden(
            torch,
            residual,
            predictor,
            gru_predictor,
            token_position_predictor,
            anchor_hidden,
        )
        after_anchor_scores = _acsr_score_rows_for_hidden(
            torch,
            transfer_residual,
            predictor,
            gru_predictor,
            token_position_predictor,
            anchor_hidden,
        )
        before_transfer_scores = _acsr_score_rows_for_hidden(
            torch,
            residual,
            predictor,
            gru_predictor,
            token_position_predictor,
            transfer_hidden,
        )
        after_transfer_scores = _acsr_score_rows_for_hidden(
            torch,
            transfer_residual,
            predictor,
            gru_predictor,
            token_position_predictor,
            transfer_hidden,
        )

        rows = []
        for variant in [
            "causal_feature_safe_contextual_topk2",
            "acsr_mlp_predicted_future",
            "acsr_gru_predicted_future",
            "shuffled_predicted_features",
            "token_position_only_predicted_features",
        ]:
            before_scores = before_anchor_scores[variant]
            after_scores = after_anchor_scores[variant]
            before_support = before_scores.topk(top_k, dim=-1).indices
            after_support = after_scores.topk(top_k, dim=-1).indices
            before_anchor_logits = _decode_for_support(
                torch,
                base,
                residual,
                anchor_hidden,
                before_support,
                before_scores.gather(dim=-1, index=before_support),
            )
            after_anchor_logits = _decode_for_support(
                torch,
                base,
                transfer_residual,
                anchor_hidden,
                after_support,
                after_scores.gather(dim=-1, index=after_support),
            )
            before_transfer_loss = _forced_support_loss(
                torch,
                F,
                base,
                residual,
                transfer_hidden,
                transfer_targets,
                vocab_size,
                before_transfer_scores[variant],
                top_k=top_k,
            )
            after_transfer_loss = _forced_support_loss(
                torch,
                F,
                base,
                transfer_residual,
                transfer_hidden,
                transfer_targets,
                vocab_size,
                after_transfer_scores[variant],
                top_k=top_k,
            )
            before_anchor_loss = F.cross_entropy(
                before_anchor_logits[:, :-1, :].reshape(-1, vocab_size),
                anchor_targets[:, :-1].reshape(-1),
            )
            after_anchor_loss = F.cross_entropy(
                after_anchor_logits[:, :-1, :].reshape(-1, vocab_size),
                anchor_targets[:, :-1].reshape(-1),
            )
            rows.append(
                {
                    "phase": "second_context_transfer",
                    "variant": variant,
                    "transfer_steps": int(transfer_steps),
                    "anchor_ce_before_transfer": float(before_anchor_loss.item()),
                    "anchor_ce_after_transfer": float(after_anchor_loss.item()),
                    "anchor_ce_drift": float(
                        after_anchor_loss.item() - before_anchor_loss.item()
                    ),
                    "transfer_ce_before_update": before_transfer_loss,
                    "transfer_ce_after_update": after_transfer_loss,
                    "transfer_ce_improvement": before_transfer_loss
                    - after_transfer_loss,
                    "anchor_support_churn_after_transfer": float(
                        (before_support != after_support)
                        .to(dtype=anchor_hidden.dtype)
                        .mean()
                        .item()
                    ),
                    "anchor_logit_mse_after_transfer": float(
                        F.mse_loss(after_anchor_logits, before_anchor_logits).item()
                    ),
                }
            )
    return rows


def _acsr_score_rows_for_hidden(
    torch: Any,
    residual: Any,
    predictor: Any,
    gru_predictor: Any,
    token_position_predictor: Any,
    hidden: Any,
) -> dict[str, Any]:
    chunks = _contextual_chunks(torch, hidden)
    causal_inputs = _causal_predictor_inputs(torch, chunks)
    position_inputs = _position_predictor_inputs(torch, chunks)
    targets_future = torch.cat([chunks["next"], chunks["next_delta"]], dim=-1)
    predicted = predictor(causal_inputs)
    gru_predicted = gru_predictor(causal_inputs)
    token_position_predicted = token_position_predictor(position_inputs)
    zero_predicted = torch.zeros_like(targets_future)
    shuffled_predicted = _shuffle_tokens(torch, predicted)
    return {
        "causal_feature_safe_contextual_topk2": _score_from_features(
            residual,
            _feature_tensor(torch, chunks, zero_predicted),
        ),
        "acsr_mlp_predicted_future": _score_from_features(
            residual,
            _feature_tensor(torch, chunks, predicted),
        ),
        "acsr_gru_predicted_future": _score_from_features(
            residual,
            _feature_tensor(torch, chunks, gru_predicted),
        ),
        "shuffled_predicted_features": _score_from_features(
            residual,
            _feature_tensor(torch, chunks, shuffled_predicted),
        ),
        "token_position_only_predicted_features": _score_from_features(
            residual,
            _feature_tensor(torch, chunks, token_position_predicted),
        ),
    }


def _teacher_context_churn_rows(
    torch: Any,
    F: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    score_rows: dict[str, Any],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    teacher_scores = score_rows["full_context_contextual_topk2_teacher"]
    rows = []
    for variant in [
        "causal_feature_safe_contextual_topk2",
        "acsr_mlp_predicted_future",
        "acsr_gru_predicted_future",
        "shuffled_predicted_features",
        "token_position_only_predicted_features",
    ]:
        scores = score_rows[variant]
        support = scores.topk(top_k, dim=-1).indices
        teacher_support = teacher_scores.topk(top_k, dim=-1).indices
        support_churn = float((support != teacher_support).to(dtype=hidden.dtype).mean().item())
        logits = _decode_for_support(
            torch,
            base,
            residual,
            hidden,
            support,
            scores.gather(dim=-1, index=support),
        )
        teacher_logits = _decode_for_support(
            torch,
            base,
            residual,
            hidden,
            teacher_support,
            teacher_scores.gather(dim=-1, index=teacher_support),
        )
        logit_mse = float(F.mse_loss(logits, teacher_logits).item())
        rows.append(
            {
                "phase": "fixed_context_teacher_reference",
                "variant": variant,
                "teacher_support_churn": support_churn,
                "teacher_logit_mse": logit_mse,
            }
        )
    return rows


def _parameter_count_rows(
    *,
    residual: Any,
    predictor: Any,
    gru_predictor: Any,
    token_position_predictor: Any,
    parameter_matched_causal_score_mlp: Any,
    hidden_dim: int,
    num_columns: int,
    top_k: int,
    acsr_predictor_parameter_count: int,
    contextual_router_parameter_count: int,
    parameter_matched_row: dict[str, Any],
) -> list[dict[str, Any]]:
    residual_count = _parameter_count(residual)
    active_sparse_proxy = top_k * hidden_dim * int(residual.atom_logits.shape[-1])
    control_count = _parameter_count(parameter_matched_causal_score_mlp)
    target_count = acsr_predictor_parameter_count + contextual_router_parameter_count
    return [
        {
            "component": "residual_columns",
            "active_parameter_count": active_sparse_proxy,
            "stored_parameter_count": residual_count,
            "basis": f"hidden_dim={hidden_dim};num_columns={num_columns};top_k={top_k}",
            "status": "available",
        },
        {
            "component": "acsr_mlp_predictor",
            "active_parameter_count": acsr_predictor_parameter_count,
            "stored_parameter_count": acsr_predictor_parameter_count,
            "basis": "causal_inputs_to_predicted_future_chunks",
            "status": "available",
        },
        {
            "component": "acsr_gru_predictor",
            "active_parameter_count": _parameter_count(gru_predictor),
            "stored_parameter_count": _parameter_count(gru_predictor),
            "basis": "causal_inputs_to_predicted_future_chunks",
            "status": "available",
        },
        {
            "component": "token_position_predictor",
            "active_parameter_count": _parameter_count(token_position_predictor),
            "stored_parameter_count": _parameter_count(token_position_predictor),
            "basis": "position_sin_cos_to_predicted_future_chunks",
            "status": "available",
        },
        {
            "component": "contextual_router_mlp",
            "active_parameter_count": contextual_router_parameter_count,
            "stored_parameter_count": contextual_router_parameter_count,
            "basis": "contextual_feature_tensor_to_column_scores",
            "status": "available",
        },
        {
            "component": "acsr_mlp_predictor_plus_contextual_router",
            "active_parameter_count": target_count,
            "stored_parameter_count": target_count,
            "basis": "predictor plus shared contextual score MLP",
            "status": "available",
        },
        {
            "component": "parameter_matched_causal_mlp_control",
            "active_parameter_count": control_count,
            "stored_parameter_count": control_count,
            "basis": (
                "causal_inputs_directly_to_column_scores;"
                f"target_parameter_count={target_count};"
                f"holdout_r2={parameter_matched_row.get('holdout_r2')};"
                f"holdout_topk_support_match={parameter_matched_row.get('holdout_topk_support_match')}"
            ),
            "status": "available",
            "parameter_count_ratio_to_acsr_path": (
                control_count / target_count if target_count else ""
            ),
        },
    ]


def _parameter_count(module: Any) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _shuffle_tokens(torch: Any, tensor: Any) -> Any:
    flat = tensor.reshape(-1, tensor.shape[-1])
    order = torch.randperm(flat.shape[0], device=tensor.device)
    return flat[order].reshape_as(tensor)


def _used_columns(support: Any) -> int:
    return len({int(value) for value in support.reshape(-1).detach().cpu().tolist()})


def _unique_support_sets(support: Any) -> int:
    return len(
        {
            tuple(int(value) for value in row)
            for row in support.reshape(-1, support.shape[-1]).detach().cpu().tolist()
        }
    )


def _support_entropy(torch: Any, support: Any, num_columns: int) -> float:
    counts = torch.bincount(support.reshape(-1), minlength=num_columns).to(dtype=torch.float32)
    probs = counts / counts.sum().clamp_min(1.0)
    entropy = -(probs[probs > 0.0] * torch.log(probs[probs > 0.0])).sum()
    return float(entropy.item())


def _mean_topk_margin(scores: Any, top_k: int) -> float:
    if scores.shape[-1] <= top_k:
        return 0.0
    sorted_scores = scores.sort(dim=-1, descending=True).values
    margin = sorted_scores[..., top_k - 1] - sorted_scores[..., top_k]
    return float(margin.mean().item())


def _gate_failures(
    router_by_name: dict[str, dict[str, Any]],
    predictor_rows: list[dict[str, Any]],
    perturbation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures = []
    if any(
        not row["passed"]
        for row in perturbation_rows
        if row.get("control_type") != "leaky_future_positive"
    ):
        failures.append(
            {
                "gate": "future_perturbation_invariance",
                "reason": "future perturbation changed prefix predictions, scores, or support",
            }
        )
    if not any(
        row.get("control_type") == "leaky_future_positive" and row["passed"]
        for row in perturbation_rows
    ):
        failures.append(
            {
                "gate": "leaky_future_positive_control",
                "reason": "future perturbation power check did not detect deliberately leaky features",
            }
        )
    acsr = router_by_name.get("acsr_mlp_predicted_future")
    if not acsr:
        failures.append({"gate": "acsr_variant", "reason": "missing ACSR router row"})
    for row in predictor_rows:
        if row["predictor"] == "mlp_causal" and row["holdout_r2"] <= -1.0:
            failures.append(
                {
                    "gate": "predictor_quality",
                    "reason": "MLP predictor is worse than a broad mean baseline",
                }
            )
    return failures


def _metric_lt(
    router_by_name: dict[str, dict[str, Any]],
    left: str,
    right: str,
    field: str,
) -> bool:
    return float(router_by_name[left][field]) < float(router_by_name[right][field])


def _metric_le(
    router_by_name: dict[str, dict[str, Any]],
    left: str,
    right: str,
    field: str,
) -> bool:
    left_value = router_by_name[left].get(field)
    right_value = router_by_name[right].get(field)
    if left_value == "" or right_value == "":
        return False
    return float(left_value) <= float(right_value)


def _optional_delta(
    left: dict[str, Any],
    right: dict[str, Any],
    field: str,
) -> float | None:
    if field not in left or field not in right:
        return None
    return float(left[field]) - float(right[field])


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML object in {path}")
    return value


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _failure_summary(
    *,
    out_dir: Path,
    start: float,
    config_path: Path,
    reason: str,
) -> dict[str, Any]:
    return {
        "status": "fail",
        "decision": "anticipatory_contextual_support_routing_runtime_unavailable",
        "claim_status": "local_acsr_smoke_not_executed",
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "failures": [{"gate": "runtime", "reason": reason}],
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    *,
    predictor_rows: list[dict[str, Any]],
    router_rows: list[dict[str, Any]],
    same_student_rows: list[dict[str, Any]],
    support_agreement_rows: list[dict[str, Any]],
    perturbation_rows: list[dict[str, Any]],
    sequence_rows: list[dict[str, Any]],
    per_sequence_rows: list[dict[str, Any]],
    margin_rows: list[dict[str, Any]],
    parameter_rows: list[dict[str, Any]],
    retention_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "predictor_metrics.csv", predictor_rows)
    _write_csv(out_dir / "router_metrics.csv", router_rows)
    _write_csv(out_dir / "same_student_metrics.csv", same_student_rows)
    _write_csv(out_dir / "support_agreement.csv", support_agreement_rows)
    _write_csv(out_dir / "feature_perturbation.csv", perturbation_rows)
    _write_csv(out_dir / "sequence_heldout_metrics.csv", sequence_rows)
    _write_csv(out_dir / "per_sequence_paired_deltas.csv", per_sequence_rows)
    _write_csv(out_dir / "margin_fragility.csv", margin_rows)
    _write_csv(out_dir / "parameter_counts.csv", parameter_rows)
    _write_csv(out_dir / "retention_churn_metrics.csv", retention_rows)
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    else:
        fieldnames = ["status"]
        rows = [{"status": "missing"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Anticipatory Contextual Support Routing Smoke",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Train steps: `{summary.get('train_steps', '')}`",
        f"- Predictor steps: `{summary.get('predictor_steps', '')}`",
        "",
        "This local CPU pilot trains the promoted full-context contextual router, "
        "trains a causal MLP to predict the future contextual feature chunks, "
        "and routes from predicted chunks at evaluation. The artifact remains "
        "a smoke pilot; promotion requires the null, same-student, regret, and "
        "retention/churn gates to stay discriminative in broader evidence.",
    ]
    if summary.get("failures"):
        lines.extend(["", "## Failures"])
        for failure in summary["failures"]:
            lines.append(f"- `{failure.get('gate')}`: {failure.get('reason')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--predictor-steps", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_anticipatory_contextual_support_routing(
        config_path=args.config,
        out_dir=args.out,
        max_steps=args.max_steps,
        predictor_steps=args.predictor_steps,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
