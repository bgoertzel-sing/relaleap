"""Executable local ACSR cross-value transfer-objective probe."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from relaleap.experiments.acsr_margin_aware_transfer_objective_design import (
    DEFAULT_OUT_DIR as DEFAULT_DESIGN_DIR,
)
from relaleap.experiments.anticipatory_contextual_support_routing import (
    _CausalScoreMLP,
    _causal_predictor_inputs,
    _contextual_chunks,
    _feature_tensor,
    _frequency_matched_random_support,
    _matched_mlp_width,
    _oracle_loss_and_support,
    _position_predictor_inputs,
    _score_from_features,
    _shuffle_tokens,
    _support_disagreement_mask,
    _support_entropy,
    _support_eval_metrics,
    _support_jaccard,
    _topk_margin_tensor,
    _train_causal_score_control_row,
    _train_independent_residual,
    _train_predictor_row,
    _unique_support_sets,
    _used_columns,
    _FuturePredictor,
)


DEFAULT_CONFIG = Path(
    "configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml"
)
DEFAULT_DESIGN = DEFAULT_DESIGN_DIR / "summary.json"
DEFAULT_OUT_DIR = Path("results/audits/acsr_transfer_objective_probe")
REQUIRED_ARTIFACTS = (
    "summary.json",
    "metrics.csv",
    "arm_metrics.csv",
    "per_token_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_acsr_transfer_objective_probe(
    *,
    config_path: Path = DEFAULT_CONFIG,
    design_summary: Path = DEFAULT_DESIGN,
    out_dir: Path = DEFAULT_OUT_DIR,
    max_steps: int = 12,
    router_steps: int = 40,
) -> dict[str, Any]:
    """Train one bounded local cross-value transfer router and write artifacts."""

    start = time.time()
    design = _read_json_object(design_summary)
    design_gate_rows = _design_gate_rows(design, design_summary)
    if any(not row["passed"] for row in design_gate_rows):
        summary = _summary(
            status="fail",
            decision="acsr_transfer_objective_probe_failed_closed",
            claim_status="transfer_objective_probe_not_run",
            start=start,
            config_path=config_path,
            design_summary=design_summary,
            max_steps=max_steps,
            router_steps=router_steps,
            objective_weights={},
            primary_metrics={},
            gate_rows=design_gate_rows,
            metric_rows=[],
            arm_rows=[],
            per_token_rows=[],
            failures=[row for row in design_gate_rows if not row["passed"]],
            out_dir=out_dir,
        )
        _write_artifacts(out_dir, summary, [], [], [], design_gate_rows)
        return summary

    try:
        import torch
        import torch.nn.functional as F

        from relaleap.smoke import (
            ResidualColumns,
            TinyCharTransformer,
            _build_batch,
            _residual_loss,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        gate_rows = design_gate_rows + [
            _criterion(
                "torch_and_harness_imports",
                False,
                "torch and RelaLeap harness imports succeed",
                str(exc),
                "cannot run executable local probe without torch/harness imports",
            )
        ]
        summary = _summary(
            status="fail",
            decision="acsr_transfer_objective_probe_failed_closed",
            claim_status="transfer_objective_probe_not_run",
            start=start,
            config_path=config_path,
            design_summary=design_summary,
            max_steps=max_steps,
            router_steps=router_steps,
            objective_weights=_objective_weights(design),
            primary_metrics={},
            gate_rows=gate_rows,
            metric_rows=[],
            arm_rows=[],
            per_token_rows=[],
            failures=[row for row in gate_rows if not row["passed"]],
            out_dir=out_dir,
        )
        _write_artifacts(out_dir, summary, [], [], [], gate_rows)
        return summary

    config = _read_yaml(config_path)
    run_cfg = _dict(config.get("run"))
    data_cfg = _dict(config.get("data"))
    model_cfg = _dict(config.get("model"))
    base_cfg = _dict(model_cfg.get("base"))
    column_cfg = _dict(model_cfg.get("columns"))
    training_cfg = _dict(config.get("training"))

    seed = int(run_cfg.get("seed", 1))
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
    objective_weights = _objective_weights(design)

    torch.manual_seed(seed)
    inputs, targets, vocab_size = _build_batch(dataset=dataset, seq_len=seq_len, batch_size=4)
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
    for _ in range(max(1, max_steps)):
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

    partner = _train_independent_residual(
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
        train_steps=max(1, max_steps),
        residual_objective=residual_objective,
        seed=seed + 1009,
    )

    with torch.no_grad():
        hidden = base.encode(inputs)
        chunks = _contextual_chunks(torch, hidden)
        causal_inputs = _causal_predictor_inputs(torch, chunks)
        position_inputs = _position_predictor_inputs(torch, chunks)
        future_targets = torch.cat([chunks["next"], chunks["next_delta"]], dim=-1)

    predictor = _FuturePredictor(
        __import__("torch.nn").nn,
        causal_inputs.shape[-1],
        hidden_dim * 2,
        hidden_dim,
    )
    position_predictor = _FuturePredictor(
        __import__("torch.nn").nn,
        position_inputs.shape[-1],
        hidden_dim * 2,
        max(8, min(hidden_dim, 64)),
    )
    _train_predictor_row(
        torch,
        F,
        predictor,
        causal_inputs,
        future_targets,
        steps=max(4, router_steps // 2),
        label="transfer_probe_future_predictor",
    )
    _train_predictor_row(
        torch,
        F,
        position_predictor,
        position_inputs,
        future_targets,
        steps=max(4, router_steps // 2),
        label="transfer_probe_position_predictor",
    )

    nn = __import__("torch.nn").nn
    target_parameter_count = _parameter_count(predictor) + _parameter_count(
        residual.contextual_column_scores
    )
    matched_width = _matched_mlp_width(
        input_dim=causal_inputs.shape[-1],
        output_dim=num_columns,
        target_parameter_count=target_parameter_count,
    )
    direct_router = _CausalScoreMLP(
        nn,
        causal_inputs.shape[-1],
        num_columns,
        matched_width,
    )
    transfer_router = _CausalScoreMLP(
        nn,
        causal_inputs.shape[-1],
        num_columns,
        matched_width,
    )
    token_position_router = _CausalScoreMLP(
        nn,
        position_inputs.shape[-1],
        num_columns,
        matched_width,
    )

    with torch.no_grad():
        predicted = predictor(causal_inputs)
        position_predicted = position_predictor(position_inputs)
        shuffled_predicted = _shuffle_tokens(torch, predicted)
        zero_predicted = torch.zeros_like(future_targets)
        teacher_scores = _score_from_features(residual, _feature_tensor(torch, chunks, future_targets))
        token_position_scores = _score_from_features(
            residual,
            _feature_tensor(torch, chunks, position_predicted),
        )
        shuffled_scores = _score_from_features(
            residual,
            _feature_tensor(torch, chunks, shuffled_predicted),
        )
        causal_zero_scores = _score_from_features(
            residual,
            _feature_tensor(torch, chunks, zero_predicted),
        )

    _train_causal_score_control_row(
        torch,
        F,
        direct_router,
        causal_inputs,
        teacher_scores.detach(),
        steps=router_steps,
        label="direct_causal_mlp_baseline",
    )
    _train_transfer_router(
        torch,
        F,
        base,
        residual,
        partner,
        transfer_router,
        causal_inputs,
        hidden,
        targets,
        vocab_size,
        teacher_scores,
        token_position_scores,
        top_k=top_k,
        steps=router_steps,
        weights=objective_weights,
    )
    _train_transfer_router(
        torch,
        F,
        base,
        residual,
        partner,
        token_position_router,
        position_inputs,
        hidden,
        targets,
        vocab_size,
        teacher_scores,
        token_position_scores,
        top_k=top_k,
        steps=router_steps,
        weights=objective_weights,
    )

    with torch.no_grad():
        direct_scores = direct_router(causal_inputs) + residual.score_tie_breaker.to(
            device=hidden.device, dtype=hidden.dtype
        )
        transfer_scores = transfer_router(causal_inputs) + residual.score_tie_breaker.to(
            device=hidden.device, dtype=hidden.dtype
        )
        token_position_trained_scores = token_position_router(position_inputs) + residual.score_tie_breaker.to(
            device=hidden.device, dtype=hidden.dtype
        )
        score_rows = {
            "direct_causal_mlp_baseline": direct_scores,
            "transfer_objective_router": transfer_scores,
            "token_position_only_transfer_null": token_position_trained_scores,
            "teacher_full_context_diagnostic": teacher_scores,
            "token_position_untrained_null": token_position_scores,
            "shuffled_predicted_null": shuffled_scores,
            "causal_zero_null": causal_zero_scores,
        }
        arm_rows, per_token_rows, primary_metrics = _evaluate_arms(
            torch,
            F,
            base,
            own_residual=residual,
            partner_residual=partner,
            hidden=hidden,
            targets=targets,
            vocab_size=vocab_size,
            score_rows=score_rows,
            top_k=top_k,
            seed=seed + 313,
        )

    metric_rows = _metric_rows(primary_metrics)
    gate_rows = design_gate_rows + _probe_gate_rows(primary_metrics)
    failures = [row for row in gate_rows if not row["passed"]]
    status = "pass" if not failures else "fail"
    summary = _summary(
        status=status,
        decision=(
            "acsr_transfer_objective_probe_recorded"
            if status == "pass"
            else "acsr_transfer_objective_probe_failed_gate"
        ),
        claim_status=(
            "local_transfer_objective_supported_not_promoted"
            if status == "pass"
            else "local_transfer_objective_not_supported"
        ),
        start=start,
        config_path=config_path,
        design_summary=design_summary,
        max_steps=max_steps,
        router_steps=router_steps,
        objective_weights=objective_weights,
        primary_metrics=primary_metrics,
        gate_rows=gate_rows,
        metric_rows=metric_rows,
        arm_rows=arm_rows,
        per_token_rows=per_token_rows,
        failures=failures,
        out_dir=out_dir,
    )
    _write_artifacts(out_dir, summary, metric_rows, arm_rows, per_token_rows, gate_rows)
    return summary


def _train_transfer_router(
    torch: Any,
    F: Any,
    base: Any,
    own_residual: Any,
    partner_residual: Any,
    router: Any,
    inputs: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    teacher_scores: Any,
    token_position_scores: Any,
    *,
    top_k: int,
    steps: int,
    weights: dict[str, float],
) -> None:
    del top_k
    optimizer = torch.optim.AdamW(router.parameters(), lr=3e-3)
    split = max(1, int(inputs.shape[1]) // 2)
    train_x = inputs[:, :split, :]
    train_hidden = hidden[:, :split, :]
    train_targets = targets[:, :split]
    teacher_train = teacher_scores[:, :split, :]
    token_position_train = token_position_scores[:, :split, :]
    high_regret_mask, disagreement_mask, low_margin_mask = _training_masks(
        torch,
        F,
        base,
        partner_residual,
        train_hidden,
        train_targets,
        vocab_size,
        teacher_train,
        token_position_train,
    )
    token_weights = torch.ones_like(high_regret_mask, dtype=train_hidden.dtype)
    token_weights = token_weights + weights.get("high_regret_cross_value_focus", 0.0) * high_regret_mask
    token_weights = token_weights + weights.get("support_disagreement_focus", 0.0) * disagreement_mask
    if weights.get("low_margin_suppression", 0.0) <= 0.0:
        token_weights = torch.where(low_margin_mask > 0.0, token_weights * 0.25, token_weights)
    for _ in range(max(1, steps)):
        optimizer.zero_grad(set_to_none=True)
        scores = router(train_x)
        partner_loss = _soft_value_ce(
            torch,
            F,
            base,
            partner_residual,
            train_hidden,
            train_targets,
            vocab_size,
            scores,
            token_weights=token_weights,
        )
        own_loss = _soft_value_ce(
            torch,
            F,
            base,
            own_residual,
            train_hidden,
            train_targets,
            vocab_size,
            scores,
            token_weights=torch.ones_like(token_weights),
        )
        teacher_loss = F.mse_loss(scores, teacher_train)
        loss = (
            weights.get("cross_value_partner_support_ce", 1.0) * partner_loss
            + 0.25 * own_loss
            + 0.05 * teacher_loss
        )
        loss.backward()
        optimizer.step()


def _soft_value_ce(
    torch: Any,
    F: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    scores: Any,
    *,
    token_weights: Any,
) -> Any:
    column_weights = torch.softmax(scores, dim=-1)
    atom_weights = torch.softmax(residual.atom_logits, dim=-1)
    column_values = torch.einsum("ca,cah->ch", atom_weights, residual.atom_values)
    residual_update = torch.einsum("bsc,ch->bsh", column_weights, column_values)
    logits = base.decode(hidden + residual_update)
    per_token = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab_size),
        targets[:, :-1].reshape(-1),
        reduction="none",
    )
    weights = token_weights[:, :-1].reshape(-1).clamp_min(0.0)
    return (per_token * weights).sum() / weights.sum().clamp_min(1e-6)


def _training_masks(
    torch: Any,
    F: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    teacher_scores: Any,
    token_position_scores: Any,
) -> tuple[Any, Any, Any]:
    oracle_loss, oracle_support = _oracle_loss_and_support(
        torch,
        F,
        base,
        residual,
        hidden,
        targets,
        vocab_size,
        teacher_scores,
    )
    del oracle_loss
    token_support = token_position_scores.topk(2, dim=-1).indices
    token_metrics = _support_eval_metrics(
        torch,
        F,
        base,
        residual,
        hidden,
        targets,
        vocab_size,
        support=token_support,
        target_scores=token_position_scores,
    )
    oracle_metrics = _support_eval_metrics(
        torch,
        F,
        base,
        residual,
        hidden,
        targets,
        vocab_size,
        support=oracle_support,
        target_scores=teacher_scores,
    )
    regret = token_metrics["per_token_losses"] - oracle_metrics["per_token_losses"]
    threshold = torch.quantile(regret, 0.75)
    high_regret = (regret >= threshold).view(hidden.shape[0], hidden.shape[1] - 1)
    teacher_support = teacher_scores.topk(2, dim=-1).indices
    disagreement = _support_disagreement_mask(torch, teacher_support, token_support).view_as(
        high_regret
    )
    margin = _topk_margin_tensor(teacher_scores, 2)[:, :-1]
    low_margin = margin < 0.01
    pad = torch.zeros(
        hidden.shape[0],
        1,
        dtype=hidden.dtype,
        device=hidden.device,
    )
    return (
        torch.cat([high_regret.to(dtype=hidden.dtype), pad], dim=1),
        torch.cat([disagreement.to(dtype=hidden.dtype), pad], dim=1),
        torch.cat([low_margin.to(dtype=hidden.dtype), pad], dim=1),
    )


def _evaluate_arms(
    torch: Any,
    F: Any,
    base: Any,
    *,
    own_residual: Any,
    partner_residual: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    score_rows: dict[str, Any],
    top_k: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    arm_rows: list[dict[str, Any]] = []
    per_token_rows: list[dict[str, Any]] = []
    primary: dict[str, Any] = {}
    value_paths = {
        "own_values": own_residual,
        "partner_values": partner_residual,
    }
    oracle_by_path = {}
    for value_path, residual in value_paths.items():
        oracle_loss, oracle_support = _oracle_loss_and_support(
            torch,
            F,
            base,
            residual,
            hidden,
            targets,
            vocab_size,
            score_rows["teacher_full_context_diagnostic"],
        )
        oracle_by_path[value_path] = (oracle_loss, oracle_support)
        random_support = _frequency_matched_random_support(
            torch,
            score_rows["direct_causal_mlp_baseline"].topk(top_k, dim=-1).indices,
            num_columns=residual.num_columns,
            seed=seed,
        )
        for arm, scores in score_rows.items():
            support = scores.topk(top_k, dim=-1).indices
            metrics = _support_eval_metrics(
                torch,
                F,
                base,
                residual,
                hidden,
                targets,
                vocab_size,
                support=support,
                target_scores=scores,
            )
            arm_rows.append(
                _arm_row(
                    torch,
                    value_path,
                    arm,
                    metrics,
                    support,
                    scores,
                    residual.num_columns,
                    oracle_loss,
                    top_k,
                )
            )
            per_token_rows.extend(
                _per_token_rows(
                    value_path,
                    arm,
                    metrics["per_token_losses"],
                    metrics["residual_update_l2_per_token"],
                    support,
                )
            )
        for arm, support, scores in [
            ("random_frequency_support_null", random_support, score_rows["direct_causal_mlp_baseline"]),
            ("oracle_best_support_diagnostic", oracle_support, score_rows["teacher_full_context_diagnostic"]),
        ]:
            metrics = _support_eval_metrics(
                torch,
                F,
                base,
                residual,
                hidden,
                targets,
                vocab_size,
                support=support,
                target_scores=scores,
            )
            arm_rows.append(
                _arm_row(
                    torch,
                    value_path,
                    arm,
                    metrics,
                    support,
                    scores,
                    residual.num_columns,
                    oracle_loss,
                    top_k,
                )
            )
            per_token_rows.extend(
                _per_token_rows(
                    value_path,
                    arm,
                    metrics["per_token_losses"],
                    metrics["residual_update_l2_per_token"],
                    support,
                )
            )
    by_key = {(row["value_path"], row["arm"]): row for row in arm_rows}
    primary.update(
        {
            "partner_transfer_minus_direct_ce": _delta(
                by_key,
                ("partner_values", "transfer_objective_router"),
                ("partner_values", "direct_causal_mlp_baseline"),
                "ce_loss",
            ),
            "own_transfer_minus_direct_ce": _delta(
                by_key,
                ("own_values", "transfer_objective_router"),
                ("own_values", "direct_causal_mlp_baseline"),
                "ce_loss",
            ),
            "partner_transfer_minus_token_position_ce": _delta(
                by_key,
                ("partner_values", "transfer_objective_router"),
                ("partner_values", "token_position_only_transfer_null"),
                "ce_loss",
            ),
            "partner_transfer_minus_random_ce": _delta(
                by_key,
                ("partner_values", "transfer_objective_router"),
                ("partner_values", "random_frequency_support_null"),
                "ce_loss",
            ),
            "partner_transfer_oracle_regret": by_key[
                ("partner_values", "transfer_objective_router")
            ]["oracle_regret"],
            "partner_direct_oracle_regret": by_key[
                ("partner_values", "direct_causal_mlp_baseline")
            ]["oracle_regret"],
            "partner_transfer_residual_norm_normalized_delta_vs_direct": _norm_delta(
                by_key,
                ("partner_values", "transfer_objective_router"),
                ("partner_values", "direct_causal_mlp_baseline"),
            ),
            "transfer_support_jaccard_with_direct": _support_jaccard(
                score_rows["transfer_objective_router"].topk(top_k, dim=-1).indices,
                score_rows["direct_causal_mlp_baseline"].topk(top_k, dim=-1).indices,
            ),
            "oracle_paths_evaluated": sorted(oracle_by_path),
        }
    )
    return arm_rows, per_token_rows, primary


def _arm_row(
    torch: Any,
    value_path: str,
    arm: str,
    metrics: dict[str, Any],
    support: Any,
    scores: Any,
    num_columns: int,
    oracle_loss: float,
    top_k: int,
) -> dict[str, Any]:
    return {
        "value_path": value_path,
        "arm": arm,
        "top_k": int(top_k),
        "ce_loss": metrics["ce_loss"],
        "oracle_loss": oracle_loss,
        "oracle_regret": metrics["ce_loss"] - oracle_loss,
        "residual_update_l2_mean": metrics["residual_update_l2_mean"],
        "used_columns": _used_columns(support),
        "unique_support_sets": _unique_support_sets(support),
        "support_entropy": _support_entropy(torch, support, num_columns),
        "mean_topk_margin": _mean_topk_margin(scores, top_k),
    }


def _per_token_rows(
    value_path: str,
    arm: str,
    losses: Any,
    residual_l2: Any,
    support: Any,
) -> list[dict[str, Any]]:
    flat_support = support[:, :-1, :].reshape(losses.numel(), support.shape[-1])
    rows = []
    for index in range(losses.numel()):
        rows.append(
            {
                "value_path": value_path,
                "arm": arm,
                "token_index": index,
                "ce_loss": float(losses[index].item()),
                "residual_update_l2": float(residual_l2[index].item()),
                "support": ";".join(str(int(item)) for item in flat_support[index].tolist()),
            }
        )
    return rows


def _probe_gate_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _criterion(
            "partner_transfer_beats_token_position_null",
            _lt(metrics.get("partner_transfer_minus_token_position_ce"), 0.0),
            "transfer objective improves partner-through-values CE over token-position trained null",
            metrics.get("partner_transfer_minus_token_position_ce"),
            "transfer objective did not beat the token-position null",
        ),
        _criterion(
            "partner_transfer_beats_random_null",
            _lt(metrics.get("partner_transfer_minus_random_ce"), 0.0),
            "transfer objective improves partner-through-values CE over frequency-random support null",
            metrics.get("partner_transfer_minus_random_ce"),
            "transfer objective did not beat the random-frequency null",
        ),
        _criterion(
            "own_ce_guardrail",
            _le(metrics.get("own_transfer_minus_direct_ce"), 0.02),
            "own-value CE worsens by no more than 0.02 versus direct causal MLP",
            metrics.get("own_transfer_minus_direct_ce"),
            "transfer objective damages own-value CE beyond guardrail",
        ),
        _criterion(
            "residual_norm_normalized_gain_available",
            metrics.get("partner_transfer_residual_norm_normalized_delta_vs_direct") not in ("", None),
            "residual-norm-normalized gain versus direct baseline is recorded",
            metrics.get("partner_transfer_residual_norm_normalized_delta_vs_direct"),
            "residual norm normalization is missing",
        ),
    ]


def _design_gate_rows(design: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    return [
        _criterion(
            "objective_design_present",
            bool(design),
            "margin-aware transfer-objective design summary exists",
            str(path),
            "missing design summary",
        ),
        _criterion(
            "objective_design_passed",
            design.get("status") == "pass",
            "design status is pass",
            design.get("status", "missing"),
            "design summary is absent or failed",
        ),
        _criterion(
            "objective_terms_present",
            bool(design.get("objective_terms")),
            "recorded objective weights are available",
            len(design.get("objective_terms", [])) if isinstance(design.get("objective_terms"), list) else 0,
            "objective terms missing",
        ),
    ]


def _summary(
    *,
    status: str,
    decision: str,
    claim_status: str,
    start: float,
    config_path: Path,
    design_summary: Path,
    max_steps: int,
    router_steps: int,
    objective_weights: dict[str, float],
    primary_metrics: dict[str, Any],
    gate_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    per_token_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    return {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "config_path": str(config_path),
        "design_summary": str(design_summary),
        "max_steps": max_steps,
        "router_steps": router_steps,
        "objective_weights": objective_weights,
        "primary_metrics": primary_metrics,
        "gate_criteria": gate_rows,
        "metric_count": len(metric_rows),
        "arm_count": len(arm_rows),
        "per_token_count": len(per_token_rows),
        "failures": failures,
        "selected_next_step": (
            "run a second seed local transfer-objective probe before GPU validation"
            if status == "pass"
            else "stop ACSR transfer-objective promotion and inspect failed guardrails"
        ),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }


def _objective_weights(design: dict[str, Any]) -> dict[str, float]:
    rows = design.get("objective_terms")
    if not isinstance(rows, list):
        return {}
    weights: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        term = str(row.get("term", ""))
        try:
            weights[term] = float(row.get("weight", 0.0))
        except (TypeError, ValueError):
            weights[term] = 0.0
    return weights


def _metric_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"metric": key, "value": value} for key, value in sorted(metrics.items())]


def _delta(
    rows: dict[tuple[str, str], dict[str, Any]],
    left: tuple[str, str],
    right: tuple[str, str],
    metric: str,
) -> float | str:
    try:
        return float(rows[left][metric]) - float(rows[right][metric])
    except (KeyError, TypeError, ValueError):
        return ""


def _norm_delta(
    rows: dict[tuple[str, str], dict[str, Any]],
    left: tuple[str, str],
    right: tuple[str, str],
) -> float | str:
    numerator = _delta(rows, left, right, "ce_loss")
    denominator = _delta(rows, left, right, "residual_update_l2_mean")
    if not isinstance(numerator, float) or not isinstance(denominator, float):
        return ""
    if abs(denominator) <= 1e-12:
        return ""
    return numerator / denominator


def _mean_topk_margin(scores: Any, top_k: int) -> float:
    if top_k < 2:
        return 0.0
    top_values = scores.topk(top_k, dim=-1).values
    return float((top_values[..., -2] - top_values[..., -1]).abs().mean().item())


def _lt(value: Any, threshold: float) -> bool:
    try:
        return float(value) < threshold
    except (TypeError, ValueError):
        return False


def _le(value: Any, threshold: float) -> bool:
    try:
        return float(value) <= threshold
    except (TypeError, ValueError):
        return False


def _criterion(
    criterion: str,
    passed: bool,
    threshold: str,
    actual: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": actual,
        "failure_reason": "" if passed else failure_reason,
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _parameter_count(module: Any) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    per_token_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "metrics.csv", metric_rows)
    _write_csv(out_dir / "arm_metrics.csv", arm_rows)
    _write_csv(out_dir / "per_token_metrics.csv", per_token_rows)
    _write_csv(out_dir / "gate_criteria.csv", gate_rows)
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [{"status": "missing"}]
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ACSR Transfer Objective Probe",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Max residual steps: `{summary.get('max_steps')}`",
        f"- Router steps: `{summary.get('router_steps')}`",
        "",
        "This bounded local probe trains a causal support router with the recorded "
        "cross-value transfer objective and compares it with the direct causal MLP, "
        "token-position, shuffled, random-frequency, own-support, partner-value, and "
        "oracle diagnostic arms. It is not a GPU validation or default promotion.",
    ]
    if summary.get("primary_metrics"):
        lines.extend(["", "## Primary Metrics"])
        for key, value in sorted(summary["primary_metrics"].items()):
            lines.append(f"- `{key}`: `{value}`")
    if summary.get("failures"):
        lines.extend(["", "## Failures"])
        for row in summary["failures"]:
            lines.append(f"- `{row['criterion']}`: {row['failure_reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--design-summary", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--router-steps", type=int, default=40)
    args = parser.parse_args()
    summary = run_acsr_transfer_objective_probe(
        config_path=args.config,
        design_summary=args.design_summary,
        out_dir=args.out,
        max_steps=args.max_steps,
        router_steps=args.router_steps,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
