"""Run the first local controls for dense-teacher pair composition."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from relaleap.experiments.dense_teacher_failure_localization import REQUIRED_TENSORS
from relaleap.experiments.dense_teacher_failure_localization import _cosine
from relaleap.experiments.dense_teacher_failure_localization import _load_evaluator_tensors
from relaleap.experiments.dense_teacher_failure_localization import _oracle_support_trained_values
from relaleap.experiments.dense_teacher_failure_localization import _r2
from relaleap.experiments.dense_teacher_failure_localization import _ridge_fit
from relaleap.experiments.dense_teacher_failure_localization import _shuffled_support_pair_design
from relaleap.experiments.dense_teacher_failure_localization import _split_pregate_metric_row
from relaleap.experiments.dense_teacher_failure_localization import _support_pair_design


DEFAULT_DISTILLATION_DIR = Path("results/audits/token_larger_dense_teacher_residual_distillation_comparison")
DEFAULT_DESIGN = Path("results/reports/dense_teacher_pair_composer_control_extension_design/summary.json")
DEFAULT_TRUTH_AUDIT = Path("results/reports/dense_teacher_pair_composer_truth_audit/summary.json")
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_pair_composer_control_extension_probe")

NEXT_ACTION = "extend_pair_composer_probe_with_norm_churn_commutator_dense_controls"
REPAIR_ACTION = "repair_pair_composer_control_extension_probe_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "router_rows.csv",
    "sentinel_rows.csv",
    "gate_criteria.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_dense_teacher_pair_composer_control_extension_probe(
    *,
    distillation_dir: Path = DEFAULT_DISTILLATION_DIR,
    design_path: Path = DEFAULT_DESIGN,
    truth_audit_path: Path = DEFAULT_TRUTH_AUDIT,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write local learned-router and leakage-sentinel rows for pair composition."""

    start = time.time()
    design = _read_json(design_path)
    truth_audit = _read_json(truth_audit_path)
    source_rows = [
        _source_json("control_extension_design", design_path, design),
        _source_json("pair_composer_truth_audit", truth_audit_path, truth_audit),
    ]
    source_rows.extend(_tensor_source_rows(distillation_dir))
    source_failures = [row for row in source_rows if row["required"] and not row["present"]]

    if source_failures:
        summary = _summary(
            status="fail",
            decision="dense_teacher_pair_composer_control_extension_probe_failed_closed",
            claim_status="pair_composer_control_probe_sources_missing",
            selected_next_action=REPAIR_ACTION,
            start=start,
            out_dir=out_dir,
            source_rows=source_rows,
            router_rows=[],
            sentinel_rows=[],
            gate_criteria=[
                _criterion("required_sources_present", False, "design, truth audit, and exported tensors must exist")
            ],
            failures=source_failures,
            rationale="The probe cannot run until the design, truth audit, and exported dense-teacher tensors are present.",
        )
        _write_artifacts(out_dir, summary)
        return summary

    try:
        router_rows, sentinel_rows = _run_probe_rows(distillation_dir)
        runtime_failure = ""
    except Exception as exc:  # pragma: no cover - defensive fail-closed path
        router_rows = []
        sentinel_rows = []
        runtime_failure = f"{type(exc).__name__}: {exc}"

    criteria = _gate_criteria(design, truth_audit, router_rows, sentinel_rows, runtime_failure)
    failures = [row for row in criteria if not row["passed"] and row["criterion"] in {"probe_runtime_completed"}]
    status = "fail" if failures else "pass"
    learned_holdout = _row(router_rows, "learned_causal_pair_router", "holdout")
    oracle_holdout = _row(router_rows, "oracle_pair_composer_reference", "holdout")
    majority_holdout = _row(sentinel_rows, "majority_pair_router_null", "holdout")
    delayed_holdout = _row(sentinel_rows, "delayed_pair_target_null", "holdout")
    misaligned_holdout = _row(sentinel_rows, "misaligned_support_pair_null", "holdout")
    token_position_holdout = _row(sentinel_rows, "token_position_pair_router_null", "holdout")

    decision = (
        "dense_teacher_pair_composer_control_extension_probe_gpu_blocked"
        if status == "pass"
        else "dense_teacher_pair_composer_control_extension_probe_failed_closed"
    )
    claim_status = (
        "learned_router_and_leakage_sentinels_recorded_but_controls_incomplete"
        if status == "pass"
        else "pair_composer_control_probe_runtime_failed"
    )
    rationale = (
        "The probe records a first deployable pair-router row and delayed/misaligned/token-position sentinel rows. "
        "It remains local evidence only: norm/churn/retention, exact finite-update commutator, and matched dense/MLP "
        "interference controls are still absent."
        if status == "pass"
        else "The probe did not complete, so no scientific interpretation is made."
    )
    candidate_actions = _candidate_actions(status)
    summary = _summary(
        status=status,
        decision=decision,
        claim_status=claim_status,
        selected_next_action=NEXT_ACTION if status == "pass" else REPAIR_ACTION,
        start=start,
        out_dir=out_dir,
        source_rows=source_rows,
        router_rows=router_rows,
        sentinel_rows=sentinel_rows,
        gate_criteria=criteria,
        failures=failures,
        rationale=rationale,
    )
    summary.update(
        {
            "candidate_actions": candidate_actions,
            "distillation_dir": str(distillation_dir),
            "design_path": str(design_path),
            "truth_audit_path": str(truth_audit_path),
            "runtime_failure": runtime_failure,
            "oracle_holdout_true_decoder_ce_loss": _float(oracle_holdout.get("true_decoder_ce_loss")),
            "learned_router_holdout_true_decoder_ce_loss": _float(learned_holdout.get("true_decoder_ce_loss")),
            "learned_router_holdout_support_pair_accuracy": _float(learned_holdout.get("support_pair_accuracy")),
            "learned_router_holdout_support_pair_jaccard": _float(learned_holdout.get("support_pair_jaccard")),
            "holdout_support_pair_entropy": _float(oracle_holdout.get("support_pair_entropy")),
            "holdout_support_pair_normalized_entropy": _float(oracle_holdout.get("support_pair_normalized_entropy")),
            "majority_pair_holdout_true_decoder_ce_loss": _float(majority_holdout.get("true_decoder_ce_loss")),
            "majority_pair_holdout_support_pair_accuracy": _float(majority_holdout.get("support_pair_accuracy")),
            "delayed_null_holdout_true_decoder_ce_loss": _float(delayed_holdout.get("true_decoder_ce_loss")),
            "misaligned_null_holdout_true_decoder_ce_loss": _float(misaligned_holdout.get("true_decoder_ce_loss")),
            "token_position_null_holdout_true_decoder_ce_loss": _float(token_position_holdout.get("true_decoder_ce_loss")),
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "advance_to_gpu_validation": False,
            "backend_policy": "local CPU probe only; RunPod and Colab remain blocked",
        }
    )
    _write_artifacts(out_dir, summary)
    return summary


def _run_probe_rows(distillation_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch
    import torch.nn.functional as F

    tensors = _load_evaluator_tensors(torch, distillation_dir)
    base_hidden = tensors["base_hidden"]
    base_logits = tensors["base_logits"]
    targets = tensors["targets"]
    teacher_logits = tensors["teacher_logits"]
    teacher_hidden_residual = tensors["teacher_hidden_residual"]
    teacher_logit_residual = tensors["teacher_logit_residual"]
    learned_scores = tensors["learned_support_scores"]
    per_column_hidden = tensors["per_column_hidden_contributions"]
    per_column_logits = tensors["per_column_logit_contributions"]
    sparse_state = tensors["sparse_column_value_state"]
    top_k = int(sparse_state.get("top_k", learned_scores.shape[-1]))
    num_columns = int(sparse_state.get("num_columns", per_column_hidden.shape[2]))

    oracle_support, _, _ = _oracle_support_trained_values(
        torch,
        per_column_hidden,
        per_column_logits,
        teacher_logit_residual,
        top_k=top_k,
    )
    valid_indices, train_indices, holdout_indices = _split_indices(torch, targets, split_seed=1729)
    flat_oracle = oracle_support.reshape(-1, top_k)
    pair_tuples = sorted({tuple(sorted(int(item) for item in row.tolist())) for row in flat_oracle})
    pair_lookup = {pair: index for index, pair in enumerate(pair_tuples)}
    train_pair_ids = _pair_ids(torch, flat_oracle[train_indices], pair_lookup)

    learned_support = _learned_pair_router_support(
        torch,
        inputs=tensors["inputs"],
        base_hidden=base_hidden,
        learned_scores=learned_scores,
        flat_oracle=flat_oracle,
        pair_tuples=pair_tuples,
        pair_lookup=pair_lookup,
        train_indices=train_indices,
    )
    delayed_support = _support_from_pair_ids(
        torch,
        _delayed_pair_ids(torch, train_pair_ids, train_indices, flat_oracle, pair_lookup),
        pair_tuples,
        top_k=top_k,
        token_count=flat_oracle.shape[0],
        fallback=flat_oracle,
    )
    misaligned_support = flat_oracle.roll(shifts=7, dims=0)
    majority_support = _majority_pair_support(
        torch,
        flat_oracle=flat_oracle,
        train_indices=train_indices,
        pair_tuples=pair_tuples,
        pair_lookup=pair_lookup,
        top_k=top_k,
    )
    token_position_support = _token_position_null_support(
        torch,
        inputs=tensors["inputs"],
        flat_oracle=flat_oracle,
        train_indices=train_indices,
        pair_tuples=pair_tuples,
        pair_lookup=pair_lookup,
        top_k=top_k,
    )

    oracle_design, feature_count = _support_pair_design(torch, flat_oracle, num_columns=num_columns, pair_tuples=pair_tuples)
    composer_values = _ridge_fit(
        torch,
        oracle_design[train_indices],
        teacher_hidden_residual.reshape(-1, teacher_hidden_residual.shape[-1])[train_indices],
        ridge=1e-4,
    )
    shuffled_design, shuffled_feature_count = _shuffled_support_pair_design(
        torch,
        flat_oracle,
        num_columns=num_columns,
        pair_tuples=pair_tuples,
    )
    shuffled_values = _ridge_fit(
        torch,
        shuffled_design[train_indices],
        teacher_hidden_residual.reshape(-1, teacher_hidden_residual.shape[-1])[train_indices],
        ridge=1e-4,
    )
    decoder_weight, decoder_bias = _decoder(torch, base_hidden, tensors["frozen_decoder_state"])

    router_specs = [
        ("oracle_pair_composer_reference", flat_oracle, oracle_design, composer_values, feature_count, "oracle support reference"),
        (
            "learned_causal_pair_router",
            learned_support,
            _support_pair_design(torch, learned_support, num_columns=num_columns, pair_tuples=pair_tuples)[0],
            composer_values,
            feature_count,
            "ridge classifier over prefix-safe token, position, hidden, and router-score features",
        ),
    ]
    sentinel_specs = [
        (
            "majority_pair_router_null",
            majority_support,
            _support_pair_design(torch, majority_support, num_columns=num_columns, pair_tuples=pair_tuples)[0],
            composer_values,
            feature_count,
            "global majority support pair from train split; class-balance null",
        ),
        (
            "delayed_pair_target_null",
            delayed_support,
            _support_pair_design(torch, delayed_support, num_columns=num_columns, pair_tuples=pair_tuples)[0],
            composer_values,
            feature_count,
            "support-pair labels delayed by one valid training/eval token; should fail",
        ),
        (
            "misaligned_support_pair_null",
            misaligned_support,
            _support_pair_design(torch, misaligned_support, num_columns=num_columns, pair_tuples=pair_tuples)[0],
            composer_values,
            feature_count,
            "oracle support pairs rolled across tokens; should fail",
        ),
        (
            "token_position_pair_router_null",
            token_position_support,
            _support_pair_design(torch, token_position_support, num_columns=num_columns, pair_tuples=pair_tuples)[0],
            composer_values,
            feature_count,
            "majority train pair by token and position bucket; shortcut null",
        ),
        (
            "feature_count_shuffled_pair_null",
            flat_oracle,
            shuffled_design,
            shuffled_values,
            shuffled_feature_count,
            "feature-count matched shuffled-pair composer retained as regression null",
        ),
    ]
    router_rows = _metric_rows(
        torch,
        F,
        specs=router_specs,
        oracle_support=flat_oracle,
        base_hidden=base_hidden,
        base_logits=base_logits,
        targets=targets,
        teacher_logits=teacher_logits,
        teacher_hidden_residual=teacher_hidden_residual,
        decoder_weight=decoder_weight,
        decoder_bias=decoder_bias,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
    )
    sentinel_rows = _metric_rows(
        torch,
        F,
        specs=sentinel_specs,
        oracle_support=flat_oracle,
        base_hidden=base_hidden,
        base_logits=base_logits,
        targets=targets,
        teacher_logits=teacher_logits,
        teacher_hidden_residual=teacher_hidden_residual,
        decoder_weight=decoder_weight,
        decoder_bias=decoder_bias,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
    )
    return router_rows, sentinel_rows


def _split_indices(torch: Any, targets: Any, *, split_seed: int) -> tuple[Any, Any, Any]:
    batch, seq_len = targets.shape
    valid_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=targets.device)
    valid_mask[:, -1] = False
    valid_indices = torch.nonzero(valid_mask.reshape(-1), as_tuple=False).reshape(-1)
    generator = torch.Generator(device=valid_indices.device)
    generator.manual_seed(split_seed)
    ordered = valid_indices[torch.randperm(int(valid_indices.numel()), generator=generator, device=valid_indices.device)]
    train_count = max(1, int(round(0.6 * int(ordered.numel()))))
    train_indices = ordered[:train_count]
    holdout_indices = ordered[train_count:]
    if int(holdout_indices.numel()) == 0:
        holdout_indices = ordered[-1:]
        train_indices = ordered[:-1]
    return valid_indices, train_indices, holdout_indices


def _learned_pair_router_support(
    torch: Any,
    *,
    inputs: Any,
    base_hidden: Any,
    learned_scores: Any,
    flat_oracle: Any,
    pair_tuples: list[tuple[int, ...]],
    pair_lookup: dict[tuple[int, ...], int],
    train_indices: Any,
) -> Any:
    x = _prefix_features(torch, inputs, base_hidden, learned_scores)
    y = _pair_ids(torch, flat_oracle, pair_lookup)
    train_x = x[train_indices]
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    x = (x - mean) / std
    train_x = x[train_indices]
    train_y = y[train_indices]
    one_hot = torch.zeros(train_x.shape[0], len(pair_tuples), dtype=train_x.dtype, device=train_x.device)
    one_hot[torch.arange(train_x.shape[0], device=train_x.device), train_y] = 1.0
    design = torch.cat([torch.ones(train_x.shape[0], 1, dtype=train_x.dtype, device=train_x.device), train_x], dim=1)
    weights = _ridge_fit(torch, design, one_hot, ridge=1e-3)
    full_design = torch.cat([torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device), x], dim=1)
    pred_ids = full_design.matmul(weights).argmax(dim=1)
    return _support_from_pair_ids(
        torch,
        pred_ids,
        pair_tuples,
        top_k=flat_oracle.shape[1],
        token_count=flat_oracle.shape[0],
        fallback=flat_oracle,
    )


def _prefix_features(torch: Any, inputs: Any, base_hidden: Any, learned_scores: Any) -> Any:
    batch, seq_len = inputs.shape
    token = inputs.reshape(-1, 1).to(dtype=base_hidden.dtype)
    token = token / max(float(inputs.max().item()), 1.0)
    position = torch.arange(seq_len, dtype=base_hidden.dtype, device=inputs.device).repeat(batch).reshape(-1, 1)
    position = position / max(float(seq_len - 1), 1.0)
    hidden = base_hidden.reshape(batch * seq_len, base_hidden.shape[-1])
    scores = learned_scores.reshape(batch * seq_len, learned_scores.shape[-1]).to(dtype=base_hidden.dtype)
    top2 = torch.topk(scores, k=min(2, scores.shape[-1]), dim=1).values
    margin = (top2[:, :1] - top2[:, 1:2]) if top2.shape[1] == 2 else top2[:, :1]
    return torch.cat([token, position, hidden, scores, margin], dim=1).to(dtype=torch.float32)


def _token_position_null_support(
    torch: Any,
    *,
    inputs: Any,
    flat_oracle: Any,
    train_indices: Any,
    pair_tuples: list[tuple[int, ...]],
    pair_lookup: dict[tuple[int, ...], int],
    top_k: int,
) -> Any:
    flat_inputs = inputs.reshape(-1)
    seq_len = inputs.shape[1]
    train_pairs = _pair_ids(torch, flat_oracle[train_indices], pair_lookup).tolist()
    counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    for idx, pair_id in zip(train_indices.tolist(), train_pairs):
        counts[(int(flat_inputs[idx].item()), int(idx % seq_len) % 8)].update([int(pair_id)])
    global_pair = Counter(train_pairs).most_common(1)[0][0]
    pred_ids = []
    for idx in range(flat_oracle.shape[0]):
        key = (int(flat_inputs[idx].item()), int(idx % seq_len) % 8)
        pred_ids.append(counts.get(key, Counter({global_pair: 1})).most_common(1)[0][0])
    return _support_from_pair_ids(
        torch,
        torch.as_tensor(pred_ids, dtype=torch.long, device=flat_oracle.device),
        pair_tuples,
        top_k=top_k,
        token_count=flat_oracle.shape[0],
        fallback=flat_oracle,
    )


def _majority_pair_support(
    torch: Any,
    *,
    flat_oracle: Any,
    train_indices: Any,
    pair_tuples: list[tuple[int, ...]],
    pair_lookup: dict[tuple[int, ...], int],
    top_k: int,
) -> Any:
    train_pair_ids = _pair_ids(torch, flat_oracle[train_indices], pair_lookup).tolist()
    majority_pair_id = Counter(train_pair_ids).most_common(1)[0][0]
    pair_ids = torch.full(
        (flat_oracle.shape[0],),
        int(majority_pair_id),
        dtype=torch.long,
        device=flat_oracle.device,
    )
    return _support_from_pair_ids(
        torch,
        pair_ids,
        pair_tuples,
        top_k=top_k,
        token_count=flat_oracle.shape[0],
        fallback=flat_oracle,
    )


def _delayed_pair_ids(torch: Any, train_pair_ids: Any, train_indices: Any, flat_oracle: Any, pair_lookup: dict[tuple[int, ...], int]) -> Any:
    del train_pair_ids, train_indices
    return _pair_ids(torch, flat_oracle.roll(shifts=1, dims=0), pair_lookup)


def _pair_ids(torch: Any, flat_support: Any, pair_lookup: dict[tuple[int, ...], int]) -> Any:
    return torch.as_tensor(
        [pair_lookup[tuple(sorted(int(item) for item in row.tolist()))] for row in flat_support],
        dtype=torch.long,
        device=flat_support.device,
    )


def _support_from_pair_ids(
    torch: Any,
    pair_ids: Any,
    pair_tuples: list[tuple[int, ...]],
    *,
    top_k: int,
    token_count: int,
    fallback: Any,
) -> Any:
    rows: list[list[int]] = []
    for pair_id in pair_ids.reshape(-1).tolist():
        pair = list(pair_tuples[int(pair_id)])
        if len(pair) < top_k:
            pair = pair + pair[:1] * (top_k - len(pair))
        rows.append(pair[:top_k])
    support = torch.as_tensor(rows, dtype=torch.long, device=fallback.device)
    if support.shape[0] != token_count:
        raise ValueError("predicted support row count does not match token count")
    return support


def _decoder(torch: Any, base_hidden: Any, frozen_decoder_state: dict[str, Any]) -> tuple[Any, Any | None]:
    weight = frozen_decoder_state.get("lm_head_weight")
    bias = frozen_decoder_state.get("lm_head_bias")
    if weight is None or not hasattr(weight, "to"):
        raise ValueError("frozen_decoder_state.pt does not contain lm_head_weight")
    weight = weight.to(dtype=base_hidden.dtype, device=base_hidden.device)
    if bias is not None and hasattr(bias, "to"):
        bias = bias.to(dtype=base_hidden.dtype, device=base_hidden.device)
    else:
        bias = None
    return weight, bias


def _metric_rows(
    torch: Any,
    F: Any,
    *,
    specs: list[tuple[str, Any, Any, Any, int, str]],
    oracle_support: Any,
    base_hidden: Any,
    base_logits: Any,
    targets: Any,
    teacher_logits: Any,
    teacher_hidden_residual: Any,
    decoder_weight: Any,
    decoder_bias: Any | None,
    train_indices: Any,
    holdout_indices: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batch, seq_len = targets.shape
    hidden_dim = base_hidden.shape[-1]
    for arm, support, design, values, feature_count, notes in specs:
        hidden_update = design.matmul(values).reshape(batch, seq_len, hidden_dim)
        logits = torch.matmul(base_hidden + hidden_update, decoder_weight.t())
        if decoder_bias is not None:
            logits = logits + decoder_bias
        for split, indices in (("train", train_indices), ("holdout", holdout_indices)):
            row = _split_pregate_metric_row(
                torch,
                F,
                arm=arm,
                split=split,
                logits=logits,
                base_hidden=base_hidden,
                base_logits=base_logits,
                hidden_update=hidden_update,
                targets=targets,
                teacher_logits=teacher_logits,
                teacher_hidden_residual=teacher_hidden_residual,
                indices=indices,
                feature_count=feature_count,
                split_seed=1729,
            )
            row.update(_support_metrics(torch, support, oracle_support, indices))
            row.update(_support_distribution_metrics(torch, oracle_support, indices))
            row["residual_norm_ratio"] = _norm_ratio(hidden_update.reshape(-1, hidden_dim)[indices], teacher_hidden_residual.reshape(-1, hidden_dim)[indices])
            row["residual_direction_error"] = 1.0 - _cosine(
                F,
                hidden_update.reshape(-1, hidden_dim)[indices],
                teacher_hidden_residual.reshape(-1, hidden_dim)[indices],
            )
            row["notes"] = notes
            rows.append(row)
    return rows


def _support_distribution_metrics(torch: Any, oracle_support: Any, indices: Any) -> dict[str, Any]:
    selected = oracle_support[indices]
    labels = [tuple(sorted(int(item) for item in row)) for row in selected.tolist()]
    counts = Counter(labels)
    total = max(len(labels), 1)
    probs = torch.as_tensor([count / total for count in counts.values()], dtype=torch.float32)
    entropy = float((-(probs * probs.clamp_min(1e-12).log())).sum().item())
    max_entropy = float(torch.log(torch.as_tensor(float(max(len(counts), 1)))).item())
    return {
        "support_pair_unique_count": len(counts),
        "support_pair_entropy": entropy,
        "support_pair_normalized_entropy": entropy / max(max_entropy, 1e-12) if len(counts) > 1 else 0.0,
        "support_pair_majority_fraction": max(counts.values()) / total,
    }


def _support_metrics(torch: Any, support: Any, oracle_support: Any, indices: Any) -> dict[str, Any]:
    pred = support[indices]
    oracle = oracle_support[indices]
    exact = (pred == oracle).all(dim=1).to(dtype=torch.float32)
    jaccards = []
    for pred_row, oracle_row in zip(pred.tolist(), oracle.tolist()):
        pred_set = set(int(item) for item in pred_row)
        oracle_set = set(int(item) for item in oracle_row)
        jaccards.append(len(pred_set & oracle_set) / max(len(pred_set | oracle_set), 1))
    return {
        "support_pair_accuracy": float(exact.mean().item()) if int(exact.numel()) else 0.0,
        "support_pair_jaccard": float(sum(jaccards) / max(len(jaccards), 1)),
    }


def _norm_ratio(prediction: Any, target: Any) -> float:
    return float(prediction.norm(dim=-1).mean().item() / max(target.norm(dim=-1).mean().item(), 1e-12))


def _gate_criteria(
    design: dict[str, Any],
    truth_audit: dict[str, Any],
    router_rows: list[dict[str, Any]],
    sentinel_rows: list[dict[str, Any]],
    runtime_failure: str,
) -> list[dict[str, Any]]:
    oracle_holdout = _row(router_rows, "oracle_pair_composer_reference", "holdout")
    learned_holdout = _row(router_rows, "learned_causal_pair_router", "holdout")
    majority_holdout = _row(sentinel_rows, "majority_pair_router_null", "holdout")
    delayed_holdout = _row(sentinel_rows, "delayed_pair_target_null", "holdout")
    misaligned_holdout = _row(sentinel_rows, "misaligned_support_pair_null", "holdout")
    source_ready = (
        design.get("status") == "pass"
        and design.get("selected_next_action") == "implement_pair_composer_control_extension_probe_locally"
        and truth_audit.get("status") == "pass"
    )
    oracle_ce = _float(oracle_holdout.get("true_decoder_ce_loss"))
    learned_ce = _float(learned_holdout.get("true_decoder_ce_loss"))
    majority_ce = _float(majority_holdout.get("true_decoder_ce_loss"))
    normalized_entropy = _float(oracle_holdout.get("support_pair_normalized_entropy"))
    majority_fraction = _float(oracle_holdout.get("support_pair_majority_fraction"))
    delayed_ce = _float(delayed_holdout.get("true_decoder_ce_loss"))
    misaligned_ce = _float(misaligned_holdout.get("true_decoder_ce_loss"))
    return [
        _criterion("required_sources_present", source_ready, "design and truth-audit sources must select the local probe"),
        _criterion("probe_runtime_completed", runtime_failure == "", "local tensor probe must complete without exception", runtime_failure),
        _criterion("oracle_reference_reproduced", oracle_ce is not None and oracle_ce > 0.0, "oracle pair-composer reference CE must be measured", oracle_ce),
        _criterion("learned_router_measured", learned_ce is not None and learned_ce > 0.0, "learned causal pair router CE must be measured", learned_ce),
        _criterion(
            "support_pair_class_balance_sufficient",
            normalized_entropy is not None
            and majority_fraction is not None
            and normalized_entropy >= 0.25
            and majority_fraction <= 0.8,
            "oracle support-pair labels must not be dominated by one class",
            {"normalized_entropy": normalized_entropy, "majority_fraction": majority_fraction},
        ),
        _criterion(
            "learned_router_beats_majority_pair_null",
            learned_ce is not None and majority_ce is not None and learned_ce <= majority_ce - 0.05,
            "learned router must beat the global majority-pair null by at least 0.05 CE",
            {"learned": learned_ce, "majority": majority_ce},
        ),
        _criterion(
            "delayed_and_misaligned_sentinels_measured",
            delayed_ce is not None and misaligned_ce is not None,
            "delayed and misaligned leakage sentinels must be measured",
            {"delayed": delayed_ce, "misaligned": misaligned_ce},
        ),
        _criterion(
            "leakage_sentinels_do_not_beat_oracle_reference",
            oracle_ce is not None
            and delayed_ce is not None
            and misaligned_ce is not None
            and delayed_ce >= oracle_ce
            and misaligned_ce >= oracle_ce,
            "sentinel controls should not beat the oracle pair-composer reference",
            {"oracle": oracle_ce, "delayed": delayed_ce, "misaligned": misaligned_ce},
        ),
        _criterion(
            "remaining_controls_complete_for_gpu",
            False,
            "norm/churn/retention, exact commutator, and matched dense/MLP interference controls are still absent",
        ),
    ]


def _candidate_actions(status: str) -> list[dict[str, str]]:
    if status != "pass":
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "probe sources or runtime failed",
                "repair the local probe before interpreting pair-composer controls",
                "source_or_runtime_repair_required",
            )
        ]
    return [
        _candidate(
            NEXT_ACTION,
            "selected",
            "learned-router and leakage-sentinel rows now exist, but the full mechanism/interference gate is incomplete",
            "extend the probe with norm/churn/retention, exact commutator, and matched dense/MLP interference controls",
            "pair_composer_probe_partial_controls_recorded_no_gpu",
        ),
        _candidate(
            "run_gpu_pair_composer_validation",
            "rejected",
            "local mechanism and interference controls are incomplete",
            "keep RunPod and Colab blocked",
            "gpu_validation_blocked",
        ),
    ]


def _summary(
    *,
    status: str,
    decision: str,
    claim_status: str,
    selected_next_action: str,
    start: float,
    out_dir: Path,
    source_rows: list[dict[str, Any]],
    router_rows: list[dict[str, Any]],
    sentinel_rows: list[dict[str, Any]],
    gate_criteria: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    rationale: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_action": selected_next_action,
        "selected_next_step": (
            "extend pair-composer control probe with norm/churn/retention, exact commutator, and matched dense/MLP controls"
            if status == "pass"
            else "repair pair-composer control-extension probe sources/runtime"
        ),
        "source_rows": source_rows,
        "router_rows": router_rows,
        "sentinel_rows": sentinel_rows,
        "gate_criteria": gate_criteria,
        "candidate_actions": _candidate_actions(status),
        "failures": failures,
        "rationale": rationale,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "out_dir": str(out_dir),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "router_rows.csv", summary["router_rows"])
    _write_csv(out_dir / "sentinel_rows.csv", summary["sentinel_rows"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Dense-Teacher Pair-Composer Control Extension Probe",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Claim status: `{summary['claim_status']}`",
            f"- Selected next action: `{summary['selected_next_action']}`",
            f"- Advance to GPU validation: `{summary.get('advance_to_gpu_validation', False)}`",
            "",
            "## Holdout Metrics",
            "",
            f"- Oracle pair-composer CE: `{summary.get('oracle_holdout_true_decoder_ce_loss')}`",
            f"- Learned causal pair-router CE: `{summary.get('learned_router_holdout_true_decoder_ce_loss')}`",
            f"- Learned support-pair accuracy: `{summary.get('learned_router_holdout_support_pair_accuracy')}`",
            f"- Learned support-pair Jaccard: `{summary.get('learned_router_holdout_support_pair_jaccard')}`",
            f"- Holdout support-pair normalized entropy: `{summary.get('holdout_support_pair_normalized_entropy')}`",
            f"- Majority-pair null CE: `{summary.get('majority_pair_holdout_true_decoder_ce_loss')}`",
            f"- Delayed null CE: `{summary.get('delayed_null_holdout_true_decoder_ce_loss')}`",
            f"- Misaligned null CE: `{summary.get('misaligned_null_holdout_true_decoder_ce_loss')}`",
            "",
            "## Interpretation",
            "",
            summary["rationale"],
            "",
            "GPU validation remains blocked.",
            "",
        ]
    )


def _source_json(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(payload),
        "required": True,
        "status": payload.get("status", "missing"),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
    }


def _tensor_source_rows(distillation_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for spec in REQUIRED_TENSORS:
        path = distillation_dir / str(spec["filename"])
        rows.append(
            {
                "source": f"tensor:{spec['tensor']}",
                "path": str(path),
                "present": path.is_file(),
                "required": True,
                "status": "present" if path.is_file() else "missing",
                "decision": "",
                "claim_status": str(spec["required_for"]),
            }
        )
    return rows


def _criterion(name: str, passed: bool, requirement: str, actual: Any = "") -> dict[str, Any]:
    return {
        "criterion": name,
        "passed": bool(passed),
        "requirement": requirement,
        "actual": actual,
        "failure_reason": "" if passed else requirement,
    }


def _candidate(action: str, disposition: str, reason: str, next_step: str, claim_status: str) -> dict[str, str]:
    return {
        "candidate_action": action,
        "disposition": disposition,
        "reason": reason,
        "next_step": next_step,
        "claim_status": claim_status,
    }


def _row(rows: list[dict[str, Any]], arm: str, split: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("arm") == arm and row.get("split") == split), {})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distillation-dir", type=Path, default=DEFAULT_DISTILLATION_DIR)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--truth-audit", type=Path, default=DEFAULT_TRUTH_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_dense_teacher_pair_composer_control_extension_probe(
        distillation_dir=args.distillation_dir,
        design_path=args.design,
        truth_audit_path=args.truth_audit,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("status", "decision", "selected_next_action", "advance_to_gpu_validation")
                if key in summary
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
