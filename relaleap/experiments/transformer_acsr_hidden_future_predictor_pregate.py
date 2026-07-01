"""Train a local prefix-safe Transformer-ACSR hidden/future pregate.

This command consumes ``transformer_acsr_hidden_future_sequence_dataset``. It
uses only prefix-safe current/previous hidden tensors and position features to
predict the nondeployable teacher top-k2 support pair, then scores predicted
pairs through the exact same-student forced-loss lookup. Future tensors, teacher
logits/support, target tokens, oracle labels, and loss columns remain target or
evaluation fields and are never predictor inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


DEFAULT_DATASET = Path("results/reports/transformer_acsr_hidden_future_sequence_dataset/dataset_rows.csv")
DEFAULT_LOSS_LOOKUP = Path("results/reports/transformer_acsr_hidden_future_sequence_dataset/loss_lookup.csv")
DEFAULT_DATASET_SUMMARY = Path("results/reports/transformer_acsr_hidden_future_sequence_dataset/summary.json")
DEFAULT_OUT_DIR = Path("results/reports/transformer_acsr_hidden_future_predictor_pregate")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "model_metrics.csv",
    "null_margin.csv",
    "heldout_predictions.csv",
    "control_contract.csv",
    "notes.md",
)


@dataclass(frozen=True)
class SequencePacket:
    key: str
    split: str
    rows: tuple[dict[str, str], ...]


class TinyHiddenFutureTransformer(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        num_columns: int,
        d_model: int,
        nhead: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, num_columns)

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        seq_len = features.shape[1]
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=features.device),
            diagonal=1,
        )
        hidden = self.input_proj(features)
        encoded = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.output(encoded)


def run_transformer_acsr_hidden_future_predictor_pregate(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    loss_lookup_path: Path = DEFAULT_LOSS_LOOKUP,
    dataset_summary_path: Path = DEFAULT_DATASET_SUMMARY,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 23,
    epochs: int = 60,
    learning_rate: float = 0.003,
    margin: float = 0.03,
    min_loss_improvement: float = 0.0,
) -> dict[str, Any]:
    """Train prefix-safe hidden predictor controls and write a fail-closed report."""

    start = time.time()
    random.seed(seed)
    torch.manual_seed(seed)
    failures: list[dict[str, Any]] = []
    dataset_summary = _read_json(dataset_summary_path)
    if not dataset_path.is_file():
        failures.append({"source": "hidden_future_sequence_dataset", "path": str(dataset_path), "reason": "dataset_rows.csv missing"})
    if not loss_lookup_path.is_file():
        failures.append({"source": "hidden_future_sequence_dataset", "path": str(loss_lookup_path), "reason": "loss_lookup.csv missing"})
    if dataset_summary and dataset_summary.get("trainability_gate_passes") is not True:
        failures.append(
            {
                "source": "hidden_future_sequence_dataset",
                "path": str(dataset_summary_path),
                "reason": "dataset summary does not mark hidden/future rows locally trainable",
            }
        )
    if failures:
        summary = _failed_summary(out_dir, failures, start)
        _write_artifacts(out_dir, summary, [], [], [], [])
        return summary

    rows = _read_csv(dataset_path)
    loss_lookup = _loss_lookup(_read_csv(loss_lookup_path))
    packets = _sequence_packets(rows)
    train_packets = [packet for packet in packets if packet.split == "train"]
    heldout_packets = [packet for packet in packets if packet.split == "heldout"]
    num_columns = _num_columns(rows)
    hidden_dim = _hidden_dim(rows)
    if not train_packets or not heldout_packets or num_columns <= 2 or hidden_dim <= 0:
        failures.append(
            {
                "source": "hidden_future_sequence_dataset",
                "path": str(dataset_path),
                "reason": "dataset lacks train/heldout sequences or valid hidden/support dimensions",
            }
        )
    if not _all_heldout_pairs_scored(heldout_packets, loss_lookup):
        failures.append(
            {
                "source": "hidden_future_sequence_dataset",
                "path": str(loss_lookup_path),
                "reason": "exact loss lookup is incomplete for heldout contexts",
            }
        )
    if failures:
        summary = _failed_summary(out_dir, failures, start)
        _write_artifacts(out_dir, summary, [], [], [], [])
        return summary

    primary = _train_and_score(
        train_packets=train_packets,
        heldout_packets=heldout_packets,
        loss_lookup=loss_lookup,
        num_columns=num_columns,
        feature_mode="prefix_hidden",
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        target_mode="teacher",
    )
    token_position = _train_and_score(
        train_packets=train_packets,
        heldout_packets=heldout_packets,
        loss_lookup=loss_lookup,
        num_columns=num_columns,
        feature_mode="token_position",
        seed=seed + 1,
        epochs=epochs,
        learning_rate=learning_rate,
        target_mode="teacher",
    )
    shuffled = _train_and_score(
        train_packets=train_packets,
        heldout_packets=heldout_packets,
        loss_lookup=loss_lookup,
        num_columns=num_columns,
        feature_mode="prefix_hidden",
        seed=seed + 2,
        epochs=epochs,
        learning_rate=learning_rate,
        target_mode="shuffled",
    )
    delayed = _delayed_control(heldout_packets, loss_lookup, num_columns)
    frequency = _frequency_control(train_packets, heldout_packets, loss_lookup, num_columns)

    metrics = [
        _metric_row("prefix_hidden_causal_transformer", "model", primary),
        _metric_row("token_position_only_transformer", "shortcut_null", token_position),
        _metric_row("shuffled_target_transformer", "target_alignment_null", shuffled),
        _metric_row("delayed_previous_teacher_support", "temporal_null", delayed),
        _metric_row("frequency_support_pair", "frequency_null", frequency),
    ]
    metric_by_name = {row["model"]: row for row in metrics}
    null_margin_rows = _null_margin_rows(metric_by_name, margin)
    null_margin_gate_passes = all(row["gate_passes"] for row in null_margin_rows)
    same_student_loss_gate_passes = (
        float(metric_by_name["prefix_hidden_causal_transformer"]["mean_forced_minus_student_router_loss"])
        <= -min_loss_improvement
    )
    control_rows = _control_contract_rows()
    downstream_gate_passes = same_student_loss_gate_passes and all(
        row["status"] == "available" for row in control_rows
    )
    advance_to_gpu = bool(null_margin_gate_passes and downstream_gate_passes)

    summary = {
        "status": "pass",
        "decision": (
            "transformer_acsr_hidden_future_predictor_pregate_passed_local_gpu_ready"
            if advance_to_gpu
            else "transformer_acsr_hidden_future_predictor_pregate_gpu_blocked"
        ),
        "claim_status": (
            "prefix_safe_hidden_transformer_clears_registered_local_gates"
            if advance_to_gpu
            else "prefix_safe_hidden_transformer_does_not_clear_full_mechanism_gate"
        ),
        "selected_next_step": (
            "run_runpod_hidden_future_transformer_acsr_validation_with_artifact_checks"
            if advance_to_gpu
            else "write_hidden_future_predictor_closeout_or_scale_local_capture_before_gpu"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": advance_to_gpu,
        "dataset_path": str(dataset_path),
        "loss_lookup_path": str(loss_lookup_path),
        "dataset_summary_path": str(dataset_summary_path),
        "row_count": len(rows),
        "loss_lookup_row_count": len(loss_lookup),
        "train_sequence_count": len(train_packets),
        "heldout_sequence_count": len(heldout_packets),
        "heldout_row_count": sum(len(packet.rows) for packet in heldout_packets),
        "num_columns": num_columns,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "null_margin": margin,
        "min_loss_improvement": min_loss_improvement,
        "prefix_hidden_jaccard": metric_by_name["prefix_hidden_causal_transformer"]["heldout_mean_jaccard"],
        "token_position_jaccard": metric_by_name["token_position_only_transformer"]["heldout_mean_jaccard"],
        "shuffled_target_jaccard": metric_by_name["shuffled_target_transformer"]["heldout_mean_jaccard"],
        "delayed_jaccard": metric_by_name["delayed_previous_teacher_support"]["heldout_mean_jaccard"],
        "frequency_jaccard": metric_by_name["frequency_support_pair"]["heldout_mean_jaccard"],
        "prefix_hidden_mean_forced_loss": metric_by_name["prefix_hidden_causal_transformer"]["mean_forced_support_loss"],
        "prefix_hidden_mean_forced_minus_oracle_loss": metric_by_name["prefix_hidden_causal_transformer"]["mean_forced_minus_oracle_loss"],
        "prefix_hidden_mean_forced_minus_student_router_loss": metric_by_name["prefix_hidden_causal_transformer"]["mean_forced_minus_student_router_loss"],
        "null_margin_gate_passes": null_margin_gate_passes,
        "same_student_loss_gate_passes": same_student_loss_gate_passes,
        "downstream_intervention_budget_gate_passes": downstream_gate_passes,
        "missing_downstream_controls": [
            row["control"] for row in control_rows if row["status"] != "available"
        ],
        "null_margin_rows": null_margin_rows,
        "backend_policy": (
            "local CPU hidden/future pregate; RunPod/Colab remain blocked unless "
            "null margins, exact same-student loss, retention/churn, commutator, "
            "and future-perturbation controls pass"
        ),
        "forbidden_predictor_fields_enforced": [
            "future_hidden_json_target_only",
            "future_delta_json_target_only",
            "teacher_support_logits_json_target_only",
            "teacher_topk_support_target_only",
            "target_token_eval_only",
            "oracle_support_eval_only",
            "loss_lookup_fields",
        ],
        "failures": failures,
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    _write_artifacts(out_dir, summary, metrics, null_margin_rows, primary["predictions"], control_rows)
    return summary


def _failed_summary(out_dir: Path, failures: list[dict[str, Any]], start: float) -> dict[str, Any]:
    return {
        "status": "fail",
        "decision": "transformer_acsr_hidden_future_predictor_pregate_failed_closed",
        "claim_status": "hidden_future_predictor_dataset_unavailable_no_gpu",
        "selected_next_step": "repair_hidden_future_sequence_dataset_before_training",
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "failures": failures,
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }


def _train_and_score(
    *,
    train_packets: list[SequencePacket],
    heldout_packets: list[SequencePacket],
    loss_lookup: dict[tuple[str, int, str], dict[str, float]],
    num_columns: int,
    feature_mode: str,
    seed: int,
    epochs: int,
    learning_rate: float,
    target_mode: str,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    train_targets = _target_sequences(train_packets, num_columns)
    if target_mode == "shuffled":
        train_targets = _shuffled_targets(train_targets, seed)
    train_features = _feature_sequences(train_packets, num_columns, feature_mode)
    model = TinyHiddenFutureTransformer(
        feature_dim=train_features.shape[-1],
        num_columns=num_columns,
        d_model=48,
        nhead=4,
        num_layers=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    train_padding = _padding_mask(train_packets)
    train_valid = ~train_padding
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(train_features, train_padding)
        loss = _masked_bce(logits, train_targets, train_valid)
        loss.backward()
        optimizer.step()

    heldout_features = _feature_sequences(heldout_packets, num_columns, feature_mode)
    heldout_targets = _target_sequences(heldout_packets, num_columns)
    heldout_padding = _padding_mask(heldout_packets)
    heldout_valid = ~heldout_padding
    model.eval()
    with torch.no_grad():
        logits = model(heldout_features, heldout_padding)
        bce = _masked_bce(logits, heldout_targets, heldout_valid).item()
        predictions = _prediction_rows(heldout_packets, logits, loss_lookup)
    scores = _score_predictions(predictions)
    scores.update({"heldout_bce": bce, "predictions": predictions})
    return scores


def _delayed_control(
    heldout_packets: list[SequencePacket],
    loss_lookup: dict[tuple[str, int, str], dict[str, float]],
    num_columns: int,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    for packet in heldout_packets:
        previous = None
        for row in packet.rows:
            if previous is None:
                pair = _parse_pair(row["student_router_support_eval_only"])
            else:
                pair = _parse_pair(previous["teacher_topk_support_target_only"])
            predictions.append(_prediction_row(packet, row, pair[0], pair[1], loss_lookup))
            previous = row
    scores = _score_predictions(predictions)
    scores["heldout_bce"] = ""
    scores["predictions"] = predictions
    return scores


def _frequency_control(
    train_packets: list[SequencePacket],
    heldout_packets: list[SequencePacket],
    loss_lookup: dict[tuple[str, int, str], dict[str, float]],
    num_columns: int,
) -> dict[str, Any]:
    counts = [0] * num_columns
    for packet in train_packets:
        for row in packet.rows:
            left, right = _parse_pair(row["teacher_topk_support_target_only"])
            counts[left] += 1
            counts[right] += 1
    left, right = sorted(range(num_columns), key=lambda idx: (-counts[idx], idx))[:2]
    predictions = [
        _prediction_row(packet, row, left, right, loss_lookup)
        for packet in heldout_packets
        for row in packet.rows
    ]
    scores = _score_predictions(predictions)
    scores["heldout_bce"] = ""
    scores["predictions"] = predictions
    return scores


def _feature_sequences(
    packets: list[SequencePacket],
    num_columns: int,
    feature_mode: str,
) -> torch.Tensor:
    max_len = max(len(packet.rows) for packet in packets)
    features: list[list[list[float]]] = []
    for packet in packets:
        seq_features = [_row_features(row, num_columns, feature_mode) for row in packet.rows]
        while len(seq_features) < max_len:
            seq_features.append([0.0] * len(seq_features[0]))
        features.append(seq_features)
    return torch.tensor(features, dtype=torch.float32)


def _row_features(row: dict[str, str], num_columns: int, feature_mode: str) -> list[float]:
    position_index = float(row["position_index"])
    denom = max(1.0, float(position_index + 1.0))
    features = [
        position_index / denom,
        float(int(position_index) % 2),
        math.sin(position_index * math.tau / 64.0),
        math.cos(position_index * math.tau / 64.0),
    ]
    if feature_mode == "prefix_hidden":
        features.extend(_json_vector(row["current_hidden_json"]))
        features.extend(_json_vector(row["previous_hidden_json"]))
    elif feature_mode != "token_position":
        raise ValueError(f"unknown feature_mode: {feature_mode}")
    return features


def _target_sequences(packets: list[SequencePacket], num_columns: int) -> torch.Tensor:
    max_len = max(len(packet.rows) for packet in packets)
    targets: list[list[list[float]]] = []
    for packet in packets:
        seq_targets = [_target_vector(row, num_columns) for row in packet.rows]
        while len(seq_targets) < max_len:
            seq_targets.append([0.0] * num_columns)
        targets.append(seq_targets)
    return torch.tensor(targets, dtype=torch.float32)


def _target_vector(row: dict[str, str], num_columns: int) -> list[float]:
    target = [0.0] * num_columns
    left, right = _parse_pair(row["teacher_topk_support_target_only"])
    target[left] = 1.0
    target[right] = 1.0
    return target


def _shuffled_targets(targets: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    flat = targets.reshape((-1, targets.shape[-1]))
    perm = torch.randperm(flat.shape[0], generator=generator)
    return flat[perm].reshape_as(targets)


def _padding_mask(packets: list[SequencePacket]) -> torch.Tensor:
    max_len = max(len(packet.rows) for packet in packets)
    return torch.tensor(
        [[False] * len(packet.rows) + [True] * (max_len - len(packet.rows)) for packet in packets],
        dtype=torch.bool,
    )


def _masked_bce(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    per_entry = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return per_entry.mean(dim=-1)[valid].mean()


def _prediction_rows(
    packets: list[SequencePacket],
    logits: torch.Tensor,
    loss_lookup: dict[tuple[str, int, str], dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for packet_index, packet in enumerate(packets):
        for row_index, row in enumerate(packet.rows):
            top = torch.topk(logits[packet_index, row_index], k=2).indices.tolist()
            left, right = sorted(int(value) for value in top)
            rows.append(_prediction_row(packet, row, left, right, loss_lookup))
    return rows


def _prediction_row(
    packet: SequencePacket,
    row: dict[str, str],
    pred_left: int,
    pred_right: int,
    loss_lookup: dict[tuple[str, int, str], dict[str, float]],
) -> dict[str, Any]:
    teacher_left, teacher_right = _parse_pair(row["teacher_topk_support_target_only"])
    teacher = {teacher_left, teacher_right}
    pred_left, pred_right = sorted((pred_left, pred_right))
    pred = {pred_left, pred_right}
    pair = f"{pred_left},{pred_right}"
    loss_row = loss_lookup[(row["sequence_id"], int(row["position_index"]), pair)]
    return {
        "sequence_id": row["sequence_id"],
        "split": row["split"],
        "fold": row["fold"],
        "flat_position": row["flat_position"],
        "position_index": row["position_index"],
        "sequence_key": packet.key,
        "teacher_support_left": min(teacher),
        "teacher_support_right": max(teacher),
        "predicted_support_left": pred_left,
        "predicted_support_right": pred_right,
        "predicted_support_pair": pair,
        "exact_pair_match": pred == teacher,
        "overlap_count": len(pred & teacher),
        "jaccard": len(pred & teacher) / len(pred | teacher),
        "forced_support_loss": loss_row["forced_support_loss"],
        "forced_minus_oracle_loss": loss_row["forced_minus_oracle_loss"],
        "forced_minus_student_router_loss": loss_row["forced_minus_student_router_loss"],
    }


def _score_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {
            "heldout_exact_pair_match_rate": 0.0,
            "heldout_mean_overlap": 0.0,
            "heldout_mean_jaccard": 0.0,
            "mean_forced_support_loss": 0.0,
            "mean_forced_minus_oracle_loss": 0.0,
            "mean_forced_minus_student_router_loss": 0.0,
        }
    count = len(predictions)
    return {
        "heldout_exact_pair_match_rate": sum(1 for row in predictions if row["exact_pair_match"]) / count,
        "heldout_mean_overlap": sum(float(row["overlap_count"]) for row in predictions) / count,
        "heldout_mean_jaccard": sum(float(row["jaccard"]) for row in predictions) / count,
        "mean_forced_support_loss": sum(float(row["forced_support_loss"]) for row in predictions) / count,
        "mean_forced_minus_oracle_loss": sum(float(row["forced_minus_oracle_loss"]) for row in predictions) / count,
        "mean_forced_minus_student_router_loss": sum(float(row["forced_minus_student_router_loss"]) for row in predictions) / count,
    }


def _metric_row(model: str, role: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "role": role,
        "heldout_exact_pair_match_rate": result["heldout_exact_pair_match_rate"],
        "heldout_mean_overlap": result["heldout_mean_overlap"],
        "heldout_mean_jaccard": result["heldout_mean_jaccard"],
        "heldout_bce": result.get("heldout_bce", ""),
        "mean_forced_support_loss": result["mean_forced_support_loss"],
        "mean_forced_minus_oracle_loss": result["mean_forced_minus_oracle_loss"],
        "mean_forced_minus_student_router_loss": result["mean_forced_minus_student_router_loss"],
    }


def _null_margin_rows(metric_by_name: dict[str, dict[str, Any]], margin: float) -> list[dict[str, Any]]:
    primary = float(metric_by_name["prefix_hidden_causal_transformer"]["heldout_mean_jaccard"])
    rows = []
    for null_name in (
        "token_position_only_transformer",
        "shuffled_target_transformer",
        "delayed_previous_teacher_support",
        "frequency_support_pair",
    ):
        null_score = float(metric_by_name[null_name]["heldout_mean_jaccard"])
        delta = primary - null_score
        rows.append(
            {
                "gate": f"beats_{null_name}",
                "primary_jaccard": primary,
                "null_jaccard": null_score,
                "delta": delta,
                "required_margin": margin,
                "gate_passes": delta >= margin,
            }
        )
    return rows


def _control_contract_rows() -> list[dict[str, str]]:
    return [
        {"control": "token_position_only_transformer", "status": "available", "evidence": "trained in this report"},
        {"control": "shuffled_target_transformer", "status": "available", "evidence": "trained in this report"},
        {"control": "delayed_previous_teacher_support", "status": "available", "evidence": "computed from prior prefix target"},
        {"control": "frequency_support_pair", "status": "available", "evidence": "computed from train sequences only"},
        {"control": "exact_arbitrary_pair_same_student_intervention", "status": "available", "evidence": "scored from loss_lookup.csv after prediction only"},
        {"control": "retention_churn_budget", "status": "missing", "evidence": "hidden/future dataset has no retention/churn rows"},
        {"control": "finite_update_commutator_budget", "status": "missing", "evidence": "hidden/future dataset has no commutator rows"},
        {"control": "future_perturbation_invariance", "status": "missing", "evidence": "no perturbed-future reroute rows in this dataset"},
    ]


def _sequence_packets(rows: list[dict[str, str]]) -> list[SequencePacket]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["sequence_id"], []).append(row)
    packets = []
    for key, group in sorted(grouped.items()):
        ordered = tuple(sorted(group, key=lambda item: int(item["position_index"])))
        packets.append(SequencePacket(key=key, split=ordered[0]["split"], rows=ordered))
    return packets


def _loss_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, int, str], dict[str, float]]:
    lookup: dict[tuple[str, int, str], dict[str, float]] = {}
    for row in rows:
        pair = _pair_string(row["forced_support_pair"])
        lookup[(row["sequence_id"], int(row["position_index"]), pair)] = {
            "forced_support_loss": float(row["forced_support_loss"]),
            "forced_minus_oracle_loss": float(row["forced_minus_oracle_loss"]),
            "forced_minus_student_router_loss": float(row["forced_minus_student_router_loss"]),
        }
    return lookup


def _all_heldout_pairs_scored(
    packets: list[SequencePacket],
    lookup: dict[tuple[str, int, str], dict[str, float]],
) -> bool:
    for packet in packets:
        for row in packet.rows:
            if row["split"] != "heldout":
                continue
            for left in range(int(row["teacher_support_logit_dim"])):
                for right in range(left + 1, int(row["teacher_support_logit_dim"])):
                    if (row["sequence_id"], int(row["position_index"]), f"{left},{right}") not in lookup:
                        return False
    return True


def _num_columns(rows: list[dict[str, str]]) -> int:
    if not rows:
        return 0
    return int(rows[0]["teacher_support_logit_dim"])


def _hidden_dim(rows: list[dict[str, str]]) -> int:
    if not rows:
        return 0
    return int(rows[0]["hidden_dim"])


def _parse_pair(value: str) -> tuple[int, int]:
    left, right = _pair_string(value).split(",", maxsplit=1)
    return tuple(sorted((int(left), int(right))))  # type: ignore[return-value]


def _pair_string(value: str) -> str:
    left, right = [int(part.strip()) for part in value.split(",", maxsplit=1)]
    left, right = sorted((left, right))
    return f"{left},{right}"


def _json_vector(value: str) -> list[float]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("expected JSON vector")
    return [float(item) for item in parsed]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    metrics: list[dict[str, Any]],
    null_margins: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    controls: list[dict[str, str]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "model_metrics.csv", metrics)
    _write_csv(out_dir / "null_margin.csv", null_margins)
    _write_csv(out_dir / "heldout_predictions.csv", predictions)
    _write_csv(out_dir / "control_contract.csv", controls)
    notes = [
        "# Transformer-ACSR Hidden/Future Predictor Pregate",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Prefix-hidden heldout Jaccard: `{summary.get('prefix_hidden_jaccard', '')}`",
        f"- Token/position heldout Jaccard: `{summary.get('token_position_jaccard', '')}`",
        f"- Mean predicted forced-minus-student loss: `{summary.get('prefix_hidden_mean_forced_minus_student_router_loss', '')}`",
        f"- Null margin gate passes: `{summary.get('null_margin_gate_passes', False)}`",
        f"- Same-student loss gate passes: `{summary.get('same_student_loss_gate_passes', False)}`",
        f"- Downstream intervention/budget gate passes: `{summary.get('downstream_intervention_budget_gate_passes', False)}`",
        f"- Next step: `{summary['selected_next_step']}`",
        "",
        "Predictor inputs are limited to current hidden, previous hidden, and",
        "position features. Exact forced losses are used only after prediction.",
        "Retention/churn, finite-update commutator, and future-perturbation controls",
        "are still missing, so GPU validation remains blocked unless they are later",
        "materialized and pass.",
    ]
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--loss-lookup", type=Path, default=DEFAULT_LOSS_LOOKUP)
    parser.add_argument("--dataset-summary", type=Path, default=DEFAULT_DATASET_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--min-loss-improvement", type=float, default=0.0)
    args = parser.parse_args(argv)
    summary = run_transformer_acsr_hidden_future_predictor_pregate(
        dataset_path=args.dataset,
        loss_lookup_path=args.loss_lookup,
        dataset_summary_path=args.dataset_summary,
        out_dir=args.out,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        margin=args.margin,
        min_loss_improvement=args.min_loss_improvement,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "selected_next_step": summary["selected_next_step"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
