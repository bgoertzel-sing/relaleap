"""Train a local support-only Transformer-ACSR pregate.

This command consumes the row dataset produced by
``transformer_acsr_sequence_dataset``. It trains a tiny causal transformer using
only prefix-safe row fields to predict the nondeployable teacher top-k2 support
pair, then compares it with token/position-only, shuffled-target, delayed, and
frequency controls.

The report remains fail-closed for GPU validation unless the local support
predictor clears registered null margins and the downstream intervention/churn
budget evidence exists.
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


DEFAULT_DATASET = Path("results/reports/transformer_acsr_sequence_dataset/dataset_rows.csv")
DEFAULT_DATASET_SUMMARY = Path("results/reports/transformer_acsr_sequence_dataset/summary.json")
DEFAULT_OUT_DIR = Path("results/reports/transformer_acsr_support_predictor_pregate")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "model_metrics.csv",
    "heldout_predictions.csv",
    "control_contract.csv",
    "notes.md",
)


@dataclass(frozen=True)
class SequencePacket:
    key: tuple[str, int, int]
    split: str
    rows: tuple[dict[str, str], ...]


class TinyCausalSupportTransformer(nn.Module):
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


def run_transformer_acsr_support_predictor_pregate(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    dataset_summary_path: Path = DEFAULT_DATASET_SUMMARY,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 17,
    epochs: int = 80,
    learning_rate: float = 0.01,
    margin: float = 0.03,
) -> dict[str, Any]:
    """Train local support predictor controls and write a fail-closed report."""

    start = time.time()
    random.seed(seed)
    torch.manual_seed(seed)
    failures: list[dict[str, Any]] = []
    dataset_summary = _read_json(dataset_summary_path)
    if not dataset_path.is_file():
        failures.append(
            {
                "source": "transformer_acsr_sequence_dataset",
                "path": str(dataset_path),
                "reason": "dataset_rows.csv missing",
            }
        )
    if dataset_summary and dataset_summary.get("trainable_support_only_now") is not True:
        failures.append(
            {
                "source": "transformer_acsr_sequence_dataset",
                "path": str(dataset_summary_path),
                "reason": "dataset summary does not mark support-only rows trainable",
            }
        )
    if failures:
        summary = _failed_summary(out_dir, failures, start)
        _write_artifacts(out_dir, summary, [], [], [])
        return summary

    rows = _read_csv(dataset_path)
    packets = _sequence_packets(rows)
    train_packets = [packet for packet in packets if packet.split == "train"]
    heldout_packets = [packet for packet in packets if packet.split == "heldout"]
    num_columns = _num_columns(rows)
    if not train_packets or not heldout_packets or num_columns <= 2:
        failures.append(
            {
                "source": "transformer_acsr_sequence_dataset",
                "path": str(dataset_path),
                "reason": "dataset lacks train/heldout sequences or valid support columns",
            }
        )
        summary = _failed_summary(out_dir, failures, start)
        _write_artifacts(out_dir, summary, [], [], [])
        return summary

    full_result = _train_and_score(
        train_packets=train_packets,
        heldout_packets=heldout_packets,
        num_columns=num_columns,
        feature_mode="prefix_support",
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        target_mode="teacher",
    )
    token_position_result = _train_and_score(
        train_packets=train_packets,
        heldout_packets=heldout_packets,
        num_columns=num_columns,
        feature_mode="token_position",
        seed=seed + 1,
        epochs=epochs,
        learning_rate=learning_rate,
        target_mode="teacher",
    )
    shuffled_result = _train_and_score(
        train_packets=train_packets,
        heldout_packets=heldout_packets,
        num_columns=num_columns,
        feature_mode="prefix_support",
        seed=seed + 2,
        epochs=epochs,
        learning_rate=learning_rate,
        target_mode="shuffled",
    )
    delayed_result = _delayed_control(heldout_packets, num_columns)
    frequency_result = _frequency_control(train_packets, heldout_packets, num_columns)

    metrics = [
        _metric_row("prefix_support_causal_transformer", "model", full_result),
        _metric_row("token_position_only_transformer", "shortcut_null", token_position_result),
        _metric_row("shuffled_target_transformer", "target_alignment_null", shuffled_result),
        _metric_row("delayed_previous_support", "temporal_null", delayed_result),
        _metric_row("frequency_support_pair", "frequency_null", frequency_result),
    ]
    metric_by_name = {row["model"]: row for row in metrics}
    null_margin_rows = _null_margin_rows(metric_by_name, margin)
    null_margin_gate_passes = all(row["gate_passes"] for row in null_margin_rows)
    downstream_rows = _control_contract_rows()
    downstream_gate_passes = all(row["status"] == "available" for row in downstream_rows)
    advance_to_gpu = bool(null_margin_gate_passes and downstream_gate_passes)

    summary = {
        "status": "pass",
        "decision": (
            "transformer_acsr_support_predictor_pregate_passed_local_gpu_ready"
            if advance_to_gpu
            else "transformer_acsr_support_predictor_pregate_gpu_blocked"
        ),
        "claim_status": (
            "support_only_prefix_transformer_clears_registered_gates"
            if advance_to_gpu
            else "support_only_prefix_transformer_does_not_clear_full_mechanism_gate"
        ),
        "selected_next_step": (
            "run_runpod_transformer_acsr_support_predictor_validation_with_artifact_checks"
            if advance_to_gpu
            else "extend_local_support_predictor_with_exact_same_student_churn_commutator_controls_or_close_branch"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": advance_to_gpu,
        "dataset_path": str(dataset_path),
        "dataset_summary_path": str(dataset_summary_path),
        "row_count": len(rows),
        "train_sequence_count": len(train_packets),
        "heldout_sequence_count": len(heldout_packets),
        "heldout_row_count": sum(len(packet.rows) for packet in heldout_packets),
        "num_columns": num_columns,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "null_margin": margin,
        "prefix_support_jaccard": metric_by_name[
            "prefix_support_causal_transformer"
        ]["heldout_mean_jaccard"],
        "token_position_jaccard": metric_by_name[
            "token_position_only_transformer"
        ]["heldout_mean_jaccard"],
        "shuffled_target_jaccard": metric_by_name[
            "shuffled_target_transformer"
        ]["heldout_mean_jaccard"],
        "delayed_jaccard": metric_by_name["delayed_previous_support"]["heldout_mean_jaccard"],
        "frequency_jaccard": metric_by_name["frequency_support_pair"]["heldout_mean_jaccard"],
        "null_margin_gate_passes": null_margin_gate_passes,
        "downstream_intervention_budget_gate_passes": downstream_gate_passes,
        "missing_downstream_controls": [
            row["control"] for row in downstream_rows if row["status"] != "available"
        ],
        "null_margin_rows": null_margin_rows,
        "backend_policy": (
            "local CPU support-only pregate; RunPod/Colab remain blocked unless null "
            "margins and exact same-student/churn/commutator controls pass"
        ),
        "failures": failures,
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    _write_artifacts(out_dir, summary, metrics, full_result["predictions"], downstream_rows)
    return summary


def _failed_summary(out_dir: Path, failures: list[dict[str, Any]], start: float) -> dict[str, Any]:
    return {
        "status": "fail",
        "decision": "transformer_acsr_support_predictor_pregate_failed_closed",
        "claim_status": "support_predictor_dataset_unavailable_no_gpu",
        "selected_next_step": "repair_transformer_acsr_sequence_dataset_before_training",
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
    model = TinyCausalSupportTransformer(
        feature_dim=train_features.shape[-1],
        num_columns=num_columns,
        d_model=32,
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
        predictions = _prediction_rows(heldout_packets, logits, num_columns)
    scores = _score_predictions(predictions)
    scores.update(
        {
            "heldout_bce": bce,
            "predictions": predictions,
        }
    )
    return scores


def _delayed_control(heldout_packets: list[SequencePacket], num_columns: int) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    for packet in heldout_packets:
        for row in packet.rows:
            left = int(row["previous_teacher_support_left"])
            right = int(row["previous_teacher_support_right"])
            if left < 0 or right < 0:
                left, right = 0, min(1, num_columns - 1)
            predictions.append(_prediction_row(packet, row, left, right))
    scores = _score_predictions(predictions)
    scores["heldout_bce"] = ""
    scores["predictions"] = predictions
    return scores


def _frequency_control(
    train_packets: list[SequencePacket],
    heldout_packets: list[SequencePacket],
    num_columns: int,
) -> dict[str, Any]:
    counts = [0] * num_columns
    for packet in train_packets:
        for row in packet.rows:
            counts[int(row["teacher_support_left"])] += 1
            counts[int(row["teacher_support_right"])] += 1
    left, right = sorted(range(num_columns), key=lambda idx: (-counts[idx], idx))[:2]
    predictions = [
        _prediction_row(packet, row, left, right)
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
    features = [
        float(row["position_fraction"]),
        float(int(row["position_parity"])),
        math.sin(float(row["position_fraction"]) * math.tau),
        math.cos(float(row["position_fraction"]) * math.tau),
    ]
    if feature_mode == "prefix_support":
        previous = [0.0] * num_columns
        for field in ("previous_teacher_support_left", "previous_teacher_support_right"):
            value = int(row[field])
            if 0 <= value < num_columns:
                previous[value] = 1.0
        features.extend(previous)
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
    target[int(row["teacher_support_left"])] = 1.0
    target[int(row["teacher_support_right"])] = 1.0
    return target


def _shuffled_targets(targets: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    flat = targets.reshape((-1, targets.shape[-1]))
    perm = torch.randperm(flat.shape[0], generator=generator)
    return flat[perm].reshape_as(targets)


def _padding_mask(packets: list[SequencePacket]) -> torch.Tensor:
    max_len = max(len(packet.rows) for packet in packets)
    rows = []
    for packet in packets:
        rows.append([False] * len(packet.rows) + [True] * (max_len - len(packet.rows)))
    return torch.tensor(rows, dtype=torch.bool)


def _masked_bce(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    per_entry = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    per_row = per_entry.mean(dim=-1)
    return per_row[valid].mean()


def _prediction_rows(
    packets: list[SequencePacket],
    logits: torch.Tensor,
    num_columns: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for packet_index, packet in enumerate(packets):
        for row_index, row in enumerate(packet.rows):
            top = torch.topk(logits[packet_index, row_index], k=2).indices.tolist()
            left, right = sorted(int(value) for value in top)
            rows.append(_prediction_row(packet, row, left, right))
    return rows


def _prediction_row(
    packet: SequencePacket,
    row: dict[str, str],
    pred_left: int,
    pred_right: int,
) -> dict[str, Any]:
    teacher = {int(row["teacher_support_left"]), int(row["teacher_support_right"])}
    pred = {pred_left, pred_right}
    return {
        "source": row["source"],
        "seed_index": row["seed_index"],
        "fold": row["fold"],
        "flat_position": row["flat_position"],
        "sequence_key": ":".join(str(part) for part in packet.key),
        "teacher_support_left": min(teacher),
        "teacher_support_right": max(teacher),
        "predicted_support_left": pred_left,
        "predicted_support_right": pred_right,
        "exact_pair_match": pred == teacher,
        "overlap_count": len(pred & teacher),
        "jaccard": len(pred & teacher) / len(pred | teacher),
    }


def _score_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {
            "heldout_exact_pair_match_rate": 0.0,
            "heldout_mean_overlap": 0.0,
            "heldout_mean_jaccard": 0.0,
        }
    return {
        "heldout_exact_pair_match_rate": sum(
            1 for row in predictions if row["exact_pair_match"]
        )
        / len(predictions),
        "heldout_mean_overlap": sum(float(row["overlap_count"]) for row in predictions)
        / len(predictions),
        "heldout_mean_jaccard": sum(float(row["jaccard"]) for row in predictions)
        / len(predictions),
    }


def _metric_row(model: str, role: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "role": role,
        "heldout_exact_pair_match_rate": result["heldout_exact_pair_match_rate"],
        "heldout_mean_overlap": result["heldout_mean_overlap"],
        "heldout_mean_jaccard": result["heldout_mean_jaccard"],
        "heldout_bce": result.get("heldout_bce", ""),
    }


def _null_margin_rows(metric_by_name: dict[str, dict[str, Any]], margin: float) -> list[dict[str, Any]]:
    primary = float(metric_by_name["prefix_support_causal_transformer"]["heldout_mean_jaccard"])
    rows = []
    for null_name in (
        "token_position_only_transformer",
        "shuffled_target_transformer",
        "delayed_previous_support",
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
        {
            "control": "token_position_only_transformer",
            "status": "available",
            "evidence": "trained in this report",
        },
        {
            "control": "shuffled_target_transformer",
            "status": "available",
            "evidence": "trained in this report",
        },
        {
            "control": "delayed_previous_support",
            "status": "available",
            "evidence": "computed from prefix-safe previous teacher support",
        },
        {
            "control": "frequency_support_pair",
            "status": "available",
            "evidence": "computed from train folds only",
        },
        {
            "control": "exact_arbitrary_pair_same_student_intervention",
            "status": "missing",
            "evidence": "dataset has forced losses only for teacher/student/oracle/token-null pairs",
        },
        {
            "control": "retention_churn_budget",
            "status": "missing",
            "evidence": "support-row dataset has no functional churn rows",
        },
        {
            "control": "finite_update_commutator_budget",
            "status": "missing",
            "evidence": "support-row dataset has no commutator rows",
        },
        {
            "control": "hidden_future_chunk_targets",
            "status": "missing",
            "evidence": "sequence dataset records missing current/future hidden tensors",
        },
    ]


def _sequence_packets(rows: list[dict[str, str]]) -> list[SequencePacket]:
    grouped: dict[tuple[str, int, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["source"], int(row["seed_index"]), int(row["fold"]))
        grouped.setdefault(key, []).append(row)
    packets = []
    for key, group in sorted(grouped.items()):
        ordered = tuple(sorted(group, key=lambda item: int(item["flat_position"])))
        split = ordered[0]["split"]
        packets.append(SequencePacket(key=key, split=split, rows=ordered))
    return packets


def _num_columns(rows: list[dict[str, str]]) -> int:
    max_column = 0
    for row in rows:
        for field in (
            "teacher_support_left",
            "teacher_support_right",
            "student_support_left",
            "student_support_right",
            "oracle_support_left",
            "oracle_support_right",
            "token_position_null_support_left",
            "token_position_null_support_right",
        ):
            max_column = max(max_column, int(row[field]))
    return max_column + 1


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
    predictions: list[dict[str, Any]],
    controls: list[dict[str, str]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "model_metrics.csv", metrics)
    _write_csv(out_dir / "heldout_predictions.csv", predictions)
    _write_csv(out_dir / "control_contract.csv", controls)
    notes = [
        "# Transformer-ACSR Support Predictor Pregate",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Prefix-support heldout Jaccard: `{summary.get('prefix_support_jaccard', '')}`",
        f"- Token/position heldout Jaccard: `{summary.get('token_position_jaccard', '')}`",
        f"- Null margin gate passes: `{summary.get('null_margin_gate_passes', False)}`",
        f"- Downstream intervention/budget gate passes: `{summary.get('downstream_intervention_budget_gate_passes', False)}`",
        f"- Next step: `{summary['selected_next_step']}`",
        "",
        "This is support-only evidence. It does not contain hidden/future chunks,",
        "arbitrary predicted-pair same-student intervention losses, retention/churn rows,",
        "or finite-update commutator rows, so GPU validation remains blocked unless",
        "those controls are later materialized and pass.",
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
    parser.add_argument("--dataset-summary", type=Path, default=DEFAULT_DATASET_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=0.03)
    args = parser.parse_args(argv)
    summary = run_transformer_acsr_support_predictor_pregate(
        dataset_path=args.dataset,
        dataset_summary_path=args.dataset_summary,
        out_dir=args.out,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        margin=args.margin,
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
