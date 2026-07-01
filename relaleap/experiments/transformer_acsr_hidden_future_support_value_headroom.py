"""Audit same-student support-value headroom for hidden/future Transformer-ACSR."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATASET = Path("results/reports/transformer_acsr_hidden_future_sequence_dataset/dataset_rows.csv")
DEFAULT_LOSS_LOOKUP = Path("results/reports/transformer_acsr_hidden_future_sequence_dataset/loss_lookup.csv")
DEFAULT_PREDICTIONS = Path(
    "results/reports/transformer_acsr_hidden_future_predictor_pregate/heldout_predictions.csv"
)
DEFAULT_OUT_DIR = Path("results/reports/transformer_acsr_hidden_future_support_value_headroom")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "context_value_headroom.csv",
    "split_summary.csv",
    "notes.md",
)


def run_transformer_acsr_hidden_future_support_value_headroom(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    loss_lookup_path: Path = DEFAULT_LOSS_LOOKUP,
    predictions_path: Path = DEFAULT_PREDICTIONS,
    out_dir: Path = DEFAULT_OUT_DIR,
    headroom_threshold: float = 0.005,
    entropy_temperature: float = 1.0,
) -> dict[str, Any]:
    """Compute oracle-vs-router support-value headroom from exact loss rows."""

    start = time.time()
    failures: list[dict[str, Any]] = []
    for name, path in (("dataset_rows", dataset_path), ("loss_lookup", loss_lookup_path)):
        if not path.is_file():
            failures.append({"source": name, "path": str(path), "reason": "required artifact missing"})
    if entropy_temperature <= 0.0:
        failures.append(
            {
                "source": "arguments",
                "path": "",
                "reason": "entropy_temperature must be positive",
            }
        )
    if failures:
        summary = _failed_summary(out_dir, failures, start)
        _write_artifacts(out_dir, summary, [], [])
        return summary

    dataset_rows = _read_csv(dataset_path)
    loss_rows = _read_csv(loss_lookup_path)
    predictions = _read_csv(predictions_path) if predictions_path.is_file() else []
    prediction_by_context = _prediction_by_context(predictions)
    context_rows = _dataset_context_rows(dataset_rows, failures)
    lookup = _loss_lookup(loss_rows, failures)
    num_columns = _num_columns(dataset_rows)
    expected_pairs = _all_pairs(num_columns)

    audit_rows: list[dict[str, Any]] = []
    for context_key, dataset_row in sorted(context_rows.items()):
        split, fold, sequence_id, position_index = context_key
        pair_losses = lookup.get(context_key, {})
        missing_pairs = [pair for pair in expected_pairs if pair not in pair_losses]
        if missing_pairs:
            failures.append(
                {
                    "source": "loss_lookup",
                    "path": str(loss_lookup_path),
                    "reason": "context missing exact support pairs",
                    "context": _context_id(context_key),
                    "missing_pair_count": len(missing_pairs),
                    "example_missing_pair": missing_pairs[0],
                }
            )
            continue

        ranked = sorted(pair_losses.items(), key=lambda item: (item[1], item[0]))
        oracle_pair, oracle_loss = ranked[0]
        second_loss = ranked[1][1] if len(ranked) > 1 else oracle_loss
        tenth_index = min(9, len(ranked) - 1)
        tenth_loss = ranked[tenth_index][1] if ranked else oracle_loss
        rank_by_pair = {pair: index + 1 for index, (pair, _) in enumerate(ranked)}

        student_pair = _pair_string(dataset_row["student_router_support_eval_only"])
        teacher_pair = _pair_string(dataset_row["teacher_topk_support_target_only"])
        predicted_pair = prediction_by_context.get(context_key)
        student_loss = pair_losses[student_pair]
        teacher_loss = pair_losses[teacher_pair]
        predicted_loss = pair_losses[predicted_pair] if predicted_pair else None

        audit_rows.append(
            {
                "split": split,
                "fold": fold,
                "sequence_id": sequence_id,
                "position_index": position_index,
                "student_router_support_pair": student_pair,
                "student_router_loss": student_loss,
                "student_router_rank": rank_by_pair[student_pair],
                "teacher_support_pair": teacher_pair,
                "teacher_loss": teacher_loss,
                "teacher_rank": rank_by_pair[teacher_pair],
                "predicted_support_pair": predicted_pair or "",
                "predicted_loss": "" if predicted_loss is None else predicted_loss,
                "predicted_rank": "" if predicted_pair is None else rank_by_pair[predicted_pair],
                "oracle_support_pair": oracle_pair,
                "oracle_loss": oracle_loss,
                "oracle_router_gap": student_loss - oracle_loss,
                "teacher_router_delta": teacher_loss - student_loss,
                "teacher_oracle_regret": teacher_loss - oracle_loss,
                "predicted_router_delta": "" if predicted_loss is None else predicted_loss - student_loss,
                "predicted_oracle_regret": "" if predicted_loss is None else predicted_loss - oracle_loss,
                "top2_loss_margin": second_loss - oracle_loss,
                "top10_loss_margin": tenth_loss - oracle_loss,
                "value_entropy": _value_entropy([loss for _, loss in ranked], entropy_temperature),
                "pair_count": len(ranked),
            }
        )

    if failures:
        summary = _failed_summary(out_dir, failures, start)
        _write_artifacts(out_dir, summary, audit_rows, [])
        return summary

    split_rows = _split_summary_rows(audit_rows, headroom_threshold)
    split_by_name = {row["split"]: row for row in split_rows}
    train_headroom = float(split_by_name.get("train", {}).get("mean_oracle_router_gap", 0.0))
    heldout_headroom = float(split_by_name.get("heldout", {}).get("mean_oracle_router_gap", 0.0))
    train_gate = bool(split_by_name.get("train", {}).get("oracle_headroom_gate_passes", False))
    heldout_gate = bool(split_by_name.get("heldout", {}).get("oracle_headroom_gate_passes", False))
    value_target_training_allowed = bool(train_gate and heldout_gate)
    decision = (
        "support_value_headroom_nontrivial_train_value_router_locally"
        if value_target_training_allowed
        else "support_value_headroom_negligible_close_teacher_imitation_before_gpu"
    )
    claim_status = (
        "oracle_support_has_action_value_room_over_student_router"
        if value_target_training_allowed
        else "teacher_support_imitation_has_insufficient_same_student_value_headroom"
    )
    summary = {
        "status": "pass",
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": (
            "train_prefix_safe_value_target_router_locally"
            if value_target_training_allowed
            else "close_transformer_acsr_teacher_support_imitation_and_select_next_local_mechanism"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "value_target_training_allowed": value_target_training_allowed,
        "headroom_threshold": headroom_threshold,
        "entropy_temperature": entropy_temperature,
        "dataset_path": str(dataset_path),
        "loss_lookup_path": str(loss_lookup_path),
        "predictions_path": str(predictions_path),
        "row_count": len(dataset_rows),
        "loss_lookup_row_count": len(loss_rows),
        "context_count": len(audit_rows),
        "expected_pair_count_per_context": len(expected_pairs),
        "train_mean_oracle_router_gap": train_headroom,
        "heldout_mean_oracle_router_gap": heldout_headroom,
        "split_summary": split_rows,
        "backend_policy": "local exact-loss headroom audit only; RunPod and Colab remain blocked",
        "strategy_review_handling": (
            "Accepted latest GPT-5.5-Pro recommendation to replace exact-commutator-first "
            "work with a same-student support-value headroom audit before training or GPU."
        ),
        "failures": [],
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    _write_artifacts(out_dir, summary, audit_rows, split_rows)
    return summary


def _dataset_context_rows(
    rows: list[dict[str, str]],
    failures: list[dict[str, Any]],
) -> dict[tuple[str, str, str, int], dict[str, str]]:
    contexts: dict[tuple[str, str, str, int], dict[str, str]] = {}
    for row in rows:
        key = _context_key(row)
        previous = contexts.setdefault(key, row)
        if previous is not row:
            failures.append(
                {
                    "source": "dataset_rows",
                    "path": "",
                    "reason": "duplicate dataset context",
                    "context": _context_id(key),
                }
            )
    return contexts


def _loss_lookup(
    rows: list[dict[str, str]],
    failures: list[dict[str, Any]],
) -> dict[tuple[str, str, str, int], dict[str, float]]:
    lookup: dict[tuple[str, str, str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = _context_key(row)
        pair = _pair_string(row["forced_support_pair"])
        if pair in lookup[key]:
            failures.append(
                {
                    "source": "loss_lookup",
                    "path": "",
                    "reason": "duplicate exact loss row for context/support pair",
                    "context": _context_id(key),
                    "support_pair": pair,
                }
            )
            continue
        lookup[key][pair] = float(row["forced_support_loss"])
    return dict(lookup)


def _prediction_by_context(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, int], str]:
    predictions: dict[tuple[str, str, str, int], str] = {}
    for row in rows:
        predictions[_context_key(row)] = _pair_string(row["predicted_support_pair"])
    return predictions


def _split_summary_rows(
    rows: list[dict[str, Any]],
    headroom_threshold: float,
) -> list[dict[str, Any]]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row["split"])].append(row)
    summaries: list[dict[str, Any]] = []
    for split, split_rows in sorted(by_split.items()):
        predicted_rows = [row for row in split_rows if row["predicted_loss"] != ""]
        mean_gap = _mean(split_rows, "oracle_router_gap")
        summaries.append(
            {
                "split": split,
                "context_count": len(split_rows),
                "mean_student_router_loss": _mean(split_rows, "student_router_loss"),
                "mean_teacher_loss": _mean(split_rows, "teacher_loss"),
                "mean_oracle_loss": _mean(split_rows, "oracle_loss"),
                "mean_predicted_loss": "" if not predicted_rows else _mean(predicted_rows, "predicted_loss"),
                "mean_oracle_router_gap": mean_gap,
                "median_oracle_router_gap": _median(split_rows, "oracle_router_gap"),
                "p90_oracle_router_gap": _quantile(split_rows, "oracle_router_gap", 0.9),
                "mean_teacher_router_delta": _mean(split_rows, "teacher_router_delta"),
                "mean_teacher_oracle_regret": _mean(split_rows, "teacher_oracle_regret"),
                "mean_predicted_router_delta": ""
                if not predicted_rows
                else _mean(predicted_rows, "predicted_router_delta"),
                "mean_predicted_oracle_regret": ""
                if not predicted_rows
                else _mean(predicted_rows, "predicted_oracle_regret"),
                "mean_student_router_rank": _mean(split_rows, "student_router_rank"),
                "mean_teacher_rank": _mean(split_rows, "teacher_rank"),
                "mean_predicted_rank": "" if not predicted_rows else _mean(predicted_rows, "predicted_rank"),
                "mean_top2_loss_margin": _mean(split_rows, "top2_loss_margin"),
                "mean_top10_loss_margin": _mean(split_rows, "top10_loss_margin"),
                "mean_value_entropy": _mean(split_rows, "value_entropy"),
                "headroom_threshold": headroom_threshold,
                "oracle_headroom_gate_passes": mean_gap >= headroom_threshold,
            }
        )
    return summaries


def _failed_summary(out_dir: Path, failures: list[dict[str, Any]], start: float) -> dict[str, Any]:
    return {
        "status": "fail",
        "decision": "support_value_headroom_failed_closed",
        "claim_status": "exact_loss_lookup_unusable_no_gpu",
        "selected_next_step": "repair_hidden_future_exact_loss_lookup_before_value_training",
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "value_target_training_allowed": False,
        "failures": failures,
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    audit_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "context_value_headroom.csv", audit_rows)
    _write_csv(out_dir / "split_summary.csv", split_rows)
    notes = [
        "# Transformer-ACSR Hidden/Future Support-Value Headroom",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Train mean oracle-router gap: `{summary.get('train_mean_oracle_router_gap', '')}`",
        f"- Heldout mean oracle-router gap: `{summary.get('heldout_mean_oracle_router_gap', '')}`",
        f"- Value-target training allowed: `{summary.get('value_target_training_allowed', False)}`",
        f"- Next step: `{summary['selected_next_step']}`",
        "",
        "This audit uses exact same-student forced-support losses only. It does",
        "not train a router and does not permit GPU validation by itself.",
    ]
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def _context_key(row: dict[str, str]) -> tuple[str, str, str, int]:
    return (row.get("split", ""), row.get("fold", ""), row["sequence_id"], int(row["position_index"]))


def _context_id(key: tuple[str, str, str, int]) -> str:
    split, fold, sequence_id, position_index = key
    return f"{split}/fold{fold}/{sequence_id}/pos{position_index}"


def _all_pairs(num_columns: int) -> list[str]:
    return [f"{left},{right}" for left in range(num_columns) for right in range(left + 1, num_columns)]


def _num_columns(rows: list[dict[str, str]]) -> int:
    if not rows:
        return 0
    return int(rows[0]["teacher_support_logit_dim"])


def _value_entropy(losses: list[float], temperature: float) -> float:
    if not losses:
        return 0.0
    offset = min(losses)
    weights = [math.exp(-(loss - offset) / temperature) for loss in losses]
    total = sum(weights)
    probs = [weight / total for weight in weights if weight > 0.0]
    return -sum(prob * math.log(prob) for prob in probs)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row[key] != ""]
    return sum(values) / len(values) if values else 0.0


def _median(rows: list[dict[str, Any]], key: str) -> float:
    values = sorted(float(row[key]) for row in rows if row[key] != "")
    if not values:
        return 0.0
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return (values[midpoint - 1] + values[midpoint]) / 2.0


def _quantile(rows: list[dict[str, Any]], key: str, q: float) -> float:
    values = sorted(float(row[key]) for row in rows if row[key] != "")
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return values[index]


def _pair_string(value: str) -> str:
    left, right = [int(part.strip()) for part in value.split(",", maxsplit=1)]
    left, right = sorted((left, right))
    return f"{left},{right}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--loss-lookup", type=Path, default=DEFAULT_LOSS_LOOKUP)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--headroom-threshold", type=float, default=0.005)
    parser.add_argument("--entropy-temperature", type=float, default=1.0)
    args = parser.parse_args(argv)
    summary = run_transformer_acsr_hidden_future_support_value_headroom(
        dataset_path=args.dataset,
        loss_lookup_path=args.loss_lookup,
        predictions_path=args.predictions,
        out_dir=args.out,
        headroom_threshold=args.headroom_threshold,
        entropy_temperature=args.entropy_temperature,
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
