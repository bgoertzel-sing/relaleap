"""Materialize a conservative row dataset for Transformer-ACSR support targets.

The dataset emitted here is intentionally narrow. Existing artifacts expose
per-token teacher support labels and sequence-heldout folds, but not raw hidden
tensors or future-context chunks. This command turns those row artifacts into a
prefix-safe support-target dataset and records the missing fields needed before
any hidden-chunk Transformer-ACSR training claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path("results/reports/transformer_acsr_sequence_dataset")
DEFAULT_SOURCE_PATHS: tuple[tuple[str, Path, bool], ...] = (
    (
        "seed1_distillation_agreement",
        Path("results/audits/token_larger_causal_contextual_router_distillation_agreement/per_token_supports.csv"),
        True,
    ),
    (
        "seed2_distillation_agreement",
        Path("results/audits/token_larger_causal_contextual_router_distillation_agreement_seed2/per_token_supports.csv"),
        True,
    ),
    (
        "seed3_distillation_agreement",
        Path("results/audits/token_larger_causal_contextual_router_distillation_agreement_seed3/per_token_supports.csv"),
        True,
    ),
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "dataset_rows.csv",
    "source_inventory.csv",
    "field_provenance.csv",
    "split_manifest.csv",
    "missing_tensor_fields.csv",
    "notes.md",
)

TRAIN_FOLDS = {0, 1, 2}
HELDOUT_FOLDS = {3}


def run_transformer_acsr_sequence_dataset(
    *,
    source_paths: tuple[tuple[str, Path, bool], ...] = DEFAULT_SOURCE_PATHS,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a sequence-fold support-target row dataset from existing artifacts."""

    start = time.time()
    source_inventory: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for seed_index, (source_name, path, required) in enumerate(source_paths, start=1):
        if not path.is_file():
            source_inventory.append(_source_inventory_row(source_name, path, required, 0, "missing"))
            if required:
                failures.append(
                    {
                        "source": source_name,
                        "path": str(path),
                        "reason": "required per-token support source missing",
                    }
                )
            continue
        source_rows = _read_csv(path)
        source_inventory.append(
            _source_inventory_row(source_name, path, required, len(source_rows), "available")
        )
        dataset_rows.extend(
            _materialize_source_rows(
                source_name=source_name,
                seed_index=seed_index,
                rows=source_rows,
            )
        )

    field_rows = _field_provenance_rows()
    split_rows = _split_manifest_rows(dataset_rows)
    missing_tensor_rows = _missing_tensor_field_rows()
    support_target_dataset_available = bool(dataset_rows) and not failures
    sequence_split_available = _has_train_and_heldout(split_rows)
    trainable_support_only_now = support_target_dataset_available and sequence_split_available
    hidden_or_future_chunk_targets_available = False

    decision = (
        "transformer_acsr_support_target_row_dataset_materialized"
        if trainable_support_only_now
        else "transformer_acsr_support_target_row_dataset_failed_closed"
    )
    claim_status = (
        "support_distribution_dataset_ready_hidden_chunks_missing_no_gpu"
        if trainable_support_only_now
        else "support_distribution_dataset_unavailable_no_gpu"
    )
    selected_next_step = (
        "train_local_support_only_prefix_safe_transformer_acsr_pregate_with_nulls"
        if trainable_support_only_now
        else "repair_missing_per_token_support_sources_before_training"
    )
    hidden_chunk_next_step = (
        "extend command-driven artifact capture for current_hidden future_hidden future_delta tensors"
    )

    summary = {
        "status": "fail" if failures else "pass",
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "hidden_chunk_next_step": hidden_chunk_next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "support_target_dataset_available": support_target_dataset_available,
        "sequence_split_available": sequence_split_available,
        "trainable_support_only_now": trainable_support_only_now,
        "hidden_or_future_chunk_targets_available": hidden_or_future_chunk_targets_available,
        "row_count": len(dataset_rows),
        "source_count": len(source_inventory),
        "available_source_count": sum(1 for row in source_inventory if row["status"] == "available"),
        "train_row_count": sum(1 for row in dataset_rows if row["split"] == "train"),
        "heldout_row_count": sum(1 for row in dataset_rows if row["split"] == "heldout"),
        "prefix_safe_feature_count": sum(1 for row in field_rows if row["allowed_as_predictor_input"]),
        "nondeployable_teacher_target_count": sum(
            1 for row in field_rows if row["nondeployable_teacher_target"]
        ),
        "forbidden_predictor_field_count": sum(
            1 for row in field_rows if row["forbidden_predictor_input"]
        ),
        "missing_tensor_field_count": len(missing_tensor_rows),
        "failures": failures,
        "backend_policy": (
            "local row-materialization only; RunPod and Colab remain blocked until a "
            "local prefix-safe support predictor clears null/intervention/churn gates"
        ),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    _write_artifacts(
        out_dir=out_dir,
        summary=summary,
        dataset_rows=dataset_rows,
        source_inventory=source_inventory,
        field_rows=field_rows,
        split_rows=split_rows,
        missing_tensor_rows=missing_tensor_rows,
    )
    return summary


def _materialize_source_rows(
    *,
    source_name: str,
    seed_index: int,
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (int(row["fold"]), int(row["flat_position"])))
    previous_teacher_by_fold: dict[int, tuple[int, int] | None] = {}
    materialized: list[dict[str, Any]] = []
    for row in ordered:
        fold = int(row["fold"])
        flat_position = int(row["flat_position"])
        teacher_left, teacher_right = _parse_support_pair(row["teacher_support"])
        student_left, student_right = _parse_support_pair(row["student_support"])
        oracle_left, oracle_right = _parse_support_pair(row["oracle_support"])
        token_null_left, token_null_right = _parse_support_pair(row["token_position_null_support"])
        previous_teacher = previous_teacher_by_fold.get(fold)
        split = "train" if fold in TRAIN_FOLDS else "heldout"
        materialized.append(
            {
                "source": source_name,
                "seed_index": seed_index,
                "fold": fold,
                "split": split,
                "flat_position": flat_position,
                "position_fraction": _position_fraction(flat_position, rows),
                "position_parity": flat_position % 2,
                "previous_teacher_support_left": -1 if previous_teacher is None else previous_teacher[0],
                "previous_teacher_support_right": -1 if previous_teacher is None else previous_teacher[1],
                "teacher_support_left": teacher_left,
                "teacher_support_right": teacher_right,
                "student_support_left": student_left,
                "student_support_right": student_right,
                "oracle_support_left": oracle_left,
                "oracle_support_right": oracle_right,
                "token_position_null_support_left": token_null_left,
                "token_position_null_support_right": token_null_right,
                "teacher_student_exact_pair_match": row["teacher_student_exact_pair_match"],
                "target_token_eval_only": row["target_token"],
                "teacher_support_forced_into_student_loss": row[
                    "teacher_support_forced_into_student_loss"
                ],
                "student_router_support_loss": row["student_router_support_loss"],
                "oracle_best_support_for_student_loss": row[
                    "oracle_best_support_for_student_loss"
                ],
                "token_position_null_support_forced_into_student_loss": row[
                    "token_position_null_support_forced_into_student_loss"
                ],
            }
        )
        previous_teacher_by_fold[fold] = (teacher_left, teacher_right)
    return materialized


def _position_fraction(flat_position: int, rows: list[dict[str, str]]) -> float:
    max_position = max(int(row["flat_position"]) for row in rows)
    if max_position <= 0:
        return 0.0
    return round(flat_position / max_position, 8)


def _parse_support_pair(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"expected top-k2 support pair, got {value!r}")
    return int(parts[0]), int(parts[1])


def _source_inventory_row(
    source_name: str,
    path: Path,
    required: bool,
    row_count: int,
    status: str,
) -> dict[str, Any]:
    return {
        "source": source_name,
        "path": str(path),
        "required": required,
        "status": status,
        "row_count": row_count,
    }


def _field_provenance_rows() -> list[dict[str, Any]]:
    return [
        _field("seed_index", "source_id", True, False, False, False),
        _field("fold", "sequence_split_key", True, False, False, False),
        _field("flat_position", "prefix_safe_position_feature", True, False, False, False),
        _field("position_fraction", "prefix_safe_position_feature", True, False, False, False),
        _field("position_parity", "prefix_safe_position_feature", True, False, False, False),
        _field(
            "previous_teacher_support_left/right",
            "teacher_forced_past_support_summary",
            True,
            False,
            False,
            False,
        ),
        _field("teacher_support_left/right", "nondeployable_teacher_support_target", False, True, True, False),
        _field("student_support_left/right", "student_router_reference_label", False, False, False, False),
        _field("oracle_support_left/right", "nondeployable_oracle_eval_label", False, False, True, False),
        _field("target_token_eval_only", "evaluation_label_only", False, False, True, True),
        _field("loss_columns", "evaluation_metrics_only", False, False, False, True),
    ]


def _field(
    field: str,
    role: str,
    prefix_safe: bool,
    nondeployable_teacher_target: bool,
    future_or_target_leaking: bool,
    forbidden_predictor_input: bool,
) -> dict[str, Any]:
    return {
        "field": field,
        "role": role,
        "prefix_safe": prefix_safe,
        "nondeployable_teacher_target": nondeployable_teacher_target,
        "future_or_target_leaking": future_or_target_leaking,
        "forbidden_predictor_input": forbidden_predictor_input,
        "allowed_as_predictor_input": prefix_safe and not forbidden_predictor_input,
        "allowed_as_training_target": nondeployable_teacher_target,
    }


def _split_manifest_rows(dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in sorted({row["seed_index"] for row in dataset_rows}):
        seed_rows = [row for row in dataset_rows if row["seed_index"] == seed]
        for split in ("train", "heldout"):
            split_rows = [row for row in seed_rows if row["split"] == split]
            rows.append(
                {
                    "seed_index": seed,
                    "split": split,
                    "folds": ",".join(
                        str(fold) for fold in sorted({row["fold"] for row in split_rows})
                    ),
                    "row_count": len(split_rows),
                    "source": split_rows[0]["source"] if split_rows else "",
                }
            )
    return rows


def _has_train_and_heldout(split_rows: list[dict[str, Any]]) -> bool:
    train = sum(int(row["row_count"]) for row in split_rows if row["split"] == "train")
    heldout = sum(int(row["row_count"]) for row in split_rows if row["split"] == "heldout")
    return train > 0 and heldout > 0


def _missing_tensor_field_rows() -> list[dict[str, str]]:
    return [
        {
            "field": "current_hidden",
            "required_for": "strict hidden-feature Transformer-ACSR predictor",
            "reason": "not present in per_token_supports.csv",
        },
        {
            "field": "previous_hidden",
            "required_for": "strict hidden-feature Transformer-ACSR predictor",
            "reason": "not present in per_token_supports.csv",
        },
        {
            "field": "future_hidden",
            "required_for": "teacher future-chunk prediction target",
            "reason": "not present in per_token_supports.csv",
        },
        {
            "field": "future_delta",
            "required_for": "teacher future-delta prediction target",
            "reason": "not present in per_token_supports.csv",
        },
        {
            "field": "teacher_support_logits_or_soft_distribution",
            "required_for": "support KL training target",
            "reason": "only hard top-k2 teacher support pairs are present",
        },
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_artifacts(
    *,
    out_dir: Path,
    summary: dict[str, Any],
    dataset_rows: list[dict[str, Any]],
    source_inventory: list[dict[str, Any]],
    field_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    missing_tensor_rows: list[dict[str, str]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "dataset_rows.csv", dataset_rows)
    _write_csv(out_dir / "source_inventory.csv", source_inventory)
    _write_csv(out_dir / "field_provenance.csv", field_rows)
    _write_csv(out_dir / "split_manifest.csv", split_rows)
    _write_csv(out_dir / "missing_tensor_fields.csv", missing_tensor_rows)
    notes = [
        "# Transformer-ACSR Sequence Dataset",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Rows: `{summary['row_count']}`",
        f"- Train rows: `{summary['train_row_count']}`",
        f"- Heldout rows: `{summary['heldout_row_count']}`",
        f"- Support-only trainable now: `{summary['trainable_support_only_now']}`",
        f"- Hidden/future chunk targets available: `{summary['hidden_or_future_chunk_targets_available']}`",
        f"- Missing tensor fields: `{summary['missing_tensor_field_count']}`",
        f"- Next step: `{summary['selected_next_step']}`",
        "",
        "This packet is suitable only for a local support-target pregate. It does not",
        "contain hidden tensors, future chunks, or soft teacher logits, so it cannot",
        "support a full hidden-chunk Transformer-ACSR claim.",
        "RunPod/Colab validation remains blocked.",
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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_transformer_acsr_sequence_dataset(out_dir=args.out)
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
