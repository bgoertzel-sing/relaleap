"""Materialize the Transformer-ACSR hidden/future sequence dataset.

This command consumes the opt-in hidden/future capture artifacts and the exact
same-student forced-support intervention rows. It is deliberately conservative:
future tensors, teacher support logits, target tokens, oracle labels, and loss
columns are preserved only as target/evaluation fields and are registered as
forbidden predictor inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_DIR = Path(
    "results/audits/token_larger_causal_contextual_router_distillation_agreement_hidden_future_capture"
)
DEFAULT_HIDDEN_FUTURE_ROWS = DEFAULT_SOURCE_DIR / "hidden_future_rows.csv"
DEFAULT_INTERVENTION_ROWS = DEFAULT_SOURCE_DIR / "intervention_rows_exact.csv"
DEFAULT_OUT_DIR = Path("results/reports/transformer_acsr_hidden_future_sequence_dataset")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "dataset_rows.csv",
    "loss_lookup.csv",
    "source_inventory.csv",
    "split_manifest.csv",
    "field_provenance.csv",
    "validation_failures.csv",
    "notes.md",
)

HIDDEN_REQUIRED_FIELDS = {
    "current_hidden_json",
    "flat_position",
    "fold",
    "forbidden_predictor_fields",
    "future_delta_json",
    "future_hidden_json",
    "future_targets_nondeployable",
    "oracle_support_eval_only",
    "position_index",
    "prefix_safe_fields",
    "previous_hidden_json",
    "sequence_id",
    "split",
    "student_router_support",
    "target_token_eval_only",
    "teacher_support_logits_json",
    "teacher_target_fields",
    "teacher_topk_support",
    "token_position_null_support",
}

INTERVENTION_REQUIRED_FIELDS = {
    "flat_position",
    "fold",
    "forced_minus_oracle_loss",
    "forced_minus_student_router_loss",
    "forced_support_loss",
    "forced_support_pair",
    "forced_support_pair_index",
    "is_oracle_support_pair",
    "is_student_router_support_pair",
    "is_teacher_support_pair",
    "oracle_support",
    "oracle_support_loss",
    "position_index",
    "row_family",
    "sequence_id",
    "split",
    "student_router_support",
    "student_router_support_loss",
    "teacher_support",
    "teacher_support_forced_loss",
}


def run_transformer_acsr_hidden_future_sequence_dataset(
    *,
    hidden_future_rows_path: Path = DEFAULT_HIDDEN_FUTURE_ROWS,
    intervention_rows_path: Path = DEFAULT_INTERVENTION_ROWS,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a leakage-labeled hidden/future dataset and exact loss lookup."""

    start = time.time()
    hidden_rows = _read_csv(hidden_future_rows_path)
    intervention_rows = _read_csv(intervention_rows_path)
    failures: list[dict[str, Any]] = []
    source_inventory = [
        _source_row("hidden_future_rows", hidden_future_rows_path, hidden_rows),
        _source_row("intervention_rows_exact", intervention_rows_path, intervention_rows),
    ]

    failures.extend(_schema_failures("hidden_future_rows", hidden_rows, HIDDEN_REQUIRED_FIELDS))
    failures.extend(
        _schema_failures("intervention_rows_exact", intervention_rows, INTERVENTION_REQUIRED_FIELDS)
    )

    dataset_rows: list[dict[str, Any]] = []
    loss_lookup: list[dict[str, Any]] = []
    pair_count_expected = 0
    hidden_dim = 0
    teacher_support_logit_dim = 0
    sequence_split_consistent = True
    exact_loss_lookup_complete = False
    leakage_contract_passes = False

    if not failures:
        dataset_rows, dataset_failures, hidden_dim, teacher_support_logit_dim = _dataset_rows(
            hidden_rows
        )
        failures.extend(dataset_failures)
        pair_count_expected = _expected_pair_count(teacher_support_logit_dim)
        loss_lookup, lookup_failures = _loss_lookup_rows(
            intervention_rows=intervention_rows,
            hidden_rows=hidden_rows,
            expected_pair_count=pair_count_expected,
        )
        failures.extend(lookup_failures)
        sequence_split_consistent = _sequence_split_consistent(dataset_rows)
        if not sequence_split_consistent:
            failures.append(
                {
                    "source": "hidden_future_rows",
                    "check": "sequence_split_consistency",
                    "reason": "at least one sequence_id appears in multiple splits",
                }
            )
        exact_loss_lookup_complete = not any(
            failure["check"].startswith("exact_pair_count")
            or failure["check"].startswith("hidden_intervention_key")
            for failure in failures
        )
        leakage_contract_passes = _leakage_contract_passes(dataset_rows)
        if not leakage_contract_passes:
            failures.append(
                {
                    "source": "hidden_future_rows",
                    "check": "leakage_contract",
                    "reason": "captured prefix/forbidden field labels do not match expected contract",
                }
            )

    split_rows = _split_manifest_rows(dataset_rows)
    train_sequence_count = sum(
        int(row["sequence_count"]) for row in split_rows if row["split"] == "train"
    )
    heldout_sequence_count = sum(
        int(row["sequence_count"]) for row in split_rows if row["split"] == "heldout"
    )
    train_row_count = sum(int(row["row_count"]) for row in split_rows if row["split"] == "train")
    heldout_row_count = sum(
        int(row["row_count"]) for row in split_rows if row["split"] == "heldout"
    )
    split_coverage_available = train_sequence_count > 0 and heldout_sequence_count > 0
    trainability_gate_passes = bool(
        not failures
        and split_coverage_available
        and exact_loss_lookup_complete
        and leakage_contract_passes
    )

    if failures:
        decision = "transformer_acsr_hidden_future_sequence_dataset_failed_closed"
        claim_status = "hidden_future_sequence_dataset_invalid_no_gpu"
        selected_next_step = "repair hidden/future capture schemas or exact intervention rows"
        status = "fail"
    elif trainability_gate_passes:
        decision = "transformer_acsr_hidden_future_sequence_dataset_trainable_locally"
        claim_status = "hidden_future_sequence_dataset_ready_for_cpu_pregate_no_gpu"
        selected_next_step = "train_tiny_prefix_safe_transformer_acsr_hidden_future_pregate_with_nulls"
        status = "pass"
    else:
        decision = "transformer_acsr_hidden_future_sequence_dataset_materialized_trainability_blocked"
        claim_status = "hidden_future_sequence_dataset_heldout_only_no_training_no_gpu"
        selected_next_step = "extend hidden/future capture to include sequence-level train and heldout splits"
        status = "pass"

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "hidden_future_row_count": len(dataset_rows),
        "exact_intervention_row_count": len(loss_lookup),
        "sequence_count": len({row["sequence_id"] for row in dataset_rows}),
        "train_sequence_count": train_sequence_count,
        "heldout_sequence_count": heldout_sequence_count,
        "train_row_count": train_row_count,
        "heldout_row_count": heldout_row_count,
        "split_coverage_available": split_coverage_available,
        "trainability_gate_passes": trainability_gate_passes,
        "trainability_status": "pass" if trainability_gate_passes else "fail_closed",
        "sequence_split_consistent": sequence_split_consistent,
        "hidden_dim": hidden_dim,
        "teacher_support_logit_dim": teacher_support_logit_dim,
        "expected_pair_count_per_position": pair_count_expected,
        "exact_loss_lookup_complete": exact_loss_lookup_complete,
        "leakage_contract_passes": leakage_contract_passes,
        "prefix_safe_feature_fields": [
            "current_hidden_json",
            "previous_hidden_json",
            "position_index",
        ],
        "forbidden_predictor_fields": [
            "future_hidden_json",
            "future_delta_json",
            "teacher_support_logits_json",
            "teacher_topk_support",
            "target_token_eval_only",
            "oracle_support_eval_only",
            "loss_lookup_fields",
        ],
        "failure_count": len(failures),
        "failures": failures,
        "backend_policy": "local dataset materialization only; RunPod and Colab remain blocked",
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(
        out_dir=out_dir,
        summary=summary,
        dataset_rows=dataset_rows,
        loss_lookup=loss_lookup,
        source_inventory=source_inventory,
        split_rows=split_rows,
        field_rows=_field_provenance_rows(),
        failures=failures,
    )
    return summary


def _dataset_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    materialized: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    hidden_dim = 0
    teacher_support_logit_dim = 0
    for index, row in enumerate(
        sorted(rows, key=lambda item: (item["sequence_id"], int(item["position_index"]))),
        start=1,
    ):
        current_hidden = _json_vector(row["current_hidden_json"])
        previous_hidden = _json_vector(row["previous_hidden_json"])
        future_hidden = _json_vector(row["future_hidden_json"])
        future_delta = _json_vector(row["future_delta_json"])
        teacher_logits = _json_vector(row["teacher_support_logits_json"])
        if index == 1:
            hidden_dim = len(current_hidden)
            teacher_support_logit_dim = len(teacher_logits)
        dims = {
            "current_hidden_json": len(current_hidden),
            "previous_hidden_json": len(previous_hidden),
            "future_hidden_json": len(future_hidden),
            "future_delta_json": len(future_delta),
        }
        if len(set(dims.values())) != 1:
            failures.append(
                {
                    "source": "hidden_future_rows",
                    "check": "hidden_dimension_consistency",
                    "sequence_id": row["sequence_id"],
                    "position_index": row["position_index"],
                    "reason": json.dumps(dims, sort_keys=True),
                }
            )
        if len(teacher_logits) != teacher_support_logit_dim:
            failures.append(
                {
                    "source": "hidden_future_rows",
                    "check": "teacher_logit_dimension_consistency",
                    "sequence_id": row["sequence_id"],
                    "position_index": row["position_index"],
                    "reason": f"expected {teacher_support_logit_dim}, got {len(teacher_logits)}",
                }
            )
        materialized.append(
            {
                "sequence_id": row["sequence_id"],
                "split": row["split"],
                "fold": int(row["fold"]),
                "batch_index": row.get("batch_index", ""),
                "flat_position": int(row["flat_position"]),
                "position_index": int(row["position_index"]),
                "current_hidden_json": row["current_hidden_json"],
                "previous_hidden_json": row["previous_hidden_json"],
                "future_hidden_json_target_only": row["future_hidden_json"],
                "future_delta_json_target_only": row["future_delta_json"],
                "teacher_support_logits_json_target_only": row["teacher_support_logits_json"],
                "teacher_topk_support_target_only": row["teacher_topk_support"],
                "student_router_support_eval_only": row["student_router_support"],
                "oracle_support_eval_only": row["oracle_support_eval_only"],
                "token_position_null_support_eval_only": row["token_position_null_support"],
                "target_token_eval_only": row["target_token_eval_only"],
                "prefix_safe_fields": row["prefix_safe_fields"],
                "forbidden_predictor_fields": row["forbidden_predictor_fields"],
                "teacher_target_fields": row["teacher_target_fields"],
                "future_targets_nondeployable": _parse_bool(row["future_targets_nondeployable"]),
                "hidden_dim": len(current_hidden),
                "teacher_support_logit_dim": len(teacher_logits),
            }
        )
    return materialized, failures, hidden_dim, teacher_support_logit_dim


def _loss_lookup_rows(
    *,
    intervention_rows: list[dict[str, str]],
    hidden_rows: list[dict[str, str]],
    expected_pair_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    hidden_keys = {
        (row["sequence_id"], int(row["flat_position"]), int(row["position_index"]))
        for row in hidden_rows
    }
    grouped: dict[tuple[str, int, int], list[dict[str, str]]] = {}
    for row in intervention_rows:
        key = (row["sequence_id"], int(row["flat_position"]), int(row["position_index"]))
        grouped.setdefault(key, []).append(row)
    if set(grouped) != hidden_keys:
        failures.append(
            {
                "source": "intervention_rows_exact",
                "check": "hidden_intervention_key_match",
                "reason": f"hidden_keys={len(hidden_keys)} intervention_keys={len(grouped)}",
            }
        )
    materialized: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        if len(rows) != expected_pair_count:
            failures.append(
                {
                    "source": "intervention_rows_exact",
                    "check": "exact_pair_count_per_position",
                    "sequence_id": key[0],
                    "flat_position": key[1],
                    "position_index": key[2],
                    "reason": f"expected {expected_pair_count}, got {len(rows)}",
                }
            )
        seen_pairs = {row["forced_support_pair"] for row in rows}
        if len(seen_pairs) != len(rows):
            failures.append(
                {
                    "source": "intervention_rows_exact",
                    "check": "exact_pair_count_unique_pairs",
                    "sequence_id": key[0],
                    "flat_position": key[1],
                    "position_index": key[2],
                    "reason": f"rows={len(rows)} unique_pairs={len(seen_pairs)}",
                }
            )
        for row in sorted(rows, key=lambda item: int(item["forced_support_pair_index"])):
            materialized.append(
                {
                    "sequence_id": row["sequence_id"],
                    "split": row["split"],
                    "fold": int(row["fold"]),
                    "flat_position": int(row["flat_position"]),
                    "position_index": int(row["position_index"]),
                    "forced_support_pair_index": int(row["forced_support_pair_index"]),
                    "forced_support_pair": row["forced_support_pair"],
                    "forced_support_loss": float(row["forced_support_loss"]),
                    "forced_minus_oracle_loss": float(row["forced_minus_oracle_loss"]),
                    "forced_minus_student_router_loss": float(
                        row["forced_minus_student_router_loss"]
                    ),
                    "is_teacher_support_pair": _parse_bool(row["is_teacher_support_pair"]),
                    "is_student_router_support_pair": _parse_bool(
                        row["is_student_router_support_pair"]
                    ),
                    "is_oracle_support_pair": _parse_bool(row["is_oracle_support_pair"]),
                    "teacher_support": row["teacher_support"],
                    "teacher_support_forced_loss": float(row["teacher_support_forced_loss"]),
                    "student_router_support": row["student_router_support"],
                    "student_router_support_loss": float(row["student_router_support_loss"]),
                    "oracle_support": row["oracle_support"],
                    "oracle_support_loss": float(row["oracle_support_loss"]),
                    "row_family": row["row_family"],
                }
            )
    return materialized, failures


def _field_provenance_rows() -> list[dict[str, Any]]:
    return [
        _field("sequence_id", "sequence_split_key", True, False, False),
        _field("fold", "sequence_split_key", True, False, False),
        _field("split", "sequence_split_label", True, False, False),
        _field("position_index", "prefix_safe_position_feature", True, False, False),
        _field("current_hidden_json", "prefix_safe_feature_tensor", True, False, False),
        _field("previous_hidden_json", "prefix_safe_feature_tensor", True, False, False),
        _field("future_hidden_json_target_only", "nondeployable_teacher_target", False, True, True),
        _field("future_delta_json_target_only", "nondeployable_teacher_target", False, True, True),
        _field(
            "teacher_support_logits_json_target_only",
            "nondeployable_teacher_target",
            False,
            True,
            True,
        ),
        _field("teacher_topk_support_target_only", "nondeployable_teacher_target", False, True, True),
        _field("target_token_eval_only", "evaluation_label_only", False, False, True),
        _field("oracle_support_eval_only", "oracle_evaluation_label", False, False, True),
        _field("loss_lookup", "same_student_intervention_eval_lookup", False, False, True),
    ]


def _field(
    field: str,
    role: str,
    prefix_safe: bool,
    nondeployable_teacher_target: bool,
    forbidden_predictor_input: bool,
) -> dict[str, Any]:
    return {
        "field": field,
        "role": role,
        "prefix_safe": prefix_safe,
        "nondeployable_teacher_target": nondeployable_teacher_target,
        "forbidden_predictor_input": forbidden_predictor_input,
        "allowed_as_predictor_input": prefix_safe and not forbidden_predictor_input,
        "allowed_as_training_target": nondeployable_teacher_target,
    }


def _split_manifest_rows(dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "heldout"):
        split_rows = [row for row in dataset_rows if row["split"] == split]
        rows.append(
            {
                "split": split,
                "sequence_count": len({row["sequence_id"] for row in split_rows}),
                "row_count": len(split_rows),
                "folds": ",".join(str(fold) for fold in sorted({row["fold"] for row in split_rows})),
                "trainability_role": (
                    "required_for_predictor_training"
                    if split == "train"
                    else "required_for_final_eval"
                ),
            }
        )
    return rows


def _schema_failures(
    source: str,
    rows: list[dict[str, str]],
    required_fields: set[str],
) -> list[dict[str, Any]]:
    if not rows:
        return [{"source": source, "check": "source_nonempty", "reason": "no rows available"}]
    missing = sorted(required_fields.difference(rows[0].keys()))
    if not missing:
        return []
    return [
        {
            "source": source,
            "check": "required_columns",
            "reason": ",".join(missing),
        }
    ]


def _source_row(source: str, path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "row_count": len(rows),
        "status": "available" if rows else "missing_or_empty",
    }


def _sequence_split_consistent(dataset_rows: list[dict[str, Any]]) -> bool:
    split_by_sequence: dict[str, set[str]] = {}
    for row in dataset_rows:
        split_by_sequence.setdefault(row["sequence_id"], set()).add(row["split"])
    return all(len(splits) == 1 for splits in split_by_sequence.values())


def _leakage_contract_passes(dataset_rows: list[dict[str, Any]]) -> bool:
    if not dataset_rows:
        return False
    required_prefix = {"current_hidden_json", "previous_hidden_json", "position_index"}
    required_forbidden = {
        "future_hidden_json",
        "future_delta_json",
        "teacher_support_logits_json",
        "teacher_topk_support",
        "target_token_eval_only",
        "oracle_support_eval_only",
    }
    for row in dataset_rows:
        prefix = _semicolon_set(str(row["prefix_safe_fields"]))
        forbidden = _semicolon_set(str(row["forbidden_predictor_fields"]))
        teacher_targets = _semicolon_set(str(row["teacher_target_fields"]))
        if not required_prefix.issubset(prefix):
            return False
        if not required_forbidden.issubset(forbidden):
            return False
        if not {
            "future_hidden_json",
            "future_delta_json",
            "teacher_support_logits_json",
            "teacher_topk_support",
        }.issubset(teacher_targets):
            return False
        if not row["future_targets_nondeployable"]:
            return False
    return True


def _expected_pair_count(column_count: int) -> int:
    if column_count < 2:
        return 0
    return math.comb(column_count, 2)


def _json_vector(value: str) -> list[float]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("expected JSON vector")
    return [float(item) for item in parsed]


def _semicolon_set(value: str) -> set[str]:
    return {part.strip() for part in value.split(";") if part.strip()}


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() == "true"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_artifacts(
    *,
    out_dir: Path,
    summary: dict[str, Any],
    dataset_rows: list[dict[str, Any]],
    loss_lookup: list[dict[str, Any]],
    source_inventory: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    field_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "dataset_rows.csv", dataset_rows)
    _write_csv(out_dir / "loss_lookup.csv", loss_lookup)
    _write_csv(out_dir / "source_inventory.csv", source_inventory)
    _write_csv(out_dir / "split_manifest.csv", split_rows)
    _write_csv(out_dir / "field_provenance.csv", field_rows)
    _write_csv(out_dir / "validation_failures.csv", failures)
    notes = [
        "# Transformer-ACSR Hidden/Future Sequence Dataset",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Hidden/future rows: `{summary['hidden_future_row_count']}`",
        f"- Exact intervention rows: `{summary['exact_intervention_row_count']}`",
        f"- Expected pair count per position: `{summary['expected_pair_count_per_position']}`",
        f"- Train rows: `{summary['train_row_count']}`",
        f"- Heldout rows: `{summary['heldout_row_count']}`",
        f"- Trainability gate: `{summary['trainability_status']}`",
        f"- Leakage contract passes: `{summary['leakage_contract_passes']}`",
        f"- Exact loss lookup complete: `{summary['exact_loss_lookup_complete']}`",
        f"- Next step: `{summary['selected_next_step']}`",
        "",
        "Future hidden, future delta, teacher logits/support, target tokens, oracle",
        "labels, and exact loss lookup fields are target/evaluation-only and must",
        "not be used as predictor inputs.",
        "",
        "RunPod and Colab validation remain blocked.",
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
    parser.add_argument("--hidden-future-rows", type=Path, default=DEFAULT_HIDDEN_FUTURE_ROWS)
    parser.add_argument("--intervention-rows", type=Path, default=DEFAULT_INTERVENTION_ROWS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_transformer_acsr_hidden_future_sequence_dataset(
        hidden_future_rows_path=args.hidden_future_rows,
        intervention_rows_path=args.intervention_rows,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "trainability_gate_passes": summary["trainability_gate_passes"],
                "selected_next_step": summary["selected_next_step"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
