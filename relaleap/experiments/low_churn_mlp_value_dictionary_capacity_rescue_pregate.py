"""Run a local pregate for richer low-churn value dictionaries."""

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

import numpy as np


DEFAULT_DESIGN_DIR = Path("results/reports/low_churn_mlp_value_dictionary_capacity_rescue_design")
DEFAULT_VECTOR_CAPTURE_DIR = Path("results/reports/low_churn_mlp_sparse_factorization_vector_capture")
DEFAULT_DECISION_AUDIT_DIR = Path("results/reports/low_churn_mlp_sparse_factorization_decision_audit")
DEFAULT_OUT_DIR = Path("results/reports/low_churn_mlp_value_dictionary_capacity_rescue_pregate")

NEXT_ACTION = "close_value_dictionary_capacity_rescue_or_request_strategy_review"
GPU_ACTION = "run_gpu_validation_after_value_dictionary_rescue_pregate"
REPAIR_ACTION = "repair_value_dictionary_capacity_rescue_pregate_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "candidate_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_low_churn_mlp_value_dictionary_capacity_rescue_pregate(
    *,
    design_dir: Path = DEFAULT_DESIGN_DIR,
    vector_capture_dir: Path = DEFAULT_VECTOR_CAPTURE_DIR,
    decision_audit_dir: Path = DEFAULT_DECISION_AUDIT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    dictionary_size: int = 8,
) -> dict[str, Any]:
    """Measure richer reusable value-dictionary ceilings on captured vectors."""

    start = time.time()
    design_summary = _read_json(design_dir / "summary.json")
    audit_summary = _read_json(decision_audit_dir / "summary.json")
    vector_rows = _read_csv(vector_capture_dir / "raw_teacher_residual_vectors.csv")
    parsed_rows = [_parse_vector_row(row) for row in vector_rows]
    source_rows = _source_rows(design_dir, vector_capture_dir, decision_audit_dir, design_summary, audit_summary, vector_rows)
    runtime_source_failure = any(not row["present"] or row["row_count"] <= 0 for row in source_rows)
    candidate_rows = [] if runtime_source_failure else _candidate_metrics(parsed_rows, dictionary_size)
    gate_rows = _gate_rows(design_summary, audit_summary, source_rows, parsed_rows, candidate_rows)
    runtime_failures = [row for row in gate_rows if not row["passed"] and row["gate_type"] == "runtime"]
    advancement_failures = [row for row in gate_rows if not row["passed"] and row["gate_type"] == "scientific_advancement"]
    status = "pass" if not runtime_failures else "fail"
    advance_to_gpu = status == "pass" and not advancement_failures
    selected_next_action = REPAIR_ACTION if status != "pass" else (GPU_ACTION if advance_to_gpu else NEXT_ACTION)
    best_sparse = _best(candidate_rows, family="sparse_oracle")
    best_control = _best(candidate_rows, family="capacity_control")
    best_null = _best(candidate_rows, family="null")
    best_valid_null = _best([row for row in candidate_rows if row.get("valid_null_for_target_access")], family="null")
    valid_null_delta = _r2_delta(best_sparse, best_valid_null)
    summary = {
        "status": status,
        "decision": (
            "low_churn_mlp_value_dictionary_capacity_rescue_pregate_recorded"
            if status == "pass"
            else "low_churn_mlp_value_dictionary_capacity_rescue_pregate_failed_closed"
        ),
        "claim_status": (
            "value_dictionary_capacity_rescue_local_gates_passed_gpu_still_requires_validation"
            if advance_to_gpu
            else "value_dictionary_capacity_rescue_local_gates_block_gpu"
        ),
        "selected_next_action": selected_next_action,
        "selected_next_step": _selected_next_step(selected_next_action),
        "requires_gpu_now": advance_to_gpu,
        "promotion_allowed": advance_to_gpu,
        "advance_to_gpu_validation": advance_to_gpu,
        "backend_policy": "local pregate only; RunPod/Colab remain blocked unless richer reusable sparse dictionary clears all local controls",
        "design_dir": str(design_dir),
        "vector_capture_dir": str(vector_capture_dir),
        "decision_audit_dir": str(decision_audit_dir),
        "out_dir": str(out_dir),
        "dictionary_size": dictionary_size,
        "source_rows": source_rows,
        "candidate_metrics": candidate_rows,
        "gate_criteria": gate_rows,
        "runtime_failures": runtime_failures,
        "advancement_failures": advancement_failures,
        "best_sparse_oracle": best_sparse,
        "best_capacity_control": best_control,
        "best_null": best_null,
        "best_valid_null": best_valid_null,
        "best_sparse_oracle_r2": best_sparse.get("heldout_reconstruction_r2", ""),
        "best_capacity_control_r2": best_control.get("heldout_reconstruction_r2", ""),
        "best_null_r2": best_null.get("heldout_reconstruction_r2", ""),
        "valid_null_delta_r2": valid_null_delta,
        "target_access_warning": "sparse oracle and shuffled-null rows are nondeployable target-aware ceilings, not deployable routing evidence",
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, source_rows, candidate_rows, gate_rows)
    return summary


def _source_rows(
    design_dir: Path,
    vector_capture_dir: Path,
    decision_audit_dir: Path,
    design_summary: dict[str, Any],
    audit_summary: dict[str, Any],
    vector_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    return [
        _source(
            "value_dictionary_capacity_rescue_design",
            design_dir / "summary.json",
            1 if design_summary else 0,
            design_summary.get("status", ""),
            design_summary.get("selected_next_action", ""),
        ),
        _source(
            "sparse_factorization_decision_audit",
            decision_audit_dir / "summary.json",
            1 if audit_summary else 0,
            audit_summary.get("status", ""),
            audit_summary.get("selected_next_action", ""),
        ),
        _source(
            "raw_teacher_residual_vectors",
            vector_capture_dir / "raw_teacher_residual_vectors.csv",
            len(vector_rows),
            "read" if vector_rows else "missing_or_empty",
            "",
        ),
    ]


def _source(source: str, path: Path, row_count: int, status: str, selected_next_action: str) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": status,
        "selected_next_action": selected_next_action,
        "row_count": row_count,
    }


def _candidate_metrics(rows: list[dict[str, Any]], dictionary_size: int) -> list[dict[str, Any]]:
    train = [row for row in rows if row["split"] != "heldout" and row["vector"]]
    heldout = [row for row in rows if row["split"] == "heldout" and row["vector"]]
    if not train or not heldout:
        return []
    train_array = np.asarray([row["vector"] for row in train], dtype=float)
    heldout_array = np.asarray([row["vector"] for row in heldout], dtype=float)
    baseline_mse = _baseline_mse(heldout_array)
    rows_out: list[dict[str, Any]] = []

    single = _fit_dictionary(train_array, dictionary_size)
    rows_out.append(_dictionary_row("single_codebook_vector_centroid_baseline", "sparse_baseline", heldout_array, single, baseline_mse))

    first = _fit_dictionary(train_array, dictionary_size)
    train_first = _nearest_vectors(train_array, first)
    residual = _fit_dictionary(train_array - train_first, dictionary_size)
    pred_multi = _nearest_stagewise(heldout_array, first, residual)
    rows_out.append(_prediction_row("multi_codebook_residual_dictionary", "sparse_oracle", heldout_array, pred_multi, baseline_mse, dictionary_size * 2, "nondeployable oracle selects residual code using heldout target residual"))

    for rank in sorted({3, 7, 9, min(16, train_array.shape[1]), train_array.shape[1]}):
        pred = _low_rank_reconstruct(train_array, heldout_array, rank)
        rows_out.append(_prediction_row(f"low_rank_svd_rank{rank}", "capacity_control", heldout_array, pred, baseline_mse, rank, "nondeployable low-rank projection control"))

    conditioned = _conditioned_dictionary_predictions(train, heldout, dictionary_size)
    rows_out.append(_prediction_row("rule_conditioned_token_mod_dictionary", "sparse_oracle", heldout_array, conditioned, baseline_mse, dictionary_size, "oracle nearest code within token-mod stratum"))

    shuffled = np.roll(train_array, shift=1, axis=0)
    shuffled_first = _fit_dictionary(shuffled, dictionary_size)
    shuffled_residual = _fit_dictionary(shuffled - _nearest_vectors(shuffled, shuffled_first), dictionary_size)
    pred_shuffled = _nearest_stagewise(heldout_array, shuffled_first, shuffled_residual)
    rows_out.append(_prediction_row("shuffled_teacher_dictionary", "null", heldout_array, pred_shuffled, baseline_mse, dictionary_size * 2, "misaligned train teacher residual null"))

    scrambled = _route_scrambled_predictions(heldout, single)
    rows_out.append(_prediction_row("route_scrambled_dictionary", "null", heldout_array, scrambled, baseline_mse, dictionary_size, "same values with token-index scrambled support"))

    dense_mean = np.repeat(train_array.mean(axis=0, keepdims=True), len(heldout_array), axis=0)
    rows_out.append(_prediction_row("dense_mean_same_rows", "capacity_control", heldout_array, dense_mean, baseline_mse, 1, "mean-vector dense baseline"))
    _annotate_candidate_rows(rows_out, vector_dim=train_array.shape[1])
    return rows_out


def _dictionary_row(name: str, family: str, targets: np.ndarray, dictionary: np.ndarray, baseline_mse: float) -> dict[str, Any]:
    return _prediction_row(
        name,
        family,
        targets,
        _nearest_vectors(targets, dictionary),
        baseline_mse,
        len(dictionary),
        "nondeployable oracle nearest-code ceiling",
    )


def _prediction_row(
    candidate: str,
    family: str,
    targets: np.ndarray,
    predictions: np.ndarray,
    baseline_mse: float,
    active_value_count: int,
    leakage_note: str,
) -> dict[str, Any]:
    mse_values = ((targets - predictions) ** 2).mean(axis=1)
    mse = float(mse_values.mean())
    r2 = 1.0 - (mse / baseline_mse) if baseline_mse > 0 else 0.0
    counts = _assignment_counts(targets, predictions)
    return {
        "candidate": candidate,
        "family": family,
        "heldout_row_count": int(len(targets)),
        "heldout_reconstruction_mse": mse,
        "heldout_reconstruction_r2": r2,
        "active_value_count": int(active_value_count),
        "support_entropy": _entropy(counts),
        "support_load_max_fraction": max(counts.values()) / sum(counts.values()) if counts else 0.0,
        "beats_single_codebook_baseline": "",
        "value_fit_split": "",
        "support_source": "",
        "target_access_at_eval": "",
        "effective_dof": "",
        "budget_match_group": "",
        "valid_null_for_target_access": False,
        "candidate_role": "",
        "deployable": False,
        "leakage_note": leakage_note,
    }


def _annotate_candidate_rows(rows: list[dict[str, Any]], *, vector_dim: int) -> None:
    single = next((row for row in rows if row["candidate"] == "single_codebook_vector_centroid_baseline"), {})
    single_r2 = _float(single.get("heldout_reconstruction_r2")) or 0.0
    for row in rows:
        candidate = row["candidate"]
        active = int(row.get("active_value_count", 0) or 0)
        row["beats_single_codebook_baseline"] = float(row["heldout_reconstruction_r2"]) > single_r2
        row["value_fit_split"] = "train_only"
        row["support_source"] = "heldout_target_nearest_code"
        row["target_access_at_eval"] = "target_residual_vector"
        row["effective_dof"] = active * vector_dim
        row["budget_match_group"] = "sparse_value_budget"
        row["valid_null_for_target_access"] = False
        row["candidate_role"] = "candidate"
        if candidate.startswith("low_rank_svd_rank"):
            rank = int(candidate.rsplit("rank", 1)[1])
            row["support_source"] = "heldout_target_projection"
            row["effective_dof"] = rank * vector_dim
            row["budget_match_group"] = "budget_matched_low_rank" if rank < vector_dim else "full_rank_ceiling"
            row["candidate_role"] = "capacity_ceiling" if rank >= vector_dim else "capacity_control"
        elif candidate == "dense_mean_same_rows":
            row["support_source"] = "train_mean_no_support"
            row["target_access_at_eval"] = "none"
            row["effective_dof"] = vector_dim
            row["budget_match_group"] = "mean_baseline"
            row["candidate_role"] = "capacity_control"
        elif candidate == "route_scrambled_dictionary":
            row["support_source"] = "token_index_scrambled_support"
            row["target_access_at_eval"] = "none"
            row["valid_null_for_target_access"] = False
            row["candidate_role"] = "route_null"
        elif candidate == "shuffled_teacher_dictionary":
            row["valid_null_for_target_access"] = True
            row["candidate_role"] = "target_aware_misaligned_null"
        elif candidate == "single_codebook_vector_centroid_baseline":
            row["candidate_role"] = "current_sparse_baseline"


def _gate_rows(
    design_summary: dict[str, Any],
    audit_summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    best_sparse = _best(candidate_rows, family="sparse_oracle")
    best_control = _best(candidate_rows, family="capacity_control")
    best_null = _best(candidate_rows, family="null")
    single = next((row for row in candidate_rows if row["candidate"] == "single_codebook_vector_centroid_baseline"), {})
    sparse_r2 = _float(best_sparse.get("heldout_reconstruction_r2"))
    control_r2 = _float(best_control.get("heldout_reconstruction_r2"))
    null_r2 = _float(best_null.get("heldout_reconstruction_r2"))
    single_r2 = _float(single.get("heldout_reconstruction_r2"))
    sparse_load = _float(best_sparse.get("support_load_max_fraction"))
    return [
        _criterion(
            "design_selected_pregate",
            design_summary.get("status") == "pass"
            and design_summary.get("selected_next_action") == "implement_value_dictionary_capacity_rescue_local_pregate",
            design_summary.get("selected_next_action", "missing"),
            "value-dictionary design must select local pregate implementation",
            "runtime",
        ),
        _criterion(
            "decision_audit_blocks_gpu_before_pregate",
            audit_summary.get("status") == "pass" and audit_summary.get("advance_to_gpu_validation") is False,
            audit_summary.get("advance_to_gpu_validation", "missing"),
            "decision audit must be present and keep GPU blocked before this pregate",
            "runtime",
        ),
        _criterion(
            "required_sources_present",
            all(row["present"] and row["row_count"] > 0 for row in source_rows),
            source_rows,
            "design, decision audit, and captured raw teacher vectors must exist and be nonempty",
            "runtime",
        ),
        _criterion(
            "heldout_vectors_available",
            any(row["split"] == "heldout" for row in parsed_rows) and any(row["split"] != "heldout" for row in parsed_rows),
            {"row_count": len(parsed_rows), "heldout": sum(1 for row in parsed_rows if row["split"] == "heldout")},
            "train and heldout teacher-vector rows are required",
            "runtime",
        ),
        _criterion(
            "richer_sparse_oracle_min_r2",
            sparse_r2 is not None and sparse_r2 >= 0.65,
            sparse_r2,
            "best richer reusable sparse oracle must reach heldout R2 >= 0.65",
            "scientific_advancement",
        ),
        _criterion(
            "richer_sparse_improves_single_codebook",
            sparse_r2 is not None and single_r2 is not None and sparse_r2 >= single_r2 + 0.15,
            {"best_sparse_r2": sparse_r2, "single_codebook_r2": single_r2},
            "richer sparse oracle must improve over the current single-codebook ceiling by >= 0.15 R2",
            "scientific_advancement",
        ),
        _criterion(
            "dense_low_rank_control_not_dominant",
            sparse_r2 is not None and control_r2 is not None and control_r2 <= sparse_r2 + 0.10,
            {"best_sparse_r2": sparse_r2, "best_capacity_control_r2": control_r2},
            "dense/low-rank controls must not beat best sparse oracle by > 0.10 R2",
            "scientific_advancement",
        ),
        _criterion(
            "shuffled_and_route_nulls_rejected",
            sparse_r2 is not None and null_r2 is not None and sparse_r2 >= null_r2 + 0.20,
            {"best_sparse_r2": sparse_r2, "best_null_r2": null_r2},
            "best sparse oracle must beat shuffled/route-scrambled nulls by >= 0.20 R2",
            "scientific_advancement",
        ),
        _criterion(
            "support_load_noncollapse",
            sparse_load is not None and sparse_load <= 0.50,
            {"best_sparse_support_load_max_fraction": sparse_load},
            "best sparse oracle support max-load fraction must be <= 0.50",
            "scientific_advancement",
        ),
    ]


def _criterion(criterion: str, passed: bool, actual: Any, threshold: str, gate_type: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "actual": actual,
        "threshold": threshold,
        "gate_type": gate_type,
        "failure_reason": "" if passed else threshold,
    }


def _fit_dictionary(vectors: np.ndarray, size: int) -> np.ndarray:
    k = max(1, min(int(size), len(vectors)))
    mean = vectors.mean(axis=0)
    centers = [vectors[int(np.argmin(((vectors - mean) ** 2).sum(axis=1)))].copy()]
    while len(centers) < k:
        current = np.asarray(centers)
        distances = ((vectors[:, None, :] - current[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        centers.append(vectors[int(np.argmax(distances))].copy())
    centers_array = np.asarray(centers, dtype=float)
    for _ in range(16):
        assignments = ((vectors[:, None, :] - centers_array[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        updated = centers_array.copy()
        for index in range(k):
            members = vectors[assignments == index]
            if len(members):
                updated[index] = members.mean(axis=0)
        if np.allclose(updated, centers_array):
            break
        centers_array = updated
    return centers_array


def _nearest_vectors(targets: np.ndarray, dictionary: np.ndarray) -> np.ndarray:
    assignments = ((targets[:, None, :] - dictionary[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    return dictionary[assignments]


def _nearest_stagewise(targets: np.ndarray, first: np.ndarray, residual: np.ndarray) -> np.ndarray:
    first_pred = _nearest_vectors(targets, first)
    residual_pred = _nearest_vectors(targets - first_pred, residual)
    return first_pred + residual_pred


def _low_rank_reconstruct(train: np.ndarray, heldout: np.ndarray, rank: int) -> np.ndarray:
    mean = train.mean(axis=0)
    centered = train - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[: max(1, min(rank, vt.shape[0]))]
    return mean + ((heldout - mean) @ basis.T) @ basis


def _conditioned_dictionary_predictions(train: list[dict[str, Any]], heldout: list[dict[str, Any]], dictionary_size: int) -> np.ndarray:
    fallback = _fit_dictionary(np.asarray([row["vector"] for row in train], dtype=float), dictionary_size)
    dictionaries: dict[int, np.ndarray] = {}
    for bucket in range(4):
        vectors = np.asarray([row["vector"] for row in train if row["token_index"] % 4 == bucket], dtype=float)
        dictionaries[bucket] = _fit_dictionary(vectors, max(1, dictionary_size // 2)) if len(vectors) else fallback
    predictions = []
    for row in heldout:
        target = np.asarray([row["vector"]], dtype=float)
        predictions.append(_nearest_vectors(target, dictionaries[row["token_index"] % 4])[0])
    return np.asarray(predictions)


def _route_scrambled_predictions(heldout: list[dict[str, Any]], dictionary: np.ndarray) -> np.ndarray:
    predictions = []
    for row in heldout:
        predictions.append(dictionary[(row["token_index"] * 3 + 1) % len(dictionary)])
    return np.asarray(predictions)


def _baseline_mse(vectors: np.ndarray) -> float:
    mean = vectors.mean(axis=0)
    return float(((vectors - mean) ** 2).mean(axis=1).mean())


def _assignment_counts(targets: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pred in predictions:
        key = json.dumps([round(float(value), 8) for value in pred.tolist()])
        counts[key] = counts.get(key, 0) + 1
    return counts


def _entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    return -sum((count / total) * math.log(count / total) for count in counts.values()) if total else 0.0


def _best(rows: list[dict[str, Any]], *, family: str) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("family") == family]
    if not candidates:
        return {}
    return max(candidates, key=lambda row: float(row.get("heldout_reconstruction_r2", -math.inf)))


def _parse_vector_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "teacher_row_id": row.get("teacher_row_id", ""),
        "split": row.get("split", ""),
        "token_index": _int(row.get("token_index"), 0),
        "vector": _json_list(row.get("teacher_residual_update_vector")),
    }


def _selected_next_step(action: str) -> str:
    if action == REPAIR_ACTION:
        return "repair missing design/audit/vector artifacts, then rerun the pregate"
    if action == GPU_ACTION:
        return "run bounded GPU validation only after local artifact checks confirm this pregate"
    return "close the current value-dictionary capacity rescue path or request strategy review before new architecture work"


def _r2_delta(left: dict[str, Any], right: dict[str, Any]) -> float | str:
    left_r2 = _float(left.get("heldout_reconstruction_r2"))
    right_r2 = _float(right.get("heldout_reconstruction_r2"))
    if left_r2 is None or right_r2 is None:
        return ""
    return left_r2 - right_r2


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", source_rows)
    _write_csv(out_dir / "candidate_metrics.csv", candidate_rows)
    _write_csv(out_dir / "gate_criteria.csv", gate_rows)
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"], lineterminator="\n")
        writer.writeheader()
        for row in rows or [{"status": "missing"}]:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in writer.fieldnames or []})


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Low-Churn MLP Value-Dictionary Capacity Rescue Pregate",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Claim status: `{summary['claim_status']}`",
            f"- Selected action: `{summary['selected_next_action']}`",
            f"- Best sparse oracle R2: `{summary['best_sparse_oracle_r2']}`",
            f"- Best capacity-control R2: `{summary['best_capacity_control_r2']}`",
            f"- Best null R2: `{summary['best_null_r2']}`",
            f"- Valid null delta R2: `{summary['valid_null_delta_r2']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            "",
            "All sparse candidates here are local ceilings, not deployable claims, because their support/code choices can inspect heldout target residual vectors. GPU validation remains blocked unless every local scientific advancement gate passes.",
            "",
        ]
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _json_list(value: Any) -> list[float]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return [float(item) for item in parsed] if isinstance(parsed, list) else []


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN_DIR)
    parser.add_argument("--vector-capture-dir", type=Path, default=DEFAULT_VECTOR_CAPTURE_DIR)
    parser.add_argument("--decision-audit-dir", type=Path, default=DEFAULT_DECISION_AUDIT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dictionary-size", type=int, default=8)
    args = parser.parse_args(argv)
    summary = run_low_churn_mlp_value_dictionary_capacity_rescue_pregate(
        design_dir=args.design_dir,
        vector_capture_dir=args.vector_capture_dir,
        decision_audit_dir=args.decision_audit_dir,
        out_dir=args.out,
        dictionary_size=args.dictionary_size,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "selected_next_action": summary["selected_next_action"],
                "best_sparse_oracle_r2": summary["best_sparse_oracle_r2"],
                "advance_to_gpu_validation": summary["advance_to_gpu_validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
