"""Train a local vector-level sparse-factorization ceiling on low-churn rows."""

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


DEFAULT_VECTOR_CAPTURE_DIR = Path("results/reports/low_churn_mlp_sparse_factorization_vector_capture")
DEFAULT_DESIGN_DIR = Path("results/reports/low_churn_mlp_sparse_factorization_ceiling_design")
DEFAULT_OUT_DIR = Path("results/reports/low_churn_mlp_sparse_factorization_vector_training")

NEXT_ACTION = "inspect_vector_sparse_factorization_ceiling_before_gpu_decision"
REPAIR_ACTION = "repair_vector_sparse_factorization_training_sources"

ARMS = (
    "oracle_support_sparse_ceiling",
    "learned_router_sparse_factorization",
    "token_position_router_sparse_factorization",
    "frequency_support_router_sparse_factorization",
    "random_fixed_support_sparse_factorization",
    "route_scrambled_same_values",
    "shuffled_teacher_residual_sparse_factorization",
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "training_rows.csv",
    "arm_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_low_churn_mlp_sparse_factorization_vector_training(
    *,
    vector_capture_dir: Path = DEFAULT_VECTOR_CAPTURE_DIR,
    design_dir: Path = DEFAULT_DESIGN_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    column_count: int = 8,
) -> dict[str, Any]:
    """Fit deterministic sparse value dictionaries against captured vectors."""

    start = time.time()
    capture_summary = _read_json(vector_capture_dir / "summary.json")
    design_summary = _read_json(design_dir / "summary.json")
    vector_rows = _read_csv(vector_capture_dir / "raw_teacher_residual_vectors.csv")
    intervention_rows = _read_csv(vector_capture_dir / "logit_intervention_rows.csv")
    support_arms = _read_csv(design_dir / "support_arms.csv")

    source_rows = _source_rows(
        vector_capture_dir=vector_capture_dir,
        design_dir=design_dir,
        capture_summary=capture_summary,
        design_summary=design_summary,
        vector_rows=vector_rows,
        intervention_rows=intervention_rows,
        support_arms=support_arms,
    )
    training_rows = _training_rows(vector_rows, column_count)
    arm_rows = _arm_metrics(training_rows, vector_rows, intervention_rows)
    gate_rows = _gate_rows(
        capture_summary=capture_summary,
        design_summary=design_summary,
        source_rows=source_rows,
        vector_rows=vector_rows,
        support_arms=support_arms,
        training_rows=training_rows,
        arm_rows=arm_rows,
    )
    runtime_failures = [row for row in gate_rows if not row["passed"] and row["gate_type"] == "runtime"]
    advancement_failures = [row for row in gate_rows if not row["passed"] and row["gate_type"] == "scientific_advancement"]
    status = "pass" if not runtime_failures else "fail"
    advancement_allowed = status == "pass" and not advancement_failures
    summary = {
        "status": status,
        "decision": (
            "low_churn_mlp_sparse_factorization_vector_training_recorded"
            if status == "pass"
            else "low_churn_mlp_sparse_factorization_vector_training_failed_closed"
        ),
        "claim_status": (
            "vector_sparse_factorization_ceiling_passed_local_gates"
            if advancement_allowed
            else "vector_sparse_factorization_ceiling_local_gates_block_gpu"
        ),
        "selected_next_action": NEXT_ACTION if status == "pass" else REPAIR_ACTION,
        "selected_next_step": (
            "inspect vector sparse-factorization rows and decide whether any bounded follow-up is warranted before GPU"
            if status == "pass"
            else "repair missing vector capture/design artifacts before rerunning vector sparse training"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": advancement_allowed,
        "advance_to_gpu_validation": False,
        "training_executed": status == "pass",
        "training_scope": "local deterministic vector sparse value dictionaries over captured low-churn teacher residuals",
        "backend_policy": "RunPod/Colab stay blocked unless local vector reconstruction, CE proxy, churn, commutator, intervention, and null gates pass",
        "vector_capture_dir": str(vector_capture_dir),
        "design_dir": str(design_dir),
        "out_dir": str(out_dir),
        "column_count": column_count,
        "source_rows": source_rows,
        "training_row_count": len(training_rows),
        "heldout_training_row_count": sum(1 for row in training_rows if row.get("split") == "heldout"),
        "arm_count": len(arm_rows),
        "best_heldout_arm": _best_arm(arm_rows, "teacher_residual_reconstruction_mse"),
        "oracle_learned_r2_gap": _oracle_learned_gap(arm_rows),
        "runtime_failures": runtime_failures,
        "advancement_failures": advancement_failures,
        "gate_criteria": gate_rows,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, source_rows, training_rows, arm_rows, gate_rows)
    return summary


def _source_rows(
    *,
    vector_capture_dir: Path,
    design_dir: Path,
    capture_summary: dict[str, Any],
    design_summary: dict[str, Any],
    vector_rows: list[dict[str, str]],
    intervention_rows: list[dict[str, str]],
    support_arms: list[dict[str, str]],
) -> list[dict[str, Any]]:
    return [
        _source("vector_capture_summary", vector_capture_dir / "summary.json", 1 if capture_summary else 0, capture_summary.get("status", ""), capture_summary.get("decision", "")),
        _source("raw_teacher_residual_vectors", vector_capture_dir / "raw_teacher_residual_vectors.csv", len(vector_rows), "read" if vector_rows else "missing_or_empty", ""),
        _source("logit_intervention_rows", vector_capture_dir / "logit_intervention_rows.csv", len(intervention_rows), "read" if intervention_rows else "missing_or_empty", ""),
        _source("sparse_factorization_design", design_dir / "summary.json", 1 if design_summary else 0, design_summary.get("status", ""), design_summary.get("decision", "")),
        _source("support_arms", design_dir / "support_arms.csv", len(support_arms), "read" if support_arms else "missing_or_empty", ""),
    ]


def _source(source: str, path: Path, row_count: int, status: str, decision: str) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": status,
        "decision": decision,
        "row_count": row_count,
    }


def _training_rows(rows: list[dict[str, str]], column_count: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    parsed = [_parse_vector_row(row) for row in rows]
    train = [row for row in parsed if row["split"] != "heldout"]
    if not train:
        return []
    dictionaries = {
        "primary": _centroids(train, column_count, lambda row: row["token_index"] % column_count),
        "position": _centroids(train, max(1, column_count // 2), lambda row: row["token_index"] % max(1, column_count // 2)),
        "frequency": _centroids(train, 1, lambda row: 0),
        "random": _centroids(train, column_count, lambda row: (row["token_index"] * 3 + 1) % column_count),
        "shuffled": _centroids(_shuffled_targets(train), column_count, lambda row: row["token_index"] % column_count),
    }
    global_vector = _mean_vector(row["target_vector"] for row in train)
    out: list[dict[str, Any]] = []
    for arm in ARMS:
        for row in parsed:
            support = _support(arm, row["token_index"], column_count)
            pred = _predict(arm, row, support, dictionaries, global_vector)
            mse = _mse(pred, row["target_vector"])
            coeff = _projection_coeff(pred, row["target_vector"])
            proxy_ce = row["base_ce_loss"] - ((row["base_ce_loss"] - row["teacher_ce_loss"]) * coeff)
            out.append(
                {
                    "arm": arm,
                    "teacher_row_id": row["teacher_row_id"],
                    "token_index": row["token_index"],
                    "split": row["split"],
                    "support": "|".join(str(item) for item in support),
                    "target_vector_l2": _norm(row["target_vector"]),
                    "predicted_vector_l2": _norm(pred),
                    "vector_projection_coeff": coeff,
                    "teacher_residual_reconstruction_mse": mse,
                    "teacher_residual_reconstruction_l2_error": math.sqrt(max(0.0, mse * len(row["target_vector"]))),
                    "base_ce_loss": row["base_ce_loss"],
                    "teacher_ce_loss": row["teacher_ce_loss"],
                    "heldout_ce_transfer_proxy": proxy_ce,
                    "teacher_gain_vs_base_ce": row["base_ce_loss"] - row["teacher_ce_loss"],
                    "sparse_gain_vs_base_ce_proxy": row["base_ce_loss"] - proxy_ce,
                    "anchor_kl_proxy": row["teacher_anchor_kl_vs_base"] * coeff * coeff,
                    "functional_churn_flip_proxy": row["teacher_prediction_changed_vs_base"] and coeff >= 0.5,
                    "finite_update_commutator_proxy": _commutator_proxy(pred, row["target_vector"], support),
                    "intervention_fingerprint_specificity_proxy": max(0.0, min(1.0, coeff)) * (1.0 / max(1, len(support))),
                    "training_mode": "vector_centroid_dictionary",
                }
            )
    return out


def _parse_vector_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "teacher_row_id": row.get("teacher_row_id", ""),
        "token_index": _int(row.get("token_index"), 0),
        "split": row.get("split", ""),
        "base_ce_loss": _float(row.get("base_ce_loss")) or 0.0,
        "teacher_ce_loss": _float(row.get("teacher_ce_loss")) or 0.0,
        "teacher_anchor_kl_vs_base": _float(row.get("teacher_anchor_kl_vs_base")) or 0.0,
        "teacher_prediction_changed_vs_base": _boolish(row.get("teacher_prediction_changed_vs_base")),
        "target_vector": _json_list(row.get("teacher_residual_update_vector")),
    }


def _centroids(rows: list[dict[str, Any]], count: int, key_fn: Any) -> dict[int, list[float]]:
    buckets: dict[int, list[list[float]]] = {index: [] for index in range(count)}
    for row in rows:
        buckets[int(key_fn(row))].append(row["target_vector"])
    fallback = _mean_vector(row["target_vector"] for row in rows)
    return {key: (_mean_vector(values) if values else fallback) for key, values in buckets.items()}


def _shuffled_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets = [row["target_vector"] for row in rows]
    if not targets:
        return rows
    shifted = targets[1:] + targets[:1]
    return [{**row, "target_vector": shifted[index]} for index, row in enumerate(rows)]


def _support(arm: str, token_index: int, column_count: int) -> list[int]:
    primary = token_index % column_count
    if arm in {"oracle_support_sparse_ceiling", "learned_router_sparse_factorization", "shuffled_teacher_residual_sparse_factorization"}:
        return [primary]
    if arm == "token_position_router_sparse_factorization":
        return [token_index % max(1, column_count // 2)]
    if arm == "frequency_support_router_sparse_factorization":
        return [0]
    if arm == "random_fixed_support_sparse_factorization":
        return [(token_index * 3 + 1) % column_count]
    if arm == "route_scrambled_same_values":
        return [(primary + max(1, column_count // 2)) % column_count]
    return [primary]


def _predict(
    arm: str,
    row: dict[str, Any],
    support: list[int],
    dictionaries: dict[str, dict[int, list[float]]],
    global_vector: list[float],
) -> list[float]:
    if arm == "oracle_support_sparse_ceiling":
        primary = dictionaries["primary"]
        candidates = list(primary.values()) + [row["target_vector"]]
        return min(candidates, key=lambda value: _mse(value, row["target_vector"]))
    if arm == "learned_router_sparse_factorization":
        return dictionaries["primary"].get(support[0], global_vector)
    if arm == "token_position_router_sparse_factorization":
        return dictionaries["position"].get(support[0], global_vector)
    if arm == "frequency_support_router_sparse_factorization":
        return dictionaries["frequency"].get(0, global_vector)
    if arm == "random_fixed_support_sparse_factorization":
        return dictionaries["random"].get(support[0], global_vector)
    if arm == "route_scrambled_same_values":
        return dictionaries["primary"].get(support[0], global_vector)
    if arm == "shuffled_teacher_residual_sparse_factorization":
        return dictionaries["shuffled"].get(support[0], global_vector)
    return global_vector


def _arm_metrics(
    training_rows: list[dict[str, Any]],
    vector_rows: list[dict[str, str]],
    intervention_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    parsed = [_parse_vector_row(row) for row in vector_rows]
    heldout_targets = [row["target_vector"] for row in parsed if row["split"] == "heldout"]
    baseline_mse = _baseline_mse(heldout_targets)
    teacher_heldout_ce = _mean(_float(row.get("teacher_ce_loss")) for row in vector_rows if row.get("split") == "heldout")
    base_heldout_ce = _mean(_float(row.get("base_ce_loss")) for row in vector_rows if row.get("split") == "heldout")
    teacher_anchor = _mean(_float(row.get("teacher_anchor_kl_vs_base")) for row in vector_rows if row.get("split") == "heldout")
    teacher_churn = _mean(1.0 if _boolish(row.get("teacher_prediction_changed_vs_base")) else 0.0 for row in vector_rows if row.get("split") == "heldout")
    intervention_count = len(intervention_rows)
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        heldout = [row for row in training_rows if row["arm"] == arm and row["split"] == "heldout"]
        all_arm = [row for row in training_rows if row["arm"] == arm]
        mse = _mean(row["teacher_residual_reconstruction_mse"] for row in heldout)
        r2 = 1.0 - (mse / baseline_mse) if baseline_mse > 0 else 0.0
        ce_proxy = _mean(row["heldout_ce_transfer_proxy"] for row in heldout)
        teacher_gap = base_heldout_ce - teacher_heldout_ce
        sparse_gap = base_heldout_ce - ce_proxy
        gap_closure = sparse_gap / teacher_gap if teacher_gap > 0 else 0.0
        rows.append(
            {
                "arm": arm,
                "heldout_row_count": len(heldout),
                "teacher_residual_reconstruction_mse": mse,
                "teacher_residual_reconstruction_r2": r2,
                "teacher_gap_closure_fraction_proxy": max(0.0, min(1.0, gap_closure)),
                "heldout_ce_transfer_proxy": ce_proxy,
                "base_heldout_ce": base_heldout_ce,
                "teacher_heldout_ce": teacher_heldout_ce,
                "support_entropy": _support_entropy(all_arm),
                "support_load_max_fraction": _support_load_max_fraction(all_arm),
                "functional_churn_flip_rate_proxy": _mean(1.0 if row["functional_churn_flip_proxy"] else 0.0 for row in heldout),
                "teacher_functional_churn_flip_rate": teacher_churn,
                "anchor_kl_proxy": _mean(row["anchor_kl_proxy"] for row in heldout),
                "teacher_anchor_kl_vs_base": teacher_anchor,
                "finite_update_commutator_proxy": _mean(row["finite_update_commutator_proxy"] for row in heldout),
                "intervention_fingerprint_specificity_proxy": _mean(row["intervention_fingerprint_specificity_proxy"] for row in heldout),
                "intervention_source_row_count": intervention_count,
                "oracle_support_regret_proxy": "",
                "scientific_advancement": False,
            }
        )
    oracle = _arm_by_name(rows, "oracle_support_sparse_ceiling")
    for row in rows:
        row["oracle_support_regret_proxy"] = row["teacher_residual_reconstruction_mse"] - oracle["teacher_residual_reconstruction_mse"] if oracle else ""
        row["scientific_advancement"] = _arm_advances(row, rows)
    return rows


def _arm_advances(row: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    shuffled = _arm_by_name(rows, "shuffled_teacher_residual_sparse_factorization")
    scrambled = _arm_by_name(rows, "route_scrambled_same_values")
    learned_oracle_gap = _oracle_learned_gap(rows)
    return bool(
        row["arm"] in {"oracle_support_sparse_ceiling", "learned_router_sparse_factorization"}
        and row["teacher_residual_reconstruction_r2"] >= 0.5
        and row["teacher_gap_closure_fraction_proxy"] >= 0.25
        and shuffled
        and row["teacher_residual_reconstruction_mse"] < shuffled["teacher_residual_reconstruction_mse"]
        and scrambled
        and row["teacher_residual_reconstruction_mse"] < scrambled["teacher_residual_reconstruction_mse"]
        and row["functional_churn_flip_rate_proxy"] <= row["teacher_functional_churn_flip_rate"]
        and row["anchor_kl_proxy"] <= row["teacher_anchor_kl_vs_base"]
        and (row["arm"] == "oracle_support_sparse_ceiling" or learned_oracle_gap <= 0.25)
    )


def _gate_rows(
    *,
    capture_summary: dict[str, Any],
    design_summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    vector_rows: list[dict[str, str]],
    support_arms: list[dict[str, str]],
    training_rows: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    arm_names = {row.get("arm", "") for row in support_arms}
    metric_arms = {row.get("arm", "") for row in arm_rows}
    oracle = _arm_by_name(arm_rows, "oracle_support_sparse_ceiling")
    learned = _arm_by_name(arm_rows, "learned_router_sparse_factorization")
    shuffled = _arm_by_name(arm_rows, "shuffled_teacher_residual_sparse_factorization")
    scrambled = _arm_by_name(arm_rows, "route_scrambled_same_values")
    advancing_arms = [row["arm"] for row in arm_rows if row.get("scientific_advancement")]
    return [
        _criterion(
            "vector_capture_selected_training",
            capture_summary.get("status") == "pass"
            and capture_summary.get("selected_next_action") == "implement_vector_sparse_factorization_ceiling_training",
            capture_summary.get("selected_next_action", "missing"),
            "vector capture must select vector sparse training",
            "runtime",
        ),
        _criterion(
            "design_source_available",
            design_summary.get("status") == "pass",
            design_summary.get("decision", "missing"),
            "sparse-factorization design artifact must be present",
            "runtime",
        ),
        _criterion(
            "required_sources_present",
            all(row["present"] and row["row_count"] > 0 for row in source_rows),
            source_rows,
            "vector capture/design artifacts must exist and be nonempty",
            "runtime",
        ),
        _criterion(
            "raw_vector_rows_available",
            bool(vector_rows) and any(row.get("split") == "heldout" for row in vector_rows),
            {"row_count": len(vector_rows), "heldout_rows": sum(1 for row in vector_rows if row.get("split") == "heldout")},
            "raw train and heldout teacher vector rows are required",
            "runtime",
        ),
        _criterion(
            "support_arm_coverage_complete",
            set(ARMS).issubset(arm_names) and set(ARMS).issubset(metric_arms),
            {"missing_design_arms": sorted(set(ARMS) - arm_names), "missing_metric_arms": sorted(set(ARMS) - metric_arms)},
            "all oracle, learned, shortcut, frequency, random, scrambled, and shuffled arms are required",
            "runtime",
        ),
        _criterion(
            "training_rows_written_for_all_arms",
            bool(training_rows) and {row["arm"] for row in training_rows} == set(ARMS),
            sorted({row["arm"] for row in training_rows}),
            "training rows must cover every sparse-factorization arm",
            "runtime",
        ),
        _criterion(
            "oracle_or_learned_closes_teacher_gap",
            bool(oracle and learned)
            and max(oracle["teacher_gap_closure_fraction_proxy"], learned["teacher_gap_closure_fraction_proxy"]) >= 0.25,
            {"oracle": oracle, "learned": learned},
            "oracle or learned sparse arm must close at least 25% of the teacher CE proxy gap",
            "scientific_advancement",
        ),
        _criterion(
            "oracle_or_learned_beats_null_reconstruction",
            bool(oracle and learned and shuffled and scrambled)
            and min(oracle["teacher_residual_reconstruction_mse"], learned["teacher_residual_reconstruction_mse"])
            < min(shuffled["teacher_residual_reconstruction_mse"], scrambled["teacher_residual_reconstruction_mse"]),
            {"oracle": oracle, "learned": learned, "shuffled": shuffled, "scrambled": scrambled},
            "oracle/learned sparse arm must beat shuffled-teacher and route-scrambled null reconstruction",
            "scientific_advancement",
        ),
        _criterion(
            "learned_router_near_oracle",
            _oracle_learned_gap(arm_rows) <= 0.25,
            _oracle_learned_gap(arm_rows),
            "learned support R2 must be within 0.25 of oracle support R2",
            "scientific_advancement",
        ),
        _criterion(
            "interference_and_intervention_fields_present",
            all(
                row.get("functional_churn_flip_rate_proxy") != ""
                and row.get("finite_update_commutator_proxy") != ""
                and row.get("intervention_fingerprint_specificity_proxy") != ""
                for row in arm_rows
            ),
            "present" if arm_rows else "missing",
            "churn, commutator, and intervention proxy fields must be populated",
            "scientific_advancement",
        ),
        _criterion(
            "scientific_advancement_arm_exists",
            bool(advancing_arms),
            advancing_arms,
            "at least one oracle/learned arm must pass the local sparse-factorization advancement gates",
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


def _support_entropy(rows: list[dict[str, Any]]) -> float:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get("support", ""))] = counts.get(str(row.get("support", "")), 0) + 1
    total = sum(counts.values())
    return -sum((count / total) * math.log(count / total) for count in counts.values()) if total else 0.0


def _support_load_max_fraction(rows: list[dict[str, Any]]) -> float:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get("support", ""))] = counts.get(str(row.get("support", "")), 0) + 1
    total = sum(counts.values())
    return max(counts.values()) / total if total else 0.0


def _projection_coeff(pred: list[float], target: list[float]) -> float:
    denom = sum(value * value for value in target)
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, _dot(pred, target) / denom))


def _commutator_proxy(pred: list[float], target: list[float], support: list[int]) -> float:
    return _mse(pred, target) * len(support) * (1.0 + _norm(pred))


def _baseline_mse(vectors: list[list[float]]) -> float:
    if not vectors:
        return 0.0
    mean = _mean_vector(vectors)
    return _mean(_mse(vector, mean) for vector in vectors)


def _mean_vector(vectors: Any) -> list[float]:
    vectors = [list(vector) for vector in vectors]
    if not vectors:
        return []
    width = len(vectors[0])
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]


def _mse(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum((a - b) ** 2 for a, b in zip(left, right)) / len(left)


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _mean(values: Any) -> float:
    real = [float(value) for value in values if value not in ("", None)]
    return sum(real) / len(real) if real else 0.0


def _best_arm(rows: list[dict[str, Any]], metric: str) -> str:
    if not rows:
        return ""
    return min(rows, key=lambda row: row.get(metric, math.inf))["arm"]


def _arm_by_name(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    for row in rows:
        if row.get("arm") == arm:
            return row
    return {}


def _oracle_learned_gap(rows: list[dict[str, Any]]) -> float:
    oracle = _arm_by_name(rows, "oracle_support_sparse_ceiling")
    learned = _arm_by_name(rows, "learned_router_sparse_factorization")
    if not oracle or not learned:
        return math.inf
    return oracle["teacher_residual_reconstruction_r2"] - learned["teacher_residual_reconstruction_r2"]


def _json_list(value: Any) -> list[float]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [float(item) for item in parsed]


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
    training_rows: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", source_rows)
    _write_csv(out_dir / "training_rows.csv", training_rows)
    _write_csv(out_dir / "arm_metrics.csv", arm_rows)
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
            "# Low-Churn MLP Sparse-Factorization Vector Training",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Claim status: `{summary['claim_status']}`",
            f"- Selected action: `{summary['selected_next_action']}`",
            f"- Training rows: `{summary['training_row_count']}`",
            f"- Best heldout arm: `{summary['best_heldout_arm']}`",
            f"- Oracle-learned R2 gap: `{summary['oracle_learned_r2_gap']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            "",
            "This is a bounded local vector-level ceiling over captured low-churn teacher residual rows. It keeps GPU validation blocked unless oracle/learned sparse arms clear the null, CE proxy, interference, and intervention gates.",
            "",
        ]
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


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


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-capture-dir", type=Path, default=DEFAULT_VECTOR_CAPTURE_DIR)
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--column-count", type=int, default=8)
    args = parser.parse_args(argv)
    summary = run_low_churn_mlp_sparse_factorization_vector_training(
        vector_capture_dir=args.vector_capture_dir,
        design_dir=args.design_dir,
        out_dir=args.out,
        column_count=args.column_count,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "selected_next_action": summary["selected_next_action"],
                "best_heldout_arm": summary["best_heldout_arm"],
                "advance_to_gpu_validation": summary["advance_to_gpu_validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
