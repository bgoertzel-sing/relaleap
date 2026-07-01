"""Audit low-churn MLP sparse-factorization evidence before GPU validation."""

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


DEFAULT_VECTOR_TRAINING_DIR = Path("results/reports/low_churn_mlp_sparse_factorization_vector_training")
DEFAULT_VECTOR_CAPTURE_DIR = Path("results/reports/low_churn_mlp_sparse_factorization_vector_capture")
DEFAULT_OUT_DIR = Path("results/reports/low_churn_mlp_sparse_factorization_decision_audit")

NEXT_ACTION = "redesign_value_dictionary_or_close_sparse_factorization_ceiling"
ROUTER_REDESIGN_ACTION = "redesign_prefix_safe_support_router_before_gpu"
REPAIR_ACTION = "repair_sparse_factorization_decision_audit_sources"

EXACT_ORACLE_ARM = "oracle_support_sparse_ceiling"
LEARNED_ARM = "learned_router_sparse_factorization"
GLOBAL_ORACLE_ARM = "global_dictionary_oracle_support_sparse_factorization"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "decision_rows.csv",
    "blame_rows.csv",
    "global_dictionary_metrics.csv",
    "covariance_spectrum.csv",
    "notes.md",
)


def run_low_churn_mlp_sparse_factorization_decision_audit(
    *,
    vector_training_dir: Path = DEFAULT_VECTOR_TRAINING_DIR,
    vector_capture_dir: Path = DEFAULT_VECTOR_CAPTURE_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    dictionary_size: int = 8,
) -> dict[str, Any]:
    """Write a fail-closed support/value/proxy-artifact decision audit."""

    start = time.time()
    training_summary = _read_json(vector_training_dir / "summary.json")
    arm_metrics = _read_csv(vector_training_dir / "arm_metrics.csv")
    vector_rows = _read_csv(vector_capture_dir / "raw_teacher_residual_vectors.csv")
    parsed_vectors = [_parse_vector_row(row) for row in vector_rows]

    source_rows = [
        _source("vector_training_summary", vector_training_dir / "summary.json", training_summary, 1 if training_summary else 0),
        _source("arm_metrics", vector_training_dir / "arm_metrics.csv", {"status": "read" if arm_metrics else ""}, len(arm_metrics)),
        _source("raw_teacher_residual_vectors", vector_capture_dir / "raw_teacher_residual_vectors.csv", {"status": "read" if parsed_vectors else ""}, len(parsed_vectors)),
    ]
    runtime_failures = [
        {"source": row["source"], "reason": f"{row['path']} missing or empty"}
        for row in source_rows
        if not row["present"] or row["row_count"] <= 0
    ]

    global_metrics = _global_dictionary_metrics(parsed_vectors, dictionary_size) if not runtime_failures else []
    covariance_rows = _covariance_spectrum(parsed_vectors) if not runtime_failures else []
    decision_rows = _decision_rows(training_summary, arm_metrics, global_metrics, covariance_rows)
    blame_rows = _blame_rows(decision_rows)
    blocking_blame = [row for row in blame_rows if row["disposition"] == "blocking"]
    status = "pass" if not runtime_failures else "fail"
    selected_next_action = _selected_next_action(status, blame_rows)
    summary = {
        "status": status,
        "decision": (
            "low_churn_mlp_sparse_factorization_decision_audit_recorded"
            if status == "pass"
            else "low_churn_mlp_sparse_factorization_decision_audit_failed_closed"
        ),
        "claim_status": (
            "sparse_factorization_proxy_artifact_and_deployable_gap_block_gpu"
            if status == "pass"
            else "sparse_factorization_decision_audit_sources_incomplete"
        ),
        "selected_next_action": selected_next_action,
        "selected_next_step": _selected_next_step(selected_next_action),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local decision audit only; exact oracle leakage, global dictionary ceiling, and deployable learned support must be separated before RunPod/Colab",
        "vector_training_dir": str(vector_training_dir),
        "vector_capture_dir": str(vector_capture_dir),
        "out_dir": str(out_dir),
        "dictionary_size": dictionary_size,
        "source_rows": source_rows,
        "decision_rows": decision_rows,
        "blame_rows": blame_rows,
        "global_dictionary_metrics": global_metrics,
        "covariance_summary": _covariance_summary(covariance_rows),
        "runtime_failures": runtime_failures,
        "blocking_blame": blocking_blame,
        "exact_oracle_nondeployable": _decision_value(decision_rows, "exact_oracle_nondeployable"),
        "learned_router_blocks_gpu": _decision_value(decision_rows, "learned_router_blocks_gpu"),
        "global_dictionary_oracle_r2": _metric_value(global_metrics, GLOBAL_ORACLE_ARM, "heldout_reconstruction_r2"),
        "learned_router_heldout_r2": _metric_value(arm_metrics, LEARNED_ARM, "teacher_residual_reconstruction_r2"),
        "oracle_learned_r2_gap": training_summary.get("oracle_learned_r2_gap", ""),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, source_rows, decision_rows, blame_rows, global_metrics, covariance_rows)
    return summary


def _source(source: str, path: Path, summary: dict[str, Any], row_count: int) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": summary.get("status", ""),
        "decision": summary.get("decision", ""),
        "claim_status": summary.get("claim_status", ""),
        "row_count": row_count,
    }


def _decision_rows(
    training_summary: dict[str, Any],
    arm_metrics: list[dict[str, str]],
    global_metrics: list[dict[str, Any]],
    covariance_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    exact_r2 = _metric_value(arm_metrics, EXACT_ORACLE_ARM, "teacher_residual_reconstruction_r2")
    learned_r2 = _metric_value(arm_metrics, LEARNED_ARM, "teacher_residual_reconstruction_r2")
    learned_gap = _float(training_summary.get("oracle_learned_r2_gap"))
    global_r2 = _metric_value(global_metrics, GLOBAL_ORACLE_ARM, "heldout_reconstruction_r2")
    train_global_r2 = _metric_value(global_metrics, GLOBAL_ORACLE_ARM, "train_reconstruction_r2")
    dim95 = _covariance_summary(covariance_rows).get("effective_dim_95pct", "")
    return [
        _decision(
            "exact_oracle_nondeployable",
            exact_r2 is not None and exact_r2 >= 0.99,
            exact_r2,
            "Exact per-row oracle reconstructs by construction and must not count as deployable sparse-column evidence.",
            "proxy_artifact",
        ),
        _decision(
            "learned_router_blocks_gpu",
            learned_r2 is None or learned_r2 < 0.5 or (learned_gap is not None and learned_gap > 0.25),
            {"learned_r2": learned_r2, "oracle_learned_r2_gap": learned_gap},
            "Deployable learned support must reach R2 >= 0.5 and stay within 0.25 R2 of the oracle before GPU validation.",
            "support_router",
        ),
        _decision(
            "global_dictionary_oracle_strong",
            global_r2 is not None and global_r2 >= 0.5,
            {"heldout_r2": global_r2, "train_r2": train_global_r2},
            "Reusable fixed dictionary with oracle support should reconstruct heldout residual vectors with R2 >= 0.5.",
            "value_dictionary",
        ),
        _decision(
            "global_dictionary_oracle_weak",
            global_r2 is None or global_r2 < 0.5,
            {"heldout_r2": global_r2, "train_r2": train_global_r2},
            "Weak reusable-dictionary oracle suggests value capacity or target non-columnability, not just router failure.",
            "value_dictionary",
        ),
        _decision(
            "covariance_intrinsic_dimension_recorded",
            dim95 != "",
            {"effective_dim_95pct": dim95, "spectrum_rows": len(covariance_rows)},
            "Residual-vector covariance spectrum must be recorded for intrinsic-dimension context.",
            "target_geometry",
        ),
    ]


def _decision(signal: str, passed: bool, actual: Any, interpretation: str, category: str) -> dict[str, Any]:
    return {
        "signal": signal,
        "passed": bool(passed),
        "actual": actual,
        "category": category,
        "interpretation": interpretation,
    }


def _blame_rows(decision_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exact_artifact = _decision_value(decision_rows, "exact_oracle_nondeployable")
    learned_block = _decision_value(decision_rows, "learned_router_blocks_gpu")
    global_strong = _decision_value(decision_rows, "global_dictionary_oracle_strong")
    global_weak = _decision_value(decision_rows, "global_dictionary_oracle_weak")
    return [
        {
            "blame_category": "proxy_artifact_failure",
            "disposition": "blocking" if exact_artifact else "not_observed",
            "evidence": "Exact oracle uses row-level target leakage, so oracle R2=1.0 is only a sanity check.",
            "next_implication": "Do not promote or run GPU from exact-oracle reconstruction.",
        },
        {
            "blame_category": "support_router_failure",
            "disposition": "blocking" if learned_block and global_strong else "secondary_or_unresolved",
            "evidence": "Learned/deployable support is far from oracle while reusable global dictionary is strong." if global_strong else "Learned/deployable support is weak, but global dictionary strength is not established.",
            "next_implication": "Redesign prefix-safe support routing only if global dictionary oracle stays strong.",
        },
        {
            "blame_category": "value_dictionary_capacity_or_target_noncolumnability",
            "disposition": "blocking" if global_weak else "not_primary",
            "evidence": "Reusable global dictionary oracle is weak on heldout residual vectors." if global_weak else "Reusable global dictionary oracle clears the local R2 threshold.",
            "next_implication": "If weak, redesign value dictionary or close the sparse-factorization ceiling path before GPU.",
        },
    ]


def _selected_next_action(status: str, blame_rows: list[dict[str, Any]]) -> str:
    if status != "pass":
        return REPAIR_ACTION
    router = next((row for row in blame_rows if row["blame_category"] == "support_router_failure"), {})
    if router.get("disposition") == "blocking":
        return ROUTER_REDESIGN_ACTION
    return NEXT_ACTION


def _selected_next_step(action: str) -> str:
    if action == REPAIR_ACTION:
        return "repair missing vector-training or raw-vector artifacts, then rerun the decision audit"
    if action == ROUTER_REDESIGN_ACTION:
        return "design a prefix-safe support router against the reusable global dictionary oracle before any GPU validation"
    return "redesign the reusable value dictionary or close the sparse-factorization ceiling path before any GPU validation"


def _global_dictionary_metrics(rows: list[dict[str, Any]], dictionary_size: int) -> list[dict[str, Any]]:
    train = [row for row in rows if row["split"] != "heldout"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    if not train or not heldout:
        return []
    dictionary = _fit_dictionary([row["vector"] for row in train], dictionary_size)
    train_mse, train_r2 = _oracle_dictionary_score([row["vector"] for row in train], dictionary)
    heldout_mse, heldout_r2 = _oracle_dictionary_score([row["vector"] for row in heldout], dictionary)
    counts = _support_counts([row["vector"] for row in heldout], dictionary)
    return [
        {
            "arm": GLOBAL_ORACLE_ARM,
            "dictionary_size": len(dictionary),
            "train_row_count": len(train),
            "heldout_row_count": len(heldout),
            "train_reconstruction_mse": train_mse,
            "train_reconstruction_r2": train_r2,
            "heldout_reconstruction_mse": heldout_mse,
            "heldout_reconstruction_r2": heldout_r2,
            "support_entropy": _entropy(counts),
            "support_load_max_fraction": max(counts.values()) / sum(counts.values()) if counts else 0.0,
            "deployable": False,
            "why_non_deployable": "oracle chooses nearest reusable dictionary vector using heldout target residual",
        }
    ]


def _fit_dictionary(vectors: list[list[float]], size: int) -> list[list[float]]:
    array = np.asarray(vectors, dtype=float)
    k = max(1, min(size, len(array)))
    mean = array.mean(axis=0)
    first = int(np.argmin(np.sum((array - mean) ** 2, axis=1)))
    centers = [array[first].copy()]
    while len(centers) < k:
        distances = np.min([np.sum((array - center) ** 2, axis=1) for center in centers], axis=0)
        centers.append(array[int(np.argmax(distances))].copy())
    centers_array = np.asarray(centers)
    for _ in range(12):
        assignments = np.argmin(((array[:, None, :] - centers_array[None, :, :]) ** 2).sum(axis=2), axis=1)
        new_centers = centers_array.copy()
        for index in range(k):
            members = array[assignments == index]
            if len(members):
                new_centers[index] = members.mean(axis=0)
        if np.allclose(new_centers, centers_array):
            break
        centers_array = new_centers
    return centers_array.tolist()


def _oracle_dictionary_score(vectors: list[list[float]], dictionary: list[list[float]]) -> tuple[float, float]:
    if not vectors or not dictionary:
        return 0.0, 0.0
    array = np.asarray(vectors, dtype=float)
    centers = np.asarray(dictionary, dtype=float)
    mse = float(np.mean(np.min(((array[:, None, :] - centers[None, :, :]) ** 2).mean(axis=2), axis=1)))
    baseline = float(np.mean(((array - array.mean(axis=0)) ** 2).mean(axis=1)))
    r2 = 1.0 - (mse / baseline) if baseline > 0 else 0.0
    return mse, r2


def _support_counts(vectors: list[list[float]], dictionary: list[list[float]]) -> dict[int, int]:
    if not vectors or not dictionary:
        return {}
    array = np.asarray(vectors, dtype=float)
    centers = np.asarray(dictionary, dtype=float)
    assignments = np.argmin(((array[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
    counts: dict[int, int] = {}
    for assignment in assignments:
        counts[int(assignment)] = counts.get(int(assignment), 0) + 1
    return counts


def _covariance_spectrum(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    vectors = [row["vector"] for row in rows]
    if len(vectors) < 2:
        return []
    array = np.asarray(vectors, dtype=float)
    cov = np.cov(array, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]
    total = float(np.sum(np.maximum(eigenvalues, 0.0)))
    rows_out = []
    cumulative = 0.0
    for index, value in enumerate(eigenvalues, start=1):
        clean = max(float(value), 0.0)
        cumulative += clean
        rows_out.append(
            {
                "component": index,
                "eigenvalue": clean,
                "explained_variance_fraction": clean / total if total > 0 else 0.0,
                "cumulative_explained_variance_fraction": cumulative / total if total > 0 else 0.0,
            }
        )
    return rows_out


def _covariance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"component_count": 0, "effective_dim_90pct": "", "effective_dim_95pct": "", "participation_ratio": ""}
    eigs = [float(row["eigenvalue"]) for row in rows]
    total = sum(eigs)
    return {
        "component_count": len(rows),
        "effective_dim_90pct": _dim_for_fraction(rows, 0.9),
        "effective_dim_95pct": _dim_for_fraction(rows, 0.95),
        "participation_ratio": (total * total / sum(value * value for value in eigs)) if total > 0 else 0.0,
    }


def _dim_for_fraction(rows: list[dict[str, Any]], fraction: float) -> int:
    for row in rows:
        if float(row["cumulative_explained_variance_fraction"]) >= fraction:
            return int(row["component"])
    return len(rows)


def _parse_vector_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "teacher_row_id": row.get("teacher_row_id", ""),
        "split": row.get("split", ""),
        "token_index": _int(row.get("token_index"), 0),
        "vector": _json_list(row.get("teacher_residual_update_vector")),
    }


def _decision_value(rows: list[dict[str, Any]], signal: str) -> bool:
    for row in rows:
        if row.get("signal") == signal:
            return bool(row.get("passed"))
    return False


def _metric_value(rows: list[dict[str, Any]] | list[dict[str, str]], arm: str, key: str) -> float | None:
    for row in rows:
        if row.get("arm") == arm:
            return _float(row.get(key))
    return None


def _entropy(counts: dict[int, int]) -> float:
    total = sum(counts.values())
    return -sum((count / total) * math.log(count / total) for count in counts.values()) if total else 0.0


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
    decision_rows: list[dict[str, Any]],
    blame_rows: list[dict[str, Any]],
    global_metrics: list[dict[str, Any]],
    covariance_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", source_rows)
    _write_csv(out_dir / "decision_rows.csv", decision_rows)
    _write_csv(out_dir / "blame_rows.csv", blame_rows)
    _write_csv(out_dir / "global_dictionary_metrics.csv", global_metrics)
    _write_csv(out_dir / "covariance_spectrum.csv", covariance_rows)
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
            "# Low-Churn MLP Sparse-Factorization Decision Audit",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Claim status: `{summary['claim_status']}`",
            f"- Selected action: `{summary['selected_next_action']}`",
            f"- Exact oracle nondeployable: `{summary['exact_oracle_nondeployable']}`",
            f"- Learned router blocks GPU: `{summary['learned_router_blocks_gpu']}`",
            f"- Learned heldout R2: `{summary['learned_router_heldout_r2']}`",
            f"- Global dictionary oracle heldout R2: `{summary['global_dictionary_oracle_r2']}`",
            f"- Covariance summary: `{json.dumps(summary['covariance_summary'], sort_keys=True)}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            "",
            "The exact oracle is labeled nondeployable because it can use heldout row target information. The reusable global-dictionary oracle is a separate ceiling: if it is weak, the blocker is value capacity or target non-columnability; if it is strong while learned support is weak, the blocker is support routing.",
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
    parser.add_argument("--vector-training-dir", type=Path, default=DEFAULT_VECTOR_TRAINING_DIR)
    parser.add_argument("--vector-capture-dir", type=Path, default=DEFAULT_VECTOR_CAPTURE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dictionary-size", type=int, default=8)
    args = parser.parse_args(argv)
    summary = run_low_churn_mlp_sparse_factorization_decision_audit(
        vector_training_dir=args.vector_training_dir,
        vector_capture_dir=args.vector_capture_dir,
        out_dir=args.out,
        dictionary_size=args.dictionary_size,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "selected_next_action": summary["selected_next_action"],
                "global_dictionary_oracle_r2": summary["global_dictionary_oracle_r2"],
                "advance_to_gpu_validation": summary["advance_to_gpu_validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
