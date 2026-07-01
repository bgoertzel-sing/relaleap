"""Run a deterministic local pilot for the orthogonalized sparse core/periphery branch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_PREGATE_DIR = Path("results/reports/orthogonalized_sparse_core_periphery_interference_pregate")
DEFAULT_OUT_DIR = Path("results/reports/orthogonalized_sparse_core_periphery_interference_pilot")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "arm_metrics.csv",
    "observable_gates.csv",
    "matched_control_matrix.csv",
    "leakage_null_matrix.csv",
    "notes.md",
)

CANDIDATE_ARM = "orthogonalized_sparse_additive_core_periphery"
NEXT_ACTION = "replace_synthetic_rows_with_bounded_local_cpu_training_pilot"
REPAIR_ACTION = "repair_orthogonalized_sparse_core_periphery_pilot_sources"


def run_orthogonalized_sparse_core_periphery_interference_pilot(
    *,
    pregate_dir: Path = DEFAULT_PREGATE_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write deterministic pilot rows and fail-closed gates for the selected branch."""

    start = time.time()
    pregate_summary_path = pregate_dir / "summary.json"
    pregate = _read_json(pregate_summary_path)
    source_rows = [_source_row("orthogonalized_sparse_core_periphery_interference_pregate", pregate_summary_path, pregate)]
    preflight = _preflight_rows(pregate_dir, pregate)
    arm_rows = _synthetic_arm_rows() if all(row["passed"] for row in preflight) else []
    control_rows = _matched_control_rows(arm_rows)
    null_rows = _leakage_null_rows(arm_rows)
    observable_rows = _observable_rows(arm_rows)
    artifact_rows = _artifact_gate_rows(arm_rows, control_rows, null_rows)
    failures = [row for row in preflight + artifact_rows if not row["passed"]]
    scientific_failures = [row for row in observable_rows if not row["passed"]]
    status = "pass" if not failures else "fail"
    scientific_gate = "blocked" if scientific_failures or status != "pass" else "advances_local_review_only"
    summary = {
        "status": status,
        "decision": (
            "orthogonalized_sparse_core_periphery_interference_pilot_recorded"
            if status == "pass"
            else "orthogonalized_sparse_core_periphery_interference_pilot_failed_closed"
        ),
        "claim_status": (
            "deterministic_schema_pilot_blocks_gpu_until_real_training_rows_clear_dense_mlp_gates"
            if status == "pass"
            else "pilot_sources_or_artifact_contract_incomplete"
        ),
        "scientific_gate": scientific_gate,
        "selected_next_action": NEXT_ACTION if status == "pass" else REPAIR_ACTION,
        "selected_next_step": (
            "replace deterministic synthetic pilot rows with a bounded local CPU training pilot using the same artifacts and gates"
            if status == "pass"
            else "repair the missing pregate source before running the pilot"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local CPU schema/pilot work only; RunPod and Colab remain blocked",
        "pregate_dir": str(pregate_dir),
        "out_dir": str(out_dir),
        "source_rows": source_rows,
        "arm_metrics": arm_rows,
        "matched_control_matrix": control_rows,
        "leakage_null_matrix": null_rows,
        "observable_gates": observable_rows,
        "gate_criteria": preflight + artifact_rows + observable_rows,
        "failures": failures,
        "scientific_failures": scientific_failures,
        "candidate_arm": CANDIDATE_ARM,
        "arm_count": len(arm_rows),
        "matched_control_row_count": len(control_rows),
        "leakage_null_row_count": len(null_rows),
        "observable_gate_count": len(observable_rows),
        "synthetic_rows_only": True,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _preflight_rows(pregate_dir: Path, pregate: dict[str, Any]) -> list[dict[str, Any]]:
    required_files = (
        "summary.json",
        "mechanism_arms.csv",
        "matched_controls.csv",
        "observable_gates.csv",
        "leakage_nulls.csv",
    )
    return [
        _criterion(
            "pregate_summary_selected_pilot",
            pregate.get("status") == "pass"
            and pregate.get("selected_next_action") == "implement_local_orthogonalized_sparse_core_periphery_interference_pilot"
            and pregate.get("requires_gpu_now") is False
            and pregate.get("advance_to_gpu_validation") is False,
            pregate.get("selected_next_action", ""),
            "pregate must pass and select this local CPU pilot with GPU blocked",
        ),
        _criterion(
            "pregate_artifacts_present",
            all((pregate_dir / name).is_file() for name in required_files),
            [name for name in required_files if (pregate_dir / name).is_file()],
            "pregate summary, arms, controls, observable gates, and leakage nulls must exist",
        ),
    ]


def _synthetic_arm_rows() -> list[dict[str, Any]]:
    rows = [
        _arm(CANDIDATE_ARM, "sparse_mechanism_candidate", 0.48, 1.00, 1.12, 384, 1536, 0.19, 0.88, 0.010, 0.62, 0.58, 0.08),
        _arm("orthogonalized_sparse_no_norm_controller_ablation", "mechanism_ablation", 0.62, 1.44, 1.76, 384, 1536, 0.28, 0.82, 0.017, 0.44, 0.42, 0.04),
        _arm("orthogonalized_sparse_no_core_protection_ablation", "mechanism_ablation", 0.50, 1.03, 1.17, 384, 1536, 0.30, 0.78, 0.019, 0.57, 0.50, -0.01),
        _arm("orthogonalized_sparse_no_update_masks_ablation", "mechanism_ablation", 0.47, 1.02, 1.18, 384, 1536, 0.34, 0.75, 0.024, 0.59, 0.51, 0.01),
        _arm("dense_ridge_residual", "matched_control", 0.43, 1.00, 1.10, 384, 384, 0.31, 0.79, 0.020, 0.20, 0.12, 0.0),
        _arm("random_feature_mlp_residual", "matched_control", 0.39, 1.01, 1.11, 384, 1536, 0.29, 0.80, 0.018, 0.24, 0.14, 0.0),
        _arm("low_rank_residual", "matched_control", 0.45, 0.99, 1.09, 256, 768, 0.27, 0.81, 0.016, 0.18, 0.11, 0.0),
        _arm("same_router_flat_value_mlp", "matched_control", 0.41, 1.00, 1.12, 384, 1536, 0.26, 0.82, 0.015, 0.28, 0.16, 0.0),
        _arm("random_sparse_columns", "leakage_or_null_control", 0.91, 1.00, 1.14, 384, 1536, 0.25, 0.83, 0.014, 0.07, 0.05, 0.0),
        _arm("frequency_matched_sparse_router", "leakage_or_null_control", 0.74, 1.00, 1.13, 384, 1536, 0.24, 0.84, 0.013, 0.12, 0.09, 0.0),
        _arm("token_position_only_router", "leakage_or_null_control", 0.86, 1.00, 1.12, 384, 1536, 0.23, 0.84, 0.012, 0.08, 0.07, 0.0),
        _arm("shuffled_teacher_residual_targets", "leakage_or_null_control", 1.02, 1.00, 1.11, 384, 1536, 0.21, 0.86, 0.011, 0.03, 0.02, 0.0),
        _arm("delayed_teacher_residual_targets", "leakage_or_null_control", 0.88, 1.00, 1.11, 384, 1536, 0.22, 0.85, 0.012, 0.06, 0.04, 0.0),
    ]
    best_control_ce = min(row["ce"] for row in rows if row["family"] == "matched_control")
    for row in rows:
        row["ce_delta_vs_best_matched_dense_mlp"] = round(row["ce"] - best_control_ce, 6)
        row["feature_schema_hash"] = _hash_text("token_id,position_id,prefix_hidden_only")
        row["uses_future_hidden_or_delta"] = False
        row["uses_teacher_residual_or_logits_at_eval"] = False
        row["uses_oracle_support_at_eval"] = False
    return rows


def _arm(
    arm: str,
    family: str,
    ce: float,
    residual_l2_mean: float,
    residual_l2_p95: float,
    active_params: int,
    stored_params: int,
    functional_churn_flip_rate: float,
    retention_after_updates: float,
    finite_update_commutator_symmetric_kl: float,
    intervention_selectivity: float,
    context_reuse_score: float,
    periphery_first_pruning_delta: float,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "family": family,
        "row_source": "deterministic_tiny_synthetic_schema_pilot",
        "ce": ce,
        "residual_l2_mean": residual_l2_mean,
        "residual_l2_p95": residual_l2_p95,
        "active_params": active_params,
        "stored_params": stored_params,
        "functional_churn_flip_rate": functional_churn_flip_rate,
        "retention_after_sequential_updates": retention_after_updates,
        "finite_update_commutator_symmetric_kl": finite_update_commutator_symmetric_kl,
        "intervention_selectivity": intervention_selectivity,
        "context_reuse_score": context_reuse_score,
        "periphery_first_pruning_delta": periphery_first_pruning_delta,
    }


def _matched_control_rows(arm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate = _row_for(arm_rows, CANDIDATE_ARM)
    return [
        {
            "control": row["arm"],
            "candidate_ce_delta": round(candidate.get("ce", 0.0) - row["ce"], 6),
            "candidate_churn_delta": round(candidate.get("functional_churn_flip_rate", 0.0) - row["functional_churn_flip_rate"], 6),
            "candidate_retention_delta": round(candidate.get("retention_after_sequential_updates", 0.0) - row["retention_after_sequential_updates"], 6),
            "candidate_commutator_delta": round(candidate.get("finite_update_commutator_symmetric_kl", 0.0) - row["finite_update_commutator_symmetric_kl"], 6),
            "matched_residual_l2": abs(candidate.get("residual_l2_mean", 0.0) - row["residual_l2_mean"]) <= 0.02,
            "matched_active_params": candidate.get("active_params") == row["active_params"],
            "matched_stored_params": candidate.get("stored_params") == row["stored_params"],
        }
        for row in arm_rows
        if row.get("family") == "matched_control"
    ]


def _leakage_null_rows(arm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate = _row_for(arm_rows, CANDIDATE_ARM)
    return [
        {
            "null_control": row["arm"],
            "candidate_ce_better_by": round(row["ce"] - candidate.get("ce", 0.0), 6),
            "candidate_selectivity_better_by": round(candidate.get("intervention_selectivity", 0.0) - row["intervention_selectivity"], 6),
            "null_matches_candidate_within_tolerance": abs(row["ce"] - candidate.get("ce", 0.0)) <= 0.03,
            "uses_nondeployable_features": bool(
                row.get("uses_future_hidden_or_delta")
                or row.get("uses_teacher_residual_or_logits_at_eval")
                or row.get("uses_oracle_support_at_eval")
            ),
        }
        for row in arm_rows
        if row.get("family") == "leakage_or_null_control"
    ]


def _observable_rows(arm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not arm_rows:
        return []
    candidate = _row_for(arm_rows, CANDIDATE_ARM)
    dense_mlp = [row for row in arm_rows if row["arm"] in {"dense_ridge_residual", "random_feature_mlp_residual"}]
    ablations = [row for row in arm_rows if row.get("family") == "mechanism_ablation"]
    nulls = [row for row in arm_rows if row.get("family") == "leakage_or_null_control"]
    best_dense_mlp_ce = min(row["ce"] for row in dense_mlp)
    ce_threshold = best_dense_mlp_ce + min(0.05, best_dense_mlp_ce * 0.05)
    return [
        _criterion("ce_guardrail", candidate["ce"] <= ce_threshold, candidate["ce"], f"candidate CE must be <= {ce_threshold:.6f}"),
        _criterion("residual_l2_budget", all(abs(candidate["residual_l2_mean"] - row["residual_l2_mean"]) <= 0.02 for row in dense_mlp), candidate["residual_l2_mean"], "candidate and dense/MLP controls must match residual L2 mean within 0.02"),
        _criterion("active_and_stored_params", all(row["active_params"] == candidate["active_params"] for row in dense_mlp), candidate["active_params"], "candidate must match dense/MLP active params in this schema pilot"),
        _criterion("functional_churn_flip_rate", candidate["functional_churn_flip_rate"] <= min(row["functional_churn_flip_rate"] for row in dense_mlp), candidate["functional_churn_flip_rate"], "candidate churn must beat or match best dense/MLP"),
        _criterion("retention_after_sequential_updates", candidate["retention_after_sequential_updates"] >= max(row["retention_after_sequential_updates"] for row in dense_mlp), candidate["retention_after_sequential_updates"], "candidate retention must beat or match best dense/MLP"),
        _criterion("finite_update_commutator_symmetric_kl", candidate["finite_update_commutator_symmetric_kl"] <= min(row["finite_update_commutator_symmetric_kl"] for row in dense_mlp), candidate["finite_update_commutator_symmetric_kl"], "candidate commutator must beat or match best dense/MLP"),
        _criterion("intervention_selectivity", candidate["intervention_selectivity"] >= 0.50, candidate["intervention_selectivity"], "candidate selectivity must be at least 0.50"),
        _criterion("context_reuse_score", candidate["context_reuse_score"] >= 0.50, candidate["context_reuse_score"], "candidate reuse score must be at least 0.50"),
        _criterion("periphery_first_pruning_delta", candidate["periphery_first_pruning_delta"] > max(row["periphery_first_pruning_delta"] for row in ablations), candidate["periphery_first_pruning_delta"], "full candidate must have stronger periphery-first pruning signal than ablations"),
        _criterion("null_control_rejection", all(row["ce"] - candidate["ce"] > 0.03 for row in nulls), candidate["ce"], "all leakage/null controls must be worse than candidate by >0.03 CE"),
        _criterion("deployable_feature_schema", not any(row.get("uses_future_hidden_or_delta") or row.get("uses_teacher_residual_or_logits_at_eval") or row.get("uses_oracle_support_at_eval") for row in arm_rows), candidate["feature_schema_hash"], "no evaluation row may use future hidden/delta, teacher residual/logits, or oracle supports"),
    ]


def _artifact_gate_rows(
    arm_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    null_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _criterion("candidate_and_ablations_present", {CANDIDATE_ARM, "orthogonalized_sparse_no_norm_controller_ablation", "orthogonalized_sparse_no_core_protection_ablation", "orthogonalized_sparse_no_update_masks_ablation"}.issubset({row["arm"] for row in arm_rows}), len(arm_rows), "candidate plus norm/core/mask ablations are required"),
        _criterion("matched_controls_present", len(control_rows) >= 4, len(control_rows), "dense, MLP, low-rank, and flat controls are required"),
        _criterion("leakage_nulls_present", len(null_rows) >= 5, len(null_rows), "token-position, shuffled, delayed, random, and frequency null rows are required"),
    ]


def _row_for(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    for row in rows:
        if row.get("arm") == arm:
            return row
    return {}


def _criterion(criterion: str, passed: bool, actual: Any, threshold: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "actual": actual,
        "threshold": threshold,
        "failure_reason": "" if passed else threshold,
    }


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file() and bool(payload),
        "sha256": _file_sha256(path),
        "mtime": _mtime(path),
        "status": payload.get("status", "missing" if not path.is_file() else ""),
        "decision": payload.get("decision", ""),
        "selected_next_action": payload.get("selected_next_action", ""),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "arm_metrics.csv", summary["arm_metrics"])
    _write_csv(out_dir / "observable_gates.csv", summary["observable_gates"])
    _write_csv(out_dir / "matched_control_matrix.csv", summary["matched_control_matrix"])
    _write_csv(out_dir / "leakage_null_matrix.csv", summary["leakage_null_matrix"])
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


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Orthogonalized Sparse Core/Periphery Interference Pilot",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Scientific gate: `{summary['scientific_gate']}`",
            f"- Selected action: `{summary['selected_next_action']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            f"- Synthetic rows only: `{summary['synthetic_rows_only']}`",
            "",
            "GPU validation remains blocked. These rows are a deterministic local schema pilot, not training evidence.",
            "",
        ]
    )


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mtime(path: Path) -> str:
    if not path.is_file():
        return ""
    return str(path.stat().st_mtime)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pregate-dir", type=Path, default=DEFAULT_PREGATE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_orthogonalized_sparse_core_periphery_interference_pilot(
        pregate_dir=args.pregate_dir,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "scientific_gate": summary["scientific_gate"],
                "selected_next_action": summary["selected_next_action"],
                "requires_gpu_now": summary["requires_gpu_now"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
