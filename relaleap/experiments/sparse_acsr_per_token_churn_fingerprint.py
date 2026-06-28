"""Sparse ACSR per-token churn fingerprint coverage report.

This report consumes existing command-generated sparse and dense artifacts. It
does not promote sparse ACSR claims when only aggregate churn fields are
available.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_COMMON_BENCHMARK_DIR = Path("results/reports/acsr_common_causal_residual_benchmark")
DEFAULT_SPARSE_GATE_DIR = Path("results/reports/acsr_sparse_dense_mechanism_gate")
DEFAULT_DENSE_OBSERVABLES_DIR = Path("results/reports/acsr_dense_mechanism_observables")
DEFAULT_MLP_FINGERPRINT_DIR = Path("results/reports/mlp_churn_intervention_fingerprint")
DEFAULT_OUT_DIR = Path("results/reports/sparse_acsr_per_token_churn_fingerprint")

PRIMARY_SPARSE_ARM = "sparse_contextual_topk2"
SPARSE_ARMS = (
    PRIMARY_SPARSE_ARM,
    "sparse_rank_matched_topk1",
    "sparse_teacher_distilled_norm_topk2",
    "sparse_frequency_matched_random_topk1",
)
DENSE_MLP_ARMS = (
    "dense_rank16_best_norm",
    "dense_rank24_best_norm",
    "parameter_matched_causal_mlp_control",
)
REQUIRED_PER_TOKEN_FIELDS = (
    "arm",
    "split",
    "base_ce_loss",
    "ce_loss",
    "delta_vs_base_ce",
    "residual_update_l2",
)
REQUIRED_CHURN_FIELDS = (
    "logit_mse_vs_base",
    "prediction_changed_vs_base",
)
REQUIRED_RAW_FIELDS = (
    "residual_update_vector",
    "base_logits",
    "candidate_logits",
)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "available_sparse_rows.csv",
    "sparse_proxy_strata.csv",
    "dense_mlp_reference.csv",
    "missing_field_matrix.csv",
    "decision_criteria.csv",
    "notes.md",
)


def run_sparse_acsr_per_token_churn_fingerprint(
    *,
    common_benchmark_dir: Path = DEFAULT_COMMON_BENCHMARK_DIR,
    sparse_gate_dir: Path = DEFAULT_SPARSE_GATE_DIR,
    dense_observables_dir: Path = DEFAULT_DENSE_OBSERVABLES_DIR,
    mlp_fingerprint_dir: Path = DEFAULT_MLP_FINGERPRINT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    min_heldout_rows_per_sparse_arm: int = 16,
) -> dict[str, Any]:
    """Write sparse proxy CE/L2 strata and fail closed on missing churn/raw fields."""

    start = time.time()
    common_summary = _read_json(common_benchmark_dir / "summary.json")
    common_arm_rows = _read_csv(common_benchmark_dir / "arm_metrics.csv")
    sparse_per_token_rows = _read_csv(common_benchmark_dir / "per_token_metrics.csv")
    sparse_gate_summary = _read_json(sparse_gate_dir / "summary.json")
    sparse_gate_rows = _read_csv(sparse_gate_dir / "mechanism_metrics.csv")
    dense_mlp_rows = _read_csv(dense_observables_dir / "per_token_observables.csv")
    mlp_summary = _read_json(mlp_fingerprint_dir / "summary.json")
    mlp_available_rows = _read_csv(mlp_fingerprint_dir / "available_arms.csv")

    available_sparse_rows = _available_sparse_rows(common_arm_rows, sparse_gate_rows, sparse_per_token_rows)
    sparse_proxy_strata = _sparse_proxy_strata(sparse_per_token_rows)
    dense_mlp_reference = _dense_mlp_reference(dense_mlp_rows, mlp_available_rows)
    missing_field_matrix = _missing_field_matrix(sparse_per_token_rows, dense_mlp_rows)
    criteria = _criteria(
        common_summary=common_summary,
        sparse_gate_summary=sparse_gate_summary,
        mlp_summary=mlp_summary,
        available_sparse_rows=available_sparse_rows,
        missing_field_matrix=missing_field_matrix,
        min_heldout_rows_per_sparse_arm=min_heldout_rows_per_sparse_arm,
    )
    failures = [row for row in criteria if not row["passed"]]
    sparse_missing_churn = [
        row for row in missing_field_matrix
        if row["family"] == "sparse"
        and int(row["rows"] or 0) > 0
        and (row["missing_churn_fields"] or row["missing_raw_fields"])
    ]

    if failures or sparse_missing_churn:
        status = "fail"
        decision = "sparse_acsr_per_token_churn_fingerprint_blocked_by_missing_sparse_fields"
        claim_status = "sparse_proxy_ce_l2_rows_available_but_churn_fingerprint_not_decisive"
        selected_next_step = (
            "extend acsr_common_causal_residual_benchmark sparse evaluation to emit "
            "per-token logit_mse_vs_base, prediction_changed_vs_base, residual_update_vector, "
            "base_logits, and candidate_logits"
        )
    else:
        status = "pass"
        decision = "sparse_acsr_per_token_churn_fingerprint_available"
        claim_status = "sparse_per_token_ce_l2_churn_raw_fields_available_for_matching"
        selected_next_step = (
            "join sparse, dense, and MLP per-token rows into one CE/L2/churn matched "
            "intervention decision report"
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "promotion_allowed": False,
        "requires_gpu_now": False,
        "source_dirs": {
            "common_benchmark": str(common_benchmark_dir),
            "sparse_gate": str(sparse_gate_dir),
            "dense_observables": str(dense_observables_dir),
            "mlp_fingerprint": str(mlp_fingerprint_dir),
        },
        "required_per_token_fields": list(REQUIRED_PER_TOKEN_FIELDS),
        "required_churn_fields": list(REQUIRED_CHURN_FIELDS),
        "required_raw_fields": list(REQUIRED_RAW_FIELDS),
        "available_sparse_row_count": len(available_sparse_rows),
        "sparse_proxy_strata_row_count": len(sparse_proxy_strata),
        "dense_mlp_reference_row_count": len(dense_mlp_reference),
        "missing_sparse_churn_or_raw_rows": len(sparse_missing_churn),
        "criteria": criteria,
        "failures": failures,
        "selected_next_step": selected_next_step,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(
        out_dir,
        summary,
        available_sparse_rows,
        sparse_proxy_strata,
        dense_mlp_reference,
        missing_field_matrix,
        criteria,
    )
    return summary


def _available_sparse_rows(
    common_arm_rows: list[dict[str, str]],
    sparse_gate_rows: list[dict[str, str]],
    sparse_per_token_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    common_by_arm = {row.get("arm", ""): row for row in common_arm_rows}
    gate_by_arm = {row.get("arm", ""): row for row in sparse_gate_rows}
    counts: dict[str, int] = defaultdict(int)
    heldout_counts: dict[str, int] = defaultdict(int)
    for row in sparse_per_token_rows:
        arm = row.get("arm", "")
        counts[arm] += 1
        if row.get("split") == "heldout":
            heldout_counts[arm] += 1

    rows: list[dict[str, Any]] = []
    for arm in SPARSE_ARMS:
        common = common_by_arm.get(arm, {})
        gate = gate_by_arm.get(_gate_equivalent_arm(arm), {})
        if not common and not counts.get(arm):
            continue
        rows.append(
            {
                "arm": arm,
                "gate_equivalent_arm": _gate_equivalent_arm(arm),
                "per_token_rows": counts.get(arm, 0),
                "heldout_rows": heldout_counts.get(arm, 0),
                "heldout_ce_loss": _float_or_blank(common.get("heldout_ce_loss")),
                "heldout_delta_vs_base_ce": _float_or_blank(common.get("heldout_delta_vs_base_ce")),
                "heldout_residual_update_l2": _float_or_blank(common.get("heldout_residual_update_l2")),
                "active_params_proxy": _float_or_blank(common.get("active_params_proxy")),
                "aggregate_gate_ce_loss": _float_or_blank(gate.get("ce_loss")),
                "aggregate_gate_functional_churn": _float_or_blank(gate.get("functional_churn")),
                "aggregate_gate_anchor_kl_or_logit_mse": _float_or_blank(gate.get("anchor_kl_or_logit_mse")),
                "aggregate_gate_fingerprint_purity": _float_or_blank(gate.get("intervention_fingerprint_purity")),
            }
        )
    return rows


def _sparse_proxy_strata(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        arm = row.get("arm", "")
        if arm not in SPARSE_ARMS:
            continue
        grouped[(arm, "split", row.get("split", ""))].append(row)
        grouped[(arm, "residual_l2_bin", _l2_bin(_float(row.get("residual_update_l2"))))].append(row)
        grouped[(arm, "residual_gain_bin", _gain_bin(_float(row.get("delta_vs_base_ce"))))].append(row)
        position = int(_float(row.get("position_index")) or 0)
        grouped[(arm, "position_parity", "even" if position % 2 == 0 else "odd")].append(row)

    strata: list[dict[str, Any]] = []
    for (arm, stratum_type, stratum), bucket in sorted(grouped.items()):
        strata.append(
            {
                "arm": arm,
                "stratum_type": stratum_type,
                "stratum": stratum,
                "row_count": len(bucket),
                "base_ce_loss": _mean(bucket, "base_ce_loss"),
                "ce_loss": _mean(bucket, "ce_loss"),
                "delta_vs_base_ce": _mean(bucket, "delta_vs_base_ce"),
                "residual_update_l2": _mean(bucket, "residual_update_l2"),
                "improvement_purity_proxy": _improvement_purity(bucket),
            }
        )
    return strata


def _dense_mlp_reference(
    dense_mlp_rows: list[dict[str, str]],
    mlp_available_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    aggregate_by_arm = {row.get("arm", ""): row for row in mlp_available_rows}
    rows: list[dict[str, Any]] = []
    for arm in DENSE_MLP_ARMS:
        arm_rows = [row for row in dense_mlp_rows if row.get("arm") == arm]
        heldout = [row for row in arm_rows if row.get("split") == "heldout"]
        aggregate = aggregate_by_arm.get(arm, {})
        rows.append(
            {
                "arm": arm,
                "per_token_rows": len(arm_rows),
                "heldout_rows": len(heldout),
                "heldout_ce_loss": _mean(heldout, "ce_loss"),
                "heldout_delta_vs_base_ce": _mean(heldout, "delta_vs_base_ce"),
                "heldout_residual_update_l2": _mean(heldout, "residual_update_l2"),
                "heldout_logit_mse_vs_base": _mean(heldout, "logit_mse_vs_base"),
                "heldout_prediction_changed_vs_base": _mean_bool(heldout, "prediction_changed_vs_base"),
                "aggregate_functional_churn": _float_or_blank(aggregate.get("functional_churn")),
            }
        )
    return rows


def _missing_field_matrix(
    sparse_rows: list[dict[str, str]],
    dense_mlp_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, source_rows, arms in (
        ("sparse", sparse_rows, SPARSE_ARMS),
        ("dense_mlp", dense_mlp_rows, DENSE_MLP_ARMS),
    ):
        for arm in arms:
            arm_rows = [row for row in source_rows if row.get("arm") == arm]
            heldout = [row for row in arm_rows if row.get("split") == "heldout"]
            rows.append(
                {
                    "family": family,
                    "arm": arm,
                    "rows": len(arm_rows),
                    "heldout_rows": len(heldout),
                    "missing_required_fields": ";".join(_missing_fields(arm_rows, REQUIRED_PER_TOKEN_FIELDS)),
                    "missing_churn_fields": ";".join(_missing_fields(arm_rows, REQUIRED_CHURN_FIELDS)),
                    "missing_raw_fields": ";".join(_missing_fields(arm_rows, REQUIRED_RAW_FIELDS)),
                }
            )
    return rows


def _criteria(
    *,
    common_summary: dict[str, Any],
    sparse_gate_summary: dict[str, Any],
    mlp_summary: dict[str, Any],
    available_sparse_rows: list[dict[str, Any]],
    missing_field_matrix: list[dict[str, Any]],
    min_heldout_rows_per_sparse_arm: int,
) -> list[dict[str, Any]]:
    primary = next((row for row in available_sparse_rows if row.get("arm") == PRIMARY_SPARSE_ARM), {})
    sparse_missing_required = [
        row for row in missing_field_matrix
        if row["family"] == "sparse" and int(row["rows"] or 0) > 0 and row["missing_required_fields"]
    ]
    sparse_missing_churn = [
        row for row in missing_field_matrix
        if row["family"] == "sparse" and int(row["rows"] or 0) > 0 and row["missing_churn_fields"]
    ]
    sparse_missing_raw = [
        row for row in missing_field_matrix
        if row["family"] == "sparse" and int(row["rows"] or 0) > 0 and row["missing_raw_fields"]
    ]
    dense_mlp_missing_churn_raw = [
        row for row in missing_field_matrix
        if row["family"] == "dense_mlp" and (row["missing_churn_fields"] or row["missing_raw_fields"])
    ]
    return [
        _criterion(
            "common_benchmark_completed_to_gate",
            common_summary.get("decision") in {
                "acsr_common_causal_residual_benchmark_supported",
                "acsr_common_causal_residual_benchmark_failed_gate",
            },
            "common sparse-vs-dense benchmark must complete to its scientific gate",
            {"status": common_summary.get("status"), "decision": common_summary.get("decision")},
            "common benchmark missing or failed before writing usable arm/per-token evidence",
        ),
        _criterion(
            "sparse_gate_available",
            sparse_gate_summary.get("status") == "pass",
            "sparse mechanism gate summary should be available as aggregate context",
            {"status": sparse_gate_summary.get("status"), "decision": sparse_gate_summary.get("decision")},
            "sparse gate aggregate context is missing or not passing",
        ),
        _criterion(
            "mlp_fingerprint_available",
            mlp_summary.get("status") == "pass",
            "MLP/dense fingerprint packet should pass so dense/MLP references are valid",
            {"status": mlp_summary.get("status"), "decision": mlp_summary.get("decision")},
            "MLP/dense fingerprint packet missing or not passing",
        ),
        _criterion(
            "primary_sparse_has_heldout_rows",
            int(primary.get("heldout_rows") or 0) >= min_heldout_rows_per_sparse_arm,
            f"primary sparse arm must have at least {min_heldout_rows_per_sparse_arm} heldout rows",
            primary,
            "primary sparse arm lacks enough heldout per-token rows",
        ),
        _criterion(
            "sparse_required_ce_l2_fields_present",
            not sparse_missing_required,
            "sparse rows must include CE and residual-L2 proxy fields",
            sparse_missing_required,
            "one or more sparse arms lacks required CE/L2 fields",
        ),
        _criterion(
            "sparse_churn_fields_present",
            not sparse_missing_churn,
            "sparse rows must include per-token logit MSE and prediction-change fields",
            sparse_missing_churn,
            "sparse per-token churn fields are missing",
        ),
        _criterion(
            "sparse_raw_intervention_fields_present",
            not sparse_missing_raw,
            "sparse rows must include raw residual vectors and base/candidate logits",
            sparse_missing_raw,
            "sparse raw fields needed for lambda-scaled intervention matching are missing",
        ),
        _criterion(
            "dense_mlp_churn_raw_fields_present",
            not dense_mlp_missing_churn_raw,
            "dense/MLP reference rows must include churn and raw intervention fields",
            dense_mlp_missing_churn_raw,
            "dense/MLP reference rows are missing churn/raw fields",
        ),
    ]


def _gate_equivalent_arm(arm: str) -> str:
    if arm == PRIMARY_SPARSE_ARM:
        return "acsr_mlp_predicted_future"
    if arm == "sparse_frequency_matched_random_topk1":
        return "random_fixed_topk2"
    return arm


def _missing_fields(rows: list[dict[str, str]], fields: tuple[str, ...]) -> list[str]:
    if not rows:
        return list(fields)
    missing: list[str] = []
    for field in fields:
        if all(row.get(field, "") == "" for row in rows):
            missing.append(field)
    return missing


def _criterion(
    criterion: str,
    passed: bool,
    threshold: Any,
    actual: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": actual,
        "failure_reason": "" if passed else failure_reason,
    }


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    available_sparse_rows: list[dict[str, Any]],
    sparse_proxy_strata: list[dict[str, Any]],
    dense_mlp_reference: list[dict[str, Any]],
    missing_field_matrix: list[dict[str, Any]],
    criteria: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "available_sparse_rows.csv", available_sparse_rows)
    _write_csv(out_dir / "sparse_proxy_strata.csv", sparse_proxy_strata)
    _write_csv(out_dir / "dense_mlp_reference.csv", dense_mlp_reference)
    _write_csv(out_dir / "missing_field_matrix.csv", missing_field_matrix)
    _write_csv(out_dir / "decision_criteria.csv", criteria)
    lines = [
        "# Sparse ACSR Per-Token Churn Fingerprint",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Promotion allowed: `{summary['promotion_allowed']}`",
        f"- Sparse proxy strata rows: `{summary['sparse_proxy_strata_row_count']}`",
        f"- Dense/MLP reference rows: `{summary['dense_mlp_reference_row_count']}`",
        f"- Next step: {summary['selected_next_step']}",
        "",
        "This local report accepts sparse CE/L2 proxy rows only as coverage evidence. It refuses a sparse low-churn or matched-intervention claim unless sparse rows include per-token churn fields and raw logits/vectors comparable to the dense/MLP packet.",
    ]
    if summary["failures"]:
        lines.extend(["", "## Failures"])
        lines.extend(f"- `{row['criterion']}`: {row['failure_reason']}" for row in summary["failures"])
    (out_dir / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["status"]
        rows = [{"status": "missing"}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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


def _mean(rows: list[dict[str, str]], field: str) -> Any:
    values = [_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return "" if not values else sum(values) / len(values)


def _mean_bool(rows: list[dict[str, str]], field: str) -> Any:
    values = [_bool_or_none(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return "" if not values else sum(1.0 if value else 0.0 for value in values) / len(values)


def _improvement_purity(rows: list[dict[str, str]]) -> Any:
    values = [_float(row.get("delta_vs_base_ce")) for row in rows]
    values = [value for value in values if value is not None]
    gains = sum(max(0.0, -value) for value in values)
    damages = sum(max(0.0, value) for value in values)
    return "" if gains + damages <= 0 else gains / (gains + damages)


def _gain_bin(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < -0.5:
        return "large_gain"
    if value < 0.0:
        return "small_gain"
    return "damage_or_no_gain"


def _l2_bin(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 0.5:
        return "low"
    if value < 1.5:
        return "mid"
    return "high"


def _float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_blank(value: Any) -> Any:
    parsed = _float(value)
    return "" if parsed is None else parsed


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in ("True", "true", "1", 1):
        return True
    if value in ("False", "false", "0", 0):
        return False
    return None


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-benchmark-dir", type=Path, default=DEFAULT_COMMON_BENCHMARK_DIR)
    parser.add_argument("--sparse-gate-dir", type=Path, default=DEFAULT_SPARSE_GATE_DIR)
    parser.add_argument("--dense-observables-dir", type=Path, default=DEFAULT_DENSE_OBSERVABLES_DIR)
    parser.add_argument("--mlp-fingerprint-dir", type=Path, default=DEFAULT_MLP_FINGERPRINT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-heldout-rows-per-sparse-arm", type=int, default=16)
    args = parser.parse_args()
    summary = run_sparse_acsr_per_token_churn_fingerprint(
        common_benchmark_dir=args.common_benchmark_dir,
        sparse_gate_dir=args.sparse_gate_dir,
        dense_observables_dir=args.dense_observables_dir,
        mlp_fingerprint_dir=args.mlp_fingerprint_dir,
        out_dir=args.out,
        min_heldout_rows_per_sparse_arm=args.min_heldout_rows_per_sparse_arm,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
