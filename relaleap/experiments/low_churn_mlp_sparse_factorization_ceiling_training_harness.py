"""Run a bounded local sparse-factorization ceiling harness on extracted rows."""

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


DEFAULT_EXTRACTOR_DIR = Path("results/reports/low_churn_mlp_sparse_factorization_ceiling_extractor")
DEFAULT_OUT_DIR = Path("results/reports/low_churn_mlp_sparse_factorization_ceiling_training_harness")

NEXT_ACTION = "capture_raw_low_churn_teacher_residual_vectors_for_sparse_factorization"
REPAIR_ACTION = "repair_low_churn_mlp_sparse_factorization_ceiling_training_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "training_rows.csv",
    "arm_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_low_churn_mlp_sparse_factorization_ceiling_training_harness(
    *,
    extractor_dir: Path = DEFAULT_EXTRACTOR_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    column_count: int = 8,
) -> dict[str, Any]:
    """Write deterministic proxy sparse-factorization rows from extractor output.

    The extractor currently exposes scalar teacher proxies rather than raw teacher
    residual vectors. This harness intentionally trains only a small deterministic
    scalar ceiling over those rows and fails closed for scientific/GPU promotion.
    """

    start = time.time()
    extractor_summary = _read_json(extractor_dir / "summary.json")
    teacher_rows = _read_csv(extractor_dir / "teacher_residual_rows.csv")
    support_schema = _read_csv(extractor_dir / "support_arm_schema.csv")
    budget_rows = _read_csv(extractor_dir / "teacher_budget_rows.csv")
    factorization_schema = _read_csv(extractor_dir / "factorization_schema.csv")

    source_rows = _source_rows(
        extractor_dir=extractor_dir,
        extractor_summary=extractor_summary,
        teacher_rows=teacher_rows,
        support_schema=support_schema,
        budget_rows=budget_rows,
        factorization_schema=factorization_schema,
    )
    training_rows = _training_rows(teacher_rows, support_schema, column_count)
    arm_rows = _arm_metrics(training_rows, teacher_rows, budget_rows)
    gate_rows = _gate_rows(
        extractor_summary=extractor_summary,
        source_rows=source_rows,
        teacher_rows=teacher_rows,
        support_schema=support_schema,
        factorization_schema=factorization_schema,
        training_rows=training_rows,
        arm_rows=arm_rows,
    )
    runtime_failures = [row for row in gate_rows if not row["passed"] and row["gate_type"] == "runtime"]
    advancement_failures = [row for row in gate_rows if not row["passed"] and row["gate_type"] == "scientific_advancement"]
    status = "pass" if not runtime_failures else "fail"
    summary = {
        "status": status,
        "decision": (
            "low_churn_mlp_sparse_factorization_ceiling_training_harness_recorded"
            if status == "pass"
            else "low_churn_mlp_sparse_factorization_ceiling_training_harness_failed_closed"
        ),
        "claim_status": "proxy_scalar_factorization_only_no_sparse_ceiling_claim",
        "selected_next_action": NEXT_ACTION if status == "pass" else REPAIR_ACTION,
        "selected_next_step": (
            "capture raw low-churn teacher residual vectors/logit intervention rows before real sparse-factorization training"
            if status == "pass"
            else "repair missing extractor artifacts before rerunning the harness"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "training_executed": True,
        "training_scope": "local deterministic scalar proxy harness; no raw residual-vector optimization",
        "backend_policy": "RunPod/Colab remain blocked until raw residual vectors, CE transfer, churn, commutator, and intervention gates exist",
        "extractor_dir": str(extractor_dir),
        "out_dir": str(out_dir),
        "column_count": column_count,
        "source_rows": source_rows,
        "training_row_count": len(training_rows),
        "heldout_training_row_count": sum(1 for row in training_rows if row["split"] == "heldout"),
        "arm_count": len(arm_rows),
        "runtime_failures": runtime_failures,
        "advancement_failures": advancement_failures,
        "gate_criteria": gate_rows,
        "best_proxy_arm": _best_proxy_arm(arm_rows),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, source_rows, training_rows, arm_rows, gate_rows)
    return summary


def _source_rows(
    *,
    extractor_dir: Path,
    extractor_summary: dict[str, Any],
    teacher_rows: list[dict[str, str]],
    support_schema: list[dict[str, str]],
    budget_rows: list[dict[str, str]],
    factorization_schema: list[dict[str, str]],
) -> list[dict[str, Any]]:
    return [
        _source("extractor_summary", extractor_dir / "summary.json", 1 if extractor_summary else 0, extractor_summary.get("status", ""), extractor_summary.get("decision", "")),
        _source("teacher_residual_rows", extractor_dir / "teacher_residual_rows.csv", len(teacher_rows), "read" if teacher_rows else "missing_or_empty", ""),
        _source("support_arm_schema", extractor_dir / "support_arm_schema.csv", len(support_schema), "read" if support_schema else "missing_or_empty", ""),
        _source("teacher_budget_rows", extractor_dir / "teacher_budget_rows.csv", len(budget_rows), "read" if budget_rows else "missing_or_empty", ""),
        _source("factorization_schema", extractor_dir / "factorization_schema.csv", len(factorization_schema), "read" if factorization_schema else "missing_or_empty", ""),
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


def _training_rows(
    teacher_rows: list[dict[str, str]],
    support_schema: list[dict[str, str]],
    column_count: int,
) -> list[dict[str, Any]]:
    if not teacher_rows or not support_schema:
        return []
    arms = [row["arm"] for row in support_schema if row.get("arm")]
    train_rows = [row for row in teacher_rows if row.get("split") != "heldout"]
    means = _column_means(train_rows, column_count)
    global_mean = _mean(_target(row) for row in train_rows)
    rows: list[dict[str, Any]] = []
    for arm in arms:
        for row in teacher_rows:
            token_index = _int(row.get("token_index"), 0)
            target = _target(row)
            support = _support_for_arm(arm, token_index, column_count)
            prediction = _prediction_for_arm(arm, support, means, global_mean, target)
            rows.append(
                {
                    "arm": arm,
                    "teacher_row_id": row.get("teacher_row_id", ""),
                    "token_index": token_index,
                    "split": row.get("split", ""),
                    "support": "|".join(str(item) for item in support),
                    "target_teacher_residual_update_l2": target,
                    "predicted_sparse_residual_update_l2": prediction,
                    "squared_reconstruction_error": (prediction - target) ** 2,
                    "teacher_ce_loss": _float(row.get("teacher_ce_loss")),
                    "base_ce_loss": _float(row.get("base_ce_loss")),
                    "proxy_ce_transfer": _proxy_ce_transfer(row, prediction, target),
                    "raw_teacher_vector_available": _boolish(row.get("raw_teacher_vector_available")),
                    "raw_intervention_available": _boolish(row.get("raw_intervention_available")),
                    "training_mode": "scalar_proxy_centroid",
                }
            )
    return rows


def _column_means(rows: list[dict[str, str]], column_count: int) -> dict[int, float]:
    buckets: dict[int, list[float]] = {index: [] for index in range(column_count)}
    for row in rows:
        token_index = _int(row.get("token_index"), 0)
        buckets[token_index % column_count].append(_target(row))
    global_mean = _mean(_target(row) for row in rows)
    return {index: (_mean(values) if values else global_mean) for index, values in buckets.items()}


def _support_for_arm(arm: str, token_index: int, column_count: int) -> list[int]:
    primary = token_index % column_count
    if arm == "oracle_support_sparse_ceiling":
        return [primary]
    if arm == "learned_router_sparse_factorization":
        return [primary]
    if arm == "token_position_router_sparse_factorization":
        return [token_index % max(1, column_count // 2)]
    if arm == "frequency_support_router_sparse_factorization":
        return [0]
    if arm == "random_fixed_support_sparse_factorization":
        return [(token_index * 3 + 1) % column_count]
    if arm == "route_scrambled_same_values":
        return [(primary + max(1, column_count // 2)) % column_count]
    if arm == "shuffled_teacher_residual_sparse_factorization":
        return [primary]
    return [primary]


def _prediction_for_arm(
    arm: str,
    support: list[int],
    means: dict[int, float],
    global_mean: float,
    target: float,
) -> float:
    if arm == "oracle_support_sparse_ceiling":
        return target
    if arm == "shuffled_teacher_residual_sparse_factorization":
        return global_mean
    values = [means.get(item, global_mean) for item in support]
    return _mean(values)


def _arm_metrics(
    training_rows: list[dict[str, Any]],
    teacher_rows: list[dict[str, str]],
    budget_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    arms = sorted({row["arm"] for row in training_rows})
    budget = {row.get("metric", ""): _float(row.get("value")) for row in budget_rows}
    teacher_ce = budget.get("teacher_heldout_ce_loss")
    shuffled_ce = budget.get("shuffled_null_heldout_ce_loss")
    target_values = [_target(row) for row in teacher_rows if row.get("split") == "heldout"]
    baseline_mse = _variance(target_values)
    rows: list[dict[str, Any]] = []
    for arm in arms:
        heldout = [row for row in training_rows if row["arm"] == arm and row["split"] == "heldout"]
        all_arm = [row for row in training_rows if row["arm"] == arm]
        mse = _mean(row["squared_reconstruction_error"] for row in heldout)
        r2 = 1.0 - (mse / baseline_mse) if baseline_mse > 0 else 0.0
        rows.append(
            {
                "arm": arm,
                "heldout_row_count": len(heldout),
                "teacher_residual_reconstruction_mse": mse,
                "teacher_residual_reconstruction_r2": r2,
                "teacher_gap_closure_fraction": max(0.0, min(1.0, r2)),
                "heldout_ce_transfer_proxy": _mean(row["proxy_ce_transfer"] for row in heldout),
                "oracle_support_regret_proxy": mse,
                "support_entropy": _support_entropy(all_arm),
                "functional_churn_flip_rate_proxy": "",
                "anchor_kl_proxy": budget.get("teacher_heldout_anchor_kl_vs_base"),
                "finite_update_commutator_proxy": "",
                "intervention_fingerprint_specificity_proxy": "",
                "beats_shuffled_teacher_ce_proxy": (
                    teacher_ce is not None and shuffled_ce is not None and teacher_ce < shuffled_ce
                ),
                "scientific_advancement": False,
                "advancement_blocker": "raw residual vectors/logit interventions unavailable; scalar proxy harness cannot establish sparse factorization",
            }
        )
    return rows


def _gate_rows(
    *,
    extractor_summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    teacher_rows: list[dict[str, str]],
    support_schema: list[dict[str, str]],
    factorization_schema: list[dict[str, str]],
    training_rows: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    required_arms = {
        "oracle_support_sparse_ceiling",
        "learned_router_sparse_factorization",
        "token_position_router_sparse_factorization",
        "frequency_support_router_sparse_factorization",
        "random_fixed_support_sparse_factorization",
        "route_scrambled_same_values",
        "shuffled_teacher_residual_sparse_factorization",
    }
    arms = {row.get("arm", "") for row in support_schema}
    schema_fields = {row.get("field", "") for row in factorization_schema}
    raw_vectors = [_boolish(row.get("raw_teacher_vector_available")) for row in teacher_rows]
    return [
        _criterion(
            "extractor_selected_training_harness",
            extractor_summary.get("status") == "pass"
            and extractor_summary.get("selected_next_action") == "implement_low_churn_mlp_sparse_factorization_ceiling_training_harness",
            extractor_summary.get("selected_next_action"),
            "extractor must select this harness",
            "runtime",
        ),
        _criterion(
            "required_sources_present",
            all(row["present"] and row["row_count"] > 0 for row in source_rows),
            source_rows,
            "all extractor artifacts must exist and be nonempty",
            "runtime",
        ),
        _criterion(
            "heldout_teacher_rows_available",
            any(row.get("split") == "heldout" for row in teacher_rows),
            {"teacher_rows": len(teacher_rows), "heldout_rows": sum(1 for row in teacher_rows if row.get("split") == "heldout")},
            "heldout teacher rows are required",
            "runtime",
        ),
        _criterion(
            "support_arm_coverage_complete",
            required_arms.issubset(arms),
            sorted(required_arms - arms),
            "all sparse ceiling arms and nulls are required",
            "runtime",
        ),
        _criterion(
            "factorization_observable_schema_present",
            {
                "teacher_residual_reconstruction_mse",
                "teacher_gap_closure_fraction",
                "finite_update_commutator",
                "intervention_fingerprint_specificity",
            }.issubset(schema_fields),
            sorted(schema_fields),
            "quality, interference, and causal observable fields are required",
            "runtime",
        ),
        _criterion(
            "training_rows_written_for_all_arms",
            bool(training_rows) and {row["arm"] for row in arm_rows} == required_arms,
            sorted({row["arm"] for row in arm_rows}),
            "proxy training rows must cover oracle, learned, shortcut, random, scrambled, and shuffled-teacher arms",
            "runtime",
        ),
        _criterion(
            "raw_teacher_vectors_available",
            any(raw_vectors),
            {"raw_teacher_vector_available_count": sum(1 for value in raw_vectors if value), "teacher_rows": len(raw_vectors)},
            "raw teacher residual vectors are required for a real sparse-factorization claim",
            "scientific_advancement",
        ),
        _criterion(
            "ce_churn_commutator_intervention_gates_available",
            False,
            "proxy harness records reconstruction only; no real CE transfer/churn/commutator/intervention gates",
            "real CE transfer, churn, commutator, and intervention-fingerprint gates must exist before GPU validation",
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


def _proxy_ce_transfer(row: dict[str, str], prediction: float, target: float) -> float | str:
    base = _float(row.get("base_ce_loss"))
    teacher = _float(row.get("teacher_ce_loss"))
    if base is None or teacher is None or target == 0:
        return ""
    closure = max(0.0, min(1.0, prediction / target))
    return base - ((base - teacher) * closure)


def _target(row: dict[str, str]) -> float:
    return _float(row.get("teacher_residual_update_l2")) or 0.0


def _support_entropy(rows: list[dict[str, Any]]) -> float:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get("support", ""))] = counts.get(str(row.get("support", "")), 0) + 1
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def _best_proxy_arm(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    return min(rows, key=lambda row: row["teacher_residual_reconstruction_mse"])["arm"]


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
            "# Low-Churn MLP Sparse-Factorization Ceiling Training Harness",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Claim status: `{summary['claim_status']}`",
            f"- Selected action: `{summary['selected_next_action']}`",
            f"- Training rows: `{summary['training_row_count']}`",
            f"- Best proxy arm: `{summary['best_proxy_arm']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            "",
            "This is a local scalar-proxy harness. It records the sparse-factorization arms and nulls but blocks scientific advancement because the extractor does not yet expose raw low-churn teacher residual vectors or real logit/intervention gates.",
            "",
        ]
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _mean(values: Any) -> float:
    real = [float(value) for value in values if value not in ("", None)]
    return sum(real) / len(real) if real else 0.0


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


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
    parser.add_argument("--extractor-dir", type=Path, default=DEFAULT_EXTRACTOR_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--column-count", type=int, default=8)
    args = parser.parse_args(argv)
    summary = run_low_churn_mlp_sparse_factorization_ceiling_training_harness(
        extractor_dir=args.extractor_dir,
        out_dir=args.out,
        column_count=args.column_count,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "selected_next_action": summary["selected_next_action"],
                "advance_to_gpu_validation": summary["advance_to_gpu_validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
