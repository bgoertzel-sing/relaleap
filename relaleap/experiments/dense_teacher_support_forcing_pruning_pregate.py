"""Run a dense-teacher support-forcing and pruning pregate.

This bounded local pregate follows the post-dense-teacher selector's next
step. It keeps the sparse value model fixed and swaps only the support source
between oracle, learned, and permuted/null supports so support quality is not
confounded with value capacity. It also records a simple causal-efficacy
pruning pass before any GPU/backend validation is considered.
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

from relaleap.experiments.dense_teacher_residual_value_capacity_norm_assay import (
    _Teacher,
    _arm_metrics,
    _fit_dictionary_model,
    _make_data,
    _norm_match,
    _predict_dictionary,
    _source_row,
    _support_rows,
    _train_flat_value_head,
    _train_support_router,
)


DEFAULT_SELECTOR = Path("results/reports/post_dense_teacher_sparse_dictionary_branch_selector/summary.json")
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_support_forcing_pruning_pregate")

DECISION = "dense_teacher_support_forcing_pruning_pregate_recorded"
FAIL_DECISION = "dense_teacher_support_forcing_pruning_pregate_failed_closed"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "support_forcing_rows.csv",
    "pruning_rows.csv",
    "gate_criteria.csv",
    "notes.md",
)

SUPPORT_FORCING_ARMS = (
    "oracle_support_same_values",
    "learned_support_same_values",
    "load_permuted_support_same_values",
    "random_support_same_values",
    "frequency_support_same_values",
    "same_router_flat_value_control",
    "pruned_oracle_support_same_values",
    "pruned_learned_support_same_values",
    "pruned_load_permuted_support_same_values",
)


def run_dense_teacher_support_forcing_pruning_pregate(
    *,
    selector_path: Path = DEFAULT_SELECTOR,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 31,
    teacher_steps: int = 100,
    router_steps: int = 80,
    value_steps: int = 80,
    control_steps: int = 80,
    column_count: int = 6,
    values_per_column: int = 3,
) -> dict[str, Any]:
    """Run the local pregate and write command-generated artifacts."""

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError("dense-teacher support-forcing/pruning pregate requires torch") from exc

    if min(teacher_steps, router_steps, value_steps, control_steps) < 1:
        raise ValueError("all training step counts must be positive")
    if column_count < 2:
        raise ValueError("column_count must be at least 2")
    if values_per_column < 2:
        raise ValueError("values_per_column must be at least 2")

    start = time.time()
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    selector = _read_json(selector_path)
    source_rows = [_source_row("post_dense_teacher_sparse_dictionary_branch_selector", selector_path, selector)]

    data = _make_data(torch, seed=seed, column_count=column_count)
    teacher = _Teacher(torch, data["input_dim"], data["classes"])
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=0.01)
    for _ in range(teacher_steps):
        logits = data["base_logits_train"] + teacher(data["x_train"])
        loss = F.cross_entropy(logits, data["y_train"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        teacher_residual_train = teacher(data["x_train"])
        teacher_residual_holdout = teacher(data["x_holdout"])
        base_holdout_ce = float(F.cross_entropy(data["base_logits_holdout"], data["y_holdout"]).item())
        teacher_holdout_ce = float(
            F.cross_entropy(data["base_logits_holdout"] + teacher_residual_holdout, data["y_holdout"]).item()
        )

    value_model = _fit_dictionary_model(
        torch,
        data["x_train"],
        teacher_residual_train,
        data["support_train"],
        column_count,
        values_per_column,
        steps=value_steps,
        lr=0.035,
    )
    support_router = _train_support_router(
        torch,
        data["x_train"],
        data["support_train"],
        data["input_dim"],
        column_count,
        steps=router_steps,
    )
    flat_value = _train_flat_value_head(
        torch,
        F,
        data["x_train"],
        teacher_residual_train,
        data["input_dim"],
        data["classes"],
        steps=control_steps,
    )

    supports = _support_rows(torch, data, support_router, column_count)
    supports["oracle"] = data["support_holdout"]
    supports["load_permuted"] = _load_permuted_support(torch, data["support_holdout"], column_count)

    same_value_model_id = "oracle_trained_multi_value_dictionary_v1"
    base_predictions = {
        "oracle_support_same_values": (
            _predict_dictionary(torch, value_model, data["x_holdout"], supports["oracle"], teacher_residual_train),
            supports["oracle"],
            True,
            "nondeployable oracle support with the same sparse values",
        ),
        "learned_support_same_values": (
            _predict_dictionary(torch, value_model, data["x_holdout"], supports["learned"], teacher_residual_train),
            supports["learned"],
            False,
            "deployable learned support with the same sparse values",
        ),
        "load_permuted_support_same_values": (
            _predict_dictionary(torch, value_model, data["x_holdout"], supports["load_permuted"], teacher_residual_train),
            supports["load_permuted"],
            False,
            "load-preserving permuted support null with the same sparse values",
        ),
        "random_support_same_values": (
            _predict_dictionary(torch, value_model, data["x_holdout"], supports["random"], teacher_residual_train),
            supports["random"],
            False,
            "deterministic random support null with the same sparse values",
        ),
        "frequency_support_same_values": (
            _predict_dictionary(torch, value_model, data["x_holdout"], supports["frequency"], teacher_residual_train),
            supports["frequency"],
            False,
            "most-frequent support null with the same sparse values",
        ),
        "same_router_flat_value_control": (
            _norm_match(torch, flat_value(data["x_holdout"]), teacher_residual_train),
            supports["learned"],
            False,
            "same learned router support with a flat value head control",
        ),
    }

    pruning_rows, retained_columns = _causal_efficacy_pruning_rows(
        torch,
        F,
        base_predictions["oracle_support_same_values"][0],
        supports["oracle"],
        data,
        column_count,
    )
    predictions = dict(base_predictions)
    for source_arm, pruned_arm in (
        ("oracle_support_same_values", "pruned_oracle_support_same_values"),
        ("learned_support_same_values", "pruned_learned_support_same_values"),
        ("load_permuted_support_same_values", "pruned_load_permuted_support_same_values"),
    ):
        pred, support, oracle, note = base_predictions[source_arm]
        predictions[pruned_arm] = (
            _apply_pruning(torch, pred, support, retained_columns),
            support,
            oracle,
            f"causal-efficacy pruned variant of {source_arm}; {note}",
        )

    support_forcing_rows = []
    for arm, (pred, support, oracle, note) in predictions.items():
        row = _arm_metrics(
            torch,
            F,
            arm,
            pred,
            support,
            teacher_residual_holdout,
            data,
            teacher_holdout_ce,
            base_holdout_ce,
            oracle,
            note,
            column_count,
            values_per_column,
        )
        row["value_model_id"] = same_value_model_id if "flat_value" not in arm else "flat_value_control_v1"
        row["support_forcing_condition"] = _support_condition(arm)
        row["causal_efficacy_pruned"] = arm.startswith("pruned_")
        row["retained_column_count"] = len(retained_columns) if arm.startswith("pruned_") else column_count
        support_forcing_rows.append(row)

    gate_rows = _gate_rows(source_rows, support_forcing_rows, pruning_rows, base_holdout_ce, teacher_holdout_ce)
    runtime_failures = [row for row in gate_rows if row["required"] and not row["passed"]]
    scientific_failures = [row for row in gate_rows if row["gate_type"] == "scientific" and not row["passed"]]
    status = "fail" if runtime_failures else "pass"
    claim_status = (
        "support_forcing_pruning_sparse_specific_gate_ready_for_repeat_no_gpu"
        if status == "pass" and not scientific_failures
        else "support_forcing_pruning_local_gates_block_gpu"
    )
    summary = {
        "status": status,
        "decision": DECISION if status == "pass" else FAIL_DECISION,
        "claim_status": claim_status,
        "selected_next_step": _selected_next_step(status, scientific_failures),
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "backend_policy": "local CPU support-forcing/pruning pregate only; RunPod and Colab remain blocked",
        "training_executed": True,
        "teacher_trained": True,
        "same_sparse_values_across_support_conditions": True,
        "causal_efficacy_pruning_executed": True,
        "teacher_train_steps": teacher_steps,
        "router_train_steps": router_steps,
        "value_train_steps": value_steps,
        "control_train_steps": control_steps,
        "seed": seed,
        "column_count": column_count,
        "values_per_column": values_per_column,
        "base_holdout_ce": round(base_holdout_ce, 6),
        "dense_teacher_holdout_ce": round(teacher_holdout_ce, 6),
        "dense_teacher_ce_improvement": round(base_holdout_ce - teacher_holdout_ce, 6),
        "retained_columns_after_pruning": retained_columns,
        "source_rows": source_rows,
        "support_forcing_rows": support_forcing_rows,
        "pruning_rows": pruning_rows,
        "gate_criteria": gate_rows,
        "failures": runtime_failures + scientific_failures,
        "strategy_review_handling": (
            "Accepted the latest GPT-5.5-Pro dense-teacher recommendation and instantiated the selected "
            "local support-forcing/pruning pregate. No GPU, Colab, or RunPod validation was used."
        ),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _load_permuted_support(torch: Any, support: Any, column_count: int) -> Any:
    rows = []
    for column in range(column_count):
        indices = torch.nonzero(support == column, as_tuple=False).flatten()
        if len(indices) > 0:
            rows.append((indices, torch.full((len(indices),), (column + 1) % column_count, dtype=torch.long)))
    permuted = support.clone()
    for indices, values in rows:
        permuted[indices] = values
    return permuted


def _causal_efficacy_pruning_rows(
    torch: Any,
    F: Any,
    oracle_pred: Any,
    oracle_support: Any,
    data: dict[str, Any],
    column_count: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    full_ce = float(F.cross_entropy(data["base_logits_holdout"] + oracle_pred, data["y_holdout"]).item())
    rows = []
    for column in range(column_count):
        mask = oracle_support == column
        ablated = oracle_pred.clone()
        ablated[mask] = 0.0
        ablated_ce = float(F.cross_entropy(data["base_logits_holdout"] + ablated, data["y_holdout"]).item())
        efficacy = ablated_ce - full_ce
        rows.append(
            {
                "column": column,
                "support_count": int(mask.sum().item()),
                "full_oracle_ce": round(full_ce, 6),
                "ablated_ce": round(ablated_ce, 6),
                "causal_efficacy_ce_delta": round(efficacy, 6),
                "retain_after_pruning": False,
            }
        )
    sorted_deltas = sorted(float(row["causal_efficacy_ce_delta"]) for row in rows)
    threshold = sorted_deltas[max(0, len(sorted_deltas) // 2)]
    retained = []
    for row in rows:
        retain = row["support_count"] > 0 and float(row["causal_efficacy_ce_delta"]) >= threshold
        row["retain_after_pruning"] = bool(retain)
        if retain:
            retained.append(int(row["column"]))
    if not retained:
        best = max(rows, key=lambda row: float(row["causal_efficacy_ce_delta"]))
        best["retain_after_pruning"] = True
        retained.append(int(best["column"]))
    return rows, retained


def _apply_pruning(torch: Any, pred: Any, support: Any, retained_columns: list[int]) -> Any:
    retained = torch.tensor(retained_columns, dtype=support.dtype, device=support.device)
    keep = (support.unsqueeze(1) == retained.unsqueeze(0)).any(dim=1)
    return pred * keep.float().unsqueeze(1)


def _support_condition(arm: str) -> str:
    if "oracle" in arm:
        return "oracle_non_deployable"
    if "learned" in arm or "same_router_flat" in arm:
        return "learned_deployable"
    if "load_permuted" in arm:
        return "load_permuted_null"
    if "random" in arm:
        return "random_null"
    if "frequency" in arm:
        return "frequency_null"
    return "unknown"


def _gate_rows(
    source_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    pruning_rows: list[dict[str, Any]],
    base_ce: float,
    teacher_ce: float,
) -> list[dict[str, Any]]:
    by_arm = {row["arm"]: row for row in rows}
    oracle = by_arm.get("oracle_support_same_values", {})
    learned = by_arm.get("learned_support_same_values", {})
    permuted = by_arm.get("load_permuted_support_same_values", {})
    random = by_arm.get("random_support_same_values", {})
    flat = by_arm.get("same_router_flat_value_control", {})
    pruned_oracle = by_arm.get("pruned_oracle_support_same_values", {})
    sparse_value_ids = {
        row.get("value_model_id")
        for row in rows
        if row["arm"] != "same_router_flat_value_control" and "same_values" in row["arm"]
    }
    oracle_mse = _float(oracle.get("teacher_residual_reconstruction_mse"), math.inf)
    best_support_null_mse = min(
        _float(permuted.get("teacher_residual_reconstruction_mse"), math.inf),
        _float(random.get("teacher_residual_reconstruction_mse"), math.inf),
    )
    return [
        _gate("selector_source_present", all(row["present"] for row in source_rows), True, "runtime", str(source_rows)),
        _gate("required_support_forcing_arms_present", set(SUPPORT_FORCING_ARMS).issubset(by_arm), True, "runtime", ",".join(sorted(by_arm))),
        _gate("same_sparse_values_across_support_conditions", sparse_value_ids == {"oracle_trained_multi_value_dictionary_v1"}, True, "runtime", str(sparse_value_ids)),
        _gate("causal_efficacy_pruning_rows_present", bool(pruning_rows), True, "runtime", f"rows={len(pruning_rows)}"),
        _gate("oracle_support_non_deployable_labeled", bool(oracle.get("oracle_support_non_deployable")), True, "runtime", "oracle support is ceiling evidence only"),
        _gate("gpu_blocked", True, True, "runtime", "requires_gpu_now=false; advance_to_gpu_validation=false"),
        _gate("dense_teacher_improves_base", teacher_ce < base_ce, False, "scientific", f"base_ce={base_ce:.6f}; teacher_ce={teacher_ce:.6f}"),
        _gate("oracle_support_beats_support_nulls_same_values", oracle_mse + 0.02 < best_support_null_mse, False, "scientific", f"oracle_mse={oracle_mse:.6f}; best_null_mse={best_support_null_mse:.6f}"),
        _gate("learned_support_low_forcing_regret", _float(learned.get("teacher_residual_reconstruction_mse"), math.inf) - oracle_mse <= 0.08, False, "scientific", f"learned_mse={learned.get('teacher_residual_reconstruction_mse')}; oracle_mse={oracle.get('teacher_residual_reconstruction_mse')}"),
        _gate("pruning_retains_oracle_ce_gap_closure", _float(pruned_oracle.get("teacher_ce_gap_closure_fraction"), -math.inf) >= 0.75 * _float(oracle.get("teacher_ce_gap_closure_fraction"), math.inf), False, "scientific", f"pruned_closure={pruned_oracle.get('teacher_ce_gap_closure_fraction')}; oracle_closure={oracle.get('teacher_ce_gap_closure_fraction')}"),
        _gate("sparse_specific_beats_flat_value_control", _float(learned.get("teacher_residual_reconstruction_r2"), -math.inf) >= _float(flat.get("teacher_residual_reconstruction_r2"), math.inf) and _float(learned.get("ce"), math.inf) <= base_ce + 0.02, False, "scientific", f"learned_r2={learned.get('teacher_residual_reconstruction_r2')}; flat_r2={flat.get('teacher_residual_reconstruction_r2')}; learned_ce={learned.get('ce')}"),
    ]


def _gate(criterion: str, passed: bool, required: bool, gate_type: str, evidence: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "required": required,
        "gate_type": gate_type,
        "evidence": evidence,
    }


def _selected_next_step(status: str, scientific_failures: list[dict[str, Any]]) -> str:
    if status != "pass":
        return "repair support-forcing/pruning runtime artifacts before interpretation"
    failed = {row["criterion"] for row in scientific_failures}
    if "oracle_support_beats_support_nulls_same_values" in failed:
        return "treat support forcing as insufficient; redesign sparse value basis before GPU"
    if "learned_support_low_forcing_regret" in failed:
        return "inspect learned support forcing regret under identical values before new mechanisms"
    if "sparse_specific_beats_flat_value_control" in failed:
        return "keep GPU blocked and close or redesign sparse-specific value/support mechanism against flat controls"
    if scientific_failures:
        return "inspect failed pruning or guardrail gates before any backend validation"
    return "repeat support-forcing/pruning pregate across seeds before GPU validation"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "support_forcing_rows.csv", summary["support_forcing_rows"])
    _write_csv(out_dir / "pruning_rows.csv", summary["pruning_rows"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Dense-Teacher Support-Forcing/Pruning Pregate",
            "",
            f"Decision: `{summary['decision']}`.",
            f"Claim status: `{summary['claim_status']}`.",
            "",
            "The same sparse value dictionary is evaluated under oracle, learned, and permuted/null supports.",
            "Oracle support remains nondeployable ceiling evidence. GPU validation remains blocked.",
            "",
            f"Retained columns after pruning: `{summary['retained_columns_after_pruning']}`.",
            f"Next step: {summary['selected_next_step']}",
            "",
        ]
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--teacher-steps", type=int, default=100)
    parser.add_argument("--router-steps", type=int, default=80)
    parser.add_argument("--value-steps", type=int, default=80)
    parser.add_argument("--control-steps", type=int, default=80)
    args = parser.parse_args(argv)
    summary = run_dense_teacher_support_forcing_pruning_pregate(
        selector_path=args.selector,
        out_dir=args.out,
        seed=args.seed,
        teacher_steps=args.teacher_steps,
        router_steps=args.router_steps,
        value_steps=args.value_steps,
        control_steps=args.control_steps,
    )
    print(json.dumps({key: summary[key] for key in ("status", "decision", "claim_status", "selected_next_step")}, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
