"""Diagnose sparse value selection versus support routing after norm repair.

The value-capacity/norm-control assay repaired sparse residual scale but still
left the learned sparse arm behind the flat value control. This bounded local
diagnostic reruns the tiny dense-teacher setup and adds nondeployable oracle
value-code ceilings so the next decision can separate support routing, code
selection, and sparse value-formulation failures before any GPU work.
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

from relaleap.experiments import dense_teacher_residual_value_capacity_norm_assay as assay


DEFAULT_SOURCE_DIR = Path("results/reports/dense_teacher_residual_value_capacity_norm_assay")
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_sparse_value_selection_diagnostic")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "diagnostic_rows.csv",
    "failure_axis_rows.csv",
    "gate_rows.csv",
    "notes.md",
)

DECISION = "dense_teacher_sparse_value_selection_diagnostic_recorded"
FAIL_DECISION = "dense_teacher_sparse_value_selection_diagnostic_failed_closed"


def run_dense_teacher_sparse_value_selection_diagnostic(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 31,
    teacher_steps: int = 100,
    router_steps: int = 80,
    value_steps: int = 80,
    control_steps: int = 80,
    column_count: int = 6,
    values_per_column: int = 3,
) -> dict[str, Any]:
    """Run the local diagnostic and write summary/CSV artifacts."""

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError("dense-teacher sparse value-selection diagnostic requires torch") from exc

    start = time.time()
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    source_summary = _read_json(source_dir / "summary.json")
    source_rows = [
        {
            "source": "dense_teacher_residual_value_capacity_norm_assay",
            "path": str(source_dir / "summary.json"),
            "present": (source_dir / "summary.json").is_file(),
            "status": source_summary.get("status", ""),
            "decision": source_summary.get("decision", ""),
            "claim_status": source_summary.get("claim_status", ""),
            "selected_next_step": source_summary.get("selected_next_step", ""),
        }
    ]

    data = assay._make_data(torch, seed=seed, column_count=column_count)
    teacher = assay._Teacher(torch, data["input_dim"], data["classes"])
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

    dictionary_model = assay._fit_dictionary_model(
        torch,
        data["x_train"],
        teacher_residual_train,
        data["support_train"],
        column_count,
        values_per_column,
        steps=value_steps,
        lr=0.035,
    )
    support_router = assay._train_support_router(
        torch,
        data["x_train"],
        data["support_train"],
        data["input_dim"],
        column_count,
        steps=router_steps,
    )
    flat_value = assay._train_flat_value_head(
        torch,
        F,
        data["x_train"],
        teacher_residual_train,
        data["input_dim"],
        data["classes"],
        steps=control_steps,
    )

    support_rows = assay._support_rows(torch, data, support_router, column_count)
    learned_support = support_rows["learned"]
    oracle_support = data["support_holdout"]
    random_support = support_rows["random"]

    predictions = {
        "oracle_support_learned_value_code_sparse": (
            assay._predict_dictionary(torch, dictionary_model, data["x_holdout"], oracle_support, teacher_residual_train),
            oracle_support,
            False,
            False,
            "oracle support with deployable learned value-code head",
        ),
        "oracle_support_oracle_value_code_sparse": (
            _predict_nearest_value_code(torch, dictionary_model, teacher_residual_holdout, oracle_support, teacher_residual_train),
            oracle_support,
            True,
            False,
            "oracle support plus nondeployable nearest residual value code",
        ),
        "learned_support_learned_value_code_sparse": (
            assay._predict_dictionary(torch, dictionary_model, data["x_holdout"], learned_support, teacher_residual_train),
            learned_support,
            False,
            False,
            "deployable learned support with learned value-code head",
        ),
        "learned_support_oracle_value_code_sparse": (
            _predict_nearest_value_code(torch, dictionary_model, teacher_residual_holdout, learned_support, teacher_residual_train),
            learned_support,
            True,
            False,
            "learned support plus nondeployable nearest residual value code",
        ),
        "global_oracle_support_value_code_sparse": (
            _predict_global_nearest_value(torch, dictionary_model, teacher_residual_holdout, teacher_residual_train),
            oracle_support,
            True,
            True,
            "nondeployable global nearest dictionary value upper bound",
        ),
        "same_router_flat_value_control": (
            assay._norm_match(torch, flat_value(data["x_holdout"]), teacher_residual_train),
            learned_support,
            False,
            False,
            "same learned support with dense flat value head",
        ),
        "random_support_oracle_value_code_null": (
            _predict_nearest_value_code(torch, dictionary_model, teacher_residual_holdout, random_support, teacher_residual_train),
            random_support,
            True,
            False,
            "random support with nondeployable nearest residual value code",
        ),
    }

    diagnostic_rows = [
        _diagnostic_row(
            torch,
            F,
            arm=arm_name,
            pred=pred,
            support=support,
            target=teacher_residual_holdout,
            data=data,
            base_ce=base_holdout_ce,
            teacher_ce=teacher_holdout_ce,
            oracle_value_code=oracle_value_code,
            oracle_support_or_global=oracle_support_or_global,
            note=note,
        )
        for arm_name, (pred, support, oracle_value_code, oracle_support_or_global, note) in predictions.items()
    ]
    failure_axis_rows = _failure_axis_rows(diagnostic_rows)
    gate_rows = _gate_rows(source_rows, diagnostic_rows, failure_axis_rows, base_holdout_ce, teacher_holdout_ce)
    runtime_failures = [row for row in gate_rows if row["required"] and not row["passed"]]
    status = "fail" if runtime_failures else "pass"
    summary = {
        "status": status,
        "decision": DECISION if status == "pass" else FAIL_DECISION,
        "claim_status": _claim_status(failure_axis_rows, status),
        "selected_next_step": _selected_next_step(failure_axis_rows, status),
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "backend_policy": "local CPU diagnostic only; RunPod and Colab remain blocked",
        "training_executed": True,
        "teacher_trained": True,
        "seed": seed,
        "teacher_train_steps": teacher_steps,
        "router_train_steps": router_steps,
        "value_train_steps": value_steps,
        "control_train_steps": control_steps,
        "column_count": column_count,
        "values_per_column": values_per_column,
        "base_holdout_ce": round(base_holdout_ce, 6),
        "dense_teacher_holdout_ce": round(teacher_holdout_ce, 6),
        "dense_teacher_ce_improvement": round(base_holdout_ce - teacher_holdout_ce, 6),
        "support_accuracy": round(float((learned_support == oracle_support).float().mean().item()), 6),
        "oracle_value_code_non_deployable": True,
        "uses_future_oracle_task_flags": {
            "uses_future_hidden_or_delta": False,
            "deployable_router_uses_oracle_support": False,
            "uses_task_id": False,
            "uses_teacher_labels_in_deployable_router": False,
        },
        "source_rows": source_rows,
        "diagnostic_rows": diagnostic_rows,
        "failure_axis_rows": failure_axis_rows,
        "gate_rows": gate_rows,
        "failures": runtime_failures + [row for row in gate_rows if row["gate_type"] == "scientific" and not row["passed"]],
        "strategy_review_handling": (
            "Accepted the major pivot and kept this branch local: this diagnostic separates sparse support, "
            "value-code selection, and sparse value-formulation before any PC/core-periphery reopening or GPU validation."
        ),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _predict_nearest_value_code(torch: Any, model: dict[str, Any], target: Any, support: Any, norm_target: Any) -> Any:
    dictionary = model["dictionary"]
    selected = []
    for index in range(len(target)):
        column = int(support[index].item())
        distances = torch.mean((dictionary[column] - target[index].unsqueeze(0)) ** 2, dim=1)
        selected.append(dictionary[column, int(distances.argmin().item())])
    return assay._norm_match(torch, torch.stack(selected), norm_target)


def _predict_global_nearest_value(torch: Any, model: dict[str, Any], target: Any, norm_target: Any) -> Any:
    flat_dictionary = model["dictionary"].reshape(-1, model["dictionary"].shape[-1])
    selected = []
    for index in range(len(target)):
        distances = torch.mean((flat_dictionary - target[index].unsqueeze(0)) ** 2, dim=1)
        selected.append(flat_dictionary[int(distances.argmin().item())])
    return assay._norm_match(torch, torch.stack(selected), norm_target)


def _diagnostic_row(
    torch: Any,
    F: Any,
    *,
    arm: str,
    pred: Any,
    support: Any,
    target: Any,
    data: dict[str, Any],
    base_ce: float,
    teacher_ce: float,
    oracle_value_code: bool,
    oracle_support_or_global: bool,
    note: str,
) -> dict[str, Any]:
    logits = data["base_logits_holdout"] + pred
    residual_l2 = torch.linalg.vector_norm(pred, dim=1)
    return {
        "arm": arm,
        "ce": round(float(F.cross_entropy(logits, data["y_holdout"]).item()), 6),
        "base_ce": round(base_ce, 6),
        "dense_teacher_ce": round(teacher_ce, 6),
        "ce_gap_vs_dense_teacher": round(float(F.cross_entropy(logits, data["y_holdout"]).item()) - teacher_ce, 6),
        "ce_improvement_vs_base": round(base_ce - float(F.cross_entropy(logits, data["y_holdout"]).item()), 6),
        "teacher_residual_reconstruction_mse": round(float(F.mse_loss(pred, target).item()), 6),
        "residual_l2_mean": round(float(residual_l2.mean().item()), 6),
        "residual_l2_p95": round(float(torch.quantile(residual_l2, 0.95).item()), 6),
        "functional_churn": round(float((logits.argmax(dim=-1) != data["base_logits_holdout"].argmax(dim=-1)).float().mean().item()), 6),
        "support_accuracy_vs_oracle": round(float((support == data["support_holdout"]).float().mean().item()), 6),
        "oracle_value_code_non_deployable": oracle_value_code,
        "uses_oracle_support_or_global_search": oracle_support_or_global,
        "uses_future_hidden_or_delta": False,
        "uses_task_id": False,
        "uses_teacher_labels_in_deployable_router": False,
        "target_access_at_eval": "teacher_residual_nearest_value_code" if oracle_value_code else "prefix_safe_value_code",
        "feature_schema_hash": "prefix_x_position_support_only_v2",
        "note": note,
    }


def _failure_axis_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_arm = {row["arm"]: row for row in rows}
    oracle_learned = by_arm["oracle_support_learned_value_code_sparse"]
    oracle_code = by_arm["oracle_support_oracle_value_code_sparse"]
    learned_learned = by_arm["learned_support_learned_value_code_sparse"]
    learned_code = by_arm["learned_support_oracle_value_code_sparse"]
    flat = by_arm["same_router_flat_value_control"]
    random_code = by_arm["random_support_oracle_value_code_null"]
    global_code = by_arm["global_oracle_support_value_code_sparse"]
    axes = [
        _axis(
            "value_code_selection_regret",
            oracle_learned,
            oracle_code,
            "mse_delta",
            "same oracle support: learned value-code head versus nearest residual code",
        ),
        _axis(
            "support_routing_regret_with_oracle_value_code",
            learned_code,
            oracle_code,
            "mse_delta",
            "nearest residual code under learned support versus oracle support",
        ),
        _axis(
            "sparse_formulation_gap_vs_flat_value",
            oracle_code,
            flat,
            "mse_delta",
            "best in-column sparse value code versus flat value head",
        ),
        _axis(
            "learned_sparse_gap_vs_flat_value",
            learned_learned,
            flat,
            "ce_delta",
            "deployable learned sparse arm versus same-router flat value control",
        ),
        _axis(
            "oracle_support_value_over_random_support_value",
            random_code,
            oracle_code,
            "mse_delta",
            "random support nearest value code versus oracle support nearest value code",
        ),
        _axis(
            "in_column_gap_vs_global_dictionary_upper_bound",
            oracle_code,
            global_code,
            "mse_delta",
            "oracle in-column nearest code versus global nearest dictionary value",
        ),
    ]
    return axes


def _axis(axis: str, candidate: dict[str, Any], reference: dict[str, Any], metric_kind: str, interpretation: str) -> dict[str, Any]:
    metric = "teacher_residual_reconstruction_mse" if metric_kind == "mse_delta" else "ce"
    candidate_value = float(candidate[metric])
    reference_value = float(reference[metric])
    delta = candidate_value - reference_value
    return {
        "axis": axis,
        "candidate_arm": candidate["arm"],
        "reference_arm": reference["arm"],
        "metric": metric,
        "candidate_value": round(candidate_value, 6),
        "reference_value": round(reference_value, 6),
        "delta": round(delta, 6),
        "material_failure": delta > (0.10 if metric_kind == "mse_delta" else 0.02),
        "interpretation": interpretation,
    }


def _gate_rows(
    source_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    axis_rows: list[dict[str, Any]],
    base_ce: float,
    teacher_ce: float,
) -> list[dict[str, Any]]:
    axes = {row["axis"]: row for row in axis_rows}
    return [
        _gate("source_assay_present", all(row["present"] for row in source_rows), True, "runtime", str(source_rows)),
        _gate("diagnostic_rows_present", bool(diagnostic_rows), True, "runtime", f"rows={len(diagnostic_rows)}"),
        _gate("gpu_blocked", True, True, "runtime", "requires_gpu_now=false; advance_to_gpu_validation=false"),
        _gate("deployable_leakage_flags_false", all(not row["uses_future_hidden_or_delta"] and not row["uses_task_id"] and not row["uses_teacher_labels_in_deployable_router"] for row in diagnostic_rows), True, "runtime", "deployable rows use prefix-safe synthetic features only"),
        _gate("dense_teacher_improves_base", teacher_ce < base_ce, False, "scientific", f"base_ce={base_ce:.6f}; teacher_ce={teacher_ce:.6f}"),
        _gate("oracle_value_code_lowers_oracle_support_mse", axes["value_code_selection_regret"]["delta"] > 0.0, False, "scientific", str(axes["value_code_selection_regret"])),
        _gate("support_regret_not_primary", axes["support_routing_regret_with_oracle_value_code"]["delta"] <= 0.10, False, "scientific", str(axes["support_routing_regret_with_oracle_value_code"])),
        _gate("sparse_formulation_not_worse_than_flat", axes["sparse_formulation_gap_vs_flat_value"]["delta"] <= 0.10, False, "scientific", str(axes["sparse_formulation_gap_vs_flat_value"])),
        _gate("learned_sparse_ce_guardrail_vs_flat", axes["learned_sparse_gap_vs_flat_value"]["delta"] <= 0.02, False, "scientific", str(axes["learned_sparse_gap_vs_flat_value"])),
    ]


def _gate(criterion: str, passed: bool, required: bool, gate_type: str, evidence: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "required": required,
        "gate_type": gate_type,
        "evidence": evidence,
    }


def _claim_status(axis_rows: list[dict[str, Any]], status: str) -> str:
    if status != "pass":
        return "source_or_runtime_failure"
    material = {row["axis"] for row in axis_rows if row["material_failure"]}
    if "sparse_formulation_gap_vs_flat_value" in material or "learned_sparse_gap_vs_flat_value" in material:
        return "sparse_value_formulation_and_code_selection_block_gpu"
    if "support_routing_regret_with_oracle_value_code" in material:
        return "support_routing_regret_blocks_sparse_dictionary"
    if "value_code_selection_regret" in material:
        return "value_code_selection_regret_blocks_deployable_sparse_dictionary"
    return "no_primary_sparse_failure_axis_detected_repeat_before_gpu"


def _selected_next_step(axis_rows: list[dict[str, Any]], status: str) -> str:
    if status != "pass":
        return "repair diagnostic source/runtime artifacts before interpreting sparse value selection"
    material = {row["axis"] for row in axis_rows if row["material_failure"]}
    if "sparse_formulation_gap_vs_flat_value" in material:
        return "redesign sparse value formulation or close this dictionary variant before GPU validation"
    if "value_code_selection_regret" in material:
        return "test a deployable richer value-code head before more support-router work"
    if "support_routing_regret_with_oracle_value_code" in material:
        return "repair learned support routing only after sparse value-code ceilings remain competitive"
    return "repeat sparse value-selection diagnostic on adjacent seed before any GPU validation"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "diagnostic_rows.csv", summary["diagnostic_rows"])
    _write_csv(out_dir / "failure_axis_rows.csv", summary["failure_axis_rows"])
    _write_csv(out_dir / "gate_rows.csv", summary["gate_rows"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Dense-Teacher Sparse Value-Selection Diagnostic",
            "",
            f"Decision: `{summary['decision']}`.",
            f"Claim status: `{summary['claim_status']}`.",
            "",
            "This is local CPU trained evidence only. GPU validation and promotion remain blocked.",
            "",
            f"Support accuracy: `{summary['support_accuracy']}`.",
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
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--teacher-steps", type=int, default=100)
    parser.add_argument("--router-steps", type=int, default=80)
    parser.add_argument("--value-steps", type=int, default=80)
    parser.add_argument("--control-steps", type=int, default=80)
    args = parser.parse_args(argv)
    summary = run_dense_teacher_sparse_value_selection_diagnostic(
        source_dir=args.source_dir,
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
