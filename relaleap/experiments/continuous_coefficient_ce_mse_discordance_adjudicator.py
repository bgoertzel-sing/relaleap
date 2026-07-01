"""Adjudicate continuous-coefficient CE/MSE discordance locally.

This follow-up tests whether the continuous-coefficient CE gain survives
same-objective flat controls. It is intentionally local CPU evidence only:
GPU validation remains blocked unless CE, teacher-MSE, sparsity, and scale
checks all agree.
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

from relaleap.experiments import continuous_coefficient_sparse_value_pregate as pregate
from relaleap.experiments import dense_teacher_residual_value_capacity_norm_assay as assay


DEFAULT_PREGATE = Path("results/reports/continuous_coefficient_sparse_value_pregate/summary.json")
DEFAULT_OUT_DIR = Path("results/reports/continuous_coefficient_ce_mse_discordance_adjudicator")

DECISION = "continuous_coefficient_ce_mse_discordance_adjudicator_recorded"
FAIL_DECISION = "continuous_coefficient_ce_mse_discordance_adjudicator_failed_closed"
OBJECTIVES = ("ce_only", "mse_only", "ce_mse_combined")
VARIANTS = ("raw", "norm_matched", "half_scale")
REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "objective_rows.csv",
    "scale_rows.csv",
    "coefficient_rows.csv",
    "gate_rows.csv",
    "notes.md",
)


def run_continuous_coefficient_ce_mse_discordance_adjudicator(
    *,
    pregate_path: Path = DEFAULT_PREGATE,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 31,
    teacher_steps: int = 100,
    router_steps: int = 80,
    value_steps: int = 100,
    control_steps: int = 100,
    column_count: int = 6,
    coeff_dim: int = 3,
) -> dict[str, Any]:
    """Run local objective-parity adjudication and write artifacts."""

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError("continuous-coefficient CE/MSE adjudicator requires torch") from exc

    if min(teacher_steps, router_steps, value_steps, control_steps) < 1:
        raise ValueError("all training step counts must be positive")
    if column_count < 2:
        raise ValueError("column_count must be at least 2")
    if coeff_dim < 1:
        raise ValueError("coeff_dim must be positive")

    start = time.time()
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    source = _read_json(pregate_path)
    source_rows = [_source_row(pregate_path, source)]

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

    router = assay._train_support_router(
        torch,
        data["x_train"],
        data["support_train"],
        data["input_dim"],
        column_count,
        steps=router_steps,
    )
    learned_support = assay._support_rows(torch, data, router, column_count)["learned"]
    train_support = data["support_train"]

    objective_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for objective in OBJECTIVES:
        continuous = _train_continuous_objective(
            torch,
            F,
            data,
            teacher_residual_train,
            train_support,
            column_count,
            coeff_dim,
            objective=objective,
            steps=value_steps,
        )
        flat = _train_flat_objective(
            torch,
            F,
            data,
            teacher_residual_train,
            objective=objective,
            steps=control_steps,
        )
        continuous_raw, _coeff = pregate._raw_predict_continuous(torch, continuous, data["x_holdout"], learned_support)
        flat_raw = flat(data["x_holdout"])
        predictions = {
            "continuous_coeff": continuous_raw,
            "same_router_flat": flat_raw,
        }
        for family, pred in predictions.items():
            for variant, adjusted in _scaled_predictions(torch, pred, teacher_residual_train).items():
                row = _objective_row(
                    torch,
                    F,
                    objective=objective,
                    family=family,
                    variant=variant,
                    pred=adjusted,
                    support=learned_support,
                    data=data,
                    target=teacher_residual_holdout,
                    base_ce=base_holdout_ce,
                    teacher_ce=teacher_holdout_ce,
                    column_count=column_count,
                    coeff_dim=coeff_dim,
                )
                objective_rows.append(row)
                scale_rows.append(
                    {
                        "objective": objective,
                        "family": family,
                        "variant": variant,
                        "residual_l2_mean": row["residual_l2_mean"],
                        "teacher_residual_l2_mean": row["teacher_residual_l2_mean"],
                        "residual_l2_mean_ratio_vs_teacher": row["residual_l2_mean_ratio_vs_teacher"],
                        "ce": row["ce"],
                        "teacher_residual_reconstruction_mse": row["teacher_residual_reconstruction_mse"],
                    }
                )
        coefficient_rows.extend(_coefficient_rows(torch, objective, continuous, data["x_holdout"], learned_support))

    gate_rows = _gate_rows(source_rows, objective_rows, coefficient_rows, base_holdout_ce, teacher_holdout_ce)
    runtime_failures = [row for row in gate_rows if row["required"] and not row["passed"]]
    scientific_failures = [row for row in gate_rows if row["gate_type"] == "scientific" and not row["passed"]]
    status = "fail" if runtime_failures else "pass"
    claim_status = _claim_status(status, scientific_failures)
    summary = {
        "status": status,
        "decision": DECISION if status == "pass" else FAIL_DECISION,
        "claim_status": claim_status,
        "selected_next_step": _selected_next_step(status, scientific_failures),
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "backend_policy": "local CPU adjudicator only; RunPod and Colab remain blocked",
        "training_executed": True,
        "teacher_trained": True,
        "seed": seed,
        "teacher_train_steps": teacher_steps,
        "router_train_steps": router_steps,
        "value_train_steps": value_steps,
        "control_train_steps": control_steps,
        "column_count": column_count,
        "coeff_dim": coeff_dim,
        "objectives": list(OBJECTIVES),
        "variants": list(VARIANTS),
        "base_holdout_ce": round(base_holdout_ce, 6),
        "dense_teacher_holdout_ce": round(teacher_holdout_ce, 6),
        "source_rows": source_rows,
        "objective_rows": objective_rows,
        "scale_rows": scale_rows,
        "coefficient_rows": coefficient_rows,
        "gate_rows": gate_rows,
        "failures": runtime_failures + scientific_failures,
        "strategy_review_handling": (
            "Accepted the minor GPT-5.5-Pro FIX recommendation: keep the CE-positive/MSE-negative pregate "
            "non-promotional and run this same-objective flat-control adjudicator locally with no GPU."
        ),
        "deferred_or_rejected_recommendations": [],
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _train_continuous_objective(
    torch: Any,
    F: Any,
    data: dict[str, Any],
    targets: Any,
    support: Any,
    column_count: int,
    coeff_dim: int,
    *,
    objective: str,
    steps: int,
) -> dict[str, Any]:
    model = {
        "basis": torch.nn.Parameter(torch.randn(column_count, coeff_dim, data["classes"]) * 0.08),
        "coeff_head": torch.nn.Sequential(
            torch.nn.Linear(data["input_dim"], 24),
            torch.nn.Tanh(),
            torch.nn.Linear(24, column_count * coeff_dim),
        ),
    }
    params = [model["basis"], *list(model["coeff_head"].parameters())]
    optimizer = torch.optim.AdamW(params, lr=0.018)
    for _ in range(steps):
        pred, coeff = pregate._raw_predict_continuous(torch, model, data["x_train"], support)
        loss = _objective_loss(F, objective, pred, targets, data["base_logits_train"], data["y_train"])
        loss = loss + 0.0025 * coeff.abs().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def _train_flat_objective(
    torch: Any,
    F: Any,
    data: dict[str, Any],
    targets: Any,
    *,
    objective: str,
    steps: int,
) -> Any:
    model = torch.nn.Sequential(torch.nn.Linear(data["input_dim"], 24), torch.nn.Tanh(), torch.nn.Linear(24, data["classes"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(steps):
        pred = model(data["x_train"])
        loss = _objective_loss(F, objective, pred, targets, data["base_logits_train"], data["y_train"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def _objective_loss(F: Any, objective: str, pred: Any, target: Any, base_logits: Any, y: Any) -> Any:
    ce = F.cross_entropy(base_logits + pred, y)
    mse = F.mse_loss(pred, target)
    if objective == "ce_only":
        return ce
    if objective == "mse_only":
        return mse
    if objective == "ce_mse_combined":
        return ce + 0.25 * mse
    raise ValueError(f"unknown objective: {objective}")


def _scaled_predictions(torch: Any, pred: Any, target_train: Any) -> dict[str, Any]:
    norm_matched = assay._norm_match(torch, pred, target_train)
    return {
        "raw": pred,
        "norm_matched": norm_matched,
        "half_scale": norm_matched * 0.5,
    }


def _objective_row(
    torch: Any,
    F: Any,
    *,
    objective: str,
    family: str,
    variant: str,
    pred: Any,
    support: Any,
    data: dict[str, Any],
    target: Any,
    base_ce: float,
    teacher_ce: float,
    column_count: int,
    coeff_dim: int,
) -> dict[str, Any]:
    logits = data["base_logits_holdout"] + pred
    ce = float(F.cross_entropy(logits, data["y_holdout"]).item())
    mse = float(F.mse_loss(pred, target).item())
    residual_l2 = torch.linalg.vector_norm(pred, dim=1)
    teacher_l2 = torch.linalg.vector_norm(target, dim=1)
    return {
        "objective": objective,
        "family": family,
        "variant": variant,
        "arm": f"{objective}_{family}_{variant}",
        "ce": round(ce, 6),
        "base_ce": round(base_ce, 6),
        "dense_teacher_ce": round(teacher_ce, 6),
        "ce_gap_vs_dense_teacher": round(ce - teacher_ce, 6),
        "ce_improvement_vs_base": round(base_ce - ce, 6),
        "teacher_residual_reconstruction_mse": round(mse, 6),
        "functional_churn": round(
            float((logits.argmax(dim=-1) != data["base_logits_holdout"].argmax(dim=-1)).float().mean().item()), 6
        ),
        "finite_update_commutator_proxy": round(assay._commutator_proxy(torch, pred, support), 6),
        "intervention_selectivity_proxy": round(assay._selectivity_proxy(torch, pred, target, support), 6),
        "residual_l2_mean": round(float(residual_l2.mean().item()), 6),
        "residual_l2_p95": round(float(torch.quantile(residual_l2, 0.95).item()), 6),
        "teacher_residual_l2_mean": round(float(teacher_l2.mean().item()), 6),
        "residual_l2_mean_ratio_vs_teacher": round(float((residual_l2.mean() / teacher_l2.mean().clamp_min(1e-6)).item()), 6),
        "active_params": data["classes"] * coeff_dim + coeff_dim if family == "continuous_coeff" else data["classes"],
        "stored_params": (
            column_count * coeff_dim * data["classes"]
            + data["input_dim"] * 24
            + 24 * column_count * coeff_dim
            + 24
            + column_count * coeff_dim
            if family == "continuous_coeff"
            else data["input_dim"] * 24 + 24 * data["classes"] + 24 + data["classes"]
        ),
        "oracle_support_non_deployable": False,
        "uses_future_hidden_or_delta": False,
        "uses_task_id": False,
        "uses_teacher_labels_in_deployable_router": False,
        "target_access_at_eval": "prefix_safe_learned_support",
        "feature_schema_hash": "prefix_x_position_support_only_v2",
    }


def _coefficient_rows(torch: Any, objective: str, model: dict[str, Any], x: Any, support: Any) -> list[dict[str, Any]]:
    _pred, coeff = pregate._raw_predict_continuous(torch, model, x, support)
    abs_coeff = coeff.abs()
    soft = torch.softmax(abs_coeff, dim=-1)
    entropy = -(soft * torch.log(soft.clamp_min(1e-8))).sum(dim=-1)
    return [
        {
            "objective": objective,
            "row": "aggregate",
            "coeff_abs_mean": round(float(abs_coeff.mean().item()), 6),
            "coeff_abs_p95": round(float(torch.quantile(abs_coeff, 0.95).item()), 6),
            "coeff_entropy_mean": round(float(entropy.mean().item()), 6),
            "coeff_near_zero_fraction": round(float((abs_coeff < 0.05).float().mean().item()), 6),
        }
    ]


def _gate_rows(
    source_rows: list[dict[str, Any]],
    objective_rows: list[dict[str, Any]],
    coefficient_rows: list[dict[str, Any]],
    base_ce: float,
    teacher_ce: float,
) -> list[dict[str, Any]]:
    rows = {(row["objective"], row["family"], row["variant"]): row for row in objective_rows}
    coeffs = {row["objective"]: row for row in coefficient_rows if row["row"] == "aggregate"}
    ce_cont = rows.get(("ce_only", "continuous_coeff", "norm_matched"), {})
    ce_flat = rows.get(("ce_only", "same_router_flat", "norm_matched"), {})
    mse_cont = rows.get(("mse_only", "continuous_coeff", "norm_matched"), {})
    mse_flat = rows.get(("mse_only", "same_router_flat", "norm_matched"), {})
    combined_cont = rows.get(("ce_mse_combined", "continuous_coeff", "norm_matched"), {})
    combined_flat = rows.get(("ce_mse_combined", "same_router_flat", "norm_matched"), {})
    half_cont = rows.get(("ce_only", "continuous_coeff", "half_scale"), {})
    raw_cont = rows.get(("ce_only", "continuous_coeff", "raw"), {})
    required = {(objective, family, variant) for objective in OBJECTIVES for family in ("continuous_coeff", "same_router_flat") for variant in VARIANTS}
    present = set(rows)
    return [
        _gate("pregate_source_present", all(row["present"] for row in source_rows), True, "runtime", str(source_rows)),
        _gate(
            "pregate_was_discordant",
            source_rows[0].get("claim_status") == "continuous_coeff_ce_mse_discordant_no_promotion",
            True,
            "runtime",
            str(source_rows[0]),
        ),
        _gate("required_objective_rows_present", required.issubset(present), True, "runtime", f"rows={len(objective_rows)}"),
        _gate("coefficient_rows_present", {row["objective"] for row in coefficient_rows} == set(OBJECTIVES), True, "runtime", str(coefficient_rows)),
        _gate("gpu_blocked", True, True, "runtime", "requires_gpu_now=false; promotion_allowed=false"),
        _gate(
            "deployable_leakage_flags_false",
            all(
                not row["uses_future_hidden_or_delta"]
                and not row["uses_task_id"]
                and not row["uses_teacher_labels_in_deployable_router"]
                for row in objective_rows
            ),
            True,
            "runtime",
            "no deployable row uses future hidden/delta, task id, or teacher labels",
        ),
        _gate("dense_teacher_improves_base", teacher_ce < base_ce, False, "scientific", f"base_ce={base_ce:.6f}; teacher_ce={teacher_ce:.6f}"),
        _gate(
            "ce_objective_continuous_beats_flat_ce",
            _metric(ce_cont, "ce") + 0.002 < _metric(ce_flat, "ce"),
            False,
            "scientific",
            f"continuous_ce={ce_cont.get('ce')}; flat_ce={ce_flat.get('ce')}",
        ),
        _gate(
            "combined_objective_continuous_beats_flat_ce",
            _metric(combined_cont, "ce") + 0.002 < _metric(combined_flat, "ce"),
            False,
            "scientific",
            f"continuous_ce={combined_cont.get('ce')}; flat_ce={combined_flat.get('ce')}",
        ),
        _gate(
            "mse_objective_continuous_matches_flat_mse",
            _metric(mse_cont, "teacher_residual_reconstruction_mse")
            <= _metric(mse_flat, "teacher_residual_reconstruction_mse") + 0.10,
            False,
            "scientific",
            f"continuous_mse={mse_cont.get('teacher_residual_reconstruction_mse')}; flat_mse={mse_flat.get('teacher_residual_reconstruction_mse')}",
        ),
        _gate(
            "combined_objective_continuous_matches_flat_mse",
            _metric(combined_cont, "teacher_residual_reconstruction_mse")
            <= _metric(combined_flat, "teacher_residual_reconstruction_mse") + 0.10,
            False,
            "scientific",
            f"continuous_mse={combined_cont.get('teacher_residual_reconstruction_mse')}; flat_mse={combined_flat.get('teacher_residual_reconstruction_mse')}",
        ),
        _gate(
            "scale_control_not_only_amplitude",
            _metric(half_cont, "ce") <= _metric(raw_cont, "ce") + 0.02,
            False,
            "scientific",
            f"half_scale_ce={half_cont.get('ce')}; raw_ce={raw_cont.get('ce')}",
        ),
        _gate(
            "coefficients_not_dense_like",
            min(float(row.get("coeff_near_zero_fraction", 0.0) or 0.0) for row in coeffs.values()) >= 0.05 if coeffs else False,
            False,
            "scientific",
            f"coefficients={coeffs}",
        ),
    ]


def _gate(criterion: str, passed: bool, required: bool, gate_type: str, evidence: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "required": bool(required),
        "gate_type": gate_type,
        "evidence": evidence,
    }


def _metric(row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("inf")


def _claim_status(status: str, scientific_failures: list[dict[str, Any]]) -> str:
    if status != "pass":
        return "continuous_coeff_ce_mse_adjudicator_runtime_failed"
    failed = {row["criterion"] for row in scientific_failures}
    if not failed:
        return "continuous_coeff_objective_parity_local_signal_repeat_before_gpu"
    if "ce_objective_continuous_beats_flat_ce" in failed or "combined_objective_continuous_beats_flat_ce" in failed:
        return "continuous_coeff_ce_gain_not_objective_parity_supported_no_gpu"
    if "mse_objective_continuous_matches_flat_mse" in failed or "combined_objective_continuous_matches_flat_mse" in failed:
        return "continuous_coeff_ce_mse_discordance_persists_no_promotion"
    if "coefficients_not_dense_like" in failed:
        return "continuous_coeff_dense_like_coefficients_block_promotion"
    return "continuous_coeff_adjudicator_partial_local_signal_no_gpu"


def _selected_next_step(status: str, scientific_failures: list[dict[str, Any]]) -> str:
    if status != "pass":
        return "repair continuous CE/MSE adjudicator source/runtime artifacts before interpretation"
    failed = {row["criterion"] for row in scientific_failures}
    if not failed:
        return "repeat continuous CE/MSE adjudicator on adjacent seed before any GPU validation"
    if "ce_objective_continuous_beats_flat_ce" in failed or "combined_objective_continuous_beats_flat_ce" in failed:
        return "close continuous coefficients as a flat-control-confounded CE artifact before GPU"
    if "mse_objective_continuous_matches_flat_mse" in failed or "combined_objective_continuous_matches_flat_mse" in failed:
        return "redesign continuous coefficients with sparse/scale constraints before any GPU validation"
    return "inspect remaining scale or coefficient sparsity failures before any GPU validation"


def _source_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "continuous_coefficient_sparse_value_pregate",
        "path": str(path),
        "present": bool(payload),
        "status": payload.get("status", "") if payload else "missing",
        "decision": payload.get("decision", "") if payload else "",
        "claim_status": payload.get("claim_status", "") if payload else "",
        "selected_next_step": payload.get("selected_next_step", "") if payload else "",
        "ce_mse_discordant": payload.get("ce_mse_discordant", "") if payload else "",
        "git_commit": payload.get("git_commit", "") if payload else "",
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "objective_rows.csv", summary["objective_rows"])
    _write_csv(out_dir / "scale_rows.csv", summary["scale_rows"])
    _write_csv(out_dir / "coefficient_rows.csv", summary["coefficient_rows"])
    _write_csv(out_dir / "gate_rows.csv", summary["gate_rows"])
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
            "# Continuous-Coefficient CE/MSE Discordance Adjudicator",
            "",
            f"- Status: {summary['status']}",
            f"- Decision: {summary['decision']}",
            f"- Claim status: {summary['claim_status']}",
            f"- Selected next step: {summary['selected_next_step']}",
            "- Same-objective flat controls, norm-matched rows, and half-scale rows are included.",
            "- GPU validation remains blocked: requires_gpu_now=false, promotion_allowed=false, advance_to_gpu_validation=false.",
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
    parser.add_argument("--pregate", type=Path, default=DEFAULT_PREGATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--teacher-steps", type=int, default=100)
    parser.add_argument("--router-steps", type=int, default=80)
    parser.add_argument("--value-steps", type=int, default=100)
    parser.add_argument("--control-steps", type=int, default=100)
    args = parser.parse_args(argv)
    summary = run_continuous_coefficient_ce_mse_discordance_adjudicator(
        pregate_path=args.pregate,
        out_dir=args.out,
        seed=args.seed,
        teacher_steps=args.teacher_steps,
        router_steps=args.router_steps,
        value_steps=args.value_steps,
        control_steps=args.control_steps,
    )
    print(
        json.dumps(
            {key: summary[key] for key in ("status", "decision", "claim_status", "selected_next_step")},
            indent=2,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
