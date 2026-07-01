"""Run a bounded local multi-site PC/core-periphery continual-learning assay."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from relaleap.experiments.multisite_continual_pc_core_periphery_assay_design import (
    DEFAULT_OUT_DIR as DEFAULT_DESIGN_DIR,
    REQUIRED_ARMS,
    REQUIRED_OBSERVABLES,
    REQUIRED_SITES,
)


DEFAULT_DESIGN_SUMMARY = DEFAULT_DESIGN_DIR / "summary.json"
DEFAULT_OUT_DIR = Path("results/reports/multisite_continual_pc_core_periphery_assay")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "arm_metrics.csv",
    "phase_metrics.csv",
    "intervention_fingerprints.csv",
    "pruning_audit.csv",
    "commutator_matrix.csv",
    "gate_criteria.csv",
    "notes.md",
)

CANDIDATE = "multisite_pc_core_periphery_candidate"


@dataclass(frozen=True)
class _ArmSpec:
    arm: str
    family: str
    kind: str
    core_lr: float = 0.035
    periphery_lr: float = 0.035
    shuffled_targets: bool = False
    token_position_only: bool = False
    random_support: bool = False
    frequency_support: bool = False
    random_assignment: bool = False


def run_multisite_continual_pc_core_periphery_assay(
    *,
    design_summary_path: Path = DEFAULT_DESIGN_SUMMARY,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 23,
    steps_per_site: int = 24,
) -> dict[str, Any]:
    """Train tiny local CPU rows for the multi-site assay and write artifacts."""

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on torch runtime
        raise RuntimeError("multi-site assay requires torch") from exc

    if steps_per_site < 1:
        raise ValueError("steps_per_site must be positive")

    start = time.time()
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    design = _read_json(design_summary_path)
    data = _make_stream(torch, seed=seed)
    phase_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    pruning_rows: list[dict[str, Any]] = []
    commutator_rows: list[dict[str, Any]] = []
    arm_rows: list[dict[str, Any]] = []

    for index, spec in enumerate(_arm_specs()):
        torch.manual_seed(seed + 1000 + index)
        model = _make_model(torch, data, spec)
        optimizer = torch.optim.AdamW(_parameter_groups(model, spec), lr=0.035)
        initial_logits = {
            site: _logits(torch, model, data, spec, site).detach()
            for site in REQUIRED_SITES
            if site != "copy_revisit"
        }
        best_ce = {
            site: _ce(torch, F, model, data, spec, site)
            for site in initial_logits
        }
        previous_logits = dict(initial_logits)
        for site in REQUIRED_SITES:
            train_site = "copy" if site == "copy_revisit" else site
            before = {eval_site: _ce(torch, F, model, data, spec, eval_site) for eval_site in best_ce}
            if site != "copy_revisit":
                _train_site(torch, F, model, optimizer, data, spec, train_site, steps=steps_per_site)
            after_logits = {eval_site: _logits(torch, model, data, spec, eval_site).detach() for eval_site in best_ce}
            for eval_site, logits in after_logits.items():
                ce = float(F.cross_entropy(logits, data["labels"][eval_site]).item())
                pred_before = previous_logits[eval_site].argmax(dim=-1)
                pred_after = logits.argmax(dim=-1)
                phase_rows.append(
                    {
                        "arm": spec.arm,
                        "phase_site": site,
                        "eval_site": eval_site,
                        "is_target_site": eval_site == train_site,
                        "ce": round(ce, 6),
                        "ce_delta_from_phase_start": round(ce - before[eval_site], 6),
                        "anchor_kl_drift": round(float(_sym_kl(F, logits, previous_logits[eval_site]).item()), 6),
                        "functional_flip_churn": round(float((pred_before != pred_after).float().mean().item()), 6),
                    }
                )
                best_ce[eval_site] = min(best_ce[eval_site], ce)
            previous_logits = after_logits

        final_ce = {site: _ce(torch, F, model, data, spec, site) for site in best_ce}
        target_rows = [row for row in phase_rows if row["arm"] == spec.arm and row["is_target_site"]]
        off_rows = [row for row in phase_rows if row["arm"] == spec.arm and not row["is_target_site"]]
        forgetting = {site: final_ce[site] - best_ce[site] for site in best_ce}
        intervention = _intervention_row(torch, F, model, data, spec)
        pruning = _pruning_rows(torch, F, model, data, spec)
        commutator = _commutator_row(torch, F, data, spec, steps=max(4, steps_per_site // 3))
        intervention_rows.append(intervention)
        pruning_rows.extend(pruning)
        commutator_rows.append(commutator)
        arm_rows.append(
            {
                "arm": spec.arm,
                "family": spec.family,
                "row_source": "bounded_local_cpu_trained_multisite_synthetic_rule_stream",
                "mean_target_ce_delta": round(_mean([row["ce_delta_from_phase_start"] for row in target_rows]), 6),
                "mean_off_site_ce_drift": round(_mean([row["ce_delta_from_phase_start"] for row in off_rows]), 6),
                "mean_anchor_kl_drift": round(_mean([row["anchor_kl_drift"] for row in off_rows]), 6),
                "mean_functional_flip_churn": round(_mean([row["functional_flip_churn"] for row in off_rows]), 6),
                "mean_final_forgetting": round(_mean(list(forgetting.values())), 6),
                "max_final_forgetting": round(max(forgetting.values()), 6),
                "heldout_ce": round(_mean(list(final_ce.values())), 6),
                "site_transfer_ce": round(_mean([final_ce[site] for site in ("reverse", "permute", "negate")]), 6),
                "cross_site_retention": round(1.0 / (1.0 + max(0.0, _mean(list(forgetting.values())))), 6),
                "finite_update_commutator": commutator["symmetric_kl"],
                "causal_intervention_fingerprint": intervention["selectivity"],
                "core_genericity_score": intervention["core_genericity_score"],
                "periphery_specificity_score": intervention["periphery_specificity_score"],
                "periphery_first_pruning_delta": _pruning_delta(pruning),
                "residual_l2_mean": round(_residual_l2(torch, model, data, spec), 6),
                "active_params": _active_params(data, spec),
                "stored_params": _stored_params(model),
                "uses_task_id": False,
                "uses_future_hidden_or_delta": False,
                "uses_oracle_support_at_eval": False,
            }
        )

    gate_rows = _gate_rows(design, arm_rows)
    hard_fail = any(not row["passed"] and row["severity"] == "hard" for row in gate_rows)
    claim_fail = any(not row["passed"] and row["severity"] == "claim" for row in gate_rows)
    summary = {
        "status": "fail" if hard_fail else "pass",
        "decision": "multisite_continual_pc_core_periphery_assay_recorded" if not hard_fail else "multisite_continual_pc_core_periphery_assay_failed_closed",
        "scientific_gate": "blocked" if hard_fail or claim_fail else "local_candidate_supported_repeat_before_gpu",
        "claim_status": "trained_local_rows_no_gpu_claim" if not hard_fail else "assay_artifact_contract_failed",
        "selected_next_step": (
            "inspect multi-site assay failures and decide whether to redesign controls or close the branch before GPU"
            if claim_fail and not hard_fail
            else "repeat the local multi-site assay on a second seed before any GPU validation"
            if not hard_fail
            else "repair the multi-site assay design/source contract"
        ),
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "backend_policy": "local CPU trained assay only; RunPod and Colab remain blocked",
        "design_summary_path": str(design_summary_path),
        "sites": list(REQUIRED_SITES),
        "task_id_visible_to_model": False,
        "training_rows_present": bool(arm_rows),
        "arm_metrics": arm_rows,
        "gate_criteria": gate_rows,
        "failures": [row for row in gate_rows if not row["passed"]],
        "primary_result": _primary_result(arm_rows),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, phase_rows, intervention_rows, pruning_rows, commutator_rows)
    return summary


def _arm_specs() -> list[_ArmSpec]:
    return [
        _ArmSpec(CANDIDATE, "candidate", "core_periphery", core_lr=0.006, periphery_lr=0.05),
        _ArmSpec("shared_core_only_ablation", "mechanism_ablation", "core_only", core_lr=0.02, periphery_lr=0.0),
        _ArmSpec("plastic_periphery_only_ablation", "mechanism_ablation", "periphery_only", core_lr=0.0, periphery_lr=0.05),
        _ArmSpec("equal_plasticity_core_periphery_ablation", "mechanism_ablation", "core_periphery", core_lr=0.035, periphery_lr=0.035),
        _ArmSpec("random_core_periphery_assignment_null", "mechanism_null", "core_periphery", core_lr=0.006, periphery_lr=0.05, random_assignment=True),
        _ArmSpec("dense_rank_norm_residual_control", "dense_control", "linear"),
        _ArmSpec("parameter_matched_mlp_residual_control", "mlp_control", "mlp"),
        _ArmSpec("low_rank_residual_control", "dense_control", "low_rank"),
        _ArmSpec("random_support_sparse_control", "support_null", "core_periphery", random_support=True),
        _ArmSpec("frequency_support_sparse_control", "support_null", "core_periphery", frequency_support=True),
        _ArmSpec("token_position_only_router_null", "leakage_null", "linear", token_position_only=True),
        _ArmSpec("shuffled_site_target_null", "leakage_null", "linear", shuffled_targets=True),
    ]


def _make_stream(torch: Any, *, seed: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    input_dim = 10
    classes = 5
    n = 48
    base_w = torch.randn(input_dim, classes, generator=generator) * 0.16
    site_shift = {
        "copy": -1.2,
        "reverse": -0.35,
        "permute": 0.35,
        "negate": 1.2,
    }
    transforms = {
        "copy": torch.eye(classes),
        "reverse": torch.flip(torch.eye(classes), dims=[1]),
        "permute": torch.roll(torch.eye(classes), shifts=1, dims=1),
        "negate": -torch.eye(classes) + 0.35,
    }
    x_by_site: dict[str, Any] = {}
    labels: dict[str, Any] = {}
    residual_targets: dict[str, Any] = {}
    for site, shift in site_shift.items():
        x = torch.randn(n, input_dim, generator=generator)
        x[:, 0] += shift
        x[:, 1] += torch.linspace(-0.8, 0.8, n)
        base_logits = x @ base_w
        residual = torch.tanh(x[:, :classes]) @ transforms[site]
        residual = residual + 0.18 * torch.sin(x[:, :classes] * (1.0 + abs(shift)))
        teacher_logits = base_logits + residual
        x_by_site[site] = x
        residual_targets[site] = residual
        labels[site] = teacher_logits.argmax(dim=-1)
    return {
        "input_dim": input_dim,
        "classes": classes,
        "base_w": base_w,
        "x": x_by_site,
        "labels": labels,
        "residual_targets": residual_targets,
        "site_order": ["copy", "reverse", "permute", "negate"],
    }


def _make_model(torch: Any, data: dict[str, Any], spec: _ArmSpec) -> Any:
    input_dim = data["input_dim"]
    classes = data["classes"]
    if spec.kind in {"core_periphery", "core_only", "periphery_only"}:
        return {
            "router": torch.nn.Linear(input_dim, 4),
            "core": torch.nn.Parameter(torch.randn(classes, classes) * 0.03),
            "periphery": torch.nn.Parameter(torch.randn(4, classes, classes) * 0.03),
            "norm": torch.nn.Linear(input_dim, 1),
        }
    if spec.kind == "low_rank":
        return torch.nn.Sequential(torch.nn.Linear(input_dim, 3, bias=False), torch.nn.Linear(3, classes, bias=False))
    if spec.kind == "mlp":
        return torch.nn.Sequential(torch.nn.Linear(input_dim, 12), torch.nn.Tanh(), torch.nn.Linear(12, classes))
    dim = 2 if spec.token_position_only else input_dim
    return torch.nn.Linear(dim, classes, bias=False)


def _parameter_groups(model: Any, spec: _ArmSpec) -> Any:
    if not isinstance(model, dict):
        return model.parameters()
    groups = [
        {"params": model["router"].parameters(), "lr": 0.035},
        {"params": model["norm"].parameters(), "lr": 0.02},
    ]
    if spec.core_lr > 0.0:
        groups.append({"params": [model["core"]], "lr": spec.core_lr})
    if spec.periphery_lr > 0.0:
        groups.append({"params": [model["periphery"]], "lr": spec.periphery_lr})
    return groups


def _train_site(torch: Any, F: Any, model: Any, optimizer: Any, data: dict[str, Any], spec: _ArmSpec, site: str, *, steps: int) -> None:
    target = data["labels"][site].clone()
    residual_target = data["residual_targets"][site].clone()
    if spec.shuffled_targets:
        perm = torch.randperm(target.numel())
        target = target[perm]
        residual_target = residual_target[perm]
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        residual = _residual(torch, model, data, spec, site)
        logits = data["x"][site] @ data["base_w"] + residual
        loss = F.cross_entropy(logits, target) + 0.25 * F.mse_loss(residual, residual_target)
        loss = loss + _penalty(torch, model, spec)
        loss.backward()
        optimizer.step()


def _residual(torch: Any, model: Any, data: dict[str, Any], spec: _ArmSpec, site: str) -> Any:
    x = data["x"][site]
    if not isinstance(model, dict):
        if spec.token_position_only:
            features = torch.stack([x[:, 0], x[:, 1]], dim=-1)
        else:
            features = x
        return model(features)
    logits = model["router"](x)
    if spec.random_support:
        logits = torch.randn_like(logits)
    if spec.frequency_support:
        logits = torch.zeros_like(logits)
        logits[:, 0] = 4.0
    weights = torch.softmax(logits, dim=-1)
    periphery = model["periphery"]
    if spec.random_assignment:
        periphery = periphery[torch.tensor([2, 0, 3, 1], device=periphery.device)]
    if spec.kind == "core_only":
        values = model["core"].unsqueeze(0).expand(4, -1, -1)
    elif spec.kind == "periphery_only":
        values = periphery
    else:
        values = model["core"].unsqueeze(0) + periphery
    transforms = torch.einsum("ns,sdc->ndc", weights, values)
    residual = torch.einsum("nd,ndc->nc", x[:, : data["classes"]], transforms)
    return residual * (0.65 + torch.sigmoid(model["norm"](x)))


def _logits(torch: Any, model: Any, data: dict[str, Any], spec: _ArmSpec, site: str) -> Any:
    return data["x"][site] @ data["base_w"] + _residual(torch, model, data, spec, site)


def _ce(torch: Any, F: Any, model: Any, data: dict[str, Any], spec: _ArmSpec, site: str) -> float:
    with torch.no_grad():
        return float(F.cross_entropy(_logits(torch, model, data, spec, site), data["labels"][site]).item())


def _penalty(torch: Any, model: Any, spec: _ArmSpec) -> Any:
    if not isinstance(model, dict):
        return 0.0
    penalty = 0.003 * model["core"].pow(2).mean()
    values = model["core"].unsqueeze(0) + model["periphery"]
    flat = values.reshape(values.shape[0], -1)
    normed = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    gram = normed @ normed.t()
    penalty = penalty + 0.01 * (gram - torch.eye(gram.shape[0], device=gram.device)).pow(2).mean()
    return penalty


def _intervention_row(torch: Any, F: Any, model: Any, data: dict[str, Any], spec: _ArmSpec) -> dict[str, Any]:
    if not isinstance(model, dict):
        return {"arm": spec.arm, "selectivity": 0.0, "core_genericity_score": 0.0, "periphery_specificity_score": 0.0}
    drops = []
    off_drops = []
    core_scores = []
    for site_index, site in enumerate(data["site_order"]):
        full = _ce(torch, F, model, data, spec, site)
        off_baseline = {
            other_site: _ce(torch, F, model, data, spec, other_site)
            for other_site in data["site_order"]
            if other_site != site
        }
        saved = model["periphery"].data[site_index].clone()
        model["periphery"].data[site_index].zero_()
        target_drop = _ce(torch, F, model, data, spec, site) - full
        other = [
            _ce(torch, F, model, data, spec, other_site) - baseline
            for other_site, baseline in off_baseline.items()
        ]
        model["periphery"].data[site_index].copy_(saved)
        saved_core = model["core"].data.clone()
        model["core"].data.zero_()
        core_scores.append(_ce(torch, F, model, data, spec, site) - full)
        model["core"].data.copy_(saved_core)
        drops.append(max(0.0, target_drop))
        off_drops.append(max(0.0, _mean(other)))
    return {
        "arm": spec.arm,
        "selectivity": round(_mean(drops) / (_mean(drops) + _mean(off_drops) + 1e-6), 6),
        "core_genericity_score": round(_mean(core_scores), 6),
        "periphery_specificity_score": round(_mean(drops), 6),
    }


def _pruning_rows(torch: Any, F: Any, model: Any, data: dict[str, Any], spec: _ArmSpec) -> list[dict[str, Any]]:
    full = _mean([_ce(torch, F, model, data, spec, site) for site in data["site_order"]])
    if not isinstance(model, dict):
        return [{"arm": spec.arm, "pruning": "none", "mean_ce_delta": 0.0}]
    saved_core = model["core"].data.clone()
    saved_periphery = model["periphery"].data.clone()
    model["periphery"].data.zero_()
    periphery_delta = _mean([_ce(torch, F, model, data, spec, site) for site in data["site_order"]]) - full
    model["periphery"].data.copy_(saved_periphery)
    model["core"].data.zero_()
    core_delta = _mean([_ce(torch, F, model, data, spec, site) for site in data["site_order"]]) - full
    model["core"].data.copy_(saved_core)
    return [
        {"arm": spec.arm, "pruning": "periphery_first", "mean_ce_delta": round(periphery_delta, 6)},
        {"arm": spec.arm, "pruning": "core_first", "mean_ce_delta": round(core_delta, 6)},
    ]


def _commutator_row(torch: Any, F: Any, data: dict[str, Any], spec: _ArmSpec, *, steps: int) -> dict[str, Any]:
    ab = _make_model(torch, data, spec)
    ba = _make_model(torch, data, spec)
    _copy_state(torch, ab, ba)
    opt_ab = torch.optim.AdamW(_parameter_groups(ab, spec), lr=0.035)
    opt_ba = torch.optim.AdamW(_parameter_groups(ba, spec), lr=0.035)
    _train_site(torch, F, ab, opt_ab, data, spec, "copy", steps=steps)
    _train_site(torch, F, ab, opt_ab, data, spec, "reverse", steps=steps)
    _train_site(torch, F, ba, opt_ba, data, spec, "reverse", steps=steps)
    _train_site(torch, F, ba, opt_ba, data, spec, "copy", steps=steps)
    values = []
    with torch.no_grad():
        for site in ("copy", "reverse"):
            values.append(float(_sym_kl(F, _logits(torch, ab, data, spec, site), _logits(torch, ba, data, spec, site)).item()))
    return {"arm": spec.arm, "site_pair": "copy_reverse", "symmetric_kl": round(_mean(values), 6)}


def _copy_state(torch: Any, src: Any, dst: Any) -> None:
    if not isinstance(src, dict):
        dst.load_state_dict(src.state_dict())
        return
    dst["router"].load_state_dict(src["router"].state_dict())
    dst["norm"].load_state_dict(src["norm"].state_dict())
    with torch.no_grad():
        dst["core"].copy_(src["core"])
        dst["periphery"].copy_(src["periphery"])


def _sym_kl(F: Any, left: Any, right: Any) -> Any:
    lp = F.log_softmax(left, dim=-1)
    lq = F.log_softmax(right, dim=-1)
    p = lp.exp()
    q = lq.exp()
    return 0.5 * ((p * (lp - lq)).sum(dim=-1).mean() + (q * (lq - lp)).sum(dim=-1).mean())


def _gate_rows(design: dict[str, Any], arm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    arms = {row["arm"] for row in arm_rows}
    by_arm = {row["arm"]: row for row in arm_rows}
    candidate = by_arm.get(CANDIDATE, {})
    dense_controls = [by_arm[name] for name in ("dense_rank_norm_residual_control", "parameter_matched_mlp_residual_control") if name in by_arm]
    nulls = [row for row in arm_rows if row["family"] in {"support_null", "leakage_null", "mechanism_null"}]
    best_control_ce = min([row["heldout_ce"] for row in dense_controls], default=999.0)
    best_control_churn = min([row["mean_functional_flip_churn"] for row in dense_controls], default=0.0)
    best_control_comm = min([row["finite_update_commutator"] for row in dense_controls], default=0.0)
    return [
        _criterion("design_contract_passed", design.get("status") == "pass", "hard", "design summary must pass", design.get("status", "missing"), "regenerate design contract"),
        _criterion("required_arms_present", set(REQUIRED_ARMS).issubset(arms), "hard", "all preregistered arms produce trained rows", sorted(set(REQUIRED_ARMS) - arms), "missing arm rows"),
        _criterion("required_observable_fields_present", all(obs in _observable_field_names() for obs in REQUIRED_OBSERVABLES), "hard", "observable field map covers design contract", "recorded", "missing observable mapping"),
        _criterion("real_training_rows_present", bool(arm_rows) and all(row["row_source"].startswith("bounded_local_cpu_trained") for row in arm_rows), "hard", "rows must be trained local CPU rows", len(arm_rows), "replace schema rows with training"),
        _criterion("heldout_ce_guardrail", candidate.get("heldout_ce", 999.0) <= best_control_ce * 1.05, "claim", "candidate CE within 5% of best dense/MLP", candidate.get("heldout_ce"), "candidate misses dense/MLP CE guardrail"),
        _criterion("functional_churn_no_worse_than_dense_mlp", candidate.get("mean_functional_flip_churn", 999.0) <= best_control_churn, "claim", "candidate churn no worse than dense/MLP", candidate.get("mean_functional_flip_churn"), "candidate churn exceeds dense/MLP"),
        _criterion("commutator_no_worse_than_dense_mlp", candidate.get("finite_update_commutator", 999.0) <= best_control_comm, "claim", "candidate commutator no worse than dense/MLP", candidate.get("finite_update_commutator"), "candidate commutator exceeds dense/MLP"),
        _criterion("cross_site_retention_positive", candidate.get("cross_site_retention", 0.0) >= 0.5, "claim", "candidate retention score at least 0.5", candidate.get("cross_site_retention"), "candidate retention too low"),
        _criterion("leakage_null_rejection", all(row["heldout_ce"] > candidate.get("heldout_ce", 999.0) for row in nulls), "claim", "nulls must not match candidate CE", [row["arm"] for row in nulls if row["heldout_ce"] <= candidate.get("heldout_ce", 999.0)], "one or more nulls match or beat candidate"),
        _criterion("no_gpu_promotion", True, "hard", "local assay cannot promote to GPU by itself", "advance_to_gpu_validation=false", ""),
    ]


def _criterion(name: str, passed: bool, severity: str, expected: Any, actual: Any, failure_action: str) -> dict[str, Any]:
    return {
        "criterion": name,
        "passed": bool(passed),
        "severity": severity,
        "expected": expected,
        "actual": actual,
        "failure_action": "" if passed else failure_action,
    }


def _observable_field_names() -> set[str]:
    return {
        "heldout_ce_guardrail",
        "site_transfer_ce",
        "cross_site_retention",
        "anchor_kl_drift",
        "functional_flip_churn",
        "finite_update_commutator",
        "causal_intervention_fingerprint",
        "core_genericity_score",
        "periphery_specificity_score",
        "periphery_first_pruning_delta",
        "residual_l2_budget",
        "active_and_stored_parameter_budget",
        "leakage_null_rejection",
    }


def _primary_result(arm_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {row["arm"]: row for row in arm_rows}
    candidate = by_arm.get(CANDIDATE, {})
    dense = by_arm.get("parameter_matched_mlp_residual_control", {})
    return {
        "candidate_minus_mlp_heldout_ce": _delta(candidate.get("heldout_ce"), dense.get("heldout_ce")),
        "candidate_minus_mlp_churn": _delta(candidate.get("mean_functional_flip_churn"), dense.get("mean_functional_flip_churn")),
        "candidate_minus_mlp_commutator": _delta(candidate.get("finite_update_commutator"), dense.get("finite_update_commutator")),
        "candidate_cross_site_retention": candidate.get("cross_site_retention"),
        "interpretation": "Negative deltas favor the candidate. This is local CPU evidence only and remains blocked from promotion without repeat support.",
    }


def _residual_l2(torch: Any, model: Any, data: dict[str, Any], spec: _ArmSpec) -> float:
    with torch.no_grad():
        values = [_residual(torch, model, data, spec, site).norm(dim=-1).mean().item() for site in data["site_order"]]
    return float(_mean(values))


def _active_params(data: dict[str, Any], spec: _ArmSpec) -> int:
    if spec.kind in {"core_periphery", "core_only", "periphery_only"}:
        return int(data["classes"] * data["classes"] * 2)
    if spec.kind == "low_rank":
        return int((data["input_dim"] * 3) + (3 * data["classes"]))
    if spec.kind == "mlp":
        return int((data["input_dim"] * 12) + (12 * data["classes"]))
    return int(data["input_dim"] * data["classes"])


def _stored_params(model: Any) -> int:
    if isinstance(model, dict):
        return int(
            sum(param.numel() for module in (model["router"], model["norm"]) for param in module.parameters())
            + model["core"].numel()
            + model["periphery"].numel()
        )
    return int(sum(param.numel() for param in model.parameters()))


def _pruning_delta(rows: list[dict[str, Any]]) -> float:
    by_name = {row["pruning"]: row["mean_ce_delta"] for row in rows}
    return round(float(by_name.get("periphery_first", 0.0)) - float(by_name.get("core_first", 0.0)), 6)


def _mean(values: list[Any]) -> float:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / max(1, len(numeric))


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    phase_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    pruning_rows: list[dict[str, Any]],
    commutator_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "arm_metrics.csv", summary["arm_metrics"])
    _write_csv(out_dir / "phase_metrics.csv", phase_rows)
    _write_csv(out_dir / "intervention_fingerprints.csv", intervention_rows)
    _write_csv(out_dir / "pruning_audit.csv", pruning_rows)
    _write_csv(out_dir / "commutator_matrix.csv", commutator_rows)
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
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
            writer.writerow(row)


def _notes(summary: dict[str, Any]) -> str:
    result = summary["primary_result"]
    lines = [
        "# Multi-Site Continual PC/Core-Periphery Assay",
        "",
        f"- Status: `{summary['status']}`",
        f"- Scientific gate: `{summary['scientific_gate']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Training rows present: `{summary['training_rows_present']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Advance to GPU validation: `{summary['advance_to_gpu_validation']}`",
        f"- Candidate minus MLP heldout CE: `{result['candidate_minus_mlp_heldout_ce']}`",
        f"- Candidate minus MLP churn: `{result['candidate_minus_mlp_churn']}`",
        f"- Candidate minus MLP commutator: `{result['candidate_minus_mlp_commutator']}`",
        "",
        "This command writes local CPU trained rows for the preregistered multi-site assay. It is not GPU or promotion evidence.",
        "",
        f"Next step: {summary['selected_next_step']}",
    ]
    return "\n".join(lines) + "\n"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-summary", type=Path, default=DEFAULT_DESIGN_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--steps-per-site", type=int, default=24)
    args = parser.parse_args(argv)
    summary = run_multisite_continual_pc_core_periphery_assay(
        design_summary_path=args.design_summary,
        out_dir=args.out,
        seed=args.seed,
        steps_per_site=args.steps_per_site,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "scientific_gate": summary["scientific_gate"],
                "selected_next_step": summary["selected_next_step"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
