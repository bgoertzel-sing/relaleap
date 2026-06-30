"""Run a local low-churn MLP residual-control pilot from the pregate budgets."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.norm_budgeted_churn_regularized_residual_pilot import (
    _GatedMLPResidual,
    _base_prediction_flip_margin_penalty,
    _criterion,
    _float,
    _heldout_mask,
    _module_update,
    _per_token_anchor_kl,
    _per_token_ce,
)


DEFAULT_PREGATE_DIR = Path("results/reports/low_churn_mlp_residual_control_pregate")
DEFAULT_MLP_FINGERPRINT_DIR = Path("results/reports/mlp_churn_intervention_fingerprint")
DEFAULT_SPARSE_COMPARATOR_DIR = Path("results/reports/acsr_common_causal_residual_benchmark")
DEFAULT_OUT_DIR = Path("results/reports/low_churn_mlp_residual_control_pilot")

LOW_CHURN_ARM = "low_churn_mlp_residual_control"
SHUFFLED_NULL_ARM = "low_churn_mlp_shuffled_target_null"
DENSE24_ARM = "dense_rank24_reference"
RAW_MLP_ARM = "raw_parameter_matched_mlp_reference"
SCALED_MLP_ARM = "scaled_mlp_dense24_l2_reference"
SPARSE_ARM = "sparse_acsr_contextual_topk2_comparator"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "arm_metrics.csv",
    "per_token_metrics.csv",
    "pareto_rows.csv",
    "intervention_fingerprints.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_low_churn_mlp_residual_control_pilot(
    *,
    pregate_dir: Path = DEFAULT_PREGATE_DIR,
    mlp_fingerprint_dir: Path = DEFAULT_MLP_FINGERPRINT_DIR,
    sparse_comparator_dir: Path = DEFAULT_SPARSE_COMPARATOR_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    train_steps: int = 12,
    seed: int = 1,
) -> dict[str, Any]:
    """Train bounded MLP residual controls and evaluate dense/sparse references."""

    start = time.time()
    pregate = _read_json(pregate_dir / "summary.json")
    budgets = _budget_values(pregate.get("budget_rows", []))
    preflight = _preflight_rows(pregate_dir, pregate, budgets, train_steps)
    arm_rows: list[dict[str, Any]] = []
    per_token_rows: list[dict[str, Any]] = []
    fingerprint_rows: list[dict[str, Any]] = []
    runtime_error = ""

    source_reference_rows = _source_reference_rows(
        budgets=budgets,
        mlp_fingerprint_dir=mlp_fingerprint_dir,
        sparse_comparator_dir=sparse_comparator_dir,
    )
    sparse_rows = _read_csv(sparse_comparator_dir / "per_token_metrics.csv")

    if all(row["passed"] for row in preflight):
        try:
            trained_arms, token_rows, fingerprint_rows = _run_training_pilot(
                budgets=budgets,
                train_steps=train_steps,
                seed=seed,
            )
            arm_rows = source_reference_rows + trained_arms
            per_token_rows = token_rows + _sparse_token_proxy_rows(sparse_rows)
        except Exception as exc:  # pragma: no cover - depends on torch runtime
            runtime_error = f"{type(exc).__name__}: {exc}"

    pareto_rows = _pareto_rows(arm_rows)
    _add_advancement_gates(arm_rows, budgets)
    gate_rows = preflight + _pilot_gate_rows(
        arm_rows=arm_rows,
        per_token_rows=per_token_rows,
        pareto_rows=pareto_rows,
        fingerprint_rows=fingerprint_rows,
        runtime_error=runtime_error,
    )
    failures = [row for row in gate_rows if not row["passed"]]
    advancing = [row for row in arm_rows if row.get("advancement_gate") == "advances_local_review_only"]
    status = "pass" if not failures else "fail"
    selected_next_action = (
        "inspect_low_churn_mlp_pilot_per_token_rows_before_gpu"
        if advancing and status == "pass"
        else "return_to_sparse_core_periphery_mechanism_work"
    )
    summary = {
        "status": status,
        "decision": (
            "low_churn_mlp_residual_control_pilot_completed"
            if status == "pass"
            else "low_churn_mlp_residual_control_pilot_failed_closed"
        ),
        "claim_status": (
            "low_churn_mlp_has_budgeted_local_signal_needs_review"
            if advancing and status == "pass"
            else "low_churn_mlp_no_budgeted_advancement_claim"
        ),
        "promotion_allowed": False,
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "selected_next_action": selected_next_action,
        "backend_policy": "local CPU pilot only; RunPod and Colab remain blocked unless a later review selects GPU validation",
        "pregate_dir": str(pregate_dir),
        "out_dir": str(out_dir),
        "seed": seed,
        "train_steps": train_steps,
        "budgets": budgets,
        "arm_count": len(arm_rows),
        "per_token_row_count": len(per_token_rows),
        "pareto_row_count": len(pareto_rows),
        "intervention_fingerprint_row_count": len(fingerprint_rows),
        "advancement_row_count": len(advancing),
        "scientific_gate": "weak_pass_needs_review" if advancing and status == "pass" else "blocked",
        "gate_criteria": gate_rows,
        "failures": failures,
        "runtime_error": runtime_error,
        "selected_next_step": (
            "inspect local low-churn MLP per-token rows against sparse comparator before any GPU validation"
            if advancing and status == "pass"
            else "return to sparse/core-periphery mechanism work unless a follow-up review requests another matched dense null"
        ),
        "strategy_review_handling": (
            "Accepted the latest minor GPT-5.5-Pro recommendation to implement a local low-churn MLP matched-control pilot with per-token rows and fail-closed gates. No Ben notification required."
        ),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, arm_rows, per_token_rows, pareto_rows, fingerprint_rows, gate_rows)
    return summary


def _run_training_pilot(
    *,
    budgets: dict[str, float],
    train_steps: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import torch
    import torch.nn.functional as F

    from relaleap.smoke import TinyCharTransformer, _build_batch

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    inputs, targets, vocab_size = _build_batch("tiny_shakespeare_char", seq_len=32, batch_size=4)
    base = TinyCharTransformer(vocab_size=vocab_size, seq_len=32, hidden_dim=32, layers=2)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base.eval()
    with torch.no_grad():
        hidden = base.encode(inputs)
        base_logits = base.decode(hidden)
        base_losses = _per_token_ce(F, base_logits, targets, vocab_size)
    heldout_mask = _heldout_mask(base_losses.reshape(-1).shape)

    arms = [
        (LOW_CHURN_ARM, False, _GatedMLPResidual(torch.nn, 32, bottleneck=16)),
        (SHUFFLED_NULL_ARM, True, _GatedMLPResidual(torch.nn, 32, bottleneck=16)),
    ]
    arm_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    fingerprint_rows: list[dict[str, Any]] = []
    for arm, shuffled_targets, module in arms:
        train_stats = _train_low_churn_module(
            torch=torch,
            F=F,
            base=base,
            module=module,
            hidden=hidden,
            targets=targets,
            vocab_size=vocab_size,
            base_logits=base_logits,
            budgets=budgets,
            train_steps=train_steps,
            shuffled_targets=shuffled_targets,
        )
        row, rows, fingerprints = _evaluate_low_churn_module(
            torch=torch,
            F=F,
            base=base,
            module=module,
            hidden=hidden,
            targets=targets,
            vocab_size=vocab_size,
            base_logits=base_logits,
            base_losses=base_losses,
            heldout_mask=heldout_mask,
            arm=arm,
            train_steps=train_steps,
            budgets=budgets,
        )
        row.update(train_stats)
        arm_rows.append(row)
        token_rows.extend(rows)
        fingerprint_rows.extend(fingerprints)
    return arm_rows, token_rows, fingerprint_rows


def _train_low_churn_module(
    *,
    torch: Any,
    F: Any,
    base: Any,
    module: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    base_logits: Any,
    budgets: dict[str, float],
    train_steps: int,
    shuffled_targets: bool,
) -> dict[str, Any]:
    module.train()
    optimizer = torch.optim.AdamW(module.parameters(), lr=2e-3)
    flat_targets = targets[:, :-1].reshape(-1)
    if shuffled_targets:
        flat_targets = flat_targets[torch.randperm(flat_targets.numel())]
    anchor_mask = ~_heldout_mask((flat_targets.numel(),))
    l2_budget = budgets["dense24_residual_l2_ceiling"]
    anchor_budget = budgets["dense24_anchor_logit_mse_ceiling"]
    flip_budget = budgets["dense24_flip_churn_ceiling"]
    losses: list[float] = []
    l2s: list[float] = []
    anchors: list[float] = []
    flips: list[float] = []
    for _ in range(max(1, train_steps)):
        optimizer.zero_grad(set_to_none=True)
        raw_update = _module_update(module, hidden)
        update = _project_to_l2_budget(raw_update, l2_budget)
        logits = base.decode(hidden + update)
        flat_logits = logits[:, :-1, :].reshape(-1, vocab_size)
        flat_base = base_logits[:, :-1, :].reshape(-1, vocab_size)
        ce = F.cross_entropy(flat_logits, flat_targets)
        token_l2 = update[:, :-1, :].reshape(-1, update.shape[-1]).norm(dim=-1).mean()
        logit_mse = F.mse_loss(flat_logits, flat_base)
        anchor_kl = _per_token_anchor_kl(F, flat_logits, flat_base)[anchor_mask].mean()
        flip_margin = _base_prediction_flip_margin_penalty(torch, flat_logits, flat_base, anchor_mask)
        budget_loss = (
            torch.relu(token_l2 - l2_budget).pow(2)
            + torch.relu(logit_mse - anchor_budget).pow(2) * 12.0
            + torch.relu(_soft_flip_rate(torch, flat_logits, flat_base) - flip_budget).pow(2) * 4.0
        )
        loss = ce + 0.20 * logit_mse + 1.25 * anchor_kl + 0.75 * flip_margin + budget_loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
        l2s.append(float(token_l2.detach().item()))
        anchors.append(float(anchor_kl.detach().item()))
        flips.append(float(flip_margin.detach().item()))
    module.eval()
    return {
        "target_shuffled": shuffled_targets,
        "train_loss_trajectory": ";".join(f"{value:.6f}" for value in losses),
        "train_l2_trajectory": ";".join(f"{value:.6f}" for value in l2s),
        "train_anchor_kl_trajectory": ";".join(f"{value:.6f}" for value in anchors),
        "train_flip_margin_trajectory": ";".join(f"{value:.6f}" for value in flips),
    }


def _evaluate_low_churn_module(
    *,
    torch: Any,
    F: Any,
    base: Any,
    module: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    base_logits: Any,
    base_losses: Any,
    heldout_mask: Any,
    arm: str,
    train_steps: int,
    budgets: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    with torch.no_grad():
        raw_update = _module_update(module, hidden)
        raw_l2 = raw_update[:, :-1, :].reshape(-1, raw_update.shape[-1]).norm(dim=-1)
        scale = min(1.0, budgets["dense24_residual_l2_ceiling"] / max(float(raw_l2[heldout_mask].mean().item()), 1e-12))
        update = raw_update * scale
        logits = base.decode(hidden + update)
        losses = _per_token_ce(F, logits, targets, vocab_size).reshape(-1)
        base_flat = base_losses.reshape(-1)
        l2 = update[:, :-1, :].reshape(-1, update.shape[-1]).norm(dim=-1)
        flat_logits = logits[:, :-1, :].reshape(-1, vocab_size)
        flat_base = base_logits[:, :-1, :].reshape(-1, vocab_size)
        logit_mse = ((flat_logits - flat_base) ** 2).mean(dim=-1)
        anchor_kl = _per_token_anchor_kl(F, flat_logits, flat_base)
        flips = flat_logits.argmax(dim=-1) != flat_base.argmax(dim=-1)
    anchor_mask = ~heldout_mask
    row = {
        "arm": arm,
        "family": "mlp_control" if arm == LOW_CHURN_ARM else "mlp_null",
        "source": "local_training_pilot",
        "train_steps": train_steps,
        "heldout_ce_loss": float(losses[heldout_mask].mean().item()),
        "heldout_delta_vs_base_ce": float((losses[heldout_mask] - base_flat[heldout_mask]).mean().item()),
        "heldout_residual_update_l2": float(l2[heldout_mask].mean().item()),
        "heldout_logit_mse_vs_base": float(logit_mse[heldout_mask].mean().item()),
        "heldout_anchor_kl_vs_base": float(anchor_kl[heldout_mask].mean().item()),
        "heldout_prediction_flip_rate": float(flips[heldout_mask].float().mean().item()),
        "off_target_anchor_ce_loss": float(losses[anchor_mask].mean().item()),
        "off_target_anchor_kl_vs_base": float(anchor_kl[anchor_mask].mean().item()),
        "off_target_prediction_flip_rate": float(flips[anchor_mask].float().mean().item()),
        "posthoc_residual_norm_scale": scale,
        "active_params": _param_count(module),
        "stored_params": _param_count(module),
    }
    token_rows = _token_rows(arm, row["family"], losses, base_flat, l2, logit_mse, anchor_kl, flips, heldout_mask)
    fingerprints = _fingerprint_rows(arm, row, token_rows)
    return row, token_rows, fingerprints


def _project_to_l2_budget(update: Any, budget: float) -> Any:
    token_l2 = update[:, :-1, :].reshape(-1, update.shape[-1]).norm(dim=-1).mean().clamp_min(1e-12)
    return update * (budget / token_l2).clamp(max=1.0)


def _soft_flip_rate(torch: Any, logits: Any, base_logits: Any) -> Any:
    base_top = base_logits.argmax(dim=-1, keepdim=True)
    base_prob = torch.softmax(logits, dim=-1).gather(dim=-1, index=base_top).squeeze(-1)
    return 1.0 - base_prob.mean()


def _token_rows(
    arm: str,
    family: str,
    losses: Any,
    base_losses: Any,
    l2: Any,
    logit_mse: Any,
    anchor_kl: Any,
    flips: Any,
    heldout_mask: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(int(losses.numel())):
        heldout = bool(heldout_mask[index].item())
        rows.append(
            {
                "arm": arm,
                "family": family,
                "source": "local_training_pilot",
                "token_index": index,
                "split": "heldout" if heldout else "train_anchor",
                "ce_loss": float(losses[index].item()),
                "base_ce_loss": float(base_losses[index].item()),
                "delta_vs_base_ce": float(losses[index].item() - base_losses[index].item()),
                "residual_update_l2": float(l2[index].item()),
                "logit_mse_vs_base": float(logit_mse[index].item()),
                "anchor_kl_vs_base": float(anchor_kl[index].item()),
                "prediction_changed_vs_base": bool(flips[index].item()),
                "raw_intervention_available": True,
            }
        )
    return rows


def _source_reference_rows(
    *,
    budgets: dict[str, float],
    mlp_fingerprint_dir: Path,
    sparse_comparator_dir: Path,
) -> list[dict[str, Any]]:
    scaled = _read_csv(mlp_fingerprint_dir / "scaled_interventions.csv")
    sparse = _read_csv(sparse_comparator_dir / "per_token_metrics.csv")
    return [
        _scaled_reference_row(scaled, "dense_rank24_best_norm", 1.0, DENSE24_ARM, "dense_control", budgets),
        _scaled_reference_row(scaled, "parameter_matched_causal_mlp_control", 1.0, RAW_MLP_ARM, "mlp_control", budgets),
        _scaled_reference_row(scaled, "parameter_matched_causal_mlp_control", 0.25, SCALED_MLP_ARM, "mlp_control", budgets),
        _sparse_reference_row(sparse, budgets),
    ]


def _scaled_reference_row(
    rows: list[dict[str, str]],
    source_arm: str,
    scale: float,
    arm: str,
    family: str,
    budgets: dict[str, float],
) -> dict[str, Any]:
    row = next(
        (
            item
            for item in rows
            if item.get("arm") == source_arm and abs((_float(item.get("lambda")) or -1.0) - scale) < 1e-9
        ),
        {},
    )
    return {
        "arm": arm,
        "family": family,
        "source": "mlp_churn_intervention_fingerprint",
        "heldout_ce_loss": _float_or_blank(row.get("ce_loss")),
        "heldout_delta_vs_base_ce": _float_or_blank(row.get("delta_vs_base_ce")),
        "heldout_residual_update_l2": _float_or_blank(row.get("residual_update_l2")),
        "heldout_logit_mse_vs_base": _float_or_blank(row.get("logit_mse_vs_base")),
        "heldout_anchor_kl_vs_base": "",
        "heldout_prediction_flip_rate": _float_or_blank(row.get("prediction_changed_vs_base")),
        "posthoc_residual_norm_scale": scale,
        "active_params": _active_params_for_reference(arm),
        "stored_params": _active_params_for_reference(arm),
        "budget_reference": budgets.get("dense24_residual_l2_ceiling", ""),
    }


def _sparse_reference_row(rows: list[dict[str, str]], budgets: dict[str, float]) -> dict[str, Any]:
    sparse = [row for row in rows if row.get("arm") == "sparse_contextual_topk2"]
    heldout = [row for row in sparse if row.get("split") in ("heldout", "test")]
    selected = heldout or sparse
    return {
        "arm": SPARSE_ARM,
        "family": "sparse_acsr",
        "source": "acsr_common_causal_residual_benchmark",
        "heldout_ce_loss": _mean(selected, "ce_loss"),
        "heldout_delta_vs_base_ce": _mean(selected, "delta_vs_base_ce"),
        "heldout_residual_update_l2": _mean(selected, "residual_update_l2"),
        "heldout_logit_mse_vs_base": _mean(selected, "logit_mse_vs_base"),
        "heldout_anchor_kl_vs_base": "",
        "heldout_prediction_flip_rate": _mean_bool(selected, "prediction_changed_vs_base"),
        "posthoc_residual_norm_scale": "",
        "active_params": 384,
        "stored_params": 384,
        "budget_reference": budgets.get("dense24_residual_l2_ceiling", ""),
    }


def _sparse_token_proxy_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if row.get("arm") != "sparse_contextual_topk2":
            continue
        output.append(
            {
                "arm": SPARSE_ARM,
                "family": "sparse_acsr",
                "source": "acsr_common_causal_residual_benchmark",
                "token_index": row.get("token_index", ""),
                "split": row.get("split", ""),
                "ce_loss": _float_or_blank(row.get("ce_loss")),
                "base_ce_loss": _float_or_blank(row.get("base_ce_loss")),
                "delta_vs_base_ce": _float_or_blank(row.get("delta_vs_base_ce")),
                "residual_update_l2": _float_or_blank(row.get("residual_update_l2")),
                "logit_mse_vs_base": _float_or_blank(row.get("logit_mse_vs_base")),
                "anchor_kl_vs_base": "",
                "prediction_changed_vs_base": row.get("prediction_changed_vs_base", ""),
                "raw_intervention_available": bool(row.get("base_logits") and row.get("candidate_logits") and row.get("residual_update_vector")),
            }
        )
    return output


def _fingerprint_rows(arm: str, arm_row: dict[str, Any], token_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    heldout = [row for row in token_rows if row.get("split") == "heldout"]
    anchor = [row for row in token_rows if row.get("split") == "train_anchor"]
    return [
        {
            "arm": arm,
            "fingerprint": "heldout_ce_gain_vs_base",
            "value": -(_float(arm_row.get("heldout_delta_vs_base_ce")) or 0.0),
            "interpretation": "positive means the residual improves heldout CE versus the frozen base",
        },
        {
            "arm": arm,
            "fingerprint": "heldout_anchor_kl_vs_base",
            "value": _float_or_blank(arm_row.get("heldout_anchor_kl_vs_base")),
            "interpretation": "lower means less logit-distribution drift on heldout tokens",
        },
        {
            "arm": arm,
            "fingerprint": "anchor_offtarget_damage",
            "value": _mean_any(anchor, "delta_vs_base_ce"),
            "interpretation": "lower means less CE damage on off-target anchor tokens",
        },
        {
            "arm": arm,
            "fingerprint": "per_token_gain_variance",
            "value": _variance([-(_float(row.get("delta_vs_base_ce")) or 0.0) for row in heldout]),
            "interpretation": "rough raw intervention fingerprint dispersion across heldout tokens",
        },
    ]


def _pareto_rows(arm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in arm_rows:
        rows.append(
            {
                "arm": row.get("arm", ""),
                "family": row.get("family", ""),
                "heldout_ce_loss": row.get("heldout_ce_loss", ""),
                "ce_gain_vs_base": -(_float(row.get("heldout_delta_vs_base_ce")) or 0.0) if row.get("heldout_delta_vs_base_ce") != "" else "",
                "residual_update_l2": row.get("heldout_residual_update_l2", ""),
                "anchor_logit_mse": row.get("heldout_logit_mse_vs_base", ""),
                "anchor_kl": row.get("heldout_anchor_kl_vs_base", ""),
                "flip_churn": row.get("heldout_prediction_flip_rate", ""),
                "active_params": row.get("active_params", ""),
                "stored_params": row.get("stored_params", ""),
            }
        )
    return rows


def _add_advancement_gates(arm_rows: list[dict[str, Any]], budgets: dict[str, float]) -> None:
    required = {
        "dense24_residual_l2_ceiling",
        "dense24_anchor_logit_mse_ceiling",
        "dense24_flip_churn_ceiling",
        "dense24_ce_reference",
    }
    if not required.issubset(budgets):
        for row in arm_rows:
            row["passes_l2_budget"] = False
            row["passes_anchor_drift_budget"] = False
            row["passes_flip_churn_budget"] = False
            row["ce_delta_vs_dense24_reference"] = ""
            row["advancement_gate"] = "blocked_missing_budget"
        return
    dense_ce = budgets["dense24_ce_reference"]
    for row in arm_rows:
        l2 = _float(row.get("heldout_residual_update_l2"))
        mse = _float(row.get("heldout_logit_mse_vs_base"))
        flip = _float(row.get("heldout_prediction_flip_rate"))
        ce = _float(row.get("heldout_ce_loss"))
        row["passes_l2_budget"] = l2 is not None and l2 <= budgets["dense24_residual_l2_ceiling"] * 1.0001
        row["passes_nontrivial_l2_fraction"] = l2 is not None and l2 >= 0.5 * budgets["dense24_residual_l2_ceiling"]
        row["passes_anchor_drift_budget"] = (
            mse is not None and mse <= budgets["dense24_anchor_logit_mse_ceiling"] * 1.0001
        )
        row["passes_flip_churn_budget"] = flip is not None and flip <= budgets["dense24_flip_churn_ceiling"] * 1.0001
        row["ce_delta_vs_dense24_reference"] = "" if ce is None else ce - dense_ce
        row["advancement_gate"] = (
            "advances_local_review_only"
            if row.get("arm") == LOW_CHURN_ARM
            and row["passes_l2_budget"]
            and row["passes_nontrivial_l2_fraction"]
            and row["passes_anchor_drift_budget"]
            and row["passes_flip_churn_budget"]
            and ce is not None
            and ce <= dense_ce
            else "blocked_or_reference"
        )


def _preflight_rows(
    pregate_dir: Path,
    pregate: dict[str, Any],
    budgets: dict[str, float],
    train_steps: int,
) -> list[dict[str, Any]]:
    required_budgets = {
        "dense24_residual_l2_ceiling",
        "dense24_anchor_logit_mse_ceiling",
        "dense24_flip_churn_ceiling",
        "dense24_ce_reference",
    }
    return [
        _criterion(
            "pregate_passed_and_selected_pilot",
            pregate.get("status") == "pass"
            and pregate.get("selected_next_action") == "implement_low_churn_mlp_residual_control_pilot",
            "pregate must pass and select the local pilot",
            pregate.get("selected_next_action", "missing"),
            "pregate did not select this pilot",
        ),
        _criterion(
            "required_budget_values_present",
            required_budgets.issubset(budgets) and all(budgets[key] > 0.0 for key in required_budgets),
            sorted(required_budgets),
            budgets,
            "one or more dense24 budget values is missing",
        ),
        _criterion(
            "pregate_artifacts_present",
            (pregate_dir / "pregate_arms.csv").is_file() and (pregate_dir / "budget_rows.csv").is_file(),
            "pregate CSV artifacts exist",
            str(pregate_dir),
            "pregate CSV artifacts missing",
        ),
        _criterion("train_steps_bounded", 1 <= train_steps <= 32, "1 <= train_steps <= 32", train_steps, "train_steps out of bounded local range"),
    ]


def _pilot_gate_rows(
    *,
    arm_rows: list[dict[str, Any]],
    per_token_rows: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
    fingerprint_rows: list[dict[str, Any]],
    runtime_error: str,
) -> list[dict[str, Any]]:
    arms = {row.get("arm") for row in arm_rows}
    token_fields = set(per_token_rows[0]) if per_token_rows else set()
    pareto_fields = set(pareto_rows[0]) if pareto_rows else set()
    return [
        _criterion("pilot_runtime_completed", not runtime_error, "local pilot runtime completes", runtime_error or "ok", "pilot runtime failed"),
        _criterion(
            "required_arms_present",
            {DENSE24_ARM, RAW_MLP_ARM, SCALED_MLP_ARM, LOW_CHURN_ARM, SHUFFLED_NULL_ARM, SPARSE_ARM}.issubset(arms),
            "dense/raw/scaled/low-churn/null/sparse arms present",
            sorted(arms),
            "one or more required arms missing",
        ),
        _criterion(
            "per_token_sparse_and_mlp_rows_present",
            {"ce_loss", "residual_update_l2", "logit_mse_vs_base", "prediction_changed_vs_base", "raw_intervention_available"}.issubset(token_fields)
            and any(row.get("arm") == LOW_CHURN_ARM for row in per_token_rows)
            and any(row.get("arm") == SPARSE_ARM for row in per_token_rows),
            "per-token CE/L2/drift/churn/raw rows exist for MLP and sparse comparator",
            sorted(token_fields),
            "per-token comparator rows are missing",
        ),
        _criterion(
            "pareto_schema_present",
            {"heldout_ce_loss", "residual_update_l2", "anchor_logit_mse", "flip_churn", "active_params", "stored_params"}.issubset(pareto_fields),
            "Pareto rows include CE, L2, drift, churn, and parameter fields",
            sorted(pareto_fields),
            "Pareto schema incomplete",
        ),
        _criterion(
            "raw_intervention_fingerprints_present",
            len(fingerprint_rows) >= 8,
            "low-churn and shuffled-null fingerprint rows exist",
            len(fingerprint_rows),
            "raw intervention fingerprint rows missing",
        ),
        _criterion(
            "budget_gates_fail_closed",
            any(row.get("arm") == LOW_CHURN_ARM and row.get("advancement_gate") in {"advances_local_review_only", "blocked_or_reference"} for row in arm_rows),
            "low-churn arm has explicit advancement gate",
            [row for row in arm_rows if row.get("arm") == LOW_CHURN_ARM],
            "low-churn budget gate missing",
        ),
    ]


def _budget_values(rows: Any) -> dict[str, float]:
    budgets: dict[str, float] = {}
    if not isinstance(rows, list):
        return budgets
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = _float(row.get("value"))
        if value is not None:
            budgets[str(row.get("metric"))] = value
    return budgets


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
    arm_rows: list[dict[str, Any]],
    per_token_rows: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
    fingerprint_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "arm_metrics.csv", arm_rows)
    _write_csv(out_dir / "per_token_metrics.csv", per_token_rows)
    _write_csv(out_dir / "pareto_rows.csv", pareto_rows)
    _write_csv(out_dir / "intervention_fingerprints.csv", fingerprint_rows)
    _write_csv(out_dir / "gate_criteria.csv", gate_rows)
    lines = [
        "# Low-Churn MLP Residual-Control Pilot",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Promotion allowed: `{summary['promotion_allowed']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Scientific gate: `{summary['scientific_gate']}`",
        f"- Advancement rows: `{summary['advancement_row_count']}`",
        f"- Next step: {summary['selected_next_step']}",
        "",
        "This bounded local CPU pilot trains a low-churn MLP residual control and a shuffled-target null under the dense24 L2, drift, and flip-churn budgets from the pregate. It also carries forward dense/raw/scaled MLP references plus a sparse ACSR per-token comparator. Budget failures block advancement and promotion.",
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, str]], field: str) -> float | str:
    values = [_float(row.get(field)) for row in rows]
    real = [value for value in values if value is not None]
    return sum(real) / len(real) if real else ""


def _mean_any(rows: list[dict[str, Any]], field: str) -> float | str:
    values = [_float(row.get(field)) for row in rows]
    real = [value for value in values if value is not None]
    return sum(real) / len(real) if real else ""


def _mean_bool(rows: list[dict[str, str]], field: str) -> float | str:
    values = [row.get(field, "").lower() for row in rows]
    real = [value in {"true", "1", "yes"} for value in values if value]
    return sum(1 for value in real if value) / len(real) if real else ""


def _variance(values: list[float]) -> float | str:
    if not values:
        return ""
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _float_or_blank(value: Any) -> float | str:
    parsed = _float(value)
    return "" if parsed is None else parsed


def _param_count(module: Any) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _active_params_for_reference(arm: str) -> int | str:
    if arm == DENSE24_ARM:
        return 1536
    if arm in {RAW_MLP_ARM, SCALED_MLP_ARM}:
        return 1072
    return ""


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pregate-dir", type=Path, default=DEFAULT_PREGATE_DIR)
    parser.add_argument("--mlp-fingerprint-dir", type=Path, default=DEFAULT_MLP_FINGERPRINT_DIR)
    parser.add_argument("--sparse-comparator-dir", type=Path, default=DEFAULT_SPARSE_COMPARATOR_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    summary = run_low_churn_mlp_residual_control_pilot(
        pregate_dir=args.pregate_dir,
        mlp_fingerprint_dir=args.mlp_fingerprint_dir,
        sparse_comparator_dir=args.sparse_comparator_dir,
        out_dir=args.out,
        train_steps=args.train_steps,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
