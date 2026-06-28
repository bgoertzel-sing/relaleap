"""Local frozen-hidden-state core/periphery PC-column pilot.

This command is a bounded CPU mechanism check. It consumes the command-driven
smoke harness to create frozen base hidden/logit tensors, then trains small
residual arms on a training-only hidden-delta PC target. It is not GPU,
promotion, or default-router evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_DESIGN = Path("results/reports/core_periphery_pc_column_nonsynthetic_pilot_design/summary.json")
DEFAULT_CONFIG = Path("configs/char_validation_support_wide_hep_temporal_clipped_objective_gate.yaml")
DEFAULT_OUT_DIR = Path("results/reports/core_periphery_pc_column_nonsynthetic_pilot")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "variant_metrics.csv",
    "intervention_fingerprints.csv",
    "failed_gate_forensics.csv",
    "gate_criteria.csv",
    "hidden_state_manifest.csv",
    "notes.md",
)

REQUIRED_VARIANTS = (
    "repaired_shared_core_residual_periphery",
    "core_periphery_pc_contextual_router",
    "current_sparse_acsr_contextual_router",
    "dense_rank_norm_residual",
    "parameter_matched_causal_mlp",
    "random_support_router",
    "frequency_support_router",
    "token_position_only_router",
    "lambda_zero_residual",
    "no_core_ablation",
    "no_periphery_ablation",
    "equal_plasticity_core_periphery",
    "shuffled_core_periphery_assignment",
)

ACTIVE_CANDIDATE = "repaired_shared_core_residual_periphery"
LEGACY_CANDIDATE = "core_periphery_pc_contextual_router"


@dataclass(frozen=True)
class _VariantSpec:
    name: str
    family: str
    kind: str = "split"
    training_mode: str = "joint"
    use_core: bool = True
    use_periphery: bool = True
    core_lr_scale: float = 0.25
    periphery_lr_scale: float = 1.0
    router: str = "contextual"
    shuffled_eval: bool = False


def run_core_periphery_pc_column_nonsynthetic_pilot(
    *,
    design_path: Path = DEFAULT_DESIGN,
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path = DEFAULT_OUT_DIR,
    train_steps: int = 8,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run the local pilot and write fail-closed report artifacts."""

    start = time.time()
    design = _read_json(design_path)
    config = _read_yaml(config_path)
    preflight = _preflight_gates(design_path, design, config_path, config, train_steps)
    variant_rows: list[dict[str, Any]] = []
    fingerprint_rows: list[dict[str, Any]] = []
    forensic_rows: list[dict[str, Any]] = []
    hidden_manifest: list[dict[str, Any]] = []
    runtime_error = ""

    if all(row["passed"] for row in preflight):
        try:
            variant_rows, fingerprint_rows, hidden_manifest = _run_torch_pilot(
                config=config,
                seed=seed,
                train_steps=train_steps,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            runtime_error = f"{type(exc).__name__}: {exc}"

    gate_rows = preflight + _pilot_gates(variant_rows, fingerprint_rows, hidden_manifest, runtime_error)
    forensic_rows = _failed_gate_forensics(variant_rows, fingerprint_rows, gate_rows)
    hard_failures = [row for row in gate_rows if not row["passed"] and row["severity"] == "hard"]
    claim_failures = [row for row in gate_rows if not row["passed"] and row["severity"] != "hard"]
    status = "fail" if hard_failures else "pass"
    if hard_failures:
        decision = "core_periphery_pc_column_nonsynthetic_pilot_failed_closed"
        claim_status = "runtime_or_artifact_contract_failed"
        scientific_gate = "blocked"
        selected_next_step = "repair the local non-synthetic pilot contract before interpretation"
    elif claim_failures:
        decision = "core_periphery_pc_column_nonsynthetic_pilot_recorded_but_blocked"
        claim_status = "local_nonsynthetic_signal_insufficient_for_gpu_or_promotion"
        scientific_gate = "blocked"
        selected_next_step = (
            "inspect failed claim gates and adjust the local split mechanism before any RunPod or Colab validation"
        )
    else:
        decision = "core_periphery_pc_column_nonsynthetic_pilot_local_candidate"
        claim_status = "local_nonsynthetic_candidate_not_gpu_or_promotion_evidence"
        scientific_gate = "ready_for_local_repeat_only"
        selected_next_step = "repeat the local non-synthetic pilot on a second seed before GPU validation"

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "scientific_gate": scientific_gate,
        "requires_gpu_now": False,
        "backend_policy": "local CPU artifact gate only; RunPod/Colab remain blocked by policy",
        "design_path": str(design_path),
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "seed": _resolved_seed(config, seed),
        "train_steps": train_steps,
        "variant_metrics": variant_rows,
        "intervention_fingerprints": fingerprint_rows,
        "failed_gate_forensics": forensic_rows,
        "hidden_state_manifest": hidden_manifest,
        "gate_criteria": gate_rows,
        "failures": hard_failures + claim_failures,
        "runtime_error": runtime_error,
        "primary_result": _primary_result(variant_rows),
        "selected_next_step": selected_next_step,
        "interpretation": _interpretation(scientific_gate),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "generated_from_head": _git_commit(),
        "dirty_diff_hash": _dirty_diff_hash(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _run_torch_pilot(
    *,
    config: dict[str, Any],
    seed: int | None,
    train_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from relaleap.smoke import TinyCharTransformer, _build_batch

    torch.manual_seed(_resolved_seed(config, seed))
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    base_cfg = model_cfg.get("base", {}) if isinstance(model_cfg.get("base"), dict) else {}
    column_cfg = model_cfg.get("columns", {}) if isinstance(model_cfg.get("columns"), dict) else {}
    dataset = str(data_cfg.get("dataset", "tiny_shakespeare_char"))
    seq_len = int(data_cfg.get("seq_len", 64))
    hidden_dim = int(base_cfg.get("hidden_dim", 64))
    layers = int(base_cfg.get("layers", 2))
    num_columns = int(column_cfg.get("num_columns", 12))
    top_k = int(column_cfg.get("top_k", 2))

    inputs, targets, vocab_size = _build_batch(dataset, seq_len=seq_len, batch_size=4)
    base = TinyCharTransformer(vocab_size=vocab_size, seq_len=seq_len, hidden_dim=hidden_dim, layers=layers)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base.eval()
    with torch.no_grad():
        hidden = base.encode(inputs).detach()
        base_logits = base.decode(hidden).detach()
        base_losses = _per_token_ce(F, base_logits, targets, vocab_size).detach()

    train_mask = torch.zeros_like(targets, dtype=torch.bool)
    anchor_mask = torch.zeros_like(targets, dtype=torch.bool)
    heldout_mask = torch.zeros_like(targets, dtype=torch.bool)
    train_mask[:2, :-1] = True
    anchor_mask[2:3, :-1] = True
    heldout_mask[3:4, :-1] = True
    teacher_delta = _training_hidden_delta(torch, F, base, hidden, targets, train_mask, vocab_size)
    teacher_target = (hidden + teacher_delta).detach()

    specs = _variant_specs()
    rows: list[dict[str, Any]] = []
    fingerprints: list[dict[str, Any]] = []
    for spec in specs:
        model = _make_model(torch, nn, hidden_dim, num_columns, top_k, spec)
        optimizer = _optimizer(torch, model, spec)
        if spec.name != "lambda_zero_residual":
            _train_variant_model(F, model, optimizer, hidden, teacher_target, train_mask, train_steps, spec)
        ba_model = _make_model(torch, nn, hidden_dim, num_columns, top_k, spec)
        ba_optimizer = _optimizer(torch, ba_model, spec)
        if spec.name != "lambda_zero_residual":
            _train_variant_model(F, ba_model, ba_optimizer, hidden, teacher_target, heldout_mask, train_steps, spec)
            _train_variant_model(F, ba_model, ba_optimizer, hidden, teacher_target, train_mask, train_steps, spec)
        row = _evaluate_variant(
            F,
            base,
            model,
            ba_model,
            hidden,
            teacher_target,
            base_logits,
            base_losses,
            targets,
            vocab_size,
            train_mask,
            anchor_mask,
            heldout_mask,
            spec,
        )
        rows.append(row)
        fingerprints.extend(
            _fingerprint_rows(F, base, model, hidden, teacher_target, targets, vocab_size, anchor_mask, heldout_mask, spec)
        )

    hidden_manifest = [
        _manifest("frozen_base_hidden_train", hidden[:2], "training-only PC target construction"),
        _manifest("frozen_base_hidden_anchor", hidden[2:3], "retention/churn/anchor KL baseline"),
        _manifest("frozen_base_hidden_heldout", hidden[3:4], "heldout CE and intervention evaluation"),
        _manifest("frozen_base_logits_anchor", base_logits[2:3], "anchor KL and flip churn"),
        _manifest("token_ids", inputs, "token/position controls and supervised CE labels where allowed"),
        _manifest("positions", torch.arange(seq_len), "causal position feature"),
        _manifest("support_router_inputs", hidden.detach(), "current hidden plus causal token/position features"),
        _manifest("teacher_hidden_delta_training_only", teacher_delta[:2], "training-only local hidden-delta PC target"),
    ]
    return rows, fingerprints, hidden_manifest


def _variant_specs() -> list[_VariantSpec]:
    return [
        _VariantSpec(
            "repaired_shared_core_residual_periphery",
            "candidate",
            training_mode="shared_core_residual_periphery",
            core_lr_scale=0.18,
            periphery_lr_scale=0.35,
        ),
        _VariantSpec("core_periphery_pc_contextual_router", "legacy_candidate"),
        _VariantSpec("current_sparse_acsr_contextual_router", "sparse_control", use_periphery=False, core_lr_scale=1.0),
        _VariantSpec("dense_rank_norm_residual", "dense_control", kind="dense"),
        _VariantSpec("parameter_matched_causal_mlp", "mlp_control", kind="mlp"),
        _VariantSpec("random_support_router", "support_null", router="random"),
        _VariantSpec("frequency_support_router", "support_null", router="frequency"),
        _VariantSpec("token_position_only_router", "router_null", router="token_position"),
        _VariantSpec("lambda_zero_residual", "residual_null", use_core=False, use_periphery=False),
        _VariantSpec("no_core_ablation", "mechanism_ablation", use_core=False),
        _VariantSpec("no_periphery_ablation", "mechanism_ablation", use_periphery=False),
        _VariantSpec("equal_plasticity_core_periphery", "mechanism_ablation", core_lr_scale=1.0, periphery_lr_scale=1.0),
        _VariantSpec("shuffled_core_periphery_assignment", "mechanism_null", shuffled_eval=True),
    ]


def _training_hidden_delta(torch: Any, F: Any, base: Any, hidden: Any, targets: Any, mask: Any, vocab_size: int) -> Any:
    probe = hidden.detach().clone().requires_grad_(True)
    logits = base.decode(probe)
    losses = _per_token_ce(F, logits, targets, vocab_size)
    loss = losses[mask].mean()
    gradient = torch.autograd.grad(loss, probe)[0]
    return (-0.35 * gradient.detach()).clamp(min=-0.25, max=0.25)


def _make_model(torch: Any, nn: Any, hidden_dim: int, num_columns: int, top_k: int, spec: _VariantSpec) -> Any:
    if spec.kind == "dense":
        return _DenseResidual(nn, hidden_dim)
    if spec.kind == "mlp":
        return _MLPResidual(nn, hidden_dim)
    return _SplitResidual(torch, nn, hidden_dim, num_columns, top_k, spec)


def _SplitResidual(torch: Any, nn: Any, hidden_dim: int, num_columns: int, top_k: int, spec: _VariantSpec) -> Any:
    class SplitResidual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spec = spec
            self.num_columns = max(2, num_columns)
            self.top_k = max(1, min(top_k, self.num_columns))
            self.core = nn.Parameter(torch.zeros(hidden_dim))
            self.periphery = nn.Parameter(torch.zeros(self.num_columns, hidden_dim))
            self.router = nn.Linear(hidden_dim + 3, self.num_columns, bias=False)
            self.dummy = nn.Parameter(torch.zeros(()))
            if self.spec.training_mode == "shared_core_residual_periphery":
                core_mask = torch.ones(hidden_dim)
                periphery_mask = torch.ones(hidden_dim)
            else:
                core_mask = torch.zeros(hidden_dim)
                core_mask[: max(1, hidden_dim // 2)] = 1.0
                periphery_mask = 1.0 - core_mask
            self.register_buffer("core_mask", core_mask)
            self.register_buffer("periphery_mask", periphery_mask)
            self.initial_core = self.core.detach().clone()
            self.initial_periphery = self.periphery.detach().clone()
            self.initial_router = self.router.weight.detach().clone()
            nn.init.zeros_(self.router.weight)

        def _features(self, hidden: Any) -> Any:
            seq_len = hidden.shape[1]
            pos = torch.linspace(0.0, 1.0, seq_len, dtype=hidden.dtype, device=hidden.device).view(1, seq_len, 1)
            return torch.cat([hidden, pos.expand(hidden.shape[0], -1, -1), torch.sin(pos * 6.28318530718).expand(hidden.shape[0], -1, -1), torch.cos(pos * 6.28318530718).expand(hidden.shape[0], -1, -1)], dim=-1)

        def _indices(self, hidden: Any) -> Any:
            if self.spec.router == "random":
                flat = torch.arange(hidden.shape[0] * hidden.shape[1], device=hidden.device)
                return (flat.reshape(hidden.shape[0], hidden.shape[1]) % self.num_columns).long()
            if self.spec.router == "frequency":
                return torch.zeros(hidden.shape[:2], dtype=torch.long, device=hidden.device)
            if self.spec.router == "token_position":
                seq_len = hidden.shape[1]
                pos = torch.arange(seq_len, device=hidden.device).view(1, seq_len).expand(hidden.shape[0], -1)
                return (pos % self.num_columns).long()
            return self.router(self._features(hidden)).argmax(dim=-1)

        def forward(self, hidden: Any, ablate: str | None = None) -> Any:
            indices = self._indices(hidden)
            if self.spec.shuffled_eval:
                indices = (indices + 1) % self.num_columns
            residual = torch.zeros_like(hidden) + self.dummy * 0.0
            if self.spec.use_core and ablate != "core":
                residual = residual + self.core.view(1, 1, -1) * self.core_mask
            if self.spec.use_periphery and ablate != "periphery":
                residual = residual + self.periphery[indices] * self.periphery_mask
            return hidden + residual

        def core_parameters(self) -> list[Any]:
            return [self.core] if self.spec.use_core else []

        def periphery_parameters(self) -> list[Any]:
            return [self.periphery, self.router.weight] if self.spec.use_periphery else []

        def core_residual_norm(self) -> float:
            return float((self.core.detach() * self.core_mask).norm().item()) if self.spec.use_core else 0.0

        def periphery_residual_norm(self) -> float:
            return float((self.periphery.detach() * self.periphery_mask).norm().item()) if self.spec.use_periphery else 0.0

        def core_update_norm(self) -> float:
            return float((self.core.detach() - self.initial_core).norm().item()) if self.spec.use_core else 0.0

        def periphery_update_norm(self) -> float:
            if not self.spec.use_periphery:
                return 0.0
            periphery_delta = (self.periphery.detach() - self.initial_periphery).norm()
            router_delta = (self.router.weight.detach() - self.initial_router).norm()
            return float((periphery_delta + router_delta).item())

    return SplitResidual()


def _DenseResidual(nn: Any, hidden_dim: int) -> Any:
    class DenseResidual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.delta = nn.Parameter(__import__("torch").zeros(hidden_dim))
            self.initial = self.delta.detach().clone()

        def forward(self, hidden: Any, ablate: str | None = None) -> Any:
            if ablate in {"core", "periphery"}:
                return hidden
            return hidden + self.delta.view(1, 1, -1)

        def core_update_norm(self) -> float:
            return float((self.delta.detach() - self.initial).norm().item())

        def core_residual_norm(self) -> float:
            return float(self.delta.detach().norm().item())

        def periphery_update_norm(self) -> float:
            return 0.0

        def periphery_residual_norm(self) -> float:
            return 0.0

    return DenseResidual()


def _MLPResidual(nn: Any, hidden_dim: int) -> Any:
    class MLPResidual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim))
            for module in self.net.modules():
                if hasattr(module, "weight"):
                    nn.init.zeros_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)

        def forward(self, hidden: Any, ablate: str | None = None) -> Any:
            if ablate in {"core", "periphery"}:
                return hidden
            return hidden + self.net(hidden)

    return MLPResidual()


def _optimizer(torch: Any, model: Any, spec: _VariantSpec) -> Any:
    if spec.kind == "dense":
        return torch.optim.AdamW(model.parameters(), lr=0.03)
    if spec.kind == "mlp":
        return torch.optim.AdamW(model.parameters(), lr=0.01)
    groups = []
    if list(model.core_parameters()):
        groups.append({"params": list(model.core_parameters()), "lr": 0.025 * spec.core_lr_scale})
    if list(model.periphery_parameters()):
        groups.append({"params": list(model.periphery_parameters()), "lr": 0.025 * spec.periphery_lr_scale})
    if not groups:
        groups.append({"params": [model.dummy], "lr": 0.0})
    return torch.optim.AdamW(groups)


def _train_variant_model(
    F: Any,
    model: Any,
    optimizer: Any,
    hidden: Any,
    target: Any,
    mask: Any,
    steps: int,
    spec: _VariantSpec,
) -> None:
    if spec.kind != "split" or spec.training_mode != "shared_core_residual_periphery":
        _train_model(F, model, optimizer, hidden, target, mask, steps)
        return

    import torch

    core_params = list(model.core_parameters())
    if core_params:
        core_optimizer = torch.optim.AdamW(core_params, lr=0.025 * spec.core_lr_scale)
        for _ in range(steps):
            core_optimizer.zero_grad(set_to_none=True)
            predicted = hidden + model.core.view(1, 1, -1) * model.core_mask
            shared_target = target[mask].mean(dim=0, keepdim=True).expand_as(target[mask])
            loss = F.mse_loss(predicted[mask], shared_target)
            loss = loss + 0.0005 * model.core.pow(2).mean()
            loss.backward()
            core_optimizer.step()

    periphery_params = list(model.periphery_parameters())
    if periphery_params:
        periphery_optimizer = torch.optim.AdamW(periphery_params, lr=0.025 * spec.periphery_lr_scale)
        for _ in range(steps):
            periphery_optimizer.zero_grad(set_to_none=True)
            predicted = model(hidden)
            loss = F.mse_loss(predicted[mask], target[mask])
            loss = loss + 0.002 * (model.periphery * model.periphery_mask).pow(2).mean()
            loss.backward()
            periphery_optimizer.step()
    model.eval()


def _train_model(F: Any, model: Any, optimizer: Any, hidden: Any, target: Any, mask: Any, steps: int) -> None:
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(hidden)[mask], target[mask])
        if hasattr(model, "core"):
            loss = loss + 0.001 * model.core.pow(2).mean()
        loss.backward()
        optimizer.step()
    model.eval()


def _evaluate_variant(
    F: Any,
    base: Any,
    model: Any,
    ba_model: Any,
    hidden: Any,
    teacher_target: Any,
    base_logits: Any,
    base_losses: Any,
    targets: Any,
    vocab_size: int,
    train_mask: Any,
    anchor_mask: Any,
    heldout_mask: Any,
    spec: _VariantSpec,
) -> dict[str, Any]:
    with __import__("torch").no_grad():
        adapted = model(hidden)
        logits = base.decode(adapted)
        losses = _per_token_ce(F, logits, targets, vocab_size)
        anchor_kl = _masked_kl(F, base_logits, logits, anchor_mask)
        flip_churn = _flip_churn(base_logits, logits, anchor_mask)
        core_prune = _masked_ce_delta(F, base, model, hidden, targets, vocab_size, anchor_mask, "core")
        periphery_prune = _masked_ce_delta(F, base, model, hidden, targets, vocab_size, anchor_mask, "periphery")
        core_prune_heldout = _masked_ce_delta(F, base, model, hidden, targets, vocab_size, heldout_mask, "core")
        periphery_prune_heldout = _masked_ce_delta(F, base, model, hidden, targets, vocab_size, heldout_mask, "periphery")
        commutator = float(F.mse_loss(model(hidden)[train_mask], ba_model(hidden)[train_mask]).detach().item())
        residual = adapted - hidden
        return {
            "variant": spec.name,
            "family": spec.family,
            "train_pc_mse": float(F.mse_loss(adapted[train_mask], teacher_target[train_mask]).detach().item()),
            "anchor_ce": float(losses[anchor_mask].mean().detach().item()),
            "heldout_ce": float(losses[heldout_mask].mean().detach().item()),
            "base_anchor_ce": float(base_losses[anchor_mask].mean().detach().item()),
            "base_heldout_ce": float(base_losses[heldout_mask].mean().detach().item()),
            "ce_guardrail_delta": float((losses[heldout_mask].mean() - base_losses[heldout_mask].mean()).detach().item()),
            "anchor_kl_drift": anchor_kl,
            "flip_churn": flip_churn,
            "functional_churn": flip_churn,
            "residual_stream_churn": float(residual[anchor_mask].pow(2).mean().detach().item()),
            "finite_update_commutator": commutator,
            "residual_l2": float(residual.norm(dim=-1).mean().detach().item()),
            "core_residual_norm": float(getattr(model, "core_residual_norm", lambda: 0.0)()),
            "periphery_residual_norm": float(getattr(model, "periphery_residual_norm", lambda: 0.0)()),
            "core_update_norm": float(getattr(model, "core_update_norm", lambda: 0.0)()),
            "periphery_update_norm": float(getattr(model, "periphery_update_norm", lambda: 0.0)()),
            "plasticity_ratio": _safe_divide(float(getattr(model, "periphery_update_norm", lambda: 0.0)()), float(getattr(model, "core_update_norm", lambda: 0.0)())),
            "core_first_prune_delta": core_prune,
            "periphery_first_prune_delta": periphery_prune,
            "periphery_first_minus_core_first_prune_delta": core_prune - periphery_prune,
            "core_first_prune_delta_heldout": core_prune_heldout,
            "periphery_first_prune_delta_heldout": periphery_prune_heldout,
            "periphery_first_minus_core_first_prune_delta_heldout": core_prune_heldout - periphery_prune_heldout,
        }


def _fingerprint_rows(
    F: Any,
    base: Any,
    model: Any,
    hidden: Any,
    teacher_target: Any,
    targets: Any,
    vocab_size: int,
    anchor_mask: Any,
    heldout_mask: Any,
    spec: _VariantSpec,
) -> list[dict[str, Any]]:
    rows = []
    full_anchor = _masked_ce(F, base, model(hidden), targets, vocab_size, anchor_mask)
    full_heldout = _masked_ce(F, base, model(hidden), targets, vocab_size, heldout_mask)
    for unit in ("core", "periphery"):
        ablated = model(hidden, ablate=unit)
        ablate_anchor = _masked_ce(F, base, ablated, targets, vocab_size, anchor_mask)
        ablate_heldout = _masked_ce(F, base, ablated, targets, vocab_size, heldout_mask)
        necessity_anchor = ablate_anchor - full_anchor
        necessity_heldout = ablate_heldout - full_heldout
        rows.append(
            {
                "variant": spec.name,
                "unit": unit,
                "necessity_anchor_delta": necessity_anchor,
                "necessity_heldout_delta": necessity_heldout,
                "sufficiency_anchor_pc_mse": float(F.mse_loss(ablated[anchor_mask], teacher_target[anchor_mask]).detach().item()),
                "selectivity_delta": necessity_heldout - necessity_anchor,
                "off_target_leakage": max(0.0, necessity_anchor),
            }
        )
    return rows


def _per_token_ce(F: Any, logits: Any, targets: Any, vocab_size: int) -> Any:
    losses = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab_size), targets[:, :-1].reshape(-1), reduction="none")
    pad = __import__("torch").zeros(logits.shape[0], 1, dtype=losses.dtype, device=logits.device)
    return __import__("torch").cat([losses.reshape(logits.shape[0], logits.shape[1] - 1), pad], dim=1)


def _masked_ce(F: Any, base: Any, hidden: Any, targets: Any, vocab_size: int, mask: Any) -> float:
    return float(_per_token_ce(F, base.decode(hidden), targets, vocab_size)[mask].mean().detach().item())


def _masked_ce_delta(F: Any, base: Any, model: Any, hidden: Any, targets: Any, vocab_size: int, mask: Any, ablate: str) -> float:
    full = _masked_ce(F, base, model(hidden), targets, vocab_size, mask)
    ablated = _masked_ce(F, base, model(hidden, ablate=ablate), targets, vocab_size, mask)
    return ablated - full


def _masked_kl(F: Any, base_logits: Any, logits: Any, mask: Any) -> float:
    import torch

    base_logp = F.log_softmax(base_logits, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    probs = torch.softmax(base_logits, dim=-1)
    kl = (probs * (base_logp - logp)).sum(dim=-1)
    return float(kl[mask].mean().detach().item())


def _flip_churn(base_logits: Any, logits: Any, mask: Any) -> float:
    return float((base_logits.argmax(dim=-1)[mask] != logits.argmax(dim=-1)[mask]).to(dtype=logits.dtype).mean().detach().item())


def _manifest(name: str, tensor: Any, purpose: str) -> dict[str, Any]:
    return {
        "field": name,
        "shape": "x".join(str(dim) for dim in tensor.shape),
        "dtype": str(tensor.dtype),
        "purpose": purpose,
        "leakage_rule": "training-only targets are not used by evaluation-time routing",
        "present": True,
    }


def _preflight_gates(design_path: Path, design: dict[str, Any], config_path: Path, config: dict[str, Any], train_steps: int) -> list[dict[str, Any]]:
    return [
        _criterion("design_present", design_path.is_file(), "hard", "non-synthetic pilot design summary exists", str(design_path), "run core_periphery_pc_column_nonsynthetic_pilot_design first"),
        _criterion(
            "design_ready_for_local_implementation",
            design.get("status") == "pass" and design.get("scientific_gate") == "ready_for_local_nonsynthetic_pilot_implementation",
            "hard",
            "design explicitly permits local non-synthetic pilot implementation",
            {"status": design.get("status"), "scientific_gate": design.get("scientific_gate")},
            "repair or rerun the design report",
        ),
        _criterion("config_present", config_path.is_file() and bool(config), "hard", "pilot config is readable", str(config_path), "restore config before running pilot"),
        _criterion("train_steps_positive", train_steps >= 1, "hard", "train_steps is positive", train_steps, "use at least one local train step"),
    ]


def _pilot_gates(
    variant_rows: list[dict[str, Any]],
    fingerprint_rows: list[dict[str, Any]],
    hidden_manifest: list[dict[str, Any]],
    runtime_error: str,
) -> list[dict[str, Any]]:
    variants = {row.get("variant") for row in variant_rows}
    fields = {row.get("field") for row in hidden_manifest}
    primary = _row(variant_rows, ACTIVE_CANDIDATE)
    dense = _row(variant_rows, "dense_rank_norm_residual")
    mlp = _row(variant_rows, "parameter_matched_causal_mlp")
    null = _row(variant_rows, "lambda_zero_residual")
    return [
        _criterion("runtime_completed", not runtime_error, "hard", "torch pilot runs without exception", runtime_error or "ok", "fix runtime error"),
        _criterion("required_variants_present", set(REQUIRED_VARIANTS).issubset(variants), "hard", "candidate plus all controls are present", sorted(str(v) for v in variants), "add missing variant rows"),
        _criterion("intervention_rows_present", len(fingerprint_rows) >= 2 * len(REQUIRED_VARIANTS), "hard", "core/periphery fingerprint rows exist for every variant", len(fingerprint_rows), "regenerate intervention fingerprints"),
        _criterion(
            "hidden_state_manifest_complete",
            {"frozen_base_hidden_train", "frozen_base_hidden_anchor", "frozen_base_hidden_heldout", "frozen_base_logits_anchor", "token_ids", "positions", "support_router_inputs", "teacher_hidden_delta_training_only"}.issubset(fields),
            "hard",
            "all frozen-hidden-state contract fields are recorded",
            sorted(str(field) for field in fields),
            "record missing hidden-state contract fields",
        ),
        _criterion("ce_guardrail_not_worse_than_null", _float(primary.get("heldout_ce")) <= _float(null.get("heldout_ce")) + 0.05, "claim", "active candidate heldout CE stays within 0.05 of frozen/null residual", {"candidate_variant": ACTIVE_CANDIDATE, "candidate": primary.get("heldout_ce"), "null": null.get("heldout_ce")}, "CE guardrail blocks GPU validation"),
        _criterion("matched_dense_retention", _float(primary.get("anchor_kl_drift")) <= _float(dense.get("anchor_kl_drift")) + 1e-8, "claim", "active candidate anchor KL no worse than dense", {"candidate_variant": ACTIVE_CANDIDATE, "candidate": primary.get("anchor_kl_drift"), "dense": dense.get("anchor_kl_drift")}, "dense control remains active"),
        _criterion("matched_mlp_retention", _float(primary.get("anchor_kl_drift")) <= _float(mlp.get("anchor_kl_drift")) + 1e-8, "claim", "active candidate anchor KL no worse than MLP", {"candidate_variant": ACTIVE_CANDIDATE, "candidate": primary.get("anchor_kl_drift"), "mlp": mlp.get("anchor_kl_drift")}, "MLP control remains active"),
        _criterion("core_periphery_update_separation", _float(primary.get("plasticity_ratio")) > 1.5, "claim", "periphery update norm exceeds protected core update norm", primary.get("plasticity_ratio"), "split may be accounting-only"),
        _criterion("periphery_first_pruning_signal", _float(primary.get("periphery_first_minus_core_first_prune_delta")) > 0.0, "claim", "core pruning is more damaging than periphery pruning on anchors", primary.get("periphery_first_minus_core_first_prune_delta"), "protected core not causally distinguished"),
    ]


def _primary_result(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = _row(rows, ACTIVE_CANDIDATE)
    dense = _row(rows, "dense_rank_norm_residual")
    mlp = _row(rows, "parameter_matched_causal_mlp")
    return {
        "primary_variant": ACTIVE_CANDIDATE,
        "heldout_ce": primary.get("heldout_ce"),
        "anchor_kl_drift": primary.get("anchor_kl_drift"),
        "core_minus_dense_anchor_kl_drift": _float(primary.get("anchor_kl_drift")) - _float(dense.get("anchor_kl_drift")),
        "core_minus_mlp_anchor_kl_drift": _float(primary.get("anchor_kl_drift")) - _float(mlp.get("anchor_kl_drift")),
        "core_periphery_update_norm_ratio": primary.get("plasticity_ratio"),
        "periphery_first_minus_core_first_prune_delta": primary.get("periphery_first_minus_core_first_prune_delta"),
    }


def _failed_gate_forensics(
    variant_rows: list[dict[str, Any]],
    fingerprint_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    interesting_variants = [
        ACTIVE_CANDIDATE,
        LEGACY_CANDIDATE,
        "parameter_matched_causal_mlp",
        "dense_rank_norm_residual",
        "lambda_zero_residual",
    ]
    failed_claims = ",".join(
        row["criterion"] for row in gate_rows if row.get("severity") == "claim" and not row.get("passed")
    )
    mlp = _row(variant_rows, "parameter_matched_causal_mlp")
    dense = _row(variant_rows, "dense_rank_norm_residual")
    rows: list[dict[str, Any]] = []
    for variant in interesting_variants:
        row = _row(variant_rows, variant)
        if not row:
            continue
        rows.append(
            {
                "variant": variant,
                "failed_claims": failed_claims,
                "anchor_kl_drift": row.get("anchor_kl_drift"),
                "anchor_kl_minus_mlp": _float(row.get("anchor_kl_drift")) - _float(mlp.get("anchor_kl_drift")),
                "anchor_kl_minus_dense": _float(row.get("anchor_kl_drift")) - _float(dense.get("anchor_kl_drift")),
                "heldout_ce": row.get("heldout_ce"),
                "residual_l2": row.get("residual_l2"),
                "core_residual_norm": row.get("core_residual_norm"),
                "periphery_residual_norm": row.get("periphery_residual_norm"),
                "core_update_norm": row.get("core_update_norm"),
                "periphery_update_norm": row.get("periphery_update_norm"),
                "plasticity_ratio": row.get("plasticity_ratio"),
                "core_prune_anchor_delta": row.get("core_first_prune_delta"),
                "periphery_prune_anchor_delta": row.get("periphery_first_prune_delta"),
                "core_minus_periphery_prune_anchor_delta": row.get("periphery_first_minus_core_first_prune_delta"),
                "core_prune_heldout_delta": row.get("core_first_prune_delta_heldout"),
                "periphery_prune_heldout_delta": row.get("periphery_first_prune_delta_heldout"),
                "core_minus_periphery_prune_heldout_delta": row.get("periphery_first_minus_core_first_prune_delta_heldout"),
                "anchor_core_necessity": _fingerprint_value(fingerprint_rows, variant, "core", "necessity_anchor_delta"),
                "anchor_periphery_necessity": _fingerprint_value(fingerprint_rows, variant, "periphery", "necessity_anchor_delta"),
                "heldout_core_necessity": _fingerprint_value(fingerprint_rows, variant, "core", "necessity_heldout_delta"),
                "heldout_periphery_necessity": _fingerprint_value(fingerprint_rows, variant, "periphery", "necessity_heldout_delta"),
            }
        )
    return rows


def _fingerprint_value(rows: list[dict[str, Any]], variant: str, unit: str, field: str) -> Any:
    for row in rows:
        if row.get("variant") == variant and row.get("unit") == unit:
            return row.get(field)
    return ""


def _row(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    for row in rows:
        if row.get("variant") == variant:
            return row
    return {}


def _criterion(criterion: str, passed: bool, severity: str, expected: Any, actual: Any, failure_action: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "severity": severity,
        "expected": expected,
        "actual": actual,
        "failure_action": failure_action,
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return float("inf") if numerator > 0.0 else 0.0
    return numerator / denominator


def _float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _resolved_seed(config: dict[str, Any], seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    run_cfg = config.get("run", {}) if isinstance(config.get("run"), dict) else {}
    return int(run_cfg.get("seed", 1))


def _interpretation(scientific_gate: str) -> str:
    if scientific_gate == "ready_for_local_repeat_only":
        return (
            "The local frozen-hidden-state pilot cleared artifact, dense/MLP retention, "
            "plasticity, pruning, and CE guardrail gates. This only permits another local repeat."
        )
    return (
        "The local frozen-hidden-state pilot is recorded but does not justify GPU validation "
        "or promotion unless all claim gates pass on a repeat."
    )


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "variant_metrics.csv", summary["variant_metrics"])
    _write_csv(out_dir / "intervention_fingerprints.csv", summary["intervention_fingerprints"])
    _write_csv(out_dir / "failed_gate_forensics.csv", summary["failed_gate_forensics"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_csv(out_dir / "hidden_state_manifest.csv", summary["hidden_state_manifest"])
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("\n")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Core/Periphery PC Column Non-Synthetic Pilot",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Scientific gate: `{summary['scientific_gate']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        "",
        summary["interpretation"],
        "",
        f"Next step: {summary['selected_next_step']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _dirty_diff_hash() -> str:
    try:
        diff = subprocess.check_output(["git", "diff", "--no-ext-diff"], text=True)
    except Exception:
        return "unknown"
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-steps", type=int, default=8)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    summary = run_core_periphery_pc_column_nonsynthetic_pilot(
        design_path=args.design,
        config_path=args.config,
        out_dir=args.out,
        train_steps=args.train_steps,
        seed=args.seed,
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
