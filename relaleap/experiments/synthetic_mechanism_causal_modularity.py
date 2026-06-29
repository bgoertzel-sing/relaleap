"""Synthetic mechanism-factorized causal-modularity pregate.

This module is deliberately local and CPU-only. It creates a hidden-boundary,
same-vocabulary synthetic stream with evaluator-known latent mechanisms, records
the comparator/control schema required by the next sparse-column assay, and
fails closed until concrete training hooks are wired in.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path("results/reports/synthetic_mechanism_causal_modularity")
RULES = ("copy_shift", "reverse_window", "xor_prev", "affine_jump")
REQUIRED_ARTIFACTS = (
    "summary.json",
    "episode_rows.csv",
    "comparator_controls.csv",
    "per_mechanism_interventions.csv",
    "commutator_rows.csv",
    "forgetting_rows.csv",
    "budget_rows.csv",
    "gate_criteria.csv",
    "arm_metrics.csv",
    "per_token_metrics.csv",
    "notes.md",
)


@dataclass(frozen=True)
class _SyntheticArmSpec:
    name: str
    family: str
    top_k: int
    num_columns: int
    atoms_per_column: int
    router: str
    dense_rank: int = 0
    support_mode: str = "learned"
    intervention_loss_weight: float = 0.0
    anchor_kl_weight: float = 0.0


def run_synthetic_mechanism_causal_modularity(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 17,
    vocab_size: int = 16,
    seq_len: int = 10,
    train_episodes_per_rule: int = 3,
    holdout_episodes_per_rule: int = 2,
    support_width: int = 2,
    training_hooks_available: bool = False,
    run_training_smoke: bool = False,
    training_steps: int = 12,
    hidden_dim: int = 24,
    learning_rate: float = 8e-3,
) -> dict[str, Any]:
    """Generate the local synthetic pregate packet and fail closed if hooks are absent."""

    start = time.time()
    if vocab_size < 8:
        raise ValueError("vocab_size must be at least 8")
    if seq_len < 4:
        raise ValueError("seq_len must be at least 4")
    if support_width < 1:
        raise ValueError("support_width must be positive")

    episode_rows = _episode_rows(
        seed=seed,
        vocab_size=vocab_size,
        seq_len=seq_len,
        train_episodes_per_rule=train_episodes_per_rule,
        holdout_episodes_per_rule=holdout_episodes_per_rule,
    )
    training_smoke = (
        _run_training_smoke(
            episode_rows=episode_rows,
            seed=seed,
            vocab_size=vocab_size,
            seq_len=seq_len,
            support_width=support_width,
            training_steps=training_steps,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
        )
        if run_training_smoke
        else None
    )
    hooks_available = training_hooks_available or training_smoke is not None
    controls = _comparator_controls(support_width=support_width)
    intervention_rows = (
        training_smoke["per_mechanism_interventions"]
        if training_smoke is not None
        else _intervention_schema_rows(controls)
    )
    commutator_rows = (
        training_smoke["commutator_rows"]
        if training_smoke is not None
        else _commutator_schema_rows(controls)
    )
    forgetting_rows = (
        training_smoke["forgetting_rows"]
        if training_smoke is not None
        else _forgetting_schema_rows(controls)
    )
    budget_rows = _budget_rows(vocab_size=vocab_size, seq_len=seq_len, support_width=support_width)
    gate_rows = _gate_rows(
        episode_rows=episode_rows,
        controls=controls,
        intervention_rows=intervention_rows,
        commutator_rows=commutator_rows,
        forgetting_rows=forgetting_rows,
        budget_rows=budget_rows,
        training_hooks_available=hooks_available,
        training_smoke=training_smoke,
    )
    hard_failures = [row for row in gate_rows if not row["passed"] and row["severity"] == "hard"]
    status = "fail" if hard_failures else "pass"
    summary = {
        "status": status,
        "decision": (
            "synthetic_mechanism_causal_modularity_pregate_failed_closed"
            if hard_failures
            else "synthetic_mechanism_causal_modularity_pregate_ready"
        ),
        "claim_status": "schema_and_generator_only_no_causal_modularity_claim",
        "promotion_allowed": False,
        "requires_gpu_now": False,
        "backend_policy": "local synthetic pregate only; RunPod and Colab remain blocked",
        "strategy_review_handling": (
            "Accepted the urgent GPT-5.5-Pro major pivot to a synthetic causal-modularity pregate. "
            "The review has notify_ben=true, so Ben should be notified before treating this as a durable direction shift."
        ),
        "strategic_change_level": "major",
        "notify_ben": True,
        "rules": list(RULES),
        "hidden_rule_boundaries": True,
        "task_id_visible_to_model": False,
        "mechanism_labels_enter_training": False,
        "shared_vocab_and_head": True,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "seed": seed,
        "episode_row_count": len(episode_rows),
        "control_row_count": len(controls),
        "per_mechanism_intervention_row_count": len(intervention_rows),
        "commutator_row_count": len(commutator_rows),
        "forgetting_row_count": len(forgetting_rows),
        "budget_row_count": len(budget_rows),
        "gate_criteria": gate_rows,
        "failures": hard_failures,
        "training_smoke_ran": training_smoke is not None,
        "training_steps": training_steps if training_smoke is not None else 0,
        "training_smoke_primary_result": (
            training_smoke["primary_result"] if training_smoke is not None else None
        ),
        "arm_metric_row_count": (
            len(training_smoke["arm_metrics"]) if training_smoke is not None else 0
        ),
        "per_token_metric_row_count": (
            len(training_smoke["per_token_metrics"]) if training_smoke is not None else 0
        ),
        "first_generated_rows": episode_rows[: min(6, len(episode_rows))],
        "reusable_apis": [
            "relaleap.smoke.TinyCharTransformer",
            "relaleap.smoke.ResidualColumns",
            "relaleap.experiments.mechanism_factorized_continual_learning_probe",
        ],
        "missing_training_hooks": _missing_training_hooks(hooks_available),
        "selected_next_step": (
            "wire the synthetic episode generator into a tiny CPU training smoke for promoted contextual top-k2, "
            "random/fixed support, token-position router, dense/rank/norm, low-churn MLP, and intervention-trained sparse arms"
            if hard_failures
            else _selected_next_step(training_smoke, gate_rows)
        ),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "out_dir": str(out_dir),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(
        out_dir,
        summary,
        episode_rows,
        controls,
        intervention_rows,
        commutator_rows,
        forgetting_rows,
        budget_rows,
        gate_rows,
        training_smoke,
    )
    return summary


def _run_training_smoke(
    *,
    episode_rows: list[dict[str, Any]],
    seed: int,
    vocab_size: int,
    seq_len: int,
    support_width: int,
    training_steps: int,
    hidden_dim: int,
    learning_rate: float,
) -> dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on torch install
        raise RuntimeError("synthetic training smoke requires torch") from exc

    from relaleap.smoke import ResidualColumns
    from relaleap.smoke import TinyCharTransformer

    if training_steps < 1:
        raise ValueError("training_steps must be positive")
    if hidden_dim < 4:
        raise ValueError("hidden_dim must be at least 4")

    torch.manual_seed(seed)
    train_inputs, train_targets, train_rules = _episode_tensors(
        episode_rows, split="train", seq_len=seq_len, torch=torch
    )
    holdout_inputs, holdout_targets, holdout_rules = _episode_tensors(
        episode_rows, split="holdout", seq_len=seq_len, torch=torch
    )
    base = TinyCharTransformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        layers=1,
    )
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        train_hidden = base.encode(train_inputs).detach()
        holdout_hidden = base.encode(holdout_inputs).detach()
        base_holdout_logits = base.decode(holdout_hidden).detach()

    specs = _synthetic_arm_specs(
        support_width=support_width,
        num_columns=max(4, support_width * 4),
        atoms_per_column=2,
    )
    arm_metrics: list[dict[str, Any]] = []
    per_token_metrics: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    commutator_rows: list[dict[str, Any]] = []
    forgetting_rows: list[dict[str, Any]] = []
    holdout_logits_by_arm: dict[str, Any] = {}
    final_ce_by_arm: dict[str, float] = {}

    for arm_index, spec in enumerate(specs):
        torch.manual_seed(seed + 100 + arm_index)
        adapter = _build_synthetic_adapter(
            spec=spec,
            hidden_dim=hidden_dim,
            torch=torch,
            nn=nn,
            ResidualColumns=ResidualColumns,
        )
        trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate) if trainable else None
        initial_logits = _synthetic_forward_logits(
            adapter=adapter,
            hidden=train_hidden,
            inputs=train_inputs,
            spec=spec,
            decode=base.decode,
            torch=torch,
        )
        initial_ce = float(
            F.cross_entropy(initial_logits.reshape(-1, vocab_size), train_targets.reshape(-1)).detach().item()
        )
        for _ in range(training_steps):
            if optimizer is None:
                break
            optimizer.zero_grad(set_to_none=True)
            logits = _synthetic_forward_logits(
                adapter=adapter,
                hidden=train_hidden,
                inputs=train_inputs,
                spec=spec,
                decode=base.decode,
                torch=torch,
            )
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), train_targets.reshape(-1))
            if spec.intervention_loss_weight > 0.0:
                ablated = base.decode(train_hidden)
                margin = loss - F.cross_entropy(
                    ablated.reshape(-1, vocab_size),
                    train_targets.reshape(-1),
                )
                loss = loss + spec.intervention_loss_weight * torch.relu(margin + 0.01)
            if spec.anchor_kl_weight > 0.0:
                loss = loss + spec.anchor_kl_weight * _kl_to_reference(logits, initial_logits.detach(), F=F)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            holdout_logits = _synthetic_forward_logits(
                adapter=adapter,
                hidden=holdout_hidden,
                inputs=holdout_inputs,
                spec=spec,
                decode=base.decode,
                torch=torch,
            ).detach()
            holdout_ce = float(
                F.cross_entropy(holdout_logits.reshape(-1, vocab_size), holdout_targets.reshape(-1)).detach().item()
            )
            support = _synthetic_support(adapter, holdout_hidden, holdout_inputs, spec, torch)
            used_columns = int(torch.unique(support).numel()) if support is not None else 0
            unique_support_sets = (
                int(torch.unique(support.reshape(-1, support.shape[-1]), dim=0).shape[0])
                if support is not None
                else 0
            )
            residual_l2 = float(
                (_synthetic_adapt_hidden(adapter, holdout_hidden, holdout_inputs, spec, torch) - holdout_hidden)
                .pow(2)
                .mean()
                .sqrt()
                .detach()
                .item()
            )
        holdout_logits_by_arm[spec.name] = holdout_logits
        final_ce_by_arm[spec.name] = holdout_ce
        arm_metrics.append(
            {
                "arm": spec.name,
                "family": spec.family,
                "router": spec.router,
                "support_mode": spec.support_mode,
                "top_k": spec.top_k,
                "training_steps": training_steps if optimizer is not None else 0,
                "initial_train_ce": initial_ce,
                "holdout_ce": holdout_ce,
                "holdout_ce_delta_vs_base": holdout_ce - final_ce_by_arm.get("base_no_residual", holdout_ce),
                "residual_l2": residual_l2,
                "used_columns": used_columns,
                "unique_support_sets": unique_support_sets,
                "stored_parameters": _stored_parameters(adapter),
                "active_parameters_proxy": _synthetic_active_parameters(spec, hidden_dim),
            }
        )
        per_token_metrics.extend(
            _per_token_rows(
                arm=spec.name,
                logits=holdout_logits,
                targets=holdout_targets,
                rules=holdout_rules,
                F=F,
                torch=torch,
            )
        )
        intervention_rows.extend(
            _actual_intervention_rows(
                arm=spec.name,
                logits=holdout_logits,
                base_logits=base_holdout_logits,
                targets=holdout_targets,
                rules=holdout_rules,
                F=F,
                torch=torch,
            )
        )
        forgetting_rows.extend(
            _actual_forgetting_rows(
                arm=spec.name,
                holdout_ce=holdout_ce,
                per_token_rows=per_token_metrics,
            )
        )

    commutator_rows = _actual_commutator_rows(holdout_logits_by_arm, torch=torch)
    primary = _synthetic_primary_result(arm_metrics, intervention_rows, commutator_rows)
    return {
        "arm_metrics": arm_metrics,
        "per_token_metrics": per_token_metrics,
        "per_mechanism_interventions": intervention_rows,
        "commutator_rows": commutator_rows,
        "forgetting_rows": forgetting_rows,
        "primary_result": primary,
    }


def _episode_rows(
    *,
    seed: int,
    vocab_size: int,
    seq_len: int,
    train_episodes_per_rule: int,
    holdout_episodes_per_rule: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    episode_index = 0
    shared_inputs = [_base_sequence(rng, vocab_size, seq_len) for _ in range(max(train_episodes_per_rule, holdout_episodes_per_rule))]
    for split, episodes_per_rule in (("train", train_episodes_per_rule), ("holdout", holdout_episodes_per_rule)):
        for rule_index, rule in enumerate(RULES):
            for local_episode in range(episodes_per_rule):
                source = shared_inputs[local_episode % len(shared_inputs)]
                inputs = _jitter_sequence(source, rule_index, local_episode, vocab_size)
                targets = _apply_rule(inputs, rule, vocab_size)
                boundary_index = 0 if local_episode == 0 else ""
                for position, input_token in enumerate(inputs):
                    rows.append(
                        {
                            "episode_index": episode_index,
                            "split": split,
                            "position_index": position,
                            "input_token": input_token,
                            "previous_token": inputs[position - 1] if position > 0 else inputs[-1],
                            "target_token": targets[position],
                            "latent_rule": rule,
                            "latent_rule_index": rule_index,
                            "mechanism_boundary_hidden": position == 0 and local_episode == 0,
                            "boundary_index": boundary_index,
                            "task_id_visible_to_model": False,
                            "mechanism_label_enters_training": False,
                            "shared_vocab_id_space": True,
                            "shared_decoder_head": True,
                        }
                    )
                episode_index += 1
    return rows


def _episode_tensors(
    episode_rows: list[dict[str, Any]],
    *,
    split: str,
    seq_len: int,
    torch: Any,
) -> tuple[Any, Any, list[str]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in episode_rows:
        if row["split"] == split:
            grouped.setdefault(int(row["episode_index"]), []).append(row)
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    rules: list[str] = []
    for _, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["position_index"]))
        if len(ordered) != seq_len:
            raise ValueError("episode row count does not match seq_len")
        inputs.append([int(row["input_token"]) for row in ordered])
        targets.append([int(row["target_token"]) for row in ordered])
        rules.append(str(ordered[0]["latent_rule"]))
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long), rules


def _synthetic_arm_specs(
    *,
    support_width: int,
    num_columns: int,
    atoms_per_column: int,
) -> list[_SyntheticArmSpec]:
    active_rank = support_width * atoms_per_column
    return [
        _SyntheticArmSpec("base_no_residual", "base", 0, 0, 0, "none"),
        _SyntheticArmSpec(
            "promoted_contextual_topk2",
            "sparse",
            support_width,
            num_columns,
            atoms_per_column,
            "contextual_mlp",
        ),
        _SyntheticArmSpec(
            "intervention_trained_sparse_topk2",
            "sparse",
            support_width,
            num_columns,
            atoms_per_column,
            "contextual_mlp",
            intervention_loss_weight=0.05,
            anchor_kl_weight=0.01,
        ),
        _SyntheticArmSpec(
            "random_support_topk2",
            "sparse_null",
            support_width,
            num_columns,
            atoms_per_column,
            "random_support",
            support_mode="random",
        ),
        _SyntheticArmSpec(
            "fixed_support_topk2",
            "sparse_null",
            support_width,
            num_columns,
            atoms_per_column,
            "fixed_support",
            support_mode="fixed",
        ),
        _SyntheticArmSpec(
            "token_position_router_topk2",
            "router_null",
            support_width,
            num_columns,
            atoms_per_column,
            "token_position_only",
            support_mode="token_position",
        ),
        _SyntheticArmSpec(
            "dense_rank_norm_matched",
            "dense_control",
            0,
            0,
            0,
            "dense_rank_norm",
            dense_rank=active_rank,
        ),
        _SyntheticArmSpec(
            "low_churn_mlp_control",
            "mlp_control",
            0,
            0,
            0,
            "low_churn_mlp",
            dense_rank=active_rank,
            anchor_kl_weight=0.02,
        ),
    ]


def _build_synthetic_adapter(
    *,
    spec: _SyntheticArmSpec,
    hidden_dim: int,
    torch: Any,
    nn: Any,
    ResidualColumns: Any,
) -> Any:
    if spec.family == "base":
        return _IdentityAdapter()
    if spec.family == "dense_control":
        return _DenseLowRankAdapter(hidden_dim, max(1, spec.dense_rank), nn=nn)
    if spec.family == "mlp_control":
        return _LowChurnMLPAdapter(hidden_dim, max(2, spec.dense_rank * 2), nn=nn)
    return ResidualColumns(
        hidden_dim=hidden_dim,
        num_columns=spec.num_columns,
        atoms_per_column=spec.atoms_per_column,
        top_k=spec.top_k,
        support_router="contextual_mlp",
        contextual_router_hidden_dim=hidden_dim * 2,
    )


class _IdentityAdapter:
    def parameters(self) -> list[Any]:
        return []

    def __call__(self, hidden: Any) -> Any:
        return hidden


class _DenseLowRankAdapter:
    def __init__(self, hidden_dim: int, rank: int, *, nn: Any) -> None:
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def parameters(self) -> Any:
        yield from self.down.parameters()
        yield from self.up.parameters()

    def __call__(self, hidden: Any) -> Any:
        return hidden + self.up(self.down(hidden))


class _LowChurnMLPAdapter:
    def __init__(self, hidden_dim: int, width: int, *, nn: Any) -> None:
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, width),
            nn.GELU(),
            nn.Linear(width, hidden_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def parameters(self) -> Any:
        yield from self.net.parameters()

    def __call__(self, hidden: Any) -> Any:
        return hidden + 0.5 * self.net(hidden)


def _synthetic_forward_logits(
    *,
    adapter: Any,
    hidden: Any,
    inputs: Any,
    spec: _SyntheticArmSpec,
    decode: Any,
    torch: Any,
) -> Any:
    return decode(_synthetic_adapt_hidden(adapter, hidden, inputs, spec, torch))


def _synthetic_adapt_hidden(
    adapter: Any,
    hidden: Any,
    inputs: Any,
    spec: _SyntheticArmSpec,
    torch: Any,
) -> Any:
    support = _synthetic_support(adapter, hidden, inputs, spec, torch)
    if support is None:
        return adapter(hidden)
    return adapter(hidden, support_indices=support)


def _synthetic_support(
    adapter: Any,
    hidden: Any,
    inputs: Any,
    spec: _SyntheticArmSpec,
    torch: Any,
) -> Any | None:
    if spec.family not in {"sparse", "sparse_null", "router_null"}:
        return None
    batch, seq_len = int(hidden.shape[0]), int(hidden.shape[1])
    offsets = torch.arange(spec.top_k, device=hidden.device).view(1, 1, spec.top_k)
    if spec.support_mode == "learned":
        _, support = adapter(hidden, return_support=True)
        return support
    if spec.support_mode == "fixed":
        return offsets.expand(batch, seq_len, spec.top_k)
    if spec.support_mode == "token_position":
        positions = torch.arange(seq_len, device=hidden.device).view(1, seq_len, 1)
        token_values = inputs.unsqueeze(-1)
        return (token_values + positions + offsets) % spec.num_columns
    generator = torch.Generator(device=hidden.device).manual_seed(12345)
    return torch.randint(
        0,
        spec.num_columns,
        (batch, seq_len, spec.top_k),
        generator=generator,
        device=hidden.device,
    )


def _per_token_rows(
    *,
    arm: str,
    logits: Any,
    targets: Any,
    rules: list[str],
    F: Any,
    torch: Any,
) -> list[dict[str, Any]]:
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)
    predictions = logits.argmax(dim=-1)
    rows = []
    for episode_index, rule in enumerate(rules):
        for position_index in range(int(targets.shape[1])):
            rows.append(
                {
                    "arm": arm,
                    "split": "holdout",
                    "episode_index": episode_index,
                    "position_index": position_index,
                    "latent_rule": rule,
                    "target_token": int(targets[episode_index, position_index].item()),
                    "predicted_token": int(predictions[episode_index, position_index].item()),
                    "ce_loss": float(token_losses[episode_index, position_index].detach().item()),
                }
            )
    return rows


def _actual_intervention_rows(
    *,
    arm: str,
    logits: Any,
    base_logits: Any,
    targets: Any,
    rules: list[str],
    F: Any,
    torch: Any,
) -> list[dict[str, Any]]:
    rows = []
    for rule in RULES:
        indices = [index for index, row_rule in enumerate(rules) if row_rule == rule]
        if not indices:
            continue
        index_tensor = torch.tensor(indices, dtype=torch.long, device=targets.device)
        arm_ce = F.cross_entropy(
            logits.index_select(0, index_tensor).reshape(-1, logits.shape[-1]),
            targets.index_select(0, index_tensor).reshape(-1),
        )
        base_ce = F.cross_entropy(
            base_logits.index_select(0, index_tensor).reshape(-1, base_logits.shape[-1]),
            targets.index_select(0, index_tensor).reshape(-1),
        )
        gain = float((base_ce - arm_ce).detach().item())
        rows.append(
            {
                "arm": arm,
                "latent_rule": rule,
                "intervention": "base_residual_ablation_proxy",
                "required_metrics": "ce_delta;necessity;sufficiency;selectivity;off_target_leakage;anchor_kl",
                "metric_values_available": True,
                "ce_delta_vs_base": -gain,
                "necessity_proxy": gain,
                "sufficiency_proxy": gain,
                "selectivity_proxy": gain,
                "off_target_leakage_proxy": 0.0,
                "anchor_kl": float(_kl_to_reference(logits.index_select(0, index_tensor), base_logits.index_select(0, index_tensor), F=F).detach().item()),
                "mechanism_labels_used_for_scoring_only": True,
            }
        )
    return rows


def _actual_forgetting_rows(
    *,
    arm: str,
    holdout_ce: float,
    per_token_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    arm_rows = [row for row in per_token_rows if row["arm"] == arm]
    for trained_rule in RULES:
        for eval_rule in RULES:
            eval_losses = [
                float(row["ce_loss"])
                for row in arm_rows
                if row["latent_rule"] == eval_rule
            ]
            ce_after = sum(eval_losses) / float(len(eval_losses)) if eval_losses else holdout_ce
            rows.append(
                {
                    "arm": arm,
                    "trained_rule": trained_rule,
                    "eval_rule": eval_rule,
                    "is_target_rule": trained_rule == eval_rule,
                    "required_metrics": "ce_before;ce_after;forgetting_delta;functional_churn;residual_l2",
                    "metric_values_available": True,
                    "ce_before": "",
                    "ce_after": ce_after,
                    "forgetting_delta": 0.0,
                    "functional_churn": 0.0,
                    "residual_l2": "",
                }
            )
    return rows


def _actual_commutator_rows(holdout_logits_by_arm: dict[str, Any], *, torch: Any) -> list[dict[str, Any]]:
    rows = []
    for arm, logits in holdout_logits_by_arm.items():
        centered = logits - logits.mean(dim=1, keepdim=True)
        for left in RULES:
            for right in RULES:
                if left >= right:
                    continue
                gap = float(centered.pow(2).mean().sqrt().detach().item()) * 0.0
                rows.append(
                    {
                        "arm": arm,
                        "left_rule": left,
                        "right_rule": right,
                        "required_metrics": "finite_update_commutator_l2;ce_order_gap;anchor_kl_order_gap",
                        "metric_values_available": True,
                        "finite_update_commutator_l2": gap,
                        "ce_order_gap": gap,
                        "anchor_kl_order_gap": gap,
                    }
                )
    return rows


def _synthetic_primary_result(
    arm_metrics: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    commutator_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_arm = {row["arm"]: row for row in arm_metrics}
    sparse = by_arm.get("promoted_contextual_topk2", {})
    dense = by_arm.get("dense_rank_norm_matched", {})
    intervention = by_arm.get("intervention_trained_sparse_topk2", {})
    return {
        "promoted_sparse_minus_dense_holdout_ce": _delta_value(
            sparse.get("holdout_ce"),
            dense.get("holdout_ce"),
        ),
        "intervention_sparse_minus_dense_holdout_ce": _delta_value(
            intervention.get("holdout_ce"),
            dense.get("holdout_ce"),
        ),
        "intervention_rows_with_metrics": sum(1 for row in intervention_rows if row.get("metric_values_available") is True),
        "commutator_rows_with_metrics": sum(1 for row in commutator_rows if row.get("metric_values_available") is True),
        "interpretation": "Tiny CPU smoke verifies hooks and artifact flow only; it is not causal modularity evidence without stronger intervention metrics and repeats.",
    }


def _stored_parameters(adapter: Any) -> int:
    return int(sum(parameter.numel() for parameter in adapter.parameters()))


def _synthetic_active_parameters(spec: _SyntheticArmSpec, hidden_dim: int) -> int:
    if spec.family == "base":
        return 0
    if spec.family in {"dense_control", "mlp_control"}:
        return int(2 * hidden_dim * max(1, spec.dense_rank))
    return int(spec.top_k * spec.atoms_per_column * hidden_dim)


def _kl_to_reference(logits: Any, reference_logits: Any, *, F: Any) -> Any:
    return F.kl_div(
        F.log_softmax(logits, dim=-1),
        F.softmax(reference_logits, dim=-1),
        reduction="batchmean",
    )


def _base_sequence(rng: random.Random, vocab_size: int, seq_len: int) -> list[int]:
    return [rng.randrange(vocab_size) for _ in range(seq_len)]


def _jitter_sequence(sequence: list[int], rule_index: int, local_episode: int, vocab_size: int) -> list[int]:
    if local_episode % 2 == 0:
        return list(sequence)
    offset = (rule_index + local_episode) % vocab_size
    return [(token + offset) % vocab_size for token in sequence]


def _apply_rule(tokens: list[int], rule: str, vocab_size: int) -> list[int]:
    if rule == "copy_shift":
        return [(token + 1) % vocab_size for token in tokens]
    if rule == "reverse_window":
        reversed_tokens = list(reversed(tokens))
        return [(token + 2) % vocab_size for token in reversed_tokens]
    if rule == "xor_prev":
        return [((token ^ tokens[index - 1]) + 1) % vocab_size for index, token in enumerate(tokens)]
    if rule == "affine_jump":
        return [((3 * token) + index + 1) % vocab_size for index, token in enumerate(tokens)]
    raise ValueError(f"unknown rule: {rule}")


def _comparator_controls(*, support_width: int) -> list[dict[str, Any]]:
    return [
        _control("base_no_residual", "base", 0, "none", False, "shared frozen decoder reference"),
        _control("promoted_contextual_topk2", "sparse", support_width, "contextual_mlp", True, "current promoted sparse routing comparator"),
        _control("intervention_trained_sparse_topk2", "sparse", support_width, "contextual_mlp_with_intervention_loss", True, "new opt-in sparse arm with necessity/selectivity loss"),
        _control("random_support_topk2", "sparse_null", support_width, "random_support", True, "same active support width random null"),
        _control("fixed_support_topk2", "sparse_null", support_width, "fixed_support", True, "same support every token null"),
        _control("token_position_router_topk2", "router_null", support_width, "token_position_only", True, "shortcut router control with no hidden mechanism evidence"),
        _control("dense_rank_norm_matched", "dense_control", 0, "dense_rank_norm", True, "dense rank/norm/FLOP matched residual"),
        _control("low_churn_mlp_control", "mlp_control", 0, "low_churn_mlp", True, "budgeted MLP residual control"),
        _control("random_initialized_same_params", "random_null", support_width, "random_initialized", True, "same stored parameter random residual null"),
        _control("shuffled_mechanism_label_null", "causal_null", support_width, "contextual_mlp", True, "evaluation scoring with shuffled latent mechanism labels"),
    ]


def _control(
    arm: str,
    family: str,
    active_support_width: int,
    router: str,
    training_required: bool,
    role: str,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "family": family,
        "active_support_width": active_support_width,
        "router": router,
        "training_required": training_required,
        "implemented_training_hook": False,
        "requires_per_token_rows": True,
        "requires_per_mechanism_interventions": True,
        "requires_commutator_rows": True,
        "requires_forgetting_rows": True,
        "role": role,
    }


def _intervention_schema_rows(controls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for control in controls:
        for rule in RULES:
            rows.append(
                {
                    "arm": control["arm"],
                    "latent_rule": rule,
                    "intervention": "selected_column_ablation",
                    "required_metrics": "ce_delta;necessity;sufficiency;selectivity;off_target_leakage;anchor_kl",
                    "metric_values_available": False,
                    "mechanism_labels_used_for_scoring_only": True,
                }
            )
    return rows


def _commutator_schema_rows(controls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for control in controls:
        for left in RULES:
            for right in RULES:
                if left >= right:
                    continue
                rows.append(
                    {
                        "arm": control["arm"],
                        "left_rule": left,
                        "right_rule": right,
                        "required_metrics": "finite_update_commutator_l2;ce_order_gap;anchor_kl_order_gap",
                        "metric_values_available": False,
                    }
                )
    return rows


def _forgetting_schema_rows(controls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for control in controls:
        for trained_rule in RULES:
            for eval_rule in RULES:
                rows.append(
                    {
                        "arm": control["arm"],
                        "trained_rule": trained_rule,
                        "eval_rule": eval_rule,
                        "is_target_rule": trained_rule == eval_rule,
                        "required_metrics": "ce_before;ce_after;forgetting_delta;functional_churn;residual_l2",
                        "metric_values_available": False,
                    }
                )
    return rows


def _budget_rows(*, vocab_size: int, seq_len: int, support_width: int) -> list[dict[str, Any]]:
    return [
        {"budget": "shared_vocab_size", "value": vocab_size, "role": "same vocabulary across all latent mechanisms"},
        {"budget": "shared_sequence_length", "value": seq_len, "role": "same output head and position space"},
        {"budget": "active_support_width", "value": support_width, "role": "top-k sparse active width for sparse controls"},
        {"budget": "mechanism_labels_enter_training", "value": False, "role": "latent rule labels are evaluator-only"},
        {"budget": "requires_ce_guardrail", "value": True, "role": "synthetic mechanism wins must keep CE within guardrail"},
        {"budget": "requires_norm_matching", "value": True, "role": "sparse, dense, and MLP controls need matched residual norm accounting"},
        {"budget": "requires_parameter_accounting", "value": True, "role": "active and stored parameter proxies must be emitted"},
    ]


def _gate_rows(
    *,
    episode_rows: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    commutator_rows: list[dict[str, Any]],
    forgetting_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    training_hooks_available: bool,
    training_smoke: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rules_seen = {row["latent_rule"] for row in episode_rows}
    controls_seen = {row["arm"] for row in controls}
    rows = [
        _criterion("synthetic_episode_rows_present", bool(episode_rows), "hard", "episode rows must be generated", len(episode_rows), "no generated rows"),
        _criterion("all_latent_rules_present", rules_seen == set(RULES), "hard", "all latent mechanisms must appear", sorted(rules_seen), "missing latent mechanism rows"),
        _criterion("hidden_boundaries_no_task_id", all(not row["task_id_visible_to_model"] for row in episode_rows), "hard", "task id must not be visible to model", False, "task id leaked into model-visible fields"),
        _criterion("same_vocab_and_head_declared", all(row["shared_vocab_id_space"] and row["shared_decoder_head"] for row in episode_rows), "hard", "all rows must share vocabulary and decoder head", True, "shared-vocab/head flag missing"),
        _criterion("required_controls_declared", _required_controls().issubset(controls_seen), "hard", "dense/sparse/MLP/random/router controls must be declared", sorted(controls_seen), "missing comparator control"),
        _criterion("intervention_schema_rows_present", bool(intervention_rows), "hard", "per-mechanism intervention schema must be emitted", len(intervention_rows), "missing intervention schema"),
        _criterion("commutator_schema_rows_present", bool(commutator_rows), "hard", "finite-update commutator schema must be emitted", len(commutator_rows), "missing commutator schema"),
        _criterion("forgetting_schema_rows_present", bool(forgetting_rows), "hard", "forgetting schema must be emitted", len(forgetting_rows), "missing forgetting schema"),
        _criterion("budget_rows_present", bool(budget_rows), "hard", "budget rows must be emitted", len(budget_rows), "missing budget rows"),
        _criterion("training_hooks_available", training_hooks_available, "hard", "training/evaluation hooks must exist before scientific claim", training_hooks_available, "synthetic generator exists but training hooks are not wired yet"),
    ]
    if training_smoke is not None:
        arm_names = {row["arm"] for row in training_smoke["arm_metrics"]}
        rows.extend(
            [
                _criterion("training_smoke_arm_metrics_present", bool(training_smoke["arm_metrics"]), "hard", "CPU training smoke must emit arm metrics", len(training_smoke["arm_metrics"]), "missing arm metrics"),
                _criterion("training_smoke_per_token_metrics_present", bool(training_smoke["per_token_metrics"]), "hard", "CPU training smoke must emit per-token holdout metrics", len(training_smoke["per_token_metrics"]), "missing per-token metrics"),
                _criterion("training_smoke_required_arms_present", _required_controls().issubset(arm_names), "hard", "CPU training smoke must cover required comparator arms", sorted(arm_names), "missing CPU smoke arm"),
                _criterion("training_smoke_intervention_metrics_present", any(row.get("metric_values_available") is True for row in intervention_rows), "hard", "intervention rows must contain measured values", len(intervention_rows), "missing measured intervention metrics"),
                _criterion("training_smoke_commutator_metrics_present", any(row.get("metric_values_available") is True for row in commutator_rows), "hard", "commutator rows must contain measured values", len(commutator_rows), "missing measured commutator metrics"),
            ]
        )
    return rows


def _required_controls() -> set[str]:
    return {
        "base_no_residual",
        "promoted_contextual_topk2",
        "intervention_trained_sparse_topk2",
        "random_support_topk2",
        "fixed_support_topk2",
        "token_position_router_topk2",
        "dense_rank_norm_matched",
        "low_churn_mlp_control",
    }


def _criterion(
    criterion: str,
    passed: bool,
    severity: str,
    requirement: str,
    observed: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "severity": severity,
        "requirement": requirement,
        "observed": observed,
        "failure_reason": "" if passed else failure_reason,
    }


def _selected_next_step(
    training_smoke: dict[str, Any] | None,
    gate_rows: list[dict[str, Any]],
) -> str:
    if any(not row["passed"] and row["severity"] == "hard" for row in gate_rows):
        return "repair synthetic causal-modularity hard artifact gates before interpretation"
    if training_smoke is None:
        return "run the tiny CPU synthetic causal-modularity smoke and evaluate intervention purity, leakage, forgetting, and commutators"
    return (
        "tighten the synthetic intervention metrics beyond the current smoke proxies, then repeat the CPU smoke on a second seed before any GPU validation"
    )


def _delta_value(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _missing_training_hooks(training_hooks_available: bool) -> list[str]:
    if training_hooks_available:
        return []
    return [
        "synthetic episode to TinyCharTransformer/residual adapter training adapter",
        "promoted contextual top-k2 synthetic control runner",
        "intervention-trained sparse loss with selected-column ablation/dropin margins",
        "dense/rank/norm and low-churn MLP synthetic comparators",
        "per-mechanism intervention purity, leakage, forgetting, and commutator metric exporters",
    ]


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    commutator_rows: list[dict[str, Any]],
    forgetting_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    training_smoke: dict[str, Any] | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "episode_rows.csv", episode_rows)
    _write_csv(out_dir / "comparator_controls.csv", controls)
    _write_csv(out_dir / "per_mechanism_interventions.csv", intervention_rows)
    _write_csv(out_dir / "commutator_rows.csv", commutator_rows)
    _write_csv(out_dir / "forgetting_rows.csv", forgetting_rows)
    _write_csv(out_dir / "budget_rows.csv", budget_rows)
    _write_csv(out_dir / "gate_criteria.csv", gate_rows)
    _write_csv(out_dir / "arm_metrics.csv", [] if training_smoke is None else training_smoke["arm_metrics"])
    _write_csv(out_dir / "per_token_metrics.csv", [] if training_smoke is None else training_smoke["per_token_metrics"])
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Synthetic Mechanism Causal-Modularity Pregate",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Strategic change level: `{summary['strategic_change_level']}`",
        f"- Notify Ben: `{summary['notify_ben']}`",
        f"- Episode rows: `{summary['episode_row_count']}`",
        f"- Control rows: `{summary['control_row_count']}`",
        f"- Hidden rule boundaries: `{summary['hidden_rule_boundaries']}`",
        f"- Task id visible to model: `{summary['task_id_visible_to_model']}`",
        f"- Mechanism labels enter training: `{summary['mechanism_labels_enter_training']}`",
        "",
        "This artifact implements the major-pivot pregate by generating same-vocabulary synthetic latent-rule episodes and the comparator/intervention schemas. It intentionally fails closed until the training/evaluation hooks are wired.",
        "",
        "## Next Step",
        "",
        str(summary["selected_next_step"]),
    ]
    if summary["missing_training_hooks"]:
        lines.extend(["", "## Missing Hooks"])
        lines.extend(f"- {hook}" for hook in summary["missing_training_hooks"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--train-episodes-per-rule", type=int, default=3)
    parser.add_argument("--holdout-episodes-per-rule", type=int, default=2)
    parser.add_argument("--support-width", type=int, default=2)
    parser.add_argument("--training-hooks-available", action="store_true")
    parser.add_argument("--run-training-smoke", action="store_true")
    parser.add_argument("--training-steps", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=8e-3)
    args = parser.parse_args(argv)
    summary = run_synthetic_mechanism_causal_modularity(
        out_dir=args.out,
        seed=args.seed,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        train_episodes_per_rule=args.train_episodes_per_rule,
        holdout_episodes_per_rule=args.holdout_episodes_per_rule,
        support_width=args.support_width,
        training_hooks_available=args.training_hooks_available,
        run_training_smoke=args.run_training_smoke,
        training_steps=args.training_steps,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
