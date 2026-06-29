"""Synthetic mechanism-factorized causal-modularity pregate.

This module is deliberately local and CPU-only. It creates a hidden-boundary,
same-vocabulary synthetic stream with evaluator-known latent mechanisms, records
the comparator/control schema required by the next sparse-column assay, and
fails closed until concrete training hooks are wired in.
"""

from __future__ import annotations

import argparse
import copy
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
    "local_scientific_gates.csv",
    "arm_metrics.csv",
    "ce_gap_decomposition.csv",
    "oracle_support_sparse_topk2.csv",
    "router_value_regret_decomposition.csv",
    "router_regret_ceiling_budget.csv",
    "teacher_distillation_closeout.csv",
    "per_token_metrics.csv",
    "ce_by_rule_position.csv",
    "residual_budget_accounting.csv",
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
    teacher_distillation_weight: float = 0.0
    shuffled_teacher_null: bool = False
    stored_parameter_floor: int = 0
    control_budget_role: str = "sparse_or_null_or_base_reference"


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
    include_teacher_distillation: bool = False,
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
            include_teacher_distillation=include_teacher_distillation,
        )
        if run_training_smoke
        else None
    )
    hooks_available = training_hooks_available or training_smoke is not None
    controls = _comparator_controls(
        support_width=support_width,
        include_teacher_distillation=include_teacher_distillation,
    )
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
    local_scientific_gate_rows = _local_scientific_gate_rows(training_smoke)
    hard_failures = [row for row in gate_rows if not row["passed"] and row["severity"] == "hard"]
    local_scientific_failures = [
        row for row in local_scientific_gate_rows if not row["passed"]
    ]
    stored_upper_bound_gap_status = _stored_upper_bound_gap_status(training_smoke)
    status = "fail" if hard_failures else "pass"
    decision = (
        "synthetic_mechanism_causal_modularity_pregate_failed_closed"
        if hard_failures
        else "synthetic_mechanism_causal_modularity_local_gates_failed_closed"
        if local_scientific_failures
        else "synthetic_mechanism_causal_modularity_active_matched_passed_stored_upper_bound_blocks_promotion"
        if stored_upper_bound_gap_status == "fail"
        else "synthetic_mechanism_causal_modularity_local_diagnostics_ready_no_promotion"
    )
    summary = {
        "status": status,
        "decision": decision,
        "claim_status": "schema_and_generator_only_no_causal_modularity_claim",
        "promotion_allowed": False,
        "requires_gpu_now": False,
        "backend_policy": "local synthetic pregate only; RunPod and Colab remain blocked",
        "strategy_review_handling": (
            "Accepted the latest urgent GPT-5.5-Pro review: treat the active/stored comparator split as a minor "
            "local diagnostic reclassification, keep GPU and promotion blocked, and add oracle-support/value "
            "decomposition before any modularity claim."
        ),
        "strategic_change_level": "minor",
        "notify_ben": False,
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
        "local_scientific_gate_status": (
            "not_run"
            if training_smoke is None
            else "fail"
            if local_scientific_failures
            else "pass"
        ),
        "active_matched_local_gate_status": (
            "not_run"
            if training_smoke is None
            else "fail"
            if local_scientific_failures
            else "pass"
        ),
        "stored_upper_bound_gap_status": stored_upper_bound_gap_status,
        "local_scientific_gates": local_scientific_gate_rows,
        "failures": hard_failures,
        "local_scientific_failures": local_scientific_failures,
        "training_smoke_ran": training_smoke is not None,
        "training_steps": training_steps if training_smoke is not None else 0,
        "training_smoke_primary_result": (
            training_smoke["primary_result"] if training_smoke is not None else None
        ),
        "arm_metric_row_count": (
            len(training_smoke["arm_metrics"]) if training_smoke is not None else 0
        ),
        "ce_gap_decomposition_row_count": (
            len(training_smoke["ce_gap_decomposition"]) if training_smoke is not None else 0
        ),
        "oracle_support_sparse_topk2_row_count": (
            len(training_smoke["oracle_support_sparse_topk2"]) if training_smoke is not None else 0
        ),
        "oracle_support_primary_result": (
            _oracle_support_summary(training_smoke["oracle_support_sparse_topk2"])
            if training_smoke is not None
            else None
        ),
        "router_value_regret_decomposition_row_count": (
            len(training_smoke["router_value_regret_decomposition"]) if training_smoke is not None else 0
        ),
        "router_value_regret_primary_result": (
            _router_value_regret_summary(training_smoke["router_value_regret_decomposition"])
            if training_smoke is not None
            else None
        ),
        "router_regret_ceiling_budget_row_count": (
            len(training_smoke["router_regret_ceiling_budget"]) if training_smoke is not None else 0
        ),
        "router_regret_ceiling_budget_primary_result": (
            _router_regret_ceiling_budget_summary(training_smoke["router_regret_ceiling_budget"])
            if training_smoke is not None
            else None
        ),
        "teacher_distillation_closeout_row_count": (
            len(training_smoke["teacher_distillation_closeout"]) if training_smoke is not None else 0
        ),
        "teacher_distillation_closeout_primary_result": (
            _teacher_distillation_closeout_summary(training_smoke["teacher_distillation_closeout"])
            if training_smoke is not None
            else None
        ),
        "per_token_metric_row_count": (
            len(training_smoke["per_token_metrics"]) if training_smoke is not None else 0
        ),
        "ce_by_rule_position_row_count": (
            len(training_smoke["ce_by_rule_position"]) if training_smoke is not None else 0
        ),
        "residual_budget_accounting_row_count": (
            len(training_smoke["residual_budget_accounting"]) if training_smoke is not None else 0
        ),
        "teacher_distillation_included": bool(include_teacher_distillation and training_smoke is not None),
        "teacher_distillation_arm_count": (
            sum(1 for row in training_smoke["arm_metrics"] if row.get("teacher_distillation_enabled") is True)
            if training_smoke is not None
            else 0
        ),
        "teacher_distillation_primary_result": (
            _teacher_distillation_summary(training_smoke["arm_metrics"])
            if training_smoke is not None
            else None
        ),
        "residual_budget_primary_result": (
            _residual_budget_summary(training_smoke["residual_budget_accounting"])
            if training_smoke is not None
            else None
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
        local_scientific_gate_rows,
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
    include_teacher_distillation: bool,
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

    sparse_stored_parameter_floor = _sparse_contextual_stored_parameter_count(
        hidden_dim=hidden_dim,
        num_columns=max(4, support_width * 4),
        atoms_per_column=2,
        contextual_router_hidden_dim=hidden_dim * 2,
    )
    specs = _synthetic_arm_specs(
        support_width=support_width,
        num_columns=max(4, support_width * 4),
        atoms_per_column=2,
        hidden_dim=hidden_dim,
        sparse_stored_parameter_floor=sparse_stored_parameter_floor,
        include_teacher_distillation=include_teacher_distillation,
    )
    dense_teacher = (
        _train_dense_teacher_adapter(
            hidden_dim=hidden_dim,
            dense_rank=_dense_rank_for_parameter_floor(hidden_dim, sparse_stored_parameter_floor),
            train_hidden=train_hidden,
            train_targets=train_targets,
            vocab_size=vocab_size,
            training_steps=training_steps,
            learning_rate=learning_rate,
            decode=base.decode,
            seed=seed,
            torch=torch,
            nn=nn,
            F=F,
        )
        if include_teacher_distillation
        else None
    )
    arm_metrics: list[dict[str, Any]] = []
    per_token_metrics: list[dict[str, Any]] = []
    oracle_support_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    commutator_rows: list[dict[str, Any]] = []
    forgetting_rows: list[dict[str, Any]] = []
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
            if spec.teacher_distillation_weight > 0.0 and dense_teacher is not None:
                student_delta = _synthetic_adapt_hidden(adapter, train_hidden, train_inputs, spec, torch) - train_hidden
                teacher_delta = _teacher_residual_delta(
                    dense_teacher,
                    train_hidden,
                    shuffle=spec.shuffled_teacher_null,
                    torch=torch,
                )
                loss = loss + spec.teacher_distillation_weight * F.mse_loss(student_delta, teacher_delta)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            adapted_holdout_hidden = _synthetic_adapt_hidden(adapter, holdout_hidden, holdout_inputs, spec, torch)
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
                (adapted_holdout_hidden - holdout_hidden)
                .pow(2)
                .mean()
                .sqrt()
                .detach()
                .item()
            )
            teacher_residual_mse = None
            if spec.teacher_distillation_weight > 0.0 and dense_teacher is not None:
                teacher_delta = _teacher_residual_delta(
                    dense_teacher,
                    holdout_hidden,
                    shuffle=spec.shuffled_teacher_null,
                    torch=torch,
                )
                teacher_residual_mse = float(
                    F.mse_loss(adapted_holdout_hidden - holdout_hidden, teacher_delta)
                    .detach()
                    .item()
                )
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
                "teacher_distillation_enabled": spec.teacher_distillation_weight > 0.0,
                "teacher_distillation_weight": spec.teacher_distillation_weight,
                "shuffled_teacher_null": spec.shuffled_teacher_null,
                "teacher_residual_mse": teacher_residual_mse,
                "stored_parameters": _stored_parameters(adapter),
                "stored_parameter_floor": spec.stored_parameter_floor,
                "active_parameters_proxy": _synthetic_active_parameters(spec, hidden_dim),
                "control_budget_role": spec.control_budget_role,
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
        if spec.family == "sparse" and spec.support_mode == "learned" and spec.top_k == 2:
            oracle_support_rows.extend(
                _oracle_support_sparse_topk2_rows(
                    arm=spec.name,
                    adapter=adapter,
                    hidden=holdout_hidden,
                    inputs=holdout_inputs,
                    targets=holdout_targets,
                    rules=holdout_rules,
                    spec=spec,
                    decode=base.decode,
                    F=F,
                    torch=torch,
                )
            )
        intervention_rows.extend(
            _actual_intervention_rows(
                arm=spec.name,
                adapter=adapter,
                hidden=holdout_hidden,
                inputs=holdout_inputs,
                logits=holdout_logits,
                base_logits=base_holdout_logits,
                targets=holdout_targets,
                rules=holdout_rules,
                spec=spec,
                decode=base.decode,
                F=F,
                torch=torch,
            )
        )
        commutator_rows.extend(
            _actual_commutator_rows(
                arm=spec.name,
                adapter=adapter,
                spec=spec,
                train_hidden=train_hidden,
                train_inputs=train_inputs,
                train_targets=train_targets,
                train_rules=train_rules,
                holdout_hidden=holdout_hidden,
                holdout_inputs=holdout_inputs,
                holdout_targets=holdout_targets,
                decode=base.decode,
                learning_rate=learning_rate,
                F=F,
                torch=torch,
            )
        )
        forgetting_rows.extend(
            _actual_forgetting_rows(
                arm=spec.name,
                adapter=adapter,
                spec=spec,
                train_hidden=train_hidden,
                train_inputs=train_inputs,
                train_targets=train_targets,
                train_rules=train_rules,
                holdout_hidden=holdout_hidden,
                holdout_inputs=holdout_inputs,
                holdout_targets=holdout_targets,
                holdout_rules=holdout_rules,
                decode=base.decode,
                learning_rate=learning_rate,
                F=F,
                torch=torch,
            )
        )

    primary = _synthetic_primary_result(arm_metrics, intervention_rows, commutator_rows)
    ce_by_rule_position = _ce_by_rule_position_rows(per_token_metrics)
    return {
        "arm_metrics": arm_metrics,
        "ce_gap_decomposition": _ce_gap_decomposition_rows(arm_metrics),
        "oracle_support_sparse_topk2": oracle_support_rows,
        "router_value_regret_decomposition": _router_value_regret_decomposition_rows(
            oracle_support_rows
        ),
        "router_regret_ceiling_budget": _router_regret_ceiling_budget_rows(
            arm_metrics,
            oracle_support_rows,
        ),
        "teacher_distillation_closeout": _teacher_distillation_closeout_rows(
            arm_metrics,
            oracle_support_rows,
        ),
        "per_token_metrics": per_token_metrics,
        "ce_by_rule_position": ce_by_rule_position,
        "residual_budget_accounting": _residual_budget_accounting_rows(arm_metrics),
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
    hidden_dim: int,
    sparse_stored_parameter_floor: int,
    include_teacher_distillation: bool = False,
) -> list[_SyntheticArmSpec]:
    active_rank = support_width * atoms_per_column
    sparse_active_proxy = active_rank * hidden_dim
    active_matched_dense_rank = max(1, (sparse_active_proxy + (2 * hidden_dim) - 1) // (2 * hidden_dim))
    active_matched_mlp_rank = active_matched_dense_rank
    stored_matched_dense_rank = max(
        active_rank,
        _dense_rank_for_parameter_floor(hidden_dim, sparse_stored_parameter_floor),
    )
    stored_matched_mlp_rank = max(
        active_rank,
        _mlp_rank_for_parameter_floor(hidden_dim, sparse_stored_parameter_floor),
    )
    specs = [
        _SyntheticArmSpec("base_no_residual", "base", 0, 0, 0, "none"),
        _SyntheticArmSpec(
            "promoted_contextual_topk2",
            "sparse",
            support_width,
            num_columns,
            atoms_per_column,
            "contextual_mlp",
            stored_parameter_floor=sparse_stored_parameter_floor,
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
            stored_parameter_floor=sparse_stored_parameter_floor,
        ),
        _SyntheticArmSpec(
            "random_support_topk2",
            "sparse_null",
            support_width,
            num_columns,
            atoms_per_column,
            "random_support",
            support_mode="random",
            stored_parameter_floor=sparse_stored_parameter_floor,
        ),
        _SyntheticArmSpec(
            "fixed_support_topk2",
            "sparse_null",
            support_width,
            num_columns,
            atoms_per_column,
            "fixed_support",
            support_mode="fixed",
            stored_parameter_floor=sparse_stored_parameter_floor,
        ),
        _SyntheticArmSpec(
            "token_position_router_topk2",
            "router_null",
            support_width,
            num_columns,
            atoms_per_column,
            "token_position_only",
            support_mode="token_position",
            stored_parameter_floor=sparse_stored_parameter_floor,
        ),
        _SyntheticArmSpec(
            "dense_rank_norm_matched",
            "dense_control",
            0,
            0,
            0,
            "dense_rank_norm",
            dense_rank=active_matched_dense_rank,
            stored_parameter_floor=sparse_stored_parameter_floor,
            control_budget_role="active_proxy_matched_dense_mlp_control",
        ),
        _SyntheticArmSpec(
            "low_churn_mlp_active_matched",
            "mlp_control",
            0,
            0,
            0,
            "low_churn_mlp",
            dense_rank=active_matched_mlp_rank,
            anchor_kl_weight=0.02,
            stored_parameter_floor=sparse_stored_parameter_floor,
            control_budget_role="active_proxy_matched_dense_mlp_control",
        ),
        _SyntheticArmSpec(
            "dense_stored_parameter_matched",
            "dense_control",
            0,
            0,
            0,
            "dense_rank_norm",
            dense_rank=stored_matched_dense_rank,
            stored_parameter_floor=sparse_stored_parameter_floor,
            control_budget_role="stored_parameter_matched_dense_mlp_upper_bound",
        ),
        _SyntheticArmSpec(
            "low_churn_mlp_stored_parameter_matched",
            "mlp_control",
            0,
            0,
            0,
            "low_churn_mlp",
            dense_rank=stored_matched_mlp_rank,
            anchor_kl_weight=0.02,
            stored_parameter_floor=sparse_stored_parameter_floor,
            control_budget_role="stored_parameter_matched_dense_mlp_upper_bound",
        ),
    ]
    if include_teacher_distillation:
        specs.extend(
            [
                _SyntheticArmSpec(
                    "dense_teacher_distilled_sparse_topk2",
                    "sparse",
                    support_width,
                    num_columns,
                    atoms_per_column,
                    "contextual_mlp",
                    teacher_distillation_weight=1.0,
                    anchor_kl_weight=0.01,
                    stored_parameter_floor=sparse_stored_parameter_floor,
                ),
                _SyntheticArmSpec(
                    "shuffled_teacher_distilled_sparse_topk2",
                    "sparse_null",
                    support_width,
                    num_columns,
                    atoms_per_column,
                    "contextual_mlp",
                    teacher_distillation_weight=1.0,
                    shuffled_teacher_null=True,
                    anchor_kl_weight=0.01,
                    stored_parameter_floor=sparse_stored_parameter_floor,
                ),
            ]
        )
    return specs


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


def _train_dense_teacher_adapter(
    *,
    hidden_dim: int,
    dense_rank: int,
    train_hidden: Any,
    train_targets: Any,
    vocab_size: int,
    training_steps: int,
    learning_rate: float,
    decode: Any,
    seed: int,
    torch: Any,
    nn: Any,
    F: Any,
) -> Any:
    torch.manual_seed(seed + 707)
    teacher = _DenseLowRankAdapter(hidden_dim, max(1, dense_rank), nn=nn)
    optimizer = torch.optim.AdamW(list(teacher.parameters()), lr=learning_rate)
    for _ in range(training_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = decode(teacher(train_hidden))
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), train_targets.reshape(-1))
        loss.backward()
        optimizer.step()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def _teacher_residual_delta(
    teacher: Any,
    hidden: Any,
    *,
    shuffle: bool,
    torch: Any,
) -> Any:
    with torch.no_grad():
        delta = teacher(hidden).detach() - hidden
        if not shuffle:
            return delta
        flat = delta.reshape(-1, delta.shape[-1])
        order = torch.arange(flat.shape[0] - 1, -1, -1, device=flat.device)
        return flat.index_select(0, order).reshape_as(delta)


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


def _oracle_support_sparse_topk2_rows(
    *,
    arm: str,
    adapter: Any,
    hidden: Any,
    inputs: Any,
    targets: Any,
    rules: list[str],
    spec: _SyntheticArmSpec,
    decode: Any,
    F: Any,
    torch: Any,
) -> list[dict[str, Any]]:
    learned_support = _synthetic_support(adapter, hidden, inputs, spec, torch)
    if learned_support is None:
        return []
    singleton_supports = [(column,) for column in range(spec.num_columns)]
    pair_supports = [
        (left, right)
        for left in range(spec.num_columns)
        for right in range(left + 1, spec.num_columns)
    ]
    learned_logits = decode(adapter(hidden, support_indices=learned_support)).detach()
    learned_losses = F.cross_entropy(
        learned_logits.reshape(-1, learned_logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)
    support_loss_rows: list[tuple[tuple[int, ...], Any]] = []
    batch, seq_len = int(targets.shape[0]), int(targets.shape[1])
    for support_tuple in singleton_supports + pair_supports:
        support_tensor = torch.tensor(
            support_tuple,
            dtype=torch.long,
            device=hidden.device,
        ).view(1, 1, len(support_tuple)).expand(batch, seq_len, len(support_tuple))
        logits = decode(adapter(hidden, support_indices=support_tensor)).detach()
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)
        support_loss_rows.append((support_tuple, losses))

    rows: list[dict[str, Any]] = []
    for episode_index, rule in enumerate(rules):
        for position_index in range(seq_len):
            learned_tuple = tuple(
                int(value.detach().item())
                for value in learned_support[episode_index, position_index]
            )
            learned_loss = float(learned_losses[episode_index, position_index].detach().item())
            best_singleton_tuple: tuple[int, ...] | None = None
            best_singleton_loss: float | None = None
            best_pair_tuple: tuple[int, ...] | None = None
            best_pair_loss: float | None = None
            best_one_swap_tuple: tuple[int, ...] | None = None
            best_one_swap_loss: float | None = None
            learned_set = set(learned_tuple)
            for support_tuple, losses in support_loss_rows:
                loss = float(losses[episode_index, position_index].detach().item())
                if len(support_tuple) == 1:
                    if best_singleton_loss is None or loss < best_singleton_loss:
                        best_singleton_tuple = support_tuple
                        best_singleton_loss = loss
                    continue
                if best_pair_loss is None or loss < best_pair_loss:
                    best_pair_tuple = support_tuple
                    best_pair_loss = loss
                if len(learned_set.intersection(support_tuple)) == 1:
                    if best_one_swap_loss is None or loss < best_one_swap_loss:
                        best_one_swap_tuple = support_tuple
                        best_one_swap_loss = loss
            oracle_tuple = best_pair_tuple
            oracle_loss = best_pair_loss
            if best_singleton_loss is not None and (
                oracle_loss is None or best_singleton_loss < oracle_loss
            ):
                oracle_tuple = best_singleton_tuple
                oracle_loss = best_singleton_loss
            regret = _delta_value(learned_loss, oracle_loss)
            pair_regret = _delta_value(learned_loss, best_pair_loss)
            one_swap_regret = _delta_value(learned_loss, best_one_swap_loss)
            total_recoverable = regret if regret is not None and regret > 0.0 else None
            one_swap_recovery = (
                ((learned_loss - best_one_swap_loss) / total_recoverable)
                if total_recoverable is not None and best_one_swap_loss is not None
                else None
            )
            rows.append(
                {
                    "arm": arm,
                    "split": "holdout",
                    "episode_index": episode_index,
                    "position_index": position_index,
                    "latent_rule": rule,
                    "target_token": int(targets[episode_index, position_index].item()),
                    "learned_support": _support_key(learned_tuple),
                    "learned_ce_loss": learned_loss,
                    "best_singleton_support": _support_key(best_singleton_tuple),
                    "best_singleton_ce_loss": best_singleton_loss,
                    "best_pair_support": _support_key(best_pair_tuple),
                    "best_pair_ce_loss": best_pair_loss,
                    "oracle_support": _support_key(oracle_tuple),
                    "oracle_support_size": len(oracle_tuple) if oracle_tuple is not None else None,
                    "oracle_ce_loss": oracle_loss,
                    "oracle_regret": regret,
                    "pair_oracle_regret": pair_regret,
                    "best_one_swap_support": _support_key(best_one_swap_tuple),
                    "best_one_swap_ce_loss": best_one_swap_loss,
                    "one_swap_regret": one_swap_regret,
                    "one_swap_recovery_fraction": one_swap_recovery,
                    "singleton_supports_evaluated": len(singleton_supports),
                    "pair_supports_evaluated": len(pair_supports),
                    "mechanism_labels_used_for_scoring_only": True,
                }
            )
    return rows


def _support_key(support: tuple[int, ...] | None) -> str:
    if support is None:
        return ""
    return ",".join(str(column) for column in support)


def _actual_intervention_rows(
    *,
    arm: str,
    adapter: Any,
    hidden: Any,
    inputs: Any,
    logits: Any,
    base_logits: Any,
    targets: Any,
    rules: list[str],
    spec: _SyntheticArmSpec,
    decode: Any,
    F: Any,
    torch: Any,
) -> list[dict[str, Any]]:
    rows = []
    support = _synthetic_support(adapter, hidden, inputs, spec, torch)
    for rule in RULES:
        indices = [index for index, row_rule in enumerate(rules) if row_rule == rule]
        if not indices:
            continue
        index_tensor = torch.tensor(indices, dtype=torch.long, device=targets.device)
        arm_rule_logits = logits.index_select(0, index_tensor)
        arm_ce = _ce_for_indices(logits, targets, index_tensor, F=F)
        base_ce = _ce_for_indices(base_logits, targets, index_tensor, F=F)
        if support is not None:
            mechanism_columns = _modal_support_columns(support.index_select(0, index_tensor), spec.top_k, torch)
            ablated_support = _replace_support_columns(support, mechanism_columns, spec.num_columns, torch)
            ablated_logits = decode(adapter(hidden, support_indices=ablated_support))
            dropin_support = torch.tensor(
                mechanism_columns,
                dtype=torch.long,
                device=hidden.device,
            ).view(1, 1, len(mechanism_columns)).expand_as(support)
            dropin_logits = decode(adapter(hidden, support_indices=dropin_support))
            ablated_ce = _ce_for_indices(ablated_logits, targets, index_tensor, F=F)
            dropin_ce = _ce_for_indices(dropin_logits, targets, index_tensor, F=F)
            other_indices = [index for index, row_rule in enumerate(rules) if row_rule != rule]
            if other_indices:
                other_index_tensor = torch.tensor(other_indices, dtype=torch.long, device=targets.device)
                other_arm_ce = _ce_for_indices(logits, targets, other_index_tensor, F=F)
                other_ablated_ce = _ce_for_indices(ablated_logits, targets, other_index_tensor, F=F)
                off_target_leakage = float((other_ablated_ce - other_arm_ce).detach().abs().item())
            else:
                off_target_leakage = 0.0
            necessity = float((ablated_ce - arm_ce).detach().item())
            sufficiency = float((base_ce - dropin_ce).detach().item())
            selectivity = necessity - off_target_leakage
            intervention_name = "selected_column_ablation_dropin"
            anchor_kl = float(
                _kl_to_reference(
                    ablated_logits.index_select(0, index_tensor),
                    arm_rule_logits,
                    F=F,
                )
                .detach()
                .item()
            )
            selected_columns = ",".join(str(column) for column in mechanism_columns)
        else:
            ablated_logits = base_logits
            ablated_ce = base_ce
            dropin_ce = arm_ce
            necessity = float((ablated_ce - arm_ce).detach().item())
            sufficiency = float((base_ce - dropin_ce).detach().item())
            selectivity = necessity
            off_target_leakage = 0.0
            intervention_name = "global_residual_ablation_control"
            anchor_kl = float(
                _kl_to_reference(
                    ablated_logits.index_select(0, index_tensor),
                    arm_rule_logits,
                    F=F,
                )
                .detach()
                .item()
            )
            selected_columns = ""
        rows.append(
            {
                "arm": arm,
                "latent_rule": rule,
                "intervention": intervention_name,
                "selected_columns": selected_columns,
                "required_metrics": "ce_delta;necessity;sufficiency;selectivity;off_target_leakage;anchor_kl",
                "metric_values_available": True,
                "normal_ce": float(arm_ce.detach().item()),
                "ablated_ce": float(ablated_ce.detach().item()),
                "dropin_ce": float(dropin_ce.detach().item()),
                "base_ce": float(base_ce.detach().item()),
                "ce_delta_vs_base": float((arm_ce - base_ce).detach().item()),
                "necessity": necessity,
                "sufficiency": sufficiency,
                "selectivity": selectivity,
                "off_target_leakage": off_target_leakage,
                "anchor_kl": anchor_kl,
                "mechanism_labels_used_for_scoring_only": True,
            }
        )
    return rows


def _actual_forgetting_rows(
    *,
    arm: str,
    adapter: Any,
    spec: _SyntheticArmSpec,
    train_hidden: Any,
    train_inputs: Any,
    train_targets: Any,
    train_rules: list[str],
    holdout_hidden: Any,
    holdout_inputs: Any,
    holdout_targets: Any,
    holdout_rules: list[str],
    decode: Any,
    learning_rate: float,
    F: Any,
    torch: Any,
) -> list[dict[str, Any]]:
    rows = []
    with torch.no_grad():
        before_logits = _synthetic_forward_logits(
            adapter=adapter,
            hidden=holdout_hidden,
            inputs=holdout_inputs,
            spec=spec,
            decode=decode,
            torch=torch,
        ).detach()
    for trained_rule in RULES:
        updated_adapter = copy.deepcopy(adapter)
        _single_rule_update(
            adapter=updated_adapter,
            rule=trained_rule,
            train_hidden=train_hidden,
            train_inputs=train_inputs,
            train_targets=train_targets,
            train_rules=train_rules,
            spec=spec,
            decode=decode,
            learning_rate=learning_rate,
            F=F,
            torch=torch,
        )
        with torch.no_grad():
            after_logits = _synthetic_forward_logits(
                adapter=updated_adapter,
                hidden=holdout_hidden,
                inputs=holdout_inputs,
                spec=spec,
                decode=decode,
                torch=torch,
            ).detach()
            after_hidden = _synthetic_adapt_hidden(
                updated_adapter,
                holdout_hidden,
                holdout_inputs,
                spec,
                torch,
            ).detach()
        for eval_rule in RULES:
            indices = [index for index, rule in enumerate(holdout_rules) if rule == eval_rule]
            if not indices:
                continue
            index_tensor = torch.tensor(indices, dtype=torch.long, device=holdout_targets.device)
            ce_before = _ce_for_indices(before_logits, holdout_targets, index_tensor, F=F)
            ce_after = _ce_for_indices(after_logits, holdout_targets, index_tensor, F=F)
            before_eval = before_logits.index_select(0, index_tensor)
            after_eval = after_logits.index_select(0, index_tensor)
            churn = (after_eval - before_eval).pow(2).mean().sqrt()
            residual_l2 = (
                after_hidden.index_select(0, index_tensor)
                - holdout_hidden.index_select(0, index_tensor)
            ).pow(2).mean().sqrt()
            rows.append(
                {
                    "arm": arm,
                    "trained_rule": trained_rule,
                    "eval_rule": eval_rule,
                    "is_target_rule": trained_rule == eval_rule,
                    "required_metrics": "ce_before;ce_after;forgetting_delta;functional_churn;residual_l2",
                    "metric_values_available": True,
                    "ce_before": float(ce_before.detach().item()),
                    "ce_after": float(ce_after.detach().item()),
                    "forgetting_delta": float((ce_after - ce_before).detach().item()),
                    "functional_churn": float(churn.detach().item()),
                    "residual_l2": float(residual_l2.detach().item()),
                }
            )
    return rows


def _actual_commutator_rows(
    *,
    arm: str,
    adapter: Any,
    spec: _SyntheticArmSpec,
    train_hidden: Any,
    train_inputs: Any,
    train_targets: Any,
    train_rules: list[str],
    holdout_hidden: Any,
    holdout_inputs: Any,
    holdout_targets: Any,
    decode: Any,
    learning_rate: float,
    F: Any,
    torch: Any,
) -> list[dict[str, Any]]:
    rows = []
    base_logits = _synthetic_forward_logits(
        adapter=adapter,
        hidden=holdout_hidden,
        inputs=holdout_inputs,
        spec=spec,
        decode=decode,
        torch=torch,
    ).detach()
    for left in RULES:
        for right in RULES:
            if left >= right:
                continue
            if not list(adapter.parameters()):
                ab_logits = base_logits
                ba_logits = base_logits
            else:
                ab_adapter = copy.deepcopy(adapter)
                ba_adapter = copy.deepcopy(adapter)
                _single_rule_update(
                    adapter=ab_adapter,
                    rule=left,
                    train_hidden=train_hidden,
                    train_inputs=train_inputs,
                    train_targets=train_targets,
                    train_rules=train_rules,
                    spec=spec,
                    decode=decode,
                    learning_rate=learning_rate,
                    F=F,
                    torch=torch,
                )
                _single_rule_update(
                    adapter=ab_adapter,
                    rule=right,
                    train_hidden=train_hidden,
                    train_inputs=train_inputs,
                    train_targets=train_targets,
                    train_rules=train_rules,
                    spec=spec,
                    decode=decode,
                    learning_rate=learning_rate,
                    F=F,
                    torch=torch,
                )
                _single_rule_update(
                    adapter=ba_adapter,
                    rule=right,
                    train_hidden=train_hidden,
                    train_inputs=train_inputs,
                    train_targets=train_targets,
                    train_rules=train_rules,
                    spec=spec,
                    decode=decode,
                    learning_rate=learning_rate,
                    F=F,
                    torch=torch,
                )
                _single_rule_update(
                    adapter=ba_adapter,
                    rule=left,
                    train_hidden=train_hidden,
                    train_inputs=train_inputs,
                    train_targets=train_targets,
                    train_rules=train_rules,
                    spec=spec,
                    decode=decode,
                    learning_rate=learning_rate,
                    F=F,
                    torch=torch,
                )
                with torch.no_grad():
                    ab_logits = _synthetic_forward_logits(
                        adapter=ab_adapter,
                        hidden=holdout_hidden,
                        inputs=holdout_inputs,
                        spec=spec,
                        decode=decode,
                        torch=torch,
                    ).detach()
                    ba_logits = _synthetic_forward_logits(
                        adapter=ba_adapter,
                        hidden=holdout_hidden,
                        inputs=holdout_inputs,
                        spec=spec,
                        decode=decode,
                        torch=torch,
                    ).detach()
            ab_ce = F.cross_entropy(ab_logits.reshape(-1, ab_logits.shape[-1]), holdout_targets.reshape(-1))
            ba_ce = F.cross_entropy(ba_logits.reshape(-1, ba_logits.shape[-1]), holdout_targets.reshape(-1))
            rows.append(
                {
                    "arm": arm,
                    "left_rule": left,
                    "right_rule": right,
                    "required_metrics": "finite_update_commutator_l2;ce_order_gap;anchor_kl_order_gap",
                    "metric_values_available": True,
                    "finite_update_commutator_l2": float((ab_logits - ba_logits).pow(2).mean().sqrt().detach().item()),
                    "ce_order_gap": float((ab_ce - ba_ce).detach().abs().item()),
                    "anchor_kl_order_gap": float(_kl_to_reference(ab_logits, ba_logits, F=F).detach().item()),
                }
            )
    return rows


def _single_rule_update(
    *,
    adapter: Any,
    rule: str,
    train_hidden: Any,
    train_inputs: Any,
    train_targets: Any,
    train_rules: list[str],
    spec: _SyntheticArmSpec,
    decode: Any,
    learning_rate: float,
    F: Any,
    torch: Any,
) -> None:
    parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    if not parameters:
        return
    indices = [index for index, train_rule in enumerate(train_rules) if train_rule == rule]
    if not indices:
        return
    index_tensor = torch.tensor(indices, dtype=torch.long, device=train_targets.device)
    optimizer = torch.optim.SGD(parameters, lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    logits = _synthetic_forward_logits(
        adapter=adapter,
        hidden=train_hidden.index_select(0, index_tensor),
        inputs=train_inputs.index_select(0, index_tensor),
        spec=spec,
        decode=decode,
        torch=torch,
    )
    targets = train_targets.index_select(0, index_tensor)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    loss.backward()
    optimizer.step()


def _ce_for_indices(logits: Any, targets: Any, index_tensor: Any, *, F: Any) -> Any:
    return F.cross_entropy(
        logits.index_select(0, index_tensor).reshape(-1, logits.shape[-1]),
        targets.index_select(0, index_tensor).reshape(-1),
    )


def _modal_support_columns(support: Any, top_k: int, torch: Any) -> list[int]:
    flat = support.reshape(-1)
    values, counts = torch.unique(flat, return_counts=True)
    order = torch.argsort(counts, descending=True)
    selected = [int(values[index].detach().item()) for index in order[:top_k]]
    while len(selected) < top_k:
        selected.append(selected[-1] if selected else 0)
    return selected


def _replace_support_columns(support: Any, mechanism_columns: list[int], num_columns: int, torch: Any) -> Any:
    blocked = torch.tensor(mechanism_columns, dtype=torch.long, device=support.device)
    replacement = torch.zeros_like(support)
    for offset in range(num_columns):
        candidate = (support + offset + 1) % num_columns
        allowed = ~torch.isin(candidate, blocked)
        replacement = torch.where((replacement == 0) & allowed, candidate, replacement)
    mask = torch.isin(support, blocked)
    return torch.where(mask, replacement, support)


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
        "interpretation": "Tiny CPU smoke verifies measured intervention and finite-update artifact flow only; it is not causal modularity evidence without stricter gates and repeats.",
    }


def _ce_gap_decomposition_rows(arm_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not arm_metrics:
        return []
    sparse_rows = [
        row
        for row in arm_metrics
        if row.get("arm") in {"promoted_contextual_topk2", "intervention_trained_sparse_topk2"}
    ]
    dense_mlp_rows = [
        row
        for row in arm_metrics
        if row.get("family") in {"dense_control", "mlp_control"}
    ]
    active_dense_mlp_rows = [
        row
        for row in dense_mlp_rows
        if row.get("control_budget_role") == "active_proxy_matched_dense_mlp_control"
    ]
    stored_dense_mlp_rows = [
        row
        for row in dense_mlp_rows
        if row.get("control_budget_role") == "stored_parameter_matched_dense_mlp_upper_bound"
    ]
    best_sparse = _best_ce_row(sparse_rows)
    best_dense_mlp = _best_ce_row(dense_mlp_rows)
    best_active_dense_mlp = _best_ce_row(active_dense_mlp_rows)
    best_stored_dense_mlp = _best_ce_row(stored_dense_mlp_rows)
    best_sparse_ce = _metric_float(best_sparse.get("holdout_ce")) if best_sparse else None
    best_dense_mlp_ce = _metric_float(best_dense_mlp.get("holdout_ce")) if best_dense_mlp else None
    best_active_dense_mlp_ce = (
        _metric_float(best_active_dense_mlp.get("holdout_ce")) if best_active_dense_mlp else None
    )
    best_stored_dense_mlp_ce = (
        _metric_float(best_stored_dense_mlp.get("holdout_ce")) if best_stored_dense_mlp else None
    )
    best_sparse_active = _metric_float(best_sparse.get("active_parameters_proxy")) if best_sparse else None
    best_sparse_stored = _metric_float(best_sparse.get("stored_parameters")) if best_sparse else None
    rows: list[dict[str, Any]] = []
    for row in arm_metrics:
        holdout_ce = _metric_float(row.get("holdout_ce"))
        active_parameters = _metric_float(row.get("active_parameters_proxy"))
        stored_parameters = _metric_float(row.get("stored_parameters"))
        family = str(row.get("family", ""))
        is_dense_or_mlp = family in {"dense_control", "mlp_control"}
        control_budget_role = str(row.get("control_budget_role") or "")
        rows.append(
            {
                "arm": row.get("arm", ""),
                "family": family,
                "router": row.get("router", ""),
                "support_mode": row.get("support_mode", ""),
                "holdout_ce": holdout_ce,
                "residual_l2": _metric_float(row.get("residual_l2")),
                "active_parameters_proxy": active_parameters,
                "stored_parameters": stored_parameters,
                "stored_parameter_floor": _metric_float(row.get("stored_parameter_floor")),
                "active_to_best_sparse_ratio": _safe_ratio(active_parameters, best_sparse_active),
                "stored_to_best_sparse_ratio": _safe_ratio(stored_parameters, best_sparse_stored),
                "best_sparse_arm": best_sparse.get("arm", "") if best_sparse else "",
                "best_sparse_holdout_ce": best_sparse_ce,
                "best_dense_mlp_arm": best_dense_mlp.get("arm", "") if best_dense_mlp else "",
                "best_dense_mlp_holdout_ce": best_dense_mlp_ce,
                "best_active_matched_dense_mlp_arm": (
                    best_active_dense_mlp.get("arm", "") if best_active_dense_mlp else ""
                ),
                "best_active_matched_dense_mlp_holdout_ce": best_active_dense_mlp_ce,
                "best_stored_matched_dense_mlp_arm": (
                    best_stored_dense_mlp.get("arm", "") if best_stored_dense_mlp else ""
                ),
                "best_stored_matched_dense_mlp_holdout_ce": best_stored_dense_mlp_ce,
                "holdout_ce_minus_best_sparse_ce": _delta_value(holdout_ce, best_sparse_ce),
                "holdout_ce_minus_best_dense_mlp_ce": _delta_value(holdout_ce, best_dense_mlp_ce),
                "holdout_ce_minus_best_active_matched_dense_mlp_ce": _delta_value(
                    holdout_ce,
                    best_active_dense_mlp_ce,
                ),
                "holdout_ce_minus_best_stored_matched_dense_mlp_ce": _delta_value(
                    holdout_ce,
                    best_stored_dense_mlp_ce,
                ),
                "best_sparse_ce_minus_this_dense_mlp_ce": (
                    _delta_value(best_sparse_ce, holdout_ce) if is_dense_or_mlp else None
                ),
                "best_sparse_ce_minus_best_dense_mlp_ce": _delta_value(best_sparse_ce, best_dense_mlp_ce),
                "best_sparse_ce_minus_best_active_matched_dense_mlp_ce": _delta_value(
                    best_sparse_ce,
                    best_active_dense_mlp_ce,
                ),
                "best_sparse_ce_minus_best_stored_matched_dense_mlp_ce": _delta_value(
                    best_sparse_ce,
                    best_stored_dense_mlp_ce,
                ),
                "control_budget_role": control_budget_role or "sparse_or_null_or_base_reference",
            }
        )
    return rows


def _ce_by_rule_position_rows(per_token_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in per_token_metrics:
        grouped.setdefault(
            (
                str(row.get("arm", "")),
                str(row.get("latent_rule", "")),
                int(row.get("position_index", 0)),
            ),
            [],
        ).append(row)
    rows: list[dict[str, Any]] = []
    for (arm, latent_rule, position_index), group_rows in sorted(grouped.items()):
        losses = [_metric_float(row.get("ce_loss")) for row in group_rows]
        losses = [loss for loss in losses if loss is not None]
        correct = [
            int(row.get("predicted_token")) == int(row.get("target_token"))
            for row in group_rows
            if row.get("predicted_token") not in {None, ""} and row.get("target_token") not in {None, ""}
        ]
        rows.append(
            {
                "arm": arm,
                "split": "holdout",
                "latent_rule": latent_rule,
                "position_index": position_index,
                "token_count": len(losses),
                "mean_ce_loss": (sum(losses) / float(len(losses))) if losses else None,
                "min_ce_loss": min(losses) if losses else None,
                "max_ce_loss": max(losses) if losses else None,
                "accuracy": (sum(1 for value in correct if value) / float(len(correct))) if correct else None,
                "mechanism_labels_used_for_scoring_only": True,
            }
        )
    return rows


def _residual_budget_accounting_rows(arm_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sparse_rows = [
        row
        for row in arm_metrics
        if row.get("arm") in {"promoted_contextual_topk2", "intervention_trained_sparse_topk2"}
    ]
    active_dense_mlp_rows = [
        row
        for row in arm_metrics
        if row.get("control_budget_role") == "active_proxy_matched_dense_mlp_control"
    ]
    stored_dense_mlp_rows = [
        row
        for row in arm_metrics
        if row.get("control_budget_role") == "stored_parameter_matched_dense_mlp_upper_bound"
    ]
    best_sparse = _best_ce_row(sparse_rows)
    best_active_dense_mlp = _best_ce_row(active_dense_mlp_rows)
    best_stored_dense_mlp = _best_ce_row(stored_dense_mlp_rows)
    best_sparse_norm = _metric_float(best_sparse.get("residual_l2")) if best_sparse else None
    best_sparse_active = _metric_float(best_sparse.get("active_parameters_proxy")) if best_sparse else None
    best_sparse_stored = _metric_float(best_sparse.get("stored_parameters")) if best_sparse else None
    best_active_flops = _flop_proxy(best_active_dense_mlp) if best_active_dense_mlp else None
    best_stored_flops = _flop_proxy(best_stored_dense_mlp) if best_stored_dense_mlp else None
    rows: list[dict[str, Any]] = []
    for row in arm_metrics:
        active_parameters = _metric_float(row.get("active_parameters_proxy"))
        stored_parameters = _metric_float(row.get("stored_parameters"))
        residual_l2 = _metric_float(row.get("residual_l2"))
        flop_proxy = _flop_proxy(row)
        rows.append(
            {
                "arm": row.get("arm", ""),
                "family": row.get("family", ""),
                "router": row.get("router", ""),
                "support_mode": row.get("support_mode", ""),
                "control_budget_role": row.get("control_budget_role", ""),
                "holdout_ce": _metric_float(row.get("holdout_ce")),
                "residual_l2": residual_l2,
                "residual_l2_ratio_vs_best_sparse": _safe_ratio(residual_l2, best_sparse_norm),
                "active_parameters_proxy": active_parameters,
                "stored_parameters": stored_parameters,
                "flop_proxy_per_token": flop_proxy,
                "active_to_best_sparse_ratio": _safe_ratio(active_parameters, best_sparse_active),
                "stored_to_best_sparse_ratio": _safe_ratio(stored_parameters, best_sparse_stored),
                "flop_to_best_active_dense_mlp_ratio": _safe_ratio(flop_proxy, best_active_flops),
                "flop_to_best_stored_dense_mlp_ratio": _safe_ratio(flop_proxy, best_stored_flops),
                "best_sparse_arm": best_sparse.get("arm", "") if best_sparse else "",
                "best_active_matched_dense_mlp_arm": (
                    best_active_dense_mlp.get("arm", "") if best_active_dense_mlp else ""
                ),
                "best_stored_matched_dense_mlp_arm": (
                    best_stored_dense_mlp.get("arm", "") if best_stored_dense_mlp else ""
                ),
                "accounting_is_proxy": True,
                "flop_proxy_notes": (
                    "Approximate per-token residual-path multiply-add proxy; excludes frozen base and decoder."
                ),
            }
        )
    return rows


def _router_value_regret_decomposition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        arm = str(row.get("arm", ""))
        rule = str(row.get("latent_rule", ""))
        grouped.setdefault((arm, "all"), []).append(row)
        grouped.setdefault((arm, rule), []).append(row)

    output_rows: list[dict[str, Any]] = []
    for (arm, latent_rule), group_rows in sorted(grouped.items()):
        oracle_sizes = [_metric_float(row.get("oracle_support_size")) for row in group_rows]
        oracle_sizes = [size for size in oracle_sizes if size is not None]
        oracle_regrets = [_metric_float(row.get("oracle_regret")) for row in group_rows]
        oracle_regrets = [value for value in oracle_regrets if value is not None]
        one_swap_recovery = [
            _metric_float(row.get("one_swap_recovery_fraction")) for row in group_rows
        ]
        one_swap_recovery = [value for value in one_swap_recovery if value is not None]
        learned_equals_oracle = [
            str(row.get("learned_support", "")) == str(row.get("oracle_support", ""))
            for row in group_rows
            if row.get("learned_support") not in {None, ""}
            and row.get("oracle_support") not in {None, ""}
        ]
        best_pair_minus_singleton = [
            _delta_value(row.get("best_pair_ce_loss"), row.get("best_singleton_ce_loss"))
            for row in group_rows
        ]
        best_pair_minus_singleton = [
            value for value in best_pair_minus_singleton if value is not None
        ]
        mean_regret = _mean_optional(group_rows, "oracle_regret")
        output_rows.append(
            {
                "arm": arm,
                "latent_rule": latent_rule,
                "token_count": len(group_rows),
                "mean_learned_ce_loss": _mean_optional(group_rows, "learned_ce_loss"),
                "mean_oracle_ce_loss": _mean_optional(group_rows, "oracle_ce_loss"),
                "mean_oracle_regret": mean_regret,
                "max_oracle_regret": max(oracle_regrets) if oracle_regrets else None,
                "positive_oracle_regret_fraction": (
                    sum(1 for value in oracle_regrets if value > 1e-12) / float(len(oracle_regrets))
                    if oracle_regrets
                    else None
                ),
                "mean_pair_oracle_regret": _mean_optional(group_rows, "pair_oracle_regret"),
                "mean_one_swap_regret": _mean_optional(group_rows, "one_swap_regret"),
                "mean_one_swap_recovery_fraction": (
                    sum(one_swap_recovery) / float(len(one_swap_recovery))
                    if one_swap_recovery
                    else None
                ),
                "oracle_pair_fraction": (
                    sum(1 for size in oracle_sizes if int(size) == 2) / float(len(oracle_sizes))
                    if oracle_sizes
                    else None
                ),
                "mean_best_pair_ce_minus_best_singleton_ce": (
                    sum(best_pair_minus_singleton) / float(len(best_pair_minus_singleton))
                    if best_pair_minus_singleton
                    else None
                ),
                "learned_support_matches_oracle_fraction": (
                    sum(1 for value in learned_equals_oracle if value) / float(len(learned_equals_oracle))
                    if learned_equals_oracle
                    else None
                ),
                "router_value_status": (
                    "router_regret_present"
                    if mean_regret is not None and mean_regret > 0.02
                    else "low_router_regret"
                    if mean_regret is not None
                    else "missing"
                ),
                "mechanism_labels_used_for_scoring_only": True,
            }
        )
    return output_rows


def _router_regret_ceiling_budget_rows(
    arm_metrics: list[dict[str, Any]],
    oracle_support_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not arm_metrics or not oracle_support_rows:
        return []

    by_arm = {str(row.get("arm")): row for row in arm_metrics}
    regret_all = {
        str(row.get("arm")): row
        for row in _router_value_regret_decomposition_rows(oracle_support_rows)
        if row.get("latent_rule") == "all"
    }
    token_position = by_arm.get("token_position_router_topk2")
    active_controls = [
        row
        for row in arm_metrics
        if row.get("control_budget_role") == "active_proxy_matched_dense_mlp_control"
    ]
    stored_controls = [
        row
        for row in arm_metrics
        if row.get("control_budget_role") == "stored_parameter_matched_dense_mlp_upper_bound"
    ]
    best_active = _best_ce_row(active_controls)
    best_stored = _best_ce_row(stored_controls)
    token_position_ce = _metric_float(token_position.get("holdout_ce")) if token_position else None
    active_ce = _metric_float(best_active.get("holdout_ce")) if best_active else None
    stored_ce = _metric_float(best_stored.get("holdout_ce")) if best_stored else None

    rows: list[dict[str, Any]] = []
    for arm in sorted(regret_all):
        arm_row = by_arm.get(arm)
        if arm_row is None:
            continue
        learned_ce = _metric_float(arm_row.get("holdout_ce"))
        regret_row = regret_all[arm]
        oracle_ce = _metric_float(regret_row.get("mean_oracle_ce_loss"))
        oracle_gain = _delta_value(learned_ce, oracle_ce)
        token_gap = _positive_gap(learned_ce, token_position_ce)
        active_gap = _positive_gap(learned_ce, active_ce)
        stored_gap = _positive_gap(learned_ce, stored_ce)
        token_closes = _ceiling_closes_gap(oracle_gain, token_gap)
        active_closes = _ceiling_closes_gap(oracle_gain, active_gap)
        stored_closes = _ceiling_closes_gap(oracle_gain, stored_gap)
        rows.append(
            {
                "arm": arm,
                "family": arm_row.get("family", ""),
                "router": arm_row.get("router", ""),
                "support_mode": arm_row.get("support_mode", ""),
                "learned_holdout_ce": learned_ce,
                "mean_learned_token_ce": _metric_float(regret_row.get("mean_learned_ce_loss")),
                "oracle_support_ce_ceiling": oracle_ce,
                "oracle_support_ce_gain": oracle_gain,
                "mean_oracle_regret": _metric_float(regret_row.get("mean_oracle_regret")),
                "token_position_null_arm": "token_position_router_topk2" if token_position else "",
                "token_position_null_ce": token_position_ce,
                "learned_ce_gap_to_token_position_null": token_gap,
                "oracle_gain_fraction_of_token_position_gap": _safe_ratio(oracle_gain, token_gap),
                "router_only_can_close_token_position_gap": token_closes,
                "oracle_ce_beats_token_position_null": _ce_beats_or_ties(oracle_ce, token_position_ce),
                "active_matched_control_arm": best_active.get("arm", "") if best_active else "",
                "active_matched_control_ce": active_ce,
                "learned_ce_gap_to_active_matched_control": active_gap,
                "oracle_gain_fraction_of_active_matched_gap": _safe_ratio(oracle_gain, active_gap),
                "router_only_can_close_active_matched_gap": active_closes,
                "oracle_ce_beats_active_matched_control": _ce_beats_or_ties(oracle_ce, active_ce),
                "stored_matched_control_arm": best_stored.get("arm", "") if best_stored else "",
                "stored_matched_control_ce": stored_ce,
                "learned_ce_gap_to_stored_matched_control": stored_gap,
                "oracle_gain_fraction_of_stored_matched_gap": _safe_ratio(oracle_gain, stored_gap),
                "router_only_can_close_stored_matched_gap": stored_closes,
                "oracle_ce_beats_stored_matched_control": _ce_beats_or_ties(oracle_ce, stored_ce),
                "router_only_sufficiency_status": _router_only_sufficiency_status(
                    token_closes,
                    active_closes,
                    stored_closes,
                    oracle_gain,
                    stored_gap,
                ),
                "mechanism_labels_used_for_scoring_only": True,
                "interpretation": (
                    "Diagnostic only: the oracle ceiling uses holdout labels to measure how much a perfect "
                    "support selector could recover with fixed sparse values."
                ),
            }
        )
    return rows


def _positive_gap(learned_ce: float | None, comparator_ce: float | None) -> float | None:
    gap = _delta_value(learned_ce, comparator_ce)
    if gap is None:
        return None
    return max(0.0, gap)


def _ceiling_closes_gap(oracle_gain: float | None, positive_gap: float | None) -> bool | None:
    if oracle_gain is None or positive_gap is None:
        return None
    return oracle_gain + 1e-12 >= positive_gap


def _ce_beats_or_ties(candidate_ce: float | None, comparator_ce: float | None) -> bool | None:
    if candidate_ce is None or comparator_ce is None:
        return None
    return candidate_ce <= comparator_ce + 1e-12


def _router_only_sufficiency_status(
    token_closes: bool | None,
    active_closes: bool | None,
    stored_closes: bool | None,
    oracle_gain: float | None,
    stored_gap: float | None,
) -> str:
    if token_closes is None or active_closes is None or stored_closes is None:
        return "missing_comparator"
    if stored_closes:
        return "router_ceiling_closes_stored_gap"
    if active_closes:
        return "router_ceiling_closes_active_gap_but_not_stored_gap"
    if token_closes:
        return "router_ceiling_only_addresses_null_gap"
    if oracle_gain is not None and stored_gap is not None and oracle_gain < stored_gap:
        return "router_ceiling_insufficient_for_stored_gap"
    return "router_ceiling_insufficient"


def _flop_proxy(row: dict[str, Any]) -> float | None:
    active_parameters = _metric_float(row.get("active_parameters_proxy"))
    if active_parameters is None:
        return None
    family = str(row.get("family", ""))
    if family == "base":
        return 0.0
    if family == "mlp_control":
        return float(active_parameters * 2.0)
    if family in {"dense_control", "sparse", "sparse_null", "router_null"}:
        return float(active_parameters * 2.0)
    return float(active_parameters)


def _best_ce_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [(row, _metric_float(row.get("holdout_ce"))) for row in rows]
    scored = [(row, ce) for row, ce in scored if ce is not None]
    if not scored:
        return None
    return min(scored, key=lambda item: item[1])[0]


def _stored_parameters(adapter: Any) -> int:
    return int(sum(parameter.numel() for parameter in adapter.parameters()))


def _sparse_contextual_stored_parameter_count(
    *,
    hidden_dim: int,
    num_columns: int,
    atoms_per_column: int,
    contextual_router_hidden_dim: int,
) -> int:
    contextual_feature_dim = hidden_dim * 5 + 3
    linear_router = hidden_dim * num_columns
    contextual_router = (
        2 * contextual_feature_dim
        + contextual_feature_dim * contextual_router_hidden_dim
        + contextual_router_hidden_dim
        + contextual_router_hidden_dim * num_columns
    )
    atom_logits = num_columns * atoms_per_column
    atom_values = num_columns * atoms_per_column * hidden_dim
    return int(linear_router + contextual_router + atom_logits + atom_values)


def _dense_rank_for_parameter_floor(hidden_dim: int, parameter_floor: int) -> int:
    if parameter_floor <= 0:
        return 1
    return max(1, (parameter_floor + (2 * hidden_dim) - 1) // (2 * hidden_dim))


def _mlp_rank_for_parameter_floor(hidden_dim: int, parameter_floor: int) -> int:
    if parameter_floor <= 0:
        return 1
    fixed_parameters = (3 * hidden_dim)
    per_width_parameters = (2 * hidden_dim) + 1
    remaining = max(0, parameter_floor - fixed_parameters)
    return max(1, (remaining + per_width_parameters - 1) // per_width_parameters)


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


def _comparator_controls(
    *,
    support_width: int,
    include_teacher_distillation: bool = False,
) -> list[dict[str, Any]]:
    controls = [
        _control("base_no_residual", "base", 0, "none", False, "shared frozen decoder reference"),
        _control("promoted_contextual_topk2", "sparse", support_width, "contextual_mlp", True, "current promoted sparse routing comparator"),
        _control("intervention_trained_sparse_topk2", "sparse", support_width, "contextual_mlp_with_intervention_loss", True, "new opt-in sparse arm with necessity/selectivity loss"),
        _control("random_support_topk2", "sparse_null", support_width, "random_support", True, "same active support width random null"),
        _control("fixed_support_topk2", "sparse_null", support_width, "fixed_support", True, "same support every token null"),
        _control("token_position_router_topk2", "router_null", support_width, "token_position_only", True, "shortcut router control with no hidden mechanism evidence"),
        _control("dense_rank_norm_matched", "dense_control", 0, "dense_rank_norm", True, "active-proxy matched dense low-rank residual"),
        _control("low_churn_mlp_active_matched", "mlp_control", 0, "low_churn_mlp", True, "active-proxy matched low-churn MLP residual"),
        _control("dense_stored_parameter_matched", "dense_control", 0, "dense_rank_norm", True, "stored-parameter matched dense upper-bound residual"),
        _control("low_churn_mlp_stored_parameter_matched", "mlp_control", 0, "low_churn_mlp", True, "stored-parameter matched low-churn MLP upper-bound residual"),
        _control("random_initialized_same_params", "random_null", support_width, "random_initialized", True, "same stored parameter random residual null"),
        _control("shuffled_mechanism_label_null", "causal_null", support_width, "contextual_mlp", True, "evaluation scoring with shuffled latent mechanism labels"),
    ]
    if include_teacher_distillation:
        controls.extend(
            [
                _control(
                    "dense_teacher_distilled_sparse_topk2",
                    "sparse",
                    support_width,
                    "contextual_mlp_dense_teacher_residual_distillation",
                    True,
                    "opt-in sparse student trained with dense-teacher residual MSE and evaluated as hard top-k2",
                ),
                _control(
                    "shuffled_teacher_distilled_sparse_topk2",
                    "teacher_null",
                    support_width,
                    "contextual_mlp_shuffled_teacher_residual_distillation",
                    True,
                    "same sparse student objective with shuffled dense-teacher residual targets as a null",
                ),
            ]
        )
    return controls


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
                _criterion("training_smoke_ce_by_rule_position_present", bool(training_smoke["ce_by_rule_position"]), "hard", "CPU training smoke must emit per-rule/per-position CE decomposition", len(training_smoke["ce_by_rule_position"]), "missing CE decomposition by rule/position"),
                _criterion("training_smoke_residual_budget_accounting_present", bool(training_smoke["residual_budget_accounting"]), "hard", "CPU training smoke must emit residual norm and FLOP proxy accounting", len(training_smoke["residual_budget_accounting"]), "missing residual budget accounting"),
                _criterion("training_smoke_required_arms_present", _required_controls().issubset(arm_names), "hard", "CPU training smoke must cover required comparator arms", sorted(arm_names), "missing CPU smoke arm"),
                _criterion("training_smoke_intervention_metrics_present", any(row.get("metric_values_available") is True for row in intervention_rows), "hard", "intervention rows must contain measured values", len(intervention_rows), "missing measured intervention metrics"),
                _criterion("training_smoke_commutator_metrics_present", any(row.get("metric_values_available") is True for row in commutator_rows), "hard", "commutator rows must contain measured values", len(commutator_rows), "missing measured commutator metrics"),
            ]
        )
    return rows


def _local_scientific_gate_rows(training_smoke: dict[str, Any] | None) -> list[dict[str, Any]]:
    if training_smoke is None:
        return []
    arm_metrics = training_smoke["arm_metrics"]
    intervention_rows = training_smoke["per_mechanism_interventions"]
    commutator_rows = training_smoke["commutator_rows"]
    forgetting_rows = training_smoke["forgetting_rows"]
    by_arm = {row["arm"]: row for row in arm_metrics}
    sparse_arms = ["promoted_contextual_topk2", "intervention_trained_sparse_topk2"]
    active_dense_mlp_arms = ["dense_rank_norm_matched", "low_churn_mlp_active_matched"]
    stored_dense_mlp_arms = [
        "dense_stored_parameter_matched",
        "low_churn_mlp_stored_parameter_matched",
    ]
    dense_mlp_arms = active_dense_mlp_arms + stored_dense_mlp_arms
    router_null_arms = ["random_support_topk2", "fixed_support_topk2", "token_position_router_topk2"]
    base_ce = _metric_float(by_arm.get("base_no_residual", {}).get("holdout_ce"))
    sparse_ces = _arm_metric_values(by_arm, sparse_arms, "holdout_ce")
    active_dense_mlp_ces = _arm_metric_values(by_arm, active_dense_mlp_arms, "holdout_ce")
    stored_dense_mlp_ces = _arm_metric_values(by_arm, stored_dense_mlp_arms, "holdout_ce")
    sparse_norms = _arm_metric_values(by_arm, sparse_arms, "residual_l2")
    active_dense_mlp_norms = _arm_metric_values(by_arm, active_dense_mlp_arms, "residual_l2")
    sparse_stored = _arm_metric_values(by_arm, sparse_arms, "stored_parameters")
    stored_dense_mlp_stored = _arm_metric_values(by_arm, stored_dense_mlp_arms, "stored_parameters")
    sparse_active = _arm_metric_values(by_arm, sparse_arms, "active_parameters_proxy")
    active_dense_mlp_active = _arm_metric_values(by_arm, active_dense_mlp_arms, "active_parameters_proxy")
    sparse_selectivity = _mean_metric(intervention_rows, sparse_arms, "selectivity")
    sparse_necessity = _mean_metric(intervention_rows, sparse_arms, "necessity")
    sparse_leakage = _mean_metric(intervention_rows, sparse_arms, "off_target_leakage")
    null_selectivity = _mean_metric(intervention_rows, router_null_arms, "selectivity")
    sparse_commutator = _mean_metric(commutator_rows, sparse_arms, "finite_update_commutator_l2")
    dense_mlp_commutator = _mean_metric(commutator_rows, dense_mlp_arms, "finite_update_commutator_l2")
    forgetting_nonzero = any(
        abs(_metric_float(row.get("forgetting_delta"))) > 1e-12
        or abs(_metric_float(row.get("functional_churn"))) > 1e-12
        for row in forgetting_rows
    )
    return [
        _scientific_gate(
            "measured_training_smoke_metrics_present",
            bool(arm_metrics and intervention_rows and commutator_rows and forgetting_rows),
            "arm, per-mechanism intervention, commutator, and forgetting rows must all be measured",
            {
                "arm_rows": len(arm_metrics),
                "intervention_rows": len(intervention_rows),
                "commutator_rows": len(commutator_rows),
                "forgetting_rows": len(forgetting_rows),
            },
            "missing measured local synthetic rows",
        ),
        _scientific_gate(
            "ce_guardrail_vs_base",
            bool(base_ce is not None and sparse_ces and max(sparse_ces) <= base_ce + 0.05),
            "both sparse arms must stay within +0.05 CE of base/no-residual",
            {"base_ce": base_ce, "sparse_ces": sparse_ces, "tolerance": 0.05},
            "sparse CE violates base guardrail",
        ),
        _scientific_gate(
            "ce_competitive_with_dense_or_mlp",
            bool(
                sparse_ces
                and active_dense_mlp_ces
                and min(sparse_ces) <= min(active_dense_mlp_ces) + 0.02
            ),
            "best sparse arm must be within +0.02 CE of the best active-proxy matched dense/MLP control",
            {
                "best_sparse_ce": min(sparse_ces) if sparse_ces else None,
                "best_active_matched_dense_mlp_ce": min(active_dense_mlp_ces) if active_dense_mlp_ces else None,
                "best_stored_matched_dense_mlp_ce": min(stored_dense_mlp_ces) if stored_dense_mlp_ces else None,
                "tolerance": 0.02,
            },
            "sparse arm is not CE-competitive with the best active-proxy matched dense/MLP control",
        ),
        _scientific_gate(
            "positive_sparse_intervention_selectivity",
            bool(sparse_selectivity is not None and sparse_selectivity > 0.0),
            "mean sparse selected-column selectivity must be positive",
            {"mean_sparse_selectivity": sparse_selectivity},
            "sparse selected-column ablation/drop-in selectivity is not positive",
        ),
        _scientific_gate(
            "sparse_selectivity_beats_router_nulls",
            bool(
                sparse_selectivity is not None
                and null_selectivity is not None
                and sparse_selectivity >= null_selectivity + 0.005
            ),
            "mean sparse selectivity must beat random/fixed/token-position support nulls by at least 0.005",
            {"mean_sparse_selectivity": sparse_selectivity, "mean_router_null_selectivity": null_selectivity, "margin": 0.005},
            "sparse selectivity does not beat router nulls by margin",
        ),
        _scientific_gate(
            "off_target_leakage_below_necessity",
            bool(
                sparse_leakage is not None
                and sparse_necessity is not None
                and sparse_leakage <= sparse_necessity
            ),
            "mean sparse off-target leakage must not exceed mean necessity",
            {"mean_sparse_off_target_leakage": sparse_leakage, "mean_sparse_necessity": sparse_necessity},
            "off-target leakage exceeds selected-column necessity",
        ),
        _scientific_gate(
            "finite_update_commutator_not_worse_than_dense_mlp",
            bool(
                sparse_commutator is not None
                and dense_mlp_commutator is not None
                and sparse_commutator <= dense_mlp_commutator * 1.1
            ),
            "mean sparse finite-update commutator must be no worse than 1.1x dense/MLP controls",
            {"mean_sparse_commutator_l2": sparse_commutator, "mean_dense_mlp_commutator_l2": dense_mlp_commutator, "max_ratio": 1.1},
            "sparse finite-update commutator is worse than dense/MLP controls",
        ),
        _scientific_gate(
            "residual_norm_budget",
            bool(
                sparse_norms
                and active_dense_mlp_norms
                and max(sparse_norms) <= max(active_dense_mlp_norms)
            ),
            "sparse residual norm must not exceed the larger active-proxy matched dense/MLP residual norm",
            {
                "max_sparse_residual_l2": max(sparse_norms) if sparse_norms else None,
                "max_active_matched_dense_mlp_residual_l2": (
                    max(active_dense_mlp_norms) if active_dense_mlp_norms else None
                ),
            },
            "sparse residual norm exceeds active-proxy matched dense/MLP norm budget",
        ),
        _scientific_gate(
            "stored_parameter_budget",
            bool(
                sparse_stored
                and stored_dense_mlp_stored
                and max(sparse_stored) <= max(stored_dense_mlp_stored)
            ),
            "sparse stored parameters must not exceed the larger stored-parameter matched dense/MLP upper-bound control",
            {
                "max_sparse_stored_parameters": max(sparse_stored) if sparse_stored else None,
                "max_stored_matched_dense_mlp_stored_parameters": (
                    max(stored_dense_mlp_stored) if stored_dense_mlp_stored else None
                ),
            },
            "sparse stored-parameter budget is not covered by stored-matched dense/MLP controls",
        ),
        _scientific_gate(
            "active_parameter_budget",
            bool(
                sparse_active
                and active_dense_mlp_active
                and max(sparse_active) <= max(active_dense_mlp_active)
            ),
            "sparse active parameter proxy must not exceed the larger active-proxy matched dense/MLP active proxy",
            {
                "max_sparse_active_parameters": max(sparse_active) if sparse_active else None,
                "max_active_matched_dense_mlp_active_parameters": (
                    max(active_dense_mlp_active) if active_dense_mlp_active else None
                ),
            },
            "sparse active-parameter budget exceeds active-proxy matched dense/MLP controls",
        ),
        _scientific_gate(
            "forgetting_and_functional_churn_measured",
            forgetting_nonzero,
            "forgetting rows must contain non-placeholder forgetting_delta or functional_churn values before interpretation",
            {"any_nonzero_forgetting_or_churn": forgetting_nonzero},
            "forgetting/functional-churn rows are still placeholder zeros",
        ),
    ]


def _scientific_gate(
    criterion: str,
    passed: bool,
    requirement: str,
    observed: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "severity": "scientific",
        "requirement": requirement,
        "observed": observed,
        "failure_reason": "" if passed else failure_reason,
    }


def _mean_metric(rows: list[dict[str, Any]], arms: list[str], metric: str) -> float | None:
    values = [
        _metric_float(row.get(metric))
        for row in rows
        if row.get("arm") in arms and row.get("metric_values_available") is True
    ]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / float(len(values))


def _arm_metric_values(by_arm: dict[str, dict[str, Any]], arms: list[str], metric: str) -> list[float]:
    values = [
        _metric_float(by_arm[arm].get(metric))
        for arm in arms
        if arm in by_arm
    ]
    return [value for value in values if value is not None]


def _metric_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _required_controls() -> set[str]:
    return {
        "base_no_residual",
        "promoted_contextual_topk2",
        "intervention_trained_sparse_topk2",
        "random_support_topk2",
        "fixed_support_topk2",
        "token_position_router_topk2",
        "dense_rank_norm_matched",
        "low_churn_mlp_active_matched",
        "dense_stored_parameter_matched",
        "low_churn_mlp_stored_parameter_matched",
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
    if training_smoke.get("router_regret_ceiling_budget"):
        return (
            "use the router-regret ceiling budget to decide whether a sequence-heldout support-head diagnostic is scientifically justified; keep GPU and promotion blocked"
        )
    if any(row.get("teacher_distillation_enabled") is True for row in training_smoke.get("arm_metrics", [])):
        if training_smoke.get("teacher_distillation_closeout"):
            return (
                "test whether reducing learned support/router regret, rather than value distillation, closes the remaining sparse gap"
            )
        if training_smoke.get("router_value_regret_decomposition"):
            return (
                "close the teacher-distillation branch as non-improving on this local seed and test whether lower router regret, not value distillation, explains the remaining sparse gap"
            )
        return (
            "inspect dense-teacher sparse student versus shuffled-teacher null, then add oracle-support regret decomposition for the teacher-distilled arm if it clears CE guardrails"
        )
    if training_smoke.get("ce_by_rule_position") and training_smoke.get("residual_budget_accounting"):
        return (
            "add an opt-in soft-to-hard dense-teacher-distilled sparse arm with a shuffled-teacher null in the local synthetic pregate"
        )
    if training_smoke.get("oracle_support_sparse_topk2"):
        return (
            "add per-rule/per-position CE decomposition and residual norm/FLOP accounting before any GPU validation or modularity claim"
        )
    if _stored_upper_bound_gap_status(training_smoke) == "fail":
        return (
            "add oracle-support sparse diagnostics on the same local seed-17 pregate before any GPU validation or modularity claim"
        )
    return (
        "add oracle-support sparse diagnostics and per-rule CE decomposition before any second seed or GPU validation"
    )


def _stored_upper_bound_gap_status(training_smoke: dict[str, Any] | None) -> str:
    if training_smoke is None:
        return "not_run"
    rows = training_smoke.get("ce_gap_decomposition", [])
    sparse_rows = [row for row in rows if row.get("arm") == "promoted_contextual_topk2"]
    if not sparse_rows:
        return "missing"
    gap = _metric_float(sparse_rows[0].get("best_sparse_ce_minus_best_stored_matched_dense_mlp_ce"))
    if gap is None:
        return "missing"
    return "fail" if gap > 0.02 else "pass"


def _oracle_support_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "mean_oracle_regret": None,
            "mean_pair_oracle_regret": None,
            "mean_one_swap_regret": None,
            "mean_one_swap_recovery_fraction": None,
            "interpretation": "oracle-support sparse top-k2 diagnostics were not run",
        }
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in sorted({str(row["arm"]) for row in rows}):
        arm_rows = [row for row in rows if row.get("arm") == arm]
        by_arm[arm] = {
            "row_count": len(arm_rows),
            "mean_learned_ce_loss": _mean_optional(arm_rows, "learned_ce_loss"),
            "mean_oracle_ce_loss": _mean_optional(arm_rows, "oracle_ce_loss"),
            "mean_oracle_regret": _mean_optional(arm_rows, "oracle_regret"),
            "mean_pair_oracle_regret": _mean_optional(arm_rows, "pair_oracle_regret"),
            "mean_one_swap_regret": _mean_optional(arm_rows, "one_swap_regret"),
            "mean_one_swap_recovery_fraction": _mean_optional(
                arm_rows,
                "one_swap_recovery_fraction",
            ),
        }
    return {
        "row_count": len(rows),
        "mean_oracle_regret": _mean_optional(rows, "oracle_regret"),
        "mean_pair_oracle_regret": _mean_optional(rows, "pair_oracle_regret"),
        "mean_one_swap_regret": _mean_optional(rows, "one_swap_regret"),
        "mean_one_swap_recovery_fraction": _mean_optional(
            rows,
            "one_swap_recovery_fraction",
        ),
        "by_arm": by_arm,
        "interpretation": (
            "Diagnostic only: oracle supports use holdout labels for scoring and expose router/value regret; "
            "they are not deployable training evidence."
        ),
    }


def _router_value_regret_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "worst_arm": "",
            "worst_latent_rule": "",
            "worst_mean_oracle_regret": None,
            "interpretation": "router/value regret decomposition was not run",
        }
    all_rows = [row for row in rows if row.get("latent_rule") == "all"]
    worst = max(
        rows,
        key=lambda row: _metric_float(row.get("mean_oracle_regret")) or float("-inf"),
    )
    worst_all = (
        max(
            all_rows,
            key=lambda row: _metric_float(row.get("mean_oracle_regret")) or float("-inf"),
        )
        if all_rows
        else {}
    )
    return {
        "row_count": len(rows),
        "arm_count": len({str(row.get("arm", "")) for row in all_rows}),
        "worst_arm": worst.get("arm", ""),
        "worst_latent_rule": worst.get("latent_rule", ""),
        "worst_mean_oracle_regret": _metric_float(worst.get("mean_oracle_regret")),
        "worst_overall_arm": worst_all.get("arm", ""),
        "worst_overall_mean_oracle_regret": _metric_float(
            worst_all.get("mean_oracle_regret")
        ),
        "interpretation": (
            "Diagnostic only: learned-vs-oracle support gaps use holdout labels for scoring. "
            "High regret indicates router/support selection headroom for already-trained sparse values; "
            "it does not prove deployable causal modularity."
        ),
    }


def _router_regret_ceiling_budget_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "best_router_only_status": "",
            "max_oracle_support_ce_gain": None,
            "min_stored_gap": None,
            "interpretation": "router-regret ceiling budget was not run",
        }
    max_gain_row = max(
        rows,
        key=lambda row: _metric_float(row.get("oracle_support_ce_gain")) or float("-inf"),
    )
    stored_gaps = [
        _metric_float(row.get("learned_ce_gap_to_stored_matched_control")) for row in rows
    ]
    stored_gaps = [gap for gap in stored_gaps if gap is not None]
    statuses = {str(row.get("router_only_sufficiency_status", "")) for row in rows}
    if "router_ceiling_closes_stored_gap" in statuses:
        best_status = "router_ceiling_closes_stored_gap"
    elif "router_ceiling_closes_active_gap_but_not_stored_gap" in statuses:
        best_status = "router_ceiling_closes_active_gap_but_not_stored_gap"
    elif "router_ceiling_only_addresses_null_gap" in statuses:
        best_status = "router_ceiling_only_addresses_null_gap"
    else:
        best_status = "router_ceiling_insufficient_for_stored_gap"
    return {
        "row_count": len(rows),
        "arm_count": len({str(row.get("arm", "")) for row in rows}),
        "best_router_only_status": best_status,
        "max_oracle_support_ce_gain": _metric_float(max_gain_row.get("oracle_support_ce_gain")),
        "max_gain_arm": max_gain_row.get("arm", ""),
        "min_stored_gap": min(stored_gaps) if stored_gaps else None,
        "stored_gap_closable_by_router_only": any(
            row.get("router_only_can_close_stored_matched_gap") is True for row in rows
        ),
        "active_gap_closable_by_router_only": any(
            row.get("router_only_can_close_active_matched_gap") is True for row in rows
        ),
        "token_position_gap_closable_by_router_only": any(
            row.get("router_only_can_close_token_position_gap") is True for row in rows
        ),
        "interpretation": (
            "Diagnostic only: compares the fixed-value oracle-support ceiling with token/position null, "
            "active-matched controls, and stored-matched upper bounds before any support-head or GPU branch."
        ),
    }


def _teacher_distillation_closeout_rows(
    arm_metrics: list[dict[str, Any]],
    oracle_support_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_arm = {str(row.get("arm")): row for row in arm_metrics}
    distilled = by_arm.get("dense_teacher_distilled_sparse_topk2")
    shuffled = by_arm.get("shuffled_teacher_distilled_sparse_topk2")
    if distilled is None or shuffled is None:
        return []

    sparse_arms = [
        "promoted_contextual_topk2",
        "intervention_trained_sparse_topk2",
    ]
    best_sparse = _best_ce_row([by_arm[arm] for arm in sparse_arms if arm in by_arm])
    active_controls = [
        row
        for row in arm_metrics
        if row.get("control_budget_role") == "active_proxy_matched_dense_mlp_control"
    ]
    stored_controls = [
        row
        for row in arm_metrics
        if row.get("control_budget_role") == "stored_parameter_matched_dense_mlp_upper_bound"
    ]
    best_active = _best_ce_row(active_controls)
    best_stored = _best_ce_row(stored_controls)

    regret_rows = _router_value_regret_decomposition_rows(oracle_support_rows)
    regret_all = {
        str(row.get("arm")): row
        for row in regret_rows
        if row.get("latent_rule") == "all"
    }
    distilled_ce = _metric_float(distilled.get("holdout_ce"))
    shuffled_ce = _metric_float(shuffled.get("holdout_ce"))
    best_sparse_ce = _metric_float(best_sparse.get("holdout_ce")) if best_sparse else None
    best_active_ce = _metric_float(best_active.get("holdout_ce")) if best_active else None
    best_stored_ce = _metric_float(best_stored.get("holdout_ce")) if best_stored else None
    distilled_regret = _metric_float(
        regret_all.get("dense_teacher_distilled_sparse_topk2", {}).get("mean_oracle_regret")
    )
    promoted_regret = _metric_float(
        regret_all.get("promoted_contextual_topk2", {}).get("mean_oracle_regret")
    )
    intervention_regret = _metric_float(
        regret_all.get("intervention_trained_sparse_topk2", {}).get("mean_oracle_regret")
    )
    best_sparse_regret = min(
        [
            value
            for value in (promoted_regret, intervention_regret)
            if value is not None
        ],
        default=None,
    )
    distilled_improves_best_sparse = (
        distilled_ce is not None
        and best_sparse_ce is not None
        and distilled_ce < best_sparse_ce - 0.01
    )
    distilled_beats_shuffled = (
        distilled_ce is not None
        and shuffled_ce is not None
        and distilled_ce < shuffled_ce - 0.01
    )
    active_guardrail_ok = (
        distilled_ce is not None
        and best_active_ce is not None
        and distilled_ce <= best_active_ce + 0.02
    )
    stored_guardrail_ok = (
        distilled_ce is not None
        and best_stored_ce is not None
        and distilled_ce <= best_stored_ce + 0.02
    )
    router_regret_remains = distilled_regret is not None and distilled_regret > 0.02
    if not distilled_improves_best_sparse:
        closeout_status = "closed_non_improving"
        interpretation = (
            "dense-teacher sparse distillation does not improve best trained sparse CE on this seed"
        )
    elif not distilled_beats_shuffled:
        closeout_status = "closed_teacher_target_not_specific"
        interpretation = (
            "dense-teacher sparse distillation does not beat the shuffled-teacher null by the CE guardrail"
        )
    elif router_regret_remains:
        closeout_status = "router_regret_explains_remaining_gap"
        interpretation = (
            "teacher target helps CE guardrails, but oracle-support regret remains above threshold"
        )
    elif not (active_guardrail_ok and stored_guardrail_ok):
        closeout_status = "value_distillation_insufficient_vs_dense_controls"
        interpretation = (
            "teacher target is structured but still fails active or stored dense/MLP CE guardrails"
        )
    else:
        closeout_status = "needs_repeat_before_branch_reopen"
        interpretation = (
            "teacher distillation clears local guardrails once; repeat before reopening the branch"
        )

    return [
        {
            "branch": "dense_teacher_distilled_sparse_topk2",
            "closeout_status": closeout_status,
            "distilled_holdout_ce": distilled_ce,
            "shuffled_null_holdout_ce": shuffled_ce,
            "distilled_minus_shuffled_holdout_ce": _delta_value(distilled_ce, shuffled_ce),
            "best_sparse_arm": best_sparse.get("arm", "") if best_sparse else "",
            "best_sparse_holdout_ce": best_sparse_ce,
            "distilled_minus_best_sparse_ce": _delta_value(distilled_ce, best_sparse_ce),
            "best_active_matched_dense_mlp_arm": best_active.get("arm", "") if best_active else "",
            "best_active_matched_dense_mlp_ce": best_active_ce,
            "distilled_minus_best_active_matched_dense_mlp_ce": _delta_value(
                distilled_ce,
                best_active_ce,
            ),
            "best_stored_matched_dense_mlp_arm": best_stored.get("arm", "") if best_stored else "",
            "best_stored_matched_dense_mlp_ce": best_stored_ce,
            "distilled_minus_best_stored_matched_dense_mlp_ce": _delta_value(
                distilled_ce,
                best_stored_ce,
            ),
            "distilled_mean_oracle_regret": distilled_regret,
            "best_existing_sparse_mean_oracle_regret": best_sparse_regret,
            "distilled_minus_best_existing_sparse_oracle_regret": _delta_value(
                distilled_regret,
                best_sparse_regret,
            ),
            "distilled_improves_best_sparse_by_0p01": distilled_improves_best_sparse,
            "distilled_beats_shuffled_by_0p01": distilled_beats_shuffled,
            "active_guardrail_ok_0p02": active_guardrail_ok,
            "stored_guardrail_ok_0p02": stored_guardrail_ok,
            "router_regret_remains_above_0p02": router_regret_remains,
            "mechanism_labels_used_for_scoring_only": True,
            "interpretation": interpretation,
        }
    ]


def _residual_budget_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "best_sparse_arm": "",
            "best_active_matched_dense_mlp_arm": "",
            "best_stored_matched_dense_mlp_arm": "",
            "interpretation": "residual budget accounting was not run",
        }
    sparse_rows = [
        row
        for row in rows
        if row.get("arm") in {"promoted_contextual_topk2", "intervention_trained_sparse_topk2"}
    ]
    active_dense_mlp_rows = [
        row
        for row in rows
        if row.get("control_budget_role") == "active_proxy_matched_dense_mlp_control"
    ]
    stored_dense_mlp_rows = [
        row
        for row in rows
        if row.get("control_budget_role") == "stored_parameter_matched_dense_mlp_upper_bound"
    ]
    best_sparse = _best_ce_row(sparse_rows)
    best_active_dense_mlp = _best_ce_row(active_dense_mlp_rows)
    best_stored_dense_mlp = _best_ce_row(stored_dense_mlp_rows)
    return {
        "row_count": len(rows),
        "best_sparse_arm": best_sparse.get("arm", "") if best_sparse else "",
        "best_sparse_residual_l2": _metric_float(best_sparse.get("residual_l2")) if best_sparse else None,
        "best_sparse_flop_proxy_per_token": (
            _metric_float(best_sparse.get("flop_proxy_per_token")) if best_sparse else None
        ),
        "best_active_matched_dense_mlp_arm": (
            best_active_dense_mlp.get("arm", "") if best_active_dense_mlp else ""
        ),
        "best_active_matched_dense_mlp_flop_proxy_per_token": (
            _metric_float(best_active_dense_mlp.get("flop_proxy_per_token"))
            if best_active_dense_mlp
            else None
        ),
        "best_stored_matched_dense_mlp_arm": (
            best_stored_dense_mlp.get("arm", "") if best_stored_dense_mlp else ""
        ),
        "best_stored_matched_dense_mlp_flop_proxy_per_token": (
            _metric_float(best_stored_dense_mlp.get("flop_proxy_per_token"))
            if best_stored_dense_mlp
            else None
        ),
        "interpretation": (
            "Diagnostic only: residual norms and FLOP proxies expose budget confounds; "
            "they are not causal-modularity evidence by themselves."
        ),
    }


def _teacher_distillation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    teacher_rows = [
        row
        for row in rows
        if row.get("arm")
        in {
            "dense_teacher_distilled_sparse_topk2",
            "shuffled_teacher_distilled_sparse_topk2",
        }
    ]
    if not teacher_rows:
        return {
            "row_count": 0,
            "distilled_holdout_ce": None,
            "shuffled_null_holdout_ce": None,
            "distilled_minus_shuffled_holdout_ce": None,
            "distilled_teacher_residual_mse": None,
            "shuffled_teacher_residual_mse": None,
            "interpretation": "dense-teacher sparse distillation was not included in this run",
        }
    by_arm = {str(row.get("arm")): row for row in teacher_rows}
    distilled = by_arm.get("dense_teacher_distilled_sparse_topk2", {})
    shuffled = by_arm.get("shuffled_teacher_distilled_sparse_topk2", {})
    distilled_ce = _metric_float(distilled.get("holdout_ce"))
    shuffled_ce = _metric_float(shuffled.get("holdout_ce"))
    return {
        "row_count": len(teacher_rows),
        "distilled_holdout_ce": distilled_ce,
        "shuffled_null_holdout_ce": shuffled_ce,
        "distilled_minus_shuffled_holdout_ce": _delta_value(distilled_ce, shuffled_ce),
        "distilled_teacher_residual_mse": _metric_float(distilled.get("teacher_residual_mse")),
        "shuffled_teacher_residual_mse": _metric_float(shuffled.get("teacher_residual_mse")),
        "interpretation": (
            "Diagnostic only: the teacher is trained locally from the same synthetic episodes; "
            "teacher labels do not enter mechanism scoring, and final sparse evaluation uses hard top-k2 support."
        ),
    }


def _teacher_distillation_closeout_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "closeout_status": "not_run",
            "interpretation": "teacher-distillation closeout was not run",
        }
    row = rows[0]
    return {
        "row_count": len(rows),
        "closeout_status": row.get("closeout_status", ""),
        "distilled_minus_best_sparse_ce": _metric_float(
            row.get("distilled_minus_best_sparse_ce")
        ),
        "distilled_minus_shuffled_holdout_ce": _metric_float(
            row.get("distilled_minus_shuffled_holdout_ce")
        ),
        "distilled_mean_oracle_regret": _metric_float(
            row.get("distilled_mean_oracle_regret")
        ),
        "router_regret_remains_above_0p02": bool(
            row.get("router_regret_remains_above_0p02")
        ),
        "interpretation": row.get("interpretation", ""),
    }


def _mean_optional(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [_metric_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / float(len(values))


def _delta_value(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _safe_ratio(left: Any, right: Any) -> float | None:
    if left is None or right in {None, 0, 0.0}:
        return None
    return float(left) / float(right)


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
    local_scientific_gate_rows: list[dict[str, Any]],
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
    _write_csv(out_dir / "local_scientific_gates.csv", local_scientific_gate_rows)
    _write_csv(out_dir / "arm_metrics.csv", [] if training_smoke is None else training_smoke["arm_metrics"])
    _write_csv(
        out_dir / "ce_gap_decomposition.csv",
        [] if training_smoke is None else training_smoke["ce_gap_decomposition"],
    )
    _write_csv(
        out_dir / "oracle_support_sparse_topk2.csv",
        [] if training_smoke is None else training_smoke["oracle_support_sparse_topk2"],
    )
    _write_csv(
        out_dir / "router_value_regret_decomposition.csv",
        [] if training_smoke is None else training_smoke["router_value_regret_decomposition"],
    )
    _write_csv(
        out_dir / "router_regret_ceiling_budget.csv",
        [] if training_smoke is None else training_smoke["router_regret_ceiling_budget"],
    )
    _write_csv(
        out_dir / "teacher_distillation_closeout.csv",
        [] if training_smoke is None else training_smoke["teacher_distillation_closeout"],
    )
    _write_csv(out_dir / "per_token_metrics.csv", [] if training_smoke is None else training_smoke["per_token_metrics"])
    _write_csv(
        out_dir / "ce_by_rule_position.csv",
        [] if training_smoke is None else training_smoke["ce_by_rule_position"],
    )
    _write_csv(
        out_dir / "residual_budget_accounting.csv",
        [] if training_smoke is None else training_smoke["residual_budget_accounting"],
    )
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
        f"- Local scientific gates: `{summary['local_scientific_gate_status']}`",
        f"- Oracle-support sparse top-k2 rows: `{summary['oracle_support_sparse_topk2_row_count']}`",
        f"- Router/value regret decomposition rows: `{summary['router_value_regret_decomposition_row_count']}`",
        f"- Router-regret ceiling budget rows: `{summary['router_regret_ceiling_budget_row_count']}`",
        f"- Teacher distillation closeout rows: `{summary['teacher_distillation_closeout_row_count']}`",
        f"- CE by rule/position rows: `{summary['ce_by_rule_position_row_count']}`",
        f"- Residual budget accounting rows: `{summary['residual_budget_accounting_row_count']}`",
        f"- Teacher distillation included: `{summary['teacher_distillation_included']}`",
        f"- Teacher distillation arm count: `{summary['teacher_distillation_arm_count']}`",
        "",
        "This artifact implements the major-pivot pregate by generating same-vocabulary synthetic latent-rule episodes and the comparator/intervention schemas. It intentionally separates artifact readiness from scientific gates; GPU validation remains blocked until local scientific gates pass.",
        "",
        "The oracle-support sparse top-k2 artifact is diagnostic only: it uses holdout labels to score exhaustive singleton/pair supports for trained sparse values and must not be treated as deployable training evidence.",
        "",
        "The router/value regret decomposition summarizes the oracle-support rows by arm and latent rule. It separates support-selection headroom from the adequacy of the already-trained sparse values, but still uses evaluator-only holdout labels.",
        "",
        "The router-regret ceiling budget compares the fixed-value oracle-support ceiling against token/position nulls, active-matched dense/MLP controls, and stored-matched dense/MLP upper bounds. It is the local fail-closed budget check before any support-head or GPU branch.",
        "",
        "The CE by rule/position and residual budget accounting artifacts are local diagnostics. FLOP values are residual-path proxies only and exclude the frozen base and decoder.",
        "",
        "The opt-in dense-teacher distillation arms are diagnostics only. The dense teacher is trained on the synthetic training episodes, sparse students are evaluated with hard top-k2 supports, and the shuffled-teacher arm is a null for teacher residual target structure.",
        "",
        "The teacher-distillation closeout artifact compares the dense-teacher sparse student to the shuffled-teacher null, the best existing sparse arm, active-matched dense/MLP controls, stored-matched upper bounds, and oracle-support regret. It is a local branch triage artifact, not promotion evidence.",
        "",
        "## Next Step",
        "",
        str(summary["selected_next_step"]),
    ]
    if summary["local_scientific_failures"]:
        lines.extend(["", "## Local Scientific Gate Failures"])
        lines.extend(
            f"- {row['criterion']}: {row['failure_reason']}"
            for row in summary["local_scientific_failures"]
        )
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
    parser.add_argument("--include-teacher-distillation", action="store_true")
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
        include_teacher_distillation=args.include_teacher_distillation,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
