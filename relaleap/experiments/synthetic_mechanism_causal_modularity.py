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
    "support_head_sequence_heldout_diagnostic.csv",
    "router_only_branch_selection.csv",
    "teacher_distillation_closeout.csv",
    "value_capacity_core_periphery_diagnostic.csv",
    "core_periphery_sparse_value_capacity_probe.csv",
    "core_periphery_update_stability_bracket.csv",
    "core_periphery_branch_closeout.csv",
    "sparse_value_redesign_selector.csv",
    "budget_normalized_gated_value_mixture_pregate.csv",
    "budget_normalized_gated_value_mixture_closeout.csv",
    "soft_mixture_low_churn_dense_modular_design.csv",
    "pc_core_periphery_residual_inference_pregate.csv",
    "pc_residual_inference_mechanism_inspection.csv",
    "pc_error_target_inference_path_audit.csv",
    "pc_decoder_adjoint_target_alignment_probe.csv",
    "pc_decoder_adjoint_minimal_retrain_probe.csv",
    "pc_decoder_adjoint_closeout.csv",
    "pc_amortized_error_pregate_design.csv",
    "pc_amortized_error_pregate.csv",
    "pc_amortized_error_pregate_closeout.csv",
    "transformer_acsr_design.csv",
    "transformer_acsr_cpu_smoke_pilot.csv",
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
    value_head_variant: str = "none"
    core_rank: int = 0
    periphery_rank: int = 0
    core_lr_scale: float = 1.0
    core_drift_penalty_weight: float = 0.0
    periphery_l1_weight: float = 0.0
    residual_norm_clip: float = 0.0
    gate_l1_weight: float = 0.0
    pc_inference_steps: int = 0
    pc_error_prediction_weight: float = 0.0
    pc_error_target_mode: str = "decoder_embedding_delta"


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
            "Accepted the latest GPT-5.5-Pro major-pivot review: close the one-site decoder-adjoint PC retrain "
            "path with explicit row-level null/control evidence before any GPU validation, and redirect only to "
            "a tiny label-free amortized multi-site PC pregate. Ben should be notified because this changes the "
            "active PC branch direction."
        ),
        "strategic_change_level": (
            "major"
            if training_smoke is not None and training_smoke.get("pc_decoder_adjoint_closeout")
            else "minor"
        ),
        "notify_ben": bool(training_smoke is not None and training_smoke.get("pc_decoder_adjoint_closeout")),
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
        "support_head_sequence_heldout_diagnostic_row_count": (
            len(training_smoke["support_head_sequence_heldout_diagnostic"]) if training_smoke is not None else 0
        ),
        "support_head_sequence_heldout_diagnostic_primary_result": (
            _support_head_sequence_heldout_diagnostic_summary(
                training_smoke["support_head_sequence_heldout_diagnostic"]
            )
            if training_smoke is not None
            else None
        ),
        "router_only_branch_selection_row_count": (
            len(training_smoke["router_only_branch_selection"]) if training_smoke is not None else 0
        ),
        "router_only_branch_selection_primary_result": (
            _router_only_branch_selection_summary(training_smoke["router_only_branch_selection"])
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
        "value_capacity_core_periphery_diagnostic_row_count": (
            len(training_smoke["value_capacity_core_periphery_diagnostic"]) if training_smoke is not None else 0
        ),
        "value_capacity_core_periphery_diagnostic_primary_result": (
            _value_capacity_core_periphery_diagnostic_summary(
                training_smoke["value_capacity_core_periphery_diagnostic"]
            )
            if training_smoke is not None
            else None
        ),
        "core_periphery_sparse_value_capacity_probe_row_count": (
            len(training_smoke["core_periphery_sparse_value_capacity_probe"]) if training_smoke is not None else 0
        ),
        "core_periphery_sparse_value_capacity_probe_primary_result": (
            _core_periphery_sparse_value_capacity_probe_summary(
                training_smoke["core_periphery_sparse_value_capacity_probe"]
            )
            if training_smoke is not None
            else None
        ),
        "core_periphery_update_stability_bracket_row_count": (
            len(training_smoke["core_periphery_update_stability_bracket"]) if training_smoke is not None else 0
        ),
        "core_periphery_update_stability_bracket_primary_result": (
            _core_periphery_update_stability_bracket_summary(
                training_smoke["core_periphery_update_stability_bracket"]
            )
            if training_smoke is not None
            else None
        ),
        "core_periphery_branch_closeout_row_count": (
            len(training_smoke["core_periphery_branch_closeout"]) if training_smoke is not None else 0
        ),
        "core_periphery_branch_closeout_primary_result": (
            _core_periphery_branch_closeout_summary(
                training_smoke["core_periphery_branch_closeout"]
            )
            if training_smoke is not None
            else None
        ),
        "sparse_value_redesign_selector_row_count": (
            len(training_smoke["sparse_value_redesign_selector"]) if training_smoke is not None else 0
        ),
        "sparse_value_redesign_selector_primary_result": (
            _sparse_value_redesign_selector_summary(
                training_smoke["sparse_value_redesign_selector"]
            )
            if training_smoke is not None
            else None
        ),
        "budget_normalized_gated_value_mixture_pregate_row_count": (
            len(training_smoke["budget_normalized_gated_value_mixture_pregate"]) if training_smoke is not None else 0
        ),
        "budget_normalized_gated_value_mixture_pregate_primary_result": (
            _budget_normalized_gated_value_mixture_pregate_summary(
                training_smoke["budget_normalized_gated_value_mixture_pregate"]
            )
            if training_smoke is not None
            else None
        ),
        "budget_normalized_gated_value_mixture_closeout_row_count": (
            len(training_smoke["budget_normalized_gated_value_mixture_closeout"]) if training_smoke is not None else 0
        ),
        "budget_normalized_gated_value_mixture_closeout_primary_result": (
            _budget_normalized_gated_value_mixture_closeout_summary(
                training_smoke["budget_normalized_gated_value_mixture_closeout"]
            )
            if training_smoke is not None
            else None
        ),
        "soft_mixture_low_churn_dense_modular_design_row_count": (
            len(training_smoke["soft_mixture_low_churn_dense_modular_design"]) if training_smoke is not None else 0
        ),
        "soft_mixture_low_churn_dense_modular_design_primary_result": (
            _soft_mixture_low_churn_dense_modular_design_summary(
                training_smoke["soft_mixture_low_churn_dense_modular_design"]
            )
            if training_smoke is not None
            else None
        ),
        "pc_core_periphery_residual_inference_pregate_row_count": (
            len(training_smoke["pc_core_periphery_residual_inference_pregate"]) if training_smoke is not None else 0
        ),
        "pc_core_periphery_residual_inference_pregate_primary_result": (
            _pc_core_periphery_residual_inference_pregate_summary(
                training_smoke["pc_core_periphery_residual_inference_pregate"]
            )
            if training_smoke is not None
            else None
        ),
        "pc_residual_inference_mechanism_inspection_row_count": (
            len(training_smoke["pc_residual_inference_mechanism_inspection"]) if training_smoke is not None else 0
        ),
        "pc_residual_inference_mechanism_inspection_primary_result": (
            _pc_residual_inference_mechanism_inspection_summary(
                training_smoke["pc_residual_inference_mechanism_inspection"]
            )
            if training_smoke is not None
            else None
        ),
        "pc_error_target_inference_path_audit_row_count": (
            len(training_smoke["pc_error_target_inference_path_audit"]) if training_smoke is not None else 0
        ),
        "pc_error_target_inference_path_audit_primary_result": (
            _pc_error_target_inference_path_audit_summary(
                training_smoke["pc_error_target_inference_path_audit"]
            )
            if training_smoke is not None
            else None
        ),
        "pc_decoder_adjoint_target_alignment_probe_row_count": (
            len(training_smoke["pc_decoder_adjoint_target_alignment_probe"]) if training_smoke is not None else 0
        ),
        "pc_decoder_adjoint_target_alignment_probe_primary_result": (
            _pc_decoder_adjoint_target_alignment_probe_summary(
                training_smoke["pc_decoder_adjoint_target_alignment_probe"]
            )
            if training_smoke is not None
            else None
        ),
        "pc_decoder_adjoint_minimal_retrain_probe_row_count": (
            len(training_smoke["pc_decoder_adjoint_minimal_retrain_probe"]) if training_smoke is not None else 0
        ),
        "pc_decoder_adjoint_minimal_retrain_probe_primary_result": (
            _pc_decoder_adjoint_minimal_retrain_probe_summary(
                training_smoke["pc_decoder_adjoint_minimal_retrain_probe"]
            )
            if training_smoke is not None
            else None
        ),
        "pc_decoder_adjoint_closeout_row_count": (
            len(training_smoke["pc_decoder_adjoint_closeout"]) if training_smoke is not None else 0
        ),
        "pc_decoder_adjoint_closeout_primary_result": (
            _pc_decoder_adjoint_closeout_summary(
                training_smoke["pc_decoder_adjoint_closeout"]
            )
            if training_smoke is not None
            else None
        ),
        "pc_amortized_error_pregate_design_row_count": (
            len(training_smoke["pc_amortized_error_pregate_design"]) if training_smoke is not None else 0
        ),
        "pc_amortized_error_pregate_design_primary_result": (
            _pc_amortized_error_pregate_design_summary(
                training_smoke["pc_amortized_error_pregate_design"]
            )
            if training_smoke is not None
            else None
        ),
        "pc_amortized_error_pregate_row_count": (
            len(training_smoke["pc_amortized_error_pregate"]) if training_smoke is not None else 0
        ),
        "pc_amortized_error_pregate_primary_result": (
            _pc_amortized_error_pregate_summary(
                training_smoke["pc_amortized_error_pregate"]
            )
            if training_smoke is not None
            else None
        ),
        "pc_amortized_error_pregate_closeout_row_count": (
            len(training_smoke["pc_amortized_error_pregate_closeout"]) if training_smoke is not None else 0
        ),
        "pc_amortized_error_pregate_closeout_primary_result": (
            _pc_amortized_error_pregate_closeout_summary(
                training_smoke["pc_amortized_error_pregate_closeout"]
            )
            if training_smoke is not None
            else None
        ),
        "transformer_acsr_design_row_count": (
            len(training_smoke["transformer_acsr_design"]) if training_smoke is not None else 0
        ),
        "transformer_acsr_design_primary_result": (
            _transformer_acsr_design_summary(training_smoke["transformer_acsr_design"])
            if training_smoke is not None
            else None
        ),
        "transformer_acsr_cpu_smoke_pilot_row_count": (
            len(training_smoke["transformer_acsr_cpu_smoke_pilot"]) if training_smoke is not None else 0
        ),
        "transformer_acsr_cpu_smoke_pilot_primary_result": (
            _transformer_acsr_cpu_smoke_pilot_summary(training_smoke["transformer_acsr_cpu_smoke_pilot"])
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
    support_head_rows: list[dict[str, Any]] = []

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
        optimizer = (
            torch.optim.AdamW(
                adapter.optimizer_param_groups(learning_rate)
                if hasattr(adapter, "optimizer_param_groups")
                else trainable,
                lr=learning_rate,
            )
            if trainable
            else None
        )
        core_reference = (
            adapter.core_parameter_snapshot()
            if hasattr(adapter, "core_parameter_snapshot")
            else []
        )
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
            if spec.core_drift_penalty_weight > 0.0 and hasattr(adapter, "core_drift_penalty"):
                loss = loss + spec.core_drift_penalty_weight * adapter.core_drift_penalty(core_reference)
            if spec.periphery_l1_weight > 0.0 and hasattr(adapter, "periphery_l1"):
                loss = loss + spec.periphery_l1_weight * adapter.periphery_l1()
            if spec.gate_l1_weight > 0.0 and hasattr(adapter, "gate_l1"):
                loss = loss + spec.gate_l1_weight * adapter.gate_l1(train_hidden)
            if spec.pc_error_prediction_weight > 0.0 and hasattr(adapter, "residual_error_prediction_loss"):
                loss = loss + spec.pc_error_prediction_weight * adapter.residual_error_prediction_loss(
                    train_hidden,
                    train_inputs,
                    train_targets,
                    base.lm_head.weight,
                    decode=base.decode,
                    target_mode=spec.pc_error_target_mode,
                    shuffle_targets=spec.shuffled_teacher_null,
                    F=F,
                )
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
                "anchor_kl_weight": spec.anchor_kl_weight,
                "teacher_distillation_weight": spec.teacher_distillation_weight,
                "shuffled_teacher_null": spec.shuffled_teacher_null,
                "teacher_residual_mse": teacher_residual_mse,
                "stored_parameters": _stored_parameters(adapter),
                "stored_parameter_floor": spec.stored_parameter_floor,
                "active_parameters_proxy": _synthetic_active_parameters(spec, hidden_dim),
                "control_budget_role": spec.control_budget_role,
                "value_head_variant": spec.value_head_variant,
                "core_rank": spec.core_rank,
                "periphery_rank": spec.periphery_rank,
                "core_lr_scale": spec.core_lr_scale,
                "core_drift_penalty_weight": spec.core_drift_penalty_weight,
                "periphery_l1_weight": spec.periphery_l1_weight,
                "gate_l1_weight": spec.gate_l1_weight,
                "residual_norm_clip": spec.residual_norm_clip,
                "residual_norm_clipped": spec.residual_norm_clip > 0.0,
                "pc_inference_steps": spec.pc_inference_steps,
                "pc_error_prediction_weight": spec.pc_error_prediction_weight,
                "pc_error_target_mode": spec.pc_error_target_mode,
                "core_parameter_drift_l2": (
                    adapter.core_parameter_drift_l2(core_reference)
                    if hasattr(adapter, "core_parameter_drift_l2")
                    else None
                ),
                "periphery_l1": (
                    float(adapter.periphery_l1().detach().item())
                    if hasattr(adapter, "periphery_l1")
                    else None
                ),
                "mean_gate_value": (
                    float(adapter.mean_gate_value(holdout_hidden).detach().item())
                    if hasattr(adapter, "mean_gate_value")
                    else None
                ),
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
        if spec.family in {
            "sparse",
            "core_periphery_sparse",
            "pc_core_periphery_sparse",
            "pc_amortized_sparse",
        } and spec.support_mode == "learned" and spec.top_k == 2:
            train_oracle_rows = _oracle_support_sparse_topk2_rows(
                arm=spec.name,
                split="train",
                adapter=adapter,
                hidden=train_hidden,
                inputs=train_inputs,
                targets=train_targets,
                rules=train_rules,
                spec=spec,
                decode=base.decode,
                F=F,
                torch=torch,
            )
            holdout_oracle_rows = _oracle_support_sparse_topk2_rows(
                arm=spec.name,
                split="holdout",
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
            oracle_support_rows.extend(holdout_oracle_rows)
            support_head_rows.extend(
                _support_head_sequence_heldout_diagnostic_rows(
                    arm=spec.name,
                    adapter=adapter,
                    train_hidden=train_hidden,
                    train_inputs=train_inputs,
                    train_targets=train_targets,
                    train_oracle_rows=train_oracle_rows,
                    holdout_hidden=holdout_hidden,
                    holdout_inputs=holdout_inputs,
                    holdout_targets=holdout_targets,
                    holdout_oracle_rows=holdout_oracle_rows,
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
    router_regret_ceiling_budget = _router_regret_ceiling_budget_rows(
        arm_metrics,
        oracle_support_rows,
    )
    router_only_branch_selection = _router_only_branch_selection_rows(
        router_regret_ceiling_budget,
        support_head_rows,
    )
    residual_budget_accounting = _residual_budget_accounting_rows(arm_metrics)
    value_capacity_core_periphery_diagnostic = _value_capacity_core_periphery_diagnostic_rows(
        arm_metrics=arm_metrics,
        residual_budget_rows=residual_budget_accounting,
        commutator_rows=commutator_rows,
        forgetting_rows=forgetting_rows,
        router_only_branch_rows=router_only_branch_selection,
    )
    core_periphery_sparse_value_capacity_probe = _core_periphery_sparse_value_capacity_probe_rows(
        arm_metrics=arm_metrics,
        residual_budget_rows=residual_budget_accounting,
        commutator_rows=commutator_rows,
        forgetting_rows=forgetting_rows,
        router_only_branch_rows=router_only_branch_selection,
    )
    core_periphery_update_stability_bracket = _core_periphery_update_stability_bracket_rows(
        arm_metrics=arm_metrics,
        commutator_rows=commutator_rows,
        forgetting_rows=forgetting_rows,
    )
    core_periphery_branch_closeout = _core_periphery_branch_closeout_rows(
        probe_rows=core_periphery_sparse_value_capacity_probe,
        stability_rows=core_periphery_update_stability_bracket,
    )
    sparse_value_redesign_selector = _sparse_value_redesign_selector_rows(
        closeout_rows=core_periphery_branch_closeout,
        arm_metrics=arm_metrics,
        residual_budget_rows=residual_budget_accounting,
        commutator_rows=commutator_rows,
        forgetting_rows=forgetting_rows,
    )
    budget_normalized_gated_value_mixture_pregate = _budget_normalized_gated_value_mixture_pregate_rows(
        redesign_rows=sparse_value_redesign_selector,
        arm_metrics=arm_metrics,
        residual_budget_rows=residual_budget_accounting,
        commutator_rows=commutator_rows,
        forgetting_rows=forgetting_rows,
    )
    budget_normalized_gated_value_mixture_closeout = _budget_normalized_gated_value_mixture_closeout_rows(
        pregate_rows=budget_normalized_gated_value_mixture_pregate,
    )
    soft_mixture_low_churn_dense_modular_design = _soft_mixture_low_churn_dense_modular_design_rows(
        closeout_rows=budget_normalized_gated_value_mixture_closeout,
        arm_metrics=arm_metrics,
        residual_budget_rows=residual_budget_accounting,
        commutator_rows=commutator_rows,
        forgetting_rows=forgetting_rows,
    )
    pc_core_periphery_residual_inference_pregate = _pc_core_periphery_residual_inference_pregate_rows(
        gated_pregate_rows=budget_normalized_gated_value_mixture_pregate,
        arm_metrics=arm_metrics,
        residual_budget_rows=residual_budget_accounting,
        commutator_rows=commutator_rows,
        forgetting_rows=forgetting_rows,
    )
    pc_residual_inference_mechanism_inspection = _pc_residual_inference_mechanism_inspection_rows(
        pc_pregate_rows=pc_core_periphery_residual_inference_pregate,
        arm_metrics=arm_metrics,
        residual_budget_rows=residual_budget_accounting,
        commutator_rows=commutator_rows,
        forgetting_rows=forgetting_rows,
    )
    pc_error_target_inference_path_audit = _pc_error_target_inference_path_audit_rows(
        pc_inspection_rows=pc_residual_inference_mechanism_inspection,
        arm_metrics=arm_metrics,
        residual_budget_rows=residual_budget_accounting,
        commutator_rows=commutator_rows,
        forgetting_rows=forgetting_rows,
    )
    pc_decoder_adjoint_target_alignment_probe = _pc_decoder_adjoint_target_alignment_probe_rows(
        pc_audit_rows=pc_error_target_inference_path_audit,
        hidden=holdout_hidden,
        targets=holdout_targets,
        decode=base.decode,
        decoder_weight=base.lm_head.weight,
        F=F,
        torch=torch,
    )
    pc_decoder_adjoint_minimal_retrain_probe = _pc_decoder_adjoint_minimal_retrain_probe_rows(
        target_probe_rows=pc_decoder_adjoint_target_alignment_probe,
        arm_metrics=arm_metrics,
        train_hidden=train_hidden,
        train_targets=train_targets,
        holdout_hidden=holdout_hidden,
        holdout_targets=holdout_targets,
        decode=base.decode,
        decoder_weight=base.lm_head.weight,
        training_steps=max(4, training_steps),
        learning_rate=learning_rate,
        seed=seed,
        torch=torch,
        nn=nn,
        F=F,
    )
    pc_decoder_adjoint_closeout = _pc_decoder_adjoint_closeout_rows(
        target_probe_rows=pc_decoder_adjoint_target_alignment_probe,
        retrain_probe_rows=pc_decoder_adjoint_minimal_retrain_probe,
        arm_metrics=arm_metrics,
        residual_budget_rows=residual_budget_accounting,
        commutator_rows=commutator_rows,
        forgetting_rows=forgetting_rows,
    )
    pc_amortized_error_pregate_design = _pc_amortized_error_pregate_design_rows(
        closeout_rows=pc_decoder_adjoint_closeout,
        arm_metrics=arm_metrics,
    )
    pc_amortized_error_pregate = _pc_amortized_error_pregate_rows(
        design_rows=pc_amortized_error_pregate_design,
        arm_metrics=arm_metrics,
        residual_budget_rows=residual_budget_accounting,
        commutator_rows=commutator_rows,
        forgetting_rows=forgetting_rows,
    )
    pc_amortized_error_pregate_closeout = _pc_amortized_error_pregate_closeout_rows(
        pregate_rows=pc_amortized_error_pregate,
    )
    transformer_acsr_design = _transformer_acsr_design_rows(
        pc_closeout_rows=pc_amortized_error_pregate_closeout,
        soft_design_rows=soft_mixture_low_churn_dense_modular_design,
        arm_metrics=arm_metrics,
        residual_budget_rows=residual_budget_accounting,
        commutator_rows=commutator_rows,
        forgetting_rows=forgetting_rows,
    )
    transformer_acsr_cpu_smoke_pilot = _transformer_acsr_cpu_smoke_pilot_rows(
        design_rows=transformer_acsr_design,
        train_hidden=train_hidden,
        train_inputs=train_inputs,
        holdout_hidden=holdout_hidden,
        holdout_inputs=holdout_inputs,
        holdout_targets=holdout_targets,
        decode=base.decode,
        training_steps=training_steps,
        learning_rate=learning_rate,
        seed=seed,
        torch=torch,
        nn=nn,
        F=F,
    )
    return {
        "arm_metrics": arm_metrics,
        "ce_gap_decomposition": _ce_gap_decomposition_rows(arm_metrics),
        "oracle_support_sparse_topk2": oracle_support_rows,
        "router_value_regret_decomposition": _router_value_regret_decomposition_rows(
            oracle_support_rows
        ),
        "router_regret_ceiling_budget": router_regret_ceiling_budget,
        "support_head_sequence_heldout_diagnostic": support_head_rows,
        "router_only_branch_selection": router_only_branch_selection,
        "teacher_distillation_closeout": _teacher_distillation_closeout_rows(
            arm_metrics,
            oracle_support_rows,
        ),
        "value_capacity_core_periphery_diagnostic": value_capacity_core_periphery_diagnostic,
        "core_periphery_sparse_value_capacity_probe": core_periphery_sparse_value_capacity_probe,
        "core_periphery_update_stability_bracket": core_periphery_update_stability_bracket,
        "core_periphery_branch_closeout": core_periphery_branch_closeout,
        "sparse_value_redesign_selector": sparse_value_redesign_selector,
        "budget_normalized_gated_value_mixture_pregate": budget_normalized_gated_value_mixture_pregate,
        "budget_normalized_gated_value_mixture_closeout": budget_normalized_gated_value_mixture_closeout,
        "soft_mixture_low_churn_dense_modular_design": soft_mixture_low_churn_dense_modular_design,
        "pc_core_periphery_residual_inference_pregate": pc_core_periphery_residual_inference_pregate,
        "pc_residual_inference_mechanism_inspection": pc_residual_inference_mechanism_inspection,
        "pc_error_target_inference_path_audit": pc_error_target_inference_path_audit,
        "pc_decoder_adjoint_target_alignment_probe": pc_decoder_adjoint_target_alignment_probe,
        "pc_decoder_adjoint_minimal_retrain_probe": pc_decoder_adjoint_minimal_retrain_probe,
        "pc_decoder_adjoint_closeout": pc_decoder_adjoint_closeout,
        "pc_amortized_error_pregate_design": pc_amortized_error_pregate_design,
        "pc_amortized_error_pregate": pc_amortized_error_pregate,
        "pc_amortized_error_pregate_closeout": pc_amortized_error_pregate_closeout,
        "transformer_acsr_design": transformer_acsr_design,
        "transformer_acsr_cpu_smoke_pilot": transformer_acsr_cpu_smoke_pilot,
        "per_token_metrics": per_token_metrics,
        "ce_by_rule_position": ce_by_rule_position,
        "residual_budget_accounting": residual_budget_accounting,
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
    core_rank = max(1, hidden_dim // 6)
    periphery_rank = max(1, atoms_per_column)
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
            "core_periphery_sparse_topk2",
            "core_periphery_sparse",
            support_width,
            num_columns,
            atoms_per_column,
            "contextual_mlp",
            stored_parameter_floor=sparse_stored_parameter_floor,
            value_head_variant="core_periphery_both",
            core_rank=core_rank,
            periphery_rank=periphery_rank,
            core_lr_scale=0.25,
            core_drift_penalty_weight=0.05,
            periphery_l1_weight=1e-4,
            residual_norm_clip=0.075,
        ),
        _SyntheticArmSpec(
            "flat_column_value_mlp_topk2",
            "core_periphery_control",
            support_width,
            num_columns,
            atoms_per_column,
            "contextual_mlp",
            stored_parameter_floor=sparse_stored_parameter_floor,
            control_budget_role="flat_same_router_value_capacity_control",
            value_head_variant="flat_column_mlp",
            core_rank=0,
            periphery_rank=max(1, core_rank + periphery_rank),
            periphery_l1_weight=1e-4,
            residual_norm_clip=0.075,
        ),
        _SyntheticArmSpec(
            "core_only_sparse_topk2",
            "core_periphery_control",
            support_width,
            num_columns,
            atoms_per_column,
            "contextual_mlp",
            stored_parameter_floor=sparse_stored_parameter_floor,
            control_budget_role="core_only_value_capacity_control",
            value_head_variant="core_only",
            core_rank=core_rank,
            periphery_rank=periphery_rank,
            core_lr_scale=0.25,
            core_drift_penalty_weight=0.05,
        ),
        _SyntheticArmSpec(
            "periphery_only_sparse_topk2",
            "core_periphery_control",
            support_width,
            num_columns,
            atoms_per_column,
            "contextual_mlp",
            stored_parameter_floor=sparse_stored_parameter_floor,
            control_budget_role="periphery_only_value_capacity_control",
            value_head_variant="periphery_only",
            core_rank=core_rank,
            periphery_rank=periphery_rank,
            periphery_l1_weight=1e-4,
            residual_norm_clip=0.075,
        ),
        _SyntheticArmSpec(
            "core_periphery_stability_slow_core_topk2",
            "core_periphery_control",
            support_width,
            num_columns,
            atoms_per_column,
            "contextual_mlp",
            anchor_kl_weight=0.02,
            stored_parameter_floor=sparse_stored_parameter_floor,
            control_budget_role="update_stability_slow_core_anchor_control",
            value_head_variant="core_periphery_both",
            core_rank=core_rank,
            periphery_rank=periphery_rank,
            core_lr_scale=0.10,
            core_drift_penalty_weight=0.15,
            periphery_l1_weight=1e-4,
            residual_norm_clip=0.075,
        ),
        _SyntheticArmSpec(
            "flat_column_value_mlp_anchor_topk2",
            "core_periphery_control",
            support_width,
            num_columns,
            atoms_per_column,
            "contextual_mlp",
            anchor_kl_weight=0.02,
            stored_parameter_floor=sparse_stored_parameter_floor,
            control_budget_role="update_stability_flat_anchor_control",
            value_head_variant="flat_column_mlp",
            core_rank=0,
            periphery_rank=max(1, core_rank + periphery_rank),
            periphery_l1_weight=1e-4,
            residual_norm_clip=0.075,
        ),
        _SyntheticArmSpec(
            "budget_normalized_gated_low_rank_value_mixture_topk2",
            "core_periphery_control",
            support_width,
            num_columns,
            atoms_per_column,
            "contextual_mlp",
            anchor_kl_weight=0.01,
            stored_parameter_floor=sparse_stored_parameter_floor,
            control_budget_role="budget_normalized_gated_low_rank_value_mixture_pregate",
            value_head_variant="gated_low_rank_value_mixture",
            core_rank=max(1, core_rank),
            periphery_rank=max(1, periphery_rank),
            core_lr_scale=0.25,
            core_drift_penalty_weight=0.05,
            periphery_l1_weight=5e-5,
            residual_norm_clip=0.055,
            gate_l1_weight=1e-4,
        ),
        _SyntheticArmSpec(
            "pc_core_periphery_residual_inference_topk2",
            "pc_core_periphery_sparse",
            support_width,
            num_columns,
            atoms_per_column,
            "contextual_mlp",
            anchor_kl_weight=0.02,
            stored_parameter_floor=sparse_stored_parameter_floor,
            control_budget_role="primary_pc_core_periphery_residual_inference_pregate",
            value_head_variant="pc_core_periphery_residual_inference",
            core_rank=max(1, core_rank),
            periphery_rank=max(1, periphery_rank),
            core_lr_scale=0.10,
            core_drift_penalty_weight=0.15,
            periphery_l1_weight=1e-4,
            residual_norm_clip=0.055,
            gate_l1_weight=1e-4,
            pc_inference_steps=2,
            pc_error_prediction_weight=0.05,
        ),
        _SyntheticArmSpec(
            "pc_same_router_flat_mlp_control_topk2",
            "core_periphery_control",
            support_width,
            num_columns,
            atoms_per_column,
            "contextual_mlp",
            anchor_kl_weight=0.02,
            stored_parameter_floor=sparse_stored_parameter_floor,
            control_budget_role="pc_same_router_flat_mlp_control",
            value_head_variant="flat_column_mlp",
            core_rank=0,
            periphery_rank=max(1, core_rank + periphery_rank),
            periphery_l1_weight=1e-4,
            residual_norm_clip=0.055,
        ),
        _SyntheticArmSpec(
            "pc_shuffled_residual_error_target_null_topk2",
            "pc_core_periphery_null",
            support_width,
            num_columns,
            atoms_per_column,
            "contextual_mlp",
            anchor_kl_weight=0.02,
            shuffled_teacher_null=True,
            stored_parameter_floor=sparse_stored_parameter_floor,
            control_budget_role="pc_shuffled_residual_error_target_null",
            value_head_variant="pc_core_periphery_residual_inference",
            core_rank=max(1, core_rank),
            periphery_rank=max(1, periphery_rank),
            core_lr_scale=0.10,
            core_drift_penalty_weight=0.15,
            periphery_l1_weight=1e-4,
            residual_norm_clip=0.055,
            gate_l1_weight=1e-4,
            pc_inference_steps=2,
            pc_error_prediction_weight=0.05,
        ),
        _SyntheticArmSpec(
            "pc_amortized_multisite_error_topk2",
            "pc_amortized_sparse",
            support_width,
            num_columns,
            atoms_per_column,
            "contextual_mlp",
            anchor_kl_weight=0.02,
            stored_parameter_floor=sparse_stored_parameter_floor,
            control_budget_role="pc_amortized_label_free_multisite_pregate",
            value_head_variant="pc_amortized_multisite_error",
            core_rank=max(1, core_rank),
            periphery_rank=max(1, periphery_rank),
            core_lr_scale=0.10,
            core_drift_penalty_weight=0.15,
            periphery_l1_weight=1e-4,
            residual_norm_clip=0.050,
            gate_l1_weight=1e-4,
            pc_inference_steps=2,
            pc_error_prediction_weight=0.06,
            pc_error_target_mode="decoder_adjoint",
        ),
        _SyntheticArmSpec(
            "pc_amortized_shuffled_error_null_topk2",
            "pc_amortized_null",
            support_width,
            num_columns,
            atoms_per_column,
            "contextual_mlp",
            anchor_kl_weight=0.02,
            stored_parameter_floor=sparse_stored_parameter_floor,
            control_budget_role="pc_amortized_shuffled_error_target_null",
            value_head_variant="pc_amortized_multisite_error",
            core_rank=max(1, core_rank),
            periphery_rank=max(1, periphery_rank),
            core_lr_scale=0.10,
            core_drift_penalty_weight=0.15,
            periphery_l1_weight=1e-4,
            residual_norm_clip=0.050,
            gate_l1_weight=1e-4,
            pc_inference_steps=2,
            pc_error_prediction_weight=0.06,
            pc_error_target_mode="decoder_adjoint_shuffled",
        ),
        _SyntheticArmSpec(
            "pc_amortized_sign_flipped_error_null_topk2",
            "pc_amortized_null",
            support_width,
            num_columns,
            atoms_per_column,
            "contextual_mlp",
            anchor_kl_weight=0.02,
            stored_parameter_floor=sparse_stored_parameter_floor,
            control_budget_role="pc_amortized_sign_flipped_error_target_null",
            value_head_variant="pc_amortized_multisite_error",
            core_rank=max(1, core_rank),
            periphery_rank=max(1, periphery_rank),
            core_lr_scale=0.10,
            core_drift_penalty_weight=0.15,
            periphery_l1_weight=1e-4,
            residual_norm_clip=0.050,
            gate_l1_weight=1e-4,
            pc_inference_steps=2,
            pc_error_prediction_weight=0.06,
            pc_error_target_mode="decoder_adjoint_sign_flipped",
        ),
        _SyntheticArmSpec(
            "pc_amortized_token_position_error_null_topk2",
            "pc_amortized_null",
            support_width,
            num_columns,
            atoms_per_column,
            "contextual_mlp",
            support_mode="token_position",
            anchor_kl_weight=0.02,
            stored_parameter_floor=sparse_stored_parameter_floor,
            control_budget_role="pc_amortized_token_position_error_predictor_null",
            value_head_variant="pc_amortized_multisite_error",
            core_rank=max(1, core_rank),
            periphery_rank=max(1, periphery_rank),
            core_lr_scale=0.10,
            core_drift_penalty_weight=0.15,
            periphery_l1_weight=1e-4,
            residual_norm_clip=0.050,
            gate_l1_weight=1e-4,
            pc_inference_steps=2,
            pc_error_prediction_weight=0.06,
            pc_error_target_mode="token_position",
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
    if spec.family in {
        "pc_core_periphery_sparse",
        "pc_core_periphery_null",
        "pc_amortized_sparse",
        "pc_amortized_null",
    }:
        return _PCCorePeripheryResidualInferenceAdapter(
            hidden_dim=hidden_dim,
            num_columns=spec.num_columns,
            top_k=spec.top_k,
            core_rank=max(1, spec.core_rank),
            periphery_rank=max(1, spec.periphery_rank),
            contextual_router_hidden_dim=hidden_dim * 2,
            core_lr_scale=spec.core_lr_scale,
            residual_norm_clip=spec.residual_norm_clip,
            inference_steps=max(1, spec.pc_inference_steps),
            torch=torch,
            nn=nn,
        )
    if spec.family in {"core_periphery_sparse", "core_periphery_control"}:
        return _CorePeripherySparseAdapter(
            hidden_dim=hidden_dim,
            num_columns=spec.num_columns,
            top_k=spec.top_k,
            core_rank=max(1, spec.core_rank),
            periphery_rank=max(1, spec.periphery_rank),
            variant=spec.value_head_variant,
            contextual_router_hidden_dim=hidden_dim * 2,
            core_lr_scale=spec.core_lr_scale,
            residual_norm_clip=spec.residual_norm_clip,
            torch=torch,
            nn=nn,
        )
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


class _CausalTransformerFuturePredictor:
    """Tiny batch-first causal transformer used by the local Transformer-ACSR smoke."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        seq_len: int,
        predictor_dim: int,
        num_heads: int,
        nn: Any,
        torch: Any,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(hidden_dim, predictor_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, predictor_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=predictor_dim * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.output_projection = nn.Linear(predictor_dim, hidden_dim)

    def parameters(self) -> Any:
        yield from self.input_projection.parameters()
        yield self.position_embedding
        yield from self.encoder.parameters()
        yield from self.output_projection.parameters()

    def __call__(self, features: Any) -> Any:
        import torch

        seq_len = int(features.shape[1])
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=features.device),
            diagonal=1,
        )
        projected = self.input_projection(features) + self.position_embedding[:, :seq_len, :]
        return self.output_projection(self.encoder(projected, mask=mask))


def _causal_transformer_future_perturbation_max_delta(
    *,
    predictor: Any,
    features: Any,
    perturb_from_position: int,
    perturb_scale: float,
) -> float:
    """Return max pre-perturbation-position output change after perturbing future inputs."""

    import torch

    with torch.no_grad():
        original = predictor(features).detach()
        perturbed_features = features.detach().clone()
        if perturb_from_position < int(features.shape[1]):
            perturb = torch.linspace(
                perturb_scale,
                perturb_scale * 2.0,
                steps=perturbed_features[:, perturb_from_position:, :].numel(),
                dtype=features.dtype,
                device=features.device,
            ).reshape_as(perturbed_features[:, perturb_from_position:, :])
            perturbed_features[:, perturb_from_position:, :] = (
                perturbed_features[:, perturb_from_position:, :] + perturb
            )
        perturbed = predictor(perturbed_features).detach()
        prefix_end = min(max(perturb_from_position, 0), int(features.shape[1]))
        if prefix_end == 0:
            return 0.0
        return float((original[:, :prefix_end, :] - perturbed[:, :prefix_end, :]).abs().max().item())


class _CorePeripherySparseAdapter:
    """Sparse top-k adapter with protected shared core and plastic per-column value paths."""

    def __init__(
        self,
        hidden_dim: int,
        num_columns: int,
        top_k: int,
        core_rank: int,
        periphery_rank: int,
        variant: str,
        contextual_router_hidden_dim: int,
        core_lr_scale: float,
        residual_norm_clip: float,
        *,
        torch: Any,
        nn: Any,
    ) -> None:
        if top_k < 1 or top_k > num_columns:
            raise ValueError("top_k must be between 1 and num_columns")
        self.hidden_dim = hidden_dim
        self.num_columns = num_columns
        self.top_k = top_k
        self.variant = variant
        self.core_lr_scale = core_lr_scale
        self.residual_norm_clip = float(residual_norm_clip)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        contextual_feature_dim = hidden_dim * 5 + 3
        self.contextual_column_scores = nn.Sequential(
            nn.LayerNorm(contextual_feature_dim),
            nn.Linear(contextual_feature_dim, contextual_router_hidden_dim),
            nn.GELU(),
            nn.Linear(contextual_router_hidden_dim, num_columns, bias=False),
        )
        self.core_down = nn.Linear(hidden_dim, core_rank, bias=False)
        self.core_up = nn.Linear(core_rank, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, 1)
        self.periphery_down = nn.Parameter(torch.empty(num_columns, hidden_dim, periphery_rank))
        self.periphery_up = nn.Parameter(torch.zeros(num_columns, periphery_rank, hidden_dim))
        self.score_tie_breaker = torch.arange(num_columns, 0, -1, dtype=self.periphery_down.dtype) * 1e-6
        nn.init.normal_(self.core_down.weight, std=0.02)
        nn.init.zeros_(self.core_up.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.normal_(self.periphery_down, std=0.02)

    def parameters(self) -> Any:
        yield from self.layer_norm.parameters()
        yield from self.contextual_column_scores.parameters()
        yield from self.core_down.parameters()
        yield from self.core_up.parameters()
        yield from self.gate.parameters()
        yield self.periphery_down
        yield self.periphery_up

    def optimizer_param_groups(self, learning_rate: float) -> list[dict[str, Any]]:
        core_params = list(self.core_down.parameters()) + list(self.core_up.parameters())
        other_params = (
            list(self.layer_norm.parameters())
            + list(self.contextual_column_scores.parameters())
            + list(self.gate.parameters())
            + [self.periphery_down, self.periphery_up]
        )
        return [
            {"params": other_params, "lr": learning_rate},
            {"params": core_params, "lr": learning_rate * self.core_lr_scale},
        ]

    def core_parameter_snapshot(self) -> list[Any]:
        return [parameter.detach().clone() for parameter in list(self.core_down.parameters()) + list(self.core_up.parameters())]

    def core_drift_penalty(self, reference: list[Any]) -> Any:
        penalty = None
        for parameter, start in zip(list(self.core_down.parameters()) + list(self.core_up.parameters()), reference, strict=True):
            value = (parameter - start).pow(2).mean()
            penalty = value if penalty is None else penalty + value
        return penalty if penalty is not None else self.periphery_up.sum() * 0.0

    def core_parameter_drift_l2(self, reference: list[Any]) -> float:
        values = [
            (parameter.detach() - start).pow(2).mean().sqrt().item()
            for parameter, start in zip(list(self.core_down.parameters()) + list(self.core_up.parameters()), reference, strict=True)
        ]
        return float(sum(values) / len(values)) if values else 0.0

    def periphery_l1(self) -> Any:
        return self.periphery_up.abs().mean()

    def mean_gate_value(self, hidden: Any) -> Any:
        return self._gate_values(hidden).mean()

    def gate_l1(self, hidden: Any) -> Any:
        return self._gate_values(hidden).abs().mean()

    def _gate_values(self, hidden: Any) -> Any:
        import torch

        if self.variant == "gated_low_rank_value_mixture":
            return 0.5 * torch.sigmoid(self.gate(self.layer_norm(hidden)))
        return torch.ones(
            hidden.shape[0],
            hidden.shape[1],
            1,
            dtype=hidden.dtype,
            device=hidden.device,
        )

    def _contextual_features(self, hidden: Any) -> Any:
        import torch

        current = hidden
        previous = torch.cat([current[:, :1, :], current[:, :-1, :]], dim=1)
        next_hidden = torch.cat([current[:, 1:, :], current[:, -1:, :]], dim=1)
        seq_len = int(current.shape[1])
        if seq_len <= 1:
            normalized_position = torch.zeros(
                current.shape[0],
                seq_len,
                1,
                dtype=current.dtype,
                device=current.device,
            )
        else:
            normalized_position = torch.linspace(
                0.0,
                1.0,
                seq_len,
                dtype=current.dtype,
                device=current.device,
            ).view(1, seq_len, 1).expand(current.shape[0], seq_len, 1)
        angle = normalized_position * (2.0 * torch.pi)
        return torch.cat(
            [
                current,
                previous,
                next_hidden,
                current - previous,
                next_hidden - current,
                normalized_position,
                torch.sin(angle),
                torch.cos(angle),
            ],
            dim=-1,
        )

    def __call__(
        self,
        hidden: Any,
        support_indices: Any | None = None,
        return_support: bool = False,
    ) -> Any:
        import torch
        import torch.nn.functional as F

        normalized = self.layer_norm(hidden)
        scores = self.contextual_column_scores(self._contextual_features(hidden))
        scores = scores + self.score_tie_breaker.to(device=hidden.device, dtype=hidden.dtype)
        if support_indices is None:
            top_values, top_indices = scores.topk(self.top_k, dim=-1)
        else:
            top_indices = support_indices
            top_values = scores.gather(dim=-1, index=top_indices)
        column_weights = F.softmax(top_values, dim=-1)
        core_delta = self.core_up(self.core_down(normalized))
        flat_delta = torch.einsum("bsh,chr,crd->bscd", normalized, self.periphery_down, self.periphery_up)
        selected_periphery = flat_delta.gather(
            dim=2,
            index=top_indices.unsqueeze(-1).expand(-1, -1, -1, flat_delta.shape[-1]),
        )
        periphery_delta = torch.einsum("bsk,bskh->bsh", column_weights, selected_periphery)
        if self.variant == "core_only":
            residual = core_delta
        elif self.variant in {"periphery_only", "flat_column_mlp"}:
            residual = periphery_delta
        elif self.variant == "gated_low_rank_value_mixture":
            residual = self._gate_values(hidden) * (core_delta + periphery_delta)
        else:
            residual = core_delta + periphery_delta
        if self.residual_norm_clip > 0.0:
            residual_rms = residual.pow(2).mean(dim=-1, keepdim=True).add(1e-12).sqrt()
            residual = residual * torch.clamp(self.residual_norm_clip / residual_rms, max=1.0)
        output = hidden + residual
        if return_support:
            return output, top_indices.detach()
        return output


class _PCCorePeripheryResidualInferenceAdapter(_CorePeripherySparseAdapter):
    """Two-step predictive residual inference with slow core and plastic periphery."""

    def __init__(
        self,
        hidden_dim: int,
        num_columns: int,
        top_k: int,
        core_rank: int,
        periphery_rank: int,
        contextual_router_hidden_dim: int,
        core_lr_scale: float,
        residual_norm_clip: float,
        inference_steps: int,
        *,
        torch: Any,
        nn: Any,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_columns=num_columns,
            top_k=top_k,
            core_rank=core_rank,
            periphery_rank=periphery_rank,
            variant="pc_core_periphery_residual_inference",
            contextual_router_hidden_dim=contextual_router_hidden_dim,
            core_lr_scale=core_lr_scale,
            residual_norm_clip=residual_norm_clip,
            torch=torch,
            nn=nn,
        )
        self.inference_steps = max(1, int(inference_steps))
        self.error_down = nn.Linear(hidden_dim, periphery_rank, bias=False)
        self.error_up = nn.Linear(periphery_rank, hidden_dim, bias=False)
        nn.init.normal_(self.error_down.weight, std=0.02)
        nn.init.zeros_(self.error_up.weight)

    def parameters(self) -> Any:
        yield from super().parameters()
        yield from self.error_down.parameters()
        yield from self.error_up.parameters()

    def optimizer_param_groups(self, learning_rate: float) -> list[dict[str, Any]]:
        groups = super().optimizer_param_groups(learning_rate)
        groups[0]["params"] = list(groups[0]["params"]) + list(self.error_down.parameters()) + list(self.error_up.parameters())
        return groups

    def _residual(
        self,
        hidden: Any,
        support_indices: Any | None = None,
    ) -> tuple[Any, Any]:
        import torch
        import torch.nn.functional as F

        normalized = self.layer_norm(hidden)
        scores = self.contextual_column_scores(self._contextual_features(hidden))
        scores = scores + self.score_tie_breaker.to(device=hidden.device, dtype=hidden.dtype)
        if support_indices is None:
            top_values, top_indices = scores.topk(self.top_k, dim=-1)
        else:
            top_indices = support_indices
            top_values = scores.gather(dim=-1, index=top_indices)
        column_weights = F.softmax(top_values, dim=-1)
        core_prediction = self.core_up(self.core_down(normalized))
        column_error = torch.einsum("bsh,chr,crd->bscd", normalized, self.periphery_down, self.periphery_up)
        selected_error = column_error.gather(
            dim=2,
            index=top_indices.unsqueeze(-1).expand(-1, -1, -1, column_error.shape[-1]),
        )
        peripheral_error = torch.einsum("bsk,bskh->bsh", column_weights, selected_error)
        inferred = torch.zeros_like(core_prediction)
        for _ in range(self.inference_steps):
            prediction_error = peripheral_error - inferred
            inferred = inferred + 0.5 * prediction_error + 0.25 * self.error_up(self.error_down(prediction_error))
        residual = self._gate_values(hidden) * (core_prediction + inferred)
        if self.residual_norm_clip > 0.0:
            residual_rms = residual.pow(2).mean(dim=-1, keepdim=True).add(1e-12).sqrt()
            residual = residual * torch.clamp(self.residual_norm_clip / residual_rms, max=1.0)
        return residual, top_indices

    def __call__(
        self,
        hidden: Any,
        support_indices: Any | None = None,
        return_support: bool = False,
    ) -> Any:
        residual, top_indices = self._residual(hidden, support_indices=support_indices)
        output = hidden + residual
        if return_support:
            return output, top_indices.detach()
        return output

    def residual_error_prediction_loss(
        self,
        hidden: Any,
        inputs: Any,
        targets: Any,
        decoder_weight: Any,
        *,
        decode: Any,
        target_mode: str,
        shuffle_targets: bool,
        F: Any,
    ) -> Any:
        import torch

        residual, _ = self._residual(hidden)
        if target_mode.startswith("decoder_adjoint"):
            target_error = _decoder_adjoint_hidden_ce_error(
                hidden,
                targets,
                decoder_weight,
                decode=decode,
                F=F,
                normalize=True,
            )
        elif target_mode == "token_position":
            token_vectors = decoder_weight.index_select(0, inputs.reshape(-1)).reshape(
                inputs.shape[0],
                inputs.shape[1],
                -1,
            )
            seq_len = int(inputs.shape[1])
            if seq_len <= 1:
                position_scale = torch.zeros(
                    inputs.shape[0],
                    seq_len,
                    1,
                    dtype=hidden.dtype,
                    device=hidden.device,
                )
            else:
                position_scale = torch.linspace(
                    -1.0,
                    1.0,
                    seq_len,
                    dtype=hidden.dtype,
                    device=hidden.device,
                ).view(1, seq_len, 1).expand(inputs.shape[0], seq_len, 1)
            target_error = F.layer_norm(token_vectors - hidden, (hidden.shape[-1],)) * position_scale
        else:
            target_vectors = decoder_weight.index_select(0, targets.reshape(-1)).reshape(
                targets.shape[0],
                targets.shape[1],
                -1,
            )
            target_error = F.layer_norm(target_vectors - hidden, (hidden.shape[-1],))
        if shuffle_targets or target_mode.endswith("_shuffled"):
            flat = target_error.reshape(-1, target_error.shape[-1])
            order = torch.arange(flat.shape[0] - 1, -1, -1, device=flat.device)
            target_error = flat.index_select(0, order).reshape_as(target_error)
        if target_mode.endswith("_sign_flipped"):
            target_error = -target_error
        target_error = target_error.detach()
        return F.mse_loss(residual, target_error)


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


def _decoder_adjoint_hidden_ce_error(
    hidden: Any,
    targets: Any,
    decoder_weight: Any,
    *,
    decode: Any,
    F: Any,
    normalize: bool = True,
) -> Any:
    """Return the hidden-space CE descent direction for a linear decoder."""

    logits = decode(hidden)
    probabilities = F.softmax(logits, dim=-1)
    expected_decoder = probabilities @ decoder_weight
    gold_decoder = decoder_weight.index_select(0, targets.reshape(-1)).reshape(
        targets.shape[0],
        targets.shape[1],
        -1,
    )
    target = gold_decoder - expected_decoder
    if normalize:
        target = F.layer_norm(target, (target.shape[-1],))
    return target


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
    if spec.family not in {
        "sparse",
        "sparse_null",
        "router_null",
        "core_periphery_sparse",
        "core_periphery_control",
        "pc_core_periphery_sparse",
        "pc_core_periphery_null",
        "pc_amortized_sparse",
        "pc_amortized_null",
    }:
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
    split: str = "holdout",
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
        for right in range(spec.num_columns)
        if right != left
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
                    "split": split,
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


def _support_head_sequence_heldout_diagnostic_rows(
    *,
    arm: str,
    adapter: Any,
    train_hidden: Any,
    train_inputs: Any,
    train_targets: Any,
    train_oracle_rows: list[dict[str, Any]],
    holdout_hidden: Any,
    holdout_inputs: Any,
    holdout_targets: Any,
    holdout_oracle_rows: list[dict[str, Any]],
    spec: _SyntheticArmSpec,
    decode: Any,
    F: Any,
    torch: Any,
) -> list[dict[str, Any]]:
    if not train_oracle_rows or not holdout_oracle_rows:
        return []

    train_flat_hidden = train_hidden.reshape(-1, train_hidden.shape[-1]).detach()
    holdout_flat_hidden = holdout_hidden.reshape(-1, holdout_hidden.shape[-1]).detach()
    train_labels = [
        _parse_support_key(str(row.get("best_pair_support", "")), expected_width=spec.top_k)
        for row in sorted(
            train_oracle_rows,
            key=lambda row: (int(row["episode_index"]), int(row["position_index"])),
        )
    ]
    shuffled_labels = list(reversed(train_labels))
    holdout_rows = sorted(
        holdout_oracle_rows,
        key=lambda row: (int(row["episode_index"]), int(row["position_index"])),
    )
    holdout_pair_oracle_labels = [
        _parse_support_key(str(row.get("best_pair_support", "")), expected_width=spec.top_k)
        for row in holdout_rows
    ]
    token_position_lookup = _token_position_support_lookup(
        train_inputs=train_inputs,
        train_oracle_rows=train_oracle_rows,
        expected_width=spec.top_k,
    )
    global_support = _modal_support(train_labels, expected_width=spec.top_k)

    distances = torch.cdist(holdout_flat_hidden, train_flat_hidden)
    nearest_indices = torch.argmin(distances, dim=1).detach().cpu().tolist()
    policies = {
        "support_regret_trained_contextual_router_topk2": [
            train_labels[index] for index in nearest_indices
        ],
        "shuffled_oracle_target_null_topk2": [
            shuffled_labels[index] for index in nearest_indices
        ],
        "token_position_only_support_head_topk2": [
            token_position_lookup.get(
                (
                    int(holdout_inputs.reshape(-1)[flat_index].detach().item()),
                    flat_index % int(holdout_inputs.shape[1]),
                ),
                global_support,
            )
            for flat_index in range(int(holdout_inputs.numel()))
        ],
        "global_modal_support_null_topk2": [
            global_support for _ in range(int(holdout_inputs.numel()))
        ],
    }
    learned_support = _synthetic_support(adapter, holdout_hidden, holdout_inputs, spec, torch)
    if learned_support is None:
        return []
    policy_support = {
        name: _support_tensor(
            supports,
            batch=int(holdout_targets.shape[0]),
            seq_len=int(holdout_targets.shape[1]),
            device=holdout_targets.device,
            torch=torch,
        )
        for name, supports in policies.items()
    }
    learned_logits = decode(adapter(holdout_hidden, support_indices=learned_support)).detach()
    learned_losses = F.cross_entropy(
        learned_logits.reshape(-1, learned_logits.shape[-1]),
        holdout_targets.reshape(-1),
        reduction="none",
    ).reshape(holdout_targets.shape)
    learned_ce = float(learned_losses.mean().detach().item())
    oracle_pair_ce = _mean_optional(holdout_rows, "best_pair_ce_loss")
    learned_pair_regret = _mean_optional(holdout_rows, "pair_oracle_regret")

    rows: list[dict[str, Any]] = []
    for policy_name, support_tensor in policy_support.items():
        logits = decode(adapter(holdout_hidden, support_indices=support_tensor)).detach()
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            holdout_targets.reshape(-1),
            reduction="none",
        ).reshape(holdout_targets.shape)
        predicted = [
            tuple(int(value.detach().item()) for value in support_tensor.reshape(-1, spec.top_k)[index])
            for index in range(int(support_tensor.numel() // spec.top_k))
        ]
        matches_oracle = [
            set(predicted_support) == set(oracle_support)
            for predicted_support, oracle_support in zip(predicted, holdout_pair_oracle_labels, strict=True)
        ]
        predicted_ce = float(losses.mean().detach().item())
        gain = learned_ce - predicted_ce
        recovery = _safe_ratio(gain, learned_pair_regret)
        residual_l2 = float(
            (adapter(holdout_hidden, support_indices=support_tensor).detach() - holdout_hidden)
            .pow(2)
            .mean()
            .sqrt()
            .item()
        )
        rows.append(
            {
                "arm": arm,
                "diagnostic": policy_name,
                "split": "sequence_heldout",
                "train_token_count": int(train_inputs.numel()),
                "holdout_token_count": int(holdout_inputs.numel()),
                "target_source": (
                    "train_split_oracle_pair_supports"
                    if policy_name != "token_position_only_support_head_topk2"
                    and policy_name != "global_modal_support_null_topk2"
                    else "train_split_oracle_pair_support_marginals"
                ),
                "uses_hidden_features": policy_name
                in {
                    "support_regret_trained_contextual_router_topk2",
                    "shuffled_oracle_target_null_topk2",
                },
                "uses_token_position_features": policy_name == "token_position_only_support_head_topk2",
                "uses_shuffled_targets": policy_name == "shuffled_oracle_target_null_topk2",
                "oracle_targets_enter_auxiliary_training": policy_name
                in {
                    "support_regret_trained_contextual_router_topk2",
                    "shuffled_oracle_target_null_topk2",
                    "token_position_only_support_head_topk2",
                    "global_modal_support_null_topk2",
                },
                "deployable_training_evidence": False,
                "learned_router_ce": learned_ce,
                "oracle_pair_ce_ceiling": oracle_pair_ce,
                "learned_pair_oracle_regret": learned_pair_regret,
                "predicted_support_ce": predicted_ce,
                "predicted_support_ce_gain_vs_learned": gain,
                "oracle_pair_regret_recovery_fraction": recovery,
                "support_accuracy_vs_oracle_pair": (
                    sum(1 for value in matches_oracle if value) / float(len(matches_oracle))
                    if matches_oracle
                    else None
                ),
                "unique_support_sets": _unique_support_count(predicted),
                "support_load_entropy": _support_load_entropy(predicted),
                "support_change_fraction": _support_change_fraction(predicted, int(holdout_targets.shape[1])),
                "residual_l2": residual_l2,
                "advance_if_gain_gt_0p02_or_recovery_ge_0p5": bool(
                    gain > 0.02 or (recovery is not None and recovery >= 0.5)
                ),
                "beats_shuffled_target_null": None,
                "beats_token_position_null": None,
                "mechanism_labels_used_for_scoring_only": True,
                "interpretation": (
                    "Diagnostic only: support labels are train-split oracle pair targets and are not deployable "
                    "training evidence; heldout CE uses fixed sparse values with hard top-k2 support swaps."
                ),
            }
        )

    by_name = {str(row["diagnostic"]): row for row in rows}
    contextual = by_name.get("support_regret_trained_contextual_router_topk2")
    shuffled = by_name.get("shuffled_oracle_target_null_topk2")
    token_position = by_name.get("token_position_only_support_head_topk2")
    if contextual is not None:
        contextual_ce = _metric_float(contextual.get("predicted_support_ce"))
        shuffled_ce = _metric_float(shuffled.get("predicted_support_ce")) if shuffled else None
        token_position_ce = _metric_float(token_position.get("predicted_support_ce")) if token_position else None
        contextual["beats_shuffled_target_null"] = _ce_beats_or_ties(contextual_ce, shuffled_ce)
        contextual["beats_token_position_null"] = _ce_beats_or_ties(contextual_ce, token_position_ce)
        contextual["advance_if_gain_gt_0p02_or_recovery_ge_0p5"] = bool(
            contextual["advance_if_gain_gt_0p02_or_recovery_ge_0p5"]
            and contextual["beats_shuffled_target_null"] is True
            and contextual["beats_token_position_null"] is True
        )
    return rows


def _parse_support_key(value: str, *, expected_width: int) -> tuple[int, ...]:
    parsed = tuple(int(part) for part in value.split(",") if part != "")
    if not parsed:
        parsed = (0,)
    while len(parsed) < expected_width:
        parsed = parsed + (parsed[-1],)
    return parsed[:expected_width]


def _support_tensor(
    supports: list[tuple[int, ...]],
    *,
    batch: int,
    seq_len: int,
    device: Any,
    torch: Any,
) -> Any:
    return torch.tensor(supports, dtype=torch.long, device=device).view(batch, seq_len, -1)


def _token_position_support_lookup(
    *,
    train_inputs: Any,
    train_oracle_rows: list[dict[str, Any]],
    expected_width: int,
) -> dict[tuple[int, int], tuple[int, ...]]:
    counts: dict[tuple[int, int], dict[tuple[int, ...], int]] = {}
    ordered_rows = sorted(
        train_oracle_rows,
        key=lambda row: (int(row["episode_index"]), int(row["position_index"])),
    )
    flat_inputs = train_inputs.reshape(-1)
    seq_len = int(train_inputs.shape[1])
    for flat_index, row in enumerate(ordered_rows):
        key = (int(flat_inputs[flat_index].detach().item()), int(row["position_index"]) % seq_len)
        support = _parse_support_key(str(row.get("best_pair_support", "")), expected_width=expected_width)
        counts.setdefault(key, {})
        counts[key][support] = counts[key].get(support, 0) + 1
    return {
        key: max(support_counts.items(), key=lambda item: (item[1], item[0]))[0]
        for key, support_counts in counts.items()
    }


def _modal_support(
    supports: list[tuple[int, ...]],
    *,
    expected_width: int,
) -> tuple[int, ...]:
    if not supports:
        return tuple(0 for _ in range(expected_width))
    counts: dict[tuple[int, ...], int] = {}
    for support in supports:
        counts[support] = counts.get(support, 0) + 1
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _unique_support_count(supports: list[tuple[int, ...]]) -> int:
    return len({tuple(support) for support in supports})


def _support_load_entropy(supports: list[tuple[int, ...]]) -> float | None:
    if not supports:
        return None
    import math

    counts: dict[int, int] = {}
    total = 0
    for support in supports:
        for column in support:
            counts[column] = counts.get(column, 0) + 1
            total += 1
    entropy = 0.0
    for count in counts.values():
        probability = count / float(total)
        entropy -= probability * math.log(probability)
    return entropy


def _support_change_fraction(supports: list[tuple[int, ...]], seq_len: int) -> float | None:
    if not supports or seq_len <= 1:
        return None
    changes = 0
    comparisons = 0
    for offset in range(0, len(supports), seq_len):
        episode_supports = supports[offset : offset + seq_len]
        for left, right in zip(episode_supports, episode_supports[1:]):
            changes += int(set(left) != set(right))
            comparisons += 1
    return changes / float(comparisons) if comparisons else None


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


def _router_only_branch_selection_rows(
    ceiling_rows: list[dict[str, Any]],
    support_head_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not ceiling_rows:
        return []
    contextual_rows = [
        row
        for row in support_head_rows
        if row.get("diagnostic") == "support_regret_trained_contextual_router_topk2"
    ]
    contextual_by_arm = {str(row.get("arm", "")): row for row in contextual_rows}
    rows: list[dict[str, Any]] = []
    for ceiling in ceiling_rows:
        arm = str(ceiling.get("arm", ""))
        support_head = contextual_by_arm.get(arm, {})
        ceiling_closes_stored = ceiling.get("router_only_can_close_stored_matched_gap") is True
        ceiling_beats_token_null = ceiling.get("oracle_ce_beats_token_position_null") is True
        support_head_advances = support_head.get("advance_if_gain_gt_0p02_or_recovery_ge_0p5") is True
        support_head_beats_token_null = support_head.get("beats_token_position_null") is True
        support_head_gain = _metric_float(support_head.get("predicted_support_ce_gain_vs_learned"))
        support_head_recovery = _metric_float(support_head.get("oracle_pair_regret_recovery_fraction"))
        stored_gap = _metric_float(ceiling.get("learned_ce_gap_to_stored_matched_control"))
        oracle_gain = _metric_float(ceiling.get("oracle_support_ce_gain"))
        close_router_only_branch = not (
            ceiling_closes_stored and ceiling_beats_token_null and support_head_advances
        )
        if not ceiling_closes_stored:
            decision = "close_router_only_branch_stored_gap_not_closable"
        elif not ceiling_beats_token_null:
            decision = "close_router_only_branch_oracle_fails_token_position_null"
        elif support_head and not support_head_advances:
            decision = "close_router_only_branch_support_head_fails_null_or_gain_gate"
        elif not support_head:
            decision = "defer_router_only_branch_missing_support_head_diagnostic"
        else:
            decision = "continue_router_only_branch_needs_repeat"
        rows.append(
            {
                "arm": arm,
                "branch": "router_only_support_head",
                "decision": decision,
                "close_or_deprioritize_router_only_path": close_router_only_branch,
                "recommend_next_path": (
                    "value_capacity_or_core_periphery_residual_design"
                    if close_router_only_branch
                    else "repeat_sequence_heldout_support_head_before_any_gpu"
                ),
                "requires_gpu_now": False,
                "promotion_allowed": False,
                "oracle_support_ce_gain": oracle_gain,
                "stored_matched_gap": stored_gap,
                "router_only_can_close_stored_gap": ceiling_closes_stored,
                "oracle_ce_beats_token_position_null": ceiling_beats_token_null,
                "support_head_ce_gain_vs_learned": support_head_gain,
                "support_head_regret_recovery_fraction": support_head_recovery,
                "support_head_beats_shuffled_target_null": support_head.get("beats_shuffled_target_null"),
                "support_head_beats_token_position_null": support_head.get("beats_token_position_null"),
                "support_head_advances": support_head_advances,
                "oracle_targets_enter_auxiliary_training": bool(
                    support_head.get("oracle_targets_enter_auxiliary_training")
                ),
                "deployable_training_evidence": False,
                "mechanism_labels_used_for_scoring_only": True,
                "interpretation": (
                    "Fail-closed branch selector: router/support-head evidence is diagnostic only. "
                    "Close or deprioritize the router-only path unless its oracle ceiling can close the "
                    "stored-control gap and its sequence-heldout support head beats strong nulls."
                ),
            }
        )
    return rows


def _value_capacity_core_periphery_diagnostic_rows(
    *,
    arm_metrics: list[dict[str, Any]],
    residual_budget_rows: list[dict[str, Any]],
    commutator_rows: list[dict[str, Any]],
    forgetting_rows: list[dict[str, Any]],
    router_only_branch_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not arm_metrics:
        return []

    by_arm = {str(row.get("arm", "")): row for row in arm_metrics}
    budget_by_arm = {str(row.get("arm", "")): row for row in residual_budget_rows}
    sparse_candidates = [
        by_arm[arm]
        for arm in (
            "promoted_contextual_topk2",
            "intervention_trained_sparse_topk2",
            "dense_teacher_distilled_sparse_topk2",
        )
        if arm in by_arm
    ]
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
    best_sparse = _best_ce_row(sparse_candidates)
    best_active = _best_ce_row(active_controls)
    best_stored = _best_ce_row(stored_controls)
    if best_sparse is None:
        return []

    router_only_closed = bool(
        router_only_branch_rows
        and all(row.get("close_or_deprioritize_router_only_path") is True for row in router_only_branch_rows)
    )
    sparse_arm = str(best_sparse.get("arm", ""))
    sparse_ce = _metric_float(best_sparse.get("holdout_ce"))
    sparse_commutator = _mean_metric(commutator_rows, [sparse_arm], "finite_update_commutator_l2")
    sparse_churn = _mean_abs_metric(forgetting_rows, [sparse_arm], "functional_churn")
    sparse_budget = budget_by_arm.get(sparse_arm, {})

    rows: list[dict[str, Any]] = []
    for branch, comparator in (
        ("active_value_capacity_control", best_active),
        ("stored_value_capacity_upper_bound", best_stored),
    ):
        if comparator is None:
            continue
        comparator_arm = str(comparator.get("arm", ""))
        comparator_ce = _metric_float(comparator.get("holdout_ce"))
        ce_gap = _positive_gap(sparse_ce, comparator_ce)
        comparator_commutator = _mean_metric(
            commutator_rows,
            [comparator_arm],
            "finite_update_commutator_l2",
        )
        comparator_churn = _mean_abs_metric(
            forgetting_rows,
            [comparator_arm],
            "functional_churn",
        )
        comparator_budget = budget_by_arm.get(comparator_arm, {})
        rows.append(
            {
                "branch": branch,
                "reference_sparse_arm": sparse_arm,
                "comparator_arm": comparator_arm,
                "reference_sparse_ce": sparse_ce,
                "comparator_ce": comparator_ce,
                "sparse_ce_gap_to_comparator": ce_gap,
                "reference_sparse_residual_l2": _metric_float(best_sparse.get("residual_l2")),
                "comparator_residual_l2": _metric_float(comparator.get("residual_l2")),
                "comparator_residual_l2_ratio_vs_sparse": _safe_ratio(
                    _metric_float(comparator.get("residual_l2")),
                    _metric_float(best_sparse.get("residual_l2")),
                ),
                "reference_sparse_active_parameters_proxy": _metric_float(
                    best_sparse.get("active_parameters_proxy")
                ),
                "comparator_active_parameters_proxy": _metric_float(
                    comparator.get("active_parameters_proxy")
                ),
                "comparator_active_ratio_vs_sparse": _safe_ratio(
                    _metric_float(comparator.get("active_parameters_proxy")),
                    _metric_float(best_sparse.get("active_parameters_proxy")),
                ),
                "reference_sparse_stored_parameters": _metric_float(best_sparse.get("stored_parameters")),
                "comparator_stored_parameters": _metric_float(comparator.get("stored_parameters")),
                "comparator_stored_ratio_vs_sparse": _safe_ratio(
                    _metric_float(comparator.get("stored_parameters")),
                    _metric_float(best_sparse.get("stored_parameters")),
                ),
                "reference_sparse_flop_proxy_per_token": _metric_float(
                    sparse_budget.get("flop_proxy_per_token")
                ),
                "comparator_flop_proxy_per_token": _metric_float(
                    comparator_budget.get("flop_proxy_per_token")
                ),
                "comparator_flop_ratio_vs_sparse": _safe_ratio(
                    _metric_float(comparator_budget.get("flop_proxy_per_token")),
                    _metric_float(sparse_budget.get("flop_proxy_per_token")),
                ),
                "reference_sparse_mean_commutator_l2": sparse_commutator,
                "comparator_mean_commutator_l2": comparator_commutator,
                "comparator_commutator_ratio_vs_sparse": _safe_ratio(
                    comparator_commutator,
                    sparse_commutator,
                ),
                "reference_sparse_mean_abs_functional_churn": sparse_churn,
                "comparator_mean_abs_functional_churn": comparator_churn,
                "comparator_churn_ratio_vs_sparse": _safe_ratio(comparator_churn, sparse_churn),
                "router_only_path_closed": router_only_closed,
                "requires_gpu_now": False,
                "promotion_allowed": False,
                "candidate_status": (
                    "stored_capacity_gap_demands_new_local_column_design"
                    if branch == "stored_value_capacity_upper_bound" and ce_gap is not None and ce_gap > 0.02
                    else "active_control_not_decisive_for_new_design"
                    if branch == "active_value_capacity_control"
                    else "stored_capacity_gap_not_material_on_this_seed"
                ),
                "recommend_next_path": (
                    "core_periphery_sparse_value_capacity_probe"
                    if router_only_closed and branch == "stored_value_capacity_upper_bound" and ce_gap is not None and ce_gap > 0.02
                    else "repeat_or_repair_local_value_capacity_diagnostic"
                ),
                "mechanism_labels_used_for_scoring_only": True,
                "interpretation": (
                    "Diagnostic only: compares the best trained sparse arm with dense/MLP value-capacity "
                    "controls using already-measured CE, budget, commutator, and functional-churn rows."
                ),
            }
        )

    if best_stored is not None:
        stored_gap = _positive_gap(sparse_ce, _metric_float(best_stored.get("holdout_ce")))
        rows.append(
            {
                "branch": "core_periphery_sparse_design_probe",
                "reference_sparse_arm": sparse_arm,
                "comparator_arm": str(best_stored.get("arm", "")),
                "reference_sparse_ce": sparse_ce,
                "comparator_ce": _metric_float(best_stored.get("holdout_ce")),
                "sparse_ce_gap_to_comparator": stored_gap,
                "reference_sparse_residual_l2": _metric_float(best_sparse.get("residual_l2")),
                "comparator_residual_l2": _metric_float(best_stored.get("residual_l2")),
                "comparator_residual_l2_ratio_vs_sparse": _safe_ratio(
                    _metric_float(best_stored.get("residual_l2")),
                    _metric_float(best_sparse.get("residual_l2")),
                ),
                "reference_sparse_active_parameters_proxy": _metric_float(
                    best_sparse.get("active_parameters_proxy")
                ),
                "comparator_active_parameters_proxy": _metric_float(
                    best_stored.get("active_parameters_proxy")
                ),
                "comparator_active_ratio_vs_sparse": _safe_ratio(
                    _metric_float(best_stored.get("active_parameters_proxy")),
                    _metric_float(best_sparse.get("active_parameters_proxy")),
                ),
                "reference_sparse_stored_parameters": _metric_float(best_sparse.get("stored_parameters")),
                "comparator_stored_parameters": _metric_float(best_stored.get("stored_parameters")),
                "comparator_stored_ratio_vs_sparse": _safe_ratio(
                    _metric_float(best_stored.get("stored_parameters")),
                    _metric_float(best_sparse.get("stored_parameters")),
                ),
                "reference_sparse_flop_proxy_per_token": _metric_float(
                    sparse_budget.get("flop_proxy_per_token")
                ),
                "comparator_flop_proxy_per_token": _metric_float(
                    budget_by_arm.get(str(best_stored.get("arm", "")), {}).get("flop_proxy_per_token")
                ),
                "comparator_flop_ratio_vs_sparse": _safe_ratio(
                    _metric_float(
                        budget_by_arm.get(str(best_stored.get("arm", "")), {}).get("flop_proxy_per_token")
                    ),
                    _metric_float(sparse_budget.get("flop_proxy_per_token")),
                ),
                "reference_sparse_mean_commutator_l2": sparse_commutator,
                "comparator_mean_commutator_l2": _mean_metric(
                    commutator_rows,
                    [str(best_stored.get("arm", ""))],
                    "finite_update_commutator_l2",
                ),
                "comparator_commutator_ratio_vs_sparse": _safe_ratio(
                    _mean_metric(
                        commutator_rows,
                        [str(best_stored.get("arm", ""))],
                        "finite_update_commutator_l2",
                    ),
                    sparse_commutator,
                ),
                "reference_sparse_mean_abs_functional_churn": sparse_churn,
                "comparator_mean_abs_functional_churn": _mean_abs_metric(
                    forgetting_rows,
                    [str(best_stored.get("arm", ""))],
                    "functional_churn",
                ),
                "comparator_churn_ratio_vs_sparse": _safe_ratio(
                    _mean_abs_metric(
                        forgetting_rows,
                        [str(best_stored.get("arm", ""))],
                        "functional_churn",
                    ),
                    sparse_churn,
                ),
                "router_only_path_closed": router_only_closed,
                "requires_gpu_now": False,
                "promotion_allowed": False,
                "candidate_status": (
                    "selected_next_local_probe"
                    if router_only_closed and stored_gap is not None and stored_gap > 0.02
                    else "blocked_until_router_or_stored_gap_evidence_is_clear"
                ),
                "recommend_next_path": (
                    "core_periphery_sparse_value_capacity_probe"
                    if router_only_closed and stored_gap is not None and stored_gap > 0.02
                    else "repeat_or_repair_local_value_capacity_diagnostic"
                ),
                "mechanism_labels_used_for_scoring_only": True,
                "interpretation": (
                    "Design-selection row: the next bounded local probe should add within-column "
                    "core/periphery value structure only if router-only is closed and the stored-control "
                    "gap remains too large for support selection to explain."
                ),
            }
        )
    return rows


def _mean_abs_metric(rows: list[dict[str, Any]], arms: list[str], metric: str) -> float | None:
    values = [
        abs(value)
        for value in (
            _metric_float(row.get(metric))
            for row in rows
            if row.get("arm") in arms and row.get("metric_values_available") is True
        )
        if value is not None
    ]
    if not values:
        return None
    return sum(values) / float(len(values))


def _flop_proxy(row: dict[str, Any]) -> float | None:
    active_parameters = _metric_float(row.get("active_parameters_proxy"))
    if active_parameters is None:
        return None
    family = str(row.get("family", ""))
    if family == "base":
        return 0.0
    if family == "mlp_control":
        return float(active_parameters * 2.0)
    if family in {
        "dense_control",
        "sparse",
        "sparse_null",
        "router_null",
        "core_periphery_sparse",
        "core_periphery_control",
        "pc_core_periphery_sparse",
        "pc_core_periphery_null",
    }:
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
    if spec.family in {"core_periphery_sparse", "core_periphery_control", "pc_core_periphery_sparse", "pc_core_periphery_null"}:
        core_active = (
            hidden_dim * max(1, spec.core_rank)
            if spec.value_head_variant not in {"periphery_only", "flat_column_mlp"}
            else 0
        )
        periphery_active = (
            spec.top_k * hidden_dim * max(1, spec.periphery_rank)
            if spec.value_head_variant != "core_only"
            else 0
        )
        return int(core_active + periphery_active)
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
        _control("core_periphery_sparse_topk2", "core_periphery_sparse", support_width, "contextual_mlp_core_periphery", True, "protected shared core plus plastic per-column periphery sparse value-capacity probe"),
        _control("flat_column_value_mlp_topk2", "core_periphery_control", support_width, "contextual_mlp_flat_column_value", True, "same-router flat per-column value MLP control"),
        _control("core_only_sparse_topk2", "core_periphery_control", support_width, "contextual_mlp_core_only", True, "core-only value-capacity ablation control"),
        _control("periphery_only_sparse_topk2", "core_periphery_control", support_width, "contextual_mlp_periphery_only", True, "periphery-only value-capacity ablation control"),
        _control("core_periphery_stability_slow_core_topk2", "core_periphery_control", support_width, "contextual_mlp_core_periphery_slow_core_anchor", True, "same-router core/periphery update-stability bracket with slower core and anchor KL"),
        _control("flat_column_value_mlp_anchor_topk2", "core_periphery_control", support_width, "contextual_mlp_flat_column_value_anchor", True, "same-router flat value-capacity update-stability bracket with anchor KL"),
        _control("budget_normalized_gated_low_rank_value_mixture_topk2", "core_periphery_control", support_width, "contextual_mlp_gated_low_rank_value_mixture", True, "selected local pregate with residual budget normalization and contextual low-rank value gating"),
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
    if training_smoke.get("transformer_acsr_cpu_smoke_pilot"):
        summary = _transformer_acsr_cpu_smoke_pilot_summary(
            training_smoke["transformer_acsr_cpu_smoke_pilot"]
        )
        if summary.get("selected_next_experiment"):
            return str(summary["selected_next_experiment"]).replace("_", " ")
    if training_smoke.get("transformer_acsr_design"):
        summary = _transformer_acsr_design_summary(
            training_smoke["transformer_acsr_design"]
        )
        if summary.get("selected_next_experiment"):
            return str(summary["selected_next_experiment"]).replace("_", " ")
    if training_smoke.get("pc_amortized_error_pregate_closeout"):
        summary = _pc_amortized_error_pregate_closeout_summary(
            training_smoke["pc_amortized_error_pregate_closeout"]
        )
        if summary.get("selected_next_experiment"):
            soft_design = _soft_mixture_low_churn_dense_modular_design_summary(
                training_smoke.get("soft_mixture_low_churn_dense_modular_design", [])
            )
            if soft_design.get("selected_next_experiment"):
                return str(soft_design["selected_next_experiment"]).replace("_", " ")
            budget_closeout = _budget_normalized_gated_value_mixture_closeout_summary(
                training_smoke.get("budget_normalized_gated_value_mixture_closeout", [])
            )
            if budget_closeout.get("branch_closed") and budget_closeout.get("selected_next_experiment"):
                return str(budget_closeout["selected_next_experiment"]).replace("_", " ")
            return str(summary["selected_next_experiment"]).replace("_", " ")
    if training_smoke.get("pc_amortized_error_pregate"):
        summary = _pc_amortized_error_pregate_summary(
            training_smoke["pc_amortized_error_pregate"]
        )
        if summary.get("selected_next_experiment"):
            return str(summary["selected_next_experiment"]).replace("_", " ")
    if training_smoke.get("pc_amortized_error_pregate_design"):
        summary = _pc_amortized_error_pregate_design_summary(
            training_smoke["pc_amortized_error_pregate_design"]
        )
        if summary.get("selected_next_experiment"):
            return str(summary["selected_next_experiment"]).replace("_", " ")
    if training_smoke.get("pc_decoder_adjoint_closeout"):
        summary = _pc_decoder_adjoint_closeout_summary(
            training_smoke["pc_decoder_adjoint_closeout"]
        )
        if summary.get("selected_next_experiment"):
            return str(summary["selected_next_experiment"]).replace("_", " ")
    if training_smoke.get("pc_decoder_adjoint_minimal_retrain_probe"):
        summary = _pc_decoder_adjoint_minimal_retrain_probe_summary(
            training_smoke["pc_decoder_adjoint_minimal_retrain_probe"]
        )
        if summary.get("selected_next_experiment"):
            return str(summary["selected_next_experiment"]).replace("_", " ")
    if training_smoke.get("pc_decoder_adjoint_target_alignment_probe"):
        summary = _pc_decoder_adjoint_target_alignment_probe_summary(
            training_smoke["pc_decoder_adjoint_target_alignment_probe"]
        )
        if summary.get("selected_next_experiment"):
            return str(summary["selected_next_experiment"]).replace("_", " ")
    if training_smoke.get("pc_error_target_inference_path_audit"):
        summary = _pc_error_target_inference_path_audit_summary(
            training_smoke["pc_error_target_inference_path_audit"]
        )
        if summary.get("selected_next_experiment"):
            return str(summary["selected_next_experiment"]).replace("_", " ")
    if training_smoke.get("pc_residual_inference_mechanism_inspection"):
        summary = _pc_residual_inference_mechanism_inspection_summary(
            training_smoke["pc_residual_inference_mechanism_inspection"]
        )
        if summary.get("selected_next_experiment"):
            return str(summary["selected_next_experiment"]).replace("_", " ")
    if training_smoke.get("pc_core_periphery_residual_inference_pregate"):
        summary = _pc_core_periphery_residual_inference_pregate_summary(
            training_smoke["pc_core_periphery_residual_inference_pregate"]
        )
        if summary.get("trainable_pc_packet_implemented"):
            if summary.get("pregate_passes"):
                return (
                    "repeat the local pc_core_periphery_residual_inference_pregate on one adjacent seed "
                    "before any RunPod validation or promotion"
                )
            return (
                "inspect or redesign the local pc_core_periphery_residual_inference_pregate because the "
                "first trainable packet did not clear signal, flat-control, norm, commutator, and churn gates"
            )
        if summary.get("selected_next_experiment"):
            return (
                "implement the local pc_core_periphery_residual_inference_pregate trainable arm with the "
                "fixed contextual top-k2 router, two-step residual-error inference, anchor-KL, norm, "
                "commutator, churn, flat same-router, norm-clipped dense/stored, token-position, random-support, "
                "and shuffled-target null controls before any GPU validation"
            )
    if training_smoke.get("budget_normalized_gated_value_mixture_pregate"):
        soft_design = _soft_mixture_low_churn_dense_modular_design_summary(
            training_smoke.get("soft_mixture_low_churn_dense_modular_design", [])
        )
        if soft_design.get("selected_next_experiment"):
            return str(soft_design["selected_next_experiment"]).replace("_", " ")
        summary = _budget_normalized_gated_value_mixture_pregate_summary(
            training_smoke["budget_normalized_gated_value_mixture_pregate"]
        )
        closeout = _budget_normalized_gated_value_mixture_closeout_summary(
            training_smoke.get("budget_normalized_gated_value_mixture_closeout", [])
        )
        if closeout.get("branch_closed") and closeout.get("selected_next_experiment"):
            return str(closeout["selected_next_experiment"]).replace("_", " ")
        if summary.get("pregate_passes"):
            return (
                "repeat the budget-normalized gated low-rank value-mixture pregate on one adjacent local seed "
                "before considering RunPod validation"
            )
        return (
            "close or redesign the budget-normalized gated low-rank sparse value-mixture branch locally; "
            "do not run GPU until a sparse value arm clears flat-control and interference gates"
        )
    if training_smoke.get("sparse_value_redesign_selector"):
        summary = _sparse_value_redesign_selector_summary(
            training_smoke["sparse_value_redesign_selector"]
        )
        if summary.get("selected_next_experiment"):
            return (
                "implement the local budget-normalized gated low-rank value-mixture pregate selected by "
                "the sparse value redesign selector; keep GPU validation blocked until flat-control, norm, "
                "commutator, and churn gates pass"
            )
    if training_smoke.get("core_periphery_branch_closeout"):
        summary = _core_periphery_branch_closeout_summary(
            training_smoke["core_periphery_branch_closeout"]
        )
        if summary.get("closeout_status") == "closed_redesign_required":
            return (
                "redesign the local sparse value mechanism away from the current clipped core/periphery "
                "split; require a flat-control and interference-budget pregate before any GPU validation"
            )
        if summary.get("closeout_status") == "repeat_before_closeout":
            return (
                "repeat the local core/periphery branch on one adjacent seed before deciding closeout or redesign"
            )
    if training_smoke.get("core_periphery_update_stability_bracket"):
        summary = _core_periphery_update_stability_bracket_summary(
            training_smoke["core_periphery_update_stability_bracket"]
        )
        if summary.get("stability_candidate_count", 0) > 0:
            return (
                "repeat the local seed-17 update-stability candidate with one adjacent regularization setting "
                "before any GPU validation"
            )
        return (
            "close or redesign the current clipped core/periphery value-capacity branch locally because "
            "the update-stability bracket did not clear commutator and churn budgets"
        )
    if training_smoke.get("value_capacity_core_periphery_diagnostic"):
        summary = _value_capacity_core_periphery_diagnostic_summary(
            training_smoke["value_capacity_core_periphery_diagnostic"]
        )
        if summary.get("selected_next_path") == "core_periphery_sparse_value_capacity_probe":
            return (
                "implement the bounded local core/periphery sparse value-capacity probe selected by the "
                "value/capacity diagnostic, preserving stored-matched dense/MLP, token/position-null, "
                "residual-norm, functional-churn, and commutator controls before any GPU validation"
            )
        return (
            "repair or repeat the local value/capacity diagnostic before designing a new residual mechanism"
        )
    if training_smoke.get("router_only_branch_selection"):
        return (
            "deprioritize the router-only/support-head path on this seed and design the next local non-GPU mechanism probe around value/capacity or core/periphery residual structure"
        )
    if training_smoke.get("support_head_sequence_heldout_diagnostic"):
        return (
            "inspect the sequence-heldout support-head diagnostic against shuffled-target and token-position nulls; keep GPU and promotion blocked unless the diagnostic beats nulls and the stored upper-bound gap is separately addressed"
        )
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


def _support_head_sequence_heldout_diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "contextual_gain_vs_learned": None,
            "contextual_recovery_fraction": None,
            "advance_support_head_branch": False,
            "interpretation": "sequence-heldout support-head diagnostic was not run",
        }
    contextual_rows = [
        row
        for row in rows
        if row.get("diagnostic") == "support_regret_trained_contextual_router_topk2"
    ]
    best_contextual = min(
        contextual_rows,
        key=lambda row: _metric_float(row.get("predicted_support_ce")) or float("inf"),
    ) if contextual_rows else {}
    return {
        "row_count": len(rows),
        "arm_count": len({str(row.get("arm", "")) for row in rows}),
        "diagnostic_count": len({str(row.get("diagnostic", "")) for row in rows}),
        "best_contextual_arm": best_contextual.get("arm", ""),
        "contextual_ce": _metric_float(best_contextual.get("predicted_support_ce")),
        "learned_router_ce": _metric_float(best_contextual.get("learned_router_ce")),
        "oracle_pair_ce_ceiling": _metric_float(best_contextual.get("oracle_pair_ce_ceiling")),
        "contextual_gain_vs_learned": _metric_float(
            best_contextual.get("predicted_support_ce_gain_vs_learned")
        ),
        "contextual_recovery_fraction": _metric_float(
            best_contextual.get("oracle_pair_regret_recovery_fraction")
        ),
        "contextual_support_accuracy_vs_oracle_pair": _metric_float(
            best_contextual.get("support_accuracy_vs_oracle_pair")
        ),
        "contextual_beats_shuffled_target_null": best_contextual.get("beats_shuffled_target_null"),
        "contextual_beats_token_position_null": best_contextual.get("beats_token_position_null"),
        "advance_support_head_branch": any(
            row.get("advance_if_gain_gt_0p02_or_recovery_ge_0p5") is True
            for row in contextual_rows
        ),
        "deployable_training_evidence": False,
        "interpretation": (
            "Diagnostic only: train-split oracle pair supports supervise a sequence-heldout support selector. "
            "Advance only when contextual support swaps beat shuffled-target and token/position nulls while "
            "recovering enough learned-router regret; this is not deployment evidence."
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


def _router_only_branch_selection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "closed_or_deprioritized_arm_count": 0,
            "recommended_next_path": "",
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "interpretation": "router-only branch selection was not run",
        }
    closed_rows = [
        row for row in rows if row.get("close_or_deprioritize_router_only_path") is True
    ]
    recommended_paths = {
        str(row.get("recommend_next_path", "")) for row in rows if row.get("recommend_next_path")
    }
    if "value_capacity_or_core_periphery_residual_design" in recommended_paths:
        recommended_next_path = "value_capacity_or_core_periphery_residual_design"
    else:
        recommended_next_path = sorted(recommended_paths)[0] if recommended_paths else ""
    return {
        "row_count": len(rows),
        "arm_count": len({str(row.get("arm", "")) for row in rows}),
        "closed_or_deprioritized_arm_count": len(closed_rows),
        "all_router_only_arms_closed_or_deprioritized": len(closed_rows) == len(rows),
        "recommended_next_path": recommended_next_path,
        "decisions": sorted({str(row.get("decision", "")) for row in rows}),
        "requires_gpu_now": any(row.get("requires_gpu_now") is True for row in rows),
        "promotion_allowed": any(row.get("promotion_allowed") is True for row in rows),
        "deployable_training_evidence": False,
        "interpretation": (
            "Fail-closed local branch selector. A router-only/support-head path remains blocked unless "
            "the oracle support ceiling can close the stored-control gap and the sequence-heldout "
            "support head beats strong nulls."
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


def _value_capacity_core_periphery_diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "selected_next_path": "",
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "interpretation": "value/capacity and core/periphery diagnostic was not run",
        }
    selected_rows = [
        row for row in rows if row.get("candidate_status") == "selected_next_local_probe"
    ]
    if selected_rows:
        selected_next_path = str(selected_rows[0].get("recommend_next_path", ""))
    else:
        recommended = sorted(
            {
                str(row.get("recommend_next_path", ""))
                for row in rows
                if row.get("recommend_next_path")
            }
        )
        selected_next_path = recommended[0] if recommended else ""
    stored_rows = [row for row in rows if row.get("branch") == "stored_value_capacity_upper_bound"]
    stored_gap = (
        _metric_float(stored_rows[0].get("sparse_ce_gap_to_comparator"))
        if stored_rows
        else None
    )
    return {
        "row_count": len(rows),
        "branch_count": len({str(row.get("branch", "")) for row in rows}),
        "selected_next_path": selected_next_path,
        "stored_capacity_gap": stored_gap,
        "router_only_path_closed": all(row.get("router_only_path_closed") is True for row in rows),
        "requires_gpu_now": any(row.get("requires_gpu_now") is True for row in rows),
        "promotion_allowed": any(row.get("promotion_allowed") is True for row in rows),
        "candidate_statuses": sorted({str(row.get("candidate_status", "")) for row in rows}),
        "interpretation": (
            "Fail-closed local synthesis over measured CE, budget, commutator, and churn rows. "
            "A core/periphery sparse value-capacity probe is a local next step only; it is not GPU "
            "or promotion evidence."
        ),
    }


def _core_periphery_sparse_value_capacity_probe_rows(
    *,
    arm_metrics: list[dict[str, Any]],
    residual_budget_rows: list[dict[str, Any]],
    commutator_rows: list[dict[str, Any]],
    forgetting_rows: list[dict[str, Any]],
    router_only_branch_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not arm_metrics:
        return []
    by_arm = {str(row.get("arm", "")): row for row in arm_metrics}
    budget_by_arm = {str(row.get("arm", "")): row for row in residual_budget_rows}
    reference = by_arm.get("promoted_contextual_topk2")
    core_periphery = by_arm.get("core_periphery_sparse_topk2")
    stored_controls = [
        row
        for row in arm_metrics
        if row.get("control_budget_role") == "stored_parameter_matched_dense_mlp_upper_bound"
    ]
    stored = _best_ce_row(stored_controls)
    if reference is None or core_periphery is None:
        return []

    router_only_closed = bool(
        router_only_branch_rows
        and all(row.get("close_or_deprioritize_router_only_path") is True for row in router_only_branch_rows)
    )
    reference_ce = _metric_float(reference.get("holdout_ce"))
    probe_ce = _metric_float(core_periphery.get("holdout_ce"))
    stored_ce = _metric_float(stored.get("holdout_ce")) if stored else None
    stored_gap = _positive_gap(reference_ce, stored_ce)
    ce_gain = _delta_value(reference_ce, probe_ce)
    stored_gap_closed_fraction = _safe_ratio(ce_gain, stored_gap)
    reference_norm = _metric_float(reference.get("residual_l2"))
    reference_commutator = _mean_metric(
        commutator_rows,
        ["promoted_contextual_topk2"],
        "finite_update_commutator_l2",
    )
    reference_churn = _mean_abs_metric(
        forgetting_rows,
        ["promoted_contextual_topk2"],
        "functional_churn",
    )

    rows: list[dict[str, Any]] = []
    for arm in (
        "core_periphery_sparse_topk2",
        "flat_column_value_mlp_topk2",
        "core_only_sparse_topk2",
        "periphery_only_sparse_topk2",
    ):
        row = by_arm.get(arm)
        if row is None:
            continue
        arm_ce = _metric_float(row.get("holdout_ce"))
        arm_norm = _metric_float(row.get("residual_l2"))
        arm_commutator = _mean_metric(commutator_rows, [arm], "finite_update_commutator_l2")
        arm_churn = _mean_abs_metric(forgetting_rows, [arm], "functional_churn")
        arm_budget = budget_by_arm.get(arm, {})
        is_primary = arm == "core_periphery_sparse_topk2"
        beats_reference = ce_gain is not None and ce_gain >= 0.05 if is_primary else (
            reference_ce is not None and arm_ce is not None and arm_ce <= reference_ce - 0.01
        )
        closes_stored_gap = (
            stored_gap_closed_fraction is not None and stored_gap_closed_fraction >= 0.10
            if is_primary
            else False
        )
        norm_ok = arm_norm is not None and reference_norm is not None and arm_norm <= reference_norm * 1.05
        commutator_ok = (
            arm_commutator is not None
            and reference_commutator is not None
            and arm_commutator <= reference_commutator * 1.10
        )
        churn_ok = (
            arm_churn is not None
            and reference_churn is not None
            and arm_churn <= reference_churn * 1.10
        )
        advance = bool(
            is_primary
            and router_only_closed
            and (beats_reference or closes_stored_gap)
            and norm_ok
            and commutator_ok
            and churn_ok
        )
        rows.append(
            {
                "arm": arm,
                "probe_role": "primary_core_periphery_probe" if is_primary else str(row.get("control_budget_role", "")),
                "value_head_variant": row.get("value_head_variant", ""),
                "reference_sparse_arm": "promoted_contextual_topk2",
                "reference_sparse_ce": reference_ce,
                "holdout_ce": arm_ce,
                "ce_gain_vs_reference_sparse": _delta_value(reference_ce, arm_ce),
                "stored_control_arm": stored.get("arm", "") if stored else "",
                "stored_control_ce": stored_ce,
                "reference_sparse_gap_to_stored_control": stored_gap,
                "stored_gap_closed_fraction": _safe_ratio(_delta_value(reference_ce, arm_ce), stored_gap),
                "residual_l2": arm_norm,
                "reference_sparse_residual_l2": reference_norm,
                "residual_l2_ratio_vs_reference_sparse": _safe_ratio(arm_norm, reference_norm),
                "active_parameters_proxy": _metric_float(row.get("active_parameters_proxy")),
                "stored_parameters": _metric_float(row.get("stored_parameters")),
                "flop_proxy_per_token": _metric_float(arm_budget.get("flop_proxy_per_token")),
                "core_rank": _metric_float(row.get("core_rank")),
                "periphery_rank": _metric_float(row.get("periphery_rank")),
                "core_lr_scale": _metric_float(row.get("core_lr_scale")),
                "core_drift_penalty_weight": _metric_float(row.get("core_drift_penalty_weight")),
                "core_parameter_drift_l2": _metric_float(row.get("core_parameter_drift_l2")),
                "periphery_l1_weight": _metric_float(row.get("periphery_l1_weight")),
                "periphery_l1": _metric_float(row.get("periphery_l1")),
                "residual_norm_clip": _metric_float(row.get("residual_norm_clip")),
                "residual_norm_clipped": row.get("residual_norm_clipped") is True,
                "mean_commutator_l2": arm_commutator,
                "reference_sparse_mean_commutator_l2": reference_commutator,
                "commutator_ratio_vs_reference_sparse": _safe_ratio(arm_commutator, reference_commutator),
                "mean_abs_functional_churn": arm_churn,
                "reference_sparse_mean_abs_functional_churn": reference_churn,
                "functional_churn_ratio_vs_reference_sparse": _safe_ratio(arm_churn, reference_churn),
                "router_only_path_closed": router_only_closed,
                "beats_reference_by_0p05_ce": bool(beats_reference) if is_primary else False,
                "closes_at_least_10pct_stored_gap": bool(closes_stored_gap) if is_primary else False,
                "norm_budget_ok": bool(norm_ok),
                "commutator_budget_ok": bool(commutator_ok),
                "functional_churn_budget_ok": bool(churn_ok),
                "advance_to_gpu_validation": advance,
                "requires_gpu_now": False,
                "promotion_allowed": False,
                "mechanism_labels_used_for_scoring_only": True,
                "interpretation": (
                    "Local trainable value-capacity probe only: labels are used for CE scoring, the router "
                    "is the same contextual top-k2 mechanism, and advancement requires CE/gap improvement "
                    "without worsening norm, commutator, or churn budgets."
                ),
            }
        )
    return rows


def _core_periphery_sparse_value_capacity_probe_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "advance_to_gpu_validation": False,
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "interpretation": "core/periphery sparse value-capacity probe was not run",
        }
    primary = next((row for row in rows if row.get("probe_role") == "primary_core_periphery_probe"), {})
    return {
        "row_count": len(rows),
        "primary_arm": primary.get("arm", ""),
        "primary_holdout_ce": _metric_float(primary.get("holdout_ce")),
        "ce_gain_vs_reference_sparse": _metric_float(primary.get("ce_gain_vs_reference_sparse")),
        "stored_gap_closed_fraction": _metric_float(primary.get("stored_gap_closed_fraction")),
        "core_parameter_drift_l2": _metric_float(primary.get("core_parameter_drift_l2")),
        "periphery_l1": _metric_float(primary.get("periphery_l1")),
        "norm_budget_ok": primary.get("norm_budget_ok") is True,
        "commutator_budget_ok": primary.get("commutator_budget_ok") is True,
        "functional_churn_budget_ok": primary.get("functional_churn_budget_ok") is True,
        "router_only_path_closed": primary.get("router_only_path_closed") is True,
        "advance_to_gpu_validation": any(row.get("advance_to_gpu_validation") is True for row in rows),
        "requires_gpu_now": any(row.get("requires_gpu_now") is True for row in rows),
        "promotion_allowed": any(row.get("promotion_allowed") is True for row in rows),
        "control_arms": [
            str(row.get("arm", ""))
            for row in rows
            if row.get("probe_role") != "primary_core_periphery_probe"
        ],
        "interpretation": (
            "Fail-closed local probe. A core/periphery arm is only worth GPU validation if it improves "
            "the promoted sparse reference or closes at least 10% of the stored-control gap while keeping "
            "norm, commutator, and churn budgets intact."
        ),
    }


def _core_periphery_update_stability_bracket_rows(
    *,
    arm_metrics: list[dict[str, Any]],
    commutator_rows: list[dict[str, Any]],
    forgetting_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not arm_metrics:
        return []
    by_arm = {str(row.get("arm", "")): row for row in arm_metrics}
    reference = by_arm.get("promoted_contextual_topk2")
    primary = by_arm.get("core_periphery_sparse_topk2")
    flat = by_arm.get("flat_column_value_mlp_topk2")
    if reference is None:
        return []

    reference_ce = _metric_float(reference.get("holdout_ce"))
    reference_norm = _metric_float(reference.get("residual_l2"))
    reference_commutator = _mean_metric(
        commutator_rows,
        ["promoted_contextual_topk2"],
        "finite_update_commutator_l2",
    )
    reference_churn = _mean_abs_metric(
        forgetting_rows,
        ["promoted_contextual_topk2"],
        "functional_churn",
    )
    primary_ce = _metric_float(primary.get("holdout_ce")) if primary else None
    primary_commutator = (
        _mean_metric(commutator_rows, ["core_periphery_sparse_topk2"], "finite_update_commutator_l2")
        if primary
        else None
    )
    primary_churn = (
        _mean_abs_metric(forgetting_rows, ["core_periphery_sparse_topk2"], "functional_churn")
        if primary
        else None
    )
    flat_ce = _metric_float(flat.get("holdout_ce")) if flat else None
    flat_commutator = (
        _mean_metric(commutator_rows, ["flat_column_value_mlp_topk2"], "finite_update_commutator_l2")
        if flat
        else None
    )
    flat_churn = (
        _mean_abs_metric(forgetting_rows, ["flat_column_value_mlp_topk2"], "functional_churn")
        if flat
        else None
    )

    rows: list[dict[str, Any]] = []
    for arm in (
        "core_periphery_stability_slow_core_topk2",
        "flat_column_value_mlp_anchor_topk2",
    ):
        row = by_arm.get(arm)
        if row is None:
            continue
        arm_ce = _metric_float(row.get("holdout_ce"))
        arm_norm = _metric_float(row.get("residual_l2"))
        arm_commutator = _mean_metric(commutator_rows, [arm], "finite_update_commutator_l2")
        arm_churn = _mean_abs_metric(forgetting_rows, [arm], "functional_churn")
        compares_to = (
            "core_periphery_sparse_topk2"
            if arm == "core_periphery_stability_slow_core_topk2"
            else "flat_column_value_mlp_topk2"
        )
        comparator_ce = primary_ce if compares_to == "core_periphery_sparse_topk2" else flat_ce
        comparator_commutator = (
            primary_commutator
            if compares_to == "core_periphery_sparse_topk2"
            else flat_commutator
        )
        comparator_churn = primary_churn if compares_to == "core_periphery_sparse_topk2" else flat_churn
        norm_ok = arm_norm is not None and reference_norm is not None and arm_norm <= reference_norm * 1.05
        commutator_ok = (
            arm_commutator is not None
            and reference_commutator is not None
            and arm_commutator <= reference_commutator * 1.10
        )
        churn_ok = (
            arm_churn is not None
            and reference_churn is not None
            and arm_churn <= reference_churn * 1.10
        )
        improves_vs_unregularized = (
            comparator_ce is not None
            and arm_ce is not None
            and arm_ce <= comparator_ce + 0.01
            and (
                comparator_commutator is None
                or arm_commutator is None
                or arm_commutator <= comparator_commutator
            )
            and (
                comparator_churn is None
                or arm_churn is None
                or arm_churn <= comparator_churn
            )
        )
        rows.append(
            {
                "arm": arm,
                "bracket_role": str(row.get("control_budget_role", "")),
                "value_head_variant": row.get("value_head_variant", ""),
                "reference_sparse_arm": "promoted_contextual_topk2",
                "unregularized_comparator_arm": compares_to,
                "reference_sparse_ce": reference_ce,
                "unregularized_comparator_ce": comparator_ce,
                "holdout_ce": arm_ce,
                "ce_delta_vs_reference_sparse": _delta_value(arm_ce, reference_ce),
                "ce_delta_vs_unregularized_comparator": _delta_value(arm_ce, comparator_ce),
                "residual_l2": arm_norm,
                "reference_sparse_residual_l2": reference_norm,
                "residual_l2_ratio_vs_reference_sparse": _safe_ratio(arm_norm, reference_norm),
                "mean_commutator_l2": arm_commutator,
                "reference_sparse_mean_commutator_l2": reference_commutator,
                "unregularized_comparator_mean_commutator_l2": comparator_commutator,
                "commutator_ratio_vs_reference_sparse": _safe_ratio(arm_commutator, reference_commutator),
                "commutator_ratio_vs_unregularized_comparator": _safe_ratio(arm_commutator, comparator_commutator),
                "mean_abs_functional_churn": arm_churn,
                "reference_sparse_mean_abs_functional_churn": reference_churn,
                "unregularized_comparator_mean_abs_functional_churn": comparator_churn,
                "functional_churn_ratio_vs_reference_sparse": _safe_ratio(arm_churn, reference_churn),
                "functional_churn_ratio_vs_unregularized_comparator": _safe_ratio(arm_churn, comparator_churn),
                "anchor_kl_weight": _metric_float(row.get("anchor_kl_weight")),
                "core_lr_scale": _metric_float(row.get("core_lr_scale")),
                "core_drift_penalty_weight": _metric_float(row.get("core_drift_penalty_weight")),
                "core_parameter_drift_l2": _metric_float(row.get("core_parameter_drift_l2")),
                "periphery_l1_weight": _metric_float(row.get("periphery_l1_weight")),
                "periphery_l1": _metric_float(row.get("periphery_l1")),
                "residual_norm_clip": _metric_float(row.get("residual_norm_clip")),
                "residual_norm_clipped": row.get("residual_norm_clipped") is True,
                "norm_budget_ok": bool(norm_ok),
                "commutator_budget_ok": bool(commutator_ok),
                "functional_churn_budget_ok": bool(churn_ok),
                "stability_candidate": bool(norm_ok and commutator_ok and churn_ok and improves_vs_unregularized),
                "requires_gpu_now": False,
                "promotion_allowed": False,
                "mechanism_labels_used_for_scoring_only": True,
                "interpretation": (
                    "Local update-stability bracket only: anchor/core-LR variants must retain CE while "
                    "reducing commutator and churn before they can replace the value-capacity probe."
                ),
            }
        )
    return rows


def _core_periphery_update_stability_bracket_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "stability_candidate_count": 0,
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "interpretation": "core/periphery update-stability bracket was not run",
        }
    candidates = [row for row in rows if row.get("stability_candidate") is True]
    best = _best_ce_row(rows)
    return {
        "row_count": len(rows),
        "stability_candidate_count": len(candidates),
        "best_ce_arm": best.get("arm", "") if best else "",
        "best_ce": _metric_float(best.get("holdout_ce")) if best else None,
        "candidate_arms": [str(row.get("arm", "")) for row in candidates],
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "interpretation": (
            "Fail-closed local bracket over clipped value-capacity arms. A variant only becomes a "
            "follow-up candidate if it keeps CE near its unregularized comparator and improves the "
            "commutator/churn budget while staying norm-matched."
        ),
    }


def _core_periphery_branch_closeout_rows(
    *,
    probe_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not probe_rows:
        return []
    primary = next((row for row in probe_rows if row.get("probe_role") == "primary_core_periphery_probe"), {})
    flat = next((row for row in probe_rows if row.get("arm") == "flat_column_value_mlp_topk2"), {})
    stability_candidates = [row for row in stability_rows if row.get("stability_candidate") is True]

    primary_ce = _metric_float(primary.get("holdout_ce"))
    flat_ce = _metric_float(flat.get("holdout_ce"))
    ce_gain = _metric_float(primary.get("ce_gain_vs_reference_sparse"))
    stored_gap_closed = _metric_float(primary.get("stored_gap_closed_fraction"))
    flat_ce_advantage = _delta_value(primary_ce, flat_ce)
    primary_budget_passes = bool(
        primary.get("norm_budget_ok") is True
        and primary.get("commutator_budget_ok") is True
        and primary.get("functional_churn_budget_ok") is True
    )
    primary_signal_passes = bool(
        (ce_gain is not None and ce_gain >= 0.05)
        or (stored_gap_closed is not None and stored_gap_closed >= 0.10)
    )
    flat_control_concern = bool(flat_ce_advantage is not None and flat_ce_advantage > 0.01)
    closeout_status = (
        "continue_local_branch"
        if primary_signal_passes and primary_budget_passes and not flat_control_concern
        else "repeat_before_closeout"
        if stability_candidates
        else "closed_redesign_required"
    )
    return [
        {
            "branch": "clipped_core_periphery_sparse_value_capacity",
            "closeout_status": closeout_status,
            "primary_arm": primary.get("arm", ""),
            "reference_sparse_arm": primary.get("reference_sparse_arm", ""),
            "primary_holdout_ce": primary_ce,
            "reference_sparse_ce": _metric_float(primary.get("reference_sparse_ce")),
            "primary_ce_gain_vs_reference_sparse": ce_gain,
            "primary_stored_gap_closed_fraction": stored_gap_closed,
            "flat_control_arm": flat.get("arm", ""),
            "flat_control_holdout_ce": flat_ce,
            "primary_ce_minus_flat_control_ce": flat_ce_advantage,
            "flat_control_stronger_by_gt_0p01": flat_control_concern,
            "norm_budget_ok": primary.get("norm_budget_ok") is True,
            "commutator_budget_ok": primary.get("commutator_budget_ok") is True,
            "functional_churn_budget_ok": primary.get("functional_churn_budget_ok") is True,
            "primary_budget_passes": primary_budget_passes,
            "primary_signal_passes": primary_signal_passes,
            "update_stability_candidate_count": len(stability_candidates),
            "update_stability_candidate_arms": ";".join(
                str(row.get("arm", "")) for row in stability_candidates
            ),
            "advance_to_gpu_validation": False,
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "recommend_next_path": (
                "redesign_sparse_value_mechanism_with_flat_control_and_interference_pregate"
                if closeout_status == "closed_redesign_required"
                else "repeat_or_extend_local_core_periphery_branch"
            ),
            "mechanism_labels_used_for_scoring_only": True,
            "interpretation": (
                "Fail-closed branch closeout. The current clipped protected-core/plastic-periphery "
                "implementation should not advance unless its primary arm clears CE or stored-gap "
                "thresholds, norm, commutator, and churn budgets, and is not explained by the flat "
                "same-router value-capacity control."
            ),
        }
    ]


def _core_periphery_branch_closeout_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "closeout_status": "not_run",
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "interpretation": "core/periphery branch closeout was not run",
        }
    row = rows[0]
    return {
        "row_count": len(rows),
        "closeout_status": row.get("closeout_status", ""),
        "primary_ce_gain_vs_reference_sparse": _metric_float(
            row.get("primary_ce_gain_vs_reference_sparse")
        ),
        "primary_stored_gap_closed_fraction": _metric_float(
            row.get("primary_stored_gap_closed_fraction")
        ),
        "primary_ce_minus_flat_control_ce": _metric_float(
            row.get("primary_ce_minus_flat_control_ce")
        ),
        "primary_budget_passes": row.get("primary_budget_passes") is True,
        "primary_signal_passes": row.get("primary_signal_passes") is True,
        "flat_control_stronger_by_gt_0p01": row.get("flat_control_stronger_by_gt_0p01") is True,
        "update_stability_candidate_count": int(row.get("update_stability_candidate_count") or 0),
        "recommended_next_path": row.get("recommend_next_path", ""),
        "requires_gpu_now": any(item.get("requires_gpu_now") is True for item in rows),
        "promotion_allowed": any(item.get("promotion_allowed") is True for item in rows),
        "interpretation": row.get("interpretation", ""),
    }


def _sparse_value_redesign_selector_rows(
    *,
    closeout_rows: list[dict[str, Any]],
    arm_metrics: list[dict[str, Any]],
    residual_budget_rows: list[dict[str, Any]],
    commutator_rows: list[dict[str, Any]],
    forgetting_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not closeout_rows:
        return []
    closeout = closeout_rows[0]
    if closeout.get("closeout_status") != "closed_redesign_required":
        return []

    by_arm = {str(row.get("arm", "")): row for row in arm_metrics}
    budget_by_arm = {str(row.get("arm", "")): row for row in residual_budget_rows}
    primary_arm = str(closeout.get("primary_arm", ""))
    flat_arm = str(closeout.get("flat_control_arm", ""))
    reference_arm = str(closeout.get("reference_sparse_arm", "promoted_contextual_topk2"))

    primary = by_arm.get(primary_arm, {})
    flat = by_arm.get(flat_arm, {})
    reference = by_arm.get(reference_arm, {})
    stored = _best_ce_row(
        [row for row in arm_metrics if row.get("control_budget_role") == "stored_parameter_matched_dense_mlp_upper_bound"]
    )
    active = _best_ce_row(
        [row for row in arm_metrics if row.get("control_budget_role") == "active_proxy_matched_dense_mlp_control"]
    )

    primary_ce = _metric_float(primary.get("holdout_ce"))
    flat_ce = _metric_float(flat.get("holdout_ce"))
    reference_ce = _metric_float(reference.get("holdout_ce"))
    stored_ce = _metric_float(stored.get("holdout_ce")) if stored else None
    active_ce = _metric_float(active.get("holdout_ce")) if active else None
    primary_commutator = _mean_metric(commutator_rows, [primary_arm], "finite_update_commutator_l2")
    primary_churn = _mean_abs_metric(forgetting_rows, [primary_arm], "functional_churn")
    flat_commutator = _mean_metric(commutator_rows, [flat_arm], "finite_update_commutator_l2")
    flat_churn = _mean_abs_metric(forgetting_rows, [flat_arm], "functional_churn")

    flat_beats_primary = bool(
        primary_ce is not None and flat_ce is not None and primary_ce - flat_ce > 0.01
    )
    budgets_failed = not bool(closeout.get("primary_budget_passes") is True)
    stored_gap = _delta_value(reference_ce, stored_ce)
    active_gap = _delta_value(reference_ce, active_ce)
    residual_ratio = _metric_float(
        budget_by_arm.get(primary_arm, {}).get("residual_l2_ratio_vs_best_sparse")
    )

    common = {
        "source_closeout_branch": closeout.get("branch", ""),
        "source_closeout_status": closeout.get("closeout_status", ""),
        "reference_sparse_arm": reference_arm,
        "reference_sparse_ce": reference_ce,
        "primary_closed_arm": primary_arm,
        "primary_closed_ce": primary_ce,
        "flat_control_arm": flat_arm,
        "flat_control_ce": flat_ce,
        "stored_upper_bound_arm": stored.get("arm", "") if stored else "",
        "stored_upper_bound_ce": stored_ce,
        "active_matched_control_arm": active.get("arm", "") if active else "",
        "active_matched_control_ce": active_ce,
        "reference_to_stored_upper_bound_ce_gap": stored_gap,
        "reference_to_active_matched_ce_gap": active_gap,
        "primary_residual_l2_ratio_vs_best_sparse": residual_ratio,
        "primary_commutator_l2": primary_commutator,
        "primary_functional_churn": primary_churn,
        "flat_control_commutator_l2": flat_commutator,
        "flat_control_functional_churn": flat_churn,
        "flat_control_stronger_by_gt_0p01": flat_beats_primary,
        "closed_branch_budget_failure": budgets_failed,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "mechanism_labels_used_for_scoring_only": True,
    }

    candidates = [
        {
            "candidate_rank": 1,
            "candidate_path": "budget_normalized_gated_low_rank_value_mixture",
            "selected": True,
            "next_experiment": "implement_local_budget_normalized_gated_value_mixture_pregate",
            "pregate_required": (
                "beat promoted sparse by >=0.05 CE or close >=10% of stored-control gap, "
                "not lose to flat same-router control by >0.01 CE, and pass norm/commutator/churn budgets"
            ),
            "reason": (
                "The clipped core/periphery split is closed, but the flat control shows value-capacity headroom; "
                "the next local design should normalize residual scale and gate low-rank value capacity before GPU."
            ),
        },
        {
            "candidate_rank": 2,
            "candidate_path": "soft_mixture_low_churn_dense_modular_residual",
            "selected": False,
            "next_experiment": "hold_as_pivot_if_sparse_value_pregate_fails",
            "pregate_required": (
                "only promote if sparse redesign again fails flat-control/interference gates and soft mixture keeps "
                "lower churn or commutator than stored MLP controls"
            ),
            "reason": (
                "Stored MLP controls remain much stronger on CE, but they carry larger interference costs; keep this "
                "as the fallback architecture rather than the immediate sparse-column step."
            ),
        },
        {
            "candidate_rank": 3,
            "candidate_path": "router_only_or_support_head_retry",
            "selected": False,
            "next_experiment": "defer_until_value_mechanism_changes",
            "pregate_required": "requires new value mechanism evidence because router-only ceilings do not close the stored-control gap",
            "reason": (
                "Existing oracle-support and sequence-heldout support-head diagnostics already close the router-only "
                "branch for this packet."
            ),
        },
    ]
    return [{**common, **candidate} for candidate in candidates]


def _sparse_value_redesign_selector_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "selected_candidate_path": "",
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "interpretation": "sparse value redesign selector was not run",
        }
    selected = next((row for row in rows if row.get("selected") is True), rows[0])
    return {
        "row_count": len(rows),
        "selected_candidate_path": selected.get("candidate_path", ""),
        "selected_next_experiment": selected.get("next_experiment", ""),
        "flat_control_stronger_by_gt_0p01": selected.get("flat_control_stronger_by_gt_0p01") is True,
        "closed_branch_budget_failure": selected.get("closed_branch_budget_failure") is True,
        "reference_to_stored_upper_bound_ce_gap": _metric_float(
            selected.get("reference_to_stored_upper_bound_ce_gap")
        ),
        "primary_residual_l2_ratio_vs_best_sparse": _metric_float(
            selected.get("primary_residual_l2_ratio_vs_best_sparse")
        ),
        "requires_gpu_now": any(row.get("requires_gpu_now") is True for row in rows),
        "promotion_allowed": any(row.get("promotion_allowed") is True for row in rows),
        "interpretation": (
            "Fail-closed local redesign selector after clipped core/periphery closeout. It selects one "
            "bounded sparse-value pregate and explicitly blocks GPU/promotion until flat-control, norm, "
            "commutator, and churn gates pass."
        ),
    }


def _budget_normalized_gated_value_mixture_pregate_rows(
    *,
    redesign_rows: list[dict[str, Any]],
    arm_metrics: list[dict[str, Any]],
    residual_budget_rows: list[dict[str, Any]],
    commutator_rows: list[dict[str, Any]],
    forgetting_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = next(
        (
            row
            for row in redesign_rows
            if row.get("selected") is True
            and row.get("candidate_path") == "budget_normalized_gated_low_rank_value_mixture"
        ),
        None,
    )
    if selected is None:
        return []

    by_arm = {str(row.get("arm", "")): row for row in arm_metrics}
    budget_by_arm = {str(row.get("arm", "")): row for row in residual_budget_rows}
    reference = by_arm.get("promoted_contextual_topk2")
    pregate = by_arm.get("budget_normalized_gated_low_rank_value_mixture_topk2")
    flat = by_arm.get("flat_column_value_mlp_topk2")
    stored = _best_ce_row(
        [row for row in arm_metrics if row.get("control_budget_role") == "stored_parameter_matched_dense_mlp_upper_bound"]
    )
    if reference is None or pregate is None or flat is None:
        return []

    reference_ce = _metric_float(reference.get("holdout_ce"))
    flat_ce = _metric_float(flat.get("holdout_ce"))
    stored_ce = _metric_float(stored.get("holdout_ce")) if stored else None
    stored_gap = _positive_gap(reference_ce, stored_ce)
    reference_norm = _metric_float(reference.get("residual_l2"))
    flat_norm = _metric_float(flat.get("residual_l2"))
    reference_commutator = _mean_metric(
        commutator_rows,
        ["promoted_contextual_topk2"],
        "finite_update_commutator_l2",
    )
    reference_churn = _mean_abs_metric(
        forgetting_rows,
        ["promoted_contextual_topk2"],
        "functional_churn",
    )
    flat_commutator = _mean_metric(commutator_rows, ["flat_column_value_mlp_topk2"], "finite_update_commutator_l2")
    flat_churn = _mean_abs_metric(forgetting_rows, ["flat_column_value_mlp_topk2"], "functional_churn")

    rows: list[dict[str, Any]] = []
    for arm, role in (
        ("budget_normalized_gated_low_rank_value_mixture_topk2", "primary_budget_normalized_gated_low_rank_value_mixture"),
        ("flat_column_value_mlp_topk2", "flat_same_router_value_capacity_control"),
        ("promoted_contextual_topk2", "promoted_sparse_reference"),
    ):
        row = by_arm.get(arm)
        if row is None:
            continue
        arm_ce = _metric_float(row.get("holdout_ce"))
        arm_norm = _metric_float(row.get("residual_l2"))
        arm_commutator = _mean_metric(commutator_rows, [arm], "finite_update_commutator_l2")
        arm_churn = _mean_abs_metric(forgetting_rows, [arm], "functional_churn")
        budget = budget_by_arm.get(arm, {})
        is_primary = arm == "budget_normalized_gated_low_rank_value_mixture_topk2"
        ce_gain = _delta_value(reference_ce, arm_ce)
        stored_gap_closed = _safe_ratio(ce_gain, stored_gap)
        flat_margin = _delta_value(arm_ce, flat_ce)
        norm_ok = arm_norm is not None and reference_norm is not None and arm_norm <= reference_norm * 1.05
        commutator_ok = (
            arm_commutator is not None
            and reference_commutator is not None
            and arm_commutator <= reference_commutator * 1.10
        )
        churn_ok = (
            arm_churn is not None
            and reference_churn is not None
            and arm_churn <= reference_churn * 1.10
        )
        signal_ok = bool(
            (ce_gain is not None and ce_gain >= 0.05)
            or (stored_gap_closed is not None and stored_gap_closed >= 0.10)
        )
        flat_control_ok = bool(flat_margin is None or flat_margin <= 0.01)
        pregate_passes = bool(
            is_primary
            and signal_ok
            and flat_control_ok
            and norm_ok
            and commutator_ok
            and churn_ok
        )
        rows.append(
            {
                "arm": arm,
                "pregate_role": role,
                "selected_candidate_path": selected.get("candidate_path", ""),
                "reference_sparse_arm": "promoted_contextual_topk2",
                "reference_sparse_ce": reference_ce,
                "holdout_ce": arm_ce,
                "ce_gain_vs_reference_sparse": ce_gain,
                "stored_control_arm": stored.get("arm", "") if stored else "",
                "stored_control_ce": stored_ce,
                "reference_sparse_gap_to_stored_control": stored_gap,
                "stored_gap_closed_fraction": stored_gap_closed,
                "flat_control_arm": "flat_column_value_mlp_topk2",
                "flat_control_ce": flat_ce,
                "ce_minus_flat_control_ce": flat_margin,
                "flat_control_ok": flat_control_ok,
                "residual_l2": arm_norm,
                "reference_sparse_residual_l2": reference_norm,
                "flat_control_residual_l2": flat_norm,
                "residual_l2_ratio_vs_reference_sparse": _safe_ratio(arm_norm, reference_norm),
                "residual_norm_clip": _metric_float(row.get("residual_norm_clip")),
                "residual_norm_clipped": row.get("residual_norm_clipped") is True,
                "mean_gate_value": _metric_float(row.get("mean_gate_value")),
                "gate_l1_weight": _metric_float(row.get("gate_l1_weight")),
                "active_parameters_proxy": _metric_float(row.get("active_parameters_proxy")),
                "stored_parameters": _metric_float(row.get("stored_parameters")),
                "flop_proxy_per_token": _metric_float(budget.get("flop_proxy_per_token")),
                "mean_commutator_l2": arm_commutator,
                "reference_sparse_mean_commutator_l2": reference_commutator,
                "flat_control_mean_commutator_l2": flat_commutator,
                "commutator_ratio_vs_reference_sparse": _safe_ratio(arm_commutator, reference_commutator),
                "mean_abs_functional_churn": arm_churn,
                "reference_sparse_mean_abs_functional_churn": reference_churn,
                "flat_control_mean_abs_functional_churn": flat_churn,
                "functional_churn_ratio_vs_reference_sparse": _safe_ratio(arm_churn, reference_churn),
                "signal_gate_ok": signal_ok if is_primary else False,
                "norm_budget_ok": norm_ok,
                "commutator_budget_ok": commutator_ok,
                "functional_churn_budget_ok": churn_ok,
                "pregate_passes": pregate_passes,
                "advance_to_gpu_validation": False,
                "requires_gpu_now": False,
                "promotion_allowed": False,
                "mechanism_labels_used_for_scoring_only": True,
                "interpretation": (
                    "Local pregate only. The selected gated low-rank sparse value mixture must clear CE or "
                    "stored-gap signal thresholds, not lose to the flat same-router control, and pass residual "
                    "norm, commutator, and functional-churn budgets before any backend validation."
                ),
            }
        )
    return rows


def _budget_normalized_gated_value_mixture_pregate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "pregate_passes": False,
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "interpretation": "budget-normalized gated value-mixture pregate was not run",
        }
    primary = next(
        (
            row
            for row in rows
            if row.get("pregate_role") == "primary_budget_normalized_gated_low_rank_value_mixture"
        ),
        {},
    )
    return {
        "row_count": len(rows),
        "primary_arm": primary.get("arm", ""),
        "primary_holdout_ce": _metric_float(primary.get("holdout_ce")),
        "ce_gain_vs_reference_sparse": _metric_float(primary.get("ce_gain_vs_reference_sparse")),
        "stored_gap_closed_fraction": _metric_float(primary.get("stored_gap_closed_fraction")),
        "ce_minus_flat_control_ce": _metric_float(primary.get("ce_minus_flat_control_ce")),
        "mean_gate_value": _metric_float(primary.get("mean_gate_value")),
        "signal_gate_ok": primary.get("signal_gate_ok") is True,
        "flat_control_ok": primary.get("flat_control_ok") is True,
        "norm_budget_ok": primary.get("norm_budget_ok") is True,
        "commutator_budget_ok": primary.get("commutator_budget_ok") is True,
        "functional_churn_budget_ok": primary.get("functional_churn_budget_ok") is True,
        "pregate_passes": primary.get("pregate_passes") is True,
        "advance_to_gpu_validation": any(row.get("advance_to_gpu_validation") is True for row in rows),
        "requires_gpu_now": any(row.get("requires_gpu_now") is True for row in rows),
        "promotion_allowed": any(row.get("promotion_allowed") is True for row in rows),
        "interpretation": (
            "Fail-closed local pregate for the selected budget-normalized gated low-rank sparse value mixture. "
            "It records whether the new value mechanism has signal after flat-control and interference budgets; "
            "it never promotes or requests GPU by itself."
        ),
    }


def _budget_normalized_gated_value_mixture_closeout_rows(
    *,
    pregate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary = _budget_normalized_gated_value_mixture_pregate_summary(pregate_rows)
    if summary.get("row_count", 0) <= 0:
        return []
    primary = next(
        (
            row
            for row in pregate_rows
            if row.get("pregate_role") == "primary_budget_normalized_gated_low_rank_value_mixture"
        ),
        pregate_rows[0],
    )
    pregate_passes = summary.get("pregate_passes") is True
    closeout_status = "repeat_before_gpu_validation" if pregate_passes else "closed_non_improving_flat_control_blocked"
    failure_reasons = []
    if summary.get("signal_gate_ok") is not True:
        failure_reasons.append("no_ce_or_stored_gap_signal")
    if summary.get("flat_control_ok") is not True:
        failure_reasons.append("same_router_flat_control_stronger")
    if summary.get("norm_budget_ok") is not True:
        failure_reasons.append("residual_norm_budget_failed")
    if summary.get("commutator_budget_ok") is not True:
        failure_reasons.append("finite_update_commutator_budget_failed")
    if summary.get("functional_churn_budget_ok") is not True:
        failure_reasons.append("functional_churn_budget_failed")
    selected_next_experiment = (
        "repeat_budget_normalized_gated_low_rank_value_mixture_on_adjacent_seed"
        if pregate_passes
        else "pivot_to_soft_mixture_low_churn_dense_modular_residual_design"
    )
    return [
        {
            "closeout_name": "budget_normalized_gated_value_mixture_closeout",
            "closeout_status": closeout_status,
            "source_primary_arm": summary.get("primary_arm", ""),
            "source_pregate_passes": pregate_passes,
            "source_primary_holdout_ce": summary.get("primary_holdout_ce"),
            "source_ce_gain_vs_reference_sparse": summary.get("ce_gain_vs_reference_sparse"),
            "source_stored_gap_closed_fraction": summary.get("stored_gap_closed_fraction"),
            "source_ce_minus_flat_control_ce": summary.get("ce_minus_flat_control_ce"),
            "source_mean_gate_value": summary.get("mean_gate_value"),
            "signal_gate_ok": summary.get("signal_gate_ok") is True,
            "flat_control_ok": summary.get("flat_control_ok") is True,
            "norm_budget_ok": summary.get("norm_budget_ok") is True,
            "commutator_budget_ok": summary.get("commutator_budget_ok") is True,
            "functional_churn_budget_ok": summary.get("functional_churn_budget_ok") is True,
            "interference_budgets_clear": bool(
                summary.get("norm_budget_ok") is True
                and summary.get("commutator_budget_ok") is True
                and summary.get("functional_churn_budget_ok") is True
            ),
            "flat_and_signal_gates_clear": bool(
                summary.get("signal_gate_ok") is True and summary.get("flat_control_ok") is True
            ),
            "branch_closed": not pregate_passes,
            "selected_next_experiment": selected_next_experiment,
            "source_failure_reasons": ";".join(failure_reasons),
            "primary_row_reference_sparse_ce": primary.get("reference_sparse_ce"),
            "primary_row_flat_control_ce": primary.get("flat_control_ce"),
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "advance_to_gpu_validation": False,
            "mechanism_labels_used_for_scoring_only": True,
            "interpretation": (
                "Closeout row for the selected budget-normalized gated low-rank value-mixture pregate. "
                "The branch stays local unless it has CE/stored-gap signal, does not lose to the flat "
                "same-router value control, and clears residual norm, commutator, and churn budgets."
            ),
        }
    ]


def _budget_normalized_gated_value_mixture_closeout_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "closeout_status": "",
            "selected_next_experiment": "",
            "requires_gpu_now": False,
            "promotion_allowed": False,
        }
    row = rows[0]
    return {
        "row_count": len(rows),
        "closeout_status": row.get("closeout_status", ""),
        "source_primary_arm": row.get("source_primary_arm", ""),
        "source_pregate_passes": row.get("source_pregate_passes") is True,
        "branch_closed": row.get("branch_closed") is True,
        "signal_gate_ok": row.get("signal_gate_ok") is True,
        "flat_control_ok": row.get("flat_control_ok") is True,
        "interference_budgets_clear": row.get("interference_budgets_clear") is True,
        "source_failure_reasons": row.get("source_failure_reasons", ""),
        "selected_next_experiment": row.get("selected_next_experiment", ""),
        "requires_gpu_now": any(closeout.get("requires_gpu_now") is True for closeout in rows),
        "promotion_allowed": any(closeout.get("promotion_allowed") is True for closeout in rows),
        "advance_to_gpu_validation": any(closeout.get("advance_to_gpu_validation") is True for closeout in rows),
        "interpretation": (
            "Closeout row for the failed budget-normalized gated value-mixture pregate. "
            "It records the flat-control/signal blocker before pivoting to the next non-PC local branch."
        ),
    }


def _soft_mixture_low_churn_dense_modular_design_rows(
    *,
    closeout_rows: list[dict[str, Any]],
    arm_metrics: list[dict[str, Any]],
    residual_budget_rows: list[dict[str, Any]],
    commutator_rows: list[dict[str, Any]],
    forgetting_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    closeout = _budget_normalized_gated_value_mixture_closeout_summary(closeout_rows)
    if closeout.get("branch_closed") is not True:
        return []

    by_arm = {str(row.get("arm", "")): row for row in arm_metrics}
    budget_by_arm = {str(row.get("arm", "")): row for row in residual_budget_rows}
    promoted = by_arm.get("promoted_contextual_topk2")
    low_churn_active = by_arm.get("low_churn_mlp_active_matched")
    dense_active = by_arm.get("dense_rank_norm_matched")
    low_churn_stored = by_arm.get("low_churn_mlp_stored_parameter_matched")
    if promoted is None or low_churn_active is None:
        return []

    promoted_ce = _metric_float(promoted.get("holdout_ce"))
    active_ce = _metric_float(low_churn_active.get("holdout_ce"))
    dense_ce = _metric_float(dense_active.get("holdout_ce")) if dense_active else None
    stored_ce = _metric_float(low_churn_stored.get("holdout_ce")) if low_churn_stored else None
    promoted_norm = _metric_float(promoted.get("residual_l2"))
    active_norm = _metric_float(low_churn_active.get("residual_l2"))
    active_commutator = _mean_metric(
        commutator_rows,
        ["low_churn_mlp_active_matched"],
        "finite_update_commutator_l2",
    )
    promoted_commutator = _mean_metric(
        commutator_rows,
        ["promoted_contextual_topk2"],
        "finite_update_commutator_l2",
    )
    active_churn = _mean_abs_metric(
        forgetting_rows,
        ["low_churn_mlp_active_matched"],
        "functional_churn",
    )
    promoted_churn = _mean_abs_metric(
        forgetting_rows,
        ["promoted_contextual_topk2"],
        "functional_churn",
    )
    ce_gap_vs_active = _positive_gap(promoted_ce, active_ce)
    ce_gap_vs_stored = _positive_gap(promoted_ce, stored_ce)
    dense_active_gap = _delta_value(active_ce, dense_ce)
    churn_ratio = _safe_ratio(active_churn, promoted_churn)
    commutator_ratio = _safe_ratio(active_commutator, promoted_commutator)
    norm_ratio = _safe_ratio(active_norm, promoted_norm)
    design_allowed = bool(
        closeout.get("selected_next_experiment")
        == "pivot_to_soft_mixture_low_churn_dense_modular_residual_design"
    )
    return [
        {
            "design_name": "soft_mixture_low_churn_dense_modular_residual",
            "source_closeout_status": closeout.get("closeout_status", ""),
            "source_failed_branch": closeout.get("source_primary_arm", ""),
            "source_failure_reasons": closeout.get("source_failure_reasons", ""),
            "candidate_family": "dense_modular_control",
            "candidate_mechanism": "soft_mixture_of_low_churn_experts_with_modularity_penalties",
            "candidate_status": "design_scaffold_selected" if design_allowed else "blocked_by_source_closeout",
            "reference_sparse_arm": "promoted_contextual_topk2",
            "reference_sparse_ce": promoted_ce,
            "active_low_churn_control_arm": "low_churn_mlp_active_matched",
            "active_low_churn_control_ce": active_ce,
            "dense_active_control_arm": "dense_rank_norm_matched" if dense_active else "",
            "dense_active_control_ce": dense_ce,
            "stored_low_churn_control_arm": "low_churn_mlp_stored_parameter_matched" if low_churn_stored else "",
            "stored_low_churn_control_ce": stored_ce,
            "reference_sparse_gap_to_active_low_churn_ce": ce_gap_vs_active,
            "reference_sparse_gap_to_stored_low_churn_ce": ce_gap_vs_stored,
            "active_low_churn_ce_minus_dense_active_ce": dense_active_gap,
            "active_low_churn_residual_l2": active_norm,
            "reference_sparse_residual_l2": promoted_norm,
            "active_low_churn_norm_ratio_vs_reference_sparse": norm_ratio,
            "active_low_churn_mean_commutator_l2": active_commutator,
            "reference_sparse_mean_commutator_l2": promoted_commutator,
            "active_low_churn_commutator_ratio_vs_reference_sparse": commutator_ratio,
            "active_low_churn_mean_abs_functional_churn": active_churn,
            "reference_sparse_mean_abs_functional_churn": promoted_churn,
            "active_low_churn_churn_ratio_vs_reference_sparse": churn_ratio,
            "active_low_churn_flop_proxy_per_token": _metric_float(
                budget_by_arm.get("low_churn_mlp_active_matched", {}).get("flop_proxy_per_token")
            ),
            "reference_sparse_flop_proxy_per_token": _metric_float(
                budget_by_arm.get("promoted_contextual_topk2", {}).get("flop_proxy_per_token")
            ),
            "required_controls": (
                "promoted_contextual_topk2;dense_rank_norm_matched;low_churn_mlp_active_matched;"
                "low_churn_mlp_stored_parameter_matched;token_position_router_topk2;random_support_topk2"
            ),
            "required_next_packet": (
                "trainable_soft_mixture_low_churn_dense_modular_residual_with_expert_dropout;"
                "entropy_or_load_balance_penalty;norm_clamp;commutator_churn_intervention_fingerprints"
            ),
            "implemented_in_current_packet": False,
            "design_selected": design_allowed,
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "advance_to_gpu_validation": False,
            "mechanism_labels_used_for_scoring_only": True,
            "interpretation": (
                "Local design selector after the gated sparse value-mixture closeout. It records why the "
                "next non-PC step should test a soft-mixture low-churn dense modular residual against the "
                "promoted sparse router and dense/MLP controls, but it does not train or promote that arm."
            ),
        }
    ]


def _soft_mixture_low_churn_dense_modular_design_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "design_selected": False,
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "interpretation": "soft-mixture low-churn dense modular design selector was not run",
        }
    row = rows[0]
    return {
        "row_count": len(rows),
        "design_name": row.get("design_name", ""),
        "candidate_status": row.get("candidate_status", ""),
        "source_failed_branch": row.get("source_failed_branch", ""),
        "source_failure_reasons": row.get("source_failure_reasons", ""),
        "reference_sparse_gap_to_active_low_churn_ce": _metric_float(
            row.get("reference_sparse_gap_to_active_low_churn_ce")
        ),
        "reference_sparse_gap_to_stored_low_churn_ce": _metric_float(
            row.get("reference_sparse_gap_to_stored_low_churn_ce")
        ),
        "active_low_churn_churn_ratio_vs_reference_sparse": _metric_float(
            row.get("active_low_churn_churn_ratio_vs_reference_sparse")
        ),
        "active_low_churn_commutator_ratio_vs_reference_sparse": _metric_float(
            row.get("active_low_churn_commutator_ratio_vs_reference_sparse")
        ),
        "implemented_in_current_packet": row.get("implemented_in_current_packet") is True,
        "design_selected": row.get("design_selected") is True,
        "requires_gpu_now": any(item.get("requires_gpu_now") is True for item in rows),
        "promotion_allowed": any(item.get("promotion_allowed") is True for item in rows),
        "advance_to_gpu_validation": any(item.get("advance_to_gpu_validation") is True for item in rows),
        "selected_next_experiment": (
            "implement_local_soft_mixture_low_churn_dense_modular_residual_pregate"
            if row.get("design_selected") is True
            else ""
        ),
        "interpretation": (
            "Design-only local branch selector. It consumes the failed gated sparse value-mixture closeout "
            "and defines the next bounded non-PC packet without requesting GPU validation."
        ),
    }


def _pc_core_periphery_residual_inference_pregate_rows(
    *,
    gated_pregate_rows: list[dict[str, Any]],
    arm_metrics: list[dict[str, Any]],
    residual_budget_rows: list[dict[str, Any]],
    commutator_rows: list[dict[str, Any]],
    forgetting_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    gated_summary = _budget_normalized_gated_value_mixture_pregate_summary(gated_pregate_rows)
    if not gated_pregate_rows or gated_summary.get("pregate_passes") is True:
        return []

    by_arm = {str(row.get("arm", "")): row for row in arm_metrics}
    budget_by_arm = {str(row.get("arm", "")): row for row in residual_budget_rows}
    reference = by_arm.get("promoted_contextual_topk2")
    flat = by_arm.get("pc_same_router_flat_mlp_control_topk2") or by_arm.get("flat_column_value_mlp_topk2")
    gated = by_arm.get("budget_normalized_gated_low_rank_value_mixture_topk2")
    pc_primary = by_arm.get("pc_core_periphery_residual_inference_topk2")
    stored = _best_ce_row(
        [row for row in arm_metrics if row.get("control_budget_role") == "stored_parameter_matched_dense_mlp_upper_bound"]
    )
    if reference is None or flat is None or gated is None:
        return []

    reference_ce = _metric_float(reference.get("holdout_ce"))
    flat_ce = _metric_float(flat.get("holdout_ce"))
    gated_ce = _metric_float(gated.get("holdout_ce"))
    stored_ce = _metric_float(stored.get("holdout_ce")) if stored else None
    stored_gap = _positive_gap(reference_ce, stored_ce)
    reference_norm = _metric_float(reference.get("residual_l2"))
    reference_commutator = _mean_metric(commutator_rows, ["promoted_contextual_topk2"], "finite_update_commutator_l2")
    reference_churn = _mean_abs_metric(forgetting_rows, ["promoted_contextual_topk2"], "functional_churn")

    rows: list[dict[str, Any]] = []
    for role, source_arm, control_family, required in (
        (
            "primary_pc_core_periphery_residual_inference_design",
            "pc_core_periphery_residual_inference_topk2" if pc_primary is not None else "not_yet_trained",
            "pc_core_periphery_sparse",
            True,
        ),
        ("same_router_flat_mlp_control_required", "pc_same_router_flat_mlp_control_topk2", "flat_same_router_value_capacity_control", True),
        ("norm_clipped_stored_dense_control_required", "low_churn_mlp_stored_parameter_matched", "stored_parameter_matched_dense_mlp_upper_bound", True),
        ("token_position_router_null_required", "token_position_router_topk2", "token_position_only_router_null", True),
        ("random_support_histogram_null_required", "random_support_topk2", "random_support_null", True),
        ("shuffled_target_null_required", "pc_shuffled_residual_error_target_null_topk2", "shuffled_target_null", True),
    ):
        source = by_arm.get(source_arm, {}) if source_arm != "not_yet_trained" else {}
        source_ce = _metric_float(source.get("holdout_ce"))
        arm_commutator = _mean_metric(commutator_rows, [source_arm], "finite_update_commutator_l2")
        arm_churn = _mean_abs_metric(forgetting_rows, [source_arm], "functional_churn")
        budget = budget_by_arm.get(source_arm, {})
        is_primary = role == "primary_pc_core_periphery_residual_inference_design"
        ce_gain = _delta_value(reference_ce, source_ce)
        stored_gap_closed = _safe_ratio(ce_gain, stored_gap)
        flat_margin = _delta_value(source_ce, flat_ce)
        norm = _metric_float(source.get("residual_l2"))
        signal_ok = bool(
            is_primary
            and (
                (ce_gain is not None and ce_gain >= 0.05)
                or (stored_gap_closed is not None and stored_gap_closed >= 0.10)
            )
        )
        flat_control_ok = bool(not is_primary or flat_margin is None or flat_margin <= 0.01)
        norm_ok = bool(
            not is_primary
            or (norm is not None and reference_norm is not None and norm <= reference_norm * 1.05)
        )
        commutator_ok = bool(
            not is_primary
            or (
                arm_commutator is not None
                and reference_commutator is not None
                and arm_commutator <= reference_commutator * 1.10
            )
        )
        churn_ok = bool(
            not is_primary
            or (
                arm_churn is not None
                and reference_churn is not None
                and arm_churn <= reference_churn * 1.10
            )
        )
        pregate_passes = bool(
            is_primary
            and bool(source)
            and signal_ok
            and flat_control_ok
            and norm_ok
            and commutator_ok
            and churn_ok
        )
        rows.append(
            {
                "pregate_name": "pc_core_periphery_residual_inference_pregate",
                "pregate_role": role,
                "source_arm": source_arm,
                "control_family": control_family,
                "required_for_next_packet": required,
                "implemented_in_current_packet": source_arm != "not_yet_trained" and bool(source),
                "selected": role == "primary_pc_core_periphery_residual_inference_design",
                "source_failed_branch": "budget_normalized_gated_low_rank_value_mixture",
                "source_failed_branch_holdout_ce": gated_ce,
                "source_failed_branch_pregate_passes": gated_summary.get("pregate_passes") is True,
                "source_failed_branch_signal_gate_ok": gated_summary.get("signal_gate_ok") is True,
                "source_failed_branch_flat_control_ok": gated_summary.get("flat_control_ok") is True,
                "source_failed_branch_norm_budget_ok": gated_summary.get("norm_budget_ok") is True,
                "source_failed_branch_commutator_budget_ok": gated_summary.get("commutator_budget_ok") is True,
                "source_failed_branch_functional_churn_budget_ok": gated_summary.get("functional_churn_budget_ok") is True,
                "reference_sparse_arm": "promoted_contextual_topk2",
                "reference_sparse_ce": reference_ce,
                "stored_control_arm": stored.get("arm", "") if stored else "",
                "stored_control_ce": stored_ce,
                "reference_sparse_gap_to_stored_control": stored_gap,
                "flat_control_arm": "flat_column_value_mlp_topk2",
                "flat_control_ce": flat_ce,
                "source_holdout_ce": source_ce,
                "source_ce_minus_reference_sparse_ce": _delta_value(source_ce, reference_ce),
                "source_ce_minus_flat_control_ce": _delta_value(source_ce, flat_ce),
                "ce_gain_vs_reference_sparse": ce_gain,
                "stored_gap_closed_fraction": stored_gap_closed,
                "source_residual_l2": norm,
                "source_active_parameters_proxy": _metric_float(source.get("active_parameters_proxy")),
                "source_stored_parameters": _metric_float(source.get("stored_parameters")),
                "source_flop_proxy_per_token": _metric_float(budget.get("flop_proxy_per_token")),
                "source_mean_commutator_l2": arm_commutator,
                "source_mean_abs_functional_churn": arm_churn,
                "pc_inference_steps": _metric_float(source.get("pc_inference_steps")),
                "pc_error_prediction_weight": _metric_float(source.get("pc_error_prediction_weight")),
                "source_core_parameter_drift_l2": _metric_float(source.get("core_parameter_drift_l2")),
                "source_mean_gate_value": _metric_float(source.get("mean_gate_value")),
                "signal_gate_ok": signal_ok,
                "flat_control_ok": flat_control_ok,
                "norm_budget_ok": norm_ok,
                "commutator_budget_ok": commutator_ok,
                "functional_churn_budget_ok": churn_ok,
                "pregate_passes": pregate_passes,
                "mechanism": (
                    "fixed contextual top-k2 router; protected shared predictive core; plastic peripheral "
                    "residual-error units; two local inference steps; residual norm clamp"
                ),
                "training_objective": "supervised_ce_plus_anchor_kl_plus_local_residual_error_prediction_plus_commutator_churn_penalties",
                "advance_gate": (
                    "beat promoted sparse by >=0.05 CE or close >=10% of stored-control gap, while flat control is "
                    "not stronger and norm, churn, commutator, pruning-selectivity, and anchor-KL budgets pass"
                ),
                "next_experiment": (
                    "repeat_trainable_pc_core_periphery_residual_inference_pregate_on_adjacent_seed"
                    if pregate_passes
                    else "inspect_or_redesign_local_pc_core_periphery_residual_inference_pregate"
                    if is_primary and bool(source)
                    else "implement_trainable_pc_core_periphery_residual_inference_pregate"
                ),
                "requires_gpu_now": False,
                "promotion_allowed": False,
                "advance_to_gpu_validation": False,
                "strategic_change_level": "major",
                "notify_ben": True,
                "mechanism_labels_used_for_scoring_only": True,
                "interpretation": (
                    "Major-pivot scaffold from the external GPT-5.5-Pro review. The current sparse value-capacity "
                    "branch is closed locally; this row defines the next command-driven trainable PC-style "
                    "residual-inference packet and required same-router/null controls. It is not evidence that the "
                    "PC arm works yet."
                ),
            }
        )
    return rows


def _pc_core_periphery_residual_inference_pregate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "selected_next_experiment": "",
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "notify_ben": False,
            "strategic_change_level": "minor",
            "interpretation": "PC core/periphery residual-inference pregate scaffold was not emitted",
        }
    selected = next((row for row in rows if row.get("selected") is True), rows[0])
    return {
        "row_count": len(rows),
        "selected_next_experiment": selected.get("next_experiment", ""),
        "source_failed_branch": selected.get("source_failed_branch", ""),
        "source_failed_branch_holdout_ce": _metric_float(selected.get("source_failed_branch_holdout_ce")),
        "reference_sparse_ce": _metric_float(selected.get("reference_sparse_ce")),
        "flat_control_ce": _metric_float(selected.get("flat_control_ce")),
        "stored_control_ce": _metric_float(selected.get("stored_control_ce")),
        "primary_arm": selected.get("source_arm", ""),
        "primary_holdout_ce": _metric_float(selected.get("source_holdout_ce")),
        "ce_gain_vs_reference_sparse": _metric_float(selected.get("ce_gain_vs_reference_sparse")),
        "stored_gap_closed_fraction": _metric_float(selected.get("stored_gap_closed_fraction")),
        "signal_gate_ok": selected.get("signal_gate_ok") is True,
        "flat_control_ok": selected.get("flat_control_ok") is True,
        "norm_budget_ok": selected.get("norm_budget_ok") is True,
        "commutator_budget_ok": selected.get("commutator_budget_ok") is True,
        "functional_churn_budget_ok": selected.get("functional_churn_budget_ok") is True,
        "pregate_passes": selected.get("pregate_passes") is True,
        "trainable_pc_packet_implemented": selected.get("implemented_in_current_packet") is True,
        "required_control_count": sum(1 for row in rows if row.get("required_for_next_packet") is True),
        "current_packet_implemented_control_count": sum(
            1 for row in rows if row.get("implemented_in_current_packet") is True
        ),
        "requires_gpu_now": any(row.get("requires_gpu_now") is True for row in rows),
        "promotion_allowed": any(row.get("promotion_allowed") is True for row in rows),
        "advance_to_gpu_validation": any(row.get("advance_to_gpu_validation") is True for row in rows),
        "notify_ben": any(row.get("notify_ben") is True for row in rows),
        "strategic_change_level": "major" if any(row.get("strategic_change_level") == "major" for row in rows) else "minor",
        "interpretation": (
            "Fail-closed scaffold for the major pivot to PC-style core/periphery residual inference. "
            "It closes the current value-capacity branch as unsupported and defines the next local trainable "
            "packet with required controls before any GPU validation."
        ),
    }


def _pc_residual_inference_mechanism_inspection_rows(
    *,
    pc_pregate_rows: list[dict[str, Any]],
    arm_metrics: list[dict[str, Any]],
    residual_budget_rows: list[dict[str, Any]],
    commutator_rows: list[dict[str, Any]],
    forgetting_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pc_summary = _pc_core_periphery_residual_inference_pregate_summary(pc_pregate_rows)
    if not pc_pregate_rows or pc_summary.get("trainable_pc_packet_implemented") is not True:
        return []
    if pc_summary.get("pregate_passes") is True:
        return []

    by_arm = {str(row.get("arm", "")): row for row in arm_metrics}
    budget_by_arm = {str(row.get("arm", "")): row for row in residual_budget_rows}
    reference = by_arm.get("promoted_contextual_topk2")
    pc_primary = by_arm.get("pc_core_periphery_residual_inference_topk2")
    flat = by_arm.get("pc_same_router_flat_mlp_control_topk2")
    shuffled = by_arm.get("pc_shuffled_residual_error_target_null_topk2")
    gated = by_arm.get("budget_normalized_gated_low_rank_value_mixture_topk2")
    stored = _best_ce_row(
        [row for row in arm_metrics if row.get("control_budget_role") == "stored_parameter_matched_dense_mlp_upper_bound"]
    )
    if reference is None or pc_primary is None:
        return []

    reference_ce = _metric_float(reference.get("holdout_ce"))
    primary_ce = _metric_float(pc_primary.get("holdout_ce"))
    flat_ce = _metric_float(flat.get("holdout_ce")) if flat else None
    shuffled_ce = _metric_float(shuffled.get("holdout_ce")) if shuffled else None
    gated_ce = _metric_float(gated.get("holdout_ce")) if gated else None
    stored_ce = _metric_float(stored.get("holdout_ce")) if stored else None
    stored_gap = _positive_gap(reference_ce, stored_ce)
    reference_commutator = _mean_metric(commutator_rows, ["promoted_contextual_topk2"], "finite_update_commutator_l2")
    primary_commutator = _mean_metric(commutator_rows, ["pc_core_periphery_residual_inference_topk2"], "finite_update_commutator_l2")
    primary_churn = _mean_abs_metric(forgetting_rows, ["pc_core_periphery_residual_inference_topk2"], "functional_churn")
    reference_churn = _mean_abs_metric(forgetting_rows, ["promoted_contextual_topk2"], "functional_churn")
    flat_margin = _delta_value(primary_ce, flat_ce)
    shuffled_margin = _delta_value(primary_ce, shuffled_ce)
    reference_margin = _delta_value(primary_ce, reference_ce)
    gated_margin = _delta_value(primary_ce, gated_ce)
    stored_gap_closed = _safe_ratio(_delta_value(reference_ce, primary_ce), stored_gap)
    commutator_ratio = _safe_ratio(primary_commutator, reference_commutator)
    churn_ratio = _safe_ratio(primary_churn, reference_churn)
    primary_budget = budget_by_arm.get("pc_core_periphery_residual_inference_topk2", {})

    failure_reasons = []
    if pc_summary.get("signal_gate_ok") is not True:
        failure_reasons.append("no_signal_vs_promoted_sparse_or_stored_gap")
    if pc_summary.get("flat_control_ok") is not True:
        failure_reasons.append("same_router_flat_control_stronger")
    if pc_summary.get("commutator_budget_ok") is not True:
        failure_reasons.append("finite_update_commutator_budget_failed")
    if pc_summary.get("norm_budget_ok") is not True:
        failure_reasons.append("residual_norm_budget_failed")
    if pc_summary.get("functional_churn_budget_ok") is not True:
        failure_reasons.append("functional_churn_budget_failed")

    if shuffled_margin is not None and shuffled_margin >= -0.01:
        selected_next = "audit_pc_error_target_and_inference_path_before_retraining"
        redesign_hint = "shuffled target null is competitive with the PC primary; inspect residual-error target alignment and inference update before adding capacity"
    elif flat_margin is not None and flat_margin > 0.01:
        selected_next = "prototype_minimal_pc_inference_ablation_against_flat_control"
        redesign_hint = "same-router flat value control is stronger; ablate core prediction, inferred error path, and gate before a larger PC redesign"
    elif commutator_ratio is not None and commutator_ratio > 1.10:
        selected_next = "add_commutator_budgeted_pc_inference_ablation"
        redesign_hint = "PC primary fails update-order stability; test an inference/anchor ablation that targets commutator before any GPU"
    else:
        selected_next = "redesign_local_pc_residual_inference_mechanism_without_gpu"
        redesign_hint = "PC primary fails the local pregate; redesign remains local and diagnostic"

    rows: list[dict[str, Any]] = []
    for role, arm, family, selected in (
        ("primary_pc_failure_fingerprint", "pc_core_periphery_residual_inference_topk2", "pc_core_periphery_sparse", True),
        ("same_router_flat_comparator", "pc_same_router_flat_mlp_control_topk2", "flat_same_router_value_capacity_control", False),
        ("shuffled_target_null_comparator", "pc_shuffled_residual_error_target_null_topk2", "shuffled_target_null", False),
        ("promoted_sparse_reference", "promoted_contextual_topk2", "sparse_reference", False),
        ("gated_value_branch_reference", "budget_normalized_gated_low_rank_value_mixture_topk2", "closed_value_capacity_branch", False),
    ):
        source = by_arm.get(arm, {})
        source_ce = _metric_float(source.get("holdout_ce"))
        source_commutator = _mean_metric(commutator_rows, [arm], "finite_update_commutator_l2")
        source_churn = _mean_abs_metric(forgetting_rows, [arm], "functional_churn")
        rows.append(
            {
                "inspection_name": "pc_residual_inference_mechanism_inspection",
                "inspection_role": role,
                "arm": arm,
                "control_family": family,
                "selected": selected,
                "source_holdout_ce": source_ce,
                "source_ce_minus_reference_sparse_ce": _delta_value(source_ce, reference_ce),
                "source_ce_minus_primary_pc_ce": _delta_value(source_ce, primary_ce),
                "source_ce_minus_flat_control_ce": _delta_value(source_ce, flat_ce),
                "source_ce_minus_shuffled_target_null_ce": _delta_value(source_ce, shuffled_ce),
                "source_mean_commutator_l2": source_commutator,
                "source_commutator_ratio_vs_reference_sparse": _safe_ratio(source_commutator, reference_commutator),
                "source_mean_abs_functional_churn": source_churn,
                "source_churn_ratio_vs_reference_sparse": _safe_ratio(source_churn, reference_churn),
                "source_residual_l2": _metric_float(source.get("residual_l2")),
                "source_active_parameters_proxy": _metric_float(source.get("active_parameters_proxy")),
                "source_stored_parameters": _metric_float(source.get("stored_parameters")),
                "source_flop_proxy_per_token": _metric_float(budget_by_arm.get(arm, {}).get("flop_proxy_per_token")),
                "primary_ce_gain_vs_reference_sparse": _delta_value(reference_ce, primary_ce),
                "primary_ce_minus_flat_control_ce": flat_margin,
                "primary_ce_minus_shuffled_target_null_ce": shuffled_margin,
                "primary_ce_minus_gated_value_branch_ce": gated_margin,
                "primary_stored_gap_closed_fraction": stored_gap_closed,
                "primary_commutator_ratio_vs_reference_sparse": commutator_ratio,
                "primary_churn_ratio_vs_reference_sparse": churn_ratio,
                "primary_pc_inference_steps": _metric_float(pc_primary.get("pc_inference_steps")),
                "primary_pc_error_prediction_weight": _metric_float(pc_primary.get("pc_error_prediction_weight")),
                "primary_mean_gate_value": _metric_float(pc_primary.get("mean_gate_value")),
                "primary_flop_proxy_per_token": _metric_float(primary_budget.get("flop_proxy_per_token")),
                "failed_gate_count": len(failure_reasons),
                "failure_reasons": ";".join(failure_reasons),
                "selected_next_experiment": selected_next,
                "redesign_hint": redesign_hint,
                "requires_gpu_now": False,
                "promotion_allowed": False,
                "advance_to_gpu_validation": False,
                "strategic_change_level": "major",
                "notify_ben": True,
                "mechanism_labels_used_for_scoring_only": True,
            }
        )
    return rows


def _pc_residual_inference_mechanism_inspection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "selected_next_experiment": "",
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "notify_ben": False,
            "strategic_change_level": "minor",
        }
    selected = next((row for row in rows if row.get("selected") is True), rows[0])
    return {
        "row_count": len(rows),
        "selected_next_experiment": selected.get("selected_next_experiment", ""),
        "redesign_hint": selected.get("redesign_hint", ""),
        "failed_gate_count": int(selected.get("failed_gate_count", 0)),
        "failure_reasons": selected.get("failure_reasons", ""),
        "primary_ce_gain_vs_reference_sparse": _metric_float(selected.get("primary_ce_gain_vs_reference_sparse")),
        "primary_ce_minus_flat_control_ce": _metric_float(selected.get("primary_ce_minus_flat_control_ce")),
        "primary_ce_minus_shuffled_target_null_ce": _metric_float(
            selected.get("primary_ce_minus_shuffled_target_null_ce")
        ),
        "primary_stored_gap_closed_fraction": _metric_float(selected.get("primary_stored_gap_closed_fraction")),
        "primary_commutator_ratio_vs_reference_sparse": _metric_float(
            selected.get("primary_commutator_ratio_vs_reference_sparse")
        ),
        "primary_churn_ratio_vs_reference_sparse": _metric_float(selected.get("primary_churn_ratio_vs_reference_sparse")),
        "requires_gpu_now": any(row.get("requires_gpu_now") is True for row in rows),
        "promotion_allowed": any(row.get("promotion_allowed") is True for row in rows),
        "advance_to_gpu_validation": any(row.get("advance_to_gpu_validation") is True for row in rows),
        "notify_ben": any(row.get("notify_ben") is True for row in rows),
        "strategic_change_level": "major" if any(row.get("strategic_change_level") == "major" for row in rows) else "minor",
        "interpretation": (
            "Fail-closed PC mechanism inspection after the trainable residual-inference pregate. "
            "It fingerprints whether the local negative is dominated by missing CE signal, stronger "
            "same-router flat control, shuffled-target null competitiveness, or update-order instability."
        ),
    }


def _pc_error_target_inference_path_audit_rows(
    *,
    pc_inspection_rows: list[dict[str, Any]],
    arm_metrics: list[dict[str, Any]],
    residual_budget_rows: list[dict[str, Any]],
    commutator_rows: list[dict[str, Any]],
    forgetting_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    inspection_summary = _pc_residual_inference_mechanism_inspection_summary(pc_inspection_rows)
    if inspection_summary.get("selected_next_experiment") != "audit_pc_error_target_and_inference_path_before_retraining":
        return []

    by_arm = {str(row.get("arm", "")): row for row in arm_metrics}
    budget_by_arm = {str(row.get("arm", "")): row for row in residual_budget_rows}
    reference = by_arm.get("promoted_contextual_topk2")
    primary = by_arm.get("pc_core_periphery_residual_inference_topk2")
    flat = by_arm.get("pc_same_router_flat_mlp_control_topk2")
    shuffled = by_arm.get("pc_shuffled_residual_error_target_null_topk2")
    gated = by_arm.get("budget_normalized_gated_low_rank_value_mixture_topk2")
    if reference is None or primary is None:
        return []

    reference_ce = _metric_float(reference.get("holdout_ce"))
    primary_ce = _metric_float(primary.get("holdout_ce"))
    flat_ce = _metric_float(flat.get("holdout_ce")) if flat else None
    shuffled_ce = _metric_float(shuffled.get("holdout_ce")) if shuffled else None
    gated_ce = _metric_float(gated.get("holdout_ce")) if gated else None
    reference_commutator = _mean_metric(commutator_rows, ["promoted_contextual_topk2"], "finite_update_commutator_l2")
    primary_commutator = _mean_metric(commutator_rows, ["pc_core_periphery_residual_inference_topk2"], "finite_update_commutator_l2")
    primary_churn = _mean_abs_metric(forgetting_rows, ["pc_core_periphery_residual_inference_topk2"], "functional_churn")
    reference_churn = _mean_abs_metric(forgetting_rows, ["promoted_contextual_topk2"], "functional_churn")
    primary_budget = budget_by_arm.get("pc_core_periphery_residual_inference_topk2", {})

    ce_gain = _delta_value(reference_ce, primary_ce)
    flat_margin = _delta_value(primary_ce, flat_ce)
    shuffled_margin = _delta_value(primary_ce, shuffled_ce)
    gated_margin = _delta_value(primary_ce, gated_ce)
    commutator_ratio = _safe_ratio(primary_commutator, reference_commutator)
    churn_ratio = _safe_ratio(primary_churn, reference_churn)
    target_alignment_ok = bool(shuffled_margin is not None and shuffled_margin <= -0.01)
    inference_value_ok = bool(ce_gain is not None and ce_gain > 0.0)
    flat_control_ok = bool(flat_margin is None or flat_margin <= 0.01)
    commutator_ok = bool(commutator_ratio is not None and commutator_ratio <= 1.10)
    retraining_ready = bool(target_alignment_ok and inference_value_ok and flat_control_ok and commutator_ok)
    selected_next = (
        "prototype_minimal_pc_inference_ablation_against_flat_control"
        if retraining_ready
        else "fix_pc_residual_error_target_alignment_before_retraining"
    )
    blockers = []
    if not target_alignment_ok:
        blockers.append("shuffled_target_null_competitive")
    if not inference_value_ok:
        blockers.append("pc_inference_no_ce_signal_vs_promoted_sparse")
    if not flat_control_ok:
        blockers.append("same_router_flat_control_stronger")
    if not commutator_ok:
        blockers.append("finite_update_commutator_budget_failed")

    rows: list[dict[str, Any]] = []
    for role, diagnostic, selected, pass_value, interpretation in (
        (
            "primary_audit_decision",
            "pc_error_target_and_inference_path",
            True,
            retraining_ready,
            "Fail-closed local audit before retraining the PC arm; GPU remains blocked until target alignment, inference signal, flat-control, and commutator checks pass.",
        ),
        (
            "residual_error_target_alignment",
            "pc_primary_vs_shuffled_residual_error_target_null",
            False,
            target_alignment_ok,
            "The PC target is considered aligned only when the trained PC primary beats the shuffled residual-error target null by at least 0.01 CE.",
        ),
        (
            "inference_path_signal",
            "pc_primary_vs_promoted_sparse_and_closed_gated_branch",
            False,
            inference_value_ok,
            "The current inference path must show positive CE signal versus promoted sparse before adding capacity or retraining variants.",
        ),
        (
            "flat_control_and_update_stability",
            "pc_primary_vs_same_router_flat_control_and_commutator",
            False,
            flat_control_ok and commutator_ok,
            "The PC mechanism must not lose to the same-router flat MLP and must satisfy the finite-update commutator budget.",
        ),
    ):
        rows.append(
            {
                "audit_name": "pc_error_target_inference_path_audit",
                "audit_role": role,
                "diagnostic": diagnostic,
                "selected": selected,
                "diagnostic_passes": pass_value,
                "primary_arm": "pc_core_periphery_residual_inference_topk2",
                "reference_sparse_arm": "promoted_contextual_topk2",
                "flat_control_arm": "pc_same_router_flat_mlp_control_topk2" if flat else "",
                "shuffled_target_null_arm": "pc_shuffled_residual_error_target_null_topk2" if shuffled else "",
                "closed_gated_branch_arm": "budget_normalized_gated_low_rank_value_mixture_topk2" if gated else "",
                "primary_holdout_ce": primary_ce,
                "reference_sparse_ce": reference_ce,
                "flat_control_ce": flat_ce,
                "shuffled_target_null_ce": shuffled_ce,
                "closed_gated_branch_ce": gated_ce,
                "primary_ce_gain_vs_reference_sparse": ce_gain,
                "primary_ce_minus_flat_control_ce": flat_margin,
                "primary_ce_minus_shuffled_target_null_ce": shuffled_margin,
                "primary_ce_minus_closed_gated_branch_ce": gated_margin,
                "primary_residual_l2": _metric_float(primary.get("residual_l2")),
                "primary_flop_proxy_per_token": _metric_float(primary_budget.get("flop_proxy_per_token")),
                "primary_pc_inference_steps": _metric_float(primary.get("pc_inference_steps")),
                "primary_pc_error_prediction_weight": _metric_float(primary.get("pc_error_prediction_weight")),
                "primary_mean_gate_value": _metric_float(primary.get("mean_gate_value")),
                "primary_commutator_ratio_vs_reference_sparse": commutator_ratio,
                "primary_churn_ratio_vs_reference_sparse": churn_ratio,
                "target_alignment_ok": target_alignment_ok,
                "inference_path_signal_ok": inference_value_ok,
                "flat_control_ok": flat_control_ok,
                "commutator_budget_ok": commutator_ok,
                "retraining_ready": retraining_ready,
                "failure_reasons": ";".join(blockers),
                "selected_next_experiment": selected_next,
                "requires_gpu_now": False,
                "promotion_allowed": False,
                "advance_to_gpu_validation": False,
                "strategic_change_level": "major",
                "notify_ben": True,
                "mechanism_labels_used_for_scoring_only": True,
                "interpretation": interpretation,
            }
        )
    return rows


def _pc_error_target_inference_path_audit_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "selected_next_experiment": "",
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "notify_ben": False,
            "strategic_change_level": "minor",
        }
    selected = next((row for row in rows if row.get("selected") is True), rows[0])
    return {
        "row_count": len(rows),
        "selected_next_experiment": selected.get("selected_next_experiment", ""),
        "target_alignment_ok": selected.get("target_alignment_ok") is True,
        "inference_path_signal_ok": selected.get("inference_path_signal_ok") is True,
        "flat_control_ok": selected.get("flat_control_ok") is True,
        "commutator_budget_ok": selected.get("commutator_budget_ok") is True,
        "retraining_ready": selected.get("retraining_ready") is True,
        "failure_reasons": selected.get("failure_reasons", ""),
        "primary_ce_gain_vs_reference_sparse": _metric_float(selected.get("primary_ce_gain_vs_reference_sparse")),
        "primary_ce_minus_flat_control_ce": _metric_float(selected.get("primary_ce_minus_flat_control_ce")),
        "primary_ce_minus_shuffled_target_null_ce": _metric_float(
            selected.get("primary_ce_minus_shuffled_target_null_ce")
        ),
        "primary_commutator_ratio_vs_reference_sparse": _metric_float(
            selected.get("primary_commutator_ratio_vs_reference_sparse")
        ),
        "requires_gpu_now": any(row.get("requires_gpu_now") is True for row in rows),
        "promotion_allowed": any(row.get("promotion_allowed") is True for row in rows),
        "advance_to_gpu_validation": any(row.get("advance_to_gpu_validation") is True for row in rows),
        "notify_ben": any(row.get("notify_ben") is True for row in rows),
        "strategic_change_level": "major" if any(row.get("strategic_change_level") == "major" for row in rows) else "minor",
        "interpretation": (
            "Fail-closed local audit of the PC residual-error target and inference path before any retraining. "
            "It blocks GPU and promotion when the shuffled-target null, same-router flat control, or commutator "
            "budget explains the PC result."
        ),
    }


def _pc_decoder_adjoint_target_alignment_probe_rows(
    *,
    pc_audit_rows: list[dict[str, Any]],
    hidden: Any,
    targets: Any,
    decode: Any,
    decoder_weight: Any,
    F: Any,
    torch: Any,
) -> list[dict[str, Any]]:
    audit_summary = _pc_error_target_inference_path_audit_summary(pc_audit_rows)
    if audit_summary.get("selected_next_experiment") != "fix_pc_residual_error_target_alignment_before_retraining":
        return []

    base_logits = decode(hidden)
    base_ce = float(F.cross_entropy(base_logits.reshape(-1, base_logits.shape[-1]), targets.reshape(-1)).detach().item())
    step_rms = 0.05
    current_target = F.layer_norm(
        decoder_weight.index_select(0, targets.reshape(-1)).reshape(
            targets.shape[0],
            targets.shape[1],
            -1,
        )
        - hidden,
        (hidden.shape[-1],),
    ).detach()
    decoder_adjoint = _decoder_adjoint_hidden_ce_error(
        hidden,
        targets,
        decoder_weight,
        decode=decode,
        F=F,
        normalize=True,
    ).detach()
    shuffled_targets = targets.reshape(-1).index_select(
        0,
        torch.arange(targets.numel() - 1, -1, -1, device=targets.device),
    ).reshape_as(targets)
    shuffled_decoder_adjoint = _decoder_adjoint_hidden_ce_error(
        hidden,
        shuffled_targets,
        decoder_weight,
        decode=decode,
        F=F,
        normalize=True,
    ).detach()
    finite_difference = _finite_difference_hidden_ce_descent_proxy(
        hidden=hidden,
        targets=targets,
        decode=decode,
        F=F,
    ).detach()

    target_map = {
        "current_decoder_embedding_minus_hidden": current_target,
        "decoder_adjoint_ce_descent": decoder_adjoint,
        "finite_difference_hidden_ce_descent_proxy": finite_difference,
        "shuffled_decoder_adjoint_target_null": shuffled_decoder_adjoint,
        "sign_flipped_decoder_adjoint_null": -decoder_adjoint,
    }
    metrics = {
        name: _target_injection_metrics(
            target=target,
            hidden=hidden,
            targets=targets,
            base_ce=base_ce,
            step_rms=step_rms,
            decode=decode,
            F=F,
        )
        for name, target in target_map.items()
    }
    decoder_delta = metrics["decoder_adjoint_ce_descent"]["injection_ce_delta"]
    current_delta = metrics["current_decoder_embedding_minus_hidden"]["injection_ce_delta"]
    shuffled_delta = metrics["shuffled_decoder_adjoint_target_null"]["injection_ce_delta"]
    sign_flipped_delta = metrics["sign_flipped_decoder_adjoint_null"]["injection_ce_delta"]
    finite_delta = metrics["finite_difference_hidden_ce_descent_proxy"]["injection_ce_delta"]
    decoder_lowers_ce = decoder_delta < 0.0
    decoder_beats_current = decoder_delta < current_delta - 1e-4
    decoder_beats_shuffled = decoder_delta < shuffled_delta - 1e-4
    decoder_beats_sign_flip = decoder_delta < sign_flipped_delta - 1e-4
    decoder_tracks_finite_difference = decoder_delta <= finite_delta + 1e-4
    alignment_gate_passes = bool(
        decoder_lowers_ce
        and decoder_beats_current
        and decoder_beats_shuffled
        and decoder_beats_sign_flip
        and decoder_tracks_finite_difference
    )
    selected_next = (
        "wire_decoder_adjoint_pc_target_into_minimal_retrain_probe"
        if alignment_gate_passes
        else "close_or_redesign_pc_target_before_retraining"
    )

    rows: list[dict[str, Any]] = []
    for name, role, selected in (
        ("decoder_adjoint_ce_descent", "primary_decoder_adjoint_ce_target", True),
        ("current_decoder_embedding_minus_hidden", "current_pc_target_control", False),
        ("finite_difference_hidden_ce_descent_proxy", "finite_difference_descent_proxy", False),
        ("shuffled_decoder_adjoint_target_null", "shuffled_target_null", False),
        ("sign_flipped_decoder_adjoint_null", "sign_flipped_target_null", False),
    ):
        row_metrics = metrics[name]
        rows.append(
            {
                "probe_name": "pc_decoder_adjoint_target_alignment_probe",
                "target_variant": name,
                "probe_role": role,
                "selected": selected,
                "base_ce": base_ce,
                "injected_ce": row_metrics["injected_ce"],
                "injection_ce_delta": row_metrics["injection_ce_delta"],
                "matched_step_rms": step_rms,
                "target_rms": row_metrics["target_rms"],
                "cosine_to_decoder_adjoint": _target_cosine(target_map[name], decoder_adjoint, F=F),
                "cosine_to_finite_difference_proxy": _target_cosine(target_map[name], finite_difference, F=F),
                "decoder_adjoint_injection_ce_delta": decoder_delta,
                "current_target_injection_ce_delta": current_delta,
                "finite_difference_injection_ce_delta": finite_delta,
                "shuffled_target_injection_ce_delta": shuffled_delta,
                "sign_flipped_target_injection_ce_delta": sign_flipped_delta,
                "decoder_lowers_ce": decoder_lowers_ce,
                "decoder_beats_current_target": decoder_beats_current,
                "decoder_beats_shuffled_target": decoder_beats_shuffled,
                "decoder_beats_sign_flip": decoder_beats_sign_flip,
                "decoder_tracks_finite_difference": decoder_tracks_finite_difference,
                "alignment_gate_passes": alignment_gate_passes,
                "selected_next_experiment": selected_next,
                "requires_gpu_now": False,
                "promotion_allowed": False,
                "advance_to_gpu_validation": False,
                "strategic_change_level": "minor",
                "notify_ben": False,
                "label_derived_training_only_target": True,
                "mechanism_labels_used_for_scoring_only": True,
                "interpretation": (
                    "Diagnostic target-alignment probe only. Decoder-adjoint CE targets use labels and can "
                    "validate sign/alignment for training-time PC experiments, but they are not deployable "
                    "label-free inference evidence."
                ),
            }
        )
    return rows


def _finite_difference_hidden_ce_descent_proxy(
    *,
    hidden: Any,
    targets: Any,
    decode: Any,
    F: Any,
) -> Any:
    probe_hidden = hidden.detach().clone().requires_grad_(True)
    logits = decode(probe_hidden)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    loss.backward()
    gradient = probe_hidden.grad.detach()
    return F.layer_norm(-gradient, (gradient.shape[-1],))


def _target_injection_metrics(
    *,
    target: Any,
    hidden: Any,
    targets: Any,
    base_ce: float,
    step_rms: float,
    decode: Any,
    F: Any,
) -> dict[str, float]:
    target_rms_tensor = target.pow(2).mean(dim=-1, keepdim=True).add(1e-12).sqrt()
    residual = target * (float(step_rms) / target_rms_tensor)
    logits = decode(hidden + residual)
    injected_ce = float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).detach().item())
    return {
        "injected_ce": injected_ce,
        "injection_ce_delta": injected_ce - float(base_ce),
        "target_rms": float(target.pow(2).mean().sqrt().detach().item()),
    }


def _target_cosine(left: Any, right: Any, *, F: Any) -> float:
    return float(
        F.cosine_similarity(
            left.reshape(-1, left.shape[-1]),
            right.reshape(-1, right.shape[-1]),
            dim=-1,
        )
        .mean()
        .detach()
        .item()
    )


def _pc_decoder_adjoint_target_alignment_probe_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "selected_next_experiment": "",
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "notify_ben": False,
            "strategic_change_level": "minor",
        }
    selected = next((row for row in rows if row.get("selected") is True), rows[0])
    return {
        "row_count": len(rows),
        "selected_next_experiment": selected.get("selected_next_experiment", ""),
        "decoder_adjoint_injection_ce_delta": _metric_float(selected.get("decoder_adjoint_injection_ce_delta")),
        "current_target_injection_ce_delta": _metric_float(selected.get("current_target_injection_ce_delta")),
        "finite_difference_injection_ce_delta": _metric_float(selected.get("finite_difference_injection_ce_delta")),
        "shuffled_target_injection_ce_delta": _metric_float(selected.get("shuffled_target_injection_ce_delta")),
        "sign_flipped_target_injection_ce_delta": _metric_float(selected.get("sign_flipped_target_injection_ce_delta")),
        "decoder_lowers_ce": selected.get("decoder_lowers_ce") is True,
        "decoder_beats_current_target": selected.get("decoder_beats_current_target") is True,
        "decoder_beats_shuffled_target": selected.get("decoder_beats_shuffled_target") is True,
        "decoder_beats_sign_flip": selected.get("decoder_beats_sign_flip") is True,
        "decoder_tracks_finite_difference": selected.get("decoder_tracks_finite_difference") is True,
        "alignment_gate_passes": selected.get("alignment_gate_passes") is True,
        "requires_gpu_now": any(row.get("requires_gpu_now") is True for row in rows),
        "promotion_allowed": any(row.get("promotion_allowed") is True for row in rows),
        "advance_to_gpu_validation": any(row.get("advance_to_gpu_validation") is True for row in rows),
        "notify_ben": any(row.get("notify_ben") is True for row in rows),
        "strategic_change_level": "major" if any(row.get("strategic_change_level") == "major" for row in rows) else "minor",
        "interpretation": (
            "Local decoder-adjoint CE target alignment probe. It allows only a minimal retrain probe when "
            "the decoder-adjoint target lowers CE and beats current, shuffled, and sign-flipped controls "
            "under matched norm."
        ),
    }


def _pc_decoder_adjoint_minimal_retrain_probe_rows(
    *,
    target_probe_rows: list[dict[str, Any]],
    arm_metrics: list[dict[str, Any]],
    train_hidden: Any,
    train_targets: Any,
    holdout_hidden: Any,
    holdout_targets: Any,
    decode: Any,
    decoder_weight: Any,
    training_steps: int,
    learning_rate: float,
    seed: int,
    torch: Any,
    nn: Any,
    F: Any,
) -> list[dict[str, Any]]:
    target_summary = _pc_decoder_adjoint_target_alignment_probe_summary(target_probe_rows)
    if target_summary.get("selected_next_experiment") != "wire_decoder_adjoint_pc_target_into_minimal_retrain_probe":
        return []

    by_arm = {str(row.get("arm", "")): row for row in arm_metrics}
    promoted = by_arm.get("promoted_contextual_topk2")
    flat = by_arm.get("pc_same_router_flat_mlp_control_topk2") or by_arm.get("flat_column_value_mlp_topk2")
    promoted_ce = _metric_float(promoted.get("holdout_ce")) if promoted else None
    flat_ce = _metric_float(flat.get("holdout_ce")) if flat else None
    vocab_size = int(decoder_weight.shape[0])
    hidden_dim = int(train_hidden.shape[-1])
    rank = max(1, hidden_dim // 4)
    target_weight = 0.25

    def current_target(hidden: Any, targets: Any) -> Any:
        return F.layer_norm(
            decoder_weight.index_select(0, targets.reshape(-1)).reshape(
                targets.shape[0],
                targets.shape[1],
                -1,
            )
            - hidden,
            (hidden.shape[-1],),
        ).detach()

    def shuffled_targets(targets: Any) -> Any:
        return targets.reshape(-1).index_select(
            0,
            torch.arange(targets.numel() - 1, -1, -1, device=targets.device),
        ).reshape_as(targets)

    train_decoder = _decoder_adjoint_hidden_ce_error(
        train_hidden,
        train_targets,
        decoder_weight,
        decode=decode,
        F=F,
        normalize=True,
    ).detach()
    holdout_decoder = _decoder_adjoint_hidden_ce_error(
        holdout_hidden,
        holdout_targets,
        decoder_weight,
        decode=decode,
        F=F,
        normalize=True,
    ).detach()
    train_shuffled = _decoder_adjoint_hidden_ce_error(
        train_hidden,
        shuffled_targets(train_targets),
        decoder_weight,
        decode=decode,
        F=F,
        normalize=True,
    ).detach()
    holdout_shuffled = _decoder_adjoint_hidden_ce_error(
        holdout_hidden,
        shuffled_targets(holdout_targets),
        decoder_weight,
        decode=decode,
        F=F,
        normalize=True,
    ).detach()
    target_specs = [
        ("decoder_adjoint_aux_target", "primary_decoder_adjoint_retrain", train_decoder, holdout_decoder, True),
        ("current_embedding_minus_hidden_aux_target", "current_pc_target_retrain_control", current_target(train_hidden, train_targets), current_target(holdout_hidden, holdout_targets), False),
        ("shuffled_decoder_adjoint_aux_target_null", "shuffled_target_retrain_null", train_shuffled, holdout_shuffled, False),
        ("sign_flipped_decoder_adjoint_aux_target_null", "sign_flipped_target_retrain_null", -train_decoder, -holdout_decoder, False),
        ("ce_only_low_rank_dense_control", "no_aux_target_ce_control", None, None, False),
    ]

    metrics: dict[str, dict[str, Any]] = {}
    for offset, (variant, _, train_target, holdout_target, _) in enumerate(target_specs):
        torch.manual_seed(seed + 1400 + offset)
        adapter = _DenseLowRankAdapter(hidden_dim, rank, nn=nn)
        optimizer = torch.optim.AdamW(list(adapter.parameters()), lr=learning_rate)
        for _ in range(training_steps):
            optimizer.zero_grad(set_to_none=True)
            adapted = adapter(train_hidden)
            logits = decode(adapted)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), train_targets.reshape(-1))
            if train_target is not None:
                residual = adapted - train_hidden
                loss = loss + target_weight * F.mse_loss(
                    F.layer_norm(residual, (hidden_dim,)),
                    train_target,
                )
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            adapted_train = adapter(train_hidden)
            train_logits = decode(adapted_train)
            train_ce = float(F.cross_entropy(train_logits.reshape(-1, vocab_size), train_targets.reshape(-1)).item())
            adapted_holdout = adapter(holdout_hidden)
            holdout_logits = decode(adapted_holdout)
            holdout_ce = float(F.cross_entropy(holdout_logits.reshape(-1, vocab_size), holdout_targets.reshape(-1)).item())
            residual = adapted_holdout - holdout_hidden
            residual_l2 = float(residual.pow(2).mean().sqrt().item())
            target_cosine = (
                _target_cosine(residual, holdout_target, F=F)
                if holdout_target is not None
                else None
            )
        metrics[variant] = {
            "train_ce": train_ce,
            "holdout_ce": holdout_ce,
            "residual_l2": residual_l2,
            "target_cosine": target_cosine,
        }

    primary = metrics["decoder_adjoint_aux_target"]
    current = metrics["current_embedding_minus_hidden_aux_target"]
    shuffled = metrics["shuffled_decoder_adjoint_aux_target_null"]
    sign_flipped = metrics["sign_flipped_decoder_adjoint_aux_target_null"]
    ce_only = metrics["ce_only_low_rank_dense_control"]
    decoder_beats_current = primary["holdout_ce"] <= current["holdout_ce"] - 0.005
    decoder_beats_shuffled = primary["holdout_ce"] <= shuffled["holdout_ce"] - 0.005
    decoder_beats_sign_flip = primary["holdout_ce"] <= sign_flipped["holdout_ce"] - 0.005
    decoder_beats_ce_only = primary["holdout_ce"] <= ce_only["holdout_ce"] + 0.005
    flat_control_ok = bool(flat_ce is None or primary["holdout_ce"] <= flat_ce + 0.01)
    promoted_signal_ok = bool(promoted_ce is None or primary["holdout_ce"] <= promoted_ce)
    retrain_gate_passes = bool(
        decoder_beats_current
        and decoder_beats_shuffled
        and decoder_beats_sign_flip
        and decoder_beats_ce_only
        and flat_control_ok
        and promoted_signal_ok
    )
    selected_next = (
        "prototype_decoder_adjoint_pc_sparse_retrain_with_flat_and_commutator_controls"
        if retrain_gate_passes
        else "close_or_redesign_decoder_adjoint_pc_retrain_path_before_gpu"
    )
    blockers = []
    if not decoder_beats_current:
        blockers.append("current_pc_target_control_not_beaten")
    if not decoder_beats_shuffled:
        blockers.append("shuffled_target_retrain_null_not_beaten")
    if not decoder_beats_sign_flip:
        blockers.append("sign_flipped_target_retrain_null_not_beaten")
    if not decoder_beats_ce_only:
        blockers.append("ce_only_dense_control_not_beaten")
    if not flat_control_ok:
        blockers.append("same_router_flat_control_still_stronger")
    if not promoted_signal_ok:
        blockers.append("no_ce_signal_vs_promoted_sparse")

    rows: list[dict[str, Any]] = []
    for variant, role, _, _, selected in target_specs:
        row = metrics[variant]
        rows.append(
            {
                "probe_name": "pc_decoder_adjoint_minimal_retrain_probe",
                "target_variant": variant,
                "probe_role": role,
                "selected": selected,
                "rank": rank,
                "training_steps": training_steps,
                "target_mse_weight": target_weight if variant != "ce_only_low_rank_dense_control" else 0.0,
                "train_ce": row["train_ce"],
                "holdout_ce": row["holdout_ce"],
                "residual_l2": row["residual_l2"],
                "target_cosine": row["target_cosine"],
                "promoted_sparse_ce": promoted_ce,
                "same_router_flat_control_ce": flat_ce,
                "holdout_ce_minus_promoted_sparse_ce": _delta_value(row["holdout_ce"], promoted_ce),
                "holdout_ce_minus_flat_control_ce": _delta_value(row["holdout_ce"], flat_ce),
                "decoder_adjoint_holdout_ce": primary["holdout_ce"],
                "current_target_holdout_ce": current["holdout_ce"],
                "shuffled_target_holdout_ce": shuffled["holdout_ce"],
                "sign_flipped_target_holdout_ce": sign_flipped["holdout_ce"],
                "ce_only_holdout_ce": ce_only["holdout_ce"],
                "decoder_beats_current_target": decoder_beats_current,
                "decoder_beats_shuffled_target": decoder_beats_shuffled,
                "decoder_beats_sign_flip": decoder_beats_sign_flip,
                "decoder_beats_ce_only_control": decoder_beats_ce_only,
                "flat_control_ok": flat_control_ok,
                "promoted_signal_ok": promoted_signal_ok,
                "retrain_gate_passes": retrain_gate_passes,
                "failure_reasons": ";".join(blockers),
                "selected_next_experiment": selected_next,
                "requires_gpu_now": False,
                "promotion_allowed": False,
                "advance_to_gpu_validation": False,
                "strategic_change_level": "minor",
                "notify_ben": False,
                "label_derived_training_only_target": variant != "ce_only_low_rank_dense_control",
                "mechanism_labels_used_for_scoring_only": True,
                "interpretation": (
                    "Minimal local retrain probe for decoder-adjoint PC target semantics. It uses label-derived "
                    "CE targets for training-time diagnosis only and blocks GPU unless the target beats matched "
                    "current, shuffled, sign-flipped, no-target, promoted sparse, and flat-control references."
                ),
            }
        )
    return rows


def _pc_decoder_adjoint_minimal_retrain_probe_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "selected_next_experiment": "",
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "notify_ben": False,
            "strategic_change_level": "minor",
        }
    selected = next((row for row in rows if row.get("selected") is True), rows[0])
    return {
        "row_count": len(rows),
        "selected_next_experiment": selected.get("selected_next_experiment", ""),
        "decoder_adjoint_holdout_ce": _metric_float(selected.get("decoder_adjoint_holdout_ce")),
        "current_target_holdout_ce": _metric_float(selected.get("current_target_holdout_ce")),
        "shuffled_target_holdout_ce": _metric_float(selected.get("shuffled_target_holdout_ce")),
        "sign_flipped_target_holdout_ce": _metric_float(selected.get("sign_flipped_target_holdout_ce")),
        "ce_only_holdout_ce": _metric_float(selected.get("ce_only_holdout_ce")),
        "holdout_ce_minus_promoted_sparse_ce": _metric_float(selected.get("holdout_ce_minus_promoted_sparse_ce")),
        "holdout_ce_minus_flat_control_ce": _metric_float(selected.get("holdout_ce_minus_flat_control_ce")),
        "decoder_beats_current_target": selected.get("decoder_beats_current_target") is True,
        "decoder_beats_shuffled_target": selected.get("decoder_beats_shuffled_target") is True,
        "decoder_beats_sign_flip": selected.get("decoder_beats_sign_flip") is True,
        "decoder_beats_ce_only_control": selected.get("decoder_beats_ce_only_control") is True,
        "flat_control_ok": selected.get("flat_control_ok") is True,
        "promoted_signal_ok": selected.get("promoted_signal_ok") is True,
        "retrain_gate_passes": selected.get("retrain_gate_passes") is True,
        "failure_reasons": selected.get("failure_reasons", ""),
        "requires_gpu_now": any(row.get("requires_gpu_now") is True for row in rows),
        "promotion_allowed": any(row.get("promotion_allowed") is True for row in rows),
        "advance_to_gpu_validation": any(row.get("advance_to_gpu_validation") is True for row in rows),
        "notify_ben": any(row.get("notify_ben") is True for row in rows),
        "strategic_change_level": "major" if any(row.get("strategic_change_level") == "major" for row in rows) else "minor",
        "interpretation": (
            "Fail-closed local minimal retrain probe for decoder-adjoint PC target semantics. "
            "It does not request GPU or promotion."
        ),
    }


def _pc_decoder_adjoint_closeout_rows(
    *,
    target_probe_rows: list[dict[str, Any]],
    retrain_probe_rows: list[dict[str, Any]],
    arm_metrics: list[dict[str, Any]],
    residual_budget_rows: list[dict[str, Any]],
    commutator_rows: list[dict[str, Any]],
    forgetting_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    retrain_summary = _pc_decoder_adjoint_minimal_retrain_probe_summary(retrain_probe_rows)
    if not retrain_probe_rows:
        return []

    target_summary = _pc_decoder_adjoint_target_alignment_probe_summary(target_probe_rows)
    retrain_by_variant = {str(row.get("target_variant", "")): row for row in retrain_probe_rows}
    primary = retrain_by_variant.get("decoder_adjoint_aux_target", retrain_probe_rows[0])
    by_arm = {str(row.get("arm", "")): row for row in arm_metrics}
    promoted = by_arm.get("promoted_contextual_topk2")
    flat = by_arm.get("pc_same_router_flat_mlp_control_topk2") or by_arm.get("flat_column_value_mlp_topk2")
    pc_arm_name = "pc_core_periphery_residual_inference_topk2"
    pc_budget = next((row for row in residual_budget_rows if row.get("arm") == pc_arm_name), {})
    promoted_budget = next((row for row in residual_budget_rows if row.get("arm") == "promoted_contextual_topk2"), {})
    pc_comm = next((row for row in commutator_rows if row.get("arm") == pc_arm_name), {})
    promoted_comm = next((row for row in commutator_rows if row.get("arm") == "promoted_contextual_topk2"), {})
    pc_forgetting = next((row for row in forgetting_rows if row.get("arm") == pc_arm_name), {})
    promoted_forgetting = next((row for row in forgetting_rows if row.get("arm") == "promoted_contextual_topk2"), {})

    primary_ce = _metric_float(primary.get("holdout_ce"))
    promoted_ce = _metric_float(primary.get("promoted_sparse_ce"))
    if promoted_ce is None and promoted is not None:
        promoted_ce = _metric_float(promoted.get("holdout_ce"))
    flat_ce = _metric_float(primary.get("same_router_flat_control_ce"))
    if flat_ce is None and flat is not None:
        flat_ce = _metric_float(flat.get("holdout_ce"))
    no_aux_ce = _metric_float(primary.get("ce_only_holdout_ce"))
    shuffled_ce = _metric_float(primary.get("shuffled_target_holdout_ce"))
    sign_flip_ce = _metric_float(primary.get("sign_flipped_target_holdout_ce"))
    current_ce = _metric_float(primary.get("current_target_holdout_ce"))
    residual_l2 = _metric_float(primary.get("residual_l2"))
    promoted_residual_l2 = _metric_float(promoted_budget.get("residual_l2"))
    residual_norm_ratio = _safe_ratio(residual_l2, promoted_residual_l2)
    commutator_ratio = _safe_ratio(
        pc_comm.get("mean_abs_order_delta"),
        promoted_comm.get("mean_abs_order_delta"),
    )
    churn_ratio = _safe_ratio(
        pc_forgetting.get("functional_churn_fraction"),
        promoted_forgetting.get("functional_churn_fraction"),
    )

    beats_current = primary.get("decoder_beats_current_target") is True
    beats_shuffled = primary.get("decoder_beats_shuffled_target") is True
    beats_sign_flip = primary.get("decoder_beats_sign_flip") is True
    beats_no_aux = primary.get("decoder_beats_ce_only_control") is True
    flat_control_ok = primary.get("flat_control_ok") is True
    promoted_signal_ok = primary.get("promoted_signal_ok") is True
    norm_budget_ok = bool(residual_norm_ratio is not None and residual_norm_ratio <= 1.25)
    commutator_budget_ok = bool(commutator_ratio is not None and commutator_ratio <= 1.0)
    functional_churn_budget_ok = bool(churn_ratio is not None and churn_ratio <= 1.0)
    closeout_blocks = not (
        target_summary.get("alignment_gate_passes") is True
        and retrain_summary.get("retrain_gate_passes") is True
        and beats_current
        and beats_shuffled
        and beats_sign_flip
        and beats_no_aux
        and flat_control_ok
        and promoted_signal_ok
        and norm_budget_ok
        and commutator_budget_ok
        and functional_churn_budget_ok
    )

    blockers: list[str] = []
    if target_summary.get("alignment_gate_passes") is not True:
        blockers.append("target_alignment_gate_failed")
    if not beats_current:
        blockers.append("current_pc_target_control_not_beaten")
    if not beats_shuffled:
        blockers.append("shuffled_target_retrain_null_not_beaten")
    if not beats_sign_flip:
        blockers.append("sign_flipped_target_retrain_null_not_beaten")
    if not beats_no_aux:
        blockers.append("no_aux_ce_control_not_beaten")
    if not flat_control_ok:
        blockers.append("same_router_flat_control_not_beaten")
    if not promoted_signal_ok:
        blockers.append("promoted_sparse_reference_not_beaten")
    if not norm_budget_ok:
        blockers.append("residual_norm_budget_failed_or_missing")
    if not commutator_budget_ok:
        blockers.append("finite_update_commutator_budget_failed_or_missing")
    if not functional_churn_budget_ok:
        blockers.append("functional_churn_budget_failed_or_missing")

    closeout_status = (
        "closed_one_site_decoder_adjoint_retrain_path"
        if closeout_blocks
        else "unexpected_pass_requires_amortized_pc_redesign_review"
    )
    selected_next = (
        "prototype_tiny_label_free_amortized_multi_site_pc_pregate_with_flat_dense_shuffled_signflip_position_nulls"
        if closeout_blocks
        else "request_strategy_review_before_any_gpu_validation"
    )
    return [
        {
            "closeout_name": "pc_decoder_adjoint_closeout",
            "closeout_status": closeout_status,
            "target_alignment_gate_passes": target_summary.get("alignment_gate_passes") is True,
            "minimal_retrain_gate_passes": retrain_summary.get("retrain_gate_passes") is True,
            "decoder_adjoint_holdout_ce": primary_ce,
            "current_target_holdout_ce": current_ce,
            "shuffled_target_holdout_ce": shuffled_ce,
            "sign_flipped_target_holdout_ce": sign_flip_ce,
            "no_aux_ce_control_holdout_ce": no_aux_ce,
            "promoted_sparse_ce": promoted_ce,
            "same_router_flat_control_ce": flat_ce,
            "decoder_minus_current_target_ce": _delta_value(primary_ce, current_ce),
            "decoder_minus_shuffled_target_ce": _delta_value(primary_ce, shuffled_ce),
            "decoder_minus_sign_flipped_target_ce": _delta_value(primary_ce, sign_flip_ce),
            "decoder_minus_no_aux_control_ce": _delta_value(primary_ce, no_aux_ce),
            "decoder_minus_promoted_sparse_ce": _delta_value(primary_ce, promoted_ce),
            "decoder_minus_same_router_flat_control_ce": _delta_value(primary_ce, flat_ce),
            "decoder_beats_current_target": beats_current,
            "decoder_beats_shuffled_target": beats_shuffled,
            "decoder_beats_sign_flip": beats_sign_flip,
            "decoder_beats_no_aux_ce_control": beats_no_aux,
            "flat_control_ok": flat_control_ok,
            "promoted_signal_ok": promoted_signal_ok,
            "residual_l2": residual_l2,
            "promoted_sparse_residual_l2": promoted_residual_l2,
            "residual_norm_ratio_vs_promoted_sparse": residual_norm_ratio,
            "finite_update_commutator_ratio_vs_promoted_sparse": commutator_ratio,
            "functional_churn_ratio_vs_promoted_sparse": churn_ratio,
            "norm_budget_ok": norm_budget_ok,
            "commutator_budget_ok": commutator_budget_ok,
            "functional_churn_budget_ok": functional_churn_budget_ok,
            "failure_reasons": ";".join(blockers),
            "selected_next_experiment": selected_next,
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "advance_to_gpu_validation": False,
            "label_derived_training_only_target": True,
            "one_site_decoder_adjoint_path_closed": closeout_blocks,
            "amortized_label_free_pc_pregate_allowed": closeout_blocks,
            "strategic_change_level": "major",
            "notify_ben": True,
            "interpretation": (
                "Fail-closed row-level closeout of the one-site decoder-adjoint PC retrain path. "
                "It records target/null/no-aux/flat/promoted sparse and interference-budget comparisons, "
                "blocks GPU and promotion, and redirects only to a tiny label-free amortized multi-site PC pregate."
            ),
        }
    ]


def _pc_decoder_adjoint_closeout_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "closeout_status": "",
            "selected_next_experiment": "",
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "notify_ben": False,
            "strategic_change_level": "minor",
        }
    row = rows[0]
    return {
        "row_count": len(rows),
        "closeout_status": row.get("closeout_status", ""),
        "target_alignment_gate_passes": row.get("target_alignment_gate_passes") is True,
        "minimal_retrain_gate_passes": row.get("minimal_retrain_gate_passes") is True,
        "decoder_adjoint_holdout_ce": _metric_float(row.get("decoder_adjoint_holdout_ce")),
        "current_target_holdout_ce": _metric_float(row.get("current_target_holdout_ce")),
        "shuffled_target_holdout_ce": _metric_float(row.get("shuffled_target_holdout_ce")),
        "sign_flipped_target_holdout_ce": _metric_float(row.get("sign_flipped_target_holdout_ce")),
        "no_aux_ce_control_holdout_ce": _metric_float(row.get("no_aux_ce_control_holdout_ce")),
        "promoted_sparse_ce": _metric_float(row.get("promoted_sparse_ce")),
        "same_router_flat_control_ce": _metric_float(row.get("same_router_flat_control_ce")),
        "decoder_minus_current_target_ce": _metric_float(row.get("decoder_minus_current_target_ce")),
        "decoder_minus_shuffled_target_ce": _metric_float(row.get("decoder_minus_shuffled_target_ce")),
        "decoder_minus_sign_flipped_target_ce": _metric_float(row.get("decoder_minus_sign_flipped_target_ce")),
        "decoder_minus_no_aux_control_ce": _metric_float(row.get("decoder_minus_no_aux_control_ce")),
        "decoder_minus_promoted_sparse_ce": _metric_float(row.get("decoder_minus_promoted_sparse_ce")),
        "decoder_minus_same_router_flat_control_ce": _metric_float(row.get("decoder_minus_same_router_flat_control_ce")),
        "norm_budget_ok": row.get("norm_budget_ok") is True,
        "commutator_budget_ok": row.get("commutator_budget_ok") is True,
        "functional_churn_budget_ok": row.get("functional_churn_budget_ok") is True,
        "failure_reasons": row.get("failure_reasons", ""),
        "selected_next_experiment": row.get("selected_next_experiment", ""),
        "one_site_decoder_adjoint_path_closed": row.get("one_site_decoder_adjoint_path_closed") is True,
        "amortized_label_free_pc_pregate_allowed": row.get("amortized_label_free_pc_pregate_allowed") is True,
        "requires_gpu_now": any(closeout.get("requires_gpu_now") is True for closeout in rows),
        "promotion_allowed": any(closeout.get("promotion_allowed") is True for closeout in rows),
        "advance_to_gpu_validation": any(closeout.get("advance_to_gpu_validation") is True for closeout in rows),
        "notify_ben": any(closeout.get("notify_ben") is True for closeout in rows),
        "strategic_change_level": "major" if any(closeout.get("strategic_change_level") == "major" for closeout in rows) else "minor",
        "interpretation": (
            "One-site decoder-adjoint PC retrain path closeout with explicit null/control and interference-budget fields. "
            "GPU remains blocked."
        ),
    }


def _pc_amortized_error_pregate_design_rows(
    *,
    closeout_rows: list[dict[str, Any]],
    arm_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    closeout_summary = _pc_decoder_adjoint_closeout_summary(closeout_rows)
    if closeout_summary.get("amortized_label_free_pc_pregate_allowed") is not True:
        return []

    by_arm = {str(row.get("arm", "")): row for row in arm_metrics}
    promoted = by_arm.get("promoted_contextual_topk2", {})
    flat = by_arm.get("pc_same_router_flat_mlp_control_topk2", {})
    dense = by_arm.get("dense_rank_norm_matched", {})
    promoted_ce = _metric_float(promoted.get("holdout_ce"))
    flat_ce = _metric_float(flat.get("holdout_ce"))
    dense_ce = _metric_float(dense.get("holdout_ce"))

    base = {
        "design_name": "pc_amortized_error_pregate",
        "source_closeout_status": closeout_summary.get("closeout_status", ""),
        "source_one_site_decoder_adjoint_path_closed": closeout_summary.get(
            "one_site_decoder_adjoint_path_closed"
        )
        is True,
        "label_free_eval_required": True,
        "mechanism_labels_used_for_scoring_only": True,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "implemented_in_current_packet": False,
        "strategic_change_level": "major",
        "notify_ben": True,
    }
    rows = [
        {
            **base,
            "design_role": "primary_label_free_amortized_multi_site_pc_micrograph",
            "selected": True,
            "error_predictor": "causal_feature_decoder_adjoint_hidden_error_predictor",
            "training_target": "decoder_adjoint_hidden_ce_error_training_only",
            "deploy_time_labels_required": False,
            "residual_update_sites": "2-4",
            "inference_steps": 2,
            "proximal_norm_clamp_required": True,
            "anchor_kl_required": True,
            "promoted_sparse_reference_ce": promoted_ce,
            "same_router_flat_control_ce": flat_ce,
            "dense_rank_norm_control_ce": dense_ce,
            "required_signal_gate": "ce_guardrail_plus_commutator_churn_fingerprint_gain_vs_controls",
            "required_controls": "same_router_flat;dense_rank_norm;shuffled_error_target;sign_flipped_error_target;token_position_error_predictor;promoted_sparse_reference",
            "selected_next_experiment": "implement_local_tiny_label_free_amortized_multi_site_pc_pregate",
            "interpretation": (
                "Design-only scaffold after decoder-adjoint closeout. It permits only a tiny label-free "
                "amortized multi-site PC pregate with explicit flat/dense/shuffled/sign-flipped/token-position nulls."
            ),
        },
        {
            **base,
            "design_role": "same_router_flat_control",
            "selected": False,
            "error_predictor": "same_router_flat_mlp_value_control",
            "training_target": "supervised_ce_or_matched_error_aux_control",
            "deploy_time_labels_required": False,
            "residual_update_sites": "matched",
            "inference_steps": 0,
            "proximal_norm_clamp_required": True,
            "anchor_kl_required": True,
            "selected_next_experiment": "",
        },
        {
            **base,
            "design_role": "dense_rank_norm_control",
            "selected": False,
            "error_predictor": "rank_norm_matched_dense_residual_control",
            "training_target": "matched_supervised_ce_control",
            "deploy_time_labels_required": False,
            "residual_update_sites": "dense",
            "inference_steps": 0,
            "proximal_norm_clamp_required": True,
            "anchor_kl_required": True,
            "selected_next_experiment": "",
        },
        {
            **base,
            "design_role": "shuffled_error_target_null",
            "selected": False,
            "error_predictor": "causal_feature_error_predictor_shuffled_target",
            "training_target": "shuffled_decoder_adjoint_hidden_error",
            "deploy_time_labels_required": False,
            "residual_update_sites": "2-4",
            "inference_steps": 2,
            "proximal_norm_clamp_required": True,
            "anchor_kl_required": True,
            "selected_next_experiment": "",
        },
        {
            **base,
            "design_role": "sign_flipped_error_target_null",
            "selected": False,
            "error_predictor": "causal_feature_error_predictor_sign_flipped_target",
            "training_target": "sign_flipped_decoder_adjoint_hidden_error",
            "deploy_time_labels_required": False,
            "residual_update_sites": "2-4",
            "inference_steps": 2,
            "proximal_norm_clamp_required": True,
            "anchor_kl_required": True,
            "selected_next_experiment": "",
        },
        {
            **base,
            "design_role": "token_position_error_predictor_null",
            "selected": False,
            "error_predictor": "token_position_only_error_predictor",
            "training_target": "decoder_adjoint_hidden_error_training_only",
            "deploy_time_labels_required": False,
            "residual_update_sites": "2-4",
            "inference_steps": 2,
            "proximal_norm_clamp_required": True,
            "anchor_kl_required": True,
            "selected_next_experiment": "",
        },
    ]
    return rows


def _pc_amortized_error_pregate_design_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "selected_next_experiment": "",
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "notify_ben": False,
            "strategic_change_level": "minor",
        }
    selected = next((row for row in rows if row.get("selected") is True), rows[0])
    return {
        "row_count": len(rows),
        "selected_design_role": selected.get("design_role", ""),
        "source_closeout_status": selected.get("source_closeout_status", ""),
        "source_one_site_decoder_adjoint_path_closed": selected.get(
            "source_one_site_decoder_adjoint_path_closed"
        )
        is True,
        "label_free_eval_required": selected.get("label_free_eval_required") is True,
        "deploy_time_labels_required": selected.get("deploy_time_labels_required") is True,
        "implemented_in_current_packet": selected.get("implemented_in_current_packet") is True,
        "required_controls": selected.get("required_controls", ""),
        "selected_next_experiment": selected.get("selected_next_experiment", ""),
        "requires_gpu_now": any(row.get("requires_gpu_now") is True for row in rows),
        "promotion_allowed": any(row.get("promotion_allowed") is True for row in rows),
        "advance_to_gpu_validation": any(row.get("advance_to_gpu_validation") is True for row in rows),
        "notify_ben": any(row.get("notify_ben") is True for row in rows),
        "strategic_change_level": "major" if any(row.get("strategic_change_level") == "major" for row in rows) else "minor",
        "interpretation": (
            "Design-only local scaffold for the next label-free amortized multi-site PC pregate. "
            "It is not evidence and keeps GPU blocked until implemented local metrics beat controls."
        ),
    }


def _pc_amortized_error_pregate_rows(
    *,
    design_rows: list[dict[str, Any]],
    arm_metrics: list[dict[str, Any]],
    residual_budget_rows: list[dict[str, Any]],
    commutator_rows: list[dict[str, Any]],
    forgetting_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    design_summary = _pc_amortized_error_pregate_design_summary(design_rows)
    if design_summary.get("selected_next_experiment") != "implement_local_tiny_label_free_amortized_multi_site_pc_pregate":
        return []
    by_arm = {str(row.get("arm", "")): row for row in arm_metrics}
    budget_by_arm = {str(row.get("arm", "")): row for row in residual_budget_rows}
    primary = by_arm.get("pc_amortized_multisite_error_topk2")
    reference = by_arm.get("promoted_contextual_topk2")
    flat = by_arm.get("pc_same_router_flat_mlp_control_topk2")
    dense = by_arm.get("dense_rank_norm_matched")
    shuffled = by_arm.get("pc_amortized_shuffled_error_null_topk2")
    sign_flipped = by_arm.get("pc_amortized_sign_flipped_error_null_topk2")
    token_position = by_arm.get("pc_amortized_token_position_error_null_topk2")
    stored = _best_ce_row(
        [row for row in arm_metrics if row.get("control_budget_role") == "stored_parameter_matched_dense_mlp_upper_bound"]
    )
    if primary is None or reference is None:
        return []

    reference_ce = _metric_float(reference.get("holdout_ce"))
    primary_ce = _metric_float(primary.get("holdout_ce"))
    flat_ce = _metric_float(flat.get("holdout_ce")) if flat else None
    dense_ce = _metric_float(dense.get("holdout_ce")) if dense else None
    shuffled_ce = _metric_float(shuffled.get("holdout_ce")) if shuffled else None
    sign_flipped_ce = _metric_float(sign_flipped.get("holdout_ce")) if sign_flipped else None
    token_position_ce = _metric_float(token_position.get("holdout_ce")) if token_position else None
    stored_ce = _metric_float(stored.get("holdout_ce")) if stored else None
    stored_gap = _positive_gap(reference_ce, stored_ce)
    ce_gain = _delta_value(reference_ce, primary_ce)
    stored_gap_closed = _safe_ratio(ce_gain, stored_gap)
    flat_margin = _delta_value(primary_ce, flat_ce)
    dense_margin = _delta_value(primary_ce, dense_ce)
    shuffled_margin = _delta_value(primary_ce, shuffled_ce)
    sign_flipped_margin = _delta_value(primary_ce, sign_flipped_ce)
    token_position_margin = _delta_value(primary_ce, token_position_ce)
    reference_norm = _metric_float(reference.get("residual_l2"))
    primary_norm = _metric_float(primary.get("residual_l2"))
    reference_commutator = _mean_metric(commutator_rows, ["promoted_contextual_topk2"], "finite_update_commutator_l2")
    primary_commutator = _mean_metric(commutator_rows, ["pc_amortized_multisite_error_topk2"], "finite_update_commutator_l2")
    reference_churn = _mean_abs_metric(forgetting_rows, ["promoted_contextual_topk2"], "functional_churn")
    primary_churn = _mean_abs_metric(forgetting_rows, ["pc_amortized_multisite_error_topk2"], "functional_churn")

    signal_ok = bool(
        (ce_gain is not None and ce_gain >= 0.02)
        or (stored_gap_closed is not None and stored_gap_closed >= 0.05)
    )
    flat_control_ok = bool(flat_margin is None or flat_margin <= 0.01)
    dense_control_ok = bool(dense_margin is None or dense_margin <= 0.01)
    shuffled_ok = bool(shuffled_margin is not None and shuffled_margin <= -0.005)
    sign_flipped_ok = bool(sign_flipped_margin is not None and sign_flipped_margin <= -0.005)
    token_position_ok = bool(token_position_margin is not None and token_position_margin <= -0.005)
    norm_ok = bool(
        primary_norm is not None
        and reference_norm is not None
        and primary_norm <= reference_norm * 1.05
    )
    commutator_ok = bool(
        primary_commutator is not None
        and reference_commutator is not None
        and primary_commutator <= reference_commutator * 1.10
    )
    churn_ok = bool(
        primary_churn is not None
        and reference_churn is not None
        and primary_churn <= reference_churn * 1.10
    )
    pregate_passes = bool(
        signal_ok
        and flat_control_ok
        and dense_control_ok
        and shuffled_ok
        and sign_flipped_ok
        and token_position_ok
        and norm_ok
        and commutator_ok
        and churn_ok
    )

    failures = []
    if not signal_ok:
        failures.append("no_ce_or_stored_gap_signal")
    if not flat_control_ok:
        failures.append("same_router_flat_control_stronger")
    if not dense_control_ok:
        failures.append("dense_rank_norm_control_stronger")
    if not shuffled_ok:
        failures.append("shuffled_error_target_null_competitive")
    if not sign_flipped_ok:
        failures.append("sign_flipped_error_target_null_competitive")
    if not token_position_ok:
        failures.append("token_position_error_predictor_null_competitive")
    if not norm_ok:
        failures.append("residual_norm_budget_failed")
    if not commutator_ok:
        failures.append("finite_update_commutator_budget_failed")
    if not churn_ok:
        failures.append("functional_churn_budget_failed")

    rows: list[dict[str, Any]] = []
    for role, arm, family, selected in (
        ("primary_label_free_amortized_multisite_pc", "pc_amortized_multisite_error_topk2", "pc_amortized_sparse", True),
        ("same_router_flat_control", "pc_same_router_flat_mlp_control_topk2", "flat_same_router_control", False),
        ("dense_rank_norm_control", "dense_rank_norm_matched", "dense_rank_norm_control", False),
        ("shuffled_error_target_null", "pc_amortized_shuffled_error_null_topk2", "target_null", False),
        ("sign_flipped_error_target_null", "pc_amortized_sign_flipped_error_null_topk2", "target_null", False),
        ("token_position_error_predictor_null", "pc_amortized_token_position_error_null_topk2", "target_null", False),
        ("no_pc_promoted_sparse_reference", "promoted_contextual_topk2", "no_pc_reference", False),
    ):
        source = by_arm.get(arm, {})
        source_ce = _metric_float(source.get("holdout_ce"))
        source_commutator = _mean_metric(commutator_rows, [arm], "finite_update_commutator_l2")
        source_churn = _mean_abs_metric(forgetting_rows, [arm], "functional_churn")
        rows.append(
            {
                "pregate_name": "pc_amortized_error_pregate",
                "pregate_role": role,
                "arm": arm,
                "control_family": family,
                "selected": selected,
                "implemented_in_current_packet": bool(source),
                "label_free_eval": True,
                "deploy_time_labels_required": False,
                "training_target": source.get("pc_error_target_mode", ""),
                "residual_update_sites": "2",
                "proximal_norm_clamp": True,
                "promoted_sparse_reference_ce": reference_ce,
                "source_holdout_ce": source_ce,
                "source_ce_minus_primary_ce": _delta_value(source_ce, primary_ce),
                "source_ce_minus_reference_sparse_ce": _delta_value(source_ce, reference_ce),
                "primary_holdout_ce": primary_ce,
                "primary_ce_gain_vs_reference_sparse": ce_gain,
                "stored_gap_closed_fraction": stored_gap_closed,
                "primary_ce_minus_flat_control_ce": flat_margin,
                "primary_ce_minus_dense_rank_norm_control_ce": dense_margin,
                "primary_ce_minus_shuffled_error_null_ce": shuffled_margin,
                "primary_ce_minus_sign_flipped_error_null_ce": sign_flipped_margin,
                "primary_ce_minus_token_position_null_ce": token_position_margin,
                "source_residual_l2": _metric_float(source.get("residual_l2")),
                "source_flop_proxy_per_token": _metric_float(budget_by_arm.get(arm, {}).get("flop_proxy_per_token")),
                "source_mean_commutator_l2": source_commutator,
                "source_mean_abs_functional_churn": source_churn,
                "primary_commutator_ratio_vs_reference_sparse": _safe_ratio(primary_commutator, reference_commutator),
                "primary_churn_ratio_vs_reference_sparse": _safe_ratio(primary_churn, reference_churn),
                "primary_pc_inference_steps": _metric_float(primary.get("pc_inference_steps")),
                "primary_pc_error_prediction_weight": _metric_float(primary.get("pc_error_prediction_weight")),
                "signal_gate_ok": signal_ok,
                "flat_control_ok": flat_control_ok,
                "dense_control_ok": dense_control_ok,
                "shuffled_target_ok": shuffled_ok,
                "sign_flipped_target_ok": sign_flipped_ok,
                "token_position_null_ok": token_position_ok,
                "norm_budget_ok": norm_ok,
                "commutator_budget_ok": commutator_ok,
                "functional_churn_budget_ok": churn_ok,
                "pregate_passes": pregate_passes,
                "failure_reasons": ";".join(failures),
                "selected_next_experiment": (
                    "repeat_label_free_amortized_multisite_pc_on_adjacent_seed"
                    if pregate_passes
                    else "close_or_redesign_label_free_amortized_multisite_pc_locally"
                ),
                "requires_gpu_now": False,
                "promotion_allowed": False,
                "advance_to_gpu_validation": False,
                "strategic_change_level": "minor",
                "notify_ben": False,
                "mechanism_labels_used_for_scoring_only": True,
                "interpretation": (
                    "Local CPU pregate for the GPT-5.5-Pro-recommended label-free amortized multi-site PC branch. "
                    "Training may use decoder-adjoint targets on train data, but holdout inference is label-free and "
                    "the row remains blocked by flat/dense/null and interference gates before any GPU validation."
                ),
            }
        )
    return rows


def _pc_amortized_error_pregate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "selected_next_experiment": "",
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "notify_ben": False,
            "strategic_change_level": "minor",
        }
    selected = next((row for row in rows if row.get("selected") is True), rows[0])
    return {
        "row_count": len(rows),
        "primary_arm": selected.get("arm", ""),
        "primary_holdout_ce": _metric_float(selected.get("primary_holdout_ce")),
        "primary_ce_gain_vs_reference_sparse": _metric_float(selected.get("primary_ce_gain_vs_reference_sparse")),
        "stored_gap_closed_fraction": _metric_float(selected.get("stored_gap_closed_fraction")),
        "flat_control_ok": selected.get("flat_control_ok") is True,
        "dense_control_ok": selected.get("dense_control_ok") is True,
        "shuffled_target_ok": selected.get("shuffled_target_ok") is True,
        "sign_flipped_target_ok": selected.get("sign_flipped_target_ok") is True,
        "token_position_null_ok": selected.get("token_position_null_ok") is True,
        "norm_budget_ok": selected.get("norm_budget_ok") is True,
        "commutator_budget_ok": selected.get("commutator_budget_ok") is True,
        "functional_churn_budget_ok": selected.get("functional_churn_budget_ok") is True,
        "pregate_passes": selected.get("pregate_passes") is True,
        "failure_reasons": selected.get("failure_reasons", ""),
        "selected_next_experiment": selected.get("selected_next_experiment", ""),
        "requires_gpu_now": any(row.get("requires_gpu_now") is True for row in rows),
        "promotion_allowed": any(row.get("promotion_allowed") is True for row in rows),
        "advance_to_gpu_validation": any(row.get("advance_to_gpu_validation") is True for row in rows),
        "notify_ben": any(row.get("notify_ben") is True for row in rows),
        "strategic_change_level": "major" if any(row.get("strategic_change_level") == "major" for row in rows) else "minor",
        "interpretation": (
            "Implemented local label-free amortized multi-site PC pregate with flat, dense, shuffled, "
            "sign-flipped, token-position, and no-PC controls. GPU remains blocked unless the local gate passes."
        ),
    }


def _pc_amortized_error_pregate_closeout_rows(
    *,
    pregate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary = _pc_amortized_error_pregate_summary(pregate_rows)
    if summary.get("row_count", 0) <= 0:
        return []
    selected = next((row for row in pregate_rows if row.get("selected") is True), pregate_rows[0])
    pregate_passes = summary.get("pregate_passes") is True
    nulls_all_clear = bool(
        summary.get("shuffled_target_ok") is True
        and summary.get("sign_flipped_target_ok") is True
        and summary.get("token_position_null_ok") is True
    )
    controls_clear = bool(
        summary.get("flat_control_ok") is True
        and summary.get("dense_control_ok") is True
    )
    interference_clear = bool(
        summary.get("norm_budget_ok") is True
        and summary.get("commutator_budget_ok") is True
        and summary.get("functional_churn_budget_ok") is True
    )
    close_current_path = not bool(
        pregate_passes
        and nulls_all_clear
        and controls_clear
        and interference_clear
    )
    closeout_status = (
        "repeat_before_gpu_validation"
        if not close_current_path
        else "closed_current_label_free_amortized_pc_target_path"
    )
    selected_next_experiment = (
        "repeat_label_free_amortized_multisite_pc_on_adjacent_seed"
        if not close_current_path
        else "return_to_non_pc_sparse_value_or_low_churn_dense_control_branch"
    )
    failure_reasons = str(summary.get("failure_reasons", ""))
    return [
        {
            "closeout_name": "pc_amortized_error_pregate_closeout",
            "closeout_status": closeout_status,
            "source_pregate_name": selected.get("pregate_name", ""),
            "source_primary_arm": summary.get("primary_arm", ""),
            "source_pregate_passes": pregate_passes,
            "source_primary_holdout_ce": summary.get("primary_holdout_ce"),
            "source_primary_ce_gain_vs_reference_sparse": summary.get("primary_ce_gain_vs_reference_sparse"),
            "source_stored_gap_closed_fraction": summary.get("stored_gap_closed_fraction"),
            "source_failure_reasons": failure_reasons,
            "signal_gate_ok": bool(
                "no_ce_or_stored_gap_signal" not in failure_reasons
                and summary.get("primary_ce_gain_vs_reference_sparse") is not None
            ),
            "flat_control_ok": summary.get("flat_control_ok") is True,
            "dense_control_ok": summary.get("dense_control_ok") is True,
            "shuffled_target_ok": summary.get("shuffled_target_ok") is True,
            "sign_flipped_target_ok": summary.get("sign_flipped_target_ok") is True,
            "token_position_null_ok": summary.get("token_position_null_ok") is True,
            "all_target_nulls_clear": nulls_all_clear,
            "flat_dense_controls_clear": controls_clear,
            "interference_budgets_clear": interference_clear,
            "current_error_target_path_closed": close_current_path,
            "redesign_error_target_allowed": False,
            "branch_reopen_requires_new_causal_signal": True,
            "selected_next_experiment": selected_next_experiment,
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "advance_to_gpu_validation": False,
            "strategic_change_level": "minor",
            "notify_ben": False,
            "mechanism_labels_used_for_scoring_only": True,
            "interpretation": (
                "Fail-closed closeout for the local label-free amortized multi-site PC pregate. "
                "The current decoder-adjoint-derived amortized error target is closed unless it beats "
                "flat/dense controls, shuffled/sign-flipped/token-position nulls, and interference budgets. "
                "No GPU validation or promotion is allowed from this packet."
            ),
        }
    ]


def _pc_amortized_error_pregate_closeout_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "closeout_status": "",
            "selected_next_experiment": "",
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "notify_ben": False,
            "strategic_change_level": "minor",
        }
    row = rows[0]
    return {
        "row_count": len(rows),
        "closeout_status": row.get("closeout_status", ""),
        "source_primary_arm": row.get("source_primary_arm", ""),
        "source_pregate_passes": row.get("source_pregate_passes") is True,
        "current_error_target_path_closed": row.get("current_error_target_path_closed") is True,
        "redesign_error_target_allowed": row.get("redesign_error_target_allowed") is True,
        "branch_reopen_requires_new_causal_signal": row.get("branch_reopen_requires_new_causal_signal") is True,
        "all_target_nulls_clear": row.get("all_target_nulls_clear") is True,
        "flat_dense_controls_clear": row.get("flat_dense_controls_clear") is True,
        "interference_budgets_clear": row.get("interference_budgets_clear") is True,
        "selected_next_experiment": row.get("selected_next_experiment", ""),
        "requires_gpu_now": any(closeout.get("requires_gpu_now") is True for closeout in rows),
        "promotion_allowed": any(closeout.get("promotion_allowed") is True for closeout in rows),
        "advance_to_gpu_validation": any(closeout.get("advance_to_gpu_validation") is True for closeout in rows),
        "notify_ben": any(closeout.get("notify_ben") is True for closeout in rows),
        "strategic_change_level": "major" if any(closeout.get("strategic_change_level") == "major" for closeout in rows) else "minor",
        "interpretation": (
            "Closeout row for the failed amortized PC pregate. The branch stays local and closed unless "
            "new causal signal justifies redesigning the error target."
        ),
    }


def _transformer_acsr_design_rows(
    *,
    pc_closeout_rows: list[dict[str, Any]],
    soft_design_rows: list[dict[str, Any]],
    arm_metrics: list[dict[str, Any]],
    residual_budget_rows: list[dict[str, Any]],
    commutator_rows: list[dict[str, Any]],
    forgetting_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pc_closeout = _pc_amortized_error_pregate_closeout_summary(pc_closeout_rows)
    soft_design = _soft_mixture_low_churn_dense_modular_design_summary(soft_design_rows)
    if pc_closeout.get("current_error_target_path_closed") is not True:
        return []
    by_arm = {str(row.get("arm", "")): row for row in arm_metrics}
    budget_by_arm = {str(row.get("arm", "")): row for row in residual_budget_rows}
    promoted = by_arm.get("promoted_contextual_topk2", {})
    token_position = by_arm.get("token_position_router_topk2", {})
    linear = by_arm.get("intervention_trained_sparse_topk2", {})
    random_support = by_arm.get("random_support_topk2", {})
    dense = by_arm.get("dense_rank_norm_matched", {})
    promoted_ce = _metric_float(promoted.get("holdout_ce"))
    token_position_ce = _metric_float(token_position.get("holdout_ce"))
    linear_ce = _metric_float(linear.get("holdout_ce"))
    random_ce = _metric_float(random_support.get("holdout_ce"))
    dense_ce = _metric_float(dense.get("holdout_ce"))
    promoted_commutator = _mean_metric(commutator_rows, ["promoted_contextual_topk2"], "finite_update_commutator_l2")
    promoted_churn = _mean_abs_metric(forgetting_rows, ["promoted_contextual_topk2"], "functional_churn")
    rows = [
        (
            "primary_transformer_acsr_design",
            "transformer_acsr",
            "small_causal_transformer_predicts_future_context_chunks_and_router_logits",
            "current_hidden;previous_hidden_window;position_encoding;past_support;past_residual_norm;past_entropy",
            "next_hidden;next_delta;full_context_router_logits;softened_topk2_support_distribution",
            "causal_masked_transformer_2_layers_4_heads_hidden_32_context_window_16",
            "implement_local_transformer_acsr_cpu_smoke_pilot",
            True,
        ),
        (
            "causal_feature_safe_topk2_control",
            "causal_contextual_topk2",
            "router_uses_only_current_and_past_features_no_predicted_future_chunks",
            "current_hidden;previous_hidden_window;position_encoding",
            "",
            "same_router_budget_as_primary_without_future_target_head",
            "control_for_router_capacity_without_anticipatory_prediction",
            False,
        ),
        (
            "mlp_gru_predictor_control",
            "mlp_gru_acsr_control",
            "prior_shallow_acsr_predictors_as_ablations_not_primary_branch",
            "current_hidden;previous_hidden_summary;position_encoding",
            "next_hidden;next_delta;full_context_router_logits",
            "parameter_matched_mlp_and_gru_predictors",
            "control_for_sequence_model_choice",
            False,
        ),
        (
            "token_position_transformer_null",
            "token_position_only_transformer_null",
            "causal_transformer_without_hidden_features",
            "token_id;position_encoding",
            "next_hidden;next_delta;full_context_router_logits",
            "same_depth_small_causal_transformer",
            "reject_position_shortcut_explanation",
            False,
        ),
        (
            "shuffled_delayed_target_null",
            "misaligned_future_target_null",
            "same_predictor_trained_on_shuffled_or_delayed_future_chunks",
            "current_hidden;previous_hidden_window;position_encoding",
            "shuffled_next_hidden;delayed_next_delta;misaligned_router_logits",
            "same_primary_transformer_budget",
            "reject_target_alignment_shortcut",
            False,
        ),
        (
            "same_student_support_intervention_check",
            "same_student_support_swap",
            "evaluate_predicted_feature_supports_through_same_residual_values",
            "heldout_hidden;predicted_support;null_support",
            "fixed_residual_value_ce_and_intervention_fingerprints",
            "no_extra_training",
            "separate_router_support_signal_from_value_capacity",
            False,
        ),
        (
            "future_perturbation_invariance_check",
            "causal_leakage_check",
            "perturb_future_tokens_and_require_router_outputs_unchanged",
            "prefix_features_before_current_position",
            "support_logits_before_and_after_future_perturbation",
            "max_support_logit_delta_threshold_1e-5",
            "fail_if_future_positions_affect_deployable_router",
            False,
        ),
        (
            "retention_churn_commutator_gate",
            "non_ce_guardrail",
            "require_ce_guardrail_plus_support_regret_churn_and_commutator_non_regression",
            "a_to_b_adaptation_rows;commutator_rows;oracle_regret_rows",
            "support_churn;functional_churn;finite_update_commutator;oracle_support_regret",
            "local_cpu_smoke_then_repeat_only_if_discriminative",
            "block_gpu_until_ce_and_non_ce_gates_pass",
            False,
        ),
    ]
    return [
        {
            "design_name": "transformer_acsr",
            "design_role": role,
            "candidate_family": family,
            "candidate_mechanism": mechanism,
            "causal_input_tensors": causal_inputs,
            "future_context_targets": targets,
            "predictor_spec": predictor_spec,
            "first_local_config": "configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml",
            "first_output_dir": "results/reports/transformer_acsr_design",
            "artifact_schema": (
                "transformer_acsr_design.csv;summary.json;predictor_target_metrics.csv;"
                "support_intervention_metrics.csv;causal_leakage_checks.csv;retention_churn_commutator.csv"
            ),
            "required_controls": (
                "full_context_contextual_mlp_teacher;causal_feature_safe_topk2;mlp_gru_acsr;"
                "token_position_transformer;shuffled_delayed_targets;linear_topk2;random_fixed_topk2;"
                "rank_matched_topk1;dense_rank_norm_matched"
            ),
            "pass_fail_criteria": (
                "match_or_improve_ce_vs_causal_feature_safe_topk2;close_nonzero_full_context_teacher_gap;"
                "beat_shuffled_delayed_and_token_position_nulls;no_future_perturbation_effect;"
                "nonworse_oracle_regret_churn_commutator_vs_causal_topk2"
            ),
            "smallest_implementation_patch": (
                "add command-driven transformer_acsr pilot module, causal feature extractor, teacher target "
                "extractor, small causal transformer predictor, support swap evaluator, and focused artifact tests"
            ),
            "source_pc_closeout_status": pc_closeout.get("closeout_status", ""),
            "source_soft_mixture_selected_next_experiment": soft_design.get("selected_next_experiment", ""),
            "soft_mixture_deferred_by_transformer_acsr_priority": True,
            "promoted_full_context_teacher_arm": "promoted_contextual_topk2",
            "promoted_full_context_teacher_ce": promoted_ce,
            "causal_feature_safe_proxy_arm": "token_position_router_topk2",
            "causal_feature_safe_proxy_ce": token_position_ce,
            "linear_or_sparse_proxy_arm": "intervention_trained_sparse_topk2",
            "linear_or_sparse_proxy_ce": linear_ce,
            "random_support_proxy_ce": random_ce,
            "dense_rank_norm_control_ce": dense_ce,
            "promoted_teacher_commutator_l2": promoted_commutator,
            "promoted_teacher_abs_functional_churn": promoted_churn,
            "promoted_teacher_flop_proxy_per_token": _metric_float(
                budget_by_arm.get("promoted_contextual_topk2", {}).get("flop_proxy_per_token")
            ),
            "selected": selected,
            "design_selected": True,
            "implemented_in_current_packet": False,
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "advance_to_gpu_validation": False,
            "strategic_change_level": "major",
            "notify_ben": True,
            "selected_next_experiment": next_experiment,
            "interpretation": (
                "Ben's 2026-06-30 direction supersedes the soft-mixture residual branch: create a fail-closed "
                "Transformer-ACSR design before GPU work. The promoted full-context contextual router is treated "
                "as a nondeployable teacher, not deployable evidence."
            ),
        }
        for role, family, mechanism, causal_inputs, targets, predictor_spec, next_experiment, selected in rows
    ]


def _transformer_acsr_design_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "design_selected": False,
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "notify_ben": False,
            "strategic_change_level": "minor",
            "selected_next_experiment": "",
        }
    selected = next((row for row in rows if row.get("selected") is True), rows[0])
    return {
        "row_count": len(rows),
        "design_name": selected.get("design_name", ""),
        "candidate_family": selected.get("candidate_family", ""),
        "future_context_targets": selected.get("future_context_targets", ""),
        "causal_input_tensors": selected.get("causal_input_tensors", ""),
        "first_local_config": selected.get("first_local_config", ""),
        "artifact_schema": selected.get("artifact_schema", ""),
        "required_controls": selected.get("required_controls", ""),
        "pass_fail_criteria": selected.get("pass_fail_criteria", ""),
        "soft_mixture_deferred_by_transformer_acsr_priority": any(
            row.get("soft_mixture_deferred_by_transformer_acsr_priority") is True for row in rows
        ),
        "design_selected": any(row.get("design_selected") is True for row in rows),
        "implemented_in_current_packet": any(row.get("implemented_in_current_packet") is True for row in rows),
        "requires_gpu_now": any(row.get("requires_gpu_now") is True for row in rows),
        "promotion_allowed": any(row.get("promotion_allowed") is True for row in rows),
        "advance_to_gpu_validation": any(row.get("advance_to_gpu_validation") is True for row in rows),
        "notify_ben": any(row.get("notify_ben") is True for row in rows),
        "strategic_change_level": "major" if any(row.get("strategic_change_level") == "major" for row in rows) else "minor",
        "selected_next_experiment": selected.get("selected_next_experiment", ""),
        "interpretation": (
            "Fail-closed Transformer-ACSR design report. It records exact future targets, causal inputs, controls, "
            "local artifact schema, and pass/fail gates before any local pilot or GPU validation."
        ),
    }


def _next_hidden_targets(hidden: Any, torch: Any) -> Any:
    return torch.cat([hidden[:, 1:, :], hidden[:, -1:, :]], dim=1).detach()


def _train_future_delta_predictor(
    *,
    train_features: Any,
    train_target_delta: Any,
    holdout_features: Any,
    hidden_dim: int,
    training_steps: int,
    learning_rate: float,
    seed: int,
    torch: Any,
    nn: Any,
    F: Any,
) -> tuple[Any, Any]:
    torch.manual_seed(seed)
    predictor_dim = max(8, min(32, hidden_dim * 2))
    num_heads = next(candidate for candidate in (4, 2, 1) if predictor_dim % candidate == 0)
    predictor = _CausalTransformerFuturePredictor(
        hidden_dim,
        seq_len=int(train_features.shape[1]),
        predictor_dim=predictor_dim,
        num_heads=num_heads,
        nn=nn,
        torch=torch,
    )
    optimizer = torch.optim.AdamW(list(predictor.parameters()), lr=learning_rate)
    for _ in range(max(2, min(8, training_steps))):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(predictor(train_features), train_target_delta)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        holdout_predicted = predictor(holdout_features).detach()
    return predictor, holdout_predicted


def _support_intervention_ce_from_predicted_delta(
    *,
    hidden: Any,
    predicted_delta: Any,
    targets: Any,
    decode: Any,
    support_width: int,
    seed: int,
    torch: Any,
    F: Any,
) -> tuple[float, float, Any]:
    generator = torch.Generator(device=hidden.device)
    generator.manual_seed(seed)
    hidden_dim = int(hidden.shape[-1])
    num_columns = max(4, support_width * 4)
    values = torch.randn(
        num_columns,
        hidden_dim,
        generator=generator,
        device=hidden.device,
        dtype=hidden.dtype,
    )
    values = F.normalize(values, dim=-1)
    scores = torch.einsum("bth,ch->btc", predicted_delta, values)
    support = torch.topk(scores, k=min(support_width, num_columns), dim=-1).indices
    selected = values[support].mean(dim=-2)
    logits = decode(hidden + 0.05 * selected)
    ce = float(F.cross_entropy(logits.reshape(-1, int(logits.shape[-1])), targets.reshape(-1)).detach().item())
    support_churn = (
        float((support[:, 1:, :] != support[:, :-1, :]).float().mean().detach().item())
        if support.shape[1] > 1
        else 0.0
    )
    return ce, support_churn, support.detach()


def _support_overlap(left: Any, right: Any) -> float:
    if left.shape != right.shape:
        return 0.0
    return float((left == right).float().mean().detach().item())


def _transformer_acsr_cpu_smoke_pilot_rows(
    *,
    design_rows: list[dict[str, Any]],
    train_hidden: Any,
    train_inputs: Any,
    holdout_hidden: Any,
    holdout_inputs: Any,
    holdout_targets: Any,
    decode: Any,
    training_steps: int,
    learning_rate: float,
    seed: int,
    torch: Any,
    nn: Any,
    F: Any,
) -> list[dict[str, Any]]:
    design = _transformer_acsr_design_summary(design_rows)
    if not design.get("design_selected"):
        return []

    hidden_dim = int(train_hidden.shape[-1])
    support_width = 2
    train_delta = _next_hidden_targets(train_hidden, torch) - train_hidden
    holdout_delta = _next_hidden_targets(holdout_hidden, torch) - holdout_hidden

    primary_predictor, primary_delta = _train_future_delta_predictor(
        train_features=train_hidden,
        train_target_delta=train_delta,
        holdout_features=holdout_hidden,
        hidden_dim=hidden_dim,
        training_steps=training_steps,
        learning_rate=learning_rate,
        seed=seed + 700,
        torch=torch,
        nn=nn,
        F=F,
    )
    shuffled_target = train_delta[torch.randperm(train_delta.shape[0])]
    _, shuffled_delta = _train_future_delta_predictor(
        train_features=train_hidden,
        train_target_delta=shuffled_target,
        holdout_features=holdout_hidden,
        hidden_dim=hidden_dim,
        training_steps=training_steps,
        learning_rate=learning_rate,
        seed=seed + 701,
        torch=torch,
        nn=nn,
        F=F,
    )
    train_token_position_features = torch.zeros_like(train_hidden)
    train_token_position_features[..., 0] = train_inputs.float() / max(1.0, float(train_inputs.max().item()))
    train_token_position_features[..., 1] = torch.linspace(
        0.0,
        1.0,
        steps=int(train_inputs.shape[1]),
        dtype=train_hidden.dtype,
        device=train_hidden.device,
    ).unsqueeze(0)
    holdout_token_position_features = torch.zeros_like(holdout_hidden)
    holdout_token_position_features[..., 0] = holdout_inputs.float() / max(1.0, float(holdout_inputs.max().item()))
    holdout_token_position_features[..., 1] = torch.linspace(
        0.0,
        1.0,
        steps=int(holdout_inputs.shape[1]),
        dtype=holdout_hidden.dtype,
        device=holdout_hidden.device,
    ).unsqueeze(0)
    _, token_position_delta = _train_future_delta_predictor(
        train_features=train_token_position_features,
        train_target_delta=train_delta,
        holdout_features=holdout_token_position_features,
        hidden_dim=hidden_dim,
        training_steps=training_steps,
        learning_rate=learning_rate,
        seed=seed + 702,
        torch=torch,
        nn=nn,
        F=F,
    )
    torch.manual_seed(seed + 703)
    mlp = nn.Sequential(
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=learning_rate)
    for _ in range(max(2, min(8, training_steps))):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(mlp(train_hidden), train_delta)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        mlp_delta = mlp(holdout_hidden).detach()

    primary_mse = float(F.mse_loss(primary_delta, holdout_delta).detach().item())
    token_position_mse = float(F.mse_loss(token_position_delta, holdout_delta).detach().item())
    shuffled_mse = float(F.mse_loss(shuffled_delta, holdout_delta).detach().item())
    mlp_mse = float(F.mse_loss(mlp_delta, holdout_delta).detach().item())
    primary_cosine = float(
        F.cosine_similarity(
            primary_delta.reshape(-1, hidden_dim),
            holdout_delta.reshape(-1, hidden_dim),
            dim=-1,
        ).mean().detach().item()
    )
    leakage_delta = _causal_transformer_future_perturbation_max_delta(
        predictor=primary_predictor,
        features=holdout_hidden,
        perturb_from_position=max(1, int(holdout_hidden.shape[1]) // 2),
        perturb_scale=0.25,
    )
    base_logits = decode(holdout_hidden)
    base_ce = float(F.cross_entropy(base_logits.reshape(-1, int(base_logits.shape[-1])), holdout_targets.reshape(-1)).detach().item())
    primary_ce, primary_churn, primary_support = _support_intervention_ce_from_predicted_delta(
        hidden=holdout_hidden,
        predicted_delta=primary_delta,
        targets=holdout_targets,
        decode=decode,
        support_width=support_width,
        seed=seed + 710,
        torch=torch,
        F=F,
    )
    token_position_ce, token_position_churn, token_position_support = _support_intervention_ce_from_predicted_delta(
        hidden=holdout_hidden,
        predicted_delta=token_position_delta,
        targets=holdout_targets,
        decode=decode,
        support_width=support_width,
        seed=seed + 710,
        torch=torch,
        F=F,
    )
    oracle_ce, oracle_churn, oracle_support = _support_intervention_ce_from_predicted_delta(
        hidden=holdout_hidden,
        predicted_delta=holdout_delta,
        targets=holdout_targets,
        decode=decode,
        support_width=support_width,
        seed=seed + 710,
        torch=torch,
        F=F,
    )
    support_assay_valid = oracle_ce < token_position_ce and oracle_ce < base_ce
    primary_beats_token_position_support = primary_ce <= token_position_ce
    gates_pass = (
        leakage_delta <= 1e-5
        and primary_mse < token_position_mse
        and primary_mse < shuffled_mse
        and support_assay_valid
        and primary_beats_token_position_support
    )
    selected_next_experiment = (
        "repeat_transformer_acsr_cpu_smoke_across_seeds"
        if gates_pass
        else (
            "replace_support_intervention_assay_with_trained_same_student_residual_values"
            if not support_assay_valid
            else "tighten_transformer_acsr_pilot_against_null_controls_before_gpu"
        )
    )
    primary_oracle_overlap = _support_overlap(primary_support, oracle_support)
    token_position_oracle_overlap = _support_overlap(token_position_support, oracle_support)
    common = {
        "design_name": "transformer_acsr",
        "training_steps": max(2, min(8, training_steps)),
        "support_width": support_width,
        "future_perturbation_max_prefix_delta": leakage_delta,
        "leakage_gate_passes": leakage_delta <= 1e-5,
        "base_ce": base_ce,
        "oracle_support_intervention_ce": oracle_ce,
        "token_position_support_intervention_ce": token_position_ce,
        "oracle_ce_gain_vs_token_position": token_position_ce - oracle_ce,
        "oracle_ce_gain_vs_base": base_ce - oracle_ce,
        "support_intervention_assay_valid": support_assay_valid,
        "support_assay_failure_reason": (
            ""
            if support_assay_valid
            else "oracle_support_does_not_improve_same_student_ce_over_base_and_token_position_null"
        ),
        "primary_ce_gain_vs_token_position_support": token_position_ce - primary_ce,
        "primary_support_overlap_with_oracle": primary_oracle_overlap,
        "token_position_support_overlap_with_oracle": token_position_oracle_overlap,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "implemented_in_current_packet": True,
    }
    rows = [
        {
            **common,
            "row_role": "primary_transformer_acsr_cpu_smoke_pilot",
            "predictor_family": "causal_transformer",
            "target_alignment": "next_hidden_delta",
            "target_mse": primary_mse,
            "target_cosine": primary_cosine,
            "support_intervention_ce": primary_ce,
            "support_churn": primary_churn,
            "beats_token_position_mse": primary_mse < token_position_mse,
            "beats_shuffled_target_mse": primary_mse < shuffled_mse,
            "beats_mlp_mse": primary_mse < mlp_mse,
            "primary_beats_token_position_support_ce": primary_beats_token_position_support,
            "pilot_gates_pass": gates_pass,
            "selected_next_experiment": selected_next_experiment,
            "interpretation": (
                "Local CPU smoke only: predicted future chunks route through fixed same-student residual values. "
                "Promotion and GPU validation remain blocked until null/control, support-assay-validity, and "
                "non-CE gates pass robustly."
            ),
        },
        {
            **common,
            "row_role": "token_position_transformer_null",
            "predictor_family": "token_position_only_transformer",
            "target_alignment": "next_hidden_delta",
            "target_mse": token_position_mse,
            "target_cosine": float(F.cosine_similarity(token_position_delta.reshape(-1, hidden_dim), holdout_delta.reshape(-1, hidden_dim), dim=-1).mean().detach().item()),
            "support_intervention_ce": token_position_ce,
            "support_churn": token_position_churn,
            "pilot_gates_pass": False,
            "selected_next_experiment": "",
            "interpretation": "Null control for token/position shortcuts.",
        },
        {
            **common,
            "row_role": "shuffled_target_null",
            "predictor_family": "causal_transformer",
            "target_alignment": "batch_shuffled_next_hidden_delta",
            "target_mse": shuffled_mse,
            "target_cosine": float(F.cosine_similarity(shuffled_delta.reshape(-1, hidden_dim), holdout_delta.reshape(-1, hidden_dim), dim=-1).mean().detach().item()),
            "support_intervention_ce": None,
            "support_churn": None,
            "pilot_gates_pass": False,
            "selected_next_experiment": "",
            "interpretation": "Null control for future-target alignment.",
        },
        {
            **common,
            "row_role": "mlp_predictor_control",
            "predictor_family": "causal_mlp",
            "target_alignment": "next_hidden_delta",
            "target_mse": mlp_mse,
            "target_cosine": float(F.cosine_similarity(mlp_delta.reshape(-1, hidden_dim), holdout_delta.reshape(-1, hidden_dim), dim=-1).mean().detach().item()),
            "support_intervention_ce": None,
            "support_churn": None,
            "pilot_gates_pass": False,
            "selected_next_experiment": "",
            "interpretation": "Ablation for sequence-model choice.",
        },
        {
            **common,
            "row_role": "same_student_oracle_support_ceiling",
            "predictor_family": "evaluator_oracle_delta",
            "target_alignment": "actual_holdout_next_hidden_delta",
            "target_mse": 0.0,
            "target_cosine": 1.0,
            "support_intervention_ce": oracle_ce,
            "support_churn": oracle_churn,
            "pilot_gates_pass": False,
            "selected_next_experiment": "",
            "interpretation": "Evaluator-only same-student support ceiling; not deployable evidence.",
        },
    ]
    for row in rows:
        row["row_count"] = len(rows)
    return rows


def _transformer_acsr_cpu_smoke_pilot_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "implemented_in_current_packet": False,
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "advance_to_gpu_validation": False,
            "pilot_gates_pass": False,
            "selected_next_experiment": "",
        }
    primary = next((row for row in rows if row.get("row_role") == "primary_transformer_acsr_cpu_smoke_pilot"), rows[0])
    return {
        "row_count": len(rows),
        "implemented_in_current_packet": any(row.get("implemented_in_current_packet") is True for row in rows),
        "primary_target_mse": primary.get("target_mse"),
        "primary_target_cosine": primary.get("target_cosine"),
        "primary_support_intervention_ce": primary.get("support_intervention_ce"),
        "token_position_support_intervention_ce": primary.get("token_position_support_intervention_ce"),
        "oracle_support_intervention_ce": primary.get("oracle_support_intervention_ce"),
        "oracle_ce_gain_vs_token_position": primary.get("oracle_ce_gain_vs_token_position"),
        "oracle_ce_gain_vs_base": primary.get("oracle_ce_gain_vs_base"),
        "support_intervention_assay_valid": primary.get("support_intervention_assay_valid") is True,
        "support_assay_failure_reason": primary.get("support_assay_failure_reason", ""),
        "primary_ce_gain_vs_token_position_support": primary.get("primary_ce_gain_vs_token_position_support"),
        "primary_support_overlap_with_oracle": primary.get("primary_support_overlap_with_oracle"),
        "token_position_support_overlap_with_oracle": primary.get("token_position_support_overlap_with_oracle"),
        "future_perturbation_max_prefix_delta": primary.get("future_perturbation_max_prefix_delta"),
        "leakage_gate_passes": primary.get("leakage_gate_passes") is True,
        "beats_token_position_mse": primary.get("beats_token_position_mse") is True,
        "beats_shuffled_target_mse": primary.get("beats_shuffled_target_mse") is True,
        "beats_mlp_mse": primary.get("beats_mlp_mse") is True,
        "primary_beats_token_position_support_ce": primary.get("primary_beats_token_position_support_ce") is True,
        "pilot_gates_pass": primary.get("pilot_gates_pass") is True,
        "requires_gpu_now": any(row.get("requires_gpu_now") is True for row in rows),
        "promotion_allowed": any(row.get("promotion_allowed") is True for row in rows),
        "advance_to_gpu_validation": any(row.get("advance_to_gpu_validation") is True for row in rows),
        "selected_next_experiment": primary.get("selected_next_experiment", ""),
        "interpretation": (
            "Command-driven local Transformer-ACSR CPU smoke. It trains a tiny causal transformer on prefix-safe "
            "hidden features and reports null/control, leakage, and same-student fixed residual-value support metrics."
        ),
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
        out_dir / "support_head_sequence_heldout_diagnostic.csv",
        [] if training_smoke is None else training_smoke["support_head_sequence_heldout_diagnostic"],
    )
    _write_csv(
        out_dir / "router_only_branch_selection.csv",
        [] if training_smoke is None else training_smoke["router_only_branch_selection"],
    )
    _write_csv(
        out_dir / "teacher_distillation_closeout.csv",
        [] if training_smoke is None else training_smoke["teacher_distillation_closeout"],
    )
    _write_csv(
        out_dir / "value_capacity_core_periphery_diagnostic.csv",
        [] if training_smoke is None else training_smoke["value_capacity_core_periphery_diagnostic"],
    )
    _write_csv(
        out_dir / "core_periphery_sparse_value_capacity_probe.csv",
        [] if training_smoke is None else training_smoke["core_periphery_sparse_value_capacity_probe"],
    )
    _write_csv(
        out_dir / "core_periphery_update_stability_bracket.csv",
        [] if training_smoke is None else training_smoke["core_periphery_update_stability_bracket"],
    )
    _write_csv(
        out_dir / "core_periphery_branch_closeout.csv",
        [] if training_smoke is None else training_smoke["core_periphery_branch_closeout"],
    )
    _write_csv(
        out_dir / "sparse_value_redesign_selector.csv",
        [] if training_smoke is None else training_smoke["sparse_value_redesign_selector"],
    )
    _write_csv(
        out_dir / "budget_normalized_gated_value_mixture_pregate.csv",
        [] if training_smoke is None else training_smoke["budget_normalized_gated_value_mixture_pregate"],
    )
    _write_csv(
        out_dir / "budget_normalized_gated_value_mixture_closeout.csv",
        [] if training_smoke is None else training_smoke["budget_normalized_gated_value_mixture_closeout"],
    )
    _write_csv(
        out_dir / "soft_mixture_low_churn_dense_modular_design.csv",
        [] if training_smoke is None else training_smoke["soft_mixture_low_churn_dense_modular_design"],
    )
    _write_csv(
        out_dir / "pc_core_periphery_residual_inference_pregate.csv",
        [] if training_smoke is None else training_smoke["pc_core_periphery_residual_inference_pregate"],
    )
    _write_csv(
        out_dir / "pc_residual_inference_mechanism_inspection.csv",
        [] if training_smoke is None else training_smoke["pc_residual_inference_mechanism_inspection"],
    )
    _write_csv(
        out_dir / "pc_error_target_inference_path_audit.csv",
        [] if training_smoke is None else training_smoke["pc_error_target_inference_path_audit"],
    )
    _write_csv(
        out_dir / "pc_decoder_adjoint_target_alignment_probe.csv",
        [] if training_smoke is None else training_smoke["pc_decoder_adjoint_target_alignment_probe"],
    )
    _write_csv(
        out_dir / "pc_decoder_adjoint_minimal_retrain_probe.csv",
        [] if training_smoke is None else training_smoke["pc_decoder_adjoint_minimal_retrain_probe"],
    )
    _write_csv(
        out_dir / "pc_decoder_adjoint_closeout.csv",
        [] if training_smoke is None else training_smoke["pc_decoder_adjoint_closeout"],
    )
    _write_csv(
        out_dir / "pc_amortized_error_pregate_design.csv",
        [] if training_smoke is None else training_smoke["pc_amortized_error_pregate_design"],
    )
    _write_csv(
        out_dir / "pc_amortized_error_pregate.csv",
        [] if training_smoke is None else training_smoke["pc_amortized_error_pregate"],
    )
    _write_csv(
        out_dir / "pc_amortized_error_pregate_closeout.csv",
        [] if training_smoke is None else training_smoke["pc_amortized_error_pregate_closeout"],
    )
    _write_csv(
        out_dir / "transformer_acsr_design.csv",
        [] if training_smoke is None else training_smoke["transformer_acsr_design"],
    )
    _write_csv(
        out_dir / "transformer_acsr_cpu_smoke_pilot.csv",
        [] if training_smoke is None else training_smoke["transformer_acsr_cpu_smoke_pilot"],
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
        f"- Support-head sequence-heldout diagnostic rows: `{summary['support_head_sequence_heldout_diagnostic_row_count']}`",
        f"- Router-only branch-selection rows: `{summary['router_only_branch_selection_row_count']}`",
        f"- Teacher distillation closeout rows: `{summary['teacher_distillation_closeout_row_count']}`",
        f"- Value/capacity core-periphery diagnostic rows: `{summary['value_capacity_core_periphery_diagnostic_row_count']}`",
        f"- Core/periphery sparse value-capacity probe rows: `{summary['core_periphery_sparse_value_capacity_probe_row_count']}`",
        f"- Core/periphery update-stability bracket rows: `{summary['core_periphery_update_stability_bracket_row_count']}`",
        f"- Core/periphery branch closeout rows: `{summary['core_periphery_branch_closeout_row_count']}`",
        f"- Sparse value redesign selector rows: `{summary['sparse_value_redesign_selector_row_count']}`",
        f"- Budget-normalized gated value-mixture pregate rows: `{summary['budget_normalized_gated_value_mixture_pregate_row_count']}`",
        f"- Budget-normalized gated value-mixture closeout rows: `{summary['budget_normalized_gated_value_mixture_closeout_row_count']}`",
        f"- Soft-mixture low-churn dense modular design rows: `{summary['soft_mixture_low_churn_dense_modular_design_row_count']}`",
        f"- PC core/periphery residual-inference pregate rows: `{summary['pc_core_periphery_residual_inference_pregate_row_count']}`",
        f"- PC residual-inference mechanism inspection rows: `{summary['pc_residual_inference_mechanism_inspection_row_count']}`",
        f"- PC error-target inference-path audit rows: `{summary['pc_error_target_inference_path_audit_row_count']}`",
        f"- PC decoder-adjoint target-alignment probe rows: `{summary['pc_decoder_adjoint_target_alignment_probe_row_count']}`",
        f"- PC decoder-adjoint minimal retrain probe rows: `{summary['pc_decoder_adjoint_minimal_retrain_probe_row_count']}`",
        f"- PC decoder-adjoint closeout rows: `{summary['pc_decoder_adjoint_closeout_row_count']}`",
        f"- PC amortized error pregate design rows: `{summary['pc_amortized_error_pregate_design_row_count']}`",
        f"- PC amortized error pregate rows: `{summary['pc_amortized_error_pregate_row_count']}`",
        f"- PC amortized error pregate closeout rows: `{summary['pc_amortized_error_pregate_closeout_row_count']}`",
        f"- Transformer-ACSR design rows: `{summary['transformer_acsr_design_row_count']}`",
        f"- Transformer-ACSR CPU smoke pilot rows: `{summary['transformer_acsr_cpu_smoke_pilot_row_count']}`",
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
        "The support-head sequence-heldout diagnostic trains only diagnostic support selectors from train-split oracle pair supports, then swaps hard top-k2 supports into fixed sparse values on heldout sequences. Oracle targets enter auxiliary diagnostic training, so the artifact can justify or close a branch but cannot support deployment or promotion claims.",
        "",
        "The router-only branch-selection artifact closes or deprioritizes the router/support-head path unless the fixed-value oracle ceiling can close the stored-control gap and the sequence-heldout support head beats strong nulls. It keeps GPU validation and promotion blocked.",
        "",
        "The CE by rule/position and residual budget accounting artifacts are local diagnostics. FLOP values are residual-path proxies only and exclude the frozen base and decoder.",
        "",
        "The opt-in dense-teacher distillation arms are diagnostics only. The dense teacher is trained on the synthetic training episodes, sparse students are evaluated with hard top-k2 supports, and the shuffled-teacher arm is a null for teacher residual target structure.",
        "",
        "The teacher-distillation closeout artifact compares the dense-teacher sparse student to the shuffled-teacher null, the best existing sparse arm, active-matched dense/MLP controls, stored-matched upper bounds, and oracle-support regret. It is a local branch triage artifact, not promotion evidence.",
        "",
        "The value/capacity core-periphery diagnostic synthesizes the measured CE, residual-budget, finite-update commutator, and functional-churn rows after the router-only branch is closed. It selects a local core/periphery sparse value-capacity probe only when the stored-control gap remains too large for support selection to explain; it does not request GPU validation or promotion.",
        "",
        "The core/periphery branch closeout artifact is fail-closed. It closes or redirects the current clipped core/periphery value-capacity branch unless the primary arm clears CE or stored-gap thresholds, norm, commutator, and churn budgets, and is not explained by the flat same-router value-capacity control.",
        "",
        "The budget-normalized gated value-mixture pregate is the selected local sparse-value redesign. It keeps the promoted contextual top-k2 router, adds residual-budget clipping plus a learned low-rank value gate, compares against the flat same-router value MLP control, and remains fail-closed with no GPU or promotion request.",
        "",
        "The soft-mixture low-churn dense modular design selector consumes the failed gated sparse value-mixture closeout and defines the next bounded non-PC packet. It is design-only: no trainable arm, GPU validation, or promotion is implied until a future local pregate beats controls and interference budgets.",
        "",
        "The Transformer-ACSR design report supersedes the soft-mixture residual branch under Ben's 2026-06-30 direction. It treats the promoted full-context contextual router as a nondeployable teacher, specifies future-context targets and prefix-safe causal inputs, and keeps GPU validation blocked until a local command-driven pilot clears CE, support-regret, churn, commutator, null, and future-perturbation gates.",
        "",
        "The Transformer-ACSR CPU smoke pilot is command-driven local evidence only. It trains a tiny causal-mask transformer on prefix-safe hidden features to predict next-hidden deltas, compares against token/position, shuffled-target, and MLP controls, checks future-perturbation invariance, and routes predicted supports through fixed same-student residual values. Promotion and GPU validation remain blocked unless these gates become robust across follow-up repeats.",
        "",
        "The PC core/periphery residual-inference pregate is a major-pivot scaffold from the external strategy review. It closes the current sparse value-capacity branch as unsupported, records that Ben should be notified, and defines the next local trainable packet: fixed contextual top-k2 router, protected predictive core, plastic residual-error periphery, two local inference steps, anchor KL, norm clamp, commutator/churn penalties, and flat/dense/token-position/random/shuffled controls. It is not promotion evidence.",
        "",
        "The PC error-target inference-path audit is the local follow-up after the trainable PC pregate failed. It checks whether the shuffled residual-error target null is competitive, whether the inference path has CE signal versus promoted sparse, whether the same-router flat control explains the result, and whether the finite-update commutator budget remains failed. It blocks retraining/GPU until those local checks support a coherent PC mechanism.",
        "",
        "The PC decoder-adjoint target-alignment probe compares the current decoder-embedding-minus-hidden target with a hidden-space CE descent target, finite-difference descent proxy, shuffled target, and sign-flipped null under matched injection norm. It is label-derived and training-time diagnostic only.",
        "",
        "The PC decoder-adjoint minimal retrain probe is a local fail-closed target-semantics gate. It trains tiny matched dense residual probes with decoder-adjoint, current, shuffled, sign-flipped, and no auxiliary targets, then blocks GPU unless the decoder-adjoint target survives those controls plus promoted sparse and same-router flat references.",
        "",
        "The PC amortized error pregate design is a design-only scaffold after the decoder-adjoint closeout. It defines the only allowed next PC path: a tiny label-free amortized multi-site error predictor with same-router flat, dense/rank/norm, shuffled-error, sign-flipped-error, token-position-only, and promoted sparse controls. It does not train a model or request GPU validation.",
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
