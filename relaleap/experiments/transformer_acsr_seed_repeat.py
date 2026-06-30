"""Repeat the local Transformer-ACSR CPU smoke pilot across adjacent seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from relaleap.experiments.synthetic_mechanism_causal_modularity import (
    run_synthetic_mechanism_causal_modularity,
)


DEFAULT_OUT_DIR = Path("results/reports/transformer_acsr_seed_repeat")
DEFAULT_SEEDS = (17, 18, 19)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _mean_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_float_or_none(row.get(key)) for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def _safe_gain(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _safe_regret_recovery(
    *,
    learned_ce: float | None,
    candidate_ce: float | None,
    oracle_ce: float | None,
) -> float | None:
    if learned_ce is None or candidate_ce is None or oracle_ce is None:
        return None
    learned_regret = learned_ce - oracle_ce
    if learned_regret <= 0.0:
        return None
    candidate_regret = candidate_ce - oracle_ce
    return (learned_regret - candidate_regret) / learned_regret


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_transformer_acsr_seed_repeat(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    vocab_size: int = 16,
    seq_len: int = 10,
    train_episodes_per_rule: int = 3,
    holdout_episodes_per_rule: int = 2,
    support_width: int = 2,
    training_steps: int = 12,
    hidden_dim: int = 24,
    learning_rate: float = 8e-3,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict[str, Any]] = []
    seed_root = out_dir / "seeds"
    seed_root.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        seed_out = seed_root / f"seed_{seed}"
        summary = run_synthetic_mechanism_causal_modularity(
            out_dir=seed_out,
            seed=seed,
            vocab_size=vocab_size,
            seq_len=seq_len,
            train_episodes_per_rule=train_episodes_per_rule,
            holdout_episodes_per_rule=holdout_episodes_per_rule,
            support_width=support_width,
            run_training_smoke=True,
            training_steps=training_steps,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            include_teacher_distillation=True,
        )
        pilot = summary["transformer_acsr_cpu_smoke_pilot_primary_result"]
        sequence_audit = summary.get("support_head_sequence_heldout_diagnostic_primary_result") or {}
        hidden_classifier_ce = _float_or_none(
            pilot["direct_hidden_support_classifier_ce"]
        )
        learned_router_ce = _float_or_none(sequence_audit.get("learned_router_ce"))
        oracle_pair_ce = _float_or_none(sequence_audit.get("oracle_pair_ce_ceiling"))
        hidden_gain_vs_learned = _safe_gain(learned_router_ce, hidden_classifier_ce)
        hidden_regret_recovery = _safe_regret_recovery(
            learned_ce=learned_router_ce,
            candidate_ce=hidden_classifier_ce,
            oracle_ce=oracle_pair_ce,
        )
        hidden_sequence_heldout_gate_passes = bool(
            hidden_gain_vs_learned is not None
            and hidden_regret_recovery is not None
            and (hidden_gain_vs_learned > 0.0 or hidden_regret_recovery >= 0.25)
        )
        hidden_rule_ood_evidence_available = False
        hidden_churn_budget_evidence_available = False
        hidden_commutator_budget_evidence_available = False
        hidden_churn_budget_gate_passes = False
        seed_rows.append(
            {
                "seed": seed,
                "status": summary["status"],
                "source_decision": summary["decision"],
                "pilot_gates_pass": pilot["pilot_gates_pass"],
                "value_aware_gate_passes": pilot["value_aware_gate_passes"],
                "support_intervention_assay_valid": pilot["support_intervention_assay_valid"],
                "leakage_gate_passes": pilot["leakage_gate_passes"],
                "value_aware_leakage_gate_passes": pilot["value_aware_leakage_gate_passes"],
                "primary_support_intervention_ce": pilot["primary_support_intervention_ce"],
                "token_position_support_intervention_ce": pilot[
                    "token_position_support_intervention_ce"
                ],
                "value_aware_support_intervention_ce": pilot[
                    "value_aware_support_intervention_ce"
                ],
                "oracle_support_intervention_ce": pilot["oracle_support_intervention_ce"],
                "primary_ce_gain_vs_token_position_support": pilot[
                    "primary_ce_gain_vs_token_position_support"
                ],
                "value_aware_ce_gain_vs_token_position_support": pilot[
                    "value_aware_ce_gain_vs_token_position_support"
                ],
                "value_aware_ce_gain_vs_primary_support": pilot[
                    "value_aware_ce_gain_vs_primary_support"
                ],
                "primary_support_overlap_with_oracle": pilot[
                    "primary_support_overlap_with_oracle"
                ],
                "value_aware_support_overlap_with_oracle": pilot[
                    "value_aware_support_overlap_with_oracle"
                ],
                "direct_hidden_support_classifier_gate_passes": pilot[
                    "direct_hidden_support_classifier_gate_passes"
                ],
                "direct_hidden_support_classifier_ce": pilot[
                    "direct_hidden_support_classifier_ce"
                ],
                "direct_hidden_support_classifier_churn": pilot[
                    "direct_hidden_support_classifier_churn"
                ],
                "direct_hidden_support_classifier_overlap_with_oracle": pilot[
                    "direct_hidden_support_classifier_overlap_with_oracle"
                ],
                "direct_hidden_support_classifier_exact_match_with_oracle": pilot[
                    "direct_hidden_support_classifier_exact_match_with_oracle"
                ],
                "direct_hidden_support_classifier_future_perturbation_max_prefix_delta": pilot[
                    "direct_hidden_support_classifier_future_perturbation_max_prefix_delta"
                ],
                "direct_hidden_support_classifier_ce_gain_vs_token_position_null": pilot[
                    "direct_hidden_support_classifier_ce_gain_vs_token_position_null"
                ],
                "direct_hidden_support_classifier_ce_gain_vs_shuffled_null": pilot[
                    "direct_hidden_support_classifier_ce_gain_vs_shuffled_null"
                ],
                "direct_hidden_support_classifier_ce_gain_vs_frequency_null": pilot[
                    "direct_hidden_support_classifier_ce_gain_vs_frequency_null"
                ],
                "sequence_audit_learned_router_ce": learned_router_ce,
                "sequence_audit_oracle_pair_ce_ceiling": oracle_pair_ce,
                "direct_hidden_support_classifier_ce_gain_vs_learned_router": hidden_gain_vs_learned,
                "direct_hidden_support_classifier_oracle_regret_recovery_vs_learned_router": (
                    hidden_regret_recovery
                ),
                "direct_hidden_support_classifier_sequence_heldout_gate_passes": (
                    hidden_sequence_heldout_gate_passes
                ),
                "direct_hidden_support_classifier_rule_ood_evidence_available": (
                    hidden_rule_ood_evidence_available
                ),
                "direct_hidden_support_classifier_churn_budget_evidence_available": (
                    hidden_churn_budget_evidence_available
                ),
                "direct_hidden_support_classifier_churn_budget_gate_passes": (
                    hidden_churn_budget_gate_passes
                ),
                "direct_hidden_support_classifier_commutator_budget_evidence_available": (
                    hidden_commutator_budget_evidence_available
                ),
                "sequence_audit_contextual_ce": _float_or_none(sequence_audit.get("contextual_ce")),
                "sequence_audit_contextual_gain_vs_learned": _float_or_none(
                    sequence_audit.get("contextual_gain_vs_learned")
                ),
                "sequence_audit_advance_support_head_branch": bool(
                    sequence_audit.get("advance_support_head_branch")
                ),
                "selected_next_experiment": pilot["selected_next_experiment"],
                "requires_gpu_now": pilot["requires_gpu_now"],
                "promotion_allowed": pilot["promotion_allowed"],
                "seed_artifact_dir": str(seed_out),
            }
        )

    completed_count = sum(1 for row in seed_rows if row["status"] == "pass")
    value_gate_pass_count = sum(1 for row in seed_rows if row["value_aware_gate_passes"])
    hidden_classifier_gate_pass_count = sum(
        1 for row in seed_rows if row["direct_hidden_support_classifier_gate_passes"]
    )
    leakage_pass_count = sum(
        1
        for row in seed_rows
        if row["leakage_gate_passes"] and row["value_aware_leakage_gate_passes"]
    )
    hidden_classifier_leakage_pass_count = sum(
        1
        for row in seed_rows
        if row["direct_hidden_support_classifier_future_perturbation_max_prefix_delta"] is not None
        and row["direct_hidden_support_classifier_future_perturbation_max_prefix_delta"] <= 1e-5
    )
    assay_valid_count = sum(1 for row in seed_rows if row["support_intervention_assay_valid"])
    mean_overlap = _mean_present(seed_rows, "value_aware_support_overlap_with_oracle")
    mean_hidden_classifier_overlap = _mean_present(
        seed_rows, "direct_hidden_support_classifier_overlap_with_oracle"
    )
    mean_hidden_classifier_exact_match = _mean_present(
        seed_rows, "direct_hidden_support_classifier_exact_match_with_oracle"
    )
    mean_gain_vs_token_position = _mean_present(
        seed_rows, "value_aware_ce_gain_vs_token_position_support"
    )
    mean_hidden_classifier_gain_vs_token_position = _mean_present(
        seed_rows, "direct_hidden_support_classifier_ce_gain_vs_token_position_null"
    )
    mean_hidden_classifier_gain_vs_shuffled = _mean_present(
        seed_rows, "direct_hidden_support_classifier_ce_gain_vs_shuffled_null"
    )
    mean_hidden_classifier_gain_vs_frequency = _mean_present(
        seed_rows, "direct_hidden_support_classifier_ce_gain_vs_frequency_null"
    )
    mean_hidden_classifier_gain_vs_learned_router = _mean_present(
        seed_rows, "direct_hidden_support_classifier_ce_gain_vs_learned_router"
    )
    mean_hidden_classifier_oracle_regret_recovery_vs_learned_router = _mean_present(
        seed_rows,
        "direct_hidden_support_classifier_oracle_regret_recovery_vs_learned_router",
    )
    robust_value_gate = (
        completed_count == len(seed_rows)
        and value_gate_pass_count == len(seed_rows)
        and leakage_pass_count == len(seed_rows)
        and assay_valid_count == len(seed_rows)
    )
    robust_hidden_classifier_gate = (
        completed_count == len(seed_rows)
        and hidden_classifier_gate_pass_count == len(seed_rows)
        and hidden_classifier_leakage_pass_count == len(seed_rows)
        and assay_valid_count == len(seed_rows)
    )
    overlap_gate_passes = mean_overlap is not None and mean_overlap >= 0.25
    hidden_classifier_overlap_gate_passes = (
        mean_hidden_classifier_overlap is not None and mean_hidden_classifier_overlap >= 0.25
    )
    hidden_classifier_null_margin_gate_passes = all(
        value is not None and value > 0.0
        for value in (
            mean_hidden_classifier_gain_vs_token_position,
            mean_hidden_classifier_gain_vs_shuffled,
            mean_hidden_classifier_gain_vs_frequency,
        )
    )
    hidden_classifier_learned_router_comparison_available = all(
        row["sequence_audit_learned_router_ce"] is not None
        and row["sequence_audit_oracle_pair_ce_ceiling"] is not None
        and row["direct_hidden_support_classifier_ce"] is not None
        for row in seed_rows
    )
    hidden_classifier_learned_router_gate_passes = bool(
        hidden_classifier_learned_router_comparison_available
        and all(
            row["direct_hidden_support_classifier_sequence_heldout_gate_passes"]
            for row in seed_rows
        )
    )
    hidden_classifier_sequence_heldout_gate_passes = all(
        row["direct_hidden_support_classifier_sequence_heldout_gate_passes"]
        for row in seed_rows
    )
    hidden_classifier_rule_ood_evidence_available = all(
        row["direct_hidden_support_classifier_rule_ood_evidence_available"]
        for row in seed_rows
    )
    hidden_classifier_rule_ood_gate_passes = False
    hidden_classifier_churn_budget_evidence_available = all(
        row["direct_hidden_support_classifier_churn_budget_evidence_available"]
        for row in seed_rows
    )
    hidden_classifier_churn_budget_gate_passes = bool(
        hidden_classifier_churn_budget_evidence_available
        and all(
            row["direct_hidden_support_classifier_churn_budget_gate_passes"]
            for row in seed_rows
        )
    )
    hidden_classifier_commutator_budget_evidence_available = all(
        row["direct_hidden_support_classifier_commutator_budget_evidence_available"]
        for row in seed_rows
    )
    hidden_classifier_commutator_budget_gate_passes = False
    hidden_classifier_sequence_ood_budget_audit_available = bool(
        hidden_classifier_sequence_heldout_gate_passes
        and hidden_classifier_rule_ood_evidence_available
        and hidden_classifier_rule_ood_gate_passes
        and hidden_classifier_churn_budget_evidence_available
        and hidden_classifier_churn_budget_gate_passes
        and hidden_classifier_commutator_budget_evidence_available
        and hidden_classifier_commutator_budget_gate_passes
    )
    hidden_classifier_gpu_gate_passes = bool(
        robust_hidden_classifier_gate
        and hidden_classifier_overlap_gate_passes
        and hidden_classifier_null_margin_gate_passes
        and hidden_classifier_learned_router_comparison_available
        and hidden_classifier_learned_router_gate_passes
        and hidden_classifier_sequence_ood_budget_audit_available
    )
    advance_to_gpu_validation = bool(
        (robust_value_gate and overlap_gate_passes)
        or hidden_classifier_gpu_gate_passes
    )
    selected_next_step = (
        "run_runpod_transformer_acsr_validation_with_artifact_checks"
        if advance_to_gpu_validation
        else "run_hidden_support_classifier_sequence_ood_budget_audit_before_gpu"
        if robust_hidden_classifier_gate
        else "tighten_value_aware_transformer_acsr_oracle_overlap_and_null_controls_before_gpu"
        if robust_value_gate
        else "close_or_redesign_value_aware_transformer_acsr_support_router_locally"
    )
    summary = {
        "status": "pass" if completed_count == len(seed_rows) else "fail",
        "decision": (
            "transformer_acsr_seed_repeat_passed_gpu_validation_ready"
            if advance_to_gpu_validation
            else "transformer_acsr_seed_repeat_local_only_gpu_blocked"
        ),
        "seed_count": len(seed_rows),
        "completed_seed_count": completed_count,
        "value_aware_gate_pass_count": value_gate_pass_count,
        "hidden_classifier_gate_pass_count": hidden_classifier_gate_pass_count,
        "leakage_pass_count": leakage_pass_count,
        "hidden_classifier_leakage_pass_count": hidden_classifier_leakage_pass_count,
        "support_intervention_assay_valid_count": assay_valid_count,
        "mean_value_aware_ce_gain_vs_token_position_support": mean_gain_vs_token_position,
        "mean_value_aware_support_overlap_with_oracle": mean_overlap,
        "mean_hidden_classifier_ce_gain_vs_token_position_null": mean_hidden_classifier_gain_vs_token_position,
        "mean_hidden_classifier_ce_gain_vs_shuffled_null": mean_hidden_classifier_gain_vs_shuffled,
        "mean_hidden_classifier_ce_gain_vs_frequency_null": mean_hidden_classifier_gain_vs_frequency,
        "mean_hidden_classifier_ce_gain_vs_learned_router": (
            mean_hidden_classifier_gain_vs_learned_router
        ),
        "mean_hidden_classifier_oracle_regret_recovery_vs_learned_router": (
            mean_hidden_classifier_oracle_regret_recovery_vs_learned_router
        ),
        "mean_hidden_classifier_support_overlap_with_oracle": mean_hidden_classifier_overlap,
        "mean_hidden_classifier_exact_match_with_oracle": mean_hidden_classifier_exact_match,
        "robust_value_gate_passes": robust_value_gate,
        "robust_hidden_classifier_gate_passes": robust_hidden_classifier_gate,
        "oracle_overlap_gate_passes": overlap_gate_passes,
        "hidden_classifier_overlap_gate_passes": hidden_classifier_overlap_gate_passes,
        "hidden_classifier_null_margin_gate_passes": hidden_classifier_null_margin_gate_passes,
        "hidden_classifier_learned_router_comparison_available": hidden_classifier_learned_router_comparison_available,
        "hidden_classifier_learned_router_gate_passes": hidden_classifier_learned_router_gate_passes,
        "hidden_classifier_sequence_heldout_gate_passes": hidden_classifier_sequence_heldout_gate_passes,
        "hidden_classifier_rule_ood_evidence_available": hidden_classifier_rule_ood_evidence_available,
        "hidden_classifier_rule_ood_gate_passes": hidden_classifier_rule_ood_gate_passes,
        "hidden_classifier_churn_budget_evidence_available": hidden_classifier_churn_budget_evidence_available,
        "hidden_classifier_churn_budget_gate_passes": hidden_classifier_churn_budget_gate_passes,
        "hidden_classifier_commutator_budget_evidence_available": (
            hidden_classifier_commutator_budget_evidence_available
        ),
        "hidden_classifier_commutator_budget_gate_passes": hidden_classifier_commutator_budget_gate_passes,
        "hidden_classifier_sequence_ood_budget_audit_available": hidden_classifier_sequence_ood_budget_audit_available,
        "hidden_classifier_gpu_gate_passes": hidden_classifier_gpu_gate_passes,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": advance_to_gpu_validation,
        "selected_next_step": selected_next_step,
        "artifacts": {
            "seed_rows_csv": str(out_dir / "seed_rows.csv"),
            "summary_json": str(out_dir / "summary.json"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }

    _write_csv(out_dir / "seed_rows.csv", seed_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    notes = [
        "# Transformer-ACSR Seed Repeat",
        "",
        f"- Seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"- Value-aware gate pass count: `{value_gate_pass_count}/{len(seed_rows)}`",
        f"- Hidden support-classifier gate pass count: `{hidden_classifier_gate_pass_count}/{len(seed_rows)}`",
        f"- Leakage pass count: `{leakage_pass_count}/{len(seed_rows)}`",
        f"- Hidden support-classifier leakage pass count: `{hidden_classifier_leakage_pass_count}/{len(seed_rows)}`",
        f"- Mean value-aware CE gain vs token/position: `{mean_gain_vs_token_position}`",
        f"- Mean value-aware oracle overlap: `{mean_overlap}`",
        f"- Mean hidden support-classifier CE gain vs token/position null: `{mean_hidden_classifier_gain_vs_token_position}`",
        f"- Mean hidden support-classifier CE gain vs shuffled null: `{mean_hidden_classifier_gain_vs_shuffled}`",
        f"- Mean hidden support-classifier CE gain vs frequency null: `{mean_hidden_classifier_gain_vs_frequency}`",
        f"- Mean hidden support-classifier CE gain vs learned router: `{mean_hidden_classifier_gain_vs_learned_router}`",
        f"- Mean hidden support-classifier oracle-regret recovery vs learned router: `{mean_hidden_classifier_oracle_regret_recovery_vs_learned_router}`",
        f"- Mean hidden support-classifier oracle overlap: `{mean_hidden_classifier_overlap}`",
        f"- Hidden support-classifier learned-router comparison available: `{hidden_classifier_learned_router_comparison_available}`",
        f"- Hidden support-classifier learned-router gate passes: `{hidden_classifier_learned_router_gate_passes}`",
        f"- Hidden support-classifier sequence-heldout gate passes: `{hidden_classifier_sequence_heldout_gate_passes}`",
        f"- Hidden support-classifier rule-OOD evidence available: `{hidden_classifier_rule_ood_evidence_available}`",
        f"- Hidden support-classifier rule-OOD gate passes: `{hidden_classifier_rule_ood_gate_passes}`",
        f"- Hidden support-classifier churn budget evidence available: `{hidden_classifier_churn_budget_evidence_available}`",
        f"- Hidden support-classifier churn budget gate passes: `{hidden_classifier_churn_budget_gate_passes}`",
        f"- Hidden support-classifier commutator budget evidence available: `{hidden_classifier_commutator_budget_evidence_available}`",
        f"- Hidden support-classifier commutator budget gate passes: `{hidden_classifier_commutator_budget_gate_passes}`",
        f"- Hidden support-classifier sequence/OOD budget audit available: `{hidden_classifier_sequence_ood_budget_audit_available}`",
        f"- Decision: `{summary['decision']}`",
        f"- Next step: `{selected_next_step}`",
        "",
        (
            "GPU validation and promotion remain blocked unless the repeated local value-aware gate and oracle-overlap "
            "gate pass. Hidden support-classifier evidence is pre-GPU only until it explicitly beats the learned "
            "router or recovers at least 25% of router-oracle regret on heldout sequences, includes rule-OOD rows, "
            "and emits nonworse residual-norm, functional-churn, and commutator budget rows."
        ),
    ]
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--train-episodes-per-rule", type=int, default=3)
    parser.add_argument("--holdout-episodes-per-rule", type=int, default=2)
    parser.add_argument("--support-width", type=int, default=2)
    parser.add_argument("--training-steps", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=8e-3)
    args = parser.parse_args(argv)
    summary = run_transformer_acsr_seed_repeat(
        out_dir=args.out,
        seeds=tuple(args.seeds),
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        train_episodes_per_rule=args.train_episodes_per_rule,
        holdout_episodes_per_rule=args.holdout_episodes_per_rule,
        support_width=args.support_width,
        training_steps=args.training_steps,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
