"""Fail-closed local audit for the Transformer-ACSR hidden support classifier."""

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


DEFAULT_OUT_DIR = Path("results/reports/hidden_support_classifier_sequence_ood_budget_audit")
DEFAULT_SEEDS = (17, 18, 19)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _mean_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_float_or_none(row.get(key)) for row in rows if row.get(key) not in (None, "")]
    return mean(values) if values else None


def _safe_gain(reference_ce: float | None, candidate_ce: float | None) -> float | None:
    if reference_ce is None or candidate_ce is None:
        return None
    return reference_ce - candidate_ce


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


def _primary_support_head_row(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if (
            row.get("arm") == "promoted_contextual_topk2"
            and row.get("diagnostic") == "support_regret_trained_contextual_router_topk2"
            and row.get("split") == "sequence_heldout"
        ):
            return row
    return rows[0] if rows else {}


def _first_pilot_row(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if row.get("row_role") == "primary_transformer_acsr_cpu_smoke_pilot":
            return row
    return rows[0] if rows else {}


def run_hidden_support_classifier_sequence_ood_budget_audit(
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
    seed_root = out_dir / "seeds"
    seed_root.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    closeout_rows: list[dict[str, Any]] = []

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
        pilot = _first_pilot_row(_read_csv(seed_out / "transformer_acsr_cpu_smoke_pilot.csv"))
        support_head = _primary_support_head_row(
            _read_csv(seed_out / "support_head_sequence_heldout_diagnostic.csv")
        )
        hidden_ce = _float_or_none(pilot.get("direct_hidden_support_classifier_ce"))
        learned_ce = _float_or_none(support_head.get("learned_router_ce"))
        oracle_ce = _float_or_none(support_head.get("oracle_pair_ce_ceiling"))
        token_null_gain = _float_or_none(
            pilot.get("direct_hidden_support_classifier_ce_gain_vs_token_position_null")
        )
        shuffled_null_gain = _float_or_none(
            pilot.get("direct_hidden_support_classifier_ce_gain_vs_shuffled_null")
        )
        frequency_null_gain = _float_or_none(
            pilot.get("direct_hidden_support_classifier_ce_gain_vs_frequency_null")
        )
        gain_vs_learned = _safe_gain(learned_ce, hidden_ce)
        regret_recovery = _safe_regret_recovery(
            learned_ce=learned_ce,
            candidate_ce=hidden_ce,
            oracle_ce=oracle_ce,
        )
        sequence_gate = bool(
            gain_vs_learned is not None
            and regret_recovery is not None
            and (gain_vs_learned > 0.0 or regret_recovery >= 0.25)
            and all(
                value is not None and value > 0.0
                for value in (token_null_gain, shuffled_null_gain, frequency_null_gain)
            )
        )
        audit_rows.append(
            {
                "seed": seed,
                "split": "sequence_heldout",
                "diagnostic": "direct_hidden_support_classifier",
                "evidence_measured": True,
                "learned_router_ce": learned_ce,
                "oracle_pair_ce_ceiling": oracle_ce,
                "hidden_classifier_ce": hidden_ce,
                "hidden_classifier_ce_gain_vs_learned_router": gain_vs_learned,
                "oracle_regret_recovery_vs_learned_router": regret_recovery,
                "hidden_classifier_ce_gain_vs_token_position_null": token_null_gain,
                "hidden_classifier_ce_gain_vs_shuffled_null": shuffled_null_gain,
                "hidden_classifier_ce_gain_vs_frequency_null": frequency_null_gain,
                "hidden_classifier_overlap_with_oracle": _float_or_none(
                    pilot.get("direct_hidden_support_classifier_overlap_with_oracle")
                ),
                "hidden_classifier_exact_match_with_oracle": _float_or_none(
                    pilot.get("direct_hidden_support_classifier_exact_match_with_oracle")
                ),
                "future_perturbation_max_prefix_delta": _float_or_none(
                    pilot.get("direct_hidden_support_classifier_future_perturbation_max_prefix_delta")
                ),
                "gate_passes": sequence_gate,
                "failure_reason": ""
                if sequence_gate
                else "must beat learned router or recover >=25% router-oracle regret and beat token/position, shuffled, and frequency nulls",
                "source_artifact_dir": str(seed_out),
                "source_status": summary["status"],
            }
        )
        audit_rows.append(
            {
                "seed": seed,
                "split": "rule_combo_heldout",
                "diagnostic": "direct_hidden_support_classifier",
                "evidence_measured": False,
                "gate_passes": False,
                "failure_reason": "rule-combo-heldout support-classifier intervention rows are not emitted by the current harness",
                "source_artifact_dir": str(seed_out),
                "source_status": summary["status"],
            }
        )
        hidden_churn = _float_or_none(pilot.get("direct_hidden_support_classifier_churn"))
        budget_rows.extend(
            [
                {
                    "seed": seed,
                    "budget": "residual_norm",
                    "evidence_measured": False,
                    "candidate_value": None,
                    "reference_value": None,
                    "gate_passes": False,
                    "failure_reason": "direct hidden support-classifier residual norm budget is not emitted",
                },
                {
                    "seed": seed,
                    "budget": "functional_churn",
                    "evidence_measured": hidden_churn is not None,
                    "candidate_value": hidden_churn,
                    "reference_value": None,
                    "gate_passes": False,
                    "failure_reason": "nonworse functional-churn reference for the learned router is not emitted",
                },
                {
                    "seed": seed,
                    "budget": "finite_update_commutator",
                    "evidence_measured": False,
                    "candidate_value": None,
                    "reference_value": None,
                    "gate_passes": False,
                    "failure_reason": "direct hidden support-classifier finite-update commutator rows are not emitted",
                },
            ]
        )

    sequence_gate_passes = all(
        row["gate_passes"] for row in audit_rows if row["split"] == "sequence_heldout"
    )
    rule_ood_gate_passes = all(
        row["gate_passes"] for row in audit_rows if row["split"] == "rule_combo_heldout"
    )
    budget_gate_passes = all(row["gate_passes"] for row in budget_rows)
    advance_to_gpu_validation = bool(sequence_gate_passes and rule_ood_gate_passes and budget_gate_passes)
    sequence_rows = [row for row in audit_rows if row["split"] == "sequence_heldout"]
    sequence_evidence_measured = bool(sequence_rows) and all(
        row["evidence_measured"] for row in sequence_rows
    )
    close_hidden_classifier_branch = bool(
        sequence_evidence_measured and not sequence_gate_passes
    )
    closeout_status = (
        "closed_hidden_support_classifier_branch_before_gpu"
        if close_hidden_classifier_branch
        else "hidden_support_classifier_gpu_ready"
        if advance_to_gpu_validation
        else "requires_exact_rule_combo_and_budget_rows_before_decision"
    )
    closeout_rows.append(
        {
            "branch": "direct_hidden_support_classifier",
            "status": closeout_status,
            "sequence_evidence_measured": sequence_evidence_measured,
            "sequence_heldout_gate_passes": sequence_gate_passes,
            "rule_combo_heldout_gate_passes": rule_ood_gate_passes,
            "budget_gate_passes": budget_gate_passes,
            "mean_hidden_classifier_ce_gain_vs_learned_router": _mean_present(
                audit_rows, "hidden_classifier_ce_gain_vs_learned_router"
            ),
            "mean_oracle_regret_recovery_vs_learned_router": _mean_present(
                audit_rows, "oracle_regret_recovery_vs_learned_router"
            ),
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "next_step": (
                "close_or_redesign_hidden_support_classifier_branch_before_gpu"
                if close_hidden_classifier_branch
                else "run_runpod_hidden_support_classifier_validation_with_artifact_checks"
                if advance_to_gpu_validation
                else "add_rule_combo_heldout_and_exact_budget_rows_before_gpu"
            ),
            "deferred_exact_row_reason": (
                "sequence-heldout same-student intervention rows already lose to the learned router on the necessary gate"
                if close_hidden_classifier_branch
                else ""
            ),
        }
    )
    if close_hidden_classifier_branch:
        selected_next_step = "close_or_redesign_hidden_support_classifier_branch_before_gpu"
    elif advance_to_gpu_validation:
        selected_next_step = "run_runpod_hidden_support_classifier_validation_with_artifact_checks"
    else:
        selected_next_step = "add_rule_combo_heldout_and_exact_budget_rows_or_close_hidden_classifier_branch"
    summary = {
        "status": "pass",
        "decision": (
            "hidden_support_classifier_sequence_ood_budget_audit_passed_gpu_ready"
            if advance_to_gpu_validation
            else "hidden_support_classifier_sequence_ood_budget_audit_gpu_blocked"
        ),
        "seed_count": len(seeds),
        "sequence_heldout_gate_passes": sequence_gate_passes,
        "rule_combo_heldout_gate_passes": rule_ood_gate_passes,
        "budget_gate_passes": budget_gate_passes,
        "residual_norm_budget_gate_passes": all(
            row["gate_passes"] for row in budget_rows if row["budget"] == "residual_norm"
        ),
        "functional_churn_budget_gate_passes": all(
            row["gate_passes"] for row in budget_rows if row["budget"] == "functional_churn"
        ),
        "commutator_budget_gate_passes": all(
            row["gate_passes"] for row in budget_rows if row["budget"] == "finite_update_commutator"
        ),
        "mean_hidden_classifier_ce_gain_vs_learned_router": _mean_present(
            audit_rows, "hidden_classifier_ce_gain_vs_learned_router"
        ),
        "mean_oracle_regret_recovery_vs_learned_router": _mean_present(
            audit_rows, "oracle_regret_recovery_vs_learned_router"
        ),
        "closeout_status": closeout_status,
        "close_hidden_classifier_branch": close_hidden_classifier_branch,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": advance_to_gpu_validation,
        "selected_next_step": selected_next_step,
        "artifacts": {
            "audit_rows_csv": str(out_dir / "audit_rows.csv"),
            "budget_rows_csv": str(out_dir / "budget_rows.csv"),
            "closeout_rows_csv": str(out_dir / "closeout_rows.csv"),
            "summary_json": str(out_dir / "summary.json"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    _write_csv(out_dir / "audit_rows.csv", audit_rows)
    _write_csv(out_dir / "budget_rows.csv", budget_rows)
    _write_csv(out_dir / "closeout_rows.csv", closeout_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    notes = [
        "# Hidden Support-Classifier Sequence/OOD/Budget Audit",
        "",
        f"- Seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"- Sequence-heldout gate passes: `{sequence_gate_passes}`",
        f"- Rule-combo-heldout gate passes: `{rule_ood_gate_passes}`",
        f"- Budget gate passes: `{budget_gate_passes}`",
        f"- Mean CE gain vs learned router: `{summary['mean_hidden_classifier_ce_gain_vs_learned_router']}`",
        f"- Mean oracle-regret recovery vs learned router: `{summary['mean_oracle_regret_recovery_vs_learned_router']}`",
        f"- Closeout status: `{closeout_status}`",
        f"- Decision: `{summary['decision']}`",
        f"- Next step: `{summary['selected_next_step']}`",
        "",
        (
            "GPU validation remains blocked. Exact rule-combo and budget rows are still required for any "
            "positive hidden-classifier claim, but the current branch can be closed or redesigned now because "
            "the measured sequence-heldout same-student intervention gate already fails against the learned router."
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
    summary = run_hidden_support_classifier_sequence_ood_budget_audit(
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
