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
    "notes.md",
)


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
    controls = _comparator_controls(support_width=support_width)
    intervention_rows = _intervention_schema_rows(controls)
    commutator_rows = _commutator_schema_rows(controls)
    forgetting_rows = _forgetting_schema_rows(controls)
    budget_rows = _budget_rows(vocab_size=vocab_size, seq_len=seq_len, support_width=support_width)
    gate_rows = _gate_rows(
        episode_rows=episode_rows,
        controls=controls,
        intervention_rows=intervention_rows,
        commutator_rows=commutator_rows,
        forgetting_rows=forgetting_rows,
        budget_rows=budget_rows,
        training_hooks_available=training_hooks_available,
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
        "first_generated_rows": episode_rows[: min(6, len(episode_rows))],
        "reusable_apis": [
            "relaleap.smoke.TinyCharTransformer",
            "relaleap.smoke.ResidualColumns",
            "relaleap.experiments.mechanism_factorized_continual_learning_probe",
        ],
        "missing_training_hooks": _missing_training_hooks(training_hooks_available),
        "selected_next_step": (
            "wire the synthetic episode generator into a tiny CPU training smoke for promoted contextual top-k2, "
            "random/fixed support, token-position router, dense/rank/norm, low-churn MLP, and intervention-trained sparse arms"
            if hard_failures
            else "run the tiny CPU synthetic causal-modularity smoke and evaluate intervention purity, leakage, forgetting, and commutators"
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
    )
    return summary


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
) -> list[dict[str, Any]]:
    rules_seen = {row["latent_rule"] for row in episode_rows}
    controls_seen = {row["arm"] for row in controls}
    return [
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
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
