"""Pregate a prefix-safe Transformer-ACSR mechanism test.

This module records whether the repository currently has enough command-driven
artifacts to start a deployable Transformer-ACSR experiment. It deliberately
fails closed: summary reports are useful provenance, but they are not a
train/heldout tensor dataset.
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


DEFAULT_OUT_DIR = Path("results/reports/transformer_acsr_pregate")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")

DEFAULT_SOURCE_PATHS: tuple[tuple[str, Path, bool], ...] = (
    (
        "anticipatory_contextual_support_routing_design",
        Path("results/reports/token_larger_anticipatory_contextual_support_routing_design/summary.json"),
        True,
    ),
    (
        "causal_contextual_router_gate",
        Path("results/reports/token_larger_causal_contextual_router_gate/summary.json"),
        True,
    ),
    (
        "causal_contextual_router_distillation_synthesis",
        Path("results/reports/token_larger_causal_contextual_router_distillation_synthesis/summary.json"),
        True,
    ),
    (
        "transformer_acsr_seed_repeat",
        Path("results/reports/transformer_acsr_seed_repeat/summary.json"),
        True,
    ),
    (
        "transformer_acsr_hidden_feature_redesign_gate",
        Path("results/reports/transformer_acsr_hidden_feature_redesign_gate/summary.json"),
        True,
    ),
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "feature_provenance.csv",
    "control_contract.csv",
    "pregate_requirements.csv",
    "notes.md",
)


def run_transformer_acsr_pregate(
    *,
    source_paths: tuple[tuple[str, Path, bool], ...] = DEFAULT_SOURCE_PATHS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed local pregate for the next Transformer-ACSR branch."""

    start = time.time()
    sources = [_source_row(name, path, required) for name, path, required in source_paths]
    strategy = _strategy_review(strategy_review_path)
    feature_rows = _feature_provenance_rows(sources)
    control_rows = _control_contract_rows()
    requirement_rows = _pregate_requirement_rows(sources, feature_rows, control_rows)
    failures = [
        {
            "source": row["source"],
            "path": row["path"],
            "reason": "required source artifact missing",
        }
        for row in sources
        if row["required"] and not row["present"]
    ]
    missing_requirements = [
        row["requirement"]
        for row in requirement_rows
        if row["status"] not in {"available", "registered"}
    ]
    tensor_dataset_available = _requirement_status(
        requirement_rows, "sequence_split_prefix_feature_tensor_dataset"
    ) == "available"
    same_student_available = _requirement_status(
        requirement_rows, "same_student_intervention_rows"
    ) == "available"
    trainable_now = bool(not failures and tensor_dataset_available and same_student_available)

    decision = (
        "transformer_acsr_pregate_ready_for_local_cpu_training"
        if trainable_now
        else "transformer_acsr_pregate_inventory_recorded_gpu_blocked"
    )
    selected_next_step = (
        "train_prefix_safe_transformer_acsr_cpu_pregate_with_registered_nulls"
        if trainable_now
        else "materialize_sequence_split_prefix_feature_teacher_tensor_dataset_before_training"
    )
    claim_status = (
        "prefix_safe_transformer_acsr_sources_ready_no_gpu"
        if trainable_now
        else "prefix_safe_transformer_acsr_training_data_incomplete_no_gpu"
    )

    summary = {
        "status": "fail" if failures else "pass",
        "decision": decision if not failures else "transformer_acsr_pregate_failed_closed",
        "claim_status": claim_status if not failures else "required_transformer_acsr_sources_missing",
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "trainable_now": trainable_now,
        "sequence_split_tensor_dataset_available": tensor_dataset_available,
        "same_student_intervention_rows_available": same_student_available,
        "feature_provenance": {
            "prefix_safe_count": sum(1 for row in feature_rows if row["prefix_safe"]),
            "future_or_target_leaking_count": sum(
                1 for row in feature_rows if row["future_or_target_leaking"]
            ),
            "nondeployable_teacher_count": sum(
                1 for row in feature_rows if row["nondeployable_teacher_only"]
            ),
        },
        "registered_control_count": len(control_rows),
        "missing_requirements": missing_requirements,
        "failures": failures,
        "source_rows": sources,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "backend_policy": (
            "local pregate only; RunPod and Colab remain blocked until a prefix-safe "
            "train/heldout mechanism gate passes"
        ),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    _write_artifacts(out_dir, summary, feature_rows, control_rows, requirement_rows)
    return summary


def _source_row(name: str, path: Path, required: bool) -> dict[str, Any]:
    if not path.is_file():
        return {
            "source": name,
            "path": str(path),
            "required": required,
            "present": False,
            "status": "missing",
            "decision": "",
            "claim_status": "",
            "selected_next_step": "",
            "selected_next_action": "",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "source": name,
        "path": str(path),
        "required": required,
        "present": True,
        "status": payload.get("status", "present"),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", payload.get("claim_statuses", "")),
        "selected_next_step": payload.get("selected_next_step", payload.get("next_step", "")),
        "selected_next_action": payload.get("selected_next_action", ""),
        "requires_gpu_now": payload.get("requires_gpu_now", ""),
        "promotion_allowed": payload.get("promotion_allowed", ""),
        "advance_to_gpu_validation": payload.get("advance_to_gpu_validation", ""),
    }


def _feature_provenance_rows(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    design_present = _present(sources, "anticipatory_contextual_support_routing_design")
    gate_present = _present(sources, "causal_contextual_router_gate")
    distill_present = _present(sources, "causal_contextual_router_distillation_synthesis")
    return [
        _feature(
            "current_hidden",
            "prefix_safe_router_input",
            "causal input available at current position",
            True,
            False,
            False,
            design_present,
        ),
        _feature(
            "previous_hidden",
            "prefix_safe_router_input",
            "causal history feature",
            True,
            False,
            False,
            design_present,
        ),
        _feature(
            "current_minus_previous_hidden",
            "prefix_safe_router_input",
            "causal local delta feature",
            True,
            False,
            False,
            design_present,
        ),
        _feature(
            "position_features",
            "prefix_safe_router_input",
            "normalized/sinusoidal position features",
            True,
            False,
            False,
            design_present,
        ),
        _feature(
            "past_support_summary",
            "optional_prefix_safe_router_input",
            "past-only support summary; must be masked at sequence start",
            True,
            False,
            False,
            design_present,
        ),
        _feature(
            "future_hidden",
            "teacher_prediction_target",
            "nondeployable full-context contextual_mlp next_hidden chunk",
            False,
            True,
            True,
            design_present,
        ),
        _feature(
            "future_delta",
            "teacher_prediction_target",
            "nondeployable full-context contextual_mlp next_hidden-current_hidden chunk",
            False,
            True,
            True,
            design_present,
        ),
        _feature(
            "full_context_contextual_mlp_support",
            "nondeployable_teacher_support_target",
            "teacher/ceiling support distribution, never a deployable router input",
            False,
            True,
            True,
            gate_present or distill_present,
        ),
        _feature(
            "target_token_or_loss",
            "forbidden_predictor_feature",
            "target/loss fields can be evaluation labels only",
            False,
            True,
            True,
            False,
        ),
    ]


def _feature(
    name: str,
    role: str,
    provenance: str,
    prefix_safe: bool,
    future_or_target_leaking: bool,
    nondeployable_teacher_only: bool,
    source_available: bool,
) -> dict[str, Any]:
    return {
        "feature": name,
        "role": role,
        "provenance": provenance,
        "prefix_safe": prefix_safe,
        "future_or_target_leaking": future_or_target_leaking,
        "nondeployable_teacher_only": nondeployable_teacher_only,
        "source_available": source_available,
        "allowed_as_router_input": prefix_safe and not future_or_target_leaking,
        "allowed_as_training_target": (
            role in {"teacher_prediction_target", "nondeployable_teacher_support_target"}
        ),
    }


def _control_contract_rows() -> list[dict[str, Any]]:
    return [
        _control("token_position_only_transformer", "shortcut null", "must lose to real prefix features"),
        _control("shuffled_future_targets", "target-alignment null", "must lose on support KL/top-k overlap"),
        _control("delayed_or_misaligned_targets", "temporal null", "must lose on heldout support metrics"),
        _control("frequency_preserving_support_permutation", "route null", "must lose in same-student interventions"),
        _control("causal_feature_safe_topk2", "deployable router control", "ACSR must not worsen regret/churn"),
        _control("mlp_or_gru_acsr_ablation", "architecture control", "Transformer must add sequence value"),
        _control("dense_rank_norm_matched_adapter", "capacity control", "sparse routing must survive budget controls"),
        _control("future_perturbation_invariance", "leakage check", "prefix scores/support must be invariant"),
        _control("retention_churn_commutator_budget", "interference check", "no hidden churn/commutator regression"),
    ]


def _control(name: str, role: str, pass_condition: str) -> dict[str, str]:
    return {
        "control": name,
        "role": role,
        "pass_condition": pass_condition,
        "registered": True,
    }


def _pregate_requirement_rows(
    sources: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "requirement": "nondeployable_teacher_labeled",
            "status": "available"
            if any(row["nondeployable_teacher_only"] for row in feature_rows)
            else "missing",
            "evidence": "feature_provenance.csv",
        },
        {
            "requirement": "prefix_safe_feature_contract",
            "status": "available"
            if any(row["allowed_as_router_input"] for row in feature_rows)
            else "missing",
            "evidence": "feature_provenance.csv",
        },
        {
            "requirement": "sequence_split_prefix_feature_tensor_dataset",
            "status": "missing",
            "evidence": (
                "summary artifacts found, but no command-driven tensor/row dataset with "
                "train/heldout prefix features and teacher targets is registered"
            ),
        },
        {
            "requirement": "same_student_intervention_rows",
            "status": "available"
            if _present(sources, "transformer_acsr_hidden_feature_redesign_gate")
            else "missing",
            "evidence": "transformer_acsr_hidden_feature_redesign_gate source",
        },
        {
            "requirement": "null_and_control_contract",
            "status": "registered" if len(control_rows) >= 6 else "missing",
            "evidence": "control_contract.csv",
        },
        {
            "requirement": "gpu_validation_gate",
            "status": "blocked",
            "evidence": "requires local train/heldout pregate pass before RunPod",
        },
    ]


def _requirement_status(rows: list[dict[str, Any]], requirement: str) -> str:
    for row in rows:
        if row["requirement"] == requirement:
            return str(row["status"])
    return "missing"


def _present(sources: list[dict[str, Any]], source: str) -> bool:
    return any(row["source"] == source and row["present"] for row in sources)


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "present": False,
            "path": str(path),
            "strategic_change_level": "missing",
            "notify_ben": False,
            "verdict": "missing",
            "recommended_next_action": "",
        }
    lines = path.read_text(encoding="utf-8").splitlines()
    header: dict[str, str] = {}
    for line in lines[:12]:
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    return {
        "present": True,
        "path": str(path),
        "strategic_change_level": header.get("strategic_change_level", ""),
        "notify_ben": header.get("notify_ben", "").lower() == "true",
        "verdict": header.get("verdict", ""),
        "recommended_next_action": header.get("recommended_next_action", ""),
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No strategy review was present; pregate still fails closed and keeps GPU blocked."
    notify = " Ben should be notified." if strategy["notify_ben"] else ""
    return (
        "Accepted GPT-5.5-Pro recommendation to start a bounded local Transformer-ACSR "
        "pregate, avoid another selector, and keep GPU blocked until prefix-safe local "
        f"mechanism gates pass. strategic_change_level={strategy['strategic_change_level']}; "
        f"notify_ben={strategy['notify_ben']}.{notify}"
    )


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    feature_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    requirement_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "feature_provenance.csv", feature_rows)
    _write_csv(out_dir / "control_contract.csv", control_rows)
    _write_csv(out_dir / "pregate_requirements.csv", requirement_rows)
    notes = [
        "# Transformer-ACSR Pregate",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Trainable now: `{summary['trainable_now']}`",
        f"- Sequence-split tensor dataset available: `{summary['sequence_split_tensor_dataset_available']}`",
        f"- Same-student intervention rows available: `{summary['same_student_intervention_rows_available']}`",
        f"- Registered controls: `{summary['registered_control_count']}`",
        f"- Next step: `{summary['selected_next_step']}`",
        "",
        "Full-context contextual-router fields are teacher targets only, not deployable router inputs.",
        "RunPod/Colab validation remains blocked until a local prefix-safe train/heldout pregate passes.",
    ]
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


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


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    args = parser.parse_args(argv)
    summary = run_transformer_acsr_pregate(
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "selected_next_step": summary["selected_next_step"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
