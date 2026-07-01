"""Design the hidden/future tensor capture packet for Transformer-ACSR.

The support-only Transformer-ACSR branch is closed by local null evidence. This
command records the exact command-driven artifact substrate required before any
new Transformer-ACSR training: prefix-safe hidden inputs, nondeployable teacher
future targets, leakage labels, sequence split keys, and same-student
intervention rows.
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


DEFAULT_CLOSEOUT = Path("results/reports/transformer_acsr_support_predictor_closeout/summary.json")
DEFAULT_SEQUENCE_DATASET = Path("results/reports/transformer_acsr_sequence_dataset/summary.json")
DEFAULT_MISSING_TENSORS = Path("results/reports/transformer_acsr_sequence_dataset/missing_tensor_fields.csv")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/transformer_acsr_hidden_future_capture_design")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "capture_requirements.csv",
    "field_capture_plan.csv",
    "leakage_contract.csv",
    "intervention_contract.csv",
    "notes.md",
)


def run_transformer_acsr_hidden_future_capture_design(
    *,
    closeout_path: Path = DEFAULT_CLOSEOUT,
    sequence_dataset_path: Path = DEFAULT_SEQUENCE_DATASET,
    missing_tensors_path: Path = DEFAULT_MISSING_TENSORS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed design packet for future tensor capture."""

    start = time.time()
    closeout = _read_json(closeout_path)
    sequence_dataset = _read_json(sequence_dataset_path)
    missing_tensors = _read_csv(missing_tensors_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_json("support_predictor_closeout", closeout_path, closeout),
        _source_json("sequence_dataset", sequence_dataset_path, sequence_dataset),
        {
            "source": "missing_tensor_fields",
            "path": str(missing_tensors_path),
            "present": missing_tensors_path.is_file(),
            "status": "read" if missing_tensors_path.is_file() else "missing",
            "decision": f"row_count={len(missing_tensors)}",
            "claim_status": ",".join(row.get("field", "") for row in missing_tensors),
        },
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}; verdict={strategy['verdict']}"
            ),
        },
    ]
    source_failures = [
        {"source": row["source"], "path": row["path"], "reason": "required source artifact missing"}
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]
    support_branch_closed = bool(
        closeout.get("support_branch_closed")
        and closeout.get("hidden_future_capture_required")
        and closeout.get("advance_to_gpu_validation") is False
    )
    support_dataset_available = bool(
        sequence_dataset.get("support_target_dataset_available")
        and sequence_dataset.get("sequence_split_available")
    )
    capture_requirements = _capture_requirements(
        support_branch_closed=support_branch_closed,
        support_dataset_available=support_dataset_available,
        missing_tensors=missing_tensors,
    )
    field_plan = _field_capture_plan()
    leakage_contract = _leakage_contract()
    intervention_contract = _intervention_contract()
    failed_requirements = [
        row for row in capture_requirements if row["required_before_training"] and not row["registered"]
    ]
    design_ready = bool(not source_failures and not failed_requirements and support_branch_closed)
    summary = {
        "status": "pass" if design_ready else "fail",
        "decision": (
            "transformer_acsr_hidden_future_capture_design_recorded"
            if design_ready
            else "transformer_acsr_hidden_future_capture_design_failed_closed"
        ),
        "claim_status": (
            "hidden_future_capture_contract_ready_no_gpu"
            if design_ready
            else "hidden_future_capture_contract_sources_incomplete_no_gpu"
        ),
        "selected_next_step": (
            "extend command-driven teacher artifact capture for hidden/future tensors and exact same-student interventions"
            if design_ready
            else "repair support closeout and sequence-dataset artifacts before capture implementation"
        ),
        "support_branch_closed": support_branch_closed,
        "support_dataset_available": support_dataset_available,
        "capture_design_ready": design_ready,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local design/report only; RunPod and Colab remain blocked",
        "ben_should_be_notified": strategy["ben_notification_required"],
        "direction_shift_recorded": strategy["ben_notification_required"],
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "source_rows": source_rows,
        "capture_requirement_count": len(capture_requirements),
        "field_capture_count": len(field_plan),
        "leakage_contract_count": len(leakage_contract),
        "intervention_contract_count": len(intervention_contract),
        "missing_tensor_fields": [row.get("field", "") for row in missing_tensors],
        "failures": source_failures + failed_requirements,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(
        out_dir=out_dir,
        summary=summary,
        capture_requirements=capture_requirements,
        field_plan=field_plan,
        leakage_contract=leakage_contract,
        intervention_contract=intervention_contract,
    )
    return summary


def _capture_requirements(
    *,
    support_branch_closed: bool,
    support_dataset_available: bool,
    missing_tensors: list[dict[str, str]],
) -> list[dict[str, Any]]:
    missing_fields = {row.get("field", "") for row in missing_tensors}
    return [
        _requirement(
            "support_only_branch_closed",
            support_branch_closed,
            "support-only predictor must be closed before designing new capture",
        ),
        _requirement(
            "sequence_split_support_rows_available",
            support_dataset_available,
            "existing support rows provide split keys and hard support labels",
        ),
        _requirement(
            "prefix_hidden_input_fields_registered",
            {"current_hidden", "previous_hidden"}.issubset(missing_fields),
            "capture current_hidden and previous_hidden as prefix-safe predictor inputs",
        ),
        _requirement(
            "future_teacher_target_fields_registered",
            {"future_hidden", "future_delta"}.issubset(missing_fields),
            "capture future_hidden and future_delta as nondeployable teacher targets only",
        ),
        _requirement(
            "soft_teacher_support_target_registered",
            "teacher_support_logits_or_soft_distribution" in missing_fields,
            "capture teacher logits or softened support distribution for KL/top-k overlap targets",
        ),
        _requirement(
            "same_student_intervention_rows_registered",
            True,
            "capture exact arbitrary-pair same-student forced-support rows for behavior-level gates",
        ),
        _requirement(
            "churn_and_commutator_budget_rows_registered",
            True,
            "capture retention/churn and finite-update commutator rows alongside support interventions",
        ),
    ]


def _requirement(name: str, registered: bool, reason: str) -> dict[str, Any]:
    return {
        "requirement": name,
        "required_before_training": True,
        "registered": registered,
        "status": "registered" if registered else "missing",
        "reason": reason,
    }


def _field_capture_plan() -> list[dict[str, Any]]:
    return [
        _field("sequence_id", "split_key", True, False, False, "stable sequence identifier"),
        _field("fold", "split_key", True, False, False, "train/heldout fold assignment"),
        _field("position_index", "prefix_safe_input", True, False, False, "causal position feature"),
        _field("token_id", "prefix_safe_input", True, False, False, "current token or prefix token id"),
        _field("current_hidden", "prefix_safe_input", True, False, False, "student/base hidden state at current position"),
        _field("previous_hidden", "prefix_safe_input", True, False, False, "previous-position hidden state with BOS mask"),
        _field("past_support_summary", "prefix_safe_input", True, False, False, "past-only support summary; masked at sequence start"),
        _field("future_hidden", "teacher_target", False, True, True, "nondeployable teacher next-hidden chunk target"),
        _field("future_delta", "teacher_target", False, True, True, "nondeployable teacher future-minus-current target"),
        _field("teacher_support_logits", "teacher_target", False, True, True, "softened contextual_mlp support distribution target"),
        _field("teacher_topk_support", "teacher_target", False, True, True, "hard top-k support label for overlap metric"),
        _field("target_token", "eval_only", False, False, True, "evaluation label; forbidden as predictor input"),
        _field("student_loss_fields", "eval_only", False, False, True, "behavior metrics; forbidden as predictor input"),
    ]


def _field(
    field: str,
    role: str,
    prefix_safe: bool,
    nondeployable_teacher_target: bool,
    forbidden_predictor_input: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "field": field,
        "role": role,
        "prefix_safe": prefix_safe,
        "nondeployable_teacher_target": nondeployable_teacher_target,
        "future_or_target_leaking": nondeployable_teacher_target or forbidden_predictor_input,
        "allowed_as_predictor_input": prefix_safe and not forbidden_predictor_input,
        "forbidden_predictor_input": forbidden_predictor_input,
        "reason": reason,
    }


def _leakage_contract() -> list[dict[str, Any]]:
    return [
        {
            "check": "future_targets_never_predictor_inputs",
            "required": True,
            "expected": "future_hidden, future_delta, teacher_support_logits, teacher_topk_support are target-only",
        },
        {
            "check": "target_token_eval_only",
            "required": True,
            "expected": "target_token and loss fields are excluded from predictor feature tensors",
        },
        {
            "check": "sequence_split_no_position_leakage",
            "required": True,
            "expected": "all rows for a sequence_id stay in one split and heldout fold is never used for frequency/null fitting",
        },
        {
            "check": "suffix_perturbation_prefix_invariance",
            "required": True,
            "expected": "prefix features are invariant under suffix-only perturbations up to numerical tolerance",
        },
        {
            "check": "misaligned_delayed_target_nulls",
            "required": True,
            "expected": "shuffled, delayed, and frequency-preserving support targets are generated from train folds only",
        },
    ]


def _intervention_contract() -> list[dict[str, Any]]:
    return [
        {
            "row_family": "same_student_forced_support",
            "required": True,
            "minimum_fields": "sequence_id,position_index,student_id,forced_support_pair,loss,teacher_loss,student_router_loss,oracle_loss",
            "gate_use": "behavior-level support intervention and oracle-regret comparison",
        },
        {
            "row_family": "retention_churn_budget",
            "required": True,
            "minimum_fields": "sequence_id,position_index,anchor_logits_before,anchor_logits_after,functional_churn,residual_norm",
            "gate_use": "interference guardrail",
        },
        {
            "row_family": "finite_update_commutator_budget",
            "required": True,
            "minimum_fields": "sequence_pair,order_ab_loss,order_ba_loss,commutator_delta",
            "gate_use": "finite-update order sensitivity guardrail",
        },
        {
            "row_family": "dense_rank_mlp_controls",
            "required": True,
            "minimum_fields": "control_name,active_params,stored_params,loss,churn,commutator_delta",
            "gate_use": "capacity and dense-control deconfounding",
        },
    ]


def _source_json(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status", "missing") if payload else "missing",
        "decision": payload.get("decision", "") if payload else "",
        "claim_status": payload.get("claim_status", "") if payload else "",
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    result = {
        "present": path.is_file(),
        "strategic_change_level": "",
        "notify_ben": False,
        "recommended_next_action": "",
        "verdict": "",
        "ben_notification_required": False,
    }
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines()[:12]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "strategic_change_level":
            result["strategic_change_level"] = value
        elif key == "notify_ben":
            result["notify_ben"] = value.lower() == "true"
        elif key == "recommended_next_action":
            result["recommended_next_action"] = value
        elif key == "verdict":
            result["verdict"] = value
    result["ben_notification_required"] = bool(
        result["notify_ben"] or str(result["strategic_change_level"]).lower() == "major"
    )
    return result


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No strategy review was present; design relies on command-generated local artifacts."
    if strategy["ben_notification_required"]:
        return "Strategy review requested Ben notification or a major shift; this design records the direction shift."
    return "Strategy review was read and is compatible with local no-GPU Transformer-ACSR capture design."


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_artifacts(
    *,
    out_dir: Path,
    summary: dict[str, Any],
    capture_requirements: list[dict[str, Any]],
    field_plan: list[dict[str, Any]],
    leakage_contract: list[dict[str, Any]],
    intervention_contract: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "capture_requirements.csv", capture_requirements)
    _write_csv(out_dir / "field_capture_plan.csv", field_plan)
    _write_csv(out_dir / "leakage_contract.csv", leakage_contract)
    _write_csv(out_dir / "intervention_contract.csv", intervention_contract)
    notes = [
        "# Transformer-ACSR Hidden/Future Capture Design",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Capture design ready: `{summary['capture_design_ready']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Promotion allowed: `{summary['promotion_allowed']}`",
        f"- Next step: `{summary['selected_next_step']}`",
        "",
        "This is a capture contract only. It does not train a predictor and does not",
        "advance any Transformer-ACSR claim to GPU validation.",
        "",
        "RunPod and Colab validation remain blocked until the command-driven capture",
        "exists and a local prefix-safe mechanism gate beats registered nulls and controls.",
    ]
    if summary["ben_should_be_notified"]:
        notes.append("")
        notes.append("The latest strategy review requested Ben notification or recorded a major direction shift.")
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
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closeout", type=Path, default=DEFAULT_CLOSEOUT)
    parser.add_argument("--sequence-dataset", type=Path, default=DEFAULT_SEQUENCE_DATASET)
    parser.add_argument("--missing-tensors", type=Path, default=DEFAULT_MISSING_TENSORS)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_transformer_acsr_hidden_future_capture_design(
        closeout_path=args.closeout,
        sequence_dataset_path=args.sequence_dataset,
        missing_tensors_path=args.missing_tensors,
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
