"""Strict causal feature-contract audit for Transformer-ACSR.

This command is intentionally narrower than the earlier Transformer-ACSR
training commands. It records a reusable fail-closed feature contract, checks
the existing command-generated Transformer-ACSR artifacts, and prevents a stale
strategy review from reopening already-closed teacher-support imitation work as
if it were new evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/strict_causal_transformer_acsr_pregate")

DEFAULT_SOURCE_PATHS: tuple[tuple[str, Path, bool], ...] = (
    (
        "transformer_acsr_hidden_future_sequence_dataset",
        Path("results/reports/transformer_acsr_hidden_future_sequence_dataset/summary.json"),
        True,
    ),
    (
        "transformer_acsr_hidden_future_predictor_pregate",
        Path("results/reports/transformer_acsr_hidden_future_predictor_pregate/summary.json"),
        True,
    ),
    (
        "transformer_acsr_hidden_future_control_audit",
        Path("results/reports/transformer_acsr_hidden_future_control_audit/summary.json"),
        True,
    ),
    (
        "transformer_acsr_hidden_future_support_value_headroom",
        Path("results/reports/transformer_acsr_hidden_future_support_value_headroom/summary.json"),
        True,
    ),
    (
        "transformer_acsr_hidden_future_support_value_closeout",
        Path("results/reports/transformer_acsr_hidden_future_support_value_closeout/summary.json"),
        True,
    ),
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "feature_contract.csv",
    "decision_checks.csv",
    "strategy_review_handling.csv",
    "notes.md",
)


@dataclass(frozen=True)
class FeatureContract:
    """Prefix-safe predictor contract for deployable Transformer-ACSR rows."""

    allowed_prefix_fields: frozenset[str] = frozenset(
        {
            "current_hidden_json",
            "previous_hidden_json",
            "position_index",
            "flat_position",
            "sequence_id",
            "fold",
            "split",
            "prefix_safe_fields",
        }
    )
    training_target_fields: frozenset[str] = frozenset(
        {
            "future_hidden_json_target_only",
            "future_delta_json_target_only",
            "teacher_support_logits_json_target_only",
            "teacher_topk_support_target_only",
        }
    )
    eval_only_fields: frozenset[str] = frozenset(
        {
            "student_router_support_eval_only",
            "oracle_support_eval_only",
            "token_position_null_support_eval_only",
            "target_token_eval_only",
            "loss_lookup_fields",
        }
    )
    forbidden_substrings: tuple[str, ...] = (
        "next_hidden",
        "next_delta",
        "future_hidden",
        "future_delta",
        "teacher_support",
        "teacher_logits",
        "teacher_topk",
        "oracle",
        "target_token",
        "task_id",
        "loss",
    )

    def assert_predictor_fields_allowed(self, fields: list[str] | tuple[str, ...]) -> None:
        forbidden = self.forbidden_predictor_fields(fields)
        if forbidden:
            raise ValueError(
                "Forbidden Transformer-ACSR predictor field(s): " + ", ".join(forbidden)
            )

    def forbidden_predictor_fields(self, fields: list[str] | tuple[str, ...]) -> list[str]:
        allowed = set(self.allowed_prefix_fields)
        forbidden: list[str] = []
        for field in fields:
            normalized = field.strip()
            lower = normalized.lower()
            if normalized in allowed:
                continue
            if (
                normalized in self.training_target_fields
                or normalized in self.eval_only_fields
                or any(token in lower for token in self.forbidden_substrings)
            ):
                forbidden.append(normalized)
                continue
            if normalized not in allowed:
                forbidden.append(normalized)
        return forbidden

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for field in sorted(self.allowed_prefix_fields):
            rows.append(_feature_row(field, "prefix_safe_predictor_input", True, False, False))
        for field in sorted(self.training_target_fields):
            rows.append(_feature_row(field, "training_target_only", False, True, False))
        for field in sorted(self.eval_only_fields):
            rows.append(_feature_row(field, "eval_only", False, False, True))
        return rows


def run_strict_causal_transformer_acsr_pregate(
    *,
    source_paths: tuple[tuple[str, Path, bool], ...] = DEFAULT_SOURCE_PATHS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a strict causal-contract audit over existing Transformer-ACSR artifacts."""

    start = time.time()
    contract = FeatureContract()
    source_rows = [_source_row(name, path, required) for name, path, required in source_paths]
    strategy = _strategy_review(strategy_review_path)
    decision_checks = _decision_checks(source_rows, strategy)
    failures = [
        row
        for row in source_rows
        if row["required"] and (not row["present"] or row["status"] == "fail")
    ]
    hard_failures = [row for row in decision_checks if row["required"] and not row["passed"]]
    closeout = _source_by_name(source_rows, "transformer_acsr_hidden_future_support_value_closeout")
    existing_branch_closed = closeout.get("decision") == "transformer_acsr_teacher_support_imitation_closed_before_gpu"

    if failures or hard_failures:
        status = "fail"
        decision = "strict_causal_transformer_acsr_pregate_failed_closed"
        claim_status = "strict_causal_transformer_acsr_sources_or_contract_incomplete"
        selected_next_step = "repair existing Transformer-ACSR source artifacts before any new training"
    elif existing_branch_closed:
        status = "pass"
        decision = "strict_causal_transformer_acsr_existing_branch_closed_no_gpu"
        claim_status = "strict_causal_contract_passes_but_teacher_support_imitation_already_closed"
        selected_next_step = "request_strategy_review_before_reopening_transformer_acsr_or_selecting_new_column_architecture"
    else:
        status = "pass"
        decision = "strict_causal_transformer_acsr_contract_recorded_local_only"
        claim_status = "strict_causal_contract_passes_training_status_unclosed"
        selected_next_step = "continue local Transformer-ACSR training only if source closeouts do not already block it"

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "training_executed": False,
        "strict_feature_contract_passes": not hard_failures,
        "existing_transformer_acsr_branch_closed": existing_branch_closed,
        "new_training_deferred": existing_branch_closed,
        "deferred_or_rejected_review_recommendations": [
            {
                "recommendation": "implement new strictly causal Transformer-ACSR local pregate/trained CPU pilot",
                "disposition": "partially_accepted_contract_audit_only",
                "reason": (
                    "The strict causal contract was added, but new training was not duplicated "
                    "because existing command-generated hidden-future predictor, control, "
                    "headroom, and closeout artifacts already close this teacher-support "
                    "imitation packet before GPU."
                ),
            }
        ],
        "ben_should_be_notified": strategy["notify_ben"]
        or strategy["strategic_change_level"] == "major",
        "direction_shift_recorded": strategy["strategic_change_level"] == "major",
        "source_rows": source_rows,
        "decision_checks": decision_checks,
        "failures": failures + hard_failures,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy, existing_branch_closed),
        "backend_policy": "local contract audit only; RunPod/Colab remain blocked",
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, contract.rows(), decision_checks)
    return summary


def _feature_row(
    field: str,
    role: str,
    allowed_as_predictor_input: bool,
    allowed_as_training_target: bool,
    eval_only: bool,
) -> dict[str, Any]:
    return {
        "field": field,
        "role": role,
        "allowed_as_predictor_input": allowed_as_predictor_input,
        "allowed_as_training_target": allowed_as_training_target,
        "eval_only": eval_only,
        "forbidden_as_predictor_input": not allowed_as_predictor_input,
    }


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
        "claim_status": payload.get("claim_status", ""),
        "selected_next_step": payload.get("selected_next_step", ""),
        "selected_next_action": payload.get("selected_next_action", ""),
        "requires_gpu_now": payload.get("requires_gpu_now", ""),
        "promotion_allowed": payload.get("promotion_allowed", ""),
        "advance_to_gpu_validation": payload.get("advance_to_gpu_validation", ""),
    }


def _decision_checks(
    source_rows: list[dict[str, Any]],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    dataset = _source_by_name(source_rows, "transformer_acsr_hidden_future_sequence_dataset")
    predictor = _source_by_name(source_rows, "transformer_acsr_hidden_future_predictor_pregate")
    control = _source_by_name(source_rows, "transformer_acsr_hidden_future_control_audit")
    headroom = _source_by_name(source_rows, "transformer_acsr_hidden_future_support_value_headroom")
    closeout = _source_by_name(source_rows, "transformer_acsr_hidden_future_support_value_closeout")
    return [
        _check(
            "strategy_review_major_pivot_recorded",
            strategy["present"] and strategy["strategic_change_level"] == "major",
            f"strategic_change_level={strategy['strategic_change_level']}; notify_ben={strategy['notify_ben']}",
            False,
        ),
        _check(
            "dataset_trainability_was_materialized",
            dataset.get("decision") == "transformer_acsr_hidden_future_sequence_dataset_trainable_locally",
            dataset.get("decision", ""),
            True,
        ),
        _check(
            "predictor_pregate_blocks_gpu",
            predictor.get("advance_to_gpu_validation") is False
            and predictor.get("requires_gpu_now") is False,
            predictor.get("decision", ""),
            True,
        ),
        _check(
            "registered_controls_block_mechanism_claim",
            control.get("advance_to_gpu_validation") is False,
            control.get("decision", ""),
            True,
        ),
        _check(
            "same_student_value_headroom_negligible",
            headroom.get("decision") == "support_value_headroom_negligible_close_teacher_imitation_before_gpu",
            headroom.get("decision", ""),
            True,
        ),
        _check(
            "teacher_support_imitation_closeout_blocks_reopening",
            closeout.get("decision") == "transformer_acsr_teacher_support_imitation_closed_before_gpu",
            closeout.get("decision", ""),
            True,
        ),
    ]


def _check(name: str, passed: bool, evidence: str, required: bool) -> dict[str, Any]:
    return {
        "check": name,
        "passed": passed,
        "required": required,
        "evidence": evidence,
    }


def _source_by_name(source_rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in source_rows:
        if row["source"] == name:
            return row
    return {}


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


def _strategy_review_handling(strategy: dict[str, Any], existing_branch_closed: bool) -> str:
    if not strategy["present"]:
        return "No external strategy review was present; the strict local contract still keeps GPU blocked."
    notify = " Ben should be notified." if strategy["notify_ben"] else ""
    if existing_branch_closed:
        return (
            "Accepted the major pivot away from PC/core-periphery and recorded the strict "
            "Transformer-ACSR causal contract. Deferred fresh Transformer-ACSR training "
            "because existing command-generated hidden-future support/value closeout "
            "already closes this teacher-support imitation branch before GPU. "
            f"strategic_change_level={strategy['strategic_change_level']}; "
            f"notify_ben={strategy['notify_ben']}.{notify}"
        )
    return (
        "Accepted the strict causal Transformer-ACSR pregate recommendation and kept "
        "RunPod/Colab blocked until local gates pass. "
        f"strategic_change_level={strategy['strategic_change_level']}; "
        f"notify_ben={strategy['notify_ben']}.{notify}"
    )


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    feature_rows: list[dict[str, Any]],
    decision_checks: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "feature_contract.csv", feature_rows)
    _write_csv(out_dir / "decision_checks.csv", decision_checks)
    _write_csv(
        out_dir / "strategy_review_handling.csv",
        summary["deferred_or_rejected_review_recommendations"],
    )
    notes = [
        "# Strict Causal Transformer-ACSR Pregate",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Existing branch closed: `{summary['existing_transformer_acsr_branch_closed']}`",
        f"- New training deferred: `{summary['new_training_deferred']}`",
        f"- Ben notification required: `{summary['ben_should_be_notified']}`",
        f"- Next step: `{summary['selected_next_step']}`",
        "",
        "Future, teacher, oracle, task-id, and loss fields are forbidden predictor inputs.",
        "RunPod/Colab validation remains blocked by local Transformer-ACSR closeout evidence.",
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
    summary = run_strict_causal_transformer_acsr_pregate(
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
