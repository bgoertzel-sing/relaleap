"""Decision/gate report for the causal contextual support-router candidate."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_SEQUENCE_REPORT = Path(
    "results/reports/token_larger_contextual_router_sequence_kfold_ablation/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/token_larger_causal_contextual_router_gate")
DEFAULT_FUTURE_PERTURBATION_REPORT = Path(
    "results/reports/causal_router_future_perturbation/summary.json"
)

CAUSAL_GATE_PREREGISTERED = "causal_contextual_router_gate_preregistered"
CAUSAL_LOCAL_GATE_PASSED = "causal_contextual_router_local_gate_passed"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
SELECTED_NEXT_ACTION = "causal_router_future_perturbation_test"
RUNPOD_REPEAT_ACTION = "runpod_repeat_matrix_now"

FULL_CONTEXT_VARIANT = "promoted_contextual_topk2:actual_full_context"
FULL_CONTEXT_CAUSAL_VIEW = "promoted_contextual_topk2:causal_current_past_position"
CAUSAL_VARIANT = "causal_contextual_topk2:actual_causal_context"
LINEAR_VARIANT = "linear_topk2_control:linear_actual"
CAUSAL_NO_POSITION_VARIANT = "causal_contextual_topk2:current_past_no_position"
POSITION_ONLY_VARIANT = "causal_contextual_topk2:position_only"

FUTURE_CONTEXT_MATERIAL_DELTA = 0.01
MIN_FOLD_WIN_FRACTION = 0.75
MAX_POSITION_ONLY_RELATIVE_TO_CAUSAL_DELTA = 0.25

_CLAIM_STATUSES = {
    "contextual_mlp": "nondeployable_full_context_oracle_diagnostic_only",
    "contextual_mlp_causal": "local_sequence_holdout_candidate_not_promoted",
    "promoted_default": "blocked_pending_causal_feature_safe_gate",
    "autoregressive_deployable_router": "not_established_for_full_context_contextual_mlp",
    "gpu_repeat": "deferred_until_local_causal_gate_passes",
}

_LOCAL_GATE_CRITERIA = (
    {
        "criterion": "future_context_safeguard",
        "threshold": f"full-context trained router future-feature ablation delta > {FUTURE_CONTEXT_MATERIAL_DELTA}",
        "purpose": "keeps contextual_mlp classified as oracle-only when future features are material",
    },
    {
        "criterion": "causal_beats_linear_mean_ce",
        "threshold": "causal contextual top-k-2 mean sequence-heldout CE <= linear top-k-2 control",
        "purpose": "requires deployable causal candidate to clear the linear router control",
    },
    {
        "criterion": "causal_fold_consistency",
        "threshold": f"causal contextual beats linear on at least {MIN_FOLD_WIN_FRACTION:.0%} of folds",
        "purpose": "blocks one-fold mean-only wins",
    },
    {
        "criterion": "support_utilization_not_collapsed",
        "threshold": "causal used-column count and unique support-set count >= linear control means",
        "purpose": "requires the causal router to preserve the support-routing signal",
    },
    {
        "criterion": "position_shortcut_not_sufficient",
        "threshold": f"position-only mean CE is at least {MAX_POSITION_ONLY_RELATIVE_TO_CAUSAL_DELTA} worse than actual causal features",
        "purpose": "guards against a position-only shortcut interpretation",
    },
    {
        "criterion": "future_perturbation_invariance",
        "threshold": "future-position perturbations do not change earlier causal-router scores/support",
        "purpose": "must be added as a unit test before RunPod/Colab validation",
    },
)

_CANDIDATE_ACTIONS = (
    {
        "candidate_action": SELECTED_NEXT_ACTION,
        "disposition": "selected",
        "reason": "the K-fold report is supportive, but the causal-router invariance test is the missing fail-closed deployability safeguard before GPU repeats",
    },
    {
        "candidate_action": RUNPOD_REPEAT_ACTION,
        "disposition": "deferred",
        "reason": "GPU validation is deferred until the local causal gate includes future-perturbation invariance",
    },
    {
        "candidate_action": "promote_contextual_mlp_causal_default",
        "disposition": "disqualified",
        "reason": "one local sequence-heldout report is not enough for a default change",
    },
    {
        "candidate_action": "restore_full_context_contextual_mlp_deployable_claim",
        "disposition": "disqualified",
        "reason": "future-token feature groups are material, so full-context contextual_mlp is oracle-style only",
    },
)


def run_causal_contextual_router_gate_report(
    *,
    sequence_report_path: Path = DEFAULT_SEQUENCE_REPORT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    future_perturbation_report_path: Path = DEFAULT_FUTURE_PERTURBATION_REPORT,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Consume K-fold evidence and pre-register the local causal-router gate."""

    start = time.time()
    sequence = _read_json_object(sequence_report_path)
    future_perturbation = _read_json_object(future_perturbation_report_path)
    strategy_review = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("contextual_router_sequence_kfold_ablation", sequence_report_path, sequence),
        _source_row(
            "causal_router_future_perturbation",
            future_perturbation_report_path,
            future_perturbation,
        ),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy_review["present"],
            "status": "present" if strategy_review["present"] else "missing_optional",
            "decision": strategy_review["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy_review['strategic_change_level']}; "
                f"notify_ben={strategy_review['notify_ben']}"
            ),
        },
    ]
    evidence = _evidence_snapshot(sequence, future_perturbation)
    gate = _evaluate_gate(evidence)
    failures = _failures(source_rows=source_rows, sequence=sequence, evidence=evidence)
    candidate_actions = _candidate_actions(gate)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        selected_next_action = None
        next_step = "repair missing or inconsistent causal contextual-router K-fold evidence"
        rationale = (
            "The causal contextual-router gate cannot be pre-registered because "
            "required sequence-heldout evidence is missing or internally inconsistent."
        )
    else:
        status = "pass"
        if gate["passes_full_local_gate"]:
            decision = CAUSAL_LOCAL_GATE_PASSED
            selected_next_action = RUNPOD_REPEAT_ACTION
            next_step = (
                "run the RunPod causal-router repeat matrix before any default-promotion work"
            )
            rationale = (
                "The full-context contextual router is classified as a nondeployable "
                "oracle-style baseline because future features are material. The causal "
                "contextual router now passes the local sequence-heldout criteria and "
                "the fail-closed future-perturbation invariance check, so the next "
                "bounded scientific step is GPU repeat validation."
            )
        else:
            decision = CAUSAL_GATE_PREREGISTERED
            selected_next_action = SELECTED_NEXT_ACTION
            next_step = (
                "add a causal-router future-perturbation invariance unit test before any "
                "RunPod repeat or default-promotion work"
            )
            rationale = (
                "The full-context contextual router is now classified as a nondeployable "
                "oracle-style baseline because future features are material. The causal "
                "contextual router is a strong local candidate versus the linear control, "
                "but default promotion and GPU repeats remain gated on a fail-closed "
                "causal-invariance test plus the recorded sequence-heldout criteria."
            )

    summary = {
        "status": status,
        "decision": decision,
        "selected_next_action": selected_next_action,
        "next_step": next_step,
        "next_command": None,
        "claim_statuses": _claim_statuses(gate),
        "local_gate_criteria": list(_LOCAL_GATE_CRITERIA),
        "local_gate_status": gate,
        "candidate_actions": candidate_actions,
        "source_rows": source_rows,
        "evidence": evidence,
        "strategy_review": strategy_review,
        "failures": failures,
        "rationale": rationale,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "gate_criteria_csv": str(out_dir / "gate_criteria.csv"),
            "candidate_actions_csv": str(out_dir / "candidate_actions.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        out_dir / "source_rows.csv",
        ["source", "path", "present", "status", "decision", "claim_status"],
        source_rows,
    )
    _write_csv(
        out_dir / "gate_criteria.csv",
        ["criterion", "threshold", "purpose"],
        list(_LOCAL_GATE_CRITERIA),
    )
    _write_csv(
        out_dir / "candidate_actions.csv",
        ["candidate_action", "disposition", "reason"],
        candidate_actions,
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _claim_statuses(gate: dict[str, Any]) -> dict[str, str]:
    claim_statuses = dict(_CLAIM_STATUSES)
    if gate.get("passes_full_local_gate"):
        claim_statuses["gpu_repeat"] = "unlocked_pending_runpod_validation"
    return claim_statuses


def _candidate_actions(gate: dict[str, Any]) -> list[dict[str, str]]:
    if not gate.get("passes_full_local_gate"):
        return list(_CANDIDATE_ACTIONS)
    return [
        {
            "candidate_action": SELECTED_NEXT_ACTION,
            "disposition": "completed",
            "reason": "future-perturbation invariance is now present and passing in the local evidence artifact",
        },
        {
            "candidate_action": RUNPOD_REPEAT_ACTION,
            "disposition": "selected",
            "reason": "the fail-closed local causal gate now passes, so GPU repeat validation is the next evidence step",
        },
        {
            "candidate_action": "promote_contextual_mlp_causal_default",
            "disposition": "disqualified",
            "reason": "default promotion remains blocked until GPU repeats and follow-up causal support audits pass",
        },
        {
            "candidate_action": "restore_full_context_contextual_mlp_deployable_claim",
            "disposition": "disqualified",
            "reason": "future-token feature groups are material, so full-context contextual_mlp is oracle-style only",
        },
    ]


def _evidence_snapshot(
    sequence: dict[str, Any], future_perturbation: dict[str, Any]
) -> dict[str, Any]:
    ablation = sequence.get("ablation", {}) if isinstance(sequence.get("ablation"), dict) else {}
    variants = ablation.get("variants", {}) if isinstance(ablation.get("variants"), dict) else {}
    comparisons = (
        ablation.get("key_comparisons", {})
        if isinstance(ablation.get("key_comparisons"), dict)
        else {}
    )
    causal_vs_linear = comparisons.get("causal_contextual_vs_linear", {})
    full_vs_linear = comparisons.get("full_context_oracle_baseline_vs_linear", {})
    causal_vs_full = comparisons.get("causal_contextual_vs_full_context_oracle_baseline", {})
    causal = variants.get(CAUSAL_VARIANT, {})
    linear = variants.get(LINEAR_VARIANT, {})
    full = variants.get(FULL_CONTEXT_VARIANT, {})
    full_causal_view = variants.get(FULL_CONTEXT_CAUSAL_VIEW, {})
    no_position = variants.get(CAUSAL_NO_POSITION_VARIANT, {})
    position_only = variants.get(POSITION_ONLY_VARIANT, {})
    return {
        "sequence_status": sequence.get("status"),
        "sequence_decision": sequence.get("decision"),
        "sequence_claim_status": sequence.get("claim_status"),
        "fold_count": ablation.get("fold_count"),
        "future_context_material_loss_delta": ablation.get("future_context_material_loss_delta"),
        "promoted_vs_linear_loss_delta": ablation.get("promoted_vs_linear_loss_delta"),
        "causal_contextual_vs_linear_loss_delta": ablation.get(
            "causal_contextual_vs_linear_loss_delta"
        ),
        "causal_contextual_vs_promoted_full_loss_delta": ablation.get(
            "causal_contextual_vs_promoted_full_loss_delta"
        ),
        "full_context_mean_ce": full.get("mean_router_loss"),
        "full_context_causal_view_mean_ce": full_causal_view.get("mean_router_loss"),
        "full_context_uses_future_context": full.get("uses_future_context"),
        "causal_mean_ce": causal.get("mean_router_loss"),
        "causal_oracle_gap": causal.get("mean_router_oracle_gap"),
        "causal_feature_safe": causal.get("causal_feature_safe"),
        "causal_mean_used_columns": causal.get("mean_used_columns"),
        "causal_mean_unique_support_sets": causal.get("mean_unique_support_sets"),
        "linear_mean_ce": linear.get("mean_router_loss"),
        "linear_oracle_gap": linear.get("mean_router_oracle_gap"),
        "linear_mean_used_columns": linear.get("mean_used_columns"),
        "linear_mean_unique_support_sets": linear.get("mean_unique_support_sets"),
        "causal_current_past_no_position_mean_ce": no_position.get("mean_router_loss"),
        "causal_position_only_mean_ce": position_only.get("mean_router_loss"),
        "causal_vs_linear_fold_count": causal_vs_linear.get("fold_count"),
        "causal_vs_linear_left_wins": causal_vs_linear.get("left_wins"),
        "causal_vs_linear_right_wins": causal_vs_linear.get("right_wins"),
        "causal_vs_linear_fold_deltas": [
            row.get("loss_delta") for row in causal_vs_linear.get("fold_deltas", [])
        ],
        "full_context_vs_linear_mean_delta": full_vs_linear.get("mean_loss_delta"),
        "causal_vs_full_mean_delta": causal_vs_full.get("mean_loss_delta"),
        "future_perturbation_status": future_perturbation.get("status"),
        "future_perturbation_decision": future_perturbation.get("decision"),
        "future_perturbation_claim_status": future_perturbation.get("claim_status"),
        "future_perturbation_invariance": future_perturbation.get(
            "future_perturbation_invariance"
        ),
    }


def _evaluate_gate(evidence: dict[str, Any]) -> dict[str, Any]:
    fold_count = _number(evidence.get("causal_vs_linear_fold_count"))
    left_wins = _number(evidence.get("causal_vs_linear_left_wins"))
    causal_mean = _number(evidence.get("causal_mean_ce"))
    position_mean = _number(evidence.get("causal_position_only_mean_ce"))
    criteria = {
        "future_context_safeguard": _gt(
            evidence.get("future_context_material_loss_delta"),
            FUTURE_CONTEXT_MATERIAL_DELTA,
        ),
        "causal_beats_linear_mean_ce": _le(
            evidence.get("causal_contextual_vs_linear_loss_delta"),
            0.0,
        ),
        "causal_fold_consistency": (
            fold_count is not None
            and fold_count > 0
            and left_wins is not None
            and (left_wins / fold_count) >= MIN_FOLD_WIN_FRACTION
        ),
        "support_utilization_not_collapsed": (
            _ge(evidence.get("causal_mean_used_columns"), evidence.get("linear_mean_used_columns"))
            and _ge(
                evidence.get("causal_mean_unique_support_sets"),
                evidence.get("linear_mean_unique_support_sets"),
            )
        ),
        "position_shortcut_not_sufficient": (
            causal_mean is not None
            and position_mean is not None
            and position_mean - causal_mean >= MAX_POSITION_ONLY_RELATIVE_TO_CAUSAL_DELTA
        ),
        "future_perturbation_invariance": evidence.get("future_perturbation_invariance") is True,
    }
    passed_without_perturbation = all(
        value for key, value in criteria.items() if key != "future_perturbation_invariance"
    )
    return {
        "criteria": criteria,
        "passed_without_future_perturbation_test": passed_without_perturbation,
        "passes_full_local_gate": all(criteria.values()),
        "missing_required_next_check": (
            "future_perturbation_invariance"
            if passed_without_perturbation and not criteria["future_perturbation_invariance"]
            else None
        ),
    }


def _failures(
    *,
    source_rows: list[dict[str, Any]],
    sequence: dict[str, Any],
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if not source_rows[0]["present"]:
        failures.append(
            {
                "source": "contextual_router_sequence_kfold_ablation",
                "field": "source_artifact",
                "expected": "file exists",
                "actual": "missing",
                "path": source_rows[0]["path"],
            }
        )
    if source_rows[1]["present"]:
        if evidence.get("future_perturbation_status") != "pass":
            failures.append(
                {
                    "source": "causal_router_future_perturbation",
                    "field": "future_perturbation_status",
                    "expected": "pass",
                    "actual": evidence.get("future_perturbation_status"),
                }
            )
        if evidence.get("future_perturbation_invariance") is not True:
            failures.append(
                {
                    "source": "causal_router_future_perturbation",
                    "field": "future_perturbation_invariance",
                    "expected": True,
                    "actual": evidence.get("future_perturbation_invariance"),
                }
            )
    expected = {
        "sequence_status": "ok",
        "sequence_decision": "causal_contextual_router_sequence_holdout_candidate",
        "sequence_claim_status": "causal_feature_safe_router_local_sequence_holdout_supported",
    }
    for field, expected_value in expected.items():
        if evidence.get(field) != expected_value:
            failures.append(
                {
                    "source": "contextual_router_sequence_kfold_ablation",
                    "field": field,
                    "expected": expected_value,
                    "actual": evidence.get(field),
                }
            )
    required_variants = (
        FULL_CONTEXT_VARIANT,
        FULL_CONTEXT_CAUSAL_VIEW,
        CAUSAL_VARIANT,
        LINEAR_VARIANT,
        CAUSAL_NO_POSITION_VARIANT,
        POSITION_ONLY_VARIANT,
    )
    variants = (sequence.get("ablation", {}) or {}).get("variants", {})
    for variant in required_variants:
        if variant not in variants:
            failures.append(
                {
                    "source": "contextual_router_sequence_kfold_ablation",
                    "field": "variant",
                    "expected": variant,
                    "actual": "missing",
                }
            )
    if evidence.get("full_context_uses_future_context") is not True:
        failures.append(
            {
                "source": "contextual_router_sequence_kfold_ablation",
                "field": "full_context_uses_future_context",
                "expected": True,
                "actual": evidence.get("full_context_uses_future_context"),
            }
        )
    if evidence.get("causal_feature_safe") is not True:
        failures.append(
            {
                "source": "contextual_router_sequence_kfold_ablation",
                "field": "causal_feature_safe",
                "expected": True,
                "actual": evidence.get("causal_feature_safe"),
            }
        )
    return failures


def _source_row(source: str, path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": packet.get("status"),
        "decision": packet.get("decision"),
        "claim_status": packet.get("claim_status") or packet.get("claim_policy"),
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "strategic_change_level": None,
            "notify_ben": None,
            "recommended_next_action": None,
            "incorporation": "optional review not present",
            "ben_notification_required": False,
        }
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:12]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() in {
            "strategic_change_level",
            "notify_ben",
            "recommended_next_action",
        }:
            header[key.strip()] = value.strip()
    notify_ben = _bool_or_none(header.get("notify_ben"))
    major = header.get("strategic_change_level") == "major"
    return {
        "path": str(path),
        "present": True,
        "strategic_change_level": header.get("strategic_change_level"),
        "notify_ben": notify_ben,
        "recommended_next_action": header.get("recommended_next_action"),
        "incorporation": (
            "accepted: reclassified full-context contextual_mlp as a "
            "nondeployable oracle diagnostic, kept contextual_mlp_causal as a "
            "local candidate only, and pre-registered a fail-closed local causal "
            "gate before GPU repeat or default promotion"
        ),
        "ben_notification_required": bool(notify_ben) or major,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _bool_or_none(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gt(left: Any, right: Any) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    return left_number is not None and right_number is not None and left_number > right_number


def _ge(left: Any, right: Any) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    return left_number is not None and right_number is not None and left_number >= right_number


def _le(left: Any, right: Any) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    return left_number is not None and right_number is not None and left_number <= right_number


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    evidence = summary["evidence"]
    gate = summary["local_gate_status"]
    lines = [
        "# Causal Contextual Router Gate Report",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Next step: `{summary['next_step']}`",
        f"- Ben notification required: `{summary['strategy_review']['ben_notification_required']}`",
        "",
        "## Claim Statuses",
    ]
    for name, status in summary["claim_statuses"].items():
        lines.append(f"- {name}: `{status}`")
    lines.extend(
        [
            "",
            "## Evidence",
            f"- Future-context material loss delta: `{evidence['future_context_material_loss_delta']}`",
            f"- Full-context mean CE: `{evidence['full_context_mean_ce']}`",
            f"- Causal contextual mean CE: `{evidence['causal_mean_ce']}`",
            f"- Linear mean CE: `{evidence['linear_mean_ce']}`",
            f"- Causal-vs-linear fold deltas: `{evidence['causal_vs_linear_fold_deltas']}`",
            f"- Future perturbation invariance: `{evidence['future_perturbation_invariance']}`",
            f"- Causal used columns / unique support sets: `{evidence['causal_mean_used_columns']}` / `{evidence['causal_mean_unique_support_sets']}`",
            f"- Linear used columns / unique support sets: `{evidence['linear_mean_used_columns']}` / `{evidence['linear_mean_unique_support_sets']}`",
            "",
            "## Local Gate",
        ]
    )
    for name, passed in gate["criteria"].items():
        lines.append(f"- {name}: `{passed}`")
    lines.extend(
        [
            f"- Passed without future perturbation test: `{gate['passed_without_future_perturbation_test']}`",
            f"- Passes full local gate: `{gate['passes_full_local_gate']}`",
            "",
            summary["rationale"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-report", type=Path, default=DEFAULT_SEQUENCE_REPORT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument(
        "--future-perturbation-report",
        type=Path,
        default=DEFAULT_FUTURE_PERTURBATION_REPORT,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_causal_contextual_router_gate_report(
        sequence_report_path=args.sequence_report,
        strategy_review_path=args.strategy_review,
        future_perturbation_report_path=args.future_perturbation_report,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
