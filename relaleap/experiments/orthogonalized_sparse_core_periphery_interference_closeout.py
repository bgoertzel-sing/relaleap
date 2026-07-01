"""Close out the trained orthogonalized sparse core/periphery pilot."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_PILOT = Path("results/reports/orthogonalized_sparse_core_periphery_interference_pilot/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/orthogonalized_sparse_core_periphery_interference_closeout")

CLOSE_ONE_SITE_BRANCH_ACTION = "close_one_site_orthogonalized_sparse_core_periphery_branch_before_gpu"
MULTISITE_PC_ASSAY_ACTION = "design_multisite_continual_pc_core_periphery_assay_before_gpu"
REPAIR_SOURCES_ACTION = "repair_orthogonalized_sparse_core_periphery_closeout_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "failure_matrix.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_orthogonalized_sparse_core_periphery_interference_closeout(
    *,
    pilot_path: Path = DEFAULT_PILOT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Summarize trained-pilot failures and choose the next local branch."""

    start = time.time()
    pilot = _read_json(pilot_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("orthogonalized_sparse_core_periphery_interference_pilot", pilot_path, pilot),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "present" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}; verdict={strategy['verdict']}"
            ),
        },
    ]
    evidence = _evidence(pilot, strategy)
    failure_matrix = _failure_matrix(evidence)
    source_failures = [row for row in source_rows[:1] if not row["present"]]
    selected_actions = _candidate_actions(evidence, bool(source_failures))
    selected = [row for row in selected_actions if row["disposition"] == "selected"]

    if source_failures or len(selected) != 1:
        status = "fail"
        decision = "orthogonalized_sparse_core_periphery_closeout_failed_closed"
        selected_next_action = REPAIR_SOURCES_ACTION
        next_step = "repair or regenerate the trained pilot summary before choosing a branch"
        claim_status = "closeout_source_artifact_incomplete"
        rationale = "The closeout cannot interpret the branch without the trained pilot artifact."
    else:
        selected_row = selected[0]
        status = "pass"
        decision = "orthogonalized_sparse_core_periphery_branch_closed_or_redirected"
        selected_next_action = selected_row["candidate_action"]
        next_step = selected_row["next_step"]
        claim_status = selected_row["claim_status"]
        rationale = selected_row["reason"]

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_action": selected_next_action,
        "next_step": next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local closeout only; RunPod and Colab remain blocked by trained local observable failures",
        "source_rows": source_rows,
        "evidence": evidence,
        "failure_matrix": failure_matrix,
        "candidate_actions": selected_actions,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(evidence, strategy),
        "failures": source_failures,
        "rationale": rationale,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _evidence(pilot: dict[str, Any], strategy: dict[str, Any]) -> dict[str, Any]:
    arms = pilot.get("arm_metrics")
    arms = arms if isinstance(arms, list) else []
    gates = pilot.get("observable_gates")
    gates = gates if isinstance(gates, list) else []
    candidate = _row_for(arms, "orthogonalized_sparse_additive_core_periphery")
    dense = _row_for(arms, "dense_ridge_residual")
    mlp = _row_for(arms, "random_feature_mlp_residual")
    ablations = [row for row in arms if isinstance(row, dict) and row.get("family") == "mechanism_ablation"]
    ablation_pruning_deltas = [
        _float_or_none(row.get("periphery_first_pruning_delta"))
        for row in ablations
        if _float_or_none(row.get("periphery_first_pruning_delta")) is not None
    ]
    max_ablation_pruning_delta = max(ablation_pruning_deltas) if ablation_pruning_deltas else None
    best_dense_mlp_ce = min(
        value
        for value in (_float_or_none(dense.get("ce")), _float_or_none(mlp.get("ce")))
        if value is not None
    ) if dense and mlp else None
    failed_gates = [str(row.get("criterion")) for row in gates if isinstance(row, dict) and row.get("passed") is False]
    return {
        "pilot_status": pilot.get("status"),
        "pilot_decision": pilot.get("decision"),
        "pilot_scientific_gate": pilot.get("scientific_gate"),
        "training_rows_present": pilot.get("training_rows_present"),
        "synthetic_rows_only": pilot.get("synthetic_rows_only"),
        "candidate_ce": _float_or_none(candidate.get("ce")),
        "best_dense_mlp_ce": best_dense_mlp_ce,
        "candidate_ce_delta_vs_best_dense_mlp": _delta(_float_or_none(candidate.get("ce")), best_dense_mlp_ce),
        "candidate_churn": _float_or_none(candidate.get("functional_churn_flip_rate")),
        "dense_churn": _float_or_none(dense.get("functional_churn_flip_rate")),
        "mlp_churn": _float_or_none(mlp.get("functional_churn_flip_rate")),
        "candidate_retention": _float_or_none(candidate.get("retention_after_sequential_updates")),
        "dense_retention": _float_or_none(dense.get("retention_after_sequential_updates")),
        "mlp_retention": _float_or_none(mlp.get("retention_after_sequential_updates")),
        "candidate_commutator": _float_or_none(candidate.get("finite_update_commutator_symmetric_kl")),
        "dense_commutator": _float_or_none(dense.get("finite_update_commutator_symmetric_kl")),
        "mlp_commutator": _float_or_none(mlp.get("finite_update_commutator_symmetric_kl")),
        "candidate_selectivity": _float_or_none(candidate.get("intervention_selectivity")),
        "candidate_periphery_pruning_delta": _float_or_none(candidate.get("periphery_first_pruning_delta")),
        "max_ablation_periphery_pruning_delta": max_ablation_pruning_delta,
        "failed_observable_gates": failed_gates,
        "ce_or_churn_blocked": "ce_guardrail" in failed_gates or "functional_churn_flip_rate" in failed_gates,
        "mechanism_pruning_blocked": "periphery_first_pruning_delta" in failed_gates,
        "positive_diagnostics": [
            name
            for name in (
                "retention_after_sequential_updates",
                "finite_update_commutator_symmetric_kl",
                "intervention_selectivity",
                "context_reuse_score",
                "null_control_rejection",
            )
            if name not in failed_gates
        ],
        "strategy_verdict": strategy["verdict"],
        "strategy_recommended_next_action": strategy["recommended_next_action"],
        "ben_notification_required": strategy["ben_notification_required"],
    }


def _failure_matrix(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _matrix_row(
            "trained_rows_are_real",
            evidence["training_rows_present"] is True and evidence["synthetic_rows_only"] is False,
            {
                "training_rows_present": evidence["training_rows_present"],
                "synthetic_rows_only": evidence["synthetic_rows_only"],
            },
            "closeout only applies to bounded local CPU training rows",
        ),
        _matrix_row(
            "dense_mlp_quality_guardrail",
            not evidence["ce_or_churn_blocked"],
            {
                "candidate_ce": evidence["candidate_ce"],
                "best_dense_mlp_ce": evidence["best_dense_mlp_ce"],
                "candidate_ce_delta": evidence["candidate_ce_delta_vs_best_dense_mlp"],
                "candidate_churn": evidence["candidate_churn"],
                "dense_churn": evidence["dense_churn"],
                "mlp_churn": evidence["mlp_churn"],
            },
            "candidate must be in the dense/MLP CE band and beat/match dense/MLP churn",
        ),
        _matrix_row(
            "protected_core_periphery_mechanism",
            not evidence["mechanism_pruning_blocked"],
            {
                "candidate_periphery_pruning_delta": evidence["candidate_periphery_pruning_delta"],
                "max_ablation_periphery_pruning_delta": evidence["max_ablation_periphery_pruning_delta"],
            },
            "full candidate must have stronger periphery-first pruning signal than ablations",
        ),
        _matrix_row(
            "interference_diagnostics",
            len(evidence["positive_diagnostics"]) >= 3,
            evidence["positive_diagnostics"],
            "positive local diagnostics may be preserved without promoting the branch",
        ),
        _matrix_row(
            "strategy_review_response",
            True,
            {
                "verdict": evidence["strategy_verdict"],
                "recommended_next_action": evidence["strategy_recommended_next_action"],
            },
            "latest review's trained-pilot recommendation has already been implemented; its radical fallback is now the sensible local next design",
        ),
    ]


def _candidate_actions(evidence: dict[str, Any], source_failed: bool) -> list[dict[str, str]]:
    if source_failed:
        return [
            _candidate(
                REPAIR_SOURCES_ACTION,
                "selected",
                "trained pilot summary is missing or unreadable",
                "repair or regenerate the trained pilot summary",
                "source_artifact_repair_required",
            )
        ]
    branch_blocked = evidence["pilot_scientific_gate"] == "blocked" or evidence["ce_or_churn_blocked"] or evidence["mechanism_pruning_blocked"]
    if branch_blocked:
        return [
            _candidate(
                CLOSE_ONE_SITE_BRANCH_ACTION,
                "closed",
                "one-site orthogonalized sparse core/periphery fails dense/MLP quality and protected-periphery mechanism gates",
                "do not run GPU validation for this branch",
                "one_site_orthogonalized_sparse_core_periphery_closed_no_gpu",
            ),
            _candidate(
                MULTISITE_PC_ASSAY_ACTION,
                "selected",
                (
                    "the pilot retains useful selectivity/retention/commutator diagnostics, but the one-site approximation is dominated by dense/MLP quality controls; "
                    "the next local test should make forgetting, commutators, and causal fingerprints primary in a multi-site continual-learning PC/core-periphery assay"
                ),
                "design the local multi-site continual-learning PC/core-periphery assay with dense/MLP/null controls before any GPU validation",
                "redirect_to_multisite_continual_pc_core_periphery_assay_no_gpu",
            ),
        ]
    return [
        _candidate(
            MULTISITE_PC_ASSAY_ACTION,
            "selected",
            "the one-site pilot did not close the mechanism, but still needs a stronger local repeat before GPU",
            "design the next local repeat with multi-site continual-learning stressors and matched controls",
            "local_repeat_required_before_gpu",
        )
    ]


def _strategy_review_handling(evidence: dict[str, Any], strategy: dict[str, Any]) -> dict[str, Any]:
    return {
        "latest_review_read": strategy["present"],
        "accepted": True,
        "deferred_or_rejected": "",
        "reason": (
            "The review recommended replacing schema rows with trained local rows; that is complete. "
            "Because the trained pilot loses dense/MLP quality and mechanism-pruning gates, the review's radical fallback is selected as a local design step."
        ),
        "ben_should_be_notified": evidence["ben_notification_required"],
    }


def _candidate(action: str, disposition: str, reason: str, next_step: str, claim_status: str) -> dict[str, str]:
    return {
        "candidate_action": action,
        "disposition": disposition,
        "reason": reason,
        "next_step": next_step,
        "claim_status": claim_status,
    }


def _matrix_row(signal: str, passed: bool, actual: Any, expected: str) -> dict[str, Any]:
    return {
        "signal": signal,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
        "failure_reason": "" if passed else expected,
    }


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file() and bool(payload),
        "status": payload.get("status", "missing" if not path.is_file() else ""),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
    }


def _row_for(rows: list[Any], arm: str) -> dict[str, Any]:
    for row in rows:
        if isinstance(row, dict) and row.get("arm") == arm:
            return row
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _strategy_review(path: Path) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "present": path.is_file(),
        "strategic_change_level": None,
        "notify_ben": None,
        "recommended_next_action": None,
        "verdict": None,
    }
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key.strip() in fields:
                fields[key.strip()] = value.strip()
    fields["ben_notification_required"] = (
        str(fields.get("notify_ben")).lower() == "true" or fields.get("strategic_change_level") == "major"
    )
    return fields


def _float_or_none(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(left - right, 6)


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "failure_matrix.csv", summary["failure_matrix"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"], lineterminator="\n")
        writer.writeheader()
        for row in rows or [{"status": "missing"}]:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in writer.fieldnames or []})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _notes(summary: dict[str, Any]) -> str:
    evidence = summary["evidence"]
    return "\n".join(
        [
            "# Orthogonalized Sparse Core/Periphery Interference Closeout",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Claim status: `{summary['claim_status']}`",
            f"- Selected action: `{summary['selected_next_action']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            f"- Candidate CE delta vs best dense/MLP: `{evidence['candidate_ce_delta_vs_best_dense_mlp']}`",
            f"- Failed observable gates: `{', '.join(evidence['failed_observable_gates'])}`",
            f"- Positive diagnostics preserved: `{', '.join(evidence['positive_diagnostics'])}`",
            "",
            "GPU validation remains blocked. The one-site orthogonalized sparse branch is closed as a mechanism claim; the next step is local multi-site continual-learning PC/core-periphery assay design.",
            "",
        ]
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_orthogonalized_sparse_core_periphery_interference_closeout(
        pilot_path=args.pilot,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
