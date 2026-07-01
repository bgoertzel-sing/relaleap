"""Close out the local multi-site PC/core-periphery assay."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_ASSAY_DIR = Path("results/reports/multisite_continual_pc_core_periphery_assay")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/multisite_continual_pc_core_periphery_closeout")

REPAIR_ACTION = "repair_multisite_pc_core_periphery_closeout_sources"
CLOSE_BRANCH_ACTION = "close_multisite_pc_core_periphery_branch_before_gpu"
REQUEST_STRATEGY_ACTION = "request_strategy_review_before_new_column_architecture"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "failure_matrix.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_multisite_continual_pc_core_periphery_closeout(
    *,
    assay_dir: Path = DEFAULT_ASSAY_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Consume the trained multi-site assay and write a conservative closeout."""

    start = time.time()
    summary_path = assay_dir / "summary.json"
    gate_path = assay_dir / "gate_criteria.csv"
    arm_path = assay_dir / "arm_metrics.csv"
    assay = _read_json(summary_path)
    gates = _read_csv(gate_path)
    arms = _read_csv(arm_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source("multisite_assay_summary", summary_path, assay, 1 if assay else 0),
        _source("multisite_assay_gate_criteria", gate_path, {}, len(gates)),
        _source("multisite_assay_arm_metrics", arm_path, {}, len(arms)),
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
            "row_count": 1 if strategy["present"] else 0,
        },
    ]
    evidence = _evidence(assay, gates, arms, strategy)
    failure_matrix = _failure_matrix(evidence)
    required_failures = _source_failures(source_rows) + [
        row for row in failure_matrix if row["required"] and not row["passed"]
    ]
    candidate_actions = _candidate_actions(evidence, bool(required_failures))
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if required_failures or len(selected) != 1:
        status = "fail"
        decision = "multisite_pc_core_periphery_closeout_failed_closed"
        selected_next_action = REPAIR_ACTION
        claim_status = "multisite_pc_core_periphery_closeout_sources_incomplete_or_inconsistent"
        selected_next_step = "repair or regenerate the multi-site assay artifacts, then rerun this closeout"
        rationale = "The closeout cannot choose a scientific branch until required source artifacts and hard gates are coherent."
    else:
        selected_row = selected[0]
        status = "pass"
        decision = "multisite_pc_core_periphery_branch_closed"
        selected_next_action = selected_row["candidate_action"]
        claim_status = selected_row["claim_status"]
        selected_next_step = selected_row["next_step"]
        rationale = selected_row["reason"]

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_action": selected_next_action,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local closeout only; RunPod and Colab remain blocked by local dense/MLP/null gate failures",
        "source_rows": source_rows,
        "evidence": evidence,
        "failure_matrix": failure_matrix,
        "candidate_actions": candidate_actions,
        "failures": required_failures,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "rationale": rationale,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _evidence(
    assay: dict[str, Any],
    gates: list[dict[str, str]],
    arms: list[dict[str, str]],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    candidate = _row_for(arms, "multisite_pc_core_periphery_candidate")
    mlp = _row_for(arms, "parameter_matched_mlp_residual_control")
    dense = _row_for(arms, "dense_rank_norm_residual_control")
    low_rank = _row_for(arms, "low_rank_residual_control")
    failed_claim_gates = [
        row.get("criterion", "")
        for row in gates
        if row.get("severity") == "claim" and str(row.get("passed")).lower() != "true"
    ]
    failed_hard_gates = [
        row.get("criterion", "")
        for row in gates
        if row.get("severity") == "hard" and str(row.get("passed")).lower() != "true"
    ]
    nulls_beating_candidate = _parse_list_field(_gate_actual(gates, "leakage_null_rejection"))
    best_dense_mlp_ce = min(
        value
        for value in (_float(dense.get("heldout_ce")), _float(mlp.get("heldout_ce")))
        if value is not None
    ) if dense and mlp else None
    best_dense_mlp_churn = min(
        value
        for value in (_float(dense.get("mean_functional_flip_churn")), _float(mlp.get("mean_functional_flip_churn")))
        if value is not None
    ) if dense and mlp else None
    best_dense_mlp_commutator = min(
        value
        for value in (_float(dense.get("finite_update_commutator")), _float(mlp.get("finite_update_commutator")))
        if value is not None
    ) if dense and mlp else None
    return {
        "assay_status": assay.get("status"),
        "assay_decision": assay.get("decision"),
        "assay_scientific_gate": assay.get("scientific_gate"),
        "training_rows_present": assay.get("training_rows_present"),
        "advance_to_gpu_validation": assay.get("advance_to_gpu_validation"),
        "promotion_allowed": assay.get("promotion_allowed"),
        "candidate_heldout_ce": _float(candidate.get("heldout_ce")),
        "candidate_churn": _float(candidate.get("mean_functional_flip_churn")),
        "candidate_commutator": _float(candidate.get("finite_update_commutator")),
        "candidate_retention": _float(candidate.get("cross_site_retention")),
        "candidate_selectivity": _float(candidate.get("causal_intervention_fingerprint")),
        "candidate_periphery_pruning_delta": _float(candidate.get("periphery_first_pruning_delta")),
        "best_dense_mlp_ce": best_dense_mlp_ce,
        "best_dense_mlp_churn": best_dense_mlp_churn,
        "best_dense_mlp_commutator": best_dense_mlp_commutator,
        "low_rank_heldout_ce": _float(low_rank.get("heldout_ce")),
        "candidate_minus_mlp_heldout_ce": _nested(assay, "primary_result", "candidate_minus_mlp_heldout_ce"),
        "candidate_minus_mlp_churn": _nested(assay, "primary_result", "candidate_minus_mlp_churn"),
        "candidate_minus_mlp_commutator": _nested(assay, "primary_result", "candidate_minus_mlp_commutator"),
        "failed_claim_gates": failed_claim_gates,
        "failed_hard_gates": failed_hard_gates,
        "nulls_matching_or_beating_candidate": nulls_beating_candidate,
        "only_positive_candidate_signal": (
            "candidate beats MLP commutator but fails best dense/MLP commutator gate"
            if _float(candidate.get("finite_update_commutator")) is not None
            and _float(mlp.get("finite_update_commutator")) is not None
            and _float(candidate.get("finite_update_commutator")) < _float(mlp.get("finite_update_commutator"))
            else "none"
        ),
        "strategy_verdict": strategy["verdict"],
        "strategy_recommended_next_action": strategy["recommended_next_action"],
        "ben_notification_required": strategy["ben_notification_required"],
    }


def _failure_matrix(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _matrix_row(
            "assay_completed_with_real_training_rows",
            (
                evidence["assay_status"] == "pass"
                and evidence["training_rows_present"] is True
                and evidence["advance_to_gpu_validation"] is False
                and evidence["promotion_allowed"] is False
                and not evidence["failed_hard_gates"]
            ),
            {
                "assay_status": evidence["assay_status"],
                "training_rows_present": evidence["training_rows_present"],
                "failed_hard_gates": evidence["failed_hard_gates"],
            },
            "required assay artifacts must be complete, trained, and locally gated",
            True,
        ),
        _matrix_row(
            "dense_mlp_quality_and_churn_guardrails",
            (
                "heldout_ce_guardrail" not in evidence["failed_claim_gates"]
                and "functional_churn_no_worse_than_dense_mlp" not in evidence["failed_claim_gates"]
            ),
            {
                "candidate_heldout_ce": evidence["candidate_heldout_ce"],
                "best_dense_mlp_ce": evidence["best_dense_mlp_ce"],
                "candidate_churn": evidence["candidate_churn"],
                "best_dense_mlp_churn": evidence["best_dense_mlp_churn"],
            },
            "candidate must stay inside dense/MLP quality and churn guardrails before any redesign or GPU claim",
            False,
        ),
        _matrix_row(
            "retention_and_null_rejection",
            (
                "cross_site_retention_positive" not in evidence["failed_claim_gates"]
                and "leakage_null_rejection" not in evidence["failed_claim_gates"]
            ),
            {
                "candidate_retention": evidence["candidate_retention"],
                "nulls_matching_or_beating_candidate": evidence["nulls_matching_or_beating_candidate"],
            },
            "candidate must retain across hidden sites and beat mechanism/support/leakage nulls",
            False,
        ),
        _matrix_row(
            "commutator_signal_is_sufficient",
            "commutator_no_worse_than_dense_mlp" not in evidence["failed_claim_gates"],
            {
                "candidate_commutator": evidence["candidate_commutator"],
                "best_dense_mlp_commutator": evidence["best_dense_mlp_commutator"],
                "candidate_minus_mlp_commutator": evidence["candidate_minus_mlp_commutator"],
                "only_positive_candidate_signal": evidence["only_positive_candidate_signal"],
            },
            "an isolated MLP-relative commutator win is insufficient when low-rank/dense controls and other gates fail",
            False,
        ),
        _matrix_row(
            "gpu_validation_remains_blocked",
            True,
            {"requires_gpu_now": False, "advance_to_gpu_validation": False},
            "RunPod and Colab should remain unused until a local mechanism gate passes",
            True,
        ),
    ]


def _candidate_actions(evidence: dict[str, Any], required_failed: bool) -> list[dict[str, str]]:
    if required_failed:
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "required assay sources or hard gates are incomplete",
                "repair or regenerate the multi-site assay artifacts",
                "source_artifact_repair_required",
            ),
            _candidate(
                CLOSE_BRANCH_ACTION,
                "blocked",
                "scientific closeout requires coherent trained assay sources",
                "rerun after source repair",
                "source_artifact_repair_required",
            ),
        ]
    local_claim_failed = bool(evidence["failed_claim_gates"])
    if local_claim_failed:
        return [
            _candidate(
                CLOSE_BRANCH_ACTION,
                "selected",
                (
                    "the trained multi-site candidate fails dense/MLP CE, churn, retention, commutator, and null-rejection claim gates; "
                    "the isolated MLP-relative commutator improvement is not enough to justify redesign-by-retuning or GPU validation"
                ),
                "close the current multi-site PC/core-periphery branch and request/choose a new local mechanism direction before GPU",
                "multisite_pc_core_periphery_closed_no_gpu",
            ),
            _candidate(
                REQUEST_STRATEGY_ACTION,
                "next_after_closeout",
                "new architecture work should be strategy-selected rather than another local threshold retune of this failed branch",
                "run an external strategy review if no newer human direction or branch selector exists",
                "strategy_review_recommended_before_new_architecture",
            ),
            _candidate(
                "run_runpod_multisite_pc_core_periphery_validation",
                "rejected",
                "local trained gates are negative and explicitly block promotion",
                "do not run RunPod or Colab for this branch",
                "gpu_validation_blocked",
            ),
            _candidate(
                "retune_multisite_pc_core_periphery_candidate",
                "rejected",
                "multiple independent local failures and null matches indicate mechanism failure rather than a single hyperparameter miss",
                "only reopen if a later strategy review specifies a materially different mechanism",
                "local_retune_rejected",
            ),
        ]
    return [
        _candidate(
            REQUEST_STRATEGY_ACTION,
            "selected",
            "local claim gates did not fail, but the local assay cannot promote itself",
            "request strategy review or repeat with a second seed before GPU",
            "local_repeat_or_strategy_review_required",
        )
    ]


def _source(source: str, path: Path, payload: dict[str, Any], row_count: int) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": "present" if path.is_file() else "missing",
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
        "row_count": row_count,
    }


def _source_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows[:3] if not row["present"] or int(row.get("row_count", 0)) < 1]


def _matrix_row(signal: str, passed: bool, actual: Any, expected: str, required: bool) -> dict[str, Any]:
    return {
        "signal": signal,
        "required": required,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }


def _candidate(action: str, disposition: str, reason: str, next_step: str, claim_status: str) -> dict[str, str]:
    return {
        "candidate_action": action,
        "disposition": disposition,
        "reason": reason,
        "next_step": next_step,
        "claim_status": claim_status,
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "present": False,
            "strategic_change_level": "",
            "notify_ben": False,
            "recommended_next_action": "",
            "verdict": "",
            "ben_notification_required": False,
        }
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:8]:
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    notify_ben = header.get("notify_ben", "").lower() == "true"
    return {
        "present": True,
        "strategic_change_level": header.get("strategic_change_level", ""),
        "notify_ben": notify_ben,
        "recommended_next_action": header.get("recommended_next_action", ""),
        "verdict": header.get("verdict", ""),
        "ben_notification_required": notify_ben or header.get("strategic_change_level", "") == "major",
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No external strategy review was present; closeout used command-generated assay artifacts only."
    notification = (
        "Ben should be notified because the review requested a major/notify direction."
        if strategy["ben_notification_required"]
        else "No Ben notification is required by the review header."
    )
    return (
        "Read the latest GPT-5.5-Pro review before selecting the closeout action. "
        "Its earlier trained-pilot recommendation was already satisfied by the multi-site assay; "
        "this closeout follows the newer command-generated negative assay evidence. "
        f"{notification}"
    )


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
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_for(rows: list[dict[str, str]], arm: str) -> dict[str, str]:
    return next((row for row in rows if row.get("arm") == arm), {})


def _gate_actual(rows: list[dict[str, str]], criterion: str) -> str:
    row = next((item for item in rows if item.get("criterion") == criterion), {})
    return row.get("actual", "")


def _parse_list_field(value: str) -> list[str]:
    return [
        item.strip().strip("'\"")
        for item in value.strip().strip("[]").split(",")
        if item.strip()
    ]


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


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
            writer.writerow(row)


def _notes(summary: dict[str, Any]) -> str:
    evidence = summary["evidence"]
    lines = [
        "# Multi-Site PC/Core-Periphery Closeout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Advance to GPU validation: `{summary['advance_to_gpu_validation']}`",
        f"- Failed claim gates: `{evidence['failed_claim_gates']}`",
        f"- Candidate minus MLP heldout CE: `{evidence['candidate_minus_mlp_heldout_ce']}`",
        f"- Candidate minus MLP churn: `{evidence['candidate_minus_mlp_churn']}`",
        f"- Candidate minus MLP commutator: `{evidence['candidate_minus_mlp_commutator']}`",
        "",
        summary["rationale"],
        "",
        f"Next step: {summary['selected_next_step']}",
    ]
    return "\n".join(lines) + "\n"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assay-dir", type=Path, default=DEFAULT_ASSAY_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_multisite_continual_pc_core_periphery_closeout(
        assay_dir=args.assay_dir,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "selected_next_action": summary["selected_next_action"]}, sort_keys=True))


if __name__ == "__main__":
    main()
