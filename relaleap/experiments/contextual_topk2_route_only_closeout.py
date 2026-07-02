"""Close out contextual top-k-2 route-only support redesign."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_PREGATE = Path("results/audits/contextual_topk2_support_quality_pregate_pilot/summary.json")
DEFAULT_BRANCH_SELECTOR = Path("results/reports/post_core_periphery_contextual_dense_branch_selector/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/contextual_topk2_route_only_closeout")

REPAIR_ACTION = "repair_contextual_topk2_route_only_closeout_sources"
CLOSE_ACTION = "close_contextual_topk2_route_only_redesign_no_gpu"
REDIRECT_ACTION = "redirect_to_non_router_mechanism_branch"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "failure_matrix.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_contextual_topk2_route_only_closeout(
    *,
    pregate_path: Path = DEFAULT_PREGATE,
    branch_selector_path: Path = DEFAULT_BRANCH_SELECTOR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Consume the deployable-candidate pregate and record the route-only closeout."""

    start = time.time()
    pregate = _read_json(pregate_path)
    selector = _read_json(branch_selector_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("contextual_topk2_support_quality_pregate", pregate_path, pregate),
        _source_row("non_router_mechanism_branch_selector", branch_selector_path, selector),
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
    evidence = _evidence(pregate, selector, strategy)
    failure_matrix = _failure_matrix(evidence)
    source_failures = [row for row in source_rows[:1] if not row["present"]]
    required_failures = source_failures + [
        row for row in failure_matrix if row["required"] and not row["passed"]
    ]
    candidate_actions = _candidate_actions(evidence, bool(required_failures))
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if required_failures or len(selected) != 1:
        status = "fail"
        decision = "contextual_topk2_route_only_closeout_failed_closed"
        claim_status = "contextual_topk2_route_only_closeout_sources_incomplete"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair or regenerate the contextual top-k-2 pregate summary, then rerun this closeout"
        rationale = "The closeout cannot interpret the route-only branch without coherent pregate evidence."
    else:
        selected_row = selected[0]
        status = "pass"
        decision = "contextual_topk2_route_only_branch_closed"
        claim_status = selected_row["claim_status"]
        selected_next_action = selected_row["candidate_action"]
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
        "backend_policy": "local closeout only; RunPod and Colab remain blocked by route-only pregate failures",
        "source_rows": source_rows,
        "evidence": evidence,
        "failure_matrix": failure_matrix,
        "candidate_actions": candidate_actions,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "failures": required_failures,
        "rationale": rationale,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _evidence(pregate: dict[str, Any], selector: dict[str, Any], strategy: dict[str, Any]) -> dict[str, Any]:
    backend = ((pregate.get("evidence") or {}).get("backend_summaries") or {}).get("local") or {}
    gates = pregate.get("gate_criteria")
    gates = gates if isinstance(gates, list) else []
    failed_gates = [str(row.get("criterion")) for row in gates if row.get("passed") is False]
    return {
        "pregate_status": pregate.get("status"),
        "pregate_decision": pregate.get("decision"),
        "pregate_claim_status": pregate.get("claim_status"),
        "pregate_selected_next_action": pregate.get("selected_next_action"),
        "pregate_selected_next_step": pregate.get("selected_next_step"),
        "pregate_advance_to_gpu_validation": pregate.get("advance_to_gpu_validation"),
        "pregate_promotion_allowed": pregate.get("promotion_allowed"),
        "training_executed": pregate.get("training_executed"),
        "generated_candidate_present": backend.get("deployable_candidate_generation_present"),
        "all_pair_coverage_present": backend.get("all_pair_one_swap_candidate_loss_coverage_present"),
        "generated_candidate_swap_fraction": backend.get("generated_candidate_accepted_one_swap_fraction"),
        "generated_candidate_mean_regret_delta": backend.get("generated_candidate_minus_linear_oracle_regret"),
        "generated_candidate_p90_regret_delta": backend.get("generated_candidate_p90_minus_linear_oracle_regret"),
        "generated_candidate_churn_delta": backend.get("generated_candidate_minus_linear_support_churn_proxy"),
        "generated_candidate_loss_minus_shuffled": backend.get("generated_candidate_loss_minus_shuffled_label_control"),
        "generated_candidate_same_student_regret_delta": backend.get(
            "generated_candidate_same_student_forced_regret_delta_vs_linear"
        ),
        "trained_pair_quality_swap_fraction": backend.get("trained_pair_quality_candidate_accepted_one_swap_fraction"),
        "failed_gate_criteria": failed_gates,
        "selector_status": selector.get("status"),
        "selector_selected_next_action": selector.get("selected_next_action"),
        "selector_next_step": selector.get("next_step") or selector.get("selected_next_step"),
        "selector_claim_status": selector.get("claim_status"),
        "strategy_verdict": strategy["verdict"],
        "strategy_recommended_next_action": strategy["recommended_next_action"],
        "ben_notification_required": strategy["ben_notification_required"],
    }


def _failure_matrix(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _matrix_row(
            "pregate_completed_and_blocks_gpu",
            evidence["pregate_status"] == "pass"
            and evidence["pregate_selected_next_action"] == "record_contextual_topk2_route_only_closeout_no_gpu"
            and evidence["pregate_advance_to_gpu_validation"] is False
            and evidence["pregate_promotion_allowed"] is False,
            {
                "status": evidence["pregate_status"],
                "selected_next_action": evidence["pregate_selected_next_action"],
                "advance_to_gpu_validation": evidence["pregate_advance_to_gpu_validation"],
                "promotion_allowed": evidence["pregate_promotion_allowed"],
            },
            "pregate must be complete and explicitly select route-only closeout",
            True,
        ),
        _matrix_row(
            "deployable_candidate_generation_tested",
            evidence["training_executed"] is True and evidence["generated_candidate_present"] is True,
            {
                "training_executed": evidence["training_executed"],
                "generated_candidate_present": evidence["generated_candidate_present"],
            },
            "closeout requires the generated deployable candidate arm, not only oracle-best labels",
            True,
        ),
        _matrix_row(
            "generated_candidate_fails_regret_churn_null_same_student",
            _generated_candidate_failed(evidence),
            {
                "swap_fraction": evidence["generated_candidate_swap_fraction"],
                "mean_regret_delta": evidence["generated_candidate_mean_regret_delta"],
                "p90_regret_delta": evidence["generated_candidate_p90_regret_delta"],
                "churn_delta": evidence["generated_candidate_churn_delta"],
                "loss_minus_shuffled": evidence["generated_candidate_loss_minus_shuffled"],
                "same_student_regret_delta": evidence["generated_candidate_same_student_regret_delta"],
            },
            "generated arm must fail to beat linear/null/same-student gates before closeout",
            True,
        ),
        _matrix_row(
            "all_pair_one_swap_coverage_absent_fail_closed",
            evidence["all_pair_coverage_present"] is False,
            {"all_pair_coverage_present": evidence["all_pair_coverage_present"]},
            "schema lacks exhaustive all-pair one-swap candidate losses, so claims stay fail-closed",
            True,
        ),
        _matrix_row(
            "non_router_selector_available",
            evidence["selector_status"] == "pass",
            {
                "selector_status": evidence["selector_status"],
                "selector_selected_next_action": evidence["selector_selected_next_action"],
                "selector_next_step": evidence["selector_next_step"],
            },
            "a command-generated non-router branch selector is available as redirect context",
            False,
        ),
        _matrix_row(
            "strategy_review_closeout_condition_satisfied",
            "predeclared closeout" in str(evidence["strategy_recommended_next_action"]).lower()
            or "close contextual" in str(evidence["pregate_selected_next_step"]).lower(),
            {
                "verdict": evidence["strategy_verdict"],
                "recommended_next_action": evidence["strategy_recommended_next_action"],
                "pregate_selected_next_step": evidence["pregate_selected_next_step"],
            },
            "latest review's closeout condition is either explicit or already reflected in the pregate next step",
            False,
        ),
    ]


def _generated_candidate_failed(evidence: dict[str, Any]) -> bool:
    mean_delta = _float(evidence["generated_candidate_mean_regret_delta"])
    p90_delta = _float(evidence["generated_candidate_p90_regret_delta"])
    churn_delta = _float(evidence["generated_candidate_churn_delta"])
    null_delta = _float(evidence["generated_candidate_loss_minus_shuffled"])
    same_student = _float(evidence["generated_candidate_same_student_regret_delta"])
    swap_fraction = _float(evidence["generated_candidate_swap_fraction"])
    return bool(
        swap_fraction == 0.0
        and mean_delta is not None
        and mean_delta <= 0.0
        and p90_delta is not None
        and p90_delta > 0.0
        and churn_delta is not None
        and churn_delta > 0.0
        and null_delta is not None
        and null_delta > 0.0
        and same_student == 0.0
    )


def _candidate_actions(evidence: dict[str, Any], source_failed: bool) -> list[dict[str, str]]:
    if source_failed:
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "required contextual top-k-2 route-only closeout evidence is missing or inconsistent",
                "repair or regenerate the contextual top-k-2 pregate summary",
                "source_artifact_repair_required",
            )
        ]
    return [
        _candidate(
            CLOSE_ACTION,
            "selected",
            (
                "deployable generated-candidate support routing accepts no swaps, worsens churn and p90 regret, "
                "loses the shuffled-label control, has no same-student gain, and lacks exhaustive all-pair loss coverage"
            ),
            "redirect the architecture loop to a non-router mechanism branch using the latest command-generated selector",
            "contextual_topk2_route_only_redesign_closed_no_gpu",
        ),
        _candidate(
            REDIRECT_ACTION,
            "next_after_closeout",
            (
                "the non-router selector remains local-only and currently points to a dense/MLP mechanism track; "
                "use it as branch context rather than reopening contextual top-k-2 routing"
            ),
            str(evidence.get("selector_next_step") or "choose one non-router local mechanism branch before GPU"),
            "non_router_mechanism_redirect_context",
        ),
        _candidate(
            "run_contextual_topk2_gpu_validation",
            "rejected",
            "local route-only support-quality, generated-candidate, null, churn, and same-student gates failed",
            "do not use RunPod or Colab for contextual top-k-2 route-only validation",
            "gpu_validation_blocked",
        ),
        _candidate(
            "rerun_oracle_best_one_swap_acceptance",
            "rejected",
            "oracle-label headroom is nondeployable and was not recovered by trained or generated deployable policies",
            "do not treat recorded oracle-best one-swap labels as a deployable routing mechanism",
            "oracle_best_headroom_not_deployable",
        ),
    ]


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(payload),
        "status": payload.get("status", "missing") if payload else "missing",
        "decision": payload.get("decision", "") if payload else "",
        "claim_status": payload.get("claim_status", "") if payload else "",
    }


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
    for line in path.read_text(encoding="utf-8").splitlines()[:10]:
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
        "ben_notification_required": notify_ben or header.get("strategic_change_level", "").lower() == "major",
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No external strategy review was present; closeout used command-generated pregate evidence."
    notification = (
        "Ben should be notified because the review requested a major/notify direction."
        if strategy["ben_notification_required"]
        else "No Ben notification is required by the review header."
    )
    return (
        "Accepted the latest GPT-5.5-Pro closeout recommendation after the deployable generated-candidate "
        "test failed regret, churn, null, and same-student gates. "
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


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        "# Contextual Top-K-2 Route-Only Closeout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Advance to GPU validation: `{summary['advance_to_gpu_validation']}`",
        f"- Generated candidate swap fraction: `{evidence['generated_candidate_swap_fraction']}`",
        f"- Generated candidate mean regret delta vs linear: `{evidence['generated_candidate_mean_regret_delta']}`",
        f"- Generated candidate churn delta vs linear: `{evidence['generated_candidate_churn_delta']}`",
        f"- All-pair one-swap coverage present: `{evidence['all_pair_coverage_present']}`",
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
    parser.add_argument("--pregate", type=Path, default=DEFAULT_PREGATE)
    parser.add_argument("--branch-selector", type=Path, default=DEFAULT_BRANCH_SELECTOR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_contextual_topk2_route_only_closeout(
        pregate_path=args.pregate,
        branch_selector_path=args.branch_selector,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "selected_next_action": summary["selected_next_action"]}, sort_keys=True))


if __name__ == "__main__":
    main()
