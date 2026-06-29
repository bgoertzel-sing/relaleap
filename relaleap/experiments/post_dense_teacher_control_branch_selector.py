"""Select the next branch after the dense-teacher control gate blocks."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_DENSE_TEACHER_CONTROL = Path("results/reports/dense_teacher_control_mechanism_assay/summary.json")
DEFAULT_DENSE_PRIMARY = Path("results/reports/dense_primary_mechanism_assay/summary.json")
DEFAULT_MLP_FOLLOWUP = Path("results/reports/mlp_dense_heldout_mechanism_followup/summary.json")
DEFAULT_MATCHED_DECISION = Path("results/reports/sparse_dense_mlp_matched_intervention_decision/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/post_dense_teacher_control_branch_selector")

LOW_CHURN_MLP_ACTION = "design_low_churn_mlp_residual_control_pregate"
REOPEN_SPARSE_ACTION = "reopen_sparse_acsr_after_new_positive_control_signal"
RUNPOD_REPEAT_ACTION = "run_runpod_dense_teacher_repeat"
REPAIR_SOURCES_ACTION = "repair_missing_dense_teacher_control_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "decision_matrix.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_post_dense_teacher_control_branch_selector(
    *,
    dense_teacher_control_path: Path = DEFAULT_DENSE_TEACHER_CONTROL,
    dense_primary_path: Path = DEFAULT_DENSE_PRIMARY,
    mlp_followup_path: Path = DEFAULT_MLP_FOLLOWUP,
    matched_decision_path: Path = DEFAULT_MATCHED_DECISION,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Choose one bounded local branch after dense-teacher ACSR is blocked."""

    start = time.time()
    dense_teacher_control = _read_json(dense_teacher_control_path)
    dense_primary = _read_json(dense_primary_path)
    mlp_followup = _read_json(mlp_followup_path)
    matched_decision = _read_json(matched_decision_path)
    strategy = _strategy_review(strategy_review_path)
    sources = [
        _source_row("dense_teacher_control_mechanism_assay", dense_teacher_control_path, dense_teacher_control),
        _source_row("dense_primary_mechanism_assay", dense_primary_path, dense_primary),
        _source_row("mlp_dense_heldout_mechanism_followup", mlp_followup_path, mlp_followup),
        _source_row("sparse_dense_mlp_matched_intervention_decision", matched_decision_path, matched_decision),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": f"strategic_change_level={strategy['strategic_change_level']}; notify_ben={strategy['notify_ben']}",
        },
    ]
    failures = _source_failures(sources)
    decision_matrix = _decision_matrix(dense_teacher_control, dense_primary, mlp_followup, matched_decision, strategy)
    candidate_actions = _candidate_actions(decision_matrix, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "post_dense_teacher_control_branch_selector_failed_closed"
        selected_next_action = REPAIR_SOURCES_ACTION
        selected_next_step = "repair missing dense-teacher-control source artifacts"
        claim_status = "dense_teacher_control_branch_sources_incomplete"
        rationale = "The selector cannot choose a research branch until required local source artifacts are present."
    else:
        status = "pass"
        decision = "post_dense_teacher_control_branch_selected"
        selected_next_action = selected[0]["candidate_action"]
        selected_next_step = selected[0]["next_step"]
        claim_status = selected[0]["claim_status"]
        rationale = selected[0]["reason"]

    summary = {
        "status": status,
        "decision": decision,
        "selected_next_action": selected_next_action,
        "selected_next_step": selected_next_step,
        "claim_status": claim_status,
        "promotion_allowed": False,
        "requires_gpu_now": selected_next_action == RUNPOD_REPEAT_ACTION,
        "backend_policy": "local branch selection only; RunPod remains blocked unless the dense-teacher control gate passes",
        "source_rows": sources,
        "decision_matrix": decision_matrix,
        "candidate_actions": candidate_actions,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "failures": failures,
        "rationale": rationale,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _decision_matrix(
    dense_teacher_control: dict[str, Any],
    dense_primary: dict[str, Any],
    mlp_followup: dict[str, Any],
    matched_decision: dict[str, Any],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    mlp_row = _mechanism_row(mlp_followup, "parameter_matched_causal_mlp_control")
    dense24_row = _mechanism_row(mlp_followup, "dense_rank24_best_norm")
    mlp_churn = _float(mlp_row.get("heldout_prediction_changed_vs_base"))
    dense24_churn = _float(dense24_row.get("heldout_prediction_changed_vs_base"))
    mlp_l2 = _float(mlp_row.get("heldout_residual_update_l2"))
    dense24_l2 = _float(dense24_row.get("heldout_residual_update_l2"))
    return [
        {
            "signal": "dense_teacher_control_gate",
            "status": dense_teacher_control.get("status"),
            "decision": dense_teacher_control.get("decision"),
            "claim_status": dense_teacher_control.get("claim_status"),
            "supports_runpod_repeat": dense_teacher_control.get("scientific_gate") == "pass",
            "observed": {
                "scientific_gate": dense_teacher_control.get("scientific_gate"),
                "requires_gpu_now": dense_teacher_control.get("requires_gpu_now"),
                "failure_count": len(_as_list(dense_teacher_control.get("failures"))),
            },
        },
        {
            "signal": "dense_primary_assay",
            "status": dense_primary.get("status"),
            "decision": dense_primary.get("decision"),
            "claim_status": dense_primary.get("claim_status"),
            "supports_mlp_as_high_power_baseline": dense_primary.get("primary_arm") == "parameter_matched_causal_mlp_control",
            "observed": {
                "primary_arm": dense_primary.get("primary_arm"),
                "primary_family": dense_primary.get("primary_family"),
            },
        },
        {
            "signal": "mlp_dense_churn_tradeoff",
            "status": mlp_followup.get("status"),
            "decision": mlp_followup.get("decision"),
            "claim_status": mlp_followup.get("claim_status"),
            "supports_low_churn_mlp_pregate": (
                mlp_churn is not None
                and dense24_churn is not None
                and mlp_churn > dense24_churn
                and mlp_l2 is not None
                and dense24_l2 is not None
                and mlp_l2 > dense24_l2
            ),
            "observed": {
                "mlp_heldout_prediction_changed_vs_base": mlp_churn,
                "dense24_heldout_prediction_changed_vs_base": dense24_churn,
                "mlp_heldout_residual_update_l2": mlp_l2,
                "dense24_heldout_residual_update_l2": dense24_l2,
            },
        },
        {
            "signal": "matched_intervention_guardrail",
            "status": matched_decision.get("status"),
            "decision": matched_decision.get("decision"),
            "claim_status": matched_decision.get("claim_status"),
            "supports_reopening_sparse": matched_decision.get("scientific_gate") == "pass",
            "observed": {
                "scientific_gate": matched_decision.get("scientific_gate"),
                "advancement_row_count": matched_decision.get("advancement_row_count"),
            },
        },
        {
            "signal": "external_strategy_review",
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": f"verdict={strategy['verdict']}",
            "supports_dense_local_fix": "dense" in strategy["recommended_next_action"].lower()
            or "composer" in strategy["recommended_next_action"].lower()
            or "mlp" in strategy["recommended_next_action"].lower(),
            "observed": {
                "strategic_change_level": strategy["strategic_change_level"],
                "notify_ben": strategy["notify_ben"],
            },
        },
    ]


def _candidate_actions(decision_matrix: list[dict[str, Any]], failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if failures:
        return [_candidate(REPAIR_SOURCES_ACTION, "selected", "required source artifacts are missing", "repair missing local source artifacts", "incomplete")]

    dense_gate = _signal(decision_matrix, "dense_teacher_control_gate")
    matched_gate = _signal(decision_matrix, "matched_intervention_guardrail")
    mlp_tradeoff = _signal(decision_matrix, "mlp_dense_churn_tradeoff")
    primary = _signal(decision_matrix, "dense_primary_assay")

    if dense_gate.get("supports_runpod_repeat") is True:
        return [
            _candidate(RUNPOD_REPEAT_ACTION, "selected", "the local dense-teacher control gate passed", "run one RunPod repeat and fetch/check artifacts", "local_gate_passed_gpu_repeat_allowed"),
            _candidate(LOW_CHURN_MLP_ACTION, "deferred", "GPU repeat takes precedence only after local gate pass", "revisit after RunPod evidence", "deferred"),
            _candidate(REOPEN_SPARSE_ACTION, "deferred", "sparse reopening waits for GPU-confirmed positive evidence", "revisit after RunPod evidence", "deferred"),
        ]

    if primary.get("supports_mlp_as_high_power_baseline") and mlp_tradeoff.get("supports_low_churn_mlp_pregate"):
        return [
            _candidate(
                LOW_CHURN_MLP_ACTION,
                "selected",
                "the MLP is the strongest dense/MLP CE baseline, but its residual norm and prediction-flip churn exceed dense24, so the next useful test is a local low-churn pregate",
                "design a local low-churn MLP residual-control pregate with dense24 residual-L2, anchor-KL, flip-churn, and raw intervention-fingerprint gates",
                "mlp_high_power_baseline_needs_low_churn_pregate",
            ),
            _candidate(REOPEN_SPARSE_ACTION, "rejected", "matched sparse/dense/MLP guardrail stayed blocked", "only reopen sparse after new positive matched-control evidence", "rejected"),
            _candidate(RUNPOD_REPEAT_ACTION, "rejected", "local dense-teacher control gate did not pass", "do not use RunPod for this blocked branch", "rejected"),
        ]

    if matched_gate.get("supports_reopening_sparse"):
        return [
            _candidate(REOPEN_SPARSE_ACTION, "selected", "a matched-intervention challenger cleared the guardrail", "design a sparse follow-up with dense/null controls", "matched_sparse_signal_reopened"),
            _candidate(LOW_CHURN_MLP_ACTION, "deferred", "sparse matched-control signal is narrower", "revisit after sparse follow-up", "deferred"),
            _candidate(RUNPOD_REPEAT_ACTION, "rejected", "RunPod still waits for a local scientific gate naming a repeat", "keep work local", "rejected"),
        ]

    return [
        _candidate(LOW_CHURN_MLP_ACTION, "selected", "dense-teacher ACSR and sparse matched controls are blocked, leaving dense/MLP control refinement as the least ambiguous local path", "design a low-churn dense/MLP residual-control pregate", "dense_mlp_control_refinement_selected"),
        _candidate(REOPEN_SPARSE_ACTION, "rejected", "no current sparse matched-control signal supports reopening the branch", "only reopen after new matched-control evidence", "rejected"),
        _candidate(RUNPOD_REPEAT_ACTION, "rejected", "no local gate requires GPU", "keep work local", "rejected"),
    ]


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status") if path.is_file() else "missing",
        "decision": payload.get("decision"),
        "claim_status": payload.get("claim_status"),
    }


def _source_failures(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"source": row["source"], "field": "source_artifact", "reason": f"{row['path']} is missing"}
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]


def _candidate(action: str, disposition: str, reason: str, next_step: str, claim_status: str) -> dict[str, str]:
    return {
        "candidate_action": action,
        "disposition": disposition,
        "reason": reason,
        "next_step": next_step,
        "claim_status": claim_status,
    }


def _signal(rows: list[dict[str, Any]], signal: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("signal") == signal), {})


def _mechanism_row(summary: dict[str, Any], arm: str) -> dict[str, Any]:
    return next((row for row in _as_list(summary.get("mechanism_comparison")) if row.get("arm") == arm), {})


def _strategy_review(path: Path) -> dict[str, Any]:
    defaults = {
        "present": path.is_file(),
        "strategic_change_level": "",
        "notify_ben": "false",
        "ben_notification_required": False,
        "recommended_next_action": "",
        "verdict": "",
    }
    if not path.is_file():
        return defaults
    for raw_line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in defaults:
            defaults[key] = value
    defaults["ben_notification_required"] = (
        str(defaults["notify_ben"]).lower() == "true"
        or str(defaults["strategic_change_level"]).lower() == "major"
    )
    return defaults


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "no external strategy review present"
    if strategy["ben_notification_required"]:
        return "review requested Ben notification or major shift; this selector records direction but does not use GPU"
    return "review consumed; no major direction shift or Ben notification requested"


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "decision_matrix.csv", summary["decision_matrix"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["status"]
        rows = [{"status": "missing"}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Post Dense-Teacher Control Branch Selector",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Selected action: `{summary['selected_next_action']}`",
            f"- Claim status: `{summary['claim_status']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            f"- Promotion allowed: `{summary['promotion_allowed']}`",
            f"- Next step: {summary['selected_next_step']}",
            "",
            "This selector consumes the blocked dense-teacher control gate and keeps the loop local unless that gate passes. It treats the high-power MLP as a control baseline that needs an explicit low-churn, norm-budgeted pregate before any GPU validation or default claim.",
            "",
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[dict[str, Any]]:
    return value if isinstance(value, list) else []


def _float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-teacher-control-path", type=Path, default=DEFAULT_DENSE_TEACHER_CONTROL)
    parser.add_argument("--dense-primary-path", type=Path, default=DEFAULT_DENSE_PRIMARY)
    parser.add_argument("--mlp-followup-path", type=Path, default=DEFAULT_MLP_FOLLOWUP)
    parser.add_argument("--matched-decision-path", type=Path, default=DEFAULT_MATCHED_DECISION)
    parser.add_argument("--strategy-review-path", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_post_dense_teacher_control_branch_selector(
        dense_teacher_control_path=args.dense_teacher_control_path,
        dense_primary_path=args.dense_primary_path,
        mlp_followup_path=args.mlp_followup_path,
        matched_decision_path=args.matched_decision_path,
        strategy_review_path=args.strategy_review_path,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "selected_next_action": summary["selected_next_action"]}, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
