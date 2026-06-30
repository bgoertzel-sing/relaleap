"""Close out the dense-teacher residual distillation comparison."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_COMPARISON = Path("results/audits/token_larger_dense_teacher_residual_distillation_comparison/summary.json")
DEFAULT_GATE_ROWS = Path("results/audits/token_larger_dense_teacher_residual_distillation_comparison/gate_criteria.csv")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_residual_distillation_closeout")

CLOSE_ACTION = "close_dense_teacher_residual_distillation_before_gpu"
REPAIR_ACTION = "repair_dense_teacher_residual_distillation_sources"
FOLLOWUP_ACTION = "return_to_acsr_broader_mechanism_benchmark"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "closeout_rows.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_dense_teacher_residual_distillation_closeout(
    *,
    comparison_path: Path = DEFAULT_COMPARISON,
    gate_rows_path: Path = DEFAULT_GATE_ROWS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Interpret the dense-teacher comparison and choose one local follow-up."""

    start = time.time()
    comparison = _read_json(comparison_path)
    gate_rows = _read_csv(gate_rows_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_json("dense_teacher_residual_distillation_comparison", comparison_path, comparison),
        _source_csv("dense_teacher_residual_distillation_gate_rows", gate_rows_path, gate_rows),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}"
            ),
            "row_count": "",
        },
    ]
    evidence = _evidence(comparison, gate_rows, strategy)
    failures = _source_failures(source_rows, evidence)
    closeout_rows = _closeout_rows(evidence)
    candidate_actions = _candidate_actions(evidence, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "dense_teacher_residual_distillation_closeout_failed_closed"
        claim_status = "dense_teacher_distillation_source_artifacts_incomplete"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair or regenerate the dense-teacher residual distillation comparison artifacts"
        rationale = "Required source artifacts are missing, contradictory, or require Ben notification."
    else:
        status = "pass"
        decision = "dense_teacher_residual_distillation_branch_closed"
        claim_status = selected[0]["claim_status"]
        selected_next_action = selected[0]["candidate_action"]
        selected_next_step = selected[0]["next_step"]
        rationale = selected[0]["reason"]

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_action": selected_next_action,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local closeout only; RunPod and Colab remain blocked until a local mechanism gate passes",
        "source_rows": source_rows,
        "evidence": evidence,
        "closeout_rows": closeout_rows,
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


def _evidence(
    comparison: dict[str, Any],
    gate_rows: list[dict[str, str]],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    gates = {row.get("criterion", ""): _bool(row.get("passed")) for row in gate_rows}
    failures = _as_list(comparison.get("failures"))
    return {
        "comparison_status": comparison.get("status"),
        "comparison_decision": comparison.get("decision"),
        "comparison_claim_status": comparison.get("claim_status"),
        "comparison_gate_passes": _nested(comparison, "gate_status", "passes_dense_teacher_distillation_gate"),
        "failure_criteria": [row.get("criterion") for row in failures if isinstance(row, dict)],
        "dense_teacher_ce_loss": _float(comparison.get("dense_teacher_ce_loss")),
        "base_ce_loss": _float(comparison.get("base_ce_loss")),
        "primary_ce_margin_gate": gates.get("acsr_ce_not_worse_than_teacher_by_large_margin"),
        "calibrated_teacher_scale_gate": gates.get("calibrated_teacher_scale_gate"),
        "acsr_null_mse_gate": gates.get("acsr_beats_token_position_and_shuffled_distillation_nulls"),
        "source_gates_present": gates.get("source_gates_present_and_passing"),
        "strategy_verdict": strategy["verdict"],
        "strategy_recommended_next_action": strategy["recommended_next_action"],
        "ben_notification_required": strategy["notify_ben"] or strategy["strategic_change_level"] == "major",
    }


def _closeout_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "branch": "dense_teacher_residual_distillation",
            "source_decision": evidence["comparison_decision"],
            "disposition": "closed_before_gpu"
            if evidence["comparison_gate_passes"] is False
            else "eligible_for_repeat",
            "reason": (
                "ACSR support/value distillation fails CE-to-teacher or calibrated-scale/null gates"
                if evidence["comparison_gate_passes"] is False
                else "all local dense-teacher distillation gates passed"
            ),
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
        {
            "branch": "acsr_broader_mechanism_benchmark",
            "source_decision": "local_closeout",
            "disposition": "redirect_target"
            if evidence["comparison_gate_passes"] is False
            else "deferred_for_repeat",
            "reason": "dense-teacher distillation is not the next GPU path; return to broader ACSR mechanism evidence",
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
        {
            "branch": "gpu_validation",
            "source_decision": "local_closeout",
            "disposition": "blocked",
            "reason": "no local dense-teacher distillation gate permits backend validation",
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
    ]


def _candidate_actions(evidence: dict[str, Any], failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if failures:
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "required dense-teacher distillation source artifacts are missing, incoherent, or require Ben notification",
                "repair or regenerate the dense-teacher residual distillation comparison artifacts",
                "source_repair_required",
            )
        ]
    if evidence["comparison_gate_passes"] is True:
        return [
            _candidate(
                "repeat_dense_teacher_residual_distillation_before_gpu",
                "selected",
                "the local dense-teacher distillation gate passed and needs a repeat before backend spend",
                "repeat the dense-teacher residual distillation comparison on an adjacent local seed",
                "local_repeat_required_before_gpu",
            )
        ]
    return [
        _candidate(
            CLOSE_ACTION,
            "selected",
            "dense-teacher distillation remains local negative evidence: the sparse ACSR student is far from the teacher CE and the calibrated 0.25-scale row does not beat all nulls",
            "record the branch closeout and use the broader ACSR mechanism benchmark as the next local source of truth",
            "dense_teacher_distillation_negative_closeout_no_gpu",
        ),
        _candidate(
            FOLLOWUP_ACTION,
            "deferred",
            "the broader benchmark is the next source to inspect after this closeout is recorded",
            "inspect or regenerate the broader ACSR mechanism benchmark before any GPU validation",
            "deferred_until_closeout_recorded",
        ),
    ]


def _source_json(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status") if path.is_file() else "missing",
        "decision": payload.get("decision"),
        "claim_status": payload.get("claim_status"),
        "row_count": "",
    }


def _source_csv(source: str, path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": "present" if path.is_file() else "missing",
        "decision": "",
        "claim_status": "",
        "row_count": len(rows),
    }


def _source_failures(source_rows: list[dict[str, Any]], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    failures = [
        {"source": row["source"], "field": "source_artifact", "reason": f"{row['path']} is missing"}
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]
    if not failures and evidence["comparison_gate_passes"] is None:
        failures.append(
            {
                "source": "dense_teacher_residual_distillation_comparison",
                "field": "gate_status.passes_dense_teacher_distillation_gate",
                "reason": "missing local gate pass/fail field",
            }
        )
    if evidence["ben_notification_required"]:
        failures.append(
            {
                "source": "strategy_review",
                "field": "notify_ben_or_major_strategy_change",
                "reason": "external review requests Ben notification before branch selection",
            }
        )
    return failures


def _candidate(
    action: str,
    disposition: str,
    reason: str,
    next_step: str,
    claim_status: str,
) -> dict[str, Any]:
    return {
        "candidate_action": action,
        "disposition": disposition,
        "reason": reason,
        "next_step": next_step,
        "claim_status": claim_status,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return {
        "present": path.is_file(),
        "strategic_change_level": _header(text, "strategic_change_level", "none"),
        "notify_ben": _header(text, "notify_ben", "false").lower() == "true",
        "recommended_next_action": _header(text, "recommended_next_action", ""),
        "verdict": _header(text, "verdict", ""),
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No external strategy review was present; this closeout used local dense-teacher artifacts only."
    if strategy["notify_ben"] or strategy["strategic_change_level"] == "major":
        return "External review requires Ben notification; closeout fails closed until that is handled."
    return (
        "Accepted the review's no-RunPod/fail-closed direction; this closeout records local negative "
        "dense-teacher evidence and keeps backend validation blocked."
    )


def _header(text: str, key: str, default: str) -> str:
    prefix = f"{key}:"
    for line in text.splitlines()[:20]:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return default


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "closeout_rows.csv", summary["closeout_rows"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    notes = [
        "# Dense Teacher Residual Distillation Closeout",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Selected next step: {summary['selected_next_step']}",
        f"- GPU validation remains blocked: `{not summary['advance_to_gpu_validation']}`",
        f"- Rationale: {summary['rationale']}",
        f"- Strategy review handling: {summary['strategy_review_handling']}",
    ]
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--gate-rows", type=Path, default=DEFAULT_GATE_ROWS)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_dense_teacher_residual_distillation_closeout(
        comparison_path=args.comparison,
        gate_rows_path=args.gate_rows,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
