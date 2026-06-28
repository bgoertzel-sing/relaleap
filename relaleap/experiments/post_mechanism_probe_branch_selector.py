"""Select the next branch after local mechanism-probe closeouts."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_MECHANISM_REPEAT = Path(
    "results/reports/mechanism_factorized_continual_learning_repeat/summary.json"
)
DEFAULT_COMMUTATOR_ASSAY = Path("results/reports/acsr_finite_update_commutator_assay/summary.json")
DEFAULT_DENSE_TEACHER = Path(
    "results/audits/token_larger_dense_teacher_residual_distillation_comparison/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/post_mechanism_probe_branch_selector")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "branch_criteria.csv",
    "notes.md",
)


def run_post_mechanism_probe_branch_selector(
    *,
    mechanism_repeat_path: Path = DEFAULT_MECHANISM_REPEAT,
    commutator_assay_path: Path = DEFAULT_COMMUTATOR_ASSAY,
    dense_teacher_path: Path = DEFAULT_DENSE_TEACHER,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Close local mechanism branches and select exactly one next step."""

    start = time.time()
    mechanism = _read_json(mechanism_repeat_path)
    commutator = _read_json(commutator_assay_path)
    dense_teacher = _read_json(dense_teacher_path)
    review = _strategy_review(strategy_review_path)

    source_rows = [
        _source_row("mechanism_factorized_cl_repeat", mechanism_repeat_path, mechanism),
        _source_row("finite_update_commutator_assay", commutator_assay_path, commutator),
        _source_row("dense_teacher_residual_distillation", dense_teacher_path, dense_teacher),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy_review_path.is_file(),
            "status": "pass" if strategy_review_path.is_file() else "missing_optional",
            "decision": review["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={review['strategic_change_level']}; "
                f"notify_ben={review['notify_ben']}"
            ),
        },
    ]
    criteria = _branch_criteria(mechanism, commutator, dense_teacher, source_rows)
    hard_failures = [
        row for row in criteria if row["severity"] == "hard" and not row["passed"]
    ]
    claim_candidates = [
        row for row in criteria if row["severity"] == "claim" and row["passed"]
    ]
    if hard_failures:
        status = "fail"
        decision = "post_mechanism_probe_branch_selector_failed_closed"
        claim_status = "post_mechanism_probe_sources_not_interpretable"
        selected_next_step = "repair_missing_or_failed_local_mechanism_probe_artifacts"
    elif claim_candidates:
        status = "pass"
        decision = "post_mechanism_probe_branch_candidate_found"
        claim_status = "local_mechanism_branch_candidate_requires_strategy_review"
        selected_next_step = "request_strategy_review_before_gpu_validation_or_new_promotion_claim"
    else:
        status = "pass"
        decision = "post_mechanism_probe_branches_closed"
        claim_status = "sparse_retention_commutator_and_dense_teacher_mechanisms_not_established"
        selected_next_step = "run_acsr_broader_mechanism_gate_with_existing_local_packets"

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "promotion_allowed": False,
        "requires_gpu_now": False,
        "backend_policy": "local CPU branch selection; no RunPod/Colab validation authorized",
        "source_rows": source_rows,
        "branch_criteria": criteria,
        "strategy_review": review,
        "direction_shift": {
            "strategic_change_level": review["strategic_change_level"],
            "notify_ben": review["notify_ben"],
            "record": _direction_shift_record(review),
        },
        "rationale": _rationale(decision),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, source_rows, criteria)
    return summary


def _branch_criteria(
    mechanism: dict[str, Any],
    commutator: dict[str, Any],
    dense_teacher: dict[str, Any],
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _criterion(
            "required_sources_present",
            all(row["present"] for row in source_rows[:3]),
            "hard",
            "mechanism repeat, commutator assay, and dense-teacher summaries exist",
            [row["path"] for row in source_rows[:3] if row["present"]],
            "missing local branch artifact",
        ),
        _criterion(
            "mechanism_repeat_runtime_passed",
            mechanism.get("status") == "pass",
            "hard",
            "mechanism-factorized CL repeat must pass runtime/schema checks",
            mechanism.get("status"),
            "mechanism repeat is not interpretable",
        ),
        _criterion(
            "commutator_assay_runtime_interpretable",
            commutator.get("status") in {"pass", "fail"}
            and commutator.get("decision") != "acsr_finite_update_commutator_assay_failed_closed",
            "hard",
            "commutator assay must have completed rather than failed closed operationally",
            commutator.get("decision"),
            "commutator assay did not complete",
        ),
        _criterion(
            "dense_teacher_runtime_interpretable",
            dense_teacher.get("status") in {"pass", "fail"}
            and dense_teacher.get("decision") != "dense_teacher_residual_distillation_runtime_failed",
            "hard",
            "dense-teacher pilot must have completed rather than failed at runtime",
            dense_teacher.get("decision"),
            "dense-teacher pilot did not complete",
        ),
        _criterion(
            "mechanism_sparse_retention_claim_supported",
            mechanism.get("claim_status")
            == "mechanism_factorized_sparse_retention_candidate_supported_not_promoted",
            "claim",
            "mechanism-factorized sparse retention claim survives repeat gates",
            mechanism.get("claim_status"),
            "mechanism-factorized sparse retention claim is blocked",
        ),
        _criterion(
            "commutator_sparse_advantage_candidate",
            commutator.get("decision") == "acsr_sparse_commutator_lower_than_dense_control",
            "claim",
            "sparse ACSR commutator lower than dense control with material magnitude",
            commutator.get("decision"),
            "commutator branch did not produce a sparse advantage candidate",
        ),
        _criterion(
            "dense_teacher_distillation_candidate",
            dense_teacher.get("decision")
            == "dense_teacher_residual_distillation_acsr_pilot_supported_not_promoted",
            "claim",
            "ACSR predicted support distills dense teacher at least as well as controls",
            dense_teacher.get("decision"),
            "dense-teacher branch did not support ACSR distillation",
        ),
    ]


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


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status") if payload else "missing",
        "decision": payload.get("decision") if payload else "",
        "claim_status": payload.get("claim_status") if payload else "",
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    values: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"strategic_change_level", "notify_ben", "recommended_next_action", "verdict"}:
            values[key] = value.strip()
    return {
        "present": path.is_file(),
        "strategic_change_level": values.get("strategic_change_level", "missing"),
        "notify_ben": values.get("notify_ben", "false").lower() == "true",
        "recommended_next_action": values.get("recommended_next_action", ""),
        "verdict": values.get("verdict", ""),
    }


def _direction_shift_record(review: dict[str, Any]) -> str:
    if review["notify_ben"] or review["strategic_change_level"] == "major":
        return "Ben should be notified before treating this branch selector as routine."
    return "No major direction shift or Ben notification requested by the latest strategy review."


def _rationale(decision: str) -> str:
    if decision == "post_mechanism_probe_branches_closed":
        return (
            "The local mechanism-factorized sparse-retention repeat did not replicate, "
            "the commutator assay was too small for a sparse mechanism claim, and the "
            "dense-teacher pilot failed its CE guardrail. Return to the existing broader "
            "ACSR mechanism gate rather than promoting or GPU-validating any branch."
        )
    if decision == "post_mechanism_probe_branch_candidate_found":
        return (
            "At least one local mechanism branch produced a candidate signal. Because "
            "promotion remains disallowed, strategy review is required before GPU validation."
        )
    return "A required local mechanism-probe artifact is missing or operationally failed."


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    criteria: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_rows.csv", source_rows)
    _write_csv(out_dir / "branch_criteria.csv", criteria)
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Post-Mechanism Probe Branch Selector",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Decision: `{summary.get('decision')}`",
        f"- Claim status: `{summary.get('claim_status')}`",
        f"- Selected next step: {summary.get('selected_next_step')}",
        f"- Requires GPU now: `{summary.get('requires_gpu_now')}`",
        "",
        str(summary.get("rationale")),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism-repeat", type=Path, default=DEFAULT_MECHANISM_REPEAT)
    parser.add_argument("--commutator-assay", type=Path, default=DEFAULT_COMMUTATOR_ASSAY)
    parser.add_argument("--dense-teacher", type=Path, default=DEFAULT_DENSE_TEACHER)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_post_mechanism_probe_branch_selector(
        mechanism_repeat_path=args.mechanism_repeat,
        commutator_assay_path=args.commutator_assay,
        dense_teacher_path=args.dense_teacher,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
