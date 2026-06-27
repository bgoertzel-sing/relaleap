"""Dense residual control baseline report after negative ACSR evidence."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_COMMON_BENCHMARK = Path("results/reports/acsr_common_causal_residual_benchmark/summary.json")
DEFAULT_BRANCH_SELECTOR = Path("results/reports/acsr_post_negative_branch_selector/summary.json")
DEFAULT_DENSE_TEACHER = Path(
    "results/audits/token_larger_dense_teacher_residual_distillation_comparison/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/dense_residual_control_baseline")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "candidate_steps.csv",
    "gate_criteria.csv",
    "notes.md",
)

SELECTED_STEP = "dense_residual_rank_norm_interference_benchmark"


def run_dense_residual_control_baseline_report(
    *,
    common_benchmark_path: Path = DEFAULT_COMMON_BENCHMARK,
    branch_selector_path: Path = DEFAULT_BRANCH_SELECTOR,
    dense_teacher_path: Path = DEFAULT_DENSE_TEACHER,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed report that makes dense controls the active baseline."""

    start = time.time()
    common = _read_json(common_benchmark_path)
    selector = _read_json(branch_selector_path)
    dense_teacher = _read_json(dense_teacher_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("common_sparse_vs_dense_benchmark", common_benchmark_path, common),
        _source_row("post_negative_branch_selector", branch_selector_path, selector),
        _source_row("dense_teacher_distillation_pilot", dense_teacher_path, dense_teacher),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "present" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}"
            ),
        },
    ]
    evidence = _evidence(common, selector, dense_teacher, strategy)
    gate_rows = _gate_rows(source_rows, evidence)
    candidate_steps = _candidate_steps(evidence, gate_rows)
    selected = [row for row in candidate_steps if row["disposition"] == "selected"]

    if any(not row["passed"] for row in gate_rows) or len(selected) != 1:
        status = "fail"
        decision = "dense_residual_control_baseline_failed_closed"
        claim_status = "dense_residual_controls_not_activated_due_to_missing_or_inconsistent_sources"
        next_step = "repair dense-control source artifacts before selecting another experiment"
        rationale = "Required post-negative ACSR artifacts were missing or inconsistent."
    else:
        status = "pass"
        decision = "dense_residual_control_baseline_selected"
        claim_status = "dense_residual_controls_active_sparse_support_claim_retired"
        next_step = selected[0]["next_step"]
        rationale = selected[0]["reason"]

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_action": selected[0]["candidate_step"] if status == "pass" else None,
        "next_step": next_step,
        "rationale": rationale,
        "evidence": evidence,
        "source_rows": source_rows,
        "candidate_steps": candidate_steps,
        "gate_criteria": gate_rows,
        "failures": [row for row in gate_rows if not row["passed"]],
        "claim_statuses": {
            "acsr_default_router_promotion": "retired",
            "sparse_support_identity": "not_supported_by_current_dense_controls",
            "dense_residual_controls": "active_baseline" if status == "pass" else "blocked",
            "ben_notification": "required" if strategy["ben_notification_required"] else "not_required",
        },
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, source_rows, candidate_steps, gate_rows)
    return summary


def _evidence(
    common: dict[str, Any],
    selector: dict[str, Any],
    dense_teacher: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    common_sparse = _arm(common, "sparse_contextual_topk2")
    common_dense = _arm(common, "rank_flop_matched_causal_dense")
    dense_margin = None
    if common_sparse and common_dense:
        dense_margin = _float_or_none(common_dense.get("heldout_delta_vs_base_ce")) - _float_or_none(
            common_sparse.get("heldout_delta_vs_base_ce")
        )
    return {
        "common_benchmark_status": common.get("status"),
        "common_claim_status": common.get("claim_status"),
        "common_sparse_topk2_heldout_delta": _float_or_none(
            common_sparse.get("heldout_delta_vs_base_ce") if common_sparse else None
        ),
        "common_dense_heldout_delta": _float_or_none(
            common_dense.get("heldout_delta_vs_base_ce") if common_dense else None
        ),
        "dense_minus_sparse_heldout_delta": dense_margin,
        "branch_selector_status": selector.get("status"),
        "branch_selected_action": selector.get("selected_next_action"),
        "branch_next_step": selector.get("next_step"),
        "dense_teacher_status": dense_teacher.get("status"),
        "dense_teacher_claim_status": dense_teacher.get("claim_status"),
        "dense_teacher_ce_loss": _float_or_none(dense_teacher.get("dense_teacher_ce_loss")),
        "acsr_student_ce_loss": _variant_metric(
            dense_teacher,
            "acsr_predicted_future_support",
            "ce_loss",
        ),
        "strategy_verdict": strategy.get("verdict"),
        "strategy_major_pivot": strategy.get("strategic_change_level") == "major",
        "ben_notification_required": strategy.get("ben_notification_required"),
    }


def _gate_rows(source_rows: list[dict[str, Any]], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    present = all(row["present"] for row in source_rows[:3])
    selector_retired = (
        evidence["branch_selector_status"] == "pass"
        and evidence["branch_selected_action"] == "retire_acsr_promotion_in_favor_of_dense_residual_controls"
    )
    dense_better = (
        evidence["common_dense_heldout_delta"] is not None
        and evidence["common_sparse_topk2_heldout_delta"] is not None
        and evidence["common_dense_heldout_delta"] < evidence["common_sparse_topk2_heldout_delta"]
    )
    teacher_not_supported = evidence["dense_teacher_status"] == "fail"
    strategy_pivot = evidence["strategy_verdict"] == "PIVOT" and evidence["strategy_major_pivot"]
    return [
        _criterion(
            "required_post_negative_sources_present",
            present,
            "common benchmark, branch selector, and dense-teacher artifacts exist",
            [row["present"] for row in source_rows[:3]],
            "required dense-control source artifact missing",
        ),
        _criterion(
            "branch_selector_retired_acsr",
            selector_retired,
            "post-negative selector retires ACSR promotion",
            evidence["branch_selected_action"],
            "branch selector did not select dense-control retirement",
        ),
        _criterion(
            "common_dense_beats_sparse_topk2",
            dense_better,
            "causal dense held-out CE delta is more negative than sparse contextual top-k2",
            {
                "dense_delta": evidence["common_dense_heldout_delta"],
                "sparse_delta": evidence["common_sparse_topk2_heldout_delta"],
            },
            "common benchmark does not support dense-control baseline",
        ),
        _criterion(
            "dense_teacher_does_not_rescue_acsr",
            teacher_not_supported,
            "dense-teacher sparse distillation pilot remains unsupported",
            evidence["dense_teacher_status"],
            "dense-teacher pilot rescued sparse ACSR and should be handled separately",
        ),
        _criterion(
            "strategy_review_major_pivot_recorded",
            strategy_pivot,
            "latest strategy review records a major PIVOT",
            {
                "verdict": evidence["strategy_verdict"],
                "major": evidence["strategy_major_pivot"],
            },
            "strategy review does not support the dense-control pivot",
        ),
    ]


def _candidate_steps(evidence: dict[str, Any], gate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if any(not row["passed"] for row in gate_rows):
        return [
            _candidate(
                SELECTED_STEP,
                "blocked",
                "dense-control evidence is incomplete or inconsistent",
                "repair dense-control source artifacts before selecting another experiment",
            ),
            _candidate(
                "revive_acsr_sparse_support_promotion",
                "blocked",
                "source inconsistency prevents a defensible promotion decision",
                "only reconsider after common dense-control artifacts are repaired",
            ),
        ]
    return [
        _candidate(
            SELECTED_STEP,
            "selected",
            "dense residual controls currently explain the strongest CE gain; the next local assay should test rank, norm, and interference sensitivity around the dense baseline",
            "implement a local dense residual rank/norm/interference benchmark with sparse top-k2 retained only as a comparator",
        ),
        _candidate(
            "revive_acsr_sparse_support_promotion",
            "rejected",
            "current common-benchmark and dense-teacher evidence do not support sparse support identity",
            "not eligible unless a future common dense-control benchmark reverses the evidence",
        ),
        _candidate(
            "gpu_repeat_acsr_promotion",
            "rejected",
            "GPU promotion repeats would spend validation time on a retired sparse claim",
            "use GPU only after a local dense-control benchmark identifies a coherent validation target",
        ),
    ]


def _source_row(source: str, path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    present = bool(summary)
    return {
        "source": source,
        "path": str(path),
        "present": present,
        "status": summary.get("status") if present else "missing",
        "decision": summary.get("decision") if present else None,
        "claim_status": summary.get("claim_status") or summary.get("claim_statuses") if present else None,
    }


def _criterion(
    criterion: str,
    passed: bool,
    threshold: Any,
    actual: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": actual,
        "failure_reason": "" if passed else failure_reason,
    }


def _candidate(candidate_step: str, disposition: str, reason: str, next_step: str) -> dict[str, str]:
    return {
        "candidate_step": candidate_step,
        "disposition": disposition,
        "reason": reason,
        "next_step": next_step,
    }


def _arm(summary: dict[str, Any], name: str) -> dict[str, Any] | None:
    rows = summary.get("arm_metrics")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and row.get("arm") == name:
            return row
    return None


def _variant_metric(summary: dict[str, Any], variant: str, key: str) -> float | None:
    rows = summary.get("variant_rows")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and row.get("variant") == variant:
            return _float_or_none(row.get(key))
    return None


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "present": False,
            "strategic_change_level": None,
            "notify_ben": None,
            "ben_notification_required": False,
            "recommended_next_action": None,
            "verdict": None,
        }
    fields: dict[str, Any] = {"present": True}
    for line in path.read_text(encoding="utf-8").splitlines()[:12]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    fields["ben_notification_required"] = str(fields.get("notify_ben", "")).lower() == "true"
    return {
        "present": True,
        "strategic_change_level": fields.get("strategic_change_level"),
        "notify_ben": fields.get("notify_ben"),
        "ben_notification_required": fields["ben_notification_required"],
        "recommended_next_action": fields.get("recommended_next_action"),
        "verdict": fields.get("verdict"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_none(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


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
    return result.stdout.strip()


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    candidate_steps: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_rows.csv", source_rows)
    _write_csv(out_dir / "candidate_steps.csv", candidate_steps)
    _write_csv(out_dir / "gate_criteria.csv", gate_rows)
    notes = [
        "# Dense Residual Control Baseline",
        "",
        f"Status: `{summary['status']}`",
        f"Decision: `{summary['decision']}`",
        f"Selected next action: `{summary['selected_next_action']}`",
        "",
        summary["rationale"],
        "",
        "The latest strategy review is recorded as a major pivot with Ben notification "
        f"{'required' if summary['claim_statuses']['ben_notification'] == 'required' else 'not required'}.",
    ]
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-benchmark", type=Path, default=DEFAULT_COMMON_BENCHMARK)
    parser.add_argument("--branch-selector", type=Path, default=DEFAULT_BRANCH_SELECTOR)
    parser.add_argument("--dense-teacher", type=Path, default=DEFAULT_DENSE_TEACHER)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    summary = run_dense_residual_control_baseline_report(
        common_benchmark_path=args.common_benchmark,
        branch_selector_path=args.branch_selector,
        dense_teacher_path=args.dense_teacher,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
