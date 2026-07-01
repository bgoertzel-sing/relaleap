"""Close the context-contrastive core/periphery branch before GPU validation."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_PROBE = Path("results/reports/context_contrastive_core_periphery_probe/summary.json")
DEFAULT_LOW_CHURN_PILOT = Path("results/reports/low_churn_mlp_residual_control_pilot/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/context_contrastive_core_periphery_closeout")

SPARSE_FACTOR_ACTION = "design_low_churn_mlp_sparse_factorization_ceiling"
REPAIR_ACTION = "repair_context_contrastive_core_periphery_closeout_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "decision_matrix.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_context_contrastive_core_periphery_closeout(
    *,
    probe_path: Path = DEFAULT_PROBE,
    low_churn_pilot_path: Path = DEFAULT_LOW_CHURN_PILOT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed closeout and select the sparse-factorization ceiling."""

    start = time.time()
    probe = _read_json(probe_path)
    low_churn = _read_json(low_churn_pilot_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source("context_contrastive_core_periphery_probe", probe_path, probe),
        _source("low_churn_mlp_residual_control_pilot", low_churn_pilot_path, low_churn),
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
        {"source": row["source"], "reason": f"{row['path']} is missing"}
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]
    decision_matrix = _decision_matrix(probe, low_churn, strategy)
    selected_ok = (
        not source_failures
        and _signal(decision_matrix, "context_probe_blocks_gpu").get("passed") is True
        and _signal(decision_matrix, "low_churn_control_is_current_ce_ceiling").get("passed") is True
        and _signal(decision_matrix, "major_strategy_pivot_to_sparse_factorization").get("passed") is True
    )
    selected_next_action = SPARSE_FACTOR_ACTION if selected_ok else REPAIR_ACTION
    status = "pass" if selected_ok else "fail"
    summary = {
        "status": status,
        "decision": (
            "context_contrastive_core_periphery_branch_closed"
            if selected_ok
            else "context_contrastive_core_periphery_closeout_failed_closed"
        ),
        "claim_status": (
            "context_contrastive_core_periphery_closed_sparse_factorization_ceiling_selected"
            if selected_ok
            else "context_contrastive_core_periphery_closeout_sources_incomplete"
        ),
        "selected_next_action": selected_next_action,
        "selected_next_step": (
            "design the local low-churn-MLP sparse-factorization ceiling with oracle, learned, random, frequency, token/position, route-scrambled, and shuffled-teacher controls"
            if selected_ok
            else "repair missing or inconsistent local closeout sources before selecting a next mechanism step"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local closeout/design only; RunPod and Colab remain blocked",
        "ben_should_be_notified": strategy["ben_notification_required"],
        "direction_shift_recorded": strategy["ben_notification_required"],
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "source_rows": source_rows,
        "decision_matrix": decision_matrix,
        "candidate_actions": _candidate_actions(selected_next_action),
        "failures": source_failures
        + [
            row
            for row in decision_matrix
            if row.get("required") and row.get("passed") is not True
        ],
        "rationale": _rationale(probe, low_churn, strategy),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _decision_matrix(
    probe: dict[str, Any],
    low_churn: dict[str, Any],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    probe_failures = probe.get("failures") if isinstance(probe.get("failures"), list) else []
    low_churn_arm = _low_churn_arm(low_churn)
    candidate = probe.get("candidate_observables") if isinstance(probe.get("candidate_observables"), dict) else {}
    candidate_ce = _float(candidate.get("heldout_ce"))
    low_churn_ce = _float(low_churn_arm.get("heldout_ce_loss"))
    return [
        {
            "signal": "context_probe_blocks_gpu",
            "required": True,
            "passed": (
                probe.get("status") == "pass"
                and probe.get("selected_next_action") == "close_or_redesign_context_contrastive_core_periphery_before_gpu"
                and probe.get("advance_to_gpu_validation") is False
                and bool(probe_failures)
            ),
            "actual": {
                "decision": probe.get("decision"),
                "claim_status": probe.get("claim_status"),
                "failure_count": len(probe_failures),
            },
            "expected": "existing probe must block GPU and contain claim-gate failures",
        },
        {
            "signal": "low_churn_control_is_current_ce_ceiling",
            "required": True,
            "passed": low_churn_ce is not None and candidate_ce is not None and low_churn_ce < candidate_ce,
            "actual": {
                "context_candidate_heldout_ce": candidate_ce,
                "low_churn_mlp_heldout_ce": low_churn_ce,
                "low_churn_advancement": low_churn.get("advance_to_gpu_validation"),
            },
            "expected": "low-churn MLP control should beat the context-contrastive candidate on CE while remaining unpromoted",
        },
        {
            "signal": "low_churn_mlp_remains_local_control",
            "required": True,
            "passed": (
                low_churn.get("status") == "pass"
                and low_churn.get("advance_to_gpu_validation") is False
                and low_churn.get("promotion_allowed") is False
            ),
            "actual": {
                "decision": low_churn.get("decision"),
                "claim_status": low_churn.get("claim_status"),
                "selected_next_action": low_churn.get("selected_next_action"),
            },
            "expected": "low-churn MLP evidence is a local control/teacher source, not a GPU promotion",
        },
        {
            "signal": "major_strategy_pivot_to_sparse_factorization",
            "required": True,
            "passed": (
                strategy["present"]
                and str(strategy["strategic_change_level"]).lower() == "major"
                and "sparse-factorization ceiling" in strategy["recommended_next_action"]
            ),
            "actual": {
                "strategic_change_level": strategy["strategic_change_level"],
                "notify_ben": strategy["notify_ben"],
                "recommended_next_action": strategy["recommended_next_action"],
            },
            "expected": "latest GPT-5.5-Pro review must select the local sparse-factorization ceiling pivot",
        },
    ]


def _candidate_actions(selected: str) -> list[dict[str, str]]:
    return [
        {
            "candidate_action": SPARSE_FACTOR_ACTION,
            "disposition": "selected" if selected == SPARSE_FACTOR_ACTION else "blocked",
            "reason": "Context-contrastive core/periphery failed local control, churn, commutator, and pruning gates; the major strategy review recommends a local sparse-factorization ceiling against the low-churn MLP teacher.",
            "next_step": "write the command-driven sparse-factorization ceiling design artifact",
        },
        {
            "candidate_action": "run_runpod_context_contrastive_validation",
            "disposition": "rejected",
            "reason": "Local gates block GPU validation and the latest strategic review explicitly pivots away from this path.",
            "next_step": "keep RunPod and Colab unused for this branch",
        },
        {
            "candidate_action": "continue_context_contrastive_architecture_tuning",
            "disposition": "rejected",
            "reason": "The current evidence beats weak nulls but loses to the low-churn MLP control and fails interference gates.",
            "next_step": "only revisit with a materially new mechanism after the sparse-factorization ceiling",
        },
    ]


def _rationale(probe: dict[str, Any], low_churn: dict[str, Any], strategy: dict[str, Any]) -> str:
    return (
        "The context-contrastive candidate is recorded as negative local evidence: it beats weak nulls but "
        "does not establish protected-core/plastic-periphery behavior against the low-churn MLP control, "
        "retention/churn, commutator, or pruning gates. The GPT-5.5-Pro review is a major pivot and requests "
        f"Ben notification={strategy['ben_notification_required']}; this report records that direction shift and "
        "selects the local sparse-factorization ceiling before any architecture tuning or GPU validation."
    )


def _low_churn_arm(summary: dict[str, Any]) -> dict[str, Any]:
    for row in summary.get("gate_criteria", []):
        if isinstance(row, dict) and row.get("criterion") == "budget_gates_fail_closed":
            actual = row.get("actual")
            if isinstance(actual, list) and actual and isinstance(actual[0], dict):
                return actual[0]
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    strategy = {
        "path": str(path),
        "present": bool(text),
        "strategic_change_level": _header_value(text, "strategic_change_level") or "unknown",
        "notify_ben": _header_value(text, "notify_ben") or "unknown",
        "recommended_next_action": _header_value(text, "recommended_next_action") or "",
        "verdict": _header_value(text, "verdict") or "",
    }
    strategy["ben_notification_required"] = (
        str(strategy["notify_ben"]).lower() == "true"
        or str(strategy["strategic_change_level"]).lower() == "major"
    )
    return strategy


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No strategy review was present; this closeout fails closed."
    if strategy["ben_notification_required"]:
        return "Accepted the major GPT-5.5-Pro pivot; Ben should be notified, and the direction shift is recorded here."
    return "Accepted the external review and preserved the no-GPU local closeout policy."


def _source(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status", "missing" if not path.is_file() else ""),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
    }


def _signal(rows: list[dict[str, Any]], signal: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("signal") == signal), {})


def _header_value(text: str, key: str) -> str:
    prefix = f"{key}:"
    for line in text.splitlines()[:20]:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


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
    return "\n".join(
        [
            "# Context-Contrastive Core/Periphery Closeout",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Selected action: `{summary['selected_next_action']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            f"- Ben should be notified: `{summary['ben_should_be_notified']}`",
            f"- Next step: {summary['selected_next_step']}",
            "",
            "GPU validation remains blocked. This is a major direction shift toward a local sparse-factorization ceiling over the low-churn MLP residual teacher.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--low-churn-pilot", type=Path, default=DEFAULT_LOW_CHURN_PILOT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_context_contrastive_core_periphery_closeout(
        probe_path=args.probe,
        low_churn_pilot_path=args.low_churn_pilot,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "selected_next_action": summary["selected_next_action"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
