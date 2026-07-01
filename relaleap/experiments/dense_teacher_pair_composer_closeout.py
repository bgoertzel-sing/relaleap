"""Close or redirect the dense-teacher pair-composer branch before GPU validation."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_PROBE = Path("results/reports/dense_teacher_pair_composer_control_extension_probe/summary.json")
DEFAULT_TRUTH_AUDIT = Path("results/reports/dense_teacher_pair_composer_truth_audit/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_pair_composer_closeout")

REDIRECT_ACTION = "redirect_from_pair_composer_to_dense_mlp_control_synthesis"
REPAIR_ACTION = "repair_dense_teacher_pair_composer_closeout_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "decision_matrix.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_dense_teacher_pair_composer_closeout(
    *,
    probe_path: Path = DEFAULT_PROBE,
    truth_audit_path: Path = DEFAULT_TRUTH_AUDIT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed pair-composer closeout and select one local redirect."""

    start = time.time()
    probe = _read_json(probe_path)
    truth = _read_json(truth_audit_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source("pair_composer_control_extension_probe", probe_path, probe),
        _source("pair_composer_truth_audit", truth_audit_path, truth),
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
    decision_matrix = _decision_matrix(probe, truth, strategy)
    required_failures = [row for row in decision_matrix if row.get("required") and row.get("passed") is not True]
    selected_ok = not source_failures and not required_failures
    selected_next_action = REDIRECT_ACTION if selected_ok else REPAIR_ACTION
    status = "pass" if selected_ok else "fail"
    summary = {
        "status": status,
        "decision": (
            "dense_teacher_pair_composer_branch_closed"
            if selected_ok
            else "dense_teacher_pair_composer_closeout_failed_closed"
        ),
        "claim_status": (
            "pair_composer_closed_dense_mlp_controls_dominate_no_gpu"
            if selected_ok
            else "pair_composer_closeout_sources_or_gates_incomplete"
        ),
        "selected_next_action": selected_next_action,
        "selected_next_step": (
            "synthesize the dense/MLP control evidence and choose the next local mechanism branch before any GPU validation"
            if selected_ok
            else "repair missing or inconsistent pair-composer closeout sources"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local closeout only; RunPod and Colab remain blocked",
        "ben_should_be_notified": strategy["ben_notification_required"],
        "direction_shift_recorded": strategy["ben_notification_required"],
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "source_rows": source_rows,
        "decision_matrix": decision_matrix,
        "candidate_actions": _candidate_actions(selected_next_action),
        "failures": source_failures + required_failures,
        "rationale": _rationale(probe, truth),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _decision_matrix(probe: dict[str, Any], truth: dict[str, Any], strategy: dict[str, Any]) -> list[dict[str, Any]]:
    oracle_ce = _float(probe.get("oracle_holdout_true_decoder_ce_loss"))
    learned_ce = _float(probe.get("learned_router_holdout_true_decoder_ce_loss"))
    majority_ce = _float(probe.get("majority_pair_holdout_true_decoder_ce_loss"))
    truth_pair_gain = _truth_pair_vs_independent_gain(truth)
    best_control = _best_control_holdout(probe)
    best_control_ce = _float(best_control.get("true_decoder_ce_loss"))
    pair_beats_control = _criterion(probe, "pair_composer_beats_best_matched_control")
    gpu_complete = _criterion(probe, "remaining_controls_complete_for_gpu")
    class_balance = _criterion(probe, "support_pair_class_balance_sufficient")
    return [
        {
            "signal": "truth_audit_positive_pair_interaction_recorded",
            "required": True,
            "passed": (
                truth.get("status") == "pass"
                and truth.get("advance_to_gpu_validation") is False
                and truth_pair_gain is not None
            ),
            "actual": {
                "decision": truth.get("decision"),
                "claim_status": truth.get("claim_status"),
                "pair_composer_vs_independent_ce_gain": truth_pair_gain,
            },
            "expected": "truth audit records the pair-interaction signal but blocks GPU",
        },
        {
            "signal": "learned_router_survives_majority_and_balance_nulls",
            "required": True,
            "passed": (
                probe.get("status") == "pass"
                and learned_ce is not None
                and majority_ce is not None
                and learned_ce < majority_ce
                and class_balance.get("passed") is True
            ),
            "actual": {
                "learned_holdout_ce": learned_ce,
                "majority_holdout_ce": majority_ce,
                "class_balance": class_balance.get("actual"),
            },
            "expected": "learned deployable router must beat majority-pair null with nontrivial support-pair entropy",
        },
        {
            "signal": "interference_controls_measured",
            "required": True,
            "passed": (
                _criterion(probe, "matched_dense_mlp_control_rows_measured").get("passed") is True
                and _criterion(probe, "exact_finite_update_commutator_measured").get("passed") is True
                and _criterion(probe, "retention_churn_measured").get("passed") is True
            ),
            "actual": {
                "matched_controls": _criterion(probe, "matched_dense_mlp_control_rows_measured").get("actual"),
                "commutator": _criterion(probe, "exact_finite_update_commutator_measured").get("actual"),
                "retention_churn": _criterion(probe, "retention_churn_measured").get("actual"),
            },
            "expected": "matched controls plus commutator and retention/churn rows must be present before closeout",
        },
        {
            "signal": "dense_mlp_controls_dominate_pair_composer",
            "required": True,
            "passed": (
                oracle_ce is not None
                and best_control_ce is not None
                and best_control_ce + 0.01 < oracle_ce
                and pair_beats_control.get("passed") is False
                and gpu_complete.get("passed") is False
            ),
            "actual": {
                "oracle_pair_holdout_ce": oracle_ce,
                "learned_router_holdout_ce": learned_ce,
                "best_matched_control": best_control.get("arm"),
                "best_matched_control_ce": best_control_ce,
                "pair_beats_control_gate": pair_beats_control,
                "gpu_completion_gate": gpu_complete,
            },
            "expected": "best matched dense/MLP residual control must beat the pair composer, keeping GPU blocked",
        },
        {
            "signal": "external_strategy_review_supports_local_controls",
            "required": True,
            "passed": (
                strategy["present"]
                and str(strategy["strategic_change_level"]).lower() in {"minor", "major"}
                and "control" in strategy["recommended_next_action"].lower()
            ),
            "actual": {
                "strategic_change_level": strategy["strategic_change_level"],
                "notify_ben": strategy["notify_ben"],
                "recommended_next_action": strategy["recommended_next_action"],
            },
            "expected": "latest review must support local control/closeout work before GPU",
        },
    ]


def _candidate_actions(selected: str) -> list[dict[str, str]]:
    return [
        {
            "candidate_action": REDIRECT_ACTION,
            "disposition": "selected" if selected == REDIRECT_ACTION else "blocked",
            "claim_status": "pair_composer_closed_dense_mlp_controls_dominate_no_gpu",
            "reason": "The learned pair-router signal survives majority/null checks, but matched dense/MLP controls dominate the pair composer on holdout CE.",
            "next_step": "write a local dense/MLP control synthesis or branch selector before any GPU validation",
        },
        {
            "candidate_action": "run_runpod_pair_composer_validation",
            "disposition": "rejected",
            "claim_status": "gpu_validation_blocked",
            "reason": "The local pair-composer gate fails against matched dense/MLP controls.",
            "next_step": "keep RunPod and Colab unused for this branch",
        },
        {
            "candidate_action": "continue_pair_composer_tuning",
            "disposition": "rejected",
            "claim_status": "local_tuning_not_prioritized",
            "reason": "The current positive is already explained by stronger dense/MLP residual controls, so more pair-composer tuning would not answer the causal-separability question.",
            "next_step": "only revisit pair composition after a materially new low-interference mechanism is specified",
        },
    ]


def _rationale(probe: dict[str, Any], truth: dict[str, Any]) -> str:
    return (
        "The pair-composer branch has a real local signal: the oracle and learned-router pair rows beat independent "
        "and majority/null baselines. It is still not a sparse causal-separability or GPU-readiness result because "
        f"the best matched dense/MLP control CE ({_float(_best_control_holdout(probe).get('true_decoder_ce_loss'))}) "
        f"is far below the oracle pair-composer CE ({_float(probe.get('oracle_holdout_true_decoder_ce_loss'))}). "
        f"The truth audit decision was {truth.get('decision')}; this closeout preserves that positive signal while "
        "redirecting away from GPU validation."
    )


def _best_control_holdout(summary: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in summary.get("control_rows", [])
        if isinstance(row, dict) and row.get("split") == "holdout" and _float(row.get("true_decoder_ce_loss")) is not None
    ]
    return min(rows, key=lambda row: float(row["true_decoder_ce_loss"])) if rows else {}


def _truth_pair_vs_independent_gain(summary: dict[str, Any]) -> float | None:
    direct = _float(summary.get("pair_composer_vs_independent_ce_gain"))
    if direct is not None:
        return direct
    pair_metrics = summary.get("pair_metrics")
    if isinstance(pair_metrics, dict):
        return _float(pair_metrics.get("pair_vs_independent_holdout_ce_gain"))
    return None


def _criterion(summary: dict[str, Any], name: str) -> dict[str, Any]:
    for row in summary.get("gate_criteria", []):
        if isinstance(row, dict) and row.get("criterion") == name:
            return row
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
        return "Accepted the GPT-5.5-Pro direction shift; Ben should be notified, and the shift is recorded here."
    return "Accepted the external review's local-control recommendation and kept GPU validation blocked."


def _source(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status", "missing" if not path.is_file() else ""),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
    }


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
            "# Dense-Teacher Pair-Composer Closeout",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Selected action: `{summary['selected_next_action']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            f"- Ben should be notified: `{summary['ben_should_be_notified']}`",
            f"- Next step: {summary['selected_next_step']}",
            "",
            "GPU validation remains blocked because matched dense/MLP residual controls dominate the pair composer.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--truth-audit", type=Path, default=DEFAULT_TRUTH_AUDIT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_dense_teacher_pair_composer_closeout(
        probe_path=args.probe,
        truth_audit_path=args.truth_audit,
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
