"""Close out or scale the hidden/future Transformer-ACSR pregate branch."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_PREGATE = Path("results/reports/transformer_acsr_hidden_future_predictor_pregate/summary.json")
DEFAULT_CONTROL_CONTRACT = Path(
    "results/reports/transformer_acsr_hidden_future_predictor_pregate/control_contract.csv"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/transformer_acsr_hidden_future_predictor_closeout")

SCALE_CAPTURE_ACTION = "scale_hidden_future_capture_locally_before_gpu"
CLOSE_BRANCH_ACTION = "hold_hidden_future_transformer_acsr_before_gpu"
REPAIR_ACTION = "repair_hidden_future_predictor_closeout_sources"
GPU_ACTION = "run_runpod_hidden_future_transformer_acsr_validation"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "decision_matrix.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_transformer_acsr_hidden_future_predictor_closeout(
    *,
    pregate_path: Path = DEFAULT_PREGATE,
    control_contract_path: Path = DEFAULT_CONTROL_CONTRACT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Consume hidden/future pregate artifacts and write a fail-closed decision."""

    start = time.time()
    pregate = _read_json(pregate_path)
    controls = _read_csv(control_contract_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("transformer_acsr_hidden_future_predictor_pregate", pregate_path, pregate),
        {
            "source": "transformer_acsr_hidden_future_predictor_control_contract",
            "path": str(control_contract_path),
            "present": control_contract_path.is_file(),
            "status": "read" if control_contract_path.is_file() else "missing",
            "decision": f"row_count={len(controls)}",
            "claim_status": _control_status(controls),
        },
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
    failures = _source_failures(source_rows)
    decision_matrix = _decision_matrix(pregate, controls, strategy)
    candidate_actions = _candidate_actions(decision_matrix, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "transformer_acsr_hidden_future_predictor_closeout_failed_closed"
        claim_status = "hidden_future_predictor_closeout_sources_incomplete"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair or regenerate hidden/future predictor pregate artifacts"
        rationale = "The closeout cannot choose a branch until required local source artifacts are present."
    else:
        selected_row = selected[0]
        status = "pass"
        decision = "transformer_acsr_hidden_future_predictor_closeout_gpu_blocked"
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
        "backend_policy": "local closeout/scale decision only; RunPod and Colab remain blocked",
        "ben_should_be_notified": strategy["ben_notification_required"],
        "direction_shift_recorded": strategy["ben_notification_required"],
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "source_rows": source_rows,
        "decision_matrix": decision_matrix,
        "candidate_actions": candidate_actions,
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
    pregate: dict[str, Any],
    controls: list[dict[str, str]],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    missing_controls = [
        row.get("control", "")
        for row in controls
        if row.get("status") not in {"available", "present"}
    ]
    heldout_sequence_count = _int(pregate.get("heldout_sequence_count"))
    train_sequence_count = _int(pregate.get("train_sequence_count"))
    same_student_delta = _float(pregate.get("prefix_hidden_mean_forced_minus_student_router_loss"))
    prefix_jaccard = _float(pregate.get("prefix_hidden_jaccard"))
    token_position_jaccard = _float(pregate.get("token_position_jaccard"))
    shuffled_jaccard = _float(pregate.get("shuffled_target_jaccard"))
    frequency_jaccard = _float(pregate.get("frequency_jaccard"))
    return [
        {
            "signal": "pregate_completed_no_gpu",
            "required": True,
            "passed": (
                pregate.get("status") == "pass"
                and pregate.get("advance_to_gpu_validation") is False
                and pregate.get("promotion_allowed") is False
            ),
            "actual": {
                "decision": pregate.get("decision"),
                "claim_status": pregate.get("claim_status"),
                "row_count": pregate.get("row_count"),
                "heldout_row_count": pregate.get("heldout_row_count"),
            },
            "expected": "completed local hidden/future pregate with no GPU advancement",
        },
        {
            "signal": "prefix_hidden_beats_registered_nulls",
            "required": False,
            "passed": pregate.get("null_margin_gate_passes") is True,
            "actual": {
                "prefix_hidden_jaccard": prefix_jaccard,
                "token_position_jaccard": token_position_jaccard,
                "shuffled_target_jaccard": shuffled_jaccard,
                "frequency_jaccard": frequency_jaccard,
            },
            "expected": "prefix-hidden predictor beats token/position, shuffled, delayed, and frequency controls",
        },
        {
            "signal": "same_student_loss_gate_failed",
            "required": True,
            "passed": (
                pregate.get("same_student_loss_gate_passes") is False
                and same_student_delta is not None
                and same_student_delta > 0.0
            ),
            "actual": {
                "prefix_hidden_mean_forced_minus_student_router_loss": same_student_delta,
                "same_student_loss_gate_passes": pregate.get("same_student_loss_gate_passes"),
            },
            "expected": "predicted support must not be treated as deployable if forced loss is worse than the same frozen student router",
        },
        {
            "signal": "downstream_controls_missing",
            "required": True,
            "passed": (
                pregate.get("downstream_intervention_budget_gate_passes") is False
                and "retention_churn_budget" in missing_controls
                and "finite_update_commutator_budget" in missing_controls
                and "future_perturbation_invariance" in missing_controls
            ),
            "actual": {"missing_controls": missing_controls},
            "expected": "retention/churn, commutator, and future-perturbation controls remain unavailable",
        },
        {
            "signal": "capture_too_small_for_promotion",
            "required": True,
            "passed": train_sequence_count <= 3 and heldout_sequence_count <= 1,
            "actual": {
                "train_sequence_count": train_sequence_count,
                "heldout_sequence_count": heldout_sequence_count,
            },
            "expected": "current packet is a smoke pregate, not promotion-scale evidence",
        },
        {
            "signal": "strategy_review_supports_local_no_gpu",
            "required": False,
            "passed": (
                strategy["present"]
                and str(strategy["strategic_change_level"]).lower() != "major"
                and "gpu" in strategy["recommended_next_action"].lower()
            ),
            "actual": {
                "strategic_change_level": strategy["strategic_change_level"],
                "notify_ben": strategy["notify_ben"],
                "recommended_next_action": strategy["recommended_next_action"],
            },
            "expected": "latest review should remain compatible with local no-GPU hidden/future work",
        },
    ]


def _candidate_actions(
    decision_matrix: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if failures:
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "required local pregate/control artifacts are missing",
                "repair or regenerate hidden/future predictor pregate sources",
                "source_artifact_repair_required",
            )
        ]

    pregate_ok = _signal(decision_matrix, "pregate_completed_no_gpu").get("passed") is True
    nulls_beat = _signal(decision_matrix, "prefix_hidden_beats_registered_nulls").get("passed") is True
    same_student_failed = _signal(decision_matrix, "same_student_loss_gate_failed").get("passed") is True
    controls_missing = _signal(decision_matrix, "downstream_controls_missing").get("passed") is True
    tiny_capture = _signal(decision_matrix, "capture_too_small_for_promotion").get("passed") is True
    select_scale = pregate_ok and nulls_beat and same_student_failed and controls_missing and tiny_capture
    select_hold = pregate_ok and (same_student_failed or controls_missing)

    return [
        _candidate(
            SCALE_CAPTURE_ACTION,
            "selected" if select_scale else "deferred",
            (
                "Prefix-hidden support prediction beats registered nulls, but the current 3-train/1-heldout packet is too small and the predicted pair is still worse than the same student router under exact forced loss."
            ),
            "scale local hidden/future capture and add current-hidden-shuffled/null-stratified, retention/churn, commutator, and future-perturbation controls before any GPU validation",
            "hidden_future_branch_interesting_but_gpu_blocked",
        ),
        _candidate(
            CLOSE_BRANCH_ACTION,
            "selected" if select_hold and not select_scale else "recorded",
            (
                "The branch cannot advance while same-student forced loss or downstream mechanism-budget controls fail."
            ),
            "hold this branch before GPU unless a scaled local control packet clears same-student loss and mechanism-budget gates",
            "hidden_future_transformer_acsr_not_deployable",
        ),
        _candidate(
            GPU_ACTION,
            "rejected",
            "Local exact same-student loss, retention/churn, commutator, and future-perturbation gates do not all pass.",
            "do not use RunPod or Colab for this branch yet",
            "gpu_validation_scientifically_blocked",
        ),
        _candidate(
            "tune_hidden_future_transformer_on_current_packet",
            "rejected",
            "Tuning against a single heldout sequence risks fitting a smoke packet and does not address forced-loss or missing mechanism controls.",
            "change the artifact/control substrate before tuning the predictor",
            "current_packet_tuning_rejected",
        ),
    ]


def _candidate(
    action: str,
    disposition: str,
    reason: str,
    next_step: str,
    claim_status: str,
) -> dict[str, str]:
    return {
        "candidate_action": action,
        "disposition": disposition,
        "reason": reason,
        "next_step": next_step,
        "claim_status": claim_status,
        "requires_gpu_now": "false",
        "promotion_allowed": "false",
        "advance_to_gpu_validation": "false",
    }


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status", "missing") if payload else "missing",
        "decision": payload.get("decision", "") if payload else "",
        "claim_status": payload.get("claim_status", "") if payload else "",
    }


def _source_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"source": row["source"], "reason": f"{row['path']} is missing"}
        for row in rows
        if row["source"] != "strategy_review" and not row["present"]
    ]


def _control_status(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "missing"
    missing = [row.get("control", "") for row in rows if row.get("status") != "available"]
    return "missing=" + ";".join(missing) if missing else "all_available"


def _signal(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in rows:
        if row.get("signal") == name:
            return row
    return {}


def _strategy_review(path: Path) -> dict[str, Any]:
    result = {
        "present": path.is_file(),
        "strategic_change_level": "",
        "notify_ben": False,
        "recommended_next_action": "",
        "verdict": "",
        "ben_notification_required": False,
    }
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines()[:12]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "strategic_change_level":
            result["strategic_change_level"] = value
        elif key == "notify_ben":
            result["notify_ben"] = value.lower() == "true"
        elif key == "recommended_next_action":
            result["recommended_next_action"] = value
        elif key == "verdict":
            result["verdict"] = value
    result["ben_notification_required"] = bool(
        result["notify_ben"] or str(result["strategic_change_level"]).lower() == "major"
    )
    return result


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No strategy review was present; closeout relies on command-generated pregate artifacts."
    if strategy["ben_notification_required"]:
        return "Strategy review requested Ben notification or a major shift; this closeout records the direction shift."
    return "Strategy review was read and remains compatible with local no-GPU hidden/future closeout."


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "decision_matrix.csv", summary["decision_matrix"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    notes = [
        "# Transformer-ACSR Hidden/Future Predictor Closeout",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Promotion allowed: `{summary['promotion_allowed']}`",
        f"- Next action: `{summary['selected_next_action']}`",
        "",
        summary["rationale"],
        "",
        "RunPod and Colab validation remain blocked for this branch.",
    ]
    if summary["ben_should_be_notified"]:
        notes.append("")
        notes.append("The latest strategy review requested Ben notification or recorded a major direction shift.")
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pregate", type=Path, default=DEFAULT_PREGATE)
    parser.add_argument("--control-contract", type=Path, default=DEFAULT_CONTROL_CONTRACT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_transformer_acsr_hidden_future_predictor_closeout(
        pregate_path=args.pregate,
        control_contract_path=args.control_contract,
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
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
