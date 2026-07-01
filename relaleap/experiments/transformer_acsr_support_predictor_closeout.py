"""Close out the support-only Transformer-ACSR pregate branch."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_PREGATE = Path("results/reports/transformer_acsr_support_predictor_pregate/summary.json")
DEFAULT_CONTROL_CONTRACT = Path(
    "results/reports/transformer_acsr_support_predictor_pregate/control_contract.csv"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/transformer_acsr_support_predictor_closeout")

CLOSE_ACTION = "close_support_only_transformer_acsr_branch"
CAPTURE_ACTION = "design_transformer_acsr_hidden_future_tensor_capture_before_more_training"
REPAIR_ACTION = "repair_transformer_acsr_support_predictor_pregate_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "decision_matrix.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_transformer_acsr_support_predictor_closeout(
    *,
    pregate_path: Path = DEFAULT_PREGATE,
    control_contract_path: Path = DEFAULT_CONTROL_CONTRACT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Consume the support-only pregate and write a fail-closed decision report."""

    start = time.time()
    pregate = _read_json(pregate_path)
    controls = _read_csv(control_contract_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("transformer_acsr_support_predictor_pregate", pregate_path, pregate),
        {
            "source": "transformer_acsr_support_predictor_control_contract",
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
    source_failures = [
        {"source": row["source"], "reason": f"{row['path']} is missing"}
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]
    decision_matrix = _decision_matrix(pregate, controls, strategy)
    required_failures = [
        row for row in decision_matrix if row.get("required") and row.get("passed") is not True
    ]
    support_branch_closed = (
        not source_failures
        and _signal(decision_matrix, "pregate_completed").get("passed") is True
        and _signal(decision_matrix, "support_predictor_loses_to_nulls").get("passed") is True
        and _signal(decision_matrix, "downstream_controls_missing").get("passed") is True
    )
    capture_required = (
        support_branch_closed
        and _signal(decision_matrix, "hidden_future_capture_missing").get("passed") is True
    )
    selected_next_action = (
        CAPTURE_ACTION
        if capture_required
        else CLOSE_ACTION
        if support_branch_closed
        else REPAIR_ACTION
    )
    status = "pass" if support_branch_closed else "fail"
    summary = {
        "status": status,
        "decision": (
            "transformer_acsr_support_only_branch_closed_hidden_future_capture_required"
            if capture_required
            else "transformer_acsr_support_only_branch_closed"
            if support_branch_closed
            else "transformer_acsr_support_predictor_closeout_failed_closed"
        ),
        "claim_status": (
            "support_row_transformer_acsr_negative_no_gpu"
            if support_branch_closed
            else "support_predictor_closeout_sources_incomplete"
        ),
        "selected_next_action": selected_next_action,
        "selected_next_step": (
            "record a local hidden/future tensor capture design before any further Transformer-ACSR training or GPU validation"
            if capture_required
            else "keep the support-only branch closed and request a new nonduplicative local mechanism direction"
            if support_branch_closed
            else "repair or regenerate the support-predictor pregate artifact before closeout"
        ),
        "support_branch_closed": support_branch_closed,
        "hidden_future_capture_required": capture_required,
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
        "rationale": _rationale(pregate, controls, capture_required),
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
    prefix = _float(pregate.get("prefix_support_jaccard"))
    token_position = _float(pregate.get("token_position_jaccard"))
    shuffled = _float(pregate.get("shuffled_target_jaccard"))
    frequency = _float(pregate.get("frequency_jaccard"))
    return [
        {
            "signal": "pregate_completed",
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
            "expected": "completed local support-only pregate with no GPU advancement",
        },
        {
            "signal": "support_predictor_loses_to_nulls",
            "required": True,
            "passed": (
                pregate.get("null_margin_gate_passes") is False
                and prefix is not None
                and token_position is not None
                and shuffled is not None
                and frequency is not None
                and prefix <= max(token_position, shuffled, frequency)
            ),
            "actual": {
                "prefix_support_jaccard": prefix,
                "token_position_jaccard": token_position,
                "shuffled_target_jaccard": shuffled,
                "frequency_jaccard": frequency,
                "null_margin_gate_passes": pregate.get("null_margin_gate_passes"),
            },
            "expected": "primary support predictor must fail registered null-margin gates",
        },
        {
            "signal": "downstream_controls_missing",
            "required": True,
            "passed": (
                pregate.get("downstream_intervention_budget_gate_passes") is False
                and "exact_arbitrary_pair_same_student_intervention" in missing_controls
                and "retention_churn_budget" in missing_controls
                and "finite_update_commutator_budget" in missing_controls
            ),
            "actual": {"missing_controls": missing_controls},
            "expected": "same-student, retention/churn, and commutator controls remain unavailable",
        },
        {
            "signal": "hidden_future_capture_missing",
            "required": True,
            "passed": "hidden_future_chunk_targets" in missing_controls,
            "actual": {"missing_controls": missing_controls},
            "expected": "support rows lack the hidden/future tensor targets needed for Ben's Transformer-ACSR branch",
        },
        {
            "signal": "strategy_review_supports_local_no_gpu",
            "required": False,
            "passed": (
                strategy["present"]
                and str(strategy["strategic_change_level"]).lower() != "major"
                and "Transformer-ACSR" in strategy["recommended_next_action"]
            ),
            "actual": {
                "strategic_change_level": strategy["strategic_change_level"],
                "notify_ben": strategy["notify_ben"],
                "recommended_next_action": strategy["recommended_next_action"],
            },
            "expected": "latest review should remain compatible with local no-GPU Transformer-ACSR work",
        },
    ]


def _candidate_actions(selected: str) -> list[dict[str, str]]:
    return [
        {
            "candidate_action": CAPTURE_ACTION,
            "disposition": "selected" if selected == CAPTURE_ACTION else "deferred",
            "reason": "The support-row transformer ties or loses to shortcut/frequency/shuffled nulls and lacks the hidden/future targets Ben requested for the real Transformer-ACSR test.",
            "next_step": "write a bounded local tensor-capture design/report that identifies exact source artifacts, leakage labels, split keys, and same-student intervention requirements",
        },
        {
            "candidate_action": CLOSE_ACTION,
            "disposition": "recorded" if selected in {CAPTURE_ACTION, CLOSE_ACTION} else "blocked",
            "reason": "The support-only row branch is negative local evidence and should not be extended as a support-only predictor.",
            "next_step": "do not rerun support-only training or send it to RunPod",
        },
        {
            "candidate_action": "run_runpod_transformer_acsr_support_predictor_validation",
            "disposition": "rejected",
            "reason": "Local null-margin and downstream intervention/budget gates failed.",
            "next_step": "keep RunPod unused for this branch",
        },
        {
            "candidate_action": "continue_support_only_predictor_tuning",
            "disposition": "rejected",
            "reason": "Additional support-row tuning would duplicate a failed support-only mechanism surface without hidden/future tensors or exact intervention rows.",
            "next_step": "only revisit after source capture changes the artifact substrate",
        },
    ]


def _rationale(
    pregate: dict[str, Any],
    controls: list[dict[str, str]],
    capture_required: bool,
) -> str:
    missing = [
        row.get("control", "")
        for row in controls
        if row.get("status") not in {"available", "present"}
    ]
    return (
        "The local support-only Transformer-ACSR pregate is negative: prefix-support heldout "
        f"Jaccard={pregate.get('prefix_support_jaccard')} ties token/position and frequency "
        f"controls and loses to shuffled-target Jaccard={pregate.get('shuffled_target_jaccard')}. "
        f"Missing controls are {', '.join(missing) or 'none'}. "
        + (
            "The support-only branch is closed; further Transformer-ACSR work requires hidden/future tensor capture before training."
            if capture_required
            else "The support-only branch is closed and no GPU validation is justified."
        )
    )


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status", "missing") if payload else "missing",
        "decision": payload.get("decision", "") if payload else "",
        "claim_status": payload.get("claim_status", "") if payload else "",
    }


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
    return "Strategy review was read and remains compatible with local no-GPU Transformer-ACSR closeout."


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


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "decision_matrix.csv", summary["decision_matrix"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    notes = [
        "# Transformer-ACSR Support Predictor Closeout",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Support branch closed: `{summary['support_branch_closed']}`",
        f"- Hidden/future capture required: `{summary['hidden_future_capture_required']}`",
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
    summary = run_transformer_acsr_support_predictor_closeout(
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
