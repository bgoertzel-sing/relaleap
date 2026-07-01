"""Close out hidden/future teacher-support imitation after support-value headroom."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_HEADROOM = Path("results/reports/transformer_acsr_hidden_future_support_value_headroom/summary.json")
DEFAULT_CONTROL_AUDIT = Path("results/reports/transformer_acsr_hidden_future_control_audit/summary.json")
DEFAULT_PREGATE = Path("results/reports/transformer_acsr_hidden_future_predictor_pregate/summary.json")
DEFAULT_POST_SELECTOR = Path("results/reports/post_core_periphery_contextual_dense_branch_selector/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/transformer_acsr_hidden_future_support_value_closeout")

CLOSE_ACTION = "close_transformer_acsr_teacher_support_imitation_before_gpu"
REPAIR_ACTION = "repair_transformer_acsr_support_value_closeout_sources"
DENSE_TRACK_ACTION = "continue_dense_mlp_mechanism_track_with_causal_router_diagnostics"
VALUE_ROUTER_ACTION = "train_prefix_safe_value_target_router_locally"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "decision_matrix.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_transformer_acsr_hidden_future_support_value_closeout(
    *,
    headroom_path: Path = DEFAULT_HEADROOM,
    control_audit_path: Path = DEFAULT_CONTROL_AUDIT,
    pregate_path: Path = DEFAULT_PREGATE,
    post_selector_path: Path = DEFAULT_POST_SELECTOR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Consume local hidden/future artifacts and select one bounded next action."""

    start = time.time()
    headroom = _read_json(headroom_path)
    control = _read_json(control_audit_path)
    pregate = _read_json(pregate_path)
    post_selector = _read_json(post_selector_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("support_value_headroom", headroom_path, headroom),
        _source_row("hidden_future_control_audit", control_audit_path, control),
        _source_row("hidden_future_predictor_pregate", pregate_path, pregate),
        _source_row("post_core_periphery_contextual_dense_branch_selector", post_selector_path, post_selector),
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
    decision_matrix = _decision_matrix(headroom, control, pregate, post_selector, strategy)
    failures = _source_failures(source_rows) + [
        row for row in decision_matrix if row["required"] and not row["passed"]
    ]
    candidate_actions = _candidate_actions(headroom, post_selector, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "transformer_acsr_support_value_closeout_failed_closed"
        claim_status = "support_value_closeout_sources_incomplete"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair or regenerate hidden/future support-value closeout source artifacts"
        rationale = "Required local source artifacts are missing or do not support a coherent closeout."
    else:
        selected_row = selected[0]
        status = "pass"
        decision = "transformer_acsr_teacher_support_imitation_closed_before_gpu"
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
        "backend_policy": "local closeout/branch selection only; RunPod and Colab remain blocked",
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
    headroom: dict[str, Any],
    control: dict[str, Any],
    pregate: dict[str, Any],
    post_selector: dict[str, Any],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _criterion(
            "support_value_headroom_audit_passed",
            headroom.get("status") == "pass",
            True,
            "support-value headroom audit must pass before closeout",
            headroom.get("decision"),
        ),
        _criterion(
            "oracle_headroom_negligible_on_train_and_heldout",
            (
                headroom.get("value_target_training_allowed") is False
                and _float(headroom.get("train_mean_oracle_router_gap")) < _float(headroom.get("headroom_threshold"))
                and _float(headroom.get("heldout_mean_oracle_router_gap")) < _float(headroom.get("headroom_threshold"))
            ),
            True,
            "train and heldout oracle-router gaps must be below the preregistered headroom threshold",
            {
                "train_mean_oracle_router_gap": headroom.get("train_mean_oracle_router_gap"),
                "heldout_mean_oracle_router_gap": headroom.get("heldout_mean_oracle_router_gap"),
                "headroom_threshold": headroom.get("headroom_threshold"),
            },
        ),
        _criterion(
            "teacher_and_predicted_support_underperform_student_router",
            (
                _split_metric(headroom, "heldout", "mean_teacher_router_delta") > 0.0
                and _split_metric(headroom, "heldout", "mean_predicted_router_delta") > 0.0
            ),
            True,
            "heldout teacher and predicted supports should not beat the current same-student router",
            {
                "heldout_mean_teacher_router_delta": _split_metric(
                    headroom, "heldout", "mean_teacher_router_delta"
                ),
                "heldout_mean_predicted_router_delta": _split_metric(
                    headroom, "heldout", "mean_predicted_router_delta"
                ),
            },
        ),
        _criterion(
            "local_controls_do_not_clear_mechanism_gate",
            (
                control.get("status") == "pass"
                and control.get("same_student_loss_gate_passes") is False
                and control.get("advance_to_gpu_validation") is False
            ),
            True,
            "control audit must keep mechanism and GPU gates closed",
            control.get("decision"),
        ),
        _criterion(
            "prefix_hidden_predicts_teacher_but_not_value",
            (
                pregate.get("status") == "pass"
                and pregate.get("null_margin_gate_passes") is True
                and pregate.get("same_student_loss_gate_passes") is False
            ),
            False,
            "prefix hidden may reconstruct teacher support but still fail value utility",
            {
                "prefix_hidden_jaccard": pregate.get("prefix_hidden_jaccard"),
                "prefix_hidden_mean_forced_minus_student_router_loss": pregate.get(
                    "prefix_hidden_mean_forced_minus_student_router_loss"
                ),
            },
        ),
        _criterion(
            "post_sparse_selector_has_local_next_mechanism",
            (
                post_selector.get("status") == "pass"
                and post_selector.get("selected_next_action") == DENSE_TRACK_ACTION
                and post_selector.get("requires_gpu_now") is False
            ),
            True,
            "after closing teacher imitation, an existing local selector must provide the next mechanism track",
            post_selector.get("selected_next_action"),
        ),
        _criterion(
            "strategy_review_accepts_no_gpu_headroom_gate",
            (
                strategy["present"]
                and strategy["strategic_change_level"] != "major"
                and "headroom" in strategy["recommended_next_action"].lower()
            ),
            False,
            "latest strategy review recommends support-value headroom before training or GPU",
            strategy["recommended_next_action"],
        ),
    ]


def _candidate_actions(
    headroom: dict[str, Any],
    post_selector: dict[str, Any],
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if failures:
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "required closeout sources are missing or fail required gates",
                "repair or regenerate the hidden/future support-value closeout source artifacts",
                "source_repair_required",
            )
        ]
    value_training_allowed = headroom.get("value_target_training_allowed") is True
    dense_next_step = str(post_selector.get("next_step") or post_selector.get("selected_next_step") or "")
    return [
        _candidate(
            VALUE_ROUTER_ACTION,
            "selected" if value_training_allowed else "rejected",
            "only train a value-target router if oracle support has nontrivial same-student value headroom",
            "train a prefix-safe value-target router locally",
            "value_router_allowed_by_headroom" if value_training_allowed else "blocked_by_negligible_headroom",
        ),
        _candidate(
            CLOSE_ACTION,
            "selected" if not value_training_allowed else "deferred",
            "oracle-vs-router headroom is below threshold on train and heldout, while teacher/predicted supports are worse than the same-student router",
            "record teacher-support imitation as closed for this packet before GPU",
            "teacher_support_imitation_closed_negligible_action_value",
        ),
        _candidate(
            DENSE_TRACK_ACTION,
            "next_after_closeout" if not value_training_allowed else "deferred",
            "the existing post-sparse selector chooses the dense/MLP mechanism track as the next local source of truth",
            dense_next_step
            or "run a bounded local dense/MLP mechanism follow-up using causal-router diagnostics as context",
            "next_local_mechanism_after_closeout",
        ),
    ]


def _criterion(
    signal: str,
    passed: bool,
    required: bool,
    expected: str,
    actual: Any,
) -> dict[str, Any]:
    return {
        "signal": signal,
        "required": required,
        "passed": bool(passed),
        "expected": expected,
        "actual": actual,
    }


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
    failures = []
    for row in source_rows[:4]:
        if not row["present"] or row["status"] not in {"pass", "ok"}:
            failures.append(
                {
                    "source": row["source"],
                    "path": row["path"],
                    "reason": "required source artifact missing or not passing",
                    "status": row["status"],
                }
            )
    return failures


def _split_metric(summary: dict[str, Any], split: str, key: str) -> float:
    for row in summary.get("split_summary", []):
        if row.get("split") == split:
            return _float(row.get(key))
    return 0.0


def _float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _strategy_review(path: Path) -> dict[str, Any]:
    strategy = {
        "present": path.is_file(),
        "path": str(path),
        "strategic_change_level": "unknown",
        "notify_ben": False,
        "verdict": "unknown",
        "recommended_next_action": "",
        "ben_notification_required": False,
    }
    if not path.is_file():
        return strategy
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", maxsplit=1)]
        if key in strategy:
            strategy[key] = value
    notify = str(strategy["notify_ben"]).lower() == "true"
    major = str(strategy["strategic_change_level"]).lower() == "major"
    strategy["notify_ben"] = notify
    strategy["ben_notification_required"] = notify or major
    return strategy


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No strategy review was available; the closeout used local command-generated artifacts."
    if strategy["ben_notification_required"]:
        return (
            "Accepted the review direction where scientifically sensible; the review requires Ben "
            "notification because notify_ben is true or the change level is major."
        )
    return (
        "Accepted the latest GPT-5.5-Pro recommendation to use local same-student "
        "support-value headroom before training, commutator work, or GPU validation."
    )


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "decision_matrix.csv", summary["decision_matrix"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    notes = [
        "# Transformer-ACSR Hidden/Future Support-Value Closeout",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected action: `{summary['selected_next_action']}`",
        f"- Selected next step: `{summary['selected_next_step']}`",
        f"- RunPod/Colab validation: `{summary['advance_to_gpu_validation']}`",
        "",
        "Teacher-support imitation is closed for this packet when exact same-student",
        "support-value headroom is negligible. The dense/MLP mechanism selector is the",
        "next local source to follow after recording this closeout.",
    ]
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
    parser.add_argument("--headroom", type=Path, default=DEFAULT_HEADROOM)
    parser.add_argument("--control-audit", type=Path, default=DEFAULT_CONTROL_AUDIT)
    parser.add_argument("--pregate", type=Path, default=DEFAULT_PREGATE)
    parser.add_argument("--post-selector", type=Path, default=DEFAULT_POST_SELECTOR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_transformer_acsr_hidden_future_support_value_closeout(
        headroom_path=args.headroom,
        control_audit_path=args.control_audit,
        pregate_path=args.pregate,
        post_selector_path=args.post_selector,
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
