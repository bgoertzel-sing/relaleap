"""Recover closed mechanism-branch state before opening the next pregate."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_CONTEXT_CLOSEOUT = Path("results/reports/context_contrastive_core_periphery_closeout/summary.json")
DEFAULT_SPARSE_FACTOR_CLOSEOUT = Path(
    "results/reports/low_churn_mlp_sparse_factorization_ceiling_closeout/summary.json"
)
DEFAULT_VALUE_DICTIONARY_CLOSEOUT = Path(
    "results/reports/low_churn_mlp_value_dictionary_capacity_rescue_closeout/summary.json"
)
DEFAULT_POST_VALUE_SELECTOR = Path("results/reports/post_value_dictionary_branch_selector/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/mechanism_branch_inventory")

DENSE_TEACHER_COLUMNABILITY_ACTION = "start_dense_teacher_columnability_pregate_before_gpu"
REPAIR_ACTION = "repair_mechanism_branch_inventory_sources"

EXPECTED_CHAIN = (
    (
        "context_contrastive_core_periphery_closeout",
        "context_contrastive_core_periphery_branch_closed",
        "design_low_churn_mlp_sparse_factorization_ceiling",
    ),
    (
        "low_churn_mlp_sparse_factorization_ceiling_closeout",
        "low_churn_mlp_sparse_factorization_vector_centroid_ceiling_closed",
        "design_value_dictionary_capacity_rescue_before_gpu",
    ),
    (
        "low_churn_mlp_value_dictionary_capacity_rescue_closeout",
        "low_churn_mlp_value_dictionary_capacity_rescue_closed",
        "select_next_post_value_dictionary_local_branch_before_gpu",
    ),
    (
        "post_value_dictionary_branch_selector",
        "post_value_dictionary_branch_selected",
        "request_strategy_review_before_new_post_value_dictionary_branch",
    ),
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "transition_rows.csv",
    "candidate_actions.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_mechanism_branch_inventory(
    *,
    context_closeout_path: Path = DEFAULT_CONTEXT_CLOSEOUT,
    sparse_factor_closeout_path: Path = DEFAULT_SPARSE_FACTOR_CLOSEOUT,
    value_dictionary_closeout_path: Path = DEFAULT_VALUE_DICTIONARY_CLOSEOUT,
    post_value_selector_path: Path = DEFAULT_POST_VALUE_SELECTOR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed inventory of closed local mechanism branches."""

    start = time.time()
    paths = {
        "context_contrastive_core_periphery_closeout": context_closeout_path,
        "low_churn_mlp_sparse_factorization_ceiling_closeout": sparse_factor_closeout_path,
        "low_churn_mlp_value_dictionary_capacity_rescue_closeout": value_dictionary_closeout_path,
        "post_value_dictionary_branch_selector": post_value_selector_path,
    }
    sources = {name: _read_json(path) for name, path in paths.items()}
    strategy = _strategy_review(strategy_review_path)
    source_rows = [_source_row(name, path, sources[name]) for name, path in paths.items()]
    source_rows.append(
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
            "selected_next_action": "",
            "requires_gpu_now": False,
            "promotion_allowed": False,
        }
    )
    transition_rows = _transition_rows(sources)
    criteria = _criteria(source_rows, transition_rows, strategy)
    failures = [row for row in criteria if not row["passed"]]
    selected_ok = not failures
    selected_next_action = DENSE_TEACHER_COLUMNABILITY_ACTION if selected_ok else REPAIR_ACTION
    candidate_actions = _candidate_actions(selected_next_action)

    summary = {
        "status": "pass" if selected_ok else "fail",
        "decision": "mechanism_branch_inventory_recorded" if selected_ok else "mechanism_branch_inventory_failed_closed",
        "claim_status": (
            "dense_teacher_columnability_pregate_selected_after_closed_branches"
            if selected_ok
            else "mechanism_branch_inventory_sources_incomplete_or_contradictory"
        ),
        "selected_next_action": selected_next_action,
        "selected_next_step": (
            "implement a local dense-teacher columnability/continual-interference pregate with matched nulls before any GPU validation"
            if selected_ok
            else "repair missing or contradictory branch closeout summaries before selecting a new mechanism branch"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local branch inventory only; RunPod and Colab remain blocked",
        "ben_should_be_notified": strategy["ben_notification_required"],
        "direction_shift_recorded": strategy["ben_notification_required"],
        "source_rows": source_rows,
        "transition_rows": transition_rows,
        "candidate_actions": candidate_actions,
        "gate_criteria": criteria,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "failures": failures,
        "rationale": _rationale(selected_ok, strategy),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status") if payload else "missing",
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
        "selected_next_action": payload.get("selected_next_action", ""),
        "requires_gpu_now": payload.get("requires_gpu_now", ""),
        "promotion_allowed": payload.get("promotion_allowed", ""),
    }


def _transition_rows(sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, expected_decision, expected_action in EXPECTED_CHAIN:
        payload = sources.get(source, {})
        rows.append(
            {
                "source": source,
                "expected_decision": expected_decision,
                "actual_decision": payload.get("decision", ""),
                "decision_matches": payload.get("decision") == expected_decision,
                "expected_selected_next_action": expected_action,
                "actual_selected_next_action": payload.get("selected_next_action", ""),
                "selected_next_action_matches": payload.get("selected_next_action") == expected_action,
                "requires_gpu_now": payload.get("requires_gpu_now", ""),
                "promotion_allowed": payload.get("promotion_allowed", ""),
            }
        )
    return rows


def _criteria(
    source_rows: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    required_sources = [row for row in source_rows if row["source"] != "strategy_review"]
    all_sources_pass = all(
        row["present"] and row["status"] == "pass" and row["requires_gpu_now"] is False
        for row in required_sources
    )
    transitions_match = all(
        row["decision_matches"] and row["selected_next_action_matches"] for row in transition_rows
    )
    no_gpu_or_promotion = all(
        row["requires_gpu_now"] is False and row["promotion_allowed"] is False for row in required_sources
    )
    review_selects_dense_columnability = (
        strategy["present"]
        and str(strategy["strategic_change_level"]).lower() == "major"
        and strategy["ben_notification_required"]
        and "dense-teacher columnability" in strategy["recommended_next_action"].lower()
    )
    return [
        _criterion(
            "required_closeouts_present_and_passing",
            all_sources_pass,
            "context, sparse-factorization, value-dictionary, and post-value summaries must pass and stay local",
            {row["source"]: {"present": row["present"], "status": row["status"]} for row in required_sources},
            "one or more required branch summaries is missing, failing, or requesting GPU",
        ),
        _criterion(
            "closed_branch_transitions_match",
            transitions_match,
            "branch summaries must encode the expected closed-branch transition chain",
            transition_rows,
            "one or more branch summaries has a contradictory decision or next action",
        ),
        _criterion(
            "closed_branches_do_not_request_gpu_or_promotion",
            no_gpu_or_promotion,
            "closed branches must not request GPU validation or default promotion",
            {row["source"]: {"requires_gpu_now": row["requires_gpu_now"], "promotion_allowed": row["promotion_allowed"]} for row in required_sources},
            "a closed branch is still requesting GPU or promotion",
        ),
        _criterion(
            "major_review_selects_dense_teacher_columnability",
            review_selects_dense_columnability,
            "urgent GPT-5.5-Pro review must select dense-teacher columnability and request Ben notification",
            {
                "present": strategy["present"],
                "strategic_change_level": strategy["strategic_change_level"],
                "notify_ben": strategy["notify_ben"],
                "recommended_next_action": strategy["recommended_next_action"],
            },
            "latest strategy review does not select the dense-teacher columnability pivot",
        ),
    ]


def _candidate_actions(selected: str) -> list[dict[str, str]]:
    return [
        {
            "candidate_action": DENSE_TEACHER_COLUMNABILITY_ACTION,
            "disposition": "selected" if selected == DENSE_TEACHER_COLUMNABILITY_ACTION else "blocked",
            "reason": "Closed-branch selector state is coherent and the urgent review selects dense-teacher columnability with matched nulls.",
            "next_step": "implement dense_teacher_columnability_pregate.py with fixture tests and fail-closed local gates",
        },
        {
            "candidate_action": "reopen_acsr_support_imitation",
            "disposition": "rejected",
            "reason": "Transformer-ACSR support imitation is closed by negligible same-student support-value headroom.",
            "next_step": "do not reopen unless a new value-bearing support landscape is captured",
        },
        {
            "candidate_action": "rerun_closed_mlp_or_sparse_selectors",
            "disposition": "rejected",
            "reason": "Dense/MLP, sparse-factorization, and value-dictionary selector chains are already closed locally.",
            "next_step": "avoid duplicate selector churn",
        },
        {
            "candidate_action": "run_gpu_validation_now",
            "disposition": "rejected",
            "reason": "No local mechanism gate has passed; the urgent review says local pivot, no GPU.",
            "next_step": "keep RunPod and Colab unused",
        },
    ]


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


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""

    def field(name: str, default: str = "") -> str:
        prefix = f"{name}:"
        for line in text.splitlines():
            if line.startswith(prefix):
                return line.split(":", 1)[1].strip()
        return default

    notify = field("notify_ben", "false")
    return {
        "path": str(path),
        "present": path.is_file(),
        "strategic_change_level": field("strategic_change_level", "unknown"),
        "notify_ben": notify,
        "ben_notification_required": notify.lower() == "true",
        "recommended_next_action": field("recommended_next_action", ""),
        "verdict": field("verdict", ""),
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if strategy["ben_notification_required"]:
        return "Accepted the major GPT-5.5-Pro pivot; Ben should be notified, and the direction shift is recorded here."
    return "Read the latest strategy review; no Ben notification requested."


def _rationale(selected_ok: bool, strategy: dict[str, Any]) -> str:
    if not selected_ok:
        return "The branch inventory failed closed because required closeout state or the strategy pivot is missing or contradictory."
    return (
        "Sparse-factorization, value-dictionary, context-contrastive, and post-value selector states are closed and coherent. "
        "The urgent GPT-5.5-Pro review is a major pivot that selects dense-teacher columnability/continual-interference "
        "as the next local pregate, so ACSR/support-imitation and MLP-selector branches stay closed and GPU remains blocked."
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "transition_rows.csv", summary["transition_rows"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    lines = [
        "# Mechanism Branch Inventory",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Ben should be notified: `{summary['ben_should_be_notified']}`",
        f"- Direction shift recorded: `{summary['direction_shift_recorded']}`",
        "",
        summary["rationale"],
    ]
    if summary["failures"]:
        lines.extend(["", "## Failures"])
        lines.extend(f"- `{row['criterion']}`: {row['failure_reason']}" for row in summary["failures"])
    (out_dir / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-closeout", type=Path, default=DEFAULT_CONTEXT_CLOSEOUT)
    parser.add_argument("--sparse-factor-closeout", type=Path, default=DEFAULT_SPARSE_FACTOR_CLOSEOUT)
    parser.add_argument("--value-dictionary-closeout", type=Path, default=DEFAULT_VALUE_DICTIONARY_CLOSEOUT)
    parser.add_argument("--post-value-selector", type=Path, default=DEFAULT_POST_VALUE_SELECTOR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_mechanism_branch_inventory(
        context_closeout_path=args.context_closeout,
        sparse_factor_closeout_path=args.sparse_factor_closeout,
        value_dictionary_closeout_path=args.value_dictionary_closeout,
        post_value_selector_path=args.post_value_selector,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
