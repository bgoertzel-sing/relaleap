"""Close out the regret-soft utility-head branch before GPU validation."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_PROBE = Path("results/reports/regret_soft_utility_head_probe/summary.json")
DEFAULT_POST_FLAT_SELECTOR = Path("results/reports/post_flat_value_branch_selector/summary.json")
DEFAULT_DENSE_TEACHER_CLOSEOUT = Path("results/reports/dense_teacher_residual_distillation_closeout/summary.json")
DEFAULT_POST_DENSE_SELECTOR = Path("results/reports/post_dense_teacher_control_branch_selector/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/regret_soft_utility_head_closeout")

CLOSE_REGRET_ACTION = "close_regret_soft_utility_head_branch_before_gpu"
LOW_CHURN_MLP_ACTION = "design_low_churn_mlp_residual_control_pregate"
REPAIR_ACTION = "repair_regret_soft_utility_head_closeout_sources"
REPEAT_DIRECT_ACTION = "repeat_direct_regret_soft_utility_head_before_gpu"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "decision_matrix.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_regret_soft_utility_head_closeout(
    *,
    probe_path: Path = DEFAULT_PROBE,
    post_flat_selector_path: Path = DEFAULT_POST_FLAT_SELECTOR,
    dense_teacher_closeout_path: Path = DEFAULT_DENSE_TEACHER_CLOSEOUT,
    post_dense_selector_path: Path = DEFAULT_POST_DENSE_SELECTOR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed closeout and select one local next mechanism path."""

    start = time.time()
    probe = _read_json(probe_path)
    post_flat = _read_json(post_flat_selector_path)
    dense_closeout = _read_json(dense_teacher_closeout_path)
    post_dense = _read_json(post_dense_selector_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("regret_soft_utility_head_probe", probe_path, probe),
        _source_row("post_flat_value_branch_selector", post_flat_selector_path, post_flat),
        _source_row("dense_teacher_residual_distillation_closeout", dense_teacher_closeout_path, dense_closeout),
        _source_row("post_dense_teacher_control_branch_selector", post_dense_selector_path, post_dense),
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
        },
    ]
    failures = _source_failures(source_rows)
    decision_matrix = _decision_matrix(probe, post_flat, dense_closeout, post_dense, strategy)
    candidate_actions = _candidate_actions(decision_matrix, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "regret_soft_utility_head_closeout_failed_closed"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair missing regret-soft utility-head closeout source artifacts"
        claim_status = "regret_soft_utility_head_closeout_sources_incomplete"
        rationale = "The closeout cannot choose a branch until required local source artifacts are present."
    else:
        selected_row = selected[0]
        status = "pass"
        decision = "regret_soft_utility_head_branch_closed"
        selected_next_action = selected_row["candidate_action"]
        selected_next_step = selected_row["next_step"]
        claim_status = selected_row["claim_status"]
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
        "backend_policy": "local closeout only; RunPod and Colab remain blocked until a local mechanism gate passes",
        "source_rows": source_rows,
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
    probe: dict[str, Any],
    post_flat: dict[str, Any],
    dense_closeout: dict[str, Any],
    post_dense: dict[str, Any],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "signal": "direct_regret_soft_probe_failed",
            "status": probe.get("status"),
            "decision": probe.get("decision"),
            "claim_status": probe.get("claim_status"),
            "supports_closeout": (
                probe.get("status") == "pass"
                and probe.get("selected_next_action") == "close_regret_soft_utility_head_probe_before_gpu"
                and probe.get("advance_to_gpu_validation") is False
                and _int(probe.get("direct_regret_soft_row_count")) > 0
                and _int(probe.get("passing_direct_row_count")) == 0
            ),
            "observed": {
                "direct_regret_soft_row_count": probe.get("direct_regret_soft_row_count"),
                "passing_direct_row_count": probe.get("passing_direct_row_count"),
                "proxy_row_count": probe.get("proxy_row_count"),
            },
        },
        {
            "signal": "post_flat_selector_already_redirected",
            "status": post_flat.get("status"),
            "decision": post_flat.get("decision"),
            "claim_status": post_flat.get("claim_status"),
            "supports_not_repeating_flat_value": (
                post_flat.get("status") == "pass"
                and post_flat.get("selected_next_action") == "run_local_dense_teacher_residual_distillation_comparison"
                and post_flat.get("advance_to_gpu_validation") is False
            ),
            "observed": {"selected_next_action": post_flat.get("selected_next_action")},
        },
        {
            "signal": "dense_teacher_distillation_closed",
            "status": dense_closeout.get("status"),
            "decision": dense_closeout.get("decision"),
            "claim_status": dense_closeout.get("claim_status"),
            "supports_downstream_local_selector": (
                dense_closeout.get("status") == "pass"
                and dense_closeout.get("selected_next_action") == "close_dense_teacher_residual_distillation_before_gpu"
                and dense_closeout.get("advance_to_gpu_validation") is False
            ),
            "observed": {"selected_next_step": dense_closeout.get("selected_next_step")},
        },
        {
            "signal": "post_dense_selector_local_path",
            "status": post_dense.get("status"),
            "decision": post_dense.get("decision"),
            "claim_status": post_dense.get("claim_status"),
            "supports_low_churn_mlp_path": (
                post_dense.get("status") == "pass"
                and post_dense.get("selected_next_action") == LOW_CHURN_MLP_ACTION
                and post_dense.get("requires_gpu_now") is False
            ),
            "observed": {"selected_next_action": post_dense.get("selected_next_action")},
        },
        {
            "signal": "external_strategy_review",
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": f"verdict={strategy['verdict']}",
            "supports_no_gpu": "runpod" in strategy["recommended_next_action"].lower()
            or "gpu" in strategy["recommended_next_action"].lower(),
            "observed": {
                "strategic_change_level": strategy["strategic_change_level"],
                "notify_ben": strategy["notify_ben"],
            },
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
                "required source artifacts are missing",
                "repair or regenerate the missing local source reports",
                "source_artifact_repair_required",
            )
        ]

    regret_failed = _signal(decision_matrix, "direct_regret_soft_probe_failed").get("supports_closeout") is True
    flat_redirected = (
        _signal(decision_matrix, "post_flat_selector_already_redirected").get("supports_not_repeating_flat_value")
        is True
    )
    dense_closed = _signal(decision_matrix, "dense_teacher_distillation_closed").get(
        "supports_downstream_local_selector"
    ) is True
    low_churn_ready = _signal(decision_matrix, "post_dense_selector_local_path").get(
        "supports_low_churn_mlp_path"
    ) is True

    if regret_failed and flat_redirected and dense_closed and low_churn_ready:
        return [
            _candidate(
                LOW_CHURN_MLP_ACTION,
                "selected",
                (
                    "direct regret-soft/listwise utility rows failed learned-router and null gates, flat-value "
                    "and dense-teacher redirects are already closed locally, and the downstream dense/MLP "
                    "selector names a low-churn MLP pregate as the least ambiguous local mechanism path"
                ),
                "design the local low-churn MLP residual-control pregate before any GPU validation",
                "regret_soft_closed_low_churn_mlp_pregate_selected_no_gpu",
            ),
            _candidate(
                CLOSE_REGRET_ACTION,
                "recorded",
                "the regret-soft utility-head branch is closed before GPU by direct local gate failure",
                "keep regret-soft/listwise utility heads closed unless a materially new local signal appears",
                "regret_soft_utility_head_closed_before_gpu",
            ),
        ]

    if regret_failed:
        return [
            _candidate(
                CLOSE_REGRET_ACTION,
                "selected",
                "direct regret-soft rows failed, but downstream selector sources are incomplete",
                "record the regret-soft closeout and repair downstream branch-selection sources",
                "regret_soft_closed_downstream_sources_incomplete",
            )
        ]

    return [
        _candidate(
            REPEAT_DIRECT_ACTION,
            "selected",
            "the direct regret-soft probe has not failed in the expected closeout shape",
            "repeat or repair the direct regret-soft utility-head probe before closing the branch",
            "regret_soft_probe_not_ready_for_closeout",
        )
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


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


def _strategy_review(path: Path) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "present": path.is_file(),
        "strategic_change_level": "minor",
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
        if key in defaults:
            defaults[key] = value.strip()
    defaults["ben_notification_required"] = (
        str(defaults["notify_ben"]).lower() == "true"
        or str(defaults["strategic_change_level"]).lower() == "major"
    )
    return defaults


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No external strategy review was present; used local closeout artifacts only."
    if strategy["ben_notification_required"]:
        return "External review requested Ben notification or marked a major shift; this closeout records it and stays local."
    return "External review consumed; its no-GPU/fail-closed recommendation is preserved by this local closeout."


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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
    if not fieldnames:
        fieldnames = ["status"]
        rows = [{"status": "missing"}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Regret-Soft Utility-Head Closeout",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Selected action: `{summary['selected_next_action']}`",
            f"- Claim status: `{summary['claim_status']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            f"- Promotion allowed: `{summary['promotion_allowed']}`",
            f"- Next step: {summary['selected_next_step']}",
            "",
            "GPU validation remains blocked. Direct regret-soft/listwise utility-head rows failed local gates, so this report closes that branch and redirects only to one local mechanism step.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--post-flat-selector", type=Path, default=DEFAULT_POST_FLAT_SELECTOR)
    parser.add_argument("--dense-teacher-closeout", type=Path, default=DEFAULT_DENSE_TEACHER_CLOSEOUT)
    parser.add_argument("--post-dense-selector", type=Path, default=DEFAULT_POST_DENSE_SELECTOR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_regret_soft_utility_head_closeout(
        probe_path=args.probe,
        post_flat_selector_path=args.post_flat_selector,
        dense_teacher_closeout_path=args.dense_teacher_closeout,
        post_dense_selector_path=args.post_dense_selector,
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
