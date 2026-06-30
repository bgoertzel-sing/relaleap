"""Select the next local branch after the low-churn MLP control pilot."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_LOW_CHURN_PILOT = Path("results/reports/low_churn_mlp_residual_control_pilot/summary.json")
DEFAULT_ACSR_CLOSEOUT = Path("results/reports/acsr_negative_evidence_closeout/summary.json")
DEFAULT_CORE_CLOSEOUT = Path("results/reports/core_periphery_negative_evidence_closeout/summary.json")
DEFAULT_POST_CORE_SELECTOR = Path("results/reports/post_core_periphery_contextual_dense_branch_selector/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/post_low_churn_mlp_branch_selector")

REPAIR_ACTION = "repair_post_low_churn_mlp_branch_selector_sources"
CONTEXT_CONTRASTIVE_CORE_PERIPHERY_ACTION = (
    "design_context_contrastive_core_periphery_probe_before_gpu"
)
REPEAT_LOW_CHURN_MLP_ACTION = "repeat_low_churn_mlp_control_only_after_advancement_gate"
REOPEN_ACSR_ACTION = "reopen_acsr_only_after_new_dense_control_signal"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "decision_matrix.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_post_low_churn_mlp_branch_selector(
    *,
    low_churn_pilot_path: Path = DEFAULT_LOW_CHURN_PILOT,
    acsr_closeout_path: Path = DEFAULT_ACSR_CLOSEOUT,
    core_closeout_path: Path = DEFAULT_CORE_CLOSEOUT,
    post_core_selector_path: Path = DEFAULT_POST_CORE_SELECTOR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Reconcile the latest local dense/MLP control result with sparse branch closeouts."""

    start = time.time()
    low_churn = _read_json(low_churn_pilot_path)
    acsr = _read_json(acsr_closeout_path)
    core = _read_json(core_closeout_path)
    post_core = _read_json(post_core_selector_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("low_churn_mlp_residual_control_pilot", low_churn_pilot_path, low_churn),
        _source_row("acsr_negative_evidence_closeout", acsr_closeout_path, acsr),
        _source_row("core_periphery_negative_evidence_closeout", core_closeout_path, core),
        _source_row("post_core_periphery_contextual_dense_branch_selector", post_core_selector_path, post_core),
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
    decision_matrix = _decision_matrix(low_churn, acsr, core, post_core, strategy)
    failures = _source_failures(source_rows)
    candidate_actions = _candidate_actions(decision_matrix, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "post_low_churn_mlp_branch_selector_failed_closed"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair missing post-low-churn branch selector source artifacts"
        claim_status = "post_low_churn_branch_sources_incomplete"
        rationale = "The selector cannot choose a branch until required local source artifacts are present."
    else:
        selected_row = selected[0]
        status = "pass"
        decision = "post_low_churn_mlp_branch_selected"
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
        "backend_policy": "local branch selection only; RunPod and Colab remain blocked until a local mechanism gate passes",
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
    low_churn: dict[str, Any],
    acsr: dict[str, Any],
    core: dict[str, Any],
    post_core: dict[str, Any],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "signal": "low_churn_mlp_control_pilot_blocked",
            "status": low_churn.get("status"),
            "decision": low_churn.get("decision"),
            "claim_status": low_churn.get("claim_status"),
            "supports_sparse_core_redirect": (
                low_churn.get("status") == "pass"
                and low_churn.get("scientific_gate") == "blocked"
                and low_churn.get("advancement_row_count") == 0
                and low_churn.get("selected_next_action") == "return_to_sparse_core_periphery_mechanism_work"
                and low_churn.get("advance_to_gpu_validation") is False
            ),
            "observed": {
                "selected_next_action": low_churn.get("selected_next_action"),
                "advancement_row_count": low_churn.get("advancement_row_count"),
                "scientific_gate": low_churn.get("scientific_gate"),
            },
        },
        {
            "signal": "acsr_promotion_path_demoted",
            "status": acsr.get("status"),
            "decision": acsr.get("decision"),
            "claim_status": acsr.get("claim_status"),
            "supports_no_acsr_reopen": (
                acsr.get("status") == "pass"
                and acsr.get("selected_next_action") == "demote_acsr_to_diagnostic_status"
                and acsr.get("requires_gpu_now") is False
            ),
            "observed": {"selected_next_action": acsr.get("selected_next_action")},
        },
        {
            "signal": "current_core_periphery_attempt_demoted",
            "status": core.get("status"),
            "decision": core.get("decision"),
            "claim_status": core.get("claim_status"),
            "supports_new_core_design_only": (
                core.get("status") == "pass"
                and core.get("selected_next_action") == "demote_current_core_periphery_mechanism_to_diagnostic_status"
                and core.get("requires_gpu_now") is False
            ),
            "observed": {"next_step": core.get("next_step")},
        },
        {
            "signal": "post_core_selector_dense_track_exhausted_by_low_churn_pilot",
            "status": post_core.get("status"),
            "decision": post_core.get("decision"),
            "claim_status": post_core.get("claim_status"),
            "supports_return_to_mechanism_design": (
                post_core.get("status") == "pass"
                and post_core.get("selected_next_action")
                == "continue_dense_mlp_mechanism_track_with_causal_router_diagnostics"
                and post_core.get("requires_gpu_now") is False
            ),
            "observed": {"selected_next_action": post_core.get("selected_next_action")},
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
) -> list[dict[str, Any]]:
    if failures:
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "required post-low-churn source artifacts are missing",
                "repair or regenerate the missing local source reports",
                "source_artifact_repair_required",
            )
        ]

    low_churn_blocked = _signal(decision_matrix, "low_churn_mlp_control_pilot_blocked").get(
        "supports_sparse_core_redirect"
    )
    acsr_demoted = _signal(decision_matrix, "acsr_promotion_path_demoted").get("supports_no_acsr_reopen")
    core_design_only = _signal(decision_matrix, "current_core_periphery_attempt_demoted").get(
        "supports_new_core_design_only"
    )
    dense_track_consumed = _signal(
        decision_matrix,
        "post_core_selector_dense_track_exhausted_by_low_churn_pilot",
    ).get("supports_return_to_mechanism_design")

    if low_churn_blocked and acsr_demoted and core_design_only and dense_track_consumed:
        return [
            _candidate(
                CONTEXT_CONTRASTIVE_CORE_PERIPHERY_ACTION,
                "selected",
                (
                    "the dense/MLP low-churn pilot did not produce an advancement row, ACSR remains "
                    "diagnostic only, and the current core/periphery mechanism was demoted; the next "
                    "coherent local step is a new context-contrastive core/periphery design with null, "
                    "dense/MLP, retention, pruning, and commutator gates"
                ),
                "design a bounded local context-contrastive core/periphery probe before any GPU validation",
                "context_contrastive_core_periphery_design_selected_no_gpu",
            ),
            _candidate(
                REPEAT_LOW_CHURN_MLP_ACTION,
                "rejected",
                "the low-churn MLP pilot has zero advancement rows, so repeating it would duplicate a blocked control",
                "only repeat after a new low-churn variant clears the advancement gate locally",
                "rejected",
            ),
            _candidate(
                REOPEN_ACSR_ACTION,
                "rejected",
                "ACSR promotion is demoted by parameter-matched, dense, churn, and commutator evidence",
                "only reopen ACSR after new positive matched-control evidence",
                "rejected",
            ),
        ]

    return [
        _candidate(
            REPAIR_ACTION,
            "selected",
            "source artifacts are present but do not encode the expected blocked dense/MLP plus demoted sparse state",
            "inspect source reports and regenerate the stale branch selector",
            "branch_state_inconsistent",
        )
    ]


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(payload),
        "status": payload.get("status") if payload else "missing",
        "decision": payload.get("decision") if payload else "",
        "claim_status": payload.get("claim_status") if payload else "",
    }


def _source_failures(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"source": row["source"], "reason": "missing_required_source", "path": row["path"]}
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]


def _signal(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("signal") == name), {})


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
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except FileNotFoundError:
        return {}


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    fields = {
        "present": bool(text),
        "strategic_change_level": "unknown",
        "notify_ben": "unknown",
        "recommended_next_action": "",
        "verdict": "",
    }
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in fields:
            fields[key] = value
    fields["ben_notification_required"] = (
        str(fields["notify_ben"]).lower() == "true"
        or fields["strategic_change_level"] == "major"
    )
    return fields


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if strategy.get("ben_notification_required"):
        return (
            "Read the latest external review; it requests Ben notification or a major shift, "
            "so this selector records the condition but still keeps all GPU/default changes blocked."
        )
    return (
        "Read the latest external review. Its no-RunPod/fail-closed recommendation remains compatible "
        "with the local branch-selection result; no recommendation is rejected in this selector."
    )


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "decision_matrix.csv", summary["decision_matrix"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


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
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Post Low-Churn MLP Branch Selector",
            "",
            f"- Status: {summary['status']}",
            f"- Decision: {summary['decision']}",
            f"- Claim status: {summary['claim_status']}",
            f"- Selected next action: {summary['selected_next_action']}",
            f"- Selected next step: {summary['selected_next_step']}",
            "- GPU validation remains blocked: requires_gpu_now=false, promotion_allowed=false, advance_to_gpu_validation=false.",
            f"- Strategy review handling: {summary['strategy_review_handling']}",
            "",
            "## Rationale",
            "",
            str(summary["rationale"]),
            "",
        ]
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-churn-pilot", type=Path, default=DEFAULT_LOW_CHURN_PILOT)
    parser.add_argument("--acsr-closeout", type=Path, default=DEFAULT_ACSR_CLOSEOUT)
    parser.add_argument("--core-closeout", type=Path, default=DEFAULT_CORE_CLOSEOUT)
    parser.add_argument("--post-core-selector", type=Path, default=DEFAULT_POST_CORE_SELECTOR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_post_low_churn_mlp_branch_selector(
        low_churn_pilot_path=args.low_churn_pilot,
        acsr_closeout_path=args.acsr_closeout,
        core_closeout_path=args.core_closeout,
        post_core_selector_path=args.post_core_selector,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
