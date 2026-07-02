"""Select the next branch after the mechanism-factorized CL repeat gate."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_SCALE_CLOSEOUT = Path(
    "results/reports/scale_constrained_sparse_residual_compression_closeout/summary.json"
)
DEFAULT_MECHANISM_REPEAT = Path(
    "results/reports/mechanism_factorized_continual_learning_repeat/summary.json"
)
DEFAULT_INTERFERENCE_MITIGATION = Path(
    "results/reports/residual_interference_mitigation_probe/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/mechanism_factorized_cl_branch_selector")

REPAIR_ACTION = "repair_mechanism_factorized_cl_branch_selector_sources"
PIVOT_ACTION = "pivot_to_commutator_dense_teacher_source_inventory"
REPEAT_ACTION = "repeat_mechanism_factorized_cl_or_topk2_mitigation"
GPU_ACTION = "launch_gpu_validation_for_mechanism_factorized_cl"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "decision_matrix.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_mechanism_factorized_cl_branch_selector(
    *,
    scale_closeout_path: Path = DEFAULT_SCALE_CLOSEOUT,
    mechanism_repeat_path: Path = DEFAULT_MECHANISM_REPEAT,
    interference_mitigation_path: Path = DEFAULT_INTERFERENCE_MITIGATION,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Reconcile the mechanism CL repeat with the prior compression closeout."""

    start = time.time()
    scale_closeout = _read_json(scale_closeout_path)
    repeat = _read_json(mechanism_repeat_path)
    mitigation = _read_json(interference_mitigation_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("scale_constrained_sparse_residual_compression_closeout", scale_closeout_path, scale_closeout),
        _source_row("mechanism_factorized_continual_learning_repeat", mechanism_repeat_path, repeat),
        _source_row("residual_interference_mitigation_probe", interference_mitigation_path, mitigation),
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
            "selected_next_step": "",
        },
    ]
    failures = _source_failures(source_rows)
    decision_matrix = _decision_matrix(scale_closeout, repeat, mitigation, strategy)
    candidate_actions = _candidate_actions(decision_matrix, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "mechanism_factorized_cl_branch_selector_failed_closed"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair missing or inconsistent mechanism-factorized CL selector sources"
        claim_status = "mechanism_factorized_cl_selector_sources_incomplete"
        rationale = "Required local source artifacts are missing or inconsistent, so no branch can be selected."
    else:
        selected_row = selected[0]
        status = "pass"
        decision = "mechanism_factorized_cl_branch_selected"
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
        "training_executed": False,
        "backend_policy": "local branch selection only; RunPod and Colab remain blocked",
        "source_rows": source_rows,
        "decision_matrix": decision_matrix,
        "candidate_actions": candidate_actions,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy, scale_closeout),
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
    scale_closeout: dict[str, Any],
    repeat: dict[str, Any],
    mitigation: dict[str, Any],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    repeat_primary = _as_dict(repeat.get("primary_result"))
    return [
        {
            "signal": "compression_path_closed_and_redirected_here",
            "status": scale_closeout.get("status"),
            "decision": scale_closeout.get("decision"),
            "claim_status": scale_closeout.get("claim_status"),
            "passed": (
                scale_closeout.get("status") == "pass"
                and scale_closeout.get("selected_next_action")
                == "redirect_to_mechanism_factorized_continual_learning_local_gate"
                and scale_closeout.get("advance_to_gpu_validation") is False
            ),
            "observed": {
                "selected_next_action": scale_closeout.get("selected_next_action"),
                "ce_gap_sparse_minus_flat": _as_dict(scale_closeout.get("evidence")).get("ce_gap_sparse_minus_flat"),
                "mse_gap_sparse_minus_flat": _as_dict(scale_closeout.get("evidence")).get("mse_gap_sparse_minus_flat"),
            },
        },
        {
            "signal": "mechanism_repeat_blocks_sparse_retention_claim",
            "status": repeat.get("status"),
            "decision": repeat.get("decision"),
            "claim_status": repeat.get("claim_status"),
            "passed": (
                repeat.get("status") == "pass"
                and repeat.get("claim_status") == "mechanism_factorized_sparse_retention_not_established"
                and repeat.get("topk2_tradeoff_repeat_status") == "not_replicated"
                and repeat.get("requires_gpu_now") is False
            ),
            "observed": {
                "topk2_tradeoff_repeat_status": repeat.get("topk2_tradeoff_repeat_status"),
                "topk2_tradeoff_supporting_seed_count": repeat_primary.get(
                    "topk2_tradeoff_supporting_seed_count"
                ),
                "full_sparse_claim_supporting_seed_count": repeat_primary.get(
                    "full_sparse_claim_supporting_seed_count"
                ),
                "selected_next_step": repeat.get("selected_next_step"),
            },
        },
        {
            "signal": "first_seed_interference_mitigation_is_partial_only",
            "status": mitigation.get("status"),
            "decision": mitigation.get("decision"),
            "claim_status": mitigation.get("claim_status"),
            "passed": (
                mitigation.get("status") == "pass"
                and mitigation.get("claim_status") == "support_width_mitigation_partial_candidate_not_promoted"
                and mitigation.get("requires_gpu_now") is False
            ),
            "observed": {
                "selected_next_step": mitigation.get("selected_next_step"),
                "topk2_minus_dense_target_ce_delta": _as_dict(mitigation.get("primary_result")).get(
                    "topk2_minus_dense_target_ce_delta"
                ),
            },
        },
        {
            "signal": "strategy_review_contingency_accepted",
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": f"verdict={strategy['verdict']}",
            "passed": (
                not strategy["present"]
                or "scale-constrained sparse residual-compression pilot" in strategy["recommended_next_action"]
            ),
            "observed": {
                "strategic_change_level": strategy["strategic_change_level"],
                "notify_ben": strategy["notify_ben"],
                "ben_notification_required": strategy["ben_notification_required"],
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
                "required local source artifacts are missing",
                "repair or regenerate the missing mechanism-factorized CL selector sources",
                "source_artifact_repair_required",
            )
        ]

    compression_redirect = _signal(decision_matrix, "compression_path_closed_and_redirected_here").get("passed")
    repeat_blocks = _signal(decision_matrix, "mechanism_repeat_blocks_sparse_retention_claim").get("passed")
    mitigation_partial = _signal(decision_matrix, "first_seed_interference_mitigation_is_partial_only").get("passed")
    if compression_redirect and repeat_blocks and mitigation_partial:
        return [
            _candidate(
                PIVOT_ACTION,
                "selected",
                (
                    "residual compression is retired, the two-seed mechanism-factorized CL repeat "
                    "does not replicate the top-k2 tradeoff, and the only support-width mitigation "
                    "signal is a first-seed partial candidate; the next useful local step is an "
                    "inventory that chooses between commutator and dense-teacher mechanisms without GPU"
                ),
                "add a local commutator/dense-teacher source inventory before any new training or GPU validation",
                "mechanism_factorized_cl_closed_local_inventory_selected_no_gpu",
            ),
            _candidate(
                REPEAT_ACTION,
                "rejected",
                "another repeat would duplicate an already non-replicated two-seed sparse-retention gate",
                "only repeat after a new mechanism changes the local gates",
                "repeat_rejected_after_nonreplication",
            ),
            _candidate(
                GPU_ACTION,
                "rejected",
                "no mechanism-factorized sparse-retention or support-width claim survived local repeat controls",
                "do not use RunPod or Colab for this branch",
                "gpu_blocked_by_local_repeat",
            ),
        ]

    return [
        _candidate(
            REPAIR_ACTION,
            "selected",
            "source artifacts are present but do not encode the expected closed-compression plus blocked-repeat state",
            "inspect source summaries and regenerate this selector after resolving stale evidence",
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
        "selected_next_step": payload.get("selected_next_step") if payload else "",
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
    fields: dict[str, Any] = {
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
        if key in fields:
            fields[key] = value.strip()
    fields["ben_notification_required"] = (
        str(fields["notify_ben"]).lower() == "true"
        or fields["strategic_change_level"] == "major"
    )
    fields["path"] = str(path)
    return fields


def _strategy_review_handling(
    strategy: dict[str, Any],
    scale_closeout: dict[str, Any],
) -> str:
    if strategy.get("ben_notification_required"):
        return (
            "Read the latest external review; it requests Ben notification or a major shift, "
            "so this selector records that condition and keeps GPU/default changes blocked."
        )
    if scale_closeout.get("strategy_review_handling"):
        return (
            "Read the latest external review. The scale-constrained closeout already accepted its "
            "pilot recommendation and contingency; this selector accepts the resulting local pivot "
            "and rejects no GPT-5.5-Pro recommendation."
        )
    return "Read the latest external review; no recommendation is rejected in this selector."


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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
            "# Mechanism-Factorized CL Branch Selector",
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
    parser.add_argument("--scale-closeout", type=Path, default=DEFAULT_SCALE_CLOSEOUT)
    parser.add_argument("--mechanism-repeat", type=Path, default=DEFAULT_MECHANISM_REPEAT)
    parser.add_argument("--interference-mitigation", type=Path, default=DEFAULT_INTERFERENCE_MITIGATION)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_mechanism_factorized_cl_branch_selector(
        scale_closeout_path=args.scale_closeout,
        mechanism_repeat_path=args.mechanism_repeat,
        interference_mitigation_path=args.interference_mitigation,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
