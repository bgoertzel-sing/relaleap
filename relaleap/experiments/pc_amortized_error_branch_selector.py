"""Select the next branch after the amortized PC error-target closeout."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_SYNTHETIC_SUMMARY = Path("results/reports/synthetic_mechanism_causal_modularity/summary.json")
DEFAULT_PC_CLOSEOUT_ROWS = Path(
    "results/reports/synthetic_mechanism_causal_modularity/pc_amortized_error_pregate_closeout.csv"
)
DEFAULT_LEARNED_ROUTER_CLOSEOUT = Path("results/reports/learned_router_sparse_value_closeout/summary.json")
DEFAULT_FLAT_VALUE_CLOSEOUT = Path("results/reports/same_router_flat_value_capacity_closeout/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/pc_amortized_error_branch_selector")

COMMUTATOR_MITIGATION_ACTION = "return_to_flat_value_commutator_mitigation"
REOPEN_PC_ACTION = "reopen_pc_error_target_only_with_new_causal_signal"
REPAIR_ACTION = "repair_pc_amortized_error_branch_selector_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_pc_amortized_error_branch_selector(
    *,
    synthetic_summary_path: Path = DEFAULT_SYNTHETIC_SUMMARY,
    pc_closeout_rows_path: Path = DEFAULT_PC_CLOSEOUT_ROWS,
    learned_router_closeout_path: Path = DEFAULT_LEARNED_ROUTER_CLOSEOUT,
    flat_value_closeout_path: Path = DEFAULT_FLAT_VALUE_CLOSEOUT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed selector after the local amortized-PC pregate failure."""

    start = time.time()
    synthetic = _read_json(synthetic_summary_path)
    pc_closeout_rows = _read_csv(pc_closeout_rows_path)
    learned_router = _read_json(learned_router_closeout_path)
    flat_value = _read_json(flat_value_closeout_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_json("synthetic_mechanism_causal_modularity", synthetic_summary_path, synthetic),
        _source_csv("pc_amortized_error_closeout_rows", pc_closeout_rows_path, pc_closeout_rows),
        _source_json("learned_router_sparse_value_closeout", learned_router_closeout_path, learned_router),
        _source_json("same_router_flat_value_capacity_closeout", flat_value_closeout_path, flat_value),
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
            "row_count": "",
        },
    ]
    failures = _source_failures(source_rows)
    evidence = _evidence(synthetic, pc_closeout_rows, learned_router, flat_value)
    candidate_actions = _candidate_actions(evidence, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "pc_amortized_error_branch_selector_failed_closed"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair PC amortized error branch-selector source artifacts"
        claim_status = "source_artifacts_incomplete"
        rationale = "The selector cannot choose a branch until required source artifacts are present."
    else:
        status = "pass"
        decision = "pc_amortized_error_branch_selected"
        selected_next_action = selected[0]["candidate_action"]
        selected_next_step = selected[0]["next_step"]
        claim_status = selected[0]["claim_status"]
        rationale = selected[0]["reason"]

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_action": selected_next_action,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "source_rows": source_rows,
        "evidence": evidence,
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
    _write_artifacts(out_dir, summary, source_rows, candidate_actions)
    return summary


def _evidence(
    synthetic: dict[str, Any],
    pc_closeout_rows: list[dict[str, str]],
    learned_router: dict[str, Any],
    flat_value: dict[str, Any],
) -> dict[str, Any]:
    closeout = pc_closeout_rows[0] if pc_closeout_rows else {}
    pc_summary = synthetic.get("pc_amortized_error_pregate_closeout_primary_result")
    if not isinstance(pc_summary, dict):
        pc_summary = {}
    return {
        "synthetic_status": synthetic.get("status"),
        "synthetic_decision": synthetic.get("decision"),
        "synthetic_selected_next_step": synthetic.get("selected_next_step"),
        "pc_closeout_status": pc_summary.get("closeout_status") or closeout.get("closeout_status"),
        "pc_current_error_target_path_closed": _bool(
            pc_summary.get("current_error_target_path_closed"),
            closeout.get("current_error_target_path_closed"),
        ),
        "pc_source_pregate_passes": _bool(
            pc_summary.get("source_pregate_passes"),
            closeout.get("source_pregate_passes"),
        ),
        "pc_all_target_nulls_clear": _bool(
            pc_summary.get("all_target_nulls_clear"),
            closeout.get("all_target_nulls_clear"),
        ),
        "pc_flat_dense_controls_clear": _bool(
            pc_summary.get("flat_dense_controls_clear"),
            closeout.get("flat_dense_controls_clear"),
        ),
        "pc_interference_budgets_clear": _bool(
            pc_summary.get("interference_budgets_clear"),
            closeout.get("interference_budgets_clear"),
        ),
        "pc_branch_reopen_requires_new_causal_signal": _bool(
            pc_summary.get("branch_reopen_requires_new_causal_signal"),
            closeout.get("branch_reopen_requires_new_causal_signal"),
        ),
        "pc_selected_next_experiment": (
            pc_summary.get("selected_next_experiment") or closeout.get("selected_next_experiment", "")
        ),
        "pc_source_failure_reasons": closeout.get("source_failure_reasons", ""),
        "learned_router_closeout_status": learned_router.get("status"),
        "learned_router_selected_next_action": learned_router.get("selected_next_action"),
        "learned_router_claim_status": learned_router.get("claim_status"),
        "flat_value_closeout_status": flat_value.get("status"),
        "flat_value_selected_next_action": flat_value.get("selected_next_action"),
        "flat_value_claim_status": flat_value.get("claim_status"),
        "flat_value_commutator_budget_passes": (
            flat_value.get("evidence", {}).get("commutator_budget_passes")
            if isinstance(flat_value.get("evidence"), dict)
            else None
        ),
    }


def _candidate_actions(evidence: dict[str, Any], failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if failures:
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "required source artifacts are missing",
                "repair PC amortized error branch-selector source artifacts",
                "source_artifacts_incomplete",
            )
        ]

    pc_closed = bool(
        evidence["synthetic_status"] == "pass"
        and evidence["pc_closeout_status"] == "closed_current_label_free_amortized_pc_target_path"
        and evidence["pc_current_error_target_path_closed"] is True
        and evidence["pc_source_pregate_passes"] is False
    )
    flat_mitigation_ready = bool(
        evidence["flat_value_closeout_status"] == "pass"
        and evidence["flat_value_selected_next_action"] == "design_flat_value_finite_update_commutator_mitigation"
    )
    if pc_closed and flat_mitigation_ready:
        return [
            _candidate(
                COMMUTATOR_MITIGATION_ACTION,
                "selected",
                (
                    "the current label-free amortized PC target path fails signal, null, flat-control, and "
                    "commutator gates; the strongest remaining local non-PC branch is flat-value commutator mitigation"
                ),
                (
                    "implement the local flat-value finite-update commutator mitigation probe with the same "
                    "CE/null/dense controls and nonworse norm/churn gates before any GPU validation"
                ),
                "pc_closed_flat_value_commutator_mitigation_active",
            ),
            _candidate(
                REOPEN_PC_ACTION,
                "deferred",
                "the PC closeout explicitly requires a new causal signal before reopening the error-target path",
                "reopen only if a new causal error target is specified with stronger null and budget gates",
                "requires_new_causal_signal",
            ),
        ]

    return [
        _candidate(
            REOPEN_PC_ACTION,
            "selected" if not pc_closed else "deferred",
            (
                "the PC closeout is not complete enough to redirect"
                if not pc_closed
                else "flat-value mitigation source is unavailable, so no non-PC redirect can be selected"
            ),
            "repair or redesign PC/non-PC branch sources before GPU",
            "pc_or_redirect_sources_incomplete",
        )
    ]


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _source_json(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(payload),
        "status": payload.get("status", "missing"),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", payload.get("closeout_status", "")),
        "row_count": "",
    }


def _source_csv(source: str, path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    row = rows[0] if rows else {}
    return {
        "source": source,
        "path": str(path),
        "present": bool(rows),
        "status": row.get("closeout_status", "missing"),
        "decision": row.get("selected_next_experiment", ""),
        "claim_status": row.get("source_failure_reasons", ""),
        "row_count": len(rows),
    }


def _source_failures(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = {
        "synthetic_mechanism_causal_modularity",
        "pc_amortized_error_closeout_rows",
        "learned_router_sparse_value_closeout",
        "same_router_flat_value_capacity_closeout",
    }
    return [
        {"source": row["source"], "path": row["path"], "reason": "required source artifact missing"}
        for row in source_rows
        if row["source"] in required and not row["present"]
    ]


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "present": False,
            "strategic_change_level": "unknown",
            "notify_ben": False,
            "recommended_next_action": "",
            "verdict": "missing",
        }
    parsed: dict[str, Any] = {
        "present": True,
        "strategic_change_level": "unknown",
        "notify_ben": False,
        "recommended_next_action": "",
        "verdict": "unknown",
    }
    for line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if key == "notify_ben":
            parsed[key] = value.lower() == "true"
        elif key in {"strategic_change_level", "recommended_next_action", "verdict"}:
            parsed[key] = value
    return parsed


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "no external strategy review present; local source artifacts selected the branch"
    return (
        "Accepted the latest review's no-RunPod/fail-closed hidden-classifier direction as already satisfied by "
        "source artifacts. This selector does not duplicate that work; it follows the newer local PC closeout and "
        "keeps GPU validation blocked."
    )


def _bool(*values: Any) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value == "":
                continue
            if value.lower() == "true":
                return True
            if value.lower() == "false":
                return False
    return None


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    candidate_actions: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", source_rows)
    _write_csv(out_dir / "candidate_actions.csv", candidate_actions)
    notes = [
        "# PC Amortized Error Branch Selector",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Selected next step: `{summary['selected_next_step']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Advance to GPU validation: `{summary['advance_to_gpu_validation']}`",
        "",
        str(summary["rationale"]),
        "",
        "GPU validation remains blocked until a local branch clears the relevant CE, null/control, and interference gates.",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-summary", type=Path, default=DEFAULT_SYNTHETIC_SUMMARY)
    parser.add_argument("--pc-closeout-rows", type=Path, default=DEFAULT_PC_CLOSEOUT_ROWS)
    parser.add_argument("--learned-router-closeout", type=Path, default=DEFAULT_LEARNED_ROUTER_CLOSEOUT)
    parser.add_argument("--flat-value-closeout", type=Path, default=DEFAULT_FLAT_VALUE_CLOSEOUT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_pc_amortized_error_branch_selector(
        synthetic_summary_path=args.synthetic_summary,
        pc_closeout_rows_path=args.pc_closeout_rows,
        learned_router_closeout_path=args.learned_router_closeout,
        flat_value_closeout_path=args.flat_value_closeout,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
