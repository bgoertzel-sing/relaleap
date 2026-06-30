"""Select the next branch after the hidden support-classifier closeout."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_HIDDEN_AUDIT = Path("results/reports/hidden_support_classifier_sequence_ood_budget_audit/summary.json")
DEFAULT_CLOSEOUT_ROWS = Path("results/reports/hidden_support_classifier_sequence_ood_budget_audit/closeout_rows.csv")
DEFAULT_SEED_REPEAT = Path("results/reports/transformer_acsr_seed_repeat/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/hidden_support_classifier_branch_selector")

REDESIGN_SUPPORT_OBJECTIVE_ACTION = "redesign_hidden_support_objective_before_any_gpu"
RETURN_LEARNED_ROUTER_ACTION = "return_to_learned_router_non_pc_sparse_value_branch"
REPAIR_SOURCES_ACTION = "repair_hidden_support_classifier_branch_selector_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_hidden_support_classifier_branch_selector(
    *,
    hidden_audit_path: Path = DEFAULT_HIDDEN_AUDIT,
    closeout_rows_path: Path = DEFAULT_CLOSEOUT_ROWS,
    seed_repeat_path: Path = DEFAULT_SEED_REPEAT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Choose between hidden-head redesign and returning to stronger learned-router work."""

    start = time.time()
    hidden_audit = _read_json(hidden_audit_path)
    closeout_rows = _read_csv(closeout_rows_path)
    seed_repeat = _read_json(seed_repeat_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("hidden_support_classifier_sequence_ood_budget_audit", hidden_audit_path, hidden_audit),
        _csv_source_row("hidden_support_classifier_closeout_rows", closeout_rows_path, closeout_rows),
        _source_row("transformer_acsr_seed_repeat", seed_repeat_path, seed_repeat),
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
    failures = _failures(source_rows)
    evidence = _evidence(hidden_audit, closeout_rows, seed_repeat, strategy)
    candidate_actions = _candidate_actions(evidence, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "hidden_support_classifier_branch_selector_failed_closed"
        selected_next_action = REPAIR_SOURCES_ACTION
        selected_next_step = "repair hidden support-classifier branch-selector source artifacts"
        claim_status = "source_artifacts_incomplete"
        rationale = "The selector cannot choose a branch until the hidden audit and closeout artifacts are present."
    else:
        status = "pass"
        decision = "hidden_support_classifier_branch_selected"
        selected_next_action = selected[0]["candidate_action"]
        selected_next_step = selected[0]["next_step"]
        claim_status = selected[0]["claim_status"]
        rationale = selected[0]["reason"]

    summary = {
        "status": status,
        "decision": decision,
        "selected_next_action": selected_next_action,
        "selected_next_step": selected_next_step,
        "claim_status": claim_status,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "source_rows": source_rows,
        "evidence": evidence,
        "candidate_actions": candidate_actions,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy, selected_next_action),
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
    hidden_audit: dict[str, Any],
    closeout_rows: list[dict[str, str]],
    seed_repeat: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    closeout = closeout_rows[0] if closeout_rows else {}
    return {
        "hidden_audit_status": hidden_audit.get("status"),
        "hidden_audit_decision": hidden_audit.get("decision"),
        "close_hidden_classifier_branch": hidden_audit.get("close_hidden_classifier_branch"),
        "closeout_status": hidden_audit.get("closeout_status") or closeout.get("status"),
        "sequence_heldout_gate_passes": hidden_audit.get("sequence_heldout_gate_passes"),
        "rule_combo_heldout_gate_passes": hidden_audit.get("rule_combo_heldout_gate_passes"),
        "budget_gate_passes": hidden_audit.get("budget_gate_passes"),
        "mean_hidden_classifier_ce_gain_vs_learned_router": _float_or_none(
            hidden_audit.get("mean_hidden_classifier_ce_gain_vs_learned_router")
        ),
        "mean_oracle_regret_recovery_vs_learned_router": _float_or_none(
            hidden_audit.get("mean_oracle_regret_recovery_vs_learned_router")
        ),
        "deferred_exact_row_reason": closeout.get("deferred_exact_row_reason", ""),
        "seed_repeat_hidden_gate_pass_count": seed_repeat.get("hidden_classifier_gate_pass_count"),
        "seed_repeat_hidden_null_margin_gate_passes": seed_repeat.get("hidden_classifier_null_margin_gate_passes"),
        "seed_repeat_value_aware_gate_pass_count": seed_repeat.get("value_aware_gate_pass_count"),
        "seed_repeat_advance_to_gpu_validation": seed_repeat.get("advance_to_gpu_validation"),
        "strategy_recommended_next_action": strategy["recommended_next_action"],
        "strategy_notify_ben": strategy["notify_ben"],
        "strategy_change_level": strategy["strategic_change_level"],
    }


def _candidate_actions(evidence: dict[str, Any], failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if failures:
        return [
            _candidate(
                REPAIR_SOURCES_ACTION,
                "selected",
                "required source artifacts are missing",
                "repair hidden support-classifier branch-selector source artifacts",
                "source_artifacts_incomplete",
            )
        ]

    hidden_closed = (
        evidence["hidden_audit_status"] == "pass"
        and evidence["close_hidden_classifier_branch"] is True
        and evidence["sequence_heldout_gate_passes"] is False
        and _float_or_none(evidence["mean_hidden_classifier_ce_gain_vs_learned_router"]) is not None
        and _float_or_none(evidence["mean_hidden_classifier_ce_gain_vs_learned_router"]) < 0.0
    )
    if hidden_closed:
        return [
            _candidate(
                RETURN_LEARNED_ROUTER_ACTION,
                "selected",
                "the direct hidden classifier beats weak nulls but loses sequence-heldout same-student CE to the learned router, so another hidden-head redesign is not the next bounded branch",
                "return to the stronger learned-router/non-PC sparse-value branch with local null and interference gates before any GPU validation",
                "direct_hidden_classifier_closed_learned_router_branch_active",
            ),
            _candidate(
                REDESIGN_SUPPORT_OBJECTIVE_ACTION,
                "deferred",
                "redesign needs a new causal signal beyond the closed direct classifier objective",
                "only reopen after a new objective/head can specify stronger learned-router, rule-OOD, and budget gates",
                "requires_new_causal_signal",
            ),
        ]

    return [
        _candidate(
            REDESIGN_SUPPORT_OBJECTIVE_ACTION,
            "selected",
            "the hidden branch is not formally closed by the available source artifacts",
            "design a stronger hidden support objective with learned-router, rule-OOD, and exact budget gates before GPU",
            "hidden_classifier_needs_exact_redesign_gate",
        ),
        _candidate(
            RETURN_LEARNED_ROUTER_ACTION,
            "deferred",
            "returning to learned-router work should wait until the hidden branch is explicitly closed",
            "revisit after closeout evidence",
            "deferred",
        ),
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


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(payload),
        "status": payload.get("status", "missing"),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", payload.get("closeout_status", "")),
    }


def _csv_source_row(source: str, path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    row = rows[0] if rows else {}
    return {
        "source": source,
        "path": str(path),
        "present": bool(rows),
        "status": row.get("status", "missing"),
        "decision": row.get("next_step", ""),
        "claim_status": row.get("deferred_exact_row_reason", ""),
        "row_count": len(rows),
    }


def _failures(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = {
        "hidden_support_classifier_sequence_ood_budget_audit",
        "hidden_support_classifier_closeout_rows",
        "transformer_acsr_seed_repeat",
    }
    return [
        {
            "source": row["source"],
            "reason": "required source artifact missing",
            "path": row["path"],
        }
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


def _strategy_review_handling(strategy: dict[str, Any], selected_next_action: str) -> str:
    if not strategy["present"]:
        return "no external strategy review present; local source artifacts selected the branch"
    if selected_next_action == RETURN_LEARNED_ROUTER_ACTION:
        return (
            "Accepted the no-RunPod/fail-closed part of the latest review. "
            "Did not duplicate hidden-classifier audit work because the current closeout already shows "
            "negative learned-router comparison; returning to the learned-router/non-PC sparse-value branch."
        )
    return "Accepted the latest review's hidden-classifier fail-closed direction."


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


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
        "# Hidden Support-Classifier Branch Selector",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Selected next step: `{summary['selected_next_step']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Advance to GPU validation: `{summary['advance_to_gpu_validation']}`",
        "",
        summary["rationale"],
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
    parser.add_argument("--hidden-audit", type=Path, default=DEFAULT_HIDDEN_AUDIT)
    parser.add_argument("--closeout-rows", type=Path, default=DEFAULT_CLOSEOUT_ROWS)
    parser.add_argument("--seed-repeat", type=Path, default=DEFAULT_SEED_REPEAT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_hidden_support_classifier_branch_selector(
        hidden_audit_path=args.hidden_audit,
        closeout_rows_path=args.closeout_rows,
        seed_repeat_path=args.seed_repeat,
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
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
