"""Close out or redirect the same-router flat value-capacity diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_DIAGNOSTIC = Path("results/reports/same_router_flat_value_capacity_diagnostic/summary.json")
DEFAULT_BUDGET_ROWS = Path("results/reports/same_router_flat_value_capacity_diagnostic/budget_rows.csv")
DEFAULT_GATE_ROWS = Path("results/reports/same_router_flat_value_capacity_diagnostic/gate_rows.csv")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/same_router_flat_value_capacity_closeout")

REPAIR_ACTION = "repair_same_router_flat_value_capacity_closeout_sources"
COMMUTATOR_MITIGATION_ACTION = "design_flat_value_finite_update_commutator_mitigation"
CLOSE_GENERIC_ACTION = "close_flat_value_capacity_as_generic_capacity_before_gpu"
REPEAT_ACTION = "repeat_same_router_flat_value_capacity_diagnostic_before_gpu"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "closeout_rows.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_same_router_flat_value_capacity_closeout(
    *,
    diagnostic_path: Path = DEFAULT_DIAGNOSTIC,
    budget_rows_path: Path = DEFAULT_BUDGET_ROWS,
    gate_rows_path: Path = DEFAULT_GATE_ROWS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Consume the flat-value diagnostic and select exactly one local follow-up."""

    start = time.time()
    diagnostic = _read_json(diagnostic_path)
    budget_rows = _read_csv(budget_rows_path)
    gate_rows = _read_csv(gate_rows_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_json("same_router_flat_value_capacity_diagnostic", diagnostic_path, diagnostic),
        _source_csv("same_router_flat_value_capacity_budget_rows", budget_rows_path, budget_rows),
        _source_csv("same_router_flat_value_capacity_gate_rows", gate_rows_path, gate_rows),
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
    evidence = _evidence(diagnostic, budget_rows, gate_rows, strategy)
    failures = _failures(source_rows, evidence)
    closeout_rows = _closeout_rows(evidence)
    candidate_actions = _candidate_actions(evidence, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "same_router_flat_value_capacity_closeout_failed_closed"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair same-router flat-value closeout source artifacts"
        claim_status = "source_artifacts_incomplete"
        rationale = "Required diagnostic source artifacts are missing, contradictory, or require Ben notification."
    else:
        selected_row = selected[0]
        status = "pass"
        decision = "same_router_flat_value_capacity_branch_closed_or_redirected"
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
        "backend_policy": "local closeout only; RunPod and Colab remain blocked until a local mitigation clears CE/control and budget gates",
        "source_rows": source_rows,
        "evidence": evidence,
        "closeout_rows": closeout_rows,
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


def _evidence(
    diagnostic: dict[str, Any],
    budget_rows: list[dict[str, str]],
    gate_rows: list[dict[str, str]],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    primary = _as_dict(diagnostic.get("primary_result"))
    gates = {row.get("gate", ""): _bool_or_none(row.get("passes")) for row in gate_rows}
    commutator_budget = next(
        (row for row in budget_rows if row.get("budget") == "finite_update_commutator"),
        {},
    )
    return {
        "diagnostic_status": diagnostic.get("status"),
        "diagnostic_decision": diagnostic.get("decision"),
        "diagnostic_claim_status": diagnostic.get("claim_status"),
        "diagnostic_passes": diagnostic.get("diagnostic_passes"),
        "selected_next_action": diagnostic.get("selected_next_action"),
        "flat_beats_promoted_sparse": _first_bool(
            primary.get("flat_beats_promoted_sparse"),
            gates.get("flat_beats_promoted_sparse"),
        ),
        "flat_beats_support_policy_nulls": _first_bool(
            primary.get("flat_beats_support_policy_nulls"),
            gates.get("flat_beats_support_policy_nulls"),
        ),
        "flat_beats_dense_mlp_controls": _first_bool(
            primary.get("flat_beats_dense_mlp_controls"),
            gates.get("flat_beats_dense_mlp_controls"),
        ),
        "interference_budgets_nonworse": _first_bool(
            primary.get("interference_budgets_nonworse"),
            gates.get("interference_budgets_nonworse"),
        ),
        "oracle_regret_strata_available": _first_bool(
            primary.get("oracle_regret_strata_available"),
            gates.get("oracle_regret_strata_available"),
        ),
        "commutator_budget_passes": _bool_or_none(commutator_budget.get("nonworse_gate_passes")),
        "commutator_candidate_value": _float_or_none(commutator_budget.get("candidate_value")),
        "commutator_reference_value": _float_or_none(commutator_budget.get("reference_budget_value")),
        "commutator_failure_reason": commutator_budget.get("failure_reason", ""),
        "ben_notification_required": strategy["notify_ben"]
        or strategy["strategic_change_level"] == "major",
        "strategy_verdict": strategy["verdict"],
        "strategy_recommended_next_action": strategy["recommended_next_action"],
    }


def _closeout_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "branch": "same_router_flat_value_capacity",
            "source_decision": evidence["diagnostic_decision"],
            "disposition": "blocked_before_gpu",
            "reason": (
                "flat value capacity clears CE/control gates but fails the finite-update commutator budget"
                if _ce_controls_pass(evidence) and evidence["commutator_budget_passes"] is False
                else "flat value capacity did not clear all local diagnostic gates"
            ),
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
        {
            "branch": "finite_update_commutator_mitigation",
            "source_decision": evidence["diagnostic_decision"],
            "disposition": "redirect_target"
            if _ce_controls_pass(evidence) and evidence["commutator_budget_passes"] is False
            else "deferred",
            "reason": evidence["commutator_failure_reason"],
            "candidate_value": evidence["commutator_candidate_value"],
            "reference_value": evidence["commutator_reference_value"],
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
        {
            "branch": "gpu_validation",
            "source_decision": "local_closeout",
            "disposition": "blocked",
            "reason": "no local flat-value capacity gate permits GPU validation",
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
    ]


def _candidate_actions(evidence: dict[str, Any], failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if failures:
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "required closeout sources are missing, contradictory, or require Ben notification",
                "repair source artifacts before continuing",
                "source_repair_required",
            )
        ]
    if evidence["diagnostic_passes"] is True:
        return [
            _candidate(
                REPEAT_ACTION,
                "selected",
                "the flat-value diagnostic passed locally and needs a repeat before backend spend",
                "repeat same-router flat-value capacity diagnostic on an adjacent seed before any GPU validation",
                "flat_value_repeat_required_before_gpu",
            )
        ]
    if _ce_controls_pass(evidence) and evidence["commutator_budget_passes"] is False:
        return [
            _candidate(
                COMMUTATOR_MITIGATION_ACTION,
                "selected",
                "flat value capacity has a local CE/control signal, but the finite-update commutator budget is the sole blocking mechanism gate",
                "design a local flat-value finite-update commutator mitigation with the same CE/null/dense controls and nonworse norm/churn gates",
                "flat_value_signal_blocked_by_commutator_mitigation_selected",
            ),
            _candidate(
                CLOSE_GENERIC_ACTION,
                "deferred",
                "closing as generic capacity should wait until a bounded commutator mitigation fails",
                "revisit after commutator mitigation design/probe",
                "deferred",
            ),
        ]
    return [
        _candidate(
            CLOSE_GENERIC_ACTION,
            "selected",
            "flat value capacity did not clear enough local gates to justify a focused mitigation",
            "close flat value capacity as generic capacity before GPU and return to dense/MLP control-first work",
            "flat_value_capacity_closed_as_generic_capacity",
        ),
        _candidate(
            COMMUTATOR_MITIGATION_ACTION,
            "rejected",
            "commutator mitigation is not warranted without a clean CE/control signal",
            "only reconsider if flat value capacity clears support-policy and dense/MLP controls",
            "rejected",
        ),
    ]


def _ce_controls_pass(evidence: dict[str, Any]) -> bool:
    return all(
        evidence.get(key) is True
        for key in (
            "flat_beats_promoted_sparse",
            "flat_beats_support_policy_nulls",
            "flat_beats_dense_mlp_controls",
            "oracle_regret_strata_available",
        )
    )


def _failures(source_rows: list[dict[str, Any]], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    failures = [
        {"source": row["source"], "reason": "missing_required_source", "path": row["path"]}
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]
    if evidence["diagnostic_status"] != "pass":
        failures.append({"source": "diagnostic", "reason": "diagnostic_status_not_pass"})
    if evidence["ben_notification_required"]:
        failures.append({"source": "strategy_review", "reason": "ben_notification_required"})
    return failures


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
        "claim_status": payload.get("claim_status", ""),
    }


def _source_csv(source: str, path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(rows),
        "status": "present" if rows else "missing",
        "decision": "",
        "claim_status": "",
        "row_count": len(rows),
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "present": False,
            "strategic_change_level": "minor",
            "notify_ben": False,
            "recommended_next_action": "",
            "verdict": "",
        }
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:12]:
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    return {
        "present": True,
        "strategic_change_level": header.get("strategic_change_level", "minor"),
        "notify_ben": header.get("notify_ben", "false").lower() == "true",
        "recommended_next_action": header.get("recommended_next_action", ""),
        "verdict": header.get("verdict", ""),
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No external strategy review was present; proceeded from local diagnostic artifacts."
    return (
        "Read the latest external review. Its no-RunPod hidden-classifier recommendation remains satisfied; "
        "this closeout records the downstream flat-value commutator blocker without launching GPU validation."
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_bool(*values: Any) -> bool | None:
    for value in values:
        parsed = _bool_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    return None


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "closeout_rows.csv", summary["closeout_rows"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    _write_notes(out_dir / "notes.md", summary)


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


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    evidence = summary["evidence"]
    lines = [
        "# Same-Router Flat-Value-Capacity Closeout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected action: `{summary['selected_next_action']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Advance to GPU validation: `{summary['advance_to_gpu_validation']}`",
        f"- Flat clears CE/control gates: `{_ce_controls_pass(evidence)}`",
        f"- Commutator budget passes: `{evidence['commutator_budget_passes']}`",
        f"- Next step: {summary['selected_next_step']}",
        "",
        "GPU validation remains blocked. The flat-value branch needs a local finite-update commutator mitigation before any repeat or promotion claim.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--budget-rows", type=Path, default=DEFAULT_BUDGET_ROWS)
    parser.add_argument("--gate-rows", type=Path, default=DEFAULT_GATE_ROWS)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_same_router_flat_value_capacity_closeout(
        diagnostic_path=args.diagnostic,
        budget_rows_path=args.budget_rows,
        gate_rows_path=args.gate_rows,
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
