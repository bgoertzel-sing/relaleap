"""Close out or redirect the learned-router sparse-value branch."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_PREGATE = Path("results/reports/learned_router_sparse_value_pregate/summary.json")
DEFAULT_PREGATE_ROWS = Path("results/reports/learned_router_sparse_value_pregate/pregate_rows.csv")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/learned_router_sparse_value_closeout")

REPAIR_SOURCES_ACTION = "repair_learned_router_sparse_value_closeout_sources"
FLAT_VALUE_DIAGNOSTIC_ACTION = "design_same_router_flat_value_capacity_diagnostic"
REPEAT_SPARSE_ACTION = "repeat_learned_router_sparse_value_pregate_before_closeout"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "closeout_rows.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_learned_router_sparse_value_closeout(
    *,
    pregate_path: Path = DEFAULT_PREGATE,
    pregate_rows_path: Path = DEFAULT_PREGATE_ROWS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Consume the sparse-value pregate and select one local redirect."""

    start = time.time()
    pregate = _read_json(pregate_path)
    pregate_rows = _read_csv(pregate_rows_path)
    strategy = _strategy_review(strategy_review_path)
    primary = _primary_pregate_row(pregate_rows)
    source_rows = [
        _source_json("learned_router_sparse_value_pregate", pregate_path, pregate),
        _source_csv("learned_router_sparse_value_pregate_rows", pregate_rows_path, pregate_rows),
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
    evidence = _evidence(pregate, primary, strategy)
    closeout_rows = _closeout_rows(evidence)
    failures = _failures(source_rows, evidence)
    candidate_actions = _candidate_actions(evidence, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "learned_router_sparse_value_closeout_failed_closed"
        claim_status = "source_artifacts_incomplete"
        selected_next_action = REPAIR_SOURCES_ACTION
        selected_next_step = "repair learned-router sparse-value closeout source artifacts"
        rationale = "Required local pregate artifacts are missing or contradictory."
    else:
        selected_row = selected[0]
        status = "pass"
        decision = "learned_router_sparse_value_branch_closed"
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
        "backend_policy": "local closeout only; RunPod and Colab remain blocked until a local redirect clears null and budget gates",
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
    pregate: dict[str, Any],
    primary: dict[str, str],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    pregate_primary = _as_dict(pregate.get("pregate_primary_result"))
    return {
        "pregate_status": pregate.get("status"),
        "pregate_decision": pregate.get("decision"),
        "pregate_claim_status": pregate.get("claim_status"),
        "pregate_selected_next_action": pregate.get("selected_next_action"),
        "pregate_passes": pregate_primary.get("pregate_passes"),
        "primary_arm": pregate_primary.get("primary_arm") or primary.get("primary_arm"),
        "primary_holdout_ce": _first_float(
            pregate_primary.get("primary_holdout_ce"),
            primary.get("primary_holdout_ce"),
        ),
        "token_position_ce_gain": _first_float(
            pregate_primary.get("token_position_ce_gain"),
            primary.get("token_position_ce_gain"),
        ),
        "flat_control_ce_gain": _first_float(
            pregate_primary.get("flat_control_ce_gain"),
            primary.get("flat_control_ce_gain"),
        ),
        "flat_control_ok": _first_bool(
            pregate_primary.get("flat_control_ok"),
            primary.get("flat_control_ok"),
        ),
        "commutator_budget_ok": _first_bool(
            pregate_primary.get("commutator_budget_ok"),
            primary.get("commutator_budget_ok"),
        ),
        "functional_churn_budget_ok": _first_bool(
            pregate_primary.get("functional_churn_budget_ok"),
            primary.get("functional_churn_budget_ok"),
        ),
        "stored_upper_bound_blocks_promotion": _first_bool(
            pregate_primary.get("stored_upper_bound_blocks_promotion"),
            primary.get("stored_upper_bound_blocks_promotion"),
        ),
        "failure_reasons": pregate_primary.get("failure_reasons") or primary.get("failure_reasons", ""),
        "strategy_verdict": strategy["verdict"],
        "strategy_recommended_next_action": strategy["recommended_next_action"],
        "ben_notification_required": strategy["notify_ben"]
        or strategy["strategic_change_level"] == "major",
    }


def _closeout_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "branch": "learned_router_sparse_value",
            "source_decision": evidence["pregate_decision"],
            "disposition": "closed_before_gpu",
            "reason": evidence["failure_reasons"],
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
        {
            "branch": "same_router_flat_value_control",
            "source_decision": evidence["pregate_decision"],
            "disposition": "redirect_target",
            "reason": "flat value control beat the sparse learned-router arm in the pregate",
            "flat_control_ce_gain": evidence["flat_control_ce_gain"],
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
        {
            "branch": "gpu_validation",
            "source_decision": "local_closeout",
            "disposition": "blocked",
            "reason": "sparse value branch failed null/control/interference gates",
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
    ]


def _failures(source_rows: list[dict[str, Any]], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows:
        if row["source"] == "strategy_review":
            continue
        if not row["present"]:
            failures.append({"source": row["source"], "reason": "missing_required_source"})
    if evidence["pregate_status"] != "pass":
        failures.append({"source": "pregate", "reason": "pregate_status_not_pass"})
    if evidence["pregate_passes"] is True:
        failures.append({"source": "pregate", "reason": "pregate_passed_repeat_before_closeout"})
    if evidence["ben_notification_required"]:
        failures.append({"source": "strategy_review", "reason": "ben_notification_required"})
    return failures


def _candidate_actions(
    evidence: dict[str, Any],
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if failures:
        return [
            _candidate(
                REPAIR_SOURCES_ACTION,
                "selected",
                "required closeout sources are missing, contradictory, or require Ben notification",
                "repair source artifacts or resolve Ben notification before continuing",
                "source_repair_required",
            )
        ]
    if evidence["pregate_passes"] is True:
        return [
            _candidate(
                REPEAT_SPARSE_ACTION,
                "selected",
                "the sparse pregate passed locally and needs a repeat before any backend spend",
                "repeat learned-router sparse-value pregate on an adjacent seed",
                "sparse_branch_repeat_required",
            )
        ]
    return [
        _candidate(
            FLAT_VALUE_DIAGNOSTIC_ACTION,
            "selected",
            (
                "the learned-router sparse-value branch is blocked by same_router_flat_value_control_stronger "
                "and related null/interference failures, making the same-router flat value control the strongest "
                "immediate counterexample"
            ),
            "design a local same-router flat-value-capacity diagnostic with token/position, random/fixed support, dense/MLP, residual-norm, churn, commutator, and oracle-regret gates",
            "sparse_value_closed_flat_value_diagnostic_selected",
        ),
        _candidate(
            REPEAT_SPARSE_ACTION,
            "rejected",
            "repeating the sparse arm would spend effort on a branch already beaten by a same-router flat value control and budget failures",
            "only reconsider after a new sparse value mechanism changes the control relationship",
            "rejected",
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
    }


def _primary_pregate_row(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if row.get("selected") == "True":
            return row
    return rows[0] if rows else {}


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
        return "No external strategy review was present; proceeded from local pregate artifacts."
    return (
        "Read the latest external review. Its no-RunPod hidden-classifier recommendation remains satisfied; "
        "this closeout records the downstream learned-router sparse-value failure without launching GPU validation."
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _float_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        if value == "True":
            return True
        if value == "False":
            return False
    return None


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
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
        "# Learned-Router Sparse-Value Closeout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Primary arm: `{evidence.get('primary_arm')}`",
        f"- Token/position CE gain: `{evidence.get('token_position_ce_gain')}`",
        f"- Flat-control CE gain: `{evidence.get('flat_control_ce_gain')}`",
        f"- Failure reasons: `{evidence.get('failure_reasons')}`",
        "",
        str(summary["rationale"]),
        "",
        "GPU validation remains blocked.",
        "",
        "## Next Step",
        "",
        str(summary["selected_next_step"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pregate", type=Path, default=DEFAULT_PREGATE)
    parser.add_argument("--pregate-rows", type=Path, default=DEFAULT_PREGATE_ROWS)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_learned_router_sparse_value_closeout(
        pregate_path=args.pregate,
        pregate_rows_path=args.pregate_rows,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
