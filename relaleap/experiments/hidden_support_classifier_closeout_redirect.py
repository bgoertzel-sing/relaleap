"""Close the direct hidden support-classifier branch and redirect locally."""

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
DEFAULT_HIDDEN_CLOSEOUT_ROWS = Path("results/reports/hidden_support_classifier_sequence_ood_budget_audit/closeout_rows.csv")
DEFAULT_ORACLE_PREGATE = Path("results/reports/transformer_acsr_oracle_overlap_training_pregate/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/hidden_support_classifier_closeout_redirect")

CLOSE_AND_REDIRECT_ACTION = "select_oracle_overlap_aware_transformer_acsr_support_objective_redesign"
REPAIR_ACTION = "repair_hidden_support_classifier_closeout_redirect_sources"
KEEP_CLOSED_ACTION = "keep_hidden_support_classifier_closed_until_new_objective_exists"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "closeout_decision.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_hidden_support_classifier_closeout_redirect(
    *,
    hidden_audit_path: Path = DEFAULT_HIDDEN_AUDIT,
    hidden_closeout_rows_path: Path = DEFAULT_HIDDEN_CLOSEOUT_ROWS,
    oracle_pregate_path: Path = DEFAULT_ORACLE_PREGATE,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Emit a fail-closed redirect after the hidden-classifier learned-router loss."""

    start = time.time()
    hidden_audit = _read_json(hidden_audit_path)
    hidden_closeout_rows = _read_csv(hidden_closeout_rows_path)
    oracle_pregate = _read_json(oracle_pregate_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_json("hidden_support_classifier_sequence_ood_budget_audit", hidden_audit_path, hidden_audit),
        _source_csv(
            "hidden_support_classifier_closeout_rows",
            hidden_closeout_rows_path,
            hidden_closeout_rows,
        ),
        _source_json("transformer_acsr_oracle_overlap_training_pregate", oracle_pregate_path, oracle_pregate),
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
    evidence = _evidence(hidden_audit, hidden_closeout_rows, oracle_pregate)
    hidden_closed = _hidden_branch_closed(evidence)
    oracle_redesign_available = _oracle_redesign_available(evidence)
    candidate_actions = _candidate_actions(
        failures=failures,
        hidden_closed=hidden_closed,
        oracle_redesign_available=oracle_redesign_available,
        evidence=evidence,
    )
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]
    if failures or len(selected) != 1:
        status = "fail"
        decision = "hidden_support_classifier_closeout_redirect_failed_closed"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair hidden support-classifier closeout/redirect source artifacts"
        claim_status = "source_artifacts_incomplete"
        rationale = "The redirect cannot choose a branch until required source artifacts are present."
    else:
        status = "pass"
        decision = "hidden_support_classifier_closed_redirect_selected"
        selected_next_action = selected[0]["candidate_action"]
        selected_next_step = selected[0]["next_step"]
        claim_status = selected[0]["claim_status"]
        rationale = selected[0]["reason"]

    closeout_decision = [
        {
            "branch": "direct_hidden_support_classifier",
            "hidden_branch_closed": hidden_closed,
            "mean_ce_gain_vs_learned_router": evidence["mean_hidden_classifier_ce_gain_vs_learned_router"],
            "mean_oracle_regret_recovery_vs_learned_router": evidence[
                "mean_oracle_regret_recovery_vs_learned_router"
            ],
            "selected_next_action": selected_next_action,
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "advance_to_gpu_validation": False,
            "failure_reason": "" if hidden_closed else "hidden audit has not explicitly closed the branch",
        }
    ]
    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_action": selected_next_action,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "hidden_branch_closed": hidden_closed,
        "oracle_overlap_redesign_available": oracle_redesign_available,
        "source_rows": source_rows,
        "evidence": evidence,
        "closeout_decision": closeout_decision,
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
    hidden_audit: dict[str, Any],
    hidden_closeout_rows: list[dict[str, str]],
    oracle_pregate: dict[str, Any],
) -> dict[str, Any]:
    closeout = hidden_closeout_rows[0] if hidden_closeout_rows else {}
    primary = oracle_pregate.get("primary_result") if isinstance(oracle_pregate.get("primary_result"), dict) else {}
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
        "oracle_pregate_status": oracle_pregate.get("status"),
        "oracle_pregate_decision": oracle_pregate.get("decision"),
        "oracle_pregate_passes": oracle_pregate.get("pregate_passes"),
        "oracle_pregate_selected_next_step": oracle_pregate.get("selected_next_step"),
        "oracle_pregate_uses_target_token": primary.get("uses_target_token_as_predictor_feature"),
        "oracle_pregate_uses_oracle_loss": primary.get("uses_oracle_loss_as_predictor_feature"),
        "oracle_pregate_prefix_safe_feature_names": primary.get("prefix_safe_feature_names", ""),
        "oracle_pregate_regret_recovery_fraction_vs_learned": _float_or_none(
            primary.get("regret_recovery_fraction_vs_learned")
        ),
        "oracle_pregate_oracle_mean_jaccard_overlap": _float_or_none(
            primary.get("oracle_mean_jaccard_overlap")
        ),
    }


def _hidden_branch_closed(evidence: dict[str, Any]) -> bool:
    return bool(
        evidence["hidden_audit_status"] == "pass"
        and evidence["close_hidden_classifier_branch"] is True
        and evidence["sequence_heldout_gate_passes"] is False
        and _float_or_none(evidence["mean_hidden_classifier_ce_gain_vs_learned_router"]) is not None
        and _float_or_none(evidence["mean_hidden_classifier_ce_gain_vs_learned_router"]) < 0.0
    )


def _oracle_redesign_available(evidence: dict[str, Any]) -> bool:
    return bool(
        evidence["oracle_pregate_status"] == "pass"
        and evidence["oracle_pregate_uses_target_token"] is False
        and evidence["oracle_pregate_uses_oracle_loss"] is False
        and "learned_support_multihot" in str(evidence["oracle_pregate_prefix_safe_feature_names"])
    )


def _candidate_actions(
    *,
    failures: list[dict[str, Any]],
    hidden_closed: bool,
    oracle_redesign_available: bool,
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    if failures:
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "required source artifacts are missing",
                "repair hidden support-classifier closeout/redirect source artifacts",
                "source_artifacts_incomplete",
            )
        ]
    if hidden_closed and oracle_redesign_available:
        return [
            _candidate(
                CLOSE_AND_REDIRECT_ACTION,
                "selected",
                (
                    "the direct hidden classifier loses the learned-router same-student gate, while the "
                    "oracle-overlap pregate provides a prefix-safe local redesign target and remains GPU-blocked"
                ),
                (
                    "replace the oracle-overlap proxy pregate with hidden-feature same-student intervention "
                    "training and learned-router/null/budget gates before any GPU validation"
                ),
                "direct_hidden_classifier_closed_oracle_overlap_redesign_selected",
            ),
            _candidate(
                KEEP_CLOSED_ACTION,
                "deferred",
                "a prefix-safe oracle-overlap redesign source artifact is already present",
                "keep branch closed only if the oracle-overlap redesign source becomes unavailable",
                "deferred",
            ),
        ]
    return [
        _candidate(
            KEEP_CLOSED_ACTION,
            "selected",
            (
                "the hidden classifier is closed, but no complete prefix-safe oracle-overlap redesign source "
                "is available"
                if hidden_closed
                else "the hidden classifier closeout evidence is not sufficient to redirect"
            ),
            "keep the direct hidden support-classifier branch closed until a new objective source exists",
            "hidden_classifier_closed_no_redesign_source"
            if hidden_closed
            else "hidden_classifier_not_closed_by_sources",
        ),
        _candidate(
            CLOSE_AND_REDIRECT_ACTION,
            "deferred",
            f"oracle pregate decision: {evidence.get('oracle_pregate_decision')}",
            "revisit after source repair or new pregate",
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
        "status": row.get("status", "missing"),
        "decision": row.get("next_step", ""),
        "claim_status": row.get("deferred_exact_row_reason", ""),
        "row_count": len(rows),
    }


def _source_failures(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"source": row["source"], "path": row["path"], "reason": "required source artifact missing"}
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]


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
        return "No external strategy review was present; proceeded from local hidden-audit and oracle-pregate artifacts."
    return (
        "Accepted the review's no-RunPod/fail-closed hidden-classifier direction. "
        "The redirect closes the direct classifier branch and keeps the oracle-overlap redesign local before GPU."
    )


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
    _write_csv(out_dir / "closeout_decision.csv", summary["closeout_decision"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    notes = [
        "# Hidden Support-Classifier Closeout Redirect",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Selected next step: `{summary['selected_next_step']}`",
        f"- Hidden branch closed: `{summary['hidden_branch_closed']}`",
        f"- Oracle-overlap redesign available: `{summary['oracle_overlap_redesign_available']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Advance to GPU validation: `{summary['advance_to_gpu_validation']}`",
        "",
        summary["rationale"],
        "",
        "GPU validation remains blocked until the redesigned objective beats learned-router, null, OOD, churn, and commutator gates.",
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
    parser.add_argument("--hidden-closeout-rows", type=Path, default=DEFAULT_HIDDEN_CLOSEOUT_ROWS)
    parser.add_argument("--oracle-pregate", type=Path, default=DEFAULT_ORACLE_PREGATE)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_hidden_support_classifier_closeout_redirect(
        hidden_audit_path=args.hidden_audit,
        hidden_closeout_rows_path=args.hidden_closeout_rows,
        oracle_pregate_path=args.oracle_pregate,
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
