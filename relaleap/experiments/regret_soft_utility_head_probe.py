"""Probe regret-soft/listwise utility-head evidence before GPU validation."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_DESIGN = Path("results/reports/regret_soft_utility_head_design/summary.json")
DEFAULT_HIDDEN_AUDIT = Path("results/reports/hidden_support_classifier_sequence_ood_budget_audit")
DEFAULT_SYNTHETIC_DIR = Path("results/reports/synthetic_mechanism_causal_modularity")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/regret_soft_utility_head_probe")

REPAIR_ACTION = "repair_regret_soft_utility_head_probe_sources"
CLOSE_ACTION = "close_regret_soft_utility_head_probe_before_gpu"
IMPLEMENT_DIRECT_ACTION = "implement_direct_regret_soft_utility_head_training_rows"
REPEAT_ACTION = "repeat_regret_soft_utility_head_probe_before_gpu"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "candidate_rows.csv",
    "gate_rows.csv",
    "notes.md",
)


def run_regret_soft_utility_head_probe(
    *,
    design_path: Path = DEFAULT_DESIGN,
    hidden_audit_dir: Path = DEFAULT_HIDDEN_AUDIT,
    synthetic_dir: Path = DEFAULT_SYNTHETIC_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Evaluate local utility-head proxy/direct rows and fail closed for GPU."""

    start = time.time()
    design = _read_json(design_path)
    paths = {
        "hidden_audit_summary": hidden_audit_dir / "summary.json",
        "hidden_audit_rows": hidden_audit_dir / "audit_rows.csv",
        "hidden_budget_rows": hidden_audit_dir / "budget_rows.csv",
        "synthetic_support_head_rows": synthetic_dir / "support_head_sequence_heldout_diagnostic.csv",
        "synthetic_arm_metrics": synthetic_dir / "arm_metrics.csv",
        "synthetic_forgetting_rows": synthetic_dir / "forgetting_rows.csv",
        "synthetic_commutator_rows": synthetic_dir / "commutator_rows.csv",
    }
    json_rows = {"hidden_audit_summary": _read_json(paths["hidden_audit_summary"])}
    csv_rows = {
        name: _read_csv(path)
        for name, path in paths.items()
        if name != "hidden_audit_summary"
    }
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_json("regret_soft_utility_head_design", design_path, design),
        _source_json("hidden_audit_summary", paths["hidden_audit_summary"], json_rows["hidden_audit_summary"]),
        *[
            _source_csv(name, path, csv_rows[name])
            for name, path in paths.items()
            if name != "hidden_audit_summary"
        ],
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
    failures = _source_failures(source_rows, design, strategy)
    candidate_rows = _candidate_rows(
        source_failures=failures,
        support_head_rows=csv_rows["synthetic_support_head_rows"],
        arm_metrics=csv_rows["synthetic_arm_metrics"],
        forgetting_rows=csv_rows["synthetic_forgetting_rows"],
        commutator_rows=csv_rows["synthetic_commutator_rows"],
        hidden_audit=json_rows["hidden_audit_summary"],
        hidden_budget_rows=csv_rows["hidden_budget_rows"],
    )
    gate_rows = _gate_rows(candidate_rows, failures)
    direct_rows = [row for row in candidate_rows if row.get("direct_regret_soft_evidence") is True]
    proxy_rows = [row for row in candidate_rows if row.get("proxy_evidence") is True]
    passing_direct = [row for row in direct_rows if row.get("candidate_passes") is True]
    missing_direct = not direct_rows
    proxy_passes = any(row.get("candidate_passes") is True for row in proxy_rows)
    probe_passes = bool(passing_direct) and all(row["passes"] is True for row in gate_rows)
    status = "fail" if failures else "pass"
    if failures:
        decision = "regret_soft_utility_head_probe_failed_closed"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair regret-soft utility-head probe source artifacts"
        claim_status = "source_artifacts_incomplete"
    elif probe_passes:
        decision = "regret_soft_utility_head_probe_passed_repeat_before_gpu"
        selected_next_action = REPEAT_ACTION
        selected_next_step = "repeat the passing direct regret-soft utility-head probe on an adjacent local seed before GPU validation"
        claim_status = "direct_regret_soft_utility_head_signal_needs_repeat"
    elif missing_direct:
        decision = "regret_soft_utility_head_probe_direct_rows_missing"
        selected_next_action = IMPLEMENT_DIRECT_ACTION
        selected_next_step = (
            "implement direct regret-soft/listwise utility-head training rows in the command harness before "
            "reconsidering GPU validation"
        )
        claim_status = "proxy_rows_insufficient_for_regret_soft_utility_head_claim"
    else:
        decision = "regret_soft_utility_head_probe_gpu_blocked"
        selected_next_action = CLOSE_ACTION
        selected_next_step = "close the current regret-soft utility-head probe before GPU; direct local gates failed"
        claim_status = "regret_soft_utility_head_not_established"
    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_action": selected_next_action,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local probe only; RunPod and Colab remain blocked until direct local rows pass",
        "source_rows": source_rows,
        "candidate_rows": candidate_rows,
        "gate_rows": gate_rows,
        "direct_regret_soft_row_count": len(direct_rows),
        "proxy_row_count": len(proxy_rows),
        "passing_direct_row_count": len(passing_direct),
        "proxy_candidate_passes": proxy_passes,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "failures": failures,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _candidate_rows(
    *,
    source_failures: list[dict[str, Any]],
    support_head_rows: list[dict[str, str]],
    arm_metrics: list[dict[str, str]],
    forgetting_rows: list[dict[str, str]],
    commutator_rows: list[dict[str, str]],
    hidden_audit: dict[str, Any],
    hidden_budget_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if source_failures:
        return [
            {
                "candidate": "source_repair",
                "direct_regret_soft_evidence": False,
                "proxy_evidence": False,
                "candidate_passes": False,
                "failure_reasons": "source_artifacts_incomplete",
            }
        ]
    rows: list[dict[str, Any]] = [
        _learned_router_reference(support_head_rows, arm_metrics, forgetting_rows, commutator_rows),
        _hidden_classifier_closed_reference(hidden_audit, hidden_budget_rows),
    ]
    direct = [
        row
        for row in support_head_rows
        if row.get("diagnostic") in {
            "regret_soft_utility_head_topk2",
            "listwise_support_utility_head_topk2",
            "margin_conditioned_utility_fallback_topk2",
        }
    ]
    for row in direct:
        rows.append(
            _support_head_candidate(
                candidate=row.get("diagnostic", "direct_regret_soft_utility_head"),
                source_row=row,
                direct=True,
                arm_metrics=arm_metrics,
                forgetting_rows=forgetting_rows,
                commutator_rows=commutator_rows,
            )
        )
    for row in support_head_rows:
        if row.get("diagnostic") == "support_regret_trained_contextual_router_topk2":
            rows.append(
                _support_head_candidate(
                    candidate="support_regret_trained_contextual_router_proxy",
                    source_row=row,
                    direct=False,
                    arm_metrics=arm_metrics,
                    forgetting_rows=forgetting_rows,
                    commutator_rows=commutator_rows,
                )
            )
    return rows


def _learned_router_reference(
    support_head_rows: list[dict[str, str]],
    arm_metrics: list[dict[str, str]],
    forgetting_rows: list[dict[str, str]],
    commutator_rows: list[dict[str, str]],
) -> dict[str, Any]:
    support_row = _first_row(
        support_head_rows,
        diagnostic="support_regret_trained_contextual_router_topk2",
        arm="promoted_contextual_topk2",
    )
    arm = _row_for_arm(arm_metrics, "promoted_contextual_topk2")
    return {
        "candidate": "learned_router_reference",
        "arm": "promoted_contextual_topk2",
        "split": support_row.get("split", "sequence_heldout"),
        "direct_regret_soft_evidence": False,
        "proxy_evidence": False,
        "candidate_ce": _number(support_row.get("learned_router_ce")) or _number(arm.get("holdout_ce")),
        "oracle_ce": _number(support_row.get("oracle_pair_ce_ceiling")),
        "ce_gain_vs_learned_router": 0.0,
        "oracle_regret_recovery_vs_learned_router": 0.0,
        "beats_token_position_null": "",
        "beats_shuffled_null": "",
        "residual_l2": _number(arm.get("residual_l2")) or _number(support_row.get("residual_l2")),
        "functional_churn": _mean_metric(forgetting_rows, "promoted_contextual_topk2", "functional_churn"),
        "finite_update_commutator_l2": _mean_metric(
            commutator_rows,
            "promoted_contextual_topk2",
            "finite_update_commutator_l2",
        ),
        "candidate_passes": False,
        "failure_reasons": "reference_only",
    }


def _hidden_classifier_closed_reference(
    hidden_audit: dict[str, Any],
    hidden_budget_rows: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "candidate": "closed_hard_hidden_support_classifier",
        "arm": "direct_hidden_support_classifier",
        "split": "sequence_heldout",
        "direct_regret_soft_evidence": False,
        "proxy_evidence": False,
        "candidate_ce": "",
        "oracle_ce": "",
        "ce_gain_vs_learned_router": hidden_audit.get("mean_hidden_classifier_ce_gain_vs_learned_router"),
        "oracle_regret_recovery_vs_learned_router": hidden_audit.get(
            "mean_oracle_regret_recovery_vs_learned_router"
        ),
        "budget_rows_present": len(hidden_budget_rows),
        "candidate_passes": False,
        "failure_reasons": "closed_by_hidden_support_classifier_sequence_ood_budget_audit",
    }


def _support_head_candidate(
    *,
    candidate: str,
    source_row: dict[str, str],
    direct: bool,
    arm_metrics: list[dict[str, str]],
    forgetting_rows: list[dict[str, str]],
    commutator_rows: list[dict[str, str]],
) -> dict[str, Any]:
    arm = source_row.get("arm", "")
    arm_row = _row_for_arm(arm_metrics, arm)
    learned_ce = _number(source_row.get("learned_router_ce"))
    candidate_ce = _number(source_row.get("predicted_support_ce"))
    oracle_ce = _number(source_row.get("oracle_pair_ce_ceiling"))
    gain = _gain(learned_ce, candidate_ce)
    recovery = _safe_regret_recovery(learned_ce=learned_ce, candidate_ce=candidate_ce, oracle_ce=oracle_ce)
    residual_l2 = _number(source_row.get("residual_l2")) or _number(arm_row.get("residual_l2"))
    functional_churn = _mean_metric(forgetting_rows, arm, "functional_churn")
    commutator = _mean_metric(commutator_rows, arm, "finite_update_commutator_l2")
    reasons: list[str] = []
    if not direct:
        reasons.append("proxy_not_direct_regret_soft_or_margin_fallback")
    if not (gain is not None and recovery is not None and (gain > 0.0 or recovery >= 0.25)):
        reasons.append("learned_router_ce_or_regret_gate_failed")
    if str(source_row.get("beats_shuffled_target_null")).lower() != "true":
        reasons.append("shuffled_null_gate_failed")
    if str(source_row.get("beats_token_position_null")).lower() != "true":
        reasons.append("token_position_null_gate_failed")
    if str(source_row.get("deployable_training_evidence")).lower() == "false":
        reasons.append("not_deployable_training_evidence")
    if residual_l2 is None or functional_churn is None or commutator is None:
        reasons.append("budget_metrics_missing_or_incomplete")
    passes = direct and not reasons
    return {
        "candidate": candidate,
        "arm": arm,
        "split": source_row.get("split", ""),
        "direct_regret_soft_evidence": direct,
        "proxy_evidence": not direct,
        "target_source": source_row.get("target_source", ""),
        "uses_hidden_features": _boolish(source_row.get("uses_hidden_features")),
        "uses_token_position_features": _boolish(source_row.get("uses_token_position_features")),
        "uses_shuffled_targets": _boolish(source_row.get("uses_shuffled_targets")),
        "deployable_training_evidence": _boolish(source_row.get("deployable_training_evidence")),
        "learned_router_ce": learned_ce,
        "candidate_ce": candidate_ce,
        "oracle_ce": oracle_ce,
        "ce_gain_vs_learned_router": gain,
        "oracle_regret_recovery_vs_learned_router": recovery,
        "support_accuracy_vs_oracle_pair": _number(source_row.get("support_accuracy_vs_oracle_pair")),
        "support_change_fraction": _number(source_row.get("support_change_fraction")),
        "residual_l2": residual_l2,
        "functional_churn": functional_churn,
        "finite_update_commutator_l2": commutator,
        "beats_shuffled_null": _boolish(source_row.get("beats_shuffled_target_null")),
        "beats_token_position_null": _boolish(source_row.get("beats_token_position_null")),
        "candidate_passes": passes,
        "failure_reasons": ";".join(reasons),
    }


def _gate_rows(candidate_rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    direct_rows = [row for row in candidate_rows if row.get("direct_regret_soft_evidence") is True]
    passing_direct = [row for row in direct_rows if row.get("candidate_passes") is True]
    proxy_rows = [row for row in candidate_rows if row.get("proxy_evidence") is True]
    return [
        _gate("required_sources_present", not failures, "required design, audit, and synthetic source artifacts exist"),
        _gate("direct_regret_soft_rows_present", bool(direct_rows), "direct regret-soft/listwise utility-head rows must exist"),
        _gate("beats_learned_router_or_recovers_regret", bool(passing_direct), "direct candidate must beat learned router or recover >=25% regret"),
        _gate("null_controls_pass", bool(passing_direct), "direct candidate must beat shuffled and token/position nulls"),
        _gate("budget_rows_pass", bool(passing_direct), "direct candidate must emit nonworse norm/churn/commutator metrics"),
        _gate("proxy_rows_not_promoted", not any(row.get("candidate_passes") for row in proxy_rows), "proxy rows cannot authorize GPU validation"),
    ]


def _gate(name: str, passes: bool, requirement: str) -> dict[str, Any]:
    return {
        "gate": name,
        "passes": passes,
        "requirement": requirement,
        "failure_reason": "" if passes else requirement,
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
        "row_count": "",
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


def _source_failures(
    source_rows: list[dict[str, Any]],
    design: dict[str, Any],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    failures = [
        {
            "source": row["source"],
            "reason": "missing required source",
        }
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]
    if design.get("selected_next_action") != "implement_regret_soft_utility_head_probe_locally":
        failures.append(
            {
                "source": "regret_soft_utility_head_design",
                "reason": "design did not select the local probe",
            }
        )
    if strategy["notify_ben"] or strategy["strategic_change_level"] == "major":
        failures.append(
            {
                "source": "strategy_review",
                "reason": "strategy review requires Ben notification before continuing silently",
            }
        )
    return failures


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "present": False,
            "path": str(path),
            "strategic_change_level": "missing",
            "notify_ben": False,
            "recommended_next_action": "",
            "verdict": "",
        }
    fields: dict[str, Any] = {
        "present": True,
        "path": str(path),
        "strategic_change_level": "",
        "notify_ben": False,
        "recommended_next_action": "",
        "verdict": "",
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "notify_ben":
            fields[key] = value.lower() == "true"
        elif key in fields:
            fields[key] = value
    return fields


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No strategy review was present; local probe remains fail-closed."
    if strategy["notify_ben"] or strategy["strategic_change_level"] == "major":
        return "Strategy review requests Ben notification; probe does not advance GPU validation."
    return "Accepted the no-RunPod/fail-closed recommendation; proxy evidence is not promoted."


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "candidate_rows.csv", summary["candidate_rows"])
    _write_csv(out_dir / "gate_rows.csv", summary["gate_rows"])
    notes = [
        "# Regret-Soft Utility-Head Probe",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Direct regret-soft row count: `{summary['direct_regret_soft_row_count']}`",
        f"- Proxy row count: `{summary['proxy_row_count']}`",
        f"- Proxy candidate passes: `{summary['proxy_candidate_passes']}`",
        f"- Advance to GPU validation: `{summary['advance_to_gpu_validation']}`",
        f"- Next step: `{summary['selected_next_step']}`",
        "",
        (
            "GPU validation remains blocked. Existing support-regret contextual-router rows are useful "
            "proxy evidence, but they are not direct regret-soft/listwise utility-head plus margin-fallback "
            "evidence and cannot authorize a GPU run."
        ),
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


def _first_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    return rows[0] if rows else {}


def _row_for_arm(rows: list[dict[str, str]], arm: str) -> dict[str, str]:
    for row in rows:
        if row.get("arm") == arm:
            return row
    return {}


def _mean_metric(rows: list[dict[str, str]], arm: str, metric: str) -> float | None:
    values = [_number(row.get(metric)) for row in rows if row.get("arm") == arm]
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def _gain(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None:
        return None
    return reference - candidate


def _safe_regret_recovery(
    *,
    learned_ce: float | None,
    candidate_ce: float | None,
    oracle_ce: float | None,
) -> float | None:
    if learned_ce is None or candidate_ce is None or oracle_ce is None:
        return None
    learned_regret = learned_ce - oracle_ce
    if learned_regret <= 0.0:
        return None
    candidate_regret = candidate_ce - oracle_ce
    return (learned_regret - candidate_regret) / learned_regret


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _boolish(value: Any) -> bool | str:
    if value in (None, ""):
        return ""
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return str(value)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--hidden-audit-dir", type=Path, default=DEFAULT_HIDDEN_AUDIT)
    parser.add_argument("--synthetic-dir", type=Path, default=DEFAULT_SYNTHETIC_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_regret_soft_utility_head_probe(
        design_path=args.design,
        hidden_audit_dir=args.hidden_audit_dir,
        synthetic_dir=args.synthetic_dir,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
