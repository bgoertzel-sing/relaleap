"""Design a local finite-update commutator mitigation for flat value capacity."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_CLOSEOUT = Path("results/reports/same_router_flat_value_capacity_closeout/summary.json")
DEFAULT_BUDGET_ROWS = Path("results/reports/same_router_flat_value_capacity_diagnostic/budget_rows.csv")
DEFAULT_GATE_ROWS = Path("results/reports/same_router_flat_value_capacity_diagnostic/gate_rows.csv")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/same_router_flat_value_commutator_mitigation_design")

SELECTED_ACTION = "implement_flat_value_commutator_mitigation_probe_locally"
REPAIR_ACTION = "repair_flat_value_commutator_mitigation_design_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "mitigation_design.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_same_router_flat_value_commutator_mitigation_design(
    *,
    closeout_path: Path = DEFAULT_CLOSEOUT,
    budget_rows_path: Path = DEFAULT_BUDGET_ROWS,
    gate_rows_path: Path = DEFAULT_GATE_ROWS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Record the next local mitigation probe contract without running GPU work."""

    start = time.time()
    closeout = _read_json(closeout_path)
    budget_rows = _read_csv(budget_rows_path)
    gate_rows = _read_csv(gate_rows_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_json("same_router_flat_value_capacity_closeout", closeout_path, closeout),
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
            "row_count": "",
        },
    ]
    evidence = _evidence(closeout, budget_rows, gate_rows, strategy)
    design_rows = _design_rows(evidence)
    gate_criteria = _gate_criteria(evidence, source_rows)
    failures = [row for row in gate_criteria if row["passed"] is False]
    status = "pass" if not failures else "fail"
    summary = {
        "status": status,
        "decision": (
            "same_router_flat_value_commutator_mitigation_design_recorded"
            if status == "pass"
            else "same_router_flat_value_commutator_mitigation_design_failed_closed"
        ),
        "claim_status": (
            "design_only_flat_value_commutator_mitigation_not_yet_evidence"
            if status == "pass"
            else "source_artifacts_incomplete_or_mitigation_not_selected"
        ),
        "selected_next_action": SELECTED_ACTION if status == "pass" else REPAIR_ACTION,
        "selected_next_step": (
            "implement a local flat-value commutator mitigation probe with order-averaging, value-norm clipping, "
            "and commutator-penalized variants, preserving CE/null/dense controls and nonworse norm/churn/commutator gates"
            if status == "pass"
            else "repair flat-value commutator mitigation design source artifacts"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local design/probe only; RunPod and Colab remain blocked until mitigation evidence passes",
        "source_rows": source_rows,
        "evidence": evidence,
        "mitigation_design": design_rows,
        "gate_criteria": gate_criteria,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "failures": failures,
        "rationale": (
            "The flat same-router value arm has a local CE/control signal but fails the finite-update "
            "commutator budget. A mitigation probe is warranted only if it preserves the existing "
            "CE/null/dense-control advantages while reducing commutator without increasing residual norm or churn."
        ),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _evidence(
    closeout: dict[str, Any],
    budget_rows: list[dict[str, str]],
    gate_rows: list[dict[str, str]],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    budget_by_name = {row.get("budget", ""): row for row in budget_rows}
    gate_by_name = {row.get("gate", ""): _bool_or_none(row.get("passes")) for row in gate_rows}
    commutator = budget_by_name.get("finite_update_commutator", {})
    residual_norm = budget_by_name.get("residual_norm", {})
    functional_churn = budget_by_name.get("functional_churn", {})
    return {
        "closeout_status": closeout.get("status"),
        "closeout_decision": closeout.get("decision"),
        "closeout_selected_next_action": closeout.get("selected_next_action"),
        "closeout_claim_status": closeout.get("claim_status"),
        "flat_beats_promoted_sparse": gate_by_name.get("flat_beats_promoted_sparse"),
        "flat_beats_support_policy_nulls": gate_by_name.get("flat_beats_support_policy_nulls"),
        "flat_beats_dense_mlp_controls": gate_by_name.get("flat_beats_dense_mlp_controls"),
        "oracle_regret_strata_available": gate_by_name.get("oracle_regret_strata_available"),
        "residual_norm_budget_passes": _bool_or_none(residual_norm.get("nonworse_gate_passes")),
        "functional_churn_budget_passes": _bool_or_none(functional_churn.get("nonworse_gate_passes")),
        "commutator_budget_passes": _bool_or_none(commutator.get("nonworse_gate_passes")),
        "commutator_candidate_value": _float_or_none(commutator.get("candidate_value")),
        "commutator_reference_value": _float_or_none(commutator.get("reference_budget_value")),
        "commutator_ratio_to_reference": _ratio(
            _float_or_none(commutator.get("candidate_value")),
            _float_or_none(commutator.get("reference_budget_value")),
        ),
        "strategy_verdict": strategy["verdict"],
        "ben_notification_required": strategy["notify_ben"]
        or strategy["strategic_change_level"] == "major",
    }


def _design_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "variant": "flat_value_order_averaged_updates",
            "mechanism": "average both finite update orders for the same router/value pair before scoring",
            "primary_hypothesis": "commutator excess is update-order asymmetry rather than useful value capacity",
            "required_controls": "unmitigated_flat_value;promoted_contextual_topk2;token_position;random_support;fixed_support;dense_rank_norm;low_churn_mlp",
            "primary_gate": "commutator <= reference_budget * 1.10 and CE advantage vs sparse/null/dense controls is preserved",
            "risk": "order averaging may erase the same value-capacity CE advantage",
        },
        {
            "variant": "flat_value_norm_clipped_updates",
            "mechanism": "clip flat value residual/update norm to the promoted sparse reference budget before intervention",
            "primary_hypothesis": "commutator excess is driven by residual magnitude, not flat value representation",
            "required_controls": "same controls plus residual_norm and functional_churn rows",
            "primary_gate": "residual norm and churn remain nonworse while commutator falls below budget",
            "risk": "norm clipping can convert the branch into an ordinary low-capacity sparse control",
        },
        {
            "variant": "flat_value_commutator_penalty_probe",
            "mechanism": "add a local finite-update order penalty during the flat value probe",
            "primary_hypothesis": "the flat value head can retain CE gains while learning more order-stable updates",
            "required_controls": "same CE/null/dense controls plus pre/post penalty budget rows",
            "primary_gate": "commutator budget passes without worsening CE beyond 0.005 or increasing churn/norm budgets",
            "risk": "penalty may optimize a diagnostic metric without improving causal support quality",
        },
    ]


def _gate_criteria(evidence: dict[str, Any], source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required_sources_present = all(
        row["present"] for row in source_rows if row["source"] != "strategy_review"
    )
    closeout_selected = (
        evidence["closeout_selected_next_action"]
        == "design_flat_value_finite_update_commutator_mitigation"
    )
    ce_controls_pass = all(
        evidence.get(key) is True
        for key in (
            "flat_beats_promoted_sparse",
            "flat_beats_support_policy_nulls",
            "flat_beats_dense_mlp_controls",
            "oracle_regret_strata_available",
        )
    )
    isolated_commutator_failure = (
        evidence["residual_norm_budget_passes"] is True
        and evidence["functional_churn_budget_passes"] is True
        and evidence["commutator_budget_passes"] is False
    )
    no_ben_notify = not evidence["ben_notification_required"]
    return [
        _criterion("required_sources_present", required_sources_present, "source artifacts must exist"),
        _criterion("closeout_selected_mitigation", closeout_selected, "closeout must select commutator mitigation"),
        _criterion("ce_and_control_signal_present", ce_controls_pass, "flat arm must retain CE/control signal"),
        _criterion(
            "isolated_commutator_budget_failure",
            isolated_commutator_failure,
            "residual norm and churn pass while commutator fails",
        ),
        _criterion("no_ben_notification_required", no_ben_notify, "strategy review must not require Ben notification"),
    ]


def _criterion(name: str, passed: bool, requirement: str) -> dict[str, Any]:
    return {
        "criterion": name,
        "passed": passed,
        "requirement": requirement,
        "failure_reason": "" if passed else requirement,
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


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "mitigation_design.csv", summary["mitigation_design"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    notes = [
        "# Same-Router Flat-Value Commutator Mitigation Design",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Commutator ratio to reference: `{summary['evidence']['commutator_ratio_to_reference']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Advance to GPU validation: `{summary['advance_to_gpu_validation']}`",
        f"- Next step: `{summary['selected_next_step']}`",
        "",
        (
            "GPU validation remains blocked. This is a design artifact only: a later probe must show that "
            "order averaging, norm clipping, or a commutator penalty preserves the flat value CE/control signal "
            "while making residual norm, functional churn, and finite-update commutator budgets nonworse."
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


def _bool_or_none(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return {
        "path": str(path),
        "present": bool(text),
        "strategic_change_level": _header_value(text, "strategic_change_level") or "unknown",
        "notify_ben": (_header_value(text, "notify_ben") or "false").lower() == "true",
        "recommended_next_action": _header_value(text, "recommended_next_action") or "",
        "verdict": _header_value(text, "verdict") or "",
    }


def _header_value(text: str, key: str) -> str:
    prefix = f"{key}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No strategy review was present; continued with local fail-closed design from automation status."
    if strategy["notify_ben"] or strategy["strategic_change_level"] == "major":
        return "Strategy review requires Ben notification; design fails closed until direction is acknowledged."
    return "Accepted the no-RunPod/fail-closed local-gating recommendation; this design keeps GPU validation blocked."


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closeout", type=Path, default=DEFAULT_CLOSEOUT)
    parser.add_argument("--budget-rows", type=Path, default=DEFAULT_BUDGET_ROWS)
    parser.add_argument("--gate-rows", type=Path, default=DEFAULT_GATE_ROWS)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_same_router_flat_value_commutator_mitigation_design(
        closeout_path=args.closeout,
        budget_rows_path=args.budget_rows,
        gate_rows_path=args.gate_rows,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
