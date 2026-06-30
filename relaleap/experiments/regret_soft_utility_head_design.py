"""Design the local regret-soft utility-head follow-up for Transformer-ACSR."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_HIDDEN_GATE = Path("results/reports/transformer_acsr_hidden_feature_redesign_gate/summary.json")
DEFAULT_HIDDEN_AUDIT = Path("results/reports/hidden_support_classifier_sequence_ood_budget_audit/summary.json")
DEFAULT_HIDDEN_CLOSEOUT = Path("results/reports/hidden_support_classifier_closeout_redirect/summary.json")
DEFAULT_SEED_REPEAT = Path("results/reports/transformer_acsr_seed_repeat/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/regret_soft_utility_head_design")

SELECTED_ACTION = "implement_regret_soft_utility_head_probe_locally"
REPAIR_ACTION = "repair_regret_soft_utility_head_design_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "utility_head_design.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_regret_soft_utility_head_design(
    *,
    hidden_gate_path: Path = DEFAULT_HIDDEN_GATE,
    hidden_audit_path: Path = DEFAULT_HIDDEN_AUDIT,
    hidden_closeout_path: Path = DEFAULT_HIDDEN_CLOSEOUT,
    seed_repeat_path: Path = DEFAULT_SEED_REPEAT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Record a bounded local probe contract without running GPU validation."""

    start = time.time()
    hidden_gate = _read_json(hidden_gate_path)
    hidden_audit = _read_json(hidden_audit_path)
    hidden_closeout = _read_json(hidden_closeout_path)
    seed_repeat = _read_json(seed_repeat_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_json("transformer_acsr_hidden_feature_redesign_gate", hidden_gate_path, hidden_gate),
        _source_json("hidden_support_classifier_sequence_ood_budget_audit", hidden_audit_path, hidden_audit),
        _source_json("hidden_support_classifier_closeout_redirect", hidden_closeout_path, hidden_closeout),
        _source_json("transformer_acsr_seed_repeat", seed_repeat_path, seed_repeat),
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
    evidence = _evidence(hidden_gate, hidden_audit, hidden_closeout, seed_repeat, strategy)
    design_rows = _design_rows(evidence)
    gate_criteria = _gate_criteria(evidence, source_rows)
    failures = [row for row in gate_criteria if row["passed"] is False]
    status = "pass" if not failures else "fail"
    summary = {
        "status": status,
        "decision": (
            "regret_soft_utility_head_design_recorded"
            if status == "pass"
            else "regret_soft_utility_head_design_failed_closed"
        ),
        "claim_status": (
            "design_only_regret_soft_utility_head_not_yet_evidence"
            if status == "pass"
            else "source_artifacts_incomplete_or_branch_not_selected"
        ),
        "selected_next_action": SELECTED_ACTION if status == "pass" else REPAIR_ACTION,
        "selected_next_step": (
            "implement a local regret-soft/listwise support-utility head with margin-conditioned learned-router "
            "fallback and learned-router/null/OOD/churn/commutator gates"
            if status == "pass"
            else "repair regret-soft utility-head design source artifacts"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local design/probe only; RunPod and Colab remain blocked until the local probe passes",
        "source_rows": source_rows,
        "evidence": evidence,
        "utility_head_design": design_rows,
        "gate_criteria": gate_criteria,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "failures": failures,
        "rationale": (
            "The hard hidden support classifier is prefix-safe and beats weak nulls, but loses the learned-router "
            "same-student gate. The next local question is whether a soft/listwise utility objective can recover "
            "oracle-support regret only when its predicted margin justifies overriding the learned router."
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
    hidden_gate: dict[str, Any],
    hidden_audit: dict[str, Any],
    hidden_closeout: dict[str, Any],
    seed_repeat: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    mean_gain = _first_present(
        hidden_gate.get("aggregates", {}).get("mean_ce_gain_vs_learned_router"),
        hidden_audit.get("mean_hidden_classifier_ce_gain_vs_learned_router"),
        seed_repeat.get("mean_hidden_classifier_ce_gain_vs_learned_router"),
    )
    mean_recovery = _first_present(
        hidden_gate.get("aggregates", {}).get("mean_oracle_regret_recovery_vs_learned_router"),
        hidden_audit.get("mean_hidden_classifier_oracle_regret_recovery_vs_learned_router"),
        seed_repeat.get("mean_hidden_classifier_oracle_regret_recovery_vs_learned_router"),
    )
    return {
        "hidden_gate_status": hidden_gate.get("status"),
        "hidden_gate_decision": hidden_gate.get("decision"),
        "hidden_gate_claim_status": hidden_gate.get("claim_status"),
        "hidden_gate_selected_next_step": hidden_gate.get("selected_next_step"),
        "hidden_feature_gate_passes": hidden_gate.get("hidden_feature_gate_passes"),
        "learned_router_gate_passes": hidden_gate.get("learned_router_gate_passes"),
        "null_gate_passes": hidden_gate.get("null_gate_passes"),
        "future_perturbation_leakage_gate_passes": hidden_gate.get("future_perturbation_leakage_gate_passes"),
        "hidden_classifier_branch_closed": _bool_or_none(
            hidden_audit.get("close_hidden_classifier_branch")
            if "close_hidden_classifier_branch" in hidden_audit
            else hidden_closeout.get("hidden_branch_closed")
        ),
        "hidden_audit_decision": hidden_audit.get("decision"),
        "hidden_audit_closeout_status": hidden_audit.get("closeout_status"),
        "hidden_closeout_selected_next_action": hidden_closeout.get("selected_next_action"),
        "seed_repeat_advance_to_gpu_validation": seed_repeat.get("advance_to_gpu_validation"),
        "seed_repeat_hidden_gpu_gate_passes": seed_repeat.get("hidden_classifier_gpu_gate_passes"),
        "mean_ce_gain_vs_learned_router": _float_or_none(mean_gain),
        "mean_oracle_regret_recovery_vs_learned_router": _float_or_none(mean_recovery),
        "strategy_verdict": strategy["verdict"],
        "ben_notification_required": strategy["notify_ben"]
        or strategy["strategic_change_level"] == "major",
    }


def _design_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "component": "regret_soft_support_utility_head",
            "specification": (
                "train on exhaustive support-loss tables as a soft/listwise target; score candidate support "
                "sets by predicted utility or negative regret rather than hard oracle labels"
            ),
            "required_inputs": "prefix-safe hidden features;token;position;history summaries;no target token;no future hidden chunks",
            "required_controls": "learned_router;oracle_support;global_best;frequency;token_position_transformer;shuffled_labels;delayed_labels;frequency_preserving_permutation;linear_hidden_probe;mlp_hidden_probe",
            "primary_gate": "candidate beats learned router or recovers >=25% router-oracle regret on sequence and rule-combo heldouts",
            "failure_closes": "direct utility-head branch remains local-only and GPU-blocked",
        },
        {
            "component": "margin_conditioned_learned_router_fallback",
            "specification": (
                "deploy the utility head only when calibrated utility margin over the learned router exceeds a "
                "heldout threshold; otherwise keep the current learned router support"
            ),
            "required_inputs": "candidate utility margin;learned-router support utility estimate;calibration split",
            "required_controls": "always_learned_router;always_utility_head;oracle_margin_threshold;random_margin_threshold",
            "primary_gate": "fallback policy improves same-student CE/regret without worsening churn or commutator budgets",
            "failure_closes": "use learned router as default; do not spend GPU",
        },
        {
            "component": "budget_and_mechanism_audit",
            "specification": (
                "emit residual norm, support churn, functional churn KL, finite-update commutator, per-rule and "
                "per-position deltas, NDCG/MRR, and calibration rows for every arm"
            ),
            "required_inputs": "same-student intervention rows;oracle losses;support changes;finite-update order rows",
            "required_controls": "hidden hard classifier;learned router;sparse value reference;flat value control;dense/rank/norm control",
            "primary_gate": "nonworse residual norm, functional churn, and finite-update commutator versus learned-router reference",
            "failure_closes": "no ACSR promotion or GPU validation claim",
        },
    ]


def _gate_criteria(evidence: dict[str, Any], source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required_sources_present = all(
        row["present"] for row in source_rows if row["source"] != "strategy_review"
    )
    selected_by_hidden_gate = (
        evidence["hidden_gate_selected_next_step"]
        == "design_regret_soft_utility_head_with_margin_conditioned_learned_router_fallback"
    )
    direct_hidden_closed = (
        evidence["hidden_feature_gate_passes"] is False
        and evidence["learned_router_gate_passes"] is False
        and evidence["hidden_classifier_branch_closed"] is True
    )
    weak_signal_available = (
        evidence["null_gate_passes"] is True
        and evidence["future_perturbation_leakage_gate_passes"] is True
    )
    gpu_blocked = (
        evidence["seed_repeat_advance_to_gpu_validation"] is False
        and evidence["seed_repeat_hidden_gpu_gate_passes"] is False
    )
    no_ben_notify = not evidence["ben_notification_required"]
    return [
        _criterion("required_sources_present", required_sources_present, "required source artifacts must exist"),
        _criterion("hidden_gate_selected_utility_head_design", selected_by_hidden_gate, "hidden gate must select utility-head design"),
        _criterion("direct_hidden_classifier_closed", direct_hidden_closed, "hard hidden classifier must be closed versus learned router"),
        _criterion("prefix_safe_weak_signal_available", weak_signal_available, "weak-null and prefix-safety gates should justify redesign rather than abandonment"),
        _criterion("gpu_validation_still_blocked", gpu_blocked, "seed-repeat gates must block GPU before this design step"),
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


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "utility_head_design.csv", summary["utility_head_design"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    notes = [
        "# Regret-Soft Utility-Head Design",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Mean CE gain vs learned router: `{summary['evidence']['mean_ce_gain_vs_learned_router']}`",
        f"- Mean oracle-regret recovery vs learned router: `{summary['evidence']['mean_oracle_regret_recovery_vs_learned_router']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Advance to GPU validation: `{summary['advance_to_gpu_validation']}`",
        f"- Next step: `{summary['selected_next_step']}`",
        "",
        (
            "GPU validation remains blocked. This is a design artifact only: a later local probe must show that "
            "a regret-soft/listwise utility head plus margin-conditioned learned-router fallback beats the learned "
            "router on same-student sequence and rule-combo heldouts while preserving residual-norm, functional-churn, "
            "and commutator budgets."
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


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


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
    return "Accepted the no-RunPod/fail-closed recommendation; this design keeps GPU validation blocked."


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-gate", type=Path, default=DEFAULT_HIDDEN_GATE)
    parser.add_argument("--hidden-audit", type=Path, default=DEFAULT_HIDDEN_AUDIT)
    parser.add_argument("--hidden-closeout", type=Path, default=DEFAULT_HIDDEN_CLOSEOUT)
    parser.add_argument("--seed-repeat", type=Path, default=DEFAULT_SEED_REPEAT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_regret_soft_utility_head_design(
        hidden_gate_path=args.hidden_gate,
        hidden_audit_path=args.hidden_audit,
        hidden_closeout_path=args.hidden_closeout,
        seed_repeat_path=args.seed_repeat,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
