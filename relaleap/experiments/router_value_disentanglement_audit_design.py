"""Design the next no-training router/value disentanglement audit."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_router_value_disentanglement_audit_design"
)
DEFAULT_CHECKPOINT = Path(
    "results/reports/token_larger_non_ce_support_quality_checkpoint/summary.json"
)
DEFAULT_ROUTER_POLICY_PROBE = Path(
    "results/audits/token_larger_promoted_topk2_router_policy_mitigation_probe/summary.json"
)
DEFAULT_UPDATE_DECOMPOSITION_AUDIT = Path(
    "results/audits/token_larger_promoted_topk2_update_decomposition_audit/summary.json"
)
DEFAULT_VALUE_MITIGATION_GATE = Path(
    "results/audits/token_larger_promoted_topk2_value_mitigation_gate/summary.json"
)
DEFAULT_COMMUTATOR_VALUE_PENALTY_PROBE = Path(
    "results/audits/token_larger_promoted_topk2_commutator_value_penalty_probe/summary.json"
)
DEFAULT_DISTILLATION_AGREEMENT = Path(
    "results/audits/token_larger_causal_contextual_router_distillation_agreement/summary.json"
)
DEFAULT_DISTILLATION_INTERVENTIONS = Path(
    "results/audits/token_larger_causal_contextual_router_distillation_agreement/intervention_metrics.csv"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")


DESIGN_RECORDED = "router_value_disentanglement_audit_design_recorded"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
SELECTED_NEXT_ACTION = "implement_no_training_router_value_disentanglement_audit"


def run_router_value_disentanglement_audit_design(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    router_policy_probe_path: Path = DEFAULT_ROUTER_POLICY_PROBE,
    update_decomposition_audit_path: Path = DEFAULT_UPDATE_DECOMPOSITION_AUDIT,
    value_mitigation_gate_path: Path = DEFAULT_VALUE_MITIGATION_GATE,
    commutator_value_penalty_probe_path: Path = DEFAULT_COMMUTATOR_VALUE_PENALTY_PROBE,
    distillation_agreement_path: Path = DEFAULT_DISTILLATION_AGREEMENT,
    distillation_interventions_path: Path = DEFAULT_DISTILLATION_INTERVENTIONS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
) -> dict[str, Any]:
    """Write a fail-closed design artifact for a local no-training audit."""

    start = time.time()
    packets = {
        "non_ce_checkpoint": _read_json_object(checkpoint_path),
        "router_policy_probe": _read_json_object(router_policy_probe_path),
        "update_decomposition_audit": _read_json_object(update_decomposition_audit_path),
        "value_mitigation_gate": _read_json_object(value_mitigation_gate_path),
        "commutator_value_penalty_probe": _read_json_object(
            commutator_value_penalty_probe_path
        ),
        "distillation_agreement": _read_json_object(distillation_agreement_path),
    }
    paths = {
        "non_ce_checkpoint": checkpoint_path,
        "router_policy_probe": router_policy_probe_path,
        "update_decomposition_audit": update_decomposition_audit_path,
        "value_mitigation_gate": value_mitigation_gate_path,
        "commutator_value_penalty_probe": commutator_value_penalty_probe_path,
        "distillation_agreement": distillation_agreement_path,
    }
    intervention_rows = _read_csv_rows(distillation_interventions_path)
    strategy_review = _strategy_review(strategy_review_path)
    source_rows = [_source_row(name, paths[name], packets[name]) for name in paths]
    source_rows.append(
        {
            "source": "distillation_interventions",
            "path": str(distillation_interventions_path),
            "present": distillation_interventions_path.is_file(),
            "status": "present" if intervention_rows else None,
            "decision": f"rows={len(intervention_rows)}",
            "claim_status": "same_student_support_swap_rows",
        }
    )
    source_rows.append(
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy_review["present"],
            "status": "present" if strategy_review["present"] else "missing_optional",
            "decision": strategy_review["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy_review['strategic_change_level']}; "
                f"notify_ben={strategy_review['notify_ben']}"
            ),
        }
    )
    evidence = _evidence_snapshot(packets, intervention_rows)
    design_rows = _design_rows(evidence)
    failures = _failures(source_rows, evidence)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        selected_next_action = None
        next_command = None
        next_step = "repair missing router/value disentanglement design sources"
        rationale = (
            "The audit design cannot be treated as evidence because required "
            "command-generated source artifacts or support-swap rows are missing."
        )
    else:
        status = "pass"
        decision = DESIGN_RECORDED
        selected_next_action = SELECTED_NEXT_ACTION
        next_command = (
            "./.venv-conda/bin/python -m "
            "relaleap.experiments.router_value_disentanglement_audit"
        )
        next_step = (
            "implement the no-training router/value disentanglement audit over "
            "existing support-swap, router-policy, and value-path artifacts"
        )
        rationale = (
            "The available artifacts already contain same-student support swaps "
            "for learned values, router-policy interventions with fixed learned "
            "values, and router-only versus value-only transfer paths. The design "
            "therefore targets a no-training synthesis audit before any new "
            "mitigation, distillation, or GPU repeat. Hub-pair and order-averaging "
            "promotion remain deferred because the localization evidence is diffuse."
        )

    summary = {
        "status": status,
        "decision": decision,
        "selected_next_action": selected_next_action,
        "next_command": next_command,
        "next_step": next_step,
        "claim_statuses": {
            "contextual_topk2_router": "operational_default_support_routing_baseline",
            "causal_contextual_router": "ce_baseline_only_not_promoted",
            "teacher_support_distillation": "closed_not_promoted",
            "router_value_disentanglement": "designed_not_yet_executed",
            "hub_pair_mitigation": "deferred_rejected_diffuse_localization",
            "order_averaging": "diagnostic_only_not_promoted",
            "topk2_causal_cooperation": "not_supported",
        },
        "source_rows": source_rows,
        "design_rows": design_rows,
        "evidence": evidence,
        "strategy_review": strategy_review,
        "failures": failures,
        "rationale": rationale,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "design_rows_csv": str(out_dir / "design_rows.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        out_dir / "source_rows.csv",
        ["source", "path", "present", "status", "decision", "claim_status"],
        source_rows,
    )
    _write_csv(
        out_dir / "design_rows.csv",
        [
            "contrast",
            "artifact_basis",
            "availability",
            "primary_metric",
            "current_signal",
            "interpretation",
            "audit_requirement",
        ],
        design_rows,
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _evidence_snapshot(
    packets: dict[str, dict[str, Any]], intervention_rows: list[dict[str, str]]
) -> dict[str, Any]:
    checkpoint = packets["non_ce_checkpoint"]
    router_probe = packets["router_policy_probe"]
    decomposition = packets["update_decomposition_audit"]
    value_gate = packets["value_mitigation_gate"]
    penalty = packets["commutator_value_penalty_probe"]
    distillation = packets["distillation_agreement"]
    interventions = _intervention_delta_summary(intervention_rows)
    return {
        "checkpoint_decision": checkpoint.get("decision"),
        "checkpoint_selected_next_action": checkpoint.get("selected_next_action"),
        "router_policy_decision": router_probe.get("decision"),
        "best_router_policy_reduction_fraction": _best_numeric(
            router_probe.get("router_policy_rows", []),
            "commutator_anchor_logit_mse_reduction_fraction",
        ),
        "update_decomposition_decision": decomposition.get("decision"),
        "value_only_fraction_of_full": _float_or_none(
            (decomposition.get("metrics", {}) or {}).get("value_only_fraction_of_full")
        ),
        "router_only_fraction_of_full": _float_or_none(
            (decomposition.get("metrics", {}) or {}).get("router_only_fraction_of_full")
        ),
        "value_mitigation_decision": value_gate.get("decision"),
        "best_value_mitigation_reduction_fraction": _float_or_none(
            (value_gate.get("metrics", {}) or {}).get(
                "best_value_mitigation_reduction_fraction"
            )
        ),
        "commutator_value_penalty_decision": penalty.get("decision"),
        "best_penalty_reduction_fraction": _float_or_none(
            (penalty.get("metrics", {}) or {}).get("best_penalty_reduction_fraction")
        ),
        "distillation_agreement_decision": distillation.get("decision"),
        "support_swap_intervention_count": len(intervention_rows),
        "support_swap_interventions": sorted(interventions),
        "teacher_support_mean_delta": interventions.get("teacher_support_forced_into_student"),
        "oracle_support_mean_delta": interventions.get("oracle_best_support_for_student"),
        "linear_support_mean_delta": interventions.get("linear_support_forced_into_student"),
        "random_support_mean_delta": interventions.get("uniform_random_support"),
    }


def _design_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "contrast": "same_values_alternate_supports",
            "artifact_basis": "causal_contextual_router_distillation_agreement/intervention_metrics.csv",
            "availability": _availability(evidence["support_swap_intervention_count"] > 0),
            "primary_metric": "delta_vs_student_router_support",
            "current_signal": (
                f"teacher={evidence['teacher_support_mean_delta']}; "
                f"oracle={evidence['oracle_support_mean_delta']}; "
                f"linear={evidence['linear_support_mean_delta']}; "
                f"random={evidence['random_support_mean_delta']}"
            ),
            "interpretation": "learned values can be evaluated under alternate supports without retraining",
            "audit_requirement": "aggregate all-token and disagreement-token deltas by intervention",
        },
        {
            "contrast": "same_values_alternate_router_policies",
            "artifact_basis": "promoted_topk2_router_policy_mitigation_probe/router_policy_rows",
            "availability": _availability(
                evidence["best_router_policy_reduction_fraction"] is not None
            ),
            "primary_metric": "commutator_anchor_logit_mse_reduction_fraction",
            "current_signal": evidence["best_router_policy_reduction_fraction"],
            "interpretation": "pinned/sticky policy changes did not clear the router-policy mitigation gate",
            "audit_requirement": "report policy deltas separately from value-transfer paths",
        },
        {
            "contrast": "same_supports_alternate_value_paths",
            "artifact_basis": "promoted_topk2_update_decomposition_audit",
            "availability": _availability(
                evidence["value_only_fraction_of_full"] is not None
                and evidence["router_only_fraction_of_full"] is not None
            ),
            "primary_metric": "value_only_fraction_of_full_vs_router_only_fraction_of_full",
            "current_signal": (
                f"value_only={evidence['value_only_fraction_of_full']}; "
                f"router_only={evidence['router_only_fraction_of_full']}"
            ),
            "interpretation": "value-transfer effects dominate router-only effects in the existing decomposition",
            "audit_requirement": "treat value path as a separate factor, not as support-selection quality",
        },
        {
            "contrast": "same_router_value_regularizer_controls",
            "artifact_basis": "value_mitigation_gate and commutator_value_penalty_probe",
            "availability": _availability(
                evidence["best_value_mitigation_reduction_fraction"] is not None
                and evidence["best_penalty_reduction_fraction"] is not None
            ),
            "primary_metric": "best_commutator_reduction_fraction",
            "current_signal": (
                f"simple={evidence['best_value_mitigation_reduction_fraction']}; "
                f"penalty={evidence['best_penalty_reduction_fraction']}"
            ),
            "interpretation": "simple value constraints improved too little to promote a mitigation",
            "audit_requirement": "include as negative controls, not as candidate promotions",
        },
    ]


def _failures(
    source_rows: list[dict[str, Any]], evidence: dict[str, Any]
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows[:-1]:
        if not row["present"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "source_artifact",
                    "expected": "file exists",
                    "actual": "missing",
                    "path": row["path"],
                }
            )
        elif row["source"] != "distillation_interventions" and row["status"] not in {
            "pass",
            "ok",
        }:
            failures.append(
                {
                    "source": row["source"],
                    "field": "status",
                    "expected": "pass or ok",
                    "actual": row["status"],
                }
            )
    expected = {
        "checkpoint_decision": "non_ce_support_quality_checkpoint_selected",
        "checkpoint_selected_next_action": "router_value_disentanglement_audit_design",
        "update_decomposition_decision": "value_update_dominated_order_sensitivity",
        "value_mitigation_decision": "value_mitigation_not_established",
        "commutator_value_penalty_decision": "commutator_value_penalty_not_established",
    }
    for field, expected_value in expected.items():
        if evidence.get(field) != expected_value:
            failures.append(
                {
                    "source": "evidence",
                    "field": field,
                    "expected": expected_value,
                    "actual": evidence.get(field),
                }
            )
    if evidence["support_swap_intervention_count"] <= 0:
        failures.append(
            {
                "source": "distillation_interventions",
                "field": "support_swap_intervention_count",
                "expected": "> 0",
                "actual": evidence["support_swap_intervention_count"],
            }
        )
    if evidence["value_only_fraction_of_full"] is None:
        failures.append(
            {
                "source": "update_decomposition_audit",
                "field": "value_only_fraction_of_full",
                "expected": "numeric",
                "actual": None,
            }
        )
    return failures


def _source_row(name: str, path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": name,
        "path": str(path),
        "present": path.is_file(),
        "status": packet.get("status"),
        "decision": packet.get("decision"),
        "claim_status": packet.get("claim_status"),
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "present": False,
            "path": str(path),
            "strategic_change_level": None,
            "notify_ben": None,
            "ben_notification_required": False,
            "recommended_next_action": None,
            "incorporation": "missing optional review; proceeded from command artifacts",
        }
    headers: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()
    notify_ben = headers.get("notify_ben", "").lower() == "true"
    strategic_change_level = headers.get("strategic_change_level")
    return {
        "present": True,
        "path": str(path),
        "strategic_change_level": strategic_change_level,
        "notify_ben": notify_ben,
        "ben_notification_required": notify_ben or strategic_change_level == "major",
        "recommended_next_action": headers.get("recommended_next_action"),
        "incorporation": (
            "accepted: no hub-pair/order-averaging mitigation, no GPU repeat, "
            "and no distillation promotion before a local router/value "
            "disentanglement audit"
        ),
    }


def _intervention_delta_summary(rows: list[dict[str, str]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        intervention = row.get("intervention")
        delta = _float_or_none(row.get("delta_vs_student_router_support"))
        if intervention and delta is not None:
            grouped.setdefault(intervention, []).append(delta)
    return {
        key: sum(values) / len(values)
        for key, values in grouped.items()
        if values
    }


def _best_numeric(rows: Any, field: str) -> float | None:
    if not isinstance(rows, list):
        return None
    values = [
        value
        for value in (_float_or_none(row.get(field)) for row in rows if isinstance(row, dict))
        if value is not None
    ]
    return max(values) if values else None


def _availability(condition: bool) -> str:
    return "available" if condition else "missing"


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Router/Value Disentanglement Audit Design",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        "",
        summary["rationale"],
        "",
        f"Next step: {summary['next_step']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_router_value_disentanglement_audit_design(out_dir=args.out)
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
