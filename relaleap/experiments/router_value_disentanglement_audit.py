"""Run a no-training router/value disentanglement synthesis audit."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path("results/audits/token_larger_router_value_disentanglement_audit")
DEFAULT_DESIGN = Path(
    "results/reports/token_larger_router_value_disentanglement_audit_design/summary.json"
)
DEFAULT_DISTILLATION_INTERVENTIONS = Path(
    "results/audits/token_larger_causal_contextual_router_distillation_agreement/"
    "intervention_metrics.csv"
)
DEFAULT_ROUTER_POLICY_ROWS = Path(
    "results/audits/token_larger_promoted_topk2_router_policy_mitigation_probe/"
    "router_policy_rows.csv"
)
DEFAULT_UPDATE_DECOMPOSITION_ROWS = Path(
    "results/audits/token_larger_promoted_topk2_update_decomposition_audit/"
    "decomposition_rows.csv"
)
DEFAULT_VALUE_MITIGATION_ROWS = Path(
    "results/audits/token_larger_promoted_topk2_value_mitigation_gate/"
    "value_mitigation_rows.csv"
)
DEFAULT_COMMUTATOR_VALUE_PENALTY_ROWS = Path(
    "results/audits/token_larger_promoted_topk2_commutator_value_penalty_probe/"
    "commutator_value_penalty_rows.csv"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")


AUDIT_RECORDED_NO_PROMOTION = "router_value_disentanglement_audit_recorded_no_promotion"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def run_router_value_disentanglement_audit(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    design_path: Path = DEFAULT_DESIGN,
    distillation_interventions_path: Path = DEFAULT_DISTILLATION_INTERVENTIONS,
    router_policy_rows_path: Path = DEFAULT_ROUTER_POLICY_ROWS,
    update_decomposition_rows_path: Path = DEFAULT_UPDATE_DECOMPOSITION_ROWS,
    value_mitigation_rows_path: Path = DEFAULT_VALUE_MITIGATION_ROWS,
    commutator_value_penalty_rows_path: Path = DEFAULT_COMMUTATOR_VALUE_PENALTY_ROWS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
) -> dict[str, Any]:
    """Synthesize existing command artifacts without retraining."""

    start = time.time()
    design = _read_json_object(design_path)
    support_rows = _read_csv_rows(distillation_interventions_path)
    router_policy_rows = _read_csv_rows(router_policy_rows_path)
    decomposition_rows = _read_csv_rows(update_decomposition_rows_path)
    value_rows = _read_csv_rows(value_mitigation_rows_path)
    penalty_rows = _read_csv_rows(commutator_value_penalty_rows_path)
    strategy_review = _strategy_review(strategy_review_path)

    source_rows = [
        _source_row("design", design_path, design.get("status"), design.get("decision")),
        _source_row(
            "distillation_interventions",
            distillation_interventions_path,
            "present" if support_rows else None,
            f"rows={len(support_rows)}",
        ),
        _source_row(
            "router_policy_rows",
            router_policy_rows_path,
            "present" if router_policy_rows else None,
            f"rows={len(router_policy_rows)}",
        ),
        _source_row(
            "update_decomposition_rows",
            update_decomposition_rows_path,
            "present" if decomposition_rows else None,
            f"rows={len(decomposition_rows)}",
        ),
        _source_row(
            "value_mitigation_rows",
            value_mitigation_rows_path,
            "present" if value_rows else None,
            f"rows={len(value_rows)}",
        ),
        _source_row(
            "commutator_value_penalty_rows",
            commutator_value_penalty_rows_path,
            "present" if penalty_rows else None,
            f"rows={len(penalty_rows)}",
        ),
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
        },
    ]

    support_summary_rows = _support_summary_rows(support_rows)
    router_summary_rows = _router_policy_summary_rows(router_policy_rows)
    decomposition_summary_rows = _decomposition_summary_rows(decomposition_rows)
    control_summary_rows = _control_summary_rows(value_rows, penalty_rows)
    factor_rows = (
        support_summary_rows
        + router_summary_rows
        + decomposition_summary_rows
        + control_summary_rows
    )
    evidence = _evidence_snapshot(
        design,
        support_summary_rows,
        router_summary_rows,
        decomposition_summary_rows,
        control_summary_rows,
    )
    failures = _failures(source_rows, evidence)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        selected_next_action = None
        next_command = None
        next_step = "repair missing router/value disentanglement source artifacts"
        rationale = (
            "The no-training audit cannot be interpreted because required "
            "command-generated source rows or the design artifact are missing or "
            "inconsistent."
        )
    else:
        status = "pass"
        decision = AUDIT_RECORDED_NO_PROMOTION
        selected_next_action = "local_same_student_retention_functional_churn_gate"
        next_command = None
        next_step = (
            "run a local same-student retention/functional-churn gate using the "
            "existing promoted top-k-2 baseline, causal-feature-safe top-k-2, "
            "linear top-k-2, rank-matched top-k-1, and random/fixed top-k-2 "
            "controls"
        )
        rationale = (
            "Existing no-training artifacts separate support swaps, router-policy "
            "pinning/stickiness, value-only versus router-only transfer, and value "
            "regularizer controls. They point to value-path and support-selection "
            "entanglement, but do not establish a promotion-worthy mitigation or "
            "causal-router support quality. The next discriminative evidence should "
            "therefore be a same-student retention/functional-churn gate, not a "
            "hub-pair, order-averaging, distillation, or GPU branch."
        )

    summary = {
        "status": status,
        "decision": decision,
        "selected_next_action": selected_next_action,
        "next_command": next_command,
        "next_step": next_step,
        "claim_statuses": {
            "contextual_topk2_router": "operational_support_routing_baseline",
            "router_value_disentanglement": (
                "recorded_value_path_and_support_selection_entangled"
            ),
            "causal_contextual_router": "ce_baseline_only_not_promoted",
            "teacher_support_distillation": "closed_not_promoted",
            "hub_pair_mitigation": "rejected_diffuse_localization",
            "order_averaging": "diagnostic_only_not_promoted",
            "topk2_causal_cooperation": "not_supported",
        },
        "source_rows": source_rows,
        "factor_rows": factor_rows,
        "support_summary_rows": support_summary_rows,
        "router_policy_summary_rows": router_summary_rows,
        "decomposition_summary_rows": decomposition_summary_rows,
        "control_summary_rows": control_summary_rows,
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
            "factor_rows_csv": str(out_dir / "factor_rows.csv"),
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
        out_dir / "factor_rows.csv",
        [
            "factor_family",
            "variant",
            "token_subset",
            "primary_metric",
            "primary_value",
            "secondary_metric",
            "secondary_value",
            "interpretation",
        ],
        factor_rows,
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _support_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        intervention = row.get("intervention", "")
        token_subset = row.get("token_subset", "")
        delta = _float_or_none(row.get("delta_vs_student_router_support"))
        if intervention and token_subset and delta is not None:
            grouped.setdefault((intervention, token_subset), []).append(delta)
    out: list[dict[str, Any]] = []
    for (intervention, token_subset), values in sorted(grouped.items()):
        mean_delta = sum(values) / len(values)
        out.append(
            {
                "factor_family": "same_values_alternate_supports",
                "variant": intervention,
                "token_subset": token_subset,
                "primary_metric": "mean_delta_vs_student_router_support",
                "primary_value": mean_delta,
                "secondary_metric": "fold_count",
                "secondary_value": len(values),
                "interpretation": _support_interpretation(intervention, mean_delta),
            }
        )
    return out


def _router_policy_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        variant = row.get("variant", "")
        reduction = _float_or_none(row.get("commutator_anchor_logit_mse_reduction_fraction"))
        ce_delta = _float_or_none(row.get("anchor_ce_delta_vs_dynamic"))
        if variant and reduction is not None:
            out.append(
                {
                    "factor_family": "same_values_alternate_router_policies",
                    "variant": variant,
                    "token_subset": "anchor",
                    "primary_metric": "commutator_anchor_logit_mse_reduction_fraction",
                    "primary_value": reduction,
                    "secondary_metric": "anchor_ce_delta_vs_dynamic",
                    "secondary_value": ce_delta,
                    "interpretation": (
                        "router policy mitigation gate not established"
                        if row.get("passes_router_policy_gate") in {"False", "false", ""}
                        else "router policy row requires inspection"
                    ),
                }
            )
    return out


def _decomposition_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        variant = row.get("variant", "")
        fraction = _float_or_none(row.get("commutator_anchor_fraction_of_full"))
        retention = _float_or_none(row.get("transfer_retention_fraction"))
        if variant and fraction is not None:
            out.append(
                {
                    "factor_family": "same_supports_alternate_value_paths",
                    "variant": variant,
                    "token_subset": "anchor",
                    "primary_metric": "commutator_anchor_fraction_of_full",
                    "primary_value": fraction,
                    "secondary_metric": "transfer_retention_fraction",
                    "secondary_value": retention,
                    "interpretation": (
                        "value-path contribution dominates"
                        if row.get("transfer_update_group") == "value_only"
                        else "router-only contribution is smaller than full update"
                    ),
                }
            )
    return out


def _control_summary_rows(
    value_rows: list[dict[str, str]], penalty_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for family, rows in [
        ("simple_value_regularizer_control", value_rows),
        ("commutator_value_penalty_control", penalty_rows),
    ]:
        for row in rows:
            variant = row.get("variant", "")
            reduction = _float_or_none(
                row.get("commutator_anchor_logit_mse_reduction_fraction")
            )
            retention = _float_or_none(row.get("transfer_retention_fraction"))
            if variant and reduction is not None:
                out.append(
                    {
                        "factor_family": family,
                        "variant": variant,
                        "token_subset": "anchor",
                        "primary_metric": (
                            "commutator_anchor_logit_mse_reduction_fraction"
                        ),
                        "primary_value": reduction,
                        "secondary_metric": "transfer_retention_fraction",
                        "secondary_value": retention,
                        "interpretation": "negative control; not promotion-worthy",
                    }
                )
    return out


def _evidence_snapshot(
    design: dict[str, Any],
    support_rows: list[dict[str, Any]],
    router_rows: list[dict[str, Any]],
    decomposition_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    support_lookup = {
        (row["variant"], row["token_subset"]): _float_or_none(row["primary_value"])
        for row in support_rows
    }
    decomp_lookup = {
        row["variant"]: _float_or_none(row["primary_value"])
        for row in decomposition_rows
    }
    return {
        "design_decision": design.get("decision"),
        "design_selected_next_action": design.get("selected_next_action"),
        "support_summary_count": len(support_rows),
        "router_policy_summary_count": len(router_rows),
        "decomposition_summary_count": len(decomposition_rows),
        "control_summary_count": len(control_rows),
        "teacher_all_token_delta": support_lookup.get(
            ("teacher_support_forced_into_student", "all_tokens")
        ),
        "oracle_all_token_delta": support_lookup.get(
            ("oracle_best_support_for_student", "all_tokens")
        ),
        "linear_all_token_delta": support_lookup.get(
            ("linear_support_forced_into_student", "all_tokens")
        ),
        "random_all_token_delta": support_lookup.get(("uniform_random_support", "all_tokens")),
        "best_router_policy_reduction_fraction": _max_primary(router_rows),
        "value_only_fraction_of_full": decomp_lookup.get("value_only_transfer_topk2"),
        "router_only_fraction_of_full": decomp_lookup.get("router_only_transfer_topk2"),
        "best_regularizer_reduction_fraction": _max_primary(control_rows),
    }


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
                    "expected": "file exists with rows",
                    "actual": "missing_or_empty",
                    "path": row["path"],
                }
            )
    expected = {
        "design_decision": "router_value_disentanglement_audit_design_recorded",
        "design_selected_next_action": "implement_no_training_router_value_disentanglement_audit",
    }
    for field, expected_value in expected.items():
        if evidence.get(field) != expected_value:
            failures.append(
                {
                    "source": "design",
                    "field": field,
                    "expected": expected_value,
                    "actual": evidence.get(field),
                }
            )
    for field in [
        "teacher_all_token_delta",
        "oracle_all_token_delta",
        "linear_all_token_delta",
        "best_router_policy_reduction_fraction",
        "value_only_fraction_of_full",
        "router_only_fraction_of_full",
        "best_regularizer_reduction_fraction",
    ]:
        if evidence.get(field) is None:
            failures.append(
                {
                    "source": "evidence",
                    "field": field,
                    "expected": "numeric",
                    "actual": None,
                }
            )
    return failures


def _support_interpretation(intervention: str, mean_delta: float) -> str:
    if intervention == "student_router_support":
        return "student support baseline"
    if intervention == "oracle_best_support_for_student":
        return "oracle support improves learned values" if mean_delta < 0 else "oracle support does not improve"
    if mean_delta > 0:
        return "alternate support worsens learned values"
    return "alternate support improves learned values"


def _source_row(source: str, path: Path, status: Any, decision: Any) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": status,
        "decision": decision,
        "claim_status": None,
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
            "accepted: close pairwise/hub mitigation, keep distillation closed, "
            "and move to local same-student retention/churn after this audit"
        ),
    }


def _max_primary(rows: list[dict[str, Any]]) -> float | None:
    values = [
        value
        for value in (_float_or_none(row.get("primary_value")) for row in rows)
        if value is not None
    ]
    return max(values) if values else None


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
        "# Router/Value Disentanglement Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        "",
        summary["rationale"],
        "",
        "Key evidence:",
        (
            f"- Teacher/all-token support delta: "
            f"`{summary['evidence']['teacher_all_token_delta']}`"
        ),
        (
            f"- Oracle/all-token support delta: "
            f"`{summary['evidence']['oracle_all_token_delta']}`"
        ),
        (
            f"- Value-only versus router-only fractions: "
            f"`{summary['evidence']['value_only_fraction_of_full']}` / "
            f"`{summary['evidence']['router_only_fraction_of_full']}`"
        ),
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
    summary = run_router_value_disentanglement_audit(out_dir=args.out)
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
