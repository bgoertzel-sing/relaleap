"""Close out the current ACSR branch after broader negative mechanism evidence."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_ACSR_GATE = Path("results/audits/acsr_broader_mechanism_gate_local/summary.json")
DEFAULT_DENSE_TEACHER = Path(
    "results/audits/token_larger_dense_teacher_residual_distillation_comparison/summary.json"
)
DEFAULT_DENSE_RANK_NORM = Path("results/reports/acsr_dense_rank_norm_synthesis/summary.json")
DEFAULT_MLP_CHURN = Path("results/reports/mlp_churn_decision/summary.json")
DEFAULT_NORM_BUDGETED = Path("results/reports/post_norm_budgeted_branch_selector/summary.json")
DEFAULT_COMMUTATOR = Path("results/reports/acsr_finite_update_commutator_assay/summary.json")
DEFAULT_MECHANISM_CL = Path("results/reports/mechanism_factorized_continual_learning_repeat/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/acsr_negative_evidence_closeout")

DEMOTE_ACSR_ACTION = "demote_acsr_to_diagnostic_status"
PREFIX_NORM_RESCUE_ACTION = "design_prefix_only_norm_normalized_acsr_rescue"
REPAIR_SOURCES_ACTION = "repair_missing_acsr_closeout_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "candidate_actions.csv",
    "evidence_matrix.csv",
    "notes.md",
)


def run_acsr_negative_evidence_closeout_report(
    *,
    acsr_gate_path: Path = DEFAULT_ACSR_GATE,
    dense_teacher_path: Path = DEFAULT_DENSE_TEACHER,
    dense_rank_norm_path: Path = DEFAULT_DENSE_RANK_NORM,
    mlp_churn_path: Path = DEFAULT_MLP_CHURN,
    norm_budgeted_path: Path = DEFAULT_NORM_BUDGETED,
    commutator_path: Path = DEFAULT_COMMUTATOR,
    mechanism_cl_path: Path = DEFAULT_MECHANISM_CL,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Select a bounded branch after local ACSR mechanism evidence fails closed."""

    start = time.time()
    acsr_gate = _read_json(acsr_gate_path)
    dense_teacher = _read_json(dense_teacher_path)
    dense_rank_norm = _read_json(dense_rank_norm_path)
    mlp_churn = _read_json(mlp_churn_path)
    norm_budgeted = _read_json(norm_budgeted_path)
    commutator = _read_json(commutator_path)
    mechanism_cl = _read_json(mechanism_cl_path)
    strategy = _strategy_review(strategy_review_path)

    source_rows = [
        _source_row("acsr_broader_mechanism_gate", acsr_gate_path, acsr_gate),
        _source_row("dense_teacher_distillation", dense_teacher_path, dense_teacher),
        _source_row("dense24_rank_norm_synthesis", dense_rank_norm_path, dense_rank_norm),
        _source_row("mlp_churn_decision", mlp_churn_path, mlp_churn),
        _source_row("norm_budgeted_branch_selector", norm_budgeted_path, norm_budgeted),
        _source_row("finite_update_commutator", commutator_path, commutator),
        _source_row("mechanism_factorized_cl_repeat", mechanism_cl_path, mechanism_cl),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "present" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}"
            ),
        },
    ]
    evidence = _evidence_snapshot(
        acsr_gate,
        dense_teacher,
        dense_rank_norm,
        mlp_churn,
        norm_budgeted,
        commutator,
        mechanism_cl,
        strategy,
    )
    evidence_matrix = _evidence_matrix(evidence)
    failures = _source_failures(source_rows)
    candidate_actions = _candidate_actions(evidence, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "acsr_negative_evidence_closeout_failed_closed"
        selected_next_action = REPAIR_SOURCES_ACTION
        next_step = "repair missing ACSR closeout source artifacts"
        claim_status = "acsr_closeout_source_evidence_incomplete"
        rationale = "The closeout cannot choose a scientific branch until all required local source artifacts are present."
    else:
        status = "pass"
        decision = "acsr_negative_evidence_closeout_branch_selected"
        selected_next_action = selected[0]["candidate_action"]
        next_step = selected[0]["next_step"]
        claim_status = selected[0]["claim_status"]
        rationale = selected[0]["reason"]

    summary = {
        "status": status,
        "decision": decision,
        "selected_next_action": selected_next_action,
        "next_step": next_step,
        "claim_status": claim_status,
        "requires_gpu_now": False,
        "backend_policy": "local closeout only; RunPod/Colab validation remains blocked until a local rescue clears dense/null/churn/norm gates",
        "source_rows": source_rows,
        "evidence": evidence,
        "evidence_matrix": evidence_matrix,
        "candidate_actions": candidate_actions,
        "strategy_review": strategy,
        "direction_shift": _direction_shift_record(strategy),
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


def _evidence_snapshot(
    acsr_gate: dict[str, Any],
    dense_teacher: dict[str, Any],
    dense_rank_norm: dict[str, Any],
    mlp_churn: dict[str, Any],
    norm_budgeted: dict[str, Any],
    commutator: dict[str, Any],
    mechanism_cl: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    acsr_gates = _as_dict(acsr_gate.get("gates"))
    commutator_metrics = _as_dict(commutator.get("metrics"))
    mechanism_primary = _as_dict(mechanism_cl.get("primary_result"))
    dense_gate = _as_dict(dense_teacher.get("gate_status"))
    dense_criteria = dense_gate.get("criteria") if isinstance(dense_gate.get("criteria"), list) else []
    dense_rank_norm_metrics = dense_rank_norm.get("comparison_metrics")
    dense_rank_norm_metrics = dense_rank_norm_metrics if isinstance(dense_rank_norm_metrics, list) else []
    dense_ce_guardrail_passed = not any(
        row.get("criterion") == "acsr_ce_not_worse_than_teacher_by_large_margin"
        and row.get("passed") is False
        for row in dense_criteria
        if isinstance(row, dict)
    )
    return {
        "acsr_gate_status": acsr_gate.get("status"),
        "acsr_gate_decision": acsr_gate.get("decision"),
        "acsr_claim_status": acsr_gate.get("claim_status"),
        "acsr_beats_nulls": bool(acsr_gates.get("acsr_beats_nulls_on_available_packets")),
        "acsr_beats_parameter_matched": bool(acsr_gates.get("acsr_beats_parameter_matched_causal_control")),
        "acsr_retention_churn_guardrail": bool(acsr_gates.get("acsr_no_worse_retention_churn_than_contextual")),
        "acsr_intervention_l2_guardrail": bool(
            acsr_gates.get("acsr_no_worse_intervention_residual_l2_than_parameter_matched")
        ),
        "dense_teacher_status": dense_teacher.get("status"),
        "dense_teacher_decision": dense_teacher.get("decision"),
        "dense_teacher_claim_status": dense_teacher.get("claim_status"),
        "dense_teacher_ce_loss": _float_or_none(dense_teacher.get("dense_teacher_ce_loss")),
        "acsr_student_ce_loss": _variant_metric(dense_teacher, "acsr_predicted_future_support", "ce_loss"),
        "dense_teacher_ce_guardrail_passed": dense_ce_guardrail_passed,
        "dense_rank_norm_status": dense_rank_norm.get("status"),
        "dense_rank_norm_decision": dense_rank_norm.get("decision"),
        "dense_rank_norm_claim_status": dense_rank_norm.get("claim_status"),
        "minimal_dense_rank_beating_sparse": _named_metric(
            dense_rank_norm_metrics,
            "minimal_dense_rank_beating_sparse",
        ),
        "rank24_delta_minus_sparse": _named_metric(
            dense_rank_norm_metrics,
            "rank24_delta_minus_sparse",
        ),
        "dense24_or_rank_norm_blocks_sparse": dense_rank_norm.get("status") == "fail"
        or dense_rank_norm.get("claim_status") == "dense_rank16_24_controls_explain_ce_gain_threshold",
        "mlp_churn_status": mlp_churn.get("status"),
        "mlp_churn_decision": mlp_churn.get("decision"),
        "mlp_churn_claim_status": mlp_churn.get("claim_status"),
        "mlp_selected_next_action": _selected_action(mlp_churn),
        "norm_budgeted_status": norm_budgeted.get("status"),
        "norm_budgeted_decision": norm_budgeted.get("decision"),
        "norm_budgeted_claim_status": norm_budgeted.get("claim_status"),
        "norm_budgeted_selected_next_action": norm_budgeted.get("selected_next_action"),
        "commutator_status": commutator.get("status"),
        "commutator_decision": commutator.get("decision"),
        "commutator_claim_status": commutator.get("claim_status"),
        "sparse_mean_logit_mse": _float_or_none(commutator_metrics.get("sparse_mean_logit_mse")),
        "dense_mean_logit_mse": _float_or_none(commutator_metrics.get("dense_mean_logit_mse")),
        "commutator_material": commutator.get("status") == "pass",
        "mechanism_cl_status": mechanism_cl.get("status"),
        "mechanism_cl_decision": mechanism_cl.get("decision"),
        "mechanism_cl_claim_status": mechanism_cl.get("claim_status"),
        "mechanism_cl_topk2_repeat_status": mechanism_cl.get("topk2_tradeoff_repeat_status"),
        "mechanism_cl_supporting_seed_count": mechanism_primary.get("topk2_tradeoff_supporting_seed_count"),
        "strategy_recommended_next_action": strategy.get("recommended_next_action"),
        "strategy_verdict": strategy.get("verdict"),
        "ben_notification_required": strategy.get("ben_notification_required"),
    }


def _evidence_matrix(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _matrix_row(
            "broader_acsr_gate",
            evidence["acsr_gate_status"],
            evidence["acsr_claim_status"],
            evidence["acsr_gate_decision"],
            "blocks rescue unless ACSR beats parameter-matched controls and churn guardrails",
        ),
        _matrix_row(
            "dense_teacher_distillation",
            evidence["dense_teacher_status"],
            evidence["dense_teacher_claim_status"],
            evidence["dense_teacher_decision"],
            "blocks rescue when sparse ACSR cannot approach the dense teacher CE guardrail",
        ),
        _matrix_row(
            "dense24_rank_norm_synthesis",
            evidence["dense_rank_norm_status"],
            evidence["dense_rank_norm_claim_status"],
            evidence["dense_rank_norm_decision"],
            "blocks sparse-specific CE claims when dense rank16/24 controls explain or beat the sparse gain",
        ),
        _matrix_row(
            "mlp_churn_decision",
            evidence["mlp_churn_status"],
            evidence["mlp_churn_claim_status"],
            evidence["mlp_churn_decision"],
            "keeps raw MLP/dense controls active while requiring matched CE/L2/churn before mechanism claims",
        ),
        _matrix_row(
            "norm_budgeted_branch_selector",
            evidence["norm_budgeted_status"],
            evidence["norm_budgeted_claim_status"],
            evidence["norm_budgeted_decision"],
            "records that sparse norm-target, retention, and finite-update branches are locally blocked",
        ),
        _matrix_row(
            "finite_update_commutator",
            evidence["commutator_status"],
            evidence["commutator_claim_status"],
            evidence["commutator_decision"],
            "blocks sparse-interference claims when the commutator is too small to interpret",
        ),
        _matrix_row(
            "mechanism_factorized_cl_repeat",
            evidence["mechanism_cl_status"],
            evidence["mechanism_cl_claim_status"],
            evidence["mechanism_cl_decision"],
            "blocks retention claims when the top-k2 tradeoff does not replicate",
        ),
    ]


def _candidate_actions(
    evidence: dict[str, Any],
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if failures:
        return [
            _candidate(REPAIR_SOURCES_ACTION, "selected", "required source artifacts are missing", "repair missing ACSR closeout source artifacts", "incomplete"),
            _candidate(DEMOTE_ACSR_ACTION, "blocked", "source artifacts are incomplete", "rerun after source repair", "blocked"),
            _candidate(PREFIX_NORM_RESCUE_ACTION, "blocked", "source artifacts are incomplete", "rerun after source repair", "blocked"),
        ]

    rescue_supported = (
        evidence["acsr_gate_status"] == "pass"
        and evidence["acsr_beats_nulls"]
        and evidence["acsr_beats_parameter_matched"]
        and evidence["acsr_retention_churn_guardrail"]
        and not evidence["dense24_or_rank_norm_blocks_sparse"]
        and (
            evidence["dense_teacher_status"] == "pass"
            or evidence["commutator_status"] == "pass"
            or evidence["mechanism_cl_topk2_repeat_status"] == "replicated"
        )
    )
    if rescue_supported:
        return [
            _candidate(
                PREFIX_NORM_RESCUE_ACTION,
                "selected",
                "ACSR survived dense/null/churn guardrails and at least one downstream sparse-mechanism artifact remains supportive",
                "design one prefix-only, residual-norm-normalized ACSR rescue with dense/null/churn controls before GPU validation",
                "acsr_rescue_design_allowed_local_only",
            ),
            _candidate(
                DEMOTE_ACSR_ACTION,
                "deferred",
                "current evidence would still justify one bounded local rescue",
                "revisit after the rescue gate",
                "deferred",
            ),
        ]

    return [
        _candidate(
            DEMOTE_ACSR_ACTION,
            "selected",
            "ACSR beats simple nulls but fails parameter-matched and retention-churn guardrails, while dense-teacher, dense24/rank-norm, MLP, norm-budgeted, commutator, and CL-repeat evidence do not establish a sparse-specific mechanism",
            "treat ACSR as a diagnostic probe; make the next experiment target dense/MLP residual controls rather than ACSR/default-router promotion",
            "acsr_promotion_path_demoted_to_diagnostic_no_default_change",
        ),
        _candidate(
            PREFIX_NORM_RESCUE_ACTION,
            "rejected",
            "a rescue would optimize a branch whose current mechanism evidence is explained by parameter-matched controls or fails downstream sparse-specific gates",
            "only reconsider if a future local ACSR gate beats dense/null/churn/norm controls",
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


def _matrix_row(source: str, status: Any, claim_status: Any, decision: Any, interpretation: str) -> dict[str, Any]:
    return {
        "source": source,
        "status": status,
        "claim_status": claim_status,
        "decision": decision,
        "interpretation": interpretation,
    }


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status") if path.is_file() else "missing",
        "decision": payload.get("decision"),
        "claim_status": payload.get("claim_status"),
    }


def _source_failures(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows:
        if row["source"] == "strategy_review":
            continue
        if not row["present"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "source_artifact",
                    "reason": f"{row['path']} is missing",
                }
            )
    return failures


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        out_dir / "source_rows.csv",
        ["source", "path", "present", "status", "decision", "claim_status"],
        summary["source_rows"],
    )
    _write_csv(
        out_dir / "candidate_actions.csv",
        ["candidate_action", "disposition", "reason", "next_step", "claim_status"],
        summary["candidate_actions"],
    )
    _write_csv(
        out_dir / "evidence_matrix.csv",
        ["source", "status", "claim_status", "decision", "interpretation"],
        summary["evidence_matrix"],
    )
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ACSR Negative Evidence Closeout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected action: `{summary['selected_next_action']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Next step: {summary['next_step']}",
        "",
        summary["rationale"],
        "",
        "This is a local branch selector. It does not promote ACSR or request RunPod/Colab validation.",
    ]
    if summary["direction_shift"]["notify_ben"]:
        lines.extend(
            [
                "",
                "Ben notification is required by the strategy review before treating this as a routine direction shift.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _strategy_review(path: Path) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "present": path.is_file(),
        "strategic_change_level": None,
        "notify_ben": None,
        "recommended_next_action": None,
        "verdict": None,
    }
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key in fields:
                fields[key] = value
    notify = str(fields.get("notify_ben")).lower() == "true"
    major = fields.get("strategic_change_level") == "major"
    fields["ben_notification_required"] = notify or major
    return fields


def _direction_shift_record(strategy: dict[str, Any]) -> dict[str, Any]:
    notify_ben = bool(strategy.get("ben_notification_required"))
    major = strategy.get("strategic_change_level") == "major"
    if major or notify_ben:
        record = "Strategy review requested a major/notify-Ben direction marker; record before treating this as routine."
    else:
        record = "No major strategy-review direction shift; local ACSR closeout follows the accepted fail-closed recommendation."
    return {
        "level": strategy.get("strategic_change_level"),
        "notify_ben": notify_ben,
        "record": record,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _variant_metric(payload: dict[str, Any], variant: str, metric: str) -> float | None:
    for row in payload.get("variant_rows", []):
        if isinstance(row, dict) and row.get("variant") == variant:
            return _float_or_none(row.get(metric))
    return None


def _named_metric(rows: list[Any], metric: str) -> float | None:
    for row in rows:
        if isinstance(row, dict) and row.get("metric") == metric:
            return _float_or_none(row.get("value"))
    return None


def _selected_action(payload: dict[str, Any]) -> str | None:
    for row in payload.get("candidate_actions", []):
        if isinstance(row, dict) and row.get("disposition") == "selected":
            action = row.get("candidate_action")
            return str(action) if action is not None else None
    selected = payload.get("selected_next_action")
    return str(selected) if selected is not None else None


def _float_or_none(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acsr-gate", type=Path, default=DEFAULT_ACSR_GATE)
    parser.add_argument("--dense-teacher", type=Path, default=DEFAULT_DENSE_TEACHER)
    parser.add_argument("--dense-rank-norm", type=Path, default=DEFAULT_DENSE_RANK_NORM)
    parser.add_argument("--mlp-churn", type=Path, default=DEFAULT_MLP_CHURN)
    parser.add_argument("--norm-budgeted", type=Path, default=DEFAULT_NORM_BUDGETED)
    parser.add_argument("--commutator", type=Path, default=DEFAULT_COMMUTATOR)
    parser.add_argument("--mechanism-cl", type=Path, default=DEFAULT_MECHANISM_CL)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_acsr_negative_evidence_closeout_report(
        acsr_gate_path=args.acsr_gate,
        dense_teacher_path=args.dense_teacher,
        dense_rank_norm_path=args.dense_rank_norm,
        mlp_churn_path=args.mlp_churn,
        norm_budgeted_path=args.norm_budgeted,
        commutator_path=args.commutator,
        mechanism_cl_path=args.mechanism_cl,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
