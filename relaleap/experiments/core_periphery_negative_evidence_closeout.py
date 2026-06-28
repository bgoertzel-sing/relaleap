"""Close out the current core/periphery PC-column branch after negative local evidence."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_PILOT = Path("results/reports/core_periphery_pc_column_nonsynthetic_pilot/summary.json")
DEFAULT_DESIGN = Path("results/reports/core_periphery_pc_column_nonsynthetic_pilot_design/summary.json")
DEFAULT_SYNTHESIS = Path("results/reports/core_periphery_pc_column_synthesis/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/core_periphery_negative_evidence_closeout")

DEMOTE_CORE_PERIPHERY_ACTION = "demote_current_core_periphery_mechanism_to_diagnostic_status"
RETURN_TO_CONTEXTUAL_ROUTING_ACTION = "return_to_promoted_contextual_routing_dense_teacher_columnability_track"
REPAIR_SOURCES_ACTION = "repair_core_periphery_closeout_source_artifacts"
LOCAL_MECHANISM_REPAIR_ACTION = "design_new_core_periphery_mechanism_before_gpu"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "candidate_actions.csv",
    "evidence_matrix.csv",
    "notes.md",
)


def run_core_periphery_negative_evidence_closeout(
    *,
    pilot_path: Path = DEFAULT_PILOT,
    design_path: Path = DEFAULT_DESIGN,
    synthesis_path: Path = DEFAULT_SYNTHESIS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Select the next local branch after the current core/periphery mechanism fails gates."""

    start = time.time()
    pilot = _read_json(pilot_path)
    design = _read_json(design_path)
    synthesis = _read_json(synthesis_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("core_periphery_nonsynthetic_pilot", pilot_path, pilot),
        _source_row("core_periphery_nonsynthetic_pilot_design", design_path, design),
        _source_row("core_periphery_synthesis", synthesis_path, synthesis),
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
    evidence = _evidence_snapshot(pilot, design, synthesis, strategy)
    evidence_matrix = _evidence_matrix(evidence)
    failures = _source_failures(source_rows)
    candidate_actions = _candidate_actions(evidence, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "core_periphery_negative_evidence_closeout_failed_closed"
        selected_next_action = REPAIR_SOURCES_ACTION
        next_step = "repair missing core/periphery closeout source artifacts"
        claim_status = "core_periphery_closeout_source_evidence_incomplete"
        rationale = "The closeout cannot choose a scientific branch until required local source artifacts are present."
    else:
        status = "pass"
        decision = "core_periphery_negative_evidence_closeout_branch_selected"
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
        "backend_policy": (
            "local closeout only; RunPod/Colab validation remains blocked until a local "
            "mechanism clears useful-periphery, retention, pruning, and dense/MLP control gates"
        ),
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
    pilot: dict[str, Any],
    design: dict[str, Any],
    synthesis: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    primary = _as_dict(pilot.get("primary_result"))
    gate_rows = pilot.get("gate_criteria")
    gate_rows = gate_rows if isinstance(gate_rows, list) else []
    failed_claims = [
        str(row.get("criterion"))
        for row in gate_rows
        if isinstance(row, dict) and row.get("severity") == "claim" and row.get("passed") is False
    ]
    return {
        "pilot_status": pilot.get("status"),
        "pilot_decision": pilot.get("decision"),
        "pilot_claim_status": pilot.get("claim_status"),
        "pilot_scientific_gate": pilot.get("scientific_gate"),
        "pilot_selected_next_step": pilot.get("selected_next_step"),
        "design_status": design.get("status"),
        "design_scientific_gate": design.get("scientific_gate"),
        "synthesis_status": synthesis.get("status"),
        "synthesis_decision": synthesis.get("decision"),
        "synthesis_claim_status": synthesis.get("claim_status"),
        "primary_variant": primary.get("primary_variant"),
        "heldout_ce": _float_or_none(primary.get("heldout_ce")),
        "anchor_kl_drift": _float_or_none(primary.get("anchor_kl_drift")),
        "core_minus_dense_anchor_kl_drift": _float_or_none(primary.get("core_minus_dense_anchor_kl_drift")),
        "core_minus_mlp_anchor_kl_drift": _float_or_none(primary.get("core_minus_mlp_anchor_kl_drift")),
        "periphery_deployment_fraction": _float_or_none(primary.get("periphery_deployment_fraction")),
        "effective_periphery_residual_norm": _float_or_none(primary.get("effective_periphery_residual_norm")),
        "paired_train_periphery_utility_mean": _float_or_none(primary.get("paired_train_periphery_utility_mean")),
        "paired_heldout_periphery_utility_mean": _float_or_none(primary.get("paired_heldout_periphery_utility_mean")),
        "paired_heldout_periphery_utility_positive_fraction": _float_or_none(
            primary.get("paired_heldout_periphery_utility_positive_fraction")
        ),
        "periphery_first_minus_core_first_prune_delta": _float_or_none(
            primary.get("periphery_first_minus_core_first_prune_delta")
        ),
        "failed_claims": failed_claims,
        "useful_periphery_observed": (
            _float_or_none(primary.get("periphery_deployment_fraction"), 0.0) > 0.0
            and _float_or_none(primary.get("effective_periphery_residual_norm"), 0.0) > 1e-6
            and _float_or_none(primary.get("paired_heldout_periphery_utility_mean"), 0.0) > 0.0
        ),
        "retention_worse_than_dense_or_mlp": (
            _float_or_none(primary.get("core_minus_dense_anchor_kl_drift"), 0.0) > 0.0
            or _float_or_none(primary.get("core_minus_mlp_anchor_kl_drift"), 0.0) > 0.0
        ),
        "protected_core_factorization_failed": (
            _float_or_none(primary.get("periphery_first_minus_core_first_prune_delta"), 0.0) <= 0.0
        ),
        "strategy_verdict": strategy.get("verdict"),
        "strategy_recommended_next_action": strategy.get("recommended_next_action"),
        "ben_notification_required": strategy.get("ben_notification_required"),
    }


def _evidence_matrix(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _matrix_row(
            "local_nonsynthetic_pilot",
            evidence["pilot_status"],
            evidence["pilot_claim_status"],
            evidence["pilot_decision"],
            "source of truth for current core/periphery mechanism readiness",
        ),
        _matrix_row(
            "useful_periphery_gate",
            "pass" if evidence["useful_periphery_observed"] else "fail",
            "nonzero deployment plus positive paired heldout utility",
            evidence["primary_variant"],
            "necessary but not sufficient for promotion or GPU validation",
        ),
        _matrix_row(
            "dense_mlp_retention_control",
            "fail" if evidence["retention_worse_than_dense_or_mlp"] else "pass",
            "retention worse than dense or MLP controls",
            {
                "minus_dense": evidence["core_minus_dense_anchor_kl_drift"],
                "minus_mlp": evidence["core_minus_mlp_anchor_kl_drift"],
            },
            "blocks current mechanism as protected-core evidence",
        ),
        _matrix_row(
            "protected_core_pruning_signal",
            "fail" if evidence["protected_core_factorization_failed"] else "pass",
            "core pruning should be more damaging than periphery pruning on anchors",
            evidence["periphery_first_minus_core_first_prune_delta"],
            "blocks core/periphery factorization claim",
        ),
        _matrix_row(
            "strategy_review",
            evidence["strategy_verdict"],
            "external review recommendation",
            evidence["strategy_recommended_next_action"],
            "minor/non-notify review keeps next step local",
        ),
    ]


def _candidate_actions(evidence: dict[str, Any], failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if failures:
        return [
            _candidate(
                REPAIR_SOURCES_ACTION,
                "selected",
                "required closeout source artifacts are missing",
                "repair missing core/periphery closeout source artifacts",
                "source_artifact_repair_required",
            ),
            _candidate(
                LOCAL_MECHANISM_REPAIR_ACTION,
                "blocked",
                "cannot design the next local mechanism from incomplete source evidence",
                "rerun after source artifact repair",
                "source_artifact_repair_required",
            ),
            _candidate(
                RETURN_TO_CONTEXTUAL_ROUTING_ACTION,
                "blocked",
                "cannot redirect branch from incomplete source evidence",
                "rerun after source artifact repair",
                "source_artifact_repair_required",
            ),
        ]

    current_failed = (
        evidence["pilot_scientific_gate"] == "blocked"
        or evidence["retention_worse_than_dense_or_mlp"]
        or evidence["protected_core_factorization_failed"]
    )
    if current_failed:
        return [
            _candidate(
                DEMOTE_CORE_PERIPHERY_ACTION,
                "selected",
                "the current mechanism has useful periphery CE signal but fails dense/MLP retention and protected-core pruning gates, so it is diagnostic rather than GPU/promotion evidence",
                "record this closeout, keep RunPod blocked, and redirect the next experiment to promoted contextual routing plus dense-teacher columnability/causal probes unless Ben requests another mechanism attempt",
                "current_core_periphery_mechanism_demoted_no_gpu_or_default_change",
            ),
            _candidate(
                LOCAL_MECHANISM_REPAIR_ACTION,
                "deferred",
                "another core/periphery mechanism attempt would need a new design, not another validation run of this mechanism",
                "only resume if Ben requests another core/periphery attempt or a new local design materially changes the retention/pruning gates",
                "core_periphery_mechanism_repair_requires_new_design",
            ),
            _candidate(
                RETURN_TO_CONTEXTUAL_ROUTING_ACTION,
                "deferred",
                "this is the operational branch after the demotion record is written",
                "start the next bounded report/probe on promoted contextual routing plus dense-teacher columnability and causal controls",
                "contextual_routing_dense_teacher_track_active_after_closeout",
            ),
        ]

    return [
        _candidate(
            DEMOTE_CORE_PERIPHERY_ACTION,
            "rejected",
            "the local mechanism cleared retention and pruning gates",
            "not applicable unless future local repeats reverse the gates",
            "local_candidate_not_demoted",
        ),
        _candidate(
            LOCAL_MECHANISM_REPAIR_ACTION,
            "selected",
            "the local mechanism did not fail closeout gates; the next bounded step is a repeat or repair before GPU",
            "run a second local seed or a focused local mechanism repair before any RunPod validation",
            "local_candidate_requires_repeat_before_gpu",
        ),
        _candidate(
            RETURN_TO_CONTEXTUAL_ROUTING_ACTION,
            "rejected",
            "core/periphery local evidence is not negative enough to redirect the branch",
            "revisit only after repeat evidence",
            "core_periphery_branch_still_local_candidate",
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


def _matrix_row(
    source: str,
    status: Any,
    claim_status: Any,
    decision: Any,
    interpretation: str,
) -> dict[str, Any]:
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
    for row in source_rows[:3]:
        if not row["present"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "source_artifact",
                    "failure_reason": f"{row['path']} is missing",
                }
            )
    return failures


def _direction_shift_record(strategy: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategic_change_level": strategy.get("strategic_change_level"),
        "notify_ben": strategy.get("notify_ben"),
        "ben_should_be_notified": bool(strategy.get("ben_notification_required")),
        "direction_shift": (
            "major_or_notify_review_present"
            if strategy.get("ben_notification_required")
            else "no_major_external_direction_shift"
        ),
        "recorded_response": (
            "Ben should be notified before treating this as routine automation"
            if strategy.get("ben_notification_required")
            else "external review did not request Ben notification"
        ),
    }


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
    direction = summary["direction_shift"]
    lines = [
        "# Core/Periphery Negative-Evidence Closeout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Ben notification: `{direction['recorded_response']}`",
        "",
        summary["rationale"],
        "",
        "Interpretation: the current local mechanism is not promotion or RunPod evidence when "
        "dense/MLP retention or protected-core pruning gates fail, even if the periphery has "
        "positive paired heldout utility.",
        "",
        f"Next step: {summary['next_step']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
    fields["ben_notification_required"] = notify or fields.get("strategic_change_level") == "major"
    return fields


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


def _float_or_none(value: Any, default: float | None = None) -> float | None:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--synthesis", type=Path, default=DEFAULT_SYNTHESIS)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_core_periphery_negative_evidence_closeout(
        pilot_path=args.pilot,
        design_path=args.design,
        synthesis_path=args.synthesis,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
