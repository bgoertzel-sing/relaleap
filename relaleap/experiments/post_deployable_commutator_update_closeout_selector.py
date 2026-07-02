"""Close out the deployable commutator-update line after its local probe."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_DEPLOYABLE_PROBE = Path(
    "results/audits/deployable_commutator_regularized_sparse_update_probe/summary.json"
)
DEFAULT_ORDER_CLOSEOUT = Path(
    "results/reports/token_larger_promoted_topk2_order_averaging_closeout/summary.json"
)
DEFAULT_FLAT_CLOSEOUT = Path(
    "results/reports/same_router_flat_value_commutator_mitigation_closeout/summary.json"
)
DEFAULT_MECHANISM_REPEAT = Path(
    "results/reports/mechanism_factorized_continual_learning_repeat/summary.json"
)
DEFAULT_MECHANISM_SELECTOR = Path(
    "results/reports/mechanism_factorized_cl_branch_selector/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/post_deployable_commutator_update_closeout_selector")

CLOSE_ACTION = "close_deployable_commutator_update_line_before_gpu"
REPAIR_ACTION = "repair_post_deployable_commutator_update_closeout_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "decision_matrix.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_post_deployable_commutator_update_closeout_selector(
    *,
    deployable_probe_path: Path = DEFAULT_DEPLOYABLE_PROBE,
    order_closeout_path: Path = DEFAULT_ORDER_CLOSEOUT,
    flat_closeout_path: Path = DEFAULT_FLAT_CLOSEOUT,
    mechanism_repeat_path: Path = DEFAULT_MECHANISM_REPEAT,
    mechanism_selector_path: Path = DEFAULT_MECHANISM_SELECTOR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Consume local mechanism evidence and choose a single non-GPU disposition."""

    start = time.time()
    deployable_probe = _read_json(deployable_probe_path)
    order_closeout = _read_json(order_closeout_path)
    flat_closeout = _read_json(flat_closeout_path)
    mechanism_repeat = _read_json(mechanism_repeat_path)
    mechanism_selector = _read_json(mechanism_selector_path)
    strategy = _strategy_review(strategy_review_path)

    source_rows = [
        _source_row("deployable_commutator_update_probe", deployable_probe_path, deployable_probe),
        _source_row("explicit_order_averaging_closeout", order_closeout_path, order_closeout),
        _source_row("flat_value_commutator_closeout", flat_closeout_path, flat_closeout),
        _source_row("mechanism_factorized_cl_repeat", mechanism_repeat_path, mechanism_repeat),
        _source_row("mechanism_factorized_cl_selector", mechanism_selector_path, mechanism_selector),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}; verdict={strategy['verdict']}"
            ),
            "selected_next_action": "",
        },
    ]
    evidence = _evidence(
        deployable_probe,
        order_closeout,
        flat_closeout,
        mechanism_repeat,
        mechanism_selector,
        strategy,
    )
    decision_matrix = _decision_matrix(evidence, source_rows)
    hard_failures = [
        row for row in decision_matrix if row["severity"] == "hard" and not row["passed"]
    ]
    candidate_actions = _candidate_actions(hard_failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if hard_failures or len(selected) != 1:
        status = "fail"
        decision = "post_deployable_commutator_update_closeout_failed_closed"
        claim_status = "closeout_sources_incomplete_no_research_claim"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair missing or inconsistent closeout sources before choosing another branch"
        rationale = "Required local source artifacts are missing or inconsistent."
    else:
        status = "pass"
        decision = "deployable_commutator_update_line_closed_no_gpu"
        claim_status = "commutator_update_mechanisms_not_established"
        selected_next_action = CLOSE_ACTION
        selected_next_step = (
            "stop commutator-update mechanism work and run a fresh local branch selector "
            "before any new non-GPU mechanism branch or backend validation"
        )
        rationale = (
            "The deployable candidate is interpretable but fails sparse-specific, dense-control, "
            "and support-overlap gates; explicit order averaging is already closed as a "
            "nondeployable diagnostic; flat-value capacity and mechanism-factorized CL do not "
            "rescue a sparse commutator-update claim."
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_action": selected_next_action,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local CPU closeout only; RunPod and Colab remain blocked",
        "source_rows": source_rows,
        "evidence": evidence,
        "decision_matrix": decision_matrix,
        "candidate_actions": candidate_actions,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "rationale": rationale,
        "failures": hard_failures,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _evidence(
    deployable_probe: dict[str, Any],
    order_closeout: dict[str, Any],
    flat_closeout: dict[str, Any],
    mechanism_repeat: dict[str, Any],
    mechanism_selector: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    probe_gates = {
        str(row.get("criterion")): row
        for row in deployable_probe.get("gate_criteria", [])
        if isinstance(row, dict)
    }
    comparisons = {
        str(row.get("control")): row
        for row in deployable_probe.get("control_comparison", [])
        if isinstance(row, dict)
    }
    return {
        "deployable_probe_status": deployable_probe.get("status"),
        "deployable_probe_decision": deployable_probe.get("decision"),
        "deployable_probe_claim_status": deployable_probe.get("claim_status"),
        "deployable_probe_requires_gpu_now": deployable_probe.get("requires_gpu_now"),
        "deployable_probe_advance_to_gpu_validation": deployable_probe.get(
            "advance_to_gpu_validation"
        ),
        "deployable_probe_promotion_allowed": deployable_probe.get("promotion_allowed"),
        "candidate_improves_sparse_gate_passed": _gate_passed(
            probe_gates, "candidate_improves_sparse_commutator"
        ),
        "candidate_beats_dense_gate_passed": _gate_passed(
            probe_gates, "candidate_beats_dense_and_random_controls"
        ),
        "support_overlap_bins_populated": _gate_passed(
            probe_gates, "support_overlap_bins_populated"
        ),
        "candidate_minus_sparse_commutator": comparisons.get(
            "sparse_unregularized_update", {}
        ).get("candidate_minus_control_commutator_anchor_logit_mse"),
        "candidate_minus_dense_active_commutator": comparisons.get(
            "dense_active_matched_update", {}
        ).get("candidate_minus_control_commutator_anchor_logit_mse"),
        "candidate_minus_dense_stored_commutator": comparisons.get(
            "dense_stored_matched_update", {}
        ).get("candidate_minus_control_commutator_anchor_logit_mse"),
        "candidate_minus_random_support_commutator": comparisons.get(
            "random_support_sparse_update", {}
        ).get("candidate_minus_control_commutator_anchor_logit_mse"),
        "order_closeout_decision": order_closeout.get("decision"),
        "order_closeout_claim_status": order_closeout.get("claim_status"),
        "flat_closeout_decision": flat_closeout.get("decision"),
        "flat_closeout_claim_status": flat_closeout.get("claim_status"),
        "mechanism_repeat_claim_status": mechanism_repeat.get("claim_status"),
        "mechanism_selector_selected_next_action": mechanism_selector.get("selected_next_action"),
        "strategy_recommended_next_action": strategy["recommended_next_action"],
        "strategy_verdict": strategy["verdict"],
        "ben_notification_required": strategy["ben_notification_required"],
    }


def _gate_passed(gates: dict[str, dict[str, Any]], name: str) -> bool | None:
    row = gates.get(name)
    if not row:
        return None
    return bool(row.get("passed"))


def _decision_matrix(
    evidence: dict[str, Any], source_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        _criterion(
            "required_sources_present",
            all(row["present"] for row in source_rows[:5]),
            "hard",
            "deployable probe, order closeout, flat closeout, mechanism repeat, and mechanism selector exist",
            [row["path"] for row in source_rows[:5] if row["present"]],
            "missing required source artifact",
        ),
        _criterion(
            "deployable_probe_interpretable",
            evidence["deployable_probe_status"] == "pass"
            and evidence["deployable_probe_decision"]
            == "deployable_commutator_regularized_sparse_update_probe_recorded_gpu_blocked",
            "hard",
            "deployable probe completed and recorded a local GPU-blocked result",
            evidence["deployable_probe_decision"],
            "deployable probe did not complete interpretably",
        ),
        _criterion(
            "gpu_validation_blocked",
            evidence["deployable_probe_requires_gpu_now"] is False
            and evidence["deployable_probe_advance_to_gpu_validation"] is False
            and evidence["deployable_probe_promotion_allowed"] is False,
            "hard",
            "probe blocks RunPod/Colab validation",
            {
                "requires_gpu_now": evidence["deployable_probe_requires_gpu_now"],
                "advance_to_gpu_validation": evidence[
                    "deployable_probe_advance_to_gpu_validation"
                ],
                "promotion_allowed": evidence["deployable_probe_promotion_allowed"],
            },
            "backend validation is not safely blocked",
        ),
        _criterion(
            "sparse_specific_candidate_not_established",
            evidence["deployable_probe_claim_status"] == "deployable_sparse_update_not_established",
            "claim",
            "candidate must not be promoted when local sparse-specific gates fail",
            evidence["deployable_probe_claim_status"],
            "deployable sparse-update claim appears supported unexpectedly",
        ),
        _criterion(
            "candidate_loses_to_dense_controls",
            evidence["candidate_beats_dense_gate_passed"] is False,
            "claim",
            "dense/random control gate must fail for this closeout",
            {
                "dense_active_delta": evidence["candidate_minus_dense_active_commutator"],
                "dense_stored_delta": evidence["candidate_minus_dense_stored_commutator"],
                "random_delta": evidence["candidate_minus_random_support_commutator"],
            },
            "candidate did not lose the dense/random control gate",
        ),
        _criterion(
            "support_overlap_incomplete",
            evidence["support_overlap_bins_populated"] is False,
            "claim",
            "support-overlap strata incompleteness must be recorded before closing",
            evidence["support_overlap_bins_populated"],
            "support-overlap bins were complete",
        ),
        _criterion(
            "order_averaging_already_closed",
            evidence["order_closeout_decision"] == "promoted_topk2_order_averaging_closed_no_gpu",
            "claim",
            "nondeployable order averaging is already closed",
            evidence["order_closeout_decision"],
            "order averaging closeout is missing or inconsistent",
        ),
        _criterion(
            "flat_value_not_sparse_rescue",
            evidence["flat_closeout_claim_status"] == "flat_value_capacity_closed_as_generic_capacity",
            "claim",
            "flat-value mitigation is closed as generic capacity, not sparse-update evidence",
            evidence["flat_closeout_claim_status"],
            "flat-value branch was not closed as a generic-capacity control",
        ),
        _criterion(
            "mechanism_factorized_retention_not_rescue",
            evidence["mechanism_repeat_claim_status"]
            == "mechanism_factorized_sparse_retention_not_established",
            "claim",
            "mechanism-factorized CL repeat does not rescue this commutator-update line",
            evidence["mechanism_repeat_claim_status"],
            "mechanism-factorized CL repeat appears to support a retention claim",
        ),
    ]


def _criterion(
    criterion: str,
    passed: bool,
    severity: str,
    requirement: str,
    observed: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "severity": severity,
        "requirement": requirement,
        "observed": observed,
        "failure_reason": "" if passed else failure_reason,
    }


def _candidate_actions(hard_failures: list[dict[str, Any]]) -> list[dict[str, str]]:
    if hard_failures:
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "closeout source artifacts are missing or inconsistent",
                "repair source artifacts before choosing another branch",
                "source_repair_required",
            )
        ]
    return [
        _candidate(
            CLOSE_ACTION,
            "selected",
            (
                "deployable commutator regularization failed local sparse-specific gates, "
                "lost to dense controls, and lacks complete support-overlap evidence"
            ),
            (
                "stop commutator-update mechanism work and run a fresh local branch selector "
                "before any new non-GPU mechanism branch or backend validation"
            ),
            "commutator_update_mechanisms_not_established",
        ),
        _candidate(
            "rerun_deployable_commutator_probe",
            "rejected",
            "the probe already completed and the failures are scientific gates rather than runtime gaps",
            "do not duplicate the completed local probe",
            "duplicate_probe_rejected",
        ),
        _candidate(
            "promote_explicit_order_averaging",
            "rejected",
            "explicit order averaging is a nondeployable upper-bound diagnostic using both orders",
            "keep order averaging closed before GPU validation",
            "nondeployable_order_averaging_rejected",
        ),
        _candidate(
            "run_runpod_or_colab_validation",
            "rejected",
            "all consumed local artifacts keep requires_gpu_now=false and promotion_allowed=false",
            "keep RunPod and Colab unused for this closeout",
            "gpu_validation_blocked",
        ),
    ]


def _candidate(
    candidate_action: str,
    disposition: str,
    reason: str,
    next_step: str,
    claim_status: str,
) -> dict[str, str]:
    return {
        "candidate_action": candidate_action,
        "disposition": disposition,
        "reason": reason,
        "next_step": next_step,
        "claim_status": claim_status,
    }


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status") if payload else "missing",
        "decision": payload.get("decision") if payload else "",
        "claim_status": payload.get("claim_status") if payload else "",
        "selected_next_action": payload.get("selected_next_action")
        or payload.get("selected_next_step")
        or "",
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _strategy_review(path: Path) -> dict[str, Any]:
    fields = {
        "strategic_change_level": "missing",
        "notify_ben": False,
        "recommended_next_action": "",
        "verdict": "missing",
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
    notify = str(fields["notify_ben"]).lower() == "true"
    strategic_change = str(fields["strategic_change_level"])
    return {
        "path": str(path),
        "present": path.is_file(),
        "strategic_change_level": strategic_change,
        "notify_ben": notify,
        "ben_notification_required": notify or strategic_change == "major",
        "recommended_next_action": str(fields["recommended_next_action"]),
        "verdict": str(fields["verdict"]),
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if strategy["ben_notification_required"]:
        return (
            "Read the external review and recorded that Ben should be notified before "
            "treating the direction shift as settled."
        )
    return (
        "Read the external review. Its local commutator recommendation has now been "
        "followed through the deployable probe and closed without GPU validation."
    )


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "summary.json", summary)
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "decision_matrix.csv", summary["decision_matrix"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Post-Deployable-Commutator Update Closeout Selector",
            "",
            f"- status: {summary['status']}",
            f"- decision: {summary['decision']}",
            f"- claim_status: {summary['claim_status']}",
            f"- selected_next_action: {summary['selected_next_action']}",
            f"- selected_next_step: {summary['selected_next_step']}",
            "- GPU validation remains blocked: requires_gpu_now=false, promotion_allowed=false, advance_to_gpu_validation=false.",
            f"- rationale: {summary['rationale']}",
            f"- strategy_review_handling: {summary['strategy_review_handling']}",
            "",
        ]
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_post_deployable_commutator_update_closeout_selector(out_dir=args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
