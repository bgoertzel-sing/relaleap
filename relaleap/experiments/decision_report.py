"""Make local HEP mechanism decisions from completed comparison artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from relaleap.experiments.check_artifacts import check_comparison_artifacts


DEFAULT_COMPARISON_DIR = Path(
    "results/comparisons/colab_support_stress_pinned_vs_repicked"
)
DEFAULT_OUT_DIR = Path("results/reports/pinned_support_decision")
DEFAULT_CLIPPED_COMPARISON_DIR = Path(
    "results/comparisons/colab_support_stress_clipped_hep"
)
DEFAULT_CLIPPED_OUT_DIR = Path("results/reports/clipped_hep_decision")
DEFAULT_GUIDED_CLIPPED_COMPARISON_DIR = Path(
    "results/comparisons/colab_support_stress_guided_clipped_hep"
)
DEFAULT_GUIDED_CLIPPED_OUT_DIR = Path("results/reports/guided_clipped_hep_decision")
DEFAULT_TEMPORAL_CLIPPED_COMPARISON_DIR = Path(
    "results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep"
)
DEFAULT_TEMPORAL_CLIPPED_OUT_DIR = Path("results/reports/temporal_clipped_hep_decision")
DEFAULT_TEMPORAL_CLIPPED_AGGREGATE_REPORTS = (
    Path("results/reports/temporal_clipped_hep_seed1_local_decision/decision_report.json"),
    Path("results/reports/temporal_clipped_hep_decision/decision_report.json"),
    Path("results/reports/temporal_clipped_hep_seed2_decision/decision_report.json"),
    Path("results/reports/temporal_clipped_hep_seed2_colab_decision/decision_report.json"),
    Path("results/reports/temporal_clipped_hep_seed3_local_decision/decision_report.json"),
    Path("results/reports/temporal_clipped_hep_seed3_colab_decision/decision_report.json"),
    Path("results/reports/temporal_clipped_hep_seed4_local_decision/decision_report.json"),
    Path("results/reports/temporal_clipped_hep_seed4_colab_decision/decision_report.json"),
)
DEFAULT_TEMPORAL_CLIPPED_AGGREGATE_OUT_DIR = Path(
    "results/reports/temporal_clipped_hep_multiseed_aggregate"
)
DEFAULT_TEMPORAL_CLIPPED_CROSS_SCALE_AGGREGATE_REPORTS = (
    *DEFAULT_TEMPORAL_CLIPPED_AGGREGATE_REPORTS,
    Path(
        "results/reports/temporal_clipped_hep_validation_local_decision/decision_report.json"
    ),
    Path(
        "results/reports/temporal_clipped_hep_validation_colab_decision/decision_report.json"
    ),
    Path(
        "results/reports/temporal_clipped_hep_extended_local_decision/decision_report.json"
    ),
    Path(
        "results/reports/temporal_clipped_hep_extended_colab_decision/decision_report.json"
    ),
)
DEFAULT_TEMPORAL_CLIPPED_CROSS_SCALE_AGGREGATE_OUT_DIR = Path(
    "results/reports/temporal_clipped_hep_cross_scale_aggregate"
)
DEFAULT_TEMPORAL_CLIPPED_PROMOTION_GATE_CROSS_SCALE_REPORT = (
    DEFAULT_TEMPORAL_CLIPPED_CROSS_SCALE_AGGREGATE_OUT_DIR / "decision_report.json"
)
DEFAULT_TEMPORAL_CLIPPED_PROMOTION_GATE_OUT_DIR = Path(
    "results/reports/temporal_clipped_hep_promotion_gate"
)
DEFAULT_TEMPORAL_CLIPPED_PROMOTION_GATE_SATISFACTION_REPORT = (
    DEFAULT_TEMPORAL_CLIPPED_PROMOTION_GATE_OUT_DIR / "decision_report.json"
)
DEFAULT_TEMPORAL_CLIPPED_PROMOTION_GATE_SATISFACTION_REPORTS = (
    Path("results/reports/temporal_clipped_hep_larger_local_decision/decision_report.json"),
    Path("results/reports/temporal_clipped_hep_larger_colab_decision/decision_report.json"),
    Path(
        "results/reports/temporal_clipped_hep_token_larger_local_decision/decision_report.json"
    ),
    Path(
        "results/reports/temporal_clipped_hep_token_larger_colab_decision/decision_report.json"
    ),
)
DEFAULT_TEMPORAL_CLIPPED_PROMOTION_GATE_SATISFACTION_OUT_DIR = Path(
    "results/reports/temporal_clipped_hep_promotion_gate_satisfaction"
)
DEFAULT_POST_PROMOTION_GATE_SATISFACTION_REPORT = (
    DEFAULT_TEMPORAL_CLIPPED_PROMOTION_GATE_SATISFACTION_OUT_DIR
    / "decision_report.json"
)
DEFAULT_POST_PROMOTION_GATE_CONFIG = Path("configs/char_smoke_hep_support_stress.yaml")
DEFAULT_POST_PROMOTION_GATE_OUT_DIR = Path(
    "results/reports/post_promotion_residual_learning_gate"
)
DEFAULT_RESIDUAL_OBJECTIVE_GATE_COMPARISON_DIRS = (
    Path("results/comparisons/validation_pc_vs_supervised_temporal_clipped_objective_gate"),
    Path(
        "results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_objective_gate"
    ),
)
DEFAULT_RESIDUAL_OBJECTIVE_GATE_ARTIFACT_CHECKS = (
    DEFAULT_RESIDUAL_OBJECTIVE_GATE_COMPARISON_DIRS[0] / "artifact_check_local.json",
    DEFAULT_RESIDUAL_OBJECTIVE_GATE_COMPARISON_DIRS[1] / "artifact_check_local.json",
)
DEFAULT_RESIDUAL_OBJECTIVE_GATE_OUT_DIR = Path(
    "results/reports/residual_objective_gate_decision"
)
DEFAULT_PC_RESIDUAL_OBJECTIVE_DIAGNOSTICS_OUT_DIR = Path(
    "results/reports/pc_residual_objective_diagnostics"
)
DEFAULT_ANCHORED_PC_RESIDUAL_OBJECTIVE_COMPARISON_DIRS = (
    Path("results/comparisons/validation_pc_anchor_temporal_clipped_objective_gate"),
    Path(
        "results/comparisons/colab_validation_pc_anchor_temporal_clipped_objective_gate"
    ),
)
DEFAULT_ANCHORED_PC_RESIDUAL_OBJECTIVE_ARTIFACT_CHECKS = (
    DEFAULT_ANCHORED_PC_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[0]
    / "artifact_check_local.json",
    DEFAULT_ANCHORED_PC_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[1]
    / "artifact_check_local.json",
)
DEFAULT_ANCHORED_PC_RESIDUAL_OBJECTIVE_OUT_DIR = Path(
    "results/reports/anchored_pc_residual_objective_decision"
)
DEFAULT_MAX_LOGIT_DELTA = 0.1
DEFAULT_MAX_PINNED_VS_REPICKED_DELTA = 0.1
PROMOTE = "promote_to_default_phase0_baseline"
PROMOTE_CLIPPED_HEP = "promote_to_default_support_stress_mitigation"
GUIDED_ORACLE_CONFIRMED = "guided_oracle_confirmed"
SELECT_TEMPORAL_CLIPPED_HEP = "select_temporal_label_free_support_stress_candidate"
SELECT_TEMPORAL_CLIPPED_HEP_AGGREGATE = (
    "select_temporal_label_free_support_stress_candidate_across_seed_smoke_evidence"
)
SELECT_TEMPORAL_CLIPPED_HEP_CROSS_SCALE_AGGREGATE = (
    "select_temporal_label_free_support_stress_candidate_across_cross_scale_evidence"
)
DEFINE_TEMPORAL_CLIPPED_HEP_PROMOTION_GATE = (
    "define_temporal_label_free_support_stress_promotion_gate"
)
SATISFY_TEMPORAL_CLIPPED_HEP_PROMOTION_GATE = (
    "satisfy_temporal_label_free_support_stress_promotion_gate"
)
DEFINE_POST_PROMOTION_RESIDUAL_LEARNING_GATE = (
    "define_post_promotion_residual_layer_learning_gate"
)
KEEP_SUPERVISED_CE_RESIDUAL_OBJECTIVE_DEFAULT = (
    "keep_supervised_ce_residual_objective_default"
)
CONTINUE_PC_RESIDUAL_OBJECTIVE_VALIDATION = (
    "continue_pc_residual_objective_validation"
)
DIAGNOSE_PC_RESIDUAL_OBJECTIVE = "diagnose_pc_residual_objective_gap"
STOP_PC_RESIDUAL_OBJECTIVE_VALIDATION = "stop_pc_residual_objective_validation"
KEEP_OPT_IN = "keep_opt_in"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def write_pinned_support_decision_report(
    comparison_dir: Path = DEFAULT_COMPARISON_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    artifact_check_path: Path | None = None,
    max_logit_delta: float = DEFAULT_MAX_LOGIT_DELTA,
) -> dict[str, Any]:
    """Write a JSON and Markdown decision report for pinned-support HEP."""

    if max_logit_delta < 0.0:
        raise ValueError("max_logit_delta must be non-negative")

    comparison = _read_json_object(comparison_dir / "summary.json")
    artifact_check = (
        _read_json_object(artifact_check_path)
        if artifact_check_path is not None and artifact_check_path.is_file()
        else check_comparison_artifacts(comparison_dir)
    )
    runs = comparison.get("runs") if isinstance(comparison.get("runs"), list) else []
    pinned_runs = [
        run
        for run in runs
        if isinstance(run, dict) and run.get("pinned_support") is True
    ]
    repicked_runs = [
        run
        for run in runs
        if isinstance(run, dict) and run.get("pinned_support") is False
    ]
    evidence = {
        "comparison_dir": str(comparison_dir),
        "artifact_check_status": artifact_check.get("status"),
        "comparison_status": comparison.get("status"),
        "verdict_status": (comparison.get("verdict") or {}).get("status")
        if isinstance(comparison.get("verdict"), dict)
        else None,
        "pinned_run_count": len(pinned_runs),
        "repicked_run_count": len(repicked_runs),
        "support_stress_run_count": len(
            [
                run
                for run in runs
                if isinstance(run, dict) and run.get("support_stress") is True
            ]
        ),
        "max_support_change_fraction": _max_nested_metric(
            runs,
            "support_instability",
            "support_change_fraction",
        ),
        "max_pinned_vs_repicked_logit_delta": _max_nested_metric(
            runs,
            "support_instability",
            "pinned_vs_repicked_logit_delta",
        ),
        "pinned_alpha_candidates": _alpha_candidates(pinned_runs),
        "repicked_alpha_candidates": _alpha_candidates(repicked_runs),
    }
    decision = _decision(evidence, max_logit_delta=max_logit_delta)
    report = {
        "status": "pass" if decision["decision"] != INSUFFICIENT_EVIDENCE else "fail",
        "decision": decision["decision"],
        "promote_to_default_phase0_baseline": decision["promote"],
        "policy": {
            "max_logit_delta_from_ordinary": max_logit_delta,
            "requires_passing_artifact_check": True,
            "requires_pinned_nonzero_alpha_loss_improvement": True,
            "requires_pinned_nonzero_alpha_within_logit_delta_budget": True,
        },
        "evidence": evidence,
        "rationale": decision["rationale"],
        "next_step": decision["next_step"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(out_dir / "decision_report.md", report)
    return report


def write_clipped_hep_decision_report(
    comparison_dir: Path = DEFAULT_CLIPPED_COMPARISON_DIR,
    out_dir: Path = DEFAULT_CLIPPED_OUT_DIR,
    *,
    artifact_check_path: Path | None = None,
    max_logit_delta: float = DEFAULT_MAX_LOGIT_DELTA,
    max_pinned_vs_repicked_delta: float = DEFAULT_MAX_PINNED_VS_REPICKED_DELTA,
) -> dict[str, Any]:
    """Write a JSON and Markdown decision report for clipped HEP settling."""

    if max_logit_delta < 0.0:
        raise ValueError("max_logit_delta must be non-negative")
    if max_pinned_vs_repicked_delta < 0.0:
        raise ValueError("max_pinned_vs_repicked_delta must be non-negative")

    comparison = _read_json_object(comparison_dir / "summary.json")
    artifact_check = (
        _read_json_object(artifact_check_path)
        if artifact_check_path is not None and artifact_check_path.is_file()
        else check_comparison_artifacts(comparison_dir)
    )
    runs = comparison.get("runs") if isinstance(comparison.get("runs"), list) else []
    clipped_runs = [
        run
        for run in runs
        if isinstance(run, dict) and run.get("hep_update_clip_norm") is not None
    ]
    unclipped_runs = [
        run
        for run in runs
        if isinstance(run, dict) and run.get("hep_update_clip_norm") is None
    ]
    max_clipped_divergence = _max_alpha_metric(
        clipped_runs,
        "pinned_vs_repicked_logit_delta",
    )
    max_unclipped_divergence = _max_alpha_metric(
        unclipped_runs,
        "pinned_vs_repicked_logit_delta",
    )
    evidence = {
        "comparison_dir": str(comparison_dir),
        "artifact_check_status": artifact_check.get("status"),
        "comparison_status": comparison.get("status"),
        "verdict_status": (comparison.get("verdict") or {}).get("status")
        if isinstance(comparison.get("verdict"), dict)
        else None,
        "clipped_run_count": len(clipped_runs),
        "unclipped_run_count": len(unclipped_runs),
        "support_stress_run_count": len(
            [
                run
                for run in runs
                if isinstance(run, dict) and run.get("support_stress") is True
            ]
        ),
        "max_support_change_fraction": _max_nested_metric(
            runs,
            "support_instability",
            "support_change_fraction",
        ),
        "max_unclipped_pinned_vs_repicked_logit_delta": max_unclipped_divergence,
        "max_clipped_pinned_vs_repicked_logit_delta": max_clipped_divergence,
        "max_pinned_vs_repicked_logit_delta_reduction": (
            None
            if max_unclipped_divergence is None or max_clipped_divergence is None
            else max_unclipped_divergence - max_clipped_divergence
        ),
        "clipped_alpha_candidates": _alpha_candidates(clipped_runs),
        "unclipped_alpha_candidates": _alpha_candidates(unclipped_runs),
    }
    decision = _clipped_decision(
        evidence,
        max_logit_delta=max_logit_delta,
        max_pinned_vs_repicked_delta=max_pinned_vs_repicked_delta,
    )
    report = {
        "status": "pass" if decision["decision"] != INSUFFICIENT_EVIDENCE else "fail",
        "decision": decision["decision"],
        "promote_to_default_support_stress_mitigation": decision["promote"],
        "policy": {
            "max_logit_delta_from_ordinary": max_logit_delta,
            "max_pinned_vs_repicked_logit_delta": max_pinned_vs_repicked_delta,
            "requires_passing_artifact_check": True,
            "requires_clipped_nonzero_alpha_loss_improvement": True,
            "requires_clipped_nonzero_alpha_within_logit_delta_budget": True,
            "requires_clipped_nonzero_alpha_within_pinned_vs_repicked_budget": True,
            "requires_nonzero_support_repick_evidence": True,
        },
        "evidence": evidence,
        "rationale": decision["rationale"],
        "next_step": decision["next_step"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_clipped_markdown(out_dir / "decision_report.md", report)
    return report


def write_guided_clipped_hep_decision_report(
    comparison_dir: Path = DEFAULT_GUIDED_CLIPPED_COMPARISON_DIR,
    out_dir: Path = DEFAULT_GUIDED_CLIPPED_OUT_DIR,
    *,
    artifact_check_path: Path | None = None,
    max_logit_delta: float = DEFAULT_MAX_LOGIT_DELTA,
    max_pinned_vs_repicked_delta: float = DEFAULT_MAX_PINNED_VS_REPICKED_DELTA,
) -> dict[str, Any]:
    """Write a JSON and Markdown decision report for guided clipped HEP."""

    if max_logit_delta < 0.0:
        raise ValueError("max_logit_delta must be non-negative")
    if max_pinned_vs_repicked_delta < 0.0:
        raise ValueError("max_pinned_vs_repicked_delta must be non-negative")

    comparison = _read_json_object(comparison_dir / "summary.json")
    artifact_check = (
        _read_json_object(artifact_check_path)
        if artifact_check_path is not None and artifact_check_path.is_file()
        else check_comparison_artifacts(comparison_dir)
    )
    runs = comparison.get("runs") if isinstance(comparison.get("runs"), list) else []
    guided_runs = [
        run
        for run in runs
        if isinstance(run, dict)
        and run.get("hep_settling_objective") == "supervised_ce_gradient"
    ]
    clipped_baseline_runs = [
        run
        for run in runs
        if isinstance(run, dict)
        and run.get("hep_update_clip_norm") is not None
        and run.get("hep_settling_objective") != "supervised_ce_gradient"
    ]
    evidence = {
        "comparison_dir": str(comparison_dir),
        "artifact_check_status": artifact_check.get("status"),
        "comparison_status": comparison.get("status"),
        "verdict_status": (comparison.get("verdict") or {}).get("status")
        if isinstance(comparison.get("verdict"), dict)
        else None,
        "guided_run_count": len(guided_runs),
        "clipped_baseline_run_count": len(clipped_baseline_runs),
        "support_stress_run_count": len(
            [
                run
                for run in runs
                if isinstance(run, dict) and run.get("support_stress") is True
            ]
        ),
        "max_support_change_fraction": _max_nested_metric(
            runs,
            "support_instability",
            "support_change_fraction",
        ),
        "max_guided_pinned_vs_repicked_logit_delta": _max_alpha_metric(
            guided_runs,
            "pinned_vs_repicked_logit_delta",
        ),
        "guided_alpha_candidates": _alpha_candidates(guided_runs),
        "clipped_baseline_alpha_candidates": _alpha_candidates(clipped_baseline_runs),
    }
    decision = _guided_clipped_decision(
        evidence,
        max_logit_delta=max_logit_delta,
        max_pinned_vs_repicked_delta=max_pinned_vs_repicked_delta,
    )
    report = {
        "status": "pass" if decision["decision"] != INSUFFICIENT_EVIDENCE else "fail",
        "decision": decision["decision"],
        "promote_to_default_support_stress_mitigation": False,
        "diagnostic_oracle_only": True,
        "policy": {
            "max_logit_delta_from_ordinary": max_logit_delta,
            "max_pinned_vs_repicked_logit_delta": max_pinned_vs_repicked_delta,
            "requires_passing_artifact_check": True,
            "requires_guided_nonzero_alpha_loss_improvement": True,
            "requires_guided_nonzero_alpha_within_logit_delta_budget": True,
            "requires_guided_nonzero_alpha_within_pinned_vs_repicked_budget": True,
            "requires_nonzero_support_repick_evidence": True,
            "allows_default_promotion": False,
            "reason_default_promotion_is_blocked": (
                "supervised_ce_gradient uses labels during settling and is an "
                "oracle probe, not a deployable inference signal"
            ),
        },
        "evidence": evidence,
        "rationale": decision["rationale"],
        "next_step": decision["next_step"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_guided_clipped_markdown(out_dir / "decision_report.md", report)
    return report


def write_temporal_clipped_hep_decision_report(
    comparison_dir: Path = DEFAULT_TEMPORAL_CLIPPED_COMPARISON_DIR,
    out_dir: Path = DEFAULT_TEMPORAL_CLIPPED_OUT_DIR,
    *,
    artifact_check_path: Path | None = None,
    max_logit_delta: float = DEFAULT_MAX_LOGIT_DELTA,
    max_pinned_vs_repicked_delta: float = DEFAULT_MAX_PINNED_VS_REPICKED_DELTA,
) -> dict[str, Any]:
    """Write a JSON and Markdown decision report for temporal clipped HEP."""

    if max_logit_delta < 0.0:
        raise ValueError("max_logit_delta must be non-negative")
    if max_pinned_vs_repicked_delta < 0.0:
        raise ValueError("max_pinned_vs_repicked_delta must be non-negative")

    comparison = _read_json_object(comparison_dir / "summary.json")
    artifact_check = (
        _read_json_object(artifact_check_path)
        if artifact_check_path is not None and artifact_check_path.is_file()
        else check_comparison_artifacts(comparison_dir)
    )
    runs = comparison.get("runs") if isinstance(comparison.get("runs"), list) else []
    temporal_runs = _runs_with_objective(runs, "temporal_consistency_gradient")
    entropy_runs = _runs_with_objective(runs, "prediction_entropy_gradient")
    guided_runs = _runs_with_objective(runs, "supervised_ce_gradient")
    clipped_baseline_runs = [
        run
        for run in runs
        if isinstance(run, dict)
        and run.get("hep_update_clip_norm") is not None
        and run.get("hep_settling_objective")
        not in {
            "prediction_entropy_gradient",
            "temporal_consistency_gradient",
            "supervised_ce_gradient",
        }
    ]
    evidence = {
        "comparison_dir": str(comparison_dir),
        "artifact_check_status": artifact_check.get("status"),
        "comparison_status": comparison.get("status"),
        "verdict_status": (comparison.get("verdict") or {}).get("status")
        if isinstance(comparison.get("verdict"), dict)
        else None,
        "temporal_run_count": len(temporal_runs),
        "entropy_run_count": len(entropy_runs),
        "guided_run_count": len(guided_runs),
        "clipped_baseline_run_count": len(clipped_baseline_runs),
        "support_stress_run_count": len(
            [
                run
                for run in runs
                if isinstance(run, dict) and run.get("support_stress") is True
            ]
        ),
        "max_support_change_fraction": _max_nested_metric(
            runs,
            "support_instability",
            "support_change_fraction",
        ),
        "max_temporal_pinned_vs_repicked_logit_delta": _max_alpha_metric(
            temporal_runs,
            "pinned_vs_repicked_logit_delta",
        ),
        "temporal_alpha_candidates": _alpha_candidates(temporal_runs),
        "entropy_alpha_candidates": _alpha_candidates(entropy_runs),
        "guided_alpha_candidates": _alpha_candidates(guided_runs),
        "clipped_baseline_alpha_candidates": _alpha_candidates(clipped_baseline_runs),
    }
    decision = _temporal_clipped_decision(
        evidence,
        max_logit_delta=max_logit_delta,
        max_pinned_vs_repicked_delta=max_pinned_vs_repicked_delta,
    )
    report = {
        "status": "pass" if decision["decision"] != INSUFFICIENT_EVIDENCE else "fail",
        "decision": decision["decision"],
        "selected_label_free_support_stress_candidate": decision["selected"],
        "promote_to_default_support_stress_mitigation": False,
        "deployable_label_free_signal": True,
        "policy": {
            "max_logit_delta_from_ordinary": max_logit_delta,
            "max_pinned_vs_repicked_logit_delta": max_pinned_vs_repicked_delta,
            "requires_passing_artifact_check": True,
            "requires_temporal_nonzero_alpha_loss_improvement": True,
            "requires_temporal_nonzero_alpha_within_logit_delta_budget": True,
            "requires_temporal_nonzero_alpha_within_pinned_vs_repicked_budget": True,
            "requires_nonzero_support_repick_evidence": True,
            "requires_entropy_and_guided_context_runs": True,
            "allows_default_promotion": False,
            "reason_default_promotion_is_blocked": (
                "temporal_consistency_gradient is deployable, but this report only "
                "selects the next label-free support-stress candidate; default "
                "promotion requires broader evidence than the current smoke comparison"
            ),
        },
        "evidence": evidence,
        "rationale": decision["rationale"],
        "next_step": decision["next_step"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_temporal_clipped_markdown(out_dir / "decision_report.md", report)
    return report


def write_temporal_clipped_hep_aggregate_report(
    decision_report_paths: list[Path] | tuple[Path, ...] = (
        DEFAULT_TEMPORAL_CLIPPED_AGGREGATE_REPORTS
    ),
    out_dir: Path = DEFAULT_TEMPORAL_CLIPPED_AGGREGATE_OUT_DIR,
    *,
    max_logit_delta: float = DEFAULT_MAX_LOGIT_DELTA,
    max_pinned_vs_repicked_delta: float = DEFAULT_MAX_PINNED_VS_REPICKED_DELTA,
) -> dict[str, Any]:
    """Write a multi-seed aggregate report from temporal clipped decisions."""

    if max_logit_delta < 0.0:
        raise ValueError("max_logit_delta must be non-negative")
    if max_pinned_vs_repicked_delta < 0.0:
        raise ValueError("max_pinned_vs_repicked_delta must be non-negative")

    entries = []
    failures = []
    for path in decision_report_paths:
        path = Path(path)
        if not path.is_file():
            failures.append(
                {
                    "field": "decision_report",
                    "expected": "file exists",
                    "actual": "missing",
                    "path": str(path),
                }
            )
            continue
        report = _read_json_object(path)
        entry = _temporal_aggregate_entry(
            path,
            report,
            max_logit_delta=max_logit_delta,
            max_pinned_vs_repicked_delta=max_pinned_vs_repicked_delta,
        )
        entries.append(entry)
        failures.extend(entry["failures"])

    seed_backend_pairs = sorted(
        {
            (entry["seed"], entry["backend"])
            for entry in entries
            if entry["seed"] is not None and entry["backend"] is not None
        }
    )
    selected_entries = [
        entry for entry in entries if entry["selected_label_free_support_stress_candidate"]
    ]
    accepted_entries = [
        entry for entry in entries if entry["best_temporal_alpha"] is not None
    ]
    improvements = [
        entry["best_temporal_alpha"]["loss_improvement_from_alpha0"]
        for entry in accepted_entries
        if entry["best_temporal_alpha"]["loss_improvement_from_alpha0"] is not None
    ]
    logit_deltas = [
        entry["best_temporal_alpha"]["max_logit_delta_from_ordinary"]
        for entry in accepted_entries
        if entry["best_temporal_alpha"]["max_logit_delta_from_ordinary"] is not None
    ]
    pinned_deltas = [
        entry["best_temporal_alpha"]["pinned_vs_repicked_logit_delta"]
        for entry in accepted_entries
        if entry["best_temporal_alpha"]["pinned_vs_repicked_logit_delta"] is not None
    ]
    support_changes = [
        entry["max_support_change_fraction"]
        for entry in entries
        if entry["max_support_change_fraction"] is not None
    ]
    decision = _temporal_aggregate_decision(entries, failures)
    report = {
        "status": "pass" if decision["decision"] != INSUFFICIENT_EVIDENCE else "fail",
        "decision": decision["decision"],
        "selected_label_free_support_stress_candidate": decision["selected"],
        "promote_to_default_support_stress_mitigation": False,
        "deployable_label_free_signal": True,
        "policy": {
            "max_logit_delta_from_ordinary": max_logit_delta,
            "max_pinned_vs_repicked_logit_delta": max_pinned_vs_repicked_delta,
            "requires_all_decision_reports_present": True,
            "requires_all_decision_reports_passing": True,
            "requires_all_reports_select_temporal_candidate": True,
            "requires_each_report_accepted_temporal_nonzero_alpha": True,
            "allows_default_promotion": False,
            "reason_default_promotion_is_blocked": (
                "This aggregate covers deterministic char-smoke seed evidence only; "
                "default promotion requires broader non-smoke validation."
            ),
        },
        "evidence": {
            "decision_report_paths": [str(path) for path in decision_report_paths],
            "report_count": len(entries),
            "selected_report_count": len(selected_entries),
            "accepted_temporal_report_count": len(accepted_entries),
            "seed_backend_pairs": [
                {"seed": seed, "backend": backend} for seed, backend in seed_backend_pairs
            ],
            "seed_count": len({seed for seed, _backend in seed_backend_pairs}),
            "backend_count": len({backend for _seed, backend in seed_backend_pairs}),
            "min_temporal_loss_improvement_from_alpha0": min(improvements)
            if improvements
            else None,
            "mean_temporal_loss_improvement_from_alpha0": (
                sum(improvements) / len(improvements) if improvements else None
            ),
            "max_temporal_loss_improvement_from_alpha0": max(improvements)
            if improvements
            else None,
            "max_temporal_logit_delta_from_ordinary": max(logit_deltas)
            if logit_deltas
            else None,
            "max_temporal_pinned_vs_repicked_logit_delta": max(pinned_deltas)
            if pinned_deltas
            else None,
            "max_support_change_fraction": max(support_changes)
            if support_changes
            else None,
            "entries": entries,
            "failures": failures,
        },
        "rationale": decision["rationale"],
        "next_step": decision["next_step"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_temporal_aggregate_markdown(out_dir / "decision_report.md", report)
    return report


def write_temporal_clipped_hep_cross_scale_aggregate_report(
    decision_report_paths: list[Path] | tuple[Path, ...] = (
        DEFAULT_TEMPORAL_CLIPPED_CROSS_SCALE_AGGREGATE_REPORTS
    ),
    out_dir: Path = DEFAULT_TEMPORAL_CLIPPED_CROSS_SCALE_AGGREGATE_OUT_DIR,
    *,
    max_logit_delta: float = DEFAULT_MAX_LOGIT_DELTA,
    max_pinned_vs_repicked_delta: float = DEFAULT_MAX_PINNED_VS_REPICKED_DELTA,
) -> dict[str, Any]:
    """Write a cross-scale aggregate report from temporal clipped decisions."""

    report = write_temporal_clipped_hep_aggregate_report(
        decision_report_paths,
        out_dir,
        max_logit_delta=max_logit_delta,
        max_pinned_vs_repicked_delta=max_pinned_vs_repicked_delta,
    )
    evidence = report["evidence"]
    scale_backend_pairs = sorted(
        {
            (entry["scale"], entry["backend"])
            for entry in evidence["entries"]
            if entry.get("scale") is not None and entry.get("backend") is not None
        }
    )
    required_pairs = {
        ("seed_smoke", "local"),
        ("seed_smoke", "colab"),
        ("validation", "local"),
        ("validation", "colab"),
        ("extended", "local"),
        ("extended", "colab"),
    }
    present_pairs = set(scale_backend_pairs)
    missing_pairs = sorted(required_pairs - present_pairs)
    failures = list(evidence["failures"])
    for scale, backend in missing_pairs:
        failures.append(
            {
                "field": "decision_report.scale_backend_pair",
                "expected": f"{scale}/{backend}",
                "actual": "missing",
            }
        )

    decision = _temporal_cross_scale_aggregate_decision(
        evidence["entries"],
        failures,
    )
    evidence["scale_backend_pairs"] = [
        {"scale": scale, "backend": backend} for scale, backend in scale_backend_pairs
    ]
    evidence["scale_count"] = len({scale for scale, _backend in scale_backend_pairs})
    evidence["failures"] = failures
    report["status"] = "pass" if decision["decision"] != INSUFFICIENT_EVIDENCE else "fail"
    report["decision"] = decision["decision"]
    report["selected_label_free_support_stress_candidate"] = decision["selected"]
    report["policy"]["requires_seed_smoke_validation_and_extended_evidence"] = True
    report["policy"]["requires_local_and_colab_evidence_per_scale"] = True
    report["policy"]["reason_default_promotion_is_blocked"] = (
        "This aggregate selects temporal consistency across seed-smoke, validation, "
        "and extended char support-stress evidence; default promotion still needs "
        "a separately defined promotion gate and broader non-char validation."
    )
    report["rationale"] = decision["rationale"]
    report["next_step"] = decision["next_step"]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_temporal_cross_scale_aggregate_markdown(
        out_dir / "decision_report.md",
        report,
    )
    return report


def write_temporal_clipped_hep_promotion_gate_report(
    cross_scale_report_path: Path = DEFAULT_TEMPORAL_CLIPPED_PROMOTION_GATE_CROSS_SCALE_REPORT,
    out_dir: Path = DEFAULT_TEMPORAL_CLIPPED_PROMOTION_GATE_OUT_DIR,
) -> dict[str, Any]:
    """Define the next promotion gate after cross-scale temporal evidence."""

    failures = []
    cross_scale_report: dict[str, Any] | None = None
    if not cross_scale_report_path.is_file():
        failures.append(
            {
                "field": "cross_scale_report",
                "expected": "file exists",
                "actual": "missing",
                "path": str(cross_scale_report_path),
            }
        )
    else:
        cross_scale_report = _read_json_object(cross_scale_report_path)
        if cross_scale_report.get("status") != "pass":
            failures.append(
                {
                    "field": "cross_scale_report.status",
                    "expected": "pass",
                    "actual": cross_scale_report.get("status"),
                    "path": str(cross_scale_report_path),
                }
            )
        if (
            cross_scale_report.get("decision")
            != SELECT_TEMPORAL_CLIPPED_HEP_CROSS_SCALE_AGGREGATE
        ):
            failures.append(
                {
                    "field": "cross_scale_report.decision",
                    "expected": SELECT_TEMPORAL_CLIPPED_HEP_CROSS_SCALE_AGGREGATE,
                    "actual": cross_scale_report.get("decision"),
                    "path": str(cross_scale_report_path),
                }
            )
        if (
            cross_scale_report.get("selected_label_free_support_stress_candidate")
            is not True
        ):
            failures.append(
                {
                    "field": (
                        "cross_scale_report."
                        "selected_label_free_support_stress_candidate"
                    ),
                    "expected": True,
                    "actual": cross_scale_report.get(
                        "selected_label_free_support_stress_candidate"
                    ),
                    "path": str(cross_scale_report_path),
                }
            )

    evidence = (
        cross_scale_report.get("evidence", {})
        if isinstance(cross_scale_report, dict)
        and isinstance(cross_scale_report.get("evidence"), dict)
        else {}
    )
    required_evidence = [
        {
            "gate": "larger_char_local_colab",
            "description": (
                "Run the temporal-vs-entropy-vs-guided clipped support-stress "
                "comparison on a larger char-level setting than the current "
                "extended check."
            ),
            "minimum_scale": {
                "seq_len": 128,
                "hidden_dim": 96,
                "num_columns": 24,
                "pc_steps": 4,
                "training_steps": 50,
            },
            "required_backends": ["local", "colab"],
            "required_runs": [
                "clipped_baseline",
                "prediction_entropy_gradient",
                "temporal_consistency_gradient",
                "supervised_ce_gradient",
            ],
        },
        {
            "gate": "non_char_tokenized_local_colab",
            "description": (
                "Run the same deployable temporal candidate against entropy and "
                "the guided oracle on a non-char tokenized language-model setting."
            ),
            "minimum_scale": {
                "seq_len": 64,
                "hidden_dim": 96,
                "num_columns": 24,
                "pc_steps": 4,
                "training_steps": 50,
            },
            "required_backends": ["local", "colab"],
            "required_runs": [
                "clipped_baseline",
                "prediction_entropy_gradient",
                "temporal_consistency_gradient",
                "supervised_ce_gradient",
            ],
        },
    ]
    acceptance_policy = {
        "requires_cross_scale_aggregate_pass": True,
        "requires_each_gate_local_and_colab": True,
        "requires_passing_artifact_checks": True,
        "requires_temporal_selected_in_each_gate": True,
        "requires_accepted_nonzero_temporal_alpha_in_each_gate": True,
        "requires_temporal_loss_improvement_from_alpha0": True,
        "max_logit_delta_from_ordinary": DEFAULT_MAX_LOGIT_DELTA,
        "max_pinned_vs_repicked_logit_delta": DEFAULT_MAX_PINNED_VS_REPICKED_DELTA,
        "requires_nonzero_support_repick_evidence": True,
        "requires_entropy_and_guided_context_runs": True,
    }
    status = "fail" if failures else "pass"
    report = {
        "status": status,
        "decision": (
            DEFINE_TEMPORAL_CLIPPED_HEP_PROMOTION_GATE
            if status == "pass"
            else INSUFFICIENT_EVIDENCE
        ),
        "selected_label_free_support_stress_candidate": status == "pass",
        "promote_to_default_support_stress_mitigation": False,
        "deployable_label_free_signal": True,
        "policy": {
            **acceptance_policy,
            "allows_default_promotion": False,
            "reason_default_promotion_is_blocked": (
                "This report defines the next promotion gate. It does not satisfy "
                "that gate or change the default support-stress mitigation path."
            ),
        },
        "evidence": {
            "cross_scale_report_path": str(cross_scale_report_path),
            "cross_scale_status": None
            if cross_scale_report is None
            else cross_scale_report.get("status"),
            "cross_scale_decision": None
            if cross_scale_report is None
            else cross_scale_report.get("decision"),
            "cross_scale_report_count": evidence.get("report_count"),
            "cross_scale_scale_count": evidence.get("scale_count"),
            "cross_scale_accepted_temporal_report_count": evidence.get(
                "accepted_temporal_report_count"
            ),
            "cross_scale_mean_temporal_loss_improvement_from_alpha0": evidence.get(
                "mean_temporal_loss_improvement_from_alpha0"
            ),
            "cross_scale_max_temporal_logit_delta_from_ordinary": evidence.get(
                "max_temporal_logit_delta_from_ordinary"
            ),
            "cross_scale_max_temporal_pinned_vs_repicked_logit_delta": evidence.get(
                "max_temporal_pinned_vs_repicked_logit_delta"
            ),
            "cross_scale_max_support_change_fraction": evidence.get(
                "max_support_change_fraction"
            ),
            "required_evidence": required_evidence,
            "failures": failures,
        },
        "rationale": (
            "The current cross-scale char evidence selects temporal consistency, "
            "but default promotion needs evidence outside the completed char "
            "settings. The gate requires one larger char setting and one non-char "
            "tokenized setting, each with local and Colab artifact-backed temporal "
            "decisions against the clipped baseline, entropy probe, and guided oracle."
            if status == "pass"
            else (
                "The promotion gate cannot be defined until the cross-scale temporal "
                "aggregate is present, passing, and selecting temporal consistency."
            )
        ),
        "next_step": (
            "add the non-char tokenized promotion-gate configs and run their local comparison"
            if status == "pass"
            else "repair or regenerate the cross-scale temporal aggregate report"
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_temporal_promotion_gate_markdown(out_dir / "decision_report.md", report)
    return report


def write_temporal_clipped_hep_promotion_gate_satisfaction_report(
    promotion_gate_report_path: Path = DEFAULT_TEMPORAL_CLIPPED_PROMOTION_GATE_SATISFACTION_REPORT,
    decision_report_paths: list[Path] | tuple[Path, ...] = (
        DEFAULT_TEMPORAL_CLIPPED_PROMOTION_GATE_SATISFACTION_REPORTS
    ),
    out_dir: Path = DEFAULT_TEMPORAL_CLIPPED_PROMOTION_GATE_SATISFACTION_OUT_DIR,
    *,
    max_logit_delta: float = DEFAULT_MAX_LOGIT_DELTA,
    max_pinned_vs_repicked_delta: float = DEFAULT_MAX_PINNED_VS_REPICKED_DELTA,
) -> dict[str, Any]:
    """Decide whether the temporal clipped HEP promotion gate is satisfied."""

    if max_logit_delta < 0.0:
        raise ValueError("max_logit_delta must be non-negative")
    if max_pinned_vs_repicked_delta < 0.0:
        raise ValueError("max_pinned_vs_repicked_delta must be non-negative")

    failures = []
    promotion_gate_report: dict[str, Any] | None = None
    if not promotion_gate_report_path.is_file():
        failures.append(
            {
                "field": "promotion_gate_report",
                "expected": "file exists",
                "actual": "missing",
                "path": str(promotion_gate_report_path),
            }
        )
    else:
        promotion_gate_report = _read_json_object(promotion_gate_report_path)
        if promotion_gate_report.get("status") != "pass":
            failures.append(
                {
                    "field": "promotion_gate_report.status",
                    "expected": "pass",
                    "actual": promotion_gate_report.get("status"),
                    "path": str(promotion_gate_report_path),
                }
            )
        if (
            promotion_gate_report.get("decision")
            != DEFINE_TEMPORAL_CLIPPED_HEP_PROMOTION_GATE
        ):
            failures.append(
                {
                    "field": "promotion_gate_report.decision",
                    "expected": DEFINE_TEMPORAL_CLIPPED_HEP_PROMOTION_GATE,
                    "actual": promotion_gate_report.get("decision"),
                    "path": str(promotion_gate_report_path),
                }
            )

    entries = []
    for path in decision_report_paths:
        path = Path(path)
        if not path.is_file():
            failures.append(
                {
                    "field": "decision_report",
                    "expected": "file exists",
                    "actual": "missing",
                    "path": str(path),
                }
            )
            continue
        report = _read_json_object(path)
        entry = _temporal_promotion_gate_entry(
            path,
            report,
            max_logit_delta=max_logit_delta,
            max_pinned_vs_repicked_delta=max_pinned_vs_repicked_delta,
        )
        entries.append(entry)
        failures.extend(entry["failures"])

    gate_backend_pairs = sorted(
        {
            (entry["gate"], entry["backend"])
            for entry in entries
            if entry.get("gate") is not None and entry.get("backend") is not None
        }
    )
    required_pairs = {
        ("larger_char_local_colab", "local"),
        ("larger_char_local_colab", "colab"),
        ("non_char_tokenized_local_colab", "local"),
        ("non_char_tokenized_local_colab", "colab"),
    }
    for gate, backend in sorted(required_pairs - set(gate_backend_pairs)):
        failures.append(
            {
                "field": "decision_report.gate_backend_pair",
                "expected": f"{gate}/{backend}",
                "actual": "missing",
            }
        )

    accepted_entries = [
        entry for entry in entries if entry.get("best_temporal_alpha") is not None
    ]
    improvements = [
        entry["best_temporal_alpha"]["loss_improvement_from_alpha0"]
        for entry in accepted_entries
        if entry["best_temporal_alpha"]["loss_improvement_from_alpha0"] is not None
    ]
    logit_deltas = [
        entry["best_temporal_alpha"]["max_logit_delta_from_ordinary"]
        for entry in accepted_entries
        if entry["best_temporal_alpha"]["max_logit_delta_from_ordinary"] is not None
    ]
    pinned_deltas = [
        entry["best_temporal_alpha"]["pinned_vs_repicked_logit_delta"]
        for entry in accepted_entries
        if entry["best_temporal_alpha"]["pinned_vs_repicked_logit_delta"] is not None
    ]
    support_changes = [
        entry["max_support_change_fraction"]
        for entry in entries
        if entry["max_support_change_fraction"] is not None
    ]
    status = "fail" if failures else "pass"
    report = {
        "status": status,
        "decision": (
            SATISFY_TEMPORAL_CLIPPED_HEP_PROMOTION_GATE
            if status == "pass"
            else INSUFFICIENT_EVIDENCE
        ),
        "selected_label_free_support_stress_candidate": status == "pass",
        "promotion_gate_satisfied": status == "pass",
        "promote_to_default_support_stress_mitigation": status == "pass",
        "deployable_label_free_signal": True,
        "policy": {
            "max_logit_delta_from_ordinary": max_logit_delta,
            "max_pinned_vs_repicked_logit_delta": max_pinned_vs_repicked_delta,
            "requires_promotion_gate_report_pass": True,
            "requires_larger_char_local_and_colab": True,
            "requires_non_char_tokenized_local_and_colab": True,
            "requires_all_decision_reports_passing": True,
            "requires_all_reports_select_temporal_candidate": True,
            "requires_each_report_accepted_temporal_nonzero_alpha": True,
            "requires_passing_artifact_checks": True,
            "requires_nonzero_support_repick_evidence": True,
            "allows_default_promotion": True,
        },
        "evidence": {
            "promotion_gate_report_path": str(promotion_gate_report_path),
            "promotion_gate_status": None
            if promotion_gate_report is None
            else promotion_gate_report.get("status"),
            "promotion_gate_decision": None
            if promotion_gate_report is None
            else promotion_gate_report.get("decision"),
            "decision_report_paths": [str(path) for path in decision_report_paths],
            "report_count": len(entries),
            "accepted_temporal_report_count": len(accepted_entries),
            "gate_backend_pairs": [
                {"gate": gate, "backend": backend}
                for gate, backend in gate_backend_pairs
            ],
            "gate_count": len({gate for gate, _backend in gate_backend_pairs}),
            "backend_count": len({backend for _gate, backend in gate_backend_pairs}),
            "min_temporal_loss_improvement_from_alpha0": min(improvements)
            if improvements
            else None,
            "mean_temporal_loss_improvement_from_alpha0": (
                sum(improvements) / len(improvements) if improvements else None
            ),
            "max_temporal_loss_improvement_from_alpha0": max(improvements)
            if improvements
            else None,
            "max_temporal_logit_delta_from_ordinary": max(logit_deltas)
            if logit_deltas
            else None,
            "max_temporal_pinned_vs_repicked_logit_delta": max(pinned_deltas)
            if pinned_deltas
            else None,
            "max_support_change_fraction": max(support_changes)
            if support_changes
            else None,
            "entries": entries,
            "failures": failures,
        },
        "rationale": (
            "The defined promotion gate is satisfied: larger-char and non-char "
            "tokenized local/Colab reports all pass, select temporal consistency, "
            "show nonzero support repicking, and include accepted nonzero temporal "
            "alphas inside both stability budgets."
            if status == "pass"
            else (
                "The promotion gate is not satisfied until the gate definition and "
                "all larger-char and non-char tokenized local/Colab decisions pass "
                "with accepted temporal alphas and nonzero support repicking."
            )
        ),
        "next_step": (
            "make the explicit default support-stress mitigation change to temporal clipped HEP"
            if status == "pass"
            else "repair or regenerate the missing or failing promotion-gate evidence reports"
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_temporal_promotion_gate_satisfaction_markdown(
        out_dir / "decision_report.md",
        report,
    )
    return report


def write_post_promotion_residual_learning_gate_report(
    promotion_satisfaction_report_path: Path = DEFAULT_POST_PROMOTION_GATE_SATISFACTION_REPORT,
    default_support_stress_config_path: Path = DEFAULT_POST_PROMOTION_GATE_CONFIG,
    out_dir: Path = DEFAULT_POST_PROMOTION_GATE_OUT_DIR,
) -> dict[str, Any]:
    """Define the next residual-layer learning gate after temporal HEP promotion."""

    failures = []
    promotion_satisfaction_report: dict[str, Any] | None = None
    if not promotion_satisfaction_report_path.is_file():
        failures.append(
            {
                "field": "promotion_satisfaction_report",
                "expected": "file exists",
                "actual": "missing",
                "path": str(promotion_satisfaction_report_path),
            }
        )
    else:
        promotion_satisfaction_report = _read_json_object(
            promotion_satisfaction_report_path
        )
        if promotion_satisfaction_report.get("status") != "pass":
            failures.append(
                {
                    "field": "promotion_satisfaction_report.status",
                    "expected": "pass",
                    "actual": promotion_satisfaction_report.get("status"),
                    "path": str(promotion_satisfaction_report_path),
                }
            )
        if (
            promotion_satisfaction_report.get("decision")
            != SATISFY_TEMPORAL_CLIPPED_HEP_PROMOTION_GATE
        ):
            failures.append(
                {
                    "field": "promotion_satisfaction_report.decision",
                    "expected": SATISFY_TEMPORAL_CLIPPED_HEP_PROMOTION_GATE,
                    "actual": promotion_satisfaction_report.get("decision"),
                    "path": str(promotion_satisfaction_report_path),
                }
            )
        if promotion_satisfaction_report.get("promotion_gate_satisfied") is not True:
            failures.append(
                {
                    "field": "promotion_satisfaction_report.promotion_gate_satisfied",
                    "expected": True,
                    "actual": promotion_satisfaction_report.get(
                        "promotion_gate_satisfied"
                    ),
                    "path": str(promotion_satisfaction_report_path),
                }
            )
        if (
            promotion_satisfaction_report.get(
                "promote_to_default_support_stress_mitigation"
            )
            is not True
        ):
            failures.append(
                {
                    "field": (
                        "promotion_satisfaction_report."
                        "promote_to_default_support_stress_mitigation"
                    ),
                    "expected": True,
                    "actual": promotion_satisfaction_report.get(
                        "promote_to_default_support_stress_mitigation"
                    ),
                    "path": str(promotion_satisfaction_report_path),
                }
            )

    default_config: dict[str, Any] | None = None
    if not default_support_stress_config_path.is_file():
        failures.append(
            {
                "field": "default_support_stress_config",
                "expected": "file exists",
                "actual": "missing",
                "path": str(default_support_stress_config_path),
            }
        )
    else:
        default_config = _read_config_object(default_support_stress_config_path)
        inference = (
            default_config.get("inference", {})
            if isinstance(default_config.get("inference"), dict)
            else {}
        )
        if inference.get("hep_settling_objective") != "temporal_consistency_gradient":
            failures.append(
                {
                    "field": "default_support_stress_config.hep_settling_objective",
                    "expected": "temporal_consistency_gradient",
                    "actual": inference.get("hep_settling_objective"),
                    "path": str(default_support_stress_config_path),
                }
            )
        if inference.get("hep_update_clip_norm") != 0.01:
            failures.append(
                {
                    "field": "default_support_stress_config.hep_update_clip_norm",
                    "expected": 0.01,
                    "actual": inference.get("hep_update_clip_norm"),
                    "path": str(default_support_stress_config_path),
                }
            )

    required_evidence = [
        {
            "gate": "pc_residual_objective_under_promoted_temporal_default",
            "description": (
                "Compare supervised residual training against PC-style residual "
                "training while keeping the promoted temporal clipped "
                "support-stress default active."
            ),
            "required_backends": ["local", "colab"],
            "required_runs": [
                "supervised_ce_temporal_clipped_support_stress",
                "pc_logit_mse_temporal_clipped_support_stress",
            ],
            "minimum_scale": {
                "dataset": "tiny_shakespeare_char",
                "seq_len": 64,
                "hidden_dim": 64,
                "num_columns": 12,
                "pc_steps": 3,
                "training_steps": 25,
            },
        },
        {
            "gate": "frozen_base_and_zero_identity_regression",
            "description": (
                "Require both residual objectives to preserve the frozen-base and "
                "zero-initialized residual invariants under the promoted default."
            ),
            "required_invariants": [
                "zero_init_identity",
                "frozen_base_unchanged",
                "hep_alpha_0_equivalence",
                "residual_parameters_updated",
            ],
        },
        {
            "gate": "artifact_backed_residual_learning_decision",
            "description": (
                "Write a command-driven decision report from completed local and "
                "Colab artifacts before changing the default residual objective."
            ),
            "required_artifacts": [
                "summary.json",
                "metrics.csv",
                "notes.md",
                "artifact_check.json",
                "decision_report.json",
            ],
        },
    ]
    status = "fail" if failures else "pass"
    report = {
        "status": status,
        "decision": (
            DEFINE_POST_PROMOTION_RESIDUAL_LEARNING_GATE
            if status == "pass"
            else INSUFFICIENT_EVIDENCE
        ),
        "promoted_temporal_support_stress_default_confirmed": status == "pass",
        "promote_residual_learning_method": False,
        "policy": {
            "requires_promotion_gate_satisfaction_pass": True,
            "requires_default_support_stress_temporal_consistency": True,
            "requires_default_support_stress_clip_norm": 0.01,
            "requires_local_and_colab_evidence": True,
            "requires_supervised_and_pc_residual_objective_runs": True,
            "requires_passing_artifact_checks": True,
            "allows_residual_objective_promotion": False,
            "reason_residual_objective_promotion_is_blocked": (
                "This report defines the next gate after temporal clipped HEP "
                "promotion. It does not execute the residual-objective comparison "
                "or change the default residual training objective."
            ),
        },
        "evidence": {
            "promotion_satisfaction_report_path": str(
                promotion_satisfaction_report_path
            ),
            "promotion_satisfaction_status": None
            if promotion_satisfaction_report is None
            else promotion_satisfaction_report.get("status"),
            "promotion_satisfaction_decision": None
            if promotion_satisfaction_report is None
            else promotion_satisfaction_report.get("decision"),
            "promotion_gate_satisfied": None
            if promotion_satisfaction_report is None
            else promotion_satisfaction_report.get("promotion_gate_satisfied"),
            "default_support_stress_config_path": str(
                default_support_stress_config_path
            ),
            "default_support_stress_experiment_id": (
                ((default_config or {}).get("run") or {}).get("experiment_id")
                if isinstance((default_config or {}).get("run"), dict)
                else None
            ),
            "default_support_stress_settling_objective": (
                ((default_config or {}).get("inference") or {}).get(
                    "hep_settling_objective"
                )
                if isinstance((default_config or {}).get("inference"), dict)
                else None
            ),
            "default_support_stress_clip_norm": (
                ((default_config or {}).get("inference") or {}).get(
                    "hep_update_clip_norm"
                )
                if isinstance((default_config or {}).get("inference"), dict)
                else None
            ),
            "required_evidence": required_evidence,
            "failures": failures,
        },
        "rationale": (
            "Temporal clipped HEP has passed its promotion gate and is present in "
            "the default support-stress config. The next bounded research gate "
            "should return to residual-layer learning itself: compare supervised "
            "residual updates against the existing PC-style objective under the "
            "promoted temporal clipped support-stress default, with local and "
            "Colab artifact-backed decisions before any residual objective change."
            if status == "pass"
            else (
                "The post-promotion residual-layer learning gate cannot be defined "
                "until temporal clipped HEP promotion evidence is passing and the "
                "default support-stress config contains the promoted temporal "
                "clipped settling path."
            )
        ),
        "next_step": (
            "add PC residual-objective support-stress temporal-clipped validation configs and run the local comparison"
            if status == "pass"
            else "repair the temporal promotion evidence or default support-stress config"
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_post_promotion_residual_learning_gate_markdown(
        out_dir / "decision_report.md",
        report,
    )
    return report


def write_residual_objective_gate_decision_report(
    comparison_dirs: list[Path] | tuple[Path, ...] = (
        DEFAULT_RESIDUAL_OBJECTIVE_GATE_COMPARISON_DIRS
    ),
    out_dir: Path = DEFAULT_RESIDUAL_OBJECTIVE_GATE_OUT_DIR,
    *,
    artifact_check_paths: list[Path] | tuple[Path, ...] | None = (
        DEFAULT_RESIDUAL_OBJECTIVE_GATE_ARTIFACT_CHECKS
    ),
    max_logit_delta: float = DEFAULT_MAX_LOGIT_DELTA,
) -> dict[str, Any]:
    """Decide the supervised-vs-PC residual objective gate from artifacts."""

    if max_logit_delta < 0.0:
        raise ValueError("max_logit_delta must be non-negative")

    entries = []
    failures = []
    artifact_paths = list(artifact_check_paths or [])
    for index, comparison_dir in enumerate(comparison_dirs):
        comparison_dir = Path(comparison_dir)
        artifact_check_path = (
            Path(artifact_paths[index]) if index < len(artifact_paths) else None
        )
        entry = _residual_objective_gate_entry(
            comparison_dir,
            artifact_check_path=artifact_check_path,
            max_logit_delta=max_logit_delta,
        )
        entries.append(entry)
        failures.extend(entry["failures"])

    backend_pairs = sorted(
        {
            entry["backend"]
            for entry in entries
            if entry.get("backend") in {"local", "colab"}
        }
    )
    for backend in sorted({"local", "colab"} - set(backend_pairs)):
        failures.append(
            {
                "field": "comparison.backend",
                "expected": backend,
                "actual": "missing",
            }
        )

    supervised_entries = [
        entry["supervised_run"] for entry in entries if entry.get("supervised_run")
    ]
    pc_entries = [entry["pc_run"] for entry in entries if entry.get("pc_run")]
    supervised_best_losses = [
        run["best_hep_loss"] for run in supervised_entries if run["best_hep_loss"] is not None
    ]
    pc_best_losses = [
        run["best_hep_loss"] for run in pc_entries if run["best_hep_loss"] is not None
    ]
    supervised_deltas = [
        run["residual_loss_delta"]
        for run in supervised_entries
        if run["residual_loss_delta"] is not None
    ]
    pc_deltas = [
        run["residual_loss_delta"]
        for run in pc_entries
        if run["residual_loss_delta"] is not None
    ]
    pc_ce_wins = [
        entry
        for entry in entries
        if entry.get("supervised_run")
        and entry.get("pc_run")
        and entry["pc_run"]["best_hep_loss"] is not None
        and entry["supervised_run"]["best_hep_loss"] is not None
        and entry["pc_run"]["best_hep_loss"] < entry["supervised_run"]["best_hep_loss"]
    ]
    decision = _residual_objective_gate_decision(entries, failures, pc_ce_wins)
    report = {
        "status": "pass" if decision["decision"] != INSUFFICIENT_EVIDENCE else "fail",
        "decision": decision["decision"],
        "continue_pc_residual_objective_validation": decision["continue_pc"],
        "promote_residual_learning_method": False,
        "default_residual_objective": "supervised_ce",
        "policy": {
            "max_logit_delta_from_ordinary": max_logit_delta,
            "requires_local_and_colab_evidence": True,
            "requires_passing_artifact_checks": True,
            "requires_supervised_and_pc_runs": True,
            "requires_support_stress_preset_disabled": True,
            "requires_temporal_clipped_hep_path": True,
            "requires_both_objectives_improve_own_training_loss": True,
            "requires_pc_lower_supervised_ce_hep_loss_for_pc_promotion": True,
            "allows_residual_objective_promotion": False,
            "reason_residual_objective_promotion_is_blocked": (
                "This report decides whether the current objective-gate evidence "
                "justifies continued PC validation. It does not change the default "
                "residual training objective."
            ),
        },
        "evidence": {
            "comparison_dirs": [str(path) for path in comparison_dirs],
            "artifact_check_paths": [str(path) for path in artifact_paths],
            "backend_count": len(backend_pairs),
            "backends": backend_pairs,
            "comparison_count": len(entries),
            "supervised_run_count": len(supervised_entries),
            "pc_run_count": len(pc_entries),
            "pc_ce_win_count": len(pc_ce_wins),
            "min_supervised_residual_loss_delta": min(supervised_deltas)
            if supervised_deltas
            else None,
            "min_pc_residual_loss_delta": min(pc_deltas) if pc_deltas else None,
            "mean_supervised_best_hep_loss": (
                sum(supervised_best_losses) / len(supervised_best_losses)
                if supervised_best_losses
                else None
            ),
            "mean_pc_best_hep_loss": (
                sum(pc_best_losses) / len(pc_best_losses) if pc_best_losses else None
            ),
            "entries": entries,
            "failures": failures,
        },
        "rationale": decision["rationale"],
        "next_step": decision["next_step"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_residual_objective_gate_markdown(out_dir / "decision_report.md", report)
    return report


def write_pc_residual_objective_diagnostics_report(
    comparison_dirs: list[Path] | tuple[Path, ...] = (
        DEFAULT_RESIDUAL_OBJECTIVE_GATE_COMPARISON_DIRS
    ),
    out_dir: Path = DEFAULT_PC_RESIDUAL_OBJECTIVE_DIAGNOSTICS_OUT_DIR,
    *,
    artifact_check_paths: list[Path] | tuple[Path, ...] | None = (
        DEFAULT_RESIDUAL_OBJECTIVE_GATE_ARTIFACT_CHECKS
    ),
    max_logit_delta: float = DEFAULT_MAX_LOGIT_DELTA,
) -> dict[str, Any]:
    """Write artifact-only diagnostics for the supervised-vs-PC objective gap."""

    if max_logit_delta < 0.0:
        raise ValueError("max_logit_delta must be non-negative")

    entries = []
    failures = []
    artifact_paths = list(artifact_check_paths or [])
    for index, comparison_dir in enumerate(comparison_dirs):
        comparison_dir = Path(comparison_dir)
        artifact_check_path = (
            Path(artifact_paths[index]) if index < len(artifact_paths) else None
        )
        entry = _residual_objective_gate_entry(
            comparison_dir,
            artifact_check_path=artifact_check_path,
            max_logit_delta=max_logit_delta,
        )
        _add_pc_diagnostics(entry)
        entries.append(entry)
        failures.extend(entry["failures"])

    backend_pairs = sorted(
        {
            entry["backend"]
            for entry in entries
            if entry.get("backend") in {"local", "colab"}
        }
    )
    for backend in sorted({"local", "colab"} - set(backend_pairs)):
        failures.append(
            {
                "field": "comparison.backend",
                "expected": backend,
                "actual": "missing",
            }
        )

    gaps = [
        entry["pc_minus_supervised_best_hep_loss"]
        for entry in entries
        if entry.get("pc_minus_supervised_best_hep_loss") is not None
    ]
    pc_own_ratios = [
        entry["pc_run"]["residual_loss_ratio"]
        for entry in entries
        if entry.get("pc_run") and entry["pc_run"]["residual_loss_ratio"] is not None
    ]
    supervised_own_ratios = [
        entry["supervised_run"]["residual_loss_ratio"]
        for entry in entries
        if entry.get("supervised_run")
        and entry["supervised_run"]["residual_loss_ratio"] is not None
    ]
    pc_hep_improvements = [
        entry["pc_best_hep_loss_improvement_from_alpha0"]
        for entry in entries
        if entry.get("pc_best_hep_loss_improvement_from_alpha0") is not None
    ]
    supervised_hep_improvements = [
        entry["supervised_best_hep_loss_improvement_from_alpha0"]
        for entry in entries
        if entry.get("supervised_best_hep_loss_improvement_from_alpha0") is not None
    ]
    pc_worse_count = len([gap for gap in gaps if gap > 0.0])
    status = "fail" if failures else "pass"
    report = {
        "status": status,
        "decision": DIAGNOSE_PC_RESIDUAL_OBJECTIVE
        if status == "pass"
        else INSUFFICIENT_EVIDENCE,
        "promote_residual_learning_method": False,
        "default_residual_objective": "supervised_ce",
        "policy": {
            "max_logit_delta_from_ordinary": max_logit_delta,
            "requires_local_and_colab_evidence": True,
            "requires_passing_artifact_checks": True,
            "requires_valid_residual_objective_gate_inputs": True,
            "diagnostic_only": True,
        },
        "evidence": {
            "comparison_dirs": [str(path) for path in comparison_dirs],
            "artifact_check_paths": [str(path) for path in artifact_paths],
            "backend_count": len(backend_pairs),
            "backends": backend_pairs,
            "comparison_count": len(entries),
            "pc_worse_ce_backend_count": pc_worse_count,
            "mean_pc_minus_supervised_best_hep_loss": (
                sum(gaps) / len(gaps) if gaps else None
            ),
            "min_pc_minus_supervised_best_hep_loss": min(gaps) if gaps else None,
            "max_pc_minus_supervised_best_hep_loss": max(gaps) if gaps else None,
            "mean_supervised_residual_loss_ratio": (
                sum(supervised_own_ratios) / len(supervised_own_ratios)
                if supervised_own_ratios
                else None
            ),
            "mean_pc_residual_loss_ratio": (
                sum(pc_own_ratios) / len(pc_own_ratios) if pc_own_ratios else None
            ),
            "mean_supervised_best_hep_loss_improvement_from_alpha0": (
                sum(supervised_hep_improvements) / len(supervised_hep_improvements)
                if supervised_hep_improvements
                else None
            ),
            "mean_pc_best_hep_loss_improvement_from_alpha0": (
                sum(pc_hep_improvements) / len(pc_hep_improvements)
                if pc_hep_improvements
                else None
            ),
            "entries": entries,
            "failures": failures,
        },
        "rationale": (
            "The artifact-backed objective-gate inputs are valid. PC improves its "
            "own logit-MSE objective, but its best supervised CE HEP loss remains "
            "higher than supervised residual training across the checked backends; "
            "the gap is objective alignment rather than a failed HEP alpha sweep."
            if status == "pass" and pc_worse_count == len(gaps)
            else (
                "The diagnostics are valid, but the PC-vs-supervised CE gap is not "
                "uniform across all checked backends."
                if status == "pass"
                else (
                    "The diagnostics require the same valid local and Colab "
                    "objective-gate artifacts as the residual-objective decision."
                )
            )
        ),
        "next_step": (
            "test a PC residual objective variant with an explicit supervised CE anchor or report why PC validation should stop"
            if status == "pass"
            else "repair or regenerate the objective-gate comparison artifacts"
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_pc_residual_objective_diagnostics_markdown(
        out_dir / "decision_report.md",
        report,
    )
    return report


def write_anchored_pc_residual_objective_decision_report(
    comparison_dirs: list[Path] | tuple[Path, ...] = (
        DEFAULT_ANCHORED_PC_RESIDUAL_OBJECTIVE_COMPARISON_DIRS
    ),
    out_dir: Path = DEFAULT_ANCHORED_PC_RESIDUAL_OBJECTIVE_OUT_DIR,
    *,
    artifact_check_paths: list[Path] | tuple[Path, ...] | None = (
        DEFAULT_ANCHORED_PC_RESIDUAL_OBJECTIVE_ARTIFACT_CHECKS
    ),
    max_logit_delta: float = DEFAULT_MAX_LOGIT_DELTA,
) -> dict[str, Any]:
    """Decide whether the CE-anchored PC objective merits more validation."""

    if max_logit_delta < 0.0:
        raise ValueError("max_logit_delta must be non-negative")

    entries = []
    failures = []
    artifact_paths = list(artifact_check_paths or [])
    for index, comparison_dir in enumerate(comparison_dirs):
        comparison_dir = Path(comparison_dir)
        artifact_check_path = (
            Path(artifact_paths[index]) if index < len(artifact_paths) else None
        )
        entry = _anchored_pc_residual_objective_entry(
            comparison_dir,
            artifact_check_path=artifact_check_path,
            max_logit_delta=max_logit_delta,
        )
        entries.append(entry)
        failures.extend(entry["failures"])

    backends = sorted(
        {
            entry["backend"]
            for entry in entries
            if entry.get("backend") in {"local", "colab"}
        }
    )
    for backend in sorted({"local", "colab"} - set(backends)):
        failures.append(
            {
                "field": "comparison.backend",
                "expected": backend,
                "actual": "missing",
            }
        )

    supervised_runs = [
        entry["supervised_run"] for entry in entries if entry.get("supervised_run")
    ]
    pc_runs = [entry["pc_run"] for entry in entries if entry.get("pc_run")]
    anchored_runs = [
        entry["anchored_pc_run"] for entry in entries if entry.get("anchored_pc_run")
    ]
    anchored_ce_wins = [
        entry
        for entry in entries
        if entry.get("supervised_run")
        and entry.get("anchored_pc_run")
        and entry["anchored_pc_run"]["best_hep_loss"] is not None
        and entry["supervised_run"]["best_hep_loss"] is not None
        and entry["anchored_pc_run"]["best_hep_loss"]
        < entry["supervised_run"]["best_hep_loss"]
    ]
    anchored_minus_supervised = [
        entry["anchored_pc_minus_supervised_best_hep_loss"]
        for entry in entries
        if entry.get("anchored_pc_minus_supervised_best_hep_loss") is not None
    ]
    pc_minus_supervised = [
        entry["pc_minus_supervised_best_hep_loss"]
        for entry in entries
        if entry.get("pc_minus_supervised_best_hep_loss") is not None
    ]
    anchor_gap_reductions = [
        entry["pc_to_anchored_gap_reduction"]
        for entry in entries
        if entry.get("pc_to_anchored_gap_reduction") is not None
    ]
    status = "fail" if failures else "pass"
    continue_pc = status == "pass" and bool(anchored_ce_wins)
    report = {
        "status": status,
        "decision": (
            CONTINUE_PC_RESIDUAL_OBJECTIVE_VALIDATION
            if continue_pc
            else (
                STOP_PC_RESIDUAL_OBJECTIVE_VALIDATION
                if status == "pass"
                else INSUFFICIENT_EVIDENCE
            )
        ),
        "continue_pc_residual_objective_validation": continue_pc,
        "selected_pc_residual_objective_variant": (
            "pc_logit_mse_ce_anchor" if continue_pc else None
        ),
        "promote_residual_learning_method": False,
        "default_residual_objective": "supervised_ce",
        "policy": {
            "max_logit_delta_from_ordinary": max_logit_delta,
            "requires_local_and_colab_evidence": True,
            "requires_passing_artifact_checks": True,
            "requires_supervised_unanchored_and_anchored_pc_runs": True,
            "requires_support_stress_preset_disabled": True,
            "requires_temporal_clipped_hep_path": True,
            "requires_all_objectives_improve_own_training_loss": True,
            "requires_anchored_pc_lower_supervised_ce_hep_loss_to_continue_pc": True,
            "allows_residual_objective_promotion": False,
            "diagnostic_decision_only": True,
        },
        "evidence": {
            "comparison_dirs": [str(path) for path in comparison_dirs],
            "artifact_check_paths": [str(path) for path in artifact_paths],
            "backend_count": len(backends),
            "backends": backends,
            "comparison_count": len(entries),
            "supervised_run_count": len(supervised_runs),
            "pc_run_count": len(pc_runs),
            "anchored_pc_run_count": len(anchored_runs),
            "anchored_pc_ce_win_count": len(anchored_ce_wins),
            "mean_pc_minus_supervised_best_hep_loss": (
                sum(pc_minus_supervised) / len(pc_minus_supervised)
                if pc_minus_supervised
                else None
            ),
            "mean_anchored_pc_minus_supervised_best_hep_loss": (
                sum(anchored_minus_supervised) / len(anchored_minus_supervised)
                if anchored_minus_supervised
                else None
            ),
            "mean_pc_to_anchored_gap_reduction": (
                sum(anchor_gap_reductions) / len(anchor_gap_reductions)
                if anchor_gap_reductions
                else None
            ),
            "entries": entries,
            "failures": failures,
        },
        "rationale": (
            "The CE-anchored PC objective closes much of the unanchored PC "
            "supervised-CE HEP loss gap, but it still does not beat supervised CE "
            "residual training in the checked local and Colab artifacts. PC "
            "objective validation should stop under the current gate."
            if status == "pass" and not continue_pc
            else (
                "The CE-anchored PC objective beats supervised CE HEP loss in at "
                "least one artifact-backed backend, so it merits a broader "
                "PC-objective validation before any default change."
                if status == "pass"
                else (
                    "The anchored-PC decision requires matching local and Colab "
                    "comparisons with passing artifact checks and valid supervised, "
                    "unanchored PC, and anchored PC temporal-clipped runs."
                )
            )
        ),
        "next_step": (
            "stop PC residual-objective validation under the current gate and select a non-PC residual-learning variant to test next"
            if status == "pass" and not continue_pc
            else (
                "run a broader anchored-PC objective comparison outside the current char validation setting"
                if status == "pass"
                else "repair or regenerate the anchored-PC objective comparison artifacts"
            )
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_anchored_pc_residual_objective_markdown(
        out_dir / "decision_report.md",
        report,
    )
    return report


def _residual_objective_gate_entry(
    comparison_dir: Path,
    *,
    artifact_check_path: Path | None,
    max_logit_delta: float,
) -> dict[str, Any]:
    failures = []
    comparison: dict[str, Any] | None = None
    artifact_check: dict[str, Any] | None = None
    if not (comparison_dir / "summary.json").is_file():
        failures.append(
            {
                "field": "comparison.summary.json",
                "expected": "file exists",
                "actual": "missing",
                "path": str(comparison_dir / "summary.json"),
            }
        )
    else:
        comparison = _read_json_object(comparison_dir / "summary.json")
    if artifact_check_path is not None and artifact_check_path.is_file():
        artifact_check = _read_json_object(artifact_check_path)
    elif comparison is not None:
        artifact_check = check_comparison_artifacts(comparison_dir)

    runs = (
        comparison.get("runs", [])
        if isinstance(comparison, dict) and isinstance(comparison.get("runs"), list)
        else []
    )
    supervised_runs = _runs_with_residual_objective(runs, "supervised_ce")
    pc_runs = _runs_with_residual_objective(runs, "pc_logit_mse")
    supervised_run = (
        _residual_objective_run_entry(supervised_runs[0], max_logit_delta=max_logit_delta)
        if supervised_runs
        else None
    )
    pc_run = (
        _residual_objective_run_entry(pc_runs[0], max_logit_delta=max_logit_delta)
        if pc_runs
        else None
    )
    verdict = comparison.get("verdict") if isinstance(comparison, dict) else None
    verdict = verdict if isinstance(verdict, dict) else {}
    backend = _infer_backend_from_report(comparison_dir, comparison_dir)

    if artifact_check is None or artifact_check.get("status") != "pass":
        failures.append(
            {
                "field": "artifact_check.status",
                "expected": "pass",
                "actual": None if artifact_check is None else artifact_check.get("status"),
                "path": str(artifact_check_path or comparison_dir),
            }
        )
    if comparison is None or comparison.get("status") != "ok":
        failures.append(
            {
                "field": "comparison.status",
                "expected": "ok",
                "actual": None if comparison is None else comparison.get("status"),
                "path": str(comparison_dir),
            }
        )
    if verdict.get("status") != "pass":
        failures.append(
            {
                "field": "comparison.verdict.status",
                "expected": "pass",
                "actual": verdict.get("status"),
                "path": str(comparison_dir),
            }
        )
    if not supervised_runs:
        failures.append(
            {
                "field": "comparison.runs.supervised_ce",
                "expected": "one run",
                "actual": 0,
                "path": str(comparison_dir),
            }
        )
    if not pc_runs:
        failures.append(
            {
                "field": "comparison.runs.pc_logit_mse",
                "expected": "one run",
                "actual": 0,
                "path": str(comparison_dir),
            }
        )

    for run_entry in (supervised_run, pc_run):
        if run_entry is None:
            continue
        prefix = f"run.{run_entry['experiment_id']}"
        if run_entry["status"] != "ok":
            failures.append(
                {
                    "field": f"{prefix}.status",
                    "expected": "ok",
                    "actual": run_entry["status"],
                    "path": str(comparison_dir),
                }
            )
        if run_entry["support_stress_preset"] is not False:
            failures.append(
                {
                    "field": f"{prefix}.support_stress_preset",
                    "expected": False,
                    "actual": run_entry["support_stress_preset"],
                    "path": str(comparison_dir),
                }
            )
        if run_entry["hep_settling_objective"] != "temporal_consistency_gradient":
            failures.append(
                {
                    "field": f"{prefix}.hep_settling_objective",
                    "expected": "temporal_consistency_gradient",
                    "actual": run_entry["hep_settling_objective"],
                    "path": str(comparison_dir),
                }
            )
        if run_entry["hep_update_clip_norm"] != 0.01:
            failures.append(
                {
                    "field": f"{prefix}.hep_update_clip_norm",
                    "expected": 0.01,
                    "actual": run_entry["hep_update_clip_norm"],
                    "path": str(comparison_dir),
                }
            )
        if not run_entry["own_loss_improved"]:
            failures.append(
                {
                    "field": f"{prefix}.residual_loss_delta",
                    "expected": "< 0.0",
                    "actual": run_entry["residual_loss_delta"],
                    "path": str(comparison_dir),
                }
            )
        if not run_entry["accepted_hep_alphas"]:
            failures.append(
                {
                    "field": f"{prefix}.hep_alpha_sweep",
                    "expected": "accepted nonzero alpha",
                    "actual": "none",
                    "path": str(comparison_dir),
                }
            )
        for invariant, passed in run_entry["invariants"].items():
            if passed is not True:
                failures.append(
                    {
                        "field": f"{prefix}.invariants.{invariant}",
                        "expected": True,
                        "actual": passed,
                        "path": str(comparison_dir),
                    }
                )

    return {
        "comparison_dir": str(comparison_dir),
        "artifact_check_path": str(artifact_check_path) if artifact_check_path else None,
        "backend": backend,
        "artifact_check_status": None if artifact_check is None else artifact_check.get("status"),
        "comparison_status": None if comparison is None else comparison.get("status"),
        "verdict_status": verdict.get("status"),
        "supervised_run": supervised_run,
        "pc_run": pc_run,
        "failures": failures,
    }


def _anchored_pc_residual_objective_entry(
    comparison_dir: Path,
    *,
    artifact_check_path: Path | None,
    max_logit_delta: float,
) -> dict[str, Any]:
    entry = _residual_objective_gate_entry(
        comparison_dir,
        artifact_check_path=artifact_check_path,
        max_logit_delta=max_logit_delta,
    )
    comparison = (
        _read_json_object(comparison_dir / "summary.json")
        if (comparison_dir / "summary.json").is_file()
        else None
    )
    runs = (
        comparison.get("runs", [])
        if isinstance(comparison, dict) and isinstance(comparison.get("runs"), list)
        else []
    )
    anchored_runs = _runs_with_residual_objective(runs, "pc_logit_mse_ce_anchor")
    anchored_run = (
        _residual_objective_run_entry(
            anchored_runs[0],
            max_logit_delta=max_logit_delta,
        )
        if anchored_runs
        else None
    )
    entry["anchored_pc_run"] = anchored_run
    if not anchored_runs:
        entry["failures"].append(
            {
                "field": "comparison.runs.pc_logit_mse_ce_anchor",
                "expected": "one run",
                "actual": 0,
                "path": str(comparison_dir),
            }
        )
    elif anchored_run is not None:
        _append_residual_objective_run_failures(
            entry["failures"],
            comparison_dir,
            anchored_run,
        )

    supervised = entry.get("supervised_run")
    pc = entry.get("pc_run")
    entry["pc_minus_supervised_best_hep_loss"] = _best_loss_delta(pc, supervised)
    entry["anchored_pc_minus_supervised_best_hep_loss"] = _best_loss_delta(
        anchored_run,
        supervised,
    )
    entry["anchored_pc_minus_unanchored_pc_best_hep_loss"] = _best_loss_delta(
        anchored_run,
        pc,
    )
    entry["pc_to_anchored_gap_reduction"] = (
        None
        if entry["pc_minus_supervised_best_hep_loss"] is None
        or entry["anchored_pc_minus_supervised_best_hep_loss"] is None
        else (
            entry["pc_minus_supervised_best_hep_loss"]
            - entry["anchored_pc_minus_supervised_best_hep_loss"]
        )
    )
    return entry


def _append_residual_objective_run_failures(
    failures: list[dict[str, Any]],
    comparison_dir: Path,
    run_entry: dict[str, Any],
) -> None:
    prefix = f"run.{run_entry['experiment_id']}"
    if run_entry["status"] != "ok":
        failures.append(
            {
                "field": f"{prefix}.status",
                "expected": "ok",
                "actual": run_entry["status"],
                "path": str(comparison_dir),
            }
        )
    if run_entry["support_stress_preset"] is not False:
        failures.append(
            {
                "field": f"{prefix}.support_stress_preset",
                "expected": False,
                "actual": run_entry["support_stress_preset"],
                "path": str(comparison_dir),
            }
        )
    if run_entry["hep_settling_objective"] != "temporal_consistency_gradient":
        failures.append(
            {
                "field": f"{prefix}.hep_settling_objective",
                "expected": "temporal_consistency_gradient",
                "actual": run_entry["hep_settling_objective"],
                "path": str(comparison_dir),
            }
        )
    if run_entry["hep_update_clip_norm"] != 0.01:
        failures.append(
            {
                "field": f"{prefix}.hep_update_clip_norm",
                "expected": 0.01,
                "actual": run_entry["hep_update_clip_norm"],
                "path": str(comparison_dir),
            }
        )
    if not run_entry["own_loss_improved"]:
        failures.append(
            {
                "field": f"{prefix}.residual_loss_delta",
                "expected": "< 0.0",
                "actual": run_entry["residual_loss_delta"],
                "path": str(comparison_dir),
            }
        )
    if not run_entry["accepted_hep_alphas"]:
        failures.append(
            {
                "field": f"{prefix}.hep_alpha_sweep",
                "expected": "accepted nonzero alpha",
                "actual": "none",
                "path": str(comparison_dir),
            }
        )
    for invariant, passed in run_entry["invariants"].items():
        if passed is not True:
            failures.append(
                {
                    "field": f"{prefix}.invariants.{invariant}",
                    "expected": True,
                    "actual": passed,
                    "path": str(comparison_dir),
                }
            )


def _best_loss_delta(
    left: Any,
    right: Any,
) -> float | None:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None
    left_loss = left.get("best_hep_loss")
    right_loss = right.get("best_hep_loss")
    if left_loss is None or right_loss is None:
        return None
    return float(left_loss) - float(right_loss)


def _residual_objective_run_entry(
    run: dict[str, Any],
    *,
    max_logit_delta: float,
) -> dict[str, Any]:
    initial_loss = _optional_float(run.get("initial_residual_loss"))
    final_loss = _optional_float(run.get("final_residual_loss"))
    residual_delta = (
        None if initial_loss is None or final_loss is None else final_loss - initial_loss
    )
    alpha_candidates = _alpha_candidates([run])
    accepted = [
        candidate
        for candidate in alpha_candidates
        if candidate["alpha"] != 0.0
        and candidate["loss_improvement_from_alpha0"] is not None
        and candidate["loss_improvement_from_alpha0"] > 0.0
        and candidate["max_logit_delta_from_ordinary"] <= max_logit_delta
    ]
    best_hep = min(
        (
            candidate
            for candidate in alpha_candidates
            if candidate["loss"] is not None
        ),
        key=lambda candidate: candidate["loss"],
        default=None,
    )
    return {
        "experiment_id": run.get("experiment_id"),
        "residual_objective": run.get("residual_objective"),
        "status": run.get("status"),
        "dataset": run.get("dataset"),
        "training_steps": run.get("training_steps"),
        "support_stress": run.get("support_stress"),
        "support_stress_preset": run.get("support_stress_preset"),
        "hep_settling_objective": run.get("hep_settling_objective"),
        "hep_update_clip_norm": run.get("hep_update_clip_norm"),
        "initial_residual_loss": initial_loss,
        "final_residual_loss": final_loss,
        "residual_loss_delta": residual_delta,
        "residual_loss_ratio": _optional_float(run.get("residual_loss_ratio")),
        "own_loss_improved": residual_delta is not None and residual_delta < 0.0,
        "best_hep_loss": None if best_hep is None else best_hep["loss"],
        "best_hep_alpha": best_hep,
        "accepted_hep_alphas": accepted,
        "invariants": run.get("invariants")
        if isinstance(run.get("invariants"), dict)
        else {},
    }


def _add_pc_diagnostics(entry: dict[str, Any]) -> None:
    supervised = entry.get("supervised_run")
    pc = entry.get("pc_run")
    if not isinstance(supervised, dict) or not isinstance(pc, dict):
        entry["pc_minus_supervised_best_hep_loss"] = None
        entry["supervised_best_hep_loss_improvement_from_alpha0"] = None
        entry["pc_best_hep_loss_improvement_from_alpha0"] = None
        return
    supervised_best = supervised.get("best_hep_loss")
    pc_best = pc.get("best_hep_loss")
    entry["pc_minus_supervised_best_hep_loss"] = (
        None if supervised_best is None or pc_best is None else pc_best - supervised_best
    )
    entry["supervised_best_hep_loss_improvement_from_alpha0"] = (
        _best_hep_improvement_from_alpha0(supervised.get("best_hep_alpha"))
    )
    entry["pc_best_hep_loss_improvement_from_alpha0"] = (
        _best_hep_improvement_from_alpha0(pc.get("best_hep_alpha"))
    )


def _best_hep_improvement_from_alpha0(best_hep_alpha: Any) -> float | None:
    if not isinstance(best_hep_alpha, dict):
        return None
    improvement = best_hep_alpha.get("loss_improvement_from_alpha0")
    return None if improvement is None else float(improvement)


def _residual_objective_gate_decision(
    entries: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    pc_ce_wins: list[dict[str, Any]],
) -> dict[str, Any]:
    if failures or not entries:
        return {
            "decision": INSUFFICIENT_EVIDENCE,
            "continue_pc": False,
            "rationale": (
                "The residual objective gate requires matching local and Colab "
                "comparisons with passing artifact checks, supervised and PC runs, "
                "disabled support-stress presets, temporal clipped HEP, passing "
                "invariants, and own-objective loss improvement."
            ),
            "next_step": "repair or regenerate the objective-gate comparison artifacts",
        }
    if pc_ce_wins:
        return {
            "decision": CONTINUE_PC_RESIDUAL_OBJECTIVE_VALIDATION,
            "continue_pc": True,
            "rationale": (
                "PC-style residual training improved its own objective and produced "
                "lower supervised CE HEP loss than supervised residual training in "
                "at least one artifact-backed backend, so it merits broader PC "
                "objective validation before any default change."
            ),
            "next_step": "run a broader supervised-vs-PC objective-gate comparison outside the current char validation setting",
        }
    return {
        "decision": KEEP_SUPERVISED_CE_RESIDUAL_OBJECTIVE_DEFAULT,
        "continue_pc": False,
        "rationale": (
            "The objective-discriminative local and Colab evidence is valid and both "
            "objectives improve their own training losses, but PC-style residual "
            "training does not beat supervised residual training on supervised CE "
            "HEP loss. The default residual objective should remain supervised CE."
        ),
        "next_step": "inspect PC residual objective variants or diagnostics before another promotion-style objective gate",
    }


def _runs_with_residual_objective(
    runs: list[Any],
    residual_objective: str,
) -> list[dict[str, Any]]:
    return [
        run
        for run in runs
        if isinstance(run, dict) and run.get("residual_objective") == residual_objective
    ]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _decision(evidence: dict[str, Any], *, max_logit_delta: float) -> dict[str, Any]:
    if (
        evidence["artifact_check_status"] != "pass"
        or evidence["comparison_status"] != "ok"
        or evidence["verdict_status"] != "pass"
        or evidence["pinned_run_count"] < 1
        or evidence["repicked_run_count"] < 1
    ):
        return {
            "decision": INSUFFICIENT_EVIDENCE,
            "promote": False,
            "rationale": (
                "The comparison and artifact evidence must pass and include both "
                "pinned and ordinary repicked runs before making a baseline decision."
            ),
            "next_step": "repair or rerun the support-stress comparison artifacts",
        }

    accepted_pinned = [
        candidate
        for candidate in evidence["pinned_alpha_candidates"]
        if candidate["alpha"] != 0.0
        and candidate["loss_improvement_from_alpha0"] is not None
        and candidate["loss_improvement_from_alpha0"] > 0.0
        and candidate["max_logit_delta_from_ordinary"] <= max_logit_delta
    ]
    if accepted_pinned:
        return {
            "decision": PROMOTE,
            "promote": True,
            "rationale": (
                "Pinned support produced a nonzero HEP alpha with loss improvement "
                "within the default ordinary-logit delta budget."
            ),
            "next_step": "promote pinned support into the default Phase 0 comparison baseline",
        }

    return {
        "decision": KEEP_OPT_IN,
        "promote": False,
        "rationale": (
            "The support-stress evidence is valid and exposes support repicking, but "
            "pinned support has no accepted nonzero HEP alpha under the default "
            "loss-improvement and logit-delta policy."
        ),
        "next_step": "keep pinned support as an opt-in diagnostic and move to the next HEP mechanism",
    }


def _clipped_decision(
    evidence: dict[str, Any],
    *,
    max_logit_delta: float,
    max_pinned_vs_repicked_delta: float,
) -> dict[str, Any]:
    if (
        evidence["artifact_check_status"] != "pass"
        or evidence["comparison_status"] != "ok"
        or evidence["verdict_status"] != "pass"
        or evidence["clipped_run_count"] < 1
        or evidence["unclipped_run_count"] < 1
        or evidence["support_stress_run_count"] < 2
    ):
        return {
            "decision": INSUFFICIENT_EVIDENCE,
            "promote": False,
            "rationale": (
                "The comparison and artifact evidence must pass and include both "
                "clipped and unclipped support-stress runs before making a clipped "
                "HEP mitigation decision."
            ),
            "next_step": "repair or rerun the clipped support-stress comparison artifacts",
        }

    if not evidence["max_support_change_fraction"] or evidence[
        "max_support_change_fraction"
    ] <= 0.0:
        return {
            "decision": INSUFFICIENT_EVIDENCE,
            "promote": False,
            "rationale": (
                "The clipped HEP comparison did not exercise support repicking, so "
                "it cannot decide a support-stress mitigation."
            ),
            "next_step": "rerun clipped support-stress with nonzero support repicking",
        }

    accepted_clipped = [
        candidate
        for candidate in evidence["clipped_alpha_candidates"]
        if candidate["alpha"] != 0.0
        and candidate["loss_improvement_from_alpha0"] is not None
        and candidate["loss_improvement_from_alpha0"] > 0.0
        and candidate["max_logit_delta_from_ordinary"] <= max_logit_delta
        and candidate["pinned_vs_repicked_logit_delta"] <= max_pinned_vs_repicked_delta
    ]
    if accepted_clipped:
        return {
            "decision": PROMOTE_CLIPPED_HEP,
            "promote": True,
            "rationale": (
                "Clipped HEP produced a nonzero alpha with loss improvement while "
                "staying within both the ordinary-logit and pinned-vs-repicked "
                "delta budgets."
            ),
            "next_step": "promote clipped HEP into the default support-stress mitigation path",
        }

    return {
        "decision": KEEP_OPT_IN,
        "promote": False,
        "rationale": (
            "The clipped support-stress evidence is valid and clipping reduces "
            "pinned-vs-repicked settling divergence, but no clipped nonzero HEP "
            "alpha improves loss under the default policy."
        ),
        "next_step": "keep clipped HEP opt-in and test a mechanism that can improve loss under support stress",
    }


def _guided_clipped_decision(
    evidence: dict[str, Any],
    *,
    max_logit_delta: float,
    max_pinned_vs_repicked_delta: float,
) -> dict[str, Any]:
    if (
        evidence["artifact_check_status"] != "pass"
        or evidence["comparison_status"] != "ok"
        or evidence["verdict_status"] != "pass"
        or evidence["guided_run_count"] < 1
        or evidence["clipped_baseline_run_count"] < 1
        or evidence["support_stress_run_count"] < 2
    ):
        return {
            "decision": INSUFFICIENT_EVIDENCE,
            "promote": False,
            "rationale": (
                "The comparison and artifact evidence must pass and include both "
                "guided clipped and unguided clipped support-stress runs before "
                "recording the oracle decision."
            ),
            "next_step": "repair or rerun the guided clipped support-stress comparison artifacts",
        }

    if not evidence["max_support_change_fraction"] or evidence[
        "max_support_change_fraction"
    ] <= 0.0:
        return {
            "decision": INSUFFICIENT_EVIDENCE,
            "promote": False,
            "rationale": (
                "The guided clipped comparison did not exercise support repicking, "
                "so it cannot decide whether the oracle probe helped under support stress."
            ),
            "next_step": "rerun guided clipped support-stress with nonzero support repicking",
        }

    accepted_guided = [
        candidate
        for candidate in evidence["guided_alpha_candidates"]
        if candidate["alpha"] != 0.0
        and candidate["loss_improvement_from_alpha0"] is not None
        and candidate["loss_improvement_from_alpha0"] > 0.0
        and candidate["max_logit_delta_from_ordinary"] <= max_logit_delta
        and candidate["pinned_vs_repicked_logit_delta"] <= max_pinned_vs_repicked_delta
    ]
    if accepted_guided:
        return {
            "decision": GUIDED_ORACLE_CONFIRMED,
            "promote": False,
            "rationale": (
                "Guided clipped HEP produced a nonzero alpha with loss improvement "
                "inside both stability budgets, but it uses supervised labels during "
                "settling and therefore remains diagnostic-only."
            ),
            "next_step": "choose a deployable error signal to test against the guided clipped oracle",
        }

    return {
        "decision": KEEP_OPT_IN,
        "promote": False,
        "rationale": (
            "The guided clipped support-stress evidence is valid, but no guided "
            "nonzero HEP alpha improves loss under the default stability policy."
        ),
        "next_step": "keep guided clipped HEP as an opt-in oracle probe and inspect alternative error signals",
    }


def _temporal_clipped_decision(
    evidence: dict[str, Any],
    *,
    max_logit_delta: float,
    max_pinned_vs_repicked_delta: float,
) -> dict[str, Any]:
    if (
        evidence["artifact_check_status"] != "pass"
        or evidence["comparison_status"] != "ok"
        or evidence["verdict_status"] != "pass"
        or evidence["temporal_run_count"] < 1
        or evidence["entropy_run_count"] < 1
        or evidence["guided_run_count"] < 1
        or evidence["clipped_baseline_run_count"] < 1
        or evidence["support_stress_run_count"] < 4
    ):
        return {
            "decision": INSUFFICIENT_EVIDENCE,
            "selected": False,
            "rationale": (
                "The comparison and artifact evidence must pass and include temporal, "
                "entropy, guided oracle, and clipped baseline support-stress runs "
                "before selecting a label-free mitigation candidate."
            ),
            "next_step": "repair or rerun the temporal-vs-entropy guided clipped comparison artifacts",
        }

    if not evidence["max_support_change_fraction"] or evidence[
        "max_support_change_fraction"
    ] <= 0.0:
        return {
            "decision": INSUFFICIENT_EVIDENCE,
            "selected": False,
            "rationale": (
                "The temporal clipped comparison did not exercise support repicking, "
                "so it cannot decide whether the label-free signal helped under support stress."
            ),
            "next_step": "rerun temporal clipped support-stress with nonzero support repicking",
        }

    accepted_temporal = [
        candidate
        for candidate in evidence["temporal_alpha_candidates"]
        if candidate["alpha"] != 0.0
        and candidate["loss_improvement_from_alpha0"] is not None
        and candidate["loss_improvement_from_alpha0"] > 0.0
        and candidate["max_logit_delta_from_ordinary"] <= max_logit_delta
        and candidate["pinned_vs_repicked_logit_delta"] <= max_pinned_vs_repicked_delta
    ]
    if accepted_temporal:
        return {
            "decision": SELECT_TEMPORAL_CLIPPED_HEP,
            "selected": True,
            "rationale": (
                "Temporal clipped HEP is deployable at inference time and produced "
                "a nonzero alpha with support-stress loss improvement inside both "
                "stability budgets, while entropy did not improve loss in the same "
                "comparison and the guided oracle remains diagnostic-only."
            ),
            "next_step": "use temporal consistency as the selected label-free candidate for the next support-stress mitigation experiment",
        }

    return {
        "decision": KEEP_OPT_IN,
        "selected": False,
        "rationale": (
            "The temporal clipped support-stress evidence is valid, but no temporal "
            "nonzero HEP alpha improves loss under the default stability policy."
        ),
        "next_step": "keep temporal clipped HEP diagnostic-only and inspect alternative label-free error signals",
    }


def _temporal_aggregate_entry(
    path: Path,
    report: dict[str, Any],
    *,
    max_logit_delta: float,
    max_pinned_vs_repicked_delta: float,
) -> dict[str, Any]:
    evidence = report.get("evidence") if isinstance(report.get("evidence"), dict) else {}
    comparison_dir = evidence.get("comparison_dir")
    temporal_candidates = evidence.get("temporal_alpha_candidates")
    if not isinstance(temporal_candidates, list):
        temporal_candidates = []
    accepted = [
        candidate
        for candidate in temporal_candidates
        if isinstance(candidate, dict)
        and float(candidate.get("alpha", 0.0)) != 0.0
        and candidate.get("loss_improvement_from_alpha0") is not None
        and float(candidate["loss_improvement_from_alpha0"]) > 0.0
        and float(candidate.get("max_logit_delta_from_ordinary", 0.0))
        <= max_logit_delta
        and float(candidate.get("pinned_vs_repicked_logit_delta", 0.0))
        <= max_pinned_vs_repicked_delta
    ]
    best_temporal_alpha = (
        max(accepted, key=lambda candidate: float(candidate["loss_improvement_from_alpha0"]))
        if accepted
        else None
    )
    seed = _infer_seed_from_report(path, comparison_dir, temporal_candidates)
    backend = _infer_backend_from_report(path, comparison_dir)
    scale = _infer_scale_from_report(path, comparison_dir, temporal_candidates)
    selected = report.get("selected_label_free_support_stress_candidate") is True
    failures = []
    if report.get("status") != "pass":
        failures.append(
            {
                "field": "decision_report.status",
                "expected": "pass",
                "actual": report.get("status"),
                "path": str(path),
            }
        )
    if report.get("decision") != SELECT_TEMPORAL_CLIPPED_HEP:
        failures.append(
            {
                "field": "decision_report.decision",
                "expected": SELECT_TEMPORAL_CLIPPED_HEP,
                "actual": report.get("decision"),
                "path": str(path),
            }
        )
    if not selected:
        failures.append(
            {
                "field": "decision_report.selected_label_free_support_stress_candidate",
                "expected": True,
                "actual": report.get("selected_label_free_support_stress_candidate"),
                "path": str(path),
            }
        )
    if best_temporal_alpha is None:
        failures.append(
            {
                "field": "decision_report.temporal_alpha_candidates",
                "expected": "accepted nonzero temporal alpha",
                "actual": "none",
                "path": str(path),
            }
        )
    if seed is None:
        failures.append(
            {
                "field": "decision_report.seed",
                "expected": "inferable seed",
                "actual": None,
                "path": str(path),
            }
        )
    if backend is None:
        failures.append(
            {
                "field": "decision_report.backend",
                "expected": "local or colab",
                "actual": None,
                "path": str(path),
            }
        )

    return {
        "path": str(path),
        "comparison_dir": comparison_dir,
        "scale": scale,
        "seed": seed,
        "backend": backend,
        "status": report.get("status"),
        "decision": report.get("decision"),
        "selected_label_free_support_stress_candidate": selected,
        "artifact_check_status": evidence.get("artifact_check_status"),
        "comparison_status": evidence.get("comparison_status"),
        "verdict_status": evidence.get("verdict_status"),
        "max_support_change_fraction": evidence.get("max_support_change_fraction"),
        "best_temporal_alpha": best_temporal_alpha,
        "failures": failures,
    }


def _temporal_promotion_gate_entry(
    path: Path,
    report: dict[str, Any],
    *,
    max_logit_delta: float,
    max_pinned_vs_repicked_delta: float,
) -> dict[str, Any]:
    entry = _temporal_aggregate_entry(
        path,
        report,
        max_logit_delta=max_logit_delta,
        max_pinned_vs_repicked_delta=max_pinned_vs_repicked_delta,
    )
    evidence = report.get("evidence") if isinstance(report.get("evidence"), dict) else {}
    temporal_candidates = evidence.get("temporal_alpha_candidates")
    if not isinstance(temporal_candidates, list):
        temporal_candidates = []
    gate = _infer_promotion_gate_from_report(
        path,
        evidence.get("comparison_dir"),
        temporal_candidates,
    )
    entry["gate"] = gate
    if gate is None:
        entry["failures"].append(
            {
                "field": "decision_report.gate",
                "expected": "larger char or non-char tokenized gate",
                "actual": None,
                "path": str(path),
            }
        )
    if entry.get("artifact_check_status") != "pass":
        entry["failures"].append(
            {
                "field": "decision_report.artifact_check_status",
                "expected": "pass",
                "actual": entry.get("artifact_check_status"),
                "path": str(path),
            }
        )
    if (
        entry.get("max_support_change_fraction") is None
        or float(entry["max_support_change_fraction"]) <= 0.0
    ):
        entry["failures"].append(
            {
                "field": "decision_report.max_support_change_fraction",
                "expected": "> 0.0",
                "actual": entry.get("max_support_change_fraction"),
                "path": str(path),
            }
        )
    return entry


def _infer_seed_from_report(
    path: Path,
    comparison_dir: Any,
    temporal_candidates: list[Any],
) -> int | None:
    texts = [str(path), str(comparison_dir or "")]
    texts.extend(
        str(candidate.get("experiment_id", ""))
        for candidate in temporal_candidates
        if isinstance(candidate, dict)
    )
    for text in texts:
        marker = "seed"
        if marker not in text:
            continue
        suffix = text.split(marker, 1)[1]
        digits = []
        for char in suffix:
            if char.isdigit():
                digits.append(char)
            elif digits:
                break
        if digits:
            return int("".join(digits))
    if any("temporal_clipped_hep" in text for text in texts):
        return 1
    return None


def _infer_scale_from_report(
    path: Path,
    comparison_dir: Any,
    temporal_candidates: list[Any],
) -> str | None:
    texts = [str(path), str(comparison_dir or "")]
    texts.extend(
        str(candidate.get("experiment_id", ""))
        for candidate in temporal_candidates
        if isinstance(candidate, dict)
    )
    joined = " ".join(texts)
    if "token_larger" in joined:
        return "token_larger"
    if "larger" in joined:
        return "larger"
    if "extended" in joined:
        return "extended"
    if "validation" in joined:
        return "validation"
    if "smoke" in joined or "seed" in joined or "temporal_clipped_hep" in joined:
        return "seed_smoke"
    return None


def _infer_backend_from_report(path: Path, comparison_dir: Any) -> str | None:
    text = f"{path} {comparison_dir or ''}"
    if "colab_" in text or "_colab_" in text:
        return "colab"
    if "local" in text:
        return "local"
    if "results/comparisons/" in text or "temporal_clipped_hep" in text:
        return "local"
    return None


def _infer_promotion_gate_from_report(
    path: Path,
    comparison_dir: Any,
    temporal_candidates: list[Any],
) -> str | None:
    texts = [str(path), str(comparison_dir or "")]
    texts.extend(
        str(candidate.get("experiment_id", ""))
        for candidate in temporal_candidates
        if isinstance(candidate, dict)
    )
    joined = " ".join(texts)
    if "token_larger" in joined:
        return "non_char_tokenized_local_colab"
    if "larger" in joined:
        return "larger_char_local_colab"
    return None


def _temporal_aggregate_decision(
    entries: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    if failures or not entries:
        return {
            "decision": INSUFFICIENT_EVIDENCE,
            "selected": False,
            "rationale": (
                "The aggregate requires every temporal clipped decision report to "
                "pass, select temporal consistency, and include an accepted nonzero "
                "temporal alpha inside the stability budgets."
            ),
            "next_step": (
                "repair or regenerate the missing or failing temporal clipped "
                "decision reports"
            ),
        }

    return {
        "decision": SELECT_TEMPORAL_CLIPPED_HEP_AGGREGATE,
        "selected": True,
        "rationale": (
            "All included local and Colab seed-smoke decision reports select temporal "
            "consistency as the deployable label-free support-stress candidate and "
            "include a nonzero temporal alpha with loss improvement inside both "
            "stability budgets."
        ),
        "next_step": (
            "run a broader non-smoke temporal-clipped validation before any "
            "default-promotion decision"
        ),
    }


def _temporal_cross_scale_aggregate_decision(
    entries: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    if failures or not entries:
        return {
            "decision": INSUFFICIENT_EVIDENCE,
            "selected": False,
            "rationale": (
                "The cross-scale aggregate requires every temporal clipped decision "
                "report to pass, select temporal consistency, include an accepted "
                "nonzero temporal alpha inside the stability budgets, and cover "
                "seed-smoke, validation, and extended local/Colab evidence."
            ),
            "next_step": (
                "repair or regenerate the missing or failing cross-scale temporal "
                "clipped decision reports"
            ),
        }

    return {
        "decision": SELECT_TEMPORAL_CLIPPED_HEP_CROSS_SCALE_AGGREGATE,
        "selected": True,
        "rationale": (
            "All included seed-smoke, validation, and extended local/Colab decision "
            "reports select temporal consistency as the deployable label-free "
            "support-stress candidate and include a nonzero temporal alpha with "
            "loss improvement inside both stability budgets."
        ),
        "next_step": (
            "define and run the next broader promotion gate before changing the "
            "default support-stress mitigation path"
        ),
    }


def _runs_with_objective(
    runs: list[Any],
    settling_objective: str,
) -> list[dict[str, Any]]:
    return [
        run
        for run in runs
        if isinstance(run, dict)
        and run.get("hep_settling_objective") == settling_objective
    ]


def _alpha_candidates(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for run in runs:
        alpha0_loss = _alpha0_loss(run.get("hep_alpha_sweep") or [])
        for entry in run.get("hep_alpha_sweep") or []:
            if not isinstance(entry, dict) or entry.get("loss") is None:
                continue
            loss = float(entry["loss"])
            candidates.append(
                {
                    "experiment_id": run.get("experiment_id"),
                    "alpha": float(entry["alpha"]),
                    "loss": loss,
                    "loss_improvement_from_alpha0": (
                        None if alpha0_loss is None else alpha0_loss - loss
                    ),
                    "max_logit_delta_from_ordinary": float(
                        entry.get("max_logit_delta_from_ordinary", 0.0)
                    ),
                    "support_change_fraction": float(
                        entry.get("support_change_fraction", 0.0)
                    ),
                    "pinned_vs_repicked_logit_delta": float(
                        entry.get("pinned_vs_repicked_logit_delta", 0.0)
                    ),
                }
            )
    return candidates


def _alpha0_loss(sweep: list[dict[str, Any]]) -> float | None:
    losses = [
        float(entry["loss"])
        for entry in sweep
        if isinstance(entry, dict)
        and float(entry.get("alpha", -1.0)) == 0.0
        and entry.get("loss") is not None
    ]
    return min(losses) if losses else None


def _max_nested_metric(
    runs: list[Any],
    parent_key: str,
    metric_key: str,
) -> float | None:
    values = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        parent = run.get(parent_key)
        if isinstance(parent, dict) and parent.get(metric_key) is not None:
            values.append(float(parent[metric_key]))
    return max(values) if values else None


def _max_alpha_metric(runs: list[dict[str, Any]], metric_key: str) -> float | None:
    values = []
    for run in runs:
        for entry in run.get("hep_alpha_sweep") or []:
            if isinstance(entry, dict) and entry.get(metric_key) is not None:
                values.append(float(entry[metric_key]))
    return max(values) if values else None


def _read_json_object(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def _read_config_object(path: Path) -> dict[str, Any]:
    from relaleap.experiments.run import _read_config

    loaded = _read_config(path)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a config object")
    return loaded


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    evidence = report["evidence"]
    lines = [
        "# Pinned Support Decision Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        (
            "- Promote to default Phase 0 baseline: "
            f"`{report['promote_to_default_phase0_baseline']}`"
        ),
        f"- Artifact check: `{evidence['artifact_check_status']}`",
        f"- Comparison verdict: `{evidence['verdict_status']}`",
        (
            "- Max support change fraction: "
            f"`{_format_metric(evidence['max_support_change_fraction'])}`"
        ),
        (
            "- Max pinned-vs-repicked logit delta: "
            f"`{_format_metric(evidence['max_pinned_vs_repicked_logit_delta'])}`"
        ),
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Pinned HEP Candidates",
        "",
        "| Alpha | Loss | Improvement vs alpha 0 | Logit delta | Support change | Pinned-vs-repicked |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in evidence["pinned_alpha_candidates"]:
        lines.append(
            (
                f"| {_format_metric(candidate['alpha'])} "
                f"| {_format_metric(candidate['loss'])} "
                f"| {_format_metric(candidate['loss_improvement_from_alpha0'])} "
                f"| {_format_metric(candidate['max_logit_delta_from_ordinary'])} "
                f"| {_format_metric(candidate['support_change_fraction'])} "
                f"| {_format_metric(candidate['pinned_vs_repicked_logit_delta'])} |"
            )
        )
    lines.extend(["", "## Next Step", "", report["next_step"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_clipped_markdown(path: Path, report: dict[str, Any]) -> None:
    evidence = report["evidence"]
    lines = [
        "# Clipped HEP Decision Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        (
            "- Promote to default support-stress mitigation: "
            f"`{report['promote_to_default_support_stress_mitigation']}`"
        ),
        f"- Artifact check: `{evidence['artifact_check_status']}`",
        f"- Comparison verdict: `{evidence['verdict_status']}`",
        (
            "- Max support change fraction: "
            f"`{_format_metric(evidence['max_support_change_fraction'])}`"
        ),
        (
            "- Max unclipped pinned-vs-repicked logit delta: "
            f"`{_format_metric(evidence['max_unclipped_pinned_vs_repicked_logit_delta'])}`"
        ),
        (
            "- Max clipped pinned-vs-repicked logit delta: "
            f"`{_format_metric(evidence['max_clipped_pinned_vs_repicked_logit_delta'])}`"
        ),
        (
            "- Max pinned-vs-repicked logit delta reduction: "
            f"`{_format_metric(evidence['max_pinned_vs_repicked_logit_delta_reduction'])}`"
        ),
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Clipped HEP Candidates",
        "",
        "| Alpha | Loss | Improvement vs alpha 0 | Logit delta | Support change | Pinned-vs-repicked |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in evidence["clipped_alpha_candidates"]:
        lines.append(
            (
                f"| {_format_metric(candidate['alpha'])} "
                f"| {_format_metric(candidate['loss'])} "
                f"| {_format_metric(candidate['loss_improvement_from_alpha0'])} "
                f"| {_format_metric(candidate['max_logit_delta_from_ordinary'])} "
                f"| {_format_metric(candidate['support_change_fraction'])} "
                f"| {_format_metric(candidate['pinned_vs_repicked_logit_delta'])} |"
            )
        )
    lines.extend(["", "## Next Step", "", report["next_step"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_guided_clipped_markdown(path: Path, report: dict[str, Any]) -> None:
    evidence = report["evidence"]
    lines = [
        "# Guided Clipped HEP Decision Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        (
            "- Promote to default support-stress mitigation: "
            f"`{report['promote_to_default_support_stress_mitigation']}`"
        ),
        f"- Diagnostic oracle only: `{report['diagnostic_oracle_only']}`",
        f"- Artifact check: `{evidence['artifact_check_status']}`",
        f"- Comparison verdict: `{evidence['verdict_status']}`",
        (
            "- Max support change fraction: "
            f"`{_format_metric(evidence['max_support_change_fraction'])}`"
        ),
        (
            "- Max guided pinned-vs-repicked logit delta: "
            f"`{_format_metric(evidence['max_guided_pinned_vs_repicked_logit_delta'])}`"
        ),
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Guided Clipped HEP Candidates",
        "",
        "| Alpha | Loss | Improvement vs alpha 0 | Logit delta | Support change | Pinned-vs-repicked |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in evidence["guided_alpha_candidates"]:
        lines.append(
            (
                f"| {_format_metric(candidate['alpha'])} "
                f"| {_format_metric(candidate['loss'])} "
                f"| {_format_metric(candidate['loss_improvement_from_alpha0'])} "
                f"| {_format_metric(candidate['max_logit_delta_from_ordinary'])} "
                f"| {_format_metric(candidate['support_change_fraction'])} "
                f"| {_format_metric(candidate['pinned_vs_repicked_logit_delta'])} |"
            )
        )
    lines.extend(["", "## Next Step", "", report["next_step"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_temporal_clipped_markdown(path: Path, report: dict[str, Any]) -> None:
    evidence = report["evidence"]
    lines = [
        "# Temporal Clipped HEP Decision Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        (
            "- Selected label-free support-stress candidate: "
            f"`{report['selected_label_free_support_stress_candidate']}`"
        ),
        (
            "- Promote to default support-stress mitigation: "
            f"`{report['promote_to_default_support_stress_mitigation']}`"
        ),
        f"- Deployable label-free signal: `{report['deployable_label_free_signal']}`",
        f"- Artifact check: `{evidence['artifact_check_status']}`",
        f"- Comparison verdict: `{evidence['verdict_status']}`",
        (
            "- Max support change fraction: "
            f"`{_format_metric(evidence['max_support_change_fraction'])}`"
        ),
        (
            "- Max temporal pinned-vs-repicked logit delta: "
            f"`{_format_metric(evidence['max_temporal_pinned_vs_repicked_logit_delta'])}`"
        ),
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Temporal Clipped HEP Candidates",
        "",
        "| Alpha | Loss | Improvement vs alpha 0 | Logit delta | Support change | Pinned-vs-repicked |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in evidence["temporal_alpha_candidates"]:
        lines.append(
            (
                f"| {_format_metric(candidate['alpha'])} "
                f"| {_format_metric(candidate['loss'])} "
                f"| {_format_metric(candidate['loss_improvement_from_alpha0'])} "
                f"| {_format_metric(candidate['max_logit_delta_from_ordinary'])} "
                f"| {_format_metric(candidate['support_change_fraction'])} "
                f"| {_format_metric(candidate['pinned_vs_repicked_logit_delta'])} |"
            )
        )
    lines.extend(["", "## Next Step", "", report["next_step"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_temporal_aggregate_markdown(path: Path, report: dict[str, Any]) -> None:
    evidence = report["evidence"]
    lines = [
        "# Temporal Clipped HEP Multi-Seed Aggregate",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        (
            "- Selected label-free support-stress candidate: "
            f"`{report['selected_label_free_support_stress_candidate']}`"
        ),
        (
            "- Promote to default support-stress mitigation: "
            f"`{report['promote_to_default_support_stress_mitigation']}`"
        ),
        f"- Report count: `{evidence['report_count']}`",
        f"- Selected report count: `{evidence['selected_report_count']}`",
        f"- Accepted temporal report count: `{evidence['accepted_temporal_report_count']}`",
        (
            "- Mean temporal loss improvement from alpha 0: "
            f"`{_format_metric(evidence['mean_temporal_loss_improvement_from_alpha0'])}`"
        ),
        (
            "- Max temporal logit delta from ordinary: "
            f"`{_format_metric(evidence['max_temporal_logit_delta_from_ordinary'])}`"
        ),
        (
            "- Max temporal pinned-vs-repicked logit delta: "
            f"`{_format_metric(evidence['max_temporal_pinned_vs_repicked_logit_delta'])}`"
        ),
        (
            "- Max support change fraction: "
            f"`{_format_metric(evidence['max_support_change_fraction'])}`"
        ),
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Evidence",
        "",
        (
            "| Seed | Backend | Status | Selected | Alpha | Loss improvement "
            "| Logit delta | Pinned-vs-repicked | Source |"
        ),
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for entry in evidence["entries"]:
        alpha = entry.get("best_temporal_alpha") or {}
        lines.append(
            (
                f"| {entry.get('seed') or ''} "
                f"| {entry.get('backend') or ''} "
                f"| {entry.get('status') or ''} "
                f"| {entry.get('selected_label_free_support_stress_candidate')} "
                f"| {_format_metric(alpha.get('alpha'))} "
                f"| {_format_metric(alpha.get('loss_improvement_from_alpha0'))} "
                f"| {_format_metric(alpha.get('max_logit_delta_from_ordinary'))} "
                f"| {_format_metric(alpha.get('pinned_vs_repicked_logit_delta'))} "
                f"| `{entry.get('path')}` |"
            )
        )
    if evidence["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in evidence["failures"]:
            lines.append(
                (
                    f"- `{failure.get('field')}` expected "
                    f"`{failure.get('expected')}`, got `{failure.get('actual')}` "
                    f"at `{failure.get('path', '')}`"
                )
            )
    lines.extend(["", "## Next Step", "", report["next_step"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_temporal_cross_scale_aggregate_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    evidence = report["evidence"]
    lines = [
        "# Temporal Clipped HEP Cross-Scale Aggregate",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        (
            "- Selected label-free support-stress candidate: "
            f"`{report['selected_label_free_support_stress_candidate']}`"
        ),
        (
            "- Promote to default support-stress mitigation: "
            f"`{report['promote_to_default_support_stress_mitigation']}`"
        ),
        f"- Report count: `{evidence['report_count']}`",
        f"- Scale count: `{evidence['scale_count']}`",
        f"- Selected report count: `{evidence['selected_report_count']}`",
        f"- Accepted temporal report count: `{evidence['accepted_temporal_report_count']}`",
        (
            "- Mean temporal loss improvement from alpha 0: "
            f"`{_format_metric(evidence['mean_temporal_loss_improvement_from_alpha0'])}`"
        ),
        (
            "- Max temporal logit delta from ordinary: "
            f"`{_format_metric(evidence['max_temporal_logit_delta_from_ordinary'])}`"
        ),
        (
            "- Max temporal pinned-vs-repicked logit delta: "
            f"`{_format_metric(evidence['max_temporal_pinned_vs_repicked_logit_delta'])}`"
        ),
        (
            "- Max support change fraction: "
            f"`{_format_metric(evidence['max_support_change_fraction'])}`"
        ),
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Evidence",
        "",
        (
            "| Scale | Seed | Backend | Status | Selected | Alpha | Loss improvement "
            "| Logit delta | Pinned-vs-repicked | Source |"
        ),
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for entry in evidence["entries"]:
        alpha = entry.get("best_temporal_alpha") or {}
        lines.append(
            (
                f"| {entry.get('scale') or ''} "
                f"| {entry.get('seed') or ''} "
                f"| {entry.get('backend') or ''} "
                f"| {entry.get('status') or ''} "
                f"| {entry.get('selected_label_free_support_stress_candidate')} "
                f"| {_format_metric(alpha.get('alpha'))} "
                f"| {_format_metric(alpha.get('loss_improvement_from_alpha0'))} "
                f"| {_format_metric(alpha.get('max_logit_delta_from_ordinary'))} "
                f"| {_format_metric(alpha.get('pinned_vs_repicked_logit_delta'))} "
                f"| `{entry.get('path')}` |"
            )
        )
    if evidence["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in evidence["failures"]:
            lines.append(
                (
                    f"- `{failure.get('field')}` expected "
                    f"`{failure.get('expected')}`, got `{failure.get('actual')}` "
                    f"at `{failure.get('path', '')}`"
                )
            )
    lines.extend(["", "## Next Step", "", report["next_step"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_temporal_promotion_gate_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    evidence = report["evidence"]
    lines = [
        "# Temporal Clipped HEP Promotion Gate",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        (
            "- Promote to default support-stress mitigation: "
            f"`{report['promote_to_default_support_stress_mitigation']}`"
        ),
        f"- Cross-scale report: `{evidence['cross_scale_report_path']}`",
        f"- Cross-scale status: `{evidence['cross_scale_status']}`",
        f"- Cross-scale decision: `{evidence['cross_scale_decision']}`",
        (
            "- Cross-scale accepted temporal report count: "
            f"`{evidence['cross_scale_accepted_temporal_report_count']}`"
        ),
        (
            "- Cross-scale mean temporal loss improvement from alpha 0: "
            f"`{_format_metric(evidence['cross_scale_mean_temporal_loss_improvement_from_alpha0'])}`"
        ),
        (
            "- Cross-scale max temporal logit delta from ordinary: "
            f"`{_format_metric(evidence['cross_scale_max_temporal_logit_delta_from_ordinary'])}`"
        ),
        (
            "- Cross-scale max temporal pinned-vs-repicked logit delta: "
            f"`{_format_metric(evidence['cross_scale_max_temporal_pinned_vs_repicked_logit_delta'])}`"
        ),
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Required Evidence",
        "",
        (
            "| Gate | Backends | Minimum seq len | Minimum hidden dim "
            "| Minimum columns | Minimum steps |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for requirement in evidence["required_evidence"]:
        minimum = requirement["minimum_scale"]
        lines.append(
            (
                f"| {requirement['gate']} "
                f"| {', '.join(requirement['required_backends'])} "
                f"| {minimum['seq_len']} "
                f"| {minimum['hidden_dim']} "
                f"| {minimum['num_columns']} "
                f"| {minimum['training_steps']} |"
            )
        )
    if evidence["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in evidence["failures"]:
            lines.append(
                (
                    f"- `{failure.get('field')}` expected "
                    f"`{failure.get('expected')}`, got `{failure.get('actual')}` "
                    f"at `{failure.get('path', '')}`"
                )
            )
    lines.extend(["", "## Next Step", "", report["next_step"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_temporal_promotion_gate_satisfaction_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    evidence = report["evidence"]
    lines = [
        "# Temporal Clipped HEP Promotion Gate Satisfaction",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        f"- Promotion gate satisfied: `{report['promotion_gate_satisfied']}`",
        (
            "- Promote to default support-stress mitigation: "
            f"`{report['promote_to_default_support_stress_mitigation']}`"
        ),
        f"- Promotion gate report: `{evidence['promotion_gate_report_path']}`",
        f"- Promotion gate status: `{evidence['promotion_gate_status']}`",
        f"- Promotion gate decision: `{evidence['promotion_gate_decision']}`",
        f"- Report count: `{evidence['report_count']}`",
        (
            "- Accepted temporal report count: "
            f"`{evidence['accepted_temporal_report_count']}`"
        ),
        (
            "- Mean temporal loss improvement from alpha 0: "
            f"`{_format_metric(evidence['mean_temporal_loss_improvement_from_alpha0'])}`"
        ),
        (
            "- Max temporal logit delta from ordinary: "
            f"`{_format_metric(evidence['max_temporal_logit_delta_from_ordinary'])}`"
        ),
        (
            "- Max temporal pinned-vs-repicked logit delta: "
            f"`{_format_metric(evidence['max_temporal_pinned_vs_repicked_logit_delta'])}`"
        ),
        (
            "- Max support change fraction: "
            f"`{_format_metric(evidence['max_support_change_fraction'])}`"
        ),
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Evidence",
        "",
        (
            "| Gate | Backend | Status | Artifact check | Selected | Alpha "
            "| Loss improvement | Logit delta | Pinned-vs-repicked | Support change | Source |"
        ),
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for entry in evidence["entries"]:
        alpha = entry.get("best_temporal_alpha") or {}
        lines.append(
            (
                f"| {entry.get('gate') or ''} "
                f"| {entry.get('backend') or ''} "
                f"| {entry.get('status') or ''} "
                f"| {entry.get('artifact_check_status') or ''} "
                f"| {entry.get('selected_label_free_support_stress_candidate')} "
                f"| {_format_metric(alpha.get('alpha'))} "
                f"| {_format_metric(alpha.get('loss_improvement_from_alpha0'))} "
                f"| {_format_metric(alpha.get('max_logit_delta_from_ordinary'))} "
                f"| {_format_metric(alpha.get('pinned_vs_repicked_logit_delta'))} "
                f"| {_format_metric(entry.get('max_support_change_fraction'))} "
                f"| `{entry.get('path')}` |"
            )
        )
    if evidence["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in evidence["failures"]:
            lines.append(
                (
                    f"- `{failure.get('field')}` expected "
                    f"`{failure.get('expected')}`, got `{failure.get('actual')}` "
                    f"at `{failure.get('path', '')}`"
                )
            )
    lines.extend(["", "## Next Step", "", report["next_step"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_post_promotion_residual_learning_gate_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    evidence = report["evidence"]
    lines = [
        "# Post-Promotion Residual Learning Gate",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        (
            "- Promoted temporal support-stress default confirmed: "
            f"`{report['promoted_temporal_support_stress_default_confirmed']}`"
        ),
        (
            "- Promote residual learning method: "
            f"`{report['promote_residual_learning_method']}`"
        ),
        (
            "- Promotion satisfaction report: "
            f"`{evidence['promotion_satisfaction_report_path']}`"
        ),
        (
            "- Promotion satisfaction status: "
            f"`{evidence['promotion_satisfaction_status']}`"
        ),
        (
            "- Default support-stress config: "
            f"`{evidence['default_support_stress_config_path']}`"
        ),
        (
            "- Default settling objective: "
            f"`{evidence['default_support_stress_settling_objective']}`"
        ),
        (
            "- Default clip norm: "
            f"`{_format_metric(evidence['default_support_stress_clip_norm'])}`"
        ),
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Required Evidence",
        "",
        "| Gate | Required evidence |",
        "| --- | --- |",
    ]
    for requirement in evidence["required_evidence"]:
        required = []
        for key in (
            "required_backends",
            "required_runs",
            "required_invariants",
            "required_artifacts",
        ):
            values = requirement.get(key)
            if isinstance(values, list):
                required.append(f"{key}: {', '.join(str(value) for value in values)}")
        minimum = requirement.get("minimum_scale")
        if isinstance(minimum, dict):
            required.append(
                "minimum_scale: "
                + ", ".join(f"{key}={value}" for key, value in minimum.items())
            )
        lines.append(f"| {requirement['gate']} | {'; '.join(required)} |")
    if evidence["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in evidence["failures"]:
            lines.append(
                (
                    f"- `{failure.get('field')}` expected "
                    f"`{failure.get('expected')}`, got `{failure.get('actual')}` "
                    f"at `{failure.get('path', '')}`"
                )
            )
    lines.extend(["", "## Next Step", "", report["next_step"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_residual_objective_gate_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    evidence = report["evidence"]
    lines = [
        "# Residual Objective Gate Decision",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        (
            "- Continue PC residual objective validation: "
            f"`{report['continue_pc_residual_objective_validation']}`"
        ),
        (
            "- Promote residual learning method: "
            f"`{report['promote_residual_learning_method']}`"
        ),
        f"- Default residual objective: `{report['default_residual_objective']}`",
        f"- Backends: `{', '.join(evidence['backends'])}`",
        f"- Supervised run count: `{evidence['supervised_run_count']}`",
        f"- PC run count: `{evidence['pc_run_count']}`",
        f"- PC CE win count: `{evidence['pc_ce_win_count']}`",
        (
            "- Mean supervised best HEP loss: "
            f"`{_format_metric(evidence['mean_supervised_best_hep_loss'])}`"
        ),
        (
            "- Mean PC best HEP loss: "
            f"`{_format_metric(evidence['mean_pc_best_hep_loss'])}`"
        ),
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Evidence",
        "",
        (
            "| Backend | Artifact check | Verdict | Objective | Own loss delta "
            "| Best HEP alpha | Best HEP loss | Support preset disabled | Source |"
        ),
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for entry in evidence["entries"]:
        for key in ("supervised_run", "pc_run"):
            run = entry.get(key)
            if not run:
                continue
            alpha = run.get("best_hep_alpha") or {}
            lines.append(
                (
                    f"| {entry.get('backend') or ''} "
                    f"| {entry.get('artifact_check_status') or ''} "
                    f"| {entry.get('verdict_status') or ''} "
                    f"| {run.get('residual_objective') or ''} "
                    f"| {_format_metric(run.get('residual_loss_delta'))} "
                    f"| {_format_metric(alpha.get('alpha'))} "
                    f"| {_format_metric(run.get('best_hep_loss'))} "
                    f"| {run.get('support_stress_preset') is False} "
                    f"| `{entry.get('comparison_dir')}` |"
                )
            )
    if evidence["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in evidence["failures"]:
            lines.append(
                (
                    f"- `{failure.get('field')}` expected "
                    f"`{failure.get('expected')}`, got `{failure.get('actual')}` "
                    f"at `{failure.get('path', '')}`"
                )
            )
    lines.extend(["", "## Next Step", "", report["next_step"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_pc_residual_objective_diagnostics_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    evidence = report["evidence"]
    lines = [
        "# PC Residual Objective Diagnostics",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        f"- Default residual objective: `{report['default_residual_objective']}`",
        f"- Backends: `{', '.join(evidence['backends'])}`",
        (
            "- Mean PC minus supervised best HEP loss: "
            f"`{_format_metric(evidence['mean_pc_minus_supervised_best_hep_loss'])}`"
        ),
        (
            "- Mean supervised residual loss ratio: "
            f"`{_format_metric(evidence['mean_supervised_residual_loss_ratio'])}`"
        ),
        (
            "- Mean PC residual loss ratio: "
            f"`{_format_metric(evidence['mean_pc_residual_loss_ratio'])}`"
        ),
        (
            "- Mean supervised best HEP improvement from alpha 0: "
            f"`{_format_metric(evidence['mean_supervised_best_hep_loss_improvement_from_alpha0'])}`"
        ),
        (
            "- Mean PC best HEP improvement from alpha 0: "
            f"`{_format_metric(evidence['mean_pc_best_hep_loss_improvement_from_alpha0'])}`"
        ),
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Evidence",
        "",
        (
            "| Backend | Artifact check | Supervised best HEP loss | PC best HEP loss "
            "| PC minus supervised | Supervised own ratio | PC own ratio | Source |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for entry in evidence["entries"]:
        supervised = entry.get("supervised_run") or {}
        pc = entry.get("pc_run") or {}
        lines.append(
            (
                f"| {entry.get('backend') or ''} "
                f"| {entry.get('artifact_check_status') or ''} "
                f"| {_format_metric(supervised.get('best_hep_loss'))} "
                f"| {_format_metric(pc.get('best_hep_loss'))} "
                f"| {_format_metric(entry.get('pc_minus_supervised_best_hep_loss'))} "
                f"| {_format_metric(supervised.get('residual_loss_ratio'))} "
                f"| {_format_metric(pc.get('residual_loss_ratio'))} "
                f"| `{entry.get('comparison_dir')}` |"
            )
        )
    if evidence["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in evidence["failures"]:
            lines.append(
                (
                    f"- `{failure.get('field')}` expected "
                    f"`{failure.get('expected')}`, got `{failure.get('actual')}` "
                    f"at `{failure.get('path', '')}`"
                )
            )
    lines.extend(["", "## Next Step", "", report["next_step"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_anchored_pc_residual_objective_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    evidence = report["evidence"]
    lines = [
        "# Anchored PC Residual Objective Decision",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        (
            "- Continue PC residual objective validation: "
            f"`{report['continue_pc_residual_objective_validation']}`"
        ),
        (
            "- Selected PC variant: "
            f"`{report['selected_pc_residual_objective_variant']}`"
        ),
        f"- Default residual objective: `{report['default_residual_objective']}`",
        f"- Backends: `{', '.join(evidence['backends'])}`",
        (
            "- Mean PC minus supervised best HEP loss: "
            f"`{_format_metric(evidence['mean_pc_minus_supervised_best_hep_loss'])}`"
        ),
        (
            "- Mean anchored PC minus supervised best HEP loss: "
            f"`{_format_metric(evidence['mean_anchored_pc_minus_supervised_best_hep_loss'])}`"
        ),
        (
            "- Mean PC-to-anchored gap reduction: "
            f"`{_format_metric(evidence['mean_pc_to_anchored_gap_reduction'])}`"
        ),
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Evidence",
        "",
        (
            "| Backend | Artifact check | Supervised best HEP loss "
            "| PC best HEP loss | Anchored PC best HEP loss "
            "| Anchored minus supervised | Gap reduction | Source |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for entry in evidence["entries"]:
        supervised = entry.get("supervised_run") or {}
        pc = entry.get("pc_run") or {}
        anchored = entry.get("anchored_pc_run") or {}
        lines.append(
            (
                f"| {entry.get('backend') or ''} "
                f"| {entry.get('artifact_check_status') or ''} "
                f"| {_format_metric(supervised.get('best_hep_loss'))} "
                f"| {_format_metric(pc.get('best_hep_loss'))} "
                f"| {_format_metric(anchored.get('best_hep_loss'))} "
                f"| {_format_metric(entry.get('anchored_pc_minus_supervised_best_hep_loss'))} "
                f"| {_format_metric(entry.get('pc_to_anchored_gap_reduction'))} "
                f"| `{entry.get('comparison_dir')}` |"
            )
        )
    if evidence["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in evidence["failures"]:
            lines.append(
                (
                    f"- `{failure.get('field')}` expected "
                    f"`{failure.get('expected')}`, got `{failure.get('actual')}` "
                    f"at `{failure.get('path', '')}`"
                )
            )
    lines.extend(["", "## Next Step", "", report["next_step"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_metric(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.8f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a HEP mechanism decision report from artifacts."
    )
    parser.add_argument(
        "--report",
        choices=(
            "pinned-support",
            "clipped-hep",
            "guided-clipped-hep",
            "temporal-clipped-hep",
            "temporal-clipped-hep-aggregate",
            "temporal-clipped-hep-cross-scale-aggregate",
            "temporal-clipped-hep-promotion-gate",
            "temporal-clipped-hep-promotion-gate-satisfaction",
            "post-promotion-residual-learning-gate",
            "residual-objective-gate",
            "pc-residual-objective-diagnostics",
            "anchored-pc-residual-objective-decision",
        ),
        default="pinned-support",
        help="Decision report to write.",
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        help="Completed comparison directory. Defaults depend on --report.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        help=(
            "Config path to inspect for reports that validate promoted defaults. "
            "Defaults depend on --report."
        ),
    )
    parser.add_argument(
        "--artifact-check",
        type=Path,
        help="Optional existing artifact check JSON to use as evidence.",
    )
    parser.add_argument(
        "--decision-report",
        action="append",
        type=Path,
        help=(
            "Decision report JSON to include in an aggregate report. Repeat for "
            "multiple reports. Defaults are used only for aggregate reports."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Directory for decision_report.json and decision_report.md.",
    )
    parser.add_argument(
        "--max-logit-delta",
        default=DEFAULT_MAX_LOGIT_DELTA,
        type=float,
        help="Ordinary-logit delta budget for promoting pinned support.",
    )
    parser.add_argument(
        "--max-pinned-vs-repicked-delta",
        default=DEFAULT_MAX_PINNED_VS_REPICKED_DELTA,
        type=float,
        help="Pinned-vs-repicked delta budget for promoting clipped HEP.",
    )
    args = parser.parse_args()
    if args.report == "clipped-hep":
        report = write_clipped_hep_decision_report(
            args.comparison_dir or DEFAULT_CLIPPED_COMPARISON_DIR,
            args.out or DEFAULT_CLIPPED_OUT_DIR,
            artifact_check_path=args.artifact_check,
            max_logit_delta=args.max_logit_delta,
            max_pinned_vs_repicked_delta=args.max_pinned_vs_repicked_delta,
        )
    elif args.report == "guided-clipped-hep":
        report = write_guided_clipped_hep_decision_report(
            args.comparison_dir or DEFAULT_GUIDED_CLIPPED_COMPARISON_DIR,
            args.out or DEFAULT_GUIDED_CLIPPED_OUT_DIR,
            artifact_check_path=args.artifact_check,
            max_logit_delta=args.max_logit_delta,
            max_pinned_vs_repicked_delta=args.max_pinned_vs_repicked_delta,
        )
    elif args.report == "temporal-clipped-hep":
        report = write_temporal_clipped_hep_decision_report(
            args.comparison_dir or DEFAULT_TEMPORAL_CLIPPED_COMPARISON_DIR,
            args.out or DEFAULT_TEMPORAL_CLIPPED_OUT_DIR,
            artifact_check_path=args.artifact_check,
            max_logit_delta=args.max_logit_delta,
            max_pinned_vs_repicked_delta=args.max_pinned_vs_repicked_delta,
        )
    elif args.report == "temporal-clipped-hep-aggregate":
        report = write_temporal_clipped_hep_aggregate_report(
            args.decision_report or DEFAULT_TEMPORAL_CLIPPED_AGGREGATE_REPORTS,
            args.out or DEFAULT_TEMPORAL_CLIPPED_AGGREGATE_OUT_DIR,
            max_logit_delta=args.max_logit_delta,
            max_pinned_vs_repicked_delta=args.max_pinned_vs_repicked_delta,
        )
    elif args.report == "temporal-clipped-hep-cross-scale-aggregate":
        report = write_temporal_clipped_hep_cross_scale_aggregate_report(
            args.decision_report
            or DEFAULT_TEMPORAL_CLIPPED_CROSS_SCALE_AGGREGATE_REPORTS,
            args.out or DEFAULT_TEMPORAL_CLIPPED_CROSS_SCALE_AGGREGATE_OUT_DIR,
            max_logit_delta=args.max_logit_delta,
            max_pinned_vs_repicked_delta=args.max_pinned_vs_repicked_delta,
        )
    elif args.report == "temporal-clipped-hep-promotion-gate":
        report = write_temporal_clipped_hep_promotion_gate_report(
            args.decision_report[0]
            if args.decision_report
            else DEFAULT_TEMPORAL_CLIPPED_PROMOTION_GATE_CROSS_SCALE_REPORT,
            args.out or DEFAULT_TEMPORAL_CLIPPED_PROMOTION_GATE_OUT_DIR,
        )
    elif args.report == "temporal-clipped-hep-promotion-gate-satisfaction":
        decision_reports = (
            args.decision_report[1:]
            if args.decision_report and len(args.decision_report) > 1
            else DEFAULT_TEMPORAL_CLIPPED_PROMOTION_GATE_SATISFACTION_REPORTS
        )
        report = write_temporal_clipped_hep_promotion_gate_satisfaction_report(
            args.decision_report[0]
            if args.decision_report
            else DEFAULT_TEMPORAL_CLIPPED_PROMOTION_GATE_SATISFACTION_REPORT,
            decision_reports,
            args.out or DEFAULT_TEMPORAL_CLIPPED_PROMOTION_GATE_SATISFACTION_OUT_DIR,
            max_logit_delta=args.max_logit_delta,
            max_pinned_vs_repicked_delta=args.max_pinned_vs_repicked_delta,
        )
    elif args.report == "post-promotion-residual-learning-gate":
        report = write_post_promotion_residual_learning_gate_report(
            args.decision_report[0]
            if args.decision_report
            else DEFAULT_POST_PROMOTION_GATE_SATISFACTION_REPORT,
            args.config_path or DEFAULT_POST_PROMOTION_GATE_CONFIG,
            args.out or DEFAULT_POST_PROMOTION_GATE_OUT_DIR,
        )
    elif args.report == "residual-objective-gate":
        report = write_residual_objective_gate_decision_report(
            tuple(args.decision_report)
            if args.decision_report
            else DEFAULT_RESIDUAL_OBJECTIVE_GATE_COMPARISON_DIRS,
            args.out or DEFAULT_RESIDUAL_OBJECTIVE_GATE_OUT_DIR,
            artifact_check_paths=DEFAULT_RESIDUAL_OBJECTIVE_GATE_ARTIFACT_CHECKS
            if not args.artifact_check
            else (args.artifact_check,),
            max_logit_delta=args.max_logit_delta,
        )
    elif args.report == "pc-residual-objective-diagnostics":
        report = write_pc_residual_objective_diagnostics_report(
            tuple(args.decision_report)
            if args.decision_report
            else DEFAULT_RESIDUAL_OBJECTIVE_GATE_COMPARISON_DIRS,
            args.out or DEFAULT_PC_RESIDUAL_OBJECTIVE_DIAGNOSTICS_OUT_DIR,
            artifact_check_paths=DEFAULT_RESIDUAL_OBJECTIVE_GATE_ARTIFACT_CHECKS
            if not args.artifact_check
            else (args.artifact_check,),
            max_logit_delta=args.max_logit_delta,
        )
    elif args.report == "anchored-pc-residual-objective-decision":
        report = write_anchored_pc_residual_objective_decision_report(
            tuple(args.decision_report)
            if args.decision_report
            else DEFAULT_ANCHORED_PC_RESIDUAL_OBJECTIVE_COMPARISON_DIRS,
            args.out or DEFAULT_ANCHORED_PC_RESIDUAL_OBJECTIVE_OUT_DIR,
            artifact_check_paths=DEFAULT_ANCHORED_PC_RESIDUAL_OBJECTIVE_ARTIFACT_CHECKS
            if not args.artifact_check
            else (args.artifact_check,),
            max_logit_delta=args.max_logit_delta,
        )
    else:
        report = write_pinned_support_decision_report(
            args.comparison_dir or DEFAULT_COMPARISON_DIR,
            args.out or DEFAULT_OUT_DIR,
            artifact_check_path=args.artifact_check,
            max_logit_delta=args.max_logit_delta,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
