"""Make local HEP mechanism decisions from completed comparison artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

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
DEFAULT_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_COMPARISON_DIRS = (
    Path("results/comparisons/validation_confidence_penalty_temporal_clipped_objective_gate"),
    Path(
        "results/comparisons/colab_validation_confidence_penalty_temporal_clipped_objective_gate"
    ),
)
DEFAULT_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_ARTIFACT_CHECKS = (
    DEFAULT_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[0]
    / "artifact_check_local.json",
    DEFAULT_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[1]
    / "artifact_check_local.json",
)
DEFAULT_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_OUT_DIR = Path(
    "results/reports/confidence_penalty_residual_objective_decision"
)
DEFAULT_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_COMPARISON_DIRS = (
    Path("results/comparisons/validation_margin_penalty_temporal_clipped_objective_gate"),
    Path(
        "results/comparisons/colab_validation_margin_penalty_temporal_clipped_objective_gate"
    ),
)
DEFAULT_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_ARTIFACT_CHECKS = (
    DEFAULT_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[0]
    / "artifact_check_local.json",
    DEFAULT_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[1]
    / "artifact_check_local.json",
)
DEFAULT_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_OUT_DIR = Path(
    "results/reports/margin_penalty_residual_objective_decision"
)
DEFAULT_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_COMPARISON_DIRS = (
    Path("results/comparisons/validation_label_smoothing_temporal_clipped_objective_gate"),
    Path(
        "results/comparisons/colab_validation_label_smoothing_temporal_clipped_objective_gate"
    ),
)
DEFAULT_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_ARTIFACT_CHECKS = (
    DEFAULT_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[0]
    / "artifact_check_local.json",
    DEFAULT_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[1]
    / "artifact_check_local.json",
)
DEFAULT_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_OUT_DIR = Path(
    "results/reports/label_smoothing_residual_objective_decision"
)
DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_COMPARISON_DIRS = (
    Path("results/comparisons/validation_focal_temporal_clipped_objective_gate"),
    Path("results/comparisons/colab_validation_focal_temporal_clipped_objective_gate"),
    Path("results/comparisons/extended_focal_temporal_clipped_objective_gate"),
    Path("results/comparisons/colab_extended_focal_temporal_clipped_objective_gate"),
    Path("results/comparisons/larger_focal_temporal_clipped_objective_gate"),
    Path("results/comparisons/colab_larger_focal_temporal_clipped_objective_gate"),
    Path("results/comparisons/token_larger_focal_temporal_clipped_objective_gate"),
    Path(
        "results/comparisons/colab_token_larger_focal_temporal_clipped_objective_gate"
    ),
    Path("results/comparisons/char_xlarge_focal_temporal_clipped_objective_gate"),
    Path("results/comparisons/colab_char_xlarge_focal_temporal_clipped_objective_gate"),
    Path("results/comparisons/char_xxlarge_focal_temporal_clipped_objective_gate"),
    Path("results/comparisons/colab_char_xxlarge_focal_temporal_clipped_objective_gate"),
)
DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_ARTIFACT_CHECKS = (
    DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[0]
    / "artifact_check_local.json",
    DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[1]
    / "artifact_check_local.json",
    DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[2]
    / "artifact_check_local.json",
    DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[3]
    / "artifact_check_local.json",
    DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[4]
    / "artifact_check_local.json",
    DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[5]
    / "artifact_check_local.json",
    DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[6]
    / "artifact_check_local.json",
    DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[7]
    / "artifact_check_local.json",
    DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[8]
    / "artifact_check_local.json",
    DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[9]
    / "artifact_check_local.json",
    DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[10]
    / "artifact_check_local.json",
    DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[11]
    / "artifact_check_local.json",
)
DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_OUT_DIR = Path(
    "results/reports/focal_residual_objective_decision"
)
DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_DECISION_REPORT = (
    DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_OUT_DIR / "decision_report.json"
)
DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_OUT_DIR = Path(
    "results/reports/focal_residual_objective_promotion_gate"
)
DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_SATISFACTION_REPORT = (
    DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_OUT_DIR / "decision_report.json"
)
DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_SATISFACTION_COMPARISON_DIRS = (
    Path("results/comparisons/char_xxlarge_focal_temporal_clipped_objective_gate_seed2"),
    Path(
        "results/comparisons/colab_char_xxlarge_focal_temporal_clipped_objective_gate_seed2"
    ),
    Path("results/comparisons/token_larger_focal_temporal_clipped_objective_gate_seed2"),
    Path(
        "results/comparisons/colab_token_larger_focal_temporal_clipped_objective_gate_seed2"
    ),
)
DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_SATISFACTION_ARTIFACT_CHECKS = (
    DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_SATISFACTION_COMPARISON_DIRS[0]
    / "artifact_check_local.json",
    DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_SATISFACTION_COMPARISON_DIRS[1]
    / "artifact_check_local.json",
    DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_SATISFACTION_COMPARISON_DIRS[2]
    / "artifact_check_local.json",
    DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_SATISFACTION_COMPARISON_DIRS[3]
    / "artifact_check_local.json",
)
DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_SATISFACTION_OUT_DIR = Path(
    "results/reports/focal_residual_objective_promotion_gate_satisfaction"
)
DEFAULT_TEMPORAL_CONSISTENCY_RESIDUAL_OBJECTIVE_COMPARISON_DIRS = (
    Path("results/comparisons/validation_temporal_consistency_weight_sweep_temporal_clipped_objective_gate"),
    Path("results/comparisons/extended_temporal_consistency_weight_sweep_temporal_clipped_objective_gate"),
)
DEFAULT_TEMPORAL_CONSISTENCY_RESIDUAL_OBJECTIVE_ARTIFACT_CHECKS = (
    DEFAULT_TEMPORAL_CONSISTENCY_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[0]
    / "artifact_check_local.json",
    DEFAULT_TEMPORAL_CONSISTENCY_RESIDUAL_OBJECTIVE_COMPARISON_DIRS[1]
    / "artifact_check_local.json",
)
DEFAULT_TEMPORAL_CONSISTENCY_RESIDUAL_OBJECTIVE_OUT_DIR = Path(
    "results/reports/temporal_consistency_residual_objective_decision"
)
DEFAULT_RESIDUAL_LEARNING_NEXT_DIRECTION_REPORTS = (
    Path("results/reports/residual_objective_gate_decision/decision_report.json"),
    Path("results/reports/anchored_pc_residual_objective_decision/decision_report.json"),
    Path(
        "results/reports/confidence_penalty_residual_objective_decision/decision_report.json"
    ),
    Path("results/reports/margin_penalty_residual_objective_decision/decision_report.json"),
    Path(
        "results/reports/label_smoothing_residual_objective_decision/decision_report.json"
    ),
    Path(
        "results/reports/focal_residual_objective_promotion_gate_satisfaction/decision_report.json"
    ),
    Path(
        "results/reports/temporal_consistency_residual_objective_decision/decision_report.json"
    ),
)
DEFAULT_RESIDUAL_LEARNING_NEXT_DIRECTION_OUT_DIR = Path(
    "results/reports/residual_learning_next_direction"
)
DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_GATE_REPORT = (
    DEFAULT_RESIDUAL_LEARNING_NEXT_DIRECTION_OUT_DIR / "decision_report.json"
)
DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_GATE_CONFIGS = (
    Path("configs/char_validation_hep_temporal_clipped_objective_gate.yaml"),
    Path(
        "configs/char_validation_capacity_hep_temporal_clipped_objective_gate.yaml"
    ),
    Path(
        "configs/char_validation_support_wide_hep_temporal_clipped_objective_gate.yaml"
    ),
    Path(
        "configs/char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate.yaml"
    ),
)
DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_GATE_OUT_DIR = Path(
    "results/reports/residual_capacity_support_diagnostic_gate"
)
DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_COMPARISON_DIR = Path(
    "results/comparisons/validation_residual_capacity_support_temporal_clipped_objective_gate"
)
DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_ARTIFACT_CHECK = (
    DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_COMPARISON_DIR
    / "artifact_check_local.json"
)
DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_DECISION_OUT_DIR = Path(
    "results/reports/residual_capacity_support_diagnostic_decision"
)
DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_COLAB_COMPARISON_DIRS = (
    DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_COMPARISON_DIR,
    Path(
        "results/comparisons/colab_validation_residual_capacity_support_temporal_clipped_objective_gate"
    ),
)
DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_COLAB_ARTIFACT_CHECKS = (
    DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_COMPARISON_DIR
    / "artifact_check_local.json",
    Path(
        "results/comparisons/colab_validation_residual_capacity_support_temporal_clipped_objective_gate"
    )
    / "artifact_check_local.json",
)
DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_COLAB_DECISION_OUT_DIR = Path(
    "results/reports/residual_capacity_support_diagnostic_colab_decision"
)
DEFAULT_RESIDUAL_SUPPORT_WIDTH_VALIDATION_GATE_REPORT = (
    DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_COLAB_DECISION_OUT_DIR
    / "decision_report.json"
)
DEFAULT_RESIDUAL_SUPPORT_WIDTH_VALIDATION_GATE_CONFIGS = (
    Path("configs/char_larger_hep_temporal_clipped_objective_gate.yaml"),
    Path("configs/char_larger_support_wide_hep_temporal_clipped_objective_gate.yaml"),
    Path("configs/token_larger_hep_temporal_clipped_objective_gate.yaml"),
    Path("configs/token_larger_support_wide_hep_temporal_clipped_objective_gate.yaml"),
)
DEFAULT_RESIDUAL_SUPPORT_WIDTH_VALIDATION_GATE_OUT_DIR = Path(
    "results/reports/residual_support_width_validation_gate"
)
DEFAULT_RESIDUAL_SUPPORT_WIDTH_VALIDATION_COMPARISON_DIRS = (
    Path(
        "results/comparisons/support_width_larger_char_token_temporal_clipped_objective_gate"
    ),
    Path(
        "results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate"
    ),
)
DEFAULT_RESIDUAL_SUPPORT_WIDTH_VALIDATION_ARTIFACT_CHECKS = (
    DEFAULT_RESIDUAL_SUPPORT_WIDTH_VALIDATION_COMPARISON_DIRS[0]
    / "artifact_check_local.json",
    DEFAULT_RESIDUAL_SUPPORT_WIDTH_VALIDATION_COMPARISON_DIRS[1]
    / "artifact_check_local.json",
)
DEFAULT_RESIDUAL_SUPPORT_WIDTH_VALIDATION_DECISION_OUT_DIR = Path(
    "results/reports/residual_support_width_validation_decision"
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
CONTINUE_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION = (
    "continue_confidence_penalty_residual_objective_validation"
)
CONTINUE_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION = (
    "continue_margin_penalty_residual_objective_validation"
)
CONTINUE_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_VALIDATION = (
    "continue_label_smoothing_residual_objective_validation"
)
CONTINUE_FOCAL_RESIDUAL_OBJECTIVE_VALIDATION = (
    "continue_focal_residual_objective_validation"
)
DEFINE_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE = (
    "define_focal_residual_objective_promotion_gate"
)
SATISFY_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE = (
    "satisfy_focal_residual_objective_promotion_gate"
)
DIAGNOSE_PC_RESIDUAL_OBJECTIVE = "diagnose_pc_residual_objective_gap"
STOP_PC_RESIDUAL_OBJECTIVE_VALIDATION = "stop_pc_residual_objective_validation"
STOP_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION = (
    "stop_confidence_penalty_residual_objective_validation"
)
STOP_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION = (
    "stop_margin_penalty_residual_objective_validation"
)
STOP_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_VALIDATION = (
    "stop_label_smoothing_residual_objective_validation"
)
STOP_FOCAL_RESIDUAL_OBJECTIVE_VALIDATION = "stop_focal_residual_objective_validation"
STOP_TEMPORAL_CONSISTENCY_RESIDUAL_OBJECTIVE_VALIDATION = (
    "stop_temporal_consistency_residual_objective_validation"
)
CONTINUE_TEMPORAL_CONSISTENCY_RESIDUAL_OBJECTIVE_VALIDATION = (
    "continue_temporal_consistency_residual_objective_validation"
)
DEFINE_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_GATE = (
    "define_residual_capacity_support_diagnostic_gate"
)
RUN_COLAB_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC = (
    "run_colab_residual_capacity_support_diagnostic"
)
CONTINUE_RESIDUAL_CAPACITY_SUPPORT_VALIDATION = (
    "continue_residual_capacity_support_validation"
)
DEFINE_RESIDUAL_SUPPORT_WIDTH_VALIDATION_GATE = (
    "define_residual_support_width_validation_gate"
)
CONTINUE_RESIDUAL_SUPPORT_WIDTH_VALIDATION = (
    "continue_residual_support_width_validation"
)
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


def write_confidence_penalty_residual_objective_decision_report(
    comparison_dirs: list[Path] | tuple[Path, ...] = (
        DEFAULT_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_COMPARISON_DIRS
    ),
    out_dir: Path = DEFAULT_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_OUT_DIR,
    *,
    artifact_check_paths: list[Path] | tuple[Path, ...] | None = (
        DEFAULT_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_ARTIFACT_CHECKS
    ),
    max_logit_delta: float = DEFAULT_MAX_LOGIT_DELTA,
) -> dict[str, Any]:
    """Decide whether confidence-penalty CE merits more objective validation."""

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
        entry = _confidence_penalty_residual_objective_entry(
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
    confidence_runs = [
        entry["confidence_penalty_run"]
        for entry in entries
        if entry.get("confidence_penalty_run")
    ]
    confidence_ce_wins = [
        entry
        for entry in entries
        if entry.get("supervised_run")
        and entry.get("confidence_penalty_run")
        and entry["confidence_penalty_run"]["best_hep_loss"] is not None
        and entry["supervised_run"]["best_hep_loss"] is not None
        and entry["confidence_penalty_run"]["best_hep_loss"]
        < entry["supervised_run"]["best_hep_loss"]
    ]
    confidence_minus_supervised = [
        entry["confidence_penalty_minus_supervised_best_hep_loss"]
        for entry in entries
        if entry.get("confidence_penalty_minus_supervised_best_hep_loss") is not None
    ]
    residual_loss_deltas = [
        entry["confidence_penalty_minus_supervised_final_residual_loss"]
        for entry in entries
        if entry.get("confidence_penalty_minus_supervised_final_residual_loss")
        is not None
    ]
    status = "fail" if failures else "pass"
    continue_variant = status == "pass" and bool(confidence_ce_wins)
    report = {
        "status": status,
        "decision": (
            CONTINUE_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION
            if continue_variant
            else (
                STOP_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION
                if status == "pass"
                else INSUFFICIENT_EVIDENCE
            )
        ),
        "continue_confidence_penalty_residual_objective_validation": continue_variant,
        "selected_residual_objective_variant": (
            "supervised_ce_confidence_penalty" if continue_variant else None
        ),
        "promote_residual_learning_method": False,
        "default_residual_objective": "supervised_ce",
        "policy": {
            "max_logit_delta_from_ordinary": max_logit_delta,
            "requires_local_and_colab_evidence": True,
            "requires_passing_artifact_checks": True,
            "requires_supervised_and_confidence_penalty_runs": True,
            "requires_support_stress_preset_disabled": True,
            "requires_temporal_clipped_hep_path": True,
            "requires_both_objectives_improve_own_training_loss": True,
            "requires_confidence_penalty_lower_supervised_ce_hep_loss_to_continue": True,
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
            "confidence_penalty_run_count": len(confidence_runs),
            "confidence_penalty_ce_win_count": len(confidence_ce_wins),
            "mean_confidence_penalty_minus_supervised_best_hep_loss": (
                sum(confidence_minus_supervised) / len(confidence_minus_supervised)
                if confidence_minus_supervised
                else None
            ),
            "mean_confidence_penalty_minus_supervised_final_residual_loss": (
                sum(residual_loss_deltas) / len(residual_loss_deltas)
                if residual_loss_deltas
                else None
            ),
            "entries": entries,
            "failures": failures,
        },
        "rationale": (
            "The confidence-penalty objective improves its own residual training "
            "loss but does not beat supervised CE on best temporal-clipped HEP "
            "supervised loss in the checked local and Colab artifacts. It should "
            "not continue under the current objective gate."
            if status == "pass" and not continue_variant
            else (
                "The confidence-penalty objective beats supervised CE HEP loss in "
                "at least one artifact-backed backend, so it merits broader "
                "objective validation before any default change."
                if status == "pass"
                else (
                    "The confidence-penalty decision requires matching local and "
                    "Colab comparisons with passing artifact checks and valid "
                    "supervised and confidence-penalty temporal-clipped runs."
                )
            )
        ),
        "next_step": (
            "select the next non-PC residual objective variant to test under the objective gate"
            if status == "pass" and not continue_variant
            else (
                "run a broader confidence-penalty objective comparison outside the current char validation setting"
                if status == "pass"
                else "repair or regenerate the confidence-penalty objective comparison artifacts"
            )
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_confidence_penalty_residual_objective_markdown(
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


def _confidence_penalty_residual_objective_entry(
    comparison_dir: Path,
    *,
    artifact_check_path: Path | None,
    max_logit_delta: float,
) -> dict[str, Any]:
    entry = _residual_objective_variant_entry(
        comparison_dir,
        variant_objective="supervised_ce_confidence_penalty",
        variant_field="confidence_penalty_run",
        missing_field="comparison.runs.supervised_ce_confidence_penalty",
        artifact_check_path=artifact_check_path,
        max_logit_delta=max_logit_delta,
    )
    supervised = entry.get("supervised_run")
    confidence = entry.get("confidence_penalty_run")
    entry["confidence_penalty_minus_supervised_best_hep_loss"] = _best_loss_delta(
        confidence,
        supervised,
    )
    entry["confidence_penalty_minus_supervised_final_residual_loss"] = (
        None
        if not isinstance(confidence, dict)
        or not isinstance(supervised, dict)
        or confidence.get("final_residual_loss") is None
        or supervised.get("final_residual_loss") is None
        else float(confidence["final_residual_loss"])
        - float(supervised["final_residual_loss"])
    )
    return entry


def write_margin_penalty_residual_objective_decision_report(
    comparison_dirs: list[Path] | tuple[Path, ...] = (
        DEFAULT_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_COMPARISON_DIRS
    ),
    out_dir: Path = DEFAULT_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_OUT_DIR,
    *,
    artifact_check_paths: list[Path] | tuple[Path, ...] | None = (
        DEFAULT_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_ARTIFACT_CHECKS
    ),
    max_logit_delta: float = DEFAULT_MAX_LOGIT_DELTA,
) -> dict[str, Any]:
    """Decide whether margin-penalty CE merits more objective validation."""

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
        entry = _margin_penalty_residual_objective_entry(
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
    margin_runs = [
        entry["margin_penalty_run"]
        for entry in entries
        if entry.get("margin_penalty_run")
    ]
    margin_ce_wins = [
        entry
        for entry in entries
        if entry.get("supervised_run")
        and entry.get("margin_penalty_run")
        and entry["margin_penalty_run"]["best_hep_loss"] is not None
        and entry["supervised_run"]["best_hep_loss"] is not None
        and entry["margin_penalty_run"]["best_hep_loss"]
        < entry["supervised_run"]["best_hep_loss"]
    ]
    margin_minus_supervised = [
        entry["margin_penalty_minus_supervised_best_hep_loss"]
        for entry in entries
        if entry.get("margin_penalty_minus_supervised_best_hep_loss") is not None
    ]
    residual_loss_deltas = [
        entry["margin_penalty_minus_supervised_final_residual_loss"]
        for entry in entries
        if entry.get("margin_penalty_minus_supervised_final_residual_loss") is not None
    ]
    status = "fail" if failures else "pass"
    continue_variant = status == "pass" and bool(margin_ce_wins)
    report = {
        "status": status,
        "decision": (
            CONTINUE_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION
            if continue_variant
            else (
                STOP_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION
                if status == "pass"
                else INSUFFICIENT_EVIDENCE
            )
        ),
        "continue_margin_penalty_residual_objective_validation": continue_variant,
        "selected_residual_objective_variant": (
            "supervised_ce_margin_penalty" if continue_variant else None
        ),
        "promote_residual_learning_method": False,
        "default_residual_objective": "supervised_ce",
        "policy": {
            "max_logit_delta_from_ordinary": max_logit_delta,
            "requires_local_and_colab_evidence": True,
            "requires_passing_artifact_checks": True,
            "requires_supervised_and_margin_penalty_runs": True,
            "requires_support_stress_preset_disabled": True,
            "requires_temporal_clipped_hep_path": True,
            "requires_both_objectives_improve_own_training_loss": True,
            "requires_margin_penalty_lower_supervised_ce_hep_loss_to_continue": True,
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
            "margin_penalty_run_count": len(margin_runs),
            "margin_penalty_ce_win_count": len(margin_ce_wins),
            "mean_margin_penalty_minus_supervised_best_hep_loss": (
                sum(margin_minus_supervised) / len(margin_minus_supervised)
                if margin_minus_supervised
                else None
            ),
            "mean_margin_penalty_minus_supervised_final_residual_loss": (
                sum(residual_loss_deltas) / len(residual_loss_deltas)
                if residual_loss_deltas
                else None
            ),
            "entries": entries,
            "failures": failures,
        },
        "rationale": (
            "The margin-penalty objective improves its own residual training "
            "loss but does not beat supervised CE on best temporal-clipped HEP "
            "supervised loss in the checked local and Colab artifacts. It should "
            "not continue under the current objective gate."
            if status == "pass" and not continue_variant
            else (
                "The margin-penalty objective beats supervised CE HEP loss in "
                "at least one artifact-backed backend, so it merits broader "
                "objective validation before any default change."
                if status == "pass"
                else (
                    "The margin-penalty decision requires matching local and "
                    "Colab comparisons with passing artifact checks and valid "
                    "supervised and margin-penalty temporal-clipped runs."
                )
            )
        ),
        "next_step": (
            "select the next non-PC residual objective variant to test under the objective gate"
            if status == "pass" and not continue_variant
            else (
                "run a broader margin-penalty objective comparison outside the current char validation setting"
                if status == "pass"
                else "repair or regenerate the margin-penalty objective comparison artifacts"
            )
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_margin_penalty_residual_objective_markdown(
        out_dir / "decision_report.md",
        report,
    )
    return report


def _margin_penalty_residual_objective_entry(
    comparison_dir: Path,
    *,
    artifact_check_path: Path | None,
    max_logit_delta: float,
) -> dict[str, Any]:
    entry = _residual_objective_variant_entry(
        comparison_dir,
        variant_objective="supervised_ce_margin_penalty",
        variant_field="margin_penalty_run",
        missing_field="comparison.runs.supervised_ce_margin_penalty",
        artifact_check_path=artifact_check_path,
        max_logit_delta=max_logit_delta,
    )
    supervised = entry.get("supervised_run")
    margin = entry.get("margin_penalty_run")
    entry["margin_penalty_minus_supervised_best_hep_loss"] = _best_loss_delta(
        margin,
        supervised,
    )
    entry["margin_penalty_minus_supervised_final_residual_loss"] = (
        None
        if not isinstance(margin, dict)
        or not isinstance(supervised, dict)
        or margin.get("final_residual_loss") is None
        or supervised.get("final_residual_loss") is None
        else float(margin["final_residual_loss"])
        - float(supervised["final_residual_loss"])
    )
    return entry


def write_label_smoothing_residual_objective_decision_report(
    comparison_dirs: list[Path] | tuple[Path, ...] = (
        DEFAULT_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_COMPARISON_DIRS
    ),
    out_dir: Path = DEFAULT_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_OUT_DIR,
    *,
    artifact_check_paths: list[Path] | tuple[Path, ...] | None = (
        DEFAULT_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_ARTIFACT_CHECKS
    ),
    max_logit_delta: float = DEFAULT_MAX_LOGIT_DELTA,
) -> dict[str, Any]:
    """Decide whether label-smoothing CE merits more objective validation."""

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
        entry = _label_smoothing_residual_objective_entry(
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
    label_smoothing_runs = [
        entry["label_smoothing_run"]
        for entry in entries
        if entry.get("label_smoothing_run")
    ]
    label_smoothing_ce_wins = [
        entry
        for entry in entries
        if entry.get("supervised_run")
        and entry.get("label_smoothing_run")
        and entry["label_smoothing_run"]["best_hep_loss"] is not None
        and entry["supervised_run"]["best_hep_loss"] is not None
        and entry["label_smoothing_run"]["best_hep_loss"]
        < entry["supervised_run"]["best_hep_loss"]
    ]
    label_smoothing_minus_supervised = [
        entry["label_smoothing_minus_supervised_best_hep_loss"]
        for entry in entries
        if entry.get("label_smoothing_minus_supervised_best_hep_loss") is not None
    ]
    residual_loss_deltas = [
        entry["label_smoothing_minus_supervised_final_residual_loss"]
        for entry in entries
        if entry.get("label_smoothing_minus_supervised_final_residual_loss")
        is not None
    ]
    status = "fail" if failures else "pass"
    continue_variant = status == "pass" and bool(label_smoothing_ce_wins)
    report = {
        "status": status,
        "decision": (
            CONTINUE_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_VALIDATION
            if continue_variant
            else (
                STOP_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_VALIDATION
                if status == "pass"
                else INSUFFICIENT_EVIDENCE
            )
        ),
        "continue_label_smoothing_residual_objective_validation": continue_variant,
        "selected_residual_objective_variant": (
            "supervised_ce_label_smoothing" if continue_variant else None
        ),
        "promote_residual_learning_method": False,
        "default_residual_objective": "supervised_ce",
        "policy": {
            "max_logit_delta_from_ordinary": max_logit_delta,
            "requires_local_and_colab_evidence": True,
            "requires_passing_artifact_checks": True,
            "requires_supervised_and_label_smoothing_runs": True,
            "requires_support_stress_preset_disabled": True,
            "requires_temporal_clipped_hep_path": True,
            "requires_both_objectives_improve_own_training_loss": True,
            "requires_label_smoothing_lower_supervised_ce_hep_loss_to_continue": True,
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
            "label_smoothing_run_count": len(label_smoothing_runs),
            "label_smoothing_ce_win_count": len(label_smoothing_ce_wins),
            "mean_label_smoothing_minus_supervised_best_hep_loss": (
                sum(label_smoothing_minus_supervised)
                / len(label_smoothing_minus_supervised)
                if label_smoothing_minus_supervised
                else None
            ),
            "mean_label_smoothing_minus_supervised_final_residual_loss": (
                sum(residual_loss_deltas) / len(residual_loss_deltas)
                if residual_loss_deltas
                else None
            ),
            "entries": entries,
            "failures": failures,
        },
        "rationale": (
            "The label-smoothing objective improves its own residual training "
            "loss but does not beat supervised CE on best temporal-clipped HEP "
            "supervised loss in the checked local and Colab artifacts. It should "
            "not continue under the current objective gate."
            if status == "pass" and not continue_variant
            else (
                "The label-smoothing objective beats supervised CE HEP loss in "
                "at least one artifact-backed backend, so it merits broader "
                "objective validation before any default change."
                if status == "pass"
                else (
                    "The label-smoothing decision requires matching local and "
                    "Colab comparisons with passing artifact checks and valid "
                    "supervised and label-smoothing temporal-clipped runs."
                )
            )
        ),
        "next_step": (
            "select the next non-PC residual objective variant to test under the objective gate"
            if status == "pass" and not continue_variant
            else (
                "run a broader label-smoothing objective comparison outside the current char validation setting"
                if status == "pass"
                else "repair or regenerate the label-smoothing objective comparison artifacts"
            )
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_label_smoothing_residual_objective_markdown(
        out_dir / "decision_report.md",
        report,
    )
    return report


def _label_smoothing_residual_objective_entry(
    comparison_dir: Path,
    *,
    artifact_check_path: Path | None,
    max_logit_delta: float,
) -> dict[str, Any]:
    entry = _residual_objective_variant_entry(
        comparison_dir,
        variant_objective="supervised_ce_label_smoothing",
        variant_field="label_smoothing_run",
        missing_field="comparison.runs.supervised_ce_label_smoothing",
        artifact_check_path=artifact_check_path,
        max_logit_delta=max_logit_delta,
    )
    supervised = entry.get("supervised_run")
    label_smoothing = entry.get("label_smoothing_run")
    entry["label_smoothing_minus_supervised_best_hep_loss"] = _best_loss_delta(
        label_smoothing,
        supervised,
    )
    entry["label_smoothing_minus_supervised_final_residual_loss"] = (
        None
        if not isinstance(label_smoothing, dict)
        or not isinstance(supervised, dict)
        or label_smoothing.get("final_residual_loss") is None
        or supervised.get("final_residual_loss") is None
        else float(label_smoothing["final_residual_loss"])
        - float(supervised["final_residual_loss"])
    )
    return entry


def write_focal_residual_objective_decision_report(
    comparison_dirs: list[Path] | tuple[Path, ...] = (
        DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_COMPARISON_DIRS
    ),
    out_dir: Path = DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_OUT_DIR,
    *,
    artifact_check_paths: list[Path] | tuple[Path, ...] | None = (
        DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_ARTIFACT_CHECKS
    ),
    max_logit_delta: float = DEFAULT_MAX_LOGIT_DELTA,
) -> dict[str, Any]:
    """Decide whether focal CE merits more objective validation."""

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
        entry = _focal_residual_objective_entry(
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
    focal_runs = [entry["focal_run"] for entry in entries if entry.get("focal_run")]
    focal_ce_wins = [
        entry
        for entry in entries
        if entry.get("supervised_run")
        and entry.get("focal_run")
        and entry["focal_run"]["best_hep_loss"] is not None
        and entry["supervised_run"]["best_hep_loss"] is not None
        and entry["focal_run"]["best_hep_loss"]
        < entry["supervised_run"]["best_hep_loss"]
    ]
    focal_minus_supervised = [
        entry["focal_minus_supervised_best_hep_loss"]
        for entry in entries
        if entry.get("focal_minus_supervised_best_hep_loss") is not None
    ]
    residual_loss_deltas = [
        entry["focal_minus_supervised_final_residual_loss"]
        for entry in entries
        if entry.get("focal_minus_supervised_final_residual_loss") is not None
    ]
    status = "fail" if failures else "pass"
    continue_variant = status == "pass" and len(focal_ce_wins) == len(entries)
    report = {
        "status": status,
        "decision": (
            CONTINUE_FOCAL_RESIDUAL_OBJECTIVE_VALIDATION
            if continue_variant
            else (
                STOP_FOCAL_RESIDUAL_OBJECTIVE_VALIDATION
                if status == "pass"
                else INSUFFICIENT_EVIDENCE
            )
        ),
        "continue_focal_residual_objective_validation": continue_variant,
        "selected_residual_objective_variant": (
            "supervised_ce_focal" if continue_variant else None
        ),
        "promote_residual_learning_method": False,
        "default_residual_objective": "supervised_ce",
        "policy": {
            "max_logit_delta_from_ordinary": max_logit_delta,
            "requires_local_and_colab_evidence": True,
            "requires_passing_artifact_checks": True,
            "requires_supervised_and_focal_runs": True,
            "requires_support_stress_preset_disabled": True,
            "requires_temporal_clipped_hep_path": True,
            "requires_both_objectives_improve_own_training_loss": True,
            "requires_focal_lower_supervised_ce_hep_loss_in_every_comparison_to_continue": True,
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
            "focal_run_count": len(focal_runs),
            "focal_ce_win_count": len(focal_ce_wins),
            "mean_focal_minus_supervised_best_hep_loss": (
                sum(focal_minus_supervised) / len(focal_minus_supervised)
                if focal_minus_supervised
                else None
            ),
            "mean_focal_minus_supervised_final_residual_loss": (
                sum(residual_loss_deltas) / len(residual_loss_deltas)
                if residual_loss_deltas
                else None
            ),
            "entries": entries,
            "failures": failures,
        },
        "rationale": (
            "The focal objective improves its own residual training loss but "
            "does not beat supervised CE on best temporal-clipped HEP supervised "
            "loss in the checked local and Colab artifacts. It should not "
            "continue under the current objective gate."
            if status == "pass" and not continue_variant
            else (
                "The focal objective beats supervised CE HEP loss in every "
                "artifact-backed comparison, including the broader extended, "
                "larger, tokenized larger, xlarge, and xxlarge local and "
                "Colab checks, so it remains the selected objective variant "
                "for the next scale before any default change."
                if status == "pass"
                else (
                    "The focal decision requires matching local and Colab "
                    "comparisons with passing artifact checks and valid "
                    "supervised and focal temporal-clipped runs."
                )
            )
        ),
        "next_step": (
            "select the next non-PC residual objective variant to test under the objective gate"
            if status == "pass" and not continue_variant
            else (
                "define the next focal objective promotion or stop gate"
                if status == "pass"
                else "run or extract the missing matching focal objective comparison artifacts"
            )
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_focal_residual_objective_markdown(
        out_dir / "decision_report.md",
        report,
    )
    return report


def write_focal_residual_objective_promotion_gate_report(
    focal_decision_report_path: Path = (
        DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_DECISION_REPORT
    ),
    out_dir: Path = DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_OUT_DIR,
) -> dict[str, Any]:
    """Define the next promotion/stop gate after focal objective evidence."""

    failures = []
    focal_decision_report: dict[str, Any] | None = None
    if not focal_decision_report_path.is_file():
        failures.append(
            {
                "field": "focal_decision_report",
                "expected": "file exists",
                "actual": "missing",
                "path": str(focal_decision_report_path),
            }
        )
    else:
        focal_decision_report = _read_json_object(focal_decision_report_path)
        if focal_decision_report.get("status") != "pass":
            failures.append(
                {
                    "field": "focal_decision_report.status",
                    "expected": "pass",
                    "actual": focal_decision_report.get("status"),
                    "path": str(focal_decision_report_path),
                }
            )
        if (
            focal_decision_report.get("decision")
            != CONTINUE_FOCAL_RESIDUAL_OBJECTIVE_VALIDATION
        ):
            failures.append(
                {
                    "field": "focal_decision_report.decision",
                    "expected": CONTINUE_FOCAL_RESIDUAL_OBJECTIVE_VALIDATION,
                    "actual": focal_decision_report.get("decision"),
                    "path": str(focal_decision_report_path),
                }
            )
        if (
            focal_decision_report.get("selected_residual_objective_variant")
            != "supervised_ce_focal"
        ):
            failures.append(
                {
                    "field": (
                        "focal_decision_report."
                        "selected_residual_objective_variant"
                    ),
                    "expected": "supervised_ce_focal",
                    "actual": focal_decision_report.get(
                        "selected_residual_objective_variant"
                    ),
                    "path": str(focal_decision_report_path),
                }
            )

    evidence = (
        focal_decision_report.get("evidence", {})
        if isinstance(focal_decision_report, dict)
        and isinstance(focal_decision_report.get("evidence"), dict)
        else {}
    )
    required_evidence = [
        {
            "gate": "char_xxlarge_seed2_local_colab",
            "description": (
                "Repeat the xxlarge char objective-discriminative temporal "
                "clipped focal-vs-supervised comparison at seed 2."
            ),
            "minimum_scale": {
                "dataset": "tiny_shakespeare_char",
                "seq_len": 192,
                "hidden_dim": 160,
                "num_columns": 40,
                "pc_steps": 4,
                "training_steps": 70,
            },
            "required_backends": ["local", "colab"],
        },
        {
            "gate": "token_larger_seed2_local_colab",
            "description": (
                "Repeat the non-char tokenized larger objective gate at seed 2 "
                "so promotion is not based on one deterministic token seed."
            ),
            "minimum_scale": {
                "dataset": "tiny_shakespeare_word",
                "seq_len": 64,
                "hidden_dim": 96,
                "num_columns": 24,
                "pc_steps": 4,
                "training_steps": 50,
            },
            "required_backends": ["local", "colab"],
        },
    ]
    status = "fail" if failures else "pass"
    report = {
        "status": status,
        "decision": (
            DEFINE_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE
            if status == "pass"
            else INSUFFICIENT_EVIDENCE
        ),
        "selected_residual_objective_variant": (
            "supervised_ce_focal" if status == "pass" else None
        ),
        "promote_residual_learning_method": False,
        "default_residual_objective": "supervised_ce",
        "policy": {
            "requires_focal_decision_report_pass": True,
            "requires_existing_decision_to_continue_focal": True,
            "requires_each_gate_local_and_colab": True,
            "requires_passing_artifact_checks": True,
            "requires_support_stress_preset_disabled": True,
            "requires_temporal_clipped_hep_path": True,
            "requires_both_objectives_improve_own_training_loss": True,
            "requires_focal_lower_supervised_ce_hep_loss_in_every_gate_comparison": True,
            "requires_focal_best_hep_loss_mean_improvement_negative": True,
            "max_logit_delta_from_ordinary": DEFAULT_MAX_LOGIT_DELTA,
            "allows_default_promotion": False,
            "reason_default_promotion_is_blocked": (
                "This report defines the next promotion or stop gate. It does "
                "not satisfy that gate or change the default residual objective."
            ),
        },
        "evidence": {
            "focal_decision_report_path": str(focal_decision_report_path),
            "focal_decision_status": None
            if focal_decision_report is None
            else focal_decision_report.get("status"),
            "focal_decision": None
            if focal_decision_report is None
            else focal_decision_report.get("decision"),
            "focal_comparison_count": evidence.get("comparison_count"),
            "focal_backend_count": evidence.get("backend_count"),
            "focal_ce_win_count": evidence.get("focal_ce_win_count"),
            "mean_focal_minus_supervised_best_hep_loss": evidence.get(
                "mean_focal_minus_supervised_best_hep_loss"
            ),
            "mean_focal_minus_supervised_final_residual_loss": evidence.get(
                "mean_focal_minus_supervised_final_residual_loss"
            ),
            "required_evidence": required_evidence,
            "failures": failures,
        },
        "rationale": (
            "Focal CE has beaten supervised CE on best temporal-clipped HEP "
            "supervised CE loss across the current artifact-backed local and "
            "Colab validation, extended, larger, tokenized larger, xlarge, and "
            "xxlarge gates. The next gate should test seed robustness at the "
            "largest char scale and at the tokenized scale before any default "
            "residual objective change."
            if status == "pass"
            else (
                "The focal promotion/stop gate cannot be defined until the "
                "focal residual objective decision report is present, passing, "
                "and continuing supervised_ce_focal validation."
            )
        ),
        "next_step": (
            "add seed-2 focal objective-gate configs for xxlarge char and token larger settings"
            if status == "pass"
            else "repair or regenerate the focal residual objective decision report"
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_focal_promotion_gate_markdown(out_dir / "decision_report.md", report)
    return report


def write_focal_residual_objective_promotion_gate_satisfaction_report(
    promotion_gate_report_path: Path = (
        DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_SATISFACTION_REPORT
    ),
    comparison_dirs: list[Path] | tuple[Path, ...] = (
        DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_SATISFACTION_COMPARISON_DIRS
    ),
    out_dir: Path = (
        DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_SATISFACTION_OUT_DIR
    ),
    *,
    artifact_check_paths: list[Path] | tuple[Path, ...] | None = (
        DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_SATISFACTION_ARTIFACT_CHECKS
    ),
    max_logit_delta: float = DEFAULT_MAX_LOGIT_DELTA,
) -> dict[str, Any]:
    """Decide whether the focal objective promotion/stop gate is satisfied."""

    if max_logit_delta < 0.0:
        raise ValueError("max_logit_delta must be non-negative")

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
            != DEFINE_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE
        ):
            failures.append(
                {
                    "field": "promotion_gate_report.decision",
                    "expected": DEFINE_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE,
                    "actual": promotion_gate_report.get("decision"),
                    "path": str(promotion_gate_report_path),
                }
            )

    entries = []
    artifact_paths = list(artifact_check_paths or [])
    for index, comparison_dir in enumerate(comparison_dirs):
        comparison_dir = Path(comparison_dir)
        artifact_check_path = (
            Path(artifact_paths[index]) if index < len(artifact_paths) else None
        )
        entry = _focal_residual_objective_entry(
            comparison_dir,
            artifact_check_path=artifact_check_path,
            max_logit_delta=max_logit_delta,
        )
        entry["gate"] = _infer_focal_promotion_gate(comparison_dir, entry)
        entries.append(entry)
        failures.extend(entry["failures"])

    gate_backend_pairs = sorted(
        {
            (entry.get("gate"), entry.get("backend"))
            for entry in entries
            if entry.get("gate") is not None and entry.get("backend") is not None
        }
    )
    required_pairs = {
        ("char_xxlarge_seed2_local_colab", "local"),
        ("char_xxlarge_seed2_local_colab", "colab"),
        ("token_larger_seed2_local_colab", "local"),
        ("token_larger_seed2_local_colab", "colab"),
    }
    for gate, backend in sorted(required_pairs - set(gate_backend_pairs)):
        failures.append(
            {
                "field": "comparison.gate_backend_pair",
                "expected": f"{gate}/{backend}",
                "actual": "missing",
            }
        )

    focal_ce_wins = [
        entry
        for entry in entries
        if entry.get("supervised_run")
        and entry.get("focal_run")
        and entry["focal_run"]["best_hep_loss"] is not None
        and entry["supervised_run"]["best_hep_loss"] is not None
        and entry["focal_run"]["best_hep_loss"]
        < entry["supervised_run"]["best_hep_loss"]
    ]
    focal_minus_supervised = [
        entry["focal_minus_supervised_best_hep_loss"]
        for entry in entries
        if entry.get("focal_minus_supervised_best_hep_loss") is not None
    ]
    residual_loss_deltas = [
        entry["focal_minus_supervised_final_residual_loss"]
        for entry in entries
        if entry.get("focal_minus_supervised_final_residual_loss") is not None
    ]
    status = "fail" if failures else "pass"
    promote = status == "pass" and len(focal_ce_wins) == len(entries)
    report = {
        "status": status,
        "decision": (
            SATISFY_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE
            if promote
            else (
                STOP_FOCAL_RESIDUAL_OBJECTIVE_VALIDATION
                if status == "pass"
                else INSUFFICIENT_EVIDENCE
            )
        ),
        "promotion_gate_satisfied": promote,
        "selected_residual_objective_variant": "supervised_ce_focal"
        if promote
        else None,
        "promote_residual_learning_method": promote,
        "default_residual_objective": "supervised_ce",
        "policy": {
            "max_logit_delta_from_ordinary": max_logit_delta,
            "requires_promotion_gate_report_pass": True,
            "requires_char_xxlarge_seed2_local_and_colab": True,
            "requires_token_larger_seed2_local_and_colab": True,
            "requires_passing_artifact_checks": True,
            "requires_support_stress_preset_disabled": True,
            "requires_temporal_clipped_hep_path": True,
            "requires_both_objectives_improve_own_training_loss": True,
            "requires_focal_lower_supervised_ce_hep_loss_in_every_gate_comparison": True,
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
            "comparison_dirs": [str(path) for path in comparison_dirs],
            "artifact_check_paths": [str(path) for path in artifact_paths],
            "comparison_count": len(entries),
            "focal_ce_win_count": len(focal_ce_wins),
            "gate_backend_pairs": [
                {"gate": gate, "backend": backend}
                for gate, backend in gate_backend_pairs
            ],
            "gate_count": len({gate for gate, _backend in gate_backend_pairs}),
            "backend_count": len({backend for _gate, backend in gate_backend_pairs}),
            "mean_focal_minus_supervised_best_hep_loss": (
                sum(focal_minus_supervised) / len(focal_minus_supervised)
                if focal_minus_supervised
                else None
            ),
            "mean_focal_minus_supervised_final_residual_loss": (
                sum(residual_loss_deltas) / len(residual_loss_deltas)
                if residual_loss_deltas
                else None
            ),
            "entries": entries,
            "failures": failures,
        },
        "rationale": (
            "The focal promotion gate is satisfied: seed-2 xxlarge char and "
            "tokenized larger local/Colab comparisons all pass and focal CE has "
            "lower best temporal-clipped supervised CE HEP loss than supervised CE."
            if promote
            else (
                "The focal promotion/stop gate evidence is valid, but focal CE "
                "does not beat supervised CE in every required seed-2 comparison. "
                "Focal validation should stop under the current gate and the "
                "default residual objective should remain supervised CE."
                if status == "pass"
                else (
                    "The focal promotion/stop gate requires the gate definition "
                    "plus seed-2 xxlarge char and tokenized larger local/Colab "
                    "comparisons with passing artifact checks and valid focal and "
                    "supervised temporal-clipped runs."
                )
            )
        ),
        "next_step": (
            "make the explicit default residual objective change to supervised_ce_focal"
            if promote
            else (
                "stop focal residual-objective validation under the current gate and select the next residual-learning direction"
                if status == "pass"
                else "repair or regenerate the missing or failing focal promotion-gate evidence"
            )
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_focal_promotion_gate_satisfaction_markdown(
        out_dir / "decision_report.md",
        report,
    )
    return report


def write_temporal_consistency_residual_objective_decision_report(
    comparison_dirs: list[Path] | tuple[Path, ...] = (
        DEFAULT_TEMPORAL_CONSISTENCY_RESIDUAL_OBJECTIVE_COMPARISON_DIRS
    ),
    out_dir: Path = DEFAULT_TEMPORAL_CONSISTENCY_RESIDUAL_OBJECTIVE_OUT_DIR,
    *,
    artifact_check_paths: list[Path] | tuple[Path, ...] | None = (
        DEFAULT_TEMPORAL_CONSISTENCY_RESIDUAL_OBJECTIVE_ARTIFACT_CHECKS
    ),
    max_logit_delta: float = DEFAULT_MAX_LOGIT_DELTA,
    min_best_hep_loss_improvement: float = 1.0e-4,
) -> dict[str, Any]:
    """Decide whether train-time temporal consistency merits broader validation."""

    if max_logit_delta < 0.0:
        raise ValueError("max_logit_delta must be non-negative")
    if min_best_hep_loss_improvement < 0.0:
        raise ValueError("min_best_hep_loss_improvement must be non-negative")

    entries = []
    failures = []
    artifact_paths = list(artifact_check_paths or [])
    for index, comparison_dir in enumerate(comparison_dirs):
        comparison_dir = Path(comparison_dir)
        artifact_check_path = (
            Path(artifact_paths[index]) if index < len(artifact_paths) else None
        )
        entry = _temporal_consistency_residual_objective_entry(
            comparison_dir,
            artifact_check_path=artifact_check_path,
            max_logit_delta=max_logit_delta,
        )
        entries.append(entry)
        failures.extend(entry["failures"])

    scales = sorted({entry["scale"] for entry in entries if entry.get("scale")})
    for scale in sorted({"validation", "extended"} - set(scales)):
        failures.append(
            {
                "field": "comparison.scale",
                "expected": scale,
                "actual": "missing",
            }
        )

    temporal_runs = [
        run
        for entry in entries
        for run in entry.get("temporal_consistency_runs", [])
    ]
    improving_runs = [
        run
        for run in temporal_runs
        if run.get("best_hep_loss_improvement_vs_supervised") is not None
        and run["best_hep_loss_improvement_vs_supervised"]
        >= min_best_hep_loss_improvement
    ]
    all_temporal_minus_supervised = [
        run["best_hep_loss_delta_vs_supervised"]
        for run in temporal_runs
        if run.get("best_hep_loss_delta_vs_supervised") is not None
    ]
    final_loss_deltas = [
        run["final_residual_loss_delta_vs_supervised"]
        for run in temporal_runs
        if run.get("final_residual_loss_delta_vs_supervised") is not None
    ]
    best_run = min(
        (
            run
            for run in temporal_runs
            if run.get("best_hep_loss_delta_vs_supervised") is not None
        ),
        key=lambda run: run["best_hep_loss_delta_vs_supervised"],
        default=None,
    )
    best_improvement = (
        None
        if best_run is None
        else -float(best_run["best_hep_loss_delta_vs_supervised"])
    )
    status = "fail" if failures else "pass"
    continue_variant = (
        status == "pass"
        and bool(temporal_runs)
        and len(improving_runs) == len(temporal_runs)
    )
    report = {
        "status": status,
        "decision": (
            CONTINUE_TEMPORAL_CONSISTENCY_RESIDUAL_OBJECTIVE_VALIDATION
            if continue_variant
            else (
                STOP_TEMPORAL_CONSISTENCY_RESIDUAL_OBJECTIVE_VALIDATION
                if status == "pass"
                else INSUFFICIENT_EVIDENCE
            )
        ),
        "continue_temporal_consistency_residual_objective_validation": continue_variant,
        "selected_residual_objective_variant": (
            "supervised_ce_temporal_consistency" if continue_variant else None
        ),
        "promote_residual_learning_method": False,
        "default_residual_objective": "supervised_ce",
        "policy": {
            "max_logit_delta_from_ordinary": max_logit_delta,
            "min_best_hep_loss_improvement": min_best_hep_loss_improvement,
            "requires_validation_and_extended_local_evidence": True,
            "requires_passing_artifact_checks": True,
            "requires_supervised_and_temporal_consistency_runs": True,
            "requires_support_stress_preset_disabled": True,
            "requires_temporal_clipped_hep_path": True,
            "requires_all_objectives_improve_own_training_loss": True,
            "requires_each_temporal_consistency_run_to_clear_min_best_hep_loss_improvement": True,
            "allows_residual_objective_promotion": False,
            "local_only_decision": True,
        },
        "evidence": {
            "comparison_dirs": [str(path) for path in comparison_dirs],
            "artifact_check_paths": [str(path) for path in artifact_paths],
            "scale_count": len(scales),
            "scales": scales,
            "comparison_count": len(entries),
            "temporal_consistency_run_count": len(temporal_runs),
            "temporal_consistency_clear_margin_count": len(improving_runs),
            "best_temporal_consistency_run": best_run,
            "best_temporal_consistency_improvement": best_improvement,
            "mean_temporal_consistency_minus_supervised_best_hep_loss": (
                sum(all_temporal_minus_supervised) / len(all_temporal_minus_supervised)
                if all_temporal_minus_supervised
                else None
            ),
            "mean_temporal_consistency_minus_supervised_final_residual_loss": (
                sum(final_loss_deltas) / len(final_loss_deltas)
                if final_loss_deltas
                else None
            ),
            "entries": entries,
            "failures": failures,
        },
        "rationale": (
            "The validation and extended local sweeps are artifact-valid, but the "
            "train-time temporal-consistency regularizer only produces tiny best "
            "temporal-clipped HEP CE-loss changes while substantially increasing "
            "the regularized residual objective loss at larger weights. This does "
            "not justify Colab time or broader validation under the current gate."
            if status == "pass" and not continue_variant
            else (
                "Every temporal-consistency regularized run clears the minimum "
                "best-HEP CE improvement margin, so the variant merits matching "
                "Colab evidence before any promotion-style decision."
                if status == "pass"
                else (
                    "The temporal-consistency residual-objective decision requires "
                    "valid validation and extended local comparisons with passing "
                    "artifact checks and temporal-clipped objective-gate runs."
                )
            )
        ),
        "next_step": (
            "stop train-time temporal-consistency regularizer validation under the current gate and select a different residual-learning direction"
            if status == "pass" and not continue_variant
            else (
                "run matching Colab temporal-consistency regularizer sweeps before any broader decision"
                if status == "pass"
                else "repair or regenerate the temporal-consistency weight-sweep artifacts"
            )
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_temporal_consistency_residual_objective_markdown(
        out_dir / "decision_report.md",
        report,
    )
    return report


def write_residual_learning_next_direction_report(
    decision_report_paths: list[Path] | tuple[Path, ...] = (
        DEFAULT_RESIDUAL_LEARNING_NEXT_DIRECTION_REPORTS
    ),
    out_dir: Path = DEFAULT_RESIDUAL_LEARNING_NEXT_DIRECTION_OUT_DIR,
) -> dict[str, Any]:
    """Select the next residual-learning direction from completed stop gates."""

    expected_decisions = {
        "residual_objective_gate_decision": KEEP_SUPERVISED_CE_RESIDUAL_OBJECTIVE_DEFAULT,
        "anchored_pc_residual_objective_decision": STOP_PC_RESIDUAL_OBJECTIVE_VALIDATION,
        "confidence_penalty_residual_objective_decision": STOP_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION,
        "margin_penalty_residual_objective_decision": STOP_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION,
        "label_smoothing_residual_objective_decision": STOP_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_VALIDATION,
        "focal_residual_objective_promotion_gate_satisfaction": STOP_FOCAL_RESIDUAL_OBJECTIVE_VALIDATION,
        "temporal_consistency_residual_objective_decision": STOP_TEMPORAL_CONSISTENCY_RESIDUAL_OBJECTIVE_VALIDATION,
    }
    entries = []
    failures = []
    seen_keys = set()
    for path in decision_report_paths:
        path = Path(path)
        key = path.parent.name
        expected_decision = expected_decisions.get(key)
        if expected_decision is None:
            failures.append(
                {
                    "field": "decision_report.kind",
                    "expected": sorted(expected_decisions),
                    "actual": key,
                    "path": str(path),
                }
            )
            continue
        seen_keys.add(key)
        if not path.is_file():
            failures.append(
                {
                    "field": "decision_report",
                    "expected": "file exists",
                    "actual": "missing",
                    "path": str(path),
                }
            )
            entries.append(
                {
                    "key": key,
                    "path": str(path),
                    "status": None,
                    "decision": None,
                    "expected_decision": expected_decision,
                }
            )
            continue
        report = _read_json_object(path)
        entry = {
            "key": key,
            "path": str(path),
            "status": report.get("status"),
            "decision": report.get("decision"),
            "expected_decision": expected_decision,
            "promote_residual_learning_method": report.get(
                "promote_residual_learning_method"
            ),
            "default_residual_objective": report.get("default_residual_objective"),
            "next_step": report.get("next_step"),
        }
        entries.append(entry)
        if report.get("status") != "pass":
            failures.append(
                {
                    "field": "decision_report.status",
                    "expected": "pass",
                    "actual": report.get("status"),
                    "path": str(path),
                }
            )
        if report.get("decision") != expected_decision:
            failures.append(
                {
                    "field": "decision_report.decision",
                    "expected": expected_decision,
                    "actual": report.get("decision"),
                    "path": str(path),
                }
            )

    for key in sorted(set(expected_decisions) - seen_keys):
        failures.append(
            {
                "field": "decision_report.kind",
                "expected": key,
                "actual": "missing",
            }
        )

    status = "fail" if failures else "pass"
    report = {
        "status": status,
        "decision": (
            DEFINE_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_GATE
            if status == "pass"
            else INSUFFICIENT_EVIDENCE
        ),
        "selected_next_direction": (
            "residual_capacity_support_diagnostic" if status == "pass" else None
        ),
        "promote_residual_learning_method": False,
        "default_residual_objective": "supervised_ce",
        "policy": {
            "requires_completed_residual_objective_gate": True,
            "requires_stopped_pc_and_non_pc_objective_variants": True,
            "requires_stopped_focal_promotion_gate": True,
            "requires_stopped_train_time_temporal_consistency_variant": True,
            "allows_residual_objective_promotion": False,
            "diagnostic_gate_only": True,
        },
        "evidence": {
            "decision_report_paths": [str(path) for path in decision_report_paths],
            "report_count": len(entries),
            "expected_report_count": len(expected_decisions),
            "entries": entries,
            "failures": failures,
        },
        "rationale": (
            "Completed objective-gate reports keep supervised CE as the default "
            "and stop the PC, anchored-PC, confidence-penalty, margin-penalty, "
            "label-smoothing, focal, and train-time temporal-consistency "
            "branches under their current gates. The next bounded residual-layer "
            "learning direction should therefore test whether residual capacity "
            "and sparse-support behavior, rather than another CE-adjacent loss "
            "variant, is limiting learned residual improvements."
            if status == "pass"
            else (
                "The next residual-learning direction cannot be selected until "
                "the completed residual-objective reports are present, passing, "
                "and stopped under their expected decisions."
            )
        ),
        "next_step": (
            "define a local residual capacity/support diagnostic gate that compares the supervised CE objective across increased column capacity, top-k support width, and temporal-clipped HEP artifacts"
            if status == "pass"
            else "repair or regenerate the missing or unexpected residual-objective decision reports"
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_residual_learning_next_direction_markdown(
        out_dir / "decision_report.md",
        report,
    )
    return report


def write_residual_capacity_support_diagnostic_gate_report(
    next_direction_report_path: Path = (
        DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_GATE_REPORT
    ),
    config_paths: list[Path] | tuple[Path, ...] = (
        DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_GATE_CONFIGS
    ),
    out_dir: Path = DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_GATE_OUT_DIR,
) -> dict[str, Any]:
    """Define the local residual capacity/support diagnostic gate."""

    failures = []
    next_direction = (
        _read_json_object(next_direction_report_path)
        if Path(next_direction_report_path).is_file()
        else None
    )
    if next_direction is None:
        failures.append(
            {
                "field": "next_direction_report",
                "expected": "file exists",
                "actual": "missing",
                "path": str(next_direction_report_path),
            }
        )
    else:
        if next_direction.get("status") != "pass":
            failures.append(
                {
                    "field": "next_direction_report.status",
                    "expected": "pass",
                    "actual": next_direction.get("status"),
                    "path": str(next_direction_report_path),
                }
            )
        if (
            next_direction.get("decision")
            != DEFINE_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_GATE
        ):
            failures.append(
                {
                    "field": "next_direction_report.decision",
                    "expected": DEFINE_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_GATE,
                    "actual": next_direction.get("decision"),
                    "path": str(next_direction_report_path),
                }
            )

    config_entries = []
    for path in config_paths:
        path = Path(path)
        if not path.is_file():
            failures.append(
                {
                    "field": "config",
                    "expected": "file exists",
                    "actual": "missing",
                    "path": str(path),
                }
            )
            config_entries.append({"path": str(path), "status": "missing"})
            continue
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            config = {}
        entry = _residual_capacity_support_config_entry(path, config)
        config_entries.append(entry)
        failures.extend(entry["failures"])

    baseline = config_entries[0] if config_entries else None
    matrix_failures = _residual_capacity_support_matrix_failures(config_entries)
    failures.extend(matrix_failures)

    status = "fail" if failures else "pass"
    comparison_dir = (
        "results/comparisons/"
        "validation_residual_capacity_support_temporal_clipped_objective_gate"
    )
    compare_command = " ".join(
        [
            "python -m relaleap.experiments.compare",
            *[f"--config {path}" for path in config_paths],
            f"--out {comparison_dir}",
        ]
    )
    check_command = " ".join(
        [
            "python -m relaleap.experiments.check_artifacts",
            f"--comparison-dir {comparison_dir}",
            f"--out {comparison_dir}/artifact_check_local.json",
        ]
    )
    report = {
        "status": status,
        "decision": (
            DEFINE_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_GATE
            if status == "pass"
            else INSUFFICIENT_EVIDENCE
        ),
        "selected_next_direction": (
            "residual_capacity_support_diagnostic" if status == "pass" else None
        ),
        "promote_residual_learning_method": False,
        "default_residual_objective": "supervised_ce",
        "policy": {
            "requires_passing_next_direction_report": True,
            "requires_supervised_ce_objective": True,
            "requires_temporal_clipped_hep": True,
            "requires_support_stress_preset_disabled": True,
            "requires_baseline_capacity_width_control": True,
            "requires_increased_column_capacity_variant": True,
            "requires_wider_top_k_support_variant": True,
            "requires_combined_capacity_and_support_variant": True,
            "diagnostic_gate_only": True,
            "allows_residual_objective_promotion": False,
        },
        "evidence": {
            "next_direction_report_path": str(next_direction_report_path),
            "next_direction_report_status": None
            if next_direction is None
            else next_direction.get("status"),
            "next_direction_report_decision": None
            if next_direction is None
            else next_direction.get("decision"),
            "baseline_config": None if baseline is None else baseline.get("path"),
            "config_paths": [str(path) for path in config_paths],
            "config_count": len(config_entries),
            "configs": config_entries,
            "failures": failures,
        },
        "commands": {
            "compare": compare_command,
            "check_artifacts": check_command,
        },
        "rationale": (
            "The completed residual-objective stop gates selected a capacity and "
            "support diagnostic rather than another CE-adjacent objective. This "
            "gate keeps the promoted temporal-clipped supervised CE path fixed "
            "and varies only residual column capacity and top-k support width, "
            "so the next local evidence can distinguish under-capacity from "
            "sparse-support bottlenecks."
            if status == "pass"
            else (
                "The residual capacity/support diagnostic gate is not ready "
                "until the next-direction report passes and every diagnostic "
                "config preserves the promoted temporal-clipped supervised CE "
                "harness."
            )
        ),
        "next_step": (
            "run the local residual capacity/support diagnostic comparison and artifact check recorded in commands.compare and commands.check_artifacts"
            if status == "pass"
            else "repair the missing or drifting next-direction report/config matrix, then regenerate this gate report"
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_residual_capacity_support_diagnostic_gate_markdown(
        out_dir / "decision_report.md",
        report,
    )
    return report


def write_residual_capacity_support_diagnostic_decision_report(
    comparison_dir: Path = DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_COMPARISON_DIR,
    out_dir: Path = DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_DECISION_OUT_DIR,
    *,
    artifact_check_path: Path | None = DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_ARTIFACT_CHECK,
    max_logit_delta: float = DEFAULT_MAX_LOGIT_DELTA,
) -> dict[str, Any]:
    """Decide whether local capacity/support evidence merits Colab validation."""

    if max_logit_delta < 0.0:
        raise ValueError("max_logit_delta must be non-negative")

    comparison = _read_json_object(comparison_dir / "summary.json")
    artifact_check = (
        _read_json_object(artifact_check_path)
        if artifact_check_path is not None and artifact_check_path.is_file()
        else check_comparison_artifacts(comparison_dir)
    )
    runs = comparison.get("runs") if isinstance(comparison.get("runs"), list) else []
    entries = [
        _residual_capacity_support_run_entry(run, max_logit_delta=max_logit_delta)
        for run in runs
        if isinstance(run, dict)
    ]
    entry_by_variant = {
        entry["variant"]: entry
        for entry in entries
        if entry.get("variant") is not None
    }
    failures = _residual_capacity_support_decision_failures(
        comparison_dir,
        comparison,
        artifact_check,
        entries,
        entry_by_variant,
    )
    baseline = entry_by_variant.get("baseline")
    support = entry_by_variant.get("support_width")
    capacity = entry_by_variant.get("capacity")
    combined = entry_by_variant.get("capacity_support_width")
    best_entry = min(
        [entry for entry in entries if entry.get("best_hep_loss") is not None],
        key=lambda entry: float(entry["best_hep_loss"]),
        default=None,
    )
    support_minus_baseline = _entry_loss_delta(support, baseline)
    capacity_minus_baseline = _entry_loss_delta(capacity, baseline)
    combined_minus_baseline = _entry_loss_delta(combined, baseline)
    support_beats_baseline = (
        support_minus_baseline is not None and support_minus_baseline < 0.0
    )
    support_is_best = (
        best_entry is not None and best_entry.get("variant") == "support_width"
    )
    accepted_support_alpha = (
        None
        if support is None
        else _best_accepted_alpha(
            support.get("alpha_candidates") or [],
            max_logit_delta=max_logit_delta,
        )
    )
    decision = (
        RUN_COLAB_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC
        if not failures
        and support_beats_baseline
        and support_is_best
        and accepted_support_alpha is not None
        else INSUFFICIENT_EVIDENCE
    )
    if not support_beats_baseline:
        failures.append(
            {
                "field": "support_width.best_hep_loss",
                "expected": "< baseline best HEP loss",
                "actual": None
                if support_minus_baseline is None
                else f"delta {support_minus_baseline}",
                "path": str(comparison_dir),
            }
        )
    if not support_is_best:
        failures.append(
            {
                "field": "best_variant",
                "expected": "support_width",
                "actual": None if best_entry is None else best_entry.get("variant"),
                "path": str(comparison_dir),
            }
        )
    if accepted_support_alpha is None:
        failures.append(
            {
                "field": "support_width.hep_alpha_sweep",
                "expected": "accepted nonzero alpha within logit-delta budget",
                "actual": "none",
                "path": str(comparison_dir),
            }
        )
    if decision == INSUFFICIENT_EVIDENCE:
        failures = _dedupe_failures(failures)

    report = {
        "status": "pass" if decision != INSUFFICIENT_EVIDENCE else "fail",
        "decision": decision,
        "selected_next_direction": (
            "colab_residual_capacity_support_diagnostic"
            if decision == RUN_COLAB_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC
            else None
        ),
        "promote_residual_learning_method": False,
        "default_residual_objective": "supervised_ce",
        "policy": {
            "max_logit_delta_from_ordinary": max_logit_delta,
            "requires_passing_artifact_check": True,
            "requires_passing_comparison_verdict": True,
            "requires_four_expected_variants": True,
            "requires_support_width_to_beat_baseline": True,
            "requires_support_width_to_be_best_variant": True,
            "requires_support_width_accepted_nonzero_alpha": True,
            "allows_residual_objective_promotion": False,
            "local_decision_only": True,
        },
        "evidence": {
            "comparison_dir": str(comparison_dir),
            "artifact_check_path": None
            if artifact_check_path is None
            else str(artifact_check_path),
            "artifact_check_status": artifact_check.get("status"),
            "comparison_status": comparison.get("status"),
            "verdict_status": (comparison.get("verdict") or {}).get("status")
            if isinstance(comparison.get("verdict"), dict)
            else None,
            "run_count": len(entries),
            "baseline_variant": baseline,
            "capacity_variant": capacity,
            "support_width_variant": support,
            "capacity_support_width_variant": combined,
            "best_variant": best_entry,
            "support_minus_baseline_best_hep_loss": support_minus_baseline,
            "capacity_minus_baseline_best_hep_loss": capacity_minus_baseline,
            "capacity_support_width_minus_baseline_best_hep_loss": combined_minus_baseline,
            "accepted_support_width_alpha": accepted_support_alpha,
            "entries": entries,
            "failures": failures,
        },
        "rationale": (
            "The local diagnostic keeps supervised CE and temporal-clipped HEP "
            "fixed while varying only residual column count and support width. "
            "Widened support is the best local variant, improves best HEP CE "
            "loss over the baseline inside the logit-delta budget, and the "
            "artifact contract passes, so the bounded next step is a matching "
            "Colab validation run."
            if decision == RUN_COLAB_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC
            else (
                "The local diagnostic does not yet justify spending Colab time "
                "because the completed artifact-backed comparison is missing, "
                "failing, or does not select widened support under the policy."
            )
        ),
        "next_step": (
            "run the matching Colab residual capacity/support diagnostic comparison through the real-Chrome CDP bridge"
            if decision == RUN_COLAB_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC
            else "repair or rerun the local residual capacity/support diagnostic comparison before Colab validation"
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_residual_capacity_support_diagnostic_decision_markdown(
        out_dir / "decision_report.md",
        report,
    )
    return report


def write_residual_capacity_support_diagnostic_colab_decision_report(
    comparison_dirs: tuple[Path, ...] = DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_COLAB_COMPARISON_DIRS,
    out_dir: Path = DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_COLAB_DECISION_OUT_DIR,
    *,
    artifact_check_paths: tuple[Path, ...] = DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_COLAB_ARTIFACT_CHECKS,
    max_logit_delta: float = DEFAULT_MAX_LOGIT_DELTA,
) -> dict[str, Any]:
    """Confirm local residual capacity/support evidence against Colab artifacts."""

    if max_logit_delta < 0.0:
        raise ValueError("max_logit_delta must be non-negative")

    backend_names = ("local", "colab")
    backend_evidence = []
    failures: list[dict[str, Any]] = []
    for index, comparison_dir in enumerate(comparison_dirs):
        artifact_check_path = (
            artifact_check_paths[index] if index < len(artifact_check_paths) else None
        )
        backend = (
            backend_names[index]
            if index < len(backend_names)
            else f"backend_{index + 1}"
        )
        evidence = _residual_capacity_support_decision_evidence(
            comparison_dir,
            artifact_check_path=artifact_check_path,
            max_logit_delta=max_logit_delta,
        )
        evidence["backend"] = backend
        backend_evidence.append(evidence)
        failures.extend(evidence["failures"])
        if not evidence["support_beats_baseline"]:
            failures.append(
                {
                    "field": f"{backend}.support_width.best_hep_loss",
                    "expected": "< baseline best HEP loss",
                    "actual": None
                    if evidence["support_minus_baseline_best_hep_loss"] is None
                    else f"delta {evidence['support_minus_baseline_best_hep_loss']}",
                    "path": str(comparison_dir),
                }
            )
        if not evidence["support_is_best"]:
            best = evidence["best_variant"]
            failures.append(
                {
                    "field": f"{backend}.best_variant",
                    "expected": "support_width",
                    "actual": None if best is None else best.get("variant"),
                    "path": str(comparison_dir),
                }
            )
        if evidence["accepted_support_width_alpha"] is None:
            failures.append(
                {
                    "field": f"{backend}.support_width.hep_alpha_sweep",
                    "expected": "accepted nonzero alpha within logit-delta budget",
                    "actual": "none",
                    "path": str(comparison_dir),
                }
            )

    if len(backend_evidence) < 2:
        failures.append(
            {
                "field": "comparison_dirs",
                "expected": "local and colab comparison directories",
                "actual": len(backend_evidence),
            }
        )

    failures = _dedupe_failures(failures)
    decision = (
        CONTINUE_RESIDUAL_CAPACITY_SUPPORT_VALIDATION
        if not failures
        else INSUFFICIENT_EVIDENCE
    )
    report = {
        "status": "pass" if decision != INSUFFICIENT_EVIDENCE else "fail",
        "decision": decision,
        "selected_next_direction": (
            "residual_capacity_support_validation_gate"
            if decision == CONTINUE_RESIDUAL_CAPACITY_SUPPORT_VALIDATION
            else None
        ),
        "promote_residual_learning_method": False,
        "default_residual_objective": "supervised_ce",
        "policy": {
            "max_logit_delta_from_ordinary": max_logit_delta,
            "requires_local_and_colab_artifact_checks": True,
            "requires_passing_comparison_verdicts": True,
            "requires_four_expected_variants_per_backend": True,
            "requires_support_width_to_beat_baseline_per_backend": True,
            "requires_support_width_to_be_best_variant_per_backend": True,
            "requires_support_width_accepted_nonzero_alpha_per_backend": True,
            "allows_residual_objective_promotion": False,
        },
        "evidence": {
            "backend_count": len(backend_evidence),
            "backends": backend_evidence,
            "failures": failures,
        },
        "rationale": (
            "The residual capacity/support diagnostic now has matching local "
            "and Colab artifact-backed evidence. In both backends, widened "
            "support is the best variant and accepts a nonzero temporal-clipped "
            "HEP alpha inside the ordinary-logit budget, while increased column "
            "capacity alone does not explain the gain. This supports continuing "
            "with a broader support-width validation gate, not changing the "
            "default residual objective yet."
            if decision == CONTINUE_RESIDUAL_CAPACITY_SUPPORT_VALIDATION
            else (
                "The paired local/Colab diagnostic is missing, failing, or does "
                "not consistently select widened support under the policy."
            )
        ),
        "next_step": (
            "define a command-driven support-width validation gate at larger char and tokenized scales"
            if decision == CONTINUE_RESIDUAL_CAPACITY_SUPPORT_VALIDATION
            else "diagnose the local/Colab residual capacity/support divergence before any broader support-width gate"
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_residual_capacity_support_diagnostic_colab_decision_markdown(
        out_dir / "decision_report.md",
        report,
    )
    return report


def write_residual_support_width_validation_gate_report(
    colab_decision_report_path: Path = DEFAULT_RESIDUAL_SUPPORT_WIDTH_VALIDATION_GATE_REPORT,
    config_paths: list[Path] | tuple[Path, ...] = (
        DEFAULT_RESIDUAL_SUPPORT_WIDTH_VALIDATION_GATE_CONFIGS
    ),
    out_dir: Path = DEFAULT_RESIDUAL_SUPPORT_WIDTH_VALIDATION_GATE_OUT_DIR,
) -> dict[str, Any]:
    """Define the broader support-width validation gate."""

    failures = []
    colab_decision = (
        _read_json_object(colab_decision_report_path)
        if Path(colab_decision_report_path).is_file()
        else None
    )
    if colab_decision is None:
        failures.append(
            {
                "field": "colab_decision_report",
                "expected": "file exists",
                "actual": "missing",
                "path": str(colab_decision_report_path),
            }
        )
    else:
        if colab_decision.get("status") != "pass":
            failures.append(
                {
                    "field": "colab_decision_report.status",
                    "expected": "pass",
                    "actual": colab_decision.get("status"),
                    "path": str(colab_decision_report_path),
                }
            )
        if (
            colab_decision.get("decision")
            != CONTINUE_RESIDUAL_CAPACITY_SUPPORT_VALIDATION
        ):
            failures.append(
                {
                    "field": "colab_decision_report.decision",
                    "expected": CONTINUE_RESIDUAL_CAPACITY_SUPPORT_VALIDATION,
                    "actual": colab_decision.get("decision"),
                    "path": str(colab_decision_report_path),
                }
            )

    config_entries = []
    for path in config_paths:
        path = Path(path)
        if not path.is_file():
            failures.append(
                {
                    "field": "config",
                    "expected": "file exists",
                    "actual": "missing",
                    "path": str(path),
                }
            )
            config_entries.append({"path": str(path), "status": "missing"})
            continue
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            config = {}
        entry = _residual_support_width_validation_config_entry(path, config)
        config_entries.append(entry)
        failures.extend(entry["failures"])
    failures.extend(_residual_support_width_validation_matrix_failures(config_entries))

    status = "fail" if failures else "pass"
    comparison_dir = (
        "results/comparisons/"
        "support_width_larger_char_token_temporal_clipped_objective_gate"
    )
    compare_command = " ".join(
        [
            "python -m relaleap.experiments.compare",
            *[f"--config {path}" for path in config_paths],
            f"--out {comparison_dir}",
        ]
    )
    check_command = " ".join(
        [
            "python -m relaleap.experiments.check_artifacts",
            f"--comparison-dir {comparison_dir}",
            f"--out {comparison_dir}/artifact_check_local.json",
        ]
    )
    report = {
        "status": status,
        "decision": (
            DEFINE_RESIDUAL_SUPPORT_WIDTH_VALIDATION_GATE
            if status == "pass"
            else INSUFFICIENT_EVIDENCE
        ),
        "selected_next_direction": (
            "support_width_larger_char_token_validation" if status == "pass" else None
        ),
        "promote_residual_learning_method": False,
        "default_residual_objective": "supervised_ce",
        "policy": {
            "requires_passing_colab_capacity_support_decision": True,
            "requires_supervised_ce_objective": True,
            "requires_temporal_clipped_hep": True,
            "requires_support_stress_preset_disabled": True,
            "requires_larger_char_baseline_and_support_width": True,
            "requires_tokenized_baseline_and_support_width": True,
            "requires_support_width_top_k_increase_only": True,
            "allows_residual_objective_promotion": False,
            "validation_gate_only": True,
        },
        "evidence": {
            "colab_decision_report_path": str(colab_decision_report_path),
            "colab_decision_report_status": None
            if colab_decision is None
            else colab_decision.get("status"),
            "colab_decision_report_decision": None
            if colab_decision is None
            else colab_decision.get("decision"),
            "config_paths": [str(path) for path in config_paths],
            "config_count": len(config_entries),
            "configs": config_entries,
            "failures": failures,
        },
        "commands": {
            "compare": compare_command,
            "check_artifacts": check_command,
        },
        "rationale": (
            "The paired local/Colab residual capacity/support diagnostic selected "
            "widened support as the best variant. This gate broadens that result "
            "without changing the supervised CE residual objective or promoted "
            "temporal-clipped HEP path by comparing baseline top-k support "
            "against widened support at larger char and tokenized scales."
            if status == "pass"
            else (
                "The support-width validation gate cannot be defined until the "
                "paired local/Colab capacity-support diagnostic passes and the "
                "larger char/tokenized config matrix preserves the promoted "
                "temporal-clipped supervised CE harness."
            )
        ),
        "next_step": (
            "run the local support-width validation comparison and artifact check recorded in commands.compare and commands.check_artifacts"
            if status == "pass"
            else "repair the missing or drifting Colab diagnostic decision/config matrix, then regenerate this gate report"
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_residual_support_width_validation_gate_markdown(
        out_dir / "decision_report.md",
        report,
    )
    return report


def write_residual_support_width_validation_decision_report(
    comparison_dirs: tuple[Path, ...] = DEFAULT_RESIDUAL_SUPPORT_WIDTH_VALIDATION_COMPARISON_DIRS,
    out_dir: Path = DEFAULT_RESIDUAL_SUPPORT_WIDTH_VALIDATION_DECISION_OUT_DIR,
    *,
    artifact_check_paths: tuple[Path, ...] = DEFAULT_RESIDUAL_SUPPORT_WIDTH_VALIDATION_ARTIFACT_CHECKS,
    max_logit_delta: float = DEFAULT_MAX_LOGIT_DELTA,
) -> dict[str, Any]:
    """Confirm larger-char/tokenized support-width validation evidence."""

    if max_logit_delta < 0.0:
        raise ValueError("max_logit_delta must be non-negative")

    backend_names = ("local", "colab")
    backend_evidence = []
    failures: list[dict[str, Any]] = []
    for index, comparison_dir in enumerate(comparison_dirs):
        artifact_check_path = (
            artifact_check_paths[index] if index < len(artifact_check_paths) else None
        )
        backend = (
            backend_names[index]
            if index < len(backend_names)
            else f"backend_{index + 1}"
        )
        evidence = _residual_support_width_validation_decision_evidence(
            comparison_dir,
            artifact_check_path=artifact_check_path,
            max_logit_delta=max_logit_delta,
        )
        evidence["backend"] = backend
        backend_evidence.append(evidence)
        failures.extend(evidence["failures"])
        for scale in ("larger_char", "tokenized"):
            scale_evidence = evidence["scales"].get(scale)
            if scale_evidence is None:
                failures.append(
                    {
                        "field": f"{backend}.{scale}",
                        "expected": "baseline and support-width runs",
                        "actual": "missing",
                        "path": str(comparison_dir),
                    }
                )
                continue
            if not scale_evidence["support_beats_baseline_alpha0_loss"]:
                failures.append(
                    {
                        "field": f"{backend}.{scale}.support_width.alpha0_loss",
                        "expected": "< baseline alpha0 loss",
                        "actual": None
                        if scale_evidence["support_minus_baseline_alpha0_loss"] is None
                        else (
                            "delta "
                            f"{scale_evidence['support_minus_baseline_alpha0_loss']}"
                        ),
                        "path": str(comparison_dir),
                    }
                )
            if not scale_evidence["support_beats_baseline_final_loss"]:
                failures.append(
                    {
                        "field": f"{backend}.{scale}.support_width.final_residual_loss",
                        "expected": "< baseline final residual loss",
                        "actual": None
                        if scale_evidence["support_minus_baseline_final_loss"] is None
                        else (
                            "delta "
                            f"{scale_evidence['support_minus_baseline_final_loss']}"
                        ),
                        "path": str(comparison_dir),
                    }
                )

    if len(backend_evidence) < 2:
        failures.append(
            {
                "field": "comparison_dirs",
                "expected": "local and colab comparison directories",
                "actual": len(backend_evidence),
            }
        )

    failures = _dedupe_failures(failures)
    decision = (
        CONTINUE_RESIDUAL_SUPPORT_WIDTH_VALIDATION
        if not failures
        else INSUFFICIENT_EVIDENCE
    )
    report = {
        "status": "pass" if decision != INSUFFICIENT_EVIDENCE else "fail",
        "decision": decision,
        "selected_next_direction": (
            "support_width_repeat_or_capacity_interaction_validation"
            if decision == CONTINUE_RESIDUAL_SUPPORT_WIDTH_VALIDATION
            else None
        ),
        "promote_residual_learning_method": False,
        "default_residual_objective": "supervised_ce",
        "default_support_stress_mitigation": "temporal_clipped_hep",
        "policy": {
            "max_logit_delta_from_ordinary": max_logit_delta,
            "requires_local_and_colab_artifact_checks": True,
            "requires_passing_comparison_verdicts": True,
            "requires_larger_char_baseline_and_support_width_per_backend": True,
            "requires_tokenized_baseline_and_support_width_per_backend": True,
            "requires_wide_support_alpha0_loss_improvement_per_scale": True,
            "requires_wide_support_final_residual_loss_improvement_per_scale": True,
            "requires_temporal_clipped_hep_stability_context": True,
            "allows_residual_objective_promotion": False,
        },
        "evidence": {
            "backend_count": len(backend_evidence),
            "backends": backend_evidence,
            "failures": failures,
        },
        "rationale": (
            "The broader support-width validation has matching local and Colab "
            "artifact-backed evidence. In both backends, top-k 2 support improves "
            "ordinary alpha-0 supervised CE loss and final residual loss over "
            "the top-k 1 baseline at larger-char and tokenized scales while "
            "leaving the supervised CE objective and temporal-clipped HEP path "
            "fixed. This supports continuing support-width validation, not "
            "changing the residual objective."
            if decision == CONTINUE_RESIDUAL_SUPPORT_WIDTH_VALIDATION
            else (
                "The paired support-width validation is missing, failing, or "
                "does not consistently improve widened-support ordinary CE loss "
                "across local and Colab evidence."
            )
        ),
        "next_step": (
            "define a bounded repeat or capacity-interaction support-width validation gate before any default support-width change"
            if decision == CONTINUE_RESIDUAL_SUPPORT_WIDTH_VALIDATION
            else "repair or rerun the local/Colab support-width validation artifacts before selecting another support-width step"
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_residual_support_width_validation_decision_markdown(
        out_dir / "decision_report.md",
        report,
    )
    return report


def _focal_residual_objective_entry(
    comparison_dir: Path,
    *,
    artifact_check_path: Path | None,
    max_logit_delta: float,
) -> dict[str, Any]:
    entry = _residual_objective_variant_entry(
        comparison_dir,
        variant_objective="supervised_ce_focal",
        variant_field="focal_run",
        missing_field="comparison.runs.supervised_ce_focal",
        artifact_check_path=artifact_check_path,
        max_logit_delta=max_logit_delta,
    )
    supervised = entry.get("supervised_run")
    focal = entry.get("focal_run")
    entry["focal_minus_supervised_best_hep_loss"] = _best_loss_delta(
        focal,
        supervised,
    )
    entry["focal_minus_supervised_final_residual_loss"] = (
        None
        if not isinstance(focal, dict)
        or not isinstance(supervised, dict)
        or focal.get("final_residual_loss") is None
        or supervised.get("final_residual_loss") is None
        else float(focal["final_residual_loss"])
        - float(supervised["final_residual_loss"])
    )
    return entry


def _temporal_consistency_residual_objective_entry(
    comparison_dir: Path,
    *,
    artifact_check_path: Path | None,
    max_logit_delta: float,
) -> dict[str, Any]:
    entry = _residual_objective_variant_entry(
        comparison_dir,
        variant_objective="supervised_ce_temporal_consistency",
        variant_field="temporal_consistency_run",
        missing_field="comparison.runs.supervised_ce_temporal_consistency",
        artifact_check_path=artifact_check_path,
        max_logit_delta=max_logit_delta,
    )
    comparison = (
        _read_json_object(comparison_dir / "summary.json")
        if (comparison_dir / "summary.json").is_file()
        else {}
    )
    runs = comparison.get("runs", []) if isinstance(comparison.get("runs"), list) else []
    supervised_runs = _runs_with_residual_objective(runs, "supervised_ce")
    supervised = (
        _residual_objective_run_entry(supervised_runs[0], max_logit_delta=max_logit_delta)
        if supervised_runs
        else None
    )
    temporal_runs = [
        _residual_objective_run_entry(run, max_logit_delta=max_logit_delta)
        for run in _runs_with_residual_objective(
            runs,
            "supervised_ce_temporal_consistency",
        )
    ]
    for run in temporal_runs:
        _append_residual_objective_run_failures(
            entry["failures"],
            comparison_dir,
            run,
        )
        run["temporal_consistency_weight"] = _infer_temporal_consistency_weight(
            str(run.get("experiment_id") or "")
        )
        run["best_hep_loss_delta_vs_supervised"] = _best_loss_delta(run, supervised)
        run["best_hep_loss_improvement_vs_supervised"] = (
            None
            if run["best_hep_loss_delta_vs_supervised"] is None
            else -float(run["best_hep_loss_delta_vs_supervised"])
        )
        run["final_residual_loss_delta_vs_supervised"] = (
            None
            if not isinstance(supervised, dict)
            or run.get("final_residual_loss") is None
            or supervised.get("final_residual_loss") is None
            else float(run["final_residual_loss"])
            - float(supervised["final_residual_loss"])
        )
    if not temporal_runs:
        entry["temporal_consistency_runs"] = []
    else:
        entry["temporal_consistency_runs"] = temporal_runs
        entry["temporal_consistency_run"] = temporal_runs[0]
    entry["scale"] = _infer_temporal_consistency_scale(comparison_dir)
    return entry


def _infer_temporal_consistency_weight(experiment_id: str) -> float | None:
    if "_w005_" in experiment_id:
        return 0.05
    if "_w010_" in experiment_id:
        return 0.1
    if "_w020_" in experiment_id:
        return 0.2
    if "temporal_consistency" in experiment_id:
        return 0.01
    return None


def _infer_temporal_consistency_scale(comparison_dir: Path) -> str | None:
    name = comparison_dir.name
    if "validation" in name:
        return "validation"
    if "extended" in name:
        return "extended"
    return None


def _residual_objective_variant_entry(
    comparison_dir: Path,
    *,
    variant_objective: str,
    variant_field: str,
    missing_field: str,
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
    variant_runs = _runs_with_residual_objective(runs, variant_objective)
    supervised_run = (
        _residual_objective_run_entry(supervised_runs[0], max_logit_delta=max_logit_delta)
        if supervised_runs
        else None
    )
    variant_run = (
        _residual_objective_run_entry(variant_runs[0], max_logit_delta=max_logit_delta)
        if variant_runs
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
    if not variant_runs:
        failures.append(
            {
                "field": missing_field,
                "expected": "one run",
                "actual": 0,
                "path": str(comparison_dir),
            }
        )
    for run_entry in (supervised_run, variant_run):
        if run_entry is not None:
            _append_residual_objective_run_failures(
                failures,
                comparison_dir,
                run_entry,
            )

    return {
        "comparison_dir": str(comparison_dir),
        "artifact_check_path": str(artifact_check_path) if artifact_check_path else None,
        "backend": backend,
        "artifact_check_status": None if artifact_check is None else artifact_check.get("status"),
        "comparison_status": None if comparison is None else comparison.get("status"),
        "verdict_status": verdict.get("status"),
        "supervised_run": supervised_run,
        variant_field: variant_run,
        "failures": failures,
    }


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


def _infer_focal_promotion_gate(
    comparison_dir: Path,
    entry: dict[str, Any],
) -> str | None:
    texts = [str(comparison_dir)]
    for run_key in ("supervised_run", "focal_run"):
        run = entry.get(run_key)
        if isinstance(run, dict):
            texts.append(str(run.get("experiment_id", "")))
            texts.append(str(run.get("dataset", "")))
    joined = " ".join(texts)
    if "token_larger" in joined or "tiny_shakespeare_word" in joined:
        return "token_larger_seed2_local_colab"
    if "char_xxlarge" in joined or "tiny_shakespeare_char" in joined:
        return "char_xxlarge_seed2_local_colab"
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


def _residual_capacity_support_config_entry(
    path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    run = config.get("run") if isinstance(config.get("run"), dict) else {}
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    training = (
        config.get("training") if isinstance(config.get("training"), dict) else {}
    )
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    base = model.get("base") if isinstance(model.get("base"), dict) else {}
    columns = (
        model.get("columns") if isinstance(model.get("columns"), dict) else {}
    )
    inference = (
        config.get("inference") if isinstance(config.get("inference"), dict) else {}
    )
    outputs = config.get("outputs") if isinstance(config.get("outputs"), dict) else {}
    entry = {
        "path": str(path),
        "experiment_id": run.get("experiment_id"),
        "dataset": data.get("dataset"),
        "seq_len": data.get("seq_len"),
        "max_steps": run.get("max_steps"),
        "residual_objective": training.get("residual_objective"),
        "hidden_dim": base.get("hidden_dim"),
        "num_columns": columns.get("num_columns"),
        "atoms_per_column": columns.get("atoms_per_column"),
        "top_k": columns.get("top_k"),
        "support_stress": columns.get("support_stress"),
        "support_stress_preset": columns.get("support_stress_preset"),
        "hep_update_clip_norm": inference.get("hep_update_clip_norm"),
        "hep_settling_objective": inference.get("hep_settling_objective"),
        "hep_alpha_sweep": inference.get("hep_alpha_sweep"),
        "require_summary_json": outputs.get("require_summary_json"),
        "require_metrics_csv": outputs.get("require_metrics_csv"),
        "require_notes_md": outputs.get("require_notes_md"),
        "failures": [],
    }
    required_values = {
        "dataset": "tiny_shakespeare_char",
        "seq_len": 64,
        "max_steps": 25,
        "residual_objective": "supervised_ce",
        "hidden_dim": 64,
        "atoms_per_column": 4,
        "support_stress": True,
        "support_stress_preset": False,
        "hep_update_clip_norm": 0.01,
        "hep_settling_objective": "temporal_consistency_gradient",
        "hep_alpha_sweep": "0.0,0.25,0.5,1.0",
        "require_summary_json": True,
        "require_metrics_csv": True,
        "require_notes_md": True,
    }
    for field, expected in required_values.items():
        if entry.get(field) != expected:
            entry["failures"].append(
                {
                    "field": f"config.{field}",
                    "expected": expected,
                    "actual": entry.get(field),
                    "path": str(path),
                }
            )
    if not isinstance(entry["num_columns"], int) or entry["num_columns"] <= 0:
        entry["failures"].append(
            {
                "field": "config.num_columns",
                "expected": "positive integer",
                "actual": entry["num_columns"],
                "path": str(path),
            }
        )
    if not isinstance(entry["top_k"], int) or entry["top_k"] <= 0:
        entry["failures"].append(
            {
                "field": "config.top_k",
                "expected": "positive integer",
                "actual": entry["top_k"],
                "path": str(path),
            }
        )
    return entry


def _residual_capacity_support_matrix_failures(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if len(entries) != 4:
        return [
            {
                "field": "config_matrix.count",
                "expected": 4,
                "actual": len(entries),
            }
        ]
    if any(entry.get("failures") for entry in entries):
        return failures
    baseline, capacity, support, combined = entries
    baseline_columns = baseline.get("num_columns")
    baseline_top_k = baseline.get("top_k")
    specs = [
        ("capacity_variant", capacity, "num_columns"),
        ("support_width_variant", support, "top_k"),
        ("capacity_support_width_variant", combined, "both"),
    ]
    for name, entry, variant in specs:
        columns = entry.get("num_columns")
        top_k = entry.get("top_k")
        if variant in ("num_columns", "both") and not columns > baseline_columns:
            failures.append(
                {
                    "field": f"config_matrix.{name}.num_columns",
                    "expected": f"> {baseline_columns}",
                    "actual": columns,
                    "path": entry.get("path"),
                }
            )
        if variant == "top_k" and columns != baseline_columns:
            failures.append(
                {
                    "field": f"config_matrix.{name}.num_columns",
                    "expected": baseline_columns,
                    "actual": columns,
                    "path": entry.get("path"),
                }
            )
        if variant in ("top_k", "both") and not top_k > baseline_top_k:
            failures.append(
                {
                    "field": f"config_matrix.{name}.top_k",
                    "expected": f"> {baseline_top_k}",
                    "actual": top_k,
                    "path": entry.get("path"),
                }
            )
        if variant == "num_columns" and top_k != baseline_top_k:
            failures.append(
                {
                    "field": f"config_matrix.{name}.top_k",
                    "expected": baseline_top_k,
                    "actual": top_k,
                    "path": entry.get("path"),
                }
            )
    return failures


def _residual_support_width_validation_config_entry(
    path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    run = config.get("run") if isinstance(config.get("run"), dict) else {}
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    training = (
        config.get("training") if isinstance(config.get("training"), dict) else {}
    )
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    base = model.get("base") if isinstance(model.get("base"), dict) else {}
    columns = (
        model.get("columns") if isinstance(model.get("columns"), dict) else {}
    )
    inference = (
        config.get("inference") if isinstance(config.get("inference"), dict) else {}
    )
    outputs = config.get("outputs") if isinstance(config.get("outputs"), dict) else {}
    entry = {
        "path": str(path),
        "experiment_id": run.get("experiment_id"),
        "dataset": data.get("dataset"),
        "seq_len": data.get("seq_len"),
        "max_steps": run.get("max_steps"),
        "residual_objective": training.get("residual_objective"),
        "hidden_dim": base.get("hidden_dim"),
        "num_columns": columns.get("num_columns"),
        "atoms_per_column": columns.get("atoms_per_column"),
        "top_k": columns.get("top_k"),
        "support_stress": columns.get("support_stress"),
        "support_stress_preset": columns.get("support_stress_preset"),
        "pc_steps": inference.get("pc_steps"),
        "hep_update_clip_norm": inference.get("hep_update_clip_norm"),
        "hep_settling_objective": inference.get("hep_settling_objective"),
        "hep_alpha_sweep": inference.get("hep_alpha_sweep"),
        "require_summary_json": outputs.get("require_summary_json"),
        "require_metrics_csv": outputs.get("require_metrics_csv"),
        "require_notes_md": outputs.get("require_notes_md"),
        "scale": _support_width_validation_scale(run.get("experiment_id"), path),
        "support_width": _support_width_validation_is_wide(
            run.get("experiment_id"),
            path,
        ),
        "failures": [],
    }
    required_values = {
        "max_steps": 50,
        "residual_objective": "supervised_ce",
        "hidden_dim": 96,
        "num_columns": 24,
        "atoms_per_column": 4,
        "support_stress": True,
        "support_stress_preset": False,
        "pc_steps": 4,
        "hep_update_clip_norm": 0.01,
        "hep_settling_objective": "temporal_consistency_gradient",
        "hep_alpha_sweep": "0.0,0.25,0.5,1.0",
        "require_summary_json": True,
        "require_metrics_csv": True,
        "require_notes_md": True,
    }
    for field, expected in required_values.items():
        if entry.get(field) != expected:
            entry["failures"].append(
                {
                    "field": f"config.{field}",
                    "expected": expected,
                    "actual": entry.get(field),
                    "path": str(path),
                }
            )
    expected_by_scale = {
        "larger_char": {"dataset": "tiny_shakespeare_char", "seq_len": 128},
        "tokenized": {"dataset": "tiny_shakespeare_word", "seq_len": 64},
    }
    scale = entry.get("scale")
    if scale not in expected_by_scale:
        entry["failures"].append(
            {
                "field": "config.scale",
                "expected": "larger_char or tokenized",
                "actual": scale,
                "path": str(path),
            }
        )
    else:
        for field, expected in expected_by_scale[scale].items():
            if entry.get(field) != expected:
                entry["failures"].append(
                    {
                        "field": f"config.{field}",
                        "expected": expected,
                        "actual": entry.get(field),
                        "path": str(path),
                    }
                )
    if entry.get("support_width") is True and entry.get("top_k") != 2:
        entry["failures"].append(
            {
                "field": "config.top_k",
                "expected": 2,
                "actual": entry.get("top_k"),
                "path": str(path),
            }
        )
    if entry.get("support_width") is False and entry.get("top_k") != 1:
        entry["failures"].append(
            {
                "field": "config.top_k",
                "expected": 1,
                "actual": entry.get("top_k"),
                "path": str(path),
            }
        )
    return entry


def _support_width_validation_scale(experiment_id: Any, path: Path) -> str | None:
    text = " ".join((str(experiment_id or ""), str(path)))
    if "char_larger" in text:
        return "larger_char"
    if "token_larger" in text:
        return "tokenized"
    return None


def _support_width_validation_is_wide(experiment_id: Any, path: Path) -> bool:
    text = " ".join((str(experiment_id or ""), str(path)))
    return "support_wide" in text


def _residual_support_width_validation_matrix_failures(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if len(entries) != 4:
        return [
            {
                "field": "config_matrix.count",
                "expected": 4,
                "actual": len(entries),
            }
        ]
    if any(entry.get("failures") for entry in entries):
        return failures
    by_key = {
        (entry.get("scale"), entry.get("support_width")): entry for entry in entries
    }
    for scale in ("larger_char", "tokenized"):
        baseline = by_key.get((scale, False))
        support = by_key.get((scale, True))
        if baseline is None:
            failures.append(
                {
                    "field": "config_matrix.scale_baseline",
                    "expected": f"{scale} baseline",
                    "actual": "missing",
                }
            )
            continue
        if support is None:
            failures.append(
                {
                    "field": "config_matrix.scale_support_width",
                    "expected": f"{scale} support-width variant",
                    "actual": "missing",
                }
            )
            continue
        for field in ("dataset", "seq_len", "max_steps", "hidden_dim", "num_columns"):
            if support.get(field) != baseline.get(field):
                failures.append(
                    {
                        "field": f"config_matrix.{scale}.{field}",
                        "expected": baseline.get(field),
                        "actual": support.get(field),
                        "path": support.get("path"),
                    }
                )
        if not support.get("top_k") > baseline.get("top_k"):
            failures.append(
                {
                    "field": f"config_matrix.{scale}.top_k",
                    "expected": f"> {baseline.get('top_k')}",
                    "actual": support.get("top_k"),
                    "path": support.get("path"),
                }
            )
    return failures


def _residual_support_width_validation_run_entry(
    run: dict[str, Any],
    *,
    max_logit_delta: float,
) -> dict[str, Any]:
    alpha_candidates = _alpha_candidates([run])
    alpha0 = next(
        (
            candidate
            for candidate in alpha_candidates
            if float(candidate.get("alpha", 0.0)) == 0.0
        ),
        None,
    )
    best_alpha = min(
        alpha_candidates,
        key=lambda candidate: float(candidate["loss"]),
        default=None,
    )
    accepted_alpha = _best_accepted_alpha(
        alpha_candidates,
        max_logit_delta=max_logit_delta,
    )
    return {
        "experiment_id": run.get("experiment_id"),
        "config_path": run.get("config_path"),
        "scale": _support_width_validation_scale(
            run.get("experiment_id"),
            Path(str(run.get("config_path") or "")),
        ),
        "support_width": _support_width_validation_is_wide(
            run.get("experiment_id"),
            Path(str(run.get("config_path") or "")),
        ),
        "status": run.get("status"),
        "dataset": run.get("dataset"),
        "top_k": run.get("top_k"),
        "residual_objective": run.get("residual_objective"),
        "support_stress": run.get("support_stress"),
        "support_stress_preset": run.get("support_stress_preset"),
        "hep_settling_objective": run.get("hep_settling_objective"),
        "hep_update_clip_norm": run.get("hep_update_clip_norm"),
        "training_steps": run.get("training_steps"),
        "final_residual_loss": run.get("final_residual_loss"),
        "alpha0_loss": None if alpha0 is None else alpha0.get("loss"),
        "best_hep_alpha": None if best_alpha is None else best_alpha.get("alpha"),
        "best_hep_loss": None if best_alpha is None else best_alpha.get("loss"),
        "best_hep_logit_delta": None
        if best_alpha is None
        else best_alpha.get("max_logit_delta_from_ordinary"),
        "accepted_hep_alpha": None
        if accepted_alpha is None
        else accepted_alpha.get("alpha"),
        "accepted_hep_loss": None
        if accepted_alpha is None
        else accepted_alpha.get("loss"),
        "max_support_change_fraction": _max_alpha_metric(
            [run],
            "support_change_fraction",
        ),
        "max_pinned_vs_repicked_logit_delta": _max_alpha_metric(
            [run],
            "pinned_vs_repicked_logit_delta",
        ),
        "invariants": run.get("invariants")
        if isinstance(run.get("invariants"), dict)
        else {},
        "artifact_invariants": run.get("artifact_invariants")
        if isinstance(run.get("artifact_invariants"), dict)
        else {},
        "alpha_candidates": alpha_candidates,
    }


def _residual_support_width_validation_decision_evidence(
    comparison_dir: Path,
    *,
    artifact_check_path: Path | None,
    max_logit_delta: float,
) -> dict[str, Any]:
    comparison = _read_json_object(comparison_dir / "summary.json")
    artifact_check = (
        _read_json_object(artifact_check_path)
        if artifact_check_path is not None and artifact_check_path.is_file()
        else check_comparison_artifacts(comparison_dir)
    )
    runs = comparison.get("runs") if isinstance(comparison.get("runs"), list) else []
    entries = [
        _residual_support_width_validation_run_entry(
            run,
            max_logit_delta=max_logit_delta,
        )
        for run in runs
        if isinstance(run, dict)
    ]
    failures = _residual_support_width_validation_decision_failures(
        comparison_dir,
        comparison,
        artifact_check,
        entries,
    )
    scales = {}
    for scale in ("larger_char", "tokenized"):
        baseline = next(
            (
                entry
                for entry in entries
                if entry.get("scale") == scale and entry.get("support_width") is False
            ),
            None,
        )
        support = next(
            (
                entry
                for entry in entries
                if entry.get("scale") == scale and entry.get("support_width") is True
            ),
            None,
        )
        if baseline is None or support is None:
            continue
        alpha0_delta = _entry_metric_delta(support, baseline, "alpha0_loss")
        final_delta = _entry_metric_delta(support, baseline, "final_residual_loss")
        best_delta = _entry_metric_delta(support, baseline, "best_hep_loss")
        scales[scale] = {
            "baseline": baseline,
            "support_width": support,
            "support_minus_baseline_alpha0_loss": alpha0_delta,
            "support_minus_baseline_final_loss": final_delta,
            "support_minus_baseline_best_hep_loss": best_delta,
            "support_beats_baseline_alpha0_loss": (
                alpha0_delta is not None and alpha0_delta < 0.0
            ),
            "support_beats_baseline_final_loss": (
                final_delta is not None and final_delta < 0.0
            ),
            "support_beats_baseline_best_hep_loss": (
                best_delta is not None and best_delta < 0.0
            ),
        }
    return {
        "comparison_dir": str(comparison_dir),
        "artifact_check_path": None
        if artifact_check_path is None
        else str(artifact_check_path),
        "artifact_check_status": artifact_check.get("status"),
        "comparison_status": comparison.get("status"),
        "verdict_status": (comparison.get("verdict") or {}).get("status")
        if isinstance(comparison.get("verdict"), dict)
        else None,
        "run_count": len(entries),
        "entries": entries,
        "scales": scales,
        "failures": failures,
    }


def _residual_support_width_validation_decision_failures(
    comparison_dir: Path,
    comparison: dict[str, Any],
    artifact_check: dict[str, Any],
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if artifact_check.get("status") != "pass":
        failures.append(
            {
                "field": "artifact_check.status",
                "expected": "pass",
                "actual": artifact_check.get("status"),
                "path": str(comparison_dir),
            }
        )
    verdict = comparison.get("verdict") if isinstance(comparison.get("verdict"), dict) else {}
    if comparison.get("status") != "ok":
        failures.append(
            {
                "field": "comparison.status",
                "expected": "ok",
                "actual": comparison.get("status"),
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
    if len(entries) != 4:
        failures.append(
            {
                "field": "comparison.runs.count",
                "expected": 4,
                "actual": len(entries),
                "path": str(comparison_dir),
            }
        )
    by_key = {
        (entry.get("scale"), entry.get("support_width")): entry for entry in entries
    }
    for scale in ("larger_char", "tokenized"):
        for support_width in (False, True):
            if (scale, support_width) not in by_key:
                failures.append(
                    {
                        "field": "comparison.runs.scale_support_width",
                        "expected": f"{scale}/{support_width}",
                        "actual": "missing",
                        "path": str(comparison_dir),
                    }
                )
    for entry in entries:
        prefix = f"comparison.runs.{entry.get('experiment_id')}"
        if entry.get("status") != "ok":
            failures.append(
                {
                    "field": f"{prefix}.status",
                    "expected": "ok",
                    "actual": entry.get("status"),
                    "path": str(comparison_dir),
                }
            )
        if entry.get("residual_objective") != "supervised_ce":
            failures.append(
                {
                    "field": f"{prefix}.residual_objective",
                    "expected": "supervised_ce",
                    "actual": entry.get("residual_objective"),
                    "path": str(comparison_dir),
                }
            )
        if entry.get("hep_settling_objective") != "temporal_consistency_gradient":
            failures.append(
                {
                    "field": f"{prefix}.hep_settling_objective",
                    "expected": "temporal_consistency_gradient",
                    "actual": entry.get("hep_settling_objective"),
                    "path": str(comparison_dir),
                }
            )
        if entry.get("support_stress_preset") is not False:
            failures.append(
                {
                    "field": f"{prefix}.support_stress_preset",
                    "expected": False,
                    "actual": entry.get("support_stress_preset"),
                    "path": str(comparison_dir),
                }
            )
        for invariant, passed in entry.get("invariants", {}).items():
            if passed is not True:
                failures.append(
                    {
                        "field": f"{prefix}.invariants.{invariant}",
                        "expected": True,
                        "actual": passed,
                        "path": str(comparison_dir),
                    }
                )
        for invariant, passed in entry.get("artifact_invariants", {}).items():
            if passed is not True:
                failures.append(
                    {
                        "field": f"{prefix}.artifact_invariants.{invariant}",
                        "expected": True,
                        "actual": passed,
                        "path": str(comparison_dir),
                    }
                )
    return failures


def _residual_capacity_support_run_entry(
    run: dict[str, Any],
    *,
    max_logit_delta: float,
) -> dict[str, Any]:
    alpha_candidates = _alpha_candidates([run])
    best_alpha = min(
        alpha_candidates,
        key=lambda candidate: float(candidate["loss"]),
        default=None,
    )
    accepted_alpha = _best_accepted_alpha(
        alpha_candidates,
        max_logit_delta=max_logit_delta,
    )
    return {
        "experiment_id": run.get("experiment_id"),
        "config_path": run.get("config_path"),
        "variant": _residual_capacity_support_variant(run),
        "status": run.get("status"),
        "residual_objective": run.get("residual_objective"),
        "support_stress": run.get("support_stress"),
        "support_stress_preset": run.get("support_stress_preset"),
        "hep_settling_objective": run.get("hep_settling_objective"),
        "hep_update_clip_norm": run.get("hep_update_clip_norm"),
        "training_steps": run.get("training_steps"),
        "final_residual_loss": run.get("final_residual_loss"),
        "best_hep_alpha": None if best_alpha is None else best_alpha.get("alpha"),
        "best_hep_loss": None if best_alpha is None else best_alpha.get("loss"),
        "best_hep_logit_delta": None
        if best_alpha is None
        else best_alpha.get("max_logit_delta_from_ordinary"),
        "accepted_hep_alpha": None
        if accepted_alpha is None
        else accepted_alpha.get("alpha"),
        "accepted_hep_loss": None
        if accepted_alpha is None
        else accepted_alpha.get("loss"),
        "max_support_change_fraction": _max_alpha_metric(
            [run],
            "support_change_fraction",
        ),
        "max_pinned_vs_repicked_logit_delta": _max_alpha_metric(
            [run],
            "pinned_vs_repicked_logit_delta",
        ),
        "invariants": run.get("invariants")
        if isinstance(run.get("invariants"), dict)
        else {},
        "artifact_invariants": run.get("artifact_invariants")
        if isinstance(run.get("artifact_invariants"), dict)
        else {},
        "alpha_candidates": alpha_candidates,
    }


def _residual_capacity_support_variant(run: dict[str, Any]) -> str | None:
    text = " ".join(
        str(value or "")
        for value in (run.get("experiment_id"), run.get("config_path"))
    )
    has_capacity = "capacity" in text
    has_support_width = "support_wide" in text or "capacity_support" in text
    if has_capacity and has_support_width:
        return "capacity_support_width"
    if has_capacity:
        return "capacity"
    if has_support_width:
        return "support_width"
    if "char_validation_hep_temporal_clipped_objective_gate" in text:
        return "baseline"
    return None


def _best_accepted_alpha(
    candidates: list[dict[str, Any]],
    *,
    max_logit_delta: float,
) -> dict[str, Any] | None:
    accepted = [
        candidate
        for candidate in candidates
        if float(candidate.get("alpha", 0.0)) > 0.0
        and (candidate.get("loss_improvement_from_alpha0") or 0.0) > 0.0
        and float(candidate.get("max_logit_delta_from_ordinary", 0.0))
        <= max_logit_delta
    ]
    return min(accepted, key=lambda candidate: float(candidate["loss"]), default=None)


def _entry_loss_delta(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> float | None:
    return _entry_metric_delta(left, right, "best_hep_loss")


def _entry_metric_delta(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    metric: str,
) -> float | None:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None
    left_value = left.get(metric)
    right_value = right.get(metric)
    if left_value is None or right_value is None:
        return None
    return float(left_value) - float(right_value)


def _residual_capacity_support_decision_failures(
    comparison_dir: Path,
    comparison: dict[str, Any],
    artifact_check: dict[str, Any],
    entries: list[dict[str, Any]],
    entry_by_variant: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if artifact_check.get("status") != "pass":
        failures.append(
            {
                "field": "artifact_check.status",
                "expected": "pass",
                "actual": artifact_check.get("status"),
                "path": str(comparison_dir),
            }
        )
    verdict = comparison.get("verdict") if isinstance(comparison.get("verdict"), dict) else {}
    if comparison.get("status") != "ok":
        failures.append(
            {
                "field": "comparison.status",
                "expected": "ok",
                "actual": comparison.get("status"),
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
    expected_variants = {
        "baseline",
        "capacity",
        "support_width",
        "capacity_support_width",
    }
    for variant in sorted(expected_variants - set(entry_by_variant)):
        failures.append(
            {
                "field": "comparison.runs.variant",
                "expected": variant,
                "actual": "missing",
                "path": str(comparison_dir),
            }
        )
    for entry in entries:
        prefix = f"comparison.runs.{entry.get('experiment_id')}"
        if entry.get("status") != "ok":
            failures.append(
                {
                    "field": f"{prefix}.status",
                    "expected": "ok",
                    "actual": entry.get("status"),
                    "path": str(comparison_dir),
                }
            )
        for invariant, passed in entry.get("invariants", {}).items():
            if passed is not True:
                failures.append(
                    {
                        "field": f"{prefix}.invariants.{invariant}",
                        "expected": True,
                        "actual": passed,
                        "path": str(comparison_dir),
                    }
                )
        for invariant, passed in entry.get("artifact_invariants", {}).items():
            if passed is not True:
                failures.append(
                    {
                        "field": f"{prefix}.artifact_invariants.{invariant}",
                        "expected": True,
                        "actual": passed,
                        "path": str(comparison_dir),
                    }
                )
    return failures


def _residual_capacity_support_decision_evidence(
    comparison_dir: Path,
    *,
    artifact_check_path: Path | None,
    max_logit_delta: float,
) -> dict[str, Any]:
    comparison = _read_json_object(comparison_dir / "summary.json")
    artifact_check = (
        _read_json_object(artifact_check_path)
        if artifact_check_path is not None and artifact_check_path.is_file()
        else check_comparison_artifacts(comparison_dir)
    )
    runs = comparison.get("runs") if isinstance(comparison.get("runs"), list) else []
    entries = [
        _residual_capacity_support_run_entry(run, max_logit_delta=max_logit_delta)
        for run in runs
        if isinstance(run, dict)
    ]
    entry_by_variant = {
        entry["variant"]: entry
        for entry in entries
        if entry.get("variant") is not None
    }
    failures = _residual_capacity_support_decision_failures(
        comparison_dir,
        comparison,
        artifact_check,
        entries,
        entry_by_variant,
    )
    baseline = entry_by_variant.get("baseline")
    support = entry_by_variant.get("support_width")
    capacity = entry_by_variant.get("capacity")
    combined = entry_by_variant.get("capacity_support_width")
    best_entry = min(
        [entry for entry in entries if entry.get("best_hep_loss") is not None],
        key=lambda entry: float(entry["best_hep_loss"]),
        default=None,
    )
    support_minus_baseline = _entry_loss_delta(support, baseline)
    support_is_best = (
        best_entry is not None and best_entry.get("variant") == "support_width"
    )
    accepted_support_alpha = (
        None
        if support is None
        else _best_accepted_alpha(
            support.get("alpha_candidates") or [],
            max_logit_delta=max_logit_delta,
        )
    )
    return {
        "comparison_dir": str(comparison_dir),
        "artifact_check_path": None
        if artifact_check_path is None
        else str(artifact_check_path),
        "artifact_check_status": artifact_check.get("status"),
        "comparison_status": comparison.get("status"),
        "verdict_status": (comparison.get("verdict") or {}).get("status")
        if isinstance(comparison.get("verdict"), dict)
        else None,
        "run_count": len(entries),
        "baseline_variant": baseline,
        "capacity_variant": capacity,
        "support_width_variant": support,
        "capacity_support_width_variant": combined,
        "best_variant": best_entry,
        "support_minus_baseline_best_hep_loss": support_minus_baseline,
        "capacity_minus_baseline_best_hep_loss": _entry_loss_delta(capacity, baseline),
        "capacity_support_width_minus_baseline_best_hep_loss": _entry_loss_delta(
            combined,
            baseline,
        ),
        "accepted_support_width_alpha": accepted_support_alpha,
        "support_beats_baseline": (
            support_minus_baseline is not None and support_minus_baseline < 0.0
        ),
        "support_is_best": support_is_best,
        "entries": entries,
        "failures": failures,
    }


def _dedupe_failures(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = []
    seen = set()
    for failure in failures:
        key = (
            failure.get("field"),
            failure.get("expected"),
            failure.get("actual"),
            failure.get("path"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(failure)
    return deduped


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


def _write_confidence_penalty_residual_objective_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    evidence = report["evidence"]
    lines = [
        "# Confidence-Penalty Residual Objective Decision",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        (
            "- Continue confidence-penalty validation: "
            f"`{report['continue_confidence_penalty_residual_objective_validation']}`"
        ),
        (
            "- Selected variant: "
            f"`{report['selected_residual_objective_variant']}`"
        ),
        f"- Default residual objective: `{report['default_residual_objective']}`",
        f"- Backends: `{', '.join(evidence['backends'])}`",
        (
            "- Mean confidence-penalty minus supervised best HEP loss: "
            f"`{_format_metric(evidence['mean_confidence_penalty_minus_supervised_best_hep_loss'])}`"
        ),
        (
            "- Mean confidence-penalty minus supervised final residual loss: "
            f"`{_format_metric(evidence['mean_confidence_penalty_minus_supervised_final_residual_loss'])}`"
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
            "| Confidence-penalty best HEP loss | Confidence minus supervised "
            "| Confidence final residual loss | Source |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for entry in evidence["entries"]:
        supervised = entry.get("supervised_run") or {}
        confidence = entry.get("confidence_penalty_run") or {}
        lines.append(
            (
                f"| {entry.get('backend') or ''} "
                f"| {entry.get('artifact_check_status') or ''} "
                f"| {_format_metric(supervised.get('best_hep_loss'))} "
                f"| {_format_metric(confidence.get('best_hep_loss'))} "
                f"| {_format_metric(entry.get('confidence_penalty_minus_supervised_best_hep_loss'))} "
                f"| {_format_metric(confidence.get('final_residual_loss'))} "
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


def _write_margin_penalty_residual_objective_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    evidence = report["evidence"]
    lines = [
        "# Margin-Penalty Residual Objective Decision",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        (
            "- Continue margin-penalty validation: "
            f"`{report['continue_margin_penalty_residual_objective_validation']}`"
        ),
        (
            "- Selected variant: "
            f"`{report['selected_residual_objective_variant']}`"
        ),
        f"- Default residual objective: `{report['default_residual_objective']}`",
        f"- Backends: `{', '.join(evidence['backends'])}`",
        (
            "- Mean margin-penalty minus supervised best HEP loss: "
            f"`{_format_metric(evidence['mean_margin_penalty_minus_supervised_best_hep_loss'])}`"
        ),
        (
            "- Mean margin-penalty minus supervised final residual loss: "
            f"`{_format_metric(evidence['mean_margin_penalty_minus_supervised_final_residual_loss'])}`"
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
            "| Margin-penalty best HEP loss | Margin minus supervised "
            "| Margin final residual loss | Source |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for entry in evidence["entries"]:
        supervised = entry.get("supervised_run") or {}
        margin = entry.get("margin_penalty_run") or {}
        lines.append(
            (
                f"| {entry.get('backend') or ''} "
                f"| {entry.get('artifact_check_status') or ''} "
                f"| {_format_metric(supervised.get('best_hep_loss'))} "
                f"| {_format_metric(margin.get('best_hep_loss'))} "
                f"| {_format_metric(entry.get('margin_penalty_minus_supervised_best_hep_loss'))} "
                f"| {_format_metric(margin.get('final_residual_loss'))} "
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


def _write_label_smoothing_residual_objective_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    evidence = report["evidence"]
    lines = [
        "# Label-Smoothing Residual Objective Decision",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        (
            "- Continue label-smoothing validation: "
            f"`{report['continue_label_smoothing_residual_objective_validation']}`"
        ),
        (
            "- Selected variant: "
            f"`{report['selected_residual_objective_variant']}`"
        ),
        f"- Default residual objective: `{report['default_residual_objective']}`",
        f"- Backends: `{', '.join(evidence['backends'])}`",
        (
            "- Mean label-smoothing minus supervised best HEP loss: "
            f"`{_format_metric(evidence['mean_label_smoothing_minus_supervised_best_hep_loss'])}`"
        ),
        (
            "- Mean label-smoothing minus supervised final residual loss: "
            f"`{_format_metric(evidence['mean_label_smoothing_minus_supervised_final_residual_loss'])}`"
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
            "| Label-smoothing best HEP loss | Label smoothing minus supervised "
            "| Label smoothing final residual loss | Source |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for entry in evidence["entries"]:
        supervised = entry.get("supervised_run") or {}
        label_smoothing = entry.get("label_smoothing_run") or {}
        lines.append(
            (
                f"| {entry.get('backend') or ''} "
                f"| {entry.get('artifact_check_status') or ''} "
                f"| {_format_metric(supervised.get('best_hep_loss'))} "
                f"| {_format_metric(label_smoothing.get('best_hep_loss'))} "
                f"| {_format_metric(entry.get('label_smoothing_minus_supervised_best_hep_loss'))} "
                f"| {_format_metric(label_smoothing.get('final_residual_loss'))} "
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


def _write_focal_residual_objective_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    evidence = report["evidence"]
    lines = [
        "# Focal Residual Objective Decision",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        (
            "- Continue focal validation: "
            f"`{report['continue_focal_residual_objective_validation']}`"
        ),
        (
            "- Selected variant: "
            f"`{report['selected_residual_objective_variant']}`"
        ),
        f"- Default residual objective: `{report['default_residual_objective']}`",
        f"- Backends: `{', '.join(evidence['backends'])}`",
        (
            "- Mean focal minus supervised best HEP loss: "
            f"`{_format_metric(evidence['mean_focal_minus_supervised_best_hep_loss'])}`"
        ),
        (
            "- Mean focal minus supervised final residual loss: "
            f"`{_format_metric(evidence['mean_focal_minus_supervised_final_residual_loss'])}`"
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
            "| Focal best HEP loss | Focal minus supervised "
            "| Focal final residual loss | Source |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for entry in evidence["entries"]:
        supervised = entry.get("supervised_run") or {}
        focal = entry.get("focal_run") or {}
        lines.append(
            (
                f"| {entry.get('backend') or ''} "
                f"| {entry.get('artifact_check_status') or ''} "
                f"| {_format_metric(supervised.get('best_hep_loss'))} "
                f"| {_format_metric(focal.get('best_hep_loss'))} "
                f"| {_format_metric(entry.get('focal_minus_supervised_best_hep_loss'))} "
                f"| {_format_metric(focal.get('final_residual_loss'))} "
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


def _write_temporal_consistency_residual_objective_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    evidence = report["evidence"]
    lines = [
        "# Temporal-Consistency Residual Objective Decision",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        (
            "- Continue temporal-consistency validation: "
            f"`{report['continue_temporal_consistency_residual_objective_validation']}`"
        ),
        (
            "- Selected variant: "
            f"`{report['selected_residual_objective_variant']}`"
        ),
        f"- Default residual objective: `{report['default_residual_objective']}`",
        f"- Scales: `{', '.join(evidence['scales'])}`",
        (
            "- Best temporal-consistency improvement: "
            f"`{_format_metric(evidence['best_temporal_consistency_improvement'])}`"
        ),
        (
            "- Mean temporal-consistency minus supervised best HEP loss: "
            f"`{_format_metric(evidence['mean_temporal_consistency_minus_supervised_best_hep_loss'])}`"
        ),
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Evidence",
        "",
        (
            "| Scale | Artifact check | Weight | Supervised best HEP loss "
            "| Temporal best HEP loss | Temporal minus supervised "
            "| Temporal final residual loss | Source |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for entry in evidence["entries"]:
        supervised = entry.get("supervised_run") or {}
        temporal_runs = entry.get("temporal_consistency_runs") or []
        for temporal in temporal_runs:
            lines.append(
                (
                    f"| {entry.get('scale') or ''} "
                    f"| {entry.get('artifact_check_status') or ''} "
                    f"| {_format_metric(temporal.get('temporal_consistency_weight'))} "
                    f"| {_format_metric(supervised.get('best_hep_loss'))} "
                    f"| {_format_metric(temporal.get('best_hep_loss'))} "
                    f"| {_format_metric(temporal.get('best_hep_loss_delta_vs_supervised'))} "
                    f"| {_format_metric(temporal.get('final_residual_loss'))} "
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


def _write_residual_learning_next_direction_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    evidence = report["evidence"]
    lines = [
        "# Residual Learning Next Direction",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        f"- Selected direction: `{report['selected_next_direction']}`",
        f"- Default residual objective: `{report['default_residual_objective']}`",
        (
            "- Promote residual learning method: "
            f"`{report['promote_residual_learning_method']}`"
        ),
        f"- Report count: `{evidence['report_count']}`",
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Evidence",
        "",
        "| Report | Status | Decision | Expected decision | Source |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in evidence["entries"]:
        lines.append(
            (
                f"| {entry.get('key') or ''} "
                f"| {entry.get('status') or ''} "
                f"| {entry.get('decision') or ''} "
                f"| {entry.get('expected_decision') or ''} "
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


def _write_residual_capacity_support_diagnostic_gate_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    evidence = report["evidence"]
    lines = [
        "# Residual Capacity Support Diagnostic Gate",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        f"- Selected direction: `{report['selected_next_direction']}`",
        f"- Default residual objective: `{report['default_residual_objective']}`",
        (
            "- Promote residual learning method: "
            f"`{report['promote_residual_learning_method']}`"
        ),
        f"- Config count: `{evidence['config_count']}`",
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Commands",
        "",
        "```bash",
        report["commands"]["compare"],
        report["commands"]["check_artifacts"],
        "```",
        "",
        "## Config Matrix",
        "",
        "| Config | Columns | Top-k | Objective | HEP objective | Failures |",
        "| --- | ---: | ---: | --- | --- | ---: |",
    ]
    for entry in evidence["configs"]:
        lines.append(
            (
                f"| `{entry.get('path')}` "
                f"| {entry.get('num_columns') or ''} "
                f"| {entry.get('top_k') or ''} "
                f"| {entry.get('residual_objective') or ''} "
                f"| {entry.get('hep_settling_objective') or ''} "
                f"| {len(entry.get('failures') or [])} |"
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


def _write_residual_capacity_support_diagnostic_decision_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    evidence = report["evidence"]
    best = evidence["best_variant"] or {}
    accepted = evidence["accepted_support_width_alpha"] or {}
    lines = [
        "# Residual Capacity Support Diagnostic Decision",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        f"- Selected direction: `{report['selected_next_direction']}`",
        f"- Default residual objective: `{report['default_residual_objective']}`",
        f"- Artifact check: `{evidence['artifact_check_status']}`",
        f"- Comparison verdict: `{evidence['verdict_status']}`",
        f"- Best variant: `{best.get('variant')}`",
        (
            "- Support minus baseline best HEP loss: "
            f"`{_format_metric(evidence['support_minus_baseline_best_hep_loss'])}`"
        ),
        (
            "- Accepted support-width alpha: "
            f"`{_format_metric(accepted.get('alpha'))}`"
        ),
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Evidence",
        "",
        "| Variant | Experiment | Best HEP loss | Best alpha | Accepted alpha | Final residual loss | Max support change | Max pinned-vs-repicked |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in evidence["entries"]:
        lines.append(
            (
                f"| {entry.get('variant') or ''} "
                f"| `{entry.get('experiment_id')}` "
                f"| {_format_metric(entry.get('best_hep_loss'))} "
                f"| {_format_metric(entry.get('best_hep_alpha'))} "
                f"| {_format_metric(entry.get('accepted_hep_alpha'))} "
                f"| {_format_metric(entry.get('final_residual_loss'))} "
                f"| {_format_metric(entry.get('max_support_change_fraction'))} "
                f"| {_format_metric(entry.get('max_pinned_vs_repicked_logit_delta'))} |"
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


def _write_residual_capacity_support_diagnostic_colab_decision_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    lines = [
        "# Residual Capacity Support Colab Decision",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        f"- Selected direction: `{report['selected_next_direction']}`",
        f"- Default residual objective: `{report['default_residual_objective']}`",
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Evidence",
        "",
        "| Backend | Artifact check | Verdict | Best variant | Support minus baseline | Accepted support alpha |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for backend in report["evidence"]["backends"]:
        best = backend["best_variant"] or {}
        accepted = backend["accepted_support_width_alpha"] or {}
        lines.append(
            (
                f"| {backend['backend']} "
                f"| `{backend['artifact_check_status']}` "
                f"| `{backend['verdict_status']}` "
                f"| `{best.get('variant')}` "
                f"| {_format_metric(backend['support_minus_baseline_best_hep_loss'])} "
                f"| {_format_metric(accepted.get('alpha'))} |"
            )
        )
    if report["evidence"]["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in report["evidence"]["failures"]:
            lines.append(
                (
                    f"- `{failure.get('field')}` expected "
                    f"`{failure.get('expected')}`, got `{failure.get('actual')}` "
                    f"at `{failure.get('path', '')}`"
                )
            )
    lines.extend(["", "## Next Step", "", report["next_step"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_residual_support_width_validation_gate_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    evidence = report["evidence"]
    lines = [
        "# Residual Support Width Validation Gate",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        f"- Selected direction: `{report['selected_next_direction']}`",
        f"- Default residual objective: `{report['default_residual_objective']}`",
        f"- Config count: `{evidence['config_count']}`",
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Commands",
        "",
        "```bash",
        report["commands"]["compare"],
        report["commands"]["check_artifacts"],
        "```",
        "",
        "## Config Matrix",
        "",
        "| Scale | Support-wide | Config | Dataset | Seq len | Columns | Top-k | Failures |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for entry in evidence["configs"]:
        lines.append(
            (
                f"| {entry.get('scale') or ''} "
                f"| `{entry.get('support_width')}` "
                f"| `{entry.get('path')}` "
                f"| {entry.get('dataset') or ''} "
                f"| {entry.get('seq_len') or ''} "
                f"| {entry.get('num_columns') or ''} "
                f"| {entry.get('top_k') or ''} "
                f"| {len(entry.get('failures') or [])} |"
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


def _write_residual_support_width_validation_decision_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    lines = [
        "# Residual Support Width Validation Decision",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        f"- Selected direction: `{report['selected_next_direction']}`",
        f"- Default residual objective: `{report['default_residual_objective']}`",
        f"- Default support-stress mitigation: `{report['default_support_stress_mitigation']}`",
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Evidence",
        "",
        "| Backend | Scale | Artifact check | Verdict | Baseline alpha-0 | Support alpha-0 | Delta | Baseline final | Support final | Delta |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for backend in report["evidence"]["backends"]:
        for scale in ("larger_char", "tokenized"):
            scale_evidence = backend["scales"].get(scale)
            if scale_evidence is None:
                lines.append(
                    (
                        f"| {backend['backend']} | {scale} "
                        f"| `{backend['artifact_check_status']}` "
                        f"| `{backend['verdict_status']}` "
                        "|  |  |  |  |  |  |"
                    )
                )
                continue
            baseline = scale_evidence["baseline"]
            support = scale_evidence["support_width"]
            lines.append(
                (
                    f"| {backend['backend']} "
                    f"| {scale} "
                    f"| `{backend['artifact_check_status']}` "
                    f"| `{backend['verdict_status']}` "
                    f"| {_format_metric(baseline.get('alpha0_loss'))} "
                    f"| {_format_metric(support.get('alpha0_loss'))} "
                    f"| {_format_metric(scale_evidence['support_minus_baseline_alpha0_loss'])} "
                    f"| {_format_metric(baseline.get('final_residual_loss'))} "
                    f"| {_format_metric(support.get('final_residual_loss'))} "
                    f"| {_format_metric(scale_evidence['support_minus_baseline_final_loss'])} |"
                )
            )
    if report["evidence"]["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in report["evidence"]["failures"]:
            lines.append(
                (
                    f"- `{failure.get('field')}` expected "
                    f"`{failure.get('expected')}`, got `{failure.get('actual')}` "
                    f"at `{failure.get('path', '')}`"
                )
            )
    lines.extend(["", "## Next Step", "", report["next_step"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_focal_promotion_gate_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    evidence = report["evidence"]
    lines = [
        "# Focal Residual Objective Promotion Gate",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        (
            "- Promote residual learning method: "
            f"`{report['promote_residual_learning_method']}`"
        ),
        (
            "- Selected variant: "
            f"`{report['selected_residual_objective_variant']}`"
        ),
        f"- Default residual objective: `{report['default_residual_objective']}`",
        f"- Focal decision report: `{evidence['focal_decision_report_path']}`",
        f"- Focal decision status: `{evidence['focal_decision_status']}`",
        f"- Focal decision: `{evidence['focal_decision']}`",
        (
            "- Mean focal minus supervised best HEP loss: "
            f"`{_format_metric(evidence['mean_focal_minus_supervised_best_hep_loss'])}`"
        ),
        (
            "- Mean focal minus supervised final residual loss: "
            f"`{_format_metric(evidence['mean_focal_minus_supervised_final_residual_loss'])}`"
        ),
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Required Evidence",
        "",
        (
            "| Gate | Dataset | Backends | Minimum seq len | Minimum hidden dim "
            "| Minimum columns | Minimum steps |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for requirement in evidence["required_evidence"]:
        minimum = requirement["minimum_scale"]
        lines.append(
            (
                f"| {requirement['gate']} "
                f"| {minimum['dataset']} "
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


def _write_focal_promotion_gate_satisfaction_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    evidence = report["evidence"]
    lines = [
        "# Focal Residual Objective Promotion Gate Satisfaction",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        f"- Promotion gate satisfied: `{report['promotion_gate_satisfied']}`",
        (
            "- Promote residual learning method: "
            f"`{report['promote_residual_learning_method']}`"
        ),
        (
            "- Selected variant: "
            f"`{report['selected_residual_objective_variant']}`"
        ),
        f"- Default residual objective: `{report['default_residual_objective']}`",
        f"- Promotion gate report: `{evidence['promotion_gate_report_path']}`",
        f"- Promotion gate status: `{evidence['promotion_gate_status']}`",
        f"- Promotion gate decision: `{evidence['promotion_gate_decision']}`",
        f"- Focal CE win count: `{evidence['focal_ce_win_count']}`",
        (
            "- Mean focal minus supervised best HEP loss: "
            f"`{_format_metric(evidence['mean_focal_minus_supervised_best_hep_loss'])}`"
        ),
        (
            "- Mean focal minus supervised final residual loss: "
            f"`{_format_metric(evidence['mean_focal_minus_supervised_final_residual_loss'])}`"
        ),
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Evidence",
        "",
        (
            "| Gate | Backend | Artifact check | Supervised best HEP loss "
            "| Focal best HEP loss | Focal minus supervised | Source |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for entry in evidence["entries"]:
        supervised = entry.get("supervised_run") or {}
        focal = entry.get("focal_run") or {}
        lines.append(
            (
                f"| {entry.get('gate') or ''} "
                f"| {entry.get('backend') or ''} "
                f"| {entry.get('artifact_check_status') or ''} "
                f"| {_format_metric(supervised.get('best_hep_loss'))} "
                f"| {_format_metric(focal.get('best_hep_loss'))} "
                f"| {_format_metric(entry.get('focal_minus_supervised_best_hep_loss'))} "
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
            "confidence-penalty-residual-objective-decision",
            "margin-penalty-residual-objective-decision",
            "label-smoothing-residual-objective-decision",
            "focal-residual-objective-decision",
            "focal-residual-objective-promotion-gate",
            "focal-residual-objective-promotion-gate-satisfaction",
            "temporal-consistency-residual-objective-decision",
            "residual-learning-next-direction",
            "residual-capacity-support-diagnostic-gate",
            "residual-capacity-support-diagnostic-decision",
            "residual-capacity-support-diagnostic-colab-decision",
            "residual-support-width-validation-gate",
            "residual-support-width-validation-decision",
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
    elif args.report == "confidence-penalty-residual-objective-decision":
        report = write_confidence_penalty_residual_objective_decision_report(
            tuple(args.decision_report)
            if args.decision_report
            else DEFAULT_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_COMPARISON_DIRS,
            args.out or DEFAULT_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_OUT_DIR,
            artifact_check_paths=DEFAULT_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_ARTIFACT_CHECKS
            if not args.artifact_check
            else (args.artifact_check,),
            max_logit_delta=args.max_logit_delta,
        )
    elif args.report == "margin-penalty-residual-objective-decision":
        report = write_margin_penalty_residual_objective_decision_report(
            tuple(args.decision_report)
            if args.decision_report
            else DEFAULT_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_COMPARISON_DIRS,
            args.out or DEFAULT_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_OUT_DIR,
            artifact_check_paths=DEFAULT_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_ARTIFACT_CHECKS
            if not args.artifact_check
            else (args.artifact_check,),
            max_logit_delta=args.max_logit_delta,
        )
    elif args.report == "label-smoothing-residual-objective-decision":
        report = write_label_smoothing_residual_objective_decision_report(
            tuple(args.decision_report)
            if args.decision_report
            else DEFAULT_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_COMPARISON_DIRS,
            args.out or DEFAULT_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_OUT_DIR,
            artifact_check_paths=DEFAULT_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_ARTIFACT_CHECKS
            if not args.artifact_check
            else (args.artifact_check,),
            max_logit_delta=args.max_logit_delta,
        )
    elif args.report == "focal-residual-objective-decision":
        report = write_focal_residual_objective_decision_report(
            tuple(args.decision_report)
            if args.decision_report
            else DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_COMPARISON_DIRS,
            args.out or DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_OUT_DIR,
            artifact_check_paths=DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_ARTIFACT_CHECKS
            if not args.artifact_check
            else (args.artifact_check,),
            max_logit_delta=args.max_logit_delta,
        )
    elif args.report == "focal-residual-objective-promotion-gate":
        report = write_focal_residual_objective_promotion_gate_report(
            args.decision_report[0]
            if args.decision_report
            else DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_DECISION_REPORT,
            args.out or DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_OUT_DIR,
        )
    elif args.report == "focal-residual-objective-promotion-gate-satisfaction":
        comparison_dirs = (
            tuple(args.decision_report[1:])
            if args.decision_report and len(args.decision_report) > 1
            else DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_SATISFACTION_COMPARISON_DIRS
        )
        report = write_focal_residual_objective_promotion_gate_satisfaction_report(
            args.decision_report[0]
            if args.decision_report
            else DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_SATISFACTION_REPORT,
            comparison_dirs,
            args.out
            or DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_SATISFACTION_OUT_DIR,
            artifact_check_paths=DEFAULT_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE_SATISFACTION_ARTIFACT_CHECKS
            if not args.artifact_check
            else (args.artifact_check,),
            max_logit_delta=args.max_logit_delta,
        )
    elif args.report == "temporal-consistency-residual-objective-decision":
        report = write_temporal_consistency_residual_objective_decision_report(
            tuple(args.decision_report)
            if args.decision_report
            else DEFAULT_TEMPORAL_CONSISTENCY_RESIDUAL_OBJECTIVE_COMPARISON_DIRS,
            args.out or DEFAULT_TEMPORAL_CONSISTENCY_RESIDUAL_OBJECTIVE_OUT_DIR,
            artifact_check_paths=DEFAULT_TEMPORAL_CONSISTENCY_RESIDUAL_OBJECTIVE_ARTIFACT_CHECKS
            if not args.artifact_check
            else (args.artifact_check,),
            max_logit_delta=args.max_logit_delta,
        )
    elif args.report == "residual-learning-next-direction":
        report = write_residual_learning_next_direction_report(
            tuple(args.decision_report)
            if args.decision_report
            else DEFAULT_RESIDUAL_LEARNING_NEXT_DIRECTION_REPORTS,
            args.out or DEFAULT_RESIDUAL_LEARNING_NEXT_DIRECTION_OUT_DIR,
        )
    elif args.report == "residual-capacity-support-diagnostic-gate":
        report = write_residual_capacity_support_diagnostic_gate_report(
            args.decision_report[0]
            if args.decision_report
            else DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_GATE_REPORT,
            tuple(args.decision_report[1:])
            if args.decision_report and len(args.decision_report) > 1
            else DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_GATE_CONFIGS,
            args.out or DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_GATE_OUT_DIR,
        )
    elif args.report == "residual-capacity-support-diagnostic-decision":
        report = write_residual_capacity_support_diagnostic_decision_report(
            args.comparison_dir
            or DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_COMPARISON_DIR,
            args.out
            or DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_DECISION_OUT_DIR,
            artifact_check_path=args.artifact_check
            or DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_ARTIFACT_CHECK,
            max_logit_delta=args.max_logit_delta,
        )
    elif args.report == "residual-capacity-support-diagnostic-colab-decision":
        report = write_residual_capacity_support_diagnostic_colab_decision_report(
            tuple(args.decision_report)
            if args.decision_report
            else DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_COLAB_COMPARISON_DIRS,
            args.out
            or DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_COLAB_DECISION_OUT_DIR,
            artifact_check_paths=DEFAULT_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_COLAB_ARTIFACT_CHECKS
            if not args.artifact_check
            else (args.artifact_check,),
            max_logit_delta=args.max_logit_delta,
        )
    elif args.report == "residual-support-width-validation-gate":
        report = write_residual_support_width_validation_gate_report(
            args.decision_report[0]
            if args.decision_report
            else DEFAULT_RESIDUAL_SUPPORT_WIDTH_VALIDATION_GATE_REPORT,
            tuple(args.decision_report[1:])
            if args.decision_report and len(args.decision_report) > 1
            else DEFAULT_RESIDUAL_SUPPORT_WIDTH_VALIDATION_GATE_CONFIGS,
            args.out or DEFAULT_RESIDUAL_SUPPORT_WIDTH_VALIDATION_GATE_OUT_DIR,
        )
    elif args.report == "residual-support-width-validation-decision":
        report = write_residual_support_width_validation_decision_report(
            tuple(args.decision_report)
            if args.decision_report
            else DEFAULT_RESIDUAL_SUPPORT_WIDTH_VALIDATION_COMPARISON_DIRS,
            args.out or DEFAULT_RESIDUAL_SUPPORT_WIDTH_VALIDATION_DECISION_OUT_DIR,
            artifact_check_paths=DEFAULT_RESIDUAL_SUPPORT_WIDTH_VALIDATION_ARTIFACT_CHECKS
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
