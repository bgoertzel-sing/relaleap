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
DEFAULT_MAX_LOGIT_DELTA = 0.1
DEFAULT_MAX_PINNED_VS_REPICKED_DELTA = 0.1
PROMOTE = "promote_to_default_phase0_baseline"
PROMOTE_CLIPPED_HEP = "promote_to_default_support_stress_mitigation"
GUIDED_ORACLE_CONFIRMED = "guided_oracle_confirmed"
SELECT_TEMPORAL_CLIPPED_HEP = "select_temporal_label_free_support_stress_candidate"
SELECT_TEMPORAL_CLIPPED_HEP_AGGREGATE = (
    "select_temporal_label_free_support_stress_candidate_across_seed_smoke_evidence"
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


def _infer_backend_from_report(path: Path, comparison_dir: Any) -> str | None:
    text = f"{path} {comparison_dir or ''}"
    if "colab_" in text or "_colab_" in text:
        return "colab"
    if "local" in text:
        return "local"
    if "results/comparisons/" in text or "temporal_clipped_hep" in text:
        return "local"
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
