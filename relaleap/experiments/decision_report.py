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
DEFAULT_MAX_LOGIT_DELTA = 0.1
DEFAULT_MAX_PINNED_VS_REPICKED_DELTA = 0.1
PROMOTE = "promote_to_default_phase0_baseline"
PROMOTE_CLIPPED_HEP = "promote_to_default_support_stress_mitigation"
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
        choices=("pinned-support", "clipped-hep"),
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
