"""Next-evidence selector after active top-k-1 post-decomposition closeout."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.active_topk1_backend_provenance_manifest import (
    ACTIVE_TOPK1_BACKEND_PROVENANCE_ESTABLISHED,
)
from relaleap.experiments.active_topk1_functional_retention_audit import (
    FUNCTIONAL_RETENTION_BRACKET_ONLY,
)
from relaleap.experiments.active_topk1_post_decomposition_decision_report import (
    BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED,
    COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS,
)
from relaleap.experiments.active_topk1_runpod_post_decomposition_closeout_report import (
    RUNPOD_POST_DECOMPOSITION_VALIDATED,
)
from relaleap.experiments.active_topk1_singleton_reconciliation_audit import (
    CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
)


DEFAULT_CLOSEOUT_DIR = Path(
    "results/reports/token_larger_active_topk1_runpod_post_decomposition_closeout"
)
DEFAULT_DIRECTION_DIR = Path(
    "results/reports/token_larger_active_topk1_post_bracket_research_direction"
)
DEFAULT_RETENTION_DIR = Path(
    "results/reports/token_larger_active_topk1_functional_retention_audit"
)
DEFAULT_RETENTION_STABILITY_DIR = Path(
    "results/reports/token_larger_active_rank_matched_topk1_retention_churn_stability"
)
DEFAULT_PROVENANCE_DIR = Path(
    "results/reports/token_larger_active_topk1_backend_provenance_manifest"
)
DEFAULT_CAUSAL_BRACKET_DIR = Path(
    "results/reports/token_larger_active_rank_matched_topk1_causal_bracket_audit"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_active_topk1_next_evidence_selection"
)

NEXT_EVIDENCE_SELECTED = "active_topk1_next_evidence_selected"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
SELECTED_EXPERIMENT = "context_gate_suppression_calibration_audit"

_SELECTED_EXPERIMENT_ROWS = (
    {
        "component": "threshold_sweep_on_existing_context_gate_holdout",
        "purpose": "find a gate threshold that keeps positive own-context holdout gain while suppressing contexts with predicted off-context harm",
    },
    {
        "component": "abstention_vs_routed_baseline_control",
        "purpose": "distinguish clean gating from simply disabling residual columns everywhere",
    },
    {
        "component": "offcontext_harm_suppression_metric",
        "purpose": "measure how much negative forced-singleton reuse remains after the gate abstains",
    },
    {
        "component": "own_context_gain_retention_metric",
        "purpose": "ensure the gate preserves the context-conditioned singleton benefit",
    },
    {
        "component": "matched_topk2_random_dense_controls",
        "purpose": "keep the existing reference and low-rank controls attached to the next claim",
    },
)

_REQUIRED_SIGNALS = (
    "own_context_singleton_gain_positive",
    "offcontext_singleton_interference_present",
    "context_gate_holdout_net_gain_positive",
    "context_gate_improves_over_ungated_holdout",
    "matched_topk2_reference_present",
    "random_control_present",
    "exhaustive_control_present",
)


def run_active_topk1_next_evidence_selection_report(
    *,
    closeout_dir: Path = DEFAULT_CLOSEOUT_DIR,
    direction_dir: Path = DEFAULT_DIRECTION_DIR,
    retention_dir: Path = DEFAULT_RETENTION_DIR,
    retention_stability_dir: Path = DEFAULT_RETENTION_STABILITY_DIR,
    provenance_dir: Path = DEFAULT_PROVENANCE_DIR,
    causal_bracket_dir: Path = DEFAULT_CAUSAL_BRACKET_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Select exactly one non-duplicative follow-up experiment from source reports."""

    start = time.time()
    closeout = _read_json_object(closeout_dir / "summary.json")
    direction = _read_json_object(direction_dir / "summary.json")
    retention = _read_json_object(retention_dir / "summary.json")
    retention_stability = _read_json_object(retention_stability_dir / "summary.json")
    provenance = _read_json_object(provenance_dir / "summary.json")
    causal_bracket = _read_json_object(causal_bracket_dir / "decision_report.json")
    strategy_review = _strategy_review(strategy_review_path)
    metrics = _closeout_metrics(closeout)
    signals = _closeout_signals(closeout)

    source_rows = _source_rows(
        closeout_dir=closeout_dir,
        direction_dir=direction_dir,
        retention_dir=retention_dir,
        retention_stability_dir=retention_stability_dir,
        provenance_dir=provenance_dir,
        causal_bracket_dir=causal_bracket_dir,
        closeout=closeout,
        direction=direction,
        retention=retention,
        retention_stability=retention_stability,
        provenance=provenance,
        causal_bracket=causal_bracket,
        strategy_review=strategy_review,
    )
    failures = _failures(
        source_rows=source_rows,
        closeout=closeout,
        direction=direction,
        retention=retention,
        retention_stability=retention_stability,
        provenance=provenance,
        causal_bracket=causal_bracket,
        metrics=metrics,
        signals=signals,
    )

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        selected_experiment = None
        claim_status = INSUFFICIENT_EVIDENCE
        claim_policy = INSUFFICIENT_EVIDENCE
        next_step = "repair_missing_or_inconsistent_next_evidence_sources"
        rationale = (
            "The next-evidence selector cannot choose a follow-up because one or "
            "more required source reports are missing, failing, or inconsistent with "
            "the validated column-plus-context-gate interpretation."
        )
    else:
        status = "pass"
        decision = NEXT_EVIDENCE_SELECTED
        selected_experiment = SELECTED_EXPERIMENT
        claim_status = COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS
        claim_policy = BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED
        next_step = (
            "implement and run the local no-training context-gate suppression "
            "calibration audit using the validated interference CSV artifacts"
        )
        rationale = (
            "The RunPod-validated decomposition already establishes positive "
            "own-context singleton gain, negative off-context forced-singleton reuse, "
            "and positive context-gated holdout gain. Repeating backend validation or "
            "adding another wrapper would duplicate completed work. The next useful "
            "evidence is a bounded no-training gate-suppression audit that tests "
            "whether off-context interference can be cleanly gated away while "
            "retaining the in-context singleton gain."
        )

    summary = {
        "status": status,
        "decision": decision,
        "selected_experiment": selected_experiment,
        "claim_status": claim_status,
        "claim_policy": claim_policy,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "source_rows": source_rows,
        "evidence_snapshot": {
            "metrics": metrics,
            "signals": signals,
        },
        "selection_gate": {
            "requires_gpu_now": False,
            "new_training_required": False,
            "duplicates_completed_work": False,
            "selected_because": [
                "runpod_closeout_already_validated",
                "offcontext_singleton_interference_remains_present",
                "context_gate_positive_but_not_yet_clean_suppression_policy",
                "broad_reusable_singleton_claim_still_excluded",
            ],
        },
        "experiment_design": {
            "components": list(_SELECTED_EXPERIMENT_ROWS),
            "success_criteria": {
                "offcontext_harm_after_gate_mean": "<= 0.0 or materially below ungated off-context harm",
                "own_context_gain_retention_fraction": ">= 0.8 of validated own-context singleton gain",
                "context_gated_net_gain_holdout_mean": "> 0.0",
                "broad_reusable_singleton_claim_policy": BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED,
            },
            "source_artifacts": [
                "results/audits/token_larger_active_topk1_context_conditioned_singleton_interference/context_gate_holdout.csv",
                "results/audits/token_larger_active_topk1_context_conditioned_singleton_interference/singleton_interference_by_context.csv",
                "results/audits/token_larger_active_topk1_context_conditioned_singleton_interference/singleton_interference_by_stratum.csv",
            ],
        },
        "strategy_review": strategy_review,
        "failures": failures,
        "rationale": rationale,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "selected_experiment_csv": str(out_dir / "selected_experiment.csv"),
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
        out_dir / "selected_experiment.csv",
        ["component", "purpose"],
        list(_SELECTED_EXPERIMENT_ROWS),
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _source_rows(
    *,
    closeout_dir: Path,
    direction_dir: Path,
    retention_dir: Path,
    retention_stability_dir: Path,
    provenance_dir: Path,
    causal_bracket_dir: Path,
    closeout: dict[str, Any],
    direction: dict[str, Any],
    retention: dict[str, Any],
    retention_stability: dict[str, Any],
    provenance: dict[str, Any],
    causal_bracket: dict[str, Any],
    strategy_review: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _row("runpod_post_decomposition_closeout", closeout_dir / "summary.json", closeout),
        _row("post_bracket_direction", direction_dir / "summary.json", direction),
        _row("functional_retention_audit", retention_dir / "summary.json", retention),
        _row(
            "retention_churn_stability",
            retention_stability_dir / "summary.json",
            retention_stability,
        ),
        _row("backend_provenance_manifest", provenance_dir / "summary.json", provenance),
        _row(
            "active_rank_matched_topk1_causal_bracket",
            causal_bracket_dir / "decision_report.json",
            causal_bracket,
        ),
        {
            "source": "strategy_review",
            "path": strategy_review.get("path"),
            "present": strategy_review.get("present"),
            "status": "present" if strategy_review.get("present") else "missing_optional",
            "decision": strategy_review.get("recommended_next_action"),
            "claim_status": (
                f"strategic_change_level={strategy_review.get('strategic_change_level')}; "
                f"notify_ben={strategy_review.get('notify_ben')}"
            ),
        },
    ]


def _row(source: str, path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": packet.get("status"),
        "decision": packet.get("decision"),
        "claim_status": packet.get("claim_status") or packet.get("claim_policy"),
    }


def _failures(
    *,
    source_rows: list[dict[str, Any]],
    closeout: dict[str, Any],
    direction: dict[str, Any],
    retention: dict[str, Any],
    retention_stability: dict[str, Any],
    provenance: dict[str, Any],
    causal_bracket: dict[str, Any],
    metrics: dict[str, Any],
    signals: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows[:6]:
        if not row["present"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "summary_json",
                    "expected": "file exists",
                    "actual": "missing",
                    "path": row["path"],
                }
            )
    expectations = (
        (
            "runpod_post_decomposition_closeout",
            closeout,
            "decision",
            RUNPOD_POST_DECOMPOSITION_VALIDATED,
        ),
        (
            "runpod_post_decomposition_closeout",
            closeout,
            "claim_status",
            COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS,
        ),
        (
            "runpod_post_decomposition_closeout",
            closeout,
            "claim_policy",
            BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED,
        ),
        ("post_bracket_direction", direction, "status", "pass"),
        (
            "functional_retention_audit",
            retention,
            "decision",
            FUNCTIONAL_RETENTION_BRACKET_ONLY,
        ),
        (
            "functional_retention_audit",
            retention,
            "claim_status",
            CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
        ),
        (
            "retention_churn_stability",
            retention_stability,
            "decision",
            "active_topk1_retention_churn_stable_across_local_seeds",
        ),
        (
            "backend_provenance_manifest",
            provenance,
            "decision",
            ACTIVE_TOPK1_BACKEND_PROVENANCE_ESTABLISHED,
        ),
        (
            "active_rank_matched_topk1_causal_bracket",
            causal_bracket,
            "decision",
            "confirm_active_rank_matched_topk1_causal_bracket",
        ),
    )
    for source, packet, field, expected in expectations:
        if packet.get(field) != expected:
            failures.append(
                {
                    "source": source,
                    "field": field,
                    "expected": expected,
                    "actual": packet.get(field),
                }
            )
    for signal in _REQUIRED_SIGNALS:
        if not signals.get(signal):
            failures.append(
                {
                    "source": "runpod_post_decomposition_closeout",
                    "field": signal,
                    "expected": True,
                    "actual": signals.get(signal),
                }
            )
    numeric_expectations = (
        ("own_context_singleton_gain_mean", ">", 0.0),
        ("off_context_singleton_gain_mean", "<", 0.0),
        ("context_gated_net_gain_holdout_mean", ">", 0.0),
        ("context_gate_gain_minus_ungated_holdout_mean", ">", 0.0),
    )
    for field, operator, threshold in numeric_expectations:
        value = metrics.get(field)
        ok = _gt(value, threshold) if operator == ">" else _lt(value, threshold)
        if not ok:
            failures.append(
                {
                    "source": "runpod_post_decomposition_closeout",
                    "field": field,
                    "expected": f"{operator} {threshold}",
                    "actual": value,
                }
            )
    return failures


def _closeout_metrics(closeout: dict[str, Any]) -> dict[str, Any]:
    return {
        str(row.get("field")): row.get("local")
        for row in closeout.get("metric_comparison", [])
        if isinstance(row, dict) and row.get("match")
    }


def _closeout_signals(closeout: dict[str, Any]) -> dict[str, Any]:
    return {
        str(row.get("field")): row.get("local")
        for row in closeout.get("signal_comparison", [])
        if isinstance(row, dict) and row.get("match")
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "strategic_change_level": None,
            "notify_ben": None,
            "recommended_next_action": None,
            "incorporation": "optional review not present",
            "ben_notification_required": False,
        }
    lines = path.read_text(encoding="utf-8").splitlines()
    header = {}
    for line in lines[:12]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"strategic_change_level", "notify_ben", "recommended_next_action"}:
            header[key] = value.strip()
    notify_ben = _bool_or_none(header.get("notify_ben"))
    major = header.get("strategic_change_level") == "major"
    return {
        "path": str(path),
        "present": True,
        "strategic_change_level": header.get("strategic_change_level"),
        "notify_ben": notify_ben,
        "recommended_next_action": header.get("recommended_next_action"),
        "incorporation": (
            "followed where still applicable: the recommended context-conditioned "
            "interference decomposition and RunPod closeout are complete; this report "
            "selects the next bounded local gate-suppression audit rather than "
            "duplicating backend validation"
        ),
        "ben_notification_required": bool(notify_ben) or major,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _bool_or_none(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def _gt(value: Any, threshold: float) -> bool:
    return isinstance(value, (float, int)) and float(value) > threshold


def _lt(value: Any, threshold: float) -> bool:
    return isinstance(value, (float, int)) and float(value) < threshold


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


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["evidence_snapshot"]["metrics"]
    lines = [
        "# Active Top-k-1 Next-Evidence Selection",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected experiment: `{summary['selected_experiment']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Claim policy: `{summary['claim_policy']}`",
        f"- Requires GPU now: `{summary['selection_gate']['requires_gpu_now']}`",
        f"- New training required: `{summary['selection_gate']['new_training_required']}`",
        f"- Git commit: `{summary['git_commit']}`",
        "",
        "## Key Evidence",
        "",
        f"- Own-context singleton gain mean: `{metrics.get('own_context_singleton_gain_mean')}`",
        f"- Off-context singleton gain mean: `{metrics.get('off_context_singleton_gain_mean')}`",
        f"- Context-gated holdout net gain: `{metrics.get('context_gated_net_gain_holdout_mean')}`",
        f"- Context gate minus ungated holdout: `{metrics.get('context_gate_gain_minus_ungated_holdout_mean')}`",
        "",
        "## Rationale",
        "",
        summary["rationale"],
        "",
        "## Next Step",
        "",
        summary["next_step"],
        "",
        "## Strategy Review",
        "",
        f"- Present: `{summary['strategy_review']['present']}`",
        f"- Strategic change level: `{summary['strategy_review']['strategic_change_level']}`",
        f"- Notify Ben: `{summary['strategy_review']['notify_ben']}`",
        f"- Ben notification required: `{summary['strategy_review']['ben_notification_required']}`",
        f"- Incorporation: {summary['strategy_review']['incorporation']}",
        "",
    ]
    if summary["failures"]:
        lines.extend(["## Failures", ""])
        for failure in summary["failures"]:
            lines.append(f"- `{failure}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closeout-dir", type=Path, default=DEFAULT_CLOSEOUT_DIR)
    parser.add_argument("--direction-dir", type=Path, default=DEFAULT_DIRECTION_DIR)
    parser.add_argument("--retention-dir", type=Path, default=DEFAULT_RETENTION_DIR)
    parser.add_argument(
        "--retention-stability-dir", type=Path, default=DEFAULT_RETENTION_STABILITY_DIR
    )
    parser.add_argument("--provenance-dir", type=Path, default=DEFAULT_PROVENANCE_DIR)
    parser.add_argument("--causal-bracket-dir", type=Path, default=DEFAULT_CAUSAL_BRACKET_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_active_topk1_next_evidence_selection_report(
        closeout_dir=args.closeout_dir,
        direction_dir=args.direction_dir,
        retention_dir=args.retention_dir,
        retention_stability_dir=args.retention_stability_dir,
        provenance_dir=args.provenance_dir,
        causal_bracket_dir=args.causal_bracket_dir,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "selected_experiment": summary["selected_experiment"],
                "next_step": summary["next_step"],
                "failures": summary["failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
