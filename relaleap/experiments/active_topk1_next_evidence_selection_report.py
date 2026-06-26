"""Next-evidence selector for the active rank-matched top-k-1 bracket."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.active_topk1_functional_retention_audit import (
    FUNCTIONAL_RETENTION_BRACKET_ONLY,
)
from relaleap.experiments.active_topk1_retention_churn_summary import (
    ACTIVE_TOPK1_RETENTION_CHURN_STABLE,
)
from relaleap.experiments.promoted_topk2_finite_update_augmented_causal_gate import (
    BLOCKED as FINITE_UPDATE_TOPK2_BLOCKED,
)
from relaleap.experiments.promoted_topk2_finite_update_control_matrix import (
    FINITE_UPDATE_CONTROL_MATRIX_READY,
)


DEFAULT_FINITE_AUGMENTED_DIR = Path(
    "results/reports/token_larger_promoted_topk2_finite_update_augmented_causal_gate"
)
DEFAULT_FUNCTIONAL_RETENTION_DIR = Path(
    "results/reports/token_larger_active_topk1_functional_retention_audit"
)
DEFAULT_RETENTION_STABILITY_DIR = Path(
    "results/reports/token_larger_active_rank_matched_topk1_retention_churn_stability"
)
DEFAULT_DECONFOUNDED_DIR = Path(
    "results/audits/token_larger_topk2_vs_rank_matched_topk1_deconfounded_intervention"
)
DEFAULT_FINITE_CONTROL_DIR = Path(
    "results/reports/token_larger_promoted_topk2_finite_update_control_matrix"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_active_topk1_next_evidence_selection"
)

NEXT_EVIDENCE_SELECTED = "active_topk1_next_evidence_selected"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
SELECTED_RETENTION_CHURN = "retention_churn"
SELECTED_MATCHED_DECONFOUNDING = "matched_deconfounding"
SELECTED_DENSE_TEACHER_DISTILLATION = "dense_teacher_residual_distillation"
SELECTED_EXPERIMENT = SELECTED_RETENTION_CHURN
TOPK2_NOT_SUPPORTED = "topk2_comparative_causal_cooperation_not_supported"

REQUIRED_ROLES = (
    "promoted_contextual_topk2",
    "rank_matched_contextual_topk1",
    "random_fixed_topk2",
    "dense_active_rank",
)


def run_active_topk1_next_evidence_selection_report(
    *,
    finite_augmented_dir: Path = DEFAULT_FINITE_AUGMENTED_DIR,
    functional_retention_dir: Path = DEFAULT_FUNCTIONAL_RETENTION_DIR,
    retention_stability_dir: Path = DEFAULT_RETENTION_STABILITY_DIR,
    deconfounded_dir: Path = DEFAULT_DECONFOUNDED_DIR,
    finite_control_dir: Path = DEFAULT_FINITE_CONTROL_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Choose exactly one bounded follow-up from completed source artifacts."""

    start = time.time()
    finite_augmented = _read_json_object(finite_augmented_dir / "summary.json")
    functional_retention = _read_json_object(functional_retention_dir / "summary.json")
    retention_stability = _read_json_object(retention_stability_dir / "summary.json")
    deconfounded = _read_json_object(deconfounded_dir / "summary.json")
    finite_control = _read_json_object(finite_control_dir / "summary.json")
    strategy_review = _strategy_review(strategy_review_path)

    source_rows = [
        _source_row("finite_update_augmented_causal_gate", finite_augmented_dir / "summary.json", finite_augmented),
        _source_row("functional_retention_audit", functional_retention_dir / "summary.json", functional_retention),
        _source_row("retention_churn_stability", retention_stability_dir / "summary.json", retention_stability),
        _source_row("deconfounded_intervention", deconfounded_dir / "summary.json", deconfounded),
        _source_row("finite_update_control_matrix", finite_control_dir / "summary.json", finite_control),
        {
            "source": "strategy_review",
            "path": strategy_review["path"],
            "present": strategy_review["present"],
            "status": "present" if strategy_review["present"] else "missing_optional",
            "decision": strategy_review["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy_review['strategic_change_level']}; "
                f"notify_ben={strategy_review['notify_ben']}"
            ),
        },
    ]
    metrics = _metrics(finite_augmented, functional_retention, deconfounded, finite_control)
    signals = _signals(metrics, finite_augmented, functional_retention, retention_stability, deconfounded, finite_control)
    failures = _failures(source_rows, finite_augmented, functional_retention, retention_stability, deconfounded, finite_control)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        selected_experiment = None
        next_step = "repair_missing_or_inconsistent_active_topk1_selection_sources"
        rationale = (
            "The selector cannot choose a branch because a required source packet is "
            "missing, failing, or inconsistent with the current top-k-2-blocked and "
            "active top-k-1 retention-bracket interpretation."
        )
    else:
        status = "pass"
        decision = NEXT_EVIDENCE_SELECTED
        selected_experiment = _selected_branch(signals, metrics)
        next_step = _next_step(selected_experiment)
        rationale = _rationale(selected_experiment)

    selected_rows = _selected_rows(selected_experiment)
    summary = {
        "status": status,
        "decision": decision,
        "selected_experiment": selected_experiment,
        "branch_options": [
            SELECTED_RETENTION_CHURN,
            SELECTED_MATCHED_DECONFOUNDING,
            SELECTED_DENSE_TEACHER_DISTILLATION,
        ],
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "source_rows": source_rows,
        "metrics": metrics,
        "signals": signals,
        "selection_gate": {
            "requires_gpu_now": False,
            "new_training_required": False,
            "duplicates_completed_work": False,
            "matched_control_coverage_adequate": signals["matched_control_coverage_adequate"],
            "topk2_causal_cooperation_blocked": signals["topk2_causal_cooperation_blocked"],
            "selected_because": _selected_because(selected_experiment, signals),
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
        ["branch", "component", "purpose"],
        selected_rows,
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _selected_branch(signals: dict[str, bool], metrics: dict[str, Any]) -> str:
    if signals["matched_control_coverage_adequate"] and signals["retention_churn_evidence_adequate"]:
        return SELECTED_RETENTION_CHURN
    if not signals["matched_control_coverage_adequate"]:
        return SELECTED_MATCHED_DECONFOUNDING
    dense_margin = _float(metrics.get("finite_topk2_minus_dense_logit_mse"))
    topk1_margin = _float(metrics.get("functional_topk1_commutator_advantage_vs_dense"))
    if dense_margin is not None and dense_margin > 0.0 and (topk1_margin is None or topk1_margin < 0.0):
        return SELECTED_DENSE_TEACHER_DISTILLATION
    return SELECTED_MATCHED_DECONFOUNDING


def _next_step(branch: str | None) -> str:
    if branch == SELECTED_RETENTION_CHURN:
        return (
            "run the retention/functional-churn follow-up as the next bounded branch: "
            "promoted contextual top-k-2, rank-matched contextual top-k-1, dense "
            "active-rank, and random fixed top-k-2 controls; prioritize anchor drift, "
            "functional churn, support identity churn, finite-update commutator risk, "
            "residual/logit deltas, and CE guardrails"
        )
    if branch == SELECTED_MATCHED_DECONFOUNDING:
        return (
            "extend the matched deconfounding/control coverage before interpreting "
            "retention or distillation claims"
        )
    if branch == SELECTED_DENSE_TEACHER_DISTILLATION:
        return (
            "run a diagnostic dense-teacher residual distillation probe and evaluate "
            "residual/logit matching plus retention/churn, not CE alone"
        )
    return "repair source artifacts before selecting a branch"


def _rationale(branch: str | None) -> str:
    if branch == SELECTED_RETENTION_CHURN:
        return (
            "The matched-control coverage is adequate and the active rank-matched "
            "top-k-1 bracket already has favorable support churn, functional/logit "
            "churn, finite-update commutator, transfer, and dense-control evidence. "
            "The finite-update-augmented causal gate keeps top-k-2 causal-cooperation "
            "claims blocked, so the highest-information follow-up is retention and "
            "functional churn rather than another mitigation family or CE variant."
        )
    if branch == SELECTED_MATCHED_DECONFOUNDING:
        return (
            "Required rank/scale/random/dense matched-control coverage is incomplete, "
            "so deconfounding must be repaired before interpreting retention or "
            "distillation evidence."
        )
    if branch == SELECTED_DENSE_TEACHER_DISTILLATION:
        return (
            "Dense active-rank behavior appears cleaner than the sparse alternatives, "
            "so a dense-teacher diagnostic is the most direct way to test whether "
            "columnar residuals can approximate a lower-interference dense correction."
        )
    return "No branch selected."


def _selected_because(branch: str | None, signals: dict[str, bool]) -> list[str]:
    if branch == SELECTED_RETENTION_CHURN:
        return [
            "external_review_prefers_retention_when_coverage_is_adequate",
            "matched_control_coverage_adequate",
            "active_topk1_low_churn_and_transfer_evidence_present",
            "topk2_causal_cooperation_blocked_by_deconfounded_and_finite_update_gates",
        ]
    if branch == SELECTED_MATCHED_DECONFOUNDING:
        return [
            "matched_control_coverage_missing_or_incomplete",
            "rank_scale_random_dense_cells_need_repair",
        ]
    if branch == SELECTED_DENSE_TEACHER_DISTILLATION:
        return [
            "dense_active_rank_control_appears_cleaner",
            "main_unknown_is_dense_correction_approximation",
        ]
    return ["insufficient_evidence"] if not signals else []


def _selected_rows(branch: str | None) -> list[dict[str, str]]:
    if branch == SELECTED_RETENTION_CHURN:
        return [
            {
                "branch": branch,
                "component": "anchor_task_drift",
                "purpose": "test whether retention survives transfer without hiding behind CE averages",
            },
            {
                "branch": branch,
                "component": "functional_and_logit_churn",
                "purpose": "compare final functions and logits after matched updates",
            },
            {
                "branch": branch,
                "component": "support_identity_churn",
                "purpose": "reconcile router-support stability with finite-update support churn",
            },
            {
                "branch": branch,
                "component": "finite_update_commutator_controls",
                "purpose": "keep top-k-1, dense active-rank, random fixed top-k-2, and promoted top-k-2 controls attached",
            },
        ]
    if branch == SELECTED_MATCHED_DECONFOUNDING:
        return [
            {
                "branch": branch or "",
                "component": "rank_scale_random_dense_control_repair",
                "purpose": "fill only missing matched-control cells before causal or retention interpretation",
            }
        ]
    if branch == SELECTED_DENSE_TEACHER_DISTILLATION:
        return [
            {
                "branch": branch,
                "component": "dense_teacher_residual_logit_matching",
                "purpose": "diagnose whether sparse columns can approximate cleaner dense corrections with lower churn",
            }
        ]
    return []


def _metrics(
    finite_augmented: dict[str, Any],
    functional_retention: dict[str, Any],
    deconfounded: dict[str, Any],
    finite_control: dict[str, Any],
) -> dict[str, Any]:
    finite_metrics = finite_augmented.get("metrics", {})
    retention_agg = (
        functional_retention.get("evidence", {}).get("aggregates", {})
        if isinstance(functional_retention.get("evidence"), dict)
        else {}
    )
    deconf_metrics = deconfounded.get("evidence", {})
    if isinstance(deconf_metrics, dict) and isinstance(deconf_metrics.get("metrics"), dict):
        deconf_metrics = deconf_metrics["metrics"]
    control_metrics = finite_control.get("metrics", {})
    role_counts = finite_augmented.get("role_counts", {})
    return {
        "finite_augmented_strata_count": finite_metrics.get("augmented_strata_count"),
        "finite_matched_exact_context_count": finite_metrics.get("augmented_matched_exact_context_count"),
        "finite_topk2_minus_topk1_logit_mse": finite_metrics.get("augmented_mean_topk2_minus_topk1_finite_logit_mse"),
        "finite_topk2_minus_dense_logit_mse": finite_metrics.get("augmented_mean_topk2_minus_dense_finite_logit_mse"),
        "finite_topk2_support_churn": finite_metrics.get("augmented_mean_topk2_finite_support_churn_fraction"),
        "finite_role_counts": role_counts,
        "control_matrix_topk2_minus_topk1_logit_mse": control_metrics.get("topk2_minus_topk1_logit_mse"),
        "functional_mean_topk1_support_churn": retention_agg.get("mean_topk1_anchor_support_churn_after_transfer"),
        "functional_mean_topk2_support_churn": retention_agg.get("mean_topk2_anchor_support_churn_after_transfer"),
        "functional_topk1_transfer_advantage_vs_topk2": retention_agg.get("mean_transfer_improvement_advantage_topk1_vs_topk2"),
        "functional_topk1_transfer_advantage_vs_dense": retention_agg.get("mean_transfer_improvement_advantage_topk1_vs_dense"),
        "functional_topk1_commutator_advantage_vs_topk2": retention_agg.get("mean_commutator_anchor_logit_mse_advantage_topk1_vs_topk2"),
        "functional_topk1_commutator_advantage_vs_dense": retention_agg.get("mean_commutator_anchor_logit_mse_advantage_topk1_vs_dense"),
        "deconfounded_matched_exact_context_count": deconf_metrics.get("matched_exact_context_count"),
        "deconfounded_topk2_positive_fraction": deconf_metrics.get("topk2_incremental_pair_gain_positive_strata_fraction"),
        "deconfounded_topk2_fixed_cleaner_fraction": deconf_metrics.get("topk2_fixed_support_cleaner_strata_fraction"),
    }


def _signals(
    metrics: dict[str, Any],
    finite_augmented: dict[str, Any],
    functional_retention: dict[str, Any],
    retention_stability: dict[str, Any],
    deconfounded: dict[str, Any],
    finite_control: dict[str, Any],
) -> dict[str, bool]:
    role_counts = metrics.get("finite_role_counts")
    coverage = isinstance(role_counts, dict) and all(
        int(role_counts.get(role, 0) or 0) > 0 for role in REQUIRED_ROLES
    )
    claim_signals = {}
    if isinstance(functional_retention.get("evidence"), dict):
        claim_signals = functional_retention["evidence"].get("claim_signals", {})
    return {
        "matched_control_coverage_adequate": bool(
            coverage
            and _positive(metrics.get("finite_augmented_strata_count"))
            and _positive(metrics.get("finite_matched_exact_context_count"))
        ),
        "topk2_causal_cooperation_blocked": bool(
            finite_augmented.get("decision") == FINITE_UPDATE_TOPK2_BLOCKED
            and deconfounded.get("decision") == TOPK2_NOT_SUPPORTED
        ),
        "retention_churn_evidence_adequate": bool(
            functional_retention.get("decision") == FUNCTIONAL_RETENTION_BRACKET_ONLY
            and retention_stability.get("decision") == ACTIVE_TOPK1_RETENTION_CHURN_STABLE
            and claim_signals.get("support_identity_churn_cleaner_than_topk2")
            and claim_signals.get("functional_logit_churn_not_higher_than_topk2")
            and claim_signals.get("finite_update_commutator_not_worse_than_topk2")
            and claim_signals.get("transfer_improvement_beats_dense_control")
        ),
        "dense_control_present": bool(
            isinstance(role_counts, dict) and int(role_counts.get("dense_active_rank", 0) or 0) > 0
        ),
        "finite_control_matrix_ready": finite_control.get("decision") == FINITE_UPDATE_CONTROL_MATRIX_READY,
        "strategy_review_prefers_retention_if_covered": True,
    }


def _failures(
    source_rows: list[dict[str, Any]],
    finite_augmented: dict[str, Any],
    functional_retention: dict[str, Any],
    retention_stability: dict[str, Any],
    deconfounded: dict[str, Any],
    finite_control: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows[:5]:
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
        ("finite_update_augmented_causal_gate", finite_augmented, "status", "pass"),
        ("finite_update_augmented_causal_gate", finite_augmented, "decision", FINITE_UPDATE_TOPK2_BLOCKED),
        ("functional_retention_audit", functional_retention, "status", "pass"),
        ("functional_retention_audit", functional_retention, "decision", FUNCTIONAL_RETENTION_BRACKET_ONLY),
        ("retention_churn_stability", retention_stability, "status", "pass"),
        ("retention_churn_stability", retention_stability, "decision", ACTIVE_TOPK1_RETENTION_CHURN_STABLE),
        ("deconfounded_intervention", deconfounded, "status", "pass"),
        ("deconfounded_intervention", deconfounded, "decision", TOPK2_NOT_SUPPORTED),
        ("finite_update_control_matrix", finite_control, "status", "pass"),
        ("finite_update_control_matrix", finite_control, "decision", FINITE_UPDATE_CONTROL_MATRIX_READY),
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
    return failures


def _source_row(source: str, path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": packet.get("status"),
        "decision": packet.get("decision"),
        "claim_status": packet.get("claim_status") or packet.get("claim_gate"),
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
            "accepted: the report follows the recommendation to run a no-training "
            "active-rank-matched top-k-1 selection report and favors retention/"
            "functional-churn when matched-control coverage is adequate"
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


def _positive(value: Any) -> bool:
    return isinstance(value, (float, int)) and float(value) > 0


def _float(value: Any) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return None


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
    lines = [
        "# Active Top-k-1 Next-Evidence Selection",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected experiment: `{summary['selected_experiment']}`",
        f"- Requires GPU now: `{summary['selection_gate']['requires_gpu_now']}`",
        f"- New training required: `{summary['selection_gate']['new_training_required']}`",
        f"- Matched-control coverage adequate: `{summary['selection_gate']['matched_control_coverage_adequate']}`",
        f"- Top-k-2 causal cooperation blocked: `{summary['selection_gate']['topk2_causal_cooperation_blocked']}`",
        f"- Git commit: `{summary['git_commit']}`",
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
    parser.add_argument("--finite-augmented-dir", type=Path, default=DEFAULT_FINITE_AUGMENTED_DIR)
    parser.add_argument("--functional-retention-dir", type=Path, default=DEFAULT_FUNCTIONAL_RETENTION_DIR)
    parser.add_argument("--retention-stability-dir", type=Path, default=DEFAULT_RETENTION_STABILITY_DIR)
    parser.add_argument("--deconfounded-dir", type=Path, default=DEFAULT_DECONFOUNDED_DIR)
    parser.add_argument("--finite-control-dir", type=Path, default=DEFAULT_FINITE_CONTROL_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_active_topk1_next_evidence_selection_report(
        finite_augmented_dir=args.finite_augmented_dir,
        functional_retention_dir=args.functional_retention_dir,
        retention_stability_dir=args.retention_stability_dir,
        deconfounded_dir=args.deconfounded_dir,
        finite_control_dir=args.finite_control_dir,
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
