"""Close out active top-k-1 after deployable gate calibration."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.active_topk1_context_gate_suppression_calibration_audit import (
    CONTEXT_GATE_SUPPRESSION_CALIBRATION_FAILED,
    CONTEXT_GATE_SUPPRESSION_CALIBRATION_PASSED,
)
from relaleap.experiments.active_topk1_post_decomposition_decision_report import (
    BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED,
    COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS,
)
from relaleap.experiments.active_topk1_runpod_post_decomposition_closeout_report import (
    RUNPOD_POST_DECOMPOSITION_VALIDATED,
)


DEFAULT_RUNPOD_CLOSEOUT_DIR = Path(
    "results/reports/token_larger_active_topk1_runpod_post_decomposition_closeout"
)
DEFAULT_SUPPRESSION_CALIBRATION_DIR = Path(
    "results/audits/token_larger_active_topk1_context_gate_suppression_calibration"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_active_topk1_post_suppression_calibration_closeout"
)

TOPK1_DIAGNOSTIC_ONLY_RETURN_TO_TOPK2 = (
    "topk1_diagnostic_only_return_to_contextual_topk2_support_routing"
)
TOPK1_DEPLOYABLE_GATE_PROMOTION_CANDIDATE = (
    "topk1_deployable_context_gate_promotion_candidate"
)
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def run_active_topk1_post_suppression_calibration_closeout_report(
    *,
    runpod_closeout_dir: Path = DEFAULT_RUNPOD_CLOSEOUT_DIR,
    suppression_calibration_dir: Path = DEFAULT_SUPPRESSION_CALIBRATION_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Select the next branch after validated decomposition plus gate calibration."""

    start = time.time()
    runpod_closeout = _read_json_object(runpod_closeout_dir / "summary.json")
    calibration = _read_json_object(suppression_calibration_dir / "summary.json")
    strategy_review = _strategy_review(strategy_review_path)
    source_rows = _source_rows(
        runpod_closeout_dir=runpod_closeout_dir,
        suppression_calibration_dir=suppression_calibration_dir,
        runpod_closeout=runpod_closeout,
        calibration=calibration,
        strategy_review=strategy_review,
    )
    metrics = calibration.get("evidence", {}).get("metrics", {})
    signals = calibration.get("evidence", {}).get("signals", {})
    failures = _failures(
        source_rows=source_rows,
        runpod_closeout=runpod_closeout,
        calibration=calibration,
        metrics=metrics,
        signals=signals,
    )

    deployable_passed = (
        calibration.get("decision") == CONTEXT_GATE_SUPPRESSION_CALIBRATION_PASSED
        and signals.get("deployable_gate_passes_pre_registered_criteria") is True
    )
    deployable_failed = (
        calibration.get("decision") == CONTEXT_GATE_SUPPRESSION_CALIBRATION_FAILED
        and signals.get("deployable_gate_passes_pre_registered_criteria") is False
    )

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        claim_status = INSUFFICIENT_EVIDENCE
        selected_next_step = "repair_missing_post_suppression_calibration_sources"
        rationale = (
            "The post-suppression closeout cannot select a branch because the "
            "validated post-decomposition packet, suppression-calibration packet, "
            "or required calibration metrics/signals are missing or inconsistent."
        )
    elif deployable_passed:
        status = "pass"
        decision = TOPK1_DEPLOYABLE_GATE_PROMOTION_CANDIDATE
        claim_status = COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS
        selected_next_step = (
            "implement_trainable_context_gate_against_topk2_topk1_random_dense_controls"
        )
        rationale = (
            "The RunPod-validated decomposition and local deployable calibration "
            "both support a column-plus-context-gate hypothesis. Because the "
            "deployable gate passed the pre-registered retained-gain, random-control, "
            "and off-context-harm criteria, the next coherent step is a small "
            "trainable context-gate implementation against the existing controls."
        )
    elif deployable_failed:
        status = "pass"
        decision = TOPK1_DIAGNOSTIC_ONLY_RETURN_TO_TOPK2
        claim_status = COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS
        selected_next_step = "return_main_architecture_loop_to_contextual_topk2_support_routing"
        rationale = (
            "The RunPod-validated decomposition supports context-gated singleton "
            "efficacy with off-context interference, but the deployable stratum gate "
            "does not beat ungated reuse, does not clear the coverage-matched random "
            "advantage threshold, and does not suppress enough off-context harm. "
            "Top-k-1 therefore remains diagnostic-only; broad reusable singleton "
            "claims stay excluded, and the main architecture loop should return to "
            "contextual top-k-2 support routing."
        )
    else:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        claim_status = INSUFFICIENT_EVIDENCE
        selected_next_step = "repair_unrecognized_suppression_calibration_decision"
        rationale = (
            "The suppression calibration packet is present but has an unrecognized "
            "decision/signals combination, so this closeout refuses to infer a branch."
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "claim_policy": BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED,
        "selected_next_step": selected_next_step,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "source_rows": source_rows,
        "evidence": {
            "metrics": metrics,
            "signals": signals,
            "calibration_decision": calibration.get("decision"),
            "runpod_closeout_decision": runpod_closeout.get("decision"),
            "interpretation": {
                "validated_context_gated_packet": (
                    runpod_closeout.get("decision") == RUNPOD_POST_DECOMPOSITION_VALIDATED
                ),
                "deployable_calibration_passed": deployable_passed,
                "deployable_calibration_failed": deployable_failed,
            },
        },
        "strategy_review": strategy_review,
        "failures": failures,
        "rationale": rationale,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
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
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _source_rows(
    *,
    runpod_closeout_dir: Path,
    suppression_calibration_dir: Path,
    runpod_closeout: dict[str, Any],
    calibration: dict[str, Any],
    strategy_review: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "source": "runpod_post_decomposition_closeout",
            "path": str(runpod_closeout_dir / "summary.json"),
            "present": (runpod_closeout_dir / "summary.json").is_file(),
            "status": runpod_closeout.get("status"),
            "decision": runpod_closeout.get("decision"),
            "claim_status": runpod_closeout.get("claim_status"),
        },
        {
            "source": "context_gate_suppression_calibration",
            "path": str(suppression_calibration_dir / "summary.json"),
            "present": (suppression_calibration_dir / "summary.json").is_file(),
            "status": calibration.get("status"),
            "decision": calibration.get("decision"),
            "claim_status": calibration.get("claim_status"),
        },
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


def _failures(
    *,
    source_rows: list[dict[str, Any]],
    runpod_closeout: dict[str, Any],
    calibration: dict[str, Any],
    metrics: dict[str, Any],
    signals: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows[:2]:
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
            runpod_closeout,
            "status",
            "pass",
        ),
        (
            "runpod_post_decomposition_closeout",
            runpod_closeout,
            "decision",
            RUNPOD_POST_DECOMPOSITION_VALIDATED,
        ),
        ("context_gate_suppression_calibration", calibration, "status", "pass"),
        (
            "context_gate_suppression_calibration",
            calibration,
            "claim_status",
            COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS,
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
    for field in (
        "deployable_holdout_net_gain",
        "deployable_gain_minus_ungated",
        "deployable_gain_minus_coverage_matched_random",
        "deployable_retained_gain_fraction",
        "deployable_offcontext_harm_suppression_fraction",
    ):
        if not isinstance(metrics.get(field), (int, float)):
            failures.append(
                {
                    "source": "context_gate_suppression_calibration",
                    "field": field,
                    "expected": "numeric value",
                    "actual": metrics.get(field),
                }
            )
    for field in (
        "deployable_gate_passes_pre_registered_criteria",
        "deployable_holdout_net_gain_positive",
        "deployable_retains_enough_own_context_gain",
        "deployable_beats_coverage_matched_random",
        "deployable_suppresses_offcontext_harm",
        "topk2_reference_present",
    ):
        if field not in signals:
            failures.append(
                {
                    "source": "context_gate_suppression_calibration",
                    "field": field,
                    "expected": "boolean signal",
                    "actual": None,
                }
            )
    return failures


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "strategic_change_level": None,
            "notify_ben": False,
            "ben_notification_required": False,
            "recommended_next_action": None,
            "incorporation": "missing_optional: no external strategy review found",
        }
    fields: dict[str, Any] = {
        "path": str(path),
        "present": True,
        "strategic_change_level": None,
        "notify_ben": False,
        "recommended_next_action": None,
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {"strategic_change_level", "recommended_next_action"}:
            fields[key] = value
        elif key == "notify_ben":
            fields[key] = value.lower() == "true"
    fields["ben_notification_required"] = (
        fields.get("notify_ben") is True
        or fields.get("strategic_change_level") == "major"
    )
    fields["incorporation"] = (
        "followed where applicable: this closeout consumes the completed retention/"
        "interference/calibration branch and records why top-k-1 promotion is "
        "deferred when the deployable gate fails"
    )
    return fields


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


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
    metrics = summary.get("evidence", {}).get("metrics", {})
    signals = summary.get("evidence", {}).get("signals", {})
    lines = [
        "# Active Top-k-1 Post-Suppression Calibration Closeout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Claim policy: `{summary['claim_policy']}`",
        f"- Selected next step: `{summary['selected_next_step']}`",
        f"- Deployable holdout net gain: `{metrics.get('deployable_holdout_net_gain')}`",
        f"- Deployable gain minus ungated: `{metrics.get('deployable_gain_minus_ungated')}`",
        "- Deployable gain minus coverage-matched random: "
        f"`{metrics.get('deployable_gain_minus_coverage_matched_random')}`",
        "- Deployable retained gain fraction: "
        f"`{metrics.get('deployable_retained_gain_fraction')}`",
        "- Deployable off-context harm suppression fraction: "
        f"`{metrics.get('deployable_offcontext_harm_suppression_fraction')}`",
        "- Pre-registered calibration passed: "
        f"`{signals.get('deployable_gate_passes_pre_registered_criteria')}`",
        "",
        "## Rationale",
        "",
        summary["rationale"],
        "",
    ]
    if summary["failures"]:
        lines.extend(["## Failures", ""])
        for failure in summary["failures"]:
            lines.append(f"- `{failure}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runpod-closeout-dir", type=Path, default=DEFAULT_RUNPOD_CLOSEOUT_DIR)
    parser.add_argument(
        "--suppression-calibration-dir",
        type=Path,
        default=DEFAULT_SUPPRESSION_CALIBRATION_DIR,
    )
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_active_topk1_post_suppression_calibration_closeout_report(
        runpod_closeout_dir=args.runpod_closeout_dir,
        suppression_calibration_dir=args.suppression_calibration_dir,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "claim_status": summary["claim_status"],
                "selected_next_step": summary["selected_next_step"],
                "failures": summary["failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
