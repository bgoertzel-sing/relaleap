"""Sequence-holdout coverage report for promoted contextual top-k-2 support selection."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_SUPPORT_SELECTION_DIR = Path(
    "results/reports/token_larger_promoted_topk2_support_selection_quality_audit"
)
DEFAULT_EXHAUSTIVE_AUDIT_DIR = Path(
    "results/audits/token_larger_support_wide_promoted_default_exhaustive_support"
)
DEFAULT_CAUSAL_ADEQUACY_DIR = Path(
    "results/reports/token_larger_promoted_topk2_causal_adequacy_matrix"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_promoted_topk2_sequence_holdout_coverage"
)

SEQUENCE_HOLDOUT_COVERAGE_READY = "sequence_holdout_coverage_ready"
SEQUENCE_HOLDOUT_EXTENSION_REQUIRED = "sequence_holdout_extension_required"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def run_promoted_topk2_sequence_holdout_coverage_report(
    *,
    support_selection_dir: Path = DEFAULT_SUPPORT_SELECTION_DIR,
    exhaustive_audit_dir: Path = DEFAULT_EXHAUSTIVE_AUDIT_DIR,
    causal_adequacy_dir: Path = DEFAULT_CAUSAL_ADEQUACY_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Check whether contextual support-selection evidence includes sequence holdout."""

    start = time.time()
    support_selection_path = support_selection_dir / "summary.json"
    exhaustive_path = exhaustive_audit_dir / "summary.json"
    causal_path = causal_adequacy_dir / "summary.json"
    support_selection = _read_json_object(support_selection_path)
    exhaustive = _read_json_object(exhaustive_path)
    causal = _read_json_object(causal_path)
    strategy_review = _strategy_review(strategy_review_path)

    audit = exhaustive.get("audit", {}) if isinstance(exhaustive.get("audit"), dict) else {}
    split_rows = _split_rows(audit)
    metrics = _metrics(audit, support_selection, causal, split_rows)
    source_rows = [
        _source_row("support_selection_quality", support_selection_path, support_selection),
        _source_row("exhaustive_support_audit", exhaustive_path, exhaustive),
        _source_row("causal_adequacy_matrix", causal_path, causal),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy_review["present"],
            "status": "present" if strategy_review["present"] else "missing_optional",
            "decision": strategy_review["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy_review['strategic_change_level']}; "
                f"notify_ben={strategy_review['notify_ben']}"
            ),
        },
    ]
    failures = _failures(source_rows, metrics)
    sequence_holdout_present = metrics["sequence_level_holdout_present"]

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        claim_status = "support_selection_holdout_coverage_uninterpretable"
        next_step = "repair_missing_sequence_holdout_coverage_sources"
        rationale = (
            "The sequence-holdout coverage report cannot be interpreted because "
            "a required support-selection or causal-adequacy source artifact is "
            "missing or inconsistent."
        )
    elif sequence_holdout_present:
        status = "pass"
        decision = SEQUENCE_HOLDOUT_COVERAGE_READY
        claim_status = "sequence_level_support_prediction_coverage_present"
        next_step = (
            "consume the sequence-level holdout rows in the next promoted top-k-2 "
            "support-selection quality synthesis"
        )
        rationale = (
            "Existing support-selection artifacts include sequence-level holdout "
            "coverage, so the deployability gap raised by the strategy review is "
            "closed at the artifact-coverage level."
        )
    else:
        status = "pass"
        decision = SEQUENCE_HOLDOUT_EXTENSION_REQUIRED
        claim_status = "sequence_level_support_prediction_not_yet_tested"
        next_step = (
            "extend relaleap.experiments.support_audit with a sequence-level "
            "holdout split for contextual support prediction and rerun the "
            "promoted token-larger exhaustive support audit"
        )
        rationale = (
            "The promoted top-k-2 support-selection packet uses even flattened "
            "token positions for training and odd flattened token positions for "
            "holdout. That is useful but does not satisfy the current strategy "
            "review's stricter sequence-level held-out evaluation recommendation. "
            "The causal-adequacy matrix can stand as the current matched-control "
            "claim blocker, but deployable contextual support-selection claims "
            "should remain limited until this artifact extension exists."
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "out_dir": str(out_dir),
        "source_rows": source_rows,
        "split_rows": split_rows,
        "metrics": metrics,
        "signals": {
            "position_holdout_present": metrics["position_holdout_present"],
            "sequence_level_holdout_present": sequence_holdout_present,
            "strategy_review_requested_sequence_holdout": strategy_review[
                "sequence_holdout_recommended"
            ],
            "deployable_support_selection_claim_blocked_by_split_coverage": (
                status == "pass" and not sequence_holdout_present
            ),
        },
        "strategy_review": strategy_review,
        "failures": failures,
        "rationale": rationale,
        "next_step": next_step,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "split_rows_csv": str(out_dir / "split_rows.csv"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_rows.csv", source_rows)
    _write_csv(out_dir / "split_rows.csv", split_rows)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _split_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in (
        "router_oracle_target_diagnostic",
        "router_oracle_target_nonlinear_diagnostic",
        "router_oracle_target_contextual_diagnostic",
        "contextual_router_support_intervention",
        "contextual_router_support_head",
    ):
        packet = audit.get(name)
        if not isinstance(packet, dict):
            continue
        train_split = _string_or_empty(packet.get("train_split"))
        holdout_split = _string_or_empty(packet.get("holdout_split"))
        split_text = f"{train_split} {holdout_split}".lower()
        rows.append(
            {
                "artifact": name,
                "train_split": train_split,
                "holdout_split": holdout_split,
                "position_holdout": _is_position_split(split_text),
                "sequence_level_holdout": _is_sequence_split(split_text),
                "holdout_oracle_gap_recovery_fraction": _nested_float(
                    packet, "holdout", "oracle_gap_recovery_fraction"
                ),
            }
        )
    return rows


def _metrics(
    audit: dict[str, Any],
    support_selection: dict[str, Any],
    causal: dict[str, Any],
    split_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    support_metrics = support_selection.get("metrics", {})
    causal_metrics = causal.get("metrics", {})
    return {
        "config_path": audit.get("config_path") or support_metrics.get("config_path"),
        "dataset": audit.get("dataset") or support_metrics.get("dataset"),
        "support_router": audit.get("support_router") or support_metrics.get("support_router"),
        "split_row_count": len(split_rows),
        "position_holdout_present": any(row["position_holdout"] for row in split_rows),
        "sequence_level_holdout_present": any(
            row["sequence_level_holdout"] for row in split_rows
        ),
        "contextual_support_head_holdout_gap_recovery": support_metrics.get(
            "contextual_support_head_holdout_gap_recovery"
        ),
        "contextual_oracle_target_holdout_gap_recovery": support_metrics.get(
            "contextual_oracle_target_holdout_gap_recovery"
        ),
        "oracle_support_regret": support_metrics.get("oracle_support_regret"),
        "causal_adequacy_decision": causal.get("decision"),
        "topk2_support_churn": causal_metrics.get("topk2_support_churn"),
        "topk2_to_topk1_finite_update_logit_mse_ratio": causal_metrics.get(
            "topk2_to_topk1_finite_update_logit_mse_ratio"
        ),
    }


def _failures(
    source_rows: list[dict[str, Any]], metrics: dict[str, Any]
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    required = {
        "support_selection_quality": ("pass", "promoted_topk2_support_selection_quality_established"),
        "exhaustive_support_audit": ("ok", None),
        "causal_adequacy_matrix": ("pass", "predictive_default_causal_adequacy_not_established"),
    }
    for row in source_rows:
        expected = required.get(row["source"])
        if expected is None:
            continue
        expected_status, expected_decision = expected
        if not row["present"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "source_artifact",
                    "expected": "file exists",
                    "actual": "missing",
                    "path": row["path"],
                }
            )
            continue
        if row["status"] != expected_status:
            failures.append(
                {
                    "source": row["source"],
                    "field": "status",
                    "expected": expected_status,
                    "actual": row["status"],
                }
            )
        if expected_decision is not None and row["decision"] != expected_decision:
            failures.append(
                {
                    "source": row["source"],
                    "field": "decision",
                    "expected": expected_decision,
                    "actual": row["decision"],
                }
            )
    if metrics["split_row_count"] <= 0:
        failures.append(
            {
                "source": "exhaustive_support_audit",
                "field": "split_rows",
                "expected": "at least one support-prediction split row",
                "actual": metrics["split_row_count"],
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
        "claim_status": packet.get("claim_status"),
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "strategic_change_level": None,
            "notify_ben": False,
            "ben_notification_required": False,
            "recommended_next_action": None,
            "sequence_holdout_recommended": False,
            "incorporation": "no external strategy review was present",
        }
    text = path.read_text(encoding="utf-8")
    strategic_change_level = _header_value(text, "strategic_change_level")
    notify_ben = (_header_value(text, "notify_ben") or "").lower() == "true"
    sequence_holdout = "sequence-level" in text.lower() and "holdout" in text.lower()
    return {
        "path": str(path),
        "present": True,
        "strategic_change_level": strategic_change_level,
        "notify_ben": notify_ben,
        "ben_notification_required": notify_ben or strategic_change_level == "major",
        "recommended_next_action": _header_value(text, "recommended_next_action"),
        "sequence_holdout_recommended": sequence_holdout,
        "incorporation": (
            "accepted the sequence-level holdout recommendation as a sensible "
            "deployability-coverage requirement; no major direction shift"
            if sequence_holdout
            else "no sequence-level holdout recommendation found in the review"
        ),
    }


def _header_value(text: str, key: str) -> str | None:
    prefix = f"{key}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    signals = summary["signals"]
    lines = [
        "# Promoted Top-k-2 Sequence-Holdout Coverage",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Position holdout present: `{signals['position_holdout_present']}`",
        f"- Sequence-level holdout present: `{signals['sequence_level_holdout_present']}`",
        "- Strategy review requested sequence holdout: "
        f"`{signals['strategy_review_requested_sequence_holdout']}`",
        "- Contextual support-head holdout gap recovery: "
        f"`{metrics['contextual_support_head_holdout_gap_recovery']}`",
        f"- Oracle support regret: `{metrics['oracle_support_regret']}`",
        "- Causal-adequacy decision: "
        f"`{metrics['causal_adequacy_decision']}`",
        "",
        summary["rationale"],
        "",
        f"Next step: {summary['next_step']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _nested_float(packet: dict[str, Any], *path: str) -> float | None:
    value: Any = packet
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_or_empty(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _is_position_split(text: str) -> bool:
    return "position" in text or "flattened token" in text or "even" in text or "odd" in text


def _is_sequence_split(text: str) -> bool:
    return "sequence" in text or "corpus" in text or "document" in text


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-selection-dir", type=Path, default=DEFAULT_SUPPORT_SELECTION_DIR)
    parser.add_argument("--exhaustive-audit-dir", type=Path, default=DEFAULT_EXHAUSTIVE_AUDIT_DIR)
    parser.add_argument("--causal-adequacy-dir", type=Path, default=DEFAULT_CAUSAL_ADEQUACY_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_promoted_topk2_sequence_holdout_coverage_report(
        support_selection_dir=args.support_selection_dir,
        exhaustive_audit_dir=args.exhaustive_audit_dir,
        causal_adequacy_dir=args.causal_adequacy_dir,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, indent=2))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
