"""RunPod closeout report for active top-k-1 post-decomposition validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.active_topk1_context_conditioned_singleton_interference_audit import (
    CONTEXT_GATE_REDUCES_OFFCONTEXT_INTERFERENCE,
)
from relaleap.experiments.active_topk1_post_decomposition_decision_report import (
    BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED,
    COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS,
    POST_DECOMPOSITION_RUNPOD_VALIDATION_RECOMMENDED,
)


DEFAULT_LOCAL_INTERFERENCE_DIR = Path(
    "results/audits/token_larger_active_topk1_context_conditioned_singleton_interference"
)
DEFAULT_RUNPOD_INTERFERENCE_DIR = Path(
    "results/runpod_fetch/audits/runpod_token_larger_active_topk1_context_conditioned_singleton_interference"
)
DEFAULT_LOCAL_DECISION_DIR = Path(
    "results/reports/token_larger_active_topk1_post_decomposition_decision"
)
DEFAULT_LOCAL_CHECKED_RUNPOD_DECISION_DIR = Path(
    "results/reports/local_checked_runpod_active_topk1_post_decomposition_decision"
)
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_active_topk1_runpod_post_decomposition_closeout"
)

RUNPOD_POST_DECOMPOSITION_VALIDATED = "runpod_post_decomposition_validated"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

METRIC_FIELDS = (
    "context_count",
    "selected_context_count",
    "offcontext_context_count",
    "random_control_context_count",
    "exhaustive_control_context_count",
    "source_row_count",
    "own_context_singleton_gain_mean",
    "off_context_singleton_gain_mean",
    "context_gated_net_gain_holdout_mean",
    "context_gate_gain_minus_ungated_holdout_mean",
    "topk2_reference_gain_mean",
    "random_singleton_gain_mean",
    "exhaustive_singleton_gain_mean",
)
SIGNAL_FIELDS = (
    "own_context_singleton_gain_positive",
    "offcontext_singleton_interference_present",
    "context_gate_holdout_net_gain_positive",
    "context_gate_improves_over_ungated_holdout",
    "matched_topk2_reference_present",
    "random_control_present",
    "exhaustive_control_present",
)
REQUIRED_INTERFERENCE_ARTIFACTS = (
    "summary.json",
    "singleton_interference_by_context.csv",
    "singleton_interference_by_stratum.csv",
    "context_gate_holdout.csv",
    "notes.md",
)


def run_active_topk1_runpod_post_decomposition_closeout_report(
    *,
    local_interference_dir: Path = DEFAULT_LOCAL_INTERFERENCE_DIR,
    runpod_interference_dir: Path = DEFAULT_RUNPOD_INTERFERENCE_DIR,
    local_decision_dir: Path = DEFAULT_LOCAL_DECISION_DIR,
    local_checked_runpod_decision_dir: Path = DEFAULT_LOCAL_CHECKED_RUNPOD_DECISION_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    prerequisite_sync_note: str = (
        "RunPod prerequisite source packets were synced to /workspace/relaleap/results "
        "before rerunning the bounded validation command because the fresh pod clone "
        "does not contain ignored local result artifacts."
    ),
) -> dict[str, Any]:
    """Compare local and fetched RunPod post-decomposition packets and close the loop."""

    start = time.time()
    local_interference = _read_json_object(local_interference_dir / "summary.json")
    runpod_interference = _read_json_object(runpod_interference_dir / "summary.json")
    local_decision = _read_json_object(local_decision_dir / "summary.json")
    checked_decision = _read_json_object(
        local_checked_runpod_decision_dir / "summary.json"
    )

    artifact_rows = [
        _artifact_row("local_interference", local_interference_dir / name)
        for name in REQUIRED_INTERFERENCE_ARTIFACTS
    ] + [
        _artifact_row("runpod_interference", runpod_interference_dir / name)
        for name in REQUIRED_INTERFERENCE_ARTIFACTS
    ] + [
        _artifact_row("local_decision", local_decision_dir / name)
        for name in ("summary.json", "decision_sources.csv", "notes.md")
    ] + [
        _artifact_row(
            "local_checked_runpod_decision",
            local_checked_runpod_decision_dir / name,
        )
        for name in ("summary.json", "decision_sources.csv", "notes.md")
    ]
    metric_rows = _metric_rows(local_interference, runpod_interference)
    signal_rows = _signal_rows(local_interference, runpod_interference)
    failures = _failures(
        artifact_rows=artifact_rows,
        local_interference=local_interference,
        runpod_interference=runpod_interference,
        local_decision=local_decision,
        checked_decision=checked_decision,
        metric_rows=metric_rows,
        signal_rows=signal_rows,
    )

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        claim_status = INSUFFICIENT_EVIDENCE
        selected_next_step = (
            "repair_missing_or_mismatched_runpod_post_decomposition_closeout_sources"
        )
        rationale = (
            "The RunPod post-decomposition validation cannot be closed out because "
            "one or more local/fetched artifacts, decision fields, metrics, or "
            "signals are missing or mismatched."
        )
    else:
        status = "pass"
        decision = RUNPOD_POST_DECOMPOSITION_VALIDATED
        claim_status = COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS
        selected_next_step = (
            "use_the_validated_column_plus_context_gate_packet_as_source_evidence"
        )
        rationale = (
            "The local and fetched RunPod context-conditioned singleton interference "
            "packets agree on the required decisions, controls, signals, and key "
            "metrics. The local fetched-artifact decision check also passes. This "
            "closes the backend validation step while preserving the policy that a "
            "broad reusable singleton claim remains excluded."
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
        "source_dirs": {
            "local_interference_dir": str(local_interference_dir),
            "runpod_interference_dir": str(runpod_interference_dir),
            "local_decision_dir": str(local_decision_dir),
            "local_checked_runpod_decision_dir": str(
                local_checked_runpod_decision_dir
            ),
        },
        "prerequisite_sync_provenance": {
            "synced_ignored_result_packets": True,
            "note": prerequisite_sync_note,
        },
        "metric_comparison": metric_rows,
        "signal_comparison": signal_rows,
        "artifact_manifest": artifact_rows,
        "failures": failures,
        "rationale": rationale,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "metric_comparison_csv": str(out_dir / "metric_comparison.csv"),
            "signal_comparison_csv": str(out_dir / "signal_comparison.csv"),
            "artifact_manifest_csv": str(out_dir / "artifact_manifest.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        out_dir / "metric_comparison.csv",
        ["field", "local", "runpod", "match"],
        metric_rows,
    )
    _write_csv(
        out_dir / "signal_comparison.csv",
        ["field", "local", "runpod", "match"],
        signal_rows,
    )
    _write_csv(
        out_dir / "artifact_manifest.csv",
        ["source", "path", "present", "sha256", "size_bytes"],
        artifact_rows,
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _metric_rows(
    local_interference: dict[str, Any],
    runpod_interference: dict[str, Any],
) -> list[dict[str, Any]]:
    local_metrics = local_interference.get("evidence", {}).get("metrics", {})
    runpod_metrics = runpod_interference.get("evidence", {}).get("metrics", {})
    return [
        {
            "field": field,
            "local": local_metrics.get(field),
            "runpod": runpod_metrics.get(field),
            "match": _values_match(local_metrics.get(field), runpod_metrics.get(field)),
        }
        for field in METRIC_FIELDS
    ]


def _signal_rows(
    local_interference: dict[str, Any],
    runpod_interference: dict[str, Any],
) -> list[dict[str, Any]]:
    local_signals = local_interference.get("evidence", {}).get("signals", {})
    runpod_signals = runpod_interference.get("evidence", {}).get("signals", {})
    return [
        {
            "field": field,
            "local": local_signals.get(field),
            "runpod": runpod_signals.get(field),
            "match": local_signals.get(field) == runpod_signals.get(field),
        }
        for field in SIGNAL_FIELDS
    ]


def _failures(
    *,
    artifact_rows: list[dict[str, Any]],
    local_interference: dict[str, Any],
    runpod_interference: dict[str, Any],
    local_decision: dict[str, Any],
    checked_decision: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    signal_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in artifact_rows:
        if not row["present"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": Path(str(row["path"])).name,
                    "expected": "file exists",
                    "actual": "missing",
                    "path": row["path"],
                }
            )
    for source, packet in (
        ("local_interference", local_interference),
        ("runpod_interference", runpod_interference),
    ):
        _expect(
            failures,
            source,
            packet,
            "status",
            "pass",
        )
        _expect(
            failures,
            source,
            packet,
            "decision",
            CONTEXT_GATE_REDUCES_OFFCONTEXT_INTERFERENCE,
        )
        _expect(
            failures,
            source,
            packet,
            "claim_policy",
            BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED,
        )
    for source, packet in (
        ("local_decision", local_decision),
        ("local_checked_runpod_decision", checked_decision),
    ):
        _expect(failures, source, packet, "status", "pass")
        _expect(
            failures,
            source,
            packet,
            "decision",
            POST_DECOMPOSITION_RUNPOD_VALIDATION_RECOMMENDED,
        )
        _expect(
            failures,
            source,
            packet,
            "claim_status",
            COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS,
        )
        _expect(
            failures,
            source,
            packet,
            "claim_policy",
            BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED,
        )
    for row in metric_rows + signal_rows:
        if not row["match"]:
            failures.append(
                {
                    "source": "local_vs_runpod",
                    "field": row["field"],
                    "expected": "matching local and fetched RunPod values",
                    "actual": {"local": row["local"], "runpod": row["runpod"]},
                }
            )
    return failures


def _expect(
    failures: list[dict[str, Any]],
    source: str,
    packet: dict[str, Any],
    field: str,
    expected: Any,
) -> None:
    if packet.get(field) != expected:
        failures.append(
            {
                "source": source,
                "field": field,
                "expected": expected,
                "actual": packet.get(field),
            }
        )


def _artifact_row(source: str, path: Path) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def _values_match(left: Any, right: Any) -> bool:
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return abs(float(left) - float(right)) <= 1e-12
    return left == right


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "# Active Top-k-1 RunPod Post-Decomposition Closeout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Claim policy: `{summary['claim_policy']}`",
        f"- Git commit: `{summary['git_commit']}`",
        "",
        "## Rationale",
        "",
        summary["rationale"],
        "",
        "## Prerequisite Sync Provenance",
        "",
        f"- Synced ignored result packets: `{summary['prerequisite_sync_provenance']['synced_ignored_result_packets']}`",
        f"- Note: {summary['prerequisite_sync_provenance']['note']}",
        "",
        "## Metric Agreement",
        "",
    ]
    for row in summary["metric_comparison"]:
        lines.append(
            f"- `{row['field']}`: local `{row['local']}`, RunPod `{row['runpod']}`, match `{row['match']}`"
        )
    lines.extend(
        [
            "",
            "## Signal Agreement",
            "",
        ]
    )
    for row in summary["signal_comparison"]:
        lines.append(
            f"- `{row['field']}`: local `{row['local']}`, RunPod `{row['runpod']}`, match `{row['match']}`"
        )
    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in summary["failures"]:
            lines.append(f"- `{failure}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-interference-dir", type=Path, default=DEFAULT_LOCAL_INTERFERENCE_DIR
    )
    parser.add_argument(
        "--runpod-interference-dir", type=Path, default=DEFAULT_RUNPOD_INTERFERENCE_DIR
    )
    parser.add_argument("--local-decision-dir", type=Path, default=DEFAULT_LOCAL_DECISION_DIR)
    parser.add_argument(
        "--local-checked-runpod-decision-dir",
        type=Path,
        default=DEFAULT_LOCAL_CHECKED_RUNPOD_DECISION_DIR,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prerequisite-sync-note", default=None)
    args = parser.parse_args(argv)
    kwargs: dict[str, Any] = {
        "local_interference_dir": args.local_interference_dir,
        "runpod_interference_dir": args.runpod_interference_dir,
        "local_decision_dir": args.local_decision_dir,
        "local_checked_runpod_decision_dir": args.local_checked_runpod_decision_dir,
        "out_dir": args.out,
    }
    if args.prerequisite_sync_note is not None:
        kwargs["prerequisite_sync_note"] = args.prerequisite_sync_note
    summary = run_active_topk1_runpod_post_decomposition_closeout_report(**kwargs)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "claim_status": summary["claim_status"],
                "claim_policy": summary["claim_policy"],
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
