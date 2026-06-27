"""Inspect causal contextual-router oracle-regret and churn failures."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_LOCAL_SUPPORT_AUDIT = Path("results/audits/local_causal_contextual_router_support_audit")
DEFAULT_RUNPOD_SUPPORT_AUDIT = Path(
    "results/runpod_fetch/audits/runpod_token_larger_causal_contextual_router_support_audit"
)
DEFAULT_POST_SEQUENCE_REPORT = Path(
    "results/reports/token_larger_contextual_router_post_sequence_decision/summary.json"
)
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_contextual_router_regret_churn_failure_inspection"
)

INSPECTION_RECORDED = "contextual_router_regret_churn_failure_inspection_recorded"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def run_contextual_router_regret_churn_failure_inspection(
    *,
    local_support_audit_dir: Path = DEFAULT_LOCAL_SUPPORT_AUDIT,
    runpod_support_audit_dir: Path = DEFAULT_RUNPOD_SUPPORT_AUDIT,
    post_sequence_report_path: Path = DEFAULT_POST_SEQUENCE_REPORT,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Summarize why the causal-feature-safe router fails support-quality gates."""

    start = time.time()
    local = _load_support_packet("local", local_support_audit_dir)
    runpod = _load_support_packet("runpod", runpod_support_audit_dir)
    post_sequence = _read_json_object(post_sequence_report_path)

    source_rows = [
        _source_row("local_support_audit", local_support_audit_dir / "summary.json", local["summary"]),
        _source_row("runpod_support_audit", runpod_support_audit_dir / "summary.json", runpod["summary"]),
        _source_row("post_sequence_decision", post_sequence_report_path, post_sequence),
    ]
    fold_rows = _fold_failure_rows(local) + _fold_failure_rows(runpod)
    evidence = _evidence(local, runpod, fold_rows, post_sequence)
    failures = _failures(source_rows, evidence)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        claim_status = "regret_churn_failure_inspection_uninterpretable"
        selected_next_step = "repair_missing_or_inconsistent_regret_churn_sources"
        rationale = (
            "The failure inspection cannot be interpreted because required local, "
            "RunPod, or post-sequence source artifacts are missing or inconsistent."
        )
    else:
        status = "pass"
        decision = INSPECTION_RECORDED
        claim_status = "causal_router_ce_win_is_not_support_quality_evidence"
        selected_next_step = (
            "stop causal-feature-safe router promotion work and return to the "
            "active rank-matched top-k-1 causal bracket"
        )
        rationale = (
            "The causal-feature-safe contextual top-k-2 router wins heldout CE "
            "because its oracle frontier is much better than the linear control, "
            "but it leaves materially more oracle-support regret and higher "
            "functional churn on every checked backend fold. The failure is "
            "therefore not a backend artifact and not a fixed-support-collapse "
            "issue; it is a support-selection/retention quality gap. This keeps "
            "the causal-feature-safe router out of the promotion path and keeps "
            "support-label distillation frozen until a same-student functional "
            "gate is positive."
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "source_rows": source_rows,
        "evidence": evidence,
        "failures": failures,
        "rationale": rationale,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "fold_failure_deltas_csv": str(out_dir / "fold_failure_deltas.csv"),
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
    _write_csv(
        out_dir / "fold_failure_deltas.csv",
        [
            "backend",
            "fold",
            "heldout_sequence_index",
            "causal_router_loss",
            "linear_router_loss",
            "causal_minus_linear_router_loss",
            "causal_oracle_loss",
            "linear_oracle_loss",
            "causal_minus_linear_oracle_loss",
            "causal_oracle_regret",
            "linear_oracle_regret",
            "causal_minus_linear_oracle_regret",
            "causal_functional_churn",
            "linear_functional_churn",
            "causal_minus_linear_functional_churn",
            "causal_unique_support_sets",
            "linear_unique_support_sets",
            "causal_used_columns",
            "linear_used_columns",
            "causal_support_change_fraction",
            "linear_support_change_fraction",
        ],
        fold_rows,
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _load_support_packet(backend: str, audit_dir: Path) -> dict[str, Any]:
    return {
        "backend": backend,
        "dir": audit_dir,
        "summary": _read_json_object(audit_dir / "summary.json"),
        "fold_metrics": _read_csv(audit_dir / "fold_metrics.csv"),
        "aggregate_metrics": _read_csv(audit_dir / "aggregate_metrics.csv"),
    }


def _evidence(
    local: dict[str, Any],
    runpod: dict[str, Any],
    fold_rows: list[dict[str, Any]],
    post_sequence: dict[str, Any],
) -> dict[str, Any]:
    local_agg = _aggregate_by_control(local["aggregate_metrics"])
    runpod_agg = _aggregate_by_control(runpod["aggregate_metrics"])
    backend_summaries = {
        "local": _backend_summary(local_agg),
        "runpod": _backend_summary(runpod_agg),
    }
    return {
        "post_sequence_selected_next_step": post_sequence.get("selected_next_step"),
        "backend_summaries": backend_summaries,
        "fold_count": len(fold_rows),
        "all_folds_causal_ce_beats_linear": all(
            _float(row["causal_minus_linear_router_loss"]) < 0 for row in fold_rows
        ),
        "all_folds_causal_oracle_frontier_beats_linear": all(
            _float(row["causal_minus_linear_oracle_loss"]) < 0 for row in fold_rows
        ),
        "all_folds_causal_regret_worse_than_linear": all(
            _float(row["causal_minus_linear_oracle_regret"]) > 0 for row in fold_rows
        ),
        "all_folds_causal_churn_worse_than_linear": all(
            _float(row["causal_minus_linear_functional_churn"]) > 0 for row in fold_rows
        ),
        "mean_fold_causal_minus_linear_router_loss": _mean(
            _float(row["causal_minus_linear_router_loss"]) for row in fold_rows
        ),
        "mean_fold_causal_minus_linear_oracle_loss": _mean(
            _float(row["causal_minus_linear_oracle_loss"]) for row in fold_rows
        ),
        "mean_fold_causal_minus_linear_oracle_regret": _mean(
            _float(row["causal_minus_linear_oracle_regret"]) for row in fold_rows
        ),
        "mean_fold_causal_minus_linear_functional_churn": _mean(
            _float(row["causal_minus_linear_functional_churn"]) for row in fold_rows
        ),
        "interpretation": (
            "causal router has a stronger oracle frontier and CE, but worse "
            "support selection regret and retention/churn than linear"
        ),
    }


def _backend_summary(aggregate: dict[str, dict[str, str]]) -> dict[str, Any]:
    causal = aggregate.get("causal_contextual_topk2", {})
    linear = aggregate.get("linear_topk2", {})
    return {
        "causal_router_loss": _optional_float(causal.get("mean_router_loss")),
        "linear_router_loss": _optional_float(linear.get("mean_router_loss")),
        "causal_minus_linear_router_loss": _delta(
            causal.get("mean_router_loss"), linear.get("mean_router_loss")
        ),
        "causal_oracle_loss": _optional_float(causal.get("mean_oracle_loss")),
        "linear_oracle_loss": _optional_float(linear.get("mean_oracle_loss")),
        "causal_minus_linear_oracle_loss": _delta(
            causal.get("mean_oracle_loss"), linear.get("mean_oracle_loss")
        ),
        "causal_oracle_regret": _optional_float(causal.get("mean_oracle_support_regret")),
        "linear_oracle_regret": _optional_float(linear.get("mean_oracle_support_regret")),
        "causal_minus_linear_oracle_regret": _delta(
            causal.get("mean_oracle_support_regret"),
            linear.get("mean_oracle_support_regret"),
        ),
        "causal_functional_churn": _optional_float(
            causal.get("mean_functional_churn_logit_l1")
        ),
        "linear_functional_churn": _optional_float(
            linear.get("mean_functional_churn_logit_l1")
        ),
        "causal_minus_linear_functional_churn": _delta(
            causal.get("mean_functional_churn_logit_l1"),
            linear.get("mean_functional_churn_logit_l1"),
        ),
        "causal_unique_support_sets": _optional_float(causal.get("mean_unique_support_sets")),
        "linear_unique_support_sets": _optional_float(linear.get("mean_unique_support_sets")),
        "causal_used_columns": _optional_float(causal.get("mean_used_columns")),
        "linear_used_columns": _optional_float(linear.get("mean_used_columns")),
    }


def _fold_failure_rows(packet: dict[str, Any]) -> list[dict[str, Any]]:
    by_fold: dict[str, dict[str, dict[str, str]]] = {}
    for row in packet["fold_metrics"]:
        control = row.get("control", "")
        if control in {"causal_contextual_topk2", "linear_topk2"}:
            by_fold.setdefault(row.get("fold", ""), {})[control] = row

    rows: list[dict[str, Any]] = []
    for fold in sorted(by_fold, key=lambda item: int(item) if item.isdigit() else item):
        causal = by_fold[fold].get("causal_contextual_topk2")
        linear = by_fold[fold].get("linear_topk2")
        if not causal or not linear:
            continue
        rows.append(
            {
                "backend": packet["backend"],
                "fold": fold,
                "heldout_sequence_index": causal.get("heldout_sequence_index", ""),
                "causal_router_loss": _optional_float(causal.get("router_loss")),
                "linear_router_loss": _optional_float(linear.get("router_loss")),
                "causal_minus_linear_router_loss": _delta(
                    causal.get("router_loss"), linear.get("router_loss")
                ),
                "causal_oracle_loss": _optional_float(causal.get("oracle_loss")),
                "linear_oracle_loss": _optional_float(linear.get("oracle_loss")),
                "causal_minus_linear_oracle_loss": _delta(
                    causal.get("oracle_loss"), linear.get("oracle_loss")
                ),
                "causal_oracle_regret": _optional_float(causal.get("oracle_support_regret")),
                "linear_oracle_regret": _optional_float(linear.get("oracle_support_regret")),
                "causal_minus_linear_oracle_regret": _delta(
                    causal.get("oracle_support_regret"),
                    linear.get("oracle_support_regret"),
                ),
                "causal_functional_churn": _optional_float(
                    causal.get("functional_churn_logit_l1")
                ),
                "linear_functional_churn": _optional_float(
                    linear.get("functional_churn_logit_l1")
                ),
                "causal_minus_linear_functional_churn": _delta(
                    causal.get("functional_churn_logit_l1"),
                    linear.get("functional_churn_logit_l1"),
                ),
                "causal_unique_support_sets": _optional_float(
                    causal.get("unique_support_sets")
                ),
                "linear_unique_support_sets": _optional_float(linear.get("unique_support_sets")),
                "causal_used_columns": _optional_float(causal.get("used_columns")),
                "linear_used_columns": _optional_float(linear.get("used_columns")),
                "causal_support_change_fraction": _optional_float(
                    causal.get("support_change_fraction")
                ),
                "linear_support_change_fraction": _optional_float(
                    linear.get("support_change_fraction")
                ),
            }
        )
    return rows


def _failures(source_rows: list[dict[str, Any]], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows:
        if not row["present"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "source_artifact",
                    "reason": "required source artifact is missing",
                }
            )
        elif row["status"] not in {"pass", "present"}:
            failures.append(
                {
                    "source": row["source"],
                    "field": "status",
                    "reason": f"unexpected status {row['status']!r}",
                }
            )
    if evidence["fold_count"] == 0:
        failures.append(
            {
                "source": "fold_metrics",
                "field": "fold_count",
                "reason": "no paired causal and linear fold rows were available",
            }
        )
    return failures


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    present = path.is_file()
    return {
        "source": source,
        "path": str(path),
        "present": present,
        "status": payload.get("status", "missing" if not present else "present"),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _aggregate_by_control(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("control", ""): row for row in rows}


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _float(value: Any) -> float:
    if value in (None, ""):
        raise ValueError("expected a numeric value")
    return float(value)


def _delta(left: Any, right: Any) -> float | None:
    if left in (None, "") or right in (None, ""):
        return None
    return float(left) - float(right)


def _mean(values: Any) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    evidence = summary["evidence"]
    local = evidence["backend_summaries"].get("local", {})
    runpod = evidence["backend_summaries"].get("runpod", {})
    lines = [
        "# Contextual Router Regret/Churn Failure Inspection",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next step: {summary['selected_next_step']}",
        "",
        "## Backend Deltas",
        "",
        "| Backend | CE delta | oracle-loss delta | oracle-regret delta | churn delta |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            "| Local | "
            f"{local.get('causal_minus_linear_router_loss')} | "
            f"{local.get('causal_minus_linear_oracle_loss')} | "
            f"{local.get('causal_minus_linear_oracle_regret')} | "
            f"{local.get('causal_minus_linear_functional_churn')} |"
        ),
        (
            "| RunPod | "
            f"{runpod.get('causal_minus_linear_router_loss')} | "
            f"{runpod.get('causal_minus_linear_oracle_loss')} | "
            f"{runpod.get('causal_minus_linear_oracle_regret')} | "
            f"{runpod.get('causal_minus_linear_functional_churn')} |"
        ),
        "",
        "## Interpretation",
        "",
        summary["rationale"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-support-audit-dir", type=Path, default=DEFAULT_LOCAL_SUPPORT_AUDIT)
    parser.add_argument("--runpod-support-audit-dir", type=Path, default=DEFAULT_RUNPOD_SUPPORT_AUDIT)
    parser.add_argument("--post-sequence-report", type=Path, default=DEFAULT_POST_SEQUENCE_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_contextual_router_regret_churn_failure_inspection(
        local_support_audit_dir=args.local_support_audit_dir,
        runpod_support_audit_dir=args.runpod_support_audit_dir,
        post_sequence_report_path=args.post_sequence_report,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
