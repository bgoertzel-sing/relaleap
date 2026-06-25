"""Singleton-gain/regret diagnostic for the active rank-matched top-k-1 bracket."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


DEFAULT_SOURCE_AUDIT_DIR = Path(
    "results/audits/token_larger_support_wide_promoted_default_causal_column_fingerprint_stability_topk1"
)
DEFAULT_DECONFOUNDED_AUDIT_DIR = Path(
    "results/audits/token_larger_topk2_vs_rank_matched_topk1_deconfounded_intervention"
)
DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_active_rank_matched_topk1_singleton_gain_regret_diagnostic"
)

LIKELY_REAL_SINGLETON_GAIN_FAILURE = "likely_real_singleton_gain_failure_mode"
MATCHING_ARTIFACT_POSSIBLE = "matching_artifact_possible"
MIXED_SINGLETON_GAIN_EVIDENCE = "mixed_singleton_gain_evidence"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

TOPK1_VARIANT = "rank_matched_topk1_contextual"
TOPK1_INTERVENTION = "fixed_dominant_router_singleton"
CONTEXT_FIELDS = ("batch_index", "position_index", "token_index", "target_token")
STRATUM_FIELDS = (
    "position_bin",
    "token_class",
    "residual_norm_bin",
    "residual_gain_bin",
    "router_support_count_bin",
)


def run_active_topk1_singleton_gain_regret_diagnostic(
    *,
    source_audit_dir: Path = DEFAULT_SOURCE_AUDIT_DIR,
    deconfounded_audit_dir: Path = DEFAULT_DECONFOUNDED_AUDIT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Audit whether negative top-k-1 singleton gain is broad or matching-specific."""

    start = time.time()
    failures = _source_failures(source_audit_dir, deconfounded_audit_dir)
    source_summary: dict[str, Any] = {}
    intervention_rows: list[dict[str, str]] = []
    matched_context_rows: list[dict[str, str]] = []
    if not failures:
        source_summary = _read_json_object(source_audit_dir / "summary.json")
        intervention_rows = _read_csv_rows(
            source_audit_dir / "per_token_pair_interventions.csv"
        )
        matched_context_rows = _read_csv_rows(
            deconfounded_audit_dir / "paired_exact_context_deltas.csv"
        )

    topk1_rows = [
        row
        for row in intervention_rows
        if row.get("variant") == TOPK1_VARIANT
        and row.get("intervention") == TOPK1_INTERVENTION
    ]
    if intervention_rows and not topk1_rows:
        failures.append(
            {
                "field": "topk1_singleton_intervention_rows",
                "expected": f"{TOPK1_VARIANT}/{TOPK1_INTERVENTION} rows",
                "actual": 0,
            }
        )

    required_fields = {
        *CONTEXT_FIELDS,
        "position_bin",
        "token_class",
        "router_support_count",
        "router_loss",
        "singleton_left_gain",
        "fixed_support_loss_delta",
        "fixed_support_logit_mse",
        "active_rank_proxy",
    }
    if topk1_rows:
        missing = sorted(required_fields - set(topk1_rows[0]))
        if missing:
            failures.append(
                {
                    "field": "topk1_singleton_required_fields",
                    "expected": sorted(required_fields),
                    "actual_missing": missing,
                }
            )

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        evidence = {"failures": failures}
        context_rows: list[dict[str, Any]] = []
        stratum_rows: list[dict[str, Any]] = []
        rationale = (
            "The singleton-gain/regret diagnostic could not be established because "
            "required source artifacts or fields are missing."
        )
        next_step = (
            "repair or regenerate the source causal-column fingerprint and "
            "deconfounded intervention artifacts before interpreting singleton gain"
        )
    else:
        matched_contexts = {_context_key(row) for row in matched_context_rows}
        context_rows = _context_gain_rows(topk1_rows, matched_contexts)
        stratum_rows = _stratum_gain_rows(context_rows)
        evidence = _build_evidence(
            source_summary,
            topk1_rows,
            context_rows,
            stratum_rows,
            matched_contexts,
        )
        decision = _decision(evidence)
        status = "pass"
        if decision == LIKELY_REAL_SINGLETON_GAIN_FAILURE:
            rationale = (
                "Negative singleton gain persists in the raw rank-matched top-k-1 "
                "source rows, in context-level aggregates, and in the matched "
                "deconfounded subset. This points to a broad singleton-gain failure "
                "mode rather than a matched-strata artifact."
            )
            next_step = (
                "treat rank-matched top-k-1 as a low-interference control only; "
                "probe why selected singleton columns are often harmful before any "
                "GPU replication of a causal-retention claim"
            )
        elif decision == MATCHING_ARTIFACT_POSSIBLE:
            rationale = (
                "The negative singleton-gain blocker is concentrated in the matched "
                "subset or stratum aggregation while broader raw context evidence is "
                "not negative, so matching or aggregation artifacts remain plausible."
            )
            next_step = (
                "regenerate the deconfounded matching packet with explicit unmatched "
                "context accounting before treating singleton gain as a real failure"
            )
        else:
            rationale = (
                "Singleton-gain evidence is mixed across raw, matched, and stratum "
                "views, so the blocker remains active but its cause is not isolated."
            )
            next_step = (
                "add a targeted source-artifact extension that logs oracle singleton "
                "regret and random singleton controls for the same contexts"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "singleton_gain_by_context.csv", _CONTEXT_OUT_FIELDS, context_rows)
    _write_csv(out_dir / "singleton_gain_by_stratum.csv", _STRATUM_OUT_FIELDS, stratum_rows)
    summary = {
        "status": status,
        "decision": decision,
        "source_audit_dir": str(source_audit_dir),
        "deconfounded_audit_dir": str(deconfounded_audit_dir),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "evidence": evidence,
        "rationale": rationale,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "singleton_gain_by_context_csv": str(out_dir / "singleton_gain_by_context.csv"),
            "singleton_gain_by_stratum_csv": str(out_dir / "singleton_gain_by_stratum.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _source_failures(source_audit_dir: Path, deconfounded_audit_dir: Path) -> list[dict[str, Any]]:
    failures = []
    for field, path in (
        ("source_summary_json", source_audit_dir / "summary.json"),
        (
            "source_per_token_pair_interventions_csv",
            source_audit_dir / "per_token_pair_interventions.csv",
        ),
        (
            "deconfounded_paired_exact_context_deltas_csv",
            deconfounded_audit_dir / "paired_exact_context_deltas.csv",
        ),
    ):
        if not path.is_file():
            failures.append(
                {"field": field, "expected": "file exists", "actual": "missing", "path": str(path)}
            )
    return failures


def _context_gain_rows(
    topk1_rows: list[dict[str, str]],
    matched_contexts: set[tuple[str, ...]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in topk1_rows:
        grouped[_context_key(row)].append(row)
    rows = []
    for context, source_rows in sorted(grouped.items()):
        gains = _float_values(source_rows, "singleton_left_gain")
        fixed_deltas = _float_values(source_rows, "fixed_support_loss_delta")
        logit_mses = _float_values(source_rows, "fixed_support_logit_mse")
        router_losses = _float_values(source_rows, "router_loss")
        first = source_rows[0]
        rows.append(
            {
                "batch_index": context[0],
                "position_index": context[1],
                "token_index": context[2],
                "target_token": context[3],
                "matched_deconfounded_context": context in matched_contexts,
                "row_count": len(source_rows),
                "position_bin": first.get("position_bin", ""),
                "token_class": first.get("token_class", ""),
                "router_support_count_bin": _support_count_bin(first.get("router_support_count")),
                "active_rank_proxy": ",".join(sorted({row.get("active_rank_proxy", "") for row in source_rows})),
                "singleton_gain_mean": _mean_or_none(gains),
                "singleton_gain_min": _min_or_none(gains),
                "singleton_gain_positive_fraction": _fraction(value > 0.0 for value in gains),
                "singleton_regret_mean": _mean_or_none([max(0.0, -value) for value in gains]),
                "fixed_support_loss_delta_mean": _mean_or_none(fixed_deltas),
                "fixed_support_logit_mse_mean": _mean_or_none(logit_mses),
                "router_loss_mean": _mean_or_none(router_losses),
            }
        )
    return rows


def _stratum_gain_rows(context_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in context_rows:
        key = (
            str(row.get("position_bin", "")),
            str(row.get("token_class", "")),
            _numeric_bin(row.get("router_loss_mean"), "router_loss"),
            str(row.get("router_support_count_bin", "")),
            str(row.get("matched_deconfounded_context", "")),
        )
        grouped[key].append(row)
    out_rows = []
    for key, rows in sorted(grouped.items()):
        gains = _numeric_values(rows, "singleton_gain_mean")
        regrets = _numeric_values(rows, "singleton_regret_mean")
        (
            position_bin,
            token_class,
            router_loss_bin,
            router_support_count_bin,
            matched_context,
        ) = key
        out_rows.append(
            {
                "position_bin": position_bin,
                "token_class": token_class,
                "router_loss_bin": router_loss_bin,
                "router_support_count_bin": router_support_count_bin,
                "matched_deconfounded_context": matched_context,
                "context_count": len(rows),
                "singleton_gain_mean": _mean_or_none(gains),
                "singleton_gain_positive_context_fraction": _fraction(
                    value > 0.0 for value in gains
                ),
                "singleton_regret_mean": _mean_or_none(regrets),
            }
        )
    return out_rows


def _build_evidence(
    source_summary: dict[str, Any],
    topk1_rows: list[dict[str, str]],
    context_rows: list[dict[str, Any]],
    stratum_rows: list[dict[str, Any]],
    matched_contexts: set[tuple[str, ...]],
) -> dict[str, Any]:
    raw_gains = _float_values(topk1_rows, "singleton_left_gain")
    context_gains = _numeric_values(context_rows, "singleton_gain_mean")
    matched_context_gains = _numeric_values(
        [row for row in context_rows if row.get("matched_deconfounded_context")],
        "singleton_gain_mean",
    )
    unmatched_context_gains = _numeric_values(
        [row for row in context_rows if not row.get("matched_deconfounded_context")],
        "singleton_gain_mean",
    )
    stratum_gains = _numeric_values(stratum_rows, "singleton_gain_mean")
    active_ranks = sorted({row.get("active_rank_proxy", "") for row in topk1_rows if row.get("active_rank_proxy")})
    metrics = {
        "source_status": source_summary.get("status"),
        "source_decision": source_summary.get("decision"),
        "raw_topk1_singleton_row_count": len(topk1_rows),
        "raw_context_count": len(context_rows),
        "matched_deconfounded_context_count": sum(
            1 for row in context_rows if row.get("matched_deconfounded_context")
        ),
        "unmatched_context_count": sum(
            1 for row in context_rows if not row.get("matched_deconfounded_context")
        ),
        "deconfounded_matched_context_keys": len(matched_contexts),
        "active_rank_proxy_values": active_ranks,
        "raw_singleton_gain_mean": _mean_or_none(raw_gains),
        "raw_singleton_gain_positive_fraction": _fraction(value > 0.0 for value in raw_gains),
        "raw_singleton_regret_mean": _mean_or_none([max(0.0, -value) for value in raw_gains]),
        "context_singleton_gain_mean": _mean_or_none(context_gains),
        "context_singleton_gain_positive_fraction": _fraction(
            value > 0.0 for value in context_gains
        ),
        "context_singleton_regret_mean": _mean_or_none(
            [max(0.0, -value) for value in context_gains]
        ),
        "matched_context_singleton_gain_mean": _mean_or_none(matched_context_gains),
        "matched_context_singleton_gain_positive_fraction": _fraction(
            value > 0.0 for value in matched_context_gains
        ),
        "unmatched_context_singleton_gain_mean": _mean_or_none(unmatched_context_gains),
        "unmatched_context_singleton_gain_positive_fraction": _fraction(
            value > 0.0 for value in unmatched_context_gains
        ),
        "stratum_singleton_gain_mean": _mean_or_none(stratum_gains),
        "stratum_singleton_gain_positive_fraction": _fraction(
            value > 0.0 for value in stratum_gains
        ),
    }
    return {
        "metrics": metrics,
        "signals": {
            "active_rank_matched": active_ranks == ["1"],
            "raw_singleton_gain_negative": _lt(metrics["raw_singleton_gain_mean"], 0.0),
            "context_singleton_gain_negative": _lt(
                metrics["context_singleton_gain_mean"], 0.0
            ),
            "matched_context_singleton_gain_negative": _lt(
                metrics["matched_context_singleton_gain_mean"], 0.0
            ),
            "unmatched_context_singleton_gain_negative_or_absent": (
                metrics["unmatched_context_count"] == 0
                or _lt(metrics["unmatched_context_singleton_gain_mean"], 0.0)
            ),
            "raw_positive_fraction_below_half": _lt(
                metrics["raw_singleton_gain_positive_fraction"], 0.5
            ),
            "context_positive_fraction_below_half": _lt(
                metrics["context_singleton_gain_positive_fraction"], 0.5
            ),
            "matched_subset_covers_most_contexts": (
                metrics["raw_context_count"] > 0
                and metrics["matched_deconfounded_context_count"]
                / metrics["raw_context_count"]
                >= 0.75
            ),
            "broad_negative_singleton_gain": all(
                (
                    _lt(metrics["raw_singleton_gain_mean"], 0.0),
                    _lt(metrics["context_singleton_gain_mean"], 0.0),
                    _lt(metrics["matched_context_singleton_gain_mean"], 0.0),
                    _lt(metrics["raw_singleton_gain_positive_fraction"], 0.5),
                    _lt(metrics["context_singleton_gain_positive_fraction"], 0.5),
                )
            ),
        },
    }


def _decision(evidence: dict[str, Any]) -> str:
    signals = evidence["signals"]
    metrics = evidence["metrics"]
    if not signals["active_rank_matched"]:
        return INSUFFICIENT_EVIDENCE
    if (
        signals["broad_negative_singleton_gain"]
        and signals["unmatched_context_singleton_gain_negative_or_absent"]
        and signals["matched_subset_covers_most_contexts"]
    ):
        return LIKELY_REAL_SINGLETON_GAIN_FAILURE
    if (
        signals["matched_context_singleton_gain_negative"]
        and not signals["context_singleton_gain_negative"]
    ) or (
        metrics["unmatched_context_count"] > 0
        and not signals["unmatched_context_singleton_gain_negative_or_absent"]
    ):
        return MATCHING_ARTIFACT_POSSIBLE
    return MIXED_SINGLETON_GAIN_EVIDENCE


def _context_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in CONTEXT_FIELDS)


def _float_values(rows: list[dict[str, str]], field: str) -> list[float]:
    return [value for row in rows if (value := _float_or_none(row.get(field))) is not None]


def _numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [value for row in rows if isinstance((value := row.get(field)), float)]


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _min_or_none(values: list[float]) -> float | None:
    return min(values) if values else None


def _fraction(values: Iterable[bool]) -> float | None:
    materialized = list(values)
    if not materialized:
        return None
    return sum(1 for value in materialized if value) / len(materialized)


def _lt(value: Any, threshold: float) -> bool:
    return isinstance(value, float) and value < threshold


def _support_count_bin(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return "unknown"
    if number <= 8:
        return "low"
    if number <= 16:
        return "mid"
    return "high"


def _numeric_bin(value: Any, label: str) -> str:
    number = _float_or_none(value)
    if number is None:
        return f"{label}_unknown"
    if number < 2.5:
        return f"{label}_low"
    if number < 3.5:
        return f"{label}_mid"
    return f"{label}_high"


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["evidence"].get("metrics", {})
    lines = [
        "# Active Top-k-1 Singleton-Gain/Regret Diagnostic",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Raw singleton rows: `{metrics.get('raw_topk1_singleton_row_count')}`",
        f"- Raw context count: `{metrics.get('raw_context_count')}`",
        "- Raw singleton gain mean: "
        f"`{metrics.get('raw_singleton_gain_mean')}`",
        "- Context singleton gain mean: "
        f"`{metrics.get('context_singleton_gain_mean')}`",
        "- Matched context singleton gain mean: "
        f"`{metrics.get('matched_context_singleton_gain_mean')}`",
        "- Unmatched context singleton gain mean: "
        f"`{metrics.get('unmatched_context_singleton_gain_mean')}`",
        "- Raw singleton regret mean: "
        f"`{metrics.get('raw_singleton_regret_mean')}`",
        "",
        "## Rationale",
        "",
        summary["rationale"],
        "",
        "## Next Step",
        "",
        summary["next_step"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


_CONTEXT_OUT_FIELDS = [
    "batch_index",
    "position_index",
    "token_index",
    "target_token",
    "matched_deconfounded_context",
    "row_count",
    "position_bin",
    "token_class",
    "router_support_count_bin",
    "active_rank_proxy",
    "singleton_gain_mean",
    "singleton_gain_min",
    "singleton_gain_positive_fraction",
    "singleton_regret_mean",
    "fixed_support_loss_delta_mean",
    "fixed_support_logit_mse_mean",
    "router_loss_mean",
]

_STRATUM_OUT_FIELDS = [
    "position_bin",
    "token_class",
    "router_loss_bin",
    "router_support_count_bin",
    "matched_deconfounded_context",
    "context_count",
    "singleton_gain_mean",
    "singleton_gain_positive_context_fraction",
    "singleton_regret_mean",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit-dir", type=Path, default=DEFAULT_SOURCE_AUDIT_DIR)
    parser.add_argument(
        "--deconfounded-audit-dir", type=Path, default=DEFAULT_DECONFOUNDED_AUDIT_DIR
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_active_topk1_singleton_gain_regret_diagnostic(
        source_audit_dir=args.source_audit_dir,
        deconfounded_audit_dir=args.deconfounded_audit_dir,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "evidence": summary["evidence"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
