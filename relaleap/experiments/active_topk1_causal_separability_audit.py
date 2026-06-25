"""Local causal-separability audit for the active rank-matched top-k-1 bracket."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path
from typing import Any


DEFAULT_DECONFOUNDED_AUDIT_DIR = Path(
    "results/audits/token_larger_topk2_vs_rank_matched_topk1_deconfounded_intervention"
)
DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_active_rank_matched_topk1_causal_separability"
)
ACTIVE_TOPK1_SEPARABILITY_AUDIT_ESTABLISHED = (
    "active_topk1_causal_separability_audit_established"
)
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def run_active_topk1_causal_separability_audit(
    audit_dir: Path = DEFAULT_DECONFOUNDED_AUDIT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Summarize active top-k-1 singleton evidence from exact-context artifacts."""

    start = time.time()
    failures: list[dict[str, Any]] = []
    summary_path = audit_dir / "summary.json"
    matched_path = audit_dir / "matched_deconfounded_strata.csv"
    paired_path = audit_dir / "paired_exact_context_deltas.csv"
    for field, path in (
        ("deconfounded_summary_json", summary_path),
        ("matched_deconfounded_strata_csv", matched_path),
        ("paired_exact_context_deltas_csv", paired_path),
    ):
        if not path.is_file():
            failures.append(
                {
                    "field": field,
                    "expected": "file exists",
                    "actual": "missing",
                    "path": str(path),
                }
            )

    source_summary: dict[str, Any] = {}
    matched_rows: list[dict[str, str]] = []
    paired_rows: list[dict[str, str]] = []
    if not failures:
        source_summary = _read_json_object(summary_path)
        matched_rows = _read_csv_rows(matched_path)
        paired_rows = _read_csv_rows(paired_path)
        if source_summary.get("status") != "pass":
            failures.append(
                {
                    "field": "deconfounded_summary.status",
                    "expected": "pass",
                    "actual": source_summary.get("status"),
                    "path": str(summary_path),
                }
            )
        if not matched_rows:
            failures.append(
                {
                    "field": "matched_deconfounded_strata",
                    "expected": "at least one matched stratum",
                    "actual": 0,
                    "path": str(matched_path),
                }
            )
        if not paired_rows:
            failures.append(
                {
                    "field": "paired_exact_context_deltas",
                    "expected": "at least one exact paired context",
                    "actual": 0,
                    "path": str(paired_path),
                }
            )

    evidence = _build_evidence(
        source_summary,
        matched_rows,
        paired_rows,
        failures=failures,
        audit_dir=audit_dir,
    )
    signals = evidence["signals"]
    selected = (
        not failures
        and signals["topk1_ce_primary"]
        and signals["exact_context_coverage_present"]
        and signals["active_rank_matched_topk1_rows_present"]
        and source_summary.get("decision")
        == "topk2_comparative_causal_cooperation_not_supported"
    )
    if selected:
        status = "pass"
        decision = ACTIVE_TOPK1_SEPARABILITY_AUDIT_ESTABLISHED
        rationale = (
            "The active rank-matched contextual top-k-1 bracket has complete "
            "exact-context deconfounded artifacts and remains CE-primary versus "
            "the promoted top-k-2 reference. This audit records singleton gain, "
            "fixed-support loss delta, logit churn, and residual-stream churn as "
            "the local top-k-1 separability packet; it does not reopen the closed "
            "top-k-2 causal-cooperation claim."
        )
        next_step = (
            "run a bounded local retention/functional-churn probe for the active "
            "rank-matched contextual top-k-1 bracket using the separability packet "
            "as the source artifact"
        )
    else:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        rationale = (
            "The active top-k-1 separability packet cannot be established from "
            "the current exact-context deconfounded artifacts."
        )
        next_step = (
            "repair or regenerate the deconfounded intervention audit before "
            "building top-k-1 separability evidence"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    stratum_rows = _topk1_stratum_rows(matched_rows)
    context_rows = _topk1_context_rows(paired_rows)
    _write_csv(out_dir / "topk1_separability_by_stratum.csv", _STRATUM_FIELDS, stratum_rows)
    _write_csv(out_dir / "topk1_separability_by_context.csv", _CONTEXT_FIELDS, context_rows)
    summary = {
        "status": status,
        "decision": decision,
        "audit_dir": str(audit_dir),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "evidence": evidence,
        "rationale": rationale,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "topk1_separability_by_stratum_csv": str(
                out_dir / "topk1_separability_by_stratum.csv"
            ),
            "topk1_separability_by_context_csv": str(
                out_dir / "topk1_separability_by_context.csv"
            ),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _build_evidence(
    source_summary: dict[str, Any],
    matched_rows: list[dict[str, str]],
    paired_rows: list[dict[str, str]],
    *,
    failures: list[dict[str, Any]],
    audit_dir: Path,
) -> dict[str, Any]:
    source_evidence = source_summary.get("evidence", {})
    if not isinstance(source_evidence, dict):
        source_evidence = {}
    topk1_ce = _float_or_none(source_evidence.get("topk1_alpha0_ce_loss"))
    topk2_ce = _float_or_none(source_evidence.get("topk2_alpha0_ce_loss"))
    topk1_active_ranks = sorted(
        {
            value
            for row in matched_rows
            for value in str(row.get("topk1_active_rank_proxy", "")).split(",")
            if value
        }
    )
    topk1_singleton_gains = _float_values(matched_rows, "topk1_singleton_gain_mean")
    topk1_fixed_deltas = _float_values(matched_rows, "topk1_fixed_support_loss_delta_mean")
    topk1_logit_mses = _float_values(matched_rows, "topk1_fixed_support_logit_mse_mean")
    topk1_residual_l2 = _float_values(matched_rows, "topk1_residual_stream_l2_delta_mean")
    topk2_incremental_minus_topk1 = _float_values(
        matched_rows, "topk2_incremental_pair_gain_minus_topk1_singleton"
    )
    context_singleton_gains = _float_values(paired_rows, "topk1_singleton_gain_mean")
    context_topk2_minus_topk1_fixed_delta = _float_values(
        paired_rows, "topk2_fixed_delta_minus_topk1"
    )
    metrics = {
        "source_decision": source_summary.get("decision"),
        "topk1_alpha0_ce_loss": topk1_ce,
        "topk2_alpha0_ce_loss": topk2_ce,
        "topk2_ce_deficit_vs_topk1": source_evidence.get(
            "topk2_ce_deficit_vs_topk1"
        ),
        "matched_deconfounded_strata_count": len(matched_rows),
        "paired_exact_context_count": len(paired_rows),
        "source_matched_exact_context_count": source_evidence.get(
            "matched_exact_context_count"
        ),
        "matched_topk1_context_fraction": source_evidence.get(
            "matched_topk1_context_fraction"
        ),
        "unmatched_topk1_context_count": source_evidence.get(
            "unmatched_topk1_context_count"
        ),
        "topk1_active_rank_proxy_values": topk1_active_ranks,
        "topk1_singleton_gain_mean": _mean(topk1_singleton_gains),
        "topk1_singleton_gain_positive_strata_fraction": _fraction(
            value > 0.0 for value in topk1_singleton_gains
        ),
        "topk1_fixed_support_loss_delta_mean": _mean(topk1_fixed_deltas),
        "topk1_fixed_support_loss_delta_negative_strata_fraction": _fraction(
            value < 0.0 for value in topk1_fixed_deltas
        ),
        "topk1_fixed_support_logit_mse_mean": _mean(topk1_logit_mses),
        "topk1_residual_stream_l2_delta_mean": _mean(topk1_residual_l2),
        "context_level_topk1_singleton_gain_mean": _mean(context_singleton_gains),
        "context_level_topk1_singleton_gain_positive_fraction": _fraction(
            value > 0.0 for value in context_singleton_gains
        ),
        "context_level_topk2_fixed_delta_minus_topk1_mean": _mean(
            context_topk2_minus_topk1_fixed_delta
        ),
        "topk2_incremental_pair_gain_minus_topk1_singleton_mean": _mean(
            topk2_incremental_minus_topk1
        ),
        "topk2_incremental_pair_gain_positive_strata_fraction": source_evidence.get(
            "topk2_incremental_pair_gain_positive_strata_fraction"
        ),
        "topk2_fixed_support_cleaner_strata_fraction": source_evidence.get(
            "topk2_fixed_support_cleaner_strata_fraction"
        ),
        "topk2_functional_churn_cleaner_strata_fraction": source_evidence.get(
            "topk2_functional_churn_cleaner_strata_fraction"
        ),
        "deconfounded_topk2_pair_synergy_mean": source_evidence.get(
            "deconfounded_topk2_pair_synergy_mean"
        ),
    }
    return {
        "audit_dir": str(audit_dir),
        "metrics": metrics,
        "signals": {
            "topk1_ce_primary": (
                topk1_ce is not None and topk2_ce is not None and topk1_ce < topk2_ce
            ),
            "exact_context_coverage_present": (
                _float_or_none(source_evidence.get("matched_exact_context_count"))
                is not None
                and _float_or_none(source_evidence.get("matched_topk1_context_fraction"))
                is not None
                and len(paired_rows) > 0
            ),
            "active_rank_matched_topk1_rows_present": topk1_active_ranks == ["1"],
            "topk2_reference_only": (
                source_summary.get("decision")
                == "topk2_comparative_causal_cooperation_not_supported"
            ),
        },
        "failures": failures,
    }


def _topk1_stratum_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        out_rows.append(
            {
                "position_bin": row.get("position_bin"),
                "token_class": row.get("token_class"),
                "residual_norm_bin": row.get("residual_norm_bin"),
                "residual_gain_bin": row.get("residual_gain_bin"),
                "support_count_bin": row.get("support_count_bin"),
                "matched_exact_context_count": row.get("matched_exact_context_count"),
                "topk1_row_count": row.get("topk1_row_count"),
                "topk1_active_rank_proxy": row.get("topk1_active_rank_proxy"),
                "topk1_router_loss_mean": row.get("topk1_router_loss_mean"),
                "topk1_singleton_gain_mean": row.get("topk1_singleton_gain_mean"),
                "topk1_fixed_support_loss_delta_mean": row.get(
                    "topk1_fixed_support_loss_delta_mean"
                ),
                "topk1_fixed_support_logit_mse_mean": row.get(
                    "topk1_fixed_support_logit_mse_mean"
                ),
                "topk1_residual_stream_l2_delta_mean": row.get(
                    "topk1_residual_stream_l2_delta_mean"
                ),
                "topk2_incremental_pair_gain_minus_topk1_singleton": row.get(
                    "topk2_incremental_pair_gain_minus_topk1_singleton"
                ),
                "topk2_pair_synergy_mean": row.get("topk2_pair_synergy_mean"),
            }
        )
    return out_rows


def _topk1_context_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        out_rows.append(
            {
                "batch_index": row.get("batch_index"),
                "position_index": row.get("position_index"),
                "token_index": row.get("token_index"),
                "target_token": row.get("target_token"),
                "matched_context_stratum_count": row.get(
                    "matched_context_stratum_count"
                ),
                "topk1_row_count": row.get("topk1_row_count"),
                "topk1_router_loss_mean": row.get("topk1_router_loss_mean"),
                "topk1_singleton_gain_mean": row.get("topk1_singleton_gain_mean"),
                "topk2_fixed_delta_minus_topk1": row.get(
                    "topk2_fixed_delta_minus_topk1"
                ),
                "topk2_logit_mse_minus_topk1": row.get(
                    "topk2_logit_mse_minus_topk1"
                ),
            }
        )
    return out_rows


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float_values(rows: list[dict[str, str]], field: str) -> list[float]:
    return [
        value
        for row in rows
        if (value := _float_or_none(row.get(field))) is not None
    ]


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


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _fraction(values: Any) -> float | None:
    materialized = list(values)
    if not materialized:
        return None
    return float(sum(1 for value in materialized if value) / len(materialized))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["evidence"]["metrics"]
    lines = [
        "# Active Top-k-1 Causal Separability Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Source audit: `{summary['audit_dir']}`",
        f"- Top-k-1 alpha-0 CE: `{metrics['topk1_alpha0_ce_loss']}`",
        f"- Top-k-2 reference alpha-0 CE: `{metrics['topk2_alpha0_ce_loss']}`",
        f"- Matched strata: `{metrics['matched_deconfounded_strata_count']}`",
        f"- Paired exact contexts: `{metrics['paired_exact_context_count']}`",
        f"- Top-k-1 active-rank values: `{metrics['topk1_active_rank_proxy_values']}`",
        f"- Top-k-1 singleton gain mean: `{metrics['topk1_singleton_gain_mean']}`",
        "- Top-k-1 singleton gain positive strata fraction: "
        f"`{metrics['topk1_singleton_gain_positive_strata_fraction']}`",
        "- Top-k-1 fixed-support loss-delta mean: "
        f"`{metrics['topk1_fixed_support_loss_delta_mean']}`",
        "- Top-k-1 fixed-support negative-delta strata fraction: "
        f"`{metrics['topk1_fixed_support_loss_delta_negative_strata_fraction']}`",
        "- Top-k-1 fixed-support logit MSE mean: "
        f"`{metrics['topk1_fixed_support_logit_mse_mean']}`",
        f"- Rationale: {summary['rationale']}",
        "",
        "## Next Step",
        "",
        summary["next_step"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


_STRATUM_FIELDS = [
    "position_bin",
    "token_class",
    "residual_norm_bin",
    "residual_gain_bin",
    "support_count_bin",
    "matched_exact_context_count",
    "topk1_row_count",
    "topk1_active_rank_proxy",
    "topk1_router_loss_mean",
    "topk1_singleton_gain_mean",
    "topk1_fixed_support_loss_delta_mean",
    "topk1_fixed_support_logit_mse_mean",
    "topk1_residual_stream_l2_delta_mean",
    "topk2_incremental_pair_gain_minus_topk1_singleton",
    "topk2_pair_synergy_mean",
]

_CONTEXT_FIELDS = [
    "batch_index",
    "position_index",
    "token_index",
    "target_token",
    "matched_context_stratum_count",
    "topk1_row_count",
    "topk1_router_loss_mean",
    "topk1_singleton_gain_mean",
    "topk2_fixed_delta_minus_topk1",
    "topk2_logit_mse_minus_topk1",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_DECONFOUNDED_AUDIT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_active_topk1_causal_separability_audit(args.audit_dir, args.out)
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
