"""Functional-retention audit for the active rank-matched top-k-1 bracket."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from relaleap.experiments.active_topk1_retention_churn_probe import (
    ACTIVE_TOPK1_RETENTION_CHURN_PROBE_ESTABLISHED,
)


DEFAULT_PROBE_DIRS = (
    Path("results/audits/token_larger_active_rank_matched_topk1_retention_churn_probe"),
    Path(
        "results/audits/token_larger_active_rank_matched_topk1_retention_churn_probe_seed2"
    ),
)
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_active_topk1_functional_retention_audit"
)

FUNCTIONAL_RETENTION_CLAIM_SUPPORTED = "functional_retention_claim_supported"
FUNCTIONAL_RETENTION_BRACKET_ONLY = "functional_retention_bracket_only"
BLOCKED_BY_NEGATIVE_SINGLETON_GAIN = "blocked_by_negative_singleton_gain"
BLOCKED_BY_CONTROL_MATCH_FAILURE = "blocked_by_control_match_failure"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

_REQUIRED_VARIANTS = (
    "rank_matched_contextual_topk1",
    "promoted_contextual_topk2",
    "norm_matched_dense_active_rank",
)
_REQUIRED_SIGNALS = (
    "required_variants_present",
    "topk1_support_churn_lower_than_topk2",
    "topk1_logit_churn_not_higher_than_topk2",
    "topk1_transfer_improvement_at_least_topk2",
    "source_singleton_gain_still_negative",
)
_METRIC_FIELDS = (
    "anchor_support_churn_after_transfer",
    "anchor_logit_mse_drift",
    "anchor_residual_stream_l2_drift",
    "anchor_ce_drift",
    "transfer_ce_improvement",
    "commutator_anchor_logit_mse",
    "commutator_transfer_logit_mse",
    "commutator_anchor_residual_stream_l2",
    "commutator_transfer_residual_stream_l2",
)


def run_active_topk1_functional_retention_audit(
    *,
    probe_dirs: tuple[Path, ...] = DEFAULT_PROBE_DIRS,
    out_dir: Path = DEFAULT_OUT_DIR,
    ce_guardrail: float = 0.05,
) -> dict[str, Any]:
    """Summarize completed probe packets as a functional-retention bracket."""

    packet_rows = [_packet_row(index, path) for index, path in enumerate(probe_dirs, 1)]
    failures = [failure for row in packet_rows for failure in _packet_failures(row)]
    aggregates = _aggregate(packet_rows, ce_guardrail=ce_guardrail)
    claim_signals = _claim_signals(aggregates, packet_rows)
    enough_packets = len(packet_rows) >= 2

    if not enough_packets:
        failures.append(
            {
                "field": "packet_count",
                "expected": "at least 2",
                "actual": len(packet_rows),
            }
        )

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        claim_status = INSUFFICIENT_EVIDENCE
        rationale = (
            "The functional-retention audit could not be established because one "
            "or more required probe packets, variants, or fields are missing or "
            "failing."
        )
        next_step = (
            "repair or regenerate the active top-k-1 retention/churn probe packets "
            "before interpreting functional-retention evidence"
        )
    elif claim_signals["claim_supported"]:
        status = "pass"
        decision = FUNCTIONAL_RETENTION_CLAIM_SUPPORTED
        claim_status = FUNCTIONAL_RETENTION_CLAIM_SUPPORTED
        rationale = (
            "The active rank-matched contextual top-k-1 bracket shows lower "
            "functional/logit drift and lower support-identity churn than the "
            "promoted top-k-2 reference across local packets, while satisfying CE "
            "guardrails and not being explained away by the dense active-rank "
            "control."
        )
        next_step = (
            "replicate the functional-retention audit on Colab/GPU before treating "
            "it as a backend-stable causal-retention claim"
        )
    else:
        status = "pass"
        decision = FUNCTIONAL_RETENTION_BRACKET_ONLY
        claim_status = _blocked_claim_status(claim_signals)
        rationale = (
            "The active rank-matched contextual top-k-1 bracket remains useful as "
            "a low-churn functional-retention bracket, but the current packets do "
            "not support a singleton causal-retention claim. The source singleton "
            "gain is still negative, so any finite-update order-sensitivity "
            "advantage remains bracket evidence rather than a causal-retention "
            "claim."
        )
        next_step = (
            "use the finite-update order-sensitivity evidence to decide whether a "
            "targeted Colab/GPU repeat is worth running despite the negative "
            "singleton-gain blocker"
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "probe_dirs": [str(path) for path in probe_dirs],
        "out_dir": str(out_dir),
        "ce_guardrail": ce_guardrail,
        "evidence": {
            "packets": packet_rows,
            "aggregates": aggregates,
            "claim_signals": claim_signals,
            "missing_evidence": {
                "finite_update_commutator": (
                    "present"
                    if claim_signals["finite_update_commutator_present"]
                    else "missing: retention/churn probe packets do not expose A-to-B versus B-to-A order metrics"
                ),
                "support_jaccard_churn": (
                    "missing: existing packets expose exact support churn only"
                ),
                "oracle_support_regret": "missing from retention/churn probe packets",
            },
        },
        "failures": failures,
        "rationale": rationale,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "packet_metrics_csv": str(out_dir / "packet_metrics.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_packet_metrics(out_dir / "packet_metrics.csv", packet_rows)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _packet_row(index: int, probe_dir: Path) -> dict[str, Any]:
    summary_path = probe_dir / "summary.json"
    summary = _read_json_object(summary_path)
    evidence = summary.get("evidence", {})
    metrics = evidence.get("metrics", {}) if isinstance(evidence, dict) else {}
    signals = evidence.get("signals", {}) if isinstance(evidence, dict) else {}
    variants = {
        str(row.get("variant")): row
        for row in summary.get("audit", {}).get("variants", [])
        if isinstance(row, dict)
    }
    row: dict[str, Any] = {
        "packet": f"seed{index}",
        "probe_dir": str(probe_dir),
        "summary_path": str(summary_path),
        "summary_present": summary_path.is_file(),
        "status": summary.get("status"),
        "decision": summary.get("decision"),
        "required_variants_present": all(name in variants for name in _REQUIRED_VARIANTS)
        or bool(signals.get("required_variants_present")),
    }
    for signal in _REQUIRED_SIGNALS:
        row[signal] = bool(signals.get(signal))
    row["source_topk1_singleton_gain_mean"] = _float_or_none(
        metrics.get("source_topk1_singleton_gain_mean")
    )
    row["source_context_level_topk1_singleton_gain_mean"] = _float_or_none(
        metrics.get("source_context_level_topk1_singleton_gain_mean")
    )
    for variant in _REQUIRED_VARIANTS:
        source = variants.get(variant, {})
        prefix = _variant_prefix(variant)
        for field in _METRIC_FIELDS:
            value = source.get(field)
            if value is None:
                value = metrics.get(f"{prefix}_{field}")
            row[f"{prefix}_{field}"] = _float_or_none(value)
    row["support_churn_advantage_topk1_vs_topk2"] = _delta(
        row["topk2_anchor_support_churn_after_transfer"],
        row["topk1_anchor_support_churn_after_transfer"],
    )
    row["logit_churn_advantage_topk1_vs_topk2"] = _delta(
        row["topk2_anchor_logit_mse_drift"],
        row["topk1_anchor_logit_mse_drift"],
    )
    row["residual_stream_churn_advantage_topk1_vs_topk2"] = _delta(
        row["topk2_anchor_residual_stream_l2_drift"],
        row["topk1_anchor_residual_stream_l2_drift"],
    )
    row["commutator_anchor_logit_mse_advantage_topk1_vs_topk2"] = _delta(
        row["topk2_commutator_anchor_logit_mse"],
        row["topk1_commutator_anchor_logit_mse"],
    )
    row["commutator_transfer_logit_mse_advantage_topk1_vs_topk2"] = _delta(
        row["topk2_commutator_transfer_logit_mse"],
        row["topk1_commutator_transfer_logit_mse"],
    )
    row["commutator_anchor_logit_mse_advantage_topk1_vs_dense"] = _delta(
        row["dense_commutator_anchor_logit_mse"],
        row["topk1_commutator_anchor_logit_mse"],
    )
    row["commutator_transfer_logit_mse_advantage_topk1_vs_dense"] = _delta(
        row["dense_commutator_transfer_logit_mse"],
        row["topk1_commutator_transfer_logit_mse"],
    )
    row["transfer_improvement_advantage_topk1_vs_topk2"] = _delta(
        row["topk1_transfer_ce_improvement"],
        row["topk2_transfer_ce_improvement"],
    )
    row["transfer_improvement_advantage_topk1_vs_dense"] = _delta(
        row["topk1_transfer_ce_improvement"],
        row["dense_transfer_ce_improvement"],
    )
    return row


def _packet_failures(row: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if not row["summary_present"]:
        return [
            {
                "packet": row["packet"],
                "field": "summary_json",
                "expected": "file exists",
                "actual": "missing",
                "path": row["summary_path"],
            }
        ]
    if row["status"] != "pass":
        failures.append(
            {
                "packet": row["packet"],
                "field": "status",
                "expected": "pass",
                "actual": row["status"],
            }
        )
    if row["decision"] != ACTIVE_TOPK1_RETENTION_CHURN_PROBE_ESTABLISHED:
        failures.append(
            {
                "packet": row["packet"],
                "field": "decision",
                "expected": ACTIVE_TOPK1_RETENTION_CHURN_PROBE_ESTABLISHED,
                "actual": row["decision"],
            }
        )
    if not row["required_variants_present"]:
        failures.append(
            {
                "packet": row["packet"],
                "field": "required_variants_present",
                "expected": True,
                "actual": row["required_variants_present"],
            }
        )
    required_metric_fields = (
        "topk1_anchor_support_churn_after_transfer",
        "topk2_anchor_support_churn_after_transfer",
        "topk1_anchor_logit_mse_drift",
        "topk2_anchor_logit_mse_drift",
        "topk1_anchor_ce_drift",
        "topk2_anchor_ce_drift",
        "dense_anchor_ce_drift",
        "topk1_transfer_ce_improvement",
        "topk2_transfer_ce_improvement",
        "dense_transfer_ce_improvement",
        "topk1_commutator_anchor_logit_mse",
        "topk2_commutator_anchor_logit_mse",
        "dense_commutator_anchor_logit_mse",
        "topk1_commutator_transfer_logit_mse",
        "topk2_commutator_transfer_logit_mse",
        "dense_commutator_transfer_logit_mse",
        "source_topk1_singleton_gain_mean",
    )
    for field in required_metric_fields:
        if row.get(field) is None:
            failures.append(
                {
                    "packet": row["packet"],
                    "field": field,
                    "expected": "numeric value",
                    "actual": None,
                }
            )
    return failures


def _aggregate(rows: list[dict[str, Any]], *, ce_guardrail: float) -> dict[str, Any]:
    def values(field: str) -> list[float]:
        return [value for value in (row.get(field) for row in rows) if isinstance(value, float)]

    aggregate_fields = (
        "topk1_anchor_support_churn_after_transfer",
        "topk2_anchor_support_churn_after_transfer",
        "topk1_anchor_logit_mse_drift",
        "topk2_anchor_logit_mse_drift",
        "topk1_anchor_residual_stream_l2_drift",
        "topk2_anchor_residual_stream_l2_drift",
        "topk1_anchor_ce_drift",
        "topk2_anchor_ce_drift",
        "dense_anchor_ce_drift",
        "topk1_transfer_ce_improvement",
        "topk2_transfer_ce_improvement",
        "dense_transfer_ce_improvement",
        "topk1_commutator_anchor_logit_mse",
        "topk2_commutator_anchor_logit_mse",
        "dense_commutator_anchor_logit_mse",
        "topk1_commutator_transfer_logit_mse",
        "topk2_commutator_transfer_logit_mse",
        "dense_commutator_transfer_logit_mse",
        "source_topk1_singleton_gain_mean",
        "source_context_level_topk1_singleton_gain_mean",
        "support_churn_advantage_topk1_vs_topk2",
        "logit_churn_advantage_topk1_vs_topk2",
        "residual_stream_churn_advantage_topk1_vs_topk2",
        "commutator_anchor_logit_mse_advantage_topk1_vs_topk2",
        "commutator_transfer_logit_mse_advantage_topk1_vs_topk2",
        "commutator_anchor_logit_mse_advantage_topk1_vs_dense",
        "commutator_transfer_logit_mse_advantage_topk1_vs_dense",
        "transfer_improvement_advantage_topk1_vs_topk2",
        "transfer_improvement_advantage_topk1_vs_dense",
    )
    aggregates = {
        f"mean_{field}": _mean_or_none(values(field))
        for field in aggregate_fields
    }
    aggregates.update(
        {
            f"min_{field}": _min_or_none(values(field))
            for field in (
                "support_churn_advantage_topk1_vs_topk2",
                "logit_churn_advantage_topk1_vs_topk2",
                "residual_stream_churn_advantage_topk1_vs_topk2",
                "commutator_anchor_logit_mse_advantage_topk1_vs_topk2",
                "commutator_transfer_logit_mse_advantage_topk1_vs_topk2",
                "commutator_anchor_logit_mse_advantage_topk1_vs_dense",
                "commutator_transfer_logit_mse_advantage_topk1_vs_dense",
                "transfer_improvement_advantage_topk1_vs_topk2",
                "transfer_improvement_advantage_topk1_vs_dense",
            )
        }
    )
    aggregates["ce_guardrail_all_packets"] = all(
        _ce_guardrail_passes(row.get(field), ce_guardrail)
        for row in rows
        for field in (
            "topk1_anchor_ce_drift",
            "topk2_anchor_ce_drift",
            "dense_anchor_ce_drift",
        )
    )
    aggregates["negative_singleton_gain_all_packets"] = all(
        isinstance(row.get("source_topk1_singleton_gain_mean"), float)
        and row["source_topk1_singleton_gain_mean"] < 0.0
        for row in rows
    )
    return aggregates


def _claim_signals(
    aggregates: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, bool]:
    support_churn_cleaner = all(
        isinstance(row.get("support_churn_advantage_topk1_vs_topk2"), float)
        and row["support_churn_advantage_topk1_vs_topk2"] > 0.0
        for row in rows
    )
    logit_churn_cleaner = all(
        isinstance(row.get("logit_churn_advantage_topk1_vs_topk2"), float)
        and row["logit_churn_advantage_topk1_vs_topk2"] >= 0.0
        for row in rows
    )
    transfer_not_worse_than_topk2 = all(
        isinstance(row.get("transfer_improvement_advantage_topk1_vs_topk2"), float)
        and row["transfer_improvement_advantage_topk1_vs_topk2"] >= 0.0
        for row in rows
    )
    beats_dense_control = all(
        isinstance(row.get("transfer_improvement_advantage_topk1_vs_dense"), float)
        and row["transfer_improvement_advantage_topk1_vs_dense"] > 0.0
        for row in rows
    )
    singleton_gain_positive = all(
        isinstance(row.get("source_topk1_singleton_gain_mean"), float)
        and row["source_topk1_singleton_gain_mean"] > 0.0
        for row in rows
    )
    commutator_present = all(
        isinstance(row.get(field), float)
        for row in rows
        for field in (
            "topk1_commutator_anchor_logit_mse",
            "topk2_commutator_anchor_logit_mse",
            "dense_commutator_anchor_logit_mse",
            "topk1_commutator_transfer_logit_mse",
            "topk2_commutator_transfer_logit_mse",
            "dense_commutator_transfer_logit_mse",
        )
    )
    commutator_not_worse_than_topk2 = commutator_present and all(
        isinstance(row.get(field), float) and row[field] >= 0.0
        for row in rows
        for field in (
            "commutator_anchor_logit_mse_advantage_topk1_vs_topk2",
            "commutator_transfer_logit_mse_advantage_topk1_vs_topk2",
        )
    )
    claim_supported = all(
        (
            support_churn_cleaner,
            logit_churn_cleaner,
            transfer_not_worse_than_topk2,
            beats_dense_control,
            singleton_gain_positive,
            bool(aggregates["ce_guardrail_all_packets"]),
            commutator_present,
            commutator_not_worse_than_topk2,
        )
    )
    return {
        "support_identity_churn_cleaner_than_topk2": support_churn_cleaner,
        "functional_logit_churn_not_higher_than_topk2": logit_churn_cleaner,
        "transfer_improvement_not_worse_than_topk2": transfer_not_worse_than_topk2,
        "transfer_improvement_beats_dense_control": beats_dense_control,
        "singleton_gain_positive": singleton_gain_positive,
        "ce_guardrail_all_packets": bool(aggregates["ce_guardrail_all_packets"]),
        "finite_update_commutator_present": commutator_present,
        "finite_update_commutator_not_worse_than_topk2": commutator_not_worse_than_topk2,
        "claim_supported": claim_supported,
    }


def _blocked_claim_status(signals: dict[str, bool]) -> str:
    if not signals["singleton_gain_positive"]:
        return BLOCKED_BY_NEGATIVE_SINGLETON_GAIN
    if not signals["transfer_improvement_beats_dense_control"]:
        return BLOCKED_BY_CONTROL_MATCH_FAILURE
    return FUNCTIONAL_RETENTION_BRACKET_ONLY


def _variant_prefix(variant: str) -> str:
    if variant == "rank_matched_contextual_topk1":
        return "topk1"
    if variant == "promoted_contextual_topk2":
        return "topk2"
    if variant == "norm_matched_dense_active_rank":
        return "dense"
    raise ValueError(f"unknown variant: {variant}")


def _write_packet_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "packet",
        "probe_dir",
        "status",
        "decision",
        "required_variants_present",
        "source_topk1_singleton_gain_mean",
        "source_context_level_topk1_singleton_gain_mean",
    ]
    for prefix in ("topk1", "topk2", "dense"):
        fieldnames.extend(f"{prefix}_{field}" for field in _METRIC_FIELDS)
    fieldnames.extend(
        [
            "support_churn_advantage_topk1_vs_topk2",
            "logit_churn_advantage_topk1_vs_topk2",
            "residual_stream_churn_advantage_topk1_vs_topk2",
            "commutator_anchor_logit_mse_advantage_topk1_vs_topk2",
            "commutator_transfer_logit_mse_advantage_topk1_vs_topk2",
            "commutator_anchor_logit_mse_advantage_topk1_vs_dense",
            "commutator_transfer_logit_mse_advantage_topk1_vs_dense",
            "transfer_improvement_advantage_topk1_vs_topk2",
            "transfer_improvement_advantage_topk1_vs_dense",
        ]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    aggregates = summary["evidence"]["aggregates"]
    lines = [
        "# Active Top-k-1 Functional-Retention Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        "- Mean top-k-1 support churn: "
        f"`{aggregates['mean_topk1_anchor_support_churn_after_transfer']}`",
        "- Mean top-k-2 support churn: "
        f"`{aggregates['mean_topk2_anchor_support_churn_after_transfer']}`",
        "- Minimum support-churn advantage: "
        f"`{aggregates['min_support_churn_advantage_topk1_vs_topk2']}`",
        "- Minimum logit-churn advantage: "
        f"`{aggregates['min_logit_churn_advantage_topk1_vs_topk2']}`",
        "- Minimum commutator anchor-logit advantage: "
        f"`{aggregates['min_commutator_anchor_logit_mse_advantage_topk1_vs_topk2']}`",
        "- Minimum commutator transfer-logit advantage: "
        f"`{aggregates['min_commutator_transfer_logit_mse_advantage_topk1_vs_topk2']}`",
        "- Mean source singleton gain: "
        f"`{aggregates['mean_source_topk1_singleton_gain_mean']}`",
        "",
        "## Rationale",
        "",
        summary["rationale"],
        "",
        "## Evidence Separation",
        "",
        "- Support identity churn: exact support-set churn from the completed probe packets.",
        "- Functional/logit churn: anchor logit MSE drift after transfer.",
        "- Causal gain/regret: source singleton gain remains the causal-gain caveat.",
        f"- CE guardrail: positive anchor CE deterioration must stay within `{summary['ce_guardrail']}`.",
        "- Finite-update commutator: A-to-B versus B-to-A final-function logit MSE when present.",
        "",
        "## Next Step",
        "",
        summary["next_step"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(left: Any, right: Any) -> float | None:
    left_value = _float_or_none(left)
    right_value = _float_or_none(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _min_or_none(values: list[float]) -> float | None:
    return min(values) if values else None


def _ce_guardrail_passes(value: Any, threshold: float) -> bool:
    number = _float_or_none(value)
    return number is not None and number <= threshold


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-dir",
        type=Path,
        action="append",
        dest="probe_dirs",
        help="Completed active top-k-1 retention/churn probe directory.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ce-guardrail", type=float, default=0.05)
    args = parser.parse_args(argv)
    probe_dirs = tuple(args.probe_dirs) if args.probe_dirs else DEFAULT_PROBE_DIRS
    summary = run_active_topk1_functional_retention_audit(
        probe_dirs=probe_dirs,
        out_dir=args.out,
        ce_guardrail=args.ce_guardrail,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "claim_status": summary["claim_status"],
                "aggregates": summary["evidence"]["aggregates"],
                "claim_signals": summary["evidence"]["claim_signals"],
                "failures": summary["failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
