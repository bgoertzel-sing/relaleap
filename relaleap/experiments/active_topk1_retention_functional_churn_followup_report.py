"""Selected retention/functional-churn follow-up for active top-k-1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from relaleap.experiments.active_topk1_next_evidence_selection_report import (
    NEXT_EVIDENCE_SELECTED,
    SELECTED_RETENTION_CHURN,
)
from relaleap.experiments.active_topk1_retention_churn_probe import (
    ACTIVE_TOPK1_RETENTION_CHURN_PROBE_ESTABLISHED,
)


DEFAULT_SELECTION_DIR = Path(
    "results/reports/token_larger_active_topk1_next_evidence_selection"
)
DEFAULT_PROBE_DIRS = (
    Path("results/audits/token_larger_active_rank_matched_topk1_retention_churn_probe"),
    Path(
        "results/audits/token_larger_active_rank_matched_topk1_retention_churn_probe_seed2"
    ),
)
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_active_topk1_retention_functional_churn_followup"
)

RETENTION_FUNCTIONAL_CHURN_BRACKET_SUPPORTED = (
    "retention_functional_churn_bracket_supported"
)
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
REQUIRED_VARIANTS = (
    "promoted_contextual_topk2",
    "rank_matched_contextual_topk1",
    "random_fixed_topk2",
    "norm_matched_dense_active_rank",
)
METRIC_FIELDS = (
    "anchor_ce_drift",
    "anchor_logit_mse_drift",
    "anchor_residual_stream_l2_drift",
    "anchor_support_churn_after_transfer",
    "transfer_ce_improvement",
    "commutator_anchor_logit_mse",
    "commutator_transfer_logit_mse",
    "commutator_anchor_residual_stream_l2",
    "commutator_transfer_residual_stream_l2",
)


def run_active_topk1_retention_functional_churn_followup_report(
    *,
    selection_dir: Path = DEFAULT_SELECTION_DIR,
    probe_dirs: tuple[Path, ...] = DEFAULT_PROBE_DIRS,
    out_dir: Path = DEFAULT_OUT_DIR,
    ce_guardrail: float = 0.05,
) -> dict[str, Any]:
    """Package the selected retention/churn branch across all four controls."""

    selection = _read_json_object(selection_dir / "summary.json")
    rows = [_packet_row(index, path) for index, path in enumerate(probe_dirs, 1)]
    failures = _selection_failures(selection_dir, selection)
    failures.extend(failure for row in rows for failure in _packet_failures(row))
    aggregates = _aggregates(rows, ce_guardrail=ce_guardrail)
    signals = _signals(rows, aggregates)

    if not failures and signals["branch_supported"]:
        status = "pass"
        decision = RETENTION_FUNCTIONAL_CHURN_BRACKET_SUPPORTED
        claim_status = "local_retention_functional_churn_bracket_only"
        rationale = (
            "The selected retention/functional-churn branch is supported locally "
            "across the refreshed probe packets. Rank-matched contextual top-k-1 "
            "has much lower support-identity churn than promoted top-k-2, no "
            "higher functional/logit churn, lower finite-update commutator risk, "
            "and better transfer CE improvement than promoted top-k-2, dense "
            "active-rank, and random fixed top-k-2 controls. This remains a "
            "retention/churn bracket, not a renewed top-k-2 causal-cooperation "
            "claim."
        )
        next_step = (
            "run a backend-stable RunPod repeat of the retention/functional-churn "
            "follow-up only if the next causal-retention claim needs GPU parity; "
            "otherwise use this local bracket to design the next discriminative "
            "causal-retention audit"
        )
    else:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        claim_status = INSUFFICIENT_EVIDENCE
        rationale = (
            "The selected retention/functional-churn branch could not be "
            "established because the branch selection, refreshed probe packets, "
            "or required four-control metrics are missing or unfavorable."
        )
        next_step = (
            "repair or rerun the active top-k-1 retention/churn probe packets "
            "with promoted top-k-2, rank-matched top-k-1, random fixed top-k-2, "
            "and dense active-rank controls before interpreting this branch"
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selection_dir": str(selection_dir),
        "probe_dirs": [str(path) for path in probe_dirs],
        "out_dir": str(out_dir),
        "ce_guardrail": ce_guardrail,
        "required_variants": list(REQUIRED_VARIANTS),
        "rows": rows,
        "aggregates": aggregates,
        "signals": signals,
        "failures": failures,
        "rationale": rationale,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "packet_variant_metrics_csv": str(out_dir / "packet_variant_metrics.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_rows(out_dir / "packet_variant_metrics.csv", rows)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _selection_failures(selection_dir: Path, selection: dict[str, Any]) -> list[dict[str, Any]]:
    path = selection_dir / "summary.json"
    failures = []
    if not path.is_file():
        return [
            {
                "source": "selection",
                "field": "summary_json",
                "expected": "file exists",
                "actual": "missing",
                "path": str(path),
            }
        ]
    if selection.get("status") != "pass":
        failures.append(
            {
                "source": "selection",
                "field": "status",
                "expected": "pass",
                "actual": selection.get("status"),
            }
        )
    if selection.get("decision") != NEXT_EVIDENCE_SELECTED:
        failures.append(
            {
                "source": "selection",
                "field": "decision",
                "expected": NEXT_EVIDENCE_SELECTED,
                "actual": selection.get("decision"),
            }
        )
    if selection.get("selected_experiment") != SELECTED_RETENTION_CHURN:
        failures.append(
            {
                "source": "selection",
                "field": "selected_experiment",
                "expected": SELECTED_RETENTION_CHURN,
                "actual": selection.get("selected_experiment"),
            }
        )
    return failures


def _packet_row(index: int, probe_dir: Path) -> dict[str, Any]:
    summary_path = probe_dir / "summary.json"
    summary = _read_json_object(summary_path)
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
        "config_path": summary.get("config_path"),
    }
    for variant in REQUIRED_VARIANTS:
        source = variants.get(variant, {})
        prefix = _prefix(variant)
        row[f"{prefix}_present"] = bool(source)
        for field in METRIC_FIELDS:
            row[f"{prefix}_{field}"] = _float_or_none(source.get(field))
    row["support_churn_advantage_topk1_vs_topk2"] = _delta(
        row["topk2_anchor_support_churn_after_transfer"],
        row["topk1_anchor_support_churn_after_transfer"],
    )
    row["logit_churn_advantage_topk1_vs_topk2"] = _delta(
        row["topk2_anchor_logit_mse_drift"],
        row["topk1_anchor_logit_mse_drift"],
    )
    row["transfer_advantage_topk1_vs_topk2"] = _delta(
        row["topk1_transfer_ce_improvement"],
        row["topk2_transfer_ce_improvement"],
    )
    row["transfer_advantage_topk1_vs_dense"] = _delta(
        row["topk1_transfer_ce_improvement"],
        row["dense_transfer_ce_improvement"],
    )
    row["transfer_advantage_topk1_vs_random_fixed_topk2"] = _delta(
        row["topk1_transfer_ce_improvement"],
        row["random_fixed_topk2_transfer_ce_improvement"],
    )
    row["commutator_anchor_advantage_topk1_vs_topk2"] = _delta(
        row["topk2_commutator_anchor_logit_mse"],
        row["topk1_commutator_anchor_logit_mse"],
    )
    row["commutator_anchor_advantage_topk1_vs_dense"] = _delta(
        row["dense_commutator_anchor_logit_mse"],
        row["topk1_commutator_anchor_logit_mse"],
    )
    row["commutator_anchor_advantage_topk1_vs_random_fixed_topk2"] = _delta(
        row["random_fixed_topk2_commutator_anchor_logit_mse"],
        row["topk1_commutator_anchor_logit_mse"],
    )
    return row


def _packet_failures(row: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
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
    for variant in REQUIRED_VARIANTS:
        prefix = _prefix(variant)
        if not row[f"{prefix}_present"]:
            failures.append(
                {
                    "packet": row["packet"],
                    "field": f"{variant}.present",
                    "expected": True,
                    "actual": False,
                }
            )
    for field in (
        "topk1_anchor_support_churn_after_transfer",
        "topk2_anchor_support_churn_after_transfer",
        "random_fixed_topk2_anchor_support_churn_after_transfer",
        "topk1_anchor_logit_mse_drift",
        "topk2_anchor_logit_mse_drift",
        "topk1_transfer_ce_improvement",
        "topk2_transfer_ce_improvement",
        "random_fixed_topk2_transfer_ce_improvement",
        "dense_transfer_ce_improvement",
        "topk1_commutator_anchor_logit_mse",
        "topk2_commutator_anchor_logit_mse",
        "random_fixed_topk2_commutator_anchor_logit_mse",
        "dense_commutator_anchor_logit_mse",
    ):
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


def _aggregates(rows: list[dict[str, Any]], *, ce_guardrail: float) -> dict[str, Any]:
    fields = (
        "topk1_anchor_support_churn_after_transfer",
        "topk2_anchor_support_churn_after_transfer",
        "random_fixed_topk2_anchor_support_churn_after_transfer",
        "topk1_anchor_logit_mse_drift",
        "topk2_anchor_logit_mse_drift",
        "topk1_transfer_ce_improvement",
        "topk2_transfer_ce_improvement",
        "random_fixed_topk2_transfer_ce_improvement",
        "dense_transfer_ce_improvement",
        "topk1_commutator_anchor_logit_mse",
        "topk2_commutator_anchor_logit_mse",
        "random_fixed_topk2_commutator_anchor_logit_mse",
        "dense_commutator_anchor_logit_mse",
        "support_churn_advantage_topk1_vs_topk2",
        "logit_churn_advantage_topk1_vs_topk2",
        "transfer_advantage_topk1_vs_topk2",
        "transfer_advantage_topk1_vs_dense",
        "transfer_advantage_topk1_vs_random_fixed_topk2",
        "commutator_anchor_advantage_topk1_vs_topk2",
        "commutator_anchor_advantage_topk1_vs_dense",
        "commutator_anchor_advantage_topk1_vs_random_fixed_topk2",
    )
    aggregates = {f"mean_{field}": _mean_or_none(_values(rows, field)) for field in fields}
    for field in fields:
        aggregates[f"min_{field}"] = _min_or_none(_values(rows, field))
    aggregates["ce_guardrail_all_packets"] = all(
        _ce_guardrail_passes(row.get(field), ce_guardrail)
        for row in rows
        for field in (
            "topk1_anchor_ce_drift",
            "topk2_anchor_ce_drift",
            "dense_anchor_ce_drift",
        )
    )
    return aggregates


def _signals(rows: list[dict[str, Any]], aggregates: dict[str, Any]) -> dict[str, bool]:
    required_present = all(
        bool(row.get(f"{_prefix(variant)}_present"))
        for row in rows
        for variant in REQUIRED_VARIANTS
    )
    topk1_support_cleaner = all(
        _positive(row.get("support_churn_advantage_topk1_vs_topk2")) for row in rows
    )
    topk1_logit_not_higher = all(
        _nonnegative(row.get("logit_churn_advantage_topk1_vs_topk2")) for row in rows
    )
    topk1_transfer_not_worse = all(
        _nonnegative(row.get("transfer_advantage_topk1_vs_topk2")) for row in rows
    )
    topk1_transfer_beats_controls = all(
        _positive(row.get("transfer_advantage_topk1_vs_dense"))
        and _positive(row.get("transfer_advantage_topk1_vs_random_fixed_topk2"))
        for row in rows
    )
    topk1_commutator_cleaner = all(
        _positive(row.get("commutator_anchor_advantage_topk1_vs_topk2"))
        and _positive(row.get("commutator_anchor_advantage_topk1_vs_dense"))
        and _positive(row.get("commutator_anchor_advantage_topk1_vs_random_fixed_topk2"))
        for row in rows
    )
    return {
        "required_four_controls_present": required_present,
        "topk1_support_identity_churn_cleaner_than_topk2": topk1_support_cleaner,
        "topk1_functional_logit_churn_not_higher_than_topk2": topk1_logit_not_higher,
        "topk1_transfer_not_worse_than_topk2": topk1_transfer_not_worse,
        "topk1_transfer_beats_dense_and_random_controls": topk1_transfer_beats_controls,
        "topk1_finite_update_commutator_cleaner_than_controls": topk1_commutator_cleaner,
        "ce_guardrail_all_packets": bool(aggregates["ce_guardrail_all_packets"]),
        "branch_supported": all(
            (
                required_present,
                topk1_support_cleaner,
                topk1_logit_not_higher,
                topk1_transfer_not_worse,
                topk1_transfer_beats_controls,
                topk1_commutator_cleaner,
                bool(aggregates["ce_guardrail_all_packets"]),
            )
        ),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "packet",
        "probe_dir",
        "status",
        "decision",
        "config_path",
    ]
    for variant in REQUIRED_VARIANTS:
        prefix = _prefix(variant)
        fieldnames.append(f"{prefix}_present")
        fieldnames.extend(f"{prefix}_{field}" for field in METRIC_FIELDS)
    fieldnames.extend(
        [
            "support_churn_advantage_topk1_vs_topk2",
            "logit_churn_advantage_topk1_vs_topk2",
            "transfer_advantage_topk1_vs_topk2",
            "transfer_advantage_topk1_vs_dense",
            "transfer_advantage_topk1_vs_random_fixed_topk2",
            "commutator_anchor_advantage_topk1_vs_topk2",
            "commutator_anchor_advantage_topk1_vs_dense",
            "commutator_anchor_advantage_topk1_vs_random_fixed_topk2",
        ]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    aggregates = summary["aggregates"]
    lines = [
        "# Active Top-k-1 Retention/Functional-Churn Follow-up",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        "- Mean top-k-1 support churn: "
        f"`{aggregates['mean_topk1_anchor_support_churn_after_transfer']}`",
        "- Mean top-k-2 support churn: "
        f"`{aggregates['mean_topk2_anchor_support_churn_after_transfer']}`",
        "- Mean random fixed top-k-2 support churn: "
        f"`{aggregates['mean_random_fixed_topk2_anchor_support_churn_after_transfer']}`",
        "- Mean top-k-1 transfer advantage vs top-k-2: "
        f"`{aggregates['mean_transfer_advantage_topk1_vs_topk2']}`",
        "- Mean top-k-1 transfer advantage vs dense: "
        f"`{aggregates['mean_transfer_advantage_topk1_vs_dense']}`",
        "- Mean top-k-1 transfer advantage vs random fixed top-k-2: "
        f"`{aggregates['mean_transfer_advantage_topk1_vs_random_fixed_topk2']}`",
        "- Minimum top-k-1 commutator advantage vs controls: "
        f"`{min(value for value in (aggregates['min_commutator_anchor_advantage_topk1_vs_topk2'], aggregates['min_commutator_anchor_advantage_topk1_vs_dense'], aggregates['min_commutator_anchor_advantage_topk1_vs_random_fixed_topk2']) if value is not None)}`",
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


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _prefix(variant: str) -> str:
    if variant == "rank_matched_contextual_topk1":
        return "topk1"
    if variant == "promoted_contextual_topk2":
        return "topk2"
    if variant == "random_fixed_topk2":
        return "random_fixed_topk2"
    if variant == "norm_matched_dense_active_rank":
        return "dense"
    raise ValueError(f"unknown variant: {variant}")


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


def _values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [value for value in (row.get(field) for row in rows) if isinstance(value, float)]


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _min_or_none(values: list[float]) -> float | None:
    return min(values) if values else None


def _ce_guardrail_passes(value: Any, threshold: float) -> bool:
    number = _float_or_none(value)
    return number is not None and number <= threshold


def _positive(value: Any) -> bool:
    number = _float_or_none(value)
    return number is not None and number > 0.0


def _nonnegative(value: Any) -> bool:
    number = _float_or_none(value)
    return number is not None and number >= 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR)
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
    summary = run_active_topk1_retention_functional_churn_followup_report(
        selection_dir=args.selection_dir,
        probe_dirs=probe_dirs,
        out_dir=args.out,
        ce_guardrail=args.ce_guardrail,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "signals": summary["signals"],
                "failures": summary["failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
