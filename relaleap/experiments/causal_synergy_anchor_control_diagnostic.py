"""Per-anchor sampled-control diagnostic for causal pair synergy artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from relaleap.experiments.causal_synergy_null_audit import (
    _ci,
    _mean,
    _optional_float,
    _read_matched_keys,
    _support_count_bins,
)
from relaleap.experiments.deconfounded_intervention_audit import (
    DEFAULT_AUDIT_DIR,
    DEFAULT_OUT_DIR as DEFAULT_DECONFOUNDED_DIR,
    TOPK2_INTERVENTION,
    TOPK2_VARIANT,
)


DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_topk2_causal_synergy_anchor_control_diagnostic"
)
DEFAULT_CONTROL_INTERVENTIONS = (
    "fixed_support_frequency_matched_control",
    "fixed_random_nonrouter_control",
    "fixed_loss_matched_nonrouter_control",
    "fixed_best_support_swap",
)
OUTCOME_PROXIMAL_CONTROLS = {"fixed_loss_matched_nonrouter_control"}


def run_causal_synergy_anchor_control_diagnostic(
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    deconfounded_dir: Path = DEFAULT_DECONFOUNDED_DIR,
    variant: str = TOPK2_VARIANT,
    observed_intervention: str = TOPK2_INTERVENTION,
    control_interventions: tuple[str, ...] = DEFAULT_CONTROL_INTERVENTIONS,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Compare each observed anchor against available sampled nonrouter controls."""

    start = time.time()
    failures: list[dict[str, Any]] = []
    per_token_path = audit_dir / "per_token_pair_interventions.csv"
    matched_path = deconfounded_dir / "matched_deconfounded_strata.csv"
    for field, path in (
        ("per_token_pair_interventions_csv", per_token_path),
        ("matched_deconfounded_strata_csv", matched_path),
    ):
        if not path.is_file():
            failures.append({"field": field, "expected": str(path)})

    rows: list[dict[str, str]] = []
    matched_keys: set[tuple[str, str, str, str, str]] = set()
    if not failures:
        with per_token_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        matched_keys = _read_matched_keys(matched_path)

    required_fields = {
        "variant",
        "intervention",
        "support",
        "anchor_support",
        "control_support",
        "anchor_router_support_count",
        "control_router_support_count",
        "support_count_difference",
        "fixed_support_loss_difference",
        "control_match_status",
        "token_index",
        "position_bin",
        "token_class",
        "residual_norm_bin",
        "residual_gain_bin",
        "router_support_count",
        "pair_synergy",
    }
    if rows:
        missing_fields = sorted(required_fields - set(rows[0]))
        if missing_fields:
            failures.append(
                {
                    "field": "per_token_anchor_control_fields",
                    "expected": sorted(required_fields),
                    "actual_missing": missing_fields,
                }
            )

    anchor_rows: list[dict[str, Any]] = []
    control_summaries: dict[str, Any] = {}
    status = "fail" if failures else "pass"
    decision = "insufficient_evidence"

    evidence: dict[str, Any] = {
        "failures": failures,
        "source_audit_dir": str(audit_dir),
        "deconfounded_dir": str(deconfounded_dir),
        "variant": variant,
        "observed_intervention": observed_intervention,
        "control_interventions": list(control_interventions),
        "ci_level": ci_level,
        "matched_dimensions": [
            "position_bin",
            "token_class",
            "residual_norm_bin",
            "residual_gain_bin",
            "support_count_bin",
        ],
        "control_notes": {
            name: (
                "outcome_proximal_loss_matched_secondary_bound"
                if name in OUTCOME_PROXIMAL_CONTROLS
                else "selection_control"
            )
            for name in control_interventions
        },
    }

    if not failures:
        support_count_bins = _support_count_bins(rows)
        observed_by_anchor_token = _observed_anchor_rows(
            rows,
            variant=variant,
            intervention=observed_intervention,
            support_count_bins=support_count_bins,
            matched_keys=matched_keys,
        )
        anchor_rows = _paired_anchor_rows(
            rows,
            observed_by_anchor_token=observed_by_anchor_token,
            variant=variant,
            control_interventions=control_interventions,
        )
        control_summaries = {
            intervention: _control_summary(
                [
                    row
                    for row in anchor_rows
                    if row["control_intervention"] == intervention
                ],
                ci_level=ci_level,
            )
            for intervention in control_interventions
        }
        evidence.update(
            {
                "observed_anchor_token_count": len(observed_by_anchor_token),
                "paired_control_token_count": len(anchor_rows),
                "anchor_count": len({row["anchor_support"] for row in anchor_rows}),
                "control_summaries": control_summaries,
            }
        )
        decision = _decision(control_summaries)

    if failures:
        evidence["failures"] = failures

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "per_anchor_control_deltas.csv", _FIELDNAMES, anchor_rows)
    summary = {
        "status": status,
        "decision": decision,
        "audit_dir": str(audit_dir),
        "deconfounded_dir": str(deconfounded_dir),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "evidence": evidence,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "per_anchor_control_deltas_csv": str(
                out_dir / "per_anchor_control_deltas.csv"
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


def _observed_anchor_rows(
    rows: list[dict[str, str]],
    *,
    variant: str,
    intervention: str,
    support_count_bins: dict[int, str],
    matched_keys: set[tuple[str, str, str, str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    observed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("variant") != variant or row.get("intervention") != intervention:
            continue
        support_count = int(_optional_float(row.get("router_support_count")) or 0)
        key = (
            str(row.get("position_bin")),
            str(row.get("token_class")),
            str(row.get("residual_norm_bin")),
            str(row.get("residual_gain_bin")),
            support_count_bins.get(support_count, "unknown"),
        )
        if matched_keys and key not in matched_keys:
            continue
        observed[(str(row.get("support")), str(row.get("token_index")))] = row
    return observed


def _paired_anchor_rows(
    rows: list[dict[str, str]],
    *,
    observed_by_anchor_token: dict[tuple[str, str], dict[str, str]],
    variant: str,
    control_interventions: tuple[str, ...],
) -> list[dict[str, Any]]:
    paired: list[dict[str, Any]] = []
    control_set = set(control_interventions)
    for control in rows:
        if control.get("variant") != variant:
            continue
        if control.get("intervention") not in control_set:
            continue
        anchor_support = str(control.get("anchor_support") or "")
        control_support = str(control.get("control_support") or control.get("support") or "")
        if not anchor_support or not control_support or anchor_support == control_support:
            continue
        observed = observed_by_anchor_token.get(
            (anchor_support, str(control.get("token_index")))
        )
        if observed is None:
            continue
        observed_synergy = _optional_float(observed.get("pair_synergy"))
        control_synergy = _optional_float(control.get("pair_synergy"))
        if observed_synergy is None or control_synergy is None:
            continue
        paired.append(
            {
                "anchor_support": anchor_support,
                "control_intervention": control.get("intervention"),
                "control_support": control_support,
                "token_index": control.get("token_index"),
                "batch_index": control.get("batch_index"),
                "position_index": control.get("position_index"),
                "position_bin": observed.get("position_bin"),
                "token_class": observed.get("token_class"),
                "residual_norm_bin": observed.get("residual_norm_bin"),
                "residual_gain_bin": observed.get("residual_gain_bin"),
                "observed_pair_synergy": observed_synergy,
                "control_pair_synergy": control_synergy,
                "observed_minus_control_pair_synergy": observed_synergy
                - control_synergy,
                "observed_pair_gain": _optional_float(observed.get("pair_gain")),
                "control_pair_gain": _optional_float(control.get("pair_gain")),
                "anchor_router_support_count": _optional_float(
                    control.get("anchor_router_support_count")
                ),
                "control_router_support_count": _optional_float(
                    control.get("control_router_support_count")
                ),
                "support_count_difference": _optional_float(
                    control.get("support_count_difference")
                ),
                "fixed_support_loss_difference": _optional_float(
                    control.get("fixed_support_loss_difference")
                ),
                "control_match_rank": _optional_float(control.get("control_match_rank")),
                "control_match_status": control.get("control_match_status"),
            }
        )
    return paired


def _control_summary(rows: list[dict[str, Any]], *, ci_level: float) -> dict[str, Any]:
    deltas = [
        float(row["observed_minus_control_pair_synergy"])
        for row in rows
        if row.get("observed_minus_control_pair_synergy") is not None
    ]
    by_anchor: dict[str, list[float]] = defaultdict(list)
    controls_by_anchor: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        anchor = str(row["anchor_support"])
        by_anchor[anchor].append(float(row["observed_minus_control_pair_synergy"]))
        controls_by_anchor[anchor].add(str(row["control_support"]))
    anchor_means = [_mean(values) for values in by_anchor.values()]
    anchor_means_float = [float(value) for value in anchor_means if value is not None]
    ci = _ci(anchor_means_float, ci_level)
    return {
        "token_pair_count": len(deltas),
        "anchor_count": len(by_anchor),
        "control_support_count": len({str(row["control_support"]) for row in rows}),
        "control_supports_per_anchor_mean": _mean(
            [float(len(supports)) for supports in controls_by_anchor.values()]
        ),
        "observed_minus_control_token_mean": _mean(deltas),
        "observed_minus_control_anchor_mean": _mean(anchor_means_float),
        "observed_minus_control_anchor_ci": list(ci),
        "positive_anchor_fraction": _fraction(value > 0.0 for value in anchor_means_float),
        "support_count_difference_mean": _mean(
            _numeric_field_values(rows, "support_count_difference")
        ),
        "support_count_difference_abs_mean": _mean(
            [abs(value) for value in _numeric_field_values(rows, "support_count_difference")]
        ),
        "fixed_support_loss_difference_mean": _mean(
            _numeric_field_values(rows, "fixed_support_loss_difference")
        ),
        "fixed_support_loss_difference_abs_mean": _mean(
            [
                abs(value)
                for value in _numeric_field_values(rows, "fixed_support_loss_difference")
            ]
        ),
        "control_match_status_counts": _value_counts(
            row.get("control_match_status") for row in rows
        ),
        "supported": ci[0] is not None and ci[0] > 0.0,
        "control_role": (
            "outcome_proximal_loss_matched_secondary_bound"
            if rows
            and rows[0].get("control_intervention") in OUTCOME_PROXIMAL_CONTROLS
            else "selection_control"
        ),
    }


def _decision(control_summaries: dict[str, Any]) -> str:
    if not control_summaries:
        return "no_controls_available"
    selection_controls = {
        name: summary
        for name, summary in control_summaries.items()
        if name not in OUTCOME_PROXIMAL_CONTROLS
    }
    available_selection = {
        name: summary
        for name, summary in selection_controls.items()
        if summary.get("anchor_count", 0) > 0
    }
    if not available_selection:
        return "no_selection_controls_available"
    failed = [
        name
        for name, summary in available_selection.items()
        if not bool(summary.get("supported"))
    ]
    if failed:
        return "anchor_pair_synergy_not_supported_against_selection_controls"
    loss_summary = control_summaries.get("fixed_loss_matched_nonrouter_control")
    if loss_summary and loss_summary.get("anchor_count", 0) > 0 and not loss_summary.get(
        "supported"
    ):
        return "selection_controls_pass_but_loss_matched_bound_fails"
    return "anchor_pair_synergy_supported_against_available_controls"


def _numeric_field_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value is None or value == "":
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric) or math.isinf(numeric):
            continue
        values.append(numeric)
    return values


def _fraction(values: Any) -> float | None:
    materialized = list(values)
    if not materialized:
        return None
    return float(sum(1 for value in materialized if value) / len(materialized))


def _value_counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None or value == "":
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    evidence = summary["evidence"]
    lines = [
        "# Causal Synergy Anchor-Control Diagnostic",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Source audit: `{summary['audit_dir']}`",
        f"- Deconfounded audit: `{summary['deconfounded_dir']}`",
    ]
    if summary["status"] == "pass":
        lines.extend(
            [
                f"- Observed anchor-token rows: `{evidence['observed_anchor_token_count']}`",
                f"- Paired control-token rows: `{evidence['paired_control_token_count']}`",
                f"- Anchor count: `{evidence['anchor_count']}`",
            ]
        )
        for name, control in evidence["control_summaries"].items():
            lines.extend(
                [
                    f"- `{name}` anchor mean delta: `{control['observed_minus_control_anchor_mean']}`",
                    f"- `{name}` anchor CI: `{control['observed_minus_control_anchor_ci']}`",
                    f"- `{name}` supported: `{control['supported']}`",
                    f"- `{name}` control role: `{control['control_role']}`",
                ]
            )
    else:
        for failure in evidence.get("failures", []):
            lines.append(f"- Missing `{failure['field']}`: expected `{failure.get('expected')}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


_FIELDNAMES = [
    "anchor_support",
    "control_intervention",
    "control_support",
    "token_index",
    "batch_index",
    "position_index",
    "position_bin",
    "token_class",
    "residual_norm_bin",
    "residual_gain_bin",
    "observed_pair_synergy",
    "control_pair_synergy",
    "observed_minus_control_pair_synergy",
    "observed_pair_gain",
    "control_pair_gain",
    "anchor_router_support_count",
    "control_router_support_count",
    "support_count_difference",
    "fixed_support_loss_difference",
    "control_match_rank",
    "control_match_status",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--deconfounded-dir", type=Path, default=DEFAULT_DECONFOUNDED_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--variant", default=TOPK2_VARIANT)
    parser.add_argument("--observed-intervention", default=TOPK2_INTERVENTION)
    parser.add_argument(
        "--control-intervention",
        action="append",
        dest="control_interventions",
        help="Control intervention to include; repeat to include several.",
    )
    parser.add_argument("--ci-level", type=float, default=0.95)
    args = parser.parse_args(argv)
    summary = run_causal_synergy_anchor_control_diagnostic(
        args.audit_dir,
        args.out,
        deconfounded_dir=args.deconfounded_dir,
        variant=args.variant,
        observed_intervention=args.observed_intervention,
        control_interventions=tuple(
            args.control_interventions or DEFAULT_CONTROL_INTERVENTIONS
        ),
        ci_level=args.ci_level,
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
