"""Non-claim diagnostic for support-frequency candidate-control blockers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_AUDIT_DIR = Path(
    "results/audits/"
    "token_larger_support_wide_promoted_default_causal_column_fingerprint_"
    "support_frequency_candidates"
)
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_support_frequency_blocker_diagnostic"
)
DISTANCE_FIELDS = (
    "support_count_abs_difference",
    "fixed_support_loss_abs_difference",
    "pair_gain_abs_difference",
    "singleton_gain_sum_abs_difference",
    "pair_value_norm_abs_difference",
    "pair_synergy_abs_difference",
)
RELAXED_SUPPORT_COUNT_CALIPERS = (1, 2, 4, 8, 16, 32)


def run_support_frequency_blocker_diagnostic(
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    relaxed_support_count_calipers: tuple[int, ...] = RELAXED_SUPPORT_COUNT_CALIPERS,
) -> dict[str, Any]:
    """Summarize why support-frequency candidate controls lack claim matches."""

    start = time.time()
    candidate_path = audit_dir / "support_frequency_candidate_controls.csv"
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, str]] = []
    if not candidate_path.is_file():
        failures.append(
            {
                "field": "support_frequency_candidate_controls_csv",
                "expected": str(candidate_path),
            }
        )
    else:
        with candidate_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    required_fields = {
        "variant",
        "anchor_index",
        "anchor_support",
        "candidate_support",
        "support_count_caliper",
        "within_support_count_caliper",
        "candidate_match_status",
        "support_count_abs_difference",
        "fixed_support_loss_difference",
        "pair_gain_difference",
        "singleton_gain_sum_difference",
        "pair_value_norm_difference",
        "pair_synergy_difference",
    }
    if rows:
        missing_fields = sorted(required_fields - set(rows[0]))
        if missing_fields:
            failures.append(
                {
                    "field": "support_frequency_candidate_control_fields",
                    "expected": sorted(required_fields),
                    "actual_missing": missing_fields,
                }
            )

    status = "fail" if failures else "pass"
    evidence: dict[str, Any] = {
        "failures": failures,
        "source_audit_dir": str(audit_dir),
        "candidate_controls_csv": str(candidate_path),
        "claim_bearing": False,
        "diagnostic_role": "non_claim_support_frequency_blocker_diagnostic",
        "unmatched_policy": "exclude_from_primary_percentile_denominator_no_loose_fallback",
    }
    anchor_rows: list[dict[str, Any]] = []
    if status == "pass":
        enriched_rows = [_enrich_row(row) for row in rows]
        anchor_rows = _anchor_diagnostics(enriched_rows, relaxed_support_count_calipers)
        calipered_count = sum(
            1 for row in enriched_rows if bool(row["within_support_count_caliper"])
        )
        unmatched_count = len(enriched_rows) - calipered_count
        failed_counts = {
            "support_count_caliper": sum(
                1 for row in enriched_rows if not row["within_support_count_caliper"]
            )
        }
        status_counts = Counter(
            str(row["candidate_match_status"]) for row in enriched_rows
        )
        primary_caliper = _mode_int(
            row["support_count_caliper"] for row in enriched_rows
        )
        evidence.update(
            {
                "candidate_row_count": len(enriched_rows),
                "anchor_count": len(
                    {
                        (str(row["variant"]), int(row["anchor_index"]))
                        for row in enriched_rows
                    }
                ),
                "support_count_caliper": primary_caliper,
                "calipered_candidate_row_count": calipered_count,
                "unmatched_candidate_row_count": unmatched_count,
                "failed_caliper_dimension_counts": failed_counts,
                "candidate_match_status_counts": dict(sorted(status_counts.items())),
                "nearest_neighbor_distance_summary": {
                    field: _summary_stats(
                        [
                            float(row[field])
                            for row in enriched_rows
                            if row[field] is not None
                        ]
                    )
                    for field in DISTANCE_FIELDS
                },
                "per_anchor_nearest_neighbor_summary": {
                    field: _summary_stats(
                        [
                            float(anchor[f"nearest_{field}"])
                            for anchor in anchor_rows
                            if anchor.get(f"nearest_{field}") is not None
                        ]
                    )
                    for field in DISTANCE_FIELDS
                },
                "relaxed_support_count_caliper_diagnostics": _relaxed_calipers(
                    enriched_rows, relaxed_support_count_calipers
                ),
            }
        )

    if status == "fail":
        decision = "insufficient_artifacts_for_support_frequency_blocker_diagnostic"
    elif evidence["calipered_candidate_row_count"] == 0:
        decision = "support_frequency_percentile_claim_remains_blocked_by_support_count_caliper"
    else:
        decision = "diagnostic_only_calipered_candidates_exist_no_claim_change"

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": status,
        "decision": decision,
        "audit_dir": str(audit_dir),
        "out_dir": str(out_dir),
        "evidence": evidence,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "per_anchor_blockers_csv": str(out_dir / "per_anchor_blockers.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "elapsed_seconds": time.time() - start,
    }
    _write_json(out_dir / "summary.json", summary)
    _write_csv(out_dir / "per_anchor_blockers.csv", _ANCHOR_FIELDNAMES, anchor_rows)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _enrich_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        **row,
        "anchor_index": int(row["anchor_index"]),
        "support_count_caliper": _optional_int(row.get("support_count_caliper")),
        "within_support_count_caliper": _parse_bool(
            row.get("within_support_count_caliper")
        ),
        "support_count_abs_difference": _optional_float(
            row.get("support_count_abs_difference")
        ),
        "fixed_support_loss_abs_difference": abs(
            _optional_float(row.get("fixed_support_loss_difference")) or 0.0
        ),
        "pair_gain_abs_difference": abs(
            _optional_float(row.get("pair_gain_difference")) or 0.0
        ),
        "singleton_gain_sum_abs_difference": abs(
            _optional_float(row.get("singleton_gain_sum_difference")) or 0.0
        ),
        "pair_value_norm_abs_difference": abs(
            _optional_float(row.get("pair_value_norm_difference")) or 0.0
        ),
        "pair_synergy_abs_difference": abs(
            _optional_float(row.get("pair_synergy_difference")) or 0.0
        ),
    }


def _anchor_diagnostics(
    rows: list[dict[str, Any]],
    relaxed_support_count_calipers: tuple[int, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["variant"]), int(row["anchor_index"]), str(row["anchor_support"]))
        ].append(row)

    diagnostics: list[dict[str, Any]] = []
    for (variant, anchor_index, anchor_support), group in sorted(grouped.items()):
        calipered_count = sum(1 for row in group if row["within_support_count_caliper"])
        out: dict[str, Any] = {
            "variant": variant,
            "anchor_index": anchor_index,
            "anchor_support": anchor_support,
            "candidate_row_count": len(group),
            "calipered_candidate_count": calipered_count,
            "unmatched_candidate_count": len(group) - calipered_count,
            "support_count_caliper": _mode_int(
                row["support_count_caliper"] for row in group
            ),
            "failed_support_count_caliper_count": sum(
                1 for row in group if not row["within_support_count_caliper"]
            ),
            "candidate_match_status_counts": json.dumps(
                dict(
                    sorted(
                        Counter(
                            str(row["candidate_match_status"]) for row in group
                        ).items()
                    )
                ),
                sort_keys=True,
            ),
        }
        for field in DISTANCE_FIELDS:
            values = [float(row[field]) for row in group if row[field] is not None]
            out[f"nearest_{field}"] = min(values) if values else None
        for caliper in relaxed_support_count_calipers:
            out[f"relaxed_caliper_{caliper}_candidate_count"] = sum(
                1
                for row in group
                if row["support_count_abs_difference"] is not None
                and row["support_count_abs_difference"] <= caliper
            )
        diagnostics.append(out)
    return diagnostics


def _relaxed_calipers(
    rows: list[dict[str, Any]], relaxed_support_count_calipers: tuple[int, ...]
) -> list[dict[str, Any]]:
    out = []
    for caliper in relaxed_support_count_calipers:
        count = sum(
            1
            for row in rows
            if row["support_count_abs_difference"] is not None
            and row["support_count_abs_difference"] <= caliper
        )
        out.append(
            {
                "support_count_caliper": caliper,
                "candidate_row_count": count,
                "candidate_fraction": count / len(rows) if rows else None,
            }
        )
    return out


def _summary_stats(values: list[float]) -> dict[str, Any]:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(clean),
        "min": clean[0],
        "p25": _quantile(clean, 0.25),
        "median": _quantile(clean, 0.5),
        "p75": _quantile(clean, 0.75),
        "max": clean[-1],
        "mean": sum(clean) / len(clean),
    }


def _quantile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _mode_int(values: Any) -> int | None:
    counts = Counter(value for value in values if value is not None)
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    evidence = summary["evidence"]
    lines = [
        "# Support-Frequency Blocker Diagnostic",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim-bearing: `{evidence['claim_bearing']}`",
        f"- Source audit: `{summary['audit_dir']}`",
    ]
    if summary["status"] == "pass":
        lines.extend(
            [
                f"- Candidate rows: `{evidence['candidate_row_count']}`",
                f"- Anchors: `{evidence['anchor_count']}`",
                f"- Calipered candidate rows: `{evidence['calipered_candidate_row_count']}`",
                f"- Unmatched candidate rows: `{evidence['unmatched_candidate_row_count']}`",
                "- Failed caliper dimension counts: "
                f"`{evidence['failed_caliper_dimension_counts']}`",
                "- Nearest support-count distance summary: "
                f"`{evidence['per_anchor_nearest_neighbor_summary']['support_count_abs_difference']}`",
                "",
                "This diagnostic is intentionally non-claim-bearing. Relaxed-caliper "
                "counts are exploratory and do not enter the support-frequency "
                "candidate-percentile denominator.",
            ]
        )
    else:
        for failure in evidence.get("failures", []):
            lines.append(f"- Missing `{failure['field']}`: expected `{failure.get('expected')}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


_ANCHOR_FIELDNAMES = [
    "variant",
    "anchor_index",
    "anchor_support",
    "candidate_row_count",
    "calipered_candidate_count",
    "unmatched_candidate_count",
    "support_count_caliper",
    "failed_support_count_caliper_count",
    "candidate_match_status_counts",
    "nearest_support_count_abs_difference",
    "nearest_fixed_support_loss_abs_difference",
    "nearest_pair_gain_abs_difference",
    "nearest_singleton_gain_sum_abs_difference",
    "nearest_pair_value_norm_abs_difference",
    "nearest_pair_synergy_abs_difference",
    "relaxed_caliper_1_candidate_count",
    "relaxed_caliper_2_candidate_count",
    "relaxed_caliper_4_candidate_count",
    "relaxed_caliper_8_candidate_count",
    "relaxed_caliper_16_candidate_count",
    "relaxed_caliper_32_candidate_count",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_support_frequency_blocker_diagnostic(args.audit_dir, args.out)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "evidence": summary["evidence"],
                "artifacts": summary["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
