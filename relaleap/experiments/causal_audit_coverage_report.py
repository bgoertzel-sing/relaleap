"""Source-artifact coverage report for the next causal audit."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path
from typing import Any


DEFAULT_AUDIT_DIR = Path(
    "results/audits/token_larger_support_wide_promoted_default_causal_column_fingerprint_stability_topk1"
)
DEFAULT_MATCHED_STRATA_DIR = Path(
    "results/audits/token_larger_rank_matched_topk1_vs_topk2_matched_strata_intervention"
)
DEFAULT_CAUSAL_FINGERPRINT_REPORT = Path(
    "results/reports/token_larger_causal_column_fingerprint_stability_topk1_audit/decision_report.json"
)
DEFAULT_BRACKET_REPORT = Path(
    "results/reports/token_larger_causal_audit_bracket_decision/decision_report.json"
)
DEFAULT_RANK_MATCHED_REPORT = Path(
    "results/reports/token_larger_rank_matched_topk1_causal_bracket_audit/decision_report.json"
)
DEFAULT_RETENTION_REPORT = Path(
    "results/reports/token_larger_retention_churn_microtest_decision/decision_report.json"
)
DEFAULT_OUT_DIR = Path("results/reports/token_larger_causal_audit_coverage")

EXISTING_ARTIFACTS_SUFFICIENT = "existing_artifacts_sufficient_for_next_no_training_audit"
SPECIFIC_MISSING_FIELDS = "specific_missing_fields_require_artifact_extension"
NEW_TRAINING_REQUIRED = "new_training_required_for_deconfounded_causal_matrix"

TOPK2_VARIANT = "baseline"
TOPK1_VARIANT = "rank_matched_topk1_contextual"


def write_causal_audit_coverage_report(
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    matched_strata_dir: Path = DEFAULT_MATCHED_STRATA_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    causal_fingerprint_report_path: Path = DEFAULT_CAUSAL_FINGERPRINT_REPORT,
    bracket_report_path: Path = DEFAULT_BRACKET_REPORT,
    rank_matched_report_path: Path = DEFAULT_RANK_MATCHED_REPORT,
    retention_report_path: Path = DEFAULT_RETENTION_REPORT,
) -> dict[str, Any]:
    """Write a decision-bearing ledger for the next no-training causal audit."""

    start = time.time()
    source_artifacts = [
        _audit_summary_entry(audit_dir),
        _csv_entry(audit_dir / "pair_interventions.csv", "pair_interventions"),
        _csv_entry(
            audit_dir / "per_token_pair_interventions.csv",
            "per_token_pair_interventions",
        ),
        _csv_entry(audit_dir / "column_fingerprints.csv", "column_fingerprints"),
        _matched_strata_entry(matched_strata_dir),
        _report_entry(causal_fingerprint_report_path, "causal_fingerprint_report"),
        _report_entry(bracket_report_path, "causal_audit_bracket_decision"),
        _report_entry(rank_matched_report_path, "rank_matched_topk1_bracket_report"),
        _report_entry(retention_report_path, "retention_churn_microtest_report"),
    ]

    audit_summary = _read_json_if_present(audit_dir / "summary.json")
    matched_summary = _read_json_if_present(matched_strata_dir / "summary.json")
    fingerprint_report = _read_json_if_present(causal_fingerprint_report_path)
    bracket_report = _read_json_if_present(bracket_report_path)
    rank_report = _read_json_if_present(rank_matched_report_path)
    retention_report = _read_json_if_present(retention_report_path)
    pair_fields = _csv_fieldnames(audit_dir / "pair_interventions.csv")
    per_token_pair_fields = _csv_fieldnames(
        audit_dir / "per_token_pair_interventions.csv"
    )
    column_fields = _csv_fieldnames(audit_dir / "column_fingerprints.csv")
    matched_fields = _csv_fieldnames(matched_strata_dir / "matched_strata.csv")

    coverage = _coverage_summary(
        audit_summary,
        matched_summary,
        fingerprint_report,
        bracket_report,
        rank_report,
        retention_report,
        pair_fields,
        per_token_pair_fields,
        column_fields,
        matched_fields,
    )
    decision, status, next_step = _coverage_decision(coverage)
    report = {
        "status": status,
        "decision": decision,
        "audit_dir": str(audit_dir),
        "matched_strata_dir": str(matched_strata_dir),
        "out_dir": str(out_dir),
        "platform": platform.platform(),
        "runtime_seconds": round(time.time() - start, 4),
        "source_artifacts": source_artifacts,
        "coverage": coverage,
        "next_no_training_causal_audit_matrix": _next_matrix(coverage),
        "next_step": next_step,
        "artifacts": {
            "decision_report_json": str(out_dir / "decision_report.json"),
            "decision_report_md": str(out_dir / "decision_report.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(out_dir / "decision_report.md", report)
    return report


def _coverage_summary(
    audit_summary: dict[str, Any],
    matched_summary: dict[str, Any],
    fingerprint_report: dict[str, Any],
    bracket_report: dict[str, Any],
    rank_report: dict[str, Any],
    retention_report: dict[str, Any],
    pair_fields: list[str],
    per_token_pair_fields: list[str],
    column_fields: list[str],
    matched_fields: list[str],
) -> dict[str, Any]:
    audit = audit_summary.get("audit", {}) if isinstance(audit_summary, dict) else {}
    variants = {
        str(row.get("variant")): row
        for row in audit.get("variants", [])
        if isinstance(row, dict) and row.get("variant")
    }
    signals = (fingerprint_report.get("evidence", {}) or {}).get("signals", {})
    matched_evidence = matched_summary.get("evidence", {})
    rank_signals = (rank_report.get("evidence", {}) or {}).get("signals", {})
    retention_signals = (retention_report.get("evidence", {}) or {}).get("signals", {})
    pair_field_set = set(pair_fields)
    per_token_pair_field_set = set(per_token_pair_fields)
    column_field_set = set(column_fields)
    matched_field_set = set(matched_fields)
    intervention_field_set = pair_field_set | per_token_pair_field_set
    matching_fields = {
        "position_bin": "position_bin" in intervention_field_set
        and "position_bin" in matched_field_set,
        "token_class": "token_class" in intervention_field_set
        and "token_class" in matched_field_set,
        "support_frequency": "router_support_count" in intervention_field_set
        or "router_support_fraction" in column_field_set,
        "residual_norm_or_gain": bool(
            {
                "fixed_support_residual_stream_l2_delta",
                "force_residual_stream_l2_delta",
                "ablate_residual_stream_l2_delta",
            }
            & (intervention_field_set | column_field_set)
        ),
        "residual_norm_bin": bool(
            {"residual_norm", "residual_norm_bin", "residual_gain_bin"}
            & intervention_field_set
        ),
        "active_rank_proxy": bool(
            {"active_rank", "active_rank_proxy", "stored_parameter_count"}
            & intervention_field_set
        ),
        "per_token_rows": bool(
            {"token_index", "position_index", "example_index", "batch_index"}
            & intervention_field_set
        ),
    }
    controls = {
        "promoted_topk2": TOPK2_VARIANT in variants,
        "rank_matched_topk1": TOPK1_VARIANT in variants
        or bool(rank_signals.get("rank_matched_topk1_present")),
        "random_support": bool(signals.get("random_fixed_topk2_worse_than_learned")),
        "norm_matched_dense": bool(signals.get("norm_matched_dense_worse_than_learned")),
        "retention_dense_control": bool(
            retention_signals.get("sparse_beats_dense_after_transfer")
        ),
    }
    required_missing_fields = [
        name
        for name in (
            "per_token_rows",
            "residual_norm_bin",
            "active_rank_proxy",
        )
        if not matching_fields[name]
    ]
    required_missing_controls = [
        name
        for name in (
            "promoted_topk2",
            "rank_matched_topk1",
            "random_support",
            "norm_matched_dense",
        )
        if not controls[name]
    ]
    return {
        "variants_present": sorted(variants),
        "variant_count": len(variants),
        "intervention_rows_present": bool(pair_fields),
        "per_token_intervention_rows_present": bool(per_token_pair_fields),
        "matched_strata_count": matched_evidence.get("matched_strata_count"),
        "matched_strata": matched_evidence.get("matched_strata", []),
        "row_granularity": "per_token"
        if matching_fields["per_token_rows"]
        else "aggregate",
        "matching_fields": matching_fields,
        "support_churn_fields_present": bool(audit.get("functional_churn")),
        "functional_churn_fields": sorted(
            {
                key
                for row in audit.get("functional_churn", [])
                if isinstance(row, dict)
                for key in row
            }
        ),
        "controls_available": controls,
        "topk2_ce_deficit_vs_rank_matched_topk1": _ce_deficit(variants),
        "topk2_pair_synergy_mean_across_strata": matched_evidence.get(
            "topk2_pair_synergy_mean_across_strata"
        ),
        "topk2_functional_churn_cleaner_than_topk1": matched_evidence.get(
            "topk2_functional_churn_cleaner_than_topk1"
        ),
        "rank_matched_topk1_router_ce_better_than_topk2": matched_evidence.get(
            "rank_matched_topk1_router_ce_better_than_topk2"
        ),
        "missing_fields_for_deconfounded_no_training_audit": required_missing_fields,
        "missing_controls_for_deconfounded_matrix": required_missing_controls,
    }


def _coverage_decision(coverage: dict[str, Any]) -> tuple[str, str, str]:
    if coverage["missing_controls_for_deconfounded_matrix"]:
        return (
            NEW_TRAINING_REQUIRED,
            "pass",
            "run a focused causal matrix that includes the missing controls before interpreting top-k-2 synergy",
        )
    if coverage["missing_fields_for_deconfounded_no_training_audit"]:
        return (
            SPECIFIC_MISSING_FIELDS,
            "pass",
            "extend the existing causal-fingerprint artifact schema with per-token residual-norm/active-rank matching fields, then rerun the no-training deconfounding ledger",
        )
    return (
        EXISTING_ARTIFACTS_SUFFICIENT,
        "pass",
        "run the no-training residual-norm/active-rank/support-stratum matched top-k-2 versus rank-matched top-k-1 causal audit",
    )


def _next_matrix(coverage: dict[str, Any]) -> dict[str, Any]:
    return {
        "brackets": [
            {
                "name": "promoted_contextual_topk2",
                "variant": TOPK2_VARIANT,
                "role": "test whether pair synergy survives CE deficit after deconfounding",
            },
            {
                "name": "rank_matched_contextual_topk1",
                "variant": TOPK1_VARIANT,
                "role": "CE-clean causal-audit bracket",
            },
        ],
        "match_or_bin_by": [
            "position_bin",
            "token_class",
            "support_frequency_or_dominance",
            "residual_norm_or_gain_bin",
            "active_rank_proxy",
        ],
        "metrics": [
            "alpha0_ce_loss",
            "pair_synergy_or_singleton_gain",
            "fixed_support_loss_delta",
            "support_identity_churn",
            "changed_support_logit_mse",
        ],
        "ce_guardrail": {
            "current_topk2_deficit": coverage["topk2_ce_deficit_vs_rank_matched_topk1"],
            "interpretation": "CE is a guardrail; do not promote top-k-2 cooperation from coarse positive synergy alone.",
        },
    }


def _audit_summary_entry(audit_dir: Path) -> dict[str, Any]:
    summary = _read_json_if_present(audit_dir / "summary.json")
    audit = summary.get("audit", {}) if isinstance(summary, dict) else {}
    return {
        "name": "causal_fingerprint_summary",
        "path": str(audit_dir / "summary.json"),
        "present": bool(summary),
        "variants_present": [
            row.get("variant") for row in audit.get("variants", []) if isinstance(row, dict)
        ],
        "intervention_rows_present": bool(audit.get("pair_intervention_count")),
        "per_token_intervention_rows_present": bool(
            audit.get("per_token_pair_intervention_count")
        ),
        "strata_coverage": "see pair_interventions.csv",
        "row_granularity": "per_token"
        if audit.get("per_token_pair_intervention_count")
        else "aggregate",
        "support_churn_fields_present": bool(audit.get("functional_churn")),
        "functional_churn_fields_present": bool(audit.get("functional_churn")),
    }


def _matched_strata_entry(matched_dir: Path) -> dict[str, Any]:
    summary = _read_json_if_present(matched_dir / "summary.json")
    evidence = summary.get("evidence", {}) if isinstance(summary, dict) else {}
    return {
        "name": "matched_strata_audit",
        "path": str(matched_dir / "summary.json"),
        "present": bool(summary),
        "variants_present": [TOPK2_VARIANT, TOPK1_VARIANT] if summary else [],
        "intervention_rows_present": bool(evidence.get("matched_strata_count")),
        "strata_coverage": evidence.get("matched_strata", []),
        "row_granularity": "aggregate",
        "support_churn_fields_present": "topk2_changed_support_logit_mse" in evidence,
        "functional_churn_fields_present": "topk2_changed_support_logit_mse" in evidence,
    }


def _csv_entry(path: Path, name: str) -> dict[str, Any]:
    fieldnames = _csv_fieldnames(path)
    field_set = set(fieldnames)
    return {
        "name": name,
        "path": str(path),
        "present": bool(fieldnames),
        "variants_present": _csv_unique(path, "variant"),
        "intervention_rows_present": "intervention" in field_set,
        "strata_coverage": {
            "position_bin": _csv_unique(path, "position_bin"),
            "token_class": _csv_unique(path, "token_class"),
        },
        "row_granularity": "per_token"
        if {"token_index", "position_index", "example_index", "batch_index"} & field_set
        else "aggregate",
        "residual_norm_or_gain_fields_present": bool(
            {
                "fixed_support_residual_stream_l2_delta",
                "force_residual_stream_l2_delta",
                "ablate_residual_stream_l2_delta",
            }
            & field_set
        ),
        "active_rank_or_stored_parameter_fields_present": bool(
            {"active_rank", "active_rank_proxy", "stored_parameter_count"} & field_set
        ),
        "support_churn_fields_present": bool(
            {"router_support_count", "router_support_fraction"} & field_set
        ),
        "functional_churn_fields_present": bool(
            {"fixed_support_logit_mse", "ablate_logit_mse", "force_logit_mse"} & field_set
        ),
    }


def _report_entry(path: Path, name: str) -> dict[str, Any]:
    report = _read_json_if_present(path)
    evidence = report.get("evidence", {}) if isinstance(report, dict) else {}
    signals = evidence.get("signals", {}) if isinstance(evidence, dict) else {}
    return {
        "name": name,
        "path": str(path),
        "present": bool(report),
        "status": report.get("status") if isinstance(report, dict) else None,
        "decision": report.get("decision") if isinstance(report, dict) else None,
        "variants_present": [],
        "intervention_rows_present": bool(
            signals.get("rank_matched_topk1_intervention_present")
            or signals.get("exact_pair_synergy_supported")
        ),
        "strata_coverage": "decision report aggregate",
        "row_granularity": "aggregate",
        "random_support_control_available": bool(
            signals.get("random_fixed_topk2_worse_than_learned")
        ),
        "dense_control_available": bool(
            signals.get("norm_matched_dense_worse_than_learned")
            or signals.get("sparse_beats_dense_after_transfer")
        ),
        "rank_matched_control_available": bool(
            signals.get("rank_matched_topk1_present")
            or signals.get("rank_matched_topk1_fingerprint_present")
        ),
    }


def _ce_deficit(variants: dict[str, dict[str, Any]]) -> float | None:
    topk2 = _float_or_none((variants.get(TOPK2_VARIANT) or {}).get("alpha0_ce_loss"))
    topk1 = _float_or_none((variants.get(TOPK1_VARIANT) or {}).get("alpha0_ce_loss"))
    if topk2 is None or topk1 is None:
        return None
    return topk2 - topk1


def _csv_fieldnames(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def _csv_unique(path: Path, field: str) -> list[str]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        if field not in (rows.fieldnames or []):
            return []
        return sorted({row.get(field, "") for row in rows if row.get(field, "")})


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    coverage = report["coverage"]
    lines = [
        "# Causal Audit Coverage Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        f"- Source artifacts: `{len(report['source_artifacts'])}`",
        f"- Row granularity: `{coverage['row_granularity']}`",
        f"- Matched strata count: `{coverage['matched_strata_count']}`",
        f"- Top-k-2 CE deficit vs rank-matched top-k-1: `{coverage['topk2_ce_deficit_vs_rank_matched_topk1']}`",
        f"- Missing fields: `{coverage['missing_fields_for_deconfounded_no_training_audit']}`",
        f"- Missing controls: `{coverage['missing_controls_for_deconfounded_matrix']}`",
        "",
        "## Controls",
        "",
    ]
    for key, value in coverage["controls_available"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Matching Fields", ""])
    for key, value in coverage["matching_fields"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Next Matrix", ""])
    matrix = report["next_no_training_causal_audit_matrix"]
    lines.append(
        "- Brackets: `"
        + ", ".join(row["name"] for row in matrix["brackets"])
        + "`"
    )
    lines.append("- Match/bin by: `" + ", ".join(matrix["match_or_bin_by"]) + "`")
    lines.extend(["", "## Next Step", "", report["next_step"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument(
        "--matched-strata-dir", type=Path, default=DEFAULT_MATCHED_STRATA_DIR
    )
    parser.add_argument(
        "--causal-fingerprint-report",
        type=Path,
        default=DEFAULT_CAUSAL_FINGERPRINT_REPORT,
    )
    parser.add_argument("--bracket-report", type=Path, default=DEFAULT_BRACKET_REPORT)
    parser.add_argument(
        "--rank-matched-report", type=Path, default=DEFAULT_RANK_MATCHED_REPORT
    )
    parser.add_argument(
        "--retention-report", type=Path, default=DEFAULT_RETENTION_REPORT
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    report = write_causal_audit_coverage_report(
        args.audit_dir,
        args.matched_strata_dir,
        args.out,
        causal_fingerprint_report_path=args.causal_fingerprint_report,
        bracket_report_path=args.bracket_report,
        rank_matched_report_path=args.rank_matched_report,
        retention_report_path=args.retention_report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
