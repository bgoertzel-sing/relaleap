"""Close out the promoted top-k-2 dead-column load-balance branch."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_PROBE_DIRS = (
    Path("results/audits/token_larger_support_wide_promoted_default_dead_column_probe_low_weight_bracket"),
    Path("results/audits/token_larger_support_wide_promoted_default_dead_column_probe_low_weight_bracket_seed2"),
)
DEFAULT_CAUSAL_DIRS = (
    Path("results/audits/token_larger_support_wide_promoted_default_causal_column_fingerprint_low_weight_bracket"),
    Path("results/audits/token_larger_support_wide_promoted_default_causal_column_fingerprint_low_weight_bracket_seed2"),
)
DEFAULT_LOAD_BALANCE_REPORT_DIR = Path("results/reports/dead_column_load_balance_probe")
DEFAULT_SUPPORT_QUALITY_DIR = Path(
    "results/reports/token_larger_promoted_topk2_support_selection_quality_audit"
)
DEFAULT_GATE_AUDIT_DIR = Path(
    "results/audits/token_larger_active_topk1_context_gate_suppression_calibration"
)
DEFAULT_RETENTION_REFERENCE_DIR = Path(
    "results/reports/token_larger_promoted_topk2_retention_reference_audit"
)
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_promoted_topk2_load_balance_closeout"
)

KEEP_LOAD_BALANCE_OPT_IN_CLOSED = "keep_load_balance_opt_in_branch_closed"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def run_promoted_topk2_load_balance_closeout_report(
    *,
    probe_dirs: tuple[Path, ...] = DEFAULT_PROBE_DIRS,
    causal_dirs: tuple[Path, ...] = DEFAULT_CAUSAL_DIRS,
    load_balance_report_dir: Path = DEFAULT_LOAD_BALANCE_REPORT_DIR,
    support_quality_dir: Path = DEFAULT_SUPPORT_QUALITY_DIR,
    gate_audit_dir: Path = DEFAULT_GATE_AUDIT_DIR,
    retention_reference_dir: Path = DEFAULT_RETENTION_REFERENCE_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Synthesize existing no-training load-balance evidence without retraining."""

    failures: list[dict[str, Any]] = []
    probe_entries = [_probe_entry(path, failures) for path in probe_dirs]
    causal_entries = [_causal_entry(path, failures) for path in causal_dirs]
    load_balance_report = _read_json_object(
        load_balance_report_dir / "decision_report.json"
    )
    support_quality = _read_json_object(support_quality_dir / "summary.json")
    gate_audit = _read_json_object(gate_audit_dir / "summary.json")
    retention_reference = _read_json_object(retention_reference_dir / "summary.json")

    source_rows = [
        *[
            _source_row(f"dead_column_probe_seed{index + 1}", path / "summary.json")
            for index, path in enumerate(probe_dirs)
        ],
        *[
            _source_row(f"load_balance_causal_fingerprint_seed{index + 1}", path / "summary.json")
            for index, path in enumerate(causal_dirs)
        ],
        _source_row(
            "prior_load_balance_decision",
            load_balance_report_dir / "decision_report.json",
        ),
        _source_row("promoted_topk2_support_quality", support_quality_dir / "summary.json"),
        _source_row("topk1_context_gate", gate_audit_dir / "summary.json"),
        _source_row(
            "promoted_topk2_retention_reference",
            retention_reference_dir / "summary.json",
        ),
    ]

    metrics = _metrics(probe_entries, causal_entries)
    signals = _signals(
        load_balance_report=load_balance_report,
        support_quality=support_quality,
        gate_audit=gate_audit,
        retention_reference=retention_reference,
        probe_entries=probe_entries,
        causal_entries=causal_entries,
    )
    _extend_context_failures(
        failures,
        load_balance_report=load_balance_report,
        support_quality=support_quality,
        gate_audit=gate_audit,
        retention_reference=retention_reference,
    )

    status = "fail" if failures or not all(signals.values()) else "pass"
    decision = KEEP_LOAD_BALANCE_OPT_IN_CLOSED if status == "pass" else INSUFFICIENT_EVIDENCE
    rationale = (
        "The seed-1/seed-2 low-weight load-balance probes consistently recruit "
        "dead or underused columns within the configured CE tolerance, but the "
        "matched causal fingerprints do not make the recruited columns cleaner "
        "or less disruptive under direct interventions. Combined with the newer "
        "top-k-2 support-selection quality packet and the failed deployable "
        "top-k-1 gate, load balancing remains an opt-in utilization diagnostic "
        "rather than a support-router default change."
        if status == "pass"
        else "The load-balance closeout cannot be interpreted because required source packets are missing, failing, or inconsistent."
    )
    next_step = (
        "run a local no-training promoted top-k-2 support-retention gap selector to choose between functional-churn controls, residual-sum normalization controls, or a bounded backend repeat"
        if status == "pass"
        else "repair the missing or inconsistent load-balance closeout source packets"
    )

    summary = {
        "status": status,
        "decision": decision,
        "out_dir": str(out_dir),
        "source_rows": source_rows,
        "metrics": metrics,
        "signals": signals,
        "failures": failures,
        "promote_router_load_balance_default": False,
        "rationale": rationale,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_source_rows(out_dir / "source_rows.csv", source_rows)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _probe_entry(path: Path, failures: list[dict[str, Any]]) -> dict[str, Any]:
    summary_path = path / "summary.json"
    value = _read_json_object(summary_path)
    probe = value.get("probe") if isinstance(value.get("probe"), dict) else {}
    decision = probe.get("decision") if isinstance(probe.get("decision"), dict) else {}
    baseline = _find_variant(probe, probe.get("baseline_variant"))
    selected = _find_variant(probe, decision.get("selected_variant"))
    entry = {
        "path": str(summary_path),
        "status": value.get("status"),
        "decision_status": decision.get("status"),
        "config_path": value.get("config_path"),
        "baseline_variant": probe.get("baseline_variant"),
        "selected_variant": decision.get("selected_variant"),
        "baseline_alpha0_ce_loss": _float_or_none(decision.get("baseline_alpha0_ce_loss")),
        "selected_alpha0_ce_loss": _float_or_none(decision.get("selected_alpha0_ce_loss")),
        "baseline_used_columns": _int_or_none(decision.get("baseline_used_columns")),
        "selected_used_columns": _int_or_none(decision.get("selected_used_columns")),
        "selected_dead_columns": _int_or_none(selected.get("dead_columns")),
        "baseline_oracle_support_regret": _float_or_none(baseline.get("oracle_support_regret")),
        "selected_oracle_support_regret": _float_or_none(selected.get("oracle_support_regret")),
        "selected_load_balance_weight": _float_or_none(selected.get("load_balance_weight")),
    }
    if not summary_path.is_file():
        failures.append(_failure("probe.summary_json", "file exists", "missing", summary_path))
    if entry["status"] != "ok":
        failures.append(_failure("probe.status", "ok", entry["status"], summary_path))
    if entry["decision_status"] != "recruited_without_ce_hurt":
        failures.append(
            _failure(
                "probe.decision.status",
                "recruited_without_ce_hurt",
                entry["decision_status"],
                summary_path,
            )
        )
    if entry["selected_used_columns"] is None or entry["baseline_used_columns"] is None:
        failures.append(_failure("probe.used_columns", "baseline and selected counts", None, summary_path))
    elif entry["selected_used_columns"] <= entry["baseline_used_columns"]:
        failures.append(
            _failure(
                "probe.used_column_gain",
                "selected_used_columns > baseline_used_columns",
                entry["selected_used_columns"] - entry["baseline_used_columns"],
                summary_path,
            )
        )
    return entry


def _causal_entry(path: Path, failures: list[dict[str, Any]]) -> dict[str, Any]:
    summary_path = path / "summary.json"
    value = _read_json_object(summary_path)
    audit = value.get("audit") if isinstance(value.get("audit"), dict) else {}
    variants = (
        audit.get("variant_summaries")
        if isinstance(audit.get("variant_summaries"), list)
        else audit.get("variants")
        if isinstance(audit.get("variants"), list)
        else []
    )
    baseline = _variant_summary(variants, "baseline")
    selected = next(
        (
            row
            for row in variants
            if isinstance(row, dict)
            and row.get("variant") != "baseline"
            and (
                row.get("selected_for_load_balance_bracket")
                or _float_or_none(row.get("load_balance_weight")) not in (None, 0.0)
            )
        ),
        {},
    )
    entry = {
        "path": str(summary_path),
        "status": value.get("status"),
        "baseline_variant": baseline.get("variant"),
        "selected_variant": selected.get("variant"),
        "baseline_mean_abs_ablate_loss_delta": _float_or_none(
            baseline.get("mean_abs_ablate_loss_delta")
        ),
        "selected_mean_abs_ablate_loss_delta": _float_or_none(
            selected.get("mean_abs_ablate_loss_delta")
        ),
        "baseline_mean_abs_force_loss_delta": _float_or_none(
            baseline.get("mean_abs_force_loss_delta")
        ),
        "selected_mean_abs_force_loss_delta": _float_or_none(
            selected.get("mean_abs_force_loss_delta")
        ),
    }
    if not summary_path.is_file():
        failures.append(_failure("causal.summary_json", "file exists", "missing", summary_path))
    if entry["status"] != "ok":
        failures.append(_failure("causal.status", "ok", entry["status"], summary_path))
    if not entry["selected_variant"]:
        failures.append(
            _failure(
                "causal.selected_variant",
                "selected load-balance bracket variant",
                None,
                summary_path,
            )
        )
    return entry


def _metrics(
    probe_entries: list[dict[str, Any]],
    causal_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    used_gains = [
        entry["selected_used_columns"] - entry["baseline_used_columns"]
        for entry in probe_entries
        if entry.get("selected_used_columns") is not None
        and entry.get("baseline_used_columns") is not None
    ]
    ce_deltas = [
        entry["selected_alpha0_ce_loss"] - entry["baseline_alpha0_ce_loss"]
        for entry in probe_entries
        if entry.get("selected_alpha0_ce_loss") is not None
        and entry.get("baseline_alpha0_ce_loss") is not None
    ]
    ablate_gains = [
        entry["selected_mean_abs_ablate_loss_delta"]
        - entry["baseline_mean_abs_ablate_loss_delta"]
        for entry in causal_entries
        if entry.get("selected_mean_abs_ablate_loss_delta") is not None
        and entry.get("baseline_mean_abs_ablate_loss_delta") is not None
    ]
    force_gains = [
        entry["selected_mean_abs_force_loss_delta"]
        - entry["baseline_mean_abs_force_loss_delta"]
        for entry in causal_entries
        if entry.get("selected_mean_abs_force_loss_delta") is not None
        and entry.get("baseline_mean_abs_force_loss_delta") is not None
    ]
    return {
        "probe_count": len(probe_entries),
        "causal_fingerprint_count": len(causal_entries),
        "min_used_column_gain": min(used_gains) if used_gains else None,
        "max_used_column_gain": max(used_gains) if used_gains else None,
        "min_alpha0_ce_delta": min(ce_deltas) if ce_deltas else None,
        "max_alpha0_ce_delta": max(ce_deltas) if ce_deltas else None,
        "min_mean_abs_ablate_delta_gain": min(ablate_gains) if ablate_gains else None,
        "max_mean_abs_ablate_delta_gain": max(ablate_gains) if ablate_gains else None,
        "min_mean_abs_force_delta_gain": min(force_gains) if force_gains else None,
        "max_mean_abs_force_delta_gain": max(force_gains) if force_gains else None,
        "probe_entries": probe_entries,
        "causal_fingerprint_entries": causal_entries,
    }


def _signals(
    *,
    load_balance_report: dict[str, Any],
    support_quality: dict[str, Any],
    gate_audit: dict[str, Any],
    retention_reference: dict[str, Any],
    probe_entries: list[dict[str, Any]],
    causal_entries: list[dict[str, Any]],
) -> dict[str, bool]:
    return {
        "all_probes_recruited_without_ce_hurt": all(
            entry.get("decision_status") == "recruited_without_ce_hurt"
            for entry in probe_entries
        ),
        "all_causal_fingerprints_present": all(
            entry.get("status") == "ok" and entry.get("selected_variant")
            for entry in causal_entries
        ),
        "prior_report_keeps_load_balance_opt_in": load_balance_report.get("decision")
        == "keep_router_load_balance_probe_opt_in",
        "prior_report_blocks_default_promotion": load_balance_report.get(
            "promote_router_load_balance_default"
        )
        is False,
        "topk2_support_selection_quality_established": support_quality.get("decision")
        == "promoted_topk2_support_selection_quality_established",
        "topk1_deployable_gate_failed": gate_audit.get("decision")
        == "deployable_context_gate_suppression_calibration_failed",
        "topk2_retention_reference_established": retention_reference.get("decision")
        == "promoted_topk2_router_default_retention_reference",
    }


def _extend_context_failures(
    failures: list[dict[str, Any]],
    *,
    load_balance_report: dict[str, Any],
    support_quality: dict[str, Any],
    gate_audit: dict[str, Any],
    retention_reference: dict[str, Any],
) -> None:
    expected = (
        (
            "prior_load_balance_decision.decision",
            "keep_router_load_balance_probe_opt_in",
            load_balance_report.get("decision"),
        ),
        (
            "prior_load_balance_decision.promote_router_load_balance_default",
            False,
            load_balance_report.get("promote_router_load_balance_default"),
        ),
        (
            "support_quality.decision",
            "promoted_topk2_support_selection_quality_established",
            support_quality.get("decision"),
        ),
        (
            "topk1_gate.decision",
            "deployable_context_gate_suppression_calibration_failed",
            gate_audit.get("decision"),
        ),
        (
            "retention_reference.decision",
            "promoted_topk2_router_default_retention_reference",
            retention_reference.get("decision"),
        ),
    )
    for field, wanted, actual in expected:
        if actual != wanted:
            failures.append({"field": field, "expected": wanted, "actual": actual})


def _find_variant(probe: dict[str, Any], name: Any) -> dict[str, Any]:
    variants = probe.get("variants") if isinstance(probe.get("variants"), list) else []
    return _variant_summary(variants, name)


def _variant_summary(variants: list[Any], name: Any) -> dict[str, Any]:
    for row in variants:
        if isinstance(row, dict) and row.get("variant") == name:
            return row
    return {}


def _source_row(source: str, path: Path) -> dict[str, Any]:
    value = _read_json_object(path)
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": value.get("status"),
        "decision": value.get("decision"),
    }


def _write_source_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["source", "path", "present", "status", "decision"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    lines = [
        "# Promoted Top-k-2 Load-Balance Closeout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Probe count: `{metrics['probe_count']}`",
        f"- Causal fingerprint count: `{metrics['causal_fingerprint_count']}`",
        f"- Used-column gain range: `{metrics['min_used_column_gain']}` to `{metrics['max_used_column_gain']}`",
        f"- Alpha-0 CE delta range: `{metrics['min_alpha0_ce_delta']}` to `{metrics['max_alpha0_ce_delta']}`",
        f"- Mean absolute ablation-delta gain range: `{metrics['min_mean_abs_ablate_delta_gain']}` to `{metrics['max_mean_abs_ablate_delta_gain']}`",
        f"- Mean absolute force-delta gain range: `{metrics['min_mean_abs_force_delta_gain']}` to `{metrics['max_mean_abs_force_delta_gain']}`",
        f"- Promote router load-balance default: `{summary['promote_router_load_balance_default']}`",
        "",
        "## Signals",
        "",
    ]
    for key, value in summary["signals"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Rationale", "", summary["rationale"], "", "## Next Step", "", summary["next_step"], ""])
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


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _failure(field: str, expected: Any, actual: Any, path: Path) -> dict[str, Any]:
    return {
        "field": field,
        "expected": expected,
        "actual": actual,
        "path": str(path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_promoted_topk2_load_balance_closeout_report(out_dir=args.out)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "metrics": {
                    key: value
                    for key, value in summary["metrics"].items()
                    if not key.endswith("_entries")
                },
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
