"""Reconcile active top-k-1 singleton gain estimands under one sign convention."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


DEFAULT_SOURCE_AUDIT_DIR = Path(
    "results/audits/token_larger_support_wide_promoted_default_causal_column_fingerprint_stability_topk1"
)
DEFAULT_CONTROL_AUDIT_DIR = Path(
    "results/audits/token_larger_active_rank_matched_topk1_singleton_control_diagnostic"
)
DEFAULT_GAIN_REGRET_AUDIT_DIR = Path(
    "results/audits/token_larger_active_rank_matched_topk1_singleton_gain_regret_diagnostic"
)
DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_active_rank_matched_topk1_singleton_reconciliation_audit"
)

TOPK1_VARIANT = "rank_matched_topk1_contextual"
SELECTED_INTERVENTION = "fixed_dominant_router_singleton"
LOGGED_ORACLE_INTERVENTION = "fixed_best_singleton_swap"
RANDOM_INTERVENTION = "fixed_random_singleton_control"
CONTEXT_FIELDS = ("batch_index", "position_index", "token_index", "target_token")

CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE = (
    "context_gated_singleton_efficacy_with_offcontext_interference"
)
UNRESOLVED_SINGLETON_ESTIMAND_MISMATCH = "unresolved_singleton_estimand_mismatch"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def run_active_topk1_singleton_reconciliation_audit(
    *,
    source_audit_dir: Path = DEFAULT_SOURCE_AUDIT_DIR,
    control_audit_dir: Path = DEFAULT_CONTROL_AUDIT_DIR,
    gain_regret_audit_dir: Path = DEFAULT_GAIN_REGRET_AUDIT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a single selected/oracle/off-context singleton reconciliation packet."""

    start = time.time()
    failures = _source_failures(source_audit_dir, control_audit_dir, gain_regret_audit_dir)
    source_summary: dict[str, Any] = {}
    control_summary: dict[str, Any] = {}
    gain_regret_summary: dict[str, Any] = {}
    source_rows: list[dict[str, str]] = []
    if not failures:
        source_summary = _read_json_object(source_audit_dir / "summary.json")
        control_summary = _read_json_object(control_audit_dir / "summary.json")
        gain_regret_summary = _read_json_object(gain_regret_audit_dir / "summary.json")
        source_rows = _read_csv_rows(source_audit_dir / "per_token_pair_interventions.csv")

    topk1_rows = [row for row in source_rows if row.get("variant") == TOPK1_VARIANT]
    required_fields = {
        *CONTEXT_FIELDS,
        "variant",
        "intervention",
        "support",
        "router_support_matches_fixed",
        "empty_loss",
        "fixed_support_loss",
        "singleton_left_gain",
        "position_bin",
        "token_class",
    }
    if topk1_rows:
        missing = sorted(required_fields - set(topk1_rows[0]))
        if missing:
            failures.append(
                {
                    "field": "topk1_reconciliation_required_fields",
                    "expected": sorted(required_fields),
                    "actual_missing": missing,
                }
            )
    elif source_rows:
        failures.append(
            {
                "field": "topk1_source_rows",
                "expected": f"variant={TOPK1_VARIANT}",
                "actual": 0,
            }
        )

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        context_rows: list[dict[str, Any]] = []
        stratum_rows: list[dict[str, Any]] = []
        evidence = {"failures": failures}
        rationale = (
            "The singleton reconciliation audit could not run because required "
            "source artifacts, prior diagnostic summaries, or source fields are missing."
        )
        next_step = "repair the source singleton diagnostics before changing the causal-bracket label"
    else:
        context_rows = _context_rows(topk1_rows)
        stratum_rows = _stratum_rows(context_rows)
        evidence = _build_evidence(
            source_summary=source_summary,
            control_summary=control_summary,
            gain_regret_summary=gain_regret_summary,
            source_audit_dir=source_audit_dir,
            control_audit_dir=control_audit_dir,
            gain_regret_audit_dir=gain_regret_audit_dir,
            source_rows=topk1_rows,
            context_rows=context_rows,
        )
        decision = _decision(evidence)
        status = "pass"
        if decision == CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE:
            rationale = (
                "The same source artifact separates two estimands: in-context "
                "router-selected singleton supports are beneficial on average, while "
                "forced off-context dominant singleton rows are negative. This supports "
                "a context-gated singleton-efficacy interpretation with off-context "
                "interference, not a global singleton-value failure."
            )
            next_step = (
                "extend the source causal-column fingerprint artifact with random and "
                "exhaustive singleton rows on the same contexts before changing the "
                "top-k-1 bracket label or spending GPU validation cycles"
            )
        else:
            rationale = (
                "The selected, logged-oracle, and off-context singleton estimates do "
                "not yet form a clean reconciled pattern, or required random/exhaustive "
                "controls remain absent."
            )
            next_step = (
                "extend the source artifact with random and exhaustive singleton "
                "controls on the same context keys, then rerun this reconciliation audit"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "singleton_reconciliation_by_context.csv", _CONTEXT_FIELDS_OUT, context_rows)
    _write_csv(out_dir / "singleton_reconciliation_by_stratum.csv", _STRATUM_FIELDS_OUT, stratum_rows)
    summary = {
        "status": status,
        "decision": decision,
        "source_audit_dir": str(source_audit_dir),
        "control_audit_dir": str(control_audit_dir),
        "gain_regret_audit_dir": str(gain_regret_audit_dir),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "evidence": evidence,
        "rationale": rationale,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "singleton_reconciliation_by_context_csv": str(
                out_dir / "singleton_reconciliation_by_context.csv"
            ),
            "singleton_reconciliation_by_stratum_csv": str(
                out_dir / "singleton_reconciliation_by_stratum.csv"
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


def _source_failures(
    source_audit_dir: Path, control_audit_dir: Path, gain_regret_audit_dir: Path
) -> list[dict[str, Any]]:
    failures = []
    for field, path in (
        ("source_summary_json", source_audit_dir / "summary.json"),
        (
            "source_per_token_pair_interventions_csv",
            source_audit_dir / "per_token_pair_interventions.csv",
        ),
        ("control_summary_json", control_audit_dir / "summary.json"),
        ("gain_regret_summary_json", gain_regret_audit_dir / "summary.json"),
    ):
        if not path.is_file():
            failures.append(
                {"field": field, "expected": "file exists", "actual": "missing", "path": str(path)}
            )
    return failures


def _context_rows(source_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        grouped[_context_key(row)].append(row)

    rows = []
    for context, values in sorted(grouped.items()):
        selected = [
            row
            for row in values
            if row.get("intervention") == SELECTED_INTERVENTION
            and _bool_value(row.get("router_support_matches_fixed"))
        ]
        offcontext = [
            row
            for row in values
            if row.get("intervention") == SELECTED_INTERVENTION
            and not _bool_value(row.get("router_support_matches_fixed"))
        ]
        oracle = [row for row in values if row.get("intervention") == LOGGED_ORACLE_INTERVENTION]
        random_rows = [row for row in values if row.get("intervention") == RANDOM_INTERVENTION]
        first = values[0]
        selected_gains = _gains(selected)
        offcontext_gains = _gains(offcontext)
        oracle_gains = _gains(_best_loss_rows(selected + oracle))
        random_gains = _gains(random_rows)
        rows.append(
            {
                "batch_index": context[0],
                "position_index": context[1],
                "token_index": context[2],
                "target_token": context[3],
                "position_bin": first.get("position_bin", ""),
                "token_class": first.get("token_class", ""),
                "selected_row_count": len(selected),
                "logged_oracle_row_count": len(oracle),
                "offcontext_row_count": len(offcontext),
                "random_singleton_row_count": len(random_rows),
                "selected_singleton_gain_mean": _mean_or_none(selected_gains),
                "logged_oracle_singleton_gain_mean": _mean_or_none(oracle_gains),
                "offcontext_fixed_dominant_singleton_gain_mean": _mean_or_none(offcontext_gains),
                "random_singleton_gain_mean": _mean_or_none(random_gains),
                "exhaustive_singleton_gain_mean": None,
                "selected_positive": _positive_mean(selected_gains),
                "logged_oracle_positive": _positive_mean(oracle_gains),
                "offcontext_negative": _negative_mean(offcontext_gains),
                "selected_context": bool(selected),
                "offcontext_context": bool(offcontext),
                "selected_and_offcontext_context": bool(selected) and bool(offcontext),
            }
        )
    return rows


def _stratum_rows(context_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in context_rows:
        grouped[(str(row.get("position_bin", "")), str(row.get("token_class", "")))].append(row)
    rows = []
    for (position_bin, token_class), values in sorted(grouped.items()):
        rows.append(
            {
                "position_bin": position_bin,
                "token_class": token_class,
                "context_count": len(values),
                "selected_context_count": sum(1 for row in values if row.get("selected_context")),
                "offcontext_context_count": sum(1 for row in values if row.get("offcontext_context")),
                "selected_singleton_gain_mean": _mean_field(values, "selected_singleton_gain_mean"),
                "logged_oracle_singleton_gain_mean": _mean_field(
                    values, "logged_oracle_singleton_gain_mean"
                ),
                "offcontext_fixed_dominant_singleton_gain_mean": _mean_field(
                    values, "offcontext_fixed_dominant_singleton_gain_mean"
                ),
                "random_singleton_gain_mean": _mean_field(values, "random_singleton_gain_mean"),
            }
        )
    return rows


def _build_evidence(
    *,
    source_summary: dict[str, Any],
    control_summary: dict[str, Any],
    gain_regret_summary: dict[str, Any],
    source_audit_dir: Path,
    control_audit_dir: Path,
    gain_regret_audit_dir: Path,
    source_rows: list[dict[str, str]],
    context_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_context_rows = [row for row in context_rows if row.get("selected_context")]
    offcontext_context_rows = [row for row in context_rows if row.get("offcontext_context")]
    selected_keys = {_context_key(row) for row in selected_context_rows}
    offcontext_keys = {_context_key(row) for row in offcontext_context_rows}
    random_rows = [row for row in source_rows if row.get("intervention") == RANDOM_INTERVENTION]
    metrics = {
        "source_status": source_summary.get("status"),
        "source_decision": source_summary.get("decision"),
        "control_diagnostic_decision": control_summary.get("decision"),
        "gain_regret_diagnostic_decision": gain_regret_summary.get("decision"),
        "topk1_source_row_count": len(source_rows),
        "context_count": len(context_rows),
        "selected_context_count": len(selected_context_rows),
        "logged_oracle_context_count": sum(
            1 for row in context_rows if int(row.get("logged_oracle_row_count") or 0) > 0
        ),
        "offcontext_context_count": len(offcontext_context_rows),
        "selected_offcontext_context_overlap_count": len(selected_keys & offcontext_keys),
        "random_singleton_row_count": len(random_rows),
        "random_singleton_context_count": sum(
            1 for row in context_rows if int(row.get("random_singleton_row_count") or 0) > 0
        ),
        "exhaustive_singleton_context_count": 0,
        "selected_singleton_gain_mean": _mean_field(
            selected_context_rows, "selected_singleton_gain_mean"
        ),
        "logged_oracle_singleton_gain_mean": _mean_field(
            selected_context_rows, "logged_oracle_singleton_gain_mean"
        ),
        "offcontext_fixed_dominant_singleton_gain_mean": _mean_field(
            offcontext_context_rows, "offcontext_fixed_dominant_singleton_gain_mean"
        ),
        "random_singleton_gain_mean": _mean_field(context_rows, "random_singleton_gain_mean"),
        "selected_positive_fraction": _fraction(
            bool(row.get("selected_positive")) for row in selected_context_rows
        ),
        "logged_oracle_positive_fraction": _fraction(
            bool(row.get("logged_oracle_positive")) for row in selected_context_rows
        ),
        "offcontext_negative_fraction": _fraction(
            bool(row.get("offcontext_negative")) for row in offcontext_context_rows
        ),
    }
    signals = {
        "selected_incontext_positive": _gt(metrics["selected_singleton_gain_mean"], 0.0),
        "logged_oracle_incontext_positive": _gt(
            metrics["logged_oracle_singleton_gain_mean"], 0.0
        ),
        "offcontext_fixed_dominant_negative": _lt(
            metrics["offcontext_fixed_dominant_singleton_gain_mean"], 0.0
        ),
        "selected_positive_on_majority": _gt(metrics["selected_positive_fraction"], 0.5),
        "offcontext_negative_on_majority": _gt(metrics["offcontext_negative_fraction"], 0.5),
        "random_singleton_control_present": metrics["random_singleton_row_count"] > 0,
        "exhaustive_singleton_control_present": False,
    }
    return {
        "metrics": metrics,
        "signals": signals,
        "provenance": {
            "source_audit_dir": str(source_audit_dir),
            "control_audit_dir": str(control_audit_dir),
            "gain_regret_audit_dir": str(gain_regret_audit_dir),
            "source_summary_sha256": _sha256(source_audit_dir / "summary.json"),
            "source_per_token_pair_interventions_sha256": _sha256(
                source_audit_dir / "per_token_pair_interventions.csv"
            ),
            "control_summary_sha256": _sha256(control_audit_dir / "summary.json"),
            "gain_regret_summary_sha256": _sha256(gain_regret_audit_dir / "summary.json"),
            "git_commit": _git_commit(),
            "variant": TOPK1_VARIANT,
            "selected_intervention": SELECTED_INTERVENTION,
            "logged_oracle_intervention": LOGGED_ORACLE_INTERVENTION,
            "offcontext_filter": (
                "dominant-router singleton rows with router_support_matches_fixed == false"
            ),
            "gain_sign_convention": (
                "singleton_gain = empty_loss - fixed_support_loss; positive means "
                "the singleton support lowers token loss relative to the empty residual"
            ),
            "context_key_fields": list(CONTEXT_FIELDS),
        },
        "missing_controls": {
            "random_singleton_control": (
                "present"
                if signals["random_singleton_control_present"]
                else "missing: current source artifact does not include random singleton rows"
            ),
            "exhaustive_singleton_context_oracle": (
                "missing: current source artifact logs a best-singleton swap row, not all singleton rows"
            ),
        },
    }


def _decision(evidence: dict[str, Any]) -> str:
    signals = evidence["signals"]
    if (
        signals["selected_incontext_positive"]
        and signals["logged_oracle_incontext_positive"]
        and signals["offcontext_fixed_dominant_negative"]
        and signals["selected_positive_on_majority"]
        and signals["offcontext_negative_on_majority"]
    ):
        return CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE
    return UNRESOLVED_SINGLETON_ESTIMAND_MISMATCH


def _context_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in CONTEXT_FIELDS)


def _bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _gains(rows: list[dict[str, str]]) -> list[float]:
    gains = []
    for row in rows:
        logged_gain = _float_or_none(row.get("singleton_left_gain"))
        if logged_gain is not None:
            gains.append(logged_gain)
            continue
        empty_loss = _float_or_none(row.get("empty_loss"))
        fixed_loss = _float_or_none(row.get("fixed_support_loss"))
        if empty_loss is not None and fixed_loss is not None:
            gains.append(empty_loss - fixed_loss)
    return gains


def _best_loss_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best_row: dict[str, str] | None = None
    best_loss = float("inf")
    for row in rows:
        loss = _float_or_none(row.get("fixed_support_loss"))
        if loss is not None and loss < best_loss:
            best_row = row
            best_loss = loss
    return [] if best_row is None else [best_row]


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


def _mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    return _mean_or_none([value for row in rows if isinstance((value := row.get(field)), float)])


def _fraction(values: Iterable[bool]) -> float | None:
    materialized = list(values)
    if not materialized:
        return None
    return sum(1 for value in materialized if value) / len(materialized)


def _positive_mean(values: list[float]) -> bool | None:
    value = _mean_or_none(values)
    return None if value is None else value > 0.0


def _negative_mean(values: list[float]) -> bool | None:
    value = _mean_or_none(values)
    return None if value is None else value < 0.0


def _gt(value: Any, threshold: float) -> bool:
    return isinstance(value, float) and value > threshold


def _lt(value: Any, threshold: float) -> bool:
    return isinstance(value, float) and value < threshold


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["evidence"].get("metrics", {})
    missing = summary["evidence"].get("missing_controls", {})
    lines = [
        "# Active Top-k-1 Singleton Reconciliation Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Contexts: `{metrics.get('context_count')}`",
        f"- Selected in-context contexts: `{metrics.get('selected_context_count')}`",
        f"- Off-context fixed/dominant contexts: `{metrics.get('offcontext_context_count')}`",
        "- Selected singleton gain mean: "
        f"`{metrics.get('selected_singleton_gain_mean')}`",
        "- Logged-oracle singleton gain mean: "
        f"`{metrics.get('logged_oracle_singleton_gain_mean')}`",
        "- Off-context fixed/dominant singleton gain mean: "
        f"`{metrics.get('offcontext_fixed_dominant_singleton_gain_mean')}`",
        f"- Random singleton control: `{missing.get('random_singleton_control')}`",
        f"- Exhaustive singleton control: `{missing.get('exhaustive_singleton_context_oracle')}`",
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


_CONTEXT_FIELDS_OUT = [
    "batch_index",
    "position_index",
    "token_index",
    "target_token",
    "position_bin",
    "token_class",
    "selected_row_count",
    "logged_oracle_row_count",
    "offcontext_row_count",
    "random_singleton_row_count",
    "selected_singleton_gain_mean",
    "logged_oracle_singleton_gain_mean",
    "offcontext_fixed_dominant_singleton_gain_mean",
    "random_singleton_gain_mean",
    "exhaustive_singleton_gain_mean",
    "selected_positive",
    "logged_oracle_positive",
    "offcontext_negative",
    "selected_context",
    "offcontext_context",
    "selected_and_offcontext_context",
]

_STRATUM_FIELDS_OUT = [
    "position_bin",
    "token_class",
    "context_count",
    "selected_context_count",
    "offcontext_context_count",
    "selected_singleton_gain_mean",
    "logged_oracle_singleton_gain_mean",
    "offcontext_fixed_dominant_singleton_gain_mean",
    "random_singleton_gain_mean",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit-dir", type=Path, default=DEFAULT_SOURCE_AUDIT_DIR)
    parser.add_argument("--control-audit-dir", type=Path, default=DEFAULT_CONTROL_AUDIT_DIR)
    parser.add_argument(
        "--gain-regret-audit-dir", type=Path, default=DEFAULT_GAIN_REGRET_AUDIT_DIR
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_active_topk1_singleton_reconciliation_audit(
        source_audit_dir=args.source_audit_dir,
        control_audit_dir=args.control_audit_dir,
        gain_regret_audit_dir=args.gain_regret_audit_dir,
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
