"""Selected-vs-oracle singleton control diagnostic for active top-k-1."""

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
DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_active_rank_matched_topk1_singleton_control_diagnostic"
)

TOPK1_VARIANT = "rank_matched_topk1_contextual"
SELECTED_INTERVENTION = "fixed_dominant_router_singleton"
LOGGED_ORACLE_INTERVENTION = "fixed_best_singleton_swap"
RANDOM_INTERVENTION = "fixed_random_singleton_control"
CONTEXT_FIELDS = ("batch_index", "position_index", "token_index", "target_token")

LIKELY_ROUTER_SELECTION_FAILURE = "likely_router_selection_failure"
LIKELY_SINGLETON_CAPACITY_OR_VALUE_FAILURE = "likely_singleton_capacity_or_value_failure"
MIXED_SINGLETON_CONTROL_EVIDENCE = "mixed_singleton_control_evidence"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def run_active_topk1_singleton_control_diagnostic(
    *,
    source_audit_dir: Path = DEFAULT_SOURCE_AUDIT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    regret_threshold: float = 0.05,
) -> dict[str, Any]:
    """Compare selected top-k-1 singleton rows to logged oracle singleton rows."""

    start = time.time()
    failures = _source_failures(source_audit_dir)
    source_summary: dict[str, Any] = {}
    source_rows: list[dict[str, str]] = []
    if not failures:
        source_summary = _read_json_object(source_audit_dir / "summary.json")
        source_rows = _read_csv_rows(source_audit_dir / "per_token_pair_interventions.csv")

    topk1_rows = [row for row in source_rows if row.get("variant") == TOPK1_VARIANT]
    selected_rows = [
        row
        for row in topk1_rows
        if row.get("intervention") == SELECTED_INTERVENTION
        and _bool_value(row.get("router_support_matches_fixed"))
    ]
    oracle_rows = [
        row for row in topk1_rows if row.get("intervention") == LOGGED_ORACLE_INTERVENTION
    ]
    random_rows = [row for row in topk1_rows if row.get("intervention") == RANDOM_INTERVENTION]

    required_fields = {
        *CONTEXT_FIELDS,
        "intervention",
        "support",
        "router_support_matches_fixed",
        "empty_loss",
        "router_loss",
        "fixed_support_loss",
        "singleton_left_gain",
        "fixed_support_logit_mse",
        "active_rank_proxy",
    }
    if topk1_rows:
        missing = sorted(required_fields - set(topk1_rows[0]))
        if missing:
            failures.append(
                {
                    "field": "topk1_singleton_control_required_fields",
                    "expected": sorted(required_fields),
                    "actual_missing": missing,
                }
            )
    if source_rows and not selected_rows:
        failures.append(
            {
                "field": "selected_router_matched_singleton_rows",
                "expected": f"{SELECTED_INTERVENTION} rows with router_support_matches_fixed",
                "actual": 0,
            }
        )
    if source_rows and not oracle_rows:
        failures.append(
            {
                "field": "logged_oracle_singleton_rows",
                "expected": f"{LOGGED_ORACLE_INTERVENTION} rows",
                "actual": 0,
            }
        )

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        context_rows: list[dict[str, Any]] = []
        stratum_rows: list[dict[str, Any]] = []
        evidence = {
            "failures": failures,
            "missing_controls": {
                "random_singleton_control": "not evaluated because required source evidence is missing",
            },
        }
        rationale = (
            "The selected-vs-oracle singleton control diagnostic could not be "
            "established because required source artifacts or fields are missing."
        )
        next_step = (
            "repair or regenerate the source causal-column fingerprint artifact "
            "before interpreting selected singleton controls"
        )
    else:
        context_rows = _context_rows(selected_rows, oracle_rows, random_rows)
        if not context_rows:
            failures.append(
                {
                    "field": "matched_context_count",
                    "expected": "at least one context with selected and logged-oracle rows",
                    "actual": 0,
                }
            )
            status = "fail"
            decision = INSUFFICIENT_EVIDENCE
            stratum_rows = []
            evidence = {
                "failures": failures,
                "missing_controls": {
                    "random_singleton_control": "not present in current source artifact",
                },
            }
            rationale = (
                "No exact contexts had both a selected router singleton and a "
                "logged oracle singleton row."
            )
            next_step = (
                "extend or regenerate the causal-column fingerprint artifact with "
                "same-context selected and oracle singleton rows"
            )
        else:
            stratum_rows = _stratum_rows(context_rows)
            evidence = _build_evidence(
                source_summary=source_summary,
                context_rows=context_rows,
                random_rows=random_rows,
                source_audit_dir=source_audit_dir,
                regret_threshold=regret_threshold,
            )
            decision = _decision(evidence)
            status = "pass"
            if decision == LIKELY_ROUTER_SELECTION_FAILURE:
                rationale = (
                    "The router-selected top-k-1 singleton is negative on average, "
                    "but the best logged singleton on the same contexts is positive "
                    "and substantially better. Within the logged singleton bracket, "
                    "the failure is primarily support selection rather than absence "
                    "of useful learned singleton values."
                )
                next_step = (
                    "extend the causal-column fingerprint source artifact with "
                    "random and exhaustive singleton controls, then test whether a "
                    "train-time singleton router can close the selected-oracle gap"
                )
            elif decision == LIKELY_SINGLETON_CAPACITY_OR_VALUE_FAILURE:
                rationale = (
                    "Even the best logged singleton rows are non-beneficial on "
                    "average, so the current evidence points toward singleton "
                    "capacity or learned-value quality rather than just router "
                    "selection."
                )
                next_step = (
                    "run an artifact extension with exhaustive singleton rows and "
                    "rank/dense controls before investing in a new singleton router"
                )
            else:
                rationale = (
                    "Selected singleton controls are mixed across same-context "
                    "selected, logged-oracle, and random-control availability, so "
                    "the negative singleton-gain cause remains unresolved."
                )
                next_step = (
                    "extend the source artifact with random singleton controls and "
                    "more complete singleton alternatives on the same contexts"
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "singleton_control_by_context.csv", _CONTEXT_FIELDS_OUT, context_rows)
    _write_csv(out_dir / "singleton_control_by_stratum.csv", _STRATUM_FIELDS_OUT, stratum_rows)
    summary = {
        "status": status,
        "decision": decision,
        "source_audit_dir": str(source_audit_dir),
        "out_dir": str(out_dir),
        "regret_threshold": regret_threshold,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "evidence": evidence,
        "rationale": rationale,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "singleton_control_by_context_csv": str(
                out_dir / "singleton_control_by_context.csv"
            ),
            "singleton_control_by_stratum_csv": str(
                out_dir / "singleton_control_by_stratum.csv"
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


def _source_failures(source_audit_dir: Path) -> list[dict[str, Any]]:
    failures = []
    for field, path in (
        ("source_summary_json", source_audit_dir / "summary.json"),
        (
            "source_per_token_pair_interventions_csv",
            source_audit_dir / "per_token_pair_interventions.csv",
        ),
    ):
        if not path.is_file():
            failures.append(
                {"field": field, "expected": "file exists", "actual": "missing", "path": str(path)}
            )
    return failures


def _context_rows(
    selected_rows: list[dict[str, str]],
    oracle_rows: list[dict[str, str]],
    random_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    selected_by_context: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    alternatives_by_context: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    random_by_context: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in selected_rows:
        selected_by_context[_context_key(row)].append(row)
        alternatives_by_context[_context_key(row)].append(row)
    for row in oracle_rows:
        alternatives_by_context[_context_key(row)].append(row)
    for row in random_rows:
        random_by_context[_context_key(row)].append(row)

    out_rows = []
    for context, rows in sorted(selected_by_context.items()):
        selected = min(rows, key=lambda row: _float_or_inf(row.get("fixed_support_loss")))
        alternatives = [
            row
            for row in alternatives_by_context.get(context, [])
            if _float_or_none(row.get("fixed_support_loss")) is not None
        ]
        if not alternatives:
            continue
        oracle = min(alternatives, key=lambda row: _float_or_inf(row.get("fixed_support_loss")))
        random_losses = _float_values(random_by_context.get(context, []), "fixed_support_loss")
        random_gains = _float_values(random_by_context.get(context, []), "singleton_left_gain")
        selected_loss = _float_or_none(selected.get("fixed_support_loss"))
        oracle_loss = _float_or_none(oracle.get("fixed_support_loss"))
        if selected_loss is None or oracle_loss is None:
            continue
        selected_gain = _float_or_none(selected.get("singleton_left_gain"))
        oracle_gain = _float_or_none(oracle.get("singleton_left_gain"))
        first = selected
        out_rows.append(
            {
                "batch_index": context[0],
                "position_index": context[1],
                "token_index": context[2],
                "target_token": context[3],
                "position_bin": first.get("position_bin", ""),
                "token_class": first.get("token_class", ""),
                "selected_support": selected.get("support", ""),
                "logged_oracle_support": oracle.get("support", ""),
                "selected_intervention": selected.get("intervention", ""),
                "logged_oracle_intervention": oracle.get("intervention", ""),
                "selected_singleton_gain": selected_gain,
                "logged_oracle_singleton_gain": oracle_gain,
                "selected_fixed_support_loss": selected_loss,
                "logged_oracle_fixed_support_loss": oracle_loss,
                "selected_to_logged_oracle_regret": selected_loss - oracle_loss,
                "selected_positive_gain": _gt(selected_gain, 0.0),
                "logged_oracle_positive_gain": _gt(oracle_gain, 0.0),
                "selected_negative_oracle_positive": _lt(selected_gain, 0.0)
                and _gt(oracle_gain, 0.0),
                "random_singleton_count": len(random_losses),
                "random_singleton_loss_mean": _mean_or_none(random_losses),
                "random_singleton_gain_mean": _mean_or_none(random_gains),
                "selected_minus_random_loss_mean": (
                    selected_loss - mean(random_losses) if random_losses else None
                ),
                "active_rank_proxy": first.get("active_rank_proxy", ""),
                "fixed_support_logit_mse": _float_or_none(
                    selected.get("fixed_support_logit_mse")
                ),
            }
        )
    return out_rows


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
                "selected_singleton_gain_mean": _mean_field(
                    values, "selected_singleton_gain"
                ),
                "logged_oracle_singleton_gain_mean": _mean_field(
                    values, "logged_oracle_singleton_gain"
                ),
                "selected_to_logged_oracle_regret_mean": _mean_field(
                    values, "selected_to_logged_oracle_regret"
                ),
                "selected_positive_fraction": _fraction(
                    bool(row.get("selected_positive_gain")) for row in values
                ),
                "logged_oracle_positive_fraction": _fraction(
                    bool(row.get("logged_oracle_positive_gain")) for row in values
                ),
                "selected_negative_oracle_positive_fraction": _fraction(
                    bool(row.get("selected_negative_oracle_positive")) for row in values
                ),
            }
        )
    return rows


def _build_evidence(
    *,
    source_summary: dict[str, Any],
    context_rows: list[dict[str, Any]],
    random_rows: list[dict[str, str]],
    source_audit_dir: Path,
    regret_threshold: float,
) -> dict[str, Any]:
    metrics = {
        "source_status": source_summary.get("status"),
        "source_decision": source_summary.get("decision"),
        "context_count": len(context_rows),
        "selected_singleton_gain_mean": _mean_field(
            context_rows, "selected_singleton_gain"
        ),
        "logged_oracle_singleton_gain_mean": _mean_field(
            context_rows, "logged_oracle_singleton_gain"
        ),
        "selected_to_logged_oracle_regret_mean": _mean_field(
            context_rows, "selected_to_logged_oracle_regret"
        ),
        "selected_positive_fraction": _fraction(
            bool(row.get("selected_positive_gain")) for row in context_rows
        ),
        "logged_oracle_positive_fraction": _fraction(
            bool(row.get("logged_oracle_positive_gain")) for row in context_rows
        ),
        "selected_negative_oracle_positive_fraction": _fraction(
            bool(row.get("selected_negative_oracle_positive")) for row in context_rows
        ),
        "random_singleton_context_count": sum(
            1 for row in context_rows if int(row.get("random_singleton_count") or 0) > 0
        ),
        "random_singleton_row_count": len(random_rows),
    }
    signals = {
        "selected_singleton_gain_negative": _lt(
            metrics["selected_singleton_gain_mean"], 0.0
        ),
        "logged_oracle_singleton_gain_positive": _gt(
            metrics["logged_oracle_singleton_gain_mean"], 0.0
        ),
        "selected_to_oracle_regret_material": _gt(
            metrics["selected_to_logged_oracle_regret_mean"], regret_threshold
        ),
        "logged_oracle_positive_on_majority": _gt(
            metrics["logged_oracle_positive_fraction"], 0.5
        ),
        "selected_negative_oracle_positive_common": _gt(
            metrics["selected_negative_oracle_positive_fraction"], 0.25
        ),
        "random_singleton_control_present": metrics["random_singleton_row_count"] > 0,
    }
    return {
        "metrics": metrics,
        "signals": signals,
        "provenance": {
            "source_audit_dir": str(source_audit_dir),
            "source_summary_sha256": _sha256(source_audit_dir / "summary.json"),
            "source_per_token_pair_interventions_sha256": _sha256(
                source_audit_dir / "per_token_pair_interventions.csv"
            ),
            "git_commit": _git_commit(),
            "variant": TOPK1_VARIANT,
            "selected_intervention": SELECTED_INTERVENTION,
            "logged_oracle_intervention": LOGGED_ORACLE_INTERVENTION,
            "random_intervention": RANDOM_INTERVENTION,
            "gain_sign_convention": (
                "singleton_gain = empty_loss - fixed_support_loss; positive means "
                "the singleton support lowers token loss relative to the empty residual"
            ),
            "context_key_fields": list(CONTEXT_FIELDS),
            "selected_context_filter": (
                "dominant-router singleton rows with router_support_matches_fixed == true"
            ),
        },
        "missing_controls": {
            "random_singleton_control": (
                "present"
                if signals["random_singleton_control_present"]
                else "missing: current source artifact does not include random singleton rows"
            ),
            "exhaustive_singleton_context_oracle": (
                "missing: logged oracle is limited to source-artifact selected singleton alternatives"
            ),
        },
    }


def _decision(evidence: dict[str, Any]) -> str:
    signals = evidence["signals"]
    if (
        signals["selected_singleton_gain_negative"]
        and signals["logged_oracle_singleton_gain_positive"]
        and signals["selected_to_oracle_regret_material"]
        and signals["logged_oracle_positive_on_majority"]
    ):
        return LIKELY_ROUTER_SELECTION_FAILURE
    if (
        signals["selected_singleton_gain_negative"]
        and not signals["logged_oracle_singleton_gain_positive"]
    ):
        return LIKELY_SINGLETON_CAPACITY_OR_VALUE_FAILURE
    return MIXED_SINGLETON_CONTROL_EVIDENCE


def _context_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in CONTEXT_FIELDS)


def _bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float_values(rows: list[dict[str, str]], field: str) -> list[float]:
    return [value for row in rows if (value := _float_or_none(row.get(field))) is not None]


def _float_or_inf(value: Any) -> float:
    parsed = _float_or_none(value)
    return float("inf") if parsed is None else parsed


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
        "# Active Top-k-1 Singleton Control Diagnostic",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Context count: `{metrics.get('context_count')}`",
        "- Selected singleton gain mean: "
        f"`{metrics.get('selected_singleton_gain_mean')}`",
        "- Logged-oracle singleton gain mean: "
        f"`{metrics.get('logged_oracle_singleton_gain_mean')}`",
        "- Selected-to-logged-oracle regret mean: "
        f"`{metrics.get('selected_to_logged_oracle_regret_mean')}`",
        "- Selected negative / oracle positive fraction: "
        f"`{metrics.get('selected_negative_oracle_positive_fraction')}`",
        f"- Random singleton control: `{missing.get('random_singleton_control')}`",
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
    "selected_support",
    "logged_oracle_support",
    "selected_intervention",
    "logged_oracle_intervention",
    "selected_singleton_gain",
    "logged_oracle_singleton_gain",
    "selected_fixed_support_loss",
    "logged_oracle_fixed_support_loss",
    "selected_to_logged_oracle_regret",
    "selected_positive_gain",
    "logged_oracle_positive_gain",
    "selected_negative_oracle_positive",
    "random_singleton_count",
    "random_singleton_loss_mean",
    "random_singleton_gain_mean",
    "selected_minus_random_loss_mean",
    "active_rank_proxy",
    "fixed_support_logit_mse",
]

_STRATUM_FIELDS_OUT = [
    "position_bin",
    "token_class",
    "context_count",
    "selected_singleton_gain_mean",
    "logged_oracle_singleton_gain_mean",
    "selected_to_logged_oracle_regret_mean",
    "selected_positive_fraction",
    "logged_oracle_positive_fraction",
    "selected_negative_oracle_positive_fraction",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit-dir", type=Path, default=DEFAULT_SOURCE_AUDIT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--regret-threshold", type=float, default=0.05)
    args = parser.parse_args(argv)
    summary = run_active_topk1_singleton_control_diagnostic(
        source_audit_dir=args.source_audit_dir,
        out_dir=args.out,
        regret_threshold=args.regret_threshold,
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
