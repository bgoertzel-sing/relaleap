"""Context-conditioned singleton interference audit for active top-k-1."""

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

from relaleap.experiments.active_topk1_post_bracket_research_direction_report import (
    POST_BRACKET_DIRECTION_SELECTED,
    SELECTED_EXPERIMENT,
)


DEFAULT_SOURCE_AUDIT_DIR = Path(
    "results/audits/token_larger_support_wide_promoted_default_causal_column_fingerprint_stability_topk1"
)
DEFAULT_DIRECTION_REPORT_DIR = Path(
    "results/reports/token_larger_active_topk1_post_bracket_research_direction"
)
DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_active_topk1_context_conditioned_singleton_interference"
)

TOPK2_VARIANT = "baseline"
TOPK1_VARIANT = "rank_matched_topk1_contextual"
TOPK2_INTERVENTION = "fixed_dominant_router_support"
SELECTED_SINGLETON_INTERVENTION = "fixed_dominant_router_singleton"
LOGGED_ORACLE_INTERVENTION = "fixed_best_singleton_swap"
RANDOM_SINGLETON_INTERVENTION = "fixed_random_singleton_control"
EXHAUSTIVE_SINGLETON_INTERVENTION = "fixed_exhaustive_singleton"
CONTEXT_FIELDS = ("batch_index", "position_index", "token_index", "target_token")

CONTEXT_CONDITIONED_SINGLETON_INTERFERENCE_AUDIT_ESTABLISHED = (
    "context_conditioned_singleton_interference_audit_established"
)
CONTEXT_GATE_REDUCES_OFFCONTEXT_INTERFERENCE = (
    "context_gate_reduces_offcontext_interference"
)
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def run_active_topk1_context_conditioned_singleton_interference_audit(
    *,
    source_audit_dir: Path = DEFAULT_SOURCE_AUDIT_DIR,
    direction_report_dir: Path = DEFAULT_DIRECTION_REPORT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a bounded no-training singleton interference decomposition."""

    start = time.time()
    failures = _source_failures(source_audit_dir, direction_report_dir)
    source_summary: dict[str, Any] = {}
    direction_summary: dict[str, Any] = {}
    source_rows: list[dict[str, str]] = []
    if not failures:
        source_summary = _read_json_object(source_audit_dir / "summary.json")
        direction_summary = _read_json_object(direction_report_dir / "summary.json")
        source_rows = _read_csv_rows(source_audit_dir / "per_token_pair_interventions.csv")

    if direction_summary and (
        direction_summary.get("decision") != POST_BRACKET_DIRECTION_SELECTED
        or direction_summary.get("selected_experiment") != SELECTED_EXPERIMENT
        or direction_summary.get("status") != "pass"
    ):
        failures.append(
            {
                "field": "direction_report",
                "expected": f"pass/{POST_BRACKET_DIRECTION_SELECTED}/{SELECTED_EXPERIMENT}",
                "actual": {
                    "status": direction_summary.get("status"),
                    "decision": direction_summary.get("decision"),
                    "selected_experiment": direction_summary.get("selected_experiment"),
                },
            }
        )

    required_fields = {
        *CONTEXT_FIELDS,
        "variant",
        "intervention",
        "support",
        "router_support_matches_fixed",
        "empty_loss",
        "router_loss",
        "fixed_support_loss",
        "singleton_left_gain",
        "pair_gain",
        "fixed_support_logit_mse",
        "fixed_support_residual_stream_l2_delta",
        "position_bin",
        "token_class",
        "residual_norm_bin",
        "residual_gain_bin",
        "active_rank_proxy",
    }
    if source_rows:
        missing = sorted(required_fields - set(source_rows[0]))
        if missing:
            failures.append(
                {
                    "field": "source_required_fields",
                    "expected": sorted(required_fields),
                    "actual_missing": missing,
                }
            )

    context_rows: list[dict[str, Any]] = []
    stratum_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        evidence = {"failures": failures}
        rationale = (
            "The context-conditioned singleton interference audit could not run "
            "because required source rows, fields, or the post-bracket direction "
            "packet are missing or inconsistent."
        )
        next_step = "repair the source audit packet before interpreting singleton interference"
    else:
        context_rows = _context_rows(source_rows)
        stratum_rows = _stratum_rows(context_rows)
        gate_rows = _gate_holdout_rows(context_rows)
        evidence = _evidence(
            source_audit_dir=source_audit_dir,
            direction_report_dir=direction_report_dir,
            source_rows=source_rows,
            context_rows=context_rows,
            gate_rows=gate_rows,
            source_summary=source_summary,
            direction_summary=direction_summary,
        )
        failures = _evidence_failures(evidence)
        if failures:
            status = "fail"
            decision = INSUFFICIENT_EVIDENCE
            evidence["failures"] = failures
            rationale = (
                "The source artifact was present, but it did not contain enough "
                "matched contexts and controls for a claim-bearing interference "
                "decomposition."
            )
            next_step = "refresh the causal fingerprint artifact with the missing matched controls"
        else:
            status = "pass"
            decision = CONTEXT_CONDITIONED_SINGLETON_INTERFERENCE_AUDIT_ESTABLISHED
            if evidence["signals"]["context_gate_holdout_net_gain_positive"]:
                decision = CONTEXT_GATE_REDUCES_OFFCONTEXT_INTERFERENCE
            rationale = (
                "The audit decomposes the active top-k-1 packet into own-context "
                "selected singleton gain, matched off-context singleton harm, a "
                "context-gated holdout net estimate, and top-k-2/random/exhaustive "
                "controls. This keeps the claim at column-plus-context-gate level "
                "and continues to exclude a broad reusable singleton claim."
            )
            next_step = (
                "treat the local decomposition as the next source packet; spend GPU "
                "time only if this local packet changes the causal-retention claim"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "singleton_interference_by_context.csv", _CONTEXT_FIELDS_OUT, context_rows)
    _write_csv(out_dir / "singleton_interference_by_stratum.csv", _STRATUM_FIELDS_OUT, stratum_rows)
    _write_csv(out_dir / "context_gate_holdout.csv", _GATE_FIELDS_OUT, gate_rows)
    summary = {
        "status": status,
        "decision": decision,
        "source_audit_dir": str(source_audit_dir),
        "direction_report_dir": str(direction_report_dir),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "evidence": evidence,
        "rationale": rationale,
        "claim_policy": "broad_reusable_singleton_claim_excluded",
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "singleton_interference_by_context_csv": str(
                out_dir / "singleton_interference_by_context.csv"
            ),
            "singleton_interference_by_stratum_csv": str(
                out_dir / "singleton_interference_by_stratum.csv"
            ),
            "context_gate_holdout_csv": str(out_dir / "context_gate_holdout.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _source_failures(source_audit_dir: Path, direction_report_dir: Path) -> list[dict[str, Any]]:
    failures = []
    for field, path in (
        ("source_summary_json", source_audit_dir / "summary.json"),
        ("source_per_token_pair_interventions_csv", source_audit_dir / "per_token_pair_interventions.csv"),
        ("direction_report_summary_json", direction_report_dir / "summary.json"),
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
        topk2 = _rows(values, TOPK2_VARIANT, TOPK2_INTERVENTION)
        topk1_selected = [
            row
            for row in _rows(values, TOPK1_VARIANT, SELECTED_SINGLETON_INTERVENTION)
            if _bool_value(row.get("router_support_matches_fixed"))
        ]
        topk1_offcontext = [
            row
            for row in _rows(values, TOPK1_VARIANT, SELECTED_SINGLETON_INTERVENTION)
            if not _bool_value(row.get("router_support_matches_fixed"))
        ]
        logged_oracle = _rows(values, TOPK1_VARIANT, LOGGED_ORACLE_INTERVENTION)
        random_singleton = _rows(values, TOPK1_VARIANT, RANDOM_SINGLETON_INTERVENTION)
        exhaustive = _rows(values, TOPK1_VARIANT, EXHAUSTIVE_SINGLETON_INTERVENTION)
        first = values[0]
        selected_gain = _mean_or_none(_gains(topk1_selected))
        offcontext_gain = _mean_or_none(_gains(topk1_offcontext))
        topk2_gain = _mean_or_none(_field_values(topk2, "pair_gain"))
        random_gain = _mean_or_none(_gains(random_singleton))
        logged_oracle_gain = _mean_or_none(_gains(_best_loss_rows(topk1_selected + logged_oracle + exhaustive)))
        exhaustive_gain = _mean_or_none(_gains(_best_loss_rows(exhaustive)))
        selected_vs_random = _delta(selected_gain, random_gain)
        offcontext_harm_vs_random = _delta(random_gain, offcontext_gain)
        rows.append(
            {
                "batch_index": context[0],
                "position_index": context[1],
                "token_index": context[2],
                "target_token": context[3],
                "position_bin": first.get("position_bin", ""),
                "token_class": first.get("token_class", ""),
                "residual_norm_bin": first.get("residual_norm_bin", ""),
                "residual_gain_bin": first.get("residual_gain_bin", ""),
                "topk2_row_count": len(topk2),
                "selected_singleton_row_count": len(topk1_selected),
                "offcontext_singleton_row_count": len(topk1_offcontext),
                "random_singleton_row_count": len(random_singleton),
                "exhaustive_singleton_row_count": len(exhaustive),
                "no_residual_loss": _mean_or_none(_field_values(values, "empty_loss")),
                "routed_baseline_loss": _mean_or_none(_field_values(topk2, "router_loss")),
                "topk2_reference_gain": topk2_gain,
                "own_context_singleton_gain": selected_gain,
                "off_context_singleton_gain": offcontext_gain,
                "off_context_singleton_harm": _negate(offcontext_gain),
                "logged_oracle_singleton_gain": logged_oracle_gain,
                "random_singleton_gain": random_gain,
                "exhaustive_singleton_gain": exhaustive_gain,
                "selected_minus_random_gain": selected_vs_random,
                "offcontext_harm_minus_random": offcontext_harm_vs_random,
                "topk2_fixed_support_logit_mse": _mean_or_none(_field_values(topk2, "fixed_support_logit_mse")),
                "topk1_selected_logit_mse": _mean_or_none(_field_values(topk1_selected, "fixed_support_logit_mse")),
                "topk2_residual_stream_l2_delta": _mean_or_none(
                    _field_values(topk2, "fixed_support_residual_stream_l2_delta")
                ),
                "topk1_selected_residual_stream_l2_delta": _mean_or_none(
                    _field_values(topk1_selected, "fixed_support_residual_stream_l2_delta")
                ),
                "has_selected_context": bool(topk1_selected),
                "has_offcontext_match": bool(topk1_offcontext),
                "has_topk2_reference": bool(topk2),
                "has_random_control": bool(random_singleton),
                "has_exhaustive_control": bool(exhaustive),
            }
        )
    return rows


def _stratum_rows(context_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in context_rows:
        grouped[
            (
                str(row.get("position_bin", "")),
                str(row.get("token_class", "")),
                str(row.get("residual_norm_bin", "")),
                str(row.get("residual_gain_bin", "")),
            )
        ].append(row)
    rows = []
    for key, values in sorted(grouped.items()):
        selected = [row for row in values if row.get("has_selected_context")]
        offcontext = [row for row in values if row.get("has_offcontext_match")]
        rows.append(
            {
                "position_bin": key[0],
                "token_class": key[1],
                "residual_norm_bin": key[2],
                "residual_gain_bin": key[3],
                "context_count": len(values),
                "selected_context_count": len(selected),
                "offcontext_context_count": len(offcontext),
                "topk2_reference_context_count": sum(1 for row in values if row.get("has_topk2_reference")),
                "random_control_context_count": sum(1 for row in values if row.get("has_random_control")),
                "exhaustive_control_context_count": sum(1 for row in values if row.get("has_exhaustive_control")),
                "own_context_singleton_gain": _mean_field(selected, "own_context_singleton_gain"),
                "off_context_singleton_gain": _mean_field(offcontext, "off_context_singleton_gain"),
                "off_context_singleton_harm": _mean_field(offcontext, "off_context_singleton_harm"),
                "context_gated_net_gain_proxy": _mean_field(selected, "own_context_singleton_gain"),
                "topk2_reference_gain": _mean_field(values, "topk2_reference_gain"),
                "random_singleton_gain": _mean_field(values, "random_singleton_gain"),
                "exhaustive_singleton_gain": _mean_field(values, "exhaustive_singleton_gain"),
                "selected_minus_random_gain": _mean_field(selected, "selected_minus_random_gain"),
                "offcontext_harm_minus_random": _mean_field(offcontext, "offcontext_harm_minus_random"),
            }
        )
    return rows


def _gate_holdout_rows(context_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train = [row for row in context_rows if _int_or_zero(row.get("position_index")) % 3 != 0]
    holdout = [row for row in context_rows if _int_or_zero(row.get("position_index")) % 3 == 0]
    train_by_gate: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    holdout_by_gate: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in train:
        train_by_gate[_gate_key(row)].append(row)
    for row in holdout:
        holdout_by_gate[_gate_key(row)].append(row)

    rows = []
    for key in sorted(set(train_by_gate) | set(holdout_by_gate)):
        train_rows = train_by_gate.get(key, [])
        holdout_rows = holdout_by_gate.get(key, [])
        train_selected = _mean_field(
            [row for row in train_rows if row.get("has_selected_context")],
            "own_context_singleton_gain",
        )
        train_offcontext = _mean_field(
            [row for row in train_rows if row.get("has_offcontext_match")],
            "off_context_singleton_gain",
        )
        train_random = _mean_field(train_rows, "random_singleton_gain")
        gate_active = _gt(train_selected, 0.0) and (
            train_offcontext is None or train_selected > train_offcontext
        ) and (train_random is None or train_selected >= train_random)
        holdout_selected = _mean_field(
            [row for row in holdout_rows if row.get("has_selected_context")],
            "own_context_singleton_gain",
        )
        holdout_offcontext = _mean_field(
            [row for row in holdout_rows if row.get("has_offcontext_match")],
            "off_context_singleton_gain",
        )
        ungated_holdout_gain = _mean_or_none(
            [
                value
                for value in (holdout_selected, holdout_offcontext)
                if isinstance(value, float)
            ]
        )
        gated_holdout_gain = holdout_selected if gate_active else 0.0
        rows.append(
            {
                "position_bin": key[0],
                "token_class": key[1],
                "residual_norm_bin": key[2],
                "residual_gain_bin": key[3],
                "train_context_count": len(train_rows),
                "holdout_context_count": len(holdout_rows),
                "train_selected_singleton_gain": train_selected,
                "train_offcontext_singleton_gain": train_offcontext,
                "train_random_singleton_gain": train_random,
                "gate_active": gate_active,
                "holdout_selected_singleton_gain": holdout_selected,
                "holdout_offcontext_singleton_gain": holdout_offcontext,
                "holdout_ungated_forced_singleton_gain": ungated_holdout_gain,
                "holdout_context_gated_net_gain": gated_holdout_gain,
                "holdout_context_gate_gain_minus_ungated": _delta(
                    gated_holdout_gain, ungated_holdout_gain
                ),
            }
        )
    return rows


def _evidence(
    *,
    source_audit_dir: Path,
    direction_report_dir: Path,
    source_rows: list[dict[str, str]],
    context_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    source_summary: dict[str, Any],
    direction_summary: dict[str, Any],
) -> dict[str, Any]:
    selected = [row for row in context_rows if row.get("has_selected_context")]
    offcontext = [row for row in context_rows if row.get("has_offcontext_match")]
    metrics = {
        "source_status": source_summary.get("status"),
        "direction_report_decision": direction_summary.get("decision"),
        "source_row_count": len(source_rows),
        "context_count": len(context_rows),
        "selected_context_count": len(selected),
        "offcontext_context_count": len(offcontext),
        "topk2_reference_context_count": sum(1 for row in context_rows if row.get("has_topk2_reference")),
        "random_control_context_count": sum(1 for row in context_rows if row.get("has_random_control")),
        "exhaustive_control_context_count": sum(1 for row in context_rows if row.get("has_exhaustive_control")),
        "own_context_singleton_gain_mean": _mean_field(selected, "own_context_singleton_gain"),
        "off_context_singleton_gain_mean": _mean_field(offcontext, "off_context_singleton_gain"),
        "off_context_singleton_harm_mean": _mean_field(offcontext, "off_context_singleton_harm"),
        "context_gated_net_gain_holdout_mean": _mean_field(gate_rows, "holdout_context_gated_net_gain"),
        "ungated_forced_singleton_gain_holdout_mean": _mean_field(gate_rows, "holdout_ungated_forced_singleton_gain"),
        "context_gate_gain_minus_ungated_holdout_mean": _mean_field(
            gate_rows, "holdout_context_gate_gain_minus_ungated"
        ),
        "topk2_reference_gain_mean": _mean_field(context_rows, "topk2_reference_gain"),
        "random_singleton_gain_mean": _mean_field(context_rows, "random_singleton_gain"),
        "exhaustive_singleton_gain_mean": _mean_field(context_rows, "exhaustive_singleton_gain"),
        "selected_minus_random_gain_mean": _mean_field(selected, "selected_minus_random_gain"),
        "offcontext_harm_minus_random_mean": _mean_field(offcontext, "offcontext_harm_minus_random"),
        "topk1_selected_logit_mse_mean": _mean_field(selected, "topk1_selected_logit_mse"),
        "topk2_fixed_support_logit_mse_mean": _mean_field(context_rows, "topk2_fixed_support_logit_mse"),
        "topk1_selected_residual_stream_l2_delta_mean": _mean_field(
            selected, "topk1_selected_residual_stream_l2_delta"
        ),
        "topk2_residual_stream_l2_delta_mean": _mean_field(
            context_rows, "topk2_residual_stream_l2_delta"
        ),
    }
    signals = {
        "own_context_singleton_gain_positive": _gt(metrics["own_context_singleton_gain_mean"], 0.0),
        "offcontext_singleton_interference_present": _lt(
            metrics["off_context_singleton_gain_mean"], 0.0
        ),
        "random_control_present": metrics["random_control_context_count"] > 0,
        "exhaustive_control_present": metrics["exhaustive_control_context_count"] > 0,
        "matched_topk2_reference_present": metrics["topk2_reference_context_count"] > 0,
        "context_gate_holdout_net_gain_positive": _gt(
            metrics["context_gated_net_gain_holdout_mean"], 0.0
        ),
        "context_gate_improves_over_ungated_holdout": _gt(
            metrics["context_gate_gain_minus_ungated_holdout_mean"], 0.0
        ),
        "topk1_logit_churn_not_higher_than_topk2": _lte(
            metrics["topk1_selected_logit_mse_mean"],
            metrics["topk2_fixed_support_logit_mse_mean"],
        ),
    }
    return {
        "metrics": metrics,
        "signals": signals,
        "provenance": {
            "source_audit_dir": str(source_audit_dir),
            "direction_report_dir": str(direction_report_dir),
            "source_per_token_pair_interventions_sha256": _sha256(
                source_audit_dir / "per_token_pair_interventions.csv"
            ),
            "direction_report_summary_sha256": _sha256(direction_report_dir / "summary.json"),
            "git_commit": _git_commit(),
            "gain_sign_convention": (
                "singleton_gain = empty_loss - fixed_support_loss; positive means "
                "the fixed support lowers token loss relative to no residual"
            ),
            "context_key_fields": list(CONTEXT_FIELDS),
            "gate_features": [
                "position_bin",
                "token_class",
                "residual_norm_bin",
                "residual_gain_bin",
            ],
            "gate_split": "position_index modulo 3 holdout; remaining contexts train",
            "dense_rank_matched_control_source": (
                "not recomputed here; dense active-rank retention control remains "
                "in the functional-retention packet consumed by the direction report"
            ),
        },
    }


def _evidence_failures(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = evidence["metrics"]
    failures = []
    for field in (
        "selected_context_count",
        "offcontext_context_count",
        "topk2_reference_context_count",
        "random_control_context_count",
        "exhaustive_control_context_count",
    ):
        if not metrics.get(field):
            failures.append({"field": field, "expected": "> 0", "actual": metrics.get(field)})
    return failures


def _rows(values: list[dict[str, str]], variant: str, intervention: str) -> list[dict[str, str]]:
    return [
        row
        for row in values
        if row.get("variant") == variant and row.get("intervention") == intervention
    ]


def _context_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in CONTEXT_FIELDS)


def _gate_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("position_bin", "")),
        str(row.get("token_class", "")),
        str(row.get("residual_norm_bin", "")),
        str(row.get("residual_gain_bin", "")),
    )


def _gains(rows: list[dict[str, str]]) -> list[float]:
    gains = []
    for row in rows:
        logged = _float_or_none(row.get("singleton_left_gain"))
        if logged is not None:
            gains.append(logged)
            continue
        empty = _float_or_none(row.get("empty_loss"))
        fixed = _float_or_none(row.get("fixed_support_loss"))
        if empty is not None and fixed is not None:
            gains.append(empty - fixed)
    return gains


def _best_loss_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best_row = None
    best_loss = float("inf")
    for row in rows:
        loss = _float_or_none(row.get("fixed_support_loss"))
        if loss is not None and loss < best_loss:
            best_row = row
            best_loss = loss
    return [] if best_row is None else [best_row]


def _field_values(rows: Iterable[dict[str, Any]], field: str) -> list[float]:
    return [value for row in rows if (value := _float_or_none(row.get(field))) is not None]


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


def _mean_field(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    return _mean_or_none(
        [value for row in rows if isinstance((value := row.get(field)), float)]
    )


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _negate(value: float | None) -> float | None:
    return None if value is None else -value


def _bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _int_or_zero(value: Any) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _gt(value: Any, threshold: float) -> bool:
    return isinstance(value, float) and value > threshold


def _lt(value: Any, threshold: float) -> bool:
    return isinstance(value, float) and value < threshold


def _lte(left: Any, right: Any) -> bool:
    return isinstance(left, float) and isinstance(right, float) and left <= right


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
    lines = [
        "# Active Top-k-1 Context-Conditioned Singleton Interference Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Contexts: `{metrics.get('context_count')}`",
        f"- Selected singleton contexts: `{metrics.get('selected_context_count')}`",
        f"- Off-context singleton contexts: `{metrics.get('offcontext_context_count')}`",
        f"- Own-context singleton gain mean: `{metrics.get('own_context_singleton_gain_mean')}`",
        f"- Off-context singleton gain mean: `{metrics.get('off_context_singleton_gain_mean')}`",
        f"- Context-gated holdout net gain: `{metrics.get('context_gated_net_gain_holdout_mean')}`",
        f"- Top-k-2 reference gain mean: `{metrics.get('topk2_reference_gain_mean')}`",
        f"- Random singleton gain mean: `{metrics.get('random_singleton_gain_mean')}`",
        f"- Exhaustive singleton gain mean: `{metrics.get('exhaustive_singleton_gain_mean')}`",
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
    "residual_norm_bin",
    "residual_gain_bin",
    "topk2_row_count",
    "selected_singleton_row_count",
    "offcontext_singleton_row_count",
    "random_singleton_row_count",
    "exhaustive_singleton_row_count",
    "no_residual_loss",
    "routed_baseline_loss",
    "topk2_reference_gain",
    "own_context_singleton_gain",
    "off_context_singleton_gain",
    "off_context_singleton_harm",
    "logged_oracle_singleton_gain",
    "random_singleton_gain",
    "exhaustive_singleton_gain",
    "selected_minus_random_gain",
    "offcontext_harm_minus_random",
    "topk2_fixed_support_logit_mse",
    "topk1_selected_logit_mse",
    "topk2_residual_stream_l2_delta",
    "topk1_selected_residual_stream_l2_delta",
    "has_selected_context",
    "has_offcontext_match",
    "has_topk2_reference",
    "has_random_control",
    "has_exhaustive_control",
]

_STRATUM_FIELDS_OUT = [
    "position_bin",
    "token_class",
    "residual_norm_bin",
    "residual_gain_bin",
    "context_count",
    "selected_context_count",
    "offcontext_context_count",
    "topk2_reference_context_count",
    "random_control_context_count",
    "exhaustive_control_context_count",
    "own_context_singleton_gain",
    "off_context_singleton_gain",
    "off_context_singleton_harm",
    "context_gated_net_gain_proxy",
    "topk2_reference_gain",
    "random_singleton_gain",
    "exhaustive_singleton_gain",
    "selected_minus_random_gain",
    "offcontext_harm_minus_random",
]

_GATE_FIELDS_OUT = [
    "position_bin",
    "token_class",
    "residual_norm_bin",
    "residual_gain_bin",
    "train_context_count",
    "holdout_context_count",
    "train_selected_singleton_gain",
    "train_offcontext_singleton_gain",
    "train_random_singleton_gain",
    "gate_active",
    "holdout_selected_singleton_gain",
    "holdout_offcontext_singleton_gain",
    "holdout_ungated_forced_singleton_gain",
    "holdout_context_gated_net_gain",
    "holdout_context_gate_gain_minus_ungated",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit-dir", type=Path, default=DEFAULT_SOURCE_AUDIT_DIR)
    parser.add_argument("--direction-report-dir", type=Path, default=DEFAULT_DIRECTION_REPORT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_active_topk1_context_conditioned_singleton_interference_audit(
        source_audit_dir=args.source_audit_dir,
        direction_report_dir=args.direction_report_dir,
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
