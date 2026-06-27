"""Broader local mechanism gate for anticipatory contextual support routing."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_SOURCE_DIRS = [
    Path("results/audits/token_larger_anticipatory_contextual_support_routing"),
    Path("results/audits/token_larger_anticipatory_contextual_support_routing_seed2"),
]
DEFAULT_OUT_DIR = Path("results/audits/acsr_broader_mechanism_gate_local")
REQUIRED_ARTIFACTS = [
    "summary.json",
    "variant_metrics.csv",
    "same_student_cross_forcing.csv",
    "perturbation_metrics.csv",
    "margin_fragility.csv",
    "parameter_counts.csv",
    "notes.md",
]

REQUIRED_SOURCE_ARTIFACTS = [
    "summary.json",
    "router_metrics.csv",
    "same_student_metrics.csv",
    "feature_perturbation.csv",
    "sequence_heldout_metrics.csv",
    "margin_fragility.csv",
    "parameter_counts.csv",
    "retention_churn_metrics.csv",
]

REQUIRED_VARIANTS = [
    "causal_feature_safe_contextual_topk2",
    "acsr_mlp_predicted_future",
    "shuffled_predicted_features",
    "token_position_only_predicted_features",
    "mean_predicted_features",
    "zero_predicted_features",
    "parameter_matched_causal_mlp_control",
    "random_fixed_topk2",
]


def run_acsr_broader_mechanism_gate(
    *,
    source_dirs: list[Path] | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    strategy_review: Path | None = None,
) -> dict[str, Any]:
    """Validate and aggregate existing ACSR packets into a fail-closed gate."""

    start = time.time()
    source_dirs = source_dirs or list(DEFAULT_SOURCE_DIRS)
    packets = [_load_packet(source_dir) for source_dir in source_dirs]
    failures = _packet_failures(packets)

    variant_rows = _variant_rows(packets)
    same_student_rows = _same_student_rows(packets)
    perturbation_rows = _perturbation_rows(packets)
    margin_rows = _margin_rows(packets)
    parameter_rows = _parameter_rows(packets)

    aggregate = _aggregate_metrics(variant_rows, same_student_rows, perturbation_rows)
    missing_controls = _missing_required_controls(
        variant_rows,
        same_student_rows,
        perturbation_rows,
        margin_rows,
        parameter_rows,
    )
    failures.extend(
        {"gate": "required_control", "reason": reason}
        for reason in missing_controls
    )
    if aggregate["parameter_matched_causal_control_available"] and not aggregate[
        "acsr_beats_parameter_matched_causal_control"
    ]:
        failures.append(
            {
                "gate": "acsr_anticipation_specific_claim",
                "reason": "acsr_not_better_than_parameter_matched_causal_control",
            }
        )

    accepted_review_notes = _strategy_review_notes(strategy_review)
    status = "pass" if not failures else "fail"
    summary = {
        "status": status,
        "decision": (
            "acsr_broader_mechanism_gate_passed"
            if status == "pass"
            else "acsr_broader_mechanism_gate_failed_closed"
        ),
        "claim_status": (
            "acsr_broader_local_mechanism_gate_supported"
            if status == "pass"
            else (
                "acsr_anticipation_specific_claim_blocked_no_default_change"
                if aggregate["parameter_matched_causal_control_available"]
                and not aggregate["acsr_beats_parameter_matched_causal_control"]
                else "acsr_broader_local_mechanism_gate_incomplete_no_default_change"
            )
        ),
        "source_dirs": [str(path) for path in source_dirs],
        "source_packet_count": len(packets),
        "loaded_packet_count": sum(1 for packet in packets if packet["loaded"]),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "gates": {
            "source_artifacts_present": not any(
                failure["gate"] == "source_artifact" for failure in failures
            ),
            "required_variants_present": not any(
                failure["gate"] == "source_variant" for failure in failures
            ),
            "acsr_beats_nulls_on_available_packets": aggregate[
                "acsr_beats_nulls_on_available_packets"
            ],
            "future_perturbation_negative_control": aggregate[
                "future_perturbation_negative_control"
            ],
            "same_student_available": bool(same_student_rows),
            "sequence_heldout_available": any(packet["sequence_rows"] for packet in packets),
            "dual_student_cross_forcing_available": False,
            "leaky_positive_control_available": any(
                row.get("control_type") == "leaky_future_positive"
                and str(row.get("passed", "")).lower() == "true"
                for row in perturbation_rows
            ),
            "parameter_matched_causal_control_available": any(
                row.get("component") == "parameter_matched_causal_mlp_control"
                and row.get("status") == "available"
                for row in parameter_rows
            ),
            "acsr_beats_parameter_matched_causal_control": aggregate[
                "acsr_beats_parameter_matched_causal_control"
            ],
        },
        "aggregate_metrics": aggregate,
        "failures": failures,
        "strategy_review": accepted_review_notes,
        "direction_shift": {
            "level": accepted_review_notes.get("strategic_change_level"),
            "notify_ben": accepted_review_notes.get("notify_ben"),
            "record": _direction_shift_record(accepted_review_notes),
        },
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(
        out_dir,
        summary,
        variant_rows=variant_rows,
        same_student_rows=same_student_rows,
        perturbation_rows=perturbation_rows,
        margin_rows=margin_rows,
        parameter_rows=parameter_rows,
    )
    return summary


def _load_packet(source_dir: Path) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "source_dir": str(source_dir),
        "loaded": False,
        "missing_artifacts": [],
        "summary": {},
        "router_rows": [],
        "same_student_rows": [],
        "perturbation_rows": [],
        "sequence_rows": [],
        "margin_rows": [],
        "parameter_rows": [],
        "retention_rows": [],
    }
    missing = [
        name for name in REQUIRED_SOURCE_ARTIFACTS if not (source_dir / name).is_file()
    ]
    if missing:
        packet["missing_artifacts"] = missing
        return packet
    packet["summary"] = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    packet["router_rows"] = _read_csv(source_dir / "router_metrics.csv")
    packet["same_student_rows"] = _read_csv(source_dir / "same_student_metrics.csv")
    packet["perturbation_rows"] = _read_csv(source_dir / "feature_perturbation.csv")
    packet["sequence_rows"] = _read_csv(source_dir / "sequence_heldout_metrics.csv")
    packet["margin_rows"] = _read_csv(source_dir / "margin_fragility.csv")
    packet["parameter_rows"] = _read_csv(source_dir / "parameter_counts.csv")
    packet["retention_rows"] = _read_csv(source_dir / "retention_churn_metrics.csv")
    packet["loaded"] = True
    return packet


def _packet_failures(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for packet in packets:
        source_dir = packet["source_dir"]
        for artifact in packet["missing_artifacts"]:
            failures.append(
                {
                    "gate": "source_artifact",
                    "source_dir": source_dir,
                    "reason": f"missing {artifact}",
                }
            )
        if not packet["loaded"]:
            continue
        if packet["summary"].get("status") != "pass":
            failures.append(
                {
                    "gate": "source_packet_status",
                    "source_dir": source_dir,
                    "reason": f"source status is {packet['summary'].get('status')}",
                }
            )
        variants = {row.get("variant") for row in packet["router_rows"]}
        for variant in REQUIRED_VARIANTS:
            if variant not in variants:
                failures.append(
                    {
                        "gate": "source_variant",
                        "source_dir": source_dir,
                        "reason": f"missing router variant {variant}",
                    }
                )
    return failures


def _variant_rows(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for packet in packets:
        if not packet["loaded"]:
            continue
        summary = packet["summary"]
        for row in packet["router_rows"]:
            enriched = {
                "source_dir": packet["source_dir"],
                "seed": _seed_label(summary),
                "variant": row.get("variant", ""),
                "control_family": _control_family(row.get("variant", "")),
                "status": "available",
                "ce_loss": _float_or_blank(row.get("ce_loss")),
                "oracle_regret": _float_or_blank(row.get("oracle_regret")),
                "used_columns": _int_or_blank(row.get("used_columns")),
                "unique_support_sets": _int_or_blank(row.get("unique_support_sets")),
                "support_entropy": _float_or_blank(row.get("support_entropy")),
                "mean_topk_margin": _float_or_blank(row.get("mean_topk_margin")),
                "top_k": _int_or_blank(row.get("top_k")),
            }
            rows.append(enriched)
        acsr = _row_by_variant(packet["router_rows"], "acsr_mlp_predicted_future")
        causal = _row_by_variant(packet["router_rows"], "causal_feature_safe_contextual_topk2")
        if acsr and causal:
            rows.append(
                _margin_gated_proxy_row(packet["source_dir"], summary, acsr, causal)
            )
        for row in packet["sequence_rows"]:
            rows.append(
                {
                    "source_dir": packet["source_dir"],
                    "seed": _seed_label(summary),
                    "split": row.get("split", "sequence_suffix_holdout"),
                    "variant": row.get("variant", ""),
                    "control_family": _control_family(row.get("variant", "")),
                    "status": "available_sequence_heldout",
                    "ce_loss": _float_or_blank(row.get("ce_loss")),
                    "oracle_regret": _float_or_blank(row.get("oracle_regret")),
                    "oracle_loss": _float_or_blank(row.get("oracle_loss")),
                    "used_columns": "",
                    "unique_support_sets": "",
                    "support_entropy": "",
                    "mean_topk_margin": "",
                    "top_k": _int_or_blank(row.get("top_k")),
                    "holdout_start": _int_or_blank(row.get("holdout_start")),
                    "heldout_positions": _int_or_blank(row.get("heldout_positions")),
                }
            )
    return rows


def _margin_gated_proxy_row(
    source_dir: str,
    summary: dict[str, Any],
    acsr: dict[str, str],
    causal: dict[str, str],
) -> dict[str, Any]:
    acsr_margin = _float(acsr.get("mean_topk_margin"))
    causal_margin = _float(causal.get("mean_topk_margin"))
    use_acsr = acsr_margin >= causal_margin
    selected = acsr if use_acsr else causal
    return {
        "source_dir": source_dir,
        "seed": _seed_label(summary),
        "variant": "margin_gated_acsr_proxy",
        "control_family": "margin_gated_acsr",
        "status": "available_proxy_from_aggregate_margin",
        "ce_loss": _float_or_blank(selected.get("ce_loss")),
        "oracle_regret": _float_or_blank(selected.get("oracle_regret")),
        "used_columns": _int_or_blank(selected.get("used_columns")),
        "unique_support_sets": _int_or_blank(selected.get("unique_support_sets")),
        "support_entropy": _float_or_blank(selected.get("support_entropy")),
        "mean_topk_margin": acsr_margin,
        "top_k": _int_or_blank(selected.get("top_k")),
        "fallback_variant": "acsr_mlp_predicted_future" if use_acsr else "causal_feature_safe_contextual_topk2",
        "note": "Proxy only: true token-level margin fallback requires score tensors.",
    }


def _same_student_rows(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for packet in packets:
        if not packet["loaded"]:
            continue
        for row in packet["same_student_rows"]:
            rows.append(
                {
                    "source_dir": packet["source_dir"],
                    "seed": _seed_label(packet["summary"]),
                    "forcing_type": "same_student",
                    "comparison": row.get("comparison", ""),
                    "acsr_forced_ce_loss": _float_or_blank(
                        row.get("acsr_forced_ce_loss")
                    ),
                    "control_forced_ce_loss": _float_or_blank(
                        row.get("control_forced_ce_loss")
                    ),
                    "acsr_minus_control_ce_loss": _float_or_blank(
                        row.get("acsr_minus_control_ce_loss")
                    ),
                    "status": "available",
                }
            )
        rows.append(
            {
                "source_dir": packet["source_dir"],
                "seed": _seed_label(packet["summary"]),
                "forcing_type": "dual_student_cross_forcing",
                "comparison": "acsr_support_cross_forced_through_independent_student",
                "status": "not_available_in_source_artifact",
                "reason": "existing ACSR packets do not store independent student values/support tensors",
            }
        )
    return rows


def _perturbation_rows(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for packet in packets:
        if not packet["loaded"]:
            continue
        for row in packet["perturbation_rows"]:
            enriched = {
                "source_dir": packet["source_dir"],
                "seed": _seed_label(packet["summary"]),
                "control_type": "future_perturbation_negative",
                "status": "available",
            }
            enriched.update(row)
            rows.append(enriched)
        if not any(
            row.get("control_type") == "leaky_future_positive"
            for row in packet["perturbation_rows"]
        ):
            rows.append(
                {
                    "source_dir": packet["source_dir"],
                    "seed": _seed_label(packet["summary"]),
                    "control_type": "leaky_future_positive",
                    "status": "not_available_in_source_artifact",
                    "reason": "existing perturbation artifact lacks intentionally leaky positive control",
                }
            )
    return rows


def _margin_rows(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for packet in packets:
        if not packet["loaded"]:
            continue
        if packet["margin_rows"]:
            for row in packet["margin_rows"]:
                enriched = {
                    "source_dir": packet["source_dir"],
                    "seed": _seed_label(packet["summary"]),
                    "status": "available",
                }
                enriched.update(row)
                rows.append(enriched)
        else:
            for row in packet["router_rows"]:
                if not row.get("variant"):
                    continue
                rows.append(
                    {
                        "source_dir": packet["source_dir"],
                        "seed": _seed_label(packet["summary"]),
                        "variant": row["variant"],
                        "mean_topk_margin": _float_or_blank(row.get("mean_topk_margin")),
                        "feature_noise_flip_rate": "",
                        "status": "margin_available_fragility_noise_not_available",
                    }
                )
    return rows


def _parameter_rows(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for packet in packets:
        if not packet["loaded"]:
            continue
        summary = packet["summary"]
        if packet["parameter_rows"]:
            for row in packet["parameter_rows"]:
                enriched = {
                    "source_dir": packet["source_dir"],
                    "seed": _seed_label(summary),
                    "status": row.get("status", "available"),
                }
                enriched.update(row)
                rows.append(enriched)
            continue
        hidden_dim = int(summary.get("hidden_dim", 0) or 0)
        num_columns = int(summary.get("num_columns", 0) or 0)
        top_k = int(summary.get("top_k", 0) or 0)
        rows.extend(
            [
                {
                    "source_dir": packet["source_dir"],
                    "seed": _seed_label(summary),
                    "component": "residual_columns",
                    "active_parameter_count": "",
                    "stored_parameter_count": "",
                    "basis": f"hidden_dim={hidden_dim};num_columns={num_columns};top_k={top_k}",
                    "status": "not_available_in_source_artifact",
                },
                {
                    "source_dir": packet["source_dir"],
                    "seed": _seed_label(summary),
                    "component": "parameter_matched_causal_mlp_control",
                    "active_parameter_count": "",
                    "stored_parameter_count": "",
                    "basis": "required by broader gate",
                    "status": "not_available_in_source_artifact",
                },
            ]
        )
    return rows


def _aggregate_metrics(
    variant_rows: list[dict[str, Any]],
    same_student_rows: list[dict[str, Any]],
    perturbation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    acsr_rows = [row for row in variant_rows if row["variant"] == "acsr_mlp_predicted_future"]
    null_names = {
        "shuffled_predicted_features",
        "token_position_only_predicted_features",
        "mean_predicted_features",
        "zero_predicted_features",
    }
    null_rows = [row for row in variant_rows if row["variant"] in null_names]
    acsr_ce = [_float(row["ce_loss"]) for row in acsr_rows if row["ce_loss"] != ""]
    null_ce = [_float(row["ce_loss"]) for row in null_rows if row["ce_loss"] != ""]
    same_student_deltas = [
        _float(row["acsr_minus_control_ce_loss"])
        for row in same_student_rows
        if row.get("forcing_type") == "same_student"
        and row.get("acsr_minus_control_ce_loss") not in ("", None)
    ]
    parameter_matched_deltas = _paired_acsr_parameter_matched_deltas(variant_rows)
    acsr_beats_parameter_matched = bool(parameter_matched_deltas) and all(
        delta["acsr_minus_parameter_matched_ce_loss"] < 0.0
        and (
            delta["acsr_minus_parameter_matched_oracle_regret"] in ("", None)
            or delta["acsr_minus_parameter_matched_oracle_regret"] < 0.0
        )
        for delta in parameter_matched_deltas
    )
    negative_control_passed = all(
        str(row.get("passed", "")).lower() == "true"
        for row in perturbation_rows
        if row.get("control_type") == "future_perturbation_negative"
    )
    return {
        "mean_acsr_ce_loss": _mean_or_none(acsr_ce),
        "mean_null_ce_loss": _mean_or_none(null_ce),
        "mean_acsr_minus_null_same_student_ce": _mean_or_none(same_student_deltas),
        "acsr_beats_nulls_on_available_packets": bool(
            acsr_ce and null_ce and max(acsr_ce) < min(null_ce)
        ),
        "future_perturbation_negative_control": negative_control_passed,
        "same_student_available_count": len(same_student_deltas),
        "available_packet_count": len({row["source_dir"] for row in variant_rows}),
        "parameter_matched_causal_control_available": bool(parameter_matched_deltas),
        "acsr_beats_parameter_matched_causal_control": acsr_beats_parameter_matched,
        "acsr_parameter_matched_paired_deltas": parameter_matched_deltas,
    }


def _paired_acsr_parameter_matched_deltas(
    variant_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in variant_rows:
        variant = row.get("variant")
        if variant not in {
            "acsr_mlp_predicted_future",
            "parameter_matched_causal_mlp_control",
        }:
            continue
        key = (
            row.get("source_dir"),
            row.get("seed"),
            row.get("status"),
            row.get("split", "fixed_context"),
            row.get("holdout_start", ""),
        )
        grouped.setdefault(key, {})[variant] = row
    for key, group in grouped.items():
        acsr = group.get("acsr_mlp_predicted_future")
        control = group.get("parameter_matched_causal_mlp_control")
        if not acsr or not control:
            continue
        acsr_ce = _float_or_none(acsr.get("ce_loss"))
        control_ce = _float_or_none(control.get("ce_loss"))
        acsr_regret = _float_or_none(acsr.get("oracle_regret"))
        control_regret = _float_or_none(control.get("oracle_regret"))
        if acsr_ce is None or control_ce is None:
            continue
        rows.append(
            {
                "source_dir": key[0],
                "seed": key[1],
                "status": key[2],
                "split": key[3],
                "holdout_start": key[4],
                "acsr_ce_loss": acsr_ce,
                "parameter_matched_ce_loss": control_ce,
                "acsr_minus_parameter_matched_ce_loss": acsr_ce - control_ce,
                "acsr_oracle_regret": acsr_regret if acsr_regret is not None else "",
                "parameter_matched_oracle_regret": (
                    control_regret if control_regret is not None else ""
                ),
                "acsr_minus_parameter_matched_oracle_regret": (
                    acsr_regret - control_regret
                    if acsr_regret is not None and control_regret is not None
                    else ""
                ),
            }
        )
    return rows


def _missing_required_controls(
    variant_rows: list[dict[str, Any]],
    same_student_rows: list[dict[str, Any]],
    perturbation_rows: list[dict[str, Any]],
    margin_rows: list[dict[str, Any]],
    parameter_rows: list[dict[str, Any]],
) -> list[str]:
    reasons = []
    variants = {row["variant"] for row in variant_rows}
    for required in REQUIRED_VARIANTS:
        if required not in variants:
            reasons.append(f"missing required variant {required}")
    if "margin_gated_acsr_proxy" not in variants:
        reasons.append("missing margin-gated ACSR variant")
    if not any(
        row.get("forcing_type") == "dual_student_cross_forcing"
        and row.get("status") == "available"
        for row in same_student_rows
    ):
        reasons.append("dual-student cross-forcing rows missing")
    if not any(
        row.get("control_type") == "leaky_future_positive"
        and row.get("status") == "available"
        and str(row.get("passed", "")).lower() == "true"
        for row in perturbation_rows
    ):
        reasons.append("leaky positive future-control rows missing")
    if not any(row.get("feature_noise_flip_rate") not in ("", None) for row in margin_rows):
        reasons.append("margin fragility under feature noise is missing")
    if not any(
        row.get("component") == "parameter_matched_causal_mlp_control"
        and row.get("status") == "available"
        for row in parameter_rows
    ):
        reasons.append("parameter-matched causal MLP control is missing")
    if not any(row.get("split") == "sequence_suffix_holdout" for row in variant_rows):
        reasons.append("sequence-heldout split is missing from source artifacts")
    return reasons


def _strategy_review_notes(strategy_review: Path | None) -> dict[str, Any]:
    if not strategy_review or not strategy_review.is_file():
        return {
            "path": str(strategy_review) if strategy_review else "",
            "status": "not_provided",
        }
    notes: dict[str, Any] = {
        "path": str(strategy_review),
        "status": "read",
        "recommendation_accepted": True,
        "deferred_or_rejected": [],
    }
    for line in strategy_review.read_text(encoding="utf-8").splitlines()[:20]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in {"strategic_change_level", "notify_ben", "recommended_next_action", "verdict"}:
            notes[key] = value.strip()
    return notes


def _direction_shift_record(review_notes: dict[str, Any]) -> str:
    if review_notes.get("strategic_change_level") == "major":
        notify = review_notes.get("notify_ben")
        return (
            "Major urgent-review pivot accepted: freeze ACSR-as-anticipation "
            "promotion/GPU repeats and pivot locally to a capacity-matched causal "
            f"support-router mechanism audit. Ben should be notified: {notify}."
        )
    return "No major direction shift; accepted broader local ACSR gate recommendation."


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    *,
    variant_rows: list[dict[str, Any]],
    same_student_rows: list[dict[str, Any]],
    perturbation_rows: list[dict[str, Any]],
    margin_rows: list[dict[str, Any]],
    parameter_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "variant_metrics.csv", variant_rows)
    _write_csv(out_dir / "same_student_cross_forcing.csv", same_student_rows)
    _write_csv(out_dir / "perturbation_metrics.csv", perturbation_rows)
    _write_csv(out_dir / "margin_fragility.csv", margin_rows)
    _write_csv(out_dir / "parameter_counts.csv", parameter_rows)
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ACSR Broader Mechanism Gate",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Loaded packets: `{summary['loaded_packet_count']}/{summary['source_packet_count']}`",
        "",
        "This gate aggregates existing command-generated ACSR packets and fails "
        "closed when broader mechanism controls are absent. It does not promote "
        "ACSR by default.",
        "",
        "## Support-Score Hook",
        "",
        "`relaleap.experiments.anticipatory_contextual_support_routing._score_from_features` "
        "computes `residual.contextual_column_scores(features)` plus the tie "
        "breaker; support is selected by `scores.topk(top_k)`. The current "
        "margin-gated row is a packet-level proxy, so token-level fallback still "
        "requires score tensors or an in-run implementation.",
    ]
    if summary.get("strategy_review", {}).get("status") == "read":
        lines.extend(
            [
                "",
                "## Strategy Review",
                "",
                f"- Strategic change level: `{summary['strategy_review'].get('strategic_change_level', '')}`",
                f"- Notify Ben: `{summary['strategy_review'].get('notify_ben', '')}`",
                "- Recommendation accepted: broader local ACSR gate before GPU repeat or dense-teacher revisit.",
            ]
        )
    aggregate = summary.get("aggregate_metrics", {})
    if not aggregate.get("acsr_beats_parameter_matched_causal_control", True):
        lines.extend(
            [
                "",
                "## Anticipation-Specific Blocker",
                "",
                "The parameter-matched direct causal MLP control is available and "
                "ACSR does not beat it on the paired CE/regret gate. This blocks "
                "the claim that predicted future-context features are currently "
                "needed for the observed support-routing gain.",
            ]
        )
    if summary.get("failures"):
        lines.extend(["", "## Fail-Closed Reasons"])
        for failure in summary["failures"]:
            lines.append(f"- `{failure.get('gate')}`: {failure.get('reason')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    else:
        fieldnames = ["status"]
        rows = [{"status": "missing"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row_by_variant(rows: list[dict[str, str]], variant: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("variant") == variant:
            return row
    return None


def _seed_label(summary: dict[str, Any]) -> str:
    config_path = str(summary.get("config_path", ""))
    if "seed2" in config_path or "seed_2" in config_path:
        return "seed2"
    return "seed1"


def _control_family(variant: str) -> str:
    if variant.startswith("acsr_"):
        return "acsr"
    if "shuffled" in variant:
        return "shuffled"
    if "token_position" in variant:
        return "token_position"
    if "zero" in variant or "mean" in variant:
        return "zero_mean"
    if "parameter_matched_causal" in variant:
        return "causal_contextual_parameter_matched"
    if "random" in variant or "fixed" in variant:
        return "random_fixed"
    if "causal" in variant:
        return "causal_contextual"
    if "teacher" in variant:
        return "nondeployable_teacher"
    return "other"


def _float(value: Any) -> float:
    return float(value)


def _float_or_blank(value: Any) -> float | str:
    if value in ("", None):
        return ""
    return float(value)


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _int_or_blank(value: Any) -> int | str:
    if value in ("", None):
        return ""
    return int(float(value))


def _mean_or_none(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--source-dir",
        type=Path,
        action="append",
        dest="source_dirs",
        help="Existing ACSR artifact directory. May be supplied multiple times.",
    )
    parser.add_argument("--strategy-review", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_acsr_broader_mechanism_gate(
        source_dirs=args.source_dirs,
        out_dir=args.out,
        strategy_review=args.strategy_review,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
