"""Capacity-matched causal-router audit for ACSR source packets."""

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


DEFAULT_SOURCE_DIRS = (
    Path("results/audits/token_larger_anticipatory_contextual_support_routing"),
    Path("results/audits/token_larger_anticipatory_contextual_support_routing_seed2"),
)
DEFAULT_GATE_DIR = Path("results/audits/acsr_broader_mechanism_gate_local")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/audits/acsr_causal_router_capacity_audit_local")
PRIMARY_VARIANT = "acsr_mlp_predicted_future"
CONTROL_VARIANT = "parameter_matched_causal_mlp_control"
REQUIRED_FILES = (
    "summary.json",
    "router_metrics.csv",
    "same_student_metrics.csv",
    "support_agreement.csv",
    "sequence_heldout_metrics.csv",
    "margin_fragility.csv",
    "parameter_counts.csv",
)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "packet_status.csv",
    "paired_capacity_deltas.csv",
    "parameter_match.csv",
    "same_student_capacity.csv",
    "support_agreement.csv",
    "margin_fragility_capacity.csv",
    "missing_mechanism_evidence.csv",
    "notes.md",
)


def run_acsr_causal_router_capacity_audit(
    *,
    source_dirs: tuple[Path, ...] = DEFAULT_SOURCE_DIRS,
    gate_dir: Path = DEFAULT_GATE_DIR,
    strategy_review: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Audit whether ACSR beats its capacity-matched direct causal-router null."""

    start = time.time()
    packets = [_load_packet(index + 1, source_dir) for index, source_dir in enumerate(source_dirs)]
    paired_rows = _paired_capacity_deltas(packets)
    parameter_rows = _parameter_rows(packets)
    same_student_rows = _same_student_rows(packets)
    support_agreement_rows = _support_agreement_rows(packets)
    margin_rows = _margin_rows(packets)
    missing_rows = _missing_mechanism_rows(
        packets,
        gate_dir,
        same_student_rows=same_student_rows,
        support_agreement_rows=support_agreement_rows,
    )
    review = _strategy_review_notes(strategy_review)

    failures = _failures(packets, paired_rows, parameter_rows, missing_rows)
    status = "pass" if not failures else "fail"
    acsr_wins = [
        row
        for row in paired_rows
        if row["acsr_minus_parameter_matched_ce_loss"] < 0.0
        and row["acsr_minus_parameter_matched_oracle_regret"] < 0.0
    ]
    aggregate = {
        "paired_delta_count": len(paired_rows),
        "acsr_strict_win_count": len(acsr_wins),
        "mean_acsr_minus_parameter_matched_ce_loss": _mean_key(
            paired_rows, "acsr_minus_parameter_matched_ce_loss"
        ),
        "mean_acsr_minus_parameter_matched_oracle_regret": _mean_key(
            paired_rows, "acsr_minus_parameter_matched_oracle_regret"
        ),
        "mean_parameter_count_ratio_to_acsr_path": _mean_key(
            parameter_rows, "parameter_count_ratio_to_acsr_path"
        ),
        "mean_same_student_acsr_minus_parameter_matched_ce_loss": _mean_key(
            same_student_rows, "acsr_minus_control_ce_loss"
        ),
        "support_agreement_available": any(
            row["evidence"] == "support_agreement" and row["status"] == "available"
            for row in missing_rows
        ),
        "dual_student_cross_forcing_available": any(
            row["evidence"] == "dual_student_cross_forcing" and row["status"] == "available"
            for row in missing_rows
        ),
    }
    summary = {
        "status": status,
        "decision": (
            "acsr_capacity_matched_causal_router_audit_passed"
            if status == "pass"
            else "acsr_capacity_matched_causal_router_audit_failed_closed"
        ),
        "claim_status": (
            "acsr_beats_capacity_matched_causal_router_not_promoted"
            if status == "pass"
            else "acsr_as_anticipation_blocked_by_capacity_matched_causal_router"
        ),
        "selected_next_step": (
            "implement dual-student support cross-forcing with stored support tensors"
            if not aggregate["dual_student_cross_forcing_available"]
            else "run paired sequence bootstrap over ACSR versus causal-router deltas"
        ),
        "source_dirs": [str(path) for path in source_dirs],
        "gate_dir": str(gate_dir),
        "strategy_review": review,
        "direction_shift": _direction_shift(review),
        "aggregate_metrics": aggregate,
        "failures": failures,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(
        out_dir,
        summary,
        packet_rows=[_public_packet(packet) for packet in packets],
        paired_rows=paired_rows,
        parameter_rows=parameter_rows,
        same_student_rows=same_student_rows,
        support_agreement_rows=support_agreement_rows,
        margin_rows=margin_rows,
        missing_rows=missing_rows,
    )
    return summary


def _load_packet(index: int, source_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (source_dir / name).is_file()]
    packet = {
        "packet": f"packet{index}",
        "source_dir": str(source_dir),
        "present": source_dir.is_dir(),
        "required_files_present": not missing,
        "missing_files": ";".join(missing),
        "summary_status": "",
        "config_path": "",
        "_summary": {},
        "_router_rows": [],
        "_same_student_rows": [],
        "_support_agreement_rows": [],
        "_sequence_rows": [],
        "_margin_rows": [],
        "_parameter_rows": [],
    }
    if missing:
        return packet
    summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    packet.update(
        {
            "summary_status": summary.get("status", ""),
            "config_path": summary.get("config_path", ""),
            "_summary": summary,
            "_router_rows": _read_csv(source_dir / "router_metrics.csv"),
            "_same_student_rows": _read_csv(source_dir / "same_student_metrics.csv"),
            "_support_agreement_rows": _read_csv(source_dir / "support_agreement.csv"),
            "_sequence_rows": _read_csv(source_dir / "sequence_heldout_metrics.csv"),
            "_margin_rows": _read_csv(source_dir / "margin_fragility.csv"),
            "_parameter_rows": _read_csv(source_dir / "parameter_counts.csv"),
        }
    )
    return packet


def _paired_capacity_deltas(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for packet in packets:
        if not packet["required_files_present"]:
            continue
        seed = _seed_label(packet)
        rows.extend(
            _paired_rows_for_split(
                packet=packet,
                seed=seed,
                split="fixed_context",
                source_rows=packet["_router_rows"],
            )
        )
        rows.extend(
            _paired_rows_for_split(
                packet=packet,
                seed=seed,
                split="sequence_suffix_holdout",
                source_rows=packet["_sequence_rows"],
            )
        )
    return rows


def _paired_rows_for_split(
    *,
    packet: dict[str, Any],
    seed: str,
    split: str,
    source_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    by_variant = {row.get("variant"): row for row in source_rows}
    acsr = by_variant.get(PRIMARY_VARIANT)
    control = by_variant.get(CONTROL_VARIANT)
    if not acsr or not control:
        return []
    acsr_ce = _number(acsr.get("ce_loss"))
    control_ce = _number(control.get("ce_loss"))
    acsr_regret = _number(acsr.get("oracle_regret"))
    control_regret = _number(control.get("oracle_regret"))
    if None in {acsr_ce, control_ce, acsr_regret, control_regret}:
        return []
    return [
        {
            "packet": packet["packet"],
            "source_dir": packet["source_dir"],
            "seed": seed,
            "split": split,
            "holdout_start": acsr.get("holdout_start", ""),
            "acsr_ce_loss": acsr_ce,
            "parameter_matched_ce_loss": control_ce,
            "acsr_minus_parameter_matched_ce_loss": acsr_ce - control_ce,
            "acsr_oracle_regret": acsr_regret,
            "parameter_matched_oracle_regret": control_regret,
            "acsr_minus_parameter_matched_oracle_regret": acsr_regret - control_regret,
            "acsr_strictly_better": acsr_ce < control_ce and acsr_regret < control_regret,
        }
    ]


def _parameter_rows(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for packet in packets:
        if not packet["required_files_present"]:
            continue
        for row in packet["_parameter_rows"]:
            if row.get("component") != CONTROL_VARIANT:
                continue
            rows.append(
                {
                    "packet": packet["packet"],
                    "source_dir": packet["source_dir"],
                    "seed": _seed_label(packet),
                    "component": row.get("component"),
                    "status": row.get("status", ""),
                    "stored_parameter_count": _number(row.get("stored_parameter_count")),
                    "active_parameter_count": _number(row.get("active_parameter_count")),
                    "parameter_count_ratio_to_acsr_path": _number(
                        row.get("parameter_count_ratio_to_acsr_path")
                    ),
                    "basis": row.get("basis", ""),
                }
            )
    return rows


def _same_student_rows(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    suffix = f"{PRIMARY_VARIANT}_support_vs_{CONTROL_VARIANT}"
    for packet in packets:
        if not packet["required_files_present"]:
            continue
        for row in packet["_same_student_rows"]:
            if row.get("comparison") != suffix:
                continue
            rows.append(
                {
                    "packet": packet["packet"],
                    "source_dir": packet["source_dir"],
                    "seed": _seed_label(packet),
                    "forcing_type": row.get("forcing_type", "same_student"),
                    "status": row.get("status", "available"),
                    "target_student": row.get("target_student", ""),
                    "comparison": row.get("comparison"),
                    "acsr_forced_ce_loss": _number(row.get("acsr_forced_ce_loss")),
                    "control_forced_ce_loss": _number(row.get("control_forced_ce_loss")),
                    "acsr_minus_control_ce_loss": _number(
                        row.get("acsr_minus_control_ce_loss")
                    ),
                }
            )
    return rows


def _support_agreement_rows(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    suffix = f"{PRIMARY_VARIANT}_support_vs_{CONTROL_VARIANT}"
    for packet in packets:
        if not packet["required_files_present"]:
            continue
        for row in packet["_support_agreement_rows"]:
            if row.get("comparison") != suffix:
                continue
            rows.append(
                {
                    "packet": packet["packet"],
                    "source_dir": packet["source_dir"],
                    "seed": _seed_label(packet),
                    "comparison": row.get("comparison"),
                    "status": row.get("status", "available"),
                    "slot_match_fraction": _number(row.get("slot_match_fraction")),
                    "set_match_fraction": _number(row.get("set_match_fraction")),
                    "changed_support_fraction": _number(row.get("changed_support_fraction")),
                }
            )
    return rows


def _margin_rows(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for packet in packets:
        if not packet["required_files_present"]:
            continue
        by_variant = {row.get("variant"): row for row in packet["_margin_rows"]}
        acsr = by_variant.get(PRIMARY_VARIANT, {})
        control = by_variant.get(CONTROL_VARIANT, {})
        rows.append(
            {
                "packet": packet["packet"],
                "source_dir": packet["source_dir"],
                "seed": _seed_label(packet),
                "acsr_mean_topk_margin": _number(acsr.get("mean_topk_margin")),
                "parameter_matched_mean_topk_margin": _number(
                    control.get("mean_topk_margin")
                ),
                "acsr_feature_noise_flip_rate": _number(
                    acsr.get("feature_noise_flip_rate")
                ),
                "parameter_matched_feature_noise_flip_rate": _number(
                    control.get("feature_noise_flip_rate")
                ),
            }
        )
    return rows


def _missing_mechanism_rows(
    packets: list[dict[str, Any]],
    gate_dir: Path,
    *,
    same_student_rows: list[dict[str, Any]],
    support_agreement_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    gate_summary = _read_json_object(gate_dir / "summary.json")
    gate_status = gate_summary.get("gates", {})
    dual_available = bool(gate_status.get("dual_student_cross_forcing_available")) or any(
        row.get("forcing_type") == "dual_student_cross_forcing"
        and row.get("status") == "available"
        for row in same_student_rows
    )
    support_agreement_available = bool(support_agreement_rows)
    rows = [
        {
            "evidence": "dual_student_cross_forcing",
            "status": "available" if dual_available else "missing",
            "reason": (
                "dual-student cross-forcing rows are available"
                if dual_available
                else "existing packets do not store independent student values/support tensors"
            ),
        },
        {
            "evidence": "support_agreement",
            "status": "available" if support_agreement_available else "missing",
            "reason": (
                "source packets expose ACSR/control support agreement rows"
                if support_agreement_available
                else (
                    "source packets expose support counts and unique support sets, "
                    "but not per-token ACSR/control support indices needed for agreement"
                )
            ),
        },
        {
            "evidence": "per_sequence_bootstrap",
            "status": "missing",
            "reason": "source packets expose aggregate sequence-heldout deltas, not per-sequence paired losses",
        },
    ]
    if not any(packet["required_files_present"] for packet in packets):
        rows.append(
            {
                "evidence": "source_packets",
                "status": "missing",
                "reason": "no complete ACSR packet was available",
            }
        )
    return rows


def _failures(
    packets: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    parameter_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures = []
    for packet in packets:
        if not packet["required_files_present"]:
            failures.append(
                {
                    "gate": "source_packet",
                    "source_dir": packet["source_dir"],
                    "reason": f"missing required files: {packet['missing_files']}",
                }
            )
        elif packet["summary_status"] != "pass":
            failures.append(
                {
                    "gate": "source_packet",
                    "source_dir": packet["source_dir"],
                    "reason": f"source packet status is {packet['summary_status']}",
                }
            )
    if not paired_rows:
        failures.append(
            {
                "gate": "paired_capacity_delta",
                "reason": "ACSR/control paired CE and oracle-regret rows are missing",
            }
        )
    if paired_rows and not all(row["acsr_strictly_better"] for row in paired_rows):
        failures.append(
            {
                "gate": "capacity_matched_causal_router",
                "reason": "acsr_not_strictly_better_than_parameter_matched_causal_mlp",
            }
        )
    if not parameter_rows:
        failures.append(
            {
                "gate": "parameter_match",
                "reason": "parameter-matched causal MLP count row is missing",
            }
        )
    for row in missing_rows:
        if row["evidence"] in {"dual_student_cross_forcing", "support_agreement"} and row[
            "status"
        ] != "available":
            failures.append(
                {
                    "gate": row["evidence"],
                    "reason": row["reason"],
                }
            )
    return failures


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    *,
    packet_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    parameter_rows: list[dict[str, Any]],
    same_student_rows: list[dict[str, Any]],
    support_agreement_rows: list[dict[str, Any]],
    margin_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "packet_status.csv", packet_rows)
    _write_csv(out_dir / "paired_capacity_deltas.csv", paired_rows)
    _write_csv(out_dir / "parameter_match.csv", parameter_rows)
    _write_csv(out_dir / "same_student_capacity.csv", same_student_rows)
    _write_csv(out_dir / "support_agreement.csv", support_agreement_rows)
    _write_csv(out_dir / "margin_fragility_capacity.csv", margin_rows)
    _write_csv(out_dir / "missing_mechanism_evidence.csv", missing_rows)
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    aggregate = summary["aggregate_metrics"]
    lines = [
        "# ACSR Causal-Router Capacity Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Mean ACSR-minus-control CE: `{aggregate['mean_acsr_minus_parameter_matched_ce_loss']}`",
        f"- Mean ACSR-minus-control oracle regret: `{aggregate['mean_acsr_minus_parameter_matched_oracle_regret']}`",
        "",
        "This report treats the parameter-matched direct causal MLP as the primary "
        "stronger null for ACSR-as-anticipation. ACSR must beat that control on "
        "paired CE and oracle regret before any anticipation-specific claim is "
        "reopened.",
    ]
    if summary["strategy_review"].get("strategic_change_level") == "major":
        lines.extend(
            [
                "",
                "## Direction Shift",
                "",
                summary["direction_shift"],
            ]
        )
    if summary["failures"]:
        lines.extend(["", "## Fail-Closed Reasons"])
        for failure in summary["failures"]:
            lines.append(f"- `{failure['gate']}`: {failure['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _strategy_review_notes(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "status": "not_found"}
    notes = {"path": str(path), "status": "read", "recommendation_accepted": True}
    for line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in {"strategic_change_level", "notify_ben", "recommended_next_action", "verdict"}:
            notes[key] = value.strip()
    return notes


def _direction_shift(review: dict[str, Any]) -> str:
    if review.get("strategic_change_level") == "major":
        return (
            "Major GPT-5.5-Pro review pivot accepted: freeze ACSR-as-anticipation "
            "promotion/GPU repeats, keep ACSR non-promoted, and audit the "
            "capacity-matched causal support-router mechanism locally. "
            f"Ben should be notified: {review.get('notify_ben')}."
        )
    return "No major strategy-review direction shift recorded for this audit."


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [{"status": "missing"}]
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _public_packet(packet: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in packet.items() if not key.startswith("_")}


def _seed_label(packet: dict[str, Any]) -> str:
    config_path = str(packet.get("config_path", ""))
    source_dir = str(packet.get("source_dir", ""))
    if "seed2" in config_path or "seed2" in source_dir or "seed_2" in config_path:
        return "seed2"
    return "seed1"


def _number(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_key(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
    return float(mean(values)) if values else None


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # pragma: no cover - environment dependent
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gate-dir", type=Path, default=DEFAULT_GATE_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument(
        "--source-dir",
        type=Path,
        action="append",
        default=None,
        help="ACSR source packet directory. May be provided multiple times.",
    )
    args = parser.parse_args()
    summary = run_acsr_causal_router_capacity_audit(
        source_dirs=tuple(args.source_dir) if args.source_dir else DEFAULT_SOURCE_DIRS,
        gate_dir=args.gate_dir,
        strategy_review=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
