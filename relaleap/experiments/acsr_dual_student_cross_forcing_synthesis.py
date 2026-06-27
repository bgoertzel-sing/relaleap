"""Synthesize ACSR dual-student support cross-forcing rows."""

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
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/acsr_dual_student_cross_forcing_synthesis")

REQUIRED_SUPPORT_SOURCES = (
    "own",
    "partner",
    "token_position_null",
    "position_stratified_shuffled_null",
    "random_frequency_matched_null",
    "oracle_diagnostic",
    "full_context_teacher_diagnostic",
)
REQUIRED_VALUE_STUDENTS = (
    "acsr_student",
    "parameter_matched_direct_causal_mlp_student",
)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_packets.csv",
    "value_student_support_synthesis.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_acsr_dual_student_cross_forcing_synthesis(
    *,
    source_dirs: tuple[Path, ...] = DEFAULT_SOURCE_DIRS,
    strategy_review: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Interpret dual-student cross-forcing against null and diagnostic rows."""

    start = time.time()
    packets = [_load_packet(index + 1, path) for index, path in enumerate(source_dirs)]
    source_rows = [_source_packet_row(packet) for packet in packets]
    support_rows = _support_synthesis_rows(packets)
    review = _strategy_review(strategy_review)
    gate_rows = _gate_rows(source_rows, support_rows, review)
    failures = [
        {"gate": row["criterion"], "reason": row["failure_reason"]}
        for row in gate_rows
        if not row["passed"]
    ]
    status = "pass" if not failures else "fail"
    transfer_rows = [row for row in support_rows if row.get("status") == "available"]
    all_partner_beats_required_nulls = bool(transfer_rows) and all(
        row["partner_beats_token_position_null"]
        and row["partner_beats_shuffled_null"]
        and row["partner_beats_random_frequency_null"]
        for row in transfer_rows
    )
    residual_norm_available = False
    claim_status = (
        "cross_value_support_transfer_suggestive_not_established"
        if status == "pass" and all_partner_beats_required_nulls
        else "cross_value_support_transfer_not_established"
    )
    if status == "pass" and all_partner_beats_required_nulls and residual_norm_available:
        claim_status = "cross_value_support_transfer_supported_not_promoted"
    summary = {
        "status": status,
        "decision": (
            "acsr_dual_student_cross_forcing_synthesis_recorded"
            if status == "pass"
            else "acsr_dual_student_cross_forcing_synthesis_failed_closed"
        ),
        "claim_status": claim_status,
        "selected_next_step": _selected_next_step(status, all_partner_beats_required_nulls),
        "source_dirs": [str(path) for path in source_dirs],
        "strategy_review": review,
        "direction_shift": _direction_shift(review),
        "source_packets": source_rows,
        "value_student_support_synthesis": support_rows,
        "gate_criteria": gate_rows,
        "failures": failures,
        "aggregate_metrics": {
            "available_value_student_synthesis_count": len(transfer_rows),
            "mean_partner_delta_vs_token_position_null": _mean_key(
                transfer_rows, "partner_delta_vs_token_position_null"
            ),
            "mean_partner_delta_vs_own_support": _mean_key(
                transfer_rows, "partner_delta_vs_own_support"
            ),
            "mean_partner_oracle_headroom_recovered_fraction": _mean_key(
                transfer_rows, "partner_oracle_headroom_recovered_fraction"
            ),
            "all_partner_beats_required_nulls": all_partner_beats_required_nulls,
            "residual_norm_control_available": residual_norm_available,
        },
        "claim_boundaries": {
            "supported": [
                "dual-student cross-forcing rows are present and interpreted by value path",
                "partner support can now be compared against token-position, shuffled, random, oracle, teacher, and own-support rows",
            ],
            "not_supported": [
                "ACSR-as-anticipation",
                "default ACSR or direct-router promotion",
                "residual-norm-normalized value-invariant support transfer",
            ],
        },
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _load_packet(index: int, source_dir: Path) -> dict[str, Any]:
    path = source_dir / "dual_student_cross_forcing.csv"
    summary_path = source_dir / "summary.json"
    rows = _read_csv(path)
    summary = _read_json(summary_path)
    return {
        "packet": f"packet{index}",
        "seed": _seed_label(source_dir, summary),
        "source_dir": str(source_dir),
        "summary_present": summary_path.is_file(),
        "summary_status": summary.get("status", ""),
        "dual_student_path": str(path),
        "dual_student_present": path.is_file(),
        "dual_student_row_count": len(rows),
        "_rows": rows,
    }


def _source_packet_row(packet: dict[str, Any]) -> dict[str, Any]:
    rows = packet["_rows"]
    support_sources = sorted({row.get("support_source", "") for row in rows})
    value_students = sorted({row.get("value_student", "") for row in rows})
    return {
        "packet": packet["packet"],
        "seed": packet["seed"],
        "source_dir": packet["source_dir"],
        "summary_present": packet["summary_present"],
        "summary_status": packet["summary_status"],
        "dual_student_present": packet["dual_student_present"],
        "dual_student_row_count": packet["dual_student_row_count"],
        "value_students": ";".join(value_students),
        "support_sources": ";".join(support_sources),
    }


def _support_synthesis_rows(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for packet in packets:
        available_rows = [
            row for row in packet["_rows"] if row.get("status", "available") == "available"
        ]
        for value_student in REQUIRED_VALUE_STUDENTS:
            value_rows = [row for row in available_rows if row.get("value_student") == value_student]
            by_source = {row.get("support_source"): row for row in value_rows}
            missing = [source for source in REQUIRED_SUPPORT_SOURCES if source not in by_source]
            if missing:
                rows.append(
                    {
                        "packet": packet["packet"],
                        "seed": packet["seed"],
                        "source_dir": packet["source_dir"],
                        "value_student": value_student,
                        "status": "missing",
                        "missing_support_sources": ";".join(missing),
                    }
                )
                continue
            partner = by_source["partner"]
            own = by_source["own"]
            token = by_source["token_position_null"]
            shuffled = by_source["position_stratified_shuffled_null"]
            random_null = by_source["random_frequency_matched_null"]
            oracle = by_source["oracle_diagnostic"]
            teacher = by_source["full_context_teacher_diagnostic"]
            partner_ce = _number(partner.get("ce_loss"))
            token_ce = _number(token.get("ce_loss"))
            oracle_ce = _number(oracle.get("ce_loss"))
            row = {
                "packet": packet["packet"],
                "seed": packet["seed"],
                "source_dir": packet["source_dir"],
                "value_student": value_student,
                "eval_split": partner.get("eval_split", ""),
                "status": "available",
                "partner_support_variant": partner.get("support_variant", ""),
                "own_support_ce_loss": _number(own.get("ce_loss")),
                "partner_ce_loss": partner_ce,
                "token_position_null_ce_loss": token_ce,
                "shuffled_null_ce_loss": _number(shuffled.get("ce_loss")),
                "random_frequency_null_ce_loss": _number(random_null.get("ce_loss")),
                "oracle_diagnostic_ce_loss": oracle_ce,
                "teacher_diagnostic_ce_loss": _number(teacher.get("ce_loss")),
                "partner_delta_vs_own_support": _number(
                    partner.get("loss_delta_vs_own_support")
                ),
                "partner_delta_vs_token_position_null": _number(
                    partner.get("loss_delta_vs_token_position_null")
                ),
                "partner_delta_vs_shuffled_null": _delta_number(
                    partner.get("ce_loss"), shuffled.get("ce_loss")
                ),
                "partner_delta_vs_random_frequency_null": _delta_number(
                    partner.get("ce_loss"), random_null.get("ce_loss")
                ),
                "partner_delta_vs_teacher_diagnostic": _delta_number(
                    partner.get("ce_loss"), teacher.get("ce_loss")
                ),
                "partner_delta_vs_oracle_diagnostic": _delta_number(
                    partner.get("ce_loss"), oracle.get("ce_loss")
                ),
                "partner_oracle_regret": _number(partner.get("oracle_regret")),
                "partner_support_jaccard_with_own": _number(
                    partner.get("support_jaccard_with_own")
                ),
                "partner_topk_margin_bin": partner.get("topk_margin_bin", ""),
                "partner_oracle_headroom_recovered_fraction": _headroom_fraction(
                    partner_ce, token_ce, oracle_ce
                ),
            }
            row["partner_beats_token_position_null"] = _lt(
                row["partner_ce_loss"], row["token_position_null_ce_loss"]
            )
            row["partner_beats_shuffled_null"] = _lt(
                row["partner_ce_loss"], row["shuffled_null_ce_loss"]
            )
            row["partner_beats_random_frequency_null"] = _lt(
                row["partner_ce_loss"], row["random_frequency_null_ce_loss"]
            )
            row["partner_matches_or_beats_own_support"] = _le(
                row["partner_ce_loss"], row["own_support_ce_loss"]
            )
            rows.append(row)
    return rows


def _gate_rows(
    source_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    review: dict[str, Any],
) -> list[dict[str, Any]]:
    available = [row for row in support_rows if row.get("status") == "available"]
    expected_count = len(source_rows) * len(REQUIRED_VALUE_STUDENTS)
    return [
        _criterion(
            "strategy_review_consumed",
            review.get("status") == "read",
            "latest strategy review is read before interpreting cross-forcing",
            review.get("status", ""),
            "latest strategy review was not available/read",
        ),
        _criterion(
            "source_packets_present",
            bool(source_rows) and all(row["dual_student_present"] for row in source_rows),
            "all selected source packets contain dual_student_cross_forcing.csv",
            ";".join(str(row["dual_student_present"]) for row in source_rows),
            "one or more selected source packets are missing cross-forcing rows",
        ),
        _criterion(
            "required_value_student_rows_present",
            len(available) == expected_count,
            "each packet has available rows for both value students",
            f"available={len(available)} expected={expected_count}",
            "missing one or more required value-student support-source rows",
        ),
        _criterion(
            "partner_token_position_null_comparison_present",
            all(row.get("partner_delta_vs_token_position_null") is not None for row in available),
            "partner support has paired token-position-null deltas",
            f"available={len(available)}",
            "partner-vs-token-position-null deltas are missing",
        ),
        _criterion(
            "random_and_shuffled_nulls_present",
            all(
                row.get("partner_delta_vs_shuffled_null") is not None
                and row.get("partner_delta_vs_random_frequency_null") is not None
                for row in available
            ),
            "shuffled and random/frequency-matched null comparisons are available",
            f"available={len(available)}",
            "shuffled or random/frequency-matched null comparisons are missing",
        ),
    ]


def _criterion(
    criterion: str,
    passed: bool,
    requirement: str,
    observed: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "requirement": requirement,
        "observed": observed,
        "failure_reason": "" if passed else failure_reason,
    }


def _selected_next_step(status: str, all_partner_beats_required_nulls: bool) -> str:
    if status != "pass":
        return "repair missing dual-student source rows before interpreting support transfer"
    if all_partner_beats_required_nulls:
        return (
            "add residual-norm-normalized and per-token paired cross-forcing metrics "
            "before any causal support-router mechanism claim"
        )
    return (
        "inspect which value path or null comparison blocks support transfer before "
        "running GPU repeats"
    )


def _headroom_fraction(
    partner_ce: float | None,
    token_ce: float | None,
    oracle_ce: float | None,
) -> float | None:
    if partner_ce is None or token_ce is None or oracle_ce is None:
        return None
    denominator = token_ce - oracle_ce
    if denominator <= 0.0:
        return None
    return (token_ce - partner_ce) / denominator


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "status": "not_found"}
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        header[key.strip()] = value.strip()
    return {
        "path": str(path),
        "status": "read",
        "recommendation_accepted": True,
        "strategic_change_level": header.get("strategic_change_level", ""),
        "notify_ben": header.get("notify_ben", ""),
        "recommended_next_action": header.get("recommended_next_action", ""),
        "verdict": header.get("verdict", ""),
    }


def _direction_shift(review: dict[str, Any]) -> str:
    if review.get("strategic_change_level") == "major" or review.get("notify_ben") == "true":
        return (
            "GPT-5.5-Pro review requested a major or notify-Ben direction shift: "
            f"{review.get('recommended_next_action', '')} Ben should be notified: "
            f"{review.get('notify_ben', '')}."
        )
    return "No major strategy-review direction shift recorded for this synthesis."


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_packets.csv", summary["source_packets"])
    _write_csv(
        out_dir / "value_student_support_synthesis.csv",
        summary["value_student_support_synthesis"],
    )
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [{"status": "missing"}]
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["aggregate_metrics"]
    lines = [
        "# ACSR Dual-Student Cross-Forcing Synthesis",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Mean partner delta vs token-position null: `{metrics['mean_partner_delta_vs_token_position_null']}`",
        f"- Mean partner oracle-headroom fraction: `{metrics['mean_partner_oracle_headroom_recovered_fraction']}`",
        "",
        summary["direction_shift"],
        "",
        "Partner support is interpreted only inside the same value path and against "
        "the token-position, shuffled, random/frequency, oracle, teacher, and own "
        "support rows. Residual-norm-normalized evidence is still required before "
        "a value-invariant causal support-router mechanism claim.",
    ]
    if summary["failures"]:
        lines.extend(["", "## Fail-Closed Reasons"])
        for failure in summary["failures"]:
            lines.append(f"- `{failure['gate']}`: {failure['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _seed_label(source_dir: Path, summary: dict[str, Any]) -> str:
    config = str(summary.get("config_path", ""))
    text = f"{source_dir} {config}"
    if "seed2" in text or "seed_2" in text:
        return "seed2"
    if "seed3" in text or "seed_3" in text:
        return "seed3"
    return "seed1"


def _delta_number(left: Any, right: Any) -> float | None:
    left_number = _number(left)
    right_number = _number(right)
    if left_number is None or right_number is None:
        return None
    return left_number - right_number


def _number(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _lt(left: Any, right: Any) -> bool:
    return isinstance(left, (int, float)) and isinstance(right, (int, float)) and left < right


def _le(left: Any, right: Any) -> bool:
    return isinstance(left, (int, float)) and isinstance(right, (int, float)) and left <= right


def _mean_key(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return float(mean(values)) if values else None


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, action="append", default=None)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_acsr_dual_student_cross_forcing_synthesis(
        source_dirs=tuple(args.source_dir) if args.source_dir else DEFAULT_SOURCE_DIRS,
        strategy_review=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
