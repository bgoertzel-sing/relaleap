"""Repeat gate for the mechanism-factorized continual-learning probe."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.mechanism_factorized_continual_learning_probe import (
    run_mechanism_factorized_continual_learning_probe,
)


DEFAULT_SEED7_DIR = Path("results/reports/mechanism_factorized_continual_learning_probe")
DEFAULT_SEED11_DIR = Path("results/reports/mechanism_factorized_continual_learning_probe_seed11")
DEFAULT_OUT_DIR = Path("results/reports/mechanism_factorized_continual_learning_repeat")
DEFAULT_SEEDS = (7, 11)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "repeat_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_mechanism_factorized_continual_learning_repeat(
    *,
    seed_dirs: list[Path] | None = None,
    seeds: list[int] | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    steps_per_phase: int = 18,
    refresh_missing: bool = True,
) -> dict[str, Any]:
    """Run or consume two local seed reports and synthesize repeat gates."""

    start = time.time()
    seed_dirs = seed_dirs or [DEFAULT_SEED7_DIR, DEFAULT_SEED11_DIR]
    seeds = seeds or list(DEFAULT_SEEDS)
    if len(seed_dirs) != len(seeds):
        raise ValueError("seed_dirs and seeds must have the same length")

    packets: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for seed, seed_dir in zip(seeds, seed_dirs):
        summary_path = seed_dir / "summary.json"
        if refresh_missing and not summary_path.is_file():
            run_mechanism_factorized_continual_learning_probe(
                out_dir=seed_dir,
                seed=seed,
                steps_per_phase=steps_per_phase,
            )
        packet = _read_json(summary_path)
        packets.append(packet)
        source_rows.append(_source_row(seed, summary_path, packet))

    repeat_rows = [_repeat_row(seed, packet) for seed, packet in zip(seeds, packets)]
    gate_rows = _gate_rows(repeat_rows, source_rows)
    hard_fail = any(not row["passed"] and row["severity"] == "hard" for row in gate_rows)
    repeat_topk2_supported = _gate_passed(gate_rows, "topk2_tradeoff_survives_all_seeds")
    repeat_claim_supported = repeat_topk2_supported and _gate_passed(
        gate_rows,
        "full_sparse_claim_survives_all_seeds",
    )
    summary = {
        "status": "fail" if hard_fail else "pass",
        "decision": (
            "mechanism_factorized_cl_repeat_failed_closed"
            if hard_fail
            else "mechanism_factorized_cl_second_seed_repeat_recorded"
        ),
        "claim_status": (
            "mechanism_factorized_sparse_retention_candidate_supported_not_promoted"
            if repeat_claim_supported
            else "mechanism_factorized_sparse_retention_not_established"
        ),
        "topk2_tradeoff_repeat_status": (
            "survives_second_seed" if repeat_topk2_supported else "not_replicated"
        ),
        "promotion_allowed": False,
        "requires_gpu_now": False,
        "backend_policy": "local CPU second-seed mechanism screen; no RunPod/Colab validation used",
        "seeds": seeds,
        "seed_dirs": [str(path) for path in seed_dirs],
        "repeat_row_count": len(repeat_rows),
        "source_rows": source_rows,
        "repeat_metrics": repeat_rows,
        "gate_criteria": gate_rows,
        "primary_result": _primary_result(repeat_rows),
        "selected_next_step": _selected_next_step(
            hard_fail=hard_fail,
            repeat_topk2_supported=repeat_topk2_supported,
            repeat_claim_supported=repeat_claim_supported,
        ),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _source_row(seed: int, path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": seed,
        "path": str(path),
        "present": bool(packet),
        "status": packet.get("status") if packet else "missing",
        "decision": packet.get("decision") if packet else "",
        "claim_status": packet.get("claim_status") if packet else "",
        "selected_next_step": packet.get("selected_next_step") if packet else "",
    }


def _repeat_row(seed: int, packet: dict[str, Any]) -> dict[str, Any]:
    gate_by_name = {
        str(row.get("criterion")): row
        for row in packet.get("gate_criteria", [])
        if isinstance(row, dict)
    }
    primary = packet.get("primary_result", {}) if isinstance(packet.get("primary_result"), dict) else {}
    return {
        "seed": seed,
        "status": packet.get("status"),
        "decision": packet.get("decision"),
        "claim_status": packet.get("claim_status"),
        "topk1_target_gate_passed": _passed(gate_by_name, "topk1_target_adaptation_no_worse_than_dense"),
        "topk1_off_target_kl_gate_passed": _passed(gate_by_name, "topk1_off_target_kl_no_worse_than_dense"),
        "topk1_forgetting_gate_passed": _passed(gate_by_name, "topk1_forgetting_no_worse_than_dense"),
        "topk2_interference_per_gain_gate_passed": _passed(
            gate_by_name,
            "topk2_interference_per_gain_no_worse_than_dense",
        ),
        "topk2_random_null_tradeoff_gate_passed": _passed(
            gate_by_name,
            "topk2_beats_random_support_tradeoff_null",
        ),
        "anchor_kl_sparse_gate_passed": _passed(
            gate_by_name,
            "anchor_kl_sparse_no_worse_than_dense_anchor_kl",
        ),
        "topk1_minus_dense_target_delta": primary.get("topk1_minus_dense_mean_target_ce_delta"),
        "topk1_minus_dense_off_target_kl": primary.get("topk1_minus_dense_mean_off_target_kl"),
        "topk2_minus_dense_forgetting_per_target_improvement": primary.get(
            "topk2_minus_dense_forgetting_per_target_improvement"
        ),
        "topk2_minus_dense_mean_final_forgetting": primary.get(
            "topk2_minus_dense_mean_final_forgetting"
        ),
    }


def _gate_rows(
    repeat_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _criterion(
            "required_seed_reports_present",
            all(row["present"] for row in source_rows) and len(source_rows) >= 2,
            "hard",
            "at least two seed summaries must be present",
            [row["path"] for row in source_rows if row["present"]],
            "missing seed summary artifact",
        ),
        _criterion(
            "all_seed_reports_pass_runtime_schema",
            all(row["status"] == "pass" for row in source_rows),
            "hard",
            "each seed report must pass runtime/schema checks",
            [row["status"] for row in source_rows],
            "at least one seed report failed",
        ),
        _criterion(
            "topk2_tradeoff_survives_all_seeds",
            all(
                row["topk2_interference_per_gain_gate_passed"]
                and row["topk2_random_null_tradeoff_gate_passed"]
                for row in repeat_rows
            ),
            "claim",
            "top-k2 must beat dense and random-support tradeoff gates on every seed",
            [
                {
                    "seed": row["seed"],
                    "dense_gate": row["topk2_interference_per_gain_gate_passed"],
                    "random_null_gate": row["topk2_random_null_tradeoff_gate_passed"],
                }
                for row in repeat_rows
            ],
            "top-k2 tradeoff did not replicate across all seeds",
        ),
        _criterion(
            "full_sparse_claim_survives_all_seeds",
            all(
                row["topk1_target_gate_passed"]
                and row["topk1_off_target_kl_gate_passed"]
                and row["topk1_forgetting_gate_passed"]
                and row["anchor_kl_sparse_gate_passed"]
                for row in repeat_rows
            ),
            "claim",
            "all sparse retention/adaptation guardrails must pass on every seed",
            [
                {
                    "seed": row["seed"],
                    "topk1_target": row["topk1_target_gate_passed"],
                    "topk1_off_target_kl": row["topk1_off_target_kl_gate_passed"],
                    "topk1_forgetting": row["topk1_forgetting_gate_passed"],
                    "anchor_kl_sparse": row["anchor_kl_sparse_gate_passed"],
                }
                for row in repeat_rows
            ],
            "full sparse-retention claim remains blocked by at least one seed/gate",
        ),
    ]


def _primary_result(repeat_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "seed_count": len(repeat_rows),
        "topk2_tradeoff_supporting_seed_count": sum(
            1
            for row in repeat_rows
            if row["topk2_interference_per_gain_gate_passed"]
            and row["topk2_random_null_tradeoff_gate_passed"]
        ),
        "full_sparse_claim_supporting_seed_count": sum(
            1
            for row in repeat_rows
            if row["topk1_target_gate_passed"]
            and row["topk1_off_target_kl_gate_passed"]
            and row["topk1_forgetting_gate_passed"]
            and row["anchor_kl_sparse_gate_passed"]
        ),
        "mean_topk2_minus_dense_forgetting_per_target_improvement": _mean(
            [row["topk2_minus_dense_forgetting_per_target_improvement"] for row in repeat_rows]
        ),
        "mean_topk1_minus_dense_target_delta": _mean(
            [row["topk1_minus_dense_target_delta"] for row in repeat_rows]
        ),
        "interpretation": (
            "Negative top-k2-minus-dense tradeoff deltas favor top-k2. "
            "This is repeat-local mechanism evidence, not GPU or promotion evidence."
        ),
    }


def _selected_next_step(
    *,
    hard_fail: bool,
    repeat_topk2_supported: bool,
    repeat_claim_supported: bool,
) -> str:
    if hard_fail:
        return "repair_missing_or_failed_mechanism_factorized_cl_seed_report"
    if repeat_claim_supported:
        return "request_strategy_review_before_any_gpu_validation_or_promotion_claim"
    if repeat_topk2_supported:
        return "use_repeated_topk2_tradeoff_signal_to_design_stricter_dense_null_controlled_interference_mitigation"
    return "stop_mechanism_factorized_sparse_retention_branch_and_pivot_to_commutator_or_dense_teacher_probe"


def _criterion(
    criterion: str,
    passed: bool,
    severity: str,
    requirement: str,
    observed: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "severity": severity,
        "requirement": requirement,
        "observed": observed,
        "failure_reason": "" if passed else failure_reason,
    }


def _passed(gate_by_name: dict[str, dict[str, Any]], criterion: str) -> bool:
    return bool(gate_by_name.get(criterion, {}).get("passed"))


def _gate_passed(gate_rows: list[dict[str, Any]], criterion: str) -> bool:
    return any(row["criterion"] == criterion and row["passed"] for row in gate_rows)


def _mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None and value != ""]
    if not numeric:
        return None
    return sum(numeric) / float(len(numeric))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "repeat_metrics.csv", summary["repeat_metrics"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    result = summary["primary_result"]
    lines = [
        "# Mechanism-Factorized CL Second-Seed Repeat",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Top-k2 tradeoff repeat status: `{summary['topk2_tradeoff_repeat_status']}`",
        f"- Seeds: `{summary['seeds']}`",
        f"- Top-k2 supporting seeds: `{result['topk2_tradeoff_supporting_seed_count']}` / `{result['seed_count']}`",
        f"- Full sparse-claim supporting seeds: `{result['full_sparse_claim_supporting_seed_count']}` / `{result['seed_count']}`",
        f"- Mean top-k2 minus dense tradeoff delta: `{result['mean_topk2_minus_dense_forgetting_per_target_improvement']}`",
        "",
        "Promotion remains blocked. This report only tests whether the local mechanism-factorized top-k2 tradeoff signal survives a second seed.",
        "",
        "## Next Step",
        "",
        str(summary["selected_next_step"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed7-dir", type=Path, default=DEFAULT_SEED7_DIR)
    parser.add_argument("--seed11-dir", type=Path, default=DEFAULT_SEED11_DIR)
    parser.add_argument("--steps-per-phase", type=int, default=18)
    parser.add_argument("--no-refresh-missing", action="store_true")
    args = parser.parse_args(argv)
    summary = run_mechanism_factorized_continual_learning_repeat(
        seed_dirs=[args.seed7_dir, args.seed11_dir],
        seeds=list(DEFAULT_SEEDS),
        out_dir=args.out,
        steps_per_phase=args.steps_per_phase,
        refresh_missing=not args.no_refresh_missing,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
