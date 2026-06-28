"""Synthesize tiny local core/periphery PC-column pilot repeats."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_PILOT_DIRS = (
    Path("results/reports/core_periphery_pc_column_pilot"),
    Path("results/reports/core_periphery_pc_column_pilot_seed11"),
)
DEFAULT_OUT_DIR = Path("results/reports/core_periphery_pc_column_synthesis")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_status.csv",
    "seed_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)

REQUIRED_PRIMARY_METRICS = (
    "core_periphery_update_norm_ratio",
    "core_minus_dense_anchor_mse_drift",
    "core_minus_mlp_anchor_mse_drift",
    "periphery_first_minus_core_first_prune_delta",
)


def run_core_periphery_pc_column_synthesis(
    *,
    pilot_dirs: tuple[Path, ...] = DEFAULT_PILOT_DIRS,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed synthesis report from local pilot artifacts."""

    start = time.time()
    packets = [_source_packet(path) for path in pilot_dirs]
    source_rows = [_source_row(packet) for packet in packets]
    seed_rows = [_seed_row(packet) for packet in packets]
    gate_rows = _gate_rows(source_rows, seed_rows)
    hard_failures = [row for row in gate_rows if not row["passed"] and row["severity"] == "hard"]
    claim_failures = [row for row in gate_rows if not row["passed"] and row["severity"] != "hard"]

    if hard_failures:
        status = "fail"
        decision = "core_periphery_pc_column_synthesis_failed_closed"
        claim_status = "source_artifacts_missing_or_failed"
        scientific_gate = "blocked"
        selected_next_step = "repair missing or failing local core/periphery pilot artifacts"
    elif claim_failures:
        status = "pass"
        decision = "core_periphery_pc_column_synthesis_recorded_but_blocked"
        claim_status = "local_repeat_signal_insufficient_for_design_progress"
        scientific_gate = "blocked"
        selected_next_step = "tighten the synthetic split mechanism or add another local repeat before GPU work"
    else:
        status = "pass"
        decision = "core_periphery_pc_column_local_repeats_supported"
        claim_status = "synthetic_local_repeat_only_not_gpu_or_promotion_evidence"
        scientific_gate = "ready_for_non_synthetic_pilot_design"
        selected_next_step = (
            "design a non-synthetic command-driven core/periphery PC-column pilot "
            "that consumes frozen hidden states and preserves dense/MLP/null controls"
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "scientific_gate": scientific_gate,
        "requires_gpu_now": False,
        "backend_policy": (
            "RunPod/Colab validation remains blocked; this synthesis only permits a "
            "non-synthetic local pilot design when all local repeat gates pass"
        ),
        "selected_next_step": selected_next_step,
        "source_status": source_rows,
        "seed_metrics": seed_rows,
        "gate_criteria": gate_rows,
        "failures": hard_failures + claim_failures,
        "aggregate_metrics": _aggregate_metrics(seed_rows),
        "interpretation": _interpretation(scientific_gate),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "generated_from_head": _git_commit(),
        "dirty_diff_hash": _dirty_diff_hash(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _source_packet(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    return {
        "dir": path,
        "summary_path": summary_path,
        "summary": _read_json(summary_path),
    }


def _source_row(packet: dict[str, Any]) -> dict[str, Any]:
    summary = packet["summary"]
    gate_rows = list(summary.get("gate_criteria") or [])
    failed_gates = [row.get("criterion") for row in gate_rows if not row.get("passed")]
    return {
        "source_dir": str(packet["dir"]),
        "summary_path": str(packet["summary_path"]),
        "present": packet["summary_path"].is_file(),
        "status": summary.get("status"),
        "decision": summary.get("decision"),
        "scientific_gate": summary.get("scientific_gate"),
        "claim_status": summary.get("claim_status"),
        "seed": summary.get("seed"),
        "failed_gate_count": len(failed_gates),
        "failed_gates": ",".join(str(item) for item in failed_gates),
    }


def _seed_row(packet: dict[str, Any]) -> dict[str, Any]:
    summary = packet["summary"]
    primary = dict(summary.get("primary_result") or {})
    row = {
        "source_dir": str(packet["dir"]),
        "seed": summary.get("seed"),
        "steps_per_task": summary.get("steps_per_task"),
    }
    for metric in REQUIRED_PRIMARY_METRICS:
        row[metric] = primary.get(metric)
    return row


def _gate_rows(source_rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seeds = {row.get("seed") for row in seed_rows if row.get("seed") is not None}
    return [
        _criterion(
            "two_distinct_local_sources_present",
            len(source_rows) >= 2 and all(row["present"] for row in source_rows) and len(seeds) >= 2,
            "hard",
            "at least two distinct local pilot summary artifacts exist",
            {"source_count": len(source_rows), "seeds": sorted(seeds)},
            "rerun the default and seed-11 local pilots before synthesis",
        ),
        _criterion(
            "all_sources_passed",
            all(row["status"] == "pass" for row in source_rows),
            "hard",
            "every source pilot has status pass",
            [{"source": row["source_dir"], "status": row["status"]} for row in source_rows],
            "repair failing pilot source artifacts",
        ),
        _criterion(
            "all_sources_ready_for_repeat_only",
            all(row["scientific_gate"] == "ready_for_repeat_only" for row in source_rows),
            "hard",
            "every source remains a local repeat-only candidate",
            [{"source": row["source_dir"], "scientific_gate": row["scientific_gate"]} for row in source_rows],
            "do not synthesize artifacts whose scientific gate changed unexpectedly",
        ),
        _criterion(
            "all_source_gates_passed",
            all(int(row["failed_gate_count"]) == 0 for row in source_rows),
            "hard",
            "no pilot hard or claim gate failed in any source",
            [{"source": row["source_dir"], "failed_gates": row["failed_gates"]} for row in source_rows],
            "interpretation blocked by failed source gate",
        ),
        _criterion(
            "required_primary_metrics_present",
            all(_has_required_metrics(row) for row in seed_rows),
            "hard",
            "every seed exposes retention, plasticity, and pruning primary metrics",
            seed_rows,
            "regenerate pilot artifacts with the current schema",
        ),
        _criterion(
            "repeat_update_separation_consistent",
            all(_float(row.get("core_periphery_update_norm_ratio")) > 1.5 for row in seed_rows),
            "claim",
            "periphery update norm exceeds core update norm on every repeat",
            [row.get("core_periphery_update_norm_ratio") for row in seed_rows],
            "split may be accounting-only across repeats",
        ),
        _criterion(
            "repeat_dense_retention_consistent",
            all(_float(row.get("core_minus_dense_anchor_mse_drift")) <= 0.0 for row in seed_rows),
            "claim",
            "core/periphery retention drift is no worse than dense on every repeat",
            [row.get("core_minus_dense_anchor_mse_drift") for row in seed_rows],
            "dense control remains stronger than the split mechanism",
        ),
        _criterion(
            "repeat_mlp_retention_consistent",
            all(_float(row.get("core_minus_mlp_anchor_mse_drift")) <= 0.0 for row in seed_rows),
            "claim",
            "core/periphery retention drift is no worse than MLP on every repeat",
            [row.get("core_minus_mlp_anchor_mse_drift") for row in seed_rows],
            "MLP control remains stronger than the split mechanism",
        ),
        _criterion(
            "repeat_periphery_first_pruning_consistent",
            all(_float(row.get("periphery_first_minus_core_first_prune_delta")) > 0.0 for row in seed_rows),
            "claim",
            "core pruning is more damaging than periphery pruning on every repeat",
            [row.get("periphery_first_minus_core_first_prune_delta") for row in seed_rows],
            "protected core is not causally distinguished by pruning across repeats",
        ),
    ]


def _aggregate_metrics(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregates: dict[str, Any] = {"seed_count": len(seed_rows)}
    for metric in REQUIRED_PRIMARY_METRICS:
        values = [_float(row.get(metric)) for row in seed_rows if row.get(metric) is not None]
        aggregates[f"mean_{metric}"] = sum(values) / len(values) if values else None
        aggregates[f"min_{metric}"] = min(values) if values else None
        aggregates[f"max_{metric}"] = max(values) if values else None
    return aggregates


def _has_required_metrics(row: dict[str, Any]) -> bool:
    return all(row.get(metric) is not None for metric in REQUIRED_PRIMARY_METRICS)


def _interpretation(scientific_gate: str) -> str:
    if scientific_gate == "ready_for_non_synthetic_pilot_design":
        return (
            "The two tiny synthetic local repeats consistently clear the preregistered "
            "plasticity, dense/MLP retention, and periphery-first pruning gates. This "
            "supports designing a non-synthetic local pilot only; it is not RunPod, "
            "Colab, default-router, or promotion evidence."
        )
    return (
        "The local repeat packet is incomplete or scientifically blocked. GPU "
        "validation and promotion remain invalid until the local artifact gates pass."
    )


def _criterion(
    criterion: str,
    passed: bool,
    severity: str,
    expected: Any,
    actual: Any,
    failure_action: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "severity": severity,
        "expected": expected,
        "actual": actual,
        "failure_action": failure_action,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_status.csv", summary["source_status"])
    _write_csv(out_dir / "seed_metrics.csv", summary["seed_metrics"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("\n")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Core/Periphery PC Column Synthesis",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Scientific gate: `{summary['scientific_gate']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        "",
        summary["interpretation"],
        "",
        f"Next step: {summary['selected_next_step']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _dirty_diff_hash() -> str:
    try:
        diff = subprocess.check_output(["git", "diff", "--no-ext-diff"], text=True)
    except Exception:
        return "unknown"
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, action="append", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_core_periphery_pc_column_synthesis(
        pilot_dirs=tuple(args.pilot_dir) if args.pilot_dir else DEFAULT_PILOT_DIRS,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "scientific_gate": summary["scientific_gate"],
                "claim_status": summary["claim_status"],
                "selected_next_step": summary["selected_next_step"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
