"""Compare small RelaLeap experiment runs from their standard artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.run import run


DEFAULT_CONFIGS = (
    Path("configs/char_smoke.yaml"),
    Path("configs/char_smoke_pc.yaml"),
    Path("configs/char_smoke_hep.yaml"),
)


def run_comparison(config_paths: list[Path], out_dir: Path) -> dict[str, Any]:
    """Run configs into sibling directories and write a compact comparison."""

    if len(config_paths) < 2:
        raise ValueError("comparison requires at least two config paths")

    start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_root = out_dir / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    entries = []
    combined_rows = []
    for config_path in config_paths:
        run_dir = run_root / config_path.stem
        summary = run(config_path, run_dir)
        metric_rows = _read_metrics(run_dir / "metrics.csv")
        entry = _comparison_entry(config_path, run_dir, summary, metric_rows)
        entries.append(entry)
        combined_rows.extend(_combined_rows(entry, metric_rows))

    status = "ok" if all(entry["status"] == "ok" for entry in entries) else "failed"
    verdict = _comparison_verdict(entries, status)
    comparison = {
        "status": status,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "loss_scale_note": (
            "Residual objectives may use different loss scales; compare each "
            "trajectory against its own initial loss."
        ),
        "verdict": verdict,
        "runs": entries,
    }
    _write_metrics(out_dir / "metrics.csv", combined_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_notes(out_dir / "notes.md", comparison)
    return comparison


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _comparison_entry(
    config_path: Path,
    run_dir: Path,
    summary: dict[str, Any],
    metric_rows: list[dict[str, str]],
) -> dict[str, Any]:
    initial = _first_metric(metric_rows, "residual_loss")
    final = _last_metric(metric_rows, "residual_loss")
    loss_delta = None if initial is None or final is None else final - initial
    loss_ratio = None
    if initial not in {None, 0.0} and final is not None:
        loss_ratio = final / initial
    phase0 = summary.get("phase0") or {}
    return {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "experiment_id": summary.get("experiment_id"),
        "status": summary.get("status"),
        "error": summary.get("error"),
        "residual_objective": phase0.get("residual_objective", ""),
        "training_steps": phase0.get("training_steps"),
        "initial_residual_loss": initial,
        "final_residual_loss": final,
        "residual_loss_delta": loss_delta,
        "residual_loss_ratio": loss_ratio,
        "base_loss": phase0.get("base_loss"),
        "zero_init_loss": phase0.get("zero_init_loss"),
        "hep_alpha_sweep": phase0.get("hep_alpha_sweep") or [],
        "invariants": phase0.get("invariants") or {},
    }


def _comparison_verdict(
    entries: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    failed_invariants = []
    invariant_count = 0
    for entry in entries:
        invariants = entry.get("invariants") or {}
        invariant_count += len(invariants)
        for name, value in sorted(invariants.items()):
            if not value:
                failed_invariants.append(
                    {
                        "experiment_id": entry["experiment_id"],
                        "invariant": name,
                    }
                )

    best_hep = _best_hep_alpha(entries)
    invariants_passed = bool(invariant_count) and not failed_invariants
    verdict_status = "pass" if status == "ok" and invariants_passed else "fail"
    return {
        "status": verdict_status,
        "invariants_passed": invariants_passed,
        "invariant_count": invariant_count,
        "failed_invariants": failed_invariants,
        "best_hep_alpha_by_loss": best_hep,
    }


def _best_hep_alpha(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for entry in entries:
        for sweep_entry in entry.get("hep_alpha_sweep") or []:
            loss = sweep_entry.get("loss")
            if loss is None:
                continue
            candidates.append(
                {
                    "experiment_id": entry["experiment_id"],
                    "alpha": float(sweep_entry["alpha"]),
                    "loss": float(loss),
                    "max_logit_delta_from_ordinary": float(
                        sweep_entry["max_logit_delta_from_ordinary"]
                    ),
                }
            )
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate["loss"])


def _combined_rows(
    entry: dict[str, Any],
    metric_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    initial = entry["initial_residual_loss"]
    for row in metric_rows:
        residual_loss = _parse_float(row.get("residual_loss", ""))
        loss_delta = None
        if initial is not None and residual_loss is not None:
            loss_delta = residual_loss - initial
        rows.append(
            {
                "experiment_id": entry["experiment_id"],
                "config_path": entry["config_path"],
                "run_dir": entry["run_dir"],
                "residual_objective": entry["residual_objective"],
                "step": row.get("step", ""),
                "phase": row.get("phase", ""),
                "base_loss": row.get("base_loss", ""),
                "residual_loss": row.get("residual_loss", ""),
                "loss_delta_from_initial": _format_optional(loss_delta),
                "hep_alpha": row.get("hep_alpha", ""),
                "hep_loss": row.get("hep_loss", ""),
                "max_hep_logit_delta_from_ordinary": row.get(
                    "max_hep_logit_delta_from_ordinary", ""
                ),
                "status": row.get("status", entry["status"]),
            }
        )
    return rows


def _first_metric(rows: list[dict[str, str]], field: str) -> float | None:
    for row in rows:
        value = _parse_float(row.get(field, ""))
        if value is not None:
            return value
    return None


def _last_metric(rows: list[dict[str, str]], field: str) -> float | None:
    for row in reversed(rows):
        value = _parse_float(row.get(field, ""))
        if value is not None:
            return value
    return None


def _parse_float(value: str | None) -> float | None:
    if value in {"", None}:
        return None
    return float(value)


def _format_optional(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.8f}"


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "experiment_id",
        "config_path",
        "run_dir",
        "residual_objective",
        "step",
        "phase",
        "base_loss",
        "residual_loss",
        "loss_delta_from_initial",
        "hep_alpha",
        "hep_loss",
        "max_hep_logit_delta_from_ordinary",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, comparison: dict[str, Any]) -> None:
    verdict = comparison["verdict"]
    lines = [
        "# Char Smoke Objective Comparison",
        "",
        "Command-driven comparison of Phase 0 char-smoke residual objectives.",
        "",
        f"- Status: `{comparison['status']}`",
        f"- Verdict: `{verdict['status']}`",
        (
            f"- Phase 0 invariants: `{verdict['invariant_count']}` checked, "
            f"passed `{verdict['invariants_passed']}`"
        ),
        f"- Loss scale note: {comparison['loss_scale_note']}",
        "",
        "## Runs",
        "",
        "| Experiment | Objective | Status | Initial loss | Final loss | Delta | Ratio |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for entry in comparison["runs"]:
        row_template = (
            "| {experiment_id} | {objective} | {status} | {initial} | "
            "{final} | {delta} | {ratio} |"
        )
        lines.append(
            row_template.format(
                experiment_id=entry["experiment_id"],
                objective=entry["residual_objective"],
                status=entry["status"],
                initial=_format_note_metric(entry["initial_residual_loss"]),
                final=_format_note_metric(entry["final_residual_loss"]),
                delta=_format_note_metric(entry["residual_loss_delta"]),
                ratio=_format_note_metric(entry["residual_loss_ratio"]),
            )
        )
    lines.extend(["", "## Artifacts", ""])
    for entry in comparison["runs"]:
        lines.append(f"- `{entry['experiment_id']}`: `{entry['run_dir']}`")
    hep_entries = [
        entry for entry in comparison["runs"] if entry.get("hep_alpha_sweep")
    ]
    if hep_entries:
        lines.extend(["", "## HEP Alpha Sweeps", ""])
        for entry in hep_entries:
            sweep = ", ".join(
                (
                    f"alpha {sweep_entry['alpha']}: "
                    f"loss {_format_note_metric(sweep_entry['loss'])}, "
                    "delta "
                    f"{_format_note_metric(sweep_entry['max_logit_delta_from_ordinary'])}"
                )
                for sweep_entry in entry["hep_alpha_sweep"]
            )
            lines.append(f"- `{entry['experiment_id']}`: {sweep}")
    if verdict["failed_invariants"]:
        lines.extend(["", "## Failed Invariants", ""])
        for failed in verdict["failed_invariants"]:
            lines.append(
                f"- `{failed['experiment_id']}`: `{failed['invariant']}`"
            )
    if verdict["best_hep_alpha_by_loss"]:
        best = verdict["best_hep_alpha_by_loss"]
        lines.extend(
            [
                "",
                "## Verdict",
                "",
                (
                    "- Best HEP alpha by loss: "
                    f"`{best['alpha']}` in `{best['experiment_id']}` "
                    f"with loss `{_format_note_metric(best['loss'])}` "
                    "and ordinary-logit delta "
                    f"`{_format_note_metric(best['max_logit_delta_from_ordinary'])}`"
                ),
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_note_metric(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.8f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and compare RelaLeap experiment configs."
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        type=Path,
        help="Config to include. Repeat for multiple configs.",
    )
    parser.add_argument(
        "--out",
        default=Path("results/comparisons/char_smoke_objectives"),
        type=Path,
    )
    args = parser.parse_args()
    config_paths = args.configs or list(DEFAULT_CONFIGS)
    comparison = run_comparison(config_paths, args.out)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    if comparison["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
