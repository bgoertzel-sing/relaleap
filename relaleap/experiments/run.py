"""Minimal config-driven experiment runner."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

from relaleap.smoke import run_phase0_smoke


def _load_torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on environment
        return {
            "torch_available": False,
            "torch_error": repr(exc),
            "cuda_available": False,
            "device": "cpu",
        }

    cuda_available = bool(torch.cuda.is_available())
    device = "cuda" if cuda_available else "cpu"
    info: dict[str, Any] = {
        "torch_available": True,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device": device,
    }
    if cuda_available:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()
    return info


def _read_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        loaded = yaml.safe_load(text)
        return loaded or {}
    except ModuleNotFoundError:
        return _read_simple_yaml(text)


def _read_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small nested-key YAML shape used by the smoke config.

    Real experiment configs should use PyYAML. This fallback exists so the
    smoke command can validate the Colab bridge even in sparse environments.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, sep, value = raw_line.strip().partition(":")
        if not sep:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def run(config_path: Path, out_dir: Path) -> dict[str, Any]:
    config = _read_config(config_path)
    run_cfg = config.get("run", {})
    seed = int(run_cfg.get("seed", 1))
    max_steps = int(run_cfg.get("max_steps", 10))
    experiment_id = str(run_cfg.get("experiment_id", "smoke"))

    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_info = _load_torch_info()
    start = time.time()
    rows = _build_placeholder_rows(
        max_steps=max_steps,
        seed=seed,
        experiment_id=experiment_id,
        device=torch_info["device"],
    )

    phase0: dict[str, Any] | None = None
    status = "ok"
    error: str | None = None
    try:
        phase0_result = run_phase0_smoke(config)
        phase0 = phase0_result.to_summary()
        if not all(phase0_result.invariants.values()):
            status = "failed"
            error = "Phase 0 invariant failure"
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"

    for row in rows:
        row["status"] = status

    _write_metrics(out_dir / "metrics.csv", rows)
    final_smoke_loss = float(rows[-1]["smoke_loss"])
    summary = {
        "experiment_id": experiment_id,
        "seed": seed,
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "status": status,
        "error": error,
        "final_smoke_loss": final_smoke_loss,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "phase0": phase0,
        **torch_info,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "config.yaml").write_text(
        config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _write_notes(out_dir / "notes.md", experiment_id, summary)

    required = config.get("outputs", {})
    artifact_invariants = {
        "summary_json": not required.get("require_summary_json", True)
        or (out_dir / "summary.json").is_file(),
        "metrics_csv": not required.get("require_metrics_csv", True)
        or (out_dir / "metrics.csv").is_file(),
        "notes_md": not required.get("require_notes_md", True)
        or (out_dir / "notes.md").is_file(),
    }
    summary["artifact_invariants"] = artifact_invariants
    if not all(artifact_invariants.values()):
        summary["status"] = "failed"
        summary["error"] = summary["error"] or "Required artifact missing"
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_placeholder_rows(
    *,
    max_steps: int,
    seed: int,
    experiment_id: str,
    device: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    loss = 1.0
    for step in range(max_steps + 1):
        loss = 0.92 * loss + 0.01 * random.random()
        rows.append(
            {
                "step": step,
                "seed": seed,
                "experiment_id": experiment_id,
                "smoke_loss": f"{loss:.8f}",
                "device": device,
            }
        )
    return rows


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, experiment_id: str, summary: dict[str, Any]) -> None:
    phase0 = summary.get("phase0") or {}
    invariants = phase0.get("invariants") or {}
    invariant_lines = [
        f"- {name}: `{value}`" for name, value in sorted(invariants.items())
    ]
    if not invariant_lines:
        invariant_lines = ["- Phase 0 invariants: `not run`"]

    path.write_text(
        "\n".join(
            [
                f"# {experiment_id}",
                "",
                "RelaLeap char-level Phase 0 smoke run.",
                "",
                f"- Status: `{summary['status']}`",
                f"- Error: `{summary['error'] or 'none'}`",
                f"- Device: `{summary['device']}`",
                f"- CUDA available: `{summary['cuda_available']}`",
                f"- Final smoke loss: `{summary['final_smoke_loss']}`",
                "",
                "## Invariants",
                "",
                *invariant_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a RelaLeap experiment config.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", default=Path("results/runs/smoke"), type=Path)
    args = parser.parse_args()
    summary = run(args.config, args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
