"""Minimal config-driven experiment runner.

This is a placeholder harness for the first Colab/local smoke path. It verifies
that the repo can be cloned, installed, run from a YAML config, and write the
standard run artifacts before the real LCR/HEP experiments are implemented.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path
from typing import Any


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
    rows = []
    loss = 1.0
    for step in range(max_steps + 1):
        loss = 0.92 * loss + 0.01 * random.random()
        rows.append(
            {
                "step": step,
                "seed": seed,
                "experiment_id": experiment_id,
                "smoke_loss": f"{loss:.8f}",
                "device": torch_info["device"],
            }
        )

    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment_id": experiment_id,
        "seed": seed,
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "status": "ok",
        "final_smoke_loss": float(rows[-1]["smoke_loss"]),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
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
    (out_dir / "notes.md").write_text(
        "\n".join(
            [
                f"# {experiment_id}",
                "",
                "This is the initial RelaLeap smoke run.",
                "",
                f"- Status: `{summary['status']}`",
                f"- Device: `{summary['device']}`",
                f"- CUDA available: `{summary['cuda_available']}`",
                f"- Final smoke loss: `{summary['final_smoke_loss']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a RelaLeap experiment config.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", default=Path("results/runs/smoke"), type=Path)
    args = parser.parse_args()
    summary = run(args.config, args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
