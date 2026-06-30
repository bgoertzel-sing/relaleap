"""Oracle-overlap targets for a redesigned Transformer-ACSR support objective."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUT_DIR = Path("results/reports/transformer_acsr_oracle_overlap")
DEFAULT_CONTEXT_KEYS = ("batch_index", "position_index", "token_index", "target_token")
LOSS_KEYS = ("support_loss", "loss", "ce_loss", "intervention_ce")
SUPPORT_KEYS = ("support", "support_pair", "support_set", "columns", "candidate_support")


def _coerce_support(value: Any) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(item) for item in value)
    if isinstance(value, list):
        return tuple(int(item) for item in value)
    if isinstance(value, int):
        return (value,)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        if text[0] in "[(":
            loaded = json.loads(text.replace("(", "[").replace(")", "]"))
            return _coerce_support(loaded)
        return tuple(int(part.strip()) for part in text.split(",") if part.strip())
    raise TypeError(f"cannot parse support value {value!r}")


def _row_support(row: dict[str, Any]) -> tuple[int, ...]:
    for key in SUPPORT_KEYS:
        if key in row and row[key] not in (None, ""):
            return _coerce_support(row[key])
    if "left_column" in row and "right_column" in row:
        return (int(row["left_column"]), int(row["right_column"]))
    if "column_left" in row and "column_right" in row:
        return (int(row["column_left"]), int(row["column_right"]))
    if "column" in row:
        return (int(row["column"]),)
    raise KeyError("support row is missing a support identifier")


def _row_loss(row: dict[str, Any]) -> float:
    for key in LOSS_KEYS:
        if key in row and row[key] not in (None, ""):
            return float(row[key])
    raise KeyError("support row is missing a loss field")


def _context_key(row: dict[str, Any], context_keys: Iterable[str]) -> tuple[Any, ...]:
    keys = tuple(context_keys)
    present = tuple(row.get(key) for key in keys)
    if any(value is not None for value in present):
        return present
    return ("global_context",)


def build_soft_oracle_support_targets(
    rows: Iterable[dict[str, Any]],
    *,
    temperature: float = 0.05,
    context_keys: tuple[str, ...] = DEFAULT_CONTEXT_KEYS,
) -> list[dict[str, Any]]:
    """Convert exhaustive support-loss rows into regret-aware soft support targets.

    Lower-loss supports receive larger probabilities via ``softmax(-loss / temperature)``
    within each context. Oracle labels are training/evaluator teachers only; inference
    must stay prefix-safe and cannot consume these losses.
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        support = _row_support(row)
        loss = _row_loss(row)
        context = _context_key(row, context_keys)
        grouped[context].append({"source": dict(row), "support": support, "loss": loss})

    targets: list[dict[str, Any]] = []
    for context, candidates in grouped.items():
        if not candidates:
            continue
        best_loss = min(candidate["loss"] for candidate in candidates)
        logits = [-(candidate["loss"] - best_loss) / temperature for candidate in candidates]
        max_logit = max(logits)
        weights = [math.exp(logit - max_logit) for logit in logits]
        normalizer = sum(weights)
        ranked = sorted(
            zip(candidates, weights, strict=True),
            key=lambda item: (item[0]["loss"], item[0]["support"]),
        )
        for rank, (candidate, weight) in enumerate(ranked, start=1):
            source = candidate["source"]
            output = {
                **{key: source.get(key) for key in context_keys if key in source},
                "support": json.dumps(list(candidate["support"])),
                "support_loss": candidate["loss"],
                "oracle_best_loss": best_loss,
                "oracle_regret": candidate["loss"] - best_loss,
                "oracle_rank": rank,
                "target_probability": weight / normalizer,
                "target_temperature": temperature,
            }
            targets.append(output)
    return targets


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_oracle_overlap_target_report(
    *,
    source_csv: Path,
    out_dir: Path = DEFAULT_OUT_DIR,
    temperature: float = 0.05,
) -> dict[str, Any]:
    rows = _read_csv(source_csv)
    targets = build_soft_oracle_support_targets(rows, temperature=temperature)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / "oracle_support_targets.csv"
    _write_csv(target_path, targets)
    contexts = {tuple(row.get(key) for key in DEFAULT_CONTEXT_KEYS) for row in targets}
    top_probability = max((row["target_probability"] for row in targets), default=None)
    summary = {
        "status": "pass",
        "decision": "oracle_overlap_transformer_acsr_target_construction_ready",
        "source_csv": str(source_csv),
        "candidate_row_count": len(rows),
        "target_row_count": len(targets),
        "context_count": len(contexts),
        "temperature": temperature,
        "max_target_probability": top_probability,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "selected_next_step": (
            "wire_local_prefix_safe_transformer_training_with_oracle_overlap_and_null_gates"
        ),
        "artifacts": {
            "oracle_support_targets_csv": str(target_path),
            "summary_json": str(out_dir / "summary.json"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    notes = [
        "# Transformer-ACSR Oracle-Overlap Targets",
        "",
        f"- Source CSV: `{source_csv}`",
        f"- Candidate rows: `{len(rows)}`",
        f"- Target rows: `{len(targets)}`",
        f"- Temperature: `{temperature}`",
        "- GPU validation and promotion remain blocked; this is a local training-target primitive.",
    ]
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--temperature", type=float, default=0.05)
    args = parser.parse_args(argv)
    summary = write_oracle_overlap_target_report(
        source_csv=args.source_csv,
        out_dir=args.out,
        temperature=args.temperature,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
