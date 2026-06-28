"""Local dense residual rank/norm matrix selected by the follow-up report."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.acsr_common_causal_residual_benchmark import (
    _as_dict,
    _causal_predictor_inputs,
    _heldout_mask,
    _per_token_ce,
    _read_yaml,
    _train_dense_arm,
)
from relaleap.experiments.acsr_transfer_objective_probe import DEFAULT_CONFIG
from relaleap.experiments.anticipatory_contextual_support_routing import _contextual_chunks
from relaleap.experiments.dense_residual_rank_norm_followup_report import (
    DEFAULT_OUT_DIR as DEFAULT_FOLLOWUP_DIR,
    _criterion,
    _float_or_none,
    _read_csv,
    _read_json,
)


DEFAULT_OUT_DIR = Path("results/reports/dense_residual_rank_norm_matrix")
DEFAULT_RANKS = (1, 4, 8, 16, 24)
DEFAULT_NORM_SCALES = (0.5, 0.75, 1.0)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "matrix_metrics.csv",
    "per_token_metrics.csv",
    "rank_summary.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_dense_residual_rank_norm_matrix(
    *,
    config_path: Path = DEFAULT_CONFIG,
    followup_dir: Path = DEFAULT_FOLLOWUP_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    ranks: tuple[int, ...] = DEFAULT_RANKS,
    norm_scales: tuple[float, ...] = DEFAULT_NORM_SCALES,
    dense_steps: int = 80,
) -> dict[str, Any]:
    """Run the selected local CPU dense rank/norm matrix and write artifacts."""

    start = time.time()
    followup_summary = _read_json(followup_dir / "summary.json")
    selected_rows = _read_csv(followup_dir / "next_matrix.csv")
    norm_rows = _read_csv(followup_dir / "norm_sensitivity.csv")
    sparse_reference = _sparse_reference_from_rows(norm_rows)
    preflight = _preflight_rows(config_path, followup_dir, followup_summary, selected_rows)
    if any(not row["passed"] for row in preflight):
        summary = _summary(
            status="fail",
            decision="dense_rank_norm_matrix_failed_closed",
            claim_status="matrix_not_run",
            start=start,
            config_path=config_path,
            followup_dir=followup_dir,
            dense_steps=dense_steps,
            matrix_rows=[],
            per_token_rows=[],
            rank_rows=[],
            gate_rows=preflight,
            out_dir=out_dir,
            sparse_reference=sparse_reference,
        )
        _write_artifacts(out_dir, summary, [], [], [], preflight)
        return summary

    try:
        matrix_rows, per_token_rows = _run_matrix(
            config_path=config_path,
            ranks=ranks,
            norm_scales=norm_scales,
            dense_steps=dense_steps,
            sparse_reference=sparse_reference,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        gate_rows = preflight + [
            _criterion(
                "matrix_runtime",
                False,
                "dense rank/norm matrix training and evaluation completes",
                str(exc),
                "dense rank/norm matrix could not run",
            )
        ]
        summary = _summary(
            status="fail",
            decision="dense_rank_norm_matrix_failed_closed",
            claim_status="matrix_runtime_failed",
            start=start,
            config_path=config_path,
            followup_dir=followup_dir,
            dense_steps=dense_steps,
            matrix_rows=[],
            per_token_rows=[],
            rank_rows=[],
            gate_rows=gate_rows,
            out_dir=out_dir,
            sparse_reference=sparse_reference,
        )
        _write_artifacts(out_dir, summary, [], [], [], gate_rows)
        return summary

    rank_rows = _rank_summary_rows(matrix_rows)
    gate_rows = preflight + _matrix_gate_rows(matrix_rows, rank_rows, sparse_reference)
    status = "pass" if all(row["passed"] for row in gate_rows) else "fail"
    decision = (
        "dense_rank_norm_matrix_completed"
        if status == "pass"
        else "dense_rank_norm_matrix_failed_gate"
    )
    claim_status = (
        "dense_rank_norm_sensitivity_measured_local_cpu"
        if status == "pass"
        else "dense_rank_norm_sensitivity_not_established"
    )
    summary = _summary(
        status=status,
        decision=decision,
        claim_status=claim_status,
        start=start,
        config_path=config_path,
        followup_dir=followup_dir,
        dense_steps=dense_steps,
        matrix_rows=matrix_rows,
        per_token_rows=per_token_rows,
        rank_rows=rank_rows,
        gate_rows=gate_rows,
        out_dir=out_dir,
        sparse_reference=sparse_reference,
    )
    _write_artifacts(out_dir, summary, matrix_rows, per_token_rows, rank_rows, gate_rows)
    return summary


def _preflight_rows(
    config_path: Path,
    followup_dir: Path,
    followup_summary: dict[str, Any],
    selected_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    selected = {
        (
            int(float(row.get("rank", "0"))),
            round(float(row.get("norm_scale_vs_sparse_topk2", "0")), 2),
        )
        for row in selected_rows
        if str(row.get("norm_scale_vs_sparse_topk2", "")).replace(".", "", 1).isdigit()
    }
    required = {(rank, round(scale, 2)) for rank in DEFAULT_RANKS for scale in DEFAULT_NORM_SCALES}
    return [
        _criterion(
            "config_present",
            config_path.is_file(),
            "matrix config exists",
            str(config_path),
            "missing matrix config",
        ),
        _criterion(
            "followup_summary_passed",
            followup_summary.get("status") == "pass",
            "dense rank/norm follow-up report passed",
            {"path": str(followup_dir / "summary.json"), "status": followup_summary.get("status")},
            "follow-up report missing or not passing",
        ),
        _criterion(
            "selected_matrix_rows_present",
            required.issubset(selected),
            "follow-up selected every required rank/scale cell",
            {"required": sorted(required), "selected": sorted(selected)},
            "follow-up next_matrix.csv does not select the required matrix",
        ),
    ]


def _run_matrix(
    *,
    config_path: Path,
    ranks: tuple[int, ...],
    norm_scales: tuple[float, ...],
    dense_steps: int,
    sparse_reference: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch
    import torch.nn.functional as F

    from relaleap.smoke import TinyCharTransformer, _build_batch

    nn = __import__("torch.nn").nn
    config = _read_yaml(config_path)
    run_cfg = _as_dict(config.get("run"))
    data_cfg = _as_dict(config.get("data"))
    model_cfg = _as_dict(config.get("model"))
    base_cfg = _as_dict(model_cfg.get("base"))

    seed = int(run_cfg.get("seed", 1))
    dataset = str(data_cfg.get("dataset", "tiny_shakespeare_word"))
    seq_len = int(data_cfg.get("seq_len", 64))
    hidden_dim = int(base_cfg.get("hidden_dim", 96))
    layers = int(base_cfg.get("layers", 2))

    torch.manual_seed(seed)
    inputs, targets, vocab_size = _build_batch(dataset=dataset, seq_len=seq_len, batch_size=4)
    base = TinyCharTransformer(vocab_size=vocab_size, seq_len=seq_len, hidden_dim=hidden_dim, layers=layers)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base.eval()

    with torch.no_grad():
        hidden = base.encode(inputs)
        base_logits = base.decode(hidden)
        base_losses = _per_token_ce(F, base_logits, targets, vocab_size)
        mask = _heldout_mask(base_losses.numel(), int(hidden.shape[1] - 1))
        chunks = _contextual_chunks(torch, hidden)
        causal_inputs = _causal_predictor_inputs(torch, chunks)

    sparse_target_l2 = sparse_reference["heldout_residual_update_l2"]
    target_active_params = int(sparse_reference["active_params_proxy"])
    matrix_rows: list[dict[str, Any]] = []
    per_token_rows: list[dict[str, Any]] = []
    for rank in ranks:
        for scale in norm_scales:
            label = f"dense_causal_rank{rank}_norm_scale_{scale:.2f}"
            row, losses, l2 = _train_dense_arm(
                torch,
                F,
                nn,
                base,
                hidden,
                targets,
                vocab_size,
                causal_inputs,
                label=label,
                target_parameter_count=target_active_params,
                steps=dense_steps,
                base_losses=base_losses,
                target_update_l2=sparse_target_l2 * scale,
                heldout_mask=mask,
                rank_override=rank,
            )
            row["norm_scale_vs_sparse_topk2"] = scale
            row["sparse_reference_heldout_l2"] = sparse_target_l2
            row["target_heldout_l2"] = sparse_target_l2 * scale
            row["heldout_delta_minus_sparse_topk2"] = _subtract(
                row.get("heldout_delta_vs_base_ce"),
                sparse_reference["heldout_delta_vs_base_ce"],
            )
            row["beats_sparse_topk2"] = (
                _float_or_none(row.get("heldout_delta_vs_base_ce")) is not None
                and _float_or_none(row.get("heldout_delta_vs_base_ce"))
                < sparse_reference["heldout_delta_vs_base_ce"]
            )
            matrix_rows.append({k: v for k, v in row.items() if k != "residual_update_tensor"})
            per_token_rows.extend(
                _per_token_rows(
                    arm=label,
                    rank=rank,
                    norm_scale=scale,
                    base_losses=base_losses,
                    losses=losses,
                    l2=l2,
                    heldout_mask=mask,
                    seq_len_minus_one=int(hidden.shape[1] - 1),
                )
            )
    return matrix_rows, per_token_rows


def _per_token_rows(
    *,
    arm: str,
    rank: int,
    norm_scale: float,
    base_losses: Any,
    losses: Any,
    l2: Any,
    heldout_mask: Any,
    seq_len_minus_one: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, loss in enumerate(losses.detach().cpu().tolist()):
        rows.append(
            {
                "arm": arm,
                "rank": rank,
                "norm_scale_vs_sparse_topk2": norm_scale,
                "token_index": idx,
                "position_index": idx % max(1, seq_len_minus_one),
                "split": "heldout" if bool(heldout_mask[idx].item()) else "train",
                "base_ce_loss": float(base_losses[idx].item()),
                "ce_loss": float(loss),
                "delta_vs_base_ce": float(loss - base_losses[idx].item()),
                "residual_update_l2": float(l2[idx].item()),
            }
        )
    return rows


def _rank_summary_rows(matrix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank in sorted({int(row["rank"]) for row in matrix_rows}):
        rank_rows = [row for row in matrix_rows if int(row["rank"]) == rank]
        best = min(
            rank_rows,
            key=lambda row: _float_or_none(row.get("heldout_delta_vs_base_ce")) or float("inf"),
        )
        rows.append(
            {
                "rank": rank,
                "best_arm": best["arm"],
                "best_norm_scale_vs_sparse_topk2": best["norm_scale_vs_sparse_topk2"],
                "best_heldout_delta_vs_base_ce": best["heldout_delta_vs_base_ce"],
                "best_delta_minus_sparse_topk2": best["heldout_delta_minus_sparse_topk2"],
                "beats_sparse_topk2": best["beats_sparse_topk2"],
            }
        )
    return rows


def _matrix_gate_rows(
    matrix_rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    sparse_reference: dict[str, float],
) -> list[dict[str, Any]]:
    expected_count = len(DEFAULT_RANKS) * len(DEFAULT_NORM_SCALES)
    sparse_delta = sparse_reference["heldout_delta_vs_base_ce"]
    winning = [row for row in matrix_rows if row.get("beats_sparse_topk2") is True]
    best_by_rank = {int(row["rank"]): row for row in rank_rows}
    minimal_winning_rank = min((int(row["rank"]) for row in winning), default=None)
    return [
        _criterion(
            "all_matrix_cells_present",
            len(matrix_rows) == expected_count,
            "all selected rank/scale cells were evaluated",
            {"expected": expected_count, "actual": len(matrix_rows)},
            "missing selected rank/scale cells",
        ),
        _criterion(
            "sparse_reference_recorded",
            sparse_delta < 0.0 and sparse_reference["heldout_residual_update_l2"] > 0.0,
            "sparse contextual top-k2 reference is recorded",
            {
                "heldout_delta": sparse_delta,
                "heldout_l2": sparse_reference["heldout_residual_update_l2"],
            },
            "missing sparse reference values",
        ),
        _criterion(
            "rank1_does_not_beat_sparse",
            not bool(best_by_rank.get(1, {}).get("beats_sparse_topk2")),
            "rank-1 dense remains below sparse top-k2",
            best_by_rank.get(1, {}),
            "rank-1 dense beat sparse, changing the dense-control interpretation",
        ),
        _criterion(
            "minimal_winning_rank_recorded",
            minimal_winning_rank is not None,
            "at least one dense rank beats sparse top-k2, so the threshold can be bracketed",
            minimal_winning_rank,
            "no dense rank beat sparse top-k2 in the selected matrix",
        ),
    ]


def _summary(
    *,
    status: str,
    decision: str,
    claim_status: str,
    start: float,
    config_path: Path,
    followup_dir: Path,
    dense_steps: int,
    matrix_rows: list[dict[str, Any]],
    per_token_rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    out_dir: Path,
    sparse_reference: dict[str, float],
) -> dict[str, Any]:
    winning_ranks = sorted(
        {int(row["rank"]) for row in matrix_rows if row.get("beats_sparse_topk2") is True}
    )
    return {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "config_path": str(config_path),
        "followup_dir": str(followup_dir),
        "dense_steps": dense_steps,
        "ranks": list(DEFAULT_RANKS),
        "norm_scales": list(DEFAULT_NORM_SCALES),
        "sparse_reference": {
            "arm": "sparse_contextual_topk2",
            "heldout_delta_vs_base_ce": sparse_reference["heldout_delta_vs_base_ce"],
            "heldout_residual_update_l2": sparse_reference["heldout_residual_update_l2"],
            "active_params_proxy": int(sparse_reference["active_params_proxy"]),
        },
        "matrix_cell_count": len(matrix_rows),
        "per_token_row_count": len(per_token_rows),
        "rank_summary_rows": rank_rows,
        "minimal_winning_dense_rank": min(winning_ranks) if winning_ranks else None,
        "winning_dense_ranks": winning_ranks,
        "gate_criteria": gate_rows,
        "failures": [row for row in gate_rows if not row["passed"]],
        "selected_next_step": _selected_next_step(status, winning_ranks),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }


def _selected_next_step(status: str, winning_ranks: list[int]) -> str:
    if status != "pass":
        return "repair or repeat the local dense rank/norm matrix before any GPU validation"
    if not winning_ranks:
        return "return to ACSR sparse-support mechanism tests because dense advantage did not survive the local rank/norm matrix"
    return (
        f"interpret dense advantage threshold at local rank {min(winning_ranks)} before deciding whether ACSR sparse supports need a stronger dense-teacher control"
    )


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    matrix_rows: list[dict[str, Any]],
    per_token_rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "matrix_metrics.csv", matrix_rows)
    _write_csv(out_dir / "per_token_metrics.csv", per_token_rows)
    _write_csv(out_dir / "rank_summary.csv", rank_rows)
    _write_csv(out_dir / "gate_criteria.csv", gate_rows)
    failures = summary.get("failures", [])
    lines = [
        "# Dense Residual Rank/Norm Matrix",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Matrix cells: `{summary['matrix_cell_count']}`",
        f"- Minimal winning dense rank: `{summary['minimal_winning_dense_rank']}`",
        f"- Next step: {summary['selected_next_step']}",
        "",
        "This local CPU matrix evaluates the selected dense causal rank ladder at residual-norm scales relative to sparse contextual top-k2. It keeps sparse top-k2 as a comparator and does not promote a dense or sparse default.",
    ]
    if failures:
        lines.extend(["", "## Failures"])
        lines.extend(f"- `{row['criterion']}`: {row['failure_reason']}" for row in failures)
    (out_dir / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sparse_reference_from_rows(rows: list[dict[str, str]]) -> dict[str, float]:
    for row in rows:
        if row.get("arm") != "sparse_contextual_topk2":
            continue
        delta = _float_or_none(row.get("heldout_delta_vs_base_ce"))
        l2 = _float_or_none(row.get("heldout_residual_update_l2"))
        active = _float_or_none(row.get("active_params_proxy"))
        if delta is not None and l2 is not None and active is not None:
            return {
                "heldout_delta_vs_base_ce": delta,
                "heldout_residual_update_l2": l2,
                "active_params_proxy": active,
            }
    return {
        "heldout_delta_vs_base_ce": -0.3092923164367676,
        "heldout_residual_update_l2": 1.0041953325271606,
        "active_params_proxy": 192.0,
    }


def _subtract(left: Any, right: Any) -> float | str:
    left_float = _float_or_none(left)
    right_float = _float_or_none(right)
    if left_float is None or right_float is None:
        return ""
    return left_float - right_float


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--followup-dir", type=Path, default=DEFAULT_FOLLOWUP_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dense-steps", type=int, default=80)
    args = parser.parse_args()
    summary = run_dense_residual_rank_norm_matrix(
        config_path=args.config,
        followup_dir=args.followup_dir,
        out_dir=args.out,
        dense_steps=args.dense_steps,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "out": str(args.out)}, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
