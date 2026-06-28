"""Extract dense rank-16/24 non-CE mechanism observables for the ACSR gate."""

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
    _criterion,
    _float_or_none,
    _read_csv,
    _read_json,
)
from relaleap.experiments.dense_residual_rank_norm_matrix import DEFAULT_OUT_DIR as DEFAULT_DENSE_MATRIX_DIR


DEFAULT_OUT_DIR = Path("results/reports/acsr_dense_mechanism_observables")
REQUIRED_RANKS = (16, 24)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "dense_mechanism_observables.csv",
    "per_token_observables.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_acsr_dense_mechanism_observable_extractor(
    *,
    config_path: Path = DEFAULT_CONFIG,
    dense_matrix_dir: Path = DEFAULT_DENSE_MATRIX_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    dense_steps: int = 80,
) -> dict[str, Any]:
    """Rerun only the selected dense rank controls and write non-CE observables."""

    start = time.time()
    dense_summary = _read_json(dense_matrix_dir / "summary.json")
    rank_rows = _read_csv(dense_matrix_dir / "rank_summary.csv")
    matrix_rows = _read_csv(dense_matrix_dir / "matrix_metrics.csv")
    preflight = _preflight_rows(config_path, dense_matrix_dir, dense_summary, rank_rows, matrix_rows)
    if any(not row["passed"] for row in preflight):
        summary = _summary(
            status="fail",
            decision="acsr_dense_mechanism_observables_failed_closed",
            claim_status="dense_observables_not_extracted",
            start=start,
            config_path=config_path,
            dense_matrix_dir=dense_matrix_dir,
            dense_steps=dense_steps,
            observable_rows=[],
            per_token_rows=[],
            gate_rows=preflight,
            out_dir=out_dir,
        )
        _write_artifacts(out_dir, summary, [], [], preflight)
        return summary

    try:
        observable_rows, per_token_rows = _extract_observables(
            config_path=config_path,
            rank_rows=rank_rows,
            matrix_rows=matrix_rows,
            dense_steps=dense_steps,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        gate_rows = preflight + [
            _criterion(
                "dense_observable_runtime",
                False,
                "rank-16/24 dense observable extraction completes",
                str(exc),
                "dense observable extraction could not run",
            )
        ]
        summary = _summary(
            status="fail",
            decision="acsr_dense_mechanism_observables_failed_closed",
            claim_status="dense_observable_runtime_failed",
            start=start,
            config_path=config_path,
            dense_matrix_dir=dense_matrix_dir,
            dense_steps=dense_steps,
            observable_rows=[],
            per_token_rows=[],
            gate_rows=gate_rows,
            out_dir=out_dir,
        )
        _write_artifacts(out_dir, summary, [], [], gate_rows)
        return summary

    gate_rows = preflight + _observable_gate_rows(observable_rows, per_token_rows)
    status = "pass" if all(row["passed"] for row in gate_rows) else "fail"
    summary = _summary(
        status=status,
        decision=(
            "acsr_dense_mechanism_observables_extracted"
            if status == "pass"
            else "acsr_dense_mechanism_observables_failed_gate"
        ),
        claim_status=(
            "dense_rank16_24_non_ce_observables_available_for_sparse_gate"
            if status == "pass"
            else "dense_rank16_24_non_ce_observables_incomplete"
        ),
        start=start,
        config_path=config_path,
        dense_matrix_dir=dense_matrix_dir,
        dense_steps=dense_steps,
        observable_rows=observable_rows,
        per_token_rows=per_token_rows,
        gate_rows=gate_rows,
        out_dir=out_dir,
    )
    _write_artifacts(out_dir, summary, observable_rows, per_token_rows, gate_rows)
    return summary


def _preflight_rows(
    config_path: Path,
    dense_matrix_dir: Path,
    dense_summary: dict[str, Any],
    rank_rows: list[dict[str, str]],
    matrix_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rank_by_rank = {int(_float_or_none(row.get("rank")) or 0): row for row in rank_rows}
    matrix_by_arm = {row.get("arm", ""): row for row in matrix_rows}
    selected = {
        rank: rank_by_rank.get(rank, {}).get("best_arm", "")
        for rank in REQUIRED_RANKS
    }
    return [
        _criterion("config_present", config_path.is_file(), "config exists", str(config_path), "missing config"),
        _criterion(
            "dense_matrix_passed",
            dense_summary.get("status") == "pass",
            "dense rank/norm matrix passed",
            {"path": str(dense_matrix_dir / "summary.json"), "status": dense_summary.get("status")},
            "dense rank/norm matrix is missing or not passing",
        ),
        _criterion(
            "rank16_24_best_arms_present",
            all(selected.get(rank) in matrix_by_arm for rank in REQUIRED_RANKS),
            "rank-16/24 best dense arms are present in matrix_metrics.csv",
            selected,
            "missing rank-16/24 best dense arm rows",
        ),
        _criterion(
            "sparse_reference_present",
            isinstance(dense_summary.get("sparse_reference"), dict)
            and _float_or_none(dense_summary.get("sparse_reference", {}).get("heldout_residual_update_l2")) is not None
            and _float_or_none(dense_summary.get("sparse_reference", {}).get("active_params_proxy")) is not None,
            "dense matrix summary records sparse norm and active parameter reference",
            dense_summary.get("sparse_reference"),
            "missing sparse reference in dense matrix summary",
        ),
    ]


def _extract_observables(
    *,
    config_path: Path,
    rank_rows: list[dict[str, str]],
    matrix_rows: list[dict[str, str]],
    dense_steps: int,
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

    matrix_by_arm = {row.get("arm", ""): row for row in matrix_rows}
    rank_by_rank = {int(_float_or_none(row.get("rank")) or 0): row for row in rank_rows}
    observable_rows: list[dict[str, Any]] = []
    per_token_rows: list[dict[str, Any]] = []
    for rank in REQUIRED_RANKS:
        selected = rank_by_rank[rank]
        matrix = matrix_by_arm[selected["best_arm"]]
        target_l2 = _float_or_none(matrix.get("target_heldout_l2"))
        if target_l2 is None:
            target_l2 = _float_or_none(matrix.get("heldout_residual_update_l2")) or 1.0
        target_active = int(_float_or_none(matrix.get("target_active_params_proxy")) or 192)
        label = f"dense_rank{rank}_best_norm"
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
            target_parameter_count=target_active,
            steps=dense_steps,
            base_losses=base_losses,
            target_update_l2=target_l2,
            heldout_mask=mask,
            rank_override=rank,
        )
        update = row.get("residual_update_tensor")
        with torch.no_grad():
            dense_logits = base.decode(hidden + update)
        observable_rows.append(
            _observable_row(
                torch=torch,
                F=F,
                rank=rank,
                source_arm=selected["best_arm"],
                row=row,
                base_logits=base_logits,
                dense_logits=dense_logits,
                base_losses=base_losses,
                losses=losses,
                l2=l2,
                heldout_mask=mask,
            )
        )
        per_token_rows.extend(
            _per_token_observable_rows(
                torch=torch,
                arm=label,
                rank=rank,
                source_arm=selected["best_arm"],
                base_logits=base_logits,
                dense_logits=dense_logits,
                base_losses=base_losses,
                losses=losses,
                l2=l2,
                heldout_mask=mask,
                seq_len_minus_one=int(hidden.shape[1] - 1),
            )
        )
    return observable_rows, per_token_rows


def _observable_row(
    *,
    torch: Any,
    F: Any,
    rank: int,
    source_arm: str,
    row: dict[str, Any],
    base_logits: Any,
    dense_logits: Any,
    base_losses: Any,
    losses: Any,
    l2: Any,
    heldout_mask: Any,
) -> dict[str, Any]:
    flat_base_logits = base_logits[:, :-1, :].reshape(base_losses.numel(), -1)
    flat_dense_logits = dense_logits[:, :-1, :].reshape(base_losses.numel(), -1)
    anchor_mask = ~heldout_mask
    dense_losses = losses.reshape(-1)
    delta = dense_losses - base_losses
    base_argmax = flat_base_logits.argmax(dim=-1)
    dense_argmax = flat_dense_logits.argmax(dim=-1)
    anchor_logit_mse = float(F.mse_loss(flat_dense_logits[anchor_mask], flat_base_logits[anchor_mask]).item())
    functional_churn = float((base_argmax[anchor_mask] != dense_argmax[anchor_mask]).float().mean().item())
    retention = float(delta[anchor_mask].mean().item())
    heldout_gain = torch.clamp(-delta[heldout_mask], min=0.0).sum()
    anchor_damage = torch.clamp(delta[anchor_mask], min=0.0).sum()
    purity = float((heldout_gain / (heldout_gain + anchor_damage + 1e-12)).item())
    heldout_kl = float(
        F.kl_div(
            F.log_softmax(flat_dense_logits[heldout_mask], dim=-1),
            F.softmax(flat_base_logits[heldout_mask], dim=-1),
            reduction="batchmean",
        ).item()
    )
    return {
        "arm": f"dense_rank{rank}_best_norm",
        "source_arm": source_arm,
        "rank": rank,
        "ce_loss": float(row["heldout_ce_loss"]),
        "delta_vs_base_ce": float(row["heldout_delta_vs_base_ce"]),
        "residual_l2": float(row["heldout_residual_update_l2"]),
        "active_params": int(row["active_params_proxy"]),
        "anchor_kl_or_logit_mse": anchor_logit_mse,
        "anchor_kl_vs_base": heldout_kl,
        "functional_churn": functional_churn,
        "retention_or_forgetting": retention,
        "intervention_fingerprint_purity": purity,
        "heldout_improvement_purity_proxy": purity,
        "observable_note": (
            "dense proxy: anchor logit MSE and prediction churn are measured against base on "
            "train/anchor tokens; purity is heldout CE improvement mass divided by heldout "
            "improvement plus anchor damage"
        ),
    }


def _per_token_observable_rows(
    *,
    torch: Any,
    arm: str,
    rank: int,
    source_arm: str,
    base_logits: Any,
    dense_logits: Any,
    base_losses: Any,
    losses: Any,
    l2: Any,
    heldout_mask: Any,
    seq_len_minus_one: int,
) -> list[dict[str, Any]]:
    flat_base_logits = base_logits[:, :-1, :].reshape(base_losses.numel(), -1)
    flat_dense_logits = dense_logits[:, :-1, :].reshape(base_losses.numel(), -1)
    base_argmax = flat_base_logits.argmax(dim=-1)
    dense_argmax = flat_dense_logits.argmax(dim=-1)
    logit_mse = ((flat_dense_logits - flat_base_logits) ** 2).mean(dim=-1)
    dense_losses = losses.reshape(-1)
    deltas = dense_losses - base_losses
    rows: list[dict[str, Any]] = []
    for idx in range(base_losses.numel()):
        rows.append(
            {
                "arm": arm,
                "source_arm": source_arm,
                "rank": rank,
                "token_index": idx,
                "position_index": idx % max(1, seq_len_minus_one),
                "split": "heldout" if bool(heldout_mask[idx].item()) else "anchor",
                "base_ce_loss": float(base_losses[idx].item()),
                "ce_loss": float(dense_losses[idx].item()),
                "delta_vs_base_ce": float(deltas[idx].item()),
                "residual_update_l2": float(l2[idx].item()),
                "logit_mse_vs_base": float(logit_mse[idx].item()),
                "prediction_changed_vs_base": bool(base_argmax[idx].item() != dense_argmax[idx].item()),
            }
        )
    return rows


def _observable_gate_rows(
    observable_rows: list[dict[str, Any]],
    per_token_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_rank = {int(row.get("rank", 0)): row for row in observable_rows}
    required_fields = {
        "anchor_kl_or_logit_mse",
        "functional_churn",
        "retention_or_forgetting",
        "intervention_fingerprint_purity",
    }
    missing = {
        rank: sorted(field for field in required_fields if by_rank.get(rank, {}).get(field, "") == "")
        for rank in REQUIRED_RANKS
    }
    return [
        _criterion(
            "rank16_24_observable_rows_present",
            all(rank in by_rank for rank in REQUIRED_RANKS),
            "observable rows exist for dense rank 16 and rank 24",
            sorted(by_rank),
            "missing required dense rank observable row",
        ),
        _criterion(
            "mechanism_fields_present",
            all(not missing[rank] for rank in REQUIRED_RANKS),
            "rank-16/24 dense rows include every required mechanism field",
            missing,
            "one or more dense mechanism fields are missing",
        ),
        _criterion(
            "per_token_observables_present",
            len(per_token_rows) > 0,
            "per-token dense observable rows were written",
            len(per_token_rows),
            "missing per-token dense observable rows",
        ),
    ]


def _summary(
    *,
    status: str,
    decision: str,
    claim_status: str,
    start: float,
    config_path: Path,
    dense_matrix_dir: Path,
    dense_steps: int,
    observable_rows: list[dict[str, Any]],
    per_token_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    return {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "config_path": str(config_path),
        "dense_matrix_dir": str(dense_matrix_dir),
        "dense_steps": dense_steps,
        "required_ranks": list(REQUIRED_RANKS),
        "observable_rows": observable_rows,
        "observable_row_count": len(observable_rows),
        "per_token_row_count": len(per_token_rows),
        "gate_criteria": gate_rows,
        "failures": [row for row in gate_rows if not row["passed"]],
        "selected_next_step": (
            "rerun acsr_sparse_dense_mechanism_gate so dense rank16/24 non-CE fields are compared"
            if status == "pass"
            else "repair dense observable extraction before rerunning the sparse-vs-dense mechanism gate"
        ),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    observable_rows: list[dict[str, Any]],
    per_token_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "dense_mechanism_observables.csv", observable_rows)
    _write_csv(out_dir / "per_token_observables.csv", per_token_rows)
    _write_csv(out_dir / "gate_criteria.csv", gate_rows)
    lines = [
        "# ACSR Dense Mechanism Observables",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Observable rows: `{summary['observable_row_count']}`",
        f"- Next step: {summary['selected_next_step']}",
        "",
        "This bounded CPU extractor reruns only the dense rank-16/24 best-norm controls selected by the rank/norm matrix. It writes dense-side anchor logit MSE, prediction churn, retention/forgetting, and a heldout-improvement purity proxy for the sparse-vs-dense mechanism gate.",
    ]
    if summary["failures"]:
        lines.extend(["", "## Failures"])
        lines.extend(f"- `{row['criterion']}`: {row['failure_reason']}" for row in summary["failures"])
    (out_dir / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dense-matrix-dir", type=Path, default=DEFAULT_DENSE_MATRIX_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dense-steps", type=int, default=80)
    args = parser.parse_args()
    summary = run_acsr_dense_mechanism_observable_extractor(
        config_path=args.config,
        dense_matrix_dir=args.dense_matrix_dir,
        out_dir=args.out,
        dense_steps=args.dense_steps,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "out": str(args.out)}, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
