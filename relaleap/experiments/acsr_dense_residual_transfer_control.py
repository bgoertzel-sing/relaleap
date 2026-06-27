"""Rank-matched dense residual control for the ACSR transfer objective."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from relaleap.experiments.acsr_transfer_objective_probe import (
    DEFAULT_CONFIG,
    DEFAULT_OUT_DIR as DEFAULT_SOURCE_PROBE,
)
from relaleap.experiments.anticipatory_contextual_support_routing import (
    _causal_predictor_inputs,
    _contextual_chunks,
    _position_predictor_inputs,
)


DEFAULT_OUT_DIR = Path("results/reports/acsr_dense_residual_transfer_control")
REQUIRED_ARTIFACTS = (
    "summary.json",
    "dense_control_metrics.csv",
    "source_probe_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)
PARTNER_VALUE_PATH = "partner_values"
OWN_VALUE_PATH = "own_values"
TRANSFER_ARM = "transfer_objective_router"
DIRECT_ARM = "direct_causal_mlp_baseline"


def run_acsr_dense_residual_transfer_control(
    *,
    source_probe_dir: Path = DEFAULT_SOURCE_PROBE,
    config_path: Path | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    dense_steps: int = 120,
) -> dict[str, Any]:
    """Train a local rank-matched dense causal adapter and compare held-out CE."""

    start = time.time()
    source_summary, source_rows, source_gate_rows = _load_source_probe(source_probe_dir)
    if config_path is None:
        config_path = Path(str(source_summary.get("config_path") or DEFAULT_CONFIG))
    source_metrics = _source_probe_metrics(source_rows)
    early_gates = _source_gate_rows(source_probe_dir, source_summary, source_rows, source_metrics)
    if any(not row["passed"] for row in early_gates):
        summary = _summary(
            status="fail",
            decision="acsr_dense_residual_transfer_control_failed_closed",
            claim_status="dense_control_not_run",
            start=start,
            source_probe_dir=source_probe_dir,
            config_path=config_path,
            dense_steps=dense_steps,
            dense_rows=[],
            source_metrics=source_metrics,
            gate_rows=early_gates,
            failures=[row for row in early_gates if not row["passed"]],
            out_dir=out_dir,
        )
        _write_artifacts(out_dir, summary, [], source_metrics, early_gates)
        return summary

    try:
        dense_rows = _run_dense_controls(
            config_path=config_path,
            source_summary=source_summary,
            source_metrics=source_metrics,
            dense_steps=dense_steps,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        gate_rows = early_gates + [
            _criterion(
                "dense_control_runtime",
                False,
                "dense control training/evaluation completes",
                str(exc),
                "dense residual control could not run",
            )
        ]
        summary = _summary(
            status="fail",
            decision="acsr_dense_residual_transfer_control_failed_closed",
            claim_status="dense_control_runtime_failed",
            start=start,
            source_probe_dir=source_probe_dir,
            config_path=config_path,
            dense_steps=dense_steps,
            dense_rows=[],
            source_metrics=source_metrics,
            gate_rows=gate_rows,
            failures=[row for row in gate_rows if not row["passed"]],
            out_dir=out_dir,
        )
        _write_artifacts(out_dir, summary, [], source_metrics, gate_rows)
        return summary

    gate_rows = early_gates + _dense_gate_rows(source_metrics, dense_rows)
    failures = [row for row in gate_rows if not row["passed"]]
    status = "pass" if not failures else "fail"
    summary = _summary(
        status=status,
        decision=(
            "acsr_dense_residual_transfer_control_supported"
            if status == "pass"
            else "acsr_dense_residual_transfer_control_failed_gate"
        ),
        claim_status=(
            "sparse_transfer_survives_rank_matched_dense_control_not_promoted"
            if status == "pass"
            else "sparse_transfer_not_separated_from_dense_control"
        ),
        start=start,
        source_probe_dir=source_probe_dir,
        config_path=config_path,
        dense_steps=dense_steps,
        dense_rows=dense_rows,
        source_metrics=source_metrics,
        gate_rows=gate_rows,
        failures=failures,
        out_dir=out_dir,
    )
    _write_artifacts(out_dir, summary, dense_rows, source_metrics, gate_rows)
    return summary


def _run_dense_controls(
    *,
    config_path: Path,
    source_summary: dict[str, Any],
    source_metrics: list[dict[str, Any]],
    dense_steps: int,
) -> list[dict[str, Any]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from relaleap.smoke import ResidualColumns, TinyCharTransformer, _build_batch, _residual_loss

    config = _read_yaml(config_path)
    run_cfg = _as_dict(config.get("run"))
    data_cfg = _as_dict(config.get("data"))
    model_cfg = _as_dict(config.get("model"))
    base_cfg = _as_dict(model_cfg.get("base"))
    column_cfg = _as_dict(model_cfg.get("columns"))
    training_cfg = _as_dict(config.get("training"))

    seed = int(run_cfg.get("seed", 1))
    learning_rate = float(run_cfg.get("learning_rate", 1e-2))
    dataset = str(data_cfg.get("dataset", "tiny_shakespeare_word"))
    seq_len = int(data_cfg.get("seq_len", 64))
    hidden_dim = int(base_cfg.get("hidden_dim", 96))
    layers = int(base_cfg.get("layers", 2))
    num_columns = int(column_cfg.get("num_columns", 24))
    atoms_per_column = int(column_cfg.get("atoms_per_column", 4))
    top_k = int(column_cfg.get("top_k", 2))
    contextual_width = int(column_cfg.get("contextual_router_hidden_dim", hidden_dim * 2))
    residual_objective = str(training_cfg.get("residual_objective", "supervised_ce"))
    residual_steps = int(source_summary.get("max_steps") or 6)

    torch.manual_seed(seed)
    inputs, targets, vocab_size = _build_batch(dataset=dataset, seq_len=seq_len, batch_size=4)
    base = TinyCharTransformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        layers=layers,
    )
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base.eval()

    residual = ResidualColumns(
        hidden_dim=hidden_dim,
        num_columns=num_columns,
        atoms_per_column=atoms_per_column,
        top_k=top_k,
        support_router="contextual_mlp",
        contextual_router_hidden_dim=contextual_width,
    )
    residual.train()
    optimizer = torch.optim.AdamW(residual.parameters(), lr=learning_rate)
    for _ in range(max(1, residual_steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = _residual_loss(
            base,
            residual,
            inputs,
            targets,
            vocab_size,
            objective=residual_objective,
        )
        loss.backward()
        optimizer.step()
    residual.eval()

    with torch.no_grad():
        hidden = base.encode(inputs)
        chunks = _contextual_chunks(torch, hidden)
        causal_inputs = _causal_predictor_inputs(torch, chunks)
        position_inputs = _position_predictor_inputs(torch, chunks)
        base_logits = base.decode(hidden)
        base_losses = _per_token_ce(F, base_logits, targets, vocab_size)

    sparse_value_params = int(num_columns * atoms_per_column * (hidden_dim + 1))
    target_update_l2 = _source_sparse_update_l2(source_metrics)
    rows = [
        _train_dense_control(
            torch,
            F,
            nn,
            base,
            hidden,
            targets,
            vocab_size,
            causal_inputs,
            label="rank_matched_causal_dense_residual",
            target_parameter_count=sparse_value_params,
            steps=dense_steps,
            base_losses=base_losses,
            target_update_l2=target_update_l2,
        ),
        _train_dense_control(
            torch,
            F,
            nn,
            base,
            hidden,
            targets,
            vocab_size,
            position_inputs,
            label="rank_matched_token_position_dense_residual",
            target_parameter_count=sparse_value_params,
            steps=dense_steps,
            base_losses=base_losses,
            target_update_l2=target_update_l2,
        ),
    ]
    return rows


def _train_dense_control(
    torch: Any,
    F: Any,
    nn: Any,
    base: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    features: Any,
    *,
    label: str,
    target_parameter_count: int,
    steps: int,
    base_losses: Any,
    target_update_l2: float,
) -> dict[str, Any]:
    split = max(1, int(features.shape[1]) // 2)
    rank = max(1, target_parameter_count // max(1, int(features.shape[-1]) + int(hidden.shape[-1])))
    adapter = _LowRankCausalDenseAdapter(nn, int(features.shape[-1]), int(hidden.shape[-1]), rank)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=3e-3)
    train_features = features[:, :split, :]
    train_hidden = hidden[:, :split, :]
    train_targets = targets[:, :split]
    for _ in range(max(1, steps)):
        optimizer.zero_grad(set_to_none=True)
        update = adapter(train_features)
        logits = base.decode(train_hidden + update)
        ce_loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, vocab_size),
            train_targets[:, :-1].reshape(-1),
        )
        update_l2 = update[:, :-1, :].reshape(-1, update.shape[-1]).norm(dim=-1).mean()
        norm_penalty = (update_l2 - target_update_l2) ** 2
        loss = ce_loss + norm_penalty
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        update = adapter(features)
        logits = base.decode(hidden + update)
        losses = _per_token_ce(F, logits, targets, vocab_size)
        all_loss = float(losses.mean().item())
        heldout_mask = _heldout_mask(losses.numel(), int(hidden.shape[1] - 1))
        heldout_loss = float(losses[heldout_mask].mean().item())
        train_loss = float(losses[~heldout_mask].mean().item())
        heldout_base_delta = heldout_loss - float(base_losses[heldout_mask].mean().item())
        all_base_delta = all_loss - float(base_losses.mean().item())
        update_l2 = update[:, :-1, :].reshape(-1, update.shape[-1]).norm(dim=-1)
    return {
        "control": label,
        "rank": rank,
        "target_parameter_count": target_parameter_count,
        "actual_parameter_count": _parameter_count(adapter),
        "target_residual_update_l2": target_update_l2,
        "train_ce_loss": train_loss,
        "heldout_ce_loss": heldout_loss,
        "all_ce_loss": all_loss,
        "heldout_delta_vs_base_ce": heldout_base_delta,
        "all_delta_vs_base_ce": all_base_delta,
        "residual_update_l2_mean": float(update_l2.mean().item()),
    }


class _LowRankCausalDenseAdapter:
    def __init__(self, nn: Any, input_dim: int, hidden_dim: int, rank: int) -> None:
        super_cls = nn.Module

        class Adapter(super_cls):
            def __init__(self) -> None:
                super().__init__()
                self.down = nn.Linear(input_dim, rank, bias=False)
                self.up = nn.Linear(rank, hidden_dim, bias=False)

            def forward(self, x: Any) -> Any:
                return self.up(self.down(x))

        self._module = Adapter()

    def __call__(self, x: Any) -> Any:
        return self._module(x)

    def parameters(self) -> Any:
        return self._module.parameters()


def _source_probe_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for value_path, arm in (
        (PARTNER_VALUE_PATH, TRANSFER_ARM),
        (PARTNER_VALUE_PATH, DIRECT_ARM),
        (OWN_VALUE_PATH, TRANSFER_ARM),
        (OWN_VALUE_PATH, DIRECT_ARM),
    ):
        selected = [
            row for row in rows if row.get("value_path") == value_path and row.get("arm") == arm
        ]
        if not selected:
            continue
        seq_len_minus_one = _infer_seq_len_minus_one(selected)
        heldout = [
            row for row in selected if _position(row, seq_len_minus_one) >= seq_len_minus_one // 2
        ]
        train = [
            row for row in selected if _position(row, seq_len_minus_one) < seq_len_minus_one // 2
        ]
        metrics.append(
            {
                "value_path": value_path,
                "arm": arm,
                "train_ce_loss": _mean_field(train, "ce_loss"),
                "heldout_ce_loss": _mean_field(heldout, "ce_loss"),
                "all_ce_loss": _mean_field(selected, "ce_loss"),
                "heldout_residual_update_l2": _mean_field(heldout, "residual_update_l2"),
                "token_count": len(selected),
                "heldout_token_count": len(heldout),
            }
        )
    keyed = {(row["value_path"], row["arm"]): row for row in metrics}
    for row in metrics:
        baseline = keyed.get((row["value_path"], DIRECT_ARM))
        if baseline is not None:
            row["heldout_delta_vs_direct_ce"] = (
                float(row["heldout_ce_loss"]) - float(baseline["heldout_ce_loss"])
            )
            row["all_delta_vs_direct_ce"] = (
                float(row["all_ce_loss"]) - float(baseline["all_ce_loss"])
            )
    return metrics


def _dense_gate_rows(
    source_metrics: list[dict[str, Any]],
    dense_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    keyed = {(row["value_path"], row["arm"]): row for row in source_metrics}
    transfer = keyed.get((PARTNER_VALUE_PATH, TRANSFER_ARM), {})
    own_transfer = keyed.get((OWN_VALUE_PATH, TRANSFER_ARM), {})
    own_direct = keyed.get((OWN_VALUE_PATH, DIRECT_ARM), {})
    causal_dense = _row_by_name(dense_rows, "control", "rank_matched_causal_dense_residual")
    token_dense = _row_by_name(dense_rows, "control", "rank_matched_token_position_dense_residual")
    transfer_partner_delta = _float_or_none(transfer.get("heldout_delta_vs_direct_ce"))
    own_delta = None
    if own_transfer and own_direct:
        own_delta = float(own_transfer["heldout_ce_loss"]) - float(own_direct["heldout_ce_loss"])
    causal_dense_delta = _float_or_none(causal_dense.get("heldout_delta_vs_base_ce"))
    token_dense_delta = _float_or_none(token_dense.get("heldout_delta_vs_base_ce"))
    sparse_l2 = _float_or_none(transfer.get("heldout_residual_update_l2"))
    causal_dense_l2 = _float_or_none(causal_dense.get("residual_update_l2_mean"))
    norm_budget_passes = (
        sparse_l2 is not None
        and causal_dense_l2 is not None
        and causal_dense_l2 <= max(0.25, sparse_l2 * 2.0)
    )
    return [
        _criterion(
            "dense_controls_trained",
            len(dense_rows) == 2,
            "causal and token-position rank-matched dense controls are present",
            len(dense_rows),
            "missing dense control rows",
        ),
        _criterion(
            "transfer_heldout_partner_gain_present",
            transfer_partner_delta is not None and transfer_partner_delta < 0.0,
            "source transfer objective improves held-out partner CE versus direct support",
            transfer_partner_delta,
            "source transfer objective lacks held-out partner gain",
        ),
        _criterion(
            "own_heldout_ce_guardrail",
            own_delta is not None and own_delta <= 0.02,
            "source transfer objective held-out own CE worsens by no more than 0.02",
            own_delta,
            "source transfer objective damages held-out own CE",
        ),
        _criterion(
            "causal_dense_residual_norm_guardrail",
            norm_budget_passes,
            "causal dense residual update L2 is no more than 2x sparse transfer held-out L2, with 0.25 floor",
            {
                "sparse_transfer_heldout_l2": sparse_l2,
                "causal_dense_l2": causal_dense_l2,
            },
            "causal dense residual used a much larger residual update than the sparse transfer source",
        ),
        _criterion(
            "sparse_transfer_beats_causal_dense_control",
            (
                norm_budget_passes
                and
                transfer_partner_delta is not None
                and causal_dense_delta is not None
                and transfer_partner_delta < causal_dense_delta
            ),
            "held-out sparse transfer gain is more negative than rank-matched causal dense residual gain",
            {
                "sparse_transfer_delta_vs_direct": transfer_partner_delta,
                "causal_dense_delta_vs_base": causal_dense_delta,
            },
            "rank-matched causal dense residual matches or beats sparse transfer",
        ),
        _criterion(
            "causal_dense_beats_token_position_dense_null",
            (
                causal_dense_delta is not None
                and token_dense_delta is not None
                and causal_dense_delta < token_dense_delta
            ),
            "causal dense control beats token-position dense null",
            {
                "causal_dense_delta_vs_base": causal_dense_delta,
                "token_position_dense_delta_vs_base": token_dense_delta,
            },
            "causal dense control does not beat token-position dense null",
        ),
    ]


def _source_gate_rows(
    source_probe_dir: Path,
    source_summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    source_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _criterion(
            "source_probe_present",
            bool(source_summary),
            "source transfer-objective probe summary exists",
            str(source_probe_dir / "summary.json"),
            "missing source probe summary",
        ),
        _criterion(
            "source_probe_passed",
            source_summary.get("status") == "pass",
            "source transfer-objective probe status is pass",
            source_summary.get("status", "missing"),
            "source transfer-objective probe did not pass",
        ),
        _criterion(
            "source_per_token_rows_present",
            bool(source_rows),
            "source per-token rows are available",
            len(source_rows),
            "missing source per-token rows",
        ),
        _criterion(
            "source_required_metrics_present",
            len(source_metrics) == 4,
            "source has partner/own transfer and direct metrics",
            len(source_metrics),
            "source per-token metrics are incomplete",
        ),
    ]


def _source_sparse_update_l2(source_metrics: list[dict[str, Any]]) -> float:
    row = _row_by_name(
        [
            metric
            for metric in source_metrics
            if metric.get("value_path") == PARTNER_VALUE_PATH
            and metric.get("arm") == TRANSFER_ARM
        ],
        "arm",
        TRANSFER_ARM,
    )
    value = _float_or_none(row.get("heldout_residual_update_l2"))
    return value if value is not None and value > 0.0 else 0.5


def _summary(
    *,
    status: str,
    decision: str,
    claim_status: str,
    start: float,
    source_probe_dir: Path,
    config_path: Path,
    dense_steps: int,
    dense_rows: list[dict[str, Any]],
    source_metrics: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    return {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "source_probe_dir": str(source_probe_dir),
        "config_path": str(config_path),
        "dense_steps": dense_steps,
        "dense_control_count": len(dense_rows),
        "source_metric_count": len(source_metrics),
        "dense_control_rows": dense_rows,
        "source_metrics": source_metrics,
        "gate_criteria": gate_rows,
        "failures": failures,
        "selected_next_step": (
            "replicate the dense-control transfer gate on seed2/local and fetched RunPod packets"
            if status == "pass"
            else "treat ACSR transfer as not separated from dense controls and design a stricter mechanism benchmark"
        ),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }


def _load_source_probe(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    summary = _read_json(path / "summary.json")
    rows = _read_csv(path / "per_token_metrics.csv")
    gate_rows = _read_csv(path / "gate_criteria.csv")
    return summary, rows, gate_rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _infer_seq_len_minus_one(rows: list[dict[str, Any]]) -> int:
    count = len(rows)
    for batch in range(8, 0, -1):
        if count % batch == 0:
            return max(1, count // batch)
    return max(1, int(math.sqrt(count)))


def _position(row: dict[str, Any], seq_len_minus_one: int) -> int:
    return int(float(row.get("token_index", 0))) % seq_len_minus_one


def _heldout_mask(count: int, seq_len_minus_one: int) -> Any:
    import torch

    positions = torch.arange(count) % max(1, seq_len_minus_one)
    return positions >= max(1, seq_len_minus_one // 2)


def _per_token_ce(F: Any, logits: Any, targets: Any, vocab_size: int) -> Any:
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab_size),
        targets[:, :-1].reshape(-1),
        reduction="none",
    )


def _mean_field(rows: list[dict[str, Any]], field: str) -> float | str:
    values = [_float_or_none(row.get(field)) for row in rows]
    numeric = [value for value in values if value is not None]
    if not numeric:
        return ""
    return sum(numeric) / len(numeric)


def _row_by_name(rows: list[dict[str, Any]], field: str, value: str) -> dict[str, Any]:
    for row in rows:
        if row.get(field) == value:
            return row
    return {}


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parameter_count(module: Any) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _criterion(
    criterion: str,
    passed: bool,
    threshold: str,
    actual: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": actual,
        "failure_reason": "" if passed else failure_reason,
    }


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    dense_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "dense_control_metrics.csv", dense_rows)
    _write_csv(out_dir / "source_probe_metrics.csv", source_rows)
    _write_csv(out_dir / "gate_criteria.csv", gate_rows)
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [{"status": "missing"}]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ACSR Dense Residual Transfer Control",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Dense steps: `{summary['dense_steps']}`",
        "",
        "This bounded local control trains rank-matched dense residual adapters from "
        "causal and token-position features, then compares the source ACSR transfer "
        "probe's held-out partner-through-values CE gain with the dense adapter's "
        "held-out CE gain. It is a fail-closed mechanism control, not a promotion.",
    ]
    if summary.get("failures"):
        lines.extend(["", "## Failures"])
        for row in summary["failures"]:
            lines.append(f"- `{row['criterion']}`: {row['failure_reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    parser.add_argument("--source-probe-dir", type=Path, default=DEFAULT_SOURCE_PROBE)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dense-steps", type=int, default=120)
    args = parser.parse_args()
    summary = run_acsr_dense_residual_transfer_control(
        source_probe_dir=args.source_probe_dir,
        config_path=args.config,
        out_dir=args.out,
        dense_steps=args.dense_steps,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
