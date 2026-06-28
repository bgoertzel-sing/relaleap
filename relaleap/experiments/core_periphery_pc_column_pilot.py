"""Tiny local core/periphery PC-column pilot.

This is a CPU-scale mechanism check, not promotion evidence. It uses synthetic
frozen hidden states so the split core/periphery learning rule can be measured
without spending GPU time or depending on notebook state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONTRACT = Path("results/reports/core_periphery_pc_column_design/summary.json")
DEFAULT_OUT_DIR = Path("results/reports/core_periphery_pc_column_pilot")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "variant_metrics.csv",
    "intervention_fingerprints.csv",
    "gate_criteria.csv",
    "notes.md",
)

REQUIRED_VARIANTS = (
    "core_periphery_pc",
    "current_sparse_acsr_contextual_router",
    "dense_rank_norm_residual",
    "parameter_matched_causal_mlp",
    "random_support_router",
    "frequency_support_router",
    "no_core_ablation",
    "no_periphery_ablation",
    "equal_plasticity_core_periphery",
    "shuffled_core_periphery_assignment",
    "lambda_zero_residual",
    "token_position_only_router",
)


@dataclass(frozen=True)
class _VariantSpec:
    name: str
    kind: str
    core_lr_scale: float = 0.25
    periphery_lr_scale: float = 1.0
    use_core: bool = True
    use_periphery: bool = True
    router: str = "contextual"
    shuffled_eval: bool = False


def run_core_periphery_pc_column_pilot(
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 7,
    steps_per_task: int = 24,
) -> dict[str, Any]:
    """Run the bounded local pilot and write report artifacts."""

    start = time.time()
    contract = _read_json(contract_path)
    preflight = _preflight_gates(contract_path, contract)
    if any(not row["passed"] for row in preflight):
        summary = _summary(
            status="fail",
            decision="core_periphery_pc_column_pilot_failed_closed",
            claim_status="design_contract_missing_or_not_ready",
            selected_next_step="repair core/periphery design contract before running pilot",
            start=start,
            contract_path=contract_path,
            out_dir=out_dir,
            variant_rows=[],
            fingerprint_rows=[],
            gate_rows=preflight,
            seed=seed,
            steps_per_task=steps_per_task,
        )
        _write_artifacts(out_dir, summary)
        return summary

    variant_rows, fingerprint_rows = _run_torch_pilot(seed=seed, steps_per_task=steps_per_task)
    gate_rows = preflight + _pilot_gates(variant_rows, fingerprint_rows)
    hard_failures = [row for row in gate_rows if not row["passed"] and row["severity"] == "hard"]
    claim_blockers = [row for row in gate_rows if not row["passed"] and row["severity"] != "hard"]
    primary = _row_by_variant(variant_rows, "core_periphery_pc")
    dense = _row_by_variant(variant_rows, "dense_rank_norm_residual")
    mlp = _row_by_variant(variant_rows, "parameter_matched_causal_mlp")
    status = "fail" if hard_failures else "pass"
    if status == "fail":
        decision = "core_periphery_pc_column_pilot_failed_closed"
        claim_status = "runtime_or_artifact_contract_failed"
        next_step = "repair the tiny pilot before interpreting mechanism evidence"
    elif claim_blockers:
        decision = "core_periphery_pc_column_pilot_recorded_but_blocked"
        claim_status = "pilot_evidence_insufficient_for_gpu_or_promotion"
        next_step = "tighten the split mechanism or controls locally before any RunPod or Colab validation"
    else:
        decision = "core_periphery_pc_column_pilot_local_candidate"
        claim_status = "tiny_local_candidate_not_promoted"
        next_step = "repeat the tiny local pilot on a second seed before any RunPod or Colab validation"

    summary = _summary(
        status=status,
        decision=decision,
        claim_status=claim_status,
        selected_next_step=next_step,
        start=start,
        contract_path=contract_path,
        out_dir=out_dir,
        variant_rows=variant_rows,
        fingerprint_rows=fingerprint_rows,
        gate_rows=gate_rows,
        seed=seed,
        steps_per_task=steps_per_task,
        primary_result={
            "primary_variant": "core_periphery_pc",
            "dense_control": "dense_rank_norm_residual",
            "mlp_control": "parameter_matched_causal_mlp",
            "core_minus_dense_anchor_mse_drift": _delta(
                primary, dense, "anchor_mse_drift_after_task_b"
            ),
            "core_minus_mlp_anchor_mse_drift": _delta(
                primary, mlp, "anchor_mse_drift_after_task_b"
            ),
            "core_periphery_update_norm_ratio": primary.get("plasticity_ratio"),
            "periphery_first_minus_core_first_prune_delta": primary.get(
                "periphery_first_minus_core_first_prune_delta"
            ),
            "requires_gpu_now": False,
            "interpretation": (
                "Negative drift deltas favor core/periphery retention. Positive prune delta means "
                "core pruning is more damaging than periphery pruning. This remains a synthetic "
                "CPU pilot and is not default-router or GPU evidence."
            ),
        },
    )
    _write_artifacts(out_dir, summary)
    return summary


def _run_torch_pilot(*, seed: int, steps_per_task: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("core/periphery pilot requires torch") from exc

    torch.manual_seed(seed)
    hidden_dim = 10
    n = 24
    task_a = _make_task(torch, n=n, hidden_dim=hidden_dim, context=0)
    task_b = _make_task(torch, n=n, hidden_dim=hidden_dim, context=1)
    decoder = torch.randn(hidden_dim, 5) / hidden_dim**0.5

    specs = [
        _VariantSpec("core_periphery_pc", "sparse_split"),
        _VariantSpec("current_sparse_acsr_contextual_router", "sparse", core_lr_scale=1.0, periphery_lr_scale=0.0, use_periphery=False),
        _VariantSpec("random_support_router", "sparse_null", router="random"),
        _VariantSpec("frequency_support_router", "sparse_null", router="frequency"),
        _VariantSpec("no_core_ablation", "ablation", use_core=False, periphery_lr_scale=1.0),
        _VariantSpec("no_periphery_ablation", "ablation", use_periphery=False, core_lr_scale=0.25),
        _VariantSpec("equal_plasticity_core_periphery", "ablation", core_lr_scale=1.0, periphery_lr_scale=1.0),
        _VariantSpec("shuffled_core_periphery_assignment", "null", shuffled_eval=True),
        _VariantSpec("lambda_zero_residual", "null", use_core=False, use_periphery=False),
        _VariantSpec("token_position_only_router", "router_null", router="token_position"),
        _VariantSpec("dense_rank_norm_residual", "dense"),
        _VariantSpec("parameter_matched_causal_mlp", "mlp"),
    ]

    rows: list[dict[str, Any]] = []
    fingerprints: list[dict[str, Any]] = []
    for spec in specs:
        if spec.kind == "dense":
            model = _DenseResidual(nn, hidden_dim)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.08)
        elif spec.kind == "mlp":
            model = _MLPResidual(nn, hidden_dim)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        else:
            model = _SplitResidual(nn, hidden_dim, spec)
            groups = []
            if list(model.core_parameters()):
                groups.append({"params": list(model.core_parameters()), "lr": 0.08 * spec.core_lr_scale})
            if list(model.periphery_parameters()):
                groups.append({"params": list(model.periphery_parameters()), "lr": 0.08 * spec.periphery_lr_scale})
            if not groups:
                groups.append({"params": [model.dummy], "lr": 0.0})
            optimizer = torch.optim.SGD(groups)

        if spec.name != "lambda_zero_residual":
            _train_sequence(torch, F, model, optimizer, task_a, steps_per_task)
            anchor_after_a = _mse(F, model, task_a)
            logits_after_a = _logits(torch, model, task_a, decoder)
            _train_sequence(torch, F, model, optimizer, task_b, steps_per_task)
        else:
            anchor_after_a = _mse(F, model, task_a)
            logits_after_a = _logits(torch, model, task_a, decoder)

        anchor_after_b = _mse(F, model, task_a)
        transfer_after_b = _mse(F, model, task_b)
        logits_after_b = _logits(torch, model, task_a, decoder)
        ba_model = _clone_fresh(torch, nn, spec, hidden_dim)
        ba_optimizer = _optimizer_for(torch, ba_model, spec)
        if spec.name != "lambda_zero_residual":
            _train_sequence(torch, F, ba_model, ba_optimizer, task_b, steps_per_task)
            _train_sequence(torch, F, ba_model, ba_optimizer, task_a, steps_per_task)
        commutator = float(F.mse_loss(model(task_a["hidden"]), ba_model(task_a["hidden"])).detach().item())
        core_prune, periphery_prune = _prune_deltas(F, model, task_a)
        fingerprints.extend(_fingerprints(F, spec.name, model, task_a, task_b))
        core_update = float(getattr(model, "core_update_norm", lambda: 0.0)())
        periphery_update = float(getattr(model, "periphery_update_norm", lambda: 0.0)())
        rows.append(
            {
                "variant": spec.name,
                "kind": spec.kind,
                "anchor_mse_after_task_a": float(anchor_after_a),
                "anchor_mse_after_task_b": float(anchor_after_b),
                "anchor_mse_drift_after_task_b": float(anchor_after_b - anchor_after_a),
                "transfer_mse_after_task_b": float(transfer_after_b),
                "anchor_logit_mse_churn": float(F.mse_loss(logits_after_b, logits_after_a).detach().item()),
                "finite_update_commutator": commutator,
                "core_residual_norm": float(getattr(model, "core_residual_norm", lambda: 0.0)()),
                "periphery_residual_norm": float(getattr(model, "periphery_residual_norm", lambda: 0.0)()),
                "core_update_norm": core_update,
                "periphery_update_norm": periphery_update,
                "plasticity_ratio": _safe_divide(periphery_update, core_update),
                "core_first_prune_delta": core_prune,
                "periphery_first_prune_delta": periphery_prune,
                "periphery_first_minus_core_first_prune_delta": core_prune - periphery_prune,
            }
        )
    return rows, fingerprints


def _make_task(torch: Any, *, n: int, hidden_dim: int, context: int) -> dict[str, Any]:
    hidden = torch.randn(n, hidden_dim)
    hidden[:, 0] = float(context)
    hidden[:, 1] = torch.linspace(-1.0, 1.0, n)
    generic = torch.zeros(hidden_dim)
    generic[2:5] = torch.tensor([0.35, -0.25, 0.15])
    specific = torch.zeros(hidden_dim)
    if context == 0:
        specific[5:8] = torch.tensor([0.24, -0.16, 0.10])
    else:
        specific[5:8] = torch.tensor([-0.22, 0.18, -0.12])
    target = hidden + generic + specific + 0.03 * torch.tanh(hidden)
    return {"hidden": hidden, "target": target, "context": context}


def _SplitResidual(nn: Any, hidden_dim: int, spec: _VariantSpec) -> Any:
    import torch

    class SplitResidual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spec = spec
            self.core = nn.Parameter(torch.zeros(2, hidden_dim))
            self.periphery = nn.Parameter(torch.zeros(2, hidden_dim))
            self.dummy = nn.Parameter(torch.zeros(()))
            self.initial_core = self.core.detach().clone()
            self.initial_periphery = self.periphery.detach().clone()

        def _route(self, hidden: Any) -> Any:
            if self.spec.router == "random":
                return (torch.arange(hidden.shape[0], device=hidden.device) % 2).long()
            if self.spec.router == "frequency":
                return torch.zeros(hidden.shape[0], dtype=torch.long, device=hidden.device)
            if self.spec.router == "token_position":
                return (hidden[:, 1] > 0).long()
            return (hidden[:, 0] > 0.5).long()

        def forward(self, hidden: Any, ablate: str | None = None) -> Any:
            route = self._route(hidden)
            core = self.core[route]
            periphery = self.periphery[1 - route] if self.spec.shuffled_eval else self.periphery[route]
            residual = torch.zeros_like(hidden) + self.dummy * 0.0
            if self.spec.use_core and ablate != "core":
                residual = residual + core
            if self.spec.use_periphery and ablate != "periphery":
                residual = residual + periphery
            return hidden + residual

        def core_parameters(self) -> list[Any]:
            return [self.core] if self.spec.use_core else []

        def periphery_parameters(self) -> list[Any]:
            return [self.periphery] if self.spec.use_periphery else []

        def core_residual_norm(self) -> float:
            return float(self.core.detach().norm().item()) if self.spec.use_core else 0.0

        def periphery_residual_norm(self) -> float:
            return float(self.periphery.detach().norm().item()) if self.spec.use_periphery else 0.0

        def core_update_norm(self) -> float:
            return float((self.core.detach() - self.initial_core).norm().item()) if self.spec.use_core else 0.0

        def periphery_update_norm(self) -> float:
            return float((self.periphery.detach() - self.initial_periphery).norm().item()) if self.spec.use_periphery else 0.0

    return SplitResidual()


def _DenseResidual(nn: Any, hidden_dim: int) -> Any:
    import torch

    class DenseResidual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.delta = nn.Parameter(torch.zeros(hidden_dim))
            self.initial = self.delta.detach().clone()

        def forward(self, hidden: Any, ablate: str | None = None) -> Any:
            if ablate in {"core", "periphery"}:
                return hidden
            return hidden + self.delta

        def core_update_norm(self) -> float:
            return float((self.delta.detach() - self.initial).norm().item())

    return DenseResidual()


def _MLPResidual(nn: Any, hidden_dim: int) -> Any:
    class MLPResidual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for module in self.net.modules():
                if hasattr(module, "weight"):
                    nn.init.zeros_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)

        def forward(self, hidden: Any, ablate: str | None = None) -> Any:
            if ablate in {"core", "periphery"}:
                return hidden
            return hidden + self.net(hidden)

    return MLPResidual()


def _train_sequence(torch: Any, F: Any, model: Any, optimizer: Any, task: dict[str, Any], steps: int) -> None:
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred = model(task["hidden"])
        loss = F.mse_loss(pred, task["target"])
        if hasattr(model, "core"):
            loss = loss + 0.015 * model.core.pow(2).mean()
        loss.backward()
        optimizer.step()


def _clone_fresh(torch: Any, nn: Any, spec: _VariantSpec, hidden_dim: int) -> Any:
    if spec.kind == "dense":
        return _DenseResidual(nn, hidden_dim)
    if spec.kind == "mlp":
        return _MLPResidual(nn, hidden_dim)
    return _SplitResidual(nn, hidden_dim, spec)


def _optimizer_for(torch: Any, model: Any, spec: _VariantSpec) -> Any:
    if spec.kind == "dense":
        return torch.optim.SGD(model.parameters(), lr=0.08)
    if spec.kind == "mlp":
        return torch.optim.SGD(model.parameters(), lr=0.05)
    groups = []
    if list(model.core_parameters()):
        groups.append({"params": list(model.core_parameters()), "lr": 0.08 * spec.core_lr_scale})
    if list(model.periphery_parameters()):
        groups.append({"params": list(model.periphery_parameters()), "lr": 0.08 * spec.periphery_lr_scale})
    if not groups:
        groups.append({"params": [model.dummy], "lr": 0.0})
    return torch.optim.SGD(groups)


def _mse(F: Any, model: Any, task: dict[str, Any]) -> float:
    return float(F.mse_loss(model(task["hidden"]), task["target"]).detach().item())


def _logits(torch: Any, model: Any, task: dict[str, Any], decoder: Any) -> Any:
    return model(task["hidden"]).detach() @ decoder


def _prune_deltas(F: Any, model: Any, task: dict[str, Any]) -> tuple[float, float]:
    full = F.mse_loss(model(task["hidden"]), task["target"]).detach().item()
    core = F.mse_loss(model(task["hidden"], ablate="core"), task["target"]).detach().item()
    periphery = F.mse_loss(model(task["hidden"], ablate="periphery"), task["target"]).detach().item()
    return float(core - full), float(periphery - full)


def _fingerprints(F: Any, variant: str, model: Any, task_a: dict[str, Any], task_b: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for unit in ("core", "periphery"):
        full_a = F.mse_loss(model(task_a["hidden"]), task_a["target"]).detach().item()
        full_b = F.mse_loss(model(task_b["hidden"]), task_b["target"]).detach().item()
        ablate_a = F.mse_loss(model(task_a["hidden"], ablate=unit), task_a["target"]).detach().item()
        ablate_b = F.mse_loss(model(task_b["hidden"], ablate=unit), task_b["target"]).detach().item()
        rows.append(
            {
                "variant": variant,
                "unit": unit,
                "necessity_anchor_delta": float(ablate_a - full_a),
                "necessity_transfer_delta": float(ablate_b - full_b),
                "selectivity_delta": float((ablate_b - full_b) - (ablate_a - full_a)),
                "off_target_leakage": float(max(0.0, ablate_a - full_a)),
            }
        )
    return rows


def _pilot_gates(variant_rows: list[dict[str, Any]], fingerprint_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants = {row["variant"] for row in variant_rows}
    primary = _row_by_variant(variant_rows, "core_periphery_pc")
    dense = _row_by_variant(variant_rows, "dense_rank_norm_residual")
    mlp = _row_by_variant(variant_rows, "parameter_matched_causal_mlp")
    return [
        _criterion(
            "required_controls_present",
            set(REQUIRED_VARIANTS).issubset(variants),
            "hard",
            "all contract controls are present",
            sorted(variants),
            "missing mandatory control rows",
        ),
        _criterion(
            "core_periphery_update_separation",
            float(primary.get("plasticity_ratio") or 0.0) > 1.5,
            "claim",
            "periphery update norm meaningfully exceeds core update norm",
            primary.get("plasticity_ratio"),
            "split may be accounting-only",
        ),
        _criterion(
            "matched_dense_retention",
            float(primary.get("anchor_mse_drift_after_task_b") or 0.0)
            <= float(dense.get("anchor_mse_drift_after_task_b") or 0.0),
            "claim",
            "core/periphery anchor drift is no worse than dense control",
            {
                "core_periphery": primary.get("anchor_mse_drift_after_task_b"),
                "dense": dense.get("anchor_mse_drift_after_task_b"),
            },
            "dense/rank/norm control remains active",
        ),
        _criterion(
            "matched_mlp_retention",
            float(primary.get("anchor_mse_drift_after_task_b") or 0.0)
            <= float(mlp.get("anchor_mse_drift_after_task_b") or 0.0),
            "claim",
            "core/periphery anchor drift is no worse than parameter-matched MLP",
            {
                "core_periphery": primary.get("anchor_mse_drift_after_task_b"),
                "mlp": mlp.get("anchor_mse_drift_after_task_b"),
            },
            "MLP control remains active",
        ),
        _criterion(
            "periphery_first_pruning_signal",
            float(primary.get("periphery_first_minus_core_first_prune_delta") or 0.0) > 0.0,
            "claim",
            "core pruning is more damaging than periphery pruning on anchors",
            primary.get("periphery_first_minus_core_first_prune_delta"),
            "protected core is not causally distinguished by pruning",
        ),
        _criterion(
            "intervention_rows_present",
            len(fingerprint_rows) >= len(REQUIRED_VARIANTS) * 2,
            "hard",
            "core/periphery intervention fingerprint rows exist for every variant",
            len(fingerprint_rows),
            "missing intervention fingerprint artifacts",
        ),
    ]


def _preflight_gates(contract_path: Path, contract: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _criterion(
            "contract_present",
            contract_path.is_file(),
            "hard",
            "design contract summary exists",
            str(contract_path) if contract_path.is_file() else "missing",
            "run core_periphery_pc_column_design first",
        ),
        _criterion(
            "contract_ready_for_tiny_pilot",
            contract.get("status") == "pass"
            and contract.get("scientific_gate") == "ready_for_tiny_pilot",
            "hard",
            "design contract explicitly allows a tiny local pilot",
            {
                "status": contract.get("status"),
                "scientific_gate": contract.get("scientific_gate"),
            },
            "contract is not ready for pilot",
        ),
    ]


def _summary(
    *,
    status: str,
    decision: str,
    claim_status: str,
    selected_next_step: str,
    start: float,
    contract_path: Path,
    out_dir: Path,
    variant_rows: list[dict[str, Any]],
    fingerprint_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    seed: int,
    steps_per_task: int,
    primary_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "scientific_gate": "blocked" if decision.endswith("_blocked") or status == "fail" else "ready_for_repeat_only",
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "seed": seed,
        "steps_per_task": steps_per_task,
        "contract_path": str(contract_path),
        "variant_metrics": variant_rows,
        "intervention_fingerprints": fingerprint_rows,
        "gate_criteria": gate_rows,
        "primary_result": primary_result or {},
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "generated_from_head": _git_commit(),
        "dirty_diff_hash": _dirty_diff_hash(),
    }


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "variant_metrics.csv", summary["variant_metrics"])
    _write_csv(out_dir / "intervention_fingerprints.csv", summary["intervention_fingerprints"])
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
        "# Core/Periphery PC Column Pilot",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Scientific gate: `{summary['scientific_gate']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        "",
        "This is a tiny synthetic CPU pilot. It is not GPU evidence and does not promote a default.",
        "",
        f"Next step: {summary['selected_next_step']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _row_by_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    for row in rows:
        if row.get("variant") == variant:
            return row
    return {}


def _delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    if key not in left or key not in right:
        return None
    return float(left[key]) - float(right[key])


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if abs(denominator) < 1e-12:
        return None
    return numerator / denominator


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
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps-per-task", type=int, default=24)
    args = parser.parse_args(argv)
    summary = run_core_periphery_pc_column_pilot(
        contract_path=args.contract,
        out_dir=args.out,
        seed=args.seed,
        steps_per_task=args.steps_per_task,
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
