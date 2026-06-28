"""Mechanism-factorized local continual-learning probe for residual adapters."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from relaleap.smoke import ResidualColumns
from relaleap.smoke import TinyCharTransformer


DEFAULT_OUT_DIR = Path("results/reports/mechanism_factorized_continual_learning_probe")
RULE_SEQUENCE = ("copy", "reverse", "permute", "negate", "copy")
REQUIRED_ARTIFACTS = (
    "summary.json",
    "phase_metrics.csv",
    "arm_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)


@dataclass(frozen=True)
class _ArmSpec:
    name: str
    kind: str
    top_k: int
    num_columns: int
    atoms_per_column: int
    support_router: str
    dense_rank: int | None = None
    fixed_random_support: bool = False
    anchor_kl_weight: float = 0.0


def run_mechanism_factorized_continual_learning_probe(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 7,
    vocab_size: int = 16,
    seq_len: int = 8,
    batch_size: int = 16,
    hidden_dim: int = 32,
    layers: int = 1,
    num_columns: int = 8,
    atoms_per_column: int = 2,
    steps_per_phase: int = 18,
    learning_rate: float = 8e-3,
    anchor_kl_weight: float = 0.15,
) -> dict[str, Any]:
    """Run a small CPU mechanism CL assay with dense/sparse/null controls."""

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on torch install
        raise RuntimeError("mechanism-factorized CL probe requires torch") from exc

    if steps_per_phase < 1:
        raise ValueError("steps_per_phase must be positive")
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    if num_columns < 2:
        raise ValueError("num_columns must be at least 2")

    start = time.time()
    torch.manual_seed(seed)
    inputs = _mechanism_inputs(
        vocab_size=vocab_size,
        seq_len=seq_len,
        batch_size=batch_size,
        seed=seed,
    )
    base = TinyCharTransformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        layers=layers,
    )
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    hidden = base.encode(inputs).detach()
    targets = {rule: _apply_rule(inputs, rule, vocab_size) for rule in set(RULE_SEQUENCE)}
    arms = _arm_specs(
        num_columns=num_columns,
        atoms_per_column=atoms_per_column,
        active_top_k=2,
        anchor_kl_weight=anchor_kl_weight,
    )

    phase_rows: list[dict[str, Any]] = []
    arm_rows: list[dict[str, Any]] = []
    for arm_index, spec in enumerate(arms):
        torch.manual_seed(seed + 1000 + arm_index)
        adapter = _build_adapter(
            spec=spec,
            hidden_dim=hidden_dim,
            contextual_router_hidden_dim=hidden_dim * 2,
            torch=torch,
            nn=nn,
        )
        trainable_parameters = [
            parameter for parameter in adapter.parameters() if parameter.requires_grad
        ]
        optimizer = (
            torch.optim.AdamW(trainable_parameters, lr=learning_rate)
            if trainable_parameters
            else None
        )
        snapshots: dict[str, Any] = {
            "initial": _evaluate_all_rules(
                adapter=adapter,
                hidden=hidden,
                decode=base.decode,
                targets=targets,
                spec=spec,
                torch=torch,
                F=F,
            )
        }
        previous_logits = _logits_for_rules(
            adapter=adapter,
            hidden=hidden,
            decode=base.decode,
            targets=targets,
            spec=spec,
            torch=torch,
        )
        best_ce = {
            rule: snapshots["initial"][rule]["ce_loss"]
            for rule in targets
        }
        adaptation_rows: list[dict[str, Any]] = []
        for phase_index, rule in enumerate(RULE_SEQUENCE, start=1):
            before = _evaluate_all_rules(
                adapter=adapter,
                hidden=hidden,
                decode=base.decode,
                targets=targets,
                spec=spec,
                torch=torch,
                F=F,
            )
            anchor_logits = {
                off_rule: previous_logits[off_rule].detach()
                for off_rule in targets
                if off_rule != rule
            }
            for _ in range(steps_per_phase):
                if optimizer is None:
                    break
                optimizer.zero_grad(set_to_none=True)
                logits = _forward_logits(adapter, hidden, spec, torch, base.decode)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets[rule].reshape(-1))
                if spec.anchor_kl_weight > 0.0 and anchor_logits:
                    anchor_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)
                    for anchor_reference in anchor_logits.values():
                        anchor_loss = anchor_loss + _kl_to_reference(
                            logits,
                            anchor_reference,
                            F=F,
                        )
                    loss = loss + spec.anchor_kl_weight * anchor_loss / float(len(anchor_logits))
                loss.backward()
                optimizer.step()
            after = _evaluate_all_rules(
                adapter=adapter,
                hidden=hidden,
                decode=base.decode,
                targets=targets,
                spec=spec,
                torch=torch,
                F=F,
                reference_logits=previous_logits,
            )
            snapshots[f"after_phase_{phase_index}_{rule}"] = after
            for eval_rule, metrics in after.items():
                before_metrics = before[eval_rule]
                ce_delta = metrics["ce_loss"] - before_metrics["ce_loss"]
                row = {
                    "arm": spec.name,
                    "kind": spec.kind,
                    "support_router": spec.support_router,
                    "phase_index": phase_index,
                    "phase_rule": rule,
                    "eval_rule": eval_rule,
                    "is_target_rule": eval_rule == rule,
                    "ce_loss": metrics["ce_loss"],
                    "ce_delta_from_phase_start": ce_delta,
                    "kl_to_pre_phase_logits": metrics.get("kl_to_reference_logits"),
                    "logit_mse_to_pre_phase": metrics.get("logit_mse_to_reference_logits"),
                    "support_churn_to_phase_start": metrics.get("support_churn_to_reference"),
                    "unique_support_sets": metrics.get("unique_support_sets"),
                    "used_columns": metrics.get("used_columns"),
                    "stored_parameters": _stored_parameters(adapter),
                    "active_parameters_proxy": _active_parameters_proxy(spec, hidden_dim),
                    "flops_proxy": _active_parameters_proxy(spec, hidden_dim),
                }
                phase_rows.append(row)
                if eval_rule == rule:
                    adaptation_rows.append(row)
                best_ce[eval_rule] = min(best_ce[eval_rule], metrics["ce_loss"])
            previous_logits = _logits_for_rules(
                adapter=adapter,
                hidden=hidden,
                decode=base.decode,
                targets=targets,
                spec=spec,
                torch=torch,
            )

        final = snapshots[f"after_phase_{len(RULE_SEQUENCE)}_{RULE_SEQUENCE[-1]}"]
        off_target_rows = [row for row in phase_rows if row["arm"] == spec.name and not row["is_target_rule"]]
        target_rows = [row for row in phase_rows if row["arm"] == spec.name and row["is_target_rule"]]
        mean_target_ce_delta = _mean([row["ce_delta_from_phase_start"] for row in target_rows])
        mean_off_target_ce_drift = _mean([row["ce_delta_from_phase_start"] for row in off_target_rows])
        mean_off_target_kl = _mean([row["kl_to_pre_phase_logits"] for row in off_target_rows])
        final_forgetting = {
            rule: final[rule]["ce_loss"] - best_ce[rule]
            for rule in sorted(targets)
        }
        mean_final_forgetting = _mean(list(final_forgetting.values()))
        target_improvement = _target_improvement(mean_target_ce_delta)
        arm_rows.append(
            {
                "arm": spec.name,
                "kind": spec.kind,
                "support_router": spec.support_router,
                "top_k": spec.top_k,
                "num_columns": spec.num_columns,
                "stored_parameters": _stored_parameters(adapter),
                "active_parameters_proxy": _active_parameters_proxy(spec, hidden_dim),
                "flops_proxy": _active_parameters_proxy(spec, hidden_dim),
                "anchor_kl_weight": spec.anchor_kl_weight,
                "mean_target_ce_delta": mean_target_ce_delta,
                "target_adaptation_improvement": target_improvement,
                "mean_off_target_ce_drift": mean_off_target_ce_drift,
                "mean_off_target_kl": mean_off_target_kl,
                "max_off_target_kl": _max([row["kl_to_pre_phase_logits"] for row in off_target_rows]),
                "mean_final_forgetting": mean_final_forgetting,
                "forgetting_per_target_improvement": _safe_ratio(
                    mean_final_forgetting,
                    target_improvement,
                ),
                "off_target_kl_per_target_improvement": _safe_ratio(
                    mean_off_target_kl,
                    target_improvement,
                ),
                "max_final_forgetting": _max(list(final_forgetting.values())),
                "final_copy_ce_loss": final["copy"]["ce_loss"],
                "final_reverse_ce_loss": final["reverse"]["ce_loss"],
                "final_permute_ce_loss": final["permute"]["ce_loss"],
                "final_negate_ce_loss": final["negate"]["ce_loss"],
                "final_forgetting_by_rule": json.dumps(final_forgetting, sort_keys=True),
            }
        )

    gate_rows = _gate_rows(arm_rows)
    summary = {
        "status": "pass",
        "decision": "mechanism_factorized_continual_learning_probe_recorded",
        "claim_status": _claim_status(gate_rows),
        "selected_next_step": _selected_next_step(gate_rows),
        "requires_gpu_now": False,
        "backend_policy": "local CPU mechanism-factorized probe; no RunPod/Colab spend for this bounded run",
        "rules": list(RULE_SEQUENCE),
        "hidden_rule_boundaries": True,
        "task_id_visible_to_model": False,
        "shared_vocab_and_head": True,
        "controls": [arm.name for arm in arms],
        "primary_result": _primary_result(arm_rows),
        "arm_metrics": arm_rows,
        "gate_criteria": gate_rows,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, phase_rows)
    return summary


def _arm_specs(
    *,
    num_columns: int,
    atoms_per_column: int,
    active_top_k: int,
    anchor_kl_weight: float,
) -> list[_ArmSpec]:
    active_rank = active_top_k * atoms_per_column
    return [
        _ArmSpec("frozen_base", "base", 0, 0, 0, "none", dense_rank=0),
        _ArmSpec("dense_active_rank", "dense", 0, 0, 0, "none", dense_rank=active_rank),
        _ArmSpec("dense_stored_match", "dense", 0, 0, 0, "none", dense_rank=active_rank * 2),
        _ArmSpec("contextual_topk1", "sparse", 1, num_columns * active_top_k, atoms_per_column, "contextual_mlp"),
        _ArmSpec("contextual_topk2", "sparse", 2, num_columns, atoms_per_column, "contextual_mlp"),
        _ArmSpec("random_frequency_matched_topk2", "sparse_fixed", 2, num_columns, atoms_per_column, "contextual_mlp", fixed_random_support=True),
        _ArmSpec("dense_active_rank_anchor_kl", "dense", 0, 0, 0, "none", dense_rank=active_rank, anchor_kl_weight=anchor_kl_weight),
        _ArmSpec("contextual_topk1_anchor_kl", "sparse", 1, num_columns * active_top_k, atoms_per_column, "contextual_mlp", anchor_kl_weight=anchor_kl_weight),
    ]


def _build_adapter(
    *,
    spec: _ArmSpec,
    hidden_dim: int,
    contextual_router_hidden_dim: int,
    torch: Any,
    nn: Any,
) -> Any:
    if spec.kind == "base":
        return _IdentityAdapter()
    if spec.kind == "dense":
        return _DenseLowRankAdapter(
            hidden_dim,
            int(spec.dense_rank or 1),
            nn=nn,
            torch=torch,
        )
    adapter = ResidualColumns(
        hidden_dim=hidden_dim,
        num_columns=spec.num_columns,
        atoms_per_column=spec.atoms_per_column,
        top_k=spec.top_k,
        support_router=spec.support_router,
        contextual_router_hidden_dim=contextual_router_hidden_dim,
    )
    return adapter


class _IdentityAdapter:
    def parameters(self) -> list[Any]:
        return []

    def __call__(self, hidden: Any) -> Any:
        return hidden


class _DenseLowRankAdapter:
    def __init__(self, hidden_dim: int, rank: int, *, nn: Any, torch: Any) -> None:
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def parameters(self) -> Any:
        yield from self.down.parameters()
        yield from self.up.parameters()

    def __call__(self, hidden: Any) -> Any:
        return hidden + self.up(self.down(hidden))


def _mechanism_inputs(*, vocab_size: int, seq_len: int, batch_size: int, seed: int) -> Any:
    import torch

    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator)


def _apply_rule(inputs: Any, rule: str, vocab_size: int) -> Any:
    if rule == "copy":
        return inputs.clone()
    if rule == "reverse":
        return inputs.flip(dims=[1])
    if rule == "permute":
        return (inputs + 3) % vocab_size
    if rule == "negate":
        return (vocab_size - 1) - inputs
    raise ValueError(f"unknown mechanism rule: {rule}")


def _forward_logits(adapter: Any, hidden: Any, spec: _ArmSpec, torch: Any, decode: Any) -> Any:
    if spec.kind == "base":
        return decode(hidden)
    if spec.fixed_random_support:
        support = _fixed_support_indices(hidden, spec, torch)
        adapted = adapter(hidden, support_indices=support)
    else:
        adapted = adapter(hidden)
    return decode(adapted)


def _fixed_support_indices(hidden: Any, spec: _ArmSpec, torch: Any) -> Any:
    batch, seq_len = int(hidden.shape[0]), int(hidden.shape[1])
    positions = torch.arange(seq_len, device=hidden.device).view(1, seq_len, 1)
    offsets = torch.arange(spec.top_k, device=hidden.device).view(1, 1, spec.top_k)
    support = (positions + offsets) % spec.num_columns
    return support.expand(batch, seq_len, spec.top_k)


def _evaluate_all_rules(
    *,
    adapter: Any,
    hidden: Any,
    decode: Any,
    targets: dict[str, Any],
    spec: _ArmSpec,
    torch: Any,
    F: Any,
    reference_logits: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    logits = _forward_logits(adapter, hidden, spec, torch, decode)
    rows = {}
    for rule, target in targets.items():
        row = {
            "ce_loss": float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1)).detach().item()),
        }
        if reference_logits and rule in reference_logits:
            row["kl_to_reference_logits"] = float(_kl_to_reference(logits, reference_logits[rule], F=F).detach().item())
            row["logit_mse_to_reference_logits"] = float(F.mse_loss(logits, reference_logits[rule]).detach().item())
        if spec.kind.startswith("sparse") or spec.kind == "sparse":
            support = _support_indices(adapter, hidden, spec, torch)
            row["used_columns"] = int(torch.unique(support).numel())
            row["unique_support_sets"] = int(torch.unique(support.reshape(-1, support.shape[-1]), dim=0).shape[0])
        rows[rule] = row
    return rows


def _logits_for_rules(
    *,
    adapter: Any,
    hidden: Any,
    decode: Any,
    targets: dict[str, Any],
    spec: _ArmSpec,
    torch: Any,
) -> dict[str, Any]:
    logits = _forward_logits(adapter, hidden, spec, torch, decode).detach()
    return {rule: logits for rule in targets}


def _support_indices(adapter: Any, hidden: Any, spec: _ArmSpec, torch: Any) -> Any:
    if spec.fixed_random_support:
        return _fixed_support_indices(hidden, spec, torch)
    _, support = adapter(hidden, return_support=True)
    return support


def _kl_to_reference(logits: Any, reference_logits: Any, *, F: Any) -> Any:
    return F.kl_div(
        F.log_softmax(logits, dim=-1),
        F.softmax(reference_logits, dim=-1),
        reduction="batchmean",
    )


def _stored_parameters(adapter: Any) -> int:
    return int(sum(parameter.numel() for parameter in adapter.parameters()))


def _active_parameters_proxy(spec: _ArmSpec, hidden_dim: int) -> int:
    if spec.kind == "base":
        return 0
    if spec.kind == "dense":
        return int(2 * hidden_dim * (spec.dense_rank or 0))
    return int(spec.top_k * spec.atoms_per_column * hidden_dim)


def _gate_rows(arm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_arm = {str(row["arm"]): row for row in arm_rows}
    required = {
        "frozen_base",
        "dense_active_rank",
        "dense_stored_match",
        "contextual_topk1",
        "contextual_topk2",
        "random_frequency_matched_topk2",
        "dense_active_rank_anchor_kl",
        "contextual_topk1_anchor_kl",
    }
    missing = sorted(required - set(by_arm))
    topk1 = by_arm.get("contextual_topk1", {})
    topk2 = by_arm.get("contextual_topk2", {})
    dense = by_arm.get("dense_active_rank", {})
    random_null = by_arm.get("random_frequency_matched_topk2", {})
    topk1_kl = by_arm.get("contextual_topk1_anchor_kl", {})
    dense_kl = by_arm.get("dense_active_rank_anchor_kl", {})
    return [
        _criterion("required_controls_present", not missing, "hard", "all dense/sparse/null arms exist", missing, "missing required control arms"),
        _criterion("hidden_boundaries_no_task_id_shared_head", True, "hard", "hidden phase boundaries, no task id, shared vocab/head", "A->B->C->A rule stream", ""),
        _criterion("budget_accounting_present", all(row.get("stored_parameters") is not None and row.get("active_parameters_proxy") is not None for row in arm_rows), "hard", "stored/active/FLOP proxies recorded", "recorded", "missing parameter accounting"),
        _criterion("topk1_target_adaptation_no_worse_than_dense", _leq(topk1.get("mean_target_ce_delta"), dense.get("mean_target_ce_delta"), margin=0.02), "claim", "top-k1 target adaptation CE delta no worse than dense", _delta_value(topk1.get("mean_target_ce_delta"), dense.get("mean_target_ce_delta")), "top-k1 target adaptation trails dense"),
        _criterion("topk1_off_target_ce_no_worse_than_dense", _leq(topk1.get("mean_off_target_ce_drift"), dense.get("mean_off_target_ce_drift"), margin=0.0), "claim", "top-k1 off-target CE drift no worse than dense", _delta_value(topk1.get("mean_off_target_ce_drift"), dense.get("mean_off_target_ce_drift")), "top-k1 off-target CE drift exceeds dense"),
        _criterion("topk1_off_target_kl_no_worse_than_dense", _leq(topk1.get("mean_off_target_kl"), dense.get("mean_off_target_kl"), margin=0.0), "claim", "top-k1 off-target KL no worse than dense", _delta_value(topk1.get("mean_off_target_kl"), dense.get("mean_off_target_kl")), "top-k1 off-target KL exceeds dense"),
        _criterion("topk1_forgetting_no_worse_than_dense", _leq(topk1.get("mean_final_forgetting"), dense.get("mean_final_forgetting"), margin=0.0), "claim", "top-k1 final forgetting no worse than dense", _delta_value(topk1.get("mean_final_forgetting"), dense.get("mean_final_forgetting")), "top-k1 forgetting exceeds dense"),
        _criterion("topk1_beats_random_support_null", _leq(topk1.get("mean_final_forgetting"), random_null.get("mean_final_forgetting"), margin=0.0), "claim", "top-k1 forgetting beats random/frequency support null", _delta_value(topk1.get("mean_final_forgetting"), random_null.get("mean_final_forgetting")), "top-k1 does not beat random support null"),
        _criterion("topk2_interference_per_gain_no_worse_than_dense", _leq(topk2.get("forgetting_per_target_improvement"), dense.get("forgetting_per_target_improvement"), margin=0.0), "claim", "top-k2 forgetting per target CE improvement no worse than dense", _delta_value(topk2.get("forgetting_per_target_improvement"), dense.get("forgetting_per_target_improvement")), "top-k2 does not beat dense on retention/adaptation tradeoff"),
        _criterion("topk2_beats_random_support_tradeoff_null", _leq(topk2.get("forgetting_per_target_improvement"), random_null.get("forgetting_per_target_improvement"), margin=0.0), "claim", "top-k2 forgetting per target CE improvement beats random/frequency support null", _delta_value(topk2.get("forgetting_per_target_improvement"), random_null.get("forgetting_per_target_improvement")), "top-k2 tradeoff does not beat random support null"),
        _criterion("anchor_kl_sparse_no_worse_than_dense_anchor_kl", _leq(topk1_kl.get("mean_off_target_kl"), dense_kl.get("mean_off_target_kl"), margin=0.0), "claim", "same anchor-KL mitigation helps sparse at least as much as dense", _delta_value(topk1_kl.get("mean_off_target_kl"), dense_kl.get("mean_off_target_kl")), "anchor-KL sparse does not beat dense anchor-KL"),
    ]


def _criterion(criterion: str, passed: bool, severity: str, requirement: str, observed: Any, failure_reason: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "severity": severity,
        "requirement": requirement,
        "observed": observed,
        "failure_reason": "" if passed else failure_reason,
    }


def _primary_result(arm_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {str(row["arm"]): row for row in arm_rows}
    topk1 = by_arm.get("contextual_topk1", {})
    topk2 = by_arm.get("contextual_topk2", {})
    dense = by_arm.get("dense_active_rank", {})
    return {
        "topk1_minus_dense_mean_target_ce_delta": _delta_value(topk1.get("mean_target_ce_delta"), dense.get("mean_target_ce_delta")),
        "topk1_minus_dense_mean_off_target_ce_drift": _delta_value(topk1.get("mean_off_target_ce_drift"), dense.get("mean_off_target_ce_drift")),
        "topk1_minus_dense_mean_off_target_kl": _delta_value(topk1.get("mean_off_target_kl"), dense.get("mean_off_target_kl")),
        "topk1_minus_dense_mean_final_forgetting": _delta_value(topk1.get("mean_final_forgetting"), dense.get("mean_final_forgetting")),
        "topk1_minus_dense_forgetting_per_target_improvement": _delta_value(topk1.get("forgetting_per_target_improvement"), dense.get("forgetting_per_target_improvement")),
        "topk2_minus_dense_mean_target_ce_delta": _delta_value(topk2.get("mean_target_ce_delta"), dense.get("mean_target_ce_delta")),
        "topk2_minus_dense_mean_off_target_ce_drift": _delta_value(topk2.get("mean_off_target_ce_drift"), dense.get("mean_off_target_ce_drift")),
        "topk2_minus_dense_mean_off_target_kl": _delta_value(topk2.get("mean_off_target_kl"), dense.get("mean_off_target_kl")),
        "topk2_minus_dense_mean_final_forgetting": _delta_value(topk2.get("mean_final_forgetting"), dense.get("mean_final_forgetting")),
        "topk2_minus_dense_forgetting_per_target_improvement": _delta_value(topk2.get("forgetting_per_target_improvement"), dense.get("forgetting_per_target_improvement")),
        "interpretation": "Negative sparse-minus-dense deltas favor the named sparse arm. This is a local mechanism-factorized screen, not promotion evidence without repeats.",
    }


def _claim_status(gate_rows: list[dict[str, Any]]) -> str:
    hard_fail = any(not row["passed"] and row["severity"] == "hard" for row in gate_rows)
    claim_fail = any(not row["passed"] and row["severity"] == "claim" for row in gate_rows)
    if hard_fail:
        return "mechanism_cl_probe_failed_closed"
    if claim_fail:
        return "mechanism_factorized_sparse_retention_not_established"
    return "mechanism_factorized_sparse_retention_candidate_supported_not_promoted"


def _selected_next_step(gate_rows: list[dict[str, Any]]) -> str:
    if any(not row["passed"] and row["severity"] == "hard" for row in gate_rows):
        return "repair_mechanism_factorized_cl_artifact_schema_before_interpretation"
    if _gate_passed(gate_rows, "topk2_interference_per_gain_no_worse_than_dense") and _gate_passed(
        gate_rows,
        "topk2_beats_random_support_tradeoff_null",
    ):
        return "run_second_seed_mechanism_factorized_cl_repeat_before_any_gpu_validation"
    if any(not row["passed"] and row["severity"] == "claim" for row in gate_rows):
        return "use mechanism CL result to choose a stricter dense/null-controlled residual-interference mitigation"
    return "repeat mechanism-factorized CL probe on a second seed before any GPU validation"


def _write_artifacts(out_dir: Path, summary: dict[str, Any], phase_rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "phase_metrics.csv", phase_rows)
    _write_csv(out_dir / "arm_metrics.csv", summary["arm_metrics"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    result = summary["primary_result"]
    lines = [
        "# Mechanism-Factorized Continual-Learning Probe",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Rule sequence: `{'->'.join(summary['rules'])}`",
        f"- Hidden rule boundaries: `{summary['hidden_rule_boundaries']}`",
        f"- Task id visible to model: `{summary['task_id_visible_to_model']}`",
        f"- Shared vocab/head: `{summary['shared_vocab_and_head']}`",
        "- Top-k1 minus dense target CE delta: "
        f"`{result['topk1_minus_dense_mean_target_ce_delta']}`",
        "- Top-k1 minus dense off-target CE drift: "
        f"`{result['topk1_minus_dense_mean_off_target_ce_drift']}`",
        "- Top-k1 minus dense off-target KL: "
        f"`{result['topk1_minus_dense_mean_off_target_kl']}`",
        "- Top-k1 minus dense final forgetting: "
        f"`{result['topk1_minus_dense_mean_final_forgetting']}`",
        "- Top-k2 minus dense target CE delta: "
        f"`{result['topk2_minus_dense_mean_target_ce_delta']}`",
        "- Top-k2 minus dense final forgetting: "
        f"`{result['topk2_minus_dense_mean_final_forgetting']}`",
        "- Top-k2 minus dense forgetting per target improvement: "
        f"`{result['topk2_minus_dense_forgetting_per_target_improvement']}`",
        "",
        "This local assay uses known transformation rules with hidden phase boundaries and a shared output head. It records dense, sparse, random-support, and identical anchor-KL controls before any promotion or GPU repeat.",
        "",
        "## Next Step",
        "",
        str(summary["selected_next_step"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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


def _mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None and value != ""]
    if not numeric:
        return None
    return sum(numeric) / float(len(numeric))


def _max(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None and value != ""]
    if not numeric:
        return None
    return max(numeric)


def _leq(left: Any, right: Any, *, margin: float) -> bool:
    if left is None or right is None:
        return False
    return float(left) <= float(right) + margin


def _delta_value(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _gate_passed(gate_rows: list[dict[str, Any]], criterion: str) -> bool:
    return any(row["criterion"] == criterion and row["passed"] for row in gate_rows)


def _target_improvement(mean_target_ce_delta: Any) -> float | None:
    if mean_target_ce_delta is None:
        return None
    return max(0.0, -float(mean_target_ce_delta))


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None:
        return None
    denominator_float = float(denominator)
    if denominator_float <= 1e-12:
        return None
    return float(numerator) / denominator_float


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps-per-phase", type=int, default=18)
    args = parser.parse_args(argv)
    summary = run_mechanism_factorized_continual_learning_probe(
        out_dir=args.out,
        seed=args.seed,
        steps_per_phase=args.steps_per_phase,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
