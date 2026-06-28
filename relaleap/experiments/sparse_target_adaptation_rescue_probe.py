"""Sparse target-adaptation rescue screen for mechanism-factorized CL."""

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

from relaleap.experiments.mechanism_factorized_continual_learning_probe import (
    RULE_SEQUENCE,
    _active_parameters_proxy,
    _apply_rule,
    _build_adapter,
    _forward_logits,
    _git_commit,
    _kl_to_reference,
    _logits_for_rules,
    _mean,
    _mechanism_inputs,
    _stored_parameters,
)
from relaleap.experiments.mechanism_factorized_continual_learning_probe import (
    _ArmSpec as _MechanismArmSpec,
)
from relaleap.smoke import TinyCharTransformer


DEFAULT_OUT_DIR = Path("results/reports/sparse_target_adaptation_rescue_probe")
REQUIRED_ARTIFACTS = (
    "summary.json",
    "rescue_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)


@dataclass(frozen=True)
class _RescueArm:
    name: str
    mechanism_spec: _MechanismArmSpec
    learning_rate_multiplier: float = 1.0
    value_learning_rate_multiplier: float = 1.0
    target_loss: str = "ce"
    focal_gamma: float = 0.0


def run_sparse_target_adaptation_rescue_probe(
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
    """Run one CPU rescue screen with dense, sparse, and random-support controls."""

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on torch install
        raise RuntimeError("sparse target-adaptation rescue probe requires torch") from exc

    if steps_per_phase < 1:
        raise ValueError("steps_per_phase must be positive")

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
    arms = _rescue_arms(
        num_columns=num_columns,
        atoms_per_column=atoms_per_column,
        anchor_kl_weight=anchor_kl_weight,
    )

    rescue_rows = [
        _run_arm(
            arm=arm,
            arm_index=arm_index,
            seed=seed,
            hidden=hidden,
            base=base,
            targets=targets,
            steps_per_phase=steps_per_phase,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            torch=torch,
            F=F,
        )
        for arm_index, arm in enumerate(arms)
    ]
    gate_rows = _gate_rows(rescue_rows)
    status = "pass" if all(row["passed"] for row in gate_rows if row["severity"] == "hard") else "fail"
    claim_status = _claim_status(gate_rows)
    summary = {
        "status": status,
        "decision": (
            "sparse_target_adaptation_rescue_probe_recorded"
            if status == "pass"
            else "sparse_target_adaptation_rescue_probe_failed_closed"
        ),
        "claim_status": claim_status,
        "selected_next_step": _selected_next_step(claim_status),
        "requires_gpu_now": False,
        "backend_policy": "local CPU rescue screen; no RunPod/Colab spend until a local rescue closes the dense gap",
        "rules": list(RULE_SEQUENCE),
        "hidden_rule_boundaries": True,
        "task_id_visible_to_model": False,
        "shared_vocab_and_head": True,
        "rescue_under_test": "contextual_topk2_sparse_value_lr_or_focal_target_objective_with_anchor_kl",
        "rescue_metrics": rescue_rows,
        "gate_criteria": gate_rows,
        "primary_result": _primary_result(rescue_rows),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _rescue_arms(
    *,
    num_columns: int,
    atoms_per_column: int,
    anchor_kl_weight: float,
) -> list[_RescueArm]:
    active_top_k = 2
    active_rank = active_top_k * atoms_per_column
    return [
        _RescueArm(
            "dense_active_rank",
            _MechanismArmSpec("dense_active_rank", "dense", 0, 0, 0, "none", dense_rank=active_rank),
        ),
        _RescueArm(
            "contextual_topk2_baseline",
            _MechanismArmSpec("contextual_topk2", "sparse", 2, num_columns, atoms_per_column, "contextual_mlp"),
        ),
        _RescueArm(
            "contextual_topk2_value_lr2_anchor_kl",
            _MechanismArmSpec(
                "contextual_topk2_value_lr2_anchor_kl",
                "sparse",
                2,
                num_columns,
                atoms_per_column,
                "contextual_mlp",
                anchor_kl_weight=anchor_kl_weight,
            ),
            value_learning_rate_multiplier=2.0,
        ),
        _RescueArm(
            "contextual_topk2_value_lr4_anchor_kl",
            _MechanismArmSpec(
                "contextual_topk2_value_lr4_anchor_kl",
                "sparse",
                2,
                num_columns,
                atoms_per_column,
                "contextual_mlp",
                anchor_kl_weight=anchor_kl_weight,
            ),
            value_learning_rate_multiplier=4.0,
        ),
        _RescueArm(
            "contextual_topk2_focal_gamma2_anchor_kl",
            _MechanismArmSpec(
                "contextual_topk2_focal_gamma2_anchor_kl",
                "sparse",
                2,
                num_columns,
                atoms_per_column,
                "contextual_mlp",
                anchor_kl_weight=anchor_kl_weight,
            ),
            target_loss="focal_ce",
            focal_gamma=2.0,
        ),
        _RescueArm(
            "contextual_topk2_focal_gamma2_value_lr2_anchor_kl",
            _MechanismArmSpec(
                "contextual_topk2_focal_gamma2_value_lr2_anchor_kl",
                "sparse",
                2,
                num_columns,
                atoms_per_column,
                "contextual_mlp",
                anchor_kl_weight=anchor_kl_weight,
            ),
            value_learning_rate_multiplier=2.0,
            target_loss="focal_ce",
            focal_gamma=2.0,
        ),
        _RescueArm(
            "random_frequency_matched_topk2",
            _MechanismArmSpec(
                "random_frequency_matched_topk2",
                "sparse_fixed",
                2,
                num_columns,
                atoms_per_column,
                "contextual_mlp",
                fixed_random_support=True,
            ),
        ),
    ]


def _run_arm(
    *,
    arm: _RescueArm,
    arm_index: int,
    seed: int,
    hidden: Any,
    base: Any,
    targets: dict[str, Any],
    steps_per_phase: int,
    learning_rate: float,
    hidden_dim: int,
    vocab_size: int,
    torch: Any,
    F: Any,
) -> dict[str, Any]:
    torch.manual_seed(seed + 2000 + arm_index)
    spec = arm.mechanism_spec
    adapter = _build_adapter(
        spec=spec,
        hidden_dim=hidden_dim,
        contextual_router_hidden_dim=hidden_dim * 2,
        torch=torch,
        nn=torch.nn,
    )
    optimizer = _optimizer(
        adapter=adapter,
        torch=torch,
        learning_rate=learning_rate,
        arm=arm,
    )
    previous_logits = _logits_for_rules(
        adapter=adapter,
        hidden=hidden,
        decode=base.decode,
        targets=targets,
        spec=spec,
        torch=torch,
    )
    best_ce = _evaluate_ce(adapter, hidden, base.decode, targets, spec, torch, F)
    target_deltas: list[float] = []
    off_target_deltas: list[float] = []
    off_target_kls: list[float] = []
    for rule in RULE_SEQUENCE:
        before = _evaluate_ce(adapter, hidden, base.decode, targets, spec, torch, F)
        anchor_logits = {
            off_rule: previous_logits[off_rule].detach()
            for off_rule in targets
            if off_rule != rule
        }
        for _ in range(steps_per_phase):
            optimizer.zero_grad(set_to_none=True)
            logits = _forward_logits(adapter, hidden, spec, torch, base.decode)
            loss = _target_loss(logits, targets[rule], vocab_size, arm=arm, F=F)
            if spec.anchor_kl_weight > 0.0 and anchor_logits:
                anchor_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)
                for reference_logits in anchor_logits.values():
                    anchor_loss = anchor_loss + _kl_to_reference(logits, reference_logits, F=F)
                loss = loss + spec.anchor_kl_weight * anchor_loss / float(len(anchor_logits))
            loss.backward()
            optimizer.step()
        after = _evaluate_ce(adapter, hidden, base.decode, targets, spec, torch, F)
        logits_after = _forward_logits(adapter, hidden, spec, torch, base.decode).detach()
        for eval_rule, ce_loss in after.items():
            delta = ce_loss - before[eval_rule]
            best_ce[eval_rule] = min(best_ce[eval_rule], ce_loss)
            if eval_rule == rule:
                target_deltas.append(delta)
            else:
                off_target_deltas.append(delta)
                off_target_kls.append(
                    float(_kl_to_reference(logits_after, previous_logits[eval_rule], F=F).detach().item())
                )
        previous_logits = _logits_for_rules(
            adapter=adapter,
            hidden=hidden,
            decode=base.decode,
            targets=targets,
            spec=spec,
            torch=torch,
        )
    final_ce = _evaluate_ce(adapter, hidden, base.decode, targets, spec, torch, F)
    forgetting = {rule: final_ce[rule] - best_ce[rule] for rule in sorted(targets)}
    return {
        "arm": arm.name,
        "kind": spec.kind,
        "top_k": spec.top_k,
        "support_router": spec.support_router,
        "learning_rate_multiplier": arm.learning_rate_multiplier,
        "value_learning_rate_multiplier": arm.value_learning_rate_multiplier,
        "target_loss": arm.target_loss,
        "focal_gamma": arm.focal_gamma,
        "anchor_kl_weight": spec.anchor_kl_weight,
        "stored_parameters": _stored_parameters(adapter),
        "active_parameters_proxy": _active_parameters_proxy(spec, hidden_dim),
        "mean_target_ce_delta": _mean(target_deltas),
        "mean_off_target_ce_drift": _mean(off_target_deltas),
        "mean_off_target_kl": _mean(off_target_kls),
        "mean_final_forgetting": _mean(list(forgetting.values())),
        "final_forgetting_by_rule": json.dumps(forgetting, sort_keys=True),
    }


def _optimizer(*, adapter: Any, torch: Any, learning_rate: float, arm: _RescueArm) -> Any:
    if hasattr(adapter, "named_parameters"):
        value_params = []
        other_params = []
        for name, parameter in adapter.named_parameters():
            if not parameter.requires_grad:
                continue
            if name == "atom_values":
                value_params.append(parameter)
            else:
                other_params.append(parameter)
        groups = []
        if other_params:
            groups.append({"params": other_params, "lr": learning_rate * arm.learning_rate_multiplier})
        if value_params:
            groups.append({"params": value_params, "lr": learning_rate * arm.value_learning_rate_multiplier})
        return torch.optim.AdamW(groups, lr=learning_rate)
    return torch.optim.AdamW(list(adapter.parameters()), lr=learning_rate * arm.learning_rate_multiplier)


def _target_loss(logits: Any, target: Any, vocab_size: int, *, arm: _RescueArm, F: Any) -> Any:
    flat_logits = logits.reshape(-1, vocab_size)
    flat_target = target.reshape(-1)
    if arm.target_loss == "ce":
        return F.cross_entropy(flat_logits, flat_target)
    if arm.target_loss == "focal_ce":
        token_ce = F.cross_entropy(flat_logits, flat_target, reduction="none")
        token_prob = (-token_ce).exp()
        return (((1.0 - token_prob) ** arm.focal_gamma) * token_ce).mean()
    raise ValueError(f"unknown sparse target rescue loss: {arm.target_loss}")


def _evaluate_ce(
    adapter: Any,
    hidden: Any,
    decode: Any,
    targets: dict[str, Any],
    spec: _MechanismArmSpec,
    torch: Any,
    F: Any,
) -> dict[str, float]:
    logits = _forward_logits(adapter, hidden, spec, torch, decode)
    return {
        rule: float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1)).detach().item())
        for rule, target in targets.items()
    }


def _gate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_arm = {str(row["arm"]): row for row in rows}
    required = {
        "dense_active_rank",
        "contextual_topk2_baseline",
        "contextual_topk2_value_lr2_anchor_kl",
        "contextual_topk2_value_lr4_anchor_kl",
        "contextual_topk2_focal_gamma2_anchor_kl",
        "contextual_topk2_focal_gamma2_value_lr2_anchor_kl",
        "random_frequency_matched_topk2",
    }
    missing = sorted(required - set(by_arm))
    best = _best_rescue(rows)
    dense = by_arm.get("dense_active_rank", {})
    baseline = by_arm.get("contextual_topk2_baseline", {})
    random_null = by_arm.get("random_frequency_matched_topk2", {})
    return [
        _criterion("required_dense_sparse_null_arms_present", not missing, "hard", "dense, baseline sparse, rescue, and random-support controls exist", missing, "missing required control arms"),
        _criterion("budget_accounting_present", all(row.get("stored_parameters") is not None and row.get("active_parameters_proxy") is not None for row in rows), "hard", "stored/active parameter proxies recorded", "recorded", "missing budget accounting"),
        _criterion("best_rescue_improves_target_adaptation_vs_topk2_baseline", _lt_delta(best, baseline, "mean_target_ce_delta"), "claim", "best rescue improves target CE adaptation versus top-k2 baseline", _delta(best, baseline, "mean_target_ce_delta"), "rescue did not improve target adaptation over top-k2 baseline"),
        _criterion("best_rescue_target_adaptation_dense_matched", _leq_delta(best, dense, "mean_target_ce_delta", margin=0.02), "claim", "best rescue closes the dense target-adaptation gap within 0.02 CE delta", _delta(best, dense, "mean_target_ce_delta"), "rescue still trails dense target adaptation"),
        _criterion("best_rescue_preserves_dense_off_target_ce_advantage", _leq_delta(best, dense, "mean_off_target_ce_drift", margin=0.0), "claim", "best rescue off-target CE drift no worse than dense", _delta(best, dense, "mean_off_target_ce_drift"), "rescue loses off-target CE advantage versus dense"),
        _criterion("best_rescue_preserves_dense_off_target_kl_advantage", _leq_delta(best, dense, "mean_off_target_kl", margin=0.0), "claim", "best rescue off-target KL no worse than dense", _delta(best, dense, "mean_off_target_kl"), "rescue loses off-target KL advantage versus dense"),
        _criterion("best_rescue_not_worse_than_topk2_off_target_kl_by_margin", _leq_delta(best, baseline, "mean_off_target_kl", margin=0.02), "claim", "best rescue preserves top-k2 off-target KL within 0.02", _delta(best, baseline, "mean_off_target_kl"), "rescue materially increases off-target KL versus top-k2 baseline"),
        _criterion("best_rescue_beats_random_support_null_on_target", _lt_delta(best, random_null, "mean_target_ce_delta"), "claim", "best rescue beats random support null on target adaptation", _delta(best, random_null, "mean_target_ce_delta"), "rescue does not beat random support null on target adaptation"),
    ]


def _best_rescue(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if str(row.get("arm")).startswith("contextual_topk2_")
        and str(row.get("arm")) != "contextual_topk2_baseline"
    ]
    if not candidates:
        return {}
    return min(candidates, key=lambda row: float(row.get("mean_target_ce_delta") or 0.0))


def _criterion(criterion: str, passed: bool, severity: str, requirement: str, observed: Any, failure_reason: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "severity": severity,
        "requirement": requirement,
        "observed": observed,
        "failure_reason": "" if passed else failure_reason,
    }


def _claim_status(gate_rows: list[dict[str, Any]]) -> str:
    if any(not row["passed"] and row["severity"] == "hard" for row in gate_rows):
        return "sparse_target_adaptation_rescue_failed_closed"
    if any(row["criterion"] == "best_rescue_target_adaptation_dense_matched" and row["passed"] for row in gate_rows) and all(
        row["passed"] for row in gate_rows if row["severity"] == "claim"
    ):
        return "sparse_target_adaptation_rescue_candidate_not_promoted"
    return "sparse_target_adaptation_rescue_not_established"


def _selected_next_step(claim_status: str) -> str:
    if claim_status == "sparse_target_adaptation_rescue_failed_closed":
        return "repair sparse target-adaptation rescue artifact schema before interpretation"
    if claim_status == "sparse_target_adaptation_rescue_candidate_not_promoted":
        return "repeat sparse target-adaptation rescue on a second local seed before any RunPod validation"
    return "retire the current top-k2 rescue path or replace it with a mechanistically different sparse-retention objective"


def _primary_result(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {str(row["arm"]): row for row in rows}
    best = _best_rescue(rows)
    dense = by_arm.get("dense_active_rank", {})
    baseline = by_arm.get("contextual_topk2_baseline", {})
    return {
        "best_rescue_arm": best.get("arm"),
        "best_rescue_minus_topk2_target_ce_delta": _delta(best, baseline, "mean_target_ce_delta"),
        "best_rescue_minus_dense_target_ce_delta": _delta(best, dense, "mean_target_ce_delta"),
        "best_rescue_minus_dense_off_target_ce_drift": _delta(best, dense, "mean_off_target_ce_drift"),
        "best_rescue_minus_dense_off_target_kl": _delta(best, dense, "mean_off_target_kl"),
        "best_rescue_minus_topk2_off_target_kl": _delta(best, baseline, "mean_off_target_kl"),
        "interpretation": "Negative target deltas favor the rescue. Promotion remains blocked unless dense target adaptation is matched while preserving sparse off-target advantages.",
    }


def _delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    if left.get(key) is None or right.get(key) is None:
        return None
    return float(left[key]) - float(right[key])


def _lt_delta(left: dict[str, Any], right: dict[str, Any], key: str) -> bool:
    value = _delta(left, right, key)
    return value is not None and value < 0.0


def _leq_delta(left: dict[str, Any], right: dict[str, Any], key: str, *, margin: float) -> bool:
    value = _delta(left, right, key)
    return value is not None and value <= margin


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "rescue_metrics.csv", summary["rescue_metrics"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_notes(out_dir / "notes.md", summary)


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


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    result = summary["primary_result"]
    lines = [
        "# Sparse Target-Adaptation Rescue Probe",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Rescue under test: `{summary['rescue_under_test']}`",
        f"- Best rescue arm: `{result.get('best_rescue_arm')}`",
        "- Best rescue minus top-k2 target CE delta: "
        f"`{result.get('best_rescue_minus_topk2_target_ce_delta')}`",
        "- Best rescue minus dense target CE delta: "
        f"`{result.get('best_rescue_minus_dense_target_ce_delta')}`",
        "- Best rescue minus dense off-target KL: "
        f"`{result.get('best_rescue_minus_dense_off_target_kl')}`",
        "",
        "This local screen tests whether boosting sparse top-k2 value updates or using a focal target objective under anchor KL can close the dense target-adaptation gap without surrendering the sparse off-target advantage.",
        "",
        "## Next Step",
        "",
        str(summary["selected_next_step"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps-per-phase", type=int, default=18)
    args = parser.parse_args(argv)
    summary = run_sparse_target_adaptation_rescue_probe(
        out_dir=args.out,
        seed=args.seed,
        steps_per_phase=args.steps_per_phase,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
