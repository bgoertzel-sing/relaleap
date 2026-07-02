"""Local dense-teacher oracle sparse-coding feasibility assay.

This evaluator consumes the post support-forcing strategy pivot and asks the
pre-GPU question directly: is the dense teacher residual field columnable by an
oracle top-k sparse code over an orthogonal basis before spending more effort
on deployable routers or sparse value redesigns?
"""

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

from relaleap.experiments.dense_teacher_residual_value_capacity_norm_assay import (
    _Teacher,
    _arm_metrics,
    _make_data,
    _norm_match,
    _source_row,
    _train_flat_value_head,
)


DEFAULT_SUPPORT_FORCING = Path("results/reports/dense_teacher_support_forcing_pruning_pregate/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_oracle_sparse_coding_feasibility")

DECISION = "dense_teacher_oracle_sparse_coding_feasibility_recorded"
FAIL_DECISION = "dense_teacher_oracle_sparse_coding_feasibility_failed_closed"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "spectrum_rows.csv",
    "arm_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)

ARMS = (
    "dense_teacher_residual_control",
    "oracle_topk_orthogonal_sparse_coding",
    "learned_router_topk_scalar_sparse_coding",
    "same_router_flat_value_control",
    "low_rank_dense_control",
    "random_topk_sparse_coding_null",
    "load_permuted_topk_sparse_coding_null",
    "token_position_topk_sparse_coding_null",
    "shuffled_target_oracle_topk_sparse_coding_null",
    "no_update_control",
    "current_sparse_support_forcing_reference",
)


def run_dense_teacher_oracle_sparse_coding_feasibility(
    *,
    support_forcing_path: Path = DEFAULT_SUPPORT_FORCING,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 31,
    teacher_steps: int = 100,
    router_steps: int = 80,
    control_steps: int = 80,
    basis_size: int = 8,
    top_k: int = 2,
    data_column_count: int = 6,
) -> dict[str, Any]:
    """Run the local CPU feasibility assay and write artifacts."""

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError("dense-teacher oracle sparse-coding feasibility assay requires torch") from exc

    if min(teacher_steps, router_steps, control_steps) < 1:
        raise ValueError("all training step counts must be positive")
    if basis_size < 2:
        raise ValueError("basis_size must be at least 2")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if data_column_count < 2:
        raise ValueError("data_column_count must be at least 2")

    start = time.time()
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    support_forcing = _read_json(support_forcing_path)
    review_text = strategy_review_path.read_text(encoding="utf-8") if strategy_review_path.is_file() else ""
    source_rows = [
        _source_row("dense_teacher_support_forcing_pruning_pregate", support_forcing_path, support_forcing),
        {
            "source": "gpt_5_5_pro_strategy_review",
            "path": str(strategy_review_path),
            "present": strategy_review_path.is_file(),
            "status": "present" if strategy_review_path.is_file() else "missing",
            "decision": _review_field(review_text, "verdict"),
            "claim_status": _review_field(review_text, "strategic_change_level"),
            "selected_next_step": _review_field(review_text, "recommended_next_action"),
        },
    ]

    data = _make_data(torch, seed=seed, column_count=data_column_count)
    effective_basis_size = min(basis_size, data["classes"])
    if top_k > effective_basis_size:
        raise ValueError("top_k must not exceed the effective residual basis size")
    teacher = _Teacher(torch, data["input_dim"], data["classes"])
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=0.01)
    for _ in range(teacher_steps):
        logits = data["base_logits_train"] + teacher(data["x_train"])
        loss = F.cross_entropy(logits, data["y_train"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        teacher_train = teacher(data["x_train"])
        teacher_holdout = teacher(data["x_holdout"])
        base_holdout_ce = float(F.cross_entropy(data["base_logits_holdout"], data["y_holdout"]).item())
        teacher_holdout_ce = float(F.cross_entropy(data["base_logits_holdout"] + teacher_holdout, data["y_holdout"]).item())

    mean, basis, train_coeff, holdout_coeff, spectrum_rows = _orthogonal_basis(torch, teacher_train, teacher_holdout, effective_basis_size)
    oracle_mask = _topk_mask(torch, holdout_coeff, top_k)
    train_oracle_mask = _topk_mask(torch, train_coeff, top_k)

    support_router = _train_mask_router(torch, data["x_train"], train_oracle_mask, data["input_dim"], effective_basis_size, steps=router_steps)
    coeff_head = _train_coeff_head(torch, F, data["x_train"], train_coeff, data["input_dim"], effective_basis_size, steps=control_steps)
    flat_value = _train_flat_value_head(torch, F, data["x_train"], teacher_train, data["input_dim"], data["classes"], steps=control_steps)
    low_rank = _train_low_rank_head(torch, F, data["x_train"], teacher_train, data["input_dim"], data["classes"], rank=top_k, steps=control_steps)

    learned_mask = _topk_mask(torch, support_router(data["x_holdout"]), top_k)
    learned_coeff = coeff_head(data["x_holdout"])
    random_mask = _deterministic_random_mask(torch, len(data["x_holdout"]), effective_basis_size, top_k)
    load_permuted_mask = torch.roll(oracle_mask, shifts=1, dims=1)
    token_position_mask = _token_position_mask(torch, data["position_holdout"], effective_basis_size, top_k)
    shuffled_mean, shuffled_basis, _shuffled_train_coeff, shuffled_holdout_coeff, _ = _orthogonal_basis(
        torch,
        torch.roll(teacher_train, shifts=1, dims=0),
        torch.roll(teacher_holdout, shifts=1, dims=0),
        effective_basis_size,
    )

    zero_support = torch.zeros(len(data["x_holdout"]), dtype=torch.long)
    arms: dict[str, tuple[Any, Any, bool, str]] = {
        "dense_teacher_residual_control": (
            teacher_holdout,
            _mask_to_support(torch, oracle_mask),
            False,
            "dense teacher residual target; control only",
        ),
        "oracle_topk_orthogonal_sparse_coding": (
            _decode_sparse(mean, basis, holdout_coeff, oracle_mask),
            _mask_to_support(torch, oracle_mask),
            True,
            "nondeployable oracle top-k coefficients over the teacher residual PCA basis",
        ),
        "learned_router_topk_scalar_sparse_coding": (
            _decode_sparse(mean, basis, learned_coeff, learned_mask),
            _mask_to_support(torch, learned_mask),
            False,
            "deployable linear top-k mask router plus scalar coefficient head",
        ),
        "same_router_flat_value_control": (
            _norm_match(torch, flat_value(data["x_holdout"]), teacher_train),
            _mask_to_support(torch, learned_mask),
            False,
            "same learned top-k support summarized as support id with flat value head",
        ),
        "low_rank_dense_control": (
            _norm_match(torch, low_rank(data["x_holdout"]), teacher_train),
            zero_support,
            False,
            "rank-matched dense residual control",
        ),
        "random_topk_sparse_coding_null": (
            _decode_sparse(mean, basis, holdout_coeff, random_mask),
            _mask_to_support(torch, random_mask),
            False,
            "random top-k basis mask null with oracle coefficients",
        ),
        "load_permuted_topk_sparse_coding_null": (
            _decode_sparse(mean, basis, holdout_coeff, load_permuted_mask),
            _mask_to_support(torch, load_permuted_mask),
            False,
            "load-preserving cyclic permutation of oracle top-k support",
        ),
        "token_position_topk_sparse_coding_null": (
            _decode_sparse(mean, basis, holdout_coeff, token_position_mask),
            _mask_to_support(torch, token_position_mask),
            False,
            "token/position-only deterministic sparse support null",
        ),
        "shuffled_target_oracle_topk_sparse_coding_null": (
            _decode_sparse(shuffled_mean, shuffled_basis, shuffled_holdout_coeff, _topk_mask(torch, shuffled_holdout_coeff, top_k)),
            _mask_to_support(torch, oracle_mask),
            False,
            "oracle sparse coding on a shuffled residual target null",
        ),
        "no_update_control": (
            torch.zeros_like(teacher_holdout),
            zero_support,
            False,
            "zero residual update control",
        ),
    }

    arm_rows = [
        _arm_metrics(torch, F, arm, pred, support, teacher_holdout, data, teacher_holdout_ce, base_holdout_ce, oracle, note, effective_basis_size, top_k)
        for arm, (pred, support, oracle, note) in arms.items()
    ]
    arm_rows.append(_current_sparse_reference_row(support_forcing))

    gate_rows = _gate_rows(source_rows, spectrum_rows, arm_rows, base_holdout_ce, teacher_holdout_ce)
    runtime_failures = [row for row in gate_rows if row["required"] and not row["passed"]]
    scientific_failures = [row for row in gate_rows if row["gate_type"] == "scientific" and not row["passed"]]
    status = "fail" if runtime_failures else "pass"
    failed_names = {row["criterion"] for row in scientific_failures}
    if status == "pass" and not scientific_failures:
        claim_status = "oracle_sparse_coding_feasible_ready_for_router_imitation_no_gpu"
    elif status == "pass" and failed_names == {"learned_support_retains_oracle_gain"}:
        claim_status = "oracle_sparse_coding_feasible_router_imitation_blocks_gpu"
    else:
        claim_status = "oracle_sparse_coding_feasibility_blocks_sparse_redesign_gpu"
    summary = {
        "status": status,
        "decision": DECISION if status == "pass" else FAIL_DECISION,
        "claim_status": claim_status,
        "selected_next_step": _selected_next_step(status, scientific_failures),
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "backend_policy": "local CPU oracle sparse-coding feasibility only; RunPod and Colab remain blocked",
        "training_executed": True,
        "teacher_trained": True,
        "seed": seed,
        "teacher_train_steps": teacher_steps,
        "router_train_steps": router_steps,
        "control_train_steps": control_steps,
        "requested_basis_size": basis_size,
        "basis_size": effective_basis_size,
        "top_k": top_k,
        "data_column_count": data_column_count,
        "base_holdout_ce": round(base_holdout_ce, 6),
        "dense_teacher_holdout_ce": round(teacher_holdout_ce, 6),
        "dense_teacher_ce_improvement": round(base_holdout_ce - teacher_holdout_ce, 6),
        "source_rows": source_rows,
        "spectrum_rows": spectrum_rows,
        "arm_metrics": arm_rows,
        "gate_criteria": gate_rows,
        "failures": runtime_failures + scientific_failures,
        "strategy_review_handling": (
            "Accepted the major GPT-5.5-Pro pivot: current sparse value/support redesigns stay closed, "
            "Ben should be notified, and this local oracle sparse-coding feasibility gate must beat flat/null "
            "controls before any new sparse redesign or GPU validation."
        ),
        "ben_notification_recommended": _review_field(review_text, "notify_ben").lower() == "true",
        "strategic_change_level": _review_field(review_text, "strategic_change_level"),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _orthogonal_basis(torch: Any, train: Any, holdout: Any, basis_size: int) -> tuple[Any, Any, Any, Any, list[dict[str, Any]]]:
    mean = train.mean(dim=0, keepdim=True)
    centered = train - mean
    _u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:basis_size]
    train_coeff = centered @ basis.T
    holdout_coeff = (holdout - mean) @ basis.T
    total_energy = float((singular_values**2).sum().item())
    cumulative = 0.0
    rows = []
    effective_rank_probs = (singular_values**2) / (singular_values**2).sum().clamp_min(1e-8)
    effective_rank = float(torch.exp(-(effective_rank_probs * torch.log(effective_rank_probs.clamp_min(1e-8))).sum()).item())
    for index in range(min(len(singular_values), basis_size)):
        energy = float((singular_values[index] ** 2).item())
        cumulative += energy
        rows.append(
            {
                "component": index,
                "singular_value": round(float(singular_values[index].item()), 6),
                "energy_fraction": round(energy / max(total_energy, 1e-8), 6),
                "cumulative_energy_fraction": round(cumulative / max(total_energy, 1e-8), 6),
                "effective_rank": round(effective_rank, 6),
            }
        )
    return mean, basis, train_coeff, holdout_coeff, rows


def _topk_mask(torch: Any, coeff: Any, top_k: int) -> Any:
    indices = torch.topk(coeff.abs(), k=top_k, dim=1).indices
    mask = torch.zeros_like(coeff)
    mask.scatter_(1, indices, 1.0)
    return mask


def _decode_sparse(mean: Any, basis: Any, coeff: Any, mask: Any) -> Any:
    return mean + (coeff * mask) @ basis


def _train_mask_router(torch: Any, x_train: Any, mask: Any, input_dim: int, basis_size: int, *, steps: int) -> Any:
    router = torch.nn.Linear(input_dim, basis_size)
    optimizer = torch.optim.AdamW(router.parameters(), lr=0.035)
    for _ in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(router(x_train), mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return router


def _train_coeff_head(torch: Any, F: Any, x_train: Any, coeff: Any, input_dim: int, basis_size: int, *, steps: int) -> Any:
    head = torch.nn.Linear(input_dim, basis_size)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.02)
    for _ in range(steps):
        loss = F.mse_loss(head(x_train), coeff)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return head


def _train_low_rank_head(torch: Any, F: Any, x_train: Any, targets: Any, input_dim: int, classes: int, *, rank: int, steps: int) -> Any:
    model = torch.nn.Sequential(torch.nn.Linear(input_dim, rank, bias=False), torch.nn.Linear(rank, classes, bias=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(steps):
        loss = F.mse_loss(model(x_train), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def _deterministic_random_mask(torch: Any, n: int, basis_size: int, top_k: int) -> Any:
    mask = torch.zeros(n, basis_size)
    for row in range(n):
        for offset in range(top_k):
            mask[row, (row * 3 + offset * 5 + 1) % basis_size] = 1.0
    return mask


def _token_position_mask(torch: Any, position: Any, basis_size: int, top_k: int) -> Any:
    mask = torch.zeros(len(position), basis_size)
    for row, pos in enumerate(position.tolist()):
        for offset in range(top_k):
            mask[row, (int(pos) + offset) % basis_size] = 1.0
    return mask


def _mask_to_support(torch: Any, mask: Any) -> Any:
    weights = torch.arange(mask.shape[1], dtype=torch.float32, device=mask.device) + 1.0
    return ((mask * weights).sum(dim=1).long() % mask.shape[1]).cpu()


def _current_sparse_reference_row(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary.get("support_forcing_rows", [])
    learned = next((row for row in rows if row.get("arm") == "learned_support_same_values"), {})
    return {
        "arm": "current_sparse_support_forcing_reference",
        "row_source": "consumed_support_forcing_pruning_artifact",
        "teacher_trained": True,
        "ce": _round_or_none(learned.get("ce")),
        "base_ce": _round_or_none(summary.get("base_holdout_ce")),
        "dense_teacher_ce": _round_or_none(summary.get("dense_teacher_holdout_ce")),
        "ce_gap_vs_dense_teacher": _round_or_none(learned.get("ce_gap_vs_dense_teacher")),
        "ce_improvement_vs_base": _round_or_none(learned.get("ce_improvement_vs_base")),
        "teacher_ce_gap_closure_fraction": _round_or_none(learned.get("teacher_ce_gap_closure_fraction")),
        "teacher_residual_reconstruction_mse": _round_or_none(learned.get("teacher_residual_reconstruction_mse")),
        "teacher_residual_reconstruction_r2": _round_or_none(learned.get("teacher_residual_reconstruction_r2")),
        "functional_churn": _round_or_none(learned.get("functional_churn")),
        "retention_proxy": _round_or_none(learned.get("retention_proxy")),
        "finite_update_commutator_proxy": _round_or_none(learned.get("finite_update_commutator_proxy")),
        "intervention_selectivity_proxy": _round_or_none(learned.get("intervention_selectivity_proxy")),
        "support_load_entropy": _round_or_none(learned.get("support_load_entropy")),
        "support_overlap_with_oracle": _round_or_none(learned.get("support_overlap_with_oracle")),
        "active_rank_proxy": learned.get("active_rank_proxy", ""),
        "residual_l2_mean": _round_or_none(learned.get("residual_l2_mean")),
        "residual_l2_p95": _round_or_none(learned.get("residual_l2_p95")),
        "teacher_residual_l2_mean": _round_or_none(learned.get("teacher_residual_l2_mean")),
        "residual_l2_mean_ratio_vs_teacher": _round_or_none(learned.get("residual_l2_mean_ratio_vs_teacher")),
        "active_params": learned.get("active_params", ""),
        "stored_params": learned.get("stored_params", ""),
        "oracle_support_non_deployable": False,
        "uses_future_hidden_or_delta": False,
        "uses_oracle_support_at_eval": False,
        "uses_task_id": False,
        "uses_teacher_labels_in_deployable_router": False,
        "target_access_at_eval": "artifact_reference_only",
        "feature_schema_hash": "support_forcing_reference_v1",
        "note": "Reference row copied from the completed support-forcing/pruning pregate; not retrained here.",
    }


def _gate_rows(
    source_rows: list[dict[str, Any]],
    spectrum_rows: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    base_ce: float,
    teacher_ce: float,
) -> list[dict[str, Any]]:
    arms = {row["arm"]: row for row in arm_rows}
    oracle = arms.get("oracle_topk_orthogonal_sparse_coding", {})
    learned = arms.get("learned_router_topk_scalar_sparse_coding", {})
    flat = arms.get("same_router_flat_value_control", {})
    random_null = arms.get("random_topk_sparse_coding_null", {})
    permuted_null = arms.get("load_permuted_topk_sparse_coding_null", {})
    position_null = arms.get("token_position_topk_sparse_coding_null", {})
    no_update = arms.get("no_update_control", {})
    oracle_r2 = _float(oracle.get("teacher_residual_reconstruction_r2"), -math.inf)
    flat_r2 = _float(flat.get("teacher_residual_reconstruction_r2"), -math.inf)
    learned_r2 = _float(learned.get("teacher_residual_reconstruction_r2"), -math.inf)
    no_update_r2 = _float(no_update.get("teacher_residual_reconstruction_r2"), -math.inf)
    best_null_r2 = max(
        _float(random_null.get("teacher_residual_reconstruction_r2"), -math.inf),
        _float(permuted_null.get("teacher_residual_reconstruction_r2"), -math.inf),
        _float(position_null.get("teacher_residual_reconstruction_r2"), -math.inf),
        no_update_r2,
    )
    oracle_gain = oracle_r2 - no_update_r2
    learned_gain = learned_r2 - no_update_r2
    effective_rank = _float(spectrum_rows[0].get("effective_rank") if spectrum_rows else None, math.inf)
    return [
        _gate("support_forcing_source_present", bool(source_rows[0].get("present")), True, "runtime", str(source_rows[0])),
        _gate("strategy_review_present", bool(source_rows[1].get("present")), True, "runtime", str(source_rows[1])),
        _gate("required_arms_present", set(ARMS).issubset(arms), True, "runtime", ",".join(sorted(arms))),
        _gate("spectrum_rows_present", bool(spectrum_rows), True, "runtime", f"rows={len(spectrum_rows)}; effective_rank={effective_rank:.6f}"),
        _gate("gpu_blocked", True, True, "runtime", "requires_gpu_now=false; advance_to_gpu_validation=false"),
        _gate("dense_teacher_improves_base", teacher_ce < base_ce, False, "scientific", f"base_ce={base_ce:.6f}; teacher_ce={teacher_ce:.6f}"),
        _gate("oracle_sparse_beats_flat_or_is_near", oracle_r2 >= 0.5 or oracle_r2 >= flat_r2 - 0.1, False, "scientific", f"oracle_r2={oracle_r2:.6f}; flat_r2={flat_r2:.6f}"),
        _gate("oracle_sparse_beats_null_controls", oracle_r2 > best_null_r2 + 0.05, False, "scientific", f"oracle_r2={oracle_r2:.6f}; best_null_r2={best_null_r2:.6f}"),
        _gate("learned_support_retains_oracle_gain", oracle_gain > 0.0 and learned_gain >= 0.8 * oracle_gain, False, "scientific", f"learned_gain={learned_gain:.6f}; oracle_gain={oracle_gain:.6f}"),
        _gate("spectrum_not_trivially_dense_for_topk", effective_rank <= max(4.0, float(arms['oracle_topk_orthogonal_sparse_coding']['active_rank_proxy']) + 2.0), False, "scientific", f"effective_rank={effective_rank:.6f}; oracle_active_rank={arms['oracle_topk_orthogonal_sparse_coding']['active_rank_proxy']}"),
    ]


def _selected_next_step(status: str, scientific_failures: list[dict[str, Any]]) -> str:
    if status != "pass":
        return "repair oracle sparse-coding feasibility runtime artifacts before interpretation"
    failed = {row["criterion"] for row in scientific_failures}
    if "oracle_sparse_beats_flat_or_is_near" in failed or "oracle_sparse_beats_null_controls" in failed:
        return "do not open another sparse value/support redesign; demote columns to diagnostics unless a stronger oracle basis is proposed"
    if "learned_support_retains_oracle_gain" in failed:
        return "improve deployable router/scalar imitation for the feasible oracle sparse-coding basis; keep local CPU and require low oracle-gain regret before GPU"
    if scientific_failures:
        return "inspect residual spectrum and guardrail failures before any backend validation"
    return "design deployable router imitation for the feasible oracle sparse-coding basis, still local CPU first"


def _gate(criterion: str, passed: bool, required: bool, gate_type: str, evidence: str) -> dict[str, Any]:
    return {"criterion": criterion, "passed": bool(passed), "required": required, "gate_type": gate_type, "evidence": evidence}


def _review_field(text: str, field: str) -> str:
    prefix = f"{field}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_or_none(value: Any) -> float | str:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return ""


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "spectrum_rows.csv", summary["spectrum_rows"])
    _write_csv(out_dir / "arm_metrics.csv", summary["arm_metrics"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Dense-Teacher Oracle Sparse-Coding Feasibility",
            "",
            f"Decision: `{summary['decision']}`.",
            f"Claim status: `{summary['claim_status']}`.",
            "",
            "This is local CPU feasibility evidence only. GPU validation remains blocked.",
            f"Ben notification recommended by strategy review: `{summary['ben_notification_recommended']}`.",
            f"Next step: {summary['selected_next_step']}",
            "",
        ]
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-forcing", type=Path, default=DEFAULT_SUPPORT_FORCING)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--teacher-steps", type=int, default=100)
    parser.add_argument("--router-steps", type=int, default=80)
    parser.add_argument("--control-steps", type=int, default=80)
    parser.add_argument("--basis-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--data-column-count", type=int, default=6)
    args = parser.parse_args(argv)
    summary = run_dense_teacher_oracle_sparse_coding_feasibility(
        support_forcing_path=args.support_forcing,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
        seed=args.seed,
        teacher_steps=args.teacher_steps,
        router_steps=args.router_steps,
        control_steps=args.control_steps,
        basis_size=args.basis_size,
        top_k=args.top_k,
        data_column_count=args.data_column_count,
    )
    print(json.dumps({key: summary[key] for key in ("status", "decision", "claim_status", "selected_next_step")}, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
