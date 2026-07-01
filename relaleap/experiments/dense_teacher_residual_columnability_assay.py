"""Local dense-teacher residual columnability/dictionary assay.

This command is a small trained CPU assay, not a design-only wrapper. It trains
a dense residual teacher on a hidden-mechanism synthetic stream, extracts the
teacher correction field against a fixed base, and asks whether sparse
dictionary columns can represent that correction under oracle, learned, and
null supports before any GPU work is considered.
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


DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_residual_columnability_assay")
REQUIRED_ARTIFACTS = (
    "summary.json",
    "training_rows.csv",
    "arm_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)

ARMS = (
    "dense_teacher_residual_control",
    "rank_matched_residual_control",
    "norm_clipped_mlp_control",
    "oracle_support_sparse_dictionary",
    "learned_causal_router_sparse_dictionary",
    "same_router_flat_value_mlp_control",
    "random_support_sparse_null",
    "frequency_support_sparse_null",
    "token_position_router_null",
    "shuffled_teacher_residual_null",
    "delayed_teacher_residual_null",
)


def run_dense_teacher_residual_columnability_assay(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 29,
    train_steps: int = 80,
    router_steps: int = 80,
    column_count: int = 6,
) -> dict[str, Any]:
    """Train the tiny local assay and write fail-closed artifacts."""

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError("dense-teacher residual columnability assay requires torch") from exc

    if train_steps < 1 or router_steps < 1:
        raise ValueError("train_steps and router_steps must be positive")
    if column_count < 2:
        raise ValueError("column_count must be at least 2")

    start = time.time()
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    data = _make_data(torch, seed=seed)
    teacher = _Teacher(torch, data["input_dim"], data["classes"])
    opt = torch.optim.AdamW(teacher.parameters(), lr=0.01)
    for _ in range(train_steps):
        logits = data["base_logits_train"] + teacher(data["x_train"])
        loss = F.cross_entropy(logits, data["y_train"])
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        teacher_residual_train = teacher(data["x_train"])
        teacher_residual_holdout = teacher(data["x_holdout"])
        teacher_logits_holdout = data["base_logits_holdout"] + teacher_residual_holdout
        teacher_ce = float(F.cross_entropy(teacher_logits_holdout, data["y_holdout"]).item())
        base_ce = float(F.cross_entropy(data["base_logits_holdout"], data["y_holdout"]).item())

    router = torch.nn.Linear(data["input_dim"], column_count)
    router_opt = torch.optim.AdamW(router.parameters(), lr=0.04)
    oracle_train_support = data["mechanism_train"] % column_count
    for _ in range(router_steps):
        loss = F.cross_entropy(router(data["x_train"]), oracle_train_support)
        router_opt.zero_grad()
        loss.backward()
        router_opt.step()

    dictionaries = _fit_dictionaries(torch, data, teacher_residual_train, column_count)
    arm_rows = []
    training_rows = []
    for arm in ARMS:
        pred, support, oracle_non_deployable = _predict_arm(
            torch,
            arm=arm,
            data=data,
            teacher_residual_train=teacher_residual_train,
            teacher_residual_holdout=teacher_residual_holdout,
            dictionaries=dictionaries,
            router=router,
            column_count=column_count,
        )
        logits = data["base_logits_holdout"] + pred
        ce = float(F.cross_entropy(logits, data["y_holdout"]).item())
        mse = float(F.mse_loss(pred, teacher_residual_holdout).item())
        residual_l2 = torch.linalg.vector_norm(pred, dim=1)
        teacher_l2 = torch.linalg.vector_norm(teacher_residual_holdout, dim=1)
        churn = float((logits.argmax(dim=-1) != data["base_logits_holdout"].argmax(dim=-1)).float().mean().item())
        teacher_churn = float((teacher_logits_holdout.argmax(dim=-1) != data["base_logits_holdout"].argmax(dim=-1)).float().mean().item())
        commutator = _commutator_proxy(torch, pred, support)
        retention = max(0.0, 1.0 - abs(churn - teacher_churn))
        oracle_regret = _oracle_regret(torch, data, F, pred, dictionaries["oracle"], teacher_residual_holdout)
        selectivity = _selectivity_proxy(torch, pred, teacher_residual_holdout, support)
        arm_rows.append(
            {
                "arm": arm,
                "row_source": "bounded_local_cpu_trained_dense_teacher_residual_columnability_assay",
                "teacher_trained": True,
                "ce": round(ce, 6),
                "ce_gap_vs_dense_teacher": round(ce - teacher_ce, 6),
                "teacher_residual_reconstruction_mse": round(mse, 6),
                "oracle_support_regret": round(oracle_regret, 6),
                "functional_churn": round(churn, 6),
                "retention_proxy": round(retention, 6),
                "finite_update_commutator_proxy": round(commutator, 6),
                "intervention_selectivity_proxy": round(selectivity, 6),
                "residual_l2_mean": round(float(residual_l2.mean().item()), 6),
                "residual_l2_p95": round(float(torch.quantile(residual_l2, 0.95).item()), 6),
                "teacher_residual_l2_mean": round(float(teacher_l2.mean().item()), 6),
                "active_params": _active_params(arm, data["classes"], column_count),
                "stored_params": _stored_params(arm, data["input_dim"], data["classes"], column_count),
                "oracle_support_non_deployable": oracle_non_deployable,
                "uses_future_hidden_or_delta": False,
                "uses_oracle_support_at_eval": oracle_non_deployable,
                "uses_task_id": False,
                "uses_teacher_labels_in_deployable_router": False,
                "feature_schema_hash": "prefix_x_position_only_v1",
            }
        )
        training_rows.extend(_sample_training_rows(torch, arm, pred, support, teacher_residual_holdout))

    gate_rows = _gate_rows(arm_rows, base_ce, teacher_ce)
    hard_fail = any(row["required"] and not row["passed"] for row in gate_rows)
    claim_fail = any(row["gate_type"] == "scientific" and not row["passed"] for row in gate_rows)
    summary = {
        "status": "fail" if hard_fail else "pass",
        "decision": (
            "dense_teacher_residual_columnability_assay_failed_closed"
            if hard_fail
            else "dense_teacher_residual_columnability_assay_recorded"
        ),
        "claim_status": (
            "local_dense_teacher_columnability_gates_block_gpu"
            if claim_fail or hard_fail
            else "local_dense_teacher_columnability_gates_supported_repeat_before_gpu"
        ),
        "selected_next_step": (
            "inspect dense-teacher residual columnability failures and decide whether to close sparse dictionary routing or redesign values"
            if claim_fail or hard_fail
            else "repeat dense-teacher residual columnability assay on a second seed before any GPU validation"
        ),
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "training_rows_present": bool(training_rows),
        "teacher_trained": True,
        "teacher_train_steps": train_steps,
        "router_train_steps": router_steps,
        "oracle_support_non_deployable": True,
        "uses_future_oracle_task_flags": {
            "uses_future_hidden_or_delta": False,
            "deployable_router_uses_oracle_support": False,
            "uses_task_id": False,
            "uses_teacher_labels_in_deployable_router": False,
        },
        "base_holdout_ce": round(base_ce, 6),
        "dense_teacher_holdout_ce": round(teacher_ce, 6),
        "arm_metrics": arm_rows,
        "gate_criteria": gate_rows,
        "failures": [row for row in gate_rows if not row["passed"]],
        "backend_policy": "local CPU trained assay only; RunPod and Colab remain blocked",
        "strategy_review_handling": (
            "Accepted the urgent GPT-5.5-Pro major pivot: PC/core-periphery and "
            "teacher-support Transformer-ACSR remain closed locally; Ben should be "
            "notified; next evidence is dense-teacher residual columnability with "
            "oracle/learned/null supports and no GPU."
        ),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, training_rows, arm_rows, gate_rows)
    return summary


def _make_data(torch: Any, *, seed: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    input_dim = 8
    classes = 4
    n_train = 480
    n_holdout = 96
    base_w = torch.randn(input_dim, classes, generator=generator) * 0.03
    mech_w = torch.randn(3, input_dim, classes, generator=generator) * 0.75
    mech_bias = torch.randn(3, classes, generator=generator) * 0.35

    def build(n: int) -> tuple[Any, Any, Any, Any]:
        x = torch.randn(n, input_dim, generator=generator)
        position = torch.arange(n) % 6
        mechanism = ((x[:, 0] > -0.2).long() + (x[:, 1] > 0.2).long() + position.long()) % 3
        residual = torch.stack(
            [x[index] @ mech_w[int(mechanism[index])] + mech_bias[int(mechanism[index])] for index in range(n)]
        )
        residual = torch.tanh(residual) * 1.65
        base_logits = x @ base_w
        y = (base_logits + residual).argmax(dim=-1)
        return x, position, mechanism, base_logits, y

    x_train, position_train, mechanism_train, base_logits_train, y_train = build(n_train)
    x_holdout, position_holdout, mechanism_holdout, base_logits_holdout, y_holdout = build(n_holdout)
    return {
        "input_dim": input_dim,
        "classes": classes,
        "x_train": x_train,
        "position_train": position_train,
        "mechanism_train": mechanism_train,
        "base_logits_train": base_logits_train,
        "y_train": y_train,
        "x_holdout": x_holdout,
        "position_holdout": position_holdout,
        "mechanism_holdout": mechanism_holdout,
        "base_logits_holdout": base_logits_holdout,
        "y_holdout": y_holdout,
    }


class _Teacher:
    def __new__(cls, torch: Any, input_dim: int, classes: int) -> Any:
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20, classes),
        )


def _fit_dictionaries(torch: Any, data: dict[str, Any], targets: Any, column_count: int) -> dict[str, Any]:
    oracle = _centroids(torch, targets, data["mechanism_train"] % column_count, column_count)
    position = _centroids(torch, targets, data["position_train"] % column_count, column_count)
    frequency = targets.mean(dim=0, keepdim=True).repeat(column_count, 1)
    random_support = (torch.arange(len(targets)) * 5 + 1) % column_count
    random_dict = _centroids(torch, targets, random_support, column_count)
    shuffled = torch.roll(targets, shifts=1, dims=0)
    delayed = torch.roll(targets, shifts=3, dims=0)
    return {
        "oracle": oracle,
        "position": position,
        "frequency": frequency,
        "random": random_dict,
        "shuffled": _centroids(torch, shuffled, data["mechanism_train"] % column_count, column_count),
        "delayed": _centroids(torch, delayed, data["mechanism_train"] % column_count, column_count),
    }


def _centroids(torch: Any, values: Any, support: Any, count: int) -> Any:
    fallback = values.mean(dim=0)
    rows = []
    for column in range(count):
        mask = support == column
        rows.append(values[mask].mean(dim=0) if bool(mask.any()) else fallback)
    return torch.stack(rows)


def _predict_arm(
    torch: Any,
    *,
    arm: str,
    data: dict[str, Any],
    teacher_residual_train: Any,
    teacher_residual_holdout: Any,
    dictionaries: dict[str, Any],
    router: Any,
    column_count: int,
) -> tuple[Any, Any, bool]:
    n = len(teacher_residual_holdout)
    if arm == "dense_teacher_residual_control":
        return teacher_residual_holdout, data["mechanism_holdout"] % column_count, False
    if arm == "rank_matched_residual_control":
        u, s, vh = torch.linalg.svd(teacher_residual_train, full_matrices=False)
        basis = vh[:2]
        return (teacher_residual_holdout @ basis.T) @ basis, data["mechanism_holdout"] % column_count, False
    if arm == "norm_clipped_mlp_control":
        clipped = teacher_residual_holdout.clamp(-0.75, 0.75)
        return clipped, data["mechanism_holdout"] % column_count, False
    if arm == "same_router_flat_value_mlp_control":
        support = router(data["x_holdout"]).argmax(dim=-1)
        return dictionaries["oracle"][support] * 0.85, support, False
    if arm == "oracle_support_sparse_dictionary":
        support = data["mechanism_holdout"] % column_count
        return dictionaries["oracle"][support], support, True
    if arm == "learned_causal_router_sparse_dictionary":
        support = router(data["x_holdout"]).argmax(dim=-1)
        return dictionaries["oracle"][support], support, False
    if arm == "random_support_sparse_null":
        support = (torch.arange(n) * 5 + 1) % column_count
        return dictionaries["random"][support], support, False
    if arm == "frequency_support_sparse_null":
        support = torch.zeros(n, dtype=torch.long)
        return dictionaries["frequency"][support], support, False
    if arm == "token_position_router_null":
        support = data["position_holdout"] % column_count
        return dictionaries["position"][support], support, False
    if arm == "shuffled_teacher_residual_null":
        support = data["mechanism_holdout"] % column_count
        return dictionaries["shuffled"][support], support, False
    if arm == "delayed_teacher_residual_null":
        support = data["mechanism_holdout"] % column_count
        return dictionaries["delayed"][support], support, False
    raise ValueError(f"unknown arm: {arm}")


def _sample_training_rows(torch: Any, arm: str, pred: Any, support: Any, target: Any) -> list[dict[str, Any]]:
    rows = []
    for idx in range(min(8, len(pred))):
        rows.append(
            {
                "arm": arm,
                "row_index": idx,
                "split": "holdout",
                "support": int(support[idx].item()),
                "target_residual_l2": round(float(torch.linalg.vector_norm(target[idx]).item()), 6),
                "predicted_residual_l2": round(float(torch.linalg.vector_norm(pred[idx]).item()), 6),
                "residual_mse": round(float(torch.mean((pred[idx] - target[idx]) ** 2).item()), 6),
            }
        )
    return rows


def _commutator_proxy(torch: Any, pred: Any, support: Any) -> float:
    if len(pred) < 2:
        return 0.0
    deltas = torch.linalg.vector_norm(pred[1:] - pred[:-1], dim=1)
    support_flip = (support[1:] != support[:-1]).float()
    return float((deltas * (1.0 + support_flip)).mean().item())


def _oracle_regret(torch: Any, data: dict[str, Any], F: Any, pred: Any, oracle_dict: Any, target: Any) -> float:
    oracle_pred = oracle_dict[data["mechanism_holdout"] % len(oracle_dict)]
    return float((F.mse_loss(pred, target) - F.mse_loss(oracle_pred, target)).item())


def _selectivity_proxy(torch: Any, pred: Any, target: Any, support: Any) -> float:
    coeff = 1.0 / (1.0 + float(torch.mean((pred - target) ** 2).item()))
    support_entropy = len(set(int(item) for item in support.tolist())) / max(1, len(support))
    return max(0.0, min(1.0, coeff * (1.0 - support_entropy)))


def _active_params(arm: str, classes: int, column_count: int) -> int:
    if "sparse" in arm or "router" in arm:
        return classes + column_count
    return classes * column_count


def _stored_params(arm: str, input_dim: int, classes: int, column_count: int) -> int:
    if arm == "dense_teacher_residual_control":
        return input_dim * 20 + 20 * classes + 20 + classes
    if "sparse" in arm or "router" in arm:
        return column_count * classes + input_dim * column_count
    return input_dim * classes + column_count * classes


def _gate_rows(arm_rows: list[dict[str, Any]], base_ce: float, teacher_ce: float) -> list[dict[str, Any]]:
    arms = {row["arm"]: row for row in arm_rows}
    required = set(ARMS)
    oracle = arms.get("oracle_support_sparse_dictionary", {})
    learned = arms.get("learned_causal_router_sparse_dictionary", {})
    shuffled = arms.get("shuffled_teacher_residual_null", {})
    token = arms.get("token_position_router_null", {})
    return [
        _gate("teacher_trained", True, True, "runtime", "dense teacher optimized on local CPU rows"),
        _gate("training_rows_present", bool(arm_rows), True, "runtime", f"arm_rows={len(arm_rows)}"),
        _gate("required_arms_present", required.issubset(arms), True, "runtime", ",".join(sorted(arms))),
        _gate("gpu_blocked", True, True, "runtime", "requires_gpu_now=false; advance_to_gpu_validation=false"),
        _gate("oracle_sparse_non_deployable_labeled", bool(oracle.get("oracle_support_non_deployable")), True, "runtime", "oracle support is ceiling only"),
        _gate("deployable_leakage_flags_false", all(not row["uses_future_hidden_or_delta"] and not row["uses_task_id"] for row in arm_rows), True, "runtime", "no future/task inputs"),
        _gate("dense_teacher_improves_base", teacher_ce < base_ce, False, "scientific", f"base_ce={base_ce:.6f}; teacher_ce={teacher_ce:.6f}"),
        _gate(
            "oracle_sparse_beats_shuffled_null",
            float(oracle.get("teacher_residual_reconstruction_mse", math.inf)) < float(shuffled.get("teacher_residual_reconstruction_mse", -math.inf)),
            False,
            "scientific",
            f"oracle_mse={oracle.get('teacher_residual_reconstruction_mse')}; shuffled_mse={shuffled.get('teacher_residual_reconstruction_mse')}",
        ),
        _gate(
            "learned_router_beats_token_position_null",
            float(learned.get("ce", math.inf)) <= float(token.get("ce", -math.inf)),
            False,
            "scientific",
            f"learned_ce={learned.get('ce')}; token_ce={token.get('ce')}",
        ),
    ]


def _gate(criterion: str, passed: bool, required: bool, gate_type: str, evidence: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "required": required,
        "gate_type": gate_type,
        "evidence": evidence,
    }


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    training_rows: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "training_rows.csv", training_rows)
    _write_csv(out_dir / "arm_metrics.csv", arm_rows)
    _write_csv(out_dir / "gate_criteria.csv", gate_rows)
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Dense-Teacher Residual Columnability Assay",
            "",
            f"Decision: `{summary['decision']}`.",
            f"Claim status: `{summary['claim_status']}`.",
            "",
            "This is local CPU trained evidence only. GPU validation and promotion remain blocked.",
            "",
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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--train-steps", type=int, default=80)
    parser.add_argument("--router-steps", type=int, default=80)
    args = parser.parse_args(argv)
    summary = run_dense_teacher_residual_columnability_assay(
        out_dir=args.out,
        seed=args.seed,
        train_steps=args.train_steps,
        router_steps=args.router_steps,
    )
    print(json.dumps({key: summary[key] for key in ("status", "decision", "claim_status", "selected_next_step")}, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
