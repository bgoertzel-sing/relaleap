"""Capture raw low-churn MLP residual vectors for sparse-factorization work."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.low_churn_mlp_residual_control_pilot import (
    LOW_CHURN_ARM,
    _GatedMLPResidual,
    _budget_values,
    _project_to_l2_budget,
    _train_low_churn_module,
)
from relaleap.experiments.norm_budgeted_churn_regularized_residual_pilot import (
    _float,
    _heldout_mask,
    _module_update,
    _per_token_anchor_kl,
    _per_token_ce,
)


DEFAULT_TRAINING_HARNESS_DIR = Path("results/reports/low_churn_mlp_sparse_factorization_ceiling_training_harness")
DEFAULT_PREGATE_DIR = Path("results/reports/low_churn_mlp_residual_control_pregate")
DEFAULT_OUT_DIR = Path("results/reports/low_churn_mlp_sparse_factorization_vector_capture")

NEXT_ACTION = "implement_vector_sparse_factorization_ceiling_training"
REPAIR_ACTION = "repair_low_churn_mlp_sparse_factorization_vector_capture_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "raw_teacher_residual_vectors.csv",
    "logit_intervention_rows.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_low_churn_mlp_sparse_factorization_vector_capture(
    *,
    training_harness_dir: Path = DEFAULT_TRAINING_HARNESS_DIR,
    pregate_dir: Path = DEFAULT_PREGATE_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    train_steps: int = 12,
    seed: int = 1,
) -> dict[str, Any]:
    """Train the bounded low-churn teacher and persist vector-level rows."""

    start = time.time()
    harness_summary = _read_json(training_harness_dir / "summary.json")
    pregate_summary = _read_json(pregate_dir / "summary.json")
    budgets = _budget_values(pregate_summary.get("budget_rows", []))
    source_rows = _source_rows(training_harness_dir, pregate_dir, harness_summary, pregate_summary, budgets)
    preflight = _preflight_rows(harness_summary, pregate_summary, budgets, train_steps)
    vector_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    runtime_error = ""

    if all(row["passed"] for row in preflight):
        try:
            vector_rows, intervention_rows = _capture_rows(budgets=budgets, train_steps=train_steps, seed=seed)
        except Exception as exc:  # pragma: no cover - torch/runtime dependent
            runtime_error = f"{type(exc).__name__}: {exc}"

    gate_rows = preflight + _capture_gate_rows(vector_rows, intervention_rows, runtime_error)
    runtime_failures = [row for row in gate_rows if not row["passed"] and row["gate_type"] == "runtime"]
    advancement_failures = [row for row in gate_rows if not row["passed"] and row["gate_type"] == "scientific_advancement"]
    status = "pass" if not runtime_failures else "fail"
    summary = {
        "status": status,
        "decision": (
            "low_churn_mlp_sparse_factorization_vector_capture_recorded"
            if status == "pass"
            else "low_churn_mlp_sparse_factorization_vector_capture_failed_closed"
        ),
        "claim_status": "raw_teacher_vectors_captured_no_sparse_factorization_claim",
        "selected_next_action": NEXT_ACTION if status == "pass" else REPAIR_ACTION,
        "selected_next_step": (
            "train a vector-level sparse-factorization ceiling against the captured teacher rows"
            if status == "pass"
            else "repair source artifacts or runtime before vector-level sparse factorization"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "training_executed": status == "pass",
        "training_scope": "local low-churn teacher replay plus raw vector/logit capture",
        "backend_policy": "RunPod/Colab remain blocked until vector sparse training, CE transfer, churn, commutator, and intervention gates pass",
        "training_harness_dir": str(training_harness_dir),
        "pregate_dir": str(pregate_dir),
        "out_dir": str(out_dir),
        "seed": seed,
        "train_steps": train_steps,
        "source_rows": source_rows,
        "raw_teacher_vector_row_count": len(vector_rows),
        "heldout_raw_teacher_vector_row_count": sum(1 for row in vector_rows if row.get("split") == "heldout"),
        "logit_intervention_row_count": len(intervention_rows),
        "runtime_failures": runtime_failures,
        "advancement_failures": advancement_failures,
        "gate_criteria": gate_rows,
        "runtime_error": runtime_error,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, source_rows, vector_rows, intervention_rows, gate_rows)
    return summary


def _capture_rows(*, budgets: dict[str, float], train_steps: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch
    import torch.nn.functional as F

    from relaleap.smoke import TinyCharTransformer, _build_batch

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    inputs, targets, vocab_size = _build_batch("tiny_shakespeare_char", seq_len=32, batch_size=4)
    base = TinyCharTransformer(vocab_size=vocab_size, seq_len=32, hidden_dim=32, layers=2)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base.eval()
    with torch.no_grad():
        hidden = base.encode(inputs)
        base_logits = base.decode(hidden)
        base_losses = _per_token_ce(F, base_logits, targets, vocab_size).reshape(-1)
    module = _GatedMLPResidual(torch.nn, 32, bottleneck=16)
    _train_low_churn_module(
        torch=torch,
        F=F,
        base=base,
        module=module,
        hidden=hidden,
        targets=targets,
        vocab_size=vocab_size,
        base_logits=base_logits,
        budgets=budgets,
        train_steps=train_steps,
        shuffled_targets=False,
    )
    module.eval()
    with torch.no_grad():
        raw_update = _module_update(module, hidden)
        update = _project_to_l2_budget(raw_update, budgets["dense24_residual_l2_ceiling"])
        teacher_logits = base.decode(hidden + update)
        teacher_losses = _per_token_ce(F, teacher_logits, targets, vocab_size).reshape(-1)
        update_flat = update[:, :-1, :].reshape(-1, update.shape[-1])
        base_flat = base_logits[:, :-1, :].reshape(-1, vocab_size)
        teacher_flat = teacher_logits[:, :-1, :].reshape(-1, vocab_size)
        logit_delta = teacher_flat - base_flat
        logit_mse = (logit_delta**2).mean(dim=-1)
        anchor_kl = _per_token_anchor_kl(F, teacher_flat, base_flat)
        prediction_changed = teacher_flat.argmax(dim=-1) != base_flat.argmax(dim=-1)
    heldout_mask = _heldout_mask(base_losses.shape)
    vector_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    flat_targets = targets[:, :-1].reshape(-1)
    for index in range(int(base_losses.numel())):
        split = "heldout" if bool(heldout_mask[index].item()) else "train_anchor"
        residual_vector = update_flat[index]
        base_row = base_flat[index]
        teacher_row = teacher_flat[index]
        delta_row = logit_delta[index]
        vector_rows.append(
            {
                "teacher_arm": LOW_CHURN_ARM,
                "teacher_row_id": f"{LOW_CHURN_ARM}:{split}:{index}",
                "token_index": index,
                "split": split,
                "target_token_id": int(flat_targets[index].item()),
                "hidden_dim": int(residual_vector.numel()),
                "vocab_size": int(delta_row.numel()),
                "base_ce_loss": float(base_losses[index].item()),
                "teacher_ce_loss": float(teacher_losses[index].item()),
                "teacher_delta_vs_base_ce": float(teacher_losses[index].item() - base_losses[index].item()),
                "teacher_residual_update_l2": float(residual_vector.norm().item()),
                "teacher_logit_delta_l2": float(delta_row.norm().item()),
                "teacher_logit_mse_vs_base": float(logit_mse[index].item()),
                "teacher_anchor_kl_vs_base": float(anchor_kl[index].item()),
                "teacher_prediction_changed_vs_base": bool(prediction_changed[index].item()),
                "raw_teacher_vector_available": True,
                "raw_intervention_available": True,
                "teacher_residual_update_vector": _json_vector(residual_vector),
                "base_logits": _json_vector(base_row),
                "teacher_logits": _json_vector(teacher_row),
                "teacher_logit_delta": _json_vector(delta_row),
            }
        )
        intervention_rows.append(
            {
                "teacher_arm": LOW_CHURN_ARM,
                "teacher_row_id": f"{LOW_CHURN_ARM}:{split}:{index}",
                "token_index": index,
                "split": split,
                "base_ce_loss": float(base_losses[index].item()),
                "teacher_ce_loss": float(teacher_losses[index].item()),
                "teacher_gain_vs_base_ce": float(base_losses[index].item() - teacher_losses[index].item()),
                "necessity_zero_update_ce_delta": float(base_losses[index].item() - teacher_losses[index].item()),
                "teacher_residual_update_l2": float(residual_vector.norm().item()),
                "teacher_logit_delta_l2": float(delta_row.norm().item()),
                "teacher_anchor_kl_vs_base": float(anchor_kl[index].item()),
                "teacher_prediction_changed_vs_base": bool(prediction_changed[index].item()),
                "intervention_role": "teacher_residual_applied_vs_zero_update",
            }
        )
    return vector_rows, intervention_rows


def _source_rows(
    training_harness_dir: Path,
    pregate_dir: Path,
    harness_summary: dict[str, Any],
    pregate_summary: dict[str, Any],
    budgets: dict[str, float],
) -> list[dict[str, Any]]:
    return [
        {
            "source": "sparse_factorization_training_harness",
            "path": str(training_harness_dir / "summary.json"),
            "present": (training_harness_dir / "summary.json").is_file(),
            "status": harness_summary.get("status", ""),
            "decision": harness_summary.get("decision", ""),
            "selected_next_action": harness_summary.get("selected_next_action", ""),
            "row_count": 1 if harness_summary else 0,
        },
        {
            "source": "low_churn_mlp_residual_control_pregate",
            "path": str(pregate_dir / "summary.json"),
            "present": (pregate_dir / "summary.json").is_file(),
            "status": pregate_summary.get("status", ""),
            "decision": pregate_summary.get("decision", ""),
            "selected_next_action": pregate_summary.get("selected_next_action", ""),
            "row_count": 1 if pregate_summary else 0,
        },
        {
            "source": "low_churn_budget_values",
            "path": str(pregate_dir / "summary.json"),
            "present": bool(budgets),
            "status": "read" if budgets else "missing_or_empty",
            "decision": "",
            "selected_next_action": "",
            "row_count": len(budgets),
        },
    ]


def _preflight_rows(
    harness_summary: dict[str, Any],
    pregate_summary: dict[str, Any],
    budgets: dict[str, float],
    train_steps: int,
) -> list[dict[str, Any]]:
    required_budgets = {
        "dense24_residual_l2_ceiling",
        "dense24_anchor_logit_mse_ceiling",
        "dense24_flip_churn_ceiling",
        "dense24_ce_reference",
    }
    return [
        _criterion(
            "training_harness_selected_vector_capture",
            harness_summary.get("status") == "pass"
            and harness_summary.get("selected_next_action")
            == "capture_raw_low_churn_teacher_residual_vectors_for_sparse_factorization",
            harness_summary.get("selected_next_action", "missing"),
            "previous harness must select raw vector capture",
            "runtime",
        ),
        _criterion(
            "pregate_budget_source_available",
            pregate_summary.get("status") == "pass" and required_budgets.issubset(budgets),
            {"pregate_status": pregate_summary.get("status", ""), "budget_keys": sorted(budgets)},
            "low-churn pregate must provide dense24 CE/L2/drift/churn budgets",
            "runtime",
        ),
        _criterion(
            "train_steps_bounded",
            1 <= train_steps <= 32,
            train_steps,
            "train_steps must remain in the bounded local CPU range [1, 32]",
            "runtime",
        ),
    ]


def _capture_gate_rows(
    vector_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    runtime_error: str,
) -> list[dict[str, Any]]:
    vector_fields = set(vector_rows[0]) if vector_rows else set()
    intervention_fields = set(intervention_rows[0]) if intervention_rows else set()
    return [
        _criterion("capture_runtime_completed", not runtime_error, runtime_error or "ok", "vector capture runtime must complete", "runtime"),
        _criterion(
            "raw_teacher_vectors_written",
            bool(vector_rows)
            and {"teacher_residual_update_vector", "base_logits", "teacher_logits", "teacher_logit_delta"}.issubset(vector_fields)
            and all(row.get("raw_teacher_vector_available") is True for row in vector_rows),
            {"row_count": len(vector_rows), "fields": sorted(vector_fields)},
            "raw residual vectors and logits must be serialized for every teacher row",
            "runtime",
        ),
        _criterion(
            "heldout_vector_rows_available",
            any(row.get("split") == "heldout" for row in vector_rows),
            {"heldout_rows": sum(1 for row in vector_rows if row.get("split") == "heldout"), "row_count": len(vector_rows)},
            "heldout vector rows are required for sparse-factorization evaluation",
            "runtime",
        ),
        _criterion(
            "logit_intervention_rows_written",
            bool(intervention_rows)
            and {"teacher_gain_vs_base_ce", "necessity_zero_update_ce_delta", "teacher_anchor_kl_vs_base"}.issubset(intervention_fields),
            {"row_count": len(intervention_rows), "fields": sorted(intervention_fields)},
            "logit intervention rows must include CE gain, zero-update necessity, and anchor drift",
            "runtime",
        ),
        _criterion(
            "vector_sparse_training_not_yet_executed",
            False,
            "raw capture only",
            "a later vector-level sparse-factorization harness must train oracle/learned/null arms before scientific advancement",
            "scientific_advancement",
        ),
    ]


def _criterion(criterion: str, passed: bool, actual: Any, threshold: str, gate_type: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "actual": actual,
        "threshold": threshold,
        "gate_type": gate_type,
        "failure_reason": "" if passed else threshold,
    }


def _json_vector(value: Any) -> str:
    return json.dumps([round(float(item), 8) for item in value.detach().cpu().tolist()])


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    vector_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", source_rows)
    _write_csv(out_dir / "raw_teacher_residual_vectors.csv", vector_rows)
    _write_csv(out_dir / "logit_intervention_rows.csv", intervention_rows)
    _write_csv(out_dir / "gate_criteria.csv", gate_rows)
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"], lineterminator="\n")
        writer.writeheader()
        for row in rows or [{"status": "missing"}]:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in writer.fieldnames or []})


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Low-Churn MLP Sparse-Factorization Vector Capture",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Claim status: `{summary['claim_status']}`",
            f"- Selected action: `{summary['selected_next_action']}`",
            f"- Raw teacher vector rows: `{summary['raw_teacher_vector_row_count']}`",
            f"- Logit intervention rows: `{summary['logit_intervention_row_count']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            "",
            "This local artifact captures raw low-churn teacher residual vectors and logit deltas. It does not train sparse value dictionaries or establish a sparse-factorization ceiling.",
            "",
        ]
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-harness-dir", type=Path, default=DEFAULT_TRAINING_HARNESS_DIR)
    parser.add_argument("--pregate-dir", type=Path, default=DEFAULT_PREGATE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    summary = run_low_churn_mlp_sparse_factorization_vector_capture(
        training_harness_dir=args.training_harness_dir,
        pregate_dir=args.pregate_dir,
        out_dir=args.out,
        train_steps=args.train_steps,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "selected_next_action": summary["selected_next_action"],
                "raw_teacher_vector_row_count": summary["raw_teacher_vector_row_count"],
                "advance_to_gpu_validation": summary["advance_to_gpu_validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
