"""Scaffold the dense-teacher failure-localization audit contract."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_CLOSEOUT_DIR = Path("results/reports/dense_teacher_columnability_closeout")
DEFAULT_DISTILLATION_DIR = Path(
    "results/audits/token_larger_dense_teacher_residual_distillation_comparison"
)
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_failure_localization")

CONTRACT_RECORDED = "dense_teacher_failure_localization_contract_recorded"
INSUFFICIENT_EVIDENCE = "dense_teacher_failure_localization_contract_failed_closed"
PARTIAL_EVALUATOR_RECORDED = "dense_teacher_failure_localization_partial_evaluator_recorded"
NEXT_STEP = (
    "implement retrained-oracle and gated pair-composer rows for "
    "dense_teacher_failure_localization"
)

REQUIRED_ARMS = (
    "learned_support_sparse_student",
    "oracle_support_trained_values",
    "retrained_oracle_support_values",
    "oracle_support_gated_value_pair_composer",
    "dense_teacher",
    "dense_rank_norm_control",
    "random_support_null",
    "fixed_support_null",
    "token_position_router_null",
    "shuffled_teacher_target_null",
)

REQUIRED_TENSORS = (
    {
        "tensor": "inputs",
        "filename": "inputs.pt",
        "required_for": "fixed validation batch identity and token/position nulls",
    },
    {
        "tensor": "targets",
        "filename": "targets.pt",
        "required_for": "CE guardrail and shuffled-target null checks",
    },
    {
        "tensor": "base_hidden",
        "filename": "base_hidden.pt",
        "required_for": "oracle/retrained sparse value evaluation against frozen hidden states",
    },
    {
        "tensor": "base_logits",
        "filename": "base_logits.pt",
        "required_for": "teacher logit residual and anchor/off-target leakage metrics",
    },
    {
        "tensor": "teacher_logits",
        "filename": "teacher_logits.pt",
        "required_for": "dense-teacher CE, logit residual, and shuffled-target controls",
    },
    {
        "tensor": "teacher_hidden_residual",
        "filename": "teacher_hidden_residual.pt",
        "required_for": "hidden residual MSE/cosine/R2 metrics",
    },
    {
        "tensor": "teacher_logit_residual",
        "filename": "teacher_logit_residual.pt",
        "required_for": "logit residual MSE/cosine/R2 metrics",
    },
    {
        "tensor": "learned_support_indices",
        "filename": "learned_support_indices.pt",
        "required_for": "learned-support baseline and oracle-support regret",
    },
    {
        "tensor": "learned_support_scores",
        "filename": "learned_support_scores.pt",
        "required_for": "router margin, support confidence, and token-position null calibration",
    },
    {
        "tensor": "per_column_hidden_contributions",
        "filename": "per_column_hidden_contributions.pt",
        "required_for": "exhaustive oracle support over trained values",
    },
    {
        "tensor": "per_column_logit_contributions",
        "filename": "per_column_logit_contributions.pt",
        "required_for": "oracle support logit residual and CE evaluation",
    },
    {
        "tensor": "sparse_column_value_state",
        "filename": "sparse_column_value_state.pt",
        "required_for": "retrained-oracle and gated pair-composer initialization/accounting",
    },
)

METRIC_FIELDS = (
    "arm",
    "arm_family",
    "availability",
    "support_source",
    "value_source",
    "composer",
    "train_steps",
    "active_params",
    "stored_params",
    "flops_proxy",
    "ce_loss",
    "teacher_hidden_residual_mse",
    "teacher_hidden_residual_r2",
    "teacher_hidden_residual_cosine",
    "teacher_logit_residual_mse",
    "teacher_logit_residual_r2",
    "teacher_logit_residual_cosine",
    "oracle_support_regret",
    "functional_churn",
    "anchor_kl",
    "offtarget_logit_leakage",
    "residual_norm_ratio",
    "residual_direction_error",
    "pair_synergy",
    "passes_no_gpu_pregate",
    "notes",
)


def run_dense_teacher_failure_localization_contract(
    *,
    closeout_dir: Path = DEFAULT_CLOSEOUT_DIR,
    distillation_dir: Path = DEFAULT_DISTILLATION_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed contract for the next local localization evaluator."""

    start = time.time()
    closeout_path = closeout_dir / "summary.json"
    distillation_path = distillation_dir / "summary.json"
    closeout = _read_json(closeout_path)
    distillation = _read_json(distillation_path)
    tensor_inventory = _tensor_inventory(distillation_dir)
    source_rows = [
        _source_row("dense_teacher_columnability_closeout", closeout_path, closeout),
        _source_row("dense_teacher_distillation_comparison", distillation_path, distillation),
    ]
    source_rows.extend(
        _tensor_source_row(row["tensor"], Path(row["path"]), bool(row["present"]))
        for row in tensor_inventory
    )
    contract_rows = _contract_rows()
    pregate = _pregate_rows(closeout, distillation, source_rows, tensor_inventory)
    failures = [row for row in pregate if not row["passed"]]
    status = "fail" if failures else "pass"
    decision = INSUFFICIENT_EVIDENCE if failures else CONTRACT_RECORDED
    next_command = (
        "./.venv-conda/bin/python -m relaleap.experiments.dense_teacher_failure_localization"
    )
    summary = {
        "status": status,
        "decision": decision,
        "claim_status": "contract_only_no_scientific_localization_claim",
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "selected_next_step": NEXT_STEP,
        "next_command": next_command,
        "source_rows": source_rows,
        "tensor_inventory": tensor_inventory,
        "pregate_rows": pregate,
        "contract_rows": contract_rows,
        "required_arms": list(REQUIRED_ARMS),
        "metric_fields": list(METRIC_FIELDS),
        "failures": failures,
        "rationale": (
            "This artifact records the local failure-localization audit contract selected by the "
            "dense-teacher closeout. It intentionally makes no new columnability claim. The next "
            "evaluator must separate support prediction, value representability, and pair/value "
            "composition using oracle-support, retrained-oracle, pair-composer, dense/rank/norm, "
            "and null rows before any GPU validation."
        ),
        "backend_policy": "local CPU contract only; RunPod and Colab are not used.",
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "tensor_inventory_csv": str(out_dir / "tensor_inventory.csv"),
            "pregate_rows_csv": str(out_dir / "pregate_rows.csv"),
            "contract_rows_csv": str(out_dir / "contract_rows.csv"),
            "evaluator_rows_csv": str(out_dir / "evaluator_rows.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    if not failures:
        evaluator_rows = _evaluator_rows(distillation_dir, distillation)
        summary["evaluator_rows"] = evaluator_rows
        summary["filled_evaluator_arms"] = [
            row["arm"] for row in evaluator_rows if row.get("availability") == "filled"
        ]
        summary["pending_evaluator_arms"] = [
            arm for arm in REQUIRED_ARMS if arm not in set(summary["filled_evaluator_arms"])
        ]
        summary["decision"] = PARTIAL_EVALUATOR_RECORDED
        summary["claim_status"] = "partial_local_evaluator_no_columnability_claim"
        summary["selected_next_step"] = NEXT_STEP
        summary["rationale"] = (
            "This artifact now fills learned-support and exhaustive oracle-support-trained-values "
            "rows from exported dense-teacher tensors, and records dense/rank/norm plus null "
            "rows from the source distillation summary. It still makes no columnability claim "
            "because retrained-oracle and pair-composer rows remain pending."
        )
    else:
        summary["evaluator_rows"] = []
        summary["filled_evaluator_arms"] = []
        summary["pending_evaluator_arms"] = list(REQUIRED_ARMS)
    _write_artifacts(out_dir, summary)
    return summary


def _evaluator_rows(distillation_dir: Path, distillation: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - environment dependent
        return [
            _metric_row(
                arm="runtime_import_failure",
                arm_family="runtime",
                availability="failed",
                notes=f"torch import failed: {exc}",
            )
        ]

    tensors = _load_evaluator_tensors(torch, distillation_dir)
    targets = tensors["targets"]
    base_logits = tensors["base_logits"]
    teacher_logits = tensors["teacher_logits"]
    teacher_hidden_residual = tensors["teacher_hidden_residual"]
    teacher_logit_residual = tensors["teacher_logit_residual"]
    learned_support = tensors["learned_support_indices"].to(dtype=torch.long)
    learned_scores = tensors["learned_support_scores"]
    per_column_hidden = tensors["per_column_hidden_contributions"]
    per_column_logits = tensors["per_column_logit_contributions"]
    sparse_state = tensors["sparse_column_value_state"]
    top_k = int(sparse_state.get("top_k", learned_support.shape[-1]))
    num_columns = int(sparse_state.get("num_columns", per_column_hidden.shape[2]))
    atoms_per_column = int(sparse_state.get("atoms_per_column", 1))
    stored_params = _state_param_count(sparse_state)
    active_params = float(top_k * atoms_per_column * teacher_hidden_residual.shape[-1])
    teacher_ce = _safe_number(distillation.get("dense_teacher_ce_loss"))

    learned_weights = torch.softmax(learned_scores, dim=-1)
    learned_hidden_update = _compose_selected(per_column_hidden, learned_support, learned_weights)
    learned_logit_update = _compose_selected(per_column_logits, learned_support, learned_weights)
    learned_logits = base_logits + learned_logit_update

    oracle_support, oracle_hidden_update, oracle_logit_update = _oracle_support_trained_values(
        torch,
        per_column_hidden,
        per_column_logits,
        teacher_logit_residual,
        top_k=top_k,
    )
    oracle_logits = base_logits + oracle_logit_update
    learned_logit_mse = float(F.mse_loss(learned_logit_update, teacher_logit_residual).item())
    oracle_logit_mse = float(F.mse_loss(oracle_logit_update, teacher_logit_residual).item())
    learned_minus_oracle_logit_mse = learned_logit_mse - oracle_logit_mse

    learned_row = _metrics_from_prediction(
        torch,
        F,
        arm="learned_support_sparse_student",
        arm_family="baseline",
        availability="filled",
        support_source="learned contextual top-k support",
        value_source="trained sparse values",
        composer="router-score softmax weighted independent column sum",
        logits=learned_logits,
        hidden_update=learned_hidden_update,
        targets=targets,
        teacher_logits=teacher_logits,
        teacher_hidden_residual=teacher_hidden_residual,
        teacher_logit_residual=teacher_logit_residual,
        base_logits=base_logits,
        teacher_ce=teacher_ce,
        stored_params=stored_params,
        active_params=active_params,
        flops_proxy=float(base_logits.numel() * max(top_k, 1)),
        oracle_support_regret=learned_minus_oracle_logit_mse,
        notes="filled from exported learned support and per-column trained values",
    )
    oracle_row = _metrics_from_prediction(
        torch,
        F,
        arm="oracle_support_trained_values",
        arm_family="oracle_support",
        availability="filled",
        support_source=f"exhaustive per-token best {top_k}-of-{num_columns} support",
        value_source="trained sparse values",
        composer="per-token least-squares weighted independent column sum",
        logits=oracle_logits,
        hidden_update=oracle_hidden_update,
        targets=targets,
        teacher_logits=teacher_logits,
        teacher_hidden_residual=teacher_hidden_residual,
        teacher_logit_residual=teacher_logit_residual,
        base_logits=base_logits,
        teacher_ce=teacher_ce,
        stored_params=stored_params,
        active_params=active_params,
        flops_proxy=float(base_logits.numel() * max(num_columns, 1)),
        oracle_support_regret=0.0,
        notes=(
            "filled by exhaustive trained-value support plus oracle least-squares support weights; "
            "CE/logit metrics use exported single-column logit deltas because the frozen decoder is not exported"
        ),
    )
    oracle_row["oracle_selected_support_sets"] = _unique_support_count(torch, oracle_support)
    oracle_row["learned_minus_oracle_logit_mse"] = learned_minus_oracle_logit_mse
    oracle_row["learned_minus_oracle_ce"] = learned_row["ce_loss"] - oracle_row["ce_loss"]
    source_rows = _source_summary_evaluator_rows(distillation)
    return [learned_row, oracle_row, *source_rows]


def _source_summary_evaluator_rows(distillation: dict[str, Any]) -> list[dict[str, Any]]:
    variant_rows = distillation.get("variant_rows") if isinstance(distillation.get("variant_rows"), list) else []
    mappings = (
        (
            "dense_teacher",
            ("parameter_matched_causal_mlp_control", "dense_teacher_parameter_matched_mlp"),
            "upper_bound",
            "dense",
            "dense teacher residual",
            "dense residual adapter",
        ),
        (
            "dense_rank_norm_control",
            ("dense_rank_norm_control",),
            "control",
            "dense/rank/norm matched",
            "matched dense or low-rank residual",
            "matched control",
        ),
        (
            "random_support_null",
            ("random_support_topk2",),
            "null",
            "random top-k support",
            "trained sparse values",
            "independent column sum",
        ),
        (
            "fixed_support_null",
            ("fixed_support_topk2",),
            "null",
            "fixed top-k support",
            "trained sparse values",
            "independent column sum",
        ),
        (
            "token_position_router_null",
            ("token_position_only_router_topk2",),
            "null",
            "token/position-only support predictor",
            "trained sparse values",
            "independent column sum",
        ),
        (
            "shuffled_teacher_target_null",
            ("shuffled_teacher_target_topk2",),
            "null",
            "shuffled teacher target support",
            "shuffled residual/logit target",
            "independent column sum",
        ),
    )
    rows: list[dict[str, Any]] = []
    for arm, source_arms, family, support_source, value_source, composer in mappings:
        source = _find_variant_row(variant_rows, source_arms, teacher_scale=1.0)
        if source is None:
            rows.append(
                _metric_row(
                    arm=arm,
                    arm_family=family,
                    availability="missing_source_summary_row",
                    support_source=support_source,
                    value_source=value_source,
                    composer=composer,
                    notes=f"missing source distillation row among {', '.join(source_arms)}",
                )
            )
            continue
        rows.append(
            _metric_row_from_variant(
                arm=arm,
                arm_family=family,
                source=source,
                support_source=support_source,
                value_source=value_source,
                composer=composer,
            )
        )
    return rows


def _find_variant_row(
    variant_rows: list[Any],
    source_arms: tuple[str, ...],
    *,
    teacher_scale: float,
) -> dict[str, Any] | None:
    for row in variant_rows:
        if not isinstance(row, dict):
            continue
        if row.get("arm") not in source_arms:
            continue
        if abs(float(row.get("teacher_scale", 1.0)) - teacher_scale) <= 1e-12:
            return row
    return None


def _metric_row_from_variant(
    *,
    arm: str,
    arm_family: str,
    source: dict[str, Any],
    support_source: str,
    value_source: str,
    composer: str,
) -> dict[str, Any]:
    teacher_logit_mse = _safe_number(source.get("teacher_logit_mse"))
    if teacher_logit_mse is None and _is_exact_teacher_control(source):
        teacher_logit_mse = 0.0
    return _metric_row(
        arm=arm,
        arm_family=arm_family,
        availability="filled",
        support_source=support_source,
        value_source=value_source,
        composer=composer,
        train_steps=0,
        active_params=_safe_number(source.get("active_params")),
        stored_params=_safe_number(source.get("stored_params")),
        flops_proxy=_safe_number(source.get("flops_estimate")),
        ce_loss=_safe_number(source.get("ce_loss")),
        teacher_hidden_residual_mse=_safe_number(source.get("teacher_residual_mse")),
        teacher_hidden_residual_r2=_safe_number(source.get("teacher_residual_r2")),
        teacher_hidden_residual_cosine=_safe_number(source.get("teacher_residual_cosine")),
        teacher_logit_residual_mse=teacher_logit_mse,
        teacher_logit_residual_r2="not_reported_in_source_summary",
        teacher_logit_residual_cosine="not_reported_in_source_summary",
        oracle_support_regret=_safe_number(source.get("support_regret")),
        functional_churn=_safe_number(source.get("functional_churn")),
        anchor_kl=_safe_number(source.get("anchor_kl_or_logit_mse")),
        offtarget_logit_leakage=max(
            0.0,
            (_safe_number(source.get("ce_loss")) or 0.0)
            - (_safe_number(source.get("teacher_ce_loss")) or 0.0),
        ),
        residual_norm_ratio=_safe_number(source.get("residual_norm_ratio")),
        residual_direction_error=(
            1.0 - float(source["teacher_residual_cosine"])
            if source.get("teacher_residual_cosine") is not None
            else ""
        ),
        pair_synergy="not_measured",
        passes_no_gpu_pregate=False,
        notes=f"filled from source distillation summary arm {source.get('arm')}",
    )


def _is_exact_teacher_control(source: dict[str, Any]) -> bool:
    ce_loss = _safe_number(source.get("ce_loss"))
    teacher_ce = _safe_number(source.get("teacher_ce_loss"))
    support_regret = _safe_number(source.get("support_regret"))
    hidden_mse = _safe_number(source.get("teacher_residual_mse"))
    return (
        ce_loss is not None
        and teacher_ce is not None
        and abs(ce_loss - teacher_ce) <= 1e-12
        and support_regret is not None
        and abs(support_regret) <= 1e-12
        and hidden_mse is not None
        and abs(hidden_mse) <= 1e-12
    )


def _load_evaluator_tensors(torch: Any, distillation_dir: Path) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    for spec in REQUIRED_TENSORS:
        path = distillation_dir / str(spec["filename"])
        tensors[str(spec["tensor"])] = torch.load(path, map_location="cpu")
    return tensors


def _oracle_support_trained_values(
    torch: Any,
    per_column_hidden: Any,
    per_column_logits: Any,
    teacher_logit_residual: Any,
    *,
    top_k: int,
) -> tuple[Any, Any, Any]:
    num_columns = int(per_column_logits.shape[2])
    combos = list(itertools.combinations(range(num_columns), top_k))
    if not combos:
        raise ValueError("top_k must produce at least one oracle support combination")
    combo_tensor = torch.as_tensor(combos, dtype=torch.long)
    batch, seq_len, _, vocab_size = per_column_logits.shape
    token_count = batch * seq_len
    logit_columns = per_column_logits.reshape(token_count, num_columns, vocab_size)
    hidden_columns = per_column_hidden.reshape(token_count, num_columns, per_column_hidden.shape[-1])
    target = teacher_logit_residual.reshape(token_count, vocab_size)
    combo_updates: list[Any] = []
    combo_hidden_updates: list[Any] = []
    combo_mses: list[Any] = []
    for combo in combos:
        chosen_logits = logit_columns[:, list(combo), :].transpose(1, 2)
        solution = torch.linalg.lstsq(chosen_logits, target.unsqueeze(-1)).solution.squeeze(-1)
        logit_update = torch.einsum("tk,tkv->tv", solution, logit_columns[:, list(combo), :])
        hidden_update = torch.einsum("tk,tkh->th", solution, hidden_columns[:, list(combo), :])
        combo_updates.append(logit_update)
        combo_hidden_updates.append(hidden_update)
        combo_mses.append(((logit_update - target) ** 2).mean(dim=-1))
    all_updates = torch.stack(combo_updates, dim=1)
    all_hidden_updates = torch.stack(combo_hidden_updates, dim=1)
    all_mses = torch.stack(combo_mses, dim=1)
    best = all_mses.argmin(dim=-1)
    oracle_support = combo_tensor[best].reshape(batch, seq_len, top_k)
    update_index = best.view(token_count, 1, 1).expand(-1, 1, vocab_size)
    oracle_logit_update = all_updates.gather(dim=1, index=update_index).squeeze(1)
    hidden_index = best.view(token_count, 1, 1).expand(-1, 1, per_column_hidden.shape[-1])
    oracle_hidden_update = all_hidden_updates.gather(dim=1, index=hidden_index).squeeze(1)
    return (
        oracle_support,
        oracle_hidden_update.reshape(batch, seq_len, per_column_hidden.shape[-1]),
        oracle_logit_update.reshape(batch, seq_len, vocab_size),
    )


def _compose_selected(per_column: Any, support: Any, weights: Any) -> Any:
    gather_index = support.unsqueeze(-1).expand(*support.shape, per_column.shape[-1])
    selected = per_column.gather(dim=2, index=gather_index)
    return (selected * weights.unsqueeze(-1)).sum(dim=2)


def _metrics_from_prediction(
    torch: Any,
    F: Any,
    *,
    arm: str,
    arm_family: str,
    availability: str,
    support_source: str,
    value_source: str,
    composer: str,
    logits: Any,
    hidden_update: Any,
    targets: Any,
    teacher_logits: Any,
    teacher_hidden_residual: Any,
    teacher_logit_residual: Any,
    base_logits: Any,
    teacher_ce: float | None,
    stored_params: float,
    active_params: float,
    flops_proxy: float,
    oracle_support_regret: float,
    notes: str,
) -> dict[str, Any]:
    vocab_size = int(logits.shape[-1])
    logit_update = logits - base_logits
    teacher_ce_loss = teacher_ce if teacher_ce is not None else _ce_loss_value(F, teacher_logits, targets, vocab_size)
    ce_loss = _ce_loss_value(F, logits, targets, vocab_size)
    hidden_mse = float(F.mse_loss(hidden_update, teacher_hidden_residual).item())
    logit_mse = float(F.mse_loss(logit_update, teacher_logit_residual).item())
    residual_l2 = float(hidden_update.norm(dim=-1).mean().item())
    teacher_l2 = float(teacher_hidden_residual.norm(dim=-1).mean().item())
    return _metric_row(
        arm=arm,
        arm_family=arm_family,
        availability=availability,
        support_source=support_source,
        value_source=value_source,
        composer=composer,
        train_steps=0,
        active_params=active_params,
        stored_params=stored_params,
        flops_proxy=flops_proxy,
        ce_loss=ce_loss,
        teacher_hidden_residual_mse=hidden_mse,
        teacher_hidden_residual_r2=_r2(torch, hidden_update, teacher_hidden_residual),
        teacher_hidden_residual_cosine=_cosine(F, hidden_update, teacher_hidden_residual),
        teacher_logit_residual_mse=logit_mse,
        teacher_logit_residual_r2=_r2(torch, logit_update, teacher_logit_residual),
        teacher_logit_residual_cosine=_cosine(F, logit_update, teacher_logit_residual),
        oracle_support_regret=oracle_support_regret,
        functional_churn=float((logits.argmax(dim=-1) != teacher_logits.argmax(dim=-1)).float().mean().item()),
        anchor_kl=float(F.mse_loss(logits, base_logits).item()),
        offtarget_logit_leakage=max(0.0, ce_loss - teacher_ce_loss),
        residual_norm_ratio=residual_l2 / max(teacher_l2, 1e-12),
        residual_direction_error=1.0 - _cosine(F, hidden_update, teacher_hidden_residual),
        pair_synergy="not_measured",
        passes_no_gpu_pregate=False,
        notes=notes,
    )


def _metric_row(**values: Any) -> dict[str, Any]:
    row = {field: "" for field in METRIC_FIELDS}
    row.update(values)
    return row


def _ce_loss_value(F: Any, logits: Any, targets: Any, vocab_size: int) -> float:
    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab_size), targets[:, :-1].reshape(-1))
    return float(loss.detach().item())


def _r2(torch: Any, prediction: Any, target: Any) -> float:
    residual = ((prediction - target) ** 2).sum()
    centered = ((target - target.mean()) ** 2).sum()
    if float(centered.item()) <= 1e-12:
        return 1.0 if float(residual.item()) <= 1e-12 else 0.0
    return float((1.0 - residual / centered).item())


def _cosine(F: Any, prediction: Any, target: Any) -> float:
    return float(F.cosine_similarity(prediction.reshape(1, -1), target.reshape(1, -1), dim=-1).item())


def _state_param_count(state: dict[str, Any]) -> float:
    total = 0
    for key in ("atom_logits", "atom_values"):
        value = state.get(key)
        if hasattr(value, "numel"):
            total += int(value.numel())
    return float(total)


def _safe_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unique_support_count(torch: Any, support: Any) -> int:
    flat = support.reshape(-1, support.shape[-1])
    return int(torch.unique(flat, dim=0).shape[0])


def _contract_rows() -> list[dict[str, Any]]:
    return [
        _contract(
            "learned_support_sparse_student",
            "baseline",
            "available_from_distillation_summary",
            "learned contextual/ACSR router support",
            "trained sparse values",
            "independent column sum",
            "anchors the current failed sparse branch",
        ),
        _contract(
            "oracle_support_trained_values",
            "oracle_support",
            "required_pending",
            "exhaustive per-token best top-k support",
            "trained sparse values",
            "independent column sum",
            "tests support-prediction regret while holding values fixed",
        ),
        _contract(
            "retrained_oracle_support_values",
            "oracle_value",
            "required_pending",
            "oracle support fixed during training/eval",
            "values retrained under oracle support",
            "independent column sum",
            "tests value representability after removing router error",
        ),
        _contract(
            "oracle_support_gated_value_pair_composer",
            "composition",
            "required_pending",
            "oracle support",
            "support-conditioned gates plus pair interaction",
            "gated values plus pair composer",
            "tests whether independent column summation is the bottleneck",
        ),
        _contract(
            "dense_teacher",
            "upper_bound",
            "available_from_distillation_summary",
            "dense",
            "dense teacher residual",
            "dense residual adapter",
            "records the target gap and guards against false sparse wins",
        ),
        _contract(
            "dense_rank_norm_control",
            "control",
            "available_from_distillation_summary",
            "dense/rank/norm matched",
            "matched dense or low-rank residual",
            "matched control",
            "prevents pair-composer wins from merely being extra dense capacity",
        ),
        _contract(
            "random_support_null",
            "null",
            "available_from_distillation_summary",
            "random top-k support",
            "trained sparse values",
            "independent column sum",
            "tests support specificity against utilization-matched random routing",
        ),
        _contract(
            "fixed_support_null",
            "null",
            "available_from_distillation_summary",
            "fixed top-k support",
            "trained sparse values",
            "independent column sum",
            "tests whether routing variation matters at all",
        ),
        _contract(
            "token_position_router_null",
            "null",
            "available_from_distillation_summary",
            "token/position-only support predictor",
            "trained sparse values",
            "independent column sum",
            "tests deployable support signal beyond token/position shortcuts",
        ),
        _contract(
            "shuffled_teacher_target_null",
            "null",
            "available_from_distillation_summary",
            "shuffled teacher target support",
            "shuffled residual/logit target",
            "independent column sum",
            "tests target-label leakage and noncausal teacher alignment",
        ),
    ]


def _contract(
    arm: str,
    arm_family: str,
    availability: str,
    support_source: str,
    value_source: str,
    composer: str,
    notes: str,
) -> dict[str, Any]:
    row = {field: "" for field in METRIC_FIELDS}
    row.update(
        {
            "arm": arm,
            "arm_family": arm_family,
            "availability": availability,
            "support_source": support_source,
            "value_source": value_source,
            "composer": composer,
            "passes_no_gpu_pregate": "pending",
            "notes": notes,
        }
    )
    return row


def _pregate_rows(
    closeout: dict[str, Any],
    distillation: dict[str, Any],
    source_rows: list[dict[str, Any]],
    tensor_inventory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    present = {row["source"]: row["present"] for row in source_rows}
    tensor_present = {row["tensor"]: bool(row["present"]) for row in tensor_inventory}
    variant_rows = distillation.get("variant_rows") if isinstance(distillation.get("variant_rows"), list) else []
    arms = {row.get("arm") for row in variant_rows if isinstance(row, dict)}
    return [
        _pregate(
            "closeout_branch_retired",
            closeout.get("decision") == "dense_teacher_sparse_columnability_branch_closed_for_failure_localization",
            "dense-teacher closeout must have retired the rescue branch",
            closeout.get("decision"),
        ),
        _pregate(
            "distillation_failed_closed",
            distillation.get("status") == "fail",
            "source distillation comparison must be negative evidence, not a promotion",
            distillation.get("status"),
        ),
        _pregate(
            "teacher_residual_tensors_present",
            bool(tensor_present.get("teacher_hidden_residual"))
            and bool(tensor_present.get("teacher_logit_residual")),
            "hidden and logit residual tensors are needed for residual-direction metrics",
            {
                "hidden": tensor_present.get("teacher_hidden_residual"),
                "logit": tensor_present.get("teacher_logit_residual"),
            },
        ),
        _pregate(
            "per_column_evaluator_tensors_present",
            all(bool(row["present"]) for row in tensor_inventory),
            "all required evaluator tensors must be exported before oracle/retrained/composer rows can run",
            {
                row["tensor"]: row["status"]
                for row in tensor_inventory
                if not row["present"]
            },
        ),
        _pregate(
            "baseline_and_null_rows_available",
            {
                "promoted_contextual_topk2_ce_mse_distill",
                "random_support_topk2",
                "fixed_support_topk2",
                "token_position_only_router_topk2",
                "shuffled_teacher_target_topk2",
            }.issubset(arms),
            "existing distillation artifact must expose baseline and null arms",
            sorted(arms),
        ),
        _pregate(
            "retrained_oracle_and_composer_rows_pending",
            True,
            "this evaluator records, but does not yet execute, retrained-oracle and composer rows",
            "pending evaluator implementation",
        ),
    ]


def _pregate(criterion: str, passed: bool, threshold: str, actual: Any) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": actual,
        "failure_reason": "" if passed else "required source or contract condition missing",
    }


def _source_row(source: str, path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": packet.get("status", ""),
        "decision": packet.get("decision", ""),
        "claim_status": packet.get("claim_status", ""),
        "git_commit": packet.get("git_commit", ""),
    }


def _tensor_inventory(distillation_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for spec in REQUIRED_TENSORS:
        path = distillation_dir / str(spec["filename"])
        present = path.is_file()
        rows.append(
            {
                "tensor": spec["tensor"],
                "path": str(path),
                "required_for": spec["required_for"],
                "present": present,
                "file_size_bytes": path.stat().st_size if present else 0,
                "status": "present" if present else "missing_required_export",
            }
        )
    return rows


def _tensor_source_row(source: str, path: Path, present: bool) -> dict[str, Any]:
    return {
        "source": f"{source}_tensor",
        "path": str(path),
        "present": present,
        "status": "present" if present else "missing",
        "decision": "",
        "claim_status": "dense_teacher_failure_localization_tensor_source",
        "git_commit": "",
    }


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "tensor_inventory.csv", summary["tensor_inventory"])
    _write_csv(out_dir / "pregate_rows.csv", summary["pregate_rows"])
    _write_csv(out_dir / "contract_rows.csv", summary["contract_rows"])
    _write_csv(out_dir / "evaluator_rows.csv", summary["evaluator_rows"])
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Dense Teacher Failure Localization Contract",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Promotion allowed: `{summary['promotion_allowed']}`",
        "",
        "## Required Arms",
        "",
    ]
    lines.extend(f"- `{arm}`" for arm in summary["required_arms"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(summary["rationale"]),
            "",
            "## Next Step",
            "",
            str(summary["selected_next_step"]),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closeout-dir", type=Path, default=DEFAULT_CLOSEOUT_DIR)
    parser.add_argument("--distillation-dir", type=Path, default=DEFAULT_DISTILLATION_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_dense_teacher_failure_localization_contract(
        closeout_dir=args.closeout_dir,
        distillation_dir=args.distillation_dir,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
