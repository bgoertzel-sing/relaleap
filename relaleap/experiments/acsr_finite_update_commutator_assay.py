"""Finite-update commutator assay for ACSR sparse and dense residual controls."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import time
from pathlib import Path
from typing import Any

import yaml

from relaleap.experiments.acsr_dense_residual_transfer_control import (
    _LowRankCausalDenseAdapter,
    _criterion,
    _float_or_none,
    _parameter_count,
    _per_token_ce,
)
from relaleap.experiments.acsr_transfer_objective_probe import DEFAULT_CONFIG
from relaleap.experiments.anticipatory_contextual_support_routing import (
    _causal_predictor_inputs,
    _contextual_chunks,
    _position_predictor_inputs,
)


DEFAULT_OUT_DIR = Path("results/reports/acsr_finite_update_commutator_assay")
REQUIRED_ARTIFACTS = (
    "summary.json",
    "variant_commutator.csv",
    "per_token_commutator.csv",
    "commutator_strata.csv",
    "gate_criteria.csv",
    "notes.md",
)
REQUIRED_VARIANTS = (
    "sparse_acsr_contextual_topk2",
    "sparse_acsr_rank_matched_topk1",
    "dense_causal_rank1",
    "dense_token_position_rank1",
)
REQUIRED_PER_TOKEN_FIELDS = {
    "variant",
    "family",
    "split",
    "batch_index",
    "position_index",
    "target_token",
    "forward_ce",
    "reverse_ce",
    "ce_delta_forward_minus_reverse",
    "ce_abs_delta",
    "symmetric_kl",
    "logit_mse",
    "residual_delta_l2",
    "forward_residual_l2",
    "reverse_residual_l2",
    "support_churn",
    "forward_support",
    "reverse_support",
}
SPARSE_VARIANT = "sparse_acsr_contextual_topk2"
DENSE_VARIANT = "dense_causal_rank1"


def run_acsr_finite_update_commutator_assay(
    *,
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path = DEFAULT_OUT_DIR,
    phase_steps: int = 4,
    learning_rate: float | None = None,
    ce_guardrail: float = 0.05,
    material_logit_mse_threshold: float = 0.01,
) -> dict[str, Any]:
    """Train two update orders and compare sparse ACSR to dense controls."""

    start = time.time()
    preflight = [
        _criterion(
            "config_present",
            config_path.is_file(),
            "config exists",
            str(config_path) if config_path.is_file() else "missing",
            "cannot run finite-update assay without config",
        ),
        _criterion(
            "phase_steps_positive",
            phase_steps > 0,
            "phase update steps are positive",
            phase_steps,
            "phase_steps must be positive",
        ),
    ]
    if any(not row["passed"] for row in preflight):
        summary = _summary(
            status="fail",
            decision="acsr_finite_update_commutator_assay_failed_closed",
            start=start,
            config_path=config_path,
            out_dir=out_dir,
            phase_steps=phase_steps,
            variant_rows=[],
            per_token_rows=[],
            strata_rows=[],
            gate_rows=preflight,
            ce_guardrail=ce_guardrail,
            material_logit_mse_threshold=material_logit_mse_threshold,
        )
        _write_artifacts(out_dir, summary, [], [], [], preflight)
        return summary

    try:
        variant_rows, per_token_rows, strata_rows = _run_assay(
            config_path=config_path,
            phase_steps=phase_steps,
            learning_rate=learning_rate,
        )
    except Exception as exc:  # pragma: no cover - depends on torch runtime
        gate_rows = preflight + [
            _criterion(
                "assay_runtime",
                False,
                "finite-update assay completes",
                str(exc),
                "finite-update assay runtime failed",
            )
        ]
        summary = _summary(
            status="fail",
            decision="acsr_finite_update_commutator_assay_failed_closed",
            start=start,
            config_path=config_path,
            out_dir=out_dir,
            phase_steps=phase_steps,
            variant_rows=[],
            per_token_rows=[],
            strata_rows=[],
            gate_rows=gate_rows,
            ce_guardrail=ce_guardrail,
            material_logit_mse_threshold=material_logit_mse_threshold,
        )
        _write_artifacts(out_dir, summary, [], [], [], gate_rows)
        return summary

    gate_rows = preflight + _assay_gate_rows(
        variant_rows,
        per_token_rows,
        ce_guardrail=ce_guardrail,
        material_logit_mse_threshold=material_logit_mse_threshold,
    )
    status = "pass" if all(row["passed"] for row in gate_rows) else "fail"
    sparse = _variant_row(variant_rows, SPARSE_VARIANT)
    dense = _variant_row(variant_rows, DENSE_VARIANT)
    sparse_minus_dense_logit = _delta(
        sparse.get("mean_logit_mse") if sparse else None,
        dense.get("mean_logit_mse") if dense else None,
    )
    failure_criteria = {str(row.get("criterion")) for row in gate_rows if not row["passed"]}
    missing_or_runtime_failure = bool(
        failure_criteria
        & {
            "config_present",
            "phase_steps_positive",
            "assay_runtime",
            "required_variants_present",
            "per_token_commutator_fields_present",
            "dense_control_available",
        }
    )
    if status == "fail" and missing_or_runtime_failure:
        decision = "acsr_finite_update_commutator_assay_failed_gate"
        claim_status = "finite_update_commutator_not_interpretable"
        next_step = "repair missing ACSR finite-update commutator assay controls"
    elif status == "fail":
        decision = "acsr_finite_update_commutator_assay_tiny_commutator"
        claim_status = "finite_update_commutator_too_small_for_sparse_mechanism_claim"
        next_step = (
            "do not promote ACSR on this assay; only rerun with a stronger bounded "
            "finite-update schedule if commutator magnitude becomes the primary question"
        )
    elif sparse_minus_dense_logit is not None and sparse_minus_dense_logit < 0.0:
        decision = "acsr_sparse_commutator_lower_than_dense_control"
        claim_status = "sparse_order_sensitivity_advantage_candidate_not_promoted"
        next_step = (
            "repeat on a second seed or GPU only if this local sparse-minus-dense "
            "commutator signal remains scientifically important"
        )
    else:
        decision = "acsr_sparse_commutator_not_lower_than_dense_control"
        claim_status = "dense_control_not_beaten_by_sparse_finite_update_assay"
        next_step = (
            "keep deployable ACSR support-discovery frozen and use dense residual "
            "controls as the active baseline for retention/churn work"
        )

    summary = _summary(
        status=status,
        decision=decision,
        start=start,
        config_path=config_path,
        out_dir=out_dir,
        phase_steps=phase_steps,
        variant_rows=variant_rows,
        per_token_rows=per_token_rows,
        strata_rows=strata_rows,
        gate_rows=gate_rows,
        ce_guardrail=ce_guardrail,
        material_logit_mse_threshold=material_logit_mse_threshold,
        claim_status=claim_status,
        next_step=next_step,
    )
    _write_artifacts(out_dir, summary, variant_rows, per_token_rows, strata_rows, gate_rows)
    return summary


def _run_assay(
    *,
    config_path: Path,
    phase_steps: int,
    learning_rate: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import torch
    import torch.nn.functional as F

    from relaleap.smoke import ResidualColumns, TinyCharTransformer, _build_batch

    nn = __import__("torch.nn").nn
    config = _read_yaml(config_path)
    run_cfg = _as_dict(config.get("run"))
    data_cfg = _as_dict(config.get("data"))
    model_cfg = _as_dict(config.get("model"))
    base_cfg = _as_dict(model_cfg.get("base"))
    column_cfg = _as_dict(model_cfg.get("columns"))

    seed = int(run_cfg.get("seed", 1))
    lr = float(learning_rate if learning_rate is not None else run_cfg.get("learning_rate", 1e-2))
    dataset = str(data_cfg.get("dataset", "tiny_shakespeare_word"))
    seq_len = int(data_cfg.get("seq_len", 64))
    hidden_dim = int(base_cfg.get("hidden_dim", 96))
    layers = int(base_cfg.get("layers", 2))
    num_columns = int(column_cfg.get("num_columns", 24))
    atoms_per_column = int(column_cfg.get("atoms_per_column", 4))
    contextual_width = int(column_cfg.get("contextual_router_hidden_dim", hidden_dim * 2))

    torch.manual_seed(seed)
    inputs, targets, vocab_size = _build_batch(dataset=dataset, seq_len=seq_len, batch_size=4)
    base = TinyCharTransformer(vocab_size=vocab_size, seq_len=seq_len, hidden_dim=hidden_dim, layers=layers)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base.eval()
    with torch.no_grad():
        hidden = base.encode(inputs)
        chunks = _contextual_chunks(torch, hidden)
        causal_features = _causal_predictor_inputs(torch, chunks)
        position_features = _position_predictor_inputs(torch, chunks)

    variants: list[tuple[str, str, Any, str]] = [
        (
            "sparse_acsr_contextual_topk2",
            "sparse",
            ResidualColumns(
                hidden_dim=hidden_dim,
                num_columns=num_columns,
                atoms_per_column=atoms_per_column,
                top_k=2,
                support_router="contextual_mlp_causal",
                contextual_router_hidden_dim=contextual_width,
            ),
            "hidden",
        ),
        (
            "sparse_acsr_rank_matched_topk1",
            "sparse",
            ResidualColumns(
                hidden_dim=hidden_dim,
                num_columns=num_columns,
                atoms_per_column=atoms_per_column,
                top_k=1,
                support_router="contextual_mlp_causal",
                contextual_router_hidden_dim=contextual_width,
            ),
            "hidden",
        ),
        (
            "dense_causal_rank1",
            "dense",
            _LowRankCausalDenseAdapter(nn, int(causal_features.shape[-1]), hidden_dim, rank=1),
            "causal_features",
        ),
        (
            "dense_token_position_rank1",
            "dense",
            _LowRankCausalDenseAdapter(nn, int(position_features.shape[-1]), hidden_dim, rank=1),
            "position_features",
        ),
    ]
    per_token_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    for variant, family, module, feature_key in variants:
        forward = copy.deepcopy(module)
        reverse = copy.deepcopy(module)
        _train_ordered(
            torch,
            F,
            base,
            forward,
            hidden,
            targets,
            vocab_size,
            features=causal_features if feature_key == "causal_features" else position_features,
            family=family,
            order=("anchor", "transfer"),
            steps=phase_steps,
            learning_rate=lr,
        )
        _train_ordered(
            torch,
            F,
            base,
            reverse,
            hidden,
            targets,
            vocab_size,
            features=causal_features if feature_key == "causal_features" else position_features,
            family=family,
            order=("transfer", "anchor"),
            steps=phase_steps,
            learning_rate=lr,
        )
        rows = _per_token_commutator_rows(
            torch,
            F,
            base,
            forward,
            reverse,
            hidden,
            targets,
            vocab_size,
            family=family,
            variant=variant,
            forward_features=causal_features if feature_key == "causal_features" else position_features,
            reverse_features=causal_features if feature_key == "causal_features" else position_features,
        )
        per_token_rows.extend(rows)
        variant_rows.append(_aggregate_variant(rows, variant=variant, family=family, module=forward))
    return variant_rows, per_token_rows, _strata_rows(per_token_rows)


def _train_ordered(
    torch: Any,
    F: Any,
    base: Any,
    module: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    *,
    features: Any,
    family: str,
    order: tuple[str, str],
    steps: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate)
    for split in order:
        mask = _split_mask(torch, hidden, split)
        for _ in range(max(1, steps)):
            optimizer.zero_grad(set_to_none=True)
            updated = module(hidden) if family == "sparse" else hidden + module(features)
            logits = base.decode(updated)
            per_token = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, vocab_size),
                targets[:, :-1].reshape(-1),
                reduction="none",
            )
            loss = per_token[mask].mean()
            loss.backward()
            optimizer.step()
    _eval_module(module)


def _per_token_commutator_rows(
    torch: Any,
    F: Any,
    base: Any,
    forward: Any,
    reverse: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    *,
    family: str,
    variant: str,
    forward_features: Any,
    reverse_features: Any,
) -> list[dict[str, Any]]:
    with torch.no_grad():
        if family == "sparse":
            forward_hidden, forward_support = forward(hidden, return_support=True)
            reverse_hidden, reverse_support = reverse(hidden, return_support=True)
        else:
            forward_update = forward(forward_features)
            reverse_update = reverse(reverse_features)
            forward_hidden = hidden + forward_update
            reverse_hidden = hidden + reverse_update
            forward_support = reverse_support = None
        forward_logits = base.decode(forward_hidden)
        reverse_logits = base.decode(reverse_hidden)
        forward_ce = _per_token_ce(F, forward_logits, targets, vocab_size)
        reverse_ce = _per_token_ce(F, reverse_logits, targets, vocab_size)
        logit_mse = ((forward_logits[:, :-1, :] - reverse_logits[:, :-1, :]) ** 2).mean(dim=-1)
        forward_log_probs = torch.log_softmax(forward_logits[:, :-1, :], dim=-1)
        reverse_log_probs = torch.log_softmax(reverse_logits[:, :-1, :], dim=-1)
        forward_probs = torch.softmax(forward_logits[:, :-1, :], dim=-1)
        reverse_probs = torch.softmax(reverse_logits[:, :-1, :], dim=-1)
        kl_forward_reverse = (forward_probs * (forward_log_probs - reverse_log_probs)).sum(dim=-1)
        kl_reverse_forward = (reverse_probs * (reverse_log_probs - forward_log_probs)).sum(dim=-1)
        symmetric_kl = 0.5 * (kl_forward_reverse + kl_reverse_forward)
        forward_update = forward_hidden - hidden
        reverse_update = reverse_hidden - hidden
        residual_delta_l2 = (forward_update[:, :-1, :] - reverse_update[:, :-1, :]).norm(dim=-1)
        forward_l2 = forward_update[:, :-1, :].norm(dim=-1)
        reverse_l2 = reverse_update[:, :-1, :].norm(dim=-1)

    rows = []
    seq_minus_one = int(hidden.shape[1] - 1)
    for batch_index in range(int(hidden.shape[0])):
        for position_index in range(seq_minus_one):
            split = "anchor" if position_index < seq_minus_one // 2 else "transfer"
            flat_index = batch_index * seq_minus_one + position_index
            f_support = _support_string(forward_support, batch_index, position_index)
            r_support = _support_string(reverse_support, batch_index, position_index)
            rows.append(
                {
                    "variant": variant,
                    "family": family,
                    "split": split,
                    "batch_index": batch_index,
                    "position_index": position_index,
                    "target_token": int(targets[batch_index, position_index].item()),
                    "forward_ce": float(forward_ce[flat_index].item()),
                    "reverse_ce": float(reverse_ce[flat_index].item()),
                    "ce_delta_forward_minus_reverse": float(
                        forward_ce[flat_index].item() - reverse_ce[flat_index].item()
                    ),
                    "ce_abs_delta": abs(float(forward_ce[flat_index].item() - reverse_ce[flat_index].item())),
                    "symmetric_kl": float(symmetric_kl[batch_index, position_index].item()),
                    "logit_mse": float(logit_mse[batch_index, position_index].item()),
                    "residual_delta_l2": float(residual_delta_l2[batch_index, position_index].item()),
                    "forward_residual_l2": float(forward_l2[batch_index, position_index].item()),
                    "reverse_residual_l2": float(reverse_l2[batch_index, position_index].item()),
                    "support_churn": "" if family == "dense" else str(f_support != r_support),
                    "forward_support": f_support,
                    "reverse_support": r_support,
                }
            )
    return rows


def _aggregate_variant(
    rows: list[dict[str, Any]],
    *,
    variant: str,
    family: str,
    module: Any,
) -> dict[str, Any]:
    row = _aggregate_numeric(rows)
    row.update(
        {
            "variant": variant,
            "family": family,
            "row_count": len(rows),
            "parameter_count": _parameter_count(module),
            "support_churn_fraction": _true_fraction(rows, "support_churn"),
            "unique_forward_support_count": len(
                {row.get("forward_support") for row in rows if row.get("forward_support")}
            ),
            "unique_reverse_support_count": len(
                {row.get("reverse_support") for row in rows if row.get("reverse_support")}
            ),
        }
    )
    return row


def _strata_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for stratum in ("family", "variant", "split"):
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row.get(stratum, "")), []).append(row)
        for value, group in sorted(groups.items()):
            aggregate = _aggregate_numeric(group)
            aggregate.update({"stratum": stratum, "value": value, "row_count": len(group)})
            out.append(aggregate)
    return out


def _assay_gate_rows(
    variant_rows: list[dict[str, Any]],
    per_token_rows: list[dict[str, Any]],
    *,
    ce_guardrail: float,
    material_logit_mse_threshold: float,
) -> list[dict[str, Any]]:
    variants_present = {str(row.get("variant")) for row in variant_rows}
    missing_variants = [variant for variant in REQUIRED_VARIANTS if variant not in variants_present]
    fieldnames = set().union(*(set(row) for row in per_token_rows)) if per_token_rows else set()
    missing_fields = sorted(REQUIRED_PER_TOKEN_FIELDS - fieldnames)
    sparse = _variant_row(variant_rows, SPARSE_VARIANT)
    dense = _variant_row(variant_rows, DENSE_VARIANT)
    sparse_logit = sparse.get("mean_logit_mse") if sparse else None
    dense_logit = dense.get("mean_logit_mse") if dense else None
    sparse_ce = sparse.get("mean_ce_abs_delta") if sparse else None
    return [
        _criterion(
            "required_variants_present",
            not missing_variants,
            list(REQUIRED_VARIANTS),
            sorted(variants_present),
            f"missing variants: {missing_variants}",
        ),
        _criterion(
            "per_token_commutator_fields_present",
            not missing_fields,
            sorted(REQUIRED_PER_TOKEN_FIELDS),
            sorted(fieldnames),
            f"missing fields: {missing_fields}",
        ),
        _criterion(
            "sparse_ce_order_guardrail",
            _at_most(sparse_ce, ce_guardrail),
            f"mean sparse absolute CE order delta <= {ce_guardrail}",
            sparse_ce,
            "sparse order sensitivity has material CE harm",
        ),
        _criterion(
            "sparse_logit_commutator_material",
            _at_least(sparse_logit, material_logit_mse_threshold),
            f"mean sparse logit MSE >= {material_logit_mse_threshold}",
            sparse_logit,
            "sparse commutator is too small to interpret",
        ),
        _criterion(
            "dense_control_available",
            dense is not None and dense_logit is not None,
            "dense causal commutator row exists",
            dense_logit,
            "dense causal control is missing",
        ),
    ]


def _summary(
    *,
    status: str,
    decision: str,
    start: float,
    config_path: Path,
    out_dir: Path,
    phase_steps: int,
    variant_rows: list[dict[str, Any]],
    per_token_rows: list[dict[str, Any]],
    strata_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    ce_guardrail: float,
    material_logit_mse_threshold: float,
    claim_status: str | None = None,
    next_step: str | None = None,
) -> dict[str, Any]:
    metrics = _summary_metrics(variant_rows)
    failures = [row for row in gate_rows if not row["passed"]]
    return {
        "status": status,
        "decision": decision,
        "claim_status": claim_status or "finite_update_commutator_not_interpretable",
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "phase_steps": phase_steps,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "thresholds": {
            "ce_guardrail": ce_guardrail,
            "material_logit_mse_threshold": material_logit_mse_threshold,
        },
        "variant_count": len(variant_rows),
        "per_token_row_count": len(per_token_rows),
        "strata_row_count": len(strata_rows),
        "metrics": metrics,
        "gate_criteria": gate_rows,
        "failures": failures,
        "variant_commutator": variant_rows,
        "next_step": next_step
        or (
            "repair missing ACSR finite-update commutator assay controls"
            if failures
            else "interpret sparse-vs-dense finite-update commutator deltas"
        ),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "variant_commutator_csv": str(out_dir / "variant_commutator.csv"),
            "per_token_commutator_csv": str(out_dir / "per_token_commutator.csv"),
            "commutator_strata_csv": str(out_dir / "commutator_strata.csv"),
            "gate_criteria_csv": str(out_dir / "gate_criteria.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }


def _summary_metrics(variant_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sparse = _variant_row(variant_rows, SPARSE_VARIANT)
    topk1 = _variant_row(variant_rows, "sparse_acsr_rank_matched_topk1")
    dense = _variant_row(variant_rows, DENSE_VARIANT)
    position_dense = _variant_row(variant_rows, "dense_token_position_rank1")
    return {
        "sparse_mean_logit_mse": sparse.get("mean_logit_mse") if sparse else None,
        "topk1_mean_logit_mse": topk1.get("mean_logit_mse") if topk1 else None,
        "dense_mean_logit_mse": dense.get("mean_logit_mse") if dense else None,
        "position_dense_mean_logit_mse": position_dense.get("mean_logit_mse") if position_dense else None,
        "sparse_minus_dense_logit_mse": _delta(
            sparse.get("mean_logit_mse") if sparse else None,
            dense.get("mean_logit_mse") if dense else None,
        ),
        "sparse_minus_topk1_logit_mse": _delta(
            sparse.get("mean_logit_mse") if sparse else None,
            topk1.get("mean_logit_mse") if topk1 else None,
        ),
        "sparse_mean_symmetric_kl": sparse.get("mean_symmetric_kl") if sparse else None,
        "sparse_mean_ce_abs_delta": sparse.get("mean_ce_abs_delta") if sparse else None,
        "sparse_support_churn_fraction": sparse.get("support_churn_fraction") if sparse else None,
    }


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    variant_rows: list[dict[str, Any]],
    per_token_rows: list[dict[str, Any]],
    strata_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "variant_commutator.csv", variant_rows)
    _write_csv(out_dir / "per_token_commutator.csv", per_token_rows)
    _write_csv(out_dir / "commutator_strata.csv", strata_rows)
    _write_csv(out_dir / "gate_criteria.csv", gate_rows)
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ACSR finite-update commutator assay",
        "",
        f"Status: `{summary['status']}`",
        f"Decision: `{summary['decision']}`",
        f"Claim status: `{summary['claim_status']}`",
        "",
        "This local command replays anchor and transfer residual updates in both "
        "orders for sparse ACSR and rank-1 dense causal controls. CE movement is "
        "a guardrail; the primary observables are symmetric KL, logit MSE, "
        "residual-delta L2, support churn, and dense-minus-sparse paired deltas.",
        "",
        f"Next step: {summary['next_step']}",
    ]
    if summary["failures"]:
        lines.extend(["", "## Failures"])
        lines.extend(f"- `{failure}`" for failure in summary["failures"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _split_mask(torch: Any, hidden: Any, split: str) -> Any:
    seq_minus_one = int(hidden.shape[1] - 1)
    position = torch.arange(seq_minus_one, device=hidden.device).view(1, -1)
    if split == "anchor":
        mask = position < seq_minus_one // 2
    elif split == "transfer":
        mask = position >= seq_minus_one // 2
    else:
        raise ValueError(f"unknown split: {split}")
    return mask.expand(int(hidden.shape[0]), seq_minus_one).reshape(-1)


def _support_string(support: Any | None, batch_index: int, position_index: int) -> str:
    if support is None:
        return ""
    values = support[batch_index, position_index].detach().cpu().tolist()
    return ",".join(str(int(value)) for value in values)


def _eval_module(module: Any) -> None:
    if hasattr(module, "eval"):
        module.eval()
    elif hasattr(module, "_module") and hasattr(module._module, "eval"):
        module._module.eval()


def _aggregate_numeric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in (
        "forward_ce",
        "reverse_ce",
        "ce_delta_forward_minus_reverse",
        "ce_abs_delta",
        "symmetric_kl",
        "logit_mse",
        "residual_delta_l2",
        "forward_residual_l2",
        "reverse_residual_l2",
    ):
        out[f"mean_{field}"] = _mean_field(rows, field)
    return out


def _variant_row(rows: list[dict[str, Any]], variant: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("variant") == variant:
            return row
    return None


def _mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [
        numeric
        for numeric in (_float_or_none(row.get(field)) for row in rows)
        if numeric is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _true_fraction(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [str(row.get(field, "")).strip().lower() for row in rows]
    present = [value for value in values if value in {"true", "false"}]
    if not present:
        return None
    return sum(value == "true" for value in present) / len(present)


def _delta(left: Any, right: Any) -> float | None:
    left_float = _float_or_none(left)
    right_float = _float_or_none(right)
    if left_float is None or right_float is None:
        return None
    return left_float - right_float


def _at_most(value: Any, threshold: float) -> bool:
    numeric = _float_or_none(value)
    return numeric is not None and numeric <= threshold


def _at_least(value: Any, threshold: float) -> bool:
    numeric = _float_or_none(value)
    return numeric is not None and numeric >= threshold


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--phase-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--ce-guardrail", type=float, default=0.05)
    parser.add_argument("--material-logit-mse-threshold", type=float, default=0.01)
    args = parser.parse_args(argv)
    summary = run_acsr_finite_update_commutator_assay(
        config_path=args.config,
        out_dir=args.out,
        phase_steps=args.phase_steps,
        learning_rate=args.learning_rate,
        ce_guardrail=args.ce_guardrail,
        material_logit_mse_threshold=args.material_logit_mse_threshold,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "claim_status": summary["claim_status"],
                "metrics": summary["metrics"],
                "failures": summary["failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
