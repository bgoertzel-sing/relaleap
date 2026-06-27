"""Dense-teacher residual distillation pilot for ACSR comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.anticipatory_contextual_support_routing import (
    _FuturePredictor,
    _contextual_chunks,
    _decode_for_support,
    _feature_tensor,
    _position_predictor_inputs,
    _read_yaml,
    _score_from_features,
    _shuffle_tokens,
    _support_entropy,
    _train_predictor_row,
    _unique_support_sets,
    _used_columns,
)


DEFAULT_CONFIG = Path(
    "configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml"
)
DEFAULT_ACSR_GATE = Path(
    "results/reports/token_larger_anticipatory_contextual_support_routing_causal_mechanism_gate/summary.json"
)
DEFAULT_CONTEXTUAL_GATE = Path(
    "results/comparisons/contextual_support_router_promotion_gate_larger_char_token/summary.json"
)
DEFAULT_PRIOR_DISTILLATION_CLOSEOUT = Path(
    "results/reports/token_larger_causal_contextual_router_post_stratified_null_closeout/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_dense_teacher_residual_distillation_comparison"
)

PRIMARY_VARIANT = "acsr_predicted_future_support"
CONTEXTUAL_VARIANT = "promoted_contextual_router_support"
CONTROL_VARIANTS = (
    "token_position_only_predicted_support",
    "shuffled_predicted_support",
)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "variant_metrics.csv",
    "support_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_dense_teacher_residual_distillation_comparison(
    *,
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path = DEFAULT_OUT_DIR,
    acsr_gate_path: Path = DEFAULT_ACSR_GATE,
    contextual_gate_path: Path = DEFAULT_CONTEXTUAL_GATE,
    prior_distillation_closeout_path: Path = DEFAULT_PRIOR_DISTILLATION_CLOSEOUT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    max_steps: int | None = None,
    teacher_steps: int = 35,
    student_steps: int = 45,
    predictor_steps: int = 50,
) -> dict[str, Any]:
    """Run a bounded local dense-teacher distillation comparison."""

    start = time.time()
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        from relaleap.smoke import ResidualColumns, TinyCharTransformer, _build_batch
    except Exception as exc:  # pragma: no cover - environment dependent
        summary = _runtime_failure(out_dir, start, config_path, str(exc))
        _write_artifacts(out_dir, summary, [], [], [])
        return summary

    config = _read_yaml(config_path)
    run_cfg = _as_dict(config.get("run"))
    data_cfg = _as_dict(config.get("data"))
    model_cfg = _as_dict(config.get("model"))
    base_cfg = _as_dict(model_cfg.get("base"))
    column_cfg = _as_dict(model_cfg.get("columns"))

    seed = int(run_cfg.get("seed", 1))
    train_steps = int(max_steps if max_steps is not None else run_cfg.get("max_steps", 50))
    dataset = str(data_cfg.get("dataset", "tiny_shakespeare_word"))
    seq_len = int(data_cfg.get("seq_len", 64))
    hidden_dim = int(base_cfg.get("hidden_dim", 96))
    layers = int(base_cfg.get("layers", 2))
    num_columns = int(column_cfg.get("num_columns", 24))
    atoms_per_column = int(column_cfg.get("atoms_per_column", 4))
    top_k = int(column_cfg.get("top_k", 2))
    contextual_width = int(column_cfg.get("contextual_router_hidden_dim", hidden_dim * 2))

    torch.manual_seed(seed)
    inputs, targets, vocab_size = _build_batch(
        dataset=dataset,
        seq_len=seq_len,
        batch_size=4,
    )
    base = TinyCharTransformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        layers=layers,
    )
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base.eval()
    with torch.no_grad():
        hidden = base.encode(inputs)
        base_logits = base.decode(hidden)
        base_ce = _loss_value(_ce_loss(F, base_logits, targets, vocab_size))

    teacher = _DenseResidualTeacher(nn, hidden_dim)
    _train_dense_teacher(
        torch,
        F,
        base,
        teacher,
        hidden,
        targets,
        vocab_size,
        steps=max(1, min(teacher_steps, train_steps)),
    )
    teacher.eval()
    with torch.no_grad():
        teacher_hidden = teacher(hidden)
        teacher_logits = base.decode(teacher_hidden)
        teacher_ce = _loss_value(_ce_loss(F, teacher_logits, targets, vocab_size))

    chunks = _contextual_chunks(torch, hidden)
    causal_inputs = _causal_predictor_inputs(torch, chunks)
    position_inputs = _position_predictor_inputs(torch, chunks)
    future_targets = torch.cat([chunks["next"], chunks["next_delta"]], dim=-1)
    predictor = _FuturePredictor(nn, causal_inputs.shape[-1], hidden_dim * 2, hidden_dim)
    token_position_predictor = _FuturePredictor(
        nn,
        position_inputs.shape[-1],
        hidden_dim * 2,
        max(8, min(hidden_dim, 64)),
    )
    predictor_rows = [
        _train_predictor_row(
            torch,
            F,
            predictor,
            causal_inputs,
            future_targets,
            steps=max(1, predictor_steps),
            label="mlp_causal",
        ),
        _train_predictor_row(
            torch,
            F,
            token_position_predictor,
            position_inputs,
            future_targets,
            steps=max(1, predictor_steps),
            label="token_position_only",
        ),
    ]
    with torch.no_grad():
        predicted_future = predictor(causal_inputs)
        token_position_future = token_position_predictor(position_inputs)
        shuffled_future = _shuffle_tokens(torch, predicted_future)

    variants = {
        CONTEXTUAL_VARIANT: {
            "kind": "native_contextual",
            "features": _feature_tensor(torch, chunks, future_targets),
        },
        PRIMARY_VARIANT: {
            "kind": "forced_features",
            "features": _feature_tensor(torch, chunks, predicted_future),
        },
        "token_position_only_predicted_support": {
            "kind": "forced_features",
            "features": _feature_tensor(torch, chunks, token_position_future),
        },
        "shuffled_predicted_support": {
            "kind": "forced_features",
            "features": _feature_tensor(torch, chunks, shuffled_future),
        },
    }
    variant_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    for name, spec in variants.items():
        residual = ResidualColumns(
            hidden_dim=hidden_dim,
            num_columns=num_columns,
            atoms_per_column=atoms_per_column,
            top_k=top_k,
            support_router="contextual_mlp",
            contextual_router_hidden_dim=contextual_width,
        )
        _train_student(
            torch,
            F,
            base,
            residual,
            hidden,
            targets,
            teacher_logits,
            vocab_size,
            variant_kind=str(spec["kind"]),
            features=spec["features"],
            steps=max(1, min(student_steps, train_steps)),
        )
        row, support = _student_metric_row(
            torch,
            F,
            base,
            residual,
            hidden,
            targets,
            teacher_logits,
            vocab_size,
            variant=name,
            variant_kind=str(spec["kind"]),
            features=spec["features"],
        )
        variant_rows.append(row)
        support_rows.append(
            {
                "variant": name,
                "used_columns": _used_columns(support),
                "unique_support_sets": _unique_support_sets(support),
                "support_entropy": _support_entropy(torch, support, num_columns),
            }
        )

    source_rows = _source_rows(
        acsr_gate_path=acsr_gate_path,
        contextual_gate_path=contextual_gate_path,
        prior_distillation_closeout_path=prior_distillation_closeout_path,
        strategy_review_path=strategy_review_path,
    )
    criteria = _gate_criteria(
        variant_rows=variant_rows,
        teacher_ce=teacher_ce,
        base_ce=base_ce,
        source_rows=source_rows,
    )
    failures = [row for row in criteria if not row["passed"]]
    if failures:
        status = "fail"
        decision = "dense_teacher_residual_distillation_pilot_not_supported"
        claim_status = "dense_teacher_distillation_not_interpretable_or_not_better_than_controls"
        next_step = "repair_dense_teacher_distillation_pilot_or_return_to_acsr_broader_benchmark"
    else:
        status = "pass"
        decision = "dense_teacher_residual_distillation_acsr_pilot_supported_not_promoted"
        claim_status = "dense_teacher_distillation_acsr_pilot_supported_not_promoted"
        next_step = "replicate_dense_teacher_residual_distillation_on_a_broader_local_benchmark"

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": next_step,
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "dataset": dataset,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "num_columns": num_columns,
        "top_k": top_k,
        "teacher_steps": max(1, min(teacher_steps, train_steps)),
        "student_steps": max(1, min(student_steps, train_steps)),
        "predictor_steps": max(1, predictor_steps),
        "base_ce_loss": base_ce,
        "dense_teacher_ce_loss": teacher_ce,
        "dense_teacher_ce_improvement": base_ce - teacher_ce,
        "variant_rows": variant_rows,
        "support_rows": support_rows,
        "predictor_rows": predictor_rows,
        "source_rows": source_rows,
        "gate_status": {
            "passes_dense_teacher_distillation_gate": not failures,
            "criteria": criteria,
        },
        "failures": failures,
        "claim_statuses": {
            PRIMARY_VARIANT: claim_status if status == "pass" else "not_supported",
            CONTEXTUAL_VARIANT: "promoted_contextual_router_comparison_baseline",
            "dense_teacher": "local_cpu_pilot_teacher_not_default",
            "promoted_default_router": "no_default_change",
        },
        "rationale": _rationale(status),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, variant_rows, support_rows, criteria)
    return summary


class _DenseResidualTeacher:
    def __new__(cls, nn: Any, hidden_dim: int) -> Any:
        class DenseTeacher(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                nn.init.zeros_(self.net[-1].weight)
                nn.init.zeros_(self.net[-1].bias)

            def forward(self, hidden: Any) -> Any:
                return hidden + self.net(hidden)

        return DenseTeacher()


def _train_dense_teacher(
    torch: Any,
    F: Any,
    base: Any,
    teacher: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    *,
    steps: int,
) -> None:
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=3e-3)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = base.decode(teacher(hidden.detach()))
        loss = _ce_loss(F, logits, targets, vocab_size)
        loss.backward()
        optimizer.step()


def _train_student(
    torch: Any,
    F: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    teacher_logits: Any,
    vocab_size: int,
    *,
    variant_kind: str,
    features: Any,
    steps: int,
) -> None:
    optimizer = torch.optim.AdamW(residual.parameters(), lr=3e-3)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = _student_logits_and_support(
            torch,
            base,
            residual,
            hidden,
            variant_kind=variant_kind,
            features=features,
        )
        loss = F.mse_loss(logits, teacher_logits.detach()) + 0.1 * _ce_loss(
            F,
            logits,
            targets,
            vocab_size,
        )
        loss.backward()
        optimizer.step()


def _student_metric_row(
    torch: Any,
    F: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    targets: Any,
    teacher_logits: Any,
    vocab_size: int,
    *,
    variant: str,
    variant_kind: str,
    features: Any,
) -> tuple[dict[str, Any], Any]:
    with torch.no_grad():
        logits, support = _student_logits_and_support(
            torch,
            base,
            residual,
            hidden,
            variant_kind=variant_kind,
            features=features,
        )
        ce_loss = _loss_value(_ce_loss(F, logits, targets, vocab_size))
        distill_mse = float(F.mse_loss(logits, teacher_logits).item())
    return (
        {
            "variant": variant,
            "variant_kind": variant_kind,
            "ce_loss": ce_loss,
            "teacher_logit_mse": distill_mse,
        },
        support,
    )


def _student_logits_and_support(
    torch: Any,
    base: Any,
    residual: Any,
    hidden: Any,
    *,
    variant_kind: str,
    features: Any,
) -> tuple[Any, Any]:
    if variant_kind == "native_contextual":
        scores = residual._score_columns(hidden) + residual.score_tie_breaker.to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
    else:
        scores = _score_from_features(residual, features)
    top_values, support = scores.topk(residual.top_k, dim=-1)
    logits = _decode_for_support(torch, base, residual, hidden, support, top_values)
    return logits, support.detach()


def _causal_predictor_inputs(torch: Any, chunks: dict[str, Any]) -> Any:
    return torch.cat(
        [
            chunks["current"],
            chunks["previous"],
            chunks["previous_delta"],
            chunks["position"],
            chunks["sin_position"],
            chunks["cos_position"],
        ],
        dim=-1,
    )


def _ce_loss(F: Any, logits: Any, targets: Any, vocab_size: int) -> float:
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab_size),
        targets[:, :-1].reshape(-1),
    )


def _loss_value(loss: Any) -> float:
    return float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)


def _gate_criteria(
    *,
    variant_rows: list[dict[str, Any]],
    teacher_ce: float,
    base_ce: float,
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_name = {row["variant"]: row for row in variant_rows}
    acsr = by_name.get(PRIMARY_VARIANT, {})
    contextual = by_name.get(CONTEXTUAL_VARIANT, {})
    controls = [by_name.get(name, {}) for name in CONTROL_VARIANTS]
    source_pass = all(
        row.get("present") and row.get("status") in {"pass", "ok"}
        for row in source_rows[:3]
    )
    acsr_mse = _number(acsr.get("teacher_logit_mse"))
    acsr_ce = _number(acsr.get("ce_loss"))
    contextual_mse = _number(contextual.get("teacher_logit_mse"))
    control_mses = [_number(row.get("teacher_logit_mse")) for row in controls]
    return [
        {
            "criterion": "source_gates_present_and_passing",
            "passed": source_pass,
            "threshold": "ACSR gate, contextual gate, and prior distillation closeout pass",
            "actual": ",".join(str(row.get("status")) for row in source_rows[:3]),
        },
        {
            "criterion": "dense_teacher_improves_base_ce",
            "passed": teacher_ce < base_ce,
            "threshold": "dense teacher CE < base CE",
            "actual": f"{teacher_ce:.6f} < {base_ce:.6f}",
        },
        {
            "criterion": "acsr_distills_at_least_as_well_as_promoted_contextual",
            "passed": acsr_mse is not None and contextual_mse is not None and acsr_mse <= contextual_mse,
            "threshold": "ACSR teacher logit MSE <= promoted contextual MSE",
            "actual": f"{acsr_mse} <= {contextual_mse}",
        },
        {
            "criterion": "acsr_beats_token_position_and_shuffled_distillation_nulls",
            "passed": acsr_mse is not None
            and all(value is not None and acsr_mse <= value for value in control_mses),
            "threshold": "ACSR teacher logit MSE <= null MSEs",
            "actual": f"{acsr_mse} <= {control_mses}",
        },
        {
            "criterion": "acsr_ce_not_worse_than_teacher_by_large_margin",
            "passed": acsr_ce is not None and acsr_ce <= teacher_ce + 0.25,
            "threshold": "ACSR CE <= dense teacher CE + 0.25",
            "actual": f"{acsr_ce} <= {teacher_ce + 0.25:.6f}",
        },
    ]


def _source_rows(
    *,
    acsr_gate_path: Path,
    contextual_gate_path: Path,
    prior_distillation_closeout_path: Path,
    strategy_review_path: Path,
) -> list[dict[str, Any]]:
    paths = [
        ("acsr_causal_mechanism_gate", acsr_gate_path),
        ("contextual_support_router_promotion_gate", contextual_gate_path),
        ("prior_causal_router_distillation_closeout", prior_distillation_closeout_path),
        ("strategy_review", strategy_review_path),
    ]
    rows = []
    for source, path in paths:
        if path.suffix == ".md":
            review = _strategy_review(path)
            rows.append(
                {
                    "source": source,
                    "path": str(path),
                    "present": path.is_file(),
                    "status": "pass" if path.is_file() else "missing",
                    "decision": review.get("recommended_next_action"),
                    "claim_status": (
                        f"strategic_change_level={review.get('strategic_change_level')}; "
                        f"notify_ben={review.get('notify_ben')}"
                    ),
                }
            )
            continue
        payload = _read_json(path)
        status = payload.get("status")
        if status is None and payload.get("comparison_status") == "ok":
            status = "pass"
        elif status == "ok":
            status = "pass"
        rows.append(
            {
                "source": source,
                "path": str(path),
                "present": path.is_file(),
                "status": status,
                "decision": payload.get("decision"),
                "claim_status": payload.get("claim_status"),
            }
        )
    return rows


def _rationale(status: str) -> str:
    if status == "pass":
        return (
            "The local dense-teacher pilot supports ACSR-predicted support as a "
            "distillation candidate, but it is bounded to one CPU packet and "
            "does not change the default router."
        )
    return (
        "The dense-teacher pilot failed closed because ACSR did not beat all "
        "dense-distillation controls or because required source gates were not "
        "available. No default-router or mechanism-promotion claim is made."
    )


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    variant_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    criteria: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "variant_metrics.csv", variant_rows)
    _write_csv(out_dir / "support_metrics.csv", support_rows)
    _write_csv(out_dir / "gate_criteria.csv", criteria)
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Dense-Teacher Residual Distillation Comparison",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Decision: `{summary.get('decision')}`",
        f"- Claim status: `{summary.get('claim_status')}`",
        f"- Base CE: `{summary.get('base_ce_loss')}`",
        f"- Dense-teacher CE: `{summary.get('dense_teacher_ce_loss')}`",
        f"- Selected next step: {summary.get('selected_next_step')}",
        "",
        "This is a local CPU pilot only. It records whether ACSR-predicted "
        "support distills a dense residual teacher better than the promoted "
        "contextual-router support and token/position or shuffled null supports.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    else:
        fieldnames = ["status"]
        rows = [{"status": "missing"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False, "path": str(path)}
    fields: dict[str, Any] = {"present": True, "path": str(path)}
    for line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {"strategic_change_level", "notify_ben", "recommended_next_action"}:
            fields[key] = value
    fields["notify_ben"] = str(fields.get("notify_ben", "false")).lower() == "true"
    return fields


def _runtime_failure(out_dir: Path, start: float, config_path: Path, reason: str) -> dict[str, Any]:
    return {
        "status": "fail",
        "decision": "dense_teacher_residual_distillation_runtime_unavailable",
        "claim_status": "dense_teacher_distillation_not_executed",
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "failures": [{"criterion": "runtime", "reason": reason}],
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--acsr-gate", type=Path, default=DEFAULT_ACSR_GATE)
    parser.add_argument("--contextual-gate", type=Path, default=DEFAULT_CONTEXTUAL_GATE)
    parser.add_argument(
        "--prior-distillation-closeout",
        type=Path,
        default=DEFAULT_PRIOR_DISTILLATION_CLOSEOUT,
    )
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--teacher-steps", type=int, default=35)
    parser.add_argument("--student-steps", type=int, default=45)
    parser.add_argument("--predictor-steps", type=int, default=50)
    args = parser.parse_args()
    summary = run_dense_teacher_residual_distillation_comparison(
        config_path=args.config,
        out_dir=args.out,
        acsr_gate_path=args.acsr_gate,
        contextual_gate_path=args.contextual_gate,
        prior_distillation_closeout_path=args.prior_distillation_closeout,
        strategy_review_path=args.strategy_review,
        max_steps=args.max_steps,
        teacher_steps=args.teacher_steps,
        student_steps=args.student_steps,
        predictor_steps=args.predictor_steps,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
