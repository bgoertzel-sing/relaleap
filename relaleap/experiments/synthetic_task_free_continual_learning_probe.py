"""Synthetic task-free continual-learning probe for dense-vs-sparse residuals."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.heldout_context_post_probe_decision_report import (
    DECISION_RECORDED,
    NEXT_BRANCH,
)
from relaleap.experiments.retention_churn_microtest import DEFAULT_CONFIG
from relaleap.experiments.retention_churn_microtest import run_retention_churn_microtest


DEFAULT_DECISION_DIR = Path("results/reports/heldout_context_post_probe_decision")
DEFAULT_OUT_DIR = Path("results/reports/synthetic_task_free_continual_learning_probe")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "variant_metrics.csv",
    "paired_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)

REQUIRED_VARIANTS = (
    "rank_matched_contextual_topk1",
    "promoted_contextual_topk2",
    "random_fixed_topk2",
    "norm_matched_dense_active_rank",
)

PRIMARY_SPARSE = "rank_matched_contextual_topk1"
PRIMARY_DENSE = "norm_matched_dense_active_rank"
TOPK2_REFERENCE = "promoted_contextual_topk2"
RANDOM_SUPPORT_NULL = "random_fixed_topk2"


def run_synthetic_task_free_continual_learning_probe(
    *,
    config_path: Path = DEFAULT_CONFIG,
    decision_dir: Path = DEFAULT_DECISION_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Run a bounded local retention/forgetting probe and write fail-closed artifacts."""

    start = time.time()
    decision = _read_json(decision_dir / "summary.json")
    early_gates = _preflight_gates(config_path, decision)
    if any(not row["passed"] and row["severity"] == "hard" for row in early_gates):
        summary = _summary(
            status="fail",
            decision="synthetic_task_free_continual_learning_probe_failed_closed",
            claim_status="preflight_failed",
            start=start,
            config_path=config_path,
            decision_dir=decision_dir,
            microtest={},
            variant_rows=[],
            paired_rows=[],
            gate_rows=early_gates,
            out_dir=out_dir,
        )
        _write_artifacts(out_dir, summary)
        return summary

    try:
        microtest = run_retention_churn_microtest(config_path, out_dir / "microtest")
    except Exception as exc:  # pragma: no cover - depends on torch runtime
        gate_rows = early_gates + [
            _criterion(
                "microtest_runtime",
                False,
                "hard",
                "retention/churn microtest completes",
                str(exc),
                "microtest runtime failed",
            )
        ]
        summary = _summary(
            status="fail",
            decision="synthetic_task_free_continual_learning_probe_failed_closed",
            claim_status="microtest_runtime_failed",
            start=start,
            config_path=config_path,
            decision_dir=decision_dir,
            microtest={},
            variant_rows=[],
            paired_rows=[],
            gate_rows=gate_rows,
            out_dir=out_dir,
        )
        _write_artifacts(out_dir, summary)
        return summary

    variant_rows = _variant_rows(microtest)
    paired_rows = _paired_rows(variant_rows)
    gate_rows = early_gates + _probe_gates(microtest, variant_rows, paired_rows)
    hard_failures = [row for row in gate_rows if not row["passed"] and row["severity"] == "hard"]
    claim_blockers = [row for row in gate_rows if not row["passed"] and row["severity"] != "hard"]
    status = "fail" if hard_failures else "pass"
    sparse_retention = _paired_metric(
        paired_rows,
        "topk1_minus_dense",
        "left_minus_right_anchor_ce_drift",
    )
    sparse_logit = _paired_metric(
        paired_rows,
        "topk1_minus_dense",
        "left_minus_right_anchor_logit_mse_drift",
    )
    sparse_transfer = _paired_metric(
        paired_rows,
        "topk1_minus_dense",
        "left_minus_right_transfer_ce_improvement",
    )
    is_true_synthetic_cl = _is_true_synthetic_cl_dataset(microtest)
    if status == "fail":
        decision_name = "synthetic_task_free_continual_learning_probe_failed_closed"
        claim_status = "source_artifacts_missing_or_inconsistent"
        next_step = "repair synthetic retention probe artifacts before interpreting dense-vs-sparse retention"
    elif not is_true_synthetic_cl:
        decision_name = "confounded_slice_retention_signal_recorded"
        claim_status = "confounded_slice_retention_not_synthetic_cl"
        next_step = "implement_mechanism_factorized_local_continual_learning_probe"
    elif not claim_blockers:
        decision_name = "synthetic_sparse_retention_candidate_supported"
        claim_status = "rank_matched_topk1_retention_advantage_candidate_not_promoted"
        next_step = "repeat the synthetic retention probe on a second seed before any GPU validation"
    else:
        decision_name = "synthetic_sparse_retention_candidate_blocked"
        claim_status = "dense_residual_controls_remain_active_retention_baseline"
        next_step = "stop sparse topk1 promotion unless a new mechanism changes retention or heldout CE evidence"

    summary = _summary(
        status=status,
        decision=decision_name,
        claim_status=claim_status,
        start=start,
        config_path=config_path,
        decision_dir=decision_dir,
        microtest=microtest,
        variant_rows=variant_rows,
        paired_rows=paired_rows,
        gate_rows=gate_rows,
        out_dir=out_dir,
        selected_next_step=next_step,
        primary_result={
            "dataset": _as_dict(microtest.get("audit")).get("dataset"),
            "is_true_synthetic_cl_dataset": is_true_synthetic_cl,
            "topk1_minus_dense_anchor_ce_drift": sparse_retention,
            "topk1_minus_dense_anchor_logit_mse_drift": sparse_logit,
            "topk1_minus_dense_transfer_ce_improvement": sparse_transfer,
            "interpretation": (
                "Negative CE/logit drift deltas favor top-k1 retention; positive transfer delta favors top-k1 adaptation. "
                "Non-synthetic datasets are same-domain slice-retention signals, not mechanism-factorized CL evidence."
            ),
        },
    )
    _write_artifacts(out_dir, summary)
    return summary


def _preflight_gates(config_path: Path, decision: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _criterion(
            "config_present",
            config_path.is_file(),
            "hard",
            "config exists",
            str(config_path) if config_path.is_file() else "missing",
            "cannot run local synthetic retention probe without config",
        ),
        _criterion(
            "heldout_post_probe_selected_this_branch",
            decision.get("status") == "pass"
            and decision.get("decision") == DECISION_RECORDED
            and decision.get("selected_next_step") == NEXT_BRANCH,
            "hard",
            "post-probe decision selects synthetic retention branch",
            {
                "status": decision.get("status"),
                "decision": decision.get("decision"),
                "selected_next_step": decision.get("selected_next_step"),
            },
            "do not duplicate or skip branch-selection source of truth",
        ),
    ]


def _variant_rows(microtest: dict[str, Any]) -> list[dict[str, Any]]:
    audit = _as_dict(microtest.get("audit"))
    variants = [
        row for row in audit.get("variants", []) if isinstance(row, dict)
    ]
    rows = []
    for row in variants:
        variant = str(row.get("variant", ""))
        if variant not in REQUIRED_VARIANTS:
            continue
        anchor_norm = _float(row.get("anchor_residual_norm_after_transfer"))
        transfer_gain = _float(row.get("transfer_ce_improvement"))
        rows.append(
            {
                "variant": variant,
                "kind": row.get("kind", ""),
                "support_router": row.get("support_router", ""),
                "top_k": row.get("top_k", ""),
                "num_columns": row.get("num_columns", ""),
                "stored_parameters": _float(row.get("stored_parameters")),
                "active_parameters_proxy": _float(row.get("active_parameters_proxy")),
                "flops_proxy": _float(row.get("active_parameters_proxy")),
                "task_a_retention_ce_delta_after_task_b_update": _float(row.get("anchor_ce_drift")),
                "task_b_adaptation_ce_delta": -transfer_gain if transfer_gain is not None else None,
                "task_b_transfer_ce_improvement": transfer_gain,
                "anchor_kl_or_logit_mse_churn": _float(row.get("anchor_logit_mse_drift")),
                "functional_churn_symmetric_kl_proxy": _float(row.get("commutator_anchor_logit_mse")),
                "residual_stream_churn_l2": _float(row.get("anchor_residual_stream_l2_drift")),
                "residual_norm_after_transfer": anchor_norm,
                "residual_gain_per_l2": _safe_divide(
                    _float(row.get("anchor_ce_drift")),
                    anchor_norm,
                ),
                "finite_update_commutator_anchor_logit_mse": _float(row.get("commutator_anchor_logit_mse")),
                "finite_update_commutator_transfer_logit_mse": _float(row.get("commutator_transfer_logit_mse")),
                "finite_update_commutator_anchor_residual_l2": _float(row.get("commutator_anchor_residual_stream_l2")),
                "support_identity_churn": row.get("anchor_support_churn_after_transfer", ""),
                "used_columns_after_transfer": row.get("anchor_used_columns_after_transfer", ""),
                "unique_support_sets_after_transfer": row.get("anchor_unique_support_sets_after_transfer", ""),
            }
        )
    audit_rows = [
        {
            "variant": "frozen_base_anchor",
            "kind": "base",
            "support_router": "none",
            "top_k": 0,
            "num_columns": 0,
            "stored_parameters": 0.0,
            "active_parameters_proxy": 0.0,
            "flops_proxy": 0.0,
            "task_a_retention_ce_delta_after_task_b_update": 0.0,
            "task_b_adaptation_ce_delta": 0.0,
            "task_b_transfer_ce_improvement": 0.0,
            "anchor_kl_or_logit_mse_churn": 0.0,
            "functional_churn_symmetric_kl_proxy": 0.0,
            "residual_stream_churn_l2": 0.0,
            "residual_norm_after_transfer": 0.0,
            "residual_gain_per_l2": None,
            "finite_update_commutator_anchor_logit_mse": 0.0,
            "finite_update_commutator_transfer_logit_mse": 0.0,
            "finite_update_commutator_anchor_residual_l2": 0.0,
            "support_identity_churn": "",
            "used_columns_after_transfer": "",
            "unique_support_sets_after_transfer": "",
        }
    ]
    return audit_rows + rows


def _paired_rows(variant_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant = {str(row.get("variant")): row for row in variant_rows}
    pairs = (
        ("topk1_minus_dense", PRIMARY_SPARSE, PRIMARY_DENSE),
        ("topk1_minus_topk2_reference", PRIMARY_SPARSE, TOPK2_REFERENCE),
        ("topk1_minus_random_support_null", PRIMARY_SPARSE, RANDOM_SUPPORT_NULL),
        ("topk2_minus_dense", TOPK2_REFERENCE, PRIMARY_DENSE),
    )
    rows = []
    for comparison, left_name, right_name in pairs:
        left = by_variant.get(left_name, {})
        right = by_variant.get(right_name, {})
        rows.append(
            {
                "comparison": comparison,
                "left_variant": left_name,
                "right_variant": right_name,
                "left_present": bool(left),
                "right_present": bool(right),
                "left_minus_right_anchor_ce_drift": _delta(
                    left.get("task_a_retention_ce_delta_after_task_b_update"),
                    right.get("task_a_retention_ce_delta_after_task_b_update"),
                ),
                "left_minus_right_anchor_logit_mse_drift": _delta(
                    left.get("anchor_kl_or_logit_mse_churn"),
                    right.get("anchor_kl_or_logit_mse_churn"),
                ),
                "left_minus_right_functional_churn_proxy": _delta(
                    left.get("functional_churn_symmetric_kl_proxy"),
                    right.get("functional_churn_symmetric_kl_proxy"),
                ),
                "left_minus_right_transfer_ce_improvement": _delta(
                    left.get("task_b_transfer_ce_improvement"),
                    right.get("task_b_transfer_ce_improvement"),
                ),
                "left_minus_right_residual_norm": _delta(
                    left.get("residual_norm_after_transfer"),
                    right.get("residual_norm_after_transfer"),
                ),
                "left_minus_right_commutator_anchor_logit_mse": _delta(
                    left.get("finite_update_commutator_anchor_logit_mse"),
                    right.get("finite_update_commutator_anchor_logit_mse"),
                ),
            }
        )
    return rows


def _probe_gates(
    microtest: dict[str, Any],
    variant_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    variants = {str(row.get("variant")) for row in variant_rows}
    missing = sorted(set(REQUIRED_VARIANTS) - variants)
    accounting_missing = [
        str(row.get("variant"))
        for row in variant_rows
        if row.get("variant") != "frozen_base_anchor"
        and (
            _float(row.get("active_parameters_proxy")) is None
            or _float(row.get("flops_proxy")) is None
            or _float(row.get("residual_norm_after_transfer")) is None
        )
    ]
    topk1_dense_retention = _paired_metric(
        paired_rows,
        "topk1_minus_dense",
        "left_minus_right_anchor_ce_drift",
    )
    topk1_dense_logit = _paired_metric(
        paired_rows,
        "topk1_minus_dense",
        "left_minus_right_anchor_logit_mse_drift",
    )
    topk1_dense_transfer = _paired_metric(
        paired_rows,
        "topk1_minus_dense",
        "left_minus_right_transfer_ce_improvement",
    )
    topk1_random_retention = _paired_metric(
        paired_rows,
        "topk1_minus_random_support_null",
        "left_minus_right_anchor_ce_drift",
    )
    dataset = _as_dict(microtest.get("audit")).get("dataset")
    return [
        _criterion(
            "microtest_status_ok",
            microtest.get("status") == "ok",
            "hard",
            "retention/churn microtest status is ok",
            microtest.get("status"),
            "microtest failed or returned stale schema",
        ),
        _criterion(
            "required_arms_present",
            not missing,
            "hard",
            "dense, top-k1, top-k2 reference, and random support null are present",
            missing,
            "missing required synthetic continual-learning arms",
        ),
        _criterion(
            "rank_flop_norm_accounting_present",
            not accounting_missing,
            "hard",
            "each learned arm records active params/flops proxy and residual norm",
            accounting_missing,
            "missing rank/FLOP/norm accounting",
        ),
        _criterion(
            "dataset_is_true_synthetic_cl",
            _is_true_synthetic_cl_dataset(microtest),
            "interpretation",
            "dataset is a known-rule synthetic continual-learning stream",
            dataset,
            "current result is same-domain slice retention, not true synthetic mechanism CL",
        ),
        _criterion(
            "topk1_retention_ce_drift_lower_than_dense",
            topk1_dense_retention is not None and topk1_dense_retention < 0.0,
            "claim",
            "top-k1 has lower task-A retention CE drift after task-B update than dense",
            topk1_dense_retention,
            "top-k1 did not improve retention CE drift over dense",
        ),
        _criterion(
            "topk1_anchor_logit_churn_not_higher_than_dense",
            topk1_dense_logit is not None and topk1_dense_logit <= 0.0,
            "claim",
            "top-k1 anchor logit churn is no higher than dense",
            topk1_dense_logit,
            "top-k1 logit churn exceeds dense",
        ),
        _criterion(
            "topk1_transfer_adaptation_not_materially_worse_than_dense",
            topk1_dense_transfer is not None and topk1_dense_transfer >= -0.05,
            "claim",
            "top-k1 task-B transfer improvement is within 0.05 CE of dense",
            topk1_dense_transfer,
            "top-k1 adaptation is materially worse than dense",
        ),
        _criterion(
            "topk1_beats_random_support_retention_null",
            topk1_random_retention is not None and topk1_random_retention < 0.0,
            "claim",
            "top-k1 retention CE drift beats random fixed support null",
            topk1_random_retention,
            "top-k1 retention does not beat random support null",
        ),
    ]


def _summary(
    *,
    status: str,
    decision: str,
    claim_status: str,
    start: float,
    config_path: Path,
    decision_dir: Path,
    microtest: dict[str, Any],
    variant_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    out_dir: Path,
    selected_next_step: str | None = None,
    primary_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failures = [row for row in gate_rows if not row["passed"] and row["severity"] == "hard"]
    claim_blockers = [row for row in gate_rows if not row["passed"] and row["severity"] != "hard"]
    return {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "backend_policy": "local CPU probe only; do not use RunPod unless a local repeat produces a claim-changing result",
        "config_path": str(config_path),
        "decision_dir": str(decision_dir),
        "microtest_out_dir": str(out_dir / "microtest") if microtest else "",
        "primary_result": primary_result or {},
        "variant_metrics": variant_rows,
        "paired_metrics": paired_rows,
        "gate_criteria": gate_rows,
        "failures": failures,
        "claim_blockers_preserved": claim_blockers,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(
        out_dir / "source_rows.csv",
        [
            {
                "source": "heldout_context_post_probe_decision",
                "path": str(Path(summary["decision_dir"]) / "summary.json"),
                "present": (Path(summary["decision_dir"]) / "summary.json").is_file(),
                "status": "consumed",
            },
            {
                "source": "retention_churn_microtest",
                "path": summary.get("microtest_out_dir", ""),
                "present": bool(summary.get("microtest_out_dir")),
                "status": summary["status"],
            },
        ],
    )
    _write_csv(out_dir / "variant_metrics.csv", summary["variant_metrics"])
    _write_csv(out_dir / "paired_metrics.csv", summary["paired_metrics"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    result = summary.get("primary_result", {})
    lines = [
        "# Synthetic Task-Free Continual-Learning Probe",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Config: `{summary['config_path']}`",
        f"- Dataset: `{result.get('dataset')}`",
        f"- True synthetic CL dataset: `{result.get('is_true_synthetic_cl_dataset')}`",
        "- Top-k1 minus dense retention CE drift: "
        f"`{result.get('topk1_minus_dense_anchor_ce_drift')}`",
        "- Top-k1 minus dense anchor logit MSE drift: "
        f"`{result.get('topk1_minus_dense_anchor_logit_mse_drift')}`",
        "- Top-k1 minus dense transfer CE improvement: "
        f"`{result.get('topk1_minus_dense_transfer_ce_improvement')}`",
        "",
        "This local probe trains on slice A, updates on slice B, then measures slice-A retention and slice-B adaptation against dense, sparse, top-k2, and random-support controls. When the dataset is not a known-rule synthetic mechanism stream, the result is labeled as confounded slice-retention evidence rather than synthetic continual-learning evidence.",
        "",
    ]
    if summary.get("selected_next_step"):
        lines.extend(["## Next Step", "", str(summary["selected_next_step"]), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _criterion(
    criterion: str,
    passed: bool,
    severity: str,
    requirement: str,
    observed: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "severity": severity,
        "requirement": requirement,
        "observed": observed,
        "failure_reason": "" if passed else failure_reason,
    }


def _paired_metric(rows: list[dict[str, Any]], comparison: str, field: str) -> float | None:
    for row in rows:
        if row.get("comparison") == comparison:
            return _float(row.get(field))
    return None


def _is_true_synthetic_cl_dataset(microtest: dict[str, Any]) -> bool:
    dataset = str(_as_dict(microtest.get("audit")).get("dataset", ""))
    return dataset.startswith("synthetic_") or dataset.startswith("mechanism_")


def _delta(left: Any, right: Any) -> float | None:
    left_value = _float(left)
    right_value = _float(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _safe_divide(numerator: Any, denominator: Any) -> float | None:
    left = _float(numerator)
    right = _float(denominator)
    if left is None or right is None or abs(right) <= 1e-12:
        return None
    return left / right


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


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


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--decision-dir", type=Path, default=DEFAULT_DECISION_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_synthetic_task_free_continual_learning_probe(
        config_path=args.config,
        decision_dir=args.decision_dir,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
