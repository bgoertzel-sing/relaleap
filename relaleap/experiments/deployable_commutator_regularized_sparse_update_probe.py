"""Local CPU probe for a deployable commutator-regularized sparse update."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.retention_churn_microtest import DEFAULT_CONFIG
from relaleap.experiments.retention_churn_microtest import run_retention_churn_microtest


DEFAULT_PREGATE = Path("results/reports/deployable_commutator_regularized_sparse_update_pregate/summary.json")
DEFAULT_FLAT_VALUE = Path("results/reports/same_router_flat_value_commutator_mitigation_probe/summary.json")
DEFAULT_ORDER_PROBE = Path("results/audits/token_larger_promoted_topk2_explicit_order_averaging_mitigation_probe/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/audits/deployable_commutator_regularized_sparse_update_probe")

CANDIDATE = "deployable_commutator_regularized_sparse_update"
UNREGULARIZED = "promoted_contextual_topk2"
DENSE_ACTIVE = "norm_matched_dense_active_rank"
DENSE_STORED = "norm_matched_dense_stored_rank"
RANDOM_SUPPORT = "random_fixed_topk2"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "arm_metrics.csv",
    "control_comparison.csv",
    "support_overlap_strata.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_deployable_commutator_regularized_sparse_update_probe(
    *,
    config_path: Path = DEFAULT_CONFIG,
    pregate_path: Path = DEFAULT_PREGATE,
    flat_value_path: Path = DEFAULT_FLAT_VALUE,
    order_probe_path: Path = DEFAULT_ORDER_PROBE,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
    commutator_improvement_fraction: float = 0.10,
    ce_drift_tolerance: float = 0.05,
    transfer_retention_fraction: float = 0.80,
    residual_norm_ratio_min: float = 0.80,
    residual_norm_ratio_max: float = 1.25,
) -> dict[str, Any]:
    """Run the local deployable update probe and fail promotion gates closed."""

    start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    microtest_dir = out_dir / "microtest"
    pregate = _read_json(pregate_path)
    flat_value = _read_json(flat_value_path)
    order_probe = _read_json(order_probe_path)
    strategy = _strategy_review(strategy_review_path)
    microtest = run_retention_churn_microtest(
        config_path,
        microtest_dir,
        include_deployable_commutator_regularized_sparse_update_variant=True,
    )
    variant_rows = [
        row for row in microtest.get("audit", {}).get("variants", []) if isinstance(row, dict)
    ]
    per_token_rows = _read_csv(microtest_dir / "per_token_commutator.csv")
    by_variant = {str(row.get("variant")): row for row in variant_rows}
    arm_rows = _arm_rows(by_variant, per_token_rows)
    support_overlap_rows = _support_overlap_rows(per_token_rows)
    control_rows = _control_rows(arm_rows, flat_value, order_probe)
    thresholds = {
        "commutator_improvement_fraction": commutator_improvement_fraction,
        "ce_drift_tolerance": ce_drift_tolerance,
        "transfer_retention_fraction": transfer_retention_fraction,
        "residual_norm_ratio_min": residual_norm_ratio_min,
        "residual_norm_ratio_max": residual_norm_ratio_max,
    }
    source_rows = [
        _source_row("deployable_commutator_pregate", pregate_path, pregate),
        _source_row("flat_value_commutator_control", flat_value_path, flat_value),
        _source_row("explicit_order_averaging_upper_bound", order_probe_path, order_probe),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": f"strategic_change_level={strategy['strategic_change_level']}; notify_ben={strategy['notify_ben']}; verdict={strategy['verdict']}",
        },
        {
            "source": "retention_churn_microtest",
            "path": str(microtest_dir / "summary.json"),
            "present": (microtest_dir / "summary.json").is_file(),
            "status": microtest.get("status"),
            "decision": microtest.get("experiment_id", ""),
            "claim_status": f"variant_count={len(variant_rows)}",
        },
    ]
    gate_rows = _gate_rows(
        source_rows=source_rows,
        arm_rows=arm_rows,
        control_rows=control_rows,
        support_overlap_rows=support_overlap_rows,
        pregate=pregate,
        thresholds=thresholds,
    )
    hard_failures = [row for row in gate_rows if not row["passed"] and row["severity"] == "hard"]
    claim_failures = [row for row in gate_rows if not row["passed"] and row["severity"] == "claim"]
    if hard_failures:
        decision = "deployable_commutator_regularized_sparse_update_probe_failed_closed"
        claim_status = "source_or_schema_failure_no_scientific_claim"
        selected_next_step = "repair the local deployable sparse-update probe source/schema before more mechanism work"
    elif claim_failures:
        decision = "deployable_commutator_regularized_sparse_update_probe_recorded_gpu_blocked"
        claim_status = "deployable_sparse_update_not_established"
        selected_next_step = "close or redesign the deployable commutator-regularized sparse update before any GPU validation"
    else:
        decision = "deployable_commutator_regularized_sparse_update_candidate_local_gate_passed"
        claim_status = "local_candidate_supported_not_promoted"
        selected_next_step = "repeat the deployable commutator-regularized sparse update probe on a second local seed before any GPU validation"
    summary = {
        "status": "fail" if hard_failures else "pass",
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "backend_policy": "local CPU probe only; Colab/RunPod validation remains blocked by local gates",
        "config_path": str(config_path),
        "thresholds": thresholds,
        "source_rows": source_rows,
        "arm_metrics": arm_rows,
        "control_comparison": control_rows,
        "support_overlap_strata": support_overlap_rows,
        "gate_criteria": gate_rows,
        "failures": hard_failures,
        "claim_failures": claim_failures,
        "strategy_review": strategy,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _arm_rows(by_variant: dict[str, dict[str, Any]], per_token_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for variant in (UNREGULARIZED, CANDIDATE, DENSE_ACTIVE, DENSE_STORED, RANDOM_SUPPORT):
        source = by_variant.get(variant, {})
        rows.append(_arm_row(variant, source, per_token_rows))
    rows.append(
        {
            "variant": "no_update",
            "role": "diagnostic_floor_not_learning_control",
            "present": True,
            "commutator_anchor_logit_mse": 0.0,
            "commutator_anchor_symmetric_kl": 0.0,
            "anchor_ce_drift": 0.0,
            "transfer_ce_improvement": 0.0,
            "anchor_residual_norm_after_transfer": 0.0,
            "stored_parameters": 0,
            "active_parameters_proxy": 0,
        }
    )
    return rows


def _arm_row(variant: str, source: dict[str, Any], per_token_rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "variant": variant,
        "role": _role(variant),
        "present": bool(source),
        "kind": source.get("kind", ""),
        "support_router": source.get("support_router", ""),
        "stored_parameters": _float_or_none(source.get("stored_parameters")),
        "active_parameters_proxy": _float_or_none(source.get("active_parameters_proxy")),
        "commutator_anchor_logit_mse": _float_or_none(source.get("commutator_anchor_logit_mse")),
        "commutator_transfer_logit_mse": _float_or_none(source.get("commutator_transfer_logit_mse")),
        "commutator_anchor_symmetric_kl": _mean_per_token(per_token_rows, variant, "anchor", "symmetric_kl"),
        "commutator_transfer_symmetric_kl": _mean_per_token(per_token_rows, variant, "transfer", "symmetric_kl"),
        "anchor_ce_drift": _float_or_none(source.get("anchor_ce_drift")),
        "anchor_kl_drift_proxy": _float_or_none(source.get("anchor_logit_mse_drift")),
        "transfer_ce_improvement": _float_or_none(source.get("transfer_ce_improvement")),
        "anchor_residual_norm_before_transfer": _float_or_none(source.get("anchor_residual_norm_before_transfer")),
        "anchor_residual_norm_after_transfer": _float_or_none(source.get("anchor_residual_norm_after_transfer")),
        "anchor_support_churn_after_transfer": _float_or_none(source.get("anchor_support_churn_after_transfer")),
        "commutator_anchor_support_churn": _float_or_none(source.get("commutator_anchor_support_churn")),
        "parameter_delta_after_anchor": _float_or_none(source.get("parameter_delta_after_anchor")),
        "parameter_delta_during_transfer": _float_or_none(source.get("parameter_delta_during_transfer")),
        "freeze_router_during_transfer": source.get("freeze_router_during_transfer", ""),
        "gradient_clip_norm": source.get("gradient_clip_norm", ""),
        "value_gradient_clip_norm": source.get("value_gradient_clip_norm", ""),
        "value_gradient_low_rank": source.get("value_gradient_low_rank", ""),
        "commutator_value_penalty_weight": source.get("commutator_value_penalty_weight", ""),
    }


def _role(variant: str) -> str:
    return {
        CANDIDATE: "candidate_deployable_forward_only_sparse_update",
        UNREGULARIZED: "sparse_unregularized_baseline",
        DENSE_ACTIVE: "dense_active_matched_control",
        DENSE_STORED: "dense_stored_matched_control",
        RANDOM_SUPPORT: "random_support_sparse_null",
    }.get(variant, "control")


def _control_rows(arm_rows: list[dict[str, Any]], flat_value: dict[str, Any], order_probe: dict[str, Any]) -> list[dict[str, Any]]:
    by_arm = {row["variant"]: row for row in arm_rows}
    candidate = by_arm.get(CANDIDATE, {})
    rows = [
        _comparison_row(candidate, by_arm.get(UNREGULARIZED, {}), "sparse_unregularized_update"),
        _comparison_row(candidate, by_arm.get(DENSE_ACTIVE, {}), "dense_active_matched_update"),
        _comparison_row(candidate, by_arm.get(DENSE_STORED, {}), "dense_stored_matched_update"),
        _comparison_row(candidate, by_arm.get(RANDOM_SUPPORT, {}), "random_support_sparse_update"),
        {
            "control": "same_router_flat_value_update",
            "control_present": flat_value.get("status") == "pass",
            "control_decision": flat_value.get("decision", ""),
            "control_claim_status": flat_value.get("claim_status", ""),
            "control_best_passes": any(row.get("variant_passes") is True for row in flat_value.get("variant_rows", []) if isinstance(row, dict)),
            "candidate_minus_control_commutator_anchor_logit_mse": "",
            "candidate_selective_win": False,
        },
        {
            "control": "explicit_order_averaged_sparse_update",
            "control_present": order_probe.get("status") == "pass",
            "control_decision": order_probe.get("decision", ""),
            "control_claim_status": order_probe.get("claim_status", ""),
            "control_best_passes": order_probe.get("decision") == "explicit_order_averaging_diagnostic_candidate_not_promoted",
            "candidate_minus_control_commutator_anchor_logit_mse": "",
            "candidate_selective_win": False,
        },
    ]
    return rows


def _comparison_row(candidate: dict[str, Any], control: dict[str, Any], control_name: str) -> dict[str, Any]:
    candidate_value = candidate.get("commutator_anchor_logit_mse")
    control_value = control.get("commutator_anchor_logit_mse")
    delta = _delta(candidate_value, control_value)
    return {
        "control": control_name,
        "control_present": bool(control.get("present")),
        "control_variant": control.get("variant", ""),
        "candidate_commutator_anchor_logit_mse": candidate_value,
        "control_commutator_anchor_logit_mse": control_value,
        "candidate_minus_control_commutator_anchor_logit_mse": delta,
        "candidate_selective_win": delta is not None and delta < 0.0,
    }


def _support_overlap_rows(per_token_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = {"low": [], "medium": [], "high": []}
    for row in per_token_rows:
        if row.get("variant") != CANDIDATE:
            continue
        overlap = _support_overlap(row.get("forward_support", ""), row.get("reverse_support", ""))
        if overlap is None:
            continue
        bucket = "low" if overlap < 0.34 else "medium" if overlap < 0.67 else "high"
        buckets[bucket].append(overlap)
    return [
        {
            "support_overlap_bin": bucket,
            "row_count": len(values),
            "mean_overlap": _mean(values),
            "populated": bool(values),
        }
        for bucket, values in buckets.items()
    ]


def _gate_rows(
    *,
    source_rows: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    support_overlap_rows: list[dict[str, Any]],
    pregate: dict[str, Any],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    by_arm = {row["variant"]: row for row in arm_rows}
    candidate = by_arm.get(CANDIDATE, {})
    baseline = by_arm.get(UNREGULARIZED, {})
    required_arms = {CANDIDATE, UNREGULARIZED, DENSE_ACTIVE, DENSE_STORED, RANDOM_SUPPORT, "no_update"}
    present_arms = {row["variant"] for row in arm_rows if row.get("present")}
    dense_random = [by_arm.get(DENSE_ACTIVE, {}), by_arm.get(DENSE_STORED, {}), by_arm.get(RANDOM_SUPPORT, {})]
    best_generic = min(
        [float(row["commutator_anchor_logit_mse"]) for row in dense_random if row.get("commutator_anchor_logit_mse") is not None],
        default=None,
    )
    candidate_comm = candidate.get("commutator_anchor_logit_mse")
    baseline_comm = baseline.get("commutator_anchor_logit_mse")
    candidate_norm = candidate.get("anchor_residual_norm_after_transfer")
    baseline_norm = baseline.get("anchor_residual_norm_after_transfer")
    norm_ratio = _ratio(candidate_norm, baseline_norm)
    rows = [
        _criterion("pregate_passed_and_selected_probe", pregate.get("status") == "pass" and pregate.get("selected_next_action") == "implement_local_deployable_commutator_regularized_sparse_update_probe", "hard", "pregate must select this probe", pregate.get("selected_next_action")),
        _criterion("required_source_artifacts_present", all(row["present"] for row in source_rows[:3]), "hard", "pregate, flat, and order source artifacts must be present", [row["source"] for row in source_rows if not row["present"]]),
        _criterion("required_microtest_arms_present", required_arms.issubset(present_arms), "hard", "candidate plus sparse/dense/random/no-update controls must be present", sorted(required_arms - present_arms)),
        _criterion("candidate_improves_sparse_commutator", _fractional_reduction_pass(baseline_comm, candidate_comm, thresholds["commutator_improvement_fraction"]), "claim", "candidate must reduce sparse baseline commutator", _delta(candidate_comm, baseline_comm)),
        _criterion("candidate_beats_dense_and_random_controls", best_generic is not None and candidate_comm is not None and float(candidate_comm) <= best_generic * (1.0 - thresholds["commutator_improvement_fraction"]), "claim", "candidate must beat best dense/random generic control by threshold", _delta(candidate_comm, best_generic)),
        _criterion("ce_guardrail", _leq(candidate.get("anchor_ce_drift"), baseline.get("anchor_ce_drift"), thresholds["ce_drift_tolerance"]), "claim", "candidate anchor CE drift must stay near sparse baseline", _delta(candidate.get("anchor_ce_drift"), baseline.get("anchor_ce_drift"))),
        _criterion("transfer_retention", _ratio(candidate.get("transfer_ce_improvement"), baseline.get("transfer_ce_improvement")) is not None and _ratio(candidate.get("transfer_ce_improvement"), baseline.get("transfer_ce_improvement")) >= thresholds["transfer_retention_fraction"], "claim", "candidate must retain transfer improvement", _ratio(candidate.get("transfer_ce_improvement"), baseline.get("transfer_ce_improvement"))),
        _criterion("residual_norm_parity", norm_ratio is not None and thresholds["residual_norm_ratio_min"] <= norm_ratio <= thresholds["residual_norm_ratio_max"], "claim", "candidate residual norm must remain in matched band", norm_ratio),
        _criterion("support_overlap_bins_populated", all(row["populated"] for row in support_overlap_rows), "claim", "low/medium/high support-overlap bins must be populated", support_overlap_rows),
        _criterion("flat_and_order_controls_not_promotion_evidence", all(not row.get("candidate_selective_win") for row in control_rows if row["control"] in {"same_router_flat_value_update", "explicit_order_averaged_sparse_update"}), "claim", "flat/order controls remain nonpromotion references", "diagnostic-only"),
    ]
    return rows


def _criterion(criterion: str, passed: bool, severity: str, requirement: str, observed: Any) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "severity": severity,
        "requirement": requirement,
        "observed": observed,
        "failure_reason": "" if passed else requirement,
    }


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file() and bool(payload),
        "status": payload.get("status", "missing" if not path.is_file() else ""),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return {
        "path": str(path),
        "present": bool(text),
        "strategic_change_level": _header_value(text, "strategic_change_level") or "unknown",
        "notify_ben": _header_value(text, "notify_ben") or "unknown",
        "recommended_next_action": _header_value(text, "recommended_next_action") or "",
        "verdict": _header_value(text, "verdict") or "",
        "ben_notification_required": (_header_value(text, "notify_ben").lower() == "true" or _header_value(text, "strategic_change_level").lower() == "major"),
    }


def _header_value(text: str, key: str) -> str:
    prefix = f"{key}:"
    for line in text.splitlines()[:20]:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean_per_token(rows: list[dict[str, str]], variant: str, split: str, field: str) -> float | None:
    return _mean([
        float(row[field])
        for row in rows
        if row.get("variant") == variant and row.get("split") == split and _is_float(row.get(field))
    ])


def _support_overlap(left: str, right: str) -> float | None:
    if not left or not right:
        return None
    left_set = {item.strip() for item in left.split(",") if item.strip()}
    right_set = {item.strip() for item in right.split(",") if item.strip()}
    if not left_set and not right_set:
        return None
    return len(left_set & right_set) / float(len(left_set | right_set))


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_float(value: Any) -> bool:
    return _float_or_none(value) is not None


def _mean(values: list[float]) -> float | None:
    return sum(values) / float(len(values)) if values else None


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _ratio(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    right_value = float(right)
    if abs(right_value) <= 1e-12:
        return None
    return float(left) / right_value


def _leq(left: Any, right: Any, margin: float) -> bool:
    if left is None or right is None:
        return False
    return float(left) <= float(right) + margin


def _fractional_reduction_pass(baseline: Any, candidate: Any, fraction: float) -> bool:
    if baseline is None or candidate is None:
        return False
    baseline_value = float(baseline)
    if baseline_value <= 1e-12:
        return False
    return float(candidate) <= baseline_value * (1.0 - fraction)


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "arm_metrics.csv", summary["arm_metrics"])
    _write_csv(out_dir / "control_comparison.csv", summary["control_comparison"])
    _write_csv(out_dir / "support_overlap_strata.csv", summary["support_overlap_strata"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"], lineterminator="\n")
        writer.writeheader()
        for row in rows or [{"status": "missing"}]:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in writer.fieldnames or []})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Deployable Commutator-Regularized Sparse Update Probe",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Claim status: `{summary['claim_status']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            f"- Next step: {summary['selected_next_step']}",
            "",
            "This is local CPU evidence only. Explicit order averaging remains a nondeployable upper-bound control, and Colab/RunPod validation remains blocked unless local gates pass.",
            "",
        ]
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_deployable_commutator_regularized_sparse_update_probe(config_path=args.config, out_dir=args.out)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "claim_status": summary["claim_status"]}, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
