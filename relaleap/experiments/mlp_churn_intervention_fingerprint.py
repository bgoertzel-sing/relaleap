"""Fail-closed MLP churn and intervention-fingerprint assay.

This report consumes existing command-generated artifacts. It does not
reconstruct raw residual interventions unless the source rows already include
the required raw fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_FOLLOWUP_DIR = Path("results/reports/mlp_dense_heldout_mechanism_followup")
DEFAULT_DENSE_OBSERVABLES_DIR = Path("results/reports/acsr_dense_mechanism_observables")
DEFAULT_SPARSE_GATE_DIR = Path("results/reports/acsr_sparse_dense_mechanism_gate")
DEFAULT_OUT_DIR = Path("results/reports/mlp_churn_intervention_fingerprint")

MLP_ARM = "parameter_matched_causal_mlp_control"
DENSE_ARMS = ("dense_rank16_best_norm", "dense_rank24_best_norm")
SPARSE_ARMS = (
    "acsr_mlp_predicted_future",
    "causal_feature_safe_contextual_topk2",
    "full_context_contextual_topk2_teacher",
)
NULL_ARMS = (
    "shuffled_predicted_features",
    "token_position_only_predicted_features",
    "random_fixed_topk2",
)
REQUIRED_ARMS = DENSE_ARMS + (MLP_ARM,)

REQUIRED_PROXY_PER_TOKEN_FIELDS = (
    "arm",
    "split",
    "base_ce_loss",
    "ce_loss",
    "delta_vs_base_ce",
    "residual_update_l2",
    "logit_mse_vs_base",
    "prediction_changed_vs_base",
)
REQUIRED_RAW_INTERVENTION_FIELDS = (
    "residual_update_vector",
    "base_logits",
    "candidate_logits",
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "matched_curves.csv",
    "scaled_interventions.csv",
    "scaled_match_summary.csv",
    "fingerprint_strata.csv",
    "available_arms.csv",
    "decision_criteria.csv",
    "notes.md",
)


def run_mlp_churn_intervention_fingerprint(
    *,
    followup_dir: Path = DEFAULT_FOLLOWUP_DIR,
    dense_observables_dir: Path = DEFAULT_DENSE_OBSERVABLES_DIR,
    sparse_gate_dir: Path = DEFAULT_SPARSE_GATE_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    min_heldout_rows_per_arm: int = 16,
) -> dict[str, Any]:
    """Write proxy CE/L2/churn fingerprints and fail closed on missing raw fields."""

    start = time.time()
    followup_summary = _read_json(followup_dir / "summary.json")
    per_token_rows = _read_csv(dense_observables_dir / "per_token_observables.csv")
    scorecard_rows = _read_csv(followup_dir / "mechanism_comparison.csv")
    sparse_rows = _read_csv(sparse_gate_dir / "mechanism_metrics.csv")

    available_arms = _available_arms(scorecard_rows, sparse_rows, per_token_rows)
    matched_curves = _matched_curves(per_token_rows)
    scaled_interventions = _scaled_interventions(per_token_rows)
    scaled_match_summary = _scaled_match_summary(scaled_interventions)
    fingerprint_strata = _fingerprint_strata(per_token_rows)
    missing_fields = _missing_required_fields(per_token_rows)
    criteria = _criteria(
        followup_summary=followup_summary,
        per_token_rows=per_token_rows,
        available_arms=available_arms,
        missing_fields=missing_fields,
        scaled_interventions=scaled_interventions,
        scaled_match_summary=scaled_match_summary,
        min_heldout_rows_per_arm=min_heldout_rows_per_arm,
    )
    failures = [row for row in criteria if not row["passed"]]

    decisive_raw_fields_present = not missing_fields["raw_intervention_fields"]
    if failures:
        status = "fail"
        if not decisive_raw_fields_present:
            decision = "mlp_churn_intervention_fingerprint_blocked_by_missing_raw_intervention_fields"
            claim_status = "proxy_fingerprint_written_but_raw_intervention_assay_not_decisive"
            selected_next_step = (
                "extend acsr_dense_mechanism_observable_extractor to emit raw residual updates "
                "and base/candidate logits for scaled CE/L2 intervention fingerprints"
            )
        else:
            decision = "mlp_churn_intervention_fingerprint_failed_closed"
            claim_status = "required_churn_fingerprint_inputs_missing"
            selected_next_step = "repair missing source artifacts before rerunning this local report"
    else:
        status = "pass"
        decision = "mlp_churn_intervention_fingerprint_scaled_assay_completed"
        claim_status = "raw_lambda_scaled_ce_l2_churn_fingerprints_available"
        selected_next_step = (
            "interpret the local scaled CE/L2 match summary before deciding whether a norm-budgeted "
            "MLP training variant is scientifically warranted"
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "source_dirs": {
            "followup": str(followup_dir),
            "dense_observables": str(dense_observables_dir),
            "sparse_gate": str(sparse_gate_dir),
        },
        "required_proxy_per_token_fields": list(REQUIRED_PROXY_PER_TOKEN_FIELDS),
        "required_raw_intervention_fields": list(REQUIRED_RAW_INTERVENTION_FIELDS),
        "missing_required_fields": missing_fields,
        "available_arms": available_arms,
        "matched_curve_row_count": len(matched_curves),
        "scaled_intervention_row_count": len(scaled_interventions),
        "scaled_match_summary": scaled_match_summary,
        "fingerprint_strata_row_count": len(fingerprint_strata),
        "criteria": criteria,
        "failures": failures,
        "selected_next_step": selected_next_step,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(
        out_dir,
        summary,
        matched_curves,
        scaled_interventions,
        scaled_match_summary,
        fingerprint_strata,
        available_arms,
        criteria,
    )
    return summary


def _available_arms(
    scorecard_rows: list[dict[str, str]],
    sparse_rows: list[dict[str, str]],
    per_token_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    aggregate_by_arm = {row.get("arm", ""): row for row in sparse_rows + scorecard_rows}
    per_token_counts: dict[str, int] = defaultdict(int)
    heldout_counts: dict[str, int] = defaultdict(int)
    for row in per_token_rows:
        arm = row.get("arm", "")
        per_token_counts[arm] += 1
        if row.get("split") == "heldout":
            heldout_counts[arm] += 1

    ordered = list(dict.fromkeys(list(REQUIRED_ARMS) + list(SPARSE_ARMS) + list(NULL_ARMS) + sorted(aggregate_by_arm)))
    rows: list[dict[str, Any]] = []
    for arm in ordered:
        aggregate = aggregate_by_arm.get(arm, {})
        if not aggregate and not per_token_counts.get(arm):
            continue
        rows.append(
            {
                "arm": arm,
                "family": aggregate.get("family", ""),
                "per_token_rows": per_token_counts.get(arm, 0),
                "heldout_rows": heldout_counts.get(arm, 0),
                "aggregate_ce_loss": _float_or_blank(aggregate.get("ce_loss") or aggregate.get("heldout_ce_loss")),
                "aggregate_residual_l2": _float_or_blank(aggregate.get("residual_l2") or aggregate.get("heldout_residual_update_l2")),
                "anchor_kl_or_logit_mse": _float_or_blank(aggregate.get("anchor_kl_or_logit_mse")),
                "functional_churn": _float_or_blank(aggregate.get("functional_churn")),
                "retention_or_forgetting": _float_or_blank(aggregate.get("retention_or_forgetting")),
                "intervention_fingerprint_purity": _float_or_blank(aggregate.get("intervention_fingerprint_purity")),
                "role": _arm_role(arm),
            }
        )
    return rows


def _matched_curves(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for arm in REQUIRED_ARMS:
        heldout = [row for row in rows if row.get("arm") == arm and row.get("split") == "heldout"]
        values = sorted(_float(row.get("residual_update_l2")) for row in heldout)
        values = [value for value in values if value is not None]
        if not values:
            continue
        thresholds = _thresholds(values)
        for threshold in thresholds:
            bucket = [
                row
                for row in heldout
                if (_float(row.get("residual_update_l2")) is not None and _float(row.get("residual_update_l2")) <= threshold)
            ]
            out.append(
                {
                    "arm": arm,
                    "match_type": "proxy_residual_l2_prefix",
                    "threshold": threshold,
                    "row_count": len(bucket),
                    "ce_loss": _mean(bucket, "ce_loss"),
                    "delta_vs_base_ce": _mean(bucket, "delta_vs_base_ce"),
                    "residual_update_l2": _mean(bucket, "residual_update_l2"),
                    "logit_mse_vs_base": _mean(bucket, "logit_mse_vs_base"),
                    "prediction_changed_vs_base": _mean_bool(bucket, "prediction_changed_vs_base"),
                }
            )
    return out


def _fingerprint_strata(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        arm = row.get("arm", "")
        if arm not in REQUIRED_ARMS:
            continue
        grouped[(arm, "split", row.get("split", ""))].append(row)
        grouped[(arm, "residual_gain_bin", _gain_bin(_float(row.get("delta_vs_base_ce"))))].append(row)
        grouped[(arm, "residual_l2_bin", _l2_bin(_float(row.get("residual_update_l2"))))].append(row)
        grouped[(arm, "prediction_changed", str(_bool_or_none(row.get("prediction_changed_vs_base"))))].append(row)

    strata: list[dict[str, Any]] = []
    for (arm, stratum_type, stratum), bucket in sorted(grouped.items()):
        strata.append(
            {
                "arm": arm,
                "stratum_type": stratum_type,
                "stratum": stratum,
                "row_count": len(bucket),
                "ce_loss": _mean(bucket, "ce_loss"),
                "delta_vs_base_ce": _mean(bucket, "delta_vs_base_ce"),
                "residual_update_l2": _mean(bucket, "residual_update_l2"),
                "logit_mse_vs_base": _mean(bucket, "logit_mse_vs_base"),
                "prediction_changed_vs_base": _mean_bool(bucket, "prediction_changed_vs_base"),
                "improvement_purity_proxy": _improvement_purity(bucket),
            }
        )
    return strata


def _scaled_interventions(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    lambdas = (0.0, 0.25, 0.5, 0.75, 1.0)
    out: list[dict[str, Any]] = []
    for arm in REQUIRED_ARMS:
        heldout = [row for row in rows if row.get("arm") == arm and row.get("split") == "heldout"]
        for lam in lambdas:
            scaled_rows = [_scaled_row(row, lam) for row in heldout]
            valid = [row for row in scaled_rows if row]
            out.append(
                {
                    "arm": arm,
                    "lambda": lam,
                    "row_count": len(valid),
                    "target_inference_failures": len(scaled_rows) - len(valid),
                    "ce_loss": _mean_any(valid, "scaled_ce_loss"),
                    "delta_vs_base_ce": _mean_any(valid, "scaled_delta_vs_base_ce"),
                    "residual_update_l2": _mean_any(valid, "scaled_residual_update_l2"),
                    "logit_mse_vs_base": _mean_any(valid, "scaled_logit_mse_vs_base"),
                    "prediction_changed_vs_base": _mean_bool_any(valid, "scaled_prediction_changed_vs_base"),
                }
            )
    return out


def _scaled_row(row: dict[str, str], lam: float) -> dict[str, Any] | None:
    base_logits = _json_float_list(row.get("base_logits"))
    candidate_logits = _json_float_list(row.get("candidate_logits"))
    base_ce = _float(row.get("base_ce_loss"))
    raw_l2 = _float(row.get("residual_update_l2"))
    if not base_logits or not candidate_logits or len(base_logits) != len(candidate_logits) or base_ce is None:
        return None
    target_index = _infer_target_index(base_logits, base_ce)
    if target_index is None:
        return None
    scaled_logits = [
        base + lam * (candidate - base)
        for base, candidate in zip(base_logits, candidate_logits)
    ]
    scaled_ce = _ce_for_target(scaled_logits, target_index)
    base_argmax = _argmax(base_logits)
    scaled_argmax = _argmax(scaled_logits)
    logit_mse = sum((scaled - base) ** 2 for scaled, base in zip(scaled_logits, base_logits)) / len(base_logits)
    return {
        "scaled_ce_loss": scaled_ce,
        "scaled_delta_vs_base_ce": scaled_ce - base_ce,
        "scaled_residual_update_l2": "" if raw_l2 is None else abs(lam) * raw_l2,
        "scaled_logit_mse_vs_base": logit_mse,
        "scaled_prediction_changed_vs_base": scaled_argmax != base_argmax,
    }


def _scaled_match_summary(scaled_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_arm = {
        arm: [row for row in scaled_rows if row.get("arm") == arm and int(row.get("row_count") or 0) > 0]
        for arm in REQUIRED_ARMS
    }
    refs = [row for arm in DENSE_ARMS for row in by_arm.get(arm, []) if _float(row.get("lambda")) == 1.0]
    out: list[dict[str, Any]] = []
    for ref in refs:
        ref_arm = str(ref.get("arm", ""))
        ref_l2 = _float(ref.get("residual_update_l2"))
        ref_ce = _float(ref.get("ce_loss"))
        if ref_l2 is None or ref_ce is None:
            continue
        for arm, candidates in by_arm.items():
            l2_match = _closest_row(candidates, "residual_update_l2", ref_l2)
            ce_match = _closest_row(candidates, "ce_loss", ref_ce)
            if l2_match:
                out.append(_match_row("residual_l2", ref_arm, ref_l2, ref_ce, arm, l2_match))
            if ce_match:
                out.append(_match_row("ce_loss", ref_arm, ref_l2, ref_ce, arm, ce_match))
    return out


def _match_row(
    match_type: str,
    reference_arm: str,
    reference_l2: float,
    reference_ce: float,
    arm: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "match_type": match_type,
        "reference_arm": reference_arm,
        "reference_residual_l2": reference_l2,
        "reference_ce_loss": reference_ce,
        "arm": arm,
        "lambda": row.get("lambda"),
        "ce_loss": row.get("ce_loss"),
        "delta_vs_base_ce": row.get("delta_vs_base_ce"),
        "residual_update_l2": row.get("residual_update_l2"),
        "logit_mse_vs_base": row.get("logit_mse_vs_base"),
        "prediction_changed_vs_base": row.get("prediction_changed_vs_base"),
        "distance": abs((_float(row.get("residual_update_l2")) or 0.0) - reference_l2)
        if match_type == "residual_l2"
        else abs((_float(row.get("ce_loss")) or 0.0) - reference_ce),
    }


def _missing_required_fields(rows: list[dict[str, str]]) -> dict[str, Any]:
    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())
    missing_proxy = sorted(field for field in REQUIRED_PROXY_PER_TOKEN_FIELDS if field not in fieldnames)
    missing_raw = sorted(field for field in REQUIRED_RAW_INTERVENTION_FIELDS if field not in fieldnames)
    by_arm: dict[str, Any] = {}
    for arm in REQUIRED_ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        arm_fields = set()
        for row in arm_rows:
            arm_fields.update(key for key, value in row.items() if value != "")
        by_arm[arm] = {
            "rows": len(arm_rows),
            "missing_proxy_fields": sorted(field for field in REQUIRED_PROXY_PER_TOKEN_FIELDS if field not in arm_fields),
            "missing_raw_intervention_fields": sorted(field for field in REQUIRED_RAW_INTERVENTION_FIELDS if field not in arm_fields),
        }
    return {
        "proxy_per_token_fields": missing_proxy,
        "raw_intervention_fields": missing_raw,
        "by_arm": by_arm,
    }


def _criteria(
    *,
    followup_summary: dict[str, Any],
    per_token_rows: list[dict[str, str]],
    available_arms: list[dict[str, Any]],
    missing_fields: dict[str, Any],
    scaled_interventions: list[dict[str, Any]],
    scaled_match_summary: list[dict[str, Any]],
    min_heldout_rows_per_arm: int,
) -> list[dict[str, Any]]:
    arms = {row["arm"]: row for row in available_arms}
    heldout_counts = {
        arm: len([row for row in per_token_rows if row.get("arm") == arm and row.get("split") == "heldout"])
        for arm in REQUIRED_ARMS
    }
    scaled_counts = {
        arm: max(
            [int(row.get("row_count") or 0) for row in scaled_interventions if row.get("arm") == arm],
            default=0,
        )
        for arm in REQUIRED_ARMS
    }
    scaled_failures = {
        arm: sum(
            int(row.get("target_inference_failures") or 0)
            for row in scaled_interventions
            if row.get("arm") == arm
        )
        for arm in REQUIRED_ARMS
    }
    return [
        _criterion(
            "mlp_dense_followup_passed",
            followup_summary.get("status") == "pass" and followup_summary.get("primary_arm") == MLP_ARM,
            "prior MLP dense heldout follow-up selected the MLP control",
            {"status": followup_summary.get("status"), "primary_arm": followup_summary.get("primary_arm")},
            "prior MLP dense heldout follow-up is missing, failed, or did not select MLP",
        ),
        _criterion(
            "required_dense_mlp_arms_present",
            all(arm in arms for arm in REQUIRED_ARMS),
            "dense rank16, dense rank24, and MLP arms are available",
            sorted(arms),
            "one or more required dense/MLP arms is missing",
        ),
        _criterion(
            "sparse_and_null_comparators_listed",
            any(arm in arms for arm in SPARSE_ARMS) and any(arm in arms for arm in NULL_ARMS),
            "sparse ACSR and null comparator aggregate rows are listed when available",
            sorted(arm for arm in arms if arm in set(SPARSE_ARMS + NULL_ARMS)),
            "sparse or null comparator aggregate rows are unavailable",
        ),
        _criterion(
            "heldout_proxy_rows_present",
            all(count >= min_heldout_rows_per_arm for count in heldout_counts.values()),
            f"each required arm has >= {min_heldout_rows_per_arm} heldout proxy rows",
            heldout_counts,
            "one or more required arms lacks heldout proxy rows",
        ),
        _criterion(
            "proxy_per_token_fields_present",
            not missing_fields["proxy_per_token_fields"],
            "proxy per-token CE/L2/churn fields are present",
            missing_fields["proxy_per_token_fields"],
            "proxy per-token fields are missing",
        ),
        _criterion(
            "raw_intervention_fields_present",
            not missing_fields["raw_intervention_fields"],
            "raw residual update vectors and logits are present for scaling/intervention",
            missing_fields["raw_intervention_fields"],
            "raw residual/logit fields are missing, so CE/L2-matched intervention scaling cannot be reconstructed",
        ),
        _criterion(
            "scaled_lambda_interventions_reconstructed",
            all(count >= min_heldout_rows_per_arm for count in scaled_counts.values())
            and all(count == 0 for count in scaled_failures.values()),
            f"each required arm has >= {min_heldout_rows_per_arm} heldout rows with inferred targets across lambda scaling",
            {"scaled_counts": scaled_counts, "target_inference_failures": scaled_failures},
            "lambda-scaled CE reconstruction failed for one or more required arms",
        ),
        _criterion(
            "scaled_match_summary_written",
            bool(scaled_match_summary),
            "scaled rows support nearest CE and residual-L2 matching summaries",
            {"rows": len(scaled_match_summary)},
            "scaled CE/L2 matching summary is empty",
        ),
    ]


def _criterion(
    criterion: str,
    passed: bool,
    threshold: Any,
    actual: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": actual,
        "failure_reason": "" if passed else failure_reason,
    }


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    matched_curves: list[dict[str, Any]],
    scaled_interventions: list[dict[str, Any]],
    scaled_match_summary: list[dict[str, Any]],
    fingerprint_strata: list[dict[str, Any]],
    available_arms: list[dict[str, Any]],
    criteria: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "matched_curves.csv", matched_curves)
    _write_csv(out_dir / "scaled_interventions.csv", scaled_interventions)
    _write_csv(out_dir / "scaled_match_summary.csv", scaled_match_summary)
    _write_csv(out_dir / "fingerprint_strata.csv", fingerprint_strata)
    _write_csv(out_dir / "available_arms.csv", available_arms)
    _write_csv(out_dir / "decision_criteria.csv", criteria)
    lines = [
        "# MLP Churn Intervention Fingerprint",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Promotion allowed: `{summary['promotion_allowed']}`",
        f"- Matched proxy rows: `{summary['matched_curve_row_count']}`",
        f"- Scaled intervention rows: `{summary['scaled_intervention_row_count']}`",
        f"- Fingerprint strata rows: `{summary['fingerprint_strata_row_count']}`",
        f"- Next step: {summary['selected_next_step']}",
        "",
        "This report follows the external review by checking whether the MLP advantage is confounded by residual norm and churn. When raw residual update vectors and logits are available, it reconstructs lambda-scaled logit interventions by inferring the target class from base CE.",
    ]
    if summary["failures"]:
        lines.extend(["", "## Failures"])
        lines.extend(f"- `{row['criterion']}`: {row['failure_reason']}" for row in summary["failures"])
    (out_dir / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _thresholds(values: list[float]) -> list[float]:
    if not values:
        return []
    indexes = sorted({0, len(values) // 4, len(values) // 2, (3 * len(values)) // 4, len(values) - 1})
    return [values[index] for index in indexes]


def _mean(rows: list[dict[str, str]], field: str) -> Any:
    values = [_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return ""
    return sum(values) / len(values)


def _mean_any(rows: list[dict[str, Any]], field: str) -> Any:
    values = [_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return ""
    return sum(values) / len(values)


def _mean_bool(rows: list[dict[str, str]], field: str) -> Any:
    values = [_bool_or_none(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return ""
    return sum(1.0 if value else 0.0 for value in values) / len(values)


def _mean_bool_any(rows: list[dict[str, Any]], field: str) -> Any:
    values = [_bool_or_none(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return ""
    return sum(1.0 if value else 0.0 for value in values) / len(values)


def _improvement_purity(rows: list[dict[str, str]]) -> Any:
    gains = [max(0.0, -value) for value in (_float(row.get("delta_vs_base_ce")) for row in rows) if value is not None]
    damages = [max(0.0, value) for value in (_float(row.get("delta_vs_base_ce")) for row in rows) if value is not None]
    denom = sum(gains) + sum(damages)
    if denom <= 0:
        return ""
    return sum(gains) / denom


def _gain_bin(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < -0.5:
        return "large_gain"
    if value < 0.0:
        return "small_gain"
    return "damage_or_no_gain"


def _l2_bin(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 0.5:
        return "low"
    if value < 1.5:
        return "mid"
    return "high"


def _arm_role(arm: str) -> str:
    if arm == MLP_ARM:
        return "mlp_primary"
    if arm in DENSE_ARMS:
        return "dense_control"
    if arm in SPARSE_ARMS:
        return "sparse_acsr_comparator"
    if arm in NULL_ARMS:
        return "null_comparator"
    return "other"


def _float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_blank(value: Any) -> Any:
    parsed = _float(value)
    return "" if parsed is None else parsed


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in ("", None):
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def _json_float_list(value: Any) -> list[float]:
    if value in ("", None):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    out: list[float] = []
    for item in parsed:
        number = _float(item)
        if number is None:
            return []
        out.append(number)
    return out


def _infer_target_index(logits: list[float], ce_loss: float, *, tolerance: float = 1e-4) -> int | None:
    losses = [_ce_for_target(logits, index) for index in range(len(logits))]
    best_index = min(range(len(losses)), key=lambda index: abs(losses[index] - ce_loss))
    return best_index if abs(losses[best_index] - ce_loss) <= tolerance else None


def _ce_for_target(logits: list[float], target_index: int) -> float:
    max_logit = max(logits)
    log_sum_exp = max_logit + math.log(sum(math.exp(value - max_logit) for value in logits))
    return log_sum_exp - logits[target_index]


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def _closest_row(rows: list[dict[str, Any]], field: str, target: float) -> dict[str, Any]:
    candidates = [row for row in rows if _float(row.get(field)) is not None]
    if not candidates:
        return {}
    return min(candidates, key=lambda row: abs((_float(row.get(field)) or 0.0) - target))


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--followup-dir", type=Path, default=DEFAULT_FOLLOWUP_DIR)
    parser.add_argument("--dense-observables-dir", type=Path, default=DEFAULT_DENSE_OBSERVABLES_DIR)
    parser.add_argument("--sparse-gate-dir", type=Path, default=DEFAULT_SPARSE_GATE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-heldout-rows-per-arm", type=int, default=16)
    args = parser.parse_args()
    summary = run_mlp_churn_intervention_fingerprint(
        followup_dir=args.followup_dir,
        dense_observables_dir=args.dense_observables_dir,
        sparse_gate_dir=args.sparse_gate_dir,
        out_dir=args.out,
        min_heldout_rows_per_arm=args.min_heldout_rows_per_arm,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
