"""Joined sparse/dense/MLP CE-L2-churn matched intervention decision report."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from relaleap.experiments.mlp_churn_intervention_fingerprint import (
    _argmax,
    _ce_for_target,
    _float,
    _infer_target_index,
)


DEFAULT_COMMON_BENCHMARK_DIR = Path("results/reports/acsr_common_causal_residual_benchmark")
DEFAULT_DENSE_OBSERVABLES_DIR = Path("results/reports/acsr_dense_mechanism_observables")
DEFAULT_MLP_FINGERPRINT_DIR = Path("results/reports/mlp_churn_intervention_fingerprint")
DEFAULT_SPARSE_FINGERPRINT_DIR = Path("results/reports/sparse_acsr_per_token_churn_fingerprint")
DEFAULT_OUT_DIR = Path("results/reports/sparse_dense_mlp_matched_intervention_decision")

SPARSE_ARMS = (
    "sparse_contextual_topk2",
    "sparse_rank_matched_topk1",
    "sparse_teacher_distilled_norm_topk2",
    "sparse_frequency_matched_random_topk1",
)
DENSE_ARMS = ("dense_rank16_best_norm", "dense_rank24_best_norm")
MLP_ARM = "parameter_matched_causal_mlp_control"
REQUIRED_ARMS = SPARSE_ARMS + DENSE_ARMS + (MLP_ARM,)
REFERENCE_ARMS = DENSE_ARMS
CHALLENGER_ARMS = SPARSE_ARMS + (MLP_ARM,)
REQUIRED_RAW_FIELDS = (
    "base_ce_loss",
    "base_logits",
    "candidate_logits",
    "residual_update_l2",
    "logit_mse_vs_base",
    "prediction_changed_vs_base",
)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "available_arms.csv",
    "scaled_interventions.csv",
    "matched_comparisons.csv",
    "pareto_frontier.csv",
    "domination_cases.csv",
    "decision_criteria.csv",
    "notes.md",
)


def run_sparse_dense_mlp_matched_intervention_decision(
    *,
    common_benchmark_dir: Path = DEFAULT_COMMON_BENCHMARK_DIR,
    dense_observables_dir: Path = DEFAULT_DENSE_OBSERVABLES_DIR,
    mlp_fingerprint_dir: Path = DEFAULT_MLP_FINGERPRINT_DIR,
    sparse_fingerprint_dir: Path = DEFAULT_SPARSE_FINGERPRINT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    min_heldout_rows_per_arm: int = 16,
) -> dict[str, Any]:
    """Join sparse/dense/MLP raw rows and write matched CE/L2 interference evidence."""

    start = time.time()
    sparse_rows = _tag_family(_read_csv(common_benchmark_dir / "per_token_metrics.csv"), "sparse")
    dense_rows = _tag_family(_read_csv(dense_observables_dir / "per_token_observables.csv"), "dense_mlp")
    rows = [row for row in sparse_rows + dense_rows if row.get("arm") in REQUIRED_ARMS]
    mlp_summary = _read_json(mlp_fingerprint_dir / "summary.json")
    sparse_summary = _read_json(sparse_fingerprint_dir / "summary.json")

    available_arms = _available_arms(rows)
    scaled_interventions = _scaled_interventions(rows)
    matched_comparisons = _matched_comparisons(scaled_interventions)
    pareto_frontier = _pareto_frontier(scaled_interventions)
    domination_cases = _domination_cases(scaled_interventions, pareto_frontier)
    criteria = _criteria(
        mlp_summary=mlp_summary,
        sparse_summary=sparse_summary,
        available_arms=available_arms,
        scaled_interventions=scaled_interventions,
        matched_comparisons=matched_comparisons,
        min_heldout_rows_per_arm=min_heldout_rows_per_arm,
    )
    failures = [row for row in criteria if not row["passed"]]
    advancement_rows = [row for row in domination_cases if row["challenger_advances"]]

    if failures:
        status = "fail"
        decision = "sparse_dense_mlp_matched_intervention_decision_failed_closed"
        claim_status = "joined_sparse_dense_mlp_matching_inputs_incomplete"
        selected_next_step = "repair missing raw heldout rows before interpreting sparse/dense/MLP CE-L2 matches"
    elif advancement_rows:
        status = "pass"
        decision = "matched_intervention_challenger_clears_best_dense_pareto_guardrail"
        claim_status = "one_or_more_sparse_or_mlp_arms_beat_best_dense_pareto_without_extra_churn_proxy"
        selected_next_step = "review matched_comparisons.csv and decide whether to train a norm-budgeted challenger"
    else:
        status = "pass"
        decision = "matched_intervention_challengers_do_not_clear_best_dense_pareto_guardrail"
        claim_status = "mlp_or_sparse_advantage_not_decisive_after_ce_l2_churn_matching"
        selected_next_step = "design a norm-budgeted churn-regularized MLP or dense-teacher sparse-column variant locally"

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "promotion_allowed": False,
        "requires_gpu_now": False,
        "source_dirs": {
            "common_benchmark": str(common_benchmark_dir),
            "dense_observables": str(dense_observables_dir),
            "mlp_fingerprint": str(mlp_fingerprint_dir),
            "sparse_fingerprint": str(sparse_fingerprint_dir),
        },
        "available_arm_count": len(available_arms),
        "scaled_intervention_row_count": len(scaled_interventions),
        "matched_comparison_row_count": len(matched_comparisons),
        "pareto_frontier_row_count": len(pareto_frontier),
        "domination_case_row_count": len(domination_cases),
        "advancement_row_count": len(advancement_rows),
        "scientific_gate": "pass" if advancement_rows and not failures else "blocked",
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
        available_arms,
        scaled_interventions,
        matched_comparisons,
        pareto_frontier,
        domination_cases,
        criteria,
    )
    return summary


def _available_arms(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for arm in REQUIRED_ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        heldout = [row for row in arm_rows if row.get("split") == "heldout"]
        present_fields = {key for row in heldout for key, value in row.items() if value != ""}
        out.append(
            {
                "arm": arm,
                "family": _family_for_arm(arm),
                "rows": len(arm_rows),
                "heldout_rows": len(heldout),
                "missing_raw_fields": ";".join(field for field in REQUIRED_RAW_FIELDS if field not in present_fields),
                "heldout_ce_loss": _mean(heldout, "ce_loss"),
                "heldout_delta_vs_base_ce": _mean(heldout, "delta_vs_base_ce"),
                "heldout_residual_update_l2": _mean(heldout, "residual_update_l2"),
                "heldout_logit_mse_vs_base": _mean(heldout, "logit_mse_vs_base"),
                "heldout_prediction_changed_vs_base": _mean_bool(heldout, "prediction_changed_vs_base"),
            }
        )
    return out


def _scaled_interventions(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    lambdas = (0.0, 0.25, 0.5, 0.75, 1.0)
    out: list[dict[str, Any]] = []
    for arm in REQUIRED_ARMS:
        heldout = [row for row in rows if row.get("arm") == arm and row.get("split") == "heldout"]
        for lam in lambdas:
            valid = [_scaled_row(row, lam) for row in heldout]
            valid = [row for row in valid if row]
            out.append(
                {
                    "arm": arm,
                    "family": _family_for_arm(arm),
                    "lambda": lam,
                    "row_count": len(valid),
                    "target_inference_failures": len(heldout) - len(valid),
                    "ce_loss": _mean_any(valid, "ce_loss"),
                    "delta_vs_base_ce": _mean_any(valid, "delta_vs_base_ce"),
                    "residual_update_l2": _mean_any(valid, "residual_update_l2"),
                    "logit_mse_vs_base": _mean_any(valid, "logit_mse_vs_base"),
                    "prediction_changed_vs_base": _mean_bool_any(valid, "prediction_changed_vs_base"),
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
    scaled_logits = [base + lam * (candidate - base) for base, candidate in zip(base_logits, candidate_logits)]
    ce_loss = _ce_for_target(scaled_logits, target_index)
    base_argmax = _argmax(base_logits)
    scaled_argmax = _argmax(scaled_logits)
    return {
        "ce_loss": ce_loss,
        "delta_vs_base_ce": ce_loss - base_ce,
        "residual_update_l2": "" if raw_l2 is None else abs(lam) * raw_l2,
        "logit_mse_vs_base": sum((scaled - base) ** 2 for scaled, base in zip(scaled_logits, base_logits)) / len(base_logits),
        "prediction_changed_vs_base": scaled_argmax != base_argmax,
    }


def _matched_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_arm = defaultdict(list)
    for row in rows:
        if int(row.get("row_count") or 0) > 0:
            by_arm[row["arm"]].append(row)
    out: list[dict[str, Any]] = []
    for reference_arm in REFERENCE_ARMS:
        reference = _row_at_lambda(by_arm[reference_arm], 1.0)
        if not reference:
            continue
        for challenger in CHALLENGER_ARMS:
            for match_type, field in (("residual_l2", "residual_update_l2"), ("ce_loss", "ce_loss")):
                matched = _closest_row(by_arm[challenger], field, _float(reference.get(field)))
                if matched:
                    out.append(_comparison_row(match_type, reference_arm, reference, challenger, matched, field))
    return out


def _pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dense_rows = [
        row for row in rows
        if row.get("arm") in DENSE_ARMS and int(row.get("row_count") or 0) > 0
    ]
    frontier: list[dict[str, Any]] = []
    for row in dense_rows:
        dominated_by = _dominating_dense(row, dense_rows)
        if not dominated_by:
            frontier.append(
                {
                    "arm": row.get("arm"),
                    "lambda": row.get("lambda"),
                    "ce_loss": row.get("ce_loss"),
                    "residual_update_l2": row.get("residual_update_l2"),
                    "logit_mse_vs_base": row.get("logit_mse_vs_base"),
                    "prediction_changed_vs_base": row.get("prediction_changed_vs_base"),
                    "row_count": row.get("row_count"),
                }
            )
    return frontier


def _domination_cases(
    rows: list[dict[str, Any]],
    pareto_frontier: list[dict[str, Any]],
    *,
    l2_tolerance: float = 0.15,
) -> list[dict[str, Any]]:
    frontier_rows = [row for row in pareto_frontier if _float(row.get("residual_update_l2")) is not None]
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("arm") not in CHALLENGER_ARMS or int(row.get("row_count") or 0) <= 0:
            continue
        challenger_l2 = _float(row.get("residual_update_l2"))
        if challenger_l2 is None:
            continue
        best_dense = _best_dense_within_l2(frontier_rows, challenger_l2, l2_tolerance)
        if not best_dense:
            continue
        l2_distance = abs(challenger_l2 - (_float(best_dense.get("residual_update_l2")) or 0.0))
        ce_delta = _difference(row.get("ce_loss"), best_dense.get("ce_loss"))
        mse_delta = _difference(row.get("logit_mse_vs_base"), best_dense.get("logit_mse_vs_base"))
        flip_delta = _difference(row.get("prediction_changed_vs_base"), best_dense.get("prediction_changed_vs_base"))
        dominated = (
            l2_distance <= l2_tolerance
            and ce_delta != ""
            and mse_delta != ""
            and flip_delta != ""
            and float(ce_delta) >= 0.0
            and float(mse_delta) >= 0.0
            and float(flip_delta) >= 0.0
        )
        advances = (
            l2_distance <= l2_tolerance
            and ce_delta != ""
            and mse_delta != ""
            and flip_delta != ""
            and float(ce_delta) < 0.0
            and float(mse_delta) <= 0.0
            and float(flip_delta) <= 0.0
        )
        out.append(
            {
                "challenger_arm": row.get("arm"),
                "challenger_lambda": row.get("lambda"),
                "challenger_ce_loss": row.get("ce_loss"),
                "challenger_residual_update_l2": row.get("residual_update_l2"),
                "challenger_logit_mse_vs_base": row.get("logit_mse_vs_base"),
                "challenger_prediction_changed_vs_base": row.get("prediction_changed_vs_base"),
                "best_dense_arm": best_dense.get("arm"),
                "best_dense_lambda": best_dense.get("lambda"),
                "best_dense_ce_loss": best_dense.get("ce_loss"),
                "best_dense_residual_update_l2": best_dense.get("residual_update_l2"),
                "best_dense_logit_mse_vs_base": best_dense.get("logit_mse_vs_base"),
                "best_dense_prediction_changed_vs_base": best_dense.get("prediction_changed_vs_base"),
                "l2_distance": l2_distance,
                "within_l2_tolerance": l2_distance <= l2_tolerance,
                "ce_delta_challenger_minus_best_dense": ce_delta,
                "logit_mse_delta_challenger_minus_best_dense": mse_delta,
                "flip_delta_challenger_minus_best_dense": flip_delta,
                "challenger_dominated_by_best_dense": dominated,
                "challenger_advances": advances,
            }
        )
    return out


def _best_dense_within_l2(
    rows: list[dict[str, Any]],
    challenger_l2: float,
    l2_tolerance: float,
) -> dict[str, Any]:
    candidates = [
        row for row in rows
        if _float(row.get("residual_update_l2")) is not None
        and abs(challenger_l2 - (_float(row.get("residual_update_l2")) or 0.0)) <= l2_tolerance
        and _float(row.get("ce_loss")) is not None
    ]
    if not candidates:
        return {}
    return min(
        candidates,
        key=lambda row: (
            _float(row.get("ce_loss")) or float("inf"),
            _float(row.get("logit_mse_vs_base")) or float("inf"),
            _float(row.get("prediction_changed_vs_base")) or float("inf"),
        ),
    )


def _dominating_dense(row: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    row_values = [_float(row.get(field)) for field in ("ce_loss", "residual_update_l2", "logit_mse_vs_base", "prediction_changed_vs_base")]
    if any(value is None for value in row_values):
        return {}
    for candidate in candidates:
        if candidate is row:
            continue
        candidate_values = [
            _float(candidate.get(field))
            for field in ("ce_loss", "residual_update_l2", "logit_mse_vs_base", "prediction_changed_vs_base")
        ]
        if any(value is None for value in candidate_values):
            continue
        no_worse = all(candidate <= current for candidate, current in zip(candidate_values, row_values))
        strictly_better = any(candidate < current for candidate, current in zip(candidate_values, row_values))
        if no_worse and strictly_better:
            return candidate
    return {}


def _comparison_row(
    match_type: str,
    reference_arm: str,
    reference: dict[str, Any],
    challenger_arm: str,
    challenger: dict[str, Any],
    field: str,
) -> dict[str, Any]:
    reference_value = _float(reference.get(field))
    challenger_value = _float(challenger.get(field))
    return {
        "match_type": match_type,
        "reference_arm": reference_arm,
        "reference_lambda": reference.get("lambda"),
        "reference_ce_loss": reference.get("ce_loss"),
        "reference_residual_update_l2": reference.get("residual_update_l2"),
        "reference_logit_mse_vs_base": reference.get("logit_mse_vs_base"),
        "reference_prediction_changed_vs_base": reference.get("prediction_changed_vs_base"),
        "challenger_arm": challenger_arm,
        "challenger_lambda": challenger.get("lambda"),
        "challenger_ce_loss": challenger.get("ce_loss"),
        "challenger_residual_update_l2": challenger.get("residual_update_l2"),
        "challenger_logit_mse_vs_base": challenger.get("logit_mse_vs_base"),
        "challenger_prediction_changed_vs_base": challenger.get("prediction_changed_vs_base"),
        "ce_delta_challenger_minus_reference": _difference(challenger.get("ce_loss"), reference.get("ce_loss")),
        "l2_delta_challenger_minus_reference": _difference(challenger.get("residual_update_l2"), reference.get("residual_update_l2")),
        "logit_mse_delta_challenger_minus_reference": _difference(challenger.get("logit_mse_vs_base"), reference.get("logit_mse_vs_base")),
        "flip_delta_challenger_minus_reference": _difference(challenger.get("prediction_changed_vs_base"), reference.get("prediction_changed_vs_base")),
        "match_distance": "" if reference_value is None or challenger_value is None else abs(challenger_value - reference_value),
    }


def _criteria(
    *,
    mlp_summary: dict[str, Any],
    sparse_summary: dict[str, Any],
    available_arms: list[dict[str, Any]],
    scaled_interventions: list[dict[str, Any]],
    matched_comparisons: list[dict[str, Any]],
    min_heldout_rows_per_arm: int,
) -> list[dict[str, Any]]:
    by_arm = {row["arm"]: row for row in available_arms}
    scaled_counts = {
        arm: max([int(row.get("row_count") or 0) for row in scaled_interventions if row.get("arm") == arm], default=0)
        for arm in REQUIRED_ARMS
    }
    inference_failures = {
        arm: sum(int(row.get("target_inference_failures") or 0) for row in scaled_interventions if row.get("arm") == arm)
        for arm in REQUIRED_ARMS
    }
    return [
        _criterion(
            "mlp_fingerprint_passed",
            mlp_summary.get("status") == "pass",
            "MLP raw lambda-scaled fingerprint report passed",
            mlp_summary.get("decision"),
            "MLP fingerprint source is missing or failed",
        ),
        _criterion(
            "sparse_fingerprint_passed",
            sparse_summary.get("status") == "pass",
            "sparse raw/churn fingerprint coverage report passed",
            sparse_summary.get("decision"),
            "sparse fingerprint source is missing or failed",
        ),
        _criterion(
            "required_arms_have_raw_rows",
            all(
                int(by_arm.get(arm, {}).get("heldout_rows") or 0) >= min_heldout_rows_per_arm
                and not by_arm.get(arm, {}).get("missing_raw_fields")
                for arm in REQUIRED_ARMS
            ),
            f"each required arm has >= {min_heldout_rows_per_arm} heldout rows and raw fields",
            {arm: by_arm.get(arm, {}) for arm in REQUIRED_ARMS},
            "one or more sparse/dense/MLP arms lacks heldout raw fields",
        ),
        _criterion(
            "scaled_targets_reconstructed",
            all(scaled_counts[arm] >= min_heldout_rows_per_arm and inference_failures[arm] == 0 for arm in REQUIRED_ARMS),
            "lambda-scaled CE can be reconstructed for every required arm",
            {"scaled_counts": scaled_counts, "target_inference_failures": inference_failures},
            "target inference or lambda reconstruction failed for one or more arms",
        ),
        _criterion(
            "matched_comparisons_written",
            bool(matched_comparisons),
            "nearest CE and residual-L2 comparisons against dense rank controls are written",
            {"rows": len(matched_comparisons)},
            "matched comparison table is empty",
        ),
    ]


def _criterion(criterion: str, passed: bool, threshold: Any, actual: Any, failure_reason: str) -> dict[str, Any]:
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
    available_arms: list[dict[str, Any]],
    scaled_interventions: list[dict[str, Any]],
    matched_comparisons: list[dict[str, Any]],
    pareto_frontier: list[dict[str, Any]],
    domination_cases: list[dict[str, Any]],
    criteria: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "available_arms.csv", available_arms)
    _write_csv(out_dir / "scaled_interventions.csv", scaled_interventions)
    _write_csv(out_dir / "matched_comparisons.csv", matched_comparisons)
    _write_csv(out_dir / "pareto_frontier.csv", pareto_frontier)
    _write_csv(out_dir / "domination_cases.csv", domination_cases)
    _write_csv(out_dir / "decision_criteria.csv", criteria)
    lines = [
        "# Sparse/Dense/MLP Matched Intervention Decision",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Promotion allowed: `{summary['promotion_allowed']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Matched comparisons: `{summary['matched_comparison_row_count']}`",
        f"- Pareto frontier rows: `{summary['pareto_frontier_row_count']}`",
        f"- Domination cases: `{summary['domination_case_row_count']}`",
        f"- Advancement rows: `{summary['advancement_row_count']}`",
        f"- Scientific gate: `{summary['scientific_gate']}`",
        f"- Next step: {summary['selected_next_step']}",
        "",
        "This report implements the local review recommendation by comparing sparse ACSR, dense rank controls, and the MLP control at nearest residual-L2 and CE operating points reconstructed from raw heldout logits. Artifact pass is separate from the scientific gate; challengers must beat the best dense Pareto comparator without worse logit-MSE or prediction-flip churn proxies.",
    ]
    if summary["failures"]:
        lines.extend(["", "## Failures"])
        lines.extend(f"- `{row['criterion']}`: {row['failure_reason']}" for row in summary["failures"])
    (out_dir / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _tag_family(rows: list[dict[str, str]], family: str) -> list[dict[str, str]]:
    for row in rows:
        row.setdefault("family", family)
    return rows


def _family_for_arm(arm: str) -> str:
    if arm in SPARSE_ARMS:
        return "sparse"
    if arm in DENSE_ARMS:
        return "dense"
    if arm == MLP_ARM:
        return "mlp"
    return "other"


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


def _mean(rows: list[dict[str, str]], field: str) -> Any:
    values = [_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return "" if not values else sum(values) / len(values)


def _mean_any(rows: list[dict[str, Any]], field: str) -> Any:
    values = [_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return "" if not values else sum(values) / len(values)


def _mean_bool(rows: list[dict[str, str]], field: str) -> Any:
    values = [_bool_or_none(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return "" if not values else sum(1.0 if value else 0.0 for value in values) / len(values)


def _mean_bool_any(rows: list[dict[str, Any]], field: str) -> Any:
    values = [_bool_or_none(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return "" if not values else sum(1.0 if value else 0.0 for value in values) / len(values)


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


def _row_at_lambda(rows: list[dict[str, Any]], lam: float) -> dict[str, Any]:
    return next((row for row in rows if _float(row.get("lambda")) == lam), {})


def _closest_row(rows: list[dict[str, Any]], field: str, target: float | None) -> dict[str, Any]:
    if target is None:
        return {}
    candidates = [row for row in rows if _float(row.get(field)) is not None]
    if not candidates:
        return {}
    return min(candidates, key=lambda row: abs((_float(row.get(field)) or 0.0) - target))


def _difference(left: Any, right: Any) -> Any:
    left_value = _float(left)
    right_value = _float(right)
    if left_value is None or right_value is None:
        return ""
    return left_value - right_value


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-benchmark-dir", type=Path, default=DEFAULT_COMMON_BENCHMARK_DIR)
    parser.add_argument("--dense-observables-dir", type=Path, default=DEFAULT_DENSE_OBSERVABLES_DIR)
    parser.add_argument("--mlp-fingerprint-dir", type=Path, default=DEFAULT_MLP_FINGERPRINT_DIR)
    parser.add_argument("--sparse-fingerprint-dir", type=Path, default=DEFAULT_SPARSE_FINGERPRINT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-heldout-rows-per-arm", type=int, default=16)
    args = parser.parse_args()
    summary = run_sparse_dense_mlp_matched_intervention_decision(
        common_benchmark_dir=args.common_benchmark_dir,
        dense_observables_dir=args.dense_observables_dir,
        mlp_fingerprint_dir=args.mlp_fingerprint_dir,
        sparse_fingerprint_dir=args.sparse_fingerprint_dir,
        out_dir=args.out,
        min_heldout_rows_per_arm=args.min_heldout_rows_per_arm,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
