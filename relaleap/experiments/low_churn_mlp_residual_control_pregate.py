"""Design a local low-churn MLP residual-control pregate."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_BRANCH_SELECTOR = Path("results/reports/post_dense_teacher_control_branch_selector/summary.json")
DEFAULT_MLP_FOLLOWUP = Path("results/reports/mlp_dense_heldout_mechanism_followup/summary.json")
DEFAULT_MLP_FINGERPRINT = Path("results/reports/mlp_churn_intervention_fingerprint")
DEFAULT_MLP_DECISION = Path("results/reports/mlp_churn_decision/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/low_churn_mlp_residual_control_pregate")

DENSE24_ARM = "dense_rank24_best_norm"
MLP_ARM = "parameter_matched_causal_mlp_control"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "budget_rows.csv",
    "pregate_arms.csv",
    "gate_criteria.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_low_churn_mlp_residual_control_pregate(
    *,
    branch_selector_path: Path = DEFAULT_BRANCH_SELECTOR,
    mlp_followup_path: Path = DEFAULT_MLP_FOLLOWUP,
    mlp_fingerprint_dir: Path = DEFAULT_MLP_FINGERPRINT,
    mlp_decision_path: Path = DEFAULT_MLP_DECISION,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Record the local pregate contract before any low-churn MLP training."""

    start = time.time()
    branch_selector = _read_json(branch_selector_path)
    mlp_followup = _read_json(mlp_followup_path)
    fingerprint_summary = _read_json(mlp_fingerprint_dir / "summary.json")
    scaled_rows = _read_csv(mlp_fingerprint_dir / "scaled_interventions.csv")
    match_rows = _read_csv(mlp_fingerprint_dir / "scaled_match_summary.csv")
    mlp_decision = _read_json(mlp_decision_path)
    strategy = _strategy_review(strategy_review_path)

    dense24_followup = _mechanism_row(mlp_followup, DENSE24_ARM)
    raw_mlp_followup = _mechanism_row(mlp_followup, MLP_ARM)
    dense24_scaled = _scaled_arm_row(scaled_rows, DENSE24_ARM, 1.0)
    raw_mlp_scaled = _scaled_arm_row(scaled_rows, MLP_ARM, 1.0)
    l2_matched_mlp = _best_match(
        match_rows,
        match_type="residual_l2",
        reference_arm=DENSE24_ARM,
        arm=MLP_ARM,
    )
    budgets = _budget_rows(
        dense24_followup=dense24_followup,
        raw_mlp_followup=raw_mlp_followup,
        dense24_scaled=dense24_scaled,
        raw_mlp_scaled=raw_mlp_scaled,
        l2_matched_mlp=l2_matched_mlp,
    )
    source_rows = _source_rows(
        branch_selector_path=branch_selector_path,
        branch_selector=branch_selector,
        mlp_followup_path=mlp_followup_path,
        mlp_followup=mlp_followup,
        mlp_fingerprint_dir=mlp_fingerprint_dir,
        fingerprint_summary=fingerprint_summary,
        scaled_rows=scaled_rows,
        match_rows=match_rows,
        mlp_decision_path=mlp_decision_path,
        mlp_decision=mlp_decision,
        strategy_review_path=strategy_review_path,
        strategy=strategy,
    )
    pregate_arms = _pregate_arms(budgets)
    criteria = _gate_criteria(
        branch_selector=branch_selector,
        mlp_followup=mlp_followup,
        fingerprint_summary=fingerprint_summary,
        mlp_decision=mlp_decision,
        budgets=budgets,
        scaled_rows=scaled_rows,
        match_rows=match_rows,
        strategy=strategy,
    )
    failures = [row for row in criteria if not row["passed"]]
    status = "pass" if not failures else "fail"
    selected_next_action = (
        "implement_low_churn_mlp_residual_control_pilot"
        if status == "pass"
        else "repair_low_churn_mlp_pregate_sources"
    )
    summary = {
        "status": status,
        "decision": (
            "low_churn_mlp_residual_control_pregate_recorded"
            if status == "pass"
            else "low_churn_mlp_residual_control_pregate_failed_closed"
        ),
        "claim_status": "design_only_no_low_churn_mlp_claim",
        "selected_next_action": selected_next_action,
        "selected_next_step": (
            "implement a local low-churn MLP residual-control pilot using the pregate arms and gates"
            if status == "pass"
            else "repair missing source artifacts before implementing a low-churn MLP pilot"
        ),
        "promotion_allowed": False,
        "requires_gpu_now": False,
        "backend_policy": "local pregate design only; RunPod and Colab remain blocked",
        "source_rows": source_rows,
        "budget_rows": budgets,
        "pregate_arms": pregate_arms,
        "gate_criteria": criteria,
        "candidate_actions": _candidate_actions(selected_next_action),
        "failures": failures,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "rationale": _rationale(status, budgets),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _budget_rows(
    *,
    dense24_followup: dict[str, Any],
    raw_mlp_followup: dict[str, Any],
    dense24_scaled: dict[str, Any],
    raw_mlp_scaled: dict[str, Any],
    l2_matched_mlp: dict[str, Any],
) -> list[dict[str, Any]]:
    dense_l2 = _first_float(
        dense24_scaled.get("residual_update_l2"),
        dense24_followup.get("heldout_residual_update_l2"),
    )
    dense_flip = _first_float(
        dense24_scaled.get("prediction_changed_vs_base"),
        dense24_followup.get("heldout_prediction_changed_vs_base"),
    )
    dense_anchor = _first_float(
        dense24_scaled.get("logit_mse_vs_base"),
        dense24_followup.get("heldout_logit_mse_vs_base"),
    )
    dense_ce = _first_float(dense24_scaled.get("ce_loss"), dense24_followup.get("heldout_ce_loss"))
    raw_l2 = _first_float(
        raw_mlp_scaled.get("residual_update_l2"),
        raw_mlp_followup.get("heldout_residual_update_l2"),
    )
    raw_flip = _first_float(
        raw_mlp_scaled.get("prediction_changed_vs_base"),
        raw_mlp_followup.get("heldout_prediction_changed_vs_base"),
    )
    return [
        _budget("dense24_residual_l2_ceiling", dense_l2, "heldout mean residual_update_l2 must be <= dense24"),
        _budget("dense24_anchor_logit_mse_ceiling", dense_anchor, "heldout anchor logit-MSE/anchor-KL proxy must be <= dense24"),
        _budget("dense24_flip_churn_ceiling", dense_flip, "heldout prediction-flip churn must be <= dense24"),
        _budget("dense24_ce_reference", dense_ce, "low-churn MLP CE is compared against this reference, not promoted on CE alone"),
        _budget("mlp_raw_residual_l2_reference", raw_l2, "raw MLP high-power endpoint reference"),
        _budget("mlp_raw_flip_churn_reference", raw_flip, "raw MLP churn reference"),
        _budget("mlp_l2_matched_lambda_reference", _float(l2_matched_mlp.get("lambda")), "nearest existing scaled MLP dense24-L2 operating point"),
    ]


def _budget(metric: str, value: float | None, role: str) -> dict[str, Any]:
    return {"metric": metric, "value": "" if value is None else value, "role": role}


def _pregate_arms(budgets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dense_l2 = _budget_value(budgets, "dense24_residual_l2_ceiling")
    return [
        {
            "arm": "dense_rank24_reference",
            "family": "dense_control",
            "trainable": False,
            "residual_l2_budget": dense_l2,
            "required_outputs": "heldout_ce; residual_update_l2; anchor_kl_or_logit_mse; prediction_flip_churn; raw_intervention_fingerprint",
            "role": "hard reference budget and advancement comparator",
        },
        {
            "arm": "raw_parameter_matched_mlp_reference",
            "family": "mlp_control",
            "trainable": False,
            "residual_l2_budget": "none",
            "required_outputs": "same metrics as dense reference",
            "role": "high-power upper endpoint, not a promotable candidate",
        },
        {
            "arm": "scaled_mlp_dense24_l2_reference",
            "family": "mlp_control",
            "trainable": False,
            "residual_l2_budget": dense_l2,
            "required_outputs": "same metrics as dense reference",
            "role": "existing post-hoc scaled control that motivates explicit training",
        },
        {
            "arm": "low_churn_mlp_residual_control",
            "family": "mlp_control",
            "trainable": True,
            "residual_l2_budget": dense_l2,
            "required_outputs": "heldout_ce; anchor_kl; flip_churn; residual_update_l2; raw_intervention_fingerprint; dense24-relative pass/fail",
            "role": "next local pilot candidate",
        },
        {
            "arm": "low_churn_mlp_shuffled_target_null",
            "family": "mlp_null",
            "trainable": True,
            "residual_l2_budget": dense_l2,
            "required_outputs": "same metrics as low_churn_mlp_residual_control",
            "role": "raw intervention-fingerprint specificity null",
        },
    ]


def _gate_criteria(
    *,
    branch_selector: dict[str, Any],
    mlp_followup: dict[str, Any],
    fingerprint_summary: dict[str, Any],
    mlp_decision: dict[str, Any],
    budgets: list[dict[str, Any]],
    scaled_rows: list[dict[str, str]],
    match_rows: list[dict[str, str]],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    budget_values = {row["metric"]: row["value"] for row in budgets}
    return [
        _criterion(
            "branch_selector_selected_low_churn_mlp_pregate",
            branch_selector.get("status") == "pass"
            and branch_selector.get("selected_next_action") == "design_low_churn_mlp_residual_control_pregate",
            "post-dense-teacher selector must choose this local pregate",
            branch_selector.get("selected_next_action", "missing"),
            "branch selector did not select low-churn MLP pregate",
        ),
        _criterion(
            "mlp_followup_passed_with_dense_and_mlp_rows",
            mlp_followup.get("status") == "pass"
            and bool(_mechanism_row(mlp_followup, DENSE24_ARM))
            and bool(_mechanism_row(mlp_followup, MLP_ARM)),
            "MLP followup must contain dense24 and raw MLP heldout rows",
            mlp_followup.get("decision", "missing"),
            "MLP followup row coverage is incomplete",
        ),
        _criterion(
            "raw_mlp_is_high_norm_high_churn_vs_dense24",
            _budget_float(budgets, "mlp_raw_residual_l2_reference") is not None
            and _budget_float(budgets, "dense24_residual_l2_ceiling") is not None
            and _budget_float(budgets, "mlp_raw_residual_l2_reference") > _budget_float(budgets, "dense24_residual_l2_ceiling")
            and _budget_float(budgets, "mlp_raw_flip_churn_reference") is not None
            and _budget_float(budgets, "dense24_flip_churn_ceiling") is not None
            and _budget_float(budgets, "mlp_raw_flip_churn_reference") > _budget_float(budgets, "dense24_flip_churn_ceiling"),
            "raw MLP must exceed dense24 residual L2 and flip churn, motivating a low-churn control",
            {
                "dense24_l2": budget_values.get("dense24_residual_l2_ceiling"),
                "raw_mlp_l2": budget_values.get("mlp_raw_residual_l2_reference"),
                "dense24_flip": budget_values.get("dense24_flip_churn_ceiling"),
                "raw_mlp_flip": budget_values.get("mlp_raw_flip_churn_reference"),
            },
            "raw MLP is not a high-norm/high-churn endpoint relative to dense24",
        ),
        _criterion(
            "fingerprint_assay_has_raw_scaled_rows",
            fingerprint_summary.get("status") == "pass" and bool(scaled_rows) and bool(match_rows),
            "raw/scaled intervention fingerprint rows must be available",
            {
                "fingerprint_status": fingerprint_summary.get("status", "missing"),
                "scaled_rows": len(scaled_rows),
                "match_rows": len(match_rows),
            },
            "raw scaled intervention fingerprints are missing",
        ),
        _criterion(
            "dense24_budgets_are_positive",
            all(
                (_budget_float(budgets, metric) is not None and _budget_float(budgets, metric) > 0.0)
                for metric in (
                    "dense24_residual_l2_ceiling",
                    "dense24_anchor_logit_mse_ceiling",
                    "dense24_flip_churn_ceiling",
                )
            ),
            "dense24 L2, anchor/logit-MSE, and flip-churn ceilings must be positive",
            {
                "l2": budget_values.get("dense24_residual_l2_ceiling"),
                "anchor": budget_values.get("dense24_anchor_logit_mse_ceiling"),
                "flip": budget_values.get("dense24_flip_churn_ceiling"),
            },
            "one or more dense24 pregate budgets is missing",
        ),
        _criterion(
            "prior_mlp_decision_blocks_raw_mlp_promotion",
            mlp_decision.get("status") == "pass"
            and mlp_decision.get("promotion_allowed") is False
            and mlp_decision.get("requires_gpu_now") is False,
            "prior MLP decision must keep raw MLP local and unpromoted",
            mlp_decision.get("decision", "missing"),
            "prior MLP decision does not block promotion/GPU",
        ),
        _criterion(
            "strategy_review_consumed_without_gpu",
            strategy["present"] and str(strategy["notify_ben"]).lower() != "true",
            "latest GPT-5.5-Pro review must be read; no notify-Ben escalation for this run",
            f"present={strategy['present']}; notify_ben={strategy['notify_ben']}; recommendation={strategy['recommended_next_action']}",
            "strategy review missing or requires notification handling",
        ),
    ]


def _source_rows(
    *,
    branch_selector_path: Path,
    branch_selector: dict[str, Any],
    mlp_followup_path: Path,
    mlp_followup: dict[str, Any],
    mlp_fingerprint_dir: Path,
    fingerprint_summary: dict[str, Any],
    scaled_rows: list[dict[str, str]],
    match_rows: list[dict[str, str]],
    mlp_decision_path: Path,
    mlp_decision: dict[str, Any],
    strategy_review_path: Path,
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _source("post_dense_teacher_control_branch_selector", branch_selector_path, branch_selector),
        _source("mlp_dense_heldout_mechanism_followup", mlp_followup_path, mlp_followup),
        _source("mlp_churn_intervention_fingerprint", mlp_fingerprint_dir / "summary.json", fingerprint_summary, extra=f"scaled_rows={len(scaled_rows)}; match_rows={len(match_rows)}"),
        _source("mlp_churn_decision", mlp_decision_path, mlp_decision),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": f"strategic_change_level={strategy['strategic_change_level']}; notify_ben={strategy['notify_ben']}; verdict={strategy['verdict']}",
            "extra": "accepted as already satisfied by dense-teacher pair-composer pregate closeout; this report continues local no-GPU work",
        },
    ]


def _source(source: str, path: Path, payload: dict[str, Any], *, extra: str = "") -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status", "missing" if not path.is_file() else ""),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
        "extra": extra,
    }


def _candidate_actions(selected: str) -> list[dict[str, str]]:
    rows = [
        (
            "implement_low_churn_mlp_residual_control_pilot",
            "selected" if selected == "implement_low_churn_mlp_residual_control_pilot" else "blocked",
            "Pregate sources are present and define dense24 L2, anchor/logit-MSE, flip-churn, and raw fingerprint gates.",
            "implement the local pilot and run focused artifact checks",
        ),
        (
            "run_runpod_or_colab_validation",
            "rejected",
            "This pregate is design-only and no local low-churn pilot has passed.",
            "keep GPU blocked",
        ),
        (
            "promote_raw_parameter_matched_mlp",
            "rejected",
            "Raw MLP remains high-norm/high-churn relative to dense24.",
            "do not promote raw MLP",
        ),
    ]
    if selected == "repair_low_churn_mlp_pregate_sources":
        rows.insert(
            0,
            (
                "repair_low_churn_mlp_pregate_sources",
                "selected",
                "One or more required local source artifacts or dense24 budgets is missing.",
                "repair source reports before implementation",
            ),
        )
    return [
        {
            "candidate_action": action,
            "disposition": disposition,
            "reason": reason,
            "next_step": next_step,
        }
        for action, disposition, reason, next_step in rows
    ]


def _rationale(status: str, budgets: list[dict[str, Any]]) -> str:
    if status != "pass":
        return "The low-churn MLP pregate fails closed because required source evidence or dense24 budgets are missing."
    return (
        "The raw parameter-matched MLP remains a useful high-power baseline, but it exceeds the dense24 "
        "residual-L2 and flip-churn reference. This report records the local pregate for an explicitly "
        "trained low-churn MLP control: advancement requires dense24-bounded residual L2, no worse "
        "anchor/logit drift, no worse prediction-flip churn, and raw intervention-fingerprint outputs."
    )


def _mechanism_row(summary: dict[str, Any], arm: str) -> dict[str, Any]:
    for row in _as_list(summary.get("mechanism_comparison")):
        if isinstance(row, dict) and row.get("arm") == arm:
            return row
    return {}


def _scaled_arm_row(rows: list[dict[str, str]], arm: str, lam: float) -> dict[str, str]:
    for row in rows:
        if row.get("arm") == arm and _float(row.get("lambda")) == lam:
            return dict(row)
    return {}


def _best_match(
    rows: list[dict[str, str]],
    *,
    match_type: str,
    reference_arm: str,
    arm: str,
) -> dict[str, str]:
    candidates = [
        row
        for row in rows
        if row.get("match_type") == match_type
        and row.get("reference_arm") == reference_arm
        and row.get("arm") == arm
    ]
    if not candidates:
        return {}
    return dict(min(candidates, key=lambda row: _float(row.get("distance")) or 0.0))


def _criterion(criterion: str, passed: bool, threshold: Any, actual: Any, failure_reason: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": actual,
        "failure_reason": "" if passed else failure_reason,
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
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No strategy review was present; the report fails closed through the strategy-review criterion."
    return (
        "Accepted the no-GPU/local-pregate direction. The specific pair-composer pregate recommendation "
        "is recorded as already satisfied and negative by dense_teacher_pair_composer_pregate_closeout, "
        "so this report proceeds to the next local low-churn MLP control pregate."
    )


def _header_value(text: str, key: str) -> str:
    prefix = f"{key}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _budget_value(rows: list[dict[str, Any]], metric: str) -> Any:
    value = _budget_float(rows, metric)
    return "" if value is None else value


def _budget_float(rows: list[dict[str, Any]], metric: str) -> float | None:
    for row in rows:
        if row.get("metric") == metric:
            return _float(row.get("value"))
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _float(value)
        if parsed is not None:
            return parsed
    return None


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "budget_rows.csv", summary["budget_rows"])
    _write_csv(out_dir / "pregate_arms.csv", summary["pregate_arms"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    lines = [
        "# Low-Churn MLP Residual-Control Pregate",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Promotion allowed: `{summary['promotion_allowed']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        "",
        summary["rationale"],
    ]
    if summary["failures"]:
        lines.extend(["", "## Failed Criteria"])
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


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-selector", type=Path, default=DEFAULT_BRANCH_SELECTOR)
    parser.add_argument("--mlp-followup", type=Path, default=DEFAULT_MLP_FOLLOWUP)
    parser.add_argument("--mlp-fingerprint-dir", type=Path, default=DEFAULT_MLP_FINGERPRINT)
    parser.add_argument("--mlp-decision", type=Path, default=DEFAULT_MLP_DECISION)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_low_churn_mlp_residual_control_pregate(
        branch_selector_path=args.branch_selector,
        mlp_followup_path=args.mlp_followup,
        mlp_fingerprint_dir=args.mlp_fingerprint_dir,
        mlp_decision_path=args.mlp_decision,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
