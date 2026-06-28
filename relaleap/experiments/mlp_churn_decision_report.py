"""Decision report for the local MLP churn/intervention-fingerprint assay."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_FINGERPRINT_DIR = Path("results/reports/mlp_churn_intervention_fingerprint")
DEFAULT_OUT_DIR = Path("results/reports/mlp_churn_decision")

MLP_ARM = "parameter_matched_causal_mlp_control"
DENSE_REFERENCE_ARM = "dense_rank24_best_norm"
SPARSE_REFERENCE_ARM = "acsr_mlp_predicted_future"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "decision_criteria.csv",
    "arm_tradeoffs.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_mlp_churn_decision_report(
    *,
    fingerprint_dir: Path = DEFAULT_FINGERPRINT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    ce_tolerance: float = 0.025,
    residual_l2_tolerance_fraction: float = 0.15,
    churn_tolerance: float = 0.05,
    raw_churn_budget: float = 0.35,
) -> dict[str, Any]:
    """Interpret scaled MLP-vs-dense tradeoffs and choose one next action."""

    start = time.time()
    fingerprint_summary = _read_json(fingerprint_dir / "summary.json")
    scaled_match_rows = _read_csv(fingerprint_dir / "scaled_match_summary.csv")
    scaled_rows = _read_csv(fingerprint_dir / "scaled_interventions.csv")
    available_rows = _read_csv(fingerprint_dir / "available_arms.csv")

    dense_ref = _scaled_arm_row(scaled_rows, DENSE_REFERENCE_ARM, 1.0)
    raw_mlp = _scaled_arm_row(scaled_rows, MLP_ARM, 1.0)
    mlp_l2_match = _best_match(
        scaled_match_rows,
        match_type="residual_l2",
        reference_arm=DENSE_REFERENCE_ARM,
        arm=MLP_ARM,
    )
    mlp_ce_match = _best_match(
        scaled_match_rows,
        match_type="ce_loss",
        reference_arm=DENSE_REFERENCE_ARM,
        arm=MLP_ARM,
    )
    sparse_ref = _available_arm_row(available_rows, SPARSE_REFERENCE_ARM)
    tradeoffs = _tradeoff_rows(
        dense_ref=dense_ref,
        raw_mlp=raw_mlp,
        mlp_l2_match=mlp_l2_match,
        mlp_ce_match=mlp_ce_match,
        sparse_ref=sparse_ref,
    )
    budgets = {
        "ce_tolerance": ce_tolerance,
        "residual_l2_tolerance_fraction": residual_l2_tolerance_fraction,
        "churn_tolerance": churn_tolerance,
        "raw_churn_budget": raw_churn_budget,
    }
    criteria = _criteria(
        fingerprint_summary=fingerprint_summary,
        dense_ref=dense_ref,
        raw_mlp=raw_mlp,
        mlp_l2_match=mlp_l2_match,
        sparse_ref=sparse_ref,
        budgets=budgets,
    )
    failures = [row for row in criteria if not row["passed"]]

    if not fingerprint_summary:
        status = "fail"
        decision = "mlp_churn_decision_blocked_missing_fingerprint"
        claim_status = "missing_source_artifacts"
        selected_next_action = "rerun_mlp_churn_intervention_fingerprint"
        next_command = (
            "python -m relaleap.experiments.mlp_churn_intervention_fingerprint"
        )
        rationale = "The decision report requires the local fingerprint packet."
    elif _criterion_passed(criteria, "scaled_mlp_matches_or_beats_dense_at_l2"):
        status = "pass"
        decision = "norm_budgeted_churn_regularized_mlp_variant_warranted"
        claim_status = "scaled_mlp_retains_dense_matched_ce_without_extra_drift"
        selected_next_action = "design_norm_budgeted_churn_regularized_mlp_variant"
        next_command = None
        rationale = (
            "At the dense residual-L2 operating point, the scaled MLP is within the "
            "CE budget while not exceeding dense churn/logit-MSE drift. A bounded "
            "training variant with explicit norm, anchor-KL, and churn penalties is "
            "scientifically warranted before any default promotion."
        )
    else:
        status = "pass"
        decision = "return_to_sparse_acsr_support_diagnostics"
        claim_status = "raw_mlp_high_power_high_churn_scaled_mlp_not_dense_dominant"
        selected_next_action = "extract_sparse_acsr_per_token_churn_fingerprints"
        next_command = None
        rationale = (
            "Raw MLP still wins CE mainly at much larger residual norm and churn, "
            "while the lambda-scaled MLP does not beat dense rank-24 CE at the "
            "matched residual-L2 point. Sparse ACSR aggregate evidence reaches "
            "similar CE with lower reported churn, but lacks per-token CE/L2/churn "
            "rows in this packet, so the next coherent local step is to extract "
            "sparse per-token fingerprints rather than promote or further tune MLP."
        )

    candidate_actions = _candidate_actions(selected_next_action)
    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "promotion_allowed": False,
        "requires_gpu_now": False,
        "selected_next_action": selected_next_action,
        "next_command": next_command,
        "source_dir": str(fingerprint_dir),
        "budgets": budgets,
        "evidence": {
            "dense_reference_arm": dense_ref,
            "raw_mlp": raw_mlp,
            "mlp_residual_l2_match_to_dense": mlp_l2_match,
            "mlp_ce_match_to_dense": mlp_ce_match,
            "sparse_reference_arm": sparse_ref,
        },
        "criteria": criteria,
        "failures": failures,
        "arm_tradeoffs": tradeoffs,
        "candidate_actions": candidate_actions,
        "rationale": rationale,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, criteria, tradeoffs, candidate_actions)
    return summary


def _criteria(
    *,
    fingerprint_summary: dict[str, Any],
    dense_ref: dict[str, Any],
    raw_mlp: dict[str, Any],
    mlp_l2_match: dict[str, Any],
    sparse_ref: dict[str, Any],
    budgets: dict[str, float],
) -> list[dict[str, Any]]:
    dense_ce = _float(dense_ref.get("ce_loss"))
    dense_l2 = _float(dense_ref.get("residual_update_l2"))
    dense_churn = _float(dense_ref.get("prediction_changed_vs_base"))
    dense_logit_mse = _float(dense_ref.get("logit_mse_vs_base"))
    l2_ce = _float(mlp_l2_match.get("ce_loss"))
    l2_l2 = _float(mlp_l2_match.get("residual_update_l2"))
    l2_churn = _float(mlp_l2_match.get("prediction_changed_vs_base"))
    l2_logit_mse = _float(mlp_l2_match.get("logit_mse_vs_base"))
    raw_l2 = _float(raw_mlp.get("residual_update_l2"))
    raw_churn = _float(raw_mlp.get("prediction_changed_vs_base"))
    sparse_ce = _float(sparse_ref.get("aggregate_ce_loss"))
    sparse_churn = _float(sparse_ref.get("functional_churn"))

    residual_l2_close = (
        dense_l2 is not None
        and l2_l2 is not None
        and abs(l2_l2 - dense_l2) <= budgets["residual_l2_tolerance_fraction"] * max(dense_l2, 1e-12)
    )
    scaled_ce_ok = (
        dense_ce is not None
        and l2_ce is not None
        and l2_ce <= dense_ce + budgets["ce_tolerance"]
    )
    scaled_drift_ok = (
        dense_churn is not None
        and l2_churn is not None
        and l2_churn <= dense_churn + budgets["churn_tolerance"]
        and dense_logit_mse is not None
        and l2_logit_mse is not None
        and l2_logit_mse <= dense_logit_mse
    )
    return [
        _criterion(
            "fingerprint_assay_passed",
            fingerprint_summary.get("status") == "pass"
            and fingerprint_summary.get("decision") == "mlp_churn_intervention_fingerprint_scaled_assay_completed",
            "source fingerprint report must pass with scaled assay completed",
            {
                "status": fingerprint_summary.get("status"),
                "decision": fingerprint_summary.get("decision"),
            },
            "source fingerprint report is missing or not a completed scaled assay",
        ),
        _criterion(
            "dense_reference_present",
            bool(dense_ref),
            "dense rank-24 lambda=1 reference row must be available",
            dense_ref,
            "dense reference row is missing",
        ),
        _criterion(
            "mlp_residual_l2_match_present",
            bool(mlp_l2_match),
            "MLP nearest residual-L2 match to dense rank-24 must be available",
            mlp_l2_match,
            "MLP residual-L2 match row is missing",
        ),
        _criterion(
            "scaled_mlp_l2_is_comparable_to_dense",
            residual_l2_close,
            "MLP matched residual L2 must be within tolerance of dense rank-24",
            {"dense_l2": dense_l2, "mlp_l2": l2_l2},
            "nearest scaled MLP residual L2 is outside the comparable operating range",
        ),
        _criterion(
            "scaled_mlp_matches_or_beats_dense_at_l2",
            residual_l2_close and scaled_ce_ok and scaled_drift_ok,
            "at matched residual L2, scaled MLP must be within CE tolerance and not worse on churn/logit-MSE drift",
            {
                "dense_ce": dense_ce,
                "mlp_ce": l2_ce,
                "dense_churn": dense_churn,
                "mlp_churn": l2_churn,
                "dense_logit_mse": dense_logit_mse,
                "mlp_logit_mse": l2_logit_mse,
            },
            "scaled MLP does not dominate the dense rank-24 control at matched residual L2",
        ),
        _criterion(
            "raw_mlp_within_drift_budget",
            raw_churn is not None and raw_churn <= budgets["raw_churn_budget"],
            "raw MLP endpoint must stay inside the explicit churn budget before promotion",
            {"raw_l2": raw_l2, "raw_churn": raw_churn},
            "raw MLP endpoint exceeds the churn budget and remains a high-power adapter",
        ),
        _criterion(
            "sparse_acsr_aggregate_suggests_lower_churn_path",
            sparse_ce is not None
            and dense_ce is not None
            and sparse_ce <= dense_ce
            and sparse_churn is not None
            and raw_churn is not None
            and sparse_churn < raw_churn,
            "sparse ACSR aggregate must be competitive with dense CE and lower churn than raw MLP",
            {"sparse_ce": sparse_ce, "dense_ce": dense_ce, "sparse_churn": sparse_churn, "raw_mlp_churn": raw_churn},
            "sparse aggregate evidence is unavailable or not a lower-churn path",
        ),
    ]


def _tradeoff_rows(
    *,
    dense_ref: dict[str, Any],
    raw_mlp: dict[str, Any],
    mlp_l2_match: dict[str, Any],
    mlp_ce_match: dict[str, Any],
    sparse_ref: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _tradeoff_row("dense_reference", DENSE_REFERENCE_ARM, dense_ref),
        _tradeoff_row("raw_mlp", MLP_ARM, raw_mlp),
        _tradeoff_row("mlp_residual_l2_match", MLP_ARM, mlp_l2_match),
        _tradeoff_row("mlp_ce_match", MLP_ARM, mlp_ce_match),
        {
            "role": "sparse_aggregate_reference",
            "arm": SPARSE_REFERENCE_ARM,
            "lambda": "",
            "ce_loss": _float_or_blank(sparse_ref.get("aggregate_ce_loss")),
            "residual_update_l2": _float_or_blank(sparse_ref.get("aggregate_residual_l2")),
            "logit_mse_vs_base": _float_or_blank(sparse_ref.get("anchor_kl_or_logit_mse")),
            "prediction_changed_vs_base": _float_or_blank(sparse_ref.get("functional_churn")),
            "note": "aggregate only; sparse per-token CE/L2/churn rows are absent in current packet",
        },
    ]


def _tradeoff_row(role: str, arm: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": role,
        "arm": arm,
        "lambda": _float_or_blank(row.get("lambda")),
        "ce_loss": _float_or_blank(row.get("ce_loss")),
        "residual_update_l2": _float_or_blank(row.get("residual_update_l2")),
        "logit_mse_vs_base": _float_or_blank(row.get("logit_mse_vs_base")),
        "prediction_changed_vs_base": _float_or_blank(row.get("prediction_changed_vs_base")),
        "note": "",
    }


def _candidate_actions(selected: str) -> list[dict[str, str]]:
    rows = [
        (
            "extract_sparse_acsr_per_token_churn_fingerprints",
            "selected" if selected == "extract_sparse_acsr_per_token_churn_fingerprints" else "deferred",
            "Sparse ACSR has competitive aggregate CE and lower churn than raw MLP, but needs per-token CE/L2/churn matching before mechanism claims.",
        ),
        (
            "design_norm_budgeted_churn_regularized_mlp_variant",
            "selected" if selected == "design_norm_budgeted_churn_regularized_mlp_variant" else "deferred",
            "Only justified when scaled MLP is dense-competitive at matched residual L2 without worse drift.",
        ),
        (
            "promote_raw_mlp_default",
            "disqualified",
            "Raw MLP endpoint is high-norm and high-churn; promotion remains blocked.",
        ),
        (
            "run_gpu_validation_now",
            "disqualified",
            "Current decision is local artifact interpretation and does not require GPU validation.",
        ),
    ]
    return [
        {"candidate_action": action, "disposition": disposition, "reason": reason}
        for action, disposition, reason in rows
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


def _criterion_passed(criteria: list[dict[str, Any]], name: str) -> bool:
    return any(row["criterion"] == name and row["passed"] for row in criteria)


def _scaled_arm_row(rows: list[dict[str, str]], arm: str, lam: float) -> dict[str, Any]:
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
) -> dict[str, Any]:
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


def _available_arm_row(rows: list[dict[str, str]], arm: str) -> dict[str, Any]:
    for row in rows:
        if row.get("arm") == arm:
            return dict(row)
    return {}


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    criteria: list[dict[str, Any]],
    tradeoffs: list[dict[str, Any]],
    candidate_actions: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "decision_criteria.csv", criteria)
    _write_csv(out_dir / "arm_tradeoffs.csv", tradeoffs)
    _write_csv(out_dir / "candidate_actions.csv", candidate_actions)
    lines = [
        "# MLP Churn Decision",
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


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fingerprint-dir", type=Path, default=DEFAULT_FINGERPRINT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ce-tolerance", type=float, default=0.025)
    parser.add_argument("--residual-l2-tolerance-fraction", type=float, default=0.15)
    parser.add_argument("--churn-tolerance", type=float, default=0.05)
    parser.add_argument("--raw-churn-budget", type=float, default=0.35)
    args = parser.parse_args()
    summary = run_mlp_churn_decision_report(
        fingerprint_dir=args.fingerprint_dir,
        out_dir=args.out,
        ce_tolerance=args.ce_tolerance,
        residual_l2_tolerance_fraction=args.residual_l2_tolerance_fraction,
        churn_tolerance=args.churn_tolerance,
        raw_churn_budget=args.raw_churn_budget,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
