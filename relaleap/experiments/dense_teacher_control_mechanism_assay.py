"""Dense-teacher/control mechanism assay.

This report consumes existing command-generated artifacts and fails closed
unless the dense-teacher ACSR pilot clears dense24 and parameter-matched MLP
controls on norm, KL/MSE, churn, and intervention-fingerprint gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_DENSE_TEACHER = Path("results/audits/token_larger_dense_teacher_residual_distillation_comparison")
DEFAULT_DENSE_PRIMARY = Path("results/reports/dense_primary_mechanism_assay")
DEFAULT_MLP_FOLLOWUP = Path("results/reports/mlp_dense_heldout_mechanism_followup")
DEFAULT_MLP_FINGERPRINT = Path("results/reports/mlp_churn_intervention_fingerprint")
DEFAULT_MATCHED_DECISION = Path("results/reports/sparse_dense_mlp_matched_intervention_decision")
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_control_mechanism_assay")

PRIMARY_VARIANT = "acsr_predicted_future_support"
REQUIRED_CONTROLS = ("dense_rank24_best_norm", "parameter_matched_causal_mlp_control")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "control_gate.csv",
    "decision_criteria.csv",
    "notes.md",
)


def run_dense_teacher_control_mechanism_assay(
    *,
    dense_teacher_dir: Path = DEFAULT_DENSE_TEACHER,
    dense_primary_dir: Path = DEFAULT_DENSE_PRIMARY,
    mlp_followup_dir: Path = DEFAULT_MLP_FOLLOWUP,
    mlp_fingerprint_dir: Path = DEFAULT_MLP_FINGERPRINT,
    matched_decision_dir: Path = DEFAULT_MATCHED_DECISION,
    out_dir: Path = DEFAULT_OUT_DIR,
    max_residual_l2_ratio: float = 1.10,
    max_anchor_mse_ratio: float = 1.10,
    max_flip_churn_ratio: float = 1.10,
    min_fingerprint_purity_delta: float = 0.0,
) -> dict[str, Any]:
    """Write a fail-closed dense-teacher/control mechanism gate."""

    start = time.time()
    dense_teacher = _read_json(dense_teacher_dir / "summary.json")
    dense_primary = _read_json(dense_primary_dir / "summary.json")
    mlp_followup = _read_json(mlp_followup_dir / "summary.json")
    mlp_fingerprint = _read_json(mlp_fingerprint_dir / "summary.json")
    matched_decision = _read_json(matched_decision_dir / "summary.json")

    sources = [
        _source_row("dense_teacher_distillation", dense_teacher_dir / "summary.json", dense_teacher),
        _source_row("dense_primary_mechanism_assay", dense_primary_dir / "summary.json", dense_primary),
        _source_row("mlp_dense_heldout_followup", mlp_followup_dir / "summary.json", mlp_followup),
        _source_row("mlp_churn_intervention_fingerprint", mlp_fingerprint_dir / "summary.json", mlp_fingerprint),
        _source_row("sparse_dense_mlp_matched_intervention_decision", matched_decision_dir / "summary.json", matched_decision),
    ]
    control_gate = _control_gate(
        dense_teacher=dense_teacher,
        dense_primary=dense_primary,
        mlp_followup=mlp_followup,
        matched_decision=matched_decision,
        max_residual_l2_ratio=max_residual_l2_ratio,
        max_anchor_mse_ratio=max_anchor_mse_ratio,
        max_flip_churn_ratio=max_flip_churn_ratio,
        min_fingerprint_purity_delta=min_fingerprint_purity_delta,
    )
    criteria = _criteria(
        sources=sources,
        dense_teacher=dense_teacher,
        mlp_fingerprint=mlp_fingerprint,
        matched_decision=matched_decision,
        control_gate=control_gate,
    )
    failures = [row for row in criteria if not row["passed"]]

    if failures:
        status = "pass"
        decision = "dense_teacher_control_mechanism_assay_blocked"
        scientific_gate = "blocked"
        claim_status = "dense_teacher_acsr_not_supported_against_dense24_mlp_controls"
        selected_next_step = (
            "do not run GPU validation; either extend the dense-teacher pilot with matched "
            "residual-L2/churn/fingerprint observables or pivot to a dense/MLP candidate as the mechanism baseline"
        )
    else:
        status = "pass"
        decision = "dense_teacher_control_mechanism_assay_cleared_for_gpu_repeat"
        scientific_gate = "pass"
        claim_status = "dense_teacher_acsr_survives_dense24_mlp_control_gate_local_only"
        selected_next_step = "run one RunPod repeat and local artifact check before any default or mechanism claim"

    summary = {
        "status": status,
        "decision": decision,
        "scientific_gate": scientific_gate,
        "claim_status": claim_status,
        "promotion_allowed": False,
        "requires_gpu_now": scientific_gate == "pass",
        "backend_policy": "RunPod only if this local scientific gate passes; Colab not used for this step",
        "source_dirs": {
            "dense_teacher": str(dense_teacher_dir),
            "dense_primary": str(dense_primary_dir),
            "mlp_followup": str(mlp_followup_dir),
            "mlp_fingerprint": str(mlp_fingerprint_dir),
            "matched_decision": str(matched_decision_dir),
        },
        "source_rows": sources,
        "control_gate": control_gate,
        "criteria": criteria,
        "failures": failures,
        "selected_next_step": selected_next_step,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _control_gate(
    *,
    dense_teacher: dict[str, Any],
    dense_primary: dict[str, Any],
    mlp_followup: dict[str, Any],
    matched_decision: dict[str, Any],
    max_residual_l2_ratio: float,
    max_anchor_mse_ratio: float,
    max_flip_churn_ratio: float,
    min_fingerprint_purity_delta: float,
) -> list[dict[str, Any]]:
    primary_variant = next(
        (row for row in _as_list(dense_teacher.get("variant_rows")) if row.get("variant") == PRIMARY_VARIANT),
        {},
    )
    scorecard = {row.get("arm"): row for row in _as_list(dense_primary.get("candidate_scorecard"))}
    comparisons = {row.get("arm"): row for row in _as_list(mlp_followup.get("mechanism_comparison"))}
    domination_cases = {
        row.get("challenger_arm"): row
        for row in _read_csv(Path(matched_decision.get("artifacts", {}).get("domination_cases_csv", "")))
        if row.get("challenger_arm") in REQUIRED_CONTROLS
    }

    rows: list[dict[str, Any]] = []
    for control in REQUIRED_CONTROLS:
        aggregate = scorecard.get(control, {})
        heldout = comparisons.get(control, {})
        rows.append(
            {
                "primary_variant": PRIMARY_VARIANT,
                "control_arm": control,
                "primary_ce_loss": _float_or_blank(primary_variant.get("ce_loss")),
                "primary_teacher_logit_mse": _float_or_blank(primary_variant.get("teacher_logit_mse")),
                "control_ce_loss": _float_or_blank(aggregate.get("ce_loss") or heldout.get("heldout_ce_loss")),
                "control_residual_l2": _float_or_blank(aggregate.get("residual_l2") or heldout.get("heldout_residual_update_l2")),
                "control_anchor_kl_or_logit_mse": _float_or_blank(aggregate.get("anchor_kl_or_logit_mse") or heldout.get("heldout_logit_mse_vs_base")),
                "control_flip_churn": _float_or_blank(heldout.get("heldout_prediction_changed_vs_base") or aggregate.get("functional_churn")),
                "control_functional_churn": _float_or_blank(aggregate.get("functional_churn")),
                "control_intervention_fingerprint_purity": _float_or_blank(aggregate.get("intervention_fingerprint_purity")),
                "matched_intervention_advances": _bool(domination_cases.get(control, {}).get("challenger_advances")),
                "gate_max_residual_l2_ratio": max_residual_l2_ratio,
                "gate_max_anchor_mse_ratio": max_anchor_mse_ratio,
                "gate_max_flip_churn_ratio": max_flip_churn_ratio,
                "gate_min_fingerprint_purity_delta": min_fingerprint_purity_delta,
                "control_fields_present": _control_fields_present(aggregate, heldout),
            }
        )
    return rows


def _criteria(
    *,
    sources: list[dict[str, Any]],
    dense_teacher: dict[str, Any],
    mlp_fingerprint: dict[str, Any],
    matched_decision: dict[str, Any],
    control_gate: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sources_ok = all(row["present"] and row["status"] in {"pass", "ok"} for row in sources)
    dense_teacher_supported = (
        dense_teacher.get("status") == "pass"
        and dense_teacher.get("decision") == "dense_teacher_residual_distillation_acsr_pilot_supported_not_promoted"
    )
    controls_present = all(row["control_fields_present"] for row in control_gate)
    raw_fingerprints_ok = (
        mlp_fingerprint.get("status") == "pass"
        and mlp_fingerprint.get("decision") == "mlp_churn_intervention_fingerprint_scaled_assay_completed"
    )
    matched_gate_ok = matched_decision.get("scientific_gate") == "pass"
    return [
        _criterion(
            "required_sources_present_and_passing",
            sources_ok,
            "all source summaries are present and status pass/ok",
            {row["source"]: {"present": row["present"], "status": row["status"]} for row in sources},
            "one or more source summaries is missing or failing",
        ),
        _criterion(
            "dense_teacher_acsr_pilot_supported",
            dense_teacher_supported,
            "dense-teacher ACSR pilot must pass its own CE/distillation/null gate",
            {"status": dense_teacher.get("status"), "decision": dense_teacher.get("decision"), "failures": dense_teacher.get("failures", [])},
            "dense-teacher ACSR pilot did not pass its own guardrails",
        ),
        _criterion(
            "dense24_and_mlp_control_fields_present",
            controls_present,
            "dense24 and parameter-matched MLP expose residual-L2, KL/MSE, churn/flip, and fingerprint fields",
            control_gate,
            "dense24 or MLP control fields are incomplete",
        ),
        _criterion(
            "raw_scaled_intervention_fingerprints_available",
            raw_fingerprints_ok,
            "MLP churn/intervention fingerprint report passed with raw scaled rows",
            {"status": mlp_fingerprint.get("status"), "decision": mlp_fingerprint.get("decision")},
            "raw scaled intervention fingerprints are unavailable",
        ),
        _criterion(
            "matched_dense_control_gate_passed",
            matched_gate_ok,
            "matched residual-L2 CE/MSE/flip gate must pass before GPU validation",
            {"scientific_gate": matched_decision.get("scientific_gate"), "decision": matched_decision.get("decision"), "advancement_row_count": matched_decision.get("advancement_row_count")},
            "no challenger cleared the dense Pareto guardrail at matched residual-L2/churn",
        ),
    ]


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    status = payload.get("status")
    if status == "ok":
        status = "pass"
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": status if path.is_file() else "missing",
        "decision": payload.get("decision"),
        "claim_status": payload.get("claim_status"),
    }


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


def _control_fields_present(aggregate: dict[str, Any], heldout: dict[str, Any]) -> bool:
    return all(
        _float(value) is not None
        for value in (
            aggregate.get("residual_l2") or heldout.get("heldout_residual_update_l2"),
            aggregate.get("anchor_kl_or_logit_mse") or heldout.get("heldout_logit_mse_vs_base"),
            aggregate.get("functional_churn"),
            heldout.get("heldout_prediction_changed_vs_base"),
            aggregate.get("intervention_fingerprint_purity"),
        )
    )


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "control_gate.csv", summary["control_gate"])
    _write_csv(out_dir / "decision_criteria.csv", summary["criteria"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Dense-Teacher Control Mechanism Assay",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Scientific gate: `{summary['scientific_gate']}`",
            f"- Claim status: `{summary['claim_status']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            f"- Promotion allowed: `{summary['promotion_allowed']}`",
            f"- Next step: {summary['selected_next_step']}",
            "",
            "This report treats dense24 and the parameter-matched causal MLP as controls. It blocks GPU validation unless dense-teacher ACSR first passes its own pilot and the matched residual-L2, KL/MSE, flip/churn, and intervention-fingerprint control gates.",
            "",
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_list(value: Any) -> list[dict[str, Any]]:
    return value if isinstance(value, list) else []


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_blank(value: Any) -> float | str:
    parsed = _float(value)
    return "" if parsed is None else parsed


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-teacher-dir", type=Path, default=DEFAULT_DENSE_TEACHER)
    parser.add_argument("--dense-primary-dir", type=Path, default=DEFAULT_DENSE_PRIMARY)
    parser.add_argument("--mlp-followup-dir", type=Path, default=DEFAULT_MLP_FOLLOWUP)
    parser.add_argument("--mlp-fingerprint-dir", type=Path, default=DEFAULT_MLP_FINGERPRINT)
    parser.add_argument("--matched-decision-dir", type=Path, default=DEFAULT_MATCHED_DECISION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_dense_teacher_control_mechanism_assay(
        dense_teacher_dir=args.dense_teacher_dir,
        dense_primary_dir=args.dense_primary_dir,
        mlp_followup_dir=args.mlp_followup_dir,
        mlp_fingerprint_dir=args.mlp_fingerprint_dir,
        matched_decision_dir=args.matched_decision_dir,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "scientific_gate": summary["scientific_gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
