"""Design a local value-dictionary rescue for the sparse-factorization ceiling."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_CLOSEOUT = Path("results/reports/low_churn_mlp_sparse_factorization_ceiling_closeout/summary.json")
DEFAULT_DECISION_AUDIT = Path("results/reports/low_churn_mlp_sparse_factorization_decision_audit/summary.json")
DEFAULT_OUT_DIR = Path("results/reports/low_churn_mlp_value_dictionary_capacity_rescue_design")

IMPLEMENT_ACTION = "implement_value_dictionary_capacity_rescue_local_pregate"
REPAIR_ACTION = "repair_value_dictionary_capacity_rescue_design_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "dictionary_designs.csv",
    "control_rows.csv",
    "gate_criteria.csv",
    "target_noncolumnability_gates.csv",
    "notes.md",
)


def run_low_churn_mlp_value_dictionary_capacity_rescue_design(
    *,
    closeout_path: Path = DEFAULT_CLOSEOUT,
    decision_audit_path: Path = DEFAULT_DECISION_AUDIT,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed design contract for richer reusable value dictionaries."""

    start = time.time()
    closeout = _read_json(closeout_path)
    audit = _read_json(decision_audit_path)
    source_rows = [
        _source_row("sparse_factorization_ceiling_closeout", closeout_path, closeout),
        _source_row("sparse_factorization_decision_audit", decision_audit_path, audit),
    ]
    dictionary_designs = _dictionary_designs()
    control_rows = _control_rows()
    target_gates = _target_noncolumnability_gates(audit)
    gate_criteria = _gate_criteria(closeout, audit, dictionary_designs, control_rows, target_gates)
    source_failures = _source_failures(source_rows)
    failures = source_failures + [row for row in gate_criteria if not row["passed"]]
    status = "pass" if not failures else "fail"
    selected_next_action = IMPLEMENT_ACTION if status == "pass" else REPAIR_ACTION
    summary = {
        "status": status,
        "decision": (
            "low_churn_mlp_value_dictionary_capacity_rescue_design_recorded"
            if status == "pass"
            else "low_churn_mlp_value_dictionary_capacity_rescue_design_failed_closed"
        ),
        "claim_status": "design_only_no_value_dictionary_rescue_claim",
        "selected_next_action": selected_next_action,
        "selected_next_step": (
            "implement a local value-dictionary capacity rescue pregate over captured low-churn teacher vectors"
            if status == "pass"
            else "repair missing or inconsistent sparse-factorization closeout/audit sources before implementation"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local design/pregate only; RunPod and Colab remain blocked until reusable value-dictionary and non-columnability gates pass",
        "source_rows": source_rows,
        "dictionary_designs": dictionary_designs,
        "control_rows": control_rows,
        "target_noncolumnability_gates": target_gates,
        "gate_criteria": gate_criteria,
        "failures": failures,
        "rationale": _rationale(audit),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _dictionary_designs() -> list[dict[str, Any]]:
    return [
        {
            "design": "multi_codebook_residual_dictionary",
            "value_family": "stagewise residual vector quantization",
            "support_budget": "2 to 4 dictionary codes per token with fixed reusable codebooks",
            "deployable_router": "prefix-safe router predicts stagewise codes after oracle-support pregate passes",
            "reason": "Tests whether the weak single-code vector-centroid ceiling was under-capacity rather than non-columnable.",
        },
        {
            "design": "low_rank_codebook_dictionary",
            "value_family": "shared low-rank basis plus sparse learned coefficients",
            "support_budget": "rank sweep matched to 7/9 covariance effective dimensions and dense24 budget",
            "deployable_router": "prefix-safe coefficient head only after oracle coefficient ceiling clears",
            "reason": "Separates low intrinsic dimension from reusable column value failure.",
        },
        {
            "design": "rule_conditioned_global_dictionary",
            "value_family": "small global dictionary with train-only latent-rule or token-stratum conditioning",
            "support_budget": "same total stored vector budget as multi-codebook arm",
            "deployable_router": "condition only on observed prefix-safe strata; no heldout target leakage",
            "reason": "Checks whether one global dictionary hides heterogeneity across latent rules or token contexts.",
        },
        {
            "design": "norm_budgeted_dictionary_with_residual_tail",
            "value_family": "clamped reusable code plus explicit small dense tail control",
            "support_budget": "same residual norm budget as low-churn MLP teacher/control",
            "deployable_router": "sparse dictionary must explain most teacher energy before dense tail is allowed",
            "reason": "Prevents a dense tail from laundering a sparse-column failure while measuring irreducible target tail.",
        },
    ]


def _control_rows() -> list[dict[str, str]]:
    return [
        _control("dense_ridge_same_rows", "dense", "same train/heldout teacher-vector rows and norm budget"),
        _control("low_rank_svd_same_rank_sweep", "low_rank", "rank sweep around covariance 90/95 percent dimensions"),
        _control("flat_value_mlp_same_router_budget", "flat_mlp", "generic nonlinear value capacity with matched stored/active params"),
        _control("single_codebook_vector_centroid_baseline", "sparse_baseline", "current weak reusable dictionary baseline"),
        _control("shuffled_teacher_dictionary", "target_null", "same dictionary fit to misaligned teacher residuals"),
        _control("route_scrambled_dictionary", "support_null", "same values with incoherent support assignment"),
    ]


def _control(name: str, family: str, matching_rule: str) -> dict[str, str]:
    return {
        "control": name,
        "family": family,
        "matching_rule": matching_rule,
        "required_before_gpu": "true",
    }


def _target_noncolumnability_gates(audit: dict[str, Any]) -> list[dict[str, Any]]:
    covariance = audit.get("covariance_summary", {}) if isinstance(audit.get("covariance_summary"), dict) else {}
    return [
        _target_gate(
            "richer_oracle_dictionary_min_r2",
            "oracle reusable richer dictionary heldout R2 must reach >= 0.65",
            "blocks_sparse_rescue_if_failed",
            audit.get("global_dictionary_oracle_r2", ""),
        ),
        _target_gate(
            "dense_low_rank_advantage_margin",
            "dense/ridge or low-rank controls must not beat best sparse oracle by > 0.10 heldout R2 at matched budget",
            "labels_target_noncolumnable_if_failed",
            "not_measured_yet",
        ),
        _target_gate(
            "shuffled_teacher_rejection",
            "best real teacher dictionary must beat shuffled/misaligned teacher null by >= 0.20 heldout R2",
            "labels_proxy_or_scale_artifact_if_failed",
            "not_measured_yet",
        ),
        _target_gate(
            "support_load_noncollapse",
            "heldout oracle support max load fraction must stay <= 0.50 or show stratum-specific explanation",
            "blocks_global_dictionary_claim_if_failed",
            _first_global_metric(audit, "support_load_max_fraction"),
        ),
        _target_gate(
            "intrinsic_dimension_context",
            "report covariance effective dimensions and require dictionary capacity sweep around dim90/dim95",
            "required_context",
            {
                "effective_dim_90pct": covariance.get("effective_dim_90pct", ""),
                "effective_dim_95pct": covariance.get("effective_dim_95pct", ""),
            },
        ),
    ]


def _target_gate(gate: str, threshold: str, failure_interpretation: str, current_value: Any) -> dict[str, Any]:
    return {
        "gate": gate,
        "threshold": threshold,
        "failure_interpretation": failure_interpretation,
        "current_value": current_value,
        "measured_in_design": False,
    }


def _gate_criteria(
    closeout: dict[str, Any],
    audit: dict[str, Any],
    dictionary_designs: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    target_gates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    controls = {row["control"] for row in control_rows}
    designs = {row["design"] for row in dictionary_designs}
    gates = {row["gate"] for row in target_gates}
    return [
        _criterion(
            "closeout_selects_value_dictionary_rescue",
            closeout.get("status") == "pass"
            and closeout.get("selected_next_action") == "design_value_dictionary_capacity_rescue_before_gpu",
            closeout.get("selected_next_action"),
            "latest sparse-factorization closeout must select value-dictionary rescue design",
        ),
        _criterion(
            "decision_audit_blocks_gpu_with_weak_global_dictionary",
            audit.get("status") == "pass"
            and audit.get("advance_to_gpu_validation") is False
            and audit.get("global_dictionary_oracle_r2", 1.0) < 0.5,
            {
                "advance_to_gpu_validation": audit.get("advance_to_gpu_validation"),
                "global_dictionary_oracle_r2": audit.get("global_dictionary_oracle_r2"),
            },
            "decision audit must show weak reusable-dictionary ceiling and no GPU advancement",
        ),
        _criterion(
            "richer_value_dictionary_design_coverage",
            {
                "multi_codebook_residual_dictionary",
                "low_rank_codebook_dictionary",
                "rule_conditioned_global_dictionary",
                "norm_budgeted_dictionary_with_residual_tail",
            }.issubset(designs),
            sorted(designs),
            "design must include richer reusable value families, not router-only tuning",
        ),
        _criterion(
            "dense_low_rank_and_null_controls_present",
            {
                "dense_ridge_same_rows",
                "low_rank_svd_same_rank_sweep",
                "flat_value_mlp_same_router_budget",
                "shuffled_teacher_dictionary",
                "route_scrambled_dictionary",
            }.issubset(controls),
            sorted(controls),
            "dense, low-rank, flat-MLP, shuffled, and route-scrambled controls are required",
        ),
        _criterion(
            "target_noncolumnability_gates_present",
            {
                "richer_oracle_dictionary_min_r2",
                "dense_low_rank_advantage_margin",
                "shuffled_teacher_rejection",
                "support_load_noncollapse",
                "intrinsic_dimension_context",
            }.issubset(gates),
            sorted(gates),
            "target non-columnability gates must be explicit before implementation",
        ),
    ]


def _criterion(criterion: str, passed: bool, actual: Any, threshold: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "actual": actual,
        "threshold": threshold,
        "failure_reason": "" if passed else threshold,
    }


def _rationale(audit: dict[str, Any]) -> str:
    return (
        "The vector-centroid reusable dictionary is too weak for GPU validation "
        f"(heldout R2={audit.get('global_dictionary_oracle_r2', 'unknown')}). "
        "This design tests whether richer reusable values can rescue the sparse-factorization ceiling while dense, "
        "low-rank, flat-value, shuffled-teacher, and route-scrambled controls can still falsify the columnability claim."
    )


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status", "missing" if not path.is_file() else ""),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
        "selected_next_action": payload.get("selected_next_action", ""),
    }


def _source_failures(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"source": row["source"], "reason": f"{row['path']} is missing"}
        for row in rows
        if not row["present"]
    ]


def _first_global_metric(audit: dict[str, Any], key: str) -> Any:
    rows = audit.get("global_dictionary_metrics", [])
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return rows[0].get(key, "")
    return ""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "dictionary_designs.csv", summary["dictionary_designs"])
    _write_csv(out_dir / "control_rows.csv", summary["control_rows"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_csv(out_dir / "target_noncolumnability_gates.csv", summary["target_noncolumnability_gates"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
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
            "# Low-Churn MLP Value-Dictionary Capacity Rescue Design",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Claim status: `{summary['claim_status']}`",
            f"- Selected action: `{summary['selected_next_action']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            f"- Advance to GPU validation: `{summary['advance_to_gpu_validation']}`",
            "",
            summary["rationale"],
            "",
            "This is a local design contract only. It does not promote sparse columns or reopen RunPod validation.",
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
    parser.add_argument("--closeout", type=Path, default=DEFAULT_CLOSEOUT)
    parser.add_argument("--decision-audit", type=Path, default=DEFAULT_DECISION_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_low_churn_mlp_value_dictionary_capacity_rescue_design(
        closeout_path=args.closeout,
        decision_audit_path=args.decision_audit,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "selected_next_action": summary["selected_next_action"],
                "advance_to_gpu_validation": summary["advance_to_gpu_validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
