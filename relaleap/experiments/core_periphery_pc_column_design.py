"""Record a fail-closed design contract for core/periphery PC columns."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path("results/reports/core_periphery_pc_column_design")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "mechanism_fields.csv",
    "controls.csv",
    "observables.csv",
    "gate_criteria.csv",
    "notes.md",
)

MANDATORY_MECHANISM_FIELDS = (
    "column_residual_parameterization",
    "core_parameterization",
    "periphery_parameterization",
    "local_pc_target",
    "core_plasticity_rule",
    "periphery_plasticity_rule",
    "consolidation_genericity_rule",
    "periphery_first_pruning_rule",
    "acsr_column_inputs",
    "leakage_constraints",
)

MANDATORY_CONTROLS = (
    "current_sparse_acsr_contextual_router",
    "dense_rank_norm_residual",
    "parameter_matched_causal_mlp",
    "random_support_router",
    "frequency_support_router",
    "no_core_ablation",
    "no_periphery_ablation",
    "equal_plasticity_core_periphery",
    "shuffled_core_periphery_assignment",
    "lambda_zero_residual",
    "token_position_only_router",
)

MANDATORY_OBSERVABLES = (
    "ce_guardrail",
    "retention_forgetting",
    "anchor_kl_drift",
    "flip_churn",
    "functional_churn",
    "residual_stream_churn",
    "finite_update_commutator",
    "residual_l2",
    "core_residual_norm",
    "periphery_residual_norm",
    "core_update_norm",
    "periphery_update_norm",
    "core_gradient_norm",
    "periphery_gradient_norm",
    "plasticity_ratio",
    "consolidation_score",
    "genericity_score",
    "pruning_sensitivity",
    "necessity_fingerprint",
    "sufficiency_fingerprint",
    "selectivity_fingerprint",
    "off_target_leakage",
    "periphery_first_prune_delta",
    "core_only_ablation",
    "periphery_only_ablation",
)


def run_core_periphery_pc_column_design(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
) -> dict[str, Any]:
    """Write the local design gate and return its machine-readable summary."""

    start = time.time()
    strategy_review = _strategy_review(strategy_review_path)
    mechanism_fields = _mechanism_fields()
    controls = _controls()
    observables = _observables()
    gate_criteria = _gate_criteria()
    failures = _contract_failures(mechanism_fields, controls, observables, gate_criteria)
    scientific_gate = "blocked" if failures else "ready_for_tiny_pilot"
    selected_next_action = (
        "repair_core_periphery_design_contract"
        if failures
        else "implement_tiny_local_core_periphery_pc_column_pilot"
    )
    next_step = (
        "repair missing mechanism, control, observable, or gate fields"
        if failures
        else (
            "implement a tiny local CPU pilot for the split core/periphery PC "
            "column, using this contract exactly before any RunPod or Colab validation"
        )
    )
    direction_shift = _direction_shift_record(strategy_review)
    summary = {
        "status": "pass" if not failures else "fail",
        "decision": "core_periphery_pc_column_design_recorded",
        "scientific_gate": scientific_gate,
        "claim_status": "design_contract_only_not_training_evidence",
        "selected_next_action": selected_next_action,
        "next_step": next_step,
        "requires_gpu_now": False,
        "backend_policy": (
            "local contract gate only; GPU validation is blocked until a tiny local "
            "pilot clears the preregistered mechanism and control gates"
        ),
        "mechanism_fields": mechanism_fields,
        "controls": controls,
        "observables": observables,
        "gate_criteria": gate_criteria,
        "failures": failures,
        "strategy_review": strategy_review,
        "direction_shift": direction_shift,
        "rationale": _rationale(scientific_gate),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "generated_from_head": _git_commit(),
        "dirty_diff_hash": _dirty_diff_hash(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _mechanism_fields() -> list[dict[str, str]]:
    return [
        {
            "field": "column_residual_parameterization",
            "specification": (
                "selected column emits core_gate * core_pc_correction + "
                "periphery_gate * periphery_pc_correction"
            ),
            "contract_role": "mechanism",
            "required": "true",
        },
        {
            "field": "core_parameterization",
            "specification": (
                "lower-plasticity residual submodule with protected atoms/units, "
                "separate residual/update/gradient norm logging, and pruning immunity "
                "unless causal utility is low"
            ),
            "contract_role": "mechanism",
            "required": "true",
        },
        {
            "field": "periphery_parameterization",
            "specification": (
                "higher-plasticity residual submodule with context-specific atoms/units, "
                "separate norm logging, sparsity pressure, and first-prune eligibility"
            ),
            "contract_role": "mechanism",
            "required": "true",
        },
        {
            "field": "local_pc_target",
            "specification": (
                "hybrid local hidden-delta plus optional dense-teacher residual target; "
                "teacher residual is marked training-only and unavailable to deployable routing"
            ),
            "contract_role": "objective",
            "required": "true",
        },
        {
            "field": "core_plasticity_rule",
            "specification": (
                "core learning rate lower than periphery, with EMA/EWC-style consolidation "
                "when errors recur across contexts"
            ),
            "contract_role": "learning_rule",
            "required": "true",
        },
        {
            "field": "periphery_plasticity_rule",
            "specification": (
                "periphery learning rate higher than core, context-specific update allowance, "
                "and penalty when periphery explains corrections already handled by core"
            ),
            "contract_role": "learning_rule",
            "required": "true",
        },
        {
            "field": "consolidation_genericity_rule",
            "specification": (
                "genericity score increases with cross-context causal utility, low churn, "
                "low off-target leakage, and stable anchor retention"
            ),
            "contract_role": "learning_rule",
            "required": "true",
        },
        {
            "field": "periphery_first_pruning_rule",
            "specification": (
                "prune peripheral units before core units when necessity/sufficiency is weak, "
                "off-target leakage or churn is high, or anchor retention benefit is low"
            ),
            "contract_role": "causal_pruning",
            "required": "true",
        },
        {
            "field": "acsr_column_inputs",
            "specification": (
                "causal ACSR predicted future-feature vector, support margin, prediction "
                "uncertainty, recent support stability, token/position features, and current hidden context"
            ),
            "contract_role": "router_interface",
            "required": "true",
        },
        {
            "field": "leakage_constraints",
            "specification": (
                "no labels, future hidden states, future tokens, or nondeployable teacher targets "
                "may enter evaluation-time routing, gates, or settling"
            ),
            "contract_role": "validity",
            "required": "true",
        },
    ]


def _controls() -> list[dict[str, str]]:
    rows = {
        "current_sparse_acsr_contextual_router": "current sparse ACSR/contextual-router column baseline",
        "dense_rank_norm_residual": "dense residual matched on active rank, residual norm, and compute budget",
        "parameter_matched_causal_mlp": "causal-input MLP residual with matched parameters and active compute",
        "random_support_router": "random support null under same top-k and support-width budget",
        "frequency_support_router": "frequency support null using train-context support frequency only",
        "no_core_ablation": "remove protected core and keep periphery path budget matched",
        "no_periphery_ablation": "remove periphery and keep core path budget matched",
        "equal_plasticity_core_periphery": "same learning rate/consolidation for core and periphery",
        "shuffled_core_periphery_assignment": "shuffle core/periphery assignment after training for accounting-fiction check",
        "lambda_zero_residual": "zero residual adapter to check base-model and artifact baselines",
        "token_position_only_router": "router restricted to token and position features",
    }
    return [
        {
            "control": name,
            "purpose": purpose,
            "matched_dimensions": "params, active_compute, residual_l2, data_split, seed",
            "required": "true",
        }
        for name, purpose in rows.items()
    ]


def _observables() -> list[dict[str, str]]:
    families = {
        "ce_guardrail": ("ce", "best/alpha0 CE must not degrade beyond preregistered tolerance"),
        "retention_forgetting": ("retention", "task-free continual-learning forgetting on anchor contexts"),
        "anchor_kl_drift": ("retention", "KL drift from anchor logits after adaptation"),
        "flip_churn": ("churn", "prediction flip fraction under adaptation and intervention"),
        "functional_churn": ("churn", "functional output churn beyond CE delta"),
        "residual_stream_churn": ("churn", "residual-stream delta/churn across anchor contexts"),
        "finite_update_commutator": ("interference", "finite-update order sensitivity"),
        "residual_l2": ("norm", "residual norm budget and matched-control guardrail"),
        "core_residual_norm": ("core", "core contribution norm"),
        "periphery_residual_norm": ("periphery", "periphery contribution norm"),
        "core_update_norm": ("core", "core update norm per step/context"),
        "periphery_update_norm": ("periphery", "periphery update norm per step/context"),
        "core_gradient_norm": ("core", "core gradient norm"),
        "periphery_gradient_norm": ("periphery", "periphery gradient norm"),
        "plasticity_ratio": ("mechanism", "periphery update/plasticity divided by core update/plasticity"),
        "consolidation_score": ("mechanism", "core consolidation/protection score"),
        "genericity_score": ("mechanism", "cross-context utility score"),
        "pruning_sensitivity": ("pruning", "loss/retention delta after candidate pruning"),
        "necessity_fingerprint": ("intervention", "necessity of core/periphery units by context"),
        "sufficiency_fingerprint": ("intervention", "sufficiency when forced alone"),
        "selectivity_fingerprint": ("intervention", "target-context benefit versus off-target effect"),
        "off_target_leakage": ("intervention", "off-target damage from support/unit intervention"),
        "periphery_first_prune_delta": ("pruning", "anchor and target deltas after periphery-first prune"),
        "core_only_ablation": ("ablation", "core-only path behavior"),
        "periphery_only_ablation": ("ablation", "periphery-only path behavior"),
    }
    return [
        {
            "observable": name,
            "family": family,
            "definition": definition,
            "required": "true",
        }
        for name, (family, definition) in families.items()
    ]


def _gate_criteria() -> list[dict[str, str]]:
    return [
        {
            "criterion": "contract_complete",
            "threshold": "all mandatory mechanism, control, observable, and gate fields present",
            "failure_action": "blocked",
            "required": "true",
        },
        {
            "criterion": "tiny_pilot_scope",
            "threshold": "CPU/local toy pilot only; no RunPod or Colab before local artifacts exist",
            "failure_action": "blocked",
            "required": "true",
        },
        {
            "criterion": "matched_control_pareto",
            "threshold": "must beat or match dense/rank/norm, MLP, current sparse ACSR, and nulls on non-CE metrics",
            "failure_action": "demote_to_operational_adapter_or_diagnostic",
            "required": "true",
        },
        {
            "criterion": "retention_churn_guardrail",
            "threshold": "forgetting and functional churn no worse than controls at matched residual norm",
            "failure_action": "blocked",
            "required": "true",
        },
        {
            "criterion": "commutator_guardrail",
            "threshold": "finite-update commutator/order sensitivity lower than matched controls or mechanistically explained",
            "failure_action": "blocked",
            "required": "true",
        },
        {
            "criterion": "intervention_fingerprint_guardrail",
            "threshold": "necessity, sufficiency, selectivity, and off-target leakage cleaner than controls",
            "failure_action": "blocked",
            "required": "true",
        },
        {
            "criterion": "periphery_first_pruning_guardrail",
            "threshold": "periphery-first pruning preserves anchors better than core pruning and removes harmful units first",
            "failure_action": "blocked",
            "required": "true",
        },
        {
            "criterion": "ce_guardrail",
            "threshold": "CE/perplexity not worse beyond preregistered tolerance",
            "failure_action": "blocked_or_diagnostic_only",
            "required": "true",
        },
        {
            "criterion": "leakage_guardrail",
            "threshold": "evaluation-time router/gates use causal inputs only",
            "failure_action": "invalid_artifact",
            "required": "true",
        },
    ]


def _contract_failures(
    mechanism_fields: list[dict[str, str]],
    controls: list[dict[str, str]],
    observables: list[dict[str, str]],
    gate_criteria: list[dict[str, str]],
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    _missing_rows(failures, "mechanism_fields", "field", MANDATORY_MECHANISM_FIELDS, mechanism_fields)
    _missing_rows(failures, "controls", "control", MANDATORY_CONTROLS, controls)
    _missing_rows(failures, "observables", "observable", MANDATORY_OBSERVABLES, observables)
    _missing_rows(
        failures,
        "gate_criteria",
        "criterion",
        (
            "contract_complete",
            "matched_control_pareto",
            "retention_churn_guardrail",
            "commutator_guardrail",
            "intervention_fingerprint_guardrail",
            "periphery_first_pruning_guardrail",
            "ce_guardrail",
            "leakage_guardrail",
        ),
        gate_criteria,
    )
    return failures


def _missing_rows(
    failures: list[dict[str, str]],
    source: str,
    key: str,
    required: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    present = {row.get(key) for row in rows if row.get("required") == "true"}
    for name in required:
        if name not in present:
            failures.append(
                {
                    "source": source,
                    "field": key,
                    "expected": name,
                    "actual": "missing",
                }
            )


def _rationale(scientific_gate: str) -> str:
    if scientific_gate == "ready_for_tiny_pilot":
        return (
            "The report is a design contract, not evidence. It is ready only for a "
            "tiny local pilot because the mechanism, mandatory controls, non-CE "
            "observables, leakage guardrail, and kill criteria are explicit and fail closed."
        )
    return (
        "The report blocks pilot work because at least one required mechanism, "
        "control, observable, or gate criterion is missing."
    )


def _strategy_review(path: Path) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "present": path.is_file(),
        "strategic_change_level": None,
        "notify_ben": None,
        "recommended_next_action": None,
        "verdict": None,
    }
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if key in fields:
                fields[key] = value.strip()
    notify = str(fields.get("notify_ben")).lower() == "true"
    major = fields.get("strategic_change_level") == "major"
    fields["ben_notification_required"] = notify or major
    fields["incorporation"] = (
        "accepted: created a local fail-closed core/periphery PC column contract "
        "with dense/MLP/null controls and retention/churn/commutator/intervention observables"
    )
    return fields


def _direction_shift_record(strategy: dict[str, Any]) -> dict[str, Any]:
    notify_ben = bool(strategy.get("ben_notification_required"))
    if notify_ben:
        record = "Strategy review requested Ben notification before treating this as routine."
    else:
        record = "No notify-Ben or major strategy marker; direction follows Ben's 2026-06-28 core/periphery update."
    return {
        "level": strategy.get("strategic_change_level"),
        "notify_ben": notify_ben,
        "record": record,
    }


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        out_dir / "mechanism_fields.csv",
        ["field", "specification", "contract_role", "required"],
        summary["mechanism_fields"],
    )
    _write_csv(
        out_dir / "controls.csv",
        ["control", "purpose", "matched_dimensions", "required"],
        summary["controls"],
    )
    _write_csv(
        out_dir / "observables.csv",
        ["observable", "family", "definition", "required"],
        summary["observables"],
    )
    _write_csv(
        out_dir / "gate_criteria.csv",
        ["criterion", "threshold", "failure_action", "required"],
        summary["gate_criteria"],
    )
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Core/Periphery PC Column Design",
        "",
        f"- Status: `{summary['status']}`",
        f"- Scientific gate: `{summary['scientific_gate']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        "",
        summary["rationale"],
        "",
        f"Next step: {summary['next_step']}",
        "",
        "This artifact is a preregistered local contract. It is not training evidence and does not promote ACSR or any sparse-column default.",
    ]
    if summary["direction_shift"]["notify_ben"]:
        lines.extend(["", "Ben notification is required by the strategy review header."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _dirty_diff_hash() -> str:
    try:
        diff = subprocess.check_output(["git", "diff", "--no-ext-diff"], text=True)
    except Exception:
        return "unknown"
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    args = parser.parse_args(argv)
    summary = run_core_periphery_pc_column_design(
        out_dir=args.out,
        strategy_review_path=args.strategy_review,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "scientific_gate": summary["scientific_gate"],
                "selected_next_action": summary["selected_next_action"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
