"""Evaluate the local context-contrastive core/periphery probe evidence."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_DESIGN_DIR = Path("results/reports/context_contrastive_core_periphery_probe_design")
DEFAULT_CORE_PILOT_DIR = Path("results/reports/core_periphery_pc_column_nonsynthetic_pilot")
DEFAULT_LOW_CHURN_DIR = Path("results/reports/low_churn_mlp_residual_control_pilot")
DEFAULT_OUT_DIR = Path("results/reports/context_contrastive_core_periphery_probe")

CONTEXT_CANDIDATE = "causal_gated_context_contrastive_periphery"
DEMOTED_CORE = "retention_constrained_gated_periphery"
TOKEN_POSITION_NULL = "token_position_only_router"
SHUFFLED_NULL = "permuted_periphery_target_null"
FREQUENCY_NULL = "frequency_support_router"
DENSE_CONTROL = "dense_rank_norm_residual"
MLP_CONTROL = "parameter_matched_causal_mlp"
NO_CORE = "no_core_ablation"
NO_PERIPHERY = "no_periphery_ablation"
EQUAL_PLASTICITY = "equal_plasticity_core_periphery"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "probe_rows.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_context_contrastive_core_periphery_probe(
    *,
    design_dir: Path = DEFAULT_DESIGN_DIR,
    core_pilot_dir: Path = DEFAULT_CORE_PILOT_DIR,
    low_churn_dir: Path = DEFAULT_LOW_CHURN_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed local probe report from command-generated artifacts."""

    start = time.time()
    design = _read_json(design_dir / "summary.json")
    core_pilot = _read_json(core_pilot_dir / "summary.json")
    low_churn = _read_json(low_churn_dir / "summary.json")
    variant_rows = _read_csv(core_pilot_dir / "variant_metrics.csv")
    fingerprint_rows = _read_csv(core_pilot_dir / "intervention_fingerprints.csv")
    low_churn_rows = _read_csv(low_churn_dir / "arm_metrics.csv")

    variants = {row.get("variant", ""): row for row in variant_rows}
    low_churn_arms = {row.get("arm", ""): row for row in low_churn_rows}
    candidate = variants.get(CONTEXT_CANDIDATE, {})
    probe_rows = _probe_rows(variants, low_churn_arms)
    source_rows = _source_rows(design_dir, design, core_pilot_dir, core_pilot, low_churn_dir, low_churn)
    gate_rows = _gate_rows(
        design=design,
        core_pilot=core_pilot,
        low_churn=low_churn,
        variants=variants,
        fingerprints=fingerprint_rows,
        low_churn_arms=low_churn_arms,
    )
    failures = [row for row in gate_rows if not row["passed"]]
    hard_failures = [row for row in failures if row["severity"] == "hard"]
    status = "fail" if hard_failures else "pass"
    advancement = status == "pass" and not failures
    summary = {
        "status": status,
        "decision": (
            "context_contrastive_core_periphery_probe_local_candidate"
            if advancement
            else "context_contrastive_core_periphery_probe_recorded_but_blocked"
        ),
        "claim_status": (
            "context_contrastive_core_periphery_local_signal_needs_repeat"
            if advancement
            else "context_contrastive_core_periphery_not_established"
        ),
        "scientific_gate": "ready_for_local_repeat_only" if advancement else "blocked",
        "selected_next_action": (
            "repeat_context_contrastive_core_periphery_probe_on_second_seed"
            if advancement
            else "close_or_redesign_context_contrastive_core_periphery_before_gpu"
        ),
        "selected_next_step": (
            "repeat the local probe on another seed before any GPU validation"
            if advancement
            else "write a closeout or redesign record before any GPU validation"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local artifact probe only; RunPod and Colab remain blocked",
        "candidate_variant": CONTEXT_CANDIDATE,
        "candidate_observables": _candidate_observables(candidate),
        "source_rows": source_rows,
        "probe_rows": probe_rows,
        "gate_criteria": gate_rows,
        "failures": failures,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _probe_rows(
    variants: dict[str, dict[str, str]],
    low_churn_arms: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    candidate = variants.get(CONTEXT_CANDIDATE, {})
    rows: list[dict[str, Any]] = []
    for name in (
        CONTEXT_CANDIDATE,
        DEMOTED_CORE,
        TOKEN_POSITION_NULL,
        SHUFFLED_NULL,
        FREQUENCY_NULL,
        DENSE_CONTROL,
        MLP_CONTROL,
        NO_CORE,
        NO_PERIPHERY,
        EQUAL_PLASTICITY,
    ):
        row = variants.get(name, {})
        rows.append(
            {
                "row_type": "core_periphery_variant",
                "name": name,
                "present": bool(row),
                "heldout_ce": _float(row.get("heldout_ce")),
                "ce_gain_vs_context_candidate": _gain(row, candidate, "heldout_ce"),
                "anchor_kl_drift": _float(row.get("anchor_kl_drift")),
                "functional_churn": _float(row.get("functional_churn")),
                "finite_update_commutator": _float(row.get("finite_update_commutator")),
                "periphery_first_minus_core_first_prune_delta_heldout": _float(
                    row.get("periphery_first_minus_core_first_prune_delta_heldout")
                ),
                "paired_heldout_periphery_utility_mean": _float(
                    row.get("paired_heldout_periphery_utility_mean")
                ),
            }
        )
    low_churn = low_churn_arms.get("low_churn_mlp_residual_control", {})
    rows.append(
        {
            "row_type": "low_churn_mlp_control",
            "name": "low_churn_mlp_residual_control",
            "present": bool(low_churn),
            "heldout_ce": _float(low_churn.get("heldout_ce_loss")),
            "ce_gain_vs_context_candidate": (
                _float(candidate.get("heldout_ce")) - _float(low_churn.get("heldout_ce_loss"))
                if candidate and low_churn
                else None
            ),
            "anchor_kl_drift": _float(low_churn.get("heldout_anchor_kl_vs_base")),
            "functional_churn": _float(low_churn.get("heldout_prediction_flip_rate")),
            "finite_update_commutator": None,
            "periphery_first_minus_core_first_prune_delta_heldout": None,
            "paired_heldout_periphery_utility_mean": None,
        }
    )
    return rows


def _gate_rows(
    *,
    design: dict[str, Any],
    core_pilot: dict[str, Any],
    low_churn: dict[str, Any],
    variants: dict[str, dict[str, str]],
    fingerprints: list[dict[str, str]],
    low_churn_arms: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    candidate = variants.get(CONTEXT_CANDIDATE, {})
    demoted = variants.get(DEMOTED_CORE, {})
    dense = variants.get(DENSE_CONTROL, {})
    mlp = variants.get(MLP_CONTROL, {})
    low_churn_arm = low_churn_arms.get("low_churn_mlp_residual_control", {})
    required = {
        CONTEXT_CANDIDATE,
        DEMOTED_CORE,
        TOKEN_POSITION_NULL,
        SHUFFLED_NULL,
        FREQUENCY_NULL,
        DENSE_CONTROL,
        MLP_CONTROL,
        NO_CORE,
        NO_PERIPHERY,
        EQUAL_PLASTICITY,
    }
    present = set(variants)
    context_ce = _float(candidate.get("heldout_ce"))
    null_ces = [_float(variants.get(name, {}).get("heldout_ce")) for name in (TOKEN_POSITION_NULL, SHUFFLED_NULL, FREQUENCY_NULL)]
    control_ces = [
        _float(demoted.get("heldout_ce")),
        _float(dense.get("heldout_ce")),
        _float(mlp.get("heldout_ce")),
        _float(low_churn_arm.get("heldout_ce_loss")),
    ]
    context_anchor = _float(candidate.get("anchor_kl_drift"))
    context_churn = _float(candidate.get("functional_churn"))
    context_commutator = _float(candidate.get("finite_update_commutator"))
    reference_anchor = min_present(
        _float(demoted.get("anchor_kl_drift")),
        _float(dense.get("anchor_kl_drift")),
        _float(mlp.get("anchor_kl_drift")),
        _float(low_churn_arm.get("heldout_anchor_kl_vs_base")),
    )
    reference_churn = min_present(
        _float(demoted.get("functional_churn")),
        _float(dense.get("functional_churn")),
        _float(mlp.get("functional_churn")),
        _float(low_churn_arm.get("heldout_prediction_flip_rate")),
    )
    reference_commutator = min_present(
        _float(demoted.get("finite_update_commutator")),
        _float(dense.get("finite_update_commutator")),
        _float(mlp.get("finite_update_commutator")),
    )
    return [
        _criterion(
            "design_selected_local_probe",
            design.get("status") == "pass"
            and design.get("selected_next_action") == "implement_context_contrastive_core_periphery_probe_locally",
            design.get("selected_next_action"),
            "context-contrastive design must select a local probe",
            "hard",
        ),
        _criterion(
            "source_pilot_available",
            core_pilot.get("status") == "pass" and bool(variants),
            {"status": core_pilot.get("status"), "variant_count": len(variants)},
            "core/periphery local pilot variant rows must be present",
            "hard",
        ),
        _criterion(
            "required_controls_present",
            required.issubset(present),
            sorted(required - present),
            "candidate plus all design-required controls must be present",
            "hard",
        ),
        _criterion(
            "fingerprints_present_for_candidate",
            any(row.get("variant") == CONTEXT_CANDIDATE for row in fingerprints),
            CONTEXT_CANDIDATE,
            "candidate must have raw core/periphery intervention fingerprints",
            "hard",
        ),
        _criterion(
            "beats_token_position_shuffled_frequency_nulls",
            context_ce is not None and all(value is not None and context_ce < value for value in null_ces),
            {"context_ce": context_ce, "null_ces": null_ces},
            "candidate heldout CE must beat token/position, shuffled, and frequency nulls",
            "claim",
        ),
        _criterion(
            "beats_demoted_dense_mlp_controls",
            context_ce is not None and all(value is not None and context_ce <= value for value in control_ces),
            {"context_ce": context_ce, "control_ces": control_ces},
            "candidate heldout CE must be nonworse than demoted core, dense, raw MLP, and low-churn MLP controls",
            "claim",
        ),
        _criterion(
            "retention_churn_budget_nonworse",
            _nonworse(context_anchor, reference_anchor) and _nonworse(context_churn, reference_churn),
            {
                "context_anchor_kl": context_anchor,
                "reference_anchor_kl": reference_anchor,
                "context_churn": context_churn,
                "reference_churn": reference_churn,
            },
            "candidate anchor drift and functional churn must be nonworse than the best available control",
            "claim",
        ),
        _criterion(
            "finite_update_commutator_budget_nonworse",
            _nonworse(context_commutator, reference_commutator),
            {"context_commutator": context_commutator, "reference_commutator": reference_commutator},
            "candidate finite-update commutator must be nonworse than the best sparse/dense control",
            "claim",
        ),
        _criterion(
            "periphery_first_pruning_signal_positive",
            _float(candidate.get("periphery_first_minus_core_first_prune_delta_heldout")) is not None
            and _float(candidate.get("periphery_first_minus_core_first_prune_delta_heldout")) > 0,
            _float(candidate.get("periphery_first_minus_core_first_prune_delta_heldout")),
            "periphery-first pruning should be less damaging than core-first pruning on heldout anchors",
            "claim",
        ),
        _criterion(
            "periphery_utility_positive",
            _float(candidate.get("paired_heldout_periphery_utility_mean")) is not None
            and _float(candidate.get("paired_heldout_periphery_utility_mean")) > 0,
            _float(candidate.get("paired_heldout_periphery_utility_mean")),
            "candidate periphery must have positive heldout causal utility",
            "claim",
        ),
        _criterion(
            "low_churn_mlp_remains_unpromoted",
            low_churn.get("advance_to_gpu_validation") is False
            and low_churn.get("advancement_row_count") == 0,
            {
                "advance_to_gpu_validation": low_churn.get("advance_to_gpu_validation"),
                "advancement_row_count": low_churn.get("advancement_row_count"),
            },
            "MLP control source must not already require GPU validation",
            "hard",
        ),
    ]


def _candidate_observables(candidate: dict[str, str]) -> dict[str, Any]:
    return {
        "heldout_ce": _float(candidate.get("heldout_ce")),
        "anchor_kl_drift": _float(candidate.get("anchor_kl_drift")),
        "functional_churn": _float(candidate.get("functional_churn")),
        "finite_update_commutator": _float(candidate.get("finite_update_commutator")),
        "periphery_first_minus_core_first_prune_delta_heldout": _float(
            candidate.get("periphery_first_minus_core_first_prune_delta_heldout")
        ),
        "paired_heldout_periphery_utility_mean": _float(
            candidate.get("paired_heldout_periphery_utility_mean")
        ),
    }


def _source_rows(
    design_dir: Path,
    design: dict[str, Any],
    core_pilot_dir: Path,
    core_pilot: dict[str, Any],
    low_churn_dir: Path,
    low_churn: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _source_row("context_contrastive_core_periphery_probe_design", design_dir / "summary.json", design),
        _source_row("core_periphery_pc_column_nonsynthetic_pilot", core_pilot_dir / "summary.json", core_pilot),
        _source_row("low_churn_mlp_residual_control_pilot", low_churn_dir / "summary.json", low_churn),
    ]


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(payload),
        "status": payload.get("status", "missing"),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
        "selected_next_action": payload.get("selected_next_action", ""),
    }


def _criterion(
    criterion: str,
    passed: bool,
    actual: Any,
    expected: str,
    severity: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "severity": severity,
        "actual": actual,
        "expected": expected,
        "failure_reason": "" if passed else expected,
    }


def _gain(row: dict[str, str], candidate: dict[str, str], field: str) -> float | None:
    value = _float(row.get(field))
    candidate_value = _float(candidate.get(field))
    if value is None or candidate_value is None:
        return None
    return value - candidate_value


def min_present(*values: float | None) -> float | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _nonworse(value: float | None, reference: float | None) -> bool:
    return value is not None and reference is not None and value <= reference


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except FileNotFoundError:
        return []


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "probe_rows.csv", summary["probe_rows"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _notes(summary: dict[str, Any]) -> str:
    failed = ", ".join(row["criterion"] for row in summary["failures"]) or "none"
    return (
        "# Context-Contrastive Core/Periphery Probe\n\n"
        f"Decision: `{summary['decision']}`.\n\n"
        f"Claim status: `{summary['claim_status']}`.\n\n"
        f"Failed gates: {failed}.\n\n"
        "GPU validation remains blocked. This report is local artifact evidence only.\n"
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN_DIR)
    parser.add_argument("--core-pilot-dir", type=Path, default=DEFAULT_CORE_PILOT_DIR)
    parser.add_argument("--low-churn-dir", type=Path, default=DEFAULT_LOW_CHURN_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_context_contrastive_core_periphery_probe(
        design_dir=args.design_dir,
        core_pilot_dir=args.core_pilot_dir,
        low_churn_dir=args.low_churn_dir,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
