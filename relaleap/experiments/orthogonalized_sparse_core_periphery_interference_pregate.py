"""Record the orthogonalized sparse core/periphery interference pregate."""

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


DEFAULT_SYNTHESIS = Path("results/reports/dense_mlp_control_synthesis/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/orthogonalized_sparse_core_periphery_interference_pregate")

IMPLEMENT_ACTION = "implement_local_orthogonalized_sparse_core_periphery_interference_pilot"
REPAIR_ACTION = "repair_orthogonalized_sparse_core_periphery_pregate_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "mechanism_arms.csv",
    "matched_controls.csv",
    "observable_gates.csv",
    "kill_thresholds.csv",
    "leakage_nulls.csv",
    "notes.md",
)


def run_orthogonalized_sparse_core_periphery_interference_pregate(
    *,
    synthesis_path: Path = DEFAULT_SYNTHESIS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed local pregate for the selected sparse mechanism."""

    start = time.time()
    synthesis = _read_json(synthesis_path)
    strategy = _strategy_review(strategy_review_path)
    mechanism_arms = _mechanism_arms()
    matched_controls = _matched_controls()
    observable_gates = _observable_gates()
    kill_thresholds = _kill_thresholds()
    leakage_nulls = _leakage_nulls()
    source_rows = _source_rows(synthesis_path, synthesis, strategy_review_path, strategy)
    gate_rows = _gate_rows(
        synthesis=synthesis,
        strategy=strategy,
        mechanism_arms=mechanism_arms,
        matched_controls=matched_controls,
        observable_gates=observable_gates,
        kill_thresholds=kill_thresholds,
        leakage_nulls=leakage_nulls,
    )
    failures = [row for row in gate_rows if not row["passed"]]
    source_failures = [
        {
            "criterion": f"{row['source']}_present",
            "passed": False,
            "actual": row["path"],
            "threshold": "required source artifact must exist and parse",
            "failure_reason": "required source artifact must exist and parse",
        }
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]
    failures.extend(source_failures)
    status = "pass" if not failures else "fail"
    summary = {
        "status": status,
        "decision": (
            "orthogonalized_sparse_core_periphery_interference_pregate_recorded"
            if status == "pass"
            else "orthogonalized_sparse_core_periphery_interference_pregate_failed_closed"
        ),
        "claim_status": "pregate_only_no_training_or_gpu_claim",
        "selected_next_action": IMPLEMENT_ACTION if status == "pass" else REPAIR_ACTION,
        "selected_next_step": (
            "implement the local CPU pilot harness from mechanism_arms.csv with matched controls and fail-closed interference gates"
            if status == "pass"
            else "repair missing synthesis/review source artifacts before implementing the pilot harness"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local pregate and CPU pilot first; RunPod and Colab remain blocked",
        "ben_should_be_notified": strategy["ben_notification_required"],
        "direction_shift_recorded": strategy["ben_notification_required"],
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "source_rows": source_rows,
        "gate_rows": gate_rows,
        "mechanism_arms": mechanism_arms,
        "matched_controls": matched_controls,
        "observable_gates": observable_gates,
        "kill_thresholds": kill_thresholds,
        "leakage_nulls": leakage_nulls,
        "failures": failures,
        "rationale": _rationale(synthesis, status),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _mechanism_arms() -> list[dict[str, Any]]:
    return [
        {
            "arm": "orthogonalized_sparse_additive_core_periphery",
            "family": "sparse_mechanism_candidate",
            "trainable": True,
            "residual_parameterization": "sum of orthogonalized selected column value vectors with learned scalar norm controller",
            "update_rule": "protected low-plasticity core updates plus plastic/pruneable periphery updates under explicit per-column masks",
            "required_outputs": "ce; residual_l2; active_params; stored_params; retention; functional_churn; finite_update_commutator; intervention_selectivity; context_reuse",
        },
        {
            "arm": "orthogonalized_sparse_no_norm_controller_ablation",
            "family": "mechanism_ablation",
            "trainable": True,
            "residual_parameterization": "same orthogonalized columns without learned norm scaling",
            "update_rule": "same masks and core/periphery split",
            "required_outputs": "same as candidate; isolates norm-controller contribution",
        },
        {
            "arm": "orthogonalized_sparse_no_core_protection_ablation",
            "family": "mechanism_ablation",
            "trainable": True,
            "residual_parameterization": "same columns and norm controller with equal core/periphery plasticity",
            "update_rule": "no protected low-plasticity core",
            "required_outputs": "same as candidate; tests whether protection is real rather than accounting",
        },
        {
            "arm": "orthogonalized_sparse_no_update_masks_ablation",
            "family": "mechanism_ablation",
            "trainable": True,
            "residual_parameterization": "same values with unrestricted column updates",
            "update_rule": "mask disabled",
            "required_outputs": "same as candidate; tests interference benefit of masks",
        },
    ]


def _matched_controls() -> list[dict[str, str]]:
    return [
        _control("dense_ridge_residual", "generic dense residual control", "active_params, stored_params, residual_l2, ce_band, split, seed"),
        _control("random_feature_mlp_residual", "high-capacity MLP residual control", "active_params, stored_params, residual_l2, ce_band, split, seed"),
        _control("low_rank_residual", "rank-constrained dense control", "active_rank, residual_l2, ce_band, split, seed"),
        _control("same_router_flat_value_mlp", "same support/router with noncolumnar value function", "router_inputs, active_params, residual_l2, ce_band"),
        _control("random_sparse_columns", "sparse-column null with matched sparsity", "column_count, top_k, active_params, residual_l2"),
        _control("frequency_matched_sparse_router", "support-frequency null", "support_frequency, top_k, residual_l2, ce_band"),
    ]


def _control(control: str, purpose: str, matched_dimensions: str) -> dict[str, str]:
    return {
        "control": control,
        "purpose": purpose,
        "matched_dimensions": matched_dimensions,
        "required": "true",
    }


def _observable_gates() -> list[dict[str, str]]:
    return [
        _observable("ce_guardrail", "quality", "candidate must stay within 0.05 CE or 5 percent of best matched dense/MLP control, whichever is stricter"),
        _observable("residual_l2_budget", "budget", "candidate and controls must report matched mean and p95 residual norms"),
        _observable("active_and_stored_params", "budget", "candidate and controls must report active and stored parameter budgets"),
        _observable("functional_churn_flip_rate", "interference", "candidate must beat or match best dense/MLP control"),
        _observable("retention_after_sequential_updates", "interference", "candidate must beat or match best dense/MLP control on anchor retention"),
        _observable("finite_update_commutator_symmetric_kl", "interference", "candidate must beat or match best dense/MLP and flat-value controls"),
        _observable("intervention_selectivity", "causal", "necessity/sufficiency gains must be on-target with bounded off-target KL"),
        _observable("context_reuse_score", "causal", "columns must retain causal utility across held-out contexts instead of memorizing support labels"),
        _observable("periphery_first_pruning_delta", "causal_pruning", "periphery pruning should hurt less than core pruning unless periphery is causally necessary"),
    ]


def _observable(metric: str, family: str, promotion_gate: str) -> dict[str, str]:
    return {"metric": metric, "family": family, "promotion_gate": promotion_gate, "required": "true"}


def _kill_thresholds() -> list[dict[str, Any]]:
    return [
        _threshold("dense_mlp_quality_dominance", "kill", "candidate CE outside matched dense/MLP band", "ce_guardrail fails"),
        _threshold("interference_no_better_than_dense", "kill", "candidate fails to beat or match dense/MLP on churn, retention, and commutator jointly", "any two interference gates fail"),
        _threshold("norm_or_budget_mismatch", "repair", "candidate wins only by lower residual norm or lower active/stored budget", "budget matching incomplete"),
        _threshold("leakage_or_future_feature_use", "kill", "routing or gates use future hidden/delta, teacher residuals, teacher logits, labels, or oracle supports at evaluation", "any leakage row fails"),
        _threshold("null_control_win", "kill", "token/position, shuffled, delayed, random, or frequency null matches candidate within tolerance", "strong null not beaten"),
        _threshold("core_periphery_accounting_fiction", "kill", "no-core/no-mask/equal-plasticity ablations match the full candidate on interference", "mechanism ablations not worse"),
    ]


def _threshold(name: str, disposition: str, condition: str, trigger: str) -> dict[str, Any]:
    return {"threshold": name, "disposition": disposition, "condition": condition, "trigger": trigger, "required": True}


def _leakage_nulls() -> list[dict[str, str]]:
    return [
        _null("token_position_only_router", "shortcut null using token and position only"),
        _null("shuffled_teacher_residual_targets", "misaligned residual target null"),
        _null("delayed_teacher_residual_targets", "delayed target null for temporal leakage"),
        _null("random_fixed_supports", "random support selection under same top-k"),
        _null("majority_or_frequency_supports", "support-frequency shortcut null"),
        _null("feature_schema_hash", "record deployable feature set and fail on nondeployable feature chunks"),
    ]


def _null(control: str, purpose: str) -> dict[str, str]:
    return {"control": control, "purpose": purpose, "required": "true"}


def _gate_rows(
    *,
    synthesis: dict[str, Any],
    strategy: dict[str, Any],
    mechanism_arms: list[dict[str, Any]],
    matched_controls: list[dict[str, Any]],
    observable_gates: list[dict[str, Any]],
    kill_thresholds: list[dict[str, Any]],
    leakage_nulls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    arm_names = {row["arm"] for row in mechanism_arms}
    control_names = {row["control"] for row in matched_controls}
    metric_names = {row["metric"] for row in observable_gates}
    null_names = {row["control"] for row in leakage_nulls}
    return [
        _criterion(
            "dense_mlp_synthesis_selected_this_pregate",
            synthesis.get("status") == "pass"
            and synthesis.get("selected_next_action")
            == "design_orthogonalized_sparse_additive_core_periphery_interference_pregate"
            and synthesis.get("requires_gpu_now") is False
            and synthesis.get("advance_to_gpu_validation") is False,
            synthesis.get("selected_next_action"),
            "dense/MLP synthesis must select this local pregate with GPU blocked",
        ),
        _criterion(
            "major_pivot_notification_preserved",
            strategy["present"] and strategy["ben_notification_required"],
            {
                "strategic_change_level": strategy["strategic_change_level"],
                "notify_ben": strategy["notify_ben"],
                "verdict": strategy["verdict"],
            },
            "latest major/notify-Ben strategy review must be recorded",
        ),
        _criterion(
            "mechanism_ablation_coverage_complete",
            {
                "orthogonalized_sparse_additive_core_periphery",
                "orthogonalized_sparse_no_norm_controller_ablation",
                "orthogonalized_sparse_no_core_protection_ablation",
                "orthogonalized_sparse_no_update_masks_ablation",
            }.issubset(arm_names),
            sorted(arm_names),
            "candidate plus norm/core/mask ablations are required",
        ),
        _criterion(
            "matched_control_coverage_complete",
            {
                "dense_ridge_residual",
                "random_feature_mlp_residual",
                "low_rank_residual",
                "same_router_flat_value_mlp",
                "random_sparse_columns",
                "frequency_matched_sparse_router",
            }.issubset(control_names),
            sorted(control_names),
            "dense, MLP, low-rank, flat, random sparse, and frequency controls are required",
        ),
        _criterion(
            "interference_observable_coverage_complete",
            {
                "ce_guardrail",
                "residual_l2_budget",
                "functional_churn_flip_rate",
                "retention_after_sequential_updates",
                "finite_update_commutator_symmetric_kl",
                "intervention_selectivity",
                "context_reuse_score",
            }.issubset(metric_names),
            sorted(metric_names),
            "quality, budget, churn, retention, commutator, selectivity, and reuse gates are required",
        ),
        _criterion(
            "kill_thresholds_fail_closed",
            len(kill_thresholds) >= 6 and all(row["disposition"] in {"kill", "repair"} for row in kill_thresholds),
            len(kill_thresholds),
            "pregate must specify fail-closed kill/repair thresholds before training",
        ),
        _criterion(
            "leakage_and_null_controls_complete",
            {
                "token_position_only_router",
                "shuffled_teacher_residual_targets",
                "delayed_teacher_residual_targets",
                "random_fixed_supports",
                "majority_or_frequency_supports",
                "feature_schema_hash",
            }.issubset(null_names),
            sorted(null_names),
            "shortcut, misalignment, random/frequency, and feature-schema leakage controls are required",
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


def _source_rows(
    synthesis_path: Path,
    synthesis: dict[str, Any],
    strategy_review_path: Path,
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "source": "dense_mlp_control_synthesis",
            "path": str(synthesis_path),
            "present": synthesis_path.is_file() and bool(synthesis),
            "sha256": _file_sha256(synthesis_path),
            "mtime": _mtime(synthesis_path),
            "status": synthesis.get("status", "missing" if not synthesis_path.is_file() else ""),
            "decision": synthesis.get("decision", ""),
            "claim_status": synthesis.get("claim_status", ""),
            "selected_next_action": synthesis.get("selected_next_action", ""),
        },
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "sha256": _file_sha256(strategy_review_path),
            "mtime": _mtime(strategy_review_path),
            "status": "read" if strategy["present"] else "missing_required",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}; verdict={strategy['verdict']}"
            ),
        },
    ]


def _rationale(synthesis: dict[str, Any], status: str) -> str:
    if status != "pass":
        return "The pregate could not verify its source synthesis or required contract rows, so it fails closed."
    return (
        "The dense/MLP synthesis closed pair-composer GPU validation and selected a local mechanism-factorized "
        "sparse-interference branch. This pregate turns that branch into a concrete fail-closed contract: "
        "orthogonalized additive sparse columns with norm control, update masks, and protected-core/plastic-periphery "
        "structure must beat matched dense/MLP controls on interference and reuse at comparable quality before any "
        f"backend validation. Source decision: {synthesis.get('decision', '')}."
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    strategy = {
        "path": str(path),
        "present": bool(text),
        "strategic_change_level": _header_value(text, "strategic_change_level") or "unknown",
        "notify_ben": _header_value(text, "notify_ben") or "unknown",
        "recommended_next_action": _header_value(text, "recommended_next_action") or "",
        "verdict": _header_value(text, "verdict") or "",
    }
    strategy["ben_notification_required"] = (
        str(strategy["notify_ben"]).lower() == "true"
        or str(strategy["strategic_change_level"]).lower() == "major"
    )
    return strategy


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "Strategy review was missing; pregate fails closed until the latest review is available."
    if strategy["ben_notification_required"]:
        return "Accepted the GPT-5.5-Pro major pivot; Ben should be notified, and GPU remains blocked."
    return "Accepted the GPT-5.5-Pro local pregate recommendation; GPU remains blocked."


def _header_value(text: str, key: str) -> str:
    prefix = f"{key}:"
    for line in text.splitlines()[:20]:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mtime(path: Path) -> str:
    if not path.is_file():
        return ""
    return str(path.stat().st_mtime)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "mechanism_arms.csv", summary["mechanism_arms"])
    _write_csv(out_dir / "matched_controls.csv", summary["matched_controls"])
    _write_csv(out_dir / "observable_gates.csv", summary["observable_gates"])
    _write_csv(out_dir / "kill_thresholds.csv", summary["kill_thresholds"])
    _write_csv(out_dir / "leakage_nulls.csv", summary["leakage_nulls"])
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
            "# Orthogonalized Sparse Core/Periphery Interference Pregate",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Selected action: `{summary['selected_next_action']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            f"- Ben should be notified: `{summary['ben_should_be_notified']}`",
            f"- Direction shift recorded: `{summary['direction_shift_recorded']}`",
            f"- Next step: {summary['selected_next_step']}",
            "",
            "GPU validation remains blocked. This pregate permits only a local CPU pilot harness with matched dense/MLP/null controls.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesis", type=Path, default=DEFAULT_SYNTHESIS)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_orthogonalized_sparse_core_periphery_interference_pregate(
        synthesis_path=args.synthesis,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "selected_next_action": summary["selected_next_action"],
                "ben_should_be_notified": summary["ben_should_be_notified"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
