"""Adjudicate the flat-vs-sparse value bottleneck after deployable imitation.

This report consumes the command-generated deployable sparse-coding imitation
probe. It does not retrain models: the bounded next step is to decide whether
the failed support-conditioned sparse value head should be closed locally before
opening another sparse value formulation or GPU validation.
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


DEFAULT_IMITATION_PROBE = Path("results/reports/dense_teacher_deployable_sparse_coding_imitation_probe/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_flat_vs_sparse_value_bottleneck_adjudicator")

DECISION = "dense_teacher_flat_vs_sparse_value_bottleneck_adjudicator_recorded"
FAIL_DECISION = "dense_teacher_flat_vs_sparse_value_bottleneck_adjudicator_failed_closed"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "bottleneck_rows.csv",
    "candidate_actions.csv",
    "gate_rows.csv",
    "notes.md",
)

KEY_ARMS = (
    "oracle_topk_orthogonal_sparse_coding",
    "combo_mlp_router_scalar_imitation",
    "support_conditioned_combo_sparse_value_head",
    "same_router_flat_value_control",
    "oracle_support_learned_combo_coeff_sparse_coding",
    "learned_combo_support_oracle_coeff_sparse_coding",
    "random_topk_sparse_coding_null",
    "no_update_control",
)


def run_dense_teacher_flat_vs_sparse_value_bottleneck_adjudicator(
    *,
    imitation_probe_path: Path = DEFAULT_IMITATION_PROBE,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a local flat-vs-sparse bottleneck decision report."""

    start = time.time()
    imitation = _read_json(imitation_probe_path)
    review = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("dense_teacher_deployable_sparse_coding_imitation_probe", imitation_probe_path, imitation),
        {
            "source": "gpt_5_5_pro_strategy_review",
            "path": str(strategy_review_path),
            "present": strategy_review_path.is_file(),
            "status": "present" if strategy_review_path.is_file() else "missing_optional",
            "decision": review["verdict"],
            "claim_status": f"strategic_change_level={review['strategic_change_level']}; notify_ben={review['notify_ben']}",
            "selected_next_step": review["recommended_next_action"],
        },
    ]
    rows = list(imitation.get("imitation_rows", [])) if isinstance(imitation.get("imitation_rows"), list) else []
    by_arm = {str(row.get("arm")): row for row in rows if isinstance(row, dict)}
    bottleneck_rows = _bottleneck_rows(by_arm)
    gate_rows = _gate_rows(source_rows, by_arm, bottleneck_rows)
    runtime_failures = [row for row in gate_rows if row["required"] and not row["passed"]]
    scientific_failures = [row for row in gate_rows if row["gate_type"] == "scientific" and not row["passed"]]
    status = "fail" if runtime_failures else "pass"
    candidate_actions = _candidate_actions(status, scientific_failures, bottleneck_rows)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]
    selected_next_action = selected[0]["candidate_action"] if selected else "repair_adjudicator_sources"
    selected_next_step = selected[0]["next_step"] if selected else "repair missing flat-vs-sparse adjudicator source artifacts"
    claim_status = _claim_status(status, scientific_failures, selected_next_action)
    summary = {
        "status": status,
        "decision": DECISION if status == "pass" else FAIL_DECISION,
        "claim_status": claim_status,
        "selected_next_action": selected_next_action,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "backend_policy": "local CPU artifact adjudicator only; RunPod and Colab remain blocked",
        "training_executed": False,
        "source_rows": source_rows,
        "bottleneck_rows": bottleneck_rows,
        "candidate_actions": candidate_actions,
        "gate_rows": gate_rows,
        "failures": runtime_failures + scientific_failures,
        "strategy_review": review,
        "strategy_review_handling": _strategy_review_handling(review),
        "deferred_or_rejected_recommendations": [],
        "ben_notification_recommended": review["notify_ben"] == "true",
        "strategic_change_level": review["strategic_change_level"],
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _bottleneck_rows(by_arm: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    oracle = by_arm.get("oracle_topk_orthogonal_sparse_coding", {})
    combo = by_arm.get("combo_mlp_router_scalar_imitation", {})
    conditioned = by_arm.get("support_conditioned_combo_sparse_value_head", {})
    flat = by_arm.get("same_router_flat_value_control", {})
    oracle_support_coeff = by_arm.get("oracle_support_learned_combo_coeff_sparse_coding", {})
    learned_support_oracle = by_arm.get("learned_combo_support_oracle_coeff_sparse_coding", {})
    random_null = by_arm.get("random_topk_sparse_coding_null", {})
    return [
        _comparison_row(
            "naive_support_conditioned_value_head_vs_combo",
            conditioned,
            combo,
            "Tests whether conditioning coefficient outputs on the selected support combo repaired sparse value calibration.",
        ),
        _comparison_row(
            "best_deployable_sparse_vs_same_router_flat",
            _best_sparse(combo, conditioned),
            flat,
            "Tests whether the best deployable sparse arm beats a generic flat value control under the same local setup.",
        ),
        _comparison_row(
            "oracle_support_learned_coeff_vs_oracle_sparse",
            oracle_support_coeff,
            oracle,
            "Tests coefficient/value capacity with perfect support but learned deployable coefficients.",
        ),
        _comparison_row(
            "learned_support_oracle_coeff_vs_oracle_sparse",
            learned_support_oracle,
            oracle,
            "Tests support routing quality with perfect scalar coefficients.",
        ),
        _comparison_row(
            "best_deployable_sparse_vs_random_null",
            _best_sparse(combo, conditioned),
            random_null,
            "Checks that the sparse deployable signal is real relative to a random top-k null.",
        ),
    ]


def _comparison_row(name: str, primary: dict[str, Any], comparator: dict[str, Any], note: str) -> dict[str, Any]:
    primary_r2 = _float(primary.get("teacher_residual_reconstruction_r2"), float("nan"))
    comparator_r2 = _float(comparator.get("teacher_residual_reconstruction_r2"), float("nan"))
    primary_ce = _float(primary.get("ce"), float("nan"))
    comparator_ce = _float(comparator.get("ce"), float("nan"))
    primary_retention = _float(primary.get("oracle_gain_retained_fraction"), float("nan"))
    comparator_retention = _float(comparator.get("oracle_gain_retained_fraction"), float("nan"))
    primary_coeff_mse = _float(primary.get("coefficient_mse_vs_oracle"), float("nan"))
    comparator_coeff_mse = _float(comparator.get("coefficient_mse_vs_oracle"), float("nan"))
    return {
        "comparison": name,
        "primary_arm": primary.get("arm", ""),
        "comparator_arm": comparator.get("arm", ""),
        "primary_r2": _round(primary_r2),
        "comparator_r2": _round(comparator_r2),
        "r2_delta_primary_minus_comparator": _round(primary_r2 - comparator_r2),
        "primary_ce": _round(primary_ce),
        "comparator_ce": _round(comparator_ce),
        "ce_delta_primary_minus_comparator": _round(primary_ce - comparator_ce),
        "primary_oracle_gain_retention": _round(primary_retention),
        "comparator_oracle_gain_retention": _round(comparator_retention),
        "retention_delta_primary_minus_comparator": _round(primary_retention - comparator_retention),
        "primary_coefficient_mse": _round(primary_coeff_mse),
        "comparator_coefficient_mse": _round(comparator_coeff_mse),
        "primary_selected_component_overlap": primary.get("oracle_selected_component_overlap", ""),
        "comparator_selected_component_overlap": comparator.get("oracle_selected_component_overlap", ""),
        "note": note,
    }


def _best_sparse(combo: dict[str, Any], conditioned: dict[str, Any]) -> dict[str, Any]:
    combo_retention = _float(combo.get("oracle_gain_retained_fraction"), -float("inf"))
    conditioned_retention = _float(conditioned.get("oracle_gain_retained_fraction"), -float("inf"))
    return conditioned if conditioned_retention > combo_retention else combo


def _gate_rows(
    source_rows: list[dict[str, Any]],
    by_arm: dict[str, dict[str, Any]],
    bottleneck_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    comparisons = {row["comparison"]: row for row in bottleneck_rows}
    probe_present = bool(source_rows[0].get("present"))
    probe_blocks_gpu = source_rows[0].get("advance_to_gpu_validation") is False and source_rows[0].get("promotion_allowed") is False
    required_arms_present = set(KEY_ARMS).issubset(by_arm)
    conditioned_vs_combo = comparisons.get("naive_support_conditioned_value_head_vs_combo", {})
    sparse_vs_flat = comparisons.get("best_deployable_sparse_vs_same_router_flat", {})
    oracle_support_vs_oracle = comparisons.get("oracle_support_learned_coeff_vs_oracle_sparse", {})
    learned_support_vs_oracle = comparisons.get("learned_support_oracle_coeff_vs_oracle_sparse", {})
    sparse_vs_random = comparisons.get("best_deployable_sparse_vs_random_null", {})
    return [
        _gate("imitation_probe_source_present", probe_present, True, "runtime", str(source_rows[0])),
        _gate("imitation_probe_blocks_gpu", probe_blocks_gpu, True, "runtime", str(source_rows[0])),
        _gate("required_key_arms_present", required_arms_present, True, "runtime", ",".join(sorted(by_arm))),
        _gate("gpu_blocked", True, True, "runtime", "requires_gpu_now=false; advance_to_gpu_validation=false"),
        _gate(
            "support_conditioned_improves_combo_r2",
            _float(conditioned_vs_combo.get("r2_delta_primary_minus_comparator"), -float("inf")) > 0.02,
            False,
            "scientific",
            str(conditioned_vs_combo),
        ),
        _gate(
            "support_conditioned_improves_combo_retention",
            _float(conditioned_vs_combo.get("retention_delta_primary_minus_comparator"), -float("inf")) > 0.02,
            False,
            "scientific",
            str(conditioned_vs_combo),
        ),
        _gate(
            "best_sparse_beats_flat_r2",
            _float(sparse_vs_flat.get("r2_delta_primary_minus_comparator"), -float("inf")) > 0.0,
            False,
            "scientific",
            str(sparse_vs_flat),
        ),
        _gate(
            "best_sparse_matches_flat_retention",
            _float(sparse_vs_flat.get("retention_delta_primary_minus_comparator"), -float("inf")) >= -0.05,
            False,
            "scientific",
            str(sparse_vs_flat),
        ),
        _gate(
            "oracle_support_learned_coeff_near_oracle_sparse",
            _float(oracle_support_vs_oracle.get("retention_delta_primary_minus_comparator"), -float("inf")) >= -0.2,
            False,
            "scientific",
            str(oracle_support_vs_oracle),
        ),
        _gate(
            "learned_support_oracle_coeff_near_oracle_sparse",
            _float(learned_support_vs_oracle.get("retention_delta_primary_minus_comparator"), -float("inf")) >= -0.2,
            False,
            "scientific",
            str(learned_support_vs_oracle),
        ),
        _gate(
            "best_sparse_beats_random_null_r2",
            _float(sparse_vs_random.get("r2_delta_primary_minus_comparator"), -float("inf")) > 0.05,
            False,
            "scientific",
            str(sparse_vs_random),
        ),
    ]


def _candidate_actions(
    status: str,
    failures: list[dict[str, Any]],
    bottleneck_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if status != "pass":
        return [
            _candidate(
                "repair_adjudicator_sources",
                "selected",
                "required local probe artifacts are missing or not fail-closed",
                "repair or regenerate the deployable sparse-coding imitation probe before interpretation",
                "source_repair_required",
            )
        ]
    failed = {row["criterion"] for row in failures}
    if "support_conditioned_improves_combo_r2" in failed and "best_sparse_beats_flat_r2" in failed:
        return [
            _candidate(
                "close_naive_support_conditioned_sparse_value_head",
                "selected",
                "the support-conditioned sparse coefficient head loses to the prior combo arm and the best sparse arm still loses R2 to the flat value control",
                "run a local sparse-basis demotion or alternative-basis selector before any GPU validation",
                "naive_support_conditioned_sparse_value_head_closed_flat_value_bottleneck",
            ),
            _candidate(
                "train_another_combo_conditioned_sparse_value_head",
                "rejected",
                "the last support-conditioned coefficient head worsened retention and coefficient error without improving support overlap",
                "do not add another same-basis coefficient-head variant without a new basis or value formulation",
                "duplicative_sparse_value_head_rejected",
            ),
        ]
    if "best_sparse_beats_flat_r2" in failed:
        return [
            _candidate(
                "demote_current_sparse_basis_to_diagnostic",
                "selected",
                "flat value capacity still dominates R2 under the existing sparse basis",
                "select a new residual basis/value formulation locally before GPU",
                "flat_value_capacity_dominates_current_sparse_basis",
            )
        ]
    return [
        _candidate(
            "prepare_multiseed_local_confirmation",
            "selected",
            "local flat-vs-sparse adjudicator did not find a blocking sparse-vs-flat failure",
            "run a local multi-seed confirmation before any backend validation",
            "flat_vs_sparse_local_gates_clear_single_seed",
        )
    ]


def _candidate(action: str, disposition: str, reason: str, next_step: str, claim_status: str) -> dict[str, Any]:
    return {
        "candidate_action": action,
        "disposition": disposition,
        "reason": reason,
        "next_step": next_step,
        "claim_status": claim_status,
    }


def _claim_status(status: str, failures: list[dict[str, Any]], selected_next_action: str) -> str:
    if status != "pass":
        return "flat_vs_sparse_bottleneck_adjudicator_runtime_failed_closed"
    failed = {row["criterion"] for row in failures}
    if selected_next_action == "close_naive_support_conditioned_sparse_value_head":
        return "naive_support_conditioned_sparse_value_closed_flat_control_blocks_gpu"
    if "best_sparse_beats_flat_r2" in failed:
        return "flat_value_capacity_dominates_sparse_basis_blocks_gpu"
    if failures:
        return "flat_vs_sparse_value_bottleneck_local_gates_block_gpu"
    return "flat_vs_sparse_value_bottleneck_single_seed_clears_no_gpu_yet"


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status", "missing"),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
        "selected_next_step": payload.get("selected_next_step", ""),
        "advance_to_gpu_validation": payload.get("advance_to_gpu_validation", ""),
        "promotion_allowed": payload.get("promotion_allowed", ""),
    }


def _strategy_review(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return {
        "present": "true" if path.is_file() else "false",
        "strategic_change_level": _review_field(text, "strategic_change_level"),
        "notify_ben": _review_field(text, "notify_ben"),
        "recommended_next_action": _review_field(text, "recommended_next_action"),
        "verdict": _review_field(text, "verdict"),
    }


def _strategy_review_handling(review: dict[str, str]) -> str:
    if review["present"] != "true":
        return "No strategy review was present; proceeded with the status-file local adjudicator next step."
    return (
        "Accepted the GPT-5.5-Pro recommendation to keep this local and mechanistic before GPU. "
        f"Review header: strategic_change_level={review['strategic_change_level']}, notify_ben={review['notify_ben']}; "
        "Ben should be notified when notify_ben is true."
    )


def _review_field(text: str, key: str) -> str:
    prefix = f"{key}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _gate(criterion: str, passed: bool, required: bool, gate_type: str, evidence: str) -> dict[str, Any]:
    return {"criterion": criterion, "passed": bool(passed), "required": required, "gate_type": gate_type, "evidence": evidence}


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: float) -> float | str:
    if value != value or value in {float("inf"), -float("inf")}:
        return ""
    return round(value, 6)


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "bottleneck_rows.csv", summary["bottleneck_rows"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    _write_csv(out_dir / "gate_rows.csv", summary["gate_rows"])
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
        writer.writerows(rows)


def _notes(summary: dict[str, Any]) -> str:
    failed = [row["criterion"] for row in summary["failures"]]
    return "\n".join(
        [
            "# Dense Teacher Flat-vs-Sparse Value Bottleneck Adjudicator",
            "",
            f"- Status: `{summary['status']}`",
            f"- Claim status: `{summary['claim_status']}`",
            f"- Selected next action: `{summary['selected_next_action']}`",
            f"- Selected next step: {summary['selected_next_step']}",
            "- GPU validation remains blocked: `requires_gpu_now=false`, `advance_to_gpu_validation=false`, `promotion_allowed=false`.",
            f"- Failed gates: {', '.join(failed) if failed else 'none'}",
            f"- Strategy review handling: {summary['strategy_review_handling']}",
            "",
        ]
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover - git may be unavailable in tests
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imitation-probe", type=Path, default=DEFAULT_IMITATION_PROBE)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_dense_teacher_flat_vs_sparse_value_bottleneck_adjudicator(
        imitation_probe_path=args.imitation_probe,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
